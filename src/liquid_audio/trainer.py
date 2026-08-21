from __future__ import annotations

import json
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from accelerate import Accelerator
from accelerate.utils import (
    DataLoaderConfiguration,
    DistributedDataParallelKwargs,
    GradientAccumulationPlugin,
    ProjectConfiguration,
    set_seed,
)
from torch.utils.data import DataLoader, Sampler

from liquid_audio import LFM2AudioModel
from liquid_audio.data.dataloader import lfm2_collator
from liquid_audio.model.lfm2_audio import LFM2AudioModelOutput
from liquid_audio.model.transformer import wrap_activation_checkpoint
from liquid_audio.training.augmentation import SpecAugment
from liquid_audio.training.config import FineTuneConfig, TrainerConfig
from liquid_audio.training.peft import (
    FineTuneReport,
    configure_finetuning,
    merge_lora_layers,
    optimizer_parameter_groups,
    promote_trainable_parameters_to_fp32,
)
from liquid_audio.training.sampler import EpochRandomSampler

if TYPE_CHECKING:
    from liquid_audio.data.dataloader import LFM2DataLoader, TaskBalancedDataset
    from liquid_audio.data.types import LFM2AudioModelInput


@dataclass(slots=True)
class TrainingState:
    step: int = 0
    epoch: int = 0
    best_validation_loss: float = math.inf
    validations_without_improvement: int = 0
    batches_in_epoch: int = 0

    def state_dict(self) -> dict[str, int | float]:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "best_validation_loss": self.best_validation_loss,
            "validations_without_improvement": self.validations_without_improvement,
            "batches_in_epoch": self.batches_in_epoch,
        }

    def load_state_dict(self, state_dict: dict[str, int | float]) -> None:
        self.step = int(state_dict["step"])
        self.epoch = int(state_dict["epoch"])
        self.best_validation_loss = float(state_dict["best_validation_loss"])
        self.validations_without_improvement = int(state_dict["validations_without_improvement"])
        self.batches_in_epoch = int(state_dict.get("batches_in_epoch", 0))


class Trainer:
    """Production-oriented trainer for ASR, TTS, and joint omni fine-tuning.

    Passing a :class:`TrainerConfig` enables all modern features. The original
    keyword arguments remain supported so existing training scripts continue to
    work, in which case full-parameter fine-tuning is preserved by default.
    """

    def __init__(
        self,
        model_id: str = "LiquidAI/LFM2.5-Audio-1.5B",
        train_data: LFM2DataLoader | TaskBalancedDataset | None = None,
        val_data: LFM2DataLoader | TaskBalancedDataset | None = None,
        lr: float = 3e-5,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.1,
        min_ratio: float = 0.1,
        max_steps: int = 1000,
        warmup_steps: int = 100,
        batch_size: int = 16,
        dataloader_num_workers: int = 0,
        logging_interval: int = 10,
        save_interval: int = 500,
        val_interval: int = 100,
        output_dir: str = "tmp",
        *,
        config: TrainerConfig | None = None,
        model: LFM2AudioModel | None = None,
        fine_tune_config: FineTuneConfig | None = None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        mixed_precision: str = "auto",
        seed: int = 42,
        resume_from_checkpoint: str | None = None,
    ) -> None:
        if train_data is None:
            raise ValueError("train_data is required")
        if len(train_data) == 0:
            raise ValueError("train_data must contain at least one sample")

        self.config = config or TrainerConfig(
            model_id=model_id,
            output_dir=output_dir,
            max_steps=max_steps,
            warmup_steps=warmup_steps,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=lr,
            betas=betas,
            weight_decay=weight_decay,
            min_lr_ratio=min_ratio,
            max_grad_norm=max_grad_norm,
            mixed_precision=mixed_precision,  # type: ignore[arg-type]
            trainable_parameters_fp32=False,
            seed=seed,
            dataloader_num_workers=dataloader_num_workers,
            logging_interval=logging_interval,
            save_interval=save_interval,
            validation_interval=val_interval,
            resume_from_checkpoint=resume_from_checkpoint,
            gradient_checkpointing=False,
            text_label_smoothing=0.0,
            end_token_weight=1.0,
            fine_tune=fine_tune_config or FineTuneConfig(strategy="full"),
            specaugment=None,
        )
        self.config.validate()
        self.output_dir = Path(self.config.output_dir)
        if self.output_dir.exists() and any(self.output_dir.iterdir()) and self.config.resume_from_checkpoint is None:
            raise FileExistsError(
                f"output directory is not empty: {self.output_dir}; choose a new directory or resume a checkpoint"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        precision = self._resolve_mixed_precision(self.config.mixed_precision)
        self.accelerator = Accelerator(
            mixed_precision=precision,
            gradient_accumulation_plugin=GradientAccumulationPlugin(
                num_steps=self.config.gradient_accumulation_steps,
                sync_with_dataloader=False,
            ),
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
            dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
            project_config=ProjectConfiguration(
                project_dir=str(self.output_dir),
                automatic_checkpoint_naming=True,
                total_limit=self.config.checkpoint_limit,
            ),
        )
        set_seed(self.config.seed, device_specific=True)
        self._configure_numeric_precision()

        dtype = self._model_dtype(precision)
        self.model = model or LFM2AudioModel.from_pretrained(
            self.config.model_id,
            revision=self.config.revision,
            device=self.accelerator.device,
            dtype=dtype,
        )
        self.fine_tune_report = configure_finetuning(self.model, self.config.fine_tune)
        if self.config.trainable_parameters_fp32:
            promote_trainable_parameters_to_fp32(self.model)
        self._enable_gradient_checkpointing()

        parameter_groups = optimizer_parameter_groups(
            self.model,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            audio_encoder_lr_scale=self.config.audio_encoder_lr_scale,
        )
        self.optimizer = torch.optim.AdamW(
            parameter_groups,
            betas=self.config.betas,
            eps=self.config.adam_epsilon,
            fused=self.accelerator.device.type == "cuda",
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._lr_multiplier)

        if self.config.specaugment is not None and train_data.specaugment is None:
            train_data.specaugment = SpecAugment(self.config.specaugment)
        set_augmentation_seed = getattr(train_data, "set_augmentation_seed", None)
        if set_augmentation_seed is not None:
            set_augmentation_seed(self.config.seed)
        if val_data is not None:
            val_data.specaugment = None

        self.train_data = train_data
        self.train_sampler = EpochRandomSampler(train_data, seed=self.config.seed)
        self.train_loader = self._make_dataloader(train_data, sampler=self.train_sampler)
        self.val_loader = self._make_dataloader(val_data) if val_data is not None else None

        if self.val_loader is None:
            self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
                self.model,
                self.optimizer,
                self.train_loader,
                self.scheduler,
            )
        else:
            self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler = self.accelerator.prepare(
                self.model,
                self.optimizer,
                self.train_loader,
                self.val_loader,
                self.scheduler,
            )

        self.state = TrainingState()
        self.accelerator.register_for_checkpointing(self.state)
        self.optimizer.zero_grad(set_to_none=True)
        self.started_at = 0.0
        self.last_log_at = 0.0
        self.last_log_step = 0
        self.best_checkpoint_dir = self.output_dir / "best-checkpoint"
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self._resume_if_requested()

    @property
    def step(self) -> int:
        return self.state.step

    @property
    def epoch(self) -> int:
        return self.state.epoch

    def train(self) -> None:
        self.started_at = time.monotonic()
        self.last_log_at = self.started_at
        self._print_startup_summary()
        self.model.train()
        train_iter = self._new_train_iterator()
        stopped_early = False

        while self.state.step < self.config.max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                self.state.epoch += 1
                self.state.batches_in_epoch = 0
                self.train_sampler.set_epoch(self.state.epoch)
                set_epoch = getattr(self.train_data, "set_epoch", None)
                if set_epoch is not None:
                    set_epoch(self.state.epoch)
                train_iter = self._new_train_iterator()
                batch = next(train_iter)

            output, optimizer_updated, grad_norm = self.train_step(batch)
            self.state.batches_in_epoch += 1
            if not optimizer_updated:
                continue

            self.state.step += 1
            if self.state.step % self.config.logging_interval == 0:
                self.log(output, grad_norm)

            if self.state.step % self.config.save_interval == 0:
                self.accelerator.save_state(safe_serialization=True)

            if self.val_loader is not None and self.state.step % self.config.validation_interval == 0:
                metrics = self.validate()
                stopped_early = self._record_validation(metrics)
                self.model.train()
                if stopped_early:
                    break

        completed_step = self.state.step
        completed_epoch = self.state.epoch
        self.accelerator.wait_for_everyone()
        if self.best_checkpoint_dir.exists() and self.val_loader is not None:
            self.accelerator.load_state(str(self.best_checkpoint_dir))
            self.state.step = completed_step
            self.state.epoch = completed_epoch
        self._save_final_model(stopped_early=stopped_early)
        self.accelerator.end_training()

        elapsed = self._elapsed()
        self.accelerator.print(
            f"[{elapsed}] Training complete at optimizer step {completed_step}"
            + (" (early stopping)" if stopped_early else "")
        )

    def train_step(
        self,
        batch: LFM2AudioModelInput,
    ) -> tuple[LFM2AudioModelOutput, bool, float | None]:
        batch = batch.to(self.accelerator.device)
        grad_norm: float | None = None
        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                output = self._forward(batch)
            if not torch.isfinite(output.loss.detach()):
                raise FloatingPointError(f"non-finite loss at optimizer step {self.state.step}: {output.loss.item()}")

            self.accelerator.backward(output.loss)
            optimizer_updated = self.accelerator.sync_gradients
            if optimizer_updated and self.config.max_grad_norm > 0:
                norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                grad_norm = float(norm.detach().item()) if isinstance(norm, torch.Tensor) else float(norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
        return output, optimizer_updated, grad_norm

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        totals = torch.zeros(8, device=self.accelerator.device, dtype=torch.float64)

        for batch in self.val_loader:
            batch = batch.to(self.accelerator.device)
            with self.accelerator.autocast():
                output = self._forward(batch)
            text_tokens = output.text_tokens.detach().to(torch.float64)
            audio_tokens = output.audio_out_tokens.detach().to(torch.float64)
            text_weight = output.text_loss_weight if output.text_loss_weight is not None else text_tokens
            audio_weight = output.audio_loss_weight if output.audio_loss_weight is not None else audio_tokens
            loss_weight = output.loss_weight if output.loss_weight is not None else text_tokens + audio_tokens
            text_weight = text_weight.detach().to(torch.float64)
            audio_weight = audio_weight.detach().to(torch.float64)
            loss_weight = loss_weight.detach().to(torch.float64)
            totals += torch.stack(
                (
                    output.loss.detach().to(torch.float64) * loss_weight,
                    loss_weight,
                    output.text_loss.detach().to(torch.float64) * text_weight,
                    text_weight,
                    output.audio_loss.detach().to(torch.float64) * audio_weight,
                    audio_weight,
                    text_tokens,
                    audio_tokens,
                )
            )

        totals = self.accelerator.reduce(totals, reduction="sum")
        metrics = {
            "loss": float((totals[0] / totals[1].clamp_min(1)).item()),
            "text_loss": float((totals[2] / totals[3].clamp_min(1)).item()),
            "audio_loss": float((totals[4] / totals[5].clamp_min(1)).item()),
            "text_tokens": float(totals[6].item()),
            "audio_tokens": float(totals[7].item()),
        }
        self._emit_metrics("validation", metrics)
        return metrics

    def log(self, model_output: LFM2AudioModelOutput, grad_norm: float | None = None) -> None:
        train_loss = self.accelerator.reduce(model_output.loss.detach(), reduction="mean").item()
        text_loss = self.accelerator.reduce(model_output.text_loss.detach(), reduction="mean").item()
        audio_loss = self.accelerator.reduce(model_output.audio_loss.detach(), reduction="mean").item()
        now = time.monotonic()
        steps_since_log = self.state.step - self.last_log_step
        steps_per_second = steps_since_log / max(now - self.last_log_at, 1e-6)
        metrics = {
            "loss": train_loss,
            "text_loss": text_loss,
            "audio_loss": audio_loss,
            "learning_rate": float(max(group["lr"] for group in self.optimizer.param_groups)),
            "optimizer_steps_per_second": steps_per_second,
        }
        if grad_norm is not None:
            metrics["gradient_norm"] = grad_norm
        self._emit_metrics("train", metrics)
        self.last_log_at = now
        self.last_log_step = self.state.step

    def _forward(self, batch: LFM2AudioModelInput) -> LFM2AudioModelOutput:
        return self.model(
            batch,
            text_label_smoothing=self.config.text_label_smoothing,
            audio_label_smoothing=self.config.audio_label_smoothing,
            end_token_weight=self.config.end_token_weight,
            text_loss_multiplier=self.config.text_loss_multiplier,
            audio_loss_multiplier=self.config.audio_loss_multiplier,
        )

    def _make_dataloader(
        self,
        dataset: LFM2DataLoader | TaskBalancedDataset,
        *,
        sampler: Sampler[int] | None = None,
    ) -> DataLoader[Any]:
        workers = self.config.dataloader_num_workers
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            sampler=sampler,
            collate_fn=lfm2_collator,
            num_workers=workers,
            pin_memory=self.accelerator.device.type == "cuda",
            persistent_workers=workers > 0,
            prefetch_factor=2 if workers > 0 else None,
        )

    def _new_train_iterator(self) -> Iterator[LFM2AudioModelInput]:
        loader = self.train_loader
        if self.state.batches_in_epoch:
            loader = self.accelerator.skip_first_batches(loader, self.state.batches_in_epoch)
        return iter(loader)

    def _lr_multiplier(self, step: int) -> float:
        warmup = self.config.resolved_warmup_steps
        if warmup > 0 and step < warmup:
            return max(1e-3, step / warmup)
        progress = (step - warmup) / max(1, self.config.max_steps - warmup)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.config.min_lr_ratio + (1.0 - self.config.min_lr_ratio) * cosine

    def _record_validation(self, metrics: dict[str, float]) -> bool:
        validation_loss = metrics["loss"]
        if validation_loss < self.state.best_validation_loss:
            self.state.best_validation_loss = validation_loss
            self.state.validations_without_improvement = 0
            self.accelerator.save_state(str(self.best_checkpoint_dir), safe_serialization=True)
        else:
            self.state.validations_without_improvement += 1

        patience = self.config.early_stopping_patience
        return patience is not None and self.state.validations_without_improvement >= patience

    def _save_final_model(self, *, stopped_early: bool) -> None:
        unwrapped = self.accelerator.unwrap_model(self.model)
        merged_lora_layers = merge_lora_layers(unwrapped)
        final_dir = self.output_dir / "final"
        self.accelerator.save_model(
            unwrapped,
            str(final_dir),
            max_shard_size="5GB",
            safe_serialization=True,
        )
        if self.accelerator.is_main_process:
            unwrapped.save_config(final_dir)
            with (final_dir / "training_config.json").open("w", encoding="utf-8") as stream:
                json.dump(self.config.to_dict(), stream, indent=2, sort_keys=True)
                stream.write("\n")
            summary = {
                "optimizer_steps": self.state.step,
                "epochs": self.state.epoch,
                "best_validation_loss": (
                    None if not math.isfinite(self.state.best_validation_loss) else self.state.best_validation_loss
                ),
                "stopped_early": stopped_early,
                "merged_lora_layers": merged_lora_layers,
                "fine_tuning": {
                    "strategy": self.fine_tune_report.strategy,
                    "trainable_parameters": self.fine_tune_report.trainable_parameters,
                    "total_parameters": self.fine_tune_report.total_parameters,
                    "trainable_percent": self.fine_tune_report.trainable_percent,
                },
            }
            with (final_dir / "training_summary.json").open("w", encoding="utf-8") as stream:
                json.dump(summary, stream, indent=2, sort_keys=True)
                stream.write("\n")

    def _resume_if_requested(self) -> None:
        checkpoint = self.config.resume_from_checkpoint
        if checkpoint is None:
            return
        checkpoint_path = self._latest_checkpoint() if checkpoint == "latest" else Path(checkpoint)
        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        self.accelerator.load_state(str(checkpoint_path))
        self.train_sampler.set_epoch(self.state.epoch)
        set_epoch = getattr(self.train_data, "set_epoch", None)
        if set_epoch is not None:
            set_epoch(self.state.epoch)
        self.accelerator.print(f"Resumed from {checkpoint_path} at optimizer step {self.state.step}")

    def _latest_checkpoint(self) -> Path | None:
        checkpoint_root = self.output_dir / "checkpoints"
        candidates = list(checkpoint_root.glob("checkpoint_*"))
        if not candidates:
            return None

        def checkpoint_number(path: Path) -> int:
            try:
                return int(path.name.rsplit("_", 1)[-1])
            except ValueError:
                return -1

        return max(candidates, key=checkpoint_number)

    def _enable_gradient_checkpointing(self) -> None:
        if not self.config.gradient_checkpointing:
            return
        lfm = self.model.lfm
        enable = getattr(lfm, "gradient_checkpointing_enable", None)
        if enable is None:
            self.accelerator.print("WARNING: LFM backbone does not expose gradient checkpointing; continuing without it")
            return
        try:
            enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            enable()
        if hasattr(lfm, "config"):
            lfm.config.use_cache = False

        conformer_layers = getattr(self.model.conformer, "layers", ())
        for layer in conformer_layers:
            if any(parameter.requires_grad for parameter in layer.parameters()):
                wrap_activation_checkpoint(layer)

    def _configure_numeric_precision(self) -> None:
        if self.accelerator.device.type != "cuda":
            return
        torch.backends.cuda.matmul.allow_tf32 = self.config.allow_tf32
        torch.backends.cudnn.allow_tf32 = self.config.allow_tf32

    @staticmethod
    def _resolve_mixed_precision(precision: str) -> str:
        if precision != "auto":
            return precision
        if not torch.cuda.is_available():
            return "no"
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"

    @staticmethod
    def _model_dtype(precision: str) -> torch.dtype:
        if precision == "bf16":
            return torch.bfloat16
        if precision == "fp16":
            return torch.float16
        return torch.float32

    def _print_startup_summary(self) -> None:
        report: FineTuneReport = self.fine_tune_report
        effective_batch = self.config.batch_size * self.config.gradient_accumulation_steps * self.accelerator.num_processes
        self.accelerator.print(
            f"[{self._elapsed()}] Start training: strategy={report.strategy} "
            f"trainable={report.trainable_parameters:,}/{report.total_parameters:,} "
            f"({report.trainable_percent:.2f}%) lora_layers={report.lora_layers} "
            f"effective_batch={effective_batch}"
        )

    def _emit_metrics(self, split: str, metrics: dict[str, float]) -> None:
        fields = " ".join(f"{name}={value:.5g}" for name, value in metrics.items())
        self.accelerator.print(
            f"[{self._elapsed()}] {split.upper()}: epoch={self.state.epoch} "
            f"step={self.state.step}/{self.config.max_steps} {fields}"
        )
        if self.config.log_jsonl and self.accelerator.is_main_process:
            record: dict[str, Any] = {
                "split": split,
                "step": self.state.step,
                "epoch": self.state.epoch,
                "elapsed_seconds": max(0.0, time.monotonic() - self.started_at),
                **metrics,
            }
            with self.metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")

    def _elapsed(self) -> str:
        total = int(max(0.0, time.monotonic() - self.started_at)) if self.started_at else 0
        hours, remainder = divmod(total, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
