from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Self

FineTuneStrategy = Literal["full", "asr_adapter", "asr_lora", "omni_lora"]
MixedPrecision = Literal["auto", "no", "fp16", "bf16"]


@dataclass(slots=True)
class SpecAugmentConfig:
    """Dynamic log-mel augmentation applied independently to every utterance."""

    probability: float = 0.8
    time_masks: int = 2
    max_time_width: int = 80
    max_time_ratio: float = 0.08
    frequency_masks: int = 2
    max_frequency_width: int = 18

    def validate(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("SpecAugment probability must be between 0 and 1")
        if self.time_masks < 0 or self.frequency_masks < 0:
            raise ValueError("SpecAugment mask counts must be non-negative")
        if self.max_time_width < 0 or self.max_frequency_width < 0:
            raise ValueError("SpecAugment mask widths must be non-negative")
        if not 0.0 <= self.max_time_ratio < 1.0:
            raise ValueError("SpecAugment max_time_ratio must be in [0, 1)")


@dataclass(slots=True)
class FineTuneConfig:
    """Controls which parameters are updated during fine-tuning.

    ``asr_lora`` is the practical default for a 16 GB GPU: it trains the audio
    encoder and adapter while adding low-rank updates to the LFM text backbone.
    TTS-only modules stay frozen. ``omni_lora`` additionally trains the audio
    output stack for joint ASR/TTS/speech-to-speech data.
    """

    strategy: FineTuneStrategy = "asr_lora"
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "w1",
        "w2",
        "w3",
    )
    train_audio_encoder: bool | None = None
    train_audio_adapter: bool | None = None
    train_tts_head: bool | None = None
    train_text_embeddings: bool = False
    train_layer_norms: bool = True
    train_biases: bool = False
    unfreeze_last_lfm_layers: int = 0

    def validate(self) -> None:
        if self.strategy not in {"full", "asr_adapter", "asr_lora", "omni_lora"}:
            raise ValueError(f"unsupported fine-tuning strategy: {self.strategy}")
        if self.lora_rank <= 0:
            raise ValueError("lora_rank must be positive")
        if self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be positive")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.unfreeze_last_lfm_layers < 0:
            raise ValueError("unfreeze_last_lfm_layers must be non-negative")


@dataclass(slots=True)
class TrainerConfig:
    """Serializable training configuration used by :class:`Trainer`."""

    model_id: str = "LiquidAI/LFM2.5-Audio-1.5B"
    revision: str | None = None
    output_dir: str = "outputs/lfm2-audio"
    max_steps: int = 1_000
    warmup_steps: int | None = None
    warmup_ratio: float = 0.03
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    audio_encoder_lr_scale: float = 0.25
    betas: tuple[float, float] = (0.9, 0.95)
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.1
    min_lr_ratio: float = 0.1
    max_grad_norm: float = 1.0
    mixed_precision: MixedPrecision = "auto"
    trainable_parameters_fp32: bool = True
    seed: int = 42
    dataloader_num_workers: int = 0
    logging_interval: int = 10
    save_interval: int = 500
    validation_interval: int = 100
    checkpoint_limit: int = 3
    resume_from_checkpoint: str | None = None
    early_stopping_patience: int | None = None
    gradient_checkpointing: bool = True
    allow_tf32: bool = True
    text_label_smoothing: float = 0.05
    audio_label_smoothing: float = 0.0
    end_token_weight: float = 1.5
    text_loss_multiplier: float | None = None
    audio_loss_multiplier: float | None = None
    log_jsonl: bool = True
    fine_tune: FineTuneConfig = field(default_factory=FineTuneConfig)
    specaugment: SpecAugmentConfig | None = field(default_factory=SpecAugmentConfig)

    def validate(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.batch_size <= 0 or self.gradient_accumulation_steps <= 0:
            raise ValueError("batch_size and gradient_accumulation_steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < self.audio_encoder_lr_scale <= 1.0:
            raise ValueError("audio_encoder_lr_scale must be in (0, 1]")
        if self.warmup_steps is not None and self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if self.max_grad_norm < 0:
            raise ValueError("max_grad_norm must be non-negative")
        if min(self.logging_interval, self.save_interval, self.validation_interval, self.checkpoint_limit) <= 0:
            raise ValueError("logging, save, validation, and checkpoint intervals must be positive")
        if self.mixed_precision not in {"auto", "no", "fp16", "bf16"}:
            raise ValueError(f"unsupported mixed precision mode: {self.mixed_precision}")
        for name, value in (
            ("text_label_smoothing", self.text_label_smoothing),
            ("audio_label_smoothing", self.audio_label_smoothing),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.end_token_weight <= 0:
            raise ValueError("end_token_weight must be positive")
        multipliers = (self.text_loss_multiplier, self.audio_loss_multiplier)
        if any(multiplier is not None and multiplier < 0 for multiplier in multipliers):
            raise ValueError("loss multipliers must be non-negative")
        if all(multiplier == 0 for multiplier in multipliers):
            raise ValueError("text_loss_multiplier and audio_loss_multiplier cannot both be zero")
        if self.early_stopping_patience is not None and self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        self.fine_tune.validate()
        if self.specaugment is not None:
            self.specaugment.validate()

    @property
    def resolved_warmup_steps(self) -> int:
        if self.warmup_steps is not None:
            return min(self.warmup_steps, self.max_steps)
        return min(round(self.max_steps * self.warmup_ratio), self.max_steps)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Self:
        values = dict(values)
        fine_tune = dict(values.pop("fine_tune", {}))
        specaugment = values.pop("specaugment", {})
        if "betas" in values:
            values["betas"] = tuple(values["betas"])
        if "lora_target_modules" in fine_tune:
            fine_tune["lora_target_modules"] = tuple(fine_tune["lora_target_modules"])
        config = cls(
            **values,
            fine_tune=FineTuneConfig(**fine_tune),
            specaugment=None if specaugment is None else SpecAugmentConfig(**specaugment),
        )
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> Self:
        with Path(path).open(encoding="utf-8") as stream:
            values = json.load(stream)
        if not isinstance(values, dict):
            raise TypeError("trainer config JSON must contain an object")
        return cls.from_dict(values)
