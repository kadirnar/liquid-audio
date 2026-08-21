from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from liquid_audio.training.config import FineTuneConfig


class LoRALinear(nn.Module):
    """A dependency-free LoRA wrapper that preserves the original linear layer."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0 or rank > min(base.in_features, base.out_features):
            raise ValueError(f"LoRA rank must be in [1, {min(base.in_features, base.out_features)}], got {rank}")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(
            base.in_features,
            rank,
            bias=False,
            device=base.weight.device,
            dtype=torch.float32,
        )
        self.lora_b = nn.Linear(
            rank,
            base.out_features,
            bias=False,
            device=base.weight.device,
            dtype=torch.float32,
        )
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        base_output = self.base(values)
        lora_values = self.dropout(values).to(self.lora_a.weight.dtype)
        update = self.lora_b(self.lora_a(lora_values)).to(base_output.dtype)
        return base_output + update * self.scaling

    def merged(self) -> nn.Linear:
        merged = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        with torch.no_grad():
            delta = (self.lora_b.weight @ self.lora_a.weight).to(self.base.weight.dtype)
            merged.weight.copy_(self.base.weight + delta * self.scaling)
            if self.base.bias is not None and merged.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged


@dataclass(frozen=True, slots=True)
class FineTuneReport:
    strategy: str
    trainable_parameters: int
    total_parameters: int
    lora_layers: int

    @property
    def trainable_percent(self) -> float:
        return 100.0 * self.trainable_parameters / max(1, self.total_parameters)


def _matches_target(name: str, targets: tuple[str, ...]) -> bool:
    leaf_name = name.rsplit(".", 1)[-1]
    return leaf_name in targets or any(name.endswith(target) for target in targets)


def inject_lora_layers(
    module: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: tuple[str, ...],
) -> int:
    replacements: list[tuple[nn.Module, str, nn.Linear]] = []
    for full_name, child in module.named_modules():
        if not isinstance(child, nn.Linear) or not _matches_target(full_name, target_modules):
            continue
        parent_name, _, leaf_name = full_name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        replacements.append((parent, leaf_name, child))

    for parent, name, linear in replacements:
        setattr(parent, name, LoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout))
    return len(replacements)


def merge_lora_layers(module: nn.Module) -> int:
    replacements: list[tuple[nn.Module, str, LoRALinear]] = []
    for full_name, child in module.named_modules():
        if not isinstance(child, LoRALinear):
            continue
        parent_name, _, leaf_name = full_name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        replacements.append((parent, leaf_name, child))
    for parent, name, lora in replacements:
        setattr(parent, name, lora.merged())
    return len(replacements)


def _set_trainable(module: nn.Module, trainable: bool = True) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(trainable)


def _child_module(module: nn.Module, name: str) -> nn.Module:
    child = getattr(module, name, None)
    if not isinstance(child, nn.Module):
        raise TypeError(f"expected {name} to be an nn.Module")
    return child


def _unfreeze_normalization_and_biases(model: nn.Module, config: FineTuneConfig) -> None:
    normalization_types = (nn.LayerNorm, nn.RMSNorm)
    for module in model.modules():
        is_normalization = isinstance(module, normalization_types) or module.__class__.__name__.lower().endswith(
            ("layernorm", "rmsnorm")
        )
        if config.train_layer_norms and is_normalization:
            _set_trainable(module)
        if config.train_biases:
            bias = getattr(module, "bias", None)
            if isinstance(bias, nn.Parameter):
                bias.requires_grad_(True)


def _unfreeze_last_lfm_layers(model: nn.Module, count: int) -> None:
    if count == 0:
        return
    lfm = _child_module(model, "lfm")
    layers = getattr(lfm, "layers", None)
    if layers is None:
        layers = getattr(getattr(lfm, "model", None), "layers", None)
    if layers is None:
        raise ValueError("could not locate LFM transformer layers for progressive unfreezing")
    for layer in list(layers)[-count:]:
        _set_trainable(layer)


def configure_finetuning(model: nn.Module, config: FineTuneConfig) -> FineTuneReport:
    """Freeze the base model and enable only the requested task-specific updates."""

    config.validate()
    if config.strategy == "full":
        _set_trainable(model)
        total = sum(parameter.numel() for parameter in model.parameters())
        return FineTuneReport(config.strategy, total, total, 0)

    _set_trainable(model, False)
    default_asr = config.strategy in {"asr_adapter", "asr_lora", "omni_lora"}
    default_tts = config.strategy == "omni_lora"
    train_audio_encoder = default_asr if config.train_audio_encoder is None else config.train_audio_encoder
    train_audio_adapter = default_asr if config.train_audio_adapter is None else config.train_audio_adapter
    train_tts_head = default_tts if config.train_tts_head is None else config.train_tts_head

    if train_audio_encoder:
        _set_trainable(_child_module(model, "conformer"))
    if train_audio_adapter:
        _set_trainable(_child_module(model, "audio_adapter"))
    if train_tts_head:
        for name in ("audio_embedding", "depth_linear", "depthformer", "depth_embeddings"):
            _set_trainable(getattr(model, name))
    if config.train_text_embeddings:
        _set_trainable(_child_module(_child_module(model, "lfm"), "embed_tokens"))

    lora_layers = 0
    if config.strategy in {"asr_lora", "omni_lora"}:
        lora_layers = inject_lora_layers(
            _child_module(model, "lfm"),
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
        )
        if lora_layers == 0:
            raise ValueError(f"none of the requested LoRA targets were found: {config.lora_target_modules}")

    _unfreeze_last_lfm_layers(model, config.unfreeze_last_lfm_layers)
    if config.strategy in {"asr_lora", "omni_lora"}:
        _unfreeze_normalization_and_biases(_child_module(model, "lfm"), config)

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable == 0:
        raise ValueError("fine-tuning strategy selected no trainable parameters")
    return FineTuneReport(config.strategy, trainable, total, lora_layers)


def optimizer_parameter_groups(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    audio_encoder_lr_scale: float,
) -> list[dict[str, object]]:
    """Build AdamW groups with no decay for norms/biases and a safer encoder LR."""

    groups: dict[tuple[float, float], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        use_decay = parameter.ndim >= 2 and not name.endswith("bias")
        decay = weight_decay if use_decay else 0.0
        lr_scale = audio_encoder_lr_scale if name.startswith("conformer.") else 1.0
        groups.setdefault((decay, lr_scale), []).append(parameter)

    return [
        {
            "params": parameters,
            "weight_decay": decay,
            "lr": learning_rate * lr_scale,
            "initial_lr": learning_rate * lr_scale,
        }
        for (decay, lr_scale), parameters in groups.items()
    ]


def promote_trainable_parameters_to_fp32(model: nn.Module) -> int:
    """Keep optimizer-owned weights in FP32 while autocast handles compute precision."""

    promoted = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.is_floating_point() and parameter.dtype != torch.float32
    )
    for child in model.children():
        child_parameters = list(child.parameters())
        if child_parameters and all(parameter.requires_grad for parameter in child_parameters):
            child.to(dtype=torch.float32)

    for parameter in model.parameters():
        if parameter.requires_grad and parameter.is_floating_point() and parameter.dtype != torch.float32:
            parameter.data = parameter.data.to(torch.float32)
    return promoted
