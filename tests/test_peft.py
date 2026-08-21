from __future__ import annotations

from typing import cast

import torch
from torch import nn

from liquid_audio.training import FineTuneConfig
from liquid_audio.training.peft import (
    LoRALinear,
    configure_finetuning,
    merge_lora_layers,
    promote_trainable_parameters_to_fp32,
)


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.norm = nn.RMSNorm(4)


class TinyLFM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 4)
        self.layers = nn.ModuleList([TinyBlock(), TinyBlock()])


class TinyAudioModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lfm = TinyLFM()
        self.conformer = nn.Linear(4, 4)
        self.audio_adapter = nn.Linear(4, 4)
        self.audio_embedding = nn.Embedding(16, 4)
        self.depth_linear = nn.Linear(4, 4)
        self.depthformer = nn.Linear(4, 4)
        self.depth_embeddings = nn.ModuleList([nn.Embedding(16, 4)])


def test_asr_lora_selects_only_asr_path_and_merges_exactly() -> None:
    torch.manual_seed(3)
    model = TinyAudioModel()
    report = configure_finetuning(
        model,
        FineTuneConfig(
            strategy="asr_lora",
            lora_rank=2,
            lora_alpha=4.0,
            lora_dropout=0.0,
            lora_target_modules=("q_proj",),
            train_layer_norms=False,
        ),
    )

    assert report.lora_layers == 2
    assert isinstance(model.lfm.layers[0].q_proj, LoRALinear)
    assert model.conformer.weight.requires_grad
    assert model.audio_adapter.weight.requires_grad
    assert not model.depth_linear.weight.requires_grad

    lora = model.lfm.layers[0].q_proj
    values = torch.randn(3, 4)
    with torch.no_grad():
        lora.lora_b.weight.fill_(0.1)
        expected = lora(values)
    assert merge_lora_layers(model) == 2
    actual = model.lfm.layers[0].q_proj(values)
    torch.testing.assert_close(actual, expected)


def test_only_trainable_parameters_are_promoted_to_fp32() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4)).to(torch.bfloat16)
    model[1].requires_grad_(False)
    first = cast(nn.Linear, model[0])
    second = cast(nn.Linear, model[1])

    promoted = promote_trainable_parameters_to_fp32(model)

    assert first.bias is not None
    assert promoted == first.weight.numel() + first.bias.numel()
    assert first.weight.dtype == torch.float32
    assert second.weight.dtype == torch.bfloat16
