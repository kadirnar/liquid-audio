from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from liquid_audio.data.types import LFM2AudioModelInput, LFM2AudioRow
from liquid_audio.model.lfm2_audio import LFM2AudioModelOutput
from liquid_audio.trainer import Trainer
from liquid_audio.training import FineTuneConfig, TrainerConfig


class TinyDataset:
    specaugment = None

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> LFM2AudioRow:
        del index
        return LFM2AudioRow(
            text=torch.tensor([[1, 2]], dtype=torch.long),
            audio_in=torch.empty(128, 0),
            audio_in_lens=torch.empty(0, dtype=torch.long),
            audio_out=torch.empty(8, 0, dtype=torch.long),
            modality_flag=torch.ones(1, 2, dtype=torch.long),
            supervision_mask=torch.ones(1, 2, dtype=torch.bool),
        )


class TinyTrainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch: LFM2AudioModelInput, **kwargs: Any) -> LFM2AudioModelOutput:
        del batch, kwargs
        loss = self.weight.square()
        zero = loss.detach() * 0
        return LFM2AudioModelOutput(
            loss=loss,
            audio_loss=zero,
            text_loss=loss,
            audio_out_tokens=torch.tensor(0),
            text_tokens=torch.tensor(1),
            audio_in_tokens=torch.tensor(0),
        )

    def save_config(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "config.json").write_text("{}\n", encoding="utf-8")


def test_trainer_accumulates_and_writes_loadable_final_artifacts(tmp_path) -> None:
    config = TrainerConfig(
        output_dir=str(tmp_path),
        max_steps=2,
        warmup_steps=0,
        batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-2,
        mixed_precision="no",
        dataloader_num_workers=0,
        logging_interval=1,
        save_interval=1,
        validation_interval=1,
        checkpoint_limit=2,
        gradient_checkpointing=False,
        text_label_smoothing=0.0,
        end_token_weight=1.0,
        fine_tune=FineTuneConfig(strategy="full"),
        specaugment=None,
    )
    trainer = Trainer(
        config=config,
        train_data=TinyDataset(),  # type: ignore[arg-type]
        model=TinyTrainModel(),  # type: ignore[arg-type]
    )

    trainer.train()

    assert trainer.step == 2
    assert (tmp_path / "final" / "model.safetensors").is_file()
    assert (tmp_path / "final" / "config.json").is_file()
    summary = json.loads((tmp_path / "final" / "training_summary.json").read_text(encoding="utf-8"))
    assert summary["optimizer_steps"] == 2

    resumed = Trainer(
        config=replace(config, max_steps=3, resume_from_checkpoint="latest"),
        train_data=TinyDataset(),  # type: ignore[arg-type]
        model=TinyTrainModel(),  # type: ignore[arg-type]
    )
    resumed.train()
    assert resumed.step == 3
    assert resumed.epoch == 2
