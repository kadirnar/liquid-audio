from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from datasets import load_from_disk

from liquid_audio.data.preprocess import preprocess_dataset
from liquid_audio.data.types import ChatMessage, LFM2AudioTrainingSample


class TinyMapper:
    codebooks = 8

    def __call__(self, messages: list[ChatMessage]) -> LFM2AudioTrainingSample:
        del messages
        return LFM2AudioTrainingSample(
            text=torch.tensor([[1, 2, 3]]),
            audio_in=torch.empty(128, 0),
            audio_in_lens=torch.empty(0, dtype=torch.long),
            audio_out=torch.empty(8, 0, dtype=torch.long),
            modality_flag=torch.ones(1, 3, dtype=torch.long),
            supervision_mask=torch.ones(1, 3, dtype=torch.bool),
        )


def test_preprocessing_is_atomic_and_records_provenance(tmp_path: Path) -> None:
    output = tmp_path / "dataset"
    stats = preprocess_dataset([[]], output, TinyMapper(), max_context_length=5)  # type: ignore[arg-type]

    assert stats.written_samples == 1
    assert len(load_from_disk(output)) == 1
    metadata = json.loads((output / "preprocessing_stats.json").read_text(encoding="utf-8"))
    assert metadata["codebooks"] == 8
    assert metadata["max_context_length"] == 5
    assert not list(tmp_path.glob(".dataset-*"))

    with pytest.raises(FileExistsError):
        preprocess_dataset([[]], output, TinyMapper())  # type: ignore[arg-type]


def test_preprocessing_rejects_an_all_skipped_dataset(tmp_path: Path) -> None:
    output = tmp_path / "empty"
    with pytest.raises(ValueError, match="no samples"):
        preprocess_dataset([[]], output, TinyMapper(), max_context_length=2)  # type: ignore[arg-type]
    assert not output.exists()
