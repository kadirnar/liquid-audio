from __future__ import annotations

from typing import Any, cast

from liquid_audio.data.dataloader import LFM2DataLoader, TaskBalancedDataset


class LabeledDataset:
    def __init__(self, label: str, length: int) -> None:
        self.label = label
        self.length = length
        self.specaugment = None

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.label, index


def test_task_balancing_uses_task_probabilities_not_corpus_sizes() -> None:
    small = cast(LFM2DataLoader, cast(Any, LabeledDataset("small", 2)))
    large = cast(LFM2DataLoader, cast(Any, LabeledDataset("large", 200)))
    mixture = TaskBalancedDataset([small, large], weights=[0.8, 0.2], samples_per_epoch=1_000, seed=7)

    labels = [mixture[index][0] for index in range(len(mixture))]  # type: ignore[index]

    assert 750 <= labels.count("small") <= 850
    assert labels == [mixture[index][0] for index in range(len(mixture))]  # type: ignore[index]
    mixture.set_epoch(1)
    assert labels != [mixture[index][0] for index in range(len(mixture))]  # type: ignore[index]
