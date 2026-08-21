from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset, load_from_disk
from torch.utils.data import Dataset as TorchDataset

from liquid_audio.data.types import LFM2AudioModelInput, LFM2AudioRow
from liquid_audio.training.augmentation import SpecAugment
from liquid_audio.training.config import SpecAugmentConfig
from liquid_audio.utils import LFMModality


class LFM2DataLoader(TorchDataset[LFM2AudioRow]):
    def __init__(
        self,
        dataset_path: str,
        context_length: int = 4096,
        specaugment: SpecAugmentConfig | None = None,
        dynamic_padding: bool = False,
        augmentation_seed: int = 42,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.context_length = context_length
        self.dataset: Dataset = load_from_disk(self.dataset_path)
        self.specaugment = SpecAugment(specaugment) if specaugment is not None else None
        self.dynamic_padding = dynamic_padding
        self.augmentation_seed = augmentation_seed
        self._epoch = torch.zeros((), dtype=torch.long).share_memory_()

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        self._epoch.fill_(epoch)

    def set_augmentation_seed(self, seed: int) -> None:
        self.augmentation_seed = seed

    def __getitem__(self, idx: int) -> LFM2AudioRow:
        row = self.dataset[idx]

        text = torch.as_tensor(row["text"], dtype=torch.long)
        audio_in = torch.as_tensor(row["audio_in"], dtype=torch.float32)
        audio_in_lens = torch.as_tensor(row["audio_in_lens"], dtype=torch.long)
        audio_out = torch.as_tensor(row["audio_out"], dtype=torch.long)
        modality = torch.as_tensor(row["modality_flag"], dtype=torch.long)
        supervision = torch.as_tensor(row["supervision_mask"], dtype=torch.bool)

        if self.specaugment is not None:
            epoch = int(self._epoch.item())
            generator = torch.Generator().manual_seed(self.augmentation_seed + epoch * len(self) + idx)
            audio_in = self.specaugment(audio_in, audio_in_lens, generator=generator)

        pad_len = self.context_length - int(modality.shape[1])
        if pad_len < 0:
            raise ValueError(
                f"sample at index {idx} has {modality.shape[1]} tokens, "
                f"which is longer than context_length={self.context_length}"
            )

        if not self.dynamic_padding:
            text = F.pad(text, (0, pad_len))
            modality = F.pad(modality, (0, pad_len), value=int(LFMModality.TEXT))
            supervision = F.pad(supervision, (0, pad_len), value=False)

        return LFM2AudioRow(
            text=text,
            audio_in=audio_in,
            audio_in_lens=audio_in_lens,
            audio_out=audio_out,
            modality_flag=modality,
            supervision_mask=supervision,
        )


class TaskBalancedDataset(TorchDataset[LFM2AudioRow]):
    """Sample task datasets by explicit probability instead of corpus size.

    The index-to-sample mapping is deterministic for a given seed and epoch,
    which keeps distributed workers reproducible while still changing the
    mixture each epoch.
    """

    def __init__(
        self,
        datasets: list[LFM2DataLoader],
        *,
        weights: list[float] | None = None,
        samples_per_epoch: int | None = None,
        seed: int = 42,
    ) -> None:
        if not datasets:
            raise ValueError("at least one task dataset is required")
        if any(len(dataset) == 0 for dataset in datasets):
            raise ValueError("task datasets must not be empty")
        resolved_weights = weights or [1.0] * len(datasets)
        if len(resolved_weights) != len(datasets):
            raise ValueError("provide exactly one weight per task dataset")
        if any(weight <= 0 for weight in resolved_weights):
            raise ValueError("task weights must be positive")
        resolved_samples = samples_per_epoch or sum(len(dataset) for dataset in datasets)
        if resolved_samples <= 0:
            raise ValueError("samples_per_epoch must be positive")

        self.datasets = datasets
        self.weights = torch.tensor(resolved_weights, dtype=torch.float64)
        self.weights /= self.weights.sum()
        self.samples_per_epoch = resolved_samples
        self.seed = seed
        self._epoch = torch.zeros((), dtype=torch.long).share_memory_()

    @property
    def specaugment(self) -> SpecAugment | None:
        transforms = [dataset.specaugment for dataset in self.datasets]
        return transforms[0] if transforms and all(transform is transforms[0] for transform in transforms) else None

    @specaugment.setter
    def specaugment(self, transform: SpecAugment | None) -> None:
        for dataset in self.datasets:
            dataset.specaugment = transform

    def set_epoch(self, epoch: int) -> None:
        self._epoch.fill_(epoch)
        for dataset in self.datasets:
            set_epoch = getattr(dataset, "set_epoch", None)
            if set_epoch is not None:
                set_epoch(epoch)

    def set_augmentation_seed(self, seed: int) -> None:
        for offset, dataset in enumerate(self.datasets):
            set_seed = getattr(dataset, "set_augmentation_seed", None)
            if set_seed is not None:
                set_seed(seed + offset * 1_000_003)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> LFM2AudioRow:
        epoch = int(self._epoch.item())
        generator = torch.Generator().manual_seed(self.seed + epoch * self.samples_per_epoch + idx)
        task_index = int(torch.multinomial(self.weights, 1, generator=generator).item())
        dataset = self.datasets[task_index]
        sample_index = int(torch.randint(len(dataset), (), generator=generator).item())
        return dataset[sample_index]


def lfm2_collator(batch: list[LFM2AudioRow]) -> LFM2AudioModelInput:
    if not batch:
        raise ValueError("cannot collate an empty batch")
    audio_in = torch.cat([row.audio_in for row in batch], dim=1)
    audio_in_lens = torch.cat([row.audio_in_lens for row in batch], dim=0)

    target_length = max(row.modality_flag.shape[1] for row in batch)
    text_rows: list[torch.Tensor] = []
    modality_rows: list[torch.Tensor] = []
    supervision_rows: list[torch.Tensor] = []
    for row in batch:
        pad_len = target_length - row.modality_flag.shape[1]
        text_rows.append(F.pad(row.text, (0, pad_len)))
        modality_rows.append(F.pad(row.modality_flag, (0, pad_len), value=int(LFMModality.TEXT)))
        supervision_rows.append(F.pad(row.supervision_mask, (0, pad_len), value=False))

    text = torch.cat(text_rows, dim=1)
    audio_out = torch.cat([row.audio_out for row in batch], dim=1)

    modality_flag = torch.cat(modality_rows, dim=0)
    supervision_mask = torch.cat(supervision_rows, dim=0)

    return LFM2AudioModelInput(
        text=text,
        audio_in=audio_in,
        audio_in_lens=audio_in_lens,
        audio_out=audio_out,
        modality_flag=modality_flag,
        supervision_mask=supervision_mask,
    )
