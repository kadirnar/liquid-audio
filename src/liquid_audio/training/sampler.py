from __future__ import annotations

from collections.abc import Iterator, Sized

import torch
from torch.utils.data import Sampler


class EpochRandomSampler(Sampler[int]):
    """A shuffle order determined only by ``seed`` and ``epoch``.

    Unlike a stateful generator-backed RandomSampler, this order can be
    reconstructed after a process restart and safely fast-forwarded.
    """

    def __init__(self, data_source: Sized, *, seed: int) -> None:
        self.data_source = data_source
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        yield from torch.randperm(len(self.data_source), generator=generator).tolist()

    def __len__(self) -> int:
        return len(self.data_source)
