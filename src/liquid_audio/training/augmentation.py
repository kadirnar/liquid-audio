from __future__ import annotations

import torch

from liquid_audio.training.config import SpecAugmentConfig


class SpecAugment:
    """Apply time and frequency masks to concatenated log-mel features.

    Audio segments are masked independently, so a long utterance cannot consume
    the masking budget of another utterance packed into the same training row.
    The transform is intentionally dynamic and therefore produces a new view on
    every epoch instead of baking one augmented copy into the dataset.
    """

    def __init__(self, config: SpecAugmentConfig) -> None:
        config.validate()
        self.config = config

    def __call__(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"expected [mel, time] features, got shape {tuple(features.shape)}")
        if lengths.ndim != 1:
            raise ValueError(f"expected one-dimensional lengths, got shape {tuple(lengths.shape)}")
        if int(lengths.sum()) != features.shape[1]:
            raise ValueError("audio feature lengths do not add up to the time dimension")
        if features.numel() == 0 or torch.rand((), generator=generator).item() >= self.config.probability:
            return features

        augmented = features.clone()
        offset = 0
        for raw_length in lengths.tolist():
            length = int(raw_length)
            segment = augmented[:, offset : offset + length]
            self._mask_segment(segment, generator=generator)
            offset += length
        return augmented

    def _mask_segment(self, segment: torch.Tensor, *, generator: torch.Generator | None) -> None:
        frequency_bins, time_steps = segment.shape
        if time_steps == 0:
            return

        time_limit = min(
            self.config.max_time_width,
            max(0, int(time_steps * self.config.max_time_ratio)),
            max(0, time_steps - 1),
        )
        for _ in range(self.config.time_masks):
            self._mask_axis(segment, axis=1, size=time_steps, max_width=time_limit, generator=generator)

        frequency_limit = min(self.config.max_frequency_width, max(0, frequency_bins - 1))
        for _ in range(self.config.frequency_masks):
            self._mask_axis(segment, axis=0, size=frequency_bins, max_width=frequency_limit, generator=generator)

    @staticmethod
    def _mask_axis(
        values: torch.Tensor,
        *,
        axis: int,
        size: int,
        max_width: int,
        generator: torch.Generator | None,
    ) -> None:
        if max_width <= 0:
            return
        width = int(torch.randint(0, max_width + 1, (), generator=generator).item())
        if width == 0:
            return
        start = int(torch.randint(0, size - width + 1, (), generator=generator).item())
        if axis == 0:
            values[start : start + width, :] = 0
        else:
            values[:, start : start + width] = 0
