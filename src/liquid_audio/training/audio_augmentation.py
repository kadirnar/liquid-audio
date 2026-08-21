from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torchaudio


@dataclass(frozen=True, slots=True)
class WaveformAugmentationConfig:
    """Lightweight waveform augmentation for ASR preprocessing."""

    speed_probability: float = 0.5
    min_speed: float = 0.9
    max_speed: float = 1.1
    gain_probability: float = 0.5
    min_gain_db: float = -6.0
    max_gain_db: float = 6.0
    noise_probability: float = 0.35
    min_snr_db: float = 15.0
    max_snr_db: float = 35.0

    def validate(self) -> None:
        for name, probability in (
            ("speed_probability", self.speed_probability),
            ("gain_probability", self.gain_probability),
            ("noise_probability", self.noise_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if not 0 < self.min_speed <= self.max_speed:
            raise ValueError("speed range must be positive and ordered")
        if self.min_gain_db > self.max_gain_db or self.min_snr_db > self.max_snr_db:
            raise ValueError("augmentation ranges must be ordered")


class WaveformAugmenter:
    """Apply speed perturbation, gain, and additive white noise."""

    def __init__(self, config: WaveformAugmentationConfig) -> None:
        config.validate()
        self.config = config

    def __call__(self, waveform: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        augmented = waveform
        if self._sample_event(self.config.speed_probability):
            speed = self._uniform(self.config.min_speed, self.config.max_speed)
            target_rate = max(1, round(sampling_rate / speed))
            augmented = torchaudio.functional.resample(augmented, sampling_rate, target_rate)

        if self._sample_event(self.config.gain_probability):
            gain_db = self._uniform(self.config.min_gain_db, self.config.max_gain_db)
            augmented = augmented * math.pow(10.0, gain_db / 20.0)

        if self._sample_event(self.config.noise_probability):
            signal_rms = augmented.square().mean().sqrt()
            if signal_rms > 0:
                snr_db = self._uniform(self.config.min_snr_db, self.config.max_snr_db)
                noise_rms = signal_rms / math.pow(10.0, snr_db / 20.0)
                augmented = augmented + torch.randn_like(augmented) * noise_rms

        peak = augmented.abs().amax()
        if peak > 0.999:
            augmented = augmented * (0.999 / peak)
        return augmented

    @staticmethod
    def _sample_event(probability: float) -> bool:
        return bool(torch.rand(()) < probability)

    @staticmethod
    def _uniform(low: float, high: float) -> float:
        return low + (high - low) * float(torch.rand(()).item())
