from __future__ import annotations

import torch

from liquid_audio.training import SpecAugment, SpecAugmentConfig
from liquid_audio.training.audio_augmentation import WaveformAugmentationConfig, WaveformAugmenter


def test_specaugment_preserves_shape_and_does_not_mutate_input() -> None:
    torch.manual_seed(7)
    features = torch.ones(32, 100)
    original = features.clone()
    transform = SpecAugment(
        SpecAugmentConfig(
            probability=1.0,
            time_masks=3,
            max_time_width=20,
            max_time_ratio=0.2,
            frequency_masks=3,
            max_frequency_width=8,
        )
    )

    augmented = transform(features, torch.tensor([40, 60]))

    assert augmented.shape == features.shape
    assert torch.equal(features, original)
    assert torch.count_nonzero(augmented == 0) > 0
    assert torch.count_nonzero(augmented) > 0


def test_specaugment_probability_zero_returns_original_tensor() -> None:
    features = torch.randn(8, 20)
    transform = SpecAugment(SpecAugmentConfig(probability=0.0))
    assert transform(features, torch.tensor([20])) is features


def test_specaugment_is_replayable_with_an_explicit_generator() -> None:
    features = torch.ones(32, 100)
    transform = SpecAugment(SpecAugmentConfig(probability=1.0))

    first = transform(features, torch.tensor([100]), generator=torch.Generator().manual_seed(19))
    replay = transform(features, torch.tensor([100]), generator=torch.Generator().manual_seed(19))

    assert torch.equal(first, replay)


def test_waveform_augmenter_guards_against_clipping() -> None:
    torch.manual_seed(1)
    waveform = torch.ones(1, 1600) * 0.9
    augmenter = WaveformAugmenter(
        WaveformAugmentationConfig(
            speed_probability=0.0,
            gain_probability=1.0,
            min_gain_db=12.0,
            max_gain_db=12.0,
            noise_probability=0.0,
        )
    )
    augmented = augmenter(waveform, 16_000)
    assert float(augmented.abs().max()) <= 0.9991
