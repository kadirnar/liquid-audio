from __future__ import annotations

from liquid_audio.training import EpochRandomSampler


def test_epoch_sampler_is_reconstructable_and_changes_between_epochs() -> None:
    first = EpochRandomSampler(range(20), seed=11)
    replay = EpochRandomSampler(range(20), seed=11)

    assert list(first) == list(replay)
    first.set_epoch(1)
    assert list(first) != list(replay)
    assert sorted(first) == list(range(20))
