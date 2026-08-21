from __future__ import annotations

import json

import pytest

from liquid_audio.training import TrainerConfig


def test_recipe_round_trip(tmp_path) -> None:
    path = tmp_path / "recipe.json"
    original = TrainerConfig()
    path.write_text(json.dumps(original.to_dict()), encoding="utf-8")

    loaded = TrainerConfig.from_json(path)

    assert loaded.to_dict() == original.to_dict()
    assert isinstance(loaded.betas, tuple)
    assert isinstance(loaded.fine_tune.lora_target_modules, tuple)


def test_invalid_interval_is_rejected() -> None:
    config = TrainerConfig(logging_interval=0)
    with pytest.raises(ValueError, match="intervals"):
        config.validate()
