from __future__ import annotations

import torch

from liquid_audio.decoding import (
    TextDecodingConfig,
    banned_next_tokens,
    has_repetition_loop,
    prepare_text_logits,
)


def test_banned_next_tokens_blocks_only_matching_ngram_continuations() -> None:
    history = [1, 2, 3, 1, 2]
    assert banned_next_tokens(history, 3) == {3}
    assert banned_next_tokens(history, 4) == set()


def test_repetition_penalty_does_not_mutate_logits_and_keeps_eos_available() -> None:
    logits = torch.tensor([0.0, 4.0, -2.0, 3.0])
    original = logits.clone()
    config = TextDecodingConfig(repetition_penalty=2.0, no_repeat_ngram_size=1, eos_token_id=3)

    adjusted = prepare_text_logits(logits, [1, 2, 3], config)

    assert torch.equal(logits, original)
    assert adjusted[1].isneginf()
    assert adjusted[2].isneginf()
    assert adjusted[3] == 1.5


def test_repeated_suffix_loop_detection() -> None:
    assert has_repetition_loop([4, 5, 4, 5, 4, 5], ngram_size=2, repeats=3)
    assert not has_repetition_loop([4, 5, 4, 5, 4, 6], ngram_size=2, repeats=3)
    assert not has_repetition_loop([4, 4], ngram_size=1, repeats=3)
