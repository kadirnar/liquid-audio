from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class TextDecodingConfig:
    """Optional safeguards for text generation.

    Defaults preserve the original model behavior. For long-form ASR, a mild
    repetition penalty plus a repeated 4-gram loop detector prevents rare
    decoding failures from running until ``max_new_tokens``.
    """

    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    loop_ngram_size: int = 0
    max_consecutive_loop_repeats: int = 0
    eos_token_id: int = 7

    def validate(self) -> None:
        if self.repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be at least 1.0")
        if self.no_repeat_ngram_size < 0 or self.loop_ngram_size < 0:
            raise ValueError("n-gram sizes must be non-negative")
        if self.max_consecutive_loop_repeats < 0:
            raise ValueError("max_consecutive_loop_repeats must be non-negative")
        if (self.loop_ngram_size == 0) != (self.max_consecutive_loop_repeats == 0):
            raise ValueError("loop_ngram_size and max_consecutive_loop_repeats must both be enabled or disabled")


def banned_next_tokens(history: list[int], ngram_size: int) -> set[int]:
    """Return tokens that would repeat an n-gram already present in history."""

    if ngram_size <= 0 or len(history) + 1 < ngram_size:
        return set()
    if ngram_size == 1:
        return set(history)

    prefix = tuple(history[-(ngram_size - 1) :])
    banned: set[int] = set()
    for start in range(len(history) - ngram_size + 1):
        if tuple(history[start : start + ngram_size - 1]) == prefix:
            banned.add(history[start + ngram_size - 1])
    return banned


def prepare_text_logits(
    logits: torch.Tensor,
    history: list[int],
    config: TextDecodingConfig,
) -> torch.Tensor:
    """Apply repetition controls without mutating the model's logits tensor."""

    config.validate()
    adjusted = logits.clone()
    if config.repetition_penalty > 1.0 and history:
        token_ids = torch.as_tensor(sorted(set(history)), device=adjusted.device, dtype=torch.long)
        scores = adjusted[token_ids]
        adjusted[token_ids] = torch.where(
            scores < 0,
            scores * config.repetition_penalty,
            scores / config.repetition_penalty,
        )

    banned = banned_next_tokens(history, config.no_repeat_ngram_size)
    banned.discard(config.eos_token_id)
    if banned:
        token_ids = torch.as_tensor(sorted(banned), device=adjusted.device, dtype=torch.long)
        adjusted[token_ids] = -torch.inf
    return adjusted


def has_repetition_loop(history: list[int], *, ngram_size: int, repeats: int) -> bool:
    """Detect an exact n-gram repeated consecutively at the end of generation."""

    if ngram_size <= 0 or repeats <= 1 or len(history) < ngram_size * repeats:
        return False
    suffix = history[-ngram_size:]
    return all(
        history[-ngram_size * index : -ngram_size * (index - 1) if index > 1 else None] == suffix
        for index in range(1, repeats + 1)
    )
