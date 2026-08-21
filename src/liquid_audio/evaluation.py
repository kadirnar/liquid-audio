from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from jiwer import process_words


def normalize_english_for_wer(text: str) -> str:
    """Apply a transparent, dependency-light normalization for model comparisons."""

    normalized = unicodedata.normalize("NFKC", text).casefold().replace("_", " ")
    normalized = re.sub(r"[^\w\s']", " ", normalized)
    return " ".join(normalized.split())


@dataclass(frozen=True, slots=True)
class WERSummary:
    samples: int
    reference_words: int
    substitutions: int
    deletions: int
    insertions: int
    wer_percent: float

    @property
    def word_errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


def corpus_wer(references: list[str], predictions: list[str]) -> WERSummary:
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have the same length")
    normalized_pairs = [
        (normalize_english_for_wer(reference), normalize_english_for_wer(prediction))
        for reference, prediction in zip(references, predictions, strict=True)
    ]
    normalized_pairs = [(reference, prediction) for reference, prediction in normalized_pairs if reference]
    if not normalized_pairs:
        raise ValueError("evaluation has no non-empty references after normalization")
    normalized_references, normalized_predictions = zip(*normalized_pairs, strict=True)
    details = process_words(list(normalized_references), list(normalized_predictions))
    reference_words = details.hits + details.substitutions + details.deletions
    return WERSummary(
        samples=len(normalized_pairs),
        reference_words=reference_words,
        substitutions=details.substitutions,
        deletions=details.deletions,
        insertions=details.insertions,
        wer_percent=100.0 * details.wer,
    )
