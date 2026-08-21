from __future__ import annotations

from liquid_audio.evaluation import corpus_wer, normalize_english_for_wer


def test_wer_normalization_and_error_counts() -> None:
    assert normalize_english_for_wer("  HELLO—World!  ") == "hello world"
    summary = corpus_wer(["Hello world", "good day"], ["hello word", "good day now"])
    assert summary.reference_words == 4
    assert summary.substitutions == 1
    assert summary.deletions == 0
    assert summary.insertions == 1
    assert summary.word_errors == 2
    assert summary.wer_percent == 50.0
