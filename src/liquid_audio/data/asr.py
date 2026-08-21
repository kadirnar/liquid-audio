from __future__ import annotations

import io
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import soundfile
import torch

from liquid_audio.data.types import AudioSegment, ChatMessage, TextSegment
from liquid_audio.training.audio_augmentation import WaveformAugmenter


class ASRChatIterator:
    """Convert manifest or Hugging Face rows to supervised ASR chats."""

    def __init__(
        self,
        rows: Iterable[dict[str, Any]],
        *,
        audio_column: str,
        text_column: str,
        base_dir: Path | None = None,
        augmentation_copies: int = 0,
        augmenter: WaveformAugmenter | None = None,
        system_prompt: str = "Perform ASR.",
    ) -> None:
        if augmentation_copies < 0:
            raise ValueError("augmentation_copies must be non-negative")
        if augmentation_copies and augmenter is None:
            raise ValueError("augmentation_copies requires an augmenter")
        self.rows = rows
        self.audio_column = audio_column
        self.text_column = text_column
        self.base_dir = base_dir
        self.augmentation_copies = augmentation_copies
        self.augmenter = augmenter
        self.system_prompt = system_prompt

    def __iter__(self) -> Iterator[list[ChatMessage]]:
        for index, row in enumerate(self.rows):
            raw_transcript = row.get(self.text_column)
            transcript = "" if raw_transcript is None else str(raw_transcript).strip()
            if not transcript:
                print(f"WARNING: skipping row {index} with an empty transcript")
                continue
            audio = load_audio_bytes(row[self.audio_column], base_dir=self.base_dir)
            for copy_index in range(self.augmentation_copies + 1):
                current_audio = audio
                if copy_index > 0:
                    assert self.augmenter is not None
                    current_audio = augment_audio_bytes(audio, self.augmenter)
                yield [
                    ChatMessage(role="system", content=[TextSegment(text=self.system_prompt)]),
                    ChatMessage(role="user", content=[AudioSegment(audio=current_audio)]),
                    ChatMessage(role="assistant", content=[TextSegment(text=transcript)]),
                ]


def load_audio_bytes(value: Any, *, base_dir: Path | None = None) -> bytes:
    if isinstance(value, dict):
        raw_bytes = value.get("bytes")
        if raw_bytes is not None:
            return bytes(raw_bytes)
        value = value.get("path")
    if not isinstance(value, (str, Path)):
        raise TypeError(f"audio value must be a path or an Audio(decode=False) object, got {type(value).__name__}")
    audio_path = Path(value)
    if not audio_path.is_absolute() and base_dir is not None:
        audio_path = base_dir / audio_path
    return audio_path.read_bytes()


def decode_audio_bytes(audio: bytes) -> tuple[torch.Tensor, int]:
    with io.BytesIO(audio) as source:
        values, sampling_rate = soundfile.read(source, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(values.T.copy())
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sampling_rate


def augment_audio_bytes(audio: bytes, augmenter: WaveformAugmenter) -> bytes:
    waveform, sampling_rate = decode_audio_bytes(audio)
    augmented = augmenter(waveform, sampling_rate)
    with io.BytesIO() as destination:
        soundfile.write(destination, augmented.mT.numpy(), sampling_rate, format="WAV", subtype="FLOAT")
        return destination.getvalue()


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{manifest_path}:{line_number} must contain a JSON object")
            yield row
