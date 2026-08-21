from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import datasets
from datasets import Features, Sequence, Value

from liquid_audio.data.mapper import LFM2AudioChatMapper
from liquid_audio.data.types import ChatMessage


@dataclass(frozen=True, slots=True)
class PreprocessingStats:
    seen_samples: int
    written_samples: int
    skipped_samples: int
    max_sequence_length: int


def preprocess_dataset(
    data: Iterable[list[ChatMessage]],
    output_path: str | Path,
    mapper: LFM2AudioChatMapper,
    max_context_length: int = -1,
) -> PreprocessingStats:
    out_dir = Path(output_path)
    if out_dir.exists():
        raise FileExistsError(f"output path already exists: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    features = Features(
        {
            "text": Sequence(Sequence(Value("int64"))),
            "audio_in": Sequence(Sequence(Value("float32"))),
            "audio_in_lens": Sequence(Value("int64")),
            "audio_out": Sequence(Sequence(Value("int64"))),
            "modality_flag": Sequence(Sequence(Value("int64"))),
            "supervision_mask": Sequence(Sequence(Value("bool"))),
        }
    )

    counters = {"seen": 0, "written": 0, "skipped": 0, "max_length": 0}

    def generator():
        for i, messages in enumerate(data):
            counters["seen"] += 1
            sample = mapper(messages)
            sample_len = int(sample.modality_flag.shape[-1])
            counters["max_length"] = max(counters["max_length"], sample_len)
            if 0 <= max_context_length < sample_len:
                print(f"WARNING: skipping sample {i} with {sample_len} tokens (max_context_length={max_context_length})")
                counters["skipped"] += 1
                continue
            counters["written"] += 1
            yield {
                "text": sample.text.tolist(),
                "audio_in": sample.audio_in.tolist(),
                "audio_in_lens": sample.audio_in_lens.tolist(),
                "audio_out": sample.audio_out.tolist(),
                "modality_flag": sample.modality_flag.tolist(),
                "supervision_mask": sample.supervision_mask.tolist(),
            }

    workspace = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}-", dir=out_dir.parent))
    try:
        try:
            preprocessed = datasets.Dataset.from_generator(
                generator,
                features=features,
                cache_dir=str(workspace / "cache"),
                fingerprint=uuid.uuid4().hex,
            )
        except ValueError as error:
            if counters["written"] == 0:
                raise ValueError("preprocessing produced no samples") from error
            raise
        if counters["written"] == 0:
            raise ValueError("preprocessing produced no samples")
        stats = PreprocessingStats(
            seen_samples=counters["seen"],
            written_samples=counters["written"],
            skipped_samples=counters["skipped"],
            max_sequence_length=counters["max_length"],
        )
        staging_dir = workspace / "dataset"
        preprocessed.save_to_disk(staging_dir)
        metadata = {
            **asdict(stats),
            "max_context_length": max_context_length,
            "codebooks": mapper.codebooks,
        }
        with (staging_dir / "preprocessing_stats.json").open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True)
            stream.write("\n")
        staging_dir.rename(out_dir)
    finally:
        shutil.rmtree(workspace)
    return stats
