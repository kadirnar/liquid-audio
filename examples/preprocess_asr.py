from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from datasets import Audio, load_dataset

from liquid_audio import LFM2AudioProcessor
from liquid_audio.data.asr import ASRChatIterator, iter_jsonl
from liquid_audio.data.mapper import LFM2AudioChatMapper
from liquid_audio.data.preprocess import preprocess_dataset
from liquid_audio.training import WaveformAugmentationConfig, WaveformAugmenter

MODEL_REVISION = "c362a0625dfe45aa588dce5f0ada28a7e5707628"


def optional_revision(value: str) -> str | None:
    return None if value.casefold() in {"none", "null"} else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess an ASR manifest or Hugging Face dataset for LFM2-Audio.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", help='JSONL with {"audio": "path.wav", "text": "transcript"} rows.')
    source.add_argument("--hf-dataset", help="Hugging Face dataset id.")
    parser.add_argument("--hf-config")
    parser.add_argument("--split", default="train")
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", default="LiquidAI/LFM2.5-Audio-1.5B")
    parser.add_argument("--revision", type=optional_revision, default=MODEL_REVISION)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-context-length", type=int, default=1024)
    parser.add_argument("--augmentation-copies", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.augmentation_copies < 0:
        raise ValueError("--augmentation-copies must be non-negative")
    torch.manual_seed(args.seed)

    base_dir: Path | None
    if args.manifest:
        manifest_path = Path(args.manifest)
        rows: Iterable[dict[str, Any]] = iter_jsonl(manifest_path)
        base_dir = manifest_path.parent
    else:
        dataset = load_dataset(args.hf_dataset, args.hf_config, split=args.split)
        rows = dataset.cast_column(args.audio_column, Audio(decode=False))
        base_dir = None

    augmenter = WaveformAugmenter(WaveformAugmentationConfig()) if args.augmentation_copies else None
    data = ASRChatIterator(
        rows,
        audio_column=args.audio_column,
        text_column=args.text_column,
        base_dir=base_dir,
        augmentation_copies=args.augmentation_copies,
        augmenter=augmenter,
    )
    revision = None if Path(args.model_id).exists() else args.revision
    processor = LFM2AudioProcessor.from_pretrained(args.model_id, revision=revision, device=args.device).eval()
    mapper = LFM2AudioChatMapper(processor)
    stats = preprocess_dataset(
        data=data,
        output_path=args.output,
        mapper=mapper,
        max_context_length=args.max_context_length,
    )
    print(f"Preprocessing complete: {stats}")
