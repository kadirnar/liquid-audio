from __future__ import annotations

import argparse
import gc
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from datasets import Audio, load_dataset
from tqdm import tqdm

from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor
from liquid_audio.data.asr import decode_audio_bytes, iter_jsonl, load_audio_bytes
from liquid_audio.decoding import TextDecodingConfig
from liquid_audio.evaluation import corpus_wer
from liquid_audio.utils import get_model_dir

MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B"
MODEL_REVISION = "c362a0625dfe45aa588dce5f0ada28a7e5707628"


def optional_revision(value: str) -> str | None:
    return None if value.casefold() in {"none", "null"} else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base and fine-tuned checkpoints with corpus WER.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", help='JSONL with {"audio": "path.wav", "text": "transcript"} rows.')
    source.add_argument("--hf-dataset", help="Hugging Face dataset id.")
    parser.add_argument("--hf-config")
    parser.add_argument("--split", default="test")
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--model", action="append", dest="models", help="Model id/path; repeat for comparisons.")
    parser.add_argument(
        "--model-revision",
        action="append",
        dest="model_revisions",
        type=optional_revision,
        help='One revision per --model; use "none" for local/unpinned models.',
    )
    parser.add_argument("--label", action="append", dest="labels", help="Label for each --model.")
    parser.add_argument("--processor-id", default=MODEL_ID)
    parser.add_argument("--processor-revision", type=optional_revision, default=MODEL_REVISION)
    parser.add_argument("--output", default="outputs/asr-evaluation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-repetition-guard", action="store_true")
    return parser.parse_args()


def safe_label(label: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", label).strip("-") or "model"


def source_rows(args: argparse.Namespace):
    if args.manifest:
        yield from iter_jsonl(args.manifest)
        return
    dataset = load_dataset(args.hf_dataset, args.hf_config, split=args.split)
    yield from dataset.cast_column(args.audio_column, Audio(decode=False))


def evaluate_model(
    *,
    model_id: str,
    revision: str | None,
    label: str,
    args: argparse.Namespace,
    processor: LFM2AudioProcessor,
    dtype: torch.dtype,
    output_dir: Path,
) -> dict[str, Any]:
    model_source: str | Path = Path(model_id) if Path(model_id).exists() else model_id
    resolved_model_path = get_model_dir(model_source, revision=None if isinstance(model_source, Path) else revision)
    resolved_revision = resolved_model_path.name if resolved_model_path.parent.name == "snapshots" else None
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = LFM2AudioModel.from_pretrained(resolved_model_path, device=args.device, dtype=dtype).eval()
    decoding = (
        TextDecodingConfig()
        if args.no_repetition_guard
        else TextDecodingConfig(repetition_penalty=1.05, loop_ngram_size=4, max_consecutive_loop_repeats=3)
    )
    references: list[str] = []
    predictions: list[str] = []
    audio_seconds = 0.0
    inference_seconds = 0.0
    capped = 0
    records: list[dict[str, Any]] = []

    progress = tqdm(source_rows(args), total=args.limit, desc=label, unit="sample")
    for index, row in enumerate(progress):
        if args.limit is not None and index >= args.limit:
            break
        audio_bytes = load_audio_bytes(
            row[args.audio_column],
            base_dir=Path(args.manifest).parent if args.manifest else None,
        )
        waveform, sampling_rate = decode_audio_bytes(audio_bytes)
        raw_reference = row.get(args.text_column)
        reference = "" if raw_reference is None else str(raw_reference).strip()
        if not reference:
            print(f"WARNING: skipping row {index} with an empty reference")
            continue

        chat = ChatState(processor, codebooks=model.codebooks, dtype=dtype)
        chat.new_turn("system")
        chat.add_text("Perform ASR.")
        chat.end_turn()
        chat.new_turn("user")
        chat.add_audio(waveform, sampling_rate)
        chat.end_turn()
        chat.new_turn("assistant")

        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        text_tokens = [
            token.detach()
            for token in model.generate_sequential(
                **chat,
                max_new_tokens=args.max_new_tokens,
                text_decoding=decoding,
            )
            if token.numel() == 1
        ]
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        wall_seconds = time.perf_counter() - started
        prediction = processor.text.decode(torch.cat(text_tokens), skip_special_tokens=True).strip() if text_tokens else ""
        duration_seconds = waveform.shape[-1] / sampling_rate
        hit_cap = len(text_tokens) >= args.max_new_tokens
        capped += int(hit_cap)
        audio_seconds += duration_seconds
        inference_seconds += wall_seconds
        references.append(reference)
        predictions.append(prediction)
        records.append(
            {
                "index": index,
                "reference": reference,
                "prediction": prediction,
                "audio_seconds": duration_seconds,
                "inference_seconds": wall_seconds,
                "hit_max_new_tokens": hit_cap,
            }
        )

    wer = corpus_wer(references, predictions)
    prediction_path = output_dir / f"predictions-{safe_label(label)}.jsonl"
    with prediction_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    result = {
        "label": label,
        "model": model_id,
        "resolved_revision": resolved_revision,
        "requested_revision": revision,
        **asdict(wer),
        "word_errors": wer.word_errors,
        "audio_seconds": audio_seconds,
        "inference_seconds": inference_seconds,
        "rtfx": audio_seconds / inference_seconds if inference_seconds else None,
        "hit_max_new_tokens": capped,
    }
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def write_report(output_dir: Path, results: list[dict[str, Any]], args: argparse.Namespace) -> None:
    summary = {
        "metric": "corpus WER with NFKC/lowercase/punctuation normalization",
        "device": args.device,
        "dtype": args.dtype,
        "processor_id": args.processor_id,
        "processor_revision": args.processor_revision,
        "seed": args.seed,
        "source": {
            "manifest": args.manifest,
            "hf_dataset": args.hf_dataset,
            "hf_config": args.hf_config,
            "split": args.split,
            "limit": args.limit,
        },
        "generation": {
            "system_prompt": "Perform ASR.",
            "max_new_tokens": args.max_new_tokens,
            "repetition_guard": not args.no_repetition_guard,
        },
        "results": results,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")

    lines = [
        "# ASR checkpoint comparison",
        "",
        "| Model | Samples | WER ↓ | Word errors (S/D/I) | RTFx ↑ | Max-token hits |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        (
            f"| {result['label']} | {result['samples']} | {result['wer_percent']:.3f} | "
            f"{result['word_errors']} ({result['substitutions']}/{result['deletions']}/{result['insertions']}) | "
            f"{result['rtfx']:.2f} | {result['hit_max_new_tokens']} |"
        )
        for result in results
    )
    lines.extend(
        (
            "",
            "All checkpoints use the same samples, processor, prompt, decoding settings, and WER normalization.",
            "This local normalization is intended for controlled before/after comparisons; use the pinned Open ASR "
            "Leaderboard normalizer when comparing against published model-card numbers.",
        )
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    arguments = parse_args()
    models = arguments.models or [MODEL_ID]
    labels = arguments.labels or models
    revisions = arguments.model_revisions or [MODEL_REVISION if model == MODEL_ID else None for model in models]
    if len(labels) != len(models):
        raise ValueError("provide exactly one --label for each --model")
    if len(revisions) != len(models):
        raise ValueError("provide exactly one --model-revision for each --model")
    if len({safe_label(label) for label in labels}) != len(labels):
        raise ValueError("model labels must map to unique output filenames")
    if arguments.limit is not None and arguments.limit <= 0:
        raise ValueError("--limit must be positive")

    torch_dtype = getattr(torch, arguments.dtype)
    if arguments.device == "cpu" and torch_dtype == torch.float16:
        raise ValueError("float16 inference is not supported on CPU")
    destination = Path(arguments.output)
    destination.mkdir(parents=True, exist_ok=True)
    processor_source: str | Path = (
        Path(arguments.processor_id) if Path(arguments.processor_id).exists() else arguments.processor_id
    )
    if isinstance(processor_source, Path):
        arguments.processor_revision = None
    processor_path = get_model_dir(
        processor_source,
        revision=None if isinstance(processor_source, Path) else arguments.processor_revision,
    )
    shared_processor = LFM2AudioProcessor.from_pretrained(processor_path, device=arguments.device).eval()
    benchmark_results = [
        evaluate_model(
            model_id=model_id,
            revision=revision,
            label=label,
            args=arguments,
            processor=shared_processor,
            dtype=torch_dtype,
            output_dir=destination,
        )
        for model_id, label, revision in zip(models, labels, revisions, strict=True)
    ]
    write_report(destination, benchmark_results, arguments)
    print(f"Wrote {destination / 'summary.md'}")
