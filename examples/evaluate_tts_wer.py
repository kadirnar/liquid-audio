from __future__ import annotations

import argparse
import gc
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Any

import soundfile
import torch
import torchaudio
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor
from liquid_audio.evaluation import corpus_wer
from liquid_audio.utils import get_model_dir

MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B"
MODEL_REVISION = "c362a0625dfe45aa588dce5f0ada28a7e5707628"
WHISPER_ID = "openai/whisper-large-v3"
WHISPER_REVISION = "06f233fe06e710322aca913c1bc4249a0d71fce1"
VOICES = (
    "Perform TTS. Use the US male voice.",
    "Perform TTS. Use the US female voice.",
    "Perform TTS. Use the UK male voice.",
    "Perform TTS. Use the UK female voice.",
)


def optional_revision(value: str) -> str | None:
    return None if value.casefold() in {"none", "null"} else value


def safe_label(label: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", label).strip("-") or "model"


def load_manifest(path: str | Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} must contain a JSON object")
            text = str(row.get("text") or "").strip()
            if not text:
                raise ValueError(f"{path}:{line_number} has no non-empty text")
            rows.append({**row, "text": text})
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError("TTS evaluation manifest contains no samples")
    return rows


def generate_audio(
    *,
    model_id: str,
    revision: str | None,
    label: str,
    rows: list[dict[str, Any]],
    processor: LFM2AudioProcessor,
    args: argparse.Namespace,
    dtype: torch.dtype,
    output_dir: Path,
) -> list[dict[str, Any]]:
    source: str | Path = Path(model_id) if Path(model_id).exists() else model_id
    resolved_path = get_model_dir(source, revision=None if isinstance(source, Path) else revision)
    resolved_revision = resolved_path.name if resolved_path.parent.name == "snapshots" else None
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = LFM2AudioModel.from_pretrained(resolved_path, device=args.device, dtype=dtype).eval()
    audio_dir = output_dir / "audio" / safe_label(label)
    audio_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for index, row in enumerate(tqdm(rows, desc=f"TTS {label}", unit="sample")):
        system_prompt = str(row.get("voice") or VOICES[index % len(VOICES)])
        chat = ChatState(processor, codebooks=model.codebooks, dtype=dtype)
        chat.new_turn("system")
        chat.add_text(system_prompt)
        chat.end_turn()
        chat.new_turn("user")
        chat.add_text(row["text"])
        chat.end_turn()
        chat.new_turn("assistant")

        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        audio_frames: list[torch.Tensor] = []
        generated_steps = 0
        for token in model.generate_sequential(
            **chat,
            max_new_tokens=args.max_new_tokens,
            audio_temperature=args.audio_temperature,
            audio_top_k=args.audio_top_k,
        ):
            generated_steps += 1
            if token.numel() > 1:
                audio_frames.append(token.detach())
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - started
        if audio_frames and torch.all(audio_frames[-1] == 2048):
            audio_frames.pop()
        if not audio_frames:
            raise RuntimeError(f"model {label} generated no audio for row {index}")

        codes = torch.stack(audio_frames, dim=1).unsqueeze(0)
        waveform = processor.decode(codes).float().cpu()[0]
        output_path = audio_dir / f"{index:05d}.wav"
        soundfile.write(output_path, waveform.numpy(), 24_000, subtype="PCM_16")
        duration_seconds = waveform.shape[-1] / 24_000
        records.append(
            {
                "index": index,
                "label": label,
                "model": model_id,
                "resolved_revision": resolved_revision,
                "requested_revision": revision,
                "reference": row["text"],
                "voice": system_prompt,
                "audio_path": str(output_path),
                "audio_seconds": duration_seconds,
                "generation_seconds": generation_seconds,
                "generation_rtfx": duration_seconds / generation_seconds if generation_seconds else None,
                "generated_steps": generated_steps,
                "hit_max_new_tokens": generated_steps >= args.max_new_tokens,
            }
        )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


def transcribe_generated_audio(
    records_by_model: list[list[dict[str, Any]]],
    *,
    args: argparse.Namespace,
    dtype: torch.dtype,
) -> None:
    evaluator_dtype = torch.float32 if args.device == "cpu" else dtype
    processor = WhisperProcessor.from_pretrained(WHISPER_ID, revision=WHISPER_REVISION)
    model = WhisperForConditionalGeneration.from_pretrained(
        WHISPER_ID,
        revision=WHISPER_REVISION,
        dtype=evaluator_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(args.device)
    model.eval()

    pending = [record for records in records_by_model for record in records]
    for record in tqdm(pending, desc="Whisper-large-v3 WER", unit="sample"):
        waveform, sample_rate = soundfile.read(record["audio_path"], dtype="float32", always_2d=True)
        waveform_tensor = torch.from_numpy(waveform.mean(axis=1))
        if sample_rate != 16_000:
            waveform_tensor = torchaudio.functional.resample(waveform_tensor, sample_rate, 16_000)
        inputs = processor(
            waveform_tensor.numpy(),
            sampling_rate=16_000,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(device=args.device, dtype=evaluator_dtype)
        attention_mask = inputs.attention_mask.to(args.device)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            predicted_ids = model.generate(
                input_features,
                attention_mask=attention_mask,
                language="english",
                task="transcribe",
            )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        record["evaluator_seconds"] = time.perf_counter() - started
        record["transcript"] = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        record["over_whisper_30_second_window"] = record["audio_seconds"] > 30.0

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    references = [record["reference"] for record in records]
    predictions = [record["transcript"] for record in records]
    overall = corpus_wer(references, predictions)
    utterance_wers = [
        corpus_wer([reference], [prediction]).wer_percent
        for reference, prediction in zip(references, predictions, strict=True)
    ]
    audio_seconds = sum(record["audio_seconds"] for record in records)
    generation_seconds = sum(record["generation_seconds"] for record in records)
    evaluator_seconds = sum(record["evaluator_seconds"] for record in records)
    return {
        "label": records[0]["label"],
        "model": records[0]["model"],
        "resolved_revision": records[0]["resolved_revision"],
        "requested_revision": records[0]["requested_revision"],
        **asdict(overall),
        "word_errors": overall.word_errors,
        "macro_wer_percent": fmean(utterance_wers),
        "audio_seconds": audio_seconds,
        "generation_rtfx": audio_seconds / generation_seconds,
        "evaluator_rtfx": audio_seconds / evaluator_seconds,
        "hit_max_new_tokens": sum(record["hit_max_new_tokens"] for record in records),
        "over_whisper_30_second_window": sum(record["over_whisper_30_second_window"] for record in records),
    }


def write_report(
    output_dir: Path,
    summaries: list[dict[str, Any]],
    records_by_model: list[list[dict[str, Any]]],
    args: argparse.Namespace,
) -> None:
    for summary, records in zip(summaries, records_by_model, strict=True):
        path = output_dir / f"predictions-{safe_label(summary['label'])}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    result = {
        "benchmark": "SEED-TTS-style round-trip intelligibility WER",
        "comparability": (
            "Not the official SEED-TTS zero-shot protocol: LFM uses built-in voices rather than prompt-audio cloning, "
            "so speaker similarity is not applicable."
        ),
        "manifest": args.manifest,
        "device": args.device,
        "dtype": args.dtype,
        "seed": args.seed,
        "generation": {
            "audio_temperature": args.audio_temperature,
            "audio_top_k": args.audio_top_k,
            "max_new_tokens": args.max_new_tokens,
        },
        "evaluator": {"model": WHISPER_ID, "revision": WHISPER_REVISION, "original_whisper": True},
        "normalization": "NFKC, lowercase, punctuation removal, whitespace collapse",
        "speaker_similarity": "not_applicable_without_prompt_audio_voice_cloning",
        "results": summaries,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")

    lines = [
        "# TTS checkpoint intelligibility comparison",
        "",
        "| Model | Samples | Macro WER ↓ | Corpus WER ↓ | Word errors (S/D/I) | Generation RTFx ↑ | Cap hits |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {item['label']} | {item['samples']} | {item['macro_wer_percent']:.3f} | "
        f"{item['wer_percent']:.3f} | {item['word_errors']} "
        f"({item['substitutions']}/{item['deletions']}/{item['insertions']}) | "
        f"{item['generation_rtfx']:.2f} | {item['hit_max_new_tokens']} |"
        for item in summaries
    )
    lines.extend(
        (
            "",
            f"Intelligibility evaluator: pinned original `{WHISPER_ID}` at `{WHISPER_REVISION}`.",
            "This is a SEED-TTS-style round-trip WER adaptation, not the official zero-shot protocol: LFM uses "
            "built-in voices and does not receive SEED-TTS prompt audio, so speaker similarity is not applicable.",
            "All comparison arms use identical prompts, voices, seeds, generation settings, evaluator, and normalization.",
        )
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare TTS checkpoints using pinned Whisper-large-v3 WER.")
    parser.add_argument("--manifest", required=True, help='JSONL rows with "text" and optional "voice".')
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
    parser.add_argument("--output", default="outputs/tts-wer-evaluation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--audio-temperature", type=float, default=0.8)
    parser.add_argument("--audio-top-k", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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
    if arguments.audio_temperature <= 0 or arguments.audio_top_k <= 0:
        raise ValueError("audio temperature and top-k must be positive")

    torch_dtype = getattr(torch, arguments.dtype)
    if arguments.device == "cpu" and torch_dtype == torch.float16:
        raise ValueError("float16 inference is not supported on CPU")
    destination = Path(arguments.output)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_rows = load_manifest(arguments.manifest, arguments.limit)
    processor_source: str | Path = (
        Path(arguments.processor_id) if Path(arguments.processor_id).exists() else arguments.processor_id
    )
    if isinstance(processor_source, Path):
        arguments.processor_revision = None
    processor_path = get_model_dir(
        processor_source,
        revision=None if isinstance(processor_source, Path) else arguments.processor_revision,
    )
    lfm_processor = LFM2AudioProcessor.from_pretrained(processor_path, device=arguments.device).eval()
    generated = [
        generate_audio(
            model_id=model_id,
            revision=revision,
            label=label,
            rows=manifest_rows,
            processor=lfm_processor,
            args=arguments,
            dtype=torch_dtype,
            output_dir=destination,
        )
        for model_id, label, revision in zip(models, labels, revisions, strict=True)
    ]
    del lfm_processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    transcribe_generated_audio(generated, args=arguments, dtype=torch_dtype)
    model_summaries = [summarize(records) for records in generated]
    write_report(destination, model_summaries, generated, arguments)
    print(f"Wrote {destination / 'summary.md'}")
