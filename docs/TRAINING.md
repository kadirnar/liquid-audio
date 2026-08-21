# Quality-oriented LFM2-Audio training

This repository includes a reproducible fine-tuning path for ASR-only and joint omni training. The recipes are
starting points, not claimed benchmark results: keep a frozen validation set and choose checkpoints by measured WER
and TTS/voice metrics.

## What was improved

| Area | Implementation | Quality or efficiency effect |
|---|---|---|
| Parameter-efficient tuning | Native LoRA with merge-on-save; ASR and omni strategies | Adapts the 1.2B LFM backbone without full Adam states |
| Speech front end | Configurable Conformer and audio-adapter tuning with a lower learning rate | Adapts acoustics while reducing catastrophic forgetting |
| Regularization | Dynamic SpecAugment plus optional clean-preserving speed/gain/noise copies | Improves robustness to speakers, tempo, volume, and noise |
| Objective | Text/audio label smoothing, text/audio loss weights, and extra end-of-turn token weight | Reduces overconfidence and teaches reliable termination |
| Optimization | BF16/FP16 autocast, FP32 trainable weights, TF32, gradient accumulation, clipping, cosine decay, and no-decay norm/bias groups | Improves stability and makes 16 GB training practical |
| Memory | Gradient checkpointing, trainable-only optimizer state, dynamic batch padding, and a pure-ASR fast path | Avoids computing the TTS depthformer for ASR-only batches |
| Multi-task data | Explicit task-balanced sampling independent of corpus size | Prevents a large ASR or TTS corpus from dominating omni training |
| Reliability | Seeded loaders, resumable optimizer/RNG state, bounded checkpoints, best-checkpoint restore, early stopping, JSONL metrics | Makes runs reproducible and recoverable |
| Decoding | Repetition penalty, no-repeat n-grams, and an optional repeated-loop EOS guard | Prevents rare ASR loops from consuming the token cap |
| Evaluation | Same-sample base/fine-tuned corpus WER comparison with S/D/I counts, RTFx, cap hits, JSON, JSONL, and Markdown | Measures real generation rather than teacher-forced loss only |

## Important architecture fact: Mimi codebooks do not improve STT

ASR input audio is converted to 128-bin log-mel features and passed through the FastConformer and audio adapter.
It does **not** pass through Mimi. The checkpoint's eight Mimi codebooks belong to the generated-audio path (TTS and
speech output). Changing `codebooks=8` to `32` therefore cannot improve ASR WER.

The released checkpoint is architecturally eight-codebook: the audio embedding, depth projection, depthformer
embeddings, loss weights, and saved tensor shapes all agree on eight. A native 32-codebook model would require
resizing and initializing those modules and then training them on 32-codebook targets. It would be a new output codec
experiment, not an inference switch, and should be judged with TTS intelligibility and acoustic metrics rather than
STT WER.

## 1. Prepare leakage-free data

A JSONL manifest uses paths relative to the manifest:

```json
{"audio": "audio/0001.wav", "text": "The reference transcript."}
```

Create training data. `--augmentation-copies 1` retains the clean row and adds one independently augmented copy:

```bash
uv run python examples/preprocess_asr.py \
  --manifest manifests/asr-train.jsonl \
  --output data/asr/train \
  --revision c362a0625dfe45aa588dce5f0ada28a7e5707628 \
  --augmentation-copies 1 \
  --max-context-length 1024
```

Create validation data separately and never augment it:

```bash
uv run python examples/preprocess_asr.py \
  --manifest manifests/asr-validation.jsonl \
  --output data/asr/validation \
  --max-context-length 1024
```

Hugging Face datasets are also supported:

```bash
uv run python examples/preprocess_asr.py \
  --hf-dataset openslr/librispeech_asr \
  --hf-config clean \
  --split train.100 \
  --text-column text \
  --output data/librispeech/train-clean-100
```

Each output includes `preprocessing_stats.json`; inspect skipped long samples instead of silently losing them.

## 2. Establish the frozen baseline

Run exactly the same held-out rows through the base and later the fine-tuned checkpoint. Ten samples are useful only
as a smoke test; checkpoint selection needs a larger, speaker-disjoint validation set.

```bash
uv run python examples/evaluate_asr.py \
  --manifest manifests/asr-test.jsonl \
  --model LiquidAI/LFM2.5-Audio-1.5B --label base \
  --limit 10 \
  --output outputs/baseline-smoke
```

The report explicitly separates substitutions, deletions, insertions, RTFx, and maximum-token hits. Its lightweight
normalizer is for controlled before/after comparisons. Use the pinned Open ASR Leaderboard normalizer and datasets
when comparing to published model-card values.

## 3. ASR adaptation

The recommended first stage trains the Conformer/audio adapter and LoRA updates in the LFM backbone. TTS-only modules
remain frozen.

```bash
uv run accelerate launch examples/train.py \
  --config configs/asr_lora.json \
  --data data/asr/train \
  --val-data data/asr/validation \
  --context-length 1024
```

Resume safely after interruption:

```bash
uv run accelerate launch examples/train.py \
  --config configs/asr_lora.json \
  --data data/asr/train \
  --val-data data/asr/validation \
  --resume
```

For a final in-domain polish pass, edit `configs/asr_polish.json` so `model_id` points to the first stage's merged
`final` directory, then train with a lower learning rate and weaker augmentation.

## 4. Joint omni rehearsal

Do not concatenate corpora and let the largest one dictate the objective. Repeat `--data` and provide one probability
per task. A useful starting point is 50% ASR, 25% TTS, and 25% speech-to-speech/chat; tune this using per-task held-out
metrics.

```bash
uv run accelerate launch examples/train.py \
  --config configs/omni_lora.json \
  --data data/asr/train --data-weight 0.50 \
  --data data/tts/train --data-weight 0.25 \
  --data data/s2s/train --data-weight 0.25 \
  --samples-per-epoch 100000 \
  --val-data data/omni/validation \
  --context-length 2048
```

The omni recipe enables the generated-audio stack. Its validation set should contain every trained task. For serious
model selection, report at least:

- ASR: corpus WER on clean, noisy, accented, meeting, and long-form subsets, plus maximum-token-hit rate.
- TTS: SEED-TTS-style round-trip WER, speaker similarity, DNSMOS/UTMOS, and human preference/MOS.
- Speech-to-speech: semantic task accuracy, response latency/RTFx, speaker consistency, and interruption behavior.
- Retention: the original model-card suite, to catch catastrophic forgetting.

## 5. Compare the trained checkpoint

```bash
uv run python examples/evaluate_asr.py \
  --manifest manifests/asr-test.jsonl \
  --model LiquidAI/LFM2.5-Audio-1.5B --label base \
  --model outputs/asr-lora/final --label fine-tuned \
  --output outputs/asr-comparison
```

`summary.md` is an English Markdown table and `summary.json` is machine-readable. The repetition safeguard is held
constant for both arms. Also run with `--no-repetition-guard` to separate learned quality gains from decoding-only
gains.

Run the TTS intelligibility regression with the same texts and built-in voices in both arms:

```bash
uv run python examples/evaluate_tts_wer.py \
  --manifest manifests/tts-test.jsonl \
  --model LiquidAI/LFM2.5-Audio-1.5B --label base \
  --model outputs/omni-lora/final --label fine-tuned \
  --output outputs/tts-comparison
```

This evaluator uses the pinned **original `openai/whisper-large-v3`**, not a fine-tuned derivative. It reports macro
and corpus WER with substitutions/deletions/insertions. This is intentionally labeled SEED-TTS-style rather than the
official SEED-TTS score: LFM offers built-in voices instead of prompt-audio voice cloning, so the official speaker
similarity arm is not applicable.

## Data and training priorities

In expected impact order:

1. Correct transcripts, speaker-disjoint splits, and domain-matched real audio.
2. Balanced coverage of accents, microphones, noise, overlap, numbers, proper nouns, and long utterances.
3. Conservative ASR LoRA/adapter tuning selected by held-out WER.
4. Joint rehearsal with original TTS/chat examples to retain omni capabilities.
5. Confidence-filtered pseudo-labels or teacher distillation only after a clean supervised baseline exists.
6. Real room impulse responses and real noise mixing for target environments; synthetic white noise is only a basic fallback.

Do not enable every regularizer at maximum strength. Ablate one change at a time against the same evaluation manifest,
keep only improvements that repeat across seeds, and retain the base model as the immutable control.

## Primary references

- [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://arxiv.org/abs/1904.08779)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Seed-TTS: A Family of High-Quality Versatile Speech Generation Models](https://arxiv.org/abs/2406.02430)
