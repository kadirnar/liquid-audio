# Liquid Audio Training Toolkit

[![Check](https://github.com/kadirnar/liquid-audio/actions/workflows/check.yml/badge.svg)](https://github.com/kadirnar/liquid-audio/actions/workflows/check.yml)

A practical toolkit for fine-tuning and evaluating
[LiquidAI/LFM2.5-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B).
The project focuses on improving ASR and speech-to-speech quality with reproducible training recipes.

## Included

- ASR and omni LoRA fine-tuning
- SpecAugment and waveform augmentation
- Balanced ASR/TTS task sampling
- Mixed precision, gradient checkpointing, and exact resume
- ASR WER and TTS WER evaluation
- Repetition-safe text decoding

## Setup

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), and a CUDA GPU for training.

```bash
git clone https://github.com/kadirnar/liquid-audio.git
cd liquid-audio
make setup
make check
```

## Prepare ASR data

Create a JSONL manifest with one sample per line:

```json
{"audio": "audio/sample.wav", "text": "The reference transcript."}
```

Preprocess the training and validation sets:

```bash
uv run python examples/preprocess_asr.py \
  --manifest manifests/asr-train.jsonl \
  --output data/asr/train \
  --augmentation-copies 1

uv run python examples/preprocess_asr.py \
  --manifest manifests/asr-validation.jsonl \
  --output data/asr/validation
```

Hugging Face datasets are also supported through `--hf-dataset`.

## Train

ASR LoRA:

```bash
make train-asr \
  TRAIN_DATA=data/asr/train \
  VAL_DATA=data/asr/validation
```

Balanced omni training:

```bash
uv run accelerate launch examples/train.py \
  --config configs/omni_lora.json \
  --data data/asr/train --data-weight 0.5 \
  --data data/tts/train --data-weight 0.5 \
  --val-data data/asr/validation
```

Resume the latest checkpoint by adding `--resume`.

## Evaluate

Compare the base model with a trained checkpoint using the same samples:

```bash
make evaluate-asr \
  TEST_MANIFEST=manifests/asr-test.jsonl \
  TRAINED_MODEL=outputs/asr-lora/final

make evaluate-tts \
  TTS_MANIFEST=manifests/tts-test.jsonl \
  TRAINED_MODEL=outputs/omni-lora/final
```

Results are written as JSON, JSONL, and English Markdown tables. TTS WER uses the pinned original
`openai/whisper-large-v3` checkpoint.

## Recipes

| Recipe | Purpose |
|---|---|
| [`configs/asr_lora.json`](configs/asr_lora.json) | Main ASR fine-tuning |
| [`configs/asr_polish.json`](configs/asr_polish.json) | Low-learning-rate ASR refinement |
| [`configs/omni_lora.json`](configs/omni_lora.json) | Joint ASR/TTS/omni fine-tuning |

The released model is native to 8 Mimi codebooks. Using 32 codebooks requires changing the output
architecture and retraining; it is not an inference switch and does not directly improve the ASR input path.

See [Training Guide](docs/TRAINING.md) for dataset formats, configuration details, checkpoint behavior,
evaluation metrics, and recommended training stages. See [GPU Smoke Test](docs/GPU_SMOKE_TEST.md) for the
verified hardware run.

## License

This project keeps the original [LFM Open License v1.0](LICENSE) and upstream third-party notices.
