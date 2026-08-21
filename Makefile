PYTHON ?= uv run python
ACCELERATE ?= uv run accelerate
TRAIN_DATA ?= data/asr/train
VAL_DATA ?= data/asr/validation
TEST_MANIFEST ?= manifests/asr-test.jsonl
TTS_MANIFEST ?= manifests/tts-test.jsonl
TRAINED_MODEL ?= outputs/asr-lora/final

.PHONY: setup check lint test train-asr train-omni evaluate-asr evaluate-tts

setup:
	uv sync --dev --frozen

lint:
	uv run ruff check .
	uv run ruff format --check
	uv run mypy src/liquid_audio examples tests

test:
	uv run pytest

check: lint test

train-asr:
	$(ACCELERATE) launch examples/train.py --config configs/asr_lora.json --data $(TRAIN_DATA) --val-data $(VAL_DATA)

train-omni:
	$(ACCELERATE) launch examples/train.py --config configs/omni_lora.json --data $(TRAIN_DATA) --val-data $(VAL_DATA)

evaluate-asr:
	$(PYTHON) examples/evaluate_asr.py --manifest $(TEST_MANIFEST) --model LiquidAI/LFM2.5-Audio-1.5B --label base --model $(TRAINED_MODEL) --label fine-tuned

evaluate-tts:
	$(PYTHON) examples/evaluate_tts_wer.py --manifest $(TTS_MANIFEST) --model LiquidAI/LFM2.5-Audio-1.5B --label base --model $(TRAINED_MODEL) --label fine-tuned
