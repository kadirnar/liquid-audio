# GPU integration smoke test

The quality-training path was exercised against the real
`LiquidAI/LFM2.5-Audio-1.5B` checkpoint on an NVIDIA GeForce RTX 5070 Ti (16 GB), not only mocked modules.

| Check | Result |
|---|---:|
| Precision | BF16 |
| Input | One-second 16 kHz synthetic ASR waveform |
| Strategy | `asr_lora`, rank 16, Conformer + adapter trainable |
| Matched LoRA layers | 82 |
| Trainable parameters | 124,662,528 / 1,463,303,296 (8.519%) |
| Trainable parameters promoted to FP32 | 114,864,896 |
| Forward loss | 6.842421 |
| Audio-output tokens in pure ASR path | 0 |
| Backward + fused AdamW update | Passed |
| Peak allocated VRAM | 5.789 GiB |

This is an integration and memory smoke test, not a quality benchmark. Real VRAM usage grows with audio duration,
context length, local batch size, and the selected trainable modules. Quality claims must come from the held-out WER
and TTS/omni evaluations described in [TRAINING.md](TRAINING.md).

The end-to-end WER CLI was also run on the bundled 18.356-second ASR sample: all 43 normalized reference words were
correct (0 substitutions, 0 deletions, 0 insertions), no 512-token cap was hit, and observed RTFx was 24–26. This
single sample validates the evaluation pipeline only; it is not a dataset-level quality claim.

The TTS regression CLI generated 3.36 seconds of speech for the nine-word bundled prompt and the pinned original
`openai/whisper-large-v3` recovered all nine words (0 S/D/I, no generation cap hit). Observed generation RTFx was
1.6–1.7. This
likewise validates the round-trip pipeline and is not an official SEED-TTS or dataset-level result.
