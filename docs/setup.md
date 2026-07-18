# Setup

## Prerequisites

- macOS on Apple Silicon (M-series)
- [uv](https://docs.astral.sh/uv/) (`brew install uv`)
- [ffmpeg](https://ffmpeg.org/) (`brew install ffmpeg`)
- [SoX](http://sox.sourceforge.net/) (`brew install sox`) — required by Qwen3-TTS speech tokenizer normalization
- Hugging Face account + token (for gated **Pyannote** models)

## Install

```bash
cd DubbingQwen
uv sync
cp .env.example .env
```

Edit `.env` and set `HF_TOKEN=hf_...`.

Accept the Pyannote model terms (required once per account):

- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

## Models

- **Whisper (CT2):** [`ivrit-ai/whisper-large-v3-turbo-ct2`](https://huggingface.co/ivrit-ai/whisper-large-v3-turbo-ct2) under `models/whisper-large-v3-turbo-ct2`.
- **TranslateGemma (optional CLI):** under `models/translategemma-4b-it` — `inference/translate.py` only. **`build_preview` uses mlx-lm** (default `mlx-community/Qwen2.5-7B-Instruct-4bit`); TranslateGemma often returns empty on this stack.
- **Qwen3-TTS Base (1.7B):** zero-shot clone under `models/Qwen3-TTS-12Hz-1.7B-Base`.

```bash
uv run hf download ivrit-ai/whisper-large-v3-turbo-ct2 \
  --local-dir models/whisper-large-v3-turbo-ct2

uv run hf download google/translategemma-4b-it \
  --local-dir models/translategemma-4b-it

uv run hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --local-dir models/Qwen3-TTS-12Hz-1.7B-Base
```

`uv sync` pins Python to `>=3.11,<3.14` (typically 3.12) for ML wheel compatibility.

If Pyannote / LatentSync hit missing MPS ops:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Verify

```bash
which sox ffmpeg
uv run python inference/transcribe.py --help
uv run python inference/translate.py --help
uv run python inference/extract_pipeline.py --help
uv run python inference/build_preview.py --help
```

Smoke-test ASR (needs a short Hebrew audio/video file):

```bash
uv run python inference/transcribe.py path/to/clip.wav --timestamps
```

Smoke-test translation:

```bash
uv run python inference/translate.py -s he -t en "שלום, מה שלומך?"
```

## Dependency source of truth

`pyproject.toml` is authoritative. Prefer `uv sync` / `uv add` over editing requirement lists by hand.
