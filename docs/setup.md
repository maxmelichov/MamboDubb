# Setup

## Prerequisites

- macOS on Apple Silicon (M-series)
- [uv](https://docs.astral.sh/uv/) (`brew install uv`)
- [ffmpeg](https://ffmpeg.org/) (`brew install ffmpeg`)
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
- **TranslateGemma:** under `models/translategemma-4b-it`.

```bash
uv run hf download ivrit-ai/whisper-large-v3-turbo-ct2 \
  --local-dir models/whisper-large-v3-turbo-ct2

uv run hf download google/translategemma-4b-it \
  --local-dir models/translategemma-4b-it
```

`uv sync` pins Python to `>=3.11,<3.14` (typically 3.12) for ML wheel compatibility.

## Verify

```bash
uv run python inference/transcribe.py --help
uv run python inference/translate.py --help
uv run python inference/extract_pipeline.py --help
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

`pyproject.toml` is authoritative. The files under `requirements/` remain as thin reference lists for component boundaries; prefer `uv sync` / `uv add` over editing them by hand for installs.
