# Setup

## Prerequisites

- macOS on Apple Silicon (M-series)
- [uv](https://docs.astral.sh/uv/) (`brew install uv`)
- [ffmpeg](https://ffmpeg.org/) (`brew install ffmpeg`)
- [SoX](http://sox.sourceforge.net/) (`brew install sox`) Qwen3-TTS text normalization
- Hugging Face account + token (for the gated **Pyannote** diarization models)

## Install

```bash
cd DubbingQwen
uv sync
cp .env.example .env      # then set HF_TOKEN=hf_...
```

Accept the Pyannote model terms once per account:

- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

Without a token the pipeline still runs diarization falls back to a single speaker
and the report says so.

## Models

All local, under `models/`:

```bash
# Translation Gemma 4 12B, quantisation-aware 4-bit, run through MLX (~11 GB).
# For a tighter memory budget use gemma-4-12B-it-4bit (~6.3 GB) instead and point
# translate.MODEL_PATH at it; see the trade-off in dubbing/translate.py.
uv run hf download mlx-community/gemma-4-12B-it-qat-4bit \
  --local-dir models/gemma-4-12B-it-qat-4bit

# Speech synthesis, zero-shot voice cloning
uv run hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --local-dir models/Qwen3-TTS-12Hz-1.7B-Base

# English ASR used to verify each generated clip
uv run hf download Systran/faster-whisper-base.en \
  --local-dir models/faster-whisper-base.en

# Source-language ASR only needed for videos that have no captions
uv run hf download ivrit-ai/whisper-large-v3-turbo-ct2 \
  --local-dir models/whisper-large-v3-turbo-ct2
```

Demucs (`htdemucs_ft`) and Pyannote download themselves on first use.

### Dubbing *into* Hebrew

Qwen3-TTS does not speak Hebrew. One extra download adds it a LoRA over the
same 1.7B Base checkpoint, so there is no second synthesiser to load:

```bash
uv run hf download notmax123/QwenTTS-he-1.7B --local-dir models/QwenTTS-he-1.7B
```

The grapheme→IPA model it reads from (`renikud-plus`, installed by `uv sync`)
fetches its own weights on first use; `hf download notmax123/RenikudPlus
model.onnx --local-dir models/RenikudPlus` puts them under `models/` instead.

With both present, `--tgt he` works, including `--src he --tgt he` a
same-language run re-voices every line in the cloned voice and never loads the
translator. Without them the run is refused up front, naming the command.

## Linux (CUDA)

Everything above applies, except translation: `mlx-lm` is macOS-only, and the main
venv's `transformers==4.57.3` pin (qwen-tts) predates Gemma 4. On Linux,
`translate.load()` instead spawns `translator/worker.py` in its own isolated uv
venv (`translator/pyproject.toml`: torch + transformers >= 5) and talks to it over a
JSON-lines pipe. Nothing to install by hand the first translate run builds the
venv via `uv run --project translator` automatically. It expects the bf16
transformers-format weights at:

```bash
uv run hf download unsloth/gemma-4-12b-it \
  --local-dir models/gemma-4-12b-it-cuda
```

### vLLM (optional, faster)

The same weights can be served by vLLM instead, which is markedly faster per line
and batches continuously. It is an optional extra of the *same* translator venv, so
one command adds it and nothing else changes:

```bash
uv sync --project translator --extra vllm
```

`translate.load()` then picks `translator/worker_vllm.py` automatically on Linux
with a CUDA device present; it speaks the same JSON-lines protocol, uses the same
prompts and the same post-processing, and greedy-decodes exactly as before.
Without the extra — and always on Windows, where vLLM publishes no wheels —
`translator/worker.py` remains the backend, unchanged. Two knobs, both env vars
read by the worker: `TRANSLATOR_VLLM_GPU_UTIL` (default `0.90`) and
`TRANSLATOR_VLLM_MAX_LEN` (default `8192`).

Set `DUBBING_TRANSLATOR_BACKEND` to `vllm`, `transformers` or `mlx` to force one
(default `auto`); a forced backend that cannot start fails loudly rather than
falling back, which is the point of forcing it.

## Verify

```bash
uv run python -m dubbing --help
uv run python -m pytest tests/test_dubbing.py -q
```

Then dub the first few minutes of something:

```bash
uv run python -m dubbing "https://www.youtube.com/watch?v=VIDEO_ID" --duration 300
```

Read `report.json` in the run directory afterwards it lists anything that needs a
human ear.

## Notes

- `pyproject.toml` is the dependency source of truth; use `uv sync` / `uv add`.
- `mlx-lm` is pinned to a git revision: Gemma 4's `gemma4_unified` architecture is
  only supported after a June 2026 fix that no release carries yet. It asks for
  transformers 5, which Qwen3-TTS cannot use, so `[tool.uv] override-dependencies`
  holds transformers at 4.57.3 mlx-lm only needs it for the tokenizer.
- If Pyannote hits missing MPS ops: `export PYTORCH_ENABLE_MPS_FALLBACK=1`
  (the pipeline sets this itself).
