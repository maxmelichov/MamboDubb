# Setup

## Prerequisites

- macOS on Apple Silicon (M-series)
- [uv](https://docs.astral.sh/uv/) (`brew install uv`)
- [ffmpeg](https://ffmpeg.org/) (`brew install ffmpeg`)
- [SoX](http://sox.sourceforge.net/) (`brew install sox`) Qwen3-TTS text normalization
- [Node](https://nodejs.org/) + [pnpm](https://pnpm.io/) (`brew install node pnpm`) to build the editor UI

## Install

```bash
cd MamboDubb
uv sync
```

**No Hugging Face account is needed.** Every model this pipeline opens is a public
repo. Diarization used to be the exception the upstream
[`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)
is gated so an install without a token dubbed every character in one voice. The
weights are CC-BY-4.0, so the identical 31 MB is checked into this repository at
`third_party/pyannote-speaker-diarization-community-1` (`dubbing/segments.py
DIARIZATION_DIR`) and ships inside the app. Nothing is downloaded and nothing is
signed in to; the licence text and the credit travel with the files, in that
directory's `LICENSE` and `NOTICE.md`.

A token is still *usable*, and only for one thing: set `DUB_DIARIZATION_HUB` to the
gated repo id to fetch the canonical repo from upstream instead, with `HF_TOKEN=hf_...`
in `.env` (or the field on the Setup screen, which writes it for you).

## Models

All local, under `models/`:

```bash
# Translation Gemma 4 12B, 6-bit, run through MLX (~9.7 GB). Below 6 bits this
# model does not hold Hebrew (the 4-bit reads fluently and gets it wrong, which
# is worse); see the scoring table in dubbing/translate.py before trading down.
uv run hf download mlx-community/gemma-4-12B-it-6bit \
  --local-dir models/gemma-4-12B-it-6bit

# Speech synthesis, zero-shot voice cloning
uv run hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --local-dir models/Qwen3-TTS-12Hz-1.7B-Base

# English ASR used to verify each generated clip
uv run hf download Systran/faster-whisper-base.en \
  --local-dir models/faster-whisper-base.en

# Multilingual ASR used to verify clips in any non-English target
uv run hf download Systran/faster-whisper-base \
  --local-dir models/faster-whisper-base

# Hebrew-source ASR only needed for videos that have no captions
uv run hf download ivrit-ai/whisper-large-v3-turbo-ct2 \
  --local-dir models/whisper-large-v3-turbo-ct2

# Every other source language reads with the vanilla multilingual turbo model
uv run hf download deepdml/faster-whisper-large-v3-turbo-ct2 \
  --local-dir models/faster-whisper-large-v3-turbo-ct2
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
Without the extra (and always on Windows, where vLLM publishes no wheels)
`translator/worker.py` remains the backend, unchanged. Two knobs, both env vars
read by the worker: `TRANSLATOR_VLLM_GPU_UTIL` (default `0.90`) and
`TRANSLATOR_VLLM_MAX_LEN` (default `8192`).

Set `DUBBING_TRANSLATOR_BACKEND` to `vllm`, `transformers` or `mlx` to force one
(default `auto`); a forced backend that cannot start fails loudly rather than
falling back, which is the point of forcing it.

### Low-VRAM mode

The bf16 weights above are about 24 GB, which is more than a 24 GB card can hold
once the KV cache is counted. Low-VRAM mode loads the same checkpoint as 4-bit
NF4 through bitsandbytes (bfloat16 compute), roughly 7 GB of weights, so a 12 GB
card runs and a 16 GB one is comfortable. On Apple Silicon the same switch loads
the mxfp4 MLX build (6.4 GB) instead of the 6-bit one (9.7 GB).

```bash
uv run python -m dubbing input.mp4 --low-vram      # or --no-low-vram
DUBBING_LOW_VRAM=1 uv run python -m dubbing input.mp4
```

The desktop app has the same switch on its Setup screen, which writes
`DUBBING_LOW_VRAM` to the workspace `.env`. Nothing has to be set at all: the
mode turns itself on under 28 GiB of VRAM or under 20 GiB of unified memory and
says so in the log. An explicit choice always wins over that detection, in both
directions.

It costs translation quality. Measured on Apple Silicon, thirty-one real Hebrew
lines: mxfp4 differed from the 6-bit output on 24 of them, six of those being
real losses (an idiom read literally, a name hallucinated out of the run's
context, a name spelled two ways in one run) against one improvement, at 1.6x
the speed. The CUDA 4-bit path has not been run on real hardware, so its ~7 GB
and its quality are calculated rather than measured.

bitsandbytes is an optional extra of the translator venv. The first low-VRAM
launch adds it (`uv run --extra lowvram`) on a machine with a CUDA device; to do
it by hand:

```bash
uv sync --project translator --extra lowvram
```

Where it cannot be installed, or where there is no CUDA device to quantise onto,
the worker says so on stderr and loads bfloat16, which is what that machine did
before the mode existed. It is never recorded in a run: which weights fit is a
property of the machine, and a project copied to a bigger card must not still be
asking for the small ones.

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
