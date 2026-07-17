# AGENTS.md

Guidance for AI agents and humans working in this repository.

## What this project is

**DubbingQwen** is a local Apple Silicon pipeline that turns Hebrew documentary / news video into English dubs:

1. Separate vocals from background music (Demucs)
2. Diarize + transcribe Hebrew (Pyannote + ivrit-ai Whisper)
3. Duration-aware translation (TranslateGemma / local LLM)
4. Zero-shot TTS (Qwen3-TTS 1.7B-Base clone from vocal refs; F5 optional)
5. Active-speaker-gated lip-sync (TalkNet/SyncNet → LatentSync)
6. Remix background + dubbed vocals with ducking (FFmpeg)

Build **one phase at a time**. Do not wire the full end-to-end pipeline until Phase 1–2 JSON output is solid.

## Tooling (required)

- **Package manager:** [`uv`](https://docs.astral.sh/uv/) only. Do not introduce Poetry/pipenv/conda.
- **Python:** `>=3.11,<3.14` (pin via `uv sync`; prefer 3.12 on Mac).
- **Install:** `uv sync`
- **Run:** `uv run …` (never assume a manually activated venv is present).
- **Secrets:** `.env` holds `HF_TOKEN`. Never commit `.env`. Use `.env.example` as the template.

## Layout

| Path | Role |
|------|------|
| `inference/` | First-party CLIs (`transcribe`, `translate`, `extract_pipeline`) |
| `models/` | Local HF checkpoints (gitignored) |
| `docs/` | Architecture and phase docs |
| `third_party/LatentSync` | Upstream lip-sync (Plan A; MPS may need fallbacks) |
| `third_party/Qwen3-TTS` | Upstream TTS |
| `outputs/` | Run artifacts (gitignored) |
| `pyproject.toml` | Source of truth for dependencies |

Legacy wrappers `inference_whisper_ivrit.py` / `inference_translategemma.py` only re-export the `inference/` CLIs.

## Conventions

- Prefer small, isolated scripts with clear CLI args over a mega-orchestrator.
- Write run artifacts under `outputs/<run_id>/`.
- JSON segment schema for Phase 1–2:

```json
{
 "source": "...",
 "vocals": "...",
 "background": "...",
 "segments": [
 {
   "speaker_id": "SPEAKER_00",
   "start": 0.0,
   "end": 1.2,
   "duration": 1.2,
   "language": "he",
   "keep_original": false,
   "text": "...",
   "phrases": [{"text": "...", "start": 0.0, "end": 1.2, "pause_after": 0.0}]
 }
 ]
}
```

Non-Hebrew turns (`language` ≠ `he`, `keep_original: true`) keep the original vocals (EN/AR/…). Only Hebrew is translated + TTS'd.

- On Apple Silicon: Whisper via **faster-whisper** + [`ivrit-ai/whisper-large-v3-turbo-ct2`](https://huggingface.co/ivrit-ai/whisper-large-v3-turbo-ct2); TranslateGemma / Pyannote / F5 via PyTorch MPS. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` if LatentSync or Pyannote hits missing ops.
- Do not download huge models into the repo root; always use `models/<name>/` via `uv run hf download …`.
- Do not commit `*.wav` / `*.mp4` / `models/` / `.env`.

## Current priority (Phase 1–2)

Make `uv run python inference/extract_pipeline.py <video>` reliable:

1. `ffmpeg` → `source.wav`
2. Demucs → `vocals.wav` + `background.wav`
3. Pyannote → speaker turns
4. faster-whisper (word timestamps) ∩ turns → `segments.json`

Next phases (translation, TTS, ASD, LatentSync) should consume that JSON — do not reinvent timestamps.

## Commands agents should know

```bash
uv sync
cp .env.example .env   # then set HF_TOKEN

uv run python inference/transcribe.py path/to/audio.wav --timestamps
uv run python inference/translate.py -s he -t en "שלום"
uv run python inference/extract_pipeline.py path/to/video.mp4
uv run python inference/build_preview.py outputs/<run> --skip-translate
# Legacy F5:
uv run python inference/build_preview.py outputs/<run> --skip-translate --tts-engine f5

uv run hf download google/translategemma-4b-it --local-dir models/translategemma-4b-it
uv run hf download ivrit-ai/whisper-large-v3-turbo-ct2 --local-dir models/whisper-large-v3-turbo-ct2
uv run hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir models/Qwen3-TTS-12Hz-1.7B-Base
```

## Hard constraints

- Never force-push, amend shared history, or skip hooks unless the user asks.
- Never write exploits or attack tooling.
- Never print or commit secrets from `.env`.
- Prefer editing existing files over adding parallel duplicate scripts.
- Keep docs accurate when changing CLI flags or JSON schema.
