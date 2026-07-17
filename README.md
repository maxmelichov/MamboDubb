# DubbingQwen

Local Hebrew→English video dubbing on Apple Silicon: source separation, diarization, ASR, translation, TTS, and lip-sync.

## Quick start

```bash
uv sync
cp .env.example .env   # set HF_TOKEN for Pyannote

uv run python inference/transcribe.py path/to/audio.wav --timestamps
uv run python inference/translate.py -s he -t en "שלום, מה שלומך?"
uv run python inference/extract_pipeline.py path/to/video.mp4
```

Full install notes: [docs/setup.md](docs/setup.md). Architecture: [docs/architecture.md](docs/architecture.md). Agent guidance: [AGENTS.md](AGENTS.md).

## Layout

```text
DubbingQwen/
├── AGENTS.md
├── docs/
│   ├── architecture.md
│   ├── pipeline.md
│   └── setup.md
├── inference/           # First-party CLIs
│   ├── extract_pipeline.py   # Phase 1–2: Demucs + Pyannote + faster-whisper
│   ├── transcribe.py         # ivrit-ai Whisper CT2 (Hebrew ASR)
│   └── translate.py
├── models/              # Local checkpoints (gitignored)
├── outputs/             # Run artifacts (gitignored)
├── pyproject.toml       # uv / dependency source of truth
└── third_party/
    ├── LatentSync/
    └── Qwen3-TTS/
```

## Models

```bash
uv run hf download ivrit-ai/whisper-large-v3-turbo-ct2 \
  --local-dir models/whisper-large-v3-turbo-ct2
uv run hf download google/translategemma-4b-it \
  --local-dir models/translategemma-4b-it
```

## Status

| Stage | CLI | State |
|-------|-----|-------|
| ASR (faster-whisper CT2) | `inference/transcribe.py` | Ready |
| Translate | `inference/translate.py` | Ready |
| Extract + diarize + ASR merge | `inference/extract_pipeline.py` | Ready |
| Preview dub (F5-TTS) | `inference/build_preview.py` | Ready |
| ASD / LatentSync / master | — | Planned (see docs) |
