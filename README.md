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
│   ├── extract_pipeline.py   # Phase 1–2: Demucs + Pyannote + mlx-whisper
│   ├── transcribe.py         # mlx-whisper Hebrew ASR
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
# TranslateGemma (local dir)
uv run hf download google/translategemma-4b-it \
  --local-dir models/translategemma-4b-it

# Whisper: mlx-whisper pulls mlx-community/ivrit-ai-whisper-large-v3-mlx on first run
# Optional local cache:
uv run hf download mlx-community/ivrit-ai-whisper-large-v3-mlx \
  --local-dir models/ivrit-ai-whisper-large-v3-mlx
```

## Status

| Stage | CLI | State |
|-------|-----|-------|
| ASR (mlx-whisper) | `inference/transcribe.py` | Ready |
| Translate | `inference/translate.py` | Ready |
| Extract + diarize + ASR merge | `inference/extract_pipeline.py` | Ready |
| TTS / ASD / LatentSync / master | — | Planned (see docs) |
