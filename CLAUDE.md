# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Read [AGENTS.md](AGENTS.md) first** — it is the authoritative doc for the pipeline stages, the invariants (which exist because each was a real bug class — do not break them), and Apple Silicon device notes. This file only adds what AGENTS.md doesn't cover.

## Commands

```bash
uv sync                                                  # install (uv only, Python 3.12)
uv run python -m dubbing "https://www.youtube.com/watch?v=VIDEO_ID"
uv run python -m dubbing input.mp4 --captions captions.json3 -o outputs/myrun
uv run python -m dubbing "<url>" --duration 320          # dub only the first N seconds (fast iteration)
uv run python -m dubbing "<url>" --force translate       # re-run one stage (downstream invalidates automatically)
uv run python -m pytest tests/test_dubbing.py -q         # all tests — pure logic, no models needed
uv run python -m pytest tests/test_dubbing.py -q -k <name>   # single test
uv run ruff check dubbing tests                          # lint (line-length 100)
```

## Architecture in one paragraph

A staged pipeline — `fetch → stems → transcript → segments → translate → tts → timeline → mix → report` — with one module per stage in `dubbing/`, orchestrated by `dubbing/cli.py`. All cross-stage state lives in `outputs/<run>/manifest.json` (`dubbing/manifest.py`); segment fields are whitelisted in `manifest.SEGMENT_KEYS` and enforced on save. Each stage is skipped when its inputs and outputs are unchanged, keyed by `manifest.STAGE_TAGS` — **when you change a stage's logic, bump its tag** so cached outputs invalidate. Models (Gemma 4 12B via MLX for translation, Qwen3-TTS for synthesis, faster-whisper for ASR/verification, Demucs, Pyannote) all run locally under `models/` and are loaded sequentially, never co-resident.

## Rules that bite

- Never hand-edit anything under `outputs/` — fix the cause in `dubbing/` and re-run.
- No per-video content: no glossaries, no name-specific regexes, no `if segment_id == 17`. A rule that only helps one video is a bug in the general rule.
- `timeline.place()` is the sole authority on where audio goes (asserts non-overlap); `mix` only adds into empty spans.
- Pure logic (segmentation, placement, translation guards) is tested without models in `tests/test_dubbing.py` — add a test alongside the change.
- Secrets in `.env` (`HF_TOKEN` for gated Pyannote models; without it diarization falls back to single-speaker).
- `transformers` is pinned to 4.57.3 via a uv override (qwen-tts needs it; mlx-lm tolerates it) — don't "fix" the version conflict by upgrading.
