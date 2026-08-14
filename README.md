<p align="center">
  <img src="docs/assets/mambodubb-banner.png" alt="MamboDubb" width="720">
</p>

<h3 align="center">Dub any video into another language — entirely on your own machine.</h3>

<p align="center">
  <a href="https://github.com/maxmelichov/MamboDubb/releases/latest"><b>Download for Mac (Apple Silicon)</b></a>
  ·
  <a href="#quick-start-cli">Run from source</a>
  ·
  <a href="docs/APP_ARCHITECTURE.md">Architecture</a>
</p>

---

MamboDubb takes a video (a file or a YouTube link), transcribes it, translates the
transcript with context, speaks every line **in the original speaker's own voice**
using zero-shot cloning, and mixes the new speech back over the original music at the
original timing. Nothing is uploaded anywhere: every model — ASR, translation,
voice synthesis, source separation, diarization — runs locally.

## The app

A desktop editor built around the script, not the timeline: every line shows the
original text and its translation together. Fix a translation in place, choose per
line whether it is dubbed or keeps the original audio, re-voice a single line in
seconds, and watch the source and output waveforms side by side.

- **Script-first editing** — original + translation on every row, inline edits,
  RTL-correct Hebrew alongside English.
- **Per-line control** — dub / keep original, re-translate, re-voice, split, merge,
  speaker reassignment, per-segment voice options.
- **Honest state** — every line says whether it is dubbed, kept, waiting, or failed,
  and one click fixes the ones that are waiting.
- **Hand edits are sacred** — a corrected line is locked; no re-run ever overwrites it.
- **A/B everything** — play the original span and the dubbed take of any line,
  back to back.

### Install (macOS, Apple Silicon)

Grab the `.dmg` from the [latest release](https://github.com/maxmelichov/MamboDubb/releases/latest),
drag MamboDubb to Applications, and launch. The app drives the pipeline in this
checkout, so you also need the source set up once:

```bash
git clone https://github.com/maxmelichov/MamboDubb.git
cd MamboDubb
uv sync                    # Python 3.12, uv only
cp .env.example .env       # set HF_TOKEN for speaker diarization (optional)
```

> **Apple Silicon is required** — translation runs on MLX, which needs an M-series
> chip. There is no Intel or iOS build.

## Quick start (CLI)

The whole pipeline is also a single headless command:

```bash
uv run python -m dubbing "https://www.youtube.com/watch?v=VIDEO_ID"
uv run python -m dubbing input.mp4 --duration 300        # just the first 5 minutes
```

The result lands in `outputs/<run>/preview.mp4`, next to a `report.json` that
accounts for every second of audio. Re-running is incremental — finished stages are
cached and only what changed is redone.

## Languages

| | |
|---|---|
| **Spoken language (source)** | Hebrew, English, Arabic, Russian, French, Spanish, German |
| **Dub into (target)** | English, Russian, French, Spanish, German, Italian, Portuguese, Chinese, Japanese, Korean |

Targets are bounded by what Qwen3-TTS can speak — Hebrew and Arabic are
source-only for now.

## How it works

```
fetch → stems → transcript → segments → translate → tts → timeline → mix → report
```

One module per stage, one `manifest.json` per run as the single source of truth.
The desktop app is a thin Tauri shell over a local FastAPI server that calls the
same pipeline — the app is a second front end, never a fork. The models:
faster-whisper (ASR + verification), Gemma (translation, via MLX), Qwen3-TTS
(voice cloning), Demucs (music/voice separation), Pyannote (speakers). They load
sequentially and never co-exist in memory, so 16–32 GB of unified memory is enough.

Guarantees the pipeline enforces on itself: no second of audible speech is ever
silently dropped, dubbed lines never overlap, and every synthesized take is
verified by a second ASR pass before it is accepted.

## Building the app from source

```bash
uv run scripts/build_desktop.py check   # toolchain sanity
uv run scripts/build_desktop.py build   # → app/desktop/src-tauri/target/release/bundle/
```
