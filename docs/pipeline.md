# Pipeline usage

## Phase 1–2: extract → separate → diarize → transcribe

```bash
uv run python inference/extract_pipeline.py path/to/video.mp4
```

Artifacts land in `outputs/<timestamp>_<stem>/`:

| File | Meaning |
|------|---------|
| `source.wav` | Full mix extracted from the input |
| `vocals.wav` | Demucs vocal stem (fed to diarization + ASR) |
| `background.wav` | Music / SFX stem (kept for final remix) |
| `segments.json` | Speaker-labeled Hebrew timeline |

Useful flags:

```bash
# Custom output JSON
uv run python inference/extract_pipeline.py clip.mp4 -o outputs/clip.json

# Reuse a fixed work directory
uv run python inference/extract_pipeline.py clip.mp4 --workdir outputs/debug_run

# Skip Demucs (debug ASR/diarization on the mix)
uv run python inference/extract_pipeline.py clip.mp4 --skip-demucs
```

### `segments.json` shape

```json
{
  "source": "/abs/path/clip.mp4",
  "workdir": "/abs/path/outputs/…",
  "vocals": "/abs/path/outputs/…/vocals.wav",
  "background": "/abs/path/outputs/…/background.wav",
  "language": "he",
  "segments": [
    {
      "speaker_id": "SPEAKER_00",
      "start": 12.4,
      "end": 16.1,
      "duration": 3.7,
      "text": "…"
    }
  ]
}
```

## Standalone ASR

```bash
uv run python inference/transcribe.py vocals.wav --timestamps
# default: models/whisper-large-v3-turbo-ct2 (ivrit-ai CT2)
```

## Standalone translation

```bash
uv run python inference/translate.py -s he -t en "טקסט בעברית"
```

## Later phases (not wired yet)

Planned consumers of `segments.json`:

1. **Translate** each `text` with a duration budget ≈ `duration` seconds.
2. **TTS (F5-TTS)** — zero-shot clone from `vocals.wav` refs; control pace with `--tts-speed`.
3. **Time fit** — F5 `speed` re-infer (`--tts-fit-duration`) then pad/trim to `[start, end]`.
4. **ASD** gate → optional LatentSync on that window.
5. **Mux** English vocals over `background.wav` with ducking.

### Pause-aware utterances

Same speaker + short silence is **one utterance**, not a new sentence:

- gap ≤ `--max-pause` (default 1.0s) → merge; TTS inserts matching silence between phrases
- gap > `--max-pause` or speaker change → new utterance

```bash
uv run python inference/extract_pipeline.py clip.mp4 --max-duration 60 --max-pause 1.0
uv run python inference/build_preview.py outputs/<run> --skip-translate --max-pause 1.0
```


```bash
# Reuse existing HE→EN text_en; F5-TTS with manual speed
uv run python inference/build_preview.py outputs/kan11_60s \
  --skip-translate \
  --tts-speed 1.0

# Speak faster to pack more English into short windows
uv run python inference/build_preview.py outputs/kan11_60s \
  --skip-translate --tts-speed 1.2

# Fixed speed only (no auto re-infer)
uv run python inference/build_preview.py outputs/kan11_60s \
  --skip-translate --tts-speed 1.0 --no-tts-fit-duration

# TTS-only (writes tts_clips/ + updates translated_segments.json)
uv run python inference/tts_f5.py outputs/kan11_60s --tts-speed 1.1
```

Output: `outputs/<run>/preview.mp4` (EN F5 voice over ducked BGM + soft EN subs).

See [architecture.md](architecture.md).
