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

## Preview / TTS (Phase 3–4)

Consumers of `segments.json`:

1. **Translate** each Hebrew `text` (non-Hebrew turns keep original audio).
2. **TTS (Qwen3-TTS 1.7B-Base)** — zero-shot clone from each phrase’s vocal ref → English.
3. **Time fit** — concat phrases + pad/soft-trim to `[start, end]`.
4. **ASD** gate → optional LatentSync on that window (not wired yet).
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
# Default: Qwen 1.7B-Base zero-shot from vocal refs
uv run python inference/build_preview.py outputs/kan11_60s --skip-translate

# Optional: ICL with Hebrew ref_text (slower cross-lingual)
uv run python inference/build_preview.py outputs/kan11_60s \
  --skip-translate --qwen-icl

# Legacy F5
uv run python inference/build_preview.py outputs/kan11_60s \
  --skip-translate --tts-engine f5 --tts-speed 1.0

# TTS-only
uv run python inference/tts_qwen.py outputs/kan11_60s
```

Output: `outputs/<run>/preview.mp4` (EN voice over ducked BGM + soft EN subs).

See [architecture.md](architecture.md).
