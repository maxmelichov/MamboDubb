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
# default model: mlx-community/ivrit-ai-whisper-large-v3-mlx
```

## Standalone translation

```bash
uv run python inference/translate.py -s he -t en "טקסט בעברית"
```

## Later phases (not wired yet)

Planned consumers of `segments.json`:

1. **Translate** each `text` with a duration budget ≈ `duration` seconds.
2. **TTS** per `speaker_id` using a clean vocal reference clip from `vocals.wav`.
3. **Time-stretch** TTS to `[start, end]`.
4. **ASD** gate → optional LatentSync on that window.
5. **Mux** English vocals over `background.wav` with ducking.

See [architecture.md](architecture.md).
