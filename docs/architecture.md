# Architecture

End-to-end local dubbing stack for Hebrew documentary / news video on Apple Silicon.

## Why this shape

Naively piping a Kan-style documentary into Whisper → translate → TTS → LatentSync fails on three edges:

1. **B-roll / wrong face** — archival or listening faces get warped if lip-sync runs without Active Speaker Detection.
2. **Duration mismatch** — Hebrew is dense; English TTS often overruns the original window.
3. **BGM clutter** — dramatic music tanks diarization and ASR; stems must be separated first.

## Pipeline phases

```text
┌─────────────┐   ┌──────────────┐   ┌─────────────────┐
│  Source AV  │──▶│ Demucs stems │──▶│ Pyannote + ASR  │
└─────────────┘   │ vocals/BGM   │   │ segments.json   │
                  └──────────────┘   └────────┬────────┘
                                              │
                  ┌──────────────┐   ┌────────▼────────┐
                  │ Time-stretch │◀──│ Duration-aware  │
                  │ TTS English  │   │ translation     │
                  └──────┬───────┘   └─────────────────┘
                         │
                  ┌──────▼───────┐   ┌─────────────────┐
                  │ ASD gate     │──▶│ LatentSync only │
                  │ (TalkNet…)   │   │ when face talks │
                  └──────┬───────┘   └────────┬────────┘
                         │                    │
                  ┌──────▼────────────────────▼────────┐
                  │ FFmpeg: duck BGM + mux final AV    │
                  └────────────────────────────────────┘
```

| Phase | Goal | Tools (current / planned) |
|-------|------|---------------------------|
| 1 | Vocals vs background | **HTDemucs** (`demucs`) |
| 2 | Who / when / what (Hebrew) | **Pyannote 3.1** + **faster-whisper** ([`ivrit-ai/whisper-large-v3-turbo-ct2`](https://huggingface.co/ivrit-ai/whisper-large-v3-turbo-ct2)) |
| 3 | Length-constrained EN text | TranslateGemma / MLX LLM |
| 4 | EN speech + fit window | **Qwen3-TTS 1.7B-Base** zero-shot clone; F5 optional |
| 5 | Lip-sync only active faces | TalkNet/SyncNet → **LatentSync** (Wav2Lip fallback) |
| 6 | Master | FFmpeg ducking + mux |

## Mac / Apple Silicon notes

- Prefer **uv** + Python **3.12**.
- Whisper: **faster-whisper** + local `models/whisper-large-v3-turbo-ct2` (Hub: `ivrit-ai/whisper-large-v3-turbo-ct2`).
- TranslateGemma: PyTorch **MPS**.
- LatentSync: CUDA-first upstream; expect `PYTORCH_ENABLE_MPS_FALLBACK=1` and possible CPU fallbacks. Keep Wav2Lip/SadTalker as Plan B.
- Unified memory helps large models, but still isolate phases to profile RAM.

## Data contract (Phase 1–2)

`inference/extract_pipeline.py` writes `outputs/<run_id>/segments.json`:

- `vocals` / `background` paths for later remix
- `segments[]` with `speaker_id`, `start`, `end`, `duration`, `text` (Hebrew)

Downstream stages must preserve `start`/`end` as the authority for timeline placement.

## What is implemented vs planned

| Piece | Status |
|-------|--------|
| `inference/transcribe.py` | ✅ |
| `inference/translate.py` | ✅ |
| `inference/extract_pipeline.py` (Demucs + Pyannote + Whisper merge) | ✅ |
| Duration-aware translation prompts | ⬜ |
| TTS (Qwen3-TTS 1.7B-Base clone) | ✅ `inference/tts_qwen.py` + `build_preview --tts-engine qwen` |
| ASD gate | ⬜ |
| LatentSync integration | ⬜ (vendored under `third_party/LatentSync`) |
| Final ducking/mux CLI | ⬜ |

See [pipeline.md](pipeline.md) for CLI usage and [setup.md](setup.md) for install.
