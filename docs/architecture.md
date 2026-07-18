# Architecture

End-to-end local dubbing stack for Hebrew documentary / news video on Apple Silicon.

## Why this shape

Naively piping a Kan-style documentary into Whisper → translate → TTS → LatentSync fails on three edges:

1. **B-roll / wrong face** — archival or listening faces get warped if lip-sync runs without Active Speaker Detection.
2. **Duration mismatch** — Hebrew is dense; English TTS often overruns (or underfills) the original window.
3. **BGM clutter** — dramatic music tanks diarization and ASR; stems must be separated first. Empty vocals + music also produce Whisper hallucinations that must be dropped.

## Pipeline phases

```text
┌─────────────┐   ┌──────────────┐   ┌─────────────────┐
│  Source AV  │──▶│ Demucs stems │──▶│ Pyannote + ASR  │
└─────────────┘   │ vocals/BGM   │   │ segments.json   │
                  └──────────────┘   └────────┬────────┘
                                              │
                  ┌──────────────┐   ┌────────▼────────┐
                  │ Time-fit TTS │◀──│ mlx-lm HE→EN    │
                  │ (Qwen clone) │   │ + phrase plan   │
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
| 2 | Who / when / what | **Pyannote 3.1** + **faster-whisper** ([`ivrit-ai/whisper-large-v3-turbo-ct2`](https://huggingface.co/ivrit-ai/whisper-large-v3-turbo-ct2)); lang detect + KEEP |
| 3 | Length-aware EN text | **mlx-lm** (default Qwen2.5-7B 4bit) in `build_preview`; episode synopsis + rolling HE→EN memory + glossary; sentence→phrase packing; shorten-retry |
| 4 | EN speech + fit window | **Qwen3-TTS 1.7B-Base** zero-shot clone (per-speaker cached prompt); pause cap 0.7s; even pace (rate≈1.0, speed up only if ≫ slot, cap ~1.18×); F5 optional |
| 5 | Lip-sync only active faces | TalkNet/SyncNet → **LatentSync** (Wav2Lip fallback) |
| 6 | Master | FFmpeg remix (`build_preview`: constant BGM gain + loudness-matched speech) |

## Mac / Apple Silicon notes

- Prefer **uv** + Python **3.12**.
- Whisper: **faster-whisper** + local `models/whisper-large-v3-turbo-ct2`.
- Preview translation: **mlx-lm** (not TranslateGemma — pad-only / empty on this stack).
- Qwen TTS needs system **SoX** (`brew install sox`) for ref normalization.
- LatentSync: CUDA-first upstream; expect `PYTORCH_ENABLE_MPS_FALLBACK=1` and possible CPU fallbacks. Keep Wav2Lip/SadTalker as Plan B.
- Unified memory helps large models, but still isolate phases to profile RAM.

## Data contract (Phase 1–2)

`inference/extract_pipeline.py` writes `outputs/<run_id>/segments.json`:

- `vocals` / `background` / `source` paths for remix
- `segments[]` with `speaker_id`, `language`, `keep_original`, `start`, `end`, `duration`, `text`, `phrases[]` (`pause_after`)

Downstream stages must preserve `start`/`end` as the authority for timeline placement. Non-Hebrew / `keep_original` turns keep original vocals.

### Phrase / TTS contract (Phase 3–4)

- Translate at **utterance** level with episode synopsis + rolling translation memory; map EN onto phrases without orphan relative clauses (`that…`).
- Prefer **constant speaking rate (~1.0)**; shorten EN and retry before speeding up; speed up only when still clearly longer than the HE window (>~1.12×, cap ~1.18×).
- Cap intra-utterance pauses at ~0.7s (utterance anchors preserved).
- Never slow below 1.0 to fill a long HE slot; never globally rewrite phrase times across utterances to “fix” pauses.
- One canonical voice ref per speaker (`tts_refs/speaker_bank.json`); reuse clone prompt per speaker.

## What is implemented vs planned

| Piece | Status |
|-------|--------|
| `inference/transcribe.py` | ✅ |
| `inference/translate.py` (TranslateGemma CLI) | ✅ |
| `inference/extract_pipeline.py` (Demucs + Pyannote + Whisper + gap/stitch/KEEP) | ✅ |
| Silent-vocals hallucination drop | ✅ |
| mlx-lm duration-aware translate in `build_preview` | ✅ synopsis + rolling memory + stub absorb |
| Seed translations from a shorter run | ✅ |
| EN sentence→phrase packing + dangling-`that` repair | ✅ hard-pause anchors (≥0.8s) |
| Even-pace Qwen fit (no slow-mo / shorten-retry / ≤1.18×) | ✅ shorten at ≥1.28× |
| Per-speaker voice bank; pause preserve (compact opt-in) | ✅ |
| YouTube caption ASR cross-check | ✅ `inference/youtube_subs.py` |
| Speech-run coalescing (gap <0.3s) | ✅ |
| EN-sandwich retag | ✅ |
| QA re-ASR (`--qa`) | ✅ |
| Long-phrase split retiming on `.` | ⬜ avoided (desynced dubs); extract splits at sentence ends for phrases |
| TTS (Qwen3-TTS 1.7B-Base clone + place) | ✅ `inference/tts_qwen.py` + `build_preview` |
| Preview remix + EN SRT | ✅ |
| ASD gate | ⬜ |
| LatentSync integration | ⬜ (vendored under `third_party/LatentSync`) |

See [pipeline.md](pipeline.md) for CLI usage and [setup.md](setup.md) for install.
