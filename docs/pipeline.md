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
| `segments.json` | Speaker-labeled timeline (`he` → dub, non-`he` → keep) |

Useful flags:

```bash
# Custom output JSON
uv run python inference/extract_pipeline.py clip.mp4 -o outputs/clip.json

# Reuse a fixed work directory
uv run python inference/extract_pipeline.py clip.mp4 --workdir outputs/debug_run

# Skip Demucs when stems already exist
uv run python inference/extract_pipeline.py clip.mp4 --workdir outputs/run --skip-demucs

# Cap length / utterance merge
uv run python inference/extract_pipeline.py clip.mp4 --max-duration 60 --max-pause 1.0
```

### Extract robustness (current)

- Per-turn language detect (`he` / `en` / `ar`); non-Hebrew → `keep_original: true`
- Phonetic-English HE transcripts retagged to KEEP
- Gap recovery + turn-tail residual ASR when diarization/ASR leaves holes
- Unfinished-utterance stitch (same-speaker long pauses + short cross-speaker completions)
- Speaker continuity stabilize (short-gap monologue flips)
- **Silent-vocals drop** — Whisper hallucinations on music beds with empty Demucs vocals are discarded (e.g. fake “thank you” over intro music)

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
      "language": "he",
      "keep_original": false,
      "start": 12.4,
      "end": 16.1,
      "duration": 3.7,
      "text": "…",
      "phrases": [
        {"text": "…", "start": 12.4, "end": 14.0, "pause_after": 0.3},
        {"text": "…", "start": 14.3, "end": 16.1, "pause_after": 0.0}
      ],
      "pauses": [0.3, 0.0]
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

(`build_preview` does **not** use this path — it uses **mlx-lm** with ±1 neighbor context.)

## Preview / TTS (Phase 3–4)

```bash
uv run python inference/build_preview.py outputs/<run>
```

Consumes `segments.json` (or `translated_segments.json` with `--skip-translate`):

1. Drop silent-vocals hallucinations (if `vocals.wav` present); also probe `source.wav` so applause-masked real speech is kept
2. Dedupe corrupted repeated EN / KEEP text; clamp overlapping phrases
3. Optional YouTube caption cross-check (`--youtube-url` / payload `youtube_url`) — annotate-only by default (auto-captions feed MT as a second ASR opinion; hard-replace is opt-in and guarded against bleed)
4. Retag phonetic-EN + EN-sandwich fragments → `keep_original` (never KEEP high-Hebrew loanword commentary)
5. **mlx-lm** utterance translate with episode synopsis + rolling HE→EN memory (± neighbors); reject memory-echo / EN≫HE; absorb unfinished stubs into same-speaker continuations
6. Distribute EN onto phrases (hard pauses ≥0.8s are anchors; soft packing only between them)
7. Per-speaker voice bank (canonical clean ref ≥~2s, or skip → per-phrase refs)
8. **Qwen3-TTS 1.7B-Base** zero-shot clone (one cached prompt per speaker); coalesce near-contiguous phrases into one TTS run
9. Preserve HE pauses by default (`--max-dub-pause 0`); shorten-EN retry when rate would exceed ~1.28× (or earlier before KEEP / tight next onset); cap short-slot end writeback; fit at ≤1.18×; yield ~0.12s before KEEP
10. Crossfade-mix overlapping clips → constant-gain BGM + speech → `preview.mp4` + `preview_en.srt`
11. Optional `--qa` re-ASR report → `qa_report.json`

### Pause-aware utterances

Same speaker + short silence is **one utterance**, not a new sentence:

- gap ≤ `--max-pause` (default 1.0s) → merge; TTS places phrases on HE offsets
- gap > `--max-pause` or speaker change → new utterance
- Finished sentence + gap ≥ **0.8s** + new speech-act (thanks / address) → **do not merge** (avoids dead air + wrong glue)
- Hard pauses ≥ **0.8s** inside an utterance stay as alignment anchors (EN split + silence preserved)
- Unfinished clause + next speaker’s first completing phrase → stitch first clause onto A
- `--max-dub-pause` (default **0** = off) optionally caps mid-utterance silence when set (e.g. `0.7`)

### English ↔ phrase mapping

Utterance-level EN is mapped onto `phrases[]` before TTS:

| Case | Behavior |
|------|----------|
| Same # EN sentences as phrases | 1:1 assignment |
| Hard HE pause (≥0.8s) between phrases | Split EN at punctuation/clause boundaries; **never** pack across the pause |
| Fewer EN sentences than HE phrases (soft gaps only) | Pack each full sentence onto a contiguous HE span |
| Dangling `that` / `which` / `who…` | Merge into the previous phrase (avoids orphan tails that sound “skipped”) |
| Hebrew `ו-` + English `and…` | Prefer keeping `And …` on the continuation phrase |

Do **not** retime HE phrase boundaries to invent pauses — ASR `start`/`end` stay authoritative (aside from optional `--max-dub-pause` compaction).

### TTS timing (Qwen)

Hebrew ASR `start`/`end` stay the timeline authority. Goal: **even speaking pace**, not exact slot fill.

| Situation | Behavior |
|-----------|----------|
| EN shorter than HE slot (normal pace) | Rate **1.0**; trailing silence in the slot |
| Slow HE (≲2 words/s) + short EN | Mild stretch (≥**0.90×**) + lead silence so speech sits later in the mouth window |
| EN only slightly long (≤ **1.12×**) | Rate **1.0**; slight overrun into following pause |
| EN clearly too long (> **1.28×**) | Translator shorten-and-retry once (reject >35% word drop); then slack borrow |
| Last dub phrase before KEEP-English | Yield ~**0.12s**; no gap-bleed; prefer shorten-retry; no overrun into KEEP |
| Next onset &lt; ~0.30s (any) | Prefer shorten + no overrun (avoids mid-sentence soft-trim) |
| Short HE slot overrun | Cap end writeback (~0.35s), never fill whole gap to next onset |
| KEEP-English late ASR start | Pull start back to energy onset on `source.wav` (bounded by previous end) |
| Still too long | Speed up only then (cap ~**1.18×**); overrun if still long (not into KEEP) |
| Near-contiguous phrases (gap <0.18s) | Coalesce into one TTS run (no mid-sentence cut) |
| Hard HE pause (≥0.8s) | Preserve silence; voice does not speak through it |
| Leading hush in Qwen raw | Trim before fit (~0.5s) so speech starts on the HE onset |
| Extreme mismatch only | Optionally borrow ≤~0.45s of adjacent pause for that clip |
| Missing `sox` | Warning from Qwen tokenizer; install with `brew install sox` |

Avoided (they desynced dubs): always-on mid-utterance pause compaction, always-expand short slots, proportional sentence-split retiming, hard-overwrite of overlapping clips.

```bash
# Full preview (translate + TTS + YouTube caption cross-check)
uv run python inference/build_preview.py outputs/kan11_5m \
  --youtube-url 'https://www.youtube.com/watch?v=Z2Uq4aX2KPs' --qa

# Reuse translations; regenerate TTS
uv run python inference/build_preview.py outputs/kan11_5m --skip-translate

# Seed EN from a shorter proven run (skips ellipsis / shorter-HE seeds)
uv run python inference/build_preview.py outputs/kan11_5m \
  --seed-translations outputs/kan11_60s/translated_segments.json

# Optional: compact mid-utterance silence + QA re-ASR
uv run python inference/build_preview.py outputs/kan11_5m \
  --skip-translate --max-dub-pause 0.7 --qa

# Optional: ICL with Hebrew ref_text (slower cross-lingual)
uv run python inference/build_preview.py outputs/kan11_60s \
  --skip-translate --qwen-icl

# Legacy F5
uv run python inference/build_preview.py outputs/kan11_60s \
  --skip-translate --tts-engine f5 --tts-speed 1.0

# TTS-only / subset
uv run python inference/tts_qwen.py outputs/kan11_60s
uv run python inference/build_preview.py outputs/kan11_5m \
  --skip-translate --reuse-tts --tts-segments 5,11
```

Output: `outputs/<run>/preview.mp4` (EN voice over ducked BGM + soft EN subs), plus `translated_segments.json`, `dubbed_audio.wav`, `tts_clips/`, `tts_refs/speaker_bank.json`.

See [architecture.md](architecture.md).
