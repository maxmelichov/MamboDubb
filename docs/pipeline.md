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

(`build_preview` defaults to **TranslateGemma-4B** with marker-aligned neighbor context.
Pass `--translator llm` to A/B mlx-lm with synopsis/memory.)

## Preview / TTS (Phase 3–4)

```bash
uv run python inference/build_preview.py outputs/<run>
```

Consumes `segments.json` (or `translated_segments.json` with `--skip-translate`):

1. Drop silent-vocals hallucinations on HE-dub rows only (KEEP / non-HE never dropped); also probe `source.wav` so applause-masked real speech is kept
2. Dedupe corrupted repeated EN / KEEP text; clamp overlapping phrases; stamp immutable `source_start`/`source_end`
3. Optional YouTube caption cross-check (`--youtube-url` only — off by default). ASR text is the MT source of truth
4. Retag phonetic-EN + EN-sandwich + speaker-language consistency → `keep_original` where appropriate
5. **TranslateGemma-4B** (default `--translator gemma`) with marker-aligned neighbor chunks; aim EN ≈ **0.8–1.2×** HE without hard truncation. A/B: `--translator llm`
6. Stable per-person voice bank (default `--speaker-bank`; `--no-speaker-bank` for fresh per-segment refs)
7. **Qwen3-TTS**: isochronous fit to each HE `source_*` window (rate **0.90–1.25**, escalate ~**1.40**, pad short clips); x-vector first (cross-lingual), then ICL with aligned bank `ref_text`; verify with dedicated English ASR (`models/faster-whisper-base.en`) + voice identity; units at pauses ≥1.2s
8. Remix → `preview.mp4` + `preview_en.srt`: EN TTS + continuous ducked leak-cleaned music; KEEP uses original `source.wav`. Mux blocked if any HE window lacks validated TTS (`--allow-missing-tts` to override)
9. Optional `--qa` re-ASR report → `qa_report.json` (coverage, early-ends, music jumps)

### Pause-aware utterances

Same speaker + short silence is **one utterance**, not a new sentence:

- gap ≤ `--max-pause` (default 1.0s) → merge; TTS places phrases on HE offsets
- gap > `--max-pause` or speaker change → new utterance
- Finished sentence + gap ≥ **0.8s** + new speech-act (thanks / address) → **do not merge** (avoids dead air + wrong glue)
- Hard pauses ≥ **0.8s** inside an utterance stay as alignment anchors (EN split + silence preserved)
- Unfinished clause + next speaker’s first completing phrase → stitch first clause onto A
- `--max-dub-pause` (default **0** = off) optionally caps mid-utterance silence when set (e.g. `0.7`)

### English ↔ phrase mapping

Utterance-level EN is mapped onto `phrases[]` before TTS by **HE speech-duration weight**:

| Case | Behavior |
|------|----------|
| Same # EN sentences as phrases | 1:1 assignment |
| Hard HE pause (≥0.8s) between phrases | Split EN at punctuation/clause boundaries; **never** pack across the pause |
| Fewer EN sentences than HE phrases (soft gaps only) | Pack each full sentence onto a contiguous HE span |
| JOIN of unfinished + continuation | Split EN by HE word/duration weight — never duplicate the tail onto both segments |
| Dangling `that` / `which` / `who…` | Merge into the previous phrase (avoids orphan tails that sound “skipped”) |
| Hebrew `ו-` + English `and…` | Prefer keeping `And …` on the continuation phrase |

Do **not** retime HE phrase boundaries to invent pauses — ASR `start`/`end` stay authoritative (aside from optional `--max-dub-pause` compaction).

### TTS timing (Qwen)

Hebrew ASR `source_start`/`source_end` are the immutable timeline authority.
Fitted occupancy is stored separately as `tts_start`/`spoken_end`.
Goal: **isochronous fit** — English fills the Hebrew chunk (≈0.8–1.2×) via atempo, never mid-word cut-offs.

Each dub utterance is split into **units** only at internal pauses ≥ **1.2s**.
Small HE breaths collapse into one continuous English clip. Big pauses stay as
real silence between units. Object-continuation stubs (`אני לא מכירה.` + `אף…`)
merge into one phrase before MT/TTS.

| Situation | Behavior |
|-----------|----------|
| One utterance, no long pause | One Qwen synth of full `text_en`, fit to HE `source_*` window |
| Internal pause ≥ **1.2s** | Split into units; EN split by speech-duration weight; mid-silence capped at **1.2s** |
| Long turn (> **~10s**) without big pause | Sub-split at HE sentence ends / caption gaps (timeline anchors) |
| EN shorter than HE window | Stretch toward window (rate ≥ **0.90**); **pad** canvas to `source_end` |
| EN longer than HE window | Compress with atempo ≤ **1.25×**; meaning-preserving shorten-and-retry when ≫1.2× |
| Still too long | Escalate atempo ≤ **1.40×**, then soft-fade to window (no mid-word chop) |
| Fit rate band | **0.90–1.25**; hard escalate to **1.40** |
| Last unit before KEEP-English | Yield ~**0.12s**; no gap-bleed into KEEP |
| Last unit before another dub | Tiny bleed ≤ **0.20s** + crossfade (prefer compress over overrun) |
| Clone garbled / chipmunk / Hebrew bleed | English-only ASR (`faster-whisper-base.en`) + voice-identity verify; x-vector first then ICL (ICL skipped on Hebrew bleed); alt-ref escalation; best-effort place; mux blocked if still missing (`--allow-missing-tts` to override) |
| Clone refs | **Stable per-person bank** (default). `--no-speaker-bank` for fresh per-segment refs |
| Leading hush in Qwen raw | Trim before fit (~0.5s) so speech starts on the HE onset |
| Speaker mostly KEEP-English | Consistency pass flips their weak HE-dub turns → KEEP |
| KEEP caption is Hebrew script | Re-ASR English for subtitle, or suppress |
| Overlapping diarization turns | Segment-timeline clamp splits overlap so neither dub is mid-word-cut |
| Speaker merge | Conservative embedding merge on (same real person); never share a prompt across distinct people |
| Non-Latin MT (CJK/…) | Rejected by `is_latin_english`; stripped before TTS |
| Caption chrome (`>>`, `[music]`) | Stripped in MT postprocess + TTS prep |
| Remix under HE dubs | Continuous music at `--duck-gain` (default **0.45**) with smooth attack/release; leak-cleaned bed (never hard-mute). Never mix `vocals.wav` under EN TTS |
| KEEP (non-HE) | Original `source.wav` slice (speech + music); equal-power crossfade with bed |
| Missing `sox` | Warning from Qwen tokenizer; install with `brew install sox` |

```bash
# Full preview (re-translate + TTS + QA) — TranslateGemma + speaker bank defaults
uv run python inference/build_preview.py outputs/kan11_5m --qa

# A/B mlx-lm instruct translator
uv run python inference/build_preview.py outputs/kan11_5m --translator llm --qa

# Optional YouTube caption annotate (ASR still owns MT text)
uv run python inference/build_preview.py outputs/kan11_5m \
  --youtube-url 'https://www.youtube.com/watch?v=Z2Uq4aX2KPs' --qa

# Reuse translations; regenerate TTS (stable bank)
uv run python inference/build_preview.py outputs/kan11_5m --skip-translate

# Fresh per-segment refs instead of bank
uv run python inference/build_preview.py outputs/kan11_5m --skip-translate --no-speaker-bank

# Skip clone verify (faster debug)
uv run python inference/build_preview.py outputs/kan11_5m \
  --skip-translate --no-clone-verify

# Degraded mux when some clones still fail
uv run python inference/build_preview.py outputs/kan11_5m \
  --skip-translate --allow-missing-tts

# Optional: force ICL with Hebrew ref_text
uv run python inference/build_preview.py outputs/kan11_60s \
  --skip-translate --qwen-icl

# Legacy F5
uv run python inference/build_preview.py outputs/kan11_60s \
  --skip-translate --tts-engine f5 --tts-speed 1.0

# TTS-only / subset
uv run python inference/tts_qwen.py outputs/kan11_60s
uv run python inference/build_preview.py outputs/kan11_5m \
  --skip-translate --reuse-tts --tts-segments 5,11 --allow-missing-tts
```

Output: `outputs/<run>/preview.mp4` (EN voice over continuous ducked BGM + soft EN subs), plus `translated_segments.json`, `dubbed_audio.wav`, `tts_clips/`.

See [architecture.md](architecture.md).
