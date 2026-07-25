# AGENTS.md

Local video dubbing: a video plus its transcript goes in, a dubbed video comes out.
Everything runs on this machine — no API calls, no cloud.

## Run it

```bash
uv run python -m dubbing "https://www.youtube.com/watch?v=VIDEO_ID"
uv run python -m dubbing input.mp4 --captions captions.json3 -o outputs/myrun
uv run python -m dubbing "<url>" --duration 320          # first N seconds, for iterating
uv run python -m dubbing "<url>" --force translate       # redo one stage
uv run python -m dubbing "<url>" --transcript captions    # use captions instead of ASR
uv run python -m pytest tests/test_dubbing.py -q
```

Stages run in order and each is skipped when its inputs and outputs are unchanged, so
re-running is cheap and an interrupted run resumes where it stopped. `--force <stage>`
re-runs one stage; everything downstream of it invalidates automatically.

## The pipeline

`fetch → stems → transcript → segments → translate → tts → timeline → mix → report`

One module per stage in `dubbing/`, all state in `outputs/<run>/manifest.json`.

| Stage | Does | Key output |
|---|---|---|
| fetch | yt-dlp video + captions, ffmpeg → 44.1kHz wav | `source.wav`, `captions.json3` |
| stems | Demucs `htdemucs_ft` | `stems/{vocals,background}.wav` |
| transcript | local Whisper on `source.wav`; captions mark target-language spans | `words.json` |
| segments | words → segments; Pyannote labels speakers | `manifest.segments` |
| translate | Gemma 4 12B (MLX, QAT 4-bit), one segment at a time | `segment.text_en` |
| tts | Qwen3-TTS zero-shot clone, verified | `clips/*.wav` |
| timeline | places every clip | `segment.place` |
| mix | duck bed, add speech, mux | `dub.wav`, `preview.mp4` |
| report | coverage + warnings, fails on gaps | `report.json` |

## Invariants — do not break these

Each one exists because it was a real bug class. Changing a stage means keeping them
true, not working around them.

1. **Never silent.** Every segment ends up either dubbed or playing its original audio.
   TTS that cannot be verified falls back to `keep`, never to nothing.
2. **Never truncated, never overlapping.** `timeline.place()` is the only thing that
   decides where audio goes, and it asserts non-overlap and that every clip's slot is at
   least as long as its audio. `mix` *adds* into the output and asserts the span was
   empty; there is no overwrite path.
3. **Speech presence never comes from the vocals stem.** Demucs routes speech into the
   music stem often enough that trusting it silently drops whole passages. ASR runs on
   the full mix and the dead-air scan reads `source.wav`. The vocals stem is only for
   clone references and diarization.
4. **Transcribe locally; trust captions only about script.** Auto-captions mangle the
   words that matter (`שייחה מוזה … בקטאר` → `ש חמוזה … בקטב`) and no translator recovers
   from that. Captions are still authoritative for *where* the target language is spoken:
   those spans become kept segments outright, because ASR in the source language either
   garbles or skips them.
5. **One TTS call per segment.** No sub-unit splitting, so no way for half a line to
   vanish.
6. **Shortening is bounded and logged.** A line is only shortened when it would otherwise
   drift past `DRIFT_MAX`, it must keep every name, number and negation, and every
   instance appears in `report.json`.
7. **No per-video content.** No glossaries, no name-specific regexes, no
   `if segment_id == 17`. A rule that only helps one video is a bug in the general rule —
   fix the general rule.
8. **The manifest stays small.** `manifest.SEGMENT_KEYS` is a whitelist enforced on save.
   If a stage needs a new field, add it there deliberately.

## Working on this

- Fix causes in `dubbing/`, then re-run. Do not hand-edit anything under `outputs/`.
- The pure logic (segmentation, placement, translation guards) is unit tested without
  models in `tests/test_dubbing.py` — add a test alongside the change.
- Bump a stage's tag in `manifest.STAGE_TAGS` when its logic changes; that invalidates
  its cached output and everything after it.
- `uv` only. Python 3.12. Secrets in `.env` (`HF_TOKEN` for Pyannote).

## Device notes (Apple Silicon)

- Translation runs on **MLX**, not MPS: Gemma 4 12B at 4-bit is ~11 GB live (and
  briefly twice that in MLX's buffer cache while loading, which `load()` clears).
- Gemma 4 changed the turn syntax to `<|turn>role … <turn|>`, so prompts go through
  the tokenizer's chat template, with `enable_thinking=False` to keep it from
  reasoning instead of translating.
- Qwen3-TTS: **float32 on MPS** — float16 NaNs in the code predictor.
- faster-whisper: **CPU only** (CTranslate2 has no MPS backend).
- Models are sequential, never co-resident: the translator is freed before TTS loads.
