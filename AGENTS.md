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
9. **The user's edits outrank the pipeline.** A segment's identity is `uid` (`id` is
   positional and renumbered on every re-segmentation), and a field the user edited by hand
   is flagged in `locked` — no stage rerun regenerates one. Edits go through `dubbing/edit.py`,
   never by poking the manifest. A stage's *failure* verdict is not exempt: a
   translation that fails on a span the user asked to dub (`keep.user_wants_dub`)
   leaves it visibly unfinished — `keep=false`, no `text_en`, which the editor shows as
   `untranslated` — instead of answering with a keep. A keep written over the user's
   word does not merely overrule them; it contradicts `passthrough`, and the next run's
   `apply_passthrough` flips it back and re-renders the tail of the run forever.
10. **An edit reopens the stages whose work it deleted.** Dropping a segment's
    translation, clip or placement while the run still says those stages are done is how
    an edited line ends up silent in a dub the CLI calls up to date. `edit.invalidate`
    calls `manifest.reopen_from`, which un-marks that stage onward (mix and report
    included) and keeps the *progress* marks, so the reruns resume and refill only the
    hole rather than rebuilding the whole run.

## Passthrough — the editor app's per-segment override

Some spans should play exactly as recorded: an interviewee already speaking the
target language, a piece of archive audio, a line the clone gets wrong. The
manifest carries two fields for this, both in `SEGMENT_KEYS`:

- **`passthrough`** — the user's word, written by the editor app. `true` plays the
  original audio for that segment, `false` dubs it, absent decides automatically.
- **`detected_lang`** — advisory. What the language classifier heard over that
  span (the VAD+LID runs the transcript stage now saves as `words.json:lang_runs`).
  **Nothing in the pipeline reads it**; it exists so the app can *suggest*
  passthrough. Automatic keeps are still decided by `segments.mark_keep` alone.

It rides the existing keep machinery rather than adding a parallel path: a
passthrough segment is a keep, so tts slices its original audio, the timeline
reserves its exact span (not stretchable), and mix ducks the bed to zero under it
instead of laying a dub over it. `segments.apply_passthrough` is the only thing
that reads the flag, and it runs both at the end of the segments stage and once at
startup on a finished manifest — so the app can set the flag and re-run.

A flip drops that segment's `text_en`, `text_mid`, `tts` and `place` (they were
made for the other path) and re-runs translate onward; every *other* segment keeps
its translation and its clip. Two rules the override does not get to break: it
cannot un-keep a segment with no text (nothing to speak — "never silent" wins),
and re-running the segments stage re-attaches overrides by **time**, not by id,
since ids are renumbered (`segments.carry_passthrough`).

## Language pairs

The pair is `--src` / `--tgt`, and neither half is assumed about the other.

**`--src` is a claim about the video, not about every line in it.**
`translate.segment_langs` is the one place that decides how far it holds for a
given segment. Three things know better, in order: the editor's `src_lang` /
`tgt_lang` override, the span witness `lang`, and the segment's own script —
which can only *refute* the claim, never replace it, so a Latin line inside a
Hebrew run comes back with its source **unknown** rather than guessed. Unknown is
a usable answer: the prompt then names no source language ("Translate the
following text into German") and the line goes straight to the target instead of
through a pivot hop premised on a claim just refuted. This is what an
already-dubbed source needs — told a line of English is Hebrew, the model hands
the English straight back, the echo guard correctly rejects it, and a perfectly
translatable segment ends with nothing.

**Hebrew as a target** (`dubbing/hebrew.py`). Qwen3-TTS has ten languages and
Hebrew is not one of them. A LoRA over the 1.7B Base checkpoint's `talker` adds
it *without touching a base weight*: attached, the model speaks Hebrew; disabled
(`talker.disable_adapter()`), the forward pass is the unmodified base. So one
loaded checkpoint serves both, `_Synth` toggles per call, and `merge_and_unload`
is never called — merging would bake the Hebrew deltas in and destroy that. Two
consequences to keep true:

- **Hebrew is synthesized from stressed IPA, never from Hebrew script.** The
  adapter was trained that way (Hebrew orthography leaves the vowels out), so
  `tts.synthesis_text` runs the line through ReNikud Plus and hands the model
  `ʁˈeɡa` where the record says רגע. The IPA is a synthesis input and nothing
  else: what is stored, subtitled and ASR-verified is always the orthography,
  and `SEGMENT_KEYS` has no field for it. It *is* in the clip cache key, next to
  the adapter tag — a changed G2P must re-synthesize, not replay the old reading.
- **Only the 1.7B checkpoint fits the adapter** (it carries that talker's output
  heads). The CLI refuses `--tts-model 0.6b` with `--tgt he`, and a per-segment
  `tts_opts.model` cannot override it.

Hebrew clips verify with the ivrit-ai fine-tune, not the multilingual base
model — the base model misreads Hebrew badly enough to fail good clips, and
under "never silent" a failed verification throws a correct dub away.

**Same-language pairs** (`--src he --tgt he`, `en`→`en`, …) are dubs, not no-ops:
every speech segment is re-voiced in the cloned voice. `translate.run` decides
identity *per segment* — `seg["lang"] or src` against the target — so a pure
same-language run loads no translator at all, while a third-language span inside
it still gets a real translation hop. Nothing is kept for "already the target
language": `mark_keep` voids its script and speaker evidence when the pair shares
a script, and only the pair-independent keeps remain (no text, third language,
the user's passthrough). Shortening is off for these runs — rewriting a
speaker's own words in their own language is a rewrite, not a translation — so
the timeline absorbs overhang with speed-up and drift, which `place` still keeps
non-overlapping.

## Working on this

- Fix causes in `dubbing/`, then re-run. Do not hand-edit anything under `outputs/`.
- The pure logic (segmentation, placement, translation guards) is unit tested without
  models in `tests/test_dubbing.py` — add a test alongside the change.
- Bump a stage's tag in `manifest.STAGE_TAGS` when its logic changes; that invalidates
  its cached output and everything after it. Changing what a stage's *fingerprint is
  made of* (`cli.stage_params`) invalidates it too, and is the right move when a new
  input decides the output — the tag is for logic.
- A stage that ran but could not do what was asked is marked **provisional**
  (`manifest.mark_provisional`): the transcript that fell back to captions because the
  ASR was missing runs again next time, while the stages after it keep their place in
  the chain for as long as the fallback keeps producing the same words.
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
