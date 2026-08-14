# Dubbing editor

A local web UI over the dubbing pipeline: import a video, look at the segments the
pipeline produced, correct them, save, re-run. Think "a much simpler Premiere Pro",
where the only track is the segment list and the only edit is what each segment says
and who says it.

## Run it

```bash
uv sync --extra app
uv run --extra app uvicorn editor.app:app --reload --port 8765
# then open http://127.0.0.1:8765
```

Runs are read from `outputs/` next to this package. To serve another checkout's runs
(e.g. a worktree serving the main clone's outputs):

```bash
DUBBING_OUTPUTS=/path/to/DubbingQwen/outputs uv run --extra app uvicorn editor.app:app
```

`ffmpeg` must be on PATH (it already is, for the pipeline) — the editor shells out to
it to cut per-segment preview slices out of `source.wav`.

## The flow

1. **Import** — paste a URL or pick a local file, choose source/target language and an
   optional duration limit. The server launches `python -m dubbing <source> -o
   outputs/<slug> …` as a subprocess and streams its log into the job panel. When it
   finishes, the browser jumps to that run's editor.
2. **Edit** — one card per segment: source text, translation, speaker, spoken language,
   per-segment target language, a passthrough checkbox, and ▶ buttons for the original
   and the dubbed audio of that segment.
3. **Save** — sends only the fields you touched. The response says which stage has to
   re-run for the edits to reach the mix.
4. **Re-run** — `POST /api/runs/<run>/rerun` runs the pipeline again with
   `--force <stage>`; everything downstream invalidates through the usual fingerprint
   chain, everything upstream is reused.

## Which stage an edit forces

`editor/edits.py` maps each editable field to the earliest stage that reads it, and a
save forces the earliest stage across all edits:

| field | forces | why |
|---|---|---|
| `text` | `translate` | the translator's input changed |
| `lang` | `translate` | what the span is actually spoken in |
| `tgt_lang` | `translate` | this line goes to a different target language |
| `text_en` | `tts` | the correction *is* the translation — re-running `translate` would overwrite it |
| `speaker` | `tts` | picks a different voice-clone reference |
| `passthrough` | `tts` | keep the original audio for this segment |

The static UI may still send `lang_override`; `edits.ALIASES` maps it to `tgt_lang`.
`tts_instructions` is **rejected with a 400** (`edits.REJECTED`): Qwen3-TTS's clone
path takes no instruction argument, so storing one would be a knob that silently did
nothing. Per-segment synthesis control is `tts_opts`, which this editor does not
expose — the studio app does.

`text_en` deliberately does **not** force `translate`: `translate` is skipped (its
fingerprint is unchanged), so the hand-written line survives and only synthesis,
placement and mixing redo.

## Architecture

```
editor/
  edits.py     pure manifest patching + field→stage mapping (unit tested)
  runs.py      outputs/ discovery, manifest load/save, safe file access, ffmpeg slices
  jobs.py      `python -m dubbing …` subprocesses with a pollable log tail
  app.py       FastAPI: REST + static mount
  static/      index.html, app.js, style.css — vanilla, no build step
```

No pipeline stage is imported by the server, so no model is ever loaded in-process;
everything that computes happens in a subprocess. The manifest is written through
`dubbing.manifest.save`, so the `SEGMENT_KEYS` whitelist still applies — every field
in `edits.EDITABLE` is in that whitelist, and `edits.py` asserts it at import.

## API

| endpoint | does |
|---|---|
| `GET /api/runs` | list runs in `outputs/` |
| `GET /api/runs/{run}` | manifest view: source, speakers, stages, media, segments |
| `PATCH /api/runs/{run}/segments` | `{"edits":[{"id":0,"fields":{…}}]}` → `{changed, fields, force}` |
| `POST /api/runs/{run}/rerun` | `{"force":"tts"}` → job |
| `POST /api/import` | form: `url` **or** `file`, plus `src`, `tgt`, `duration` → job |
| `GET /api/jobs`, `GET /api/jobs/{id}`, `POST /api/jobs/{id}/cancel` | job status + log tail |
| `GET /api/runs/{run}/file/{path}` | any file inside the run (preview.mp4, clips/…) |
| `GET /api/runs/{run}/segments/{id}/original` \| `/dubbed` | per-segment audio |

## v1 limits

- **`tgt_lang` does not survive a rerun.** The editor forces `translate` for it, but
  the translate stage reads the run-level target, not `seg["tgt_lang"]` — only the
  studio app's `/retranslate` honours a per-segment target.
- Re-running forces a whole stage, not one segment: correcting one line re-synthesises
  every segment in the run.
- A re-run reconstructs whatever the manifest recorded (`runs.OPT_KEYS`: genre,
  register, transcript, tts_model, device, captions, dub_foreign) plus
  src/tgt/duration/context. A run the headless CLI made records none of them, so its
  re-run uses the defaults; pass anything missing via `extra_args` on `/rerun`.
- Jobs live in server memory: restarting the server loses the log, not the work (the
  pipeline resumes from the manifest).
- No auth, no concurrency control — a local single-user tool. Two browsers editing one
  run will clobber each other.

## Tests

```bash
uv run python -m pytest tests/test_editor_edits.py -q
```

Pure logic only — no server, no run directory, no models.
