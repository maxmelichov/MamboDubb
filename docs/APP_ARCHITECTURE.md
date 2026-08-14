# Dubbing Studio — app architecture and contracts

The editor app: import a video (file or URL), run the dubbing pipeline, then review and
correct the result segment by segment — a much simpler Premiere Pro whose clips are the
pipeline's segments.

This document is the **contract** between workstreams built in parallel. Anything here
is settled; anything not here is the implementer's call. Change a contract by editing
this file in the same PR that changes the code.

## Layering (after MamboRambo)

```
web UI (React/Vite)  →  local HTTP  →  studio server (Python/FastAPI)  →  dubbing/ pipeline
```

Rules, mirroring `MamboRambo/docs/ARCHITECTURE.md`:

* **The UI never imports the pipeline.** It only speaks HTTP to the server.
* **The server never re-implements pipeline logic.** It calls `dubbing.*` functions and
  owns nothing but jobs, HTTP, and file serving.
* Everything in `dubbing/` stays runnable headless from `python -m dubbing`. The app is a
  second front end, never a fork.

A Tauri desktop shell may wrap this later. Keeping the server a standalone process with a
stdout ready-signal is what makes that a packaging change rather than a rewrite.

### Why not Tauri/Rust now

MamboRambo puts its engine in a Rust sidecar because its engine is Rust. Ours is Python
(MLX, torch, faster-whisper), so the sidecar is Python. Rust is not installed on the dev
machine, and a Tauri shell adds nothing until we package for other people.

## Process contract (server)

`uv run python -m dubbing_app.server --host 127.0.0.1 --port 0 [--outputs DIR]`

1. Bind the port (0 = OS-assigned).
2. Print **exactly one line** of JSON to stdout, then flush:
   `{"status":"ready","port":54321,"version":"0.1.0"}`
3. Serve. Logs go to stderr, never stdout — stdout is the handshake channel.
4. Watchdog: poll `os.getppid()` every second; exit(0) when it changes, so a crashed
   parent never leaves a pipeline running. Skip when the parent is already init (pid 1).

This is deliberately the same handshake `translator/worker.py` already uses, and what a
Tauri shell would parse.

## The one-job rule

**At most one pipeline job runs at a time, process-wide.** Not a design preference: the
models are sequential and never co-resident (AGENTS.md, device notes) and the dev machine
has 26 GB of unified memory. A second concurrent job would swap the machine to death.

Enqueue further requests; expose the queue. Cheap edits that touch no model (`keep` flips,
text edits, speaker relabels) are *not* jobs and must stay responsive while a job runs.

## Data model

A **project** is one run directory under `outputs/`. Its `manifest.json` is the source of
truth. The server holds no database; it reads and writes the manifest through
`dubbing.manifest`, which means every invariant in AGENTS.md still applies — in particular
`SEGMENT_KEYS` is a whitelist and **any key not listed there is silently dropped on save**.

### Stable segment identity — new

`seg["id"]` is positional and is renumbered whenever segmentation changes, so the UI cannot
use it as a key across a re-run. `uid` is in `SEGMENT_KEYS`: an opaque string minted once
when a segment is created and preserved across edits. Every API below keys on `uid`.
`id` stays as-is for ordering and for existing tests.

**Shipped** (`manifest.mint_uid` / `ensure_uids`): the uid is `"s" + sha1(start|end|text)[:11]`,
minted at the end of the segments stage once every split, splice and fill has settled, and
carried unchanged through every later edit. Content-derived rather than random so that a run
directory written before `uid` existed is back-filled *on load* with exactly the uid a fresh
re-segmentation would mint — identity survives both the migration and a re-run that
reproduces the segment. `manifest.save` back-fills too, so no segment ever reaches the UI
without one. A copied segment dict (`splice_foreign_spans` makes trimmed pieces that way) is
re-minted from its own span, so uids stay unique.

### New per-segment keys (all must be added to `SEGMENT_KEYS`)

| key | type | meaning |
|---|---|---|
| `uid` | str | stable identity across re-segmentation |
| `src_lang` | str? | per-segment source-language override; falls back to `lang`, then `m["source"]["src_lang"]` |
| `tgt_lang` | str? | per-segment target override — "tag this section and translate it to something else" |
| `tts_opts` | dict? | per-segment synthesis overrides, see below |
| `locked` | dict? | `{field: true}` — fields the user edited by hand that a re-run must not overwrite |

`locked` is what makes the editor safe: a re-translate of the whole run must skip a line
the user corrected. Honoured, as shipped, in:

* `manifest.reset_stage` — the main one. A stage reset (`--force`, a changed fingerprint,
  `rebuild`) drops `text_en` / `tts` / `place` for every segment; a locked field survives,
  and a locked `keep` is not re-decided.
* `translate.run` — `needs_translation(seg)` is False for a locked line, in the main loop,
  the keep-subtitle pass and (crucially) the run-global revision pass, which otherwise
  rewrites every dubbed line whenever anything at all was retranslated.
* `tts` — `clear_failed_keeps` leaves a locked segment's verdict and clip alone. `pending`
  needs no lock check: nothing but the user's own edit deletes a locked clip record, so a
  locked clip is still on disk and already counts as done. A lock whose clip has gone is
  unhonorable and never-silent wins — that segment is synthesized again.
* `timeline.run` — the shortening round skips a segment with a locked `text_en` or `tts`
  instead of re-translating and re-voicing it. It drifts; `place` still asserts non-overlap.

Lock fields (`manifest.LOCK_FIELDS`): `text`, `text_en`, `tts`, `place`, `keep`, `speaker`,
`bounds`. The setters below set them; a targeted `retranslate`/`resynthesize` of specific
uids clears the one it replaces.

## Edit API — `dubbing/edit.py` (new module)

The server calls only these; it never pokes the manifest directly. Every function takes a
loaded manifest `m` and the run dir, mutates `m` in place, and leaves saving to the caller.
Model-loading functions are the slow ones and are the only ones that need a job slot.

```python
find(m, uid) -> dict | None                       # segment by stable uid, or None
index_of(m, uid) -> int                           # its position; raises SegmentNotFound

# --- no models, instant -------------------------------------------------
set_text(m, uid, *, text=None, text_en=None, lock=True) -> dict
set_keep(m, uid, keep: bool, reason: str = "manual") -> dict
set_speaker(m, uid, speaker: str) -> dict
set_bounds(m, uid, start: float, end: float) -> dict   # raises on overlap with neighbours
set_langs(m, uid, *, src_lang=None, tgt_lang=None) -> dict
set_tts_opts(m, uid, **opts) -> dict              # None removes an option
split(m, uid, at: float) -> tuple[str, str]       # -> two new uids
merge(m, uid_a, uid_b) -> str                     # adjacent, same speaker -> new uid
invalidate(m, uid, *, stages: set[str]) -> set[str]    # {"translate","tts","timeline"}
enrich(m, workdir, uids=None) -> list[dict]       # segments + the clip's verify verdict

# --- models, needs the job slot -----------------------------------------
retranslate(m, workdir, uids, *, progress=None, register="narration",
            genre="documentary", respect_locked=False) -> dict[str, str]
resynthesize(m, workdir, uids, *, progress=None, device=None, model=None)
            -> dict[str, dict]
rebuild(m, workdir, *, from_stage: str, progress=None, save=None, **overrides) -> list[str]
```

Everything invalid raises `edit.EditError` (`SegmentNotFound` is a subclass) — map those to
`invalid_request` / `not_found`. `progress(fraction: float, message: str)`.

Rules the setters all follow, so the server does not have to think about them:

* **A hand edit locks the field it wrote**, so no rerun overwrites it. `set_text(lock=False)`
  is the escape hatch for a programmatic edit a rerun may still improve on.
* **Changing what is spoken invalidates what was built from it**, downstream included:
  a new line drops the clip and the placement, a new speaker or span drops the clip. Where
  the edit provably contradicts an approved artifact (a new line, a new voice, a new span,
  new `tts_opts`) the setter clears the `tts` lock rather than keeping a clip that says
  something else.
* **`split`/`merge` mint new uids.** The halves of a split are not the segment the old uid
  named; a UI holding it must be told, not shown different audio. Every word survives
  (`split` cuts at the word boundary nearest the time and refuses a one-word segment),
  the list stays sorted and non-overlapping, and `id` is renumbered.
* **Targeted beats locked, global does not.** `retranslate(uids)` / `resynthesize(uids)` is
  the user pointing at *this* segment, so it replaces a locked value and leaves it unlocked
  (machine work again). `respect_locked=True` for a bulk redo; `rebuild` always respects.

Deviations from the signatures above as first drafted, all additive: optional keyword
arguments (`lock`, `register`/`genre`, `respect_locked`, `device`/`model`, `save`,
`**overrides`) so the server can pass a project's settings without the module inventing
them; setters return the segment; `invalidate` returns the fields it dropped and expands
`stages` downstream itself (dropping a translation while keeping the clip made from it is
never right). `rebuild` accepts only `translate`/`tts`/`timeline`/`mix`/`report` — anything
earlier needs the source media and belongs to a real `python -m dubbing` run — and re-marks
each stage with the fingerprint `cli.stage_params` would compute, so a later CLI run sees
the work as done. `enrich` is new: it keeps knowledge of the clip-cache layout inside
`dubbing/`, where `GET /segments` needs the verification verdict.

`rebuild` reads the run's settings from `m["source"]` (`src_lang`, `tgt_lang`,
`duration_limit`, `context`, and optionally `register`, `genre`, `tts_model`, `transcript`,
`dub_foreign`, `device`), falling back to the CLI defaults. **The server should write
`genre`/`register`/`tts_model` into `m["source"]` when it creates a project** — top-level
`source` is not whitelisted, so anything may live there — or pass them as `**overrides`.

Implementation notes, from the pipeline audit:

* Per-segment redo already works by *deletion*: `translate.run` only fills segments whose
  `text_en` is empty (translate.py:1351) and `tts.run` only synthesizes segments with no
  usable clip (tts.py:928). `invalidate` deletes; the stage functions do the rest.
* `translate.run`'s revision pass is **run-global** — it rewrites every dubbed line whenever
  anything is retranslated. For single-segment edits, call `translate.generate` directly
  (see the recipe in the audit) rather than `translate.run`, or the user's other corrections
  get overwritten. Respect `locked`.
* `mix` is all-or-nothing with a full libx264 re-encode, so it is the interactive
  bottleneck. Treat "render preview" as an explicit user action, not an autosave.
* `--force <stage>` did **not** invalidate downstream stages (the fingerprint is unchanged
  and the downstream `outputs` lists are empty, so `stage_done` returned True). Fixed:
  `manifest.clear_downstream` drops and resets every later stage, `cli.apply_force` calls it,
  and `rebuild` does the same for the range it re-runs.

## HTTP API

JSON in, JSON out. Errors are uniform, after MamboRambo's `write_error`:

```json
{"error": {"code": "invalid_request" | "not_found" | "busy" | "internal_error",
           "message": "human readable"}}
```

| method | path | purpose |
|---|---|---|
| GET | `/health` | `{"status":"ok","version":...}` |
| GET | `/api/projects` | list run dirs: name, title, langs, duration, stage state, mtime |
| POST | `/api/projects` | `{source, src_lang, tgt_lang, duration?, name?, context?, genre?, register?}` → create dir, enqueue a full run |
| GET | `/api/projects/{name}` | manifest + report + derived stage status |
| GET | `/api/projects/{name}/segments` | the segment list, enriched (see below) |
| PATCH | `/api/projects/{name}/segments/{uid}` | any no-model edit; body mirrors the setters |
| POST | `/api/projects/{name}/segments/{uid}/split` `/merge` | structural edits |
| POST | `/api/projects/{name}/retranslate` | `{uids:[...]}` → job |
| POST | `/api/projects/{name}/resynthesize` | `{uids:[...]}` → job |
| POST | `/api/projects/{name}/render` | re-run timeline → mix → report → job |
| GET | `/api/projects/{name}/events` | **NDJSON progress stream** (see below) |
| GET | `/api/jobs` , `/api/jobs/{id}` , DELETE `/api/jobs/{id}` | queue state, cancel |
| GET | `/media/{name}/{path}` | range-capable file serving from the run dir (wav/mp4/srt/json) |

`GET /segments` returns each segment plus fields the UI needs and the manifest does not
store: absolute media URLs for `place.clip` and `tts.clip`, the verification verdict from
`clips/<hash>.json` (`heard`, `overlap` — the ready-made "did the voice say the right
thing" signal), and the segment's source-audio window as a URL.

### Progress: NDJSON, one JSON object per line

`GET /api/projects/{name}/events` streams `application/x-ndjson` and stays open:

```json
{"type":"stage","stage":"tts","status":"running","progress":0.42,"message":"clip 31/74"}
{"type":"segment","uid":"...","field":"tts","status":"done"}
{"type":"job","id":"...","status":"queued|running|done|failed","error":null}
{"type":"log","level":"info","message":"..."}
```

Chosen over MamboRambo's binary framing because our payloads are metadata, not WAV blobs;
kept from it: **errors travel as a frame**, so a mid-run failure surfaces without needing a
non-200 status on an already-streaming response. Send a heartbeat every 15 s so proxies and
sleeping laptops do not silently drop the stream.

## UI contract

Vite + React + TypeScript, pnpm, in `app/ui/`. No state library; a project store in
context, following MamboRambo's deliberately plain approach.

Two screens:

1. **Import** — file picker or URL, source/target language, duration cap, genre/register,
   optional context note. Starts a run and goes to the editor with live progress.
2. **Editor** — video player top; under it a **timeline** of segments on a time axis
   (colour-coded: dubbed / kept / failed / unclaimed), and a **segment inspector** for the
   selected segment: source text, target text, speaker, keep toggle, language tags, TTS
   options, the verification transcript, buttons to re-translate and re-voice just this
   segment, and A/B playback of original vs dubbed audio.

The editor must be usable while a job runs: no-model edits stay live, model actions queue.

Design guidance, not contract: lean on `dataviz`-style restraint — this is a work tool, so
information density over decoration, and never encode meaning in colour alone (keep/dub
state also carries an icon and a label).

## Workstreams and branches

| branch | owns |
|---|---|
| `feat/segment-ops` | `dubbing/edit.py`, `uid`, new `SEGMENT_KEYS`, `locked`, downstream-invalidation fix |
| `fix/english-muted` | target-language speech that ends up neither kept nor dubbed |
| `feat/tts-controls` | per-segment `tts_opts` (seed, greedy, reference override, style) |
| `app/server` | `dubbing_app/` FastAPI sidecar |
| `app/ui` | `app/ui/` React editor |
| `fix/tail-and-wall` | media-end wall in `timeline.place`, mux tail pad in `mix.mux` |

Each branches from `app`, stays in its own worktree, and merges back into `app` when green.
`app` merges to `main` when the whole thing hangs together.

## Non-negotiables

Everything in AGENTS.md still holds — never silent, never truncated or overlapping,
one TTS call per segment, no per-video content, the manifest stays small. Two app-specific
additions:

* **The user's edits outrank the pipeline.** A `locked` field is never overwritten by a
  re-run, and a manual `keep` is never re-decided by `mark_keep`.
* **Every destructive action is undoable or confirmed.** Re-running a stage discards
  generated work; say so before doing it.
