# MamboDubb — app architecture and contracts

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

`uv run python -m dubbing_app.server --host 127.0.0.1 --port 0 [--outputs DIR] [--ui-dir DIR] [--exit-on-stdin-close]`

1. Bind the port (0 = OS-assigned).
2. Print **exactly one line** of JSON to stdout, then flush:
   `{"status":"ready","port":54321,"version":"0.1.0"}`
3. Serve. Logs go to stderr, never stdout — stdout is the handshake channel.
4. Watchdog: poll `os.getppid()` every second; exit(0) when it changes, so a crashed
   parent never leaves a pipeline running. Skip when the parent is already init (pid 1).
5. `--exit-on-stdin-close` (the desktop shell always passes it, and pipes stdin rather
   than nulling it): exit(0) on stdin EOF. The ppid watchdog cannot see through the
   `uv run` wrapper — a wrapper that outlives the shell keeps the server's parent chain
   "alive" and leaks a listening server per app launch; the closed pipe cannot lie.

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

### `tts_opts` — per-segment synthesis controls

Defined and validated in `dubbing/ttsopts.py` (`parse` raises `ValueError`; `merge` applies
a partial patch and strips defaults, which is what `set_tts_opts` should call). Every key is
optional; absent or `{}` means "the pipeline's own choices" and reproduces a default run
byte for byte.

| key | type | default | effect |
|---|---|---|---|
| `seed` | int 0..2³²-1 | derived | replaces the text-derived seed; retry *N* uses `seed + 1000N`. Bump it to re-roll a take. |
| `greedy` | bool | `false` | deterministic decode on every attempt, not just the last of `MAX_TRIES` |
| `ref` | str | auto | wav under the run dir to clone from; also disables the canonical-reference escalation |
| `ref_text` | str | — | transcript of `ref`, switching the clone to ICL mode. **Requires `ref`.** |
| `model` | `"1.7b"`/`"0.6b"` | run's `--tts-model` | checkpoint for this segment (swaps the loaded model — one line, not a hundred) |
| `speed` | float 0.5..2.0 | `1.0` | ffmpeg `atempo` baked in before verification |
| `temperature` | float 0<t≤2 | `0.55` | sampler; rejected together with `greedy` |
| `top_p` | float 0<p≤1 | `0.85` | ″ |
| `top_k` | int 1..1000 | `30` | ″ |
| `repetition_penalty` | float 0.8..2.0 | `1.08` | |
| `max_new_tokens` | int 32..4096 | derived from the text | |
| `keep_pauses` | bool | `false` | leave long internal silences instead of compressing to `PAUSE_KEEP` |

Rules the server and UI can rely on:

* **Unknown keys and out-of-range values raise**, in `parse` and again in `tts._plan`. There
  is no silent coercion — validate on `PATCH` and surface the message.
* **Every audio-affecting option is in the clip cache key**, and the tts record carries an
  option fingerprint (`seg["tts"]["opts"]`), so `tts.run` re-synthesizes a segment whose
  options changed under an otherwise-usable clip. Editing `tts_opts` alone is enough; no
  `invalidate` call is required (one is still harmless).
* **Natural-language style instructions are not offered, because they cannot work here.**
  `instruct` exists only on `generate_voice_design` / `generate_custom_voice`, both gated on
  a different `tts_model_type` and neither taking reference audio — they design a voice from
  the description instead of cloning the speaker. `ref_text` (ICL) is the only prosody-
  carrying knob the Base checkpoint has, and it means "sound like this take", not "sound
  angry". Do not add a style textbox to the inspector.

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
| GET | `/health` | `{"status":"ok","version":...,"commit":"a1b2c3d"|null,"outputs":...,"busy":...,"queued":...}` |
| GET | `/api/setup` | first-run environment report (see **Desktop packaging**) |
| POST | `/api/setup/install` | `{id}` → run the hardcoded argv for that check (`ffmpeg`, `sox`); 400 otherwise, 409 while one runs |
| GET | `/api/setup/install` | `{running, id, ok, error, tail:[…], check}` — poll it; `check` is a fresh probe once the process exits |
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

### The report says what it is a report about

`report.run` stamps `report["manifest"]` — `manifest.content_fingerprint(m)`, a hash of the
segment list it counted. `GET /api/projects/{name}` still returns the file verbatim and
adds **`report.stale`**: `true` when that stamp is missing or no longer matches the
manifest on disk. A report is derived from decisions, and an edit changes no stage
parameter, so nothing else on disk could tell a current report from one the manifest moved
past hours ago. The data is served either way — the numbers were true of the run they
described — and the UI decides whether to caption them "as of the last render". A report
written before the stamp existed is `stale: true`: it cannot prove otherwise.

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

## Desktop packaging

Installed, there is no `pnpm dev` and no second origin: the shell starts **one** process —
this server — and points a webview at `http://127.0.0.1:<port>` from the ready line. So the
server serves the app as well as the API.

### Single-process serving of the built UI

`create_app(outputs, ..., ui_dir=None)` / `--ui-dir DIR`:

* `None` (default) → `<repo>/app/ui/dist`. An **empty string** disables UI serving; the
  process is then API-only, which is what `pnpm dev` wants.
* Serving only happens when the directory holds an `index.html`. Without one — a checkout
  that never ran `pnpm build` — behaviour is exactly as before: `GET /` is a 404 envelope.
* `dist/assets/` is a real static mount (its own traversal guard, ETag/304 for free).
* A catch-all `GET /{path}` is registered **last**: an existing file under `ui_dir` is
  served, anything else falls back to `index.html`, because `/editor/<name>` exists only in
  the browser router and a hard reload on it must still get the app.
* `health`, `api` and `media` first segments are refused by the catch-all **explicitly**,
  not merely by route order. Order alone still hands `index.html` to a *misspelled* API
  path, which turns "no such project" into HTML that `fetch().json()` cannot parse.
* Paths follow `media.resolve`'s discipline — resolve, then compare against the root — so
  `../`, an absolute path and a symlink planted in `dist/` are all 404, never a file leak.

The stdout ready line is unchanged (`{"status":"ready","port":...,"version":...}`); the
shell needs nothing new to parse.

### Version stamp

`/health` carries `commit`: `git rev-parse --short HEAD`, run once at startup and cached
(`/health` is polled — a subprocess per poll would be absurd), `null` outside a repository.
`DUBBING_STUDIO_COMMIT` overrides it, which is the packaged path: a build with no `.git`
bakes the hash in at build time, the way the translator worker's ready signal does.

### Setup / first-run API

`GET /api/setup` answers "can this machine actually run the pipeline?" — the question no
error inside the editor answers legibly, since a missing `ffmpeg` surfaces as a stage that
dies halfway and an absent model directory silently becomes a multi-gigabyte download.

```json
{"ok": true,
 "checks": [{"id": "model.translate", "label": "Translation model (Gemma 4 12B)",
             "ok": true, "detail": "9.7 GB in /…/models/gemma-4-12B-it-6bit",
             "required": true, "installable": false,
             "path": "/…", "bytes": 10424182784}]}
```

* Ids: `ffmpeg`, `sox`, `hf_token`, `model.translate`, `model.tts.<key>` (one per
  `tts.TTS_MODELS`), `model.asr.he`, `model.asr.src`, `model.asr.en`, `model.asr.tgt`,
  `model.lid`, `model.demucs`, `disk`. `label` and `detail` are for display; `id` is stable.
* **`ok` is the conjunction of the `required` checks only.** Required = a default run cannot
  work: ffmpeg, sox, the translator, the default TTS checkpoint, the English ASR that
  verifies every clip. Everything else is informational — Demucs and Pyannote download
  themselves, a missing `HF_TOKEN` degrades diarization to one speaker, the other ASR/TTS
  models matter only for particular language pairs — and must not gate first run.
* **The token is reported as present or absent, never echoed.** Env first
  (`HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN`), then a `HF_TOKEN=` line in `.env`.
* **No model is loaded and no import is heavy**: `shutil.which`, `os.environ`, `stat`,
  `shutil.disk_usage`. Milliseconds, safe to poll, and it must stay that way.
* **`installable` is the server's answer to "can the app fix this?"** — true for exactly
  the keys of `install.INSTALLERS` (`ffmpeg`, `sox` → `brew install …`), so the UI puts an
  Install button on a failing row without keeping its own copy of the whitelist. The id maps
  to a **hardcoded argv**; nothing from the request body is ever executed, and a model id is
  refused with a 400 that points at the download command already in its `detail`. One
  install at a time, tracked in-process — not in `JobQueue`, which is for work that loads
  models and would make a `brew` wait behind a render.
* **Paths come from the pipeline's own constants** (`translate.MODEL_PATH`,
  `tts.TTS_MODELS`, `transcript.WHISPER_MODEL` / `SRC_ASR_MODEL` / `EN_ASR_MODEL` /
  `TARGET_ASR_MODEL` / `LID_MODEL`), never restated. A hardcoded copy would drift and then
  report a green tick for a directory the pipeline does not open.

## UI contract

Vite + React + TypeScript, pnpm, in `app/ui/`. No state library; a project store in
context, following MamboRambo's deliberately plain approach.

Two screens:

1. **Import** — file picker or URL, source/target language, duration cap, genre/register,
   optional context note. Starts a run and goes to the editor with live progress.
2. **Editor** — script-first, in three regions. A 44px **header** (back to runs, title,
   `he→en`, Render preview, run health, keyboard help, theme). A **main row** split
   between the **script** (~58%, and it is the pane that grows) and a fixed **viewer
   column** (~42%: video, transport, then the selection panel). A 112px **timeline strip**
   full-width along the bottom.

   The script is the work surface, because reviewing a dub means comparing a translation
   with its original and that is a comparison between two pieces of text. Every row stacks
   them — original above (muted), translation below (ink) — both always visible, both
   complete, never ellipsis-truncated, for kept segments too. The row container is
   `dir="ltr"` (timecodes and buttons are chrome) while each text line is `dir="auto"` +
   `.auto-dir`, so a Hebrew original sits right-to-left directly above its left-to-right
   translation without dragging the layout around. Clicking the translation edits it in
   place; the draft is seeded on mount and never re-seeded, so a background refetch cannot
   clobber what is being typed, and an empty commit is refused rather than sent (the server
   400s on it). The row under the playhead washes and auto-scrolls to centre.

   The **selection panel** holds what is true *about* a line and no copy of the line
   itself: the dub/keep verdict, the two model actions with their prices, then four shut
   `Disclosure` shelves — Voice & speaker, Verification, Timing & languages, Advanced
   (locks, and the `PATCH {locked:{}}` that releases them). The **timeline** stays a map:
   two lanes (source vs. output — the drift picture), unclaimed hatch, click to seek or
   select, drag to scrub, zoom and split at its right edge. No dragging clips and no trim
   handles, ever: `timeline.place()` is the sole authority on where audio goes.

   Search plus four filter chips (All / Failed / Kept / Edited) make 200 rows navigable,
   and a filtered set is fixed in **one** job — `/retranslate` and `/resynthesize` both
   take `{uids:[…]}`.

   One keyboard listener with one guard: `space` play/pause, `↑/↓` rows, `↵` edit, `esc`
   leave, `a`/`b` original/dub, `k` keep, `s` split, `+/−` zoom, `⌘F` search. A/B shares a
   single `<audio>` element app-wide (`lib/clipAudio.ts`) so only one clip can ever sound.

The editor must be usable while a job runs: no-model edits stay live, model actions queue.

`lib/api.ts` is the only seam that knows a wire shape. Segment audio is `seg.media` —
`{play, tts, source, source_window}`, exactly as `Projects.enrich` writes it — and
`lib/fixtures.ts` implements *that* shape, not one of its own. A fixture that invents field
names is a second, divergent server that hides the first one's bugs.

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
