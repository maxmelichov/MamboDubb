# MamboDubb app architecture and contracts

The editor app: import a video (file or URL), run the dubbing pipeline, then review and
correct the result segment by segment a much simpler Premiere Pro whose clips are the
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
3. Serve. Logs go to stderr, never stdout stdout is the handshake channel.
4. Watchdog: poll `os.getppid()` every second; exit(0) when it changes, so a crashed
   parent never leaves a pipeline running. Skip when the parent is already init (pid 1).
5. `--exit-on-stdin-close` (the desktop shell always passes it, and pipes stdin rather
   than nulling it): exit(0) on stdin EOF. The ppid watchdog cannot see through the
   `uv run` wrapper a wrapper that outlives the shell keeps the server's parent chain
   "alive" and leaks a listening server per app launch; the closed pipe cannot lie.

This is deliberately the same handshake `translator/worker.py` already uses, and what a
Tauri shell would parse.

## The one-job rule

**At most one pipeline job runs at a time, process-wide.** Not a design preference: the
models are sequential and never co-resident (a hard rule of the pipeline) and the dev machine
has 26 GB of unified memory. A second concurrent job would swap the machine to death.

Enqueue further requests; expose the queue. Cheap edits that touch no model (`keep` flips,
text edits, speaker relabels) are *not* jobs and must stay responsive while a job runs.

## Data model

A **project** is one run directory under `outputs/`. Its `manifest.json` is the source of
truth. The server holds no database; it reads and writes the manifest through
`dubbing.manifest`, which means every pipeline invariant still applies in particular
`SEGMENT_KEYS` is a whitelist and **any key not listed there is silently dropped on save**.

### Stable segment identity new

`seg["id"]` is positional and is renumbered whenever segmentation changes, so the UI cannot
use it as a key across a re-run. `uid` is in `SEGMENT_KEYS`: an opaque string minted once
when a segment is created and preserved across edits. Every API below keys on `uid`.
`id` stays as-is for ordering and for existing tests.

**Shipped** (`manifest.mint_uid` / `ensure_uids`): the uid is `"s" + sha1(start|end|text)[:11]`,
minted at the end of the segments stage once every split, splice and fill has settled, and
carried unchanged through every later edit. Content-derived rather than random so that a run
directory written before `uid` existed is back-filled *on load* with exactly the uid a fresh
re-segmentation would mint identity survives both the migration and a re-run that
reproduces the segment. `manifest.save` back-fills too, so no segment ever reaches the UI
without one. A copied segment dict (`splice_foreign_spans` makes trimmed pieces that way) is
re-minted from its own span, so uids stay unique.

### New per-segment keys (all must be added to `SEGMENT_KEYS`)

| key | type | meaning |
|---|---|---|
| `uid` | str | stable identity across re-segmentation |
| `src_lang` | str? | per-segment source-language override; falls back to `lang`, then `m["source"]["src_lang"]` |
| `tgt_lang` | str? | per-segment target override "tag this section and translate it to something else" |
| `tts_opts` | dict? | per-segment synthesis overrides, see below |
| `locked` | dict? | `{field: true}` fields the user edited by hand that a re-run must not overwrite |

`locked` is what makes the editor safe: a re-translate of the whole run must skip a line
the user corrected. Honoured, as shipped, in:

* `manifest.reset_stage` the main one. A stage reset (`--force`, a changed fingerprint,
  `rebuild`) drops `text_en` / `tts` / `place` for every segment; a locked field survives,
  and a locked `keep` is not re-decided. `place` is dropped unconditionally placement is
  all-or-nothing and is not lockable (below).
* `translate.run` `needs_translation(seg)` is False for a locked line, in the main loop,
  the keep-subtitle pass and (crucially) the run-global revision pass, which otherwise
  rewrites every dubbed line whenever anything at all was retranslated.
* `tts` `clear_failed_keeps` leaves a locked segment's verdict and clip alone, honouring
  `locked.keep` through the same predicate `reset_stage` and `edit.invalidate` use
  (`manifest.undo_pipeline_keep`) it used to guard only `locked.tts`, so a user-locked
  keep was re-decided on every run. `pending` re-queues a clip whose *text* has moved
  under it, unless it is locked: that is a conflict, reported (`report.json:
  stale_locked_clips`), never silently kept and never silently regenerated. A lock whose
  clip has gone is unhonorable and never-silent wins that segment is synthesized again.
* `timeline.run` the shortening round skips a segment with a locked `text_en` or `tts`
  instead of re-translating and re-voicing it. It drifts; `place` still asserts non-overlap.

Lock fields (`manifest.LOCK_FIELDS`): `text`, `text_en`, `tts`, `keep`, `speaker`. The
setters below set them; a targeted `retranslate`/`resynthesize` of specific uids clears the
one it replaces.

`place` and `bounds` are **not** lockable, and `set_locked` rejects them. Nothing could
honour either: `timeline.run` lays the whole run out in one forward pass and rewrites every
`place` (which is what makes non-overlap provable), and re-segmentation rebuilds every span
from the words, carrying only `passthrough` forward by time. Accepting a lock the next run
overwrites is state that lies about the user's edit.

### `tts_opts` per-segment synthesis controls

Defined and validated in `dubbing/ttsopts.py` (`parse` raises `ValueError`; `merge` applies
a partial patch and strips defaults, which is what `set_tts_opts` should call). Every key is
optional; absent or `{}` means "the pipeline's own choices" and reproduces a default run
byte for byte.

| key | type | default | effect |
|---|---|---|---|
| `seed` | int 0..2³²-1 | derived | replaces the text-derived seed; retry *N* uses `seed + 1000N`. Bump it to re-roll a take. |
| `greedy` | bool | `false` | deterministic decode on every attempt, not just the last of `MAX_TRIES` |
| `ref` | str | auto | wav under the run dir to clone from; also disables the canonical-reference escalation |
| `ref_text` | str | | transcript of `ref`, switching the clone to ICL mode. **Requires `ref`.** |
| `model` | `"1.7b"`/`"0.6b"` | run's `--tts-model` | checkpoint for this segment (swaps the loaded model one line, not a hundred) |
| `speed` | float 0.5..2.0 | `1.0` | ffmpeg `atempo` baked in before verification |
| `temperature` | float 0<t≤2 | `0.55` | sampler; rejected together with `greedy` |
| `top_p` | float 0<p≤1 | `0.85` | ″ |
| `top_k` | int 1..1000 | `30` | ″ |
| `repetition_penalty` | float 0.8..2.0 | `1.08` | |
| `max_new_tokens` | int 32..4096 | derived from the text | |
| `keep_pauses` | bool | `false` | leave long internal silences instead of compressing to `PAUSE_KEEP` |

Rules the server and UI can rely on:

* **Unknown keys and out-of-range values raise**, in `parse` and again in `tts._plan`. There
  is no silent coercion validate on `PATCH` and surface the message.
* **Every audio-affecting option is in the clip cache key**, and the tts record carries an
  option fingerprint (`seg["tts"]["opts"]`), so `tts.run` re-synthesizes a segment whose
  options changed under an otherwise-usable clip. Editing `tts_opts` alone is enough; no
  `invalidate` call is required (one is still harmless).
* **Natural-language style instructions are not offered, because they cannot work here.**
  `instruct` exists only on `generate_voice_design` / `generate_custom_voice`, both gated on
  a different `tts_model_type` and neither taking reference audio they design a voice from
  the description instead of cloning the speaker. `ref_text` (ICL) is the only prosody-
  carrying knob the Base checkpoint has, and it means "sound like this take", not "sound
  angry". Do not add a style textbox to the inspector.

## Edit API `dubbing/edit.py` (new module)

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
add(m, start, end, *, text, speaker=None) -> str  # claim a free span -> new uid
remove(m, uid) -> dict                            # take a segment out; returns it
invalidate(m, uid, *, stages: set[str]) -> set[str]    # {"translate","tts","timeline"}
enrich(m, workdir, uids=None) -> list[dict]       # segments + the clip's verify verdict

# --- models, needs the job slot -----------------------------------------
retranslate(m, workdir, uids, *, progress=None, register="narration",
            genre="documentary", respect_locked=False) -> dict[str, str]
resynthesize(m, workdir, uids, *, progress=None, device=None, model=None)
            -> dict[str, dict]
rebuild(m, workdir, *, from_stage: str, progress=None, save=None, **overrides) -> list[str]
```

Everything invalid raises `edit.EditError` (`SegmentNotFound` is a subclass) map those to
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
* **`add`/`remove` are the other two structural edits, and they are not each other's
  inverse of `set_keep`.** `add` claims an *uncovered* span: an overlap with a neighbour
  is refused, never clamped (`timeline.place` asserts non-overlap, so a silently moved
  span would be an edit the user did not make), the span must be at least
  `segments.MIN_SEG_SEC` and inside the media, and the text is required — it is what gets
  translated and spoken. It writes `locked.text` (the words are the user's, typed because
  the ASR had none) and `passthrough: false` (`keep.user_wants_dub`, so the first
  translation or synthesis failure leaves the line visibly unfinished instead of answering
  with the original audio the user was replacing), and inherits the nearest segment's
  speaker unless one is named. `remove` deletes the record outright — **it does not mute
  the span**: `mix` adds the original vocals stem at unity gain into every span no
  placement claims, so a removed span sounds exactly like a passage the pipeline never
  detected. That is the honest reading of "take it out of the dub" and the only one
  compatible with *never silent*; "play the original here, deliberately" is `set_keep` /
  `passthrough`, which keeps the segment, places it, and ducks the bed to zero under it.
* **What each structural edit reopens.** `split`, `merge` and `add` reopen from
  `translate` — each leaves a segment with nothing generated for it. `remove` reopens from
  `timeline`: no survivor's translation or clip was made from the segment that went, but
  the span is free now and placement is laid out in one forward pass. None of them reopens
  `segments`: re-running that stage rebuilds the list from the words and undoes the edit.
  Which is also the standing caveat — a structural edit is honoured from `translate`
  onward, and a genuine re-segmentation (a changed `--transcript`, a new duration cap)
  loses it, exactly as it has always lost a split. Only `passthrough` survives that, by
  time (`segments.carry_passthrough`).
* **Targeted beats locked, global does not.** `retranslate(uids)` / `resynthesize(uids)` is
  the user pointing at *this* segment, so it replaces a locked value and leaves it unlocked
  (machine work again). `respect_locked=True` for a bulk redo; `rebuild` always respects.

Deviations from the signatures above as first drafted, all additive: optional keyword
arguments (`lock`, `register`/`genre`, `respect_locked`, `device`/`model`, `save`,
`**overrides`) so the server can pass a project's settings without the module inventing
them; setters return the segment; `invalidate` returns the fields it dropped and expands
`stages` downstream itself (dropping a translation while keeping the clip made from it is
never right). `rebuild` accepts only `translate`/`tts`/`timeline`/`mix`/`report` anything
earlier needs the source media and belongs to a real `python -m dubbing` run and re-marks
each stage with the fingerprint `cli.stage_params` would compute, so a later CLI run sees
the work as done. `enrich` is new: it keeps knowledge of the clip-cache layout inside
`dubbing/`, where `GET /segments` needs the verification verdict.

`rebuild` reads the run's settings from `m["source"]` (`src_lang`, `tgt_lang`,
`duration_limit`, `context`, and optionally `register`, `genre`, `tts_model`, `transcript`,
`dub_foreign`, `device`), falling back to the CLI defaults. **The server should write
`genre`/`register`/`tts_model` into `m["source"]` when it creates a project** top-level
`source` is not whitelisted, so anything may live there or pass them as `**overrides`.

Implementation notes, from the pipeline audit:

* Per-segment redo already works by *deletion*: `translate.run` only fills segments whose
  `text_en` is empty (translate.py:1351) and `tts.run` only synthesizes segments with no
  usable clip (tts.py:928). `invalidate` deletes; the stage functions do the rest.
* `translate.run`'s revision pass is **run-global** it rewrites every dubbed line whenever
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
| GET | `/api/setup/install` | `{running, id, ok, error, tail:[…], check}` poll it; `check` is a fresh probe once the process exits |
| GET | `/api/projects` | list run dirs: name, title, langs, duration, stage state, mtime |
| POST | `/api/projects` | `{source, src_lang, tgt_lang, duration?, name?, context?, genre?, register?, transcript?, tts_model?, dub_foreign?, captions?}` → create dir, enqueue a full run |
| GET | `/api/projects/{name}` | manifest + report (+`stale`) + `render` (+`stale`/`changed`) + stage status + jobs |
| PATCH | `/api/projects/{name}` | `{context?, genre?, register?}` → change the run's options (see below); 409 while a job runs |
| POST | `/api/projects/{name}/run` | run it again which on a stopped run is a **resume** (see below); 409 while a job runs |
| GET | `/api/projects/{name}/segments` | the segment list, enriched (see below) |
| POST | `/api/projects/{name}/segments` | `{start, end, text, speaker?}` → claim an uncovered span; 201 `{uid, segment}` |
| PATCH | `/api/projects/{name}/segments/{uid}` | any no-model edit; body mirrors the setters |
| DELETE | `/api/projects/{name}/segments/{uid}` | take the segment out of the dub; `{uid, removed, segments}` |
| POST | `/api/projects/{name}/segments/{uid}/split` `/merge` | structural edits |
| POST | `/api/projects/{name}/retranslate` | `{uids:[...]}` → job |
| POST | `/api/projects/{name}/resynthesize` | `{uids:[...]}` → job |
| POST | `/api/projects/{name}/render` | re-run timeline → mix → report → job |
| GET | `/api/projects/{name}/events` | **NDJSON progress stream** (see below) |
| GET | `/api/jobs` , `/api/jobs/{id}` , DELETE `/api/jobs/{id}[?batch=1]` | queue state, cancel one job or the whole gesture |
| GET | `/media/{name}/{path}` | range-capable file serving from the run dir (wav/mp4/srt/json) |

`GET /segments` returns each segment plus fields the UI needs and the manifest does not
store: absolute media URLs for `place.clip` and `tts.clip`, the verification verdict from
`clips/<hash>.json` (`heard`, `overlap` the ready-made "did the voice say the right
thing" signal), and the segment's source-audio window as a URL.

Two options on `POST /api/projects` are deliberately **server-side only**, with no control
on the import screen: `captions` (a filesystem path to a caption file to use instead of the
fetched one a browser cannot produce a real path, and the desktop file picker offers
videos) and `tts_model` (one checkpoint is current; the field exists so an old manifest
still re-runs). The transcript *source* `auto | captions | asr` is a different thing
and is on the screen, because "the auto-captions for this video are mangled" is a judgement
only the person watching it can make.

### Running it again is how a run resumes

`POST /api/projects/{name}/run` enqueues the same `run` job that created the project, with
the payload read back off `m["source"]`. There is no separate resume machinery and there
must not be: every stage is skipped when its inputs and outputs are unchanged, so
re-running *is* resuming, and a second implementation of "where did this stop" would be the
fork this document rules out. The local-file probe that `POST /api/projects` does is not
repeated a run past `fetch` no longer needs its input, and refusing to resume because the
source file was moved is a refusal with nothing behind it. Refused with 409 while anything
is queued or running on the project.

Recovery needs the other half too, and it is already served: `GET /api/projects/{name}`
returns `jobs`, and the most recent failed one carries the `stage` it died in and the
`error` that killed it. That is the **only** copy the event stream deliberately replays
nothing terminal (see below) so a client that wants to say more than "stopped at fetch"
reads it there.

### Genre, register and context can change; the source and the languages cannot

`PATCH /api/projects/{name}` writes those three as **flat keys on `m["source"]`**, which is
the spelling `dubbing.edit._args` resolves first (`app_opts` is the app's own, older record
and loses to it). One spelling, so the change reaches every path that re-runs anything: a
per-line re-translate, a render, a resume. `dubbing_app.ops` resolves the pair the same way
reading only `app_opts` there meant an edited genre was honoured by a re-translate and
silently ignored by a full run.

Nothing is invalidated and no job is enqueued. All three are inputs to the *translator*, so
they take effect the next time translation runs; re-translating two hundred lines because a
dropdown moved would be a worse surprise than the wait. `context: ""` clears the note the
one field with a way to be removed. The source and the language pair are **not** patchable:
changing either invalidates `fetch` and every stage after it, which is a new project
wearing an old project's name.

### A job says which segments it is about

Every job dict (`/api/jobs`, the `jobs` list on a project, `POST /retranslate` and friends)
carries **`uids`** the segments the job was asked for, `[]` for a whole-run job. It is
what a UI needs to say "re-voicing 3 lines" and to mark exactly those rows busy; it was
declared on the client's `Job` type while only the fixtures ever produced it. Always a
list, never absent.

`uids` and `batch` ride on the **job frames** too, live and replayed, for the same reason
`kind` does: a client rebuilds "which rows are busy" and "what would Cancel stop" from the
frames alone, and a replayed job that omitted them would come back as a job about nothing —
rows stop pulsing on reload mid-render.

### Stage status four values, not two

`stages` (on `GET /api/projects/{name}`, in `summary`, and in the event prelude) is
`{stage: "done" | "pending" | "running" | "failed"}`. `done`/`pending` come from the
manifest's `stages` map, which is the only truth about what *finished*; the other two come
from the project's jobs, because an unfinished stage and one that was never reached look
identical in the manifest. The stage a running job is in is `running`; the stage the most
recent failed job died in is `failed`, unless a job has succeeded since. A UI's failure
treatment ("stopped at transcript") has nothing else to key on.

### The report says what it is a report about

`report.run` stamps `report["manifest"]` `manifest.content_fingerprint(m)`, a hash of the
segment list it counted. `GET /api/projects/{name}` still returns the file verbatim and
adds **`report.stale`**: `true` when that stamp is missing or no longer matches the
manifest on disk. A report is derived from decisions, and an edit changes no stage
parameter, so nothing else on disk could tell a current report from one the manifest moved
past hours ago. The data is served either way the numbers were true of the run they
described and the UI decides whether to caption them "as of the last render". A report
written before the stamp existed is `stale: true`: it cannot prove otherwise.

### The video says what it is a render of

The same question, asked about `preview.mp4` the artifact the user actually watches.
`mix.run` stamps a run-level `m["render"]` as it writes the file:

```json
{"at": 1700000000, "fp": "<content_fingerprint>", "segments": {"<uid>": "<digest>"}}
```

Run-level, not per-segment, so `manifest.save`'s `SEGMENT_KEYS` whitelist leaves it alone;
written inside `mix.run` rather than at its two call sites so the CLI and the app's
`rebuild` both get it from the one place that writes the video. `GET /api/projects/{name}`
turns it into **`render`**:

| field | meaning |
|---|---|
| `at` | epoch seconds of that render, or `null` when the run predates the stamp or was never mixed |
| `stale` | the fingerprint no longer matches the manifest. A missing stamp is `true`, on the same rule as `report.stale` |
| `changed` | how many lines differ from the ones that render was made of |

`changed` is exact, not an estimate: `manifest.segment_digests` hashes each segment's
decisions under its `uid`, and `manifest.digest_delta` counts the uids whose digest moved
(a line added or removed counts once). `id` is excluded from the per-segment digest where
`content_fingerprint` includes it `id` is positional and renumbered by every split, so
keying on it would report the whole tail after a split as changed. Everything else
whitelisted is in, including fields that never reach the audio: over-reporting costs a
re-render the user chose, under-reporting hides one they asked for.

`changed` is `0` when there is no stamp to compare against, so a UI must not phrase
"N lines changed" off a `render` whose `at` is `null` there, the honest answer is that
staleness is unknown, not that nothing changed.

### One gesture, one thing to cancel

"Dub these 27" is one decision that becomes two jobs a `retranslate` and the
`resynthesize` queued behind it. Cancelling only the first left the second to voice 27
lines whose translation had just been abandoned: 27 `tts_failed` keeps from one click.

Jobs from one gesture therefore share a **`batch`** id. The pair is enqueued by the client
(two routes, two POSTs), so the id is minted there and sent on both `POST /retranslate`
and `POST /resynthesize` accept an optional `batch` alongside `uids`, and `Job.to_dict`
exposes it. A lone re-voice from the inspector has `batch: null`.

`DELETE /api/jobs/{id}?batch=1` cancels every non-terminal job sharing that batch and
returns `{job, cancelled: [...]}`; without the flag it cancels exactly the one job and
`cancelled` is `null`. **Order is load-bearing**: `cancel_batch` cancels the *queued*
members first, while the running one still holds the single worker. Killing the running job
first would free the worker to start the very job the call is trying to stop the race is
the disaster itself, not a corner of it.

The other half of the guarantee is in the pipeline: `edit.resynthesize` leaves a line with
no translation *unvoiced* (`keep` stays `false`, no clip) and says so in its summary, rather
than recording a `tts_failed` keep. Unfinished is a state the editor shows and offers to
fix; a keep looks like a decision someone made.

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

#### The prelude catching a new reader up

Opening the stream replays a `log` frame, one `stage` frame per pipeline stage, and the
jobs that are still **queued or running**. Two rules the client can rely on:

* **`"replay": true`** is on every prelude frame and on nothing else. The nine stage frames
  are a snapshot of all nine at once, not a progression, so a client that pins its display
  to the last stage frame it saw must skip them and read the snapshot as a whole.
* **Terminal jobs are never replayed.** A `failed`/`cancelled` job resent on every
  reconnect resurrects an error bar the user dismissed an hour ago and the UI reconnects
  on every navigation and every wake from sleep. A finished job's outcome is in
  `GET /api/projects/{name}` (`jobs`), which is where a client that wants history looks.

### What the client derives from the queue, and no longer tracks itself

Two pieces of UI state used to be kept alongside `jobs` and drift from it. Both are now
derived, and the contract above is what makes that possible.

* **`pendingUids`** which lines have model work in flight is every `uid` named by a
  non-terminal `retranslate`/`resynthesize` job. It was a `busyUids` list primed when a
  request was sent and cleared by two paths that both fired far too early: any `segment`
  frame dropped that uid, and *any* job reaching a terminal state emptied the list,
  including a job about other lines. Measured lifetime was about 100 ms against a re-voice
  that runs for a minute a line, and a reload cleared it outright. Derived, it lasts as
  long as the work and survives a reconnect, because the jobs do. `run` and `render` name
  no lines: they rebuild everything, and marking every row busy says nothing.
* **`stage`** must ignore `replay: true` frames. Replayed stage frames are a snapshot of
  all nine stages, so a client that pins its display to the last one it saw reads the
  ninth which is how the editor came to announce "Running report · 100%" the instant a
  re-voice started, over a video minutes out of date. Use them for refetch decisions only.
  A client should also clear `stage` when a **new** job goes `running`: the previous job's
  final frame is not the new job's first one, and the gap between them is a model load.

## Desktop packaging

Installed, there is no `pnpm dev` and no second origin: the shell starts **one** process —
this server and points a webview at `http://127.0.0.1:<port>` from the ready line. So the
server serves the app as well as the API.

### Single-process serving of the built UI

`create_app(outputs, ..., ui_dir=None)` / `--ui-dir DIR`:

* `None` (default) → `<repo>/app/ui/dist`. An **empty string** disables UI serving; the
  process is then API-only, which is what `pnpm dev` wants.
* Serving only happens when the directory holds an `index.html`. Without one a checkout
  that never ran `pnpm build` behaviour is exactly as before: `GET /` is a 404 envelope.
* `dist/assets/` is a real static mount (its own traversal guard, ETag/304 for free).
* A catch-all `GET /{path}` is registered **last**: an existing file under `ui_dir` is
  served, anything else falls back to `index.html`, because `/editor/<name>` exists only in
  the browser router and a hard reload on it must still get the app.
* `health`, `api` and `media` first segments are refused by the catch-all **explicitly**,
  not merely by route order. Order alone still hands `index.html` to a *misspelled* API
  path, which turns "no such project" into HTML that `fetch().json()` cannot parse.
* Paths follow `media.resolve`'s discipline resolve, then compare against the root so
  `../`, an absolute path and a symlink planted in `dist/` are all 404, never a file leak.

The stdout ready line is unchanged (`{"status":"ready","port":...,"version":...}`); the
shell needs nothing new to parse.

### Version stamp

`/health` carries `commit`: `git rev-parse --short HEAD`, run once at startup and cached
(`/health` is polled a subprocess per poll would be absurd), `null` outside a repository.
`DUBBING_STUDIO_COMMIT` overrides it, which is the packaged path: a build with no `.git`
bakes the hash in at build time, the way the translator worker's ready signal does.

### Setup / first-run API

`GET /api/setup` answers "can this machine actually run the pipeline?" the question no
error inside the editor answers legibly, since a missing `ffmpeg` surfaces as a stage that
dies halfway and an absent model directory silently becomes a multi-gigabyte download.

```json
{"ok": true,
 "checks": [{"id": "model.translate", "label": "Translation model (Gemma 4 12B)",
             "ok": true, "detail": "9.7 GB in /…/models/gemma-4-12B-it-6bit",
             "severity": "blocking", "stage": "translate",
             "required": true, "installable": false,
             "path": "/…", "bytes": 10424182784}]}
```

* Ids: `ffmpeg`, `sox`, `uv`, `hf_token`, `model.translate`, `model.tts.<key>` (one per
  `tts.TTS_MODELS`), `model.asr.he`, `model.asr.src`, `model.asr.en`, `model.asr.tgt`,
  `model.lid`, `model.diarization`, `model.demucs`, `model.tts.he`, `model.g2p.he`, `disk`. `label` and `detail`
  are for display; `id` is stable.
* **`severity` is the grade, and `required` is derived from it.** A boolean has two values
  and the question has three, so everything that was not required was reported as one
  undifferentiated "informational" which put a gated `HF_TOKEN` (every speaker in the
  video collapsed into one) on the same row as a Korean checkpoint a Hebrew→English run
  never opens.
  * `blocking` the run **fails**: `ffmpeg`, `sox`, `uv`, the translator, the default TTS
    checkpoint, the English ASR that verifies every clip. `required: true`, and these rows
    also carry **`stage`**, the pipeline stage the absence kills, so a client can say
    "runs will fail at translate" instead of "runs will fail". `stage` may be `null` and a
    client must handle that: `uv` blocks (nothing here is installed or updated without it)
    but a running server spawns its job child with `sys.executable` and never shells out to
    it, so naming a stage for it would be a guess dressed as a fact.
  * `degrades` the run **works and is worse**: `model.lid` (foreign speech is never
    detected). It stops nothing and it is not nothing.
  * `optional` irrelevant until asked for: the per-language-pair models, the
    self-downloading caches (Demucs, `model.diarization`), free disk and `hf_token`.
    The token row was `degrades` for as long as diarization loaded the gated
    `pyannote/speaker-diarization-community-1`; the pipeline reads an ungated mirror of
    the same CC-BY-4.0 weights now (`segments.diarization_sources`), so a machine with no
    Hugging Face account tells speakers apart like any other and **readiness may never
    again depend on a credential**.
* **`ok` is the conjunction of the `required` checks only** unchanged, and now equal to
  "no blocking check fails". A client must not compute readiness as "every row passes":
  that is stricter than the server and makes the "ready, with things missing" state
  unreachable.
* **The token is reported as present or absent, never echoed.** It is optional, and the
  only thing it still buys is `DUB_DIARIZATION_HUB` pointed at the gated upstream repo.
  Env first
  (`HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN`), then a `HF_TOKEN=` line in `.env` whose
  **absolute** path the failing row names, backticked, because "put it in `.env`" is a
  scavenger hunt on a machine with three checkouts and only the server knows which one it
  reads. Backticked spans in `detail` are the parts meant to be typed; the UI sets them as
  code and offers each as one click to the clipboard.
* **No model is loaded and no import is heavy**: `shutil.which`, `os.environ`, `stat`,
  `shutil.disk_usage`. Milliseconds, safe to poll, and it must stay that way.
* **`installable` is the server's answer to "can the app fix this?"** true for exactly
  the keys of `install.INSTALLERS` (`ffmpeg`, `sox` → `brew install …`), so the UI puts an
  Install button on a failing row without keeping its own copy of the whitelist. The id maps
  to a **hardcoded argv**; nothing from the request body is ever executed, and a model id is
  refused with a 400 that points at the download command already in its `detail`. One
  install at a time, tracked in-process not in `JobQueue`, which is for work that loads
  models and would make a `brew` wait behind a render.
* **Paths come from the pipeline's own constants** (`translate.MODEL_PATH`,
  `tts.TTS_MODELS`, `transcript.WHISPER_MODEL` / `SRC_ASR_MODEL` / `EN_ASR_MODEL` /
  `TARGET_ASR_MODEL` / `LID_MODEL`), never restated. A hardcoded copy would drift and then
  report a green tick for a directory the pipeline does not open.

## UI contract

Vite + React + TypeScript, pnpm, in `app/ui/`. No state library; a project store in
context, following MamboRambo's deliberately plain approach.

Two screens:

1. **Import** file picker or URL, source/target language, duration cap, genre/register,
   transcript source, optional context note. Starts a run and goes to the editor with live
   progress. The rail ends with the sentence that says which of those are final: genre,
   register and context can be changed later (`PATCH /api/projects/{name}`); the source and
   the languages cannot.
2. **Editor** script-first, in three regions. A 44px **header** (back to runs, title,
   `he→en`, Render preview, run health, keyboard help, theme). A **main row** split
   between the **script** (~58%, and it is the pane that grows) and a fixed **viewer
   column** (~42%: video, transport, then the selection panel). A 112px **timeline strip**
   full-width along the bottom.

   The script is the work surface, because reviewing a dub means comparing a translation
   with its original and that is a comparison between two pieces of text. Every row stacks
   them original above (muted), translation below (ink) both always visible, both
   complete, never ellipsis-truncated, for kept segments too. The row container is
   `dir="ltr"` (timecodes and buttons are chrome) while each text line is `dir="auto"` +
   `.auto-dir`, so a Hebrew original sits right-to-left directly above its left-to-right
   translation without dragging the layout around. Clicking the translation edits it in
   place; the draft is seeded on mount and never re-seeded, so a background refetch cannot
   clobber what is being typed, and an empty commit is refused rather than sent (the server
   400s on it). The row under the playhead washes and auto-scrolls to centre.

   The **selection panel** holds what is true *about* a line and no copy of the line
   itself: the dub/keep verdict, the two model actions with their prices, then four shut
   `Disclosure` shelves Voice & speaker, Verification, Timing & languages (the bounds
   as two number inputs, split at the playhead, merge with either neighbour, and remove),
   Advanced (locks, and the `PATCH {locked:{}}` that releases them). Adding a segment is
   offered where the absence is visible the uncovered-audio list in the rail, whose
   spans are exactly "speech nobody claimed" not as a floating "new segment" button
   with no span to attach to. The **timeline** stays a map:
   two lanes (source vs. output the drift picture), unclaimed hatch, click to seek or
   select, drag to scrub, zoom and split at its right edge. No dragging clips and no trim
   handles, ever: `timeline.place()` is the sole authority on where audio goes.

   Search plus four filter chips (All / Failed / Kept / Edited) make 200 rows navigable,
   and a filtered set is fixed in **one** job `/retranslate` and `/resynthesize` both
   take `{uids:[…]}`.

   One keyboard listener with one guard: `space` play/pause, `↑/↓` rows, `↵` edit, `esc`
   leave, `a`/`b` original/dub, `k` keep, `s` split, `+/−` zoom, `⌘F` search. A/B shares a
   single `<audio>` element app-wide (`lib/clipAudio.ts`) so only one clip can ever sound.

The editor must be usable while a job runs: no-model edits stay live, model actions queue.

`lib/api.ts` is the only seam that knows a wire shape. Segment audio is `seg.media` —
`{play, tts, source, source_window}`, exactly as `Projects.enrich` writes it and
`lib/fixtures.ts` implements *that* shape, not one of its own. A fixture that invents field
names is a second, divergent server that hides the first one's bugs.

Design guidance, not contract: lean on `dataviz`-style restraint this is a work tool, so
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

The pipeline's own invariants still hold never silent, never truncated or overlapping,
one TTS call per segment, no per-video content, the manifest stays small. Two app-specific
additions:

* **The user's edits outrank the pipeline.** A `locked` field is never overwritten by a
  re-run, and a manual `keep` is never re-decided by `mark_keep`.
* **Every destructive action is undoable or confirmed.** Re-running a stage discards
  generated work; say so before doing it.
