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
use it as a key across a re-run. Add `uid` to `SEGMENT_KEYS`: an opaque string minted once
when a segment is created and preserved across edits. Every API below keys on `uid`.
`id` stays as-is for ordering and for existing tests.

### New per-segment keys (all must be added to `SEGMENT_KEYS`)

| key | type | meaning |
|---|---|---|
| `uid` | str | stable identity across re-segmentation |
| `src_lang` | str? | per-segment source-language override; falls back to `lang`, then `m["source"]["src_lang"]` |
| `tgt_lang` | str? | per-segment target override — "tag this section and translate it to something else" |
| `tts_opts` | dict? | per-segment synthesis overrides, see below |
| `locked` | dict? | `{field: true}` — fields the user edited by hand that a re-run must not overwrite |

`locked` is what makes the editor safe: a re-translate of the whole run must skip a line
the user corrected. Honour it in `translate.run`/`tts.run` when present.

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
find(m, uid) -> dict | None          # segment by stable uid

# --- no models, instant -------------------------------------------------
set_text(m, uid, *, text=None, text_en=None)      # edit source or target text
set_keep(m, uid, keep: bool, reason: str = "manual")
set_speaker(m, uid, speaker: str)
set_bounds(m, uid, start: float, end: float)      # asserts no overlap with neighbours
set_langs(m, uid, *, src_lang=None, tgt_lang=None)
set_tts_opts(m, uid, **opts)
split(m, uid, at: float) -> tuple[str, str]       # -> two uids
merge(m, uid_a, uid_b) -> str                     # adjacent, same speaker
invalidate(m, uid, *, stages: set[str])           # drop text_en / tts / place for one segment

# --- models, needs the job slot -----------------------------------------
retranslate(m, workdir, uids, *, progress=None) -> dict[str, str]
resynthesize(m, workdir, uids, *, progress=None) -> dict[str, dict]
rebuild(m, workdir, *, from_stage: str, progress=None)   # re-run stages forward
```

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
* `--force <stage>` does **not** invalidate downstream stages (the fingerprint is unchanged
  and the downstream `outputs` lists are empty, so `stage_done` returns True). `rebuild`
  must call `manifest.clear_stage` for each downstream stage itself. Fixing the CLI to match
  its docstring belongs to the segment-ops workstream.

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
