---
name: mambodubb
description: Drive the MamboDubb local dubbing server over HTTP — create dub projects, watch pipeline progress, fix translations and verdicts per line, re-voice, render, and read run reports. Use when asked to dub a video, operate on an existing run in outputs/, or automate the dubbing studio.
---

# Driving MamboDubb

MamboDubb dubs a video into another language locally: `fetch → stems → transcript
→ segments → translate → tts → timeline → mix → report`. Everything you can do in
the studio UI is this HTTP API; the desktop app is only a shell over it.

## 1. Find or start the server

A running instance (the desktop app spawns one):

```bash
PID=$(pgrep -f "dubbing_app.server" | tail -1)
PORT=$(lsof -nP -a -p "$PID" -iTCP -sTCP:LISTEN | awk '/LISTEN/{sub(/.*:/,"",$9); print $9; exit}')
```

None running? Start one headless from the repo root (stays up until killed):

```bash
nohup uv run mambodubb --port 4400 &   # UI + API on http://127.0.0.1:4400
```

`GET /health` answers when it is up. All examples below assume `B=http://127.0.0.1:$PORT`.

## 2. The job model — read this first

- **One model job runs at a time**; new jobs queue. Job kinds: `run`,
  `retranslate`, `resynthesize`, `render`, each `202`-returns `{"job":{"id",...}}`.
- Poll `GET /api/jobs` (newest last) until the job's `status` is terminal:
  `done | failed | cancelled`. `error` carries the reason. Poll every ~10–30s;
  a full run takes minutes to an hour.
- `DELETE /api/jobs/{id}` cancels; add `?batch=1` to cancel every job sharing the
  id's `batch` (one user gesture = one batch).
- Jobs live in the server's memory: if the server dies, queued work is gone.
- Edits (PATCH/split/merge/add/delete) are instant and never jobs, but
  structural edits 409 while a stage is running on that project.

## 3. Projects

```bash
curl -s $B/api/projects                                   # list, newest first
curl -s -X POST $B/api/projects -H 'Content-Type: application/json' -d '{
  "source": "https://www.youtube.com/watch?v=VIDEO_ID",   # or an absolute file path
  "src_lang": "he", "tgt_lang": "en",
  "duration": 300,            # optional seconds cap — use while iterating
  "context": "who/what the video is about; spellings of names",
  "dub_foreign": false        # true = also dub third-language speech
}'                                                        # 201 → {"name": ..., "job": ...}
curl -s $B/api/projects/NAME                              # detail: stages, outputs, render staleness, jobs
curl -s -X POST $B/api/projects/NAME/run -d '{}'          # resume/re-run with the run's own recorded options
curl -s -X PATCH $B/api/projects/NAME -H 'Content-Type: application/json' \
     -d '{"context": "...", "genre": "movie", "register": "dialogue"}'   # translator inputs, applied on next translate
```

Targets: en/he/ru/fr/es/de/it/pt/zh/ja/ko. Sources: he/en/ar/ru/fr/es/de.
**Declare the language the video actually speaks** — the report's
`source_mismatch` field calls out a wrong declaration; if it fires, recreate the
project with the right `src_lang` rather than fighting the verdicts.

## 4. Segments (the script)

```bash
curl -s $B/api/projects/NAME/segments      # every line: text, text_en, keep, verify, media URLs
```

Address segments by their stable `uid`, never by `id` (ids renumber).
Key fields: `keep` (true = original audio plays), `keep_reason`, `text` (source),
`text_en` (translation), `verify.overlap` (did the clone say it), `locked`
(hand-edited fields a re-run must not touch), `media.play/source` (listen),
`media.fallback` (dub-wanted line currently playing original audio).

```bash
PATCH .../segments/UID   {"text_en": "Better wording"}     # edit + lock the translation
PATCH .../segments/UID   {"keep": false}                   # verdict: dub this line
PATCH .../segments/UID   {"locked": {}}                    # release all locks (hand line back to the pipeline)
POST  .../segments/UID/split  {"at": 12.34}                # structural
POST  .../segments/UID/merge  {"with": "OTHER_UID"}
POST  .../segments  {"start": 431.5, "end": 434.9, "text": "…", "speaker": null}  # claim an uncovered span (overlap = 400)
DELETE .../segments/UID                                    # remove a line outright
```

## 5. Model work on lines, then the video

```bash
POST .../retranslate   {"uids": ["u1","u2"], "batch": "fix-1"}   # re-translate those lines
POST .../resynthesize  {"uids": ["u1","u2"], "batch": "fix-1"}   # re-voice them (queue after retranslate)
POST .../render        {}                                        # remix preview.mp4
```

The project detail's `render` object says whether preview.mp4 is stale and how
many lines changed — render when `stale` and you're done editing, not after
every edit.

## 6. Results

- `GET /api/projects/NAME` → `outputs.preview` / `outputs.srt`, served under
  `/media/NAME/...`; on disk at `outputs/NAME/preview.mp4`.
- `outputs/NAME/report.json` (or via `/media/`) is the accounting:
  `unaccounted` must be `[]`; check `verify` counts, `keep_reasons`,
  `source_mismatch`, `uncovered_audible`.

## Rules

- Never edit files under `outputs/` by hand — the manifest is the pipeline's;
  use the API.
- Never two pipelines on one machine: the models need the RAM. One server, its
  queue orders everything.
- A user's locked edit outranks you: don't clear `locked` on lines you didn't
  write unless asked.
- The CLI (`uv run python -m dubbing <source> -o outputs/NAME`) is equivalent for
  headless runs — but never run it against a project while the server has a job
  on that project.
- Setup problems (missing models/tools): `GET /api/setup` lists them with
  severities and install commands; `POST /api/setup/install {"id": "ffmpeg"}`
  runs a whitelisted installer.
