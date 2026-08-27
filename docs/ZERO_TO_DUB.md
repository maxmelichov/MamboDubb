# Zero to dub

The clean-Mac walk: a factory-fresh Apple Silicon Mac, the .dmg, and nothing else no
Homebrew, no Xcode, no terminal, nobody to ask. Every step below was traced through the
code that implements it (file references inline), and every hub repo id was probed
against the live API on 2026-08-20. "WORKS IN-APP" means the user gets there with
clicks alone; "GAP" means the path dead-ends outside the app.

## The journey

| # | step | what happens | status |
|---|------|--------------|--------|
| 1 | Open the .dmg, drag to Applications | Gatekeeper: **"MamboDubb is damaged and can't be opened"** on any Mac but the build machine if the user drags the `.app`. The app is unsigned and un-notarized (no Developer ID). `install.sh` (and **Install MamboDubb.command** inside the `.dmg`) copies to `/Applications`, clears quarantine, and ad-hoc-signs the bundle so it launches. Drag-to-Applications without that step is still a blocker. | **GAP** drag-install; **WORKS** via `install.sh` |
| 2 | First launch | The shell copies the bundled payload to `~/Library/Application Support/MamboDubb/workspace`, forces owner-write, stamps `.mambodubb-payload`, stores the path (`provision.rs::resolve_workspace`). Silent and correct; upgrades refresh source dirs wholesale and never touch `.venv`/`.env`/`models`/`outputs`. | WORKS IN-APP |
| 3 | First `uv sync` | The runner spawns `uv run --project <workspace> python -m dubbing_app.server …` with the **sidecar** uv (`Contents/MacOS/uv`, `workspace.rs::find_uv` no Homebrew needed) and blocks on the ready line. `uv run` builds the ~1.7 GB `.venv`, fetching a managed CPython first. Minutes long, and `main.tsx` renders **nothing at all** until it finishes: `initApiBase()` awaits `start_server` *before the first `root.render`*, so the window is a bare `#110e16` rectangle the whole time. The stderr ring (`get_server_log`) holds the sync's progress, but **no UI code calls it** the sign of life exists and is never shown. | **GAP dead screen** |
| 4 | Setup screen: tools | `SetupGate` routes to `/setup`; `ffmpeg` and `sox` are red BLOCKING rows with Install buttons. The button runs `brew install …` (`dubbing/tools.py::RECIPES`) and a fresh Mac **has no brew**: `install.py::MANAGERS["brew"]` answers "Get it from https://brew.sh" whose own install line is a terminal command. The unattended path exists only on machines that already broke the zero-terminal rule once. | **GAP dead end without Homebrew** |
| 5 | Setup screen: models | Every red model row has a Download button with its price on it; `POST /api/setup/install` maps the id through `setup.model_downloads()` to a `snapshot_download` that resumes if torn off, with byte progress on a 2 s poll. All nine hub ids public and ungated; the one gated model in the tree, diarization, is not in that table at all because its weights ship in the payload. ~18 GB for the blocking set. | WORKS IN-APP |
| 6 | HF token | Nothing needs one, and Setup no longer says so, because a row that says it is a row about a credential. Diarization was the only gated model in the tree, and its weights are CC-BY-4.0, so the identical 31 MB is checked into `third_party/pyannote-speaker-diarization-community-1` and copied into the workspace on first launch (`segments.DIARIZATION_DIR`) no download, no account, and every speaker keeps their own voice. The `hf_token` row, its paste box and `POST|DELETE /api/setup/hf_token` are all removed; anyone who wants `DUB_DIARIZATION_HUB` pointed at the gated upstream repo puts it and `HF_TOKEN` in `.env`, as `.env.example` describes. | NOT A STEP no credential |
| 7 | First dub | With 1 to 6 survived: pick a video (native dialog, absolute path), the job runs, Demucs and the Hebrew G2P fetch their own caches mid-run (network again, progress in the job log). Renders, previews, reveals in Finder. | WORKS IN-APP |

## Hub repos, verified

Every id in `setup.model_downloads()`, probed with `GET
https://huggingface.co/api/models/<id>` all **200**, `gated: false`, `private: false`.
No wrong ids, nothing gated, nothing needing the token step 6 asks for.

| check id | hub repo | status |
|---|---|---|
| `model.translate` | `mlx-community/gemma-4-12B-it-6bit` | 200 public |
| `model.tts.1.7b` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | 200 public |
| `model.tts.0.6b` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | 200 public |
| `model.asr.he` | `ivrit-ai/whisper-large-v3-turbo-ct2` | 200 public |
| `model.asr.src` | `deepdml/faster-whisper-large-v3-turbo-ct2` | 200 public |
| `model.asr.en` | `Systran/faster-whisper-base.en` | 200 public |
| `model.asr.tgt` | `Systran/faster-whisper-base` | 200 public |
| `model.lid` | `speechbrain/lang-id-voxlingua107-ecapa` | 200 public |
| `model.tts.he` | `notmax123/QwenTTS-he-1.7B` | 200 public |

## The gaps, in the order they bite

1. **Unsigned .dmg (drag-install blocker).** Dragging the `.app` to Applications still
   dies at Gatekeeper. `install.sh` / **Install MamboDubb.command** is the path that
   actually launches until Developer ID + notarization exist. The product still owes
   people a drag-to-Applications install; the one-liner is a stopgap, not the design.
2. **Dead window through the first sync.** The multi-minute `uv sync` happens behind an
   unrendered webview. The pieces to fix it already exist and are already wired on one
   side: render first, call `start_server` from inside the app, and poll
   `get_server_log` for the sync's own progress lines while waiting.
3. **ffmpeg/sox need Homebrew, Homebrew needs a terminal.** The blocking-tool path is
   only zero-terminal on machines that already have brew. Bundling static builds (or
   downloading them the way models are downloaded) is the shape of the fix; the
   `MANAGERS` message is honest but honesty is not a dub.
4. ~~**The HF token is a hand-edit in a hidden folder.**~~ Closed by deletion rather than
   by plumbing. There was a POST that wrote the workspace `.env` and a field on the Setup
   row to drive it; then the reason to want a token went away, because the CC-BY-4.0
   diarization weights ship inside the app and a fresh machine needs no Hugging Face
   account at all. Both the row and the endpoint are gone: a step nobody has to take is
   not a step to make easier. The `.env` setting stays for the one person who wants the
   gated upstream repo.
5. **A failed server start has no surface.** If `start_server` errors (sync failed,
   offline first launch), `desktop.ts` swallows it into a `null` base URL and the
   import screen reports a generic fetch failure; the stderr tail that names the real
   reason sits unread behind `get_server_log`.

Steps 2, 3 and 5 also share one assumption worth saying out loud: the first launch, the
tool installs, the model downloads and the first run's self-caching models all need the
network. Offline, the first launch is gap 5 wearing a different cause.
