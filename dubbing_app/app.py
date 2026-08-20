"""The HTTP surface. JSON in, JSON out, one error envelope, no pipeline logic.

Routes are exactly the table in docs/APP_ARCHITECTURE.md. Two rules shape the
handlers:

* **No-model edits never become jobs.** `PATCH /segments/{uid}` and friends load
  the manifest, call one `ops` setter and save, on the request thread, in
  microseconds so the editor stays live while a render runs.
* **Model work is always a job**, and the queue serialises it. A second request
  is queued, not refused; `busy` is reserved for the edits that genuinely cannot
  interleave with a running job.
"""

from __future__ import annotations

import json
import secrets
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import Body, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from dubbing import STAGES, manifest

from . import errors, events, install, jobs as jobs_mod, media, ops, peaks, setup, ui
from .errors import ApiError, busy, invalid, not_found
from .events import EventBus
from .jobs import JobQueue
from .projects import Projects
from .runner import SubprocessRunner

# The `dubbing.cli` choice lists, restated. They cannot be imported: `dubbing.cli`
# pulls in every stage, and therefore torch and MLX, which is exactly what keeps
# the server out of this process (see `dubbing_app.runner`). A test asserts these
# are still character-for-character what the parser accepts. Restating them is
# what turns a bad option into a 400 instead of a project directory whose one job
# dies on an argparse usage dump and can never be re-run.
Genre = Literal["documentary", "movie"]
Register = Literal["narration", "dialogue"]
Transcript = Literal["auto", "captions", "asr"]
TtsModel = Literal["1.7b"]  # 0.6b is retired for new runs; old manifests that recorded it still re-run
# `cli.SRC_CHOICES` / `cli.TGT_CHOICES`, sorted, legacy aliases included. An
# unrecognised code is worse here than an unrecognised genre: `script.script_for`
# answers "latin" for anything it does not know, so a project created with "jp"
# would run to completion with every script-derived verdict quietly wrong,
# instead of failing.
SrcLang = Literal["ar", "de", "en", "es", "fr", "he", "it", "iw", "ja", "ko", "pt", "ru", "zh"]
TgtLang = Literal["de", "en", "es", "fr", "he", "it", "iw", "ja", "ko", "pt", "ru", "zh"]


# ---------------------------------------------------------------------------
# request bodies
# ---------------------------------------------------------------------------

class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class CreateProject(Strict):
    source: str
    src_lang: SrcLang = "he"
    tgt_lang: TgtLang = "en"
    duration: float | None = None
    name: str | None = None
    context: str | None = None
    genre: Genre | None = None
    # `register` is a BaseModel attribute, so the field is spelled with a trailing
    # underscore and aliased back to the wire name the CLI flag uses.
    register_: Register | None = Field(default=None, alias="register")
    transcript: Transcript | None = None
    tts_model: TtsModel | None = None
    dub_foreign: bool = False
    captions: str | None = None


class PatchProject(Strict):
    """The three run options that are still a decision after the run has started.

    Genre, register and context change what the *translator* is told; nothing
    that has already been fetched, transcribed or segmented depends on them, so
    they can be corrected without recreating the project which is what the
    import screen now promises. The source and the language pair are not here on
    purpose: changing either invalidates the fetch and every stage after it,
    which is a new project wearing an old project's name.

    Every field is optional and `None` means "not supplied". `context` is the one
    that can be *cleared*, by sending the empty string: a note that turned out to
    be wrong has to have a way out, and the pipeline reads a missing context and
    an empty one identically (`edit._args`, `cli.main`).
    """

    context: str | None = None
    genre: Genre | None = None
    register_: Register | None = Field(default=None, alias="register")


class PatchSegment(Strict):
    """Every field is optional; `None` means "not supplied", never "clear".

    The one exception is `src_lang`/`tgt_lang`, where the empty string clears the
    override a language tag has no other way to be removed.
    """

    text: str | None = None
    text_en: str | None = None
    keep: bool | None = None
    keep_reason: str | None = None
    speaker: str | None = None
    start: float | None = None
    end: float | None = None
    src_lang: str | None = None
    tgt_lang: str | None = None
    tts_opts: dict[str, Any] | None = None
    # Editing a field locks it, so unlocking needs its own way in: `{}` releases
    # every lock on the segment, letting a re-run own that field again. Without
    # this a hand-edit is permanent and the user can never hand the line back.
    locked: dict[str, bool] | None = None


class SplitBody(Strict):
    at: float


class MergeBody(Strict):
    # `with` is a keyword; the wire name is still `with`, and `uid` is accepted.
    other: str | None = Field(default=None, alias="with")
    uid: str | None = None


class AddSegmentBody(Strict):
    """A span the pipeline missed, claimed by hand.

    `text` is required and not optional-with-a-placeholder: it is what gets
    translated and spoken, and a segment with nothing to say is the one thing
    `edit.add` cannot make (see its docstring). `speaker` is optional — omitted,
    the nearest segment's voice is inherited, which is right for a line dropped
    out of the middle of a passage and is the only content-free default there is.
    """

    start: float
    end: float
    text: str
    speaker: str | None = None


class UidsBody(Strict):
    uids: list[str] = Field(default_factory=list)
    # One user gesture, two jobs: "Dub these 27" flips the verdicts, then queues a
    # retranslate and a resynthesize. Only the client knows those two POSTs came from
    # one decision, so it mints the id and sends it on both; `DELETE /api/jobs/{id}
    # ?batch=1` is what makes cancelling the decision possible instead of cancelling
    # half of it. Optional: a lone re-voice from the inspector has no batch.
    batch: str | None = Field(default=None, max_length=64)


class RenderBody(Strict):
    pass


class RunBody(Strict):
    """Nothing. A resume takes no arguments it re-runs *this* project, with the
    options this project recorded. A body that could change them would make the
    button "run it differently", which is what `PATCH /api/projects/{name}` is
    for, and the two must not be one gesture."""


class InstallBody(Strict):
    """One field, and it is a *key*, never a command.

    `id` is looked up in `install.INSTALLERS` (tools) and then in
    `setup.model_downloads()` (hub-snapshot models); a miss in both is a 400.
    Strict-extra is what keeps a hopeful `{"id": "ffmpeg", "argv": [...]}` — or
    a `repo_id` of the client's choosing — from ever being read, let alone run.
    """

    id: str


class HfTokenBody(Strict):
    """One field, and it is a *credential*.

    It is validated for shape, written to the workspace `.env`, and that is the
    whole life it has in this process: never logged, never echoed back (the
    response is the re-probed check row, which says "set" and nothing else),
    never stored anywhere a `GET` could read it out again. The 400s below are
    worded around the token, not with it — an error message is the easiest
    place to leak a secret, because it is the one string everyone pastes into
    a bug report.
    """

    token: str


# The two spellings `hf_token_check` accepts, so a delete that removed only one
# would leave the row green and the user certain they had removed it.
_HF_TOKEN_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def _rewrite_env_token(path: Path, token: str | None) -> None:
    """Swap (or drop, `token=None`) the token lines in `.env`, and touch nothing
    else the file may hold other variables the user put there by hand, and a
    writer that rewrote them from parsed values would strip their comments and
    quoting. Line surgery, not serialization. Creates the file when absent; the
    workspace directory already exists (the server is running out of it).
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        lines = []
    kept = [line for line in lines
            if not ("=" in line and line.partition("=")[0].strip() in _HF_TOKEN_KEYS)]
    if token is not None:
        kept.append(f"HF_TOKEN={token}")
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    # A credential file is nobody else's to read. Best effort — .env may sit on
    # a filesystem that ignores modes, and the write already succeeded.
    try:
        path.chmod(0o600)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------

class RequestGate:
    """The request gate, chosen by how the server is bound (dubbing_app/server.py).

    * Loopback bind (the default, and the desktop shell's only mode): no token,
      but the Host header must BE a loopback name. Strict CORS already stops a
      malicious page reading responses — DNS rebinding is the remaining move,
      serving the attacker's page from a hostname that *resolves* to 127.0.0.1:
      same-origin in the browser's eyes, foreign in the Host header, which is
      why the header is the check. ("testserver" is FastAPI's TestClient
      default; not a name a browser can be steered to via public DNS.)
    * Non-loopback bind: the Host header can legitimately be any LAN name, so
      the gate is a bearer token instead — Jupyter-style: `?token=` once sets
      an HttpOnly cookie and every later request (media, events, the lot)
      rides the cookie.

    Pure ASGI on purpose, not `@app.middleware`: that decorator is Starlette's
    BaseHTTPMiddleware, which re-streams every response through a shim that
    loses client-disconnect cancellation — the events stream then never
    released its subscription. Raw ASGI passes the stream and its cancellation
    through untouched. OPTIONS passes free: a CORS preflight carries neither
    cookie nor token by design, and answering it reveals nothing.
    """

    TRUSTED_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "testserver"})
    COOKIE = "mambodubb_token"

    def __init__(self, app, *, token: str | None):
        self.app = app
        self.token = token

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") == "OPTIONS":
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope.get("headers") or []}
        if self.token:
            return await self._token_gate(scope, receive, send, headers)
        host = headers.get("host", "").strip()
        name = (host[1:host.index("]")] if host.startswith("[") and "]" in host
                else host.rsplit(":", 1)[0] if host.count(":") == 1 else host)
        if name.lower() not in self.TRUSTED_HOSTS:
            return await self._refuse(send, 403, f"refused Host {host!r}: this "
                                      "server answers loopback names only")
        await self.app(scope, receive, send)

    async def _token_gate(self, scope, receive, send, headers) -> None:
        from urllib.parse import parse_qs

        cookie = headers.get("cookie", "")
        supplied = ""
        for part in cookie.split(";"):
            name, _, value = part.strip().partition("=")
            if name == self.COOKIE:
                supplied = value
        from_cookie = bool(supplied)
        if not supplied:
            qs = parse_qs(scope.get("query_string", b"").decode("latin-1"))
            supplied = (qs.get("token") or [""])[0]
        if not supplied:
            # RFC 7235: the scheme is case-insensitive, so `bearer x` is as
            # valid as `Bearer x`.
            scheme, _, value = headers.get("authorization", "").strip().partition(" ")
            if scheme.lower() == "bearer":
                supplied = value.strip()
        # Compare bytes: `compare_digest` raises TypeError on non-ASCII *str*
        # input, and the query string is attacker-supplied — `?token=%C3%A9`
        # must be a 401, not a 500 wearing the exception text.
        if not secrets.compare_digest(supplied.encode("utf-8", "surrogateescape"),
                                      self.token.encode("utf-8")):
            return await self._refuse(send, 401, "authentication required: open "
                                      "the ?token=… link the server printed at startup")

        async def send_with_cookie(message):
            if message["type"] == "http.response.start" and not from_cookie:
                message["headers"] = list(message.get("headers") or []) + [
                    (b"set-cookie",
                     f"{self.COOKIE}={self.token}; HttpOnly; SameSite=Strict; "
                     "Path=/".encode())]
            await send(message)

        await self.app(scope, receive, send_with_cookie)

    @staticmethod
    async def _refuse(send, status: int, message: str) -> None:
        body = json.dumps({"error": message}).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


def create_app(outputs: Path, *, runner=None, version: str | None = None,
               ui_dir: str | Path | None = None, token: str | None = None) -> FastAPI:
    from . import __version__

    version = version or __version__
    # Read once at startup, not per request: `/health` is polled.
    commit = setup.git_commit()
    ui_root = ui.resolve_dir(ui_dir)
    projects = Projects(Path(outputs))
    projects.root.mkdir(parents=True, exist_ok=True)
    bus = EventBus()
    queue = JobQueue(runner or SubprocessRunner(), bus)
    # Deliberately not the JobQueue: a `brew install` loads no model, and making
    # it wait behind a render would mean waiting an hour to fix the thing that
    # is blocking the render. Its own one-at-a-time slot, in this process.
    installer = install.Installer(setup.probe)
    # One writer at a time per project: two PATCHes arriving together would
    # otherwise both load, both edit and the second would drop the first's change.
    locks: dict[str, threading.Lock] = {}
    locks_guard = threading.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        queue.start()
        try:
            yield
        finally:
            queue.stop()

    app = FastAPI(title="MamboDubb", version=version, docs_url=None, redoc_url=None,
                  lifespan=lifespan)
    app.state.projects = projects
    app.state.bus = bus
    app.state.jobs = queue
    app.state.installer = installer
    app.state.version = version
    errors.install(app)
    # The gate goes on first so the CORS layer added after it wraps OUTSIDE it
    # (Starlette: later add_middleware = outermost) and answers preflights
    # itself — an OPTIONS request carries neither cookie nor token by design.
    app.add_middleware(RequestGate, token=token)
    # Exactly the cross-origin callers that exist, and no others. `["*"]` here
    # meant any web page open in the user's browser could read and drive this
    # API on localhost — the server can read the filesystem and run the
    # pipeline, so that wildcard was the hole, not the port. The Tauri webview
    # (tauri://localhost, and tauri.localhost on Windows) and a localhost Vite
    # dev server are the only legitimate foreign origins; the served UI is
    # same-origin and needs no CORS at all.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^(tauri://localhost|https?://tauri\.localhost"
                           r"|http://(localhost|127\.0\.0\.1)(:\d+)?)$",
        allow_methods=["*"], allow_headers=["*"],
        expose_headers=["Content-Range", "Accept-Ranges"],
    )

    def lock_for(name: str) -> threading.Lock:
        with locks_guard:
            return locks.setdefault(name, threading.Lock())

    def guard_structural(name: str) -> None:
        """Refuse split/merge/bounds while a stage is running on this project.

        They renumber `seg["id"]`, which the running stage is iterating over.
        Field edits are allowed at any time the contract requires it.

        Called **inside** `lock_for(name)`, never before it: checked outside, a job
        could start in the window between the check and the write, and the job
        child which loaded the manifest before the split existed would write
        its own segment list back over it (`worker.Journal`, which matches by uid
        and cannot re-apply a segment it has never seen). The lock does not stop
        the worker thread, so the journal reports what it finds as a conflict; this
        closes the half of the race the server owns.
        """
        for job in queue.active(name):
            if job.status == "running":
                raise busy(f"job {job.id} ({job.kind}) is running on {name!r}; "
                           "structural edits would renumber segments under it")

    def guard_idle(name: str, because: str) -> None:
        """Refuse while this project has any job queued or running.

        Stricter than `guard_structural`, which tolerates a *queued* job because a
        queued job has not read the manifest yet. These two callers cannot:

        * a resume enqueued behind a running run would re-run the stages the
          running one is in the middle of, against the same directory;
        * a run option changed while a job holds the manifest is a setting that
          job has already read, and whose effect the user would then watch not
          happen.

        Both are cheap to refuse and expensive to explain afterwards.
        """
        for job in queue.active(name):
            raise busy(f"job {job.id} ({job.kind}) is {job.status} on {name!r} {because}")

    def enqueue(kind: str, name: str, payload: dict[str, Any],
                *, batch: str | None = None) -> dict[str, Any]:
        workdir = projects.require_dir(name)
        job = queue.submit(kind, name, {**payload, "workdir": str(workdir)}, batch=batch)
        return job.to_dict()

    # -- health ------------------------------------------------------------

    @app.get("/health")
    def health() -> dict[str, Any]:
        current = queue.current
        return {"status": "ok", "version": version, "commit": commit,
                "outputs": str(projects.root),
                "busy": current.id if current else None,
                "queued": len(queue.active())}

    # -- setup / first run -------------------------------------------------

    @app.get("/api/setup")
    def setup_report() -> dict[str, Any]:
        """Can this machine run the pipeline? Filesystem and env only no model
        is loaded, so the desktop shell may call it before anything else."""
        return setup.report(projects.root)

    @app.post("/api/setup/install", status_code=202)
    def start_install(body: InstallBody) -> dict[str, Any]:
        """Install one missing tool, or download one missing model. `id` is a
        key into a server-side table (argv for the tools, hub repo + local dir
        for the models) — nothing from the body is ever executed. 400 for
        anything in neither table, 409 while another install is running."""
        return installer.start(body.id)

    @app.get("/api/setup/install")
    def install_status() -> dict[str, Any]:
        """The running install, if any, and the last lines of its output.

        Polled rather than streamed: setup has no project, so it has no event
        stream, and an install measured in minutes is served fine by a 2 s poll.
        Carries a freshly probed check row once the worker has exited; while a
        model download runs it also carries `bytes_done`/`bytes_total`, the
        directory's size on disk against the table's estimate.
        """
        return installer.status()

    @app.post("/api/setup/hf_token")
    def set_hf_token(body: HfTokenBody) -> dict[str, Any]:
        """Save the Hugging Face token into the workspace `.env` the in-app
        version of the one setup step that used to require finding a hidden
        folder and hand-editing a dotfile. Shape checks only: `hf_` prefix, no
        whitespace inside the token is *validated* against the hub the first
        time diarization presents it, and a wrong-but-well-formed token fails
        there with pyannote's own error. Other `.env` lines survive the write.
        Returns the re-probed row; the token itself never appears in a
        response, a log line, or an error message.
        """
        token = body.token.strip()
        if not token:
            raise invalid("no token: paste the whole hf_… string from "
                          "https://huggingface.co/settings/tokens")
        if any(ch.isspace() for ch in token):
            raise invalid("that token has whitespace inside it — a copy that "
                          "caught a line break or a trailing word. Copy just "
                          "the hf_… string, nothing around it.")
        if not token.startswith("hf_") or len(token) <= len("hf_"):
            raise invalid("that does not look like a Hugging Face token: they "
                          "start with hf_. Copy it from "
                          "https://huggingface.co/settings/tokens")
        _rewrite_env_token(setup.env_path(), token)
        return setup.hf_token_check()

    @app.delete("/api/setup/hf_token")
    def clear_hf_token() -> dict[str, Any]:
        """Remove the token from `.env` (both spellings), leaving every other
        line alone. The re-probed row is the honest answer to "is it gone":
        it stays green if the shell's *environment* still carries HF_TOKEN,
        which no file edit can undo and the row's `source` says so."""
        _rewrite_env_token(setup.env_path(), None)
        return setup.hf_token_check()

    # -- projects ----------------------------------------------------------

    @app.get("/api/projects")
    def list_projects() -> dict[str, Any]:
        return {"projects": projects.list()}

    @app.post("/api/projects", status_code=201)
    def create_project(body: CreateProject) -> dict[str, Any]:
        source = (body.source or "").strip()
        if not source:
            raise invalid("source is required")
        # A local-file input is probed by actually opening it: macOS TCC denies at
        # open(), not at stat(), so this is the only check that tells the truth.
        # Failing here costs the user a sentence; failing at the fetch (or worse,
        # mix) stage costs them a job that dies minutes in with an ffmpeg error
        # that says nothing about permissions.
        if not source.lower().startswith(("http://", "https://")):
            path = Path(source).expanduser()
            try:
                with open(path, "rb") as fh:
                    fh.read(1)
            except FileNotFoundError:
                raise invalid(f"input not found: {path}. Paste the file's full "
                              "path, or pick it with Choose file.")
            except OSError:
                raise invalid(f"macOS is not letting this app read {path}. Grant "
                              "access in System Settings → Privacy & Security → "
                              "Files and Folders, or move the file somewhere the "
                              "app can read.")
        # Hebrew needs two local models the other targets do not (the Qwen3-TTS
        # Hebrew LoRA and its G2P). Said here, not at the tts stage: a run that
        # found out there would already have paid for stems, ASR and diarization.
        from dubbing import hebrew

        if hebrew.is_hebrew(body.tgt_lang):
            gaps = hebrew.missing()
            if gaps:
                raise invalid("Hebrew is not available as a target yet: "
                              + "; ".join(g.replace("\n    ", " ") for g in gaps))
        from .projects import slugify

        name = projects.unique_name(body.name or slugify(source))
        workdir = projects.dir_for(name)
        workdir.mkdir(parents=True, exist_ok=False)
        record = {
            "input": source, "src_lang": body.src_lang, "tgt_lang": body.tgt_lang,
            "duration_limit": body.duration, "context": body.context,
            # The pipeline options this run was made with. `rebuild` needs them
            # later and the headless CLI never records them anywhere else.
            "app_opts": {"genre": body.genre or "documentary",
                         "register": body.register_ or "narration",
                         "transcript": body.transcript or "auto",
                         "tts_model": body.tts_model or "1.7b",
                         "dub_foreign": bool(body.dub_foreign),
                         "captions": body.captions},
        }
        m = manifest.new(record)
        manifest.save(workdir, m)
        job = enqueue("run", name, {"source": record})
        return {"project": projects.summary(name), "job": job}

    @app.get("/api/projects/{name}")
    def get_project(name: str) -> dict[str, Any]:
        m = projects.load(name)
        # The queue is the only thing that knows a stage is running right now or
        # that one died: the manifest records what finished, and an unfinished
        # stage is indistinguishable from one that was never reached.
        jobs = [j.to_dict() for j in queue.list(name)]
        return {"name": name, "manifest": m, "report": projects.report(name, m),
                "render": projects.render_state(m),
                "stages": projects.stage_status(m, jobs),
                "summary": projects.summary(name, jobs),
                "jobs": jobs}

    @app.patch("/api/projects/{name}")
    def patch_project(name: str, body: PatchProject = Body(...)) -> dict[str, Any]:
        """Change this run's genre, register or context.

        Written as **flat keys** on `m["source"]`, which is the spelling
        `dubbing.edit._args` reads first (`app_opts` is the app's own, older
        record and loses to it). So the change reaches every path that re-runs
        anything a per-line re-translate, a render, a resume without a second
        copy of the setting anywhere.

        No stage is invalidated and no job is enqueued: these three are inputs to
        the *translator*, so they take effect the next time translation runs, and
        saying so is the UI's job. Silently re-translating two hundred lines
        because a dropdown moved would be a worse surprise than the wait.
        """
        fields = body.model_dump(exclude_unset=True, by_alias=True)
        if not fields:
            raise invalid("empty patch: send at least one of context, genre, register")
        with lock_for(name):
            guard_idle(name, "run options are read when a job starts, not while it runs")
            m = projects.load(name)
            source = m.setdefault("source", {})
            for key, value in fields.items():
                # The empty string is how a context note is *removed*; for the two
                # Literals there is no such value, so `None` never gets this far.
                source[key] = (value or "").strip() or None if key == "context" else value
            projects.save(name, m)
        return {"source": m["source"], "project": projects.summary(name)}

    @app.post("/api/projects/{name}/run", status_code=202)
    def resume_run(name: str, body: RunBody = Body(default=RunBody())) -> dict[str, Any]:
        """Run this project which, on a project that has already run, is a resume.

        There is no separate resume machinery and there must not be: every stage
        is skipped when its inputs and outputs are unchanged (AGENTS.md), so
        enqueueing the same `run` job that created the project picks up exactly
        where it stopped. The payload is the one `create_project` built, read back
        off the manifest, so the resumed run reproduces the original run's argv —
        including any option changed since through `PATCH /api/projects/{name}`.

        The local-file probe `create_project` does is deliberately *not* repeated:
        a run past `fetch` no longer needs its input, and refusing to resume a
        half-finished dub because the source file has since been moved would be a
        refusal with no cause behind it.
        """
        m = projects.load(name)
        source = dict(m.get("source") or {})
        if not source.get("input"):
            raise invalid(f"project {name!r} does not record what it was made from, "
                          "so there is nothing to re-run")
        with lock_for(name):
            guard_idle(name, "the run is already going")
            return {"job": enqueue("run", name, {"source": source})}

    # -- segments ----------------------------------------------------------

    @app.get("/api/projects/{name}/segments")
    def list_segments(name: str) -> dict[str, Any]:
        m = projects.load(name)
        return {"segments": projects.enrich_all(name, m),
                "speakers": m.get("speakers") or {}}

    @app.get("/api/projects/{name}/peaks")
    def project_peaks(name: str, file: str = "source", n: int = 1000) -> dict[str, Any]:
        """Downsampled waveform for a timeline lane. Instant; reads no models."""
        return peaks.for_project(projects.require_dir(name), file, n)

    @app.patch("/api/projects/{name}/segments/{uid}")
    def patch_segment(name: str, uid: str, body: PatchSegment = Body(...)) -> dict[str, Any]:
        fields = body.model_dump(exclude_unset=True)
        if not fields:
            raise invalid("empty patch")
        if ("start" in fields) != ("end" in fields):
            raise invalid("start and end must be changed together")
        with lock_for(name):
            if "start" in fields:
                guard_structural(name)
            m = projects.load(name)
            try:
                seg = _apply_patch(m, uid, fields)
            except ops.EditError as exc:
                raise invalid(str(exc)) from exc
            projects.save(name, m)
        for field in fields:
            bus.publish(name, events.segment_event(uid, field, "done"))
        return {"segment": projects.enrich(name, m, seg)}

    @app.post("/api/projects/{name}/segments", status_code=201)
    def add_segment(name: str, body: AddSegmentBody) -> dict[str, Any]:
        """Claim an uncovered span of the timeline as a new segment.

        Structural, so it is guarded like a split: it appends to the list and
        renumbers every `id` after it, which a running stage is iterating over.

        An overlap with a neighbour is a 400, not a clamp. `timeline.place` is
        the sole authority on where audio goes and it asserts non-overlap, so a
        request that would break that has to be answered, not quietly turned
        into a different request the user did not make.
        """
        with lock_for(name):
            guard_structural(name)
            m = projects.load(name)
            try:
                uid = ops.add(m, body.start, body.end, text=body.text,
                              speaker=body.speaker)
            except ops.EditError as exc:
                raise invalid(str(exc)) from exc
            projects.save(name, m)
        bus.publish(name, events.segment_event(uid, "add", "done"))
        return {"uid": uid, "segment": projects.enrich(name, m, ops.find(m, uid))}

    @app.delete("/api/projects/{name}/segments/{uid}")
    def remove_segment(name: str, uid: str) -> dict[str, Any]:
        """Take this segment out of the dub entirely.

        Not a keep: a kept segment is still placed and still ducks the bed. A
        removed one stops existing, and `mix` plays the original vocals in the
        span nothing claims — see `dubbing.edit.remove`. The deleted record is
        returned so a client can offer an undo without having kept a copy.
        """
        with lock_for(name):
            guard_structural(name)
            m = projects.load(name)
            try:
                gone = ops.remove(m, uid)
            except ops.SegmentNotFound as exc:
                raise not_found(str(exc)) from exc
            except ops.EditError as exc:
                raise invalid(str(exc)) from exc
            projects.save(name, m)
        bus.publish(name, events.segment_event(uid, "remove", "done"))
        return {"uid": uid, "removed": gone, "segments": len(m.get("segments") or [])}

    @app.post("/api/projects/{name}/segments/{uid}/split")
    def split_segment(name: str, uid: str, body: SplitBody) -> dict[str, Any]:
        with lock_for(name):
            guard_structural(name)
            m = projects.load(name)
            try:
                uid_a, uid_b = ops.split(m, uid, body.at)
            except ops.EditError as exc:
                raise invalid(str(exc)) from exc
            projects.save(name, m)
        bus.publish(name, events.segment_event(uid, "split", "done"))
        return {"uids": [uid_a, uid_b],
                "segments": [projects.enrich(name, m, ops.find(m, u)) for u in (uid_a, uid_b)]}

    @app.post("/api/projects/{name}/segments/{uid}/merge")
    def merge_segment(name: str, uid: str, body: MergeBody = Body(default=MergeBody())
                      ) -> dict[str, Any]:
        with lock_for(name):
            guard_structural(name)
            m = projects.load(name)
            other = body.other or body.uid or _next_uid(m, uid)
            if not other:
                raise invalid("nothing to merge with: no following segment")
            try:
                merged = ops.merge(m, uid, other)
            except ops.EditError as exc:
                raise invalid(str(exc)) from exc
            projects.save(name, m)
        bus.publish(name, events.segment_event(merged, "merge", "done"))
        return {"uid": merged, "segment": projects.enrich(name, m, ops.find(m, merged))}

    # -- model jobs --------------------------------------------------------

    @app.post("/api/projects/{name}/retranslate", status_code=202)
    def retranslate(name: str, body: UidsBody) -> dict[str, Any]:
        return {"job": enqueue("retranslate", name,
                               {"uids": _check_uids(projects, name, body)},
                               batch=body.batch)}

    @app.post("/api/projects/{name}/resynthesize", status_code=202)
    def resynthesize(name: str, body: UidsBody) -> dict[str, Any]:
        return {"job": enqueue("resynthesize", name,
                               {"uids": _check_uids(projects, name, body)},
                               batch=body.batch)}

    @app.post("/api/projects/{name}/render", status_code=202)
    def render(name: str, body: RenderBody = Body(default=RenderBody())) -> dict[str, Any]:
        projects.load(name)
        return {"job": enqueue("render", name, {})}

    # -- events ------------------------------------------------------------

    @app.get("/api/projects/{name}/events")
    async def project_events(name: str, request: Request):
        """The stream, opened with a prelude that catches a new reader up.

        Two things the prelude must not do, both learned from the client:

        * **Nothing terminal is replayed.** A failed or cancelled job replayed on
          every reconnect resurrects an error bar the user dismissed an hour ago,
          and the UI reconnects on every navigation and every wake from sleep.
          Only work that is still queued or running is news.
        * **Replayed stage frames say so** (`"replay": true`). They are a snapshot
          of nine stages at once, not a progression, so a client that pins its
          display to "the last stage frame seen" would read the ninth as the stage
          the run is in.
        """
        m = projects.load(name)
        sub = bus.subscribe(name)
        prelude: list[dict[str, Any]] = [
            {"type": "log", "level": "info", "message": f"watching {name}"},
        ]
        jobs = [j.to_dict() for j in queue.list(name)]
        status = projects.stage_status(m, jobs)
        prelude += [{**events.stage_event(s, status[s],
                                          1.0 if status[s] == "done" else None),
                     "replay": True}
                    for s in STAGES]
        # `uids`/`batch` ride along for the same reason `kind` does: a client that
        # reconnects rebuilds "which rows are busy" and "what would Cancel stop" from
        # the frames alone, and a replayed job that omitted them would come back as a
        # job about nothing rows stop pulsing on reload mid-render.
        prelude += [events.job_event(j["id"], j["status"], j["error"],
                                     kind=j["kind"], project=j["project"],
                                     uids=j["uids"], batch=j["batch"], replay=True)
                    for j in jobs if j["status"] not in jobs_mod.TERMINAL]
        return StreamingResponse(
            events.stream(sub, prelude=prelude),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no",
                     "Connection": "keep-alive"},
        )

    # -- jobs --------------------------------------------------------------

    @app.get("/api/jobs")
    def list_jobs(project: str | None = None) -> dict[str, Any]:
        current = queue.current
        return {"jobs": [j.to_dict() for j in queue.list(project)],
                "running": current.id if current else None}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        return {"job": queue.get(job_id).to_dict()}

    @app.delete("/api/jobs/{job_id}")
    def cancel_job(job_id: str, batch: bool = False) -> dict[str, Any]:
        """Cancel one job, or with `?batch=1` the whole gesture it came from.

        `job` is always the job named in the path, so existing callers read the same
        field they always did; `cancelled` lists everything this call stopped, which
        for a batch is the job and its queued mates. Cancelling a translate while the
        voice job it feeds stays queued is the audit's disaster and is what `batch=1`
        exists to prevent see `JobQueue.cancel_batch` for why the order matters.
        """
        if not batch:
            return {"job": queue.cancel(job_id).to_dict(), "cancelled": None}
        stopped = [j.to_dict() for j in queue.cancel_batch(job_id)]
        return {"job": queue.get(job_id).to_dict(), "cancelled": stopped}

    # -- media -------------------------------------------------------------

    @app.api_route("/media/{name}/{path:path}", methods=["GET", "HEAD"])
    def serve_media(name: str, path: str, request: Request):
        workdir = projects.require_dir(name)
        target = media.resolve(workdir, path)
        return media.serve(target, request.headers.get("range"),
                           head=request.method == "HEAD")

    @app.get("/media/{name}")
    def media_root(name: str):
        raise not_found("no such media file")

    # -- built UI (LAST: the catch-all must never shadow a route above) ------

    ui.install(app, ui_root)

    return app


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _apply_patch(m: dict[str, Any], uid: str, fields: dict[str, Any]) -> dict[str, Any]:
    """Route a PATCH body through the `ops` setters never a raw dict update.

    Going through the setters is what applies each edit's side effects: bounds
    re-place, a manual keep locks itself, and nothing writes a key the manifest
    whitelist would drop.
    """
    if ops.find(m, uid) is None:
        raise not_found(f"no segment {uid!r}")
    if "text" in fields or "text_en" in fields:
        ops.set_text(m, uid, text=fields.get("text"), text_en=fields.get("text_en"))
    if "keep" in fields:
        ops.set_keep(m, uid, bool(fields["keep"]),
                     fields.get("keep_reason") or ("manual" if fields["keep"] else None))
    if "speaker" in fields:
        ops.set_speaker(m, uid, fields["speaker"])
    if "start" in fields and "end" in fields:
        ops.set_bounds(m, uid, fields["start"], fields["end"])
    if "src_lang" in fields or "tgt_lang" in fields:
        ops.set_langs(m, uid, src_lang=fields.get("src_lang"), tgt_lang=fields.get("tgt_lang"))
    if "tts_opts" in fields:
        ops.set_tts_opts(m, uid, **(fields["tts_opts"] or {}))
    if "locked" in fields:
        # Applied last: the setters above lock what they touch, so an explicit
        # `locked` in the same patch is the user's final word on this segment.
        ops.set_locked(m, uid, fields["locked"] or {})
    return ops.find(m, uid)


def _next_uid(m: dict[str, Any], uid: str) -> str | None:
    segs = m.get("segments") or []
    for i, seg in enumerate(segs[:-1]):
        if seg.get("uid") == uid:
            return segs[i + 1].get("uid")
    return None


def _check_uids(projects: Projects, name: str, body: UidsBody) -> list[str]:
    m = projects.load(name)
    if not body.uids:
        raise invalid("uids must not be empty")
    known = {s.get("uid") for s in m.get("segments") or []}
    missing = [u for u in body.uids if u not in known]
    if missing:
        raise not_found(f"unknown segment uid(s): {', '.join(missing)}")
    return list(dict.fromkeys(body.uids))


__all__ = ["create_app", "ApiError", "JSONResponse"]
