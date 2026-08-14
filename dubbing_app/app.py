"""The HTTP surface. JSON in, JSON out, one error envelope, no pipeline logic.

Routes are exactly the table in docs/APP_ARCHITECTURE.md. Two rules shape the
handlers:

* **No-model edits never become jobs.** `PATCH /segments/{uid}` and friends load
  the manifest, call one `ops` setter and save, on the request thread, in
  microseconds — so the editor stays live while a render runs.
* **Model work is always a job**, and the queue serialises it. A second request
  is queued, not refused; `busy` is reserved for the edits that genuinely cannot
  interleave with a running job.
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import Body, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from dubbing import STAGES, manifest

from . import errors, events, media, ops, peaks, setup, ui
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


# ---------------------------------------------------------------------------
# request bodies
# ---------------------------------------------------------------------------

class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class CreateProject(Strict):
    source: str
    src_lang: str = "he"
    tgt_lang: str = "en"
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


class PatchSegment(Strict):
    """Every field is optional; `None` means "not supplied", never "clear".

    The one exception is `src_lang`/`tgt_lang`, where the empty string clears the
    override — a language tag has no other way to be removed.
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


class UidsBody(Strict):
    uids: list[str] = Field(default_factory=list)


class RenderBody(Strict):
    pass


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------

def create_app(outputs: Path, *, runner=None, version: str | None = None,
               ui_dir: str | Path | None = None) -> FastAPI:
    from . import __version__

    version = version or __version__
    # Read once at startup, not per request: `/health` is polled.
    commit = setup.git_commit()
    ui_root = ui.resolve_dir(ui_dir)
    projects = Projects(Path(outputs))
    projects.root.mkdir(parents=True, exist_ok=True)
    bus = EventBus()
    queue = JobQueue(runner or SubprocessRunner(), bus)
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
    app.state.version = version
    errors.install(app)
    # The Vite dev server is a different origin; the server only ever binds
    # loopback, so this is not a wider hole than the port already is.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"], expose_headers=["Content-Range", "Accept-Ranges"])

    def lock_for(name: str) -> threading.Lock:
        with locks_guard:
            return locks.setdefault(name, threading.Lock())

    def guard_structural(name: str) -> None:
        """Refuse split/merge/bounds while a stage is running on this project.

        They renumber `seg["id"]`, which the running stage is iterating over.
        Field edits are allowed at any time — the contract requires it.
        """
        for job in queue.active(name):
            if job.status == "running":
                raise busy(f"job {job.id} ({job.kind}) is running on {name!r}; "
                           "structural edits would renumber segments under it")

    def enqueue(kind: str, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        workdir = projects.require_dir(name)
        job = queue.submit(kind, name, {**payload, "workdir": str(workdir)})
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
        """Can this machine run the pipeline? Filesystem and env only — no model
        is loaded, so the desktop shell may call it before anything else."""
        return setup.report(projects.root)

    # -- projects ----------------------------------------------------------

    @app.get("/api/projects")
    def list_projects() -> dict[str, Any]:
        return {"projects": projects.list()}

    @app.post("/api/projects", status_code=201)
    def create_project(body: CreateProject) -> dict[str, Any]:
        source = (body.source or "").strip()
        if not source:
            raise invalid("source is required")
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
        return {"name": name, "manifest": m, "report": projects.report(name),
                "stages": projects.stage_status(m),
                "summary": projects.summary(name),
                "jobs": [j.to_dict() for j in queue.list(name)]}

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
        if "start" in fields:
            guard_structural(name)
        with lock_for(name):
            m = projects.load(name)
            try:
                seg = _apply_patch(m, uid, fields)
            except ops.EditError as exc:
                raise invalid(str(exc)) from exc
            projects.save(name, m)
        for field in fields:
            bus.publish(name, events.segment_event(uid, field, "done"))
        return {"segment": projects.enrich(name, m, seg)}

    @app.post("/api/projects/{name}/segments/{uid}/split")
    def split_segment(name: str, uid: str, body: SplitBody) -> dict[str, Any]:
        guard_structural(name)
        with lock_for(name):
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
        guard_structural(name)
        with lock_for(name):
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
        return {"job": enqueue("retranslate", name, {"uids": _check_uids(projects, name, body)})}

    @app.post("/api/projects/{name}/resynthesize", status_code=202)
    def resynthesize(name: str, body: UidsBody) -> dict[str, Any]:
        return {"job": enqueue("resynthesize", name, {"uids": _check_uids(projects, name, body)})}

    @app.post("/api/projects/{name}/render", status_code=202)
    def render(name: str, body: RenderBody = Body(default=RenderBody())) -> dict[str, Any]:
        projects.load(name)
        return {"job": enqueue("render", name, {})}

    # -- events ------------------------------------------------------------

    @app.get("/api/projects/{name}/events")
    async def project_events(name: str, request: Request):
        m = projects.load(name)
        sub = bus.subscribe(name)
        prelude: list[dict[str, Any]] = [
            {"type": "log", "level": "info", "message": f"watching {name}"},
        ]
        status = projects.stage_status(m)
        prelude += [events.stage_event(s, status[s], 1.0 if status[s] == "done" else None)
                    for s in STAGES]
        prelude += [events.job_event(j.id, j.status, j.error, kind=j.kind, project=j.project)
                    for j in queue.list(name) if j.status not in ("done",)]
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
    def cancel_job(job_id: str) -> dict[str, Any]:
        return {"job": queue.cancel(job_id).to_dict()}

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
    """Route a PATCH body through the `ops` setters — never a raw dict update.

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
