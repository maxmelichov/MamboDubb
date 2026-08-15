"""The job queue, and the one-job rule.

**At most one pipeline job runs at a time, process-wide.** Not tuning: the
pipeline's models are sequential and never co-resident (AGENTS.md, device notes)
and the machine has 26 GB of unified memory, so a second concurrent job would
swap it to death. The rule is enforced structurally — there is exactly one worker
thread, it takes one job off the queue at a time, and there is no code path that
starts work anywhere else. Everything else queues.

No-model edits (text, keep flips, speaker relabels) never come through here: they
run inline on the request thread, which is what keeps the editor usable while a
render is running.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .errors import ApiError, invalid, not_found
from .events import EventBus, job_event

QUEUED, RUNNING, DONE, FAILED, CANCELLED = "queued", "running", "done", "failed", "cancelled"
TERMINAL = (DONE, FAILED, CANCELLED)

KINDS = ("run", "retranslate", "resynthesize", "render")

Emit = Callable[[dict[str, Any]], None]


@dataclass
class Job:
    id: str
    kind: str
    project: str
    payload: dict[str, Any] = field(default_factory=dict)
    status: str = QUEUED
    error: str | None = None
    result: Any = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    stage: str | None = None
    progress: float | None = None
    message: str | None = None
    cancelling: bool = False
    # Jobs one user gesture created. "Dub these 27" is one decision that becomes a
    # retranslate and a resynthesize, and cancelling only the first leaves the second
    # to voice 27 lines that now have nothing to say — the audit's `tts_failed` pile.
    # The pair is enqueued by the client (two routes, two POSTs), so the id is minted
    # there and travels on both requests; the server's part is to remember it and to
    # let one DELETE reach everything wearing it.
    batch: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "project": self.project,
                "status": self.status, "error": self.error, "result": self.result,
                "created": self.created, "started": self.started, "finished": self.finished,
                "stage": self.stage, "progress": self.progress, "message": self.message,
                # Which segments this job is about, hoisted out of the payload: it
                # is what a UI needs to show "re-voicing 3 lines" or to mark those
                # rows busy, and it was declared on the client's Job type while
                # only the fixtures ever produced it. Always a list, never absent.
                "uids": list(self.payload.get("uids") or []),
                "batch": self.batch,
                "payload": self.payload}


class Runner(Protocol):
    def run(self, job: Job, emit: Emit) -> Any: ...
    def cancel(self, job: Job) -> None: ...


class JobQueue:
    def __init__(self, runner: Runner, bus: EventBus):
        self.runner = runner
        self.bus = bus
        self._queue: queue.Queue[str] = queue.Queue()
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._current: Job | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._work, name="job-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        with self._lock:
            current = self._current
        if current is not None:
            try:
                self.runner.cancel(current)
            except Exception:
                pass
        self._queue.put("")                       # wake the worker
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    # -- api ---------------------------------------------------------------

    def submit(self, kind: str, project: str, payload: dict[str, Any] | None = None,
               *, batch: str | None = None) -> Job:
        if kind not in KINDS:
            raise invalid(f"unknown job kind {kind!r}")
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, project=project,
                  payload=dict(payload or {}), batch=batch or None)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
        self._queue.put(job.id)
        self._publish(job)
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise not_found(f"no job {job_id!r}")
        return job

    def list(self, project: str | None = None) -> list[Job]:
        with self._lock:
            jobs = [self._jobs[i] for i in self._order]
        if project:
            jobs = [j for j in jobs if j.project == project]
        return jobs

    def active(self, project: str | None = None) -> list[Job]:
        return [j for j in self.list(project) if j.status not in TERMINAL]

    @property
    def current(self) -> Job | None:
        with self._lock:
            return self._current

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        with self._lock:
            if job.status in TERMINAL:
                raise invalid(f"job {job_id} already {job.status}")
            running = job is self._current
            job.cancelling = True
            if not running:
                job.status, job.finished = CANCELLED, time.time()
        if running:
            self.runner.cancel(job)               # the runner ends it; worker records it
        else:
            self._publish(job)
        return job

    def cancel_batch(self, job_id: str) -> list[Job]:
        """Cancel `job_id` and every non-terminal job that shares its batch.

        The order is the whole point. `_work` skips a job that is no longer `queued`
        when it reaches it, so the queued members are cancelled **first**, while the
        running one still holds the worker; killing the running job first would free
        the worker to start the very job this call is trying to stop. That race is
        not theoretical — it is the audit's disaster exactly: cancel the translate,
        the voice job starts anyway, and 27 lines are synthesised from translations
        that were never written.

        A job with no batch cancels alone. Members already terminal are skipped
        rather than raising: half a batch finishing before the user hits Cancel is
        the normal case, not an error.
        """
        job = self.get(job_id)
        if not job.batch:
            return [self.cancel(job_id)]
        current = self.current
        members = [j for j in self.list(job.project)
                   if j.batch == job.batch and j.status not in TERMINAL]
        queued = [j for j in members if j is not current]
        running = [j for j in members if j is current]
        cancelled = []
        for member in queued + running:
            try:
                cancelled.append(self.cancel(member.id))
            except ApiError:
                continue                          # finished in the gap; nothing to stop
        return cancelled

    # -- worker ------------------------------------------------------------

    def _publish(self, job: Job) -> None:
        # A job frame carries what the job is *about*, not just that it moved: the UI
        # marks rows busy from the frame it hears first, before any refetch lands.
        self.bus.publish(job.project, job_event(job.id, job.status, job.error,
                                                kind=job.kind, project=job.project,
                                                uids=list(job.payload.get("uids") or []),
                                                batch=job.batch))

    def _work(self) -> None:
        while not self._stop.is_set():
            job_id = self._queue.get()
            if self._stop.is_set():
                break                     # `stop` pushes a sentinel to wake us here
            if not job_id:
                continue
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None or job.status != QUEUED:
                    continue              # cancelled while it sat in the queue
                job.status, job.started = RUNNING, time.time()
                self._current = job
            self._publish(job)
            try:
                job.result = self.runner.run(job, self._emitter(job))
                status, error = (CANCELLED, None) if job.cancelling else (DONE, None)
            except JobCancelled:
                status, error = CANCELLED, None
            except Exception as exc:
                status, error = FAILED, f"{type(exc).__name__}: {exc}"
            finally:
                with self._lock:
                    self._current = None
            job.status, job.error, job.finished = status, error, time.time()
            if status == DONE:
                job.progress = 1.0
            self._publish(job)

    def _emitter(self, job: Job) -> Emit:
        """The runner's channel out: keeps the job's own summary in step with the bus."""
        def emit(event: dict[str, Any]) -> None:
            if event.get("type") == "stage":
                job.stage = event.get("stage") or job.stage
                if event.get("progress") is not None:
                    job.progress = event["progress"]
                if event.get("message"):
                    job.message = event["message"]
            self.bus.publish(job.project, {**event, "job": job.id})
        return emit


class JobCancelled(Exception):
    """Raised by a runner when the user cancelled the job it was running."""
