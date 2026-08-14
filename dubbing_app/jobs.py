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

from .errors import invalid, not_found
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

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "project": self.project,
                "status": self.status, "error": self.error, "result": self.result,
                "created": self.created, "started": self.started, "finished": self.finished,
                "stage": self.stage, "progress": self.progress, "message": self.message,
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

    def submit(self, kind: str, project: str, payload: dict[str, Any] | None = None) -> Job:
        if kind not in KINDS:
            raise invalid(f"unknown job kind {kind!r}")
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, project=project,
                  payload=dict(payload or {}))
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

    # -- worker ------------------------------------------------------------

    def _publish(self, job: Job) -> None:
        self.bus.publish(job.project, job_event(job.id, job.status, job.error,
                                                kind=job.kind, project=job.project))

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
