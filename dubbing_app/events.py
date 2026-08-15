"""The NDJSON progress bus: one JSON object per line, `application/x-ndjson`.

Frames, per docs/APP_ARCHITECTURE.md:

    {"type":"stage","stage":"tts","status":"running","progress":0.42,"message":"clip 31/74"}
    {"type":"segment","uid":"...","field":"tts","status":"done"}
    {"type":"job","id":"...","status":"queued|running|done|failed","error":null}
    {"type":"log","level":"info","message":"..."}

Plus `{"type":"heartbeat"}` every 15 s a keepalive, not an event, so proxies
and sleeping laptops do not silently drop the stream.

**Errors travel as a frame.** By the time a run fails the response has been
streaming for minutes and its status line is long gone, so a failure is a
`job`/`log` frame; the HTTP status says nothing about the run.

Producers are the job worker *thread*; consumers are asyncio request handlers.
The handoff is `loop.call_soon_threadsafe`, captured when the subscriber is
created, which is the only thread-safe way into an `asyncio.Queue`.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator

HEARTBEAT_SECONDS = 15.0
# A client that stops reading must not grow the server without bound. Past this,
# the oldest frames are dropped and a marker frame says so.
MAX_PENDING = 2048

ALL = "*"


def encode(event: dict[str, Any]) -> bytes:
    return (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")


def stage_event(stage: str, status: str, progress: float | None = None,
                message: str | None = None) -> dict[str, Any]:
    return {"type": "stage", "stage": stage, "status": status,
            "progress": progress, "message": message}


def segment_event(uid: str, field: str, status: str) -> dict[str, Any]:
    return {"type": "segment", "uid": uid, "field": field, "status": status}


def job_event(job_id: str, status: str, error: str | None = None,
              **extra: Any) -> dict[str, Any]:
    return {"type": "job", "id": job_id, "status": status, "error": error, **extra}


def log_event(message: str, level: str = "info") -> dict[str, Any]:
    return {"type": "log", "level": level, "message": message}


class Subscription:
    def __init__(self, bus: "EventBus", project: str, loop: asyncio.AbstractEventLoop):
        self.bus = bus
        self.project = project
        self.loop = loop
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.dropped = 0

    def offer(self, event: dict[str, Any]) -> None:
        """Called from any thread."""
        try:
            self.loop.call_soon_threadsafe(self._put, event)
        except RuntimeError:
            pass                     # loop already closed; the reader is gone

    def _put(self, event: dict[str, Any]) -> None:
        while self.queue.qsize() >= MAX_PENDING:
            try:
                self.queue.get_nowait()
                self.dropped += 1
            except asyncio.QueueEmpty:
                break
        self.queue.put_nowait(event)

    def close(self) -> None:
        self.bus.unsubscribe(self)


class EventBus:
    """Fan-out of events to whoever is watching a project (or `*`, everything)."""

    def __init__(self) -> None:
        self._subs: list[Subscription] = []

    def subscribe(self, project: str) -> Subscription:
        sub = Subscription(self, project, asyncio.get_running_loop())
        self._subs.append(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        if sub in self._subs:
            self._subs.remove(sub)

    def publish(self, project: str | None, event: dict[str, Any]) -> None:
        """Thread-safe. `project=None` means "everyone", e.g. queue-wide news."""
        event = {"t": round(time.time(), 3), **event}
        for sub in list(self._subs):
            if project is None or sub.project in (project, ALL):
                sub.offer(event)

    def subscriber_count(self, project: str | None = None) -> int:
        return sum(1 for s in self._subs if project is None or s.project == project)


async def stream(sub: Subscription, *, prelude: list[dict[str, Any]] | None = None,
                 heartbeat: float | None = None) -> AsyncIterator[bytes]:
    """The response body: prelude, then frames, then a heartbeat whenever idle."""
    heartbeat = heartbeat or HEARTBEAT_SECONDS      # read late, so tests can shorten it
    try:
        for event in prelude or []:
            yield encode(event)
        while True:
            try:
                event = await asyncio.wait_for(sub.queue.get(), timeout=heartbeat)
            except asyncio.TimeoutError:
                yield encode({"type": "heartbeat", "t": round(time.time(), 3)})
                continue
            if sub.dropped:
                yield encode(log_event(f"dropped {sub.dropped} frames (client too slow)",
                                       level="warning"))
                sub.dropped = 0
            yield encode(event)
    except asyncio.CancelledError:
        raise
    except Exception as exc:                       # never a bare 500 mid-stream
        yield encode(log_event(f"event stream failed: {type(exc).__name__}: {exc}",
                               level="error"))
    finally:
        sub.close()
