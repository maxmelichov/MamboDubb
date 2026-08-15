"""Runs a job as a child process, and turns its two output streams into events.

**Why a subprocess and not a thread.** Three reasons, all of them the one-job
rule wearing different hats:

1. *Cancel.* A Python thread cannot be interrupted; a process can be signalled.
   `--duration`-less runs are tens of minutes long, so "stop" has to mean stop.
2. *Memory.* Gemma 4 at 4-bit is ~11 GB live and MLX's buffer cache briefly holds
   twice that. Neither MLX nor torch reliably returns that to the OS inside a
   long-lived process; process exit does, unconditionally. On a 26 GB machine
   that is the difference between the next job running and the machine swapping.
3. *Latency.* The server never imports torch, MLX or faster-whisper, so it starts
   in milliseconds and a no-model edit is never behind a model's GIL.

**Where progress comes from.** Two channels, on purpose:

* stdout is the child's own NDJSON authoritative, structured, per-segment.
* stderr is the pipeline's existing human log. `dubbing.cli` already prints
  `[stage]`, `[stage] done in Ns` and `[stage] up to date`, and `translate`/`tts`
  already print `n/N` counters. Parsing those costs nothing and works for the
  full run, where the orchestration is `dubbing.cli`'s and the app has no hook
  inside it. The alternative threading a progress callback through nine stage
  functions is a change to `dubbing/` for the app's benefit, which is the fork
  docs/APP_ARCHITECTURE.md rules out.

Everything unparsed on stderr still reaches the UI as a `log` frame, so nothing
is lost.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from .events import log_event, stage_event
from .jobs import Emit, Job, JobCancelled

REPO_ROOT = Path(__file__).resolve().parents[1]

# `[tts]`, `[tts] done in 91s`, `[tts] up to date` dubbing/cli.py
_STAGE_RE = re.compile(r"^\[(\w+)\](?:\s+(up to date|done in [\d.]+s))?$")
# `  translate: 12/74`, `  tts: 30/74` dubbing/translate.py, dubbing/tts.py
_COUNT_RE = re.compile(r"^\s+(\w+):\s+(\d+)/(\d+)$")

TERM_GRACE = 8.0


def parse_stderr(line: str) -> dict[str, Any] | None:
    """One pipeline log line → one event, or None to pass it through as a log."""
    stage = _STAGE_RE.match(line)
    if stage:
        name, tail = stage.group(1), stage.group(2)
        if tail is None:
            return stage_event(name, "running", 0.0, None)
        return stage_event(name, "done", 1.0, tail)
    count = _COUNT_RE.match(line)
    if count:
        name, done, total = count.group(1), int(count.group(2)), int(count.group(3))
        if total:
            return stage_event(name, "running", min(1.0, done / total), f"{done}/{total}")
    return None


class SubprocessRunner:
    """Spawns `python -m dubbing_app.worker` and pumps its streams into `emit`."""

    def __init__(self, python: str | None = None, cwd: Path | None = None,
                 module: str = "dubbing_app.worker"):
        self.python = python or sys.executable
        self.cwd = Path(cwd or REPO_ROOT)
        self.module = module          # a seam: tests point this at a stub child
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    def spec(self, job: Job) -> dict[str, Any]:
        return {"kind": job.kind, "workdir": job.payload.get("workdir"),
                "payload": job.payload}

    def run(self, job: Job, emit: Emit) -> Any:
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = os.pathsep.join(
            [str(self.cwd)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        proc = subprocess.Popen(
            [self.python, "-m", self.module],
            cwd=str(self.cwd), env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
            # Its own process group: the pipeline spawns yt-dlp and ffmpeg, and
            # cancelling has to take those with it rather than orphan a re-encode.
            start_new_session=True,
        )
        with self._lock:
            self._procs[job.id] = proc

        errors: list[str] = []
        stderr_thread = threading.Thread(
            target=self._pump_stderr, args=(proc, emit, errors),
            name=f"job-{job.id}-stderr", daemon=True)
        stderr_thread.start()

        result: dict[str, Any] | None = None
        try:
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write(json.dumps(self.spec(job), ensure_ascii=False) + "\n")
            proc.stdin.flush()
            proc.stdin.close()
            for raw in proc.stdout:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except Exception:
                    emit(log_event(raw))
                    continue
                if event.get("type") == "result":
                    result = event
                    continue
                emit(event)
            code = proc.wait()
        finally:
            stderr_thread.join(timeout=2.0)
            with self._lock:
                self._procs.pop(job.id, None)

        if job.cancelling:
            raise JobCancelled(f"job {job.id} cancelled")
        if result is not None and not result.get("ok", True):
            raise RuntimeError(result.get("error") or "job failed")
        if code != 0:
            tail = " | ".join(errors[-3:]) or f"exit {code}"
            raise RuntimeError(f"job process exited {code}: {tail}")
        return (result or {}).get("data")

    def cancel(self, job: Job) -> None:
        with self._lock:
            proc = self._procs.get(job.id)
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
        killer = threading.Timer(TERM_GRACE, self._kill, args=(proc,))
        killer.daemon = True
        killer.start()

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                proc.kill()

    @staticmethod
    def _pump_stderr(proc: subprocess.Popen, emit: Emit, errors: list[str]) -> None:
        assert proc.stderr is not None
        for raw in proc.stderr:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            errors.append(line.strip())
            del errors[:-50]
            print(line, file=sys.stderr)          # the server's own log, too
            event = parse_stderr(line)
            emit(event if event else log_event(line[:500]))
