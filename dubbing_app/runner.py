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
is lost. Two things are reshaped on the way, and neither drops anything: an
identical line repeated back to back arrives as one frame that names the count
(`LineRuns`), and a traceback the child fenced for the purpose stays in this
process's own stderr instead of being painted into the user's job log. The raw
lines are printed here either way, so `journalctl`, a terminal, or whoever is
reading the server's output still sees exactly what the child wrote.
"""

from __future__ import annotations

import json
import os
import re
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

# The fence `dubbing_app.worker` prints a traceback inside. Declared here rather
# than there because this module is imported by the server on startup and
# `worker` is not: the server must not pay for `ops`' imports to know two strings.
TRACEBACK_OPEN = "--- traceback, for a bug report ---"
TRACEBACK_CLOSE = "--- end of traceback ---"


class LineRuns:
    """Turns a run of identical log lines into one frame that says how many.

    The last line of defence against the log a user filed as an issue: four
    identical lines per segment, then the same four again for the next one. The
    pipeline's own stages are being fixed where the repetition is made (see
    `dubbing/tts.py`), but this catches the same shape wherever else it turns up,
    including in stages nobody has looked at yet.

    A repeated line is held rather than sent, so the count is on the frame the
    user reads instead of arriving after it. That means the *last* line of a run
    is delayed until something different happens or the stream ends, which
    `flush` guarantees. Nothing is dropped: a line that repeats once is still one
    line, and the raw text is printed to this process's stderr regardless.
    """

    def __init__(self, emit: Any) -> None:
        self.emit = emit
        self.held: str | None = None
        self.count = 0

    def offer(self, line: str) -> None:
        if line == self.held:
            self.count += 1
            return
        self.flush()
        self.held, self.count = line, 1

    def flush(self) -> None:
        if self.held is None:
            return
        line, count = self.held, self.count
        self.held, self.count = None, 0
        self.emit(log_event(line if count == 1 else f"{line}  (x{count})"))


# The platform split (detach a child, end its whole tree) lives in
# `dubbing.tools` now, where the editor, the installer and the translator can
# reach it too; re-exported unchanged because this module is where the tests
# and the callers learned the names.
from dubbing.tools import (  # noqa: E402, F401  (re-exports)
    CREATE_NEW_PROCESS_GROUP, spawn_kwargs, terminate_tree)


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
        proc = self._spawn()
        with self._lock:
            self._procs[job.id] = proc

        errors: list[str] = []
        stderr_thread = threading.Thread(
            target=self._pump_stderr, args=(proc, emit, errors),
            name=f"job-{job.id}-stderr", daemon=True)
        stderr_thread.start()

        try:
            result, code = self._drive(proc, job, emit)
        finally:
            stderr_thread.join(timeout=2.0)
            with self._lock:
                self._procs.pop(job.id, None)
        return self._outcome(job, result, code, errors)

    def _spawn(self) -> subprocess.Popen:
        """The job child, detached and speaking UTF-8 on all three streams."""
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = os.pathsep.join(
            [str(self.cwd)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        # The protocol is NDJSON carrying Hebrew, and a Windows child's stdio
        # defaults to the ANSI code page — which cannot encode it, so the child
        # would die inside `print` rather than in anything it was asked to do.
        env["PYTHONIOENCODING"] = "utf-8"
        return subprocess.Popen(
            [self.python, "-m", self.module],
            cwd=str(self.cwd), env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            # Detached: the pipeline spawns yt-dlp and ffmpeg, and cancelling has
            # to take those with it rather than orphan a re-encode.
            **spawn_kwargs(),
        )

    def _drive(self, proc: subprocess.Popen, job: Job,
               emit: Emit) -> tuple[dict[str, Any] | None, int]:
        """Hand the child its spec, then relay stdout until it exits.

        Returns the child's one `result` frame (or None) and its exit code.
        Every other frame goes straight to `emit`; anything that is not JSON at
        all is still forwarded, as a log line, so nothing the child said is lost.
        """
        assert proc.stdin is not None and proc.stdout is not None
        result: dict[str, Any] | None = None
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
        return result, proc.wait()

    @staticmethod
    def _outcome(job: Job, result: dict[str, Any] | None, code: int,
                 errors: list[str]) -> Any:
        """What the job returned, or the exception that says why it did not.

        Cancellation is asked about first: a killed child exits non-zero, and
        reporting that as a failure would put an error bar over a stop the user
        asked for.
        """
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
        terminate_tree(proc, hard=False)
        killer = threading.Timer(TERM_GRACE, self._kill, args=(proc,))
        killer.daemon = True
        killer.start()

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        if proc.poll() is None:
            terminate_tree(proc, hard=True)

    @staticmethod
    def _pump_stderr(proc: subprocess.Popen, emit: Emit, errors: list[str]) -> None:
        """Every stderr line: to this process's log raw, to the UI readable.

        `errors` is the tail `_outcome` quotes when a child dies without saying
        why, so traceback frames are kept out of it too: three lines of `File
        "...", line 812` is the least informative possible answer to "what went
        wrong", and the sentence that *is* the answer is the line above them.
        """
        assert proc.stderr is not None
        runs = LineRuns(emit)
        in_traceback = False
        for raw in proc.stderr:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            print(line, file=sys.stderr)          # the server's own log, raw
            if line.strip() == TRACEBACK_OPEN:
                runs.flush()
                in_traceback = True
                continue
            if line.strip() == TRACEBACK_CLOSE:
                in_traceback = False
                continue
            if in_traceback:
                continue
            errors.append(line.strip())
            del errors[:-50]
            event = parse_stderr(line)
            if event:
                runs.flush()
                emit(event)
            else:
                runs.offer(line[:500])
        runs.flush()
