"""Pipeline runs launched as subprocesses, with pollable logs.

One `python -m dubbing …` process per job. The pipeline is already resumable and
writes its own state to the manifest, so a job needs to keep nothing but its log
tail and exit code a server restart loses the log, not the work.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import uuid
from collections import deque
from pathlib import Path
from typing import Any

from dubbing import tools

from .runs import REPO_ROOT

LOG_LINES = 400


class Job:
    def __init__(self, cmd: list[str], run: str | None, label: str) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.cmd = cmd
        self.run = run
        self.label = label
        self.log: deque[str] = deque(maxlen=LOG_LINES)
        self.returncode: int | None = None
        # Detached (`tools.spawn_kwargs`), so cancel can end the whole tree: a
        # leader-only terminate leaves ffmpeg encoding and holding this stdout
        # pipe, and `_pump` then never sees EOF — a job reported "running"
        # forever over work that was asked to stop.
        self.proc = subprocess.Popen(
            cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, **tools.spawn_kwargs(),
        )
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.log.append(line.rstrip("\n"))
        self.returncode = self.proc.wait()

    @property
    def status(self) -> str:
        if self.returncode is None:
            return "running"
        return "done" if self.returncode == 0 else "failed"

    def state(self, tail: int = 60) -> dict[str, Any]:
        return {
            "id": self.id, "run": self.run, "label": self.label,
            "status": self.status, "returncode": self.returncode,
            "cmd": self.cmd, "log": list(self.log)[-tail:],
        }

    def cancel(self) -> None:
        if self.returncode is None:
            tools.terminate_tree(self.proc, hard=False)


_JOBS: dict[str, Job] = {}


# The `python -m dubbing` flag each recorded option is spelled with.
OPT_FLAGS = {"genre": "--genre", "register": "--register", "transcript": "--transcript",
             "tts_model": "--tts-model", "device": "--device", "captions": "--captions"}


def dub_command(source: str, out: Path, *, src: str = "he", tgt: str = "en",
                duration: float | None = None, force: str | None = None,
                opts: dict[str, Any] | None = None,
                extra: list[str] | None = None) -> list[str]:
    """The command line that re-runs `out`.

    `opts` is what the run recorded (`runs.recorded_opts`) and every one of them
    has to be passed back: an option the command omits changes that stage's
    fingerprint, and an upstream fingerprint change resets the stage rather than
    resuming it a `--genre movie` run re-run without `--genre` loses its
    segments, and with them every edit made in this editor.
    """
    cmd = [sys.executable, "-m", "dubbing", source, "-o", str(out), "--src", src, "--tgt", tgt]
    if duration:
        cmd += ["--duration", str(duration)]
    for key, flag in OPT_FLAGS.items():
        if (opts or {}).get(key) is not None:
            cmd += [flag, str(opts[key])]
    if (opts or {}).get("dub_foreign"):
        cmd += ["--dub-foreign"]
    if force:
        cmd += ["--force", force]
    return cmd + list(extra or [])


def launch(cmd: list[str], run: str | None, label: str) -> Job:
    job = Job(cmd, run, label)
    _JOBS[job.id] = job
    return job


def get(job_id: str) -> Job:
    return _JOBS[job_id]


def listing() -> list[dict[str, Any]]:
    return [j.state(tail=1) for j in reversed(list(_JOBS.values()))]
