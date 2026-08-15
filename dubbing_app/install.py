"""Installing the missing command-line tools from inside the app.

The Setup screen's job is to say what this machine is missing. For a model that
is enough the check's detail line carries the command, and nothing else could
honestly be offered for a ten-gigabyte download that wants a Hugging Face login
and half an hour. For `ffmpeg` and `sox` it is not: they are one `brew install`
away, the user is looking at a red row that says so, and asking them to find a
terminal is asking them to leave the app to type a line the app already knows.

Four rules shape this module, each of them a refusal:

* **The id maps to a hardcoded argv.** Nothing the client sends is ever executed
  or interpolated the request body is a strict model with one field, and that
  field is looked up in `INSTALLERS`. An id that is not in the table is a 400
  that says how to install it by hand; a *model* id is refused the same way,
  because auto-downloading gigabytes behind a spinner is not an install button.
* **At most one install at a time, process-wide.** Not for memory (a `brew` runs
  nothing of ours) but because two package managers writing the same prefix is a
  broken prefix, and because the screen can only honestly show one spinner.
  Tracked here, in this object, and not in `JobQueue`: a job is a pipeline stage
  in a subprocess that loads models, and putting a `brew` behind a render would
  make the user wait an hour to fix the thing that is blocking them.
* **No streaming channel.** The event bus is per project and setup has no
  project. An install is minutes long, so `GET /api/setup/install` returning the
  last N lines and a 2 s poll is the whole design a second transport for a
  progress bar nobody watches character by character is not worth its bugs.
* **The check is re-probed when the process exits, not trusted.** `brew` exiting
  0 is not the same claim as "`ffmpeg` is now on PATH" a bottle can install to
  a prefix this process cannot see. The status response carries a fresh check
  row, produced by the same probe `/api/setup` uses.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from collections import deque
from typing import Any, Callable

from .errors import busy, invalid

# id → the argv to run. The only executable strings in this feature.
INSTALLERS: dict[str, tuple[str, ...]] = {
    "ffmpeg": ("brew", "install", "ffmpeg"),
    "sox": ("brew", "install", "sox"),
}

# argv[0] → what to say when it is not on PATH. The tool that installs the tools
# cannot itself be installed from here; naming the one URL is the whole answer.
MANAGERS: dict[str, str] = {
    "brew": ("Homebrew is not on this machine, and it is what installs {tool}. "
             "Get it from https://brew.sh, then re-check or install {tool} by "
             "hand with `{command}` once brew is there."),
}

# How much of the output the status response carries. An install is chatty and
# the UI shows one line of it; the rest exists for the failure case, where the
# reason is usually the last twenty lines and never the first.
TAIL_LINES = 200

# A hung package manager must not hold the one slot forever. Half an hour is
# past any real `brew install` and well short of "the user gave up".
TIMEOUT = 1800.0

Probe = Callable[[str], dict[str, Any] | None]


def manual_command(id_: str, recipes: dict[str, tuple[str, ...]] | None = None) -> str | None:
    """What the user would type for `id_`, or None if the app has no recipe."""
    argv = (recipes if recipes is not None else INSTALLERS).get(id_)
    return " ".join(argv) if argv else None


class Installer:
    """The one install slot, and the process behind it.

    `probe` re-reads a single check by id (`setup.probe`); `recipes` overrides
    the module table, which is how the tests get a stub argv instead of a real
    package manager. Left unset it reads `INSTALLERS` *at call time*, so
    monkeypatching either the name or its contents works.
    """

    def __init__(self, probe: Probe, recipes: dict[str, tuple[str, ...]] | None = None,
                 spawn: Callable[..., subprocess.Popen] | None = None):
        self._probe = probe
        self._recipes = recipes
        self._spawn = spawn or subprocess.Popen
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._tail: deque[str] = deque(maxlen=TAIL_LINES)
        self._running = False
        self._id: str | None = None
        self._ok: bool | None = None
        self._error: str | None = None
        self._check: dict[str, Any] | None = None
        self._started: float | None = None
        self._finished: float | None = None

    @property
    def recipes(self) -> dict[str, tuple[str, ...]]:
        return self._recipes if self._recipes is not None else INSTALLERS

    # -- api ---------------------------------------------------------------

    def start(self, id_: str) -> dict[str, Any]:
        """Begin installing `id_`. Raises the 400/409 the UI renders."""
        argv = self.recipes.get(id_)
        if argv is None:
            raise invalid(self._refusal(id_))
        manager = argv[0]
        if shutil.which(manager) is None:
            template = MANAGERS.get(manager,
                                    "`{manager}` is not on PATH install `{command}` by hand.")
            raise invalid(template.format(tool=id_, command=" ".join(argv), manager=manager))

        with self._lock:
            if self._running:
                raise busy(f"an install is already running ({self._id}); "
                           "one at a time wait for it to finish")
            self._running = True
            self._id = id_
            self._ok = None
            self._error = None
            self._check = None
            self._started = time.time()
            self._finished = None
            self._tail.clear()
            self._tail.append("$ " + " ".join(argv))
            self._thread = threading.Thread(target=self._run, args=(id_, tuple(argv)),
                                            name=f"install-{id_}", daemon=True)
            self._thread.start()
        return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {"running": self._running, "id": self._id, "ok": self._ok,
                    "error": self._error, "tail": list(self._tail), "check": self._check,
                    "started": self._started, "finished": self._finished}

    def wait(self, timeout: float = 5.0) -> bool:
        """Join the install thread. For tests and shutdown, never for a request."""
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    # -- the subprocess ----------------------------------------------------

    def _refusal(self, id_: str) -> str:
        offered = ", ".join(f"`{' '.join(a)}`" for a in self.recipes.values())
        return (f"{id_!r} cannot be installed from the app. The only installs it "
                f"runs are {offered or 'none'}. Everything else is by hand: the "
                "models download themselves on first use and the command is in "
                "that check's detail line run it in a terminal from the repo.")

    def _env(self) -> dict[str, str]:
        # Non-interactive, because there is no terminal to answer a prompt on and
        # a blocked read would hold the one slot until the timeout.
        return {**os.environ, "NONINTERACTIVE": "1", "HOMEBREW_NO_AUTO_UPDATE": "1",
                "HOMEBREW_NO_ANALYTICS": "1"}

    def _run(self, id_: str, argv: tuple[str, ...]) -> None:
        ok, error = False, None
        try:
            proc = self._spawn(list(argv), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               text=True, bufsize=1, env=self._env())
        except OSError as exc:
            self._finish(id_, False, f"{type(exc).__name__}: {exc}")
            return
        killer = threading.Timer(TIMEOUT, self._kill, args=(proc,))
        killer.daemon = True
        killer.start()
        try:
            for line in proc.stdout or ():
                self._line(line.rstrip("\n"))
            code = proc.wait()
            ok = code == 0
            if not ok:
                error = f"`{' '.join(argv)}` exited {code}"
        except Exception as exc:                      # a broken pipe, a killed child
            error = f"{type(exc).__name__}: {exc}"
        finally:
            killer.cancel()
        self._finish(id_, ok, error)

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        try:
            proc.kill()
        except OSError:
            pass

    def _line(self, text: str) -> None:
        with self._lock:
            self._tail.append(text)

    def _finish(self, id_: str, ok: bool, error: str | None) -> None:
        """Record the verdict and re-probe, because the exit code is a claim
        about the package manager, not about this machine's PATH."""
        check = None
        try:
            check = self._probe(id_)
        except Exception as exc:                      # a probe must never strand the slot
            error = error or f"could not re-check {id_}: {type(exc).__name__}: {exc}"
        if ok and check is not None and not check.get("ok"):
            ok = False
            error = error or (f"`{' '.join(self.recipes.get(id_, ()))}` succeeded but "
                              f"{id_} is still not there restart the app so it picks "
                              "up the new PATH.")
        with self._lock:
            self._running = False
            self._ok = ok
            self._error = error
            self._check = check
            self._finished = time.time()


__all__ = ["INSTALLERS", "Installer", "manual_command"]
