"""The command-line tools the pipeline shells out to, and how *this* machine
installs them.

`ffmpeg` and `sox` are the same two dependencies on every platform and three
different sentences: `brew install ffmpeg` on a Mac, a `winget` id on Windows,
`sudo apt-get install` on Linux. The table lives here rather than in
`dubbing_app/install.py` because both sides need it and neither owns it: the
headless pipeline's "ffmpeg not found" line and the app's install button must
name the same command, or one of them is lying about this machine.

Two facts per recipe, and the second one is what the app reads:

* the **argv** the user would type, and
* whether its package manager can run **unattended**. `brew` and `winget` can
  (given the non-interactive env `install.Installer` sets); `sudo apt-get`
  cannot — it wants a password on a terminal this app does not have, and a
  spinner in front of a hidden password prompt is a hang, not an install. So on
  Linux there is no button and the check's detail line carries the command,
  which is exactly what `dubbing_app/install.py`'s docstring says it does for
  everything it refuses to run.
"""

from __future__ import annotations

import sys

# platform key → tool id → the argv that installs it.
RECIPES: dict[str, dict[str, tuple[str, ...]]] = {
    "darwin": {
        "ffmpeg": ("brew", "install", "ffmpeg"),
        "sox": ("brew", "install", "sox"),
    },
    "win32": {
        "ffmpeg": ("winget", "install", "--id", "Gyan.FFmpeg", "-e",
                   "--accept-source-agreements", "--accept-package-agreements"),
        "sox": ("winget", "install", "--id", "ChrisBagwell.SoX", "-e",
                "--accept-source-agreements", "--accept-package-agreements"),
    },
    "linux": {
        "ffmpeg": ("sudo", "apt-get", "install", "-y", "ffmpeg"),
        "sox": ("sudo", "apt-get", "install", "-y", "sox"),
    },
}

# The package managers that finish without asking a question. Anything else is a
# command we print and never run.
UNATTENDED = frozenset({"brew", "winget"})


def platform_key(platform: str | None = None) -> str:
    """`sys.platform`, folded to the three keys `RECIPES` has.

    Anything unrecognised (freebsd, aix) is treated as Linux: a POSIX box with a
    package manager we do not drive, which is the honest default.
    """
    name = platform if platform is not None else sys.platform
    if name.startswith("win"):
        return "win32"
    if name == "darwin":
        return "darwin"
    return "linux"


def recipes(platform: str | None = None) -> dict[str, tuple[str, ...]]:
    """Every install recipe this platform has, tool id → argv."""
    return dict(RECIPES[platform_key(platform)])


def recipe(id_: str, platform: str | None = None) -> tuple[str, ...] | None:
    return RECIPES[platform_key(platform)].get(id_)


def command(id_: str, platform: str | None = None) -> str | None:
    """What the user would type to install `id_` here, or None."""
    argv = recipe(id_, platform)
    return " ".join(argv) if argv else None


def unattended(id_: str, platform: str | None = None) -> bool:
    """Whether the app may run this recipe itself, unwatched."""
    argv = recipe(id_, platform)
    return bool(argv) and argv[0] in UNATTENDED


def auto_installers(platform: str | None = None) -> dict[str, tuple[str, ...]]:
    """The recipes an app is allowed to run — the rest are printed, not run."""
    return {id_: argv for id_, argv in recipes(platform).items()
            if unattended(id_, platform)}


def utf8_stdio(streams=None) -> None:
    """Make this process's stdout/stderr speak UTF-8 whatever the console says.

    Windows still starts Python with the ANSI code page on stdio, and every
    stage here prints Hebrew: a source line in a log, a title, a path. A
    `print` that dies with UnicodeEncodeError takes down the run for the one
    thing that cannot be allowed to fail — saying what happened — and does it
    at a random stage, looking like anything but its cause. Idempotent, and a
    no-op wherever stdio is UTF-8 already (every Mac and Linux).
    """
    for stream in (streams if streams is not None else (sys.stdout, sys.stderr)):
        encoding = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
        if encoding in ("", "utf8") or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def install_hint(id_: str, platform: str | None = None) -> str:
    """A sentence naming the command, or the empty string when there is none."""
    cmd = command(id_, platform)
    return f"install it with `{cmd}`" if cmd else ""


__all__ = ["RECIPES", "UNATTENDED", "auto_installers", "command", "install_hint",
           "platform_key", "recipe", "recipes", "unattended", "utf8_stdio"]
