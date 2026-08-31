"""The command-line tools the pipeline shells out to, and how *this* machine
installs them.

`ffmpeg` and `sox` are the same two dependencies on every platform and three
different sentences: `brew install ffmpeg` on a Mac, a `winget` id on Windows,
`sudo apt-get install` on Linux. `uv` is here too, and it is the one tool with
a *missing* row rather than three: no distribution ships it in apt, so Linux
gets no line at all instead of a `sudo apt-get install uv` that would fail on
every machine that read it. What Linux gets instead is the route
`dubbing_app/install.py` calls `uv_release_route`, which needs no package
manager anywhere. The table lives here rather than in
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

import os
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# Where the app puts binaries it installed itself (the brewless-Mac static
# ffmpeg). Inside the workspace, not in /usr/local: no sudo, no PATH edits, and
# deleting the workspace deletes everything the app ever added to this machine.
TOOLS_DIR_ENV = "DUBSTUDIO_TOOLS_DIR"

# platform key → tool id → the argv that installs it.
#
# `uv` is deliberately absent from the Linux row. Homebrew and winget both
# publish it, so `brew install uv` and `winget install --id astral-sh.uv` are
# real lines a user can type; Debian and Ubuntu do not, so `sudo apt-get install
# uv` would be a sentence this screen printed and no machine could run. A row
# with no command is honest about that, and the button on Linux comes from
# somewhere else entirely (`install.uv_release_route`, the official release
# archive, which needs no package manager and no password on any platform).
RECIPES: dict[str, dict[str, tuple[str, ...]]] = {
    "darwin": {
        "ffmpeg": ("brew", "install", "ffmpeg"),
        "sox": ("brew", "install", "sox"),
        "uv": ("brew", "install", "uv"),
    },
    "win32": {
        "ffmpeg": ("winget", "install", "--id", "Gyan.FFmpeg", "-e",
                   "--accept-source-agreements", "--accept-package-agreements"),
        "sox": ("winget", "install", "--id", "ChrisBagwell.SoX", "-e",
                "--accept-source-agreements", "--accept-package-agreements"),
        "uv": ("winget", "install", "--id", "astral-sh.uv", "-e",
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


def utf8_stdio(streams: Iterable[Any] | None = None) -> None:
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


def tools_bin() -> Path:
    """The workspace-local bin dir, whether or not anything is in it yet.

    `REPO_ROOT/tools/bin` — and in the packaged app REPO_ROOT *is* the
    workspace, because provisioning copies this source tree there. The env
    override exists for the same reason `DUBSTUDIO_UV_PATH` does: a test (or an
    unusual deployment) points it somewhere real without touching the checkout.
    """
    override = (os.environ.get(TOOLS_DIR_ENV) or "").strip()
    return Path(override) if override else REPO_ROOT / "tools" / "bin"


def resolve_tool(name: str) -> str | None:
    """Where `name` actually is, or None. The one lookup every call site uses.

    Order: a per-tool env override (`DUBSTUDIO_FFMPEG=/path/to/ffmpeg`), then
    the workspace tools dir, then PATH. The middle step is the point: a static
    build the app installed for a brewless Mac lives in `tools/bin`, and a bare
    `subprocess.run(["ffmpeg", ...])` would never find it — PATH belongs to the
    shell the app was not launched from. `shutil.which` stays last so a real
    Homebrew/winget install keeps winning exactly when it is on PATH.
    """
    override = (os.environ.get(f"DUBSTUDIO_{name.upper()}") or "").strip()
    if override and Path(override).is_file():
        return override
    exe = name + (".exe" if platform_key() == "win32" else "")
    local = tools_bin() / exe
    if local.is_file() and os.access(local, os.X_OK):
        return str(local)
    return shutil.which(name)


def install_hint(id_: str, platform: str | None = None) -> str:
    """A sentence naming the command, or the empty string when there is none."""
    cmd = command(id_, platform)
    return f"install it with `{cmd}`" if cmd else ""


__all__ = ["RECIPES", "TOOLS_DIR_ENV", "UNATTENDED", "auto_installers", "command",
           "install_hint", "platform_key", "recipe", "recipes", "resolve_tool",
           "tools_bin", "unattended", "utf8_stdio"]
