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


# ---------------------------------------------------------------------------
# uv
# ---------------------------------------------------------------------------
#
# The uv lookup lives here, beside `resolve_tool`, rather than in
# `dubbing_app/setup.py` where it was written, and the move is not tidying. Two
# callers need it and they are on opposite sides of a one-way import:
# `dubbing_app.setup` draws the Setup screen's uv row, and `dubbing.translate`
# spawns the CUDA translation worker with `uv run --project translator`, which
# is every non-Mac dub. `dubbing` is the headless pipeline and must never import
# `dubbing_app` (the app imports the pipeline, and a cycle would make `python -m
# dubbing` depend on a server it does not have), so the shared answer has to sit
# somewhere both can reach. This module already is that place for ffmpeg and
# sox. `dubbing_app.setup` re-exports every name below, so the app's spelling
# (`setup.find_uv`) is unchanged and there is still exactly one implementation.

# Set by `runner/process.rs` on the server's environment, and by the desktop
# shell generally: on a packaged install uv exists only as the Tauri sidecar
# next to the app binary, which no PATH and no home directory mentions.
UV_PATH_ENV = "DUBSTUDIO_UV_PATH"

# Absolute paths worth probing before PATH, per platform, copied from
# `workspace.rs`'s three `UV_FALLBACKS` constants. Windows genuinely has none:
# winget and uv's own installer both land on PATH or in the per-user
# `.local\bin` the tail of `find_uv` checks, so an empty tuple there is the
# answer rather than a gap.
UV_FALLBACKS: dict[str, tuple[str, ...]] = {
    "darwin": ("/opt/homebrew/bin/uv", "/usr/local/bin/uv"),
    "linux": ("/usr/local/bin/uv", "/usr/bin/uv",
              "/home/linuxbrew/.linuxbrew/bin/uv"),
    "win32": (),
}

# What Windows treats as executable when a name is typed without a suffix, for
# the machines that do not say. The real value comes from the environment; this
# is the documented default, and it is here so the lookup behaves the same in a
# test as it does on a machine whose PATHEXT was never touched.
DEFAULT_PATHEXT = ".COM;.EXE;.BAT;.CMD;.VBS;.VBE;.JS;.JSE;.WSF;.WSH;.MSC"


def uv_exe(platform: str | None = None) -> str:
    """The binary's name, which is the only part of this lookup spelled
    differently per platform (`workspace.rs`: `UV_EXE`)."""
    return "uv.exe" if platform_key(platform) == "win32" else "uv"


def uv_fallbacks(platform: str | None = None) -> tuple[str, ...]:
    return UV_FALLBACKS[platform_key(platform)]


def uv_home(env: dict[str, str] | None = None) -> Path | None:
    """The user's home directory, resolved the way `workspace.rs` resolves it:
    `HOME`, then `USERPROFILE`.

    Not `Path.home()`, and the difference is the whole point. Python's
    `expanduser` reads `USERPROFILE` on Windows and `HOME` everywhere else, so
    the two agree on every real machine. The Rust side reads `HOME` *first* on
    all three, which is what a Git-Bash or MSYS shell sets. Asking
    the same two variables in the same order is what makes "the shell and the
    server look in the same place" a fact rather than a hope. None when neither
    is set, exactly as the Rust returns None: a home-relative guess with no
    home is not a path.
    """
    environ = os.environ if env is None else env
    raw = environ.get("HOME")
    if raw is None:
        raw = environ.get("USERPROFILE")
    return Path(raw) if raw else None


def path_names(exe: str, env: dict[str, str] | None = None) -> tuple[str, ...]:
    """The filenames worth trying inside one PATH directory, in order.

    Keyed off the name rather than off `sys.platform`, and that is deliberate
    twice over. `uv_exe()` has already decided the platform question and spelled
    its answer into the name, so asking again would be two decisions where there
    is one; and a bare `"uv"` on a machine that wants `"uv.exe"` is a lookup
    nobody meant to ask for. It also means every branch here can be tested from
    a Mac, which is the only way any of it gets tested at all.

    Off Windows that is just `("uv",)`: a POSIX executable is its own name.

    For the `.exe` spelling: `uv.exe` first, because that is what astral ships
    and what `workspace.rs` tries first, so a directory holding several
    resolves the same way on both sides. Then every suffix this machine's
    PATHEXT names, which is the rung that was missing: Scoop, Chocolatey and
    most corporate wrappers install uv as a `uv.cmd` or `uv.bat` shim, and a
    scan that looked for `uv.exe` and `uv` alone walked straight past a working
    uv, reported the row not-found, and offered a button whose whole reason for
    existing (`install.uv_install_dir`) is to avoid ending up with two uvs that
    disagree. Then the bare name last, for the extensionless Git-Bash-style
    shim the Rust also looks for.

    A superset of `workspace.rs`'s two names, deliberately and in the safe
    direction: the worst a Python-only rung can do is find a real uv the shell
    would have missed, and the shell's own miss falls through to the bundled
    sidecar rather than to a second install.
    """
    if not exe.lower().endswith(".exe"):
        return (exe,)
    environ = os.environ if env is None else env
    bare = exe[:-4]
    names = [exe]
    raw = environ.get("PATHEXT") or DEFAULT_PATHEXT
    # Split on `;` and not `os.pathsep`: PATHEXT is a Windows variable and is
    # `;`-separated wherever it is read, while `os.pathsep` is `:` on the Mac
    # this gets tested from, where the split silently produced one suffix.
    for suffix in raw.split(";"):
        suffix = suffix.strip()
        if not suffix.startswith("."):
            continue
        candidate = bare + suffix.lower()
        if candidate not in names:
            names.append(candidate)
    names.append(bare)
    return tuple(names)


def uv_on_path(exe: str, env: dict[str, str] | None = None) -> str | None:
    """`which`, near enough, and near enough to `workspace.rs`'s `find_in_path`.

    Written out rather than handed to `shutil.which` because the two disagree
    on Windows in the direction that matters: `shutil.which("uv")` appends the
    PATHEXT suffixes and never tries the bare name, and `shutil.which("uv.exe")`
    never tries anything else, while the Rust tries `uv.exe` then `uv` inside
    each directory in turn. A Git-Bash-style extensionless shim is worth
    finding, and a directory holding both must resolve the same way on both
    sides or the shell and the server disagree about which uv this machine has.

    So this is the union rather than either one: `shutil.which`'s PATHEXT (see
    `path_names`) *and* the Rust's bare name, which is what closes the gap
    `resolve_tool` never had. `resolve_tool` ends at `shutil.which`, so ffmpeg
    and sox have honoured PATHEXT all along; uv was the one lookup in this repo
    that did not, and that is the whole of the difference that is left between
    them.
    """
    environ = os.environ if env is None else env
    raw = environ.get("PATH")
    if not raw:
        return None
    names = path_names(exe, environ)
    for directory in raw.split(os.pathsep):
        if not directory:
            continue
        for name in names:
            candidate = Path(directory) / name
            if candidate.is_file():
                return str(candidate)
    return None


def find_uv(platform: str | None = None) -> str | None:
    """Where `uv` actually is, not just where PATH says.

    Five rungs, the same five and in the same order as `workspace.rs`:

    1. `DUBSTUDIO_UV_PATH`. On a packaged install this is not a courtesy, it is
       *the* answer: every bundle ships uv as a Tauri `externalBin` sidecar next
       to the app binary, and `runner/process.rs` hands the resolved path down
       on the server's environment when it spawns it. The Rust side's second
       rung resolves that sidecar from `current_exe`, and this process cannot:
       its `sys.executable` is the venv's Python, nowhere near the shell. So
       the env var is how the sidecar reaches Python, and it is why a desktop
       user has uv by construction and should never see this row red.
    2. (the sidecar, which is rung 1 here; see above)
    3. this platform's literal install paths, `uv_fallbacks()`.
    4. PATH, `.exe` first on Windows and then whatever PATHEXT names.
    5. `~/.local/bin` and `~/.cargo/bin`, where uv's own installer and
       `cargo install uv` put it. A GUI process inherits almost none of the
       user's shell PATH (a Finder-launched .app, a Start menu shortcut), so
       these two literals catch the most common standalone install there is:
       the one `install-server.sh` and `install-server.ps1` perform.
    """
    override = (os.environ.get(UV_PATH_ENV) or "").strip()
    if override and Path(override).is_file():
        return override
    for candidate in uv_fallbacks(platform):
        if Path(candidate).is_file():
            return candidate
    exe = uv_exe(platform)
    found = uv_on_path(exe)
    if found:
        return found
    home = uv_home()
    if home is None:
        return None
    for candidate in (home / ".local" / "bin" / exe, home / ".cargo" / "bin" / exe):
        if candidate.is_file():
            return str(candidate)
    return None


def install_hint(id_: str, platform: str | None = None) -> str:
    """A sentence naming the command, or the empty string when there is none."""
    cmd = command(id_, platform)
    return f"install it with `{cmd}`" if cmd else ""


__all__ = ["DEFAULT_PATHEXT", "RECIPES", "TOOLS_DIR_ENV", "UNATTENDED", "UV_FALLBACKS",
           "UV_PATH_ENV", "auto_installers", "command", "find_uv", "install_hint",
           "path_names", "platform_key", "recipe", "recipes", "resolve_tool",
           "tools_bin", "unattended", "utf8_stdio", "uv_exe", "uv_fallbacks",
           "uv_home", "uv_on_path"]
