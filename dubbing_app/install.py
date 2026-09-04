"""Installing the missing tools — and downloading the missing models — from
inside the app.

The Setup screen's job is to say what this machine is missing. For `ffmpeg` and
`sox` the fix is one package-manager line away (`brew install` on a Mac,
`winget install` on Windows), the user is looking at a red row that says so,
and asking them to find a terminal is asking them to leave the app to type a
line the app already knows. Where that line needs a terminal anyway — `sudo
apt-get`, which asks for a password — there is no button and the detail line
carries the command; see `dubbing/tools.py`.

A machine with no package manager to drive gets a different route instead of a
refusal — a Mac with no Homebrew, a Windows with no winget, and *every* Linux
(its recipe is `sudo apt-get`, which wants a password on a terminal the app does
not have, so it is deliberately not an unattended one). "Get it from
https://brew.sh" is itself a terminal command plus an admin password, which is
everything the DMG promised to avoid, so for `ffmpeg` (and the `ffprobe` that
ships beside it) the button falls back to a pinned static build: the
`static-ffmpeg` wheel is `uv pip`-installed into this venv, its downloader
fetches the platform zip it ships a URL for — one wheel, a build for all three
platforms — and the two binaries are *copied*, not symlinked, into the
workspace's `tools/bin`, where `dubbing.tools.resolve_tool` looks before PATH.
Copied because the wheel is not in the lockfile and the next `uv sync` prunes
it; the binaries must outlive their installer. A real package manager stays the
route wherever there is one to drive.

The models used to be refused here on principle: auto-downloading gigabytes
behind a spinner is not an install button, and this docstring said so. That
principle was never about the gigabytes — it was about the *blindness*. What a
shipped app can offer honestly is different in every way that mattered: the
row says how big the download is before the button is pressed, the status
reports bytes done against bytes expected while it runs, `snapshot_download`
resumes a torn-off attempt instead of starting over, and the id still maps to
a server-side table (`setup.model_downloads()`) that names the repo and the
directory — nothing the client sends chooses either. A user who installed a
DMG so as never to open a terminal is not served by a detail line that ends in
`uv run hf download`; they are served by a button that says what it costs.

Four rules shape this module, each of them a refusal:

* **The id maps to a hardcoded action.** Nothing the client sends is ever
  executed or interpolated the request body is a strict model with one field,
  and that field is looked up in `INSTALLERS` (tools → argv) or in
  `setup.model_downloads()` (models → hub repo + local dir). An id in neither
  table is a 400 that says how to install it by hand, which is still where the
  Hebrew G2P ends up: it fetches its own weights from inside a package this
  environment either has or does not, and `uv sync` is the fix.

  Three ids are in neither table and have a button anyway, because their fix is
  not a snapshot. `uv` is the third and the newest: no distribution packages
  it, and on a Mac with no brew or a Windows with no winget there is no manager
  to drive either, so the button downloads astral's own release build for this
  machine and checks it against the SHA-256 published beside it
  (`install_uv`). It used to say uv could not be installed from here at all, on
  the argument below about package managers, which is an argument about brew
  and winget and about nothing else. uv is neither of them.

  Diarization's weights shipped with the app, so the button copies them back from the payload (or restores them from the checkout) and
  verifies the copy against SHA256SUMS; the gated upstream repo is never reached
  for, and no route here can ask the user for a Hugging Face account. Demucs
  fetches `htdemucs_ft` itself the first time a stems run asks for it, so the
  button runs that same fetch early (`demucs_argv`) rather than leaving the one
  row on the screen with no gesture at all, which is what it used to be: the
  only way to install it was to start a dub and sit through a silent mid-run
  download. Both are still hardcoded actions chosen by id; neither takes
  anything from the request.
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

import importlib
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

from dubbing import tools, uvrelease

from . import runner as runner_mod
from .errors import ApiError, busy, invalid

# id → the argv to run, for *this* platform. The only executable strings in this
# feature, and still a hardcoded table — `dubbing.tools` picks the row for the
# machine, nothing here is built from a request. A platform whose package
# manager cannot run unattended (Linux: `sudo apt-get` wants a password on a
# terminal the app does not have) contributes no rows at all, so the Setup
# screen shows no button and the check's detail line carries the command
# instead. That is the same refusal this module already makes for the models.
INSTALLERS: dict[str, tuple[str, ...]] = tools.auto_installers()

# argv[0] → what to say when it is not on PATH. The tool that installs the tools
# cannot itself be installed from here; naming the one URL is the whole answer.
MANAGERS: dict[str, str] = {
    "brew": ("Homebrew is not on this machine, and it is what installs {tool}. "
             "Get it from https://brew.sh, then re-check, or install {tool} by "
             "hand with `{command}` once brew is there."),
    "winget": ("The Windows Package Manager (winget) is not on this machine, and "
               "it is what installs {tool}. Get App Installer from the Microsoft "
               "Store, then re-check, or install {tool} by hand with `{command}` "
               "once winget is there."),
    "apt-get": ("`apt-get` is not on this machine: install {tool} with your "
                "distribution's package manager, or by hand with `{command}`."),
    "sudo": ("This machine installs {tool} with `{command}`, which asks for a "
             "password on a terminal the app does not have. Run it in a terminal "
             "and re-check."),
}

# How much of the output the status response carries. An install is chatty and
# the UI shows one line of it; the rest exists for the failure case, where the
# reason is usually the last twenty lines and never the first.
TAIL_LINES = 200

# A hung package manager must not hold the one slot forever. Half an hour is
# past any real `brew install` and well short of "the user gave up". Only the
# subprocess installs get this: a model download is *expected* to take longer
# than any brew, there is no child to kill, and killing it would throw away
# nothing anyway — `snapshot_download` resumes from the partial files.
TIMEOUT = 1800.0

# How long a queued item waits for the one slot when a row's own Install button
# has taken it, and how often it looks. Five minutes is longer than any tool
# install and shorter than a large download, which is the right shape: the thing
# actually worth waiting out is a `brew` or a restore, and a queue stuck behind
# somebody's hand-started 9.7 GB fetch should get on with the rest of the list.
SLOT_WAIT = 300.0
SLOT_POLL = 1.0

Probe = Callable[[str], dict[str, Any] | None]

# Which of the six routes an install actually took, recorded when the slot is
# claimed and read when it is released.
#
# It exists because `_finish` used to guess, and guessed from the wrong table. A
# row can have a `brew install` recipe *and* be installed by the static-build
# route (a Mac with no Homebrew is exactly that machine), so "is this id in
# `recipes`?" answered "package manager" for an install no package manager
# touched, and the failure sentence then named a command that never ran and
# offered a restart that could not have helped. What the message has to be about
# is what was done, and the only place that is known for certain is where it was
# decided.
MANAGER, STATIC, UV_ROUTE, RESTORE, DOWNLOAD, DEMUCS_ROUTE = (
    "manager", "static", "uv", "restore", "download", "demucs")

# The one wheel the brewless route installs, pinned: its job is to hand over a
# known ffmpeg build, and "whatever version resolved today" is not a known
# build. Bump deliberately, with the binaries it fetches re-checked.
STATIC_FFMPEG_SPEC = "static-ffmpeg==3.0"

# The diarization weights, which are the one install here that downloads nothing.
# Upstream (`segments.DIARIZATION_MODEL`) is gated: free and CC-BY-4.0, but with
# an accept-the-terms click and a read token in front of the files, and a token
# is exactly what this app promises never to ask for. So the restore route reads
# from the copy that already shipped: the 31 MB under `third_party/`, checked
# into the repo and staged into the desktop payload. It verifies that copy against the
# SHA256SUMS that travels beside it.
DIARIZATION_ID = "model.diarization"
# The tool that builds this project's environment, and the one row here whose
# install is neither a package manager nor a wheel: astral publishes a static
# binary per platform under a stable release URL with a `.sha256` beside it.
# `uv_release_route` decides when that is the route; `install_uv` is the route.
UV_ID = "uv"
# The releases page, and the triple table, and the checksum, and the archive
# unpacking: `dubbing.uvrelease`, which `scripts/stage_desktop_payload.py`
# also imports. This module used to carry its own copy of all four.
UV_RELEASES = uvrelease.UV_RELEASES
# The stem-separation weights, which are the one install here that is neither a
# package manager nor a snapshot: demucs fetches them itself the first time a
# `stems` run asks for the model, so the button just asks first. See
# `demucs_argv` and `setup.demucs_check`.
DEMUCS_ID = "model.demucs"
# An explicit source directory, for a machine that has neither a checkout nor an
# .app to read from: point it at an unpacked copy and the button works again.
DIARIZATION_SOURCE_ENV = "DUB_DIARIZATION_SOURCE"
# Where a macOS install keeps the payload the workspace was copied from
# (`provision.rs`: `Contents/Resources/workspace`). Globbed rather than named,
# because the bundle's name is the app's and this module does not own it. There
# is no equivalent constant for Windows or Linux, and inventing one that is wrong
# would be worse than the honest sentence the row falls back to.
DIARIZATION_BUNDLE_GLOB = "*.app/Contents/Resources/workspace"
DIARIZATION_BUNDLE_ROOTS = (Path("/Applications"), Path.home() / "Applications")
# The file every candidate source has to carry: it is both the proof that the
# directory is the weights and the manifest the copy is checked against.
DIARIZATION_SUMS = "SHA256SUMS"

# Terminal control sequences, which the Setup screen is not a terminal for. A
# package manager draws its progress with colour codes and by rewriting the line
# it is on, and neither survives the trip: the escapes reach the panel as
# `[32m` glued to the word after them, and the rewrites reach it as one
# `\r`-joined line the width of the whole download.
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b[@-Z\\-_]")


def readable(text: str) -> str:
    """One package-manager line as a person would have seen it in a terminal.

    Two things a terminal does that a `<pre>` does not. It interprets the colour
    escapes instead of printing them, so they are dropped. And it obeys the
    carriage returns a progress bar redraws itself with, which means the line a
    user would have been looking at is the *last* segment, not all forty of them
    end to end: `pip`'s download bar arrives here as a single line thousands of
    characters wide, and the tail row it lands in is unreadable and pushes the
    real output off the panel.

    Nothing informative is lost. Every earlier segment of a redrawn line is a
    stale copy of the one kept, which is the whole point of redrawing it.
    """
    text = _ANSI.sub("", text).replace("\x08", "")
    # `\r\n` is a line ending, not a redraw; only a bare `\r` rewinds the line.
    return text.replace("\r\n", "\n").rsplit("\r", 1)[-1].rstrip()


def registry_path_entries() -> list[str]:
    """Every directory this machine's *persisted* PATH names, on Windows.

    Empty on every other platform, and that is the whole of the platform
    difference: a Mac or a Linux process inherits the PATH its parent had, and
    `server.widen_path` already adds the Homebrew prefixes a GUI-launched app
    would otherwise miss. Windows keeps PATH in the registry and hands each new
    process a copy, so a `winget install` that just put `ffmpeg.exe` somewhere
    is invisible to a process that started before it. `install-server.ps1`
    rebuilds PATH from the registry for that exact reason and says so; this is
    the same rebuild, for the server that runs the buttons.

    Both hives, **machine then user**, which is the order Windows composes them
    in: the session's PATH is the system value with the per-user value appended,
    which is why a per-user entry can only ever be a fallback and never a
    shadow. This read had them the other way round, with a docstring asserting
    that user-first was how Windows did it, and the consequence is the exact
    failure the function exists to prevent. A machine carrying a stale
    per-user ffmpeg and a good system one resolved to the stale copy here and to
    the good one in a fresh cmd.exe, so the server and the user's own terminal
    disagreed about what this machine has, which is unfalsifiable from the
    outside and looks like the app being wrong about everything.
    `install-server.ps1`'s `Update-PathFromRegistry` joins machine and then user
    and always did; the two halves of one idea now say the same thing.

    A key that will not open is skipped rather than raised on: a PATH we could
    not widen is the state we were already in.
    """
    if sys.platform != "win32":
        return []
    import winreg

    hives = ((winreg.HKEY_LOCAL_MACHINE,
              r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
             (winreg.HKEY_CURRENT_USER, "Environment"))
    found: list[str] = []
    for root, key in hives:
        try:
            with winreg.OpenKey(root, key) as handle:
                value, _ = winreg.QueryValueEx(handle, "Path")
        except OSError:
            continue
        # Split on `;` and not `os.pathsep`, for the same reason PATHEXT does
        # in `dubbing.tools`: this is a Windows registry value and it is always
        # semicolon separated. `os.pathsep` happens to be `;` on the machine
        # that runs this for real, so the two agree in production and the bug
        # is invisible there. It shows up in the test, which fakes `winreg`
        # from a Mac, where `os.pathsep` is `:` and every `C:\...` entry gets
        # cut in half at its drive letter. A Windows-only parser that reads the
        # host's separator is a parser the host can never check.
        found += [part for part in
                  os.path.expandvars(str(value)).split(";") if part]
    return found


def refresh_path(env: dict[str, str] | None = None,
                 entries: list[str] | None = None) -> str:
    """Append to `env`'s PATH the directories this machine has gained since the
    process started, and answer the new PATH.

    Why it exists: `Installer._finish` re-probes rather than trusting a package
    manager's exit code, and on Windows that re-probe could never succeed. winget
    installs a tool and adds its directory to the registry PATH, which reaches
    processes started *afterwards*; the server was started before, so
    `shutil.which("ffmpeg")` kept saying no and every winget row ended in
    "succeeded but it is still not there, restart the app". The install had
    worked. Only the lookup was stale, and a restart is a poor answer to a
    problem the process can fix in a millisecond.

    `entries` is injectable so the merge can be tested off Windows, where there
    is no registry to read and `registry_path_entries` answers with nothing.
    Every entry is `is_dir()`-guarded for the reason `server.widen_path` guards
    its own: PATH is read by every child this process ever spawns, and filling
    it with directories that do not exist is a cost paid on every lookup for a
    guess that was wrong.
    """
    environ = os.environ if env is None else env
    incoming = registry_path_entries() if entries is None else list(entries)
    parts = [part for part in environ.get("PATH", "").split(os.pathsep) if part]
    seen = {os.path.normcase(part.rstrip("\\/")) for part in parts}
    for entry in incoming:
        key = os.path.normcase(entry.rstrip("\\/"))
        if not key or key in seen:
            continue
        try:
            if not Path(entry).is_dir():
                continue
        except OSError:
            continue
        seen.add(key)
        parts.append(entry)
    environ["PATH"] = os.pathsep.join(parts)
    return environ["PATH"]


def demucs_argv() -> tuple[str, ...]:
    """The argv that fetches the Demucs weights, and nothing else.

    `dubbing/stems.py` shells out to `python -m demucs` and demucs pulls
    `htdemucs_ft` down on that first call, so the honest way to "install" it is
    to make that call happen early: `get_model(stems.MODEL)` loads the bag of
    models by name, which is the download, and then drops it. Same interpreter as
    the server (`sys.executable`, the one `runner.SubprocessRunner` also spawns
    jobs with), same cache, same weights, so the row is green afterwards because
    the thing the run needs is genuinely there, not because a button said so.

    The model name comes from the pipeline constant rather than being spelled
    here, so this cannot fetch a different bag than the stems stage opens. It is
    a hardcoded argv like every other row's: nothing from a request reaches it.
    """
    from dubbing import stems

    return (sys.executable, "-c",
            "from demucs.pretrained import get_model; "
            f"get_model({stems.MODEL!r}); print('demucs weights ready')")


def static_route(id_: str) -> bool:
    """True when the button installs `id_` as a static build instead of running
    a package manager: `ffmpeg` (the zip carries ffprobe too), on a machine with
    no unattended manager to prefer.

    One question — "is there a package manager here this app is allowed to
    drive?" — with a different answer on each platform:

    * **macOS**: Homebrew when it is present, because it also delivers updates;
      the static build only on the factory-state Mac that has none, where "get
      it from brew.sh" is a terminal plus an admin password, which is the whole
      journey the DMG exists to avoid.
    * **Windows**: the same shape with winget.
    * **Linux**: always. The recipe there is `sudo apt-get`, which asks for a
      password on a terminal the app does not have, so it is deliberately absent
      from `tools.auto_installers()` — which used to mean a Linux user got no
      ffmpeg button at all, and (once one button installs everything) an
      "install everything" that quietly skipped the one tool every stage shells
      out to.

    The same wheel covers all three: `static-ffmpeg` ships a build per platform,
    so this is one route, not three. `sox` is not offered on any of them —
    no vetted static source, and the pipeline no longer blocks on it
    (see `setup.TOOLS`).
    """
    if id_ != "ffmpeg":
        return False
    platform = tools.platform_key()
    if platform == "darwin":
        return shutil.which("brew") is None
    if platform == "win32":
        return shutil.which("winget") is None
    return True


def uv_release_route(id_: str) -> bool:
    """True when the button installs `uv` from its official release archive
    rather than by driving a package manager.

    Same shape as `static_route` and the same one question, "is there a
    package manager here this app is allowed to drive?", with a different
    answer per platform: Homebrew on a Mac that has it, winget on a Windows that has it,
    and the release archive on the machines that have neither, which is every
    Linux (no distribution packages uv at all, so unlike ffmpeg there is not
    even a `sudo apt-get` line to print).

    The package manager is preferred wherever it exists, for the reason it is
    preferred for ffmpeg: it also delivers the next upgrade, and a binary this
    app drops in a directory is a binary this app has to remember to update.
    """
    if id_ != UV_ID:
        return False
    argv = INSTALLERS.get(id_)
    return argv is None or shutil.which(argv[0]) is None


def uv_triple() -> str | None:
    """The release asset triple for this machine, or None when astral publishes
    no build for it (a 32-bit Windows, a riscv Linux). None is what turns the
    button off rather than a download that 404s.

    The table and the fold are `dubbing.uvrelease`'s, shared with the desktop
    staging script; this is only the name the app calls it by, kept because
    `installable` and the tests both ask for it here.
    """
    return uvrelease.host_triple()


def uv_install_dir() -> Path:
    """Where the button puts uv: the per-user `~/.local/bin`.

    Not the workspace's `tools/bin`, which is where the static ffmpeg goes, and
    the difference is which chains already look there. `tools/bin` is a
    convention only `dubbing.tools.resolve_tool` knows, and uv is not resolved
    through it: uv is found by `setup.find_uv` and by `workspace.rs`'s
    `find_uv`, both of which end at `~/.local/bin` and `~/.cargo/bin`. Putting
    it anywhere else would mean teaching a Rust binary a new location and
    shipping a new shell before the button worked, so the button writes where
    both sides already look, which is also where uv's own installer,
    `install-server.sh` and `install-server.ps1` all put it, so a machine ends
    up with one uv however it was installed rather than two that disagree.

    It is outside the workspace, which is a real cost: deleting the workspace no
    longer deletes everything the app added. It is the right cost. uv is a
    user-level tool that outlives any one checkout, and a second private copy of
    a 35 MB binary per workspace is the thing a user would actually resent.
    """
    home = _uv_home()
    if home is None:
        raise RuntimeError("neither HOME nor USERPROFILE is set, so there is no "
                           "per-user bin directory to install uv into")
    return home / ".local" / "bin"


def _uv_home() -> Path | None:
    # `dubbing.tools`, which is where the uv lookup lives now; `setup.uv_home`
    # is a re-export of this same function, and going through the app to reach
    # the pipeline's own answer was one hop of indirection with nothing on it.
    return tools.uv_home()


def download(url: str, timeout: float = 180.0) -> bytes:
    """Fetch `url`. Module-level, like `snapshot_download`, so a test swaps the
    whole thing and no bytes cross the network; `install_uv` hands it to
    `uvrelease.fetch_uv` as that function's `fetch`, so the seam the tests
    already knew about is still the only one."""
    return uvrelease.download(url, timeout)


def install_uv(log: Callable[[str], None]) -> Path:
    """Put the official `uv` build for this machine in `~/.local/bin`, after
    checking it against the checksum astral publishes beside it.

    The fetch, the checksum and the archive are `dubbing.uvrelease`'s, which is
    the same code `scripts/stage_desktop_payload.py` runs and the same code
    `install-server.sh` and `install-server.ps1` describe in shell. Why an
    archive rather than a piped install script is written down there.

    What is this function's own is the last three lines: where the binary goes
    (`uv_install_dir`, the directory every uv lookup in this repo already ends
    at) and the sentence for a machine astral publishes nothing for.
    """
    triple = uv_triple()
    if triple is None:
        raise RuntimeError(
            f"astral publishes no uv build for {platform.machine()} on "
            f"{sys.platform}; install uv by hand from https://docs.astral.sh/uv/")
    payload = uvrelease.fetch_uv(triple, log=log, fetch=download)
    target = uvrelease.write_binary(payload,
                                    uv_install_dir() / uvrelease.binary_name(triple))
    log(f"installed {target}")
    return target


def diarization_target() -> Path:
    """Where the weights belong: the pipeline's own constant, never a copy."""
    from dubbing import segments

    return segments.DIARIZATION_DIR


def _is_diarization_copy(path: Path) -> bool:
    return (path / DIARIZATION_SUMS).is_file()


def diarization_source() -> tuple[str, Path] | None:
    """`(kind, where)` the app would restore the diarization weights from, or
    None when this machine carries no copy to restore from.

    Two kinds, because there are two ways this app is installed and they keep
    the same 31 MB in different places:

    * ``"copy"`` a directory holding the weights: the `DUB_DIARIZATION_SOURCE`
      override first, then the payload inside a macOS .app, which is what the
      workspace was provisioned from and is still there, read-only and intact,
      after someone deletes the working copy.
    * ``"git"`` a checkout. The weights are committed (they are payload, not
      user state), so `git checkout --` is a restore from the object store with
      no network and no account, and it is the route on a developer machine.

    Never a download. The upstream repo is the one gated thing in the tree, and
    a button that stops to ask for a Hugging Face account is the button this
    app exists not to have.
    """
    target = diarization_target()
    override = (os.environ.get(DIARIZATION_SOURCE_ENV) or "").strip()
    if override:
        candidate = Path(override)
        if candidate.resolve() != target.resolve() and _is_diarization_copy(candidate):
            return "copy", candidate
    for root in DIARIZATION_BUNDLE_ROOTS:
        try:
            payloads = sorted(root.glob(DIARIZATION_BUNDLE_GLOB))
        except OSError:                      # an unreadable /Applications
            continue
        for payload in payloads:
            candidate = payload / "third_party" / target.name
            if candidate.resolve() != target.resolve() and _is_diarization_copy(candidate):
                return "copy", candidate
    repo = target.parents[1]                 # <root>/third_party/<name>
    if (repo / ".git").exists() and shutil.which("git") is not None:
        return "git", repo
    return None


def diarization_command() -> str | None:
    """The line a user would type to do by hand what the button does, or None
    when there is nothing on this machine to restore from. The Setup row carries
    it, so a machine with no button still has an answer instead of a red row.

    The copy line is spelled for the shell that will read it. `cp -R` is a
    sentence a PowerShell prompt answers with "cp : A positional parameter
    cannot be found", which is the worst kind of instruction: it looks like
    help, it is confidently wrong, and the user has no way to tell which of the
    two of you is mistaken. `git` is spelled the same everywhere, so the other
    branch needs no such split.
    """
    source = diarization_source()
    if source is None:
        return None
    kind, where = source
    target = diarization_target()
    if kind == "git":
        return f"git -C {where} checkout -- {target.relative_to(where)}"
    if tools.platform_key() == "win32":
        return f'robocopy "{where}" "{target}" /E'
    return f"cp -R {where}/. {target}"


def diarization_route() -> str | None:
    """One phrase naming what pressing Install on the diarization row does."""
    source = diarization_source()
    if source is None:
        return None
    kind, where = source
    if kind == "git":
        return f"by restoring the committed weights from the checkout at {where}"
    return f"by copying the weights that shipped with the app, from {where}"


def installable(id_: str) -> bool:
    """Can the app install `id_` by itself? The tools it has an argv for, the
    ones it can fetch a static build of, `uv` from its official release archive
    where no package manager here publishes it, the diarization weights when
    this machine still carries the copy they shipped in, and the Demucs cache, which
    is always yes: the fetch is demucs's own and needs nothing from this machine
    but a network. Read by `setup.report` and
    `setup.probe` so the flag that puts a button on a row and the code behind
    the button are the same answer — a row whose POST is a 400 is worse than no
    button, and so is a missing button for an install that would have worked."""
    if id_ == DIARIZATION_ID:
        return diarization_source() is not None
    if id_ == DEMUCS_ID:
        return True
    if uv_release_route(id_):
        # Only when astral has a build for this machine and there is a home
        # directory to put it in. A button whose download would 404, or whose
        # destination does not resolve, is worse than the sentence naming the URL.
        return uv_triple() is not None and _uv_home() is not None
    argv = INSTALLERS.get(id_)
    # **Having a recipe is not having the manager.** `INSTALLERS` is a table of
    # what this *platform* installs with, built at import from
    # `tools.auto_installers()`, and it says `brew install sox` on every Mac
    # including the ones with no Homebrew. Answering yes off that table put an
    # Install button on the `sox` row of a factory-state Mac (and of a Windows
    # with no winget), and pressing it was exactly the 400 the docstring above
    # forbids: `_start_without_manager` has no static build and no release
    # archive to fall back to for sox, so all it can do is refuse. "Install
    # everything" queued that row too, and recorded the refusal as a failure.
    # `route()` had this right from the start and asked whether the manager is
    # on the machine; the flag that draws the button and the sentence under it
    # have to be the same answer, so this asks the same question. Per call
    # rather than at table-build time, so a brew installed while the app is open
    # turns the button on at the next poll.
    if argv is not None and shutil.which(argv[0]) is not None:
        return True
    return static_route(id_)


def route(id_: str) -> str | None:
    """One phrase naming how the Install button would install `id_`, or None
    when it would not (no recipe, or a manager that is not on this machine and
    no static fallback). The Setup row's detail carries it, so the user knows
    what pressing the button does *before* pressing it."""
    if id_ == DIARIZATION_ID:
        return diarization_route()
    if id_ == DEMUCS_ID:
        from dubbing import stems

        return (f"by fetching the {stems.MODEL} weights now, into the same cache "
                "the first stems run would fill anyway")
    if static_route(id_):
        return "as a static build into the workspace (tools/bin), with no package manager needed"
    if uv_release_route(id_):
        if not installable(id_):
            return None
        # It says where, because `~/.local/bin` is on PATH by default on
        # neither Windows nor a bare Linux login shell. The app itself does not
        # care (`setup.find_uv` and the shell's own lookup both end there), and
        # a user who wants `uv` at a prompt does, so the sentence has to say so
        # rather than let them discover it.
        return (f"by downloading the official uv build for this machine from "
                f"{UV_RELEASES}, checking it against the SHA-256 astral publishes "
                f"beside it, and putting it in {uv_install_dir()}, which this app "
                "looks in. Add that directory to PATH yourself to use uv in a "
                "terminal")
    argv = INSTALLERS.get(id_)
    if argv is None:
        return None
    manager = argv[0]
    if shutil.which(manager) is None:
        return None
    return f"via {'Homebrew' if manager == 'brew' else manager}"


def fetch_static_ffmpeg(log: Callable[[str], None]) -> tuple[str, str]:
    """Get static ffmpeg + ffprobe paths, installing `static-ffmpeg` first if
    this venv lacks it. Module-level, like `snapshot_download`, so a test swaps
    it whole and no wheel or binary ever moves.

    The wheel goes in with `uv pip --python <this interpreter>` — the same uv
    that built the venv, found the same way (`setup.find_uv`); the import cache
    is invalidated so the just-installed package is importable without a
    restart. The download itself is the package's own: a pinned zip URL per
    platform from its release repo, not an ad-hoc website scrape of ours.
    """
    try:
        from static_ffmpeg import run as static_run
    except ImportError:
        from . import setup

        uv = setup.find_uv()
        if uv is None:
            raise RuntimeError(
                "uv was not found, and it is what installs the static-ffmpeg "
                "wheel into this environment: fix the uv row first")
        argv = [uv, "pip", "install", "--python", sys.executable, STATIC_FFMPEG_SPEC]
        log("$ " + " ".join(argv))
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise RuntimeError(f"`{' '.join(argv)}` exited {proc.returncode}: "
                               f"{(proc.stderr or proc.stdout)[-500:]}")
        importlib.invalidate_caches()
        from static_ffmpeg import run as static_run

    log("fetching the pinned ffmpeg/ffprobe build for this platform")
    return static_run.get_or_fetch_platform_executables_else_raise()


def verify_diarization(root: Path) -> int:
    """Check every file `root`'s SHA256SUMS names and return how many passed.
    Raises when one is missing or does not match.

    The manifest is the same one `tests/test_phase2_segments.py` checks the
    bundled copy against, and it travels with the weights, so what is verified
    here is that *this* copy arrived whole. A half-finished `cp` into a full
    disk is the realistic failure, and pyannote's own error for it would be a
    tensor shape complaint fifty lines into a run.
    """
    import hashlib

    sums = root / DIARIZATION_SUMS
    try:
        lines = sums.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(f"no {DIARIZATION_SUMS} beside the restored weights: {exc}")
    checked = 0
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        expected, _, rel = line.partition(" ")
        rel = rel.strip().lstrip("*")
        target = root / rel
        if not target.is_file():
            raise RuntimeError(f"{rel} is missing from the restored weights")
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(f"{rel} does not match {DIARIZATION_SUMS}: the copy "
                               "is damaged, so it was not left in place")
        checked += 1
    if not checked:
        raise RuntimeError(f"{DIARIZATION_SUMS} named no files to verify")
    return checked


def restore_diarization(log: Callable[[str], None]) -> Path:
    """Put the bundled diarization weights back, and prove they arrived.

    Module-level like `fetch_static_ffmpeg` and `snapshot_download`, so a test
    swaps the whole thing and no 31 MB moves. Nothing here touches the network:
    the source is a copy this machine already has (`diarization_source`).

    The copy route stages into a sibling directory and only moves it into place
    once the hashes agree, so a full disk or a damaged payload leaves the
    workspace exactly as it was rather than leaving a half-written pipeline for
    a run to crash on. The git route cannot do that (the checkout writes where
    the weights belong), so it verifies after the fact and says so if it fails,
    which is still the checkout's own state, repairable by running it again.
    """
    source = diarization_source()
    if source is None:
        raise RuntimeError(
            "nothing on this machine carries the diarization weights to restore "
            f"from. Set {DIARIZATION_SOURCE_ENV} to an unpacked copy, or reinstall "
            "the app, or point DUB_DIARIZATION_HUB at a mirror you can reach")
    kind, where = source
    target = diarization_target()
    if kind == "git":
        argv = ["git", "-C", str(where), "checkout", "--",
                str(target.relative_to(where))]
        log("$ " + " ".join(argv))
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            raise RuntimeError(f"`{' '.join(argv)}` exited {proc.returncode}: "
                               f"{(proc.stderr or proc.stdout)[-500:]}")
        checked = verify_diarization(target)
    else:
        log(f"copying the bundled weights from {where}")
        staging = target.parent / f"{target.name}.incoming"
        shutil.rmtree(staging, ignore_errors=True)
        staging.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copytree(where, staging, dirs_exist_ok=True)
            checked = verify_diarization(staging)
            shutil.rmtree(target, ignore_errors=True)
            staging.replace(target)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    log(f"verified {checked} files against {DIARIZATION_SUMS}")
    return target


def snapshot_download(**kwargs: Any) -> Any:
    """`huggingface_hub.snapshot_download`, imported when first used.

    Deferred because rule one of `setup.py` holds here too: the tool installs
    never need huggingface_hub, and a `GET /api/setup` must not pay its import.
    Module-level so a test swaps the whole thing with one monkeypatch and no
    bytes ever move.
    """
    from huggingface_hub import snapshot_download as fetch

    return fetch(**kwargs)


def manual_command(id_: str, recipes: dict[str, tuple[str, ...]] | None = None) -> str | None:
    """What the user would type for `id_`, or None if this platform has none.

    Falls back to `dubbing.tools`, so a tool the app will not install for itself
    (no unattended package manager) still hands the user its command.
    """
    argv = (recipes if recipes is not None else INSTALLERS).get(id_)
    return " ".join(argv) if argv else tools.command(id_)


class Installer:
    """The one install slot, and the process behind it.

    `probe` re-reads a single check by id (`setup.probe`); `recipes` overrides
    the module table, which is how the tests get a stub argv instead of a real
    package manager. Left unset it reads `INSTALLERS` *at call time*, so
    monkeypatching either the name or its contents works. `downloads` is the
    same override for the model table (`setup.model_downloads()`), for tests
    that want a tiny repo pointed at a tmp dir instead of 9.7 GB at `models/`.
    """

    def __init__(self, probe: Probe, recipes: dict[str, tuple[str, ...]] | None = None,
                 spawn: Callable[..., subprocess.Popen] | None = None,
                 downloads: dict[str, dict[str, Any]] | None = None):
        self._probe = probe
        self._recipes = recipes
        self._downloads = downloads
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
        # Which route the slot is running (see the constants above). Set when it
        # is claimed, read by `_finish` when it is released.
        self._route: str | None = None
        # Set only while the slot holds a model download: where the bytes land
        # and how many are expected. `status()` reads the directory's size off
        # disk each poll — hf's tqdm has nowhere to draw, and the filesystem is
        # the one progress source that cannot disagree with reality.
        self._dl_dir: Path | None = None
        self._dl_total: int | None = None

    @property
    def recipes(self) -> dict[str, tuple[str, ...]]:
        return self._recipes if self._recipes is not None else INSTALLERS

    @property
    def downloads(self) -> dict[str, dict[str, Any]]:
        if self._downloads is not None:
            return self._downloads
        from . import setup

        return setup.model_downloads()

    # -- api ---------------------------------------------------------------

    def start(self, id_: str) -> dict[str, Any]:
        """Begin installing `id_`. Raises the 400/409 the UI renders."""
        argv = self.recipes.get(id_)
        if argv is None:
            return self._start_without_recipe(id_)
        manager = argv[0]
        if shutil.which(manager) is None:
            return self._start_without_manager(id_, tuple(argv), manager)
        return self._begin(id_, ["$ " + " ".join(argv)], route=MANAGER,
                           target=self._run, args=(id_, tuple(argv)))

    def _start_without_recipe(self, id_: str) -> dict[str, Any]:
        """The rows whose install is not a package-manager argv at all.

        A platform whose package manager cannot run unattended contributes no
        recipes at all (Linux: `sudo apt-get` wants a password), so ffmpeg
        arrives here with no argv and still has a route. So does uv, which on
        Linux has no recipe on any table because no distribution packages it.
        Then the two ids whose fix is neither a manager nor a snapshot, and last
        the hub downloads.
        An id in none of those is the 400 that names what the app does install.
        """
        if static_route(id_):
            # The static build is the only button that can exist there, and
            # without it "install everything" would silently skip the one tool
            # every stage shells out to.
            return self._start_static(
                id_, f"# no package manager here can install {id_} unattended, "
                     f"so installing a static ffmpeg/ffprobe build into "
                     f"{tools.tools_bin()}")
        if uv_release_route(id_) and installable(id_):
            # Every Linux, plus the Mac with no brew and the Windows with no
            # winget. Nothing about this needs a package manager or a password.
            return self._begin(
                id_, [f"# no package manager here can install uv, so downloading "
                      f"the official build from {UV_RELEASES} and verifying its "
                      f"published SHA-256"],
                route=UV_ROUTE, target=self._run_uv, args=(id_,))
        if id_ == DEMUCS_ID:
            # A subprocess like the package managers, and for the same reason:
            # it imports torch and demucs, which the server process spends its
            # life not importing (`setup.py` rule one).
            demucs = demucs_argv()
            return self._begin(
                id_, ["$ " + " ".join(demucs),
                      "fetching the stem-separation weights into the cache the "
                      "first stems run would fill anyway"],
                route=DEMUCS_ROUTE, target=self._run, args=(id_, demucs))
        if id_ == DIARIZATION_ID and installable(id_):
            source = diarization_source()
            assert source is not None            # installable() just said so
            return self._begin(
                id_, [f"# restoring the bundled diarization weights from "
                      f"{source[1]}, and verifying them against {DIARIZATION_SUMS}"],
                route=RESTORE, target=self._run_restore, args=(id_,))
        spec = self.downloads.get(id_)
        if spec is None:
            raise invalid(self._refusal(id_))
        return self._start_download(id_, spec)

    def _start_without_manager(self, id_: str, argv: tuple[str, ...],
                               manager: str) -> dict[str, Any]:
        """The recipe is here but the manager is not: a brewless Mac, a Windows
        with no winget. "Install the package manager first" dies at a terminal,
        so take the static route instead of refusing. Checked here, not at
        table-build time, so a brew installed mid-session is picked up on the
        next press."""
        if static_route(id_):
            return self._start_static(
                id_, f"# {manager} is not on this machine, so installing a "
                     f"static ffmpeg/ffprobe build into {tools.tools_bin()}")
        if uv_release_route(id_) and installable(id_):
            # A brewless Mac or a wingetless Windows: the recipe is in the
            # table but its manager is not here, and uv needs neither.
            return self._begin(
                id_, [f"# {manager} is not on this machine, so downloading the "
                      f"official uv build from {UV_RELEASES} and verifying its "
                      f"published SHA-256"],
                route=UV_ROUTE, target=self._run_uv, args=(id_,))
        template = MANAGERS.get(manager,
                                "`{manager}` is not on PATH: install `{command}` by hand.")
        raise invalid(template.format(tool=id_, command=" ".join(argv), manager=manager))

    def _start_static(self, id_: str, why: str) -> dict[str, Any]:
        """Claim the slot for the pinned static ffmpeg build. `why` is the line
        the tail opens with, naming which of the two ways we got here."""
        return self._begin(id_, [why], route=STATIC, target=self._run_static,
                           args=(id_,))

    def _start_download(self, id_: str, spec: dict[str, Any]) -> dict[str, Any]:
        """Claim the slot for a hub snapshot. Same slot as the tools — the
        screen has one progress row, and two downloads share one disk and one
        pipe anyway."""
        from . import setup

        hub, local_dir = str(spec["hub"]), Path(spec["path"])
        total = int(spec.get("bytes") or 0)
        lines = [f"$ snapshot_download({hub!r}, local_dir={str(local_dir)!r})",
                 (f"downloading ~{setup.human_bytes(total)}; " if total else "downloading; ")
                 + "partial files are kept, so an interrupted download resumes where it stopped"]
        return self._begin(id_, lines, route=DOWNLOAD, target=self._run_download,
                           args=(id_, hub, local_dir), dl_dir=local_dir, dl_total=total)

    def _begin(self, id_: str, lines: list[str], *, route: str,
               target: Callable[..., None], args: tuple,
               dl_dir: Path | None = None,
               dl_total: int | None = None) -> dict[str, Any]:
        with self._lock:
            if self._running:
                raise busy(f"an install is already running ({self._id}); "
                           "one at a time, so wait for it to finish")
            self._running = True
            self._id = id_
            self._ok = None
            self._error = None
            self._check = None
            self._started = time.time()
            self._finished = None
            self._route = route
            self._dl_dir = dl_dir
            self._dl_total = dl_total
            self._tail.clear()
            self._tail.extend(lines)
            self._thread = threading.Thread(target=target, args=args,
                                            name=f"install-{id_}", daemon=True)
            self._thread.start()
        return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            body = {"running": self._running, "id": self._id, "ok": self._ok,
                    "error": self._error, "tail": list(self._tail), "check": self._check,
                    "started": self._started, "finished": self._finished}
            dl_dir, dl_total = self._dl_dir, self._dl_total
        if dl_dir is not None:
            from . import setup

            # Walked outside the lock: `dir_size` on a directory mid-download is
            # real I/O, and a poll must not stall the writer to count its bytes.
            # `bytes_total` is the table's estimate, so done/total can overshoot
            # 100% by a little — a progress bar clamps, accounting never asks.
            body["bytes_done"] = setup.dir_size(dl_dir)
            body["bytes_total"] = dl_total
        return body

    def reset(self) -> None:
        """Forget a finished install, so the next answer is about the next
        gesture and not the last one.

        The mirror of `InstallQueue.clear`, and it exists for the bug that one
        did not cover. `POST /api/setup/install_all` answers the slot's status
        with a queue block beside it, so a queue that ran nothing answered with
        whatever the last single install left in the slot: `id: "model.lid"`,
        `ok: true`, "download complete". The most likely presentation of "nothing
        happened" was then "something succeeded", for a model nobody had just
        asked for. A running install is never touched, because that one is still
        the truth about right now.
        """
        with self._lock:
            if self._running:
                return
            self._id = None
            self._ok = None
            self._error = None
            self._check = None
            self._started = None
            self._finished = None
            self._route = None
            self._dl_dir = None
            self._dl_total = None
            self._tail.clear()

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
        # What this machine would type for the tools there is no button for —
        # on a platform whose package manager needs a terminal that is *every*
        # tool, and a refusal that names none of them is a scavenger hunt.
        byhand = ", ".join(f"`{cmd}`" for cmd in
                           (tools.command(t) for t in sorted(tools.recipes()))
                           if cmd and cmd not in
                           {" ".join(a) for a in self.recipes.values()})
        return (f"{id_!r} cannot be installed from the app. The only installs it "
                f"runs are {offered or 'none'} and the hub-snapshot model "
                "downloads. Everything else is by hand: "
                + (f"the tools with {byhand}, and " if byhand else "")
                + "the remaining models download themselves on first use. Where "
                "a command is needed it is in that check's detail line: run it "
                "in a terminal from the repo.")

    def _env(self) -> dict[str, str]:
        # Non-interactive, because there is no terminal to answer a prompt on and
        # a blocked read would hold the one slot until the timeout.
        return {**os.environ, "NONINTERACTIVE": "1", "HOMEBREW_NO_AUTO_UPDATE": "1",
                "HOMEBREW_NO_ANALYTICS": "1"}

    def _run(self, id_: str, argv: tuple[str, ...]) -> None:
        ok, error = False, None
        try:
            # `errors="replace"`: a package manager's progress output is full of
            # box-drawing characters, and a decode error must not be what ends an
            # install that was otherwise working.
            # Detached like a job child (`runner.spawn_kwargs`), so the timeout
            # kill below can reach the whole tree. A package manager keeps its
            # real work in children that inherit the stdout pipe; killing only
            # the leader leaves that pipe open, and the read loop then blocks
            # until the orphans exit — a "stopped, the slot is free again"
            # message over a slot that was not free at all.
            proc = self._spawn(list(argv), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               text=True, encoding="utf-8", errors="replace",
                               bufsize=1, env=self._env(),
                               **runner_mod.spawn_kwargs())
        except OSError as exc:
            self._finish(id_, False, f"{type(exc).__name__}: {exc}")
            return
        # The timer says so, rather than only doing it. A killed child exits with
        # a signal, so the sentence a user was shown for a package manager that
        # hung for half an hour was "`brew install ffmpeg` exited -9", which
        # names neither the cause nor anything to do about it. The one fact that
        # explains that number is held here and nowhere else.
        timed_out = threading.Event()

        def expire() -> None:
            timed_out.set()
            self._kill(proc)

        killer = threading.Timer(TIMEOUT, expire)
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
        if timed_out.is_set():
            ok = False
            error = (f"`{' '.join(argv)}` was still running after "
                     f"{TIMEOUT / 60:.0f} minutes and was stopped, so the one "
                     "install slot is free again. It was most likely waiting for "
                     "something this app cannot answer: run it in a terminal to "
                     "see what it wants.")
        # Before the re-probe, and only for the route where it changes the
        # answer: on Windows a package manager puts its binary on the PATH new
        # processes get, and the check below runs in this one. See `refresh_path`.
        if self._route == MANAGER:
            refresh_path()
        # This is the one route whose failure is only ever a claim about an exit
        # code, so it is the one route a green re-probe is allowed to overrule
        # (see `_finish`). Not after a timeout, though: a child that was killed
        # at thirty minutes did not finish anything, and a row that is green in
        # spite of it is green for some earlier reason that has nothing to say
        # about this attempt.
        self._finish(id_, ok, error, exit_code_only=not timed_out.is_set())

    def _run_download(self, id_: str, hub: str, local_dir: Path) -> None:
        """The worker behind a model install: one `snapshot_download` into the
        directory the pipeline will read. No timeout, deliberately — 81 GB on
        home DSL is slower than any brew, there is no child process to kill,
        and an abandoned attempt costs nothing because the next one resumes
        from the partial files.

        The one line that is not the download is the receipt. This function is
        the only place in the app that knows, as a fact rather than an estimate,
        that a model's fetch finished: `snapshot_download` returned instead of
        throwing, which means every file in the repo is on disk. That knowledge
        used to die here, and `setup.model_ready` was left to re-derive it by
        weighing the directory against a hand-typed size, which is how a
        complete language-ID model came to be reported as a partial download
        moments after the app itself finished downloading it. Writing it down
        costs one small file and ends the argument.
        """
        from . import setup

        ok, error = False, None
        try:
            # Looked up as a module global so a test's monkeypatch is what runs.
            snapshot_download(repo_id=hub, local_dir=str(local_dir))
            # Before `_finish`, because `_finish` re-probes and the whole point
            # is that the re-probe sees what this worker knows.
            setup.record_install(local_dir)
            ok = True
            self._line("download complete")
        except Exception as exc:                       # network, auth, disk — all of it
            error = (f"{type(exc).__name__}: {exc}. Partial files are kept, "
                     "so starting the install again resumes the download")
            self._line(error)
        self._finish(id_, ok, error)

    def _run_restore(self, id_: str) -> None:
        """The worker behind the diarization row: a local copy, then a hash
        check. No timeout and no network, so the only failure modes are a full
        disk and a damaged source, and both of them are named rather than left
        for a run to hit."""
        ok, error = False, None
        try:
            # Looked up as a module global so a test's monkeypatch is what runs.
            restore_diarization(self._line)
            ok = True
            self._line("restore complete")
        except Exception as exc:                       # disk, permissions, a bad copy
            error = f"{type(exc).__name__}: {exc}"
            self._line(error)
        self._finish(id_, ok, error)

    def _run_uv(self, id_: str) -> None:
        """The worker behind the uv row on a machine with no package manager to
        drive. No timeout: it is one 35 MB download, and there is no prompt to
        hang on. Looked up as a module global so a test's monkeypatch is what
        runs and no binary ever lands in a real home directory."""
        ok, error = False, None
        try:
            install_uv(self._line)
            ok = True
        except Exception as exc:                       # network, checksum, disk
            error = f"{type(exc).__name__}: {exc}"
            self._line(error)
        self._finish(id_, ok, error)

    def _run_static(self, id_: str) -> None:
        """The brewless worker: fetch the pinned static build and copy its two
        binaries into `tools.tools_bin()`, where `resolve_tool` finds them
        before PATH. Copies, never symlinks — the wheel is not in the lockfile,
        so the next `uv sync` prunes it, and a symlink into pruned
        site-packages is a broken ffmpeg one sync later. No timeout, same
        reasoning as the model downloads: the slow part is a download that
        resumes nothing but costs one retry, and there is no prompt to hang on.
        """
        ok, error = False, None
        try:
            ffmpeg, ffprobe = fetch_static_ffmpeg(self._line)
            bin_dir = tools.tools_bin()
            bin_dir.mkdir(parents=True, exist_ok=True)
            for src in (Path(ffmpeg), Path(ffprobe)):
                dst = bin_dir / src.name
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                shutil.copy2(src, dst)
                dst.chmod(dst.stat().st_mode | 0o755)
                self._line(f"installed {dst}")
            ok = True
        except Exception as exc:                       # network, disk, a bad zip
            error = f"{type(exc).__name__}: {exc}"
            self._line(error)
        self._finish(id_, ok, error)

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        # The tree, not the leader: `_run` detached the child exactly so this
        # can reach the grandchildren still holding the stdout pipe. The
        # platform split (killpg / taskkill) lives in `runner.terminate_tree`,
        # and every branch of it ends in at least `proc.kill()`.
        runner_mod.terminate_tree(proc, hard=True)

    def _line(self, text: str) -> None:
        with self._lock:
            self._tail.append(readable(text))

    def _disagreement(self, id_: str, check: dict[str, Any]) -> str:
        """The sentence for a worker that succeeded and a check that says no,
        chosen by the route that ran. See `_finish`."""
        said = str(check.get("detail") or "").strip()
        if self._route == MANAGER:
            argv = " ".join(self.recipes.get(id_, ()))
            return (f"`{argv}` succeeded but {id_} is still not there. Restart the "
                    "app so it picks up the new PATH.")
        if self._route == STATIC:
            # No PATH is involved: `tools.resolve_tool` reads `tools/bin` before
            # it reads PATH, so a restart changes nothing and offering one would
            # send the user round a loop with no exit.
            return (f"the static {id_} build was written to {tools.tools_bin()} and "
                    "the check still says it is not there. Restarting will not "
                    "change that, because that directory is read before PATH is: "
                    "it is a disagreement between the installer and the probe. "
                    "Please report it"
                    + (f". The check says: {said}" if said else "."))
        if self._route == UV_ROUTE:
            # Nothing was downloaded from a hub and no package manager's PATH is
            # at issue. The binary was written where this app chose and the probe
            # that looks there disagreed, which nothing the user presses can fix.
            return (f"uv was written to {uv_install_dir()} and verified against its "
                    "published checksum, and the check still says it is not there. "
                    "That is a disagreement between the installer and the probe, not "
                    "an unfinished install: pressing this again would download the "
                    "same binary and get the same answer. Please report it")
        if self._route == RESTORE:
            return (f"the diarization weights were copied to {diarization_target()} "
                    f"and every file matched {DIARIZATION_SUMS}, and the check still "
                    "says they are not there. Restoring them again would copy the "
                    "same verified files and get the same answer. Please report it"
                    + (f". The check says: {said}" if said else "."))
        if self._route == DEMUCS_ROUTE:
            return (f"the {id_} fetch reported the weights ready and the check still "
                    "says the cache is empty. The two are looking at different "
                    "directories, which is a bug here and not an unfinished "
                    "download. Please report it"
                    + (f". The check says: {said}" if said else "."))
        return (f"the download of {id_} finished, and its check still says it is "
                "not there. That is a bug in the check, not an unfinished "
                "download: running the install again would repeat a download "
                "that already succeeded, and get the same answer. Please report "
                "it" + (f". The check says: {said}" if said else "."))

    def _finish(self, id_: str, ok: bool, error: str | None, *,
                exit_code_only: bool = False) -> None:
        """Record the verdict and re-probe, because the exit code is a claim
        about the package manager, not about this machine.

        The re-probe is the authority in **both** directions, and for one day it
        was only allowed to overrule a success. Overruling a failure is the half
        that was missing, and it is the half a Windows user meets first: winget
        answers non-zero for "this is already installed", which this repo's own
        `install-server.ps1` has a paragraph about. A machine whose ffmpeg was
        installed but not on the server's PATH therefore had a red row, and a
        button that ran winget, got "already installed", and reported a failure
        for a tool that was sitting right there. Pressing it again did the same
        thing.

        **But it may only overrule what it actually checked**, which is
        `exit_code_only`, and the gap it closes is a real one. A re-probe answers
        one question about one id. The package-manager route is the only one
        whose failure is a *claim about the exit code* and nothing else, so it is
        the only one allowed to be overruled by a green row. Every other route
        fails because something it did raised, and what raised is not necessarily
        what the row looks at: `_run_static` copies ffmpeg and then ffprobe, and
        a machine where the ffprobe copy threw had a green ffmpeg probe, a green
        row, no failure recorded anywhere, and a dub that died in
        `dubbing.audio` at the first ffprobe call. (The ffmpeg row now probes
        ffprobe too, see `setup.COMPANIONS`, which is the other half of that fix;
        this half is the rule that a probe cannot vouch for work it did not
        look at.) The same override laundered the thirty-minute timeout kill: the
        killed child had installed nothing, the row happened to be green from an
        earlier attempt, and "was still running after 30 minutes" was replaced
        with a success.

        So a timed-out `_run` passes `exit_code_only=False` as well, and the one
        case the override was written for, an exit code that lies about a machine
        that has the tool, is exactly the case it still covers.

        The other direction, a worker that said yes and a check that says no,
        picks its sentence by **which route ran** rather than by which table the
        id happens to appear in (`_disagreement`, one sentence per route). That distinction is not academic: a
        Mac with no Homebrew installs ffmpeg by the static route while `recipes`
        still holds `brew install ffmpeg`, so keying off the table produced a
        failure message naming a command nobody ran and offering a restart that
        could not have helped, since `tools/bin` is found without PATH at all.

        Where repeating the work could genuinely help, it is offered. A package
        manager can put a binary somewhere only a new process will see, so a
        restart is a real instruction with a real outcome. Where it could not, it
        is not offered, and this is the rule that was bought the hard way: "start
        the install again and it resumes what is missing" was written for a
        torn-off fetch, but the fetch that reaches this line is the one that just
        said `download complete`. Told to try again, the user got another
        successful download, another red row and the same sentence, forever. So
        the message names the disagreement, quotes what the check said so the bug
        is reportable, and stops. An app that asks a user to repeat something
        that already worked has stopped telling them the truth.
        """
        check = None
        # One try around everything between here and the lock, because
        # everything between here and the lock can raise and the slot is not
        # released until the lock is taken. The probe had its own guard and
        # `_disagreement` did not, which is a distinction the thread that dies
        # does not make: `_disagreement`'s RESTORE branch imports
        # `dubbing.segments` and its UV_ROUTE branch calls `uv_install_dir`,
        # which raises RuntimeError when neither HOME nor USERPROFILE is set. On
        # a partly synced venv the import fails, the worker thread dies with
        # `_running` still true, and every install button answers 409 until the
        # app is restarted. A message we could not compose is worth less than a
        # slot we can still use.
        try:
            try:
                check = self._probe(id_)
            except Exception as exc:                  # a probe must never strand the slot
                error = error or f"could not re-check {id_}: {type(exc).__name__}: {exc}"
            fresh = check is not None and bool(check.get("ok"))
            if fresh and not ok and exit_code_only:
                # The exit code complained and the machine has the thing anyway.
                # Keep what it said in the tail and stop calling this a failure.
                self._line(f"the installer was unhappy ({error or 'no reason given'}), "
                           f"but the re-check found {id_} on this machine, so it is "
                           "installed and this is not a failure")
                ok, error = True, None
            elif ok and check is not None and not fresh:
                ok = False
                error = error or self._disagreement(id_, check)
        except Exception as exc:
            ok = False
            error = error or (f"the install of {id_} could not be checked: "
                              f"{type(exc).__name__}: {exc}")
        with self._lock:
            self._running = False
            self._ok = ok
            self._error = error
            self._check = check
            self._finished = time.time()


class InstallQueue:
    """Everything that is missing, from one gesture — through the same one slot.

    The Setup screen gives every red row its own button, which is correct and,
    on a fresh machine, is nine buttons and forty minutes of coming back to
    check. This is the thing that presses them in order so the user does not
    have to. It installs nothing itself: it holds a list of ids and calls
    `Installer.start` for one of them at a time, which means the one-at-a-time
    rule, the re-probe, the resume and every refusal are still exactly the
    Installer's — there is no second install path to keep honest.

    Four decisions, all of them consequences of that:

    * **The list is computed on the server, from the report.** `setup.install_plan`
      reads a fresh `report()` and returns every missing row the app can actually
      fix, in grade order: blocking, then degrades, then optional. Nothing in the
      request chooses what runs; the button sends no body at all.
    * **The progress shape is the one that already exists.** A poll of
      `GET /api/setup/install` still answers the single slot's status — the id in
      flight, its tail, its bytes — with one extra `queue` block saying which
      item that is (`pos` of `items`), what is left, and what failed. The UI
      reuses its progress bar rather than growing a second one.
    * **Cancel stops after the current item, never during it.** A half-killed
      `brew` is a broken prefix, and a killed download would throw away nothing
      only because it resumes — but the user pressing Cancel means "stop
      queueing more", and stopping at the boundary is the one reading that
      leaves the machine in a state either side of the button can describe.
    * **A failure does not end the queue.** Nine downloads and one dead mirror
      should leave eight models on disk. The failed ids are named in the queue
      block; each one's own row is red again after the re-probe and says why.
    """

    def __init__(self, installer: Installer):
        self._installer = installer
        self._lock = threading.Lock()
        self._items: list[dict[str, Any]] = []
        self._pos = 0
        self._running = False
        self._cancelled = False
        self._failed: list[str] = []
        self._thread: threading.Thread | None = None
        self._started: float | None = None
        self._finished: float | None = None

    # -- api ---------------------------------------------------------------

    def start(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        """Queue `items` (`setup.install_plan`'s rows) and begin.

        Idempotent by design: pressing it twice is one gesture repeated, not an
        error, so a call while the queue runs answers the running queue. Only a
        *single* install started from a row holds the slot against it — that is
        the Installer's 409, raised here before anything is queued, because
        half a queue that dies on its first item is worse than a refusal.
        """
        with self._lock:
            # Our own item holding the slot is not a conflict — it is this
            # queue, and the answer to pressing the button again is the queue.
            # Anything else in it is a row's own install, and starting a plan
            # behind it would mean a first item that 409s on itself.
            if not self._running and self._installer.status()["running"]:
                raise busy("an install is already running; wait for it to finish, "
                           "then install the rest in one go")
            if not self._running:
                # The slot's leftovers are the previous gesture's story, and
                # this is a new one. Without this the answer to an empty plan
                # was the last single install's success, for an unrelated id.
                self._installer.reset()
                self._items = [dict(item) for item in items]
                self._pos = 0
                self._cancelled = False
                self._failed = []
                self._started = time.time()
                # An empty plan is the honest answer to "install everything" on
                # a machine with nothing missing: a queue that ran nothing, not
                # a 400 — and `finished` says so without a thread ever starting.
                # `status` still emits the block for it (`_started`, not
                # `_items`), because "a queue that ran nothing" needs somewhere
                # to be said.
                self._running = bool(self._items)
                self._finished = None if self._running else self._started
                if self._running:
                    self._thread = threading.Thread(target=self._run,
                                                    name="install-all", daemon=True)
                    self._thread.start()
        return self.status()

    def cancel(self) -> dict[str, Any]:
        """Stop after the item in flight. A no-op answer when nothing is queued —
        the button the user pressed is gone by the time a second press could
        land, and answering 409 to "stop" would be absurd."""
        with self._lock:
            if self._running:
                self._cancelled = True
        return self.status()

    def clear(self) -> None:
        """Forget a finished queue. Called when a row's own button starts an
        install: the header line describes the last queue, and a manual install
        is the end of that story — leaving it up would report a failure the user
        is in the middle of fixing by hand."""
        with self._lock:
            if not self._running:
                self._items = []
                self._failed = []
                self._pos = 0
                # And the block goes with it: `status` emits one for as long as
                # a queue has been started, so forgetting the queue has to be
                # forgetting that too.
                self._started = None
                self._finished = None

    def status(self) -> dict[str, Any]:
        """The single slot's status, plus a `queue` block while one exists.

        One body, because the screen polls one endpoint and the two facts are
        two halves of the same sentence: *what* is installing (the slot) and
        *where in the list* it is (here). `remaining_bytes` counts the items not
        yet finished and subtracts what the current download already has on
        disk, so the header's "~14 GB to go" falls as the bar fills.

        "While one exists" is a queue that has been *started*, not a queue with
        items in it. The empty plan is the case that distinction is for: pressing
        "install all" with nothing missing used to answer with no block at all,
        which is indistinguishable on the wire from never having pressed it, and
        left the screen rendering whatever the slot happened to hold. An empty
        `items` with `running: false` and a `finished` on it is the shape that
        says "this ran, and there was nothing to run".
        """
        body = self._installer.status()
        with self._lock:
            if self._started is None:
                return body
            items = [dict(item) for item in self._items]
            pos, running, cancelled = self._pos, self._running, self._cancelled
            failed, started, finished = list(self._failed), self._started, self._finished
        done = int(body.get("bytes_done") or 0) if running else 0
        remaining = max(0, sum(int(i.get("bytes") or 0) for i in items[pos:]) - done)
        body["queue"] = {"running": running, "cancelled": cancelled, "items": items,
                         "pos": pos, "total": len(items), "failed": failed,
                         "remaining_bytes": remaining, "started": started,
                         "finished": finished}
        return body

    def wait(self, timeout: float = 30.0) -> bool:
        """Join the queue thread. For tests and shutdown, never for a request."""
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    # -- the worker --------------------------------------------------------

    def _run(self) -> None:
        while True:
            with self._lock:
                if self._cancelled or self._pos >= len(self._items):
                    break
                id_ = str(self._items[self._pos]["id"])
            self._one(id_)
            with self._lock:
                self._pos += 1
        with self._lock:
            self._running = False
            self._finished = time.time()

    def _one(self, id_: str) -> None:
        """One item, start to exit. Every failure mode is recorded and stepped
        over — a refusal (an id the tables no longer carry), a worker that threw,
        or a worker that exited with `ok: False`. The reason is already in the
        slot's own `error` and in that row's re-probe; the queue only has to
        remember *which* ones so the header can say so at the end."""
        if not self._claim(id_):
            with self._lock:
                cancelled = self._cancelled
            if not cancelled:
                self._fail(id_)
            return
        # A join loop rather than one long join: `wait` takes the thread as it
        # was, and a poll of `status()` must never be the thing that blocks.
        while not self._installer.wait(1.0):
            pass
        if self._installer.status().get("ok") is not True:
            self._fail(id_)

    def _claim(self, id_: str) -> bool:
        """Take the one slot for `id_`, waiting out anyone else who has it.

        "Anyone else" is one thing: a row's own Install button, pressed in the
        gap between two queue items. That press wins the slot fairly, and it is
        over in a moment, but the queue used to read the resulting 409 as this
        item having failed and step past it, so a user who fixed one row by hand
        while "install everything" ran silently lost a different row from the
        queue, with nothing to say why. Waiting is the whole fix, and it is safe
        because the thing being waited for is the same one-at-a-time slot that
        would have run this item anyway.

        Bounded, because an unbounded wait is a queue that never ends: a slot
        held by a nine-gigabyte download is not something to sit behind forever,
        and after `SLOT_WAIT` this item is recorded as failed like any other and
        the queue moves on. Cancel is honoured while waiting; an item never
        started because the user cancelled is not a failure and is not recorded
        as one.
        """
        deadline = time.time() + SLOT_WAIT
        while True:
            try:
                self._installer.start(id_)
                return True
            except ApiError as exc:
                if exc.code != "busy":
                    return False             # a 400: no route for this id at all
            except Exception:
                return False
            with self._lock:
                if self._cancelled:
                    return False
            if time.time() >= deadline:
                return False
            time.sleep(SLOT_POLL)

    def _fail(self, id_: str) -> None:
        with self._lock:
            if id_ not in self._failed:
                self._failed.append(id_)


__all__ = ["DEMUCS_ID", "DIARIZATION_ID", "INSTALLERS", "Installer", "InstallQueue",
           "UV_ID", "UV_RELEASES", "demucs_argv", "download",
           "diarization_command", "diarization_route", "diarization_source",
           "diarization_target", "fetch_static_ffmpeg", "installable",
           "install_uv", "manual_command", "restore_diarization", "route",
           "snapshot_download", "static_route", "uv_install_dir",
           "uv_release_route", "uv_triple", "verify_diarization",
           "refresh_path", "registry_path_entries", "MANAGER", "STATIC",
           "UV_ROUTE", "RESTORE", "DOWNLOAD", "DEMUCS_ROUTE"]
