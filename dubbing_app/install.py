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

  Two ids are in neither table and have a button anyway, because their fix is
  not a snapshot. Diarization's weights shipped with the app, so the button
  copies them back from the payload (or restores them from the checkout) and
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
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

from dubbing import tools

from .errors import busy, invalid

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

Probe = Callable[[str], dict[str, Any] | None]

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
    it, so a machine with no button still has an answer instead of a red row."""
    source = diarization_source()
    if source is None:
        return None
    kind, where = source
    target = diarization_target()
    if kind == "git":
        return f"git -C {where} checkout -- {target.relative_to(where)}"
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
    ones it can fetch a static build of, the diarization weights when this
    machine still carries the copy they shipped in, and the Demucs cache, which
    is always yes: the fetch is demucs's own and needs nothing from this machine
    but a network. Read by `setup.report` and
    `setup.probe` so the flag that puts a button on a row and the code behind
    the button are the same answer — a row whose POST is a 400 is worse than no
    button, and so is a missing button for an install that would have worked."""
    if id_ == DIARIZATION_ID:
        return diarization_source() is not None
    if id_ == DEMUCS_ID:
        return True
    return id_ in INSTALLERS or static_route(id_)


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
            # A platform whose package manager cannot run unattended contributes
            # no recipes at all (Linux: `sudo apt-get` wants a password), so
            # ffmpeg arrives here with no argv — and still has a route. The
            # static build is the only button that can exist there, and without
            # it "install everything" would silently skip the one tool every
            # stage shells out to.
            if static_route(id_):
                return self._begin(
                    id_, [f"# no package manager here can install {id_} unattended, "
                          f"so installing a static ffmpeg/ffprobe build into "
                          f"{tools.tools_bin()}"],
                    target=self._run_static, args=(id_,))
            if id_ == DEMUCS_ID:
                # A subprocess like the package managers, and for the same
                # reason: it imports torch and demucs, which the server process
                # spends its life not importing (`setup.py` rule one).
                demucs = demucs_argv()
                return self._begin(
                    id_, ["$ " + " ".join(demucs),
                          "fetching the stem-separation weights into the cache the "
                          "first stems run would fill anyway"],
                    target=self._run, args=(id_, demucs))
            if id_ == DIARIZATION_ID and installable(id_):
                source = diarization_source()
                assert source is not None            # installable() just said so
                return self._begin(
                    id_, [f"# restoring the bundled diarization weights from "
                          f"{source[1]}, and verifying them against {DIARIZATION_SUMS}"],
                    target=self._run_restore, args=(id_,))
            spec = self.downloads.get(id_)
            if spec is None:
                raise invalid(self._refusal(id_))
            return self._start_download(id_, spec)
        manager = argv[0]
        if shutil.which(manager) is None:
            # The recipe is here but the manager is not: a brewless Mac, a
            # Windows with no winget. "Install the package manager first" dies
            # at a terminal, so take the static route instead of refusing.
            # Checked here, not at table-build time, so a brew installed
            # mid-session is picked up on the next press.
            if static_route(id_):
                return self._begin(
                    id_, [f"# {manager} is not on this machine, so installing a "
                          f"static ffmpeg/ffprobe build into {tools.tools_bin()}"],
                    target=self._run_static, args=(id_,))
            template = MANAGERS.get(manager,
                                    "`{manager}` is not on PATH: install `{command}` by hand.")
            raise invalid(template.format(tool=id_, command=" ".join(argv), manager=manager))
        return self._begin(id_, ["$ " + " ".join(argv)],
                           target=self._run, args=(id_, tuple(argv)))

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
        return self._begin(id_, lines, target=self._run_download,
                           args=(id_, hub, local_dir), dl_dir=local_dir, dl_total=total)

    def _begin(self, id_: str, lines: list[str], *, target: Callable[..., None],
               args: tuple, dl_dir: Path | None = None,
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
            proc = self._spawn(list(argv), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               text=True, encoding="utf-8", errors="replace",
                               bufsize=1, env=self._env())
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

    def _run_download(self, id_: str, hub: str, local_dir: Path) -> None:
        """The worker behind a model install: one `snapshot_download` into the
        directory the pipeline will read. No timeout, deliberately — 81 GB on
        home DSL is slower than any brew, there is no child process to kill,
        and an abandoned attempt costs nothing because the next one resumes
        from the partial files.
        """
        ok, error = False, None
        try:
            # Looked up as a module global so a test's monkeypatch is what runs.
            snapshot_download(repo_id=hub, local_dir=str(local_dir))
            ok = True
            self._line("download complete")
        except Exception as exc:                       # network, auth, disk — all of it
            error = (f"{type(exc).__name__}: {exc} partial files are kept, "
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
            if id_ in self.recipes:
                error = error or (f"`{' '.join(self.recipes.get(id_, ()))}` succeeded but "
                                  f"{id_} is still not there. Restart the app so it picks "
                                  "up the new PATH.")
            else:
                error = error or (f"the download finished but {id_} still fails its "
                                  "check: the files may be incomplete; start the "
                                  "install again and it resumes what is missing.")
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
        try:
            self._installer.start(id_)
        except Exception:                        # a 400/409 raised as an HTTPException
            self._fail(id_)
            return
        # A join loop rather than one long join: `wait` takes the thread as it
        # was, and a poll of `status()` must never be the thing that blocks.
        while not self._installer.wait(1.0):
            pass
        if self._installer.status().get("ok") is not True:
            self._fail(id_)

    def _fail(self, id_: str) -> None:
        with self._lock:
            if id_ not in self._failed:
                self._failed.append(id_)


__all__ = ["DEMUCS_ID", "DIARIZATION_ID", "INSTALLERS", "Installer", "InstallQueue",
           "demucs_argv",
           "diarization_command", "diarization_route", "diarization_source",
           "diarization_target", "fetch_static_ffmpeg", "installable",
           "manual_command", "restore_diarization", "route", "snapshot_download",
           "static_route", "verify_diarization"]
