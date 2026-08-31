"""First-run environment checks "can this machine actually run the pipeline?".

The desktop shell needs an answer before it shows an import form, because every
failure mode here is undiagnosable from inside the editor: a missing `ffmpeg`
surfaces as a stage that dies halfway, and an absent model directory silently
becomes a multi-gigabyte download on the first run.

Two rules shape this module:

* **No model is loaded and no weights are touched.** Every check is `which`,
  `os.environ` or `stat`, so `GET /api/setup` answers in milliseconds and can be
  polled. The pipeline imports here are function-local and reach only for
  constants none of them pulls torch or MLX into the server process.
* **The paths come from the pipeline's own constants**, never restated here. If
  `translate.MODEL_PATH` moves, this check moves with it; a hardcoded copy would
  drift and then cheerfully report a green tick for a model the pipeline cannot
  find.

Report shape::

    {"ok": bool, "checks": [{"id", "label", "ok", "detail", "required": bool,
                             "severity": "blocking"|"degrades"|"optional",
                             "state": "missing"|"incomplete"|"ready",
                             "installable": bool, "path"?, "found_at"?, "bytes"?,
                             "hub"?, "download_bytes"?, "downloading"?, "fix"?}, ...]}

`installable` is the server's answer to "can the app fix this for me?" true for
exactly the ids `dubbing_app.install` can install (`install.installable`: an argv
for this platform, the static ffmpeg build where no unattended package manager
exists, or the bundled diarization weights when this machine still carries a copy
to restore them from) plus the models in `model_downloads()`, the hub snapshots
the app can fetch itself. The UI needs it
as a flag rather than a list of its own, or the two sides drift and a button
appears on a row whose `POST /api/setup/install` is a 400. A downloadable model
row also carries `hub` and `download_bytes`, so the button can say what it costs
("Download (~9.7 GB)") before it is pressed. `fix` is the same courtesy for a row
whose repair is not a download: the command, as data, so the UI can offer to copy
it rather than making the user select it out of a sentence.

`path` is always where the pipeline *would* load a model from, even when it is
found somewhere else; `found_at` says where it actually is when the two differ
(a model the loader auto-downloaded into the Hugging Face cache).

`ok` is the conjunction of the **required** checks only.

`severity` is what `required` could never say. A boolean has two values and this
question has three, so everything that was not required was reported as one
undifferentiated "informational" which put a missing diarization model (every
speaker in the video collapsed into one) on the same row as a Korean TTS
checkpoint a Hebrew→English run will never open. Three grades, and the third is
what makes the first two mean anything:

* ``blocking`` the run **fails** without it. `ffmpeg`, the translator, the
  default TTS checkpoint and the English verifier. Exactly the ``required``
  set, and `required` is derived from this so the two cannot drift. It is a
  short list on purpose: `sox` and `uv` are both command-line tools this
  project cannot be developed without and neither of them stops a dub, so
  neither is blocking, and reading "the command-line tools" as a category was
  how uv came to be REQUIRED on machines it was already running.
* ``degrades`` the run **works and is worse**. No language-ID model means
  foreign speech is never detected; no diarization weights mean every character
  in the video is dubbed in one voice. Nothing here stops a run, and nothing here
  is nothing.
* ``optional`` irrelevant until you ask for it: the per-language-pair models,
  the self-downloading caches, and free disk.

`state` is to `ok` what `severity` is to `required`: the boolean has two values
and a model directory has three. A first-run fetch writes `config.json`,
`tokenizer.json` and `model.safetensors.index.json` in the first second or two
and then spends minutes on the shards, so "the directory exists and is not
empty" called a 6.4 GB model READY at 1% downloaded, for the whole download and
forever after an interrupted one. Three states, and the third is the one that
was missing:

* ``missing`` nothing is there. Press Download.
* ``incomplete`` some of it is there and it cannot be loaded. Either a fetch is
  running right now (wait) or one stopped (press Download; `hf download` resumes
  from the partial files). Which of the two is on the row, because they are
  different instructions.
* ``ready`` every file the loader opens is on disk.

`ok` is `state == "ready"` and nothing else, so no client has to learn the new
field to stop believing a half-downloaded model. `downloading` is on an
incomplete row when a fetch is live, which is what lets a screen keep polling
until it turns green instead of asking the user to press Re-check.

There is no credential row of any kind, and that is deliberate rather than an
oversight. A Hugging Face token once sat here, first as ``degrades`` and then as
``optional``, because diarization loaded a gated repo. Since v0.4.0 the
CC-BY-4.0 weights ship inside the app (``segments.DIARIZATION_DIR``) and every
other model is public, so a machine that has never signed in to Hugging Face is
not missing anything and has no row to read. ``HF_TOKEN`` and
``DUB_DIARIZATION_HUB`` still work for anyone who wants the gated upstream repo
instead; that is a setting in ``.env``, not a checklist item.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# The three grades a check can be. See the module docstring; `required` is
# derived from this and never passed in, so no caller can declare a check
# blocking and not required (or the reverse) by hand.
BLOCKING, DEGRADES, OPTIONAL = "blocking", "degrades", "optional"
SEVERITIES = (BLOCKING, DEGRADES, OPTIONAL)

# The three states a check can be *in*, as opposed to what it costs. Same
# discipline as the grades: `ok` is derived from this and never passed beside
# it, so nothing can be READY and not ok, or the reverse.
MISSING, INCOMPLETE, READY = "missing", "incomplete", "ready"
STATES = (MISSING, INCOMPLETE, READY)

# The file a sharded checkpoint carries, and what makes "is this whole?" exact
# rather than probable: its `weight_map` names every shard the loader opens.
SHARD_INDEX = "model.safetensors.index.json"

# The fraction of the expected download that has to be on disk before a model
# with no shard index and no install receipt is called whole.
#
# It was 0.9 for one day, and 0.9 produced the exact lie the paragraph it was
# written under promised it would not: `model.lid` was declared at a rounded
# 100 MB, the real complete snapshot is 86.4 MB, the floor demanded 90 MB, and a
# working language-ID model went red minutes after the check shipped. Two things
# came out of that. The declared sizes below are now measured rather than
# guessed, and, more to the point, a floor is no longer the *first* answer to
# "did this download finish". A shard index answers it exactly, and a download
# this app ran itself answers it from the installer's own return value
# (`record_install`). What is left for the floor is the case where neither
# exists: a model somebody else's `hf download` or the first-run auto-fetch put
# on disk, which this app never watched and cannot ask.
#
# For that case 0.75 is both looser and still meaningful. What the floor has to
# catch is a torn-off download, and a torn-off download is not 15% short. The
# bug in #17 was a 6.4 GB model certified at 1%, and an interrupted fetch stops
# with whole shards missing, which is gigabytes, not a rounding error. What 0.9
# was catching instead was disagreement between a hand-entered number and a
# disk, which is not a fact about the model at all. A quarter of a download
# missing is still a gap no estimate explains.
SIZE_FLOOR = 0.75

# The file `record_install` leaves beside a model whose download this app ran to
# completion, and the reason the retry loop below cannot come back. See it.
INSTALL_RECEIPT = ".mambodubb-download-complete"

# How recently a download's lock or partial file must have been touched for the
# fetch to count as live. Long enough that a slow shard between two writes is
# not read as abandoned, short enough that a fetch killed with the app is not
# still "downloading" the next morning.
FETCH_FRESH_SECONDS = 120.0

# A run needs headroom for the source media, the stems, every clip and the
# re-encoded preview. Well under a real feature film; enough to catch "this disk
# is full" before a two-hour job discovers it.
DISK_WARN_BYTES = 10 * 1024**3

_COMMIT_ENV = "DUBBING_STUDIO_COMMIT"


# ---------------------------------------------------------------------------
# version stamp
# ---------------------------------------------------------------------------

_commit: str | None = None
_commit_read = False


def git_commit(*, refresh: bool = False) -> str | None:
    """The short commit this server was built from, or None.

    Read once and cached: `/health` is polled, and a `git` subprocess per poll
    would be absurd. A packaged app has no `.git` at all, so the env var is the
    baked-in path set it at build time and the binary reports its provenance
    without needing a repository, exactly as the packaged worker does.
    """
    global _commit, _commit_read
    if _commit_read and not refresh:
        return _commit
    _commit_read = True
    baked = (os.environ.get(_COMMIT_ENV) or "").strip()
    if baked:
        _commit = baked
        return _commit
    _commit = None
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=5.0)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode == 0:
        _commit = out.stdout.strip() or None
    return _commit


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------

def check(id_: str, label: str, ok: bool, detail: str, *, severity: str = BLOCKING,
          state: str | None = None, **extra: Any) -> dict[str, Any]:
    """One row. `required` is `severity == "blocking"`, computed here and nowhere
    else the old contract, still on the wire, now with one source of truth.

    `state` is the same arrangement for the other axis. Most checks have nothing
    to say beyond yes or no a binary is on PATH or it is not so they pass
    nothing and get `ready`/`missing` derived from `ok`. Only a model directory
    has a third answer, and it passes `state=INCOMPLETE` with `ok=False`: a
    partial download is not ready, and any client that has never heard of this
    field still reads it as not ready, which is the whole point of deriving one
    from the other in one place.
    """
    if severity not in SEVERITIES:
        raise ValueError(f"unknown severity {severity!r}")
    ok = bool(ok)
    if state is None:
        state = READY if ok else MISSING
    if state not in STATES:
        raise ValueError(f"unknown state {state!r}")
    if ok is not (state == READY):
        raise ValueError(f"{id_}: ok={ok} disagrees with state={state!r}")
    return {"id": id_, "label": label, "ok": ok, "detail": detail,
            "severity": severity, "required": severity == BLOCKING,
            "state": state, **extra}


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def dir_size(path: Path) -> int:
    """Bytes this model actually occupies under `path`, counting each file once.

    Symlinks are followed, and that is a correction rather than a nicety. A
    snapshot can arrive deduplicated: `huggingface_hub` links a file it already
    holds in the shared cache instead of copying it, and speechbrain's
    `from_hparams` fills its `savedir` with symlinks outright: the language-ID
    directory in the development checkout is four symlinks into
    `~/.cache/huggingface/hub` and no regular files at all. Not following them
    measured that complete model at zero bytes, which the size floor then read
    as a download that had not started. The bytes are on the disk; whose tree
    they happen to be counted under is an accident of deduplication, and a
    readiness check that turns that accident into a red row is measuring the
    wrong thing.

    Followed, but never counted twice. The Hugging Face cache is `blobs/` with
    `snapshots/<rev>/` symlinked at it, so every file in a cached repo is
    reachable by two paths and a naive sum reports a cached model at double its
    size, which would hand the floor a free pass instead of a false red: the
    same lie the other way round. Identity is `(st_dev, st_ino)`, which is what
    "the same file" means to the filesystem.

    Directories are still not recursed *through* a symlink; that is where the
    cycles are, and `rglob` declines to by default. Unreadable entries and
    broken links are skipped: a size is a nicety, never a reason to 500.
    """
    total = 0
    seen: set[tuple[int, int]] = set()
    try:
        for entry in path.rglob("*"):
            try:
                if not entry.is_file():          # follows the link; a broken one is not
                    continue
                st = entry.stat()
            except OSError:
                continue
            key = (st.st_dev, st.st_ino)
            if key in seen:
                continue
            seen.add(key)
            total += st.st_size
    except OSError:
        return total
    return total


# The command-line tools, in one table so a single one can be re-checked after
# an install without re-running (and re-`stat`ing) the whole model report. A row
# is (label, exe, why) with an optional fourth element, the severity — blocking
# when absent, because "a run dies partway through without it" is this file's
# definition. Of the three rows here it is true of exactly one, ffmpeg; sox and
# uv both say so in their own comments below, and both carry the grade to match.
TOOLS: dict[str, tuple[str, ...]] = {
    "ffmpeg": ("ffmpeg", "ffmpeg", "every stage shells out to it for audio and video"),
    # Downgraded from blocking after reading what actually runs: the only sox
    # caller in the tree is qwen_tts's 25Hz tokenizer (`speech_vq.XVectorExtractor`),
    # and this pipeline loads only 12Hz checkpoints, whose x-vector path never
    # touches it. The `sox` *Python* package imports fine with the binary absent
    # (verified: pysox logs a warning and sets NO_SOX), so a brewless Mac dubs
    # without it. The row stays so a future 25Hz experiment finds the button.
    "sox": ("SoX", "sox", "nothing the shipped pipeline runs needs it. Only "
            "qwen_tts's 25Hz tokenizer would, and this pipeline loads 12Hz "
            "checkpoints", OPTIONAL),
    # Two things about this row used to be wrong, and the second one was wrong
    # in the way this whole grading scheme exists to prevent.
    #
    # It said it was not installable from here, borrowing `install.MANAGERS`'s
    # argument that the tool that installs the tools cannot itself be installed
    # from here. That argument is about *package managers*: Homebrew and winget
    # are the thing the button drives, so a button that drives them to install
    # themselves has nowhere to stand. uv is not that. It is one static binary
    # with an official signed release per platform, astral publishes a `.sha256`
    # beside every archive, and `install-server.sh`, `install-server.ps1` and
    # `scripts/stage_desktop_payload.py` in this same repo all already download
    # and verify it that way. The button now does the same
    # (`install.uv_release_route`), and there is no circularity: installing uv
    # needs no uv. The one place the order matters is the other direction,
    # `install.fetch_static_ffmpeg`, which uses uv to put the static-ffmpeg
    # wheel in this venv, and that is an argument for the button existing
    # rather than against it.
    #
    # And it was BLOCKING, which was untrue on every machine that read it. This
    # file defines blocking as "the run **fails** without it", and no run fails:
    # the server is already up in its own environment, and
    # `runner.SubprocessRunner` spawns every job child with `sys.executable`,
    # never with uv. Nothing shells out to it during a dub. The old comment said
    # exactly that and then graded it blocking anyway, which is how a desktop
    # user whose app was *launched by* the bundled uv came to be shown REQUIRED
    # and MISSING beside a machine that dubs perfectly.
    #
    # OPTIONAL, then, and by this file's own rule rather than by taste.
    # `degrades` means the run works and is worse, and a run without uv is not
    # worse, it is identical. `optional` means irrelevant until you ask for it,
    # and what asks for it is repairing or updating the environment (`uv sync`),
    # every command in AGENTS.md, and the brewless static-ffmpeg route. Those
    # are real, they are named below, and none of them is a dub.
    "uv": ("uv", "uv", "no stage of a dub shells out to it, because this server is "
                       "already running in its environment and every job child is "
                       "spawned with the same interpreter. What needs it is repairing or "
                       "updating that environment (`uv sync`), and every command in "
                       "AGENTS.md. Get it from https://docs.astral.sh/uv/",
           OPTIONAL),
}


# `uv` discovery, kept in step with `find_uv()` in the shell's `workspace.rs`:
# same env override, same literal paths *per platform*, same PATH rule, same
# home-relative tail, same order. Two different rules for the same tool is
# exactly the bug this exists to close: the Rust side found uv, started this
# server with it, and the server then reported the tool that launched it as
# missing and required.
#
# It had drifted, and it had drifted in the one direction nobody here could
# see: every literal below was a macOS path and the home-relative tail was
# spelled `uv` with no suffix, so a Windows machine with uv installed exactly
# where its own installer puts it (`%USERPROFILE%\.local\bin\uv.exe`) had a red
# row and a working uv. A Mac-only fallback list is not a fallback list; it is
# a fallback list for the machine the author was sitting at.
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


def uv_exe(platform: str | None = None) -> str:
    """The binary's name, which is the only part of this lookup spelled
    differently per platform (`workspace.rs`: `UV_EXE`)."""
    from dubbing import tools as tool_recipes

    return "uv.exe" if tool_recipes.platform_key(platform) == "win32" else "uv"


def uv_fallbacks(platform: str | None = None) -> tuple[str, ...]:
    from dubbing import tools as tool_recipes

    return UV_FALLBACKS[tool_recipes.platform_key(platform)]


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


def uv_on_path(exe: str, env: dict[str, str] | None = None) -> str | None:
    """`which`, near enough, and near enough to `workspace.rs`'s `find_in_path`.

    Written out rather than handed to `shutil.which` because the two disagree
    on Windows in the direction that matters: `shutil.which("uv")` appends the
    PATHEXT suffixes and never tries the bare name, and `shutil.which("uv.exe")`
    never tries anything else, while the Rust tries `uv.exe` then `uv` inside
    each directory in turn. A Git-Bash-style extensionless shim is worth
    finding, and a directory holding both must resolve the same way on both
    sides or the shell and the server disagree about which uv this machine has.
    """
    environ = os.environ if env is None else env
    raw = environ.get("PATH")
    if not raw:
        return None
    bare = exe[:-4] if exe.endswith(".exe") else exe
    names = (exe,) if bare == exe else (exe, bare)
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
    4. PATH, `.exe` first on Windows.
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


def tool(id_: str, label: str, exe: str, why: str,
         severity: str = BLOCKING) -> dict[str, Any]:
    """One tool row. `tools.resolve_tool` is the whole probe — the same lookup
    every pipeline call site uses: env override, then the workspace `tools/bin`
    (where the brewless-Mac static build lands), then PATH with PATHEXT, so
    `ffmpeg.exe` on Windows answers to the same lookup as `ffmpeg` elsewhere.
    The one exception is `uv`, which gets `find_uv()`'s fuller chain (above).

    A missing row carries **this platform's** install command, because that is
    the only actionable half of "it is not here" and the row is where a user
    with no button looks for it (`dubbing.tools`; the Setup screen's button
    exists only where that command can run unattended). A platform with no
    command still gets the button's sentence when there is a route that is not
    a package manager at all, which is the Linux uv row: no apt package exists,
    so there is nothing to type, and the button downloads the official release
    instead. Where there *is* a button, the row also says which route pressing
    it takes, `via Homebrew` or the static build into the workspace, because
    a button that might run a package manager or might download a binary is a
    button nobody can trust."""
    from dubbing import tools as tool_recipes

    found = find_uv() if exe == "uv" else tool_recipes.resolve_tool(exe)
    # "not on PATH" for the tools whose probe is PATH and two overrides of it.
    # uv's probe is five rungs of which PATH is one, so naming PATH there would
    # send a user to check the one place that was least likely to be the answer.
    missing = "not found" if exe == "uv" else "not on PATH"
    detail = found or f"{exe} {missing}: {why}"
    if not found:
        command = tool_recipes.command(id_)
        if command:
            detail += f". Install it: `{command}`"
        from .install import route

        how = route(id_)
        if how:
            detail += f". The Install button installs it {how}"
    return check(id_, label, bool(found), detail, severity=severity, path=found)


def probe(id_: str) -> dict[str, Any] | None:
    """One check, by id or None if it is not one this can answer alone.

    Exactly the ids the app can install: the tools, re-checked with a fresh
    `shutil.which`, the hub-snapshot models, re-`stat`ed on disk, and the
    diarization weights the restore route puts back. A check that is none of
    those (`disk`, the token, the self-downloading caches) is nothing the
    install slot ever finishes, so nothing ever asks for it here.
    `dubbing_app.install` calls this when its worker exits, so the row the UI
    redraws is fresh evidence and not the worker's opinion of itself.
    """
    from .install import installable

    spec = TOOLS.get(id_)
    if spec is None:
        if id_ not in model_downloads() and not installable(id_):
            return None
        row = next((c for c in model_checks() if c["id"] == id_), None)
        if row is None:
            return None
        return {**row, "installable": True,
                **({"stage": blocking_stage(id_)} if row["severity"] == BLOCKING else {})}
    # Same row `report` would produce, `installable` and `stage` included: a
    # client that drops this straight into its list must not get a shape one key
    # short it would redraw a REQUIRED row as an untagged one.
    row = tool(id_, *spec)
    return {**row, "installable": installable(id_),
            **({"stage": blocking_stage(id_)} if row["severity"] == BLOCKING else {})}


def index_shards(root: Path) -> list[Path] | None:
    """Every weight file `root`'s shard index names, or None when it has no index.

    One JSON parse for an exact answer. A sharded checkpoint ships
    `model.safetensors.index.json`, whose `weight_map` maps each tensor to the
    file holding it, so the set of files the loader will open is written down
    beside them: "is this download finished?" stops being a guess about bytes
    and becomes a list of paths to `stat`.

    Looked for by walk rather than at a fixed name, because the two places a
    model lives put it in two places: `models/<dir>/model.safetensors.index.json`
    for a `--local-dir` fetch, and `snapshots/<rev>/…` inside the Hugging Face
    cache. None means "this model does not answer that way" a single-file
    checkpoint, a faster-whisper directory, a speechbrain one and the caller
    falls back to the size floor rather than treating the absence as a failure.
    """
    index = next(iter(sorted(root.rglob(SHARD_INDEX))), None)
    if index is None:
        return None
    try:
        data = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None                      # a half-written index is not a manifest
    weight_map = data.get("weight_map") if isinstance(data, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        return None
    names = dict.fromkeys(v for v in weight_map.values() if isinstance(v, str))
    return [index.parent / name for name in names]


def record_install(root: Path) -> None:
    """Write the receipt that says "this app downloaded this model, and the
    download returned rather than raised".

    The size floor is a heuristic standing in for a question the installer never
    had to guess at. `snapshot_download` either returns or throws, and when it
    returns every file in the repo is on disk by definition, with no estimate, no
    rounding and no argument with a hand-entered table. Leaving that fact
    unrecorded is what built the loop this exists to end: the installer finished
    a download, `model_ready` disagreed with a number somebody typed into
    `model_downloads()` months earlier, and the slot told the user to press
    Download again, which succeeded again, and was disagreed with again. A
    receipt turns "how many bytes does the table think this should be" into "did
    the thing that actually knows say yes", and a completed download can no
    longer be re-flagged by an arithmetic it was never party to.

    The measured size and the timestamp go in the file for a human reading it,
    and deliberately not into the check. Turning the recorded size into a second
    floor would rebuild the trap out of better numbers: `.cache/huggingface`
    metadata gets cleaned, a shared blob gets relinked, and the model would go
    red for having become tidier. The receipt is a yes or it is absent.

    Best effort. A read-only tree or a full disk must not turn a finished
    download into a failed one; without the receipt the floor still answers,
    which is exactly where this was before.
    """
    try:
        root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (root / INSTALL_RECEIPT).write_text(
            f"{stamp}\n{dir_size(root)}\n", encoding="utf-8")
    except OSError:
        pass


def install_recorded(root: Path) -> bool:
    """Did a download this app ran to completion leave its receipt in `root`?"""
    try:
        return (root / INSTALL_RECEIPT).is_file()
    except OSError:
        return False


def model_ready(root: Path, size: int, hub_bytes: int = 0) -> bool:
    """Is everything the loader opens actually in `root`?

    Three answers, strongest first, and the order is the whole design.

    A shard index is exact: its `weight_map` names every file the loader opens,
    they either exist or they do not, and nothing may overrule it: a receipt
    for a download that has since had a shard deleted is a claim about the past,
    while the index is a claim about the disk.

    An install receipt is testimony from the one party that knows.
    `record_install` writes it when `snapshot_download` returns, so for a model
    this app fetched, "is the download finished" is answered by the process that
    finished it instead of by measuring bytes against a guess. This is the rung
    that was missing, and its absence is the whole of #17's regression: a
    complete 86.4 MB language-ID snapshot, a table that said 100 MB, a floor at
    0.9, and a red row on a model the pipeline loads happily.

    A size floor is last because it is the only one that is a guess, and it is
    all that is left for a model this app did not download: the first-run
    auto-fetch, or a user's own `hf download`. A model with no expected size at
    all is still taken at its word, because there is nothing to compare it to, and
    calling every unmeasured model incomplete would be the "check that cannot
    pass" this module's second rule forbids.
    """
    shards = index_shards(root)
    if shards is not None:
        return all(shard.is_file() for shard in shards)
    if install_recorded(root):
        return True
    if hub_bytes:
        return size >= SIZE_FLOOR * hub_bytes
    return True


def fetch_in_flight(root: Path) -> bool:
    """Is something writing this model right now, or did it stop?

    The difference between "wait" and "press Download", and it is readable
    without asking any process anything: `huggingface_hub` keeps a `.lock` and a
    `.incomplete` beside every file it is fetching `<local_dir>/.cache/
    huggingface/download/` for a `--local-dir` download, `blobs/` inside the
    cache and touches them as the bytes land. Fresh means a fetch is live;
    stale means one died and the partial files are waiting for the next attempt
    to resume them.

    Only those two directories are walked, never the model tree: the shards are
    the big part of a `stat` sweep and this runs on every poll.
    """
    cutoff = time.time() - FETCH_FRESH_SECONDS
    for marks in (root / ".cache" / "huggingface" / "download", root / "blobs"):
        if not marks.is_dir():
            continue
        try:
            entries = list(marks.rglob("*"))
        except OSError:
            continue
        for entry in entries:
            if entry.suffix not in (".lock", ".incomplete"):
                continue
            try:
                if entry.stat().st_mtime >= cutoff:
                    return True
            except OSError:
                continue
    return False


def _model_location(path: Path, hub: str, hub_cached: bool,
                    hub_bytes: int) -> tuple[Path, int, str]:
    """Where this model is, how many bytes are there, and which state that is.

    The local directory first, then the Hugging Face cache when every loader of
    this model accepts the hub id (`hub_cached`). `model_ready` decides between
    READY and INCOMPLETE; nothing on disk at all is MISSING.
    """
    present = path.is_dir() and any(path.iterdir())
    where = path
    if not present and hub and hub_cached:
        cached = hf_cache_repo(hub)
        if cached is not None:
            present, where = True, cached
    size = dir_size(where) if present else 0
    if not present:
        return where, size, MISSING
    return where, size, (READY if model_ready(where, size, hub_bytes) else INCOMPLETE)


def _model_detail(state: str, path: Path, where: Path, size: int, *, hub: str,
                  hub_bytes: int, note: str, fix: str, downloading: bool) -> str:
    """The sentence under the row: what is there, and what to do about it."""
    if state == READY:
        origin = f"{path}" if where == path else f"the Hugging Face cache ({where})"
        return f"{human_bytes(size)} in {origin}"
    if state == INCOMPLETE:
        # The two halves a user needs in this state and in no other: how far in
        # it is, and whether anything is still working on it. "Incomplete" alone
        # would leave them staring at a screen with nothing to decide.
        progress = (f"{int(100 * size / hub_bytes)}% of {human_bytes(hub_bytes)}"
                    if hub_bytes else f"{human_bytes(size)} so far")
        detail = (f"incomplete: {human_bytes(size)} in {where}: "
                  + (f"downloading, {progress}" if downloading
                     else "partial download, press Download to finish"))
        if hub:
            detail += f". It resumes: `uv run hf download {hub} --local-dir {path}`"
        elif fix:
            detail += f". Restore it: `{fix}`"
        return detail
    detail = f"missing: {path}" + (f" ({note})" if note else "")
    if hub:
        approx = f" (~{human_bytes(hub_bytes)})" if hub_bytes else ""
        detail += f". Fetch it{approx}: `uv run hf download {hub} --local-dir {path}`"
    elif fix:
        detail += f". Restore it: `{fix}`"
    return detail


def _model_extra(path: Path, where: Path, size: int, *, hub: str, hub_bytes: int,
                 fix: str, downloading: bool) -> dict[str, Any]:
    """The row's data fields, beside the sentence.

    `path` stays the pipeline's own constant whichever place answered: it is
    where this model belongs, and a row that renamed itself after the cache
    would stop being checkable against `dubbing`'s constants. Where it was
    actually found, when that is somewhere else, is a second field.

    `downloading` goes on only where it is true, and only where it can be: a
    live fetch is the one state on this screen that resolves itself, and it is
    what tells a client to keep polling instead of leaving a stale row under a
    Re-check button.
    """
    extra: dict[str, Any] = {"path": str(path), "bytes": size}
    if where != path:
        extra["found_at"] = str(where)
    if hub:
        extra["hub"] = hub
        extra["download_bytes"] = hub_bytes
    if fix:
        extra["fix"] = fix
    if downloading:
        extra["downloading"] = True
    return extra


def model(id_: str, label: str, path: Path, *, severity: str = BLOCKING,
          note: str = "", hub: str = "", hub_bytes: int = 0,
          hub_cached: bool = False, fix: str = "") -> dict[str, Any]:
    """A model's presence and size, in **either** place the pipeline reads it
    from. `path` always comes from the pipeline module that loads it, so this
    cannot describe a stale location.

    Two locations, because the loaders have two. `translate.py` and `tts.py`
    both fall back to the hub id when `models/<dir>` is absent, and a hub-id
    load downloads into the Hugging Face cache and never into `models/`. A probe
    that stats only the local directory therefore reports MISSING forever on a
    machine where translation and TTS demonstrably work, and the file it is
    looking for is never going to appear. `demucs_check` learned this first and
    its reasoning is the general one: a check that cannot pass teaches the user
    to ignore the whole screen, and a *blocking* row that cannot pass teaches it
    hardest. So a known `hub` is also looked for in the cache
    (`models--org--name` with something under `blobs/`), and a row that passes
    there says which of the two places answered.

    `hub_cached` is what keeps that from becoming the opposite lie, and it is
    per model rather than global because the fallback is. Language ID
    (`transcript.py`: `if not LID_MODEL.is_dir(): return None`) and the Hebrew
    LoRA (`hebrew.attach_adapter`) read the local directory and nothing else, so
    a cached copy of those does not make the feature work, and a green row for
    one would be exactly the "cheerful tick for a model the pipeline cannot
    find" this module's second rule exists to prevent. A model is marked cached
    only when **every** loader that opens it accepts the hub id.

    A missing row's whole job is to be actionable: when the hub repo is known,
    the detail carries the exact download command (backticked so the UI sets it
    as code) and the row carries `hub` and `download_bytes`, which is what lets
    the Setup screen label its button "Download (~9.7 GB)" instead of just
    "Download". "Missing" without the command or the size is a scavenger hunt.
    `fix` is the same promise for a model with no hub to fetch from: the command
    that repairs it, backticked, and on the row as data so a UI can copy it.

    **A directory with something in it is not a model.** This used to be the
    whole test (`path.is_dir() and any(path.iterdir())`), and the auto-fetch on
    first launch turned that into the exact failure the paragraph above is about,
    pointed the other way: `hf download` writes the config, the tokenizer and the
    shard index within a second or two and then spends minutes on the weights, so
    a *blocking* row went green at 1% of a 6.4 GB download and stayed green for
    the whole fetch and forever after an interrupted one. A user who opened Setup
    during the first-run download saw every blocking row certified, started a dub
    and got a load failure from a model this screen had just passed. So presence
    is now `model_ready` an exact shard list where there is one, the installer's
    own receipt where there is one, the size floor where there is neither and
    the middle state has a name (`INCOMPLETE`) and a sentence saying which of
    "wait" and "press Download" it means (`fetch_in_flight`).

    **And a model that is whole is not a partial download.** The paragraph above
    warned that a red row on a working model would be the same lie backwards,
    and then shipped one within the hour: `model.lid`, complete at 86.4 MB,
    against a table that said 100 MB and a floor that wanted 90. The user pressed
    Download, the download reported success, the row stayed amber, and the
    install slot told them to press Download again, a loop with no exit that
    this check built by treating a rounded guess as evidence. What came out of
    it is the ordering in `model_ready`: measurement is the last resort, not the
    first, and the two things that actually know (the shard index, and the
    installer that ran the fetch) are asked before it. `install.Installer._finish`
    holds the other half of that promise, and will never again answer a finished
    download with "start it again".
    """
    where, size, state = _model_location(path, hub, hub_cached, hub_bytes)
    downloading = fetch_in_flight(where) if state == INCOMPLETE else False
    detail = _model_detail(state, path, where, size, hub=hub, hub_bytes=hub_bytes,
                           note=note, fix=fix, downloading=downloading)
    extra = _model_extra(path, where, size, hub=hub, hub_bytes=hub_bytes, fix=fix,
                         downloading=downloading)
    return check(id_, label, state == READY, detail, severity=severity,
                 state=state, **extra)


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------

# There is no `hf_token` row here any more, and the reason is worth writing
# down because the removed check argued the opposite in its own docstring. That
# argument was written while diarization loaded a gated repo: no token meant
# every speaker in the video collapsed into one voice, so a row about the token
# was a row about a real defect. Since v0.4.0 the CC-BY-4.0 weights ship with
# the app (`segments.DIARIZATION_DIR`, restorable in-app), every model installs
# with no credential, and a machine that has never signed in to Hugging Face
# dubs exactly as well as one that has. A row that can only ever say "not set"
# on a perfectly healthy install is advertising a problem nobody has, and
# "optional, and truthfully so" is still a row the user has to read and
# dismiss. The *capability* is untouched: `HF_TOKEN` in `.env` and
# `DUB_DIARIZATION_HUB` still select the gated upstream repo
# (`dubbing/segments.py diarization_sources`). It is simply not a checklist
# item, because nothing about it is unfinished.


def env_path(env_file: Path | None = None) -> Path:
    """The `.env` this process reads absolute, because that is the whole point."""
    return (REPO_ROOT / ".env" if env_file is None else Path(env_file)).resolve()


def env_file_value(path: Path, key: str) -> str | None:
    """What `key` is set to in a `.env` file, or None when it is not in there.

    Read the way python-dotenv reads it, because python-dotenv is what will
    actually apply the file (`dubbing.cli` and `dubbing_app.worker` both call
    `load_dotenv`): comments skipped, surrounding quotes stripped, and the last
    assignment winning over an earlier one. A reader that disagreed with the
    loader would show the user a setting their runs do not have.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    found = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() == key:
            found = value.strip().strip("'\"")
    return found


def gpu_memory_bytes() -> int:
    """Total VRAM on the first NVIDIA GPU, or 0 when there is none to ask.

    Asked of `nvidia-smi` rather than of torch, and that is the module's first
    rule rather than a preference: `GET /api/setup` runs in the server process,
    and importing torch there to read one number would cost half a gigabyte of
    resident memory on every machine, including the Macs that have no CUDA at
    all. The run itself still asks torch (`dubbing.translate._total_vram`);
    this is the cheap version for a screen.
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return 0
    try:
        out = subprocess.run([exe, "--query-gpu=memory.total",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return 0
    first = (out.stdout or "").strip().splitlines()
    try:
        return int(float(first[0].strip())) * 1024**2      # nvidia-smi reports MiB
    except (IndexError, ValueError):
        return 0


def low_vram_env_key() -> str:
    """The variable name the pipeline reads the mode from. Its name, not a copy:
    the route that writes `.env` has to spell it exactly as the run reads it."""
    from dubbing import translate

    return translate.LOW_VRAM_ENV


def low_vram_state(env_file: Path | None = None) -> tuple[bool, str, str, str]:
    """(enabled, source, why, backend) the mode this machine will actually run in.

    The threshold and the decision both come from `dubbing.translate`, never
    restated here, so the screen and the run cannot disagree about what this
    machine is going to do. The backend is passed as a platform fact rather than
    read from `translate._backend()`, whose vLLM probe imports torch.

    `source` is where the answer came from: "env" (a variable in the
    environment, which no file this app writes can override), "env_file" (the
    workspace `.env`, which the toggle writes) or "auto" (nobody said, so the
    memory on this machine decided).
    """
    from dubbing import translate

    path = env_path(env_file)
    env_value = os.environ.get(translate.LOW_VRAM_ENV)
    source = "env" if (env_value or "").strip() else None
    if source is None:
        env_value = env_file_value(path, translate.LOW_VRAM_ENV)
        source = "env_file" if (env_value or "").strip() else None
    forced = translate.parse_low_vram(env_value)
    if forced is None:
        source = "auto"                     # including an unreadable value
    backend = "mlx" if sys.platform == "darwin" else "transformers"
    unified = translate._total_unified() if backend == "mlx" else 0
    vram = 0 if backend == "mlx" else gpu_memory_bytes()
    enabled, why = translate.choose_low_vram(forced, backend, vram, unified)
    return enabled, source, why, backend


def low_vram_check(env_file: Path | None = None) -> dict[str, Any]:
    """Which translator weights this machine will load, and why.

    Always `ok` and always `optional`: this row is a *setting*, not a missing
    thing, and a red tick beside a deliberate choice would send people looking
    for something to install. What it carries instead is the answer and its
    provenance (`enabled` and `source`; see `low_vram_state`).
    """
    from dubbing import translate

    path = env_path(env_file)
    enabled, source, why, backend = low_vram_state(env_file)

    if enabled:
        weights = ("the mxfp4 MLX build, about 6.4 GB" if backend == "mlx"
                   else "4-bit NF4 weights, about 7 GB")
        detail = f"on: the translator loads {weights}, and translates less well"
    else:
        weights = ("the 6-bit MLX build, 9.7 GB" if backend == "mlx"
                   else "bfloat16 weights, about 24 GB")
        detail = f"off: the translator loads {weights}"
    if source == "auto":
        detail += f" ({why})"
    elif source == "env":
        detail += f", set by {translate.LOW_VRAM_ENV} in the environment"
    else:
        detail += f", set in `{path}`"
    return check("low_vram", "Low-VRAM translator", True, detail,
                 severity=OPTIONAL, enabled=enabled, source=source, path=str(path))


def disk_check(outputs: Path) -> dict[str, Any]:
    """Free space where runs are written. Informational: the number matters more
    than the verdict, and "enough" depends on how long the video is."""
    probe = outputs
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError as exc:
        return check("disk", "Free disk space", False, f"unavailable: {exc}",
                     severity=OPTIONAL, path=str(outputs))
    return check("disk", "Free disk space", usage.free >= DISK_WARN_BYTES,
                 f"{human_bytes(usage.free)} free of {human_bytes(usage.total)} at {probe}",
                 severity=OPTIONAL, path=str(outputs), bytes=usage.free)


def model_downloads() -> dict[str, dict[str, Any]]:
    """id → the hub snapshot that fills that check's directory.

    Exactly the models a plain `snapshot_download(repo_id, local_dir=path)`
    satisfies: public repos whose on-disk layout IS the repo layout, into the
    same path the pipeline constant names. Not the Hebrew G2P, which fetches its
    own cache on first use and stays a 400 at `POST /api/setup/install`. Not
    Demucs either, and for a while that meant the same thing; it has a button now
    (`install.DEMUCS_ID`), but through a route that is not a snapshot at all,
    which is why it is still not in this table.

    Diarization is deliberately **not** here, and for the opposite of the old
    reason. It used to be the one *gated* model in the tree, kept out because a
    download button that stops to ask for a Hugging Face account is worse than no
    button. Now its CC-BY-4.0 weights are checked into `third_party/` and ship
    inside the app (`segments.DIARIZATION_DIR`), so there is nothing to download.
    Its row is installable all the same, through a route that is not in this
    table because it fetches nothing: `install.restore_diarization` copies the
    weights back from the payload or the checkout they shipped in and verifies
    them against SHA256SUMS. Keeping it out of *this* table is what guarantees
    the gated repo is never something the app reaches for.

    The hub ids are the pipeline's own where the pipeline has one
    (`translate.HUB_ID`, `tts.TTS_MODELS[...]["hub"]`, `transcript.*_HUB`,
    `hebrew.ADAPTER_HUB`); the ASR fallbacks mirror `tts._ASR_CANDIDATES` and
    the LID id is the documented source of `models/lang-id-voxlingua107-ecapa`
    (docs/MULTILANG_PLAN.md).

    `bytes` is the download size, and it has two jobs that want different
    things. A button label and a progress denominator want a round number and do
    not care about a few megabytes; `model_ready`'s size floor is arithmetic
    against a disk and cares a great deal. For one day these numbers were
    hand-rounded guesses serving both, and the second job caught the first
    lying: `model.lid` was written down as 100 MB, the real snapshot is 86.4 MB,
    and a working model went red. So every figure here is now `dir_size` of a
    complete install on a real machine, rounded **down** to three significant
    figures. Down, always: rounding down costs a button label under a percent of
    accuracy, and it guarantees the declared size can never be the reason a
    finished download reads as unfinished. The hub's own "~95.4 MB" for the LID
    repo disagrees with all of this, which is the last word on why the estimate
    to trust is the one taken from the disk the check will read.

    `cached` says whether a copy in the Hugging Face cache is as good as one in
    `models/`, and it is a per-model fact because the loaders differ. The
    translator, the TTS checkpoint and the source ASRs all fall back to the hub
    id, so a cached copy is what the run will actually open. Language ID, the
    Hebrew LoRA and the two verification ASRs have at least one caller that reads
    the directory or gives up (`transcript.load_lid`, `hebrew.attach_adapter`,
    `transcript.load_target_asr`), so for them only the directory counts.
    Absent means false: a row claiming a model is present has to be right about
    every loader that opens it, not just the forgiving one.
    """
    from dubbing import hebrew, transcript, translate, tts

    # The translator row follows low-VRAM mode, and has to: with the mode on,
    # a row that reported the 6-bit build would call the machine ready for a
    # model the run is not going to open, and its Download button would fetch
    # 9.7 GB the run then ignores before quietly fetching 6.4 GB more.
    tr_path, tr_hub, _ = translate.mlx_model_for(low_vram_state()[0])
    out: dict[str, dict[str, Any]] = {
        # Measured: mxfp4 6,399,849,874 and 6-bit 9,760,955,850 bytes on disk.
        "model.translate": {"hub": tr_hub, "path": tr_path,
                            "bytes": 6_390_000_000 if tr_hub == translate.LOW_VRAM_HUB_ID
                            else 9_760_000_000, "cached": True},
    }
    # Only the default checkpoint is offered. 0.6b exists in tts.TTS_MODELS
    # solely so old manifests that recorded it can re-run; a download button
    # for it would be an invitation to a worse voice, and the user said no.
    tts_default = tts.TTS_MODELS[tts.DEFAULT_TTS_MODEL]
    out[f"model.tts.{tts.DEFAULT_TTS_MODEL}"] = {
        "hub": tts_default["hub"],
        "path": tts.REPO_ROOT / "models" / tts_default["dir"],
        "bytes": 4_540_000_000, "cached": True,          # measured 4,544,231,055
    }
    out.update({
        "model.asr.he": {"hub": transcript.WHISPER_HUB, "path": transcript.WHISPER_MODEL,
                         "bytes": 1_620_000_000, "cached": True},   # 1,621,668,269
        "model.asr.src": {"hub": transcript.SRC_ASR_HUB, "path": transcript.SRC_ASR_MODEL,
                          "bytes": 1_620_000_000, "cached": True},  # 1,621,670,683
        "model.asr.en": {"hub": "Systran/faster-whisper-base.en",  # tts._ASR_CANDIDATES
                         "path": transcript.EN_ASR_MODEL, "bytes": 147_000_000},  # 147,772,938
        "model.asr.tgt": {"hub": "Systran/faster-whisper-base",    # tts._ASR_CANDIDATES_MULTI
                          "path": transcript.TARGET_ASR_MODEL,
                          "bytes": 147_000_000},                    # 147,887,035
        # 86,400,192 measured, against the 100 MB somebody rounded to and the
        # ~95.4 MB the hub quotes. This row is why the rest of them were checked.
        "model.lid": {"hub": "speechbrain/lang-id-voxlingua107-ecapa",
                      "path": transcript.LID_MODEL, "bytes": 86_400_000},
        "model.tts.he": {"hub": hebrew.ADAPTER_HUB, "path": hebrew.ADAPTER_DIR,
                         "bytes": 247_000_000},                     # 247,523,723
    })
    return out


def model_checks() -> list[dict[str, Any]]:
    """Every model the pipeline opens, read from its own constants: each
    directory, and last the one that is a cache rather than a directory
    (`demucs_check`).

    Blocking means "the default run cannot work without it": the translator, the
    default TTS checkpoint and the English ASR that verifies each clip (without
    it verification silently stops happening, which AGENTS.md forbids).

    One model degrades rather than blocks: language ID. Its absence does not stop
    a run, it stops the run from *noticing* a third language, and a third language
    nobody noticed is kept as recorded with no subtitle. The rest are per-language
    pair a Korean checkpoint has nothing to say about a Hebrew→English run so
    they are optional and stay out of the way.
    """
    from dubbing import hebrew, segments, transcript, tts

    downloads = model_downloads()

    def m(id_: str, label: str, path: Path, **kw: Any) -> dict[str, Any]:
        # Hub repo and size come from the one table, so a row and the install
        # slot can never disagree about where a model comes from or how big it is.
        d = downloads.get(id_)
        if d:
            kw.setdefault("hub", d["hub"])
            kw.setdefault("hub_bytes", d["bytes"])
            kw.setdefault("hub_cached", bool(d.get("cached")))
        return model(id_, label, path, **kw)

    tr = downloads["model.translate"]
    out = [
        m("model.translate", "Translation model (Gemma 4 12B)", tr["path"],
          note=f"downloads from {tr['hub']} on first use"),
    ]
    # One row: the default checkpoint. See model_downloads for why 0.6b is
    # deliberately absent here even though the pipeline can still re-run it.
    tts_spec = tts.TTS_MODELS[tts.DEFAULT_TTS_MODEL]
    out.append(m(f"model.tts.{tts.DEFAULT_TTS_MODEL}",
                 f"TTS checkpoint {tts.DEFAULT_TTS_MODEL}",
                 tts.REPO_ROOT / "models" / tts_spec["dir"],
                 severity=BLOCKING,
                 note=f"downloads from {tts_spec['hub']} on first use"))
    out += [
        m("model.asr.he", "Source ASR: Hebrew (ivrit-ai)", transcript.WHISPER_MODEL,
          severity=OPTIONAL, note="only for Hebrew sources without captions"),
        m("model.asr.src", "Source ASR: multilingual", transcript.SRC_ASR_MODEL,
          severity=OPTIONAL, note="only for non-Hebrew sources without captions"),
        m("model.asr.en", "Target ASR: English (clip verification)",
          transcript.EN_ASR_MODEL,
          note="without it generated clips are never verified"),
        m("model.asr.tgt", "Target ASR: multilingual", transcript.TARGET_ASR_MODEL,
          severity=OPTIONAL, note="only for non-English targets"),
        m("model.lid", "Language ID (VoxLingua107)", transcript.LID_MODEL,
          severity=DEGRADES, note="without it foreign-speech detection is skipped"),
        # Green on every install, because the weights are in the payload: 31 MB
        # of CC-BY-4.0 pyannote checked into `third_party/` and copied into the
        # workspace on first launch. The row is here to *say* so — a screen that
        # lists nine models and silently omits the tenth reads as an omission.
        #
        # It has no Download button, and never will: the upstream repo is the one
        # gated thing in the tree. What it has instead is a *Restore* one. When
        # the copy is gone the app can put it back from the payload it shipped in
        # or from the checkout it was cloned into, with no network and no account
        # (`install.diarization_source`), and `fix` carries the same line for a
        # user who would rather type it.
        #
        # `degrades`, not `optional`, and only since that button existed. The row
        # is graded for the state it is in when it is red, and red here means
        # every character in the video is dubbed in one voice: the run works and
        # is worse, which is what `degrades` means. It sat in `optional` while
        # the only fix was a Hugging Face account, because a grade that says
        # "this matters" beside a fix the app could not offer is just noise. The
        # fix is now one button, so the grade can tell the truth, and "install
        # everything" picks it up along with the rest.
        m("model.diarization", "Speaker diarization (pyannote community-1)",
          segments.DIARIZATION_DIR, severity=DEGRADES,
          **diarization_repair()),
        # Hebrew is a dub TARGET only with both of these. Optional every other
        # target runs without them but a Hebrew run is refused up front when
        # either is missing, so the report is where a user finds out first.
        m("model.tts.he", "Hebrew TTS adapter (Qwen3-TTS LoRA)", hebrew.ADAPTER_DIR,
          severity=OPTIONAL,
          note=f"only for Hebrew targets: {hebrew.ADAPTER_DOWNLOAD}"),
        g2p_check(),
        # Last, and in this list rather than appended by `report` alone, because
        # it has a button now (`install.DEMUCS_ID`) and `probe` re-checks a row
        # by looking for it here. A row the app can install and cannot re-probe
        # would be a button whose result never reaches the screen.
        demucs_check(),
    ]
    return out


def diarization_repair() -> dict[str, str]:
    """The `note` and `fix` for the diarization row: what its absence costs, and
    the line that ends it.

    Two answers, because there are two situations and only one of them has a
    command. A machine that still carries the copy the weights shipped in gets
    that copy's restore line, which is also what the button runs. A machine that
    carries none of them (a stripped install, a workspace pruned by hand) gets
    the honest alternative instead of a command that would fail: reinstall, or
    name a mirror. What it never gets is "log in to Hugging Face": the upstream
    repo is gated, and this row exists precisely so nobody is asked for that.
    """
    from dubbing import segments

    from .install import diarization_command

    note = "ships with the app; without it every speaker is dubbed in one voice"
    fix = diarization_command()
    if fix:
        return {"note": note, "fix": fix}
    return {"note": note + ": no copy of it is left on this machine, so reinstall "
                          "the app, or point `"
                          f"{segments.DIARIZATION_HUB_ENV}` at a directory or a "
                          f"mirror of {segments.DIARIZATION_MODEL} you can reach"}


def g2p_check() -> dict[str, Any]:
    """The Hebrew grapheme→IPA model. Unlike the others this is a *package* first:
    `renikud-plus` fetches its own weights on first use, so what decides the verdict
    is whether it is importable, and a local `models/RenikudPlus` is reported when
    it exists but is never required."""
    from dubbing import hebrew

    local = hebrew.G2P_FILE.is_file()
    size = dir_size(hebrew.G2P_DIR) if local else 0
    if not hebrew.g2p_ready():
        detail = (f"{hebrew.G2P_PACKAGE} is not installed: run `uv sync`; "
                  "without it Hebrew is unavailable as a target")
    elif local:
        detail = f"{human_bytes(size)} in {hebrew.G2P_DIR}"
    else:
        detail = (f"{hebrew.G2P_PACKAGE} installed; weights download from "
                  f"{hebrew.G2P_HUB} on first use")
    return check("model.g2p.he", "Hebrew G2P (ReNikud Plus)", hebrew.g2p_ready(),
                 detail, severity=OPTIONAL, path=str(hebrew.G2P_DIR), bytes=size)


def default_cache_home() -> Path:
    """The base directory the two caches below default under: `XDG_CACHE_HOME`
    when it is set, otherwise `~/.cache`.

    Both `huggingface_hub` and `torch.hub` read that variable, on every
    platform, and neither of them was being asked. That is a Linux bug and only
    a Linux one: nothing sets `XDG_CACHE_HOME` on a stock Mac or Windows, so
    the two rows agreed with the libraries there and disagreed on exactly the
    distributions that set it (and on any machine whose user set it to move
    tens of gigabytes off a small root volume). The failure was the quiet kind
    this file keeps warning about: a model genuinely on disk, reported missing,
    with a Download button that would fetch it into the directory the check was
    already refusing to look in.
    """
    xdg = (os.environ.get("XDG_CACHE_HOME") or "").strip()
    return Path(xdg) if xdg else Path.home() / ".cache"


def hf_hub_cache() -> Path:
    """Where huggingface_hub keeps models, resolved the way the library does:
    `HF_HUB_CACHE` wins, then `HF_HOME` (cache lives under its `hub/`), then
    `XDG_CACHE_HOME`, then `~/.cache`. Restating the library's rule here is the
    price of rule one: importing huggingface_hub just to ask a path would pull
    it into every `GET /api/setup`."""
    hub_cache = (os.environ.get("HF_HUB_CACHE") or "").strip()
    if hub_cache:
        return Path(hub_cache)
    hf_home = (os.environ.get("HF_HOME") or "").strip()
    if hf_home:
        return Path(hf_home) / "hub"
    return default_cache_home() / "huggingface" / "hub"


def hf_cache_repo(hub: str) -> Path | None:
    """The Hugging Face cache directory holding `hub`, or None when it holds no
    bytes of it. The second of the two places `model()` looks, and the one a
    hub-id load actually fills.

    "Holds it" is `blobs/` with something in it, not merely the directory
    existing: an interrupted or metadata-only fetch leaves the `models--*` shell
    behind, and calling that present would be the opposite failure to the one
    this exists to fix.
    """
    repo = hf_hub_cache() / f"models--{hub.replace('/', '--')}"
    blobs = repo / "blobs"
    if repo.is_dir() and blobs.is_dir() and any(blobs.iterdir()):
        return repo
    return None


def _demucs_cache() -> Path | None:
    """The cache holding the Demucs weights, or None when neither has them.

    Two caches, because demucs changed homes: 3.x keeps `.th` weights under the
    torch hub cache, 4.x fetches from the Hugging Face Hub into the HF cache as
    `models--adefossez--*` (with the payload under `blobs/`).
    """
    # `TORCH_HOME`, else `XDG_CACHE_HOME/torch`, else `~/.cache/torch`, which is
    # `torch.hub._get_torch_home()` restated for the same reason `hf_hub_cache`
    # restates its library's rule: asking torch would import torch.
    torch_cache = Path(os.environ.get("TORCH_HOME")
                       or (default_cache_home() / "torch")) / "hub"
    if torch_cache.is_dir() and any(torch_cache.rglob("*.th")):
        return torch_cache
    hf_cache = hf_hub_cache()
    if hf_cache.is_dir():
        for repo in sorted(hf_cache.glob("models--adefossez--*")):
            blobs = repo / "blobs"
            if blobs.is_dir() and any(blobs.iterdir()):
                return repo
    return None


def demucs_check() -> dict[str, Any]:
    """Optional by contract: Demucs fetches `htdemucs_ft` the first time `stems`
    runs, so absence is a slow first run, not a broken install and not a worse
    dub.

    Optional, and installable all the same, which is not the contradiction it
    reads as. "It downloads itself" was taken to mean "there is nothing to do
    here", and what that produced was the one row on the whole screen with no
    gesture at all: no button, no place in the queue, and a detail line whose
    only offer was to start a dub and sit through a silent mid-run download. A
    fetch the user chose, on the setup screen, with a progress bar, is strictly
    better than the same fetch in the middle of a job. `install.demucs_argv` runs
    the same `get_model` the stems stage would, into the same cache, so the
    button changes when the bytes arrive and not what arrives.

    Two caches, because demucs changed homes: 3.x keeps `.th` weights under the
    torch hub cache, 4.x fetches from the Hugging Face Hub into the HF cache as
    `models--adefossez--*` (with the payload under `blobs/`). Probing only the
    old one made this a row that could never pass on a working install."""
    from dubbing import stems

    from .install import DEMUCS_ID, route

    found = _demucs_cache()
    present = found is not None
    size = dir_size(found) if found is not None else 0
    if present:
        detail = f"{stems.MODEL} cache: {human_bytes(size)} in {found}"
    else:
        # Both halves, because the row used to offer only the first one and that
        # is the sentence that ends in "so start a dub and wait".
        detail = (f"{stems.MODEL} not downloaded yet: it is fetched on the first "
                  f"stems run. The Install button installs it {route(DEMUCS_ID)}")
    # The missing row points at the HF cache: that is where a 4.x download will
    # actually land, and the torch hub path would send the user somewhere the
    # weights will never appear.
    return check("model.demucs", "Demucs stem separation", present, detail,
                 severity=OPTIONAL, path=str(found or hf_hub_cache()), bytes=size)


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

# Where a blocking check's absence actually stops the run. "Runs will fail" is
# true and useless; "runs will fail at translate" is the sentence that tells a
# user whether to fix it now or start the fetch and fix it while it downloads.
BLOCKING_STAGE: dict[str, str] = {
    "ffmpeg": "fetch",          # the first stage that shells out; they all do
    # sox has no entry: its row is OPTIONAL now (see TOOLS) and `stage` is only
    # ever attached to blocking rows — a stage on it would be the false urgency
    # the severity downgrade removed.
    "model.translate": "translate",
    "model.asr.en": "tts",      # clip verification lives inside the tts stage
}


def blocking_stage(id_: str) -> str | None:
    """The stage a missing blocking check kills, or None when there isn't one.

    `None` is a real answer and not a gap to be filled, and the row that taught
    that lesson has since been regraded rather than answered. `uv` was blocking
    with no stage, on the reasoning that nothing about this project is installed
    or updated without it while a server that is already running never shells
    out to it (`runner.SubprocessRunner` uses `sys.executable`). The second half
    of that sentence is the whole argument against the first: a check that kills
    no stage is not blocking, so uv is `optional` now and the question does not
    arise for it. What survives is the shape. A client that shows the stage must
    still be prepared not to have one, because the alternative is a future
    blocking row inventing a stage it does not really break.
    """
    if id_ in BLOCKING_STAGE:
        return BLOCKING_STAGE[id_]
    return "tts" if id_.startswith("model.tts.") else None


def report(outputs: Path) -> dict[str, Any]:
    # Imported here, not at module scope: `install` imports this module back for
    # its re-probe, and the one-directional import is what keeps that honest.
    from .install import installable

    downloads = model_downloads()
    checks: list[dict[str, Any]] = [tool(id_, *spec) for id_, spec in TOOLS.items()]
    checks += model_checks()             # the Demucs cache is the last of those
    checks.append(disk_check(Path(outputs)))
    # Last, beside the other row about this machine rather than about a file:
    # it is the only one that is a choice, and it reads as a footnote to the
    # model list above it, which is what it is.
    checks.append(low_vram_check())
    for c in checks:
        c["installable"] = installable(c["id"]) or c["id"] in downloads
        # Only on the rows where it means something: a `stage` on an optional
        # row would read as "this is where it will bite you", which is exactly
        # the false urgency `severity` exists to stop.
        if c["severity"] == BLOCKING:
            c["stage"] = blocking_stage(c["id"])
    ok = all(c["ok"] for c in checks if c["required"])
    return {"ok": ok, "checks": checks}


def install_plan(report_: dict[str, Any]) -> list[dict[str, Any]]:
    """What "install everything" actually installs, in the order to install it.

    Read off a report rather than rebuilt from the tables, so the button can
    never queue a row the screen is not showing red. Three filters, and each one
    is a refusal:

    * **Only what is missing.** A green row is not re-fetched; a torn-off
      download is (`snapshot_download` resumes from the partial files, so a row
      that is half there costs only the half that is not).
    * **Only what the app can install by itself** — `installable`, the same flag
      that puts a button on the row. Everything else is out by construction, and
      that is also where the credentials question answers itself: no check here
      asks for a token at all, and no hub id
      in `model_downloads()` is gated the one gated repo in the tree,
      `segments.DIARIZATION_MODEL`, is not in that table at all because its
      weights ship with the app. Every repo this can queue is public, and the
      one queued item that is not a repo at all is the diarization restore,
      which reads a copy already on this machine. So the queue never stops half
      way to ask the user for something.
    * **In the order the grades mean.** Blocking first, then `degrades`, then
      `optional`: the run has to work before it has to be good, and on a slow
      connection that order is the difference between a usable machine in twenty
      minutes and one in two hours.

    The third filter used to be "nothing graded `optional`", and the argument for
    it was that a Korean checkpoint has nothing to say about a Hebrew→English run
    and a button that says "everything" must not quietly mean "and 40 GB you will
    never open". That is a good rule about the multilingual checkpoints and the
    wrong one about the list it actually excluded. `optional` grades "does the
    run technically finish without it", which puts the Hebrew ASR and the Hebrew
    TTS adapter, the two models most specific to the job on a Hebrew machine, in
    the same bucket as a language nobody will open, and then one flag dropped
    both. What that left was a button labelled "install all" that, on a machine
    where only optional rows were missing, installed nothing and said nothing,
    and five red rows whose only remaining instruction was a shell command in a
    desktop app whose whole point is that it does not need one.

    So the button means what it says, and the 40 GB objection is answered where
    it was always going to have to be answered: the button carries the price. The
    sizes are on the rows already, the plan adds them up, and a user who does not
    want the extras still has each row's own button and the queue's Cancel.

    Each row carries the label the header says out loud and the download's size
    (0 for a tool, since a `brew` is seconds and has no denominator), which is
    what lets the button price itself before it is pressed.
    """
    rank = {BLOCKING: 0, DEGRADES: 1, OPTIONAL: 2}
    rows = [c for c in report_.get("checks", ())
            if not c.get("ok") and c.get("installable") and c.get("severity") in rank]
    # Stable, so within a grade the plan keeps the report's order — which is the
    # order the screen lists them in, and the order the user is reading.
    rows.sort(key=lambda c: rank[c["severity"]])
    return [{"id": c["id"], "label": c["label"],
             "bytes": int(c.get("download_bytes") or 0)} for c in rows]


__all__ = ["report", "probe", "install_plan", "git_commit", "human_bytes", "dir_size", "env_path",
           "env_file_value", "find_uv", "uv_exe", "uv_fallbacks", "uv_home",
           "uv_on_path", "UV_FALLBACKS", "UV_PATH_ENV", "default_cache_home",
           "gpu_memory_bytes", "hf_hub_cache",
           "hf_cache_repo", "diarization_repair", "index_shards", "model_ready",
           "fetch_in_flight", "record_install", "install_recorded", "INSTALL_RECEIPT",
           "low_vram_check", "low_vram_env_key", "low_vram_state", "model_downloads",
           "blocking_stage", "TOOLS", "BLOCKING", "DEGRADES", "OPTIONAL", "SEVERITIES",
           "MISSING", "INCOMPLETE", "READY", "STATES"]
