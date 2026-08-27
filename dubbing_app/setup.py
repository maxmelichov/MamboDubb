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
                             "installable": bool, "path"?, "found_at"?, "bytes"?,
                             "hub"?, "download_bytes"?, "fix"?}, ...]}

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
undifferentiated "informational" which put a missing HF token (every speaker
in the video collapsed into one) on the same row as a Korean TTS checkpoint a
Hebrew→English run will never open. Three grades, and the third is what makes
the first two mean anything:

* ``blocking`` the run **fails** without it. The command-line tools, the
  translator, the default TTS checkpoint and the English verifier. Exactly the
  ``required`` set, and `required` is derived from this so the two cannot drift.
* ``degrades`` the run **works and is worse**. No language-ID model means
  foreign speech is never detected; no diarization weights mean every character
  in the video is dubbed in one voice. Nothing here stops a run, and nothing here
  is nothing.
* ``optional`` irrelevant until you ask for it: the per-language-pair models,
  the self-downloading caches, free disk and now the HF token, which used to
  sit in ``degrades`` because diarization was gated behind it. It is not any
  more the CC-BY-4.0 weights ship inside the app
  (``segments.DIARIZATION_DIR``) so the row states a fact instead of asking
  for a credential, and a machine that has never signed in to Hugging Face
  reads as fully ready.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# The three grades a check can be. See the module docstring; `required` is
# derived from this and never passed in, so no caller can declare a check
# blocking and not required (or the reverse) by hand.
BLOCKING, DEGRADES, OPTIONAL = "blocking", "degrades", "optional"
SEVERITIES = (BLOCKING, DEGRADES, OPTIONAL)

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
          **extra: Any) -> dict[str, Any]:
    """One row. `required` is `severity == "blocking"`, computed here and nowhere
    else the old contract, still on the wire, now with one source of truth."""
    if severity not in SEVERITIES:
        raise ValueError(f"unknown severity {severity!r}")
    return {"id": id_, "label": label, "ok": bool(ok), "detail": detail,
            "severity": severity, "required": severity == BLOCKING, **extra}


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def dir_size(path: Path) -> int:
    """Bytes on disk under `path`. Symlinks are not followed and unreadable
    entries are skipped a size is a nicety, never a reason to 500."""
    total = 0
    try:
        for entry in path.rglob("*"):
            try:
                if entry.is_file() and not entry.is_symlink():
                    total += entry.stat().st_size
            except OSError:
                continue
    except OSError:
        return total
    return total


# The command-line tools, in one table so a single one can be re-checked after
# an install without re-running (and re-`stat`ing) the whole model report. A row
# is (label, exe, why) with an optional fourth element, the severity — blocking
# when absent, because "a run dies partway through without it" is this file's
# definition and it is true of ffmpeg and uv.
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
    # Not installable from here `install.MANAGERS` says the same thing about
    # Homebrew: the tool that installs the dependencies cannot be one of them.
    # And it names no stage (see `BLOCKING_STAGE`): a server that is already
    # running has its environment, and `runner.SubprocessRunner` spawns the job
    # child with `sys.executable`, not with `uv`. What breaks without it is
    # repairing or updating that environment which is why it blocks and why
    # claiming it kills a particular stage would be the same dishonesty this
    # grading exists to remove.
    "uv": ("uv", "uv", "the pipeline's dependencies are installed and pinned with it "
                       "(`uv sync`), and every command in AGENTS.md starts with it. "
                       "Get it from https://docs.astral.sh/uv/"),
}


# `uv` discovery, kept byte-for-byte in step with `find_uv()` in the shell's
# `workspace.rs`: same env override, same literal paths, same order. Two
# different rules for the same tool is exactly the bug this exists to close —
# the Rust side found uv, started this server with it, and the server then
# reported the tool that launched it as missing and required.
UV_PATH_ENV = "DUBSTUDIO_UV_PATH"
UV_FALLBACKS = ("/opt/homebrew/bin/uv", "/usr/local/bin/uv")


def find_uv() -> str | None:
    """Where `uv` actually is, not just where PATH says. A Finder-launched .app
    inherits almost none of the user's shell PATH on macOS, and uv's own
    installer puts the binary in `~/.local/bin` — which is why `shutil.which`
    alone answers "missing" on a machine where uv is running the server."""
    override = (os.environ.get(UV_PATH_ENV) or "").strip()
    if override and Path(override).is_file():
        return override
    for candidate in UV_FALLBACKS:
        if Path(candidate).is_file():
            return candidate
    found = shutil.which("uv")
    if found:
        return found
    local = Path.home() / ".local" / "bin" / "uv"
    return str(local) if local.is_file() else None


def tool(id_: str, label: str, exe: str, why: str,
         severity: str = BLOCKING) -> dict[str, Any]:
    """One tool row. `tools.resolve_tool` is the whole probe — the same lookup
    every pipeline call site uses: env override, then the workspace `tools/bin`
    (where the brewless-Mac static build lands), then PATH with PATHEXT, so
    `ffmpeg.exe` on Windows answers to the same lookup as `ffmpeg` elsewhere.
    The one exception is `uv`, which gets `find_uv()`'s fuller chain (above).

    A missing row carries **this platform's** install command, because that is
    the only actionable half of "not on PATH" and the row is where a user with
    no button looks for it (`dubbing.tools`; the Setup screen's button exists
    only where that command can run unattended). Where there *is* a button, the
    row also says which route pressing it takes — `via Homebrew`, or the static
    build into the workspace — because a button that might run a package
    manager or might download a binary is a button nobody can trust."""
    from dubbing import tools as tool_recipes

    found = find_uv() if exe == "uv" else tool_recipes.resolve_tool(exe)
    detail = found or f"{exe} not on PATH: {why}"
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


def model(id_: str, label: str, path: Path, *, severity: str = BLOCKING,
          note: str = "", hub: str = "", hub_bytes: int = 0,
          fix: str = "") -> dict[str, Any]:
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

    A missing row's whole job is to be actionable: when the hub repo is known,
    the detail carries the exact download command (backticked so the UI sets it
    as code) and the row carries `hub` and `download_bytes`, which is what lets
    the Setup screen label its button "Download (~9.7 GB)" instead of just
    "Download". "Missing" without the command or the size is a scavenger hunt.
    `fix` is the same promise for a model with no hub to fetch from: the command
    that repairs it, backticked, and on the row as data so a UI can copy it.
    """
    present = path.is_dir() and any(path.iterdir()) if path.is_dir() else False
    where = path
    if not present and hub:
        cached = hf_cache_repo(hub)
        if cached is not None:
            present, where = True, cached
    size = dir_size(where) if present else 0
    if present:
        origin = f"{path}" if where == path else f"the Hugging Face cache ({where})"
        detail = f"{human_bytes(size)} in {origin}"
    else:
        detail = f"missing: {path}" + (f" ({note})" if note else "")
        if hub:
            approx = f" (~{human_bytes(hub_bytes)})" if hub_bytes else ""
            detail += f". Fetch it{approx}: `uv run hf download {hub} --local-dir {path}`"
        elif fix:
            detail += f". Restore it: `{fix}`"
    # `path` stays the pipeline's own constant whichever place answered: it is
    # where this model belongs, and a row that renamed itself after the cache
    # would stop being checkable against `dubbing`'s constants. Where it was
    # actually found, when that is somewhere else, is a second field.
    extra: dict[str, Any] = {"path": str(path), "bytes": size}
    if where != path:
        extra["found_at"] = str(where)
    if hub:
        extra["hub"] = hub
        extra["download_bytes"] = hub_bytes
    if fix:
        extra["fix"] = fix
    return check(id_, label, present, detail, severity=severity, **extra)


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------

def hf_token_check(env_file: Path | None = None) -> dict[str, Any]:
    """Presence only. The value is a credential and never leaves this process —
    the report says "set" or "not set" and nothing that could reconstruct it.

    **Optional, and that is the whole point of this row now.** It used to be
    `degrades`, and truthfully so: diarization loaded a gated repo, no token
    meant every speaker in the video collapsed into one voice, and the only fix
    was a Hugging Face account. A shippable app cannot ask for that, so it does
    not: the CC-BY-4.0 weights ship with the app (`segments.DIARIZATION_DIR`)
    and a machine with no token diarizes exactly as well as one with. The row
    stays because a token is still *usable*
    it selects the gated upstream repo via `DUB_DIARIZATION_HUB`, and it is
    what other gated models would need but it may never again be the reason a
    fresh install is not green.

    The unset row still names the **absolute** `.env` path, backticked so the UI
    sets it as code and offers to copy it: on a machine with three checkouts,
    only this process knows which `.env` it will actually read.
    """
    path = env_path(env_file)
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if (os.environ.get(var) or "").strip():
            return check("hf_token", "Hugging Face token", True, f"set via {var}",
                         severity=OPTIONAL, source="env", path=str(path))
    if _env_file_has_token(path):
        return check("hf_token", "Hugging Face token", True, f"set in `{path}`",
                     severity=OPTIONAL, source="env_file", path=str(path))
    return check("hf_token", "Hugging Face token", False,
                 "not set: nothing needs one. The diarization weights ship with the "
                 "app, so speakers are told apart without an account. Only for fetching "
                 f"gated upstream models instead; it would go in `{path}`",
                 severity=OPTIONAL, source=None, path=str(path))


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


def _env_file_has_token(path: Path) -> bool:
    return any((env_file_value(path, var) or "").strip()
               for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"))


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
    same path the pipeline constant names. Not Demucs or the Hebrew G2P (they
    fetch their own caches on first use) those stay 400 at
    `POST /api/setup/install`.

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
    (docs/MULTILANG_PLAN.md). `bytes` is the download size measured from real
    installs approximate on purpose, good enough for a button label and a
    progress denominator, never for accounting.
    """
    from dubbing import hebrew, transcript, translate, tts

    # The translator row follows low-VRAM mode, and has to: with the mode on,
    # a row that reported the 6-bit build would call the machine ready for a
    # model the run is not going to open, and its Download button would fetch
    # 9.7 GB the run then ignores before quietly fetching 6.4 GB more.
    tr_path, tr_hub, _ = translate.mlx_model_for(low_vram_state()[0])
    out: dict[str, dict[str, Any]] = {
        "model.translate": {"hub": tr_hub, "path": tr_path,
                            "bytes": 6_400_000_000 if tr_hub == translate.LOW_VRAM_HUB_ID
                            else 9_700_000_000},
    }
    # Only the default checkpoint is offered. 0.6b exists in tts.TTS_MODELS
    # solely so old manifests that recorded it can re-run; a download button
    # for it would be an invitation to a worse voice, and the user said no.
    tts_default = tts.TTS_MODELS[tts.DEFAULT_TTS_MODEL]
    out[f"model.tts.{tts.DEFAULT_TTS_MODEL}"] = {
        "hub": tts_default["hub"],
        "path": tts.REPO_ROOT / "models" / tts_default["dir"],
        "bytes": 4_500_000_000,
    }
    out.update({
        "model.asr.he": {"hub": transcript.WHISPER_HUB, "path": transcript.WHISPER_MODEL,
                         "bytes": 1_600_000_000},
        "model.asr.src": {"hub": transcript.SRC_ASR_HUB, "path": transcript.SRC_ASR_MODEL,
                          "bytes": 1_600_000_000},
        "model.asr.en": {"hub": "Systran/faster-whisper-base.en",  # tts._ASR_CANDIDATES
                         "path": transcript.EN_ASR_MODEL, "bytes": 150_000_000},
        "model.asr.tgt": {"hub": "Systran/faster-whisper-base",    # tts._ASR_CANDIDATES_MULTI
                          "path": transcript.TARGET_ASR_MODEL, "bytes": 150_000_000},
        "model.lid": {"hub": "speechbrain/lang-id-voxlingua107-ecapa",
                      "path": transcript.LID_MODEL, "bytes": 100_000_000},
        "model.tts.he": {"hub": hebrew.ADAPTER_HUB, "path": hebrew.ADAPTER_DIR,
                         "bytes": 250_000_000},
    })
    return out


def model_checks() -> list[dict[str, Any]]:
    """Every model directory the pipeline opens, read from its own constants.

    Blocking means "the default run cannot work without it": the translator, the
    default TTS checkpoint and the English ASR that verifies each clip (without
    it verification silently stops happening, which AGENTS.md forbids).

    One model degrades rather than blocks: language ID. Its absence does not stop
    a run, it stops the run from *noticing* a third language, and a third language
    nobody noticed is kept as recorded with no subtitle. The rest are per-language
    pair a Korean checkpoint has nothing to say about a Hebrew→English run so
    they are optional and stay out of the way.
    """
    from dubbing import hebrew, segments, transcript, translate, tts

    downloads = model_downloads()

    def m(id_: str, label: str, path: Path, **kw: Any) -> dict[str, Any]:
        # Hub repo and size come from the one table, so a row and the install
        # slot can never disagree about where a model comes from or how big it is.
        d = downloads.get(id_)
        if d:
            kw.setdefault("hub", d["hub"])
            kw.setdefault("hub_bytes", d["bytes"])
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


def hf_hub_cache() -> Path:
    """Where huggingface_hub keeps models, resolved the way the library does:
    `HF_HUB_CACHE` wins, then `HF_HOME` (cache lives under its `hub/`), then
    the default. Restating the library's rule here is the price of rule one —
    importing huggingface_hub just to ask a path would pull it into every
    `GET /api/setup`."""
    hub_cache = (os.environ.get("HF_HUB_CACHE") or "").strip()
    if hub_cache:
        return Path(hub_cache)
    hf_home = (os.environ.get("HF_HOME") or "").strip()
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


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


def demucs_check() -> dict[str, Any]:
    """Optional by contract: Demucs fetches `htdemucs_ft` the first time `stems`
    runs, so absence is a slow first run, not a broken install and not a worse
    dub.

    Two caches, because demucs changed homes: 3.x keeps `.th` weights under the
    torch hub cache, 4.x fetches from the Hugging Face Hub into the HF cache as
    `models--adefossez--*` (with the payload under `blobs/`). Probing only the
    old one made this a row that could never pass on a working install."""
    from dubbing import stems

    torch_cache = Path(os.environ.get("TORCH_HOME") or (Path.home() / ".cache" / "torch")) / "hub"
    found: Path | None = None
    if torch_cache.is_dir() and any(torch_cache.rglob("*.th")):
        found = torch_cache
    else:
        hf_cache = hf_hub_cache()
        if hf_cache.is_dir():
            for repo in sorted(hf_cache.glob("models--adefossez--*")):
                blobs = repo / "blobs"
                if blobs.is_dir() and any(blobs.iterdir()):
                    found = repo
                    break
    present = found is not None
    size = dir_size(found) if found is not None else 0
    detail = (f"{stems.MODEL} cache: {human_bytes(size)} in {found}" if present
              else f"{stems.MODEL} not downloaded yet: fetched on the first stems run")
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

    `None` is a real answer, not a gap to be filled: `uv` blocks nothing about
    this project is installed or updated without it but a server that is
    already running does not shell out to it (`runner.SubprocessRunner` uses
    `sys.executable`), so naming a stage it kills would be a guess dressed as a
    fact. A client that shows the stage must be prepared not to have one.
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
    checks.append(hf_token_check())
    checks += model_checks()
    checks.append(demucs_check())
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
      that is also where the credentials question answers itself: the one check
      that mentions a token is `hf_token`, which installs nothing, and no hub id
      in `model_downloads()` is gated the one gated repo in the tree,
      `segments.DIARIZATION_MODEL`, is not in that table at all because its
      weights ship with the app. Every repo this can queue is public, and the
      one queued item that is not a repo at all is the diarization restore,
      which reads a copy already on this machine. So the queue never stops half
      way to ask the user for something.
    * **Nothing graded `optional`.** A Korean checkpoint has nothing to say
      about a Hebrew→English run, and a button that says "everything" must not
      quietly mean "and 40 GB you will never open". Blocking first, then
      `degrades`: the run has to work before it has to be good, and on a slow
      connection the order is the difference between a usable machine in twenty
      minutes and one in two hours.

    Each row carries the label the header says out loud and the download's size
    (0 for a tool — a `brew` is seconds and has no denominator), which is what
    lets the button price itself before it is pressed.
    """
    rank = {BLOCKING: 0, DEGRADES: 1}
    rows = [c for c in report_.get("checks", ())
            if not c.get("ok") and c.get("installable") and c.get("severity") in rank]
    # Stable, so within a grade the plan keeps the report's order — which is the
    # order the screen lists them in, and the order the user is reading.
    rows.sort(key=lambda c: rank[c["severity"]])
    return [{"id": c["id"], "label": c["label"],
             "bytes": int(c.get("download_bytes") or 0)} for c in rows]


__all__ = ["report", "probe", "install_plan", "git_commit", "human_bytes", "dir_size", "env_path",
           "env_file_value", "find_uv", "gpu_memory_bytes", "hf_hub_cache",
           "hf_cache_repo", "diarization_repair",
           "low_vram_check", "low_vram_env_key", "low_vram_state", "model_downloads",
           "blocking_stage", "TOOLS", "BLOCKING", "DEGRADES", "OPTIONAL", "SEVERITIES"]
