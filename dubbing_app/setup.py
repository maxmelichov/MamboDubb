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
                             "installable": bool, "path"?, "bytes"?}, ...]}

`installable` is the server's answer to "can the app fix this for me?" true for
exactly the ids `dubbing_app.install` has an argv for. The UI needs it as a flag
rather than a list of its own, or the two sides drift and a button appears on a
row whose `POST /api/setup/install` is a 400.

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
* ``degrades`` the run **works and is worse**. No HF token means diarization
  falls back to a single speaker; no language-ID model means foreign speech is
  never detected. Nothing here stops a run, and nothing here is nothing.
* ``optional`` irrelevant until you ask for it: the per-language-pair models,
  the self-downloading caches, free disk.
"""

from __future__ import annotations

import os
import shutil
import subprocess
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
# an install without re-running (and re-`stat`ing) the whole model report. All
# three are blocking: without any one of them a run dies partway through, which
# is the definition this file uses.
TOOLS: dict[str, tuple[str, str, str]] = {
    "ffmpeg": ("ffmpeg", "ffmpeg", "every stage shells out to it for audio and video"),
    "sox": ("SoX", "sox", "Qwen3-TTS text normalization needs it"),
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


def tool(id_: str, label: str, exe: str, why: str, *,
         severity: str = BLOCKING) -> dict[str, Any]:
    """One tool row. `shutil.which` is the whole probe — it honours PATHEXT, so
    `ffmpeg.exe` on Windows answers to the same lookup as `ffmpeg` elsewhere.

    A missing row carries **this platform's** install command, because that is
    the only actionable half of "not on PATH" and the row is where a user with
    no button looks for it (`dubbing.tools`; the Setup screen's button exists
    only where that command can run unattended)."""
    from dubbing import tools as tool_recipes

    found = shutil.which(exe)
    detail = found or f"{exe} not on PATH {why}"
    if not found:
        command = tool_recipes.command(id_)
        if command:
            detail += f". Install it: `{command}`"
    return check(id_, label, bool(found), detail, severity=severity, path=found)


def probe(id_: str) -> dict[str, Any] | None:
    """One check, by id or None if it is not one this can answer alone.

    Only the tools are here, and that is the whole point: a check that needs the
    outputs root (`disk`) or a pipeline import (the models) is not a thing the
    app installs, so nothing ever asks for it here. `dubbing_app.install` calls
    this when its subprocess exits, so the row the UI redraws is a fresh
    `shutil.which` and not the package manager's opinion of itself.
    """
    from .install import INSTALLERS

    spec = TOOLS.get(id_)
    if spec is None:
        return None
    # Same row `report` would produce, `installable` and `stage` included: a
    # client that drops this straight into its list must not get a shape one key
    # short it would redraw a REQUIRED row as an untagged one.
    row = tool(id_, *spec)
    return {**row, "installable": id_ in INSTALLERS,
            **({"stage": blocking_stage(id_)} if row["severity"] == BLOCKING else {})}


def model(id_: str, label: str, path: Path, *, severity: str = BLOCKING,
          note: str = "", hub: str = "") -> dict[str, Any]:
    """A model directory's presence and size. `path` always comes from the
    pipeline module that loads it, so this cannot describe a stale location.

    A missing row's whole job is to be actionable: when the hub repo is known,
    the detail carries the exact download command, backticked so the UI sets it
    as code. "Missing" without the command is a scavenger hunt.
    """
    present = path.is_dir() and any(path.iterdir()) if path.is_dir() else False
    size = dir_size(path) if present else 0
    if present:
        detail = f"{human_bytes(size)} in {path}"
    else:
        detail = f"missing: {path}" + (f" {note}" if note else "")
        if hub:
            detail += f". Fetch it: `uv run hf download {hub} --local-dir {path}`"
    return check(id_, label, present, detail, severity=severity, path=str(path), bytes=size)


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------

def hf_token_check(env_file: Path | None = None) -> dict[str, Any]:
    """Presence only. The value is a credential and never leaves this process —
    the report says "set" or "not set" and nothing that could reconstruct it.

    The failing row names the **absolute** `.env` path, backticked so the UI sets
    it as code and offers to copy it. "put HF_TOKEN in .env" is a scavenger hunt
    on a machine with three checkouts; only this process knows which `.env` it
    will actually read.
    """
    path = env_path(env_file)
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if (os.environ.get(var) or "").strip():
            return check("hf_token", "Hugging Face token", True, f"set via {var}",
                         severity=DEGRADES, source="env", path=str(path))
    if _env_file_has_token(path):
        return check("hf_token", "Hugging Face token", True, f"set in `{path}`",
                     severity=DEGRADES, source="env_file", path=str(path))
    return check("hf_token", "Hugging Face token", False,
                 "not set diarization falls back to a single speaker, so every line "
                 "is attributed to one voice. Accept Pyannote's model terms, then add "
                 f"`HF_TOKEN=hf_…` to `{path}`",
                 severity=DEGRADES, source=None, path=str(path))


def env_path(env_file: Path | None = None) -> Path:
    """The `.env` this process reads absolute, because that is the whole point."""
    return (REPO_ROOT / ".env" if env_file is None else Path(env_file)).resolve()


def _env_file_has_token(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN") and value.strip().strip("'\""):
            return True
    return False


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
    from dubbing import hebrew, transcript, translate, tts

    out = [
        model("model.translate", "Translation model (Gemma 4 12B)", translate.MODEL_PATH,
              note=f"downloads from {translate.HUB_ID} on first use"),
    ]
    for key, spec in tts.TTS_MODELS.items():
        default = key == tts.DEFAULT_TTS_MODEL
        out.append(model(f"model.tts.{key}", f"TTS checkpoint {key}"
                         + (" (default)" if default else ""),
                         tts.REPO_ROOT / "models" / spec["dir"],
                         severity=BLOCKING if default else OPTIONAL,
                         note=f"downloads from {spec['hub']} on first use"))
    out += [
        model("model.asr.he", "Source ASR Hebrew (ivrit-ai)", transcript.WHISPER_MODEL,
              severity=OPTIONAL, note="only for Hebrew sources without captions",
              hub=transcript.WHISPER_HUB),
        model("model.asr.src", "Source ASR multilingual", transcript.SRC_ASR_MODEL,
              severity=OPTIONAL, note="only for non-Hebrew sources without captions",
              hub=transcript.SRC_ASR_HUB),
        model("model.asr.en", "Target ASR English (clip verification)",
              transcript.EN_ASR_MODEL,
              note="without it generated clips are never verified",
              hub="Systran/faster-whisper-base.en"),
        model("model.asr.tgt", "Target ASR multilingual", transcript.TARGET_ASR_MODEL,
              severity=OPTIONAL, note="only for non-English targets",
              hub="Systran/faster-whisper-base"),
        model("model.lid", "Language ID (VoxLingua107)", transcript.LID_MODEL,
              severity=DEGRADES, note="without it foreign-speech detection is skipped"),
        # Hebrew is a dub TARGET only with both of these. Optional every other
        # target runs without them but a Hebrew run is refused up front when
        # either is missing, so the report is where a user finds out first.
        model("model.tts.he", "Hebrew TTS adapter (Qwen3-TTS LoRA)", hebrew.ADAPTER_DIR,
              severity=OPTIONAL,
              note=f"only for Hebrew targets {hebrew.ADAPTER_DOWNLOAD}"),
        g2p_check(),
    ]
    return out


def g2p_check() -> dict[str, Any]:
    """The Hebrew grapheme→IPA model. Unlike the others this is a *package* first:
    `renikud-plus` fetches its own weights on first use, so what decides the verdict
    is whether it is importable, and a local `models/RenikudPlus` is reported when
    it exists but is never required."""
    from dubbing import hebrew

    local = hebrew.G2P_FILE.is_file()
    size = dir_size(hebrew.G2P_DIR) if local else 0
    if not hebrew.g2p_ready():
        detail = (f"{hebrew.G2P_PACKAGE} is not installed run `uv sync`; "
                  "without it Hebrew is unavailable as a target")
    elif local:
        detail = f"{human_bytes(size)} in {hebrew.G2P_DIR}"
    else:
        detail = (f"{hebrew.G2P_PACKAGE} installed; weights download from "
                  f"{hebrew.G2P_HUB} on first use")
    return check("model.g2p.he", "Hebrew G2P (ReNikud Plus)", hebrew.g2p_ready(),
                 detail, severity=OPTIONAL, path=str(hebrew.G2P_DIR), bytes=size)


def demucs_check() -> dict[str, Any]:
    """Optional by contract: Demucs fetches `htdemucs_ft` into the torch hub
    cache the first time `stems` runs, so absence is a slow first run, not a
    broken install and not a worse dub."""
    from dubbing import stems

    cache = Path(os.environ.get("TORCH_HOME") or (Path.home() / ".cache" / "torch")) / "hub"
    present = cache.is_dir() and any(cache.rglob("*.th"))
    size = dir_size(cache) if present else 0
    detail = (f"{stems.MODEL} cache: {human_bytes(size)} in {cache}" if present
              else f"{stems.MODEL} not downloaded yet fetched on the first stems run")
    return check("model.demucs", "Demucs stem separation", present, detail,
                 severity=OPTIONAL, path=str(cache), bytes=size)


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

# Where a blocking check's absence actually stops the run. "Runs will fail" is
# true and useless; "runs will fail at translate" is the sentence that tells a
# user whether to fix it now or start the fetch and fix it while it downloads.
BLOCKING_STAGE: dict[str, str] = {
    "ffmpeg": "fetch",          # the first stage that shells out; they all do
    "sox": "tts",               # Qwen3-TTS text normalization
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
    from .install import INSTALLERS

    checks: list[dict[str, Any]] = [tool(id_, *spec) for id_, spec in TOOLS.items()]
    checks.append(hf_token_check())
    checks += model_checks()
    checks.append(demucs_check())
    checks.append(disk_check(Path(outputs)))
    for c in checks:
        c["installable"] = c["id"] in INSTALLERS
        # Only on the rows where it means something: a `stage` on an optional
        # row would read as "this is where it will bite you", which is exactly
        # the false urgency `severity` exists to stop.
        if c["severity"] == BLOCKING:
            c["stage"] = blocking_stage(c["id"])
    ok = all(c["ok"] for c in checks if c["required"])
    return {"ok": ok, "checks": checks}


__all__ = ["report", "probe", "git_commit", "human_bytes", "dir_size", "env_path",
           "blocking_stage", "TOOLS", "BLOCKING", "DEGRADES", "OPTIONAL", "SEVERITIES"]
