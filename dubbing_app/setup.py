"""First-run environment checks — "can this machine actually run the pipeline?".

The desktop shell needs an answer before it shows an import form, because every
failure mode here is undiagnosable from inside the editor: a missing `ffmpeg`
surfaces as a stage that dies halfway, and an absent model directory silently
becomes a multi-gigabyte download on the first run.

Two rules shape this module:

* **No model is loaded and no weights are touched.** Every check is `which`,
  `os.environ` or `stat`, so `GET /api/setup` answers in milliseconds and can be
  polled. The pipeline imports here are function-local and reach only for
  constants — none of them pulls torch or MLX into the server process.
* **The paths come from the pipeline's own constants**, never restated here. If
  `translate.MODEL_PATH` moves, this check moves with it; a hardcoded copy would
  drift and then cheerfully report a green tick for a model the pipeline cannot
  find.

Report shape::

    {"ok": bool, "checks": [{"id", "label", "ok", "detail",
                             "required": bool, "path"?, "bytes"?}, ...]}

`ok` is the conjunction of the **required** checks only. Everything else is
informational: Demucs and Pyannote download themselves on demand, a missing
HF_TOKEN degrades diarization to a single speaker rather than failing, and the
non-default models are only needed for particular language pairs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

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
    baked-in path — set it at build time and the binary reports its provenance
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

def check(id_: str, label: str, ok: bool, detail: str, *, required: bool = True,
          **extra: Any) -> dict[str, Any]:
    return {"id": id_, "label": label, "ok": bool(ok), "detail": detail,
            "required": bool(required), **extra}


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def dir_size(path: Path) -> int:
    """Bytes on disk under `path`. Symlinks are not followed and unreadable
    entries are skipped — a size is a nicety, never a reason to 500."""
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


def tool(id_: str, label: str, exe: str, why: str, *, required: bool = True) -> dict[str, Any]:
    found = shutil.which(exe)
    return check(id_, label, bool(found), found or f"{exe} not on PATH — {why}",
                 required=required, path=found)


def model(id_: str, label: str, path: Path, *, required: bool = True,
          note: str = "") -> dict[str, Any]:
    """A model directory's presence and size. `path` always comes from the
    pipeline module that loads it, so this cannot describe a stale location."""
    present = path.is_dir() and any(path.iterdir()) if path.is_dir() else False
    size = dir_size(path) if present else 0
    if present:
        detail = f"{human_bytes(size)} in {path}"
    else:
        detail = f"missing: {path}" + (f" — {note}" if note else "")
    return check(id_, label, present, detail, required=required, path=str(path), bytes=size)


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------

def hf_token_check(env_file: Path | None = None) -> dict[str, Any]:
    """Presence only. The value is a credential and never leaves this process —
    the report says "set" or "not set" and nothing that could reconstruct it."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if (os.environ.get(var) or "").strip():
            return check("hf_token", "Hugging Face token", True, f"set via {var}",
                         required=False, source="env")
    path = REPO_ROOT / ".env" if env_file is None else env_file
    if _env_file_has_token(path):
        return check("hf_token", "Hugging Face token", True, f"set in {path.name}",
                     required=False, source="env_file")
    return check("hf_token", "Hugging Face token", False,
                 "not set — diarization falls back to a single speaker",
                 required=False, source=None)


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
        return check("disk", "Free disk space", False, f"unavailable: {exc}", required=False,
                     path=str(outputs))
    return check("disk", "Free disk space", usage.free >= DISK_WARN_BYTES,
                 f"{human_bytes(usage.free)} free of {human_bytes(usage.total)} at {probe}",
                 required=False, path=str(outputs), bytes=usage.free)


def model_checks() -> list[dict[str, Any]]:
    """Every model directory the pipeline opens, read from its own constants.

    Required means "the default run cannot work without it": the translator, the
    default TTS checkpoint and the English ASR that verifies each clip (without
    it verification silently stops happening, which AGENTS.md forbids). The rest
    are per-language-pair or self-downloading, so they are reported and do not
    block.
    """
    from dubbing import transcript, translate, tts

    out = [
        model("model.translate", "Translation model (Gemma 4 12B)", translate.MODEL_PATH,
              note=f"downloads from {translate.HUB_ID} on first use"),
    ]
    for key, spec in tts.TTS_MODELS.items():
        default = key == tts.DEFAULT_TTS_MODEL
        out.append(model(f"model.tts.{key}", f"TTS checkpoint {key}"
                         + (" (default)" if default else ""),
                         tts.REPO_ROOT / "models" / spec["dir"],
                         required=default,
                         note=f"downloads from {spec['hub']} on first use"))
    out += [
        model("model.asr.he", "Source ASR — Hebrew (ivrit-ai)", transcript.WHISPER_MODEL,
              required=False, note="only for Hebrew sources without captions"),
        model("model.asr.src", "Source ASR — multilingual", transcript.SRC_ASR_MODEL,
              required=False, note="only for non-Hebrew sources without captions"),
        model("model.asr.en", "Target ASR — English (clip verification)",
              transcript.EN_ASR_MODEL,
              note="without it generated clips are never verified"),
        model("model.asr.tgt", "Target ASR — multilingual", transcript.TARGET_ASR_MODEL,
              required=False, note="only for non-English targets"),
        model("model.lid", "Language ID (VoxLingua107)", transcript.LID_MODEL,
              required=False, note="without it foreign-speech detection is skipped"),
    ]
    return out


def demucs_check() -> dict[str, Any]:
    """Informational by contract: Demucs fetches `htdemucs_ft` into the torch hub
    cache the first time `stems` runs, so absence is a slow first run, not a
    broken install."""
    from dubbing import stems

    cache = Path(os.environ.get("TORCH_HOME") or (Path.home() / ".cache" / "torch")) / "hub"
    present = cache.is_dir() and any(cache.rglob("*.th"))
    size = dir_size(cache) if present else 0
    detail = (f"{stems.MODEL} cache: {human_bytes(size)} in {cache}" if present
              else f"{stems.MODEL} not downloaded yet — fetched on the first stems run")
    return check("model.demucs", "Demucs stem separation", present, detail,
                 required=False, path=str(cache), bytes=size)


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

def report(outputs: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = [
        tool("ffmpeg", "ffmpeg", "ffmpeg", "every stage shells out to it for audio and video"),
        tool("sox", "SoX", "sox", "Qwen3-TTS text normalization needs it"),
        hf_token_check(),
    ]
    checks += model_checks()
    checks.append(demucs_check())
    checks.append(disk_check(Path(outputs)))
    ok = all(c["ok"] for c in checks if c["required"])
    return {"ok": ok, "checks": checks}


__all__ = ["report", "git_commit", "human_bytes", "dir_size"]
