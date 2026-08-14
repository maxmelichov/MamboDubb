"""Run directories under `outputs/` — discovery, reading, and safe file access.

The manifest is read and written through `dubbing.manifest` so the editor obeys
the same whitelist every stage does. Nothing here imports a pipeline stage: those
pull in torch, and the web server has no business loading a model.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from dubbing import manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
# DUBBING_OUTPUTS points the editor at another checkout's runs (e.g. a worktree
# serving the runs of the main clone). Default is this repo's outputs/.
OUTPUTS = Path(os.environ.get("DUBBING_OUTPUTS") or REPO_ROOT / "outputs").resolve()
UPLOADS = OUTPUTS / "_uploads"


# Pipeline options a run can have recorded, and which a re-run has to pass back.
# Leaving one out is not "use the default": it changes the stage fingerprint
# (`cli.stage_params`), and a changed fingerprint on an upstream stage is not a
# re-run of that stage — `manifest.reset_stage("segments")` empties
# `m["segments"]`, taking every translation, clip and hand-edit with it.
OPT_KEYS = ("genre", "register", "transcript", "tts_model", "device", "captions",
            "dub_foreign")


def is_url(source: str) -> bool:
    return bool(re.match(r"https?://", source.strip()))


def recorded_opts(m: dict[str, Any]) -> dict[str, Any]:
    """The pipeline options this run was made with, if it recorded any.

    Two places, because two front ends wrote them: the studio stores them under
    `source.app_opts`, and `dubbing.edit._args` reads them flat off `source`. Flat
    wins where both exist. A run made by the headless CLI records neither — it has
    nowhere to record them today — so a re-run of one still uses the defaults.
    """
    src = m.get("source") or {}
    stored = dict(src.get("app_opts") or {})
    stored.update({key: src[key] for key in OPT_KEYS if src.get(key) is not None})
    return {key: value for key, value in stored.items()
            if key in OPT_KEYS and value is not None}


def default_workdir(source: str) -> Path:
    """Where `python -m dubbing <source>` would put its run.

    Mirrors `dubbing.cli.default_workdir` — importing that module would drag the
    whole model stack into the server process.
    """
    if is_url(source):
        vid = re.search(r"(?:v=|youtu\.be/|/shorts/|/live/)([A-Za-z0-9_-]{6,})", source)
        slug = vid.group(1) if vid else re.sub(r"\W+", "_", source)[-24:]
        return OUTPUTS / f"yt_{slug}"
    return OUTPUTS / re.sub(r"\W+", "_", Path(source).stem)[:48]


def workdir(run: str) -> Path:
    """Resolve a run name to its directory, refusing anything outside outputs/."""
    path = (OUTPUTS / run).resolve()
    if path.parent != OUTPUTS.resolve() or not path.is_dir():
        raise KeyError(run)
    return path


def resolve_in_run(run: str, relative: str) -> Path:
    """A file inside a run directory, with traversal refused."""
    root = workdir(run)
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise KeyError(relative)
    return path


def list_runs() -> list[dict[str, Any]]:
    out = []
    if not OUTPUTS.is_dir():
        return out
    for path in sorted(OUTPUTS.iterdir()):
        if not path.is_dir() or path.name.startswith("_"):
            continue
        m = manifest.load(path)
        if m is None:
            continue
        src = m.get("source") or {}
        out.append({
            "run": path.name,
            "title": src.get("title") or src.get("input") or path.name,
            "input": src.get("input"),
            "src_lang": src.get("src_lang"),
            "tgt_lang": src.get("tgt_lang"),
            "duration": src.get("duration"),
            "segments": len(m.get("segments") or []),
            "stages": sorted((m.get("stages") or {}).keys()),
            "has_preview": (path / "preview.mp4").is_file(),
            "mtime": path.stat().st_mtime,
        })
    return sorted(out, key=lambda r: r["mtime"], reverse=True)


def load(run: str) -> dict[str, Any]:
    m = manifest.load(workdir(run))
    if m is None:
        raise KeyError(run)
    return m


def save(run: str, m: dict[str, Any]) -> None:
    manifest.save(workdir(run), m)


def view(run: str) -> dict[str, Any]:
    """Everything the editor screen needs for one run, in one response."""
    root = workdir(run)
    m = load(run)
    media = {
        name: name for name in ("preview.mp4", "source_video.mp4", "source.wav", "dub.wav")
        if (root / name).is_file()
    }
    return {
        "run": run,
        "source": m.get("source") or {},
        "speakers": m.get("speakers") or {},
        "stages": sorted((m.get("stages") or {}).keys()),
        "media": media,
        "segments": m.get("segments") or [],
    }


def dubbed_clip(run: str, seg: dict[str, Any]) -> Path:
    """The audio this segment currently contributes to the mix."""
    for rel in ((seg.get("place") or {}).get("clip"), (seg.get("tts") or {}).get("clip")):
        if rel:
            try:
                return resolve_in_run(run, rel)
            except KeyError:
                continue
    raise KeyError("no clip yet")


def original_slice(run: str, seg: dict[str, Any]) -> Path:
    """Cut this segment's span out of the run's source audio, for A/B preview.

    Cached outside the run directory: `outputs/` is the pipeline's, and a stray
    file there is exactly the kind of thing the no-hand-edits rule is about. The
    source never changes for a run, so a slice is only ever cut once.
    """
    start, end = float(seg["start"]), float(seg["end"])
    out = (Path(tempfile.gettempdir()) / "dubbing-editor-cache" / run
           / f"orig_{seg['id']}_{start:.2f}_{end:.2f}.wav")
    if out.is_file():
        return out
    src = workdir(run) / "source.wav"
    if not src.is_file():
        raise KeyError("source.wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{max(start, 0):.3f}",
         "-t", f"{max(end - start, 0.05):.3f}", "-i", str(src), str(out)],
        check=True,
    )
    return out
