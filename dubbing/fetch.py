"""Stage 1 — get the video, its audio, and (if available) its own captions."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from . import audio

VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".ts"}


def is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def _lang_prefs(lang: str) -> tuple[str, ...]:
    # YouTube still uses legacy ISO-639 codes for a few languages (mirrors
    # transcript._LID_ALIAS): "iw" Hebrew, "ji" Yiddish, "in" Indonesian.
    alias = {"he": ["iw", "he"], "yi": ["ji", "yi"], "id": ["in", "id"]}.get(lang, [lang])
    return tuple([f"{a}-orig" for a in alias] + alias)


def _download_video(url: str, workdir: Path) -> tuple[Path, str]:
    import yt_dlp

    opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(workdir / "source_video.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    found = sorted(workdir.glob("source_video.*"))
    found = [p for p in found if p.suffix.lower() in VIDEO_EXTS]
    if not found:
        raise SystemExit(f"yt-dlp produced no video file in {workdir}")
    return found[0], info.get("title") or ""


def _download_captions(url: str, workdir: Path, lang: str) -> Path | None:
    import yt_dlp

    prefs = _lang_prefs(lang)
    opts = {
        "skip_download": True,
        "writeautomaticsub": True,
        "writesubtitles": True,
        "subtitleslangs": list(prefs),
        "subtitlesformat": "json3",
        "outtmpl": str(workdir / "captions"),
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info(url, download=True)
    except Exception as exc:  # captions are optional — ASR is the fallback
        print(f"  fetch: caption download failed ({exc})", file=sys.stderr)
        return None

    found = [p for p in workdir.glob("captions*") if ".json3" in p.name]
    if not found:
        return None

    def rank(p: Path) -> int:
        name = p.name.lower()
        for i, pref in enumerate(prefs):
            if f".{pref.lower()}." in name:
                return i
        return len(prefs)

    best = sorted(found, key=rank)[0]
    target = workdir / "captions.json3"
    if best != target:
        best.replace(target)
    for stale in workdir.glob("captions.*.json3"):
        stale.unlink(missing_ok=True)
    return target


def run(
    m: dict[str, Any],
    workdir: Path,
    *,
    source: str,
    captions_file: Path | None,
    duration_limit: float | None,
    src_lang: str,
) -> None:
    audio.require_tools()
    workdir.mkdir(parents=True, exist_ok=True)

    if is_url(source):
        video, title = _download_video(source, workdir)
        caps = _download_captions(source, workdir, src_lang)
        m["source"]["title"] = title
    else:
        video = Path(source).expanduser().resolve()
        if not video.is_file():
            raise SystemExit(f"input not found: {video}")
        caps = None
        if captions_file:
            caps = Path(captions_file).expanduser().resolve()
            if not caps.is_file():
                raise SystemExit(f"captions file not found: {caps}")
        m["source"].setdefault("title", video.stem)

    if captions_file and is_url(source):
        caps = Path(captions_file).expanduser().resolve()

    source_wav = workdir / "source.wav"
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(video)]
    if duration_limit:
        cmd += ["-t", f"{duration_limit:g}"]
    cmd += ["-vn", "-acodec", "pcm_s16le", "-ar", str(audio.SR), str(source_wav)]
    audio.run(cmd)

    m["files"]["video"] = str(video)
    m["files"]["source_wav"] = "source.wav"
    m["files"]["captions_raw"] = str(caps) if caps else None
    m["source"]["duration"] = round(audio.duration(source_wav), 3)
    print(
        f"  fetch: {m['source']['duration']:.1f}s audio"
        f"{' + captions' if caps else ' (no captions — will use ASR)'}",
        file=sys.stderr,
    )
