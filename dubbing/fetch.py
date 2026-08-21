"""Stage 1 get the video, its audio, and (if available) its own captions."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

from . import audio, tools


def localize(video: Path, workdir: Path) -> Path:
    """Copy an outside-the-run input into the run directory, once.

    A run that keeps pointing at the original file depends on it forever: the mix
    stage re-reads the video to mux the preview, so moving the file or macOS
    denying the app access to its folder (a new TCC identity after a rename did
    exactly this) kills a job forty minutes in, at the last stage, with an error
    that looks like anything but its cause. One copy at fetch time makes the run
    self-contained; the disk cost is the price of a run that always finishes.
    """
    if workdir.resolve() in video.resolve().parents:
        return video
    copy = workdir / f"input{video.suffix.lower() or '.mp4'}"
    if copy.is_file() and copy.stat().st_size == video.stat().st_size:
        return copy
    shutil.copy2(video, copy)
    return copy

VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".ts"}


def is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def _lang_prefs(lang: str) -> tuple[str, ...]:
    # YouTube still uses legacy ISO-639 codes for a few languages (mirrors
    # transcript._LID_ALIAS): "iw" Hebrew, "ji" Yiddish, "in" Indonesian.
    # Chinese and Portuguese are the other shape of the same problem: YouTube
    # labels their tracks by script or region ("zh-Hans", "pt-BR") and almost
    # never by the bare code, and `rank()` below matches a preference literally —
    # so without the variants listed here a Chinese video's own captions are
    # invisible and the run falls back to ASR.
    alias = {
        "he": ["iw", "he"], "yi": ["ji", "yi"], "id": ["in", "id"],
        "zh": ["zh", "zh-Hans", "zh-Hant", "zh-CN", "zh-TW"],
        "pt": ["pt", "pt-BR", "pt-PT"],
    }.get(lang, [lang])
    return tuple([f"{a}-orig" for a in alias] + alias)


# Tried in order until one succeeds. YouTube periodically starts refusing the
# web client's stream URLs outright (HTTP 403, mid-2026: PO-token enforcement),
# while the android client keeps serving — but only muxed formats, so the
# fallback trades stream quality for working at all. The ladder is generic
# retry policy, not a per-site hack: each rung is (extractor_args, format).
_VIDEO_ATTEMPTS: tuple[tuple[dict, str], ...] = (
    ({}, "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"),
    ({"youtube": {"player_client": ["android"]}}, "best[ext=mp4]/best"),
    ({"youtube": {"player_client": ["ios"]}}, "best[ext=mp4]/best"),
)


def _download_video(url: str, workdir: Path) -> tuple[Path, str]:
    import yt_dlp

    last: Exception | None = None
    # yt-dlp merges bestvideo+bestaudio by looking ffmpeg up on bare PATH — the
    # one lookup that misses a static build the app installed into the
    # workspace. Hand it the resolved one so both find the same binary.
    ffmpeg = tools.resolve_tool("ffmpeg")
    for i, (extractor_args, fmt) in enumerate(_VIDEO_ATTEMPTS):
        opts = {
            "format": fmt,
            "outtmpl": str(workdir / "source_video.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }
        if ffmpeg:
            opts["ffmpeg_location"] = str(Path(ffmpeg).parent)
        if extractor_args:
            opts["extractor_args"] = extractor_args
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
            break
        except yt_dlp.utils.DownloadError as exc:
            last = exc
            if i + 1 < len(_VIDEO_ATTEMPTS):
                client = (extractor_args.get("youtube", {}).get("player_client")
                          or ["web"])[0]
                print(f"  fetch: download via {client} client failed ({exc}) "
                      "retrying with the next client", file=sys.stderr)
    else:
        raise SystemExit(f"yt-dlp could not download {url}: {last}")
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
    except Exception as exc:  # captions are optional ASR is the fallback
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
        video = localize(video, workdir)
        caps = None
        m["source"].setdefault("title", video.stem)

    # A transcript the user handed us outranks anything the download found it is
    # the whole point of supplying one and on a local video it is the only one
    # there is. Checked here for both kinds of source: the URL branch used to take
    # the path on trust, so a typo in it produced a run that quietly transcribed
    # itself instead of reading the file it was given.
    if captions_file:
        caps = Path(captions_file).expanduser().resolve()
        if not caps.is_file():
            raise SystemExit(f"captions file not found: {caps}")

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
        f"{' + captions' if caps else ' (no captions will use ASR)'}",
        file=sys.stderr,
    )
