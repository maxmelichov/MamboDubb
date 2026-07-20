#!/usr/bin/env python3
"""Fetch YouTube auto-captions and cross-check against ASR segments."""

from __future__ import annotations

import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

# Replace ASR text when caption similarity is below this (keep asr_text backup).
HARD_REPLACE_SIM = 0.60


def _norm_he(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    t = re.sub(r"[\"'״׳]", "", t)
    # Auto-caption chrome: music / applause tags, speaker arrows, stage directions.
    t = re.sub(r"\[(?:מוזיקה|applause|music|שירה|קהל)[^\]]*\]", " ", t, flags=re.I)
    t = re.sub(r"\(+[^)]*(?:מוזיקה|applause|music)[^)]*\)+", " ", t, flags=re.I)
    t = re.sub(r">{{2,}", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" -–—")
    return t


def download_youtube_captions(
    url: str,
    out_dir: Path,
    *,
    lang_prefs: tuple[str, ...] = ("iw-orig", "iw", "he"),
) -> Path | None:
    """Download Hebrew auto-captions (json3 preferred) via yt-dlp. Returns path or None."""
    try:
        import yt_dlp
    except ImportError as e:
        raise SystemExit(
            f"yt-dlp is required for YouTube captions ({e}).\n"
            "Install with: uv sync"
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(out_dir / "youtube_captions")
    ydl_opts: dict[str, Any] = {
        "skip_download": True,
        "writesubtitles": False,
        "writeautomaticsub": True,
        "subtitleslangs": list(lang_prefs),
        "subtitlesformat": "json3",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    # yt-dlp names files like youtube_captions.iw-orig.json3
    candidates: list[Path] = []
    for p in out_dir.glob("youtube_captions*"):
        if p.suffix in {".json3", ".json", ".vtt", ".srt"} or ".json3" in p.name:
            candidates.append(p)
    if not candidates:
        # Also check requested_subtitles keys for written files.
        print("  youtube_subs: no caption files written", file=sys.stderr)
        return None

    def _rank(p: Path) -> tuple[int, int]:
        name = p.name.lower()
        lang_score = 99
        for i, pref in enumerate(lang_prefs):
            if pref.lower() in name:
                lang_score = i
                break
        fmt_score = 0 if "json3" in name or p.suffix == ".json3" else 1
        return (lang_score, fmt_score)

    best = sorted(candidates, key=_rank)[0]
    print(f"  youtube_subs: using {best.name}", file=sys.stderr)
    # Stash URL on a sidecar for rebuilds.
    meta = {"url": url, "caption_file": str(best), "title": info.get("title")}
    (out_dir / "youtube_captions_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return best


def parse_json3_cues(path: Path) -> list[dict[str, Any]]:
    """Parse YouTube json3 captions into [{start, end, text}, ...]."""
    data = json.loads(path.read_text(encoding="utf-8"))
    events = data.get("events") or []
    cues: list[dict[str, Any]] = []
    for ev in events:
        segs = ev.get("segs") or []
        if not segs:
            continue
        text = "".join(s.get("utf8") or "" for s in segs)
        text = re.sub(r"\s+", " ", text).strip()
        if not text or text == "\n":
            continue
        start_ms = float(ev.get("tStartMs") or 0)
        dur_ms = float(ev.get("dDurationMs") or 0)
        start = start_ms / 1000.0
        end = (start_ms + dur_ms) / 1000.0 if dur_ms else start + 0.5
        # Prefer word-level timing when available.
        words = []
        for s in segs:
            w = (s.get("utf8") or "").strip()
            if not w or w == "\n":
                continue
            off = float(s.get("tOffsetMs") or 0) / 1000.0
            words.append({"text": w, "start": start + off})
        cues.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
                "words": words,
            }
        )
    return cues


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def align_captions_to_segments(
    segments: list[dict[str, Any]],
    cues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach text_youtube (+ youtube_words) to each segment by time overlap."""
    summary: list[dict[str, Any]] = []
    for seg in segments:
        s0 = float(seg["start"])
        s1 = float(seg["end"])
        parts: list[str] = []
        words_out: list[dict[str, Any]] = []
        for cue in cues:
            ov = _overlap(s0, s1, float(cue["start"]), float(cue["end"]))
            if ov <= 0.05:
                continue
            # Require meaningful overlap relative to cue or segment.
            cue_dur = max(0.1, float(cue["end"]) - float(cue["start"]))
            if ov / cue_dur < 0.25 and ov < 0.4:
                continue
            parts.append((cue.get("text") or "").strip())
            for w in cue.get("words") or []:
                wt = (w.get("text") or "").strip()
                if not wt:
                    continue
                ws = float(w.get("start") or cue["start"])
                if s0 - 0.15 <= ws <= s1 + 0.15:
                    words_out.append({"text": wt, "start": round(ws, 3)})
        yt_text = _norm_he(" ".join(parts))
        asr = _norm_he(seg.get("text") or "")
        sim = SequenceMatcher(None, asr, yt_text).ratio() if (asr and yt_text) else 0.0
        if yt_text:
            seg["text_youtube"] = yt_text
        if words_out:
            seg["youtube_words"] = words_out
        summary.append(
            {
                "start": s0,
                "end": s1,
                "sim": round(sim, 3),
                "asr": asr[:80],
                "youtube": yt_text[:80],
            }
        )
    return summary


def apply_youtube_crosscheck(
    segments: list[dict[str, Any]],
    cues: list[dict[str, Any]],
    *,
    hard_replace_sim: float = HARD_REPLACE_SIM,
    allow_hard_replace: bool = False,
) -> dict[str, int]:
    """Store text_youtube; optionally hard-replace ASR when similarity is very low.

    Default is annotate-only: auto-captions often bleed across segment boundaries,
    so we pass text_youtube to MT as a second opinion instead of overwriting ASR.
    """
    summary = align_captions_to_segments(segments, cues)
    replaced = 0
    annotated = 0
    for seg, row in zip(segments, summary):
        yt = (seg.get("text_youtube") or "").strip()
        if not yt:
            continue
        annotated += 1
        sim = float(row["sim"])
        asr = (seg.get("text") or "").strip()
        print(
            f"  youtube_subs: annotate sim={sim:.2f} "
            f"{float(seg['start']):.1f}-{float(seg['end']):.1f}s",
            file=sys.stderr,
        )
        if not allow_hard_replace:
            continue
        if sim >= hard_replace_sim or not yt or not asr or sim <= 0:
            continue
        # Guard against caption bleed onto short stubs / overlong cues.
        asr_words = max(1, len(asr.split()))
        yt_words = len(yt.split())
        dur = float(seg.get("end", 0) - float(seg.get("start", 0)))
        if dur < 2.0 or asr_words <= 4:
            continue
        if yt_words > asr_words * 2.2 or yt_words < asr_words * 0.4:
            continue
        lang = seg.get("language") or "he"
        keep = bool(seg.get("keep_original", lang != "he"))
        if keep or lang != "he":
            continue
        if "asr_text" not in seg:
            seg["asr_text"] = asr
        seg["text"] = yt
        phrases = seg.get("phrases") or []
        if len(phrases) == 1:
            phrases[0]["text"] = yt
        replaced += 1
        print(
            f"  youtube_subs: REPLACE ASR (sim={sim:.2f}) "
            f"{float(seg['start']):.1f}-{float(seg['end']):.1f}s",
            file=sys.stderr,
        )
        print(f"    ASR: {asr[:70]}", file=sys.stderr)
        print(f"    YT:  {yt[:70]}", file=sys.stderr)
    return {"annotated": annotated, "replaced": replaced}


def crosscheck_workdir(
    workdir: Path,
    url: str,
    segments: list[dict[str, Any]],
) -> dict[str, int]:
    """Download captions into workdir and apply cross-check to segments."""
    cap_dir = workdir / "youtube_captions"
    path = download_youtube_captions(url, cap_dir)
    if path is None:
        return {"annotated": 0, "replaced": 0}
    if "json3" in path.name or path.suffix in {".json3", ".json"}:
        cues = parse_json3_cues(path)
    else:
        print(f"  youtube_subs: unsupported format {path.name}", file=sys.stderr)
        return {"annotated": 0, "replaced": 0}
    print(f"  youtube_subs: {len(cues)} cues from captions", file=sys.stderr)
    return apply_youtube_crosscheck(segments, cues)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Fetch YouTube captions for a run dir.")
    p.add_argument("workdir", type=Path)
    p.add_argument("--url", required=True)
    args = p.parse_args()
    seg_path = args.workdir / "segments.json"
    payload = json.loads(seg_path.read_text(encoding="utf-8"))
    segments = payload.get("segments") or []
    stats = crosscheck_workdir(args.workdir, args.url, segments)
    payload["segments"] = segments
    payload["youtube_url"] = args.url
    seg_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {seg_path} ({stats})", file=sys.stderr)
