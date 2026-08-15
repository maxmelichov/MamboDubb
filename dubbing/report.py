"""Stage 9 — the safety net: prove every second of source speech was handled.

Accounting is a hard gate (a segment is dubbed or kept, nothing else exists).
Everything else is a warning surfaced to the human: shortened lines, drifted
placements, and audible stretches of source audio that no transcript word and no
placement covers — which is exactly the shape the old "dead air at 2:04" bug had.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import audio, manifest as manifest_mod, timeline, transcript

SCAN_SR = 16000
GAP_MIN_SEC = 2.0
GAP_RMS_FLOOR = 0.010


def uncovered_spans(source_wav: Path, words: list[dict], segments: list[dict],
                    total: float) -> list[dict[str, Any]]:
    """Audible stretches with neither a transcript word nor a placed clip."""
    a = audio.decode_mono(source_wav, SCAN_SR)
    hop = 0.1
    levels = audio.frame_rms(a, SCAN_SR, hop)
    covered = np.zeros(len(levels), dtype=bool)

    def mark(t0: float, t1: float) -> None:
        i0 = max(0, int(t0 / hop))
        i1 = min(len(covered), int(t1 / hop) + 1)
        if i1 > i0:
            covered[i0:i1] = True

    for w in words:
        mark(w["t"] - 0.2, w["t"] + 0.8)
    for seg in segments:
        mark(seg["start"], seg["end"])
        p = seg.get("place")
        if p:
            mark(p["start"], p["end"])

    spans: list[dict[str, Any]] = []
    i = 0
    while i < len(covered):
        if covered[i]:
            i += 1
            continue
        j = i
        while j < len(covered) and not covered[j]:
            j += 1
        t0, t1 = i * hop, min(total, j * hop)
        if t1 - t0 >= GAP_MIN_SEC:
            level = float(np.median(levels[i:j])) if j > i else 0.0
            if level >= GAP_RMS_FLOOR:
                spans.append({"start": round(t0, 2), "end": round(t1, 2),
                              "duration": round(t1 - t0, 2), "rms": round(level, 4)})
        i = j
    return spans


def run(m: dict[str, Any], workdir: Path) -> dict[str, Any]:
    segments = m["segments"]
    words = transcript.load_words(workdir, m)

    unaccounted = [s["id"] for s in segments
                   if not s.get("place") or not (workdir / s["place"]["clip"]).is_file()]
    dubbed = [s for s in segments if not s["keep"]]
    kept = [s for s in segments if s["keep"]]
    keep_reasons: dict[str, int] = {}
    for s in kept:
        keep_reasons[s["keep_reason"] or "?"] = keep_reasons.get(s["keep_reason"] or "?", 0) + 1

    shortened = [{
        "id": s["id"], "start": s["start"],
        "before": s["text_en"], "after": s["place"]["spoken"],
        "retention": round(len(s["place"]["spoken"].split())
                           / max(1, len(s["text_en"].split())), 2),
    } for s in segments if (s.get("place") or {}).get("spoken")]

    verify = {"ok": 0, "soft": 0, "keep": 0}
    for s in segments:
        verify[s["tts"]["verify"]] = verify.get(s["tts"]["verify"], 0) + 1

    drifts = [s["place"]["drift"] for s in segments if s.get("place")]
    rates = [s["place"]["rate"] for s in segments if s.get("place")]
    spans = uncovered_spans(workdir / m["files"]["source_wav"], words, segments,
                            float(m["source"]["duration"]))

    report = {
        # What this report is a report *about*. Nothing else on disk says so: an
        # edit made a minute later changes no stage parameter, so a reader had no
        # way to tell a current report from one the manifest had moved past hours
        # ago — and it was served as current either way.
        "manifest": manifest_mod.content_fingerprint(m),
        # Where the words came from. "captions" on a run that asked for ASR means
        # every line downstream was translated from a caption track that mangles
        # exactly the words that matter (AGENTS.md, invariant 4) — the single most
        # useful thing to know when a dub reads like nonsense.
        "transcript_origin": transcript.origin(m),
        "segments": len(segments),
        "dubbed": len(dubbed),
        "kept": len(kept),
        "keep_reasons": keep_reasons,
        "unaccounted": unaccounted,
        "verify": verify,
        "shortened": shortened,
        "drift": {
            "max": round(max(drifts, default=0.0), 2),
            "mean": round(sum(drifts) / len(drifts), 3) if drifts else 0.0,
            "over_soft": sum(1 for d in drifts if d > timeline.DRIFT_SOFT),
        },
        "speed": {"max": round(max(rates, default=1.0), 3),
                  "compressed": sum(1 for r in rates if r > 1.01)},
        "uncovered_audible": spans,
    }
    (workdir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                         encoding="utf-8")

    print("", file=sys.stderr)
    print(f"  {len(segments)} segments: {len(dubbed)} dubbed, {len(kept)} original "
          f"({keep_reasons})", file=sys.stderr)
    print(f"  tts verify: {verify}", file=sys.stderr)
    print(f"  drift: max {report['drift']['max']}s, {report['drift']['over_soft']} over "
          f"{timeline.DRIFT_SOFT}s | speed-up on {report['speed']['compressed']} segments "
          f"(max {report['speed']['max']}x)", file=sys.stderr)
    if shortened:
        print(f"  shortened for timing: {len(shortened)} segments", file=sys.stderr)
        for s in shortened:
            print(f"    seg {s['id']} @{s['start']:.1f}s kept {s['retention']:.0%} of words",
                  file=sys.stderr)
    if spans:
        total_gap = sum(s["duration"] for s in spans)
        print(f"  uncovered audible audio: {len(spans)} spans, {total_gap:.1f}s total "
              "(music, or transcript missed it)", file=sys.stderr)
        for s in spans[:10]:
            print(f"    {s['start']:.1f}-{s['end']:.1f}s (rms {s['rms']:.3f})", file=sys.stderr)
    if unaccounted:
        print(f"  FAIL: {len(unaccounted)} segments have no audio: {unaccounted}",
              file=sys.stderr)
    return report
