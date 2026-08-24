"""Stage 9 the safety net: prove every second of source speech was handled.

Accounting is a hard gate (a segment is dubbed or kept, nothing else exists).
Everything else is a warning surfaced to the human: shortened lines, drifted
placements, and audible stretches of source audio that no transcript word and no
placement covers which is exactly the shape the old "dead air at 2:04" bug had.

The other half of that job is accounting for what the run *could not do*, because
a verdict written by a failure path is exactly the kind that never reaches the
user otherwise. Four fields exist for it:

* `verify.unverified` clips no ASR ever heard (the verifier did not load).
  Never folded into `ok`; the summary says so out loud when nothing was checked.
  `verify.accepted` is the neighbouring verdict: heard, understood, and only
  good enough after every clone reference had been tried. `verify.wrong_voice`
  is the third: the right words in the wrong person's voice, which word overlap
  alone scores as a pass and a listener catches in one second.
* `degraded` what a stage could not load and what it fell back to
  (`m["health"]`, written by the stages themselves: diarization, turn
  refinement, the verify ASR, the reference validator).
* `overrun`, `shorten_abandoned`, `subtitles_failed` a drifted line that
  nothing could rescue, and a kept span showing "…" instead of a subtitle.
* `stale_locked_clips` a clip the user approved for a line that has since
  changed. The pipeline may not replace it and may not pass it off as current.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import audio, manifest as manifest_mod, script, timeline, transcript
from . import tts

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


def declared_source_mismatch(m: dict[str, Any],
                             segments: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The transcript disagrees with the project's declared source language.

    Script is the one witness that needs no model: when most of the speech that
    was heard is not written in the source language's script, the declaration
    itself is the likeliest bug (an English video imported as Hebrew), and every
    keep/dub verdict downstream inherits it script-keeps then file the real
    language of the video as "already the target". The report cannot fix a
    declaration; it can refuse to let it pass silently. Same-script mistakes
    (a German video declared English) are invisible to this check by nature.
    """
    src = ((m.get("source") or {}).get("src_lang") or "").lower()
    if not src:
        return None
    voiced = [(s, float(s["end"]) - float(s["start"]))
              for s in segments if (s.get("text") or "").strip()]
    total = sum(d for _, d in voiced)
    if total < 30.0:          # too little speech to accuse a declaration
        return None
    in_src = sum(d for s, d in voiced if script.is_script(s["text"], src))
    frac = in_src / total
    if frac >= 0.5:
        return None
    return {"declared": src, "in_source_script": round(frac, 2)}


def run(m: dict[str, Any], workdir: Path) -> dict[str, Any]:
    segments = m["segments"]
    words = transcript.load_words(workdir, m)

    # No placement, no clip file behind the placement, or no clip at all. The
    # third is the one this report used to miss: a segment whose tts record was
    # invalidated and never refilled has nothing to place, and the timeline that
    # deferred over it left the mix to fill its span with the original vocals.
    # It is a hole in the dub whether or not a stale placement survived beside it.
    unaccounted = sorted({
        s["id"] for s in segments
        if not (s.get("tts") or {}).get("clip")
        or not s.get("place") or not (workdir / s["place"]["clip"]).is_file()})
    dubbed = [s for s in segments if not s["keep"]]
    kept = [s for s in segments if s["keep"]]
    keep_reasons: dict[str, int] = {}
    for s in kept:
        keep_reasons[s["keep_reason"] or "?"] = keep_reasons.get(s["keep_reason"] or "?", 0) + 1

    # Retention is measured in speech units (characters for a CJK/Korean target),
    # like every other length budget: `.split()` counts a whole Japanese line as
    # one word, and every shortening then reported a retention of exactly 1.0.
    run_tgt = (m.get("source") or {}).get("tgt_lang") or "en"
    shortened = [{
        "id": s["id"], "start": s["start"],
        "before": s["text_en"], "after": s["place"]["spoken"],
        "retention": round(
            script.speech_units(s["place"]["spoken"], s.get("tgt_lang") or run_tgt)
            / max(1, script.speech_units(s["text_en"], s.get("tgt_lang") or run_tgt)), 2),
    } for s in segments if (s.get("place") or {}).get("spoken")]

    # "unverified" is its own verdict: the clip cleared the length guard and no
    # ASR ever heard it (dubbing/tts.py, NO_ASR). Counting it as "ok" made a run
    # with no verifier at all look like a run where everything passed. "accepted"
    # is the same argument one bar down: a clip that cleared the accept floor but
    # never reached CLONE_GOOD_OVERLAP, after every reference was tried. It is a
    # real dub and it is not a clean one, and folding it into "ok" is how a run
    # full of half-garbled lines read as fully verified.
    # "wrong_voice" is the third: ECAPA says the clip is a different person from
    # the segment's own audio (dubbing/tts.py, CLONE_VOICE_MIN). It is a real dub
    # of the right words and it is the failure a listener notices first, so it is
    # counted apart and named with the lines it happened on.
    verify = {"ok": 0, "accepted": 0, "soft": 0, "keep": 0, "unverified": 0,
              "wrong_voice": 0}
    wrong_voice = []
    for s in segments:
        got = (s.get("tts") or {}).get("verify")
        if got is None:
            continue          # no clip at all it is counted in `unaccounted`
        verify[got] = verify.get(got, 0) + 1
        if got == "wrong_voice":
            wrong_voice.append({"id": s["id"], "start": s["start"],
                                "speaker": s.get("speaker"),
                                "voice": (s.get("tts") or {}).get("voice")})

    drifts = [s["place"]["drift"] for s in segments if s.get("place")]
    rates = [s["place"]["rate"] for s in segments if s.get("place")]
    overruns = [s["place"]["overrun"] for s in segments
                if (s.get("place") or {}).get("overrun")]
    # A rescue that was attempted and abandoned the missing half of the
    # shortened/drift story: this line is late because nothing could fix it.
    abandoned = [{"id": s["id"], "start": s["start"],
                  "reason": s["place"]["shorten"]}
                 for s in segments if (s.get("place") or {}).get("shorten")]
    # A kept span whose subtitle is the translate stage's "…" placeholder: the
    # translator refused and the viewer gets an ellipsis where a line should be.
    # (The placeholder is written both by design a passed-through span has no
    # translation to show and by failure; telling the two apart needs a distinct
    # marker in dubbing/translate.py, which is a follow-up. Counting them is the
    # visibility that costs nothing meanwhile.)
    subtitles_failed = [s["id"] for s in kept
                        if (s.get("text_en") or "").strip() == "…"]
    # What ran degraded this run, written by the stages themselves.
    health = dict(m.get("health") or {})
    # A locked clip whose text has moved on since it was made: the pipeline may
    # not replace it (the user approved it) and may not pretend it is current.
    stale_locked = [{"id": s["id"], "uid": s.get("uid")}
                    for s in segments if tts.clip_text_stale(s)]
    spans = uncovered_spans(workdir / m["files"]["source_wav"], words, segments,
                            float(m["source"]["duration"]))

    report = {
        # What this report is a report *about*. Nothing else on disk says so: an
        # edit made a minute later changes no stage parameter, so a reader had no
        # way to tell a current report from one the manifest had moved past hours
        # ago and it was served as current either way.
        "manifest": manifest_mod.content_fingerprint(m),
        # Where the words came from. "captions" on a run that asked for ASR means
        # every line downstream was translated from a caption track that mangles
        # exactly the words that matter (AGENTS.md, invariant 4) the single most
        # useful thing to know when a dub reads like nonsense.
        "transcript_origin": transcript.origin(m),
        "segments": len(segments),
        "dubbed": len(dubbed),
        "kept": len(kept),
        "keep_reasons": keep_reasons,
        "unaccounted": unaccounted,
        "verify": verify,
        "wrong_voice": wrong_voice,
        "shortened": shortened,
        "shorten_abandoned": abandoned,
        "subtitles_failed": subtitles_failed,
        "stale_locked_clips": stale_locked,
        "degraded": health,
        "overrun": {"count": len(overruns),
                    "max": round(max(overruns, default=0.0), 2)},
        "drift": {
            "max": round(max(drifts, default=0.0), 2),
            "mean": round(sum(drifts) / len(drifts), 3) if drifts else 0.0,
            "over_soft": sum(1 for d in drifts if d > timeline.DRIFT_SOFT),
        },
        "speed": {"max": round(max(rates, default=1.0), 3),
                  "compressed": sum(1 for r in rates if r > 1.01)},
        "uncovered_audible": spans,
        "source_mismatch": declared_source_mismatch(m, segments),
    }
    (workdir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                         encoding="utf-8")

    print("", file=sys.stderr)
    print(f"  {len(segments)} segments: {len(dubbed)} dubbed, {len(kept)} original "
          f"({keep_reasons})", file=sys.stderr)
    print(f"  tts verify: {verify}", file=sys.stderr)
    if wrong_voice:
        ids = ", ".join(str(w["id"]) for w in wrong_voice[:10])
        print(f"  WARNING: {len(wrong_voice)} clip(s) speak the right words in "
              f"another voice (seg {ids}) re-voice them from a reference of the "
              "right speaker", file=sys.stderr)
    if verify["unverified"]:
        checked = verify["ok"] + verify["accepted"] + verify["soft"]
        if not checked:
            print(f"  WARNING: NOT ONE clip was verified no verification ASR "
                  f"loaded; {verify['unverified']} dub(s) accepted on length alone",
                  file=sys.stderr)
        else:
            print(f"  WARNING: {verify['unverified']} clip(s) accepted without "
                  f"verification (of {checked + verify['unverified']} dubbed)",
                  file=sys.stderr)
    if report["source_mismatch"]:
        mm = report["source_mismatch"]
        print(f"  WARNING: only {mm['in_source_script']:.0%} of the speech is written "
              f"in the declared source language ({mm['declared']}) the declaration "
              "is probably wrong; re-create the project with the language the video "
              "actually speaks", file=sys.stderr)
    for name, reason in sorted(health.items()):
        print(f"  degraded: {name} {reason}", file=sys.stderr)
    print(f"  drift: max {report['drift']['max']}s, {report['drift']['over_soft']} over "
          f"{timeline.DRIFT_SOFT}s | speed-up on {report['speed']['compressed']} segments "
          f"(max {report['speed']['max']}x)", file=sys.stderr)
    if overruns:
        print(f"  overrun: {len(overruns)} clip(s) still talking past the next "
              f"speaker's onset, worst {report['overrun']['max']}s", file=sys.stderr)
    for item in abandoned:
        print(f"    seg {item['id']} @{item['start']:.1f}s shorten attempted, "
              f"{item['reason']} original kept, still late", file=sys.stderr)
    if subtitles_failed:
        print(f"  subtitles: {len(subtitles_failed)} kept segment(s) show the "
              f"\"…\" placeholder instead of a line: {subtitles_failed[:10]}",
              file=sys.stderr)
    for item in stale_locked:
        print(f"  CONFLICT: seg {item['id']} has a locked clip made for text that "
              "has since changed it speaks the old line (resynthesize it, or "
              "release the lock)", file=sys.stderr)
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
