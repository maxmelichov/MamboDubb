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
  refinement, the verify ASR, the reference validator), plus `tts.synthesis`
  which this stage adds because no stage can report its own total absence.
* `tts_unavailable` every line that was supposed to be dubbed fell back to
  source audio, which is a stage that never ran rather than a per-segment
  fallback, and is the one warning here that also exits the CLI nonzero.
* `overrun`, `shorten_abandoned`, `subtitles_failed` a drifted line that
  nothing could rescue, and a kept span showing "…" instead of a subtitle.
* `stale_locked_clips` a clip the user approved for a line that has since
  changed. The pipeline may not replace it and may not pass it off as current.
* `faint_endings` a dubbed line whose closing fricative is so far under the line
  it ends that a listener will hear the last word go missing. The one failure in
  this list that `verify` structurally cannot see: an ASR reads the word off its
  vowel and scores a full pass on an /s/ nobody can hear.
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


# How far under a line's own speech level its closing fricative may sit before
# this says so. Set from the corpus rather than from taste: across the four demo
# runs the 113 lines that close on a fricative have a median of -10.8 dB and a
# tenth percentile of -18.5 dB, so this bar names about a tenth of lines and the
# ones it names are the genuinely dead endings, not the merely soft ones.
FAINT_ENDING_DB = -18.0


def faint_line_endings(workdir: Path, dubbed: list[dict]) -> list[dict]:
    """Dubbed lines whose closing fricative is too quiet to survive the mix.

    The acoustic counterpart to `verify`, and it exists because `verify`
    structurally cannot see this class of failure at all. Word overlap asks an
    ASR what it heard, and an ASR reads "us" off the vowel and never reports that
    the /s/ after it was inaudible under a music bed. Energy is a witness that
    can, and it costs one FFT per 30 ms of the last 0.6 s of each clip.

    What it is honestly worth. It catches an ending that is gone or nearly gone,
    which is the shape `trim_trailing_silence` used to manufacture before it
    learned to stop on a fricative. It did NOT catch the line it was written in
    response to: measured over the four demo runs, the "…to us" /s/ lands 13.8 dB
    under its own line where the median line lands at 10.8 dB and the tenth
    percentile at 18.5 dB. On every clip-level number tried (fricative level, and
    the level of whatever follows the fricative) that line is mid-distribution.
    So this is not the check that would have caught it, and it is not offered as
    one. What it is: the cheapest number that makes the failure *arguable from
    evidence* rather than from one person's ear, and a tripwire on any future
    regression that starts eating final consonants across a population.

    Reported, not enforced, and the threshold is deliberately out at the tenth
    percentile. Faint final fricatives are mostly a property of this voice rather
    than of the pipeline, so a bar tight enough to fail this line would fail half
    of every run.

    Lines that end in no fricative are not measured; `final_fricative_db` returns
    None and they are skipped. The gap that leaves is a closing fricative missing
    outright rather than merely faint, which needs per-language orthography to
    know to expect, and is not attempted here.
    """
    out = []
    for s in dubbed:
        rec = (s.get("place") or {}).get("clip") or (s.get("tts") or {}).get("clip")
        if not rec:
            continue
        path = workdir / rec
        if not path.exists():
            continue
        try:
            samples = audio.decode_mono(path, audio.SR)
        except RuntimeError:
            continue
        level = audio.final_fricative_db(samples, audio.SR)
        if level is None or level >= FAINT_ENDING_DB:
            continue
        out.append({"id": s["id"], "start": s["start"], "db": round(level, 1)})
    return out


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


def _verify_counts(segments: list[dict[str, Any]]) -> tuple[dict[str, int], list[dict]]:
    """How the clips that will actually be heard scored, and who sounds wrong.

    "unverified" is its own verdict: the clip cleared the length guard and no
    ASR ever heard it (dubbing/tts.py, NO_ASR). Counting it as "ok" made a run
    with no verifier at all look like a run where everything passed. "accepted"
    is the same argument one bar down: a clip that cleared the accept floor but
    never reached CLONE_GOOD_OVERLAP, after every reference was tried. It is a
    real dub and it is not a clean one, and folding it into "ok" is how a run
    full of half-garbled lines read as fully verified.
    "wrong_voice" is the third: ECAPA says the clip is a different person from
    the segment's own audio (dubbing/tts.py, CLONE_VOICE_MIN). It is a real dub
    of the right words and it is the failure a listener notices first, so it is
    counted apart and named with the lines it happened on.
    """
    verify = {"ok": 0, "accepted": 0, "soft": 0, "keep": 0, "unverified": 0,
              "wrong_voice": 0}
    wrong_voice = []
    for s in segments:
        # The verdict of the clip that will be HEARD. A line the timeline
        # shortened is voiced by a second clip made from the shorter text, and
        # `seg["tts"]` still records the take that one replaced: a shortened
        # segment aired soft-accepted at 0.33 overlap while this count read it as
        # one of the run's "ok"s. `place.verify` is written whenever the aired
        # clip is not the recorded one.
        got = (s.get("place") or {}).get("verify") or (s.get("tts") or {}).get("verify")
        if got is None:
            continue          # no clip at all it is counted in `unaccounted`
        verify[got] = verify.get(got, 0) + 1
        if got == "wrong_voice":
            wrong_voice.append({"id": s["id"], "start": s["start"],
                                "speaker": s.get("speaker"),
                                "voice": (s.get("tts") or {}).get("voice")})
    return verify, wrong_voice


def _shortened_lines(segments: list[dict[str, Any]], run_tgt: str) -> list[dict[str, Any]]:
    """Every line the timeline rewrote shorter, with how much of it survived.

    Retention is measured in speech units (characters for a CJK/Korean target),
    like every other length budget: `.split()` counts a whole Japanese line as
    one word, and every shortening then reported a retention of exactly 1.0.
    """
    def units(text: str, seg: dict[str, Any]) -> int:
        return script.speech_units(text, seg.get("tgt_lang") or run_tgt)

    return [{
        "id": s["id"], "start": s["start"],
        "before": s["text_en"], "after": s["place"]["spoken"],
        "retention": round(units(s["place"]["spoken"], s) / max(1, units(s["text_en"], s)), 2),
    } for s in segments if (s.get("place") or {}).get("spoken")]


def _degraded(m: dict[str, Any], keep_reasons: dict[str, int],
              dubbed: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, int] | None]:
    """What ran degraded (the stages' own `m["health"]`), plus total TTS failure.

    The second half is the one degradation no stage is in a position to report
    about itself. A `tts_failed` keep is the TTS stage's per-segment safety net:
    that line did not come out usable, the source audio stands in for it, and the
    run is a dub with one gap. The same reason on *every* line that was supposed
    to be dubbed is a different event wearing the same name. It means the engine
    never loaded and nothing was synthesized at all, so the "dub" the mix
    produces is the source audio end to end, in the source language.

    That used to finish clean (issue #15): `degraded` empty, exit 0, a
    preview.mp4 indistinguishable from a successful run until someone listened
    to it. The boundary is total failure, not any failure: a partial pile of
    tts_failed keeps is the fallback working as designed and still exits 0.
    """
    health = dict(m.get("health") or {})
    tts_unavailable = None
    if keep_reasons.get("tts_failed") and not dubbed:
        tts_unavailable = {"kept": keep_reasons["tts_failed"]}
        health["tts.synthesis"] = (
            f"produced nothing all {tts_unavailable['kept']} segment(s) fell "
            "back to source audio, so this dub is the untranslated original")
    return health, tts_unavailable


def collect(m: dict[str, Any], workdir: Path) -> dict[str, Any]:
    """Everything `report.json` says about this run. Writes nothing, prints nothing."""
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

    run_tgt = (m.get("source") or {}).get("tgt_lang") or "en"
    shortened = _shortened_lines(segments, run_tgt)
    verify, wrong_voice = _verify_counts(segments)
    health, tts_unavailable = _degraded(m, keep_reasons, dubbed)

    placed = [s for s in segments if s.get("place")]
    drifts = [s["place"]["drift"] for s in placed]
    rates = [s["place"]["rate"] for s in placed]
    overruns = [s["place"]["overrun"] for s in placed if s["place"].get("overrun")]
    # A rescue that was attempted and abandoned the missing half of the
    # shortened/drift story. Not the same thing as a late line: a refused shorten
    # means the clip kept every word and the slot had to absorb it in speed, and
    # the speed-up usually gets there. Its own `overrun` says which happened, so
    # the record carries it and nothing has to guess.
    abandoned = [{"id": s["id"], "start": s["start"],
                  "reason": s["place"]["shorten"],
                  "overrun": s["place"].get("overrun") or 0.0}
                 for s in placed if s["place"].get("shorten")]
    # A kept span whose subtitle is the translate stage's "…" placeholder: the
    # translator refused and the viewer gets an ellipsis where a line should be.
    # (The placeholder is written both by design a passed-through span has no
    # translation to show and by failure; telling the two apart needs a distinct
    # marker in dubbing/translate.py, which is a follow-up. Counting them is the
    # visibility that costs nothing meanwhile.)
    subtitles_failed = [s["id"] for s in kept
                        if (s.get("text_en") or "").strip() == "…"]
    # A locked clip whose text has moved on since it was made: the pipeline may
    # not replace it (the user approved it) and may not pretend it is current.
    stale_locked = [{"id": s["id"], "uid": s.get("uid")}
                    for s in segments if tts.clip_text_stale(s)]
    spans = uncovered_spans(workdir / m["files"]["source_wav"], words, segments,
                            float(m["source"]["duration"]))

    return {
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
        # Not a warning: the one report field that means "this run produced no
        # dub at all". `cli.main` exits 1 on it, the same as `unaccounted`.
        "tts_unavailable": tts_unavailable,
        "degraded": health,
        "overrun": {"count": len(overruns),
                    "max": round(max(overruns, default=0.0), 2)},
        "drift": {
            "max": round(max(drifts, default=0.0), 2),
            "mean": round(sum(drifts) / len(drifts), 3) if drifts else 0.0,
            "over_soft": sum(1 for d in drifts if d > timeline.DRIFT_SOFT),
        },
        "faint_endings": faint_line_endings(workdir, dubbed),
        "speed": {"max": round(max(rates, default=1.0), 3),
                  "compressed": sum(1 for r in rates if r > 1.01)},
        "uncovered_audible": spans,
        "source_mismatch": declared_source_mismatch(m, segments),
    }


def summarize(report: dict[str, Any]) -> None:
    """Print the run's headline counts and every warning in it, to stderr.

    Reads only the report, so what the human is told and what `report.json`
    records cannot drift apart.
    """
    verify = report["verify"]
    print("", file=sys.stderr)
    print(f"  {report['segments']} segments: {report['dubbed']} dubbed, "
          f"{report['kept']} original ({report['keep_reasons']})", file=sys.stderr)
    print(f"  tts verify: {verify}", file=sys.stderr)
    if report["tts_unavailable"]:
        print(f"  FAIL: TTS produced nothing all {report['tts_unavailable']['kept']} "
              "segment(s) fell back to their source audio, so this run's dub is "
              "the untranslated original. The synthesis engine did not load; the "
              "per-segment errors above say why.", file=sys.stderr)
    if report["wrong_voice"]:
        ids = ", ".join(str(w["id"]) for w in report["wrong_voice"][:10])
        print(f"  WARNING: {len(report['wrong_voice'])} clip(s) speak the right words in "
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
    for name, reason in sorted(report["degraded"].items()):
        print(f"  degraded: {name} {reason}", file=sys.stderr)
    print(f"  drift: max {report['drift']['max']}s, {report['drift']['over_soft']} over "
          f"{timeline.DRIFT_SOFT}s | speed-up on {report['speed']['compressed']} segments "
          f"(max {report['speed']['max']}x)", file=sys.stderr)
    if report["overrun"]["count"]:
        print(f"  overrun: {report['overrun']['count']} clip(s) still talking past the next "
              f"speaker's onset, worst {report['overrun']['max']}s", file=sys.stderr)
    if report["faint_endings"]:
        worst = min(report["faint_endings"], key=lambda f: f["db"])
        print(f"  faint endings: {len(report['faint_endings'])} line(s) close on a "
              f"fricative too quiet to carry, worst {worst['db']}dB under the line "
              f"at {worst['start']:.2f}s (seg {worst['id']})", file=sys.stderr)
    abandoned = report["shorten_abandoned"]
    if abandoned:
        # Its own heading, and its own truth. These lines used to print under the
        # overrun count and each claimed to be "still late", so a run with one
        # overrun listed four late segments and three of them were on time.
        print(f"  shorten refused: {len(abandoned)} segment(s) kept every word and "
              "paid for it in speed", file=sys.stderr)
    for item in abandoned:
        late = (f", still {item['overrun']:.2f}s late" if item["overrun"]
                else ", absorbed by the speed-up")
        print(f"    seg {item['id']} @{item['start']:.1f}s {item['reason']}{late}",
              file=sys.stderr)
    if report["subtitles_failed"]:
        print(f"  subtitles: {len(report['subtitles_failed'])} kept segment(s) show the "
              f"\"…\" placeholder instead of a line: {report['subtitles_failed'][:10]}",
              file=sys.stderr)
    for item in report["stale_locked_clips"]:
        print(f"  CONFLICT: seg {item['id']} has a locked clip made for text that "
              "has since changed it speaks the old line (resynthesize it, or "
              "release the lock)", file=sys.stderr)
    if report["shortened"]:
        print(f"  shortened for timing: {len(report['shortened'])} segments", file=sys.stderr)
        for s in report["shortened"]:
            print(f"    seg {s['id']} @{s['start']:.1f}s kept {s['retention']:.0%} of words",
                  file=sys.stderr)
    spans = report["uncovered_audible"]
    if spans:
        total_gap = sum(s["duration"] for s in spans)
        print(f"  uncovered audible audio: {len(spans)} spans, {total_gap:.1f}s total "
              "(music, or transcript missed it)", file=sys.stderr)
        for s in spans[:10]:
            print(f"    {s['start']:.1f}-{s['end']:.1f}s (rms {s['rms']:.3f})", file=sys.stderr)
    if report["unaccounted"]:
        print(f"  FAIL: {len(report['unaccounted'])} segments have no audio: "
              f"{report['unaccounted']}", file=sys.stderr)


def run(m: dict[str, Any], workdir: Path) -> dict[str, Any]:
    """Count, write `report.json`, tell the human what is wrong with the run."""
    report = collect(m, workdir)
    (workdir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    summarize(report)
    return report
