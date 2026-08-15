"""Derive the UI's dev fixture from a real run directory.

Kept out of the app tree on purpose: the fixture is a checked-in JSON snapshot,
not something the app generates at runtime.
"""
import hashlib
import json
import pathlib
import sys

run = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])

m = json.loads((run / "manifest.json").read_text())
report = json.loads((run / "report.json").read_text()) if (run / "report.json").exists() else None


def uid_for(seg):
    blob = f"{seg['id']}:{seg['start']:.3f}:{seg['end']:.3f}:{seg['speaker']}"
    return "s_" + hashlib.sha1(blob.encode()).hexdigest()[:12]


def verify_for(seg):
    tts = seg.get("tts") or {}
    clip = tts.get("clip")
    if not clip:
        return None
    stem = pathlib.Path(clip).stem
    path = run / "clips" / f"{stem}.json"
    if not path.exists():
        return None
    v = json.loads(path.read_text())
    return {"ok": v.get("ok", True), "overlap": v.get("overlap", 0.0), "heard": v.get("heard", "")}


segments = []
for seg in m["segments"]:
    uid = uid_for(seg)
    s = {
        "uid": uid,
        "id": seg["id"],
        "start": seg["start"],
        "end": seg["end"],
        "speaker": seg["speaker"],
        "text": seg.get("text", ""),
        "text_en": seg.get("text_en"),
        "keep": bool(seg.get("keep")),
        "keep_reason": seg.get("keep_reason"),
        "lang": seg.get("lang"),
        "src_lang": None,
        "tgt_lang": None,
        "tts_opts": None,
        "locked": None,
        "tts": seg.get("tts"),
        "place": seg.get("place"),
        "verify": verify_for(seg),
    }
    segments.append(s)

# A fixture that is all-green teaches the UI nothing. Bend three segments into the
# states a reviewer actually has to deal with: a failed synthesis, a soft-accepted
# clone, and a hand-locked translation.
by_id = {s["id"]: s for s in segments}
for sid, mutate in (
    (11, "failed"),
    (17, "soft"),
    (5, "locked"),
):
    s = by_id.get(sid)
    if not s:
        continue
    if mutate == "failed":
        # What tts.run actually leaves behind when a clip cannot be verified: a
        # keep it decided against itself, with a slice of the original audio
        # attached so the mix is never silent (`Engine.keep_clip`). The record's
        # `verify` is never "failed" — the pipeline only writes ok/soft/keep,
        # and a fixture that invented "failed" gave the UI a failure state that
        # could only ever be reached in fixture mode.
        s["keep"] = True
        s["keep_reason"] = "tts_failed"
        clip = f"clips/{s['uid']}_keep.wav"
        s["tts"] = {"clip": clip, "dur": round(s["end"] - s["start"], 2),
                    "tries": 3, "overlap": 1.0, "verify": "keep"}
        s["place"] = {"start": s["start"], "end": s["end"], "rate": 1.0,
                      "drift": 0.0, "clip": clip}
        s["verify"] = {
            "ok": False,
            "overlap": 0.31,
            "heard": "the qatari the qatari threat is is a much",
        }
    elif mutate == "soft" and s.get("tts"):
        s["tts"] = {**s["tts"], "verify": "soft", "overlap": 0.78, "tries": 3}
        if s.get("verify"):
            s["verify"] = {**s["verify"], "ok": False, "overlap": 0.78}
    elif mutate == "locked":
        s["locked"] = {"text_en": True}

# One segment tagged as a different source language, to exercise the per-segment
# language override in the inspector.
if by_id.get(23):
    by_id[23]["src_lang"] = "ar"
    by_id[23]["lang"] = "ar"

src = dict(m["source"])
src.setdefault("genre", "documentary")
src.setdefault("register", "narration")

detail = {
    "name": run.name,
    "source": src,
    "speakers": m.get("speakers", {}),
    "stages": {k: "done" for k in m.get("stages", {})},
    "outputs": m.get("outputs", {}),
    "report": report,
}

out.write_text(
    json.dumps({"project": detail, "segments": segments}, ensure_ascii=False, indent=1),
    encoding="utf-8",
)
print("wrote", out, len(segments), "segments")
