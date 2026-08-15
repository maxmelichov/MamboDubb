"""Fixtures for the studio-server tests a run directory built by hand.

The worktree cannot run models, so every test points the server at a temp
`outputs/` containing a manifest shaped exactly like a real one (whatsapp_0809:
9 segments, 38 s) with tiny real WAV files behind the clip paths.
"""

from __future__ import annotations

import json
import struct
import wave
from pathlib import Path
from typing import Any

SEGMENTS = [
    # id, start, end, speaker, keep, keep_reason, text, text_en
    (0, 0.0, 0.34, "SPEAKER_01", True, "tts_failed", "הם", "They."),
    (1, 0.34, 2.2, "SPEAKER_00", False, None, "אנטי-ציוניים.", "Anti-Zionists."),
    (2, 2.74, 11.2, "SPEAKER_00", False, None, "ועם כל הגיא.", "And with all the noise."),
    (3, 11.2, 14.0, "SPEAKER_00", False, None, "מה קרה פה?", "What happened here?"),
    (4, 14.0, 20.0, "SPEAKER_01", False, None, "שאלה טובה.", "Good question."),
]


def write_wav(path: Path, seconds: float = 0.2, rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(seconds * rate)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(struct.pack("<%dh" % frames, *([0] * frames)))


def make_project(root: Path, name: str = "whatsapp_0809") -> Path:
    """A finished run: every stage recorded, clips and verdicts on disk."""
    workdir = root / name
    (workdir / "clips").mkdir(parents=True, exist_ok=True)
    write_wav(workdir / "source.wav", 38.0)
    write_wav(workdir / "dub.wav", 38.0)
    (workdir / "preview.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"video-bytes" * 200)
    (workdir / "preview_en.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n",
                                            encoding="utf-8")

    segments: list[dict[str, Any]] = []
    for sid, start, end, spk, keep, reason, text, text_en in SEGMENTS:
        seg: dict[str, Any] = {"id": sid, "start": start, "end": end, "speaker": spk,
                               "text": text, "keep": keep, "keep_reason": reason,
                               "text_en": text_en}
        if keep:
            clip = f"clips/keep_{sid:012d}.wav"
            write_wav(workdir / clip, end - start)
            seg["tts"] = {"clip": clip, "dur": round(end - start, 3), "tries": 0,
                          "overlap": 1.0, "verify": "keep"}
            seg["place"] = {"start": start, "end": end, "rate": 1.0, "drift": 0.0,
                            "clip": clip}
        else:
            base = f"clips/{sid:016x}.wav"
            fitted = f"clips/fit_{sid:016x}_1.300.wav"
            write_wav(workdir / base, end - start)
            write_wav(workdir / fitted, (end - start) / 1.3)
            (workdir / base).with_suffix(".json").write_text(json.dumps(
                {"ok": True, "overlap": 0.94, "heard": text_en, "dur": round(end - start, 3)}),
                encoding="utf-8")
            seg["tts"] = {"clip": base, "dur": round(end - start, 3), "tries": 1,
                          "overlap": 0.94, "verify": "ok"}
            seg["place"] = {"start": start, "end": end, "rate": 1.3, "drift": 0.0,
                            "clip": fitted}
        segments.append(seg)

    m = {
        "version": 1,
        "source": {"input": "inputs/whatsapp.mp4", "src_lang": "he", "tgt_lang": "en",
                   "duration_limit": None, "title": "whatsapp", "duration": 38.127},
        "files": {"source_wav": "source.wav", "words": "words.json"},
        "stages": {s: {"fp": f"fp{i}"} for i, s in enumerate(
            ("fetch", "stems", "transcript", "segments", "translate", "tts",
             "timeline", "mix", "report"))},
        "progress": {},
        "speakers": {"SPEAKER_00": {"dur": 36.6, "ref": "refs/SPEAKER_00.wav"},
                     "SPEAKER_01": {"dur": 0.34, "ref": None}},
        "segments": segments,
        "outputs": {"dub_wav": "dub.wav", "preview": "preview.mp4"},
    }
    (workdir / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    (workdir / "report.json").write_text(json.dumps(
        {"segments": len(segments), "dubbed": 4, "kept": 1, "unaccounted": []}),
        encoding="utf-8")
    return workdir
