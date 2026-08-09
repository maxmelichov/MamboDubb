"""manifest.json — the single record every stage reads and writes.

Stage results are keyed by a fingerprint chain: each stage's fingerprint mixes in
its predecessor's, so re-running one stage with new params invalidates everything
downstream without any extra bookkeeping.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from . import STAGES

# Bump a stage tag when that stage's logic changes in a way that invalidates its
# previous output. Downstream stages invalidate automatically via the chain.
STAGE_TAGS = {
    "fetch": "fetch/v1",
    "stems": "stems/v1",
    "transcript": "transcript/v38",
    "segments": "segments/v33",
    "translate": "translate/v30",
    "tts": "tts/v12",
    "timeline": "timeline/v10",
    "mix": "mix/v7",
    "report": "report/v1",
}

# Anything not listed here is dropped on save. This is what stops the segment
# record from growing back into the 30-field soup of the old pipeline.
SEGMENT_KEYS = {
    "id",
    "start",
    "end",
    "speaker",
    "text",
    "keep",
    "keep_reason",
    "lang",       # third-language keeps only: what the span's speech is, for subtitles
    "text_en",
    "text_mid",   # pivot runs only: the English intermediate text_en was made from
    "tts",
    "place",
}


def new(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": 1,
        "source": source,
        "files": {},
        "stages": {},
        "progress": {},
        "speakers": {},
        "segments": [],
        "outputs": {},
    }


def reset_stage(m: dict[str, Any], stage: str) -> None:
    """Drop what `stage` produced, so it can be recomputed from scratch.

    Stages also resume *within* themselves (per-segment), so this is only called
    when the stage's fingerprint changed — a resumed run keeps its partial work,
    a re-parameterised one does not.
    """
    if stage == "segments":
        m["segments"] = []
        m["speakers"] = {}
        return
    fields = {"translate": ("text_en", "text_mid"), "tts": ("tts",),
              "timeline": ("place",)}.get(stage)
    if not fields:
        return
    # Undo keep-flips this stage OR anything downstream of it made, so a rerun
    # re-decides them. Downstream flips must go too: a segment kept by tts_failed
    # still holds the translation that failed, and a translate reset that skips it
    # (because it looks "kept") re-feeds the same bad text to the new TTS run —
    # the downstream stage is guaranteed to rerun anyway once this one does.
    undo = {"translate": ("mt_failed", "tts_failed"), "tts": ("tts_failed",)}.get(stage, ())
    for seg in m.get("segments") or []:
        if seg.get("keep_reason") in undo:
            seg["keep"], seg["keep_reason"] = False, None
        for field in fields:
            if (stage == "translate" and field == "text_en" and seg.get("keep")
                    and seg.get("keep_reason") != "foreign"):
                # Structurally kept segments are subtitled by the segments stage;
                # foreign keeps get their subtitle *from* translate, so theirs resets.
                continue
            seg.pop(field, None)


def load(workdir: Path) -> dict[str, Any] | None:
    path = workdir / "manifest.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save(workdir: Path, m: dict[str, Any]) -> None:
    for seg in m.get("segments") or []:
        for key in [k for k in seg if k not in SEGMENT_KEYS]:
            del seg[key]
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / "manifest.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def stage_fingerprint(m: dict[str, Any], stage: str, params: dict[str, Any]) -> str:
    idx = STAGES.index(stage)
    upstream = ""
    if idx:
        upstream = (m.get("stages") or {}).get(STAGES[idx - 1], {}).get("fp", "")
    blob = json.dumps(
        {"tag": STAGE_TAGS[stage], "params": params, "up": upstream},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def stage_done(
    m: dict[str, Any], workdir: Path, stage: str, fp: str, outputs: list[str]
) -> bool:
    rec = (m.get("stages") or {}).get(stage)
    if not rec or rec.get("fp") != fp:
        return False
    return all((workdir / o).exists() for o in outputs)


def mark_stage(m: dict[str, Any], stage: str, fp: str) -> None:
    m.setdefault("stages", {})[stage] = {"fp": fp}


def clear_stage(m: dict[str, Any], stage: str) -> None:
    """Force a full redo of `stage`, discarding even resumable partial work."""
    (m.get("stages") or {}).pop(stage, None)
    (m.get("progress") or {}).pop(stage, None)
