"""The single seam between the server and the pipeline.

Nothing else in `dubbing_app` imports `dubbing.*`. The server calls only the
functions below, whose signatures are the ones `docs/APP_ARCHITECTURE.md` gives
for `dubbing/edit.py`.

`dubbing/edit.py` is being written on `feat/segment-ops` and does not exist here
yet, so this module prefers it and falls back to provisional local
implementations. Every provisional piece is marked `PROVISIONAL` with what
replaces it, so reconciliation at merge is mechanical: delete the fallback, keep
the re-export.

The model-loading functions (`retranslate`, `resynthesize`, `rebuild`,
`full_run`) are only ever called from the job child process
(`dubbing_app.worker`), never from the server process — see `dubbing_app.runner`
for why. Their pipeline imports are therefore function-local: importing this
module must not drag torch/MLX into the server.
"""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any, Callable, Iterable

from dubbing import STAGES, manifest

Progress = Callable[[dict[str, Any]], None]

try:  # the real thing, once feat/segment-ops lands
    from dubbing import edit as _edit  # type: ignore
except ImportError:
    _edit = None

HAVE_EDIT = _edit is not None

# Keys the app adds to every segment. On `feat/segment-ops` these live in
# `manifest.SEGMENT_KEYS` itself; until then `manifest.save` would silently drop
# them (AGENTS.md invariant 8), which would lose `uid` on the very first write.
#
# PROVISIONAL: widening the whitelist at import time is the fallback. Delete this
# block when `dubbing/edit.py` exists — by then SEGMENT_KEYS lists these itself.
APP_SEGMENT_KEYS = {"uid", "src_lang", "tgt_lang", "tts_opts", "locked"}
if not HAVE_EDIT:
    manifest.SEGMENT_KEYS |= APP_SEGMENT_KEYS

# Which per-segment fields each stage owns, for `invalidate`.
STAGE_FIELDS = {
    "translate": ("text_en", "text_mid"),
    "tts": ("tts",),
    "timeline": ("place",),
}


if HAVE_EDIT:
    # One name for "the app asked for a structurally impossible edit", whichever
    # side raised it: the routes catch `ops.EditError`, and `dubbing.edit` is the
    # authority on what is impossible.
    EditError = _edit.EditError
    SegmentNotFound = _edit.SegmentNotFound
else:
    class EditError(ValueError):
        """A structurally impossible edit (overlap, non-adjacent merge, no such uid)."""

    class SegmentNotFound(EditError):
        """No segment with that uid."""


# --------------------------------------------------------------------------
# stable identity
# --------------------------------------------------------------------------

def new_uid() -> str:
    return "s" + secrets.token_hex(6)


def ensure_uids(m: dict[str, Any]) -> bool:
    """Mint a `uid` for every segment that lacks one. True when anything changed.

    `seg["id"]` is positional and is renumbered whenever segmentation changes, so
    it cannot key the UI across a re-run. Runs made by the headless CLI have no
    uids at all; this is where they acquire them, once, on first read.
    """
    if HAVE_EDIT and hasattr(_edit, "ensure_uids"):
        return bool(_edit.ensure_uids(m))
    changed = False
    seen: set[str] = set()
    for seg in m.get("segments") or []:
        uid = seg.get("uid")
        if not uid or uid in seen:
            uid = new_uid()
            seg["uid"] = uid
            changed = True
        seen.add(uid)
    return changed


def find(m: dict[str, Any], uid: str) -> dict[str, Any] | None:
    if HAVE_EDIT:
        return _edit.find(m, uid)
    for seg in m.get("segments") or []:
        if seg.get("uid") == uid:
            return seg
    return None


def _need(m: dict[str, Any], uid: str) -> dict[str, Any]:
    seg = find(m, uid)
    if seg is None:
        raise EditError(f"no segment with uid {uid!r}")
    return seg


def _lock(seg: dict[str, Any], field: str) -> None:
    """Record that the user set `field` by hand, so a re-run must not redo it."""
    seg.setdefault("locked", {})[field] = True


def _renumber(m: dict[str, Any]) -> None:
    m["segments"].sort(key=lambda s: (float(s["start"]), float(s["end"])))
    for i, seg in enumerate(m["segments"]):
        seg["id"] = i


# --------------------------------------------------------------------------
# no models, instant — safe to run while a job holds the model slot
# --------------------------------------------------------------------------

def set_text(m: dict[str, Any], uid: str, *, text: str | None = None,
             text_en: str | None = None) -> dict[str, Any]:
    """PROVISIONAL until dubbing.edit.set_text."""
    if HAVE_EDIT:
        return _edit.set_text(m, uid, text=text, text_en=text_en) or _need(m, uid)
    seg = _need(m, uid)
    if text is not None:
        seg["text"] = text
        _lock(seg, "text")
    if text_en is not None:
        seg["text_en"] = text_en
        _lock(seg, "text_en")
    return seg


def set_keep(m: dict[str, Any], uid: str, keep: bool, reason: str = "manual") -> dict[str, Any]:
    """PROVISIONAL until dubbing.edit.set_keep.

    A manual keep is never re-decided by `mark_keep` (docs/APP_ARCHITECTURE.md,
    non-negotiables), hence the lock.
    """
    if HAVE_EDIT:
        return _edit.set_keep(m, uid, keep, reason) or _need(m, uid)
    seg = _need(m, uid)
    seg["keep"] = bool(keep)
    seg["keep_reason"] = reason if keep else None
    _lock(seg, "keep")
    return seg


def set_speaker(m: dict[str, Any], uid: str, speaker: str) -> dict[str, Any]:
    """PROVISIONAL until dubbing.edit.set_speaker."""
    if HAVE_EDIT:
        return _edit.set_speaker(m, uid, speaker) or _need(m, uid)
    seg = _need(m, uid)
    seg["speaker"] = speaker
    _lock(seg, "speaker")
    return seg


def set_bounds(m: dict[str, Any], uid: str, start: float, end: float) -> dict[str, Any]:
    """PROVISIONAL until dubbing.edit.set_bounds. Asserts no neighbour overlap."""
    if HAVE_EDIT:
        return _edit.set_bounds(m, uid, start, end) or _need(m, uid)
    seg = _need(m, uid)
    start, end = float(start), float(end)
    if end <= start:
        raise EditError(f"end ({end}) must be after start ({start})")
    for other in m["segments"]:
        if other is seg:
            continue
        if float(other["start"]) < end and start < float(other["end"]):
            raise EditError(f"bounds {start}-{end} overlap segment {other.get('uid')} "
                            f"({other['start']}-{other['end']})")
    seg["start"], seg["end"] = start, end
    _lock(seg, "bounds")
    # The clip was cut for the old window; where it goes is the timeline's call.
    invalidate(m, uid, stages={"timeline"})
    _renumber(m)
    return seg


def set_langs(m: dict[str, Any], uid: str, *, src_lang: str | None = None,
              tgt_lang: str | None = None) -> dict[str, Any]:
    """PROVISIONAL until dubbing.edit.set_langs."""
    if HAVE_EDIT:
        return _edit.set_langs(m, uid, src_lang=src_lang, tgt_lang=tgt_lang) or _need(m, uid)
    seg = _need(m, uid)
    if src_lang is not None:
        seg["src_lang"] = src_lang or None
        _lock(seg, "src_lang")
    if tgt_lang is not None:
        seg["tgt_lang"] = tgt_lang or None
        _lock(seg, "tgt_lang")
    return seg


def set_tts_opts(m: dict[str, Any], uid: str, **opts: Any) -> dict[str, Any]:
    """PROVISIONAL until dubbing.edit.set_tts_opts (owned by feat/tts-controls)."""
    if HAVE_EDIT:
        return _edit.set_tts_opts(m, uid, **opts) or _need(m, uid)
    seg = _need(m, uid)
    current = dict(seg.get("tts_opts") or {})
    for key, value in opts.items():
        if value is None:
            current.pop(key, None)
        else:
            current[key] = value
    seg["tts_opts"] = current or None
    if not current:
        seg.pop("tts_opts", None)
    _lock(seg, "tts_opts")
    return seg


def set_locked(m: dict[str, Any], uid: str, locked: dict[str, bool]) -> dict[str, Any]:
    """Replace this segment's locks; `{}` releases them all."""
    if HAVE_EDIT:
        return _edit.set_locked(m, uid, locked) or _need(m, uid)
    seg = _need(m, uid)
    kept = {f: True for f, on in (locked or {}).items() if on}
    if kept:
        seg["locked"] = kept
    else:
        seg.pop("locked", None)
    return seg


def invalidate(m: dict[str, Any], uid: str, *, stages: set[str]) -> dict[str, Any]:
    """Drop one segment's output for `stages`, so the stage remakes it.

    Per-segment redo works by deletion: `translate.run` only fills segments whose
    `text_en` is empty and `tts.run` only synthesizes segments with no usable
    clip. This deletes; the stage functions do the rest.

    PROVISIONAL until dubbing.edit.invalidate.
    """
    if HAVE_EDIT:
        return _edit.invalidate(m, uid, stages=set(stages)) or _need(m, uid)
    seg = _need(m, uid)
    # Deleting a stage's output implies deleting everything downstream of it.
    order = [s for s in STAGES if s in STAGE_FIELDS]
    first = min((order.index(s) for s in stages if s in STAGE_FIELDS), default=None)
    if first is None:
        return seg
    for stage in order[first:]:
        for field in STAGE_FIELDS[stage]:
            seg.pop(field, None)
        # A keep the pipeline chose because this stage failed must be re-decided;
        # a keep the user chose by hand outranks the pipeline and stays.
        if seg.get("keep_reason") in ("mt_failed", "tts_failed"):
            seg["keep"], seg["keep_reason"] = False, None
    return seg


def split(m: dict[str, Any], uid: str, at: float) -> tuple[str, str]:
    """PROVISIONAL until dubbing.edit.split. Returns the two new uids."""
    if HAVE_EDIT:
        return _edit.split(m, uid, at)
    seg = _need(m, uid)
    at = float(at)
    start, end = float(seg["start"]), float(seg["end"])
    if not (start < at < end):
        raise EditError(f"split point {at} is not strictly inside {start}-{end}")
    words = (seg.get("text") or "").split()
    cut = round(len(words) * (at - start) / (end - start)) if words else 0
    second = dict(seg)
    second["uid"] = new_uid()
    seg["end"], second["start"] = at, at
    seg["text"] = " ".join(words[:cut])
    second["text"] = " ".join(words[cut:])
    m["segments"].insert(m["segments"].index(seg) + 1, second)
    _renumber(m)
    # Both halves say something different from what was translated and voiced.
    invalidate(m, seg["uid"], stages={"translate"})
    invalidate(m, second["uid"], stages={"translate"})
    return seg["uid"], second["uid"]


def merge(m: dict[str, Any], uid_a: str, uid_b: str) -> str:
    """PROVISIONAL until dubbing.edit.merge. Adjacent, same speaker; returns the uid."""
    if HAVE_EDIT:
        return _edit.merge(m, uid_a, uid_b)
    a, b = _need(m, uid_a), _need(m, uid_b)
    segs = m["segments"]
    i, j = segs.index(a), segs.index(b)
    if i > j:
        a, b, i, j = b, a, j, i
    if j != i + 1:
        raise EditError("only adjacent segments can be merged")
    if a.get("speaker") != b.get("speaker"):
        raise EditError(f"speakers differ ({a.get('speaker')} vs {b.get('speaker')})")
    a["end"] = b["end"]
    a["text"] = " ".join(t for t in ((a.get("text") or "").strip(),
                                     (b.get("text") or "").strip()) if t)
    segs.pop(j)
    _renumber(m)
    invalidate(m, a["uid"], stages={"translate"})
    return a["uid"]


# --------------------------------------------------------------------------
# models — these need the job slot, and only ever run in the job child process
# --------------------------------------------------------------------------

def _emit(progress: Progress | None, event: dict[str, Any]) -> None:
    if progress is not None:
        progress(event)


def _langs(m: dict[str, Any]) -> tuple[str, str]:
    src = m.get("source") or {}
    return src.get("src_lang") or "he", src.get("tgt_lang") or "en"


def _opts(m: dict[str, Any]) -> dict[str, Any]:
    """The CLI options this run was created with; defaults for headless runs.

    Stored by the server at project creation so a later `rebuild` reproduces the
    same pipeline parameters instead of silently switching genre/register.
    """
    stored = dict((m.get("source") or {}).get("app_opts") or {})
    return {"genre": stored.get("genre") or "documentary",
            "register": stored.get("register") or "narration",
            "tts_model": stored.get("tts_model") or "1.7b",
            "device": stored.get("device"),
            "transcript": stored.get("transcript") or "auto",
            "dub_foreign": bool(stored.get("dub_foreign")),
            "captions": stored.get("captions")}


def retranslate(m: dict[str, Any], workdir: Path, uids: Iterable[str], *,
                progress: Progress | None = None) -> dict[str, str]:
    """Re-translate just these segments. Returns uid → new target text.

    PROVISIONAL until dubbing.edit.retranslate.

    Calls `translate.generate` directly rather than `translate.run`: `run`'s
    revision pass is run-global and rewrites every dubbed line, which would
    silently undo the user's other corrections. `locked` is deliberately NOT
    honoured here — an explicit per-segment request outranks the lock that
    protects a segment from a *run-global* re-run.
    """
    if HAVE_EDIT:
        return _edit.retranslate(m, workdir, list(uids), progress=progress)

    from dubbing import translate

    uids = [u for u in uids]
    segs = [_need(m, u) for u in uids]
    src_lang, tgt_lang = _langs(m)
    opts = _opts(m)
    context = (m.get("source") or {}).get("context") or ""
    order = m["segments"]
    before = {s["id"]: prev.get("text") or "" for prev, s in zip(order, order[1:])}

    out: dict[str, str] = {}
    _emit(progress, {"type": "stage", "stage": "translate", "status": "running",
                     "progress": 0.0, "message": f"0/{len(segs)}"})
    tokenizer, model, device = translate.load()
    try:
        for n, seg in enumerate(segs, 1):
            seg_src = seg.get("src_lang") or seg.get("lang") or src_lang
            seg_tgt = seg.get("tgt_lang") or tgt_lang
            text = translate.generate(
                tokenizer, model, seg.get("text") or "", source=seg_src, target=seg_tgt,
                context=context, preceding=before.get(seg["id"], ""), device=device,
                register=opts["register"], genre=opts["genre"],
            )
            seg["text_en"] = text
            # The old clip says the old words; drop it so tts remakes it.
            invalidate(m, seg["uid"], stages={"tts"})
            out[seg["uid"]] = text
            _emit(progress, {"type": "segment", "uid": seg["uid"], "field": "text_en",
                             "status": "done"})
            _emit(progress, {"type": "stage", "stage": "translate", "status": "running",
                             "progress": n / len(segs), "message": f"{n}/{len(segs)}"})
    finally:
        translate.free(model)
    _emit(progress, {"type": "stage", "stage": "translate", "status": "done", "progress": 1.0})
    return out


def resynthesize(m: dict[str, Any], workdir: Path, uids: Iterable[str], *,
                 progress: Progress | None = None) -> dict[str, dict]:
    """Re-voice just these segments, then re-place the timeline. uid → tts record.

    PROVISIONAL until dubbing.edit.resynthesize.

    Drops each segment's clip and placement and re-runs the `tts` and `timeline`
    stage functions: `tts.run` synthesizes exactly the segments with no usable
    clip, so "just these" falls out of the deletion. `timeline.run` re-places
    everything from scratch, which is what keeps the never-overlapping invariant
    true after a clip changes length.
    """
    if HAVE_EDIT:
        return _edit.resynthesize(m, workdir, list(uids), progress=progress)

    uids = [u for u in uids]
    for uid in uids:
        invalidate(m, uid, stages={"tts"})
    rebuild(m, workdir, from_stage="tts", to_stage="timeline", progress=progress)
    return {uid: (find(m, uid) or {}).get("tts") for uid in uids}


def rebuild(m: dict[str, Any], workdir: Path, *, from_stage: str,
            to_stage: str | None = None, progress: Progress | None = None) -> None:
    """Re-run stages forward over whatever has been invalidated.

    PROVISIONAL until dubbing.edit.rebuild.

    Calls the stage functions directly instead of shelling out to `dubbing.cli`.
    The CLI decides what to run from a fingerprint chain, and the fingerprint
    mixes in CLI parameters (genre, register, tts model) that a run made by the
    headless CLI never recorded — guessing them wrong would flip `stage_done` to
    False and trigger `reset_stage`, discarding every translation and clip in the
    run. Calling the stage functions never touches a fingerprint, so the stage
    records stay valid for the headless CLI afterwards.
    """
    if HAVE_EDIT and hasattr(_edit, "rebuild"):
        return _edit.rebuild(m, workdir, from_stage=from_stage, progress=progress)

    from dubbing import mix, report, timeline
    from dubbing import tts as tts_mod

    if from_stage not in STAGES:
        raise EditError(f"unknown stage {from_stage!r}")
    start = STAGES.index(from_stage)
    stop = STAGES.index(to_stage) if to_stage else len(STAGES) - 1
    if stop < start:
        raise EditError(f"{to_stage!r} is upstream of {from_stage!r}")
    todo = [s for s in STAGES[start:stop + 1] if s in ("tts", "timeline", "mix", "report")]
    if len(todo) != stop - start + 1:
        raise EditError(f"rebuild covers tts..report only, not {from_stage!r}..{to_stage!r}")

    src_lang, tgt_lang = _langs(m)
    opts = _opts(m)

    def save() -> None:
        manifest.save(workdir, m)

    engine = None
    try:
        for stage in todo:
            _emit(progress, {"type": "stage", "stage": stage, "status": "running",
                             "progress": 0.0})
            if stage == "tts":
                engine = tts_mod.run(m, workdir, save=save, device=opts["device"],
                                     model=opts["tts_model"])
            elif stage == "timeline":
                engine = engine or tts_mod.Engine(m, workdir, device=opts["device"],
                                                  model=opts["tts_model"])
                timeline.run(m, workdir, genre=opts["genre"], **_retimers(m, workdir, engine,
                                                                         src_lang, tgt_lang, opts))
            elif stage == "mix":
                if engine is not None:
                    engine.close()
                    engine = None
                mix.run(m, workdir)
            elif stage == "report":
                report.run(m, workdir)
            save()
            _emit(progress, {"type": "stage", "stage": stage, "status": "done", "progress": 1.0})
    finally:
        if engine is not None:
            engine.close()


def _retimers(m, workdir: Path, engine, src_lang: str, tgt_lang: str, opts: dict):
    """The shorten/resynth callbacks `timeline.run` uses when a line will not fit.

    PROVISIONAL: reuses `dubbing.cli._retimers` so the app produces byte-identical
    timing to the headless run. `dubbing.edit.rebuild` owns this at merge.
    """
    from argparse import Namespace

    from dubbing import cli

    args = Namespace(src=src_lang, tgt=tgt_lang, genre=opts["genre"],
                     register=opts["register"], device=opts["device"],
                     tts_model=opts["tts_model"])
    shorten_many, resynth_many = cli._retimers(m, workdir, engine, args)
    return {"shorten_many": shorten_many, "resynth_many": resynth_many}


def full_run(workdir: Path, source: dict[str, Any], *,
             progress: Progress | None = None) -> int:
    """A whole pipeline run, fetch → report, via the headless entry point.

    Not part of the edit API: creating a project *is* `python -m dubbing`, and
    re-implementing its stage orchestration in the app would be exactly the fork
    docs/APP_ARCHITECTURE.md forbids. Progress comes from the stderr its stages
    already print, parsed by `dubbing_app.runner`.
    """
    from dubbing import cli

    return cli.main(full_run_argv(workdir, source))


def full_run_argv(workdir: Path, source: dict[str, Any]) -> list[str]:
    opts = _opts({"source": source})
    argv = [str(source["input"]), "-o", str(workdir),
            "--src", source.get("src_lang") or "he",
            "--tgt", source.get("tgt_lang") or "en",
            "--register", opts["register"], "--genre", opts["genre"],
            "--tts-model", opts["tts_model"], "--transcript", opts["transcript"]]
    if source.get("duration_limit"):
        argv += ["--duration", str(source["duration_limit"])]
    if source.get("context"):
        argv += ["--context", str(source["context"])]
    if opts["captions"]:
        argv += ["--captions", str(opts["captions"])]
    if opts["dub_foreign"]:
        argv += ["--dub-foreign"]
    if opts["device"]:
        argv += ["--device", str(opts["device"])]
    return argv
