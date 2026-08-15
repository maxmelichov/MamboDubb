"""Per-segment editing — the only door the app has into a run.

The pipeline produces a dub; this module is how a human corrects one. Every
function takes a loaded manifest `m`, mutates it in place and leaves saving to the
caller, so a server can batch several edits into one `manifest.save`.

Three rules hold everything together:

* **Identity is `uid`, never `id`.** `id` is positional and renumbered whenever
  segmentation changes (see `manifest.mint_uid`).
* **Redo is deletion.** `translate.run` fills segments whose `text_en` is empty and
  `tts.run` synthesizes segments with no usable clip, so `invalidate` deletes and
  the stage functions do the rest. Deleting one artifact deletes what was built on
  it: a new line means a new clip means a new placement.
* **The user's edits outrank the pipeline.** Every hand-edit sets a flag in
  `locked`, and no stage rerun regenerates a locked field (`manifest.is_locked`).
  The exception is targeted: `retranslate`/`resynthesize` on an explicit list of
  uids is the user asking for *this* segment to be redone, so it clears the lock it
  replaces. A whole-run rebuild never does.

Nothing here loads a model except `retranslate`, `resynthesize` and `rebuild` —
those are the slow ones, and the only ones that need the app's job slot.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from . import MANUAL_REASON, STAGES, manifest, ttsopts

Progress = Callable[[float, str], None]

# What each stage owns on a segment, and what its result was built on. Mirrors
# `manifest.reset_stage`, per segment instead of per run.
STAGE_FIELDS: dict[str, tuple[str, ...]] = {
    "translate": ("text_en", "text_mid"),
    "tts": ("tts",),
    "timeline": ("place",),
}
# The lock that holds a stage's result back, per stage. `timeline` has none:
# placement is all-or-nothing and is never the user's (see manifest.LOCK_FIELDS).
STAGE_LOCK = {"translate": "text_en", "tts": "tts"}
DOWNSTREAM = {"translate": ("tts", "timeline"), "tts": ("timeline",), "timeline": ()}
# Stages `rebuild` can start from. Everything earlier needs the source media and
# belongs to a full `python -m dubbing` run, not to an edit.
REBUILDABLE = ("translate", "tts", "timeline", "mix", "report")
# `tts_opts` keys that only reach the synthesis call. A kept segment has no
# synthesis, so on one they are inert — see `set_tts_opts`. (`speed` applies:
# it is post-processing on the finished slice. `keep_pauses` asks for exactly
# what a kept slice already is, so it is a documented no-op rather than a lie.)
SYNTHESIS_ONLY_OPTS = ("seed", "greedy", "ref", "ref_text", "model", "temperature",
                       "top_p", "top_k", "repetition_penalty", "max_new_tokens")


class EditError(ValueError):
    """An edit that cannot be applied — the caller sent something invalid."""


class SegmentNotFound(EditError):
    """No segment with that uid."""


class RebuildIncomplete(RuntimeError):
    """A rebuild ran to the end and the report says the dub still has holes.

    Not an `EditError`: the caller asked for something perfectly valid and the
    pipeline could not deliver it. `cli.main` says the same thing by exiting 1.
    """


def _progress(progress: Progress | None, fraction: float, message: str) -> None:
    if progress is not None:
        progress(max(0.0, min(1.0, float(fraction))), message)


def _lock(seg: dict[str, Any], *fields: str) -> None:
    locked = seg.setdefault("locked", {})
    for field in fields:
        locked[field] = True


def _unlock(seg: dict[str, Any], *fields: str) -> None:
    locked = seg.get("locked") or {}
    for field in fields:
        locked.pop(field, None)
    if not locked:
        seg.pop("locked", None)
    else:
        seg["locked"] = locked


def _renumber(m: dict[str, Any]) -> None:
    """Restore `id` as the position in time after a structural edit."""
    m["segments"].sort(key=lambda s: (float(s["start"]), float(s["end"])))
    for i, seg in enumerate(m["segments"]):
        seg["id"] = i


# --------------------------------------------------------------- lookup

def find(m: dict[str, Any], uid: str) -> dict[str, Any] | None:
    """The segment with this stable uid, or None."""
    for seg in m.get("segments") or []:
        if seg.get("uid") == uid:
            return seg
    return None


def index_of(m: dict[str, Any], uid: str) -> int:
    """Position of `uid` in the segment list. Raises `SegmentNotFound`."""
    for i, seg in enumerate(m.get("segments") or []):
        if seg.get("uid") == uid:
            return i
    raise SegmentNotFound(f"no segment {uid!r}")


def _require(m: dict[str, Any], uid: str) -> dict[str, Any]:
    seg = find(m, uid)
    if seg is None:
        raise SegmentNotFound(f"no segment {uid!r}")
    return seg


# --------------------------------------------------------------- invalidation

def invalidate(m: dict[str, Any], uid: str, *, stages: set[str]) -> set[str]:
    """Drop this segment's results for `stages`, so a rerun regenerates them.

    Downstream comes with it — a translation the clip was made from cannot be
    dropped while the clip stays. Locked fields survive: the user's text is not the
    translator's to reissue. Returns the field names actually removed.

    A keep this pipeline decided for itself (`mt_failed`, `tts_failed`) is undone
    too, exactly as `manifest.reset_stage` does — through the same predicate
    (`manifest.undo_pipeline_keep`), so the two cannot drift apart — rather than
    the segment being stuck on a verdict about text that no longer exists. That
    predicate is where the two nuances live: a keep the user asked for
    (`passthrough` True) is theirs and stays, while a `locked.keep` guarding a
    failure verdict written over the user's "dub it" (`keep.user_wants_dub`) does
    not protect it.

    A kept span's subtitle survives on the same rule
    (`manifest.keeps_own_subtitle`): translate never refills one it did not write,
    so dropping it here would leave the span with no subtitle at all. The caller
    that is *flipping* the keep drops the line itself, since the flip is what
    makes it the wrong kind (see `set_keep`).

    The run's stage marks come off with the fields (`manifest.reopen_from`, mix and
    report included). Deleting a clip while nine stages still say "done" is how an
    edited segment ended up silent in a dub the CLI called up to date: nothing
    would ever refill the hole, and `report.json` went on claiming full coverage.
    Every caller gets that for free — which is why it lives here and not in them.
    """
    seg = _require(m, uid)
    want: set[str] = set()
    for stage in stages:
        if stage not in STAGE_FIELDS:
            raise EditError(f"unknown stage {stage!r}; choose from "
                            f"{', '.join(sorted(STAGE_FIELDS))}")
        want.add(stage)
        want.update(DOWNSTREAM[stage])
    manifest.reopen_from(m, min(want, key=STAGES.index))

    dropped: set[str] = set()
    for stage in ("translate", "tts", "timeline"):
        lock = STAGE_LOCK.get(stage)
        if stage not in want or (lock and manifest.is_locked(seg, lock)):
            continue
        for field in STAGE_FIELDS[stage]:
            if (stage == "translate" and field == "text_en"
                    and manifest.keeps_own_subtitle(seg)):
                continue
            if seg.pop(field, None) is not None:
                dropped.add(field)
    for stage in ("translate", "tts"):
        if stage in want:
            manifest.undo_pipeline_keep(seg, manifest.PIPELINE_KEEPS[stage])
    return dropped


# --------------------------------------------------------------- no-model edits

def set_text(m: dict[str, Any], uid: str, *, text: str | None = None,
             text_en: str | None = None, lock: bool = True) -> dict[str, Any]:
    """Rewrite the source transcription and/or the dubbed line.

    Editing `text_en` is the common case: the user fixes a translation, and the
    clip made from the old one has to go. Editing `text` (the transcription the
    translator reads) drops the translation as well, unless the user already locked
    that. `lock=False` is for a programmatic edit that a rerun may still improve on.
    """
    seg = _require(m, uid)
    if text is None and text_en is None:
        raise EditError("set_text needs text= and/or text_en=")
    # A committed value identical to what is stored is not an edit. Without this,
    # an editor that lets the user click into a line and click out again would
    # invalidate the clip, stamp a lock, and queue a resynthesis — for nothing.
    if text is not None and text.strip() == seg.get("text"):
        text = None
    if text_en is not None and text_en.strip() == seg.get("text_en"):
        text_en = None
    if text is None and text_en is None:
        return seg
    if text is not None:
        if not text.strip():
            raise EditError("text cannot be empty — use set_keep or merge instead")
        seg["text"] = text.strip()
        if lock:
            _lock(seg, "text")
        invalidate(m, uid, stages={"translate"})
    if text_en is not None:
        if not text_en.strip():
            raise EditError("text_en cannot be empty — a dubbed segment must say "
                            "something; use set_keep(keep=True) instead")
        # The clip provably says the old line, whatever the user approved about it.
        _unlock(seg, "tts")
        # Invalidate before locking, or the write below is the very thing this
        # call's own lock would protect from it. `translate`, not `tts`: supplying
        # the line by hand replaces the translate stage's result, so a keep the
        # translator's own failure caused (`mt_failed`) is answered and reopened —
        # left standing, the user's line would only ever be a subtitle. A keep the
        # span or the user decided is untouched; see `invalidate`'s undo table.
        invalidate(m, uid, stages={"translate"})
        seg["text_en"] = text_en.strip()
        if lock:
            _lock(seg, "text_en")
        # The pivot intermediate described the old line; keeping it would let a
        # shorten re-translate from text the user has already replaced.
        seg.pop("text_mid", None)
    return seg


def set_keep(m: dict[str, Any], uid: str, keep: bool,
             reason: str = "manual") -> dict[str, Any]:
    """Play this segment's original audio (or stop doing so).

    A manual keep is never re-decided by `mark_keep` or undone by a stage rerun, so
    it locks. Either direction changes what is heard, so the clip and its placement
    go: `tts.run` gives a keep its original-audio slice and a dub its synthesis.

    So does the line. A keep's `text_en` is a *subtitle* — for a foreign or
    passed-through span it is the honest "…" placeholder — and a dub's is what the
    voice says; leaving one behind across a flip is how a segment ends up speaking
    a placeholder, since `translate.needs_translation` sees a non-empty `text_en`
    and never refills it. `segments.apply_passthrough` drops it on exactly this
    flip and so does this, minus what the user wrote by hand.
    """
    seg = _require(m, uid)
    seg["keep"] = bool(keep)
    seg["keep_reason"] = reason if keep else None
    # One concept, one key: the user's own verdict is also the pipeline's
    # `passthrough` override, so a headless re-run honours it identically
    # (segments.apply_passthrough) and it survives re-segmentation
    # (carry_passthrough). EVERY write through this door carries it, whatever
    # reason the caller names: the pipeline reads `passthrough`, not the reason
    # string, so a keep written without it is reverted by the next run — the
    # user's verdict overruled by a rerun, which is the one thing a lock exists
    # to prevent. A stage that keeps a segment for its own reasons does not come
    # through here (it writes `keep_reason` directly).
    seg["passthrough"] = bool(keep)
    _lock(seg, "keep")
    # The clip is the wrong kind now (a synthesis for a keep, or a slice of the
    # original for a dub), whatever the tts lock said about the old one.
    _unlock(seg, "tts")
    invalidate(m, uid, stages={"translate"})
    # `invalidate` protects a kept span's subtitle (it is not translate's to
    # discard) — but this call is the flip itself, and the line it left behind was
    # written for the other path. Same rule as `segments.apply_passthrough`: it
    # goes, minus what the user wrote by hand.
    if keep and not manifest.is_locked(seg, "text_en"):
        seg.pop("text_en", None)
        seg.pop("text_mid", None)
    return seg


def set_speaker(m: dict[str, Any], uid: str, speaker: str) -> dict[str, Any]:
    """Reassign the speaker — which is also the voice this line is cloned from."""
    if not (speaker or "").strip():
        raise EditError("speaker cannot be empty")
    seg = _require(m, uid)
    if seg.get("speaker") != speaker:
        seg["speaker"] = speaker
        _unlock(seg, "tts")          # the old clip is in the old voice
        invalidate(m, uid, stages={"tts"})
    _lock(seg, "speaker")
    return seg


def set_bounds(m: dict[str, Any], uid: str, start: float, end: float) -> dict[str, Any]:
    """Move this segment's span. Asserts it stays inside its neighbours.

    Segments never overlap and never reorder — `timeline.place` relies on both, and
    a keep's clip is cut from exactly this span.

    No lock is stamped: nothing honours one. Re-running the segments stage rebuilds
    every span from the words and carries only `passthrough` forward, so a
    `locked.bounds` promised the user something the next run would quietly break
    (see `manifest.LOCK_FIELDS`).
    """
    seg = _require(m, uid)
    start, end = round(float(start), 3), round(float(end), 3)
    if start < 0.0:
        raise EditError("start cannot be negative")
    if end <= start:
        raise EditError(f"end {end} must be after start {start}")
    i = index_of(m, uid)
    segs = m["segments"]
    if i and start < float(segs[i - 1]["end"]) - 1e-6:
        raise EditError(f"start {start} overlaps the previous segment "
                        f"(ends at {segs[i - 1]['end']})")
    if i + 1 < len(segs) and end > float(segs[i + 1]["start"]) + 1e-6:
        raise EditError(f"end {end} overlaps the next segment "
                        f"(starts at {segs[i + 1]['start']})")
    if (seg["start"], seg["end"]) != (start, end):
        seg["start"], seg["end"] = start, end
        _unlock(seg, "tts")          # a keep clip is cut from the span itself
        invalidate(m, uid, stages={"tts"})
    return seg


def set_langs(m: dict[str, Any], uid: str, *, src_lang: str | None = None,
              tgt_lang: str | None = None) -> dict[str, Any]:
    """Override the language pair for this segment alone.

    "This section is actually Arabic", or "translate this bit into French". Both
    change what the translator is asked for, so the translation goes with them.
    Passing "" clears an override and falls back to the run's languages.

    `tgt_lang` reaches the synthesiser too (`tts.Engine.tgt_for`): the line is
    prepared, cached, spoken and verified in that language. It used to be honoured
    by the translator alone, so a French line was voiced and ASR-checked as
    English — it failed verification every time and fell back to keep.
    """
    seg = _require(m, uid)
    changed = False
    for key, value in (("src_lang", src_lang), ("tgt_lang", tgt_lang)):
        if value is None:
            continue
        value = value.strip().lower()
        if not value:
            changed = seg.pop(key, None) is not None or changed
        elif seg.get(key) != value:
            seg[key] = value
            changed = True
    if changed:
        _unlock(seg, "text_en")      # the locked line is in the wrong language now
        invalidate(m, uid, stages={"translate"})
    return seg


def set_tts_opts(m: dict[str, Any], uid: str, **opts: Any) -> dict[str, Any]:
    """Per-segment synthesis overrides (seed, greedy, reference, style…).

    The keys and their meanings belong to `dubbing.ttsopts`, which validates the
    *result* of the patch, not just the patch: an unknown key, an out-of-range
    value or a combination that could do nothing (greedy plus a sampler) is
    rejected here. Storing one unvalidated would only move the `ValueError` into
    the middle of the next run, where `Engine._plan` raises it and takes the whole
    stage down instead of the edit that caused it. `None` removes an option; no
    options at all removes the whole record, as does a patch that only restates
    the defaults.

    On a segment that is currently a KEEP there is no synthesis call, so the
    options that only reach one (`SYNTHESIS_ONLY_OPTS`) are refused instead of
    being stored where they would sit doing nothing — the same rule ttsopts
    applies to greedy-plus-sampler, one step further out. What does apply to a
    keep is `speed`, which `tts.keep_clip` bakes into the slice.
    """
    seg = _require(m, uid)
    if seg.get("keep"):
        inert = sorted(k for k in opts if k in SYNTHESIS_ONLY_OPTS and opts[k] is not None)
        if inert:
            raise EditError(
                f"{', '.join(inert)}: this segment plays its original audio "
                f"(keep_reason={seg.get('keep_reason')!r}), so nothing synthesizes it "
                "and these would do nothing. Un-keep it first (set_keep(keep=False)), "
                "or set `speed`, which does apply to a kept slice.")
    try:
        current = ttsopts.merge(seg.get("tts_opts"), opts)
    except ValueError as exc:
        raise EditError(str(exc)) from exc
    if current == (seg.get("tts_opts") or {}):
        return seg
    if current:
        seg["tts_opts"] = current
    else:
        seg.pop("tts_opts", None)
    _unlock(seg, "tts")              # the clip predates these options
    invalidate(m, uid, stages={"tts"})
    return seg


def set_locked(m: dict[str, Any], uid: str, locked: dict[str, bool]) -> dict[str, Any]:
    """Replace this segment's locks outright. `{}` releases every one of them.

    Editing a field locks it, which is what stops a re-run overwriting the user's
    work — so releasing is the only way to hand a line back to the pipeline after
    changing your mind. It is deliberately a replace, not a merge: "unlock this"
    has to be expressible, and a merge could only ever add.
    """
    seg = _require(m, uid)
    unknown = sorted(set(locked) - set(manifest.LOCK_FIELDS))
    if unknown:
        raise EditError(f"not lockable: {', '.join(unknown)}; "
                        f"choose from {', '.join(manifest.LOCK_FIELDS)}")
    kept = {f: True for f, on in locked.items() if on}
    if kept:
        seg["locked"] = kept
    else:
        seg.pop("locked", None)
    return seg


# --------------------------------------------------------------- structure

def split(m: dict[str, Any], uid: str, at: float) -> tuple[str, str]:
    """Cut one segment into two at `at` seconds. Returns the two new uids.

    Every word survives: the text is divided at the word boundary nearest `at`, and
    a segment with fewer than two words cannot be split at all — half of it would
    have nothing to say. Both halves are new segments with new uids, and everything
    generated from the old text (translation, clip, placement) is dropped: none of
    it describes either half.
    """
    seg = _require(m, uid)
    at = round(float(at), 3)
    start, end = float(seg["start"]), float(seg["end"])
    if not (start < at < end):
        raise EditError(f"split point {at} is not inside [{start}, {end}]")
    words = (seg.get("text") or "").split()
    if len(words) < 2:
        raise EditError("cannot split a segment of fewer than two words without "
                        "leaving a half with no text")
    # Where the cut falls in the span, in words. Clamped so neither half is empty.
    k = round(len(words) * (at - start) / (end - start))
    k = max(1, min(len(words) - 1, int(k)))

    left = _derived(seg, start=start, end=at, text=" ".join(words[:k]))
    right = _derived(seg, start=at, end=end, text=" ".join(words[k:]))
    assert (left["text"] + " " + right["text"]).split() == words, "split lost a word"

    i = index_of(m, uid)
    m["segments"][i:i + 1] = [left, right]
    manifest.ensure_uids(m["segments"])
    _renumber(m)
    # Two segments with nothing generated for them, in a run that says translate
    # onward is done. Reopened here rather than in `invalidate` (no field was
    # dropped — the whole segment was replaced), and never earlier than translate:
    # re-running the segments stage would rebuild the list from the words and
    # undo the cut. `mark_keep` re-decides the halves the way it decided the whole.
    manifest.reopen_from(m, "translate")
    return left["uid"], right["uid"]


def merge(m: dict[str, Any], uid_a: str, uid_b: str) -> str:
    """Join two adjacent same-speaker segments. Returns the new uid.

    Adjacent and same-speaker only: merging across a speaker change would make one
    clip in one voice for two people, and merging across a gap would claim seconds
    the pipeline decided nobody was speaking. The merged span is `a.start`→`b.end`
    and holds both texts in order; nothing is dropped but what was generated from
    the two halves separately.
    """
    if uid_a == uid_b:
        raise EditError("cannot merge a segment with itself")
    i, j = index_of(m, uid_a), index_of(m, uid_b)
    if j < i:
        i, j = j, i
    if j != i + 1:
        raise EditError("only adjacent segments can be merged")
    a, b = m["segments"][i], m["segments"][j]
    if a.get("speaker") != b.get("speaker"):
        raise EditError(f"different speakers ({a.get('speaker')} vs {b.get('speaker')})")

    text = " ".join(t for t in ((a.get("text") or "").strip(),
                                (b.get("text") or "").strip()) if t)
    merged = _derived(a, start=float(a["start"]), end=float(b["end"]), text=text)
    # A merged span plays as one thing: it is only kept when both halves were, and
    # only carries a language or synthesis override the two halves agreed on.
    merged["keep"] = bool(a.get("keep") and b.get("keep"))
    merged["keep_reason"] = (a.get("keep_reason")
                             if merged["keep"] and a.get("keep_reason") == b.get("keep_reason")
                             else (MANUAL_REASON if merged["keep"] else None))
    for key in ("lang", "src_lang", "tgt_lang", "tts_opts", "detected_lang",
                "passthrough"):
        if a.get(key) != b.get(key):
            merged.pop(key, None)
    if merged["keep"] and merged["keep_reason"] == MANUAL_REASON:
        # The merged span is now a keep the *user* owns, and `passthrough` is the
        # only carrier of that: it is what a headless re-run honours and the only
        # thing `carry_passthrough` can re-attach after a re-segmentation. Stamping
        # the reason without it would invent a keep no later run reproduces.
        merged["passthrough"] = True
    m["segments"][i:j + 1] = [merged]
    manifest.ensure_uids(m["segments"])
    _renumber(m)
    manifest.reopen_from(m, "translate")   # one new segment, nothing generated for it
    return merged["uid"]


def _derived(seg: dict[str, Any], *, start: float, end: float, text: str) -> dict[str, Any]:
    """A new segment carved out of `seg`: same voice and settings, no results.

    Its uid is minted fresh from its own span and text — it is not the segment the
    old uid named, and a UI still pointing at that uid must be told so rather than
    silently shown a different piece of audio. Locks go too: they were the user's
    approval of text that no longer exists.

    `passthrough` does not: it is the user's word about the SPAN, not about the
    text, and every second of the span is still covered by the pieces carved out
    of it. Dropped, it would survive as far as the next re-segmentation and no
    further — `mark_keep` would re-decide both halves with no override left for
    `carry_passthrough` to re-attach, and the passage would be dubbed again.
    `detected_lang` rides along for the same reason: it is what the classifier
    heard over that span, so it still describes every piece of it, and it is what
    the app reads to suggest passthrough in the first place.
    """
    out: dict[str, Any] = {
        "id": seg.get("id", 0),
        "uid": manifest.mint_uid(start, end, text),
        "start": round(start, 3),
        "end": round(end, 3),
        "speaker": seg.get("speaker"),
        "text": text,
        "keep": bool(seg.get("keep")),
        "keep_reason": seg.get("keep_reason") if seg.get("keep") else None,
    }
    for key in ("lang", "src_lang", "tgt_lang", "tts_opts", "detected_lang",
                "passthrough"):
        if seg.get(key) is not None:
            out[key] = seg[key]
    return out


# --------------------------------------------------------------- model work

def _langs(m: dict[str, Any], seg: dict[str, Any]) -> tuple[str, str]:
    """This segment's (source, target) — per-segment knowledge, then the run's.

    One rule for both front ends: the choice is `translate.segment_langs`, which
    the pipeline stage uses too, so a line re-translated from the app takes the
    same hops it would in a headless run — including the empty source that means
    "nothing knows what language this line is" (a Latin line inside a Hebrew run).

    Normalised through `cli.normalize_lang`, so "iw" and "he" are one language
    here as they are everywhere else — the pair decides whether this segment needs
    a translation at all (`translate.same_language`).
    """
    from . import cli, translate

    src = (m.get("source") or {})
    source, target = translate.segment_langs(seg, src.get("src_lang") or "he",
                                             src.get("tgt_lang") or "en")
    return (cli.normalize_lang(source) if source else ""), cli.normalize_lang(target)


def _established(m: dict[str, Any], target: str):
    """The proper-noun spellings the finished script already uses, for consistency."""
    from . import translate

    return tuple(translate.canonical_names(
        [n for s in m.get("segments") or []
         if (s.get("text_en") or "").strip() and not s.get("keep")
         for n in translate._name_occurrences(s["text_en"], target)]))


def _preceding(m: dict[str, Any], seg: dict[str, Any], pivot: bool, target: str,
               own_pair: bool = False) -> str:
    """The line just before this one, in the language the hop writes.

    Same rule as `translate.run`: the previous English intermediate for a pivot's
    first hop, the previous target line for a direct one, and the previous source
    text when neither is usable. A segment off the run's own language pair stands
    alone — its neighbour spoke a different language and carries no signal.
    """
    from . import translate

    if own_pair or seg.get("lang") or seg.get("src_lang"):
        return ""
    i = index_of(m, seg["uid"])
    if not i:
        return ""
    prev = m["segments"][i - 1]
    if pivot:
        prior = (prev.get("text_mid") or "").strip()
    else:
        prior = (prev.get("text_en") or "").strip()
        if not translate.is_target_text(prior, target):
            prior = ""               # a kept or failed neighbour: subtitle, not output
    return prior or (prev.get("text") or "").strip()


def retranslate(m: dict[str, Any], workdir: Path, uids: Sequence[str], *,
                progress: Progress | None = None, register: str = "narration",
                genre: str = "documentary",
                respect_locked: bool = False) -> dict[str, str]:
    """Re-translate exactly these segments. Returns {uid: new text_en}.

    Deliberately not `translate.run`: that ends with a revision pass over the whole
    script, which would rewrite every other line the user has corrected. This is the
    same per-segment recipe — pivot via English where the pair needs it, numbers
    spelled in code, the run's established name spellings for consistency — applied
    to these segments only.

    Asking for a specific segment overrides its `text_en` lock (that is what the
    button in front of the user means); `respect_locked=True` skips locked lines
    instead, for a bulk redo. What it never overrides is the *verdict*: whether
    this span is dubbed or plays as recorded belongs to `keep`/`passthrough` and
    to `invalidate`'s rule about them, not to a new line arriving. A fresh
    translation of a kept span is a better subtitle for it; a translation that
    fails keeps the original audio as in the pipeline — unless the user asked for
    a dub, which stands (`translate.mark_failed`).

    The returned map holds only the segments that came back translated, so its
    size is the honest count of what this call achieved; the rest is reported on
    the progress channel and left visibly unfinished.
    """
    from . import numwords, translate

    segs = [_require(m, uid) for uid in uids]
    if not segs:
        return {}
    if respect_locked:
        segs = [s for s in segs if not manifest.is_locked(s, "text_en")]
    context = (m.get("source") or {}).get("context") or ""
    out: dict[str, str] = {}
    # Same-language segments need no model: the line is already in the target
    # language, so "re-translate" means restoring it verbatim (see `translate.run`).
    # Filtered before the load, so a purely same-language redo never pays for Gemma.
    same: list[dict[str, Any]] = []
    todo: list[dict[str, Any]] = []
    for seg in segs:
        (same if translate.same_language(*_langs(m, seg)) else todo).append(seg)
    for seg in same:
        _unlock(seg, "text_en")
        invalidate(m, seg["uid"], stages={"translate"})
        seg["text_en"] = translate._finalize_numbers(
            (seg.get("text") or "").strip(), _langs(m, seg)[1])
        out[seg["uid"]] = seg["text_en"]
    asked, segs = len(segs), todo
    if not segs:
        _progress(progress, 1.0, f"translated {len(out)} of {asked} segment(s)")
        return out
    from . import cli

    run_pair = (cli.normalize_lang((m.get("source") or {}).get("src_lang") or "he"),
                cli.normalize_lang((m.get("source") or {}).get("tgt_lang") or "en"))
    failed: list[dict[str, Any]] = []
    _progress(progress, 0.0, f"loading translator for {len(segs)} segment(s)")
    processor, model, device = translate.load()
    try:
        for n, seg in enumerate(segs):
            _progress(progress, n / len(segs), f"translating {n + 1}/{len(segs)}")
            source, target = _langs(m, seg)
            pivot = translate.pivot_via_english(source, target)
            preceding = _preceding(m, seg, pivot, target,
                                   (source, target) != run_pair)
            seg_ctx = translate.relevant_context(context, seg["text"], source)
            mid = ""
            if pivot:
                mid = translate.generate(processor, model, seg["text"], source=source,
                                         target="en", context=seg_ctx, preceding=preceding,
                                         device=device, register=register, genre=genre,
                                         names=_established(m, "en"))
                if not translate.is_target_text(mid, "en"):
                    text = ""
                else:
                    mid = numwords.spell_numbers(mid.strip(), "en")
                    text = translate.generate(processor, model, mid, source="en",
                                              target=target, context=seg_ctx,
                                              preceding="", device=device,
                                              register=register, genre=genre,
                                              names=_established(m, target),
                                              numbers_spelled=True, asr_source=False)
            else:
                src_text = seg["text"]
                en_direct = source == "en" and target != "en"
                if en_direct:
                    src_text = numwords.spell_numbers(src_text, "en")
                text = translate.generate(processor, model, src_text, source=source,
                                          target=target, context=seg_ctx,
                                          preceding=preceding, device=device,
                                          register=register, genre=genre,
                                          names=_established(m, target),
                                          numbers_spelled=en_direct)
            text = translate._finalize_numbers(text, target)
            # The result is the machine's, so the lock the user's old text held is
            # gone — and the clip built on that text with it.
            _unlock(seg, "text_en")
            invalidate(m, seg["uid"], stages={"translate"})
            if translate.is_target_text(text, target):
                seg["text_en"] = text.strip()
                if pivot:
                    seg["text_mid"] = mid
                # The verdict is not this call's to change. `invalidate` above
                # already reopened the one keep a new translation answers — the
                # pipeline's own `mt_failed` — under the rule that protects the
                # user's. Writing keep=False here reached past exactly that rule:
                # it un-kept spans the user had kept, left their `passthrough`
                # saying the opposite, and the next run's `apply_passthrough`
                # flipped them back and deleted this translation on the way.
                out[seg["uid"]] = seg["text_en"]
            else:
                failed.append(seg)
                where = ("keep original" if translate.mark_failed(seg)
                         else "still untranslated (the user asked for a dub)")
                print(f"  edit: seg {seg['id']} failed to translate → {where}",
                      file=sys.stderr)
    finally:
        translate.free(model)
    # Say what happened. A job that translated nothing and reported "translated 1
    # segment(s)" is how a line goes round the flip-and-retry loop forever: the
    # user is told the work is done and hears no change.
    done = f"translated {len(out)} of {asked} segment(s)"
    if failed:
        done += (f" — {len(failed)} failed to translate "
                 f"(seg {', '.join(str(s['id']) for s in failed)})")
    _progress(progress, 1.0, done)
    return out


class ResynthResult(dict):
    """`{uid: tts record}` — plus `summary`, how each one actually got there.

    A dict subclass so every existing caller (and the job result the app relays,
    which is JSON) sees exactly the mapping it always did. The breakdown rides
    alongside because "synthesized 4 segment(s)" was a lie in two directions: a
    segment that fell back to its original audio was counted as synthesized, and
    a placement that could not run was not mentioned at all.
    """

    def __init__(self, records: dict[str, dict[str, Any]], summary: dict[str, Any]):
        super().__init__(records)
        self.summary = dict(summary)


def _resynth_message(summary: dict[str, Any],
                     unvoiced: Sequence[dict[str, Any]] = ()) -> str:
    """The one line the user reads about the job. Says what happened, not what was asked.

    `unvoiced` rides beside the summary rather than in it: those segments were not
    a synthesis outcome at all (nothing was translated for them to speak), and the
    summary's four counts are what every caller reads as the job's result.
    """
    parts = [f"synthesized {summary['synthesized']}"]
    if summary["fell_back"]:
        parts.append(f"{summary['fell_back']} fell back to the original audio")
    if summary["kept"]:
        parts.append(f"{summary['kept']} re-sliced from the original")
    line = ", ".join(parts)
    if unvoiced:
        line += (f" — {len(unvoiced)} still needs a translation "
                 f"(seg {', '.join(str(s['id']) for s in unvoiced)})")
    if summary["deferred_placement"]:
        line += "; placement deferred to the next tts run"
    return line


def resynthesize(m: dict[str, Any], workdir: Path, uids: Sequence[str], *,
                 progress: Progress | None = None, device: str | None = None,
                 model: str | None = None) -> ResynthResult:
    """Re-voice exactly these segments. Returns {uid: tts record} + a breakdown.

    Never silent, as in the stage itself: a segment whose clip cannot be verified
    falls back to its original audio (`keep_reason="tts_failed"`) rather than to
    nothing, and a segment that is already a keep gets a fresh slice of the original.
    Asking for a segment overrides its tts lock; the new clip is unlocked, being the
    machine's work again.

    One case is not a synthesis failure at all and must not be dressed as one: a
    dub with no line to speak, on a span the user asked to dub. Nothing was voiced
    because nothing was translated (`translate.mark_failed`), and answering with a
    keep would hand back the very audio they were replacing — the second half of
    the bug where "Dub it" appears to do nothing. It is left unvoiced and counted,
    and the timeline is deferred rather than built over the hole.

    The fallback and the re-slice *are* failures to synthesize, and so is a
    re-placement that could not run (`_replace_timeline`), so every one of them is
    counted (`ResynthResult.summary`) and stated in the final message rather than
    rolled into a success total the user has no way to check.
    """
    from . import keep as keep_mod
    from . import tts as tts_mod

    counts = {"synthesized": 0, "fell_back": 0, "kept": 0}
    segs = [_require(m, uid) for uid in uids]
    if not segs:
        return ResynthResult({}, {**counts, "deferred_placement": False})
    out: dict[str, dict[str, Any]] = {}
    unvoiced: list[dict[str, Any]] = []
    _progress(progress, 0.0, f"loading synthesiser for {len(segs)} segment(s)")
    eng = tts_mod.Engine(m, workdir, device=device,
                         model=model or tts_mod.DEFAULT_TTS_MODEL)
    try:
        eng.build_speaker_refs()
        for n, seg in enumerate(segs):
            _progress(progress, n / len(segs), f"synthesizing {n + 1}/{len(segs)}")
            _unlock(seg, "tts")
            invalidate(m, seg["uid"], stages={"tts"})
            if seg.get("keep"):
                record = eng.keep_clip(seg)
                counts["kept"] += 1
            else:
                record = eng.clip_for(seg, seg.get("text_en") or "")
                if record is None:
                    if (not (seg.get("text_en") or "").strip()
                            and keep_mod.user_wants_dub(seg)):
                        unvoiced.append(seg)
                        print(f"  edit: seg {seg['id']} has no translation to "
                              f"speak → left unvoiced", file=sys.stderr)
                        continue
                    seg["keep"], seg["keep_reason"] = True, "tts_failed"
                    record = eng.keep_clip(seg)
                    counts["fell_back"] += 1
                    print(f"  edit: seg {seg['id']} unusable → keep original",
                          file=sys.stderr)
                else:
                    counts["synthesized"] += 1
            seg["tts"] = record
            out[seg["uid"]] = record
    finally:
        eng.close()
    # A new clip is a new length, so its old placement is gone (invalidate drops
    # it) and `timeline.place` is the only thing allowed to decide the new one.
    # Re-place before returning: a segment left unplaced is missing from the mix,
    # which is the never-silent invariant broken by the back door. Rendering the
    # preview stays an explicit action — this restores the manifest, not the video.
    _progress(progress, 0.95, "re-placing the timeline")
    summary = {**counts, "deferred_placement": not _replace_timeline(m, workdir)}
    _progress(progress, 1.0, _resynth_message(summary, unvoiced))
    return ResynthResult(out, summary)


def _replace_timeline(m: dict[str, Any], workdir: Path) -> bool:
    """Re-run placement only. Returns whether it could.

    No shortening round: that would reload the translator to rewrite lines the user
    did not ask about.

    Placement is all-or-nothing — `timeline.build_items` reads every segment's clip
    to lay the whole run out in one forward pass — so it can only run when every
    segment has one. Correcting several lines and re-voicing one of them is the
    ordinary studio loop and leaves the rest without a clip until they are asked
    for; there is no timeline to build over that hole, and the tts run that fills
    it re-places everything anyway. Deferring beats dying with a KeyError after the
    new clips are already in the manifest.
    """
    from . import timeline as timeline_mod

    waiting = sum(1 for s in m.get("segments") or [] if not (s.get("tts") or {}).get("clip"))
    if waiting:
        print(f"  edit: {waiting} segment(s) still without audio — placement "
              f"deferred to the next tts run", file=sys.stderr)
        return False
    timeline_mod.run(m, workdir, genre=getattr(_args(m), "genre", "documentary"))
    return True


def start_stage(m: dict[str, Any], from_stage: str) -> tuple[str, str | None]:
    """Where a rebuild from `from_stage` must really start, and why.

    A rebuild is asked for a stage; what it can *run* depends on what the segments
    hold. `timeline.build_items` reads `seg["tts"]` for every segment in one
    forward pass and `tts.run` reads `seg["text_en"]` for every dubbed one, so
    either dies with a bare `KeyError` on the ordinary post-edit manifest — the
    state every corrected-but-not-yet-revoiced line leaves behind.

    So the missing work is *made*, not tripped over: a render that finds segments
    without audio starts at tts, and one that finds dubbed segments without a line
    starts at translate. Backing up is the honest reading of "render this run" and
    it is what `_replace_timeline` already does per call; refusing would leave the
    user with a button that only works on runs that did not need it. Never the
    other direction — a rebuild asked for translate is not quietly narrowed.
    """
    idx = STAGES.index(from_stage)
    segs = m.get("segments") or []
    if idx > STAGES.index("translate"):
        untranslated = [s for s in segs
                        if not s.get("keep") and not (s.get("text_en") or "").strip()]
        if untranslated:
            return "translate", (f"{len(untranslated)} segment(s) have no line to "
                                 f"speak — starting from translate")
    if idx > STAGES.index("tts"):
        unvoiced = [s for s in segs if not (s.get("tts") or {}).get("clip")]
        if unvoiced:
            return "tts", (f"{len(unvoiced)} segment(s) have no audio — starting "
                           f"from tts")
    return from_stage, None


def rebuild(m: dict[str, Any], workdir: Path, *, from_stage: str,
            progress: Progress | None = None, save: Callable[[], None] | None = None,
            **overrides: Any) -> list[str]:
    """Re-run the pipeline from `from_stage` forward. Returns the stages run.

    The whole-run path, and the one that must honour every lock: a rerun of
    translate leaves hand-corrected lines alone, a rerun of tts leaves approved
    clips alone (`manifest.reset_stage`). Stages are re-marked with the fingerprint
    the CLI would compute for them, so a later `python -m dubbing` on the same run
    sees this work as done rather than redoing it.

    Only the manifest-side stages can be rebuilt — anything before `translate` needs
    the source media and belongs to a real run.

    What it can actually run is decided *before* anything is cleared
    (`start_stage`): this used to reset timeline, mix and report and only then die
    on the first segment with no clip, which left the run strictly worse than the
    Render button found it.

    Raises `RebuildIncomplete` when the report it ends on finds segments with no
    audio — the same verdict `python -m dubbing` exits 1 on, so a render job fails
    instead of reporting success over a dub with holes in it.
    """
    from . import cli, mix, report
    from . import timeline as timeline_mod
    from . import translate
    from . import tts as tts_mod

    if from_stage not in REBUILDABLE:
        raise EditError(f"cannot rebuild from {from_stage!r}; choose from "
                        f"{', '.join(REBUILDABLE)}")
    from_stage, why = start_stage(m, from_stage)
    if why:
        print(f"  edit: {why}", file=sys.stderr)
        _progress(progress, 0.0, why)
    args = _args(m, **overrides)
    params = cli.stage_params(args, m)
    todo = [s for s in STAGES[STAGES.index(from_stage):]]
    for stage in todo:
        manifest.clear_stage(m, stage)
    for stage in todo[1:]:
        manifest.reset_stage(m, stage)
    manifest.reset_stage(m, from_stage)

    save = save or (lambda: manifest.save(workdir, m))
    engine = None
    unaccounted: list[int] = []
    try:
        for n, stage in enumerate(todo):
            _progress(progress, n / len(todo), f"{stage}")
            if stage == "translate":
                translate.run(m, workdir, source=args.src, target=args.tgt, save=save,
                              register=args.register, genre=args.genre)
            elif stage == "tts":
                engine = tts_mod.run(m, workdir, save=save, device=args.device,
                                     model=args.tts_model)
            elif stage == "timeline":
                engine = engine or tts_mod.Engine(m, workdir, device=args.device,
                                                  model=args.tts_model)
                shorten_many, resynth_many = cli._retimers(m, workdir, engine, args)
                timeline_mod.run(m, workdir, shorten_many=shorten_many,
                                 resynth_many=resynth_many, genre=args.genre)
            elif stage == "mix":
                if engine is not None:
                    engine.close()
                    engine = None
                mix.run(m, workdir)
            elif stage == "report":
                unaccounted = list((report.run(m, workdir) or {}).get("unaccounted") or [])
            manifest.mark_stage(m, stage, manifest.stage_fingerprint(m, stage, params[stage]))
            save()
    finally:
        if engine is not None:
            engine.close()
    _progress(progress, 1.0, f"rebuilt {len(todo)} stage(s)")
    if unaccounted:
        # `cli.main` exits 1 on exactly this, and the app's job runner treats a
        # non-zero exit as a failed job. A render that ends with segments the mix
        # has no audio for is the same failure through a different door, and
        # reporting it as success is how a dub with holes reaches the user.
        raise RebuildIncomplete(
            f"{len(unaccounted)} segment(s) have no audio in the mix "
            f"({', '.join(str(i) for i in unaccounted[:8])}"
            f"{'…' if len(unaccounted) > 8 else ''}) — see report.json")
    return todo


def _args(m: dict[str, Any], **overrides: Any):
    """CLI arguments for this run: whatever it recorded, then the CLI's defaults.

    The run's settings live in `m["source"]`, so a rebuild reproduces the fingerprints
    of the run that made the manifest instead of inventing new ones — the same
    resolution `cli.main` applies to a bare re-run (`cli.resolve_settings`), so the
    two paths cannot drift. A setting the run never recorded falls back to the CLI
    default; a caller that knows better passes it in.
    """
    from . import cli

    src = m.get("source") or {}
    args = cli.parse_args([str(src.get("input") or "input")])
    # Normalised exactly as `cli.main` does, so a manifest carrying the legacy
    # spelling ("iw") is the same pair here that it is on the headless path.
    args.src = cli.normalize_lang(src.get("src_lang") or args.src)
    args.tgt = cli.normalize_lang(src.get("tgt_lang") or args.tgt)
    args.duration = src.get("duration_limit")
    args.context = src.get("context")
    # The CLI records settings as flat keys; the studio app records the same
    # settings under source["app_opts"]. Reading only the flat spelling made
    # every app-created project silently fall back to CLI defaults here — a
    # "movie" run re-placed its timeline with documentary timing on every
    # per-line re-voice. Flat wins when both exist (it is the older writer), and
    # `cli.resolve_settings` fills in whatever neither recorded, exactly as a bare
    # re-run does, so the two front ends cannot disagree about a run's settings.
    app_opts = src.get("app_opts") or {}
    recorded = {**app_opts, **{k: v for k, v in src.items() if v is not None}}
    cli.resolve_settings(args, {"source": recorded})
    if recorded.get("device") is not None:   # not a recorded setting; the app's alone
        args.device = recorded["device"]
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise EditError(f"unknown option {key!r}")
        setattr(args, key, value)
    return args


def enrich(m: dict[str, Any], workdir: Path,
           uids: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Segments plus the verification verdict the UI wants and the manifest omits.

    `clips/<key>.json` is written by the tts stage next to every clip: what the
    verifier actually heard, and how much of it matched. It is the ready-made "did
    the voice say the right thing" signal, and reading it is the only reason the app
    would otherwise need to know how the clip cache is laid out.
    """
    import json

    want = set(uids) if uids is not None else None
    out = []
    for seg in m.get("segments") or []:
        if want is not None and seg.get("uid") not in want:
            continue
        row = dict(seg)
        clip = (seg.get("tts") or {}).get("clip")
        if clip:
            meta = (workdir / clip).with_suffix(".json")
            if meta.is_file():
                try:
                    row["verify"] = json.loads(meta.read_text(encoding="utf-8"))
                except Exception:
                    pass
        out.append(row)
    return out
