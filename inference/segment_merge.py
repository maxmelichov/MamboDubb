#!/usr/bin/env python3
"""Pause-aware utterance grouping for dubbing.

Rules:
  - Speaker change → new utterance
  - Silence / gap > max_pause → new utterance (VAD-like split)
  - Same speaker with a shorter pause → keep one utterance, record the pause
    so TTS can insert silence instead of starting a new sentence
"""

from __future__ import annotations

import copy
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_MAX_PAUSE = 1.0  # seconds — above this, start a new utterance
DEFAULT_PHRASE_PAUSE = 0.18  # seconds — within an utterance, mark an internal pause
# Attach a short continuation clause across a diarization flip (unfinished → next).
DEFAULT_STITCH_GAP = 0.55
DEFAULT_SAME_SPEAKER_STITCH_GAP = 3.5
DEFAULT_SHORT_COMPLETION_GAP = 2.5
DEFAULT_GAP_RECOVERY_MIN = 1.5
MIN_PHRASE_SEC = 1.15
MIN_PHRASE_WORDS = 4
SHORT_COMPLETION_MAX_WORDS = 4
SHORT_COMPLETION_MAX_SEC = 1.5
SPEAKER_STABILIZE_MAX_GAP = 0.6
# Don't glue a finished sentence onto a new speech-act across this gap.
POST_SENTENCE_MERGE_GAP = 0.80
# Long HE clauses: split on sentence end so TTS can pause between thoughts.
LONG_PHRASE_SPLIT_SEC = 4.5
SENTENCE_PAUSE_SEC = 0.55
# Elastic dub placement (never-cut packer).
MAX_DRIFT_SEC = 0.6
RUN_BREAK_GAP_SEC = 1.5
INTER_GAP_SEC = 0.12
PLACE_KEEP_GUARD_SEC = 0.12
PLACE_MAX_SPEED = 1.50
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…؟])\s+")
# Parliamentary thanks / address — new speech-act, not monologue continuation.
_NEW_SPEECH_ACT = re.compile(
    r"^(תודה|תודות|אדוני|גברתי|כבוד|מר |גב'|שלום|"
    r"thank you|thanks|mr\.|mrs\.|madam|mister)\b",
    re.IGNORECASE,
)


def _ends_sentence_final(text: str) -> bool:
    t = (text or "").strip().rstrip("\"'”’")
    return bool(t) and t[-1] in ".!?؟…"


def looks_like_new_speech_act(text: str) -> bool:
    """True for thanks/address/short standalone that shouldn't glue onto a monologue."""
    t = (text or "").strip()
    if not t:
        return False
    if _NEW_SPEECH_ACT.search(t):
        return True
    # Very short standalone after a pause (e.g. "אין מענה.") is a new act.
    n_words = len(t.split())
    return n_words <= 4 and _ends_sentence_final(t)


def _join_text_idempotent(left: str, right: str) -> str:
    """Join two text chunks without duplicating an already-present suffix/prefix."""
    a = (left or "").strip()
    b = (right or "").strip()
    if not b:
        return a
    if not a:
        return b
    # Already merged (re-run safety): b is suffix of a, or a ends with b.
    if a.endswith(b) or b in a:
        return a
    if b.startswith(a) and len(b) > len(a):
        return b
    return f"{a} {b}".strip()


# Do not absorb a tiny phrase across a real mid-utterance silence.
HARD_PAUSE_SEC = 0.80


def merge_short_phrases(
    phrases: list[dict[str, Any]],
    *,
    min_sec: float = MIN_PHRASE_SEC,
    min_words: int = MIN_PHRASE_WORDS,
    hard_pause: float = HARD_PAUSE_SEC,
) -> list[dict[str, Any]]:
    """Merge tiny pause-split stubs so translate+TTS see real clauses.

    Idempotent: re-running on already-merged phrases does not duplicate text_en.
    Never merges across a hard pause (>= hard_pause) — those are alignment anchors.
    """
    if len(phrases) <= 1:
        return phrases
    out: list[dict[str, Any]] = []
    for p in phrases:
        cur: dict[str, Any] = {
            "text": (p.get("text") or "").strip(),
            "start": float(p["start"]),
            "end": float(p["end"]),
            "pause_after": float(p.get("pause_after") or 0.0),
        }
        if "text_en" in p:
            cur["text_en"] = (p.get("text_en") or "").strip()
        if "speaker_id" in p:
            cur["speaker_id"] = p["speaker_id"]
        for key in ("tts_fit", "tts_raw", "tts_speed_used", "tts_text"):
            if key in p:
                cur[key] = p[key]
        dur = cur["end"] - cur["start"]
        n_words = len(cur["text"].split())
        tiny = (dur < min_sec and n_words <= min_words) or n_words <= 2 or dur < 0.55
        if out and tiny:
            prev = out[-1]
            gap = float(cur["start"]) - float(prev["end"])
            prev_pause = float(prev.get("pause_after") or 0.0)
            # Preserve real silences as phrase boundaries.
            if max(gap, prev_pause) >= hard_pause:
                out.append(cur)
                continue
            # Preserve sentence-final interjections —
            # do not fold a new sentence into the previous one just because
            # it is short. Exception: object-continuation stubs
            # ("אני לא מכירה." + "אף אישה…") must stay one TTS/MT unit.
            prev_txt = (prev.get("text") or "").rstrip()
            cur_txt = (cur.get("text") or "").rstrip()
            object_cont = needs_object_continuation(prev_txt, cur_txt)
            if (
                not object_cont
                and prev_txt.endswith((".", "!", "?", "…", "؟"))
                and cur_txt.endswith((".", "!", "?", "…", "؟"))
            ):
                out.append(cur)
                continue
            prev["text"] = _join_text_idempotent(prev["text"], cur["text"])
            if cur.get("text_en") or prev.get("text_en"):
                prev["text_en"] = dedupe_repeated_sentences(
                    _join_text_idempotent(
                        prev.get("text_en") or "", cur.get("text_en") or ""
                    )
                )
            prev["end"] = max(float(prev["end"]), float(cur["end"]))
            prev["pause_after"] = cur["pause_after"]
            continue
        out.append(cur)
    # Second pass: merge object-continuation pairs even when neither is "tiny"
    # (e.g. short stub + longer object clause both punctuated).
    out = _merge_object_continuation_phrases(out, hard_pause=hard_pause)
    if len(out) >= 2:
        first = out[0]
        gap0 = float(out[1]["start"]) - float(first["end"])
        pause0 = float(first.get("pause_after") or 0.0)
        if (
            first["end"] - first["start"] < min_sec
            and len(first["text"].split()) <= min_words
            and max(gap0, pause0) < hard_pause
        ):
            nxt = out[1]
            nxt["text"] = _join_text_idempotent(first["text"], nxt["text"])
            if first.get("text_en") or nxt.get("text_en"):
                nxt["text_en"] = dedupe_repeated_sentences(
                    _join_text_idempotent(
                        first.get("text_en") or "", nxt.get("text_en") or ""
                    )
                )
            nxt["start"] = first["start"]
            out = [nxt, *out[2:]]
    if out:
        out[-1]["pause_after"] = 0.0
    return out


def _merge_object_continuation_phrases(
    phrases: list[dict[str, Any]],
    *,
    hard_pause: float = HARD_PAUSE_SEC,
) -> list[dict[str, Any]]:
    """Join object-seeking stubs with their object clauses inside one utterance.

    General pattern: "אני לא מכירה." + "אף אישה במזרח התיכון…" must become
    one phrase so MT/TTS cannot misalign "I don't know" onto the wrong clause.

    Semantic continuations override pause_after when the next phrase starts
    immediately after cur.end (gap small) — long pause_after metadata alone
    must not block the merge.
    """
    if len(phrases) <= 1:
        return phrases
    # Object continuations may sit after a brief breath; allow up to ~2s gap.
    cont_gap_max = max(float(hard_pause), 2.0)
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(phrases):
        cur = dict(phrases[i])
        if i + 1 < len(phrases):
            nxt = phrases[i + 1]
            gap = float(nxt["start"]) - float(cur["end"])
            if (
                gap < cont_gap_max
                and needs_object_continuation(
                    (cur.get("text") or "").strip(),
                    (nxt.get("text") or "").strip(),
                )
            ):
                cur["text"] = _join_text_idempotent(
                    cur.get("text") or "", nxt.get("text") or ""
                )
                if cur.get("text_en") or nxt.get("text_en"):
                    cur["text_en"] = dedupe_repeated_sentences(
                        _join_text_idempotent(
                            cur.get("text_en") or "", nxt.get("text_en") or ""
                        )
                    )
                cur["end"] = max(float(cur["end"]), float(nxt["end"]))
                cur["pause_after"] = float(nxt.get("pause_after") or 0.0)
                for key in ("tts_fit", "tts_raw", "tts_speed_used", "tts_text"):
                    cur.pop(key, None)
                out.append(cur)
                i += 2
                continue
        out.append(cur)
        i += 1
    return out


def stamp_source_timing(segments: list[dict[str, Any]]) -> int:
    """Stamp immutable ASR windows once; later TTS may not overwrite them.

    Sets segment- and phrase-level ``source_start`` / ``source_end`` from the
    current ``start`` / ``end`` when absent. Returns how many segments stamped.
    """
    n = 0
    for seg in segments:
        if "source_start" not in seg:
            seg["source_start"] = float(seg.get("start") or 0.0)
            seg["source_end"] = float(seg.get("end") or 0.0)
            n += 1
        for p in seg.get("phrases") or []:
            if "source_start" not in p:
                p["source_start"] = float(p.get("start") or 0.0)
                p["source_end"] = float(p.get("end") or 0.0)
    return n


def _phrase_text_key(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())[:48]


def restore_asr_phrase_boundaries(
    segments: list[dict[str, Any]],
    asr_segments: list[dict[str, Any]],
) -> int:
    """Re-apply ASR phrase start/end/text from segments.json.

    TTS unit writeback can bloat short stubs into multi-second windows and drop
    trailing phrases. ASR timings remain the authority for HE windows.
    Also re-anchors KEEP/original segment start/end to the matching ASR row so
    energy snap/extend from a prior run cannot permanently shift the timeline.
    """
    if not asr_segments:
        return 0
    n = 0
    for seg in segments:
        st = float(seg.get("source_start") or seg.get("start") or 0.0)
        en = float(seg.get("source_end") or seg.get("end") or 0.0)
        best = None
        best_ov = 0.0
        for src in asr_segments:
            a = float(src.get("start") or 0.0)
            b = float(src.get("end") or 0.0)
            ov = max(0.0, min(en, b) - max(st, a))
            if ov > best_ov:
                best_ov = ov
                best = src
        if best is None or best_ov < 0.5:
            continue

        keep = bool(seg.get("keep_original") or (seg.get("language") or "he") != "he")
        asr_a = float(best.get("start") or st)
        asr_b = float(best.get("end") or en)
        # Always re-anchor the segment window to ASR (placement may still
        # soft-snap for KEEP, but source_* / mix authority stays ASR).
        if abs(asr_a - st) > 0.08 or abs(asr_b - en) > 0.08:
            seg["start"] = asr_a
            seg["end"] = asr_b
            seg["source_start"] = asr_a
            seg["source_end"] = asr_b
            seg["duration"] = asr_b - asr_a
            n += 1

        if keep:
            continue

        src_phrases = best.get("phrases") or []
        if len(src_phrases) < 2:
            continue
        # Rebuild phrases from ASR; keep utterance-level text_en for redistribute.
        new_phrases: list[dict[str, Any]] = []
        for sp in src_phrases:
            row = {
                "text": (sp.get("text") or "").strip(),
                "start": float(sp["start"]),
                "end": float(sp["end"]),
                "pause_after": float(sp.get("pause_after") or 0.0),
                "source_start": float(sp["start"]),
                "source_end": float(sp["end"]),
                "speaker_id": seg.get("speaker_id"),
            }
            new_phrases.append(row)
        old = seg.get("phrases") or []
        changed = len(old) != len(new_phrases) or any(
            abs(float(o.get("start") or 0) - float(n_.get("start") or 0)) > 0.15
            or abs(float(o.get("end") or 0) - float(n_.get("end") or 0)) > 0.15
            or _phrase_text_key(o.get("text") or "") != _phrase_text_key(n_.get("text") or "")
            for o, n_ in zip(old, new_phrases)
        )
        if not changed and len(old) == len(new_phrases):
            continue
        # Preserve ASR HE text at segment level too when truncated.
        src_text = (best.get("text") or "").strip()
        if src_text and len(src_text) > len((seg.get("text") or "").strip()) + 10:
            seg["text"] = src_text
        seg["phrases"] = new_phrases
        # Drop stale TTS paths so units regenerate on restored windows.
        for key in ("tts_fit", "tts_raw", "tts_speed_used", "tts_text", "spoken_end", "tts_start"):
            seg.pop(key, None)
        n += 1
    return n


def rehydrate_missing_asr_segments(
    segments: list[dict[str, Any]],
    asr_segments: list[dict[str, Any]],
    *,
    min_hole_sec: float = 1.5,
) -> int:
    """Re-insert ASR turns that fell out of translated JSON (e.g. empty KEEP text)."""
    if not asr_segments:
        return 0

    def _covered(t0: float, t1: float) -> bool:
        for s in segments:
            a = float(s.get("source_start") or s.get("start") or 0.0)
            b = float(s.get("source_end") or s.get("end") or 0.0)
            if min(t1, b) - max(t0, a) >= 0.6 * (t1 - t0):
                return True
        return False

    added = 0
    for src in asr_segments:
        a = float(src.get("start") or 0.0)
        b = float(src.get("end") or 0.0)
        if b - a < min_hole_sec:
            continue
        if _covered(a, b):
            continue
        row = copy.deepcopy(src)
        # Ensure KEEP rows survive empty-caption filtering.
        if row.get("keep_original") or (row.get("language") or "he") != "he":
            row["keep_original"] = True
            row["language"] = row.get("language") or "en"
            if not (row.get("text") or "").strip():
                row["text"] = "[original audio]"
            if not (row.get("text_en") or "").strip():
                row["text_en"] = row["text"]
        row["source_start"] = a
        row["source_end"] = b
        segments.append(row)
        added += 1
        print(
            f"  rehydrate missing ASR [{row.get('speaker_id')}] "
            f"{a:.1f}-{b:.1f}s | {(row.get('text') or '')[:50]}",
            file=sys.stderr,
        )
    if added:
        segments.sort(
            key=lambda s: float(s.get("source_start") or s.get("start") or 0.0)
        )
    return added


def source_start(obj: dict[str, Any]) -> float:
    """ASR onset — prefers immutable source_start when present."""
    if obj.get("source_start") is not None:
        return float(obj["source_start"])
    return float(obj.get("start") or 0.0)


def source_end(obj: dict[str, Any]) -> float:
    """ASR offset — prefers immutable source_end when present."""
    if obj.get("source_end") is not None:
        return float(obj["source_end"])
    return float(obj.get("end") or 0.0)


_SENTENCE_SPLIT_EN = re.compile(r"(?<=[.!?…])\s+")


def dedupe_repeated_sentences(text: str) -> str:
    """Collapse consecutive duplicate sentences / clause runs.

    Fixes corrupted text_en like:
      '…with great artistry. it with great artistry. it with great artistry.'
    """
    t = re.sub(r"\s+", " ", (text or "").strip())
    if not t:
        return t

    def _norm_sent(s: str) -> str:
        s = s.lower().rstrip(".!?…").strip()
        s = re.sub(r"^(it|and|the|a|an|this|that)\s+", "", s)
        return s.strip()

    parts = [p.strip() for p in _SENTENCE_SPLIT_EN.split(t) if p.strip()]
    if len(parts) >= 2:
        out: list[str] = []
        for part in parts:
            norm = _norm_sent(part)
            if not norm:
                continue
            if any(_norm_sent(o) == norm for o in out[-4:]):
                continue
            # Later fragment is a short suffix of an earlier sentence
            # ("…with great artistry" then "it with great artistry").
            if len(norm.split()) >= 3 and any(
                _norm_sent(o).endswith(norm) for o in out
            ):
                continue
            out.append(part)
        rebuilt = " ".join(out).strip()
        if rebuilt:
            t = rebuilt
    # Phrase-level: collapse "X X X" where X is 4+ words repeated.
    words = t.split()
    for n in range(min(24, len(words) // 2), 3, -1):
        changed = True
        while changed and len(words) >= 2 * n:
            changed = False
            i = 0
            while i + 2 * n <= len(words):
                a = [w.lower().strip(".,;:!?\"'") for w in words[i : i + n]]
                b = [w.lower().strip(".,;:!?\"'") for w in words[i + n : i + 2 * n]]
                if a == b:
                    del words[i + n : i + 2 * n]
                    changed = True
                else:
                    i += 1
    return " ".join(words).strip()


def dedupe_segment_text_fields(seg: dict[str, Any]) -> None:
    """In-place: dedupe text / text_en on a segment and its phrases."""
    if (seg.get("text_en") or "").strip():
        seg["text_en"] = dedupe_repeated_sentences(seg["text_en"])
    if (seg.get("text") or "").strip() and not re.search(
        r"[\u0590-\u05FF]", seg.get("text") or ""
    ):
        # Only dedupe Latin KEEP transcripts (Hebrew ASR rarely self-repeats this way).
        seg["text"] = dedupe_repeated_sentences(seg["text"])
    for p in seg.get("phrases") or []:
        if (p.get("text_en") or "").strip():
            p["text_en"] = dedupe_repeated_sentences(p["text_en"])
        he = p.get("text") or ""
        if he.strip() and not re.search(r"[\u0590-\u05FF]", he):
            p["text"] = dedupe_repeated_sentences(he)


def clamp_phrase_timeline(
    phrases: list[dict[str, Any]],
    *,
    min_gap: float = 0.0,
) -> list[dict[str, Any]]:
    """Make phrase times monotonic; fold fully-overlapped phrases into the previous.

    Prevents stitch artifacts where a short completion sits inside a longer
    previous phrase window (which then causes hard-chop / skipped audio).
    """
    if not phrases:
        return []
    ordered = sorted(
        (dict(p) for p in phrases),
        key=lambda p: (float(p["start"]), float(p["end"])),
    )
    out: list[dict[str, Any]] = []
    for p in ordered:
        cur = dict(p)
        cur["start"] = float(cur["start"])
        cur["end"] = float(cur["end"])
        if cur["end"] <= cur["start"]:
            cur["end"] = cur["start"] + 0.25
        if not out:
            out.append(cur)
            continue
        prev = out[-1]
        # Fully inside previous → absorb text into prev, drop cur.
        if cur["start"] >= prev["start"] - 0.05 and cur["end"] <= prev["end"] + 0.05:
            prev["text"] = _join_text_idempotent(prev.get("text") or "", cur.get("text") or "")
            if cur.get("text_en") or prev.get("text_en"):
                prev["text_en"] = _join_text_idempotent(
                    prev.get("text_en") or "", cur.get("text_en") or ""
                )
            prev["end"] = max(prev["end"], cur["end"])
            prev["pause_after"] = float(cur.get("pause_after") or 0.0)
            continue
        # Partial overlap / out-of-order: clamp start to prev.end (+ min_gap).
        floor = prev["end"] + min_gap
        if cur["start"] < floor:
            cur["start"] = floor
        if cur["end"] <= cur["start"]:
            cur["end"] = cur["start"] + 0.25
        # Refresh pause_after on previous.
        prev["pause_after"] = round(max(0.0, cur["start"] - prev["end"]), 3)
        out.append(cur)
    if out:
        out[-1]["pause_after"] = 0.0
    return out


def clamp_segment_phrases(seg: dict[str, Any]) -> None:
    """In-place monotonic clamp + field refresh for one segment."""
    phrases = seg.get("phrases") or []
    if not phrases:
        return
    clamped = clamp_phrase_timeline(phrases)
    seg["phrases"] = clamped
    _rebuild_segment_text(seg)


def clamp_segment_timeline(
    segments: list[dict[str, Any]],
    *,
    min_gap: float = 0.02,
) -> int:
    """Make segment start/end monotonic so later turns don't overlap earlier ones.

    Diarization often emits overlapping turns (A ends 72.4, B starts 71.9).
    Without a clamp, TTS uses B's start as hard_end for A and mid-word-cuts A.
    Splits the overlap at the midpoint (or prev.end + min_gap). Returns #clamped.
    """
    if len(segments) < 2:
        return 0
    ordered = sorted(
        range(len(segments)),
        key=lambda i: (
            float(segments[i].get("start") or 0.0),
            float(segments[i].get("end") or 0.0),
        ),
    )
    changed = 0
    prev_end = None
    prev_idx = None
    for idx in ordered:
        seg = segments[idx]
        s = float(seg.get("start") or 0.0)
        e = float(seg.get("end") or s)
        if e <= s:
            e = s + 0.25
        if prev_end is not None and s < prev_end - 1e-3:
            overlap = prev_end - s
            if 0 < overlap <= 1.5 and prev_idx is not None:
                mid = round((s + prev_end) / 2.0, 3)
                other = segments[prev_idx]
                other["end"] = mid
                other["duration"] = round(
                    float(other["end"]) - float(other["start"]), 3
                )
                phrases = other.get("phrases") or []
                if phrases:
                    phrases[-1]["end"] = mid
                    phrases[-1]["pause_after"] = 0.0
                new_start = mid
            else:
                new_start = round(prev_end + min_gap, 3)
            if new_start >= e:
                e = new_start + 0.25
            print(
                f"  clamp overlap [{seg.get('speaker_id')}] "
                f"{s:.2f} → start {new_start:.2f}s",
                file=sys.stderr,
            )
            seg["start"] = new_start
            s = new_start
            phrases = seg.get("phrases") or []
            if phrases:
                phrases[0]["start"] = new_start
                if float(phrases[0]["end"]) <= new_start:
                    phrases[0]["end"] = new_start + 0.25
            changed += 1
        seg["end"] = round(e, 3)
        seg["start"] = round(s, 3)
        seg["duration"] = round(float(seg["end"]) - float(seg["start"]), 3)
        prev_end = float(seg["end"])
        prev_idx = idx
    return changed


_UNFINISHED_TAIL = re.compile(
    r"(מממן|מממנת|שפועל|שפועלת|funds|fund|operates|and|the|את|של|עם|ו|"
    r"אותך|אותה|אותו|אותם|אותן|"
    r"צובעת|צובע|צובעים|עושה|מקדמת|מקדם|מצליחה|מצליח)\.?$",
    re.IGNORECASE,
)
_OPEN_ADVERBIAL = re.compile(
    r"(באופן|בצורה|כמו|לקראת|בגלל|אחרי|לפני|עד|בתוך|בלי|עלי|על)\.?$",
    re.IGNORECASE,
)
_CLAUSE_END = re.compile(r"[.!?؟]\s+")
# Known transitive / painting verbs that need an object (את …) on the next turn.
_OBJECT_SEEKING_STUB = re.compile(
    r"^(היא |הוא |הם |הן |אני )?(צובעת|צובע|עושה|מקדמת|מקדם|מצליחה|מצליח|"
    r"מממנת|מממן|קושרת|קושר|מכירה|מכיר|יודעת|יודע)\.?$",
    re.IGNORECASE,
)
_OBJECT_CONT_NEXT = re.compile(r"^(את |אף |שום |ו[\u0590-\u05FF])")
# Generic short subject + one content word (verb-like) — used with object next only.
_SUBJECT_VERB_SHORT = re.compile(
    r"^(היא|הוא|הם|הן|אני)\s+[\u0590-\u05FF]{2,14}\.?$"
)
_ANI_LO_STUB = re.compile(r"^אני לא [\u0590-\u05FF]{2,14}\.?$")


def utterance_unfinished(text: str) -> bool:
    """True when the transcript looks mid-sentence / mid-list / open adverbial."""
    t = (text or "").strip().rstrip("\"'”’")
    if not t:
        return False
    if t.endswith((",", "،", ";", ":", "—", "-")):
        return True
    if t.endswith("…") or t.endswith("..."):
        return True
    if _OPEN_ADVERBIAL.search(t):
        return True
    if _UNFINISHED_TAIL.search(t):
        return True
    if _OBJECT_SEEKING_STUB.match(t) or _SUBJECT_VERB_SHORT.match(t):
        return True
    if _ANI_LO_STUB.match(t):
        return True
    if t[-1] not in ".!?؟":
        return True
    return False


def needs_object_continuation(text: str, next_text: str) -> bool:
    """True when a short subject+verb / 'אני לא X' stub is completed by next.

    Shared by extract stitch and preview absorb. Next must start with an
    object/negation continuation (את / אף / שום / ו…).
    """
    t = (text or "").strip()
    nxt = (next_text or "").lstrip()
    if not t or not nxt:
        return False
    if not _OBJECT_CONT_NEXT.match(nxt):
        return False
    if _OBJECT_SEEKING_STUB.match(t) or _SUBJECT_VERB_SHORT.match(t):
        return True
    if _ANI_LO_STUB.match(t):
        return True
    # Tiny 1–3 word stub ending in a verb-like token + object next.
    words = t.rstrip(".!?").split()
    if 1 <= len(words) <= 3 and utterance_unfinished(t):
        return True
    return False


def is_short_completion(seg: dict[str, Any]) -> bool:
    """True for tiny follow-ons that finish a prior open clause (e.g. אמנות רבה)."""
    text = (seg.get("text") or "").strip()
    if not text:
        return False
    n_words = len(text.split())
    dur = float(seg.get("end") or 0) - float(seg.get("start") or 0)
    return n_words <= SHORT_COMPLETION_MAX_WORDS or dur <= SHORT_COMPLETION_MAX_SEC


def is_continuation_start(text: str) -> bool:
    """Hebrew continuation markers (vav-conjunctive, etc.)."""
    t = (text or "").lstrip()
    if not t:
        return False
    if t[0] == "ו":
        return True
    return False


def _split_first_clause(text: str) -> tuple[str, str]:
    """Split leading clause (through first .!? ) from the remainder."""
    t = (text or "").strip()
    if not t:
        return "", ""
    m = _CLAUSE_END.search(t)
    if not m:
        return t, ""
    return t[: m.end()].strip(), t[m.end() :].strip()


def _rebuild_segment_text(seg: dict[str, Any]) -> None:
    phrases = seg.get("phrases") or []
    seg["text"] = " ".join(
        (p.get("text") or "").strip() for p in phrases if (p.get("text") or "").strip()
    )
    en_bits = [
        (p.get("text_en") or "").strip()
        for p in phrases
        if (p.get("text_en") or "").strip()
    ]
    if en_bits:
        seg["text_en"] = " ".join(en_bits)
    # Keep existing utterance-level text_en when phrases lack EN (e.g. mid sentence-split).
    # Never wipe a good translation here — redistribute happens in build_preview.
    if phrases:
        seg["start"] = float(phrases[0]["start"])
        seg["end"] = float(phrases[-1]["end"])
    seg["duration"] = round(float(seg["end"]) - float(seg["start"]), 3)
    seg["pauses"] = [float(p.get("pause_after") or 0.0) for p in phrases]
    for key in ("tts_fit", "tts_raw", "tts_speed_used"):
        seg.pop(key, None)


# Public alias used by build_preview / sentence splits.
refresh_segment_fields = _rebuild_segment_text


def _absorb_segment_into(
    cur: dict[str, Any],
    nxt: dict[str, Any],
    *,
    speaker_id: str | None = None,
) -> dict[str, Any]:
    """Append nxt phrases onto cur (same utterance), keeping internal pause."""
    spk = speaker_id or cur.get("speaker_id")
    cur_phrases = [dict(p) for p in (cur.get("phrases") or [])]
    nxt_phrases = [dict(p) for p in (nxt.get("phrases") or [])]
    if not nxt_phrases and (nxt.get("text") or "").strip():
        nxt_phrases = [
            {
                "text": (nxt.get("text") or "").strip(),
                "start": float(nxt["start"]),
                "end": float(nxt["end"]),
                "pause_after": 0.0,
            }
        ]
    if cur_phrases and nxt_phrases:
        gap_pause = round(
            max(0.0, float(nxt_phrases[0]["start"]) - float(cur_phrases[-1]["end"])),
            3,
        )
        cur_phrases[-1]["pause_after"] = gap_pause
    for p in nxt_phrases:
        p["speaker_id"] = spk
        cur_phrases.append(p)
    if cur_phrases:
        cur_phrases[-1]["pause_after"] = 0.0
    cur["phrases"] = clamp_phrase_timeline(cur_phrases)
    cur["speaker_id"] = spk
    _rebuild_segment_text(cur)
    dedupe_segment_text_fields(cur)
    return cur


def stitch_unfinished_continuations(
    segments: list[dict[str, Any]],
    *,
    max_gap: float = DEFAULT_STITCH_GAP,
    same_speaker_max_gap: float = DEFAULT_SAME_SPEAKER_STITCH_GAP,
    short_completion_gap: float = DEFAULT_SHORT_COMPLETION_GAP,
) -> list[dict[str, Any]]:
    """Attach unfinished tails to their completions.

    - Same speaker: allow longer pauses (documentary mid-breath).
    - Cross speaker: keep tight, but absorb *short* completions onto A
      (e.g. `באופן.` + `אמנות רבה.`).
    - Longer next segments: move only the first clause onto A when possible.
    """
    if len(segments) < 2:
        return segments

    out: list[dict[str, Any]] = []
    i = 0
    while i < len(segments):
        cur = dict(segments[i])
        cur["phrases"] = [dict(p) for p in (cur.get("phrases") or [])]
        if i + 1 >= len(segments):
            out.append(cur)
            break

        nxt = dict(segments[i + 1])
        nxt["phrases"] = [dict(p) for p in (nxt.get("phrases") or [])]
        gap = float(nxt.get("start") or 0) - float(cur.get("end") or 0)
        same_lang = (cur.get("language") or "he") == (nxt.get("language") or "he")
        same_keep = bool(cur.get("keep_original")) == bool(nxt.get("keep_original"))
        same_spk = str(cur.get("speaker_id")) == str(nxt.get("speaker_id"))
        cur_tail = (cur.get("text") or "").strip()
        if cur.get("phrases"):
            cur_tail = (cur["phrases"][-1].get("text") or cur_tail).strip()
        nxt_text = (nxt.get("text") or "").strip()
        unfinished = utterance_unfinished(cur_tail)
        object_cont = needs_object_continuation(cur_tail, nxt_text)
        short_b = is_short_completion(nxt)

        if same_spk:
            allowed_gap = same_speaker_max_gap
        elif short_b and (unfinished or object_cont):
            allowed_gap = short_completion_gap
        elif object_cont:
            allowed_gap = short_completion_gap
        else:
            allowed_gap = max_gap

        if not (
            same_lang
            and same_keep
            and not cur.get("keep_original")
            and 0 <= gap <= allowed_gap
            and (unfinished or object_cont)
            and (nxt.get("phrases") or (nxt.get("text") or "").strip())
        ):
            out.append(cur)
            i += 1
            continue

        # Object-continuation stubs (היא צובעת + את…): absorb even cross-speaker.
        if object_cont or short_b or (same_spk and unfinished and is_short_completion(nxt)):
            cur = _absorb_segment_into(cur, nxt, speaker_id=str(cur.get("speaker_id")))
            out.append(cur)
            i += 2
            continue

        # Same-speaker unfinished across a longer pause: absorb whole next if it
        # looks like a continuation (vav / no strong new-sentence marker).
        if same_spk and unfinished and (
            is_continuation_start(nxt_text) or not nxt_text[:1].isupper()
        ):
            # Hebrew has no case — treat any same-speaker unfinished+gap as absorb
            # when the next segment is not much longer than a clause.
            nxt_dur = float(nxt.get("end") or 0) - float(nxt.get("start") or 0)
            if nxt_dur <= 8.0:
                cur = _absorb_segment_into(cur, nxt, speaker_id=str(cur.get("speaker_id")))
                out.append(cur)
                i += 2
                continue

        first = dict((nxt.get("phrases") or [{}])[0]) if nxt.get("phrases") else {
            "text": nxt_text,
            "start": float(nxt["start"]),
            "end": float(nxt["end"]),
            "pause_after": 0.0,
        }
        he_head, he_rest = _split_first_clause(first.get("text") or "")
        en_src = (first.get("text_en") or "").strip()
        en_head, en_rest = _split_first_clause(en_src) if en_src else ("", "")

        # Cross-speaker unfinished: first phrase of B is itself one completing
        # clause (no further clause in that phrase). Move it onto A; keep the
        # rest of B. Fixes "שפועל" + "בעולם היא מממנת אותו." cut/skip.
        first_text = (first.get("text") or "").strip()
        first_words = len(first_text.split())
        first_dur = float(first.get("end") or 0) - float(first.get("start") or 0)
        completing_clause = (
            unfinished
            and first_text
            and (not he_rest or he_head == first_text)
            and (
                first_words <= 12
                or first_dur <= 3.5
                or is_continuation_start(first_text)
            )
        )
        if completing_clause and not he_rest:
            # Whole first phrase completes A.
            moved = dict(first)
            moved["speaker_id"] = cur.get("speaker_id")
            if cur["phrases"]:
                gap_pause = round(
                    max(0.0, float(first["start"]) - float(cur["phrases"][-1]["end"])),
                    3,
                )
                cur["phrases"][-1]["pause_after"] = gap_pause
            cur["phrases"].append(moved)
            _rebuild_segment_text(cur)
            rest_phrases = list(nxt.get("phrases") or [])[1:]
            if rest_phrases:
                nxt["phrases"] = rest_phrases
                _rebuild_segment_text(nxt)
                segments[i + 1] = nxt
                clamp_segment_phrases(cur)
                dedupe_segment_text_fields(cur)
                out.append(cur)
                i += 1
                continue
            # B was only the completing clause — absorb fully.
            cur = _absorb_segment_into(cur, nxt, speaker_id=str(cur.get("speaker_id")))
            out.append(cur)
            i += 2
            continue

        if not he_head or not he_rest:
            # No clause boundary — if unfinished, still absorb short-ish next.
            nxt_dur = float(nxt.get("end") or 0) - float(nxt.get("start") or 0)
            if unfinished and (same_spk or nxt_dur <= SHORT_COMPLETION_MAX_SEC * 2):
                cur = _absorb_segment_into(cur, nxt, speaker_id=str(cur.get("speaker_id")))
                out.append(cur)
                i += 2
                continue
            out.append(cur)
            i += 1
            continue

        # Time-split the first next phrase by character share of the head clause.
        p0 = float(first["start"])
        p1 = float(first["end"])
        frac = max(0.15, min(0.85, len(he_head) / max(1, len((first.get("text") or "").strip()))))
        split_t = round(p0 + (p1 - p0) * frac, 3)
        split_t = min(split_t, p1 - 0.2)
        split_t = max(split_t, p0 + 0.25)

        moved = {
            "text": he_head,
            "start": p0,
            "end": split_t,
            "pause_after": 0.0,
            "speaker_id": cur.get("speaker_id"),
        }
        if en_head:
            moved["text_en"] = en_head
        elif en_src:
            moved["text_en"] = en_src.split(".")[0].strip() + ("." if "." in en_src else "")

        if cur["phrases"]:
            gap_pause = round(max(0.0, p0 - float(cur["phrases"][-1]["end"])), 3)
            cur["phrases"][-1]["pause_after"] = gap_pause
        cur["phrases"].append(moved)
        _rebuild_segment_text(cur)

        rest_phrase = {
            "text": he_rest,
            "start": split_t,
            "end": p1,
            "pause_after": float(first.get("pause_after") or 0.0),
            "speaker_id": nxt.get("speaker_id"),
        }
        if en_rest:
            rest_phrase["text_en"] = en_rest
        nxt["phrases"] = [rest_phrase] + list(nxt.get("phrases") or [])[1:]
        _rebuild_segment_text(nxt)

        segments[i + 1] = nxt
        clamp_segment_phrases(cur)
        dedupe_segment_text_fields(cur)
        out.append(cur)
        i += 1

    for seg in out:
        clamp_segment_phrases(seg)
        dedupe_segment_text_fields(seg)
    return out


def stabilize_speaker_continuity(
    segments: list[dict[str, Any]],
    *,
    max_gap: float = SPEAKER_STABILIZE_MAX_GAP,
) -> list[dict[str, Any]]:
    """Reassign speaker_id across short HE monologue flips (TTS voice identity).

    Only flips *short* continuations / stubs. Never absorbs a long independent
    turn into the previous speaker — that would glue different people into one
    TTS voice across a diarization boundary.
    """
    if len(segments) < 2:
        return segments
    out: list[dict[str, Any]] = []
    for seg in segments:
        row = dict(seg)
        row["phrases"] = [dict(p) for p in (row.get("phrases") or [])]
        if out:
            prev = out[-1]
            gap = float(row.get("start") or 0) - float(prev.get("end") or 0)
            prev_he = (prev.get("language") or "he") == "he" and not prev.get(
                "keep_original"
            )
            row_he = (row.get("language") or "he") == "he" and not row.get(
                "keep_original"
            )
            prev_tail = (prev.get("text") or "").strip()
            if prev.get("phrases"):
                prev_tail = (prev["phrases"][-1].get("text") or prev_tail).strip()
            row_text = (row.get("text") or "").strip()
            # Only short follow-ons may inherit the previous speaker.
            # Long turns (≥ ~3s or ≥ ~8 words) keep their own diarization ID.
            short_follow = is_short_completion(row)
            if (
                prev_he
                and row_he
                and short_follow
                and -0.15 <= gap <= max_gap
                and (
                    utterance_unfinished(prev_tail)
                    or is_continuation_start(row_text)
                )
            ):
                spk = str(prev.get("speaker_id"))
                row["speaker_id"] = spk
                for p in row["phrases"]:
                    p["speaker_id"] = spk
        out.append(row)
    return out


def find_uncovered_gaps(
    segments: list[dict[str, Any]],
    media_end: float,
    *,
    min_gap: float = DEFAULT_GAP_RECOVERY_MIN,
    media_start: float = 0.0,
) -> list[tuple[float, float]]:
    """Return [start, end) intervals with no segment coverage and duration ≥ min_gap."""
    ordered = sorted(
        (
            (float(s["start"]), float(s["end"]))
            for s in segments
            if float(s.get("end") or 0) > float(s.get("start") or 0)
        ),
        key=lambda x: x[0],
    )
    gaps: list[tuple[float, float]] = []
    cursor = float(media_start)
    end_limit = float(media_end)
    for a, b in ordered:
        if a > cursor + min_gap:
            gaps.append((cursor, min(a, end_limit)))
        cursor = max(cursor, b)
    if end_limit > cursor + min_gap:
        gaps.append((cursor, end_limit))
    return [(round(a, 3), round(b, 3)) for a, b in gaps if b - a >= min_gap]


def gap_has_speech_energy(
    vocals_path: Path | str,
    start: float,
    end: float,
    *,
    rms_thresh: float = 0.02,
    sample_rate: int = 16000,
    min_active_sec: float = 0.35,
) -> bool:
    """True when the window has sustained speech-like energy."""
    import numpy as np

    path = Path(vocals_path)
    dur = max(0.05, float(end) - float(start))
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{dur:.3f}",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode != 0 or not result.stdout:
        return False
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if audio.size < sample_rate // 10:
        return False
    hop = max(1, sample_rate // 10)
    active = 0.0
    for i in range(0, len(audio) - hop, hop):
        rms = float(np.sqrt(np.mean(audio[i : i + hop] ** 2) + 1e-12))
        if rms >= rms_thresh:
            active += hop / sample_rate
            if active >= min_active_sec:
                return True
    return False


def drop_silent_vocal_segments(
    segments: list[dict[str, Any]],
    vocals_path: Path | str,
    *,
    source_path: Path | str | None = None,
    rms_thresh: float = 0.015,
    min_active_sec: float = 0.25,
) -> list[dict[str, Any]]:
    """Drop ASR rows whose audio bed is near-silent (music hallucinations).

    Prefer the Demucs vocals stem. When `source_path` is provided, also probe
    the full mix — applause / crowd often masks speech in the vocals stem
    while the original still has audible speech. Only drop when BOTH are
    near-silent.
    """
    path = Path(vocals_path)
    if not path.is_file() or not segments:
        return segments
    source = Path(source_path) if source_path else None
    if source is not None and not source.is_file():
        source = None
    kept: list[dict[str, Any]] = []
    for seg in segments:
        # Never drop KEEP-original (EN/AR/…) — those must stay in the mix.
        if seg.get("keep_original") or (seg.get("language") or "he") != "he":
            kept.append(seg)
            continue
        a, b = float(seg["start"]), float(seg["end"])
        dur = max(0.05, b - a)
        # Short genuine thanks clips can't meet a fixed 0.25s active floor.
        need = min(min_active_sec, max(0.08, dur * 0.45))
        # Probe a little past ASR bounds — Whisper often clips onset/offset.
        pad = 0.45
        lo = max(0.0, a - pad)
        hi = b + pad
        if gap_has_speech_energy(
            path,
            lo,
            hi,
            rms_thresh=rms_thresh,
            min_active_sec=need,
        ):
            kept.append(seg)
            continue
        # Vocals stem silent — check full mix before dropping (applause mask).
        if source is not None and gap_has_speech_energy(
            source,
            lo,
            hi,
            rms_thresh=max(rms_thresh, 0.012),
            min_active_sec=need,
        ):
            print(
                f"  keep applause-masked [{seg.get('speaker_id')}] "
                f"{a:.1f}-{b:.1f}s (silent vocals, speech in source) | "
                f"{(seg.get('text') or '')[:50]}",
                file=sys.stderr,
            )
            kept.append(seg)
            continue
        text = (seg.get("text") or "")[:70]
        print(
            f"  drop silent-vocals [{seg.get('speaker_id')}] "
            f"{a:.1f}-{b:.1f}s | {text}",
            file=sys.stderr,
        )
    return kept


def split_long_phrases_at_periods(
    phrases: list[dict[str, Any]],
    *,
    min_dur: float = LONG_PHRASE_SPLIT_SEC,
    sentence_pause: float = SENTENCE_PAUSE_SEC,
) -> list[dict[str, Any]]:
    """Split long phrases on `.` / `!` / `?` and insert a longer pause between."""
    if not phrases:
        return []
    out: list[dict[str, Any]] = []
    for phrase in phrases:
        text = (phrase.get("text") or "").strip()
        start = float(phrase["start"])
        end = float(phrase["end"])
        dur = max(0.0, end - start)
        parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
        if (
            dur < min_dur
            or len(parts) < 2
            or not re.search(r"[.!?…؟]", text)
        ):
            out.append(dict(phrase))
            continue

        weights = [max(1, len(p.split())) for p in parts]
        total_w = float(sum(weights))
        n_gaps = len(parts) - 1
        pause = min(sentence_pause, max(0.28, (dur * 0.12) / max(1, n_gaps)))
        speak_budget = max(0.6, dur - pause * n_gaps)
        en = (phrase.get("text_en") or "").strip()
        en_parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(en) if p.strip()] if en else []
        if len(en_parts) != len(parts):
            en_parts = []

        t = start
        carry = {
            k: v
            for k, v in phrase.items()
            if k not in {"text", "start", "end", "pause_after", "text_en", "tts_fit", "tts_raw", "tts_speed_used"}
        }
        for i, part in enumerate(parts):
            slot = speak_budget * (weights[i] / total_w)
            pe = t + slot
            row = {
                **carry,
                "text": part,
                "start": round(t, 3),
                "end": round(pe if i < len(parts) - 1 else end, 3),
                "pause_after": round(
                    pause if i < n_gaps else float(phrase.get("pause_after") or 0.0),
                    3,
                ),
            }
            if en_parts:
                row["text_en"] = en_parts[i]
            out.append(row)
            t = pe + pause
    return out


def split_segments_at_sentence_ends(
    segments: list[dict[str, Any]],
    *,
    min_dur: float = LONG_PHRASE_SPLIT_SEC,
    sentence_pause: float = SENTENCE_PAUSE_SEC,
) -> int:
    """In-place: split long HE phrases on sentence punctuation. Returns #phrases added."""
    added = 0
    for seg in segments:
        if seg.get("keep_original") or (seg.get("language") or "he") != "he":
            continue
        saved_en = (seg.get("text_en") or "").strip()
        phrases = list(seg.get("phrases") or [])
        if not phrases and (seg.get("text") or "").strip():
            phrases = [
                {
                    "text": (seg.get("text") or "").strip(),
                    "start": float(seg["start"]),
                    "end": float(seg["end"]),
                    "pause_after": 0.0,
                    "text_en": (seg.get("text_en") or "").strip() or None,
                }
            ]
            if phrases[0]["text_en"] is None:
                phrases[0].pop("text_en", None)
        before = len(phrases)
        split = split_long_phrases_at_periods(
            phrases, min_dur=min_dur, sentence_pause=sentence_pause
        )
        if len(split) == before:
            continue
        seg["phrases"] = split
        refresh_segment_fields(seg)
        # Restore utterance EN if rebuild didn't reassemble it from phrases.
        if saved_en and not (seg.get("text_en") or "").strip():
            seg["text_en"] = saved_en
        # Clear partial EN so caller can redistribute the utterance translation.
        if any((p.get("text_en") or "").strip() for p in split) and any(
            not (p.get("text_en") or "").strip() for p in split
        ):
            for p in split:
                p.pop("text_en", None)
            if saved_en:
                seg["text_en"] = saved_en
        added += len(split) - before
    return added


def speaker_at_time(turns: list[dict[str, Any]], t: float) -> str | None:
    """Speaker whose turn covers time t (max overlap with a tiny window)."""
    best_id: str | None = None
    best_ov = 0.0
    # Point query with a 40ms window so boundaries are stable.
    a, b = t, t + 0.04
    for turn in turns:
        ts, te = float(turn["start"]), float(turn["end"])
        ov = max(0.0, min(b, te) - max(a, ts))
        if ov > best_ov:
            best_ov = ov
            best_id = str(turn["speaker_id"])
    if best_id is not None:
        return best_id
    # Fallback: nearest turn center.
    if not turns:
        return None
    nearest = min(
        turns,
        key=lambda turn: abs(((float(turn["start"]) + float(turn["end"])) / 2.0) - t),
    )
    return str(nearest["speaker_id"])


def tag_phrases_with_speakers(
    phrases: list[dict[str, Any]],
    turns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach speaker_id to each phrase from diarization (midpoint of phrase)."""
    out: list[dict[str, Any]] = []
    for p in phrases:
        row = dict(p)
        mid = (float(p["start"]) + float(p["end"])) / 2.0
        spk = speaker_at_time(turns, mid)
        if spk:
            row["speaker_id"] = spk
        out.append(row)
    return out


def split_segment_by_phrase_speaker(seg: dict[str, Any]) -> list[dict[str, Any]]:
    """Split one ASR segment into per-speaker utterances when phrases flip speaker."""
    phrases = seg.get("phrases") or []
    if not phrases:
        return [seg]
    if not any(p.get("speaker_id") for p in phrases):
        return [seg]

    groups: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    cur_spk: str | None = None
    for p in phrases:
        spk = str(p.get("speaker_id") or seg.get("speaker_id") or "SPEAKER_??")
        if cur and spk != cur_spk:
            groups.append(cur)
            cur = []
        cur.append(dict(p))
        cur_spk = spk
    if cur:
        groups.append(cur)

    if len(groups) <= 1:
        if phrases and phrases[0].get("speaker_id"):
            seg = dict(seg)
            seg["speaker_id"] = phrases[0]["speaker_id"]
        return [seg]

    split_rows: list[dict[str, Any]] = []
    for group in groups:
        row = {
            "speaker_id": group[0].get("speaker_id") or seg.get("speaker_id"),
            "language": seg.get("language"),
            "language_score": seg.get("language_score"),
            "he_score": seg.get("he_score"),
            "en_score": seg.get("en_score"),
            "ar_score": seg.get("ar_score"),
            "keep_original": seg.get("keep_original"),
            "start": round(float(group[0]["start"]), 3),
            "end": round(float(group[-1]["end"]), 3),
            "phrases": group,
        }
        if group:
            group[-1]["pause_after"] = 0.0
        row["text"] = " ".join(
            (p.get("text") or "").strip() for p in group if (p.get("text") or "").strip()
        )
        en_bits = [
            (p.get("text_en") or "").strip()
            for p in group
            if (p.get("text_en") or "").strip()
        ]
        if en_bits:
            row["text_en"] = " ".join(en_bits)
        row["duration"] = round(float(row["end"]) - float(row["start"]), 3)
        row["pauses"] = [float(p.get("pause_after") or 0.0) for p in group]
        split_rows.append(row)
    return split_rows


def _join_text(parts: list[str]) -> str:
    out: list[str] = []
    for part in parts:
        t = (part or "").strip()
        if not t:
            continue
        if out and not out[-1].endswith((" ", "\n")) and not t.startswith((" ", ",", ".", ";", ":", "—", "-")):
            # Hebrew/English words usually need a space; faster-whisper often includes it
            if not (out[-1].endswith((".", "!", "?", "،", "…"))):
                pass
        out.append(t)
    text = "".join(out).strip()
    if text and " " not in text and len(out) > 1:
        text = " ".join(p.strip() for p in out if p.strip())
    return text


def _phrases_from_segment(seg: dict[str, Any]) -> list[dict[str, Any]]:
    """Prefer existing phrase structure; never collapse multi-phrase rows to one blob."""
    existing = seg.get("phrases") or []
    if existing:
        out: list[dict[str, Any]] = []
        for p in existing:
            text = (p.get("text") or "").strip()
            text_en = (p.get("text_en") or "").strip() or None
            if not text and not text_en:
                continue
            phrase = {
                "text": text,
                "start": float(p["start"]),
                "end": float(p["end"]),
                "pause_after": float(p.get("pause_after") or 0.0),
            }
            if text_en:
                phrase["text_en"] = text_en
            if p.get("speaker_id"):
                phrase["speaker_id"] = p["speaker_id"]
            out.append(phrase)
        if out:
            return out

    text = (seg.get("text") or "").strip()
    text_en = (seg.get("text_en") or "").strip() or None
    if not text and not text_en:
        return []
    phrase = {
        "text": text,
        "start": float(seg["start"]),
        "end": float(seg["end"]),
        "pause_after": 0.0,
    }
    if text_en:
        phrase["text_en"] = text_en
    return [phrase]


def merge_same_speaker_segments(
    segments: list[dict[str, Any]],
    *,
    max_pause: float = DEFAULT_MAX_PAUSE,
) -> list[dict[str, Any]]:
    """Merge consecutive same-speaker, same-language segments across short pauses.

    Never merges different speakers — voice identity for TTS must stay distinct.
    Never merges across languages. Preserves existing phrases[] instead of flattening.
    """
    if not segments:
        return []

    ordered = sorted(segments, key=lambda s: float(s["start"]))
    merged: list[dict[str, Any]] = []

    cur = None
    for seg in ordered:
        phrases = _phrases_from_segment(seg)
        if not phrases:
            continue

        spk = seg["speaker_id"]
        lang = seg.get("language") or "he"
        keep_original = bool(seg.get("keep_original", lang != "he"))
        start = float(phrases[0]["start"])
        end = float(phrases[-1]["end"])

        def _new_cur() -> dict[str, Any]:
            row = {
                "speaker_id": spk,
                "language": lang,
                "keep_original": keep_original,
                "start": start,
                "end": end,
                "phrases": [dict(p) for p in phrases],
            }
            # Preserve language scores + already-rendered TTS across merge.
            for key in (
                "language_score",
                "he_score",
                "en_score",
                "ar_score",
                "tts_fit",
                "tts_raw",
                "tts_speed_used",
                "ref_audio",
                "ref_start",
                "ref_end",
                "text_en",
            ):
                if key in seg and seg[key] is not None:
                    row[key] = seg[key]
            return row

        if cur is None:
            cur = _new_cur()
            continue

        gap = start - float(cur["end"])
        same_lang = (
            lang == cur.get("language", "he")
            and keep_original == bool(cur.get("keep_original"))
        )
        same_spk = spk == cur["speaker_id"]

        # Refuse finished-sentence → new speech-act merges across a real pause
        # (e.g. interview monologue + parliamentary "thank you Mr. Chairman").
        refuse_merge = False
        if same_lang and same_spk and 0 <= gap <= max_pause:
            prev_tail = ""
            if cur.get("phrases"):
                prev_tail = (cur["phrases"][-1].get("text") or "").strip()
            next_head = (phrases[0].get("text") or "").strip() if phrases else ""
            if (
                gap >= POST_SENTENCE_MERGE_GAP
                and _ends_sentence_final(prev_tail)
                and looks_like_new_speech_act(next_head)
            ):
                refuse_merge = True

        if same_lang and same_spk and 0 <= gap <= max_pause and not refuse_merge:
            cur["phrases"][-1]["pause_after"] = round(max(gap, 0.0), 3)
            cur["phrases"].extend(dict(p) for p in phrases)
            cur["end"] = end
            for key in ("tts_fit", "tts_raw", "tts_speed_used"):
                cur.pop(key, None)
        else:
            _finalize_merged(cur)
            merged.append(cur)
            cur = _new_cur()

    if cur is not None:
        _finalize_merged(cur)
        merged.append(cur)

    return merged


def _finalize_merged(cur: dict[str, Any]) -> None:
    cur["phrases"] = clamp_phrase_timeline(cur.get("phrases") or [])
    if cur["phrases"]:
        cur["phrases"][-1]["pause_after"] = 0.0
        cur["start"] = float(cur["phrases"][0]["start"])
        cur["end"] = float(cur["phrases"][-1]["end"])
    cur["text"] = " ".join(
        p["text"].strip() for p in cur["phrases"] if (p.get("text") or "").strip()
    )
    en_bits = [
        (p.get("text_en") or "").strip()
        for p in cur["phrases"]
        if (p.get("text_en") or "").strip()
    ]
    if en_bits:
        cur["text_en"] = " ".join(en_bits)
    dedupe_segment_text_fields(cur)
    cur["duration"] = round(float(cur["end"]) - float(cur["start"]), 3)
    cur["pauses"] = [float(p.get("pause_after") or 0.0) for p in cur["phrases"]]


def merge_diarization_turns(
    turns: list[dict[str, Any]],
    *,
    max_pause: float = DEFAULT_MAX_PAUSE,
) -> list[dict[str, Any]]:
    """Merge adjacent *same-speaker* diarization turns across short pauses.

    Never merges different speakers — that collapsed distinct voices into one
    ASR/TTS segment (e.g. SPEAKER_02 then SPEAKER_01 within 0.3s).
    """
    if not turns:
        return []
    ordered = sorted(turns, key=lambda t: float(t["start"]))
    out: list[dict[str, Any]] = []
    cur = dict(ordered[0])
    for turn in ordered[1:]:
        gap = float(turn["start"]) - float(cur["end"])
        same_spk = turn["speaker_id"] == cur["speaker_id"]
        if same_spk and 0 <= gap <= max_pause:
            cur["end"] = float(turn["end"])
        else:
            out.append(
                {
                    "speaker_id": cur["speaker_id"],
                    "start": round(float(cur["start"]), 3),
                    "end": round(float(cur["end"]), 3),
                }
            )
            cur = dict(turn)
    out.append(
        {
            "speaker_id": cur["speaker_id"],
            "start": round(float(cur["start"]), 3),
            "end": round(float(cur["end"]), 3),
        }
    )
    return out


def words_to_utterances(
    words: list[Any],
    speaker_turns: list[dict[str, Any]],
    *,
    max_pause: float = DEFAULT_MAX_PAUSE,
    phrase_pause: float = DEFAULT_PHRASE_PAUSE,
) -> list[dict[str, Any]]:
    """Build utterances from word timestamps + diarization turns.

    A word inherits the speaker of the turn with maximum temporal overlap.
    """
    labeled: list[dict[str, Any]] = []
    for w in words:
        if w.start is None or w.end is None:
            continue
        token = (w.word or "").strip()
        if not token:
            continue
        # keep leading space info via original when joining
        raw = w.word or token
        best_spk = "SPEAKER_00"
        best_overlap = -1.0
        for turn in speaker_turns:
            overlap = min(float(w.end), float(turn["end"])) - max(float(w.start), float(turn["start"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_spk = turn["speaker_id"]
        if best_overlap <= 0:
            # word outside turns — still keep with nearest turn
            nearest = min(
                speaker_turns,
                key=lambda t: abs((float(t["start"]) + float(t["end"])) / 2 - float(w.start)),
                default=None,
            )
            if nearest:
                best_spk = nearest["speaker_id"]
        labeled.append(
            {
                "word": raw,
                "start": float(w.start),
                "end": float(w.end),
                "speaker_id": best_spk,
            }
        )

    if not labeled:
        return []

    labeled.sort(key=lambda x: x["start"])
    utterances: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    phrase_words: list[str] = []
    phrase_start = 0.0
    phrase_end = 0.0

    def flush_phrase(pause_after: float = 0.0) -> None:
        nonlocal phrase_words, phrase_start, phrase_end, cur
        if cur is None or not phrase_words:
            phrase_words = []
            return
        text = "".join(phrase_words).strip()
        if text and " " not in text and len(phrase_words) > 1:
            text = " ".join(w.strip() for w in phrase_words if w.strip())
        if text:
            cur["phrases"].append(
                {
                    "text": text,
                    "start": phrase_start,
                    "end": phrase_end,
                    "pause_after": round(pause_after, 3),
                }
            )
        phrase_words = []

    def flush_utterance() -> None:
        nonlocal cur
        flush_phrase(0.0)
        if cur is None:
            return
        if not cur["phrases"]:
            cur = None
            return
        cur["text"] = " ".join(p["text"] for p in cur["phrases"]).strip()
        # Fix last phrase pause
        if cur["phrases"]:
            cur["phrases"][-1]["pause_after"] = 0.0
        cur["end"] = float(cur["phrases"][-1]["end"])
        cur["duration"] = round(float(cur["end"]) - float(cur["start"]), 3)
        cur["pauses"] = [float(p["pause_after"]) for p in cur["phrases"]]
        utterances.append(cur)
        cur = None

    for i, w in enumerate(labeled):
        prev = labeled[i - 1] if i > 0 else None
        gap = (w["start"] - prev["end"]) if prev else 0.0
        speaker_change = prev is not None and w["speaker_id"] != prev["speaker_id"]
        new_utt = prev is None or speaker_change or gap > max_pause

        if new_utt:
            flush_utterance()
            cur = {
                "speaker_id": w["speaker_id"],
                "start": w["start"],
                "end": w["end"],
                "phrases": [],
            }
            phrase_words = [w["word"]]
            phrase_start = w["start"]
            phrase_end = w["end"]
            continue

        # Same utterance — maybe internal phrase pause
        assert cur is not None
        if gap >= phrase_pause:
            flush_phrase(pause_after=gap)
            phrase_words = [w["word"]]
            phrase_start = w["start"]
            phrase_end = w["end"]
        else:
            if not phrase_words:
                phrase_start = w["start"]
            phrase_words.append(w["word"])
            phrase_end = w["end"]
        cur["end"] = w["end"]

    flush_utterance()
    return utterances

def retag_english_sandwich(
    segments: list[dict[str, Any]],
    *,
    max_sec: float = 3.0,
) -> int:
    """Mark short dub segments sandwiched between same-speaker EN KEEP as keep_original.

    Catches English fragments mis-ASR'd as Hebrew (e.g. "אוקיי, על כך") between
    two English turns of the same speaker.
    """
    n = 0
    for i, seg in enumerate(segments):
        if i == 0 or i + 1 >= len(segments):
            continue
        lang = seg.get("language") or "he"
        keep = bool(seg.get("keep_original", lang != "he"))
        if keep or lang != "he":
            continue
        dur = float(seg.get("end", 0) - float(seg.get("start", 0)))
        if dur > max_sec:
            continue
        prev = segments[i - 1]
        nxt = segments[i + 1]
        prev_keep = bool(
            prev.get("keep_original") or (prev.get("language") or "he") != "he"
        )
        nxt_keep = bool(
            nxt.get("keep_original") or (nxt.get("language") or "he") != "he"
        )
        if not (prev_keep and nxt_keep):
            continue
        spk = str(seg.get("speaker_id") or "")
        if not spk:
            continue
        if str(prev.get("speaker_id") or "") != spk:
            continue
        if str(nxt.get("speaker_id") or "") != spk:
            continue
        seg["language"] = "en"
        seg["keep_original"] = True
        text = (seg.get("text") or "").strip()
        seg["text_en"] = (seg.get("text_en") or text).strip() or text
        for p in seg.get("phrases") or []:
            if not (p.get("text_en") or "").strip():
                p["text_en"] = (p.get("text") or "").strip()
        n += 1
        print(
            f"  Retag EN-sandwich → KEEP [{spk}] "
            f"{float(seg['start']):.1f}-{float(seg['end']):.1f}s | {text[:60]}",
            file=sys.stderr,
        )
    return n


def _is_keep_segment(seg: dict[str, Any]) -> bool:
    return bool(seg.get("keep_original") or (seg.get("language") or "he") != "he")


def _segment_clip_len(seg: dict[str, Any]) -> float:
    """Natural TTS clip length (seconds) for elastic packing."""
    stored = seg.get("tts_clip_sec")
    if stored is not None:
        try:
            v = float(stored)
            if v > 0.05:
                return v
        except (TypeError, ValueError):
            pass
    fitted = seg.get("tts_fit")
    if fitted:
        path = Path(fitted)
        if path.is_file():
            try:
                from inference.tts_f5 import wav_duration

                return max(0.05, float(wav_duration(path)))
            except Exception:
                pass
    a = float(seg.get("source_start") or seg.get("start") or 0.0)
    b = float(seg.get("source_end") or seg.get("end") or a)
    return max(0.25, b - a)


def plan_dub_placement(
    segments: list[dict[str, Any]],
    media_duration: float,
    *,
    max_drift: float = MAX_DRIFT_SEC,
    run_break_gap: float = RUN_BREAK_GAP_SEC,
    inter_gap: float = INTER_GAP_SEC,
    keep_guard: float = PLACE_KEEP_GUARD_SEC,
    max_speed: float = PLACE_MAX_SPEED,
) -> int:
    """Plan elastic ``place_start`` / ``place_end`` / ``place_speed`` for HE dubs.

    Groups consecutive Hebrew dubs into runs bounded by KEEP/non-Hebrew segments
    and by natural gaps ≥ ``run_break_gap``. Within a run, closes small gaps
    (``inter_gap``), keeps onset drift ≤ ``max_drift``, and — when the run would
    overrun a locked KEEP anchor — applies one uniform speed-up ≤ ``max_speed``.
    Never trims speech. Last segment before a run-break gap / media end is
    allowed to finish fully into that slack.

    Mutates ``segments`` in place. Returns the number of dub segments planned.
    """
    n = len(segments)
    media_duration = max(0.25, float(media_duration))
    planned = 0

    def _src_a(seg: dict[str, Any]) -> float:
        return float(seg.get("source_start") or seg.get("start") or 0.0)

    def _src_b(seg: dict[str, Any]) -> float:
        return float(seg.get("source_end") or seg.get("end") or _src_a(seg))

    # Initialize KEEP / failed / missing with locked source placement.
    for seg in segments:
        if _is_keep_segment(seg) or seg.get("tts_failed") or not seg.get("tts_fit"):
            a = _src_a(seg)
            b = _src_b(seg)
            seg["place_start"] = round(a, 3)
            seg["place_end"] = round(b, 3)
            seg["place_speed"] = 1.0
            seg["place_drift"] = 0.0

    i = 0
    while i < n:
        seg = segments[i]
        if (
            _is_keep_segment(seg)
            or seg.get("tts_failed")
            or not seg.get("tts_fit")
        ):
            i += 1
            continue

        run = [i]
        j = i + 1
        while j < n:
            nxt = segments[j]
            if _is_keep_segment(nxt):
                break
            if nxt.get("tts_failed") or not nxt.get("tts_fit"):
                break
            prev_b = _src_b(segments[run[-1]])
            cur_a = _src_a(nxt)
            if cur_a - prev_b >= float(run_break_gap):
                break
            run.append(j)
            j += 1

        # Locked end: KEEP onset − guard; else next-run onset / media end (soft).
        must_yield = False
        if j < n and _is_keep_segment(segments[j]):
            locked_end = _src_a(segments[j]) - float(keep_guard)
            must_yield = True
        elif j < n:
            locked_end = _src_a(segments[j])
        else:
            locked_end = media_duration
        locked_end = max(locked_end, _src_a(segments[run[0]]) + 0.25)

        natural_lens = [_segment_clip_len(segments[k]) for k in run]

        def _try_pack(speed: float) -> tuple[list[tuple[float, float]], bool]:
            lens = [max(0.05, L / max(speed, 1.0)) for L in natural_lens]
            places: list[tuple[float, float]] = []
            ok = True
            for ri, idx in enumerate(run):
                src_a = _src_a(segments[idx])
                if ri == 0:
                    p0 = src_a
                else:
                    prev_src_b = _src_b(segments[run[ri - 1]])
                    orig_gap = max(0.0, src_a - prev_src_b)
                    gap = min(orig_gap, float(inter_gap)) if orig_gap > 1e-3 else 0.0
                    # Close gaps: start right after previous end (+ tiny gap).
                    p0 = places[-1][1] + gap
                    if p0 < src_a - float(max_drift):
                        p0 = src_a - float(max_drift)
                    if p0 > src_a + float(max_drift):
                        p0 = src_a + float(max_drift)
                    if p0 + 1e-3 < places[-1][1]:
                        ok = False
                p1 = p0 + lens[ri]
                if abs(p0 - src_a) > float(max_drift) + 1e-3:
                    ok = False
                places.append((p0, p1))
            # Last segment before a soft gap may finish into the slack.
            if places and places[-1][1] > locked_end + 1e-3:
                if must_yield or places[-1][1] > locked_end + 0.05:
                    ok = False
            return places, ok

        # Prefer speed 1.0; escalate uniformly up to max_speed if needed.
        chosen_speed = 1.0
        places, ok = _try_pack(1.0)
        if not ok:
            lo, hi = 1.0, float(max_speed)
            best_places = places
            best_speed = hi
            for _ in range(12):
                mid = 0.5 * (lo + hi)
                cand, cand_ok = _try_pack(mid)
                if cand_ok:
                    best_places, best_speed = cand, mid
                    hi = mid
                else:
                    lo = mid
                    best_places = cand
            places, chosen_speed = best_places, best_speed
            # Final attempt at max_speed even if slightly imperfect (never trim).
            if not _try_pack(chosen_speed)[1]:
                places, _ = _try_pack(float(max_speed))
                chosen_speed = float(max_speed)

        for ri, idx in enumerate(run):
            p0, p1 = places[ri]
            src_a = _src_a(segments[idx])
            segments[idx]["place_start"] = round(p0, 3)
            segments[idx]["place_end"] = round(p1, 3)
            segments[idx]["place_speed"] = round(float(chosen_speed), 4)
            segments[idx]["place_drift"] = round(p0 - src_a, 3)
            planned += 1

        if abs(chosen_speed - 1.0) > 0.01:
            print(
                f"  Elastic pack run [{run[0]}..{run[-1]}]: "
                f"speed={chosen_speed:.3f}× → fit before "
                f"{'KEEP' if must_yield else 'gap/end'} @ {locked_end:.2f}s",
                file=sys.stderr,
            )
        i = j

    return planned


def apply_place_speeds(
    segments: list[dict[str, Any]],
    *,
    sample_rate: int = 44100,
) -> int:
    """Apply ``place_speed`` via atempo to ``tts_fit`` clips; refresh place_end.

    Returns the number of clips rewritten. Speeds ≈1.0 are skipped. Never trims.
    """
    from inference.tts_f5 import atempo_chain, wav_duration

    n_changed = 0
    for seg in segments:
        if _is_keep_segment(seg) or seg.get("tts_failed"):
            continue
        speed = float(seg.get("place_speed") or 1.0)
        fitted = seg.get("tts_fit")
        if not fitted or not Path(fitted).is_file():
            continue
        if abs(speed - 1.0) < 0.01:
            # Sync place_end to actual clip length at place_start.
            p0 = float(seg.get("place_start") or seg.get("source_start") or seg["start"])
            dur = wav_duration(Path(fitted))
            seg["place_end"] = round(p0 + dur, 3)
            seg["tts_clip_sec"] = round(dur, 3)
            continue
        src = Path(fitted)
        dst = src.with_name(src.stem + f"_pack{speed:.2f}".replace(".", "p") + ".wav")
        chain = atempo_chain(speed)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(src),
                "-filter:a",
                chain,
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                str(dst),
            ],
            check=True,
            capture_output=True,
        )
        seg["tts_fit"] = str(dst)
        p0 = float(seg.get("place_start") or seg.get("source_start") or seg["start"])
        dur = wav_duration(dst)
        seg["place_end"] = round(p0 + dur, 3)
        seg["tts_clip_sec"] = round(dur, 3)
        seg["spoken_end"] = round(p0 + dur, 3)
        # Combine with any prior gentle fit rate for QA.
        prior = float(seg.get("tts_speed_used") or 1.0)
        seg["tts_speed_used"] = round(prior * speed, 3)
        n_changed += 1
        print(
            f"  Pack speed [{seg.get('speaker_id')}] {speed:.3f}× → {dst.name} "
            f"({dur:.2f}s @ place {p0:.2f}-{p0 + dur:.2f})",
            file=sys.stderr,
        )
    return n_changed

