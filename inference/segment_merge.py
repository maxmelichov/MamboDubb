#!/usr/bin/env python3
"""Pause-aware utterance grouping for dubbing.

Rules:
  - Speaker change → new utterance
  - Silence / gap > max_pause → new utterance (VAD-like split)
  - Same speaker with a shorter pause → keep one utterance, record the pause
    so TTS can insert silence instead of starting a new sentence
"""

from __future__ import annotations

import re
from typing import Any


DEFAULT_MAX_PAUSE = 1.0  # seconds — above this, start a new utterance
DEFAULT_PHRASE_PAUSE = 0.18  # seconds — within an utterance, mark an internal pause
# Attach a short continuation clause across a diarization flip (unfinished → next).
DEFAULT_STITCH_GAP = 0.55

_UNFINISHED_TAIL = re.compile(
    r"(מממן|מממנת|שפועל|שפועלת|funds|fund|operates|and|the|את|של|עם|ו)\.?$",
    re.IGNORECASE,
)
_CLAUSE_END = re.compile(r"[.!?؟]\s+")


def utterance_unfinished(text: str) -> bool:
    """True when the transcript looks mid-sentence / mid-list."""
    t = (text or "").strip().rstrip("\"'”’")
    if not t:
        return False
    if t.endswith((",", "،", ";", ":", "—", "-")):
        return True
    if t.endswith("…") or t.endswith("..."):
        return True
    if _UNFINISHED_TAIL.search(t):
        return True
    if t[-1] not in ".!?؟":
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


def stitch_unfinished_continuations(
    segments: list[dict[str, Any]],
    *,
    max_gap: float = DEFAULT_STITCH_GAP,
) -> list[dict[str, Any]]:
    """If A ends mid-sentence and B begins the completion, move that clause onto A.

    Keeps B's remaining text/speaker so later voices stay distinct
    (e.g. "…that operates" + "in the world — she funds it." → same speaker,
    while "Qatar funds Hamas…" stays on the next speaker).
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
        cur_tail = (cur.get("text") or "").strip()
        if cur.get("phrases"):
            cur_tail = (cur["phrases"][-1].get("text") or cur_tail).strip()

        if not (
            same_lang
            and same_keep
            and not cur.get("keep_original")
            and 0 <= gap <= max_gap
            and utterance_unfinished(cur_tail)
            and nxt.get("phrases")
        ):
            out.append(cur)
            i += 1
            continue

        first = dict(nxt["phrases"][0])
        he_head, he_rest = _split_first_clause(first.get("text") or "")
        en_src = (first.get("text_en") or "").strip()
        en_head, en_rest = _split_first_clause(en_src) if en_src else ("", "")
        if not he_head or not he_rest:
            # Need a clear clause boundary on the next segment.
            out.append(cur)
            i += 1
            continue

        # Time-split the first next phrase by character share of the head clause.
        p0 = float(first["start"])
        p1 = float(first["end"])
        frac = max(0.15, min(0.85, len(he_head) / max(1, len((first.get("text") or "").strip()))))
        split_t = round(p0 + (p1 - p0) * frac, 3)
        # Prefer a slightly earlier split so the next speaker owns the new onset.
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
            # Fallback: whole EN head estimate when EN lacked punctuation.
            moved["text_en"] = en_src.split(".")[0].strip() + ("." if "." in en_src else "")

        if cur["phrases"]:
            gap_pause = round(max(0.0, p0 - float(cur["phrases"][-1]["end"])), 3)
            cur["phrases"][-1]["pause_after"] = gap_pause
        cur["phrases"].append(moved)
        cur["end"] = split_t
        cur["text"] = " ".join(
            (p.get("text") or "").strip() for p in cur["phrases"] if p.get("text")
        )
        en_bits = [
            (p.get("text_en") or "").strip()
            for p in cur["phrases"]
            if (p.get("text_en") or "").strip()
        ]
        if en_bits:
            cur["text_en"] = " ".join(en_bits)
        cur["duration"] = round(float(cur["end"]) - float(cur["start"]), 3)
        cur["pauses"] = [float(p.get("pause_after") or 0.0) for p in cur["phrases"]]
        for key in ("tts_fit", "tts_raw", "tts_speed_used"):
            cur.pop(key, None)

        rest_phrase = {
            "text": he_rest,
            "start": split_t,
            "end": p1,
            "pause_after": float(first.get("pause_after") or 0.0),
            "speaker_id": nxt.get("speaker_id"),
        }
        if en_rest:
            rest_phrase["text_en"] = en_rest
        elif en_src and not en_head:
            pass
        nxt["phrases"] = [rest_phrase] + nxt["phrases"][1:]
        nxt["start"] = split_t
        nxt["text"] = " ".join(
            (p.get("text") or "").strip() for p in nxt["phrases"] if p.get("text")
        )
        en_bits_n = [
            (p.get("text_en") or "").strip()
            for p in nxt["phrases"]
            if (p.get("text_en") or "").strip()
        ]
        if en_bits_n:
            nxt["text_en"] = " ".join(en_bits_n)
        elif "text_en" in nxt:
            # Drop stale full-string EN; rebuild from phrases only.
            nxt.pop("text_en", None)
        nxt["duration"] = round(float(nxt["end"]) - float(nxt["start"]), 3)
        nxt["pauses"] = [float(p.get("pause_after") or 0.0) for p in nxt["phrases"]]
        for key in ("tts_fit", "tts_raw", "tts_speed_used"):
            nxt.pop(key, None)

        segments[i + 1] = nxt
        out.append(cur)
        i += 1

    return out


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
            # Preserve already-rendered TTS paths across merge (selective rebuilds).
            for key in (
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

        if same_lang and same_spk and 0 <= gap <= max_pause:
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
    if cur["phrases"]:
        cur["phrases"][-1]["pause_after"] = 0.0
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
