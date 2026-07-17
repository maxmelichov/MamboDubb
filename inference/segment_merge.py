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
# Diarization often flips speaker mid-clause; bridge tiny gaps so ASR isn't clipped.
DEFAULT_CROSS_SPEAKER_GAP = 0.45


_UNFINISHED_TAIL = re.compile(
    r"(מממן|מממנת|funds|fund|and|the|את|של|עם|ו)\.?$",
    re.IGNORECASE,
)


def utterance_unfinished(text: str) -> bool:
    """True when the transcript looks mid-sentence / mid-list (don't hard-cut)."""
    t = (text or "").strip().rstrip("\"'”’")
    if not t:
        return False
    if t.endswith((",", "،", ";", ":", "—", "-")):
        return True
    if t.endswith("…") or t.endswith("..."):
        return True
    if _UNFINISHED_TAIL.search(t):
        return True
    # No strong terminal punctuation → treat as open when bridging short gaps.
    if t[-1] not in ".!?؟":
        return True
    return False


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
    cross_speaker_gap: float = DEFAULT_CROSS_SPEAKER_GAP,
) -> list[dict[str, Any]]:
    """Merge consecutive same-language segments when the gap is a short pause.

    Also bridges *unfinished* sentences across a tiny speaker-id flip (diarization
    chatter), so we don't cut mid-list ("Qatar funds…" / "…al-Qaeda").
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
        prev_tail = ""
        if cur.get("phrases"):
            prev_tail = (cur["phrases"][-1].get("text") or "").strip()
        unfinished = utterance_unfinished(prev_tail)

        bridge_same = same_lang and same_spk and 0 <= gap <= max_pause
        # Mid-sentence across a spurious speaker flip (very short gap only).
        bridge_cross = (
            same_lang
            and not same_spk
            and unfinished
            and 0 <= gap <= cross_speaker_gap
        )

        if bridge_same or bridge_cross:
            cur["phrases"][-1]["pause_after"] = round(max(gap, 0.0), 3)
            cur["phrases"].extend(dict(p) for p in phrases)
            cur["end"] = end
            # Prefer the speaker who owns more of the merged span.
            if bridge_cross:
                cur_dur = float(cur["end"]) - float(cur["start"])
                new_dur = end - start
                if new_dur > cur_dur * 0.6:
                    cur["speaker_id"] = spk
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
    cross_speaker_gap: float = DEFAULT_CROSS_SPEAKER_GAP,
) -> list[dict[str, Any]]:
    """Merge adjacent diarization turns so ASR is not clipped mid-utterance.

    Same speaker: merge gaps ≤ max_pause.
    Different speaker: merge only micro-gaps (diarization flip) so one Whisper
    window covers a continuing clause; keep the longer turn's speaker_id.
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
        elif (not same_spk) and 0 <= gap <= cross_speaker_gap:
            cur_dur = float(cur["end"]) - float(cur["start"])
            turn_dur = float(turn["end"]) - float(turn["start"])
            if turn_dur > cur_dur:
                cur["speaker_id"] = turn["speaker_id"]
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
