#!/usr/bin/env python3
"""Pause-aware utterance grouping for dubbing.

Rules:
  - Speaker change → new utterance
  - Silence / gap > max_pause → new utterance (VAD-like split)
  - Same speaker with a shorter pause → keep one utterance, record the pause
    so TTS can insert silence instead of starting a new sentence
"""

from __future__ import annotations

from typing import Any


DEFAULT_MAX_PAUSE = 1.0  # seconds — above this, start a new utterance
DEFAULT_PHRASE_PAUSE = 0.18  # seconds — within an utterance, mark an internal pause


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


def merge_same_speaker_segments(
    segments: list[dict[str, Any]],
    *,
    max_pause: float = DEFAULT_MAX_PAUSE,
) -> list[dict[str, Any]]:
    """Merge consecutive same-speaker segments when the gap is a short pause.

    Output segment shape:
      {
        speaker_id, start, end, duration, text,
        phrases: [{text, text_en?, start, end, pause_after}, ...],
        pauses: [float, ...]  # pause after phrase i (last usually 0)
      }
    """
    if not segments:
        return []

    ordered = sorted(segments, key=lambda s: float(s["start"]))
    merged: list[dict[str, Any]] = []

    cur = None
    for seg in ordered:
        spk = seg["speaker_id"]
        start = float(seg["start"])
        end = float(seg["end"])
        text = (seg.get("text") or "").strip()
        text_en = (seg.get("text_en") or "").strip() or None
        if not text and not text_en:
            continue

        phrase = {
            "text": text,
            "start": start,
            "end": end,
            "pause_after": 0.0,
        }
        if text_en:
            phrase["text_en"] = text_en

        if cur is None:
            cur = {
                "speaker_id": spk,
                "start": start,
                "end": end,
                "text": text,
                "phrases": [phrase],
            }
            if text_en:
                cur["text_en"] = text_en
            continue

        gap = start - float(cur["end"])
        same = spk == cur["speaker_id"]
        if same and 0 <= gap <= max_pause:
            # Same speaker continuing after a pause — keep one utterance
            cur["phrases"][-1]["pause_after"] = round(max(gap, 0.0), 3)
            cur["phrases"].append(phrase)
            cur["end"] = end
            # Keep spaces between merged phrases (Whisper tokens may omit them)
            cur["text"] = " ".join(
                p["text"].strip() for p in cur["phrases"] if (p.get("text") or "").strip()
            )
            if text_en:
                prev_en = (cur.get("text_en") or "").strip()
                cur["text_en"] = (prev_en + " " + text_en).strip() if prev_en else text_en
        else:
            cur["duration"] = round(float(cur["end"]) - float(cur["start"]), 3)
            cur["pauses"] = [float(p.get("pause_after") or 0.0) for p in cur["phrases"]]
            merged.append(cur)
            cur = {
                "speaker_id": spk,
                "start": start,
                "end": end,
                "text": text,
                "phrases": [phrase],
            }
            if text_en:
                cur["text_en"] = text_en

    if cur is not None:
        cur["duration"] = round(float(cur["end"]) - float(cur["start"]), 3)
        cur["pauses"] = [float(p.get("pause_after") or 0.0) for p in cur["phrases"]]
        merged.append(cur)

    return merged


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
