"""Stage 4 — group the word stream into speakable segments.

Everything here is a pure function over the word list except `diarize`, so the
segmentation rules can be unit-tested without audio. Segments are built from
word timings only; a segment never spans a speaker change, so later stages can
treat `segment.speaker` as exact.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

GAP_SPLIT = 0.70       # silence between words that ends a segment
SPEAKER_GAP = 0.25     # diarization must be backed by a real pause to split
MAX_LEN = 10.0         # target segment length; stubs never merge past it
# A group only reaches the length splitter if it holds no interior sentence end,
# i.e. it is a single sentence. Splitting one mid-clause makes both halves
# translate badly (each fragment is ambiguous) and gives them separate, clashing
# TTS prosody — the audible "jump". Qwen voices ~13s cleanly, so a lone long
# sentence is kept whole and only genuinely huge ones are split.
SENTENCE_MAX = 13.0
MIN_LEN = 1.2          # below this a segment is a stub and wants merging
MIN_WORDS = 3
MERGE_GAP = 0.50       # only merge a stub across a gap this small
WORD_MAX = 0.80        # a word never occupies more than this
SENTENCE_END = re.compile(r"[.!?…]['\"»׳״]?$")

KEEP_TAIL_MAX = 2.5      # longest trailing original audio a keep segment may reclaim
KEEP_TAIL_FLOOR = 0.008  # vocal energy above this is still speech, not a pause
KEEP_TAIL_SILENCE = 0.35 # this much quiet ends the tail
UNCOVERED_MIN = 0.6      # shortest audible gap worth keeping as original audio
UNCOVERED_PEAK = 0.04    # a gap needs energy this loud somewhere to count as speech

LATIN = re.compile(r"[A-Za-z]")
HEBREW = re.compile(r"[֐-׿]")
SPEAKER_EN_RATIO = 0.60
FOREIGN_MIN_WORDS = 3  # a shorter Latin run is an embedded token (acronym/brand)
                       # inside source speech, not a target-language passage

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"


def script_of(text: str) -> str | None:
    """'latin', 'source', or None for characters that belong to neither."""
    lat = len(LATIN.findall(text or ""))
    heb = len(HEBREW.findall(text or ""))
    if not lat and not heb:
        return None
    return "latin" if lat > heb else "source"


def latin_ratio(text: str) -> float:
    lat = len(LATIN.findall(text or ""))
    heb = len(HEBREW.findall(text or ""))
    if lat + heb == 0:
        return 0.0
    return lat / (lat + heb)


def word_ends(words: list[dict[str, Any]]) -> list[float]:
    """End time per word.

    Whisper words carry a measured `end`; caption words give only an onset and
    fall back to a generous fixed span (a segment that ends too early would clip
    the original audio of a kept segment). Either way a word never runs past the
    next word's onset.
    """
    ends: list[float] = []
    for i, w in enumerate(words):
        nxt = words[i + 1]["t"] if i + 1 < len(words) else None
        end = w.get("end")
        if end is None:
            end = (w["t"] + WORD_MAX) if nxt is None else min(nxt, w["t"] + WORD_MAX)
        elif nxt is not None:
            end = min(end, nxt)
        ends.append(max(end, w["t"] + 0.02))
    return ends


def speech_gap(prev: dict[str, Any], cur: dict[str, Any]) -> float:
    """Silence between two words.

    Uses Whisper's measured word end when present, so the split thresholds below
    mean actual seconds of silence. Caption words, which give only an onset, fall
    back to a length-based guess of how long `prev` took to say.
    """
    end = prev.get("end")
    if end is None:
        end = prev["t"] + min(WORD_MAX, 0.10 + 0.06 * len(prev["text"]))
    return max(0.0, cur["t"] - end)


def assign_word_speakers(words: list[dict[str, Any]], turns: list[dict[str, Any]]) -> None:
    """Tag each word with the diarization turn covering it (in place)."""
    if not turns:
        for w in words:
            w["spk"] = "SPEAKER_00"
        return
    turns = sorted(turns, key=lambda t: t["start"])
    ends = word_ends(words)
    idx = 0
    prev = turns[0]["speaker"]
    for w, end in zip(words, ends):
        mid = 0.5 * (w["t"] + end)
        while idx + 1 < len(turns) and turns[idx]["end"] < mid:
            idx += 1
        best, best_ov = None, 0.0
        for t in turns[max(0, idx - 1) : idx + 2]:
            ov = min(end, t["end"]) - max(w["t"], t["start"])
            if ov > best_ov:
                best, best_ov = t["speaker"], ov
        if best is None:
            near = min(turns, key=lambda t: min(abs(t["start"] - mid), abs(t["end"] - mid)))
            gap = min(abs(near["start"] - mid), abs(near["end"] - mid))
            best = near["speaker"] if gap <= 0.5 else prev
        w["spk"] = best
        prev = best


def _make(words: list[dict[str, Any]], ends: list[float]) -> dict[str, Any]:
    # Majority vote, weighted by time: diarization boundaries are a word or two
    # off, and a stray label at the edge must not rename the whole segment.
    votes: dict[str, float] = {}
    for w, end in zip(words, ends):
        spk = w.get("spk", "SPEAKER_00")
        votes[spk] = votes.get(spk, 0.0) + max(0.05, end - w["t"])
    return {
        "start": round(words[0]["t"], 3),
        "end": round(ends[-1], 3),
        "speaker": max(votes.items(), key=lambda kv: kv[1])[0],
        "text": " ".join(w["text"] for w in words),
        "_words": words,
        "_ends": ends,
    }


def _split_long(seg: dict[str, Any]) -> list[dict[str, Any]]:
    """Recursively split an over-long single sentence at its largest interior pause."""
    if seg["end"] - seg["start"] <= SENTENCE_MAX or len(seg["_words"]) < 4:
        return [seg]
    words, ends = seg["_words"], seg["_ends"]
    best_i, best_gap = None, -1.0
    for i in range(1, len(words)):
        left = ends[i - 1] - seg["start"]
        right = seg["end"] - words[i]["t"]
        if left < 1.0 or right < 1.0:
            continue
        gap = speech_gap(words[i - 1], words[i])
        # Prefer a real pause; break ties toward the middle of the segment.
        score = gap - 0.02 * abs(left - right)
        if score > best_gap:
            best_i, best_gap = i, score
    if best_i is None:
        best_i = len(words) // 2
    left = _make(words[:best_i], ends[:best_i])
    right = _make(words[best_i:], ends[best_i:])
    return _split_long(left) + _split_long(right)


def _merge_stubs(segs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold sub-second fragments into a neighbour (same speaker, small gap only)."""
    out = list(segs)
    changed = True
    while changed:
        changed = False
        for i, seg in enumerate(out):
            short = (seg["end"] - seg["start"] < MIN_LEN) or (len(seg["_words"]) < MIN_WORDS)
            if not short:
                continue
            cands = []
            if i > 0 and out[i - 1]["speaker"] == seg["speaker"]:
                cands.append((seg["start"] - out[i - 1]["end"], i - 1))
            if i + 1 < len(out) and out[i + 1]["speaker"] == seg["speaker"]:
                cands.append((out[i + 1]["start"] - seg["end"], i + 1))
            cands = [c for c in cands if c[0] <= MERGE_GAP]
            if not cands:
                continue
            _gap, j = min(cands)
            a, b = (j, i) if j < i else (i, j)
            if out[b]["_words"][0].get("brk"):
                continue   # never merge across an explicit caption speaker change
            sa, sb = script_of(out[a]["text"]), script_of(out[b]["text"])
            if sa and sb and sa != sb:
                continue   # nor across a change of language
            merged = _make(out[a]["_words"] + out[b]["_words"], out[a]["_ends"] + out[b]["_ends"])
            if merged["end"] - merged["start"] > MAX_LEN:
                continue
            out[a : b + 1] = [merged]
            changed = True
            break
    return out


def _embedded_latin(words: list[dict[str, Any]]) -> list[bool]:
    """Mark Latin words that are short embedded tokens, not a foreign passage.

    A lone acronym or brand dropped into source speech (`...מממן את ISIS, את...`)
    is written in Latin letters but spoken by the same person in the same breath.
    Treating it as a script change would carve it into its own segment and play a
    fraction of a second of original audio — an audible voice jump. Only a
    sustained run of Latin words is a real target-language passage; anything
    shorter stays with its neighbours and is dubbed like the rest of the sentence.
    """
    n = len(words)
    latin = [script_of(w["text"]) == "latin" for w in words]
    embedded = [False] * n
    i = 0
    while i < n:
        if not latin[i]:
            i += 1
            continue
        j = i
        while j < n and latin[j]:
            j += 1
        if j - i < FOREIGN_MIN_WORDS:
            for k in range(i, j):
                embedded[k] = True
        i = j
    return embedded


def words_to_segments(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not words:
        return []
    ends = word_ends(words)
    # When the transcript marks speaker changes itself those markers are exact,
    # so diarization gets no vote on boundaries — its edges land a word or two
    # off and would cut sentences in half. Without markers (ASR fallback) a
    # diarization change still splits, but only when a real pause confirms it.
    has_markers = any(w.get("brk") for w in words)
    embedded = _embedded_latin(words)
    groups: list[list[int]] = [[0]]
    script = None if embedded[0] else script_of(words[0]["text"])
    for i in range(1, len(words)):
        w, prev = words[i], words[i - 1]
        gap = speech_gap(prev, w)
        # A change of script is a change of language, and the two halves get
        # opposite treatment — one dubbed, one kept as original audio — so they
        # can never share a segment. An embedded token (short Latin run) is not a
        # language change and does not set or break the script context.
        this_script = None if embedded[i] else script_of(w["text"])
        script_change = this_script is not None and script is not None and this_script != script
        if this_script is not None:
            script = this_script
        speaker_change = (
            not has_markers
            and w.get("spk", "") != prev.get("spk", "")
            and gap >= SPEAKER_GAP
        )
        split = (
            gap >= GAP_SPLIT
            or w.get("brk")
            or speaker_change
            or script_change
            or bool(SENTENCE_END.search(prev["text"]))
        )
        if split:
            groups.append([i])
        else:
            groups[-1].append(i)

    segs: list[dict[str, Any]] = []
    for g in groups:
        segs.extend(_split_long(_make([words[i] for i in g], [ends[i] for i in g])))
    segs = _merge_stubs(segs)

    out: list[dict[str, Any]] = []
    for i, seg in enumerate(segs):
        out.append({
            "id": i,
            "start": seg["start"],
            "end": seg["end"],
            "speaker": seg["speaker"],
            "text": seg["text"],
        })
    return out


def splice_foreign_spans(segs: list[dict[str, Any]],
                         spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace transcript segments inside target-language spans with the captions'.

    ASR in the source language either mangles target-language speech or skips it
    entirely, which would leave those seconds with no segment and therefore no
    audio at all. The captions know where that speech is and what it says, so
    those spans become segments in their own right.
    """
    # A span with no words of its own cannot replace what it displaces, and
    # dropping segments for it would leave those seconds silent.
    spans = [s for s in spans if s.get("words")]
    if not spans:
        return segs

    kept: list[dict[str, Any]] = []
    for seg in segs:
        dur = max(0.05, seg["end"] - seg["start"])
        overlap = sum(max(0.0, min(seg["end"], s["end"]) - max(seg["start"], s["start"]))
                      for s in spans)
        if overlap / dur > 0.5:
            continue                      # this segment is the captions' territory
        trimmed = False
        for s in spans:                   # trim any partial intrusion
            if s["start"] < seg["end"] <= s["end"]:
                seg["end"], trimmed = round(s["start"], 3), True
            elif s["start"] <= seg["start"] < s["end"]:
                seg["start"], trimmed = round(s["end"], 3), True
        # Only a trimmed remnant may be discarded. An untouched segment is kept
        # however short, or its words would vanish from the output entirely.
        if not trimmed or seg["end"] - seg["start"] >= 0.4:
            kept.append(seg)

    for s in spans:
        words = [{"t": w["t"], "text": w["text"], "brk": False, "spk": w.get("spk", "SPEAKER_00")}
                 for w in s.get("words") or []]
        made = words_to_segments(words) if words else []
        if not made:
            continue
        # Tile the span: these all play original audio, so any hole between them
        # would be a silent stretch in the middle of someone talking. Recovered
        # word timings in particular bunch together and cannot be trusted to
        # cover the passage on their own.
        # Clamped to the span exactly, so the passage tiles it and nothing spills
        # past the edge onto a neighbouring segment.
        made[0]["start"] = round(s["start"], 3)
        for a, b in zip(made, made[1:]):
            a["end"] = round(b["start"], 3)
        made[-1]["end"] = round(s["end"], 3)
        kept.extend(x for x in made if x["end"] > x["start"])
    return sorted(kept, key=lambda x: x["start"])


def unsegmented_words(words: list[dict[str, Any]], segs: list[dict[str, Any]],
                      spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Transcript words that no segment covers.

    A segment is the only route audio has to the output, so a word left out here
    is a word the viewer never hears. Words inside a target-language span are
    exempt: those were deliberately replaced by the captions' own words.
    """
    ranges = sorted((s["start"], s["end"]) for s in segs)
    # Span edges are approximate — they come from caption word onsets — so a word
    # just outside one is still that passage's speech and is covered by its audio.
    edge = 1.0
    span_ranges = [(s["start"] - edge, s["end"] + edge) for s in spans]
    lost = []
    for w in words:
        t = w["t"]
        if any(a <= t <= b for a, b in span_ranges):
            continue
        if not any(a - 0.05 <= t <= b + 0.05 for a, b in ranges):
            lost.append(w)
    return lost


def mark_keep(segments: list[dict[str, Any]],
              spans: list[dict[str, Any]] | None = None) -> None:
    """Flag segments whose original audio should play instead of a dub.

    Two content-free rules: the caption text is already in the target script, or
    the speaker is predominantly a target-language speaker (auto-captions often
    render their speech phonetically in the source script, which the per-segment
    test alone would miss).
    """
    def letters(text: str) -> int:
        return len(LATIN.findall(text or "")) + len(HEBREW.findall(text or ""))

    del spans   # target-language spans are handled structurally by splice_foreign_spans
    totals: dict[str, list[float]] = {}
    for seg in segments:
        if letters(seg["text"]) < 2:
            continue
        dur = seg["end"] - seg["start"]
        agg = totals.setdefault(seg["speaker"], [0.0, 0.0])
        agg[0] += dur
        if latin_ratio(seg["text"]) > 0.5:
            agg[1] += dur
    en_speakers = {
        spk for spk, (total, lat) in totals.items() if total > 0 and lat / total >= SPEAKER_EN_RATIO
    }
    for seg in segments:
        if letters(seg["text"]) < 2:
            # Transcript noise (stray glyphs). Nothing to translate, so let the
            # original audio through rather than leaving a hole.
            seg["keep"], seg["keep_reason"] = True, "no_text"
        elif latin_ratio(seg["text"]) > 0.5:
            seg["keep"], seg["keep_reason"] = True, "latin"
        elif seg["speaker"] in en_speakers:
            seg["keep"], seg["keep_reason"] = True, "speaker_en"
        else:
            seg["keep"], seg["keep_reason"] = False, None


def diarize(vocals: Path) -> list[dict[str, Any]]:
    """Pyannote turns; returns [] (single-speaker fallback) if unavailable."""
    try:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        import torch
        from pyannote.audio import Pipeline

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        try:
            pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=token)
        except TypeError:
            pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=token)
        if pipeline is None:
            raise RuntimeError("pyannote returned no pipeline (HF_TOKEN / model terms?)")
        if torch.backends.mps.is_available():
            pipeline.to(torch.device("mps"))
        elif torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        output = pipeline(str(vocals))
        annotation = getattr(output, "speaker_diarization", output)
        return [
            {"speaker": spk, "start": round(float(turn.start), 3), "end": round(float(turn.end), 3)}
            for turn, _, spk in annotation.itertracks(yield_label=True)
        ]
    except Exception as exc:
        print(f"  segments: diarization unavailable ({exc}) — single speaker", file=sys.stderr)
        return []


def extend_keeps_to_speech_end(segs: list[dict[str, Any]], levels, hop: float,
                               total: float) -> None:
    """Grow each keep segment's end through trailing speech, up to the next segment.

    Keep segments play original audio. The source-language ASR stops transcribing
    before an English speaker actually stops (and Whisper's word-end timestamps
    bunch early), so the segment boundary lands mid-sentence and the tail is
    clipped. Walk the vocal energy forward from the boundary, reclaiming speech
    until a real pause or the next segment — whichever comes first. The
    English/Hebrew boundary itself is handled precisely upstream by the language
    detector (transcript.detect_spoken_target_spans), so this never eats into a
    neighbour: it only recovers a VAD trim within a keep's own trailing silence.
    """
    n = len(levels)
    for i, s in enumerate(segs):
        if not s.get("keep"):
            continue
        nxt_start = segs[i + 1]["start"] if i + 1 < len(segs) else total
        limit = min(nxt_start, s["end"] + KEEP_TAIL_MAX)
        end, silence, t = s["end"], 0.0, s["end"]
        while t < limit - 1e-9:
            j = int(t / hop)
            if j >= n:
                break
            if levels[j] >= KEEP_TAIL_FLOOR:
                end, silence = min(limit, t + hop), 0.0
            else:
                silence += hop
                if silence >= KEEP_TAIL_SILENCE:
                    break
            t += hop
        if end > s["end"] + 0.05:
            s["end"] = round(end, 3)


def fill_uncovered_audible(segs: list[dict[str, Any]], levels, hop: float,
                           total: float) -> None:
    """Keep original audio for audible stretches no segment covers.

    A region the language detector mislabels as neither source nor target, and the
    ASR never transcribed, otherwise plays silent — the speaker vanishes (the "1:35
    goes quiet" gap). Cover it with original audio so it is at least heard.
    """
    import numpy as np

    n = len(levels)
    covered = np.zeros(n, dtype=bool)
    for s in segs:
        covered[max(0, int(s["start"] / hop)) : min(n, int(s["end"] / hop) + 1)] = True
    added: list[tuple[float, float]] = []
    i = 0
    while i < n:
        if covered[i]:
            i += 1
            continue
        j = i
        while j < n and not covered[j]:
            j += 1
        a, b = i * hop, min(total, j * hop)
        if b - a >= UNCOVERED_MIN and float(np.max(levels[i:j])) >= UNCOVERED_PEAK:
            added.append((round(a, 3), round(b, 3)))
        i = j
    for a, b in added:
        segs.append({"id": -1, "start": a, "end": b, "speaker": "SPEAKER_00",
                     "text": "", "keep": True, "keep_reason": "uncovered"})
    if added:
        segs.sort(key=lambda s: s["start"])
        for k, s in enumerate(segs):
            s["id"] = k


def run(m: dict[str, Any], workdir: Path, words: list[dict[str, Any]],
        spans: list[dict[str, Any]] | None = None) -> None:
    turns = diarize(workdir / m["files"]["vocals"])
    assign_word_speakers(words, turns)
    segs = words_to_segments(words)
    if spans:
        span_words = [w for s in spans for w in (s.get("words") or [])]
        assign_word_speakers(span_words, turns)
        segs = splice_foreign_spans(segs, spans)
        for i, seg in enumerate(segs):
            seg["id"] = i
    mark_keep(segs, spans)

    # Drop sub-word noise fragments (a lone "ב", stray glyphs). Kept as original
    # audio they play a jarring one-letter blip of the source voice between dubbed
    # lines; letting those seconds fall to the background bed is far smoother. Their
    # (sub-two-letter) words carry no meaning, so nothing is lost.
    segs = [s for s in segs if s.get("keep_reason") != "no_text"]
    for i, seg in enumerate(segs):
        seg["id"] = i

    lost = [w for w in unsegmented_words(words, segs, spans or [])
            if len(re.sub(r"[^A-Za-z֐-׿]", "", w["text"])) >= 2]
    assert not lost, (
        f"{len(lost)} transcript words fell outside every segment, starting at "
        f"{lost[0]['t']:.2f}s ({lost[0]['text']!r}) — they would never be heard"
    )
    # Keep segments (original audio) must not clip a speaker mid-sentence.
    from . import audio

    total = float(m["source"].get("duration") or 0.0)
    levels = audio.frame_rms(audio.decode_mono(workdir / m["files"]["vocals"], 16000),
                             16000, 0.1)
    extend_keeps_to_speech_end(segs, levels, 0.1, total or len(levels) * 0.1)
    fill_uncovered_audible(segs, levels, 0.1, total or len(levels) * 0.1)
    m["segments"] = segs
    m["speakers"] = {
        spk: {"dur": round(sum(s["end"] - s["start"] for s in segs if s["speaker"] == spk), 2)}
        for spk in sorted({s["speaker"] for s in segs})
    }
    kept = sum(1 for s in segs if s["keep"])
    print(
        f"  segments: {len(segs)} segments, {len(m['speakers'])} speakers, "
        f"{kept} keep-original",
        file=sys.stderr,
    )
