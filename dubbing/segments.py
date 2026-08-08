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

from . import script

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
MIN_SEG_SEC = 0.9      # below this TTS reliably fails on the line, so the segment
                       # may cross a dramatic pause (like a lone word) to rejoin a
                       # substantial same-speaker neighbour rather than stay alone
MIN_WORDS = 3
MERGE_GAP = 0.50       # only merge a stub across a gap this small
LONE_WORD_GAP = 1.5    # ...except a single-word stub: one word is almost never a
                       # complete utterance, and dubbed alone it plays as a random
                       # interjection ("קשרות" → a stray "Relations" mid-pause), so
                       # it may cross a dramatic pause to rejoin its sentence
WORD_MAX = 0.80        # a word never occupies more than this
SENTENCE_END = re.compile(r"[.!?…]['\"»׳״]?$")

KEEP_TAIL_MAX = 2.5      # longest trailing original audio a keep segment may reclaim
KEEP_TAIL_FLOOR = 0.008  # vocal energy above this is still speech, not a pause
KEEP_TAIL_SILENCE = 0.35 # this much quiet ends the tail
UNCOVERED_MIN = 0.6      # shortest audible gap worth keeping as original audio
UNCOVERED_PEAK = 0.04    # a gap needs energy this loud somewhere to count as speech
UNCOVERED_HEAD_MAX = 1.0 # how far into an uncovered gap to look for the pause that
                         # ends the previous speaker, before giving up and starting
SPLICE_MIN_REMNANT = 0.4 # a trimmed piece shorter than this holds no speakable line
WORD_OVERLAP = 0.05      # a word must overlap a piece by more than this to belong to it

SPEAKER_EN_RATIO = 0.60
FOREIGN_MIN_WORDS = 3  # a shorter target-script run is an embedded token (acronym/
                       # brand) inside source speech, not a target-language passage

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"


def text_bucket(text: str, src: str = "he", tgt: str = "en") -> str | None:
    """'target', 'source', or None when the letters name neither.

    Script is a cheap, high-precision language signal only when the pair is
    written differently. For a same-script pair (en→es) letters carry no signal
    at all, so the answer is always None and every script-based shortcut stays
    off — the LID callback path decides instead.
    """
    if script.same_script(src, tgt):
        return None
    t = script.count_letters(text or "", script.script_for(tgt))
    s = script.count_letters(text or "", script.script_for(src))
    if not t and not s:
        return None
    return "target" if t > s else "source"


def script_of(text: str) -> str | None:
    """Legacy he→en wrapper: 'latin', 'source', or None for neither."""
    bucket = text_bucket(text, "he", "en")
    return "latin" if bucket == "target" else bucket


def latin_ratio(text: str) -> float:
    """Legacy he→en wrapper: Latin letters over Latin+Hebrew letters."""
    lat = script.count_letters(text or "", "latin")
    heb = script.count_letters(text or "", "hebrew")
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


def _turn_boundaries(turns: list[dict[str, Any]]) -> list[float]:
    """Times where diarization hands the floor to a different speaker."""
    turns = sorted(turns, key=lambda t: t["start"])
    return [b["start"] for a, b in zip(turns, turns[1:]) if a["speaker"] != b["speaker"]]


def _split_speaker_turns(segs: list[dict[str, Any]],
                         turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split each segment at diarization speaker changes that fall inside it.

    Fast dialogue hands the floor over with no pause, so the gap-confirmed
    speaker split above misses it and two characters' turns fuse into one
    segment — courtroom Q+A dubbed in a single cloned voice. The turn list
    knows where the handover happened; cut between the two words nearest it,
    with at least one word on each side, so the pieces stay non-overlapping
    and keep their word alignment. With one speaker (or no turn data) there
    are no boundaries and nothing changes.
    """
    bounds = _turn_boundaries(turns)
    if not bounds:
        return segs
    out: list[dict[str, Any]] = []
    for seg in segs:
        words, ends = seg["_words"], seg["_ends"]
        cuts: list[int] = []
        for b in bounds:
            if not seg["start"] < b < seg["end"] or len(words) < 2:
                continue
            # Each word sits on the side its midpoint falls on; clamp so both
            # sides keep a word even when the boundary grazes the segment edge.
            i = sum(1 for w, e in zip(words, ends) if 0.5 * (w["t"] + e) < b)
            cuts.append(min(max(i, 1), len(words) - 1))
        prev = 0
        for i in sorted(set(cuts)):
            if i > prev:
                out.append(_make(words[prev:i], ends[prev:i]))
                prev = i
        out.append(_make(words[prev:], ends[prev:]) if prev else seg)
    return out


def _merge_stubs(segs: list[dict[str, Any]], src: str = "he",
                 tgt: str = "en") -> list[dict[str, Any]]:
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
            def _allow(j: int) -> float:
                # A lone word — or a fragment too short for TTS to voice at
                # all — may cross a dramatic pause, but only to rejoin a
                # substantial neighbour: two stranded stubs re-merging would
                # undo a real pause split.
                tgt_seg = out[j]
                substantial = (tgt_seg["end"] - tgt_seg["start"] >= MIN_LEN
                               or len(tgt_seg["_words"]) >= MIN_WORDS)
                micro = (len(seg["_words"]) == 1
                         or seg["end"] - seg["start"] < MIN_SEG_SEC)
                return LONE_WORD_GAP if micro and substantial else MERGE_GAP
            cands = [c for c in cands if c[0] <= _allow(c[1])]
            if not cands:
                continue
            _gap, j = min(cands)
            a, b = (j, i) if j < i else (i, j)
            if out[b]["_words"][0].get("brk"):
                continue   # never merge across an explicit caption speaker change
            sa = text_bucket(out[a]["text"], src, tgt)
            sb = text_bucket(out[b]["text"], src, tgt)
            if sa and sb and sa != sb:
                continue   # nor across a change of language
            merged = _make(out[a]["_words"] + out[b]["_words"], out[a]["_ends"] + out[b]["_ends"])
            if merged["end"] - merged["start"] > MAX_LEN:
                continue
            out[a : b + 1] = [merged]
            changed = True
            break
    return out


def _embedded_target(words: list[dict[str, Any]], src: str = "he",
                     tgt: str = "en") -> list[bool]:
    """Mark target-script words that are short embedded tokens, not a passage.

    A lone acronym or brand dropped into source speech (`...מממן את ISIS, את...`)
    is written in the target script but spoken by the same person in the same
    breath. Treating it as a script change would carve it into its own segment and
    play a fraction of a second of original audio — an audible voice jump. Only a
    sustained run of target-script words is a real target-language passage;
    anything shorter stays with its neighbours and is dubbed like the rest of the
    sentence. For a same-script pair no word reads as target, so nothing is marked.
    """
    n = len(words)
    latin = [text_bucket(w["text"], src, tgt) == "target" for w in words]
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


def words_to_segments(words: list[dict[str, Any]], src: str = "he",
                      tgt: str = "en",
                      turns: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    if not words:
        return []
    ends = word_ends(words)
    # When the transcript marks speaker changes itself those markers are exact,
    # so diarization gets no vote on boundaries — its edges land a word or two
    # off and would cut sentences in half. Without markers (ASR fallback) a
    # diarization change still splits, but only when a real pause confirms it.
    has_markers = any(w.get("brk") for w in words)
    embedded = _embedded_target(words, src, tgt)
    groups: list[list[int]] = [[0]]
    bucket = None if embedded[0] else text_bucket(words[0]["text"], src, tgt)
    for i in range(1, len(words)):
        w, prev = words[i], words[i - 1]
        gap = speech_gap(prev, w)
        # A change of script is a change of language, and the two halves get
        # opposite treatment — one dubbed, one kept as original audio — so they
        # can never share a segment. An embedded token (a short target-script run)
        # is not a language change and does not set or break the script context.
        this_bucket = None if embedded[i] else text_bucket(w["text"], src, tgt)
        script_change = this_bucket is not None and bucket is not None and this_bucket != bucket
        if this_bucket is not None:
            bucket = this_bucket
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
    # Turn boundaries split first, stub merging after: a split may leave a short
    # same-speaker fragment that then rejoins its own side, while a genuinely
    # short cross-speaker interjection stays its own (still dubbed) segment.
    if turns:
        segs = _split_speaker_turns(segs, turns)
    segs = _merge_stubs(segs, src, tgt)

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


def _residues(seg: dict[str, Any],
              spans: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """The stretches of a segment that no span covers, in order.

    A segment can straddle several spans — a speaker interrupted twice by English —
    and what is left of it is then more than one piece. Subtracting the spans one at
    a time keeps every piece rather than reasoning about the aggregate.
    """
    pieces = [(seg["start"], seg["end"])]
    for s in spans:
        cut: list[tuple[float, float]] = []
        for a, b in pieces:
            if s["end"] <= a or s["start"] >= b:
                cut.append((a, b))
                continue
            if a < s["start"]:
                cut.append((a, round(s["start"], 3)))
            if s["end"] < b:
                cut.append((round(s["end"], 3), b))
        pieces = cut
    return pieces


def splice_foreign_spans(segs: list[dict[str, Any]], spans: list[dict[str, Any]],
                         words: list[dict[str, Any]] | None = None,
                         src: str = "he", tgt: str = "en") -> list[dict[str, Any]]:
    """Replace transcript segments inside target-language spans with the captions'.

    ASR in the source language either mangles target-language speech or skips it
    entirely, which would leave those seconds with no segment and therefore no
    audio at all. The captions know where that speech is and what it says, so
    those spans become segments in their own right.

    What the spans do *not* cover stays, in as many pieces as it takes. Judging a
    segment by its total overlap instead threw away speech: a Hebrew sentence with
    two English interjections inside it is more than half span by area, and dropping
    it whole silently lost the Hebrew in between ("כי הבעיה היא שבאחור" at 5:00).
    `words` supplies the text for a piece; without it a multi-piece segment can only
    be trimmed to its longest piece.
    """
    # A span with no words of its own cannot replace what it displaces, and
    # dropping segments for it would leave those seconds silent.
    spans = [s for s in spans if s.get("words")]
    if not spans:
        return segs

    ends = word_ends(words) if words else []
    kept: list[dict[str, Any]] = []
    for seg in segs:
        pieces = _residues(seg, spans)
        if pieces == [(seg["start"], seg["end"])]:
            kept.append(seg)              # untouched: kept however short
            continue
        if not pieces:
            continue                      # this segment is the captions' territory
        if words is None:                 # no words to redistribute — keep the longest
            pieces = [max(pieces, key=lambda p: p[1] - p[0])]
        for a, b in pieces:
            # Only a trimmed remnant may be discarded, and only when it is too short
            # to hold speech or holds none — otherwise its words vanish from the output.
            if b - a < SPLICE_MIN_REMNANT:
                continue
            piece = dict(seg, start=a, end=b)
            # A word belongs to this piece if it *overlaps* it. Neither edge rule works
            # alone: keeping only words that start inside drops one straddling the trim
            # ("היא צובעת את קטאר" lost its verb and was dubbed "Qatar in colors…"),
            # while keeping the segment's whole text drags in what the source-language
            # ASR hallucinated over the target-language speech inside the span, which is
            # nonsense that then fails translation and airs the original instead.
            if words is not None:
                said = [w["text"] for w, end in zip(words, ends)
                        if end > a + WORD_OVERLAP and w["t"] < b - WORD_OVERLAP]
                if not said:
                    continue
                piece["text"] = " ".join(said)
            kept.append(piece)

    for s in spans:
        words = [{"t": w["t"], "text": w["text"], "brk": False, "spk": w.get("spk", "SPEAKER_00")}
                 for w in s.get("words") or []]
        made = words_to_segments(words, src, tgt) if words else []
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
        # A third-language span remembers its language: the translate stage needs
        # it to render the subtitle for these keep-original segments.
        if s.get("lang") and s["lang"] != tgt:
            for x in made:
                x["lang"] = s["lang"]
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


def mark_keep(segments: list[dict[str, Any]], spans: list[dict[str, Any]] | None = None,
              target: str = "en", src: str = "he", dub_foreign: bool = False) -> None:
    """Flag segments whose original audio should play instead of a dub.

    Three content-free rules: the segment came out of a detected non-source-language
    span, its text is already in the target script, or its speaker is predominantly a
    target-language speaker (auto-captions often render their speech phonetically in
    the source script, which the per-segment test alone would miss).

    The span rule has to be structural. A span in a language no ASR here reads carries
    no text at all, and judging it by script would file it as transcript noise and drop
    it — airing nothing where somebody is speaking.

    When source and target share a script, the two script rules are meaningless and
    stay off; only the span rule (fed by the LID path upstream) marks keeps then.

    `dub_foreign` (opt-in) sends a *confident* third-language span segment down the
    dub path instead: its language is known (not "und") and its transcription is real
    (not the "…" placeholder), so the translate stage can render it in the target and
    TTS can voice it from the segment's own audio. Anything less confident keeps its
    original audio exactly as before — never silent.
    """
    def letters(text: str) -> int:
        return sum(1 for ch in (text or "") if ch.isalpha())

    cross = not script.same_script(src, target)
    # Test-locked legacy value: the target bucket is called "latin" for Latin-script
    # targets; any other script names itself honestly.
    target_reason = "latin" if script.script_for(target) == "latin" else "target_lang"

    ranges = [(s["start"], s["end"], s.get("lang")) for s in (spans or []) if s.get("words")]

    def span_lang(seg: dict[str, Any]) -> str | None:
        """The language of the span this segment came out of, or None if it did not."""
        for a, b, lang in ranges:
            if a - 0.05 <= seg["start"] and seg["end"] <= b + 0.05:
                return lang or ""
        return None
    totals: dict[str, list[float]] = {}
    for seg in segments:
        if letters(seg["text"]) < 2:
            continue
        dur = seg["end"] - seg["start"]
        agg = totals.setdefault(seg["speaker"], [0.0, 0.0])
        agg[0] += dur
        if cross and script.is_script(seg["text"], target):
            agg[1] += dur
    en_speakers = {
        spk for spk, (total, lat) in totals.items() if total > 0 and lat / total >= SPEAKER_EN_RATIO
    }
    for seg in segments:
        lang = span_lang(seg)
        if lang is not None:
            # Not the source language, whatever it is: play it as it was recorded. The
            # span's own language names it — a target-language line that happens to
            # carry no letters ("330,000") is still the target language, not a third one.
            in_target = (lang == target
                         or (cross and script.is_script(seg["text"], target)))
            if (dub_foreign and not in_target and lang and lang != "und"
                    and (seg.get("text") or "").strip() not in ("", "…")):
                # Opted in, and the span is confident: known language, real words.
                # This one is dubbable — translate reads seg["lang"] as its source.
                seg["keep"], seg["keep_reason"] = False, None
                continue
            seg["keep"] = True
            seg["keep_reason"] = target_reason if in_target else "foreign"
        elif letters(seg["text"]) < 2:
            # Transcript noise (stray glyphs). Nothing to translate, so let the
            # original audio through rather than leaving a hole.
            seg["keep"], seg["keep_reason"] = True, "no_text"
        elif cross and script.is_script(seg["text"], target):
            seg["keep"], seg["keep_reason"] = True, target_reason
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
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda:0"))
        elif torch.backends.mps.is_available():
            pipeline.to(torch.device("mps"))
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
                           total: float, is_target_lang=None, voice_levels=None) -> None:
    """Keep original audio for audible stretches no segment covers — target only.

    A region the language detector mislabels as neither source nor target, and the
    ASR never transcribed, otherwise plays silent — the speaker vanishes (the "1:35
    goes quiet" gap). Cover it with original audio so it is at least heard.

    But original audio is only safe to play where it is the *target* language: a
    mislabelled source-language stretch must never air its own voice (the "I can
    hear the Hebrew speaker" bleed). `is_target_lang(a, b)` runs VoxLingua on the
    stretch and answers that question; without it (no LID model) nothing is filled,
    so an uncertain region falls to the background bed rather than risk the bleed.
    """
    import numpy as np

    if is_target_lang is None:
        return
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
        # An uncovered stretch opens the moment the transcript stops, but the source
        # speaker does not stop there — Whisper's word ends bunch early, so the last
        # syllable is still sounding. Playing original audio from that instant airs it:
        # the "ה" of "…באומנות רבה" was audible at 1:34 under the dub that had just
        # said the same sentence. Start at the first pause instead, if one comes soon.
        # This one question — has the voice stopped? — is the one thing the vocals
        # stem answers better than the mix, where music never falls below the floor.
        # Audibility above still reads the mix; a stem that wrongly hears silence only
        # costs a moment of original audio, it cannot drop speech.
        head = i
        while voice_levels is not None and head < j and (head - i) * hop < UNCOVERED_HEAD_MAX:
            if head >= len(voice_levels):
                break
            if voice_levels[head] < KEEP_TAIL_FLOOR:
                # One frame past the pause's leading edge: these levels are RMS over
                # the whole frame, so the first quiet frame can still open on the
                # decaying tail of the last syllable — which is the part that was
                # audible. A tenth of a second of an untranscribed gap costs nothing.
                a = min((head + 1) * hop, b)
                break
            head += 1
        if (b - a >= UNCOVERED_MIN and float(np.max(levels[int(a / hop):j])) >= UNCOVERED_PEAK
                and is_target_lang(a, b)):
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
        spans: list[dict[str, Any]] | None = None, dub_foreign: bool = False) -> None:
    src_lang = m["source"].get("src_lang") or "he"
    tgt_lang = m["source"].get("tgt_lang") or "en"
    turns = diarize(workdir / m["files"]["vocals"])
    assign_word_speakers(words, turns)
    segs = words_to_segments(words, src_lang, tgt_lang, turns=turns)
    if spans:
        span_words = [w for s in spans for w in (s.get("words") or [])]
        assign_word_speakers(span_words, turns)
        segs = splice_foreign_spans(segs, spans, words, src_lang, tgt_lang)
        for i, seg in enumerate(segs):
            seg["id"] = i
    mark_keep(segs, spans, tgt_lang, src_lang, dub_foreign)

    # Drop sub-word noise fragments (a lone "ב", stray glyphs). Kept as original
    # audio they play a jarring one-letter blip of the source voice between dubbed
    # lines; letting those seconds fall to the background bed is far smoother. Their
    # (sub-two-letter) words carry no meaning, so nothing is lost.
    segs = [s for s in segs if s.get("keep_reason") != "no_text"]
    for i, seg in enumerate(segs):
        seg["id"] = i

    lost = [w for w in unsegmented_words(words, segs, spans or [])
            if sum(1 for ch in w["text"] if ch.isalpha()) >= 2]
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
    # Audible stretches no segment covers: keep the original audio only where it is
    # the target language (VoxLingua confirms), so a missed English line is heard
    # again without a mislabelled Hebrew region airing its source voice. Audibility
    # is judged from the source mix, not the vocals — Demucs sometimes routes a
    # speaker into the music stem, and that speech must still count. Falls back to
    # no filling when the LID model is absent.
    src = workdir / m["files"]["source_wav"]
    src_levels = audio.frame_rms(audio.decode_mono(src, 16000), 16000, 0.1)
    from . import transcript

    lid = transcript.load_lid()
    is_target = None
    if lid is not None:
        tgt = tgt_lang

        def is_target(a: float, b: float) -> bool:
            clip = audio.decode_mono(src, 16000, start=a,
                                     end=min(b, a + transcript.LID_WINDOW))
            lang, prob = transcript.detect_language(lid, clip)
            return lang == tgt and prob >= transcript.LID_MIN_PROB

    fill_uncovered_audible(segs, src_levels, 0.1, total or len(src_levels) * 0.1,
                           is_target_lang=is_target, voice_levels=levels)
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
