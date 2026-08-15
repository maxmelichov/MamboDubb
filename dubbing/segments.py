"""Stage 4 group the word stream into speakable segments.

The splitting and merging rules are pure functions over the word list, so they
unit-test without audio; only `diarize` and `refine_turns` touch a model, and
`run` the disk. Segments are built from word timings only; a segment never spans
a speaker change, so later stages can treat `segment.speaker` as exact.

Two parts of the stage live next door and are re-exported below: `dubbing.keep`
(which segments play their original audio, and the user's passthrough override)
and `dubbing.interjections` (the lexicon the movie-mode keep rule consults).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

from . import script

# The keep verdict and the interjection lexicon it consults now live in their own
# modules. Re-exported here because the pipeline, both front ends and the tests all
# reach them as `segments.<name>`, and that spelling is not worth churning.
from .interjections import (  # noqa: F401  (re-export)
    INTERJECTION_MAX_SEC,
    INTERJECTION_MAX_WORDS,
    _is_interjection_token,
    is_interjection_keep,
)
from .keep import (  # noqa: F401  (re-export)
    CARRY_MIN_OVERLAP,
    DETECT_MIN_COVER,
    SPEAKER_EN_RATIO,
    SPEAKER_EN_VETO_PROB,
    apply_passthrough,
    carry_passthrough,
    mark_keep,
    saved_overrides,
    stamp_detected_lang,
)

GAP_SPLIT = 0.70       # silence between words that ends a segment
SPEAKER_GAP = 0.25     # diarization must be backed by a real pause to split
MAX_LEN = 10.0         # target segment length; stubs never merge past it
# A group only reaches the length splitter if it holds no interior sentence end,
# i.e. it is a single sentence. Splitting one mid-clause makes both halves
# translate badly (each fragment is ambiguous) and gives them separate, clashing
# TTS prosody the audible "jump". Qwen voices ~13s cleanly, so a lone long
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

FOREIGN_MIN_WORDS = 3  # a shorter target-script run is an embedded token (acronym/
                       # brand) inside source speech, not a target-language passage

TURN_GAP_SPLIT = 0.8   # silence between two diarization turns that forces a split:
                       # pyannote's segmentation hears the pause even when clustering
                       # gives both turns the same label, while Whisper's word timings
                       # smear across it and would bridge two characters' lines

# A diarization turn shorter than this is not a speaker turn. Pyannote sprinkles
# sub-half-second turns of a second label through one person's speech (a breath, a
# louder syllable, a bar of music under the voice), and every one of them used to
# cut two segment boundaries into the middle of a sentence the fragments then
# carried the wrong speaker, never re-merged (a stub only merges into a
# same-speaker neighbour) and were voiced by a different clone. Nothing this short
# is dubbable on its own either (MIN_SEG_SEC is 0.9), so a real interjection of
# this length is better off spoken in the surrounding voice than stranded.
MIN_TURN_SEC = 0.40
# ...and how much silence may sit on either side of such a blip for it to still
# count as *inside* the surrounding speech. Kept under TURN_GAP_SPLIT so absorbing
# a blip can never hide a pause that would itself have forced a split.
ISLAND_GAP_MAX = 0.60

# How far a segment's start may move back to meet the diarization onset of its own
# speaker at a handoff. Whisper's first word timestamp after a speaker change lands
# late (the new voice's onset is inside the previous speaker's decaying tail, and
# the decoder anchors on the first clean frame), which delays the dub and for a
# kept segment delays the original audio against the picture. Pyannote's turn start
# is the better estimate of when the voice actually began.
HANDOFF_SNAP = 0.30

DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"

# What this stage records in `m["health"]` when it has to run degraded, and
# therefore what a successful run of it clears (see `run`). Read by `report.run`.
HEALTH_KEYS = ("segments.diarization", "segments.turn_refinement")

# Post-diarization refinement: pyannote's clustering sometimes files two
# alternating voices under one label (courtroom Q+A dubbed in a single clone),
# while an ECAPA speaker embedding separates them decisively. A same-speaker run
# of turns is re-clustered on those embeddings and split only when the evidence
# is unambiguous (see _split_embedding_clusters).
ECAPA_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_DIR = Path(__file__).resolve().parents[1] / "models" / "spkrec-ecapa-voxceleb"
REFINE_MIN_TURNS = 3     # fewer turns cannot show alternation, so never split
REFINE_MIN_TURN_SEC = 0.4  # measured: a 0.55s turn still embeds cleanly (0.33 to its cluster)
REFINE_SPLIT_DIST = 0.80   # dendrogram cut on cosine distance. Measured on the
                           # courtroom run: same-voice average linkage reaches 0.77
                           # (a short turn), cross-voice linkage sits at 0.88 —
                           # 0.80 clears the first with margin and refuses the second
REFINE_MAX_TURNS = 12      # a longer same-label run is one person holding the floor
                           # (a news read, a lecture) with pauses in it, not two
                           # people pyannote failed to tell apart. Cutting such a run
                           # in two renames half of the dominant speaker's turns for
                           # the rest of the video the worst possible outcome, since
                           # every downstream stage then hears two speakers where the
                           # viewer sees one. Two voices genuinely fused by clustering
                           # show up as a short alternating exchange, so the evidence
                           # this rule needs is always present within a dozen turns.
REFINE_MARGIN = 1.10       # the cross-cluster linkage must also beat the widest
                           # within-cluster merge by this factor: an absolute cut
                           # alone splits one voice recorded two ways (studio vs
                           # phone line) as readily as two voices. Measured
                           # courtroom: 0.88 against 0.77 * 1.10 = 0.85 still splits.


def text_bucket(text: str, src: str = "he", tgt: str = "en") -> str | None:
    """'target', 'source', or None when the letters name neither.

    Script is a cheap, high-precision language signal only when the pair is
    written differently. For a same-script pair (en→es) letters carry no signal
    at all, so the answer is always None and every script-based shortcut stays
    off the LID callback path decides instead.
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


def _speaker_blocks(turns: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Index ranges of maximal runs of consecutive same-speaker turns."""
    blocks: list[tuple[int, int]] = []
    i = 0
    while i < len(turns):
        j = i + 1
        while j < len(turns) and turns[j]["speaker"] == turns[i]["speaker"]:
            j += 1
        blocks.append((i, j))
        i = j
    return blocks


def smooth_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Absorb sub-MIN_TURN_SEC speaker islands into the speech around them. Pure.

    A majority vote with a minimum-duration constraint, in its simplest honest
    form: a block of turns labelled X that holds less than MIN_TURN_SEC of speech,
    sits between two blocks that are both Y and both longer than it, at least one
    of which is a real turn, and is separated from them by no more than
    ISLAND_GAP_MAX of silence on either side, is relabelled Y. The shortest island
    goes first, so a *flutter* measured: 0.54/0.02/0.02/0.02/0.34s alternating
    between two labels inside one sentence collapses from the inside out into a
    single turn instead of surviving as six. Nothing that short is a speaker turn;
    it is a clustering blip, and left alone it cuts the sentence around it into
    three segments, the middle one attributed and later voiced as somebody
    else. Timings are never touched, so the audio each word belongs to does not
    move; only the label does. A genuine short interruption from a field reporter
    or an interviewee runs seconds, not milliseconds, and is left exactly as
    diarized. Returns a new list; the caller's dicts are not mutated.
    """
    out = [dict(t) for t in sorted(turns, key=lambda t: (t["start"], t["end"]))]
    if len(out) < 3:
        return out

    def spoken(a: int, b: int) -> float:
        return sum(t["end"] - t["start"] for t in out[a:b])

    while True:
        best = None
        blocks = _speaker_blocks(out)
        for k in range(1, len(blocks) - 1):
            (pa, pb), (ia, ib), (na, nb) = blocks[k - 1], blocks[k], blocks[k + 1]
            host = out[pa]["speaker"]
            if out[na]["speaker"] != host:
                continue
            island, prev, nxt = spoken(ia, ib), spoken(pa, pb), spoken(na, nb)
            if island >= MIN_TURN_SEC:
                continue
            if min(prev, nxt) < island or max(prev, nxt) < MIN_TURN_SEC:
                continue
            before = out[ia]["start"] - out[pb - 1]["end"]
            after = out[na]["start"] - out[ib - 1]["end"]
            if max(before, after) > ISLAND_GAP_MAX:
                continue
            if best is None or island < best[0]:
                best = (island, ia, ib, host)
        if best is None:
            return out
        _island, ia, ib, host = best
        for t in out[ia:ib]:
            t["speaker"] = host


def real_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The turns long enough to be treated as speaker turns.

    Word labelling and boundary cutting must read the same view of diarization,
    or a turn too short to justify a cut still steals the label of the words
    inside it and splits the segment anyway (through the word-level speaker-change
    rule). If every turn is short a very quiet file, a degenerate diarization —
    the filter would leave nothing to attribute to, so the full list stands.
    """
    kept = [t for t in turns if t["end"] - t["start"] >= MIN_TURN_SEC]
    return kept or list(turns)


def assign_word_speakers(words: list[dict[str, Any]], turns: list[dict[str, Any]]) -> None:
    """Tag each word with the diarization turn covering it (in place)."""
    if not turns:
        for w in words:
            w["spk"] = "SPEAKER_00"
        return
    turns = sorted(real_turns(turns), key=lambda t: t["start"])
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


def _turn_boundaries(turns: list[dict[str, Any]]) -> list[tuple[float, bool]]:
    """(time, protect) pairs where diarization demands a segment cut.

    A speaker handover cuts at the incoming turn's start. A silence gap of at
    least TURN_GAP_SPLIT between consecutive turns same speaker or not cuts
    at the gap's midpoint, and that cut is *protected*: clustering often labels
    two different characters' turns with one speaker, so `_merge_stubs`'s
    same-speaker rule would quietly glue the halves back together. A protected
    cut plants a `brk` marker (the caption speaker-change mechanism) on the
    split point, which `_merge_stubs` never crosses.

    Handover cuts come from the turns long enough to be speaker turns
    (`real_turns`): a sub-MIN_TURN_SEC blip would otherwise cut twice, stranding
    a word or two of somebody's sentence as a separate, differently-voiced
    segment. Silence cuts still read *every* turn as diarized, so removing a blip
    from the handover view can never turn its speech into an apparent pause.
    """
    turns = sorted(turns, key=lambda t: t["start"])
    bounds: list[tuple[float, bool]] = []
    for a, b in zip(turns, turns[1:]):
        gap = b["start"] - a["end"]
        if gap >= TURN_GAP_SPLIT:
            bounds.append((round(a["end"] + gap / 2, 3), True))
    real = real_turns(turns)
    for a, b in zip(real, real[1:]):
        if a["speaker"] != b["speaker"]:
            bounds.append((b["start"], False))
    return bounds


def _split_speaker_turns(segs: list[dict[str, Any]],
                         turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split each segment at diarization speaker changes that fall inside it.

    Fast dialogue hands the floor over with no pause, so the gap-confirmed
    speaker split above misses it and two characters' turns fuse into one
    segment courtroom Q+A dubbed in a single cloned voice. The turn list
    knows where the handover happened; cut between the two words nearest it,
    with at least one word on each side, so the pieces stay non-overlapping
    and keep their word alignment. Inter-turn silence gaps cut the same way
    (see _turn_boundaries); a protected gap cut marks the first word after the
    cut with `brk` so `_merge_stubs` cannot re-fuse the pieces. With one
    speaker in back-to-back turns (or no turn data) there are no boundaries
    and nothing changes.
    """
    bounds = _turn_boundaries(turns)
    if not bounds:
        return segs
    out: list[dict[str, Any]] = []
    for seg in segs:
        words, ends = seg["_words"], seg["_ends"]
        cuts: dict[int, bool] = {}
        for b, protect in bounds:
            if not seg["start"] < b < seg["end"] or len(words) < 2:
                continue
            # Each word sits on the side its midpoint falls on; clamp so both
            # sides keep a word even when the boundary grazes the segment edge.
            i = sum(1 for w, e in zip(words, ends) if 0.5 * (w["t"] + e) < b)
            i = min(max(i, 1), len(words) - 1)
            cuts[i] = cuts.get(i, False) or protect
        if not cuts:
            out.append(seg)
            continue
        # Copy, not mutate: the word dicts are shared with the caller's list.
        ws = [dict(w, brk=True) if cuts.get(i) else w for i, w in enumerate(words)]
        prev = 0
        for i in sorted(cuts):
            out.append(_make(ws[prev:i], ends[prev:i]))
            prev = i
        out.append(_make(ws[prev:], ends[prev:]))
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
                # A lone word or a fragment too short for TTS to voice at
                # all may cross a dramatic pause, but only to rejoin a
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
    play a fraction of a second of original audio an audible voice jump. Only a
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
    # so diarization gets no vote on boundaries its edges land a word or two
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
        # opposite treatment one dubbed, one kept as original audio so they
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


def snap_speaker_handoffs(segs: list[dict[str, Any]],
                          turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pull a segment's start back to its speaker's diarization onset. Pure-ish.

    At a handoff the incoming voice starts inside the outgoing one's decaying
    tail, and Whisper times its first word from the first frame it hears cleanly —
    consistently late, by up to a few hundred milliseconds. That lateness is the
    dub's lateness: the clip is placed at `source_start`, so the new speaker
    starts talking after they visibly started, and a *kept* segment plays its
    original audio late against the picture.

    Diarization's own onset is the better estimate, so where a real turn of this
    segment's speaker begins shortly before it (within HANDOFF_SNAP) the start
    moves back to meet it. Only ever earlier, and never past the previous
    segment's end: a start that moved later would strand the speech in front of
    it, and placements must stay ordered and non-overlapping. Segments are
    modified in place and returned.
    """
    if not turns or not segs:
        return segs
    real = sorted(real_turns(turns), key=lambda t: t["start"])
    prev_end = 0.0
    prev_spk = None
    for seg in segs:
        start = seg["start"]
        if seg["speaker"] != prev_spk:
            floor = max(prev_end, start - HANDOFF_SNAP)
            best = None
            for t in real:
                if t["speaker"] != seg["speaker"] or t["end"] <= start:
                    continue
                if floor <= t["start"] < start - 0.02:
                    best = t["start"] if best is None else max(best, t["start"])
            if best is not None:
                seg["start"] = round(best, 3)
        prev_end, prev_spk = seg["end"], seg["speaker"]
    return segs


def _residues(seg: dict[str, Any],
              spans: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """The stretches of a segment that no span covers, in order.

    A segment can straddle several spans a speaker interrupted twice by English —
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
    trimmed: set[int] = set()      # id() of the pieces a span trim left behind
    kept: list[dict[str, Any]] = []
    for seg in segs:
        pieces = _residues(seg, spans)
        if pieces == [(seg["start"], seg["end"])]:
            kept.append(seg)              # untouched: kept however short
            continue
        if not pieces:
            continue                      # this segment is the captions' territory
        if words is None:                 # no words to redistribute keep the longest
            pieces = [max(pieces, key=lambda p: p[1] - p[0])]
        for a, b in pieces:
            # Only a trimmed remnant may be discarded, and only when it is too short
            # to hold speech or holds none otherwise its words vanish from the output.
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
            trimmed.add(id(piece))
            kept.append(piece)

    made_all: list[dict[str, Any]] = []
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
        # Every span segment also carries the classifier's label as an advisory
        # stamp including a target-language one, which `lang` deliberately does
        # not record (it means "third language" to the translate stage). The
        # editor app reads this to explain why a span plays original audio.
        for x in made:
            x["detected_lang"] = s.get("lang") or tgt
        made_all.extend(x for x in made if x["end"] > x["start"])
    kept.extend(made_all)
    out = sorted(kept, key=lambda x: x["start"])
    return _absorb_trim_remnants(out, trimmed, {id(x) for x in made_all}, src, tgt)


def _absorb_trim_remnants(segs: list[dict[str, Any]], trimmed: set[int],
                          from_span: set[int], src: str = "he",
                          tgt: str = "en") -> list[dict[str, Any]]:
    """Fold a stranded trim remnant back into the span segment it touches.

    A span edge lands where the *voice* stopped, but Whisper's last word before it
    smears seconds past that point, so subtracting the span leaves a remnant of one
    or two words butting straight up against the span's own segment the same
    person, mid-phrase ("The National Rabbinical Court against" | "the"). Left
    alone that remnant is a separate line: separately translated out of context,
    separately voiced, and separated on the timeline by a placement gap that the
    original speech never had.

    It rejoins only when rejoining cannot change what is played: the two are
    contiguous, share a speaker, the span is target-language (nothing else may
    play its own audio outside its own span), and the remnant's own script says
    target too so the merged span is one speaker saying one thing in one
    language. A same-script pair, where the script test proves nothing, never
    merges. `_merge_stubs` cannot do this job: it runs before the splice exists,
    and these two pieces come from different sources.
    """
    out = list(segs)
    i = 0
    while i < len(out):
        seg = out[i]
        if id(seg) not in trimmed:
            i += 1
            continue
        short = (seg["end"] - seg["start"] < MIN_LEN
                 or len((seg.get("text") or "").split()) < MIN_WORDS)
        if not short or text_bucket(seg.get("text", ""), src, tgt) != "target":
            i += 1
            continue
        host_i = None
        for j in (i - 1, i + 1):
            if not 0 <= j < len(out) or id(out[j]) not in from_span:
                continue
            host = out[j]
            if host.get("lang") or host.get("speaker") != seg.get("speaker"):
                continue      # a third-language span, or somebody else's line
            gap = seg["start"] - host["end"] if j < i else host["start"] - seg["end"]
            if abs(gap) <= 0.05:
                host_i = j
                break
        if host_i is None:
            i += 1
            continue
        host = out[host_i]
        first, second = (host, seg) if host_i < i else (seg, host)
        host["start"] = min(host["start"], seg["start"])
        host["end"] = max(host["end"], seg["end"])
        host["text"] = " ".join(t for t in (first.get("text"), second.get("text")) if t)
        del out[i]
    return out


def unsegmented_words(words: list[dict[str, Any]], segs: list[dict[str, Any]],
                      spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Transcript words that no segment covers.

    A segment is the only route audio has to the output, so a word left out here
    is a word the viewer never hears. Words inside a target-language span are
    exempt: those were deliberately replaced by the captions' own words.
    """
    ranges = sorted((s["start"], s["end"]) for s in segs)
    # Span edges are approximate they come from caption word onsets so a word
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


def diarize(vocals: Path, *, note: Callable[[str], None] | None = None
            ) -> list[dict[str, Any]]:
    """Pyannote turns; returns [] (single-speaker fallback) if unavailable.

    The fallback is a real verdict about the run every speaker becomes one
    voice, so every line is cloned from one reference and the commonest cause
    (no `HF_TOKEN`, or the model's terms not accepted) is a five-second fix the
    user never hears about if this only prints. `note` records the reason where
    `report.run` can repeat it; without one, behaviour is exactly as before.
    """
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
        print(f"  segments: diarization unavailable ({exc}) single speaker", file=sys.stderr)
        if note is not None:
            note(f"unavailable ({exc}) every speaker cloned as one voice")
        return []


def _split_embedding_clusters(sims: Any) -> list[int] | None:
    """0/1 cluster labels per turn when a run holds exactly two alternating voices.

    `sims` is a square cosine-similarity matrix over one run's turn embeddings,
    in time order. Average-linkage agglomerative clustering on cosine distance,
    cut at REFINE_SPLIT_DIST: the run splits only when the dendrogram yields
    exactly two clusters there, each with at least two members, and the labels
    alternate at least once. A block pattern (AAABBB) has a single transition —
    that shape is more likely a scene change with one voice recorded differently
    than two people, so it never splits. Anything ambiguous returns None.
    """
    import numpy as np

    dist = 1.0 - np.asarray(sims, dtype=float)
    n = len(dist)
    if not REFINE_MIN_TURNS <= n <= REFINE_MAX_TURNS:
        return None                # too few turns to alternate, or too many to be
                                   # a fused two-person exchange at all
    clusters: list[list[int]] = [[i] for i in range(n)]

    def linkage(a: list[int], b: list[int]) -> float:
        return float(np.mean([dist[i, j] for i in a for j in b]))

    within = 0.0
    while len(clusters) > 2:
        best = min(
            ((linkage(clusters[x], clusters[y]), x, y)
             for x in range(len(clusters)) for y in range(x + 1, len(clusters))),
        )
        d, x, y = best
        if d > REFINE_SPLIT_DIST:
            return None            # the cut leaves more than two clusters
        within = max(within, d)
        clusters[x] += clusters[y]
        del clusters[y]
    cross = linkage(clusters[0], clusters[1])
    if cross <= max(REFINE_SPLIT_DIST, within * REFINE_MARGIN):
        # Either the cut leaves a single cluster, or the two "clusters" are no
        # further apart than one of them is wide one voice, two recordings.
        return None
    if min(len(c) for c in clusters) < 2:
        return None                # a lone outlier turn is not a second voice
    second = clusters[1] if 0 in clusters[0] else clusters[0]
    labels = [1 if i in second else 0 for i in range(n)]
    if sum(1 for a, b in zip(labels, labels[1:]) if a != b) < 2:
        return None                # block split, not an alternating dialogue
    return labels


_ECAPA = None
_ECAPA_FAILED = False


def _load_ecapa():
    """The ECAPA speaker-embedding model, loaded once; None if unavailable."""
    global _ECAPA, _ECAPA_FAILED
    if _ECAPA is not None or _ECAPA_FAILED:
        return _ECAPA
    try:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _ECAPA = EncoderClassifier.from_hparams(
            source=ECAPA_MODEL, savedir=str(ECAPA_DIR), run_opts={"device": device})
    except Exception as exc:
        _ECAPA_FAILED = True
        print(f"  segments: speaker embeddings unavailable ({exc}) "
              "turns left as diarized", file=sys.stderr)
    return _ECAPA


def _embed_turns(vocals: Path, run: list[dict[str, Any]]):
    """Unit-normalized ECAPA embedding per turn, or None if the model is absent."""
    model = _load_ecapa()
    if model is None:
        return None
    import numpy as np
    import torch

    from . import audio

    vecs = []
    for t in run:
        clip = audio.decode_mono(vocals, 16000, start=t["start"], end=t["end"])
        wav = torch.from_numpy(clip.astype("float32")).unsqueeze(0)
        with torch.no_grad():
            emb = model.encode_batch(wav.to(next(model.parameters()).device))
        v = emb.squeeze().cpu().numpy().astype(float)
        vecs.append(v / (np.linalg.norm(v) or 1.0))
    return np.stack(vecs)


def refine_turns(turns: list[dict[str, Any]], vocals: Path, *,
                 note: Callable[[str], None] | None = None) -> list[dict[str, Any]]:
    """Split diarization turns that ECAPA embeddings say hold two voices.

    Pyannote's clustering can label two alternating speakers as one (both sides
    of a courtroom exchange dubbed in a single clone). For every maximal run of
    one speaker's consecutive turns no other speaker between, only silence —
    with at least REFINE_MIN_TURNS turns each at least REFINE_MIN_TURN_SEC long,
    the turns are embedded and re-clustered; when exactly two alternating
    clusters emerge (see _split_embedding_clusters) the second cluster's turns
    get a fresh label (original + "b"), so splitting, speaker voting and clone
    reference selection all see two speakers. On any failure no turns, no
    vocals, model unavailable the turns come back unchanged; this never fails
    the stage.
    """
    if not turns:
        return turns
    ordered = sorted(turns, key=lambda t: (t["start"], t["end"]))
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1]["speaker"] == ordered[i]["speaker"]:
            j += 1
        run = ordered[i : j + 1]
        # Short turns embed as noise, but one of them must not veto the run
        # (a 0.55s interjection sat inside the measured courtroom exchange):
        # they are skipped for embedding and inherit a neighbour's verdict.
        long_enough = [t for t in run
                       if t["end"] - t["start"] >= REFINE_MIN_TURN_SEC]
        # A run longer than REFINE_MAX_TURNS is somebody holding the floor, not a
        # fused exchange (see the constant): skipped here so the embeddings are
        # never even computed for a news read's hundred turns.
        if REFINE_MIN_TURNS <= len(long_enough) <= REFINE_MAX_TURNS:
            runs.append((i, j + 1))
        i = j + 1
    if not runs:
        return ordered
    if not vocals or not Path(vocals).is_file():
        print(f"  segments: no vocals at {vocals} turns left as diarized",
              file=sys.stderr)
        if note is not None:
            note(f"unavailable: no vocals at {vocals} two voices pyannote fused "
                 "stay fused")
        return ordered
    out = [dict(t) for t in ordered]
    for a, b in runs:
        run = out[a:b]
        embeddable = [t for t in run
                      if t["end"] - t["start"] >= REFINE_MIN_TURN_SEC]
        try:
            embs = _embed_turns(vocals, embeddable)
            if embs is None:
                if note is not None:
                    note("unavailable: speaker embeddings could not load two "
                         "voices pyannote fused stay fused")
                return ordered
            labels = _split_embedding_clusters(embs @ embs.T)
        except Exception as exc:
            print(f"  segments: turn refinement failed ({exc}) "
                  "turns left as diarized", file=sys.stderr)
            if note is not None:
                note(f"failed ({exc}) turns left as diarized")
            return ordered
        if labels is None:
            continue
        spk = run[0]["speaker"]
        verdicts = dict(zip((id(t) for t in embeddable), labels))
        for t in run:
            lab = verdicts.get(id(t))
            if lab is None:
                # too short to embed: inherit the nearest embedded turn's side
                mid = (t["start"] + t["end"]) / 2
                near = min(embeddable, key=lambda e:
                           abs((e["start"] + e["end"]) / 2 - mid))
                lab = verdicts[id(near)]
            if lab:
                t["speaker"] = spk + "b"
        print(f"  segments: embeddings split {spk} into two alternating voices "
              f"({run[0]['start']:.1f}-{run[-1]['end']:.1f}s, {len(run)} turns)",
              file=sys.stderr)
    return out


def extend_keeps_to_speech_end(segs: list[dict[str, Any]], levels, hop: float,
                               total: float) -> None:
    """Grow each keep segment's end through trailing speech, up to the next segment.

    Keep segments play original audio. The source-language ASR stops transcribing
    before an English speaker actually stops (and Whisper's word-end timestamps
    bunch early), so the segment boundary lands mid-sentence and the tail is
    clipped. Walk the vocal energy forward from the boundary, reclaiming speech
    until a real pause or the next segment whichever comes first. The
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


def _judge_windows(a: float, b: float, win: float, is_target_lang) -> list[tuple[float, float]]:
    """The runs of [a, b] the language witness calls the target language.

    The witness answers for one classifier window at a time, so a long gap has to be
    walked window by window: asking once, at the gap's start, decides twenty seconds
    on their first four. A remainder shorter than half a window is judged together
    with the window before it rather than alone the classifier's verdicts on
    sub-second fragments are documented noise (a 0.9s piece came back Maori at p>=0.6).
    """
    runs: list[list[float]] = []
    t = a
    while t < b - 1e-9:
        c = min(b, t + win)
        if b - c < win / 2:
            c = b
        if is_target_lang(round(t, 3), round(c, 3)):
            if runs and abs(runs[-1][1] - t) < 1e-9:
                runs[-1][1] = c
            else:
                runs.append([t, c])
        t = c
    return [(x, y) for x, y in runs]


def fill_uncovered_audible(segs: list[dict[str, Any]], levels, hop: float,
                           total: float, is_target_lang=None, voice_levels=None,
                           win: float = 4.0) -> None:
    """Keep original audio for audible stretches no segment covers target only.

    A region the language detector mislabels as neither source nor target, and the
    ASR never transcribed, otherwise plays silent the speaker vanishes (the "1:35
    goes quiet" gap). Cover it with original audio so it is at least heard, as a
    real `spoken_target` keep segment the editor can see, select and correct.

    But original audio is only safe to play where it is the *target* language: a
    mislabelled source-language stretch must never air its own voice (the "I can
    hear the Hebrew speaker" bleed), and a music-only stretch kept here would play
    the mix on top of the bed that already carries it. `is_target_lang(a, b)` runs
    VoxLingua on one `win`-second window and answers that question; the gap is
    walked window by window, and only the target-language runs inside it are kept.
    Without the witness (no LID model) nothing is filled the mix's vocals fill is
    the floor there, so an unjudged region still plays the original voice.
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
        # speaker does not stop there Whisper's word ends bunch early, so the last
        # syllable is still sounding. Playing original audio from that instant airs it:
        # the "ה" of "…באומנות רבה" was audible at 1:34 under the dub that had just
        # said the same sentence. Start at the first pause instead, if one comes soon.
        # This one question has the voice stopped? is the one thing the vocals
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
                # decaying tail of the last syllable which is the part that was
                # audible. A tenth of a second of an untranscribed gap costs nothing.
                a = min((head + 1) * hop, b)
                break
            head += 1
        if b - a >= UNCOVERED_MIN and float(np.max(levels[int(a / hop):j])) >= UNCOVERED_PEAK:
            for wa, wb in _judge_windows(a, b, win, is_target_lang):
                if (wb - wa >= UNCOVERED_MIN
                        and float(np.max(levels[int(wa / hop):max(int(wa / hop) + 1,
                                                                 int(wb / hop))]))
                        >= UNCOVERED_PEAK):
                    added.append((round(wa, 3), round(wb, 3)))
        i = j
    for a, b in added:
        # `spoken_target`, not a bare "uncovered": the witness named this the target
        # language, which is why its original audio may play at all and the editor
        # can tell it apart from a stretch nobody could read.
        segs.append({"id": -1, "start": a, "end": b, "speaker": "SPEAKER_00",
                     "text": "", "keep": True, "keep_reason": "spoken_target"})
    if added:
        segs.sort(key=lambda s: s["start"])
        for k, s in enumerate(segs):
            s["id"] = k


def run(m: dict[str, Any], workdir: Path, words: list[dict[str, Any]],
        spans: list[dict[str, Any]] | None = None, dub_foreign: bool = False,
        genre: str = "documentary",
        overrides: list[tuple[float, float, bool]] | None = None,
        lang_runs: list[dict[str, Any]] | None = None) -> None:
    src_lang = m["source"].get("src_lang") or "he"
    tgt_lang = m["source"].get("tgt_lang") or "en"
    # What this run could not do is recorded, not only printed: a stage that
    # degrades itself is writing a verdict about every segment (one voice for the
    # whole file, say), and report.json is where that has to be visible. Cleared
    # first so a run that succeeds erases the previous run's excuse.
    health = m.setdefault("health", {})
    for key in HEALTH_KEYS:
        health.pop(key, None)

    def _note(key: str):
        return lambda reason: health.__setitem__(key, reason)

    turns = diarize(workdir / m["files"]["vocals"], note=_note("segments.diarization"))
    # Smooth before refining: the blips absorbed here are what chop one speaker's
    # long run into short ones, and every later decision (word labels, boundary
    # cuts, embedding runs) must read the same, smoothed view of the turns.
    smoothed = smooth_turns(turns)
    moved = sum(1 for a, b in zip(sorted(turns, key=lambda t: (t["start"], t["end"])),
                                  smoothed) if a["speaker"] != b["speaker"])
    if moved:
        print(f"  segments: absorbed {moved} sub-{MIN_TURN_SEC}s speaker blip(s)",
              file=sys.stderr)
    turns = refine_turns(smoothed, workdir / m["files"]["vocals"],
                         note=_note("segments.turn_refinement"))
    assign_word_speakers(words, turns)
    segs = words_to_segments(words, src_lang, tgt_lang, turns=turns)
    # Word timestamps run late after a handoff; diarization knows better where the
    # incoming voice started (see snap_speaker_handoffs).
    snap_speaker_handoffs(segs, turns)
    if spans:
        span_words = [w for s in spans for w in (s.get("words") or [])]
        assign_word_speakers(span_words, turns)
        segs = splice_foreign_spans(segs, spans, words, src_lang, tgt_lang)
        for i, seg in enumerate(segs):
            seg["id"] = i
    from . import audio, transcript

    src = workdir / m["files"]["source_wav"]
    lid = transcript.load_lid()

    def seg_lang(seg: dict[str, Any]) -> tuple[str, float] | None:
        """What the classifier hears in this one segment (None when it cannot ask).

        Asked from the source mix, like every other presence question: Demucs
        sometimes routes a speaker into the music stem. Only reached for segments
        the speaker-level prior would otherwise keep, so this costs one embedding
        pass per line of a mostly-target speaker, not one per segment in the file.
        """
        if lid is None:
            return None
        try:
            clip = audio.decode_mono(src, 16000, start=seg["start"],
                                     end=min(seg["end"], seg["start"] + transcript.LID_WINDOW))
            return transcript.detect_language(lid, clip)
        except Exception as exc:                                    # pragma: no cover
            print(f"  segments: per-segment language check failed ({exc})", file=sys.stderr)
            return None

    # The user's overrides survive a re-segmentation by time, not by id this
    # stage renumbers everything it rebuilds. Carried before the automatic rules
    # run so `apply_passthrough` below has the last word over them.
    if overrides:
        carried = carry_passthrough(segs, overrides)
        print(f"  segments: carried {carried}/{len(overrides)} passthrough override(s)",
              file=sys.stderr)
    mark_keep(segs, spans, tgt_lang, src_lang, dub_foreign, genre=genre,
              seg_lang=seg_lang if lid is not None else None)
    apply_passthrough(segs)

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
        f"{lost[0]['t']:.2f}s ({lost[0]['text']!r}) they would never be heard"
    )
    # Keep segments (original audio) must not clip a speaker mid-sentence.
    total = float(m["source"].get("duration") or 0.0)
    levels = audio.frame_rms(audio.decode_mono(workdir / m["files"]["vocals"], 16000),
                             16000, 0.1)
    extend_keeps_to_speech_end(segs, levels, 0.1, total or len(levels) * 0.1)
    # Audible stretches no segment covers: keep the original audio only where it is
    # the target language (VoxLingua confirms), so a missed English line is heard
    # again without a mislabelled Hebrew region airing its source voice. Audibility
    # is judged from the source mix, not the vocals Demucs sometimes routes a
    # speaker into the music stem, and that speech must still count. Falls back to
    # no filling when the LID model is absent.
    src_levels = audio.frame_rms(audio.decode_mono(src, 16000), 16000, 0.1)
    is_target = None
    if lid is not None:
        tgt = tgt_lang

        def is_target(a: float, b: float) -> bool:
            clip = audio.decode_mono(src, 16000, start=a,
                                     end=min(b, a + transcript.LID_WINDOW))
            lang, prob = transcript.detect_language(lid, clip)
            return lang == tgt and prob >= transcript.LID_MIN_PROB

    fill_uncovered_audible(segs, src_levels, 0.1, total or len(src_levels) * 0.1,
                           is_target_lang=is_target, voice_levels=levels,
                           win=transcript.LID_WINDOW)
    # Advisory only: every segment that exists now including the ones the fill
    # just added gets the classifier's label for the app to read.
    stamp_detected_lang(segs, lang_runs)
    # Identity is minted once, here, after every split/splice/fill has settled —
    # `id` is positional and renumbered above, so nothing outside the pipeline can
    # key on it. See `manifest.mint_uid`.
    from . import manifest

    manifest.ensure_uids(segs)
    m["segments"] = segs
    m["speakers"] = {
        spk: {"dur": round(sum(s["end"] - s["start"] for s in segs if s["speaker"] == spk), 2)}
        for spk in sorted({s["speaker"] for s in segs})
    }
    kept = sum(1 for s in segs if s["keep"])
    forced = sum(1 for s in segs if s.get("passthrough") is not None)
    print(
        f"  segments: {len(segs)} segments, {len(m['speakers'])} speakers, "
        f"{kept} keep-original"
        f"{f', {forced} user-set' if forced else ''}",
        file=sys.stderr,
    )
