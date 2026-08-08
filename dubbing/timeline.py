"""Stage 7 — decide where every clip starts. This is the module that makes
cut-off speech structurally impossible.

One forward pass places each clip at its source onset, or as soon after as the
previous clip allows. Nothing is ever trimmed to fit: a clip that is too long
for its slot is gently time-compressed, one much shorter than its slot is gently
time-stretched (so it does not finish early and drift out of sync), and whatever
compression still does not fit becomes *drift* on the following segment. Drift is
bounded, and it disappears on its own at the next real pause, because a start is
`max(source_start, prev_end + gap)`.

Each clip is anchored to its *own* segment's end, not to the next segment's
start: stretching never pushes a clip past the moment its original speaker
stopped just because the timeline has room, and a clip that is genuinely too
long is compressed toward its own end, allowed to spill at most TAIL_MAX into
a following gap when the next segment is the same speaker — and not at all when
the speaker changes (or there is no next segment). A speaker change makes the
next segment's original onset a hard wall: a clip that would still be talking
when the other character visibly starts is first pulled earlier into free
timeline before it (at most LEAD_MAX, never overlapping the previous clip),
then compressed up to RATE_MAX. A clip that even then cannot fit still
overruns rather than being cut; the following segment is pushed later (drift)
and the overrun is measured and reported instead of masquerading as ordinary
drift.

Placed clips are separated by a short gap, except where the source itself ran
them together — a passage of original audio split into parts stays joined, since
inserting silence at each seam would both sound wrong and push the run late.

Invariants asserted at the end of the stage:
  * placements are strictly ordered and never overlap
  * no clip is shorter than the audio it holds (nothing is cut)
  * no clip starts more than LEAD_MAX before its source onset (and only ever
    early to duck under a cross-speaker wall)
  * every segment is placed exactly once
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

from . import audio

MIN_GAP = 0.10        # silence between two placed clips that the source separated
MIN_SEAM = 0.005      # …and between two that ran together: inaudible, but enough
                      # that millisecond rounding can never make them overlap
RATE_PREF = 1.15      # compression we are willing to apply without thinking
RATE_MAX = 1.30       # hard ceiling, only used when already running late
RATE_MIN = 0.82       # slowest we stretch a short dub to fill its slot (below this
                      # it drawls); keeps a much-shorter English line from finishing
                      # early and leaving a silent tail
DRIFT_SOFT = 0.50     # lateness that justifies escalating to RATE_MAX
DRIFT_MAX = 1.50      # lateness that justifies asking for a shorter translation
SHORTEN_ROUNDS = 2
TAIL_MAX = 0.60       # how far a too-long clip may run past its own segment's end
                      # into a following gap — only when the next segment is the
                      # same speaker; a speaker change (or the end of the video)
                      # allows no deliberate tail at all
LEAD_MAX = 0.60       # how far a clip squeezed against a cross-speaker wall may
                      # start before its own source onset, into free timeline —
                      # a moment of early speech beats talking over the next
                      # speaker's opening


def rate_for(dur: float, slot: float, drift_in: float, stretchable: bool,
             own: float | None = None, tail: float = 0.0) -> float:
    """How much to speed a clip up so it fits `slot`, within policy limits.

    `own` is the segment's own span (source_end - source_start): stretching a
    short clip fills at most that far, never the whole slot — the dub should
    stop when the original speaker stopped, not when the next one starts. A
    clip longer than its own span is compressed toward `own + tail`, where
    `tail` is the deliberate overhang allowed past the segment's end (TAIL_MAX
    into a same-speaker gap, 0.0 across a speaker change). `own=None` keeps
    the pure fit-to-slot behaviour.
    """
    if not stretchable or dur <= 0.05:
        return 1.0
    if slot <= 0.05:
        # Back-to-back source segments leave no slot at all; only compress when
        # we are already behind and need to claw time back.
        return RATE_PREF if drift_in > DRIFT_SOFT else 1.0
    # How far a slow-down may fill, and how far the clip may reach at all.
    # Both are still capped by the slot: the next placement is never overlapped.
    stretch_to = slot if own is None else min(slot, max(own, 0.05))
    fit_to = slot if own is None else min(slot, max(own + tail, 0.05))
    if dur <= stretch_to:
        # The dub is shorter than the span its speaker actually used: play it at
        # 1.0 and it finishes early, leaving a silent tail. Stretch it to fill
        # that span, but never below RATE_MIN (a drawl is worse than a little
        # silence) — and never past the segment's own end just because the
        # timeline has room before the next segment.
        return max(RATE_MIN, dur / stretch_to)
    if dur <= fit_to:
        # Ends within the allowed tail: neither filled out nor squeezed.
        return 1.0
    need = dur / fit_to
    rate = min(RATE_PREF, need)
    if rate < need:
        # RATE_PREF is not enough to hit the target. Escalate to RATE_MAX when
        # the shortfall would push the next segment late (classic drift), or
        # when it would overhang past the allowed tail beyond the segment's
        # own end (talking over the next speaker's opening).
        over_slot = dur / rate - slot
        over_cap = dur / rate - fit_to
        if over_slot > DRIFT_SOFT or (own is not None and over_cap > 1e-6):
            rate = min(RATE_MAX, need)
    return rate


def place(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pure forward placement. `items` must be sorted by source_start."""
    out: list[dict[str, Any]] = []
    prev_end = -math.inf
    prev_source_end = -math.inf
    for i, it in enumerate(items):
        # Never insert more silence than the source had. Segments that run
        # straight into each other — a passage of original audio split into
        # parts, say — must stay joined, or every boundary would add a gap and
        # the whole run would slide late.
        need = min(MIN_GAP, max(MIN_SEAM, it["source_start"] - prev_source_end))
        start = max(it["source_start"], prev_end + need)
        nxt = items[i + 1]["source_start"] if i + 1 < len(items) else math.inf
        next_need = min(MIN_GAP, max(MIN_SEAM, nxt - it["source_end"]))
        slot = nxt - next_need - start
        drift = start - it["source_start"]
        # A too-long clip may run TAIL_MAX past its own end into a following
        # gap when the next segment is the same speaker; a speaker change (or
        # the end of the video) gets no deliberate tail — compress harder, and
        # let any unavoidable remainder push the next segment later (drift)
        # rather than talk over its speaker's opening.
        same_speaker = (i + 1 < len(items)
                        and items[i + 1].get("speaker") == it.get("speaker"))
        tail = TAIL_MAX if same_speaker else 0.0
        own = it["source_end"] - it["source_start"]
        rate = rate_for(it["dur"], slot, drift, it.get("stretchable", False),
                        own=own, tail=tail)
        end = start + it["dur"] / rate
        # A speaker change makes the next segment's original onset a hard wall:
        # the dub must not still be talking in voice A when character B visibly
        # starts. First pull the clip earlier into free timeline before it
        # (bounded by LEAD_MAX and the previous placement, so nothing overlaps),
        # then compress up to RATE_MAX. Whatever still does not fit overruns —
        # audio is never cut — but the overrun is measured and returned rather
        # than masquerading as ordinary drift, and because each clip is fitted
        # against its own wall, one clip's lateness never propagates past the
        # next wall that has slack to absorb it.
        wall = nxt - next_need
        overrun = 0.0
        if i + 1 < len(items) and not same_speaker and end > wall + 1e-6:
            floor = max(0.0, prev_end + need)
            lead = min(LEAD_MAX, start - floor, end - wall)
            if lead > 1e-6:
                start -= lead
                end -= lead
            if end > wall + 1e-6 and it.get("stretchable") and it["dur"] > 0.05:
                allowed = wall - start
                want = it["dur"] / allowed if allowed > 0.05 else math.inf
                rate = max(rate, min(RATE_MAX, want))
                end = start + it["dur"] / rate
            # What matters on screen is talking past the next speaker's actual
            # onset; ending inside the seam gap just before it costs nothing.
            overrun = max(0.0, end - nxt)
        out.append({"id": it["id"], "start": round(start, 3), "end": round(end, 3),
                    "rate": round(rate, 4),
                    "drift": round(start - it["source_start"], 3),
                    "overrun": round(overrun, 3)})
        prev_end = end
        prev_source_end = it["source_end"]
    return out


def assert_invariants(places: list[dict], items: list[dict]) -> None:
    assert len(places) == len(items), "every segment must be placed exactly once"
    for a, b in zip(places, places[1:]):
        assert b["start"] >= a["end"] - 1e-3, (
            f"placements overlap: seg {a['id']} ends {a['end']:.3f}, "
            f"seg {b['id']} starts {b['start']:.3f}"
        )
    for p, it in zip(places, items):
        assert p["start"] >= it["source_start"] - LEAD_MAX - 1e-3, (
            f"seg {p['id']} starts more than LEAD_MAX before source"
        )
        held = it["dur"] / p["rate"]
        assert abs((p["end"] - p["start"]) - held) < 0.02, (
            f"seg {p['id']} would be truncated: slot {p['end'] - p['start']:.3f}s "
            f"for {held:.3f}s of audio"
        )


def _worst_overrunner(items: list[dict], places: list[dict], idx: int) -> int | None:
    """The segment in this run that is pushing segment `idx` late."""
    best, best_overrun = None, 0.0
    i = idx - 1
    while i >= 0:
        overrun = places[i]["end"] - items[i + 1]["source_start"]
        if items[i].get("stretchable") and not items[i].get("shortened") and overrun > best_overrun:
            best, best_overrun = i, overrun
        if places[i]["drift"] <= 1e-6:
            break
        i -= 1
    return best


def build_items(m: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for seg in sorted(m["segments"], key=lambda s: s["start"]):
        if seg["keep"]:
            # Kept audio is the original span, so its clip must be exactly that
            # long. Anything else means the wrong audio is about to be placed.
            span = seg["end"] - seg["start"]
            assert abs(float(seg["tts"]["dur"]) - span) < 0.15, (
                f"seg {seg['id']}: kept clip is {seg['tts']['dur']:.2f}s "
                f"but its source span is {span:.2f}s"
            )
        items.append({
            "id": seg["id"],
            "source_start": float(seg["start"]),
            "source_end": float(seg["end"]),
            "dur": float(seg["tts"]["dur"]),
            "clip": seg["tts"]["clip"],
            "speaker": seg.get("speaker"),
            "stretchable": not seg["keep"],
        })
    return items


def run(m: dict[str, Any], workdir: Path, *, shorten_many=None, resynth_many=None) -> None:
    by_id = {s["id"]: s for s in m["segments"]}
    items = build_items(m)
    by_index = {it["id"]: i for i, it in enumerate(items)}
    spoken: dict[int, str] = {}   # segment id -> shortened line actually voiced
    places = place(items)

    for _round in range(SHORTEN_ROUNDS):
        late = [i for i, p in enumerate(places) if p["drift"] > DRIFT_MAX]
        if not late or shorten_many is None or resynth_many is None:
            break
        # Collect the whole round's requests before calling out, so the translator
        # and the synthesiser are each loaded once rather than swapped per segment.
        requests: dict[int, int] = {}
        for idx in sorted(late, key=lambda i: -places[i]["drift"]):
            j = _worst_overrunner(items, places, idx)
            if j is None or items[j]["id"] in requests:
                continue
            slot = items[j + 1]["source_start"] - places[j]["start"]
            budget = max(0.5, slot) * RATE_PREF
            ratio = max(0.5, min(0.95, budget / max(items[j]["dur"], 0.1)))
            words = len((by_id[items[j]["id"]].get("text_en") or "").split())
            requests[items[j]["id"]] = max(3, int(words * ratio))
            items[j]["shortened"] = True   # attempted at most once per segment
        if not requests:
            break
        texts = shorten_many([(by_id[i], n) for i, n in requests.items()]) or {}
        texts = {i: t for i, t in texts.items() if t}
        if not texts:
            break
        records = resynth_many([(by_id[i], t) for i, t in texts.items()]) or {}
        changed = False
        for seg_id, record in records.items():
            if not record:
                continue
            seg, it = by_id[seg_id], items[by_index[seg_id]]
            before = len(seg["text_en"].split())
            # Kept local to this stage: `text_en` belongs to the translator, and
            # rewriting it would make the next run shorten an already-short line.
            spoken[seg_id] = texts[seg_id]
            it["dur"], it["clip"] = float(record["dur"]), record["clip"]
            changed = True
            print(f"  timeline: shortened seg {seg_id} {before}→"
                  f"{len(texts[seg_id].split())} words", file=sys.stderr)
        if not changed:
            break
        places = place(items)

    # Apply the planned tempo change, then re-place using the *measured* result.
    # Re-measuring is what keeps planned and real durations from diverging —
    # that divergence is how the previous pipeline produced overlaps.
    for it, p in zip(items, places):
        raw = workdir / it["clip"]
        # Bake the tempo change into the clip — both a speed-up (too long for its
        # slot) and a slow-down (a much-shorter English line stretched toward the
        # source duration so it does not finish early and drift out of sync). Only
        # rate ~1.0 is left untouched. The re-place below then runs at rate 1.0 on
        # the fitted clip, so its measured length is what gets placed.
        if abs(p["rate"] - 1.0) > 0.01:
            fitted = workdir / "clips" / f"fit_{Path(it['clip']).stem}_{p['rate']:.3f}.wav"
            if not fitted.is_file():
                audio.atempo(raw, fitted, p["rate"])
            it["clip"] = f"clips/{fitted.name}"
            it["dur"] = audio.duration(fitted)
            it["applied_rate"] = p["rate"]
        it["stretchable"] = False

    final = place(items)
    assert_invariants(final, items)

    for it, p in zip(items, final):
        seg = by_id[it["id"]]
        seg["place"] = {"start": p["start"], "end": p["end"],
                        "rate": round(it.get("applied_rate", 1.0), 4),
                        "drift": p["drift"], "clip": it["clip"]}
        if it["id"] in spoken:
            seg["place"]["spoken"] = spoken[it["id"]]

    drifts = [p["drift"] for p in final]
    print(f"  timeline: {len(final)} placed, max drift {max(drifts):.2f}s, "
          f"{sum(1 for d in drifts if d > DRIFT_SOFT)} over {DRIFT_SOFT}s", file=sys.stderr)
    overruns = [p["overrun"] for p in final if p["overrun"] > 1e-3]
    if overruns:
        print(f"  timeline: {len(overruns)} cross-speaker overrun(s), "
              f"worst {max(overruns):.2f}s", file=sys.stderr)
