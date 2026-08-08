"""Phase 4 — clips are anchored to their own segment's end, not the next start.

The fitter used to slow a short dub toward the *next* segment's start, so a
clip whose speaker had long stopped kept talking into the silence (and, across
a speaker change, into the next speaker's opening). These tests pin the new
policy: never stretch past the segment's own end, cap deliberate overhang at
TAIL_MAX into a same-speaker gap and at zero across a speaker change, and keep
the drift machinery (push the next segment later rather than truncate).
"""

from __future__ import annotations

import random

import pytest

from dubbing import timeline


def item(i, start, dur, *, end=None, speaker="S0", stretchable=True):
    return {"id": i, "source_start": start, "source_end": end if end is not None else start + dur,
            "dur": dur, "speaker": speaker, "stretchable": stretchable}


def own_end(it):
    return it["source_end"]


# --------------------------------------------------------------- (a) no fill-forward

def test_short_clip_is_not_stretched_past_its_own_end():
    # 2s of dub on a 4s segment, with 16s of open timeline before the next
    # segment: the old fitter slowed it to RATE_MIN to fill forward. It must
    # now end by its own segment's end, whatever room the slot has.
    items = [item(0, 0.0, 2.0, end=4.0), item(1, 20.0, 1.0)]
    places = timeline.place(items)
    assert places[0]["end"] <= own_end(items[0]) + 1e-6
    assert places[0]["rate"] >= timeline.RATE_MIN

    # Gently slowing to fill the segment's *own* span is still wanted — that
    # fights dead air without outliving the speaker.
    items = [item(0, 0.0, 3.5, end=4.0), item(1, 20.0, 1.0)]
    places = timeline.place(items)
    assert places[0]["rate"] == pytest.approx(3.5 / 4.0)
    assert places[0]["end"] == pytest.approx(4.0, abs=1e-3)


def test_last_segment_is_not_slowed_to_fill_infinity():
    # With no next segment the slot is unbounded; the final clip must still be
    # anchored to its own end, not drawled at RATE_MIN into the credits.
    items = [item(0, 0.0, 3.0, end=4.0)]
    places = timeline.place(items)
    # 3.0/4.0 is below RATE_MIN, so the slow-down floors there — and the clip
    # still ends inside its own segment, not at some point in the open tail.
    assert places[0]["rate"] == pytest.approx(timeline.RATE_MIN)
    assert places[0]["end"] <= own_end(items[0]) + 1e-6


# --------------------------------------------------------------- (b) exact fit

def test_exact_fit_is_untouched():
    items = [item(0, 0.0, 4.0, end=4.0), item(1, 20.0, 1.0)]
    places = timeline.place(items)
    assert places[0]["rate"] == pytest.approx(1.0)
    assert places[0]["start"] == 0.0
    assert places[0]["end"] == pytest.approx(4.0, abs=1e-3)


# --------------------------------------------------------------- (c) same-speaker tail

def test_long_clip_spills_at_most_tail_max_into_same_speaker_gap():
    # 3.9s of dub on a 3s segment, same speaker next after a long gap: the
    # clip may run into the gap, but no more than TAIL_MAX past its own end.
    items = [item(0, 0.0, 3.9, end=3.0, speaker="A"),
             item(1, 10.0, 1.0, speaker="A")]
    places = timeline.place(items)
    assert places[0]["end"] <= own_end(items[0]) + timeline.TAIL_MAX + 1e-3
    assert places[0]["rate"] <= timeline.RATE_MAX + 1e-9

    # A mild overrun that already lands inside the tail allowance is neither
    # squeezed nor filled out.
    items = [item(0, 0.0, 3.4, end=3.0, speaker="A"),
             item(1, 10.0, 1.0, speaker="A")]
    places = timeline.place(items)
    assert places[0]["rate"] == pytest.approx(1.0)
    assert places[0]["end"] == pytest.approx(3.4, abs=1e-3)


# --------------------------------------------------------------- (d) speaker change

def test_speaker_change_gets_no_deliberate_tail():
    # Same 3.9s-on-3s clip as above, but the next segment is a different
    # speaker: no tail allowance — compress to end at the segment's own end.
    items = [item(0, 0.0, 3.9, end=3.0, speaker="A"),
             item(1, 10.0, 1.0, speaker="B")]
    places = timeline.place(items)
    assert places[0]["rate"] == pytest.approx(3.9 / 3.0)  # within RATE_MAX
    assert places[0]["end"] <= own_end(items[0]) + 1e-3


def test_speaker_change_prefers_pushing_next_over_overlap():
    # Even RATE_MAX cannot fit 5s of dub into a 3s segment. The remainder
    # spills, the next (different-speaker) segment is pushed later, and the
    # placements never overlap — nothing is truncated.
    items = [item(0, 0.0, 5.0, end=3.0, speaker="A"),
             item(1, 3.2, 1.0, end=4.2, speaker="B")]
    places = timeline.place(items)
    timeline.assert_invariants(places, items)
    assert places[0]["rate"] == pytest.approx(timeline.RATE_MAX)  # squeezed as hard as allowed
    assert places[1]["start"] >= places[0]["end"] - 1e-6          # no overlap: next is pushed
    assert places[1]["drift"] > 0                                  # ...i.e. it drifts later
    held = places[0]["end"] - places[0]["start"]
    assert held == pytest.approx(5.0 / timeline.RATE_MAX, abs=2e-3)  # never truncated


# --------------------------------------------------------------- (e) drift machinery

def random_items(rng, n=40):
    items, t = [], 0.0
    for i in range(n):
        span = rng.uniform(0.8, 8.0)
        items.append(item(i, round(t, 3), round(span * rng.uniform(0.6, 1.8), 3),
                          end=round(t + span, 3),
                          speaker=rng.choice(["A", "B", "C"]),
                          stretchable=rng.random() > 0.2))
        t += span + rng.choice([0.0, 0.05, 0.3, 1.0, 4.0])
    return items


def test_drift_bounds_and_invariants_still_hold():
    rng = random.Random(4242)
    for _ in range(200):
        items = random_items(rng)
        places = timeline.place(items)
        timeline.assert_invariants(places, items)
        for it, p in zip(items, places):
            assert p["rate"] <= timeline.RATE_MAX + 1e-9
            assert p["rate"] >= timeline.RATE_MIN - 1e-9
            # Anchor property: a clip no longer than its own span never ends
            # past its (drift-shifted) segment end.
            if it["dur"] <= it["source_end"] - it["source_start"]:
                assert p["end"] <= it["source_end"] + p["drift"] + 1e-3


# --------------------------------------------------------------- (f) cross-speaker wall

def test_cross_speaker_wall_respected_via_earlier_start():
    # 4.5s of dub on a 3s segment: even RATE_MAX leaves 3.46s of audio, which
    # used to spill 0.26s past the next speaker's onset at 8.2. With 5s of free
    # timeline before the segment, the clip now starts early instead — the wall
    # holds, and the next speaker begins exactly on time (no cascade).
    items = [item(0, 5.0, 4.5, end=8.0, speaker="A"),
             item(1, 8.2, 1.0, speaker="B")]
    places = timeline.place(items)
    timeline.assert_invariants(places, items)
    assert places[0]["start"] < 5.0                                # pulled earlier
    assert places[0]["start"] >= 5.0 - timeline.LEAD_MAX - 1e-3    # ...but boundedly
    assert places[0]["end"] <= items[1]["source_start"] + 1e-3     # wall holds
    assert places[0]["overrun"] == 0.0
    assert places[1]["start"] == pytest.approx(8.2)
    assert places[1]["drift"] == 0.0


def test_cross_speaker_wall_respected_via_compression_within_cap():
    # 3.6s of dub on a 3s segment, next speaker at 3.2: a 1.2x squeeze — well
    # under RATE_MAX — lands the clip at its own end, inside the wall, with no
    # early start needed.
    items = [item(0, 0.0, 3.6, end=3.0, speaker="A"),
             item(1, 3.2, 1.0, speaker="B")]
    places = timeline.place(items)
    timeline.assert_invariants(places, items)
    assert places[0]["start"] == 0.0
    assert places[0]["rate"] == pytest.approx(3.6 / 3.0)
    assert places[0]["end"] <= items[1]["source_start"] + 1e-3
    assert places[0]["overrun"] == 0.0
    assert places[1]["drift"] == 0.0


def test_impossible_fit_overruns_without_truncation_and_is_recorded():
    # 5s of dub on a 2s segment with the next speaker at 2.2 and no timeline
    # before it: neither an earlier start nor RATE_MAX can fit. The clip keeps
    # every millisecond of its audio, the next segment is pushed (no overlap),
    # and the unavoidable overrun is measured on the placement.
    items = [item(0, 0.0, 5.0, end=2.0, speaker="A"),
             item(1, 2.2, 1.0, speaker="B")]
    places = timeline.place(items)
    timeline.assert_invariants(places, items)
    assert places[0]["rate"] == pytest.approx(timeline.RATE_MAX)
    held = places[0]["end"] - places[0]["start"]
    assert held == pytest.approx(5.0 / timeline.RATE_MAX, abs=2e-3)  # never truncated
    assert places[1]["start"] >= places[0]["end"] - 1e-6             # never overlapped
    assert places[0]["overrun"] == pytest.approx(
        places[0]["end"] - items[1]["source_start"], abs=2e-3)

    # An unstretchable clip against a wall cannot compress either — it must
    # still place cleanly, with the overrun recorded rather than hidden.
    items = [item(0, 0.0, 3.0, end=2.0, speaker="A", stretchable=False),
             item(1, 2.3, 1.0, speaker="B")]
    places = timeline.place(items)
    timeline.assert_invariants(places, items)
    assert places[0]["rate"] == pytest.approx(1.0)
    assert places[0]["overrun"] == pytest.approx(0.7, abs=2e-3)


def test_late_start_past_the_wall_still_compresses_to_rate_max():
    # A predecessor's overrun pushes this clip to start after its own wall.
    # The old fitter saw "no slot" and squeezed only to RATE_PREF; now the
    # wall escalates straight to RATE_MAX so the talk-over is as short as it
    # can possibly be.
    items = [item(0, 0.0, 4.0, end=2.0, speaker="A", stretchable=False),
             item(1, 2.1, 0.8, end=2.9, speaker="B"),
             item(2, 3.0, 1.0, speaker="C")]
    places = timeline.place(items)
    timeline.assert_invariants(places, items)
    assert places[1]["rate"] == pytest.approx(timeline.RATE_MAX)
    assert places[1]["overrun"] > 0


def test_cascade_is_damped_at_a_wall_with_slack():
    # The first clip overruns hard (unstretchable, 2s over), pushing the
    # second 1.9s late. The second still fits inside its own wall — the gap
    # before speaker C has slack — so the lateness dies there: C starts
    # exactly on its source onset instead of inheriting the drift.
    items = [item(0, 0.0, 4.0, end=2.0, speaker="A", stretchable=False),
             item(1, 2.2, 2.3, end=4.2, speaker="B"),
             item(2, 7.0, 1.0, speaker="C")]
    places = timeline.place(items)
    timeline.assert_invariants(places, items)
    assert places[0]["overrun"] > 1.0                              # the trigger
    assert places[1]["drift"] > 1.0                                # pushed late...
    assert places[1]["end"] <= items[2]["source_start"] + 1e-3     # ...but held
    assert places[1]["overrun"] == 0.0
    assert places[2]["start"] == pytest.approx(7.0)
    assert places[2]["drift"] == 0.0


def test_same_speaker_tail_gets_no_lead_and_no_overrun():
    # Same-speaker behaviour is unchanged by the wall: a long clip may still
    # run past the next segment's start (TAIL_MAX policy plus drift), is never
    # pulled earlier, and records no cross-speaker overrun.
    items = [item(0, 0.0, 5.2, end=3.0, speaker="A"),
             item(1, 3.2, 1.0, speaker="A")]
    places = timeline.place(items)
    timeline.assert_invariants(places, items)
    assert places[0]["start"] == 0.0
    assert places[0]["rate"] == pytest.approx(timeline.RATE_MAX)
    assert places[0]["end"] > items[1]["source_start"]   # allowed: same speaker
    assert places[0]["overrun"] == 0.0


def test_drift_still_propagates_and_resets_at_a_pause():
    items = [item(0, 0.0, 5.0, end=2.0, speaker="A"),
             item(1, 2.0, 1.0, end=3.0, speaker="B"),
             item(2, 6.0, 1.0, speaker="B")]
    places = timeline.place(items)
    assert places[1]["drift"] > 0          # pushed late by its overlong predecessor
    assert places[2]["drift"] == 0.0       # the pause absorbed it
    assert places[2]["start"] == 6.0
