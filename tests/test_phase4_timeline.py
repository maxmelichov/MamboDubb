"""Phase 4 clips are anchored to their own segment's end, not the next start.

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

    # Gently slowing to fill the segment's *own* span is still wanted that
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
    # 3.0/4.0 is below RATE_MIN, so the slow-down floors there and the clip
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

    # A mild overrun the preferred rate can absorb is compressed onto the
    # speaker's own end rather than left to sit in the tail. The tail is where a
    # clip that does not fit goes, not what a clip is aimed at: aiming at it put
    # every dub's end a measurable fraction of a second after its speaker's.
    items = [item(0, 0.0, 3.4, end=3.0, speaker="A"),
             item(1, 10.0, 1.0, speaker="A")]
    places = timeline.place(items)
    assert places[0]["rate"] == pytest.approx(3.4 / 3.0, abs=1e-3)  # 1.133 < RATE_PREF
    assert places[0]["end"] == pytest.approx(3.0, abs=1e-3)

    # ...but only when it is affordable. 3.5s on a 3s span needs 1.167x, past
    # RATE_PREF, and 3.5s still lands inside own + TAIL_MAX, so the clip keeps
    # rate 1.0 and spends the tail rather than being squeezed that hard.
    items = [item(0, 0.0, 3.5, end=3.0, speaker="A"),
             item(1, 10.0, 1.0, speaker="A")]
    places = timeline.place(items)
    assert 3.5 / 3.0 > timeline.RATE_PREF and 3.5 < 3.0 + timeline.TAIL_MAX
    assert places[0]["rate"] == pytest.approx(1.0)
    assert places[0]["end"] == pytest.approx(3.5, abs=1e-3)


# --------------------------------------------------------------- (d) speaker change

def test_speaker_change_gets_no_deliberate_tail():
    # Same 3.9s-on-3s clip as above, but the next segment is a different
    # speaker: no tail allowance compress to end at the segment's own end.
    items = [item(0, 0.0, 3.9, end=3.0, speaker="A"),
             item(1, 10.0, 1.0, speaker="B")]
    places = timeline.place(items)
    assert places[0]["rate"] == pytest.approx(3.9 / 3.0)  # within RATE_MAX
    assert places[0]["end"] <= own_end(items[0]) + 1e-3


def test_speaker_change_prefers_pushing_next_over_overlap():
    # Even RATE_MAX cannot fit 5s of dub into a 3s segment. The remainder
    # spills, the next (different-speaker) segment is pushed later, and the
    # placements never overlap nothing is truncated.
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
    # timeline before the segment, the clip now starts early instead the wall
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
    # 3.6s of dub on a 3s segment, next speaker at 3.2: a 1.2x squeeze well
    # under RATE_MAX lands the clip at its own end, inside the wall, with no
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

    # An unstretchable clip against a wall cannot compress either it must
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
    # second 1.9s late. The second still fits inside its own wall the gap
    # before speaker C has slack so the lateness dies there: C starts
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


# --------------------------------------------------- media end as the last wall

def test_media_end_walls_the_last_clip():
    """The end of the media is a wall like a speaker change: past it the mux has
    no video, so audio there is lost rather than merely late."""
    item = {"id": 0, "source_start": 5.0, "source_end": 8.0, "dur": 4.0,
            "clip": "c.wav", "speaker": "A", "stretchable": True}
    free = timeline.place([dict(item)])
    walled = timeline.place([dict(item)], media_end=8.0)
    assert free[0]["end"] > 8.0 + 1e-3          # unbounded: runs past the video
    assert walled[0]["end"] <= 8.0 + 1e-3       # walled: pulled earlier/compressed
    assert walled[0]["overrun"] == 0.0


def test_media_end_default_is_unbounded():
    """Omitting media_end must reproduce the previous behaviour exactly."""
    items = [{"id": 0, "source_start": 1.0, "source_end": 3.0, "dur": 2.5,
              "clip": "a.wav", "speaker": "A", "stretchable": True},
             {"id": 1, "source_start": 4.0, "source_end": 6.0, "dur": 1.5,
              "clip": "b.wav", "speaker": "B", "stretchable": True}]
    assert (timeline.place([dict(i) for i in items])
            == timeline.place([dict(i) for i in items], None, float("inf")))


def test_media_end_never_truncates_audio():
    """A clip too long for the remaining media still overruns never cut."""
    item = {"id": 0, "source_start": 9.0, "source_end": 10.0, "dur": 6.0,
            "clip": "c.wav", "speaker": "A", "stretchable": True}
    p = timeline.place([dict(item)], media_end=10.0)[0]
    held = 6.0 / p["rate"]
    assert abs((p["end"] - p["start"]) - held) < 0.02


# ----------------------------------------- (f) the shorten budget is in speech units

def cjk_overrunning_manifest(tgt="ja"):
    """A dub far too long for its slot, in a language written without spaces."""
    from dubbing import manifest

    m = manifest.new({"input": "x", "src_lang": "ja", "tgt_lang": tgt})
    m["source"]["duration"] = 30.0
    m["segments"] = [
        {"id": 0, "start": 0.0, "end": 2.0, "speaker": "S0", "keep": False,
         "keep_reason": None, "text": "首相は…",
         "text_en": "首相は今日の記者会見で新しい経済政策を発表しました。",
         "tts": {"clip": "clips/a.wav", "dur": 9.0}},
        {"id": 1, "start": 2.0, "end": 4.0, "speaker": "S1", "keep": False,
         "keep_reason": None, "text": "市場は…", "text_en": "市場はこれを歓迎した。",
         "tts": {"clip": "clips/b.wav", "dur": 1.0}},
    ]
    manifest.ensure_uids(m["segments"])
    return m


def test_shorten_budget_for_a_cjk_target_is_asked_for_in_characters(monkeypatch,
                                                                    tmp_path):
    # `.split()` counts a whole Japanese line as one word, so the budget was
    # `max(3, int(1 * ratio))` = 3 for every segment — and `shorten`, measuring
    # the same way, then found every rewrite too long. This stage's only rescue
    # for a late line was permanently dead for zh/ja/ko.
    asked = {}
    monkeypatch.setattr(timeline.audio, "atempo", lambda *a, **k: None)
    monkeypatch.setattr(timeline.audio, "duration", lambda p: 7.0)
    m = cjk_overrunning_manifest()

    def shorten_many(reqs):
        asked.update({s["id"]: n for s, n in reqs})
        return {}

    timeline.run(m, tmp_path, shorten_many=shorten_many, resynth_many=lambda x: {})
    # 24 characters of Japanese, scaled by the slot ratio — not the floor of 3.
    assert asked and asked[0] > 3


def test_shorten_budget_for_a_latin_target_is_unchanged(monkeypatch, tmp_path):
    asked = {}
    monkeypatch.setattr(timeline.audio, "atempo", lambda *a, **k: None)
    monkeypatch.setattr(timeline.audio, "duration", lambda p: 7.0)
    m = cjk_overrunning_manifest(tgt="en")
    m["segments"][0]["text_en"] = "one two three four five six seven eight nine ten"

    def shorten_many(reqs):
        asked.update({s["id"]: n for s, n in reqs})
        return {}

    timeline.run(m, tmp_path, shorten_many=shorten_many, resynth_many=lambda x: {})
    assert asked and asked[0] <= 10          # a fraction of the line's ten words


def test_a_segments_own_target_language_decides_its_budget(monkeypatch, tmp_path):
    # The editor's per-segment `tgt_lang` override is the language the line is
    # actually in; the run's pair is only the default.
    asked = {}
    monkeypatch.setattr(timeline.audio, "atempo", lambda *a, **k: None)
    monkeypatch.setattr(timeline.audio, "duration", lambda p: 7.0)
    m = cjk_overrunning_manifest(tgt="en")
    m["segments"][0]["tgt_lang"] = "ja"

    def shorten_many(reqs):
        asked.update({s["id"]: n for s, n in reqs})
        return {}

    timeline.run(m, tmp_path, shorten_many=shorten_many, resynth_many=lambda x: {})
    assert asked and asked[0] > 3            # measured as Japanese, not as one word


# ------------------------- (g) placement is anchored on measured speech, not the
#                               ASR boundary

def spoken(i, start, dur, *, end=None, speech=None, speaker="S0", stretchable=True):
    """An item that also carries the speech bounds `speech_anchors` would find."""
    it = item(i, start, dur, end=end, speaker=speaker, stretchable=stretchable)
    if speech is not None:
        it["speech_start"], it["speech_end"] = speech
    return it


def test_a_clip_waits_for_the_speaker_not_for_the_segment_boundary():
    # The boundary says 10.0 and the speaker is measured to start at 10.25. Placing
    # on the boundary starts the dub a quarter of a second before the mouth moves,
    # which is the systematic error the demo runs showed on every segment of a run.
    items = [spoken(0, 10.0, 2.0, end=13.0, speech=(10.25, 13.0))]
    p = timeline.place(items)[0]
    assert p["start"] == pytest.approx(10.25)
    assert p["drift"] == pytest.approx(0.0)     # on time, against the speaker


def test_without_measured_speech_nothing_moves():
    # Every run made before this, and any run whose vocals cannot be read, must
    # place exactly where it placed before.
    items = [item(0, 10.0, 2.0, end=13.0)]
    assert timeline.place(items) == timeline.place([spoken(0, 10.0, 2.0, end=13.0)])


def test_a_clip_is_fitted_to_the_span_the_speaker_used():
    # 3.0s of dub over a boundary span of 3.0s is a fit; over the 2.0s the speaker
    # actually spent talking it is not, and pretending otherwise is how a dub keeps
    # going after the shot has moved on.
    boundary = timeline.place([item(0, 10.0, 3.0, end=13.0),
                               item(1, 20.0, 1.0, speaker="S1")])[0]
    measured = timeline.place([spoken(0, 10.0, 3.0, end=13.0, speech=(10.0, 12.0)),
                               spoken(1, 20.0, 1.0, speech=(20.0, 21.0), speaker="S1")])[0]
    assert boundary["rate"] == pytest.approx(1.0)
    assert measured["rate"] > 1.0
    assert measured["end"] <= 12.0 + timeline.TAIL_MAX + 1e-6


def test_the_anchors_wait_is_given_back_before_the_clip_overruns():
    # Waiting for the onset costs slot. When a line is already too long for its
    # slot that wait is the difference between fitting and pushing the whole scene
    # late, so it is the first thing given up — down to the boundary, no further.
    items = [spoken(0, 10.0, 3.4, end=12.6, speech=(10.4, 12.6)),
             spoken(1, 13.0, 1.0, speech=(13.0, 14.0), speaker="S1")]
    p = timeline.place(items)[0]
    assert 10.0 - 1e-6 <= p["start"] < 10.4
    assert p["overrun"] == pytest.approx(0.0)
    timeline.assert_invariants(timeline.place(items), items)


def test_anchored_placements_keep_every_invariant():
    rnd = random.Random(11)
    items, t = [], 0.0
    for i in range(40):
        span = rnd.uniform(0.8, 4.0)
        lead = rnd.uniform(0.0, timeline.ANCHOR_MAX)
        items.append(spoken(i, t, rnd.uniform(0.3, 6.0), end=t + span,
                            speech=(t + lead, t + span),
                            speaker=f"S{rnd.randint(0, 2)}"))
        t += span + rnd.uniform(0.0, 2.5)
    places = timeline.place(items)
    timeline.assert_invariants(places, items)


# ------------------------- (h) reading the anchors off the vocals stem

def write_tone(path, spans, *, dur, sr=44100):
    import numpy as np
    import soundfile as sf
    a = np.zeros(int(dur * sr), dtype="float32")
    for s, e in spans:
        n = int((e - s) * sr)
        a[int(s * sr):int(s * sr) + n] = 0.4 * np.sin(
            2 * np.pi * 220 * np.arange(n) / sr).astype("float32")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), a, sr)


def anchored_manifest(tmp_path, spans, segments):
    from dubbing import manifest
    write_tone(tmp_path / "stems/vocals.wav", spans, dur=30.0)
    m = manifest.new({"input": "x", "src_lang": "he", "tgt_lang": "en"})
    m["files"]["vocals"] = "stems/vocals.wav"
    m["segments"] = segments
    manifest.ensure_uids(m["segments"])
    return m


def seg(i, start, end, speaker="S0"):
    return {"id": i, "start": start, "end": end, "speaker": speaker, "keep": False,
            "keep_reason": None, "text": "x", "text_en": "x",
            "tts": {"clip": f"clips/{i}.wav", "dur": 1.0}}


def test_speech_anchors_find_the_onset_inside_a_padded_segment(tmp_path):
    m = anchored_manifest(tmp_path, [(5.3, 7.0)], [seg(0, 5.0, 7.2)])
    assert timeline.speech_anchors(m, tmp_path)[0][0] == pytest.approx(5.3, abs=0.05)


def test_speech_anchors_never_move_further_than_anchor_max(tmp_path):
    # A reading that disagrees with the boundary by more than ANCHOR_MAX is more
    # likely a bad reading than a bad boundary, and either way one clip is not
    # allowed to move that far on the strength of an energy envelope.
    m = anchored_manifest(tmp_path, [(9.0, 11.0)], [seg(0, 6.0, 11.0)])
    onset, _ = timeline.speech_anchors(m, tmp_path)[0]
    assert onset == pytest.approx(6.0 + timeline.ANCHOR_MAX, abs=0.02)


def test_speech_anchors_do_not_read_the_previous_line_as_this_ones_onset(tmp_path):
    # Segments 0.08s apart are ordinary; a fixed pad would hear the tail of
    # segment 0 and report segment 1 as starting before it does.
    m = anchored_manifest(tmp_path, [(3.0, 6.0), (6.4, 9.0)],
                          [seg(0, 3.0, 6.08), seg(1, 6.16, 9.0)])
    assert timeline.speech_anchors(m, tmp_path)[1][0] >= 6.08 - 1e-6


def test_a_silent_window_yields_no_anchor_and_the_boundary_stands(tmp_path):
    m = anchored_manifest(tmp_path, [], [seg(0, 3.0, 6.0)])
    assert timeline.speech_anchors(m, tmp_path) == {}


def test_no_vocals_stem_is_not_an_error(tmp_path):
    from dubbing import manifest
    m = manifest.new({"input": "x", "src_lang": "he", "tgt_lang": "en"})
    m["segments"] = [seg(0, 3.0, 6.0)]
    assert timeline.speech_anchors(m, tmp_path) == {}


# ------------------------------- (i) the tail is a fallback, not a target

def test_rate_aims_at_the_speakers_own_end_when_that_is_affordable():
    # A clip 10% longer than its speaker's span, with a whole TAIL_MAX of slack
    # available: it is compressed onto the speaker's end, because the 1.1x that
    # costs is under RATE_PREF. It used to be left alone to sit in the tail.
    assert timeline.rate_for(3.3, 10.0, 0.0, True, own=3.0,
                             tail=timeline.TAIL_MAX) == pytest.approx(1.1)


def test_a_clip_the_preferred_rate_cannot_reach_still_spends_the_tail():
    # 3.5s on a 3s span needs 1.167x, past RATE_PREF, and it lands inside
    # own + TAIL_MAX. Squeezing that hard to save half a second of overhang is a
    # worse trade than the overhang, so the clip keeps rate 1.0.
    assert 3.5 / 3.0 > timeline.RATE_PREF and 3.5 < 3.0 + timeline.TAIL_MAX
    assert timeline.rate_for(3.5, 10.0, 0.0, True, own=3.0,
                             tail=timeline.TAIL_MAX) == pytest.approx(1.0)


def test_the_tail_is_still_the_ceiling_for_a_clip_that_cannot_be_fitted():
    # Nothing above may let a clip past `own + tail`: too long for the preferred
    # rate AND too long for the tail is still compressed toward own + tail.
    rate = timeline.rate_for(6.0, 10.0, 0.0, True, own=3.0, tail=timeline.TAIL_MAX)
    assert rate == pytest.approx(min(timeline.RATE_MAX, 6.0 / 3.6))


def test_no_tail_across_a_speaker_change_is_unchanged():
    # tail=0 makes "affordable" and "the old behaviour" the same target, so a
    # speaker change is untouched by any of this.
    assert timeline.rate_for(3.3, 10.0, 0.0, True, own=3.0,
                             tail=0.0) == pytest.approx(1.1)


# ------------------- (j) a line pinned at the ceiling asks to be shortened

def pinned_manifest():
    """One line far too long for its own speaker, with a long pause after it.

    Nothing is waiting on it, so it never drifts however badly it fits: this is
    the shape the DRIFT_MAX trigger cannot see.
    """
    from dubbing import manifest

    m = manifest.new({"input": "x", "src_lang": "en", "tgt_lang": "ru"})
    m["source"]["duration"] = 60.0
    m["segments"] = [
        {"id": 0, "start": 0.0, "end": 3.0, "speaker": "S0", "keep": False,
         "keep_reason": None, "text": "a", "text_en": " ".join(["word"] * 12),
         "tts": {"clip": "clips/a.wav", "dur": 6.0}},
        {"id": 1, "start": 30.0, "end": 32.0, "speaker": "S0", "keep": False,
         "keep_reason": None, "text": "b", "text_en": "short", 
         "tts": {"clip": "clips/b.wav", "dur": 1.0}},
    ]
    manifest.ensure_uids(m["segments"])
    return m


def test_a_line_pinned_at_the_ceiling_asks_to_be_shortened_without_drifting(
        monkeypatch, tmp_path):
    monkeypatch.setattr(timeline.audio, "atempo", lambda *a, **k: None)
    monkeypatch.setattr(timeline.audio, "duration", lambda p: 4.6)
    m = pinned_manifest()
    asked = {}

    places = timeline.place(timeline.build_items(m), media_end=60.0)
    # The premise: it is pinned, it overhangs, and it is not late.
    assert places[0]["rate"] == pytest.approx(timeline.RATE_MAX)
    assert places[0]["end"] - 3.0 > timeline.OVERHANG_MAX
    assert places[0]["drift"] <= timeline.DRIFT_MAX

    def shorten_many(reqs):
        asked.update({s["id"]: n for s, n in reqs})
        return {}

    timeline.run(m, tmp_path, shorten_many=shorten_many, resynth_many=lambda x: {})
    assert list(asked) == [0]           # the pinned line, and only it


def test_a_line_that_fits_its_speaker_is_never_asked_to_shorten(monkeypatch,
                                                                tmp_path):
    monkeypatch.setattr(timeline.audio, "atempo", lambda *a, **k: None)
    monkeypatch.setattr(timeline.audio, "duration", lambda p: 3.0)
    m = pinned_manifest()
    m["segments"][0]["tts"]["dur"] = 3.0          # exactly its speaker's span
    asked = {}

    def shorten_many(reqs):
        asked.update({s["id"]: n for s, n in reqs})
        return {}

    timeline.run(m, tmp_path, shorten_many=shorten_many, resynth_many=lambda x: {})
    assert asked == {}


def test_the_shorten_budget_for_a_pinned_line_is_its_own_span_not_the_gap(
        monkeypatch, tmp_path):
    # The lateness trigger budgets against the next speaker's onset, which here
    # is 30s away and would ask for no shortening at all. A pinned line is
    # budgeted against its own 3s span instead.
    monkeypatch.setattr(timeline.audio, "atempo", lambda *a, **k: None)
    monkeypatch.setattr(timeline.audio, "duration", lambda p: 4.6)
    m = pinned_manifest()
    asked = {}

    def shorten_many(reqs):
        asked.update({s["id"]: n for s, n in reqs})
        return {}

    timeline.run(m, tmp_path, shorten_many=shorten_many, resynth_many=lambda x: {})
    # 12 words, and a 3s span at RATE_PREF holds about 3.45/6.0 of a 6s clip.
    assert asked[0] < 12
