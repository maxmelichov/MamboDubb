"""--genre movie: mode-dependent timeline rates + rate continuity, interjection
keeps in segments, and the movie prompt block in translate. The default genre
("documentary") must be exactly current behavior everywhere — several tests here
are regression pins for that.
"""

from __future__ import annotations

import pytest

from dubbing import segments, timeline, translate


def item(i, start, dur, *, end=None, speaker="S0", stretchable=True):
    return {"id": i, "source_start": start,
            "source_end": end if end is not None else start + dur,
            "dur": dur, "speaker": speaker, "stretchable": stretchable}


# ------------------------------------------------------------------ rates_for_genre

def test_documentary_rates_are_exactly_the_module_constants():
    r = timeline.rates_for_genre("documentary")
    assert r.rate_pref == timeline.RATE_PREF
    assert r.rate_max == timeline.RATE_MAX
    assert r.rate_min == timeline.RATE_MIN
    assert r.tail_max == timeline.TAIL_MAX
    # And the default argument path is the same object contents.
    assert timeline.rates_for_genre() == r


def test_documentary_place_is_byte_identical_to_default_place():
    items = [item(0, 0.0, 5.0, end=2.0, speaker="A"),
             item(1, 2.2, 1.0, speaker="B"),
             item(2, 6.0, 0.4, end=2.0, speaker="B"),
             item(3, 9.0, 3.1, end=11.0, speaker="A")]
    assert timeline.place(items) == timeline.place(
        items, timeline.rates_for_genre("documentary"))


def test_movie_rates_lower_the_caps_and_raise_the_floor():
    r = timeline.rates_for_genre("movie")
    assert r.rate_max == pytest.approx(1.15)
    assert r.rate_min == pytest.approx(0.85)
    assert r.tail_max > timeline.TAIL_MAX          # more overhang tolerated


# ----------------------------------------------------------------- movie caps in place

def test_movie_never_compresses_past_its_cap_and_records_the_overrun():
    # 5s of dub on a 2s segment against a cross-speaker wall: documentary
    # squeezes to 1.3; movie stops at 1.15 and lets the remainder overrun.
    items = [item(0, 0.0, 5.0, end=2.0, speaker="A"),
             item(1, 2.2, 1.0, speaker="B")]
    movie = timeline.rates_for_genre("movie")
    places = timeline.place(items, movie)
    timeline.assert_invariants(places, items)
    assert places[0]["rate"] == pytest.approx(1.15)
    held = places[0]["end"] - places[0]["start"]
    assert held == pytest.approx(5.0 / 1.15, abs=2e-3)      # nothing truncated
    assert places[0]["overrun"] > 0                          # recorded, not hidden
    assert places[1]["start"] >= places[0]["end"] - 1e-6     # next pushed, no overlap
    # Documentary on the same items still reaches RATE_MAX (regression pin).
    doc = timeline.place(items)
    assert doc[0]["rate"] == pytest.approx(timeline.RATE_MAX)


def test_movie_stretch_floor_is_higher():
    # A short dub stretched to fill its span floors at 0.85, not 0.82.
    items = [item(0, 0.0, 3.0, end=4.0)]
    places = timeline.place(items, timeline.rates_for_genre("movie"))
    assert places[0]["rate"] == pytest.approx(0.85)


def test_movie_allows_a_longer_same_speaker_tail():
    # 3.9s on a 3s segment, same speaker next: documentary compresses to fit
    # own+0.6; movie's larger tail lets it play at 1.0 and just run over.
    items = [item(0, 0.0, 3.9, end=3.0, speaker="A"),
             item(1, 10.0, 1.0, speaker="A")]
    places = timeline.place(items, timeline.rates_for_genre("movie"))
    assert places[0]["rate"] == pytest.approx(1.0)
    assert places[0]["end"] <= 3.0 + timeline.MOVIE_TAIL_MAX + 1e-3
    doc = timeline.place(items)
    assert doc[0]["rate"] > 1.0                              # regression pin


def test_movie_keeps_the_walls_earlier_start_behavior():
    # Cross-speaker wall with free timeline before the clip: movie mode still
    # pulls the clip earlier (bounded by LEAD_MAX) before letting it overrun.
    items = [item(0, 5.0, 4.0, end=8.0, speaker="A"),
             item(1, 8.2, 1.0, speaker="B")]
    places = timeline.place(items, timeline.rates_for_genre("movie"))
    timeline.assert_invariants(places, items)
    assert places[0]["start"] < 5.0
    assert places[0]["start"] >= 5.0 - timeline.LEAD_MAX - 1e-3


# ----------------------------------------------------------------- rate continuity

def test_smoothing_converges_same_speaker_neighbours():
    # A hard-compressed line right next to a drawl of the same voice, with
    # slack after the pair: the rates must end within RATE_STEP_MAX.
    items = [item(0, 0.0, 4.0, end=2.0, speaker="A"),     # needs compression
             item(1, 3.0, 1.7, end=5.0, speaker="A"),     # will be stretched
             item(2, 20.0, 1.0, speaker="B")]
    places = timeline.place(items, timeline.rates_for_genre("movie"))
    assert places[0]["rate"] - places[1]["rate"] > timeline.RATE_STEP_MAX
    smoothed = timeline.smooth_rates(items, places)
    assert abs(smoothed[0]["rate"] - smoothed[1]["rate"]) <= timeline.RATE_STEP_MAX + 1e-6
    # The compressed clip was nudged down, the drawl up — toward each other.
    assert smoothed[0]["rate"] < places[0]["rate"]
    assert smoothed[1]["rate"] > places[1]["rate"]
    # A drawl is only ever lifted toward natural speed, never past it.
    assert smoothed[1]["rate"] <= 1.0 + 1e-9
    # No placement start moved, and nothing overlaps.
    for p, q in zip(places, smoothed):
        assert p["start"] == q["start"]
    for a, b in zip(smoothed, smoothed[1:]):
        assert b["start"] >= a["end"] - 1e-3


def test_smoothing_never_crosses_a_downstream_wall():
    # Slowing the fast clip may only spend slack up to the NEXT placement — and
    # when that next segment is a different speaker, never past its source
    # onset (the wall), even if the placement itself drifted later.
    items = [item(0, 0.0, 4.0, end=2.0, speaker="A"),
             item(1, 2.1, 1.7, end=4.1, speaker="B")]
    places = timeline.place(items, timeline.rates_for_genre("movie"))
    smoothed = timeline.smooth_rates(items, places)
    assert smoothed == places      # cross-speaker pair: untouched entirely
    # Same-speaker pair squeezed against a cross-speaker wall right after it.
    items = [item(0, 0.0, 4.0, end=2.0, speaker="A"),
             item(1, 2.1, 1.6, end=4.0, speaker="A"),
             item(2, 4.2, 1.0, speaker="B")]
    places = timeline.place(items, timeline.rates_for_genre("movie"))
    smoothed = timeline.smooth_rates(items, places)
    assert smoothed[1]["end"] <= max(places[2]["start"],
                                     items[2]["source_start"]) + 1e-3
    for a, b in zip(smoothed, smoothed[1:]):
        assert b["start"] >= a["end"] - 1e-3


def test_smoothing_ignores_cross_speaker_and_distant_pairs():
    # Different speakers, and same speaker across a scene break (gap >= 2s):
    # both stay exactly as placed.
    items = [item(0, 0.0, 4.0, end=2.0, speaker="A"),
             item(1, 2.2, 1.7, end=4.2, speaker="B"),      # cross-speaker
             item(2, 10.0, 1.7, end=12.0, speaker="B")]    # 5.8s gap: new scene
    places = timeline.place(items, timeline.rates_for_genre("movie"))
    assert timeline.smooth_rates(items, places) == places


def test_smoothing_leaves_unstretchable_clips_alone():
    items = [item(0, 0.0, 4.0, end=2.0, speaker="A", stretchable=False),
             item(1, 3.0, 1.7, end=5.0, speaker="A"),
             item(2, 20.0, 1.0, speaker="B")]
    places = timeline.place(items, timeline.rates_for_genre("movie"))
    smoothed = timeline.smooth_rates(items, places)
    assert smoothed[0]["rate"] == places[0]["rate"] == pytest.approx(1.0)


# ------------------------------------------------------------- interjection keeps

def seg(text, start=0.0, end=1.2, speaker="S0"):
    return {"id": 0, "start": start, "end": end, "speaker": speaker, "text": text}


def test_interjection_keep_fires_only_in_movie_mode():
    for genre, keep in (("movie", True), ("documentary", False)):
        s = [seg("וולקאם")]
        segments.mark_keep(s, None, "en", "he", genre=genre)
        assert s[0]["keep"] is keep
        assert s[0]["keep_reason"] == ("interjection" if keep else None)
    # Default genre is documentary — the old call signature is unchanged.
    s = [seg("וולקאם")]
    segments.mark_keep(s, None, "en", "he")
    assert s[0]["keep"] is False


def test_interjection_keep_respects_word_and_duration_limits():
    # Three words: not a beat.
    s = [seg("אהלן אהלן אהלן")]
    segments.mark_keep(s, None, "en", "he", genre="movie")
    assert s[0]["keep"] is False
    # Two interjection words within limits: kept.
    s = [seg("אהלן ביי")]
    segments.mark_keep(s, None, "en", "he", genre="movie")
    assert s[0]["keep"] is True and s[0]["keep_reason"] == "interjection"
    # Too long in seconds, same text: not kept.
    s = [seg("אהלן ביי", end=1.6)]
    segments.mark_keep(s, None, "en", "he", genre="movie")
    assert s[0]["keep"] is False


def test_interjection_keep_only_for_listed_or_borrowed_tokens():
    # Listed cross-lingual particles fire; borrowed source-script spellings of
    # target words fire via transliteration; ordinary source words never do.
    for text in ("יאללה", "הלו", "ביי!", "אוקיי", "וולקאם"):
        s = [seg(text)]
        segments.mark_keep(s, None, "en", "he", genre="movie")
        assert s[0]["keep"] is True, text
        assert s[0]["keep_reason"] == "interjection"
    for text in ("שלום", "מלחמה", "קדימה", "טוב מאוד"):
        s = [seg(text)]
        segments.mark_keep(s, None, "en", "he", genre="movie")
        assert s[0]["keep"] is False, text


def test_interjection_detection_is_script_general():
    # The literal list works whatever the pair; the borrowed-word translit test
    # only runs for a caseless source against a Latin target it can map.
    assert segments._is_interjection_token("привет", "ru", "en")
    assert segments._is_interjection_token("wow", "en", "ru")
    assert not segments._is_interjection_token("война", "ru", "en")
    # Borrowed test needs 3+ source-script letters — קו is ambiguous, skipped.
    assert not segments._is_interjection_token("קו", "he", "en")


def test_interjection_keep_still_gets_a_translated_subtitle():
    # The keep plays original audio, but the subtitle must be the translation:
    # translate.run routes interjection keeps through the subtitle-translation
    # path instead of copying their source text into text_en.
    s = seg("וולקאם")
    s.update(keep=True, keep_reason="interjection")
    assert translate.needs_subtitle_translation(s)
    # Already-translated, plain keeps, and empty texts are not re-translated.
    assert not translate.needs_subtitle_translation(dict(s, text_en="Welcome!"))
    assert not translate.needs_subtitle_translation(
        {"keep": True, "keep_reason": "latin", "text": "OK then"})
    assert not translate.needs_subtitle_translation(dict(s, text=""))
    # The foreign-span rule is unchanged.
    assert translate.needs_subtitle_translation(
        {"keep": True, "keep_reason": "foreign", "lang": "ar", "text": "مرحبا"})
    assert not translate.needs_subtitle_translation(
        {"keep": True, "keep_reason": "foreign", "lang": "und", "text": "مرحبا"})


# ------------------------------------------------------------------ movie prompt

_MOVIE_MARK = "This is dubbed movie dialogue"


def test_movie_prompt_block_present_only_for_movie_genre():
    movie = translate._translate_instruction("שלום", "he", "en", genre="movie")
    assert _MOVIE_MARK in movie
    assert "borrowed" in movie and "never expand a short line" in movie
    # movie implies the dialogue register even when the caller left narration.
    assert "contractions are welcome" in movie
    assert _MOVIE_MARK not in translate._translate_instruction("שלום", "he", "en")


def test_documentary_genre_prompt_is_byte_identical_to_today():
    # Regression pin: passing genre="documentary" (or nothing) changes nothing,
    # for both registers and both hop shapes.
    for kwargs in ({}, {"register": "dialogue"},
                   {"context": "war film", "preceding": "prev line"},
                   {"numbers_spelled": True, "asr_source": False}):
        base = translate._translate_instruction("שלום", "he", "en", **kwargs)
        assert base == translate._translate_instruction(
            "שלום", "he", "en", genre="documentary", **kwargs)
        assert _MOVIE_MARK not in base
    ru = translate._translate_instruction("Hello there.", "en", "ru")
    assert ru == translate._translate_instruction("Hello there.", "en", "ru",
                                                  genre="documentary")


def test_movie_prompt_keeps_the_measured_blocks_intact():
    # The movie block is additive: fluency/echelon/garble stay exactly gated
    # as before on an en→ru hop.
    p = translate._translate_instruction("Hello there.", "en", "ru", genre="movie")
    assert "idiomatic" in p and "echelon" in p
    assert _MOVIE_MARK in p
