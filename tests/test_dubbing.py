"""Unit tests for the dubbing pipeline's pure logic (no models, no audio)."""

from __future__ import annotations

import json
import random

import pytest

from dubbing import manifest, mix, segments, timeline, transcript, translate, tts


# --------------------------------------------------------------------------- timeline

def item(i, start, dur, *, end=None, stretchable=True):
    return {"id": i, "source_start": start, "source_end": end if end is not None else start + dur,
            "dur": dur, "stretchable": stretchable}


def random_items(rng, n=40):
    items, t = [], 0.0
    for i in range(n):
        span = rng.uniform(0.8, 8.0)
        items.append(item(i, round(t, 3), round(span * rng.uniform(0.6, 1.8), 3),
                          end=round(t + span, 3), stretchable=rng.random() > 0.2))
        t += span + rng.choice([0.0, 0.05, 0.3, 1.0, 4.0])
    return items


def test_placements_never_overlap():
    rng = random.Random(1234)
    for _ in range(200):
        places = timeline.place(random_items(rng))
        for a, b in zip(places, places[1:]):
            assert b["start"] > a["end"] - 1e-6


def test_contiguous_source_segments_stay_joined():
    # A passage of original audio split into parts must not gain silence at each
    # seam, or a long run of them slides steadily late.
    items = [item(i, i * 2.0, 2.0, end=(i + 1) * 2.0, stretchable=False) for i in range(8)]
    places = timeline.place(items)
    assert places[-1]["drift"] < 0.05          # seams cost milliseconds, not tenths
    for a, b in zip(places, places[1:]):
        seam = b["start"] - a["end"]
        assert 0 < seam <= timeline.MIN_SEAM + 1e-6


def test_separate_source_segments_keep_a_gap():
    items = [item(0, 0.0, 3.0, end=1.0), item(1, 5.0, 1.0, end=6.0)]
    places = timeline.place(items)
    assert places[1]["start"] >= places[0]["end"] + timeline.MIN_GAP - 1e-6


def test_nothing_is_ever_truncated():
    rng = random.Random(99)
    for _ in range(100):
        items = random_items(rng)
        places = timeline.place(items)
        assert len(places) == len(items)
        for it, p in zip(items, places):
            held = p["end"] - p["start"]
            # place() rounds to milliseconds; assert_invariants allows 20ms.
            assert held == pytest.approx(it["dur"] / p["rate"], abs=2e-3)
            assert p["rate"] <= timeline.RATE_MAX + 1e-9


def test_never_starts_before_its_source_onset():
    # A short clip must not slide earlier to fill the space in front of it.
    items = [item(0, 0.0, 0.5, end=2.0), item(1, 5.0, 1.0)]
    places = timeline.place(items)
    assert places[1]["start"] == 5.0
    assert places[1]["drift"] == 0.0


def test_drift_propagates_but_resets_at_a_pause():
    items = [item(0, 0.0, 5.0, end=2.0), item(1, 2.0, 1.0, end=3.0), item(2, 6.0, 1.0)]
    places = timeline.place(items)
    assert places[1]["drift"] > 0          # pushed late by its predecessor
    assert places[2]["drift"] == 0.0       # the 3s hole absorbed it
    assert places[2]["start"] == 6.0


def test_keep_segments_are_never_stretched():
    items = [item(0, 0.0, 4.0, end=1.0, stretchable=False), item(1, 1.0, 1.0)]
    places = timeline.place(items)
    assert places[0]["rate"] == 1.0
    assert places[0]["end"] - places[0]["start"] == pytest.approx(4.0)


def test_placement_is_a_fixed_point_once_applied():
    rng = random.Random(7)
    items = random_items(rng)
    first = timeline.place(items)
    applied = [dict(it, dur=it["dur"] / p["rate"], stretchable=False)
               for it, p in zip(items, first)]
    second = timeline.place(applied)
    for a, b in zip(first, second):
        # 2ms tolerance: a stretched clip's rate is rounded to 4 dp in the output,
        # so reconstructing dur from it drifts sub-frame — harmless for placement.
        assert a["start"] == pytest.approx(b["start"], abs=2e-3)
        assert a["end"] == pytest.approx(b["end"], abs=2e-3)


def test_escalates_speed_only_when_running_late():
    tight = timeline.rate_for(dur=3.0, slot=2.9, drift_in=0.0, stretchable=True)
    assert tight == pytest.approx(3.0 / 2.9)          # exact gentle fit
    hard = timeline.rate_for(dur=6.0, slot=3.0, drift_in=0.0, stretchable=True)
    assert hard == timeline.RATE_MAX                  # cannot fit; capped
    assert timeline.rate_for(2.0, 5.0, 0.0, True) == pytest.approx(timeline.RATE_MIN)  # short dub stretched to the floor
    assert timeline.rate_for(4.5, 5.0, 0.0, True) == pytest.approx(4.5 / 5.0)          # mild: fills the slot
    assert timeline.rate_for(2.0, 0.0, 0.0, True) == 1.0
    assert timeline.rate_for(2.0, 0.0, 1.0, True) == timeline.RATE_PREF


def test_assert_invariants_catches_overlap():
    items = [item(0, 0.0, 2.0), item(1, 1.0, 1.0)]
    bad = [{"id": 0, "start": 0.0, "end": 2.0, "rate": 1.0, "drift": 0.0},
           {"id": 1, "start": 1.0, "end": 2.0, "rate": 1.0, "drift": 0.0}]
    with pytest.raises(AssertionError, match="overlap"):
        timeline.assert_invariants(bad, items)


def test_assert_invariants_catches_truncation():
    items = [item(0, 0.0, 5.0)]
    bad = [{"id": 0, "start": 0.0, "end": 1.0, "rate": 1.0, "drift": 0.0}]
    with pytest.raises(AssertionError, match="truncated"):
        timeline.assert_invariants(bad, items)


# --------------------------------------------------------------------------- transcript

def test_chrome_and_arrows_are_stripped():
    assert transcript.clean_token("[מוזיקה]") == ("", False)
    assert transcript.clean_token("(applause)") == ("", False)
    assert transcript.clean_token(">>") == ("", True)
    assert transcript.clean_token(">> שלום") == ("שלום", True)
    assert transcript.clean_token(" word ") == ("word", False)


def test_collapse_repeats_drops_looped_hallucinations():
    w = lambda t, s: {"t": t, "end": t + 0.2, "text": s}
    # "big" repeated three times as a 2-gram collapses to one; the surrounding
    # words and the first copy's timing survive.
    words = [w(0.0, "qatar"), w(0.4, "is"),
             w(0.8, "the"), w(1.0, "biggest"),
             w(1.4, "the"), w(1.6, "biggest"),
             w(2.0, "the"), w(2.2, "biggest"),
             w(2.6, "end")]
    out = transcript.collapse_repeats(words)
    assert [x["text"] for x in out] == ["qatar", "is", "the", "biggest", "end"]
    assert out[2]["t"] == 0.8  # first occurrence's timing is kept


def test_collapse_repeats_keeps_ordinary_doubling():
    w = lambda t, s: {"t": t, "text": s}
    words = [w(0.0, "very"), w(0.3, "very"), w(0.6, "good")]
    assert [x["text"] for x in transcript.collapse_repeats(words)] == ["very", "very", "good"]


class _FakeWord:
    def __init__(self, word, start, end):
        self.word, self.start, self.end = word, start, end


class _FakeSeg:
    def __init__(self, text, avg_logprob=-0.2):
        self.text = text
        self.avg_logprob = avg_logprob
        self.words = [_FakeWord(w + " ", i * 0.3, i * 0.3 + 0.25)
                      for i, w in enumerate(text.split())]


class _FakeEnModel:
    """Returns scripted subtitle text per transcribe() call, in order."""
    def __init__(self, scripted):
        self.scripted, self.i = scripted, 0

    def transcribe(self, *a, **k):
        text = self.scripted[self.i]
        self.i += 1
        return iter([_FakeSeg(text)]), object()


def _detect(monkeypatch, regions, langs, texts, **kw):
    from dubbing import audio
    monkeypatch.setattr(audio, "decode_mono", lambda *a, **k: [0.0])
    monkeypatch.setattr(transcript, "vad_regions", lambda *a, **k: regions)
    monkeypatch.setattr(transcript, "_extend_english_end", lambda en, sw, b, limit: b)
    seq = iter(langs)
    monkeypatch.setattr(transcript, "detect_language", lambda lid, clip: next(seq))
    model = _FakeEnModel(texts)
    return transcript.detect_spoken_target_spans(
        model, object(), object(), "voc.wav", 320.0, "en", **kw)


def test_detect_spoken_target_spans_keeps_only_target_language(monkeypatch):
    # Regions <= LID_WINDOW so each is one chunk; region 1 English -> span, 2 Hebrew.
    spans = _detect(monkeypatch, [(2.0, 4.0), (8.0, 10.0)],
                    [("en", 0.9), ("he", 1.0)], ["Qatar is a dangerous enemy"])
    assert len(spans) == 1
    assert spans[0]["text"] == "Qatar is a dangerous enemy"
    # Start is the VAD onset; end is clamped to the last English word (1.45s into
    # the clip) + SPAN_END_PAD, never the raw VAD/extended boundary — so the span
    # cannot claim the source-language tail that follows the last English word.
    assert (spans[0]["start"], spans[0]["end"]) == (2.0, 3.7)


def test_detect_spoken_target_spans_rejects_low_confidence(monkeypatch):
    # English but below the LID confidence floor → not kept.
    spans = _detect(monkeypatch, [(2.0, 4.0)], [("en", 0.4)], ["whatever"])
    assert spans == []


def test_detect_spoken_target_spans_skips_known_ranges(monkeypatch):
    # English and confident, but already covered by a caption/recovered span.
    spans = _detect(monkeypatch, [(2.0, 4.0)], [("en", 0.9)], ["x y"],
                    known=[(1.0, 4.5)])
    assert spans == []


def test_words_from_json3_uses_word_offsets(tmp_path):
    data = {"events": [
        {"tStartMs": 1000, "segs": [{"utf8": "one"}, {"utf8": "two", "tOffsetMs": 500}]},
        {"tStartMs": 2000, "aAppend": 1, "segs": [{"utf8": "\n"}]},
        {"tStartMs": 2000, "segs": [{"utf8": "three", "isSpeakerChange": True}]},
    ]}
    path = tmp_path / "c.json3"
    path.write_text(json.dumps(data), encoding="utf-8")
    words = transcript.words_from_json3(path)
    assert [w["text"] for w in words] == ["one", "two", "three"]
    assert [w["t"] for w in words] == [1.0, 1.5, 2.0]
    assert words[2]["brk"] is True


def test_words_from_json3_respects_duration_limit(tmp_path):
    data = {"events": [{"tStartMs": 0, "segs": [
        {"utf8": "a"}, {"utf8": "b", "tOffsetMs": 5000}, {"utf8": "c", "tOffsetMs": 9000}]}]}
    path = tmp_path / "c.json3"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert [w["text"] for w in transcript.words_from_json3(path, limit=6.0)] == ["a", "b"]


# --------------------------------------------------------------------------- segments

def mkwords(spec, spk="S0"):
    return [{"t": t, "text": w, "brk": False, "spk": spk} for t, w in spec]


def test_word_ends_prefer_measured_end_over_guess():
    # With a real end the word span is exact; without one it falls back to the
    # generous guess, clamped to the next onset.
    measured = [{"t": 0.0, "end": 0.4, "text": "aa"}, {"t": 2.0, "text": "bb"}]
    assert segments.word_ends(measured)[0] == pytest.approx(0.4)
    guessed = [{"t": 0.0, "text": "aa"}, {"t": 0.3, "text": "bb"}]
    assert segments.word_ends(guessed)[0] == pytest.approx(0.3)  # clamped to next onset


def test_speech_gap_uses_measured_end():
    # 0.4s spoken then a real 0.6s silence — the char-count guess would have
    # over-estimated the spoken time and shrunk the gap.
    prev = {"t": 0.0, "end": 0.4, "text": "aa"}
    assert segments.speech_gap(prev, {"t": 1.0, "text": "bb"}) == pytest.approx(0.6)
    guessed = {"t": 0.0, "text": "aa"}  # spoken guess 0.10 + 0.06*2 = 0.22
    assert segments.speech_gap(guessed, {"t": 1.0, "text": "bb"}) == pytest.approx(0.78)


def test_measured_ends_split_a_real_pause_that_the_guess_would_miss():
    # Onsets are only 0.5s apart, but the measured end shows a 0.45s silence —
    # below GAP_SPLIT, so it stays one segment; widen the pause and it splits.
    tight = [{"t": 0.0, "end": 0.05, "text": "aa", "spk": "S0"},
             {"t": 0.5, "end": 0.55, "text": "bb", "spk": "S0"}]
    assert len(segments.words_to_segments(tight)) == 1
    wide = [{"t": 0.0, "end": 0.05, "text": "aa", "spk": "S0"},
            {"t": 1.0, "end": 1.05, "text": "bb", "spk": "S0"}]
    assert len(segments.words_to_segments(wide)) == 2


def test_keep_tail_extends_through_trailing_speech():
    # Speech (>= floor) for 0–1.0s, silence after. A keep segment ending at 0.5s
    # reclaims the trailing speech up to the pause.
    levels = [0.05] * 10 + [0.0] * 20
    segs = [{"id": 0, "start": 0.0, "end": 0.5, "speaker": "A", "keep": True},
            {"id": 1, "start": 2.5, "end": 3.0, "speaker": "B", "keep": False}]
    segments.extend_keeps_to_speech_end(segs, levels, 0.1, 3.0)
    assert 0.9 <= segs[0]["end"] <= 1.4    # grew to the end of speech (± a frame)
    assert segs[1]["end"] == 3.0           # non-keep segment untouched


def test_keep_tail_stops_at_the_next_segment():
    # The extension never crosses into the next segment (keep or dub) — the
    # language boundary is handled upstream by the detector.
    levels = [0.05] * 30                        # continuous speech
    segs = [{"id": 0, "start": 0.0, "end": 0.5, "speaker": "A", "keep": True},
            {"id": 1, "start": 1.0, "end": 3.0, "speaker": "B", "keep": False}]
    segments.extend_keeps_to_speech_end(segs, levels, 0.1, 3.0)
    assert segs[0]["end"] == pytest.approx(1.0)  # stops at the next segment's start
    assert segs[1]["start"] == 1.0               # untouched


def test_split_on_pause():
    words = mkwords([(0.0, "aa"), (0.3, "bb"), (0.6, "cc"),
                     (3.0, "dd"), (3.3, "ee"), (3.6, "ff")])
    segs = segments.words_to_segments(words)
    assert len(segs) == 2
    assert segs[0]["text"] == "aa bb cc"
    assert segs[1]["start"] == 3.0


def test_embedded_latin_token_stays_in_the_hebrew_segment():
    # A lone Latin acronym mid-sentence must not become its own segment (which
    # would play a fraction of a second of original audio — the 0:31 voice jump).
    words = mkwords([(0.0, "קטאר"), (0.3, "מממן"), (0.6, "את"),
                     (0.9, "ISIS,"), (1.2, "את"), (1.5, "נוסרה")])
    segs = segments.words_to_segments(words)
    assert len(segs) == 1
    assert "ISIS," in segs[0]["text"]
    segments.mark_keep(segs)
    assert not segs[0]["keep"]                  # dubbed, not kept as original audio


def test_sustained_latin_run_still_splits_and_is_kept():
    # A real English passage (>= FOREIGN_MIN_WORDS) still separates from Hebrew.
    words = mkwords([(0.0, "שלום"), (0.3, "עולם"),
                     (0.6, "this"), (0.9, "is"), (1.2, "english"), (1.5, "speech")])
    segs = segments.words_to_segments(words)
    assert [segments.script_of(s["text"]) for s in segs] == ["source", "latin"]


def test_split_on_sentence_end_and_speaker_change():
    words = mkwords([(0.0, "aa"), (0.3, "bb"), (0.6, "cc"), (0.9, "dd."),
                     (1.3, "ee"), (1.6, "ff"), (1.9, "gg"), (2.2, "hh")])
    segs = segments.words_to_segments(words)
    assert [s["text"] for s in segs] == ["aa bb cc dd.", "ee ff gg hh"]


def test_speaker_change_splits_only_when_a_pause_confirms_it():
    # Diarization edges land a word or two off the real turn boundary, so a
    # change with no pause must not cut the sentence — the majority vote names it.
    tight = mkwords([(0.0, "aa"), (0.3, "bb"), (0.6, "cc"),
                     (0.9, "dd"), (1.2, "ee"), (1.5, "ff")])
    for w in tight[4:]:
        w["spk"] = "S1"
    segs = segments.words_to_segments(tight)
    assert len(segs) == 1 and segs[0]["speaker"] == "S0"

    paused = mkwords([(0.0, "aa"), (0.3, "bb"), (0.6, "cc"),
                      (1.4, "dd"), (1.7, "ee"), (2.0, "ff")])
    for w in paused[3:]:
        w["spk"] = "S1"
    assert [s["speaker"] for s in segments.words_to_segments(paused)] == ["S0", "S1"]


def test_caption_speaker_marker_splits_immediately():
    words = mkwords([(0.0, "aa"), (0.3, "bb"), (0.6, "cc"),
                     (0.9, "dd"), (1.2, "ee"), (1.5, "ff")])
    words[3]["brk"] = True
    assert len(segments.words_to_segments(words)) == 2


def test_over_long_single_sentence_is_split():
    words = mkwords([(round(i * 0.4, 2), f"w{i}") for i in range(40)])   # ~16s, no punctuation
    words[20]["t"] = 8.5   # interior pause under GAP_SPLIT
    segs = segments.words_to_segments(words)
    assert all(s["end"] - s["start"] <= segments.SENTENCE_MAX + 1e-6 for s in segs)
    assert len(segs) >= 2


def test_single_sentence_under_the_cap_stays_whole():
    # ~12s single sentence: kept as one clip, not split mid-clause into a jump.
    words = mkwords([(round(i * 0.4, 2), f"w{i}") for i in range(30)])
    segs = segments.words_to_segments(words)
    assert len(segs) == 1


def test_stubs_merge_into_a_neighbour_but_never_across_speakers():
    words = mkwords([(0.0, "aa"), (0.2, "bb"), (0.4, "cc."), (0.7, "dd")])
    segs = segments.words_to_segments(words)
    assert len(segs) == 1                       # the one-word tail folded back in
    split = mkwords([(0.0, "aa"), (0.2, "bb"), (0.4, "cc."), (0.7, "dd")])
    split[3]["spk"] = "S1"
    assert len(segments.words_to_segments(split)) == 2


def test_every_word_survives_segmentation():
    rng = random.Random(5)
    words, t = [], 0.0
    for i in range(300):
        words.append({"t": round(t, 3), "text": f"w{i}", "brk": rng.random() < 0.05, "spk": "S0"})
        t += rng.choice([0.2, 0.3, 0.9, 2.0])
    segs = segments.words_to_segments(words)
    assert " ".join(s["text"] for s in segs).split() == [w["text"] for w in words]


def test_speaker_assignment_by_overlap_and_fallback():
    words = mkwords([(0.0, "aa"), (5.0, "bb"), (50.0, "cc")])
    turns = [{"speaker": "A", "start": 0.0, "end": 2.0},
             {"speaker": "B", "start": 4.0, "end": 8.0}]
    segments.assign_word_speakers(words, turns)
    assert [w["spk"] for w in words] == ["A", "B", "B"]
    segments.assign_word_speakers(words, [])
    assert {w["spk"] for w in words} == {"SPEAKER_00"}


def test_foreign_spans_become_their_own_segments():
    # ASR either mangles or skips target-language speech, so those seconds must
    # come from the captions or they would have no audio at all.
    asr = [{"id": 0, "start": 0.0, "end": 5.0, "speaker": "A", "text": "עברית"},
           {"id": 1, "start": 10.2, "end": 14.0, "speaker": "A", "text": "שיבוש של אנגלית"},
           {"id": 2, "start": 20.0, "end": 24.0, "speaker": "A", "text": "עוד עברית"}]
    spans = [{"start": 10.0, "end": 15.0, "text": "Frankly I had the same concern",
              "words": [{"t": 10.0 + 0.4 * i, "text": w, "spk": "B"}
                        for i, w in enumerate("Frankly I had the same concern".split())]}]
    out = segments.splice_foreign_spans(asr, spans)
    assert [s["text"] for s in out] == ["עברית", "Frankly I had the same concern", "עוד עברית"]
    segments.mark_keep(out, spans)
    assert out[1]["keep"] and out[1]["keep_reason"] == "latin"
    assert not out[0]["keep"] and not out[2]["keep"]


def test_speech_between_two_spans_inside_one_segment_survives():
    # A Hebrew sentence interrupted twice by English is more than half span by area;
    # judging it on total overlap dropped it whole and lost the Hebrew in between.
    said = "כי הבעיה היא שבאחור"
    words = ([{"t": 0.2 + 0.3 * i, "text": w, "spk": "A"} for i, w in enumerate("אני עדיין מציע".split())]
             + [{"t": 4.5 + 0.4 * i, "text": w, "spk": "A"} for i, w in enumerate(said.split())]
             + [{"t": 9.2 + 0.3 * i, "text": w, "spk": "A"} for i, w in enumerate("את המילים".split())])
    asr = [{"id": 0, "start": 0.2, "end": 10.0, "speaker": "A", "text": "…"}]
    spans = [{"start": 1.5, "end": 4.2, "text": "I still would urge caution",
              "words": [{"t": 1.5 + 0.4 * i, "text": w, "spk": "B"}
                        for i, w in enumerate("I still would urge caution".split())]},
             {"start": 8.0, "end": 10.1, "text": "I am not sure",
              "words": [{"t": 8.0 + 0.4 * i, "text": w, "spk": "B"}
                        for i, w in enumerate("I am not sure".split())]}]
    out = segments.splice_foreign_spans(asr, spans, words)
    assert said in [s["text"] for s in out]
    assert not segments.unsegmented_words(words, out, spans)
    # Every piece keeps clear of both spans.
    for s in out:
        for sp in spans:
            assert s["end"] <= sp["start"] or s["start"] >= sp["end"] or s["text"] == sp["text"]


def test_span_segments_tile_the_whole_passage():
    # Recovered word timings bunch up; the passage must still play end to end.
    spans = [{"start": 10.0, "end": 20.0, "text": "hello there my friend",
              "words": [{"t": 18.6, "text": "hello"}, {"t": 18.8, "text": "there"},
                        {"t": 19.4, "text": "my"}, {"t": 19.6, "text": "friend"}]}]
    out = segments.splice_foreign_spans([], spans)
    assert out[0]["start"] == 10.0 and out[-1]["end"] == 20.0
    for a, b in zip(out, out[1:]):
        assert a["end"] == b["start"], "no silent hole inside the passage"


def test_short_untouched_segments_are_never_dropped():
    # A brief recovered phrase must survive: only a trimmed remnant may be cut.
    asr = [{"id": 0, "start": 45.16, "end": 45.53, "speaker": "A", "text": "just want to"},
           {"id": 1, "start": 45.53, "end": 50.0, "speaker": "A", "text": "עברית"}]
    spans = [{"start": 270.0, "end": 275.0, "text": "elsewhere entirely",
              "words": [{"t": 270.0, "text": "elsewhere"}, {"t": 271.0, "text": "entirely"}]}]
    out = segments.splice_foreign_spans(asr, spans)
    assert "just want to" in [s["text"] for s in out]


def test_unsegmented_words_are_detected():
    words = [{"t": 1.0, "text": "a"}, {"t": 9.0, "text": "orphan"}, {"t": 20.5, "text": "b"}]
    segs = [{"start": 0.5, "end": 2.0}, {"start": 20.0, "end": 21.0}]
    spans = []
    lost = segments.unsegmented_words(words, segs, spans)
    assert [w["text"] for w in lost] == ["orphan"]
    # words inside a target-language span were deliberately replaced
    assert segments.unsegmented_words(words, segs, [{"start": 8.0, "end": 10.0}]) == []


def test_span_trims_rather_than_overlaps_a_neighbour():
    asr = [{"id": 0, "start": 8.0, "end": 11.0, "speaker": "A", "text": "עברית"}]
    spans = [{"start": 10.0, "end": 15.0, "text": "hello there friend",
              "words": [{"t": 10.0 + 0.4 * i, "text": w, "spk": "B"}
                        for i, w in enumerate("hello there friend".split())]}]
    out = segments.splice_foreign_spans(asr, spans)
    assert out[0]["end"] == 10.0            # trimmed back to the span's edge
    assert out[1]["start"] >= 10.0
    for a, b in zip(out, out[1:]):
        assert b["start"] >= a["end"] - 1e-6


def test_foreign_spans_from_captions():
    words = [{"t": 1.0, "text": "שלום"}, {"t": 5.0, "text": "Frankly"},
             {"t": 5.4, "text": "I"}, {"t": 5.9, "text": "had"}, {"t": 30.0, "text": "עברית"}]
    spans = transcript.foreign_spans(words)
    assert len(spans) == 1
    assert spans[0]["text"] == "Frankly I had"
    assert spans[0]["start"] == 5.0


def test_keep_rules():
    segs = [
        {"id": 0, "start": 0, "end": 5, "speaker": "A", "text": "This is English speech"},
        {"id": 1, "start": 5, "end": 10, "speaker": "A", "text": "and more English here"},
        # Phonetically transcribed English: only the speaker rule can catch it.
        {"id": 2, "start": 10, "end": 15, "speaker": "A", "text": "עוד קצת עברית"},
        {"id": 3, "start": 15, "end": 20, "speaker": "B", "text": "שלום לכולם"},
    ]
    segments.mark_keep(segs)
    assert [s["keep_reason"] for s in segs] == ["latin", "latin", "speaker_en", None]
    assert not segs[3]["keep"]


# --------------------------------------------------------------------------- translate

def test_is_target_text():
    assert translate.is_target_text("Hello there")
    assert not translate.is_target_text("שלום עולם")
    assert not translate.is_target_text("")


def test_adjacent_repeat_detects_only_the_collapse():
    rep = translate._adjacent_repeat
    # The "education, education" defect (comma, coordinator, or bare adjacency).
    assert rep("disseminating education, education and human rights") == "education"
    assert rep("education and education, and human rights") == "education"
    # Phrase-level emphasis, intensifiers, stopwords and clean lists are left alone.
    assert rep("Qatar funds Hamas, Qatar funds Al-Qaeda, Qatar funds ISIS") is None
    assert rep("a very, very extreme ideological view") is None
    assert rep("education, learning, and human rights") is None
    assert rep("the the report") is None


def test_strip_adjacent_repeat_removes_the_duplicate():
    assert (translate._strip_adjacent_repeat("education, education and human rights")
            == "education and human rights")
    assert translate._strip_adjacent_repeat("scholarship and scholarship") == "scholarship"


def test_strip_editorial_removes_brackets_and_notes():
    strip = translate._strip_editorial
    # Gemma 4 brackets words it supplied and offers alternatives; both would be read aloud.
    assert strip("Qatar funds ISIS, [and] Qatar funds al-Nusra") == \
        "Qatar funds ISIS, and Qatar funds al-Nusra"
    assert strip("a country with a [connection/link] to the Brotherhood") == \
        "a country with a connection to the Brotherhood"
    # A trailing translator's note goes with it.
    assert strip("In the end, it is a state.\n\n*(Note: the Hebrew appears garbled.)*") == \
        "In the end, it is a state."
    # Clean output is untouched.
    assert strip("Through Qatari gas, Qatar is bribing Europe.") == \
        "Through Qatari gas, Qatar is bribing Europe."


def test_prompt_prefers_the_models_own_chat_template():
    class Tok:
        chat_template = "…"

        def apply_chat_template(self, messages, **kw):
            assert kw["add_generation_prompt"] and kw["enable_thinking"] is False
            return [1, 2, 3]

    class NoTemplate:
        chat_template = None

    assert translate._prompt(Tok(), "text") == [1, 2, 3]
    # Without one, fall back to writing Gemma 3's markers by hand.
    assert translate._prompt(NoTemplate(), "text").startswith("<start_of_turn>user\ntext")


def test_generate_repairs_a_repeated_word(monkeypatch):
    # First pass collapses two words; the retry (with the nudge) resolves it.
    replies = iter([
        "disseminating education, education and human rights.",
        "disseminating education, upbringing and human rights.",
    ])
    monkeypatch.setattr(translate, "_run", lambda *a, **k: next(replies))
    out = translate.generate(None, None, "he-text", source="he", target="en")
    assert out == "disseminating education, upbringing and human rights."


def test_generate_strips_repeat_when_retry_also_repeats(monkeypatch):
    # Retry still repeats → last-resort strip removes the duplicate.
    monkeypatch.setattr(translate, "_run",
                        lambda *a, **k: "disseminating education, education and human rights.")
    out = translate.generate(None, None, "he-text", source="he", target="en")
    assert out == "disseminating education and human rights."


def test_shorten_rejects_unsafe_rewrites(monkeypatch):
    original = ("The emirate of Qatar funded the group with 400 million dollars "
                "and did not deny it")

    def attempt(reply):
        monkeypatch.setattr(translate, "_run", lambda *a, **k: reply)
        return translate.shorten(None, None, "he", original, 12, source="he", target="en")

    assert attempt("Qatar funded the group with 400 million dollars, not denying it.")
    assert attempt("Qatar funded the group and did not deny it") is None       # lost the number
    assert attempt("Qatar funded the group with 400 million dollars") is None  # lost the negation
    assert attempt("The group received 400 million and did not deny it") is None  # lost the name
    assert attempt(original) is None                                          # not actually shorter
    assert attempt("קטר מימנה") is None                                        # not the target language


# --------------------------------------------------------------------------- tts

def test_prepare_en_keeps_every_speakable_word():
    assert tts.prepare_en("Hello [music] world") == "Hello world."
    assert tts.prepare_en("Qatar funds it, and more") == "Qatar funds it, and more."
    assert tts.prepare_en("mixed שלום text") == "mixed text."
    assert tts.prepare_en("שלום") == ""


def test_clone_length_gate():
    text = "one two three four five six"          # 6 words, ~2.25s expected
    assert tts.clone_length_ok(2.0, text)
    assert not tts.clone_length_ok(0.5, text)      # chipmunk
    assert not tts.clone_length_ok(9.0, text)      # stalled
    assert not tts.clone_length_ok(0.0, text)


def test_word_overlap():
    assert tts.word_overlap("hello world", "hello world") == 1.0
    assert tts.word_overlap("hello world", "hello") == 0.5
    assert tts.word_overlap("hello world", "") == 0.0


# --------------------------------------------------------------------------- mix

def test_duck_envelope():
    segs = [
        {"keep": False, "place": {"start": 1.0, "end": 3.0, "clip": "x"}},
        {"keep": True, "place": {"start": 5.0, "end": 6.0, "clip": "y"}},
    ]
    env = mix.build_envelope(segs, 10.0)
    assert env[int(0.2 * mix.CTRL_HZ)] == pytest.approx(1.0, abs=0.02)   # untouched bed
    assert env[int(2.0 * mix.CTRL_HZ)] == pytest.approx(mix.DUCK_DUB, abs=0.02)
    assert env[int(5.5 * mix.CTRL_HZ)] < 0.15                            # keep span silenced
    assert env[int(9.5 * mix.CTRL_HZ)] == pytest.approx(1.0, abs=0.02)


# --------------------------------------------------------------------------- manifest

def test_fingerprint_chain_invalidates_downstream():
    m = manifest.new({"input": "x"})
    fp_a = manifest.stage_fingerprint(m, "fetch", {"source": "a"})
    manifest.mark_stage(m, "fetch", fp_a)
    down_before = manifest.stage_fingerprint(m, "stems", {})
    manifest.mark_stage(m, "fetch", manifest.stage_fingerprint(m, "fetch", {"source": "b"}))
    assert manifest.stage_fingerprint(m, "stems", {}) != down_before


def test_reset_stage_drops_that_stage_and_its_keep_flips():
    m = manifest.new({"input": "x"})
    m["segments"] = [
        {"id": 0, "keep": False, "keep_reason": None, "text_en": "hi",
         "tts": {"clip": "a"}, "place": {"start": 0}},
        {"id": 1, "keep": True, "keep_reason": "mt_failed", "text_en": "x"},
        {"id": 2, "keep": True, "keep_reason": "latin", "text_en": "y"},
    ]
    manifest.reset_stage(m, "translate")
    assert "text_en" not in m["segments"][0]
    assert m["segments"][0]["tts"] == {"clip": "a"}          # other stages untouched
    assert not m["segments"][1]["keep"]                       # mt_failed re-decided
    assert m["segments"][2]["keep_reason"] == "latin"         # segmentation's call stands

    manifest.reset_stage(m, "timeline")
    assert "place" not in m["segments"][0]


def test_save_drops_unknown_segment_keys(tmp_path):
    m = manifest.new({"input": "x"})
    m["segments"] = [{"id": 0, "start": 0.0, "end": 1.0, "text": "hi", "junk": 1,
                      "internal_state": {"a": 1}}]
    manifest.save(tmp_path, m)
    loaded = manifest.load(tmp_path)
    assert set(loaded["segments"][0]) == {"id", "start", "end", "text"}
