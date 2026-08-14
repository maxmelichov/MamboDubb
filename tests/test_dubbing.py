"""Unit tests for the dubbing pipeline's pure logic (no models, no audio)."""

from __future__ import annotations

import json
import random
import sys

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
    # Invisible bidi marks the Hebrew ASR emits at direction changes carry no speech.
    assert transcript.clean_token("\u202bהיא") == ("היא", False)


def test_collapse_repeats_drops_looped_hallucinations():
    def w(t, s):
        return {"t": t, "end": t + 0.2, "text": s}
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
    def w(t, s):
        return {"t": t, "text": s}
    words = [w(0.0, "very"), w(0.3, "very"), w(0.6, "good")]
    assert [x["text"] for x in transcript.collapse_repeats(words)] == ["very", "very", "good"]


def test_join_split_marks_rejoins_a_geresh_word():
    def w(t, s):
        return {"t": t, "end": t + 0.2, "text": s}

    # Whisper splits ג'יהאד into "לג" + "'יהאד", stranding the preposition ל and
    # turning "calls for jihad" into "calls jihad"; likewise "אל" + "-קאעידה".
    out = transcript.join_split_marks([w(0.0, "שקוראת"), w(0.4, "לג"), w(0.6, "'יהאד."),
                                       w(1.0, "אל"), w(1.2, "-קאעידה")])
    assert [x["text"] for x in out] == ["שקוראת", "לג'יהאד.", "אל-קאעידה"]
    assert (out[1]["t"], out[1]["end"]) == (0.4, 0.8)      # first onset, second's end
    # After a sentence end the mark starts something new, and a speaker break is never
    # swallowed.
    kept = transcript.join_split_marks([w(0.0, "סוף."), w(0.4, "-אחר כך")])
    assert [x["text"] for x in kept] == ["סוף.", "-אחר כך"]
    # And a mark-initial token a pause away belongs to whatever comes next, not to the
    # word before it.
    apart = transcript.join_split_marks([w(0.0, "מילה"), w(3.0, "-ועוד")])
    assert [x["text"] for x in apart] == ["מילה", "-ועוד"]


class _FakeWhisper:
    """Stands in for faster_whisper.WhisperModel; scripted to fail per device."""

    calls: list[dict] = []
    fail_construct = frozenset()
    fail_transcribe = frozenset()

    def __init__(self, path, device="cpu", compute_type="auto", **kw):
        if device in self.fail_construct:
            raise RuntimeError(f"no {device}")
        self.device = device
        type(self).calls.append({"device": device, "compute_type": compute_type, **kw})

    def transcribe(self, *_a, **_k):
        if self.device in self.fail_transcribe:
            raise RuntimeError("cuDNN missing")
        return iter(()), None


def _picked(monkeypatch, fake):
    import faster_whisper

    fake.calls = []
    monkeypatch.setattr(faster_whisper, "WhisperModel", fake)
    monkeypatch.setattr(transcript, "_cuda_usable", lambda: True)
    monkeypatch.setattr(transcript, "_cudnn_on_path", lambda: None)
    return fake


def test_load_whisper_prefers_cuda(monkeypatch):
    fake = _picked(monkeypatch, _FakeWhisper)
    model = transcript.load_whisper("m", cpu_threads=7)
    assert model.device == "cuda"
    assert fake.calls == [{"device": "cuda", "compute_type": "float16"}]


def test_load_whisper_falls_back_when_construction_fails(monkeypatch):
    class Fake(_FakeWhisper):
        fail_construct = frozenset({"cuda"})

    fake = _picked(monkeypatch, Fake)
    model = transcript.load_whisper("m", cpu_threads=7)
    assert model.device == "cpu"
    # cpu_threads reaches only the CPU fallback, with the pre-CUDA compute_type.
    assert fake.calls == [{"device": "cpu", "compute_type": "auto", "cpu_threads": 7}]


def test_load_whisper_falls_back_when_warmup_fails(monkeypatch):
    class Fake(_FakeWhisper):
        fail_transcribe = frozenset({"cuda"})

    fake = _picked(monkeypatch, Fake)
    model = transcript.load_whisper("m")
    assert model.device == "cpu"
    assert fake.calls[-1] == {"device": "cpu", "compute_type": "auto"}


def test_load_whisper_skips_cuda_when_unavailable(monkeypatch):
    fake = _picked(monkeypatch, _FakeWhisper)
    monkeypatch.setattr(transcript, "_cuda_usable", lambda: False)
    model = transcript.load_whisper("m")
    assert model.device == "cpu"
    assert fake.calls == [{"device": "cpu", "compute_type": "auto"}]


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


def _detect(monkeypatch, regions, langs, texts, *, pause=None, extend_to=None, **kw):
    """Run the detector over scripted VAD regions, languages and subtitle texts.

    `pause` is what the voice-stop scan reports (None = the voice never stops in
    reach), `extend_to` what _extend_english_end returns (None = it does not move).
    """
    from dubbing import audio
    monkeypatch.setattr(audio, "decode_mono", lambda *a, **k: [0.0])
    monkeypatch.setattr(transcript, "vad_regions", lambda *a, **k: regions)
    monkeypatch.setattr(transcript, "_extend_english_end",
                        lambda en, sw, b, limit, tgt="en": b if extend_to is None else extend_to)
    monkeypatch.setattr(transcript, "_voice_pause_after", lambda sw, start, limit: pause)
    seq = iter(langs)
    monkeypatch.setattr(transcript, "detect_language", lambda lid, clip: next(seq))
    model = _FakeEnModel(texts)
    kw.setdefault("source", "he")
    return transcript.detect_spoken_target_spans(
        model, object(), object(), "voc.wav", 320.0, "en", **kw)


def test_detect_spoken_target_spans_keeps_only_target_language(monkeypatch):
    # Regions <= LID_WINDOW so each is one chunk; region 1 English -> span, 2 Hebrew.
    spans = _detect(monkeypatch, [(2.0, 4.0), (8.0, 10.0)],
                    [("en", 0.9), ("he", 1.0)], ["Qatar is a dangerous enemy"])
    assert len(spans) == 1
    assert spans[0]["text"] == "Qatar is a dangerous enemy"
    # Start is the VAD onset. With no pause in reach the end falls back to the last
    # English word (1.45s into the clip) + SPAN_END_PAD — stopping early is the safe
    # side, since the alternative is airing whoever speaks next.
    assert (spans[0]["start"], spans[0]["end"]) == (2.0, 3.7)


def test_span_ends_where_the_voice_stops(monkeypatch):
    # Whisper's last word bunches early; the speaker is still talking past it, so the
    # span runs to the pause instead of cutting them off mid-word (1:51).
    spans = _detect(monkeypatch, [(2.0, 4.0), (8.0, 10.0)],
                    [("en", 0.9), ("he", 1.0)], ["Qatar is a dangerous enemy"], pause=4.1)
    assert spans[0]["end"] == 4.1

    # And when the classifier calls English well past where the voice actually stopped,
    # the pause still wins — keeping the rest would air the source speaker (3:25).
    spans = _detect(monkeypatch, [(2.0, 6.0), (8.0, 10.0)],
                    [("en", 0.9), ("he", 1.0)], ["Qatar is a dangerous enemy"],
                    pause=4.2, extend_to=7.5)
    assert spans[0]["end"] == 4.2


def test_voice_pause_needs_more_than_a_plosive_gap(monkeypatch):
    from dubbing import audio
    monkeypatch.setattr(audio, "decode_mono", lambda *a, **k: [0.0])
    hop = transcript.PAUSE_HOP
    # One quiet frame is a stop consonant inside a word, not the end of the sentence.
    monkeypatch.setattr(audio, "frame_rms", lambda *a, **k: [0.09, 0.001, 0.09, 0.09])
    assert transcript._voice_pause_after("voc.wav", 100.0, 100.0 + 4 * hop) is None
    # Two in a row is a pause, reported at its first frame.
    monkeypatch.setattr(audio, "frame_rms", lambda *a, **k: [0.09, 0.09, 0.001, 0.001])
    assert transcript._voice_pause_after("voc.wav", 100.0, 100.0 + 4 * hop) == 100.0 + 2 * hop


def test_a_third_language_is_kept_as_is(monkeypatch):
    # Arabic in a Hebrew documentary: no ASR here reads it, so it is kept as recorded,
    # with no subtitle text — dubbing it from the source model's gibberish is worse.
    monkeypatch.setattr(transcript, "_sounds_foreign",
                        lambda lid, mdl, sw, a, b, src: "ar")
    spans = _detect(monkeypatch, [(2.0, 6.0), (8.0, 10.0)],
                    [("ar", 0.9), ("he", 1.0)], [])
    assert len(spans) == 1
    assert (spans[0]["start"], spans[0]["end"], spans[0]["lang"]) == (2.0, 6.0, "ar")
    assert spans[0]["text"] == "…"

    # The source language itself is never kept — that is what gets dubbed.
    assert _detect(monkeypatch, [(2.0, 6.0)], [("he", 1.0)], []) == []

    # And an unconfirmed third language is left to the dub rather than aired blind.
    monkeypatch.setattr(transcript, "_sounds_foreign",
                        lambda lid, mdl, sw, a, b, src: None)
    assert _detect(monkeypatch, [(2.0, 6.0)], [("ar", 0.9)], []) == []


def test_foreign_group_joins_the_pieces_of_one_passage():
    # The 4s windows cut a long Arabic answer into runs with an unlabelled sliver
    # between; judged alone the first piece misses the bar and gets dubbed from
    # gibberish. The group is the passage.
    runs = [(404.1, 420.1, "en"), (421.1, 425.1, "ar"), (425.1, 425.8, None),
            (426.2, 446.2, "ar"), (446.2, 451.9, "he")]
    a, b, used = transcript._foreign_group(runs, 1)
    assert (a, b) == (421.1, 446.2)
    assert used == {1, 3}
    # It never reaches over a run in another language, even a close one.
    runs = [(10.0, 14.0, "ar"), (14.2, 18.0, "he"), (18.1, 22.0, "ar")]
    assert transcript._foreign_group(runs, 0)[:2] == (10.0, 14.0)
    # Nor across a real silence.
    runs = [(10.0, 14.0, "ar"), (30.0, 34.0, "ar")]
    assert transcript._foreign_group(runs, 0)[:2] == (10.0, 14.0)


def test_a_foreign_span_ending_mid_utterance_runs_to_the_pause(monkeypatch):
    # The classifier's windows can end a passage while the speaker is still talking;
    # that last second then gets dubbed from gibberish. It runs to the next pause.
    monkeypatch.setattr(transcript, "_sounds_foreign",
                        lambda lid, mdl, sw, a, b, src: "ar")
    spans = _detect(monkeypatch, [(2.0, 6.0), (8.0, 10.0)],
                    [("ar", 0.9), ("he", 1.0)], [], pause=7.3)
    assert spans[0]["end"] == 7.3
    # Where the boundary is already a pause, nothing moves.
    spans = _detect(monkeypatch, [(2.0, 6.0), (8.0, 10.0)],
                    [("ar", 0.9), ("he", 1.0)], [], pause=6.0)
    assert spans[0]["end"] == 6.0


def test_foreign_start_walks_back_over_what_is_not_the_source(monkeypatch):
    # A clip can begin inside a run the classifier calls the source language by
    # majority; those seconds get dubbed from gibberish over the top of it.
    monkeypatch.setattr(transcript, "_utterance_start_before",
                        lambda sw, end, limit: {593.4: 591.3, 591.3: 589.7}.get(round(end, 1)))
    reads = {(591.3, 593.4): False, (589.7, 591.3): True}   # the Hebrew before it reads fine
    monkeypatch.setattr(transcript, "_reads_as_source",
                        lambda lid, mdl, sw, a, b, src: reads[(round(a, 1), round(b, 1))])
    got = transcript._extend_foreign_start(object(), object(), "voc.wav", 593.4, "he", 0.0)
    assert got == 591.3                     # took the clip's own first line, stopped at Hebrew

    # It never walks past the bound, however much is unreadable.
    monkeypatch.setattr(transcript, "_reads_as_source", lambda *a, **k: False)
    monkeypatch.setattr(transcript, "_utterance_start_before",
                        lambda sw, end, limit: max(limit, end - 0.5))
    assert transcript._extend_foreign_start(
        object(), object(), "voc.wav", 100.0, "he", 0.0) == 100.0 - transcript.FOREIGN_BACK_MAX


def test_sounds_foreign_demands_more_than_the_window_labels(monkeypatch):
    from dubbing import audio
    monkeypatch.setattr(audio, "decode_mono", lambda *a, **k: [0.0])
    monkeypatch.setattr(transcript, "detect_language", lambda lid, clip: ("ar", 0.9))
    unreadable = type("M", (), {"transcribe": lambda self, *a, **k: (
        iter([_FakeSeg("gibberish", avg_logprob=-0.64)]), object())})()
    readable = type("M", (), {"transcribe": lambda self, *a, **k: (
        iter([_FakeSeg("clean source sentence", avg_logprob=-0.34)]), object())})()
    sounds = transcript._sounds_foreign
    # Too short for the classifier to be the only witness.
    assert sounds(object(), unreadable, "voc.wav", 0.0, 1.0, "he") is None
    assert sounds(object(), unreadable, "voc.wav", 0.0, 4.0, "he") == "ar"
    # The source ASR reads it fine → the label was wrong and this is the source
    # language; airing it would put the narrator's own voice on top of the dub.
    assert sounds(object(), readable, "voc.wav", 0.0, 4.0, "he") is None
    # Without the witness at all, nothing is kept blind.
    assert sounds(object(), None, "voc.wav", 0.0, 4.0, "he") is None
    # Too unsure to name it, but the ASR still cannot read it: kept, unnamed. A
    # Chinese news clip classified as `vi` 0.43 / `tr` 0.34 is plainly not Hebrew.
    monkeypatch.setattr(transcript, "detect_language", lambda lid, clip: ("ar", 0.7))
    assert sounds(object(), unreadable, "voc.wav", 0.0, 4.0, "he") == "und"
    monkeypatch.setattr(transcript, "detect_language", lambda lid, clip: (None, 0.0))
    assert sounds(object(), unreadable, "voc.wav", 0.0, 4.0, "he") == "und"
    # But a confident "this IS the source language" outranks the ASR.
    monkeypatch.setattr(transcript, "detect_language", lambda lid, clip: ("he", 0.99))
    assert sounds(object(), unreadable, "voc.wav", 0.0, 4.0, "he") is None


def test_leading_fragment_is_reclaimed_when_it_is_speech(monkeypatch):
    # VAD breaks "Frankly," off the front of an English sentence and the classifier
    # calls the 0.9s fragment Maori; the English model, hearing it with the run it
    # abuts, reads it as confident English with a word inside the fragment.
    from dubbing import audio
    monkeypatch.setattr(audio, "decode_mono", lambda *a, **k: [0.0])
    english = _FakeEnModel(["Frankly I had the same"])
    assert transcript._reclaim_leading_fragment(english, "voc.wav", 270.8, 272.0, 277.5) == 270.8

    # Gibberish behind it (Hebrew read by the English model) is refused...
    class _Gibberish:
        def transcribe(self, *a, **k):
            return iter([_FakeSeg("...", avg_logprob=-1.2)]), object()

    assert transcript._reclaim_leading_fragment(_Gibberish(), "voc.wav", 270.8, 272.0, 277.5) == 272.0
    # ...and so is a fragment too long to be a broken-off word.
    assert transcript._reclaim_leading_fragment(
        _FakeEnModel(["Frankly I had the same"]), "voc.wav", 268.0, 272.0, 277.5) == 272.0


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


# ------------------------------------------- diarization smoothing and handoff timing
#
# The "news" shape: one speaker holds the floor, with occasional short clips from
# somebody else. Pyannote sprinkles sub-half-second turns of a second label through
# the dominant voice, and each one used to cut a sentence into three segments whose
# middle carried the wrong speaker and was voiced by a different clone.

def mkturns(spec):
    return [{"speaker": s, "start": a, "end": b} for a, b, s in spec]


def test_sub_minimum_speaker_blip_is_absorbed_by_its_host():
    turns = mkturns([(0.0, 10.0, "A"), (10.0, 10.25, "B"), (10.25, 20.0, "A")])
    assert [t["speaker"] for t in segments.smooth_turns(turns)] == ["A", "A", "A"]
    # Timings are never touched, and the caller's dicts are not mutated.
    assert [(t["start"], t["end"]) for t in segments.smooth_turns(turns)] == \
        [(t["start"], t["end"]) for t in turns]
    assert turns[1]["speaker"] == "B"


def test_a_flutter_of_labels_collapses_into_one_turn():
    # Measured (news_videos clip 3, 29.0-30.0s): pyannote flickers between two
    # labels six times inside one second of one person's sentence. Absorbing the
    # shortest island first collapses the flicker from the inside out; what is
    # left is a single sub-second turn, not six alternating ones.
    turns = mkturns([(20.500, 29.039, "S1"), (29.039, 29.579, "S0"),
                     (29.579, 29.596, "S1"), (29.596, 29.613, "S0"),
                     (29.613, 29.630, "S1"), (29.630, 29.967, "S0"),
                     (29.967, 37.510, "S1")])
    out = segments.smooth_turns(turns)
    assert [t["speaker"] for t in out] == ["S1", "S0", "S0", "S0", "S0", "S0", "S1"]


def test_a_real_short_interruption_survives_smoothing():
    # A field reporter's clip runs seconds, not milliseconds: it is a speaker turn
    # and must keep its own label (and therefore its own voice).
    turns = mkturns([(0.0, 10.0, "A"), (10.2, 12.7, "B"), (12.9, 20.0, "A")])
    assert [t["speaker"] for t in segments.smooth_turns(turns)] == ["A", "B", "A"]
    # Neither does a blip that is not surrounded by one voice, nor one sitting
    # across a real pause rather than inside somebody's speech.
    edges = mkturns([(0.0, 10.0, "A"), (10.0, 10.2, "B"), (10.2, 20.0, "C")])
    assert [t["speaker"] for t in segments.smooth_turns(edges)] == ["A", "B", "C"]
    apart = mkturns([(0.0, 10.0, "A"), (11.0, 11.2, "B"), (12.2, 20.0, "A")])
    assert [t["speaker"] for t in segments.smooth_turns(apart)] == ["A", "B", "A"]


def test_a_blip_no_longer_shatters_a_sentence():
    spec = [(0.0, "aa"), (0.4, "bb"), (0.8, "cc"), (1.2, "dd"),
            (1.6, "ee"), (2.0, "ff"), (2.4, "gg")]
    blip = mkturns([(0.0, 1.3, "A"), (1.3, 1.55, "B"), (1.55, 3.0, "A")])
    # Word labels and boundary cuts read the same view: a turn too short to be a
    # speaker turn neither renames a word nor cuts the segment.
    words = mkwords(spec)
    segments.assign_word_speakers(words, blip)
    assert {w["spk"] for w in words} == {"A"}
    segs = segments.words_to_segments(words, turns=blip)
    assert len(segs) == 1 and segs[0]["speaker"] == "A"
    # ...while a real turn of the same shape still takes its words and cuts the
    # sentence into the three lines the two voices actually spoke.
    real = mkturns([(0.0, 1.3, "A"), (1.3, 2.2, "B"), (2.2, 3.0, "A")])
    other = mkwords(spec)
    segments.assign_word_speakers(other, real)
    assert [s["speaker"] for s in segments.words_to_segments(other, turns=real)] == \
        ["A", "B", "A"]


def test_a_blip_never_hides_the_pause_around_it():
    # Silence cuts still read every turn as diarized: dropping a blip from the
    # handover view must not make its speech look like a gap in the audio.
    turns = mkturns([(0.0, 0.9, "A"), (0.9, 1.1, "B"), (1.1, 2.0, "A")])
    assert segments._turn_boundaries(turns) == []
    # The same blip with a second of silence on either side: both pauses still cut
    # (protected, at their midpoints), and neither cut is a speaker handover.
    quiet = mkturns([(0.0, 0.5, "A"), (1.5, 1.7, "B"), (2.7, 3.5, "A")])
    assert segments._turn_boundaries(quiet) == [(1.0, True), (2.2, True)]


def test_handoff_start_snaps_back_to_the_diarization_onset():
    # Whisper times the incoming voice's first word late; pyannote heard it start
    # 0.25s earlier, and the dub (or the kept original audio) belongs there.
    segs = [{"start": 5.0, "end": 8.0, "speaker": "A", "text": "aa"},
            {"start": 8.5, "end": 10.0, "speaker": "B", "text": "bb"}]
    turns = mkturns([(4.9, 8.1, "A"), (8.25, 10.1, "B")])
    segments.snap_speaker_handoffs(segs, turns)
    assert segs[1]["start"] == pytest.approx(8.25)
    assert segs[0]["start"] == pytest.approx(4.9)   # the first line is a handoff too
    # A same-speaker continuation is left alone: its onset is not a handoff, and
    # the pause before it is the source's own.
    same = [{"start": 5.0, "end": 8.0, "speaker": "A", "text": "aa"},
            {"start": 9.0, "end": 10.0, "speaker": "A", "text": "bb"}]
    segments.snap_speaker_handoffs(same, mkturns([(4.9, 8.1, "A"), (8.8, 10.1, "A")]))
    assert same[1]["start"] == 9.0


def test_handoff_snap_never_overlaps_or_moves_a_start_later():
    # An onset behind the previous segment's end is unreachable (placements must
    # stay ordered), one further than HANDOFF_SNAP is not evidence about this
    # segment, and an onset *after* the first word would strand the speech.
    def snapped(seg_start, turn_start, prev_end=8.0):
        segs = [{"start": 5.0, "end": prev_end, "speaker": "A", "text": "aa"},
                {"start": seg_start, "end": 12.0, "speaker": "B", "text": "bb"}]
        segments.snap_speaker_handoffs(segs, mkturns([(4.9, prev_end + 0.1, "A"),
                                                      (turn_start, 12.1, "B")]))
        return segs[1]["start"]

    assert snapped(8.1, 7.9) == 8.1                 # onset inside the previous segment
    assert snapped(8.5, 7.5) == 8.5                 # 1.0s away — beyond HANDOFF_SNAP
    assert snapped(8.5, 8.7) == 8.5                 # never later than the first word
    assert snapped(8.5, 8.49) == 8.5                # a rounding-level move is refused


def test_a_long_monologue_run_is_never_split_into_two_voices():
    # One person holding the floor with pauses is not two people pyannote fused:
    # splitting such a run renames half the dominant speaker's turns for good.
    n = 2 * segments.REFINE_MAX_TURNS
    alternating = [[1.0 if i == j else (0.9 if i % 2 == j % 2 else 0.05)
                    for j in range(n)] for i in range(n)]
    assert segments._split_embedding_clusters(alternating) is None
    assert segments._split_embedding_clusters([r[:6] for r in alternating[:6]]) is not None


def test_two_clusters_no_wider_apart_than_one_of_them_is_wide(monkeypatch):
    # One voice recorded two ways clears the absolute cut as easily as two voices
    # do; the split needs the gap between the clusters to beat the widest merge
    # inside them as well.
    sims = [[1.0 if i == j else (0.21 if i % 2 == j % 2 else 0.15) for j in range(6)]
            for i in range(6)]
    assert segments._split_embedding_clusters(sims) is None
    monkeypatch.setattr(segments, "REFINE_MARGIN", 1.0)
    assert segments._split_embedding_clusters(sims) == [0, 1, 0, 1, 0, 1]


def test_trim_remnant_rejoins_the_span_it_was_cut_from():
    # Measured (news_wa4, 11.8-14.6s): the span ends where the voice stopped, but
    # Whisper's last word smears a second past it, leaving "the" as a one-word
    # segment of its own — separately translated, separately voiced, and pushed
    # off the seam on the timeline.
    said = [(11.8, 12.01, "The"), (12.01, 12.35, "national"), (12.35, 12.93, "rabbinical"),
            (12.93, 13.19, "court"), (13.19, 13.49, "against"), (13.49, 14.85, "the")]
    words = [{"t": t, "end": e, "text": w, "spk": "S0"} for t, e, w in said]
    asr = [{"id": 0, "start": 11.8, "end": 14.85, "speaker": "S0",
            "text": " ".join(w["text"] for w in words)}]
    spans = [{"start": 11.8, "end": 13.57, "lang": "en",
              "text": "The national rabbinical court against",
              "words": [{"t": t, "text": w, "spk": "S0"} for t, _e, w in said[:5]]}]
    out = segments.splice_foreign_spans(asr, spans, words)
    assert [s["text"] for s in out] == ["The national rabbinical court against the"]
    assert (out[0]["start"], out[0]["end"]) == (11.8, 14.85)


def test_a_remnant_that_is_not_the_span_s_language_stays_its_own_segment():
    # The remnant is source-language speech: it has to be dubbed, so it can never
    # be absorbed into a span that plays its original audio.
    words = [{"t": 11.8, "end": 13.4, "text": "English", "spk": "S0"},
             {"t": 13.5, "end": 14.6, "text": "עברית", "spk": "S0"}]
    asr = [{"id": 0, "start": 11.8, "end": 14.6, "speaker": "S0", "text": "English עברית"}]
    spans = [{"start": 11.8, "end": 13.45, "lang": "en", "text": "English",
              "words": [{"t": 11.8, "text": "English", "spk": "S0"}]}]
    out = segments.splice_foreign_spans(asr, spans, words)
    assert [s["text"] for s in out] == ["English", "עברית"]


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


def test_trimmed_segment_keeps_straddling_words_but_not_the_span_s():
    # The source-language ASR hallucinates over target-language speech, so a trim must
    # drop what lies inside the span — but a word straddling the boundary is this
    # segment's own and has to survive, or the line loses its verb.
    words = [{"t": 9.0, "end": 9.4, "text": "hallucinated"},      # inside the span
             {"t": 9.9, "end": 10.6, "text": "straddling"},       # crosses the boundary
             {"t": 10.7, "end": 11.2, "text": "inside"}]
    asr = [{"id": 0, "start": 9.0, "end": 11.2, "speaker": "A", "text": "…"}]
    spans = [{"start": 8.8, "end": 10.2, "text": "the other speaker",
              "words": [{"t": 8.8, "text": "the"}]}]
    out = segments.splice_foreign_spans(asr, spans, words)
    trimmed = [s for s in out if s["start"] == 10.2]
    assert trimmed and trimmed[0]["text"] == "straddling inside"


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


def test_uncovered_keep_waits_for_the_previous_speaker_to_stop():
    # The gap opens where the transcript stops, but the source voice is still
    # sounding there — playing original audio from that instant airs the tail of the
    # line just dubbed (the "ה" of "…באומנות רבה" at 1:34). It starts at the pause.
    hop = 0.1
    levels = [0.09] * 20 + [0.001] * 5 + [0.09] * 40   # speech, pause at 2.0s, speech
    segs = [{"id": 0, "start": 0.0, "end": 1.5, "speaker": "A", "text": "…",
             "keep": False, "keep_reason": None}]
    segments.fill_uncovered_audible(segs, levels, hop, 6.5, is_target_lang=lambda a, b: True,
                                    voice_levels=levels)
    added = [s for s in segs if s.get("keep_reason") == "uncovered"]
    assert added and added[0]["start"] == pytest.approx(2.1, abs=1e-6)   # one frame past the pause edge

    # With no pause in reach the gap is left where it was, rather than silently
    # swallowing a second of a region that may be someone talking.
    segs = [{"id": 0, "start": 0.0, "end": 1.5, "speaker": "A", "text": "…",
             "keep": False, "keep_reason": None}]
    segments.fill_uncovered_audible(segs, [0.09] * 65, hop, 6.5, is_target_lang=lambda a, b: True,
                                    voice_levels=[0.09] * 65)
    added = [s for s in segs if s.get("keep_reason") == "uncovered"]
    assert added and added[0]["start"] == pytest.approx(1.6, abs=hop)


def test_a_foreign_span_is_kept_even_with_no_text():
    # A span in a language nothing here reads carries no words to judge by script.
    # Judged that way it would be filed as transcript noise and dropped, leaving
    # silence where somebody is speaking, so the span itself is what marks it.
    spans = [{"start": 10.0, "end": 14.0, "lang": "ar", "text": "…",
              "words": [{"t": 10.0, "text": "…"}]}]
    segs = [{"id": 0, "start": 0.0, "end": 5.0, "speaker": "A", "text": "עברית"},
            {"id": 1, "start": 10.0, "end": 14.0, "speaker": "B", "text": "…"}]
    segments.mark_keep(segs, spans)
    assert (segs[1]["keep"], segs[1]["keep_reason"]) == (True, "foreign")
    assert not segs[0]["keep"]          # the source language is still dubbed
    # A target-language span keeps its own reason, so the report still tells them apart.
    en = [{"id": 0, "start": 10.0, "end": 14.0, "speaker": "B", "text": "Frankly, I agree."}]
    segments.mark_keep(en, [{"start": 10.0, "end": 14.0, "lang": "en",
                             "text": "Frankly, I agree.", "words": [{"t": 10.0, "text": "x"}]}])
    assert en[0]["keep_reason"] == "latin"
    # ...including a target-language line with no letters in it at all.
    digits = [{"id": 0, "start": 10.0, "end": 14.0, "speaker": "B", "text": "330 ,000"}]
    segments.mark_keep(digits, [{"start": 10.0, "end": 14.0, "lang": "en",
                                 "text": "330,000", "words": [{"t": 10.0, "text": "x"}]}])
    assert digits[0]["keep_reason"] == "latin"


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
    # Any untranslated script, not Hebrew alone.
    assert not translate.is_target_text("مرحبا بالعالم")
    assert not translate.is_target_text("Привет мир")


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


def test_strip_adjacent_repeat_handles_sentence_final_echo():
    # ASR echo carried through the source ("65%. אחוז") lands as a one-word
    # sentence duplicating the word before it; the detector is punctuation-blind,
    # so the stripper must accept sentence punctuation between the copies too.
    strip = translate._strip_adjacent_repeat
    assert (strip("Асад правил шестьюдесятью пятью процентами. Процентами.", "ru")
            == "Асад правил шестьюдесятью пятью процентами.")
    assert (strip("Assad controlled sixty-five percent. Percent.", "en")
            == "Assad controlled sixty-five percent.")
    # mid-sentence echo across a period, later text preserved
    assert (strip("about sixty percent. Percent. Then he entered Idlib.", "en")
            == "about sixty percent. Then he entered Idlib.")
    # no punctuation between copies → untouched here (detector-level concern)
    assert strip("I know that that is true.", "en") == "I know that that is true."


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
    # Any trailing aside on its own line, however it is worded.
    assert strip("It is a state.\n(The source text appears to be corrupted.)") == "It is a state."
    # An inline parenthesis is part of the line and stays.
    assert strip("It is a state (a small one) after all.") == "It is a state (a small one) after all."
    # Clean output is untouched.
    assert strip("Through Qatari gas, Qatar is bribing Europe.") == \
        "Through Qatari gas, Qatar is bribing Europe."


def test_strip_editorial_removes_channel_leaks_and_leading_junk():
    strip = translate._strip_editorial
    # A re-opened thinking channel whose markers were stripped leaves a bare label
    # line; TTS spoke the word "thought" on two segments of the al-Sharaa episode.
    assert strip("thought\nThe name rises to the headlines.") == \
        "The name rises to the headlines."
    assert strip("Thinking\nIt is a state.") == "It is a state."
    # Stray leading punctuation is never how a spoken line starts.
    assert strip("/In the year 2013, I am the commander.") == \
        "In the year 2013, I am the commander."
    assert strip(".Be careful of these areas.") == "Be careful of these areas."
    # Legitimate openers survive: quotes, parentheses, digits.
    assert strip('"We are here," he said.') == '"We are here," he said.'
    assert strip("(quietly) We are here.") == "(quietly) We are here."
    assert strip("600 men crossed from Iraq.") == "600 men crossed from Iraq."
    # A sentence that merely contains the word is untouched.
    assert strip("The thought never left him.") == "The thought never left him."


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


def test_preceding_line_is_context_never_content(monkeypatch):
    # "יגנו עליה" alone reads as "condemn her"; the line before settles it as "defend".
    # It must reach the prompt marked as background, and never be translated itself.
    seen = {}

    def capture(tok, mdl, prompt, n):
        seen["p"] = prompt
        return "They will defend her."

    monkeypatch.setattr(translate, "_run", capture)
    translate.generate(None, None, "יגנו עליה", source="he", target="en",
                       preceding="יעשו הכל כדי להגן על קטאר")
    prompt = seen["p"]
    assert "להגן על קטאר" in prompt
    assert "do not translate it" in prompt
    assert prompt.rstrip().endswith("Hebrew: יגנו עליה")     # the segment is the last word
    # Without one the prompt is unchanged.
    seen.clear()
    translate.generate(None, None, "יגנו עליה", source="he", target="en")
    assert "background only" not in seen["p"]


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


def test_load_picks_the_subprocess_backend_off_mac(monkeypatch):
    # Without MLX (any Linux box) load() must return the (None, handle, None)
    # triple, spawning the worker through its own uv project venv.
    created = {}

    class FakeHandle:
        def __init__(self, cmd, **kw):
            created["cmd"] = cmd
            self.own_gpu = kw.get("own_gpu", False)

    monkeypatch.setattr(translate, "_mlx_available", lambda: False)
    monkeypatch.setattr(translate, "_spare_gpu", lambda: None)
    monkeypatch.setattr(translate, "WorkerHandle", FakeHandle)
    processor, model, device = translate.load()
    assert processor is None and device is None
    assert isinstance(model, FakeHandle)
    joined = " ".join(created["cmd"])
    assert "--project" in created["cmd"] and "worker.py" in joined


def test_own_gpu_worker_is_persistent_and_reused(monkeypatch):
    # With a spare GPU the worker survives free() and load() hands it back,
    # instead of paying the 24 GB reload every timeline round.
    spawned = []

    class FakeProc:
        def poll(self):
            return None

    class FakeHandle:
        def __init__(self, cmd, **kw):
            spawned.append(cmd)
            self.own_gpu = kw.get("own_gpu", False)
            self._proc = FakeProc()
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(translate, "_mlx_available", lambda: False)
    monkeypatch.setattr(translate, "_spare_gpu", lambda: "1")
    monkeypatch.setattr(translate, "WorkerHandle", FakeHandle)
    monkeypatch.setattr(translate, "_WORKER", None)
    assert not translate.exclusive_device()
    _, first, _ = translate.load()
    assert first.own_gpu
    translate.free(first)
    assert not first.closed
    _, again, _ = translate.load()
    assert again is first and len(spawned) == 1
    monkeypatch.setattr(translate, "_WORKER", None)  # do not leak into other tests


def test_worker_handle_protocol_round_trip(tmp_path):
    # A stand-in worker that speaks the real protocol — ready line, then one JSON
    # response per request — exercises spawn, framing, flushing and id matching
    # without any model.
    echo = tmp_path / "echo_worker.py"
    echo.write_text(
        "import json, sys\n"
        "print(json.dumps({'ready': True}), flush=True)\n"
        "for line in sys.stdin:\n"
        "    req = json.loads(line)\n"
        "    print(json.dumps({'id': req['id'], 'text': req['user_text'].upper()},\n"
        "          ensure_ascii=False), flush=True)\n",
        encoding="utf-8")
    handle = translate.WorkerHandle([sys.executable, str(echo)], ready_timeout=30)
    try:
        assert handle.request("hello world", 10) == "HELLO WORLD"
        assert handle.request('quote " and שלום', 10) == 'QUOTE " AND שלום'   # JSON + UTF-8 survive
    finally:
        handle.close()


def test_worker_handle_reports_a_dead_worker(tmp_path):
    # A worker that dies mid-request must surface as a clear error, not a hang.
    dying = tmp_path / "dying_worker.py"
    dying.write_text(
        "import json, sys\n"
        "print(json.dumps({'ready': True}), flush=True)\n"
        "sys.stdin.readline()\n"
        "sys.exit(1)\n",
        encoding="utf-8")
    handle = translate.WorkerHandle([sys.executable, str(dying)], ready_timeout=30)
    try:
        with pytest.raises(RuntimeError, match="translator worker"):
            handle.request("hello", 10)
    finally:
        handle.close()


def test_run_routes_worker_output_through_shared_postprocessing():
    # The subprocess backend must get the same editorial/marker stripping as MLX.
    class FakeHandle(translate.WorkerHandle):
        def __init__(self):
            pass

        def request(self, user_text, max_new_tokens, timeout=600.0):
            return "It is a state.\n(Note: the source appears garbled.)<end_of_turn>"

    assert translate._run(None, FakeHandle(), "text", 50) == "It is a state."


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
