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
        # so reconstructing dur from it drifts sub-frame harmless for placement.
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
    # English word (1.45s into the clip) + SPAN_END_PAD stopping early is the safe
    # side, since the alternative is airing whoever speaks next.
    assert (spans[0]["start"], spans[0]["end"]) == (2.0, 3.7)


def test_span_ends_where_the_voice_stops(monkeypatch):
    # Whisper's last word bunches early; the speaker is still talking past it, so the
    # span runs to the pause instead of cutting them off mid-word (1:51).
    spans = _detect(monkeypatch, [(2.0, 4.0), (8.0, 10.0)],
                    [("en", 0.9), ("he", 1.0)], ["Qatar is a dangerous enemy"], pause=4.1)
    assert spans[0]["end"] == 4.1

    # And when the classifier calls English well past where the voice actually stopped,
    # the pause still wins keeping the rest would air the source speaker (3:25).
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
    # with no subtitle text dubbing it from the source model's gibberish is worse.
    monkeypatch.setattr(transcript, "_sounds_foreign", lambda *a, **k: "ar")
    spans = _detect(monkeypatch, [(2.0, 6.0), (8.0, 10.0)],
                    [("ar", 0.9), ("he", 1.0)], [])
    assert len(spans) == 1
    assert (spans[0]["start"], spans[0]["end"], spans[0]["lang"]) == (2.0, 6.0, "ar")
    assert spans[0]["text"] == "…"

    # The source language itself is never kept that is what gets dubbed.
    assert _detect(monkeypatch, [(2.0, 6.0)], [("he", 1.0)], []) == []

    # And an unconfirmed third language is left to the dub rather than aired blind.
    monkeypatch.setattr(transcript, "_sounds_foreign", lambda *a, **k: None)
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
    monkeypatch.setattr(transcript, "_sounds_foreign", lambda *a, **k: "ar")
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


class _FakeAsr:
    """A whisper stand-in that reads every clip the same scripted way."""

    def __init__(self, text, logprob):
        self.text, self.logprob = text, logprob

    def transcribe(self, *a, **k):
        return iter([_FakeSeg(self.text, avg_logprob=self.logprob)]), object()


def test_target_asr_outvotes_a_confident_source_read(monkeypatch):
    # The muted-English bug. VoxLingua has no usable opinion (its documented
    # mislabels sit at 0.34-0.60) and the Hebrew fine-tune TRANSLITERATES the
    # English speech into Hebrew script at -0.38 above the fail bar so the
    # passage is judged "source language" and dubbed over a man speaking English.
    # A model that actually reads the target language is the honest witness.
    from dubbing import audio
    monkeypatch.setattr(audio, "decode_mono", lambda *a, **k: [0.0])
    monkeypatch.setattr(transcript, "detect_language", lambda lid, clip: ("mi", 0.5))
    lying = _FakeAsr("איי הד דה סיים קונסרנס", -0.38)
    english = _FakeAsr("I had the same concerns about this", -0.12)
    sounds = transcript._sounds_foreign

    # Without the witness: dubbed (today's behaviour, and the bug).
    assert sounds(object(), lying, "voc.wav", 0.0, 8.0, "he") is None
    # With it: named as the target language, so it becomes a real target span.
    assert sounds(object(), lying, "voc.wav", 0.0, 8.0, "he",
                  tgt_model=english, target="en") == "en"

    # Controls the witness must not fire on source-language speech:
    honest = _FakeAsr("משפט עברי נקי לגמרי", -0.20)
    # ...the target model reads real Hebrew as low-confidence gibberish,
    gibberish = _FakeAsr("the the the a a", -1.10)
    assert sounds(object(), honest, "voc.wav", 0.0, 8.0, "he",
                  tgt_model=gibberish, target="en") is None
    # ...a fluent read that does not beat the source model's own read proves nothing,
    assert sounds(object(), honest, "voc.wav", 0.0, 8.0, "he",
                  tgt_model=_FakeAsr("and this is what he said then", -0.22),
                  target="en") is None
    # ...a known stock hallucination is not speech ("thanks for watching" over music),
    assert sounds(object(), lying, "voc.wav", 0.0, 8.0, "he",
                  tgt_model=_FakeAsr("Thanks for watching!", -0.05), target="en") is None
    # ...and two words are too little to tell a phrase from a decode artifact.
    assert sounds(object(), lying, "voc.wav", 0.0, 8.0, "he",
                  tgt_model=_FakeAsr("Yeah okay", -0.05), target="en") is None
    # A confident source-language label still outranks everything.
    monkeypatch.setattr(transcript, "detect_language", lambda lid, clip: ("he", 0.99))
    assert sounds(object(), lying, "voc.wav", 0.0, 8.0, "he",
                  tgt_model=english, target="en") is None
    # And so does a confidently NAMED third language: a target-forced decoder always
    # returns target-language text, so it may not rename the Arabic quote to English.
    monkeypatch.setattr(transcript, "detect_language", lambda lid, clip: ("ar", 0.90))
    assert sounds(object(), _FakeAsr("gibberish", -0.64), "voc.wav", 0.0, 8.0, "he",
                  tgt_model=english, target="en") == "ar"


def test_a_target_verdict_becomes_a_target_span(monkeypatch):
    # A passage the classifier cannot name but the target ASR reads cleanly is
    # routed to the TARGET branch: it gets subtitle text, refined edges and
    # lang == target, so segments/mark_keep files it as a target keep the editor
    # can see not an unnamed "foreign" keep with "…" for text.
    monkeypatch.setattr(transcript, "_sounds_foreign", lambda *a, **k: "en")
    spans = _detect(monkeypatch, [(2.0, 6.0)], [(None, 0.2)],
                    ["I had the same concerns"])
    assert len(spans) == 1
    assert spans[0]["lang"] == "en"
    assert spans[0]["text"] == "I had the same concerns"
    assert spans[0]["words"][0]["text"] == "I"
    # And that span makes a real, keep=True, target-reason segment downstream.
    segs = [{"id": 0, "start": spans[0]["start"], "end": spans[0]["end"],
             "speaker": "A", "text": spans[0]["text"]}]
    segments.mark_keep(segs, spans)
    assert (segs[0]["keep"], segs[0]["keep_reason"]) == (True, "latin")


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
    # 0.4s spoken then a real 0.6s silence the char-count guess would have
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
    # The extension never crosses into the next segment (keep or dub) the
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
    # would play a fraction of a second of original audio the 0:31 voice jump).
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
    # change with no pause must not cut the sentence the majority vote names it.
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
    assert snapped(8.5, 7.5) == 8.5                 # 1.0s away beyond HANDOFF_SNAP
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
    # segment of its own separately translated, separately voiced, and pushed
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
    # drop what lies inside the span but a word straddling the boundary is this
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
    # sounding there playing original audio from that instant airs the tail of the
    # line just dubbed (the "ה" of "…באומנות רבה" at 1:34). It starts at the pause.
    hop = 0.1
    levels = [0.09] * 20 + [0.001] * 5 + [0.09] * 40   # speech, pause at 2.0s, speech
    segs = [{"id": 0, "start": 0.0, "end": 1.5, "speaker": "A", "text": "…",
             "keep": False, "keep_reason": None}]
    segments.fill_uncovered_audible(segs, levels, hop, 6.5, is_target_lang=lambda a, b: True,
                                    voice_levels=levels)
    added = [s for s in segs if s.get("keep_reason") == "spoken_target"]
    assert added and added[0]["start"] == pytest.approx(2.1, abs=1e-6)   # one frame past the pause edge

    # With no pause in reach the gap is left where it was, rather than silently
    # swallowing a second of a region that may be someone talking.
    segs = [{"id": 0, "start": 0.0, "end": 1.5, "speaker": "A", "text": "…",
             "keep": False, "keep_reason": None}]
    segments.fill_uncovered_audible(segs, [0.09] * 65, hop, 6.5, is_target_lang=lambda a, b: True,
                                    voice_levels=[0.09] * 65)
    added = [s for s in segs if s.get("keep_reason") == "spoken_target"]
    assert added and added[0]["start"] == pytest.approx(1.6, abs=hop)


def test_uncovered_fill_judges_the_whole_gap_not_only_its_first_window():
    # The language witness answers for one LID window (4s). Asking it once, at the
    # gap's start, decides a twenty-second gap on its first four seconds: a gap that
    # opens on a music sting and then carries an English speaker was answered "not
    # target" and filled with nothing the speaker played neither dubbed nor kept.
    # Each window is judged on its own and the target-language run is what gets kept.
    hop = 0.1
    levels = [0.09] * 200                     # audible throughout
    segs = [{"id": 0, "start": 0.0, "end": 2.0, "speaker": "A", "text": "…",
             "keep": False, "keep_reason": None}]
    asked: list[tuple[float, float]] = []

    def is_target(a: float, b: float) -> bool:
        asked.append((round(a, 1), round(b, 1)))
        return a >= 9.99                      # the English starts 8s into the gap



    segments.fill_uncovered_audible(segs, levels, hop, 20.0, is_target_lang=is_target,
                                    voice_levels=levels, win=4.0)
    added = [s for s in segs if s.get("keep_reason") == "spoken_target"]
    assert len(asked) > 1                     # the whole gap is judged, not its head
    # The gap opens at 2.1s and is walked in 4s windows; only the target-language
    # run inside it is kept, and the rest of the gap is left to the mix's floor.
    assert [(s["start"], s["end"]) for s in added] == [(10.1, 20.0)]
    assert added[0]["keep"] is True

    # A gap with no target speech in it anywhere is still left alone the mix's
    # vocals fill is the floor there, and a keep would double the music bed.
    segs = [{"id": 0, "start": 0.0, "end": 2.0, "speaker": "A", "text": "…",
             "keep": False, "keep_reason": None}]
    segments.fill_uncovered_audible(segs, levels, hop, 20.0,
                                    is_target_lang=lambda a, b: False,
                                    voice_levels=levels, win=4.0)
    assert len(segs) == 1


def test_per_segment_evidence_outvotes_the_speaker_prior():
    # SPEAKER_EN_RATIO is a speaker-level prior: once a speaker crosses it, every
    # one of their segments is kept including the ones where they genuinely speak
    # the source language, which then never get dubbed. Per-segment evidence, when
    # it is strong, must be able to outvote the prior.
    # Segments 2 and 3 are code-switched: neither script holds a majority of the
    # letters, so the text witness has nothing to say about them and the audio
    # witness is the only thing that can move them off the prior.
    def rows():
        return [
            {"id": 0, "start": 0, "end": 6, "speaker": "A", "text": "This is English speech"},
            {"id": 1, "start": 6, "end": 12, "speaker": "A", "text": "and more English here"},
            {"id": 2, "start": 12, "end": 15, "speaker": "A", "text": "OK אז"},
            {"id": 3, "start": 15, "end": 18, "speaker": "A", "text": "בסדר fine"},
        ]

    # No witness (no LID model): the prior stands, exactly as before.
    segs = rows()
    segments.mark_keep(segs)
    assert [s["keep_reason"] for s in segs] == ["latin", "latin", "speaker_en", "speaker_en"]

    # A confident "this is the source language" on segment 3 sends it to the dub,
    # while segment 2 which the witness cannot name keeps the prior's verdict.
    segs = rows()
    said = {2: ("en", 0.72), 3: ("he", 0.97)}
    segments.mark_keep(segs, seg_lang=lambda s: said.get(s["id"]))
    assert [s["keep_reason"] for s in segs] == ["latin", "latin", "speaker_en", None]
    assert not segs[3]["keep"]

    # An unsure witness is not evidence: below the bar the prior still wins, because
    # the prior was measured and the classifier's mislabels sit in exactly that band.
    segs = rows()
    segments.mark_keep(segs, seg_lang=lambda s: ("he", 0.70) if s["id"] == 3 else None)
    assert segs[3]["keep_reason"] == "speaker_en"


def test_source_script_text_outvotes_the_speaker_prior():
    # The bug: a speaker who alternates between already-dubbed target passages and
    # untranslated source ones crosses SPEAKER_EN_RATIO, and every one of their
    # segments then rode `speaker_en` including plainly source-language lines,
    # which were never translated or dubbed. The audio witness could not save them:
    # on short clips it returns nothing, or a low-probability wrong label. The
    # segment's own text is the witness that is always there.
    def rows():
        return [
            {"id": 0, "start": 0, "end": 6, "speaker": "A", "text": "This is English speech"},
            {"id": 1, "start": 6, "end": 12, "speaker": "A", "text": "and more English here"},
            # The same speaker, now genuinely speaking the source language.
            {"id": 2, "start": 12, "end": 15, "speaker": "A",
             "text": "זה וראות את גר רבה יושבים"},
            # A code-switched line: one source word inside a target sentence. Its
            # letters are still mostly target, so it is not evidence of anything.
            {"id": 3, "start": 15, "end": 18, "speaker": "A", "text": "we visited שוק yesterday"},
        ]

    segs = rows()
    segments.mark_keep(segs)
    assert segs[0]["keep_reason"] == "latin" and segs[1]["keep_reason"] == "latin"
    # Source-script text: dubbed, whatever the speaker's prior says.
    assert (segs[2]["keep"], segs[2]["keep_reason"]) == (False, None)
    # A majority is required, not a presence: the mixed line stays kept.
    assert segs[3]["keep"] is True

    # The witness the classifier gives on these clips in the wild absent, or a
    # confident-sounding nonsense label still leaves the text witness in charge.
    for verdict in (None, ("la", 0.31), ("mi", 0.92)):
        segs = rows()
        segments.mark_keep(segs, seg_lang=lambda s, v=verdict: v)
        assert segs[2]["keep_reason"] is None
        assert segs[3]["keep"] is True

    # Same-script pair (en→es): script cannot tell the two languages apart, so it
    # is void as evidence both ways no speaker is a target speaker there and no
    # segment is flipped by its text.
    same = [
        {"id": 0, "start": 0, "end": 6, "speaker": "A", "text": "This is English speech"},
        {"id": 1, "start": 6, "end": 12, "speaker": "A", "text": "and more English here"},
        {"id": 2, "start": 12, "end": 15, "speaker": "A", "text": "Esto ya esta en espanol"},
    ]
    segments.mark_keep(same, None, "es", "en")
    assert [s["keep_reason"] for s in same] == [None, None, None]


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
        # Code-switched, no script in the majority: only the speaker rule can call
        # this one (source-script text of the same speaker gets dubbed instead —
        # see test_source_script_text_outvotes_the_speaker_prior).
        {"id": 2, "start": 10, "end": 15, "speaker": "A", "text": "OK אז"},
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
    monkeypatch.setattr(translate, "_vllm_available", lambda: False)
    monkeypatch.setattr(translate, "_vllm_installed", lambda: False)
    monkeypatch.setattr(translate, "_spare_gpu", lambda: None)
    monkeypatch.setattr(translate, "WorkerHandle", FakeHandle)
    processor, model, device = translate.load()
    assert processor is None and device is None
    assert isinstance(model, FakeHandle)
    joined = " ".join(created["cmd"])
    assert "--project" in created["cmd"] and "worker.py" in joined


def test_backend_order_is_mlx_then_vllm_then_transformers():
    # The chain, with nothing forced. The transformers worker is the floor: it is
    # also the CPU path (worker.py picks cpu when torch sees no GPU) and the only
    # CUDA backend on Windows, where vLLM has no wheels.
    pick = translate.select_backend
    assert pick("auto", mlx_ok=True, vllm_ok=True) == "mlx"
    assert pick("", mlx_ok=True, vllm_ok=False) == "mlx"
    assert pick("auto", mlx_ok=False, vllm_ok=True) == "vllm"
    assert pick("auto", mlx_ok=False, vllm_ok=False) == "transformers"


def test_backend_env_override_wins_even_against_the_probes():
    # Forcing a backend is how a user says the probe is wrong about this machine,
    # so it must not fall back silently; a typo is not a backend and auto-selects.
    pick = translate.select_backend
    assert pick("vllm", mlx_ok=True, vllm_ok=False) == "vllm"
    assert pick("transformers", mlx_ok=True, vllm_ok=True) == "transformers"
    assert pick("MLX", mlx_ok=False, vllm_ok=True) == "mlx"          # case-insensitive
    assert pick("tensorrt", mlx_ok=False, vllm_ok=True) == "vllm"    # unknown → auto


def test_vllm_backend_needs_linux_and_a_gpu_and_the_extra(monkeypatch):
    # All three conditions are load-bearing; none of them may import vllm here.
    def state(installed, platform, gpus):
        monkeypatch.setattr(translate, "_vllm_installed", lambda: installed)
        monkeypatch.setattr(translate.sys, "platform", platform)
        monkeypatch.setattr(translate, "_cuda_count", lambda: gpus)
        return translate._vllm_available()

    assert state(True, "linux", 2)
    assert not state(False, "linux", 2)      # extra not synced
    assert not state(True, "win32", 2)       # no Windows wheels
    assert not state(True, "linux", 0)       # no CUDA device


def test_worker_cmd_points_at_the_right_script_and_keeps_the_extra(monkeypatch):
    # `uv run` re-syncs the venv to the extras on its own command line, so a
    # transformers launch that dropped --extra would uninstall vLLM out from
    # under the next run's probe: the flag follows the venv, not the backend.
    monkeypatch.setattr(translate, "_vllm_installed", lambda: True)
    vllm_cmd = translate._worker_cmd("vllm")
    tf_cmd = translate._worker_cmd("transformers")
    assert "worker_vllm.py" in " ".join(vllm_cmd)
    assert vllm_cmd[vllm_cmd.index("--extra") + 1] == "vllm"
    assert tf_cmd[tf_cmd.index("--extra") + 1] == "vllm"
    assert tf_cmd[-2].endswith("worker.py")

    monkeypatch.setattr(translate, "_vllm_installed", lambda: False)
    assert "--extra" not in translate._worker_cmd("transformers")


def test_worker_handle_batches_and_reorders_replies(tmp_path):
    # A batched worker may finish short prompts first; request_many must hand the
    # answers back in the order they were asked, matched by id.
    #
    # The stand-in writes the whole batch in ONE write and then goes quiet, which
    # is the shape that broke the old select()-based reader: the first readline
    # pulls every reply into Python's buffer, the fd goes empty, and select then
    # reports nothing to read while the answers are already in hand. Anything less
    # deterministic passes by luck.
    shuffling = tmp_path / "batch_worker.py"
    shuffling.write_text(
        "import json, sys\n"
        "print(json.dumps({'ready': True}), flush=True)\n"
        "for line in sys.stdin:\n"
        "    msg = json.loads(line)\n"
        "    reqs = msg['batch'] if 'batch' in msg else [msg]\n"
        "    sys.stdout.write(''.join(\n"
        "        json.dumps({'id': r['id'], 'text': r['user_text'].upper()},\n"
        "                   ensure_ascii=False) + '\\n' for r in reversed(reqs)))\n"
        "    sys.stdout.flush()\n",
        encoding="utf-8")
    handle = translate.WorkerHandle([sys.executable, str(shuffling)], ready_timeout=30)
    try:
        assert handle.request_many([("one", 10), ("two", 10), ("שלוש", 10)]) == [
            "ONE", "TWO", "שלוש"]
        assert handle.request_many([]) == []
        assert handle.request_many([("alone", 10)]) == ["ALONE"]   # falls back to request
        assert handle.request("after the batch", 10) == "AFTER THE BATCH"
    finally:
        handle.close()


def test_both_workers_parse_the_same_protocol_line():
    # The two workers are only interchangeable if they read the same messages.
    # Their parse() is importable without any model: neither imports its engine
    # at module scope, which is also what keeps vllm out of the main venv.
    import importlib.util

    root = translate.REPO_ROOT
    parsers = []
    for name in ("worker", "worker_vllm"):
        spec = importlib.util.spec_from_file_location(
            f"translator_{name}", root / "translator" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        parsers.append(module.parse)

    for parse in parsers:
        assert parse('{"id": 3, "user_text": "hi", "max_new_tokens": 40}') == [
            {"id": 3, "user_text": "hi", "max_new_tokens": 40}]
        assert parse('{"id": 1, "user_text": "hi"}')[0]["max_new_tokens"] == 400
        assert [r["id"] for r in parse(
            '{"batch": [{"id": 7, "user_text": "a"}, {"id": 8, "user_text": "b"}]}'
        )] == [7, 8]
        with pytest.raises(Exception):
            parse('{"batch": []}')
        with pytest.raises(Exception):
            parse("not json at all")


def test_worker_handle_batch_surfaces_a_failed_item(tmp_path):
    # One blank answer out of N, silently returned, is the exact failure the ids
    # exist to prevent: an erroring item fails the whole call.
    failing = tmp_path / "failing_batch_worker.py"
    failing.write_text(
        "import json, sys\n"
        "print(json.dumps({'ready': True}), flush=True)\n"
        "for line in sys.stdin:\n"
        "    for r in json.loads(line)['batch']:\n"
        "        print(json.dumps({'id': r['id'], 'error': 'OOM'}), flush=True)\n",
        encoding="utf-8")
    handle = translate.WorkerHandle([sys.executable, str(failing)], ready_timeout=30)
    try:
        with pytest.raises(RuntimeError, match="OOM"):
            handle.request_many([("a", 10), ("b", 10)])
    finally:
        handle.close()


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
    monkeypatch.setattr(translate, "_vllm_available", lambda: False)
    monkeypatch.setattr(translate, "_vllm_installed", lambda: False)
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
    # A stand-in worker that speaks the real protocol ready line, then one JSON
    # response per request exercises spawn, framing, flushing and id matching
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


def test_a_stray_byte_on_the_workers_pipes_never_kills_the_pump(tmp_path):
    # The child's stdio is not ours to guarantee: a CUDA library, a progress bar
    # or a truncated multi-byte flush puts one undecodable byte on the pipe.
    # Decoded strictly, that raises inside the pump *thread*, which then dies
    # silently and the parent sits on `_read_line` for the whole 3600s
    # ready_timeout waiting for a worker that is already talking. A mangled log
    # character is the cheaper failure, so the pipes replace rather than raise.
    import time

    noisy = tmp_path / "noisy_worker.py"
    noisy.write_text(
        "import sys\n"
        "sys.stderr.buffer.write(b'cuda \\xff init\\n'); sys.stderr.buffer.flush()\n"
        "sys.stdout.buffer.write(b'\\xff not json\\n'); sys.stdout.buffer.flush()\n"
        "sys.stdin.readline()\n",
        encoding="utf-8")
    t0 = time.time()
    with pytest.raises(RuntimeError, match="instead of the ready line"):
        translate.WorkerHandle([sys.executable, str(noisy)], ready_timeout=20)
    # The point is that it answered at all: strict decoding waits out the timeout.
    assert time.time() - t0 < 15


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
    # `uid` is minted on save for anything that lacks one, so it is always there.
    assert set(loaded["segments"][0]) == {"id", "uid", "start", "end", "text"}


# ----------------------------------------------------------------------- passthrough
# The editor app's per-segment override: `passthrough=True` plays the original
# audio for that span, `False` dubs it, absent leaves the automatic verdict alone.
# It rides the existing keep machinery, so these tests pin the flag's effect on
# `keep` and on the derived work a flip invalidates.

def seg_for_passthrough(i, start, end, **kw):
    s = {"id": i, "start": start, "end": end, "speaker": "A", "text": "some words",
         "keep": False, "keep_reason": None}
    s.update(kw)
    return s


def test_passthrough_true_makes_the_segment_play_original_audio():
    segs = [seg_for_passthrough(0, 0.0, 3.0, passthrough=True)]
    assert segments.apply_passthrough(segs) == [0]
    assert segs[0]["keep"] and segs[0]["keep_reason"] == "user"


def test_passthrough_false_sends_a_kept_segment_down_the_dub_path():
    segs = [seg_for_passthrough(0, 0.0, 3.0, keep=True, keep_reason="latin",
                                passthrough=False)]
    assert segments.apply_passthrough(segs) == [0]
    assert not segs[0]["keep"] and segs[0]["keep_reason"] is None


def test_passthrough_absent_leaves_the_automatic_verdict_alone():
    segs = [seg_for_passthrough(0, 0.0, 3.0, keep=True, keep_reason="foreign"),
            seg_for_passthrough(1, 3.0, 6.0)]
    assert segments.apply_passthrough(segs) == []
    assert segs[0]["keep"] and segs[0]["keep_reason"] == "foreign"
    assert not segs[1]["keep"]


def test_passthrough_agreeing_with_the_automatic_verdict_keeps_its_named_reason():
    # "foreign" and "interjection" tell translate to render a subtitle; an
    # override that changes nothing must not overwrite them with "user".
    segs = [seg_for_passthrough(0, 0.0, 3.0, keep=True, keep_reason="foreign",
                                lang="ar", passthrough=True)]
    assert segments.apply_passthrough(segs) == []
    assert segs[0]["keep_reason"] == "foreign"


def test_a_flip_throws_away_the_work_made_for_the_other_path():
    # The translation, the clip and the placement of a flipped segment were all
    # made for the path it is no longer on left behind, mix would lay a dub
    # over a span meant to play as recorded.
    segs = [seg_for_passthrough(0, 0.0, 3.0, text_en="hello", text_mid="hello",
                                tts={"clip": "clips/0.wav", "dur": 2.2},
                                place={"start": 0.0, "end": 2.2}, passthrough=True)]
    segments.apply_passthrough(segs)
    for field in ("text_en", "text_mid", "tts", "place"):
        assert field not in segs[0]


def test_a_user_keep_is_subtitled_honestly_under_either_of_its_two_names():
    # One verdict, two words for it: the pipeline's own door writes
    # `keep_reason="user"`, the studio's (`edit.set_keep`) writes "manual" and
    # both stamp `passthrough`. The viewer is about to hear the target language,
    # so the source-language ASR's mangled reading of that span is not the
    # subtitle; recognising only one of the two names captions half the user's
    # overrides with the very garble that made them reach for the override.
    m = manifest.new({"input": "x", "src_lang": "he", "tgt_lang": "en"})
    m["segments"] = [
        seg_for_passthrough(0, 0.0, 3.0, text="ש חמוזה בקטב", keep=True,
                            keep_reason="user", passthrough=True),
        seg_for_passthrough(1, 3.0, 6.0, text="ש חמוזה בקטב", keep=True,
                            keep_reason="manual", passthrough=True),
    ]
    # Nothing to dub and nothing to sub-translate, so no model is ever loaded.
    translate.run(m, None, source="he", target="en")
    assert [s["text_en"] for s in m["segments"]] == ["…", "…"]


def test_applying_passthrough_twice_changes_nothing_the_second_time():
    # It runs on every invocation, so a re-run must not keep re-invalidating work.
    segs = [seg_for_passthrough(0, 0.0, 3.0, passthrough=True),
            seg_for_passthrough(1, 3.0, 6.0, keep=True, keep_reason="latin",
                                passthrough=False)]
    assert len(segments.apply_passthrough(segs)) == 2
    segs[0]["tts"] = {"clip": "clips/keep_0.wav", "dur": 3.0}
    assert segments.apply_passthrough(segs) == []
    assert segs[0]["tts"] == {"clip": "clips/keep_0.wav", "dur": 3.0}


def test_a_dub_override_on_a_wordless_span_is_refused():
    # Never silent outranks the override: with no text there is nothing to
    # translate and nothing to speak, so stripping the original audio would
    # leave the bed alone under a speaking face.
    segs = [seg_for_passthrough(0, 0.0, 3.0, text="", keep=True,
                                keep_reason="uncovered", passthrough=False)]
    assert segments.apply_passthrough(segs) == []
    assert segs[0]["keep"]


def test_passthrough_keeps_reserve_their_exact_span_on_the_timeline():
    # A keep is not stretchable and its clip is the span itself, so the timeline
    # reserves exactly those seconds and everything else places around them.
    m = {"segments": [
        seg_for_passthrough(0, 0.0, 2.0, text_en="a",
                            tts={"clip": "clips/0.wav", "dur": 1.6}),
        seg_for_passthrough(1, 2.0, 5.0, keep=True, keep_reason="user",
                            tts={"clip": "clips/keep_1.wav", "dur": 3.0}),
        seg_for_passthrough(2, 5.0, 7.0, text_en="c",
                            tts={"clip": "clips/2.wav", "dur": 1.5}),
    ]}
    items = timeline.build_items(m)
    assert [it["stretchable"] for it in items] == [True, False, True]
    places = timeline.place(items)
    timeline.assert_invariants(places, items)
    assert places[1]["start"] == pytest.approx(2.0)
    assert places[1]["end"] == pytest.approx(5.0)


def test_mix_ducks_the_bed_away_under_a_passthrough_span():
    # The kept audio carries its own background, so the bed goes to zero under it
    # rather than being ducked as it is under a dub.
    segs = [seg_for_passthrough(0, 0.0, 2.0, place={"start": 0.0, "end": 2.0}),
            seg_for_passthrough(1, 2.0, 5.0, keep=True, keep_reason="user",
                                place={"start": 2.0, "end": 5.0})]
    env = mix.build_envelope(segs, 6.0)
    assert env[int(3.5 * mix.CTRL_HZ)] == pytest.approx(0.0, abs=1e-3)
    assert env[int(1.0 * mix.CTRL_HZ)] == pytest.approx(mix.DUCK_DUB, abs=0.05)


def test_overrides_survive_a_re_segmentation_by_time_not_by_id():
    # The segments stage rebuilds and renumbers everything, so the override has
    # to find the segment covering the same moment.
    old = [seg_for_passthrough(7, 10.0, 14.0, passthrough=True),
           seg_for_passthrough(8, 14.0, 18.0, passthrough=False)]
    saved = segments.saved_overrides(old)
    assert saved == [(10.0, 14.0, True), (14.0, 18.0, False)]
    rebuilt = [seg_for_passthrough(0, 0.0, 9.8),
               seg_for_passthrough(1, 9.8, 14.1),
               seg_for_passthrough(2, 14.1, 17.9)]
    assert segments.carry_passthrough(rebuilt, saved) == 2
    assert "passthrough" not in rebuilt[0]
    assert rebuilt[1]["passthrough"] is True
    assert rebuilt[2]["passthrough"] is False


def test_an_override_never_spreads_across_a_merge_or_a_split():
    # Both directions must agree, or one marked line would hand its verdict to
    # every line the re-segmentation merged it with.
    saved = [(10.0, 12.0, True)]
    merged = [seg_for_passthrough(0, 0.0, 40.0)]        # old span is 5% of the new one
    assert segments.carry_passthrough(merged, saved) == 0
    assert "passthrough" not in merged[0]
    split = [seg_for_passthrough(0, 10.0, 10.4), seg_for_passthrough(1, 10.4, 12.0)]
    assert segments.carry_passthrough(split, saved) == 1   # only the substantial half
    assert "passthrough" not in split[0]
    assert split[1]["passthrough"] is True


def test_detected_lang_is_advisory_and_decides_nothing():
    # The app reads it to SUGGEST passthrough; the pipeline must not act on it.
    segs = [seg_for_passthrough(0, 0.0, 4.0), seg_for_passthrough(1, 4.0, 8.0)]
    runs = [{"start": 0.0, "end": 4.0, "lang": "en"},
            {"start": 4.0, "end": 8.0, "lang": "he"}]
    segments.stamp_detected_lang(segs, runs)
    assert [s["detected_lang"] for s in segs] == ["en", "he"]
    assert not any(s["keep"] for s in segs)               # nothing auto-passed through


def test_detected_lang_needs_most_of_the_segment_and_never_overwrites_a_span_s():
    straddling = [seg_for_passthrough(0, 3.0, 7.0)]      # half en, half he
    segments.stamp_detected_lang(straddling, [{"start": 0.0, "end": 5.0, "lang": "en"},
                                              {"start": 5.0, "end": 9.0, "lang": "he"}])
    assert "detected_lang" not in straddling[0]
    from_span = [seg_for_passthrough(0, 0.0, 4.0, detected_lang="ar")]
    segments.stamp_detected_lang(from_span, [{"start": 0.0, "end": 4.0, "lang": "he"}])
    assert from_span[0]["detected_lang"] == "ar"
    unlabelled = [seg_for_passthrough(0, 0.0, 4.0)]
    segments.stamp_detected_lang(unlabelled, [{"start": 0.0, "end": 4.0, "lang": ""}])
    assert "detected_lang" not in unlabelled[0]


def test_passthrough_and_detected_lang_survive_a_manifest_save(tmp_path):
    m = manifest.new({"input": "x"})
    m["segments"] = [{"id": 0, "start": 0.0, "end": 1.0, "text": "hi",
                      "passthrough": True, "detected_lang": "en"}]
    manifest.save(tmp_path, m)
    assert manifest.load(tmp_path)["segments"][0]["passthrough"] is True
    assert manifest.load(tmp_path)["segments"][0]["detected_lang"] == "en"


def test_a_stage_reset_never_undoes_the_user_s_override():
    # Only the pipeline's own keep-flips (mt_failed, tts_failed) are re-decided.
    m = manifest.new({"input": "x"})
    m["segments"] = [{"id": 0, "start": 0.0, "end": 1.0, "keep": True,
                      "keep_reason": "user", "passthrough": True,
                      "text_en": "hi", "tts": {"clip": "a"}}]
    manifest.reset_stage(m, "translate")
    manifest.reset_stage(m, "tts")
    assert m["segments"][0]["keep"] and m["segments"][0]["keep_reason"] == "user"
    assert m["segments"][0]["passthrough"] is True


def test_passthrough_flip_keeps_a_hand_corrected_line():
    """A locked text_en is the user's word twice over: spoken when the flip goes
    to dub, shown as the subtitle when it goes to keep. The flip drops the clip
    and placement either way, but never the correction itself."""
    segs = [{"id": 0, "start": 0.0, "end": 2.0, "speaker": "A", "text": "מילים",
             "keep": True, "keep_reason": "latin", "passthrough": False,
             "text_en": "The user's own wording.", "locked": {"text_en": True},
             "tts": {"clip": "clips/x.wav", "dur": 1.0}, "place": {"start": 0.0}}]
    assert segments.apply_passthrough(segs) == [0]
    assert segs[0]["text_en"] == "The user's own wording."
    assert "tts" not in segs[0] and "place" not in segs[0]


def test_localize_copies_an_outside_input_into_the_run_once(tmp_path):
    from dubbing import fetch

    workdir = tmp_path / "run"
    workdir.mkdir()
    original = tmp_path / "elsewhere" / "clip.MP4"
    original.parent.mkdir()
    original.write_bytes(b"video-bytes" * 100)

    copy = fetch.localize(original, workdir)
    assert copy.parent == workdir and copy.name == "input.mp4"
    assert copy.read_bytes() == original.read_bytes()
    # Idempotent: same size means no second copy...
    before = copy.stat().st_mtime_ns
    assert fetch.localize(original, workdir) == copy
    assert copy.stat().st_mtime_ns == before
    # ...and a file already inside the run is left where it is.
    assert fetch.localize(copy, workdir) == copy


def test_mix_prefers_the_runs_own_copy_and_names_the_failure(tmp_path):
    from dubbing import mix

    workdir = tmp_path / "run"
    workdir.mkdir()
    gone = tmp_path / "moved-away.mp4"      # recorded, but no longer readable
    m = {"files": {"video": str(gone)}}

    # With the run's own copy present, the unreadable original does not matter.
    copy = workdir / "input.mp4"
    copy.write_bytes(b"x")
    assert mix.video_path(m, workdir) == copy

    # Without it, the failure says what happened and what to do.
    copy.unlink()
    with pytest.raises(SystemExit, match="moved, or macOS denied"):
        mix.video_path(m, workdir)

    # A readable original outside the run is copied in healing an old run.
    outside = tmp_path / "still-here.mp4"
    outside.write_bytes(b"y" * 32)
    m2 = {"files": {"video": str(outside)}}
    healed = mix.video_path(m2, workdir)
    assert healed.parent == workdir and healed.read_bytes() == outside.read_bytes()


def test_a_named_witness_outranks_the_target_script():
    # English speech inside a he→de run: Latin script, but NOT German. The old
    # script clause kept a whole video of English as "already the target" in a
    # German dub, so keep-and-dub were audibly the same run.
    spans = [{"start": 0.0, "end": 5.0, "lang": "en", "text": "What plane is missing",
              "words": [{"t": 0.4 * i, "text": w, "spk": "A"}
                        for i, w in enumerate("What plane is missing".split())]}]
    segs = [{"id": 0, "start": 0.0, "end": 4.9, "speaker": "A",
             "text": "What plane is missing"}]
    segments.mark_keep(segs, spans, target="de")
    assert segs[0]["keep"] and segs[0]["keep_reason"] == "foreign"
    # Opting in to foreign dubbing makes it dubbable, from its own language.
    segs2 = [dict(segs[0])]
    segs2[0].pop("keep"), segs2[0].pop("keep_reason")
    segments.mark_keep(segs2, spans, target="de", dub_foreign=True)
    assert not segs2[0]["keep"]
    # The same English line in a he→en run is the target, exactly as before.
    segs3 = [{"id": 0, "start": 0.0, "end": 4.9, "speaker": "A",
              "text": "What plane is missing"}]
    segments.mark_keep(segs3, spans, target="en")
    assert segs3[0]["keep"] and segs3[0]["keep_reason"] == "latin"


def test_speakable_none_for_failed_locked_dub():
    # translate.mark_failed on a user-locked "dub it" leaves keep=false with no
    # translation. tts.run used to crash on seg["text_en"] for exactly this
    # shape; speakable() names it a real state instead.
    failed = {"id": 3, "start": 0.0, "end": 2.0, "text": "שלום",
              "keep": False, "keep_reason": "mt_failed",
              "locked": {"keep": True}}
    assert tts.speakable(failed) is None
    assert tts.speakable({"text_en": "   "}) is None
    assert tts.speakable({"text_en": "Guten Tag"}) == "Guten Tag"


def test_audio_witness_outranks_target_script():
    # he→de, English speech, nameless caption span: script alone said "already
    # German". A confident audio-LID witness naming en (≠ de) outranks it, on
    # the same rule a NAMED span witness does; an unsure or absent classifier
    # changes nothing.
    def seg_for():
        return [{"id": 0, "start": 0.0, "end": 4.0, "speaker": "A",
                 "text": "This is from the side."}]
    spans = [{"start": 0.0, "end": 4.5, "lang": "",  # nameless witness
              "words": [{"t": 0.5 * i, "text": w, "spk": "A"}
                        for i, w in enumerate("This is from the side".split())]}]

    segs = seg_for()
    segments.mark_keep(segs, spans, target="de", seg_lang=lambda s: ("en", 0.95))
    assert segs[0]["keep"] and segs[0]["keep_reason"] == "foreign"

    # No classifier → the script clause keeps its say, as before.
    segs = seg_for()
    segments.mark_keep(segs, spans, target="de", seg_lang=None)
    assert segs[0]["keep"] and segs[0]["keep_reason"] == "latin"

    # Unsure verdict → below the veto bar, changes nothing.
    segs = seg_for()
    segments.mark_keep(segs, spans, target="de", seg_lang=lambda s: ("en", 0.5))
    assert segs[0]["keep_reason"] == "latin"

    # The classifier naming the TARGET agrees with the script still kept as target.
    segs = seg_for()
    segments.mark_keep(segs, spans, target="de", seg_lang=lambda s: ("de", 0.95))
    assert segs[0]["keep_reason"] == "latin"

    # Same rule on the no-span path (segment not from any detected span).
    segs = seg_for()
    segments.mark_keep(segs, [], target="de", seg_lang=lambda s: ("en", 0.95))
    assert segs[0]["keep"] and segs[0]["keep_reason"] == "foreign"

    # he→en: en IS the target, so the audio witness confirms rather than vetoes.
    segs = seg_for()
    segments.mark_keep(segs, spans, target="en", seg_lang=lambda s: ("en", 0.95))
    assert segs[0]["keep_reason"] == "latin"


def test_untranslated_reopens_translate():
    # mark_failed on a user-locked dub leaves keep=false with no text_en the
    # honest unfinished state. untranslated() is the stage gate's answer: those
    # lines mean the stage is NOT done, whatever the fingerprint says.
    segs = [
        {"id": 0, "keep": False, "text_en": None},         # unfinished
        {"id": 1, "keep": False, "text_en": "  "},         # unfinished (blank)
        {"id": 2, "keep": False, "text_en": "Guten Tag"},  # translated
        {"id": 3, "keep": True},                           # kept: no text needed
    ]
    assert translate.untranslated(segs) == [0, 1]
    assert translate.untranslated([{"id": 0, "keep": False, "text_en": "x"}]) == []


def test_declared_source_mismatch():
    from dubbing import report

    def seg(text, dur=10.0, at=0.0):
        return {"start": at, "end": at + dur, "text": text}

    m_he = {"source": {"src_lang": "he"}}
    # An English video imported as Hebrew: nearly all speech is Latin-script.
    english = [seg("This is from the side.", at=i * 10) for i in range(5)]
    mm = report.declared_source_mismatch(m_he, english)
    assert mm and mm["declared"] == "he" and mm["in_source_script"] < 0.5
    # A genuinely Hebrew video: no accusation.
    hebrew = [seg("זה מן הצד של הדברים", at=i * 10) for i in range(5)]
    assert report.declared_source_mismatch(m_he, hebrew) is None
    # Too little speech to judge under the 30s floor, silence.
    assert report.declared_source_mismatch(m_he, [seg("Hello there", dur=5.0)]) is None
    # Same-script mistakes are invisible by nature (declared en, actually de).
    m_en = {"source": {"src_lang": "en"}}
    german = [seg("Das ist von der Seite der Dinge", at=i * 10) for i in range(5)]
    assert report.declared_source_mismatch(m_en, german) is None


def test_bare_rerun_keeps_languages_and_cap():
    # A bare `python -m dubbing <input>` on an existing run must BE a re-run:
    # argparse's he/en/full-length used to overwrite the recorded pair and cap
    # — a 320-second iteration run silently became the whole 57-minute episode,
    # and an en→de project would have flipped to he→en.
    import argparse

    from dubbing import cli

    def parsed(**typed):
        ns = argparse.Namespace(src=None, tgt=None, duration=None, captions=None)
        for key in cli.RECORDED_DEFAULTS:
            setattr(ns, key, None)
        for key, value in typed.items():
            setattr(ns, key, value)
        return ns

    m = {"source": {"src_lang": "en", "tgt_lang": "de", "duration_limit": 320.0}}

    args = parsed()
    cli.resolve_settings(args, m)
    assert (args.src, args.tgt, args.duration) == ("en", "de", 320.0)

    # normalize_lang folds an untyped None to "" before resolve runs.
    args = parsed(src="", tgt="")
    cli.resolve_settings(args, m)
    assert (args.src, args.tgt) == ("en", "de")

    # A typed flag still wins, and 0 clears a recorded cap.
    args = parsed(tgt="fr", duration=0.0)
    cli.resolve_settings(args, m)
    assert (args.src, args.tgt, args.duration) == ("en", "fr", None)

    # First run, nothing typed: the documented defaults.
    args = parsed()
    cli.resolve_settings(args, None)
    assert (args.src, args.tgt, args.duration) == ("he", "en", None)


def test_a_bare_rerun_keeps_the_caption_file_it_was_given():
    # `captions` is in fetch's fingerprint, so a bare re-run that forgot it flips
    # fetch — and a flipped chain empties `m["segments"]` at the segments stage,
    # taking every lock and hand-edit with it, while fetch quietly falls back to
    # ASR. Same class of loss as --duration, same rule.
    from pathlib import Path as _Path

    from dubbing import cli

    made = cli.parse_args(["in.mp4", "--captions", "subs.json3"])
    cli.resolve_settings(made, None)
    m = manifest.new(cli.source_record(made))
    assert m["source"]["captions"] == "subs.json3"

    again = cli.parse_args(["in.mp4"])                # the user types no flags
    cli.resolve_settings(again, m)
    assert again.captions == _Path("subs.json3")
    assert cli.stage_params(again, m)["fetch"] == cli.stage_params(made, m)["fetch"]

    # A run that never had captions records the absence as "" — never None, which
    # the next run would read back as "nobody said".
    bare = cli.parse_args(["in.mp4"])
    cli.resolve_settings(bare, None)
    assert cli.source_record(bare)["captions"] == ""
    assert bare.captions is None


def test_a_restored_language_is_normalized_like_a_typed_one():
    # A manifest this CLI never wrote (an API-created project, a hand-restored
    # dir) can hold a legacy alias. `edit._args` normalizes its copy, so skipping
    # it here made the same run fingerprint differently in the studio and the CLI.
    from dubbing import cli, edit

    m = {"source": {"input": "in.mp4", "src_lang": "iw", "tgt_lang": "en"}}
    args = cli.parse_args(["in.mp4"])
    cli.resolve_settings(args, m)
    assert args.src == "he"
    assert edit._args(m).src == "he"
    assert cli.stage_params(args, m) == cli.stage_params(edit._args(m), m)


# ------------------------------------------- the gap pass's own speech gate

class _GapWord:
    def __init__(self, word, start, end, probability=0.9):
        self.word, self.start, self.end, self.probability = word, start, end, probability


class _GapSeg:
    """What faster-whisper hands back: one decode reading, shared by its words."""

    def __init__(self, avg_logprob, start, end, text):
        self.avg_logprob, self.start, self.end, self.text = avg_logprob, start, end, text
        self.words = [_GapWord(" " + w, start + 0.1 * i, start + 0.1 * i + 0.08)
                      for i, w in enumerate(text.split())]


def _gap_model(segs):
    class _Model:
        def transcribe(self, *a, **kw):
            return iter(segs), None

    return _Model()


def _audible_everywhere(monkeypatch, total=10.0, level=0.05):
    """No words known and audio above the floor throughout: one gap window."""
    import numpy as np

    from dubbing import audio

    monkeypatch.setattr(audio, "decode_mono", lambda *a, **kw: np.zeros(16))
    monkeypatch.setattr(audio, "frame_rms",
                        lambda *a, **kw: np.full(int(total / 0.1), level))


def _logged_reads(monkeypatch, levels=None, total=10.0, level=0.05):
    """`_audible_everywhere`, plus a log of which file every read came from.

    Two kinds of read: the whole-file one a level array is measured from (no
    start/end) and each window's decode clip (both). `levels` gives a per-file
    frame-energy array for the files that need one; anything unnamed is audible
    at `level` throughout. The stubs pass the path itself where the real ones
    pass samples nothing here looks at audio, only at which file it came from.
    """
    from pathlib import Path

    import numpy as np

    from dubbing import audio

    frames = int(total / 0.1)
    reads: list[tuple[str, Path, float | None]] = []

    def decode_mono(path, rate, start=None, end=None, **kw):
        reads.append(("window" if start is not None else "levels", Path(path), start))
        return Path(path)

    def frame_rms(samples, rate, hop):
        value = (levels or {}).get(Path(samples), level)
        if np.isscalar(value):
            return np.full(frames, float(value))
        return np.asarray(value, dtype=float)

    monkeypatch.setattr(audio, "decode_mono", decode_mono)
    monkeypatch.setattr(audio, "frame_rms", frame_rms)
    return reads


def _read_paths(reads, kind):
    return [path for what, path, _start in reads if what == kind]


def test_speech_only_drops_the_unreadable_decode_and_keeps_its_neighbours():
    segs = [_GapSeg(-0.10, 0.0, 2.0, "real speech"),
            _GapSeg(-0.90, 2.0, 4.0, "invented over music"),
            _GapSeg(-0.12, 4.0, 6.0, "real again")]
    assert [s.text for s in transcript.speech_only(segs)] == ["real speech", "real again"]


def test_the_gap_pass_declines_music_its_neighbours_would_have_carried(tmp_path,
                                                                      monkeypatch):
    # The window mean is a weaker test than the per-segment one: a span the main
    # pass declined as music re-entered here whenever the other segments in the
    # same window read well enough to carry the average.
    _audible_everywhere(monkeypatch)
    segs = [_GapSeg(-0.10, 0.0, 2.0, "real speech"),
            _GapSeg(-0.90, 2.0, 4.0, "invented over music"),
            _GapSeg(-0.12, 4.0, 6.0, "real again")]
    mean = sum(s.avg_logprob for s in segs) / len(segs)
    assert mean > transcript.GAP_MIN_LOGPROB      # the mean alone accepts the window
    assert segs[1].avg_logprob < transcript.ASR_MIN_LOGPROB

    words, spans = transcript.recover_gaps(_gap_model(segs), tmp_path / "source.wav",
                                           [], "en", 10.0, tgt_lang="he")
    assert [w["text"] for w in words] == ["real", "speech", "real", "again"]
    assert spans == []


def test_a_gap_window_that_is_only_music_recovers_nothing(tmp_path, monkeypatch):
    _audible_everywhere(monkeypatch)
    segs = [_GapSeg(-0.70, 0.0, 2.0, "one invented line"),
            _GapSeg(-0.90, 2.0, 4.0, "and another")]
    words, spans = transcript.recover_gaps(_gap_model(segs), tmp_path / "source.wav",
                                           [], "en", 10.0, tgt_lang="he")
    assert words == [] and spans == []


def test_the_gap_pass_judges_audibility_on_the_mix_but_listens_to_the_vocals(tmp_path,
                                                                            monkeypatch):
    # Two different questions, two different files. "Is something audible here"
    # is the mix's answer that is what plays over an uncovered span. "What is
    # being said" is the vocals' answer, same as the main pass; asked of the mix,
    # the music itself is what made this decoder invent lines over stings the
    # main pass had correctly declined.
    source = tmp_path / "source.wav"
    vocals = tmp_path / "stems" / "vocals.wav"
    vocals.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"x"), vocals.write_bytes(b"x")

    reads = _logged_reads(monkeypatch)
    segs = [_GapSeg(-0.10, 0.5, 2.5, "real speech")]
    # One known word mid-file, so there are two uncovered windows to decode and
    # "every window" is a claim with more than one case behind it.
    known_words = [{"t": 5.0, "end": 5.4, "text": "known", "brk": False}]

    transcript.recover_gaps(_gap_model(segs), source, list(known_words), "en", 10.0,
                            tgt_lang="he", listen_wav=vocals)
    # The mix nominates the windows; the vocals both confirm and are decoded.
    assert _read_paths(reads, "levels") == [source, vocals]
    decoded = _read_paths(reads, "window")
    assert len(decoded) == 2 and set(decoded) == {vocals}

    # No stems, or a vocals file the demucs stage never wrote: the mix is all
    # there is, which is the behaviour every run before this had and it is
    # measured once, not twice, because the second reading is the same array.
    for absent in (None, tmp_path / "stems" / "never_written.wav"):
        reads.clear()
        transcript.recover_gaps(_gap_model(segs), source, list(known_words), "en", 10.0,
                               tgt_lang="he", listen_wav=absent)
        assert _read_paths(reads, "levels") == [source]
        decoded = _read_paths(reads, "window")
        assert len(decoded) == 2 and set(decoded) == {source}


def _sting_then_speech(tmp_path, monkeypatch):
    """A run whose first uncovered window is a music sting and second is speech.

    The mix is equally loud over both (a sting measures 0.037-0.052 RMS, well
    over GAP_RMS_FLOOR), so window *selection* cannot tell them apart. The
    separated vocals can, by three orders of magnitude: the real hallucination
    sites measured 0.000016-0.0018 against 0.053-0.071 for real speech.
    """
    source = tmp_path / "source.wav"
    vocals = tmp_path / "stems" / "vocals.wav"
    vocals.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"x"), vocals.write_bytes(b"x")
    # One known word at 5.0s splits the file into windows 0.0-4.7 and 5.8-10.0.
    known_words = [{"t": 5.0, "end": 5.4, "text": "known", "brk": False}]
    quiet_vocals = [0.0005] * 48 + [0.06] * 52
    reads = _logged_reads(monkeypatch, levels={source: 0.045, vocals: quiet_vocals})
    return reads, source, vocals, known_words


def test_a_window_loud_in_the_mix_but_silent_in_the_vocals_is_never_decoded(tmp_path,
                                                                           monkeypatch,
                                                                           capsys):
    # The hole the earlier fix left: listening to the vocals does not help if the
    # window was handed over on the mix's word alone. Whisper given near-silence
    # is exactly what emits "thank you very much" — so the sting is never decoded
    # at all. The pre-rewrite pipeline had this guard; the rewrite dropped it.
    reads, source, vocals, known = _sting_then_speech(tmp_path, monkeypatch)
    segs = [_GapSeg(-0.10, 0.5, 2.5, "thank you very much")]

    words, spans = transcript.recover_gaps(_gap_model(segs), source, list(known), "en",
                                           10.0, tgt_lang="he", listen_wav=vocals)
    # Exactly one decode, and it is the speech window (clip starts at 5.8 - GAP_PAD).
    windows = [start for kind, _path, start in reads if kind == "window"]
    assert windows == [pytest.approx(5.8 - transcript.GAP_PAD)]
    # Nothing from the sting: the one recovered line is the speech window's.
    assert [w["text"] for w in words] == ["known", "thank", "you", "very", "much"]
    assert spans == []
    # Said out loud, with both readings, so a report reader can check the call.
    note = capsys.readouterr().err
    assert "gap 0.0-4.7s is music, not speech" in note
    assert "vocals RMS 0.000500" in note and "mix 0.045" in note
    assert "left uncovered" in note


def test_a_window_with_real_vocal_energy_is_still_decoded_and_recovered(tmp_path,
                                                                       monkeypatch):
    # The other half of the same judgement: the gate refuses silence, not gaps.
    # Same mix and same windows as the sting case only the vocals differ.
    source = tmp_path / "source.wav"
    vocals = tmp_path / "stems" / "vocals.wav"
    vocals.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"x"), vocals.write_bytes(b"x")
    known = [{"t": 5.0, "end": 5.4, "text": "known", "brk": False}]
    reads = _logged_reads(monkeypatch, levels={source: 0.045, vocals: [0.06] * 100})
    segs = [_GapSeg(-0.10, 0.5, 2.5, "real speech")]

    words, _spans = transcript.recover_gaps(_gap_model(segs), source, list(known), "en",
                                            10.0, tgt_lang="he", listen_wav=vocals)
    assert _read_paths(reads, "window") == [vocals, vocals]
    # Both windows' words came back, around the word that was already known.
    assert [w["text"] for w in words] == ["real", "speech", "known", "real", "speech"]


# ------------------------------------- a fallback slice keeps a keep's policy

def test_a_fallback_slice_on_a_dubbed_line_is_never_stretched():
    # tts's universal fallback is a `keep_*.wav` slice of the original span. It
    # is original audio, exactly its span long, so stretching it both speeds up
    # the speaker's own voice and renames the clip `fit_keep_*` — which silently
    # dropped the UI's `media.fallback` flag.
    m = {"segments": [
        {"id": 0, "start": 0.0, "end": 2.0, "keep": False,
         "tts": {"clip": "clips/keep_000000000000.wav", "dur": 2.0}},
        {"id": 1, "start": 2.0, "end": 4.0, "keep": False,
         "tts": {"clip": "clips/9f2c.wav", "dur": 2.4}},
        {"id": 2, "start": 4.0, "end": 6.0, "keep": True,
         "tts": {"clip": "clips/keep_000000004000.wav", "dur": 2.0, "span": 2.0}},
    ]}
    items = timeline.build_items(m)
    assert [it["stretchable"] for it in items] == [False, True, False]
    # …and the placement it earns: played at its own rate, like any original audio.
    assert timeline.place(items)[0]["rate"] == 1.0


# ------------------------------- a stale synthesis is not an original-audio slice

class _SliceEngine:
    """A synthesiser that only ever cuts keep slices. Records what it was asked for."""

    def __init__(self, workdir, *a, **kw):
        self.workdir, self.cut = workdir, []

    def build_speaker_refs(self):
        pass

    def keep_clip(self, seg):
        self.cut.append(seg["id"])
        clip = f"clips/keep_{int(seg['start'] * 1000):012d}.wav"
        (self.workdir / clip).parent.mkdir(parents=True, exist_ok=True)
        (self.workdir / clip).write_bytes(b"x")
        return {"clip": clip, "dur": round(seg["end"] - seg["start"], 3), "tries": 0,
                "overlap": 1.0, "verify": "keep", "span": round(seg["end"] - seg["start"], 3)}

    def clip_for(self, seg, text):
        return None

    def close(self):
        pass


def _failed_dub_with_a_stale_clip(workdir):
    """A user-locked dub whose translation was popped, its old synthesis still there."""
    (workdir / "clips").mkdir(parents=True, exist_ok=True)
    (workdir / "clips" / "9f2c.wav").write_bytes(b"x")
    return {"version": 1, "source": {"src_lang": "he", "tgt_lang": "en"},
            "files": {"source_wav": "source.wav"}, "speakers": {},
            "segments": [{"id": 0, "uid": "s0", "start": 0.0, "end": 2.0,
                          "speaker": "A", "text": "שלום", "keep": False,
                          "keep_reason": "mt_failed", "locked": {"keep": True},
                          "tts": {"clip": "clips/9f2c.wav", "dur": 1.4, "tries": 1,
                                  "overlap": 0.9, "verify": "ok", "opts": "",
                                  "text_sha": tts.text_sha("the line it used to say")}}]}


def test_a_stale_synthesis_is_replaced_by_the_original_slice(tmp_path, monkeypatch):
    # `keep_needs_slice` alone answers False for ANY clip on disk with matching
    # options — including the segment's own synthesis of a translation
    # `mark_failed` has since popped. That clip then stayed in place speaking the
    # previous line while the stage's log claimed the original plays.
    m = _failed_dub_with_a_stale_clip(tmp_path)
    seg = m["segments"][0]
    assert tts.speakable(seg) is None                    # nothing to say
    assert not tts.keep_needs_slice(seg, tmp_path)       # …and the old guard said "done"
    assert tts.needs_synthesis(seg, tmp_path)            # the stage does see it

    engines = []
    monkeypatch.setattr(tts, "Engine",
                        lambda mm, wd, **kw: engines.append(_SliceEngine(wd)) or engines[-1])
    tts.run(m, tmp_path)
    assert engines[0].cut == [0]
    assert seg["tts"]["clip"].startswith("clips/keep_")
    assert "text_sha" not in seg["tts"]                  # a slice speaks no line

    # …and it does not churn: the record IS a slice now, so the next run leaves it.
    tts.run(m, tmp_path)
    assert engines[1].cut == []
    assert seg["tts"]["clip"].startswith("clips/keep_")


# --------------------------- a translate re-entry reopens what was built on it

def test_a_translate_re_entry_reopens_the_stages_built_on_it(tmp_path, monkeypatch):
    """A hole re-enters translate under an UNCHANGED fingerprint, so every
    downstream fingerprint recomputes identical and tts/timeline/mix/report each
    declare no output files — `stage_done`'s `all([])` is True and they all report
    "up to date". A line the re-entry successfully translates would then keep
    playing its original-audio fallback forever."""
    from dubbing import STAGES, cli, report as report_mod

    workdir = tmp_path / "run"
    for rel in ("source.wav", "words.json", "dub.wav", "preview.mp4", "report.json",
                "stems/vocals.wav", "stems/background.wav"):
        (workdir / rel).parent.mkdir(parents=True, exist_ok=True)
        (workdir / rel).write_bytes(b"x")

    def finished_run(*segs):
        args = cli.parse_args([str(tmp_path / "in.mp4"), "-o", str(workdir)])
        cli.resolve_settings(args, None)
        m = manifest.new(cli.source_record(args))
        m["source"]["transcript_origin"] = "asr"
        m["segments"] = [dict(s) for s in segs]
        params = cli.stage_params(args, m)
        for stage in STAGES:
            manifest.mark_stage(m, stage, manifest.stage_fingerprint(m, stage,
                                                                     params[stage]))
        manifest.save(workdir, m)
        return m

    def line(i, text_en):
        s = {"id": i, "uid": f"s{i}", "start": 2.0 * i, "end": 2.0 * i + 2.0,
             "speaker": "A", "text": "שלום", "keep": False, "keep_reason": None,
             "tts": {"clip": f"clips/{i}.wav", "dur": 1.8},
             "place": {"start": 2.0 * i, "end": 2.0 * i + 1.8, "rate": 1.0,
                       "drift": 0.0, "clip": f"clips/{i}.wav"}}
        if text_en is None:      # mark_failed on a user-locked dub: the hole
            s["keep_reason"], s["locked"] = "mt_failed", {"keep": True}
        else:
            s["text_en"] = text_en
        return s

    ran: list[str] = []

    class FakeEngine:
        # In the tts stage's fingerprint: a stub without it changes what the CLI
        # computes, and every mark laid down above would stop matching.
        REF_BUILD = tts.Engine.REF_BUILD

        def __init__(self, *a, **kw):
            pass

        def close(self):
            pass

    def fake_translate(mm, wd, **kw):
        ran.append("translate")
        for s in mm["segments"]:
            s["text_en"] = s.get("text_en") or "a line at last"

    monkeypatch.setattr(translate, "run", fake_translate)
    monkeypatch.setattr(tts, "run", lambda mm, wd, **kw: (ran.append("tts"), FakeEngine())[1])
    monkeypatch.setattr(tts, "Engine", FakeEngine)
    monkeypatch.setattr(timeline, "run", lambda mm, wd, **kw: ran.append("timeline"))
    monkeypatch.setattr(mix, "run", lambda mm, wd: ran.append("mix"))
    monkeypatch.setattr(report_mod, "run",
                        lambda mm, wd: (ran.append("report"), {"unaccounted": []})[1])

    argv = [str(tmp_path / "in.mp4"), "-o", str(workdir)]

    # Control: nothing unfinished, so nothing re-runs.
    finished_run(line(0, "One"), line(1, "Two"))
    assert cli.main(argv) == 0
    assert ran == []

    # One untranslated line: translate re-enters, and so must everything after it.
    m = finished_run(line(0, None), line(1, "Two"))
    assert translate.untranslated(m["segments"]) == [0]
    assert cli.main(argv) == 0
    assert ran == ["translate", "tts", "timeline", "mix", "report"]
    assert manifest.load(workdir)["segments"][0]["text_en"] == "a line at last"
