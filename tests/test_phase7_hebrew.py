"""Phase 7 Hebrew as a dub target, and same-language pairs. No models loaded.

Two features that arrived together and share one idea: the target language is no
longer assumed to be something *other* than the source, and it is no longer
assumed to be something the base checkpoint can pronounce.

* **Hebrew target.** Qwen3-TTS does not speak Hebrew; a LoRA over its `talker`
  does, and only from stressed IPA. So a Hebrew line is synthesized from
  `G2P(text)` while everything stored, subtitled and ASR-verified stays Hebrew
  orthography and the clip's cache key has to carry both the adapter and the
  IPA, or a G2P change would replay the old pronunciation.
* **Same-language pairs.** he→he is a dub, not a no-op: every speech segment is
  re-voiced in the cloned voice, with the source line used verbatim and no
  translator loaded at all.

The G2P is stubbed throughout it is a 300 MB ONNX model and these are unit
tests. What is under test is the plumbing around it.
"""

from __future__ import annotations

import pytest

from dubbing import cli, hebrew, manifest, script, translate, tts
from dubbing.ttsopts import DEFAULT

# What the real G2P returns for these lines (checked against the adapter's own
# model card, which prints exactly this for the first one).
IPA = {
    "עוד רגע אני אתייחס ליהודה ושומרון.": "ʔˈod ʁˈeɡa ʔanˈi ʔetjaχˈes lihudˈa veʃomʁˈon.",
    "שלום, מה שלומך היום?": "ʃalˈom, mˈa ʃlomχˈa hajˈom?",
}


@pytest.fixture
def fake_g2p(monkeypatch):
    """`hebrew.phonemize` without the ONNX model behind it."""
    seen: list[str] = []

    def phonemize(text, *, speaker=hebrew.SPEAKER_UNKNOWN):
        seen.append(text)
        return IPA.get(text, f"<ipa:{text}>")

    monkeypatch.setattr(hebrew, "phonemize", phonemize)
    return seen


def _engine(tmp_path, *, tgt="he", src="he"):
    return tts.Engine({"source": {"src_lang": src, "tgt_lang": tgt},
                       "speakers": {}, "segments": []}, tmp_path)


# ------------------------------------------------------------- language codes


def test_every_spelling_of_hebrew_is_hebrew():
    assert hebrew.is_hebrew("he") and hebrew.is_hebrew("HE")
    # YouTube still emits the pre-1989 code; it names the same language.
    assert hebrew.is_hebrew("iw")
    assert not hebrew.is_hebrew("yi")        # Yiddish shares the script, not the model
    assert not hebrew.is_hebrew("") and not hebrew.is_hebrew(None)


def test_the_cli_normalizes_legacy_codes_once():
    assert cli.normalize_lang("iw") == "he"
    assert cli.normalize_lang(" HE ") == "he"
    assert cli.normalize_lang("ru") == "ru"


def test_the_engine_speaks_one_spelling_of_hebrew(tmp_path):
    # `language="iw"` is not a value faster-whisper accepts, and the verify ASR is
    # handed this string directly.
    assert _engine(tmp_path, tgt="iw").tgt_lang == "he"


# ------------------------------------------------------- text prepared for Hebrew


def test_a_hebrew_line_is_synthesized_from_ipa_and_recorded_as_hebrew(fake_g2p):
    line = "עוד רגע אני אתייחס ליהודה ושומרון."
    speak = tts.prepare_text(line, "he")
    assert script.is_script(speak, "he")               # the record stays orthography
    synth = tts.synthesis_text(speak, "he")
    assert synth == IPA[line]
    assert hebrew.STRESS in synth                      # the marker the LoRA learned
    assert not script.is_script(synth, "he")           # ... and the input is not


def test_every_other_target_is_handed_its_own_text_unchanged(fake_g2p):
    for lang, line in (("en", "Hello there."), ("ru", "Привет всем."),
                       ("ja", "こんにちは。")):
        assert tts.synthesis_text(line, lang) == line
    assert not fake_g2p                                 # the G2P was never consulted


def test_prepare_text_keeps_hebrew_and_drops_what_the_voice_cannot_say():
    # The same rule every target gets: runs of the TARGET script survive, a word in
    # another one does not take the line down with it.
    got = tts.prepare_text("שלום Hello עולם", "he")
    assert "Hello" not in got and "שלום" in got and "עולם" in got


def test_a_line_the_g2p_empties_is_not_synthesized(tmp_path, monkeypatch):
    monkeypatch.setattr(hebrew, "phonemize", lambda text, **kw: "   ")
    eng = _engine(tmp_path)
    assert eng._plan({"id": 1, "uid": "u1", "start": 0.0, "end": 2.0,
                      "speaker": "S1"}, "שלום עולם") is None


# ------------------------------------------------------------------- cache keys


def test_the_cache_key_carries_the_adapter_and_the_ipa(tmp_path):
    eng = _engine(tmp_path)
    args = ("שלום עולם.", "ref:1.00-4.00", 42, False, DEFAULT)
    base = eng._cache_key(*args, synth="ʃalˈom ʔolˈam.")
    # A different pronunciation of the same words is a different clip: without this
    # a changed G2P would silently replay audio of the old reading.
    assert base != eng._cache_key(*args, synth="ʃalˈom ʔolˈam!")
    assert base != eng._cache_key(*args, synth="")


def test_a_non_hebrew_target_hashes_exactly_as_it_did(tmp_path):
    """The regression guard: nothing about Hebrew may move an existing key."""
    import hashlib

    eng = _engine(tmp_path, tgt="en", src="he")
    blob = f"{tts.TTS_MODELS['1.7b']['tag']}|some text|ref:1.00-4.00|xvec|42|0"
    want = hashlib.sha1(blob.encode()).hexdigest()[:16]
    assert eng._cache_key("some text", "ref:1.00-4.00", 42, False) == want
    # ... and passing a synthesis text (which for en is the same string) is a no-op.
    assert eng._cache_key("some text", "ref:1.00-4.00", 42, False,
                          DEFAULT, synth="some text") == want


def test_hebrew_pins_the_checkpoint_the_adapter_fits(tmp_path):
    # The LoRA carries the 1.7B talker's output heads, so a per-segment override
    # cannot pull the 0.6B checkpoint out from under it.
    from dubbing import ttsopts

    eng = _engine(tmp_path)
    assert eng.model_for(ttsopts.parse({"model": "0.6b"})) == hebrew.ADAPTER_MODEL
    other = _engine(tmp_path, tgt="ru")
    assert other.model_for(ttsopts.parse({"model": "0.6b"})) == "0.6b"


def test_hebrew_verifies_with_the_hebrew_asr():
    from dubbing import transcript

    cands = tts._asr_candidates("he")
    assert transcript.WHISPER_HUB in cands
    assert cands[0].endswith(transcript.WHISPER_MODEL.name)
    # Unchanged for everyone else.
    assert tts._asr_candidates("en") == tts._ASR_CANDIDATES
    assert tts._asr_candidates("ru") == tts._ASR_CANDIDATES_MULTI


# --------------------------------------------------------------- CLI validation


def _args(**kw):
    from argparse import Namespace

    return Namespace(**{"src": "he", "tgt": "en", "tts_model": "1.7b", **kw})


def test_a_hebrew_target_without_its_models_is_refused_with_the_command(monkeypatch):
    monkeypatch.setattr(hebrew, "adapter_ready", lambda: False)
    monkeypatch.setattr(hebrew, "g2p_ready", lambda: True)
    with pytest.raises(SystemExit) as exc:
        cli.check_langs(_args(tgt="he"))
    assert hebrew.ADAPTER_DOWNLOAD in str(exc.value)


def test_the_fetch_commands_name_an_absolute_local_dir():
    """`--local-dir` is resolved against the working directory, so a relative one
    is a command that only works from the repo root. Pasted anywhere else it
    downloads 236 MB into a `models/` beside wherever the terminal happened to
    be, reports success, and leaves the Setup row red with nothing saying why.
    Every other row's command is absolute (`setup.model` interpolates the whole
    path); these two are hand-written and used by the CLI refusals as well."""
    from pathlib import Path

    for command, where in ((hebrew.ADAPTER_DOWNLOAD, hebrew.ADAPTER_DIR),
                           (hebrew.G2P_DOWNLOAD, hebrew.G2P_DIR)):
        local_dir = command.split("--local-dir", 1)[1].strip()
        assert Path(local_dir).is_absolute(), command
        assert local_dir == str(where)


def test_a_hebrew_target_needs_the_checkpoint_the_adapter_fits(monkeypatch):
    monkeypatch.setattr(hebrew, "adapter_ready", lambda: True)
    monkeypatch.setattr(hebrew, "g2p_ready", lambda: True)
    cli.check_langs(_args(tgt="he"))                          # 1.7b: fine
    with pytest.raises(SystemExit, match="1.7b"):
        cli.check_langs(_args(tgt="he", tts_model="0.6b"))


def test_a_same_language_pair_is_never_refused(monkeypatch):
    monkeypatch.setattr(hebrew, "adapter_ready", lambda: True)
    monkeypatch.setattr(hebrew, "g2p_ready", lambda: True)
    for pair in (("he", "he"), ("en", "en"), ("ru", "ru")):
        cli.check_langs(_args(src=pair[0], tgt=pair[1]))


# ------------------------------------------------------- same-language translate


def test_same_language_is_case_folded_and_needs_both_sides():
    assert translate.same_language("he", "he") and translate.same_language("HE", "he")
    assert not translate.same_language("he", "en")
    assert not translate.same_language("", "en")      # no source is not a match


def test_a_same_language_pair_pivots_nowhere():
    assert not translate.pivot_via_english("he", "he")
    assert not translate.pivot_via_english("en", "en")
    assert translate.pivot_via_english("he", "ru")    # unchanged for real pairs


def _manifest(segments):
    return {"source": {"src_lang": "he", "tgt_lang": "he"}, "speakers": {},
            "segments": segments}


def _seg(i, text, **kw):
    return {"id": i, "uid": f"u{i}", "start": float(i), "end": float(i) + 1.0,
            "speaker": "S1", "text": text, "keep": False, "keep_reason": None, **kw}


def test_same_language_translation_is_the_source_line_and_loads_no_model(tmp_path,
                                                                        monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("the translator must not load for a same-language pair")

    monkeypatch.setattr(translate, "load", boom)
    m = _manifest([_seg(1, "שלום עולם"), _seg(2, "מה שלומך היום")])
    translate.run(m, tmp_path, source="he", target="he")
    assert [s["text_en"] for s in m["segments"]] == ["שלום עולם", "מה שלומך היום"]
    assert all(not s["keep"] for s in m["segments"])       # every line is dubbed
    assert all("text_mid" not in s for s in m["segments"])  # nothing pivoted


def test_same_language_still_spells_digits_in_code(tmp_path, monkeypatch):
    monkeypatch.setattr(translate, "load",
                        lambda *a, **kw: pytest.fail("no model here"))
    m = _manifest([_seg(1, "היו שם 3 אנשים")])
    translate.run(m, tmp_path, source="he", target="he")
    assert "3" not in m["segments"][0]["text_en"]
    assert "שלוש" in m["segments"][0]["text_en"]


def test_same_language_keeps_are_the_ones_that_never_depended_on_the_pair(tmp_path,
                                                                          monkeypatch):
    """No segment is kept for "already the target language" that is the point."""
    segs = [_seg(1, "…", keep=True, keep_reason="no_text"),
            _seg(2, "שלום"),
            _seg(3, "שלום", keep=True, keep_reason="user")]
    script_keeps = []
    from dubbing import segments as segments_mod

    segments_mod.mark_keep(segs, [], "he", "he")
    for s in segs:
        script_keeps.append((s["keep"], s["keep_reason"]))
    # Only the text-less span is kept by the automatic rules: with one script on
    # both sides the script and speaker evidence is void (see `mark_keep`).
    assert script_keeps[0] == (True, "no_text")
    assert script_keeps[1] == (False, None)
    assert script_keeps[2] == (False, None)     # a user keep is re-applied later

    monkeypatch.setattr(translate, "load",
                        lambda *a, **kw: pytest.fail("no model here"))
    segs[2]["keep"], segs[2]["keep_reason"] = True, "user"
    m = _manifest(segs)
    translate.run(m, tmp_path, source="he", target="he")
    # A keep's subtitle is its own line the viewer hears exactly that.
    assert segs[2]["text_en"] == "שלום"
    assert segs[0]["text_en"] == "…"


def test_a_third_language_span_still_gets_a_real_translation(tmp_path, monkeypatch):
    """Same-language identity is per segment, not per run."""
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(translate, "load", lambda *a, **kw: ("proc", "model", None))
    monkeypatch.setattr(translate, "free", lambda model: None)
    monkeypatch.setattr(translate, "revise_run",
                        lambda *a, **kw: kw.get("lines") or a[2])

    def generate(processor, model, text, *, source, target, **kw):
        calls.append((source, target))
        # ar→he routes through English, as every non-English pair does.
        return "hello from the Arabic" if target == "en" else "שלום מהערבית"

    monkeypatch.setattr(translate, "generate", generate)
    m = _manifest([_seg(1, "שלום עולם"),
                   _seg(2, "مرحبا بالعالم", lang="ar")])
    translate.run(m, tmp_path, source="he", target="he")
    assert m["segments"][0]["text_en"] == "שלום עולם"       # identity, no hop
    assert m["segments"][1]["text_en"] == "שלום מהערבית"     # a real ar→he hop
    assert calls == [("ar", "en"), ("en", "he")]


def test_a_same_language_run_asks_the_timeline_for_no_rewrites(tmp_path):
    # Shortening a same-language line would mean loading Gemma to rewrite the
    # speaker's own words in their own language. The timeline absorbs the overhang
    # with speed-up and drift instead (`timeline.run` skips the round on None).
    args = _args(src="he", tgt="he", device=None, register="narration",
                 genre="documentary")
    assert cli._retimers({"segments": [], "source": {}}, tmp_path, None, args) == (None,
                                                                                   None)
    args = _args(src="he", tgt="en", device=None, register="narration",
                 genre="documentary")
    shorten, resynth = cli._retimers({"segments": [], "source": {}}, tmp_path, None, args)
    assert shorten is not None and resynth is not None


# ----------------------------------------------------------------- stage tags


def test_the_stage_tags_moved():
    # Hebrew synthesis changes what the tts stage produces, and identity
    # translation changes what translate produces; neither may replay old output.
    # v16, not v15: a second line of work claimed v15 for the honest-failure
    # records, so the merged tag moves past both claims rather than picking one.
    # v17: a dub-wanted line with no translation now replaces a stale *synthesis*
    # with its original-audio slice, which changes what this stage produces.
    # v18: reference-varying retries, the two-bar accept, and greedy-by-default
    # for Hebrew every one of which can change the clip a segment ends up with.
    # v19: the retry ladder is gated on voice identity, so a segment that used to
    # escalate to another speaker's window now stays on its own audio.
    # v20: a Hebrew line is decoded behind a warm-up carrier that is cut back off,
    # so every Hebrew clip is different audio from the one v19 would have made.
    # v23: and it is now actually cut off. Under v21's threshold the cut was refused
    # on every clip measured, so a v21/v22 Hebrew entry holds the cold decode rather
    # than the warmed one; different audio again, and this time the audio the feature
    # was written to produce in the first place.
    # v24: the verify ASR's own word boundaries no longer fail a take, so a line
    # that used to keep its original audio now keeps a clip.
    # v25: a one- or two-word clip has an absolute length ceiling again, so a
    # stalled short take is retried instead of placed.
    # v26: a line with under a second and a half of audio clones from its
    # speaker's canonical reference, so its clip is cloned from other seconds.
    assert manifest.STAGE_TAGS["tts"] == "tts/v26"
    # v36: script-derived gloss floors, negations and shorten budgets (CJK/hangul).
    # v37: "%" survives into every TTS target's own vocabulary.
    # v39: a repetition the speaker actually made survives the "X, X" repair.
    assert manifest.STAGE_TAGS["translate"] == "translate/v39"


def test_ipa_is_not_a_manifest_field():
    """The IPA is a synthesis input; the record is always the orthography."""
    assert "ipa" not in manifest.SEGMENT_KEYS
    assert "text_ipa" not in manifest.SEGMENT_KEYS


# ------------------------------------------------------- the decode Hebrew defaults to


def test_a_hebrew_target_decodes_greedily_unless_the_segment_says_otherwise(
        tmp_path, fake_g2p, monkeypatch):
    """Only honest because the retry ladder varies the reference, not the seed.

    The sampler is where the LoRA's wandering and repetition come from, so greedy
    is the steadier read of IPA. It could not be the default while a retry meant
    "same reference, next seed": under `do_sample=False` the seed does nothing,
    so greedy-by-default would have bought one take and no retries at all.
    """
    eng = _engine(tmp_path)
    monkeypatch.setattr(eng, "ref_for",
                        lambda seg, opts=DEFAULT: (tmp_path / "r.wav", "ref:1.00-4.00"))
    base = {"id": 1, "uid": "u1", "start": 0.0, "end": 2.0, "speaker": "S1"}
    assert eng._plan(base, "שלום עולם").greedy is True
    # …and the ladder still ends on a take that is not the first one again: with
    # no speaker reference bank here, the sampled rescue is the whole ladder.
    plan = eng._plan(base, "שלום עולם")
    assert [g for _p, _k, g in eng.rungs(base, plan)] == [True, False]

    # The segment always wins, whether it asks for the sampler or says so outright.
    sampled = {**base, "tts_opts": {"greedy": False}}
    assert eng._plan(sampled, "שלום עולם").greedy is False
    warm = {**base, "tts_opts": {"temperature": 0.8}}
    assert eng._plan(warm, "שלום עולם").greedy is False

    # Nothing changes for a target the base checkpoint speaks itself.
    en = tts.Engine({"source": {"src_lang": "he", "tgt_lang": "en"},
                     "speakers": {}, "segments": []}, tmp_path)
    monkeypatch.setattr(en, "ref_for",
                        lambda seg, opts=DEFAULT: (tmp_path / "r.wav", "ref:1.00-4.00"))
    assert en._plan(base, "hello world").greedy is False


# ------------------------------------------------------------- warm-up carrier

# The talker decodes from the SOURCE speaker's x-vector, so a Hebrew line that
# starts cold carries an English accent for its first seconds and only then
# settles into the adapter's Hebrew. The fix is to decode a fixed Hebrew phrase
# first and cut it back off, and what these tests hold is the part that has to be
# exact: the sentence the user hears must not contain a syllable of it, and a
# boundary that cannot be trusted must fall back to today's audio, not to a guess.


def test_the_carrier_is_phonemized_like_any_other_hebrew(fake_g2p):
    hebrew.free()
    assert hebrew.carrier() == f"<ipa:{hebrew.CARRIER_TEXT}>"
    # Once per process, not once per line: it is the same phrase every time.
    hebrew.carrier()
    assert fake_g2p.count(hebrew.CARRIER_TEXT) == 1
    hebrew.free()


# Real word streams, whisper-large-v3-turbo over clips this repo's own Hebrew path
# generated for outputs/demo_en_he. CLEAN is the usual reading, six tokens. SPLIT_8
# and SPLIT_7 are segments 25 and 27 of the same run, where the ASR broke the same
# fixed phrase into eight and seven tokens: those two are the bug this file exists
# to hold shut, because counting six words in either of them points the cut at the
# carrier's own last words and still lands inside any plausible duration band.

CLEAN = [("רגע", 0.0), ("אחד", 0.58), ("בבקשה,", 1.06), ("ועכשיו", 1.76),
         ("נמשיך", 2.38), ("הלאה.", 2.80), ("אני", 3.24), ("תמיד", 3.46),
         ("אומרת", 3.80), ("לאנשים,", 4.20)]
CLEAN_TEXT = "אני תמיד אומרת לאנשים, תעשו את מה שמפחיד אתכם."

# אחד came back as "סאגה הד", an "או" appeared out of nowhere, and הלאה became הלאק.
SPLIT_8 = [("רגע", 0.0), ("סאגה", 0.48), ("הד", 1.0), ("בבקשה,", 1.16),
           ("ועכשיו", 1.86), ("או", 2.42), ("נמשיך", 2.56), ("הלאק", 2.96),
           ("שמבינים", 3.30), ("שאם", 3.92), ("אנשים", 4.32), ("מאמינים", 4.78)]
SPLIT_8_TEXT = "כשמבינים שאם אנשים מאמינים בך ואתה מאמין שהם מאמינים בך."

# הלאה came back as two tokens, "על אשצה".
SPLIT_7 = [("רוג", 0.0), ("אחד", 0.38), ("בבקשה", 0.78), ("ועכשיו", 1.54),
           ("נמשיך", 2.08), ("על", 2.50), ("אשצה,", 2.62), ("זהו", 3.20),
           ("לוח", 3.64), ("עדות.", 3.92)]
SPLIT_7_TEXT = "זהו לוח עדות."


def test_the_boundary_is_where_the_carrier_stops_being_heard():
    assert tts.carrier_boundary(CLEAN) == 3.24


def test_a_mis_split_carrier_does_not_move_the_boundary():
    """The observed failure, both shapes of it.

    Counting words put segment 25's cut at 2.56 and segment 27's at 2.62, both
    comfortably inside [2.5, 5.0] and both in the middle of the carrier: the clips
    would have shipped opening on "נמשיך הלאק" and on "אשצה". Matching characters
    cannot be moved by a split or an invented token, because the letters either
    side of a boundary the ASR imagined are the same letters.
    """
    assert tts.carrier_boundary(SPLIT_8) == 3.30
    assert tts.carrier_boundary(SPLIT_7) == 3.20


def test_a_boundary_that_cannot_be_established_is_refused():
    """None is a refusal, not a cut: the caller re-makes the clip without one."""
    # A clip that does not begin with this carrier at all: nothing to align to, so
    # there is no cut to make and the take is thrown away.
    sentence = [("זהו", 0.0), ("לוח", 0.62), ("עדות", 1.20), ("שלם", 1.90),
                ("ומדויק", 2.60), ("מאוד", 3.20), ("היום", 3.80)]
    assert tts.carrier_boundary(sentence) is None
    # The carrier was heard but the ASR found no sentence behind it, so there is no
    # word after the match to cut to.
    assert tts.carrier_boundary(CLEAN[:6]) is None
    assert tts.carrier_boundary([]) is None
    # And the band still bounds the answer, even when the letters line up: this is
    # not what makes the cut trustworthy, but it is still an answer nothing sane
    # can produce.
    late = [(w, t * 3.0) for w, t in CLEAN]
    assert tts.carrier_boundary(late) is None


# Real word streams again, but from ten clips generated against the carrier code as
# it shipped, and these are the ones that took the feature down. Whisper does not
# transcribe the warm-up, it hallucinates over it: the phrase is rushed and content
# free, so what comes back is Hebrew-shaped gibberish. The best prefix distance on
# these ran 14 to 18 against a 26-letter carrier and the v21 gate refused every one
# of them at 10.4, which meant every Hebrew line in the run fell back to the cold
# decode the carrier exists to prevent.

# Segment 25's own clip, the line the defect was reported on. Note the carrier came
# back as "וגטא דבאפ הוא סחיון אנישייב נייל," and the line's own first word
# "כשמבינים" came back as two words, "שהם מבינים".
GIBBERISH = [
    ("וגטא", 0.0), ("דבאפ", 0.62), ("הוא", 1.04),
    ("סחיון", 1.38), ("אנישייב", 1.82), ("נייל,", 2.36),
    ("שהם", 2.76), ("מבינים", 3.02), ("שאם", 3.42),
]
GIBBERISH_TEXT = "כשמבינים שאם אנשים מאמינים בך ואתה מאמין שהם מאמינים בך."
# The same line under a sampled decode. Here the alignment's favourite prefix is the
# correct one, which is the easy case.
GIBBERISH_OK = [
    ("וגא", 0.0), ("עדת", 0.38), ("בפאף", 0.68),
    ("וסכי", 1.2), ("אונצ", 1.84), ("'וית", 2.1),
    ("הלאה.", 2.3), ("כשמבינים", 2.6), ("שאם", 3.16),
]
# The control: the same text decoded with no carrier in front of it at all.
NO_CARRIER = [
    ("כשמבינים", 0.0), ("שאם", 0.78), ("אנשים", 1.12),
    ("מאמינים", 1.72), ("בך", 2.46), ("ואתה", 2.82),
    ("מאמין", 3.36), ("שהם", 3.76), ("מאמינים", 4.04),
]


def test_a_hallucinated_carrier_transcript_is_still_cut():
    """The regression this file exists to hold shut from now on.

    Every one of these scores far worse against CARRIER_TEXT than any threshold
    could have tolerated, and every one of them has a carrier on the front that has
    to come off. Distance is not what decides that any more, so they all produce
    candidates rather than a refusal.
    """
    assert tts.carrier_candidates(GIBBERISH)
    assert tts.carrier_candidates(GIBBERISH_OK)
    # And the distances really are that bad, i.e. this is not a test that would pass
    # by accident if the old threshold came back.
    letters = tts._letters(hebrew.CARRIER_TEXT)
    for stream in (GIBBERISH, GIBBERISH_OK):
        floor = min(tts._prefix_distances([tts._letters(w) for w, _ in stream], letters))
        assert floor > hebrew.CARRIER_HEAD_MAX * len(letters)
        assert floor >= 14


def test_the_carrier_distance_cannot_rank_boundaries_by_correctness():
    """Why the threshold was not raised instead: there is nothing to raise it to.

    A distance ceiling can only work if correct boundaries score better than wrong
    ones. They do not. SPLIT_7's mid-carrier cut, the one fe713ff was written to
    stop, scores better than the FLOOR of every real clip above, so any ceiling that
    admits a good clip admits that bad cut several times over.
    """
    letters = tts._letters(hebrew.CARRIER_TEXT)
    bad = tts._prefix_distances([tts._letters(w) for w, _ in SPLIT_7], letters)[5]
    good = min(tts._prefix_distances([tts._letters(w) for w, _ in GIBBERISH], letters))
    assert bad < good
    # Which is exactly why nothing downstream may reintroduce one.
    assert not hasattr(hebrew, "CARRIER_MATCH_MAX")


def test_a_clip_with_no_carrier_on_it_is_refused():
    """Presence is decided by the shape of the curve, not by the size of the score.

    NO_CARRIER's best score (21) is not far off GIBBERISH's (17), so magnitude
    cannot separate them. The descent can: a carrier makes the score fall to a floor
    and then climb, and with no carrier it only ever climbs.
    """
    assert tts.carrier_candidates(NO_CARRIER) == []
    letters = tts._letters(hebrew.CARRIER_TEXT)
    curve = tts._prefix_distances([tts._letters(w) for w, _ in NO_CARRIER], letters)
    assert curve[0] - min(curve) < hebrew.CARRIER_DESCENT_MIN
    for stream in (GIBBERISH, GIBBERISH_OK, CLEAN):
        curve = tts._prefix_distances([tts._letters(w) for w, _ in stream], letters)
        assert curve[0] - min(curve) >= hebrew.CARRIER_DESCENT_MIN


def test_the_true_boundary_is_offered_even_when_it_scores_second():
    """Why candidates are a list and not an answer.

    On this clip the best-scoring prefix stops one word short, at 2.36, which would
    leave the carrier's last word "נייל," on the front of the shipped line. The
    correct boundary, 2.76, scores one worse. Both are offered, best first, and the
    far-side check is what picks between them.
    """
    found = tts.carrier_candidates(GIBBERISH)
    assert found[0] == 2.36
    assert 2.76 in found
    # The alignment's favourite is still what `carrier_boundary` reports.
    assert tts.carrier_boundary(GIBBERISH) == 2.36


def test_the_cut_is_proven_from_the_far_side_too():
    """`carrier_gone` hears the result instead of trusting the arithmetic."""
    # A correct cut, mis-heard the way this ASR really does mis-hear these lines.
    assert tts.carrier_gone("שמבינים שאם אנשים מאמינים", SPLIT_8_TEXT) is True
    assert tts.carrier_gone("שמי קליסטינה קוק, ואני", "שמי כריסטינה קוק, ואני מומחית")
    # The two bad cuts, i.e. what would have shipped. The first still has a whole
    # carrier word on it; the second only has the tail of one, which is why the
    # anchor and not the word list is what has to catch it.
    assert tts.carrier_gone("נמשיך הלאק שמבינים שאם", SPLIT_8_TEXT) is False
    assert tts.carrier_gone("אשצה זהו לוח עדות.", SPLIT_7_TEXT) is False
    # A cut that ate the start of the sentence instead of the end of the carrier.
    assert tts.carrier_gone("לוח עדות.", SPLIT_7_TEXT) is False
    # And a clip nothing was heard in proves nothing either way, so it is refused.
    assert tts.carrier_gone("", SPLIT_7_TEXT) is False


def test_a_re_split_first_word_does_not_reject_a_correct_cut():
    """The other half of the 10-of-10 failure, and the reason the head is letters.

    This is what segment 25's correctly cut clip actually transcribes as: the line
    opens "כשמבינים שאם" and whisper heard the first word as two, "שהם מבינים". Held
    to two words against two words the heard head is nine letters against eleven and
    the take was thrown away; matched as a prefix, a re-split costs nothing, which is
    the same argument that moved the carrier itself off token counting.
    """
    assert tts.carrier_gone("שהם מבינים שאם אנשים מאמינים", GIBBERISH_TEXT) is True


def test_a_cut_that_opens_inside_the_first_word_is_refused():
    """Matching a prefix is generous in one direction, so the length is pinned too.

    Dropping the line's opening syllable is only a couple of insertions and would
    pass on drift alone, so the winning prefix also has to be about as long as the
    head it matched. Leaving carrier on the front stays expensive either way, since
    every stray letter has to be deleted before the line can match at all.
    """
    assert tts.carrier_gone("מבינים שאם אנשים", GIBBERISH_TEXT) is False
    assert tts.carrier_gone("נייל שהם מבינים שאם", GIBBERISH_TEXT) is False


class _FakeAsr:
    """Word timestamps out of a script, keyed by which file is being read.

    `_cut_carrier` transcribes twice: the generated clip, to find the boundary,
    and the file it cut, to prove what came out. This hands each of them its own
    answer so both halves of the decision can be driven from a test.
    """

    def __init__(self, whole, after):
        self.whole, self.after = whole, after
        self.seen: list[str] = []

    def transcribe(self, path, **kw):
        self.seen.append(path)
        words = self.after if path.endswith(".cut.wav") else self.whole
        seg = type("S", (), {"words": [type("W", (), {"word": w, "start": t})()
                                       for w, t in words]})()
        return [seg], None


def _cut_engine(tmp_path, monkeypatch, asr):
    """An engine whose `_cut_carrier` can run without ffmpeg or a model."""
    eng = _engine(tmp_path, tgt="he")
    monkeypatch.setattr(tts.Engine, "asr_for", lambda self, tgt=None: asr)
    # ffmpeg's only job here is to write the cut file; the bytes stand in for it.
    monkeypatch.setattr(tts.audio, "run",
                        lambda cmd, **kw: __import__("pathlib").Path(cmd[-1])
                        .write_bytes(b"cut"))
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"whole")
    return eng, clip


def test_a_mis_split_carrier_is_cut_where_the_sentence_starts(tmp_path, monkeypatch):
    asr = _FakeAsr(SPLIT_8, [("שמבינים", 0.0), ("שאם", 0.62), ("אנשים", 1.02)])
    eng, clip = _cut_engine(tmp_path, monkeypatch, asr)
    assert eng._cut_carrier(clip, "he", SPLIT_8_TEXT) is True
    assert clip.read_bytes() == b"cut"          # the cut was moved onto the clip
    assert not clip.with_suffix(".cut.wav").exists()


def test_an_unalignable_take_is_discarded_rather_than_cut(tmp_path, monkeypatch):
    """No carrier could be found, so there is no cut to make and no clip to keep.

    The clip has to come back untouched, because the caller's answer to a False is
    to re-synthesize it cold and it must not be re-synthesizing over a half-cut file.
    """
    nothing = [("זהו", 0.0), ("לוח", 0.62), ("עדות", 1.20), ("שלם", 1.90),
               ("ומדויק", 2.60), ("מאוד", 3.20), ("היום", 3.80)]
    asr = _FakeAsr(nothing, nothing)
    eng, clip = _cut_engine(tmp_path, monkeypatch, asr)
    assert eng._cut_carrier(clip, "he", SPLIT_7_TEXT) is False
    assert clip.read_bytes() == b"whole"
    assert len(asr.seen) == 1                   # it never got as far as cutting


def test_a_cut_that_does_not_open_on_the_line_is_discarded(tmp_path, monkeypatch):
    """The far-side check, driven on its own.

    The alignment is happy here it is the audio that came back wrong, which is the
    case a boundary named on the right word but a beat early produces. The take is
    thrown away exactly as if no boundary had been found at all.
    """
    asr = _FakeAsr(CLEAN, [("נמשיך", 0.0), ("הלאה", 0.42), ("אני", 0.80)])
    eng, clip = _cut_engine(tmp_path, monkeypatch, asr)
    assert eng._cut_carrier(clip, "he", CLEAN_TEXT) is False
    assert clip.read_bytes() == b"whole"        # left exactly as it was
    assert not clip.with_suffix(".cut.wav").exists()
    # ... and it did do the work: it cut, listened, and refused what it heard.
    assert len(asr.seen) == 2


class _StepAsr(_FakeAsr):
    """Like `_FakeAsr`, but each cut it is asked about gets its own answer.

    `_cut_carrier` now cuts once per candidate until one is proven, so the far-side
    transcript is not a single fixed reply any more.
    """

    def __init__(self, whole, afters):
        super().__init__(whole, None)
        self.afters = list(afters)

    def transcribe(self, path, **kw):
        self.seen.append(path)
        words = self.afters.pop(0) if path.endswith(".cut.wav") else self.whole
        seg = type("S", (), {"words": [type("W", (), {"word": w, "start": t})()
                                       for w, t in words]})()
        return [seg], None


def test_a_boundary_that_leaves_carrier_behind_falls_to_the_next_candidate(
        tmp_path, monkeypatch):
    """The 10-of-10 clip, driven all the way through the shipped path.

    The alignment's favourite cut on this stream is 2.36, which leaves "נייל," on
    the front. That take is cut, listened to and refused, and the next candidate
    (2.76, the correct one) is cut, listened to and kept. The clip only ever ships
    because something heard it open on the line, which is the property that has to
    survive any retuning here.
    """
    asr = _StepAsr(GIBBERISH, [
        [("נייל", 0.0), ("שהם", 0.4), ("מבינים", 0.66), ("שאם", 1.06)],
        [("שהם", 0.0), ("מבינים", 0.26), ("שאם", 0.66), ("אנשים", 1.0)],
    ])
    eng, clip = _cut_engine(tmp_path, monkeypatch, asr)
    assert eng._cut_carrier(clip, "he", GIBBERISH_TEXT) is True
    assert clip.read_bytes() == b"cut"
    assert not clip.with_suffix(".cut.wav").exists()
    # One read of the whole clip, then one per candidate it had to listen to.
    assert len(asr.seen) == 3


def test_no_candidate_proving_clean_is_still_a_refusal(tmp_path, monkeypatch):
    """Trying more places must not turn into eventually accepting one of them."""
    asr = _StepAsr(GIBBERISH, [
        [("נייל", 0.0), ("שהם", 0.4), ("מבינים", 0.66)],
        [("נמשיך", 0.0), ("הלאה", 0.4), ("שהם", 0.8)],
        [("מבינים", 0.0), ("שאם", 0.4), ("אנשים", 0.8)],
    ])
    eng, clip = _cut_engine(tmp_path, monkeypatch, asr)
    assert eng._cut_carrier(clip, "he", GIBBERISH_TEXT) is False
    assert clip.read_bytes() == b"whole"        # the take is discarded, not guessed at
    assert not clip.with_suffix(".cut.wav").exists()


def test_the_carrier_widens_the_token_budget_it_shares():
    """It is decoded out of the same budget, so the sentence must not pay for it."""
    plain = tts.max_new_tokens("שלום עולם ומה שלומך היום", "he")
    assert tts.carrier_budget(plain) > plain
    # The ceiling is still `max_new_tokens`', not something the carrier may raise.
    assert tts.carrier_budget(2048) == 2048


def test_only_hebrew_is_decoded_behind_a_carrier(tmp_path, monkeypatch, fake_g2p):
    hebrew.free()
    monkeypatch.setattr(tts.Engine, "asr_for", lambda self, tgt=None: object())
    assert _engine(tmp_path, tgt="he")._carrier_for("he") is not None
    # The ten languages the checkpoint speaks natively have no warm-up to do, and
    # that includes a base-language line voiced inside a Hebrew run.
    assert _engine(tmp_path, tgt="he")._carrier_for("en") is None
    assert _engine(tmp_path, tgt="ru")._carrier_for("ru") is None
    hebrew.free()


def test_no_verification_asr_means_no_carrier(tmp_path, monkeypatch):
    """The cut is made on that ASR's word timestamps.

    Without one there is no way to show the shipped clip is free of the carrier,
    and a carrier that cannot be proven gone is one that must never be prepended.
    """
    monkeypatch.setattr(tts.Engine, "asr_for", lambda self, tgt=None: None)
    assert _engine(tmp_path, tgt="he")._carrier_for("he") is None


def test_the_cache_key_carries_the_carrier_only_where_one_was_used(tmp_path):
    eng = _engine(tmp_path)
    args = ("שלום עולם.", "ref:1.00-4.00", 42, False, DEFAULT, None, "ʃalˈom ʔolˈam.")
    # Not audible in the clip, but it changes the context the sentence decodes in
    # and so the audio: a cached clip of the cold read may not be replayed as this.
    assert eng._cache_key(*args, carrier=True) != eng._cache_key(*args, carrier=False)
    # And a Hebrew clip made without one keeps the key it had before this existed.
    assert eng._cache_key(*args, carrier=False) == eng._cache_key(*args)
