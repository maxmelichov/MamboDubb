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
    assert manifest.STAGE_TAGS["tts"] == "tts/v20"
    # v36: script-derived gloss floors, negations and shorten budgets (CJK/hangul).
    # v37: "%" survives into every TTS target's own vocabulary.
    assert manifest.STAGE_TAGS["translate"] == "translate/v38"


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


def test_the_boundary_is_the_word_after_the_carrier():
    starts = [0.0, 0.66, 1.12, 1.94, 2.70, 3.20, 3.60, 3.78]
    assert tts.carrier_boundary(starts, hebrew.CARRIER_WORDS) == 3.60


def test_a_boundary_outside_the_trusted_band_is_refused():
    """None is a refusal, not a cut: the caller re-makes the clip without one."""
    n = hebrew.CARRIER_WORDS
    # Too early to be the end of this carrier, so the ASR ran the words together.
    assert tts.carrier_boundary([0.1 * i for i in range(n + 2)], n) is None
    # Too late, so it split one of them and the count no longer means what it did.
    assert tts.carrier_boundary([1.0 * i for i in range(n + 2)], n) is None
    # And a clip the ASR heard no sentence in at all has no boundary to give.
    assert tts.carrier_boundary([0.0, 0.66, 1.12], n) is None
    assert tts.carrier_boundary([], n) is None


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
