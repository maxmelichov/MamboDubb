"""Phase 2 Hebrew → any Qwen3-TTS target. Pure logic, no models."""

from dubbing import mix, tts

# --------------------------------------------------------------------- prepare_text


def test_prepare_text_ru_keeps_cyrillic_drops_hebrew():
    out = tts.prepare_text("Привет שלום мир", "ru")
    assert "Привет" in out and "мир" in out
    assert "שלום" not in out


def test_prepare_text_ru_alone_survives():
    assert tts.prepare_text("Привет, мир", "ru") == "Привет, мир."


def test_prepare_text_en_matches_prepare_en():
    for text in ("Hello [music] world", "Qatar funds it, and more",
                 "mixed שלום text", "שלום", "It's a co-op deal."):
        assert tts.prepare_text(text, "en") == tts.prepare_en(text)


def test_prepare_en_still_drops_hebrew_entirely():
    assert tts.prepare_en("שלום") == ""
    assert tts.prepare_text("שלום", "he") == "שלום."


def test_prepare_text_cjk_survives():
    zh = tts.prepare_text("你好，世界。", "zh")
    assert "你好" in zh and "世界" in zh
    ja = tts.prepare_text("こんにちは世界", "ja")
    assert "こんにちは" in ja and "世界" in ja


def test_prepare_text_ko_survives():
    out = tts.prepare_text("안녕하세요 세계", "ko")
    assert "안녕하세요" in out and "세계" in out


# ------------------------------------------------------------- tokens and overlap


def test_tokens_english_unchanged():
    assert tts._tokens("Hello, world's 3 dogs!") == ["hello", "world's", "3", "dogs"]
    assert tts._tokens("Hello, world's 3 dogs!", "en") == ["hello", "world's", "3", "dogs"]


def test_tokens_cyrillic():
    toks = tts._tokens("Привет, мир", "ru")
    assert toks == ["привет", "мир"]


def test_tokens_cjk_single_characters():
    assert tts._tokens("你好世界", "zh") == ["你", "好", "世", "界"]
    assert tts._tokens("안녕 세계", "ko") == ["안", "녕", "세", "계"]


def test_word_overlap_cyrillic_identity():
    s = "Катар финансирует это и многое другое"
    assert tts.word_overlap(s, s, "ru") == 1.0
    assert tts.word_overlap(s, "", "ru") == 0.0
    assert tts.word_overlap("привет мир", "привет", "ru") == 0.5


def test_word_overlap_cjk_char_level():
    assert tts.word_overlap("你好世界", "你好世界", "zh") == 1.0
    assert tts.word_overlap("你好世界", "你好", "zh") == 0.5


def test_word_overlap_english_unchanged():
    assert tts.word_overlap("hello world", "hello world") == 1.0
    assert tts.word_overlap("hello world", "hello") == 0.5


def test_word_overlap_forgives_the_asr_its_own_word_boundaries():
    """The verify ASR wrote "All right." for a clip that says "Alright.".

    A one-word line has no other words to carry the score, so that spelling choice
    took the take from 1.0 to 0.0, failed every rung of the ladder, and aired the
    source audio instead. Both directions, because which side splits the word is
    the ASR's choice, not the translator's.
    """
    assert tts.word_overlap("Alright.", "All right.") == 1.0
    assert tts.word_overlap("All right.", "Alright.") == 1.0
    assert tts.word_overlap("Welcome home", "Wellcome home") == 1.0


def test_word_overlap_still_fails_a_take_that_says_something_else():
    """The letter bar is a spelling repair, not a second chance for a bad clone."""
    assert tts.word_overlap("Alright.", "Chelsea.") == 0.0
    assert tts.word_overlap("Nice, well done.", "Something else entirely here.") == 0.0
    assert tts.word_overlap("Just fine.", "") == 0.0
    # Two real words that merely look alike stay two different words.
    assert tts.word_overlap("I saw you at the prayer.",
                            "I saw you at the player.") < 1.0


def test_word_overlap_credits_each_target_word_once():
    """A repeated word in the heard text cannot pay for a word that was not said."""
    assert abs(tts.word_overlap("okay okay okay", "okay") - 1 / 3) < 1e-9
    assert tts.word_overlap("okay", "okay okay okay") == 1.0


# ----------------------------------------------------------------- length guards


def test_clone_length_ok_chinese_counts_characters():
    text = "这是一个很长的句子里面有十二个字"   # 16 chars ≈ 3.2 s at 5 chars/s
    assert tts.clone_length_ok(3.2, text, "zh")
    assert not tts.clone_length_ok(0.3, text, "zh")   # chipmunk
    assert not tts.clone_length_ok(30.0, text, "zh")  # stalled
    assert not tts.clone_length_ok(0.0, text, "zh")


def test_clone_length_ok_english_unchanged():
    text = "this line has six words total"
    assert tts.clone_length_ok(2.0, text)
    assert not tts.clone_length_ok(0.5, text)
    assert not tts.clone_length_ok(9.0, text)


def test_a_short_line_still_has_a_ceiling():
    """Below three units the per-unit rate is off, but not the absolute bound.

    A one-word "Chelsea." came back 4.61s. The verify ASR heard "Chelsea", which
    is all word_overlap asks, and the length guard did not look at short lines at
    all, so 4.6s of audio went into a 1.1s slot and pushed the next line 2.13s
    late. The short-line allowance is generous and finite.
    """
    assert tts.clone_length_ok(0.69, "Chelsea.")      # what a right one measures
    assert tts.clone_length_ok(2.10, "Chelsea.")      # a slow, emphatic delivery
    assert not tts.clone_length_ok(4.61, "Chelsea.")  # the stall that started this
    assert tts.clone_length_ok(3.00, "My dear.")      # two words get twice the rate
    assert not tts.clone_length_ok(3.30, "My dear.")
    # A chipmunk short line is refused exactly as before.
    assert not tts.clone_length_ok(0.06, "Chelsea.")


def test_max_new_tokens_chinese_not_word_counted():
    zh = "这是一个很长的句子" * 5              # 45 chars, one "word" by .split()
    # Word-based budget for 1 "word" is the 96 floor; char-based must exceed it.
    assert tts.max_new_tokens(zh) == 96
    assert tts.max_new_tokens(zh, "zh") > 96
    en = "hello world"
    assert tts.max_new_tokens(en) == tts.max_new_tokens(en, "en")


# ---------------------------------------------------------- Qwen language mapping


def test_qwen_language_name_mapping():
    supported = ["English", "Chinese", "German", "Italian", "Portuguese",
                 "Spanish", "Japanese", "Korean", "French", "Russian"]
    assert tts.qwen_language_name("ru", supported) == "Russian"
    assert tts.qwen_language_name("zh", supported) == "Chinese"
    assert tts.qwen_language_name("EN", supported) == "English"
    # case-insensitive against the checkpoint's spelling
    assert tts.qwen_language_name("fr", ["french"]) == "french"


def test_qwen_language_name_unsupported_falls_back_to_auto():
    assert tts.qwen_language_name("he", ["English", "Russian"]) == "Auto"
    assert tts.qwen_language_name("ru", ["English"]) == "Auto"
    assert tts.qwen_language_name("xx", ["English"]) == "Auto"
    assert tts.qwen_language_name("", []) == "Auto"


# ------------------------------------------------------------ source-script leak


def test_source_script_leak_hebrew_in_russian_clip():
    assert tts.source_script_leak("שלום שלום שלום", "he", "ru")
    assert tts.source_script_leak("שלום мир", "he", "ru")          # src >= tgt letters
    assert not tts.source_script_leak("Катар финансирует это", "he", "ru")


def test_source_script_leak_matches_old_en_behaviour():
    assert tts.source_script_leak("שלום עולם", "he", "en")
    assert not tts.source_script_leak("hello world", "he", "en")
    # a stray Hebrew char against solid English does not fail the clip
    assert not tts.source_script_leak("hello ש big wide world", "he", "en")


def test_source_script_leak_skipped_for_same_script_pair():
    assert not tts.source_script_leak("hola mundo", "en", "es")
    assert not tts.source_script_leak("anything at all", "en", "en")
    assert not tts.source_script_leak("привет", "ru", "uk")


# ------------------------------------------------------------------ ASR selection


def test_asr_candidates_en_keeps_dot_en_chain():
    assert tts._asr_candidates("en") == tts._ASR_CANDIDATES
    assert all(c.endswith(".en") for c in tts._asr_candidates("en"))


def test_asr_candidates_other_targets_use_multilingual():
    for tgt in ("ru", "zh", "de", "ja"):
        cands = tts._asr_candidates(tgt)
        assert cands == tts._ASR_CANDIDATES_MULTI
        assert not any(c.endswith(".en") for c in cands)


# ------------------------------------------------------------------- cache keys


def test_cache_key_mixes_target_language(tmp_path):
    e_en = tts.Engine({"source": {"src_lang": "he", "tgt_lang": "en"}}, tmp_path)
    e_ru = tts.Engine({"source": {"src_lang": "he", "tgt_lang": "ru"}}, tmp_path)
    args = ("same text", "ref:1.00-4.00", 42, False)
    assert e_en._cache_key(*args) != e_ru._cache_key(*args)


# -------------------------------------------------------------------- mix / SRT


def test_subtitle_lang3():
    assert mix.subtitle_lang3("en") == "eng"
    assert mix.subtitle_lang3("ru") == "rus"
    assert mix.subtitle_lang3("zh") == "chi"
    assert mix.subtitle_lang3("he") == "heb"
    assert mix.subtitle_lang3("xx") == "und"
    assert mix.subtitle_lang3("") == "und"


# ------------------------------------------------------- tts_failed is per-run


def test_clear_failed_keeps_makes_tts_failed_dubbable_again():
    segs = [
        {"id": 0, "keep": True, "keep_reason": "tts_failed", "text_en": "hi",
         "tts": {"clip": "clips/keep_x.wav", "tries": 0, "verify": "keep"}},
        {"id": 1, "keep": True, "keep_reason": "foreign", "text_en": "ok"},
        {"id": 2, "keep": True, "keep_reason": "latin"},
        {"id": 3, "keep": True, "keep_reason": "speaker_en"},
        {"id": 4, "keep": False, "keep_reason": None, "text_en": "x"},
    ]
    assert tts.clear_failed_keeps(segs) == [0]
    assert segs[0]["keep"] is False and segs[0]["keep_reason"] is None
    assert "tts" not in segs[0]                       # keep-clip record dropped too
    # run()'s own selection rule now includes it
    assert [s["id"] for s in segs if not s["keep"]] == [0, 4]
    # every other keep reason is untouched
    for i, reason in ((1, "foreign"), (2, "latin"), (3, "speaker_en")):
        assert segs[i]["keep"] is True and segs[i]["keep_reason"] == reason


def test_reset_stage_tts_clears_tts_failed():
    from dubbing import manifest

    m = manifest.new({"input": "x"})
    m["segments"] = [
        {"id": 0, "keep": True, "keep_reason": "tts_failed", "text_en": "hi",
         "tts": {"clip": "a"}},
        {"id": 1, "keep": True, "keep_reason": "foreign", "tts": {"clip": "b"}},
    ]
    manifest.reset_stage(m, "tts")
    assert m["segments"][0]["keep"] is False
    assert m["segments"][0]["keep_reason"] is None
    assert "tts" not in m["segments"][0] and "tts" not in m["segments"][1]
    assert m["segments"][1]["keep_reason"] == "foreign"   # not tts's flip to undo


# ------------------------------------------------------- escalation reference


def test_canonical_ref_lookup(tmp_path):
    m = {"source": {"src_lang": "he", "tgt_lang": "en"},
         "speakers": {"S1": {"ref": "refs/S1.wav"}}, "segments": []}
    eng = tts.Engine(m, tmp_path)
    (tmp_path / "refs" / "S1.wav").write_bytes(b"x")
    got = eng._canonical_ref({"speaker": "S1"})
    assert got is not None and got[0] == tmp_path / "refs/S1.wav"
    # The ref key carries a hash of the file's content, so a rebuilt canonical
    # reference can never replay clips cloned from the old file's voice.
    assert got[1].startswith("S1:canonical:") and len(got[1]) > len("S1:canonical:")
    assert eng._canonical_ref({"speaker": "S2"}) is None          # never built
    m["speakers"]["S3"] = {"ref": "refs/S3.wav"}                  # stale record
    assert eng._canonical_ref({"speaker": "S3"}) is None


def test_canonical_ref_key_tracks_content(tmp_path):
    m = {"source": {"src_lang": "he", "tgt_lang": "en"},
         "speakers": {"S1": {"ref": "refs/S1.wav"}}, "segments": []}
    (tmp_path / "refs").mkdir(parents=True)
    (tmp_path / "refs" / "S1.wav").write_bytes(b"old-voice")
    key_old = tts.Engine(m, tmp_path)._canonical_ref({"speaker": "S1"})[1]
    (tmp_path / "refs" / "S1.wav").write_bytes(b"new-longer-voice")
    key_new = tts.Engine(m, tmp_path)._canonical_ref({"speaker": "S1"})[1]
    assert key_old != key_new


def test_a_verifier_relaxes_the_fast_bound_but_not_the_slow_one():
    """A quick take is the ASR's question to answer, a stalled one is not.

    Every line here is a take a Hebrew drama's run threw away unheard and the
    large ASR read back word for word: the fast bound assumes about 1.7 syllables
    a word, and a short line made of one-syllable function words beats it while
    saying every word. The slow bound does not move — a stall costs the timeline
    real seconds, and no reading of the words makes those back.
    """
    assert not tts.clone_length_ok(0.52, "I have it.")
    assert tts.clone_length_ok(0.52, "I have it.", verified=True)
    assert not tts.clone_length_ok(1.01, "They said they're on their way.")
    assert tts.clone_length_ok(1.01, "They said they're on their way.", verified=True)
    # Still garble at a rate no read reaches, verifier or not.
    assert not tts.clone_length_ok(0.12, "I have it.", verified=True)
    # And the stall bound is the same number under both.
    assert not tts.clone_length_ok(9.0, "this line has six words total", verified=True)
    assert not tts.clone_length_ok(4.61, "Chelsea.", verified=True)


def test_an_empty_read_is_retried_without_the_vad(tmp_path):
    """The VAD calling a whole clip silence is not the clip failing to say the line."""
    clip = tmp_path / "c.wav"
    clip.write_bytes(b"")
    asked = []

    class Model:
        def transcribe(self, path, **kw):
            asked.append(kw["vad_filter"])
            if kw["vad_filter"]:
                return [], None
            return [type("S", (), {"text": " We'll learn it by hand."})()], None

    assert tts._read(Model(), clip, "en") == "We'll learn it by hand."
    assert asked == [True, False]


def test_a_clip_the_vad_read_is_not_read_twice(tmp_path):
    clip = tmp_path / "c.wav"
    clip.write_bytes(b"")
    asked = []

    class Model:
        def transcribe(self, path, **kw):
            asked.append(kw["vad_filter"])
            return [type("S", (), {"text": "hello"})()], None

    assert tts._read(Model(), clip, "en") == "hello"
    assert asked == [True]


def test_a_sentence_split_folds_an_abbreviation_forward():
    """`split_sentences` reads punctuation, and "Dr." is punctuation."""
    assert tts._sentence_parts("Dr. Smith went home. He slept.") == [
        "Dr. Smith went home.", "He slept."]
    assert tts._sentence_parts("One line only") == ["One line only"]
    # ...and a one-word sentence is not worth a breath of its own either.
    assert tts._sentence_parts(
        "Let's see, can you? Stronger, a bit more. Excellent.") == [
        "Let's see, can you?", "Stronger, a bit more. Excellent."]
    # A trailing runt joins the piece before it, not a pause of its own.
    assert tts._sentence_parts("She left. Ok.") == ["She left. Ok."]


def test_joining_sentences_puts_the_pause_back(tmp_path):
    import numpy as np
    import soundfile as sf

    sr = 16000
    for name in ("a.wav", "b.wav"):
        sf.write(str(tmp_path / name), np.ones(sr, dtype="float32") * 0.1, sr)
    out = tts._join_clips([tmp_path / "a.wav", tmp_path / "b.wav"],
                          tmp_path / "j.wav", pause=0.2)
    wav, got = sf.read(str(out), dtype="float32")
    assert got == sr
    assert abs(len(wav) / sr - 2.2) < 1e-6


def test_a_short_take_that_says_half_the_line_is_truncation_not_speed():
    """The fast bound yields to the verifier; it may not excuse a stopped decode.

    A 1.79s take of a thirteen-word line that says only its first sentence is the
    synthesis stopping early. Under the relaxed bound it reached the ranking and
    out-scored the complete take of the same line, which the base ASR had read
    less generously.
    """
    line = "What's going on with the evacuation? Schultz says he'll be here soon."
    # Says everything, at a quick clip: exactly what the relaxation is for.
    assert tts.clone_length_ok(1.79, line, verified=True)
    # ...so the decision cannot rest on length alone, and `clip_is_good` is the
    # other half: a short take below the good bar is dropped by `clip_for`.
    assert not tts.clip_is_good({"ok": True, "overlap": 0.43, "dur": 1.79})
    assert tts.clip_is_good({"ok": True, "overlap": 1.0, "dur": 1.79})
