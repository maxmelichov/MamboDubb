"""Phase 2 — Hebrew → any Qwen3-TTS target. Pure logic, no models."""

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
