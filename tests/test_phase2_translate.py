"""Phase 2: the translation guards work for any target language, not just English.

Pure logic, no models — same ground rules as tests/test_dubbing.py.
"""

from __future__ import annotations

from dubbing import translate


# ------------------------------------------------------------------ is_target_text

def test_is_target_text_accepts_the_actual_target_script():
    assert translate.is_target_text("Это перевод на русский язык", target="ru")
    assert translate.is_target_text("مرحبا بالعالم", target="ar")
    assert translate.is_target_text("שלום לכולם", target="he")
    assert translate.is_target_text("你好世界你好", target="zh")
    assert translate.is_target_text("안녕하세요 여러분", target="ko")


def test_is_target_text_rejects_untranslated_output_per_target():
    # Still-English output is a failed translation when the target is Russian.
    assert not translate.is_target_text("Hello there", target="ru")
    assert not translate.is_target_text("שלום עולם", target="ru")
    assert not translate.is_target_text("Hello there", target="ar")
    assert not translate.is_target_text("", target="ru")
    # Mostly source-script with a stray target letter is still a failure.
    assert not translate.is_target_text("Привет hello мир и ещё слова", target="en")


def test_is_target_text_legacy_one_arg_form_means_english():
    # The he→en path calls it with no target; behavior is exactly the old one.
    assert translate.is_target_text("Hello there")
    assert not translate.is_target_text("שלום עולם")
    assert not translate.is_target_text("")
    assert not translate.is_target_text("مرحبا بالعالم")
    assert not translate.is_target_text("Привет мир")


# ------------------------------------------------------------------- negation guard

def test_has_negation_per_language():
    neg = translate._has_negation
    assert neg("он не пришёл", target="ru")
    assert neg("никогда больше", target="ru")
    assert not neg("он пришёл вчера", target="ru")
    assert neg("no lo hizo", target="es")
    assert neg("nunca jamás", target="es")
    assert not neg("lo hizo ayer", target="es")
    assert neg("das ist nicht wahr", target="de")
    assert neg("ohne Zweifel", target="de")
    assert not neg("das ist wahr", target="de")
    # English default unchanged, including the contraction form.
    assert neg("they did not deny it")
    assert neg("they didn't deny it")
    assert not neg("they denied it")


def test_shorten_negation_guard_catches_a_dropped_negation(monkeypatch):
    # A Russian rewrite that silently drops "не" must be refused.
    original = "Катар не отрицал финансирование группировки"

    def attempt(reply):
        monkeypatch.setattr(translate, "_run", lambda *a, **k: reply)
        return translate.shorten(None, None, "he", original, 4,
                                 source="he", target="ru")

    assert attempt("Катар отрицал финансирование") is None       # lost не
    assert attempt("Катар не отрицал этого")                     # kept it


# ---------------------------------------------------------------- adjacent repeats

def test_adjacent_repeat_works_on_cyrillic():
    rep = translate._adjacent_repeat
    assert rep("распространяя образование, образование и права человека",
               target="ru") == "образование"
    assert rep("образование и образование", target="ru") == "образование"
    # Distinct words and short function words are left alone.
    assert rep("образование, воспитание и права человека", target="ru") is None
    assert rep("он и она или они", target="ru") is None


def test_adjacent_repeat_is_disabled_for_cjk_targets():
    assert translate._adjacent_repeat("教育、教育と人権", target="ja") is None
    assert translate._adjacent_repeat("教育,教育和人权", target="zh") is None


def test_adjacent_repeat_english_behavior_unchanged():
    rep = translate._adjacent_repeat
    assert rep("disseminating education, education and human rights") == "education"
    assert rep("Qatar funds Hamas, Qatar funds Al-Qaeda, Qatar funds ISIS") is None
    assert rep("a very, very extreme ideological view") is None
    assert rep("the the report") is None


def test_strip_adjacent_repeat_uses_the_targets_coordinators():
    strip = translate._strip_adjacent_repeat
    assert strip("образование, образование и права", target="ru") == \
        "образование и права"
    assert strip("образование и образование", target="ru") == "образование"
    # English default unchanged.
    assert strip("education, education and human rights") == \
        "education and human rights"


# ------------------------------------------------------------------- proper nouns

def test_proper_nouns_finds_cyrillic_names_mid_sentence():
    names = translate._proper_nouns(
        "Вчера Катар финансировал Хамас без ограничений", target="ru")
    assert names == {"катар", "хамас"}
    # Sentence-initial capitals are not names.
    assert translate._proper_nouns("Вчера шёл дождь", target="ru") == set()


def test_proper_nouns_empty_for_caseless_scripts():
    assert translate._proper_nouns("קטר מימנה את חמאס", target="he") == set()
    assert translate._proper_nouns("قطر مولت حماس", target="ar") == set()
    assert translate._proper_nouns("カタールはハマスに", target="ja") == set()


def test_proper_nouns_english_behavior_unchanged():
    names = translate._proper_nouns("The emirate of Qatar funded Hamas")
    assert names == {"qatar", "hamas"}


# ---------------------------------------------------------------------- the prompt

def test_translate_instruction_contractions_only_for_english():
    en = translate._translate_instruction("текст", "ru", "en")
    assert "contractions" in en and "we're" in en
    ru = translate._translate_instruction("שלום", "he", "ru")
    assert "contractions" not in ru and "we're" not in ru
    assert "text-to-speech" in ru                       # generic rule still present
    assert "Russian" in ru                              # target named via _LANG_NAMES


def test_lang_names_cover_the_qwen_targets():
    for code, name in [("zh", "Chinese"), ("ja", "Japanese"), ("ko", "Korean"),
                       ("uk", "Ukrainian"), ("ru", "Russian"), ("pt", "Portuguese")]:
        assert translate._lang(code) == name


# ------------------------------------------------------- end-to-end guard threading

def test_run_marks_untranslated_output_as_target_aware(monkeypatch):
    # The mt_failed gate uses the real target: Russian output for target="ru" is a
    # success, English output for target="ru" is a failure (the old hardcoded Latin
    # gate had this exactly backwards).
    assert translate.is_target_text("Катар подкупает Европу газом", target="ru")
    assert not translate.is_target_text("Qatar bribes Europe with gas", target="ru")


def test_generate_repairs_a_repeated_word_in_russian(monkeypatch):
    replies = iter([
        "распространяя образование, образование и права человека.",
        "распространяя образование, воспитание и права человека.",
    ])
    monkeypatch.setattr(translate, "_run", lambda *a, **k: next(replies))
    out = translate.generate(None, None, "he-text", source="he", target="ru")
    assert out == "распространяя образование, воспитание и права человека."
