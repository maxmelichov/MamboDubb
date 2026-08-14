"""Phase 5 — deterministic digit→word spelling. Pure logic, no models.

The v22 prompt asked the *model* to write numbers as words and it obeyed
unreliably: "ב-504" became "five zero four" on the English hop, which the ru hop
then read as the year 2004 and rendered "в две тысячи четвёртом году". Since
translate/v24 conversion is code (dubbing/numwords.py via num2words) applied to
the English layer, and models only translate/inflect existing number-words; a
target-side safety net spells any digits a model still passes through.
"""

from __future__ import annotations

import re

from dubbing import manifest, translate
from dubbing.numwords import spell_numbers


# ------------------------------------------------------------ spell_numbers(en)

def test_cardinal():
    assert spell_numbers("He lost 504 soldiers.", "en") == \
        "He lost five hundred and four soldiers."


def test_year_form_for_standalone_1900_to_2099():
    assert spell_numbers("In 1982 it began.", "en") == \
        "In nineteen eighty-two it began."
    assert "twenty seventeen" in spell_numbers("Since 2017.", "en")
    # Outside the year window: plain cardinal.
    assert "one thousand" in spell_numbers("about 1500 men", "en")


def test_percent():
    assert spell_numbers("about 65% of them", "en") == \
        "about sixty-five percent of them"
    assert "процентов" in spell_numbers("около 65%", "ru")


def test_thousands_separator_stripped_before_spelling():
    assert spell_numbers("some 330,000 people", "en") == \
        "some three hundred and thirty thousand people"


def test_decimal_english_only():
    assert spell_numbers("worth 3.5 billion", "en") == \
        "worth three point five billion"
    # Non-English decimals stay digits rather than risk a wrong reading.
    assert spell_numbers("около 3.5 литра", "ru") == "около 3.5 литра"


def test_times_ranges_and_glued_digits_left_untouched():
    assert spell_numbers("at 6:30 in the morning", "en") == \
        "at 6:30 in the morning"
    assert spell_numbers("from 2010-2015", "en") == "from 2010-2015"
    assert spell_numbers("the mid-1990s", "en") == "the mid-1990s"
    assert spell_numbers("the 1990s ended", "en") == "the 1990s ended"
    assert spell_numbers("version 1.2.3 here", "en") == "version 1.2.3 here"


def test_unsupported_num2words_language_returns_text_unchanged():
    assert spell_numbers("504 things", "xx") == "504 things"


# --------------------------------------------------- target-language safety net

def test_russian_digit_with_grammatical_suffix_becomes_nominative_cardinal():
    out = spell_numbers("в 504-м бою", "ru")
    assert not re.search(r"\d", out)
    assert "пятьсот" in out


# ------------------------------------------------------------- run() wiring

def test_run_spells_hop1_output_before_hop2_and_final_text_has_no_digits(
        monkeypatch, tmp_path):
    calls = []

    def fake_generate(tok, mdl, text, *, source, target, context="",
                      preceding="", device=None, **kw):
        calls.append({"text": text, "source": source, "target": target,
                      "numbers_spelled": kw.get("numbers_spelled", False)})
        if target == "en":
            return "He lost 504 soldiers."          # English hop leaves digits
        return "Он потерял 504 бойцов."             # model disobeys → safety net

    monkeypatch.setattr(translate, "load", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(translate, "free", lambda mdl: None)
    monkeypatch.setattr(translate, "generate", fake_generate)
    m = manifest.new({"input": "x"})
    m["segments"] = [{"id": 0, "start": 0.0, "end": 1.0, "speaker": "S0",
                      "text": "הוא איבד 504 חיילים", "keep": False,
                      "keep_reason": None}]
    translate.run(m, tmp_path, source="he", target="ru")

    assert [(c["source"], c["target"]) for c in calls] == [("he", "en"), ("en", "ru")]
    # Hop 1 is not told anything about spelled numbers; hop 2 is.
    assert calls[0]["numbers_spelled"] is False
    assert calls[1]["numbers_spelled"] is True
    # Hop 2's input is the code-spelled intermediate, not the raw digit line.
    assert calls[1]["text"] == "He lost five hundred and four soldiers."
    seg = m["segments"][0]
    assert seg["text_mid"] == "He lost five hundred and four soldiers."
    # The Russian digits the model passed through were spelled by the net.
    assert not re.search(r"\d", seg["text_en"])
    assert "пятьсот" in seg["text_en"]


def test_run_en_target_always_spells_the_final_english(monkeypatch, tmp_path):
    monkeypatch.setattr(translate, "load", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(translate, "free", lambda mdl: None)
    monkeypatch.setattr(translate, "generate",
                        lambda *a, **k: "He lost 504 soldiers.")
    m = manifest.new({"input": "x"})
    m["segments"] = [{"id": 0, "start": 0.0, "end": 1.0, "speaker": "S0",
                      "text": "הוא איבד 504 חיילים", "keep": False,
                      "keep_reason": None}]
    translate.run(m, tmp_path, source="he", target="en")
    assert m["segments"][0]["text_en"] == "He lost five hundred and four soldiers."
