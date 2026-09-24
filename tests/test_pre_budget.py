"""Movie translate-time length check: measure against the segment, revise once.

The invariants pinned here: documentary is untouched (its prompt-identity test
lives in test_genre.py; here, the helper itself never fires for it because the
call is genre-gated), the gate only trips on clear overruns, a shorten refusal
keeps the full translation, and the pivot path shortens from the English
intermediate exactly as the timeline's rescue does.
"""

from __future__ import annotations

from types import SimpleNamespace

from dubbing import translate


def _seg(dur: float, text_en: str, text: str = "מקור") -> dict:
    return {"id": 1, "start": 10.0, "end": 10.0 + dur,
            "text": text, "text_en": text_en}


def test_budget_is_duration_times_rate_with_a_floor():
    assert translate.pre_budget_units(_seg(2.0, ""), "en") == 5      # 2s * 2.9
    assert translate.pre_budget_units(_seg(0.4, ""), "en") == 3      # the floor
    assert translate.pre_budget_units({"start": 5.0, "end": 5.0}, "en") is None


def test_budget_counts_characters_for_cjk_targets():
    words = translate.pre_budget_units(_seg(2.0, ""), "en")
    chars = translate.pre_budget_units(_seg(2.0, ""), "ja")
    assert chars == 13 and chars > words                             # 2s * 6.5


def test_a_line_within_slack_is_left_alone(monkeypatch):
    def never(*a, **k):
        raise AssertionError("shorten must not be called")

    monkeypatch.setattr(translate, "shorten", never)
    seg = _seg(3.0, "eight words fits in three seconds just fine")
    h = SimpleNamespace(processor=None, model=None, device=None)
    assert not translate._fit_movie_budget(h, seg, seg_src="he", seg_tgt="en",
                                           seg_pivot=False, mids={}, context="",
                                           preceding="")


def test_a_clear_overrun_is_shortened_from_the_source(monkeypatch):
    calls = {}

    def fake_shorten(processor, model, src_text, current, budget, **kw):
        calls.update(src=src_text, budget=budget, source=kw["source"])
        return "short line"

    monkeypatch.setattr(translate, "shorten", fake_shorten)
    long = " ".join(["word"] * 30)
    seg = _seg(2.0, long)
    h = SimpleNamespace(processor=None, model=None, device=None)
    assert translate._fit_movie_budget(h, seg, seg_src="he", seg_tgt="en",
                                       seg_pivot=False, mids={}, context="",
                                       preceding="")
    assert seg["text_en"] == "short line"
    assert calls == {"src": "מקור", "budget": 5, "source": "he"}


def test_a_refused_shorten_keeps_the_full_translation(monkeypatch):
    monkeypatch.setattr(translate, "shorten", lambda *a, **k: None)
    long = " ".join(["word"] * 30)
    seg = _seg(2.0, long)
    h = SimpleNamespace(processor=None, model=None, device=None)
    assert not translate._fit_movie_budget(h, seg, seg_src="he", seg_tgt="en",
                                           seg_pivot=False, mids={}, context="",
                                           preceding="")
    assert seg["text_en"] == long


def test_a_pivot_line_shortens_from_its_english_intermediate(monkeypatch):
    calls = {}

    def fake_shorten(processor, model, src_text, current, budget, **kw):
        calls.update(src=src_text, source=kw["source"])
        return "короткая строка"

    monkeypatch.setattr(translate, "shorten", fake_shorten)
    seg = _seg(2.0, " ".join(["слово"] * 30))
    h = SimpleNamespace(processor=None, model=None, device=None)
    assert translate._fit_movie_budget(h, seg, seg_src="he", seg_tgt="ru",
                                       seg_pivot=True, mids={1: "the english mid"},
                                       context="", preceding="")
    assert calls == {"src": "the english mid", "source": "en"}


def test_a_pivot_line_with_no_intermediate_is_left_alone(monkeypatch):
    def never(*a, **k):
        raise AssertionError("shorten must not be called")

    monkeypatch.setattr(translate, "shorten", never)
    seg = _seg(2.0, " ".join(["слово"] * 30))
    h = SimpleNamespace(processor=None, model=None, device=None)
    assert not translate._fit_movie_budget(h, seg, seg_src="he", seg_tgt="ru",
                                           seg_pivot=True, mids={}, context="",
                                           preceding="")
