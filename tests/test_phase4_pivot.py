"""Phase 4 — English-pivot translation (src→en→tgt). Pure logic, no models."""

from __future__ import annotations

from dubbing import manifest, translate


# ------------------------------------------------------------------ pivot predicate

def test_pivot_via_english_truth_table():
    assert translate.pivot_via_english("he", "ru")
    assert translate.pivot_via_english("ar", "es")
    assert not translate.pivot_via_english("he", "en")   # en target: direct
    assert not translate.pivot_via_english("en", "ru")   # en source: direct
    assert not translate.pivot_via_english("en", "en")


# ------------------------------------------------------------------ run() call flow

def _dub_manifest(texts):
    m = manifest.new({"input": "x"})
    m["segments"] = [
        {"id": i, "start": float(i), "end": float(i) + 1.0, "speaker": "S0",
         "text": t, "keep": False, "keep_reason": None}
        for i, t in enumerate(texts)
    ]
    return m


def _fake_translate(monkeypatch, reply):
    calls = []

    def fake_generate(tok, mdl, text, *, source, target, context="", preceding="",
                      device=None, **kw):
        calls.append({"text": text, "source": source, "target": target,
                      "preceding": preceding})
        return reply(text, source, target)

    monkeypatch.setattr(translate, "load", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(translate, "free", lambda mdl: None)
    monkeypatch.setattr(translate, "generate", fake_generate)
    return calls


_EN = {"שורה אחת": "english line one", "שורה שתיים": "english line two"}
_RU = {"english line one": "русская строка один",
       "english line two": "русская строка два"}


def test_run_pivots_he_ru_in_two_hops_and_stores_the_intermediate(monkeypatch, tmp_path):
    calls = _fake_translate(monkeypatch, lambda text, s, t:
                            _EN[text] if t == "en" else _RU[text])
    m = _dub_manifest(["שורה אחת", "שורה שתיים"])
    translate.run(m, tmp_path, source="he", target="ru")

    # Two hops per segment, in order: he→en then en→ru.
    assert [(c["source"], c["target"]) for c in calls] == \
        [("he", "en"), ("en", "ru")] * 2
    # Hop 1 translates the source text; hop 2 translates the English intermediate.
    assert calls[0]["text"] == "שורה אחת"
    assert calls[1]["text"] == "english line one"
    # Preceding context: hop 1 sees the previous segment's SOURCE text, hop 2 the
    # previous segment's just-produced English intermediate (empty for the first).
    assert calls[0]["preceding"] == "" and calls[1]["preceding"] == ""
    assert calls[2]["preceding"] == "שורה אחת"
    # Hop 2 never gets a preceding line: clean English needs no disambiguation,
    # and with one the model once swapped an entity (Jabhat al-Nusra → "al-Qaeda's
    # Front", reproduced deterministically).
    assert calls[3]["preceding"] == ""

    seg0, seg1 = m["segments"]
    assert seg0["text_en"] == "русская строка один"
    assert seg0["text_mid"] == "english line one"
    assert seg1["text_en"] == "русская строка два"
    assert seg1["text_mid"] == "english line two"


def test_run_failed_first_hop_keeps_the_segment_as_mt_failed(monkeypatch, tmp_path):
    # The English hop comes back still in Hebrew → is_target_text(mid, "en")
    # fails, so the segment is kept exactly as a direct-translation failure is.
    calls = _fake_translate(monkeypatch, lambda text, s, t: "עדיין עברית")
    m = _dub_manifest(["שורה אחת"])
    translate.run(m, tmp_path, source="he", target="ru")

    assert len(calls) == 1                                # no second hop attempted
    seg = m["segments"][0]
    assert seg["keep"] and seg["keep_reason"] == "mt_failed"
    assert seg["text_en"] == "שורה אחת"
    assert "text_mid" not in seg


def test_run_failed_second_hop_keeps_the_segment_as_mt_failed(monkeypatch, tmp_path):
    # English hop succeeds, the en→ru hop echoes English → not Russian → keep.
    calls = _fake_translate(monkeypatch, lambda text, s, t:
                            _EN.get(text, text) if t == "en" else "still english")
    m = _dub_manifest(["שורה אחת"])
    translate.run(m, tmp_path, source="he", target="ru")

    assert len(calls) == 2
    seg = m["segments"][0]
    assert seg["keep"] and seg["keep_reason"] == "mt_failed"
    assert seg["text_en"] == "שורה אחת"
    assert "text_mid" not in seg


def test_run_he_en_stays_direct_and_stores_no_intermediate(monkeypatch, tmp_path):
    calls = _fake_translate(monkeypatch, lambda text, s, t: _EN[text])
    m = _dub_manifest(["שורה אחת", "שורה שתיים"])
    translate.run(m, tmp_path, source="he", target="en")

    assert [(c["source"], c["target"]) for c in calls] == [("he", "en")] * 2
    assert calls[1]["preceding"] == "שורה אחת"            # source-text context, as before
    for seg in m["segments"]:
        assert "text_mid" not in seg
    assert m["segments"][0]["text_en"] == "english line one"


def test_run_pivots_a_foreign_subtitle_keep(monkeypatch, tmp_path):
    # A third-language keep's subtitle goes span-lang → en → target too.
    calls = _fake_translate(monkeypatch, lambda text, s, t:
                            "english aside" if t == "en" else "русская вставка")
    m = _dub_manifest([])
    m["segments"] = [{"id": 0, "start": 0.0, "end": 1.0, "speaker": "S0",
                      "text": "كلام عربي", "keep": True, "keep_reason": "foreign",
                      "lang": "ar"}]
    translate.run(m, tmp_path, source="he", target="ru")

    assert [(c["source"], c["target"], c["text"]) for c in calls] == \
        [("ar", "en", "كلام عربي"), ("en", "ru", "english aside")]
    assert m["segments"][0]["text_en"] == "русская вставка"


def test_run_foreign_subtitle_failure_falls_back_to_ellipsis(monkeypatch, tmp_path):
    calls = _fake_translate(monkeypatch, lambda text, s, t: "لا يزال عربيا")
    m = _dub_manifest([])
    m["segments"] = [{"id": 0, "start": 0.0, "end": 1.0, "speaker": "S0",
                      "text": "كلام عربي", "keep": True, "keep_reason": "foreign",
                      "lang": "ar"}]
    translate.run(m, tmp_path, source="he", target="ru")
    assert len(calls) == 1                                # first hop failed, no second
    assert m["segments"][0]["text_en"] == "…"


# ------------------------------------------------------------------ manifest wiring

def test_segment_keys_carry_the_intermediate():
    assert "text_mid" in manifest.SEGMENT_KEYS


def test_reset_translate_drops_both_translation_fields():
    m = manifest.new({"input": "x"})
    m["segments"] = [{"id": 0, "keep": False, "keep_reason": None,
                      "text_en": "русская строка", "text_mid": "english line"}]
    manifest.reset_stage(m, "translate")
    assert "text_en" not in m["segments"][0]
    assert "text_mid" not in m["segments"][0]
