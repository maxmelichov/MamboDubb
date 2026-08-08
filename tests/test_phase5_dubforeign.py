"""Phase 5 — --dub-foreign: dub confident third-language spans. Pure logic, no models."""

from __future__ import annotations

from dubbing import manifest, segments, translate

# ------------------------------------------------------------------ segments gating

AR_WORDS = [{"t": 2.0, "text": "كلام"}, {"t": 2.5, "text": "عربي"},
            {"t": 3.0, "text": "واضح"}]


def _spliced(lang, words, tgt="ru"):
    """Segments carved out of one foreign span, as segments.run builds them."""
    spans = [{"start": 2.0, "end": 4.0, "lang": lang, "words": words}]
    segs = segments.splice_foreign_spans([], spans, None, "he", tgt)
    for i, s in enumerate(segs):
        s["id"] = i
    return segs, spans


def test_confident_foreign_span_keeps_by_default():
    # Current behavior pinned: without the flag a confident Arabic span on a
    # he→ru run still plays its original audio as a "foreign" keep.
    segs, spans = _spliced("ar", AR_WORDS)
    segments.mark_keep(segs, spans, "ru", "he")
    assert segs and all(s["keep"] and s["keep_reason"] == "foreign" for s in segs)


def test_confident_foreign_span_is_dubbable_with_dub_foreign():
    segs, spans = _spliced("ar", AR_WORDS)
    segments.mark_keep(segs, spans, "ru", "he", dub_foreign=True)
    assert segs
    for s in segs:
        assert s["keep"] is False and s["keep_reason"] is None
        assert s["lang"] == "ar"          # translate reads this as its source language


def test_und_span_keeps_even_with_dub_foreign():
    segs, spans = _spliced("und", AR_WORDS)
    segments.mark_keep(segs, spans, "ru", "he", dub_foreign=True)
    assert segs and all(s["keep"] and s["keep_reason"] == "foreign" for s in segs)


def test_placeholder_text_span_keeps_even_with_dub_foreign():
    # The ASR was not confident enough to read the span ("…" fallback): there is
    # nothing to translate, so original audio plays exactly as before.
    segs, spans = _spliced("ar", [{"t": 2.0, "text": "…"}])
    segments.mark_keep(segs, spans, "ru", "he", dub_foreign=True)
    assert segs and all(s["keep"] and s["keep_reason"] == "foreign" for s in segs)


def test_target_language_span_keeps_even_with_dub_foreign():
    # A span already in the target language is never dubbed — it IS the target.
    words = [{"t": 2.0, "text": "plain"}, {"t": 2.5, "text": "english"},
             {"t": 3.0, "text": "line"}]
    segs, spans = _spliced("en", words, tgt="en")
    segments.mark_keep(segs, spans, "en", "he", dub_foreign=True)
    assert segs and all(s["keep"] and s["keep_reason"] == "latin" for s in segs)


# ------------------------------------------------------------------ translate routing

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


_EN = {"שורה אחת": "english line one"}
_RU = {"english line one": "русская строка один",
       "english aside": "русская вставка"}


def _mixed_manifest():
    m = manifest.new({"input": "x"})
    m["segments"] = [
        {"id": 0, "start": 0.0, "end": 1.0, "speaker": "S0",
         "text": "שורה אחת", "keep": False, "keep_reason": None},
        {"id": 1, "start": 1.0, "end": 2.0, "speaker": "S1",
         "text": "english aside", "keep": False, "keep_reason": None, "lang": "en"},
    ]
    return m


def test_run_translates_an_english_lang_segment_in_one_hop(monkeypatch, tmp_path):
    calls = _fake_translate(monkeypatch, lambda text, s, t:
                            _EN[text] if t == "en" else _RU[text])
    m = _mixed_manifest()
    translate.run(m, tmp_path, source="he", target="ru")

    # The he segment pivots (he→en→ru); the en segment goes en→ru directly.
    assert [(c["source"], c["target"]) for c in calls] == \
        [("he", "en"), ("en", "ru"), ("en", "ru")]
    assert calls[2]["text"] == "english aside"
    # Its neighbour spoke a different language — no preceding context is passed.
    assert calls[2]["preceding"] == ""

    seg0, seg1 = m["segments"]
    assert seg0["text_en"] == "русская строка один"
    assert seg0["text_mid"] == "english line one"     # pivot happened → stored
    assert seg1["text_en"] == "русская вставка"
    assert "text_mid" not in seg1                     # single hop → no intermediate


def test_run_lang_segment_mt_failure_keeps_original(monkeypatch, tmp_path):
    # The en→ru hop comes back still in English → not target text → the segment
    # keeps its original audio as mt_failed, exactly like any other dub failure.
    calls = _fake_translate(monkeypatch, lambda text, s, t: "still english")
    m = manifest.new({"input": "x"})
    m["segments"] = [{"id": 0, "start": 0.0, "end": 1.0, "speaker": "S0",
                      "text": "english aside", "keep": False, "keep_reason": None,
                      "lang": "en"}]
    translate.run(m, tmp_path, source="he", target="ru")

    assert len(calls) == 1                            # one direct hop, no pivot
    seg = m["segments"][0]
    assert seg["keep"] and seg["keep_reason"] == "mt_failed"
    assert seg["text_en"] == "english aside"
    assert "text_mid" not in seg
