"""Phase 2 — Hebrew source to any target language (pure logic, no models).

The two-script world (Latin == target, Hebrew == source) is replaced by
`dubbing.script`, parameterized by the actual language pair. These tests pin the
generalized behaviour: a Cyrillic target is kept by script exactly as English
was, Latin text is no longer magically "the target", and a same-script pair
(en→es, ru→uk) disables every script shortcut so the LID path decides instead.
"""

from __future__ import annotations

import pytest

from dubbing import script, segments, transcript


def mkwords(spec, spk="S0"):
    return [{"t": t, "text": w, "brk": False, "spk": spk} for t, w in spec]


# --------------------------------------------------------------------------- script

def test_script_for_names_the_writing_system():
    assert script.script_for("en") == "latin"
    assert script.script_for("es") == "latin"
    assert script.script_for("he") == "hebrew"
    assert script.script_for("ru") == "cyrillic"
    assert script.script_for("ar") == "arabic"
    assert script.script_for("xx") == "latin"      # unknown codes default to latin


def test_same_script_tells_when_letters_prove_nothing():
    assert not script.same_script("he", "en")
    assert not script.same_script("he", "ru")
    assert not script.same_script("he", "ar")
    assert not script.same_script("en", "ru")
    assert script.same_script("en", "es")
    assert script.same_script("ru", "uk")
    assert script.same_script("he", "yi")


def test_script_ratio_counts_letters_of_the_language_s_script():
    assert script.script_ratio("Привет мир", "ru") == 1.0
    assert script.script_ratio("Привет world", "ru") == pytest.approx(6 / 11)
    assert script.script_ratio("שלום", "ru") == 0.0
    assert script.script_ratio("مرحبا بالعالم", "ar") == 1.0
    assert script.script_ratio("330,000 …", "en") == 0.0   # no letters, no evidence
    assert script.is_script("مرحبا بالعالم", "ar")
    assert not script.is_script("مرحبا", "he")
    # Same script: letters cannot tell es from en, and is_script does not claim to.
    assert script.is_script("hello there", "es")


def test_text_bucket_is_pair_driven_and_off_for_same_script_pairs():
    assert segments.text_bucket("Привет", "he", "ru") == "target"
    assert segments.text_bucket("שלום", "he", "ru") == "source"
    assert segments.text_bucket("hello", "he", "ru") is None   # neither of the pair
    assert segments.text_bucket("330", "he", "ru") is None
    assert segments.text_bucket("hola", "en", "es") is None    # same-script: no signal
    # Legacy he→en wrapper keeps its historical vocabulary.
    assert segments.script_of("hello") == "latin"
    assert segments.script_of("שלום") == "source"
    assert segments.script_of("330") is None


# --------------------------------------------------------------------------- mark_keep

def test_mark_keep_ru_keeps_cyrillic_and_not_latin():
    segs = [
        {"id": 0, "start": 0, "end": 5, "speaker": "A", "text": "Привет всем добрый вечер"},
        {"id": 1, "start": 5, "end": 10, "speaker": "B", "text": "This is English speech"},
        {"id": 2, "start": 10, "end": 15, "speaker": "C", "text": "שלום לכולם"},
    ]
    segments.mark_keep(segs, None, "ru")
    assert (segs[0]["keep"], segs[0]["keep_reason"]) == (True, "target_lang")
    assert not segs[1]["keep"]        # Latin is just a third language now
    assert not segs[2]["keep"]        # the source is still dubbed


def test_mark_keep_ru_speaker_rule_uses_the_target_script():
    segs = [
        {"id": 0, "start": 0, "end": 5, "speaker": "A", "text": "Привет всем добрый вечер"},
        {"id": 1, "start": 5, "end": 10, "speaker": "A", "text": "Ещё немного по-русски"},
        # Phonetically transcribed Russian: only the speaker rule can catch it.
        {"id": 2, "start": 10, "end": 15, "speaker": "A", "text": "עוד קצת עברית"},
        {"id": 3, "start": 15, "end": 20, "speaker": "B", "text": "שלום לכולם"},
    ]
    segments.mark_keep(segs, None, "ru")
    assert [s["keep_reason"] for s in segs] == ["target_lang", "target_lang",
                                                "speaker_en", None]


def test_mark_keep_ru_span_names_its_reason_honestly():
    segs = [{"id": 0, "start": 10.0, "end": 14.0, "speaker": "B", "text": "Привет мир"}]
    spans = [{"start": 10.0, "end": 14.0, "lang": "ru", "text": "Привет мир",
              "words": [{"t": 10.0, "text": "x"}]}]
    segments.mark_keep(segs, spans, "ru")
    assert (segs[0]["keep"], segs[0]["keep_reason"]) == (True, "target_lang")
    # A span in yet another language stays "foreign", as before.
    ar = [{"id": 0, "start": 10.0, "end": 14.0, "speaker": "B", "text": "…"}]
    segments.mark_keep(ar, [{"start": 10.0, "end": 14.0, "lang": "ar", "text": "…",
                             "words": [{"t": 10.0, "text": "…"}]}], "ru")
    assert (ar[0]["keep"], ar[0]["keep_reason"]) == (True, "foreign")


def test_mark_keep_en_legacy_values_are_intact():
    segs = [
        {"id": 0, "start": 0, "end": 5, "speaker": "A", "text": "This is English speech"},
        {"id": 1, "start": 5, "end": 10, "speaker": "B", "text": "שלום לכולם"},
    ]
    segments.mark_keep(segs, None, "en")
    assert (segs[0]["keep"], segs[0]["keep_reason"]) == (True, "latin")
    assert not segs[1]["keep"]


def test_mark_keep_same_script_pair_relies_on_spans_only():
    # ru→uk: letters cannot tell the two apart, so text alone never keeps —
    # only the LID-backed span rule does.
    segs = [
        {"id": 0, "start": 0, "end": 5, "speaker": "A", "text": "Привіт усім друзі"},
        {"id": 1, "start": 10, "end": 14, "speaker": "B", "text": "Дякую щиро вам"},
    ]
    segments.mark_keep(segs, None, "uk", "ru")
    assert not segs[0]["keep"] and not segs[1]["keep"]
    spans = [{"start": 10.0, "end": 14.0, "lang": "uk", "text": "Дякую щиро вам",
              "words": [{"t": 10.0, "text": "x"}]}]
    segments.mark_keep(segs, spans, "uk", "ru")
    assert not segs[0]["keep"]
    assert (segs[1]["keep"], segs[1]["keep_reason"]) == (True, "target_lang")


# --------------------------------------------------------------------------- segmentation

def test_sustained_cyrillic_run_splits_for_a_ru_target():
    words = mkwords([(0.0, "שלום"), (0.3, "עולם"),
                     (0.6, "это"), (0.9, "русская"), (1.2, "речь"), (1.5, "друзья")])
    segs = segments.words_to_segments(words, "he", "ru")
    assert [segments.text_bucket(s["text"], "he", "ru") for s in segs] == ["source", "target"]


def test_embedded_cyrillic_token_stays_in_the_hebrew_segment():
    # A lone Cyrillic brand mid-sentence must not become its own segment for a
    # ru target, exactly as a lone Latin acronym does not for an en target.
    words = mkwords([(0.0, "קטאר"), (0.3, "מממן"), (0.6, "את"),
                     (0.9, "Правда,"), (1.2, "את"), (1.5, "העיתון")])
    segs = segments.words_to_segments(words, "he", "ru")
    assert len(segs) == 1
    assert "Правда," in segs[0]["text"]


def test_same_script_pair_never_splits_on_script():
    words = mkwords([(0.0, "hello"), (0.3, "world"),
                     (0.6, "hola"), (0.9, "buenos"), (1.2, "dias"), (1.5, "amigos")])
    assert len(segments.words_to_segments(words, "en", "es")) == 1


# ------------------------------------------------------------- speaker-turn splitting

def spkwords(spec):
    """Words with explicit ends and per-word speakers: (t, end, text, spk)."""
    return [{"t": t, "end": e, "text": w, "brk": False, "spk": s} for t, e, w, s in spec]


FAST_QA = [(0.0, 0.35, "aaa", "S0"), (0.4, 0.75, "bbb", "S0"), (0.8, 1.15, "ccc", "S0"),
           (1.2, 1.55, "ddd", "S1"), (1.6, 1.95, "eee", "S1"), (2.0, 2.35, "fff", "S1")]
FAST_QA_TURNS = [{"speaker": "S0", "start": 0.0, "end": 1.18},
                 {"speaker": "S1", "start": 1.18, "end": 2.4}]


def test_interior_turn_boundary_splits_the_segment():
    # Fast dialogue: the handover pause (0.05s) is far below SPEAKER_GAP, so the
    # gap-confirmed rule fuses both turns — the turn list must cut them apart.
    segs = segments.words_to_segments(spkwords(FAST_QA), turns=FAST_QA_TURNS)
    assert [s["text"] for s in segs] == ["aaa bbb ccc", "ddd eee fff"]
    assert [s["speaker"] for s in segs] == ["S0", "S1"]
    assert segs[0]["end"] <= segs[1]["start"]          # non-overlapping


def test_no_turns_or_single_speaker_turns_change_nothing():
    # Without turn data the fused segment stays fused (degenerate diarization
    # fallback), and same-speaker turns carry no boundary at all.
    assert len(segments.words_to_segments(spkwords(FAST_QA))) == 1
    same = [{"speaker": "S0", "start": 0.0, "end": 1.2},
            {"speaker": "S0", "start": 1.2, "end": 2.4}]
    assert len(segments.words_to_segments(spkwords(FAST_QA), turns=same)) == 1


def test_turn_boundary_on_a_segment_edge_does_not_split():
    words = spkwords([(0.0, 0.35, "aaa", "S0"), (0.4, 0.75, "bbb", "S0"),
                      (0.8, 1.15, "ccc", "S0"),
                      (3.0, 3.35, "ddd", "S1"), (3.4, 3.75, "eee", "S1"),
                      (3.8, 4.15, "fff", "S1")])
    turns = [{"speaker": "S0", "start": 0.0, "end": 1.2},
             {"speaker": "S1", "start": 3.0, "end": 4.2}]
    segs = segments.words_to_segments(words, turns=turns)   # pause already split it
    assert [s["text"] for s in segs] == ["aaa bbb ccc", "ddd eee fff"]


# --------------------------------------------------------------- micro-segment merging

def test_micro_segment_rejoins_a_same_speaker_neighbour_across_a_pause():
    # Two words, half a second: too short for TTS. The 1.0s gap is past
    # MERGE_GAP, but a sub-MIN_SEG_SEC fragment may cross it like a lone word.
    words = spkwords([(0.0, 0.35, "aaa", "S0"), (0.4, 0.75, "bbb", "S0"),
                      (0.8, 1.15, "ccc", "S0"), (1.2, 1.55, "ddd", "S0"),
                      (1.6, 1.95, "eee.", "S0"),
                      (2.95, 3.15, "ff", "S0"), (3.25, 3.45, "gg", "S0")])
    segs = segments.words_to_segments(words)
    assert len(segs) == 1
    assert segs[0]["text"].endswith("eee. ff gg")


def test_micro_segment_with_only_other_speaker_neighbours_survives():
    # A short cross-speaker interjection must not merge (that would re-fuse two
    # characters) and must not be dropped — it stays its own dubbable segment.
    words = spkwords([(0.0, 0.35, "aaa", "S0"), (0.4, 0.75, "bbb", "S0"),
                      (0.8, 1.15, "ccc", "S0"), (1.2, 1.55, "ddd.", "S0"),
                      (2.55, 2.75, "ff", "S1"), (2.85, 3.05, "gg", "S1"),
                      (4.05, 4.4, "hhh", "S0"), (4.45, 4.8, "iii", "S0"),
                      (4.85, 5.2, "jjj", "S0")])
    segs = segments.words_to_segments(words)
    assert [s["speaker"] for s in segs] == ["S0", "S1", "S0"]
    assert segs[1]["text"] == "ff gg"


def test_turn_split_fragment_remerges_into_its_own_speaker():
    # The turn boundary cuts a short same-speaker tail off the fused segment;
    # the stub merge then folds it back into its speaker's previous sentence,
    # while the other speaker's turn stands alone.
    words = spkwords([(0.0, 0.35, "aaa", "S0"), (0.4, 0.75, "bbb", "S0"),
                      (0.8, 1.15, "ccc", "S0"), (1.2, 1.55, "ddd", "S0"),
                      (1.6, 1.95, "eee.", "S0"),
                      (2.25, 2.5, "fff", "S0"), (2.55, 2.8, "ggg", "S0"),
                      (2.85, 3.1, "hhh", "S1"), (3.15, 3.4, "iii", "S1"),
                      (3.45, 3.7, "jjj", "S1"), (3.75, 4.0, "kkk", "S1")])
    turns = [{"speaker": "S0", "start": 0.0, "end": 2.82},
             {"speaker": "S1", "start": 2.82, "end": 4.2}]
    segs = segments.words_to_segments(words, turns=turns)
    assert [s["text"] for s in segs] == ["aaa bbb ccc ddd eee. fff ggg",
                                        "hhh iii jjj kkk"]
    assert [s["speaker"] for s in segs] == ["S0", "S1"]
    assert segs[0]["end"] <= segs[1]["start"]


# --------------------------------------------------------------------------- transcript

def test_recovered_text_routes_by_target_script_cross_script_only():
    # recover_gaps files target-script text as a kept span and everything else as
    # dubbable words; text_is_target is that routing decision.
    assert transcript.text_is_target("Привет мир", "he", "ru")
    assert not transcript.text_is_target("שלום עולם", "he", "ru")
    assert not transcript.text_is_target("This is English", "he", "ru")
    # Same-script pair: script proves nothing; LID decides downstream.
    assert not transcript.text_is_target("Hola amigos", "en", "es")
    # The he→en path answers exactly as latin_ratio > 0.5 did.
    assert transcript.text_is_target("Frankly I had", "he", "en")
    assert not transcript.text_is_target("שיבוש של אנגלית", "he", "en")


def test_foreign_spans_ru_target_marks_cyrillic_not_latin():
    words = [{"t": 1.0, "text": "שלום"}, {"t": 5.0, "text": "Привет"},
             {"t": 5.4, "text": "мир"}, {"t": 5.9, "text": "друзья"},
             {"t": 6.3, "text": "Frankly"}, {"t": 30.0, "text": "עברית"}]
    spans = transcript.foreign_spans(words, src="he", tgt="ru")
    assert len(spans) == 1
    assert spans[0]["text"] == "Привет мир друзья"
    assert spans[0]["start"] == 5.0


def test_foreign_spans_same_script_pair_returns_no_spans():
    words = [{"t": 1.0, "text": "hola"}, {"t": 1.4, "text": "amigos"},
             {"t": 1.8, "text": "buenos"}, {"t": 2.2, "text": "dias"}]
    assert transcript.foreign_spans(words, src="en", tgt="es") == []
    # ...while the he→en case still finds the same run.
    assert len(transcript.foreign_spans(words, src="he", tgt="en")) == 1
