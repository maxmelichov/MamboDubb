"""Phase 2 Hebrew source to any target language (pure logic, no models).

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
        # Transliterated Russian: not the target script, not the source script
        # either, so only the speaker rule can catch it.
        {"id": 2, "start": 10, "end": 15, "speaker": "A", "text": "Nu ladno poyekhali"},
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
    # gap-confirmed rule fuses both turns the turn list must cut them apart.
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


# -------------------------------------------------------------- inter-turn gap splitting

# Whisper smears word timings across a real pause (word gaps here are all far
# below GAP_SPLIT), while diarization's segmentation reports the silence between
# turns even when clustering hands both turns the same speaker label.
SMEARED = [(0.0, 0.4, "aaa", "S0"), (0.5, 0.9, "bbb", "S0"), (1.0, 1.4, "ccc", "S0"),
           (1.6, 2.4, "ddd", "S0"), (2.6, 3.0, "eee", "S0"), (3.1, 3.4, "fff", "S0")]


def test_same_speaker_gap_splits_at_midpoint_and_survives_merge():
    # Turns share a label but a 0.9s silence separates them: split at the gap's
    # midpoint (2.5). The right piece (0.8s, two words) is a stub 0.2s from a
    # substantial same-speaker neighbour _merge_stubs would fold it straight
    # back in, so the split point carries a brk marker that merging never crosses.
    words = spkwords(SMEARED)
    turns = [{"speaker": "S0", "start": 0.0, "end": 2.05},
             {"speaker": "S0", "start": 2.95, "end": 4.0}]
    segs = segments.words_to_segments(words, turns=turns)
    assert [s["text"] for s in segs] == ["aaa bbb ccc ddd", "eee fff"]
    assert segs[0]["end"] <= 2.5 <= segs[1]["start"]
    assert [s["speaker"] for s in segs] == ["S0", "S0"]
    # The caller's word dicts are not mutated by the brk marking.
    assert not any(w["brk"] for w in words)


def test_sub_threshold_gap_does_not_split():
    # 0.7s of inter-turn silence is below TURN_GAP_SPLIT; same speaker, so no
    # handover boundary either the segment stays whole.
    turns = [{"speaker": "S0", "start": 0.0, "end": 2.2},
             {"speaker": "S0", "start": 2.9, "end": 4.0}]
    assert len(segments.words_to_segments(spkwords(SMEARED), turns=turns)) == 1


def test_gap_aligned_with_an_existing_boundary_is_a_noop():
    # The words themselves pause (1.6s >= GAP_SPLIT), so the segments are already
    # split there; the diarization gap's midpoint (2.2) falls between them, not
    # strictly inside either, and changes nothing.
    words = spkwords([(0.0, 0.4, "aaa", "S0"), (0.5, 0.9, "bbb", "S0"),
                      (1.0, 1.4, "ccc", "S0"),
                      (3.0, 3.4, "ddd", "S0"), (3.5, 3.9, "eee", "S0"),
                      (4.0, 4.4, "fff", "S0")])
    turns = [{"speaker": "S0", "start": 0.0, "end": 1.5},
             {"speaker": "S0", "start": 2.9, "end": 4.5}]
    segs = segments.words_to_segments(words, turns=turns)
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
    # characters) and must not be dropped it stays its own dubbable segment.
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


# ----------------------------------------------------------- embedding-based turn split

def simmat(n, pairs, default=0.10):
    """Symmetric similarity matrix with 1.0 diagonal and given off-diagonal pairs."""
    m = [[default] * n for _ in range(n)]
    for i in range(n):
        m[i][i] = 1.0
    for (i, j), v in pairs.items():
        m[i][j] = m[j][i] = v
    return m


# Measured on the courtroom exchange (yt_keRQsy-rWxI, 48.5-66.2s): two alternating
# voices pyannote clustered as one. Turns in time order at ~49, 55, 58, 60, 62, 64s;
# ECAPA cosine similarity within each true voice 0.31-0.53, across voices 0.01-0.18.
COURTROOM = simmat(6, {
    (0, 2): 0.31, (0, 4): 0.33, (2, 4): 0.32,     # voice A: turns 0, 2, 4
    (1, 3): 0.32, (1, 5): 0.53, (3, 5): 0.32,     # voice B: turns 1, 3, 5
    (0, 1): 0.05, (0, 3): 0.12, (0, 5): 0.01, (2, 1): 0.18,
    (2, 3): 0.08, (2, 5): 0.11, (4, 1): 0.14, (4, 3): 0.06, (4, 5): 0.09,
})


def test_courtroom_matrix_splits_into_two_alternating_clusters():
    assert segments._split_embedding_clusters(COURTROOM) == [0, 1, 0, 1, 0, 1]


def test_uniform_high_similarity_never_splits():
    # One voice throughout: every pair well inside the cut threshold.
    assert segments._split_embedding_clusters(simmat(6, {}, default=0.85)) is None


def test_block_pattern_does_not_split():
    # AAABBB: two clean clusters but a single transition more likely one voice
    # across a scene change than two people, so the run stays whole.
    block = simmat(6, {(i, j): 0.80 for g in ((0, 1, 2), (3, 4, 5))
                       for i in g for j in g if i < j}, default=0.05)
    assert segments._split_embedding_clusters(block) is None


def test_three_clusters_at_the_cut_do_not_split():
    # Three mutually distant voices: not the unambiguous two-voice shape.
    three = simmat(6, {(0, 3): 0.80, (1, 4): 0.80, (2, 5): 0.80}, default=0.05)
    assert segments._split_embedding_clusters(three) is None


def test_lone_outlier_turn_does_not_split():
    # Five turns of one voice plus a single distant turn: a second cluster needs
    # at least two members to count as a second speaker.
    outlier = simmat(6, {(i, j): 0.80 for i in range(5) for j in range(i + 1, 5)},
                     default=0.05)
    assert segments._split_embedding_clusters(outlier) is None


def test_too_few_turns_or_empty_input_do_not_split():
    assert segments._split_embedding_clusters(simmat(2, {(0, 1): 0.05})) is None
    assert segments._split_embedding_clusters([]) is None


def mkturns(spec, spk="S0"):
    return [{"speaker": s, "start": a, "end": b} for a, b, s in spec] if spec and \
        len(spec[0]) == 3 else [{"speaker": spk, "start": a, "end": b} for a, b in spec]


def test_refine_turns_degenerate_inputs_pass_through(tmp_path):
    from pathlib import Path
    assert segments.refine_turns([], Path("missing.wav")) == []
    # A qualifying run but no vocals file: unchanged, never touches the model.
    turns = mkturns([(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)])
    assert segments.refine_turns(turns, tmp_path / "missing.wav") == turns
    # No qualifying run (too few turns / a sub-0.6s turn): unchanged, no model.
    short = mkturns([(0.0, 1.0), (2.0, 2.3), (4.0, 5.0)])
    assert segments.refine_turns(short, tmp_path / "missing.wav") == short


def test_refine_turns_survives_model_failure(tmp_path, monkeypatch):
    vocals = tmp_path / "vocals.wav"
    vocals.write_bytes(b"not audio")
    def boom(vocals, run):
        raise RuntimeError("no speechbrain")
    monkeypatch.setattr(segments, "_embed_turns", boom)
    turns = mkturns([(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)])
    assert segments.refine_turns(turns, vocals) == turns


def test_refine_turns_relabels_the_second_cluster(tmp_path, monkeypatch):
    import numpy as np
    vocals = tmp_path / "vocals.wav"
    vocals.write_bytes(b"not audio")
    # Orthogonal-ish synthetic embeddings reproducing the courtroom similarities.
    def fake_embed(vocals, run):
        e = np.linalg.cholesky(np.array(COURTROOM) + 1e-6 * np.eye(6))
        return e / np.linalg.norm(e, axis=1, keepdims=True)
    monkeypatch.setattr(segments, "_embed_turns", fake_embed)
    turns = mkturns([(48.5, 50.2), (55.2, 56.3), (57.5, 58.5),
                     (59.5, 60.5), (61.5, 62.1), (63.7, 66.2)])
    out = segments.refine_turns(turns, vocals)
    assert [t["speaker"] for t in out] == ["S0", "S0b", "S0", "S0b", "S0", "S0b"]
    # Timings untouched; the caller's turn dicts are not mutated.
    assert [(t["start"], t["end"]) for t in out] == [(t["start"], t["end"]) for t in turns]
    assert all(t["speaker"] == "S0" for t in turns)


def test_refine_turns_other_speaker_breaks_the_run(tmp_path, monkeypatch):
    # An interposed turn from another speaker splits the run; neither half has
    # three turns, so nothing qualifies and the model is never consulted.
    vocals = tmp_path / "vocals.wav"
    vocals.write_bytes(b"not audio")
    def boom(vocals, run):
        raise AssertionError("must not embed")
    monkeypatch.setattr(segments, "_embed_turns", boom)
    turns = mkturns([(0.0, 1.0, "S0"), (2.0, 3.0, "S0"),
                     (4.0, 5.0, "S1"), (6.0, 7.0, "S0"), (8.0, 9.0, "S0")])
    assert segments.refine_turns(turns, vocals) == turns


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


def test_short_turn_does_not_veto_refinement_and_inherits_side(monkeypatch):
    # Measured courtroom run: a 0.55s interjection sat inside 5 alternating
    # turns; the old all-turns->=0.6s rule vetoed the whole run.
    from pathlib import Path
    import numpy as np
    turns = [
        {"speaker": "S6", "start": 55.23, "end": 56.26},
        {"speaker": "S6", "start": 57.52, "end": 58.45},
        {"speaker": "S6", "start": 59.50, "end": 60.53},
        {"speaker": "S6", "start": 61.51, "end": 61.81},   # 0.30s too short
        {"speaker": "S6", "start": 63.72, "end": 66.20},
    ]
    alt = np.array([[1.0, 0.05, 0.9, 0.05], [0.05, 1.0, 0.05, 0.9],
                    [0.9, 0.05, 1.0, 0.05], [0.05, 0.9, 0.05, 1.0]])

    def fake_embed(vocals, run):
        assert len(run) == 4          # short turn excluded from embedding
        e = np.linalg.cholesky(alt + 1e-9 * np.eye(4))
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    monkeypatch.setattr(segments, "_embed_turns", fake_embed)
    monkeypatch.setattr(segments.Path, "is_file", lambda self: True)
    out = segments.refine_turns(turns, Path("vocals.wav"))
    labs = [t["speaker"] for t in out]
    # embedded turns are indices 0,1,2,4; the fake matrix alternates them
    assert labs[0] != labs[1] and labs[0] == labs[2] and labs[1] == labs[4]
    # the 0.55s turn (index 3, midpoint 61.8s) inherits from its nearest
    # embedded neighbour the 59.5-60.5s turn (midpoint 60.0s)
    assert labs[3] == labs[2]


def test_measured_six_turn_courtroom_matrix_splits():
    # The production run: same-voice linkage peaks at 0.77 distance (turn at
    # 61.5s), cross-voice at 0.88 the cut must sit between them.
    sims = [
        [1.00, 0.01, 0.33, 0.04, 0.32, 0.19],
        [0.01, 1.00, 0.00, 0.26, 0.14, 0.33],
        [0.33, 0.00, 1.00, 0.15, 0.53, 0.20],
        [0.04, 0.26, 0.15, 1.00, 0.16, 0.20],
        [0.32, 0.14, 0.53, 0.16, 1.00, 0.20],
        [0.19, 0.33, 0.20, 0.20, 0.20, 1.00],
    ]
    assert segments._split_embedding_clusters(sims) == [0, 1, 0, 1, 0, 1]


# ------------------------------------------------- CJK sources: shape of the text

def test_sentence_end_knows_the_cjk_terminators():
    # Without these a Japanese read has no interior sentence end at all: the
    # grouping loop never splits it, and the length splitter then cuts it
    # mid-clause into the two 6.9s chunks that produced the audible prosody jump.
    assert segments.SENTENCE_END.search("発表しました。")
    assert segments.SENTENCE_END.search("そうですか？")
    assert segments.SENTENCE_END.search("すごい！")
    assert segments.SENTENCE_END.search("举行了记者会。")
    assert segments.SENTENCE_END.search("と言った。」")      # quoted speech
    # Separators are not ends: the ideographic comma keeps the clause open.
    assert not segments.SENTENCE_END.search("首相は今日、")
    assert not segments.SENTENCE_END.search("首相は")
    # …and everything it already knew still ends a sentence.
    assert segments.SENTENCE_END.search("done.")
    assert segments.SENTENCE_END.search('he said."')
    assert segments.SENTENCE_END.search("אמר.")


def test_a_japanese_read_segments_one_segment_per_sentence():
    # Three sentences spoken without a pause long enough to split (GAP_SPLIT is
    # 0.70s): only the terminators can separate them. As one group they were a
    # single 6s span that the length splitter carved mid-clause instead.
    spec = []
    tokens = (["首相", "は", "今日", "会見", "した。"]
              + ["新しい", "経済", "政策", "を", "発表した。"]
              + ["市場", "は", "これ", "を", "歓迎した。"])
    for i, tok in enumerate(tokens):
        spec.append((round(i * 0.4, 3), tok))
    segs = segments.words_to_segments(mkwords(spec), "ja", "en")
    assert len(segs) == 3
    assert segs[0]["text"] == "首相は今日会見した。"
    assert segs[1]["text"] == "新しい経済政策を発表した。"
    assert segs[2]["text"] == "市場はこれを歓迎した。"


def test_cjk_word_tokens_join_without_the_spaces_the_language_has_not():
    assert script.join_words(["これ", "は", "テスト", "です", "。"]) == "これはテストです。"
    assert script.join_words(["ISIS", "を", "支援"]) == "ISISを支援"
    assert script.join_words(["hello", "world"]) == "hello world"
    # Korean is written WITH spaces — only its particles glue on, inside a word.
    assert script.join_words(["안녕", "하세요"]) == "안녕 하세요"
    assert script.join_words(["שלום", "עולם"]) == "שלום עולם"
    assert script.join_words(["", "  ", "solo"]) == "solo"


def test_split_words_is_the_inverse_join_words_needs():
    for text in ("これはテストです。", "hello world", "안녕 하세요",
                 "ISISを支援した", "שלום עולם"):
        assert script.join_words(script.split_words(text)) == text
    assert script.split_words("これは") == ["こ", "れ", "は"]
    assert script.split_words("hello world") == ["hello", "world"]


def test_cjk_and_hangul_are_measured_in_characters_not_words():
    assert script.speech_units("one two three", "en") == 3
    assert script.speech_units("首相は今日会見した", "ja") == 9
    assert script.speech_units("首相今天举行了记者会", "zh") == 10
    assert script.speech_units("가지 않습니다", "ko") == 6
    # Never zero: every caller divides by it.
    assert script.speech_units("", "ja") == 1
    assert script.speech_units("。", "ja") == 1


def test_cjk_shares_one_bucket_while_hangul_stands_apart():
    assert script.same_script("ja", "zh")        # kana+Han vs Han: one bucket
    assert not script.same_script("ja", "ko")
    assert not script.same_script("zh", "en")
    assert script.same_script("it", "pt")        # both Latin: no script signal
    assert segments.text_bucket("東京", "ja", "en") == "source"
    assert segments.text_bucket("Tokyo", "ja", "en") == "target"
    assert segments.text_bucket("서울", "ko", "en") == "source"
    assert segments.text_bucket("東京", "ja", "zh") is None    # same bucket
    assert segments.text_bucket("ciao", "it", "pt") is None    # same script


# --------------------------------------------- stranded fragments (issue #8)

def _orphan_pair():
    # The issue's exact shape: music buries the middle of one line, so the
    # transcript tears into a one-word opener and its continuation 5.85s later.
    return [
        {"id": 0, "start": 6.29, "end": 8.03, "speaker": "S0",
         "text": "בשנת", "keep": False},
        {"id": 1, "start": 13.88, "end": 20.36, "speaker": "S0",
         "text": "2013, אני אז מפקד אוגדה 36", "keep": False},
    ]


def test_stranded_opener_merges_into_its_continuation():
    segs = segments.merge_stranded_fragments(_orphan_pair())
    assert len(segs) == 1
    # Voiced once, whole, at the later timestamp: the absorber keeps its timing.
    assert segs[0]["text"] == "בשנת 2013, אני אז מפקד אוגדה 36"
    assert (segs[0]["start"], segs[0]["end"]) == (13.88, 20.36)
    # ...and remembers where the orphan came from, for the editor and for
    # unsegmented_words, which must still count its words as heard.
    assert segs[0]["merged_from"] == [{"start": 6.29, "end": 8.03, "text": "בשנת"}]


def test_merged_from_keeps_the_orphan_s_words_covered():
    segs = segments.merge_stranded_fragments(_orphan_pair())
    words = [{"t": 6.3, "text": "בשנת"}, {"t": 13.9, "text": "2013"}]
    assert segments.unsegmented_words(words, segs, []) == []


def test_normal_adjacent_segments_are_never_merged():
    # A short line before an ordinary pause is not a torn fragment: the gap is
    # below ORPHAN_GAP_MIN, so both segments survive untouched.
    segs = [
        {"id": 0, "start": 6.29, "end": 8.03, "speaker": "S0",
         "text": "בשנת", "keep": False},
        {"id": 1, "start": 8.9, "end": 15.0, "speaker": "S0",
         "text": "2013 קרה משהו", "keep": False},
    ]
    assert segments.merge_stranded_fragments(segs) == segs


def test_a_complete_short_sentence_is_not_a_fragment():
    # "כן." ends its own sentence; a scene change after it is a gap, not a tear.
    segs = _orphan_pair()
    segs[0]["text"] = "כן."
    assert len(segments.merge_stranded_fragments(segs)) == 2


def test_no_merge_across_a_speaker_change_or_into_a_keep():
    other = _orphan_pair()
    other[1]["speaker"] = "S1"
    assert len(segments.merge_stranded_fragments(other)) == 2
    kept = _orphan_pair()
    kept[1]["keep"] = True          # keep plays original audio; text added to it
    assert len(segments.merge_stranded_fragments(kept)) == 2   # would never be voiced


def test_a_wordy_or_long_or_distant_opener_stands_alone():
    wordy = _orphan_pair()
    wordy[0]["text"] = "בשנת ההיא של המלחמה"          # a clause, not a torn word
    assert len(segments.merge_stranded_fragments(wordy)) == 2
    slow = _orphan_pair()
    slow[0]["end"] = 8.9                              # 2.6s: a deliberate utterance
    assert len(segments.merge_stranded_fragments(slow)) == 2
    far = _orphan_pair()
    far[1]["start"] = 16.5                            # past ORPHAN_GAP_MAX
    assert len(segments.merge_stranded_fragments(far)) == 2
