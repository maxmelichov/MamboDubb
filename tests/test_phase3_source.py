"""Phase 3 any source language (pure logic, no models).

The source ASR is no longer hard-wired to the ivrit-ai Hebrew fine-tune: a
table picks the model per source, Hebrew (and its legacy "iw" spelling) keeping
the fine-tune and everything else reading with the vanilla multilingual turbo,
told the language at transcribe time. Caption fetching mirrors the same legacy
ISO-639 aliases YouTube still uses.
"""

from __future__ import annotations

from dubbing import fetch, transcript, translate


# ----------------------------------------------------------- source ASR selection

def test_hebrew_keeps_the_ivrit_finetune():
    local, hub = transcript.source_asr_paths("he")
    assert local == transcript.WHISPER_MODEL
    assert local.name == "whisper-large-v3-turbo-ct2"
    assert hub == "ivrit-ai/whisper-large-v3-turbo-ct2"


def test_legacy_iw_behaves_like_he():
    assert transcript.source_asr_paths("iw") == transcript.source_asr_paths("he")


def test_other_sources_use_the_vanilla_turbo():
    for src in ("ru", "en", "es", "ar", "id"):
        local, hub = transcript.source_asr_paths(src)
        assert local == transcript.SRC_ASR_MODEL, src
        assert local.name == "faster-whisper-large-v3-turbo-ct2", src
        assert hub == "deepdml/faster-whisper-large-v3-turbo-ct2", src


def test_selection_is_case_insensitive():
    assert transcript.source_asr_paths("HE") == transcript.source_asr_paths("he")
    assert transcript.source_asr_paths("RU") == transcript.source_asr_paths("ru")


# -------------------------------------------------------------- caption language prefs

def test_hebrew_prefs_unchanged():
    assert fetch._lang_prefs("he") == ("iw-orig", "he-orig", "iw", "he")


def test_yiddish_and_indonesian_include_legacy_codes():
    assert fetch._lang_prefs("yi") == ("ji-orig", "yi-orig", "ji", "yi")
    assert fetch._lang_prefs("id") == ("in-orig", "id-orig", "in", "id")


def test_unlisted_codes_are_passed_through():
    assert fetch._lang_prefs("ru") == ("ru-orig", "ru")
    assert fetch._lang_prefs("en") == ("en-orig", "en")


def test_fetch_aliases_mirror_transcript_lid_aliases():
    # transcript._LID_ALIAS maps legacy -> modern; fetch expands modern -> legacy.
    for legacy, modern in transcript._LID_ALIAS.items():
        assert legacy in fetch._lang_prefs(modern)


def test_echoes_source_catches_untranslated_same_script_output():
    echo = translate._echoes_source
    src = "The second is that it's put us in a place where we have no idea"
    # The en→es failure mode: the model hands the English source back.
    assert echo('Source:** "The second is that it\'s put us in a place"', src)
    # A real Spanish translation shares almost no full tokens with its source.
    assert not echo("Lo segundo es que nos ha puesto en un lugar donde no tenemos idea", src)
    # Cross-script pairs can never overlap on tokens.
    assert not echo("They will defend her.", "יגנו עליה כי היא חשובה להם")
    # Short sources are exempt three words legitimately survive intact.
    assert not echo("Good morning.", "Good morning.")


def test_clip_exceeds_slot_only_fires_on_runaway():
    from dubbing import tts
    # 71.5s for a 5.7s slot the en→es cascade is a rejection.
    assert tts.clip_exceeds_slot(71.5, 5.7)
    # Ordinary expansion (a 9s clip in a 6s slot) is placement's business.
    assert not tts.clip_exceeds_slot(9.0, 6.0)
    # Short segments get the flat 8s allowance.
    assert not tts.clip_exceeds_slot(8.2, 0.5)
    assert tts.clip_exceeds_slot(10.0, 0.5)


def test_not_a_translation_catches_reasoning_dumps():
    bad = translate._not_a_translation
    src = "The second is that it's put us in a place where we have no idea"
    # Markdown working notes (the real en→es failure, abridged).
    assert bad('Source:** "The second…"\n- **Translation Strategy:** nos ha puesto', src)
    # Length blow-up without markdown.
    assert bad("palabra " * 80, src)
    # A real translation passes.
    assert not bad("Lo segundo es que nos ha puesto en una situación sin idea", src)
    # Short sources are never length-judged.
    assert not bad("Buenos días a todos ustedes aquí presentes hoy.", "Good morning.")


def test_translate_reset_reclaims_downstream_failure_keeps():
    from dubbing import manifest
    # A segment kept by a *downstream* tts failure still holds the translation
    # that failed; a translate reset must re-translate it, not skip it as "kept".
    m = {"segments": [
        {"id": 0, "keep": True, "keep_reason": "tts_failed", "text_en": "bad dump"},
        {"id": 1, "keep": True, "keep_reason": "latin", "text_en": "subtitle"},
        {"id": 2, "keep": False, "keep_reason": None, "text_en": "fine"},
    ]}
    manifest.reset_stage(m, "translate")
    s0, s1, s2 = m["segments"]
    assert not s0["keep"] and "text_en" not in s0
    assert s1["keep"] and s1["text_en"] == "subtitle"   # structural keeps untouched
    assert "text_en" not in s2


def test_foreign_span_language_flows_to_spliced_segments():
    from dubbing import segments as seg_mod
    span = {"start": 10.0, "end": 14.0, "lang": "ar",
            "words": [{"t": 10.2, "text": "مصلحة"}, {"t": 12.0, "text": "الجماعة"}]}
    out = seg_mod.splice_foreign_spans([], [span], words=[], src="he", tgt="en")
    assert out and all(x.get("lang") == "ar" for x in out)


def test_translate_reset_clears_foreign_keep_subtitles():
    from dubbing import manifest
    m = {"segments": [
        {"id": 0, "keep": True, "keep_reason": "foreign", "lang": "ar",
         "text": "مصلحة الجماعة", "text_en": "stale translation"},
        {"id": 1, "keep": True, "keep_reason": "latin", "text_en": "kept subtitle"},
    ]}
    manifest.reset_stage(m, "translate")
    assert "text_en" not in m["segments"][0]      # re-translated next run
    assert m["segments"][1]["text_en"] == "kept subtitle"


def test_lang_is_a_whitelisted_segment_key():
    from dubbing import manifest
    assert "lang" in manifest.SEGMENT_KEYS


def test_drop_stretched_words_removes_nested_artifact():
    # The real 61-64s artifact: "סתם" stretched over 3.1s with invented filler
    # words inside its span, then the real words re-read at the right time.
    w = [
        {"t": 61.17, "end": 61.71, "text": "לא"},
        {"t": 61.71, "end": 64.79, "text": "סתם"},          # stretched
        {"t": 62.83, "end": 63.17, "text": "עדיין"},         # nested inventions
        {"t": 63.17, "end": 63.47, "text": "קפל"},
        {"t": 64.51, "end": 64.57, "text": "לא"},
        {"t": 64.57, "end": 64.83, "text": "סתם..."},
        {"t": 64.79, "end": 65.37, "text": "אמריקאים"},
    ]
    out = transcript.drop_stretched_words(w)
    texts = [x["text"] for x in out]
    assert "עדיין" not in texts and "קפל" not in texts
    # The seamlessly attached leading "לא" is the same decode event and goes too;
    # the genuine re-read at the span's end survives.
    assert texts == ["לא", "סתם...", "אמריקאים"] and out[0]["t"] == 64.51
    # A long word with nothing nested inside survives.
    ok = [{"t": 0.0, "end": 3.0, "text": "loooong"}, {"t": 3.1, "end": 3.4, "text": "next"}]
    assert transcript.drop_stretched_words(ok) == ok


def test_drop_stock_phrases_removes_whisper_credit_hallucinations():
    drop = transcript.drop_stock_phrases
    # The Arzamas jingle case: an invented subtitler credit over music.
    w = [{"t": 0.3, "end": 0.8, "text": "Субтитры"},
         {"t": 0.8, "end": 1.3, "text": "сделал"},
         {"t": 1.3, "end": 1.9, "text": "DimaTorzok"},
         {"t": 8.4, "end": 9.5, "text": "Когда"}]
    out = drop(w)
    assert [x["text"] for x in out] == ["Когда"]
    # Legitimate dialogue containing "subscribe" alone is untouched.
    ok = [{"t": 0.0, "end": 0.4, "text": "people"},
          {"t": 0.4, "end": 0.8, "text": "subscribe"},
          {"t": 0.8, "end": 1.2, "text": "to"},
          {"t": 1.2, "end": 1.6, "text": "newspapers"}]
    assert drop(ok) == ok


def test_lone_word_stub_merges_across_a_dramatic_pause():
    from dubbing import segments as seg_mod
    # "קשרות" sat alone for 0.56s between 0.84s and 1.2s pauses and was dubbed
    # as a stray "Relations"; one word may cross a wider gap to rejoin its line.
    words = [
        {"t": 151.9, "end": 152.4, "text": "מורכבות", "brk": False, "spk": "A"},
        {"t": 152.5, "end": 153.0, "text": "של", "brk": False, "spk": "A"},
        {"t": 153.1, "end": 154.8, "text": "האיש.", "brk": False, "spk": "A"},
        {"t": 155.7, "end": 156.2, "text": "קשרות", "brk": False, "spk": "A"},
        {"t": 157.4, "end": 158.0, "text": "וגמישויות", "brk": False, "spk": "A"},
        {"t": 158.1, "end": 158.7, "text": "יוצאות", "brk": False, "spk": "A"},
        {"t": 158.8, "end": 159.4, "text": "מן", "brk": False, "spk": "A"},
    ]
    segs = seg_mod.words_to_segments(words, "he", "en")
    assert all(len(s["text"].split()) > 1 for s in segs), \
        [s["text"] for s in segs]


def test_echoed_word_dropped_but_real_repetition_kept():
    drop = transcript.drop_echo_words
    # The real 7:25 artifact: identical adjacent word overlapping its twin.
    echo = [{"t": 445.53, "end": 446.65, "text": "ומסוכנים."},
            {"t": 446.55, "end": 447.51, "text": "ומסוכנים."}]
    out = drop(echo)
    assert len(out) == 1 and out[0]["end"] == 447.51
    # Rhetorical repetition has a pause and survives.
    real = [{"t": 10.0, "end": 10.6, "text": "מסוכנים."},
            {"t": 11.1, "end": 11.7, "text": "מסוכנים."}]
    assert drop(real) == real
