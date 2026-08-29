"""Phase 5 translate-stage prompt improvements. Pure logic, no models.

A dialogue register mode, the rolling run-level established-names list, and the
military-Hebrew terms note. Since translate/v24 digit→word conversion is
deterministic code (dubbing/numwords.py, tested in test_phase5_numwords.py);
the model is only ever asked to keep already-spelled number-words as words.
"""

from __future__ import annotations

from dubbing import keep, manifest, translate

_NARRATION_EN = ("Write full words, no contractions (\"we are\" not "
                 "\"we're\", \"do not\" not \"don't\") so the text-to-speech "
                 "reads them clearly.")


# ------------------------------------------------------------- numbers as words

def test_instruction_never_asks_the_model_to_convert_digits():
    # v24: the v22 sentence ("…out in full words … never digits…") made the
    # model convert digits itself, unreliably ("504" → "five zero four"); the
    # default prompt now says nothing about writing numbers as words.
    for source, target in (("he", "ru"), ("he", "en"), ("en", "ru")):
        p = translate._translate_instruction("בשנת 1973", source, target)
        assert "out in full words" not in p
        assert "never digits" not in p
        assert "written out as words" not in p


def test_spelled_input_hop_asks_only_for_inflected_number_words():
    # The hop whose input already carries code-spelled English number-words
    # (pivot hop 2, or a direct en→tgt hop) gets exactly one light sentence.
    p = translate._translate_instruction("five hundred and four soldiers",
                                         "en", "ru", numbers_spelled=True)
    assert ("Numbers in the source are written out as words; render them as "
            "naturally inflected words in Russian, never as digits.") in p


def test_shorten_instruction_no_longer_asks_for_number_words(monkeypatch):
    prompts = []

    def fake_run(tok, mdl, user_text, max_new_tokens):
        prompts.append(user_text)
        return ""                                   # fails is_target_text → None

    monkeypatch.setattr(translate, "_run", fake_run)
    assert translate.shorten(None, None, "מאה חיילים", "one hundred soldiers went",
                             3, source="he", target="en") is None
    assert len(prompts) == 1
    assert "out in full words" not in prompts[0] and "never digits" not in prompts[0]
    assert "keeping every name, number and negation" in prompts[0]


# ------------------------------------------------------------------- register

def test_narration_english_keeps_the_exact_no_contractions_sentence():
    p = translate._translate_instruction("שלום", "he", "en")
    assert _NARRATION_EN in p
    p = translate._translate_instruction("שלום", "he", "en", register="narration")
    assert _NARRATION_EN in p


def test_dialogue_english_welcomes_contractions_and_drops_the_ban():
    p = translate._translate_instruction("שלום", "he", "en", register="dialogue")
    assert "contractions are welcome" in p
    assert "don't" in p and "we're" in p
    assert _NARRATION_EN not in p
    assert "no contractions" not in p


def test_dialogue_other_target_gets_conversational_register():
    p = translate._translate_instruction("שלום", "he", "ru", register="dialogue")
    assert "natural conversational Russian" in p
    assert "register of spoken dialogue" in p
    # No English-specific aside leaks into a Russian prompt (pinned by an
    # existing test for narration; hold it for dialogue too).
    assert "contractions" not in p and "we're" not in p


def test_narration_other_target_unchanged():
    p = translate._translate_instruction("שלום", "he", "ru")
    assert "Write full words suitable for text-to-speech." in p


# ------------------------------------------------------- established names list

def test_established_names_roll_most_recent_first_and_cap_at_12():
    names: list[str] = []
    for i in range(15):
        names = translate.update_established_names(
            names, f"Then Person{i:02d} spoke to the crowd.", "en")
    assert len(names) == translate.MAX_ESTABLISHED_NAMES == 12
    assert names[0] == "Person14"                    # most recent first
    assert "Person02" not in names                   # oldest three rolled off
    # A recurring name moves to the front instead of duplicating.
    names = translate.update_established_names(
        names, "And Person05 answered.", "en")
    assert names[0] == "Person05" and names.count("Person05") == 1
    assert len(names) == 12


def test_established_names_skip_sentence_initial_words_and_caseless_scripts():
    assert translate.update_established_names([], "The war began.", "en") == []
    assert translate.update_established_names([], "מלחמה גדולה", "he") == []


def test_names_note_appears_only_when_list_is_nonempty():
    bare = translate._translate_instruction("שלום", "he", "en")
    assert "already established" not in bare
    p = translate._translate_instruction("שלום", "he", "en",
                                         names=("Dayan", "Golda", "Sinai"))
    assert ("Names already established in this video's translation use these "
            "exact spellings when the same person or place recurs: "
            "Dayan, Golda, Sinai." in p)


def test_run_threads_names_and_register_into_generate(monkeypatch, tmp_path):
    calls = []

    def fake_generate(tok, mdl, text, *, source, target, context="", preceding="",
                      device=None, register="narration", names=(), **kw):
        calls.append({"target": target, "register": register, "names": names})
        return {"אחת": "General Elazar spoke.",
                "שתיים": "Then Elazar and Dayan left."}[text]

    monkeypatch.setattr(translate, "load", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(translate, "free", lambda mdl: None)
    monkeypatch.setattr(translate, "generate", fake_generate)
    m = manifest.new({"input": "x"})
    m["segments"] = [
        {"id": i, "start": float(i), "end": float(i) + 1.0, "speaker": "S0",
         "text": t, "keep": False, "keep_reason": None}
        for i, t in enumerate(["אחת", "שתיים"])
    ]
    translate.run(m, tmp_path, source="he", target="en", register="dialogue")

    assert [c["register"] for c in calls] == ["dialogue", "dialogue"]
    assert calls[0]["names"] == ()                   # nothing established yet
    assert calls[1]["names"] == ("Elazar",)          # from the first translation
    # After the run both names are established (most recent first).
    # (Dayan appears later in segment 2's translation than Elazar.)


# ---------------------------------------------------------- military-Hebrew note

def test_hebrew_source_gets_the_military_terms_note():
    p = translate._translate_instruction("המ\"פ נפצע", "he", "en")
    assert "מ\"פ company commander" in p
    assert "קמ\"ן intelligence officer" in p
    assert "חגורת נפץ explosive belt" in p
    assert "מודיעין means intelligence" in p
    assert "not the city Modi'in and never a person's name" in p


def test_non_hebrew_source_gets_no_military_note():
    for source in ("ru", "en", "ar"):
        p = translate._translate_instruction("text", source, "en")
        assert "company commander" not in p
        assert "מודיעין" not in p


def test_hebrew_military_note_carries_the_unit_echelons():
    p = translate._translate_instruction("המחלקה יצאה", "he", "en")
    assert "מחלקה platoon" in p
    assert "גדוד battalion" in p


# --------------------------------------------------------------------- tag bump

def test_translate_stage_tag_bumped():
    # v32: two parallel lines each claimed v31 (lock-aware reruns; the news
    # register work) the merged tag moves past both claims.
    # v33: a user keep is subtitled honestly under either of its two names.
    # v34: a same-language pair translates by identity, with no model loaded.
    # v35: a segment translates from its OWN language, and a failed translation
    # no longer overrules the user's "dub it".
    # v36: the gloss floors, `_has_negation` and `shorten`'s budget are all
    # script-derived, so zh/ja/ko sources and targets are judged in their own unit.
    # v37: "%" becomes the target's own spoken word for every TTS target —
    # ja/ko/zh percentages no longer vanish, it/pt no longer say "percent".
    # v39: a repetition the speaker actually made survives the "X, X" repair, so
    # a drama line keeps the beat the actor played.
    # v40: the fluency license reaches the hops INTO English as well as out of
    # them, and a bare honorific is ruled a title rather than a name. Both are
    # prompt text, so every translation this stage has ever written is stale.
    assert manifest.STAGE_TAGS["translate"] == "translate/v40"


# ------------------------------------------------------- per-segment gloss gating

# The measured regression's shape: one --context string mixing background (always
# wanted) with word glosses (wanted only where the word is actually spoken).
_CTX = ("Israeli documentary about Ahmed al-Sharaa and modern Syria; "
        "the word זיקית means chameleon (the color-changing lizard); "
        "the narrator's phrase בלאגן (sometimes mis-transcribed as מלאגן or מרגן) "
        "means a big mess or chaos.")


def test_gloss_clause_included_when_its_word_is_in_the_segment():
    out = translate.relevant_context(_CTX, "ראינו שם זיקית על העץ", "he")
    assert "chameleon" in out
    assert "Israeli documentary" in out              # background rides along
    assert "chaos" not in out                        # the other gloss is gated out


def test_gloss_clause_included_for_a_mis_transcription_within_one_edit():
    # ASR wrote בלגן (a dropped א edit distance 1 from בלאגן, not listed in
    # the clause itself); the gloss must still reach this segment's prompt.
    out = translate.relevant_context(_CTX, "היה שם בלגן שלם", "he")
    assert "big mess or chaos" in out
    assert "chameleon" not in out


def test_gloss_clause_excluded_when_its_word_is_absent():
    # The regression: clean Hebrew about a bourgeois neighborhood picked up the
    # בלאגן gloss and came back as "a neighborhood of chaos".
    out = translate.relevant_context(_CTX, "שכונה של בורגנים", "he")
    assert "chaos" not in out and "chameleon" not in out and "זיקית" not in out
    assert "Israeli documentary" in out


def test_background_only_context_passes_through_unchanged():
    bg = "Israeli documentary about Ahmed al-Sharaa; he became president of Syria."
    assert translate.relevant_context(bg, "שכונה של בורגנים", "he") == bg


def test_empty_context_stays_empty():
    assert translate.relevant_context("", "שלום", "he") == ""
    assert translate.relevant_context("   ", "שלום", "he") == ""


def test_run_gates_both_pivot_hops_against_the_source_text(monkeypatch, tmp_path):
    # Hop 2 translates the clean English intermediate, but its gloss gate must
    # read the ORIGINAL Hebrew segment that is how «зикит» reached Russian.
    calls = []

    def fake_generate(tok, mdl, text, *, source, target, context="", **kw):
        calls.append({"source": source, "target": target, "context": context})
        return "The elite lives there." if target == "en" else "Элита живёт там."

    monkeypatch.setattr(translate, "load", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(translate, "free", lambda mdl: None)
    monkeypatch.setattr(translate, "generate", fake_generate)
    m = manifest.new({"input": "x"})
    m["source"]["context"] = _CTX
    m["segments"] = [
        {"id": 0, "start": 0.0, "end": 1.0, "speaker": "S0",
         "text": "האליטה הסורית חיה שמה", "keep": False, "keep_reason": None},
        {"id": 1, "start": 1.0, "end": 2.0, "speaker": "S0",
         "text": "ראינו זיקית על העץ", "keep": False, "keep_reason": None},
    ]
    translate.run(m, tmp_path, source="he", target="ru")

    assert [c["target"] for c in calls] == ["en", "ru", "en", "ru"]
    # Segment 0 has neither gloss word: both its hops see background only.
    for c in calls[:2]:
        assert "chameleon" not in c["context"] and "chaos" not in c["context"]
        assert "Israeli documentary" in c["context"]
    # Segment 1 speaks זיקית: both hops including the en→ru one get the gloss.
    for c in calls[2:]:
        assert "chameleon" in c["context"]
        assert "chaos" not in c["context"]


def test_shorten_prompt_is_gloss_gated_too(monkeypatch):
    prompts = []

    def fake_run(tok, mdl, user_text, max_new_tokens):
        prompts.append(user_text)
        return ""                                   # fails is_target_text → None

    monkeypatch.setattr(translate, "_run", fake_run)
    translate.shorten(None, None, "שכונה של בורגנים", "a bourgeois neighborhood", 2,
                      source="he", target="en", context=_CTX)
    assert len(prompts) == 1
    assert "chameleon" not in prompts[0] and "chaos" not in prompts[0]
    assert "Israeli documentary" in prompts[0]


def test_latin_script_source_passes_context_through_whole():
    # Same-script gating has no signal; the note is injected as before (this is
    # also the pivot-shorten path, whose "source" is the English intermediate).
    assert translate.relevant_context(_CTX, "The elite lives there.", "en") == _CTX


def test_fluency_license_on_every_hop_that_touches_english():
    into_ru = translate._translate_instruction("Hello there.", "en", "ru")
    assert "idiomatic Russian phrasing" in into_ru
    assert "never coin" in into_ru.lower()
    # Into English too, which is where it was missing: a he→en hop was never
    # asked to prefer the plain verb, and wrote "One must know her in order to
    # know how Qatar operates" for a line three plain words would carry.
    into_en = translate._translate_instruction("שלום", "he", "en")
    assert "idiomatic English phrasing" in into_en
    assert "never coin" in into_en.lower()
    assert "idiomatic" in translate._translate_instruction("Привет.", "ru", "en")
    # A hop with English on neither side is untouched by any of this.
    assert "idiomatic" not in translate._translate_instruction("שלום", "he", "ru")


def test_garble_note_only_with_context_on_an_asr_hop():
    ctx = "Documentary about Ahmed al-Sharaa, known as al-Julani."
    with_ctx = translate._translate_instruction("שלום", "he", "en", context=ctx)
    assert "may mishear" in with_ctx
    # Bare instruction was probed and fixed nothing never emitted without context.
    assert "may mishear" not in translate._translate_instruction("שלום", "he", "en")
    # The pivot's second hop reads clean model English, not ASR.
    hop2 = translate._translate_instruction("hello", "en", "ru", context=ctx,
                                            numbers_spelled=True, asr_source=False)
    assert "may mishear" not in hop2


def test_run_disables_garble_note_on_the_second_pivot_hop(monkeypatch, tmp_path):
    calls = []

    def fake_generate(tok, mdl, text, *, source, target, **kw):
        calls.append((source, kw.get("asr_source", True)))
        return "ответ" if target == "ru" else "answer"

    monkeypatch.setattr(translate, "load", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(translate, "free", lambda mdl: None)
    monkeypatch.setattr(translate, "generate", fake_generate)
    m = manifest.new({"input": "x"})
    m["source"]["context"] = "Docu about al-Sharaa."
    m["segments"] = [{"id": 0, "start": 0.0, "end": 1.0, "speaker": "S0",
                      "text": "שלום לך", "keep": False, "keep_reason": None}]
    translate.run(m, tmp_path, source="he", target="ru")
    hop1 = [c for c in calls if c[0] == "he"]
    hop2 = [c for c in calls if c[0] == "en"]
    assert hop1 and all(asr for _, asr in hop1)
    assert hop2 and all(not asr for _, asr in hop2)


def test_trailing_clause_repeat_stripped():
    out = translate._strip_trailing_clause_repeat(
        "May God bless you and keep you, may the Lord turn His face toward you "
        "and be gracious to you, may the Lord turn His face toward you.")
    assert out == ("May God bless you and keep you, may the Lord turn His face "
                   "toward you and be gracious to you.")


def test_trailing_clause_repeat_leaves_legit_text_alone():
    for s in ("He came, he saw, he conquered.",
              "Never, never, never give up.",
              "The plan was simple. The plan was simple in name only.",
              "", "One clause only here."):
        assert translate._strip_trailing_clause_repeat(s) == s


# ----------------------------------------------------------------- revision pass

def _echo_numbered(prompt):
    import re as _re
    return "\n".join(f"{n}. {t}"
                     for n, t in _re.findall(r"(?m)^(\d+)\. (.*)$", prompt))


def test_revise_run_batches_with_overlap_context(monkeypatch):
    prompts = []

    def fake_run(tok, mdl, prompt, n):
        prompts.append(prompt)
        return _echo_numbered(prompt)

    monkeypatch.setattr(translate, "_run", fake_run)
    lines = [f"Spoken script line number here, take {chr(65 + i)}." for i in range(30)]
    out = translate.revise_run(None, None, lines, target="en")
    assert out == lines                              # echo → nothing changes
    assert len(prompts) == 2                         # 25 + 5, one call each
    assert "final context" not in prompts[0]
    assert "The first 2 lines are final context" in prompts[1]
    # Batch 2 lists the last two lines of batch 1 as context, then its own five.
    assert f"1. {lines[23]}" in prompts[1]
    assert f"2. {lines[24]}" in prompts[1]
    assert f"3. {lines[25]}" in prompts[1]
    assert f"7. {lines[29]}" in prompts[1]


def test_revise_run_ignores_revisions_to_the_overlap_context(monkeypatch):
    batch = [0]

    def fake_run(tok, mdl, prompt, n):
        batch[0] += 1
        import re as _re
        listed = _re.findall(r"(?m)^(\d+)\. ", prompt)
        return "\n".join(f"{k}. Rewritten in batch {'one two three'.split()[batch[0] - 1]} "
                         f"as line {'one two three four'.split()[int(k) - 1]}."
                         for k in listed)

    monkeypatch.setattr(translate, "_run", fake_run)
    lines = ["First spoken line here.", "Second spoken line here.",
             "Third spoken line here."]
    out = translate.revise_run(None, None, lines, target="en",
                               batch_size=2, overlap=1)
    # Batch 2 re-revised its context line (line 2), but only its own chunk
    # (line 3) is applied line 2 keeps batch 1's revision.
    assert out[1] == "Rewritten in batch one as line two."
    assert out[2] == "Rewritten in batch two as line two."


def test_revise_run_spells_digits_a_revision_introduced(monkeypatch):
    monkeypatch.setattr(translate, "_run",
                        lambda *a, **k: "1. It cost 500 dollars back then.")
    out = translate.revise_run(None, None, ["It cost much money back then."],
                               target="en")
    assert out == ["It cost five hundred dollars back then."]


def test_revise_run_survives_a_dead_model_call(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("worker died")

    monkeypatch.setattr(translate, "_run", boom)
    assert translate.revise_run(None, None, ["A line that stays."],
                                target="en") == ["A line that stays."]


def test_run_revises_the_finished_script_with_canonical_names(monkeypatch, tmp_path):
    seen = {}

    def fake_generate(tok, mdl, text, *, source, target, **kw):
        return {"אחת": "Then Jolani spoke first.",
                "שתיים": "Later Julani replied at length."}[text]

    def fake_revise(tok, mdl, lines, *, target, names=(), **kw):
        seen["lines"], seen["names"] = list(lines), tuple(names)
        return ["Then Jolani spoke first.", "Later Jolani replied at length."]

    monkeypatch.setattr(translate, "load", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(translate, "free", lambda mdl: None)
    monkeypatch.setattr(translate, "generate", fake_generate)
    monkeypatch.setattr(translate, "revise_run", fake_revise)
    m = manifest.new({"input": "x"})
    m["segments"] = [
        {"id": i, "start": float(i), "end": float(i) + 1.0, "speaker": "S0",
         "text": t, "keep": False, "keep_reason": None}
        for i, t in enumerate(["אחת", "שתיים"])
    ]
    translate.run(m, tmp_path, source="he", target="en")
    # The entity table is canonicalised from the finished lines themselves:
    # Jolani (1) vs Julani (1) tie broken by earliest → Jolani.
    assert seen["names"] == ("Jolani",)
    assert seen["lines"] == ["Then Jolani spoke first.",
                             "Later Julani replied at length."]
    assert m["segments"][1]["text_en"] == "Later Jolani replied at length."


# ------------------------------------------- a segment's own source language

# The bug this section pins: a he→de dub of a video whose speech is mostly
# ENGLISH (an already-dubbed source). The user flips one English line from
# "keep" to "Dub it"; the translator is asked for "the following HEBREW text" —
# the line is English Gemma hands the English straight back, `_echoes_source`
# correctly rejects the echo, `generate` gives up, and the segment is answered
# with a keep that undoes the user's flip and reports success. Observed on
# outputs/WhatsApp_Video_2026_08_10_at_14_07_11_2, segment 3.

def test_a_latin_line_in_a_hebrew_run_does_not_claim_to_be_hebrew():
    seg = {"text": "around this table, after the elections in October."}
    # Script refutes the run's claim, and refuting is all script can do: the
    # source is left unknown, never guessed at.
    assert translate.segment_langs(seg, "he", "de") == ("", "de")
    p = translate._translate_instruction(seg["text"], "", "de")
    assert "Translate the following text into clear, natural German" in p
    assert "Hebrew" not in p
    assert p.endswith(f"Text: {seg['text']}")


def test_a_hebrew_line_in_a_hebrew_run_still_says_hebrew():
    seg = {"text": "מסביב לשולחן הזה"}
    assert translate.segment_langs(seg, "he", "de") == ("he", "de")
    p = translate._translate_instruction(seg["text"], "he", "de")
    assert "Translate the following Hebrew text into clear, natural German" in p
    assert p.endswith(f"Hebrew: {seg['text']}")


def test_the_span_witness_and_the_editor_override_outrank_the_run():
    assert translate.segment_langs({"text": "hello there", "lang": "en"},
                                   "he", "de") == ("en", "de")
    # The user's override wins over the witness, and names the target too.
    assert translate.segment_langs({"text": "hello there", "lang": "en",
                                    "src_lang": "fr", "tgt_lang": "ru"},
                                   "he", "de") == ("fr", "ru")


def test_script_only_refutes_when_it_has_something_to_say():
    # Same-script pair: no signal, so the run's claim stands.
    assert translate.segment_langs({"text": "hola mundo"}, "en", "es") == ("en", "es")
    # Too few letters to judge a number, an interjection, a stray glyph.
    assert translate.segment_langs({"text": "330,000"}, "he", "de") == ("he", "de")
    assert translate.segment_langs({"text": "ok."}, "he", "de") == ("he", "de")


def test_an_unknown_source_goes_straight_to_the_target():
    # The pivot's first hop would have to name a language nothing knows.
    assert translate.pivot_via_english("", "de") is False
    assert translate.pivot_via_english("he", "de") is True
    assert translate.pivot_via_english("en", "de") is False
    # And the leak guard has no source script to hunt for.
    assert translate._leaks_source_script("Klartext", "", "he") is False


def test_run_translates_an_english_line_from_english_not_from_hebrew(monkeypatch,
                                                                    tmp_path):
    hops = []

    def fake_generate(tok, mdl, text, *, source, target, **kw):
        hops.append((text, source, target))
        return "Rund um diesen Tisch." if target == "de" else "english"

    monkeypatch.setattr(translate, "load", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(translate, "free", lambda mdl: None)
    monkeypatch.setattr(translate, "generate", fake_generate)
    monkeypatch.setattr(translate, "revise_run", lambda *a, **k: [])
    m = manifest.new({"input": "x", "src_lang": "he", "tgt_lang": "de"})
    m["segments"] = [
        {"id": 0, "start": 0.0, "end": 2.0, "speaker": "S0", "keep": False,
         "keep_reason": None, "text": "around this table, after the elections."},
        {"id": 1, "start": 2.0, "end": 4.0, "speaker": "S0", "keep": False,
         "keep_reason": None, "text": "גדי איזנקוט אמר את זה"},
    ]
    translate.run(m, tmp_path, source="he", target="de")
    # The English line goes straight to German, one hop, naming no source.
    assert hops[0] == ("around this table, after the elections.", "", "de")
    # The Hebrew one is untouched by the repair: he→de still pivots via English.
    assert [(s, t) for _, s, t in hops[1:]] == [("he", "en"), ("en", "de")]
    assert m["segments"][0]["text_en"] == "Rund um diesen Tisch."


def test_a_failed_translation_never_overrules_the_users_dub_it():
    # Both forms of the user's verdict count: `edit.set_keep(False)` writes the
    # lock and `passthrough` together, and either alone is still them speaking.
    for verdict in ({"locked": {"keep": True}}, {"passthrough": False}):
        seg = {"id": 3, "text": "around this table.", "keep": False,
               "text_en": "stale", "text_mid": "stale", **verdict}
        assert translate.mark_failed(seg) is False
        assert seg["keep"] is False
        assert "text_en" not in seg and "text_mid" not in seg
    # Unlocked, the pipeline's own "never silent" answer stands unchanged.
    plain = {"id": 4, "text": "around this table.", "keep": False}
    assert translate.mark_failed(plain) is True
    assert plain["keep"] is True and plain["keep_reason"] == "mt_failed"
    assert plain["text_en"] == plain["text"]
    # A locked keep=True is the user asking for the original audio: agreeing
    # with them is not overruling them.
    kept = {"id": 5, "text": "around this table.", "keep": True,
            "passthrough": True, "locked": {"keep": True}}
    assert translate.mark_failed(kept) is True


def test_run_leaves_a_locked_dub_untranslated_instead_of_asserting(monkeypatch,
                                                                  tmp_path):
    # The flip-flop this ends: keep=True beside passthrough=False is a manifest
    # that disagrees with itself, and the next run's `apply_passthrough` flips it
    # back and drops the whole tail of the run forever, never converging.
    monkeypatch.setattr(translate, "load", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(translate, "free", lambda mdl: None)
    monkeypatch.setattr(translate, "generate", lambda *a, **k: "")
    monkeypatch.setattr(translate, "revise_run", lambda *a, **k: [])
    m = manifest.new({"input": "x", "src_lang": "he", "tgt_lang": "de"})
    m["segments"] = [{"id": 0, "start": 0.0, "end": 2.0, "speaker": "S0",
                      "text": "מסביב לשולחן", "keep": False, "keep_reason": None,
                      "passthrough": False, "locked": {"keep": True}}]
    translate.run(m, tmp_path, source="he", target="de")   # must not assert
    seg = m["segments"][0]
    assert seg["keep"] is False and not seg.get("text_en")
    assert keep.apply_passthrough(m["segments"]) == []      # nothing to flip back


# ------------------------------------ gloss gating for the scripts Hebrew is not

def test_gloss_matches_a_two_character_japanese_word():
    # The floors were calibrated on Hebrew, where 4 letters is a short word. A
    # Japanese or Chinese word is one or two characters (首相 = prime minister),
    # so under the old floor no gloss in a zh/ja note ever reached a prompt and
    # the whole --context feature was dead for those sources.
    ctx = "NHK evening news; 首相 means the prime minister"
    hit = translate.relevant_context(ctx, "首相は今日会見した。", "ja")
    assert "prime minister" in hit and "NHK evening news" in hit
    miss = translate.relevant_context(ctx, "東京は雨です。", "ja")
    assert "prime minister" not in miss and "NHK evening news" in miss


def test_gloss_matches_a_chinese_word_inside_an_unspaced_sentence():
    ctx = "首相 means the prime minister"
    assert "prime minister" in translate.relevant_context(
        ctx, "首相今天举行了记者会", "zh")
    assert translate.relevant_context(ctx, "市场今天下跌了", "zh") == ""


def test_a_single_character_gloss_never_matches_loosely():
    # One Han character occurs inside unrelated compounds everywhere — exactly
    # the run-wide leak the Hebrew floors were raised to stop.
    ctx = "米 means the United States"
    assert translate.relevant_context(ctx, "新しい経済政策を発表した。", "ja") == ""


def test_gloss_matches_an_agglutinated_korean_word():
    # Korean glues its particles onto the noun: the gloss names 국무총리 and the
    # line says 국무총리는, so the exact-word rung could never fire.
    ctx = "국무총리 is the prime minister"
    assert "prime minister" in translate.relevant_context(
        ctx, "국무총리는 오늘 회견했다", "ko")


def test_a_korean_gloss_matches_by_prefix_not_by_free_substring():
    # Prefix keeps the precision the Hebrew floors bought: a shorter word that
    # merely occurs inside the spoken one is not this segment's gloss.
    ctx = "총리 is the prime minister"
    assert translate.relevant_context(ctx, "국무총리는 오늘 회견했다", "ko") == ""


def test_latin_sources_still_pass_the_whole_note_through():
    # The short-circuit for a Latin-script source is untouched: there is no
    # gloss/background signal to gate on, so the note is never trimmed.
    ctx = "Italian documentary; presidente means the president"
    for src, spoken in (("it", "Il presidente ha parlato"),
                        ("pt", "O presidente falou ontem")):
        assert translate.relevant_context(ctx, spoken, src) == ctx
        # ...including for a segment the gloss has nothing to do with.
        assert translate.relevant_context(ctx, "Buongiorno a tutti", src) == ctx


# ------------------------------------------------------------------- negations

def test_has_negation_reads_the_korean_negation_glued_inside_a_word():
    assert translate._has_negation("가지 않습니다", "ko")
    assert translate._has_negation("돈이 없어요", "ko")
    assert not translate._has_negation("여기로 갑니다", "ko")


def test_has_negation_regressions_for_the_languages_that_already_worked():
    assert translate._has_negation("我不知道", "zh")
    assert not translate._has_negation("我知道", "zh")
    assert translate._has_negation("知りません", "ja")
    assert not translate._has_negation("私は行きます", "ja")
    assert translate._has_negation("אני לא יודע", "he")
    assert not translate._has_negation("אני יודע", "he")
    assert translate._has_negation("I don't know", "en")
    assert not translate._has_negation("I know", "en")


# --------------------------------------------------- shorten's length budget

def test_shorten_measures_a_japanese_line_in_characters(monkeypatch):
    # `.split()` calls a whole Japanese line one word, so `have` was 1, every
    # rewrite failed `n >= have`, and the timeline's shorten rescue could never
    # succeed for a CJK target.
    long_line = "首相は今日の記者会見で新しい経済政策を発表しました。"
    short_line = "首相が新しい経済政策を発表した。"
    assert len(long_line.split()) == len(short_line.split()) == 1
    seen = {}

    def fake_run(p, m, instruction, n):
        seen["i"] = instruction
        return short_line

    monkeypatch.setattr(translate, "_run", fake_run)
    out = translate.shorten(None, None, "首相は…", long_line, 14,
                            source="ja", target="ja")
    assert out == short_line
    # ...and the budget is spoken in the unit it was measured in.
    assert "14 characters" in seen["i"]


def test_shorten_still_counts_words_for_a_latin_target(monkeypatch):
    seen = {}

    short_line = "Il presidente ha annunciato la nuova politica economica"

    def fake_run(p, m, instruction, n):
        seen["i"] = instruction
        return short_line

    monkeypatch.setattr(translate, "_run", fake_run)
    out = translate.shorten(None, None, "…",
                            "Il presidente della repubblica ha annunciato oggi la "
                            "nuova politica economica del governo", 8,
                            source="it", target="it")
    assert out == short_line
    assert "8 words" in seen["i"]


def test_shorten_refuses_a_rewrite_that_is_not_shorter_in_units(monkeypatch):
    line = "首相が新しい経済政策を発表した。"
    monkeypatch.setattr(translate, "_run", lambda *a, **k: line)
    assert translate.shorten(None, None, "首相は…", line, 8,
                             source="ja", target="ja") is None


def test_shorten_refuses_a_rewrite_that_romanises_a_name(monkeypatch):
    # The real he-ru failure: the full translation had transliterated the
    # disease into Cyrillic, the shortened rewrite put it back in Latin, and the
    # Russian voice had no pronunciation for it and dropped the word in silence.
    # `_proper_nouns` cannot see it — the name is sentence-initial, so it is
    # indistinguishable from ordinary capitalisation — so the script guard must.
    current = "Сквебиас — это тяжелое заболевание кожи, очень заразное для людей."
    rewrite = "Squebias — тяжелое кожное заболевание, очень заразное для людей."
    monkeypatch.setattr(translate, "_run", lambda *a, **k: rewrite)
    assert translate.shorten(None, None, "…", current, 8,
                             source="he", target="ru") is None


def test_shorten_allows_a_rewrite_that_keeps_the_name_in_the_lines_own_script(
        monkeypatch):
    current = "Сквебиас — это тяжелое заболевание кожи, очень заразное для людей."
    rewrite = "Сквебиас — тяжелое кожное заболевание, заразное для людей."
    monkeypatch.setattr(translate, "_run", lambda *a, **k: rewrite)
    assert translate.shorten(None, None, "…", current, 8,
                             source="he", target="ru") == rewrite


def test_shorten_still_allows_latin_where_the_line_already_had_it(monkeypatch):
    # A name the full translation itself left in Latin is not an introduction,
    # so the guard must not fire and cost the rescue a legitimate rewrite.
    current = "NASA отправляет Artemis к Луне в следующем году, как и планировалось."
    rewrite = "NASA отправляет Artemis к Луне в следующем году."
    monkeypatch.setattr(translate, "_run", lambda *a, **k: rewrite)
    assert translate.shorten(None, None, "…", current, 8,
                             source="en", target="ru") == rewrite


def test_scripts_used_names_only_the_scripts_present():
    from dubbing import script as script_mod
    assert script_mod.scripts_used("Привет") == {"cyrillic"}
    assert script_mod.scripts_used("NASA и Artemis") == {"latin", "cyrillic"}
    assert script_mod.scripts_used("שלום") == {"hebrew"}
    assert script_mod.scripts_used("123 — ...") == set()
