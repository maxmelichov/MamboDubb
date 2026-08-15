"""The bug-hunt campaign's regression ledger every measured lesson, pinned.

Each section pins a policy that was settled by measurement, not taste: two
independent A/B harnesses (33-segment he→ru set, 46-segment he→en drama set,
exact prompt reproduction, greedy decoding) plus full-run audits. The headline
numbers the pins protect:

- Context glosses earned their keep (drama harness: 14 fixes / 0 breaks came
  from context glosses) while template-level linguistics rules measured 0 fixes
  on target segments and regressed controls hence the prompt-policy pins.
- An ENGLISH preceding line reconciles garbled names and dangling pronouns; a
  source-language one does not, and once caused the documented entity swap.
- The established-names list self-poisons without junk filters and variant
  canonicalisation (drama run: "I'm"/"His"/"God"; he→ru run: «Его»/«Это»).

Everything here is pure logic model calls are monkeypatched. Related pins
that predate the campaign live in test_phase5_translate.py and
test_phase4_pivot.py; this file references rather than duplicates them.
"""

from __future__ import annotations

from dubbing import manifest, translate

# =============================================================================
# 1. Prompt policy what earns a place in the template, and what does not.
#
# Also pinned elsewhere: the garble note requires non-empty context AND an
# asr_source hop (test_phase5_translate.py::test_garble_note_only_with_context_
# on_an_asr_hop measured: the bare instruction fixes nothing).
# =============================================================================


def test_collocation_and_echelon_sentences_only_on_en_to_non_en_hops():
    """Both ride the fluency license's guard (source=="en" and target!="en").

    A/B'd: the anti-calque sentence added +2 fixes in combination and the
    echelon rule repaired дивизион→дивизия, each with 0 semantic breaks on the
    controls. Neither belongs on a hop that reads the noisy source.
    """
    hop2 = translate._translate_instruction("hello", "en", "ru",
                                            numbers_spelled=True, asr_source=False)
    assert "standard collocation" in hop2 and "copula-heavy" in hop2
    assert "Military unit types translate by echelon" in hop2
    assert "the Russian term for the same echelon" in hop2
    for source, target in (("he", "en"), ("he", "ru"), ("ru", "en")):
        p = translate._translate_instruction("text", source, target)
        assert "standard collocation" not in p
        assert "echelon" not in p


def test_no_dative_or_idiom_instruction_in_any_prompt():
    """Guard against reintroducing the removed faithfulness block.

    Both harnesses measured the dative/idiom instruction inert on its target
    segments (outputs byte-identical) and harmful on controls, so it was
    removed; the class is covered elsewhere (glosses, preceding line). Pin its
    distinctive phrases absent from every prompt shape.
    """
    prompts = [
        translate._translate_instruction("שלום", "he", "en"),
        translate._translate_instruction("שלום", "he", "ru", register="dialogue"),
        translate._translate_instruction("שלום", "he", "en",
                                         context="Docu about al-Sharaa."),
        translate._translate_instruction("hello", "en", "ru",
                                         numbers_spelled=True, asr_source=False),
        translate._translate_instruction("Привет", "ru", "en"),
    ]
    for p in prompts:
        assert "dative or possessive pronoun" not in p
        assert "changes who acts and who receives" not in p


# =============================================================================
# 2. Preceding line shown in the language the hop WRITES.
#
# The full chain (hop 1 previous English intermediate, hop 2 previous
# text_mid) is pinned on real call flow in test_phase4_pivot.py::
# test_run_pivots_he_ru_in_two_hops_and_stores_the_intermediate and in the
# wiring pin at the bottom of this file.
# =============================================================================


def test_preceding_falls_back_to_source_when_previous_hop_failed(
        monkeypatch, tmp_path):
    """Measured: garbled-name reconciliation fires with an English preceding
    line and not with a Hebrew one but when no English exists yet (first
    segment, failed neighbour) the source text is still better than nothing on
    hop 1, and hop 2 gets "" rather than a wrong-language line. The historic
    "Jabhat al-Nusra" → "al-Qaeda's Front" entity swap happened with a HEBREW
    preceding line on hop 2; a coherent English one measured safe (0 breaks on
    17 controls including every entity-swap guard line)."""
    calls = []

    def fake_generate(tok, mdl, text, *, source, target, preceding="", **kw):
        calls.append({"target": target, "preceding": preceding, "text": text})
        if text == "שורה אחת":
            return "עדיין עברית"                 # hop 1 fails → no English mid
        return "english line" if target == "en" else "русская строка"

    monkeypatch.setattr(translate, "load", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(translate, "free", lambda mdl: None)
    monkeypatch.setattr(translate, "generate", fake_generate)
    monkeypatch.setattr(translate, "revise_run",
                        lambda tok, mdl, lines, **kw: list(lines))
    m = manifest.new({"input": "x"})
    m["segments"] = [
        {"id": i, "start": float(i), "end": float(i) + 1.0, "speaker": "S0",
         "text": t, "keep": False, "keep_reason": None}
        for i, t in enumerate(["שורה אחת", "שורה שתיים"])
    ]
    translate.run(m, tmp_path, source="he", target="ru")
    # Segment 0's English hop failed, so segment 1's first hop has no previous
    # English intermediate it falls back to the previous SOURCE text; its
    # second hop passes "" rather than mislead with a wrong-language line.
    seg1_hop1 = [c for c in calls if c["text"] == "שורה שתיים"][0]
    assert seg1_hop1["preceding"] == "שורה אחת"
    seg1_hop2 = [c for c in calls if c["target"] == "ru"][-1]
    assert seg1_hop2["preceding"] == ""


def test_subtitle_only_foreign_keeps_get_no_preceding(monkeypatch, tmp_path):
    """The subs loop translates isolated third-language spans; their neighbours
    spoke a different language, so no preceding line carries signal there —
    unchanged by the preceding-policy switch."""
    calls = []

    def fake_generate(tok, mdl, text, *, source, target, preceding="", **kw):
        calls.append(preceding)
        return "english aside" if target == "en" else "русская вставка"

    monkeypatch.setattr(translate, "load", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(translate, "free", lambda mdl: None)
    monkeypatch.setattr(translate, "generate", fake_generate)
    m = manifest.new({"input": "x"})
    m["segments"] = [{"id": 0, "start": 0.0, "end": 1.0, "speaker": "S0",
                      "text": "كلام عربي", "keep": True, "keep_reason": "foreign",
                      "lang": "ar"}]
    translate.run(m, tmp_path, source="he", target="ru")
    assert calls and all(p == "" for p in calls)


# =============================================================================
# 3. Gloss gating length-gated matching, because both fuzz rules leaked.
#
# The positive 5+-letter case (בלאגן matches the ASR variant בלגן at edit
# distance 1) is pinned in test_phase5_translate.py::
# test_gloss_clause_included_for_a_mis_transcription_within_one_edit.
# =============================================================================


def test_three_letter_gloss_token_attaches_to_no_segment():
    """Drama harness leak trap (a): a clause glossing a token under 4 letters
    matched by substring nearly everywhere and leaked run-wide (a מ"פ-bearing
    clause reached unrelated segments). Tokens that short never participate —
    even an exact hit is indistinguishable from noise."""
    ctx = "Docu about Syria; the word גדל means grew."
    out = translate.relevant_context(ctx, "הוא גדל בעיר", "he")
    assert "grew" not in out
    assert "Docu about Syria" in out


def test_four_letter_gloss_token_matches_by_exact_word_only():
    """Drama harness leak trap (b): 4-letter common words (אותו) leaked into
    unrelated segments via the edit-distance rule and perturbed them. Exact
    word match still works; fuzz needs 5+ letters."""
    ctx = "Docu about Syria; the word אותו means him."
    hit = translate.relevant_context(ctx, "ראיתי אותו שם", "he")
    assert "means him" in hit
    miss = translate.relevant_context(ctx, "ראיתי אותם שם", "he")   # 1 edit away
    assert "means him" not in miss
    assert "Docu about Syria" in miss


# =============================================================================
# 4. Entity table the established-names list must not self-poison.
# =============================================================================


def test_canonical_names_picks_best_attested_and_maps_variants():
    """"Jolani" attested three times, "Julani" twice: every variant of the
    group maps to the best-attested form, output ordered by first appearance."""
    attested = ["Julani", "Jolani", "Jolani", "Nusra", "Jolani", "Julani"]
    assert translate.canonical_names(attested) == ["Jolani", "Nusra"]


def test_canonical_names_breaks_frequency_ties_by_earliest():
    assert translate.canonical_names(["Julani", "Nusra", "Jolani"]) == \
        ["Julani", "Nusra"]


def test_canonical_names_once_seen_hallucination_never_beats_attested():
    """Drama run: one bad translation coined a name ("Shula" for "Shlomo"),
    which re-entered the rolling list and overwrote the correct name in later
    segments. A once-seen variant must never become canonical over a
    twice-attested one, whichever appeared first."""
    assert translate.canonical_names(["Shlomi", "Shlomo", "Shlomo"]) == ["Shlomo"]


def test_canonical_names_works_on_cyrillic_and_multiword_names():
    assert translate.canonical_names(
        ["Джулани", "Джолани", "Джолани"]) == ["Джолани"]
    assert translate.canonical_names(
        ["Jabhat al-Nusra", "Jablat al-Nusra", "Jabhat al-Nusra"]) == \
        ["Jabhat al-Nusra"]


def test_canonical_names_never_merges_short_or_distinct_names():
    """One edit on a four-letter name is a different word, not a misspelling."""
    assert translate.canonical_names(["Iran", "Iraq"]) == ["Iran", "Iraq"]
    assert translate.canonical_names(["Jolani", "Golan"]) == ["Jolani", "Golan"]
    # Exact case-insensitive repeats collapse to the most frequent casing.
    assert translate.canonical_names(["HAMAS", "Hamas", "Hamas"]) == ["Hamas"]


def test_name_occurrences_keeps_repeats_in_order():
    assert translate._name_occurrences(
        "Then Elazar met Dayan, and Elazar left.", "en") == \
        ["Elazar", "Dayan", "Elazar"]


def test_english_contraction_and_pronoun_capitals_never_enter_the_table():
    """Drama-run poisoning: "I'm", "His", "God" are capitalised constantly and
    crowded real names out of the rolling window. None of them are names."""
    assert translate._name_occurrences(
        "Then I'm sure His plan pleased God, so Shlomo agreed.", "en") == \
        ["Shlomo"]
    assert translate.update_established_names(
        [], "Then I'm sure His plan pleased God.", "en") == []
    assert translate.update_established_names(
        ["Shlomo"], "So He told them, Okay, sure.", "en") == ["Shlomo"]


def test_cyrillic_pronoun_capitals_never_enter_the_table():
    """he→ru run: «Его», «Это» accumulated as junk names. Pronouns only —
    capitalised ordinary adjectives («Свободной») cannot be stoplisted without
    a vocabulary project; frequency canonicalisation absorbs those."""
    assert translate._name_occurrences(
        "Потом Его слова услышала Москва, и Это всё изменило.", "ru") == \
        ["Москва"]
    assert translate.update_established_names(
        [], "Потом Его выбор увидели все.", "ru") == []


# =============================================================================
# 5. Output guards what leaves the stage is target text, once, complete.
#
# Trailing duplicated ≥4-word clause stripped / legit repetition kept is
# pinned in test_phase5_translate.py::test_trailing_clause_repeat_stripped
# and ::test_trailing_clause_repeat_leaves_legit_text_alone.
# =============================================================================


def test_generate_retries_once_on_a_source_script_leak(monkeypatch):
    """A leaked source-script word survives is_target_text (majority rule) and
    TTS reads it aloud mid-dub; the leak guard retries once with an explicit
    instruction, then falls back to keeping the original audio."""
    replies = iter(["The leader הרמטכ\"ל spoke to the troops.",
                    "The leader spoke to the troops."])
    prompts = []

    def fake_run(tok, mdl, prompt, n):
        prompts.append(prompt)
        return next(replies)

    monkeypatch.setattr(translate, "_run", fake_run)
    out = translate.generate(None, None, "המנהיג דיבר", source="he", target="en")
    assert out == "The leader spoke to the troops."
    assert len(prompts) == 2
    assert "not a single Hebrew letter" in prompts[1]


def test_generate_gives_up_when_the_leak_persists(monkeypatch):
    monkeypatch.setattr(translate, "_run",
                        lambda *a, **k: "The leader הרמטכ\"ל spoke.")
    assert translate.generate(None, None, "המנהיג דיבר",
                              source="he", target="en") == ""


def test_same_script_pairs_never_flag_a_leak():
    assert not translate._leaks_source_script("anything at all", "en", "es")
    assert translate._leaks_source_script("mostly english עברית here", "he", "en")


def test_revise_run_applies_fixes_and_rejects_script_relapses(monkeypatch):
    """The revision pass may fix spelling drift against the canonical names,
    but a revised line that fails is_target_text (e.g. a source-script
    relapse) keeps its original revision must never lose a translation."""
    lines = ["The Jolani faction spoke.",
             "Then Julani answered them.",
             "The talks collapsed entirely."]
    reply = ("1. The Jolani faction spoke.\n"
             "2. Then Jolani answered them.\n"          # canonical spelling fix
             "3. השיחות קרסו לגמרי")                     # source-script relapse
    monkeypatch.setattr(translate, "_run", lambda *a, **k: reply)
    out = translate.revise_run(None, None, lines, target="en", names=("Jolani",))
    assert out[0] == lines[0]
    assert out[1] == "Then Jolani answered them."
    assert out[2] == lines[2]                        # fails is_target_text → kept


def test_revise_run_keeps_the_batch_on_parse_failure(monkeypatch):
    lines = ["Line alpha stays put.", "Line beta stays put."]
    for garbage in ("Sure! Here are my thoughts on the script.",
                    "1. Only one line came back."):      # line-count mismatch
        monkeypatch.setattr(translate, "_run", lambda *a, **k: garbage)
        assert translate.revise_run(None, None, lines, target="en") == lines


# =============================================================================
# 6. Wiring one run, every per-call contract at once.
# =============================================================================

_CTX = "Docu about Ahmed al-Sharaa; the word זיקית means chameleon."
_T = ["ראינו זיקית על העץ", "שכונה של בורגנים", "הצבא נכנס לעיר"]
_EN = {_T[0]: "Then Sharaa saw a chameleon.",
       _T[1]: "A bourgeois neighborhood.",
       _T[2]: "Then the army entered."}
_RU = {"Then Sharaa saw a chameleon.": "Потом Шараа увидел хамелеона.",
       "A bourgeois neighborhood.": "Буржуазный район города.",
       "Then the army entered.": "Потом армия вошла в город."}


def test_run_call_contract_over_three_segments(monkeypatch, tmp_path):
    """The complete per-call contract on a he→ru pivot run:

    - gloss gating reads the ORIGINAL source text on BOTH hops (the chameleon
      gloss reaches only segment 0's calls; background reaches every call);
    - asr_source is True exactly on the hop that reads the noisy source;
    - preceding is the previous segment's English intermediate on both hops
      ("" for the first segment);
    - names are the canonicalised established list of the hop's own language.
    """
    calls = []

    def fake_generate(tok, mdl, text, *, source, target, context="",
                      preceding="", names=(), **kw):
        calls.append({"source": source, "target": target, "context": context,
                      "preceding": preceding, "names": tuple(names),
                      "asr": kw.get("asr_source", True)})
        return _EN[text] if target == "en" else _RU[text]

    monkeypatch.setattr(translate, "load", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(translate, "free", lambda mdl: None)
    monkeypatch.setattr(translate, "generate", fake_generate)
    monkeypatch.setattr(translate, "revise_run",
                        lambda tok, mdl, lines, **kw: list(lines))
    m = manifest.new({"input": "x"})
    m["source"]["context"] = _CTX
    m["segments"] = [
        {"id": i, "start": float(i), "end": float(i) + 1.0, "speaker": "S0",
         "text": t, "keep": False, "keep_reason": None}
        for i, t in enumerate(_T)
    ]
    translate.run(m, tmp_path, source="he", target="ru")

    assert [(c["source"], c["target"]) for c in calls] == \
        [("he", "en"), ("en", "ru")] * 3
    # asr_source: True on the source-reading hop, False on the English hop.
    assert [c["asr"] for c in calls] == [True, False] * 3
    # Gloss gating against the ORIGINAL Hebrew, both hops of each segment.
    for c in calls[:2]:
        assert "chameleon" in c["context"]
    for c in calls[2:]:
        assert "chameleon" not in c["context"]
    for c in calls:
        assert "Ahmed al-Sharaa" in c["context"]
    # Preceding: previous segment's English intermediate, on BOTH hops.
    assert calls[0]["preceding"] == "" and calls[1]["preceding"] == ""
    assert calls[2]["preceding"] == _EN[_T[0]]
    assert calls[3]["preceding"] == _EN[_T[0]]
    assert calls[4]["preceding"] == _EN[_T[1]]
    assert calls[5]["preceding"] == _EN[_T[1]]
    # Names: each hop gets its own language's canonicalised established list.
    assert calls[0]["names"] == () and calls[1]["names"] == ()
    assert calls[2]["names"] == ("Sharaa",)
    assert calls[3]["names"] == ("Шараа",)
    assert calls[4]["names"] == ("Sharaa",)
    assert calls[5]["names"] == ("Шараа",)
    # And the pivot stored what it should.
    assert m["segments"][0]["text_mid"] == _EN[_T[0]]
    assert m["segments"][0]["text_en"] == _RU[_EN[_T[0]]]
