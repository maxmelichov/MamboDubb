"""Is this short beat an international interjection the actor should keep saying?

Movie mode only, and purely lexical: a fixed cross-lingual word list plus a
transliteration test, no audio and no per-video vocabulary (invariant 7). Split
out of `dubbing/segments.py`, which re-exports `is_interjection_keep` — this is
the one thing `mark_keep` needs from here.
"""

from __future__ import annotations

from typing import Any

from . import script

# Movie mode only: a standalone beat of at most this many words / seconds whose
# words are all international interjections (or source-script borrowings of a
# target interjection) keeps ORIGINAL audio — the actor's own "welcome!" beats
# any clone for a one-word line. Subtitles still show the translation.
INTERJECTION_MAX_WORDS = 2
INTERJECTION_MAX_SEC = 1.5

# Cross-lingual greeting particles and interjections, by surface form. These are
# INTERNATIONAL words — borrowed or shared across languages, recognisable in the
# original voice — not any one video's vocabulary. Hebrew-script entries cover the
# common loan spellings; Latin and Cyrillic entries the same words in those scripts.
_INTERJECTIONS = frozenset({
    # Hebrew-script loans / particles
    "אהלן", "יאללה", "הלו", "ביי", "אוקיי", "אוקי", "וואו", "או-קיי",
    "סורי", "האי", "היי", "וולקאם", "ולקאם", "צ'או", "באי",
    # Latin
    "hello", "hi", "hey", "bye", "goodbye", "okay", "ok", "wow", "welcome",
    "ahlan", "yalla", "ciao", "salut", "hallo", "sorry", "oops",
    # Cyrillic
    "привет", "пока", "чао", "окей", "ок", "алло", "хай", "вау",
})

# Latin skeletons of target-language interjection words for the borrowed-word
# test: a source-script token whose transliteration lands on one of these is a
# loanword the actor said in (near-)target pronunciation. Skeletons are the
# word lower-cased, c→k, vowels aeiou dropped, doubles collapsed.
_BORROWED_TARGET_WORDS = ("welcome", "okay", "hello", "goodbye", "bye",
                          "wow", "sorry", "hey")

# Hebrew consonant → Latin skeleton options. Matres lectionis (א ה ע י) drop;
# ו may read as the consonant w/v or as a vowel (drops). Small and general:
# it only ever has to recognise an international loanword, not Hebrew.
_HE_SKELETON: dict[str, tuple[str, ...]] = {
    "א": ("",), "ה": ("", "h"), "ע": ("",), "י": ("", "y"), "ו": ("", "w", "v"),
    "ב": ("b", "v"), "ג": ("g",), "ד": ("d",), "ז": ("z",), "ח": ("h", "kh"),
    "ט": ("t",), "כ": ("k", "kh"), "ך": ("k", "kh"), "ל": ("l",),
    "מ": ("m",), "ם": ("m",), "נ": ("n",), "ן": ("n",), "ס": ("s",),
    "פ": ("p", "f"), "ף": ("p", "f"), "צ": ("ts",), "ץ": ("ts",),
    "ק": ("k",), "ר": ("r",), "ש": ("sh", "s"), "ת": ("t",),
}


def _collapse(s: str) -> str:
    """Adjacent duplicate letters collapsed ("wwlkm" → "wlkm")."""
    out = []
    for ch in s:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


def _latin_skeleton(word: str) -> str:
    """Consonant skeleton of a Latin word: lowercase, c→k, vowels out, doubles collapsed."""
    s = (word or "").lower().replace("c", "k")
    return _collapse("".join(ch for ch in s if ch not in "aeiou"))


def _source_skeletons(token: str) -> set[str]:
    """Every plausible Latin consonant skeleton of a source-script token.

    Currently Hebrew only (the one caseless loan-heavy source measured); an
    unmapped script yields no skeletons, so the borrowed-word test simply never
    fires there and only the literal interjection list applies.
    """
    variants = {""}
    for ch in token:
        opts = _HE_SKELETON.get(ch)
        if opts is None:
            if not script.count_letters(ch, "hebrew") and not ch.isalpha():
                continue                   # punctuation/geresh inside the token
            return set()                   # a letter the map cannot read
        variants = {v + o for v in variants for o in opts}
        if len(variants) > 200:            # loanwords are short; never blow up
            return set()
    return {_collapse(v) for v in variants if v}


def _is_interjection_token(token: str, src: str, tgt: str) -> bool:
    """One word of a candidate interjection beat: listed, or a borrowed target word."""
    tok = token.strip().strip("\"'׳״.,!?…-–—()").lower()
    if not tok:
        return False
    if tok in _INTERJECTIONS:
        return True
    # Borrowed-word test: a source-script token (3+ letters — shorter ones are
    # ambiguous) whose transliteration skeleton is a target interjection word.
    if script.script_for(tgt) != "latin" or script.same_script(src, tgt):
        return False
    if script.count_letters(tok, script.script_for(src)) < 3:
        return False
    skels = _source_skeletons(tok)
    # Only skeletons of 3+ consonants discriminate ("wlkm"); shorter ones would
    # false-positive on native words (הלא → "hl" = hello) — those loans' common
    # spellings are in the literal list instead.
    return bool(skels) and any(sk in skels for w in _BORROWED_TARGET_WORDS
                               if len(sk := _latin_skeleton(w)) >= 3)


def is_interjection_keep(seg: dict[str, Any], src: str, tgt: str) -> bool:
    """True when a standalone segment is a greeting/interjection beat (movie mode).

    The segment must be short in both words and seconds — a one- or two-word
    beat, its own segment rather than part of a longer sentence — and EVERY
    word must be an international interjection or a source-script borrowing of
    a target one. Language-general by construction: a fixed cross-lingual list
    plus a script/transliteration test, never per-video vocabulary.
    """
    if seg["end"] - seg["start"] > INTERJECTION_MAX_SEC:
        return False
    words = (seg.get("text") or "").split()
    if not 1 <= len(words) <= INTERJECTION_MAX_WORDS:
        return False
    return all(_is_interjection_token(w, src, tgt) for w in words)
