"""Deterministic digit→word spelling, so no model ever converts a number.

The v22 translate prompt asked the *model* to write numbers out as words, and the
model did it unreliably: "ב-504" came back from the English hop as "five zero
four", and the ru hop then fabricated "в две тысячи четвёртом году" the year
2004 out of it. Digit→word conversion is a mechanical job, so it is done here
in code (num2words); the models are only trusted to translate and inflect
number-words that already exist in their input.

Conversion happens on the **English layer**: the English hop translates digits as
digits, `spell_numbers(mid, "en")` turns them into correct English words, and the
target hop inflects those words. Source languages (Hebrew, Arabic, …) are never
spelled here. For non-English targets `spell_numbers(text, tgt)` is only a safety
net over digits the model passed through a nominative cardinal reads better in
TTS than a digit with a grammatical suffix ("504-м"), even when the case is
imperfect.

Deliberately conservative: times ("6:30"), ranges and hyphen compounds
("2010-2015", "mid-1990s"), and non-English decimals are left as digits rather
than risk a wrong reading; an unsupported num2words language (ja, zh) keeps its
digits — the TTS voice reads those natively — except that "%" still becomes the
language's spoken percent word, because `tts.prepare_text` strips the symbol.
"""

from __future__ import annotations

import re

from num2words import num2words

from . import script as _script
from .script import script_for


def _spaceless(ch: str) -> bool:
    """A CJK/kana/hangul character. Those scripts write without spaces, so a
    digit next to such a letter is ordinary prose — not a glued token like
    "1990s" or "A4"."""
    o = ord(ch)
    return any(lo <= o <= hi
               for bucket in ("cjk", "hangul")
               for lo, hi in _script._RANGES[bucket])

# The language's word for "%", appended after the spelled cardinal. Small table,
# English fallback general vocabulary, not per-video content. Covers every
# Qwen3-TTS target: a fallback English "percent" inside a non-Latin line is
# stripped by tts.prepare_text and the dub silently loses the word.
_PERCENT = {
    "en": "percent",
    "ru": "процентов",
    "es": "por ciento",
    "fr": "pour cent",
    "de": "Prozent",
    "it": "per cento",
    "pt": "por cento",
    "ja": "パーセント",
    "ko": "퍼센트",
}
# Chinese writes the percent word *before* the number: 20% is 百分之二十, never
# 二十百分之. num2words has no zh, so the number itself stays digits and the
# prefix lands on them (百分之20) — the voice reads digits natively.
_PERCENT_PREFIX = {"zh": "百分之"}


def _percent_words(number_words: str, lang: str) -> str:
    """`number_words` with the language's word for "%" attached correctly."""
    prefix = _PERCENT_PREFIX.get(lang)
    if prefix is not None:
        return prefix + number_words
    # No space in a script that has none between its words (二十パーセント).
    sep = "" if script_for(lang) in ("cjk", "hangul") else " "
    return number_words + sep + _PERCENT.get(lang, _PERCENT["en"])


# A number token: thousands-separated integer, decimal, or plain integer —
# optionally trailed by "%", a short attached non-Latin grammatical suffix
# ("504-м", "2004-го"; Latin is excluded so "10-day" keeps its word), or an
# English ordinal suffix ("4th"). Matched suffixes other than "%" are dropped.
_TOKEN = re.compile(
    r"(\d{1,3}(?:,\d{3})+|\d+\.\d+|\d+)"
    r"(\s?%|-[^\W\da-zA-Z_]{1,3}(?![\w'’-])|(?:st|nd|rd|th)\b)?"
)


def _convert(token: str, num: str, suffix: str, lang: str, spellable: bool) -> str:
    if not spellable:
        # num2words has no such language (ja, zh): the digits stay — the TTS
        # voice reads them natively — but "%" must still become a spoken word,
        # because tts.prepare_text strips the symbol from the synthesis text.
        if "%" in suffix:
            return _percent_words(num, lang)
        return token
    if "." in num:
        if lang != "en":
            return token  # non-English decimals stay digits rather than risk it
        words = num2words(float(num), lang="en")  # 3.5 → "three point five"
    else:
        n = int(num.replace(",", ""))
        words = None
        # A standalone 4-digit 1900–2099 number is read as a year.
        if "," not in num and len(num) == 4 and 1900 <= n <= 2099 and "%" not in suffix:
            try:
                words = num2words(n, to="year", lang=lang)  # "nineteen eighty-two"
            except (NotImplementedError, TypeError):
                words = None
        if words is None:
            words = num2words(n, lang=lang)
    if "%" in suffix:
        words = _percent_words(words, lang)
    # Any other matched suffix (grammatical "-м", ordinal "th") is dropped: the
    # nominative cardinal is the safe spoken form, and the target-hop model
    # inflects words it is never asked to convert digits.
    return words


def spell_numbers(text: str, lang: str = "en") -> str:
    """`text` with standalone digit numbers spelled out as `lang` words.

    Pure and deterministic. Leaves untouched: times and ranges (any digit
    adjacent to ":" or "-"), digits glued into a word ("1990s", "A4"),
    version-like dotted chains, non-English decimals, and everything when
    num2words does not support `lang`.
    """
    if not text or not re.search(r"\d", text):
        return text
    try:
        num2words(1, lang=lang)
        spellable = True
    except NotImplementedError:
        # No spelling for this language, but "%" still needs its spoken word
        # (tts.prepare_text strips the symbol); everything else stays as-is.
        if "%" not in text:
            return text
        spellable = False

    def repl(m: re.Match[str]) -> str:
        start, end = m.start(), m.end()
        before = text[start - 1] if start else ""
        after = text[end] if end < len(text) else ""
        if before in (":", "-") or after in (":", "-"):  # time, range, compound
            return m.group(0)
        if ((before.isalnum() and not _spaceless(before))
                or (after.isalpha() and not _spaceless(after))):  # glued: "1990s"
            return m.group(0)
        if before == "." and start >= 2 and text[start - 2].isdigit():
            return m.group(0)                          # tail of a 1.2.3 chain
        if after == "." and end + 1 < len(text) and text[end + 1].isdigit():
            return m.group(0)                          # head of a 1.2.3 chain
        return _convert(m.group(0), m.group(1), m.group(2) or "", lang, spellable)

    return _TOKEN.sub(repl, text)
