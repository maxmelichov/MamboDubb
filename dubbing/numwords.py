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
than risk a wrong reading; an unsupported num2words language returns the text
unchanged.
"""

from __future__ import annotations

import re

from num2words import num2words

# The language's word for "%", appended after the spelled cardinal. Small table,
# English fallback general vocabulary, not per-video content.
_PERCENT = {
    "en": "percent",
    "ru": "процентов",
    "es": "por ciento",
    "fr": "pour cent",
    "de": "Prozent",
}

# A number token: thousands-separated integer, decimal, or plain integer —
# optionally trailed by "%", a short attached non-Latin grammatical suffix
# ("504-м", "2004-го"; Latin is excluded so "10-day" keeps its word), or an
# English ordinal suffix ("4th"). Matched suffixes other than "%" are dropped.
_TOKEN = re.compile(
    r"(\d{1,3}(?:,\d{3})+|\d+\.\d+|\d+)"
    r"(\s?%|-[^\W\da-zA-Z_]{1,3}(?![\w'’-])|(?:st|nd|rd|th)\b)?"
)


def _convert(token: str, num: str, suffix: str, lang: str) -> str:
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
        words += " " + _PERCENT.get(lang, _PERCENT["en"])
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
    except NotImplementedError:
        return text

    def repl(m: re.Match[str]) -> str:
        start, end = m.start(), m.end()
        before = text[start - 1] if start else ""
        after = text[end] if end < len(text) else ""
        if before in (":", "-") or after in (":", "-"):  # time, range, compound
            return m.group(0)
        if before.isalnum() or after.isalpha():        # glued into a word
            return m.group(0)
        if before == "." and start >= 2 and text[start - 2].isdigit():
            return m.group(0)                          # tail of a 1.2.3 chain
        if after == "." and end + 1 < len(text) and text[end + 1].isdigit():
            return m.group(0)                          # head of a 1.2.3 chain
        return _convert(m.group(0), m.group(1), m.group(2) or "", lang)

    return _TOKEN.sub(repl, text)
