"""Which writing system a language uses, and whether text is written in it.

The pipeline's original form hardcoded a two-script world Latin meant "target
(English)" and the Hebrew block meant "source" in half a dozen places. This
module is the single replacement: every "is this text in the target language's
script?" question goes through `is_script` / `script_ratio` with an explicit
language code, so a Russian or Arabic target works the same way English did.

Script detection is a *cheap, high-precision* signal and only that. It separates
languages written differently (he/en, en/ru, he/ar); for same-script pairs
(en→es, ru→uk) it cannot discriminate and callers must fall back to LID —
`same_script()` tells them when. CJK gets one shared bucket: Japanese mixes kana
with Han and the han bucket alone cannot separate zh from ja.

The other thing a script decides is how text is *shaped*: where a sentence ends
(`CJK_SENTENCE_END`), whether words take spaces between them (`join_words` /
`split_words`), and what one unit of speech is (`speech_units` — a word, or a
character where there are no word boundaries). Those live here too, so the
segmenter, the translator, the timeline and TTS cannot each guess differently.
"""

from __future__ import annotations

# Unicode block ranges per script bucket. Deliberately coarse: whole blocks,
# base + extensions that actually occur in modern text.
_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "latin": ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F),
              (0x1E00, 0x1EFF)),
    "cyrillic": ((0x0400, 0x04FF), (0x0500, 0x052F), (0x2DE0, 0x2DFF),
                 (0xA640, 0xA69F)),
    "hebrew": ((0x0590, 0x05FF), (0xFB1D, 0xFB4F)),
    "arabic": ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
               (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)),
    "greek": ((0x0370, 0x03FF), (0x1F00, 0x1FFF)),
    "devanagari": ((0x0900, 0x097F),),
    "hangul": ((0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F)),
    # Han + kana share one bucket: Japanese text mixes them freely and Chinese
    # text is Han-only, so the bucket answers "is this CJK?" not "zh or ja?".
    "cjk": ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x3040, 0x309F),
            (0x30A0, 0x30FF), (0xF900, 0xFAFF)),
}

# Language code → script bucket. Extend as targets are added; unknown codes
# default to "latin" (the least surprising answer for the long tail of codes
# whisper and Gemma emit).
_LANG_SCRIPT: dict[str, str] = {
    "en": "latin", "es": "latin", "fr": "latin", "de": "latin", "it": "latin",
    "pt": "latin", "tr": "latin", "nl": "latin", "pl": "latin", "cs": "latin",
    "ro": "latin", "hu": "latin", "sv": "latin", "da": "latin", "no": "latin",
    "fi": "latin", "id": "latin", "vi": "latin", "az": "latin",
    "ru": "cyrillic", "uk": "cyrillic", "bg": "cyrillic", "sr": "cyrillic",
    "mk": "cyrillic", "be": "cyrillic", "kk": "cyrillic",
    "he": "hebrew", "yi": "hebrew", "iw": "hebrew",
    "ar": "arabic", "fa": "arabic", "ur": "arabic", "ps": "arabic",
    "el": "greek",
    "hi": "devanagari", "mr": "devanagari", "ne": "devanagari",
    "ko": "hangul",
    "zh": "cjk", "ja": "cjk", "yue": "cjk",
}


# CJK/fullwidth punctuation, and the ASCII mark each one stands for. One table,
# because two stages ask the same question of it: the segmenter's `SENTENCE_END`
# (where does a Japanese sentence end?) and `tts.prepare_text` (which marks may
# the voice see?). They disagreed once — the segmenter knew only ASCII "." and a
# Japanese read therefore had no interior sentence end at all.
CJK_PUNCT: dict[str, str] = {
    "。": ".", "｡": ".", "！": "!", "？": "?", "，": ",", "、": ",",
    "；": ";", "：": ":",
}
# The subset of it that ends a sentence rather than separating clauses.
CJK_SENTENCE_END = "".join(k for k, v in CJK_PUNCT.items() if v in ".!?")

# Characters that need no space beside them: Han, kana, and the CJK/fullwidth
# punctuation and forms that go with them. Chinese and Japanese are written
# without word spaces, so a space at either side of one of these is a space the
# language does not have. Korean is deliberately absent — hangul IS written with
# spaces, and only its particles glue onto the preceding word.
_UNSPACED_RANGES: tuple[tuple[int, int], ...] = (
    (0x3000, 0x303F),   # CJK symbols and punctuation (。、「」…)
    (0x3040, 0x30FF),   # hiragana + katakana
    (0x31F0, 0x31FF),   # katakana phonetic extensions
    (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF),   # Han
    (0xFF00, 0xFF60),   # fullwidth forms
    (0xFF61, 0xFF9F),   # halfwidth katakana and punctuation
)


def script_for(lang: str) -> str:
    """The script bucket a language is written in ("latin" for unknown codes)."""
    return _LANG_SCRIPT.get((lang or "").lower(), "latin")


def same_script(lang_a: str, lang_b: str) -> bool:
    """True when script detection cannot tell the two languages apart."""
    return script_for(lang_a) == script_for(lang_b)


def unspaced(ch: str) -> bool:
    """True when `ch` is written without spaces around it (Han, kana, CJK marks)."""
    return bool(ch) and any(lo <= ord(ch) <= hi for lo, hi in _UNSPACED_RANGES)


def join_words(parts: list[str] | tuple[str, ...]) -> str:
    """Join word tokens into one line, without a space across a CJK seam.

    ASR and captions hand Japanese and Chinese back one token at a time, and
    space-joining them writes text no reader of either language would produce
    ("これ は テスト です 。"). That text is what the translator is shown, what
    `_not_a_translation` counts `\\w+` tokens of, and what `relevant_context`
    matches a gloss against — all three read it as a dozen one-character words.
    The seam decides, not a language code: a space is dropped only when the
    character on one side of it belongs to a script that has no word spaces.
    """
    out = ""
    for part in parts:
        part = (part or "").strip()
        if not part:
            continue
        if out and not (unspaced(out[-1]) or unspaced(part[0])):
            out += " "
        out += part
    return out


def split_words(text: str) -> list[str]:
    """The units `join_words` would put back together: words, CJK per character."""
    out: list[str] = []
    run = ""
    for ch in text or "":
        if unspaced(ch) or ch.isspace():
            if run:
                out.append(run)
                run = ""
            if not ch.isspace():
                out.append(ch)
            continue
        run += ch
    if run:
        out.append(run)
    return out


def speech_units(text: str, lang: str) -> int:
    """How many speakable units the text has: words, or characters for CJK/hangul.

    The pipeline's length budgets — clone plausibility, token ceilings, how much
    shorter a rewrite must be — are all "how much speech is this?", and `.split()`
    answers 1 for a whole Japanese sentence. Every one of them goes through here.
    """
    if script_for(lang) in ("cjk", "hangul"):
        return max(1, sum(1 for ch in text or "" if ch.isalnum()))
    return max(1, len((text or "").split()))


def count_letters(text: str, script: str) -> int:
    """Letters of `text` belonging to `script` (non-letters never count)."""
    ranges = _RANGES[script]
    return sum(1 for ch in text or ""
               if ch.isalpha() and any(lo <= ord(ch) <= hi for lo, hi in ranges))


def script_ratio(text: str, lang: str) -> float:
    """Fraction of the letters of `text` written in `lang`'s script.

    0.0 when the text has no letters at all the caller decides what "no
    evidence" means, exactly as the old `latin_ratio` did.
    """
    script = script_for(lang)
    letters = sum(1 for ch in text or "" if ch.isalpha())
    if not letters:
        return 0.0
    return count_letters(text, script) / letters


def is_script(text: str, lang: str, threshold: float = 0.5) -> bool:
    """True when more than `threshold` of the text's letters are in `lang`'s script."""
    return script_ratio(text, lang) > threshold
