"""Which writing system a language uses, and whether text is written in it.

The pipeline's original form hardcoded a two-script world — Latin meant "target
(English)" and the Hebrew block meant "source" — in half a dozen places. This
module is the single replacement: every "is this text in the target language's
script?" question goes through `is_script` / `script_ratio` with an explicit
language code, so a Russian or Arabic target works the same way English did.

Script detection is a *cheap, high-precision* signal and only that. It separates
languages written differently (he/en, en/ru, he/ar); for same-script pairs
(en→es, ru→uk) it cannot discriminate and callers must fall back to LID —
`same_script()` tells them when. CJK gets one shared bucket: Japanese mixes kana
with Han and the han bucket alone cannot separate zh from ja.
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


def script_for(lang: str) -> str:
    """The script bucket a language is written in ("latin" for unknown codes)."""
    return _LANG_SCRIPT.get((lang or "").lower(), "latin")


def same_script(lang_a: str, lang_b: str) -> bool:
    """True when script detection cannot tell the two languages apart."""
    return script_for(lang_a) == script_for(lang_b)


def count_letters(text: str, script: str) -> int:
    """Letters of `text` belonging to `script` (non-letters never count)."""
    ranges = _RANGES[script]
    return sum(1 for ch in text or ""
               if ch.isalpha() and any(lo <= ord(ch) <= hi for lo, hi in ranges))


def script_ratio(text: str, lang: str) -> float:
    """Fraction of the letters of `text` written in `lang`'s script.

    0.0 when the text has no letters at all — the caller decides what "no
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
