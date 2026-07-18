"""Per-clip language detection for mixed-language documentary audio.

Whisper's whole-file language ID is biased toward the dominant language (Hebrew).
For each short turn we score he / en / ar transcriptions by avg_logprob + script match,
then keep non-Hebrew audio as-is downstream.
"""

from __future__ import annotations

import re
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any

HE_RE = re.compile(r"[\u0590-\u05FF]")
AR_RE = re.compile(r"[\u0600-\u06FF]")
LAT_RE = re.compile(r"[A-Za-z]")

CANDIDATE_LANGS = ("he", "en", "ar")

# Whisper-he often transliterates English names; restore Latin forms for dubbing.
_CODE_SWITCH_LATIN: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bאייסיס\b"), "ISIS"),
    (re.compile(r"\bאיסיס\b"), "ISIS"),
    (re.compile(r"\bISIS\b", re.I), "ISIS"),
    (re.compile(r"\bאל-?קאעידה\b"), "Al-Qaeda"),
    (re.compile(r"\bאל-?קאידה\b"), "Al-Qaeda"),
    (re.compile(r"\bAl-?Qaeda\b", re.I), "Al-Qaeda"),
]

# Phonetic Hebrew spellings of English words (documentary interviews).
# Do NOT include common Hebrew loanwords that appear in real HE commentary
# (פילנתרופ / בנבולנט) — those alone must not force KEEP of Hebrew audio.
_PHONETIC_EN_MARKERS: tuple[str, ...] = (
    "פרנקלי",
    "פרנקליי",
    "אינפלואנס",
    "אינפלונס",
    "אופריציה",
    "אופרציה",
    "דירקטור",
    "אלג'יינס",
    "אלג'יאנס",
    "ברידג",
    "מוסלם בראודהוד",
    "בראודהוד",
    "קמפלאז",
    "וורלד וור",
    "הפסקה לקרד",  # "She's the bridge to Cardinal" style
    "לקרדלי",
    "פילנדה",  # planet-ish
    "סטיל",  # styles
    "אלג'ינס",
)

# Weak loanword markers: only count toward phonetic-EN when Latin is also present
# or Hebrew script is sparse (otherwise real HE commentary gets wrongly KEEP'd).
_WEAK_LOANWORD_MARKERS: tuple[str, ...] = (
    "פילנתרופ",
    "פילנת'",
    "בנבולנט",
    "בנבלנט",
)

# Prefer EN when its score is within this margin of HE (avoids HE bias on short EN).
_EN_SCORE_MARGIN = 0.35


def looks_like_phonetic_english(text: str) -> bool:
    """True when a 'Hebrew' transcript is mostly English spoken phonetically.

    High-Hebrew script with only loanword markers (e.g. פילנתרופ) is NOT enough —
    that is real Hebrew commentary and must stay as HE dub, not KEEP-original.
    """
    t = (text or "").strip()
    if not t:
        return False
    ratios = script_ratios(t)
    has_strong_marker = any(m in t for m in _PHONETIC_EN_MARKERS)
    has_weak_loan = any(m in t for m in _WEAK_LOANWORD_MARKERS)

    # Dense Hebrew + near-zero Latin: refuse KEEP even if a weak loanword matches.
    if ratios["he"] >= 0.70 and ratios["en"] < 0.08:
        return False

    if has_strong_marker:
        # Strong markers still need either some Latin or not-overwhelming HE.
        if ratios["en"] >= 0.08 or ratios["he"] <= 0.75:
            return True
        return False

    if has_weak_loan and ratios["en"] >= 0.12 and ratios["he"] <= 0.65:
        return True

    # Dense Latin + sparse real Hebrew grammar words → likely English audio.
    if ratios["en"] >= 0.35 and ratios["he"] <= 0.55:
        he_only = HE_RE.findall(t)
        # Short HE runs that are mostly transliteration length tokens
        if ratios["total_letters"] >= 12 and ratios["en"] >= 0.45:
            return True
        if len(he_only) >= 8 and ratios["en"] >= 0.25:
            # Many HE letters but with CIA / World War / Al-Qaeda style Latin islands
            if re.search(r"\b(CIA|ISIS|World War|Al-?Qaeda)\b", t, re.I):
                return True
    return False


def text_is_hebrew_script_heavy(text: str, *, he_min: float = 0.70) -> bool:
    """True when transcript is predominantly Hebrew letters (not Latin EN ASR)."""
    ratios = script_ratios(text or "")
    return ratios["he"] >= he_min and ratios["en"] < 0.15


def restore_latin_names(text: str) -> str:
    """Keep English org/acronym tokens in Latin inside Hebrew transcripts."""
    out = text or ""
    for pat, repl in _CODE_SWITCH_LATIN:
        out = pat.sub(repl, out)
    return out


def script_ratios(text: str) -> dict[str, float]:
    he = len(HE_RE.findall(text))
    ar = len(AR_RE.findall(text))
    lat = len(LAT_RE.findall(text))
    total = max(he + ar + lat, 1)
    return {"he": he / total, "ar": ar / total, "en": lat / total, "total_letters": float(he + ar + lat)}


def sanitize_text(text: str, lang: str) -> str:
    """Light cleanup for the winning language — keep code-switched names.

    Hebrew documentary speech often embeds Latin tokens (ISIS, Al-Qaeda, brands).
    Never strip those from `he` / `ar` transcripts.
    """
    text = (text or "").strip()
    if not text:
        return ""
    if lang == "en":
        # Drop HE/AR runs; keep Latin (including names).
        text = re.sub(r"[\u0590-\u05FF\u0600-\u06FF]+", "", text)
    elif lang == "he":
        # Drop Arabic only — keep Latin names inside Hebrew.
        text = re.sub(r"[\u0600-\u06FF]+", "", text)
    elif lang == "ar":
        # Drop Hebrew only — keep Latin names inside Arabic.
        text = re.sub(r"[\u0590-\u05FF]+", "", text)
    text = re.sub(r"\s+", " ", text).strip(" .,;:-")
    return text.strip()


def script_bonus(text: str, lang: str) -> float:
    """Reward transcripts whose script matches the forced language."""
    if not (text or "").strip():
        return -1.0
    he = len(HE_RE.findall(text))
    ar = len(AR_RE.findall(text))
    lat = len(LAT_RE.findall(text))
    # Latin inside HE/AR is code-switching, not a failed script match.
    if lang == "he":
        denom = max(he + ar, 1)
        matched = he / denom
        if he + ar < 2 and lat >= 2:
            return -0.5
    elif lang == "ar":
        denom = max(ar + he, 1)
        matched = ar / denom
        if ar + he < 2 and lat >= 2:
            return -0.5
    elif lang == "en":
        denom = max(lat + he + ar, 1)
        matched = lat / denom
    else:
        matched = 0.0
        denom = 1
    if (he + ar + lat) < 2:
        return -1.0
    if matched < 0.25:
        return -1.5
    return 0.75 * matched


def extract_mono_wav(src: Path, start: float, end: float, dst: Path, sample_rate: int = 16000) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dur = max(end - start, 0.05)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{dur:.3f}",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-acodec",
            "pcm_s16le",
            str(dst),
        ],
        check=True,
        capture_output=True,
    )
    return dst


def _letter_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z\u0590-\u05FF\u0600-\u06FF]", text or ""))


def score_language(
    model: Any,
    clip_path: Path,
    *,
    candidates: tuple[str, ...] = CANDIDATE_LANGS,
    beam_size: int = 5,
) -> tuple[str, float, list[Any], str]:
    """Return (lang, score, whisper_segments, text) for the best candidate."""
    scored: list[tuple[str, float, list[Any], str]] = []

    for lang in candidates:
        segments_iter, _info = model.transcribe(
            str(clip_path),
            language=lang,
            word_timestamps=True,
            beam_size=beam_size,
            vad_filter=False,
        )
        segments = list(segments_iter)
        if not segments:
            continue
        text = " ".join((s.text or "").strip() for s in segments).strip()
        if not text or _letter_count(text) < 2:
            continue
        logprob = statistics.mean(float(s.avg_logprob) for s in segments)
        score = logprob + script_bonus(text, lang)
        scored.append((lang, score, segments, text))

    if not scored:
        return "he", -99.0, [], ""

    scored.sort(key=lambda x: x[1], reverse=True)

    def _finalize(row: tuple[str, float, list[Any], str]) -> tuple[str, float, list[Any], str]:
        lang, score, segments, text = row
        text = sanitize_text(text, lang)
        if lang in ("he", "ar"):
            text = restore_latin_names(text)
        return lang, score, segments, text

    by_lang = {row[0]: row for row in scored}
    he_row = by_lang.get("he")
    en_row = by_lang.get("en")

    chosen = scored[0]
    lang, score, segments, text = chosen

    # Prefer English when HE "wins" but the transcript is phonetic English, or
    # when EN is within a small score margin (Whisper-he is biased on this corpus).
    if lang == "he" and en_row is not None:
        he_text = he_row[3] if he_row else text
        en_text = en_row[3]
        en_ratios = script_ratios(en_text)
        if looks_like_phonetic_english(he_text) or (
            en_row[1] >= score - _EN_SCORE_MARGIN and en_ratios["en"] >= 0.55
        ):
            chosen = en_row

    lang, score, segments, text = _finalize(chosen)

    # Reject punctuation-only / empty winners; try next usable candidate.
    if _letter_count(text) < 3:
        for row in scored:
            lang2, score2, segs2, text2 = _finalize(row)
            if _letter_count(text2) >= 3:
                lang, score, segments, text = lang2, score2, segs2, text2
                break
        else:
            return "he", -99.0, [], ""

    if lang == "he" and en_row is not None and looks_like_phonetic_english(text):
        lang, score, segments, text = _finalize(en_row)
        if _letter_count(text) < 3 and he_row is not None:
            lang, score, segments, text = _finalize(he_row)

    return lang, score, segments, text


def detect_and_transcribe_turn(
    model: Any,
    vocals_path: Path,
    start: float,
    end: float,
    *,
    candidates: tuple[str, ...] = CANDIDATE_LANGS,
    beam_size: int = 5,
) -> dict[str, Any]:
    """Transcribe one time window and attach the winning language."""
    with tempfile.TemporaryDirectory(prefix="langdetect_") as tmp:
        clip = Path(tmp) / "clip.wav"
        extract_mono_wav(vocals_path, start, end, clip)
        lang, score, segments, text = score_language(
            model, clip, candidates=candidates, beam_size=beam_size
        )

    words: list[Any] = []
    for seg in segments:
        if not seg.words:
            continue
        for w in seg.words:
            if w.start is None or w.end is None:
                continue
            token = (w.word or "").strip()
            if lang == "en" and HE_RE.search(token):
                continue
            if lang == "he" and LAT_RE.search(token) and not HE_RE.search(token):
                # Keep Latin brand names inside Hebrew (ISIS etc.)
                pass
            # Remap clip-relative times → absolute timeline
            w.start = float(w.start) + start
            w.end = float(w.end) + start
            words.append(w)

    return {
        "language": lang,
        "language_score": round(score, 4),
        "text": text,
        "words": words,
        "start": start,
        "end": end,
    }
