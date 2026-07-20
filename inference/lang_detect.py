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

# Common Hebrew function words — signal real HE commentary vs phonetic EN ASR.
_HE_FUNCTION_WORDS = re.compile(
    r"(?:^|[\s,])(של|את|על|עם|היא|הוא|אנחנו|אני|זה|זו|כי|אבל|גם|לא|"
    r"יש|אין|מה|מי|כמו|יותר|בין|אחרי|לפני|בסוף|מדובר)(?:[\s,.!?]|$)"
)


def _has_latin_island(text: str) -> bool:
    return bool(
        re.search(r"\b(CIA|ISIS|World War|Al-?Qaeda|UN|NATO)\b", text or "", re.I)
    )


def _real_hebrew_commentary(text: str) -> bool:
    """Dense Hebrew with grammar words and no EN interview signal → dub, not KEEP."""
    t = (text or "").strip()
    if not t:
        return False
    ratios = script_ratios(t)
    if ratios["he"] < 0.70 or ratios["en"] >= 0.15:
        return False
    if _has_latin_island(t):
        return False
    n_strong = sum(1 for m in _PHONETIC_EN_MARKERS if m in t)
    if n_strong >= 2:
        return False
    weak_only = any(m in t for m in _WEAK_LOANWORD_MARKERS) and n_strong == 0
    he_funcs = len(_HE_FUNCTION_WORDS.findall(t))
    if he_funcs >= 2 or weak_only:
        return True
    return he_funcs >= 1 and ratios["he"] >= 0.85 and n_strong == 0


def should_keep_original(
    text: str,
    *,
    lang: str | None = None,
    he_score: float | None = None,
    en_score: float | None = None,
    neighbors: list[dict[str, Any]] | None = None,
) -> bool:
    """Single KEEP gate: score-first; phonetic markers only as tie-break.

    1. Winning lang != he → KEEP
    2. en_score within margin of he_score → KEEP (interview EN even if HE-script ASR)
    3. Phonetic-HE rescue: strong markers AND (competitive EN / Latin island /
       adjacent KEEP-English neighbor)
    4. Real-HE veto: dense Hebrew commentary without competitive EN → never KEEP
    """
    lang = (lang or "he").lower()
    t = (text or "").strip()
    if lang != "he":
        return True
    if not t:
        return False

    competitive_en = (
        he_score is not None
        and en_score is not None
        and float(en_score) >= float(he_score) - _EN_SCORE_MARGIN
    )

    # Real HE commentary veto (unless EN score is competitive).
    if _real_hebrew_commentary(t) and not competitive_en:
        return False

    if competitive_en:
        return True

    n_strong = sum(1 for m in _PHONETIC_EN_MARKERS if m in t)
    has_strong = n_strong >= 1
    latin = _has_latin_island(t)
    neighbor_keep_en = False
    if neighbors:
        for n in neighbors:
            n_lang = (n.get("language") or "he").lower()
            if n.get("keep_original") or n_lang == "en":
                neighbor_keep_en = True
                break

    # Phonetic rescue: markers alone are not enough — need a corroborating signal.
    if has_strong and (latin or neighbor_keep_en or n_strong >= 2):
        return True
    # Competitive EN score already returned True above; neighbor alone without
    # strong markers must not KEEP (avoids false positives on HE news).
    if neighbor_keep_en and looks_like_phonetic_english(t) and has_strong:
        return True
    return False


def looks_like_phonetic_english(text: str) -> bool:
    """True when a 'Hebrew' transcript is mostly English spoken phonetically.

    High-Hebrew script with only weak loanwords (e.g. פילנתרופ) is NOT enough —
    that is real Hebrew commentary and must stay as HE dub, not KEEP-original.

    Strong phonetic markers (אינפלואנס / אופריציה / קמפלאז) DO force KEEP even
    when Whisper wrote the English interview in Hebrew letters.
    """
    t = (text or "").strip()
    if not t:
        return False
    ratios = script_ratios(t)
    has_strong_marker = any(m in t for m in _PHONETIC_EN_MARKERS)
    has_weak_loan = any(m in t for m in _WEAK_LOANWORD_MARKERS)
    has_latin_island = _has_latin_island(t)
    n_strong = sum(1 for m in _PHONETIC_EN_MARKERS if m in t)

    # Weak loanwords alone in dense HE → real Hebrew commentary, not KEEP.
    if (
        ratios["he"] >= 0.70
        and ratios["en"] < 0.08
        and not has_strong_marker
        and not has_latin_island
    ):
        return False

    # Strong phonetic spellings of EN interview speech (even if HE letters dominate).
    if has_strong_marker and (n_strong >= 2 or ratios["en"] >= 0.05 or has_latin_island):
        return True
    if has_strong_marker and ratios["he"] <= 0.80:
        return True

    if has_weak_loan and ratios["en"] >= 0.12 and ratios["he"] <= 0.65:
        return True

    # Dense Latin + sparse real Hebrew grammar words → likely English audio.
    if ratios["en"] >= 0.35 and ratios["he"] <= 0.55:
        he_only = HE_RE.findall(t)
        if ratios["total_letters"] >= 12 and ratios["en"] >= 0.45:
            return True
        if len(he_only) >= 8 and ratios["en"] >= 0.25 and has_latin_island:
            return True
    if has_latin_island and ratios["en"] >= 0.10:
        return True
    return False


def text_is_hebrew_script_heavy(text: str, *, he_min: float = 0.70) -> bool:
    """True when transcript is predominantly Hebrew letters (not Latin EN ASR).

    Does NOT treat strong phonetic-EN interviews as 'Hebrew-only' — those still
    need KEEP even though Whisper wrote them in Hebrew script.
    """
    if should_keep_original(text, lang="he") or looks_like_phonetic_english(text):
        return False
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
) -> tuple[str, float, list[Any], str, dict[str, float]]:
    """Return (lang, score, whisper_segments, text, scores_by_lang).

    When KEEP is warranted for HE-script phonetic English, keep the HE transcript
    (do not sanitize_text(..., "en") into near-empty Latin).
    """
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

    empty_scores: dict[str, float] = {}
    if not scored:
        return "he", -99.0, [], "", empty_scores

    scored.sort(key=lambda x: x[1], reverse=True)
    scores_by_lang = {row[0]: float(row[1]) for row in scored}

    def _finalize(
        row: tuple[str, float, list[Any], str], *, as_lang: str | None = None
    ) -> tuple[str, float, list[Any], str]:
        lang, score, segments, text = row
        out_lang = as_lang or lang
        # KEEP phonetic-EN: preserve HE-script ASR for debug; do not wipe to Latin.
        if as_lang == "en" and lang == "he":
            text = restore_latin_names((text or "").strip())
            return "en", score, segments, text
        text = sanitize_text(text, out_lang)
        if out_lang in ("he", "ar"):
            text = restore_latin_names(text)
        return out_lang, score, segments, text

    by_lang = {row[0]: row for row in scored}
    he_row = by_lang.get("he")
    en_row = by_lang.get("en")

    chosen = scored[0]
    lang, score, segments, text = chosen
    he_score = scores_by_lang.get("he")
    en_score = scores_by_lang.get("en")
    he_text_raw = he_row[3] if he_row else text

    # Score-first KEEP via should_keep_original — prefer HE transcript + lang=en.
    if lang == "he" and should_keep_original(
        he_text_raw,
        lang="he",
        he_score=he_score,
        en_score=en_score,
    ):
        if he_row is not None:
            lang, score, segments, text = _finalize(he_row, as_lang="en")
        elif en_row is not None:
            lang, score, segments, text = _finalize(en_row)
        else:
            lang, score, segments, text = _finalize(chosen, as_lang="en")
    else:
        lang, score, segments, text = _finalize(chosen)

    # Reject punctuation-only / empty winners; try next usable candidate.
    if _letter_count(text) < 3:
        for row in scored:
            lang2, score2, segs2, text2 = _finalize(row)
            if _letter_count(text2) >= 3:
                lang, score, segments, text = lang2, score2, segs2, text2
                break
        else:
            return "he", -99.0, [], "", scores_by_lang

    return lang, score, segments, text, scores_by_lang


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
        lang, score, segments, text, scores_by_lang = score_language(
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
                # Keep HE-script tokens for phonetic-EN KEEP (do not drop).
                if script_ratios(text or "")["he"] < 0.35:
                    continue
            if lang == "he" and LAT_RE.search(token) and not HE_RE.search(token):
                # Keep Latin brand names inside Hebrew (ISIS etc.)
                pass
            # Remap clip-relative times → absolute timeline
            w.start = float(w.start) + start
            w.end = float(w.end) + start
            words.append(w)

    keep = should_keep_original(
        text,
        lang=lang if lang != "en" or script_ratios(text or "")["en"] >= 0.35 else "he",
        he_score=scores_by_lang.get("he"),
        en_score=scores_by_lang.get("en"),
    )
    # If gate says KEEP but whisper winner was HE, tag as EN keep.
    if keep and lang == "he":
        lang = "en"

    return {
        "language": lang,
        "language_score": round(score, 4),
        "he_score": round(scores_by_lang["he"], 4) if "he" in scores_by_lang else None,
        "en_score": round(scores_by_lang["en"], 4) if "en" in scores_by_lang else None,
        "ar_score": round(scores_by_lang["ar"], 4) if "ar" in scores_by_lang else None,
        "keep_original": keep or lang != "he",
        "text": text,
        "words": words,
        "start": start,
        "end": end,
    }
