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


def script_ratios(text: str) -> dict[str, float]:
    he = len(HE_RE.findall(text))
    ar = len(AR_RE.findall(text))
    lat = len(LAT_RE.findall(text))
    total = max(he + ar + lat, 1)
    return {"he": he / total, "ar": ar / total, "en": lat / total, "total_letters": float(he + ar + lat)}


def sanitize_text(text: str, lang: str) -> str:
    """Drop characters that don't belong to the winning language's script."""
    text = (text or "").strip()
    if not text:
        return ""
    if lang == "en":
        text = re.sub(r"[\u0590-\u05FF\u0600-\u06FF]+", "", text)
    elif lang == "he":
        text = re.sub(r"[A-Za-z]+", "", text)
    elif lang == "ar":
        text = re.sub(r"[\u0590-\u05FF]+", "", text)
        text = re.sub(r"[A-Za-z]+", "", text)
    text = re.sub(r"\s+", " ", text).strip(" .,;:-")
    return text.strip()


def script_bonus(text: str, lang: str) -> float:
    """Reward transcripts whose script matches the forced language."""
    ratios = script_ratios(text)
    if ratios["total_letters"] < 2:
        return -1.0
    matched = ratios.get(lang, 0.0)
    # Heavy penalty when forced-lang script is almost absent (hallucinated wrong lang).
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


def score_language(
    model: Any,
    clip_path: Path,
    *,
    candidates: tuple[str, ...] = CANDIDATE_LANGS,
    beam_size: int = 5,
) -> tuple[str, float, list[Any], str]:
    """Return (lang, score, whisper_segments, text) for the best candidate."""
    best: tuple[str, float, list[Any], str] | None = None

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
        if not text:
            continue
        logprob = statistics.mean(float(s.avg_logprob) for s in segments)
        score = logprob + script_bonus(text, lang)
        if best is None or score > best[1]:
            best = (lang, score, segments, text)

    if best is None:
        return "he", -99.0, [], ""
    lang, score, segments, text = best
    text = sanitize_text(text, lang)
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
