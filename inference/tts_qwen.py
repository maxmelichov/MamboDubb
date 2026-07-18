#!/usr/bin/env python3
"""Qwen3-TTS 1.7B-Base zero-shot voice clone for DubbingQwen (Phase 4).

Clones from each phrase's own vocal window (~3s Hebrew ref), then synthesizes
English. Default uses speaker-embedding clone (x_vector_only) — stabler for
Hebrew→English than ICL with Hebrew ref_text.

Model: Qwen/Qwen3-TTS-12Hz-1.7B-Base (https://arxiv.org/abs/2601.15621)
  - x_vector_only=True (default): speaker embedding from ref audio
  - x_vector_only=False: ICL with ref_audio + Hebrew ref_text
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from inference.tts_f5 import (
    atempo_chain,
    extend_end_by_energy,
    extract_wav_slice,
    phrase_plan,
    place_in_slot,
    pull_start_to_energy,
    wav_duration,
)

# Prefer constant speaking pace. Only speed up when EN is clearly too long for
# the HE slot; never slow down; avoid micro-retiming that makes adjacent lines
# alternate slow/fast.
FIT_MAX_RATE = 1.18
FIT_MIN_RATE = 1.0
FIT_SPEEDUP_THRESHOLD = 1.12  # below this, keep rate=1.0 and overrun slightly
# Only shorten when even mild speedup + slack cannot fit.
SHORTEN_RETRY_RATE = 1.28
SHORTEN_SYL_PER_SEC = 3.6
SHORTEN_MAX_WORD_DROP = 0.35  # reject rewrites that drop more than ~35% of words
SHORTEN_MIN_SLOT_SEC = 1.2  # default floor; overridden when EN≫HE
# Cap how far past ASR end a short HE slot may expand into the next gap.
SHORT_SLOT_OVERRUN_CAP_SEC = 0.35
TIGHT_NEXT_GAP_SEC = 0.30  # force shorten / no overrun when next onset this close
# 0 = preserve original HE pauses (default). Set >0 to cap mid-utterance silence.
DEFAULT_MAX_DUB_PAUSE = 0.0
# Adjacent phrases with gap below this become one TTS run (no mid-sentence cut).
# Keep tight so slow/deliberate HE breaths stay as separate alignment anchors.
SPEECH_RUN_GAP_SEC = 0.18
# Slow Hebrew (low words/sec): EN at natural rate finishes early → misaligned lips.
SLOW_HE_WPS = 2.0
SLOW_FILL_RATIO = 0.82
SLOW_MIN_RATE = 0.90  # mild stretch only — not cartoon slow-mo
SLOW_LEAD_FRAC = 0.40  # put this fraction of leftover silence before speech
SLOW_LEAD_MAX_SEC = 0.85
SLOW_SLOT_MIN_SEC = 1.40
# Qwen often emits ~0.4–0.6s hush before speech; strip before fit/place.
LEAD_SILENCE_MAX_TRIM_SEC = 0.85
LEAD_SILENCE_PAD_SEC = 0.03
# Only when EN/HE mismatch would force rate outside bounds, borrow pause room.
SLACK_MAX_EARLY_SEC = 0.25
SLACK_MAX_LATE_SEC = 0.45

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "models" / "Qwen3-TTS-12Hz-1.7B-Base"
HUB_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
# Qwen card: ~3s clone refs. Longer ICL refs + Hebrew→EN blew up decode time on MPS.
REF_TARGET_SEC = 3.2
REF_MIN_SEC = 2.0
REF_MAX_SEC = 4.5
CODEC_HZ = 12.5


def _hf_noise_ratio(audio: np.ndarray, sr: int) -> float:
    """High-band / mid-band energy — Demucs hiss scores high; clean speech low."""
    if audio.size < sr // 4:
        return 1e9
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=-1)
    X = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    mid = float(np.mean(X[(freqs >= 200) & (freqs < 3500)] ** 2) + 1e-12)
    high = float(np.mean(X[(freqs >= 5500) & (freqs < 14000)] ** 2))
    return high / mid


def pick_cleanest_ref_window(
    vocals: Path,
    win_start: float,
    win_end: float,
    *,
    media_duration: float,
    target_sec: float = REF_TARGET_SEC,
) -> tuple[float, float]:
    """Choose a ~target_sec slice inside [win_start, win_end] with least HF hiss."""
    span = max(0.0, win_end - win_start)
    if span <= 0.05:
        a = max(0.0, min(win_start, media_duration - target_sec))
        return a, min(media_duration, a + target_sec)

    # Load the search region once (mono 24k).
    pad = 0.05
    load_a = max(0.0, win_start - pad)
    load_b = min(media_duration, win_end + pad)
    tmp = vocals.parent / "tts_refs" / "_ref_scan.wav"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    extract_wav_slice(vocals, load_a, load_b, tmp, sample_rate=24000)
    audio, sr = sf.read(str(tmp), dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=-1).astype(np.float32)

    want = int(round(target_sec * sr))
    if len(audio) <= want:
        return load_a, load_b

    hop = max(1, int(0.2 * sr))
    best_i = 0
    best_score = 1e18
    # Prefer windows that are actually speech, not near-silence.
    for i in range(0, len(audio) - want + 1, hop):
        chunk = audio[i : i + want]
        rms = float(np.sqrt(np.mean(chunk**2) + 1e-12))
        if rms < 0.018:
            continue
        score = _hf_noise_ratio(chunk, sr) / max(rms, 1e-3)
        if score < best_score:
            best_score = score
            best_i = i

    ref_start = load_a + best_i / sr
    ref_end = min(media_duration, ref_start + target_sec)
    if ref_end - ref_start < REF_MIN_SEC:
        ref_start = max(0.0, ref_end - REF_MIN_SEC)
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass
    return ref_start, ref_end


def _speaker_search_windows(
    seg: dict,
    phrase: dict,
    all_segments: list[dict] | None,
) -> list[tuple[float, float]]:
    """Candidate time ranges for a clean clone ref (phrase, then same speaker)."""
    p_start = float(phrase["start"])
    p_end = float(phrase["end"])
    seg_a = float(seg.get("start") or p_start)
    seg_b = float(seg.get("end") or p_end)
    windows = [
        (max(seg_a, p_start - 0.15), min(seg_b, p_end + 0.15)),
        (seg_a, seg_b),
    ]
    spk = seg.get("speaker_id")
    if spk and all_segments:
        for other in all_segments:
            if other.get("speaker_id") != spk:
                continue
            if other.get("keep_original"):
                continue
            if (other.get("language") or "he") != (seg.get("language") or "he"):
                continue
            a, b = float(other["start"]), float(other["end"])
            if b - a >= REF_MIN_SEC:
                windows.append((a, b))
    # Dedup / drop empties
    out: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for a, b in windows:
        key = (round(a, 3), round(b, 3))
        if b - a < 0.3 or key in seen:
            continue
        seen.add(key)
        out.append((a, b))
    return out


def build_qwen_phrase_ref(
    phrase: dict,
    seg: dict,
    seg_index: int,
    phrase_index: int,
    vocals: Path,
    out_dir: Path,
    media_duration: float,
    all_segments: list[dict] | None = None,
) -> dict:
    """~3s vocal slice + Hebrew transcript for Qwen Base clone.

    Picks the cleanest window on this speaker (lowest Demucs HF hiss), including
    later/earlier turns of the same speaker_id when the local phrase is hissy.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    he_text = (phrase.get("text") or seg.get("text") or "um").strip() or "um"

    best: tuple[float, float, float] | None = None  # score, start, end
    for search_a, search_b in _speaker_search_windows(seg, phrase, all_segments):
        target = min(REF_TARGET_SEC, max(REF_MIN_SEC, search_b - search_a))
        ref_start, ref_end = pick_cleanest_ref_window(
            vocals,
            search_a,
            search_b,
            media_duration=media_duration,
            target_sec=target,
        )
        # Score the chosen window
        tmp = out_dir / f"_score_{seg_index:02d}_{phrase_index:02d}.wav"
        extract_wav_slice(vocals, ref_start, ref_end, tmp, sample_rate=24000)
        audio, sr = sf.read(str(tmp), dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = np.mean(audio, axis=-1).astype(np.float32)
        rms = float(np.sqrt(np.mean(audio**2) + 1e-12))
        score = _hf_noise_ratio(audio, sr) / max(rms, 1e-3)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        if best is None or score < best[0]:
            best = (score, ref_start, ref_end)

    if best is None:
        mid = (float(phrase["start"]) + float(phrase["end"])) / 2.0
        ref_start = max(0.0, mid - REF_TARGET_SEC / 2.0)
        ref_end = min(media_duration, ref_start + REF_TARGET_SEC)
    else:
        _, ref_start, ref_end = best
    if ref_end - ref_start > REF_MAX_SEC:
        ref_end = ref_start + REF_MAX_SEC

    ref_path = out_dir / f"seg_{seg_index:02d}_p{phrase_index:02d}_qwen_ref.wav"
    extract_wav_slice(vocals, ref_start, ref_end, ref_path, sample_rate=24000)
    return {
        "path": str(ref_path.resolve()),
        "ref_text": he_text,
        "start": ref_start,
        "end": ref_end,
    }

# Hebrew→EN clone sometimes Hebrew-izes or drops names. Prefer spoken-stable forms.
_TTS_NAME_RESPPELL: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bHezbollah\b", re.I), "Hezbollah"),
    (re.compile(r"\bHizballah\b", re.I), "Hezbollah"),
    (re.compile(r"\bHizballa\b", re.I), "Hezbollah"),
    # Hyphenated Arabic org names are often swallowed in list tails.
    (re.compile(r"\bJabhat\s+al-?Nusra\b", re.I), "the Nusra Front"),
    (re.compile(r"\bJahbat\s+al-?Nusra\b", re.I), "the Nusra Front"),
    (re.compile(r"\bAl-?Qaeda\b", re.I), "Al-Qaeda"),
    (re.compile(r"\bAl\s+Qaeda\b", re.I), "Al-Qaeda"),
    (re.compile(r"\bSheikha\s+Moza\b", re.I), "Sheikha Moza"),
    (re.compile(r"\bSheikh\s+Moza\b", re.I), "Sheikha Moza"),
]


def prepare_english_tts_text(text: str) -> str:
    """Light cleanup for HE→EN voice clone (do not respell common words).

    Earlier funds→fundz made 1.7B say "funs"; keep natural English orthography.
    Map hard proper nouns to forms Qwen actually voices in list enumerations.
    """
    out = (text or "").strip()
    if not out:
        return out
    for pat, repl in _TTS_NAME_RESPPELL:
        out = pat.sub(repl, out)
    # Undo any stale fundz/fundd hacks left in curated JSON.
    out = re.sub(r"\bfundz\b", "funds", out, flags=re.I)
    out = re.sub(r"\bfundd\b", "fund", out, flags=re.I)
    if out[-1] not in ".!?":
        out += "."
    return out


def _transcribe_keep_english(vocals: Path, start: float, end: float) -> str | None:
    """Re-ASR an extended KEEP English window (picks up trailing clauses)."""
    try:
        from faster_whisper import WhisperModel
        from inference.extract_pipeline import resolve_whisper_model
    except Exception:
        return None
    if end - start < 0.4:
        return None
    tmp = vocals.parent / "tts_clips" / "_keep_en_probe.wav"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    extract_wav_slice(vocals, start, end, tmp, sample_rate=16000)
    try:
        model_path = resolve_whisper_model("ivrit-ai/whisper-large-v3-turbo-ct2")
        model = WhisperModel(str(model_path), device="cpu", compute_type="auto")
        segs, _info = model.transcribe(
            str(tmp),
            language="en",
            word_timestamps=False,
            condition_on_previous_text=False,
        )
        text = " ".join((s.text or "").strip() for s in segs).strip()
        text = re.sub(r"\s+", " ", text)
        if not text:
            return None
        # ivrit CT2 sometimes still emits Hebrew script under language=en.
        if re.search(r"[\u0590-\u05FF]", text):
            return None
        return text
    except Exception as exc:
        print(f"  KEEP re-ASR failed: {exc}", file=sys.stderr)
        return None
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def estimate_max_new_tokens(text: str, target_sec: float | None = None) -> int:
    """Allow the model to finish the full sentence (never cap to the Hebrew slot).

    Slot matching is done afterward by stretching the complete waveform.
    `target_sec` is ignored for the cap — keeping it in the signature for callers.
    """
    del target_sec  # duration fit is stretch-only; do not truncate generation
    words = max(1, len(text.split()))
    # Generous headroom so EOS ends the line, not max_new_tokens.
    sec = words * 0.65 + 2.5
    return max(96, min(2048, int(sec * CODEC_HZ)))


def _atempo_to(src: Path, dst: Path, rate: float, *, sample_rate: int) -> None:
    # Allow wide rates; atempo_chain splits into 0.5–2.0 stages.
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-af",
            atempo_chain(rate),
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            str(dst),
        ],
        check=True,
        capture_output=True,
    )


def _remap_to_n(audio: np.ndarray, target_n: int) -> np.ndarray:
    """Resample the full waveform to exactly target_n samples (no chop / no pad)."""
    if target_n <= 0:
        raise ValueError("target_n must be positive")
    if len(audio) == target_n:
        return audio.astype(np.float32, copy=False)
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=True)
    x_new = np.linspace(0.0, 1.0, num=target_n, endpoint=True)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def trim_leading_silence(
    src: Path,
    dst: Path,
    *,
    sample_rate: int = 44100,
    rms_thresh: float = 0.012,
    pad_sec: float = LEAD_SILENCE_PAD_SEC,
    max_trim_sec: float = LEAD_SILENCE_MAX_TRIM_SEC,
) -> float:
    """Drop hush before first speech energy. Returns seconds removed."""
    audio, sr = sf.read(str(src), dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=-1).astype(np.float32)
    if sr != sample_rate:
        raise RuntimeError(f"Unexpected sample rate {sr} (want {sample_rate})")
    if audio.size < sample_rate // 5:
        if Path(src) != Path(dst):
            sf.write(str(dst), audio, sample_rate)
        return 0.0

    hop = max(1, sample_rate // 50)  # 20ms
    limit = min(len(audio) - hop, int(max_trim_sec * sample_rate))
    onset = 0
    for i in range(0, max(1, limit), hop):
        rms = float(np.sqrt(np.mean(audio[i : i + hop] ** 2) + 1e-12))
        if rms >= rms_thresh:
            onset = max(0, i - int(pad_sec * sample_rate))
            break
    trimmed = onset / sample_rate
    if trimmed < 0.04:
        if Path(src) != Path(dst):
            sf.write(str(dst), audio, sample_rate)
        return 0.0
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), audio[onset:].astype(np.float32), sample_rate)
    return float(trimmed)


def compact_phrase_timeline_gaps(
    plan: list[dict],
    *,
    max_pause: float = DEFAULT_MAX_DUB_PAUSE,
    min_gap_to_compact: float = 0.85,
) -> float:
    """Cap intra-utterance pauses, shifting later phrases earlier.

    Opt-in: when max_pause <= 0, pauses are preserved at their HE anchors
    (default). When set, long HE mid-breaths become ~max_pause; freed time
    becomes trailing silence — next-utterance anchors stay put.
    """
    if max_pause <= 0 or len(plan) < 2:
        return 0.0
    total = 0.0
    for j in range(len(plan) - 1):
        gap = float(plan[j + 1]["start"]) - float(plan[j]["end"])
        if gap < min_gap_to_compact:
            continue
        shift = gap - max_pause
        if shift <= 0.02:
            continue
        for k in range(j + 1, len(plan)):
            plan[k]["start"] = round(float(plan[k]["start"]) - shift, 3)
            plan[k]["end"] = round(float(plan[k]["end"]) - shift, 3)
        plan[j]["pause_after"] = round(max_pause, 3)
        total += shift
    return float(total)


def coalesce_speech_runs(
    plan: list[dict],
    *,
    max_gap: float = SPEECH_RUN_GAP_SEC,
) -> list[dict]:
    """Merge adjacent phrases with gap < max_gap into one TTS synthesis unit.

    Hard truncation can only happen at real pause/segment boundaries, not
    mid-sentence between nearly-contiguous ASR phrases.
    """
    if len(plan) <= 1:
        return plan
    out: list[dict] = []
    for p in plan:
        cur = dict(p)
        if out:
            prev = out[-1]
            gap = float(cur["start"]) - float(prev["end"])
            prev_pause = float(prev.get("pause_after") or 0.0)
            if max(gap, prev_pause) < max_gap:
                prev_he = (prev.get("text_he") or prev.get("text") or "").rstrip()
                cur_he = (cur.get("text_he") or cur.get("text") or "").rstrip()
                # Keep sentence-final interjections as their own TTS unit.
                if prev_he.endswith((".", "!", "?", "…", "؟")) and cur_he.endswith(
                    (".", "!", "?", "…", "؟")
                ):
                    out.append(cur)
                    continue
                # Coalesce into previous run.
                prev_en = (prev.get("text") or "").strip()
                cur_en = (cur.get("text") or "").strip()
                prev["text"] = f"{prev_en} {cur_en}".strip()
                if (prev.get("text_he") or "").strip() or (cur.get("text_he") or "").strip():
                    prev["text_he"] = (
                        f"{(prev.get('text_he') or '').strip()} "
                        f"{(cur.get('text_he') or '').strip()}"
                    ).strip()
                prev_ten = (prev.get("text_en") or "").strip()
                cur_ten = (cur.get("text_en") or "").strip()
                if prev_ten or cur_ten:
                    prev["text_en"] = f"{prev_ten} {cur_ten}".strip()
                prev["end"] = float(cur["end"])
                prev["pause_after"] = float(cur.get("pause_after") or 0.0)
                continue
        out.append(cur)
    return out


def slack_phrase_window_for_rate(
    *,
    p_start: float,
    p_end: float,
    raw_sec: float,
    left_lim: float,
    right_lim: float,
    max_rate: float = FIT_MAX_RATE,
    min_rate: float = FIT_MIN_RATE,
    max_early: float = SLACK_MAX_EARLY_SEC,
    max_late: float = SLACK_MAX_LATE_SEC,
) -> tuple[float, float, str]:
    """Borrow a little adjacent pause only when EN would need extreme speed.

    Returns (new_start, new_end, note). Hebrew ASR times stay put when the
    natural rate is already inside [min_rate, max_rate].
    """
    slot = max(0.25, p_end - p_start)
    if raw_sec <= 0.05:
        return p_start, p_end, ""
    natural = raw_sec / slot
    start, end = p_start, p_end
    note = ""
    if natural > max_rate:
        # EN too long for HE slot — extend into following pause a bit.
        need = raw_sec / max_rate - slot
        late = min(max_late, max(0.0, right_lim - end), max(0.0, need))
        if late >= 0.04:
            end = end + late
            note = f"slack +{late:.2f}s end (EN long)"
    elif natural < min_rate:
        # EN too short for a tight window — optional slight early start so the
        # spoken line can sit nearer the lips without slowing to fill.
        # (Trailing silence already handles long HE slots.)
        if slot < 1.4:
            early = min(max_early, max(0.0, start - left_lim), 0.15)
            if early >= 0.04:
                start = start - early
                note = f"slack −{early:.2f}s start (short slot)"
    return start, end, note


def fit_exact_window(
    src: Path,
    dst: Path,
    target_sec: float,
    *,
    sample_rate: int = 44100,
    target_n: int | None = None,
    max_rate: float = FIT_MAX_RATE,
    min_rate: float = FIT_MIN_RATE,
    allow_overrun: bool = True,
) -> float:
    """Time-stretch toward the Hebrew slot with stable speaking pace.

    - Default: never slow below 1.0 (no slow-mo fill).
    - Callers may pass min_rate≈0.90 for slow HE slots only.
    - Keep rate=1.0 when only mildly long (≤ FIT_SPEEDUP_THRESHOLD) and overrun
      a little into the following pause — avoids slow/fast flip-flops.
    - Speed up only when clearly over-long; cap at max_rate and overrun.
    """
    target_sec = max(0.2, float(target_sec))
    if target_n is None:
        target_n = int(round(target_sec * sample_rate))
    target_n = max(1, int(target_n))
    actual = wav_duration(src)
    if actual <= 0.05:
        raise RuntimeError(f"Empty TTS clip: {src}")

    # rate>1 → faster / shorter; rate<1 → slower / longer
    natural = actual / (target_n / sample_rate)
    speedup_from = max(1.0, FIT_SPEEDUP_THRESHOLD)
    if allow_overrun and natural > max_rate:
        rate = max_rate
        exact = False
    elif natural > speedup_from:
        rate = min(natural, max_rate)
        exact = True
    elif natural < min_rate and min_rate < 1.0 - 1e-6:
        # Slow HE slot: mild stretch toward the window (not below min_rate).
        rate = max(natural, min_rate)
        exact = abs(rate - natural) > 0.02
    else:
        # Short, exact, or only slightly long: speak at natural pace.
        rate = 1.0
        exact = False

    work = src
    if abs(rate - 1.0) > 0.005:
        stretched = dst.with_name(dst.stem + "_stretch.wav")
        _atempo_to(src, stretched, rate, sample_rate=sample_rate)
        work = stretched

    audio, sr = sf.read(str(work), dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=-1).astype(np.float32)
    if sr != sample_rate:
        raise RuntimeError(f"Unexpected sample rate {sr} (want {sample_rate})")
    if exact and rate >= 1.0:
        audio = _remap_to_n(audio, target_n)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), audio, sample_rate)
    return float(rate)


def he_words_per_sec(he_text: str, slot_sec: float) -> float:
    words = len((he_text or "").split())
    return words / max(0.25, float(slot_sec))


def is_slow_he_slot(he_text: str, slot_sec: float, raw_en_sec: float) -> bool:
    """True when Hebrew is deliberate/slow and EN would finish early in the slot."""
    if float(slot_sec) < SLOW_SLOT_MIN_SEC:
        return False
    if raw_en_sec <= 0.05:
        return False
    if raw_en_sec / max(0.25, float(slot_sec)) >= SLOW_FILL_RATIO:
        return False
    return he_words_per_sec(he_text, slot_sec) < SLOW_HE_WPS


def pad_leading_silence_for_slow_slot(
    src: Path,
    dst: Path,
    *,
    slot_sec: float,
    sample_rate: int = 44100,
) -> float:
    """Insert leading silence so short EN sits later in a slow HE window.

    Returns seconds of lead silence added. Keeps speaking pace; only shifts
    where the clip begins inside the phrase.
    """
    audio, sr = sf.read(str(src), dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=-1).astype(np.float32)
    if sr != sample_rate:
        raise RuntimeError(f"Unexpected sample rate {sr} (want {sample_rate})")
    dur = len(audio) / float(sample_rate)
    leftover = max(0.0, float(slot_sec) - dur)
    lead = min(SLOW_LEAD_MAX_SEC, leftover * SLOW_LEAD_FRAC)
    if lead < 0.06:
        if src.resolve() != dst.resolve():
            sf.write(str(dst), audio, sample_rate)
        return 0.0
    pad = np.zeros(int(round(lead * sample_rate)), dtype=np.float32)
    # Do not exceed the slot after padding.
    max_n = max(1, int(round(float(slot_sec) * sample_rate)))
    out = np.concatenate([pad, audio])[:max_n]
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), out, sample_rate)
    return float(lead)


# Guard silence before a KEEP-original (usually English) onset so the dub
# does not step on the original speaker. Soft bleed into same-speaker gaps
# remains allowed when the next segment is also dubbed.
KEEP_YIELD_GUARD_SEC = 0.12
GAP_BLEED_SEC = 0.35


def assemble_on_hebrew_timeline(
    plan: list[dict],
    seg_start: float,
    seg_end: float,
    out_path: Path,
    *,
    sample_rate: int = 44100,
    hard_end: float | None = None,
    crossfade_sec: float = 0.06,
    next_is_keep: bool = False,
    gap_bleed_sec: float | None = None,
) -> Path:
    """Place each phrase clip at its (possibly compacted) start with soft mix.

    Clips may overrun a short phrase into the following pause, but not past
    `hard_end` (next segment start). Overlaps are crossfaded instead of
    overwritten so previous tails are not erased.

    When `next_is_keep` is True, disable gap bleed past the last phrase so the
    dub yields cleanly to the original-language speaker.
    """
    limit_t = float(hard_end) if hard_end is not None else float(seg_end)
    bleed = 0.0 if next_is_keep else (
        GAP_BLEED_SEC if gap_bleed_sec is None else float(gap_bleed_sec)
    )
    last_needed = float(seg_end)
    for phrase in plan:
        clip = Path(phrase.get("tts_fit") or "")
        if clip.is_file():
            last_needed = max(last_needed, float(phrase["start"]) + wav_duration(clip))
    last_needed = min(last_needed, limit_t)

    seg_a = int(round(seg_start * sample_rate))
    seg_b = int(round(last_needed * sample_rate))
    n = max(1, seg_b - seg_a)
    canvas = np.zeros(n, dtype=np.float32)
    fade_n = max(1, int(round(crossfade_sec * sample_rate)))

    for j, phrase in enumerate(plan):
        clip = Path(phrase["tts_fit"])
        if not clip.is_file():
            continue
        audio, sr = sf.read(str(clip), dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = np.mean(audio, axis=-1).astype(np.float32)
        if sr != sample_rate:
            tmp = clip.with_name(clip.stem + f"_{sample_rate}.wav")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(clip),
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    str(sample_rate),
                    "-ac",
                    "1",
                    str(tmp),
                ],
                check=True,
                capture_output=True,
            )
            audio, sr = sf.read(str(tmp), dtype="float32", always_2d=False)
            if getattr(audio, "ndim", 1) > 1:
                audio = np.mean(audio, axis=-1).astype(np.float32)

        p_a = int(round(float(phrase["start"]) * sample_rate))
        next_t = (
            float(plan[j + 1]["start"]) - 0.02
            if j + 1 < len(plan)
            else limit_t
        )
        # Soft budget: allow slight overrun into the next gap, but not past next onset.
        max_len = max(0, int(round(next_t * sample_rate)) - p_a)
        # Extra headroom into the pause between phrases before hard cut.
        # Disabled when the next segment is KEEP (would step on original speech).
        if j + 1 < len(plan) and bleed > 0:
            max_len = max(max_len, int(round((next_t + bleed) * sample_rate)) - p_a)
            max_len = min(max_len, int(round(limit_t * sample_rate)) - p_a)
        offset = p_a - seg_a
        if offset < 0:
            audio = audio[-offset:]
            offset = 0
        take = min(len(audio), max_len, n - offset)
        if take <= 0 or offset >= n:
            if len(audio) > max_len + sample_rate // 10:
                lost = (len(audio) - max(0, max_len)) / sample_rate
                print(
                    f"      WARN: truncating phrase {j} by {lost:.2f}s "
                    f"(would overrun next onset)",
                    file=sys.stderr,
                )
            continue
        chunk = audio[:take].copy()
        if take < len(audio) and take > int(0.08 * sample_rate):
            fade = min(int(0.08 * sample_rate), take // 4)
            chunk[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
            lost = (len(audio) - take) / sample_rate
            if lost > 0.08:
                print(
                    f"      WARN: soft-trim phrase {j} lost {lost:.2f}s",
                    file=sys.stderr,
                )
        # Crossfade mix into existing canvas (don't erase previous tails).
        dest = canvas[offset : offset + take]
        overlap = min(fade_n, take, len(dest))
        if overlap > 1 and float(np.max(np.abs(dest[:overlap]))) > 1e-4:
            ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
            dest[:overlap] = dest[:overlap] * (1.0 - ramp) + chunk[:overlap] * ramp
            if take > overlap:
                dest[overlap:] = chunk[overlap:]
        else:
            dest[:] = chunk
        canvas[offset : offset + take] = dest

    peak = float(np.max(np.abs(canvas))) if len(canvas) else 0.0
    if peak > 1.0:
        canvas /= peak
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), canvas, sample_rate)
    return out_path


def resolve_qwen_device(requested: str | None) -> str:
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_qwen_dtype(device: str) -> torch.dtype:
    if device.startswith("cuda"):
        return torch.bfloat16
    # float16 on MPS hits NaNs in the code predictor sampler; use float32.
    return torch.float32


def resolve_model_path(prefer: Path) -> str:
    if prefer.is_dir() and any(prefer.iterdir()):
        return str(prefer)
    return HUB_MODEL


class QwenVoiceCloneSynthesizer:
    """1.7B-Base zero-shot clone with optional prompt reuse per speaker."""

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        device: str | None = None,
        x_vector_only: bool = True,
    ) -> None:
        self.device = resolve_qwen_device(device)
        self.dtype = resolve_qwen_dtype(self.device)
        self.path = resolve_model_path(Path(model_path) if model_path else DEFAULT_MODEL)
        self.x_vector_only = bool(x_vector_only)
        self._model = None
        self._prompt_cache: dict[str, object] = {}

    def _load(self):
        from qwen_tts import Qwen3TTSModel

        if self._model is not None:
            return self._model

        print(f"Loading Qwen3-TTS 1.7B-Base (voice clone) from {self.path}", file=sys.stderr)
        kwargs: dict = {
            "dtype": self.dtype,
            "low_cpu_mem_usage": False,
        }
        if self.device.startswith("cuda"):
            kwargs["attn_implementation"] = "sdpa"
            kwargs["device_map"] = self.device
        else:
            kwargs["device_map"] = "cpu"

        model = Qwen3TTSModel.from_pretrained(self.path, **kwargs)
        if not self.device.startswith("cuda"):
            model.model.to(self.device)
            tok = getattr(model.model, "speech_tokenizer", None)
            if tok is not None and hasattr(tok, "to"):
                tok.to(self.device)
            model.device = torch.device(self.device)
        self._model = model
        return model

    def synthesize(
        self,
        text: str,
        *,
        ref_audio: Path,
        ref_text: str,
        language: str = "English",
        out_wav: Path,
        prompt_key: str | None = None,
        target_sec: float | None = None,
    ) -> Path:
        model = self._load()
        out_wav.parent.mkdir(parents=True, exist_ok=True)

        voice_clone_prompt = None
        if prompt_key is not None and prompt_key in self._prompt_cache:
            voice_clone_prompt = self._prompt_cache[prompt_key]
        elif prompt_key is not None:
            voice_clone_prompt = model.create_voice_clone_prompt(
                ref_audio=str(ref_audio),
                ref_text=ref_text if not self.x_vector_only else None,
                x_vector_only_mode=self.x_vector_only,
            )
            self._prompt_cache[prompt_key] = voice_clone_prompt

        # Cooler sampling → fewer Hebrew-phonology slips on English.
        speak = prepare_english_tts_text(text)
        gen_kwargs = {
            "max_new_tokens": estimate_max_new_tokens(speak, target_sec),
            "do_sample": True,
            "temperature": 0.55,
            "top_p": 0.85,
            "top_k": 30,
            "repetition_penalty": 1.08,
            "subtalker_dosample": True,
            "subtalker_temperature": 0.55,
            "subtalker_top_p": 0.85,
            "subtalker_top_k": 30,
        }
        if voice_clone_prompt is not None:
            wavs, sr = model.generate_voice_clone(
                text=speak,
                language=language,
                voice_clone_prompt=voice_clone_prompt,
                **gen_kwargs,
            )
        else:
            wavs, sr = model.generate_voice_clone(
                text=speak,
                language=language,
                ref_audio=str(ref_audio),
                ref_text=ref_text if not self.x_vector_only else None,
                x_vector_only_mode=self.x_vector_only,
                **gen_kwargs,
            )

        audio = np.asarray(wavs[0], dtype=np.float32)
        sf.write(str(out_wav), audio, sr)
        return out_wav


def synthesize_segments_qwen(
    segments: list[dict],
    vocals: Path,
    workdir: Path,
    *,
    language: str = "English",
    model_path: str | Path | None = None,
    device: str | None = None,
    x_vector_only: bool = True,
    reuse_speaker_prompt: bool = True,
    selected_indices: set[int] | None = None,
    merge_pauses: bool = True,
    max_pause: float = 0.75,
    max_dub_pause: float = DEFAULT_MAX_DUB_PAUSE,
    shorten_en_fn=None,
    speaker_bank: dict | None = None,
) -> list[dict]:
    from inference.segment_merge import (
        clamp_phrase_timeline,
        dedupe_repeated_sentences,
        dedupe_segment_text_fields,
        merge_same_speaker_segments,
    )

    if merge_pauses:
        before = len(segments)
        segments[:] = merge_same_speaker_segments(segments, max_pause=max_pause)
        print(
            f"Merged same-speaker pauses: {before} → {len(segments)} utterances "
            f"(max_pause={max_pause}s)",
            file=sys.stderr,
        )

    tts_dir = workdir / "tts_clips"
    ref_dir = workdir / "tts_refs"
    tts_dir.mkdir(parents=True, exist_ok=True)
    media_duration = wav_duration(vocals)
    syn = QwenVoiceCloneSynthesizer(
        model_path=model_path or DEFAULT_MODEL,
        device=device,
        x_vector_only=x_vector_only,
    )
    mode = "x_vector_only" if x_vector_only else "icl"
    print(
        f"Qwen3-TTS 1.7B-Base zero-shot clone ({mode}) language={language} "
        f"reuse_speaker_prompt={reuse_speaker_prompt} max_dub_pause={max_dub_pause}s",
        file=sys.stderr,
    )

    bank = speaker_bank or {}
    speakers_map = bank.get("speakers") if isinstance(bank.get("speakers"), dict) else bank
    bank_paths: dict[str, Path] = {}
    for k, v in (speakers_map or {}).items():
        if isinstance(v, dict) and v.get("path"):
            bank_paths[str(k)] = Path(v["path"])
        elif isinstance(v, (str, Path)):
            bank_paths[str(k)] = Path(v)

    for i, seg in enumerate(segments):
        dedupe_segment_text_fields(seg)
        start = float(seg["start"])
        end = float(seg["end"])
        target = max(end - start, 0.4)
        lang = seg.get("language") or "he"
        keep_original = bool(seg.get("keep_original", lang != "he"))

        next_start = media_duration
        next_is_keep = False
        if i + 1 < len(segments):
            nxt = segments[i + 1]
            nxt_phrases = nxt.get("phrases") or []
            next_start = float(
                (nxt_phrases[0].get("start") if nxt_phrases else None)
                or nxt.get("start")
                or media_duration
            )
            nxt_lang = nxt.get("language") or "he"
            next_is_keep = bool(nxt.get("keep_original", nxt_lang != "he"))

        prev_end = 0.0
        if i > 0:
            prev_end = float(segments[i - 1].get("end") or 0.0)

        if keep_original:
            max_end = max(start + 0.4, next_start - 0.05)
            source_mix = workdir / "source.wav"
            probe_src = source_mix if source_mix.is_file() else vocals
            # Pull start back to true speech onset (symmetric to extend_end).
            # Fixes late KEEP-English starts from coarse gap-recovery windows.
            pulled = pull_start_to_energy(
                probe_src,
                start,
                end,
                min_start=prev_end + 0.04,
                max_pull=1.25 if lang == "en" else 0.65,
                rms_thresh=0.016 if lang == "en" else 0.04,
            )
            if pulled < start - 0.04:
                print(
                    f"  KEEP [{seg['speaker_id']}] {lang} "
                    f"snap start {start:.2f} → {pulled:.2f}s (energy onset)",
                    file=sys.stderr,
                )
                start = pulled
            if lang == "en":
                extended_end = extend_end_by_energy(
                    probe_src,
                    start,
                    end,
                    max_end=max_end,
                    pad=0.18,
                    rms_thresh=0.016,
                    bridge_gap_sec=4.5,
                    max_extend=min(8.0, max(0.5, max_end - end)),
                )
            else:
                extended_end = extend_end_by_energy(
                    probe_src,
                    start,
                    end,
                    max_end=max_end,
                    pad=0.06,
                    rms_thresh=0.045,
                    bridge_gap_sec=0.12,
                    max_extend=0.28,
                )
            if extended_end > end + 0.05:
                print(
                    f"  KEEP [{seg['speaker_id']}] {lang} "
                    f"slot {start:.1f}-{end:.1f}s → extend end to {extended_end:.1f}s",
                    file=sys.stderr,
                )
                end = extended_end
            else:
                print(
                    f"  KEEP [{seg['speaker_id']}] {lang} "
                    f"slot {start:.1f}-{end:.1f}s (original audio)",
                    file=sys.stderr,
                )
            asr_end = float(seg.get("end") or end)
            seg["start"] = round(start, 3)
            seg["end"] = round(end, 3)
            seg["duration"] = round(end - start, 3)
            if lang == "en" and end > asr_end + 0.35:
                refreshed = _transcribe_keep_english(vocals, start, end)
                if refreshed:
                    refreshed = dedupe_repeated_sentences(
                        re.sub(r"\s+", " ", refreshed).strip()
                    )
                    seg["text"] = refreshed
                    seg["text_en"] = refreshed
                    print(f"    KEEP text ← {refreshed[:80]}", file=sys.stderr)
            # Always rebuild KEEP as a single phrase (never glue onto phrases[-1]).
            keep_text = (seg.get("text_en") or seg.get("text") or "").strip()
            keep_text = dedupe_repeated_sentences(keep_text)
            seg["text"] = keep_text
            seg["text_en"] = keep_text
            seg["phrases"] = [
                {
                    "text": keep_text,
                    "text_en": keep_text,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "pause_after": 0.0,
                    "speaker_id": seg.get("speaker_id"),
                }
            ]
            target = max(end - start, 0.4)
            raw = tts_dir / f"seg_{i:02d}_orig.wav"
            fitted = tts_dir / f"seg_{i:02d}_fit.wav"
            keep_src = source_mix if source_mix.is_file() else vocals
            extract_wav_slice(keep_src, start, end, raw, sample_rate=44100)
            place_in_slot(raw, fitted, target, sample_rate=44100)
            seg["keep_uses_source"] = bool(keep_src == source_mix)
            seg.pop("tts_text", None)
            seg["tts_raw"] = str(raw)
            seg["tts_fit"] = str(fitted)
            seg["tts_speed_used"] = 1.0
            seg["keep_original"] = True
            seg["tts_engine"] = "keep_original"
            continue

        if selected_indices is not None and i not in selected_indices:
            fitted = seg.get("tts_fit")
            if not fitted or not Path(fitted).is_file():
                candidate = tts_dir / f"seg_{i:02d}_fit.wav"
                if candidate.is_file():
                    seg["tts_fit"] = str(candidate)
                    print(
                        f"  REATTACH [{seg['speaker_id']}] existing {candidate.name}",
                        file=sys.stderr,
                    )
            continue

        # Preserve original Hebrew before phrase_plan mutates text fields.
        orig_he = (seg.get("text") or "").strip()
        orig_he_phrases = [
            (p.get("text") or "").strip() for p in (seg.get("phrases") or [])
        ]

        plan = phrase_plan(seg)
        if not plan:
            continue
        plan = clamp_phrase_timeline(plan)
        before_runs = len(plan)
        plan = coalesce_speech_runs(plan)
        if len(plan) != before_runs:
            print(
                f"  coalesce speech runs: {before_runs} → {len(plan)} "
                f"(gap < {SPEECH_RUN_GAP_SEC:.2f}s)",
                file=sys.stderr,
            )

        yield_guard = KEEP_YIELD_GUARD_SEC if next_is_keep else 0.02
        if next_is_keep:
            print(
                f"  yield to KEEP next @ {next_start:.2f}s "
                f"(guard {yield_guard:.2f}s)",
                file=sys.stderr,
            )

        for j, phrase in enumerate(plan):
            p_start = float(phrase["start"])
            p_end = float(phrase["end"])
            if j == 0:
                pulled = pull_start_to_energy(
                    vocals,
                    p_start,
                    p_end,
                    min_start=prev_end + 0.04,
                    max_pull=0.85,
                    rms_thresh=0.04,
                )
                if pulled < p_start - 0.04:
                    print(
                        f"  SNAP start [{seg['speaker_id']}] "
                        f"{p_start:.2f} → {pulled:.2f}s (energy onset)",
                        file=sys.stderr,
                    )
                    p_start = pulled
            hard_cap = (
                float(plan[j + 1]["start"]) - 0.02
                if j + 1 < len(plan)
                else min(max(p_end, p_start + 0.25), next_start - yield_guard)
            )
            if j + 1 < len(plan):
                p_end = min(p_end, hard_cap)
            elif next_is_keep:
                # Cap the last phrase so EN cannot spill into KEEP onset.
                p_end = min(p_end, hard_cap)
            phrase["start"] = p_start
            phrase["end"] = max(p_end, p_start + 0.25)

        compacted = compact_phrase_timeline_gaps(plan, max_pause=max_dub_pause)
        if compacted >= 0.15:
            print(
                f"  compact pauses −{compacted:.2f}s (cap {max_dub_pause:.2f}s)",
                file=sys.stderr,
            )

        he_start = float(plan[0]["start"])
        he_end = float(plan[-1]["end"])
        start = he_start
        end = he_end
        seg["start"] = round(start, 3)
        seg["end"] = round(end, 3)
        target = max(end - start, 0.4)

        print(
            f"  Qwen clone [{seg['speaker_id']}] "
            f"Hebrew window {start:.1f}-{end:.1f}s ({len(plan)} phrase(s))",
            file=sys.stderr,
        )

        spk_key = str(seg.get("speaker_id") or f"seg_{i}")
        bank_ref = bank_paths.get(spk_key)
        first_ref_meta: dict | None = None
        rates: list[float] = []
        for j, phrase in enumerate(plan):
            text = phrase["text"]
            p_start = float(phrase["start"])
            p_end = float(phrase["end"])
            p_target = max(p_end - p_start, 0.25)

            he_text = (phrase.get("text_he") or "").strip()
            if not he_text and j < len(orig_he_phrases):
                he_text = orig_he_phrases[j]
            he_text = he_text or "um"
            phrase["text_he"] = he_text

            if bank_ref is not None and bank_ref.is_file():
                ref = {
                    "path": str(bank_ref.resolve()),
                    "ref_text": he_text,
                    "start": float(seg.get("ref_start") or p_start),
                    "end": float(seg.get("ref_end") or p_end),
                }
            else:
                ref = build_qwen_phrase_ref(
                    {"start": p_start, "end": p_end, "text": he_text},
                    seg,
                    i,
                    j,
                    vocals,
                    ref_dir,
                    media_duration,
                    all_segments=segments,
                )
            if first_ref_meta is None:
                first_ref_meta = ref

            speak = prepare_english_tts_text(text)
            tok = estimate_max_new_tokens(speak, p_target)
            if speak != text:
                print(f"    tts-text: {speak[:70]}…", file=sys.stderr)
            print(
                f"    phrase {j} HE [{p_start:.2f}-{p_end:.2f}] "
                f"slot={p_target:.2f}s ref={ref['start']:.1f}-{ref['end']:.1f}s "
                f"max_tok={tok}: {text[:48]}…",
                file=sys.stderr,
            )
            chunk = tts_dir / f"seg_{i:02d}_p{j:02d}_qwen.wav"
            prompt_key = spk_key if reuse_speaker_prompt else None
            syn.synthesize(
                speak,
                ref_audio=Path(ref["path"]),
                ref_text=str(ref["ref_text"]),
                language=language,
                out_wav=chunk,
                prompt_key=prompt_key,
                target_sec=p_target,
            )
            phrase["tts_text"] = speak

            chunk_441 = tts_dir / f"seg_{i:02d}_p{j:02d}_raw441.wav"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(chunk),
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "44100",
                    "-ac",
                    "1",
                    str(chunk_441),
                ],
                check=True,
                capture_output=True,
            )
            trimmed = tts_dir / f"seg_{i:02d}_p{j:02d}_trim.wav"
            lead = trim_leading_silence(chunk_441, trimmed)
            fit_src = trimmed if lead >= 0.04 else chunk_441
            if lead >= 0.04:
                print(f"      trim lead silence {lead:.2f}s", file=sys.stderr)

            raw_sec = wav_duration(fit_src)
            natural = raw_sec / max(p_target, 0.25)
            # Prefer shorten-and-retry over hard truncation when the next
            # segment is KEEP-original OR the next onset is very close.
            is_last_phrase = j + 1 >= len(plan)
            gap_to_next = max(0.0, float(next_start) - float(p_end))
            tight_next = is_last_phrase and gap_to_next < TIGHT_NEXT_GAP_SEC
            en_words = max(1, len(speak.split()))
            he_words = max(1, len(he_text.split()))
            en_heavy = en_words > max(4, he_words * 3) and natural > FIT_MAX_RATE
            shorten_trigger = SHORTEN_RETRY_RATE
            if (next_is_keep and is_last_phrase) or tight_next:
                shorten_trigger = min(SHORTEN_RETRY_RATE, FIT_MAX_RATE)
            # Allow shorten even on sub-1.2s stubs when EN clearly overfills.
            min_slot_for_shorten = (
                0.35 if (en_heavy or tight_next or next_is_keep) else SHORTEN_MIN_SLOT_SEC
            )
            if (
                natural > shorten_trigger
                and shorten_en_fn is not None
                and p_target >= min_slot_for_shorten
            ):
                budget = max(6, int(p_target * SHORTEN_SYL_PER_SEC))
                try:
                    shorter = shorten_en_fn(
                        he_text=he_text,
                        en_text=speak,
                        duration=p_target,
                        syllable_budget=budget,
                    )
                except Exception as exc:
                    print(f"      shorten retry failed: {exc}", file=sys.stderr)
                    shorter = None
                if shorter and shorter.strip() and shorter.strip() != speak:
                    orig_words = max(1, len(speak.split()))
                    new_words = len(shorter.split())
                    drop = 1.0 - (new_words / orig_words)
                    if drop > SHORTEN_MAX_WORD_DROP:
                        print(
                            f"      shorten rejected (dropped {drop:.0%} words): "
                            f"{shorter[:70]}…",
                            file=sys.stderr,
                        )
                        shorter = None
                if shorter and shorter.strip() and shorter.strip() != speak:
                    print(
                        f"      shorten EN ({natural:.2f}× → retry): {shorter[:70]}…",
                        file=sys.stderr,
                    )
                    speak = prepare_english_tts_text(shorter)
                    phrase["text"] = speak
                    phrase["tts_text"] = speak
                    syn.synthesize(
                        speak,
                        ref_audio=Path(ref["path"]),
                        ref_text=str(ref["ref_text"]),
                        language=language,
                        out_wav=chunk,
                        prompt_key=prompt_key,
                        target_sec=p_target,
                    )
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-i",
                            str(chunk),
                            "-acodec",
                            "pcm_s16le",
                            "-ar",
                            "44100",
                            "-ac",
                            "1",
                            str(chunk_441),
                        ],
                        check=True,
                        capture_output=True,
                    )
                    lead = trim_leading_silence(chunk_441, trimmed)
                    fit_src = trimmed if lead >= 0.04 else chunk_441
                    raw_sec = wav_duration(fit_src)

            left_lim = (
                float(plan[j - 1]["end"]) + 0.08
                if j > 0
                else float(prev_end) + 0.04
            )
            right_lim = (
                float(plan[j + 1]["start"]) - 0.08
                if j + 1 < len(plan)
                else float(next_start) - yield_guard
            )
            # ASR end before slack/fit mutation — used to cap writeback.
            asr_end = float(phrase.get("end") or p_end)
            slot_max_rate = 1.12 if (p_end - p_start) < 1.2 else FIT_MAX_RATE
            # Before KEEP or a tight next onset: do not allow overrun slack.
            if (next_is_keep and is_last_phrase) or tight_next:
                slot_max_rate = max(slot_max_rate, FIT_MAX_RATE)
            p_start, p_end, slack_note = slack_phrase_window_for_rate(
                p_start=p_start,
                p_end=p_end,
                raw_sec=raw_sec,
                left_lim=left_lim,
                right_lim=right_lim,
                max_rate=slot_max_rate,
            )
            if slack_note:
                print(f"      {slack_note}", file=sys.stderr)
                phrase["start"] = round(p_start, 3)
                phrase["end"] = round(p_end, 3)
            p_target = max(p_end - p_start, 0.25)

            slow_slot = is_slow_he_slot(he_text, p_target, raw_sec)
            slot_min_rate = SLOW_MIN_RATE if slow_slot else FIT_MIN_RATE
            if slow_slot:
                print(
                    f"      slow-HE slot ({he_words_per_sec(he_text, p_target):.1f} w/s, "
                    f"EN fills {raw_sec / p_target:.0%}) → mild stretch + lead pad",
                    file=sys.stderr,
                )

            fitted_phrase = tts_dir / f"seg_{i:02d}_p{j:02d}_fit.wav"
            p_a = int(round(p_start * 44100))
            p_b = int(round(p_end * 44100))
            rate = fit_exact_window(
                fit_src,
                fitted_phrase,
                p_target,
                target_n=max(1, p_b - p_a),
                max_rate=slot_max_rate,
                min_rate=slot_min_rate,
                allow_overrun=not (
                    (next_is_keep and is_last_phrase) or tight_next
                ),
            )
            if slow_slot:
                lead = pad_leading_silence_for_slow_slot(
                    fitted_phrase,
                    fitted_phrase,
                    slot_sec=p_target,
                    sample_rate=44100,
                )
                if lead >= 0.06:
                    print(f"      slow-HE lead silence {lead:.2f}s", file=sys.stderr)
            rates.append(rate)
            actual = wav_duration(fitted_phrase)
            overrun = actual - p_target
            note = (
                f"overrun +{overrun:.2f}s into pause"
                if overrun > 0.04
                else "exact slot"
            )
            print(
                f"      qwen raw={wav_duration(chunk_441):.2f}s → "
                f"{actual:.2f}s @ rate={rate:.3f} ({note})",
                file=sys.stderr,
            )
            phrase["tts_fit"] = str(fitted_phrase)
            phrase["tts_raw"] = str(chunk_441)
            phrase["ref_audio"] = ref["path"]
            phrase["tts_speed_used"] = round(rate, 3)
            if actual > p_target + 0.04:
                # Cap expansion for short HE slots — don't fill the whole gap
                # to the next onset (that caused 0.5s stubs → multi-second repeats).
                gap_budget = max(0.0, float(next_start) - asr_end - yield_guard)
                if p_target < 1.5:
                    max_over = min(SHORT_SLOT_OVERRUN_CAP_SEC, gap_budget * 0.4)
                else:
                    max_over = min(gap_budget, SLACK_MAX_LATE_SEC)
                phrase["end"] = round(
                    min(p_start + actual, asr_end + max_over, next_start - yield_guard),
                    3,
                )

        start = min(float(p["start"]) for p in plan)
        end = max(float(p["end"]) for p in plan)
        seg["start"] = round(start, 3)
        seg["end"] = round(end, 3)
        target = max(end - start, 0.4)

        fitted = tts_dir / f"seg_{i:02d}_fit.wav"
        assemble_on_hebrew_timeline(
            plan,
            start,
            end,
            fitted,
            sample_rate=44100,
            hard_end=next_start - yield_guard,
            next_is_keep=next_is_keep,
        )
        spoken = wav_duration(fitted)
        end = max(end, start + spoken)
        # Don't let short HE stubs expand to the next onset.
        asr_seg_end = max(float(p.get("end") or 0) for p in plan) if plan else end
        if (end - start) < 1.5:
            end = min(end, asr_seg_end + SHORT_SLOT_OVERRUN_CAP_SEC)
        seg["end"] = round(min(end, next_start - yield_guard), 3)
        seg["duration"] = round(float(seg["end"]) - start, 3)
        print(
            f"    timeline {spoken:.2f}s (Hebrew slot was {target:.2f}s, "
            f"{len(plan)} phrases @ HE offsets, "
            f"hard_end={next_start - yield_guard:.2f})",
            file=sys.stderr,
        )
        seg.pop("tts_text", None)
        seg["tts_raw"] = str(plan[0].get("tts_raw") or fitted)
        seg["tts_fit"] = str(fitted)
        seg["tts_speed_used"] = (
            round(sum(rates) / len(rates), 3) if rates else 1.0
        )
        seg["keep_original"] = False
        seg["tts_engine"] = "qwen3-tts-1.7b-base"
        seg["qwen_clone_mode"] = mode
        if first_ref_meta:
            seg["ref_audio"] = first_ref_meta["path"]
            seg["ref_start"] = first_ref_meta["start"]
            seg["ref_end"] = first_ref_meta["end"]

        # REPLACE phrases with the plan (never zip onto unmerged list).
        new_phrases: list[dict] = []
        for j, out_p in enumerate(plan):
            he = (out_p.get("text_he") or "").strip()
            if not he and j < len(orig_he_phrases):
                he = orig_he_phrases[j]
            row = {
                "text": he or orig_he,
                "text_en": dedupe_repeated_sentences(
                    (out_p.get("tts_text") or out_p.get("text") or "").strip()
                ),
                "start": float(out_p["start"]),
                "end": float(out_p["end"]),
                "pause_after": float(out_p.get("pause_after") or 0.0),
                "speaker_id": seg.get("speaker_id"),
            }
            if out_p.get("tts_fit"):
                row["tts_fit"] = out_p["tts_fit"]
            if out_p.get("tts_raw"):
                row["tts_raw"] = out_p["tts_raw"]
            if out_p.get("tts_speed_used") is not None:
                row["tts_speed_used"] = out_p["tts_speed_used"]
            new_phrases.append(row)
        if new_phrases:
            new_phrases[-1]["pause_after"] = 0.0
        seg["phrases"] = new_phrases
        if orig_he:
            seg["text"] = orig_he
        elif new_phrases:
            seg["text"] = " ".join(
                (p.get("text") or "").strip() for p in new_phrases if p.get("text")
            )
        en_bits = [
            (p.get("text_en") or "").strip() for p in new_phrases if p.get("text_en")
        ]
        if en_bits:
            seg["text_en"] = dedupe_repeated_sentences(" ".join(en_bits))

    return segments


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen3-TTS 1.7B-Base zero-shot clone.")
    p.add_argument("workdir", type=Path, help="Run dir with translated_segments.json + vocals.")
    p.add_argument("--language", default="English")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument(
        "--icl",
        action="store_true",
        help="Use ICL (ref_audio + Hebrew ref_text) instead of x-vector-only.",
    )
    p.add_argument(
        "--no-reuse-speaker-prompt",
        action="store_true",
        help="Rebuild clone prompt per phrase instead of once per speaker (default: reuse).",
    )
    p.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    p.add_argument("--max-pause", type=float, default=1.0)
    p.add_argument(
        "--max-dub-pause",
        type=float,
        default=DEFAULT_MAX_DUB_PAUSE,
        help="Cap intra-utterance silence (0=preserve HE pauses, default). "
        "Set e.g. 0.7 to compact long mid-utterance silences.",
    )
    p.add_argument("--tts-segments", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    workdir = args.workdir
    translated = workdir / "translated_segments.json"
    segments_path = translated if translated.is_file() else workdir / "segments.json"
    if not segments_path.is_file():
        raise SystemExit(f"Missing {segments_path}")
    payload = json.loads(segments_path.read_text(encoding="utf-8"))
    segments = payload.get("segments") or []
    vocals = Path(payload.get("vocals") or workdir / "vocals.wav")
    if not vocals.is_file():
        raise SystemExit(f"Missing vocals: {vocals}")

    bank_path = workdir / "tts_refs" / "speaker_bank.json"
    speaker_bank = None
    if bank_path.is_file():
        speaker_bank = json.loads(bank_path.read_text(encoding="utf-8"))

    synthesize_segments_qwen(
        segments,
        vocals,
        workdir,
        language=args.language,
        model_path=args.model,
        device=None if args.device == "auto" else args.device,
        x_vector_only=not args.icl,
        reuse_speaker_prompt=not args.no_reuse_speaker_prompt,
        selected_indices=(
            {int(v) for v in args.tts_segments.split(",")} if args.tts_segments else None
        ),
        max_pause=args.max_pause,
        max_dub_pause=args.max_dub_pause,
        speaker_bank=speaker_bank,
    )
    payload["segments"] = segments
    payload["tts_engine"] = "qwen3-tts-1.7b-base"
    out = workdir / "translated_segments.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
