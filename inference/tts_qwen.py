#!/usr/bin/env python3
"""Qwen3-TTS 1.7B-Base zero-shot voice clone for DubbingQwen (Phase 4).

Clones from each segment's own vocal window (~3–4.5s Hebrew ref), then
synthesizes English and gently fits toward the Hebrew [start,end] (rate
0.90–1.15). Never trims mid-sentence — overrun is resolved later by the
elastic gap-closing packer (``plan_dub_placement``). Units split only at
pauses ≥1.2s. Clone verify + regenerate for garbled/chipmunk clips.

Model: Qwen/Qwen3-TTS-12Hz-1.7B-Base (https://arxiv.org/abs/2601.15621)
  - x_vector_only=True (default): speaker embedding from ref audio
  - x_vector_only=False: ICL with ref_audio + Hebrew ref_text
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from inference.tts_f5 import (
    atempo_chain,
    extend_end_by_energy,
    extract_wav_slice,
    place_in_slot,
    pull_start_to_energy,
    wav_duration,
)

# Gentle per-unit fit toward the Hebrew [start,end]. Never trim — overrun is
# handled by the elastic packer (plan_dub_placement) with a higher hard cap.
FIT_MAX_RATE_GENTLE = 1.15
FIT_MAX_RATE = FIT_MAX_RATE_GENTLE  # alias: normal fit never exceeds gentle
FIT_MAX_RATE_HARD = 1.50  # packer-only uniform run speed-up (never trim)
FIT_MIN_RATE = 0.90
FIT_SPEEDUP_THRESHOLD = 1.0  # always fit toward the HE window when short
UNIT_SPLIT_PAUSE_SEC = 1.2  # only split TTS units at real mid-utterance silence
# Keep big pauses as real silence, but cap so a long hole doesn't feel "stuck".
MAX_MID_SILENCE_SEC = 1.2
# Long same-speaker turns without a big pause still need time anchors or EN drifts.
MAX_UNIT_SEC = 10.0
# Shorten before audible speedup at/above FIT_MAX_RATE_GENTLE.
SHORTEN_SYL_PER_SEC = 3.6
SHORTEN_MAX_WORD_DROP = 0.40
# Legacy CLI default (pause compaction opt-in; units use UNIT_SPLIT_PAUSE_SEC).
DEFAULT_MAX_DUB_PAUSE = 0.0
# Kept for tests / callers that still import the old helpers (no longer hot path).
FIT_MAX_RATE_SHORT = FIT_MAX_RATE_GENTLE
SHORT_SLOT_SEC = 1.50
SHORTEN_MIN_SLOT_SEC = 1.0
SHORT_SLOT_OVERRUN_CAP_SEC = 0.20
TIGHT_NEXT_GAP_SEC = 0.30
SPEECH_RUN_GAP_SEC = 0.18
SLOW_HE_WPS = 2.0
SLOW_FILL_RATIO = 0.82
SLOW_MIN_RATE = 0.90
SLOW_LEAD_FRAC = 0.40
SLOW_LEAD_MAX_SEC = 0.85
SLOW_SLOT_MIN_SEC = 1.40
SLACK_MAX_EARLY_SEC = 0.15
SLACK_MAX_LATE_SEC = 0.25
# Qwen often emits ~0.4–0.6s hush before speech; strip before fit/place.
LEAD_SILENCE_MAX_TRIM_SEC = 0.85
LEAD_SILENCE_PAD_SEC = 0.03
# Clone stability: reject garbled / chipmunk clips and regenerate.
CLONE_MIN_OVERLAP = 0.35
CLONE_MAX_TRIES = 3
CLONE_MIN_SEC_PER_WORD = 0.18  # below this → likely chipmunk / truncated
CLONE_MAX_SEC_PER_WORD = 0.95  # above this → likely stalled / garbled drawl
# Boundary-aware trim helper (kept for tests; no longer used on the fit hot path).
BOUNDARY_TRIM_LOOKBACK_SEC = 0.45
BOUNDARY_TRIM_RMS = 0.012
BOUNDARY_TRIM_MIN_SILENCE_SEC = 0.04
# Intra-segment unit packing: small natural pause when closing unit gaps.
UNIT_INTER_GAP_SEC = 0.08
# Preserve real HE dramatic stops; only close micro-gaps below this.
UNIT_PRESERVE_GAP_SEC = 0.45

REPO_ROOT = Path(__file__).resolve().parents[1]

# region agent log
_DEBUG_LOG_PATH = REPO_ROOT / ".cursor" / "debug-5e8424.log"


def _debug_event(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict,
) -> None:
    try:
        payload = {
            "sessionId": "5e8424",
            "runId": os.environ.get("DUB_DEBUG_RUN", "diagnostic"),
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
# endregion

DEFAULT_MODEL = REPO_ROOT / "models" / "Qwen3-TTS-12Hz-1.7B-Base"
HUB_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
# Longer clean refs → stronger x-vector identity. Concat up to REF_CONCAT_SEC.
REF_TARGET_SEC = 4.5
REF_MIN_SEC = 2.5
REF_MAX_SEC = 6.0
REF_CONCAT_SEC = 5.5
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


# Own-segment clone ref quality gate: usable if long enough and not near-silent.
REF_OWN_MIN_RMS = 0.018
# HF/RMS score above this → own window is unusable; allow same-speaker fallback.
REF_OWN_MAX_NOISE_SCORE = 8.0


def _speaker_own_windows(
    seg: dict,
    phrase: dict,
) -> list[tuple[float, float]]:
    """Unit/phrase-local windows only for per-unit zero-shot cloning.

    Do **not** add the full parent segment as a candidate: scoring that let every
    unit in a multi-unit turn pick the same mid-segment ref (identical
    seg_N_p00/p01/p02 refs — 1:24 and 1:31 same-voice bug).
    """
    del seg  # parent span must not compete with the unit window
    p_start = float(phrase["start"])
    p_end = float(phrase["end"])
    # Slight pad inside the unit only (clamp to phrase bounds after).
    a = p_start
    b = p_end
    if b - a < 0.3:
        mid = 0.5 * (p_start + p_end)
        a = mid - 0.2
        b = mid + 0.2
    windows = [(a, b)]
    out: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for wa, wb in windows:
        key = (round(wa, 3), round(wb, 3))
        if wb - wa < 0.15 or key in seen:
            continue
        seen.add(key)
        out.append((wa, wb))
    return out


def _speaker_same_id_fallback_windows(
    seg: dict,
    phrase: dict,
    all_segments: list[dict] | None,
) -> list[tuple[float, float]]:
    """Other HE turns with the same speaker_id (never a different speaker)."""
    spk = seg.get("speaker_id")
    if not spk or not all_segments:
        return []
    p_start = float(phrase["start"])
    p_end = float(phrase["end"])
    seg_a = float(seg.get("start") or p_start)
    seg_b = float(seg.get("end") or p_end)
    mid = 0.5 * (p_start + p_end)
    nearby: list[tuple[float, float, float]] = []
    for other in all_segments:
        if other.get("speaker_id") != spk:
            continue
        if other.get("keep_original"):
            continue
        if (other.get("language") or "he") != "he":
            continue
        a, b = float(other["start"]), float(other["end"])
        if b - a < REF_MIN_SEC:
            continue
        if abs(a - seg_a) < 0.05 and abs(b - seg_b) < 0.05:
            continue
        nearby.append((abs(0.5 * (a + b) - mid), a, b))
    nearby.sort(key=lambda t: t[0])
    out: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for _dist, a, b in nearby[:4]:
        key = (round(a, 3), round(b, 3))
        if key in seen:
            continue
        seen.add(key)
        out.append((a, b))
    return out


def _speaker_search_windows(
    seg: dict,
    phrase: dict,
    all_segments: list[dict] | None,
) -> list[tuple[float, float]]:
    """Candidate clone refs: own segment first, then same-speaker_id only.

    Cross-speaker nearest-in-time windows are never candidates — scoring those
    let a cleaner foreign turn steal identity (0:50 SPEAKER_02→SPEAKER_06 bug).
    """
    windows = list(_speaker_own_windows(seg, phrase))
    for a, b in _speaker_same_id_fallback_windows(seg, phrase, all_segments):
        key = (round(a, 3), round(b, 3))
        if any((round(x, 3), round(y, 3)) == key for x, y in windows):
            continue
        windows.append((a, b))
    return windows


def _score_ref_window(
    vocals: Path,
    ref_start: float,
    ref_end: float,
    *,
    tmp: Path,
) -> tuple[float, float]:
    """Return (noise_score, rms) for a candidate ref slice."""
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
    return score, rms


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

    Prefer this **unit/phrase** window only. Only if that local audio fails a
    quality gate (near-silent / catastrophic HF) fall back to other turns of
    the same ``speaker_id``. Never clone from a different speaker, and never
    score the full parent segment against a sub-unit.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    he_text = (phrase.get("text") or seg.get("text") or "um").strip() or "um"
    tmp = out_dir / f"_score_{seg_index:02d}_{phrase_index:02d}.wav"

    def _best_in(windows: list[tuple[float, float]]) -> tuple[float, float, float, float] | None:
        # score, start, end, rms
        best: tuple[float, float, float, float] | None = None
        for search_a, search_b in windows:
            # Target at most the search span — do not ask for longer than the unit.
            span = max(0.25, search_b - search_a)
            target = min(REF_TARGET_SEC, max(min(REF_MIN_SEC, span), span))
            ref_start, ref_end = pick_cleanest_ref_window(
                vocals,
                search_a,
                search_b,
                media_duration=media_duration,
                target_sec=target,
            )
            # Clamp to the unit search window so we never inherit outside audio.
            ref_start = max(ref_start, search_a)
            ref_end = min(ref_end, search_b)
            if ref_end - ref_start < 0.15:
                continue
            score, rms = _score_ref_window(vocals, ref_start, ref_end, tmp=tmp)
            if best is None or score < best[0]:
                best = (score, ref_start, ref_end, rms)
        return best

    own_best = _best_in(_speaker_own_windows(seg, phrase))
    chosen: tuple[float, float, float, float] | None = own_best
    # Prefer any usable local audio (even short units) over a remote same-id turn.
    own_ok = (
        own_best is not None
        and own_best[3] >= REF_OWN_MIN_RMS
        and own_best[0] <= REF_OWN_MAX_NOISE_SCORE
    )
    if not own_ok:
        fb = _best_in(
            _speaker_same_id_fallback_windows(seg, phrase, all_segments)
        )
        if fb is not None and (chosen is None or fb[0] < chosen[0]):
            chosen = fb

    if chosen is None:
        p_start = float(phrase["start"])
        p_end = float(phrase["end"])
        ref_start, ref_end = p_start, p_end
        if ref_end - ref_start < 0.25:
            mid = 0.5 * (p_start + p_end)
            ref_start = max(0.0, mid - 0.2)
            ref_end = min(media_duration, mid + 0.2)
    else:
        _, ref_start, ref_end, _rms = chosen
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
    # Keep the org's proper name (do not rewrite to "the Nusra Front").
    (re.compile(r"\bJahbat\s+al-?Nusra\b", re.I), "Jabhat al-Nusra"),
    (re.compile(r"\bJabhat\s+a-Nusra\b", re.I), "Jabhat al-Nusra"),
    (re.compile(r"\bthe\s+Nusra\s+Front\b", re.I), "Jabhat al-Nusra"),
    (re.compile(r"\bAl-?Qaeda\b", re.I), "Al-Qaeda"),
    (re.compile(r"\bAl\s+Qaeda\b", re.I), "Al-Qaeda"),
    (re.compile(r"\bSheikha\s+Moza\b", re.I), "Sheikha Moza"),
    (re.compile(r"\bSheikh\s+Moza\b", re.I), "Sheikha Moza"),
]


def prepare_english_tts_text(text: str) -> str:
    """Light cleanup for HE→EN voice clone (do not respell common words).

    Earlier funds→fundz made 1.7B say "funs"; keep natural English orthography.
    Map hard proper nouns to forms Qwen actually voices in list enumerations.
    Hard-strip non-Latin scripts and caption chrome so garbage never reaches synth.
    """
    out = (text or "").strip()
    if not out:
        return out
    # Caption chrome + non-Latin (CJK/HE/AR) must never reach the voice clone.
    out = re.sub(
        r"(?:>>+\s*)|"
        r"\[(?:music|applause|מוזיקה|שירה|קהל)[^\]]*\]|"
        r"\((?:music|applause)\)",
        " ",
        out,
        flags=re.IGNORECASE,
    )
    cleaned: list[str] = []
    for ch in out:
        if ch.isalpha() and not ("A" <= ch <= "Z" or "a" <= ch <= "z"):
            cleaned.append(" ")
        else:
            cleaned.append(ch)
    out = re.sub(r"\s+", " ", "".join(cleaned)).strip(" ,.;:-")
    if not out:
        return out
    for pat, repl in _TTS_NAME_RESPPELL:
        out = pat.sub(repl, out)
    # Undo any stale fundz/fundd hacks left in curated JSON.
    out = re.sub(r"\bfundz\b", "funds", out, flags=re.I)
    out = re.sub(r"\bfundd\b", "fund", out, flags=re.I)
    if out and out[-1] not in ".!?":
        out += "."
    return out


def _transcribe_keep_english(
    vocals: Path,
    start: float,
    end: float,
    *,
    prefer_source: Path | None = None,
) -> str | None:
    """Re-ASR an extended KEEP English window (picks up trailing clauses)."""
    try:
        from faster_whisper import WhisperModel
        from inference.extract_pipeline import resolve_whisper_model
    except Exception:
        return None
    if end - start < 0.4:
        return None
    src = prefer_source if prefer_source is not None and prefer_source.is_file() else vocals
    tmp = vocals.parent / "tts_clips" / "_keep_en_probe.wav"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    extract_wav_slice(src, start, end, tmp, sample_rate=16000)
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
        if re.search(r"[\u0590-\u05FF]", text) and not re.search(r"[A-Za-z]{3,}", text):
            return None
        # Strip residual Hebrew tokens if mixed.
        if re.search(r"[\u0590-\u05FF]", text):
            text = re.sub(r"[\u0590-\u05FF]+", " ", text)
            text = re.sub(r"\s+", " ", text).strip(" ,.-")
            if len(text.split()) < 3:
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
                from inference.segment_merge import needs_object_continuation

                object_cont = needs_object_continuation(prev_he, cur_he)
                # Keep sentence-final interjections as their own TTS unit,
                # except object-continuation stubs that must stay one unit.
                if (
                    not object_cont
                    and prev_he.endswith((".", "!", "?", "…", "؟"))
                    and cur_he.endswith((".", "!", "?", "…", "؟"))
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
    max_rate: float = FIT_MAX_RATE_GENTLE,
    min_rate: float = FIT_MIN_RATE,
    allow_overrun: bool = True,
    hard_max_rate: float = FIT_MAX_RATE_HARD,
    pad_short: bool = True,
) -> float:
    """Gently fit toward the Hebrew speaking window — never trim speech.

    Escalation:
      1. rate in [min_rate, max_rate] (default 0.90–1.15) toward HE duration
      2. if still too long → keep full length (overrun); packer resolves later
      3. if short → stretch down to min_rate, then pad to ``target_n``

    ``allow_overrun`` is kept for API compatibility; overrun is always kept
    (never boundary-trim / hard-window-fade). ``hard_max_rate`` is unused here
    (reserved for the elastic packer's uniform run speed-up).
    """
    del hard_max_rate  # packer-only; per-unit fit stays gentle + never-cut
    target_sec = max(0.2, float(target_sec))
    if target_n is None:
        target_n = int(round(target_sec * sample_rate))
    target_n = max(1, int(target_n))
    actual = wav_duration(src)
    if actual <= 0.05:
        raise RuntimeError(f"Empty TTS clip: {src}")

    # rate>1 → faster / shorter; rate<1 → slower / longer
    natural = actual / (target_n / sample_rate)
    if natural > max_rate:
        # Cap at gentle max; keep leftover overrun (never trim).
        rate = max_rate
    elif natural >= FIT_SPEEDUP_THRESHOLD:
        rate = min(max(natural, 1.0), max_rate)
    elif min_rate <= natural < 1.0:
        rate = max(natural, min_rate)
    else:
        # Clearly short (natural < min_rate): stretch to min_rate, then pad.
        rate = min_rate

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
    pre_canvas_n = len(audio)
    canvas_action = "unchanged"
    pad_samples = 0
    removed_samples = 0

    # Pad short audio to the HE canvas; never trim long audio.
    if len(audio) < target_n and pad_short:
        pad_samples = target_n - len(audio)
        pad = np.zeros(pad_samples, dtype=np.float32)
        canvas_action = "pad"
        fade = min(int(0.04 * sample_rate), max(1, len(audio) // 10))
        if fade > 1 and len(audio) > fade:
            audio = audio.copy()
            audio[-fade:] *= np.linspace(1.0, 0.85, fade, dtype=np.float32)
        audio = np.concatenate([audio, pad])
    elif len(audio) > target_n:
        # Keep full speech; elastic packer closes gaps / speeds the run later.
        canvas_action = "keep_overrun"
        removed_samples = 0

    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), audio, sample_rate)

    # region agent log
    _debug_event(
        "H4",
        "inference/tts_qwen.py:fit_exact_window",
        "Record exact-fit padding or trimming",
        {
            "sourceFile": src.name,
            "targetFile": dst.name,
            "rawSec": round(actual, 4),
            "targetSec": round(target_n / sample_rate, 4),
            "naturalRate": round(natural, 4),
            "chosenRate": round(float(rate), 4),
            "preCanvasSec": round(pre_canvas_n / sample_rate, 4),
            "outputSec": round(len(audio) / sample_rate, 4),
            "action": canvas_action,
            "padSec": round(pad_samples / sample_rate, 4),
            "removedSec": round(removed_samples / sample_rate, 4),
            "allowOverrun": True,
        },
    )
    # endregion
    return float(rate)


def boundary_aware_trim(
    audio: np.ndarray,
    max_n: int,
    *,
    sample_rate: int = 44100,
    lookback_sec: float = BOUNDARY_TRIM_LOOKBACK_SEC,
    rms_thresh: float = BOUNDARY_TRIM_RMS,
    min_silence_sec: float = BOUNDARY_TRIM_MIN_SILENCE_SEC,
) -> np.ndarray:
    """Trim to ≤max_n at a silence/energy boundary — never a raw mid-word chop.

    Looks back from max_n for a quiet region; soft-fades out there. Falls back
    to a soft-fade at max_n only when no silence is found.
    """
    if max_n <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(audio) <= max_n:
        return audio.astype(np.float32, copy=False)

    hop = max(1, sample_rate // 100)  # 10ms
    lookback = int(round(lookback_sec * sample_rate))
    min_sil = max(1, int(round(min_silence_sec * sample_rate)))
    search_start = max(0, max_n - lookback)
    # Find the latest quiet run that ends near max_n.
    best_cut = max_n
    i = max_n - hop
    while i >= search_start:
        chunk = audio[max(0, i) : i + hop]
        rms = float(np.sqrt(np.mean(chunk**2) + 1e-12))
        if rms < rms_thresh:
            # Expand left through contiguous quiet.
            j = i
            while j - hop >= search_start:
                prev = audio[j - hop : j]
                if float(np.sqrt(np.mean(prev**2) + 1e-12)) >= rms_thresh:
                    break
                j -= hop
            quiet_len = i + hop - j
            if quiet_len >= min_sil:
                # Cut at the start of the quiet region (end of last speech).
                best_cut = max(1, j)
                break
        i -= hop

    take = min(best_cut, max_n, len(audio))
    chunk = audio[:take].copy()
    fade = min(int(0.06 * sample_rate), max(1, take // 8))
    if fade > 1:
        chunk[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return chunk.astype(np.float32)


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

    Legacy helper retained for tests; utterance-level TTS no longer uses it.
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
    max_n = max(1, int(round(float(slot_sec) * sample_rate)))
    out = np.concatenate([pad, audio])[:max_n]
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), out, sample_rate)
    return float(lead)


def coarse_split_en_by_weights(en: str, group_weights: list[float | int]) -> list[str]:
    """Split one EN string into N chunks by weight, preferring nearby punctuation.

    Weights should reflect each HE group's share of the utterance (speech
    duration or caption-word count). Punctuation snaps only when close to the
    target fraction — a wide snap used to dump almost all EN into the second
    unit after a mid-utterance comma (0:50 dead-air bug).
    """
    n = len(group_weights)
    if n <= 0:
        return []
    en = (en or "").strip()
    if n == 1 or not en:
        return [en] if n == 1 else [""] * n
    weights = [max(0.5, float(w)) for w in group_weights]
    total_w = float(sum(weights))
    # Prefer sentence ends; commas only as a secondary snap.
    sentence_cuts = [m.end() for m in re.finditer(r"(?<=[.!?…])\s+", en)]
    comma_cuts = [m.end() for m in re.finditer(r"[,;:—–]\s+", en)]
    words = en.split()
    if not words:
        return [en] * n

    chunks: list[str] = []
    cursor_char = 0
    assigned_words = 0
    dangling = {
        "the", "a", "an", "and", "or", "but", "of", "to", "with", "for",
        "in", "on", "at", "by", "from", "as", "is", "are", "was", "were",
        "when", "but", "that", "which", "who", "into", "onto", "about",
        "than", "then", "if", "while", "because", "so", "very",
        "she", "he", "it", "they", "we", "i",
        # Mid-NP modifiers — never end a unit on "Western |" before "conscience".
        "western", "eastern", "northern", "southern", "middle", "clear",
        "islamic", "palestinian", "israeli", "american", "european",
        "human", "modern", "extreme", "central",
    }
    # Never leave a cut that starts the next chunk with these (mid-PP / mid-clause).
    leading_glue = {
        "for", "to", "of", "in", "with", "that", "which", "who", "than",
        "into", "onto", "about", "from", "by", "on", "at", "as", "and", "or",
        "but", "if", "while", "because", "is", "are", "was", "were", "has",
        "have", "had",
    }
    # Tight snap: ~10% of string or 10 chars — not 25%.
    snap_tol = max(10, int(0.10 * len(en)))

    def _pick_cut(target_char: int) -> int | None:
        pool_sent = [c for c in sentence_cuts if cursor_char < c < len(en) - 2]
        if pool_sent:
            best = min(pool_sent, key=lambda c: abs(c - target_char))
            if abs(best - target_char) <= snap_tol:
                return best
        pool_comma = [c for c in comma_cuts if cursor_char < c < len(en) - 2]
        if pool_comma:
            best = min(pool_comma, key=lambda c: abs(c - target_char))
            # Commas must be even closer — avoid early "with the weapon," cuts.
            if abs(best - target_char) <= max(8, snap_tol // 2):
                return best
        return None

    for gi in range(n - 1):
        target_frac = sum(weights[: gi + 1]) / total_w
        target_char = int(round(target_frac * len(en)))
        cut = _pick_cut(target_char)
        if cut is not None:
            chunk = en[cursor_char:cut].strip().rstrip(",")
            if chunk:
                chunks.append(chunk)
                cursor_char = cut
                assigned_words = len(en[:cursor_char].split())
                continue
        remaining_groups = n - gi - 1
        need = max(1, round(len(words) * weights[gi] / total_w))
        need = min(need, max(1, len(words) - assigned_words - remaining_groups))
        # Prefer a nearby sentence/comma word boundary within ±2 words of need.
        for k in range(assigned_words + 1, len(words) - remaining_groups + 1):
            if words[k - 1].endswith((".", "!", "?", "…")):
                if abs((k - assigned_words) - need) <= max(2, need // 3):
                    need = k - assigned_words
                    break
        # Grow (don't shrink) past dangling ends and mid-PP starts so we never
        # cut "comfortable | for the Western" or "colors that | are very…".
        max_need = len(words) - assigned_words - remaining_groups
        guard = 0
        while assigned_words + need < len(words) and need < max_need and guard < 12:
            last = words[assigned_words + need - 1].strip(".,;:!?\"'").lower()
            nxt = words[assigned_words + need].strip(".,;:!?\"'").lower()
            if last in dangling or nxt in leading_glue:
                need += 1
                guard += 1
                continue
            break
        chunk_words = words[assigned_words : assigned_words + need]
        chunks.append(" ".join(chunk_words).strip())
        assigned_words += need
        cursor_char = len(" ".join(words[:assigned_words]))
        if cursor_char < len(en) and en[cursor_char : cursor_char + 1] == " ":
            cursor_char += 1
    tail = en[cursor_char:].strip() if cursor_char < len(en) else ""
    if not tail:
        tail = " ".join(words[assigned_words:]).strip()
    chunks.append(tail)
    while len(chunks) < n:
        chunks.append("")
    return chunks[:n]


def _group_split_weights(
    groups: list[list[dict]],
    *,
    youtube_words: list[dict] | None = None,
) -> list[float]:
    """Weight each unit by speech duration; prefer caption-word counts when present."""
    speech_durs = [
        max(0.25, float(g[-1]["end"]) - float(g[0]["start"])) for g in groups
    ]
    if youtube_words:
        counts: list[float] = []
        for g in groups:
            a, b = float(g[0]["start"]), float(g[-1]["end"])
            n = sum(
                1
                for w in youtube_words
                if a - 0.05 <= float(w.get("start") or 0.0) <= b + 0.05
                and (w.get("text") or "").strip()
            )
            counts.append(float(max(1, n)))
        if sum(counts) >= len(groups):
            return counts
    # Blend duration with HE word count so short dense clauses aren't starved.
    word_counts = [
        float(max(1, sum(len((p.get("text") or "").split()) for p in g)))
        for g in groups
    ]
    return [0.65 * d + 0.35 * w for d, w in zip(speech_durs, word_counts)]


def _coalesce_object_continuation_sents(sents: list[str]) -> list[str]:
    """Keep 'אני לא מכירה.' + 'אף אישה…' as one HE sentence unit for splitting."""
    if len(sents) <= 1:
        return sents
    try:
        from inference.segment_merge import needs_object_continuation
    except Exception:
        return sents
    out: list[str] = []
    i = 0
    while i < len(sents):
        cur = sents[i]
        if i + 1 < len(sents) and needs_object_continuation(cur, sents[i + 1]):
            out.append(f"{cur.rstrip()} {sents[i + 1].lstrip()}".strip())
            i += 2
            continue
        out.append(cur)
        i += 1
    return out


def _slice_group_at_times(
    group: list[dict],
    cut_times: list[float],
    *,
    max_unit_sec: float = MAX_UNIT_SEC,
    he_sents: list[str] | None = None,
) -> list[list[dict]]:
    """Turn absolute cut times into synthetic single-phrase subgroups."""
    g_start = float(group[0]["start"])
    g_end = float(group[-1]["end"])
    he = " ".join((p.get("text") or "").strip() for p in group if p.get("text")).strip()
    sents = he_sents or ([he] if he else [""])
    sents = _coalesce_object_continuation_sents(sents)

    bounds = [g_start] + sorted(
        t for t in cut_times if g_start + 0.8 < t < g_end - 0.8
    )
    bounds.append(g_end)
    merged = [bounds[0]]
    for t in bounds[1:-1]:
        if t - merged[-1] < 1.2:
            continue
        merged.append(t)
    if g_end - merged[-1] < 1.2 and len(merged) > 1:
        merged[-1] = g_end
    else:
        merged.append(g_end)

    final = [merged[0]]
    for t in merged[1:]:
        # When HE sentences guide the cuts, do not insert anonymous max_unit
        # chops — those re-split object-continuation spans and scramble text.
        if he_sents is None:
            while t - final[-1] > max_unit_sec + 0.5:
                final.append(final[-1] + max_unit_sec)
        final.append(t)

    out: list[list[dict]] = []
    n = max(1, len(final) - 1)
    for i in range(n):
        a, b = final[i], final[i + 1]
        lo = int(round(i * len(sents) / n))
        hi = int(round((i + 1) * len(sents) / n))
        text = " ".join(sents[lo:hi]).strip() or he
        out.append(
            [
                {
                    "text": text,
                    "start": a,
                    "end": b,
                    "pause_after": 0.0,
                }
            ]
        )
    return out if out else [group]


def _split_long_phrase_group(
    group: list[dict],
    *,
    max_unit_sec: float = MAX_UNIT_SEC,
    youtube_words: list[dict] | None = None,
) -> list[list[dict]]:
    """Sub-split a long phrase group into ≤max_unit_sec units.

    Prefer existing phrase boundaries; otherwise cut at caption word gaps or
    HE sentence ends. General fix for long single-speaker turns that otherwise
    drift when spoken as one natural-pace clip.
    """
    if not group:
        return []
    g_start = float(group[0]["start"])
    g_end = float(group[-1]["end"])
    try:
        from inference.segment_merge import needs_object_continuation
    except Exception:
        needs_object_continuation = None  # type: ignore

    # Soft allowance for object-continuation spans (stub + object ≈ 12s).
    soft_max = max_unit_sec + 3.0
    he_join = " ".join((p.get("text") or "").strip() for p in group if p.get("text")).strip()
    if (
        needs_object_continuation
        and "אני לא" in he_join
        and any(tok in he_join for tok in ("אף ", "שום ", "את "))
        and g_end - g_start <= soft_max
    ):
        return [group]
    if g_end - g_start <= max_unit_sec + 0.25:
        return [group]

    # 1) Pack existing phrases into bins ≤ max_unit_sec.
    if len(group) > 1:
        bins: list[list[dict]] = [[]]
        for p in group:
            if not bins[-1]:
                bins[-1].append(p)
                continue
            prev = bins[-1][-1]
            object_cont = bool(
                needs_object_continuation
                and needs_object_continuation(
                    (prev.get("text") or "").strip(),
                    (p.get("text") or "").strip(),
                )
            )
            span = float(p["end"]) - float(bins[-1][0]["start"])
            # Object continuations always stay with their stub, even if slightly long.
            if object_cont:
                bins[-1].append(p)
                continue
            if span > max_unit_sec and len(bins[-1]) >= 1:
                bins.append([p])
            else:
                bins[-1].append(p)
        if len(bins) > 1:
            # Recurse per bin — never discard packing and re-slice the whole span
            # (that re-introduced object-continuation cuts via max_unit chops).
            out: list[list[dict]] = []
            for b in bins:
                out.extend(
                    _split_long_phrase_group(
                        b,
                        max_unit_sec=max_unit_sec,
                        youtube_words=youtube_words,
                    )
                )
            return out

    # 2) Caption word gaps ≥ 0.45s inside the span → synthetic cuts.
    if youtube_words:
        words = sorted(
            (
                w
                for w in youtube_words
                if g_start - 0.05 <= float(w.get("start") or 0) <= g_end + 0.05
                and (w.get("text") or "").strip()
            ),
            key=lambda w: float(w["start"]),
        )
        cut_times: list[float] = []
        for i in range(len(words) - 1):
            gap = float(words[i + 1]["start"]) - float(words[i]["start"])
            if gap >= 0.45:
                t = float(words[i + 1]["start"])
                if g_start + 1.5 < t < g_end - 1.5:
                    cut_times.append(t)
        if cut_times:
            return _slice_group_at_times(
                group, cut_times, max_unit_sec=max_unit_sec
            )

    # 3) HE sentence ends → proportional time slices.
    he = " ".join((p.get("text") or "").strip() for p in group if p.get("text")).strip()
    sents = [s.strip() for s in re.split(r"(?<=[.!?…׃])\s+", he) if s.strip()]
    sents = _coalesce_object_continuation_sents(sents)
    if len(sents) < 2:
        n = max(2, int(round((g_end - g_start) / max_unit_sec)))
        step = (g_end - g_start) / n
        cuts = [g_start + step * k for k in range(1, n)]
        return _slice_group_at_times(group, cuts, max_unit_sec=max_unit_sec)

    weights = [max(1, len(s.split())) for s in sents]
    total = float(sum(weights))
    cuts = []
    acc = 0.0
    for w in weights[:-1]:
        acc += w
        cuts.append(g_start + (acc / total) * (g_end - g_start))
    return _slice_group_at_times(
        group, cuts, max_unit_sec=max_unit_sec, he_sents=sents
    )


def split_utterance_into_units(
    seg: dict,
    *,
    split_pause: float = UNIT_SPLIT_PAUSE_SEC,
    max_mid_silence: float = MAX_MID_SILENCE_SEC,
    max_unit_sec: float = MAX_UNIT_SEC,
) -> list[dict]:
    """Split a dub utterance into TTS units at big pauses and long-span cuts.

    Small HE breaths collapse. Pauses >= split_pause become (capped) silence.
    Spans longer than max_unit_sec are further split using caption gaps /
    sentence ends so natural-pace EN cannot drift across a whole monologue.
    """
    phrases = list(seg.get("phrases") or [])
    if phrases:
        try:
            from inference.segment_merge import merge_short_phrases

            phrases = merge_short_phrases(phrases)
            seg["phrases"] = phrases
        except Exception:
            pass
    seg_en = (seg.get("text_en") or "").strip()
    seg_he = (seg.get("text") or "").strip()
    if not phrases:
        if not seg_en and not seg_he:
            return []
        return [
            {
                "text_en": seg_en or seg_he,
                "text_he": seg_he,
                "start": float(seg.get("source_start") or seg["start"]),
                "end": float(seg.get("source_end") or seg["end"]),
                "source_start": float(seg.get("source_start") or seg["start"]),
                "source_end": float(seg.get("source_end") or seg["end"]),
                "pause_after": 0.0,
            }
        ]

    groups: list[list[dict]] = [[]]
    for i, p in enumerate(phrases):
        if groups[-1]:
            prev = groups[-1][-1]
            gap = float(p["start"]) - float(prev["end"])
            pause = float(prev.get("pause_after") or 0.0)
            if max(gap, pause) >= split_pause:
                groups.append([])
        groups[-1].append(p)

    yt_words = seg.get("youtube_words")
    if not isinstance(yt_words, list):
        yt_words = None

    expanded: list[list[dict]] = []
    for g in groups:
        expanded.extend(
            _split_long_phrase_group(
                g, max_unit_sec=max_unit_sec, youtube_words=yt_words
            )
        )
    groups = expanded

    weights = _group_split_weights(groups, youtube_words=yt_words)

    phrase_en_chunks = [
        " ".join(
            (p.get("text_en") or "").strip() for p in g if p.get("text_en")
        ).strip()
        for g in groups
    ]
    def _en_incomplete_tail(text: str) -> bool:
        words = (text or "").strip().split()
        if not words:
            return False
        last = words[-1].strip(".,;:!?\"'").lower()
        return last in {
            "the", "a", "an", "and", "or", "but", "of", "to", "with", "for",
            "in", "on", "at", "by", "from", "as", "is", "are", "was", "were",
            "when", "if", "while", "because", "that", "which", "who",
            "western", "eastern", "northern", "southern", "middle", "clear",
            "islamic", "palestinian", "israeli", "american", "european",
            "human", "modern", "extreme", "central",
        }

    broken_phrase_en = any(
        re.match(
            r"^(is|are|was|were|has|have|had)\s+",
            (c or "").strip(),
            flags=re.I,
        )
        for c in phrase_en_chunks
    ) or any(
        phrase_en_chunks[i + 1]
        and _en_incomplete_tail(phrase_en_chunks[i] or "")
        for i in range(len(phrase_en_chunks) - 1)
    )
    if phrase_en_chunks and all(phrase_en_chunks) and not broken_phrase_en:
        # Prefer already-aligned phrase EN (object-continuation merges, etc.).
        en_chunks = phrase_en_chunks
    elif seg_en:
        if len(groups) == 1:
            en_chunks = [seg_en]
        else:
            en_chunks = coarse_split_en_by_weights(seg_en, weights)
    else:
        en_chunks = phrase_en_chunks

    units: list[dict] = []
    for gi, g in enumerate(groups):
        he = " ".join((p.get("text") or "").strip() for p in g if p.get("text")).strip()
        en = (en_chunks[gi] if gi < len(en_chunks) else "").strip()
        if not en:
            en = he
        raw_pause = 0.0
        if gi + 1 < len(groups):
            raw_pause = max(
                0.0,
                float(groups[gi + 1][0]["start"]) - float(g[-1]["end"]),
            )
        pause_after = min(raw_pause, float(max_mid_silence)) if raw_pause else 0.0
        src_a = float(g[0].get("source_start") or g[0]["start"])
        src_b = float(g[-1].get("source_end") or g[-1]["end"])
        units.append(
            {
                "text_en": en,
                "text_he": he or seg_he,
                "start": src_a,
                "end": src_b,
                "source_start": src_a,
                "source_end": src_b,
                "pause_after": round(pause_after, 3),
                "raw_pause_after": round(raw_pause, 3),
            }
        )

    for gi in range(len(units) - 1):
        raw = float(units[gi].get("raw_pause_after") or 0.0)
        kept = float(units[gi].get("pause_after") or 0.0)
        shrink = raw - kept
        if shrink <= 0.04:
            continue
        # Compact placement gaps only — keep source_* as ASR anchors.
        for uj in range(gi + 1, len(units)):
            units[uj]["start"] = round(float(units[uj]["start"]) - shrink, 3)
            units[uj]["end"] = round(float(units[uj]["end"]) - shrink, 3)

    # region agent log
    _debug_event(
        "H2,H3,H4",
        "inference/tts_qwen.py:split_utterance_into_units",
        "Capture ASR phrases and resulting TTS units",
        {
            "speaker": seg.get("speaker_id"),
            "segmentStart": seg.get("source_start", seg.get("start")),
            "segmentEnd": seg.get("source_end", seg.get("end")),
            "phraseCount": len(phrases),
            "groupCount": len(groups),
            "phrases": [
                {
                    "start": p.get("start"),
                    "end": p.get("end"),
                    "sourceStart": p.get("source_start"),
                    "sourceEnd": p.get("source_end"),
                    "pauseAfter": p.get("pause_after"),
                    "heWords": len((p.get("text") or "").split()),
                    "enWords": len((p.get("text_en") or "").split()),
                }
                for p in phrases
            ],
            "units": [
                {
                    "start": unit.get("start"),
                    "end": unit.get("end"),
                    "sourceStart": unit.get("source_start"),
                    "sourceEnd": unit.get("source_end"),
                    "rawPauseAfter": unit.get("raw_pause_after"),
                    "pauseAfter": unit.get("pause_after"),
                    "heWords": len((unit.get("text_he") or "").split()),
                    "enWords": len((unit.get("text_en") or "").split()),
                }
                for unit in units
            ],
        },
    )
    # endregion
    return units

def _word_overlap_ratio(a: str, b: str) -> float:
    wa = set(re.findall(r"[a-z0-9]+", (a or "").lower()))
    wb = set(re.findall(r"[a-z0-9]+", (b or "").lower()))
    if not wa:
        return 0.0
    return len(wa & wb) / max(1, len(wa))


def _expected_speech_sec(text: str) -> float:
    words = max(1, len((text or "").split()))
    # Natural English ~2.5–3.5 words/sec; use mid for length sanity.
    return words / 3.0 + 0.25


def clone_length_ok(raw_sec: float, text: str) -> bool:
    """Reject chipmunk-short or stalled-long clips."""
    words = max(1, len((text or "").split()))
    if raw_sec <= 0.05:
        return False
    spw = raw_sec / words
    if spw < CLONE_MIN_SEC_PER_WORD:
        return False
    if spw > CLONE_MAX_SEC_PER_WORD and words >= 3:
        return False
    expected = _expected_speech_sec(text)
    # Chipmunk: finished in <40% of expected speaking time.
    if raw_sec < expected * 0.40:
        return False
    return True


_CLONE_ASR_MODEL = None

# Dedicated English ASR for clone content checks (not the Hebrew ivrit model).
CLONE_ASR_CANDIDATES = (
    "models/faster-whisper-base.en",
    "Systran/faster-whisper-base.en",
    "models/faster-whisper-tiny.en",
    "Systran/faster-whisper-tiny.en",
)


def _get_clone_asr_model():
    """Lazy-load a dedicated English Whisper model for clone verification."""
    global _CLONE_ASR_MODEL
    if _CLONE_ASR_MODEL is not None:
        return _CLONE_ASR_MODEL
    try:
        from faster_whisper import WhisperModel
        from inference.extract_pipeline import resolve_whisper_model

        last_exc: Exception | None = None
        for cand in CLONE_ASR_CANDIDATES:
            try:
                model_path = resolve_whisper_model(cand)
                # Prefer local dir when present under repo models/.
                local = Path(cand)
                if local.is_dir() and (local / "config.json").is_file():
                    model_path = str(local.resolve())
                _CLONE_ASR_MODEL = WhisperModel(
                    str(model_path), device="cpu", compute_type="auto"
                )
                print(
                    f"  clone verify: English ASR={model_path}",
                    file=sys.stderr,
                )
                return _CLONE_ASR_MODEL
            except Exception as exc:
                last_exc = exc
                continue
        raise RuntimeError(last_exc or "no English ASR candidate loaded")
    except Exception as exc:
        print(f"  clone verify: ASR unavailable ({exc})", file=sys.stderr)
        return None


def verify_clone_clip(
    wav_path: Path,
    target_en: str,
    *,
    min_overlap: float = CLONE_MIN_OVERLAP,
    skip_asr: bool = False,
) -> tuple[bool, float, str]:
    """Return (ok, overlap, asr_text). Length-fail → ok=False without ASR."""
    try:
        raw_sec = wav_duration(wav_path)
    except Exception:
        return False, 0.0, ""
    if not clone_length_ok(raw_sec, target_en):
        return False, 0.0, f"[len={raw_sec:.2f}s]"
    if skip_asr:
        return True, 1.0, ""
    model = _get_clone_asr_model()
    if model is None:
        # No ASR — accept on length alone.
        return True, 1.0, ""
    try:
        segs, _ = model.transcribe(
            str(wav_path),
            language="en",
            task="transcribe",
            word_timestamps=False,
            condition_on_previous_text=False,
            vad_filter=True,
        )
        asr = " ".join((s.text or "").strip() for s in segs).strip()
    except Exception as exc:
        print(f"  clone verify ASR failed: {exc}", file=sys.stderr)
        return False, 0.0, ""  # reject on ASR crash — don't accept garbage
    # Hebrew-script ASR of "English" TTS → garbled clone.
    he_chars = len(re.findall(r"[\u0590-\u05FF]", asr))
    lat_chars = len(re.findall(r"[A-Za-z]", asr))
    if he_chars >= 8 and he_chars >= lat_chars:
        return False, 0.0, asr[:120]
    if he_chars > 0 and lat_chars < 3:
        return False, 0.0, asr[:120]
    overlap = _word_overlap_ratio(target_en, asr)
    return overlap >= min_overlap, overlap, asr[:120]


def voice_similarity(clone_wav: Path, ref_wav: Path) -> float:
    """Cosine similarity of spectral fingerprints between clone and ref.

    Returns 0..1-ish; used as a soft identity check alongside content ASR.
    """
    try:
        from inference.speaker_bank import _segment_embedding_fallback

        a, sr_a = sf.read(str(clone_wav), dtype="float32", always_2d=False)
        b, sr_b = sf.read(str(ref_wav), dtype="float32", always_2d=False)
        if getattr(a, "ndim", 1) > 1:
            a = np.mean(a, axis=-1).astype(np.float32)
        if getattr(b, "ndim", 1) > 1:
            b = np.mean(b, axis=-1).astype(np.float32)
        ea = _segment_embedding_fallback(a, int(sr_a))
        eb = _segment_embedding_fallback(b, int(sr_b))
        na = float(np.linalg.norm(ea) + 1e-9)
        nb = float(np.linalg.norm(eb) + 1e-9)
        return float(np.dot(ea, eb) / (na * nb))
    except Exception:
        return 0.5  # neutral when embedding unavailable


# Guard silence before a KEEP-original (usually English) onset so the dub
# does not step on the original speaker. Soft bleed into same-speaker gaps
# remains allowed when the next segment is also dubbed.
KEEP_YIELD_GUARD_SEC = 0.12
# Soft bleed into the next dub onset (crossfade); never used before KEEP.
# Under never-cut assemble this is informational — units keep full length.
GAP_BLEED_SEC = 0.20


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
    """Place each unit clip on the HE timeline — never trim mid-sentence.

    Units keep their full synthesized length. Real HE stops
    (``orig_gap >= UNIT_PRESERVE_GAP_SEC``) are preserved at the source onset;
    only micro-gaps are closed to ``UNIT_INTER_GAP_SEC``. Pad tails are not
    source gaps — callers should pass ``pad_short=False`` on non-final units
    so short clips abut the next unit. Overlaps are crossfaded. ``hard_end`` /
    ``next_is_keep`` no longer truncate audio; the elastic packer speeds the
    whole segment clip when a KEEP anchor requires it.
    """
    del hard_end, gap_bleed_sec, next_is_keep  # packer owns yield / speed-up
    inter = float(UNIT_INTER_GAP_SEC)
    preserve = float(UNIT_PRESERVE_GAP_SEC)
    # Build placement plan: preserve real stops, close micro-gaps, keep full clips.
    place_abs: list[tuple[float, float, Path]] = []
    cursor = float(seg_start)
    for j, phrase in enumerate(plan):
        clip = Path(phrase.get("tts_fit") or "")
        if not clip.is_file():
            continue
        dur = wav_duration(clip)
        if dur <= 0.01:
            continue
        src_a = float(phrase["start"])
        if not place_abs:
            p0 = max(float(seg_start), src_a)
        else:
            prev_end = place_abs[-1][1]
            # Prefer the previous unit's HE source_end for the real gap; fall
            # back to its place end only when source timing is missing.
            prev_src_end = float(
                plan[j - 1].get("source_end")
                or plan[j - 1].get("end")
                or src_a
            )
            orig_gap = max(0.0, src_a - prev_src_end)
            if orig_gap >= preserve:
                # Real dramatic stop: park at the HE onset (keep the silence).
                p0 = max(prev_end, src_a)
            elif orig_gap > 1e-3:
                # Micro-gap: close to a tiny natural pause.
                gap = min(orig_gap, inter)
                p0 = max(prev_end + gap, min(src_a, prev_end + inter))
            else:
                p0 = prev_end
            if p0 < prev_end:
                p0 = prev_end
        p1 = p0 + dur
        place_abs.append((p0, p1, clip))
        cursor = p1

    if not place_abs:
        # Empty canvas of the HE window length so callers still get a file.
        n = max(1, int(round(max(0.25, float(seg_end) - float(seg_start)) * sample_rate)))
        canvas = np.zeros(n, dtype=np.float32)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), canvas, sample_rate)
        return out_path

    canvas_start = float(seg_start)
    canvas_end = max(float(seg_end), place_abs[-1][1])
    seg_a = int(round(canvas_start * sample_rate))
    seg_b = int(round(canvas_end * sample_rate))
    n = max(1, seg_b - seg_a)
    canvas = np.zeros(n, dtype=np.float32)
    fade_n = max(1, int(round(crossfade_sec * sample_rate)))

    for p0, _p1, clip in place_abs:
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

        offset = int(round(p0 * sample_rate)) - seg_a
        if offset < 0:
            audio = audio[-offset:]
            offset = 0
        take = min(len(audio), n - offset)
        if take <= 0 or offset >= n:
            # Grow canvas if a unit extends past the initial estimate.
            extra = offset + len(audio) - n
            if extra > 0:
                canvas = np.concatenate(
                    [canvas, np.zeros(extra, dtype=np.float32)]
                )
                n = len(canvas)
                take = len(audio)
            else:
                continue
        chunk = audio[:take].copy()
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
        seed: int | None = None,
        x_vector_only: bool | None = None,
        deterministic: bool = False,
    ) -> Path:
        model = self._load()
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        use_xvec = self.x_vector_only if x_vector_only is None else bool(x_vector_only)

        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        voice_clone_prompt = None
        # Cache key includes mode so ICL retry doesn't reuse an x-vector prompt.
        cache_key = None
        if prompt_key is not None:
            cache_key = f"{prompt_key}|{'xvec' if use_xvec else 'icl'}"
            if cache_key in self._prompt_cache:
                voice_clone_prompt = self._prompt_cache[cache_key]
            else:
                voice_clone_prompt = model.create_voice_clone_prompt(
                    ref_audio=str(ref_audio),
                    ref_text=ref_text if not use_xvec else None,
                    x_vector_only_mode=use_xvec,
                )
                self._prompt_cache[cache_key] = voice_clone_prompt

        # Cooler sampling → fewer Hebrew-phonology slips on English.
        # Deterministic fallback: greedy / near-greedy for stable acceptance.
        speak = prepare_english_tts_text(text)
        if deterministic:
            gen_kwargs = {
                "max_new_tokens": estimate_max_new_tokens(speak, target_sec),
                "do_sample": False,
                "temperature": 0.01,
                "top_p": 1.0,
                "top_k": 1,
                "repetition_penalty": 1.05,
                "subtalker_dosample": False,
                "subtalker_temperature": 0.01,
                "subtalker_top_p": 1.0,
                "subtalker_top_k": 1,
            }
        else:
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
                ref_text=ref_text if not use_xvec else None,
                x_vector_only_mode=use_xvec,
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
    verify_clone: bool = True,
) -> list[dict]:
    from inference.segment_merge import (
        dedupe_repeated_sentences,
        dedupe_segment_text_fields,
        merge_same_speaker_segments,
    )

    del max_dub_pause  # units use UNIT_SPLIT_PAUSE_SEC; kept for CLI compat

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
        f"reuse_speaker_prompt={reuse_speaker_prompt} "
        f"unit_split={UNIT_SPLIT_PAUSE_SEC:.1f}s verify_clone={verify_clone}",
        file=sys.stderr,
    )

    bank = speaker_bank or {}
    speakers_map = bank.get("speakers") if isinstance(bank.get("speakers"), dict) else bank
    bank_paths: dict[str, Path] = {}
    bank_meta: dict[str, dict] = {}
    for k, v in (speakers_map or {}).items():
        if isinstance(v, dict) and v.get("path"):
            bank_paths[str(k)] = Path(v["path"])
            bank_meta[str(k)] = v
        elif isinstance(v, (str, Path)):
            bank_paths[str(k)] = Path(v)

    # Track speakers that repeatedly fail clone verify → prefer alt refs / ICL.
    speaker_fail_counts: dict[str, int] = {}

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
            # Freeze ASR authority before energy snap/extend mutates placement.
            if seg.get("source_start") is None:
                seg["source_start"] = float(start)
            if seg.get("source_end") is None:
                seg["source_end"] = float(end)
            src_start = float(seg["source_start"])
            src_end = float(seg["source_end"])
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
            # Placement may soft-snap; immutable ASR window stays on source_*.
            seg["start"] = round(start, 3)
            seg["end"] = round(end, 3)
            seg["duration"] = round(end - start, 3)
            seg["source_start"] = round(src_start, 3)
            seg["source_end"] = round(src_end, 3)
            prefer_src = source_mix if source_mix.is_file() else None
            if lang == "en" and end > src_end + 0.35:
                refreshed = _transcribe_keep_english(
                    vocals, start, end, prefer_source=prefer_src
                )
                if refreshed:
                    refreshed = dedupe_repeated_sentences(
                        re.sub(r"\s+", " ", refreshed).strip()
                    )
                    seg["text"] = refreshed
                    seg["text_en"] = refreshed
                    print(f"    KEEP text ← {refreshed[:80]}", file=sys.stderr)
            keep_text = (seg.get("text_en") or seg.get("text") or "").strip()
            keep_text = dedupe_repeated_sentences(keep_text)
            # Suppress Hebrew-script captions on KEEP-English (mis-ASR / phonetic).
            # Latin islands like "World War II" must not keep a Hebrew caption.
            if lang == "en" and re.search(r"[\u0590-\u05FF]", keep_text):
                he_chars = len(re.findall(r"[\u0590-\u05FF]", keep_text))
                lat_chars = len(re.findall(r"[A-Za-z]", keep_text))
                mostly_hebrew = he_chars >= max(8, lat_chars * 2)
                if mostly_hebrew or not re.search(r"[A-Za-z]{3,}", keep_text):
                    refreshed = _transcribe_keep_english(
                        vocals, start, end, prefer_source=prefer_src
                    )
                    if refreshed:
                        keep_text = dedupe_repeated_sentences(refreshed)
                        print(
                            f"    KEEP caption ← EN ASR: {keep_text[:80]}",
                            file=sys.stderr,
                        )
                    else:
                        # Keep a non-empty placeholder so later runs do not
                        # filter this KEEP row out (empty text → dropped → silence).
                        keep_text = keep_text if keep_text else "[original audio]"
                        print(
                            "    KEEP caption suppressed (Hebrew-script, no EN ASR); "
                            "keeping original audio",
                            file=sys.stderr,
                        )
            if not keep_text:
                keep_text = "[original audio]"
            seg["text"] = keep_text
            seg["text_en"] = keep_text
            seg["phrases"] = [
                {
                    "text": keep_text,
                    "text_en": keep_text,
                    "start": round(src_start, 3),
                    "end": round(src_end, 3),
                    "source_start": round(src_start, 3),
                    "source_end": round(src_end, 3),
                    "tts_start": round(start, 3),
                    "spoken_end": round(end, 3),
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
            seg["tts_clip_sec"] = round(max(end - start, 0.4), 3)
            # Mix onset must match extract clock (energy snap), not later ASR
            # source_start — otherwise native speech plays late on the timeline.
            seg["place_start"] = round(start, 3)
            seg["place_end"] = round(end, 3)
            seg["place_speed"] = 1.0
            seg["place_drift"] = round(start - src_start, 3)
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

        # --- Utterance-level natural-pace dub ---
        orig_he = (seg.get("text") or "").strip()
        # Freeze ASR window before any energy-snap / fit mutation.
        if "source_start" not in seg:
            seg["source_start"] = float(seg.get("start") or 0.0)
            seg["source_end"] = float(seg.get("end") or 0.0)
        for p in seg.get("phrases") or []:
            if "source_start" not in p:
                p["source_start"] = float(p.get("start") or 0.0)
                p["source_end"] = float(p.get("end") or 0.0)
        source_a = float(seg["source_start"])
        source_b = float(seg["source_end"])
        units = split_utterance_into_units(seg)
        if not units:
            continue

        yield_guard = KEEP_YIELD_GUARD_SEC if next_is_keep else 0.02
        if next_is_keep:
            print(
                f"  yield to KEEP next @ {next_start:.2f}s "
                f"(guard {yield_guard:.2f}s)",
                file=sys.stderr,
            )

        # Pull first unit start to energy onset (placement only; source_* stays).
        u0_start = float(units[0]["start"])
        u0_end = float(units[0]["end"])
        pulled = pull_start_to_energy(
            vocals,
            u0_start,
            u0_end,
            min_start=prev_end + 0.04,
            max_pull=0.85,
            rms_thresh=0.04,
        )
        if pulled < u0_start - 0.04:
            print(
                f"  SNAP start [{seg['speaker_id']}] "
                f"{u0_start:.2f} → {pulled:.2f}s (energy onset)",
                file=sys.stderr,
            )
            units[0]["start"] = pulled

        start = float(units[0]["start"])
        end = float(units[-1]["end"])
        # Keep ASR start/end authoritative for mix/subs; track placement separately.
        seg["tts_start"] = round(start, 3)
        seg["tts_end"] = round(end, 3)
        seg["start"] = round(source_a, 3)
        seg["end"] = round(source_b, 3)

        print(
            f"  Qwen clone [{seg['speaker_id']}] "
            f"Hebrew window {source_a:.1f}-{source_b:.1f}s ({len(units)} unit(s))",
            file=sys.stderr,
        )

        spk_key = str(seg.get("speaker_id") or f"seg_{i}")
        bank_ref = bank_paths.get(spk_key)
        first_ref_meta: dict | None = None
        rates: list[float] = []
        place_plan: list[dict] = []

        for j, unit in enumerate(units):
            text_en = (unit.get("text_en") or "").strip()
            he_text = (unit.get("text_he") or orig_he or "um").strip() or "um"
            u_start = float(unit.get("source_start") or unit["start"])
            u_end = float(unit.get("source_end") or unit["end"])
            # Fit toward the HE lips window; never trim — overrun kept for packer.
            he_slot_sec = max(0.25, u_end - u_start)
            raw_pause = float(unit.get("raw_pause_after") or 0.0)
            # Always allow overrun: unit audio keeps full length; elastic packer
            # closes gaps / speeds the run before KEEP or locked anchors.
            allow_overrun = True
            if j + 1 < len(units):
                nxt_u = units[j + 1]
                hard_cap = float(nxt_u.get("source_start") or nxt_u["start"]) - 0.02
            else:
                hard_cap = (
                    next_start - yield_guard
                    if next_is_keep
                    else min(next_start + SHORT_SLOT_OVERRUN_CAP_SEC, media_duration)
                )
            hard_cap = max(hard_cap, u_start + he_slot_sec, u_start + 0.25)
            window_sec = he_slot_sec
            window_end = hard_cap

            # Build / pick clone refs — bank first (stable identity), then local.
            alt_refs: list[dict] = []
            bm = bank_meta.get(spk_key) or {}
            if bank_ref is not None and bank_ref.is_file():
                primary_ref = {
                    "path": str(bank_ref.resolve()),
                    "ref_text": str(bm.get("ref_text") or he_text),
                    "start": float(bm.get("start") or seg.get("ref_start") or u_start),
                    "end": float(bm.get("end") or seg.get("ref_end") or u_end),
                }
                alt_refs.append(primary_ref)
            phrase_ref = build_qwen_phrase_ref(
                {"start": u_start, "end": u_end, "text": he_text},
                seg,
                i,
                j,
                vocals,
                ref_dir,
                media_duration,
                all_segments=segments,
            )
            # Avoid duplicating the same path.
            if not alt_refs or Path(phrase_ref["path"]) != Path(alt_refs[0]["path"]):
                alt_refs.append(phrase_ref)
            if first_ref_meta is None:
                first_ref_meta = alt_refs[0]

            speak = prepare_english_tts_text(text_en)
            if not speak or not re.search(r"[A-Za-z]{2,}", speak):
                print(
                    f"      WARN: empty/non-English TTS text for unit {j}; skip",
                    file=sys.stderr,
                )
                seg["tts_failed"] = True
                continue
            print(
                f"    unit {j} HE [{u_start:.2f}-{u_end:.2f}] "
                f"window={window_sec:.2f}s: {speak[:60]}…",
                file=sys.stderr,
            )

            chunk = tts_dir / f"seg_{i:02d}_u{j:02d}_qwen.wav"
            chunk_441 = tts_dir / f"seg_{i:02d}_u{j:02d}_raw441.wav"
            trimmed = tts_dir / f"seg_{i:02d}_u{j:02d}_trim.wav"
            prompt_key = spk_key if reuse_speaker_prompt else None
            # Deterministic per-speaker seed → stable timbre across segments.
            import hashlib

            spk_seed = (
                int(hashlib.md5(spk_key.encode("utf-8")).hexdigest()[:8], 16) % 100_000
            )

            def _resample_trim(src_wav: Path) -> Path:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(src_wav),
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
                if lead >= 0.04:
                    print(f"      trim lead silence {lead:.2f}s", file=sys.stderr)
                    return trimmed
                return chunk_441

            best_fit_src: Path | None = None
            best_overlap = -1.0
            best_speak = speak
            best_score = -1.0
            accepted = False

            # Escalation for HE→EN clone:
            # default x_vector_only → try x-vector first (cross-lingual safe),
            # then ICL with aligned Hebrew ref_text. After clone failures, prefer ICL.
            fail_n = speaker_fail_counts.get(spk_key, 0)
            prefer_icl = (not x_vector_only) or fail_n >= 2
            mode_plan: list[tuple[int, bool]] = []  # (ref_idx, use_xvec)
            for ri in range(len(alt_refs)):
                if prefer_icl:
                    mode_plan.append((ri, False))  # ICL first
                    mode_plan.append((ri, True))
                else:
                    mode_plan.append((ri, True))
                    mode_plan.append((ri, False))

            n_tries = max(CLONE_MAX_TRIES, len(mode_plan)) if verify_clone else 1
            skip_icl = False  # set when ICL bleeds Hebrew into English target
            for attempt in range(n_tries):
                ref_i, use_xvec = mode_plan[attempt % len(mode_plan)]
                if skip_icl and not use_xvec:
                    continue
                ref = alt_refs[ref_i]
                seed = spk_seed + attempt * 97
                try:
                    syn.synthesize(
                        speak,
                        ref_audio=Path(ref["path"]),
                        ref_text=str(ref["ref_text"]),
                        language=language,
                        out_wav=chunk,
                        prompt_key=prompt_key,
                        target_sec=window_sec,
                        seed=seed,
                        x_vector_only=use_xvec,
                    )
                except Exception as exc:
                    print(
                        f"      synth attempt {attempt} failed: {exc}",
                        file=sys.stderr,
                    )
                    continue
                fit_src = _resample_trim(chunk)

                if not verify_clone:
                    best_fit_src = fit_src
                    best_overlap = 1.0
                    accepted = True
                    break

                ok, overlap, asr = verify_clone_clip(fit_src, speak)
                # Voice identity: compare clone embedding to the ref embedding.
                voice_sim = voice_similarity(fit_src, Path(ref["path"]))
                score = overlap * 0.7 + max(0.0, voice_sim) * 0.3
                print(
                    f"      clone verify attempt {attempt}: "
                    f"ok={ok} overlap={overlap:.2f} voice={voice_sim:.2f} "
                    f"xvec={use_xvec} asr={asr[:50]!r}",
                    file=sys.stderr,
                )
                he_chars = len(re.findall(r"[\u0590-\u05FF]", asr or ""))
                if he_chars >= 4 and not use_xvec:
                    # Hebrew ICL often collapses EN target → skip further ICL tries.
                    skip_icl = True
                    print(
                        "      ICL Hebrew bleed detected; preferring x-vector",
                        file=sys.stderr,
                    )
                if score > best_score and (
                    overlap >= CLONE_MIN_OVERLAP * 0.5 or (ok and voice_sim >= 0.5)
                ):
                    best_score = score
                    best_overlap = overlap
                    best_copy = tts_dir / f"seg_{i:02d}_u{j:02d}_best.wav"
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-i",
                            str(fit_src),
                            "-acodec",
                            "pcm_s16le",
                            "-ar",
                            "44100",
                            "-ac",
                            "1",
                            str(best_copy),
                        ],
                        check=True,
                        capture_output=True,
                    )
                    best_fit_src = best_copy
                if ok and voice_sim >= 0.55:
                    accepted = True
                    if best_fit_src is None:
                        best_fit_src = fit_src
                        best_overlap = overlap
                    break
                if ok and best_fit_src is not None and score >= 0.40:
                    # Content OK; voice soft — accept best so far after retries.
                    accepted = True
                    break

            # Deterministic fallback when sampled clones all failed acceptance.
            if verify_clone and not accepted:
                print(
                    f"      clone deterministic fallback "
                    f"(best_overlap={best_overlap:.2f})",
                    file=sys.stderr,
                )
                ref = alt_refs[0]
                try:
                    syn.synthesize(
                        speak,
                        ref_audio=Path(ref["path"]),
                        ref_text=str(ref["ref_text"]),
                        language=language,
                        out_wav=chunk,
                        prompt_key=prompt_key,
                        target_sec=window_sec,
                        seed=spk_seed,
                        x_vector_only=False,  # ICL deterministic
                        deterministic=True,
                    )
                    fit_src = _resample_trim(chunk)
                    ok, overlap, asr = verify_clone_clip(fit_src, speak)
                    print(
                        f"      clone verify fallback: "
                        f"ok={ok} overlap={overlap:.2f} asr={asr[:50]!r}",
                        file=sys.stderr,
                    )
                    if ok or overlap >= CLONE_MIN_OVERLAP:
                        best_fit_src = fit_src
                        best_overlap = overlap
                        accepted = True
                    elif best_fit_src is not None and best_overlap >= CLONE_MIN_OVERLAP * 0.7:
                        # Keep best-effort English rather than leaving a hole.
                        accepted = True
                        print(
                            f"      accepting best-effort clone "
                            f"(overlap={best_overlap:.2f})",
                            file=sys.stderr,
                        )
                except Exception as exc:
                    print(f"      deterministic fallback failed: {exc}", file=sys.stderr)

            if not accepted or best_fit_src is None:
                speaker_fail_counts[spk_key] = speaker_fail_counts.get(spk_key, 0) + 1
                # Last resort: keep best English-ish clip if any content overlap.
                if best_fit_src is not None and best_overlap >= 0.20:
                    print(
                        f"      WARN: weak clone for unit {j} "
                        f"(overlap={best_overlap:.2f}); placing best-effort",
                        file=sys.stderr,
                    )
                    accepted = True
                else:
                    print(
                        f"      FAIL: no accepted clone for unit {j} "
                        f"(best_overlap={best_overlap:.2f}); mark tts_failed",
                        file=sys.stderr,
                    )
                    seg["tts_failed"] = True
                    continue

            # Shorten-retry if natural rate would exceed gentle max.
            # Prefer shorten over chipmunk; never trim the waveform.
            raw_sec = wav_duration(best_fit_src)
            natural = raw_sec / max(window_sec, 0.25)
            overrun_sec = max(0.0, raw_sec - window_sec)
            pause_can_absorb = overrun_sec <= max(0.0, raw_pause - 0.05)
            if (
                natural > FIT_MAX_RATE_GENTLE
                and shorten_en_fn is not None
                and not pause_can_absorb
            ):
                budget = max(6, int(window_sec * SHORTEN_SYL_PER_SEC))
                try:
                    shorter = shorten_en_fn(
                        he_text=he_text,
                        en_text=best_speak,
                        duration=window_sec,
                        syllable_budget=budget,
                    )
                except Exception as exc:
                    print(f"      shorten retry failed: {exc}", file=sys.stderr)
                    shorter = None
                if shorter and shorter.strip() and shorter.strip() != best_speak:
                    orig_words = max(1, len(best_speak.split()))
                    new_words = len(shorter.split())
                    drop = 1.0 - (new_words / orig_words)
                    if drop <= SHORTEN_MAX_WORD_DROP:
                        print(
                            f"      shorten EN ({natural:.2f}× → retry): "
                            f"{shorter[:70]}…",
                            file=sys.stderr,
                        )
                        best_speak = prepare_english_tts_text(shorter)
                        syn.synthesize(
                            best_speak,
                            ref_audio=Path(alt_refs[0]["path"]),
                            ref_text=str(alt_refs[0]["ref_text"]),
                            language=language,
                            out_wav=chunk,
                            prompt_key=prompt_key,
                            target_sec=window_sec,
                            seed=2000 + i * 17 + j,
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
                        best_fit_src = trimmed if lead >= 0.04 else chunk_441
                        raw_sec = wav_duration(best_fit_src)

            fitted_unit = tts_dir / f"seg_{i:02d}_u{j:02d}_fit.wav"
            # Non-final units: don't pad short speech out to the HE window —
            # that creates multi-second holes before a continuing EN clause
            # (1:52 "comfortable … for the Western" bug). Final unit may pad
            # so the segment canvas still covers the HE end.
            pad_short = j + 1 >= len(units)
            rate = fit_exact_window(
                best_fit_src,
                fitted_unit,
                window_sec,
                max_rate=FIT_MAX_RATE_GENTLE,
                min_rate=FIT_MIN_RATE,
                allow_overrun=True,
                pad_short=pad_short,
            )
            rates.append(rate)
            actual = wav_duration(fitted_unit)
            he_slot = he_slot_sec
            delta = actual - he_slot
            if abs(delta) < 0.04:
                note = "exact"
            elif delta > 0:
                note = f"overrun +{delta:.2f}s (kept for packer)"
            else:
                note = f"short {-delta:.2f}s (stretched toward HE)"
            print(
                f"      qwen raw={raw_sec:.2f}s → {actual:.2f}s "
                f"@ rate={rate:.3f} he={he_slot:.2f}s ({note})",
                file=sys.stderr,
            )
            place_end = u_start + actual  # full clip; never trim to HE end
            place_plan.append(
                {
                    "start": u_start,
                    "end": place_end,
                    "source_start": u_start,
                    "source_end": u_end,
                    "tts_fit": str(fitted_unit),
                    "tts_raw": str(best_fit_src),
                    "tts_speed_used": round(rate, 3),
                    "tts_text": best_speak,
                    "text_he": he_text,
                    "text_en": best_speak,
                    "pause_after": float(unit.get("pause_after") or 0.0),
                }
            )

        if not place_plan:
            seg["tts_failed"] = True
            print(
                f"  FAIL [{seg.get('speaker_id')}] no placed units; tts_failed",
                file=sys.stderr,
            )
            continue

        seg.pop("tts_failed", None)
        # Restore immutable ASR window; track fitted occupancy separately.
        start = source_a
        end = source_b
        spoken_start = min(float(p["start"]) for p in place_plan)
        spoken_end = max(float(p["end"]) for p in place_plan)
        seg["start"] = round(start, 3)
        seg["end"] = round(end, 3)
        seg["source_start"] = round(start, 3)
        seg["source_end"] = round(end, 3)
        seg["tts_start"] = round(spoken_start, 3)
        seg["spoken_end"] = round(spoken_end, 3)

        fitted = tts_dir / f"seg_{i:02d}_fit.wav"
        # Assemble full unit audio (never trim). Elastic packer owns KEEP yield.
        assemble_on_hebrew_timeline(
            place_plan,
            start,
            end,
            fitted,
            sample_rate=44100,
            hard_end=None,
            next_is_keep=next_is_keep,
            gap_bleed_sec=0.0,
        )
        spoken = wav_duration(fitted)
        seg["spoken_end"] = round(start + spoken, 3)
        seg["tts_end"] = seg["spoken_end"]
        seg["tts_clip_sec"] = round(spoken, 3)
        seg["duration"] = round(float(seg["end"]) - start, 3)
        print(
            f"    timeline {spoken:.2f}s ({len(place_plan)} unit(s), never-cut assemble)",
            file=sys.stderr,
        )
        seg.pop("tts_text", None)
        seg["tts_raw"] = str(place_plan[0].get("tts_raw") or fitted)
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

        # Replace phrases with placed units; keep ASR source_* for mix/subs.
        new_phrases: list[dict] = []
        for out_p in place_plan:
            src_a = float(out_p.get("source_start") or out_p["start"])
            src_b = float(out_p.get("source_end") or out_p["end"])
            row = {
                "text": (out_p.get("text_he") or orig_he).strip(),
                "text_en": dedupe_repeated_sentences(
                    (out_p.get("tts_text") or out_p.get("text_en") or "").strip()
                ),
                "start": src_a,
                "end": src_b,
                "source_start": src_a,
                "source_end": src_b,
                "tts_start": float(out_p["start"]),
                "spoken_end": float(out_p["end"]),
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
        # Keep utterance-level text_en authoritative when present.
        if not (seg.get("text_en") or "").strip():
            en_bits = [
                (p.get("text_en") or "").strip()
                for p in new_phrases
                if p.get("text_en")
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
        help="Legacy; units split at UNIT_SPLIT_PAUSE_SEC (1.2s). Kept for CLI compat.",
    )
    p.add_argument(
        "--no-clone-verify",
        action="store_true",
        help="Skip clone length/ASR verification + regenerate (faster, less stable).",
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
        verify_clone=not args.no_clone_verify,
    )
    payload["segments"] = segments
    payload["tts_engine"] = "qwen3-tts-1.7b-base"
    out = workdir / "translated_segments.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
