#!/usr/bin/env python3
"""Per-speaker voice bank for consistent Qwen TTS cloning.

Builds one canonical ~3–4s clean vocal ref per diarization speaker, optionally
relabeling outlier segments via embedding proximity so rapid speaker flips
don't produce timbre jumps.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from inference.tts_f5 import extract_wav_slice, wav_duration
from inference.tts_qwen import (
    REF_MAX_SEC,
    REF_MIN_SEC,
    REF_TARGET_SEC,
    _hf_noise_ratio,
    pick_cleanest_ref_window,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_mono_slice(
    vocals: Path, start: float, end: float, *, sample_rate: int = 16000
) -> np.ndarray:
    import soundfile as sf

    tmp = vocals.parent / "tts_refs" / "_emb_probe.wav"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    extract_wav_slice(vocals, start, end, tmp, sample_rate=sample_rate)
    audio, sr = sf.read(str(tmp), dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=-1).astype(np.float32)
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass
    return audio


def _segment_embedding_fallback(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Cheap spectral fingerprint when pyannote embedding model is unavailable."""
    if audio.size < sr // 4:
        return np.zeros(32, dtype=np.float32)
    # Log-mel-ish: band energies + spectral centroid.
    X = np.abs(np.fft.rfft(audio.astype(np.float32)))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    bands = np.linspace(80, min(7000, sr / 2 - 1), 17)
    feats: list[float] = []
    for a, b in zip(bands[:-1], bands[1:]):
        mask = (freqs >= a) & (freqs < b)
        feats.append(float(np.log1p(np.mean(X[mask] ** 2) + 1e-12)))
    # Pitch-ish: peak in 80–400 Hz
    voice = (freqs >= 80) & (freqs < 400)
    if np.any(voice):
        peak_i = int(np.argmax(X[voice]))
        feats.append(float(freqs[voice][peak_i]))
    else:
        feats.append(0.0)
    # Centroid
    denom = float(np.sum(X) + 1e-12)
    feats.append(float(np.sum(freqs * X) / denom))
    # Pad / trim to 32
    while len(feats) < 32:
        feats.append(0.0)
    vec = np.asarray(feats[:32], dtype=np.float32)
    n = float(np.linalg.norm(vec) + 1e-9)
    return vec / n


_PYANNOTE_OK: bool | None = None


def _try_pyannote_embedding(audio: np.ndarray, sr: int = 16000) -> np.ndarray | None:
    global _PYANNOTE_OK
    if _PYANNOTE_OK is False:
        return None
    try:
        import torch
        from pyannote.audio import Model, Inference
        from dotenv import load_dotenv
        import os

        load_dotenv(REPO_ROOT / ".env")
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        model = Model.from_pretrained(
            "pyannote/embedding", use_auth_token=token
        )
        inference = Inference(model, window="whole")
        waveform = torch.from_numpy(audio).float().unsqueeze(0)
        if sr != 16000:
            n_out = int(round(len(audio) * 16000 / sr))
            x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
            audio = np.interp(x_new, x_old, audio).astype(np.float32)
            waveform = torch.from_numpy(audio).float().unsqueeze(0)
        emb = inference({"waveform": waveform, "sample_rate": 16000})
        vec = np.asarray(emb, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(vec) + 1e-9)
        _PYANNOTE_OK = True
        return (vec / n).astype(np.float32)
    except Exception as exc:
        if _PYANNOTE_OK is not False:
            print(
                f"  speaker_bank: pyannote embedding unavailable ({exc}); "
                "using spectral fallback without relabel",
                file=sys.stderr,
            )
        _PYANNOTE_OK = False
        return None


def embed_segment(
    vocals: Path, start: float, end: float, *, use_pyannote: bool = True
) -> tuple[np.ndarray, bool]:
    """Return (embedding, used_pyannote)."""
    dur = max(0.4, min(4.0, float(end) - float(start)))
    mid = (float(start) + float(end)) / 2.0
    a = max(0.0, mid - dur / 2.0)
    b = a + dur
    audio = _load_mono_slice(vocals, a, b, sample_rate=16000)
    if use_pyannote:
        emb = _try_pyannote_embedding(audio, 16000)
        if emb is not None:
            return emb, True
    return _segment_embedding_fallback(audio, 16000), False


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def relabel_speakers_by_embedding(
    segments: list[dict[str, Any]],
    vocals: Path,
    *,
    merge_threshold: float = 0.82,
) -> int:
    """Relabel HE dub segments whose embedding is closer to another speaker centroid.

    Only runs when pyannote embeddings are available — spectral fallback is too
    weak and collapses distinct voices.
    """
    he_idxs = [
        i
        for i, s in enumerate(segments)
        if not s.get("keep_original") and (s.get("language") or "he") == "he"
        and float(s.get("end", 0) - float(s.get("start", 0))) >= 0.6
    ]
    if len(he_idxs) < 2:
        return 0

    embs: dict[int, np.ndarray] = {}
    used_pyannote = False
    for i in he_idxs:
        s = segments[i]
        try:
            emb, ok = embed_segment(vocals, float(s["start"]), float(s["end"]))
            embs[i] = emb
            used_pyannote = used_pyannote or ok
        except Exception:
            continue
    if len(embs) < 2:
        return 0
    if not used_pyannote:
        print(
            "  speaker_bank: skip embedding relabel (no pyannote access)",
            file=sys.stderr,
        )
        return 0

    by_spk: dict[str, list[np.ndarray]] = {}
    for i, emb in embs.items():
        spk = str(segments[i].get("speaker_id") or "SPEAKER_??")
        by_spk.setdefault(spk, []).append(emb)
    centroids = {
        spk: np.mean(np.stack(vs, axis=0), axis=0)
        for spk, vs in by_spk.items()
        if vs
    }
    for spk, c in list(centroids.items()):
        centroids[spk] = c / (np.linalg.norm(c) + 1e-9)

    spk_ids = list(centroids.keys())
    merge_map: dict[str, str] = {s: s for s in spk_ids}
    for a_i in range(len(spk_ids)):
        for b_i in range(a_i + 1, len(spk_ids)):
            a, b = spk_ids[a_i], spk_ids[b_i]
            if _cosine(centroids[a], centroids[b]) >= merge_threshold + 0.05:
                keep, drop = (a, b) if a <= b else (b, a)
                merge_map[drop] = keep

    changed = 0
    for i, emb in embs.items():
        cur = merge_map.get(
            str(segments[i].get("speaker_id") or ""),
            str(segments[i].get("speaker_id") or ""),
        )
        best_spk = cur
        best = _cosine(emb, centroids.get(cur, emb))
        for spk, c in centroids.items():
            canon = merge_map.get(spk, spk)
            score = _cosine(emb, c)
            if score > best + 0.05 and score >= merge_threshold:
                best = score
                best_spk = canon
        if best_spk != str(segments[i].get("speaker_id")):
            print(
                f"  relabel {segments[i].get('speaker_id')} → {best_spk} "
                f"@ {float(segments[i]['start']):.1f}s (cos={best:.2f})",
                file=sys.stderr,
            )
            segments[i]["speaker_id"] = best_spk
            for p in segments[i].get("phrases") or []:
                p["speaker_id"] = best_spk
            changed += 1
    return changed


def _concat_ref_windows(
    vocals: Path,
    windows: list[tuple[float, float]],
    out_path: Path,
    *,
    target_sec: float = REF_TARGET_SEC,
    sample_rate: int = 24000,
) -> tuple[float, float] | None:
    """Concatenate several short clean slices until ≥ REF_MIN_SEC."""
    import soundfile as sf

    chunks: list[np.ndarray] = []
    used: list[tuple[float, float]] = []
    need = float(target_sec)
    for a, b in sorted(windows, key=lambda w: -(w[1] - w[0])):
        if need <= 0:
            break
        take = min(b - a, max(0.4, need))
        tmp = out_path.with_name(out_path.stem + f"_piece_{len(used)}.wav")
        extract_wav_slice(vocals, a, a + take, tmp, sample_rate=sample_rate)
        audio, sr = sf.read(str(tmp), dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = np.mean(audio, axis=-1).astype(np.float32)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        if audio.size < sample_rate // 10:
            continue
        chunks.append(audio.astype(np.float32))
        used.append((a, a + take))
        need -= take
    if not chunks:
        return None
    combined = np.concatenate(chunks)
    dur = len(combined) / float(sample_rate)
    if dur < REF_MIN_SEC - 0.05:
        return None
    # Trim to REF_MAX_SEC
    max_n = int(round(REF_MAX_SEC * sample_rate))
    if len(combined) > max_n:
        combined = combined[:max_n]
        dur = len(combined) / float(sample_rate)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), combined, sample_rate)
    return (used[0][0], used[0][0] + dur)


def pick_canonical_ref_for_speaker(
    speaker_id: str,
    segments: list[dict[str, Any]],
    vocals: Path,
    out_dir: Path,
    media_duration: float,
) -> dict[str, Any] | None:
    """Choose the cleanest ≥REF_MIN_SEC window across all turns of this speaker.

    Prefers the longest clean candidate. If no single turn yields REF_MIN_SEC,
    concatenates multiple clean slices. Returns None (skip bank) when still
    too short — TTS will fall back to per-phrase refs.
    """
    windows: list[tuple[float, float]] = []
    for seg in segments:
        if str(seg.get("speaker_id")) != str(speaker_id):
            continue
        if seg.get("keep_original"):
            continue
        if (seg.get("language") or "he") != "he":
            continue
        a, b = float(seg["start"]), float(seg["end"])
        if b - a >= 0.5:
            windows.append((a, b))
    if not windows:
        return None

    # Prefer longer search windows first.
    windows.sort(key=lambda w: -(w[1] - w[0]))

    best: tuple[float, float, float, float] | None = None  # score, start, end, dur
    out_dir.mkdir(parents=True, exist_ok=True)
    for search_a, search_b in windows:
        span = search_b - search_a
        if span < REF_MIN_SEC:
            continue
        target = min(REF_TARGET_SEC, max(REF_MIN_SEC, span))
        ref_start, ref_end = pick_cleanest_ref_window(
            vocals,
            search_a,
            search_b,
            media_duration=media_duration,
            target_sec=target,
        )
        dur = ref_end - ref_start
        if dur < REF_MIN_SEC - 0.05:
            continue
        tmp = out_dir / f"_score_{speaker_id}.wav"
        extract_wav_slice(vocals, ref_start, ref_end, tmp, sample_rate=24000)
        import soundfile as sf

        audio, sr = sf.read(str(tmp), dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = np.mean(audio, axis=-1).astype(np.float32)
        rms = float(np.sqrt(np.mean(audio**2) + 1e-12))
        score = _hf_noise_ratio(audio, sr) / max(rms, 1e-3)
        # Prefer longer refs when cleanliness is similar.
        score = score - 0.02 * min(dur, REF_TARGET_SEC)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        if best is None or score < best[0]:
            best = (score, ref_start, ref_end, dur)

    ref_path = out_dir / f"speaker_{speaker_id}_ref.wav"
    if best is not None:
        _, ref_start, ref_end, _ = best
        if ref_end - ref_start > REF_MAX_SEC:
            ref_end = ref_start + REF_MAX_SEC
        extract_wav_slice(vocals, ref_start, ref_end, ref_path, sample_rate=24000)
        return {
            "path": str(ref_path.resolve()),
            "start": ref_start,
            "end": ref_end,
            "speaker_id": speaker_id,
            "score": float(best[0]),
        }

    # No single window long enough — try concatenating short turns.
    concat = _concat_ref_windows(
        vocals, windows, ref_path, target_sec=REF_TARGET_SEC
    )
    if concat is None:
        print(
            f"  speaker bank [{speaker_id}] skip — no ≥{REF_MIN_SEC:.1f}s ref "
            f"(will use per-phrase refs)",
            file=sys.stderr,
        )
        return None
    ref_start, ref_end = concat
    return {
        "path": str(ref_path.resolve()),
        "start": ref_start,
        "end": ref_end,
        "speaker_id": speaker_id,
        "score": 0.0,
        "concat": True,
    }


def build_speaker_bank(
    segments: list[dict[str, Any]],
    vocals: Path,
    workdir: Path,
    *,
    relabel: bool = True,
) -> dict[str, Any]:
    """Build tts_refs/speaker_bank.json with one clean ref per speaker."""
    out_dir = workdir / "tts_refs"
    out_dir.mkdir(parents=True, exist_ok=True)
    media_duration = wav_duration(vocals)

    n_relabel = 0
    if relabel:
        try:
            n_relabel = relabel_speakers_by_embedding(segments, vocals)
        except Exception as exc:
            print(f"  speaker_bank relabel skipped: {exc}", file=sys.stderr)

    speakers = sorted(
        {
            str(s.get("speaker_id"))
            for s in segments
            if not s.get("keep_original") and (s.get("language") or "he") == "he"
        }
    )
    bank: dict[str, Any] = {"speakers": {}, "relabeled": n_relabel}
    for spk in speakers:
        meta = pick_canonical_ref_for_speaker(
            spk, segments, vocals, out_dir, media_duration
        )
        if meta:
            bank["speakers"][spk] = meta
            print(
                f"  speaker bank [{spk}] ref "
                f"{meta['start']:.1f}-{meta['end']:.1f}s → {Path(meta['path']).name}",
                file=sys.stderr,
            )
    out_path = out_dir / "speaker_bank.json"
    out_path.write_text(json.dumps(bank, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(bank['speakers'])} speakers)", file=sys.stderr)
    return bank
