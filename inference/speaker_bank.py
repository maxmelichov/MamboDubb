#!/usr/bin/env python3
"""Per-speaker voice bank for consistent Qwen TTS cloning.

Builds one canonical ~4.5–6s clean vocal ref per diarization speaker (concat
of two clean windows when a single turn is short), optionally relabeling
outlier segments via embedding proximity so rapid speaker flips don't produce
timbre jumps.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from inference.tts_f5 import extract_wav_slice, wav_duration
from inference.tts_qwen import (
    REF_CONCAT_SEC,
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
    """Token-free speaker fingerprint (log-mel bands + pitch + deltas).

    Stronger than a raw FFT snapshot so same-person turns can be merged without
    pyannote/HF access. Not as good as ECAPA, but enough for documentary
    over-segmentation collapse.
    """
    if audio.size < sr // 4:
        return np.zeros(64, dtype=np.float32)
    x = audio.astype(np.float32)
    # Frame into ~25ms windows, 10ms hop.
    win = max(1, int(0.025 * sr))
    hop = max(1, int(0.010 * sr))
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    n_frames = 1 + (len(x) - win) // hop
    # Precompute FFT freqs for mel-ish bands.
    n_fft = 512
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    bands = np.linspace(80, min(7000, sr / 2 - 1), 25)
    band_masks = [(freqs >= a) & (freqs < b) for a, b in zip(bands[:-1], bands[1:])]
    frame_feats: list[np.ndarray] = []
    for fi in range(n_frames):
        frame = x[fi * hop : fi * hop + win]
        if len(frame) < win:
            frame = np.pad(frame, (0, win - len(frame)))
        # Hann window
        frame = frame * np.hanning(win).astype(np.float32)
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        else:
            frame = frame[:n_fft]
        X = np.abs(np.fft.rfft(frame, n=n_fft))
        bands_e = np.array(
            [float(np.log1p(np.mean(X[m] ** 2) + 1e-12)) for m in band_masks],
            dtype=np.float32,
        )
        frame_feats.append(bands_e)
    mat = np.stack(frame_feats, axis=0)  # (T, 24)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    # Delta (first-order)
    if mat.shape[0] >= 3:
        delta = mat[2:] - mat[:-2]
        d_mean = delta.mean(axis=0)
    else:
        d_mean = np.zeros_like(mean)
    # Pitch-ish: peak in 80–400 Hz from full clip
    Xf = np.abs(np.fft.rfft(x))
    ff = np.fft.rfftfreq(len(x), 1.0 / sr)
    voice = (ff >= 80) & (ff < 400)
    if np.any(voice):
        peak_f = float(ff[voice][int(np.argmax(Xf[voice]))])
    else:
        peak_f = 0.0
    denom = float(np.sum(Xf) + 1e-12)
    centroid = float(np.sum(ff * Xf) / denom)
    extras = np.array([peak_f / 400.0, centroid / 8000.0], dtype=np.float32)
    # Pack to 64-d: mean(24) + std(24) + d_mean truncated + extras
    vec = np.concatenate(
        [
            mean,
            std,
            d_mean[:14],
            extras,
        ]
    ).astype(np.float32)
    if vec.size < 64:
        vec = np.pad(vec, (0, 64 - vec.size))
    vec = vec[:64]
    n = float(np.linalg.norm(vec) + 1e-9)
    return vec / n


_SPEECHBRAIN_OK: bool | None = None
_SPEECHBRAIN_CLASSIFIER = None


def _try_speechbrain_embedding(audio: np.ndarray, sr: int = 16000) -> np.ndarray | None:
    """Optional ECAPA-TDNN via SpeechBrain (no gated HF token required)."""
    global _SPEECHBRAIN_OK, _SPEECHBRAIN_CLASSIFIER
    if _SPEECHBRAIN_OK is False:
        return None
    try:
        import torch

        if _SPEECHBRAIN_CLASSIFIER is None:
            from speechbrain.inference.speaker import EncoderClassifier

            _SPEECHBRAIN_CLASSIFIER = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(REPO_ROOT / "models" / "spkrec-ecapa-voxceleb"),
                run_opts={"device": "cpu"},
            )
        wav = audio.astype(np.float32)
        if sr != 16000:
            n_out = int(round(len(wav) * 16000 / sr))
            x_old = np.linspace(0.0, 1.0, num=len(wav), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
            wav = np.interp(x_new, x_old, wav).astype(np.float32)
        tensor = torch.from_numpy(wav).unsqueeze(0)
        with torch.no_grad():
            emb = _SPEECHBRAIN_CLASSIFIER.encode_batch(tensor)
        vec = emb.squeeze().cpu().numpy().astype(np.float32).reshape(-1)
        n = float(np.linalg.norm(vec) + 1e-9)
        _SPEECHBRAIN_OK = True
        return (vec / n).astype(np.float32)
    except Exception as exc:
        if _SPEECHBRAIN_OK is not False:
            print(
                f"  speaker_bank: SpeechBrain ECAPA unavailable ({exc}); "
                "using improved spectral embedding",
                file=sys.stderr,
            )
        _SPEECHBRAIN_OK = False
        return None


def embed_segment(
    vocals: Path, start: float, end: float, *, use_pyannote: bool = True
) -> tuple[np.ndarray, str]:
    """Return (embedding, backend) where backend is pyannote|speechbrain|spectral."""
    dur = max(0.4, min(4.0, float(end) - float(start)))
    mid = (float(start) + float(end)) / 2.0
    a = max(0.0, mid - dur / 2.0)
    b = a + dur
    audio = _load_mono_slice(vocals, a, b, sample_rate=16000)
    if use_pyannote:
        emb = _try_pyannote_embedding(audio, 16000)
        if emb is not None:
            return emb, "pyannote"
    emb = _try_speechbrain_embedding(audio, 16000)
    if emb is not None:
        return emb, "speechbrain"
    return _segment_embedding_fallback(audio, 16000), "spectral"


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
                "will try SpeechBrain / spectral",
                file=sys.stderr,
            )
        _PYANNOTE_OK = False
        return None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def relabel_speakers_by_embedding(
    segments: list[dict[str, Any]],
    vocals: Path,
    *,
    merge_threshold: float = 0.82,
) -> int:
    """Relabel HE dub segments whose embedding is closer to another speaker centroid.

    Uses pyannote when available, else SpeechBrain ECAPA (token-free), else an
    improved spectral fingerprint. Spectral merges use a higher threshold to
    avoid collapsing distinct voices.
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
    backends: set[str] = set()
    for i in he_idxs:
        s = segments[i]
        try:
            emb, backend = embed_segment(vocals, float(s["start"]), float(s["end"]))
            embs[i] = emb
            backends.add(backend)
        except Exception:
            continue
    if len(embs) < 2:
        return 0

    # Spectral is weaker — only allow speaker-level centroid merges at a very
    # high threshold, and skip per-segment reassignment (too noisy / collapses
    # distinct documentary voices).
    backend = (
        "pyannote"
        if "pyannote" in backends
        else ("speechbrain" if "speechbrain" in backends else "spectral")
    )
    thresh = float(merge_threshold)
    per_segment_relabel = True
    winner_margin = 0.05
    if backend == "spectral":
        thresh = max(thresh, 0.97)
        per_segment_relabel = False
        winner_margin = 0.08
    elif backend == "speechbrain":
        thresh = max(thresh, 0.85)
        winner_margin = 0.06
    else:
        # pyannote: still require a clear winner so distinct people stay distinct.
        thresh = max(thresh, 0.88)
        winner_margin = 0.05
    print(
        f"  speaker_bank: embedding backend={backend} merge_thresh={thresh:.2f} "
        f"margin={winner_margin:.2f} per_segment={per_segment_relabel}",
        file=sys.stderr,
    )

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
            if _cosine(centroids[a], centroids[b]) >= thresh + 0.02:
                keep, drop = (a, b) if a <= b else (b, a)
                merge_map[drop] = keep

    changed = 0
    # Apply speaker-level merges first.
    for i in he_idxs:
        cur = str(segments[i].get("speaker_id") or "")
        canon = merge_map.get(cur, cur)
        if canon != cur:
            print(
                f"  merge-spk {cur} → {canon} "
                f"@ {float(segments[i]['start']):.1f}s",
                file=sys.stderr,
            )
            segments[i]["speaker_id"] = canon
            for p in segments[i].get("phrases") or []:
                p["speaker_id"] = canon
            changed += 1

    if not per_segment_relabel:
        return changed

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
            if score > best + winner_margin and score >= thresh:
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

    Prefers a long clean candidate (~REF_TARGET_SEC). When the best single
    window is shorter than REF_CONCAT_SEC, concatenates a second clean slice
    for stronger x-vector identity. Returns None when still too short — TTS
    falls back to per-phrase refs.
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
        _, ref_start, ref_end, dur = best
        # Pad with a second clean window when the best single slice is short.
        if dur < REF_CONCAT_SEC - 0.15 and len(windows) >= 2:
            concat = _concat_ref_windows(
                vocals,
                windows,
                ref_path,
                target_sec=REF_CONCAT_SEC,
            )
            if concat is not None:
                ref_start, ref_end = concat
                return {
                    "path": str(ref_path.resolve()),
                    "start": ref_start,
                    "end": ref_end,
                    "speaker_id": speaker_id,
                    "score": float(best[0]),
                    "concat": True,
                }
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
        vocals, windows, ref_path, target_sec=REF_CONCAT_SEC
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
    merge_threshold: float = 0.90,
) -> dict[str, Any]:
    """Build tts_refs/speaker_bank.json with one clean ref per speaker.

    Conservative embedding merge (high threshold + winner margin) collapses
    over-segmented diarization labels for the same real person while refusing
    to share a voice across distinct overlapping speakers.
    """
    out_dir = workdir / "tts_refs"
    out_dir.mkdir(parents=True, exist_ok=True)
    media_duration = wav_duration(vocals)

    n_relabel = 0
    if relabel:
        try:
            n_relabel = relabel_speakers_by_embedding(
                segments, vocals, merge_threshold=merge_threshold
            )
        except Exception as exc:
            print(f"  speaker_bank relabel skipped: {exc}", file=sys.stderr)
    else:
        print(
            "  speaker_bank: cross-speaker merge disabled",
            file=sys.stderr,
        )

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
            # Attach the HE transcript that overlaps the ref window for ICL.
            meta["ref_text"] = _ref_text_for_window(
                segments, spk, float(meta["start"]), float(meta["end"])
            )
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


def _ref_text_for_window(
    segments: list[dict[str, Any]],
    speaker_id: str,
    ref_start: float,
    ref_end: float,
) -> str:
    """HE transcript overlapping the reference window for ICL cloning."""
    bits: list[str] = []
    for seg in segments:
        if str(seg.get("speaker_id")) != str(speaker_id):
            continue
        if seg.get("keep_original") or (seg.get("language") or "he") != "he":
            continue
        for p in seg.get("phrases") or [{"text": seg.get("text"), "start": seg.get("start"), "end": seg.get("end")}]:
            a = float(p.get("start") or 0.0)
            b = float(p.get("end") or 0.0)
            if b <= ref_start or a >= ref_end:
                continue
            t = (p.get("text") or "").strip()
            if t:
                bits.append(t)
        if not bits:
            a = float(seg.get("start") or 0.0)
            b = float(seg.get("end") or 0.0)
            if not (b <= ref_start or a >= ref_end):
                t = (seg.get("text") or "").strip()
                if t:
                    bits.append(t)
    return " ".join(bits).strip() or "um"
