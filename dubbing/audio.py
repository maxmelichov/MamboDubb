"""ffmpeg / numpy audio helpers shared by every stage."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100  # working rate for everything that ends up in the mix


def require_tools() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} not found on PATH (brew install ffmpeg)")


def run(cmd: list[str]) -> None:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {res.stderr[-1500:]}")


def duration(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        text=True,
    ).strip()
    return float(out)


def decode_mono(path: Path, sr: int, *, start: float = 0.0, end: float | None = None) -> np.ndarray:
    """Decode (a slice of) any audio file to a mono float32 array at `sr`."""
    cmd = ["ffmpeg", "-v", "error"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(path)]
    if end is not None:
        cmd += ["-t", f"{max(0.01, end - start):.3f}"]
    cmd += ["-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "-"]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"decode failed for {path}: {res.stderr[-800:].decode(errors='ignore')}")
    return np.frombuffer(res.stdout, dtype=np.float32)


def extract_slice(src: Path, start: float, end: float, dst: Path, *, sample_rate: int = SR) -> Path:
    """Write [start, end) of `src` to `dst`. Seek after -i for frame accuracy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    dur = max(end - start, 0.05)
    run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-ss", f"{start:.3f}",
         "-t", f"{dur:.3f}", "-acodec", "pcm_s16le", "-ar", str(sample_rate),
         "-ac", "1", str(dst)])
    return dst


def atempo_chain(rate: float) -> str:
    """ffmpeg atempo filter chain; each stage must stay within 0.5–2.0."""
    rate = max(0.05, min(20.0, float(rate)))
    parts: list[str] = []
    while rate > 2.0 + 1e-6:
        parts.append("atempo=2.0")
        rate /= 2.0
    while rate < 0.5 - 1e-6:
        parts.append("atempo=0.5")
        rate /= 0.5
    parts.append(f"atempo={rate:.6f}")
    return ",".join(parts)


def atempo(src: Path, dst: Path, rate: float, *, sample_rate: int = SR) -> Path:
    """Time-compress/expand without changing pitch. Never trims."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-af", atempo_chain(rate),
         "-acodec", "pcm_s16le", "-ar", str(sample_rate), "-ac", "1", str(dst)])
    return dst


def trim_leading_silence(
    src: Path,
    dst: Path,
    *,
    sample_rate: int = SR,
    rms_thresh: float = 0.012,
    pad_sec: float = 0.03,
    max_trim_sec: float = 0.85,
) -> float:
    """Drop the hush Qwen emits before speech. Returns seconds removed."""
    a, sr = sf.read(str(src), dtype="float32", always_2d=False)
    if getattr(a, "ndim", 1) > 1:
        a = np.mean(a, axis=-1).astype(np.float32)
    if sr != sample_rate:
        raise RuntimeError(f"unexpected sample rate {sr} (want {sample_rate})")
    hop = max(1, sample_rate // 50)
    limit = min(len(a) - hop, int(max_trim_sec * sample_rate))
    onset = 0
    for i in range(0, max(1, limit), hop):
        if float(np.sqrt(np.mean(a[i : i + hop] ** 2) + 1e-12)) >= rms_thresh:
            onset = max(0, i - int(pad_sec * sample_rate))
            break
    trimmed = onset / sample_rate
    if trimmed < 0.04:
        if Path(src) != Path(dst):
            sf.write(str(dst), a, sample_rate)
        return 0.0
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), a[onset:].astype(np.float32), sample_rate)
    return float(trimmed)


def frame_rms(audio: np.ndarray, sr: int, hop_sec: float = 0.1) -> np.ndarray:
    hop = max(1, int(hop_sec * sr))
    n = len(audio) // hop
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    frames = audio[: n * hop].reshape(n, hop)
    return np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1) + 1e-12)


def hf_noise_ratio(audio: np.ndarray, sr: int) -> float:
    """High-band / mid-band energy. Demucs hiss scores high, clean speech low."""
    if audio.size < sr // 4:
        return 1e9
    x = np.asarray(audio, dtype=np.float32)
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    mid = float(np.mean(spec[(freqs >= 200) & (freqs < 3500)] ** 2) + 1e-12)
    high = float(np.mean(spec[(freqs >= 5500) & (freqs < 14000)] ** 2))
    return high / mid


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2) + 1e-12))
