"""ffmpeg / numpy audio helpers shared by every stage."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from . import tools

SR = 44100  # working rate for everything that ends up in the mix


def require_tools() -> None:
    # `resolve_tool` and not a bare exec: it checks the workspace `tools/bin`
    # (where a brewless Mac's static build lives) before PATH, honours PATHEXT
    # so this finds `ffmpeg.exe` on Windows, and the message names the command
    # *this* platform installs it with rather than a Mac's.
    for tool in ("ffmpeg", "ffprobe"):
        if tools.resolve_tool(tool) is None:
            hint = tools.command("ffmpeg")
            raise SystemExit(f"{tool} not found on PATH"
                             + (f" ({hint})" if hint else ""))


def run(cmd: list[str]) -> None:
    # Every ffmpeg in the pipeline funnels through here (fetch, tts, mix), so
    # this one line is what lets them all find a workspace-installed build.
    res = subprocess.run([tools.resolve_tool(cmd[0]) or cmd[0], *cmd[1:]],
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {res.stderr[-1500:]}")


def duration(path: Path) -> float:
    out = subprocess.check_output(
        [tools.resolve_tool("ffprobe") or "ffprobe", "-v", "error",
         "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        text=True,
    ).strip()
    return float(out)


def decode_mono(path: Path, sr: int, *, start: float = 0.0, end: float | None = None) -> np.ndarray:
    """Decode (a slice of) any audio file to a mono float32 array at `sr`."""
    cmd = [tools.resolve_tool("ffmpeg") or "ffmpeg", "-v", "error"]
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


# An unvoiced fricative is quiet and high. /s/ at the end of a word carries a
# fraction of the power of the vowel before it — measured across the demo runs a
# sentence-final /s/ sits 12 to 30 dB under the line's own speech level — but
# nearly all of what it does carry is above 4 kHz, where a vowel has almost
# nothing. So "is this frame silence?" cannot be answered by loudness alone: the
# RMS gate that correctly calls Qwen's trailing hush silence also calls the /s/
# of "us" silence, and trims it off. These two numbers are the second question,
# asked only of frames the loudness gate already rejected.
FRICATIVE_FLOOR = 0.0025   # ~-52 dBFS; Qwen's hush measures -70 dBFS and below
FRICATIVE_TILT = 1.0       # summed 4-9 kHz magnitude over summed sub-1 kHz


def is_fricative(frame: np.ndarray, sample_rate: int) -> bool:
    """True when a frame is quiet-but-high, the signature of an unvoiced fricative.

    The test is a spectral tilt: sum the FFT magnitude in 4-9 kHz and divide by
    the sum below 1 kHz. A vowel is bottom-heavy and lands far below 1; an /s/ or
    /f/ or /sh/ is top-heavy and lands above it, often by an order of magnitude.
    Digital hush has no reliable tilt either way, so `FRICATIVE_FLOOR` keeps a
    noise floor from reading as speech and stopping a trim that should happen.
    """
    if len(frame) < 32:
        return False
    if float(np.sqrt(np.mean(frame ** 2) + 1e-12)) < FRICATIVE_FLOOR:
        return False
    mag = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
    freq = np.fft.rfftfreq(len(frame), 1.0 / sample_rate)
    low = float(mag[freq < 1000].sum())
    if low <= 0:
        return False
    return float(mag[(freq >= 4000) & (freq <= 9000)].sum()) / low > FRICATIVE_TILT


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


def trim_trailing_silence(
    src: Path,
    dst: Path,
    *,
    sample_rate: int = SR,
    rms_thresh: float = 0.012,
    pad_sec: float = 0.06,
    max_trim_sec: float = 0.85,
) -> float:
    """Drop the hush Qwen emits after the sentence. Returns seconds removed.

    The mirror of `trim_leading_silence`, and it exists for the same reason: the
    clip that leaves TTS should hold the line and nothing else. A trailing hush
    is not free, because the timeline measures the *file*, and a clip whose last
    quarter-second is silence is compressed as though those were words, so the
    audible speech is squeezed harder than it needs to be and the sentence
    finishes early inside its own slot. Measured on the demo runs the tail
    averaged 0.12–0.22 s, i.e. ~5% of clip length, which is ~5% of speed the
    speech was paying for nothing.

    The pad is a touch wider than the leading one: a final consonant or a vowel
    decay trails off gradually, and clipping the release is audible in a way
    that clipping a silent lead-in is not.

    The pad alone is not enough, though, and widening it is not the fix. A frame
    is kept when it clears `rms_thresh` *or* when `is_fricative` says it is
    quiet-but-high, because a sentence-final /s/ is both: it lives 12 to 30 dB
    under the line it ends, which is under this gate, so on loudness alone the
    scan walks straight past it and stops at the vowel before it. The pad then
    measures from the wrong place and the file ends mid-fricative. That is the
    "and not the hedgehogs to us" report: the /s/ was generated, and the trim
    was what removed it. Widening the pad only buys a fixed number of
    milliseconds and pays for it on every clip that really did end in hush.
    """
    a, sr = sf.read(str(src), dtype="float32", always_2d=False)
    if getattr(a, "ndim", 1) > 1:
        a = np.mean(a, axis=-1).astype(np.float32)
    if sr != sample_rate:
        raise RuntimeError(f"unexpected sample rate {sr} (want {sample_rate})")
    hop = max(1, sample_rate // 50)
    limit = min(len(a) - hop, int(max_trim_sec * sample_rate))
    end = len(a)
    for i in range(0, max(1, limit), hop):
        j = len(a) - i - hop
        if j < 0:
            break
        frame = a[j : j + hop]
        loud = float(np.sqrt(np.mean(frame ** 2) + 1e-12)) >= rms_thresh
        if loud or is_fricative(frame, sample_rate):
            end = min(len(a), j + hop + int(pad_sec * sample_rate))
            break
    trimmed = (len(a) - end) / sample_rate
    if trimmed < 0.04:
        if Path(src) != Path(dst):
            sf.write(str(dst), a, sample_rate)
        return 0.0
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), a[:end].astype(np.float32), sample_rate)
    return float(trimmed)


def final_fricative_db(samples: np.ndarray, sample_rate: int,
                       *, win_sec: float = 0.030, tail_sec: float = 0.60,
                       ) -> float | None:
    """How far a line's closing fricative sits under the line's own level, in dB.

    Returns a negative number, or None when the tail holds no fricative at all
    (which for a line that should end in one is itself the failure, and the
    caller is the one that knows whether to expect it). The reference is the
    median of the frames within 25 dB of the loudest, i.e. the line's own working
    level, so the answer does not move with how loudly this take was generated.

    Why a fricative and not just "the last phoneme": a line can end on a
    perfectly audible sound that is not the one the words call for. Qwen likes to
    leave a voiced murmur 100 to 300 ms behind the sentence, and "the last thing
    with energy in it" finds the murmur and calls the ending healthy. The tilted
    frames are the consonant itself.

    Be clear about what this does and does not catch. It catches an ending that
    is *gone* or nearly so, which is the shape the trailing trim used to produce
    before `is_fricative` was added to it. It did NOT catch the line that started
    this work: the /s/ of "and not the hedgehogs to us", which three native
    listeners heard go missing, measures 13.8 dB under its line against a corpus
    median of 10.8 dB, i.e. squarely mid-distribution. That line is not
    acoustically unusual for this voice; see `report.faint_line_endings`.
    """
    n = int(win_sec * sample_rate)
    hop = max(1, n // 3)
    if len(samples) < n:
        return None
    levels, tilted = [], []
    for i in range(0, len(samples) - n, hop):
        seg = samples[i : i + n]
        levels.append(20.0 * np.log10(float(np.sqrt(np.mean(seg ** 2))) + 1e-12))
        tilted.append(is_fricative(seg, sample_rate))
    if not levels:
        return None
    peak = max(levels)
    if peak < -70.0:
        return None
    ref = float(np.median([v for v in levels if v > peak - 25.0]))
    # Only the closing stretch, and within it only the *last* run of tilted
    # frames. Both halves matter. An /s/ in the middle of the line says nothing
    # about whether the line finished, and taking the loudest fricative in the
    # tail is just as wrong: in "…the hedgehogs to us" the /s/ of "hedgehogs" is
    # 4 dB hotter than the /s/ of "us" and sits inside the same 0.6 s, so the
    # loudest reading scores the ending on a fricative two words early. The peak
    # of the run (not its dying edge) is what a listener hears it as.
    first = max(0, len(levels) - int(tail_sec * sample_rate / hop))
    last = None
    for i in range(len(levels) - 1, first - 1, -1):
        if tilted[i]:
            last = i
            break
    if last is None:
        return None
    start = last
    while start > first and tilted[start - 1]:
        start -= 1
    return float(max(levels[start : last + 1]) - ref)


def frame_rms(audio: np.ndarray, sr: int, hop_sec: float = 0.1) -> np.ndarray:
    hop = max(1, int(hop_sec * sr))
    n = len(audio) // hop
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    frames = audio[: n * hop].reshape(n, hop)
    return np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1) + 1e-12)


def speech_bounds(audio: np.ndarray, sr: int, *, hop_sec: float = 0.02,
                  rel: float = 0.10, floor: float = 0.004
                  ) -> tuple[float, float] | None:
    """First and last speech instant in `audio`, in seconds, or None if silent.

    The threshold is relative to the window's own loud end (the 97th percentile
    of frame RMS, so one click cannot set it) with an absolute floor underneath,
    which is what lets one function read both a quiet interview and a shouted
    line without a per-run gain constant.
    """
    e = frame_rms(audio, sr, hop_sec)
    if e.size == 0:
        return None
    thresh = max(floor, rel * float(np.percentile(e, 97)))
    idx = np.flatnonzero(e >= thresh)
    if idx.size == 0:
        return None
    return float(idx[0] * hop_sec), float((idx[-1] + 1) * hop_sec)


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
