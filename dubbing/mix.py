"""Stage 8 — assemble the final audio and mux the preview.

Speech is *added* into the output, never written over it. Because the timeline
guarantees non-overlapping placements, the occupancy check below can only fail
if that guarantee was broken, so it is a tripwire rather than a fix-up. The old
pipeline's overwrite-on-overlap is what cut the tails off spoken lines.

Time not claimed by any placed clip carries the original vocals stem: a
designed final pass adds the corresponding slice of stems/vocals.wav, at unity
gain, into every span the placements left empty. Wherever detection upstream
missed a speaker entirely (no segment, so no clip), the viewer hears the
original voice over the bed instead of a voice-stripped mute. Silent vocals
add silence, which is harmless. This does NOT use the vocals stem as a
speech-presence oracle — no decision keys off it; it is used as audio,
unconditionally, in all unclaimed time.

Audio is streamed in blocks so a 40-minute video does not need a 40-minute
buffer in memory.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from . import audio

BG_GAIN = 0.68         # music bed level under speech
DUCK_DUB = 0.86        # bed ducking under a dubbed line (gentle: the bed should
                       # sit with the dub, not disappear under it)
DUCK_KEEP = 0.0        # original audio already contains its own bed
SPEECH_GAIN = 1.2
SPEECH_RMS = 0.085     # per-clip loudness target for dubbed speech
# Kept original audio carries its own background (the bed is ducked away under
# it), so it is levelled as a finished mix — against dub+bed, not against the dub
# alone. Without this the original-language passages sit ~7dB below the dubs and
# sound muted. Its gain cap is looser because source recordings vary far more
# than the synthesiser's output does.
KEEP_RMS = 0.105
KEEP_GAIN_CAP = 4.0
GAIN_CAP = 1.25
FADE_SEC = 0.012
PEAK = 0.98
CTRL_HZ = 200
BLOCK_SEC = 30.0
# Vocals fill: original vocals played in every span no placement claimed.
FILL_SLOP = 0.05       # placements this close are treated as contiguous
FILL_MIN_SEC = 0.25    # complement slivers shorter than this are ignored
FILL_FADE_SEC = 0.030  # edge fade on each fill span
FILL_FLOOR = 0.008     # per-frame vocal rms above this is audible (ducking only —
                       # the fill itself is unconditional)


def _duck(env: np.ndarray, t0: float, t1: float, level: float,
          attack: float, release: float) -> None:
    n = len(env)
    i0 = max(0, min(n, int(t0 * CTRL_HZ)))
    i1 = max(0, min(n, int(t1 * CTRL_HZ)))
    if i1 <= i0:
        return
    a = max(1, int(attack * CTRL_HZ))
    r = max(1, int(release * CTRL_HZ))
    a_end = min(i1, i0 + a)
    if a_end > i0:
        env[i0:a_end] = np.minimum(env[i0:a_end], np.linspace(1.0, level, a_end - i0))
    r_start = max(a_end, i1 - r)
    if r_start > a_end:
        env[a_end:r_start] = np.minimum(env[a_end:r_start], level)
    if i1 > r_start:
        env[r_start:i1] = np.minimum(env[r_start:i1], np.linspace(level, 1.0, i1 - r_start))


def build_envelope(segments: list[dict[str, Any]], total: float,
                   fill_voiced: list[tuple[float, float]] = ()) -> np.ndarray:
    env = np.ones(int(total * CTRL_HZ) + 2, dtype=np.float32)
    for seg in segments:
        p = seg["place"]
        if seg["keep"]:
            _duck(env, p["start"], p["end"], DUCK_KEEP, 0.04, 0.06)
        else:
            _duck(env, p["start"], p["end"] + 0.40, DUCK_DUB, 0.08, 0.18)
    # A speaker restored by the vocals fill gets the same bed treatment as a
    # dubbed one; silent fill (no entry here) leaves the music alone.
    for a, b in fill_voiced:
        _duck(env, a, b + 0.40, DUCK_DUB, 0.08, 0.18)
    k = 12  # ~60ms smoothing so the bed never steps
    return np.convolve(env, np.ones(k, dtype=np.float32) / k, mode="same")


def unclaimed_spans(intervals: list[tuple[float, float]], total: float,
                    slop: float = FILL_SLOP,
                    min_len: float = FILL_MIN_SEC) -> list[tuple[float, float]]:
    """Complement of the placed intervals over [0, total].

    Intervals closer than `slop` are merged (a 50ms seam between two clips is
    not a hole), and complement slivers shorter than `min_len` are dropped —
    nothing intelligible fits in them and they would just chop the bed.
    """
    merged: list[list[float]] = []
    for a, b in sorted((a, b) for a, b in intervals if b > a):
        if merged and a <= merged[-1][1] + slop:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    spans: list[tuple[float, float]] = []
    t = 0.0
    for a, b in merged:
        if a - t >= min_len:
            spans.append((t, a))
        t = max(t, b)
    if total - t >= min_len:
        spans.append((t, total))
    return spans


def voiced_spans(a: np.ndarray, sr: int, floor: float = FILL_FLOOR,
                 hop_sec: float = 0.1) -> list[tuple[float, float]]:
    """Spans (relative to `a`) whose ~0.1s frame rms clears the speech floor."""
    levels = audio.frame_rms(a, sr, hop_sec)
    spans: list[tuple[float, float]] = []
    start: float | None = None
    for i, lv in enumerate(levels):
        if lv >= floor:
            if start is None:
                start = i * hop_sec
        elif start is not None:
            spans.append((start, i * hop_sec))
            start = None
    if start is not None:
        spans.append((start, len(levels) * hop_sec))
    return spans


def _read_mono(f: sf.SoundFile, start: int, n: int) -> np.ndarray:
    s = min(max(start, 0), f.frames)
    f.seek(s)
    a = f.read(min(n, f.frames - s), dtype="float32", always_2d=True)
    a = a.mean(axis=1).astype(np.float32)
    if len(a) < n:
        a = np.pad(a, (0, n - len(a)))
    return a


def load_clip(path: Path, keep: bool) -> np.ndarray:
    a, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(a, "ndim", 1) > 1:
        a = np.mean(a, axis=-1).astype(np.float32)
    r = audio.rms(a)
    if r > 1e-6:
        target, cap = (KEEP_RMS, KEEP_GAIN_CAP) if keep else (SPEECH_RMS, GAIN_CAP)
        a = a * min(cap, target / r)
    if keep:
        # Levelled above as a finished mix; undo the global speech gain so it is
        # not lifted a second time.
        a = a / SPEECH_GAIN
    fade = max(1, int(FADE_SEC * sr))
    if len(a) > 2 * fade:
        a[:fade] *= np.linspace(0.0, 1.0, fade)
        a[-fade:] *= np.linspace(1.0, 0.0, fade)
    return a.astype(np.float32)


def assemble(m: dict[str, Any], workdir: Path, raw: Path) -> float:
    """Stream the mixed track into `raw`; return its absolute peak."""
    sr = audio.SR
    segments = sorted(m["segments"], key=lambda s: s["place"]["start"])
    media_end = float(m["source"]["duration"])
    total = max(media_end, max((s["place"]["end"] for s in segments), default=0.0) + 0.2)

    # Spans the placements left empty: they carry the original vocals. The
    # timeline asserts each clip's length matches its place span within 20ms,
    # so the complement of the place intervals cannot collide with a clip.
    fill = unclaimed_spans(
        [(s["place"]["start"], s["place"]["end"]) for s in segments], media_end)
    voc_path = workdir / m["files"]["vocals"]
    voiced: list[tuple[float, float]] = []
    with sf.SoundFile(str(voc_path)) as vf:
        for t0, t1 in fill:
            s0 = int(t0 * sr)
            piece = _read_mono(vf, s0, int(t1 * sr) - s0)
            voiced += [(t0 + a, t0 + b) for a, b in voiced_spans(piece, sr)]

    env = build_envelope(segments, total, voiced)
    ctrl_t = np.arange(len(env), dtype=np.float32) / CTRL_HZ

    bg_path = workdir / m["files"]["background"]
    peak = 0.0
    idx = 0
    pending: list[tuple[int, np.ndarray]] = []   # (start sample, audio) not yet fully written
    fade_n = max(1, int(FILL_FADE_SEC * sr))

    with sf.SoundFile(str(bg_path)) as bg, sf.SoundFile(str(voc_path)) as vf, sf.SoundFile(
        str(raw), "w", samplerate=sr, channels=2, subtype="FLOAT"
    ) as out:
        pos = 0
        n_total = int(total * sr)
        block_n = int(BLOCK_SEC * sr)
        while pos < n_total:
            n = min(block_n, n_total - pos)
            bed = bg.read(n, dtype="float32", always_2d=True)
            if len(bed) < n:
                bed = np.pad(bed, ((0, n - len(bed)), (0, 0)))
            if bed.shape[1] == 1:
                bed = np.repeat(bed, 2, axis=1)

            speech = np.zeros(n, dtype=np.float32)
            occupied = np.zeros(n, dtype=bool)
            block_end = pos + n

            while idx < len(segments) and int(segments[idx]["place"]["start"] * sr) < block_end:
                seg = segments[idx]
                start = int(seg["place"]["start"] * sr)
                clip = load_clip(workdir / seg["place"]["clip"], seg["keep"])
                pending.append((start, clip))
                idx += 1

            still: list[tuple[int, np.ndarray]] = []
            for start, clip in pending:
                a = max(start, pos)
                b = min(start + len(clip), block_end)
                if b > a:
                    lo, hi = a - pos, b - pos
                    assert not occupied[lo:hi].any(), (
                        "speech overlap in mix — timeline invariant violated near "
                        f"{a / sr:.2f}s"
                    )
                    speech[lo:hi] += clip[a - start : b - start]
                    occupied[lo:hi] = True
                if start + len(clip) > block_end:
                    still.append((start, clip))
            pending = still

            # Designed final pass: original vocals into every unclaimed span.
            # Runs after all placed clips, so an overlap here would mean the
            # complement above disagrees with the placements — a real bug.
            for t0, t1 in fill:
                s0, s1 = int(t0 * sr), int(t1 * sr)
                a = max(s0, pos)
                b = min(s1, block_end)
                if b <= a:
                    continue
                lo, hi = a - pos, b - pos
                occ = occupied[lo:hi]
                # A placed clip's audio can run a few ms past its declared place
                # span (clip files are not sample-exact to the manifest), so the
                # complement may brush a clip's tail. Trim those samples from the
                # fill rather than double them; anything beyond ~0.1s means the
                # complement truly disagrees with the placements — a real bug.
                overlap = int(occ.sum())
                assert overlap <= int(0.1 * sr), (
                    "vocals fill overlaps a placed clip — unclaimed-span "
                    f"computation is wrong near {a / sr:.2f}s"
                )
                piece = _read_mono(vf, a, b - a)
                k = np.arange(a, b, dtype=np.float32)
                edge = np.minimum((k - s0) / fade_n, (s1 - k) / fade_n)
                piece *= np.clip(edge, 0.0, 1.0)
                if overlap:
                    piece[occ] = 0.0
                # Unity gain: undo the global speech lift applied below.
                speech[lo:hi] += piece / SPEECH_GAIN
                occupied[lo:hi] = True

            gain = np.interp(np.arange(pos, block_end, dtype=np.float32) / sr,
                             ctrl_t, env).astype(np.float32)
            mixed = bed * (BG_GAIN * gain)[:, None] + (speech * SPEECH_GAIN)[:, None]
            peak = max(peak, float(np.max(np.abs(mixed))) if mixed.size else 0.0)
            out.write(mixed)
            pos = block_end
    return peak


def build_track(m: dict[str, Any], workdir: Path) -> Path:
    sr = audio.SR
    raw = workdir / "dub_raw.wav"
    peak = assemble(m, workdir, raw)
    final = workdir / "dub.wav"
    volume = min(1.0, PEAK / peak) if peak > PEAK else 1.0
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(raw)]
    if volume < 1.0:
        cmd += ["-af", f"volume={volume:.4f}"]
        print(f"  mix: peak {peak:.2f} → limiting by {volume:.3f}", file=sys.stderr)
    cmd += ["-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "2", str(final)]
    audio.run(cmd)
    raw.unlink(missing_ok=True)
    return final


def _ts(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    mnt, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{mnt:02d}:{s:02d},{ms:03d}"


def write_srt(m: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    for n, seg in enumerate(sorted(m["segments"], key=lambda s: s["place"]["start"]), 1):
        p = seg["place"]
        text = (p.get("spoken") or seg.get("text_en") or "").strip()
        if text == "…":
            continue   # placeholder for an unreadable foreign span — not a subtitle
        if not text:
            continue
        lines += [str(n), f"{_ts(p['start'])} --> {_ts(p['end'])}", text, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# ISO-639-1 → ISO-639-2 (bibliographic, what mov_text metadata expects) for the
# Qwen3-TTS targets plus the pipeline's default source. Anything else is "und".
_ISO_639_2 = {
    "en": "eng", "zh": "chi", "de": "ger", "it": "ita", "pt": "por",
    "es": "spa", "ja": "jpn", "ko": "kor", "fr": "fre", "ru": "rus", "he": "heb",
}


def subtitle_lang3(code: str) -> str:
    return _ISO_639_2.get((code or "").lower(), "und")


def mux(video: Path, track: Path, srt: Path, out: Path, duration: float,
        lang: str = "en") -> None:
    audio.run([
        "ffmpeg", "-y", "-v", "error", "-i", str(video), "-i", str(track), "-i", str(srt),
        "-t", f"{duration:.3f}", "-map", "0:v:0", "-map", "1:a:0", "-map", "2:0",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k", "-c:s", "mov_text",
        "-metadata:s:s:0", f"language={subtitle_lang3(lang)}",
        "-shortest", "-movflags", "+faststart", str(out),
    ])


def run(m: dict[str, Any], workdir: Path) -> None:
    track = build_track(m, workdir)
    tgt = (m["source"].get("tgt_lang") or "en").lower()
    srt = workdir / f"preview_{tgt}.srt"
    write_srt(m, srt)
    preview = workdir / "preview.mp4"
    mux(Path(m["files"]["video"]), track, srt, preview, float(m["source"]["duration"]),
        lang=tgt)
    m["outputs"] = {"dub_wav": track.name, "srt": srt.name, "preview": preview.name}
    print(f"  mix: {preview.name}", file=sys.stderr)
