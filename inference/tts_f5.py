#!/usr/bin/env python3
"""F5-TTS zero-shot cloning for DubbingQwen (Phase 4).

Uses https://github.com/SWivid/F5-TTS with explicit speed control:
  speed > 1.0 → faster / shorter
  speed < 1.0 → slower / longer

Speaker refs: each segment clones from **its own** vocal window (not a global
SPEAKER_XX bank). Diarization IDs are often wrong on documentaries; the audio
in [start, end] is the ground-truth voice for that line.

Timing:
  - Cap auto-fit speed to [--fit-speed-min, --fit-speed-max] (default 0.85–1.25)
  - Place speech at segment start; pad trailing silence (do not drag to 0.5x)
  - If still too long at max speed, trim end (rare)

No lip-sync here — audio only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SPEED_MIN = 0.3
SPEED_MAX = 2.0
FIT_SPEED_MIN_DEFAULT = 0.85
FIT_SPEED_MAX_DEFAULT = 1.25
REF_TARGET_SEC = 6.0
REF_MIN_SEC = 2.5
REF_MAX_SEC = 11.5


def clamp_speed(speed: float, lo: float = SPEED_MIN, hi: float = SPEED_MAX) -> float:
    return max(lo, min(hi, float(speed)))


def extract_wav_slice(
    src: Path,
    start: float,
    end: float,
    dst: Path,
    *,
    sample_rate: int = 24000,
) -> Path:
    """Extract [start, end). Seek after -i for frame-accurate cuts."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    dur = max(end - start, 0.05)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{dur:.3f}",
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
    return dst


def extend_end_by_energy(
    vocals: Path,
    start: float,
    end: float,
    *,
    max_end: float,
    pad: float = 0.15,
    rms_thresh: float = 0.018,
    sample_rate: int = 16000,
    bridge_gap_sec: float = 0.45,
) -> float:
    """Push end forward while speech energy continues (fixes early ASR cuts).

    Bridges short dips (breaths) up to bridge_gap_sec so KEEP English isn't
    chopped mid-sentence when energy briefly drops.
    """
    import numpy as np

    probe_end = min(max_end, end + 2.5)
    if probe_end <= end + 0.05:
        return min(end + pad, max_end)
    dur = probe_end - start
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(vocals),
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{dur:.3f}",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    hop = max(1, sample_rate // 10)  # 100ms
    last_active = end
    quiet_run = 0.0
    for i in range(0, len(audio) - hop, hop):
        t = start + i / sample_rate
        if t < end - 0.05:
            continue
        if t >= max_end:
            break
        rms = float(np.sqrt(np.mean(audio[i : i + hop] ** 2) + 1e-12))
        if rms >= rms_thresh:
            last_active = t + hop / sample_rate
            quiet_run = 0.0
        else:
            quiet_run += hop / sample_rate
            if quiet_run >= bridge_gap_sec:
                break
    return min(max(last_active + pad, end), max_end)


def wav_duration(path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        text=True,
    ).strip()
    return float(out)


def pad_or_trim(src: Path, dst: Path, target_sec: float, sample_rate: int = 44100) -> None:
    """Pad with silence or trim to exact slot length (no pitch change)."""
    af = f"apad=whole_dur={target_sec:.6f},atrim=0:{target_sec:.6f}"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-af",
            af,
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


def detect_lead_glitch(
    src: Path,
    *,
    sample_rate: int = 44100,
    min_trim_sec: float = 0.10,
    max_trim_sec: float = 0.32,
) -> float:
    """Trim F5's leading moan / 'he' / click before real English.

    Cross-lingual F5 (Hebrew ref → English) often starts with a steady voiced
    vowel ("ohhhh", "heee") — high energy, low zero-crossing rate. Real English
    usually brings higher ZCR once consonants start.
    """
    import numpy as np

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-",
    ]
    audio = np.frombuffer(
        subprocess.run(cmd, capture_output=True, check=True).stdout, dtype=np.float32
    )
    if audio.size < sample_rate // 4:
        return min_trim_sec

    hop = max(1, int(0.02 * sample_rate))
    # Find first window that looks like articulated speech (not a steady moan)
    trim = min_trim_sec
    limit = min(len(audio) - hop, int(max_trim_sec * sample_rate))
    for i in range(0, limit, hop):
        win = audio[i : i + hop]
        rms = float(np.sqrt(np.mean(win**2) + 1e-12))
        if rms < 0.008:
            continue
        zcr = float(np.mean(np.abs(np.diff(np.signbit(win).astype(np.float32)))))
        # Moan/vowel pad: loud + very smooth (low zcr). Cut until zcr rises.
        t = i / sample_rate
        if t < min_trim_sec:
            continue
        if zcr >= 0.08:
            trim = max(min_trim_sec, t - 0.02)
            break
        trim = t + hop / sample_rate
    return float(min(max(trim, min_trim_sec), max_trim_sec))


def clean_f5_onset(
    src: Path,
    dst: Path,
    *,
    sample_rate: int = 44100,
    head_trim_sec: float | None = None,
    tail_trim_sec: float = 0.05,
    fade_in_sec: float = 0.06,
    fade_out_sec: float = 0.04,
) -> Path:
    """Drop F5 cross-lingual onset glitches (moans / 'he'), then soft-fade."""
    dur = wav_duration(src)
    if head_trim_sec is None:
        head_trim_sec = detect_lead_glitch(src, sample_rate=sample_rate)
    head = min(max(0.0, head_trim_sec), max(0.0, dur * 0.45))
    tail = min(max(0.0, tail_trim_sec), max(0.0, (dur - head) * 0.15))
    keep = max(0.25, dur - head - tail)
    fade_in = min(fade_in_sec, keep * 0.25)
    fade_out = min(fade_out_sec, keep * 0.15)
    af = (
        f"atrim={head:.4f}:{head + keep:.4f},asetpts=PTS-STARTPTS,"
        f"afade=t=in:st=0:d={fade_in:.4f},"
        f"afade=t=out:st={max(0.0, keep - fade_out):.4f}:d={fade_out:.4f}"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-af",
            af,
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
    print(f"      glitch trim {head:.2f}s / tail {tail:.2f}s", file=sys.stderr)
    return dst


def fit_phrase_natural(
    src: Path,
    dst: Path,
    target_sec: float,
    *,
    sample_rate: int = 44100,
    max_speedup: float = 1.08,
    max_overflow: float = 0.15,
) -> float:
    """Fit phrase into slot at near-natural pace.

    Returns the actual duration used (may slightly exceed target_sec by
    max_overflow so we don't swallow the last words with a hard cut).
    """
    actual = wav_duration(src)
    work = src
    if actual > target_sec + 0.12:
        rate = min(max_speedup, actual / target_sec)
        if rate > 1.02:
            sped = dst.with_name(dst.stem + "_pace.wav")
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
                    str(sped),
                ],
                check=True,
                capture_output=True,
            )
            work = sped
            actual = wav_duration(work)
            print(
                f"      gentle pace {rate:.2f}x → {actual:.2f}s (cap {max_speedup:.2f})",
                file=sys.stderr,
            )

    # Allow a little overflow rather than chopping the last syllable
    slot = max(target_sec, min(actual, target_sec + max_overflow))
    if actual > slot + 0.02:
        # Last resort: only then trim (fade the cut)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(work),
                "-af",
                f"atrim=0:{slot:.6f},asetpts=PTS-STARTPTS,"
                f"afade=t=out:st={max(0.0, slot - 0.08):.4f}:d=0.08",
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
        print(f"      soft-trim to {slot:.2f}s (kept +{slot - target_sec:.2f}s overflow)", file=sys.stderr)
        return slot

    pad_or_trim(work, dst, max(slot, target_sec), sample_rate)
    return max(slot, min(actual, target_sec) if actual < target_sec else actual)


def make_silence(path: Path, seconds: float, sample_rate: int = 44100) -> Path:
    seconds = max(0.02, float(seconds))
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r={sample_rate}:cl=mono",
            "-t",
            f"{seconds:.3f}",
            "-acodec",
            "pcm_s16le",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def concat_wavs(paths: list[Path], out: Path, sample_rate: int = 44100) -> Path:
    """Concatenate mono wavs with ffmpeg concat demuxer."""
    if len(paths) == 1:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(paths[0]),
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                str(out),
            ],
            check=True,
            capture_output=True,
        )
        return out
    list_file = out.with_suffix(".txt")
    list_file.write_text(
        "".join(f"file '{p.resolve()}'\n" for p in paths),
        encoding="utf-8",
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


def phrase_plan(seg: dict) -> list[dict]:
    """Return phrase dicts with English gen text, Hebrew ref text, and timings."""
    phrases = seg.get("phrases") or []
    if phrases:
        plan = []
        for p in phrases:
            text_en = (p.get("text_en") or "").strip()
            text_he = (p.get("text") or "").strip()
            # For Hebrew dubs: gen=EN, ref_text=HE. Fall back sensibly.
            gen = text_en or text_he
            if not gen:
                continue
            plan.append(
                {
                    "text": gen,
                    "text_he": text_he or text_en or gen,
                    "start": float(p.get("start", seg["start"])),
                    "end": float(p.get("end", seg["end"])),
                    "pause_after": float(p.get("pause_after") or 0.0),
                }
            )
        if plan:
            plan[-1]["pause_after"] = 0.0
            return plan
    text_en = (seg.get("text_en") or "").strip()
    text_he = (seg.get("text") or "").strip()
    gen = text_en or text_he
    if not gen:
        return []
    return [
        {
            "text": gen,
            "text_he": text_he or gen,
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "pause_after": 0.0,
        }
    ]


def atempo_chain(rate: float) -> str:
    """Build ffmpeg atempo filter chain (each stage must be 0.5–2.0)."""
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


def stretch_to_duration(
    src: Path,
    dst: Path,
    target_sec: float,
    *,
    sample_rate: int = 44100,
    min_rate: float = 0.7,
    max_rate: float = 1.35,
) -> None:
    """Time-stretch toward target_sec, then pad (never hard-trim speech)."""
    actual = wav_duration(src)
    if actual <= 0.05:
        pad_or_trim(src, dst, target_sec, sample_rate)
        return
    rate = actual / max(target_sec, 0.05)
    rate = max(min_rate, min(max_rate, rate))
    stretched = dst.with_name(dst.stem + "_stretch.wav")
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
            str(stretched),
        ],
        check=True,
        capture_output=True,
    )
    place_in_slot(stretched, dst, target_sec, sample_rate)


def place_in_slot(src: Path, dst: Path, target_sec: float, sample_rate: int = 44100) -> None:
    """Put speech at the start of the slot; silence fills the rest.

    Do not trim overlong speech — preserve the full phrase.
    """
    actual = wav_duration(src)
    if actual <= target_sec + 0.02:
        pad_or_trim(src, dst, target_sec, sample_rate)
        return
    print(
        f"    preserve overflow: generated {actual:.2f}s for slot {target_sec:.2f}s",
        file=sys.stderr,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
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


def assemble_phrase_timeline(
    phrases: list[dict],
    phrase_wavs: list[Path],
    seg_start: float,
    seg_end: float,
    out_path: Path,
    sample_rate: int = 44100,
) -> None:
    """Lay phrase clips onto a silence bed at their Hebrew timeline offsets."""
    target = max(seg_end - seg_start, 0.4)
    silence = out_path.with_name(out_path.stem + "_bed.wav")
    make_silence(silence, target, sample_rate=sample_rate)

    if not phrases or not phrase_wavs:
        pad_or_trim(silence, out_path, target, sample_rate)
        return

    inputs: list[str] = ["-i", str(silence)]
    filters: list[str] = []
    labels: list[str] = ["[0:a]"]
    for i, (phrase, wav) in enumerate(zip(phrases, phrase_wavs)):
        inputs.extend(["-i", str(wav)])
        delay_ms = int(round(max(0.0, float(phrase["start"]) - seg_start) * 1000))
        label = f"p{i}"
        filters.append(f"[{i + 1}:a]adelay={delay_ms}|{delay_ms}[{label}]")
        labels.append(f"[{label}]")

    n = len(labels)
    filters.append(
        f"{''.join(labels)}amix=inputs={n}:normalize=0:dropout_transition=0,"
        f"atrim=0:{target:.6f},apad=whole_dur={target:.6f}[out]"
    )
    cmd = [
        "ffmpeg",
        "-y",
        *inputs,
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[out]",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-t",
        f"{target:.3f}",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-1500:], file=sys.stderr)
        raise RuntimeError(f"phrase timeline assemble failed ({result.returncode})")



def segment_ref_window(
    seg: dict,
    all_segments: list[dict],
    media_duration: float,
) -> tuple[float, float, str]:
    """Choose a ref window centered on this segment's own speech.

    Prefer the segment itself. If too short for F5, expand into neighboring
    audio on the timeline (same continuous documentary speech), not a random
    SPEAKER_XX bank entry.
    """
    start = float(seg["start"])
    end = float(seg["end"])
    text = (seg.get("text") or "").strip() or "um"
    dur = end - start

    if REF_MIN_SEC <= dur <= REF_MAX_SEC:
        return start, end, text

    if dur > REF_MAX_SEC:
        # Take the first REF_TARGET_SEC of this turn
        return start, start + REF_TARGET_SEC, text

    # Too short: expand symmetrically, clamped to media / neighbors
    need = REF_TARGET_SEC - dur
    left = need / 2.0
    right = need - left
    ref_start = max(0.0, start - left)
    ref_end = min(media_duration, end + right)
    # If still short (near edges), take more from the other side
    if ref_end - ref_start < REF_MIN_SEC:
        ref_end = min(media_duration, ref_start + REF_TARGET_SEC)
    if ref_end - ref_start < REF_MIN_SEC:
        ref_start = max(0.0, ref_end - REF_TARGET_SEC)

    # Collect Hebrew text overlapping the expanded window for ref_text
    bits: list[str] = []
    for s in all_segments:
        if float(s["end"]) <= ref_start or float(s["start"]) >= ref_end:
            continue
        t = (s.get("text") or "").strip()
        if t:
            bits.append(t)
    ref_text = " ".join(bits) if bits else text
    return ref_start, min(ref_end, ref_start + REF_MAX_SEC), ref_text


def build_segment_ref(
    seg: dict,
    index: int,
    all_segments: list[dict],
    vocals: Path,
    out_dir: Path,
    media_duration: float,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_start, ref_end, ref_text = segment_ref_window(seg, all_segments, media_duration)
    ref_path = out_dir / f"seg_{index:02d}_ref.wav"
    extract_wav_slice(vocals, ref_start, ref_end, ref_path)
    return {
        "path": str(ref_path.resolve()),
        "ref_text": ref_text,
        "start": ref_start,
        "end": ref_end,
    }


def build_phrase_ref(
    phrase: dict,
    seg: dict,
    seg_index: int,
    phrase_index: int,
    all_segments: list[dict],
    vocals: Path,
    out_dir: Path,
    media_duration: float,
) -> dict:
    """Voice ref = this phrase's Hebrew audio + matching Hebrew text.

    Critical: F5 sizes output from len(gen_text)/len(ref_text). Using the whole
    utterance as ref_text makes short English phrases collapse to ~1s gibberish.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    p_start = float(phrase["start"])
    p_end = float(phrase["end"])
    he_text = (phrase.get("text") or seg.get("text") or "um").strip() or "um"

    # Expand short phrases so F5 has enough reference audio
    ref_start, ref_end = p_start, p_end
    dur = ref_end - ref_start
    if dur < REF_MIN_SEC:
        need = REF_TARGET_SEC - dur
        ref_start = max(0.0, p_start - need * 0.35)
        ref_end = min(media_duration, p_end + need * 0.65)
        if ref_end - ref_start < REF_MIN_SEC:
            ref_end = min(media_duration, ref_start + REF_TARGET_SEC)
        # Prefer Hebrew text that overlaps the expanded window
        bits: list[str] = []
        for s in all_segments:
            if s.get("keep_original"):
                continue
            for p in s.get("phrases") or [{"text": s.get("text"), "start": s["start"], "end": s["end"]}]:
                if float(p.get("end", 0)) <= ref_start or float(p.get("start", 0)) >= ref_end:
                    continue
                t = (p.get("text") or "").strip()
                if t:
                    bits.append(t)
        if bits:
            he_text = " ".join(bits)

    if ref_end - ref_start > REF_MAX_SEC:
        # Keep audio around the phrase itself
        mid = (p_start + p_end) / 2
        ref_start = max(0.0, mid - REF_TARGET_SEC / 2)
        ref_end = min(media_duration, ref_start + REF_TARGET_SEC)
        he_text = (phrase.get("text") or he_text).strip() or he_text

    ref_path = out_dir / f"seg_{seg_index:02d}_p{phrase_index:02d}_ref.wav"
    extract_wav_slice(vocals, ref_start, ref_end, ref_path)
    return {
        "path": str(ref_path.resolve()),
        "ref_text": he_text,
        "start": ref_start,
        "end": ref_end,
    }


class F5Synthesizer:
    def __init__(self, model: str = "F5TTS_v1_Base", device: str | None = None):
        from f5_tts.api import F5TTS

        print(f"Loading F5-TTS ({model})…", file=sys.stderr)
        self.tts = F5TTS(model=model, device=device)
        self.sample_rate = self.tts.target_sample_rate

    def synthesize(
        self,
        *,
        ref_audio: Path,
        ref_text: str,
        gen_text: str,
        out_wav: Path,
        speed: float = 1.0,
        fix_duration: float | None = None,
        seed: int | None = None,
        nfe_step: int = 32,
    ) -> Path:
        speed = clamp_speed(speed)
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        kwargs = dict(
            ref_file=str(ref_audio),
            ref_text=ref_text,
            gen_text=gen_text,
            file_wave=str(out_wav),
            speed=speed,
            seed=seed,
            remove_silence=False,
            nfe_step=nfe_step,
        )
        if fix_duration is not None and fix_duration > 0.2:
            # Force mel length ≈ Hebrew phrase window (seconds).
            kwargs["fix_duration"] = float(fix_duration)
        self.tts.infer(**kwargs)
        return out_wav

    def synthesize_fit(
        self,
        *,
        ref_audio: Path,
        ref_text: str,
        gen_text: str,
        out_raw: Path,
        out_fit: Path,
        target_sec: float,
        speed: float = 1.0,
        fit_duration: bool = True,
        fit_speed_min: float = FIT_SPEED_MIN_DEFAULT,
        fit_speed_max: float = FIT_SPEED_MAX_DEFAULT,
        seed: int | None = None,
    ) -> tuple[Path, float]:
        speed = clamp_speed(speed)
        self.synthesize(
            ref_audio=ref_audio,
            ref_text=ref_text,
            gen_text=gen_text,
            out_wav=out_raw,
            speed=speed,
            seed=seed,
        )
        actual = wav_duration(out_raw)
        final_speed = speed

        if fit_duration and target_sec > 0.2 and actual > 0.05:
            # Only nudge within a narrow band — never force 0.5x / 2.0x mush
            needed = clamp_speed(speed * (actual / target_sec), fit_speed_min, fit_speed_max)
            if abs(needed - speed) >= 0.05:
                print(
                    f"    fit: {actual:.2f}s → slot {target_sec:.2f}s; "
                    f"speed {speed:.2f} → {needed:.2f} (capped {fit_speed_min}-{fit_speed_max})",
                    file=sys.stderr,
                )
                mid = out_raw.with_name(out_raw.stem + "_fitpass.wav")
                self.synthesize(
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    gen_text=gen_text,
                    out_wav=mid,
                    speed=needed,
                    seed=seed,
                )
                out_raw = mid
                final_speed = needed
                actual = wav_duration(out_raw)
            else:
                print(
                    f"    keep speed={speed:.2f}: raw {actual:.2f}s vs slot {target_sec:.2f}s "
                    f"(pad/trim only)",
                    file=sys.stderr,
                )

        place_in_slot(out_raw, out_fit, target_sec, sample_rate=44100)
        return out_fit, final_speed


def synthesize_segments_f5(
    segments: list[dict],
    vocals: Path,
    workdir: Path,
    *,
    speed: float = 1.0,
    fit_duration: bool = True,
    fit_speed_min: float = FIT_SPEED_MIN_DEFAULT,
    fit_speed_max: float = FIT_SPEED_MAX_DEFAULT,
    model: str = "F5TTS_v1_Base",
    device: str | None = None,
    selected_indices: set[int] | None = None,
    merge_pauses: bool = True,
    max_pause: float = 0.75,
) -> list[dict]:
    from inference.segment_merge import merge_same_speaker_segments

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
    syn = F5Synthesizer(model=model, device=device)
    refs_meta: dict[str, dict] = {}

    for i, seg in enumerate(segments):
        start = float(seg["start"])
        end = float(seg["end"])
        target = max(end - start, 0.4)
        lang = seg.get("language") or "he"
        keep_original = bool(seg.get("keep_original", lang != "he"))

        # Cap expansion so we don't eat into the next utterance
        next_start = media_duration
        if i + 1 < len(segments):
            next_start = float(segments[i + 1]["start"])

        # Always (re)materialize keep-original clips — selective --tts-segments must
        # never drop EN/AR originals, and we always refresh so ends aren't truncated.
        if keep_original:
            # Stop just before the next utterance (Hebrew dub or another KEEP).
            max_end = max(start + 0.4, next_start - 0.05)
            extended_end = extend_end_by_energy(
                vocals,
                start,
                end,
                max_end=max_end,
                pad=0.18,
                rms_thresh=0.016,
                bridge_gap_sec=0.55,
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
            seg["end"] = round(end, 3)
            seg["duration"] = round(end - start, 3)
            if seg.get("phrases"):
                seg["phrases"][-1]["end"] = seg["end"]
                # Keep original text for subs
                if not (seg["phrases"][-1].get("text_en") or "").strip():
                    seg["phrases"][-1]["text_en"] = seg["phrases"][-1].get("text") or ""
            target = max(end - start, 0.4)
            raw = tts_dir / f"seg_{i:02d}_orig.wav"
            fitted = tts_dir / f"seg_{i:02d}_fit.wav"
            extract_wav_slice(vocals, start, end, raw, sample_rate=44100)
            # Copy full original speech into the slot (pad only if short — never
            # hard-trim the English tail).
            place_in_slot(raw, fitted, target, sample_rate=44100)
            seg.pop("tts_text", None)
            seg["tts_raw"] = str(raw)
            seg["tts_fit"] = str(fitted)
            seg["tts_speed_used"] = 1.0
            seg["keep_original"] = True
            continue

        if selected_indices is not None and i not in selected_indices:
            # Re-attach conventional fit path if JSON lost it but the file remains.
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

        plan = phrase_plan(seg)
        if not plan:
            continue

        seg_speed = clamp_speed(float(seg.get("tts_speed", speed)))
        print(
            f"  F5 [{seg['speaker_id']}] he slot {start:.1f}-{end:.1f}s "
            f"{len(plan)} phrase(s) speed={seg_speed:.2f} (fix_duration + phrase refs)",
            file=sys.stderr,
        )

        phrase_wavs: list[Path] = []
        used_speeds: list[float] = []
        first_ref_meta: dict | None = None
        for j, phrase in enumerate(plan):
            text = phrase["text"]
            p_start = float(phrase["start"])
            p_end = float(phrase["end"])
            # Soft cap: keep a small gap, but allow later overflow into that gap
            # so we don't hard-cut the last words.
            hard_cap = (
                float(plan[j + 1]["start"]) - 0.04
                if j + 1 < len(plan)
                else min(float(seg["end"]) + 0.35, next_start - 0.05)
            )
            p_end = min(p_end, hard_cap)
            p_target = max(p_end - p_start, 0.35)
            phrase["start"] = p_start
            phrase["end"] = p_end

            he_text = (phrase.get("text_he") or phrase.get("text") or "").strip() or "um"
            ref = build_phrase_ref(
                {
                    "start": p_start,
                    "end": p_end,
                    "text": he_text,
                },
                seg,
                i,
                j,
                segments,
                vocals,
                ref_dir,
                media_duration,
            )
            ref["ref_text"] = he_text
            refs_meta[f"seg_{i:02d}_p{j:02d}"] = ref
            if first_ref_meta is None:
                first_ref_meta = ref

            print(
                f"    phrase {j} [{p_start:.1f}-{p_end:.1f}s] "
                f"ref={ref['start']:.1f}-{ref['end']:.1f}s slot={p_target:.2f}s: "
                f"{text[:55]}…",
                file=sys.stderr,
            )
            chunk = tts_dir / f"seg_{i:02d}_p{j:02d}.wav"
            # Always speed=1.0 — mid-sentence speed-up/down is what the user hears
            # as "speeds ups speeds downs". Never re-synth at other rates.
            syn.synthesize(
                ref_audio=Path(ref["path"]),
                ref_text=ref["ref_text"],
                gen_text=text,
                out_wav=chunk,
                speed=1.0,
                fix_duration=None,
            )
            chunk_441 = tts_dir / f"seg_{i:02d}_p{j:02d}_441.wav"
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

            cleaned = tts_dir / f"seg_{i:02d}_p{j:02d}_clean.wav"
            clean_f5_onset(chunk_441, cleaned, sample_rate=44100)
            actual = wav_duration(cleaned)
            print(f"      clean {actual:.2f}s @ speed 1.0 (no atempo)", file=sys.stderr)
            phrase["tts_fit"] = str(cleaned)
            phrase_wavs.append(cleaned)
            used_speeds.append(1.0)

        # Concatenate phrases with short breaths; pad once at the end — never atempo.
        concat_parts: list[Path] = []
        for j, phrase in enumerate(plan):
            concat_parts.append(Path(phrase["tts_fit"]))
            pause = float(phrase.get("pause_after") or 0.0)
            if j < len(plan) - 1:
                pause = min(max(pause, 0.08), 0.18)
                sil = tts_dir / f"seg_{i:02d}_p{j:02d}_breath.wav"
                make_silence(sil, pause, sample_rate=44100)
                concat_parts.append(sil)

        if i > 0:
            prev_end = float(segments[i - 1]["end"])
            if start - prev_end > 0.25:
                new_start = prev_end + 0.08
                print(
                    f"    pull start {start:.2f} → {new_start:.2f} "
                    f"(was late after prev end {prev_end:.2f})",
                    file=sys.stderr,
                )
                start = new_start
                seg["start"] = round(start, 3)

        target = max(float(seg["end"]) - start, 0.4)
        raw = tts_dir / f"seg_{i:02d}_raw.wav"
        fitted = tts_dir / f"seg_{i:02d}_fit.wav"
        concat_wavs(concat_parts, raw, sample_rate=44100)
        spoken = wav_duration(raw)
        print(
            f"    concat {spoken:.2f}s into slot {target:.2f}s "
            f"({len(plan)} phrases, constant pace)",
            file=sys.stderr,
        )
        if spoken > target + 0.05:
            # Soft-trim the tail only — do NOT time-stretch (uneven pace).
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(raw),
                    "-af",
                    f"atrim=0:{target:.6f},asetpts=PTS-STARTPTS,"
                    f"afade=t=out:st={max(0.0, target - 0.08):.4f}:d=0.08",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "44100",
                    "-ac",
                    "1",
                    str(fitted),
                ],
                check=True,
                capture_output=True,
            )
            print(f"    soft-trim concat to {target:.2f}s (no speed change)", file=sys.stderr)
        else:
            place_in_slot(raw, fitted, target, sample_rate=44100)
        seg["end"] = round(start + target, 3)
        seg["duration"] = round(target, 3)

        seg.pop("tts_text", None)
        seg["tts_raw"] = str(raw)
        seg["tts_fit"] = str(fitted)
        seg["tts_speed_used"] = (
            sum(used_speeds) / len(used_speeds) if used_speeds else seg_speed
        )
        seg["keep_original"] = False
        if first_ref_meta:
            seg["ref_audio"] = first_ref_meta["path"]
            seg["ref_start"] = first_ref_meta["start"]
            seg["ref_end"] = first_ref_meta["end"]
        if seg.get("phrases"):
            for src_p, out_p in zip(seg["phrases"], plan):
                src_p["text_en"] = out_p["text"]
                if out_p.get("tts_fit"):
                    src_p["tts_fit"] = out_p["tts_fit"]

    (ref_dir / "refs.json").write_text(
        json.dumps(refs_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return segments


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="F5-TTS synthesis for translated segments.")
    p.add_argument("workdir", type=Path, help="Run dir with translated_segments.json + vocals.wav")
    p.add_argument("--tts-speed", type=float, default=1.0, help="F5 speed (0.3–2.0). >1 = faster.")
    p.add_argument(
        "--tts-fit-duration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Nudge speed within --fit-speed-* to approach slot length (default: on).",
    )
    p.add_argument("--fit-speed-min", type=float, default=FIT_SPEED_MIN_DEFAULT)
    p.add_argument("--fit-speed-max", type=float, default=FIT_SPEED_MAX_DEFAULT)
    p.add_argument("--model", default="F5TTS_v1_Base")
    p.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    p.add_argument(
        "--max-pause",
        type=float,
        default=1.0,
        help="Merge same-speaker gaps up to this many seconds into one utterance with pauses.",
    )
    p.add_argument(
        "--no-merge-pauses",
        action="store_true",
        help="Do not merge same-speaker short pauses before TTS.",
    )
    p.add_argument(
        "--segments",
        default=None,
        help="Only regenerate comma-separated zero-based segment indices.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    workdir = args.workdir
    path = workdir / "translated_segments.json"
    if not path.is_file():
        path = workdir / "segments.json"
    if not path.is_file():
        raise SystemExit(f"Missing segments JSON in {workdir}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    segments = payload.get("segments", [])
    vocals = Path(payload.get("vocals") or workdir / "vocals.wav")
    if not vocals.is_file():
        raise SystemExit(f"Missing vocals: {vocals}")
    if not any((s.get("text_en") or s.get("text") or "").strip() for s in segments):
        raise SystemExit("No text_en/text on segments — translate first.")

    synthesize_segments_f5(
        segments,
        vocals,
        workdir,
        speed=args.tts_speed,
        fit_duration=args.tts_fit_duration,
        fit_speed_min=args.fit_speed_min,
        fit_speed_max=args.fit_speed_max,
        model=args.model,
        device=args.device,
        selected_indices=(
            {int(value) for value in args.segments.split(",")} if args.segments else None
        ),
        merge_pauses=not args.no_merge_pauses,
        max_pause=args.max_pause,
    )
    out = workdir / "translated_segments.json"
    payload["segments"] = segments
    payload["tts_engine"] = "f5-tts"
    payload["tts_speed"] = args.tts_speed
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
