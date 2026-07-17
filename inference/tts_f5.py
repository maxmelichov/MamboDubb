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


def extract_wav_slice(src: Path, start: float, end: float, dst: Path) -> Path:
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
            "-acodec",
            "pcm_s16le",
            "-ar",
            "24000",
            "-ac",
            "1",
            str(dst),
        ],
        check=True,
        capture_output=True,
    )
    return dst


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
    """Return [{text_en, pause_after}, ...] for pause-aware TTS."""
    phrases = seg.get("phrases") or []
    if phrases:
        plan = []
        for p in phrases:
            text = (p.get("text_en") or p.get("text") or "").strip()
            if not text:
                continue
            plan.append(
                {
                    "text": text,
                    "pause_after": float(p.get("pause_after") or 0.0),
                }
            )
        if plan:
            plan[-1]["pause_after"] = 0.0
            return plan
    text = (seg.get("text_en") or seg.get("text") or "").strip()
    return [{"text": text, "pause_after": 0.0}] if text else []


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
        seed: int | None = None,
    ) -> Path:
        speed = clamp_speed(speed)
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        self.tts.infer(
            ref_file=str(ref_audio),
            ref_text=ref_text,
            gen_text=gen_text,
            file_wave=str(out_wav),
            speed=speed,
            seed=seed,
            remove_silence=False,
        )
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
        if selected_indices is not None and i not in selected_indices:
            continue
        plan = phrase_plan(seg)
        if not plan:
            continue

        ref = build_segment_ref(seg, i, segments, vocals, ref_dir, media_duration)
        refs_meta[f"seg_{i:02d}"] = ref

        start = float(seg["start"])
        end = float(seg["end"])
        target = max(end - start, 0.4)
        seg_speed = clamp_speed(float(seg.get("tts_speed", speed)))
        
        is_hebrew = seg.get("language", "he") == "he"
        if not is_hebrew:
            print(
                f"  Skipping TTS for non-Hebrew segment [{seg['speaker_id']}] "
                f"slot {start:.1f}-{end:.1f}s ({seg.get('language')})", 
                file=sys.stderr
            )
            raw = tts_dir / f"seg_{i:02d}_orig.wav"
            extract_wav_slice(vocals, start, end, raw)
            fitted = tts_dir / f"seg_{i:02d}_orig_fit.wav"
            place_in_slot(raw, fitted, target, sample_rate=44100)
            
            seg.pop("tts_text", None)
            seg["tts_raw"] = str(raw)
            seg["tts_fit"] = str(fitted)
            seg["tts_speed_used"] = 1.0
            seg["ref_audio"] = ref["path"]
            seg["ref_start"] = ref["start"]
            seg["ref_end"] = ref["end"]
            continue

        chunk_paths: list[Path] = []
        print(
            f"  F5 [{seg['speaker_id']}] slot {start:.1f}-{end:.1f}s "
            f"{len(plan)} phrase(s) ref={ref['start']:.1f}-{ref['end']:.1f}s "
            f"speed={seg_speed:.2f}",
            file=sys.stderr,
        )
        for j, phrase in enumerate(plan):
            text = phrase["text"]
            pause_after = float(phrase.get("pause_after") or 0.0)
            print(f"    phrase {j}: {text[:60]}…", file=sys.stderr)
            chunk = tts_dir / f"seg_{i:02d}_p{j:02d}.wav"
            syn.synthesize(
                ref_audio=Path(ref["path"]),
                ref_text=ref["ref_text"],
                gen_text=text,
                out_wav=chunk,
                speed=seg_speed,
            )
            # Resample to 44.1k mono for concat
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
            chunk_paths.append(chunk_441)
            if pause_after >= 0.05:
                sil = tts_dir / f"seg_{i:02d}_p{j:02d}_silence.wav"
                make_silence(sil, pause_after, sample_rate=44100)
                chunk_paths.append(sil)
                print(f"    + pause {pause_after:.2f}s", file=sys.stderr)

        raw = tts_dir / f"seg_{i:02d}_raw.wav"
        fitted = tts_dir / f"seg_{i:02d}_fit.wav"
        concat_wavs(chunk_paths, raw, sample_rate=44100)

        # Optional light speed nudge on the whole concatenated utterance
        used = seg_speed
        if fit_duration:
            actual = wav_duration(raw)
            needed = clamp_speed(seg_speed * (actual / target), fit_speed_min, fit_speed_max)
            if abs(needed - seg_speed) >= 0.05 and abs(actual - target) > 0.35:
                print(
                    f"    utterance fit: {actual:.2f}s → slot {target:.2f}s; "
                    f"re-speed {seg_speed:.2f} → {needed:.2f}",
                    file=sys.stderr,
                )
                # Re-synth all phrases at needed speed (keeps pauses absolute)
                chunk_paths = []
                for j, phrase in enumerate(plan):
                    chunk = tts_dir / f"seg_{i:02d}_p{j:02d}.wav"
                    syn.synthesize(
                        ref_audio=Path(ref["path"]),
                        ref_text=ref["ref_text"],
                        gen_text=phrase["text"],
                        out_wav=chunk,
                        speed=needed,
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
                    chunk_paths.append(chunk_441)
                    pause_after = float(phrase.get("pause_after") or 0.0)
                    if pause_after >= 0.05:
                        sil = tts_dir / f"seg_{i:02d}_p{j:02d}_silence.wav"
                        make_silence(sil, pause_after, sample_rate=44100)
                        chunk_paths.append(sil)
                concat_wavs(chunk_paths, raw, sample_rate=44100)
                used = needed

        place_in_slot(raw, fitted, target, sample_rate=44100)
        seg.pop("tts_text", None)
        seg["tts_raw"] = str(raw)
        seg["tts_fit"] = str(fitted)
        seg["tts_speed_used"] = used
        seg["ref_audio"] = ref["path"]
        seg["ref_start"] = ref["start"]
        seg["ref_end"] = ref["end"]

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
