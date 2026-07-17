#!/usr/bin/env python3
"""Build a watchable preview from Phase 1–2 outputs.

Takes a run directory (e.g. outputs/kan11_60s) with:
  segments.json, vocals.wav, background.wav, and the source video

Produces:
  translated_segments.json
  preview_en.srt
  dubbed_audio.wav
  preview.mp4   ← video + EN F5-TTS over ducked BGM + soft EN subs

TTS: F5-TTS (https://github.com/SWivid/F5-TTS)
  --tts-speed 1.0          manual rate (0.3–2.0; >1 = faster/shorter)
  --tts-fit-duration       nudge speed within 0.85–1.25 toward slot length
  Per-segment voice ref from that line's own vocals (not SPEAKER_XX bank)
No lip-sync yet (LatentSync = Phase 5).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from inference.tts_f5 import synthesize_segments_f5, wav_duration as f5_wav_duration

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRANSLATE_MODEL = REPO_ROOT / "models" / "translategemma-4b-it"
MIN_SEG_DURATION = 0.35


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.bfloat16


def load_translator(model_path: Path, device: torch.device):
    dtype = resolve_dtype(device)
    print(f"Loading TranslateGemma from {model_path} on {device} ({dtype})...", file=sys.stderr)
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return processor, model, dtype


def translate_text(
    processor,
    model,
    device: torch.device,
    dtype: torch.dtype,
    text: str,
    source: str = "he",
    target: str = "en",
    max_new_tokens: int = 256,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": source,
                    "target_lang_code": target,
                    "text": text.strip(),
                }
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device, dtype=dtype)
    input_length = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    return processor.decode(generated[0, input_length:], skip_special_tokens=True).strip()


def srt_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - math.floor(seconds)) * 1000))
    if ms == 1000:
        s += 1
        ms = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments: list[dict], path: Path, text_key: str = "text_en") -> None:
    lines: list[str] = []
    idx = 1
    for seg in segments:
        text = (seg.get(text_key) or "").strip()
        if not text:
            continue
        start = float(seg["start"])
        end = max(float(seg["end"]), start + 0.4)
        speaker = seg.get("speaker_id", "")
        lines.append(str(idx))
        lines.append(f"{srt_timestamp(start)} --> {srt_timestamp(end)}")
        lines.append(f"[{speaker}] {text}" if speaker else text)
        lines.append("")
        idx += 1
    path.write_text("\n".join(lines), encoding="utf-8")


def wav_duration(path: Path) -> float:
    return f5_wav_duration(path)


def build_dubbed_track(
    segments: list[dict],
    background: Path,
    total_duration: float,
    workdir: Path,
) -> Path:
    """Place pre-rendered F5 TTS clips on a timeline over ducked background."""
    silence = workdir / "silence.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=44100:cl=mono",
            "-t",
            f"{total_duration:.3f}",
            "-acodec",
            "pcm_s16le",
            str(silence),
        ],
        check=True,
        capture_output=True,
    )

    inputs = ["-i", str(background), "-i", str(silence)]
    filter_parts: list[str] = [
        f"[0:a]atrim=0:{total_duration:.3f},asetpts=PTS-STARTPTS,volume=0.35[bg]",
        f"[1:a]atrim=0:{total_duration:.3f},asetpts=PTS-STARTPTS[bed]",
    ]

    speech_labels: list[str] = []
    next_idx = 2
    for i, seg in enumerate(segments):
        fitted = seg.get("tts_fit")
        if not fitted or not Path(fitted).is_file():
            continue
        start = float(seg["start"])
        delay_ms = int(round(start * 1000))
        inputs.extend(["-i", str(fitted)])
        label = f"s{i}"
        filter_parts.append(f"[{next_idx}:a]adelay={delay_ms}|{delay_ms},volume=1.25[{label}]")
        speech_labels.append(f"[{label}]")
        next_idx += 1

    if speech_labels:
        n = 1 + len(speech_labels)
        mix_inputs = "[bed]" + "".join(speech_labels)
        filter_parts.append(f"{mix_inputs}amix=inputs={n}:normalize=0:dropout_transition=0[speech]")
        filter_parts.append("[bg][speech]amix=inputs=2:normalize=0:dropout_transition=0[out]")
    else:
        filter_parts.append("[bg]acopy[out]")

    out_wav = workdir / "dubbed_audio.wav"
    cmd = [
        "ffmpeg",
        "-y",
        *inputs,
        "-filter_complex",
        ";".join(filter_parts),
        "-map",
        "[out]",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "44100",
        "-t",
        f"{total_duration:.3f}",
        str(out_wav),
    ]
    print("Mixing dubbed audio…", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"ffmpeg mix failed (exit {result.returncode})")
    return out_wav


def find_source_video(workdir: Path, payload: dict) -> Path:
    src = Path(payload.get("source") or "")
    if src.is_file():
        return src
    p = workdir / "source_video.mp4"
    if p.is_file():
        return p
    for p in workdir.glob("*.mp4"):
        if p.name != "preview.mp4":
            return p
    raise SystemExit(f"No source video found in {workdir}")


def mux_preview(
    video: Path,
    audio: Path,
    srt: Path,
    out_mp4: Path,
    duration: float,
) -> None:
    print(f"Muxing preview → {out_mp4}", file=sys.stderr)
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-i",
            str(audio),
            "-i",
            str(srt),
            "-t",
            f"{duration:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-map",
            "2:0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-c:s",
            "mov_text",
            "-metadata:s:s:0",
            "language=eng",
            "-shortest",
            "-movflags",
            "+faststart",
            str(out_mp4),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"ffmpeg mux failed (exit {result.returncode})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a watchable dubbed preview from extract outputs.")
    p.add_argument(
        "workdir",
        type=Path,
        nargs="?",
        default=REPO_ROOT / "outputs" / "kan11_60s",
        help="Run directory with segments.json + stems (default: outputs/kan11_60s).",
    )
    p.add_argument("--model", type=Path, default=DEFAULT_TRANSLATE_MODEL)
    p.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    p.add_argument("--skip-translate", action="store_true", help="Reuse existing text_en in JSON.")
    p.add_argument("--skip-tts", action="store_true", help="Keep Hebrew vocals; only add EN subs.")
    p.add_argument(
        "--tts-speed",
        type=float,
        default=1.0,
        help="F5-TTS speed 0.3–2.0 (default 1.0). >1 faster/shorter; <1 slower/longer.",
    )
    p.add_argument(
        "--tts-fit-duration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Nudge F5 speed toward slot length within --fit-speed-* (default: on).",
    )
    p.add_argument("--fit-speed-min", type=float, default=0.85)
    p.add_argument("--fit-speed-max", type=float, default=1.25)
    p.add_argument(
        "--max-pause",
        type=float,
        default=1.0,
        help="Merge same-speaker gaps ≤ this into one utterance; TTS inserts silence for the pause.",
    )
    p.add_argument(
        "--tts-segments",
        default=None,
        help="Only regenerate comma-separated zero-based segment indices.",
    )
    p.add_argument(
        "--reuse-tts",
        action="store_true",
        help="Reuse existing tts_fit files and only remix/mux.",
    )
    p.add_argument("--f5-model", default="F5TTS_v1_Base", help="F5-TTS model name.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    workdir = args.workdir
    seg_path = workdir / "segments.json"
    if not seg_path.is_file():
        raise SystemExit(f"Missing {seg_path}")

    payload = json.loads(seg_path.read_text(encoding="utf-8"))
    translated_path = workdir / "translated_segments.json"
    if args.skip_translate and translated_path.is_file():
        payload = json.loads(translated_path.read_text(encoding="utf-8"))
        print(f"Loaded translations from {translated_path}", file=sys.stderr)

    segments = [
        s
        for s in payload.get("segments", [])
        if float(s.get("duration", s["end"] - s["start"])) >= MIN_SEG_DURATION
        and (s.get("text") or "").strip()
    ]
    if not segments:
        raise SystemExit("No usable segments after filtering micro-turns.")

    if args.skip_translate and all(s.get("text_en") for s in segments):
        print("Reusing existing English translations.", file=sys.stderr)
    else:
        if not args.model.is_dir():
            raise SystemExit(f"Translate model not found: {args.model}")
        device = resolve_device(args.device)
        processor, model, dtype = load_translator(args.model, device)
        for i, seg in enumerate(segments):
            he = seg["text"].strip()
            if seg.get("language", "he") != "he":
                print(f"Skipping translation for non-Hebrew segment [{i+1}/{len(segments)}]: {he[:50]}…", file=sys.stderr)
                seg["text_en"] = he
                continue
                
            print(f"Translating [{i+1}/{len(segments)}] {he[:50]}…", file=sys.stderr)
            seg["text_en"] = translate_text(processor, model, device, dtype, he)
            print(f"  → {seg['text_en']}", file=sys.stderr)
        del model, processor
        if device.type == "mps":
            torch.mps.empty_cache()

    out_json = workdir / "translated_segments.json"
    payload_out = {**payload, "segments": segments}
    out_json.write_text(json.dumps(payload_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}", file=sys.stderr)

    srt_path = workdir / "preview_en.srt"
    write_srt(segments, srt_path)
    print(f"Wrote {srt_path}", file=sys.stderr)

    background = Path(payload.get("background") or workdir / "background.wav")
    vocals = Path(payload.get("vocals") or workdir / "vocals.wav")
    if not background.is_file():
        raise SystemExit(f"Missing background stem: {background}")
    if not vocals.is_file():
        raise SystemExit(f"Missing vocals stem: {vocals}")

    total_duration = wav_duration(
        workdir / "source.wav" if (workdir / "source.wav").is_file() else vocals
    )

    if args.skip_tts:
        dubbed = workdir / "dubbed_audio.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(background),
                "-i",
                str(vocals),
                "-filter_complex",
                "[0:a]volume=0.35[bg];[1:a]volume=1.0[v];[bg][v]amix=inputs=2:normalize=0[out]",
                "-map",
                "[out]",
                "-t",
                f"{total_duration:.3f}",
                "-acodec",
                "pcm_s16le",
                str(dubbed),
            ],
            check=True,
            capture_output=True,
        )
    else:
        f5_device = None if args.device == "auto" else args.device
        if args.reuse_tts:
            print("Reusing existing F5-TTS clips.", file=sys.stderr)
        else:
            print(
                f"F5-TTS speed={args.tts_speed} fit_duration={args.tts_fit_duration}",
                file=sys.stderr,
            )
            synthesize_segments_f5(
                segments,
                vocals,
                workdir,
                speed=args.tts_speed,
                fit_duration=args.tts_fit_duration,
                fit_speed_min=args.fit_speed_min,
                fit_speed_max=args.fit_speed_max,
                model=args.f5_model,
                device=f5_device,
                selected_indices=(
                    {int(value) for value in args.tts_segments.split(",")}
                    if args.tts_segments
                    else None
                ),
                merge_pauses=True,
                max_pause=args.max_pause,
            )
        payload_out = {
            **payload,
            "segments": segments,
            "tts_engine": "f5-tts",
            "tts_speed": args.tts_speed,
        }
        out_json.write_text(json.dumps(payload_out, ensure_ascii=False, indent=2), encoding="utf-8")
        dubbed = build_dubbed_track(segments, background, total_duration, workdir)

    video = find_source_video(workdir, payload)
    preview = workdir / "preview.mp4"
    mux_preview(video, dubbed, srt_path, preview, total_duration)

    print(f"\nWatchable preview ready:\n  {preview}", file=sys.stderr)
    print(str(preview))


if __name__ == "__main__":
    main()
