#!/usr/bin/env python3
"""Build a watchable preview from Phase 1–2 outputs.

Takes a run directory (e.g. outputs/kan11_60s) with:
  segments.json, vocals.wav, background.wav, and the source video

Produces:
  translated_segments.json
  preview_en.srt
  dubbed_audio.wav
  preview.mp4   ← video + EN TTS over ducked BGM + soft EN subs

TTS (default: Qwen3-TTS 0.6B-Base zero-shot — https://arxiv.org/abs/2601.15621):
  --tts-engine qwen        clone from each phrase's vocal ref → English
  --tts-engine f5          F5-TTS zero-shot (legacy)

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
from inference.tts_qwen import DEFAULT_MODEL as QWEN_DEFAULT_MODEL, synthesize_segments_qwen

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
        lang = seg.get("language") or "he"
        keep = bool(seg.get("keep_original", lang != "he"))
        phrases = seg.get("phrases") or []
        if phrases:
            for p in phrases:
                text = (p.get(text_key) or p.get("text") or "").strip()
                if not text:
                    continue
                start = float(p.get("start", seg["start"]))
                end = max(float(p.get("end", seg["end"])), start + 0.4)
                speaker = seg.get("speaker_id", "")
                tag = f"{speaker}/{lang}" if keep else speaker
                lines.append(str(idx))
                lines.append(f"{srt_timestamp(start)} --> {srt_timestamp(end)}")
                lines.append(f"[{tag}] {text}" if tag else text)
                lines.append("")
                idx += 1
            continue

        text = (seg.get(text_key) or seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg["start"])
        end = max(float(seg["end"]), start + 0.4)
        speaker = seg.get("speaker_id", "")
        tag = f"{speaker}/{lang}" if keep else speaker
        lines.append(str(idx))
        lines.append(f"{srt_timestamp(start)} --> {srt_timestamp(end)}")
        lines.append(f"[{tag}] {text}" if tag else text)
        lines.append("")
        idx += 1
    path.write_text("\n".join(lines), encoding="utf-8")


def ensure_translations(segments: list[dict], processor, model, device, dtype) -> None:
    """Translate Hebrew only; copy original text for keep_original / non-he segments."""
    for i, seg in enumerate(segments):
        lang = seg.get("language") or "he"
        keep = bool(seg.get("keep_original", lang != "he"))
        phrases = seg.get("phrases") or []

        if keep:
            if phrases:
                for p in phrases:
                    p["text_en"] = (p.get("text_en") or p.get("text") or "").strip()
                seg["text_en"] = " ".join(
                    (p.get("text_en") or "").strip() for p in phrases if p.get("text_en")
                )
            else:
                seg["text_en"] = (seg.get("text_en") or seg.get("text") or "").strip()
            print(
                f"Keep original [{i+1}/{len(segments)}] {lang}: {(seg.get('text') or '')[:50]}…",
                file=sys.stderr,
            )
            continue

        if phrases:
            for j, p in enumerate(phrases):
                he = (p.get("text") or "").strip()
                if not he:
                    continue
                if (p.get("text_en") or "").strip():
                    continue
                if processor is None:
                    raise SystemExit("Translator required for Hebrew phrases without text_en")
                print(
                    f"Translating [{i+1}/{len(segments)} p{j}] {he[:50]}…",
                    file=sys.stderr,
                )
                p["text_en"] = translate_text(processor, model, device, dtype, he)
                print(f"  → {p['text_en']}", file=sys.stderr)
            seg["text_en"] = " ".join(
                (p.get("text_en") or "").strip() for p in phrases if p.get("text_en")
            )
            continue

        he = (seg.get("text") or "").strip()
        if not he:
            continue
        if (seg.get("text_en") or "").strip():
            continue
        if processor is None:
            raise SystemExit("Translator required for Hebrew segments without text_en")
        print(f"Translating [{i+1}/{len(segments)}] {he[:50]}…", file=sys.stderr)
        seg["text_en"] = translate_text(processor, model, device, dtype, he)
        print(f"  → {seg['text_en']}", file=sys.stderr)


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
        "--tts-engine",
        choices=("qwen", "f5"),
        default="qwen",
        help="TTS backend (default: qwen 0.6B-Base zero-shot clone).",
    )
    p.add_argument(
        "--qwen-model",
        type=Path,
        default=QWEN_DEFAULT_MODEL,
        help="Qwen3-TTS Base checkpoint (default: models/Qwen3-TTS-12Hz-0.6B-Base).",
    )
    p.add_argument(
        "--qwen-x-vector-only",
        action="store_true",
        help="Clone via speaker embedding only (no ICL ref_text).",
    )
    p.add_argument(
        "--qwen-reuse-speaker-prompt",
        action="store_true",
        help="Reuse one clone prompt per diarization speaker (faster; less phrase-local).",
    )
    p.add_argument(
        "--tts-speed",
        type=float,
        default=1.0,
        help="F5-TTS only: speed 0.3–2.0 (default 1.0).",
    )
    p.add_argument(
        "--tts-fit-duration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="F5 only: nudge speed toward slot length (default: on).",
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

    def _has_en(seg: dict) -> bool:
        if seg.get("keep_original") or seg.get("language", "he") != "he":
            return bool((seg.get("text") or seg.get("text_en") or "").strip())
        phrases = seg.get("phrases") or []
        if phrases:
            return all((p.get("text_en") or "").strip() for p in phrases if (p.get("text") or "").strip())
        return bool((seg.get("text_en") or "").strip())

    if args.skip_translate and all(_has_en(s) for s in segments):
        print("Reusing existing English / original-language text.", file=sys.stderr)
        # Still normalize keep_original text_en copies
        for seg in segments:
            if seg.get("keep_original") or seg.get("language", "he") != "he":
                seg["keep_original"] = True
                if not (seg.get("text_en") or "").strip():
                    seg["text_en"] = (seg.get("text") or "").strip()
                for p in seg.get("phrases") or []:
                    if not (p.get("text_en") or "").strip():
                        p["text_en"] = (p.get("text") or "").strip()
    else:
        # For keep_original we don't need the translator at all
        needs_mt = [
            s
            for s in segments
            if not (s.get("keep_original") or s.get("language", "he") != "he")
            and not _has_en(s)
        ]
        processor = model = dtype = None
        device = resolve_device(args.device)
        if needs_mt:
            if not args.model.is_dir():
                raise SystemExit(f"Translate model not found: {args.model}")
            processor, model, dtype = load_translator(args.model, device)
        ensure_translations(segments, processor, model, device, dtype)
        if model is not None:
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
        tts_device = None if args.device == "auto" else args.device
        selected = (
            {int(value) for value in args.tts_segments.split(",")}
            if args.tts_segments
            else None
        )
        if args.reuse_tts:
            print(f"Reusing existing TTS clips (engine={args.tts_engine}).", file=sys.stderr)
        elif args.tts_engine == "qwen":
            synthesize_segments_qwen(
                segments,
                vocals,
                workdir,
                model_path=args.qwen_model,
                x_vector_only=args.qwen_x_vector_only,
                reuse_speaker_prompt=args.qwen_reuse_speaker_prompt,
                device=tts_device,
                selected_indices=selected,
                merge_pauses=True,
                max_pause=args.max_pause,
            )
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
                device=tts_device,
                selected_indices=selected,
                merge_pauses=True,
                max_pause=args.max_pause,
            )
        payload_out = {
            **payload,
            "segments": segments,
            "tts_engine": (
                "qwen3-tts-0.6b-base" if args.tts_engine == "qwen" else "f5-tts"
            ),
            "tts_speed": args.tts_speed,
            "qwen_x_vector_only": bool(args.qwen_x_vector_only),
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
