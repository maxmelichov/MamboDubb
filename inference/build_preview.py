#!/usr/bin/env python3
"""Build a watchable preview from Phase 1–2 outputs.

Takes a run directory (e.g. outputs/kan11_60s) with:
  segments.json, vocals.wav, background.wav, and the source video

Produces:
  translated_segments.json
  preview_en.srt
  dubbed_audio.wav
  preview.mp4   ← video + EN TTS over ducked BGM + soft EN subs

TTS (default: Qwen3-TTS 1.7B-Base zero-shot — https://arxiv.org/abs/2601.15621):
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


def preserve_latin_tokens(source: str, translated: str) -> str:
    """Re-inject Latin tokens from the source if the translator dropped them."""
    import re

    he_re = re.compile(r"[\u0590-\u05FF]")
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9'./-]*", source or "")
    out = (translated or "").strip()
    if not out:
        return out
    # Translator sometimes echoes Hebrew or collapses to bare names — don't "fix" that.
    if he_re.search(out):
        return out
    if tokens and len(out.split()) <= len(tokens):
        return out
    lower = out.lower()
    missing = [t for t in tokens if t.lower() not in lower]
    if not missing:
        return out
    return (out.rstrip(" .,") + " " + " ".join(dict.fromkeys(missing))).strip()


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
    en = processor.decode(generated[0, input_length:], skip_special_tokens=True).strip()
    return preserve_latin_tokens(text, en)


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


def _load_mono(path: Path, sample_rate: int):
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sr == sample_rate:
        return mono
    # Linear resample for rare rate mismatches (TTS / Demucs are usually 44.1k).
    n_out = int(round(len(mono) * sample_rate / sr))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=len(mono), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, mono).astype(np.float32)


def build_dubbed_track(
    segments: list[dict],
    background: Path,
    total_duration: float,
    workdir: Path,
    *,
    bg_gain: float = 0.55,
    speech_gain: float = 1.2,
    speech_target_rms: float = 0.085,
) -> Path:
    """Place TTS clips on a timeline and sum with a *constant* background level.

    No sidechain ducking — pumping the bed under every line made playback feel
    uneven. Keep music at a steady gain; speech is loudness-matched on top.
    """
    import numpy as np
    import soundfile as sf

    # Demucs no_vocals leaves loud HF hiss in vocal holes — clean the bed only.
    from inference.extract_pipeline import write_clean_background

    raw_bg = workdir / "background_raw.wav"
    if not raw_bg.is_file() and background.is_file():
        import shutil

        shutil.copy2(background, raw_bg)
    if raw_bg.is_file():
        write_clean_background(raw_bg, background)

    bg, sr = sf.read(str(background), dtype="float32", always_2d=True)
    n_samples = max(1, int(round(total_duration * sr)))
    if bg.shape[0] < n_samples:
        bg = np.pad(bg, ((0, n_samples - bg.shape[0]), (0, 0)))
    else:
        bg = bg[:n_samples]

    speech = np.zeros(n_samples, dtype=np.float32)
    # Mute Demucs bed under KEEP-original spans (those clips already include
    # the original music from source.wav — layering gappy no_vocals makes waves).
    bg_gate = np.ones(n_samples, dtype=np.float32)
    placed = 0
    # Place in timeline order; later clips overwrite overlaps (avoids KEEP+dub sum).
    ordered = sorted(
        segments,
        key=lambda s: (float(s.get("start") or 0.0), 0 if s.get("keep_original") else 1),
    )
    for seg in ordered:
        fitted = seg.get("tts_fit")
        if not fitted:
            continue
        path = Path(fitted)
        if not path.is_file():
            print(f"  skip missing TTS clip: {path}", file=sys.stderr)
            continue
        clip = _load_mono(path, sr)
        i0 = int(round(float(seg["start"]) * sr))
        if i0 >= n_samples or len(clip) == 0:
            continue
        i1 = min(n_samples, i0 + len(clip))
        take = clip[: i1 - i0].copy()
        keep = bool(seg.get("keep_original") or seg.get("keep_uses_source"))
        if keep:
            # Source mix already has natural speech+bed balance.
            fade = min(int(0.015 * sr), len(take) // 4)
            if fade > 1:
                ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
                take[:fade] *= ramp
                take[-fade:] *= ramp[::-1]
            speech[i0:i1] = take
            bg_gate[i0:i1] = 0.0
        else:
            clip_rms = float(np.sqrt(np.mean(take**2) + 1e-12))
            if clip_rms > 1e-4:
                # Cap boost — loudness-matching hissy TTS clips made white noise
                # scream in the 20–35s range.
                gain = speech_target_rms / clip_rms
                take *= min(gain, 1.25)
            fade = min(int(0.012 * sr), len(take) // 4)
            if fade > 1:
                ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
                take[:fade] *= ramp
                take[-fade:] *= ramp[::-1]
            speech[i0:i1] = take
        placed += 1

    # Soft edges on bg mute so KEEP↔dub transitions don't click.
    fade_bg = max(1, int(0.02 * sr))
    gate = bg_gate.copy()
    # Simple box blur for mute ramps
    kernel = np.ones(fade_bg, dtype=np.float32) / float(fade_bg)
    gate = np.convolve(gate, kernel, mode="same")
    gate = np.clip(gate, 0.0, 1.0)

    mix = bg * (bg_gain * gate)[:, None] + (speech * speech_gain)[:, None]
    # KEEP clips are already full-mix; don't boost them as hard as TTS.
    # (speech_gain applied uniformly; KEEP was written at natural level — scale keep down relative)
    # Re-balance: we applied speech_gain to all; for keep regions reduce toward 1.0
    # by compensating where gate==0 (keep spans).
    keep_mask = 1.0 - gate
    if float(np.max(keep_mask)) > 0.01 and speech_gain != 1.0:
        # Remove excess speech_gain on KEEP: multiply those samples by 1/speech_gain
        # through the speech contribution only — approximate via remix of keep parts.
        mix = bg * (bg_gain * gate)[:, None] + (
            speech * (speech_gain * gate + 1.0 * keep_mask)
        )[:, None]
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.98:
        mix *= 0.98 / peak

    out_wav = workdir / "dubbed_audio.wav"
    print(
        f"Mixing dubbed audio (numpy): {placed} clips, "
        f"bg_gain={bg_gain} (constant, no duck), speech_rms≈{speech_target_rms} → {out_wav}",
        file=sys.stderr,
    )
    sf.write(str(out_wav), mix.astype(np.float32), sr, subtype="PCM_16")
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
        help="TTS backend (default: qwen 1.7B-Base zero-shot clone).",
    )
    p.add_argument(
        "--qwen-model",
        type=Path,
        default=QWEN_DEFAULT_MODEL,
        help="Qwen3-TTS Base checkpoint (default: models/Qwen3-TTS-12Hz-1.7B-Base).",
    )
    p.add_argument(
        "--qwen-icl",
        action="store_true",
        help="ICL clone with Hebrew ref_text (default: x-vector-only from ref audio).",
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
        # Preview remix without TTS: place full vocals as one "clip" at t=0.
        import numpy as np
        import soundfile as sf

        bg, sr = sf.read(str(background), dtype="float32", always_2d=True)
        voc, vsr = sf.read(str(vocals), dtype="float32", always_2d=True)
        n = max(1, int(round(total_duration * sr)))
        if bg.shape[0] < n:
            bg = np.pad(bg, ((0, n - bg.shape[0]), (0, 0)))
        else:
            bg = bg[:n]
        voc_m = voc.mean(axis=1)
        if vsr != sr and len(voc_m):
            n_out = int(round(len(voc_m) * sr / vsr))
            x_old = np.linspace(0.0, 1.0, num=len(voc_m), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
            voc_m = np.interp(x_new, x_old, voc_m).astype(np.float32)
        speech = np.zeros(n, dtype=np.float32)
        speech[: min(n, len(voc_m))] = voc_m[:n]
        mix = bg * 0.65 + speech[:, None]
        peak = float(np.max(np.abs(mix))) if mix.size else 0.0
        if peak > 0.98:
            mix *= 0.98 / peak
        dubbed = workdir / "dubbed_audio.wav"
        sf.write(str(dubbed), mix.astype(np.float32), sr, subtype="PCM_16")
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
                x_vector_only=not args.qwen_icl,
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
                "qwen3-tts-1.7b-base" if args.tts_engine == "qwen" else "f5-tts"
            ),
            "tts_speed": args.tts_speed,
            "qwen_x_vector_only": not bool(args.qwen_icl),
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
