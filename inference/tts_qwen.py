#!/usr/bin/env python3
"""Qwen3-TTS 0.6B-Base zero-shot voice clone for DubbingQwen (Phase 4).

Clones from each phrase's own vocal window (Hebrew ref audio + transcript),
then synthesizes English — same ref strategy as F5, but Qwen Base.

Model: Qwen/Qwen3-TTS-12Hz-0.6B-Base (https://arxiv.org/abs/2601.15621)
  - x_vector_only=False (default): ICL with ref_audio + ref_text
  - x_vector_only=True: speaker embedding only (often stabler cross-lingual)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from inference.tts_f5 import (
    build_phrase_ref,
    concat_wavs,
    extend_end_by_energy,
    extract_wav_slice,
    make_silence,
    phrase_plan,
    place_in_slot,
    wav_duration,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "models" / "Qwen3-TTS-12Hz-0.6B-Base"
HUB_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"


def resolve_qwen_device(requested: str | None) -> str:
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_qwen_dtype(device: str) -> torch.dtype:
    if device.startswith("cuda"):
        return torch.bfloat16
    # float16 on MPS hits NaNs in the code predictor sampler; use float32.
    return torch.float32


def resolve_model_path(prefer: Path) -> str:
    if prefer.is_dir() and any(prefer.iterdir()):
        return str(prefer)
    return HUB_MODEL


class QwenVoiceCloneSynthesizer:
    """0.6B-Base zero-shot clone with optional prompt reuse per speaker."""

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        device: str | None = None,
        x_vector_only: bool = False,
    ) -> None:
        self.device = resolve_qwen_device(device)
        self.dtype = resolve_qwen_dtype(self.device)
        self.path = resolve_model_path(Path(model_path) if model_path else DEFAULT_MODEL)
        self.x_vector_only = bool(x_vector_only)
        self._model = None
        self._prompt_cache: dict[str, object] = {}

    def _load(self):
        from qwen_tts import Qwen3TTSModel

        if self._model is not None:
            return self._model

        print(f"Loading Qwen3-TTS 0.6B-Base (voice clone) from {self.path}", file=sys.stderr)
        kwargs: dict = {
            "dtype": self.dtype,
            "low_cpu_mem_usage": False,
        }
        if self.device.startswith("cuda"):
            kwargs["attn_implementation"] = "sdpa"
            kwargs["device_map"] = self.device
        else:
            kwargs["device_map"] = "cpu"

        model = Qwen3TTSModel.from_pretrained(self.path, **kwargs)
        if not self.device.startswith("cuda"):
            model.model.to(self.device)
            tok = getattr(model.model, "speech_tokenizer", None)
            if tok is not None and hasattr(tok, "to"):
                tok.to(self.device)
            model.device = torch.device(self.device)
        self._model = model
        return model

    def synthesize(
        self,
        text: str,
        *,
        ref_audio: Path,
        ref_text: str,
        language: str = "English",
        out_wav: Path,
        prompt_key: str | None = None,
    ) -> Path:
        model = self._load()
        out_wav.parent.mkdir(parents=True, exist_ok=True)

        voice_clone_prompt = None
        if prompt_key is not None and prompt_key in self._prompt_cache:
            voice_clone_prompt = self._prompt_cache[prompt_key]
        elif prompt_key is not None:
            voice_clone_prompt = model.create_voice_clone_prompt(
                ref_audio=str(ref_audio),
                ref_text=ref_text if not self.x_vector_only else None,
                x_vector_only_mode=self.x_vector_only,
            )
            self._prompt_cache[prompt_key] = voice_clone_prompt

        if voice_clone_prompt is not None:
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=voice_clone_prompt,
            )
        else:
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=str(ref_audio),
                ref_text=ref_text if not self.x_vector_only else None,
                x_vector_only_mode=self.x_vector_only,
            )

        audio = np.asarray(wavs[0], dtype=np.float32)
        sf.write(str(out_wav), audio, sr)
        return out_wav


def synthesize_segments_qwen(
    segments: list[dict],
    vocals: Path,
    workdir: Path,
    *,
    language: str = "English",
    model_path: str | Path | None = None,
    device: str | None = None,
    x_vector_only: bool = False,
    reuse_speaker_prompt: bool = False,
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
    syn = QwenVoiceCloneSynthesizer(
        model_path=model_path or DEFAULT_MODEL,
        device=device,
        x_vector_only=x_vector_only,
    )
    mode = "x_vector_only" if x_vector_only else "icl"
    print(
        f"Qwen3-TTS 0.6B-Base zero-shot clone ({mode}) language={language}",
        file=sys.stderr,
    )

    for i, seg in enumerate(segments):
        start = float(seg["start"])
        end = float(seg["end"])
        target = max(end - start, 0.4)
        lang = seg.get("language") or "he"
        keep_original = bool(seg.get("keep_original", lang != "he"))

        next_start = media_duration
        if i + 1 < len(segments):
            next_start = float(segments[i + 1]["start"])

        if keep_original:
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
                if not (seg["phrases"][-1].get("text_en") or "").strip():
                    seg["phrases"][-1]["text_en"] = seg["phrases"][-1].get("text") or ""
            target = max(end - start, 0.4)
            raw = tts_dir / f"seg_{i:02d}_orig.wav"
            fitted = tts_dir / f"seg_{i:02d}_fit.wav"
            extract_wav_slice(vocals, start, end, raw, sample_rate=44100)
            place_in_slot(raw, fitted, target, sample_rate=44100)
            seg.pop("tts_text", None)
            seg["tts_raw"] = str(raw)
            seg["tts_fit"] = str(fitted)
            seg["tts_speed_used"] = 1.0
            seg["keep_original"] = True
            seg["tts_engine"] = "keep_original"
            continue

        if selected_indices is not None and i not in selected_indices:
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

        print(
            f"  Qwen clone [{seg['speaker_id']}] "
            f"slot {start:.1f}-{end:.1f}s {len(plan)} phrase(s)",
            file=sys.stderr,
        )

        first_ref_meta: dict | None = None
        for j, phrase in enumerate(plan):
            text = phrase["text"]
            p_start = float(phrase["start"])
            p_end = float(phrase["end"])
            hard_cap = (
                float(plan[j + 1]["start"]) - 0.04
                if j + 1 < len(plan)
                else min(float(seg["end"]) + 0.35, next_start - 0.05)
            )
            p_end = min(p_end, hard_cap)
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
            ref["ref_text"] = he_text if not (ref.get("ref_text") or "").strip() else ref["ref_text"]
            if first_ref_meta is None:
                first_ref_meta = ref

            print(
                f"    phrase {j} [{p_start:.1f}-{p_end:.1f}s] "
                f"ref={ref['start']:.1f}-{ref['end']:.1f}s: {text[:55]}…",
                file=sys.stderr,
            )
            chunk = tts_dir / f"seg_{i:02d}_p{j:02d}_qwen.wav"
            prompt_key = None
            if reuse_speaker_prompt:
                prompt_key = str(seg.get("speaker_id") or f"seg_{i}")
            syn.synthesize(
                text,
                ref_audio=Path(ref["path"]),
                ref_text=str(ref["ref_text"]),
                language=language,
                out_wav=chunk,
                prompt_key=prompt_key,
            )

            chunk_441 = tts_dir / f"seg_{i:02d}_p{j:02d}_clean.wav"
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
            actual = wav_duration(chunk_441)
            print(f"      qwen {actual:.2f}s", file=sys.stderr)
            phrase["tts_fit"] = str(chunk_441)
            phrase["ref_audio"] = ref["path"]

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
            f"({len(plan)} phrases)",
            file=sys.stderr,
        )
        if spoken > target + 0.05:
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
            print(f"    soft-trim concat to {target:.2f}s", file=sys.stderr)
        else:
            place_in_slot(raw, fitted, target, sample_rate=44100)

        seg["end"] = round(start + target, 3)
        seg["duration"] = round(target, 3)
        seg.pop("tts_text", None)
        seg["tts_raw"] = str(raw)
        seg["tts_fit"] = str(fitted)
        seg["tts_speed_used"] = 1.0
        seg["keep_original"] = False
        seg["tts_engine"] = "qwen3-tts-0.6b-base"
        seg["qwen_clone_mode"] = mode
        if first_ref_meta:
            seg["ref_audio"] = first_ref_meta["path"]
            seg["ref_start"] = first_ref_meta["start"]
            seg["ref_end"] = first_ref_meta["end"]
        if seg.get("phrases"):
            for src_p, out_p in zip(seg["phrases"], plan):
                src_p["text_en"] = out_p["text"]
                if out_p.get("tts_fit"):
                    src_p["tts_fit"] = out_p["tts_fit"]

    return segments


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen3-TTS 0.6B-Base zero-shot clone.")
    p.add_argument("workdir", type=Path, help="Run dir with translated_segments.json + vocals.")
    p.add_argument("--language", default="English")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument(
        "--x-vector-only",
        action="store_true",
        help="Clone from speaker embedding only (no ICL ref_text).",
    )
    p.add_argument(
        "--no-reuse-speaker-prompt",
        action="store_true",
        help="Rebuild clone prompt per phrase instead of once per speaker.",
    )
    p.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    p.add_argument("--max-pause", type=float, default=1.0)
    p.add_argument("--tts-segments", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    workdir = args.workdir
    translated = workdir / "translated_segments.json"
    segments_path = translated if translated.is_file() else workdir / "segments.json"
    if not segments_path.is_file():
        raise SystemExit(f"Missing {segments_path}")
    payload = json.loads(segments_path.read_text(encoding="utf-8"))
    segments = payload.get("segments") or []
    vocals = Path(payload.get("vocals") or workdir / "vocals.wav")
    if not vocals.is_file():
        raise SystemExit(f"Missing vocals: {vocals}")

    synthesize_segments_qwen(
        segments,
        vocals,
        workdir,
        language=args.language,
        model_path=args.model,
        device=None if args.device == "auto" else args.device,
        x_vector_only=args.x_vector_only,
        reuse_speaker_prompt=not args.no_reuse_speaker_prompt,
        selected_indices=(
            {int(v) for v in args.tts_segments.split(",")} if args.tts_segments else None
        ),
        max_pause=args.max_pause,
    )
    payload["segments"] = segments
    payload["tts_engine"] = "qwen3-tts-0.6b-base"
    out = workdir / "translated_segments.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
