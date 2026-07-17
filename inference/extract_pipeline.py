#!/usr/bin/env python3
"""Phase 1 & 2: extract audio, separate stems, diarize, and transcribe Hebrew speech.

Outputs a JSON timeline of:
  [{speaker_id, start, end, text}, ...]

Also writes vocals.wav / background.wav under the run output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Fix for Pyannote MPS fallback on Apple Silicon
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import mlx_whisper
    from pyannote.audio import Pipeline
except ImportError as e:
    print(f"Error importing dependencies: {e}", file=sys.stderr)
    print("Install with: uv sync or uv pip install -r requirements/extract.txt", file=sys.stderr)
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "mlx-community/ivrit-ai-whisper-large-v3-mlx"
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            "ffmpeg not found on PATH. Install it (e.g. `brew install ffmpeg`) and retry."
        )


def extract_audio(input_path: Path, output_wav: Path, sample_rate: int = 44100) -> Path:
    """Strip audio from video (or re-encode audio) to a mono/stereo WAV for Demucs."""
    require_ffmpeg()
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        str(output_wav),
    ]
    print(f"Extracting audio → {output_wav}", file=sys.stderr)
    subprocess.run(cmd, check=True, capture_output=True)
    if not output_wav.is_file():
        raise FileNotFoundError(f"ffmpeg did not produce {output_wav}")
    return output_wav


def separate_vocals(input_audio: Path, demucs_dir: Path) -> tuple[Path, Path]:
    """Run HTDemucs two-stem separation; return (vocals, background/no_vocals)."""
    demucs_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running Demucs (htdemucs, two-stems) on {input_audio}...", file=sys.stderr)
    cmd = [
        sys.executable,
        "-m",
        "demucs",
        "--two-stems=vocals",
        "-n",
        "htdemucs",
        "-o",
        str(demucs_dir),
        str(input_audio),
    ]
    subprocess.run(cmd, check=True)

    stem_dir = demucs_dir / "htdemucs" / input_audio.stem
    vocals_path = stem_dir / "vocals.wav"
    background_path = stem_dir / "no_vocals.wav"
    if not vocals_path.is_file():
        raise FileNotFoundError(f"Demucs failed to produce {vocals_path}")
    if not background_path.is_file():
        raise FileNotFoundError(f"Demucs failed to produce {background_path}")
    return vocals_path, background_path


def diarize_audio(vocals_path: Path, model_id: str = DEFAULT_DIARIZATION_MODEL) -> list[dict]:
    print(f"Running Pyannote diarization ({model_id}) on {vocals_path}...", file=sys.stderr)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print(
            "Warning: HF_TOKEN not set. Pyannote gated models need a token + accepted terms.",
            file=sys.stderr,
        )

    try:
        pipeline = Pipeline.from_pretrained(model_id, token=token)
    except TypeError:
        pipeline = Pipeline.from_pretrained(model_id, use_auth_token=token)

    if pipeline is None:
        raise SystemExit(
            "Failed to load pyannote pipeline. Set HF_TOKEN in .env and accept model "
            f"conditions at https://huggingface.co/{model_id}"
        )

    import torch

    if torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))
    elif torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    output = pipeline(str(vocals_path))
    # pyannote.audio 4.x returns DiarizeOutput; 3.x returned Annotation directly.
    annotation = getattr(output, "speaker_diarization", output)

    segments: list[dict] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append(
            {
                "speaker_id": speaker,
                "start": round(float(turn.start), 3),
                "end": round(float(turn.end), 3),
            }
        )
    return segments


def transcribe_and_merge(
    vocals_path: Path,
    segments: list[dict],
    model_name: str,
    language: str = "he",
) -> list[dict]:
    
    print(f"Transcribing {vocals_path} with mlx_whisper (model={model_name})...", file=sys.stderr)
    
    result = mlx_whisper.transcribe(
        str(vocals_path),
        path_or_hf_repo=model_name,
        language=language,
        word_timestamps=True
    )

    words = []
    for ws in result.get("segments", []):
        if "words" in ws:
            words.extend(ws["words"])

    print(
        f"Mapped {len(words)} words → {len(segments)} speaker turns...",
        file=sys.stderr,
    )

    results: list[dict] = []
    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        
        seg_words = [
            w["word"] for w in words
            if "end" in w and "start" in w and w["end"] >= seg_start and w["start"] <= seg_end
        ]
        # mlx-whisper words usually include leading spaces; join then normalize.
        text = "".join(seg_words).strip()
        if text and " " not in text and len(seg_words) > 1:
            text = " ".join(w.strip() for w in seg_words if w.strip())
        
        if not text:
            continue
        results.append(
            {
                "speaker_id": seg["speaker_id"],
                "start": seg_start,
                "end": seg_end,
                "duration": round(seg_end - seg_start, 3),
                "text": text,
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 & 2: Demucs + Pyannote + MLX Whisper → speaker-labeled Hebrew JSON."
    )
    parser.add_argument("video", type=Path, help="Input video or audio file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: outputs/<run_id>/segments.json).",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Working directory for stems/audio (default: outputs/<run_id>/).",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HuggingFace model ID or local path for MLX Whisper.")
    parser.add_argument(
        "--diarization-model",
        default=DEFAULT_DIARIZATION_MODEL,
        help=f"Pyannote model id (default: {DEFAULT_DIARIZATION_MODEL}).",
    )
    parser.add_argument("--language", "-l", default="he")
    parser.add_argument(
        "--skip-demucs",
        action="store_true",
        help="Skip source separation; diarize/transcribe the extracted mix directly.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.video.is_file():
        raise SystemExit(f"File not found: {args.video}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + args.video.stem
    workdir = args.workdir or (REPO_ROOT / "outputs" / run_id)
    workdir.mkdir(parents=True, exist_ok=True)

    source_wav = workdir / "source.wav"
    extract_audio(args.video, source_wav)

    if args.skip_demucs:
        vocals_path = source_wav
        background_path = None
    else:
        vocals_path, background_path = separate_vocals(source_wav, workdir / "demucs")

    # Copy stems to a stable location for later pipeline stages
    vocals_out = workdir / "vocals.wav"
    if vocals_path.resolve() != vocals_out.resolve():
        shutil.copy2(vocals_path, vocals_out)
        vocals_path = vocals_out
    if background_path is not None:
        background_out = workdir / "background.wav"
        shutil.copy2(background_path, background_out)
        background_path = background_out

    segments = diarize_audio(vocals_path, model_id=args.diarization_model)
    
    results = transcribe_and_merge(
        vocals_path,
        segments,
        args.model,
        language=args.language,
    )

    payload = {
        "source": str(args.video.resolve()),
        "workdir": str(workdir.resolve()),
        "vocals": str(vocals_path.resolve()),
        "background": str(background_path.resolve()) if background_path else None,
        "language": args.language,
        "whisper_model": args.model,
        "diarization_model": args.diarization_model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "segments": results,
    }

    output_path = args.output or (workdir / "segments.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(
        f"Wrote {len(results)} segments → {output_path}",
        file=sys.stderr,
    )
    print(str(output_path))


if __name__ == "__main__":
    main()
