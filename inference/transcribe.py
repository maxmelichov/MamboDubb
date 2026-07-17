#!/usr/bin/env python3
"""Transcribe audio with the local mlx-whisper large-v2-tuned model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import mlx_whisper
except ImportError:
    print("Error: mlx_whisper is not installed.", file=sys.stderr)
    print("Install with: uv pip install mlx-whisper", file=sys.stderr)
    sys.exit(1)


DEFAULT_MODEL = "mlx-community/ivrit-ai-whisper-large-v3-mlx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe audio with mlx-whisper (default: mlx-community/ivrit-ai-whisper-large-v3-mlx)."
    )
    parser.add_argument("audio", type=Path, help="Path to an audio or video file.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"HuggingFace model ID or local path (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--language",
        "-l",
        default="he",
        help="Language code for transcription (default: he).",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Print segment start/end timestamps with the text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.audio.is_file():
        raise SystemExit(f"Audio file not found: {args.audio}")

    print(f"Loading mlx-whisper model '{args.model}'...", file=sys.stderr)

    result = mlx_whisper.transcribe(
        str(args.audio),
        path_or_hf_repo=args.model,
        language=args.language,
        word_timestamps=False, # We only need segment timestamps unless requested otherwise, but mlx_whisper.transcribe returns segments by default
    )
    
    segments = result.get("segments", [])

    print(f"Detected language={result.get('language', 'unknown')}", file=sys.stderr)

    texts: list[str] = []
    for segment in segments:
        text = segment.get("text", "").strip()
        texts.append(text)
        if args.timestamps:
            start = segment.get("start", 0.0)
            end = segment.get("end", 0.0)
            print(f"[{start:7.2f} -> {end:7.2f}] {text}")
        else:
            print(text)

    if not texts:
        print("(no speech detected)", file=sys.stderr)


if __name__ == "__main__":
    main()
