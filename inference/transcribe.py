#!/usr/bin/env python3
"""Transcribe audio with ivrit-ai Whisper large-v3-turbo (CTranslate2 / faster-whisper).

Default checkpoint: models/whisper-large-v3-turbo-ct2
Hub: https://huggingface.co/ivrit-ai/whisper-large-v3-turbo-ct2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from faster_whisper import WhisperModel

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "whisper-large-v3-turbo-ct2"
DEFAULT_HUB_ID = "ivrit-ai/whisper-large-v3-turbo-ct2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe with ivrit-ai/whisper-large-v3-turbo-ct2 (faster-whisper)."
    )
    parser.add_argument("audio", type=Path, help="Path to an audio or video file.")
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help=f"Local CT2 dir or Hub id (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--language",
        "-l",
        default="he",
        help="Language code for transcription (default: he).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Inference device (default: auto).",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        help="CTranslate2 compute type, e.g. float16, int8, auto (default: auto).",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for decoding (default: 5).",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Print segment start/end timestamps with the text.",
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Also print word-level timestamps.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import ctranslate2

        if "cuda" in ctranslate2.get_supported_compute_types("cuda"):
            return "cuda"
    except Exception:
        pass
    return "cpu"


def resolve_model_path(model: str) -> str:
    """Prefer local dir when it exists; otherwise pass Hub id through."""
    path = Path(model)
    if path.is_dir():
        return str(path)
    # Default local path missing → fall back to Hub id
    if model == str(DEFAULT_MODEL_PATH) and not DEFAULT_MODEL_PATH.is_dir():
        return DEFAULT_HUB_ID
    return model


def main() -> None:
    args = parse_args()
    if not args.audio.is_file():
        raise SystemExit(f"Audio file not found: {args.audio}")

    model_path = resolve_model_path(args.model)
    if Path(model_path).is_dir() is False and model_path == str(DEFAULT_MODEL_PATH):
        raise SystemExit(
            f"Model directory not found: {DEFAULT_MODEL_PATH}\n"
            "Download with:\n"
            f"  uv run hf download {DEFAULT_HUB_ID} "
            f'--local-dir "{DEFAULT_MODEL_PATH}"'
        )

    device = resolve_device(args.device)
    print(f"Loading {model_path} on {device}...", file=sys.stderr)

    model = WhisperModel(
        model_path,
        device=device,
        compute_type=args.compute_type,
    )
    segments, info = model.transcribe(
        str(args.audio),
        language=args.language,
        beam_size=args.beam_size,
        word_timestamps=args.word_timestamps or args.timestamps,
    )

    print(
        f"Detected language={info.language} probability={info.language_probability:.2f}",
        file=sys.stderr,
    )

    texts: list[str] = []
    for segment in segments:
        text = segment.text.strip()
        texts.append(text)
        if args.timestamps:
            print(f"[{segment.start:7.2f} -> {segment.end:7.2f}] {text}")
            if args.word_timestamps and segment.words:
                for w in segment.words:
                    print(f"    [{w.start:7.2f} -> {w.end:7.2f}] {w.word}")
        else:
            print(text)

    if not texts:
        print("(no speech detected)", file=sys.stderr)


if __name__ == "__main__":
    main()
