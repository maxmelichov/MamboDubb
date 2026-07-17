#!/usr/bin/env python3
"""Translate text with the local TranslateGemma 4B checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

print(
    "WARNING: TranslateGemma currently returns empty on this stack due to a pad-token bug.\n"
    "Please use `inference/translate_pipeline.py` (which uses mlx-lm) instead.",
    file=sys.stderr,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "translategemma-4b-it"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate text with TranslateGemma 4B.")
    parser.add_argument("text", nargs="?", help="Text to translate; reads stdin when omitted.")
    parser.add_argument("--source", "-s", required=True, help="Source language code, e.g. en.")
    parser.add_argument("--target", "-t", required=True, help="Target language code, e.g. es.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Local model directory (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Inference device (default: auto).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    text = args.text if args.text is not None else sys.stdin.read()
    if not text.strip():
        raise SystemExit("No input text was provided.")
    if not args.model.is_dir():
        raise SystemExit(
            f"Model directory not found: {args.model}\n"
            "Download it with: hf download google/translategemma-4b-it "
            f'--local-dir "{DEFAULT_MODEL_PATH}"'
        )

    device = resolve_device(args.device)
    dtype = resolve_dtype(device)
    print(f"Loading {args.model} on {device} ({dtype})...", file=sys.stderr)

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": args.source,
                    "target_lang_code": args.target,
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
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
        )

    translated_tokens = generated[0, input_length:]
    print(processor.decode(translated_tokens, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
