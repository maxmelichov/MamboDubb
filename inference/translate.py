#!/usr/bin/env python3
"""Translate text with the local TranslateGemma 4B checkpoint."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

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
        # float16 on MPS yields pad-only generations with this checkpoint;
        # bfloat16 (via pipeline / generate) produces real translations.
        return torch.bfloat16
    return torch.bfloat16


def load_translategemma(model_path: Path | str, device: torch.device | None = None):
    """Load TranslateGemma processor + model. Returns (processor, model, device)."""
    path = Path(model_path)
    if not path.is_dir():
        raise SystemExit(
            f"Model directory not found: {path}\n"
            "Download it with: hf download google/translategemma-4b-it "
            f'--local-dir "{DEFAULT_MODEL_PATH}"'
        )
    device = device or resolve_device("auto")
    dtype = resolve_dtype(device)
    print(f"Loading TranslateGemma {path} on {device} ({dtype})...", file=sys.stderr)
    processor = AutoProcessor.from_pretrained(path)
    model = AutoModelForImageTextToText.from_pretrained(
        path,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return processor, model, device


def translate_gemma(
    processor,
    model,
    text: str,
    *,
    source: str = "he",
    target: str = "en",
    device: torch.device | None = None,
    max_new_tokens: int = 220,
) -> str:
    """Translate one string with TranslateGemma (source/target language codes)."""
    he = (text or "").strip()
    if not he:
        return ""
    if device is None:
        device = next(model.parameters()).device
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": source,
                    "target_lang_code": target,
                    "text": he,
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
    )
    # Move tensors to device WITHOUT casting dtype — casting input_ids to float
    # made generate emit pad-only output on this stack.
    # Drop token_type_ids — not needed for text-only TranslateGemma and can
    # confuse generation on some transformers builds.
    clean = {}
    for k, v in inputs.items():
        if k == "token_type_ids":
            continue
        clean[k] = v.to(device) if hasattr(v, "to") else v
    inputs = clean
    input_length = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )

    translated_tokens = generated[0, input_length:]
    # Guard against pad-only generations (seen on MPS float16).
    if translated_tokens.numel() == 0 or int((translated_tokens == 0).sum()) == int(
        translated_tokens.numel()
    ):
        return ""
    return processor.decode(translated_tokens, skip_special_tokens=True).strip()


_MARKER_RE = re.compile(r"\[\[C(\d+)\]\]")


def build_marked_he_window(
    chunks: list[str],
    *,
    focus_idx: int,
) -> str:
    """Build a marker-delimited HE window for contextual TranslateGemma.

    Markers look like ``[[C0]] … [[C1]] …`` so the model can return aligned EN.
    ``focus_idx`` is the chunk that must be translated; neighbors are context.
    """
    parts: list[str] = []
    for i, text in enumerate(chunks):
        t = (text or "").strip()
        if not t:
            continue
        parts.append(f"[[C{i}]] {t}")
    return "\n".join(parts)


def parse_marked_en(en: str, n_chunks: int) -> dict[int, str] | None:
    """Parse ``[[C0]] english… [[C1]] …`` back into per-chunk EN.

    Returns None when markers are missing / incomplete so callers can fall back
    to direct single-chunk translation.
    """
    text = (en or "").strip()
    if not text or "[[C" not in text:
        return None
    matches = list(_MARKER_RE.finditer(text))
    if not matches:
        return None
    out: dict[int, str] = {}
    for i, m in enumerate(matches):
        idx = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip().strip('"').strip("'")
        if chunk:
            out[idx] = chunk
    # Require the focus-range markers we asked for (at least one).
    if not out:
        return None
    if any(i < 0 or i >= n_chunks for i in out):
        return None
    return out


def translate_gemma_marked_chunks(
    processor,
    model,
    chunks: list[str],
    *,
    focus_idx: int,
    source: str = "he",
    target: str = "en",
    device: torch.device | None = None,
    max_new_tokens: int = 320,
) -> dict[int, str] | None:
    """Translate a neighbor window with markers; return per-chunk EN or None.

    Instructs TranslateGemma (via plain text — chat template only accepts the
    source string) to preserve ``[[Ci]]`` markers. On failure returns None.
    """
    nonempty = [(i, (c or "").strip()) for i, c in enumerate(chunks) if (c or "").strip()]
    if not nonempty:
        return None
    marked = build_marked_he_window(chunks, focus_idx=focus_idx)
    # TranslateGemma is a dedicated MT model: wrap markers in the source text
    # and ask it to keep them by putting them inline with the HE.
    # Extra instruction is not part of the official template, so we rely on
    # the model copying the bracket markers through; fallback handles failure.
    prompt = (
        "Preserve every [[Ci]] marker and translate only the Hebrew after each "
        "marker into English. Keep the same markers in the same order.\n"
        f"{marked}"
    )
    raw = translate_gemma(
        processor,
        model,
        prompt,
        source=source,
        target=target,
        device=device,
        max_new_tokens=max_new_tokens,
    )
    parsed = parse_marked_en(raw, len(chunks))
    if parsed is not None and focus_idx in parsed:
        return parsed
    # Retry without the instruction preface (some MT models echo it poorly).
    raw2 = translate_gemma(
        processor,
        model,
        marked,
        source=source,
        target=target,
        device=device,
        max_new_tokens=max_new_tokens,
    )
    parsed2 = parse_marked_en(raw2, len(chunks))
    if parsed2 is not None and focus_idx in parsed2:
        return parsed2
    return None


def main() -> None:
    args = parse_args()
    text = args.text if args.text is not None else sys.stdin.read()
    if not text.strip():
        raise SystemExit("No input text was provided.")

    device = resolve_device(args.device)
    processor, model, device = load_translategemma(args.model, device=device)
    out = translate_gemma(
        processor,
        model,
        text,
        source=args.source,
        target=args.target,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )
    print(out)


if __name__ == "__main__":
    main()
