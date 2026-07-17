#!/usr/bin/env python3
"""Phase 3: Context-Aware Translation using mlx-lm."""

import argparse
import json
import sys
from pathlib import Path

try:
    from mlx_lm import load, generate
except ImportError:
    print("Error: mlx-lm is not installed.", file=sys.stderr)
    print("Install with: uv pip install mlx-lm", file=sys.stderr)
    sys.exit(1)

DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate Hebrew segments to English using duration constraints.")
    parser.add_argument("input", type=Path, help="Path to input segments.json")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output JSON path (default: input_translated.json)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"HuggingFace model ID for mlx-lm (default: {DEFAULT_MODEL})")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not args.input.is_file():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
        
    output_path = args.output
    if output_path is None:
        output_path = args.input.parent / f"{args.input.stem}_translated.json"
        
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        print("No segments found in input file.", file=sys.stderr)
        sys.exit(0)
        
    print(f"Loading model {args.model} via mlx-lm...", file=sys.stderr)
    model, tokenizer = load(args.model)
    
    translated_segments = []
    
    print(f"Translating {len(segments)} segments...", file=sys.stderr)
    
    for idx, seg in enumerate(segments):
        he_text = seg.get("text", "").strip()
        duration = seg.get("duration", 0.0)
        seg_lang = seg.get("language", "he")
        
        if not he_text:
            translated_segments.append(seg)
            continue
            
        if seg_lang != "he":
            print(f"[{idx+1:02d}/{len(segments)}] {duration:.1f}s | Skipping translation for non-Hebrew ({seg_lang}) segment: {he_text}")
            new_seg = dict(seg)
            # Copy original text into text_en so it isn't completely empty, 
            # though TTS logic will handle non-hebrew segments using original audio.
            new_seg["text_en"] = he_text
            translated_segments.append(new_seg)
            continue
            
        # Rough target: ~3.5 syllables per second for natural English speech rate
        target_syllables = max(1, int(duration * 3.5))
        
        prompt = (
            f"You are an expert dubbing translator. Translate the following Hebrew text into English.\n"
            f"The original text takes {duration:.2f} seconds to speak.\n"
            f"Your English translation must be roughly {target_syllables} syllables long to match the timing.\n"
            f"Provide ONLY the English translation, with no quotes, notes, or explanations.\n\n"
            f"Hebrew: {he_text}\n"
            f"English:"
        )
        
        messages = [
            {"role": "system", "content": "You are a professional dubbing translator. Always output only the raw translation text."},
            {"role": "user", "content": prompt}
        ]
        
        if hasattr(tokenizer, "apply_chat_template"):
            text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text_prompt = prompt
            
        response = generate(model, tokenizer, prompt=text_prompt, max_tokens=150, verbose=False)
        en_text = response.strip().strip('"').strip("'")
        
        print(f"[{idx+1:02d}/{len(segments)}] {duration:.1f}s | HE: {he_text} -> EN: {en_text}")
        
        new_seg = dict(seg)
        new_seg["text_en"] = en_text
        translated_segments.append(new_seg)
        
    data["segments"] = translated_segments
    data["translation_model"] = args.model
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print(f"\nSaved translated segments to {output_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
