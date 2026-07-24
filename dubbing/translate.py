"""Stage 5 — general-purpose Gemma as a comprehension translator, one segment at a time.

A dedicated MT model (TranslateGemma) translates literally and has no world
knowledge, so it drops or garbles the references that matter — "האחים המוסלמים"
(the Muslim Brotherhood, an organisation) came back as "brotherhood among
Muslims", and "שייח'ה" (the Sheikha) as "a CEO". A general instruction model
*understands* the text: it keeps the named organisations and people, and — asked
to — every clause and detail, rather than paraphrasing them away. That is what a
dub needs: the meaning, complete, in natural English.

The model is general Gemma 3 (12B) run through MLX (Apple-native), 4-bit
quantised, so it fits in unified memory. Each segment is translated on its own,
deterministically (greedy); a segment never sees its neighbours, which keeps the
pieces from bleeding into each other.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "gemma-3-12b-it-4bit"
HUB_ID = "mlx-community/gemma-3-12b-it-4bit"

_LATIN = re.compile(r"[A-Za-z]")
_HEBREW = re.compile(r"[֐-׿]")
_NEGATIONS = {"not", "no", "never", "cannot", "none", "without", "nor", "n't"}

_LANG_NAMES = {"he": "Hebrew", "en": "English", "ar": "Arabic", "fr": "French",
               "es": "Spanish", "de": "German", "ru": "Russian", "it": "Italian",
               "pt": "Portuguese", "tr": "Turkish", "fa": "Persian"}


def _lang(code: str) -> str:
    return _LANG_NAMES.get(code, code)


def _run(tokenizer, model, user_text: str, max_new_tokens: int) -> str:
    """Greedy-decode one user turn through MLX, stripped of turn/end markers."""
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler

    prompt = f"<start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n"
    out = mlx_lm.generate(model, tokenizer, prompt, max_tokens=max_new_tokens,
                          sampler=make_sampler(temp=0.0), verbose=False)
    for marker in ("<end_of_turn>", "<eos>", "<start_of_turn>"):
        out = (out or "").split(marker)[0]
    return out.strip()


def _translate_instruction(text: str, source: str, target: str, context: str = "") -> str:
    """Comprehension prompt: faithful and complete, not literal or summarised.

    `context` is an optional per-video note — who and what the video is about, and
    the established spellings of names the ASR tends to mangle. The model uses it to
    render "שייח' עמוזה" (a mistranscription) as "Sheikha Moza" rather than guessing.
    """
    src, tgt = _lang(source), _lang(target)
    hint = f"{context.strip()}\n\n" if context and context.strip() else ""
    return (
        f"{hint}"
        f"Translate the following {src} text into clear, natural {tgt} for subtitles. "
        f"Understand the meaning and translate it faithfully and completely: keep every "
        f"clause, every detail and any repeated emphasis; preserve all names, "
        f"organizations, numbers and specific references, using their established {tgt} "
        f"names and respecting grammatical gender. Do not summarize, shorten, omit, or "
        f"translate word-for-word. Output only the {tgt} translation, nothing else.\n\n"
        f"{src}: {text}"
    )


def is_target_text(text: str) -> bool:
    """Reject empty / untranslated (still source-script) output."""
    t = (text or "").strip()
    if len(t) < 2:
        return False
    lat = len(_LATIN.findall(t))
    heb = len(_HEBREW.findall(t))
    return lat >= 2 and lat > heb


def load(device: str | None = None):
    """Load general Gemma 3 (12B, 4-bit) through MLX.

    Returns (tokenizer, model, None) to match the transformers-era
    (processor, model, device) call sites — device is unused under MLX.
    """
    import mlx_lm

    path = str(MODEL_PATH) if MODEL_PATH.is_dir() else HUB_ID
    print(f"  translate: loading {path} (mlx 4-bit)", file=sys.stderr)
    model, tokenizer = mlx_lm.load(path)
    # Gemma ends a turn with <end_of_turn>, not <eos>. Without it in the stop set
    # generation runs to max_tokens every time — slow, and the text trails a wall
    # of <end_of_turn> tokens. Register it so generation stops at the real end.
    try:
        eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if isinstance(eot, int) and eot >= 0:
            tokenizer.eos_token_ids.add(eot)
    except Exception:
        pass
    return tokenizer, model, None


def generate(tokenizer, model, text: str, *, source: str, target: str,
             context: str = "", device=None, max_new_tokens: int = 400) -> str:
    src = (text or "").strip()
    if not src:
        return ""
    return _run(tokenizer, model, _translate_instruction(src, source, target, context),
                max_new_tokens)


def _proper_nouns(text: str) -> set[str]:
    """Capitalised words that are not merely sentence-initial."""
    out: set[str] = set()
    for match in re.finditer(r"\b[A-Z][a-zA-Z'-]{2,}\b", text or ""):
        before = (text[: match.start()]).rstrip()
        if not before or before[-1] in ".!?":
            continue
        out.add(match.group(0).lower())
    return out


def _digits(text: str) -> set[str]:
    return set(re.findall(r"\d+", text or ""))


def _has_negation(text: str) -> bool:
    words = set(re.findall(r"[a-z']+", (text or "").lower()))
    return bool(words & _NEGATIONS) or "n't" in (text or "").lower()


def shorten(processor, model, source_text: str, current_en: str, max_words: int, *,
            source: str, target: str, context: str = "", device=None) -> str | None:
    """Re-translate the source more concisely. None if meaning would be at risk.

    Re-translating (rather than compressing the English) keeps the model on its
    task, and the guards below refuse any rewrite that drops a number, a name, or
    a negation.
    """
    have = len((current_en or "").split())
    want = max(3, min(max_words, have - 1))
    src, tgt = _lang(source), _lang(target)
    hint = f"{context.strip()}\n\n" if context and context.strip() else ""
    instruction = (
        f"{hint}"
        f"Translate the following {src} text into {tgt} as concisely as possible, in at "
        f"most {want} words, while keeping every name, number and negation. Output only "
        f"the {tgt} translation.\n\n{src}: {source_text}"
    )
    out = _run(processor, model, instruction, max(64, want * 4 + 40))
    out = (out or "").strip().strip('"')
    if not is_target_text(out) or "[[C" in out:
        return None
    n = len(out.split())
    if n >= have or n < max(3, 0.5 * have):
        return None
    if not _digits(current_en) <= _digits(out):
        return None
    keep_names = _proper_nouns(current_en)
    if keep_names:
        low = out.lower()
        if sum(1 for name in keep_names if name in low) < 0.8 * len(keep_names):
            return None
    if _has_negation(current_en) and not _has_negation(out):
        return None
    return out


def run(m: dict[str, Any], workdir: Path, *, source: str, target: str, save=None) -> None:
    from . import manifest

    segments = m["segments"]
    for seg in segments:
        if seg["keep"] and not (seg.get("text_en") or "").strip():
            seg["text_en"] = seg["text"]

    dub = [s for s in segments if not s["keep"]]
    todo = [s for s in dub if not (s.get("text_en") or "").strip()]
    if not todo:
        return
    context = m["source"].get("context") or ""
    processor, model, device = load()
    try:
        for n, seg in enumerate(dub, 1):
            if (seg.get("text_en") or "").strip():
                continue
            # Translate each segment on its own — standalone output is deterministic
            # and faithful, which matters more for a dub than resolving a pronoun; a
            # marked multi-segment window makes the model drift a clause onto its
            # neighbour. The per-video `context` supplies names the ASR mangles.
            text = generate(processor, model, seg["text"], source=source,
                            target=target, context=context, device=device)
            if is_target_text(text):
                seg["text_en"] = text.strip()
            else:
                seg["keep"], seg["keep_reason"] = True, "mt_failed"
                seg["text_en"] = seg["text"]
                print(f"  translate: seg {seg['id']} failed → keep original",
                      file=sys.stderr)
            if n % 8 == 0:
                print(f"  translate: {n}/{len(dub)}", file=sys.stderr)
                if save:
                    save()
    finally:
        free(model)
    manifest.save(workdir, m)
    missing = [s["id"] for s in segments if not (s.get("text_en") or "").strip()]
    assert not missing, f"segments without text_en: {missing}"


def free(model) -> None:
    import gc

    del model
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass
