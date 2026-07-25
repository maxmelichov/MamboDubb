"""Stage 5 — general-purpose Gemma as a comprehension translator, one segment at a time.

A dedicated MT model (TranslateGemma) translates literally and has no world
knowledge, so it drops or garbles the references that matter — "האחים המוסלמים"
(the Muslim Brotherhood, an organisation) came back as "brotherhood among
Muslims", and "שייח'ה" (the Sheikha) as "a CEO". A general instruction model
*understands* the text: it keeps the named organisations and people, and — asked
to — every clause and detail, rather than paraphrasing them away. That is what a
dub needs: the meaning, complete, in natural English.

The model is general Gemma 4 (12B) run through MLX (Apple-native), quantised so it
fits in unified memory. Each segment is translated on its own, deterministically
(greedy); a segment never sees its neighbours, which keeps the pieces from bleeding
into each other.

Which 12B quantisation matters more than it looks. Scored on the 30 dubbed segments
of the Qatar piece against nine things the Hebrew must survive — "הגז הקטארי" as
"Qatari gas", "שתי ההפיכות" as "two coups", "לבנה השני" as "her second son",
"שלושה עשורים" as "three decades", the named organisations intact:

  6bit       9.7 GB  9/9, and the fastest of the ones that pass. The default.
  qat-4bit  10.9 GB  9/9 too, but 2.4x slower (its MLP layers are 8-bit) and bigger.
  8bit      12.7 GB  8/9 — loses "her second son". No gain for the extra 3 GB.
  4bit       6.3 GB  5/9. Same footprint as the Gemma 3 it replaces, and reads
                     fluently, which is what makes it dangerous: "the Qatari fund",
                     "two turnovers", "his son", "the following two decades".
  mxfp4      6.3 GB  6/9, about 3x faster, but "the Qatari loot" for the gas.

For reference the Gemma 3 12B 4-bit it replaces scored 7/9. Below 6 bits this model
does not hold Hebrew, so trading precision for memory is a real quality decision, not
a free one — but if 9.7 GB is too much, MODEL_PATH is the only line to change.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "gemma-4-12B-it-6bit"
HUB_ID = "mlx-community/gemma-4-12B-it-6bit"

_LATIN = re.compile(r"[A-Za-z]")
_HEBREW = re.compile(r"[֐-׿]")
_NEGATIONS = {"not", "no", "never", "cannot", "none", "without", "nor", "n't"}

_LANG_NAMES = {"he": "Hebrew", "en": "English", "ar": "Arabic", "fr": "French",
               "es": "Spanish", "de": "German", "ru": "Russian", "it": "Italian",
               "pt": "Portuguese", "tr": "Turkish", "fa": "Persian"}


def _lang(code: str) -> str:
    return _LANG_NAMES.get(code, code)


_TURN_MARKERS = ("<end_of_turn>", "<eos>", "<start_of_turn>", "<turn|>", "<|turn>")


def _prompt(tokenizer, user_text: str):
    """One user turn, rendered by the model's own chat template.

    Gemma 4 changed the turn syntax — `<|turn>user … <turn|>` where Gemma 3 had
    `<start_of_turn>user … <end_of_turn>` — so the markers can no longer be written
    by hand. The template gets them right for either generation. Two details matter:
    it is tokenised here (the template emits `<bos>` itself, and generating from a
    string would prepend a second one), and `enable_thinking=False` makes the Gemma 4
    generation prompt pre-close the thought channel (`<|channel>thought\\n<channel|>`),
    so the model answers with the translation instead of reasoning its way there.
    """
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template([{"role": "user", "content": user_text}],
                                                 add_generation_prompt=True,
                                                 enable_thinking=False)
        except Exception as exc:                                  # template too new/old
            print(f"  translate: chat template failed ({exc}); using Gemma 3 markers",
                  file=sys.stderr)
    return f"<start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n"


def _run(tokenizer, model, user_text: str, max_new_tokens: int) -> str:
    """Greedy-decode one user turn through MLX, stripped of turn/end markers."""
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler

    out = mlx_lm.generate(model, tokenizer, _prompt(tokenizer, user_text),
                          max_tokens=max_new_tokens, sampler=make_sampler(temp=0.0),
                          verbose=False) or ""
    if "<channel|>" in out:              # a thought channel opened anyway — keep the answer
        out = out.split("<channel|>")[-1]
    for marker in _TURN_MARKERS:
        out = out.split(marker)[0]
    return _strip_editorial(out)


_NOTE = re.compile(r"\n\s*[*_(]*\s*(?:note|translation note|n\.b\.)\b.*\Z", re.S | re.I)
_BRACKET_ALT = re.compile(r"\[([^\[\]/]{1,40}?)(?:\s*/\s*[^\[\]]{1,40})?\]")


def _strip_editorial(text: str) -> str:
    """Drop a translator's brackets and trailing notes; the TTS would read them aloud.

    Gemma 4 annotates: it brackets words it supplied ("Qatar funds ISIS, [and] Qatar
    funds …"), offers alternatives ("a [connection/link] to"), and appends a note when
    it judges the source garbled ("*(Note: the Hebrew appears to be corrupted…)*"). The
    prompt asks for none of it and mostly gets none; this is the net underneath. On
    clean output every pattern misses and the text is returned as-is.
    """
    text = _NOTE.sub("", text or "").strip()
    text = _BRACKET_ALT.sub(r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _translate_instruction(text: str, source: str, target: str, context: str = "",
                           extra: str = "") -> str:
    """Comprehension prompt: faithful and complete, not literal or summarised.

    `context` is an optional per-video note — who and what the video is about, and
    the established spellings of names the ASR tends to mangle. The model uses it to
    render "שייח' עמוזה" (a mistranscription) as "Sheikha Moza" rather than guessing.
    `extra` is an optional extra sentence appended on a retry (see `_repair_repeat`).
    """
    src, tgt = _lang(source), _lang(target)
    hint = f"{context.strip()}\n\n" if context and context.strip() else ""
    tail = f" {extra.strip()}" if extra and extra.strip() else ""
    return (
        f"{hint}"
        f"Translate the following {src} text into clear, natural {tgt} for subtitles. "
        f"Understand the meaning and translate it faithfully and completely: keep every "
        f"clause, every detail and any repeated emphasis; preserve all names, "
        f"organizations, numbers and specific references, using their established {tgt} "
        f"names and respecting grammatical gender. Do not summarize, shorten, omit, or "
        f"translate word-for-word. Write full words, no contractions (\"we are\" not "
        f"\"we're\", \"do not\" not \"don't\") so the text-to-speech reads them clearly.{tail} "
        f"Output only the {tgt} translation, nothing else — no notes, no comments, no "
        f"square brackets and no alternative renderings.\n\n"
        f"{src}: {text}"
    )


# Coordinators/stopwords that may legitimately repeat; a repeat of one of these is
# never the "education, education" defect and must be left alone.
_REPEAT_SKIP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "is", "are", "was",
    "were", "be", "by", "for", "with", "that", "this", "it", "as", "so", "no", "not",
    "very", "really", "much", "more", "most", "too", "just", "even", "still",
}


def _adjacent_repeat(text: str) -> str | None:
    """The content word in an audible "X, X" / "X and X" collapse, or None.

    Catches a single content word repeated back-to-back (only a comma or a
    coordinator between) — the shape a translator produces when it flattens two
    distinct source words ("השכלה וחינוך") onto one English word. Phrase-level
    emphasis ("Qatar funds Hamas, Qatar funds Al-Qaeda") is not matched: there the
    repeated token has other words between its occurrences, so it is left intact.
    """
    toks = re.findall(r"[A-Za-z']+", text or "")
    for i in range(len(toks) - 1):
        a = toks[i].lower()
        if len(a) < 4 or a in _REPEAT_SKIP:
            continue
        if a == toks[i + 1].lower():
            return toks[i]                                   # "X, X" or "X X"
        if (i + 2 < len(toks) and toks[i + 1].lower() in {"and", "or"}
                and a == toks[i + 2].lower()):
            return toks[i]                                   # "X and X"
    return None


def _strip_adjacent_repeat(text: str) -> str:
    """Last-resort: drop the duplicate half of an "X, X" / "X and X" collapse."""
    text = re.sub(r"\b(\w{4,})\b\s*,\s*\1\b", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\w{4,})\b\s+(and|or)\s+\1\b", r"\1", text, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", text).strip()


def is_target_text(text: str) -> bool:
    """Reject empty / untranslated (still source-script) output."""
    t = (text or "").strip()
    if len(t) < 2:
        return False
    lat = len(_LATIN.findall(t))
    heb = len(_HEBREW.findall(t))
    return lat >= 2 and lat > heb


def load(device: str | None = None):
    """Load general Gemma 4 (12B, 4-bit) through MLX.

    Returns (tokenizer, model, None) to match the transformers-era
    (processor, model, device) call sites — device is unused under MLX.
    """
    import mlx_lm

    path = str(MODEL_PATH) if MODEL_PATH.is_dir() else HUB_ID
    print(f"  translate: loading {path} (mlx 4-bit)", file=sys.stderr)
    try:
        model, tokenizer = mlx_lm.load(path)
    except AttributeError:
        # Gemma 4's tokenizer_config.json writes extra_special_tokens as a list, which
        # transformers 5 accepts and 4.x (pinned here by qwen-tts) does not — it calls
        # .keys() on it. Passing a mapping overrides the file and loads the same
        # tokenizer. Harmless to retry: nothing else in load() has taken effect yet.
        model, tokenizer = mlx_lm.load(
            path, tokenizer_config={"extra_special_tokens": {"video_token": "<|video|>"}})
    # Gemma ends a turn with a turn marker, not <eos>. Without it in the stop set
    # generation runs to max_tokens every time — slow, and the text trails a wall
    # of end-of-turn tokens. Register both generations' markers (Gemma 4 already
    # carries <turn|> in its eos ids; the lookup is a no-op when a name is absent).
    # generation's markers; an unknown name resolves to <unk>, so round-trip it.
    for name in ("<end_of_turn>", "<turn|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(name)
            if isinstance(tid, int) and tid >= 0 and tokenizer.convert_ids_to_tokens(tid) == name:
                tokenizer.eos_token_ids.add(tid)
        except Exception:
            pass
    # Loading leaves as much again in MLX's buffer cache (a 9.7 GB model peaks near
    # 20 GB). It is reclaimable, but on a 26 GB machine there is no reason to hold it.
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass
    return tokenizer, model, None


def generate(tokenizer, model, text: str, *, source: str, target: str,
             context: str = "", device=None, max_new_tokens: int = 400) -> str:
    src = (text or "").strip()
    if not src:
        return ""
    out = _run(tokenizer, model, _translate_instruction(src, source, target, context),
               max_new_tokens)
    return _repair_repeat(tokenizer, model, src, out, source, target, context, max_new_tokens)


def _repair_repeat(tokenizer, model, src: str, out: str, source: str, target: str,
                   context: str, max_new_tokens: int) -> str:
    """Fix an "education, education" collapse if the first pass produced one.

    Greedy decoding is faithful but occasionally flattens two distinct source words
    onto one English word. Re-ask once, with an added instruction to use distinct
    words for distinct items; if the model still repeats, strip the duplicate so the
    listener never hears the same word twice. Both steps are no-ops on clean output,
    so the common case pays only the cheap `_adjacent_repeat` scan.
    """
    dup = _adjacent_repeat(out)
    if not dup:
        return out
    tgt = _lang(target)
    extra = (f"If the source lists several items, give each a distinct {tgt} word — "
             f"do not write the same word twice in a row or in a list.")
    retry = _run(tokenizer, model,
                 _translate_instruction(src, source, target, context, extra=extra),
                 max_new_tokens)
    if is_target_text(retry) and not _adjacent_repeat(retry):
        return retry
    fixed = _strip_adjacent_repeat(out)
    print(f"  translate: de-duplicated repeated word {dup!r}", file=sys.stderr)
    return fixed


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
    # Dubbed segments must have a translation; kept segments need not (an
    # "uncovered" keep is untranscribed audio that only plays as original sound).
    missing = [s["id"] for s in segments
               if not s.get("keep") and not (s.get("text_en") or "").strip()]
    assert not missing, f"dubbed segments without text_en: {missing}"


def free(model) -> None:
    import gc

    del model
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass
