"""Forced alignment: tighten the transcript's word times against a CTC model.

Whisper's word timestamps are decoder attention read sideways: good to a
couple hundred milliseconds, late after speaker handoffs, and every dub
placement anchors on them. A character-level CTC model gives each word the
frame span it actually occupies, which is the difference between a cut that
lands in the pause and one that clips the incoming actor.

Alignment runs per span (a stretch of words with no long silence inside),
never per file: a CTC path that goes wrong drifts to the end of whatever it
was given, so the blast radius of one bad stretch is capped at that stretch.
Every span is also gated by MAX_SHIFT after the fact a word the aligner
wants to move seconds away is a failed alignment, not a discovery, and the
span keeps its Whisper times.

Hebrew only for now (`SUPPORTED`): the model is a Hebrew fine-tune of XLS-R,
and the pipeline's other source languages keep their Whisper times untouched.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ALIGN_MODEL = REPO_ROOT / "models" / "wav2vec2-xls-r-300m-hebrew"
ALIGN_HUB = "imvladikon/wav2vec2-xls-r-300m-hebrew"
SUPPORTED = {"he"}

SPAN_GAP = 0.8       # a silence this long ends a span; CTC drift stops here
SPAN_MAX_SEC = 15.0  # and a span never exceeds this, whatever the gaps say
SPAN_PAD = 0.25      # audio slack on each side, so edge words have room
MAX_SHIFT = 1.5      # a word moved further than this fails the whole span
SR = 16000

# Niqqud and cantillation: the ASR emits bare consonantal Hebrew and the
# aligner's vocabulary matches, so pointing is stripped rather than mapped.
_POINTS = re.compile(r"[֑-ׇ]")


def _normalize(token: str, vocab: dict[str, int]) -> str:
    """The characters of one word that the aligner's vocabulary can spell."""
    bare = _POINTS.sub("", unicodedata.normalize("NFC", token))
    return "".join(ch for ch in bare.lower() if ch in vocab)


_ALIGNER: tuple[Any, Any] | None | bool = None


def load_aligner():
    """(model, processor) on first use, None when unavailable. Cached either way."""
    global _ALIGNER
    if _ALIGNER is None:
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

            source = str(ALIGN_MODEL) if ALIGN_MODEL.is_dir() else ALIGN_HUB
            model = Wav2Vec2ForCTC.from_pretrained(source).eval()
            processor = Wav2Vec2Processor.from_pretrained(source)
            import torch

            if torch.cuda.is_available():
                model = model.cuda()
            _ALIGNER = (model, processor)
        except Exception as exc:
            print(f"  transcript: aligner unavailable ({exc}) whisper times kept",
                  file=sys.stderr)
            _ALIGNER = False
    return _ALIGNER or None


def spans_of(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Consecutive words split at silences and at the span-length cap."""
    spans: list[list[dict[str, Any]]] = []
    for w in words:
        if (spans and w["t"] - spans[-1][-1]["end"] < SPAN_GAP
                and w["end"] - spans[-1][0]["t"] <= SPAN_MAX_SEC):
            spans[-1].append(w)
        else:
            spans.append([w])
    return spans


def _align_span(span: list[dict[str, Any]], wav: Path, model, processor) -> bool:
    """Realign one span's words in place. False = kept Whisper times."""
    import torch
    import torchaudio.functional as F

    from . import audio

    vocab = processor.tokenizer.get_vocab()
    spelled = [_normalize(w["text"], vocab) for w in span]
    if not all(spelled):
        return False                    # a word the vocabulary cannot spell
    sep = vocab.get("|")
    blank = processor.tokenizer.pad_token_id
    tokens: list[int] = []
    bounds: list[tuple[int, int]] = []  # each word's slice of `tokens`
    for i, s in enumerate(spelled):
        if i and sep is not None:
            tokens.append(sep)
        bounds.append((len(tokens), len(tokens) + len(s)))
        tokens.extend(vocab[ch] for ch in s)

    t0 = max(0.0, span[0]["t"] - SPAN_PAD)
    t1 = span[-1]["end"] + SPAN_PAD
    clip = audio.decode_mono(wav, SR, start=t0, end=t1)
    if len(clip) < SR // 4:
        return False
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(torch.from_numpy(clip.astype("float32"))
                       .unsqueeze(0).to(device)).logits
    emission = torch.log_softmax(logits, dim=-1).cpu()
    try:
        path, _scores = F.forced_align(
            emission, torch.tensor([tokens], dtype=torch.int32), blank=blank)
    except Exception:
        return False
    frames = path[0].tolist()
    stride = len(clip) / SR / emission.shape[1]

    # Frame index of each emitted target token, in order. forced_align labels
    # every frame with a token id or blank; a token's frames are the run that
    # consumed it, tracked by walking the target sequence.
    starts: list[float] = [0.0] * len(tokens)
    ends: list[float] = [0.0] * len(tokens)
    ti = -1
    prev = blank
    for fi, label in enumerate(frames):
        if label != blank and (label != prev or fi == 0 or frames[fi - 1] == blank):
            ti += 1
            starts[ti] = fi * stride
        if label != blank:
            ends[ti] = (fi + 1) * stride
        prev = label
    if ti != len(tokens) - 1:
        return False                    # the path did not consume every token

    new_times = []
    for w, (a, b) in zip(span, bounds):
        nt = round(t0 + starts[a], 3)
        ne = round(t0 + ends[b - 1], 3)
        if ne <= nt or abs(nt - w["t"]) > MAX_SHIFT or abs(ne - w["end"]) > MAX_SHIFT:
            return False
        new_times.append((nt, ne))
    for w, (nt, ne) in zip(span, new_times):
        w["t"], w["end"] = nt, ne
    return True


def refine_word_times(words: list[dict[str, Any]], wav: Path, lang: str) -> int:
    """Realign every span it safely can; returns how many words moved."""
    if lang not in SUPPORTED or not words:
        if words and lang not in SUPPORTED:
            print(f"  transcript: no aligner for '{lang}' whisper times kept",
                  file=sys.stderr)
        return 0
    loaded = load_aligner()
    if loaded is None:
        return 0
    model, processor = loaded
    refined = 0
    for span in spans_of(words):
        before = [(w["t"], w["end"]) for w in span]
        if _align_span(span, wav, model, processor):
            refined += sum(1 for w, old in zip(span, before)
                           if (w["t"], w["end"]) != old)
    print(f"  transcript: aligner moved {refined}/{len(words)} word times",
          file=sys.stderr)
    return refined
