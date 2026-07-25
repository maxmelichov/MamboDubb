"""Stage 3 — a flat, time-stamped word stream for the whole video.

Local ASR is the default source, not the video's own captions. Auto-captions
mangle exactly the words that matter most — on the reference video
`שייחה מוזה … בקטאר` ("Sheikha Moza … in Qatar") is captioned `ש חמוזה … בקטב`,
a broken name plus a non-word — and a translator cannot recover from that. The
dedicated Hebrew model also finds ~27% more words than the captions do.

The main ASR pass runs on the separated vocals, which is measurably more accurate
with music out of the way (`בקטר` "in Qatar" rather than `בגדה`, a non-word). But
Demucs sometimes routes speech into the music stem, so whether anyone is speaking
is judged from `source.wav` — the gap-recovery pass below re-listens to the full
mix wherever the vocals produced nothing, which is what stops those passages from
disappearing.

Captions are still used for the one thing they are reliably better at: telling us
*which script* is being spoken, so passages already in the target language can be
kept as original audio (see `foreign_spans`). They also supply proper subtitle
text for those passages.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
WHISPER_MODEL = REPO_ROOT / "models" / "whisper-large-v3-turbo-ct2"
WHISPER_HUB = "ivrit-ai/whisper-large-v3-turbo-ct2"
EN_ASR_MODEL = REPO_ROOT / "models" / "faster-whisper-base.en"
LID_MODEL = REPO_ROOT / "models" / "lang-id-voxlingua107-ecapa"
LID_MIN_PROB = 0.60    # VoxLingua confidence to trust a language label
LID_WINDOW = 4.0       # language-ID in windows this size so a long monologue that
                       # switches language partway is caught, not labelled by majority
VAD_THRESHOLD = 0.4    # a touch more sensitive than Silero's 0.5 default
VAD_PAD_MS = 150       # pad speech edges so soft word starts/ends are not trimmed
VAD_MERGE_GAP = 0.5    # join speech regions separated by less than this
LID_SHORT = 1.5        # a run shorter than this is below what the LID can call alone
VAD_MIN_SEC = 0.6      # ignore speech blips shorter than this (a short English tail
                       # like "just want to help" must still be kept, not clipped)
SPAN_TAIL_LOGPROB = -0.5  # while the English model reads this confidently past the LID
SPAN_TAIL_STEP = 0.5      # boundary, in steps of this, the trailing word is still English
SPAN_TAIL_MAX = 1.5       # ...extend the span up to this far to let the speaker finish
PAUSE_HOP = 0.05          # resolution of the "has the voice stopped?" scan
PAUSE_FLOOR = 0.008       # vocal energy below this is a pause, not speech
PAUSE_FRAMES = 2          # consecutive quiet frames a real pause needs, so the closure
                          # of a plosive inside a word does not end a span
SPAN_END_PAD = 0.25       # keep a span's end no further than this past its last English
                          # word: _extend_english_end widens the *decode* window to
                          # catch a trailing word, but the span must not then claim the
                          # Hebrew beyond it — a keep span there plays the source voice.
# VoxLingua107 reports languages by their old ISO-639 codes; map to ours.
_LID_ALIAS = {"iw": "he", "in": "id", "ji": "yi"}

GAP_MIN_SEC = 1.2      # shortest unheard stretch worth a second ASR pass
GAP_RMS_FLOOR = 0.012  # below this the stretch is silence, not missed speech
GAP_PAD = 0.35

# Caption chrome: sound tags and speaker arrows carry no speech.
_CHROME = re.compile(r"[\[\(](?:[^\]\)]{0,40})[\]\)]")
_ARROWS = re.compile(r">>+")
# Everything that is not a letter/digit, for comparing repeated words regardless
# of the trailing punctuation a looped phrase tends to end on.
_WORD_PUNCT = re.compile(r"[^\w]", re.UNICODE)


def clean_token(raw: str) -> tuple[str, bool]:
    """Return (word, is_speaker_break). Empty word means: drop this token."""
    text = (raw or "").strip()
    brk = bool(_ARROWS.search(text))
    text = _ARROWS.sub(" ", text)
    text = _CHROME.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text, brk


def words_from_json3(path: Path, *, limit: float | None = None) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    pending_break = False
    for event in data.get("events") or []:
        base = float(event.get("tStartMs") or 0.0) / 1000.0
        for seg in event.get("segs") or []:
            text, brk = clean_token(seg.get("utf8") or "")
            brk = brk or bool(seg.get("isSpeakerChange"))
            if not text:
                pending_break = pending_break or brk
                continue
            t = base + float(seg.get("tOffsetMs") or 0.0) / 1000.0
            if limit is not None and t >= limit:
                return out
            if out and t < out[-1]["t"]:
                t = out[-1]["t"]  # captions are monotonic; guard anyway
            out.append({"t": round(t, 3), "text": text, "brk": brk or pending_break})
            pending_break = False
    return out


def load_asr():
    from faster_whisper import WhisperModel

    model_path = str(WHISPER_MODEL) if WHISPER_MODEL.is_dir() else WHISPER_HUB
    print(f"  transcript: ASR {model_path} (cpu)", file=sys.stderr)
    return WhisperModel(model_path, device="cpu", compute_type="auto")


def _words_of(segments, offset: float = 0.0, limit: float | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seg in segments:
        for w in seg.words or []:
            text, _brk = clean_token(w.word)
            if not text:
                continue
            t = float(w.start) + offset
            if limit is not None and t >= limit:
                return out
            # Keep Whisper's measured word end — segmentation and placement need
            # real durations, not a guess from onsets alone (see segments.word_ends).
            end = float(w.end) + offset
            out.append({"t": round(t, 3), "end": round(max(end, t + 0.02), 3),
                        "text": text, "brk": False})
    return out


def collapse_repeats(words: list[dict[str, Any]], *, max_ngram: int = 4,
                     min_reps: int = 3) -> list[dict[str, Any]]:
    """Drop Whisper's back-to-back repetition hallucinations.

    On silent or musical stretches the decoder sometimes loops a phrase — the
    reference video's opening transcribes as `הכי גדולות הכי גדולות הכי גדולות`,
    which then "translates" to fluent nonsense. Any n-gram of up to `max_ngram`
    words repeated `min_reps` or more times in a row collapses to a single
    occurrence, keeping the first copy's timings. The threshold stays at three so
    an ordinary spoken doubling (`מאוד מאוד`) is left alone.
    """
    def key(j: int) -> str:
        # Compare on the bare word: a looped phrase often ends on punctuation
        # ("...גדולות גדולות גדולות.") that would otherwise break the match.
        return _WORD_PUNCT.sub("", words[j]["text"]).lower()

    out: list[dict[str, Any]] = []
    i, n = 0, len(words)
    while i < n:
        collapsed = False
        for length in range(min(max_ngram, (n - i) // 2), 0, -1):
            base = [key(i + k) for k in range(length)]
            reps = 1
            while (i + (reps + 1) * length <= n
                   and [key(i + reps * length + k) for k in range(length)] == base):
                reps += 1
            if reps >= min_reps:
                out.extend(words[i : i + length])
                i += reps * length
                collapsed = True
                break
        if not collapsed:
            out.append(words[i])
            i += 1
    return out


_SPLIT_MARK = re.compile(r"^['׳`’\-–]")


def join_split_marks(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rejoin a word the ASR split at a geresh or a hyphen.

    Hebrew spells foreign sounds with a geresh — ג'יהאד, אג'נדה, ג'בהת — and Whisper
    emits the geresh as the start of a *new* token: `לג` + `'יהאד`, `אל` + `-קאעידה`.
    The halves reach the translator as separate words and it misreads the grammar
    that hangs on them: "שקוראת לג'יהאד" ("which calls **for** jihad") came back as
    "which she calls jihad", because the preposition ל was stranded on its own token.
    Rejoined, the same model and prompt render it correctly.

    The merged word keeps the first half's onset and the second half's end, so timing
    is unchanged. A token after a sentence end is left alone — there the mark starts
    something new rather than continuing a word.
    """
    out: list[dict[str, Any]] = []
    for w in words:
        prev = out[-1] if out else None
        if (prev and _SPLIT_MARK.match(w.get("text") or "")
                and (prev.get("text") or "") and not prev["text"].endswith((".", "!", "?"))
                and not w.get("brk")):
            prev["text"] += w["text"]
            if w.get("end") is not None:
                prev["end"] = w["end"]
            continue
        out.append(dict(w))
    return out


def words_from_whisper(model, source_wav: Path, lang: str, *,
                       limit: float | None = None) -> list[dict[str, Any]]:
    segments, _info = model.transcribe(
        str(source_wav), language=lang, beam_size=5, word_timestamps=True,
        condition_on_previous_text=False, vad_filter=True,
    )
    return _words_of(segments, limit=limit)


def load_en_asr():
    from faster_whisper import WhisperModel

    if not EN_ASR_MODEL.is_dir():
        return None
    return WhisperModel(str(EN_ASR_MODEL), device="cpu", compute_type="auto")


def load_vad():
    from silero_vad import load_silero_vad

    return load_silero_vad()


def vad_regions(vad, source_wav: Path, *, sr: int = 16000,
                merge_gap: float = VAD_MERGE_GAP) -> list[tuple[float, float]]:
    """Speech start/end from Silero VAD, joined across brief pauses.

    A tiny model (~a millisecond of compute per chunk) that gives real speech
    boundaries — far cheaper than transcribing a region just to find where it
    begins and ends, and accurate enough to keep an utterance from being clipped.
    """
    import torch
    from silero_vad import get_speech_timestamps

    from . import audio

    wav = torch.from_numpy(audio.decode_mono(source_wav, sr).astype("float32"))
    stamps = get_speech_timestamps(wav, vad, sampling_rate=sr, return_seconds=True,
                                   threshold=VAD_THRESHOLD, speech_pad_ms=VAD_PAD_MS)
    regions: list[list[float]] = []
    for t in stamps:
        a, b = float(t["start"]), float(t["end"])
        if regions and a - regions[-1][1] <= merge_gap:
            regions[-1][1] = b
        else:
            regions.append([a, b])
    return [(a, b) for a, b in regions]


def load_lid():
    from speechbrain.inference.classifiers import EncoderClassifier

    if not LID_MODEL.is_dir():
        return None
    return EncoderClassifier.from_hparams(source=str(LID_MODEL), savedir=str(LID_MODEL))


def detect_language(lid, clip) -> tuple[str, float]:
    """Spoken-language code and probability for a clip, via VoxLingua107 ECAPA.

    One embedding forward pass, not a transcription — this is the cheap way to ask
    "what language is this?", which the Hebrew-tuned ASR cannot answer (it always
    says Hebrew, even over English speech).
    """
    import torch

    sig = torch.from_numpy(clip.astype("float32")).unsqueeze(0)
    _prob, score, _idx, label = lid.classify_batch(sig)
    code = label[0].split(":")[0].strip()
    return _LID_ALIAS.get(code, code), float(score.exp())


def language_segments(vad, lid, source_wav: Path, *, win: float = LID_WINDOW,
                      merge_gap: float = VAD_MERGE_GAP) -> list[tuple[float, float, str | None]]:
    """VAD speech split into maximal same-language runs (start, end, lang).

    Each speech region is language-ID'd in windows of at most `win` seconds, so a
    long monologue that switches language partway — English narration into a Hebrew
    interview — is caught rather than labelled by its majority. Adjacent windows of
    the same language merge back into one run with VAD-precise boundaries; a window
    the classifier is unsure about is labelled None and breaks the run.
    """
    from . import audio

    labeled: list[list] = []
    for a, b in vad_regions(vad, source_wav, merge_gap=0.2):
        t = a
        while t < b - 1e-6:
            c = min(b, t + win)
            clip = audio.decode_mono(source_wav, 16000, start=t, end=c)
            lang, prob = detect_language(lid, clip)
            labeled.append([round(t, 3), round(c, 3), lang if prob >= LID_MIN_PROB else None])
            t = c
    merged: list[list] = []
    for s, e, lang in labeled:
        if merged and merged[-1][2] == lang and s - merged[-1][1] <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e, lang])
    return [(s, e, lang) for s, e, lang in merged]


def _extend_english_end(en_model, source_wav: Path, b: float, limit: float) -> float:
    """Grow an English span end while the English model still reads English past it.

    Used to reclaim a trailing word ("...children") the coarse LID window placed on
    the Hebrew side. The English-only model decodes real English with high
    avg_logprob and Hebrew as low-confidence gibberish, so the logprob is the stop.
    """
    from . import audio

    while b + 0.25 < limit:
        end = min(b + SPAN_TAIL_STEP, limit)
        segs, _info = en_model.transcribe(audio.decode_mono(source_wav, 16000, start=b, end=end),
                                          language="en", beam_size=5, vad_filter=False)
        segs = list(segs)
        if not segs:
            break
        lp = sum(s.avg_logprob for s in segs) / len(segs)
        text = " ".join(s.text.strip() for s in segs).strip()
        if lp < SPAN_TAIL_LOGPROB or len(text) < 2:
            break
        b = end
    return round(b, 3)


def _voice_pause_after(source_wav: Path, start: float, limit: float) -> float | None:
    """When the voice next stops, or None if it does not stop before `limit`.

    A pause needs `PAUSE_FRAMES` consecutive quiet frames, so the closure of a plosive
    inside a word is not mistaken for the end of the sentence.
    """
    from . import audio

    if limit - start <= PAUSE_HOP:
        return None
    levels = audio.frame_rms(audio.decode_mono(source_wav, 16000, start=start, end=limit),
                             16000, PAUSE_HOP)
    quiet = 0
    for k, level in enumerate(levels):
        if level < PAUSE_FLOOR:
            quiet += 1
            if quiet >= PAUSE_FRAMES:
                return round(start + (k - quiet + 1) * PAUSE_HOP, 3)
        else:
            quiet = 0
    return None


def _reclaim_leading_fragment(en_model, source_wav: Path, cand: float, a: float,
                              b: float) -> float:
    """Take back the fragment VAD broke off the front of an utterance, if it is speech.

    The counterpart of `_extend_english_end`, for the same failure at the other edge:
    VAD splits a soft first word into a region of its own ("Frankly," at 270.80-271.70,
    then "I had the same concerns…" from 272.00), and the language classifier cannot
    judge a fragment that short — it labelled that one `mi`, Maori, confidently. So the
    run started on the speaker's *second* word and the viewer heard them from the
    middle of their own sentence (4:31, "the VAD wakes up too late").

    The English-only model decides instead, reading the fragment together with the run
    it abuts: real English comes back confident with a word placed inside the fragment,
    while Hebrew decodes as low-confidence gibberish and is refused. Bounded to a
    fragment of `LID_SHORT`, so a mistake costs a moment of source voice at a boundary
    and never a whole utterance.
    """
    from . import audio

    if not 0.0 < a - cand <= LID_SHORT:
        return round(a, 3)
    segs, _info = en_model.transcribe(
        audio.decode_mono(source_wav, 16000, start=cand, end=min(b, cand + LID_WINDOW)),
        language="en", beam_size=5, vad_filter=False, word_timestamps=True)
    segs = list(segs)
    if not segs:
        return round(a, 3)
    lp = sum(s.avg_logprob for s in segs) / len(segs)
    starts = [float(w.start) for s in segs for w in (s.words or []) if (w.word or "").strip()]
    if lp < SPAN_TAIL_LOGPROB or not starts or cand + min(starts) >= a - 0.05:
        return round(a, 3)
    return round(cand, 3)


def detect_spoken_target_spans(en_model, vad, lid, source_wav: Path, total: float,
                               target: str, *, known: list[tuple[float, float]] | None = None
                               ) -> list[dict[str, Any]]:
    """Speech regions actually spoken in the target language, as original-audio spans.

    Silero VAD gives the precise start/end of every utterance; VoxLingua107 says
    which language each window is. A run the language model calls the target language
    is kept as original audio (the Hebrew-tuned ASR would only render it as
    gibberish). The English-only model runs once per such run — for the subtitle
    text, not to guess the language — and the span uses the VAD boundaries.
    """
    from . import audio

    known = known or []
    lsegs = language_segments(vad, lid, source_wav)
    spans: list[dict[str, Any]] = []
    for i, (a, b, lang) in enumerate(lsegs):
        if lang != target or b - a < VAD_MIN_SEC:
            continue
        if any(a < kb and ka < b for ka, kb in known):
            continue
        # The coarse LID window can put a short trailing word ("...children") on the
        # Hebrew side; the English-only model, by contrast, reads it as confident
        # English. Step the end forward while it keeps reading English, so the
        # speaker finishes — bounded so it can't run into the real Hebrew.
        nxt = lsegs[i + 1][0] if i + 1 < len(lsegs) else total
        b = _extend_english_end(en_model, source_wav, b, min(b + SPAN_TAIL_MAX, nxt + 0.6, total))
        # Same at the leading edge, when the run just before is a fragment VAD broke
        # off this one — near enough to be the same breath, and not itself English
        # (that would already be a span of its own).
        if i and lsegs[i - 1][2] != target and a - lsegs[i - 1][1] <= VAD_MERGE_GAP:
            a = _reclaim_leading_fragment(en_model, source_wav, lsegs[i - 1][0], a, b)
        clip = audio.decode_mono(source_wav, 16000, start=a, end=b)
        segs, _info = en_model.transcribe(clip, language=target, beam_size=5,
                                          vad_filter=False, word_timestamps=True)
        segs = list(segs)
        text = " ".join(s.text.strip() for s in segs).strip()
        got = [w for s in segs for w in (s.words or []) if (w.word or "").strip()]
        if got:
            # The decode window `b` was widened by _extend_english_end to catch a
            # trailing English word, but the re-transcription only emits words where
            # English is actually spoken. Clamp the span to its last English word so
            # it never claims the Hebrew tail beyond — a keep span there plays the
            # source voice (the "I can hear the Hebrew speaker" bleed) and steals
            # those seconds from the dub that should cover them.
            #
            # Neither timestamp is the real end. Whisper's last word bunches early —
            # "…and children." ended at 110.82 of speech that ran to 111.40, and
            # clamping there cut the speaker off mid-word (1:51). The classifier's
            # boundary is coarse the other way: its windows are `win` long, so an
            # English run can be called through 206.90 when the Hebrew narrator has
            # been talking since 205.04, and keeping that span airs him (3:25).
            # The voice itself is the honest boundary: end at the first real pause
            # after the last English word, and fall back to the early word timestamp
            # when no pause is in reach — with the speakers back to back, stopping
            # early is the one that does not put the source voice on air.
            last = a + float(got[-1].end)
            stop = _voice_pause_after(source_wav, last,
                                      min(b + SPAN_END_PAD, last + SPAN_TAIL_MAX, total))
            b = stop if stop is not None else min(b, round(last + SPAN_END_PAD, 3))
        span_words = ([{"t": round(a + float(w.start), 3), "text": w.word.strip()} for w in got]
                      or [{"t": round(a, 3), "text": tok} for tok in text.split()]
                      or [{"t": round(a, 3), "text": "…"}])
        spans.append({"start": round(a, 3), "end": round(min(total, b), 3),
                      "text": text or "…", "words": span_words})
        print(f"  transcript: {target}-spoken {a:.1f}-{b:.1f}s: {text[:60]}",
              file=sys.stderr)
    return spans


def uncovered_windows(words: list[dict[str, Any]], levels, hop: float, total: float,
                      *, min_gap: float = GAP_MIN_SEC, floor: float = GAP_RMS_FLOOR,
                      known: list[tuple[float, float]] | None = None
                      ) -> list[tuple[float, float]]:
    """Audible stretches the transcript says nothing about.

    `known` marks ranges already accounted for by another source, so they are not
    transcribed a second time — two overlapping versions of the same speech would
    both be placed, one after the other.
    """
    import numpy as np

    covered = np.zeros(len(levels), dtype=bool)
    for w in words:
        i0 = max(0, int((w["t"] - 0.2) / hop))
        i1 = min(len(covered), int((w["t"] + 0.8) / hop) + 1)
        covered[i0:i1] = True
    for a, b in known or []:
        covered[max(0, int(a / hop)) : min(len(covered), int(b / hop) + 1)] = True
    out: list[tuple[float, float]] = []
    i = 0
    while i < len(covered):
        if covered[i]:
            i += 1
            continue
        j = i
        while j < len(covered) and not covered[j]:
            j += 1
        a, b = i * hop, min(total, j * hop)
        if b - a >= min_gap and float(np.median(levels[i:j])) >= floor:
            out.append((a, b))
        i = j
    return out


def recover_gaps(model, source_wav: Path, words: list[dict[str, Any]], lang: str,
                 total: float, known: list[tuple[float, float]] | None = None
                 ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Transcribe again, in isolation, wherever audible speech went unheard.

    A single pass over a whole film skips things — most often a passage in
    another language, which Whisper drops rather than transcribes. Those seconds
    would otherwise have no segment at all, and a segment is the only way audio
    of any kind reaches the output. Recovered target-language text becomes a kept
    segment; recovered source-language text gets dubbed like anything else.
    """
    from . import audio
    from .segments import latin_ratio

    levels = audio.frame_rms(audio.decode_mono(source_wav, 16000), 16000, 0.1)
    windows = uncovered_windows(words, levels, 0.1, total, known=known)
    if not windows:
        return words, []
    found: list[dict[str, Any]] = []
    spans: list[dict[str, Any]] = []
    for a, b in windows:
        clip = audio.decode_mono(source_wav, 16000, start=max(0.0, a - GAP_PAD),
                                 end=min(total, b + GAP_PAD))
        try:
            segs, _ = model.transcribe(clip, language=lang, beam_size=5,
                                       word_timestamps=True,
                                       condition_on_previous_text=False, vad_filter=True)
        except Exception as exc:
            print(f"  transcript: gap {a:.1f}-{b:.1f}s failed ({exc})", file=sys.stderr)
            continue
        got = [w for w in _words_of(segs, offset=max(0.0, a - GAP_PAD))
               if a - 0.15 <= w["t"] <= b + 0.15]
        if not got:
            continue
        text = " ".join(w["text"] for w in got)
        print(f"  transcript: recovered {len(got)} words at {a:.1f}-{b:.1f}s: {text[:60]}",
              file=sys.stderr)
        if latin_ratio(text) > 0.5:
            # Already in the target language, so it will play as original audio and
            # the word timings do not need to be trusted — only the window does.
            # Whisper bunches word times inside a short clip, which would otherwise
            # leave part of the passage uncovered and therefore silent.
            spans.append({"start": round(a, 3), "end": round(b, 3), "text": text,
                          "words": got})
        else:
            found.extend(got)
    if not found:
        return words, spans
    merged = sorted(words + found, key=lambda w: w["t"])
    # Drop anything the recovery pass duplicated at a window edge.
    out = [merged[0]]
    for w in merged[1:]:
        if w["t"] - out[-1]["t"] < 0.05 and w["text"] == out[-1]["text"]:
            continue
        out.append(w)
    return out, spans


def merge_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop target-language spans already covered by an earlier one.

    Two sources can describe the same speech — the captions and the gap-recovery
    pass — and keeping both would place the same words twice, one after the other.
    """
    out: list[dict[str, Any]] = []
    for s in sorted(spans, key=lambda x: (x["start"], -(x["end"] - x["start"]))):
        if out and s["start"] < out[-1]["end"] - 0.05:
            out[-1]["end"] = max(out[-1]["end"], s["end"])
            continue
        out.append(dict(s))
    return out


def foreign_spans(words: list[dict[str, Any]], *, min_sec: float = 0.8,
                  join_gap: float = 2.0) -> list[dict[str, Any]]:
    """Stretches of caption text already written in the target script.

    Those passages are spoken in the target language, so they should play as
    original audio rather than being "translated" from a phonetic transcription
    of themselves.
    """
    from .segments import latin_ratio

    runs: list[list[dict[str, Any]]] = []
    for w in words:
        if latin_ratio(w["text"]) <= 0.5:
            continue
        if runs and w["t"] - runs[-1][-1]["t"] <= join_gap:
            runs[-1].append(w)
        else:
            runs.append([w])
    spans = []
    for run in runs:
        start, end = run[0]["t"], run[-1]["t"] + 0.8
        if end - start >= min_sec:
            spans.append({"start": round(start, 3), "end": round(end, 3),
                          "text": " ".join(w["text"] for w in run),
                          "words": [{"t": w["t"], "text": w["text"]} for w in run]})
    return spans


def run(m: dict[str, Any], workdir: Path, *, src_lang: str, tgt_lang: str = "en",
        prefer: str = "auto") -> None:
    raw = m["files"].get("captions_raw")
    limit = m["source"].get("duration")
    has_captions = bool(raw) and Path(raw).is_file()
    recovered: list[dict[str, Any]] = []
    en_spans: list[dict[str, Any]] = []
    caption_words = words_from_json3(Path(raw), limit=limit) if has_captions else []

    if prefer == "captions" and not caption_words:
        raise SystemExit("--transcript captions was requested but no caption file is available")
    if prefer == "captions":
        words, origin = caption_words, "captions"
    else:
        try:
            source_wav = workdir / m["files"]["source_wav"]
            vocals = workdir / m["files"].get("vocals", "")
            model = load_asr()
            # Transcribe the isolated voice; judge speech presence from the mix.
            words = words_from_whisper(model, vocals if vocals.is_file() else source_wav,
                                       src_lang, limit=limit)
            caption_spans = foreign_spans(caption_words) if caption_words else []
            words, recovered = recover_gaps(
                model, source_wav, words, src_lang,
                float(limit or m["source"]["duration"]),
                known=[(s["start"], s["end"]) for s in caption_spans])
            words = join_split_marks(collapse_repeats(words))
            # Real target-language speech the source model rendered as gibberish:
            # Silero VAD finds the utterances, VoxLingua107 says which are English,
            # and those are kept as original audio instead of dubbed from nonsense.
            en_model = load_en_asr()
            vad = load_vad()
            lid = load_lid()
            if en_model is not None and vad is not None and lid is not None:
                en_spans = detect_spoken_target_spans(
                    en_model, vad, lid, vocals if vocals.is_file() else source_wav,
                    float(limit or m["source"]["duration"]), tgt_lang)
            origin = "asr"
        except Exception as exc:
            if prefer == "asr" or not caption_words:
                raise
            print(f"  transcript: ASR unavailable ({exc}) — falling back to captions",
                  file=sys.stderr)
            words, origin = caption_words, "captions"

    if not words:
        raise SystemExit(
            "No transcript words were produced. For a local file pass --captions <file.json3>, "
            "or check that the ASR model is present under models/."
        )

    # VAD+LID English spans are authoritative — their boundaries are precise, so a
    # caption or gap-recovery span (coarser, Hebrew-model or caption derived) is kept
    # only where VAD/LID found no target-language speech, never overriding it.
    others = (foreign_spans(caption_words) if caption_words else []) + recovered
    others = [s for s in others
              if not any(s["start"] < e["end"] and e["start"] < s["end"] for e in en_spans)]
    spans = merge_spans(en_spans + others)
    path = workdir / "words.json"
    path.write_text(
        json.dumps({"origin": origin, "words": words, "foreign_spans": spans},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    m["files"]["words"] = "words.json"
    m["source"]["transcript_origin"] = origin
    print(f"  transcript: {len(words)} words from {origin}"
          f"{f', {len(spans)} target-language spans from captions' if spans else ''}",
          file=sys.stderr)


def load_words(workdir: Path, m: dict[str, Any]) -> list[dict[str, Any]]:
    data = json.loads((workdir / m["files"]["words"]).read_text(encoding="utf-8"))
    return data["words"]


def load_foreign_spans(workdir: Path, m: dict[str, Any]) -> list[dict[str, Any]]:
    data = json.loads((workdir / m["files"]["words"]).read_text(encoding="utf-8"))
    return data.get("foreign_spans") or []
