"""Stage 3 a flat, time-stamped word stream for the whole video.

Local ASR is the default source, not the video's own captions. Auto-captions
mangle exactly the words that matter most on the reference video
`שייחה מוזה … בקטאר` ("Sheikha Moza … in Qatar") is captioned `ש חמוזה … בקטב`,
a broken name plus a non-word and a translator cannot recover from that. The
dedicated Hebrew model also finds ~27% more words than the captions do.

The main ASR pass runs on the separated vocals, which is measurably more accurate
with music out of the way (`בקטר` "in Qatar" rather than `בגדה`, a non-word). But
Demucs sometimes routes speech into the music stem, so whether anyone is speaking
is judged from `source.wav` the gap-recovery pass below re-examines every
stretch the mix says is audible and the vocals produced no words for, which is
what stops those passages from disappearing. What it *decodes* there is the
vocals again, and only where the vocals hold energy: see `recover_gaps`.

None of that is an argument against a transcript the *user* already has, which is
a different thing from a scraped auto-caption: somebody typed or corrected it, so
`--transcript file` takes it verbatim (.srt, .vtt or .json3) and neither the
download nor the ASR gets a vote. Whether it is any good is the user's problem,
which is exactly the trade they are asking for.

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
from collections.abc import Iterable, Iterator
from typing import Any

from . import script

REPO_ROOT = Path(__file__).resolve().parents[1]
# Hebrew keeps its dedicated fine-tune; every other source reads with the
# vanilla multilingual turbo model, told the language at transcribe time.
WHISPER_MODEL = REPO_ROOT / "models" / "whisper-large-v3-turbo-ct2"
WHISPER_HUB = "ivrit-ai/whisper-large-v3-turbo-ct2"
SRC_ASR_MODEL = REPO_ROOT / "models" / "faster-whisper-large-v3-turbo-ct2"
SRC_ASR_HUB = "deepdml/faster-whisper-large-v3-turbo-ct2"
EN_ASR_MODEL = REPO_ROOT / "models" / "faster-whisper-base.en"
# Any non-English target reads with the multilingual base model instead, told the
# language at transcribe time (`language=tgt`).
TARGET_ASR_MODEL = REPO_ROOT / "models" / "faster-whisper-base"
LID_MODEL = REPO_ROOT / "models" / "lang-id-voxlingua107-ecapa"
LID_MIN_PROB = 0.60    # VoxLingua confidence to trust a language label
LID_WINDOW = 4.0       # language-ID in windows this size so a long monologue that
                       # switches language partway is caught, not labelled by majority
VAD_THRESHOLD = 0.4    # a touch more sensitive than Silero's 0.5 default
VAD_PAD_MS = 150       # pad speech edges so soft word starts/ends are not trimmed
VAD_MERGE_GAP = 0.5    # join speech regions separated by less than this
LID_SHORT = 1.5        # a run shorter than this is below what the LID can call alone
FOREIGN_MIN_SEC = 1.5  # a run in a third language must be at least this long to keep:
                       # no ASR here can read it, so the classifier is the only witness
FOREIGN_MIN_PROB = 0.85  # ...and must be this sure, well above LID_MIN_PROB
FOREIGN_SURE_PROB = 0.95 # at this certainty the classifier outranks the ASR veto:
                         # the Hebrew fine-tune transliterates clear English speech
                         # into Hebrew script at logprob -0.46 (above the fail bar),
                         # while every documented LID mislabel sat at 0.34-0.60
FOREIGN_WINDOW = 8.0   # how much of the run to judge it on
FOREIGN_JOIN_GAP = 1.5 # pieces of one passage this close, in the same language, are
                       # one passage the windows cut a long answer into runs
FOREIGN_BACK_MAX = 3.0 # how far back a passage may reclaim utterances the classifier
                       # buried inside a majority-source run
# FOREIGN_SRC_LOGPROB and SPAN_TAIL_LOGPROB were calibrated empirically against
# the ivrit-ai Hebrew fine-tune (the numbers in the comments below are its
# readings). They are shared across all source ASR models for now and may need
# revisiting per source once other models show different confidence profiles.
# The source ASR's read is only trustworthy as a NEGATIVE witness ("I cannot read
# this"). Asserting the opposite it lies confidently: a Hebrew fine-tune renders
# clear English speech as Hebrew-script transliteration at -0.38, above the fail bar,
# and that reading alone once dubbed over a man speaking English. A model that
# actually reads the target language is the positive witness against it but only
# when it reads well in absolute terms, beats the source model on the same clip, and
# returns a phrase rather than a stock hallucination.
TARGET_READ_LOGPROB = -0.35  # target ASR read that counts as "clean target speech"
TARGET_READ_MARGIN = 0.10    # ...and it must beat the source ASR's read by this much
TARGET_READ_WORDS = 3        # ...over at least this many words
FOREIGN_SRC_LOGPROB = -0.5  # ...and the source ASR must FAIL to read it this well. Its
                       # confidence is the honest witness: on this video it read real
                       # Arabic at -0.64 as garbled non-words, but read the stretches
                       # the classifier mislabelled `mi` and `nl` at -0.38/-0.34 as
                       # clean Hebrew. Without this the label alone airs the narrator.
# An UNNAMED foreign verdict ("und") rests on a single negative witness: the source
# ASR could not read this passage. That premise is checkable, and it was never
# checked. The main source-language pass — a whole-file decode, far better
# conditioned than the isolated re-decode `_judge_span` scores — writes its own
# words over the same seconds, with a per-word probability. When it read the passage
# confidently, "no ASR here reads this" is simply false, and keeping the span throws
# a correct transcript away and airs the source language undubbed. A NAMED verdict is
# untouched: the classifier saying "Arabic, p=0.96" is a second witness, and the
# Hebrew model transliterating that Arabic into Hebrew script is the documented
# reason the name outranks the read.
UND_SRC_WORDS = 3      # source-script words the main pass must have put in the span...
UND_SRC_PROB = 0.8     # ...at this median word probability, to refute an "und" verdict

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
                          # Hebrew beyond it a keep span there plays the source voice.
# VoxLingua107 reports languages by their old ISO-639 codes; map to ours.
_LID_ALIAS = {"iw": "he", "in": "id", "ji": "yi"}

GAP_MIN_SEC = 0.7      # shortest unheard stretch worth a second ASR pass. Was 1.2;
                       # lowered once GAP_MIN_LOGPROB existed to filter the extra
                       # candidates a 0.8s gap held "אל ג'ולאני", the name of the
                       # episode's subject, spoken clear and lost between segments
GAP_RMS_FLOOR = 0.012  # below this the stretch is silence, not missed speech
                       # Read twice, of two files, for two different questions
                       # see `recover_gaps`. Against the separated vocals it is
                       # the "is anyone speaking here at all" floor, and the two
                       # populations it separates are 1.5+ orders of magnitude
                       # apart: measured across every run in outputs/, the three
                       # sites where the gap pass hallucinated a line over music
                       # read 0.000016-0.0018 RMS on the vocals stem, while real
                       # speech read 0.053-0.071. This floor sits ~7x above the
                       # loudest of the former and ~4x below the quietest of the
                       # latter. A floor scaled off each file's own speech median
                       # was considered and rejected: on a source whose vocals
                       # are quiet throughout it scales *down* into the music,
                       # which is the one failure the check exists to prevent.
GAP_PAD = 0.35
GAP_MIN_LOGPROB = -0.5  # a gap read below this is a hallucination (music sting),
                        # not recovered speech leave the window uncovered instead

# The same question one pass earlier: the MAIN ASR pass reads whatever window it
# is handed, and over music with no speech in it the model does not decline it
# invents. The opening 13.1s of theme music on "אויבים" S5E4 came back as
# " תודה רבה." — a stock pleasantry that cannot be answered with a phrase
# blacklist, because those same two words are real dialogue elsewhere in this
# very series (see _ASR_STOCK: only phrases that are *never* legitimate belong
# there). Nor does Whisper's own no-speech signal help: measured over that
# episode, `no_speech_prob` read 0.0000 on the invented line and 0.0000 on all
# 127 real speech segments, and the invented words carried per-word
# probabilities up to 0.831. The decode's `avg_logprob` is the one reading that
# separates the two populations, with room to spare:
#   real speech, 127 segments across the episode ...  -0.079 … -0.242
#   the invented line over the music intro .........  -0.843
#   music the gap pass already rejects .............  -0.55 … -1.01
# So the floor goes between them, at the value the gap pass was calibrated to
# for the same judgement. Narration over music is unaffected: the same intro
# music under speech reads -0.17, well inside the speech population.
ASR_MIN_LOGPROB = -0.5

# Caption chrome: sound tags and speaker arrows carry no speech.
_CHROME = re.compile(r"[\[\(](?:[^\]\)]{0,40})[\]\)]")
_ARROWS = re.compile(r">>+")
# Bidi and zero-width formatting marks. The Hebrew ASR emits them at direction
# changes; they are invisible, carry no speech, and travel into the translator's
# prompt and the TTS text if left in.
_INVISIBLE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
# Everything that is not a letter/digit, for comparing repeated words regardless
# of the trailing punctuation a looped phrase tends to end on.
_WORD_PUNCT = re.compile(r"[^\w]", re.UNICODE)


def clean_token(raw: str) -> tuple[str, bool]:
    """Return (word, is_speaker_break). Empty word means: drop this token."""
    text = (raw or "").strip()
    brk = bool(_ARROWS.search(text))
    text = _ARROWS.sub(" ", text)
    text = _CHROME.sub(" ", text)
    text = _INVISIBLE.sub("", text)
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


class TranscriptFileError(ValueError):
    """A transcript file the pipeline cannot read, with the reason in the message.

    A plain `ValueError` rather than the `SystemExit` every other refusal in this
    module raises, because the first thing that reads a user-supplied transcript
    is the *server*, at create time (`dubbing_app.app.create_project`), and a
    `SystemExit` there is not a 400 — it is a dead worker. The pipeline turns it
    back into a `SystemExit` at the one place it catches it.
    """


# One line is all an SRT cue and a WebVTT cue have in common, and it is the one
# that matters: `hh:mm:ss,mmm --> hh:mm:ss,mmm` (SRT) or `mm:ss.mmm --> …`
# (WebVTT, where the hours are optional and the decimal separator is a point).
# Everything else in either format — the SRT index line, a VTT cue identifier,
# NOTE and STYLE blocks, the `align:start position:10%` settings trailing the
# timestamps — is chrome around it, so the parser below finds cues by this line
# and treats every other line as either body text or noise.
_TS = r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})"
_CUE = re.compile(rf"^{_TS}\s*-->\s*{_TS}")
# WebVTT markup: `<v Speaker>`, `<i>`, the per-word `<00:00:01.000>` timings, and
# the `{\an8}` positioning some SRT writers borrow from SSA. None of it is speech.
_SUB_TAGS = re.compile(r"<[^>]{0,120}>|\{[^}]{0,60}\}")

SUBTITLE_SUFFIXES = (".srt", ".vtt")
# What `--captions` accepts. Timestamps are the whole contract: this pipeline
# places words in time and has no forced aligner, so a transcript with no clock
# on it (.txt, a pasted paragraph) cannot be turned into one that does.
TRANSCRIPT_SUFFIXES = SUBTITLE_SUFFIXES + (".json3",)


def _cue_seconds(m: re.Match[str], base: int) -> float:
    hours, minutes, seconds, frac = m.group(base + 1, base + 2, base + 3, base + 4)
    return (float(hours or 0) * 3600.0 + float(minutes) * 60.0 + float(seconds)
            + float(frac) / (10.0 ** len(frac)))


def cues_from_subtitles(text: str) -> list[tuple[float, float, str]]:
    """(start, end, text) for every cue in an SRT or WebVTT file.

    Both formats in one reader on purpose: they differ in the decimal separator,
    in whether the hours are written, and in what sits *around* a cue — never in
    what a cue is. A cue body runs from the timestamp line to the next blank line
    (or the next timestamp), which is exactly what both specs say.
    """
    cues: list[tuple[float, float, str]] = []
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    i = 0
    while i < len(lines):
        m = _CUE.match(lines[i].strip())
        if not m:
            i += 1
            continue
        start, end = _cue_seconds(m, 0), _cue_seconds(m, 4)
        body: list[str] = []
        i += 1
        while i < len(lines) and lines[i].strip() and not _CUE.match(lines[i].strip()):
            body.append(lines[i])
            i += 1
        joined = re.sub(r"\s+", " ", _SUB_TAGS.sub(" ", " ".join(body))).strip()
        if joined and end > start:
            cues.append((start, end, joined))
    return cues


def words_from_subtitles(path: Path, *, limit: float | None = None
                         ) -> list[dict[str, Any]]:
    """A subtitle file's cues as the flat word stream every later stage reads.

    A cue is a *line*, not a word, so the words inside it are spread evenly over
    its span. That is a guess, but a bounded one: no word can land outside the
    cue the subtitler timed it into, which is a tighter placement than the fixed
    `segments.WORD_MAX` guess caption words get today, and the segmenter clamps
    every end to the next word's onset anyway.

    Splitting is `script.split_words`, so a Chinese or Japanese cue becomes one
    unit per character rather than one unit per cue with no spaces in it there
    are no word boundaries to split on, and a whole cue at one timestamp would
    place the entire line at its first instant.
    """
    raw = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    out: list[dict[str, Any]] = []
    for start, end, cue in cues_from_subtitles(raw):
        text, brk = clean_token(cue)
        if not text:
            continue
        tokens = script.split_words(text)
        if not tokens:
            continue
        step = max(end - start, 0.05) / len(tokens)
        for k, token in enumerate(tokens):
            t = start + k * step
            if limit is not None and t >= limit:
                return out
            if out and t < out[-1]["t"]:
                t = out[-1]["t"]   # overlapping cues; the stream stays monotonic
            out.append({"t": round(t, 3), "end": round(max(t + 0.02, t + step), 3),
                        "text": token, "brk": brk and k == 0})
    return out


def words_from_file(path: Path, *, limit: float | None = None) -> list[dict[str, Any]]:
    """Read whichever caption/transcript file the run was handed.

    The one door for "words the user (or the fetch) supplied", so the three
    formats are decided in a single place instead of at each caller. Raises
    `TranscriptFileError` for anything it cannot read the caller decides
    whether that is a 400 or a dead run.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix not in TRANSCRIPT_SUFFIXES:
        raise TranscriptFileError(
            f"{p.name}: a transcript has to be {', '.join(TRANSCRIPT_SUFFIXES)}. "
            "Plain text has no timestamps, and nothing here can align it to the audio.")
    if not p.is_file():
        raise TranscriptFileError(f"transcript file not found: {p}")
    try:
        if suffix == ".json3":
            return words_from_json3(p, limit=limit)
        return words_from_subtitles(p, limit=limit)
    except TranscriptFileError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise TranscriptFileError(f"{p.name} could not be read as {suffix}: {exc}") from exc


def check_transcript_file(path: Path) -> int:
    """Parse a supplied transcript and return its word count, or refuse it.

    The check the *server* runs before a project exists. Everything it can say —
    wrong extension, missing file, unparseable, parses but says nothing costs a
    sentence here and costs a fetch, a stems run and a confusing job failure if
    it is left to the transcript stage instead.
    """
    words = words_from_file(Path(path))
    if not words:
        raise TranscriptFileError(
            f"{Path(path).name} parsed, but there are no timed words in it. "
            "An empty subtitle file, or one whose cues carry no text.")
    return len(words)


def _cuda_usable() -> bool:
    """Whether CTranslate2 sees a CUDA device. Never true on darwin."""
    if sys.platform == "darwin":
        return False
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _cudnn_on_path() -> None:
    """Make the nvidia-cudnn wheel's libs findable before CTranslate2 dlopens them.

    The cuDNN that ships alongside torch lives inside the `nvidia.cudnn` package,
    which is not on the default library search path. Preload its libraries and
    prepend the dir to LD_LIBRARY_PATH; harmless no-op when the wheel is absent.

    Windows is the same problem with different spelling: the DLLs sit in
    `nvidia/cudnn/bin`, `LD_LIBRARY_PATH` means nothing, and since 3.8 a DLL
    directory has to be declared with `os.add_dll_directory`. Same intent, so it
    lives here rather than in a second half-copy of this function elsewhere.
    """
    try:
        import ctypes
        import glob
        import os

        import nvidia.cudnn

        # A namespace package: __file__ is None, __path__ holds the real dir.
        pkg_dir = next(iter(nvidia.cudnn.__path__), None)
        if pkg_dir is None:
            return
        if sys.platform == "win32":
            for sub in ("bin", "lib"):
                d = Path(pkg_dir).resolve() / sub
                if d.is_dir():
                    os.add_dll_directory(str(d))
            return
        lib = Path(pkg_dir).resolve() / "lib"
        if not lib.is_dir():
            return
        current = os.environ.get("LD_LIBRARY_PATH", "")
        if str(lib) not in current.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{lib}:{current}" if current else str(lib)
        for so in sorted(glob.glob(str(lib / "libcudnn*.so.*"))):
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
    except Exception:
        pass


def load_whisper(model_path: str, *, label: str = "ASR",
                 cpu_threads: int | None = None):
    """A faster-whisper model on CUDA when that actually works, else CPU.

    CUDA is tried first (float16), and a tiny warm-up transcribe runs before the
    model is trusted CTranslate2 needs cuDNN and its absence surfaces at the
    first forward pass, not at construction. Any failure falls back to CPU with
    the same settings as before (`compute_type="auto"`), so on a Mac or a box
    without CUDA the result is exactly the old behaviour. `cpu_threads` applies
    only to the CPU path.
    """
    from faster_whisper import WhisperModel

    if _cuda_usable():
        try:
            _cudnn_on_path()
            model = WhisperModel(model_path, device="cuda", compute_type="float16")
            import numpy as np

            segs, _info = model.transcribe(np.zeros(1600, dtype=np.float32),
                                           language="en", beam_size=1)
            list(segs)  # force the forward pass; cuDNN failures raise here
            print(f"  {label}: {model_path} (cuda, float16)", file=sys.stderr)
            return model
        except Exception as exc:
            print(f"  {label}: cuda unusable ({exc}) falling back to cpu",
                  file=sys.stderr)
    kwargs: dict[str, Any] = {"cpu_threads": cpu_threads} if cpu_threads else {}
    model = WhisperModel(model_path, device="cpu", compute_type="auto", **kwargs)
    print(f"  {label}: {model_path} (cpu)", file=sys.stderr)
    return model


def source_asr_paths(src: str) -> tuple[Path, str]:
    """(local model dir, hub fallback) for the ASR that reads the source language.

    Hebrew including YouTube's legacy "iw" code keeps the ivrit-ai fine-tune;
    every other source uses the vanilla multilingual large-v3-turbo, which is told
    the language at transcribe time (`language=src`).
    """
    if (src or "").lower() in ("he", "iw"):
        return WHISPER_MODEL, WHISPER_HUB
    return SRC_ASR_MODEL, SRC_ASR_HUB


def load_asr(src: str = "he"):
    local, hub = source_asr_paths(src)
    model_path = str(local) if local.is_dir() else hub
    return load_whisper(model_path, label="transcript: ASR")


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
            # Keep Whisper's measured word end segmentation and placement need
            # real durations, not a guess from onsets alone (see segments.word_ends).
            end = float(w.end) + offset
            # Whisper's own per-word confidence rides along: hallucination analysis
            # needs it, and it costs one field.
            out.append({"t": round(t, 3), "end": round(max(end, t + 0.02), 3),
                        "text": text, "brk": False,
                        "p": round(float(getattr(w, "probability", 1.0) or 1.0), 3)})
    return out


# Stock phrases Whisper injects over music and silence subtitler credits and
# outro thanks it memorised from training data ("Субтитры сделал DimaTorzok" over
# the Arzamas jingle; the classic Amara/"thanks for watching" family). This is a
# fact about the ASR model, not about any one video: the phrases are stereotyped,
# never legitimate dialogue, and always hallucinated.
_ASR_STOCK = re.compile(
    r"субтитры|дима\s*торжок|dimatorzok|редактор\s+субтитров|"
    r"subtitles?\s+by|amara\.org|zeoranger|"
    r"продолжение\s+следует|спасибо\s+за\s+просмотр|"
    r"thanks?\s+for\s+watching|"
    r"подпи(?:шись|сывайтесь)\s+на\s+канал|subscribe\s+to\s+(?:the|my|our)\s+channel|"
    r"اشتركوا?\s+في\s+القناة",
    re.IGNORECASE)


def drop_stock_phrases(words: list[dict[str, Any]], *, window: float = 4.0
                       ) -> list[dict[str, Any]]:
    """Drop a run of words containing a known Whisper stock hallucination.

    The phrase and its immediate companions form one invented line, so the whole
    contiguous run (words within `window` seconds around the match, uninterrupted
    by a gap > 1s) goes, not just the matching token.
    """
    joined, spans = "", []
    for w in words:
        if joined:
            joined += " "
        start = len(joined)
        joined += w["text"]
        spans.append((start, len(joined)))
    bad: set[int] = set()
    hits = [i for m in _ASR_STOCK.finditer(joined)
            for i, (s, e) in enumerate(spans) if e > m.start() and s < m.end()]
    for i in hits:
        bad.add(i)
        for j in range(i - 1, -1, -1):
            if words[j + 1]["t"] - float(words[j].get("end", words[j]["t"])) > 1.0 \
                    or words[i]["t"] - words[j]["t"] > window:
                break
            bad.add(j)
        for j in range(i + 1, len(words)):
            if words[j]["t"] - float(words[j - 1].get("end", words[j - 1]["t"])) > 1.0 \
                    or words[j]["t"] - words[i]["t"] > window:
                break
            bad.add(j)
    if bad:
        gone = " ".join(words[i]["text"] for i in sorted(bad))[:70]
        print(f"  transcript: dropped stock-phrase hallucination: {gone}", file=sys.stderr)
    return [w for i, w in enumerate(words) if i not in bad]


def drop_echo_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop a word decoded twice across a chunk boundary.

    One speaker cannot overlap themselves: an adjacent identical word whose start
    precedes its twin's end ("ומסוכנים." 445.53–446.65 then again 446.55–447.51)
    is the same utterance decoded twice, and the dub then says the word twice.
    Genuine rhetorical repetition never overlaps it has a pause and is kept.
    """
    out: list[dict[str, Any]] = []
    for w in words:
        prev = out[-1] if out else None
        if (prev is not None
                and _WORD_PUNCT.sub("", w["text"]).lower()
                == _WORD_PUNCT.sub("", prev["text"]).lower()
                and _WORD_PUNCT.sub("", w["text"])
                and w["t"] < float(prev.get("end", prev["t"])) - 0.05):
            prev["end"] = max(float(prev.get("end", prev["t"])),
                              float(w.get("end", w["t"])))
            print(f"  transcript: dropped echoed word {w['text']!r} at {w['t']:.2f}s",
                  file=sys.stderr)
            continue
        out.append(w)
    return out


def drop_stretched_words(words: list[dict[str, Any]], *, max_sec: float = 2.5
                         ) -> list[dict[str, Any]]:
    """Drop decode artifacts: a words whose span swallows the words after it.

    Real word timings are monotonic. Across a chunk boundary under music the
    model once emitted "סתם" stretched over 3.1 seconds with six invented filler
    words *inside* its span (at probability up to 0.97 confidence does not
    catch this) before re-reading the real "לא סתם" at the right time. The
    stretched word and everything nested inside its span are one artifact; both
    go. A merely long word with nothing overlapping it is left alone.
    """
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(words):
        w = words[i]
        dur = float(w.get("end", w["t"])) - w["t"]
        nested = [v for v in words[i + 1:] if v["t"] < float(w.get("end", w["t"])) - 0.05]
        if dur > max_sec and nested:
            # The artifact is the stretched word plus the inventions buried well
            # inside its span; the genuine re-read arrives at the span's end and
            # must survive, so only words ending clearly before it are dropped.
            # Words seamlessly attached in front (no gap to the stretched word)
            # are the same decode event the surviving "לא" of a false "לא סתם"
            # became a stranded half-second "No" in the dub.
            span_end = float(w.get("end", w["t"]))
            # Bounded: real speech also runs words tightly together, so at most
            # two words and one second may be reclaimed, and only seamless ones.
            lead = []
            k = len(out) - 1
            edge = w["t"]
            while (k >= 0 and len(lead) < 2 and w["t"] - out[k]["t"] <= 1.0
                   and edge - float(out[k].get("end", out[k]["t"])) < 0.15):
                lead.append(out[k])
                edge = out[k]["t"]
                k -= 1
            out = out[:k + 1]
            dropped = lead + [w] + [v for v in nested
                                    if float(v.get("end", v["t"])) < span_end - 0.25]
            print(f"  transcript: dropped stretched-word artifact at {w['t']:.1f}s "
                  f"({len(dropped)} words: {' '.join(d['text'] for d in dropped)[:60]})",
                  file=sys.stderr)
            keep_ids = {id(v) for v in words[i + 1:]} - {id(d) for d in dropped}
            rest = [v for v in words[i + 1:] if id(v) in keep_ids]
            return out + drop_stretched_words(rest, max_sec=max_sec)
        out.append(w)
        i += 1
    return out


def collapse_repeats(words: list[dict[str, Any]], *, max_ngram: int = 4,
                     min_reps: int = 3) -> list[dict[str, Any]]:
    """Drop Whisper's back-to-back repetition hallucinations.

    On silent or musical stretches the decoder sometimes loops a phrase the
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
MARK_JOIN_GAP = 0.4    # a mark-initial token further than this from the word before it
                       # starts something new; only an adjacent one is a split word


def join_split_marks(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rejoin a word the ASR split at a geresh or a hyphen.

    Hebrew spells foreign sounds with a geresh ג'יהאד, אג'נדה, ג'בהת and Whisper
    emits the geresh as the start of a *new* token: `לג` + `'יהאד`, `אל` + `-קאעידה`.
    The halves reach the translator as separate words and it misreads the grammar
    that hangs on them: "שקוראת לג'יהאד" ("which calls **for** jihad") came back as
    "which she calls jihad", because the preposition ל was stranded on its own token.
    Rejoined, the same model and prompt render it correctly.

    The merged word keeps the first half's onset and the second half's end, so timing
    is unchanged. A token after a sentence end is left alone there the mark starts
    something new rather than continuing a word.
    """
    out: list[dict[str, Any]] = []
    for w in words:
        prev = out[-1] if out else None
        gap = w["t"] - float(prev.get("end", prev["t"])) if prev else 0.0
        if (prev and _SPLIT_MARK.match(w.get("text") or "")
                and (prev.get("text") or "") and not prev["text"].endswith((".", "!", "?"))
                and not w.get("brk") and gap <= MARK_JOIN_GAP):
            prev["text"] += w["text"]
            if w.get("end") is not None:
                prev["end"] = w["end"]
            continue
        out.append(dict(w))
    return out


def speech_only(segments: Iterable[Any], *,
                min_logprob: float = ASR_MIN_LOGPROB) -> Iterator[Any]:
    """Yield the decoded segments that read like speech, dropping the rest.

    `vad_filter=True` is not this gate: Silero passes the music intro as speech,
    and the model then invents a line to fill it. The reading is what tells them
    apart (see ASR_MIN_LOGPROB for the numbers).

    A reading is the decode chunk's, shared by every segment Whisper splits out
    of it, so rejecting one rejects the chunk and that is the intent: a chunk
    that reads this badly is not one whose confident-looking sub-segments can be
    believed. Lazy on purpose, so `_words_of` can still stop at its limit
    without forcing the rest of the file through the decoder.
    """
    for seg in segments:
        lp = getattr(seg, "avg_logprob", None)
        if lp is not None and float(lp) < min_logprob:
            text = (getattr(seg, "text", "") or "").strip()[:60]
            print(f"  transcript: {float(getattr(seg, 'start', 0.0)):.1f}-"
                  f"{float(getattr(seg, 'end', 0.0)):.1f}s read at logprob "
                  f"{float(lp):.2f} discarded as non-speech: {text}", file=sys.stderr)
            continue
        yield seg


def words_from_whisper(model, source_wav: Path, lang: str, *,
                       limit: float | None = None) -> list[dict[str, Any]]:
    segments, _info = model.transcribe(
        str(source_wav), language=lang, beam_size=5, word_timestamps=True,
        condition_on_previous_text=False, vad_filter=True,
    )
    return _words_of(speech_only(segments), limit=limit)


def load_target_asr(tgt: str = "en"):
    """An ASR that reads the target language, or None when its model is absent.

    English keeps the dedicated `.en` model (unchanged behaviour); every other
    target uses the multilingual base model with `language=tgt` at transcribe time.
    """
    if (tgt or "en") == "en":
        if not EN_ASR_MODEL.is_dir():
            return None
        return load_whisper(str(EN_ASR_MODEL), label="transcript: EN ASR")
    if not TARGET_ASR_MODEL.is_dir():
        return None
    return load_whisper(str(TARGET_ASR_MODEL), label=f"transcript: {tgt} ASR")


def load_vad():
    from silero_vad import load_silero_vad

    return load_silero_vad()


def vad_regions(vad, source_wav: Path, *, sr: int = 16000,
                merge_gap: float = VAD_MERGE_GAP) -> list[tuple[float, float]]:
    """Speech start/end from Silero VAD, joined across brief pauses.

    A tiny model (~a millisecond of compute per chunk) that gives real speech
    boundaries far cheaper than transcribing a region just to find where it
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

    One embedding forward pass, not a transcription this is the cheap way to ask
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
    long monologue that switches language partway English narration into a Hebrew
    interview is caught rather than labelled by its majority. Adjacent windows of
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


def _extend_english_end(en_model, source_wav: Path, b: float, limit: float,
                        target: str = "en") -> float:
    """Grow an English span end while the English model still reads English past it.

    Used to reclaim a trailing word ("...children") the coarse LID window placed on
    the Hebrew side. The English-only model decodes real English with high
    avg_logprob and Hebrew as low-confidence gibberish, so the logprob is the stop.
    """
    from . import audio

    while b + 0.25 < limit:
        end = min(b + SPAN_TAIL_STEP, limit)
        segs, _info = en_model.transcribe(audio.decode_mono(source_wav, 16000, start=b, end=end),
                                          language=target, beam_size=5, vad_filter=False)
        segs = list(segs)
        if not segs:
            break
        lp = sum(s.avg_logprob for s in segs) / len(segs)
        text = " ".join(s.text.strip() for s in segs).strip()
        if lp < SPAN_TAIL_LOGPROB or len(text) < 2:
            break
        b = end
    return round(b, 3)


def _voice_pauses(source_wav: Path, start: float, limit: float) -> list[float]:
    """Where the voice stops between `start` and `limit`, in order.

    A pause needs `PAUSE_FRAMES` consecutive quiet frames, so the closure of a plosive
    inside a word is not mistaken for the end of a sentence.
    """
    from . import audio

    if limit - start <= PAUSE_HOP:
        return []
    levels = audio.frame_rms(audio.decode_mono(source_wav, 16000, start=start, end=limit),
                             16000, PAUSE_HOP)
    out: list[float] = []
    quiet = 0
    for k, level in enumerate(levels):
        if level < PAUSE_FLOOR:
            quiet += 1
            if quiet == PAUSE_FRAMES:
                out.append(round(start + (k - quiet + 1) * PAUSE_HOP, 3))
        else:
            quiet = 0
    return out


def _voice_pause_after(source_wav: Path, start: float, limit: float) -> float | None:
    """When the voice next stops, or None if it does not stop before `limit`."""
    found = _voice_pauses(source_wav, start, limit)
    return found[0] if found else None


def _utterance_start_before(source_wav: Path, end: float, limit: float) -> float | None:
    """The pause that opens the utterance running into `end`, or None.

    Not simply the last pause: `end` itself often sits in one. This skips any pause
    within `PAUSE_HOP * PAUSE_FRAMES * 2` of `end` and takes the one before that, which
    is where the phrase now being spoken began.
    """
    edge = end - PAUSE_HOP * PAUSE_FRAMES * 2
    earlier = [p for p in _voice_pauses(source_wav, limit, end) if p <= edge]
    return earlier[-1] if earlier else None


def _reclaim_leading_fragment(en_model, source_wav: Path, cand: float, a: float,
                              b: float, target: str = "en") -> float:
    """Take back the fragment VAD broke off the front of an utterance, if it is speech.

    The counterpart of `_extend_english_end`, for the same failure at the other edge:
    VAD splits a soft first word into a region of its own ("Frankly," at 270.80-271.70,
    then "I had the same concerns…" from 272.00), and the language classifier cannot
    judge a fragment that short it labelled that one `mi`, Maori, confidently. So the
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
        language=target, beam_size=5, vad_filter=False, word_timestamps=True)
    segs = list(segs)
    if not segs:
        return round(a, 3)
    lp = sum(s.avg_logprob for s in segs) / len(segs)
    starts = [float(w.start) for s in segs for w in (s.words or []) if (w.word or "").strip()]
    if lp < SPAN_TAIL_LOGPROB or not starts or cand + min(starts) >= a - 0.05:
        return round(a, 3)
    return round(cand, 3)


def _foreign_group(lsegs: list[tuple[float, float, str | None]], i: int
                   ) -> tuple[float, float, set[int]]:
    """The whole extent of a passage the `win`-second windows broke into pieces.

    A twenty-second answer in Arabic does not arrive as one run: the classifier called
    421.10-425.10 and 426.20-446.20 Arabic with an unlabelled sliver between them, and
    judged alone the first piece missed the confidence bar and was dubbed from the
    source ASR's gibberish ("The notes from Gaza are being sent to the Philippines").
    Joined, the passage speaks for itself. Only runs of the *same* language join, and
    only across a breath, so this never reaches over a run in another language.
    """
    lang = lsegs[i][2]
    a, b = lsegs[i][0], lsegs[i][1]
    used = {i}
    for step, edge in ((1, "forward"), (-1, "back")):
        j = i + step
        while 0 <= j < len(lsegs):
            start, end, other = lsegs[j]
            if (start - b if edge == "forward" else a - end) > FOREIGN_JOIN_GAP:
                break
            if other == lang:
                if edge == "forward":
                    b = end
                else:
                    a = start
                used.add(j)
            elif not (other is None and end - start <= LID_SHORT):
                break                      # a labelled neighbour ends the passage
            j += step
    return a, b, used


def _reads_as_source(lid, src_model, source_wav: Path, a: float, b: float,
                     source: str) -> bool:
    """Whether this stretch is the source language, by both witnesses."""
    from . import audio

    clip = audio.decode_mono(source_wav, 16000, start=a, end=b)
    lang, prob = detect_language(lid, clip)
    if lang == source and prob >= LID_MIN_PROB:
        return True
    segs, _info = src_model.transcribe(clip, language=source, beam_size=5, vad_filter=False)
    segs = list(segs)
    return not segs or sum(s.avg_logprob for s in segs) / len(segs) >= FOREIGN_SRC_LOGPROB


def _extend_foreign_start(lid, src_model, source_wav: Path, a: float, source: str,
                          floor: float) -> float:
    """Carry a passage back over utterances that are not the source language either.

    A foreign clip can begin in the middle of a run the classifier calls the source
    language by majority a twenty-second Hebrew run whose last two seconds are the
    start of a Chinese report. Those seconds are then transcribed as Hebrew gibberish
    and dubbed: "-כן, תודה רבה, זה פרלטינגרס" became "Yes, thank you very much, this is
    Pirlinger" over the top of the clip. Walking back one utterance at a time, and
    stopping the moment either witness says source language, finds the real edge the
    Hebrew before it reads at -0.21 where the clip reads at -0.79.
    """
    limit = max(floor, a - FOREIGN_BACK_MAX)
    while a - limit > PAUSE_HOP:
        prev = _utterance_start_before(source_wav, a, limit)
        if prev is None or prev >= a - PAUSE_HOP:
            break
        if _reads_as_source(lid, src_model, source_wav, prev, a, source):
            break
        a = prev
    return round(a, 3)


_SPAN_ASR = None


def _read_foreign_span(source_wav: Path, a: float, b: float, lang: str
                       ) -> tuple[str, list[dict[str, Any]]]:
    """Transcribe a confirmed third-language span for its subtitle.

    The vanilla multilingual turbo (the non-Hebrew source-ASR model) reads ~100
    languages; told the confirmed language or left to detect when the verdict
    was "und" it supplies the words the subtitle track needs. Loaded lazily
    and cached: most runs have no third-language span at all. On any failure the
    span falls back to the old "…" placeholder and nothing downstream changes.
    """
    global _SPAN_ASR
    placeholder = "…", [{"t": round(a, 3), "text": "…"}]
    try:
        if _SPAN_ASR is None:
            _SPAN_ASR = load_whisper(
                str(SRC_ASR_MODEL) if SRC_ASR_MODEL.is_dir() else SRC_ASR_HUB,
                label="span-asr")
        from . import audio

        clip = audio.decode_mono(source_wav, 16000, start=a, end=b)
        segs, _ = _SPAN_ASR.transcribe(
            clip, language=None if lang == "und" else lang,
            word_timestamps=True, condition_on_previous_text=False)
        segs = list(segs)
        # Chanting/music makes the multilingual model invent stock phrases
        # ("اشتركوا في القناة" "subscribe to the channel"); a low-confidence
        # read is that, not the passage's words. Lenient floor: foreign reads
        # legitimately score lower than source-language ones.
        if segs and sum(s.avg_logprob for s in segs) / len(segs) < -0.8:
            return placeholder
        words = [{"t": round(a + w.start, 3), "text": w.word.strip()}
                 for s in segs for w in (s.words or []) if (w.word or "").strip()]
    except Exception as exc:
        print(f"  transcript: span ASR failed ({exc}); span kept without text",
              file=sys.stderr)
        return placeholder
    if not words:
        return placeholder
    return " ".join(w["text"] for w in words), words


def _reads_as_target(tgt_model, clip, target: str, src_lp: float | None) -> bool:
    """Whether the target-language ASR reads this clip as clean target speech.

    The one witness that can contradict a confident source-language read, since it
    is a different model with the opposite bias: on source-language audio it decodes
    low-confidence gibberish, on real target speech it decodes a fluent phrase. Three
    conditions together, because each alone has a known failure mode an absolute
    floor (a garbage read is never clean), a margin over the source model's read of
    the *same* clip (whoever reads it better is right, and a tie is no evidence), and
    a phrase of real words that is not one of Whisper's memorised stock lines
    ("Thanks for watching" over music scores near zero and means nothing).
    """
    if tgt_model is None or not target:
        return False
    try:
        segs, _info = tgt_model.transcribe(clip, language=target, beam_size=5,
                                           vad_filter=False)
        segs = list(segs)
    except Exception as exc:                                    # pragma: no cover
        print(f"  transcript: target-ASR witness failed ({exc})", file=sys.stderr)
        return False
    if not segs:
        return False
    lp = sum(s.avg_logprob for s in segs) / len(segs)
    if lp < TARGET_READ_LOGPROB:
        return False
    if src_lp is not None and lp < src_lp + TARGET_READ_MARGIN:
        return False
    text = " ".join(s.text.strip() for s in segs).strip()
    if _ASR_STOCK.search(text):
        return False
    words = [w for w in text.split() if any(ch.isalpha() for ch in w)]
    return len(words) >= TARGET_READ_WORDS and script.is_script(text, target)


def _sounds_foreign(lid, src_model, source_wav: Path, a: float, b: float,
                    source: str, tgt_model=None, target: str = "") -> str | None:
    """A name for a passage that is not the source language, or None to dub it.

    The deciding witness is the source-language ASR, because it is the one that can be
    wrong in a way we can measure: it read real Arabic at avg_logprob -0.64 as garbled
    non-words, and read the stretches the classifier mislabelled `mi` and `nl` at
    -0.38/-0.34 as clean Hebrew. So it must FAIL here, always.

    The classifier's job is to name the language and to veto it is asked over the
    whole passage, since its `win`-second windows are unreliable on short ones (a 0.9s
    fragment came back Maori at p≥0.6). When it is confident, its name is used; when it
    has no opinion at all the passage is still kept, unnamed, on the ASR's word alone —
    a Chinese news clip came back `vi` 0.43, `tr` 0.34, `nn` 0.09 and is plainly not
    Hebrew. What it may not do is contradict itself: if it says the source language
    confidently, that outranks the ASR and the passage is dubbed.

    `tgt_model` an ASR that actually reads the target language is the positive
    witness against a source read that is confidently wrong. When it speaks, the
    verdict is the target language itself and the caller treats the passage as
    target-language speech (subtitle text, refined edges, a target keep) rather than
    as an unnamed foreign one.
    """

    if b - a < FOREIGN_MIN_SEC or src_model is None:
        return None
    # Judge a long passage by its interior: the edges inherit a word or two of
    # source-language bleed from the neighbouring segment, and one clean Hebrew
    # fragment at the rim ("מלחמת האזרחים?" at -0.38) once vetoed 10.6 seconds of
    # an English interviewee, which then played neither dubbed nor subtitled.
    trim = min(1.0, (b - a) / 4) if b - a >= 4.0 else 0.0
    verdict = _judge_span(lid, src_model, source_wav, a + trim, b - trim, source,
                          tgt_model, target)
    if verdict != "und" or b - a >= 6.0:
        return verdict
    # An unnamed verdict on a short span is the least trustworthy combination:
    # both witnesses are unreliable there. A 2.9s music sting under narration got
    # LID "my"@0.34 and made the Hebrew ASR hallucinate "תודה רבה" four times at
    # logprob -0.55 three seconds of the source language then played undubbed.
    # Judged once more with ±2s of context the same audio is he@0.95 with clean
    # ASR. A *named* foreign language (the real Arabic quote, ar@0.96) never
    # reaches this retry.
    widened = _judge_span(lid, src_model, source_wav, max(0.0, a - 2.0), b + 2.0, source,
                          tgt_model, target)
    return verdict if widened == "und" else widened


def _judge_span(lid, src_model, source_wav: Path, a: float, b: float,
                source: str, tgt_model=None, target: str = "") -> str | None:
    from . import audio

    clip = audio.decode_mono(source_wav, 16000, start=a, end=min(b, a + FOREIGN_WINDOW))
    lang, prob = detect_language(lid, clip)
    if lang == source and prob >= LID_MIN_PROB:
        return None                                   # the classifier vetoes
    if lang and lang != source and prob >= FOREIGN_SURE_PROB:
        return lang                # near-certain foreign label outranks the ASR read
    segs, _info = src_model.transcribe(clip, language=source, beam_size=5, vad_filter=False)
    segs = list(segs)
    src_lp = (sum(s.avg_logprob for s in segs) / len(segs)) if segs else None
    # Before the source read is believed either way, ask the model that reads the
    # target language. This is the only witness that can overturn a confident lie
    # ("I read this English as Hebrew, at -0.38"), and it is what turns a muted
    # target-language speaker into a real, visible target span. It is asked only
    # where the classifier has NOT named a different language confidently: a
    # target-forced decoder always returns target-language text, so it cannot be
    # allowed to rename a passage VoxLingua has already identified as Arabic.
    named = bool(lang) and lang != source and prob >= FOREIGN_MIN_PROB
    if (not named or lang == target) and _reads_as_target(tgt_model, clip, target, src_lp):
        return target
    if src_lp is None or src_lp >= FOREIGN_SRC_LOGPROB:
        return None                                   # it reads as the source language
    return lang if named else "und"


def reads_as_source(words: list[dict[str, Any]] | None, a: float, b: float,
                    source: str) -> bool:
    """Did the main source-language pass already read [a, b] as confident source speech?

    Pure the witness against an unnamed ("und") foreign verdict, whose whole
    premise is that no ASR here can read the passage. `words` is the main pass's
    output (`t`, `text`, and the per-word probability `p`); a span it filled with
    real source-script words at a high median probability is a span it read, and an
    isolated re-decode scoring a little under the failure bar does not overrule that.

    Deliberately blind to a named verdict: this only ever answers about "und".
    """
    if not words or not source:
        return False
    inside = [w for w in words if a - 0.05 <= float(w.get("t", 0.0)) <= b + 0.05]
    real = [w for w in inside if script.is_script(w.get("text") or "", source)]
    if len(real) < UND_SRC_WORDS:
        return False
    probs = sorted(float(w.get("p", 0.0)) for w in real)
    mid = len(probs) // 2
    median = probs[mid] if len(probs) % 2 else (probs[mid - 1] + probs[mid]) / 2
    return median >= UND_SRC_PROB


def detect_spoken_target_spans(en_model, vad, lid, source_wav: Path, total: float,
                               target: str, *, source: str = "", src_model=None,
                               known: list[tuple[float, float]] | None = None,
                               lsegs: list[tuple[float, float, str | None]] | None = None,
                               src_words: list[dict[str, Any]] | None = None
                               ) -> list[dict[str, Any]]:
    """Speech regions not spoken in the source language, as original-audio spans.

    Silero VAD gives the precise start/end of every utterance; VoxLingua107 says
    which language each window is. A run the classifier calls anything other than the
    source language is kept as original audio, because the source-tuned ASR renders it
    as gibberish and dubbing that is worse than not dubbing it at all.

    The target language gets the fuller treatment: an ASR that actually reads it
    supplies the subtitle text and refines both edges. A third language Arabic in a
    Hebrew documentary is kept as-is with no subtitle, since no model here can read
    it, and it has to clear a higher confidence bar for the same reason.
    """
    from . import audio

    known = known or []
    # The caller may hand in the runs it already computed (they are also persisted
    # for the editor app), so the classifier is not paid for twice.
    lsegs = language_segments(vad, lid, source_wav) if lsegs is None else lsegs
    spans: list[dict[str, Any]] = []
    consumed: set[int] = set()
    for i, (a, b, lang) in enumerate(lsegs):
        if lang == source or i in consumed:
            continue
        if lang != target:
            # Judge a third-language passage whole; its pieces are not decidable alone.
            a, b, used = _foreign_group(lsegs, i)
            consumed |= used
        if b - a < VAD_MIN_SEC:
            continue
        if any(a < kb and ka < b for ka, kb in known):
            continue
        if lang != target:
            # The span is the classifier's word alone confirmed over the whole
            # run. Its subtitle text comes from the vanilla multilingual ASR below.
            confirmed = _sounds_foreign(lid, src_model, source_wav, a, b, source,
                                        tgt_model=en_model, target=target)
            if confirmed is None:
                continue
            if confirmed == "und" and reads_as_source(src_words, a, b, source):
                # Nobody named a language, and the source pass read these seconds
                # confidently in the source script: the "no ASR reads this" premise
                # the unnamed keep stands on is false. Dub it, and let the words the
                # main pass already wrote be the transcript (see `reads_as_source`).
                print(f"  transcript: {a:.2f}-{b:.2f}s unnamed, but the source ASR "
                      "read it confidently dubbing it", file=sys.stderr)
                continue
        if lang != target and confirmed != target:
            # A passage that ends where nobody paused ended at a window edge, not at a
            # change of speaker: the Arabic run stopped at 446.20, exactly five windows
            # in, with the voice still going to 447.30, and that last second was dubbed
            # as "And they will convert and they will be crushed." A language does not
            # change mid-utterance, so carry the span to the next real pause. Where the
            # boundary is already a pause as it was at both ends of the passage the
            # scan returns it unchanged and nothing moves.
            b = _voice_pause_after(source_wav, b, min(b + SPAN_TAIL_MAX, total)) or b
            # And the same at the leading edge, where a clip can start inside a run the
            # classifier calls the source language by majority.
            a = _extend_foreign_start(lid, src_model, source_wav, a, source,
                                      spans[-1]["end"] if spans else 0.0)
            # Read the passage with the vanilla multilingual ASR so the viewer at
            # least gets a subtitle (the source show burns its own over these).
            # The audio still plays original the text is subtitle-only, and the
            # translate stage renders it into the target language.
            text, span_words = _read_foreign_span(source_wav, a, min(total, b), confirmed)
            spans.append({"start": round(a, 3), "end": round(min(total, b), 3),
                          "lang": confirmed, "text": text, "words": span_words})
            continue
        # The coarse LID window can put a short trailing word ("...children") on the
        # Hebrew side; the English-only model, by contrast, reads it as confident
        # English. Step the end forward while it keeps reading English, so the
        # speaker finishes bounded so it can't run into the real Hebrew.
        # The next run is the one that starts after this passage ends not simply
        # `i + 1`, because a passage confirmed as the target language may have been
        # grouped out of several runs.
        nxt = next((s for s, _e, _l in lsegs if s >= b - 1e-6), total)
        b = _extend_english_end(en_model, source_wav, b,
                                min(b + SPAN_TAIL_MAX, nxt + 0.6, total), target)
        # Same at the leading edge, when the run just before is a fragment VAD broke
        # off this one near enough to be the same breath, and not itself English
        # (that would already be a span of its own).
        prev = next(((s, e, lg) for s, e, lg in reversed(lsegs) if e <= a + 1e-6), None)
        if prev and prev[2] != target and a - prev[1] <= VAD_MERGE_GAP:
            a = _reclaim_leading_fragment(en_model, source_wav, prev[0], a, b, target)
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
            # it never claims the Hebrew tail beyond a keep span there plays the
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
            # when no pause is in reach with the speakers back to back, stopping
            # early is the one that does not put the source voice on air.
            last = a + float(got[-1].end)
            stop = _voice_pause_after(source_wav, last,
                                      min(b + SPAN_END_PAD, last + SPAN_TAIL_MAX, total))
            b = stop if stop is not None else min(b, round(last + SPAN_END_PAD, 3))
        span_words = ([{"t": round(a + float(w.start), 3), "text": w.word.strip()} for w in got]
                      or [{"t": round(a, 3), "text": tok} for tok in text.split()]
                      or [{"t": round(a, 3), "text": "…"}])
        spans.append({"start": round(a, 3), "end": round(min(total, b), 3),
                      "lang": target, "text": text or "…", "words": span_words})
        print(f"  transcript: {target}-spoken {a:.1f}-{b:.1f}s: {text[:60]}",
              file=sys.stderr)
    return spans


def uncovered_windows(words: list[dict[str, Any]], levels, hop: float, total: float,
                      *, min_gap: float = GAP_MIN_SEC, floor: float = GAP_RMS_FLOOR,
                      known: list[tuple[float, float]] | None = None
                      ) -> list[tuple[float, float]]:
    """Audible stretches the transcript says nothing about.

    `known` marks ranges already accounted for by another source, so they are not
    transcribed a second time two overlapping versions of the same speech would
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


def text_is_target(text: str, src: str, tgt: str) -> bool:
    """Whether transcribed text is already written in the target language's script.

    A cross-script signal only: when the pair shares a script (en→es) letters
    prove nothing, so this is always False and the LID path decides downstream.
    """
    return not script.same_script(src, tgt) and script.is_script(text, tgt)


def window_rms(levels, a: float, b: float, hop: float = 0.1) -> float:
    """The median frame energy over [a, b) of a `frame_rms` array.

    Exactly the statistic and exactly the slice `uncovered_windows` judges a
    window's audibility by, recovered from the window's times: `a` is always
    `i * hop`, and `b` is `j * hop` clipped to the file's end, so ceiling `b`
    lands back on `j`. Written this way on purpose the vocals gate in
    `recover_gaps` reads the same slice of the same kind of array, so on a run
    with no stems (where it reads the mix a second time) it is asking the
    question the window has already answered, and cannot answer it differently.
    """
    import math

    import numpy as np

    i0 = max(0, int(a / hop + 1e-9))
    i1 = min(len(levels), max(i0 + 1, math.ceil(b / hop - 1e-9)))
    if i1 <= i0:
        return 0.0
    return float(np.median(levels[i0:i1]))


def recover_gaps(model, source_wav: Path, words: list[dict[str, Any]], lang: str,
                 total: float, known: list[tuple[float, float]] | None = None,
                 tgt_lang: str = "en", listen_wav: Path | None = None
                 ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Transcribe again, in isolation, wherever audible speech went unheard.

    A single pass over a whole film skips things most often a passage in
    another language, which Whisper drops rather than transcribes. Those seconds
    would otherwise have no segment at all, and a segment is the only way audio
    of any kind reaches the output. Recovered target-language text becomes a kept
    segment; recovered source-language text gets dubbed like anything else.

    Two different questions, two different files. "Is something audible here"
    is asked of `source_wav`, the mix, because that is what the viewer will
    hear over an uncovered span. "What is being said" is asked of `listen_wav`,
    the separated vocals, same as the main pass: this decoder used to re-listen
    to the mix, and the mix's music is exactly what made it invent lines over
    stings the main pass had correctly declined. A window whose vocals hold no
    speech recovers nothing and stays uncovered which the report already
    flags as `uncovered_audible` instead of becoming a hallucinated segment.

    Audible in the mix is therefore not enough to *earn* a decode, only to
    nominate one. A music-only sting is loud (measured 0.037-0.052 RMS, well
    over GAP_RMS_FLOOR) while its separated vocals are digitally silent
    (0.000016-0.0018), and Whisper handed near-silence is precisely what emits
    "thank you very much". So the window is nominated by the mix and confirmed
    by the vocals, and both readings are said out loud when it is refused.
    """
    from . import audio

    listen = listen_wav if listen_wav is not None and listen_wav.is_file() else source_wav
    levels = audio.frame_rms(audio.decode_mono(source_wav, 16000), 16000, 0.1)
    # The same measurement over the file the decoder will actually hear. Memoized
    # rather than branched: with no stems `listen` IS `source_wav`, so this is the
    # identical array, the gate below reads the identical slice of it, and the
    # no-stems path is today's behaviour by construction rather than by promise.
    heard = levels if listen == source_wav else audio.frame_rms(
        audio.decode_mono(listen, 16000), 16000, 0.1)
    windows = uncovered_windows(words, levels, 0.1, total, known=known)
    if not windows:
        return words, []
    found: list[dict[str, Any]] = []
    spans: list[dict[str, Any]] = []
    for a, b in windows:
        voice = window_rms(heard, a, b)
        if voice < GAP_RMS_FLOOR:
            print(f"  transcript: gap {a:.1f}-{b:.1f}s is music, not speech "
                  f"(vocals RMS {voice:.6f} against mix {window_rms(levels, a, b):.3f}) "
                  f"left uncovered", file=sys.stderr)
            continue
        clip = audio.decode_mono(listen, 16000, start=max(0.0, a - GAP_PAD),
                                 end=min(total, b + GAP_PAD))
        try:
            segs, _ = model.transcribe(clip, language=lang, beam_size=5,
                                       word_timestamps=True,
                                       condition_on_previous_text=False, vad_filter=True)
        except Exception as exc:
            print(f"  transcript: gap {a:.1f}-{b:.1f}s failed ({exc})", file=sys.stderr)
            continue
        # Per-segment first, with the main pass's own gate: without this, a
        # segment the main pass just declined as music re-enters here whenever
        # its neighbours in the same window read well enough to carry the
        # *mean*, and averaging across segments is the weaker test of the two.
        segs = list(speech_only(segs))
        # A low-confidence read of a gap is a hallucination, not recovered speech:
        # a music sting made the model invent a whole line ("עוד לא עבר יום…") that
        # was then dubbed as nonsense. Declining leaves the window uncovered, so
        # the original audio for a sting, the music itself plays, which is right.
        lp = (sum(s.avg_logprob for s in segs) / len(segs)) if segs else 0.0
        if segs and lp < GAP_MIN_LOGPROB:
            print(f"  transcript: gap {a:.1f}-{b:.1f}s read at logprob {lp:.2f} "
                  f"discarded as hallucination", file=sys.stderr)
            continue
        got = [w for w in _words_of(segs, offset=max(0.0, a - GAP_PAD))
               if a - 0.15 <= w["t"] <= b + 0.15]
        if not got:
            continue
        text = " ".join(w["text"] for w in got)
        print(f"  transcript: recovered {len(got)} words at {a:.1f}-{b:.1f}s: {text[:60]}",
              file=sys.stderr)
        if text_is_target(text, lang, tgt_lang):
            # Already in the target language, so it will play as original audio and
            # the word timings do not need to be trusted only the window does.
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

    Two sources can describe the same speech the captions and the gap-recovery
    pass and keeping both would place the same words twice, one after the other.
    """
    out: list[dict[str, Any]] = []
    for s in sorted(spans, key=lambda x: (x["start"], -(x["end"] - x["start"]))):
        if out and s["start"] < out[-1]["end"] - 0.05:
            out[-1]["end"] = max(out[-1]["end"], s["end"])
            continue
        out.append(dict(s))
    return out


def foreign_spans(words: list[dict[str, Any]], *, min_sec: float = 0.8,
                  join_gap: float = 2.0, src: str = "he",
                  tgt: str = "en") -> list[dict[str, Any]]:
    """Stretches of caption text already written in the target script.

    Those passages are spoken in the target language, so they should play as
    original audio rather than being "translated" from a phonetic transcription
    of themselves. Only meaningful for a cross-script pair when source and
    target share a script the captions cannot mark such spans, and none return.
    """
    runs: list[list[dict[str, Any]]] = []
    for w in words:
        if not text_is_target(w["text"], src, tgt):
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


def _caption_words(raw: str | None, *, has_captions: bool, limit: float | None,
                   prefer: str, supplied: bool) -> list[dict[str, Any]]:
    """The caption track's words, or none.

    A file the user handed us is fatal when it will not parse; auto-captions the
    fetch downloaded are optional, and a broken one falls back to the ASR like an
    absent one.
    """
    if not has_captions:
        return []
    try:
        return words_from_file(Path(raw), limit=limit)
    except TranscriptFileError as exc:
        if prefer == "file" or supplied:
            raise SystemExit(str(exc)) from exc
        print(f"  transcript: ignoring the downloaded captions ({exc})", file=sys.stderr)
        return []


def _refuse_unusable_captions(prefer: str, raw: str | None, *, has_captions: bool,
                              caption_words: list[dict[str, Any]]) -> None:
    """Fail now, by name, when the transcript the user asked for is not there."""
    if prefer == "file" and not has_captions:
        raise SystemExit("--transcript file was requested but no --captions file was given")
    if prefer == "file" and not caption_words:
        raise SystemExit(f"--transcript file was requested but {raw} holds no timed words")
    if prefer == "captions" and not caption_words:
        raise SystemExit("--transcript captions was requested but no caption file is available")


def _asr_transcript(m: dict[str, Any], workdir: Path, *, src_lang: str, tgt_lang: str,
                    limit: float | None, caption_words: list[dict[str, Any]]) -> tuple[
                        list[dict[str, Any]], list[dict[str, Any]],
                        list[dict[str, Any]], list[dict[str, Any]]]:
    """Transcribe locally. Returns (words, recovered spans, target spans, lang runs)."""
    source_wav = workdir / m["files"]["source_wav"]
    vocals = workdir / m["files"].get("vocals", "")
    model = load_asr(src_lang)
    # Transcribe the isolated voice; judge speech presence from the mix.
    words = words_from_whisper(model, vocals if vocals.is_file() else source_wav,
                               src_lang, limit=limit)
    caption_spans = (foreign_spans(caption_words, src=src_lang, tgt=tgt_lang)
                     if caption_words else [])
    duration = float(limit or m["source"]["duration"])
    words, recovered = recover_gaps(
        model, source_wav, words, src_lang, duration,
        known=[(s["start"], s["end"]) for s in caption_spans],
        tgt_lang=tgt_lang,
        listen_wav=vocals if vocals.is_file() else None)
    words = join_split_marks(drop_stock_phrases(drop_stretched_words(
        drop_echo_words(collapse_repeats(words)))))
    # Real target-language speech the source model rendered as gibberish:
    # Silero VAD finds the utterances, VoxLingua107 says which are English,
    # and those are kept as original audio instead of dubbed from nonsense.
    en_model = load_target_asr(tgt_lang)
    vad = load_vad()
    lid = load_lid()
    en_spans: list[dict[str, Any]] = []
    lang_runs: list[dict[str, Any]] = []
    if en_model is not None and vad is not None and lid is not None:
        lid_wav = vocals if vocals.is_file() else source_wav
        # Computed once and kept: the spans below are only the runs that
        # are NOT the source language, but the editor app wants the whole
        # picture including "this run is Hebrew, confidently" so it can
        # suggest passthrough on a segment the automatic rules left dubbed.
        runs = language_segments(vad, lid, lid_wav)
        lang_runs = [{"start": a, "end": b, "lang": lang or ""} for a, b, lang in runs]
        en_spans = detect_spoken_target_spans(
            en_model, vad, lid, lid_wav, duration, tgt_lang,
            source=src_lang, src_model=model, lsegs=runs, src_words=words)
    return words, recovered, en_spans, lang_runs


def _all_foreign_spans(caption_words: list[dict[str, Any]], recovered: list[dict[str, Any]],
                       en_spans: list[dict[str, Any]], src_lang: str,
                       tgt_lang: str) -> list[dict[str, Any]]:
    """Merge the target-language spans from every witness, best witness first.

    VAD+LID target spans are authoritative their boundaries are precise, so a
    caption or gap-recovery span (coarser, source-model or caption derived) is kept
    only where VAD/LID found no target-language speech, never overriding it.
    """
    others = ((foreign_spans(caption_words, src=src_lang, tgt=tgt_lang)
               if caption_words else []) + recovered)
    others = [s for s in others
              if not any(s["start"] < e["end"] and e["start"] < s["end"] for e in en_spans)]
    return merge_spans(en_spans + others)


def run(m: dict[str, Any], workdir: Path, *, src_lang: str, tgt_lang: str = "en",
        prefer: str = "auto") -> None:
    """Stage 3: produce `words.json` the word stream every later stage reads."""
    # Legacy ISO-639 spellings ("iw", "ji", "in") mean the same language to us and
    # to Whisper's `language=` argument; normalize once so every downstream use
    # (transcribe, LID comparison, script table) sees the modern code.
    src_lang = _LID_ALIAS.get((src_lang or "").lower(), (src_lang or "").lower())
    raw = m["files"].get("captions_raw")
    limit = m["source"].get("duration")
    has_captions = bool(raw) and Path(raw).is_file()
    supplied = bool((m["source"] or {}).get("captions"))
    recovered: list[dict[str, Any]] = []
    en_spans: list[dict[str, Any]] = []
    lang_runs: list[dict[str, Any]] = []

    caption_words = _caption_words(raw, has_captions=has_captions, limit=limit,
                                   prefer=prefer, supplied=supplied)
    _refuse_unusable_captions(prefer, raw, has_captions=has_captions,
                              caption_words=caption_words)

    if prefer in ("captions", "file"):
        # The user's own transcript and the video's own captions are read the
        # same way the only difference is which one is on record, and whether
        # its absence is an error. `origin` keeps them apart so the report, the
        # studio and `is_fallback` can all tell "we were given this" from "we
        # scraped this".
        words, origin = caption_words, ("file" if prefer == "file" else "captions")
    else:
        try:
            words, recovered, en_spans, lang_runs = _asr_transcript(
                m, workdir, src_lang=src_lang, tgt_lang=tgt_lang, limit=limit,
                caption_words=caption_words)
            origin = "asr"
        except Exception as exc:
            if prefer == "asr" or not caption_words:
                raise
            print(f"  transcript: ASR unavailable ({exc}) falling back to captions",
                  file=sys.stderr)
            words, origin = caption_words, "captions"

    if not words:
        raise SystemExit(
            "No transcript words were produced. For a local file pass --captions "
            "<file.srt|.vtt|.json3>, or check that the ASR model is present under models/."
        )

    spans = _all_foreign_spans(caption_words, recovered, en_spans, src_lang, tgt_lang)
    (workdir / "words.json").write_text(
        json.dumps({"origin": origin, "words": words, "foreign_spans": spans,
                    "lang_runs": lang_runs},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    m["files"]["words"] = "words.json"
    m["source"]["transcript_origin"] = origin
    print(f"  transcript: {len(words)} words from {origin}"
          f"{f', {len(spans)} target-language spans from captions' if spans else ''}",
          file=sys.stderr)


def origin(m: dict[str, Any]) -> str | None:
    """Where the transcript came from: "asr", "captions", "file", or nothing yet."""
    return (m.get("source") or {}).get("transcript_origin")


def is_fallback(m: dict[str, Any], prefer: str = "auto") -> bool:
    """True when the transcript is the captions fallback, not the one asked for.

    Captions are authoritative about *where* the target language is spoken and
    nowhere else they mangle exactly the words that matter (AGENTS.md, invariant
    4) so a run that asked for ASR and got captions because the model was
    missing is running on a degraded transcript. It is a real result and
    everything downstream is built on it, but it is not an answer to cache: the
    next run has to try the ASR again.
    """
    return prefer != "captions" and origin(m) == "captions"


def load_words(workdir: Path, m: dict[str, Any]) -> list[dict[str, Any]]:
    data = json.loads((workdir / m["files"]["words"]).read_text(encoding="utf-8"))
    return data["words"]


def load_foreign_spans(workdir: Path, m: dict[str, Any]) -> list[dict[str, Any]]:
    data = json.loads((workdir / m["files"]["words"]).read_text(encoding="utf-8"))
    return data.get("foreign_spans") or []


def load_lang_runs(workdir: Path, m: dict[str, Any]) -> list[dict[str, Any]]:
    """Classifier language runs over the whole video advisory, may be absent."""
    data = json.loads((workdir / m["files"]["words"]).read_text(encoding="utf-8"))
    return data.get("lang_runs") or []
