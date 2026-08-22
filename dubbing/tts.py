"""Stage 6 Qwen3-TTS zero-shot voice cloning, one clip per segment.

Two guarantees this module owns:

* **Never silent.** Every segment leaves this stage with an audio clip. A dub
  that cannot be verified after its bounded retries falls back to the original
  audio slice (`keep_reason="tts_failed"`), never to nothing.
* **Never partial.** A segment is one TTS call for its whole text. There is no
  sub-unit splitting, so there is no way for part of a line to go missing.

Clips are cached on disk by content hash, which is also the resume mechanism for
long videos: a killed run picks up at the first uncached clip. The hash covers
the text, the reference, the options AND the target language, and the record
carries a fingerprint of the line it was made from a clip whose text has moved
under it is re-queued rather than replayed (`clip_text_stale`).

The run's language pair is a default, not a rule: a segment carrying its own
`tgt_lang` (the editor's per-segment override) is prepared, keyed, synthesised
and verified in *that* language.

**Accepting is not the same as passing.** Verification has two bars: a clip below
`CLONE_MIN_OVERLAP` is not a dub and the line keeps its original audio, while a
clip below `CLONE_GOOD_OVERLAP` is used but keeps the retry ladder running and is
recorded as `verify: "accepted"`, never as "ok". One number doing both jobs is
why garbled Hebrew takes at 0.44 were stored on the first try and never retried.
What a retry varies is the *reference* (`Engine.rungs`), because that is the
ingredient a clone's quality actually turns on the seed is not, and under a
greedy decode it is inert.

**A check that did not run is a verdict.** With no verification ASR a clip is
accepted on length alone and recorded as `verify: "unverified"`, never as "ok",
and everything this stage could not load lands in `m["health"]` for report.json.

Per-segment overrides live in `seg["tts_opts"]` (see `dubbing/ttsopts.py`): seed,
forced-greedy decode, a pinned clone reference, model size, tempo, the sampler.
Every one of them is mixed into the clip's cache key, so changing one on an
already-dubbed segment cannot replay the old clip.

**Natural-language style instructions are not available on this path.** The
checkpoint exposes `instruct` only through `generate_voice_design` /
`generate_custom_voice`, which are gated on `tts_model_type` being `voice_design`
/ `custom_voice` and take no reference audio at all they invent a voice from
the description instead of cloning the speaker, which is the one thing this
pipeline may not lose. The closest thing the Base checkpoint has is ICL mode
(`tts_opts.ref_text`): conditioning on a reference's text *and* codes rather than
only its speaker embedding, which carries the reference clip's prosody. That is
a "sound like this take", not "sound angry" and it is the honest limit here.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import soundfile as sf

from . import audio
from . import hebrew as hebrew_mod
from . import script as script_mod
from . import ttsopts
from .script import count_letters, same_script, script_for, speech_units
from .ttsopts import TtsOpts

REPO_ROOT = Path(__file__).resolve().parents[1]
# Voice-clone model, selectable per run (--tts-model). The tag is mixed into every
# clip's cache key, so 0.6B and 1.7B clips never collide and switching re-generates.
TTS_MODELS = {
    "1.7b": {"dir": "Qwen3-TTS-12Hz-1.7B-Base", "hub": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
             "tag": "qwen3-tts-1.7b-base"},
    "0.6b": {"dir": "Qwen3-TTS-12Hz-0.6B-Base", "hub": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
             "tag": "qwen3-tts-0.6b-base"},
}
DEFAULT_TTS_MODEL = "1.7b"
# ttsopts validates `tts_opts.model` without importing this module (it stays free of
# numpy/torch so the editor can validate cheaply); this keeps the two lists honest.
assert set(TTS_MODELS) == set(ttsopts.MODELS), "ttsopts.MODELS is out of step with TTS_MODELS"
CODEC_HZ = 12.5

# Default sampler for a non-greedy attempt the values `tts_opts` overrides.
SAMPLED = {"temperature": 0.55, "top_p": 0.85, "top_k": 30}
GREEDY = {"temperature": 0.01, "top_p": 1.0, "top_k": 1}
REPETITION_PENALTY = 1.08

REF_SR = 24000
REF_TARGET_SEC = 4.5
MIN_REF_SEC = 2.5    # a clone reference shorter than this yields truncated/garbled clones
REF_MIN_RMS = 0.018
REF_JOIN_FADE_SEC = 0.02   # fade at the joins of a concatenated reference

# Voice-outlier rejection for candidate reference windows, thresholds measured on a
# 12-speaker drama run (109 candidate windows): per-candidate mean cosine similarity
# to the speaker's other windows had median 0.35, p25 0.24, p10 0.15. Coherent
# speakers' windows mutually average 0.36-0.70; windows later confirmed to be another
# voice sat at 0.08-0.23. Speakers with globally incoherent windows (whisper +
# music leakage) average 0.17-0.20 with no true outliers, hence the coherence gate:
# a low candidate is only rejected when the REST of the speaker's windows agree
# with each other.
REF_SIM_OUTLIER = 0.25     # candidate mean-cos below this is suspect ...
REF_SIM_COHERENT = 0.35    # ... but only when the other windows mutually average this
REF_MATCH_MIN = 0.25       # segment window vs canonical ref: same-voice acceptance

# What `_verify` reports when there is no verification ASR at all. It is a
# verdict of its own, not an "ok": the clip passed the length guard and nothing
# else looked at it. Carried onto the record as verify="unverified" so the run
# summary can say how much of the safety net actually ran.
NO_ASR = "no-asr"

# What this stage records in `m["health"]` when it runs degraded, and therefore
# what a fresh run of it clears (see `run`). Read by `report.run`.
HEALTH_KEYS = ("tts.verify_asr", "tts.speaker_embeddings")

CLONE_MIN_SEC_PER_WORD = 0.18   # faster than this is chipmunk garble
CLONE_MAX_SEC_PER_WORD = 0.95   # slower than this is a stall/drawl
# CJK/hangul speech runs ~5 characters/s; the word constants assume ~3 words/s.
CLONE_MIN_SEC_PER_CHAR = 0.08   # faster than this is chipmunk garble
CLONE_MAX_SEC_PER_CHAR = 0.60   # slower than this is a stall/drawl
# Two bars, not one. `CLONE_MIN_OVERLAP` is the floor: below it a line is not a
# dub at all and falls back to the original audio. `CLONE_GOOD_OVERLAP` is what
# stops the retry ladder. They used to be the same number, which is why a garbled
# Hebrew take at 0.44 was stored as "ok" on the first try and never retried: the
# accept bar was doing the retry bar's job. Anything between them is accepted
# only after the ladder has been walked, and is recorded as "accepted", not "ok".
CLONE_MIN_OVERLAP = 0.35        # accept floor: below this the line keeps its original
CLONE_GOOD_OVERLAP = 0.85       # good enough to stop retrying
CLONE_SOFT_OVERLAP = 0.20       # best-effort accept
# How many *references* one segment is cloned from before the decode is changed:
# the aligned window plus up to two alternatives (see `Engine._ref_ladder`).
MAX_TRIES = 3
REF_ALTS = 3                    # alternative windows kept per speaker for the ladder

_LATIN_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9''\-.]*")
_CHROME = re.compile(r"[\[\(][^\]\)]{0,40}[\]\)]")
# CJK punctuation → the ASCII marks prepare_text understands. The table itself
# lives in `script`, because `segments.SENTENCE_END` reads the same one: what
# ends a Japanese sentence for the segmenter must end it for the voice too.
_CJK_PUNCT = str.maketrans(script_mod.CJK_PUNCT)

_RUN_RES: dict[str, re.Pattern] = {"latin": _LATIN_RUN}


def _run_re(bucket: str) -> re.Pattern:
    """A `_LATIN_RUN`-shaped regex for any script bucket (letters + digits)."""
    got = _RUN_RES.get(bucket)
    if got is None:
        cls = "".join(f"{chr(lo)}-{chr(hi)}" for lo, hi in script_mod._RANGES[bucket])
        got = _RUN_RES[bucket] = re.compile(f"[{cls}0-9][{cls}0-9''\\-.]*")
    return got


def prepare_text(text: str, lang: str) -> str:
    """Strip anything the target-language voice cannot speak; keep the rest.

    Keeps runs of the TARGET script's letters plus digits and basic punctuation,
    so a Hebrew (source) word inside a Russian line is dropped without deleting
    the line the old Latin-only version erased entire non-Latin translations.
    """
    bucket = script_for(lang)
    run = _run_re(bucket)
    t = _CHROME.sub(" ", text or "")
    t = t.replace("[[", " ").replace("]]", " ")
    if bucket in ("cjk", "hangul"):
        t = t.translate(_CJK_PUNCT)
    parts = re.split(r"([.!?,;:])", t)
    out: list[str] = []
    for part in parts:
        if part in ".!?,;:":
            if out and out[-1] not in ".!?,;:":
                out.append(part)
            continue
        words = run.findall(part)
        if words:
            out.append(" ".join(words))
    speak = re.sub(r"\s+([.!?,;:])", r"\1", " ".join(out)).strip()
    speak = re.sub(r"\s+", " ", speak)
    if speak and speak[-1] not in ".!?":
        speak += "."
    return speak


def prepare_en(text: str) -> str:
    """Strip anything the English voice cannot speak; keep every word it can."""
    return prepare_text(text, "en")


def _tokens(text: str, lang: str = "en") -> list[str]:
    bucket = script_for(lang)
    if bucket in ("cjk", "hangul"):
        # No reliable word boundaries compare single characters instead.
        return [ch for ch in (text or "").lower() if ch.isalnum()]
    if (lang or "en").lower() == "en":
        return re.findall(r"[a-z0-9']+", (text or "").lower())
    return re.findall(r"[\w']+", (text or "").lower(), re.UNICODE)


def word_overlap(target: str, heard: str, lang: str = "en") -> float:
    a, b = _tokens(target, lang), _tokens(heard, lang)
    if not a:
        return 0.0
    common = sum((Counter(a) & Counter(b)).values())
    return common / len(a)


def clip_exceeds_slot(clip_sec: float, slot_sec: float) -> bool:
    """A clip so much longer than its segment that placement cannot recover.

    Translations legitimately run longer than the source and the timeline absorbs
    up to ~DRIFT_MAX of overhang via shortening and speed-up; the bound here is
    far beyond that (3x the slot plus 8s) so it only fires on runaway synthesis,
    never on ordinary expansion.
    """
    return clip_sec > slot_sec * 3.0 + 8.0


def clone_length_ok(sec: float, text: str, lang: str = "en") -> bool:
    char_based = script_for(lang) in ("cjk", "hangul")
    units = speech_units(text, lang)
    if sec <= 0.05:
        return False
    spu = sec / units
    lo, hi = ((CLONE_MIN_SEC_PER_CHAR, CLONE_MAX_SEC_PER_CHAR) if char_based
              else (CLONE_MIN_SEC_PER_WORD, CLONE_MAX_SEC_PER_WORD))
    if spu < lo:
        return False
    if spu > hi and units >= 3:
        return False
    expected = (units / 5.0 if char_based else units / 3.0) + 0.25
    return sec >= expected * 0.40


def max_new_tokens(text: str, lang: str = "en") -> int:
    units = speech_units(text, lang)
    sec = units * 0.25 + 2.5 if script_for(lang) in ("cjk", "hangul") else units * 0.65 + 2.5
    return max(96, min(2048, int(sec * CODEC_HZ)))


def source_script_leak(heard: str, src: str, tgt: str) -> bool:
    """True when an ASR transcript is dominated by the SOURCE language's script.

    Only meaningful when the pair's scripts differ (`same_script` → always False):
    for en→es both are Latin and this check cannot discriminate.
    """
    if same_script(src, tgt):
        return False
    src_letters = count_letters(heard, script_for(src))
    tgt_letters = count_letters(heard, script_for(tgt))
    return bool(src_letters and (src_letters >= tgt_letters or tgt_letters < 3))


# Qwen3-TTS names languages in English ("Russian"), the pipeline uses ISO codes.
_QWEN_LANG_NAMES = {
    "en": "English", "zh": "Chinese", "de": "German", "it": "Italian",
    "pt": "Portuguese", "es": "Spanish", "ja": "Japanese", "ko": "Korean",
    "fr": "French", "ru": "Russian",
}


def qwen_language_name(code: str, supported) -> str:
    """The checkpoint's name for a language code, or "Auto" when unsupported."""
    want = _QWEN_LANG_NAMES.get((code or "").lower())
    if want:
        for name in supported or ():
            if str(name).lower() == want.lower():
                return str(name)
    return "Auto"


def synthesis_text(speak: str, lang: str) -> str:
    """What the checkpoint is actually asked to say, for a line already prepared.

    Identical to `speak` for the ten languages the base model knows. Hebrew is the
    exception: its LoRA was trained on stressed IPA, and Hebrew orthography leaves
    the vowels out, so the text field gets `ʁˈeɡa` where the subtitle says רגע (see
    `dubbing/hebrew.py`). The orthography remains the record it is what is stored,
    subtitled and ASR-verified; the IPA never leaves this stage.
    """
    if hebrew_mod.is_hebrew(lang):
        return hebrew_mod.phonemize(speak)
    return speak


# --------------------------------------------------------------------------- refs


PAUSE_MAX = 0.35    # an internal TTS silence longer than this is unnatural
PAUSE_KEEP = 0.12   # ... and is compressed down to this


def _trim_internal_silence(wav: np.ndarray, sr: int, *, floor: float = 0.015) -> np.ndarray:
    """Compress over-long silences inside a generated clip.

    Qwen sometimes drops a ~1s pause before a trailing phrase (a comma before
    "par excellence"), which the placement then can't hide. Runs of silence longer
    than PAUSE_MAX are shortened to PAUSE_KEEP; speech is untouched.
    """
    x = np.asarray(wav, dtype=np.float32)
    hop = max(1, int(0.02 * sr))
    n = len(x) // hop
    if n < 2:
        return x
    rms = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-9)
    silent = rms < floor
    keep_f, max_f = max(1, int(PAUSE_KEEP / 0.02)), int(PAUSE_MAX / 0.02)
    pieces: list[np.ndarray] = []
    i = 0
    while i < n:
        j = i
        while j < n and silent[j] == silent[i]:
            j += 1
        seg = x[i * hop : j * hop]
        if silent[i] and (j - i) > max_f:
            seg = seg[: keep_f * hop]
        pieces.append(seg)
        i = j
    pieces.append(x[n * hop :])
    return np.concatenate(pieces) if pieces else x


def best_ref_window(voc: np.ndarray, a: float, b: float, target: float
                    ) -> tuple[float, float, float, float] | None:
    """Cleanest `target`-second window in [a, b] → (start, end, noise, rms)."""
    i0 = max(0, int(a * REF_SR))
    i1 = min(len(voc), int(b * REF_SR))
    if i1 - i0 < int(0.5 * REF_SR):
        return None
    span = voc[i0:i1]
    want = min(len(span), int(target * REF_SR))
    hop = max(1, int(0.2 * REF_SR))
    best = None
    for i in range(0, max(1, len(span) - want + 1), hop):
        chunk = span[i : i + want]
        r = audio.rms(chunk)
        if r < REF_MIN_RMS:
            continue
        score = audio.hf_noise_ratio(chunk, REF_SR) / max(r, 1e-3)
        if best is None or score < best[0]:
            best = (score, i, r)
    if best is None:
        return None
    score, i, r = best
    start = (i0 + i) / REF_SR
    return start, start + want / REF_SR, score, r


# ECAPA speaker embeddings validate reference windows (same lazy-loader pattern as
# dubbing/segments.py). The model is tiny (~80MB) and optional: every caller treats
# None as "no validation", so a missing model leaves behaviour unchanged.
ECAPA_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_DIR = REPO_ROOT / "models" / "spkrec-ecapa-voxceleb"
_ECAPA = None
_ECAPA_FAILED = False


def _load_ecapa():
    """The ECAPA speaker-embedding model, loaded once; None if unavailable."""
    global _ECAPA, _ECAPA_FAILED
    if _ECAPA is not None or _ECAPA_FAILED:
        return _ECAPA
    try:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _ECAPA = EncoderClassifier.from_hparams(
            source=ECAPA_MODEL, savedir=str(ECAPA_DIR), run_opts={"device": device})
    except Exception as exc:
        _ECAPA_FAILED = True
        print(f"  tts: speaker embeddings unavailable ({exc}) "
              "references unvalidated", file=sys.stderr)
    return _ECAPA


def _embed_clip(model, clip: np.ndarray) -> np.ndarray:
    """Unit-normalized ECAPA embedding of one 16kHz mono clip."""
    import torch

    wav = torch.from_numpy(clip.astype("float32")).unsqueeze(0)
    with torch.no_grad():
        emb = model.encode_batch(wav.to(next(model.parameters()).device))
    v = emb.squeeze().cpu().numpy().astype(float)
    return v / (np.linalg.norm(v) or 1.0)


def _embed_windows(vocals: Path, spans: list[tuple[float, float]]) -> np.ndarray | None:
    """One embedding per (start, end) span of the vocals file, or None (no model)."""
    model = _load_ecapa()
    if model is None:
        return None
    return np.stack([_embed_clip(model, audio.decode_mono(vocals, 16000, start=a, end=b))
                     for a, b in spans])


def _embed_wavfile(path: Path) -> np.ndarray | None:
    """Embedding of a whole wav file, or None (no model)."""
    model = _load_ecapa()
    if model is None:
        return None
    return _embed_clip(model, audio.decode_mono(path, 16000))


def reject_voice_outliers(vecs, *, low: float = REF_SIM_OUTLIER,
                          coherent: float = REF_SIM_COHERENT) -> list[bool]:
    """Keep-mask over one speaker's candidate windows; False = different voice.

    A candidate is rejected only when BOTH hold: its mean cosine similarity to the
    speaker's other windows is below `low`, and those other windows mutually average
    at least `coherent` i.e. the rest agree on one voice and this window doesn't
    match it (a diarization mislabel). A speaker whose windows are all mutually
    dissimilar (noisy/whispered material) keeps everything: there is no consensus
    voice to reject against. Thresholds are measured see REF_SIM_OUTLIER above.
    """
    vecs = np.asarray(vecs, dtype=float)
    n = len(vecs)
    if n < 3:
        return [True] * n       # pairwise similarity cannot say which one is wrong
    sims = vecs @ vecs.T
    keep: list[bool] = []
    for i in range(n):
        mean_i = (sims[i].sum() - sims[i, i]) / (n - 1)
        if mean_i >= low:
            keep.append(True)
            continue
        idx = [j for j in range(n) if j != i]
        sub = sims[np.ix_(idx, idx)]
        m = len(idx)
        others = (sub.sum() - np.trace(sub)) / (m * (m - 1))
        keep.append(others < coherent)
    if not any(keep):
        return [True] * n       # never reject everything
    return keep


def choose_ref_windows(cands: list[tuple[float, float, float, float]],
                       *, min_sec: float = MIN_REF_SEC,
                       target: float = REF_TARGET_SEC) -> list[tuple[float, float]]:
    """Which of a speaker's candidate windows make their reference.

    Prefers the cleanest single window at least `min_sec` long. When every window
    is shorter, takes the cleanest windows until their total reaches `target`
    (they are concatenated by the caller), so a speaker of short lines still gets
    a reference long enough to clone from. With too little material the result is
    simply everything there is possibly one short window, the old behaviour.
    Returns (start, end) spans in time order.
    """
    if not cands:
        return []
    long_enough = [c for c in cands if c[1] - c[0] >= min_sec]
    if long_enough:
        best = min(long_enough, key=lambda c: (c[2], c[0]))
        return [(best[0], best[1])]
    chosen: list[tuple[float, float, float, float]] = []
    total = 0.0
    for c in sorted(cands, key=lambda c: (c[2], c[0])):
        chosen.append(c)
        total += c[1] - c[0]
        if total >= target:
            break
    return sorted((c[0], c[1]) for c in chosen)


def concat_ref(voc: np.ndarray, spans: list[tuple[float, float]], sr: int = REF_SR,
               *, fade: float = REF_JOIN_FADE_SEC) -> np.ndarray:
    """The spans of `voc`, in time order, joined with short fades at the edges."""
    nf = max(1, int(fade * sr))
    pieces: list[np.ndarray] = []
    for a, b in sorted(spans):
        x = np.array(voc[int(a * sr): int(b * sr)], dtype=np.float32)
        if not len(x):
            continue
        k = min(nf, len(x) // 2)
        if k:
            ramp = np.linspace(0.0, 1.0, k, dtype=np.float32)
            x[:k] *= ramp
            x[-k:] *= ramp[::-1]
        pieces.append(x)
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)


def sampling_kwargs(greedy: bool, opts: TtsOpts = ttsopts.DEFAULT) -> dict[str, Any]:
    """The decode arguments for one attempt, main talker and sub-talker in step.

    A greedy attempt ignores `tts_opts`' sampler entirely which is exactly why
    `ttsopts.parse` refuses to store the two together instead of letting the
    values sit there doing nothing.
    """
    base = GREEDY if greedy else SAMPLED
    out: dict[str, Any] = {"do_sample": not greedy, "subtalker_dosample": not greedy}
    for k, default in base.items():
        v = default if greedy or getattr(opts, k) is None else getattr(opts, k)
        out[k] = v
        out[f"subtalker_{k}"] = v
    return out


def seed_id(seg: dict[str, Any]) -> str:
    """The identity the derived seed keys on stable across re-segmentation.

    `seg["id"]` is positional: inserting one segment renumbers everything after
    it, which silently re-rolls every later line's voice take. `uid` (minted once
    per segment, see APP_ARCHITECTURE.md) does not move, so it is preferred where
    it exists. Falling back to `id` keeps every pre-uid manifest on exactly the
    seeds and therefore exactly the cached clips it already had.
    """
    return str(seg.get("uid") or seg["id"])


def seed_for(seg: dict[str, Any], speak: str, opts: TtsOpts = ttsopts.DEFAULT) -> int:
    """The base seed for a segment: the user's, else derived from identity + text."""
    if opts.seed is not None:
        return opts.seed
    return int(hashlib.sha1(f"{seed_id(seg)}|{speak}".encode()).hexdigest()[:8], 16)


def _verdict(clip: Path, ok: bool, ov: float, heard: str) -> dict[str, Any]:
    """One attempt's verification result, as stored in `clips/<key>.json`.

    Built in exactly one place because the pipelined path in `run` and the
    sequential retry path in `clip_for` must record the same thing including
    `verified`, which is False when there was no ASR to ask. Without that flag the
    two paths both wrote "ok" and a run with no verifier looked, in the manifest
    and in report.json, exactly like a run where every clip passed.
    """
    return {"ok": ok, "overlap": round(ov, 3), "heard": heard,
            "dur": round(audio.duration(clip), 3), "verified": heard != NO_ASR}


def clip_is_good(verdict: dict[str, Any]) -> bool:
    """True when a take is good enough to stop the retry ladder.

    The upper of the two bars (see CLONE_GOOD_OVERLAP). An unverified clip reads
    as good — there is no ASR to disagree with, so retrying it would only spend
    GPU time re-rolling a verdict nobody can check.
    """
    return bool(verdict.get("ok")) and verdict.get("overlap", 0.0) >= CLONE_GOOD_OVERLAP


class Plan(NamedTuple):
    """What one segment's attempts are made from (see `Engine._plan`).

    `speak` is the line in its own script what gets stored, shown and verified.
    `synth` is what the checkpoint is handed, which differs only for Hebrew, where
    it is the IPA transcription of `speak` (see `synthesis_text`).
    """

    speak: str
    synth: str
    ref_path: Path
    ref_key: str
    base_seed: int
    opts: TtsOpts
    # This segment's own language pair. Usually the run's, but `seg["tgt_lang"]`
    # is a real per-segment override (edit.set_langs writes it and the translator
    # honours it), so the synthesiser, the cache key and the verifier all have to
    # read it too a French line synthesised as English, keyed as English and
    # verified against an English ASR failed verification every time and fell back
    # to keep, with no hint that a supported override was the cause.
    tgt: str
    src: str
    # The decode attempt 0 uses. Usually `opts.greedy`, but a Hebrew target
    # defaults to the deterministic decode (see `Engine._greedy_for`), and that
    # default is the plan's, not the segment's: it must not leak into the option
    # fingerprint stored on the record, or every Hebrew clip would look like its
    # options had changed under it and be re-synthesized forever.
    greedy: bool = False


def text_sha(text: str) -> str:
    """Fingerprint of the line a clip was made from (see `clip_text_stale`)."""
    return hashlib.sha1((text or "").strip().encode("utf-8")).hexdigest()[:12]


def clip_text_stale(seg: dict[str, Any]) -> bool:
    """True when this segment's clip provably says a line it no longer holds.

    A clip is audio of specific words. Nothing recorded which words, so a
    translate rerun that replaced `text_en` under an existing clip left a segment
    whose voice says the old sentence while the manifest, the subtitle and the
    editor all show the new one and the resume test ("there is a clip, and its
    options match") called that done.

    False for records written before the fingerprint existed and for keeps (whose
    clip is a slice of the original audio, not of any line).
    """
    rec = seg.get("tts") or {}
    have = rec.get("text_sha")
    if not have or seg.get("keep"):
        return False
    return have != text_sha(seg.get("text_en") or "")


def stale_locked_clip(seg: dict[str, Any]) -> bool:
    """A locked clip whose text moved: a conflict only the user can settle.

    The lock says "this take is approved"; the text says "that is not what this
    line reads any more". Regenerating silently overrules the user's approval and
    keeping silently ships the wrong words, so neither happens it is reported
    (`report.json: stale_locked_clips`) and left for the user to resynthesize or
    unlock.
    """
    from . import manifest

    return manifest.is_locked(seg, "tts") and clip_text_stale(seg)


def speakable(seg: dict[str, Any]) -> str | None:
    """The text this dubbed segment can be synthesized from, or None.

    None is a real state, not an error: `translate.mark_failed` honours a
    user-locked "dub it" by leaving `keep` false with NO translation the
    failure was the translator's, and flipping the verdict would be the
    overwrite that path refused. Such a segment reaches this stage with
    nothing to speak; the stage's answer is the universal fallback
    (`keep_clip`), not a crash and not a re-decided verdict.
    """
    text = (seg.get("text_en") or "").strip()
    return text or None


def needs_synthesis(seg: dict[str, Any], workdir: Path) -> bool:
    """True when this dubbed segment has no clip that still matches its options.

    A usable clip on disk is normally the whole resume test. It is not enough
    once options exist: editing `tts_opts` on an already-dubbed line leaves the
    old clip in place, and without this the re-run would look successful and
    change nothing. The record carries the option fingerprint it was made with,
    so a mismatch puts the segment back in the queue.

    Nor is it enough once the *text* can move under a clip: the record carries a
    fingerprint of the line it was made from too, and a mismatch re-queues the
    segment unless the user locked that clip, in which case the conflict is
    reported rather than decided (see `stale_locked_clip`).
    """
    rec = seg.get("tts")
    if not rec or not (workdir / rec["clip"]).is_file():
        return True
    if rec.get("opts", "") != ttsopts.parse(seg.get("tts_opts")).fingerprint():
        return True
    return clip_text_stale(seg) and not stale_locked_clip(seg)


class Engine:
    """Holds the loaded models and the reference bank for one run."""

    def __init__(self, m: dict[str, Any], workdir: Path, *, device: str | None = None,
                 model: str = DEFAULT_TTS_MODEL):
        self.m = m
        src = m.get("source") or {}
        self.src_lang = (src.get("src_lang") or "he").lower()
        # "iw" is Hebrew's legacy code and reaches whisper's `language=` as a
        # rejected value; the pipeline speaks one spelling of each language.
        tgt = (src.get("tgt_lang") or "en").lower()
        self.tgt_lang = "he" if hebrew_mod.is_hebrew(tgt) else tgt
        self.workdir = workdir
        self.clips = workdir / "clips"
        self.refs = workdir / "refs"
        self.clips.mkdir(parents=True, exist_ok=True)
        self.refs.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.hebrew = hebrew_mod.is_hebrew(self.tgt_lang)
        if self.hebrew:
            # The Hebrew LoRA is trained against the 1.7B Base talker and carries
            # that checkpoint's output heads, so the 0.6B one cannot wear it. The
            # CLI refuses the combination outright; this is the manifest path (a
            # run whose recorded --tts-model predates the Hebrew target).
            model = hebrew_mod.ADAPTER_MODEL
        self.tts_model = model if model in TTS_MODELS else DEFAULT_TTS_MODEL
        self.model_tag = TTS_MODELS[self.tts_model]["tag"]
        self._synth = None
        self._synth_model: str | None = None
        self._asr: dict[str, Any] = {}        # target language → verify ASR (or None)
        self._voc: np.ndarray | None = None
        self._ref_hash: dict[str, str] = {}          # canonical ref path → content hash
        self._match_cache: dict[tuple, bool] = {}    # (span, canonical path) → same voice?

    # -- lazy resources ----------------------------------------------------
    @property
    def vocals_path(self) -> Path:
        return self.workdir / self.m["files"]["vocals"]

    @property
    def vocals(self) -> np.ndarray:
        if self._voc is None:
            self._voc = audio.decode_mono(self.vocals_path, REF_SR)
        return self._voc

    @property
    def synth(self):
        return self.synth_for(ttsopts.DEFAULT)

    def model_for(self, opts: TtsOpts = ttsopts.DEFAULT) -> str:
        """Which checkpoint this segment uses its own override, else the run's.

        A Hebrew target has no choice: only the 1.7B Base checkpoint fits the LoRA,
        so a per-segment `tts_opts.model` cannot pick the 0.6B one out from under it.
        """
        if self.hebrew:
            return hebrew_mod.ADAPTER_MODEL
        return opts.model or self.tts_model

    def model_tag_for(self, opts: TtsOpts = ttsopts.DEFAULT) -> str:
        return TTS_MODELS[self.model_for(opts)]["tag"]

    def synth_for(self, opts: TtsOpts = ttsopts.DEFAULT):
        """The loaded synth for this segment's model, swapping checkpoints if asked.

        Only ever one is resident: `tts_opts.model` is a per-line escape hatch, and
        two Qwen3-TTS checkpoints in float32 would not fit alongside everything
        else (AGENTS.md device notes). Switching therefore frees the other one,
        which costs a reload fine for the handful of lines a user overrides,
        which is why it is not the run-wide default.
        """
        want = self.model_for(opts)
        if self._synth is not None and self._synth_model != want:
            print(f"  tts: switching checkpoint {self._synth_model} → {want} "
                  "for a segment override", file=sys.stderr)
            self._synth.free()
            self._synth = None
        if self._synth is None:
            self._synth = _Synth(device=self.device, model=want, lang=self.tgt_lang)
            self._synth_model = want
        return self._synth

    @property
    def asr(self):
        return self.asr_for(self.tgt_lang)

    def asr_for(self, tgt: str | None = None):
        """The verification ASR for one target language, loaded once each.

        Keyed by language because a segment may target its own (`seg["tgt_lang"]`):
        a French line checked against the English model reads as gibberish, fails
        verification every time and falls back to keep a supported override
        turned into a silent, permanent failure.
        """
        tgt = (tgt or self.tgt_lang or "en").lower()
        if tgt not in self._asr:
            self._asr[tgt] = _load_asr(tgt)
            if self._asr[tgt] is None:
                self.note_degraded(
                    "tts.verify_asr",
                    f"unavailable: no faster-whisper model for {tgt!r} "
                    "clips accepted on length alone")
        return self._asr[tgt]

    # -- degraded operation ------------------------------------------------
    def note_degraded(self, component: str, reason: str) -> None:
        """Record that part of the safety net could not run (see `report.run`).

        The never-silent principle applies to the checks as much as to the audio:
        a verification that did not happen is a verdict this stage wrote, and a
        verdict written by a failure path has to be as visible as the user's own.
        stderr scrolls past; `m["health"]` reaches report.json.
        """
        self.m.setdefault("health", {})[component] = reason

    # -- reference audio ---------------------------------------------------
    # Bumped when the canonical-ref recipe changes: reset_stage clears segments'
    # tts records but not m["speakers"], so without this marker a manifest from
    # an older run would keep its old (possibly sub-second) reference forever.
    REF_BUILD = 3

    def build_speaker_refs(self) -> None:
        """One canonical reference per speaker: their cleanest few seconds.

        Plus `ref_alts`: the runners-up. Clone quality is chaotically sensitive to
        the reference and no static metric predicts it, so the only thing a retry
        can usefully change is which audio the voice is cloned from — and the
        candidate windows measured here are the ready-made supply. They used to be
        thrown away the moment the canonical reference was written; now the best
        few survive for `_ref_ladder` to retry from.

        A reference shorter than MIN_REF_SEC reliably clones truncated or
        wrong-sounding, so when a speaker's best single window is that short their
        cleanest windows are concatenated (time order, short fades at the joins)
        up to ~REF_TARGET_SEC. Candidate windows are first ECAPA-validated: a
        window whose voice is an outlier against the speaker's other windows is a
        diarization mislabel and is dropped (see reject_voice_outliers). With no
        embedding model, validation is skipped and every candidate stands.
        """
        by_speaker: dict[str, list[dict]] = {}
        for seg in self.m["segments"]:
            if not seg["keep"]:
                by_speaker.setdefault(seg["speaker"], []).append(seg)
        for spk, segs in by_speaker.items():
            info = self.m["speakers"].setdefault(spk, {})
            if (info.get("ref") and (self.workdir / info["ref"]).is_file()
                    and info.get("ref_v") == self.REF_BUILD):
                continue
            longest = sorted(segs, key=lambda s: s["start"] - s["end"])[:30]
            cands = []
            for seg in longest:
                got = best_ref_window(self.vocals, seg["start"], seg["end"], REF_TARGET_SEC)
                if got:
                    cands.append(got)
            cands = self._validated_candidates(spk, cands)
            spans = choose_ref_windows(cands)
            wav = concat_ref(self.vocals, spans)
            if not len(wav):
                info["ref"] = None
                info["ref_alts"] = []
                info["ref_v"] = self.REF_BUILD
                print(f"  tts: no clean reference for {spk}", file=sys.stderr)
                continue
            path = self.refs / f"{spk}.wav"
            sf.write(str(path), wav, REF_SR)
            by_start = {(c[0], c[1]): c for c in cands}
            info["ref"] = f"refs/{path.name}"
            info["ref_span"] = [round(spans[0][0], 2), round(spans[-1][1], 2)]
            info["ref_windows"] = [[round(a, 2), round(b, 2)] for a, b in spans]
            info["ref_sec"] = round(len(wav) / REF_SR, 2)
            info["ref_noise"] = round(min(by_start[s][2] for s in spans), 2)
            # The retry supply: the cleanest candidate windows that are NOT part
            # of the canonical reference, so a ladder rung is genuinely different
            # audio rather than the canonical clip cut a second way.
            chosen = set(spans)
            alts = sorted((c for c in cands if (c[0], c[1]) not in chosen),
                          key=lambda c: c[2])[:REF_ALTS]
            info["ref_alts"] = [[round(c[0], 2), round(c[1], 2)] for c in alts]
            info["ref_v"] = self.REF_BUILD
            if len(spans) > 1:
                print(f"  tts: {spk} reference concatenated from {len(spans)} windows "
                      f"({info['ref_sec']:.1f}s)", file=sys.stderr)

    def _validated_candidates(self, spk: str, cands: list) -> list:
        """Drop candidate windows whose voice mismatches the speaker's consensus."""
        if len(cands) < 3:
            return cands
        vecs = _embed_windows(self.vocals_path, [(c[0], c[1]) for c in cands])
        if vecs is None:
            self.note_degraded(
                "tts.speaker_embeddings",
                "unavailable: clone references unvalidated a diarization "
                "mislabel can clone the wrong voice")
            return cands
        keep = reject_voice_outliers(vecs)
        for c, k in zip(cands, keep):
            if not k:
                print(f"  tts: {spk} ref window {c[0]:.2f}-{c[1]:.2f} rejected "
                      "(voice outlier likely another speaker)", file=sys.stderr)
        return [c for c, k in zip(cands, keep) if k]

    def pinned_ref(self, opts: TtsOpts) -> tuple[Path, str] | None:
        """The reference `tts_opts.ref` names, or None when the segment pins none.

        Loud on a missing file rather than quietly cloning something else: a
        pinned reference is the user saying "this line must sound like *this*",
        and falling back to the automatic window would look like the override was
        ignored. The key carries a content hash, so replacing the wav in place
        re-synthesizes instead of replaying the clip cloned from the old bytes.
        """
        if not opts.ref:
            return None
        path = self.workdir / opts.ref
        if not path.is_file():
            raise FileNotFoundError(
                f"tts_opts.ref: no such reference audio under the run directory: {opts.ref}")
        h = self._ref_hash.get(str(path))
        if h is None:
            h = hashlib.sha1(path.read_bytes()).hexdigest()[:10]
            self._ref_hash[str(path)] = h
        return path, f"opt:{opts.ref}:{h}"

    def ref_for(self, seg: dict[str, Any], opts: TtsOpts = ttsopts.DEFAULT
                ) -> tuple[Path, str] | None:
        """The segment's own aligned voice whoever actually speaks in this window.

        `tts_opts.ref` short-circuits all of this: the user picked the voice.

        The reference is cut strictly from the segment's own `[start, end]`, so the
        cloned voice is the real speaker at that exact moment. It deliberately does
        NOT widen into neighbouring segments: diarization here is unreliable —
        ECAPA shows two segments labelled the same speaker can be different voices —
        so widening by label blends another person into the clone (the audible
        wrong-speaker/voice-switch around 1:11).

        A short line is the one exception: a sub-MIN_REF_SEC reference reliably
        clones truncated, so when the aligned window is that short AND an ECAPA
        embedding confirms it is the same voice as the speaker's canonical
        reference, the canonical (longer, validated) reference is used instead.
        Without embeddings there is no confirmation, and the short aligned window
        stands the old behaviour, never a blind widen-by-label.
        """
        pinned = self.pinned_ref(opts)
        if pinned is not None:
            return pinned
        span = seg["end"] - seg["start"]
        got = best_ref_window(self.vocals, seg["start"], seg["end"], min(REF_TARGET_SEC, span))
        if got and got[3] >= REF_MIN_RMS:
            start, end, _score, _rms = got
            if end - start < MIN_REF_SEC:
                canon = self._canonical_ref(seg)
                if canon is not None and self._matches_canonical((start, end), canon[0]):
                    return canon
            return self.window_ref(start, end)
        # Only if the aligned window is too short or essentially silent: the
        # speaker's canonical clip, if one was built, else the synth default voice.
        return self._canonical_ref(seg)

    def window_ref(self, start: float, end: float) -> tuple[Path, str]:
        """One window of the vocals as a clone reference, cut on demand.

        Named by its audio window, never the segment id: ids shift between runs as
        segmentation changes, so an id-named file goes stale and clones a different
        moment's voice (the "1:19 voice at 1:33" bug). A window-named file, and a
        window-based cache key, cannot.
        """
        path = self.refs / f"ref_{start:.2f}-{end:.2f}.wav"
        if not path.is_file():
            sf.write(str(path), self.vocals[int(start * REF_SR) : int(end * REF_SR)],
                     REF_SR)
        return path, f"ref:{start:.2f}-{end:.2f}"

    def _ref_ladder(self, seg: dict[str, Any], opts: TtsOpts,
                    first_key: str) -> list[tuple[Path, str]]:
        """Alternative references for attempts 1+, best first.

        This is what a retry actually varies. Re-rolling the same reference cannot
        help: under a greedy decode the seed is inert, so every retry is the same
        audio, and even sampled the reference is what dominates a clone's quality.
        Nothing static predicts which reference clones well — only the verify ASR
        knows — so the ladder is "try a different voice sample and ask again".

        The speaker's canonical reference is rung 1 (it is the longest, cleanest,
        outlier-checked audio this run has of them), then the runners-up
        `build_speaker_refs` kept. A pinned `tts_opts.ref` has no ladder at all:
        the user chose that voice, and swapping it out would hand back exactly the
        one they were overriding.
        """
        if opts.ref:
            return []
        out: list[tuple[Path, str]] = []
        seen = {first_key}
        canon = self._canonical_ref(seg)
        if canon is not None and canon[1] not in seen:
            out.append(canon)
            seen.add(canon[1])
        info = self.m.get("speakers", {}).get(seg.get("speaker")) or {}
        for span in info.get("ref_alts") or []:
            got = self.window_ref(float(span[0]), float(span[1]))
            if got[1] not in seen:
                out.append(got)
                seen.add(got[1])
        return out[: MAX_TRIES - 1]

    def _matches_canonical(self, span: tuple[float, float], canon: Path) -> bool:
        """ECAPA says the aligned window and the canonical ref are the same voice.

        False whenever embeddings are unavailable the caller then keeps the
        segment's own audio, exactly the pre-embedding behaviour.
        """
        key = (round(span[0], 2), round(span[1], 2), str(canon))
        got = self._match_cache.get(key)
        if got is None:
            vecs = _embed_windows(self.vocals_path, [span])
            cvec = _embed_wavfile(canon)
            if vecs is None or cvec is None:
                self.note_degraded(
                    "tts.speaker_embeddings",
                    "unavailable: clone references unvalidated short windows "
                    "clone truncated and nothing confirms the voice")
            got = (vecs is not None and cvec is not None
                   and float(vecs[0] @ cvec) >= REF_MATCH_MIN)
            self._match_cache[key] = got
        return got

    def _canonical_ref(self, seg: dict[str, Any]) -> tuple[Path, str] | None:
        """The speaker's canonical clean reference (see build_speaker_refs).

        The ref key carries a hash of the file's bytes: the canonical wav for a
        speaker is rebuilt over time (longer windows, concatenation, outlier
        rejection), and a bare "SPK:canonical" key would keep handing back clips
        cloned from the old reference.
        """
        info = self.m["speakers"].get(seg["speaker"]) or {}
        if info.get("ref") and (self.workdir / info["ref"]).is_file():
            path = self.workdir / info["ref"]
            h = self._ref_hash.get(str(path))
            if h is None:
                h = hashlib.sha1(path.read_bytes()).hexdigest()[:10]
                self._ref_hash[str(path)] = h
            return path, f"{seg['speaker']}:canonical:{h}"
        return None

    # -- synthesis ---------------------------------------------------------
    def _cache_key(self, speak: str, ref_key: str, seed: int, greedy: bool,
                   opts: TtsOpts = ttsopts.DEFAULT, tgt: str | None = None,
                   synth: str | None = None) -> str:
        """Everything that changes the audio, hashed. Nothing that changes it may
        stay out of here, or an edited segment silently replays its old clip.

        The blob is append-only by construction: the model tag, target language,
        text, reference key, clone mode, seed and greedy flag occupy exactly the
        positions they always did, and per-segment options add a suffix only when
        they are set. A segment with no `tts_opts` therefore hashes to the same
        key it did before this existed, and every cached clip stays valid.

        `tgt` is *this segment's* target, not the run's: a segment carrying its own
        `tgt_lang` is different audio of the same words, and an fr clip that
        collided with an en one would be replayed as English forever.

        A Hebrew-synthesised clip adds one more suffix, for the two things that
        make its audio: the adapter (a different adapter is a different voice
        model) and the IPA that was actually synthesized, which `speak` alone does
        not pin down a G2P change would otherwise replay clips of the old
        pronunciation. It hangs off the *call's* language, so an English line
        voiced inside a Hebrew run (adapter disabled the base model exactly)
        keeps the key it would have had anywhere else.
        """
        # The target language is part of the key so a ru clip never collides with
        # an en clip of the same text. "en" is left out of the blob to keep every
        # existing English cache entry valid.
        tgt = (tgt or self.tgt_lang).lower()
        lang = "" if tgt == "en" else f"|{tgt}"
        blob = (f"{self.model_tag_for(opts)}{lang}|{speak}|{ref_key}"
                f"|{opts.clone_mode()}|{seed}|{int(greedy)}")
        extra = opts.cache_suffix()
        if extra:
            blob += f"|{extra}"
        if self.hebrew_for(tgt):
            ipa = hashlib.sha1((synth or "").encode("utf-8")).hexdigest()[:10]
            blob += f"|{hebrew_mod.ADAPTER_TAG}:{ipa}"
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]

    def _verify(self, clip: Path, speak: str, tgt: str | None = None,
                src: str | None = None) -> tuple[bool, float, str]:
        tgt = (tgt or self.tgt_lang).lower()
        src = (src or self.src_lang).lower()
        try:
            sec = audio.duration(clip)
        except Exception:
            return False, 0.0, "unreadable"
        if not clone_length_ok(sec, speak, tgt):
            return False, 0.0, f"len={sec:.2f}s"
        model = self.asr_for(tgt)
        if model is None:
            # Accepted, but not verified and `_verdict` turns that into a
            # verdict of its own rather than letting it read as a pass.
            return True, 1.0, NO_ASR
        try:
            segs, _ = model.transcribe(str(clip), language=tgt,
                                       word_timestamps=False,
                                       condition_on_previous_text=False, vad_filter=True)
            heard = " ".join((s.text or "").strip() for s in segs).strip()
        except Exception as exc:
            return False, 0.0, f"asr-error {exc}"
        if source_script_leak(heard, src, tgt):
            return False, 0.0, "source-script output"
        ov = word_overlap(speak, heard, tgt)
        return ov >= CLONE_MIN_OVERLAP, ov, heard[:120]

    def tgt_for(self, seg: dict[str, Any]) -> str:
        """This segment's target language its own override, else the run's.

        Same rule as `edit._langs`, which is what the translator obeys: if the two
        disagreed, the line would be written in one language and spoken in another.
        Spelled the way the run's own target is (`__init__`): "iw" is Hebrew's
        legacy code and reaches the verify ASR's `language=` as a rejected value.
        """
        tgt = (seg.get("tgt_lang") or self.tgt_lang or "en").lower()
        return "he" if hebrew_mod.is_hebrew(tgt) else tgt

    def src_for(self, seg: dict[str, Any]) -> str:
        """This segment's source language its own override, else the run's."""
        return (seg.get("src_lang") or seg.get("lang") or self.src_lang or "he").lower()

    def hebrew_for(self, tgt: str | None = None) -> bool:
        """True when a call in `tgt` is synthesised through the Hebrew LoRA.

        Two conditions, because the adapter is a *run*-level load (it decides which
        checkpoint is fetched, `model_for`) while the language is per segment:
        this run attached it, and this call speaks Hebrew. A Hebrew segment inside
        an English run therefore gets the base model and its own orthography —
        degraded, but never IPA handed to a model that never learned to read it.
        """
        return self.hebrew and hebrew_mod.is_hebrew(tgt or self.tgt_lang)

    def _plan(self, seg: dict[str, Any], text_en: str) -> Plan | None:
        """Everything an attempt needs, or None if the segment cannot be voiced.

        Raises `ValueError` on an unusable `tts_opts` a mistyped option is a
        mistake to show the user, not something to quietly synthesize around.
        """
        opts = ttsopts.parse(seg.get("tts_opts"))
        tgt, src = self.tgt_for(seg), self.src_for(seg)
        speak = prepare_text(text_en, tgt)
        if not speak or len(_tokens(speak, tgt)) == 0:
            return None
        synth = synthesis_text(speak, tgt) if self.hebrew_for(tgt) else speak
        if not synth.strip():
            return None            # nothing came back from the G2P to say
        ref = self.ref_for(seg, opts)
        if ref is None:
            return None
        ref_path, ref_key = ref
        return Plan(speak, synth, ref_path, ref_key, seed_for(seg, speak, opts),
                    opts, tgt, src, self._greedy_for(seg, opts, tgt))

    def _greedy_for(self, seg: dict[str, Any], opts: TtsOpts, tgt: str) -> bool:
        """Whether this segment's first attempt decodes deterministically.

        A Hebrew target defaults to greedy. The LoRA speaks IPA and the sampler is
        where its wandering and repetition come from; greedy is the steadier read.
        This is only an honest default because the retry ladder above varies the
        *reference* — with a seed-varying ladder, forcing greedy meant a segment
        got one take and no retries at all, which was strictly worse.

        The segment always wins: an explicit `greedy` either way is obeyed, and so
        is a sampler setting, which `ttsopts.parse` only stores when greedy is off
        and which would be silently discarded by a forced greedy decode.
        """
        if opts.greedy or not self.hebrew_for(tgt):
            return opts.greedy
        raw = seg.get("tts_opts")
        if isinstance(raw, dict) and "greedy" in raw:
            return False        # the user asked for a sampled decode on this line
        return not any(getattr(opts, k) is not None for k in ttsopts.SAMPLING_KEYS)

    def _attempt(self, seg_id: int, speak: str, ref_path: Path, ref_key: str,
                 seed: int, greedy: bool, opts: TtsOpts = ttsopts.DEFAULT,
                 tgt: str | None = None, synth: str | None = None):
        """Generate (or load-from-cache) one rung's clip.

        Returns (clip, meta, verdict). `verdict` is None when the clip was just
        generated and still needs verifying; a dict when it was cached; and
        {"failed": True} when generation raised. Isolating this from verification
        is what lets the pipeline in `run` verify one clip while the GPU makes the
        next the seed/cache scheme is identical to the sequential path.

        The seed is the caller's and does not move between rungs. It used to be
        `base_seed + 1000 * attempt`, which read like a re-roll and was one only
        while the decode sampled: under `do_sample=False` the seed does nothing,
        so a greedy ladder was MAX_TRIES byte-identical takes filed under
        different cache keys. What varies a rung now is the reference or the
        decode, both of which are in the key on their own.
        """
        synth = speak if synth is None else synth
        key = self._cache_key(speak, ref_key, seed, greedy, opts, tgt, synth)
        clip = self.clips / f"{key}.wav"
        meta = self.clips / f"{key}.json"
        if clip.is_file() and meta.is_file():
            return clip, meta, json.loads(meta.read_text(encoding="utf-8"))
        try:
            self.synth_for(opts).generate(speak, ref_path, clip, seed=seed, greedy=greedy,
                                          opts=opts, synth=synth, lang=tgt)
        except Exception as exc:
            print(f"  tts: seg {seg_id} generate failed ({exc})", file=sys.stderr)
            return clip, meta, {"failed": True}
        audio.trim_leading_silence(clip, clip)
        return clip, meta, None

    def _verify_and_store(self, clip: Path, meta: Path, speak: str,
                          tgt: str | None = None, src: str | None = None
                          ) -> dict[str, Any]:
        verdict = _verdict(clip, *self._verify(clip, speak, tgt, src))
        meta.write_text(json.dumps(verdict, ensure_ascii=False), encoding="utf-8")
        return verdict

    @staticmethod
    def _record(clip: Path, verdict: dict[str, Any], attempt: int,
                opts: TtsOpts = ttsopts.DEFAULT, text: str = "") -> dict[str, Any]:
        # "unverified" is not "ok": the clip cleared the length guard and no ASR
        # ever heard it. A cached verdict from before this existed has no
        # `verified` key and was written by a run that did have an ASR, so it
        # reads as verified every existing record stays exactly as it was.
        rec = {"clip": f"clips/{clip.name}", "dur": verdict["dur"],
               "tries": attempt + 1, "overlap": verdict["overlap"],
               "verify": "ok" if verdict.get("verified", True) else "unverified"}
        # Which options this clip was made under, so `run` can spot a segment whose
        # options changed under an otherwise-usable clip. Absent for the defaults,
        # which keeps every existing record byte-identical.
        fp = opts.fingerprint()
        if fp:
            rec["opts"] = fp
        # Which line this clip actually says, so a translate rerun cannot leave it
        # behind claiming to be current (`clip_text_stale`).
        if (text or "").strip():
            rec["text_sha"] = text_sha(text)
        return rec

    def rungs(self, seg: dict[str, Any], plan: Plan) -> list[tuple[Path, str, bool]]:
        """The ladder this segment climbs: (reference, ref key, greedy) per attempt.

        Rung 0 is the segment's own aligned window under the plan's decode. Rungs
        1+ swap the *reference* — the speaker's canonical clip first, then the
        runners-up — with the seed and the decode held fixed, because the
        reference is the one ingredient that reliably changes a clone.

        The last rung changes the decode instead, on the original reference. Under
        a sampled plan that is the deterministic final take this stage has always
        ended on. Under a greedy plan it is the one *sampled* take — the seg-27
        rescue: a greedy ladder cannot re-roll (the seed is inert), so when every
        reference has been tried and none reached CLONE_GOOD_OVERLAP, letting the
        decode sample once is the only new take left to make.
        """
        out = [(plan.ref_path, plan.ref_key, plan.greedy)]
        out += [(p, k, plan.greedy) for p, k in
                self._ref_ladder(seg, plan.opts, plan.ref_key)]
        out.append((plan.ref_path, plan.ref_key, not plan.greedy))
        return out

    def clip_for(self, seg: dict[str, Any], text_en: str) -> dict[str, Any] | None:
        """Bounded retry ladder. Returns a tts record, or None to fall back to keep.

        Two bars decide when it stops. A take at or above `CLONE_GOOD_OVERLAP` is
        "ok" and ends the ladder. A take merely above `CLONE_MIN_OVERLAP` is
        usable but not good: it is remembered and the ladder keeps going, and if
        nothing better turns up it is returned as "accepted" — a real dub, said
        out loud in report.json to be a second-class one. Below the floor the best
        take is soft-accepted, or the segment falls back to its original audio.
        """
        plan = self._plan(seg, text_en)
        if plan is None:
            return None
        speak, synth, ref_path, ref_key, base_seed, opts, tgt, src, greedy = plan
        slot = float(seg["end"]) - float(seg["start"])
        best: dict[str, Any] | None = None

        for attempt, (a_path, a_key, a_greedy) in enumerate(self.rungs(seg, plan)):
            clip, meta, verdict = self._attempt(seg["id"], speak, a_path, a_key,
                                                base_seed, a_greedy, opts, tgt, synth)
            if verdict is not None and verdict.get("failed"):
                continue
            if verdict is None:
                verdict = self._verify_and_store(clip, meta, speak, tgt, src)
            # A clip the timeline could never absorb (shorten + 1.3x speed-up
            # recover maybe 3x; a 71s clip for a 5.7s slot once shoved everything
            # after it 49s late) is a failure however well it verifies.
            if clip_exceeds_slot(verdict["dur"], slot):
                print(f"  tts: seg {seg['id']} clip {verdict['dur']:.1f}s vs "
                      f"{slot:.1f}s slot rejected", file=sys.stderr)
                continue
            record = self._record(clip, verdict, attempt, opts, text_en)
            if clip_is_good(verdict):
                return record
            if verdict["overlap"] > (best or {}).get("overlap", -1.0):
                best = dict(record, verify="accepted" if verdict["ok"] else "soft")

        if best and best["verify"] == "accepted":
            print(f"  tts: seg {seg['id']} accepted below {CLONE_GOOD_OVERLAP:.2f} "
                  f"(overlap {best['overlap']:.2f}) every reference was tried",
                  file=sys.stderr)
            return best
        if best and best["overlap"] >= CLONE_SOFT_OVERLAP:
            print(f"  tts: seg {seg['id']} soft-accept (overlap {best['overlap']:.2f})",
                  file=sys.stderr)
            return best
        return None

    def keep_clip(self, seg: dict[str, Any]) -> dict[str, Any]:
        """Original audio for this span the universal fallback.

        Named by the span itself, never by segment id: ids are renumbered
        whenever segmentation changes, and an id-keyed cache then hands back a
        clip of the wrong length for the wrong moment.

        `tts_opts.speed` applies here too, through the same atempo path a dub
        uses: it is post-processing on finished audio, and a keep is finished
        audio. It used to be accepted on a kept segment and quietly do nothing.
        The other options cannot apply there is no synthesis call to send them
        to and `edit.set_tts_opts` refuses them on a keep rather than storing a
        knob that does nothing (`keep_pauses` is the exception that is already
        true: a keep's silences are the original's, untouched, which is exactly
        what it asks for).

        The record carries the span it was cut for. The clip's *duration* is no
        longer that span once a tempo is applied, and `timeline.build_items`
        needs to know it is looking at this segment's audio the recorded span
        is what it checks.
        """
        opts = ttsopts.parse(seg.get("tts_opts"))
        span = round(float(seg["end"]) - float(seg["start"]), 3)
        fp = opts.fingerprint()
        blob = f"{seg['start']:.3f}|{seg['end']:.3f}" + (f"|{fp}" if fp else "")
        key = hashlib.sha1(blob.encode()).hexdigest()[:12]
        path = self.clips / f"keep_{key}.wav"
        want = span / opts.speed
        if not path.is_file() or abs(audio.duration(path) - want) > 0.1:
            source = self.workdir / self.m["files"]["source_wav"]
            if opts.speed == 1.0:
                audio.extract_slice(source, seg["start"], seg["end"], path)
            else:
                raw = path.with_suffix(".raw.wav")
                audio.extract_slice(source, seg["start"], seg["end"], raw)
                audio.atempo(raw, path, opts.speed)
                raw.unlink(missing_ok=True)
        rec = {"clip": f"clips/{path.name}", "dur": round(audio.duration(path), 3),
               "tries": 0, "overlap": 1.0, "verify": "keep", "span": span}
        if fp:
            rec["opts"] = fp
        return rec

    def close(self) -> None:
        if self._synth is not None:
            self._synth.free()
            self._synth = None
        self._synth_model = None
        self._voc = None
        hebrew_mod.free()


class _Synth:
    """Minimal Qwen3-TTS wrapper: x-vector-only cloning, one call per clip.

    A Hebrew target attaches the Hebrew LoRA to the loaded checkpoint's `talker`.
    That is an addition, not a swap: with the adapter disabled the forward pass is
    the unmodified base model's, so this one object still speaks all ten of the
    checkpoint's own languages `generate` picks per call (see `_adapter`).
    """

    def __init__(self, *, device: str | None = None, model: str = DEFAULT_TTS_MODEL,
                 lang: str = "en"):
        import torch

        self.lang = (lang or "en").lower()
        self.hebrew = hebrew_mod.is_hebrew(self.lang)
        self._qwen_langs: dict[str, str] = {}  # language code → the checkpoint's name

        if device and device != "auto":
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda:0"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        # float16 on MPS hits NaNs in the code predictor sampler.
        self.dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        spec = TTS_MODELS[model if model in TTS_MODELS else DEFAULT_TTS_MODEL]
        local = REPO_ROOT / "models" / spec["dir"]
        self.path = str(local) if local.is_dir() else spec["hub"]
        self._model = None
        self._prompts: dict[str, Any] = {}

    def _load(self):
        import contextlib
        import io

        import torch

        # qwen_tts eagerly imports its 25Hz tokenizer, whose module prints an
        # asterisk banner telling non-CUDA users to install flash-attn (which
        # has no macOS build, and this pipeline only loads 12Hz checkpoints).
        # Swallow stdout for just the import; real import errors still raise.
        with contextlib.redirect_stdout(io.StringIO()):
            from qwen_tts import Qwen3TTSModel

        if self._model is not None:
            return self._model
        print(f"  tts: loading {self.path} on {self.device} ({self.dtype})", file=sys.stderr)
        kwargs: dict[str, Any] = {"dtype": self.dtype, "low_cpu_mem_usage": False}
        if self.device.startswith("cuda"):
            kwargs["attn_implementation"] = "sdpa"
            kwargs["device_map"] = self.device
        else:
            kwargs["device_map"] = "cpu"
        model = Qwen3TTSModel.from_pretrained(self.path, **kwargs)
        if not self.device.startswith("cuda"):
            model.model.to(self.device)
            tok = getattr(model.model, "speech_tokenizer", None)
            if tok is not None and hasattr(tok, "to"):
                tok.to(self.device)
            model.device = torch.device(self.device)
        if self.hebrew:
            # After the device placement, so the LoRA lands where its base layers
            # are; cast to the checkpoint's dtype because the adapter ships bf16
            # and the base runs float32 off CUDA (float16/bf16 NaNs on MPS).
            model.model.talker = hebrew_mod.attach_adapter(model.model.talker)
            model.model.talker.to(device=self.device, dtype=self.dtype)
            model.model.eval()
            print(f"  tts: Hebrew adapter attached ({hebrew_mod.ADAPTER_DIR})",
                  file=sys.stderr)
        self._model = model
        return model

    def _adapter(self, model, lang: str):
        """Context in which the Hebrew LoRA is enabled for `lang`, or disabled.

        Disabled is the base model exactly that is the adapter's whole contract,
        and it is what lets one loaded checkpoint serve a Hebrew line and an English
        one. A synth that never attached the adapter has nothing to toggle and gets
        a no-op either way.
        """
        import contextlib

        if not self.hebrew or hebrew_mod.is_hebrew(lang):
            return contextlib.nullcontext()
        return model.model.talker.disable_adapter()

    def _language(self, model, lang: str | None = None) -> str:
        """The checkpoint's name for a target language, resolved once per language.

        Per language, not per run: a segment may target its own (`seg["tgt_lang"]`),
        and synthesising a French line as English is how a supported override turned
        into a permanent tts_failed.
        """
        lang = (lang or self.lang or "en").lower()
        if hebrew_mod.is_hebrew(lang):
            # Not a gap to warn about: the adapter was trained and sampled with
            # language="Auto", and the IPA in the text field carries the phonetics.
            return "Auto"
        got = self._qwen_langs.get(lang)
        if got is None:
            try:
                supported = model.get_supported_languages()
            except Exception:
                supported = _QWEN_LANG_NAMES.values()
            got = self._qwen_langs[lang] = qwen_language_name(lang, supported)
            if got == "Auto":
                print(f"  tts: target {lang!r} not in the checkpoint's supported "
                      "languages synthesising with language=Auto", file=sys.stderr)
        return got

    def generate(self, speak: str, ref: Path, out: Path, *, seed: int, greedy: bool,
                 opts: TtsOpts = ttsopts.DEFAULT, synth: str | None = None,
                 lang: str | None = None) -> Path:
        """`speak` is the line in its own script; `synth` is what the model is told.

        They differ only for Hebrew, where `synth` is the stressed IPA. The token
        budget is still measured on `speak`: IPA keeps the word boundaries, and the
        orthography is the text every other length rule in this module reads.

        `lang` names the language of this one call; it defaults to the run's target
        and exists so a segment with its own `tgt_lang` is spoken in that language,
        and so a Hebrew-loaded synth can also voice a base-language line which it
        does with the adapter switched off, i.e. as the plain base model.
        """
        import torch

        model = self._load()
        lang = (lang or self.lang).lower()
        text = synth if synth is not None else speak
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        # ICL mode conditions on the reference's codes AND its transcript, so the
        # prompt differs per (reference, ref_text) pair, not per reference alone.
        key = (str(ref), opts.ref_text)
        if key not in self._prompts:
            self._prompts[key] = model.create_voice_clone_prompt(
                ref_audio=str(ref), ref_text=opts.ref_text,
                x_vector_only_mode=not opts.icl,
            )
        with self._adapter(model, lang):
            wavs, sr = model.generate_voice_clone(
                text=text, language=self._language(model, lang),
                voice_clone_prompt=self._prompts[key],
                max_new_tokens=opts.max_new_tokens or max_new_tokens(speak, lang),
                repetition_penalty=opts.repetition_penalty or REPETITION_PENALTY,
                **sampling_kwargs(greedy, opts),
            )
        wav = np.asarray(wavs[0], dtype=np.float32)
        if not opts.keep_pauses:
            wav = _trim_internal_silence(wav, sr)
        out.parent.mkdir(parents=True, exist_ok=True)
        raw = out.with_suffix(".raw.wav")
        sf.write(str(raw), wav, sr)
        # The tempo change is baked in here, before verification, so the clip that
        # gets length-checked and ASR-checked is the clip that gets placed.
        tempo = (["-af", audio.atempo_chain(opts.speed)] if opts.speed != 1.0 else [])
        audio.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), *tempo,
                   "-acodec", "pcm_s16le", "-ar", str(audio.SR), "-ac", "1", str(out)])
        raw.unlink(missing_ok=True)
        return out

    def free(self) -> None:
        import gc

        import torch

        self._prompts.clear()
        if self._model is not None:
            try:
                self._model.model.to("cpu")
            except Exception:
                pass
            self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


_ASR_CANDIDATES = ("models/faster-whisper-base.en", "Systran/faster-whisper-base.en",
                   "models/faster-whisper-tiny.en", "Systran/faster-whisper-tiny.en")
# Non-English targets verify with the multilingual base model instead of the .en ones.
_ASR_CANDIDATES_MULTI = ("models/faster-whisper-base", "Systran/faster-whisper-base")


def _asr_candidates(tgt: str) -> tuple[str, ...]:
    """Which ASR verifies a clip in `tgt` the source stage's constants, reused.

    Hebrew gets the ivrit-ai fine-tune the transcript stage already reads Hebrew
    with. The multilingual base model transcribes Hebrew badly enough that the
    word-overlap check would fail good clips, which under "never silent" means a
    correct dub thrown away for the original audio.
    """
    from . import transcript

    tgt = (tgt or "en").lower()
    if tgt == "en":
        return _ASR_CANDIDATES
    if hebrew_mod.is_hebrew(tgt):
        return (str(transcript.WHISPER_MODEL.relative_to(REPO_ROOT)),
                transcript.WHISPER_HUB) + _ASR_CANDIDATES_MULTI
    return _ASR_CANDIDATES_MULTI
VERIFY_CPU_THREADS = 2   # Mac-only cap on the verify ASR's CPU threads. It runs on a
                         # worker thread while the next clip generates on the GPU (see
                         # run()); left uncapped on Apple Silicon, CTranslate2 grabs
                         # every core and starves the MPS dispatch thread, so the
                         # overlap would slow generation instead of hiding the verify
                         # behind it. Elsewhere the CPU fallback uses half the cores.


def _verify_cpu_threads() -> int:
    """Thread budget for the verify ASR's CPU fallback (see VERIFY_CPU_THREADS)."""
    if sys.platform == "darwin":
        return VERIFY_CPU_THREADS
    import os

    return max(1, (os.cpu_count() or 4) // 2)


def _load_asr(tgt: str = "en"):
    from .transcript import load_whisper

    for cand in _asr_candidates(tgt):
        local = REPO_ROOT / cand
        target = str(local) if local.is_dir() else cand
        try:
            model = load_whisper(target, label="tts: verify ASR",
                                 cpu_threads=_verify_cpu_threads())
            print(f"  tts: verifying with {target}", file=sys.stderr)
            return model
        except Exception:
            continue
    print("  tts: no verification ASR accepting on length only", file=sys.stderr)
    return None


def clear_failed_keeps(segments: list[dict[str, Any]]) -> list[int]:
    """Make every previously tts_failed segment dubbable again.

    A tts_failed keep is this stage's own per-run verdict, never cached state:
    whatever run set it, entering the stage again means re-attempting synthesis.
    `manifest.reset_stage` already undoes these on the invalidation path, but the
    resume path (same fingerprint after an interruption) skips reset_stage, and
    without this a resumed run would treat the failure as settled forever.
    Returns the ids that were cleared.

    Both locks are honoured, through the same predicate the other two undo paths
    use (`manifest.undo_pipeline_keep`): `locked.keep` is the user answering the
    verdict ("yes, keep the original here") and `locked.tts` is the user approving
    the slice it produced. This function used to guard only the second, so a
    user-locked keep was silently re-decided on every single tts run while
    `reset_stage` and `edit.invalidate` left it alone.
    """
    from . import manifest

    cleared: list[int] = []
    for seg in segments:
        if manifest.is_locked(seg, "tts"):
            continue          # the user approved this clip; a rerun does not replace it
        if manifest.undo_pipeline_keep(seg, manifest.PIPELINE_KEEPS["tts"]):
            seg.pop("tts", None)  # the keep-clip record; this run re-decides
            cleared.append(seg["id"])
    return cleared


def has_clip(seg: dict[str, Any], workdir: Path) -> bool:
    """True when this segment already owns audio on disk."""
    return bool(seg.get("tts")) and (workdir / seg["tts"]["clip"]).is_file()


def keep_needs_slice(seg: dict[str, Any], workdir: Path) -> bool:
    """True when this kept segment needs its original-audio slice (re)cut.

    The mirror of `needs_synthesis` for the other path. A keep's options are
    fingerprinted onto its record too, so setting `tts_opts.speed` on a kept
    segment re-cuts the slice instead of leaving the old one in place looking
    like the edit took.
    """
    if not has_clip(seg, workdir):
        return True
    return (seg["tts"].get("opts", "")
            != ttsopts.parse(seg.get("tts_opts")).fingerprint())


def pending(segments: list[dict[str, Any]], workdir: Path) -> list[dict[str, Any]]:
    """The dubbed segments this stage must synthesize.

    A segment with a usable clip is done that is the resume mechanism, and the
    editor's per-segment redo is the same deletion (`edit.invalidate`).

    That is also where a `locked` clip is safe: nothing but the user's own edit
    deletes a locked record (`manifest.reset_stage`, `clear_failed_keeps` and
    `edit.invalidate` all skip it), so a locked clip is by definition still on disk
    and never lands here. A lock whose clip is *gone* is unhonorable never-silent
    outranks it and that segment is synthesized again.
    """
    return [s for s in segments if not s.get("keep") and needs_synthesis(s, workdir)]


def run(m: dict[str, Any], workdir: Path, *, save=None, device: str | None = None,
        model: str = DEFAULT_TTS_MODEL) -> Engine:
    from . import manifest

    engine = Engine(m, workdir, device=device, model=model)
    # This run's own excuses only: whatever the last one could not load may well
    # be there now, and a stale note is a lie of the same family as a missing one.
    for key in HEALTH_KEYS:
        (m.get("health") or {}).pop(key, None)
    clear_failed_keeps(m["segments"])
    # A clip the user approved for a line that has since changed. It is not this
    # stage's to replace and not this stage's to pass off as current, so it is
    # named here and counted in report.json (`stale_locked_clips`).
    for seg in m["segments"]:
        if stale_locked_clip(seg):
            print(f"  tts: seg {seg['id']} locked clip was made for different text "
                  "left alone; it still speaks the old line", file=sys.stderr)
    engine.build_speaker_refs()
    todo = pending(m["segments"], workdir)

    # Generation runs on the GPU, verification (Whisper) on the CPU. Generate each
    # first-attempt clip in order and hand its verify to a single worker thread, so
    # it overlaps the *next* clip's generation instead of stalling the GPU. One
    # worker only: faster-whisper is not re-entrant, and this already hides all but
    # the last clip's verify. Segments whose first attempt fails (rare) fall through
    # to clip_for, which resumes at attempt 1 straight from the cache.
    import concurrent.futures as cf

    verifying: dict[int, tuple] = {}
    retry: list[dict[str, Any]] = []
    done = 0
    with cf.ThreadPoolExecutor(max_workers=1) as vpool:
        for seg in todo:
            text = speakable(seg)
            if text is None:
                # mt_failed on a user-locked dub: nothing to speak. The line
                # plays its original slice exactly what the editor promises
                # of a failed translation and stays failed until the user
                # re-translates it.
                #
                # `keep_needs_slice` alone is the wrong guard here: it answers
                # False for ANY clip on disk with matching opts — including
                # the segment's old *synthesis* of a translation mark_failed
                # has since popped. That clip would then stay in place
                # speaking the previous line while this print claims the
                # original plays. The record must already BE a keep slice for
                # the skip to be honest.
                rec_clip = str((seg.get("tts") or {}).get("clip") or "")
                is_slice = rec_clip.rsplit("/", 1)[-1].startswith("keep_")
                if not is_slice or keep_needs_slice(seg, workdir):
                    seg["tts"] = engine.keep_clip(seg)
                print(f"  tts: seg {seg['id']} has no translation the original "
                      "plays until it is re-translated", file=sys.stderr)
                continue
            plan = engine._plan(seg, text)
            if plan is None:
                retry.append(seg)
                continue
            speak, synth, ref_path, ref_key, base_seed, opts, tgt, src, greedy = plan
            clip, meta, verdict = engine._attempt(seg["id"], speak, ref_path, ref_key,
                                                  base_seed, greedy, opts, tgt, synth)
            if verdict is not None and verdict.get("failed"):
                retry.append(seg)
                continue
            if verdict is not None:                    # cache hit no verify needed
                # `clip_is_good`, not `ok`: a first take that merely clears the
                # accept floor is exactly the garble this fast path used to store
                # as "ok" and never look at again. It goes to the ladder, which
                # will hand this same cached clip back if nothing beats it.
                if clip_is_good(verdict):
                    seg["tts"] = engine._record(clip, verdict, 0, opts, seg["text_en"])
                else:
                    retry.append(seg)
                continue
            verifying[seg["id"]] = (vpool.submit(engine._verify, clip, speak, tgt, src),
                                    clip, meta, speak, opts)

        for seg in todo:
            item = verifying.get(seg["id"])
            if item is None:
                continue
            fut, clip, meta, speak, opts = item
            verdict = _verdict(clip, *fut.result())
            meta.write_text(json.dumps(verdict, ensure_ascii=False), encoding="utf-8")
            if clip_is_good(verdict):
                seg["tts"] = engine._record(clip, verdict, 0, opts, seg["text_en"])
            else:
                retry.append(seg)
            done += 1
            if done % 10 == 0:
                print(f"  tts: {done}/{len(todo)}", file=sys.stderr)
                if save:
                    save()

    # Slow path: attempts 1+ for the few that failed their first try (needs the GPU
    # again, so it runs after the pipelined pass rather than stalling it).
    for seg in retry:
        record = engine.clip_for(seg, seg["text_en"])
        if record is None:
            seg["keep"], seg["keep_reason"] = True, "tts_failed"
            print(f"  tts: seg {seg['id']} unusable → keep original", file=sys.stderr)
        else:
            seg["tts"] = record
    if save:
        save()

    # Every keep gets its original-audio slice that is the never-silent floor.
    for seg in m["segments"]:
        if not seg["keep"]:
            continue
        if manifest.is_locked(seg, "tts") and has_clip(seg, workdir):
            continue          # the user approved this slice; a rerun does not re-cut it
        if keep_needs_slice(seg, workdir):
            seg["tts"] = engine.keep_clip(seg)
    missing = [s["id"] for s in m["segments"] if not s.get("tts")]
    assert not missing, f"segments without audio: {missing}"
    return engine
