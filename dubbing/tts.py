"""Stage 6 — Qwen3-TTS zero-shot voice cloning, one clip per segment.

Two guarantees this module owns:

* **Never silent.** Every segment leaves this stage with an audio clip. A dub
  that cannot be verified after its bounded retries falls back to the original
  audio slice (`keep_reason="tts_failed"`), never to nothing.
* **Never partial.** A segment is one TTS call for its whole text. There is no
  sub-unit splitting, so there is no way for part of a line to go missing.

Clips are cached on disk by content hash, which is also the resume mechanism for
long videos: a killed run picks up at the first uncached clip.

Per-segment overrides live in `seg["tts_opts"]` (see `dubbing/ttsopts.py`): seed,
forced-greedy decode, a pinned clone reference, model size, tempo, the sampler.
Every one of them is mixed into the clip's cache key, so changing one on an
already-dubbed segment cannot replay the old clip.

**Natural-language style instructions are not available on this path.** The
checkpoint exposes `instruct` only through `generate_voice_design` /
`generate_custom_voice`, which are gated on `tts_model_type` being `voice_design`
/ `custom_voice` and take no reference audio at all — they invent a voice from
the description instead of cloning the speaker, which is the one thing this
pipeline may not lose. The closest thing the Base checkpoint has is ICL mode
(`tts_opts.ref_text`): conditioning on a reference's text *and* codes rather than
only its speaker embedding, which carries the reference clip's prosody. That is
a "sound like this take", not "sound angry" — and it is the honest limit here.
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
from . import script as script_mod
from . import ttsopts
from .script import count_letters, same_script, script_for
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

# Default sampler for a non-greedy attempt — the values `tts_opts` overrides.
SAMPLED = {"temperature": 0.55, "top_p": 0.85, "top_k": 30}
GREEDY = {"temperature": 0.01, "top_p": 1.0, "top_k": 1}
REPETITION_PENALTY = 1.08

REF_SR = 24000
REF_TARGET_SEC = 4.5
MIN_REF_SEC = 2.5    # a clone reference shorter than this yields truncated/garbled clones
REF_MIN_RMS = 0.018
REF_MAX_NOISE = 8.0
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

CLONE_MIN_SEC_PER_WORD = 0.18   # faster than this is chipmunk garble
CLONE_MAX_SEC_PER_WORD = 0.95   # slower than this is a stall/drawl
CLONE_MIN_OVERLAP = 0.35        # accept
CLONE_SOFT_OVERLAP = 0.20       # best-effort accept
MAX_TRIES = 3

_LATIN_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9''\-.]*")
_CHROME = re.compile(r"[\[\(][^\]\)]{0,40}[\]\)]")
# CJK punctuation → the ASCII marks prepare_text understands.
_CJK_PUNCT = str.maketrans("。，！？；：、", ".,!?;:,")

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
    the line — the old Latin-only version erased entire non-Latin translations.
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
        # No reliable word boundaries — compare single characters instead.
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


def _speech_units(text: str, lang: str) -> int:
    """How many speakable units the text has: words, or characters for CJK/hangul."""
    if script_for(lang) in ("cjk", "hangul"):
        return max(1, sum(1 for ch in text or "" if ch.isalnum()))
    return max(1, len((text or "").split()))


# CJK/hangul speech runs ~5 characters/s; the word constants assume ~3 words/s.
CLONE_MIN_SEC_PER_CHAR = 0.08   # faster than this is chipmunk garble
CLONE_MAX_SEC_PER_CHAR = 0.60   # slower than this is a stall/drawl


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
    units = _speech_units(text, lang)
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
    units = _speech_units(text, lang)
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
        print(f"  tts: speaker embeddings unavailable ({exc}) — "
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
    at least `coherent` — i.e. the rest agree on one voice and this window doesn't
    match it (a diarization mislabel). A speaker whose windows are all mutually
    dissimilar (noisy/whispered material) keeps everything: there is no consensus
    voice to reject against. Thresholds are measured — see REF_SIM_OUTLIER above.
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
    simply everything there is — possibly one short window, the old behaviour.
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


def seed_id(seg: dict[str, Any]) -> str:
    """The identity the derived seed keys on — stable across re-segmentation.

    `seg["id"]` is positional: inserting one segment renumbers everything after
    it, which silently re-rolls every later line's voice take. `uid` (minted once
    per segment, see APP_ARCHITECTURE.md) does not move, so it is preferred where
    it exists. Falling back to `id` keeps every pre-uid manifest on exactly the
    seeds — and therefore exactly the cached clips — it already had.
    """
    return str(seg.get("uid") or seg["id"])


def seed_for(seg: dict[str, Any], speak: str, opts: TtsOpts = ttsopts.DEFAULT) -> int:
    """The base seed for a segment: the user's, else derived from identity + text."""
    if opts.seed is not None:
        return opts.seed
    return int(hashlib.sha1(f"{seed_id(seg)}|{speak}".encode()).hexdigest()[:8], 16)


class Plan(NamedTuple):
    """What one segment's attempts are made from (see `Engine._plan`)."""

    speak: str
    ref_path: Path
    ref_key: str
    base_seed: int
    opts: TtsOpts


def needs_synthesis(seg: dict[str, Any], workdir: Path) -> bool:
    """True when this dubbed segment has no clip that still matches its options.

    A usable clip on disk is normally the whole resume test. It is not enough
    once options exist: editing `tts_opts` on an already-dubbed line leaves the
    old clip in place, and without this the re-run would look successful and
    change nothing. The record carries the option fingerprint it was made with,
    so a mismatch puts the segment back in the queue.
    """
    rec = seg.get("tts")
    if not rec or not (workdir / rec["clip"]).is_file():
        return True
    return rec.get("opts", "") != ttsopts.parse(seg.get("tts_opts")).fingerprint()


class Engine:
    """Holds the loaded models and the reference bank for one run."""

    def __init__(self, m: dict[str, Any], workdir: Path, *, device: str | None = None,
                 model: str = DEFAULT_TTS_MODEL):
        self.m = m
        src = m.get("source") or {}
        self.src_lang = (src.get("src_lang") or "he").lower()
        self.tgt_lang = (src.get("tgt_lang") or "en").lower()
        self.workdir = workdir
        self.clips = workdir / "clips"
        self.refs = workdir / "refs"
        self.clips.mkdir(parents=True, exist_ok=True)
        self.refs.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.tts_model = model if model in TTS_MODELS else DEFAULT_TTS_MODEL
        self.model_tag = TTS_MODELS[self.tts_model]["tag"]
        self._synth = None
        self._synth_model: str | None = None
        self._asr = None
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
        """Which checkpoint this segment uses — its own override, else the run's."""
        return opts.model or self.tts_model

    def model_tag_for(self, opts: TtsOpts = ttsopts.DEFAULT) -> str:
        return TTS_MODELS[self.model_for(opts)]["tag"]

    def synth_for(self, opts: TtsOpts = ttsopts.DEFAULT):
        """The loaded synth for this segment's model, swapping checkpoints if asked.

        Only ever one is resident: `tts_opts.model` is a per-line escape hatch, and
        two Qwen3-TTS checkpoints in float32 would not fit alongside everything
        else (AGENTS.md device notes). Switching therefore frees the other one,
        which costs a reload — fine for the handful of lines a user overrides,
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
        if self._asr is None:
            self._asr = _load_asr(self.tgt_lang)
        return self._asr

    # -- reference audio ---------------------------------------------------
    # Bumped when the canonical-ref recipe changes: reset_stage clears segments'
    # tts records but not m["speakers"], so without this marker a manifest from
    # an older run would keep its old (possibly sub-second) reference forever.
    REF_BUILD = 2

    def build_speaker_refs(self) -> None:
        """One canonical reference per speaker: their cleanest few seconds.

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
            return cands
        keep = reject_voice_outliers(vecs)
        for c, k in zip(cands, keep):
            if not k:
                print(f"  tts: {spk} ref window {c[0]:.2f}-{c[1]:.2f} rejected "
                      "(voice outlier — likely another speaker)", file=sys.stderr)
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
        """The segment's own aligned voice — whoever actually speaks in this window.

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
        stands — the old behaviour, never a blind widen-by-label.
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
            # Name the clip by its audio window, never the segment id: ids shift
            # between runs as segmentation changes, so an id-named file goes stale
            # and clones a different moment's voice (the "1:19 voice at 1:33" bug).
            # A window-named file, and a window-based cache key, cannot.
            path = self.refs / f"ref_{start:.2f}-{end:.2f}.wav"
            if not path.is_file():
                sf.write(str(path), self.vocals[int(start * REF_SR) : int(end * REF_SR)],
                         REF_SR)
            return path, f"ref:{start:.2f}-{end:.2f}"
        # Only if the aligned window is too short or essentially silent: the
        # speaker's canonical clip, if one was built, else the synth default voice.
        return self._canonical_ref(seg)

    def _matches_canonical(self, span: tuple[float, float], canon: Path) -> bool:
        """ECAPA says the aligned window and the canonical ref are the same voice.

        False whenever embeddings are unavailable — the caller then keeps the
        segment's own audio, exactly the pre-embedding behaviour.
        """
        key = (round(span[0], 2), round(span[1], 2), str(canon))
        got = self._match_cache.get(key)
        if got is None:
            vecs = _embed_windows(self.vocals_path, [span])
            cvec = _embed_wavfile(canon)
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
                   opts: TtsOpts = ttsopts.DEFAULT) -> str:
        """Everything that changes the audio, hashed. Nothing that changes it may
        stay out of here, or an edited segment silently replays its old clip.

        The blob is append-only by construction: the model tag, target language,
        text, reference key, clone mode, seed and greedy flag occupy exactly the
        positions they always did, and per-segment options add a suffix only when
        they are set. A segment with no `tts_opts` therefore hashes to the same
        key it did before this existed, and every cached clip stays valid.
        """
        # The target language is part of the key so a ru clip never collides with
        # an en clip of the same text. "en" is left out of the blob to keep every
        # existing English cache entry valid.
        lang = "" if self.tgt_lang == "en" else f"|{self.tgt_lang}"
        blob = (f"{self.model_tag_for(opts)}{lang}|{speak}|{ref_key}"
                f"|{opts.clone_mode()}|{seed}|{int(greedy)}")
        extra = opts.cache_suffix()
        if extra:
            blob += f"|{extra}"
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]

    def _verify(self, clip: Path, speak: str) -> tuple[bool, float, str]:
        try:
            sec = audio.duration(clip)
        except Exception:
            return False, 0.0, "unreadable"
        if not clone_length_ok(sec, speak, self.tgt_lang):
            return False, 0.0, f"len={sec:.2f}s"
        model = self.asr
        if model is None:
            return True, 1.0, "no-asr"
        try:
            segs, _ = model.transcribe(str(clip), language=self.tgt_lang,
                                       word_timestamps=False,
                                       condition_on_previous_text=False, vad_filter=True)
            heard = " ".join((s.text or "").strip() for s in segs).strip()
        except Exception as exc:
            return False, 0.0, f"asr-error {exc}"
        if source_script_leak(heard, self.src_lang, self.tgt_lang):
            return False, 0.0, "source-script output"
        ov = word_overlap(speak, heard, self.tgt_lang)
        return ov >= CLONE_MIN_OVERLAP, ov, heard[:120]

    def _plan(self, seg: dict[str, Any], text_en: str) -> Plan | None:
        """Everything an attempt needs, or None if the segment cannot be voiced.

        Raises `ValueError` on an unusable `tts_opts` — a mistyped option is a
        mistake to show the user, not something to quietly synthesize around.
        """
        opts = ttsopts.parse(seg.get("tts_opts"))
        speak = prepare_text(text_en, self.tgt_lang)
        if not speak or len(_tokens(speak, self.tgt_lang)) == 0:
            return None
        ref = self.ref_for(seg, opts)
        if ref is None:
            return None
        ref_path, ref_key = ref
        return Plan(speak, ref_path, ref_key, seed_for(seg, speak, opts), opts)

    def _attempt(self, seg_id: int, speak: str, ref_path: Path, ref_key: str,
                 base_seed: int, attempt: int, opts: TtsOpts = ttsopts.DEFAULT):
        """Generate (or load-from-cache) one attempt's clip.

        Returns (clip, meta, verdict). `verdict` is None when the clip was just
        generated and still needs verifying; a dict when it was cached; and
        {"failed": True} when generation raised. Isolating this from verification
        is what lets the pipeline in `run` verify one clip while the GPU makes the
        next — the seed/cache scheme is identical to the sequential path.
        """
        greedy = opts.greedy or attempt == MAX_TRIES - 1
        seed = base_seed + 1000 * attempt
        key = self._cache_key(speak, ref_key, seed, greedy, opts)
        clip = self.clips / f"{key}.wav"
        meta = self.clips / f"{key}.json"
        if clip.is_file() and meta.is_file():
            return clip, meta, json.loads(meta.read_text())
        try:
            self.synth_for(opts).generate(speak, ref_path, clip, seed=seed, greedy=greedy,
                                          opts=opts)
        except Exception as exc:
            print(f"  tts: seg {seg_id} generate failed ({exc})", file=sys.stderr)
            return clip, meta, {"failed": True}
        audio.trim_leading_silence(clip, clip)
        return clip, meta, None

    def _verify_and_store(self, clip: Path, meta: Path, speak: str) -> dict[str, Any]:
        ok, ov, heard = self._verify(clip, speak)
        verdict = {"ok": ok, "overlap": round(ov, 3), "heard": heard,
                   "dur": round(audio.duration(clip), 3)}
        meta.write_text(json.dumps(verdict, ensure_ascii=False))
        return verdict

    @staticmethod
    def _record(clip: Path, verdict: dict[str, Any], attempt: int,
                opts: TtsOpts = ttsopts.DEFAULT) -> dict[str, Any]:
        rec = {"clip": f"clips/{clip.name}", "dur": verdict["dur"],
               "tries": attempt + 1, "overlap": verdict["overlap"], "verify": "ok"}
        # Which options this clip was made under, so `run` can spot a segment whose
        # options changed under an otherwise-usable clip. Absent for the defaults,
        # which keeps every existing record byte-identical.
        fp = opts.fingerprint()
        if fp:
            rec["opts"] = fp
        return rec

    def clip_for(self, seg: dict[str, Any], text_en: str) -> dict[str, Any] | None:
        """Bounded retry loop. Returns a tts record, or None to fall back to keep."""
        plan = self._plan(seg, text_en)
        if plan is None:
            return None
        speak, ref_path, ref_key, base_seed, opts = plan
        slot = float(seg["end"]) - float(seg["start"])
        best: dict[str, Any] | None = None

        # Escalation: the segment's own aligned window can be too short to clone
        # from (a ~1s reference reliably yields a ~1s truncated clip whatever the
        # text says). After the bounded tries fail, one extra attempt swaps in the
        # speaker's canonical reference — different audio, different cache key, so
        # it is a genuinely new synthesis, never a replay of a cached failure.
        # A pinned `tts_opts.ref` is exempt: the user chose that voice, and
        # escalating past it to the speaker's canonical reference would hand back
        # exactly the voice they were overriding.
        attempts = [(a, ref_path, ref_key) for a in range(MAX_TRIES)]
        alt = None if opts.ref else self._canonical_ref(seg)
        if alt is not None and alt[1] != ref_key:
            attempts.append((MAX_TRIES, alt[0], alt[1]))

        for attempt, a_path, a_key in attempts:
            clip, meta, verdict = self._attempt(seg["id"], speak, a_path, a_key,
                                                base_seed, attempt, opts)
            if verdict is not None and verdict.get("failed"):
                continue
            if verdict is None:
                verdict = self._verify_and_store(clip, meta, speak)
            # A clip the timeline could never absorb (shorten + 1.3x speed-up
            # recover maybe 3x; a 71s clip for a 5.7s slot once shoved everything
            # after it 49s late) is a failure however well it verifies.
            if clip_exceeds_slot(verdict["dur"], slot):
                print(f"  tts: seg {seg['id']} clip {verdict['dur']:.1f}s vs "
                      f"{slot:.1f}s slot — rejected", file=sys.stderr)
                continue
            record = self._record(clip, verdict, attempt, opts)
            if verdict["ok"]:
                return record
            if verdict["overlap"] > (best or {}).get("overlap", -1.0):
                best = dict(record, verify="soft")

        if best and best["overlap"] >= CLONE_SOFT_OVERLAP:
            print(f"  tts: seg {seg['id']} soft-accept (overlap {best['overlap']:.2f})",
                  file=sys.stderr)
            return best
        return None

    def keep_clip(self, seg: dict[str, Any]) -> dict[str, Any]:
        """Original audio for this span — the universal fallback.

        Named by the span itself, never by segment id: ids are renumbered
        whenever segmentation changes, and an id-keyed cache then hands back a
        clip of the wrong length for the wrong moment.
        """
        want = seg["end"] - seg["start"]
        key = hashlib.sha1(f"{seg['start']:.3f}|{seg['end']:.3f}".encode()).hexdigest()[:12]
        path = self.clips / f"keep_{key}.wav"
        if not path.is_file() or abs(audio.duration(path) - want) > 0.1:
            audio.extract_slice(self.workdir / self.m["files"]["source_wav"],
                                seg["start"], seg["end"], path)
        return {"clip": f"clips/{path.name}", "dur": round(audio.duration(path), 3),
                "tries": 0, "overlap": 1.0, "verify": "keep"}

    def close(self) -> None:
        if self._synth is not None:
            self._synth.free()
            self._synth = None
        self._synth_model = None
        self._voc = None


class _Synth:
    """Minimal Qwen3-TTS wrapper: x-vector-only cloning, one call per clip."""

    def __init__(self, *, device: str | None = None, model: str = DEFAULT_TTS_MODEL,
                 lang: str = "en"):
        import torch

        self.lang = (lang or "en").lower()
        self._qwen_lang: str | None = None

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
        import torch
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
        self._model = model
        return model

    def _language(self, model) -> str:
        """The checkpoint's name for the target language, resolved once per run."""
        if self._qwen_lang is None:
            try:
                supported = model.get_supported_languages()
            except Exception:
                supported = _QWEN_LANG_NAMES.values()
            self._qwen_lang = qwen_language_name(self.lang, supported)
            if self._qwen_lang == "Auto":
                print(f"  tts: target {self.lang!r} not in the checkpoint's supported "
                      "languages — synthesising with language=Auto", file=sys.stderr)
        return self._qwen_lang

    def generate(self, speak: str, ref: Path, out: Path, *, seed: int, greedy: bool,
                 opts: TtsOpts = ttsopts.DEFAULT) -> Path:
        import torch

        model = self._load()
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
        base = GREEDY if greedy else SAMPLED
        vals = {k: (base[k] if greedy or getattr(opts, k) is None else getattr(opts, k))
                for k in ("temperature", "top_p", "top_k")}
        sampling = {"do_sample": not greedy, "subtalker_dosample": not greedy}
        for k, v in vals.items():
            sampling[k] = v
            sampling[f"subtalker_{k}"] = v
        wavs, sr = model.generate_voice_clone(
            text=speak, language=self._language(model),
            voice_clone_prompt=self._prompts[key],
            max_new_tokens=opts.max_new_tokens or max_new_tokens(speak, self.lang),
            repetition_penalty=opts.repetition_penalty or REPETITION_PENALTY, **sampling,
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
    return _ASR_CANDIDATES if (tgt or "en").lower() == "en" else _ASR_CANDIDATES_MULTI
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
    print("  tts: no verification ASR — accepting on length only", file=sys.stderr)
    return None


def clear_failed_keeps(segments: list[dict[str, Any]]) -> list[int]:
    """Make every previously tts_failed segment dubbable again.

    A tts_failed keep is this stage's own per-run verdict, never cached state:
    whatever run set it, entering the stage again means re-attempting synthesis.
    `manifest.reset_stage` already undoes these on the invalidation path, but the
    resume path (same fingerprint after an interruption) skips reset_stage, and
    without this a resumed run would treat the failure as settled forever.
    Returns the ids that were cleared.
    """
    cleared: list[int] = []
    for seg in segments:
        if seg.get("keep_reason") == "tts_failed":
            seg["keep"], seg["keep_reason"] = False, None
            seg.pop("tts", None)  # the keep-clip record; this run re-decides
            cleared.append(seg["id"])
    return cleared


def run(m: dict[str, Any], workdir: Path, *, save=None, device: str | None = None,
        model: str = DEFAULT_TTS_MODEL) -> Engine:
    engine = Engine(m, workdir, device=device, model=model)
    clear_failed_keeps(m["segments"])
    engine.build_speaker_refs()
    dub = [s for s in m["segments"] if not s["keep"]]
    todo = [s for s in dub if needs_synthesis(s, workdir)]

    # Generation runs on the GPU, verification (Whisper) on the CPU. Generate each
    # first-attempt clip in order and hand its verify to a single worker thread, so
    # it overlaps the *next* clip's generation instead of stalling the GPU. One
    # worker only: faster-whisper is not re-entrant, and this already hides all but
    # the last clip's verify. Segments whose first attempt fails (rare) fall through
    # to clip_for, which resumes at attempt 1 straight from the cache.
    import concurrent.futures as cf

    pending: dict[int, tuple] = {}
    retry: list[dict[str, Any]] = []
    done = 0
    with cf.ThreadPoolExecutor(max_workers=1) as vpool:
        for seg in todo:
            plan = engine._plan(seg, seg["text_en"])
            if plan is None:
                retry.append(seg)
                continue
            speak, ref_path, ref_key, base_seed, opts = plan
            clip, meta, verdict = engine._attempt(seg["id"], speak, ref_path, ref_key,
                                                  base_seed, 0, opts)
            if verdict is not None and verdict.get("failed"):
                retry.append(seg)
                continue
            if verdict is not None:                    # cache hit — no verify needed
                if verdict["ok"]:
                    seg["tts"] = engine._record(clip, verdict, 0, opts)
                else:
                    retry.append(seg)
                continue
            pending[seg["id"]] = (vpool.submit(engine._verify, clip, speak),
                                  clip, meta, speak, opts)

        for seg in todo:
            item = pending.get(seg["id"])
            if item is None:
                continue
            fut, clip, meta, speak, opts = item
            ok, ov, heard = fut.result()
            verdict = {"ok": ok, "overlap": round(ov, 3), "heard": heard,
                       "dur": round(audio.duration(clip), 3)}
            meta.write_text(json.dumps(verdict, ensure_ascii=False))
            if ok:
                seg["tts"] = engine._record(clip, verdict, 0, opts)
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

    for seg in m["segments"]:
        if seg["keep"] and not (seg.get("tts") and (workdir / seg["tts"]["clip"]).is_file()):
            seg["tts"] = engine.keep_clip(seg)
    missing = [s["id"] for s in m["segments"] if not s.get("tts")]
    assert not missing, f"segments without audio: {missing}"
    return engine
