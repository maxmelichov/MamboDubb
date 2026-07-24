"""Stage 6 — Qwen3-TTS zero-shot voice cloning, one clip per segment.

Two guarantees this module owns:

* **Never silent.** Every segment leaves this stage with an audio clip. A dub
  that cannot be verified after its bounded retries falls back to the original
  audio slice (`keep_reason="tts_failed"`), never to nothing.
* **Never partial.** A segment is one TTS call for its whole text. There is no
  sub-unit splitting, so there is no way for part of a line to go missing.

Clips are cached on disk by content hash, which is also the resume mechanism for
long videos: a killed run picks up at the first uncached clip.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from . import audio

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "Qwen3-TTS-12Hz-1.7B-Base"
HUB_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_TAG = "qwen3-tts-1.7b-base"
CODEC_HZ = 12.5

REF_SR = 24000
REF_TARGET_SEC = 4.5
REF_MIN_SEC = 2.5
REF_MIN_RMS = 0.018
REF_MAX_NOISE = 8.0

CLONE_MIN_SEC_PER_WORD = 0.18   # faster than this is chipmunk garble
CLONE_MAX_SEC_PER_WORD = 0.95   # slower than this is a stall/drawl
CLONE_MIN_OVERLAP = 0.35        # accept
CLONE_SOFT_OVERLAP = 0.20       # best-effort accept
MAX_TRIES = 3

_LATIN_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9''\-.]*")
_CHROME = re.compile(r"[\[\(][^\]\)]{0,40}[\]\)]")


def prepare_en(text: str) -> str:
    """Strip anything the English voice cannot speak; keep every word it can."""
    t = _CHROME.sub(" ", text or "")
    t = t.replace("[[", " ").replace("]]", " ")
    parts = re.split(r"([.!?,;:])", t)
    out: list[str] = []
    for part in parts:
        if part in ".!?,;:":
            if out and out[-1] not in ".!?,;:":
                out.append(part)
            continue
        words = _LATIN_RUN.findall(part)
        if words:
            out.append(" ".join(words))
    speak = re.sub(r"\s+([.!?,;:])", r"\1", " ".join(out)).strip()
    speak = re.sub(r"\s+", " ", speak)
    if speak and speak[-1] not in ".!?":
        speak += "."
    return speak


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def word_overlap(target: str, heard: str) -> float:
    a, b = _tokens(target), _tokens(heard)
    if not a:
        return 0.0
    common = sum((Counter(a) & Counter(b)).values())
    return common / len(a)


def clone_length_ok(sec: float, text: str) -> bool:
    words = max(1, len((text or "").split()))
    if sec <= 0.05:
        return False
    spw = sec / words
    if spw < CLONE_MIN_SEC_PER_WORD:
        return False
    if spw > CLONE_MAX_SEC_PER_WORD and words >= 3:
        return False
    expected = words / 3.0 + 0.25
    return sec >= expected * 0.40


def max_new_tokens(text: str) -> int:
    words = max(1, len(text.split()))
    return max(96, min(2048, int((words * 0.65 + 2.5) * CODEC_HZ)))


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


class Engine:
    """Holds the loaded models and the reference bank for one run."""

    def __init__(self, m: dict[str, Any], workdir: Path, *, device: str | None = None):
        self.m = m
        self.workdir = workdir
        self.clips = workdir / "clips"
        self.refs = workdir / "refs"
        self.clips.mkdir(parents=True, exist_ok=True)
        self.refs.mkdir(parents=True, exist_ok=True)
        self.device = device
        self._synth = None
        self._asr = None
        self._voc: np.ndarray | None = None

    # -- lazy resources ----------------------------------------------------
    @property
    def vocals(self) -> np.ndarray:
        if self._voc is None:
            path = self.workdir / self.m["files"]["vocals"]
            self._voc = audio.decode_mono(path, REF_SR)
        return self._voc

    @property
    def synth(self):
        if self._synth is None:
            self._synth = _Synth(device=self.device)
        return self._synth

    @property
    def asr(self):
        if self._asr is None:
            self._asr = _load_asr()
        return self._asr

    # -- reference audio ---------------------------------------------------
    def build_speaker_refs(self) -> None:
        """One canonical reference per speaker: their cleanest few seconds."""
        by_speaker: dict[str, list[dict]] = {}
        for seg in self.m["segments"]:
            if not seg["keep"]:
                by_speaker.setdefault(seg["speaker"], []).append(seg)
        for spk, segs in by_speaker.items():
            info = self.m["speakers"].setdefault(spk, {})
            if info.get("ref") and (self.workdir / info["ref"]).is_file():
                continue
            longest = sorted(segs, key=lambda s: s["start"] - s["end"])[:30]
            best = None
            for seg in longest:
                got = best_ref_window(self.vocals, seg["start"], seg["end"], REF_TARGET_SEC)
                if got and (best is None or got[2] < best[2]):
                    best = got
            if best is None:
                info["ref"] = None
                print(f"  tts: no clean reference for {spk}", file=sys.stderr)
                continue
            start, end, score, _rms = best
            path = self.refs / f"{spk}.wav"
            sf.write(str(path), self.vocals[int(start * REF_SR) : int(end * REF_SR)], REF_SR)
            info["ref"] = f"refs/{path.name}"
            info["ref_span"] = [round(start, 2), round(end, 2)]
            info["ref_noise"] = round(score, 2)

    def ref_for(self, seg: dict[str, Any]) -> tuple[Path, str] | None:
        """The segment's own aligned voice — whoever actually speaks in this window.

        The reference is cut strictly from the segment's own `[start, end]`, so the
        cloned voice is the real speaker at that exact moment. It deliberately does
        NOT widen into neighbouring segments: diarization here is unreliable —
        ECAPA shows two segments labelled the same speaker can be different voices —
        so widening by label blends another person into the clone (the audible
        wrong-speaker/voice-switch around 1:11). A short line simply gets a shorter
        reference of the correct voice.
        """
        span = seg["end"] - seg["start"]
        got = best_ref_window(self.vocals, seg["start"], seg["end"], min(REF_TARGET_SEC, span))
        if got and got[3] >= REF_MIN_RMS:
            start, end, _score, _rms = got
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
        info = self.m["speakers"].get(seg["speaker"]) or {}
        if info.get("ref"):
            return self.workdir / info["ref"], f"{seg['speaker']}:canonical"
        return None

    # -- synthesis ---------------------------------------------------------
    def _cache_key(self, speak: str, ref_key: str, seed: int, greedy: bool) -> str:
        blob = f"{MODEL_TAG}|{speak}|{ref_key}|xvec|{seed}|{int(greedy)}"
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]

    def _verify(self, clip: Path, speak: str) -> tuple[bool, float, str]:
        try:
            sec = audio.duration(clip)
        except Exception:
            return False, 0.0, "unreadable"
        if not clone_length_ok(sec, speak):
            return False, 0.0, f"len={sec:.2f}s"
        model = self.asr
        if model is None:
            return True, 1.0, "no-asr"
        try:
            segs, _ = model.transcribe(str(clip), language="en", word_timestamps=False,
                                       condition_on_previous_text=False, vad_filter=True)
            heard = " ".join((s.text or "").strip() for s in segs).strip()
        except Exception as exc:
            return False, 0.0, f"asr-error {exc}"
        heb = len(re.findall(r"[֐-׿]", heard))
        lat = len(re.findall(r"[A-Za-z]", heard))
        if heb and (heb >= lat or lat < 3):
            return False, 0.0, "source-script output"
        ov = word_overlap(speak, heard)
        return ov >= CLONE_MIN_OVERLAP, ov, heard[:120]

    def clip_for(self, seg: dict[str, Any], text_en: str) -> dict[str, Any] | None:
        """Bounded retry loop. Returns a tts record, or None to fall back to keep."""
        speak = prepare_en(text_en)
        if not speak or len(_tokens(speak)) == 0:
            return None
        ref = self.ref_for(seg)
        if ref is None:
            return None
        ref_path, ref_key = ref
        base_seed = int(hashlib.sha1(f"{seg['id']}|{speak}".encode()).hexdigest()[:8], 16)
        best: dict[str, Any] | None = None

        for attempt in range(MAX_TRIES):
            greedy = attempt == MAX_TRIES - 1
            seed = base_seed + 1000 * attempt
            key = self._cache_key(speak, ref_key, seed, greedy)
            clip = self.clips / f"{key}.wav"
            meta = self.clips / f"{key}.json"
            if clip.is_file() and meta.is_file():
                verdict = json.loads(meta.read_text())
            else:
                try:
                    self.synth.generate(speak, ref_path, clip, seed=seed, greedy=greedy)
                except Exception as exc:
                    print(f"  tts: seg {seg['id']} generate failed ({exc})", file=sys.stderr)
                    continue
                audio.trim_leading_silence(clip, clip)
                ok, ov, heard = self._verify(clip, speak)
                verdict = {"ok": ok, "overlap": round(ov, 3), "heard": heard,
                           "dur": round(audio.duration(clip), 3)}
                meta.write_text(json.dumps(verdict, ensure_ascii=False))
            record = {"clip": f"clips/{clip.name}", "dur": verdict["dur"],
                      "tries": attempt + 1, "overlap": verdict["overlap"], "verify": "ok"}
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
        self._voc = None


class _Synth:
    """Minimal Qwen3-TTS wrapper: x-vector-only cloning, one call per clip."""

    def __init__(self, *, device: str | None = None):
        import torch

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
        self.path = str(MODEL_PATH) if MODEL_PATH.is_dir() else HUB_ID
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

    def generate(self, speak: str, ref: Path, out: Path, *, seed: int, greedy: bool) -> Path:
        import torch

        model = self._load()
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        key = str(ref)
        if key not in self._prompts:
            self._prompts[key] = model.create_voice_clone_prompt(
                ref_audio=str(ref), ref_text=None, x_vector_only_mode=True
            )
        sampling = (
            {"do_sample": False, "temperature": 0.01, "top_p": 1.0, "top_k": 1,
             "subtalker_dosample": False, "subtalker_temperature": 0.01,
             "subtalker_top_p": 1.0, "subtalker_top_k": 1}
            if greedy else
            {"do_sample": True, "temperature": 0.55, "top_p": 0.85, "top_k": 30,
             "subtalker_dosample": True, "subtalker_temperature": 0.55,
             "subtalker_top_p": 0.85, "subtalker_top_k": 30}
        )
        wavs, sr = model.generate_voice_clone(
            text=speak, language="English", voice_clone_prompt=self._prompts[key],
            max_new_tokens=max_new_tokens(speak), repetition_penalty=1.08, **sampling,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        raw = out.with_suffix(".raw.wav")
        sf.write(str(raw), _trim_internal_silence(np.asarray(wavs[0], dtype=np.float32), sr), sr)
        audio.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-acodec", "pcm_s16le",
                   "-ar", str(audio.SR), "-ac", "1", str(out)])
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
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


_ASR_CANDIDATES = ("models/faster-whisper-base.en", "Systran/faster-whisper-base.en",
                   "models/faster-whisper-tiny.en", "Systran/faster-whisper-tiny.en")


def _load_asr():
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        print(f"  tts: verification ASR unavailable ({exc})", file=sys.stderr)
        return None
    for cand in _ASR_CANDIDATES:
        local = REPO_ROOT / cand
        target = str(local) if local.is_dir() else cand
        try:
            model = WhisperModel(target, device="cpu", compute_type="auto")
            print(f"  tts: verifying with {target}", file=sys.stderr)
            return model
        except Exception:
            continue
    print("  tts: no verification ASR — accepting on length only", file=sys.stderr)
    return None


def run(m: dict[str, Any], workdir: Path, *, save=None, device: str | None = None) -> Engine:
    engine = Engine(m, workdir, device=device)
    engine.build_speaker_refs()
    dub = [s for s in m["segments"] if not s["keep"]]
    for n, seg in enumerate(dub, 1):
        if seg.get("tts") and (workdir / seg["tts"]["clip"]).is_file():
            continue
        record = engine.clip_for(seg, seg["text_en"])
        if record is None:
            seg["keep"], seg["keep_reason"] = True, "tts_failed"
            print(f"  tts: seg {seg['id']} unusable → keep original", file=sys.stderr)
        else:
            seg["tts"] = record
        if n % 10 == 0:
            print(f"  tts: {n}/{len(dub)}", file=sys.stderr)
            if save:
                save()
    for seg in m["segments"]:
        if seg["keep"] and not (seg.get("tts") and (workdir / seg["tts"]["clip"]).is_file()):
            seg["tts"] = engine.keep_clip(seg)
    missing = [s["id"] for s in m["segments"] if not s.get("tts")]
    assert not missing, f"segments without audio: {missing}"
    return engine
