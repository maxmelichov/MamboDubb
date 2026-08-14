"""`seg["tts_opts"]` — the per-segment synthesis overrides, parsed and validated.

Everything here is *pure*: no models, no filesystem. `dubbing/tts.py` honours the
parsed options; the editor's `set_tts_opts` validates with `parse` before writing
one into the manifest.

Two rules the design follows:

* **A knob that cannot work is not shipped.** Every option below reaches a real
  argument of the synthesis call (or a real post-processing step). Qwen3-TTS's
  natural-language `instruct` is *not* here — see the module note in `tts.py`.
* **Unknown or out-of-range is loud.** `parse` raises `ValueError`; nothing is
  silently dropped, because a silently-ignored option looks exactly like a
  broken model to the user.

Defaults are today's hard-coded behaviour, so a segment with no `tts_opts`
(or `{}`) produces byte-identical audio and byte-identical cache keys.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from typing import Any

# Kept in step with tts.TTS_MODELS by an assertion there (importing it would be
# circular, and this module deliberately stays free of numpy/torch).
MODELS = ("1.7b", "0.6b")

SEED_MAX = 2 ** 32
SPEED_MIN, SPEED_MAX = 0.5, 2.0        # one ffmpeg atempo stage; beyond this is a chipmunk
TEMP_MAX = 2.0
TOP_K_MAX = 1000
REP_PEN_MIN, REP_PEN_MAX = 0.8, 2.0
TOKENS_MIN, TOKENS_MAX = 32, 4096      # 4096 codec tokens ≈ 5.5 min at 12.5 Hz


@dataclass(frozen=True)
class TtsOpts:
    """One segment's synthesis overrides. Every default reproduces today's run.

    * `seed` — replaces the text-derived base seed. Retry N still uses
      `seed + 1000 * N`, so the bounded retries stay distinct takes; attempt 0 is
      exactly the seed given. Re-rolling a bad take = bump this by one.
    * `greedy` — force the deterministic decode on *every* attempt instead of only
      on the last of `MAX_TRIES`. Steadier, flatter; the cure for a line that
      wanders or repeats.
    * `ref` — a wav under the run dir to clone from, replacing the window this
      stage would pick itself. This is how a wrong or ugly voice on one line gets
      fixed. A pinned reference also disables the canonical-reference escalation:
      the user's choice is final.
    * `ref_text` — the transcript of `ref`, which switches the clone to the
      checkpoint's ICL mode (`x_vector_only_mode=False`): the model conditions on
      the reference *codes and text*, not only its speaker embedding, which is the
      only prosody-carrying path this checkpoint has. Requires `ref` — the
      auto-picked window has no transcript the user could supply.
    * `model` — `"1.7b"` or `"0.6b"` for this segment only.
    * `speed` — tempo of the finished clip (ffmpeg `atempo`, pitch preserved).
      Applied before verification, so what is checked is what is used; an extreme
      value can therefore fail the length guard and fall back to `keep`.
    * `temperature` / `top_p` / `top_k` / `repetition_penalty` / `max_new_tokens`
      — the sampler, straight through to `generate_voice_clone`. The first three
      are rejected together with `greedy=True`, where they would do nothing.
    * `keep_pauses` — leave long internal silences alone instead of compressing
      them to `PAUSE_KEEP` (for a line whose dramatic pause is the point).
    """

    seed: int | None = None
    greedy: bool = False
    ref: str | None = None
    ref_text: str | None = None
    model: str | None = None
    speed: float = 1.0
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    max_new_tokens: int | None = None
    keep_pauses: bool = False

    # ------------------------------------------------------------------ output
    def to_dict(self) -> dict[str, Any]:
        """Only the options that differ from the default — the manifest stays small."""
        out: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if v != f.default:
                out[f.name] = v
        return out

    @property
    def icl(self) -> bool:
        """True when the clone conditions on reference text+codes, not the x-vector."""
        return self.ref_text is not None

    def clone_mode(self) -> str:
        """The cache-key token for the cloning mode ("xvec" is today's only one)."""
        if not self.icl:
            return "xvec"
        return "icl:" + hashlib.sha1((self.ref_text or "").encode("utf-8")).hexdigest()[:10]

    def cache_suffix(self) -> str:
        """Audio-affecting options that the cache key does not already carry.

        `seed`, `greedy` and `model` are separate ingredients of the key, `ref` is
        folded into the ref key and `ref_text` into `clone_mode`, so only the
        sampler and the post-processing appear here. Empty for the defaults,
        which is what keeps every existing cache entry valid.
        """
        parts = []
        for name, abbr in sorted(_CACHE_ABBR.items(), key=lambda kv: kv[1]):
            v = getattr(self, name)
            if v == _DEFAULTS[name]:
                continue
            parts.append(f"{abbr}={_fmt(v)}")
        return ";".join(parts)

    def fingerprint(self) -> str:
        """Short hash over *all* non-default options; "" for the defaults.

        Stored on the tts record so `tts.run` can tell that a segment's options
        changed under an otherwise-usable clip and re-synthesize it. Without it,
        editing `tts_opts` on an already-dubbed segment would do nothing at all.
        """
        d = self.to_dict()
        if not d:
            return ""
        blob = json.dumps(d, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:10]


DEFAULT = TtsOpts()
_DEFAULTS = {f.name: f.default for f in fields(TtsOpts)}
KEYS = tuple(_DEFAULTS)
_CACHE_ABBR = {
    "speed": "spd",
    "temperature": "t",
    "top_p": "tp",
    "top_k": "tk",
    "repetition_penalty": "rp",
    "max_new_tokens": "mnt",
    "keep_pauses": "kp",
}
# Options that only apply when the decode samples; with greedy they would be
# silently inert, so they are rejected together instead.
SAMPLING_KEYS = ("temperature", "top_p", "top_k")


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _bad(key: str, msg: str) -> None:
    raise ValueError(f"tts_opts.{key}: {msg}")


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _str(raw: dict, key: str) -> str | None:
    v = raw[key]
    if not isinstance(v, str) or not v.strip():
        _bad(key, f"expected a non-empty string, got {v!r}")
    return v


def _num(raw: dict, key: str, lo: float, hi: float, *, incl_lo: bool = True) -> float:
    v = raw[key]
    if not _is_num(v):
        _bad(key, f"expected a number, got {v!r}")
    v = float(v)
    if v != v or not ((lo <= v if incl_lo else lo < v) and v <= hi):
        _bad(key, f"expected {lo}{'' if incl_lo else ' (exclusive)'}..{hi}, got {v!r}")
    return v


def _int(raw: dict, key: str, lo: int, hi: int) -> int:
    v = raw[key]
    if not _is_int(v):
        _bad(key, f"expected an integer, got {v!r}")
    if not lo <= v <= hi:
        _bad(key, f"expected {lo}..{hi}, got {v!r}")
    return int(v)


def _bool(raw: dict, key: str) -> bool:
    v = raw[key]
    if not isinstance(v, bool):
        _bad(key, f"expected true or false, got {v!r}")
    return v


def _relpath(raw: dict, key: str) -> str:
    v = _str(raw, key)
    p = v.replace("\\", "/")
    if p.startswith("/") or (len(p) > 1 and p[1] == ":"):
        _bad(key, f"must be relative to the run directory, got {v!r}")
    if ".." in p.split("/"):
        _bad(key, f"must not escape the run directory, got {v!r}")
    return p


def parse(raw: Any) -> TtsOpts:
    """`seg["tts_opts"]` → `TtsOpts`. Raises `ValueError` on anything unusable.

    `None`, `{}` and a `TtsOpts` all pass through; anything else that is not a
    dict, any unknown key, and any out-of-range value raise. Nothing is coerced
    and nothing is ignored.
    """
    if raw is None:
        return DEFAULT
    if isinstance(raw, TtsOpts):
        return raw
    if not isinstance(raw, dict):
        raise ValueError(f"tts_opts: expected an object, got {type(raw).__name__}")
    unknown = [k for k in raw if k not in _DEFAULTS]
    if unknown:
        raise ValueError(
            f"tts_opts: unknown option(s) {sorted(unknown)!r}; known: {sorted(_DEFAULTS)!r}")

    got: dict[str, Any] = {}
    if "seed" in raw:
        got["seed"] = _int(raw, "seed", 0, SEED_MAX - 1)
    if "greedy" in raw:
        got["greedy"] = _bool(raw, "greedy")
    if "ref" in raw:
        got["ref"] = _relpath(raw, "ref")
    if "ref_text" in raw:
        got["ref_text"] = _str(raw, "ref_text")
    if "model" in raw:
        v = _str(raw, "model")
        if v not in MODELS:
            _bad("model", f"expected one of {list(MODELS)}, got {v!r}")
        got["model"] = v
    if "speed" in raw:
        got["speed"] = _num(raw, "speed", SPEED_MIN, SPEED_MAX)
    if "temperature" in raw:
        got["temperature"] = _num(raw, "temperature", 0.0, TEMP_MAX, incl_lo=False)
    if "top_p" in raw:
        got["top_p"] = _num(raw, "top_p", 0.0, 1.0, incl_lo=False)
    if "top_k" in raw:
        got["top_k"] = _int(raw, "top_k", 1, TOP_K_MAX)
    if "repetition_penalty" in raw:
        got["repetition_penalty"] = _num(raw, "repetition_penalty", REP_PEN_MIN, REP_PEN_MAX)
    if "max_new_tokens" in raw:
        got["max_new_tokens"] = _int(raw, "max_new_tokens", TOKENS_MIN, TOKENS_MAX)
    if "keep_pauses" in raw:
        got["keep_pauses"] = _bool(raw, "keep_pauses")

    opts = TtsOpts(**got)
    if opts.ref_text is not None and opts.ref is None:
        _bad("ref_text", "needs `ref` too — ICL mode conditions on the transcript of a "
                         "reference the caller chose, and the auto-picked window has none")
    if opts.greedy:
        inert = [k for k in SAMPLING_KEYS if getattr(opts, k) is not None]
        if inert:
            _bad("greedy", f"cannot be combined with {inert} — a greedy decode ignores "
                           "the sampler, and a knob that does nothing is worse than no knob")
    return opts


def merge(current: Any, updates: dict[str, Any]) -> dict[str, Any]:
    """Apply `updates` to a segment's stored options; `None` resets one to its default.

    Returns the dict to store (already validated, defaults stripped), so an editor
    can round-trip a partial patch without knowing the schema. No option has `None`
    as a meaningful *value*, so `None` unambiguously means "unset this one". An
    empty result means "back to the pipeline's own choices" and the caller should
    drop the key entirely.
    """
    base = parse(current).to_dict()
    for k, v in updates.items():
        if k not in _DEFAULTS:
            raise ValueError(
                f"tts_opts: unknown option {k!r}; known: {sorted(_DEFAULTS)!r}")
        if v is None:
            base.pop(k, None)
        else:
            base[k] = v
    return parse(base).to_dict()
