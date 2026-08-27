"""Hebrew as a dub TARGET: the G2P that feeds it and the LoRA that speaks it.

Qwen3-TTS has ten native languages and Hebrew is not one of them. What makes it a
target here is `notmax123/QwenTTS-he-1.7B` a LoRA adapter over the **1.7B Base**
checkpoint that adds Hebrew *without touching a single base weight*. Two properties
of that adapter shape this module:

* It wraps the model's **`talker`** submodule, and PEFT can switch it off
  (`talker.disable_adapter()`), where the forward pass is bit-identical to the
  unmodified base verified in its model card across all 404 base tensors. So one
  loaded checkpoint serves Hebrew (adapter enabled) and the ten base languages
  (adapter disabled per call): no second model, no second load, and an English or
  Russian run whose checkpoint happens to carry the adapter sounds exactly as it
  did before. `merge_and_unload()` would bake the Hebrew deltas in permanently and
  destroy that guarantee, so it is never called.
* It was trained on **stressed IPA**, not Hebrew orthography. Hebrew script leaves
  the vowels out, so feeding it raw Hebrew does not work; the text field gets
  `ʔˈod ʁˈeɡa ʔanˈi …` and the `ˈ` (U+02C8) before the stressed vowel is a phoneme
  the model learned prosody from, not decoration.

The G2P is **ReNikud Plus** (`renikud-plus` on PyPI, arXiv 2606.20179): one INT8
ONNX graph that reads unvocalized Hebrew characters and predicts, per character, a
consonant, a vowel and whether it carries the word's primary stress straight to
IPA, with no diacritization pass in between. It is the same tool the adapter's
training data was phonemized with, and it reproduces the model card's worked
example character for character:

    עוד רגע אני אתייחס ליהודה ושומרון.  →  ʔˈod ʁˈeɡa ʔanˈi ʔetjaχˈes lihudˈa veʃomʁˈon.

The IPA is a **synthesis input, never a record**. `manifest.SEGMENT_KEYS` is a
whitelist and IPA is not on it: what is stored, subtitled and ASR-verified is
always the Hebrew orthography.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# ISO-639 spellings of Hebrew that reach us. "iw" is YouTube's legacy code and
# `transcript.run` normalizes it away, but a manifest or a `--tgt` can still carry it.
CODES = frozenset({"he", "iw"})

# The Hebrew LoRA. It was trained against the 1.7B Base talker and the output heads
# saved inside it are that checkpoint's shapes, so it does not fit the 0.6B one.
ADAPTER_DIR = REPO_ROOT / "models" / "QwenTTS-he-1.7B"
ADAPTER_HUB = "notmax123/QwenTTS-he-1.7B"
ADAPTER_MODEL = "1.7b"
# Absolute, like every other fetch command this app hands a user. `--local-dir` is
# resolved against the working directory, so a relative `models/QwenTTS-he-1.7B`
# pasted into a terminal that is not the repo root puts 236 MB somewhere nothing
# will ever look for it, and the Setup row stays red with no sign of why.
ADAPTER_DOWNLOAD = f"uv run hf download {ADAPTER_HUB} --local-dir {ADAPTER_DIR}"
# Mixed into every Hebrew clip's cache key, alongside the IPA actually synthesized.
# Bump it if the adapter is ever replaced by a differently-trained one.
ADAPTER_TAG = "qwentts-he-lora-v1"

# The grapheme→IPA weights. `renikud_onnx.G2P()` fetches them from the hub into the
# HF cache on first use, so the local copy is a convenience, not a requirement —
# same contract as Demucs and Pyannote.
G2P_DIR = REPO_ROOT / "models" / "RenikudPlus"
G2P_FILE = G2P_DIR / "model.onnx"
G2P_HUB = "notmax123/RenikudPlus"
# Pinned. The hub repo doubles as the G2P's development home, and the day a new
# training run landed on main (pat40e, 2026-08-21) every fresh dub started
# speaking Hebrew with a hard foreign accent — the LoRA was trained against the
# July revision's IPA and no other. Bump only after a validated →he run.
G2P_REVISION = "3d4b716a0004c56042032cefb42a806708ed0bcb"
G2P_DOWNLOAD = (f"uv run hf download {G2P_HUB} model.onnx --revision {G2P_REVISION} "
                f"--local-dir {G2P_DIR}")
G2P_PACKAGE = "renikud-plus"

# The G2P is gender-conditioned: 0 unknown, 1 male, 2 female. The pipeline does not
# classify speaker gender, so it asks as "unknown" which is also the setting that
# reproduces the adapter's training format exactly. The parameter stays on
# `phonemize` for a caller that does know (Hebrew inflects several readings by the
# addressee's gender: "ʃlomχˈa" to a man, "ʃlomˈeχ" to a woman).
SPEAKER_UNKNOWN = 0

STRESS = "ˈ"          # U+02C8, the primary-stress marker the adapter was trained with


def is_hebrew(lang: str | None) -> bool:
    """True for every spelling of Hebrew a language code reaches us as."""
    return (lang or "").lower() in CODES


def adapter_ready() -> bool:
    """The LoRA weights are on disk. Nothing downloads this one automatically."""
    return ADAPTER_DIR.is_dir() and any(ADAPTER_DIR.glob("adapter_model*"))


def g2p_ready() -> bool:
    """The G2P is importable its weights self-download, the package does not."""
    from importlib.util import find_spec

    return find_spec("renikud_onnx") is not None


def missing() -> list[str]:
    """What a Hebrew target still needs, as lines a user can act on.

    Empty means Hebrew is available as a target. Both halves are required and
    neither substitutes for the other: without the adapter the base checkpoint
    cannot say Hebrew at all, and without the G2P there is nothing to say to it —
    Hebrew orthography in the text field is not what the adapter was trained on.
    """
    out: list[str] = []
    if not adapter_ready():
        out.append(f"Hebrew TTS adapter missing: {ADAPTER_DIR}\n    {ADAPTER_DOWNLOAD}")
    if not g2p_ready():
        out.append(f"Hebrew G2P missing: no {G2P_PACKAGE} in this environment\n"
                   f"    uv sync   (it is a dependency; the weights then download on first use)")
    return out


def require() -> None:
    """Raise with the download commands unless Hebrew can actually be spoken."""
    gaps = missing()
    if gaps:
        raise SystemExit("target 'he' needs:\n  - " + "\n  - ".join(gaps))


# ------------------------------------------------------------------ grapheme → IPA

_G2P = None


def _load_g2p():
    """The ONNX G2P session, loaded once per process.

    A local `models/RenikudPlus/model.onnx` wins so an offline machine and the
    setup report agree on where the weights are; otherwise the package fetches
    them into the HF cache itself.
    """
    global _G2P
    if _G2P is None:
        from renikud_onnx import G2P

        where = G2P_FILE if G2P_FILE.is_file() else None
        if where is None:
            # Never let the package fetch "latest" on its own — the pin above
            # is the whole point. Fetch the pinned file, then load it as local.
            from huggingface_hub import hf_hub_download

            where = Path(hf_hub_download(G2P_HUB, "model.onnx",
                                          revision=G2P_REVISION))
        _G2P = G2P(str(where))
        print(f"  tts: Hebrew G2P ({where})", file=sys.stderr)
    return _G2P


def phonemize(text: str, *, speaker: int = SPEAKER_UNKNOWN) -> str:
    """Hebrew orthography → stressed IPA, the adapter's text format.

    Punctuation, spaces and anything not Hebrew pass through in place, so the
    sentence the synthesiser sees keeps the phrasing the line was written with.
    """
    if not (text or "").strip():
        return ""
    return _load_g2p().phonemize(text, speaker=int(speaker),
                                 target_speaker=int(speaker))


def free() -> None:
    """Drop the G2P session the Engine's `close`, and the tests."""
    global _G2P
    _G2P = None


# ---------------------------------------------------------------- adapter plumbing


def attach_adapter(talker):
    """Wrap a Qwen3-TTS `talker` in the Hebrew LoRA and return the wrapped module.

    Idempotent: a talker that is already a `PeftModel` is handed straight back, so
    a second call cannot stack adapters on top of each other.
    """
    from peft import PeftModel

    if isinstance(talker, PeftModel):
        return talker
    if not adapter_ready():
        raise FileNotFoundError(
            f"Hebrew TTS adapter missing: {ADAPTER_DIR}\n    {ADAPTER_DOWNLOAD}")
    return PeftModel.from_pretrained(talker, str(ADAPTER_DIR))
