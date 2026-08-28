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

The same is true of the **warm-up carrier** further down: a fixed Hebrew phrase
decoded ahead of every line and cut back off before anything downstream sees the
clip, because the adapter's Hebrew only asserts itself a few seconds into a decode
that starts from an English speaker's x-vector. It is a synthesis input too, and it
never reaches the audio, the manifest or the subtitles.
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


# ------------------------------------------------------------------- warm-up carrier

# The talker decodes autoregressively from a reference x-vector, and that vector is
# cut from the SOURCE speaker, an English one in the pair this was found on. At
# step zero the only thing conditioning the acoustics is that English voice, and the
# adapter's Hebrew prior only takes over once there is enough of the model's own
# Hebrew in the context to condition on. Measured over 29 clips of a real en→he run,
# the VoxLingua107 Hebrew posterior starts at 0.59 and does not settle near its
# ceiling until about 2.5-3s in (0.85 past 2s, p=0.004 paired against the onset),
# which is what "the first four or five words sound wrong" is, in numbers.
#
# So the line is not synthesized cold. A fixed Hebrew phrase is decoded first and cut
# back off, and the sentence the user hears begins already warm. The phrase is
# content-free and identical for every line and every video: this is a property of
# the decoder, not of anything a particular clip says.
#
# Length is the active ingredient, not the words. In the sweep that chose this one,
# a 0.9s carrier did nothing and a 1.7s carrier was not significant (+0.055,
# p=0.56); only at ~3.5s did the onset move (+0.171 overall, p=0.019, better on 10
# of 11 lines; +0.302 on the lines that actually had a bad onset). Shorten this text
# and the fix stops working, so re-measure before touching it.
CARRIER_TEXT = "רגע אחד בבקשה, ועכשיו נמשיך הלאה."
# There used to be a CARRIER_WORDS = 6 here, and the cut was made at the start of the
# ASR's seventh word. It is gone because the count was the bug. The ASR does not
# segment a fixed phrase into a fixed number of tokens: over 8 clips of one real run
# it heard the carrier as 6 tokens six times, as 7 once (הלאה split into "על אשצה")
# and as 8 once (אחד heard as "סאגה הד" plus a spurious "או"). Both odd cases put the
# seventh word's start inside the carrier's own last words, so the cut left carrier
# audio on the front of the shipped clip. Nothing may count ASR tokens here again;
# the carrier is located by matching its CHARACTERS (see `tts.carrier_boundary`).
#
# There used to be a CARRIER_MATCH_MAX = 0.40 here as well, a ceiling on how far the
# ASR's reading of the carrier could drift from CARRIER_TEXT before the boundary was
# refused outright. It is gone because it was measuring the wrong thing, and it took
# the whole feature down with it: over 10 freshly generated carrier clips it refused
# the cut on 10, every line fell back to the cold decode the carrier exists to prevent,
# and the fix shipped disabled by its own guard.
#
# Two separate facts killed it. First, the ASR does not transcribe the carrier, it
# hallucinates over it: the warm-up is rushed and content-free, so whisper returns
# gibberish that happens to have Hebrew letters in it ("וגטא דבאפ הוא סחיון אנישייב
# נייל,", "רגשי הדת בפאב הוא השיון הניצויד הלאה,"). The best prefix distance on those
# 10 clips ran 14 to 18 against a 26-letter carrier, i.e. 0.54 to 0.69, never once
# under the 0.40 bar. Second, and fatally for the whole idea, the distance does not
# rank boundaries by whether they are RIGHT. On segment 27 of the run fe713ff was
# written from, the wrong cut (the word-count bug's, mid-carrier) scores 5 and the
# correct one also scores 5; on segment 25 the wrong cut scores 12 where every correct
# boundary above scores 14 or worse. A number that a known-bad boundary beats cannot
# be the thing that certifies a good one, at any threshold. There is no separation to
# tune, so raising the ceiling was never the fix, and it is not what happens below.
#
# What the alignment IS good for is two narrower jobs, and it keeps both. It can say
# whether a carrier is there at all, and it can rank candidate cut points. Neither
# needs an absolute distance, so neither uses one.
#
# A clip with the carrier in front of it produces a distance curve with a floor in it:
# the score falls as the prefix grows to cover the carrier and climbs steeply once the
# prefix starts eating the sentence. A clip with no carrier has no such shape, only a
# slow climb from the first word. That descent, first word's score minus the floor, is
# how presence is decided, and it separates cleanly where the raw distance does not:
# 1 on the no-carrier clip against 6 to 10 on all ten carrier clips.
CARRIER_DESCENT_MIN = 5
# How far above the floor a prefix may score and still be worth trying as a cut point.
# The floor itself is often a letter or two off the truth, because the ASR's gibberish
# does not end exactly where the carrier does: on 3 of the 10 clips the best-scoring
# prefix stopped one word short and would have left the carrier's own last word on the
# clip. Those neighbours score within 1 of the floor, so instead of trusting the argmin
# the boundaries within this slack are all offered, best first, and each is cut and
# LISTENED TO until one is proven clean (see `tts.carrier_boundary` and the far-side
# check below). Ranking is advisory; only the listening is fatal.
CARRIER_MATCH_SLACK = 2
# The cut is proven from the other side, and since the pre-cut threshold above is gone
# this is now the ONLY thing that can refuse a take: the clip is re-transcribed after
# cutting and has to open on the sentence's own first CARRIER_HEAD_WORDS words, within
# CARRIER_HEAD_MAX of character drift. That is the right place for the only fatal
# check to sit, because it is the one that reads the audio that would actually ship,
# and because the failure it looks for (carrier letters still on the front) is the
# failure that matters. Two words rather than one because a single mis-heard opening
# word is ordinary and would reject good takes; 0.34 passes every correct cut measured
# (worst 0.18, "שמי כריסטינה" heard as "שמי קליסטינה") and rejects both of the bad
# ones from fe713ff's run (0.70 and above).
CARRIER_HEAD_WORDS = 2
CARRIER_HEAD_MAX = 0.34
# The head is compared as LETTERS against the best-matching prefix of what was heard,
# not as CARRIER_HEAD_WORDS words against CARRIER_HEAD_WORDS words. Same reasoning as
# the carrier itself: the ASR re-splits words freely and a fixed word count is not a
# fixed amount of material. On 2 of the 10 clips whisper broke the sentence's own
# first word in two ("כשמבינים" heard as "שהם מבינים"), so a two-word head covered
# nine letters where the line's own two words are eleven, and a correct cut scored as
# a mismatch and was thrown away.
#
# Comparing prefixes that way is generous in one direction, though, and this bounds it.
# Carrier left on the front is expensive, because every stray letter has to be deleted
# to reach the line. A cut that ate the start of the LINE is cheap, because a couple of
# missing letters are a couple of insertions. So the match is required to be a match of
# roughly the whole head and not of a truncated one: the winning prefix may fall at
# most this many letters short of the head's own length. At 1 a clip that opens mid-word
# ("מבינים" for "כשמבינים") is refused where the drift ratio alone would have kept it.
CARRIER_HEAD_SHORT = 1
# A coarse bound on where a boundary can possibly be, not the reason it is trusted:
# see the note above about what happens when a duration band is the only guard. Over
# fe713ff's 8 clips a correct boundary sat between 2.86s and 3.62s, and the floor was
# 2.5s on that evidence. The 10 clips measured since put correct boundaries as early
# as 2.34s, and on two of them the 2.5s floor was the only thing refusing a cut that
# was otherwise right, so the floor moves down with room under the new minimum. It can
# afford to: widening a band that only ever bounded the search costs nothing now that
# every candidate inside it is cut and listened to before it can ship.
CARRIER_MIN_SEC = 2.0
CARRIER_MAX_SEC = 5.0
# Mixed into every carrier-synthesized clip's cache key. Bump it with CARRIER_TEXT,
# because a different lead-in decodes a different sentence behind it, and with any
# change to where the carrier is cut, because that changes the audio that ships.
# v1 clips were cut by counting ASR tokens and some of them still carry the tail of
# the carrier, so none of them may be replayed.
# v2 clips must not be replayed either, and for the opposite reason: almost none of
# them have a carrier cut off them at all. The v2 gate refused the cut on 10 of 10
# measured clips, and the caller's answer to a refusal is to re-decode the line cold
# and write THAT into the same cache entry, because the key records that a carrier was
# attempted and not whether one survived. So a v2 entry is, in practice, exactly the
# cold-start audio this feature exists to replace, filed as though it were the fix.
# Nothing distinguishes the two cases by key, and the cut point moved as well, so the
# whole generation goes.
CARRIER_TAG = "he-carrier-v3"

_CARRIER_IPA: str | None = None


def carrier() -> str:
    """The carrier phrase as IPA, phonemized once per process.

    Through the same G2P the real line goes through rather than a hardcoded
    transcription, so a G2P revision cannot leave the carrier spelled in a
    convention the adapter no longer speaks while the sentence moves on.
    """
    global _CARRIER_IPA
    if _CARRIER_IPA is None:
        _CARRIER_IPA = phonemize(CARRIER_TEXT)
    return _CARRIER_IPA


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
    global _G2P, _CARRIER_IPA
    _G2P = None
    _CARRIER_IPA = None      # it came out of that session; a new one re-derives it


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
