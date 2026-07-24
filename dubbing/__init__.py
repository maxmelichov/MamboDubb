"""Local video dubbing: captions → translation → voice-cloned TTS → mix.

Stages run in order and are individually resumable; see `dubbing.cli`.
The invariants that keep the output correct are documented in `dubbing.timeline`
(placement) and `dubbing.tts` (never-silent) — read those before changing either.
"""

__all__ = ["STAGES"]

STAGES = (
    "fetch",
    "stems",
    "transcript",
    "segments",
    "translate",
    "tts",
    "timeline",
    "mix",
    "report",
)
