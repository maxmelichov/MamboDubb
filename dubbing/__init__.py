"""Local video dubbing: captions → translation → voice-cloned TTS → mix.

Stages run in order and are individually resumable; see `dubbing.cli`.
The invariants that keep the output correct are documented in `dubbing.timeline`
(placement) and `dubbing.tts` (never-silent) — read those before changing either.
"""

__all__ = ["STAGES", "PASSTHROUGH_REASON"]

# `keep_reason` of a segment the user marked as passthrough (manifest field
# `passthrough`) rather than one the automatic rules decided. Lives here because
# both the segments stage that sets it and the translate stage that reads it
# need the same word for it.
PASSTHROUGH_REASON = "user"

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
