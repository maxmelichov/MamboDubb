"""Local video dubbing: captions → translation → voice-cloned TTS → mix.

Stages run in order and are individually resumable; see `dubbing.cli`.
The invariants that keep the output correct are documented in `dubbing.timeline`
(placement) and `dubbing.tts` (never-silent) — read those before changing either.
"""

__all__ = ["STAGES", "PASSTHROUGH_REASON", "MANUAL_REASON", "USER_KEEP_REASONS"]

# `keep_reason` of a segment the user marked as passthrough (manifest field
# `passthrough`) rather than one the automatic rules decided. Lives here because
# both the segments stage that sets it and the translate stage that reads it
# need the same word for it.
PASSTHROUGH_REASON = "user"
# The same verdict, written by the studio's own door: `edit.set_keep` stamps
# `passthrough` AND this reason, which its UIs key on. Two names for one thing is
# a historical accident, not a distinction — anything asking "did the user choose
# this keep?" must accept both, or half the user's overrides go unrecognised.
MANUAL_REASON = "manual"
USER_KEEP_REASONS = (PASSTHROUGH_REASON, MANUAL_REASON)

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
