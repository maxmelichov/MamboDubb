"""Segment edits — the pure logic behind the editor's Save button.

Only the fields listed in `EDITABLE` may be written from the UI, and every one of
them maps to the earliest pipeline stage that has to re-run for the edit to reach
the mix. Editing a translation does *not* force `translate` (that would throw the
edit away and regenerate the line) — it forces `tts`, the first stage that reads
`text_en`. Asking for a different target language is the one case where the
translator itself must run again.

No I/O here: `apply_edits` takes a manifest dict and returns what changed, so it
is unit-testable without a server, a run directory, or a model.
"""

from __future__ import annotations

from typing import Any, Callable

from dubbing import STAGES
from dubbing.manifest import SEGMENT_KEYS


def _text(v: Any) -> str:
    if not isinstance(v, str):
        raise ValueError("expected a string")
    return v.strip()


def _opt_lang(v: Any) -> str | None:
    if v is None:
        return None
    if not isinstance(v, str):
        raise ValueError("expected a language code or null")
    v = v.strip().lower()
    return v or None


def _speaker(v: Any) -> str:
    v = _text(v)
    if not v:
        raise ValueError("speaker cannot be empty")
    return v


def _flag(v: Any) -> bool:
    if not isinstance(v, bool):
        raise ValueError("expected true or false")
    return v


# field -> (coercer, stage that must re-run when the field changes)
#
# The studio (dubbing_app + dubbing/edit.py) and this editor share one schema:
# a concept gets ONE manifest key, whichever front end wrote it. Two fields this
# editor originally proposed were therefore renamed to the canonical keys at
# merge, and one was dropped outright:
#   lang_override    -> tgt_lang  (the studio's per-segment target language)
#   passthrough      -> keep + keep_reason="manual" (+ locked.keep, see below)
#   tts_instructions -> REJECTED: Qwen3-TTS's clone path has no instruction
#                       channel at all (generate_voice_clone takes no `instruct`;
#                       the checkpoints that do cannot clone a reference voice).
#                       Storing it would be a knob that silently does nothing —
#                       per-segment control lives in `tts_opts` (seed, greedy,
#                       ref, ref_text, …), validated by dubbing/ttsopts.py.
EDITABLE: dict[str, tuple[Callable[[Any], Any], str]] = {
    # source transcript text: the translator's input, so translation redoes.
    "text": (_text, "translate"),
    # the translated line the synthesiser speaks — keep it, just re-synthesise.
    "text_en": (_text, "tts"),
    # which voice clone reference the segment is spoken with.
    "speaker": (_speaker, "tts"),
    # what language this span is actually spoken in (subtitles + translation).
    "lang": (_opt_lang, "translate"),
    # per-segment target language override, for re-translating one line elsewhere.
    "tgt_lang": (_opt_lang, "translate"),
    # speaker already speaks the target language: keep the original audio.
    "passthrough": (_flag, "tts"),
}

# Wire-name compatibility: the static UI may still send the old names.
ALIASES = {"lang_override": "tgt_lang"}

REJECTED: dict[str, str] = {
    "tts_instructions": "the synthesiser has no instruction channel: "
                        "generate_voice_clone takes no instruct argument, and the "
                        "checkpoints that do cannot clone a voice — use tts_opts "
                        "(seed, greedy, ref, ref_text) instead",
}

# Fields the editor stores must survive manifest.save()'s whitelist. `passthrough`
# is not stored under its own name — it lands on keep/keep_reason/locked.
_MISSING = sorted(f for f in EDITABLE if f != "passthrough" and f not in SEGMENT_KEYS)
assert not _MISSING, f"add to manifest.SEGMENT_KEYS: {_MISSING}"
for _written in ("keep", "keep_reason", "locked"):
    assert _written in SEGMENT_KEYS, _written


def earliest(stages: list[str] | set[str]) -> str | None:
    """The stage furthest upstream in `stages` — the one to --force."""
    known = [s for s in STAGES if s in set(stages)]
    return known[0] if known else None


def apply_edits(m: dict[str, Any], edits: list[dict[str, Any]]) -> dict[str, Any]:
    """Write `edits` into the manifest in place.

    Each edit is `{"id": <segment id>, "fields": {name: value}}`. Values that
    match what is already stored are dropped, so saving an untouched form does
    not invalidate anything. Returns `{"changed": [ids], "fields": [names],
    "force": stage | None}` where `force` is the stage to re-run.
    """
    by_id = {seg["id"]: seg for seg in m.get("segments") or []}
    changed_ids: list[int] = []
    changed_fields: set[str] = set()
    stages: set[str] = set()

    for edit in edits:
        try:
            seg_id = int(edit["id"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"edit without a valid segment id: {edit!r}") from None
        seg = by_id.get(seg_id)
        if seg is None:
            raise ValueError(f"no segment {seg_id} in this run")
        touched = False
        for name, raw in (edit.get("fields") or {}).items():
            name = ALIASES.get(name, name)
            if name in REJECTED:
                raise ValueError(f"field {name!r}: {REJECTED[name]}")
            if name not in EDITABLE:
                raise ValueError(f"field {name!r} is not editable")
            coerce, stage = EDITABLE[name]
            try:
                value = coerce(raw)
            except ValueError as exc:
                raise ValueError(f"segment {seg_id}, field {name!r}: {exc}") from None
            # Absent and empty are the same state, so clearing an unset field is
            # a no-op rather than a manifest change that invalidates a stage.
            # `passthrough` is stored as keep/keep_reason, so it compares there.
            current = bool(seg.get("keep")) if name == "passthrough" else seg.get(name)
            if value == current or (not value and not current):
                continue
            if not value and name in ("text", "text_en"):
                # Blanking these would leave the synthesiser nothing to say.
                raise ValueError(f"segment {seg_id}, field {name!r}: cannot be empty")
            if name == "passthrough":
                # Canonical form: the studio's manual keep, locked against re-runs.
                seg["keep"], seg["keep_reason"] = bool(value), ("manual" if value else None)
                seg.setdefault("locked", {})["keep"] = True
            elif not value:
                seg.pop(name, None)   # cleared: drop it instead of storing a blank
            else:
                seg[name] = value
            touched = True
            changed_fields.add(name)
            stages.add(stage)
        if touched:
            changed_ids.append(seg_id)

    return {
        "changed": changed_ids,
        "fields": sorted(changed_fields),
        "force": earliest(stages),
    }
