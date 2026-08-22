"""The single seam between the server and the pipeline.

Nothing else in `dubbing_app` imports `dubbing.*`. The server calls only the
functions below, which are thin adapters over `dubbing/edit.py` the pipeline's
own edit API. Two things genuinely live here, and nothing else does:

* **Progress shape.** `dubbing.edit` reports `(fraction, message)`; the server
  puts NDJSON frames on a bus. `_edit_progress` translates, at the seam.
* **Run options.** The app records the pipeline parameters a project was created
  with under `m["source"]["app_opts"]`, which is the app's own convention;
  `dubbing.edit` reads flat keys and otherwise falls back to the CLI defaults. So
  every call that re-runs a stage passes what the project recorded a render of a
  `--genre movie` project must be a movie render, and must stamp the fingerprints
  a `--genre movie` CLI run would stamp, or the next headless run redoes
  everything.

The model-loading functions (`retranslate`, `resynthesize`, `rebuild`,
`full_run`) are only ever called from the job child process
(`dubbing_app.worker`), never from the server process see `dubbing_app.runner`
for why. Their pipeline imports are therefore function-local: importing this
module must not drag torch/MLX into the server.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

from dubbing import STAGES
from dubbing import edit as _edit
from dubbing import manifest

Progress = Callable[[dict[str, Any]], None]

# One name for "the app asked for a structurally impossible edit": the routes catch
# `ops.EditError`, and `dubbing.edit` is the authority on what is impossible.
EditError = _edit.EditError
SegmentNotFound = _edit.SegmentNotFound

# Which per-segment fields each stage owns. `dubbing.edit` is the authority.
STAGE_FIELDS = _edit.STAGE_FIELDS

# Pipeline parameters a project can record, and the `dubbing.cli` argument each
# one is. Anything absent keeps `dubbing.edit._args`'s own fallback.
OPT_KEYS = ("genre", "register", "tts_model", "device", "transcript", "dub_foreign")


# --------------------------------------------------------------------------
# stable identity
# --------------------------------------------------------------------------

def ensure_uids(m: dict[str, Any]) -> bool:
    """Mint a `uid` for every segment that lacks one. True when anything changed.

    `seg["id"]` is positional and is renumbered whenever segmentation changes, so
    it cannot key the UI across a re-run. Runs made by the headless CLI have no
    uids at all; this is where they acquire them, once, on first read by the
    pipeline's own content-derived rule (`manifest.mint_uid`), so the uid survives
    a re-segmentation that reproduces the same span.
    """
    return bool(manifest.ensure_uids(m.get("segments")))


def find(m: dict[str, Any], uid: str) -> dict[str, Any] | None:
    return _edit.find(m, uid)


def _need(m: dict[str, Any], uid: str) -> dict[str, Any]:
    seg = find(m, uid)
    if seg is None:
        raise SegmentNotFound(f"no segment with uid {uid!r}")
    return seg


# --------------------------------------------------------------------------
# no models, instant safe to run while a job holds the model slot
# --------------------------------------------------------------------------

def set_text(m: dict[str, Any], uid: str, *, text: str | None = None,
             text_en: str | None = None) -> dict[str, Any]:
    return _edit.set_text(m, uid, text=text, text_en=text_en)


def set_keep(m: dict[str, Any], uid: str, keep: bool, reason: str = "manual") -> dict[str, Any]:
    return _edit.set_keep(m, uid, keep, reason)


def set_speaker(m: dict[str, Any], uid: str, speaker: str) -> dict[str, Any]:
    return _edit.set_speaker(m, uid, speaker)


def set_bounds(m: dict[str, Any], uid: str, start: float, end: float) -> dict[str, Any]:
    return _edit.set_bounds(m, uid, start, end)


def set_langs(m: dict[str, Any], uid: str, *, src_lang: str | None = None,
              tgt_lang: str | None = None) -> dict[str, Any]:
    return _edit.set_langs(m, uid, src_lang=src_lang, tgt_lang=tgt_lang)


def set_tts_opts(m: dict[str, Any], uid: str, **opts: Any) -> dict[str, Any]:
    """Per-segment synthesis overrides, validated before they reach the manifest.

    `dubbing.ttsopts` is the schema and it is deliberately loud but `tts.run` is
    where it is read, minutes into a job. An unknown key or an out-of-range value
    stored here is a `ValueError` raised inside every future run of the tts stage,
    on a manifest the user is not allowed to hand-edit back. So the patch is parsed
    at the door instead: `ttsopts.merge` applies it, validates the result and
    strips defaults, and a bad one is a 400 the editor can show.
    """
    from dubbing import ttsopts

    seg = _need(m, uid)
    current = dict(seg.get("tts_opts") or {})
    try:
        merged = ttsopts.merge(current, opts)
    except ValueError as exc:
        raise EditError(str(exc)) from exc
    # `None` is how `dubbing.edit.set_tts_opts` clears one, so name every key that
    # either side knows about and let the validated result decide.
    patch = {key: merged.get(key) for key in set(current) | set(merged) | set(opts)}
    return _edit.set_tts_opts(m, uid, **patch)


def set_locked(m: dict[str, Any], uid: str, locked: dict[str, bool]) -> dict[str, Any]:
    """Replace this segment's locks; `{}` releases them all."""
    return _edit.set_locked(m, uid, locked)


def invalidate(m: dict[str, Any], uid: str, *, stages: set[str]) -> dict[str, Any]:
    """Drop one segment's output for `stages`, so the stage remakes it.

    Returns the segment `dubbing.edit.invalidate` returns the field names it
    removed, which is not what the routes want to hand back.
    """
    _edit.invalidate(m, uid, stages=set(stages))
    return _need(m, uid)


def split(m: dict[str, Any], uid: str, at: float) -> tuple[str, str]:
    """Cut a segment in two at `at` seconds. Returns the two new uids."""
    return _edit.split(m, uid, at)


def merge(m: dict[str, Any], uid_a: str, uid_b: str) -> str:
    """Join two adjacent same-speaker segments. Returns the surviving uid."""
    return _edit.merge(m, uid_a, uid_b)


def remove(m: dict[str, Any], uid: str) -> dict[str, Any]:
    """Take a segment out of the dub. Returns the record that was deleted.

    Not "mute it": `dubbing.mix` plays the original vocals in every span no
    placement claims, so a removed span sounds like a passage the pipeline never
    detected. See `dubbing.edit.remove` for why that is the only honest answer.
    """
    return _edit.remove(m, uid)


def add(m: dict[str, Any], start: float, end: float, *, text: str,
        speaker: str | None = None) -> str:
    """Claim an uncovered span as a new segment. Returns its uid."""
    return _edit.add(m, start, end, text=text, speaker=speaker)


# --------------------------------------------------------------------------
# the run's pipeline options
# --------------------------------------------------------------------------

def recorded_opts(m: dict[str, Any]) -> dict[str, Any]:
    """This run's settings: the flat keys on `m["source"]` over `app_opts`.

    Two writers, one question. `dubbing.cli` records settings as flat keys on
    `m["source"]` (`cli.source_record`); the app records the same settings under
    `m["source"]["app_opts"]`. `dubbing.edit._args` resolves the pair flat-first,
    because flat is the older writer and the one a headless re-run refreshes —
    and, since `PATCH /api/projects/{name}`, the one a user's edit lands in.

    This resolves it the same way, and it has to: reading only `app_opts` here
    meant a genre the user changed after creation was honoured by every per-line
    re-translate (which goes through `edit._args`) and silently ignored by a full
    run or a resume (which comes through this module) the same project rendered
    two ways depending on which button was pressed.
    """
    src = m.get("source") or {}
    stored = dict(src.get("app_opts") or {})
    for key in (*OPT_KEYS, "captions"):
        if src.get(key) is not None:
            stored[key] = src[key]
    return stored


def pipeline_overrides(m: dict[str, Any]) -> dict[str, Any]:
    """The `dubbing.cli` arguments this project was created with.

    Only what was actually recorded: `dubbing.edit._args` reads flat keys off
    `m["source"]` and then falls back to argparse's defaults, and overriding a key
    the project never chose would replace that fallback with a guess.
    """
    stored = recorded_opts(m)
    return {key: stored[key] for key in OPT_KEYS if stored.get(key) is not None}


def _opts(m: dict[str, Any]) -> dict[str, Any]:
    """The CLI options this run was created with, with defaults filled in.

    Stored by the server at project creation so a later run reproduces the same
    pipeline parameters instead of silently switching genre/register.
    """
    stored = recorded_opts(m)
    return {"genre": stored.get("genre") or "documentary",
            "register": stored.get("register") or "narration",
            "tts_model": stored.get("tts_model") or "1.7b",
            "device": stored.get("device"),
            "transcript": stored.get("transcript") or "auto",
            "dub_foreign": bool(stored.get("dub_foreign")),
            "captions": stored.get("captions")}


# --------------------------------------------------------------------------
# models these need the job slot, and only ever run in the job child process
# --------------------------------------------------------------------------

def _edit_progress(progress: Progress | None, stage: str):
    """Adapt this module's event callback to `dubbing.edit`'s (fraction, message).

    The two halves were specified with different progress shapes the pipeline
    reports how far along it is, the server needs a frame it can put on the wire —
    so the translation lives here, at the seam, rather than either side bending.
    """
    if progress is None:
        return None

    def report(fraction: float, message: str) -> None:
        progress({"type": "stage", "stage": stage, "status": "running",
                  "progress": round(float(fraction), 4), "message": message})

    return report


def retranslate(m: dict[str, Any], workdir: Path, uids: Iterable[str], *,
                progress: Progress | None = None) -> dict[str, str]:
    """Re-translate just these segments. Returns uid → new target text."""
    opts = pipeline_overrides(m)
    kwargs = {key: opts[key] for key in ("register", "genre") if key in opts}
    return _edit.retranslate(m, workdir, list(uids),
                             progress=_edit_progress(progress, "translate"), **kwargs)


def resynthesize(m: dict[str, Any], workdir: Path, uids: Iterable[str], *,
                 progress: Progress | None = None) -> dict[str, dict]:
    """Re-voice just these segments, then re-place the timeline. uid → tts record."""
    opts = pipeline_overrides(m)
    return _edit.resynthesize(m, workdir, list(uids),
                              progress=_edit_progress(progress, "tts"),
                              device=opts.get("device"), model=opts.get("tts_model"))


def rebuild(m: dict[str, Any], workdir: Path, *, from_stage: str,
            progress: Progress | None = None,
            save: Callable[[], None] | None = None) -> list[str]:
    """Re-run stages forward from `from_stage`. Returns the stages run.

    `save` is the caller's write path, called after each stage: the job child hands
    in one that merges the edits the user made through the server while this was
    running, instead of overwriting them (see `dubbing_app.worker`).
    """
    return _edit.rebuild(m, workdir, from_stage=from_stage,
                         progress=_rebuild_progress(progress, from_stage), save=save,
                         **pipeline_overrides(m))


def _rebuild_progress(progress: Progress | None, asked: str):
    """`_edit_progress` for a rebuild, which does not stay in one stage.

    A render is asked for `timeline` and may run tts or translate first
    (`edit.start_stage`), and reporting the asked-for stage throughout left the UI
    saying "timeline" through the twenty minutes it spent in the synthesiser —
    with nothing else moving, which is what a re-render came to look like when it
    was really only voicing one line. `edit.rebuild` announces each stage by name
    as it enters it, so that is what the frame carries.
    """
    if progress is None:
        return None
    current = asked

    def report(fraction: float, message: str) -> None:
        nonlocal current
        if message in STAGES:
            current = message
        progress({"type": "stage", "stage": current, "status": "running",
                  "progress": round(float(fraction), 4), "message": message})

    return report


def full_run(workdir: Path, source: dict[str, Any], *,
             progress: Progress | None = None) -> int:
    """A whole pipeline run, fetch → report, via the headless entry point.

    Not part of the edit API: creating a project *is* `python -m dubbing`, and
    re-implementing its stage orchestration in the app would be exactly the fork
    docs/APP_ARCHITECTURE.md forbids. Progress comes from the stderr its stages
    already print, parsed by `dubbing_app.runner`.
    """
    from dubbing import cli

    return cli.main(full_run_argv(workdir, source))


def full_run_argv(workdir: Path, source: dict[str, Any]) -> list[str]:
    from dubbing import hebrew

    opts = _opts({"source": source})
    tgt = source.get("tgt_lang") or "en"
    # The Hebrew LoRA fits only the 1.7B talker, and the CLI refuses the other
    # combination. A project whose checkpoint was chosen before its target reaches
    # here; correcting it beats failing a run over a setting the UI never linked.
    tts_model = hebrew.ADAPTER_MODEL if hebrew.is_hebrew(tgt) else opts["tts_model"]
    argv = [str(source["input"]), "-o", str(workdir),
            "--src", source.get("src_lang") or "he",
            "--tgt", tgt,
            "--register", opts["register"], "--genre", opts["genre"],
            "--tts-model", tts_model, "--transcript", opts["transcript"]]
    if source.get("duration_limit"):
        argv += ["--duration", str(source["duration_limit"])]
    if source.get("context"):
        argv += ["--context", str(source["context"])]
    if opts["captions"]:
        argv += ["--captions", str(opts["captions"])]
    if opts["dub_foreign"]:
        argv += ["--dub-foreign"]
    if opts["device"]:
        argv += ["--device", str(opts["device"])]
    return argv
