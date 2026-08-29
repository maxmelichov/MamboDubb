"""Entry point: `uv run python -m dubbing <url-or-file>`.

Stages run in order. Each one is skipped when its fingerprint and outputs are
already on disk, so re-running is cheap and an interrupted run resumes where it
stopped. `--force <stage>` re-runs that stage; everything downstream of it
invalidates automatically through the fingerprint chain.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import (
    STAGES, fetch, hebrew, manifest, mix, report, segments, stems, timeline, tools,
    transcript, translate,
)
from . import tts as tts_mod

REPO_ROOT = Path(__file__).resolve().parents[1]

# Legacy ISO-639 spellings mean the same language to us, to whisper and to the script
# table. Normalised once, at the entry point, so exactly one spelling of each language
# reaches the manifest otherwise `--src iw --tgt he` looks like a cross-language pair.
LANG_ALIASES = {"iw": "he", "ji": "yi", "in": "id"}

# The languages this pipeline is actually built for: a source needs an ASR that
# reads it and a `script` bucket that is true of it; a target needs a Qwen3-TTS
# voice as well (which is why Arabic reads but does not speak).
#
# They are `choices` on --src/--tgt because `script.script_for` answers "latin"
# for anything it does not know: a mistyped `--src jp` would not fail, it would
# quietly claim the video is written in Latin script, and every script-derived
# verdict downstream — keep vs dub, gloss matching, source-leak detection —
# would inherit the lie. Refusing the typo at the command line is the only place
# it is still cheap.
SRC_LANGS = ("he", "en", "ar", "ru", "fr", "es", "de", "it", "pt", "zh", "ja", "ko")
TGT_LANGS = ("en", "he", "ru", "fr", "es", "de", "it", "pt", "zh", "ja", "ko")


def _lang_choices(langs: tuple[str, ...]) -> tuple[str, ...]:
    """`langs` plus the legacy spellings `normalize_lang` folds into them.

    argparse checks `choices` before `normalize_lang` ever runs (it is applied in
    `main`, after the parse), so `--src iw` has to be accepted *here* or the alias
    the whole pipeline goes out of its way to support would die at the front door.
    """
    return tuple(sorted(set(langs) | {a for a, c in LANG_ALIASES.items() if c in langs}))


SRC_CHOICES = _lang_choices(SRC_LANGS)
TGT_CHOICES = _lang_choices(TGT_LANGS)


def normalize_lang(code: str) -> str:
    code = (code or "").strip().lower()
    return LANG_ALIASES.get(code, code)


def check_langs(args: argparse.Namespace) -> None:
    """Refuse a language pair this machine cannot actually dub, and say why.

    Hebrew is a target only when its two local models are present the Qwen3-TTS
    Hebrew LoRA and the G2P that feeds it stressed IPA (see `dubbing/hebrew.py`).
    A run that discovered this at the tts stage would have already paid for stems,
    ASR, diarization and translation.

    A same-language pair is *not* refused: `--src he --tgt he` is a dub, not a
    mistake every speech segment is re-voiced in the cloned voice with no
    translation step (see `translate.run`).
    """
    if hebrew.is_hebrew(args.tgt):
        hebrew.require()
        if args.tts_model != hebrew.ADAPTER_MODEL:
            raise SystemExit(
                f"target 'he' needs --tts-model {hebrew.ADAPTER_MODEL}: the Hebrew "
                f"adapter was trained against that checkpoint's talker and carries its "
                f"output heads, so it does not fit the {args.tts_model} one.")


def check_transcript(args: argparse.Namespace) -> None:
    """Refuse an unreadable `--transcript file` before the fetch stage runs.

    Same reasoning as `check_langs`: the file is not read until the transcript
    stage, which is two stages and a Demucs separation later. A typo'd path or a
    .txt with no timestamps in it costs a sentence here and a quarter of an hour
    there.
    """
    if args.transcript != "file":
        return
    if not args.captions:
        raise SystemExit("--transcript file needs --captions <file.srt|.vtt|.json3>: "
                         "there is no transcript to read without one.")
    try:
        transcript.check_transcript_file(Path(args.captions))
    except transcript.TranscriptFileError as exc:
        raise SystemExit(str(exc)) from exc


def default_workdir(source: str) -> Path:
    if fetch.is_url(source):
        vid = re.search(r"(?:v=|youtu\.be/|/shorts/|/live/)([A-Za-z0-9_-]{6,})", source)
        slug = vid.group(1) if vid else re.sub(r"\W+", "_", source)[-24:]
        return REPO_ROOT / "outputs" / f"yt_{slug}"
    return REPO_ROOT / "outputs" / re.sub(r"\W+", "_", Path(source).stem)[:48]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m dubbing",
                                description="Dub a video into another language, locally.")
    p.add_argument("source", help="YouTube URL or local video file")
    p.add_argument("-o", "--out", type=Path, help="work directory (default: outputs/<slug>)")
    p.add_argument("--captions", type=Path,
                   help="a transcript you already have (.srt, .vtt or .json3) to read "
                        "instead of the video's own captions; see --transcript file")
    # None, not the default value — same rule as RECORDED_DEFAULTS below: a
    # bare re-run must keep THIS run's languages and cap, and argparse filling
    # in he/en/full-length is indistinguishable from the user typing them. The
    # he→en/full-length effective defaults are applied in `resolve_settings`.
    # `choices` is safe beside `default=None`: argparse only checks a value it
    # actually parsed, never the default, so the None sentinel resolve_settings
    # depends on still reaches it untouched.
    p.add_argument("--src", default=None, choices=SRC_CHOICES,
                   help="source language code (default: he)")
    p.add_argument("--tgt", default=None, choices=TGT_CHOICES,
                   help="target language code (default: en)")
    p.add_argument("--duration", type=float, default=None,
                   help="only dub the first N seconds (0 = the whole video, "
                        "which also clears a recorded cap on a re-run)")
    p.add_argument("--context", help="one-line note on who/what the video is about and "
                   "the spellings of names the ASR mangles steers the translator")
    # Every option in RECORDED_DEFAULTS defaults to None, not to its value: the
    # value is applied by `resolve_settings`, which cannot tell "the user typed the
    # default" from "the user typed nothing" once argparse has filled one in and
    # the difference is whether a re-run keeps this run's settings or overwrites
    # them with the defaults. The effective default is in RECORDED_DEFAULTS.
    p.add_argument("--register", choices=("narration", "dialogue"), default=None,
                   help="translation speaking style: 'narration' (default, full words) "
                        "or 'dialogue' (natural spoken register, e.g. English "
                        "contractions)")
    p.add_argument("--genre", choices=("documentary", "movie"), default=None,
                   help="content genre: 'documentary' (default, current behavior) or "
                        "'movie' (dialogue register, gentler time-compression with "
                        "rate continuity, and short greeting/interjection beats keep "
                        "the actor's original voice)")
    p.add_argument("--dub-foreign", action=argparse.BooleanOptionalAction, default=None,
                   help="dub confident third-language passages into the target instead "
                        "of keeping original audio with a subtitle (default: off; "
                        "--no-dub-foreign turns it back off on a run that recorded it)")
    p.add_argument("--transcript", choices=("auto", "captions", "asr", "file"), default=None,
                   help="where the transcript comes from (default: captions if present). "
                        "'file' reads the --captions file and nothing else, and fails "
                        "rather than falling back to the ASR")
    p.add_argument("--stages", help=f"comma-separated subset of: {','.join(STAGES)}")
    p.add_argument("--force", help="stage to re-run, or 'all'")
    p.add_argument("--device", help="torch device override for TTS")
    p.add_argument("--tts-model", choices=("1.7b", "0.6b"), default=None,
                   help="Qwen3-TTS voice-clone model (default 1.7b; 0.6b accepted "
                        "only so old runs re-run)")
    # Deliberately NOT in RECORDED_DEFAULTS. Every other option here describes
    # the dub; this one describes the machine, and a manifest that remembered it
    # would hand a 12 GB card's setting to the 48 GB card the project was copied
    # to. It is also kept out of the translate fingerprint for the same reason: a
    # machine change must not silently re-translate a finished project.
    p.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=None,
                   help="load smaller translator weights so an ordinary GPU (or a "
                        "16 GB Mac) can run the 12B translator: 4-bit on CUDA, mxfp4 "
                        "on Apple Silicon. Costs translation quality. Detected "
                        "automatically on a small card; --no-low-vram refuses the "
                        "detection and keeps the full-precision weights")
    return p.parse_args(argv)


# Files a stage must have left on disk to count as done. Empty for the stages whose
# whole result lives in the manifest which is why forcing one has to invalidate
# the rest explicitly; see `apply_force`.
STAGE_OUTPUTS: dict[str, list[str]] = {
    "fetch": ["source.wav"],
    "stems": ["stems/vocals.wav", "stems/background.wav"],
    "transcript": ["words.json"],
    "segments": [],
    "translate": [],
    "tts": [],
    "timeline": [],
    "mix": ["dub.wav", "preview.mp4"],
    "report": ["report.json"],
}


# Settings that shape the output and are not otherwise recoverable from the
# manifest, with the value each one takes when nobody has an opinion.
# `dubbing.edit._args` reads them back to reproduce this run's own arguments;
# without them the studio silently falls back to argparse's defaults, re-places a
# movie timeline at documentary rates and stamps rebuilt stages with fingerprints
# this CLI would never compute.
RECORDED_DEFAULTS: dict[str, Any] = {
    "register": "narration",
    "genre": "documentary",
    "transcript": "auto",
    "tts_model": "1.7b",
    "dub_foreign": False,
}
RECORDED_SETTINGS = tuple(RECORDED_DEFAULTS)


def resolve_settings(args: argparse.Namespace, m: dict[str, Any] | None = None) -> None:
    """Fill in the options nobody typed: this run's own, then the built-in default.

    Three sources, in order of authority: the flag on this command line, what the
    manifest recorded about the run being re-run, and RECORDED_DEFAULTS. Without
    the middle one a bare `python -m dubbing <input>` on an existing run was not a
    re-run at all argparse's defaults overwrote `m["source"]`, which flipped the
    segments/translate/timeline fingerprints, and a changed segments fingerprint
    empties `m["segments"]`: every edit, lock and passthrough in the project, gone,
    for typing the command without its flags.
    """
    recorded = (m or {}).get("source") or {}
    # The three options whose manifest names differ from their flag names — and
    # whose loss is the catastrophic kind: --src/--tgt falling back to he/en
    # flips the language pair of any other-pair project, and a lost --duration
    # turned a 320-second iteration run into the whole 57-minute episode.
    # "not" rather than "is None": `normalize_lang` has already run by the time
    # main() calls this, and it folds an untyped None into "".
    # Re-normalize after the restore: the typed flags went through
    # `normalize_lang` in main(), but a manifest value never written by this
    # CLI (an API-created project, a hand-restored dir) can hold an alias like
    # "iw" — and `edit._args` normalizes its copy, so skipping it here made
    # the same run fingerprint differently in the studio and the CLI.
    if not args.src:
        args.src = normalize_lang(recorded.get("src_lang") or "he")
    if not args.tgt:
        args.tgt = normalize_lang(recorded.get("tgt_lang") or "en")
    if args.duration is None:
        args.duration = recorded.get("duration_limit")
    elif args.duration <= 0:
        args.duration = None               # typed 0: dub the whole video
    # Same class as --duration: `captions` is in fetch's fingerprint, so a bare
    # re-run that forgot it flips fetch — and a flipped chain empties
    # `m["segments"]` at the segments stage, taking every lock and hand-edit
    # with it, while fetch quietly falls back to ASR.
    if args.captions is None and recorded.get("captions"):
        args.captions = Path(recorded["captions"])
    for key, fallback in RECORDED_DEFAULTS.items():
        if getattr(args, key, None) is not None:
            continue                       # typed on this command line: it wins
        value = recorded.get(key)
        setattr(args, key, fallback if value is None else value)


def source_record(args: argparse.Namespace) -> dict[str, Any]:
    """What `m["source"]` must remember about the run that made this manifest.

    Call `resolve_settings` first: a `None` here would record "no opinion" as this
    run's setting, and the next run would read it back as one.
    """
    rec: dict[str, Any] = {"input": args.source, "src_lang": args.src,
                           "tgt_lang": args.tgt, "duration_limit": args.duration,
                           "captions": str(args.captions) if args.captions else ""}
    rec.update({key: getattr(args, key) if getattr(args, key) is not None else default
                for key, default in RECORDED_DEFAULTS.items()})
    return rec


def captions_key(path: Path | None) -> str:
    """A supplied transcript's identity for the fetch fingerprint: path *and* bytes.

    The path alone cannot tell one edit of a file from another, and a user who
    fixes the mangled spelling of a name in their own .srt and re-runs is doing
    exactly that: same path, new words. Cached on the path, the run would hand
    back the transcript built from the old file and every stage after it. So the
    contents are hashed into the value, and correcting the file invalidates the
    fetch, the transcript and everything downstream — which is what a changed
    input is supposed to do.

    An unreadable file degrades to the bare path: the fingerprint stays stable up
    to the `SystemExit` the fetch stage is about to raise about it anyway. No
    captions is still the empty string, so a run that never used one fingerprints
    today exactly as it did before this existed.
    """
    if not path:
        return ""
    try:
        digest = hashlib.sha1(Path(path).expanduser().read_bytes()).hexdigest()[:16]
    except OSError:
        return str(path)
    return f"{path}#{digest}"


def stage_params(args: argparse.Namespace, m: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Everything that, when changed, must invalidate a stage's cached result.

    Shared with `dubbing.edit.rebuild`, so the app re-runs a stage under exactly the
    fingerprint the CLI would compute for it.
    """
    return {
        "fetch": {"source": args.source, "captions": captions_key(args.captions),
                  "duration": args.duration, "src": args.src},
        "stems": {},
        # `origin` is this stage's own output (ASR, or the captions fallback when
        # ASR was unavailable), and the two produce different words for the same
        # parameters so a run whose transcript source changed invalidates
        # everything built on it, instead of caching a degraded transcript forever.
        "transcript": {"src": args.src, "tgt": args.tgt, "prefer": args.transcript,
                       "origin": m["source"].get("transcript_origin")},
        # segments reads tgt_lang from the manifest, so the pair must be in its
        # fingerprint with params={} changing --tgt never invalidated it.
        "segments": {"src": args.src, "tgt": args.tgt, "dub_foreign": args.dub_foreign,
                     "genre": args.genre},
        "translate": {"src": args.src, "tgt": args.tgt,
                      "context": m["source"].get("context") or "",
                      "register": args.register, "genre": args.genre},
        # Two version knobs that change the audio and are not otherwise in here:
        # the reference-building recipe (`m["speakers"]` survives a tts reset, so
        # nothing else would notice a new one) and the Hebrew LoRA's tag. Bumping
        # either used to leave every existing clip in place `needs_synthesis`
        # answers by file existence and quietly ship the old voice.
        "tts": {"model": args.tts_model, "tgt": args.tgt,
                "refs": tts_mod.Engine.REF_BUILD,
                **({"adapter": hebrew.ADAPTER_TAG} if hebrew.is_hebrew(args.tgt) else {})},
        "timeline": {"genre": args.genre},
        "mix": {},
        "report": {},
    }


def apply_force(m: dict[str, Any], force: str | None) -> list[str]:
    """Honour `--force <stage>`: re-run that stage AND everything after it.

    Downstream does not fall out of the fingerprint chain here. Forcing re-runs a
    stage with the same tag, the same params and the same upstream fingerprint, so
    every downstream fingerprint recomputes identical and translate/tts/timeline
    declare no output files, so `stage_done`'s `all([])` is True and they print "up
    to date" and skip. The re-run really does discard what they were built on, so
    their records are dropped explicitly.

    Returns the stages invalidated, forced stage first.
    """
    if not force:
        return []
    if force == "all":
        m["stages"], m["progress"] = {}, {}
        return list(STAGES)
    if force not in STAGES:
        raise SystemExit(f"unknown stage {force!r}; choose from {', '.join(STAGES)}")
    manifest.clear_stage(m, force)
    return [force] + manifest.clear_downstream(m, force)


def report_failed(result: dict[str, Any]) -> bool:
    """Whether a finished run has nothing worth handing over.

    Two ways to reach the end of every stage with no usable dub, and the exit
    code is the only channel that carries either one out of here: the app's job
    runner reads it, `dubbing_app/worker.py` turns it into a failed job, and the
    UI paints the stage red. Everything else the report has to say is a warning.

    * `unaccounted` a segment with no audio behind it, so the mix has a hole.
    * `tts_unavailable` nothing was synthesized at all, so the "dub" is the
      source audio in the source language (issue #15). Read with `.get`: a
      report from before this field existed still gets judged on the rest.
    """
    return bool(result["unaccounted"] or result.get("tts_unavailable"))


def _open_run(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    """Resolve the work directory, load or create its manifest, settle the options.

    `resolve_settings` runs before anything reads a setting and before anything is
    recorded: an option this command line did not carry is the one this run was made
    with, not argparse's default. `check_langs` reads `--tts-model`, so it waits for
    that.
    """
    workdir = (args.out or default_workdir(args.source)).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    m = manifest.load(workdir)
    resolve_settings(args, m)
    check_langs(args)
    check_transcript(args)

    m = m or manifest.new(source_record(args))
    m["source"].update(source_record(args))
    if args.context is not None:
        m["source"]["context"] = args.context
    return workdir, m


def _honour_passthrough(m: dict[str, Any]) -> dict[str, Any]:
    """Apply the editor's per-segment overrides, and reopen what a flip invalidated.

    The editor writes `passthrough` into a finished manifest and re-runs. Honour it
    before anything is skipped as up to date: a flip changes what the translate, tts,
    timeline and mix stages produced, so their "done" marks come off and they run
    again. Their *progress* marks stay, so the stages resume rather than restart —
    only the flipped segments lost their work (see `segments.apply_passthrough`), and
    every other line keeps its translation and its clip. Nothing happens at all when
    no override changed a verdict.

    Returns the saved overrides, which the segments stage re-attaches by time.
    """
    saved = m.get("segments") or []
    overrides = segments.saved_overrides(saved)
    flipped = segments.apply_passthrough(saved)
    if flipped:
        print(f"passthrough: {len(flipped)} segment(s) re-decided by the user "
              f"({', '.join(str(i) for i in flipped[:8])}"
              f"{'…' if len(flipped) > 8 else ''}) redoing from translate",
              file=sys.stderr)
        manifest.reopen_from(m, "translate")
    return overrides


def _selected_stages(args: argparse.Namespace) -> set[str]:
    """The `--stages` subset, or every stage. Refuses a name that is not one."""
    selected = set(args.stages.split(",")) if args.stages else set(STAGES)
    unknown = selected - set(STAGES)
    if unknown:
        raise SystemExit(f"unknown stage(s): {', '.join(sorted(unknown))}")
    return selected


@dataclass
class _Run:
    """What the stage loop carries from one stage to the next.

    `engine` and `words` are the two things a stage hands to a later one in memory
    rather than through the manifest: the TTS engine so timeline does not reload the
    synthesiser, and the word stream so segments does not re-read `words.json`.
    """

    m: dict[str, Any]
    workdir: Path
    args: argparse.Namespace
    overrides: dict[str, Any]
    save: Callable[[], None]
    engine: tts_mod.Engine | None = None
    words: list[dict[str, Any]] | None = None


def _run_stage(run: _Run, stage: str) -> dict[str, Any] | None:
    """Run one stage. Returns the report's result for `report`, else None."""
    m, workdir, args = run.m, run.workdir, run.args
    if stage == "fetch":
        fetch.run(m, workdir, source=args.source, captions_file=args.captions,
                  duration_limit=args.duration, src_lang=args.src)
    elif stage == "stems":
        stems.run(m, workdir)
    elif stage == "transcript":
        transcript.run(m, workdir, src_lang=args.src, tgt_lang=args.tgt,
                       prefer=args.transcript)
    elif stage == "segments":
        run.words = run.words or transcript.load_words(workdir, m)
        segments.run(m, workdir, run.words, transcript.load_foreign_spans(workdir, m),
                     dub_foreign=args.dub_foreign, genre=args.genre,
                     overrides=run.overrides,
                     lang_runs=transcript.load_lang_runs(workdir, m))
    elif stage == "translate":
        translate.run(m, workdir, source=args.src, target=args.tgt, save=run.save,
                      register=args.register, genre=args.genre)
    elif stage == "tts":
        run.engine = tts_mod.run(m, workdir, save=run.save, device=args.device,
                                 model=args.tts_model)
    elif stage == "timeline":
        run.engine = run.engine or tts_mod.Engine(m, workdir, device=args.device,
                                                  model=args.tts_model)
        shorten_many, resynth_many = _retimers(m, run.engine, args)
        timeline.run(m, workdir, shorten_many=shorten_many, resynth_many=resynth_many,
                     genre=args.genre)
    elif stage == "mix":
        # The synthesiser is done with; free the device before ffmpeg runs.
        if run.engine is not None:
            run.engine.close()
            run.engine = None
        mix.run(m, workdir)
    elif stage == "report":
        return report.run(m, workdir)
    return None


def _load_dotenv() -> None:
    """Best effort: `.env` holds HF_TOKEN, and a run without one still works."""
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    tools.utf8_stdio()          # a Windows console is not UTF-8; every stage prints Hebrew
    args = parse_args(argv)
    args.src, args.tgt = normalize_lang(args.src), normalize_lang(args.tgt)
    _load_dotenv()

    # After load_dotenv, so a `--low-vram` on the command line still beats a
    # DUBBING_LOW_VRAM in the .env the app writes, and before any stage runs, so
    # the choice is settled the first time anything loads the translator.
    translate.set_low_vram(args.low_vram)

    workdir, m = _open_run(args)
    overrides = _honour_passthrough(m)
    apply_force(m, args.force)
    selected = _selected_stages(args)

    def save() -> None:
        manifest.save(workdir, m)

    params = stage_params(args, m)
    run = _Run(m=m, workdir=workdir, args=args, overrides=overrides, save=save)

    for stage in STAGES:
        fp = manifest.stage_fingerprint(m, stage, params[stage])
        if stage not in selected:
            if stage not in m["stages"]:
                print(f"stopping before {stage!r}: not selected and never run", file=sys.stderr)
                save()
                return 0
            continue
        if manifest.stage_done(m, workdir, stage, fp, STAGE_OUTPUTS[stage]):
            # A matching fingerprint is not the whole truth for translate: a
            # user-locked dub whose translation failed is left visibly
            # unfinished (keep=false, no text_en) rather than reverted, and
            # that unfinished work must reopen the stage or no re-run ever
            # retries it.
            holes = translate.untranslated(m["segments"]) if stage == "translate" else []
            if not holes:
                print(f"[{stage}] up to date", file=sys.stderr)
                manifest.mark_stage(m, stage, fp)
                continue
            print(f"[translate] {len(holes)} line(s) still untranslated re-entering",
                  file=sys.stderr)
            # The trap `apply_force` documents, through this door too: the
            # re-entry runs translate under an *unchanged* fingerprint, so
            # every downstream fingerprint recomputes identical and tts/
            # timeline/mix/report would all report "up to date" — a line the
            # re-entry successfully translates would keep playing its
            # original-audio fallback forever. Reopen them explicitly.
            manifest.clear_downstream(m, "translate")

        print(f"[{stage}]", file=sys.stderr)
        t0 = time.time()
        # Same fingerprint as the last attempt → resume its partial work.
        # Different → that work was produced by different logic or inputs, drop it.
        if m.setdefault("progress", {}).get(stage) != fp:
            manifest.reset_stage(m, stage)
            m["progress"][stage] = fp
            save()

        result = _run_stage(run, stage)

        if stage == "transcript":
            # This stage's fingerprint contains its own verdict about where the
            # words came from, so the mark has to describe what it produced, not
            # what was known before it ran otherwise a first run would spend the
            # next one redoing the whole pipeline to reach the same answer.
            params = stage_params(args, m)
            fp = manifest.stage_fingerprint(m, stage, params[stage])
            if transcript.is_fallback(m, args.transcript):
                print("  transcript: this is the captions fallback, not ASR the "
                      "next run will try the ASR again", file=sys.stderr)
                manifest.mark_provisional(m, stage, fp)
                save()
                print(f"[{stage}] done in {time.time() - t0:.0f}s", file=sys.stderr)
                continue

        manifest.mark_stage(m, stage, fp)
        save()
        if stage == "report":
            # The report's own verdict is the exit code, and it gets no "done in"
            # line: what it took to count is not news.
            if report_failed(result):
                return 1
            continue
        print(f"[{stage}] done in {time.time() - t0:.0f}s", file=sys.stderr)

    print(f"\nPreview: {workdir / 'preview.mp4'}", file=sys.stderr)
    return 0


def _retimers(m: dict[str, Any], engine: tts_mod.Engine | None,
              args: argparse.Namespace) -> tuple[Callable | None, Callable | None]:
    """Callbacks the timeline uses when a line must be shortened to fit.

    Batched per round so each model is loaded once. Where translator and TTS
    compete for one device (MLX unified memory, single GPU) the synthesiser is
    released while the translator runs, and vice versa; on a multi-GPU box each
    keeps its own device and both stay resident.

    A same-language run has none: `(None, None)` tells `timeline.run` not to ask,
    and it absorbs overhang with speed-up and drift instead which `place` still
    keeps non-overlapping, so the invariant is untouched. Shortening a line here
    would mean loading Gemma to *rewrite the speaker's own words in their own
    language*, which is a rewrite, not a translation, and a same-language dub says
    the same number of words in about the same time anyway.
    """
    if translate.same_language(args.src, args.tgt):
        return None, None

    def shorten_many(requests):
        if translate.exclusive_device():
            engine.close()
        segs = m["segments"]
        # preceding is SOURCE-language text by convention see translate._PRECEDING.
        before = {s["id"]: prev["text"] for prev, s in zip(segs, segs[1:])}
        # On a pivot run the shorten re-translates from the English intermediate
        # (the measured-good line), so its preceding context is English too.
        before_mid = {s["id"]: prev.get("text_mid") or ""
                      for prev, s in zip(segs, segs[1:])}
        pivot = translate.pivot_via_english(args.src, args.tgt)
        out: dict[int, str | None] = {}
        with translate.loaded() as h:
            for seg, max_words in requests:
                # A pivot run re-shortens the English intermediate (the line that
                # was measured), not the source; everything else is the same call.
                use_mid = pivot and bool((seg.get("text_mid") or "").strip())
                text = seg["text_mid"] if use_mid else seg["text"]
                source = "en" if use_mid else args.src
                preceding = (before_mid if use_mid else before).get(seg["id"], "")
                out[seg["id"]] = translate.shorten(
                    h.processor, h.model, text, seg["text_en"], max_words,
                    source=source, target=args.tgt,
                    context=m["source"].get("context") or "",
                    preceding=preceding, device=h.device,
                )
                if not out[seg["id"]]:
                    print(f"  timeline: seg {seg['id']} kept full length "
                          "(no safe shorter translation)", file=sys.stderr)
        # `translate.loaded.__exit__` drops the only names, collects, and
        # clears the pool for the resynth pass — the ritual lives there now.
        return out

    def resynth_many(items):
        return {seg["id"]: engine.clip_for(seg, text) for seg, text in items}

    return shorten_many, resynth_many


if __name__ == "__main__":
    raise SystemExit(main())
