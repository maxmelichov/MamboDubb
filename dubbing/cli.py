"""Entry point: `uv run python -m dubbing <url-or-file>`.

Stages run in order. Each one is skipped when its fingerprint and outputs are
already on disk, so re-running is cheap and an interrupted run resumes where it
stopped. `--force <stage>` re-runs that stage; everything downstream of it
invalidates automatically through the fingerprint chain.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any

from . import STAGES, fetch, manifest, mix, report, segments, stems, timeline, transcript, translate
from . import tts as tts_mod

REPO_ROOT = Path(__file__).resolve().parents[1]


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
    p.add_argument("--captions", type=Path, help="json3 caption file for a local video")
    p.add_argument("--src", default="he", help="source language code (default: he)")
    p.add_argument("--tgt", default="en", help="target language code (default: en)")
    p.add_argument("--duration", type=float, help="only dub the first N seconds")
    p.add_argument("--context", help="one-line note on who/what the video is about and "
                   "the spellings of names the ASR mangles — steers the translator")
    p.add_argument("--register", choices=("narration", "dialogue"), default="narration",
                   help="translation speaking style: 'narration' (default, full words) "
                        "or 'dialogue' (natural spoken register, e.g. English "
                        "contractions)")
    p.add_argument("--genre", choices=("documentary", "movie"), default="documentary",
                   help="content genre: 'documentary' (default, current behavior) or "
                        "'movie' (dialogue register, gentler time-compression with "
                        "rate continuity, and short greeting/interjection beats keep "
                        "the actor's original voice)")
    p.add_argument("--dub-foreign", action="store_true",
                   help="dub confident third-language passages into the target instead "
                        "of keeping original audio with a subtitle")
    p.add_argument("--transcript", choices=("auto", "captions", "asr"), default="auto",
                   help="where the transcript comes from (default: captions if present)")
    p.add_argument("--stages", help=f"comma-separated subset of: {','.join(STAGES)}")
    p.add_argument("--force", help="stage to re-run, or 'all'")
    p.add_argument("--device", help="torch device override for TTS")
    p.add_argument("--tts-model", choices=("1.7b", "0.6b"), default="1.7b",
                   help="Qwen3-TTS voice-clone model (default 1.7b; 0.6b accepted only so old runs re-run)")
    return p.parse_args(argv)


# Files a stage must have left on disk to count as done. Empty for the stages whose
# whole result lives in the manifest — which is why forcing one has to invalidate
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
# manifest. `dubbing.edit._args` reads them back to reproduce this run's own
# arguments; without them the studio silently falls back to argparse's defaults,
# re-places a movie timeline at documentary rates and stamps rebuilt stages with
# fingerprints this CLI would never compute.
RECORDED_SETTINGS = ("register", "genre", "transcript", "tts_model", "dub_foreign")


def source_record(args: argparse.Namespace) -> dict[str, Any]:
    """What `m["source"]` must remember about the run that made this manifest."""
    rec: dict[str, Any] = {"input": args.source, "src_lang": args.src,
                           "tgt_lang": args.tgt, "duration_limit": args.duration}
    rec.update({key: getattr(args, key) for key in RECORDED_SETTINGS})
    return rec


def stage_params(args: argparse.Namespace, m: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Everything that, when changed, must invalidate a stage's cached result.

    Shared with `dubbing.edit.rebuild`, so the app re-runs a stage under exactly the
    fingerprint the CLI would compute for it.
    """
    return {
        "fetch": {"source": args.source, "captions": str(args.captions or ""),
                  "duration": args.duration, "src": args.src},
        "stems": {},
        "transcript": {"src": args.src, "tgt": args.tgt, "prefer": args.transcript},
        # segments reads tgt_lang from the manifest, so the pair must be in its
        # fingerprint — with params={} changing --tgt never invalidated it.
        "segments": {"src": args.src, "tgt": args.tgt, "dub_foreign": args.dub_foreign,
                     "genre": args.genre},
        "translate": {"src": args.src, "tgt": args.tgt,
                      "context": m["source"].get("context") or "",
                      "register": args.register, "genre": args.genre},
        "tts": {"model": args.tts_model, "tgt": args.tgt},
        "timeline": {"genre": args.genre},
        "mix": {},
        "report": {},
    }


def apply_force(m: dict[str, Any], force: str | None) -> list[str]:
    """Honour `--force <stage>`: re-run that stage AND everything after it.

    Downstream does not fall out of the fingerprint chain here. Forcing re-runs a
    stage with the same tag, the same params and the same upstream fingerprint, so
    every downstream fingerprint recomputes identical — and translate/tts/timeline
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except Exception:
        pass

    workdir = (args.out or default_workdir(args.source)).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    m = manifest.load(workdir) or manifest.new(source_record(args))
    m["source"].update(source_record(args))
    if args.context is not None:
        m["source"]["context"] = args.context

    # The editor writes `passthrough` into a finished manifest and re-runs.
    # Honour it before anything is skipped as up to date: a flip changes what the
    # translate, tts, timeline and mix stages produced, so their "done" marks come
    # off and they run again. Their *progress* marks stay, so the stages resume
    # rather than restart — only the flipped segments lost their work (see
    # segments.apply_passthrough), and every other line keeps its translation and
    # its clip. Nothing happens at all when no override changed a verdict.
    overrides = segments.saved_overrides(m.get("segments") or [])
    flipped = segments.apply_passthrough(m.get("segments") or [])
    if flipped:
        print(f"passthrough: {len(flipped)} segment(s) re-decided by the user "
              f"({', '.join(str(i) for i in flipped[:8])}"
              f"{'…' if len(flipped) > 8 else ''}) — redoing from translate",
              file=sys.stderr)
        for stale in ("translate", "tts", "timeline", "mix", "report"):
            (m.get("stages") or {}).pop(stale, None)

    apply_force(m, args.force)

    selected = set(args.stages.split(",")) if args.stages else set(STAGES)
    unknown = selected - set(STAGES)
    if unknown:
        raise SystemExit(f"unknown stage(s): {', '.join(sorted(unknown))}")

    def save() -> None:
        manifest.save(workdir, m)

    params = stage_params(args, m)
    outputs = STAGE_OUTPUTS

    engine: tts_mod.Engine | None = None
    words: list[dict[str, Any]] | None = None

    for stage in STAGES:
        fp = manifest.stage_fingerprint(m, stage, params[stage])
        if stage not in selected:
            if stage not in m["stages"]:
                print(f"stopping before {stage!r}: not selected and never run", file=sys.stderr)
                save()
                return 0
            continue
        if manifest.stage_done(m, workdir, stage, fp, outputs[stage]):
            print(f"[{stage}] up to date", file=sys.stderr)
            manifest.mark_stage(m, stage, fp)
            continue

        print(f"[{stage}]", file=sys.stderr)
        t0 = time.time()
        # Same fingerprint as the last attempt → resume its partial work.
        # Different → that work was produced by different logic or inputs, drop it.
        if m.setdefault("progress", {}).get(stage) != fp:
            manifest.reset_stage(m, stage)
            m["progress"][stage] = fp
            save()

        if stage == "fetch":
            fetch.run(m, workdir, source=args.source, captions_file=args.captions,
                      duration_limit=args.duration, src_lang=args.src)
        elif stage == "stems":
            stems.run(m, workdir)
        elif stage == "transcript":
            transcript.run(m, workdir, src_lang=args.src, tgt_lang=args.tgt,
                           prefer=args.transcript)
        elif stage == "segments":
            words = words or transcript.load_words(workdir, m)
            segments.run(m, workdir, words, transcript.load_foreign_spans(workdir, m),
                         dub_foreign=args.dub_foreign, genre=args.genre,
                         overrides=overrides,
                         lang_runs=transcript.load_lang_runs(workdir, m))
        elif stage == "translate":
            translate.run(m, workdir, source=args.src, target=args.tgt, save=save,
                          register=args.register, genre=args.genre)
        elif stage == "tts":
            engine = tts_mod.run(m, workdir, save=save, device=args.device, model=args.tts_model)
        elif stage == "timeline":
            engine = engine or tts_mod.Engine(m, workdir, device=args.device, model=args.tts_model)
            shorten_many, resynth_many = _retimers(m, workdir, engine, args)
            timeline.run(m, workdir, shorten_many=shorten_many, resynth_many=resynth_many,
                         genre=args.genre)
        elif stage == "mix":
            if engine is not None:
                engine.close()
                engine = None
            mix.run(m, workdir)
        elif stage == "report":
            result = report.run(m, workdir)
            manifest.mark_stage(m, stage, fp)
            save()
            if result["unaccounted"]:
                return 1
            continue

        manifest.mark_stage(m, stage, fp)
        save()
        print(f"[{stage}] done in {time.time() - t0:.0f}s", file=sys.stderr)

    print(f"\nPreview: {workdir / 'preview.mp4'}", file=sys.stderr)
    return 0


def _retimers(m, workdir: Path, engine, args):
    """Callbacks the timeline uses when a line must be shortened to fit.

    Batched per round so each model is loaded once. Where translator and TTS
    compete for one device (MLX unified memory, single GPU) the synthesiser is
    released while the translator runs, and vice versa; on a multi-GPU box each
    keeps its own device and both stay resident.
    """

    def shorten_many(requests):
        if translate.exclusive_device():
            engine.close()
        processor, model, device = translate.load()
        segs = m["segments"]
        # preceding is SOURCE-language text by convention — see translate._PRECEDING.
        before = {s["id"]: prev["text"] for prev, s in zip(segs, segs[1:])}
        # On a pivot run the shorten re-translates from the English intermediate
        # (the measured-good line), so its preceding context is English too.
        before_mid = {s["id"]: prev.get("text_mid") or ""
                      for prev, s in zip(segs, segs[1:])}
        pivot = translate.pivot_via_english(args.src, args.tgt)
        out: dict[int, str | None] = {}
        try:
            for seg, max_words in requests:
                if pivot and (seg.get("text_mid") or "").strip():
                    out[seg["id"]] = translate.shorten(
                        processor, model, seg["text_mid"], seg["text_en"], max_words,
                        source="en", target=args.tgt,
                        context=m["source"].get("context") or "",
                        preceding=before_mid.get(seg["id"], ""), device=device,
                    )
                else:
                    out[seg["id"]] = translate.shorten(
                        processor, model, seg["text"], seg["text_en"], max_words,
                        source=args.src, target=args.tgt,
                        context=m["source"].get("context") or "",
                        preceding=before.get(seg["id"], ""), device=device,
                    )
                if not out[seg["id"]]:
                    print(f"  timeline: seg {seg['id']} kept full length "
                          "(no safe shorter translation)", file=sys.stderr)
        finally:
            translate.free(model)
        return out

    def resynth_many(items):
        return {seg["id"]: engine.clip_for(seg, text) for seg, text in items}

    return shorten_many, resynth_many


if __name__ == "__main__":
    raise SystemExit(main())
