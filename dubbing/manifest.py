"""manifest.json the single record every stage reads and writes.

Stage results are keyed by a fingerprint chain: each stage's fingerprint mixes in
its predecessor's, so re-running one stage with new params invalidates everything
downstream without any extra bookkeeping.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from . import STAGES

# Bump a stage tag when that stage's logic changes in a way that invalidates its
# previous output. Downstream stages invalidate automatically via the chain.
STAGE_TAGS = {
    "fetch": "fetch/v1",
    "stems": "stems/v1",
    # Two lines of development each claimed the same numbers for different logic
    # (the target-ASR witness vs handoff snapping at transcript/v39; lock-aware
    # reruns vs the news register at translate/v31; segments carried both
    # alignment+passthrough and the spoken_target/veto work), so the merged tag
    # moves past every claim rather than picking a side.
    "transcript": "transcript/v45",
    # v45: an unnamed ("und") foreign span no longer overrides a passage the main
    # source pass read confidently. The unnamed keep's whole premise is "no ASR
    # here reads this"; where the source ASR's own words say otherwise, six
    # seconds of correctly transcribed dialogue were being aired undubbed with a
    # "…" subtitle. Named verdicts are unchanged.
    # v44: a gap window is nominated by the mix and confirmed by the vocals. A
    # music-only sting is loud in the mix but digitally silent once separated,
    # and near-silence is what Whisper answers with "thank you very much"; the
    # window is now left uncovered (and reported) instead of decoded. Restores a
    # guard the pre-rewrite pipeline had and the rewrite dropped.
    # v43: the gap-recovery decoder listens to the separated vocals, like the
    # main pass — audibility is still judged from the mix, but "what is said"
    # is no longer asked of audio with the music still in it, which is where
    # the invented lines over stings ("thank you very much") came from.
    # v42: `recover_gaps` filters its decode per-segment through `speech_only`
    # before the window mean — a span the main pass declined as music could
    # re-enter through the gap pass whenever better-reading neighbours in the
    # same window carried the average.
    # v41: the main ASR pass gained the no-speech gate the gap pass already had —
    # a decode read below ASR_MIN_LOGPROB is a hallucination over music, not
    # speech, so a music-only intro yields no words instead of an invented line.
    # v39: a confident audio-LID witness naming a non-target language outranks
    # the "already target script" clause, exactly as a named span witness does —
    # nameless English lines in a he→de run were kept as "already German".
    # v40: CJK opens as a source. `SENTENCE_END` knows the fullwidth enders
    # (。！？, from `script.CJK_PUNCT` — the table TTS already read), so a Japanese
    # read splits per sentence instead of into mid-clause length chunks; and word
    # tokens join through `script.join_words`, which writes Han/kana without the
    # spaces neither language has.
    # v41: `merge_stranded_fragments` runs in the pipeline — a one-word opener
    # torn off by a buried middle folds into its continuation instead of being
    # voiced as a stranded beat — and it refuses spliced foreign-language spans,
    # so an English orphan never rides into a `lang`-tagged neighbour's voice.
    "segments": "segments/v42",
    # v42: a keep no longer walks its end over a word that belongs to another
    # segment. `merge_stranded_fragments` moves an orphaned word's TEXT into a
    # later segment but leaves the seconds it was spoken in claimed by nobody, and
    # the keep-tail walk marched straight through them so the line played twice,
    # once in the original voice and once dubbed.
    # v34: a same-language pair (he→he, en→en) translates by identity the target
    # line is the source line and no translator loads.
    # v35: a segment translates from its OWN language (`translate.segment_langs`),
    # and a failed translation no longer overrules the user's "dub it".
    # v36: the --context gloss floors are per script bucket instead of the
    # Hebrew-calibrated 4/5 (a 2-character 首相 could never match, and Korean
    # agglutination beat the exact-word rung), `_has_negation` reads Korean
    # negations glued inside a word, and `shorten`'s budget is in speech units —
    # counting `.split()` words made every CJK rewrite fail its own length check.
    # v37: "%" becomes the target's own spoken word for every TTS target (zh as
    # the 百分之 prefix over kept digits; numbers inside spaceless-script prose
    # now spell at all) — merged past a parallel claim on v36.
    # v38: Hebrew joined _PERCENT (אחוז), so "X%" in a Hebrew dub no longer
    # borrows English "percent" for tts.prepare_text to silently strip.
    "translate": "translate/v38",
    # Two lines each claimed tts/v15 for different logic (the Hebrew LoRA vs the
    # honest-failure records), so the merged tag moves past both.
    # v16: Hebrew targets synthesize from stressed IPA through the Qwen3-TTS Hebrew
    # LoRA and every clip's cache key carries the adapter and that IPA; records
    # carry the text fingerprint and (for keeps) the span and options they were cut
    # with; "unverified" is its own verdict; and a segment's own `tgt_lang` reaches
    # synthesis, the cache key and verification.
    # v17: a dub-wanted line with no translation replaces its record with the
    # original-audio slice unless the record already IS one — a stale synthesis
    # of a since-failed translation used to satisfy `keep_needs_slice` and keep
    # speaking the old line.
    # v18: the retry ladder varies the clone reference instead of the seed (which
    # is inert under a greedy decode), acceptance splits into a floor and a
    # CLONE_GOOD_OVERLAP retry bar ("accepted" is now its own verdict), and a
    # Hebrew target decodes greedily by default every clip's decode, reference
    # and verdict can differ from v17's.
    # v19: a retry reference is offered only when ECAPA says it is this segment's
    # own voice (a label is not a person), the alternates are held to the same
    # test, and a clip that says the right words in another voice is recorded as
    # "wrong_voice" rather than "ok" different rungs, different verdicts.
    "tts": "tts/v23",
    # v23: the Hebrew warm-up carrier is no longer refused on a distance threshold
    # against the ASR's reading of it. That threshold rejected the cut on 10 of 10
    # measured clips, so every Hebrew line fell back to the cold decode the carrier
    # exists to prevent and the feature shipped disabled by its own guard. The
    # alignment now ranks candidate cut points instead of certifying one, and the
    # only fatal check is the far-side one, which cuts and re-transcribes and keeps
    # the first boundary that really does open on the line. Hebrew clips change
    # audibly: the stutter inside אנשים and the devoiced /e/ of שֶׁ both go. As in
    # v21 this tag alone would not retire the cached clips, so hebrew.CARRIER_TAG
    # goes to v3 with it, and see there for why the v2 entries are the broken audio
    # rather than the fixed audio.
    # v22: a generated clip is trimmed at BOTH ends, not only the front. The
    # trailing hush averaged 0.12-0.22s across the demo runs, ~5% of clip length,
    # and the timeline measures the file: that silence was being time-compressed
    # as though it were words, so the speech was squeezed ~5% harder than it
    # needed to be and then finished early inside its own slot.
    # v21: the carrier's cut point is found by matching the carrier's characters
    # against the verification ASR, and the cut clip is re-transcribed and made to
    # open on the sentence's own first words before it is kept. v20 counted ASR
    # tokens, which is not a fixed number for a fixed phrase: on 2 of 8 clips of a
    # real run the count was short and the cut left the tail of the carrier on the
    # front of a clip that then passed verification. This tag alone would not have
    # retired those clips, since a re-run reads them out of the clip cache by key,
    # so hebrew.CARRIER_TAG goes to v2 with it.
    # v20: a Hebrew line is decoded behind a fixed warm-up carrier that is then cut
    # back off (hebrew.CARRIER_TEXT). The talker starts from the source speaker's
    # x-vector, so the first seconds of a cold decode carry their accent and not the
    # adapter's Hebrew. Measured, the LID posterior opens at 0.59 and only settles
    # past ~2.5s, which is the "first few words sound wrong" report. Warming the
    # context first moves the sentence's own onset +0.171 (p=0.019).
    # v14: `place` now carries the overrun it measured and why a shortening was
    # abandoned report.json's drift story reads from these. (v13 was claimed on
    # a branch that never landed under that number.)
    # v15: tts's fallback slices (keep_*.wav on a dub-wanted line) get a keep's
    # placement policy — never tempo-stretched (they are original audio, cut to
    # exactly their span; stretching also renamed them fit_keep_*, dropping the
    # UI's fallback flag).
    # v16: the shorten budget asked for is in speech units (`script.speech_units`),
    # so a CJK line is measured in characters. As words it was always 1, the budget
    # floored at 3, and `shorten` refused every rewrite — this stage's only rescue
    # for a late line was dead for zh/ja/ko.
    # v17: placement is anchored on the speech measured in the vocals stem rather
    # than on the ASR segment boundary, and a clip is fitted to the span the
    # speaker actually used. The two disagree systematically and in one direction
    # per source (a median 0.22s late on the he-source demo, 0.13s long on the
    # en-source one), so every dub in a run started early by about that much.
    # v18: the same-speaker tail is a fallback rather than a target. A clip
    # longer than its speaker's own span is now compressed onto that end whenever
    # the compression costs no more than RATE_PREF, and only one that cannot be
    # fitted that cheaply spends the tail. Aiming at own + TAIL_MAX put the mean
    # end-alignment error at 0.32s (he-ru) and 0.49s (en-ru), where a Russian
    # line is reliably a little longer than the one it dubs and so reliably
    # landed in the slack. Second half: a line still ending more than
    # OVERHANG_MAX past its own speaker after being compressed all the way to
    # RATE_MAX asks for a shorter translation of itself. That line never trips
    # DRIFT_MAX, because its overhang goes into the pause the source left rather
    # than into the next speaker, so the lateness trigger could not see it: it
    # fired on none of the four demo runs while 9 to 14 lines a run sat pinned.
    "timeline": "timeline/v18",
    "mix": "mix/v9",
    # v3: honest failure accounting verify.unverified, degraded, overrun,
    # shorten_abandoned, subtitles_failed, stale_locked_clips alongside the
    # transcript-origin stamping. (v2 was claimed for the first half alone.)
    # v4: source_mismatch the report says when most of the heard speech is
    # not written in the declared source language's script (wrong declaration).
    # v5: tts_unavailable a run where every line that was supposed to be dubbed
    # fell back to source audio is a stage that never ran, not a per-segment
    # fallback. It now names itself in `degraded` and exits the CLI nonzero
    # instead of reporting a clean run over an untranslated preview.
    "report": "report/v5",
}

# Anything not listed here is dropped on save. This is what stops the segment
# record from growing back into the 30-field soup of the old pipeline.
SEGMENT_KEYS = {
    "id",
    "uid",        # stable identity across re-segmentation; see `mint_uid`
    "start",
    "end",
    "speaker",
    "text",
    "keep",
    "keep_reason",
    "lang",       # third-language keeps only: what the span's speech is, for subtitles
    "src_lang",   # editor override: this segment's source language (falls back to `lang`)
    "tgt_lang",   # editor override: translate this segment into something else
    "detected_lang",  # advisory: what the language classifier heard over this span.
                      # Never decides anything it is what the editor app reads to
                      # SUGGEST passthrough ("this one is already English").
    "passthrough",    # user override: True = play the original audio for this span,
                      # False = dub it, absent = decide automatically. The pipeline
                      # honours it in segments.apply_passthrough; both front ends
                      # write THIS key (one concept, one manifest key).
    "text_en",
    "text_mid",   # pivot runs only: the English intermediate text_en was made from
    "tts_opts",   # per-segment synthesis overrides (dubbing/ttsopts.py); user-set only
    "tts",
    "locked",     # {field: True} for fields the user edited by hand see `is_locked`
    "place",
}

# Fields a user can lock. A locked field is never regenerated by a stage rerun:
# the user's edit outranks the pipeline (docs/APP_ARCHITECTURE.md, Non-negotiables).
# `text` and `speaker` have no generator to hold off, but the editor records them
# so a UI can show what was hand-edited.
#
# `place` and `bounds` are deliberately NOT here. Nothing could honour them:
# placement is all-or-nothing by design (`timeline.run` lays the whole run out in
# one forward pass and rewrites every `place`, which is what makes non-overlap
# provable), and re-segmentation rebuilds every span from the words, carrying only
# `passthrough` forward by time. A lock the pipeline overwrites on the next run is
# state that lies about the user's edit, so the API rejects it instead see
# `edit.set_locked`.
LOCK_FIELDS = ("text", "text_en", "tts", "keep", "speaker")


# Keeps whose `text_en` is written by the translate stage rather than copied from
# the segment's own text exactly `translate.needs_subtitle_translation`'s two
# kinds. Theirs is a translation, so a translate reset must reopen it; every other
# keep's subtitle belongs to the segments stage and is not translate's to discard.
SUBTITLED_BY_TRANSLATE = ("foreign", "interjection")


def is_locked(seg: dict[str, Any], field: str) -> bool:
    """True when the user hand-edited `field` on this segment."""
    return bool((seg.get("locked") or {}).get(field))


# A keep the pipeline decided for itself, per stage: the verdicts a rerun of that
# stage (or of anything upstream of it) is entitled to re-decide. One table, read
# by `reset_stage`, `edit.invalidate` and `tts.clear_failed_keeps` alike they
# undo the same verdict, so they cannot be allowed to disagree about which ones.
PIPELINE_KEEPS = {"translate": ("mt_failed", "tts_failed"), "tts": ("tts_failed",)}


def undo_pipeline_keep(seg: dict[str, Any], reasons: tuple[str, ...]) -> bool:
    """Reopen a keep the pipeline decided for itself. True when it was undone.

    The single predicate every undo path shares. `locked.keep` is the user
    answering that verdict "yes, keep the original here" and it outranks the
    rerun, whichever door the rerun came through. Guarding this with any *other*
    lock (the tts record's, say) is how a user-locked keep ended up silently
    re-decided on every tts run while `reset_stage` left it alone.

    Two nuances the lock alone gets wrong, so they live here too rather than in
    one caller:

    * A keep the *user* asked for (`passthrough` True) is never this stage's to
      reopen, whatever reason string is sitting on it un-keeping it would leave
      the manifest disagreeing with itself until the next `apply_passthrough`
      flipped it back.
    * A `locked.keep` guarding a value the *pipeline* put there is not the user's
      verdict at all: a failure keep written over a locked `keep=False` is how a
      span the user asked to dub stayed kept through every retry. So the undo
      looks past the lock at what the user actually said
      (`keep.user_wants_dub`), and old manifests carrying that contradiction
      repair themselves on first touch.
    """
    if seg.get("keep_reason") not in reasons or seg.get("passthrough") is True:
        return False
    from . import keep as keep_mod

    if is_locked(seg, "keep") and not keep_mod.user_wants_dub(seg):
        return False
    seg["keep"], seg["keep_reason"] = False, None
    return True


def keeps_own_subtitle(seg: dict[str, Any]) -> bool:
    """True when this keep's `text_en` is the segments stage's, not translate's.

    A kept span still gets a subtitle. For exactly the two kinds in
    `SUBTITLED_BY_TRANSLATE` that subtitle is a translation, so a translate reset
    reopens it; every other keep's line belongs to the segments stage and
    `translate.needs_translation` will never refill it dropped, it is gone for
    good. Shared by `reset_stage` (whole run) and `edit.invalidate` (one segment)
    so the two doors cannot drift apart.
    """
    return bool(seg.get("keep")) and seg.get("keep_reason") not in SUBTITLED_BY_TRANSLATE


def mint_uid(start: float, end: float, text: str) -> str:
    """The stable identity of a segment, derived from what created it.

    Opaque to every caller the only promise is that it is stable. Content-derived
    rather than random on purpose: a run directory written before `uid` existed
    gets back-filled on load with exactly the uid a fresh re-segmentation would
    mint, so identity survives both the migration and a re-run that reproduces the
    same segment. Segments never overlap, so (start, end, text) is unique within a
    run; `ensure_uids` re-mints the rare duplicate anyway.

    Minted once, at creation, and then carried unchanged through every edit a
    segment whose text the user rewrote keeps the uid its original text minted.
    """
    blob = f"{float(start):.3f}|{float(end):.3f}|{text or ''}"
    return "s" + hashlib.sha1(blob.encode("utf-8")).hexdigest()[:11]


def ensure_uids(segments: list[dict[str, Any]] | None) -> list[str]:
    """Give every segment a uid, keeping the ones already there. Returns those minted.

    Idempotent, so it is safe to call on load, on save, and at the end of the
    segments stage. A duplicate (a segment dict copied to make a trimmed piece)
    is re-minted from its own content, with a counter for the pathological case.
    """
    minted: list[str] = []
    seen: set[str] = set()
    for seg in segments or []:
        uid = seg.get("uid")
        if not uid or uid in seen:
            uid = mint_uid(seg.get("start", 0.0), seg.get("end", 0.0), seg.get("text", ""))
            n = 1
            while uid in seen:
                uid = mint_uid(seg.get("start", 0.0), seg.get("end", 0.0),
                               f"{seg.get('text', '')}#{n}")
                n += 1
            seg["uid"] = uid
            minted.append(uid)
        seen.add(uid)
    return minted


def new(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": 1,
        "source": source,
        "files": {},
        "stages": {},
        "progress": {},
        "speakers": {},
        "segments": [],
        "outputs": {},
    }


def reset_stage(m: dict[str, Any], stage: str) -> None:
    """Drop what `stage` produced, so it can be recomputed from scratch.

    Stages also resume *within* themselves (per-segment), so this is only called
    when the stage's fingerprint changed a resumed run keeps its partial work,
    a re-parameterised one does not.
    """
    if stage == "segments":
        m["segments"] = []
        m["speakers"] = {}
        return
    fields = {"translate": ("text_en", "text_mid"), "tts": ("tts",),
              "timeline": ("place",)}.get(stage)
    if not fields:
        return
    # Undo keep-flips this stage OR anything downstream of it made, so a rerun
    # re-decides them. Downstream flips must go too: a segment kept by tts_failed
    # still holds the translation that failed, and a translate reset that skips it
    # (because it looks "kept") re-feeds the same bad text to the new TTS run —
    # the downstream stage is guaranteed to rerun anyway once this one does.
    undo = PIPELINE_KEEPS.get(stage, ())
    # A hand-edited field is not this stage's to discard: a global rerun must never
    # silently overwrite a human correction. `timeline` has no entry: placement is
    # all-or-nothing and is never the user's (see LOCK_FIELDS).
    lock = {"translate": "text_en", "tts": "tts"}.get(stage)
    for seg in m.get("segments") or []:
        if lock and is_locked(seg, lock):
            continue
        undo_pipeline_keep(seg, undo)
        for field in fields:
            if stage == "translate" and field == "text_en" and keeps_own_subtitle(seg):
                # Structurally kept segments are subtitled by the segments stage;
                # the keeps whose subtitle comes *from* translate reset with it.
                continue
            seg.pop(field, None)


def load(workdir: Path) -> dict[str, Any] | None:
    path = workdir / "manifest.json"
    if not path.is_file():
        return None
    m = json.loads(path.read_text(encoding="utf-8"))
    # Migration: run dirs written before `uid` existed get theirs here, by the same
    # rule a fresh run would use, so the editor can key on it from the first load.
    ensure_uids(m.get("segments"))
    return m


def save(workdir: Path, m: dict[str, Any]) -> None:
    ensure_uids(m.get("segments"))
    for seg in m.get("segments") or []:
        for key in [k for k in seg if k not in SEGMENT_KEYS]:
            del seg[key]
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / "manifest.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def stage_fingerprint(m: dict[str, Any], stage: str, params: dict[str, Any]) -> str:
    idx = STAGES.index(stage)
    upstream = ""
    if idx:
        upstream = (m.get("stages") or {}).get(STAGES[idx - 1], {}).get("fp", "")
    blob = json.dumps(
        {"tag": STAGE_TAGS[stage], "params": params, "up": upstream},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def content_fingerprint(m: dict[str, Any]) -> str:
    """A hash of the decisions in the segment list, for artifacts written beside it.

    Not a stage fingerprint: those key on a stage's *parameters*, and a hand edit
    changes no parameter at all which is how `report.json` came to describe a
    manifest that had moved on hours ago. This keys on the segments themselves, so
    anything derived from them can say whether it is still about them.

    `stages` is deliberately not in it: `report.run` is called before its own stage
    is marked, so a fingerprint that included the marks could never match again.
    """
    blob = json.dumps(
        [{key: seg.get(key) for key in sorted(SEGMENT_KEYS)}
         for seg in m.get("segments") or []],
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def segment_digests(m: dict[str, Any]) -> dict[str, str]:
    """`uid` -> a hash of that one segment's decisions.

    `content_fingerprint` answers "did anything change"; this answers "which lines".
    An artifact that stores it alongside the fingerprint can say *how many* lines it
    is behind by, instead of guessing from how many are unfinished a guess that
    counts lines which were already unfinished when it was written and never changed.

    `id` is excluded where `content_fingerprint` includes it: it is positional and is
    renumbered by every split and merge, so keying on it would report every line after
    an inserted split as changed. `uid` is the identity here and is the key. Every
    other whitelisted field is in, including ones that do not reach the audio
    (`locked`): over-reporting a change costs a re-render the user chose, while
    under-reporting hides one they asked for.
    """
    out: dict[str, str] = {}
    keys = sorted(SEGMENT_KEYS - {"id", "uid"})
    for seg in m.get("segments") or []:
        uid = seg.get("uid")
        if not uid:
            continue
        blob = json.dumps({key: seg.get(key) for key in keys},
                          sort_keys=True, ensure_ascii=False)
        out[str(uid)] = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]
    return out


def digest_delta(before: dict[str, str] | None, after: dict[str, str]) -> int:
    """How many segments differ between two `segment_digests` maps.

    A uid on one side only counts once a line that was added or removed since is
    one line's worth of difference, not two.
    """
    if not before:
        return 0
    return sum(1 for uid in set(before) | set(after)
               if before.get(uid) != after.get(uid))


def stage_done(
    m: dict[str, Any], workdir: Path, stage: str, fp: str, outputs: list[str]
) -> bool:
    rec = (m.get("stages") or {}).get(stage)
    if not rec or rec.get("fp") != fp:
        return False
    return all((workdir / o).exists() for o in outputs)


def mark_stage(m: dict[str, Any], stage: str, fp: str) -> None:
    m.setdefault("stages", {})[stage] = {"fp": fp}


# Suffix on a stage mark that says "this ran, and it is not the answer that was
# asked for" see `mark_provisional`. Never equal to a recomputed fingerprint.
PROVISIONAL = "~provisional"


def mark_provisional(m: dict[str, Any], stage: str, fp: str) -> None:
    """Record what `stage` produced while saying it is not what was asked for.

    A transcript that fell back to captions because the ASR was unavailable is a
    real result everything downstream is built on it and must not be thrown
    away but it is not the transcript the run asked for, so the next run has to
    try again. A mark that can never equal a recomputed fingerprint says both at
    once: this stage re-runs, and the stages after it keep their place in the
    chain for as long as the fallback keeps producing the same thing. Marked, not
    left blank, so `--stages` still sees a stage that has run.
    """
    mark_stage(m, stage, fp + PROVISIONAL)


def clear_stage(m: dict[str, Any], stage: str) -> None:
    """Force a full redo of `stage`, discarding even resumable partial work."""
    (m.get("stages") or {}).pop(stage, None)
    (m.get("progress") or {}).pop(stage, None)


def unmark_stage(m: dict[str, Any], stage: str) -> bool:
    """Drop `stage`'s "done" mark, keeping the partial work it can resume from.

    The weaker half of `clear_stage`. A per-segment edit hollowed out one segment
    and left every other one intact, so the stage must *run again* but it must
    run as a resume, filling the hole, not as a restart that throws away the other
    seventy clips. `clear_stage` drops the progress mark too, which is what makes
    the next run call `reset_stage` and discard all of it.
    """
    return (m.get("stages") or {}).pop(stage, None) is not None


def reopen_from(m: dict[str, Any], stage: str) -> list[str]:
    """Un-mark `stage` and everything after it. Returns the stages reopened.

    What an edit owes the run: deleting one segment's translation, clip or
    placement leaves eight stages still saying "done", so the next run skips them
    all and the hole is never filled a segment with no placement is missing from
    the mix, which is the never-silent invariant broken by the back door.

    Per-segment work is untouched (`unmark_stage`, not `clear_stage`), so the
    reopened stages resume: they regenerate exactly what was deleted.
    """
    return [later for later in STAGES[STAGES.index(stage):] if unmark_stage(m, later)]


def clear_downstream(m: dict[str, Any], stage: str) -> list[str]:
    """Invalidate every stage after `stage`. Returns the stages cleared.

    The fingerprint chain cannot do this on its own when a stage is re-run with
    unchanged parameters: `stage_fingerprint` mixes tag + params + upstream fp, all
    three unchanged, so the recomputed fingerprint matches and translate/tts/
    timeline declare no output files, so `stage_done`'s `all([])` is True and they
    report "up to date". Re-running a stage genuinely does discard what came after
    it, so the record has to be dropped explicitly.
    """
    cleared = []
    for later in STAGES[STAGES.index(stage) + 1:]:
        clear_stage(m, later)
        reset_stage(m, later)
        cleared.append(later)
    return cleared
