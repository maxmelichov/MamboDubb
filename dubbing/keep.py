"""Which segments play their original audio instead of a dub — and the user's override.

Two decisions live here, and they are the same decision reached twice. `mark_keep`
is the pipeline's automatic verdict (span, script, speaker-prior rules); the
`passthrough` functions are the editor app's per-segment override of it. Both write
`seg["keep"]`, so tts slices the original audio, the timeline reserves the exact
span, and mix ducks the bed away instead of laying a dub over it.

Pure functions over segment dicts — no audio, no models. Split out of
`dubbing/segments.py`, which re-exports the five entry points the pipeline and the
front ends call as `segments.<name>`.
"""

from __future__ import annotations

from typing import Any

from . import PASSTHROUGH_REASON, script
from .interjections import is_interjection_keep

# The speaker prior: this much of a speaker's transcribed letters being in the
# target script makes them a target-language speaker, and their segments keep by
# default. Measured over the whole speaker, so it is a prior and not a verdict —
# see `mark_keep` for the two per-segment witnesses that can overturn it.
SPEAKER_EN_RATIO = 0.60
# ...and how sure the per-segment language witness must be to overturn that prior on
# one segment. Well above LID_MIN_PROB: every documented mislabel of the classifier
# sat at 0.34-0.60, and the cost of a wrong veto is dubbing over a speaker who really
# was speaking the target language — the bug this whole class is about.
SPEAKER_EN_VETO_PROB = 0.85

# Passthrough — the editor app's per-segment override (manifest field `passthrough`).
# True plays the original audio for that span, False dubs it, absent decides
# automatically. It rides the existing keep machinery: a passthrough segment is a
# keep, so tts slices the original audio for it, the timeline reserves its exact
# span, and mix ducks the bed away under it instead of laying a dub over it.
# Its `keep_reason` is `PASSTHROUGH_REASON`, imported above because the translate
# stage subtitles that reason differently and must call it the same thing.
# Carrying an override to a rebuilt segment: the new segment must be this much
# covered by the old one, and vice versa, or they are not the same moment.
CARRY_MIN_OVERLAP = 0.5
# Advisory language stamp: the classifier run under a segment must cover this much
# of it before its label is worth showing the user. A clear majority, not half —
# a segment split evenly across a language change has no one answer, and offering
# the user either label there is offering the wrong one half the time.
DETECT_MIN_COVER = 0.6


def mark_keep(segments: list[dict[str, Any]], spans: list[dict[str, Any]] | None = None,
              target: str = "en", src: str = "he", dub_foreign: bool = False,
              genre: str = "documentary", seg_lang=None) -> None:
    """Flag segments whose original audio should play instead of a dub.

    Three content-free rules: the segment came out of a detected non-source-language
    span, its text is already in the target script, or its speaker is predominantly a
    target-language speaker and nothing about this one segment says otherwise (the
    speaker prior catches the lines the per-segment script test cannot judge — text
    with no script majority either way, and every segment of a same-script pair).

    The span rule has to be structural. A span in a language no ASR here reads carries
    no text at all, and judging it by script would file it as transcript noise and drop
    it — airing nothing where somebody is speaking.

    When source and target share a script, the two script rules are meaningless and
    stay off; only the span rule (fed by the LID path upstream) marks keeps then.

    `dub_foreign` (opt-in) sends a *confident* third-language span segment down the
    dub path instead: its language is known (not "und") and its transcription is real
    (not the "…" placeholder), so the translate stage can render it in the target and
    TTS can voice it from the segment's own audio. Anything less confident keeps its
    original audio exactly as before — never silent.

    The speaker rule is a PRIOR, not a verdict. It is measured over a whole speaker,
    so it also keeps the genuine source-language lines of a mostly-target speaker,
    which then never get dubbed at all ("why isn't this dubbed?"). Two per-segment
    witnesses can outvote it, and either one is enough:

    - the segment's own TEXT is dominantly in the source script (cross-script pairs
      only). A speaker who alternates languages transcribes as source script exactly
      where they spoke the source language, and that is the one witness always
      present — it needs no model and does not depend on clip length.
    - `seg_lang(seg)` returns the language classifier's verdict for that one segment
      as `(lang, prob)` (or None when it has none) and it names the source language
      confidently. The bar is deliberately high — the classifier's documented
      mislabels sit in the 0.34-0.60 band — so an unsure verdict changes nothing.

    The text witness is what the audio one cannot be: short source-language clips
    routinely come back from the classifier as None or as a low-probability wrong
    label, so on its own it left whole untranslated passages kept as `speaker_en`.
    """
    def letters(text: str) -> int:
        return sum(1 for ch in (text or "") if ch.isalpha())

    cross = not script.same_script(src, target)
    # Test-locked legacy value: the target bucket is called "latin" for Latin-script
    # targets; any other script names itself honestly.
    target_reason = "latin" if script.script_for(target) == "latin" else "target_lang"

    ranges = [(s["start"], s["end"], s.get("lang")) for s in (spans or []) if s.get("words")]

    def span_lang(seg: dict[str, Any]) -> str | None:
        """The language of the span this segment came out of, or None if it did not."""
        for a, b, lang in ranges:
            if a - 0.05 <= seg["start"] and seg["end"] <= b + 0.05:
                return lang or ""
        return None
    totals: dict[str, list[float]] = {}
    for seg in segments:
        if letters(seg["text"]) < 2:
            continue
        dur = seg["end"] - seg["start"]
        agg = totals.setdefault(seg["speaker"], [0.0, 0.0])
        agg[0] += dur
        if cross and script.is_script(seg["text"], target):
            agg[1] += dur
    en_speakers = {
        spk for spk, (total, lat) in totals.items() if total > 0 and lat / total >= SPEAKER_EN_RATIO
    }

    def says_source(seg: dict[str, Any]) -> bool:
        """Per-segment evidence strong enough to outvote the speaker-level prior.

        Two independent witnesses, OR'd. The TEXT witness: the segment's own
        transcript is written dominantly in the SOURCE script, which for a
        cross-script pair says this line was not spoken in the target language.
        The AUDIO witness: the language classifier confidently names the source
        language over that span. Either alone overturns the prior — the audio
        witness goes missing or guesses nonsense on short clips, which is how
        whole passages of genuine source speech used to ride a speaker's prior
        through the pipeline undubbed.

        "Dominantly" is the same majority-of-letters test the rest of this
        function measures script with, so a mostly-target line with one
        source-script word embedded in it is not evidence and does not flip.
        """
        # Text witness. Void for a same-script pair, where the two languages are
        # written identically and script says nothing about which one was spoken.
        if cross and script.is_script(seg["text"], src):
            return True
        if seg_lang is None:
            return False
        verdict = seg_lang(seg)
        if not verdict:
            return False
        lang, prob = verdict
        return lang == src and float(prob) >= SPEAKER_EN_VETO_PROB

    for seg in segments:
        lang = span_lang(seg)
        if lang is not None:
            # Not the source language, whatever it is: play it as it was recorded.
            # A witness that NAMED a language cleared its confidence threshold and
            # outranks the script: English inside a he→de run is Latin-script but
            # it is NOT German, and letting script overrule the name kept a whole
            # video of English marked "already the target" in a German dub. The
            # script clause keeps its say only when the witness is nameless — a
            # caption span with no language label, where target-script text is
            # still the best evidence there is.
            named = lang not in ("", "und")
            in_target = (lang == target
                         or (not named and cross and script.is_script(seg["text"], target)))
            if (dub_foreign and not in_target and lang and lang != "und"
                    and (seg.get("text") or "").strip() not in ("", "…")):
                # Opted in, and the span is confident: known language, real words.
                # This one is dubbable — translate reads seg["lang"] as its source.
                seg["keep"], seg["keep_reason"] = False, None
                continue
            seg["keep"] = True
            seg["keep_reason"] = target_reason if in_target else "foreign"
        elif letters(seg["text"]) < 2:
            # Transcript noise (stray glyphs). Nothing to translate, so let the
            # original audio through rather than leaving a hole.
            seg["keep"], seg["keep_reason"] = True, "no_text"
        elif cross and script.is_script(seg["text"], target):
            seg["keep"], seg["keep_reason"] = True, target_reason
        elif seg["speaker"] in en_speakers and not says_source(seg):
            seg["keep"], seg["keep_reason"] = True, "speaker_en"
        elif genre == "movie" and is_interjection_keep(seg, src, target):
            # Movie mode: a standalone greeting/interjection beat plays in the
            # actor's real voice — better than any clone for a one-word line.
            # The subtitle still shows the translation (see translate.run).
            seg["keep"], seg["keep_reason"] = True, "interjection"
        else:
            seg["keep"], seg["keep_reason"] = False, None


def apply_passthrough(segments: list[dict[str, Any]]) -> list[int]:
    """Honour the per-segment `passthrough` override; returns the ids it flipped.

    The override is the user's word, so it is applied on every run — after the
    automatic rules in `mark_keep`, and again before any downstream stage, since
    the app writes it into a finished manifest and expects the next run to obey it.

    `True` makes the segment a keep, so its original audio plays. `False` sends it
    down the dub path. Absent (the default) leaves the automatic verdict alone.
    An override that merely agrees with the automatic verdict changes nothing —
    in particular a span already kept for a *named* reason stays named, because
    "foreign" and "interjection" tell the translate stage to render a subtitle
    and overwriting them with "user" would silently drop it.

    Whatever the flip invalidates goes with it: the translation, the clip and the
    placement of a flipped segment were made for the other path, and leaving them
    behind would dub a passthrough span or play original audio over a dub's slot.
    This is what makes the function safe to call every run — it is idempotent, and
    only a real change of verdict throws work away.
    """
    flipped: list[int] = []
    for seg in segments:
        want = seg.get("passthrough")
        if want is None or bool(want) == bool(seg.get("keep")):
            continue
        if not want and not (seg.get("text") or "").strip():
            # Nothing to translate and nothing to speak: a "dub this" override on a
            # span with no words would strip the original audio and put nothing in
            # its place. The keep stands — never silent outranks the override.
            continue
        if want:
            seg["keep"], seg["keep_reason"] = True, PASSTHROUGH_REASON
        else:
            seg["keep"], seg["keep_reason"] = False, None
        # A hand-corrected line survives the flip: as a dub it is what gets
        # spoken, as a keep it is the subtitle — either way it is the user's
        # word, and this function exists to honour exactly that.
        from . import manifest as manifest_mod

        drop = (("text_mid", "tts", "place") if manifest_mod.is_locked(seg, "text_en")
                else ("text_en", "text_mid", "tts", "place"))
        for field in drop:
            seg.pop(field, None)
        # The clip is the wrong KIND now (a synthesis under a keep, or a slice of
        # the original under a dub), so a `locked.tts` on it is answered rather
        # than left standing over a record that no longer exists — exactly what
        # `edit.set_keep` does at the studio's own door. Leaving the lock behind
        # would tell every later run the user approved a clip that is gone.
        locked = seg.get("locked") or {}
        locked.pop("tts", None)
        if locked:
            seg["locked"] = locked
        else:
            seg.pop("locked", None)
        flipped.append(seg["id"])
    return flipped


def user_wants_dub(seg: dict[str, Any]) -> bool:
    """True when the user has said "dub this line", in either of its two forms.

    One verdict, written twice: `edit.set_keep(keep=False)` stamps `passthrough`
    False *and* locks `keep`, so a headless re-run honours it (`apply_passthrough`)
    and no stage rerun regenerates it (`manifest.reset_stage`). Either form alone
    is still the user speaking, so either one counts here.

    The reason this needs a name: a stage that fails to dub the line answers with
    a keep (`mt_failed`, `tts_failed`), and a keep written over this verdict does
    not merely overrule the user — it contradicts the manifest. The next run's
    `apply_passthrough` reads `passthrough=False` beside `keep=True`, flips it
    back, drops the translation, the clip and the placement, and the whole tail of
    the run is redone to reach the same failure again, forever.
    """
    from . import manifest

    if seg.get("passthrough") is False:
        return True
    return manifest.is_locked(seg, "keep") and not seg.get("keep")


def carry_passthrough(segments: list[dict[str, Any]],
                      overrides: list[tuple[float, float, bool]]) -> int:
    """Re-attach saved overrides to freshly rebuilt segments; returns how many stuck.

    Re-running the segments stage throws every segment away and renumbers what
    replaces it, so an override cannot be carried by id. It is carried by *time*:
    the new segment covering the same moment inherits it. Both directions must
    agree — the new segment mostly inside the old span and the old span mostly
    inside the new one — so a re-segmentation that merges four lines into one
    does not silently spread one line's override across all four.
    """
    stuck = 0
    for seg in segments:
        span = seg["end"] - seg["start"]
        if span <= 0:
            continue
        for a, b, want in overrides:
            if b - a <= 0:
                continue
            overlap = min(seg["end"], b) - max(seg["start"], a)
            if overlap / span >= CARRY_MIN_OVERLAP and overlap / (b - a) >= CARRY_MIN_OVERLAP:
                seg["passthrough"] = bool(want)
                stuck += 1
                break
    return stuck


def saved_overrides(segments: list[dict[str, Any]]) -> list[tuple[float, float, bool]]:
    """The overrides on these segments, as (start, end, passthrough) triples."""
    return [(float(s["start"]), float(s["end"]), bool(s["passthrough"]))
            for s in segments if s.get("passthrough") is not None]


def stamp_detected_lang(segments: list[dict[str, Any]],
                        lang_runs: list[dict[str, Any]] | None) -> None:
    """Record what the language classifier heard under each segment, advisory only.

    Nothing in the pipeline reads this. It exists so the editor app can tell the
    user "this line sounds like it is already English" and offer passthrough,
    instead of the user having to listen to a forty-minute video to find the two
    places where an interviewee switched language. Kept advisory on purpose: the
    classifier is confident enough to suggest and not confident enough to decide.

    A label is only stamped when the run it comes from covers most of the segment
    (`DETECT_MIN_COVER`); a segment straddling a language change has no one
    answer, and a half-covered label would suggest the wrong thing. A stamp that
    is already there — from the span the segment was built out of, which knows
    better — is never overwritten.
    """
    runs = [(float(r["start"]), float(r["end"]), r.get("lang") or "")
            for r in (lang_runs or []) if (r.get("lang") or "")]
    if not runs:
        return
    for seg in segments:
        if seg.get("detected_lang"):
            continue
        span = seg["end"] - seg["start"]
        if span <= 0:
            continue
        best, best_cover = "", 0.0
        for a, b, lang in runs:
            cover = max(0.0, min(seg["end"], b) - max(seg["start"], a)) / span
            if cover > best_cover:
                best, best_cover = lang, cover
        if best_cover >= DETECT_MIN_COVER:
            seg["detected_lang"] = best
