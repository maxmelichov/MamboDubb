"""The editing foundation the studio app is built on (pure logic, no models).

Three things are pinned here, because each one is what makes a human correction
survive contact with the pipeline:

* `uid` stable identity. `id` is positional and renumbered on every
  re-segmentation, so a UI keyed on it would follow the wrong audio.
* `locked` a hand-edited field is never regenerated. A whole-run re-translate,
  the run-global revision pass, a tts rerun, and the timeline's shortening round
  all have to leave it alone.
* `--force <stage>` really invalidates downstream, as its docstring has always
  claimed.

The model-driven halves of `dubbing.edit` (retranslate / resynthesize / rebuild)
are exercised for their bookkeeping only; their generation is the pipeline's own,
tested where it lives.
"""

from __future__ import annotations

import pytest

from dubbing import STAGES, cli, edit, manifest, segments, timeline, translate
from dubbing import keep as keep_mod
from dubbing import tts as tts_mod


def seg(uid=None, **kw):
    s = {"id": kw.pop("id", 0), "start": 0.0, "end": 2.0, "speaker": "S0",
         "text": "aa bb cc dd", "keep": False}
    s.update(kw)
    if uid:
        s["uid"] = uid
    return s


def mk(*segs, **source):
    m = manifest.new({"input": "x", "src_lang": "he", "tgt_lang": "en", **source})
    m["segments"] = [dict(s) for s in segs]
    for i, s in enumerate(m["segments"]):
        s["id"] = i
    manifest.ensure_uids(m["segments"])
    return m


class StubEngine:
    """A synthesiser that loads nothing. `REF_BUILD` is the real one's: it is in
    the tts stage's fingerprint, so a stub without it changes what the CLI would
    compute."""

    REF_BUILD = tts_mod.Engine.REF_BUILD

    def __init__(self, *args, **kw):
        pass

    def close(self):
        pass


def two_segs():
    """Two adjacent same-speaker segments, both fully generated."""
    return mk(
        seg(start=0.0, end=2.0, text="aa bb", text_en="one two",
            tts={"clip": "clips/a.wav", "dur": 1.8}, place={"start": 0.0, "clip": "clips/a.wav"}),
        seg(start=2.0, end=4.0, text="cc dd", text_en="three four",
            tts={"clip": "clips/b.wav", "dur": 1.9}, place={"start": 2.0, "clip": "clips/b.wav"}),
    )


# --------------------------------------------------------------------- uid

def test_uid_is_stable_where_id_is_not():
    m = two_segs()
    uids = [s["uid"] for s in m["segments"]]
    # Re-segmentation renumbers ids from scratch; a UI keyed on id follows the
    # wrong segment, one keyed on uid does not.
    for s in m["segments"]:
        s["id"] += 100
    assert [s["uid"] for s in m["segments"]] == uids
    assert edit.find(m, uids[1])["text"] == "cc dd"


def test_uid_survives_every_edit():
    m = two_segs()
    uid = m["segments"][0]["uid"]
    edit.set_text(m, uid, text_en="rewritten line")
    edit.set_speaker(m, uid, "S9")
    edit.set_keep(m, uid, True)
    edit.set_langs(m, uid, tgt_lang="ru")
    edit.set_bounds(m, uid, 0.5, 1.9)
    assert m["segments"][0]["uid"] == uid


def test_uid_is_minted_from_content_so_a_rerun_reproduces_it():
    # The migration property: a run dir written before uid existed gets the same
    # uid a fresh re-segmentation of the same words would mint.
    fresh = manifest.mint_uid(1.0, 2.0, "hello there")
    assert manifest.mint_uid(1.0, 2.0, "hello there") == fresh
    assert manifest.mint_uid(1.0, 2.5, "hello there") != fresh
    assert manifest.mint_uid(1.0, 2.0, "hello here") != fresh
    assert fresh.startswith("s") and len(fresh) == 12


def test_load_backfills_uid_for_a_manifest_written_without_it(tmp_path):
    import json

    old = {"version": 1, "source": {}, "files": {}, "stages": {}, "progress": {},
           "speakers": {}, "outputs": {},
           "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "hi"},
                        {"id": 1, "start": 1.0, "end": 2.0, "text": "there"}]}
    (tmp_path / "manifest.json").write_text(json.dumps(old), encoding="utf-8")
    m = manifest.load(tmp_path)
    uids = [s["uid"] for s in m["segments"]]
    assert all(uids) and len(set(uids)) == 2
    assert uids[0] == manifest.mint_uid(0.0, 1.0, "hi")


def test_ensure_uids_is_idempotent_and_remints_copies():
    segs = [{"start": 0.0, "end": 1.0, "text": "hi"}]
    manifest.ensure_uids(segs)
    first = segs[0]["uid"]
    manifest.ensure_uids(segs)
    assert segs[0]["uid"] == first
    # A trimmed piece is made by copying the segment dict (segments.splice_foreign_spans),
    # which would otherwise hand two segments one identity.
    segs.append(dict(segs[0], start=1.0, end=2.0))
    manifest.ensure_uids(segs)
    assert segs[1]["uid"] != first


def test_uid_is_whitelisted_and_survives_save(tmp_path):
    m = two_segs()
    uid = m["segments"][0]["uid"]
    edit.set_langs(m, uid, src_lang="ar", tgt_lang="ru")
    edit.set_tts_opts(m, uid, seed=7)
    edit.set_text(m, uid, text_en="hand written")
    manifest.save(tmp_path, m)
    back = manifest.load(tmp_path)["segments"][0]
    assert back["uid"] == uid
    assert (back["src_lang"], back["tgt_lang"]) == ("ar", "ru")
    assert back["tts_opts"] == {"seed": 7}
    assert back["locked"] == {"text_en": True}


def test_new_keys_are_whitelisted():
    assert {"uid", "src_lang", "tgt_lang", "tts_opts", "locked"} <= manifest.SEGMENT_KEYS


# --------------------------------------------------------------------- locked

def test_reset_stage_keeps_locked_fields():
    m = mk(seg(text_en="mine", tts={"clip": "a"}, place={"start": 0.0},
               locked={"text_en": True, "tts": True}),
           seg(id=1, start=2.0, end=4.0, text_en="theirs", tts={"clip": "b"},
               place={"start": 2.0}))
    manifest.reset_stage(m, "translate")
    manifest.reset_stage(m, "tts")
    assert m["segments"][0]["text_en"] == "mine"       # the user's line stands
    assert m["segments"][0]["tts"] == {"clip": "a"}    # and the clip they approved
    assert "text_en" not in m["segments"][1]           # everything else resets
    assert "tts" not in m["segments"][1]


def test_reset_stage_keeps_a_manual_keep():
    m = mk(seg(keep=True, keep_reason="tts_failed", locked={"keep": True, "tts": True},
               tts={"clip": "a"}))
    manifest.reset_stage(m, "tts")
    assert m["segments"][0]["keep"] and m["segments"][0]["keep_reason"] == "tts_failed"


def test_translate_skips_a_locked_line():
    locked = seg(text_en="my own words", locked={"text_en": True})
    assert not translate.needs_translation(locked)
    # ... even when the translation was invalidated under it.
    del locked["text_en"]
    assert not translate.needs_translation(locked)
    assert translate.needs_translation(seg(text_en=""))
    assert not translate.needs_translation(seg(text_en="machine words"))


def test_translate_subtitle_pass_skips_a_locked_keep():
    s = seg(keep=True, keep_reason="foreign", lang="ar", text="مرحبا",
            text_en="hand-written subtitle", locked={"text_en": True})
    assert not translate.needs_subtitle_translation(s)


def test_translate_revision_pass_excludes_locked_lines():
    # The run-global revision pass rewrites every dubbed line whenever anything was
    # translated the exact path that would undo a user's corrections.
    m = mk(seg(text_en="mine", locked={"text_en": True}),
           seg(id=1, start=2.0, end=4.0, text_en="theirs"))
    rev = [s for s in m["segments"]
           if not s.get("keep") and (s.get("text_en") or "").strip()
           and not manifest.is_locked(s, "text_en")]
    assert [s["text_en"] for s in rev] == ["theirs"]


def test_tts_skips_a_locked_clip(tmp_path):
    (tmp_path / "clips").mkdir()
    (tmp_path / "clips" / "a.wav").write_bytes(b"")
    (tmp_path / "clips" / "gone.wav").write_bytes(b"")
    segs = [
        seg(text_en="x", tts={"clip": "clips/a.wav"}, locked={"tts": True}),
        seg(id=1, text_en="x", tts={"clip": "clips/missing.wav"}, locked={"tts": True}),
        seg(id=2, text_en="x"),
        seg(id=3, text_en="x", tts={"clip": "clips/gone.wav"}),
        seg(id=4, text_en="x", keep=True),
    ]
    todo = tts_mod.pending(segs, tmp_path)
    # A locked clip is still on disk (nothing but the user's own edit deletes one),
    # so it is already "done" and never re-synthesized. A lock whose clip is gone is
    # unhonorable and never-silent wins. A keep is not this list's business.
    assert [s["id"] for s in todo] == [1, 2]


def test_tts_does_not_reopen_a_locked_failed_keep():
    segs = [seg(keep=True, keep_reason="tts_failed", tts={"clip": "k"},
                locked={"tts": True}),
            seg(id=1, keep=True, keep_reason="tts_failed", tts={"clip": "k2"})]
    cleared = tts_mod.clear_failed_keeps(segs)
    assert cleared == [1]
    assert segs[0]["keep"] and segs[0]["tts"] == {"clip": "k"}
    assert not segs[1]["keep"] and "tts" not in segs[1]


def test_timeline_does_not_shorten_a_locked_line(monkeypatch):
    # A dub far too long for its slot: the shortening round would re-translate and
    # re-voice it. Locked, it must drift instead and still place legally.
    m = mk(seg(start=0.0, end=2.0, text_en="one two three four five six seven eight",
               tts={"clip": "clips/a.wav", "dur": 9.0}, locked={"text_en": True}),
           seg(id=1, start=2.0, end=4.0, text_en="nine ten",
               tts={"clip": "clips/b.wav", "dur": 1.0}))
    m["segments"][0]["keep"] = False
    calls = []

    def shorten_many(requests):
        calls.append([s["id"] for s, _ in requests])
        return {s["id"]: "short" for s, _ in requests}

    monkeypatch.setattr(timeline.audio, "atempo", lambda *a, **k: None)
    monkeypatch.setattr(timeline.audio, "duration", lambda p: 1.0)
    timeline.run(m, tmp_workdir(), shorten_many=shorten_many,
                 resynth_many=lambda items: {})
    # Unlocked, this exact setup asks to shorten segment 0 (calls == [[0]]).
    assert calls == []
    # It drifted instead of being rewritten, and placement still holds.
    assert m["segments"][0]["text_en"].endswith("eight")
    assert m["segments"][0]["place"]["end"] > m["segments"][1]["place"]["start"] - 100


def tmp_workdir(clips=()):
    """A scratch run dir. `clips` are created as real 1s wav files, because
    re-placement measures every clip it places."""
    import tempfile
    from pathlib import Path

    import numpy as np
    import soundfile as sf

    root = Path(tempfile.mkdtemp())
    if clips:
        (root / "clips").mkdir(parents=True, exist_ok=True)
        for name in clips:
            sf.write(str(root / name), np.zeros(44100, dtype="float32"), 44100)
    return root


# --------------------------------------------------------------- edit: lookup

def test_find_returns_none_for_an_unknown_uid():
    m = two_segs()
    assert edit.find(m, "nope") is None
    with pytest.raises(edit.SegmentNotFound):
        edit.set_speaker(m, "nope", "S1")


# --------------------------------------------------------------- edit: invalidate

def test_invalidate_drops_downstream_too():
    m = two_segs()
    uid = m["segments"][0]["uid"]
    edit.invalidate(m, uid, stages={"translate"})
    s = m["segments"][0]
    # A new line means a new clip means a new placement none of it can be kept.
    assert "text_en" not in s and "tts" not in s and "place" not in s
    assert m["segments"][1]["text_en"] == "three four"      # neighbours untouched


def test_invalidate_respects_locks():
    m = two_segs()
    uid = m["segments"][0]["uid"]
    m["segments"][0]["locked"] = {"text_en": True}
    edit.invalidate(m, uid, stages={"translate"})
    assert m["segments"][0]["text_en"] == "one two"
    assert "tts" not in m["segments"][0]                    # unlocked results still go


def test_invalidate_undoes_the_pipelines_own_keep_verdict():
    m = mk(seg(keep=True, keep_reason="tts_failed", text_en="x", tts={"clip": "a"}))
    edit.invalidate(m, m["segments"][0]["uid"], stages={"tts"})
    assert not m["segments"][0]["keep"]
    assert m["segments"][0]["keep_reason"] is None


def test_invalidate_rejects_an_unknown_stage():
    m = two_segs()
    with pytest.raises(edit.EditError):
        edit.invalidate(m, m["segments"][0]["uid"], stages={"mix"})


# ------------------------------------------------- edit: the run's stage marks

def finished(m):
    """A run where every stage is marked done, as a finished one's manifest is."""
    m["stages"] = {s: {"fp": f"fp-{s}"} for s in STAGES}
    m["progress"] = {s: f"fp-{s}" for s in STAGES}
    return m


def test_invalidate_reopens_the_stages_whose_work_it_deleted():
    m = finished(two_segs())
    edit.invalidate(m, m["segments"][0]["uid"], stages={"translate"})
    # Deleting a translation, a clip and a placement while the run still says
    # "translate: done" is how an edited segment ends up silent: the next run
    # skips all nine stages, nothing refills the hole, and report.json goes on
    # claiming every second is covered.
    assert set(m["stages"]) == {"fetch", "stems", "transcript", "segments"}
    # …but the *progress* marks stay, so the reopened stages resume and refill
    # only what was deleted, instead of restarting and rebuilding the whole run.
    assert set(m["progress"]) == set(STAGES)
    assert m["segments"][1]["text_en"] == "three four"


def test_invalidate_reopens_no_further_back_than_the_stage_it_touched():
    m = finished(two_segs())
    edit.invalidate(m, m["segments"][0]["uid"], stages={"timeline"})
    assert set(m["stages"]) == set(STAGES) - {"timeline", "mix", "report"}


@pytest.mark.parametrize("apply,gone", [
    (lambda m, uid: edit.set_text(m, uid, text_en="a better line"),
     {"translate", "tts", "timeline", "mix", "report"}),
    (lambda m, uid: edit.set_text(m, uid, text="שורה אחרת"),
     {"translate", "tts", "timeline", "mix", "report"}),
    (lambda m, uid: edit.set_keep(m, uid, True),
     {"translate", "tts", "timeline", "mix", "report"}),
    (lambda m, uid: edit.set_langs(m, uid, tgt_lang="ru"),
     {"translate", "tts", "timeline", "mix", "report"}),
    (lambda m, uid: edit.set_speaker(m, uid, "S9"), {"tts", "timeline", "mix", "report"}),
    (lambda m, uid: edit.set_bounds(m, uid, 0.2, 1.9), {"tts", "timeline", "mix", "report"}),
    (lambda m, uid: edit.set_tts_opts(m, uid, seed=7), {"tts", "timeline", "mix", "report"}),
])
def test_every_no_model_edit_reopens_what_it_invalidated(apply, gone):
    m = finished(two_segs())
    apply(m, m["segments"][0]["uid"])
    assert set(STAGES) - set(m["stages"]) == gone


def test_a_split_reopens_from_translate_and_never_the_segmentation():
    m = finished(two_segs())
    edit.split(m, m["segments"][0]["uid"], 1.0)
    # `segments` stays done on purpose: re-running it rebuilds the list from the
    # words and would undo the very cut the user just made.
    assert set(m["stages"]) == {"fetch", "stems", "transcript", "segments"}


def test_a_merge_reopens_from_translate():
    m = finished(two_segs())
    a, b = (s["uid"] for s in m["segments"])
    edit.merge(m, a, b)
    assert set(m["stages"]) == {"fetch", "stems", "transcript", "segments"}


def test_a_headless_rerun_after_an_edit_is_not_up_to_date(tmp_path):
    """The whole point, end to end: the studio hollows out a segment, and the CLI
    that runs next has to notice. Every stage output the run declares is on disk,
    so before this fix `stage_done` answered True nine times and the run printed
    "up to date" for a manifest with a hole in it."""
    workdir = tmp_workdir(["clips/a.wav", "clips/b.wav"])
    for out in ("source.wav", "words.json", "dub.wav", "preview.mp4", "report.json",
                "stems/vocals.wav", "stems/background.wav"):
        (workdir / out).parent.mkdir(parents=True, exist_ok=True)
        (workdir / out).write_text("x")

    args = cli.parse_args([str(workdir / "in.mp4"), "-o", str(workdir)])
    m = two_segs()
    params = cli.stage_params(args, m)
    for stage in STAGES:                       # pretend a full run just finished
        manifest.mark_stage(m, stage, manifest.stage_fingerprint(m, stage, params[stage]))
        m["progress"][stage] = m["stages"][stage]["fp"]

    def verdicts():
        return {s: manifest.stage_done(m, workdir, s,
                                       manifest.stage_fingerprint(m, s, params[s]),
                                       cli.STAGE_OUTPUTS[s]) for s in STAGES}

    assert all(verdicts().values())
    edit.set_text(m, m["segments"][1]["uid"], text_en="What on earth is it")
    assert [s for s, done in verdicts().items() if not done] == \
        ["translate", "tts", "timeline", "mix", "report"]


# --------------------------------------------------------------- edit: setters

def test_set_text_en_locks_and_invalidates_the_clip():
    m = two_segs()
    uid = m["segments"][0]["uid"]
    edit.set_text(m, uid, text_en="  a better line  ")
    s = m["segments"][0]
    assert s["text_en"] == "a better line"
    assert manifest.is_locked(s, "text_en")
    assert "tts" not in s and "place" not in s


def test_set_text_en_replaces_even_a_previously_locked_line():
    m = two_segs()
    uid = m["segments"][0]["uid"]
    edit.set_text(m, uid, text_en="first go")
    edit.set_text(m, uid, text_en="second go")
    assert m["segments"][0]["text_en"] == "second go"


def test_set_text_en_reopens_the_keep_the_failed_translation_caused():
    # `mt_failed` is the pipeline's own verdict about text that no longer exists:
    # the translator gave up, so the span fell back to its original audio. Writing
    # the translation by hand is the user answering exactly that, and leaving the
    # keep standing means their line is only ever a subtitle never spoken.
    m = mk(seg(keep=True, keep_reason="mt_failed", text_en="aa bb cc dd",
               tts={"clip": "clips/keep_x.wav", "dur": 2.0}))
    uid = m["segments"][0]["uid"]
    edit.set_text(m, uid, text_en="hello world")
    s = m["segments"][0]
    assert s["text_en"] == "hello world"
    assert not s["keep"] and s["keep_reason"] is None
    assert "tts" not in s                      # the keep-slice is not what plays now


def test_set_text_en_does_not_reopen_a_keep_the_user_or_the_span_decided():
    # Editing the subtitle of a genuinely foreign span is not a request to dub it,
    # and neither is editing one the user chose to pass through.
    for reason in ("foreign", "user", "manual"):
        m = mk(seg(keep=True, keep_reason=reason, lang="ar", text_en="old sub"))
        edit.set_text(m, m["segments"][0]["uid"], text_en="better sub")
        assert m["segments"][0]["keep"], reason
        assert m["segments"][0]["keep_reason"] == reason


def test_set_source_text_invalidates_the_translation():
    m = two_segs()
    uid = m["segments"][0]["uid"]
    edit.set_text(m, uid, text="ee ff")
    s = m["segments"][0]
    assert s["text"] == "ee ff"
    assert "text_en" not in s and "tts" not in s


def test_set_text_needs_something_and_rejects_empty():
    m = two_segs()
    uid = m["segments"][0]["uid"]
    with pytest.raises(edit.EditError):
        edit.set_text(m, uid)
    with pytest.raises(edit.EditError):
        edit.set_text(m, uid, text_en="   ")


def test_set_text_with_the_same_value_is_a_no_op():
    # Clicking into a line and clicking out again is not an edit: the clip
    # stays, no lock is stamped, nothing queues a resynthesis.
    m = two_segs()
    uid = m["segments"][0]["uid"]
    edit.set_text(m, uid, text_en="  one two  ")   # equals the stored line
    s = m["segments"][0]
    assert s["tts"] == {"clip": "clips/a.wav", "dur": 1.8}
    assert "place" in s
    assert not manifest.is_locked(s, "text_en")
    edit.set_text(m, uid, text="aa bb")            # transcription, same story
    assert "text_en" in s and not manifest.is_locked(s, "text")


def test_set_keep_locks_so_mark_keep_cannot_re_decide_it():
    m = two_segs()
    uid = m["segments"][0]["uid"]
    edit.set_keep(m, uid, True)
    s = m["segments"][0]
    assert s["keep"] and s["keep_reason"] == "manual"
    assert manifest.is_locked(s, "keep")
    assert "tts" not in s and "place" not in s              # the dub is not what plays now
    edit.set_keep(m, uid, False)
    assert not s["keep"] and s["keep_reason"] is None


def test_set_keep_drops_the_line_that_was_written_for_the_other_path():
    # A keep's `text_en` is a SUBTITLE for a foreign or passed-through span it is
    # the honest placeholder, not something anyone should say out loud. Un-keeping
    # while leaving it behind hands that placeholder to the tts stage as the line to
    # speak, and `needs_translation` (text_en non-empty) never refills it. The
    # pipeline's own door for the same flip drops it: segments.apply_passthrough.
    m = mk(seg(keep=True, keep_reason="foreign", lang="und", text_en="…",
               text_mid="…", tts={"clip": "clips/k.wav", "dur": 2.0},
               place={"start": 0.0}))
    uid = m["segments"][0]["uid"]
    edit.set_keep(m, uid, False)
    s = m["segments"][0]
    assert "text_en" not in s and "text_mid" not in s
    assert "tts" not in s and "place" not in s
    assert translate.needs_translation(s)          # the real line is asked for again


def test_set_keep_leaves_a_hand_corrected_line_alone():
    # The user's own text outranks the pipeline in both directions: it is what the
    # segment says when it is dubbed and what it reads when it is kept.
    m = mk(seg(text_en="my own line", locked={"text_en": True}))
    uid = m["segments"][0]["uid"]
    edit.set_keep(m, uid, True)
    assert m["segments"][0]["text_en"] == "my own line"
    edit.set_keep(m, uid, False)
    assert m["segments"][0]["text_en"] == "my own line"


def test_set_speaker_drops_the_clip_because_the_voice_changed():
    m = two_segs()
    uid = m["segments"][0]["uid"]
    edit.set_speaker(m, uid, "S3")
    assert m["segments"][0]["speaker"] == "S3"
    assert "tts" not in m["segments"][0]
    assert m["segments"][0]["text_en"] == "one two"          # the line is still good


def test_set_bounds_asserts_no_overlap():
    m = two_segs()
    a, b = (s["uid"] for s in m["segments"])
    with pytest.raises(edit.EditError):
        edit.set_bounds(m, b, 1.5, 4.0)          # into its predecessor
    with pytest.raises(edit.EditError):
        edit.set_bounds(m, a, 0.0, 2.5)          # into its successor
    with pytest.raises(edit.EditError):
        edit.set_bounds(m, a, 1.0, 1.0)          # empty
    with pytest.raises(edit.EditError):
        edit.set_bounds(m, a, -1.0, 1.0)
    edit.set_bounds(m, a, 0.25, 1.75)
    assert (m["segments"][0]["start"], m["segments"][0]["end"]) == (0.25, 1.75)
    assert "tts" not in m["segments"][0]         # a keep clip is cut from the span


def test_set_langs_overrides_and_clears():
    m = two_segs()
    uid = m["segments"][0]["uid"]
    edit.set_langs(m, uid, src_lang="AR", tgt_lang="ru")
    s = m["segments"][0]
    assert (s["src_lang"], s["tgt_lang"]) == ("ar", "ru")
    assert "text_en" not in s                    # the old line is the wrong language
    edit.set_langs(m, uid, tgt_lang="")
    assert "tgt_lang" not in s


def test_set_langs_overrides_a_locked_line_because_it_is_now_wrong():
    m = two_segs()
    uid = m["segments"][0]["uid"]
    edit.set_text(m, uid, text_en="my english")
    edit.set_langs(m, uid, tgt_lang="ru")
    assert "text_en" not in m["segments"][0]


def test_set_tts_opts_merges_removes_and_invalidates():
    m = two_segs()
    uid = m["segments"][0]["uid"]
    edit.set_tts_opts(m, uid, seed=3, greedy=True)
    assert m["segments"][0]["tts_opts"] == {"seed": 3, "greedy": True}
    assert "tts" not in m["segments"][0]
    edit.set_tts_opts(m, uid, greedy=None)
    assert m["segments"][0]["tts_opts"] == {"seed": 3}
    edit.set_tts_opts(m, uid, seed=None)
    assert "tts_opts" not in m["segments"][0]


def test_set_tts_opts_refuses_what_the_synthesiser_could_never_honour():
    # `ttsopts` is loud on purpose a silently-ignored option looks exactly like a
    # broken model. Storing one unvalidated only moves the ValueError to the middle
    # of the next run (`Engine._plan`), where it takes the whole stage down instead
    # of failing the edit that caused it.
    m = two_segs()
    uid = m["segments"][0]["uid"]
    for bad in ({"nonsense": 1}, {"speed": 99.0}, {"model": "3b"},
                {"ref_text": "hello"},                       # ref_text needs ref
                {"greedy": True, "temperature": 0.7}):       # inert together
        with pytest.raises(edit.EditError):
            edit.set_tts_opts(m, uid, **bad)
    assert "tts_opts" not in m["segments"][0]
    assert m["segments"][0]["tts"] == {"clip": "clips/a.wav", "dur": 1.8}


def test_set_tts_opts_stores_a_value_that_equals_the_default_as_no_option_at_all():
    # `TtsOpts.to_dict` strips defaults, so `{"speed": 1.0}` fingerprints as "" —
    # a stored option that can never requeue the segment. Normalised away, the
    # record says what it means and the clip that already matches it survives.
    m = two_segs()
    uid = m["segments"][0]["uid"]
    edit.set_tts_opts(m, uid, speed=1.0)
    assert "tts_opts" not in m["segments"][0]
    assert m["segments"][0]["tts"] == {"clip": "clips/a.wav", "dur": 1.8}


# --------------------------------------------------------------- edit: structure

def test_split_preserves_every_word():
    m = two_segs()
    uid = m["segments"][0]["uid"]
    m["segments"][0]["text"] = "one two three four five"
    a, b = edit.split(m, uid, 1.0)
    left, right = m["segments"][0], m["segments"][1]
    assert (left["uid"], right["uid"]) == (a, b)
    assert (left["text"] + " " + right["text"]).split() == "one two three four five".split()
    assert (left["start"], left["end"], right["start"], right["end"]) == (0.0, 1.0, 1.0, 2.0)
    assert [s["id"] for s in m["segments"]] == [0, 1, 2]


def test_split_drops_what_was_made_from_the_whole_line():
    m = two_segs()
    a, b = edit.split(m, m["segments"][0]["uid"], 1.0)
    for s in m["segments"][:2]:
        assert "text_en" not in s and "tts" not in s and "place" not in s
    assert a != b and b != m["segments"][2]["uid"]


def test_split_refuses_what_it_cannot_do_without_losing_words():
    m = mk(seg(text="solo"))
    uid = m["segments"][0]["uid"]
    with pytest.raises(edit.EditError):
        edit.split(m, uid, 1.0)                  # one word, one of the halves is empty
    m2 = two_segs()
    with pytest.raises(edit.EditError):
        edit.split(m2, m2["segments"][0]["uid"], 5.0)     # outside the span
    with pytest.raises(edit.EditError):
        edit.split(m2, m2["segments"][0]["uid"], 0.0)     # on the edge


def test_split_never_makes_an_empty_half_even_at_the_edge():
    m = mk(seg(start=0.0, end=10.0, text="a b c"))
    left_uid, right_uid = edit.split(m, m["segments"][0]["uid"], 0.05)
    assert m["segments"][0]["text"] and m["segments"][1]["text"]


def test_merge_joins_adjacent_same_speaker_segments():
    m = two_segs()
    a, b = (s["uid"] for s in m["segments"])
    uid = edit.merge(m, a, b)
    assert len(m["segments"]) == 1
    s = m["segments"][0]
    assert s["uid"] == uid and uid not in (a, b)
    assert (s["start"], s["end"]) == (0.0, 4.0)
    assert s["text"] == "aa bb cc dd"
    assert "text_en" not in s and "tts" not in s and "place" not in s
    assert s["id"] == 0


def test_merge_order_does_not_matter():
    m = two_segs()
    a, b = (s["uid"] for s in m["segments"])
    assert edit.merge(m, b, a)
    assert len(m["segments"]) == 1


def test_merge_refuses_across_speakers_and_across_a_gap():
    m = mk(seg(start=0.0, end=2.0, text="aa"),
           seg(id=1, start=2.0, end=4.0, text="bb", speaker="S1"),
           seg(id=2, start=4.0, end=6.0, text="cc"))
    a, b, c = (s["uid"] for s in m["segments"])
    with pytest.raises(edit.EditError):
        edit.merge(m, a, b)                      # different speakers
    with pytest.raises(edit.EditError):
        edit.merge(m, a, c)                      # not adjacent
    with pytest.raises(edit.EditError):
        edit.merge(m, a, a)


def test_merge_keeps_only_when_both_halves_were_kept():
    m = mk(seg(start=0.0, end=2.0, text="aa", keep=True, keep_reason="foreign"),
           seg(id=1, start=2.0, end=4.0, text="bb", keep=True, keep_reason="foreign"))
    uid = edit.merge(m, *(s["uid"] for s in m["segments"]))
    assert edit.find(m, uid)["keep"] and edit.find(m, uid)["keep_reason"] == "foreign"

    m2 = mk(seg(start=0.0, end=2.0, text="aa", keep=True, keep_reason="foreign"),
            seg(id=1, start=2.0, end=4.0, text="bb"))
    uid2 = edit.merge(m2, *(s["uid"] for s in m2["segments"]))
    assert not edit.find(m2, uid2)["keep"]


def test_a_split_carries_the_users_verdict_about_the_span():
    # `passthrough` is the user's word about the SPAN, not about the text, and a
    # split leaves every second of that span covered. Dropped, the halves' `keep`
    # survives only until the next `python -m dubbing`: the segments stage rebuilds
    # from scratch, `mark_keep` re-decides both halves, and there is no override
    # left for `carry_passthrough` to re-attach the passage is dubbed again.
    m = mk(seg(start=0.0, end=4.0, text="aa bb cc dd", keep=True,
               keep_reason="user", passthrough=True))
    edit.split(m, m["segments"][0]["uid"], 2.0)
    assert [s.get("passthrough") for s in m["segments"]] == [True, True]
    assert segments.saved_overrides(m["segments"]) == [(0.0, 2.0, True), (2.0, 4.0, True)]


def test_a_merge_carries_an_override_only_when_both_halves_agreed():
    both = mk(seg(start=0.0, end=2.0, text="aa", keep=True, keep_reason="user",
                  passthrough=True),
              seg(id=1, start=2.0, end=4.0, text="bb", keep=True, keep_reason="user",
                  passthrough=True))
    a, b = (s["uid"] for s in both["segments"])
    edit.merge(both, a, b)
    assert both["segments"][0]["passthrough"] is True

    # Disagreement has no answer, so the merged span carries none same rule the
    # language and synthesis overrides already follow.
    split_minds = mk(seg(start=0.0, end=2.0, text="aa", keep=True,
                         keep_reason="user", passthrough=True),
                     seg(id=1, start=2.0, end=4.0, text="bb", passthrough=False))
    a, b = (s["uid"] for s in split_minds["segments"])
    edit.merge(split_minds, a, b)
    assert "passthrough" not in split_minds["segments"][0]


def test_structural_edits_leave_the_list_ordered_and_non_overlapping():
    m = two_segs()
    edit.split(m, m["segments"][1]["uid"], 3.0)
    edit.split(m, m["segments"][0]["uid"], 1.0)
    spans = [(s["start"], s["end"]) for s in m["segments"]]
    assert spans == sorted(spans)
    for (a_start, a_end), (b_start, b_end) in zip(spans, spans[1:]):
        assert a_end <= b_start
    assert [s["id"] for s in m["segments"]] == list(range(len(spans)))
    assert len({s["uid"] for s in m["segments"]}) == len(spans)


# --------------------------------------------------------------- edit: model paths

def test_retranslate_calls_generate_per_segment_and_never_the_revision_pass(monkeypatch):
    m = two_segs()
    # Real source text for the run's own pair: written in Hebrew, so nothing
    # refutes `--src he` and the hop translates from Hebrew. The placeholder
    # "aa bb" is Latin, which `translate.segment_langs` reads as evidence that
    # the run's claim is not true of this particular line.
    m["segments"][0]["text"] = "שלום עולם"
    a, b = (s["uid"] for s in m["segments"])
    edit.set_text(m, b, text_en="my own corrected line")
    seen = []

    monkeypatch.setattr(edit_translate := translate, "load",
                        lambda *a, **k: ("proc", "model", None))
    monkeypatch.setattr(edit_translate, "free", lambda model: None)
    monkeypatch.setattr(edit_translate, "revise_run",
                        lambda *a, **k: pytest.fail("revision pass must not run"))

    def fake_generate(proc, model, text, **kw):
        seen.append((text, kw["source"], kw["target"]))
        return "fresh line"

    monkeypatch.setattr(edit_translate, "generate", fake_generate)
    out = edit.retranslate(m, tmp_workdir(), [a])
    assert out == {a: "fresh line"}
    assert m["segments"][0]["text_en"] == "fresh line"
    assert [s for _, s, _ in seen] == ["he"]          # one hop: he→en is not a pivot
    # The other segment's correction is untouched the whole point.
    assert m["segments"][1]["text_en"] == "my own corrected line"


def test_retranslate_honours_the_per_segment_language_override(monkeypatch):
    m = two_segs()
    uid = m["segments"][0]["uid"]
    edit.set_langs(m, uid, src_lang="ar", tgt_lang="ru")
    hops = []
    monkeypatch.setattr(translate, "load", lambda *a, **k: ("p", "m", None))
    monkeypatch.setattr(translate, "free", lambda model: None)
    monkeypatch.setattr(translate, "generate",
                        lambda p, mo, text, **kw: (hops.append((kw["source"], kw["target"]))
                                                   or ("english" if kw["target"] == "en"
                                                       else "русский текст")))
    edit.retranslate(m, tmp_workdir(), [uid])
    assert hops == [("ar", "en"), ("en", "ru")]       # ar→ru pivots through English
    assert m["segments"][0]["text_en"] == "русский текст"
    assert m["segments"][0]["text_mid"] == "english"


def test_retranslate_clears_the_lock_it_replaces_and_drops_the_clip(monkeypatch):
    m = two_segs()
    uid = m["segments"][0]["uid"]
    edit.set_text(m, uid, text_en="mine")
    monkeypatch.setattr(translate, "load", lambda *a, **k: ("p", "m", None))
    monkeypatch.setattr(translate, "free", lambda model: None)
    monkeypatch.setattr(translate, "generate", lambda *a, **k: "machine line")
    edit.retranslate(m, tmp_workdir(), [uid])
    s = m["segments"][0]
    assert s["text_en"] == "machine line"
    assert not manifest.is_locked(s, "text_en")       # explicitly asked for, so replaced
    assert "tts" not in s
    # respect_locked= is the bulk path, and there the lock wins.
    edit.set_text(m, uid, text_en="mine again")
    assert edit.retranslate(m, tmp_workdir(), [uid], respect_locked=True) == {}
    assert s["text_en"] == "mine again"


def test_retranslate_falls_back_to_keep_when_the_model_fails(monkeypatch):
    m = two_segs()
    uid = m["segments"][0]["uid"]
    monkeypatch.setattr(translate, "load", lambda *a, **k: ("p", "m", None))
    monkeypatch.setattr(translate, "free", lambda model: None)
    monkeypatch.setattr(translate, "generate", lambda *a, **k: "")
    edit.retranslate(m, tmp_workdir(), [uid])
    s = m["segments"][0]
    assert s["keep"] and s["keep_reason"] == "mt_failed"
    assert s["text_en"] == s["text"]                  # never silent: it plays original


def test_retranslate_translates_a_foreign_line_from_its_own_language(monkeypatch):
    # A he→de run of a video whose speech is mostly English. `--src` is a claim
    # about the video; this line's own Latin script refutes it, so the prompt
    # asserts no source at all and goes straight to German one hop, not the
    # he→en→de pivot whose first hop can only produce an echo.
    m = mk(seg(start=0.0, end=2.0, text="around this table, after the elections."),
           src_lang="he", tgt_lang="de")
    uid = m["segments"][0]["uid"]
    hops = []
    monkeypatch.setattr(translate, "load", lambda *a, **k: ("p", "m", None))
    monkeypatch.setattr(translate, "free", lambda model: None)
    monkeypatch.setattr(translate, "generate",
                        lambda p, mo, text, **kw: (hops.append((kw["source"], kw["target"]))
                                                   or "Rund um diesen Tisch."))
    out = edit.retranslate(m, tmp_workdir(), [uid])
    assert hops == [("", "de")]
    assert out == {uid: "Rund um diesen Tisch."}
    assert m["segments"][0]["text_en"] == "Rund um diesen Tisch."
    assert m["segments"][0]["keep"] is False


def test_retranslate_leaves_a_locked_dub_it_unfinished_and_says_so(monkeypatch):
    # The user pressed "Dub it", which locks `keep` and writes `passthrough`.
    # When the translation fails, reverting them restores the audio they were
    # replacing so the button appears to do nothing, twice over, since the job
    # still reported a success. Their verdict stands and the message is honest.
    m = mk(seg(start=0.0, end=2.0, text="around this table, after the elections.",
               text_en="stale", keep=True, keep_reason="latin",
               tts={"clip": "clips/a.wav", "dur": 1.8}),
           src_lang="he", tgt_lang="de")
    uid = m["segments"][0]["uid"]
    edit.set_keep(m, uid, False)
    monkeypatch.setattr(translate, "load", lambda *a, **k: ("p", "m", None))
    monkeypatch.setattr(translate, "free", lambda model: None)
    monkeypatch.setattr(translate, "generate", lambda *a, **k: "")
    seen = []
    out = edit.retranslate(m, tmp_workdir(), [uid],
                           progress=lambda f, msg: seen.append((f, msg)))
    s = m["segments"][0]
    assert s["keep"] is False and s["passthrough"] is False   # manifest agrees
    assert not s.get("text_en")                 # the editor's "untranslated" state
    assert out == {}                            # nothing translated, nothing claimed
    assert seen[-1] == (1.0, "translated 0 of 1 segment(s) 1 failed to translate (seg 0)")


def test_retranslate_leaves_a_kept_spans_verdict_alone(monkeypatch):
    # Retranslating a keep improves its SUBTITLE. Un-keeping it would strand the
    # user's `passthrough=True` beside `keep=False`, and the next headless run's
    # `apply_passthrough` would flip it back and delete this translation an
    # 11 GB model load thrown away without a word.
    m = mk(seg(start=0.0, end=2.0, text="גדי איזנקוט", keep=True,
               keep_reason="manual", passthrough=True, locked={"keep": True}))
    uid = m["segments"][0]["uid"]
    monkeypatch.setattr(translate, "load", lambda *a, **k: ("p", "m", None))
    monkeypatch.setattr(translate, "free", lambda model: None)
    monkeypatch.setattr(translate, "generate", lambda *a, **k: "Gadi Eisenkot.")
    edit.retranslate(m, tmp_workdir(), [uid])
    s = m["segments"][0]
    assert s["text_en"] == "Gadi Eisenkot."
    assert s["keep"] is True and s["passthrough"] is True
    assert keep_mod.apply_passthrough(m["segments"]) == []   # nothing to undo


def test_retranslate_still_reopens_the_pipelines_own_failed_keep(monkeypatch):
    # The one keep a fresh translation does answer: `mt_failed` is the
    # translator's own verdict about text that is now gone (`invalidate`).
    m = mk(seg(start=0.0, end=2.0, text="גדי איזנקוט", keep=True,
               keep_reason="mt_failed", text_en="גדי איזנקוט"))
    uid = m["segments"][0]["uid"]
    monkeypatch.setattr(translate, "load", lambda *a, **k: ("p", "m", None))
    monkeypatch.setattr(translate, "free", lambda model: None)
    monkeypatch.setattr(translate, "generate", lambda *a, **k: "Gadi Eisenkot.")
    edit.retranslate(m, tmp_workdir(), [uid])
    s = m["segments"][0]
    assert s["keep"] is False and s["keep_reason"] is None
    assert s["text_en"] == "Gadi Eisenkot."


def test_retranslate_repairs_a_failure_keep_written_over_the_users_dub_it(monkeypatch):
    # The state three real runs are stuck in: keep=True/mt_failed sitting on top
    # of passthrough=False and a locked keep. The lock was guarding a value the
    # PIPELINE wrote, so the undo looks past it at what the user said.
    m = mk(seg(start=0.0, end=2.0, text="In the meeting, the participants tried.",
               keep=True, keep_reason="mt_failed", passthrough=False,
               locked={"keep": True},
               text_en="In the meeting, the participants tried.",
               tts={"clip": "clips/keep_a.wav", "dur": 2.0}),
           src_lang="he", tgt_lang="de")
    uid = m["segments"][0]["uid"]
    monkeypatch.setattr(translate, "load", lambda *a, **k: ("p", "m", None))
    monkeypatch.setattr(translate, "free", lambda model: None)
    monkeypatch.setattr(translate, "generate", lambda *a, **k: "Bei dem Treffen.")
    edit.retranslate(m, tmp_workdir(), [uid])
    s = m["segments"][0]
    assert s["keep"] is False and s["keep_reason"] is None
    assert s["text_en"] == "Bei dem Treffen." and "tts" not in s
    assert keep_mod.apply_passthrough(m["segments"]) == []   # no flip left to make


def test_resynthesize_does_not_voice_a_locked_dub_that_has_no_line(monkeypatch):
    # The second half of the same loop: with nothing to speak, the old code
    # called it a tts failure and handed the segment its original audio back.
    m = mk(seg(start=0.0, end=2.0, text="around this table."))
    uid = m["segments"][0]["uid"]
    edit.set_keep(m, uid, False)

    class FakeEngine:
        def __init__(self, *a, **kw):
            pass

        def build_speaker_refs(self):
            pass

        def clip_for(self, seg, text_en):
            return None                       # nothing to say

        def keep_clip(self, seg):
            pytest.fail("a missing translation is not a synthesis failure")

        def close(self):
            pass

    monkeypatch.setattr(tts_mod, "Engine", FakeEngine)
    seen = []
    out = edit.resynthesize(m, tmp_workdir(), [uid],
                            progress=lambda f, msg: seen.append((f, msg)))
    s = m["segments"][0]
    assert out == {} and "tts" not in s
    assert s["keep"] is False and s["passthrough"] is False
    # This exact shape meant to be dubbed, no translation, no clip is what the
    # editor's `segmentState` reads as "untranslated" and the Unfinished chip counts.
    # Left as a `tts_failed` keep instead (which is what a cancelled translate used
    # to produce) the line would sit in Kept, looking like a decision someone made.
    assert not (s.get("text_en") or "").strip() and s.get("keep_reason") is None
    # The message states what happened (`edit._resynth_message`): nothing was
    # synthesized, one line has no translation to speak, and the placement that
    # could not run over the hole is said out loud rather than left implied.
    assert seen[-1][1] == ("synthesized 0 1 still needs a translation (seg 0); "
                           "placement deferred to the next tts run")


def test_retranslate_reports_progress(monkeypatch):
    m = two_segs()
    monkeypatch.setattr(translate, "load", lambda *a, **k: ("p", "m", None))
    monkeypatch.setattr(translate, "free", lambda model: None)
    monkeypatch.setattr(translate, "generate", lambda *a, **k: "line")
    seen = []
    edit.retranslate(m, tmp_workdir(), [s["uid"] for s in m["segments"]],
                     progress=lambda f, msg: seen.append((f, msg)))
    assert seen[0][0] == 0.0 and seen[-1][0] == 1.0
    assert all(0.0 <= f <= 1.0 for f, _ in seen)
    assert all(isinstance(msg, str) for _, msg in seen)


def test_resynthesize_records_clips_and_falls_back_to_keep(monkeypatch):
    m = two_segs()
    a, b = (s["uid"] for s in m["segments"])
    m["segments"][1]["keep"] = True

    class FakeEngine:
        def __init__(self, *args, **kw):
            self.closed = False

        def build_speaker_refs(self):
            pass

        def clip_for(self, seg, text):
            return None if text == "one two" else {"clip": "clips/new.wav", "dur": 1.0}

        def keep_clip(self, seg):
            return {"clip": "clips/keep.wav", "dur": 2.0, "verify": "keep"}

        def close(self):
            self.closed = True

    monkeypatch.setattr(tts_mod, "Engine", FakeEngine)
    out = edit.resynthesize(m, tmp_workdir(["clips/new.wav", "clips/keep.wav", "clips/a.wav", "clips/b.wav"]), [a, b])
    # Never silent: the failed dub falls back to its original audio, not to nothing.
    assert m["segments"][0]["keep"] and m["segments"][0]["keep_reason"] == "tts_failed"
    assert out[a]["verify"] == "keep"
    assert out[b]["verify"] == "keep"                 # already a keep: fresh slice
    # The old placement was for the old clip; a new one is made for the new clip,
    # because an unplaced segment is missing from the mix entirely.
    assert m["segments"][0]["place"]["clip"] != "clips/old.wav"


def test_resynthesize_overrides_a_tts_lock_when_asked_for_that_segment(monkeypatch):
    m = two_segs()
    uid = m["segments"][0]["uid"]
    m["segments"][0]["locked"] = {"tts": True}

    class FakeEngine:
        def __init__(self, *a, **k):
            pass

        def build_speaker_refs(self):
            pass

        def clip_for(self, seg, text):
            return {"clip": "clips/new.wav", "dur": 1.0}

        def close(self):
            pass

    monkeypatch.setattr(tts_mod, "Engine", FakeEngine)
    edit.resynthesize(m, tmp_workdir(["clips/new.wav", "clips/a.wav", "clips/b.wav"]), [uid])
    assert m["segments"][0]["tts"]["clip"] == "clips/new.wav"
    assert not manifest.is_locked(m["segments"][0], "tts")


def test_the_run_records_its_own_settings_so_the_studio_reproduces_them():
    # `edit._args` rebuilds the CLI arguments from `m["source"]` it is what tells
    # `_replace_timeline` which tempo policy this run uses and what `rebuild` marks
    # its stages with. A setting the run never wrote down falls back to argparse's
    # default, so a --genre movie run got re-placed at documentary rates after any
    # resynthesize, and rebuild stamped fingerprints the CLI would never compute.
    argv = ["clip.mp4", "--genre", "movie", "--register", "dialogue",
            "--tts-model", "1.7b", "--dub-foreign", "--transcript", "asr"]
    args = cli.parse_args(argv)
    # main() resolves before anything reads or records a setting; --src/--tgt/
    # --duration now parse as None so a re-run can tell "typed" from "default".
    cli.resolve_settings(args, None)
    m = manifest.new(cli.source_record(args))
    back = edit._args(m)
    assert (back.genre, back.register, back.tts_model, back.dub_foreign,
            back.transcript) == ("movie", "dialogue", "1.7b", True, "asr")
    # The whole promise of `rebuild`: the same fingerprint the CLI would compute,
    # so a later `python -m dubbing` on this run sees the work as done.
    assert cli.stage_params(back, m) == cli.stage_params(args, m)
    # And the stakes on the timeline: the two genres are different tempo policies.
    assert timeline.rates_for_genre(back.genre) != timeline.rates_for_genre("documentary")


def test_resynthesize_defers_placement_while_another_segment_awaits_its_clip(monkeypatch):
    # Correcting several lines and then re-voicing one of them is the ordinary
    # studio loop, and it leaves the others without a clip until they are asked for.
    # `timeline.build_items` places the whole run at once and reads `seg["tts"]`
    # unconditionally, so re-placing over that hole died with a bare KeyError —
    # after the new clips had already been written into the manifest.
    m = two_segs()
    a, b = (s["uid"] for s in m["segments"])
    edit.set_text(m, b, text_en="a better line")
    assert "tts" not in m["segments"][1]

    class FakeEngine:
        def __init__(self, *args, **kw):
            pass

        def build_speaker_refs(self):
            pass

        def clip_for(self, seg, text):
            return {"clip": "clips/new.wav", "dur": 1.0}

        def close(self):
            pass

    monkeypatch.setattr(tts_mod, "Engine", FakeEngine)
    out = edit.resynthesize(m, tmp_workdir(["clips/new.wav", "clips/a.wav"]), [a])
    assert out[a]["clip"] == "clips/new.wav"
    # Nothing coherent to place yet the tts run that fills the hole re-places
    # everything but the synthesis the user asked for is recorded, not lost.
    assert "place" not in m["segments"][0] and "place" not in m["segments"][1]


def test_rebuild_refuses_stages_that_need_the_source_media():
    m = two_segs()
    with pytest.raises(edit.EditError):
        edit.rebuild(m, tmp_workdir(), from_stage="segments")
    with pytest.raises(edit.EditError):
        edit.rebuild(m, tmp_workdir(), from_stage="nonsense")


def test_rebuild_runs_forward_and_marks_the_stages_the_cli_would(monkeypatch):
    m = two_segs()
    m["stages"] = {s: {"fp": "old"} for s in ("fetch", "translate", "tts", "timeline",
                                              "mix", "report")}
    ran = []
    monkeypatch.setattr(edit, "_args", edit._args)      # keep the real defaults path
    from dubbing import mix as mix_mod
    from dubbing import report as report_mod

    monkeypatch.setattr(timeline, "run", lambda *a, **k: ran.append("timeline"))
    monkeypatch.setattr(mix_mod, "run", lambda *a, **k: ran.append("mix"))
    monkeypatch.setattr(report_mod, "run", lambda *a, **k: ran.append("report") or {})
    monkeypatch.setattr(tts_mod, "Engine", StubEngine)
    monkeypatch.setattr(cli, "_retimers", lambda *a, **k: (None, None))
    done = edit.rebuild(m, tmp_workdir(), from_stage="timeline", save=lambda: None)
    assert done == ["timeline", "mix", "report"] == ran
    assert m["stages"]["fetch"]["fp"] == "old"          # upstream untouched
    assert m["stages"]["timeline"]["fp"] != "old"       # and re-marked, not left stale


# --------------------------------------------------- rebuild: look before clearing

@pytest.mark.parametrize("asked,expect", [
    ("timeline", "timeline"), ("mix", "mix"), ("translate", "translate"),
])
def test_start_stage_leaves_a_runnable_rebuild_alone(asked, expect):
    assert edit.start_stage(two_segs(), asked)[0] == expect


def test_start_stage_backs_up_to_tts_for_a_segment_with_no_clip():
    m = two_segs()
    edit.set_text(m, m["segments"][1]["uid"], text_en="a better line")   # drops the clip
    for asked in ("timeline", "mix", "report"):
        stage, why = edit.start_stage(m, asked)
        assert stage == "tts" and "no audio" in why
    # tts is what fills the hole, so a rebuild already starting there is untouched.
    assert edit.start_stage(m, "tts")[0] == "tts"


def test_start_stage_backs_up_to_translate_for_a_line_that_was_never_made():
    m = two_segs()
    m["segments"][1].pop("text_en")
    m["segments"][1].pop("tts")
    assert edit.start_stage(m, "timeline")[0] == "translate"
    # A keep speaks its original audio; it needs no translation to be renderable.
    m["segments"][1].update(keep=True, keep_reason="manual",
                            tts={"clip": "clips/b.wav", "dur": 1.9})
    assert edit.start_stage(m, "timeline")[0] == "timeline"


def test_rebuild_fills_the_hole_instead_of_clearing_and_dying(monkeypatch):
    """The Render button on the state an ordinary text edit leaves behind. It used
    to clear timeline/mix/report first and only then hit `KeyError: 'tts'` inside
    `timeline.build_items` leaving the run strictly worse than it found it."""
    from dubbing import mix as mix_mod
    from dubbing import report as report_mod

    m = finished(two_segs())
    edit.set_text(m, m["segments"][1]["uid"], text_en="a better line")
    ran = []
    monkeypatch.setattr(tts_mod, "run",
                        lambda *a, **k: ran.append("tts") or type(
                            "E", (), {"close": lambda self: None})())
    monkeypatch.setattr(timeline, "run", lambda *a, **k: ran.append("timeline"))
    monkeypatch.setattr(mix_mod, "run", lambda *a, **k: ran.append("mix"))
    monkeypatch.setattr(report_mod, "run", lambda *a, **k: ran.append("report") or {})
    monkeypatch.setattr(cli, "_retimers", lambda *a, **k: (None, None))

    done = edit.rebuild(m, tmp_workdir(), from_stage="timeline", save=lambda: None)
    assert done == ["tts", "timeline", "mix", "report"] == ran
    assert m["segments"][0]["text_en"] == "one two"     # the other line is untouched


def test_rebuild_fails_when_the_report_says_the_dub_has_holes(monkeypatch):
    from dubbing import mix as mix_mod
    from dubbing import report as report_mod

    m = two_segs()
    monkeypatch.setattr(timeline, "run", lambda *a, **k: None)
    monkeypatch.setattr(mix_mod, "run", lambda *a, **k: None)
    monkeypatch.setattr(report_mod, "run", lambda *a, **k: {"unaccounted": [1]})
    monkeypatch.setattr(tts_mod, "Engine", StubEngine)
    monkeypatch.setattr(cli, "_retimers", lambda *a, **k: (None, None))
    with pytest.raises(edit.RebuildIncomplete, match="no audio"):
        edit.rebuild(m, tmp_workdir(), from_stage="timeline", save=lambda: None)
    # It ran to the end first: the work is marked and saved, only the verdict fails.
    assert set(m["stages"]) >= {"timeline", "mix", "report"}


# --------------------------------------------------------------- --force downstream

def test_force_invalidates_every_downstream_stage():
    m = two_segs()
    m["stages"] = {s: {"fp": f"fp-{s}"} for s in ("fetch", "stems", "transcript",
                                                  "segments", "translate", "tts",
                                                  "timeline", "mix", "report")}
    m["progress"] = dict.fromkeys(m["stages"], "x")
    cleared = cli.apply_force(m, "translate")
    assert cleared == ["translate", "tts", "timeline", "mix", "report"]
    assert set(m["stages"]) == {"fetch", "stems", "transcript", "segments"}
    assert set(m["progress"]) == {"fetch", "stems", "transcript", "segments"}
    # Downstream results are gone too, so the reruns actually redo them: without
    # this the empty `outputs` lists made translate/tts/timeline "up to date".
    assert all("tts" not in s and "place" not in s for s in m["segments"])


def test_force_keeps_upstream_results():
    m = two_segs()
    m["stages"] = {"translate": {"fp": "a"}, "tts": {"fp": "b"}, "timeline": {"fp": "c"}}
    cli.apply_force(m, "tts")
    assert m["stages"] == {"translate": {"fp": "a"}}
    assert m["segments"][0]["text_en"] == "one two"      # translate's work stands
    assert "place" not in m["segments"][0]               # the timeline's does not
    # The forced stage's own results are reset by the run loop, not here: dropping
    # its `progress` record is what makes the loop call `reset_stage` for it.
    assert "tts" not in m.get("progress", {})


def test_force_all_and_unknown_stage():
    m = two_segs()
    m["stages"] = {"fetch": {"fp": "a"}}
    assert cli.apply_force(m, "all")
    assert m["stages"] == {} and m["progress"] == {}
    assert cli.apply_force(m, None) == []
    with pytest.raises(SystemExit):
        cli.apply_force(m, "nope")


def test_force_does_not_discard_a_users_locked_work():
    m = two_segs()
    m["segments"][0]["locked"] = {"text_en": True, "tts": True}
    m["stages"] = {"translate": {"fp": "t"}, "tts": {"fp": "u"}, "timeline": {"fp": "v"}}
    cli.apply_force(m, "translate")
    kept, other = m["segments"]
    assert kept["text_en"] == "one two" and kept["tts"] == {"clip": "clips/a.wav", "dur": 1.8}
    assert "tts" not in other       # everything unlocked is invalidated as usual


# ------------------------------------------------------ fingerprint knobs

def params_for(m, argv=("in.mp4",)):
    args = cli.parse_args(list(argv))
    cli.resolve_settings(args, m)
    return cli.stage_params(args, m)


def test_the_tts_fingerprint_carries_the_reference_recipe(monkeypatch):
    # `m["speakers"]` survives a tts reset, so a new canonical-reference recipe
    # would otherwise keep cloning from the old references forever and
    # `needs_synthesis` answers by file existence, so nothing else would notice.
    m = manifest.new({"input": "in.mp4", "src_lang": "he", "tgt_lang": "en"})
    before = params_for(m)["tts"]
    monkeypatch.setattr(tts_mod.Engine, "REF_BUILD", tts_mod.Engine.REF_BUILD + 1)
    assert params_for(m)["tts"] != before


def test_the_tts_fingerprint_carries_the_hebrew_adapter(monkeypatch):
    from dubbing import hebrew

    he = manifest.new({"input": "in.mp4", "src_lang": "en", "tgt_lang": "he"})
    en = manifest.new({"input": "in.mp4", "src_lang": "he", "tgt_lang": "en"})
    tagged = params_for(he, ("in.mp4", "--tgt", "he"))["tts"]
    assert tagged["adapter"] == hebrew.ADAPTER_TAG
    # A run that does not go through the adapter does not carry its tag at all.
    assert "adapter" not in params_for(en)["tts"]
    monkeypatch.setattr(hebrew, "ADAPTER_TAG", "qwentts-he-lora-v2")
    assert params_for(he, ("in.mp4", "--tgt", "he"))["tts"] != tagged


def test_the_transcript_fingerprint_carries_where_the_words_came_from():
    # Captions and ASR produce different words for identical parameters, so the
    # source has to be in the chain: a run that started on the captions fallback
    # and later got its ASR must redo the segments built on the caption text.
    m = manifest.new({"input": "in.mp4", "src_lang": "he", "tgt_lang": "en"})
    m["source"]["transcript_origin"] = "captions"
    fallback = params_for(m)["transcript"]
    m["source"]["transcript_origin"] = "asr"
    assert params_for(m)["transcript"] != fallback


def test_a_captions_fallback_is_recorded_as_not_the_answer_that_was_asked_for():
    from dubbing import transcript as transcript_mod

    m = manifest.new({"input": "in.mp4", "src_lang": "he", "tgt_lang": "en"})
    m["source"]["transcript_origin"] = "captions"
    assert transcript_mod.is_fallback(m, "auto")
    assert not transcript_mod.is_fallback(m, "captions")   # captions were asked for
    m["source"]["transcript_origin"] = "asr"
    assert not transcript_mod.is_fallback(m, "auto")

    # A provisional mark can never equal a recomputed fingerprint, so the stage
    # runs again while everything downstream keeps its place in the chain for
    # as long as the fallback keeps producing the same words.
    fp = manifest.stage_fingerprint(m, "transcript", params_for(m)["transcript"])
    manifest.mark_provisional(m, "transcript", fp)
    assert "transcript" in m["stages"]
    assert not manifest.stage_done(m, tmp_workdir(), "transcript", fp, [])


# ------------------------------------------------ the run's recorded settings

def test_a_bare_rerun_reproduces_the_run_instead_of_overwriting_it():
    """`python -m dubbing <input>` on an existing run is a *re-run*. It used to
    overwrite `m["source"]` with argparse's defaults, which changed the segments,
    translate and timeline fingerprints and a changed segments fingerprint empties
    `m["segments"]`, taking every edit, lock and passthrough in the project with it."""
    made = cli.parse_args(["in.mp4", "--genre", "movie", "--register", "dialogue",
                           "--dub-foreign"])
    cli.resolve_settings(made)
    m = manifest.new(cli.source_record(made))
    assert m["source"]["genre"] == "movie" and m["source"]["dub_foreign"] is True

    again = cli.parse_args(["in.mp4"])                 # the user types no flags
    cli.resolve_settings(again, m)
    assert cli.source_record(again) == cli.source_record(made)
    assert cli.stage_params(again, m) == cli.stage_params(made, m)


def test_an_explicit_flag_still_beats_what_the_run_recorded():
    m = manifest.new({"input": "in.mp4", "src_lang": "he", "tgt_lang": "en",
                      "genre": "movie", "register": "dialogue", "dub_foreign": True})
    args = cli.parse_args(["in.mp4", "--genre", "documentary", "--no-dub-foreign"])
    cli.resolve_settings(args, m)
    assert args.genre == "documentary"          # the command line is the last word
    assert args.dub_foreign is False            # …including turning a flag back off
    assert args.register == "dialogue"          # and it says nothing about the rest


def test_a_run_that_recorded_nothing_gets_the_documented_defaults():
    args = cli.parse_args(["in.mp4"])
    assert args.genre is None                   # "nobody said" is a distinct answer
    cli.resolve_settings(args, manifest.new({"input": "in.mp4"}))
    assert (args.genre, args.register, args.transcript, args.tts_model,
            args.dub_foreign) == ("documentary", "narration", "auto", "1.7b", False)
    assert cli.source_record(args)["genre"] == "documentary"


def test_the_studio_resolves_a_runs_settings_the_way_the_cli_does():
    # One resolution, two front ends: `edit._args` is what `rebuild` computes its
    # fingerprints from, so a drift here is a whole tail of the pipeline redone.
    made = cli.parse_args(["in.mp4", "--genre", "movie", "--transcript", "asr"])
    cli.resolve_settings(made)
    m = manifest.new(cli.source_record(made))
    assert cli.stage_params(edit._args(m), m) == cli.stage_params(made, m)


def test_content_fingerprint_answers_what_a_derived_artifact_was_made_from(tmp_path):
    # `report.json` is written beside the manifest and describes the segments as
    # they were. An edit changes no stage parameter, so a stage fingerprint cannot
    # tell the two apart this one can, and survives the JSON round trip a report
    # is compared across.
    m = two_segs()
    before = manifest.content_fingerprint(m)
    manifest.save(tmp_path, m)
    assert manifest.content_fingerprint(manifest.load(tmp_path)) == before
    edit.set_text(m, m["segments"][0]["uid"], text_en="a better line")
    edited = manifest.content_fingerprint(m)
    assert edited != before
    # Stage marks are not in it: `report.run` is called before its own stage is
    # marked, so a fingerprint that counted them could never match afterwards.
    manifest.mark_stage(m, "report", "anything")
    assert manifest.content_fingerprint(m) == edited


def test_clear_downstream_is_the_shared_helper():
    m = two_segs()
    m["stages"] = {s: {"fp": "x"} for s in ("translate", "tts", "timeline")}
    assert manifest.clear_downstream(m, "tts") == ["timeline", "mix", "report"]
    assert set(m["stages"]) == {"translate", "tts"}
    assert "place" not in m["segments"][0]


def test_args_reads_the_apps_recorded_options():
    # The studio app records run settings under source["app_opts"]; reading only
    # the CLI's flat keys made every app project re-place its timeline with
    # documentary rules whatever genre was chosen at import.
    m = mk(seg())
    m["source"]["app_opts"] = {"genre": "movie", "register": "dialogue"}
    args = edit._args(m)
    assert args.genre == "movie" and args.register == "dialogue"
    # The flat (CLI) spelling wins when both exist.
    m["source"]["genre"] = "documentary"
    assert edit._args(m).genre == "documentary"
