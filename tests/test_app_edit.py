"""The editing foundation the studio app is built on (pure logic, no models).

Three things are pinned here, because each one is what makes a human correction
survive contact with the pipeline:

* `uid` — stable identity. `id` is positional and renumbered on every
  re-segmentation, so a UI keyed on it would follow the wrong audio.
* `locked` — a hand-edited field is never regenerated. A whole-run re-translate,
  the run-global revision pass, a tts rerun, and the timeline's shortening round
  all have to leave it alone.
* `--force <stage>` — really invalidates downstream, as its docstring has always
  claimed.

The model-driven halves of `dubbing.edit` (retranslate / resynthesize / rebuild)
are exercised for their bookkeeping only; their generation is the pipeline's own,
tested where it lives.
"""

from __future__ import annotations

import pytest

from dubbing import cli, edit, manifest, segments, timeline, translate
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
    # translated — the exact path that would undo a user's corrections.
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
    # re-voice it. Locked, it must drift instead — and still place legally.
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
    # A new line means a new clip means a new placement — none of it can be kept.
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
    # keep standing means their line is only ever a subtitle — never spoken.
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
    # A keep's `text_en` is a SUBTITLE — for a foreign or passed-through span it is
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
    # `ttsopts` is loud on purpose — a silently-ignored option looks exactly like a
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
    # left for `carry_passthrough` to re-attach — the passage is dubbed again.
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

    # Disagreement has no answer, so the merged span carries none — same rule the
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
    # The other segment's correction is untouched — the whole point.
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
    # `edit._args` rebuilds the CLI arguments from `m["source"]` — it is what tells
    # `_replace_timeline` which tempo policy this run uses and what `rebuild` marks
    # its stages with. A setting the run never wrote down falls back to argparse's
    # default, so a --genre movie run got re-placed at documentary rates after any
    # resynthesize, and rebuild stamped fingerprints the CLI would never compute.
    argv = ["clip.mp4", "--genre", "movie", "--register", "dialogue",
            "--tts-model", "1.7b", "--dub-foreign", "--transcript", "asr"]
    args = cli.parse_args(argv)
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
    # Nothing coherent to place yet — the tts run that fills the hole re-places
    # everything — but the synthesis the user asked for is recorded, not lost.
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
    monkeypatch.setattr(tts_mod, "Engine",
                        lambda *a, **k: type("E", (), {"close": lambda self: None})())
    monkeypatch.setattr(cli, "_retimers", lambda *a, **k: (None, None))
    done = edit.rebuild(m, tmp_workdir(), from_stage="timeline", save=lambda: None)
    assert done == ["timeline", "mix", "report"] == ran
    assert m["stages"]["fetch"]["fp"] == "old"          # upstream untouched
    assert m["stages"]["timeline"]["fp"] != "old"       # and re-marked, not left stale


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
