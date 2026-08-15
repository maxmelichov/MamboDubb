"""Honest failure accounting and lock coherence.

One rule holds this file together: *a verdict written by a failure path must be
as visible and as reversible as the user verdict it overrules.* Every test here
pins one place where the pipeline used to overrule the user, degrade itself, or
give up on a segment without saying so — or where two code paths disagreed about
which lock protects the same undo.
"""

from __future__ import annotations

import pytest

from dubbing import edit, manifest, segments as seg_mod, timeline
from dubbing import tts as tts_mod


def seg(uid=None, **kw):
    s = {"id": kw.pop("id", 0), "start": 0.0, "end": 2.0, "speaker": "S0",
         "text": "aa bb cc dd", "keep": False, "keep_reason": None}
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


def locked_keep(**kw):
    """A segment the pipeline gave up on and the user then settled: keep it."""
    base = seg(keep=True, keep_reason="tts_failed", text_en="Hi",
               tts={"clip": "clips/k.wav", "dur": 2.0, "verify": "keep"},
               locked={"keep": True})
    base.update(kw)
    return base


# ------------------------------------------------- one undo, one lock (D-4)

def test_every_path_that_undoes_a_pipeline_keep_honours_the_same_lock():
    # `manifest.reset_stage`, `edit.invalidate` and `tts.clear_failed_keeps` all
    # undo the same verdict (a keep the pipeline decided for itself). They used to
    # disagree about which lock protects it — clear_failed_keeps guarded `tts`
    # while the other two guarded `keep` — so a user-locked keep was silently
    # re-decided on every tts run.
    a = locked_keep()
    manifest.reset_stage({"segments": [a]}, "tts")
    assert (a["keep"], a["keep_reason"]) == (True, "tts_failed")

    m = mk(locked_keep())
    b = m["segments"][0]
    edit.invalidate(m, b["uid"], stages={"tts"})
    assert (b["keep"], b["keep_reason"]) == (True, "tts_failed")

    c = locked_keep()
    assert tts_mod.clear_failed_keeps([c]) == []
    assert (c["keep"], c["keep_reason"]) == (True, "tts_failed")
    assert c["tts"]["clip"] == "clips/k.wav"   # and its original-audio slice stands


def test_an_unlocked_pipeline_keep_is_still_re_decided_everywhere():
    a = locked_keep(locked=None)
    a.pop("locked")
    manifest.reset_stage({"segments": [a]}, "tts")
    assert (a["keep"], a["keep_reason"]) == (False, None)

    b = locked_keep()
    b.pop("locked")
    assert tts_mod.clear_failed_keeps([b]) == [0]
    assert "tts" not in b


# ------------------------------------- locks nothing could honour (D-9/S-F10)

def test_place_and_bounds_are_not_lockable_because_nothing_honours_them():
    # `timeline.run` rewrites every seg["place"] in one forward pass — placement is
    # all-or-nothing by design — and re-segmentation rebuilds every span, so a
    # `locked.place` / `locked.bounds` was state that lied. The API rejects them
    # rather than accepting a promise it cannot keep.
    assert "place" not in manifest.LOCK_FIELDS
    assert "bounds" not in manifest.LOCK_FIELDS
    m = mk(seg(text_en="Hi"))
    uid = m["segments"][0]["uid"]
    with pytest.raises(edit.EditError):
        edit.set_locked(m, uid, {"place": True})
    with pytest.raises(edit.EditError):
        edit.set_locked(m, uid, {"bounds": True})
    edit.set_bounds(m, uid, 0.0, 1.5)
    assert not manifest.is_locked(m["segments"][0], "bounds")


# --------------------------------------- one subtitle guard, two callers (D-10)

def test_invalidate_keeps_the_subtitle_reset_stage_would_have_kept():
    # A keep whose `text_en` is a subtitle written by the *segments* stage is not
    # translate's to discard: translate never refills it, so dropping it leaves a
    # kept span with no subtitle at all. `reset_stage` has always known this; the
    # per-segment door did not.
    m = mk(seg(keep=True, keep_reason=seg_mod.PASSTHROUGH_REASON, text_en="…"))
    uid = m["segments"][0]["uid"]
    edit.invalidate(m, uid, stages={"translate"})
    assert m["segments"][0]["text_en"] == "…"
    # A keep translate *does* subtitle resets with translate, as before.
    m2 = mk(seg(keep=True, keep_reason="foreign", text_en="A subtitle"))
    edit.invalidate(m2, m2["segments"][0]["uid"], stages={"translate"})
    assert "text_en" not in m2["segments"][0]


def test_set_keep_still_drops_the_line_the_flip_invalidated():
    # The guard above must not resurrect the other bug: a dub's line left behind
    # on a fresh keep is what makes a segment speak a placeholder later.
    m = mk(seg(text_en="a dubbed line", tts={"clip": "clips/a.wav", "dur": 1.0}))
    uid = m["segments"][0]["uid"]
    edit.set_keep(m, uid, True)
    assert "text_en" not in m["segments"][0]
    assert "tts" not in m["segments"][0]


# ------------------------------------------- apply_passthrough vs locks (D-11)

def test_apply_passthrough_releases_the_tts_lock_it_invalidates():
    # The flip makes the clip the wrong *kind* (a synthesis for a keep), so the
    # lock on it is answered explicitly — exactly what `edit.set_keep` does —
    # instead of the record being deleted with the lock left behind, claiming the
    # user approved a clip that no longer exists.
    segs = [seg(passthrough=True, text_en="Hello", locked={"tts": True},
                tts={"clip": "clips/a.wav", "dur": 1.0},
                place={"start": 0.0, "end": 1.0, "clip": "clips/a.wav"})]
    assert seg_mod.apply_passthrough(segs) == [0]
    assert "tts" not in segs[0] and "place" not in segs[0]
    assert not manifest.is_locked(segs[0], "tts")


# --------------------------------------------- the user's word survives (D-13)

def test_a_merged_manual_keep_carries_the_passthrough_that_re_attaches_it():
    # `carry_passthrough` re-attaches overrides after a re-segmentation, and
    # `passthrough` is the only thing it can re-attach. A merge that stamps the
    # manual keep reason without it invents a keep no later run can reproduce.
    m = mk(seg(id=0, start=0.0, end=2.0, text="aa bb", keep=True,
               keep_reason="foreign"),
           seg(id=1, start=2.0, end=4.0, text="cc dd", keep=True,
               keep_reason=seg_mod.PASSTHROUGH_REASON, passthrough=True))
    uid = edit.merge(m, m["segments"][0]["uid"], m["segments"][1]["uid"])
    merged = edit.find(m, uid)
    assert merged["keep"] and merged["keep_reason"] == "manual"
    assert merged["passthrough"] is True


def test_any_user_door_keep_carries_the_passthrough_bit():
    # The app can pass its own reason through `set_keep`; the pipeline honours
    # `passthrough`, not the reason string, so a keep written without it is
    # reverted by the next headless run.
    m = mk(seg(text_en="Hi"), seg(id=1, start=2.0, end=4.0, text_en="Ho"))
    a, b = (s["uid"] for s in m["segments"])
    edit.set_keep(m, a, True, "foreign")
    assert m["segments"][0]["passthrough"] is True
    edit.set_keep(m, b, True)
    assert m["segments"][1]["passthrough"] is True


def test_split_and_merge_preserve_the_advisory_language_label():
    # `detected_lang` is what the app reads to SUGGEST passthrough. Carved out of
    # a segment, both halves still cover the span it was heard over.
    m = mk(seg(text="aa bb cc dd", detected_lang="en"))
    left, right = edit.split(m, m["segments"][0]["uid"], 1.0)
    assert edit.find(m, left)["detected_lang"] == "en"
    assert edit.find(m, right)["detected_lang"] == "en"
    uid = edit.merge(m, left, right)
    assert edit.find(m, uid)["detected_lang"] == "en"


# ------------------------------------------------ the timeline says what it did

def overrunning_run():
    """A dub far too long for its slot, walled by a speaker change."""
    m = mk(seg(start=0.0, end=2.0, speaker="S0", text_en="one two three four",
               tts={"clip": "clips/a.wav", "dur": 9.0}),
           seg(id=1, start=2.0, end=4.0, speaker="S1", text_en="five six",
               tts={"clip": "clips/b.wav", "dur": 1.0}))
    m["source"]["duration"] = 30.0
    return m


def test_place_carries_the_overrun_it_measured(monkeypatch, tmp_path):
    # `place` measures cross-speaker overrun precisely so a drifted line is
    # explicable; `run` used to drop it on the floor when writing seg["place"].
    m = overrunning_run()
    monkeypatch.setattr(timeline.audio, "atempo", lambda *a, **k: None)
    monkeypatch.setattr(timeline.audio, "duration", lambda p: 7.0)
    timeline.run(m, tmp_path)
    assert m["segments"][0]["place"]["overrun"] > 0.0
    assert m["segments"][1]["place"]["overrun"] == 0.0


def test_a_shortening_the_synthesiser_refused_is_recorded_not_swallowed(
        monkeypatch, tmp_path):
    # The translator's refusal is logged one line up; the synthesiser's was not,
    # so a line that drifted after a failed rescue looked identical to one nobody
    # ever tried to rescue.
    m = overrunning_run()
    monkeypatch.setattr(timeline.audio, "atempo", lambda *a, **k: None)
    monkeypatch.setattr(timeline.audio, "duration", lambda p: 7.0)
    timeline.run(m, tmp_path, shorten_many=lambda reqs: {s["id"]: "short" for s, _ in reqs},
                 resynth_many=lambda items: {s["id"]: None for s, _ in items})
    assert m["segments"][0]["place"]["shorten"] == "synthesis-refused"


def test_a_shortening_the_translator_refused_is_recorded_too(monkeypatch, tmp_path):
    m = overrunning_run()
    monkeypatch.setattr(timeline.audio, "atempo", lambda *a, **k: None)
    monkeypatch.setattr(timeline.audio, "duration", lambda p: 7.0)
    timeline.run(m, tmp_path, shorten_many=lambda reqs: {s["id"]: "" for s, _ in reqs},
                 resynth_many=lambda items: {})
    assert m["segments"][0]["place"]["shorten"] == "translator-refused"
