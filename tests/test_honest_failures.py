"""Honest failure accounting and lock coherence.

One rule holds this file together: *a verdict written by a failure path must be
as visible and as reversible as the user verdict it overrules.* Every test here
pins one place where the pipeline used to overrule the user, degrade itself, or
give up on a segment without saying so or where two code paths disagreed about
which lock protects the same undo.
"""

from __future__ import annotations

import pytest

from dubbing import PASSTHROUGH_REASON
from dubbing import cli, edit, manifest, report as report_mod, segments as seg_mod
from dubbing import timeline
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
    # disagree about which lock protects it clear_failed_keeps guarded `tts`
    # while the other two guarded `keep` so a user-locked keep was silently
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
    # `timeline.run` rewrites every seg["place"] in one forward pass placement is
    # all-or-nothing by design and re-segmentation rebuilds every span, so a
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
    m = mk(seg(keep=True, keep_reason=PASSTHROUGH_REASON, text_en="…"))
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
    # lock on it is answered explicitly exactly what `edit.set_keep` does —
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
               keep_reason=PASSTHROUGH_REASON, passthrough=True))
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


# ------------------------------------------- the safety net reports on itself

def record(verdict, **kw):
    from pathlib import Path

    return tts_mod.Engine._record(Path("clips/a.wav"), verdict, 0, **kw)


def test_a_clip_no_asr_ever_heard_is_recorded_as_unverified():
    # `_verify` accepts on length alone when there is no verification ASR. That is
    # a verdict of its own the record used to say "ok", so a run where nothing
    # was checked read exactly like a run where everything passed.
    rec = record({"ok": True, "overlap": 1.0, "heard": tts_mod.NO_ASR, "dur": 1.0,
                  "verified": False})
    assert rec["verify"] == "unverified"
    # A verdict from before the flag existed was written by a run that did have an
    # ASR: it still reads as verified, so no cached clip changes meaning.
    old = record({"ok": True, "overlap": 0.9, "heard": "hello", "dur": 1.0})
    assert old["verify"] == "ok"


def fake_report_run(m, tmp_path, monkeypatch):
    monkeypatch.setattr(report_mod.transcript, "load_words", lambda wd, mm: [])
    monkeypatch.setattr(report_mod, "uncovered_spans", lambda *a, **k: [])
    m.setdefault("files", {})["source_wav"] = "source.wav"
    m["source"]["duration"] = 30.0
    return report_mod.run(m, tmp_path)


def report_segments():
    return mk(
        seg(start=0.0, end=2.0, text_en="Hello",
            tts={"clip": "clips/a.wav", "dur": 1.0, "verify": "unverified"},
            place={"start": 0.0, "end": 1.0, "clip": "clips/a.wav", "drift": 0.0,
                   "rate": 1.0, "overrun": 0.4}),
        seg(id=1, start=2.0, end=4.0, keep=True, keep_reason="user", text_en="…",
            passthrough=True,
            tts={"clip": "clips/b.wav", "dur": 2.0, "verify": "keep"},
            place={"start": 2.0, "end": 4.0, "clip": "clips/b.wav", "drift": 0.0,
                   "rate": 1.0, "overrun": 0.0, "shorten": "synthesis-refused"}))


def test_report_counts_what_the_run_could_not_do(tmp_path, monkeypatch):
    m = report_segments()
    m["health"] = {"segments.diarization": "unavailable (no HF_TOKEN) every "
                                           "speaker cloned as one voice"}
    rep = fake_report_run(m, tmp_path, monkeypatch)
    assert rep["verify"]["unverified"] == 1        # never folded into "ok"
    assert rep["overrun"] == {"count": 1, "max": 0.4}
    assert rep["subtitles_failed"] == [1]          # a kept span showing "…"
    assert rep["shorten_abandoned"][0]["reason"] == "synthesis-refused"
    assert "segments.diarization" in rep["degraded"]


def test_a_clip_in_the_wrong_voice_is_counted_and_named(tmp_path, monkeypatch, capsys):
    # The right words in another person's voice. Word overlap scores it as a pass
    # (0.889 on the line that started this), so the only place a reviewer could
    # ever find it was by embedding the clips themselves report.json says it now.
    m = report_segments()
    m["segments"][0]["tts"] = {"clip": "clips/a.wav", "dur": 1.0, "overlap": 0.889,
                               "verify": "wrong_voice", "voice": 0.066}
    rep = fake_report_run(m, tmp_path, monkeypatch)
    assert rep["verify"]["wrong_voice"] == 1 and rep["verify"]["ok"] == 0
    assert rep["wrong_voice"] == [{"id": 0, "start": 0.0, "speaker": "S0",
                                   "voice": 0.066}]
    assert "another voice" in capsys.readouterr().err


def test_the_summary_says_when_nothing_was_verified(tmp_path, monkeypatch, capsys):
    fake_report_run(report_segments(), tmp_path, monkeypatch)
    assert "NOT ONE clip was verified" in capsys.readouterr().err


def test_diarization_failure_is_recorded_not_only_printed():
    # The commonest cause is a missing HF_TOKEN a five-second fix the user never
    # hears about while the fallback (one voice for the whole file) only prints.
    from pathlib import Path

    notes: dict[str, str] = {}
    turns = seg_mod.diarize(Path("/nope/vocals.wav"),
                            note=lambda why: notes.__setitem__("diarization", why))
    assert turns == []
    assert "unavailable" in notes["diarization"]


# ------------------------------------- a clip that says the old words (D-8)

def clip_on_disk(tmp_path, name="clips/a.wav"):
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    return p


def dubbed_with_clip(text_en="Hello there", **kw):
    s = seg(text_en=text_en,
            tts={"clip": "clips/a.wav", "dur": 1.9, "tries": 1, "overlap": 1.0,
                 "verify": "ok", "text_sha": tts_mod.text_sha(text_en)})
    s.update(kw)
    return s


def test_a_clip_is_re_queued_when_the_line_moved_under_it(tmp_path):
    clip_on_disk(tmp_path)
    s = dubbed_with_clip()
    assert not tts_mod.needs_synthesis(s, tmp_path)
    s["text_en"] = "Greetings, friend"          # a translate rerun replaced the line
    assert tts_mod.clip_text_stale(s)
    assert tts_mod.needs_synthesis(s, tmp_path)
    assert [x["id"] for x in tts_mod.pending([s], tmp_path)] == [0]


def test_a_locked_clip_whose_line_moved_is_a_conflict_not_a_silent_verdict(tmp_path):
    clip_on_disk(tmp_path)
    s = dubbed_with_clip(locked={"tts": True})
    s["text_en"] = "Greetings, friend"
    # Never silently regenerated the user approved this take ...
    assert not tts_mod.needs_synthesis(s, tmp_path)
    assert tts_mod.pending([s], tmp_path) == []
    # ... and never silently kept either: it is surfaced.
    assert tts_mod.stale_locked_clip(s)


def test_records_written_before_the_fingerprint_existed_are_left_alone(tmp_path):
    clip_on_disk(tmp_path)
    s = seg(text_en="Hello", tts={"clip": "clips/a.wav", "dur": 1.0, "verify": "ok"})
    assert not tts_mod.clip_text_stale(s)
    assert not tts_mod.needs_synthesis(s, tmp_path)


# ---------------------------- a segment's own target language (D-7 / audit 14)

def fr_engine(tmp_path):
    m = {"source": {"src_lang": "he", "tgt_lang": "en"},
         "files": {"source_wav": "source.wav", "vocals": "stems/vocals.wav"},
         "speakers": {}, "segments": []}
    eng = tts_mod.Engine(m, tmp_path)
    eng.ref_for = lambda s, opts=None: (tmp_path / "refs/a.wav", "ref:1.00-4.00")
    return eng


def test_a_segments_own_target_language_reaches_the_synthesiser(tmp_path):
    # `edit.set_langs` honours seg["tgt_lang"] for translation. The Engine read
    # only the run's target, so a French line was prepared, keyed, synthesised and
    # ASR-verified as English it failed verification every time and degraded to
    # tts_failed, with nothing pointing at the override as the cause.
    eng = fr_engine(tmp_path)
    fr = seg(text_en="Bonjour tout le monde", tgt_lang="fr")
    assert eng.tgt_for(fr) == "fr" and eng.tgt_for(seg()) == "en"
    plan = eng._plan(fr, fr["text_en"])
    assert plan.tgt == "fr" and plan.src == "he"
    # The cache key MUST carry it: an en clip of the same text is different audio.
    args = ("bonjour.", "ref:1.00-4.00", 42, False)
    assert eng._cache_key(*args, tts_mod.ttsopts.DEFAULT, "fr") != eng._cache_key(*args)


def test_the_verifier_asks_the_asr_for_that_language(tmp_path, monkeypatch):
    eng = fr_engine(tmp_path)
    asked: list[str] = []
    monkeypatch.setattr(eng, "asr_for", lambda tgt=None: asked.append(tgt) or None)
    monkeypatch.setattr(tts_mod.audio, "duration", lambda p: 1.5)
    ok, _ov, heard = eng._verify(tmp_path / "clips/a.wav", "bonjour tout le monde.",
                                 "fr", "he")
    assert asked == ["fr"]
    assert ok and heard == tts_mod.NO_ASR       # no model here: unverified, not "ok"


def test_the_synth_is_told_which_language_to_speak(tmp_path, monkeypatch):
    eng = fr_engine(tmp_path)
    calls: list[str | None] = []

    class FakeSynth:
        def generate(self, speak, ref, out, *, seed, greedy, opts=None,
                     synth=None, lang=None, carrier=None):
            calls.append(lang)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"x")
            return out

    eng.synth_for = lambda opts=None: FakeSynth()
    monkeypatch.setattr(tts_mod.audio, "trim_leading_silence", lambda a, b: None)
    eng._attempt(1, "bonjour.", tmp_path / "refs/a.wav", "ref:1.00-4.00", 7, 0,
                 tts_mod.ttsopts.DEFAULT, "fr")
    assert calls == ["fr"]


# --------------------------------- tts_opts on a kept segment (D-9 / audit 9)

class KeepEngine(tts_mod.Engine):
    """A real Engine with the ffmpeg calls replaced the bookkeeping is the point."""

    def __init__(self, tmp_path, monkeypatch):
        m = {"source": {"src_lang": "he", "tgt_lang": "en"},
             "files": {"source_wav": "source.wav", "vocals": "stems/vocals.wav"},
             "speakers": {}, "segments": []}
        super().__init__(m, tmp_path)
        self.cut: list[tuple] = []
        self.tempo: list[float] = []
        monkeypatch.setattr(tts_mod.audio, "extract_slice",
                            lambda src, a, b, out: (self.cut.append((a, b, out)),
                                                    out.write_bytes(b"x")))
        monkeypatch.setattr(tts_mod.audio, "atempo",
                            lambda raw, out, rate: (self.tempo.append(rate),
                                                    out.write_bytes(b"x")))
        monkeypatch.setattr(tts_mod.audio, "duration",
                            lambda p: 2.0 / (self.tempo[-1] if self.tempo else 1.0))


def test_speed_on_a_kept_segment_reaches_the_slice(tmp_path, monkeypatch):
    eng = KeepEngine(tmp_path, monkeypatch)
    plain = eng.keep_clip(seg(keep=True, keep_reason="user"))
    assert eng.tempo == [] and plain["span"] == 2.0

    fast = eng.keep_clip(seg(keep=True, keep_reason="user", tts_opts={"speed": 1.5}))
    assert eng.tempo == [1.5]
    # A different clip for different audio, and the span it was cut for recorded
    # so the timeline can still prove it is this segment's audio.
    assert fast["clip"] != plain["clip"] and fast["span"] == 2.0
    assert fast["opts"]                        # fingerprinted, so a change re-cuts


def test_changing_speed_on_a_kept_segment_re_cuts_the_slice(tmp_path, monkeypatch):
    eng = KeepEngine(tmp_path, monkeypatch)
    s = seg(keep=True, keep_reason="user", tts_opts={"speed": 1.5})
    s["tts"] = eng.keep_clip(s)
    (tmp_path / s["tts"]["clip"]).write_bytes(b"x")
    assert not tts_mod.keep_needs_slice(s, tmp_path)
    s["tts_opts"] = {"speed": 1.2}
    assert tts_mod.keep_needs_slice(s, tmp_path)


def test_the_timeline_accepts_a_kept_clip_a_tempo_made_shorter():
    m = mk(seg(keep=True, keep_reason="user", text_en="…", tts_opts={"speed": 2.0},
               tts={"clip": "clips/k.wav", "dur": 1.0, "span": 2.0, "verify": "keep"}))
    assert timeline.build_items(m)[0]["dur"] == 1.0
    # …and still refuses a slice cut for a different moment (ids are renumbered;
    # an id-keyed cache once handed back the wrong span's audio).
    m["segments"][0]["tts"]["span"] = 7.0
    with pytest.raises(AssertionError):
        timeline.build_items(m)


def test_synthesis_only_options_are_refused_on_a_kept_segment():
    m = mk(seg(keep=True, keep_reason="user", text_en="…"))
    uid = m["segments"][0]["uid"]
    with pytest.raises(edit.EditError) as exc:
        edit.set_tts_opts(m, uid, seed=7)
    assert "original audio" in str(exc.value)
    for key, value in (("greedy", True), ("ref", "refs/x.wav"), ("model", "0.6b"),
                       ("max_new_tokens", 512), ("top_k", 5)):
        with pytest.raises(edit.EditError):
            edit.set_tts_opts(m, uid, **{key: value})
    # What does apply to a kept slice is accepted.
    edit.set_tts_opts(m, uid, speed=1.4)
    assert m["segments"][0]["tts_opts"] == {"speed": 1.4}
    # And on a dubbed segment nothing changes.
    m2 = mk(seg(text_en="Hello"))
    edit.set_tts_opts(m2, m2["segments"][0]["uid"], seed=7)
    assert m2["segments"][0]["tts_opts"] == {"seed": 7}


# ------------------------------------- the job says what it did (S-F3/F4)

class FallbackEngine:
    """Every synthesis fails; keeps get their slice. Same surface as tts.Engine."""

    def __init__(self, *a, **kw):
        pass

    def build_speaker_refs(self):
        pass

    def clip_for(self, seg, text):
        return None

    def keep_clip(self, seg):
        return {"clip": "clips/keep.wav", "dur": round(seg["end"] - seg["start"], 3),
                "tries": 0, "overlap": 1.0, "verify": "keep"}

    def close(self):
        pass


def test_resynthesize_does_not_call_a_keep_fallback_a_synthesis(monkeypatch, tmp_path):
    # "synthesized 1 segment(s)" for a segment that was NOT synthesized it fell
    # back to its original audio and no word about the placement that never ran.
    monkeypatch.setattr(tts_mod, "Engine", FallbackEngine)
    m = mk(seg(start=0.0, end=2.0, text_en="Hello there"),
           seg(id=1, start=2.0, end=4.0, text_en="still being edited"))
    msgs: list[tuple[float, str]] = []
    out = edit.resynthesize(m, tmp_path, [m["segments"][0]["uid"]],
                            progress=lambda f, msg: msgs.append((f, msg)))
    assert out.summary == {"synthesized": 0, "fell_back": 1, "kept": 0,
                           "deferred_placement": True}
    last = msgs[-1][1]
    assert "synthesized 0" in last
    assert "fell back to the original audio" in last
    assert "placement deferred" in last
    # The mapping every existing caller reads is unchanged.
    assert out[m["segments"][0]["uid"]]["verify"] == "keep"


def test_resynthesize_says_placement_ran_when_it_did(monkeypatch, tmp_path):
    monkeypatch.setattr(edit, "_replace_timeline", lambda m, wd: True)

    class Engine(FallbackEngine):
        def clip_for(self, seg, text):
            return {"clip": "clips/new.wav", "dur": 1.0, "verify": "ok"}

    monkeypatch.setattr(tts_mod, "Engine", Engine)
    m = mk(seg(text_en="Hello there"))
    msgs: list[tuple[float, str]] = []
    out = edit.resynthesize(m, tmp_path, [m["segments"][0]["uid"]],
                            progress=lambda f, msg: msgs.append((f, msg)))
    assert out.summary == {"synthesized": 1, "fell_back": 0, "kept": 0,
                           "deferred_placement": False}
    assert msgs[-1][1] == "synthesized 1"


def test_report_names_a_locked_clip_that_speaks_the_old_line(tmp_path, monkeypatch):
    m = report_segments()
    s = m["segments"][0]
    s["locked"] = {"tts": True}
    s["tts"]["text_sha"] = tts_mod.text_sha("something else entirely")
    rep = fake_report_run(m, tmp_path, monkeypatch)
    assert [x["id"] for x in rep["stale_locked_clips"]] == [0]


# ------------------------------- the segment with no clip at all (D-5)

def test_the_timeline_names_a_segment_it_has_no_audio_for():
    """The hole `edit.set_text` opens, and the silence it used to become.

    Dropping a dubbable segment's clip and never refilling it left
    `build_items` reading `seg["tts"]["dur"]` on a segment that had no `tts`.
    The KeyError was caught one frame up as "placement deferred", and the mix
    then filled the whole span with the original vocals: the line played in the
    source language and nothing anywhere said so.
    """
    m = mk(seg(text_en="Hello", tts={"clip": "clips/a.wav", "dur": 1.0, "verify": "ok"}),
           seg(id=1, start=2.0, end=4.0, text_en="World"))
    with pytest.raises(AssertionError) as exc:
        timeline.build_items(m)
    assert "[1]" in str(exc.value)          # named, not merely counted
    # …and a kept segment whose slice went missing is the same failure.
    m["segments"][1]["tts"] = {"clip": "clips/b.wav", "dur": 2.0, "verify": "ok"}
    assert len(timeline.build_items(m)) == 2


def test_report_counts_an_unvoiced_segment_as_unaccounted(tmp_path, monkeypatch):
    """A stale placement beside a dropped clip is not coverage."""
    m = report_segments()
    (tmp_path / "clips").mkdir()
    for name in ("a.wav", "b.wav"):
        (tmp_path / "clips" / name).write_bytes(b"clip")
    assert fake_report_run(m, tmp_path, monkeypatch)["unaccounted"] == []

    m["segments"][0].pop("tts")             # clip invalidated, placement left behind
    rep = fake_report_run(m, tmp_path, monkeypatch)
    assert rep["unaccounted"] == [0]
    # And counting the verdicts does not die on the segment that has none.
    assert rep["verify"]["keep"] == 1


def tts_failed_run(*, dubbed_first=False):
    """The issue #15 shape: every line kept because synthesis produced nothing.

    A `tts_failed` keep still carries a clip and a placement (the segment's own
    source audio, standing in for the dub that never arrived), so `unaccounted`
    stays empty and the run has nothing else to complain about. That is exactly
    why it used to pass for a clean run.
    """
    first = seg(start=0.0, end=2.0, text_en="Hello",
                keep=not dubbed_first,
                keep_reason=None if dubbed_first else "tts_failed",
                tts={"clip": "clips/a.wav", "dur": 2.0,
                     "verify": "ok" if dubbed_first else "keep"},
                place={"start": 0.0, "end": 2.0, "clip": "clips/a.wav",
                       "drift": 0.0, "rate": 1.0, "overrun": 0.0})
    second = seg(id=1, start=2.0, end=4.0, text_en="World", keep=True,
                 keep_reason="tts_failed",
                 tts={"clip": "clips/b.wav", "dur": 2.0, "verify": "keep"},
                 place={"start": 2.0, "end": 4.0, "clip": "clips/b.wav",
                        "drift": 0.0, "rate": 1.0, "overrun": 0.0})
    return mk(first, second)


def with_clips(tmp_path):
    (tmp_path / "clips").mkdir()
    for name in ("a.wav", "b.wav"):
        (tmp_path / "clips" / name).write_bytes(b"clip")


def test_a_run_that_dubbed_nothing_at_all_is_not_a_clean_run(tmp_path, monkeypatch,
                                                             capsys):
    # Issue #15: TTS never loaded, every segment fell back to its own audio, and
    # the report said dubbed 0 / degraded {} / exit 0 over a preview that is the
    # untranslated source video. A stage that did not run is not a fallback.
    with_clips(tmp_path)
    rep = fake_report_run(tts_failed_run(), tmp_path, monkeypatch)

    assert rep["dubbed"] == 0 and rep["keep_reasons"] == {"tts_failed": 2}
    assert rep["unaccounted"] == []          # nothing else was going to catch it
    assert rep["tts_unavailable"] == {"kept": 2}
    assert "untranslated original" in rep["degraded"]["tts.synthesis"]
    assert "FAIL: TTS produced nothing" in capsys.readouterr().err


def test_a_partial_tts_failure_is_still_the_per_segment_fallback(tmp_path,
                                                                 monkeypatch, capsys):
    # The boundary. One line dubbed and one kept is the safety net working as
    # designed: a warning in `keep_reasons`, not a failed run.
    with_clips(tmp_path)
    rep = fake_report_run(tts_failed_run(dubbed_first=True), tmp_path, monkeypatch)

    assert rep["dubbed"] == 1 and rep["keep_reasons"] == {"tts_failed": 1}
    assert rep["tts_unavailable"] is None
    assert "tts.synthesis" not in rep["degraded"]
    assert "FAIL: TTS produced nothing" not in capsys.readouterr().err


def test_a_run_that_dubbed_nothing_exits_nonzero():
    """The app's job runner reads the exit code, so this is what reaches the UI."""
    assert cli.report_failed({"unaccounted": [], "tts_unavailable": {"kept": 11}})
    assert not cli.report_failed({"unaccounted": [], "tts_unavailable": None})
    assert cli.report_failed({"unaccounted": [3], "tts_unavailable": None})
    # A stage that never learned the field still decides on the ones it has.
    assert not cli.report_failed({"unaccounted": []})


def test_render_fails_loudly_when_a_segment_ends_up_with_no_audio(tmp_path, monkeypatch):
    """`rebuild` may not report success over a dub with a hole in it."""
    monkeypatch.setattr(report_mod, "run", lambda *a, **k: {"unaccounted": [1]})
    m = mk(seg(text_en="Hello",
               tts={"clip": "clips/a.wav", "dur": 1.0, "verify": "ok"},
               place={"start": 0.0, "end": 1.0, "clip": "clips/a.wav", "drift": 0.0,
                      "rate": 1.0}))
    with pytest.raises(edit.RebuildIncomplete) as exc:
        edit.rebuild(m, tmp_path, from_stage="report")
    assert "no audio in the mix" in str(exc.value)
