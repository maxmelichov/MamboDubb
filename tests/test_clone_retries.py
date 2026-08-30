"""The clone retry ladder, and the two bars that decide when it stops.

A voice clone's quality is chaotically sensitive to which few seconds of the
speaker it was cloned from, and nothing measurable about a reference predicts
it only the verify ASR knows, after the fact. Everything here follows from
that: a retry must change the *reference*, the seed cannot help (and under a
greedy decode does literally nothing), and "good enough to keep" has to be a
lower bar than "good enough to stop looking".

No models: the synthesiser and the verifier are both stubbed. What is under
test is which take gets made, in what order, and which one is kept.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

from dubbing import audio, tts
from dubbing.ttsopts import DEFAULT

SPEAKER = {"ref": "refs/S1.wav",
           # What `build_speaker_refs` keeps for exactly this purpose.
           "ref_alts": [[10.0, 14.5], [20.0, 24.5]]}


class FakeSynth:
    """Records what each attempt asked for and writes a stand-in clip."""

    def __init__(self):
        self.calls: list[dict] = []

    def generate(self, speak, ref, out, *, seed, greedy, opts=DEFAULT,
                 synth=None, lang=None, carrier=None):
        self.calls.append({"ref": ref.name, "seed": seed, "greedy": greedy})
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), np.full(audio.SR, 0.2, dtype="float32"), audio.SR)
        return out


def voice(tag: str) -> np.ndarray:
    """A unit embedding per speaker tag; two tags are orthogonal (cosine 0)."""
    v = np.zeros(8)
    v["abc".index(tag)] = 1.0
    return v


def engine(tmp_path, monkeypatch, overlaps, *, tgt="en", voices=None,
           embeddings=True, dur=1.0):
    """An Engine whose verifier hands back `overlaps`, one per attempt.

    `voices` maps a reference or clip file name to a speaker tag; everything
    unnamed, the segment's own window included, is speaker "a". `embeddings=False`
    is the run with no ECAPA model at all.
    """
    m = {"source": {"src_lang": "he", "tgt_lang": tgt},
         "files": {"vocals": "stems/vocals.wav"},
         "speakers": {"S1": dict(SPEAKER)}, "segments": []}
    eng = tts.Engine(m, tmp_path)
    tags = voices or {}
    monkeypatch.setattr(
        tts, "_embed_windows",
        lambda voc, spans: (np.stack([voice(tags.get(f"{a:.2f}-{b:.2f}", "a"))
                                      for a, b in spans]) if embeddings else None))
    monkeypatch.setattr(
        tts, "_embed_wavfile",
        lambda p: voice(tags.get(p.name, "a")) if embeddings else None)
    (tmp_path / "refs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "refs" / "S1.wav").write_bytes(b"canonical")
    (tmp_path / "refs" / "auto.wav").write_bytes(b"auto")
    fake = FakeSynth()
    eng.synth_for = lambda opts=DEFAULT: fake
    eng.ref_for = lambda seg, opts=DEFAULT: (tmp_path / "refs/auto.wav", "ref:1.00-4.00")
    # 30s of vocals, so the alternative windows can actually be cut.
    eng._voc = np.full(30 * tts.REF_SR, 0.2, dtype="float32")
    seen = iter(overlaps)

    def verify(clip, meta, speak, *lang):
        ov = next(seen)
        return {"ok": ov >= tts.CLONE_MIN_OVERLAP, "overlap": ov, "heard": "x",
                "dur": dur, "verified": True}

    monkeypatch.setattr(eng, "_verify_and_store", verify)
    return eng, fake


def seg(**over):
    s = {"id": 27, "uid": "u27", "start": 0.0, "end": 4.0, "speaker": "S1"}
    s.update(over)
    return s


# ---------------------------------------------- the ladder varies the reference


def test_a_usable_but_not_good_take_does_not_stop_the_ladder(tmp_path, monkeypatch):
    """0.44 used to be stored as "ok" on the first try and never looked at again.

    `CLONE_MIN_OVERLAP` was both the accept bar and the retry trigger, so every
    garble that cleared the floor was accepted before a single retry ran.
    """
    eng, fake = engine(tmp_path, monkeypatch, [0.44, 0.60, 0.92])
    got = eng.clip_for(seg(), "hello world")
    assert got["verify"] == "ok" and got["tries"] == 3
    # Three takes, three different references, and the seed never moved.
    assert [c["ref"] for c in fake.calls] == ["auto.wav", "S1.wav", "ref_10.00-14.50.wav"]
    assert len({c["seed"] for c in fake.calls}) == 1


def test_the_best_take_wins_and_says_it_was_only_accepted(tmp_path, monkeypatch):
    eng, fake = engine(tmp_path, monkeypatch, [0.44, 0.61, 0.50, 0.40])
    got = eng.clip_for(seg(), "hello world")
    # Every rung was walked, and the best of them is a dub — labelled honestly.
    assert got["verify"] == "accepted" and got["overlap"] == 0.61
    assert len(fake.calls) == 4
    # No two rungs are the same take: each is a distinct (reference, decode) pair.
    assert len({(c["ref"], c["greedy"]) for c in fake.calls}) == 4


def test_below_the_floor_is_still_a_soft_accept_then_a_fallback(tmp_path, monkeypatch):
    eng, _fake = engine(tmp_path, monkeypatch, [0.30, 0.25, 0.22, 0.21])
    assert eng.clip_for(seg(), "hello world")["verify"] == "soft"

    eng, _fake = engine(tmp_path, monkeypatch, [0.05, 0.0, 0.0, 0.0])
    assert eng.clip_for(seg(), "hello world") is None


# ------------------------------------------------------- greedy gets a real ladder


def test_a_greedy_ladder_never_makes_the_same_take_twice(tmp_path, monkeypatch):
    """The seed is inert under `do_sample=False`; only the reference isn't.

    `_attempt` used to offset the seed per attempt, which under greedy produced
    MAX_TRIES byte-identical takes filed under different cache keys — a retry
    ladder that could not, even in principle, return anything new.
    """
    eng, fake = engine(tmp_path, monkeypatch, [0.44, 0.44, 0.44, 0.44])
    eng.clip_for(seg(tts_opts={"greedy": True}), "hello world")
    assert len({(c["ref"], c["greedy"]) for c in fake.calls}) == len(fake.calls)
    assert [c["ref"] for c in fake.calls[:3]] == ["auto.wav", "S1.wav",
                                                  "ref_10.00-14.50.wav"]


def test_the_sampled_rescue_fires_once_at_the_end_of_a_greedy_ladder(tmp_path,
                                                                     monkeypatch):
    eng, fake = engine(tmp_path, monkeypatch, [0.44, 0.44, 0.44, 0.90])
    got = eng.clip_for(seg(tts_opts={"greedy": True}), "hello world")
    assert [c["greedy"] for c in fake.calls] == [True, True, True, False]
    assert got["verify"] == "ok" and got["tries"] == 4

    # …and it does not fire at all once a reference has already cleared the bar.
    eng, fake = engine(tmp_path, monkeypatch, [0.44, 0.90])
    eng.clip_for(seg(tts_opts={"greedy": True}), "hello world")
    assert [c["greedy"] for c in fake.calls] == [True, True]


def test_a_pinned_reference_has_no_ladder_but_still_gets_the_other_decode(
        tmp_path, monkeypatch):
    eng, fake = engine(tmp_path, monkeypatch, [0.44, 0.44])
    (tmp_path / "refs" / "pick.wav").write_bytes(b"chosen")
    eng.ref_for = lambda s, opts=DEFAULT: (tmp_path / "refs/pick.wav", "opt:pick")
    eng.clip_for(seg(tts_opts={"ref": "refs/pick.wav"}), "hello world")
    assert {c["ref"] for c in fake.calls} == {"pick.wav"}
    assert [c["greedy"] for c in fake.calls] == [False, True]


# ------------------------------------------------- the alternatives are recorded


def test_build_speaker_refs_keeps_the_runners_up(tmp_path, monkeypatch):
    m = {"source": {"src_lang": "he", "tgt_lang": "en"},
         "files": {"vocals": "stems/vocals.wav"}, "speakers": {},
         "segments": [{"id": i, "start": i * 5.0, "end": i * 5.0 + 4.5, "keep": False,
                       "speaker": "S1"} for i in range(4)]}
    eng = tts.Engine(m, tmp_path)
    eng._voc = np.full(30 * tts.REF_SR, 0.2, dtype="float32")
    monkeypatch.setattr(tts, "_embed_windows", lambda *a, **k: None)
    monkeypatch.setattr(tts, "_embed_wavfile", lambda *a, **k: None)
    eng.build_speaker_refs()
    info = m["speakers"]["S1"]
    assert info["ref_v"] == tts.Engine.REF_BUILD
    # Kept, not discarded and never a second cut of the canonical clip itself.
    alts = [tuple(a) for a in info["ref_alts"]]
    assert 0 < len(alts) <= tts.REF_ALTS
    assert not set(alts) & {tuple(w) for w in info["ref_windows"]}


# --------------------------------------- the pipelined fast path uses the same bar


class _StubEngine:
    """`tts.run`'s collaborator, reduced to the one decision under test."""

    overlap = 0.5

    def __init__(self, m, workdir, **kw):
        self.workdir = workdir
        self.retried: list[int] = []
        self.faults = tts.Repeats()
        (workdir / "clips").mkdir(parents=True, exist_ok=True)

    def flush_faults(self):
        pass

    def build_speaker_refs(self):
        pass

    def _plan(self, seg, text):
        return tts.Plan(text, text, self.workdir / "r.wav", "ref:0.00-4.00",
                        7, DEFAULT, "en", "he", False)

    def _attempt(self, seg_id, speak, ref, key, seed, greedy, opts=DEFAULT,
                 tgt=None, synth=None):
        clip = self.workdir / "clips" / f"{seg_id}.wav"
        clip.write_bytes(b"x")
        return clip, clip.with_suffix(".json"), {
            "ok": True, "overlap": self.overlap, "heard": "x", "dur": 1.0,
            "verified": True}

    @staticmethod
    def _record(clip, verdict, attempt, opts=DEFAULT, text=""):
        return {"clip": f"clips/{clip.name}", "dur": 1.0, "tries": attempt + 1,
                "overlap": verdict["overlap"], "verify": "ok"}

    def clip_for(self, seg, text):
        self.retried.append(seg["id"])
        return {"clip": "clips/best.wav", "dur": 1.0, "tries": 4,
                "overlap": self.overlap, "verify": "accepted"}

    def close(self):
        pass


def _run_with(tmp_path, monkeypatch, overlap):
    made: list[_StubEngine] = []
    monkeypatch.setattr(_StubEngine, "overlap", overlap)
    monkeypatch.setattr(tts, "Engine",
                        lambda m, wd, **kw: made.append(_StubEngine(m, wd)) or made[-1])
    m = {"source": {"src_lang": "he", "tgt_lang": "en"},
         "files": {"source_wav": "source.wav"}, "speakers": {},
         "segments": [{"id": 0, "uid": "s0", "start": 0.0, "end": 4.0, "speaker": "S1",
                       "text": "שלום", "text_en": "hello world", "keep": False,
                       "keep_reason": None}]}
    tts.run(m, tmp_path)
    return m["segments"][0], made[0]


def test_the_first_take_is_only_kept_when_it_is_actually_good(tmp_path, monkeypatch):
    """The fast path stored anything above the floor as "ok" and moved on.

    That is the whole of bug (b): a cache hit at 0.50 never reached the retry
    ladder, so the ladder's quality never mattered.
    """
    seg, eng = _run_with(tmp_path, monkeypatch, 0.50)
    assert eng.retried == [0] and seg["tts"]["verify"] == "accepted"

    seg, eng = _run_with(tmp_path, monkeypatch, 0.95)
    assert eng.retried == [] and seg["tts"]["verify"] == "ok"


# ------------------------------------------- a rung has to be the same person

# The regression these pin: varying the reference fixed garbled audio and broke
# something worse. Diarization had pooled two people under SPEAKER_03, the ladder
# reached for the other one's window because the *label* matched, and a female
# line came back in a male voice with a word overlap of 0.889. Nothing looked at
# who was speaking, so nothing objected.


def test_a_rung_whose_reference_is_another_voice_is_skipped(tmp_path, monkeypatch):
    """The canonical clip is a stranger here, so the ladder walks past it."""
    eng, fake = engine(tmp_path, monkeypatch, [0.44, 0.44, 0.44, 0.44],
                       voices={"S1.wav": "b"})
    eng.clip_for(seg(), "hello world")
    refs = [c["ref"] for c in fake.calls]
    assert "S1.wav" not in refs
    # …and the rungs that ARE this voice still run, in order.
    assert refs == ["auto.wav", "ref_10.00-14.50.wav", "ref_20.00-24.50.wav",
                    "auto.wav"]


def test_a_ladder_with_no_valid_alternative_stays_on_its_own_window(tmp_path,
                                                                    monkeypatch):
    """Every substitute is somebody else, so the retry is the other decode.

    Not "clone from the nearest thing the label offers": that is exactly the
    trade the wrong-voice bug made. A same-window retry can only be a worse take
    of the right person; a stranger's window is a wrong dub that verifies fine.
    """
    eng, fake = engine(tmp_path, monkeypatch, [0.44, 0.44],
                       voices={"S1.wav": "b", "ref_10.00-14.50.wav": "b",
                               "ref_20.00-24.50.wav": "b"})
    eng.clip_for(seg(), "hello world")
    assert [(c["ref"], c["greedy"]) for c in fake.calls] == [("auto.wav", False),
                                                             ("auto.wav", True)]


def test_without_embeddings_the_ladder_substitutes_nothing_and_says_so(tmp_path,
                                                                       monkeypatch):
    """No ECAPA is no way to tell, and "cannot tell" is not permission."""
    eng, fake = engine(tmp_path, monkeypatch, [0.44, 0.44], embeddings=False)
    eng.clip_for(seg(), "hello world")
    assert [c["ref"] for c in fake.calls] == ["auto.wav", "auto.wav"]
    assert "tts.speaker_embeddings" in eng.m["health"]


def test_the_runners_up_exclude_a_window_the_validator_rejected(tmp_path,
                                                                monkeypatch):
    """A window dropped as another voice may not come back as retry material."""
    m = {"source": {"src_lang": "he", "tgt_lang": "en"},
         "files": {"vocals": "stems/vocals.wav"}, "speakers": {},
         "segments": [{"id": i, "start": i * 5.0, "end": i * 5.0 + 4.5, "keep": False,
                       "speaker": "S1"} for i in range(4)]}
    eng = tts.Engine(m, tmp_path)
    eng._voc = np.full(30 * tts.REF_SR, 0.2, dtype="float32")
    # One of the four candidate windows is a different person the diarization
    # mislabel this whole chain starts from.
    tags = {"10.00-14.50": "b"}
    monkeypatch.setattr(tts, "_embed_windows",
                        lambda voc, spans: np.stack(
                            [voice(tags.get(f"{a:.2f}-{b:.2f}", "a")) for a, b in spans]))
    monkeypatch.setattr(tts, "_embed_wavfile", lambda p: voice("a"))
    eng.build_speaker_refs()
    alts = [tuple(a) for a in m["speakers"]["S1"]["ref_alts"]]
    assert (10.0, 14.5) not in alts
    assert alts and set(alts) <= {(5.0, 9.5), (15.0, 19.5)}


# --------------------------------------- the words are only half of the verdict


def test_the_right_words_in_the_wrong_voice_are_not_ok():
    """0.889 overlap and a male clone of a female line. Both are true at once."""
    assert tts.clip_is_good({"ok": True, "overlap": 0.889, "verified": True})
    assert not tts.clip_is_good({"ok": True, "overlap": 0.889, "verified": True,
                                 "voice": 0.066, "voice_ok": False})
    # A check that did not run is not a failure: no embeddings, or a verdict
    # cached before this existed, still reads exactly as it did.
    assert tts.clip_is_good({"ok": True, "overlap": 0.889, "verified": True,
                             "voice_ok": True})


def test_a_wrong_voice_clip_is_recorded_and_reported_as_such(tmp_path, monkeypatch):
    """It is still a dub, and the record says whose voice it is in.

    The segment's own reference cannot be synthesised here, so the only takes are
    ladder takes and the clone drifts off the (correct) reference into somebody
    else. Rare by construction, which is why it has to be legible: the verdict is
    "wrong_voice", not "ok", and report.json counts and names it (report.run,
    pinned in tests/test_honest_failures.py).
    """
    eng, fake = engine(tmp_path, monkeypatch, [0.90, 0.90])
    real_generate = fake.generate

    def generate(speak, ref, out, **kw):
        if ref.name == "auto.wav":
            raise RuntimeError("no usable audio in this segment's own window")
        return real_generate(speak, ref, out, **kw)

    fake.generate = generate
    # Every finished clip comes back as speaker "b"; the references stay "a".
    monkeypatch.setattr(tts, "_embed_wavfile",
                        lambda p: voice("b" if p.parent.name == "clips" else "a"))
    got = eng.clip_for(seg(), "hello world")
    # A dub, kept (never-silent), and named for what is wrong with it.
    assert got["verify"] == "wrong_voice" and got["voice"] == 0.0
    assert got["overlap"] == 0.9


# ------------------------------------------- a truncated take is not a quick one

LONG = "a long line of thirteen words the voice will not finish saying here"


def test_a_take_too_short_for_its_words_never_reaches_the_ranking(tmp_path,
                                                                  monkeypatch):
    """The fast bound yields to the verifier so a correct quick read survives; a
    take that is quick AND says half the line is the decode stopping early."""
    eng, _fake = engine(tmp_path, monkeypatch, [0.5, 0.5, 0.5, 0.5], dur=1.0)
    assert eng.clip_for(seg(), LONG) is None
    # The same overlaps at a plausible length are accepted exactly as before.
    eng, _fake = engine(tmp_path, monkeypatch, [0.5, 0.5, 0.5, 0.5], dur=4.0)
    assert eng.clip_for(seg(), LONG)["verify"] == "accepted"


def test_a_line_the_voice_truncates_is_said_one_sentence_at_a_time(tmp_path,
                                                                   monkeypatch):
    """Varying the reference is no answer to a stopped decode: it stops in the
    same place from every reference. Saying the line in two breaths is."""
    # Four rungs that all fall short, then a clean take of each sentence, then
    # the joined clip read whole.
    eng, fake = engine(tmp_path, monkeypatch,
                       [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0], dur=4.0)
    got = eng.clip_for(seg(), "First sentence here. Second sentence here.")
    assert got["verify"] == "ok" and got["overlap"] == 1.0
    assert got["clip"].startswith("clips/parts_")
    # Each sentence was synthesised on its own, from the same reference as rung 0.
    assert [c["ref"] for c in fake.calls][-2:] == ["auto.wav", "auto.wav"]


def test_a_one_sentence_line_is_never_split(tmp_path, monkeypatch):
    eng, fake = engine(tmp_path, monkeypatch, [0.5, 0.5, 0.5, 0.5], dur=4.0)
    got = eng.clip_for(seg(), LONG)
    assert got["verify"] == "accepted" and len(fake.calls) == 4


def test_a_split_that_reads_no_better_is_not_used(tmp_path, monkeypatch):
    eng, _fake = engine(tmp_path, monkeypatch,
                        [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 0.4], dur=4.0)
    got = eng.clip_for(seg(), "First sentence here. Second sentence here.")
    assert got["overlap"] == 0.5 and not got["clip"].startswith("clips/parts_")


# ------------------------------------------- a crashed decode is one roll of dice

def test_a_decode_that_raises_is_re_rolled_before_it_is_a_failure(tmp_path,
                                                                  monkeypatch):
    """One seed threw IndexError out of the checkpoint's embedding lookup and
    three others voiced the same sentence cleanly. Left as a failure it cost the
    whole segment, because the crashed rung is the one the caller was on."""
    eng, fake = engine(tmp_path, monkeypatch, [0.9], dur=4.0)
    real = fake.generate
    seen: list[int] = []

    def flaky(speak, ref, out, *, seed, greedy, **kw):
        seen.append(seed)
        if len(seen) == 1:
            raise IndexError("index out of range in self")
        return real(speak, ref, out, seed=seed, greedy=greedy, **kw)

    fake.generate = flaky
    assert eng.clip_for(seg(), "hello world")["verify"] == "ok"
    assert len(seen) == 2 and seen[0] != seen[1]


def test_a_decode_that_keeps_crashing_is_still_a_failure(tmp_path, monkeypatch):
    eng, fake = engine(tmp_path, monkeypatch, [], dur=4.0)

    def always(speak, ref, out, *, seed, greedy, **kw):
        raise IndexError("index out of range in self")

    fake.generate = always
    assert eng.clip_for(seg(), "hello world") is None


def test_a_greedy_re_roll_samples_because_the_seed_is_inert():
    assert tts.reseeds(7, greedy=True)[1][1] is False
    assert tts.reseeds(7, greedy=False)[1][0] != 7
    assert tts.reseeds(7, greedy=False)[0] == (7, False)


# ------------------------------------- one fault, said once, not once per attempt
#
# The log a user filed as an issue was four identical lines per segment and then
# the same four for the next segment: one broken import, printed until it filled
# the panel, with the fact that nothing at all could be synthesised nowhere in
# it. These pin the shape that replaced it. A regression here is invisible until
# somebody files the same issue again, so the assertions are on the literal text.

def always_raising(eng, fake, exc):
    def always(speak, ref, out, *, seed, greedy, **kw):
        raise exc

    fake.generate = always
    return eng


def test_a_ladder_that_raises_every_rung_says_it_once_with_the_count(
        tmp_path, monkeypatch, capsys):
    eng, fake = engine(tmp_path, monkeypatch, [], dur=4.0)
    always_raising(eng, fake, ModuleNotFoundError("No module named 'qwen_tts.core.models'"))

    assert eng.clip_for(seg(), "hello world") is None
    lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.strip()]

    assert len(lines) == 1, lines
    said = lines[0]
    # The attempt count is on the line, so four retries of one problem cannot be
    # mistaken for four problems.
    assert f"{len(eng.rungs(seg(), eng._plan(seg(), 'hello world'))) * tts.GENERATE_TRIES}" \
        " attempt(s)" in said
    # The original exception text survives verbatim, for the bug report.
    assert "ModuleNotFoundError: No module named 'qwen_tts.core.models'" in said
    # …and it is followed by something to do about it.
    assert "git submodule update --init third_party/Qwen3-TTS" in said
    # The per-attempt lines are gone.
    assert "generate failed" not in said and "decode crashed" not in said


def test_the_same_fault_on_the_next_line_is_counted_not_repeated(
        tmp_path, monkeypatch, capsys):
    eng, fake = engine(tmp_path, monkeypatch, [], dur=4.0)
    always_raising(eng, fake, ModuleNotFoundError("No module named 'qwen_tts.core.models'"))

    for seg_id in (5, 6, 7):
        assert eng.clip_for(seg(id=seg_id, uid=f"u{seg_id}"), "hello world") is None
    eng.flush_faults()
    lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.strip()]

    assert len(lines) == 2, lines
    assert lines[0].startswith("  tts: seg 5 could not be voiced.")
    assert "seg 6" not in lines[0] and "seg 7" not in lines[0]
    # The lines it also took down are named, once, at the end.
    assert lines[1] == ("  tts: the same fault took 2 more line(s), which keep their "
                        "original audio (ModuleNotFoundError: No module named "
                        "'qwen_tts.core.models'); segments 6, 7")


def test_one_line_taken_down_alone_gets_no_summary(tmp_path, monkeypatch, capsys):
    eng, fake = engine(tmp_path, monkeypatch, [], dur=4.0)
    always_raising(eng, fake, IndexError("index out of range in self"))
    eng.clip_for(seg(), "hello world")
    capsys.readouterr()

    eng.flush_faults()
    assert capsys.readouterr().err == ""


def test_two_different_faults_are_both_announced(tmp_path, monkeypatch, capsys):
    """Collapsing is per cause. Two exceptions are two problems, and hiding the
    second one behind the first is the failure this whole change is against."""
    eng, fake = engine(tmp_path, monkeypatch, [], dur=4.0)
    always_raising(eng, fake, ModuleNotFoundError("no qwen_tts"))
    eng.clip_for(seg(id=1, uid="u1"), "hello world")
    always_raising(eng, fake, MemoryError("out of memory"))
    eng.clip_for(seg(id=2, uid="u2"), "hello world")

    said = capsys.readouterr().err
    assert "seg 1 could not be voiced" in said and "no qwen_tts" in said
    assert "seg 2 could not be voiced" in said and "out of memory" in said


def test_a_synthesis_fault_reaches_report_json_not_only_the_scrollback(
        tmp_path, monkeypatch):
    """stderr scrolls past; `m["health"]` is what report.run prints as `degraded`.

    This is the other half of collapsing: what is said once has to be findable
    afterwards, and the run's own report is where a user looks.
    """
    eng, fake = engine(tmp_path, monkeypatch, [], dur=4.0)
    always_raising(eng, fake, ModuleNotFoundError("No module named 'qwen_tts'"))
    eng.clip_for(seg(), "hello world")

    assert eng.m["health"]["tts.generate"] == ("ModuleNotFoundError: No module "
                                               "named 'qwen_tts'")
    assert "tts.generate" in tts.HEALTH_KEYS      # so a clean re-run clears it


def test_a_line_that_failed_on_its_takes_still_speaks_for_itself(
        tmp_path, monkeypatch, capsys):
    """The fallback line is only suppressed where something else already said it.

    A segment whose takes were simply bad is a fact about that one line, so it
    keeps its own line in the log; only the segments an exception took down are
    covered by the fault summary.
    """
    eng, _fake = engine(tmp_path, monkeypatch, [0.0, 0.0, 0.0, 0.0], dur=4.0)
    segment = dict(seg(), text_en="hello world", keep=False, keep_reason=None)
    tts._retry_pass(eng, [segment])

    assert "  tts: seg 27 unusable → keep original" in capsys.readouterr().err
    assert segment["keep_reason"] == "tts_failed"


def test_the_remedy_for_a_missing_module_is_the_install_and_never_a_retry():
    """An instruction that is wrong costs more than no instruction: this project
    has already shipped a message telling a user to redo an install that had
    just succeeded. Nothing here tells anyone to try the same thing again."""
    missing = tts.explain_generate_failure("ModuleNotFoundError: No module named 'x'")
    assert "Reinstall the app" in missing and "uv sync" in missing
    assert "again" not in missing

    memory = tts.explain_generate_failure("MemoryError: out of memory")
    assert "Close other" in memory and "resumes rather than restarts" in memory

    # An unknown cause promises nothing it cannot deliver.
    plain = tts.explain_generate_failure("ValueError: something odd")
    assert "keeps its original audio" in plain and "The rest of the run continues." in plain


def test_repeats_never_announces_one_segment_twice_for_one_cause():
    runs = tts.Repeats()
    assert runs.first("boom", 5) is True
    assert runs.first("boom", 5) is False         # a later rung on the same line
    assert runs.first("boom", 6) is False         # a later line, counted
    assert runs.hit(6) and not runs.hit(9)
    assert runs.rest() == [("boom", [6])]
    assert runs.rest() == []                      # and it is only said once
