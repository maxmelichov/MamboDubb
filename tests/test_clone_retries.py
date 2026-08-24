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
                 synth=None, lang=None):
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
           embeddings=True):
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
                "dur": 1.0, "verified": True}

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
        (workdir / "clips").mkdir(parents=True, exist_ok=True)

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
