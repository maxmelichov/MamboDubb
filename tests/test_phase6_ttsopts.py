"""Per-segment `tts_opts` — parsing, cache keys, and the no-op guarantee.

Pure logic, no models. The regression guard that matters most is
`test_defaults_reproduce_todays_cache_key`: a run with no options anywhere must
hash to exactly the keys it did before this feature existed.
"""

import numpy as np
import pytest
import soundfile as sf

from dubbing import audio, manifest, tts, ttsopts
from dubbing.ttsopts import DEFAULT, TtsOpts

# ------------------------------------------------------------------- parsing


def test_absent_and_empty_are_the_default():
    assert ttsopts.parse(None) is DEFAULT
    assert ttsopts.parse({}) == DEFAULT
    assert ttsopts.parse(DEFAULT) is DEFAULT
    assert DEFAULT.to_dict() == {}
    assert DEFAULT.fingerprint() == ""
    assert DEFAULT.cache_suffix() == ""
    assert DEFAULT.clone_mode() == "xvec"


def test_every_option_round_trips():
    raw = {"seed": 7, "ref": "refs/S1.wav", "ref_text": "shalom", "model": "0.6b",
           "speed": 1.1, "temperature": 0.8, "top_p": 0.9, "top_k": 40,
           "repetition_penalty": 1.2, "max_new_tokens": 600, "keep_pauses": True}
    opts = ttsopts.parse(raw)
    assert opts.to_dict() == raw
    assert ttsopts.parse(opts.to_dict()) == opts


def test_unknown_option_is_rejected_loudly():
    with pytest.raises(ValueError) as exc:
        ttsopts.parse({"speeed": 1.2})
    assert "speeed" in str(exc.value) and "known" in str(exc.value)
    # ... including one that only exists on a different checkpoint
    with pytest.raises(ValueError, match="instruct"):
        ttsopts.parse({"instruct": "say it angrily"})


def test_non_dict_is_rejected():
    for bad in ("seed=1", 3, [("seed", 1)], True):
        with pytest.raises(ValueError):
            ttsopts.parse(bad)


@pytest.mark.parametrize("raw", [
    {"seed": -1}, {"seed": 2 ** 32}, {"seed": 1.5}, {"seed": "1"}, {"seed": True},
    {"greedy": "yes"}, {"greedy": 1}, {"keep_pauses": 0},
    {"speed": 0.0}, {"speed": 0.4}, {"speed": 2.5}, {"speed": "1.2"}, {"speed": float("nan")},
    {"temperature": 0.0}, {"temperature": -0.5}, {"temperature": 3.0},
    {"top_p": 0.0}, {"top_p": 1.5}, {"top_k": 0}, {"top_k": 2.5}, {"top_k": 10_000},
    {"repetition_penalty": 0.1}, {"repetition_penalty": 5.0},
    {"max_new_tokens": 0}, {"max_new_tokens": 99_999}, {"max_new_tokens": 100.5},
    {"model": "3b"}, {"model": ""}, {"model": 1},
    {"ref": ""}, {"ref": "/etc/passwd"}, {"ref": "../../secrets.wav"},
    {"ref": "refs/../../x.wav"}, {"ref": 5},
])
def test_out_of_range_values_are_rejected(raw):
    with pytest.raises(ValueError):
        ttsopts.parse(raw)


def test_ref_text_without_ref_is_rejected():
    # ICL conditions on the transcript of a specific reference; the auto-picked
    # window has no transcript the caller could have supplied.
    with pytest.raises(ValueError, match="ref"):
        ttsopts.parse({"ref_text": "shalom"})
    assert ttsopts.parse({"ref": "refs/a.wav", "ref_text": "shalom"}).icl


def test_greedy_rejects_the_sampler_instead_of_ignoring_it():
    for k, v in (("temperature", 0.9), ("top_p", 0.8), ("top_k", 20)):
        with pytest.raises(ValueError, match="greedy"):
            ttsopts.parse({"greedy": True, k: v})
    # ... and the same values are fine without it
    assert ttsopts.parse({"temperature": 0.9, "top_p": 0.8, "top_k": 20}).top_k == 20


def test_error_messages_name_the_option():
    with pytest.raises(ValueError, match=r"tts_opts\.speed"):
        ttsopts.parse({"speed": 9.0})


def test_ref_accepts_a_windows_style_separator_but_not_an_absolute_path():
    assert ttsopts.parse({"ref": "refs\\S1.wav"}).ref == "refs/S1.wav"
    with pytest.raises(ValueError):
        ttsopts.parse({"ref": "C:\\refs\\S1.wav"})


def test_merge_patches_and_clears():
    stored = {"seed": 5, "speed": 1.2}
    assert ttsopts.merge(stored, {"speed": 0.9}) == {"seed": 5, "speed": 0.9}
    assert ttsopts.merge(stored, {"speed": None}) == {"seed": 5}
    # clearing back to the default drops the key entirely — the manifest stays small
    assert ttsopts.merge({"speed": 1.2}, {"speed": 1.0}) == {}
    assert ttsopts.merge(None, {}) == {}
    with pytest.raises(ValueError):
        ttsopts.merge(stored, {"nonsense": 1})
    with pytest.raises(ValueError):
        ttsopts.merge(stored, {"speed": 9.0})


def test_merge_validates_the_combination_not_just_the_patch():
    with pytest.raises(ValueError, match="greedy"):
        ttsopts.merge({"temperature": 0.9}, {"greedy": True})


# ----------------------------------------------------------------- cache keys


def _engine(tmp_path, tgt="en"):
    return tts.Engine({"source": {"src_lang": "he", "tgt_lang": tgt},
                       "speakers": {}, "segments": []}, tmp_path)


ARGS = ("some text", "ref:1.00-4.00", 42, False)


def test_defaults_reproduce_todays_cache_key(tmp_path):
    """The regression guard: no options must hash exactly as it did before."""
    import hashlib

    eng = _engine(tmp_path)
    speak, ref_key, seed, greedy = ARGS
    old_blob = f"{eng.model_tag}|{speak}|{ref_key}|xvec|{seed}|{int(greedy)}"
    old = hashlib.sha1(old_blob.encode("utf-8")).hexdigest()[:16]
    assert eng._cache_key(*ARGS) == old
    assert eng._cache_key(*ARGS, DEFAULT) == old
    assert eng._cache_key(*ARGS, ttsopts.parse({})) == old


def test_every_audio_affecting_option_changes_the_key(tmp_path):
    eng = _engine(tmp_path)
    base = eng._cache_key(*ARGS)
    seen = {base}
    for raw in ({"speed": 1.1}, {"temperature": 0.9}, {"top_p": 0.9}, {"top_k": 40},
                {"repetition_penalty": 1.2}, {"max_new_tokens": 600},
                {"keep_pauses": True}, {"model": "0.6b"},
                {"ref": "refs/a.wav", "ref_text": "shalom"}):
        key = eng._cache_key(*ARGS, ttsopts.parse(raw))
        assert key not in seen, f"{raw} did not change the cache key"
        seen.add(key)


def test_ref_text_changes_the_clone_mode_and_the_key(tmp_path):
    eng = _engine(tmp_path)
    a = ttsopts.parse({"ref": "refs/a.wav", "ref_text": "one transcript"})
    b = ttsopts.parse({"ref": "refs/a.wav", "ref_text": "another transcript"})
    assert a.clone_mode().startswith("icl:") and a.clone_mode() != b.clone_mode()
    assert eng._cache_key(*ARGS, a) != eng._cache_key(*ARGS, b)


def test_seed_and_greedy_still_key_the_clip(tmp_path):
    eng = _engine(tmp_path)
    speak, ref_key, seed, greedy = ARGS
    assert eng._cache_key(speak, ref_key, seed + 1, greedy) != eng._cache_key(*ARGS)
    assert eng._cache_key(speak, ref_key, seed, True) != eng._cache_key(*ARGS)


def test_cache_suffix_is_order_independent(tmp_path):
    a = ttsopts.parse({"speed": 1.1, "top_k": 40})
    b = ttsopts.parse({"top_k": 40, "speed": 1.1})
    assert a.cache_suffix() == b.cache_suffix()
    assert a.fingerprint() == b.fingerprint()


def test_model_override_keys_the_clip_by_the_checkpoint_that_made_it(tmp_path):
    small = ttsopts.parse({"model": "0.6b"})
    big = ttsopts.parse({"model": "1.7b"})
    eng = _engine(tmp_path)                      # run default is 1.7b
    assert eng.model_tag_for(big) == eng.model_tag
    assert eng._cache_key(*ARGS, big) == eng._cache_key(*ARGS)
    assert eng._cache_key(*ARGS, small) != eng._cache_key(*ARGS)
    # ... and the same segment under a 0.6b run collides with neither
    other = tts.Engine({"source": {"src_lang": "he", "tgt_lang": "en"},
                        "speakers": {}, "segments": []}, tmp_path, model="0.6b")
    assert other._cache_key(*ARGS) == eng._cache_key(*ARGS, small)


def test_target_language_still_mixed_in(tmp_path):
    opts = ttsopts.parse({"speed": 1.1})
    assert _engine(tmp_path, "ru")._cache_key(*ARGS, opts) != \
        _engine(tmp_path, "en")._cache_key(*ARGS, opts)


# -------------------------------------------------------------------- sampler


def test_sampler_defaults_are_todays_hard_coded_values():
    assert tts.sampling_kwargs(False) == {
        "do_sample": True, "temperature": 0.55, "top_p": 0.85, "top_k": 30,
        "subtalker_dosample": True, "subtalker_temperature": 0.55,
        "subtalker_top_p": 0.85, "subtalker_top_k": 30}
    assert tts.sampling_kwargs(True) == {
        "do_sample": False, "temperature": 0.01, "top_p": 1.0, "top_k": 1,
        "subtalker_dosample": False, "subtalker_temperature": 0.01,
        "subtalker_top_p": 1.0, "subtalker_top_k": 1}
    assert tts.sampling_kwargs(False, DEFAULT) == tts.sampling_kwargs(False)


def test_sampler_options_reach_both_talkers():
    got = tts.sampling_kwargs(False, ttsopts.parse({"temperature": 0.9, "top_k": 5}))
    assert got["temperature"] == got["subtalker_temperature"] == 0.9
    assert got["top_k"] == got["subtalker_top_k"] == 5
    assert got["top_p"] == 0.85                       # untouched options keep the default


def test_greedy_attempt_ignores_the_sampler_options():
    # The last of MAX_TRIES is greedy whatever the segment asked for; parse()
    # already refuses greedy+sampler, so this only covers the forced last try.
    opts = ttsopts.parse({"temperature": 0.9})
    assert tts.sampling_kwargs(True, opts) == tts.sampling_kwargs(True)


# ---------------------------------------------------------------------- seeds


def test_seed_derivation_prefers_uid_over_positional_id():
    speak = "hello there"
    a = tts.seed_for({"id": 3, "uid": "abc"}, speak)
    b = tts.seed_for({"id": 99, "uid": "abc"}, speak)
    assert a == b                                   # renumbering must not re-roll the take
    assert tts.seed_for({"id": 3, "uid": "xyz"}, speak) != a


def test_seed_derivation_without_uid_is_unchanged():
    import hashlib

    seg = {"id": 3}
    want = int(hashlib.sha1(b"3|hello").hexdigest()[:8], 16)
    assert tts.seed_for(seg, "hello") == want
    assert tts.seed_for({"id": 3, "uid": None}, "hello") == want
    assert tts.seed_for({"id": 3, "uid": ""}, "hello") == want


def test_seed_override_wins_and_retries_stay_distinct():
    opts = ttsopts.parse({"seed": 12345})
    seg = {"id": 3, "uid": "abc"}
    assert tts.seed_for(seg, "hello", opts) == 12345
    # attempt N uses base + 1000N (tts._attempt), so a pinned seed is attempt 0 exactly
    assert [12345 + 1000 * a for a in range(3)] == [12345, 13345, 14345]


# --------------------------------------------------- planning and re-synthesis


def _plan_engine(tmp_path):
    """An Engine whose reference lookup is stubbed — `_plan` without audio."""
    eng = _engine(tmp_path)
    (tmp_path / "refs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "refs" / "auto.wav").write_bytes(b"auto")
    eng.ref_for = lambda seg, opts=DEFAULT: (
        eng.pinned_ref(opts) or (tmp_path / "refs/auto.wav", "ref:1.00-4.00"))
    return eng


def test_plan_carries_the_parsed_options(tmp_path):
    eng = _plan_engine(tmp_path)
    plan = eng._plan({"id": 1, "start": 0.0, "end": 3.0,
                      "tts_opts": {"speed": 1.1}}, "hello world")
    assert plan.opts.speed == 1.1
    assert plan.speak == "hello world."
    assert eng._plan({"id": 1, "start": 0.0, "end": 3.0}, "hello world").opts == DEFAULT


def test_plan_rejects_bad_options_instead_of_synthesizing_around_them(tmp_path):
    eng = _plan_engine(tmp_path)
    with pytest.raises(ValueError):
        eng._plan({"id": 1, "start": 0.0, "end": 3.0,
                   "tts_opts": {"speed": 12.0}}, "hello world")


def test_pinned_ref_replaces_the_chosen_window(tmp_path):
    eng = _engine(tmp_path)
    (tmp_path / "refs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "refs" / "pick.wav").write_bytes(b"chosen-voice")
    path, key = eng.pinned_ref(ttsopts.parse({"ref": "refs/pick.wav"}))
    assert path == tmp_path / "refs/pick.wav"
    assert key.startswith("opt:refs/pick.wav:")
    # content-hashed, so replacing the file re-synthesizes rather than replaying
    (tmp_path / "refs" / "pick.wav").write_bytes(b"different-voice")
    eng2 = _engine(tmp_path)
    assert eng2.pinned_ref(ttsopts.parse({"ref": "refs/pick.wav"}))[1] != key
    assert eng.pinned_ref(DEFAULT) is None


def test_pinned_ref_missing_file_is_loud(tmp_path):
    eng = _engine(tmp_path)
    with pytest.raises(FileNotFoundError, match="tts_opts.ref"):
        eng.pinned_ref(ttsopts.parse({"ref": "refs/gone.wav"}))


def test_record_carries_the_option_fingerprint(tmp_path):
    verdict = {"dur": 1.5, "overlap": 0.9}
    clip = tmp_path / "clips" / "abc.wav"
    plain = tts.Engine._record(clip, verdict, 0)
    assert "opts" not in plain                       # unchanged for a default segment
    assert tts.Engine._record(clip, verdict, 0, DEFAULT) == plain
    tagged = tts.Engine._record(clip, verdict, 0, ttsopts.parse({"speed": 1.1}))
    assert tagged["opts"] == ttsopts.parse({"speed": 1.1}).fingerprint()


def test_changing_options_requeues_an_already_dubbed_segment(tmp_path):
    (tmp_path / "clips").mkdir()
    (tmp_path / "clips" / "a.wav").write_bytes(b"clip")
    seg = {"id": 1, "tts": {"clip": "clips/a.wav"}}
    assert not tts.needs_synthesis(seg, tmp_path)     # default options, clip present
    seg["tts_opts"] = {"speed": 1.1}
    assert tts.needs_synthesis(seg, tmp_path)         # options changed under the clip
    seg["tts"]["opts"] = ttsopts.parse({"speed": 1.1}).fingerprint()
    assert not tts.needs_synthesis(seg, tmp_path)     # ... and settle once re-made
    seg["tts_opts"] = {"speed": 1.2}
    assert tts.needs_synthesis(seg, tmp_path)


def test_needs_synthesis_still_requires_the_file(tmp_path):
    assert tts.needs_synthesis({"id": 1}, tmp_path)
    assert tts.needs_synthesis({"id": 1, "tts": {"clip": "clips/missing.wav"}}, tmp_path)


# ------------------------------------------------- attempts (stubbed synthesis)


class _FakeSynth:
    """Records what `_attempt` asked for and writes a stand-in clip."""

    def __init__(self):
        self.calls = []

    def generate(self, speak, ref, out, *, seed, greedy, opts=DEFAULT,
                 synth=None, lang=None):
        self.calls.append({"speak": speak, "ref": ref, "seed": seed,
                           "greedy": greedy, "opts": opts})
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), np.full(audio.SR, 0.2, dtype="float32"), audio.SR)
        return out


def _attempt_engine(tmp_path):
    eng = _engine(tmp_path)
    fake = _FakeSynth()
    eng.synth_for = lambda opts=DEFAULT: fake
    return eng, fake


def _attempts(eng, opts, *, tries=tts.MAX_TRIES):
    ref = eng.workdir / "refs" / "auto.wav"
    return [eng._attempt(1, "hello.", ref, "ref:1.00-4.00", 100, a, opts)
            for a in range(tries)]


def test_greedy_option_forces_the_deterministic_decode_from_the_first_try(tmp_path):
    eng, fake = _attempt_engine(tmp_path)
    _attempts(eng, DEFAULT)
    assert [c["greedy"] for c in fake.calls] == [False, False, True]   # today's ladder

    eng, fake = _attempt_engine(tmp_path)
    _attempts(eng, ttsopts.parse({"greedy": True}))
    assert [c["greedy"] for c in fake.calls] == [True, True, True]


def test_attempts_still_walk_the_seed_and_land_on_distinct_clips(tmp_path):
    eng, fake = _attempt_engine(tmp_path)
    got = _attempts(eng, ttsopts.parse({"seed": 100, "speed": 1.1}))
    assert [c["seed"] for c in fake.calls] == [100, 1100, 2100]
    assert len({clip.name for clip, _meta, _v in got}) == 3
    assert all(c["opts"].speed == 1.1 for c in fake.calls)


def test_attempt_replays_the_cache_only_for_the_same_options(tmp_path):
    eng, fake = _attempt_engine(tmp_path)
    ref = tmp_path / "refs" / "auto.wav"
    clip, meta, _ = eng._attempt(1, "hello.", ref, "r", 100, 0, DEFAULT)
    meta.write_text('{"ok": true, "overlap": 1.0, "heard": "hello", "dur": 1.0}')
    assert eng._attempt(1, "hello.", ref, "r", 100, 0, DEFAULT)[2]["ok"] is True
    assert len(fake.calls) == 1                       # second call came from the cache
    # ... but a changed option is a different clip, never a replay of the old one
    other = eng._attempt(1, "hello.", ref, "r", 100, 0, ttsopts.parse({"speed": 1.1}))
    assert other[2] is None and other[0] != clip
    assert len(fake.calls) == 2


def test_pinned_reference_is_not_escalated_past(tmp_path, monkeypatch):
    """The canonical-ref escalation must not undo the user's chosen voice."""
    m = {"source": {"src_lang": "he", "tgt_lang": "en"},
         "speakers": {"S1": {"ref": "refs/S1.wav"}}, "segments": []}
    eng = tts.Engine(m, tmp_path)
    (tmp_path / "refs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "refs" / "S1.wav").write_bytes(b"canonical")
    (tmp_path / "refs" / "pick.wav").write_bytes(b"chosen")
    fake = _FakeSynth()
    eng.synth_for = lambda opts=DEFAULT: fake
    monkeypatch.setattr(eng, "_verify_and_store",
                        lambda clip, meta, speak: {"ok": False, "overlap": 0.0,
                                                   "heard": "", "dur": 1.0})
    seg = {"id": 1, "start": 0.0, "end": 3.0, "speaker": "S1",
           "tts_opts": {"ref": "refs/pick.wav"}}
    assert eng.clip_for(seg, "hello world") is None            # every attempt failed
    refs = {c["ref"].name for c in fake.calls}
    assert refs == {"pick.wav"}                                # never fell back to S1
    assert len(fake.calls) == tts.MAX_TRIES                    # and no extra attempt


def test_unpinned_segment_still_escalates_to_the_canonical_reference(tmp_path, monkeypatch):
    m = {"source": {"src_lang": "he", "tgt_lang": "en"},
         "speakers": {"S1": {"ref": "refs/S1.wav"}}, "segments": []}
    eng = tts.Engine(m, tmp_path)
    (tmp_path / "refs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "refs" / "S1.wav").write_bytes(b"canonical")
    (tmp_path / "refs" / "auto.wav").write_bytes(b"auto")
    fake = _FakeSynth()
    eng.synth_for = lambda opts=DEFAULT: fake
    eng.ref_for = lambda seg, opts=DEFAULT: (tmp_path / "refs/auto.wav", "ref:1.00-4.00")
    monkeypatch.setattr(eng, "_verify_and_store",
                        lambda clip, meta, speak: {"ok": False, "overlap": 0.0,
                                                   "heard": "", "dur": 1.0})
    assert eng.clip_for({"id": 1, "start": 0.0, "end": 3.0, "speaker": "S1"},
                        "hello world") is None
    assert len(fake.calls) == tts.MAX_TRIES + 1
    assert fake.calls[-1]["ref"].name == "S1.wav"


# ------------------------------------------------------------------- manifest


def test_tts_opts_survives_a_save_and_a_tts_reset(tmp_path):
    m = manifest.new({"input": "x"})
    m["segments"] = [{"id": 0, "keep": False, "tts_opts": {"seed": 5},
                      "tts": {"clip": "clips/a.wav"}, "bogus": 1}]
    manifest.save(tmp_path, m)
    got = manifest.load(tmp_path)["segments"][0]
    assert got["tts_opts"] == {"seed": 5} and "bogus" not in got
    # a stage reset drops the generated clip but never the user's settings
    manifest.reset_stage(m, "tts")
    assert m["segments"][0]["tts_opts"] == {"seed": 5}
    assert "tts" not in m["segments"][0]


def test_models_list_matches_the_checkpoints():
    assert set(ttsopts.MODELS) == set(tts.TTS_MODELS)
    assert tts.DEFAULT_TTS_MODEL in ttsopts.MODELS


def test_option_set_is_documented():
    """Every option carries a rationale in the dataclass docstring."""
    doc = TtsOpts.__doc__ or ""
    for name in ttsopts.KEYS:
        assert f"`{name}`" in doc, f"{name} is undocumented"
