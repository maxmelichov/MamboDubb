"""Clone-reference selection: minimum duration, concatenation, voice validation.

Pure logic plus Engine-level tests on synthetic vocals; ECAPA is monkeypatched,
no models load.
"""

import copy

import numpy as np
import soundfile as sf

from dubbing import tts
from dubbing.tts import MIN_REF_SEC, REF_SR, REF_TARGET_SEC


# ------------------------------------------------------------ choose_ref_windows


def test_choose_prefers_single_long_window():
    cands = [(10.0, 11.5, 0.01, 0.1),      # short
             (20.0, 23.5, 0.05, 0.1),      # long, noisier
             (30.0, 33.0, 0.01, 0.1)]      # long, cleanest
    assert tts.choose_ref_windows(cands) == [(30.0, 33.0)]


def test_choose_concatenates_short_windows_to_target():
    cands = [(40.0, 41.5, 0.03, 0.1), (10.0, 11.5, 0.01, 0.1),
             (20.0, 21.5, 0.02, 0.1), (30.0, 31.5, 0.04, 0.1)]
    spans = tts.choose_ref_windows(cands)
    total = sum(b - a for a, b in spans)
    assert total >= MIN_REF_SEC
    assert total <= REF_TARGET_SEC + 2.0          # stops once the target is reached
    assert spans == sorted(spans)                  # time order
    assert len(spans) > 1


def test_choose_single_short_window_falls_back_gracefully():
    assert tts.choose_ref_windows([(10.0, 10.9, 0.01, 0.1)]) == [(10.0, 10.9)]
    assert tts.choose_ref_windows([]) == []


# ------------------------------------------------------------------- concat_ref


def _tone(freq: float, sec: float, amp: float = 0.1) -> np.ndarray:
    t = np.arange(int(sec * REF_SR)) / REF_SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _vocals(voice_spans, total_sec=60.0):
    """Silence with `(start, end, freq)` tone spans dropped in."""
    voc = np.zeros(int(total_sec * REF_SR), dtype=np.float32)
    for a, b, freq in voice_spans:
        voc[int(a * REF_SR): int(a * REF_SR) + int((b - a) * REF_SR)] = _tone(freq, b - a)
    return voc


def test_concat_ref_joins_in_time_order_with_fades():
    voc = _vocals([(10.0, 11.5, 220), (20.0, 21.5, 220)])
    out = tts.concat_ref(voc, [(20.0, 21.5), (10.0, 11.5)])
    assert abs(len(out) / REF_SR - 3.0) < 0.01
    # Fades pull the joins to (near) zero.
    assert abs(out[0]) < 1e-4 and abs(out[-1]) < 1e-4
    # Earlier span first regardless of argument order (past the 480-sample fade).
    first = _tone(220, 1.5)
    assert np.allclose(out[600:1200], first[600:1200], atol=1e-5)


def test_concat_ref_deterministic():
    voc = _vocals([(10.0, 11.5, 220), (20.0, 21.5, 330)])
    a = tts.concat_ref(voc, [(10.0, 11.5), (20.0, 21.5)])
    b = tts.concat_ref(voc, [(10.0, 11.5), (20.0, 21.5)])
    assert np.array_equal(a, b)


# ---------------------------------------------------------- reject_voice_outliers


def _unit(*xs):
    v = np.asarray(xs, dtype=float)
    return v / np.linalg.norm(v)


def test_outlier_rejected_when_others_cohere():
    vecs = np.stack([_unit(1, 0), _unit(1, 0.1), _unit(1, -0.1), _unit(0, 1)])
    assert tts.reject_voice_outliers(vecs) == [True, True, True, False]


def test_incoherent_speaker_keeps_everything():
    # Mutually dissimilar windows: no consensus voice, nothing to reject against.
    vecs = np.eye(4)
    assert tts.reject_voice_outliers(vecs) == [True] * 4


def test_too_few_windows_keeps_everything():
    vecs = np.stack([_unit(1, 0), _unit(0, 1)])
    assert tts.reject_voice_outliers(vecs) == [True, True]


# ---------------------------------------------------- build_speaker_refs (Engine)


def _engine(tmp_path, voc, segs):
    m = {"source": {"src_lang": "he", "tgt_lang": "en"},
         "files": {"vocals": "stems/vocals.wav"},
         "speakers": {}, "segments": segs}
    eng = tts.Engine(m, tmp_path)
    eng._voc = voc
    return eng


def _seg(i, a, b, spk="S0"):
    return {"id": i, "start": a, "end": b, "speaker": spk, "keep": False}


def test_short_line_speaker_gets_concatenated_ref(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "_embed_windows", lambda *a, **k: None)  # no model
    spans = [(10.0, 11.5), (20.0, 21.5), (30.0, 31.5), (40.0, 41.5)]
    voc = _vocals([(a, b, 220) for a, b in spans])
    eng = _engine(tmp_path, voc, [_seg(i, a, b) for i, (a, b) in enumerate(spans)])
    eng.build_speaker_refs()
    info = eng.m["speakers"]["S0"]
    assert info["ref"]
    wav, sr = sf.read(str(tmp_path / info["ref"]))
    assert len(wav) / sr >= MIN_REF_SEC
    assert len(info["ref_windows"]) > 1


def test_outlier_window_excluded_from_ref(tmp_path, monkeypatch):
    spans = [(10.0, 11.5), (20.0, 21.5), (30.0, 31.5), (40.0, 41.5)]
    voc = _vocals([(a, b, 220) for a, b in spans])

    def fake_embed(vocals, wanted):
        # The 30.0 window is another voice; the rest agree.
        return np.stack([_unit(0, 1) if a == 30.0 else _unit(1, 0) for a, b in wanted])

    monkeypatch.setattr(tts, "_embed_windows", fake_embed)
    eng = _engine(tmp_path, voc, [_seg(i, a, b) for i, (a, b) in enumerate(spans)])
    eng.build_speaker_refs()
    info = eng.m["speakers"]["S0"]
    assert info["ref"]
    assert [30.0, 31.5] not in info["ref_windows"]
    assert len(info["ref_windows"]) == 3          # the other three, concatenated


def test_single_short_window_speaker_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "_embed_windows", lambda *a, **k: None)
    voc = _vocals([(10.0, 11.2, 220)])
    eng = _engine(tmp_path, voc, [_seg(0, 10.0, 11.2)])
    eng.build_speaker_refs()
    info = eng.m["speakers"]["S0"]
    assert info["ref"]                             # short, but never None
    assert info["ref_sec"] < MIN_REF_SEC


def test_build_refs_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "_embed_windows", lambda *a, **k: None)
    spans = [(10.0, 11.5), (20.0, 21.5), (30.0, 31.5)]
    voc = _vocals([(a, b, 220) for a, b in spans])
    segs = [_seg(i, a, b) for i, (a, b) in enumerate(spans)]
    outs = []
    for sub in ("a", "b"):
        eng = _engine(tmp_path / sub, voc, copy.deepcopy(segs))
        eng.build_speaker_refs()
        outs.append((tmp_path / sub / eng.m["speakers"]["S0"]["ref"]).read_bytes())
    assert outs[0] == outs[1]


def test_old_manifest_ref_is_rebuilt(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "_embed_windows", lambda *a, **k: None)
    spans = [(10.0, 11.5), (20.0, 21.5), (30.0, 31.5)]
    voc = _vocals([(a, b, 220) for a, b in spans])
    eng = _engine(tmp_path, voc, [_seg(i, a, b) for i, (a, b) in enumerate(spans)])
    # A pre-REF_BUILD manifest entry: ref exists on disk but has no version marker.
    (tmp_path / "refs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "refs" / "S0.wav").write_bytes(b"stale")
    eng.m["speakers"]["S0"] = {"ref": "refs/S0.wav", "ref_span": [345.12, 345.84]}
    eng.build_speaker_refs()
    info = eng.m["speakers"]["S0"]
    assert info["ref_v"] == tts.Engine.REF_BUILD
    assert info["ref_sec"] >= MIN_REF_SEC          # rebuilt, not the stale 0.72s ref


# ------------------------------------------------------------------ ref_for


def test_ref_for_long_segment_uses_own_window(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "_embed_windows", lambda *a, **k: None)
    voc = _vocals([(10.0, 14.0, 220)])
    eng = _engine(tmp_path, voc, [])
    path, key = eng.ref_for(_seg(0, 10.0, 14.0))
    assert key.startswith("ref:")
    assert path.is_file()


def test_ref_for_a_one_second_line_takes_the_canonical_unasked(tmp_path, monkeypatch):
    """A second of speech carries no usable speaker embedding, so nothing is asked.

    The gate answers "not the same voice" to every candidate at that length, its
    own speaker's canonical included, and the fallback is the sub-MIN_REF_SEC
    window this branch exists to avoid. On a Hebrew drama that cost two lines
    outright: every take cloned from about a second of audio came back truncated
    and was rejected on length, and the original Hebrew aired instead.
    """
    # Embeddings that would say "different voice" if anyone asked them.
    monkeypatch.setattr(tts, "_embed_windows", lambda *a, **k: np.stack([_unit(1, 0)]))
    monkeypatch.setattr(tts, "_embed_wavfile", lambda *a, **k: _unit(0, 1))
    voc = _vocals([(10.0, 11.0, 220)])
    eng = _engine(tmp_path, voc, [])
    eng.m["speakers"]["S0"] = {"ref": "refs/S0.wav"}
    (tmp_path / "refs").mkdir(parents=True, exist_ok=True)
    sf.write(str(tmp_path / "refs" / "S0.wav"), _tone(220, 3.0), REF_SR)
    _path, key = eng.ref_for(_seg(0, 10.0, 11.0))
    assert key.startswith("S0:canonical:")


def test_ref_for_a_one_second_line_with_no_canonical_uses_its_own_window(
        tmp_path, monkeypatch):
    """Nothing to swap to is still nothing to swap to: the short window stands."""
    monkeypatch.setattr(tts, "_embed_windows", lambda *a, **k: None)
    monkeypatch.setattr(tts, "_embed_wavfile", lambda *a, **k: None)
    voc = _vocals([(10.0, 11.0, 220)])
    eng = _engine(tmp_path, voc, [])
    _path, key = eng.ref_for(_seg(0, 10.0, 11.0))
    assert key.startswith("ref:")


def test_ref_for_short_segment_no_embeddings_keeps_own_window(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "_embed_windows", lambda *a, **k: None)
    monkeypatch.setattr(tts, "_embed_wavfile", lambda *a, **k: None)
    voc = _vocals([(10.0, 12.0, 220)])
    eng = _engine(tmp_path, voc, [])
    eng.m["speakers"]["S0"] = {"ref": "refs/S0.wav"}
    (tmp_path / "refs").mkdir(parents=True, exist_ok=True)
    sf.write(str(tmp_path / "refs" / "S0.wav"), _tone(220, 3.0), REF_SR)
    path, key = eng.ref_for(_seg(0, 10.0, 12.0))
    assert key.startswith("ref:")                  # current behaviour unchanged


def test_ref_for_short_segment_matching_voice_uses_canonical(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "_embed_windows", lambda *a, **k: np.stack([_unit(1, 0)]))
    monkeypatch.setattr(tts, "_embed_wavfile", lambda *a, **k: _unit(1, 0.1))
    voc = _vocals([(10.0, 12.0, 220)])
    eng = _engine(tmp_path, voc, [])
    eng.m["speakers"]["S0"] = {"ref": "refs/S0.wav"}
    (tmp_path / "refs").mkdir(parents=True, exist_ok=True)
    sf.write(str(tmp_path / "refs" / "S0.wav"), _tone(220, 3.0), REF_SR)
    path, key = eng.ref_for(_seg(0, 10.0, 12.0))
    assert key.startswith("S0:canonical:")


def test_ref_for_short_segment_mismatched_voice_keeps_own_window(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "_embed_windows", lambda *a, **k: np.stack([_unit(1, 0)]))
    monkeypatch.setattr(tts, "_embed_wavfile", lambda *a, **k: _unit(0, 1))
    voc = _vocals([(10.0, 12.0, 220)])
    eng = _engine(tmp_path, voc, [])
    eng.m["speakers"]["S0"] = {"ref": "refs/S0.wav"}
    (tmp_path / "refs").mkdir(parents=True, exist_ok=True)
    sf.write(str(tmp_path / "refs" / "S0.wav"), _tone(220, 3.0), REF_SR)
    path, key = eng.ref_for(_seg(0, 10.0, 12.0))
    assert key.startswith("ref:")                  # this line's own voice wins
