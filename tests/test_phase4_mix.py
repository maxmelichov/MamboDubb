"""Phase 4 unclaimed time carries the original vocals (invariant 1 at the mix).

A real run muted an on-screen speaker: an English interviewee inside a he→ru
video slipped past every detection layer, no segment was created, and the mix
played only the background stem over 10 seconds of a person visibly talking.
Per-detector fixes are whack-a-mole; the structural fix is here. Any span of
the output no placed clip claims now gets the matching slice of the vocals
stem at unity gain, so a detection failure degrades to "original audio plays"
instead of "speaker muted". No decision keys off the vocals stem it is used
as audio, unconditionally, in all unclaimed time so invariant 3 stands.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from dubbing import audio, mix

SR = audio.SR


# ---------------------------------------------------------------- pure helpers

def test_unclaimed_spans_complement():
    spans = mix.unclaimed_spans([(1.0, 2.0), (3.0, 4.0)], 5.0)
    assert spans == [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)]


def test_unclaimed_spans_merges_within_slop():
    # 30ms seam between two clips is not a hole.
    spans = mix.unclaimed_spans([(1.0, 2.0), (2.03, 3.0)], 4.0)
    assert spans == [(0.0, 1.0), (3.0, 4.0)]


def test_unclaimed_spans_drops_slivers():
    # A 100ms gap clears the slop but is below the minimum span length.
    assert mix.unclaimed_spans([(0.0, 1.9), (2.0, 4.0)], 4.0) == []


def test_unclaimed_spans_no_placements_is_whole_timeline():
    assert mix.unclaimed_spans([], 3.0) == [(0.0, 3.0)]


def test_voiced_spans_finds_the_voiced_stretch():
    sr = 16000
    a = np.zeros(3 * sr, dtype=np.float32)
    a[sr : 2 * sr] = 0.1 * np.sin(np.linspace(0, 440 * 2 * np.pi, sr))
    spans = mix.voiced_spans(a, sr)
    assert len(spans) == 1
    t0, t1 = spans[0]
    assert t0 == pytest.approx(1.0, abs=0.15)
    assert t1 == pytest.approx(2.0, abs=0.15)
    assert mix.voiced_spans(np.zeros(3 * sr, dtype=np.float32), sr) == []


# ------------------------------------------------------------------- envelope

def test_envelope_ducks_under_voiced_fill_only():
    env = mix.build_envelope([], 10.0, [(2.0, 4.0)])
    assert env[int(3.0 * mix.CTRL_HZ)] == pytest.approx(mix.DUCK_DUB, abs=0.02)
    assert env[int(1.0 * mix.CTRL_HZ)] == pytest.approx(1.0, abs=0.02)
    # Silent fill contributes no span, so the bed is untouched.
    silent = mix.build_envelope([], 10.0, [])
    assert silent[int(3.0 * mix.CTRL_HZ)] == pytest.approx(1.0, abs=0.02)


# ------------------------------------------------------------------- assembly

def _wav(path, a):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), a.astype(np.float32), SR, subtype="FLOAT")


def _seg(i, start, end, clip):
    return {"id": i, "keep": False, "place": {"start": start, "end": end, "clip": clip}}


def _assemble(tmp_path, vocals, *, segs=None, dur=6.0):
    """Run mix.assemble over a silent bed with two silent 1s clips.

    Silent clips and a silent bed make the output a microscope for the vocals
    fill: any nonzero sample in the result came from the fill pass.
    """
    if segs is None:
        for name in ("a.wav", "b.wav"):
            _wav(tmp_path / name, np.zeros(SR))
        segs = [_seg(0, 0.5, 1.5, "a.wav"), _seg(1, 3.5, 4.5, "b.wav")]
    _wav(tmp_path / "bg.wav", np.zeros(int(dur * SR)))
    _wav(tmp_path / "stems" / "vocals.wav", vocals)
    m = {"segments": segs, "source": {"duration": dur},
         "files": {"background": "bg.wav", "vocals": "stems/vocals.wav"}}
    raw = tmp_path / "raw.wav"
    mix.assemble(m, tmp_path, raw)
    out, _ = sf.read(str(raw), dtype="float32", always_2d=True)
    return out[:, 0]


def _tone(dur, amp=0.1):
    t = np.arange(int(dur * SR)) / SR
    return (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_unclaimed_gap_receives_vocals_at_unity(tmp_path):
    # The 78.5–89.1s failure in miniature: a speaker between two placed clips
    # whom no segment covers. The gap must carry the vocals, at unity gain.
    vocals = np.zeros(6 * SR, dtype=np.float32)
    vocals[int(1.5 * SR) : int(3.5 * SR)] = _tone(2.0)
    out = _assemble(tmp_path, vocals)
    got = out[int(2.0 * SR) : int(3.0 * SR)]
    assert audio.rms(got) == pytest.approx(0.1 / np.sqrt(2), rel=0.1)


def test_placed_clip_regions_are_not_touched_by_fill(tmp_path):
    # Vocals sound everywhere, but inside a placed (silent) clip the output
    # must stay exactly the clip: the fill never bleeds into claimed time.
    out = _assemble(tmp_path, _tone(6.0))
    inside = out[int(0.7 * SR) : int(1.3 * SR)]
    assert float(np.max(np.abs(inside))) < 1e-6
    gap = out[int(2.0 * SR) : int(3.0 * SR)]           # sanity: fill did run
    assert audio.rms(gap) > 0.05


def test_sliver_gap_below_threshold_is_skipped(tmp_path):
    for name in ("a.wav", "b.wav"):
        _wav(tmp_path / name, np.zeros(SR))
    segs = [_seg(0, 0.5, 1.5, "a.wav"), _seg(1, 1.6, 2.6, "b.wav")]
    out = _assemble(tmp_path, _tone(6.0), segs=segs)
    sliver = out[int(1.51 * SR) : int(1.59 * SR)]
    assert float(np.max(np.abs(sliver))) < 1e-6
    tail = out[int(3.0 * SR) : int(4.0 * SR)]          # sanity: real gaps filled
    assert audio.rms(tail) > 0.05


def test_fill_fades_at_span_edges(tmp_path):
    # Constant-level vocals expose the ramp: the first samples of the fill are
    # attenuated, half a fade in they sit midway, and mid-span is unity.
    vocals = np.full(6 * SR, 0.1, dtype=np.float32)
    out = _assemble(tmp_path, vocals)
    i0, i1 = int(1.5 * SR), int(3.5 * SR)              # fill span between clips
    assert abs(out[i0 + 2]) < 0.01                     # no hard step in
    assert abs(out[i1 - 2]) < 0.01                     # no hard step out
    half = int(mix.FILL_FADE_SEC / 2 * SR)
    assert out[i0 + half] == pytest.approx(0.05, abs=0.02)
    assert out[(i0 + i1) // 2] == pytest.approx(0.1, abs=0.005)


# ------------------------------------------------------------- mux tail padding

def test_tail_pad_holds_the_last_frame_for_overhanging_audio():
    assert mix.tail_pad(39.7, 38.1) == pytest.approx(1.6)


def test_tail_pad_is_zero_when_audio_fits():
    assert mix.tail_pad(38.0, 38.1) == 0.0
    assert mix.tail_pad(38.12, 38.1) == 0.0   # within the rounding floor
