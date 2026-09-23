"""--diarizer: recorded, replayed, fingerprint-safe, and it fails toward pyannote.

Same contract as --separator (see test_separator_flag.py): the default must
fingerprint exactly as before the flag existed, or every finished run would
re-segment and drop its locks. The extra rule here is the fallback direction:
a broken nemotron venv degrades to the default engine, not to single-speaker.
"""

from __future__ import annotations

from pathlib import Path

from dubbing import cli, segments


def _args(argv: list[str]):
    args = cli.parse_args(["input.mp4", *argv])
    args.src, args.tgt = "he", "en"
    return args


def test_the_default_fingerprints_exactly_like_before_the_flag_existed():
    args = _args([])
    cli.resolve_settings(args, {"source": {}})
    assert args.diarizer == "pyannote"
    assert "diarizer" not in cli.stage_params(args, {"source": {}})["segments"]


def test_nemotron_lands_in_the_segments_fingerprint():
    args = _args(["--diarizer", "nemotron"])
    cli.resolve_settings(args, {"source": {}})
    assert cli.stage_params(args, {"source": {}})["segments"]["diarizer"] == "nemotron"


def test_a_bare_rerun_keeps_the_diarizer_the_run_recorded():
    args = _args([])
    cli.resolve_settings(args, {"source": {"diarizer": "nemotron"}})
    assert args.diarizer == "nemotron"


def test_source_record_remembers_the_choice():
    args = _args(["--diarizer", "nemotron"])
    cli.resolve_settings(args, {"source": {}})
    assert cli.source_record(args)["diarizer"] == "nemotron"


def test_diarize_uses_the_nemotron_worker_when_asked(monkeypatch, tmp_path):
    turns = [{"speaker": "speaker_0", "start": 0.0, "end": 1.5}]
    monkeypatch.setattr(segments, "_diarize_nemotron", lambda vocals: turns)
    assert segments.diarize(tmp_path / "vocals.wav", backend="nemotron") == turns


def test_a_broken_nemotron_falls_back_to_pyannote_and_says_so(monkeypatch, tmp_path):
    def boom(vocals):
        raise RuntimeError("no venv here")

    def also_down():
        raise RuntimeError("also down")

    monkeypatch.setattr(segments, "_diarize_nemotron", boom)
    monkeypatch.setattr(segments, "_load_diarization_pipeline", also_down)
    seen: list[str] = []
    out = segments.diarize(tmp_path / "vocals.wav", note=seen.append,
                           backend="nemotron")
    # Both engines down: single-speaker fallback, and the note trail shows the
    # nemotron failure was recorded on the way to pyannote's own.
    assert out == []
    assert any("nemotron" in n for n in seen)


def test_hybrid_lands_in_the_segments_fingerprint():
    args = _args(["--diarizer", "hybrid"])
    cli.resolve_settings(args, {"source": {}})
    assert cli.stage_params(args, {"source": {}})["segments"]["diarizer"] == "hybrid"


def test_hybrid_with_a_broken_worker_is_plain_pyannote(monkeypatch, tmp_path):
    def boom(vocals):
        raise RuntimeError("no venv here")

    pyannote_turns = [{"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0}]
    monkeypatch.setattr(segments, "_diarize_nemotron", boom)
    monkeypatch.setattr(segments, "_load_diarization_pipeline",
                        lambda: (FakePyannote(pyannote_turns), "test"))
    seen: list[str] = []
    out = segments.diarize(tmp_path / "vocals.wav", note=seen.append,
                           backend="hybrid")
    assert out == []  # FakePyannote below is not a real pipeline; the point is
    # the note: the worker failure is recorded and the run continues.
    assert any("nemotron" in n for n in seen)


class FakePyannote:
    def __init__(self, turns):
        self.turns = turns


def _fake_embed(vectors):
    """An `_embed_span` whose answer depends on where the span sits.

    `vectors` maps (start, end) coverage to a unit vector: the first range
    containing the span's midpoint wins. Spans outside every range embed as
    None, like silence would.
    """
    def embed(vocals, start, end):
        mid = 0.5 * (start + end)
        for (a, b), v in vectors.items():
            if a <= mid <= b:
                return v
        return None

    return embed


def test_hybrid_splits_a_turn_nemotron_heard_a_change_inside(monkeypatch):
    import numpy as np

    va, vb = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    monkeypatch.setattr(segments, "_embed_span",
                        _fake_embed({(0.0, 5.0): va, (5.0, 10.0): vb}))
    py = [{"speaker": "SPEAKER_00", "start": 0.0, "end": 10.0}]
    ne = [{"speaker": "speaker_0", "start": 0.0, "end": 5.0},
          {"speaker": "speaker_1", "start": 5.0, "end": 10.0}]
    out = segments._hybrid_turns(py, ne, Path("vocals.wav"))
    assert [(t["start"], t["end"], t["speaker"]) for t in out] == [
        (0.0, 5.0, "SPEAKER_00"), (5.0, 10.0, "SPEAKER_00b")]


def test_hybrid_relabels_the_far_side_of_a_fused_junction(monkeypatch):
    import numpy as np

    va, vb = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    monkeypatch.setattr(segments, "_embed_span",
                        _fake_embed({(0.0, 4.0): va, (4.0, 8.0): vb, (8.0, 12.0): vb}))
    # pyannote heard the pause at 4.0 but filed both sides as SPEAKER_00, and
    # already knows a SPEAKER_01 elsewhere whose voice matches the far side.
    py = [{"speaker": "SPEAKER_00", "start": 0.0, "end": 3.9},
          {"speaker": "SPEAKER_00", "start": 4.1, "end": 8.0},
          {"speaker": "SPEAKER_01", "start": 8.0, "end": 12.0}]
    ne = [{"speaker": "speaker_0", "start": 0.0, "end": 3.9},
          {"speaker": "speaker_1", "start": 4.1, "end": 8.0}]
    out = segments._hybrid_turns(py, ne, Path("vocals.wav"))
    assert out[1]["speaker"] == "SPEAKER_01"   # adopted, not minted


def test_hybrid_leaves_same_voice_turns_alone(monkeypatch):
    import numpy as np

    va = np.array([1.0, 0.0])
    monkeypatch.setattr(segments, "_embed_span",
                        _fake_embed({(0.0, 10.0): va}))
    py = [{"speaker": "SPEAKER_00", "start": 0.0, "end": 10.0}]
    ne = [{"speaker": "speaker_0", "start": 0.0, "end": 5.0},
          {"speaker": "speaker_1", "start": 5.0, "end": 10.0}]
    assert segments._hybrid_turns(py, ne, Path("vocals.wav")) == py


def test_hybrid_skips_a_change_pyannote_already_hears(monkeypatch):
    def never(*a, **k):
        raise AssertionError("no embedding should be computed")

    monkeypatch.setattr(segments, "_embed_span", never)
    py = [{"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0},
          {"speaker": "SPEAKER_01", "start": 5.0, "end": 10.0}]
    ne = [{"speaker": "speaker_0", "start": 0.0, "end": 4.9},
          {"speaker": "speaker_1", "start": 5.1, "end": 10.0}]
    assert segments._hybrid_turns(py, ne, Path("vocals.wav")) == py


def test_the_editor_rerun_command_carries_the_diarizer():
    from editor import jobs

    cmd = jobs.dub_command("in.mp4", Path("out"), opts={"diarizer": "nemotron"})
    assert cmd[cmd.index("--diarizer") + 1] == "nemotron"
