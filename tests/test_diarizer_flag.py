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


def test_the_editor_rerun_command_carries_the_diarizer():
    from editor import jobs

    cmd = jobs.dub_command("in.mp4", Path("out"), opts={"diarizer": "nemotron"})
    assert cmd[cmd.index("--diarizer") + 1] == "nemotron"
