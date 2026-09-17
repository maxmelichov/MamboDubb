"""--separator: the stems backend is recorded, replayed, and fingerprint-safe.

The dangerous property is the last one: the stems fingerprint is a hash of its
params dict, so recording {"separator": "demucs"} for runs that never chose one
would flip every existing project's stems fingerprint and re-separate,
re-transcribe and re-diarize it for a no-op. The default must fingerprint as {}.
"""

from __future__ import annotations

from pathlib import Path

from dubbing import cli, stems


def _args(argv: list[str]):
    args = cli.parse_args(["input.mp4", *argv])
    args.src, args.tgt = "he", "en"
    return args


def test_the_default_fingerprints_exactly_like_before_the_flag_existed():
    args = _args([])
    cli.resolve_settings(args, {"source": {}})
    assert args.separator == "demucs"
    assert cli.stage_params(args, {"source": {}})["stems"] == {}


def test_roformer_lands_in_the_stems_fingerprint():
    args = _args(["--separator", "roformer"])
    cli.resolve_settings(args, {"source": {}})
    assert cli.stage_params(args, {"source": {}})["stems"] == {"separator": "roformer"}


def test_a_bare_rerun_keeps_the_separator_the_run_recorded():
    args = _args([])
    cli.resolve_settings(args, {"source": {"separator": "roformer"}})
    assert args.separator == "roformer"


def test_the_command_line_beats_the_recorded_separator():
    args = _args(["--separator", "demucs"])
    cli.resolve_settings(args, {"source": {"separator": "roformer"}})
    assert args.separator == "demucs"


def test_source_record_remembers_the_choice():
    args = _args(["--separator", "roformer"])
    cli.resolve_settings(args, {"source": {}})
    assert cli.source_record(args)["separator"] == "roformer"


def test_stems_run_dispatches_on_the_separator(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(stems, "_run_demucs",
                        lambda source, out, workdir: calls.append("demucs"))
    monkeypatch.setattr(stems, "_run_roformer",
                        lambda source, out: calls.append("roformer"))
    m = {"files": {"source_wav": "source.wav"}}
    (tmp_path / "source.wav").write_bytes(b"")
    stems.run(m, tmp_path)
    stems.run(m, tmp_path, separator="roformer")
    assert calls == ["demucs", "roformer"]
    assert m["files"]["vocals"] == "stems/vocals.wav"


def test_the_editor_rerun_command_carries_the_separator():
    from editor import jobs

    cmd = jobs.dub_command("in.mp4", Path("out"), opts={"separator": "roformer"})
    assert cmd[cmd.index("--separator") + 1] == "roformer"
