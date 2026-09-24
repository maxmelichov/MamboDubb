"""The compiled diarizer is preferred exactly when it exists, and nothing else moves.

rust/diarizer builds a drop-in for the NeMo worker: same argv, same JSON file.
The pipeline must pick it up the moment the binary appears (or DUB_DIARIZER_BIN
points at one) and fall back to the uv venv the moment it does not, without
changing the subprocess contract either way. A DUB_DIARIZER_BIN that names a
missing file must not crash the run; it degrades to the venv path like any
other absent binary, keeping _diarize_nemotron's error contract intact.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

from dubbing import segments


def _stub_binary(tmp_path: Path, turns: list[dict]) -> Path:
    """An executable that honors the worker argv: wav, out.json, chunk_sec."""
    stub = tmp_path / "mambodubb-diarizer"
    stub.write_text("#!/bin/sh\n"
                    f"printf '%s' '{json.dumps(turns)}' > \"$2\"\n"
                    "echo \"stub: $1 chunk=$3\" >&2\n")
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    return stub


def test_the_env_override_wins_when_it_names_a_real_file(tmp_path, monkeypatch):
    stub = _stub_binary(tmp_path, [])
    monkeypatch.setenv("DUB_DIARIZER_BIN", str(stub))
    assert segments._nemotron_binary() == stub


def test_a_missing_env_override_degrades_to_the_venv_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("DUB_DIARIZER_BIN", str(tmp_path / "gone"))
    assert segments._nemotron_binary() is None


def test_the_repo_build_is_found_without_any_env(tmp_path, monkeypatch):
    monkeypatch.delenv("DUB_DIARIZER_BIN", raising=False)
    built = tmp_path / "target" / "release" / "mambodubb-diarizer"
    built.parent.mkdir(parents=True)
    built.write_bytes(b"")
    monkeypatch.setattr(segments, "DIARIZER_RUST", tmp_path)
    assert segments._nemotron_binary() == built


def test_no_binary_anywhere_means_the_venv_worker(tmp_path, monkeypatch):
    monkeypatch.delenv("DUB_DIARIZER_BIN", raising=False)
    monkeypatch.setattr(segments, "DIARIZER_RUST", tmp_path / "unbuilt")
    assert segments._nemotron_binary() is None


def test_the_compiled_worker_gets_the_worker_argv_and_its_json_comes_back(
        tmp_path, monkeypatch):
    turns = [{"speaker": "w0_speaker_1", "start": 1.234, "end": 5.678}]
    stub = _stub_binary(tmp_path, turns)
    monkeypatch.setenv("DUB_DIARIZER_BIN", str(stub))
    assert segments._diarize_nemotron(tmp_path / "vocals.wav", chunk_sec=150.0) == turns


def test_a_failing_compiled_worker_keeps_the_error_contract(tmp_path, monkeypatch):
    stub = tmp_path / "mambodubb-diarizer"
    stub.write_text("#!/bin/sh\nexit 3\n")
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("DUB_DIARIZER_BIN", str(stub))
    try:
        segments._diarize_nemotron(tmp_path / "vocals.wav")
    except RuntimeError as exc:
        assert "nemotron worker exited 3" in str(exc)
    else:
        raise AssertionError("a failing worker must raise, not return")
