"""The peaks endpoint waveform summaries for the editor's timeline lanes."""

from __future__ import annotations

import array
import sys
import wave
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from dubbing_app import peaks as peaks_mod  # noqa: E402
from dubbing_app.app import create_app  # noqa: E402
from tests.conftest_app import make_project  # noqa: E402

NAME = "whatsapp_0809"


def write_wav(path: Path, *, seconds: float = 1.0, rate: int = 8000,
              channels: int = 1, loud_tail: bool = True) -> None:
    """Half silence, half full-scale a waveform with a shape to assert on."""
    frames = int(seconds * rate)
    samples = array.array("h")
    for i in range(frames):
        value = 30000 if (loud_tail and i >= frames // 2) else 0
        samples.extend([value] * channels)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())


@pytest.fixture()
def outputs(tmp_path):
    root = tmp_path / "outputs"
    make_project(root, NAME)
    return root


@pytest.fixture()
def client(outputs):
    with TestClient(create_app(outputs, runner=None, ui_dir="")) as c:
        yield c


def test_peaks_summarize_the_wav_shape(client, outputs):
    write_wav(outputs / NAME / "source.wav")
    r = client.get(f"/api/projects/{NAME}/peaks", params={"file": "source", "n": 100})
    assert r.status_code == 200
    body = r.json()
    assert body["file"] == "source"
    assert body["duration"] == pytest.approx(1.0, abs=0.01)
    got = body["peaks"]
    assert len(got) == 100
    # the quiet half is quiet, the loud half is loud
    assert max(got[:45]) < 0.05
    assert min(got[55:]) > 0.8


def test_peaks_downmix_stereo_and_clamp_buckets(client, outputs):
    write_wav(outputs / NAME / "dub.wav", channels=2)
    r = client.get(f"/api/projects/{NAME}/peaks",
                   params={"file": "dub", "n": 999_999})
    assert r.status_code == 200
    assert len(r.json()["peaks"]) == peaks_mod.MAX_BUCKETS


def test_peaks_name_their_errors(client, outputs):
    # a lane the API does not know
    r = client.get(f"/api/projects/{NAME}/peaks", params={"file": "stems"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_request"
    # a legal lane whose file has not been made yet
    (outputs / NAME / "dub.wav").unlink()
    r = client.get(f"/api/projects/{NAME}/peaks", params={"file": "dub"})
    assert r.status_code == 404
    assert "dub.wav" in r.json()["error"]["message"]
    # a file that exists but is not a wav
    (outputs / NAME / "source.wav").write_bytes(b"not audio")
    r = client.get(f"/api/projects/{NAME}/peaks", params={"file": "source"})
    assert r.status_code == 400
