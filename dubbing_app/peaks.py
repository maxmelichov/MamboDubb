"""Waveform peaks for the timeline lanes.

The editor draws the run's audio as audio — two lanes of amplitude, source over
output — and a lane needs a few hundred numbers, not forty megabytes of PCM. So
the downsampling happens here, next to the files, and the wire carries the
summary.

The reader is deliberately cheap: it seeks to each bucket and inspects a small
window of frames instead of decoding the whole file. A waveform overview is a
picture, not a measurement — sampling ~256 frames out of every bucket draws the
same picture as an exact per-bucket max, and it keeps the endpoint instant on an
hour-long run. Only mono-ized 16-bit PCM is understood, which is what every
stage of the pipeline writes; anything else is a 4xx, not a guess.
"""

from __future__ import annotations

import array
import wave
from pathlib import Path

from . import errors

# The run files a lane may ask for, by their manifest-stable names. The editor's
# SOURCE lane is the original audio; OUTPUT is the finished mix.
FILES = {"source": "source.wav", "dub": "dub.wav"}

MAX_BUCKETS = 4000
# Frames inspected per bucket. Enough that a short transient inside the bucket
# still registers, small enough that the whole scan touches <1% of a long file.
PROBE_FRAMES = 256


def read_peaks(path: Path, buckets: int) -> dict:
    """`buckets` normalized peak values in [0, 1], plus the audio duration."""
    buckets = max(16, min(int(buckets), MAX_BUCKETS))
    try:
        reader = wave.open(str(path), "rb")
    except (wave.Error, EOFError) as exc:
        raise errors.invalid(f"not a readable wav: {path.name}: {exc}") from exc
    with reader:
        nch = reader.getnchannels()
        width = reader.getsampwidth()
        rate = reader.getframerate()
        nframes = reader.getnframes()
        if width != 2:
            raise errors.invalid(f"{path.name} is not 16-bit PCM")
        if nframes <= 0 or rate <= 0:
            return {"duration": 0.0, "peaks": [0.0] * buckets}

        per_bucket = max(1, nframes // buckets)
        probe = min(PROBE_FRAMES, per_bucket)
        peaks: list[float] = []
        for i in range(buckets):
            start = min(i * per_bucket, nframes - 1)
            reader.setpos(start)
            samples = array.array("h")
            samples.frombytes(reader.readframes(probe))
            if nch > 1:
                samples = samples[::nch]  # channel 0 is picture enough
            top = max((abs(s) for s in samples), default=0)
            peaks.append(round(top / 32768, 4))
    return {"duration": nframes / rate, "peaks": peaks}


def for_project(workdir: Path, file: str, buckets: int) -> dict:
    wav_name = FILES.get(file)
    if wav_name is None:
        raise errors.invalid(f"unknown peaks file {file!r}; one of {sorted(FILES)}")
    path = workdir / wav_name
    if not path.is_file():
        raise errors.not_found(f"{wav_name} does not exist yet for this run")
    out = read_peaks(path, buckets)
    out["file"] = file
    return out
