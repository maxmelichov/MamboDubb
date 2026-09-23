"""Nemotron-3 diarization worker: wav path in, JSON turns file out.

One shot per stems file, launched by `dubbing.segments._diarize_nemotron` as
`uv run --project diarizer python worker.py <vocals.wav> <out.json> [chunk_sec]`.
The JSON goes to a file rather than stdout because NeMo and its progress bars
own both standard streams; the parent reads the file and treats a missing one
as failure.

`chunk_sec > 0` diarizes the file in overlapping windows and resets the
model's speaker slots at every window. The model is hard-capped at 8 speakers
per pass; a film exceeds that across its runtime but almost never within one
scene, so windowing removes the cap where it binds. The price is that labels
mean nothing across windows (each window's `speaker_0` is somebody new), which
is exactly right for the hybrid diarizer: it reads only the change points and
discards the identities. Plain `--diarizer nemotron` runs unchunked, where
identity has to hold for the whole file.

Output shape matches what `segments.diarize` returns for pyannote:
`[{"speaker", "start", "end"}, ...]`, so everything downstream (smoothing,
ECAPA refinement, word assignment) is shared.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

MODEL = "nvidia/Nemotron-3-Diarization"
OVERLAP_SEC = 10.0


def _parse(lines: list[str], offset: float = 0.0) -> list[dict]:
    turns = []
    for line in lines:
        start, end, speaker = line.split()
        turns.append({"speaker": speaker,
                      "start": round(float(start) + offset, 3),
                      "end": round(float(end) + offset, 3)})
    return turns


def main() -> None:
    wav, out = sys.argv[1], sys.argv[2]
    chunk_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    import soundfile as sf
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel

    model = SortformerEncLabelModel.from_pretrained(MODEL).eval()
    if torch.cuda.is_available():
        model = model.cuda()

    if chunk_sec <= 0:
        turns = _parse(model.diarize(audio=[wav], batch_size=1)[0])
    else:
        samples, sr = sf.read(wav, dtype="float32", always_2d=True)
        mono = samples.mean(axis=1)
        total = len(mono) / sr
        step = chunk_sec - OVERLAP_SEC
        starts = [i * step for i in range(max(1, int(total // step) + 1))
                  if i * step < total - 0.5 or i == 0]
        turns = []
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, t0 in enumerate(starts):
                p = Path(tmp) / f"chunk{i}.wav"
                sf.write(str(p), mono[int(t0 * sr):int((t0 + chunk_sec) * sr)], sr)
                paths.append(str(p))
            results = model.diarize(audio=paths, batch_size=1)
            for i, (t0, lines) in enumerate(zip(starts, results)):
                # Each window owns the region up to the midpoint of its overlap
                # with the next; a turn belongs to the window holding its start.
                lo = 0.0 if i == 0 else t0 + OVERLAP_SEC / 2
                hi = (total if i == len(starts) - 1
                      else starts[i + 1] + OVERLAP_SEC / 2)
                for t in _parse(lines, offset=t0):
                    if lo <= t["start"] < hi:
                        t["speaker"] = f"w{i}_{t['speaker']}"
                        t["end"] = min(t["end"], hi + OVERLAP_SEC / 2)
                        turns.append(t)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(turns, fh)
    print(f"diarizer: {len(turns)} turn(s), "
          f"{len({t['speaker'] for t in turns})} label(s)", file=sys.stderr)


if __name__ == "__main__":
    main()
