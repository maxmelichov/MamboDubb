"""Nemotron-3 diarization worker: wav path in, JSON turns file out.

One shot per stems file, launched by `dubbing.segments._diarize_nemotron` as
`uv run --project diarizer python worker.py <vocals.wav> <out.json>`. The JSON
goes to a file rather than stdout because NeMo and its progress bars own both
standard streams; the parent reads the file and treats a missing one as
failure.

Output shape matches what `segments.diarize` returns for pyannote:
`[{"speaker", "start", "end"}, ...]`, so everything downstream (smoothing,
ECAPA refinement, word assignment) is shared.
"""

from __future__ import annotations

import json
import sys

MODEL = "nvidia/Nemotron-3-Diarization"


def main() -> None:
    wav, out = sys.argv[1], sys.argv[2]
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel

    model = SortformerEncLabelModel.from_pretrained(MODEL).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    results = model.diarize(audio=[wav], batch_size=1)
    turns = []
    for line in results[0]:
        start, end, speaker = line.split()
        turns.append({"speaker": speaker,
                      "start": round(float(start), 3),
                      "end": round(float(end), 3)})
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(turns, fh)
    print(f"diarizer: {len(turns)} turn(s), "
          f"{len({t['speaker'] for t in turns})} speaker(s)", file=sys.stderr)


if __name__ == "__main__":
    main()
