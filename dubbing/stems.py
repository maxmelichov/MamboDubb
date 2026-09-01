"""Stage 2 Demucs stem separation.

The vocals stem is used ONLY as a source of voice-clone reference audio and as
diarization input. It is never treated as an oracle for "is someone speaking
here" Demucs routes speech into the music stem often enough that doing so is
what produced the old pipeline's dead-air holes.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import nvlibs

MODEL = "htdemucs_ft"


def run(m: dict[str, Any], workdir: Path) -> None:
    source = workdir / m["files"]["source_wav"]
    stems = workdir / "stems"
    stems.mkdir(parents=True, exist_ok=True)
    scratch = workdir / "_demucs"

    # The device is named because this is the stage where getting it wrong is
    # expensive enough to be worth a line of log: the same separation is minutes
    # on a GPU and most of a day on a CPU, and the run that made this comment
    # necessary spent sixteen hours here without printing anything a reader
    # could have used to tell which one was happening.
    device = nvlibs.torch_device()
    print(f"  stems: demucs {MODEL} on {source.name} ({device}, slow, cached)",
          file=sys.stderr)
    nvlibs.warn_if_gpu_unused()
    subprocess.run(
        [sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", MODEL,
         "-o", str(scratch), str(source)],
        check=True, env=nvlibs.subprocess_env(),
    )

    produced = scratch / MODEL / source.stem
    for src_name, dst_name in (("vocals.wav", "vocals.wav"), ("no_vocals.wav", "background.wav")):
        src = produced / src_name
        if not src.is_file():
            raise FileNotFoundError(f"demucs did not produce {src}")
        shutil.move(str(src), str(stems / dst_name))
    shutil.rmtree(scratch, ignore_errors=True)

    m["files"]["vocals"] = "stems/vocals.wav"
    m["files"]["background"] = "stems/background.wav"
