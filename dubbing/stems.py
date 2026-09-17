"""Stage 2, stem separation: Demucs by default, MelBand RoFormer on request.

The vocals stem is used ONLY as a source of voice-clone reference audio and as
diarization input. It is never treated as an oracle for "is someone speaking
here": Demucs routes speech into the music stem often enough that doing so is
what produced the old pipeline's dead-air holes.

`--separator roformer` swaps in Kim Jensen's MelBand RoFormer vocal model
(MIT), which recovers speech under music far better than htdemucs_ft: on a
controlled speech+score mixture the SI-SDR gap was +3.1 dB at light-music
levels and +4.3 dB at loud-score levels, and the gap widens as the music gets
louder, which is exactly the movie case. Demucs stays the default because the
choice is in the stems fingerprint: flipping it re-separates, re-transcribes
and re-diarizes every existing run.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import nvlibs

MODEL = "htdemucs_ft"

SEPARATORS = ("demucs", "roformer")
# Kim Jensen's vocal checkpoint: the strongest vocal RoFormer with a clean
# (MIT) license; the higher-numbered community checkpoints carry unknown terms.
ROFORMER_MODEL = "vocals_mel_band_roformer.ckpt"
REPO_ROOT = Path(__file__).resolve().parents[1]
ROFORMER_DIR = REPO_ROOT / "models" / "audio-separator"


def run(m: dict[str, Any], workdir: Path, separator: str = "demucs") -> None:
    source = workdir / m["files"]["source_wav"]
    stems = workdir / "stems"
    stems.mkdir(parents=True, exist_ok=True)

    if separator == "roformer":
        _run_roformer(source, stems)
    else:
        _run_demucs(source, stems, workdir)

    m["files"]["vocals"] = "stems/vocals.wav"
    m["files"]["background"] = "stems/background.wav"


def _run_demucs(source: Path, stems: Path, workdir: Path) -> None:
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


def _run_roformer(source: Path, stems: Path) -> None:
    # In-process rather than a subprocess: audio_separator has no stable CLI
    # module entry point, and the surrounding stages already hold torch in this
    # process. The checkpoint is fetched into models/ on first use, next to the
    # pipeline's other weights.
    from audio_separator.separator import Separator

    device = nvlibs.torch_device()
    print(f"  stems: roformer {ROFORMER_MODEL} on {source.name} ({device}, cached)",
          file=sys.stderr)
    nvlibs.warn_if_gpu_unused()
    scratch = stems / "_roformer"
    scratch.mkdir(parents=True, exist_ok=True)
    sep = Separator(output_dir=str(scratch), model_file_dir=str(ROFORMER_DIR),
                    output_format="WAV")
    sep.load_model(model_filename=ROFORMER_MODEL)
    produced = [scratch / Path(name).name for name in sep.separate(str(source))]

    vocals = [p for p in produced if "(vocals)" in p.name.lower()]
    other = [p for p in produced if "(vocals)" not in p.name.lower()]
    if len(vocals) != 1 or len(other) != 1:
        raise FileNotFoundError(
            f"roformer produced {[p.name for p in produced]}; expected one "
            "vocals stem and one background stem")
    shutil.move(str(vocals[0]), str(stems / "vocals.wav"))
    shutil.move(str(other[0]), str(stems / "background.wav"))
    shutil.rmtree(scratch, ignore_errors=True)
