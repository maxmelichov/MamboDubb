"""Everything about finding, and about *having*, the NVIDIA libraries.

Two different failures live here because they are the same failure wearing two
operating systems, and because a user hits them in the same minute.

**Linux: the wrong copy answers.** torch's wheels carry their own cuDNN under
site-packages/nvidia/ and find it through RUNPATH, but RUNPATH does not
propagate: when libcudnn.so.9 dlopens its sub-libraries (libcudnn_cnn,
libcudnn_ops, …) they resolve through ldconfig instead, and a system-installed
cuDNN of a different version answers. The mix fails inside the first convolution
with CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH.

**Windows: no copy answers.** Since Python 3.8 the loader does *not* search PATH
for the dependent DLLs of an extension module; it searches the directories
handed to `os.add_dll_directory` and the ones next to the extension itself. So a
machine with CUDA installed system wide, with `cublas64_12.dll` sitting in a
directory that is on PATH, still fails with "Library cublas64_12.dll is not
found or cannot be loaded" the moment CTranslate2 asks for it. Registering the
wheel directories is the fix, and it has to happen in-process before the first
load, which is why `preload()` runs from `dubbing/__main__.py` rather than from
whichever stage happens to need CUDA first.

**And the one nobody sees.** PyPI's default `torch` wheel is a CUDA build on
Linux and a CPU-only build on Windows. A Windows user with a perfectly good card
gets a pipeline where every torch stage silently runs on the CPU, which turns a
four-minute Demucs pass into sixteen hours. Nothing raises, so nothing is
logged, so the only symptom is that the run is slow. `warn_if_gpu_unused()` is
the loud line that failure never had, and it is called once at startup rather
than per stage so that it is the first thing in the log and not the fortieth.

Every function here is a no-op that returns something harmless when there is no
GPU, no wheel or no torch, so Macs and CPU-only boxes pay nothing but the import.
"""

from __future__ import annotations

import ctypes
import os
import re
import shutil
import sys
from pathlib import Path


def _lib_dirs() -> list[Path]:
    try:
        import nvidia
    except ImportError:
        return []
    dirs: list[Path] = []
    for pkg_dir in map(Path, nvidia.__path__):
        dirs.extend(sorted(p for p in pkg_dir.glob("*/lib") if p.is_dir()))
    return dirs


def _torch_dir() -> Path | None:
    """Where torch is installed, without importing it.

    `find_spec` on a top-level name locates the package and stops; it does not
    execute `torch/__init__.py`, which is the half-gigabyte this module refuses
    to spend on answering a question about a file.
    """
    try:
        from importlib.util import find_spec

        spec = find_spec("torch")
    except Exception:
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def windows_dll_dirs() -> list[Path]:
    """The directories a Windows CUDA load has to be told about, in load order.

    Three sources, and they are not redundant:

    * `nvidia/*/bin` is where the `nvidia-cublas-cu12` and `nvidia-cudnn-cu12`
      wheels put their DLLs. CTranslate2 does not bundle either one, so on
      Windows these wheels are the only reason `cublas64_12.dll` exists in the
      environment at all.
    * `nvidia/*/lib` because a few of the wheels spell it that way.
    * `torch/lib` because a CUDA `torch` wheel drops its own copies of the CUDA
      runtime there. torch registers that directory for itself when it is
      imported, but "when it is imported" is too late and too conditional to
      rely on for a library that is loaded by somebody else.

    Named and returned rather than registered inline so the Setup screen can ask
    the same question without a Windows machine being involved.
    """
    dirs: list[Path] = []
    try:
        import nvidia

        for pkg_dir in map(Path, nvidia.__path__):
            for sub in ("bin", "lib"):
                dirs.extend(sorted(p for p in pkg_dir.glob(f"*/{sub}") if p.is_dir()))
    except ImportError:
        pass
    torch_dir = _torch_dir()
    if torch_dir is not None and (torch_dir / "lib").is_dir():
        dirs.append(torch_dir / "lib")
    return dirs


def preload() -> None:
    """Make the wheels' CUDA libraries loadable. Safe to call more than once."""
    if sys.platform == "win32":
        for d in windows_dll_dirs():
            try:
                os.add_dll_directory(str(d))
            except OSError:
                # A directory that vanished between the glob and here, or a
                # path the loader will not take. Neither is worth ending a run
                # that may not need CUDA at all.
                pass
        return
    if sys.platform != "linux":
        return
    for lib_dir in _lib_dirs():
        if lib_dir.parent.name != "cudnn":
            continue
        # libcudnn.so.9 first, then the sub-libraries it would otherwise dlopen
        # by bare soname through ldconfig.
        for so in sorted(lib_dir.glob("libcudnn*.so.*"), key=lambda p: len(p.name)):
            try:
                ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


def subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    if sys.platform != "linux":
        return env
    dirs = [str(d) for d in _lib_dirs()]
    if dirs:
        current = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(dirs + ([current] if current else []))
    return env


# ---------------------------------------------------------------------------
# is there a GPU, and is this torch able to use it
# ---------------------------------------------------------------------------

def nvidia_smi() -> str | None:
    """The path to `nvidia-smi`, which is the cheap proof that a driver is here.

    A driver is not a card and a card is not a usable card, but the negative is
    what this is for: no `nvidia-smi` means there is nothing to be disappointed
    about, and every warning below stays quiet. `shutil.which` only, no
    subprocess: `dubbing_app.setup.gpu_memory_bytes` is the one that pays for a
    spawn, and it caches the result for exactly that reason.
    """
    return shutil.which("nvidia-smi")


def torch_cuda_build(torch_dir: Path | None = None) -> str | None:
    """The CUDA version this installed torch was built for, or None for CPU-only.

    Read out of `torch/version.py` as text rather than by importing torch. That
    file is generated at build time and holds a literal `cuda: str = '12.4'` (or
    `cuda: str | None = None` for the CPU wheel), so the answer is on disk and
    costs a file read instead of half a gigabyte of resident memory. That
    matters because the Setup screen asks this question on every poll, in a
    server process whose whole design rule is that it never imports torch.
    """
    torch_dir = torch_dir if torch_dir is not None else _torch_dir()
    if torch_dir is None:
        return None
    try:
        text = (torch_dir / "version.py").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = re.search(r"^cuda\s*(?::[^=]*)?=\s*['\"]([^'\"]+)['\"]", text, re.MULTILINE)
    return m.group(1) if m else None


def cuda_hint() -> str:
    """One sentence naming the fix for *this* platform, for a fallback message.

    A message that says what broke and not what to do is a message that sends
    somebody to a search engine. Empty on macOS, where CUDA is not a thing that
    was supposed to work and suggesting it would be a lie.
    """
    if sys.platform == "darwin":
        return ""
    if sys.platform == "win32":
        return "run `uv sync --extra app --extra cuda`; see docs/WINDOWS.md"
    # Linux gets a CUDA torch from PyPI by default, so the missing-DLL half of
    # this can only be CTranslate2's: torch's stack is CUDA 13 and CTranslate2
    # wants the CUDA 12 cuBLAS, which nothing installs on its own.
    return ("install the CUDA 12 libraries CTranslate2 needs with "
            "`uv pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`; "
            "see docs/CROSS_PLATFORM.md")


def gpu_unused_reason() -> str | None:
    """Why a machine with an NVIDIA card is about to run on the CPU, or None.

    None covers all three of the uninteresting cases at once: no card, no torch,
    or a torch that can use the card. Only the mismatch gets a sentence, because
    only the mismatch is a thing somebody would want to stop and fix.
    """
    if sys.platform == "darwin" or not nvidia_smi():
        return None
    torch_dir = _torch_dir()
    if torch_dir is None:
        return None
    if torch_cuda_build(torch_dir) is not None:
        return None
    return ("this machine has an NVIDIA driver but the installed torch is a "
            "CPU-only build, so every model runs on the CPU (stem separation "
            "alone goes from minutes to hours)")


_warned = False


def warn_if_gpu_unused(out=None, *, once: bool = True) -> str | None:
    """Print `gpu_unused_reason()` and return what was printed, or None.

    Called from `dubbing/__main__.py` before anything loads, and again from the
    stem stage, which is where the cost actually lands. The point of the first
    placement is the log: the user whose sixteen-hour run started this was told
    nothing until the ASR stage, hours in, and then only about the ASR. The
    point of `once` is that saying it twice in one process turns a warning into
    noise; `once=False` is for the tests, which want the string back rather than
    the suppression.
    """
    global _warned
    reason = gpu_unused_reason()
    if reason is None:
        return None
    if once:
        if _warned:
            return None
        _warned = True
    hint = cuda_hint()
    line = f"warning: {reason}. To fix it, {hint}." if hint else f"warning: {reason}."
    print(line, file=out if out is not None else sys.stderr)
    return line


def torch_device() -> str:
    """"cuda", "mps" or "cpu": the device torch will pick on this machine.

    Reported, never imposed. `dubbing.stems` uses it to *say* what the demucs
    child is about to do rather than to pass `-d`, because demucs makes the same
    choice from the same torch and a device this repo forces is a device this
    repo has to keep right on every platform forever. The value of knowing is
    the log line; the sixteen-hour run had no line to read.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
