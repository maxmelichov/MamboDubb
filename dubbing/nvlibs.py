"""Make the pip-bundled NVIDIA libraries win over mismatched system copies.

On Linux, torch's wheels carry their own cuDNN under site-packages/nvidia/ and
find it through RUNPATH — but RUNPATH does not propagate: when libcudnn.so.9
dlopens its sub-libraries (libcudnn_cnn, libcudnn_ops, …) they resolve through
ldconfig instead, and a system-installed cuDNN of a different version answers.
The mix fails inside the first convolution with
CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH.

Two fixes, one per process kind: `preload()` loads the wheel's cuDNN libraries
into this process by absolute path before torch first needs them (an soname
already loaded wins any later lookup), and `subprocess_env()` prepends the wheel
lib dirs to LD_LIBRARY_PATH for children (demucs runs as a subprocess). Both are
no-ops off Linux or when no nvidia wheels are installed.
"""

from __future__ import annotations

import ctypes
import os
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


def preload() -> None:
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
