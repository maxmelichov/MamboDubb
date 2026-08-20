#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Stage the bundle payload for the MamboDubb desktop shell.

The .app ships two things beyond the web UI so a fresh Mac never needs a terminal:

1. **The pipeline source** — everything `uv sync` needs to build the venv, copied to
   `app/desktop/src-tauri/payload/workspace/`, which `tauri.conf.json` bundles as a
   resource (`Contents/Resources/workspace` in the .app). On first launch the shell
   copies it to `~/Library/Application Support/MamboDubb/workspace` (the .app is
   read-only; `uv sync` writes `.venv` there) and runs the server out of that.
2. **`uv` itself** — as a Tauri sidecar (`externalBin`), copied from a locally
   installed uv to `binaries/uv-<target-triple>`. No Homebrew on the user's machine.

What is deliberately *not* staged: `models/` (~81 GB, downloaded on demand), `.venv`
(~1.7 GB, built on the user's machine by `uv sync`), `outputs/`, `tests/`, `app/`,
`docs/`. What *must* be staged beyond the obvious source trees: `README.md` and
`editor/`, because `[tool.uv] package = true` makes `uv sync` build the project wheel,
and the pyproject names both (`readme = "README.md"`, hatch `packages = [... "editor"]`).

Wired as part of app/desktop's `pretauri` hook, so it runs before every
`pnpm tauri dev` / `pnpm tauri build`. Run it by hand before a bare `cargo check` /
`cargo test` in src-tauri, because tauri-build resolves `externalBin` at compile time:

    uv run --script scripts/stage_desktop_payload.py
    uv run --script scripts/stage_desktop_payload.py --skip-missing-submodule
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_TAURI = ROOT / "app" / "desktop" / "src-tauri"
PAYLOAD = SRC_TAURI / "payload" / "workspace"
BINARIES = SRC_TAURI / "binaries"

# The workspace payload, as an allowlist: the pipeline source plus exactly the files
# `uv sync` reads. Everything else in the repo is dev tooling, docs, or gitignored bulk.
PAYLOAD_DIRS = ("dubbing", "dubbing_app", "editor", "translator", "third_party")
PAYLOAD_FILES = ("pyproject.toml", "uv.lock", ".python-version", ".env.example", "README.md")

# Never copied, wherever they appear: build products and user state that would either
# bloat the .app or leak this machine's environment into someone else's.
EXCLUDE_NAMES = {
    "__pycache__", ".git", ".venv", ".pytest_cache", ".DS_Store", ".mypy_cache",
    ".ruff_cache", "models", "outputs", "demucs_out", "node_modules",
}
EXCLUDE_SUFFIXES = (".egg-info", ".pyc")


def _ignore(directory: str, names: list[str]) -> set[str]:
    return {
        name for name in names
        if name in EXCLUDE_NAMES or name.endswith(EXCLUDE_SUFFIXES)
    }


def _writable(path: Path) -> None:
    """Owner-writable, recursively. The provisioning copy in Rust preserves modes, and
    a read-only file that came out of a read-only .app would break `uv sync`."""
    for child in [path, *path.rglob("*")]:
        mode = child.stat().st_mode
        child.chmod(mode | stat.S_IWUSR)


def stage_workspace(skip_missing_submodule: bool) -> int:
    qwen = ROOT / "third_party" / "Qwen3-TTS" / "pyproject.toml"
    if not qwen.is_file():
        message = (
            "third_party/Qwen3-TTS is not checked out; uv sync in the provisioned "
            "workspace would fail (pyproject pins qwen-tts to that path).\n"
            "Fix: git submodule update --init third_party/Qwen3-TTS"
        )
        if not skip_missing_submodule:
            raise SystemExit(f"error: {message}")
        print(f"warning: {message} (continuing: --skip-missing-submodule)")

    if PAYLOAD.exists():
        shutil.rmtree(PAYLOAD)
    PAYLOAD.mkdir(parents=True)
    # The staged tree is a build product; keep git blind to it.
    (PAYLOAD.parent / ".gitignore").write_text("*\n", encoding="utf-8")

    for name in PAYLOAD_DIRS:
        source = ROOT / name
        if not source.is_dir():
            raise SystemExit(f"error: expected directory missing from the repo: {source}")
        shutil.copytree(source, PAYLOAD / name, ignore=_ignore, symlinks=False)
    for name in PAYLOAD_FILES:
        source = ROOT / name
        if not source.is_file():
            raise SystemExit(f"error: expected file missing from the repo: {source}")
        shutil.copy2(source, PAYLOAD / name)

    _writable(PAYLOAD)
    total = sum(f.stat().st_size for f in PAYLOAD.rglob("*") if f.is_file())
    print(f"staged workspace payload: {PAYLOAD} ({total / 1e6:.1f} MB)")
    return total


def target_triple() -> str:
    """The Rust triple Tauri appends to `externalBin` names at bundle time."""
    machine = platform.machine().lower()
    if sys.platform == "darwin":
        return "aarch64-apple-darwin" if machine == "arm64" else "x86_64-apple-darwin"
    if sys.platform.startswith("linux"):
        arch = "aarch64" if machine in ("arm64", "aarch64") else "x86_64"
        return f"{arch}-unknown-linux-gnu"
    if sys.platform == "win32":
        return "x86_64-pc-windows-msvc"
    raise SystemExit(f"error: unsupported platform for staging: {sys.platform}")


def find_local_uv() -> Path | None:
    """The same chain the shell uses at runtime, minus the sidecar it is producing."""
    override = os.environ.get("DUBSTUDIO_UV_PATH")
    if override and Path(override).is_file():
        return Path(override)
    found = shutil.which("uv")
    if found:
        return Path(found)
    for candidate in (
        Path("/opt/homebrew/bin/uv"),
        Path("/usr/local/bin/uv"),
        Path.home() / ".local" / "bin" / "uv",
    ):
        if candidate.is_file():
            return candidate
    return None


def stage_uv() -> None:
    triple = target_triple()
    exe = ".exe" if sys.platform == "win32" else ""
    dest = BINARIES / f"uv-{triple}{exe}"
    uv = find_local_uv()
    if uv is None:
        raise SystemExit(
            "error: no local uv to bundle as the sidecar. Install one (brew install uv,\n"
            "or https://github.com/astral-sh/uv/releases — the archive for\n"
            f"{triple} contains the binary), or set DUBSTUDIO_UV_PATH, then rerun."
        )
    BINARIES.mkdir(parents=True, exist_ok=True)
    (BINARIES / ".gitignore").write_text("*\n", encoding="utf-8")
    shutil.copy2(uv, dest)
    dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"staged uv sidecar: {uv} -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--skip-missing-submodule",
        action="store_true",
        help="stage even if third_party/Qwen3-TTS is empty (CI without submodules; "
        "the resulting bundle cannot uv sync)",
    )
    args = parser.parse_args(argv)
    stage_workspace(args.skip_missing_submodule)
    stage_uv()
    return 0


if __name__ == "__main__":
    sys.exit(main())
