#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Stage the bundle payload for the MamboDubb desktop shell.

The bundle — .dmg on macOS, .exe/.msi on Windows, .deb/.AppImage on Linux — ships two
things beyond the web UI so a fresh machine never needs a terminal:

1. **The pipeline source** — everything `uv sync` needs to build the venv, copied to
   `app/desktop/src-tauri/payload/workspace/`, which `tauri.conf.json` bundles as a
   resource (`Contents/Resources/workspace` in the .app, `resources/workspace` next to
   the .exe, `/usr/lib/MamboDubb/workspace` for a .deb). On first launch the shell
   copies it somewhere writable (`~/Library/Application Support/MamboDubb/workspace`,
   `%LOCALAPPDATA%\\MamboDubb\\workspace`, `~/.local/share/MamboDubb/workspace` — the
   installed bundle is read-only and `uv sync` has to write a `.venv`) and runs the
   server out of that.
2. **`uv` itself** — as a Tauri sidecar (`externalBin`) at
   `binaries/uv-<target-triple>[.exe]`. Copied from a locally installed uv when the
   target is this machine (the fast path every local rebuild takes), otherwise
   downloaded from the official astral-sh/uv release for that triple and checksum
   verified. No package manager needed on the user's machine.

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
    uv run --script scripts/stage_desktop_payload.py --target x86_64-pc-windows-msvc
    uv run --script scripts/stage_desktop_payload.py --download-uv --uv-version 0.12.5
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


# Where the official uv builds live. The asset names *are* Rust target triples, which
# is the happy accident that makes this a one-line mapping to what Tauri wants.
UV_RELEASES = "https://github.com/astral-sh/uv/releases"

# The triples we know how to bundle for: every one that has both an official uv asset
# and a Tauri bundle target. Value is the archive extension of the uv asset.
SUPPORTED_TRIPLES = {
    "aarch64-apple-darwin": ".tar.gz",
    "x86_64-apple-darwin": ".tar.gz",
    "x86_64-unknown-linux-gnu": ".tar.gz",
    "aarch64-unknown-linux-gnu": ".tar.gz",
    "x86_64-pc-windows-msvc": ".zip",
    "aarch64-pc-windows-msvc": ".zip",
}


def host_triple() -> str:
    """The Rust triple of the machine running this script."""
    machine = platform.machine().lower()
    arm = machine in ("arm64", "aarch64")
    if sys.platform == "darwin":
        return "aarch64-apple-darwin" if arm else "x86_64-apple-darwin"
    if sys.platform.startswith("linux"):
        return f"{'aarch64' if arm else 'x86_64'}-unknown-linux-gnu"
    if sys.platform == "win32":
        return f"{'aarch64' if arm else 'x86_64'}-pc-windows-msvc"
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
        Path.home() / ".local" / "bin" / "uv.exe",
    ):
        if candidate.is_file():
            return candidate
    return None


def _download(url: str) -> bytes:
    import urllib.error
    import urllib.request

    try:
        # The URL is always an https://github.com literal built from a triple this
        # script validated against SUPPORTED_TRIPLES; nothing here is user-controlled.
        with urllib.request.urlopen(url, timeout=180) as response:
            return response.read()
    except urllib.error.HTTPError as err:
        raise SystemExit(f"error: {url} -> HTTP {err.code} {err.reason}") from err
    except OSError as err:
        raise SystemExit(f"error: could not fetch {url}: {err}") from err


def download_uv(triple: str, version: str, dest: Path) -> None:
    """Fetch the official uv build for `triple` and unpack its binary to `dest`.

    Astral publishes one archive per target under a stable name, plus a `.sha256`
    sibling, so this needs no API call and no token — and the checksum is verified,
    because an unverified binary is about to be signed into someone's installer.
    """
    import hashlib
    import io

    suffix = SUPPORTED_TRIPLES[triple]
    asset = f"uv-{triple}{suffix}"
    base = f"{UV_RELEASES}/latest/download" if version == "latest" \
        else f"{UV_RELEASES}/download/{version}"
    print(f"downloading {base}/{asset}", flush=True)
    blob = _download(f"{base}/{asset}")

    expected = _download(f"{base}/{asset}.sha256").decode("utf-8", "replace").split()
    digest = hashlib.sha256(blob).hexdigest()
    if not expected or expected[0].lower() != digest:
        raise SystemExit(
            f"error: checksum mismatch for {asset}: got {digest}, "
            f"expected {expected[0] if expected else '<empty>'}"
        )

    wanted = "uv.exe" if triple.endswith("windows-msvc") else "uv"
    if suffix == ".zip":
        import zipfile

        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            member = next(
                (n for n in archive.namelist() if n.rsplit("/", 1)[-1] == wanted), None
            )
            if member is None:
                raise SystemExit(f"error: no {wanted} inside {asset}")
            payload = archive.read(member)
    else:
        import tarfile

        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
            member = next(
                (m for m in archive.getmembers()
                 if m.isfile() and m.name.rsplit("/", 1)[-1] == wanted),
                None,
            )
            if member is None:
                raise SystemExit(f"error: no {wanted} inside {asset}")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise SystemExit(f"error: could not read {wanted} out of {asset}")
            payload = extracted.read()

    dest.write_bytes(payload)


def stage_uv(triple: str, version: str, force_download: bool) -> None:
    """Put `uv-<triple>[.exe]` in `binaries/`, which is what `externalBin` resolves.

    Two paths, and the fast one exists because it is the inner loop of every local
    rebuild: when the sidecar's target *is* this machine, a locally installed uv is
    already the right binary, so copy it and skip the network entirely. For any other
    target — and for a release build that wants a pinned, checksummed uv rather than
    whatever brew last upgraded to — download the official build for that triple.
    """
    if triple not in SUPPORTED_TRIPLES:
        raise SystemExit(
            f"error: no uv sidecar known for target {triple}.\n"
            f"Known targets: {', '.join(sorted(SUPPORTED_TRIPLES))}"
        )
    exe = ".exe" if triple.endswith("windows-msvc") else ""
    dest = BINARIES / f"uv-{triple}{exe}"
    BINARIES.mkdir(parents=True, exist_ok=True)
    (BINARIES / ".gitignore").write_text("*\n", encoding="utf-8")

    uv = None if force_download or triple != host_triple() else find_local_uv()
    if uv is not None:
        shutil.copy2(uv, dest)
        source = str(uv)
    else:
        download_uv(triple, version, dest)
        source = f"astral-sh/uv {version} ({triple})"
    dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"staged uv sidecar: {source} -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--skip-missing-submodule",
        action="store_true",
        help="stage even if third_party/Qwen3-TTS is empty (CI without submodules; "
        "the resulting bundle cannot uv sync)",
    )
    parser.add_argument(
        "--target",
        default=None,
        metavar="TRIPLE",
        help="Rust target triple the bundle is for (default: this machine). "
        f"One of: {', '.join(sorted(SUPPORTED_TRIPLES))}",
    )
    parser.add_argument(
        "--download-uv",
        action="store_true",
        help="always fetch the official uv build instead of copying a local one",
    )
    parser.add_argument(
        "--uv-version",
        default="latest",
        metavar="VERSION",
        help="uv release tag to download, e.g. 0.12.5 (default: latest)",
    )
    args = parser.parse_args(argv)
    stage_workspace(args.skip_missing_submodule)
    stage_uv(args.target or host_triple(), args.uv_version, args.download_uv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
