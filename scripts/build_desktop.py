#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Build tooling for the MamboDubb desktop shell (after MamboRambo's pre_build.py).

The shell has no sidecar to compile the pipeline stays a Python checkout the app
drives with `uv` so all this does is get the web UI into place and drive Tauri.

    uv run --script scripts/build_desktop.py check     # toolchain + a dist to point Tauri at
    uv run --script scripts/build_desktop.py ui        # pnpm build in app/ui, copy dist here
    uv run --script scripts/build_desktop.py dev       # pnpm tauri dev (Vite on :1430)
    uv run --script scripts/build_desktop.py build     # pnpm tauri build  -> .app / .dmg
    uv run --script scripts/build_desktop.py build --debug

`check` is wired as app/desktop's `pretauri` hook, so it runs before every `pnpm tauri
…`; `ui` is the `build` script, which tauri.conf.json runs as `beforeBuildCommand`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UI_DIR = ROOT / "app" / "ui"
DESKTOP_DIR = ROOT / "app" / "desktop"
UI_DIST = UI_DIR / "dist"
DESKTOP_DIST = DESKTOP_DIR / "dist"

PLACEHOLDER = """<!doctype html>
<meta charset="utf-8">
<title>MamboDubb</title>
<style>
  body { font: 15px/1.6 system-ui, sans-serif; margin: 4rem auto; max-width: 34rem;
         color: #e7e7ea; background: #17171b; }
  code { background: #26262c; padding: .15rem .4rem; border-radius: .3rem; }
</style>
<h1>MamboDubb</h1>
<p>This is the placeholder shell frontend. The real UI is built from
  <code>app/ui</code>; run <code>uv run --script scripts/build_desktop.py ui</code> to put it
  here.</p>
"""


def run(cmd: list[str], cwd: Path, env: dict | None = None) -> None:
    print("+", " ".join(cmd), f"(in {cwd})", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


# Where a tool lives when it is installed but not on the PATH a GUI or a hook inherits.
# rustup's shims are the usual casualty; Homebrew's bin is the other.
FALLBACK_DIRS = (Path.home() / ".cargo" / "bin", Path("/opt/homebrew/bin"), Path("/usr/local/bin"))


def require(tool: str) -> str:
    """Locate a build tool, honouring the same env override the shell uses for uv."""
    override = os.environ.get(f"DUBSTUDIO_{tool.upper()}_PATH")
    if override and Path(override).is_file():
        return override
    found = shutil.which(tool)
    if found:
        return found
    for directory in FALLBACK_DIRS:
        candidate = directory / tool
        if candidate.is_file():
            return str(candidate)
    raise SystemExit(
        f"error: {tool} not found on PATH. Install it, or set "
        f"DUBSTUDIO_{tool.upper()}_PATH to its location."
    )


def write_placeholder() -> None:
    DESKTOP_DIST.mkdir(parents=True, exist_ok=True)
    (DESKTOP_DIST / "index.html").write_text(PLACEHOLDER, encoding="utf-8")
    print(f"wrote placeholder frontend to {DESKTOP_DIST}")


def copy_dist() -> bool:
    """Copy app/ui/dist into the desktop project. False if the UI has not been built."""
    if not (UI_DIST / "index.html").is_file():
        return False
    if DESKTOP_DIST.exists():
        shutil.rmtree(DESKTOP_DIST)
    shutil.copytree(UI_DIST, DESKTOP_DIST)
    print(f"copied {UI_DIST} -> {DESKTOP_DIST}")
    return True


def cmd_ui(args: argparse.Namespace) -> int:
    pnpm = require("pnpm")
    if not UI_DIR.is_dir():
        print(f"warning: {UI_DIR} is missing; using the placeholder frontend")
        write_placeholder()
        return 0
    if not (UI_DIR / "node_modules").is_dir():
        run([pnpm, "install"], cwd=UI_DIR)
    run([pnpm, "build"], cwd=UI_DIR)
    if not copy_dist():
        raise SystemExit(f"error: {UI_DIR} built but produced no {UI_DIST}/index.html")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Everything `pnpm tauri …` needs before it starts, and nothing slow."""
    require("pnpm")
    require("cargo")
    # `frontendDist` must resolve even for `tauri dev`, and the Tauri CLI refuses to
    # start when it does not exist. A stale-but-present dist is fine; `ui` replaces it.
    if not (DESKTOP_DIST / "index.html").is_file() and not copy_dist():
        write_placeholder()
    print("desktop toolchain ok")
    return 0


def cargo_env() -> dict:
    """The Tauri CLI shells out to cargo, so cargo has to be on *its* PATH."""
    env = os.environ.copy()
    cargo_bin = str(Path(require("cargo")).parent)
    if cargo_bin not in env.get("PATH", "").split(os.pathsep):
        env["PATH"] = cargo_bin + os.pathsep + env.get("PATH", "")
    return env


def cmd_dev(args: argparse.Namespace) -> int:
    pnpm = require("pnpm")
    ensure_desktop_install(pnpm)
    run([pnpm, "tauri", "dev"], cwd=DESKTOP_DIR, env=cargo_env())
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    pnpm = require("pnpm")
    ensure_desktop_install(pnpm)
    cmd_ui(args)
    cmd = [pnpm, "tauri", "build"]
    if args.debug:
        cmd.append("--debug")
    if args.target:
        cmd += ["--target", args.target]
    run(cmd, cwd=DESKTOP_DIR, env=cargo_env())
    profile = "debug" if args.debug else "release"
    bundle = DESKTOP_DIR / "src-tauri" / "target" / profile / "bundle"
    print(f"\nbundles under {bundle}")
    return 0


def ensure_desktop_install(pnpm: str) -> None:
    if not (DESKTOP_DIR / "node_modules").is_dir():
        run([pnpm, "install"], cwd=DESKTOP_DIR)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("check", help="verify the toolchain and that a frontendDist exists")
    sub.add_parser("ui", help="build app/ui and copy its dist into app/desktop")
    sub.add_parser("dev", help="run the shell against the Vite dev server on :1430")
    build = sub.add_parser("build", help="build the .app / .dmg bundle")
    build.add_argument("--debug", action="store_true", help="debug profile, faster to build")
    build.add_argument("--target", help="Rust target triple")
    args = parser.parse_args(argv)

    handlers = {"check": cmd_check, "ui": cmd_ui, "dev": cmd_dev, "build": cmd_build}
    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 2
    if not hasattr(args, "debug"):
        args.debug = False
    if not hasattr(args, "target"):
        args.target = None
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
