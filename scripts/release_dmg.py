#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Build the release .dmg that opens on a Mac that is not this one.

An unsigned bundle is fine for the machine that built it and dead on arrival
everywhere else: Gatekeeper quarantines anything downloaded, and for an app with no
Developer ID signature + notarization ticket the dialog says "MamboDubb is damaged
and can't be opened" — the one cure is `xattr -dc` in Terminal, which is exactly the
step this whole app exists to spare people. So the release path is signing and
notarization or it is not a release path.

    uv run --script scripts/release_dmg.py             # the one release command
    uv run --script scripts/release_dmg.py --dry-run   # print the plan, run nothing

Signing is Tauri-native, not post-hoc `codesign --deep`: the bundler signs every
Mach-O inside-out (the uv sidecar included), applies the hardened runtime, submits to
Apple, waits, and staples — all driven by four environment variables it reads itself:

    APPLE_SIGNING_IDENTITY   "Developer ID Application: Name (TEAMID)"
    APPLE_ID                 the Apple ID email
    APPLE_PASSWORD           an app-specific password, never the account password
    APPLE_TEAM_ID            the 10-character team id

What this script adds around `build_desktop.py build` is everything Tauri does not:
the submodule gate (a payload without Qwen3-TTS provisions a workspace that can never
`uv sync`), notarizing **the .dmg itself** (Tauri notarizes and staples the .app; a
stapled ticket on the container too means Gatekeeper clears the download even
offline), and the paranoid post-build verification — `codesign --verify`, `spctl`,
`stapler validate` — so a bad artifact fails here instead of on a user's Mac.

With none of the four variables set it still builds, UNSIGNED, and shouts about it:
that artifact is for this machine and for testers who are told the xattr incantation
up front. Prerequisites, certificate setup, and the release checklist live in
docs/RELEASING.md.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_desktop  # noqa: E402  the build itself; this script only wraps it

ROOT = build_desktop.ROOT
SRC_TAURI = ROOT / "app" / "desktop" / "src-tauri"
SUBMODULE = ROOT / "third_party" / "Qwen3-TTS"

# Tauri reads these itself during `tauri build`; the split matters because the first
# alone signs without notarizing, which on current macOS still gets the app blocked
# ("Apple could not verify…"), just with a politer dialog than "damaged".
SIGN_VAR = "APPLE_SIGNING_IDENTITY"
NOTARY_VARS = ("APPLE_ID", "APPLE_PASSWORD", "APPLE_TEAM_ID")

DRY_RUN = False


def run(cmd: list[str], cwd: Path = ROOT) -> None:
    # Same "+ cmd" echo as build_desktop.run, but with the app-specific password
    # redacted — a release often runs in CI, and CI keeps its logs.
    shown = list(cmd)
    if "--password" in shown:
        shown[shown.index("--password") + 1] = "•••"
    prefix = "+ (dry-run)" if DRY_RUN else "+"
    print(prefix, " ".join(shown), f"(in {cwd})", flush=True)
    if not DRY_RUN:
        subprocess.run(cmd, cwd=cwd, check=True)


def banner(lines: list[str], char: str = "!") -> None:
    width = max(len(line) for line in lines) + 4
    print()
    print(char * width)
    for line in lines:
        print(f"{char} {line.ljust(width - 4)} {char}")
    print(char * width)
    print()


def check_submodule() -> None:
    """Refuse to bundle a payload that cannot `uv sync` on the user's machine."""
    if (SUBMODULE / "pyproject.toml").is_file():
        return
    raise SystemExit(
        "error: third_party/Qwen3-TTS is not checked out, and the provisioned\n"
        "workspace could never `uv sync` without it. Run:\n"
        "    git submodule update --init --recursive"
    )


def bundle_version() -> str:
    """tauri.conf.json's version is the one that names the .dmg."""
    conf = json.loads((SRC_TAURI / "tauri.conf.json").read_text(encoding="utf-8"))
    return conf["version"]


def artifact_paths() -> tuple[Path, Path]:
    """Where `tauri build` leaves the .app and .dmg for the host's own triple."""
    bundle = SRC_TAURI / "target" / "release" / "bundle"
    app = bundle / "macos" / "MamboDubb.app"
    arch = {"arm64": "aarch64", "x86_64": "x64"}.get(os.uname().machine, os.uname().machine)
    dmg = bundle / "dmg" / f"MamboDubb_{bundle_version()}_{arch}.dmg"
    return app, dmg


def verify_signed(app: Path, dmg: Path, notarized: bool) -> None:
    """Fail here, not on a user's Mac. Every check is what Gatekeeper will run."""
    run(["codesign", "--verify", "--deep", "--strict", "--verbose=2", str(app)])
    if notarized:
        # spctl only accepts a Developer ID app once Apple has notarized it, so this
        # doubles as proof the ticket actually reached the bundle.
        run(["spctl", "--assess", "--type", "exec", "-v", str(app)])
        run(["xcrun", "stapler", "validate", str(app)])
        run(["xcrun", "stapler", "validate", str(dmg)])


def notarize_dmg(dmg: Path) -> None:
    """Tauri notarized and stapled the .app; ticket the container too.

    A stapled .dmg passes Gatekeeper's download assessment without a round-trip to
    Apple, so the install works on a Mac that is offline at first launch.
    """
    run([
        "xcrun", "notarytool", "submit", str(dmg), "--wait",
        "--apple-id", os.environ["APPLE_ID"],
        "--password", os.environ["APPLE_PASSWORD"],
        "--team-id", os.environ["APPLE_TEAM_ID"],
    ])
    run(["xcrun", "stapler", "staple", str(dmg)])


def main(argv: list[str] | None = None) -> int:
    global DRY_RUN
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print every command the release would run, run none of them",
    )
    args = parser.parse_args(argv)
    DRY_RUN = args.dry_run

    check_submodule()

    signing = bool(os.environ.get(SIGN_VAR))
    notary_missing = [v for v in NOTARY_VARS if not os.environ.get(v)]
    notarizing = signing and not notary_missing

    if not signing:
        banner([
            "BUILDING UNSIGNED — this .dmg works ONLY on this machine.",
            "On any other Mac, Gatekeeper says the app \"is damaged and",
            "can't be opened\". Testers must clear quarantine by hand:",
            "    xattr -dc /Applications/MamboDubb.app",
            f"For a real release set {SIGN_VAR}, {', '.join(NOTARY_VARS)}",
            "— see docs/RELEASING.md.",
        ])
    elif not notarizing:
        banner([
            f"{SIGN_VAR} is set but {', '.join(notary_missing)} " + ("is" if len(notary_missing) == 1 else "are") + " not:",
            "the app will be SIGNED but NOT NOTARIZED, and current macOS",
            "still blocks it (\"Apple could not verify…\"). Set all of",
            f"{', '.join(NOTARY_VARS)} for a shippable artifact.",
        ])
    else:
        print(f"signing as {os.environ[SIGN_VAR]!r}, notarizing as "
              f"{os.environ['APPLE_ID']} (team {os.environ['APPLE_TEAM_ID']})")

    # The build proper: build_desktop.py owns UI build + payload staging (via the
    # `pretauri` hook) + `pnpm tauri build`; Tauri signs/notarizes/staples the .app
    # itself from the env vars above. Release profile only — a --debug bundle is not
    # a release, whatever it is signed with.
    if DRY_RUN:
        run(["uv", "run", "--script", "scripts/build_desktop.py", "build"])
    else:
        build = argparse.Namespace(debug=False, target=None)
        build_desktop.cmd_build(build)

    app, dmg = artifact_paths()
    if not DRY_RUN and not dmg.is_file():
        raise SystemExit(f"error: expected {dmg} — did the dmg bundle step fail?")

    if notarizing:
        notarize_dmg(dmg)
    if signing:
        verify_signed(app, dmg, notarized=notarizing)

    print(f"\nrelease artifact: {dmg}")
    if not signing:
        print("(UNSIGNED — see the warning above before giving this to anyone)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
