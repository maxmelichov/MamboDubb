#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Build the release .dmg that opens on a Mac that is not this one.

An unsigned bundle is fine for the machine that built it and dead on arrival
everywhere else: Gatekeeper quarantines anything downloaded, and a linker-signed-only
.app then shows "MamboDubb is damaged and can't be opened. You should move it to the
Trash." The lasting fix is a Developer ID signature and a notarization ticket. Until
those exist, unsigned builds still have to be installable: this script ad-hoc-signs
the bundle (so the signature actually covers Info.plist and resources), drops
install.sh into the .dmg as "Install MamboDubb.command", and the README leads with

    curl -fsSL https://github.com/maxmelichov/MamboDubb/releases/latest/download/install.sh | sh

which downloads, copies to /Applications, clears quarantine, re-signs, and launches.

    uv run --script scripts/release_dmg.py                  # the one release command
    uv run --script scripts/release_dmg.py --dry-run        # print the plan, run nothing
    uv run --script scripts/release_dmg.py --finish-unsigned  # post-process an already-built .dmg

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

With none of the four variables set it still builds, unsigned, ad-hoc-signs the
bundle, and injects the installer. Prerequisites, certificate setup, and the
release checklist live in docs/RELEASING.md.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_desktop  # noqa: E402  the build itself; this script only wraps it

ROOT = build_desktop.ROOT
SRC_TAURI = ROOT / "app" / "desktop" / "src-tauri"
SUBMODULE = ROOT / "third_party" / "Qwen3-TTS"
INSTALL_SH = ROOT / "install.sh"
DMG_HELPER_NAME = "Install MamboDubb.command"

# The prebuilt web UI, attached to the release so the Windows and Linux
# installers do not have to put Node and pnpm on a user's machine to get a
# browser editor. It is produced here rather than by hand because the build has
# already happened by this point: `build_desktop.cmd_build` builds app/ui/dist
# on its way to the .app, so packing it is free and forgetting it is the only
# way it can go missing from a release.
UI_DIST = ROOT / "app" / "ui" / "dist"
UI_ASSET_NAME = "mambodubb-ui-dist.tar.gz"

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


def adhoc_sign(app: Path) -> None:
    """Replace linker-signed inner binaries with a real ad-hoc bundle signature.

    An unsigned Tauri .app still has `adhoc,linker-signed` Mach-Os. Gatekeeper then
    reports the bundle as damaged ("code has no resources but signature indicates
    they must be present") even after quarantine is cleared. `codesign -s -` of the
    bundle seals Info.plist and resources so install.sh's `xattr -cr` is enough.
    """
    if not app.is_dir():
        raise SystemExit(f"error: expected {app}")
    run(["codesign", "--force", "--deep", "--sign", "-", str(app)])
    run(["codesign", "--verify", "--deep", "--strict", "--verbose=2", str(app)])


def _attach_rw(image: Path) -> tuple[str, Path]:
    """Attach a read-write disk image; return (device, mountpoint)."""
    proc = subprocess.run(
        ["hdiutil", "attach", "-readwrite", "-nobrowse", "-noverify", str(image)],
        check=True, capture_output=True, text=True,
    )
    device = ""
    mount: Path | None = None
    for line in proc.stdout.splitlines():
        if "/Volumes/" in line:
            parts = line.split()
            device = parts[0]
            mount = Path(parts[-1])
    if not device or mount is None:
        raise SystemExit(f"error: could not parse hdiutil attach output:\n{proc.stdout}")
    return device, mount


def inject_signed_app_and_helper(app: Path, dmg: Path) -> None:
    """Put the ad-hoc-signed .app and install.sh into the already-built .dmg.

    Tauri writes the .dmg from the unsigned tree. After we sign the .app we have
    to copy that signed tree back in, or the image still carries the broken
    linker-signed bundle. The helper is install.sh under a Finder-friendly name.
    """
    if not INSTALL_SH.is_file():
        raise SystemExit(f"error: missing {INSTALL_SH}")
    if not dmg.is_file():
        raise SystemExit(f"error: expected {dmg}")

    with tempfile.TemporaryDirectory(prefix="mambodubb-dmg-") as tmp:
        tmp_path = Path(tmp)
        rw = tmp_path / "rw.dmg"
        rebuilt = tmp_path / "out.dmg"
        run(["hdiutil", "convert", str(dmg), "-format", "UDRW", "-o", str(rw)])
        # A few extra MB so replacing the .app + dropping the helper cannot fail
        # on a volume packed to the last block.
        run(["hdiutil", "resize", "-size", "180m", str(rw)])
        print(f"+ hdiutil attach {rw}", flush=True)
        device, mount = _attach_rw(rw)
        try:
            dest_app = mount / "MamboDubb.app"
            if dest_app.exists():
                shutil.rmtree(dest_app)
            print(f"+ ditto {app} -> {dest_app}", flush=True)
            subprocess.run(["ditto", str(app), str(dest_app)], check=True)
            helper = mount / DMG_HELPER_NAME
            print(f"+ copy {INSTALL_SH} -> {helper}", flush=True)
            shutil.copy2(INSTALL_SH, helper)
            helper.chmod(0o755)
            # Hide the .command extension so the icon reads as an installer.
            subprocess.run(
                ["SetFile", "-a", "E", str(helper)],
                check=False, capture_output=True,
            )
        finally:
            run(["hdiutil", "detach", "-quiet", device])
        run([
            "hdiutil", "convert", str(rw), "-format", "UDZO",
            "-imagekey", "zlib-level=9", "-o", str(rebuilt),
        ])
        print(f"+ replace {dmg}", flush=True)
        shutil.move(str(rebuilt), str(dmg))


def pack_ui_dist(dest_dir: Path) -> Path | None:
    """Write mambodubb-ui-dist.tar.gz next to the .dmg, or say why it could not.

    The archive holds the *contents* of app/ui/dist at its top level, not a
    wrapping directory: the installers untar it straight into the checkout's
    own app/ui/dist, which is where the server looks by default, so a wrapper
    would put index.html one level too deep and the server would quietly serve
    the API alone.

    Names are sorted and the entries stripped of uid/gid so two releases built
    from the same commit produce the same bytes, which is what makes preflight's
    digest comparison against the uploaded asset mean anything.
    """
    if not (UI_DIST / "index.html").is_file():
        print(f"warning: {UI_DIST} has no index.html; skipping {UI_ASSET_NAME}")
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / UI_ASSET_NAME
    if DRY_RUN:
        print(f"+ (dry-run) pack {UI_DIST} -> {out}")
        return out
    print(f"+ pack {UI_DIST} -> {out}", flush=True)
    members = sorted(p for p in UI_DIST.rglob("*") if p.is_file())

    def scrub(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = 0
        return info

    with tarfile.open(out, "w:gz") as tar:
        for member in members:
            tar.add(member, arcname=str(member.relative_to(UI_DIST)), filter=scrub)
    return out


def finish_unsigned(app: Path, dmg: Path) -> None:
    """Make an unsigned Tauri .dmg installable on a Mac that is not this one."""
    adhoc_sign(app)
    inject_signed_app_and_helper(app, dmg)


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
    parser.add_argument(
        "--finish-unsigned", action="store_true",
        help="ad-hoc-sign and inject install.sh into an already-built .dmg; skip the build",
    )
    args = parser.parse_args(argv)
    DRY_RUN = args.dry_run

    check_submodule()

    if args.finish_unsigned:
        if os.environ.get(SIGN_VAR):
            print(f"{SIGN_VAR} is set; skipping unsigned finish (Tauri already signed)")
            return 0
        app, dmg = artifact_paths()
        if DRY_RUN:
            print(f"+ (dry-run) ad-hoc-sign {app} and inject {DMG_HELPER_NAME} into {dmg}")
            pack_ui_dist(dmg.parent)
            return 0
        finish_unsigned(app, dmg)
        ui = pack_ui_dist(dmg.parent)
        print(f"\nunsigned artifact: {dmg}")
        if ui:
            print(f"web UI asset:      {ui}")
        print(f"(ad-hoc signed; {DMG_HELPER_NAME} inside; install.sh is the download path)")
        return 0

    signing = bool(os.environ.get(SIGN_VAR))
    notary_missing = [v for v in NOTARY_VARS if not os.environ.get(v)]
    notarizing = signing and not notary_missing

    if not signing:
        banner([
            "BUILDING UNSIGNED. Finder-drag of a downloaded .dmg still trips",
            "Gatekeeper (\"damaged\", move to Trash). This build ad-hoc-signs",
            "the bundle and ships Install MamboDubb.command inside the .dmg,",
            "which a downloaded volume's quarantine flag stops anyone from",
            "double-clicking: it only runs from Terminal via sh. So users",
            "install with:",
            "    curl -fsSL https://github.com/maxmelichov/MamboDubb/releases/latest/download/install.sh | sh",
            f"A real drag-to-Applications release needs {SIGN_VAR} and",
            f"{', '.join(NOTARY_VARS)}; see docs/RELEASING.md.",
        ])
    elif not notarizing:
        banner([
            f"{SIGN_VAR} is set but {', '.join(notary_missing)} "
            + ("is" if len(notary_missing) == 1 else "are") + " not:",
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
        raise SystemExit(f"error: expected {dmg}; did the dmg bundle step fail?")

    if notarizing:
        notarize_dmg(dmg)
    if signing:
        verify_signed(app, dmg, notarized=notarizing)
    elif not DRY_RUN:
        finish_unsigned(app, dmg)

    ui = pack_ui_dist(dmg.parent)

    print(f"\nrelease artifact: {dmg}")
    if ui:
        print(f"web UI asset:     {ui}")
    print("upload both, plus install.sh: the two installers that run from source"
          " read the web UI asset off the release.")
    if not signing:
        print(f"(unsigned, ad-hoc signed, {DMG_HELPER_NAME} inside)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
