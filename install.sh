#!/bin/sh
# Install MamboDubb on Apple Silicon and actually launch it.
#
# Safari and Chrome quarantine every download. An unsigned (or linker-signed-only)
# .app then makes Gatekeeper show "MamboDubb is damaged and can't be opened. You
# should move it to the Trash." That is not a corrupt file. This script copies
# the app to /Applications, clears the quarantine, and replaces the incomplete
# inner signature with a real ad-hoc bundle signature so the app opens.
#
#   curl -fsSL https://github.com/maxmelichov/MamboDubb/releases/latest/download/install.sh | sh
#   sh install.sh /path/to/MamboDubb_*.dmg
#   Double-click "Install MamboDubb.command" inside the .dmg (this file, renamed)
#
# Must work when piped to `sh`: do not read stdin.

set -eu

REPO="maxmelichov/MamboDubb"
APP_NAME="MamboDubb.app"
DEST="/Applications/${APP_NAME}"

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

info() {
    printf '%s\n' "$*"
}

[ "$(uname -s)" = Darwin ] || die "this installer is for macOS"
[ "$(uname -m)" = arm64 ] || die "MamboDubb needs Apple Silicon (an M-series Mac)"

command -v hdiutil >/dev/null || die "hdiutil not found"
command -v ditto >/dev/null || die "ditto not found"

# When Finder launches a .command, $0 is the file. When piped (`curl | sh`), $0
# is `sh` and there is no sibling app to copy.
SCRIPT_DIR=""
case $0 in
    /* | ./* | ../*)
        if [ -f "$0" ]; then
            SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
        fi
        ;;
esac

LOCAL_DMG=""
if [ "${1-}" != "" ]; then
    [ -f "$1" ] || die "not a file: $1"
    LOCAL_DMG=$1
fi

SIBLING=""
if [ -n "$SCRIPT_DIR" ] && [ -d "$SCRIPT_DIR/$APP_NAME" ]; then
    SIBLING="$SCRIPT_DIR/$APP_NAME"
fi

TMPDIR=$(mktemp -d "${TMPDIR:-/tmp}/mambodubb-install.XXXXXX")
MOUNT=""
cleanup() {
    if [ -n "$MOUNT" ] && [ -d "$MOUNT" ]; then
        hdiutil detach "$MOUNT" -quiet >/dev/null 2>&1 || true
    fi
    rm -rf "$TMPDIR"
}
trap cleanup EXIT INT HUP TERM

latest_dmg_url() {
    api="https://api.github.com/repos/${REPO}/releases/latest"
    json=$(curl -fsSL "$api") || die "could not reach GitHub releases"
    # sed first: a clean Mac may not have python3 (it prompts to install Xcode).
    url=$(printf '%s\n' "$json" | sed -n 's/.*"browser_download_url": "\([^"]*MamboDubb_[^"]*_aarch64\.dmg\)".*/\1/p' | head -n 1)
    if [ -z "$url" ] && command -v python3 >/dev/null 2>&1; then
        url=$(printf '%s\n' "$json" | python3 -c '
import json, sys
rel = json.load(sys.stdin)
for a in rel.get("assets") or []:
    name = a.get("name") or ""
    if name.startswith("MamboDubb_") and name.endswith("_aarch64.dmg"):
        print(a["browser_download_url"])
        break
else:
    sys.exit(1)
' 2>/dev/null) || url=""
    fi
    [ -n "$url" ] || die "latest GitHub release has no MamboDubb_*_aarch64.dmg"
    printf '%s\n' "$url"
}

SRC=""
if [ -n "$SIBLING" ]; then
    info "Using the app next to this installer."
    SRC=$SIBLING
else
    if [ -n "$LOCAL_DMG" ]; then
        dmg=$LOCAL_DMG
        info "Using local disk image: $dmg"
    else
        url=$(latest_dmg_url)
        dmg="$TMPDIR/MamboDubb.dmg"
        info "Downloading $url"
        curl -fL --progress-bar -o "$dmg" "$url" || die "download failed"
    fi
    # Strip quarantine on the image itself so the copy does not inherit it.
    xattr -cr "$dmg" 2>/dev/null || true
    info "Mounting disk image…"
    attach=$(hdiutil attach -nobrowse -readonly "$dmg") || die "could not mount the disk image"
    MOUNT=$(printf '%s\n' "$attach" | awk '/\/Volumes\//{print $NF; exit}')
    [ -n "$MOUNT" ] && [ -d "$MOUNT/$APP_NAME" ] || die "disk image has no $APP_NAME"
    SRC="$MOUNT/$APP_NAME"
fi

if [ -d "$DEST" ]; then
    info "Replacing the existing install at $DEST"
    osascript -e 'tell application "MamboDubb" to quit' >/dev/null 2>&1 || true
    # Give the process a beat to drop its file locks.
    sleep 1
    rm -rf "$DEST"
fi

info "Copying to $DEST"
ditto "$SRC" "$DEST" || die "copy failed"

info "Clearing Gatekeeper quarantine…"
xattr -cr "$DEST" 2>/dev/null || true

# Tauri's unsigned bundle leaves Mach-Os linker-signed. Gatekeeper then says the
# app is damaged ("code has no resources but signature indicates they must be
# present"). A real ad-hoc signature of the bundle seals Info.plist + resources.
info "Signing the app on this Mac…"
if codesign --force --deep --sign - "$DEST" >/dev/null 2>&1; then
    codesign --verify --deep --strict "$DEST" >/dev/null 2>&1 \
        || info "warning: codesign verify reported a problem; trying to launch anyway"
else
    info "warning: codesign failed; launching anyway"
fi

info "Launching MamboDubb."
open "$DEST"

info ""
info "Installed. If macOS still complains, System Settings → Privacy & Security"
info "and click Open Anyway. Dragging the .app out of a downloaded .dmg will hit"
info "the Trash dialog again; rerun this installer instead."
