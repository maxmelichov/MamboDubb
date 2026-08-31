#!/bin/sh
# Install MamboDubb from source on Linux, and on a Mac that would rather run the
# server than the desktop app. Ends with the studio serving on 127.0.0.1:4400.
#
#   curl -fsSL https://raw.githubusercontent.com/maxmelichov/MamboDubb/main/install-server.sh | sh
#
# There is no .deb, no AppImage and no .exe to download, and there cannot be one
# until the account-level GitHub Actions block is lifted (docs/CROSS_PLATFORM.md).
# So the install is the from-source route, done for you: fetch the source at the
# latest release tag, put uv on the machine, resolve the Python side, put the
# built web UI in place, and start the server. The desktop app is a thin shell
# over exactly this server, so a browser gets the identical editor.
#
# Nothing here downloads a model. That is the Setup screen's job on first run,
# and it is about 25 GB.
#
# Environment overrides, all optional:
#   MAMBODUBB_DIR      where the checkout goes (default: $HOME/MamboDubb)
#   MAMBODUBB_REF      tag or branch to install (default: the latest release)
#   MAMBODUBB_PORT     port to serve on (default: 4400)
#   MAMBODUBB_START=0  install and stop, do not start the server
#   MAMBODUBB_UI_TARBALL  a local prebuilt UI archive instead of the release one
#
# Must work when piped to `sh`: do not read stdin, and never ask a question.

set -eu

REPO="maxmelichov/MamboDubb"
DIR="${MAMBODUBB_DIR:-$HOME/MamboDubb}"
PORT="${MAMBODUBB_PORT:-4400}"
UI_ASSET="mambodubb-ui-dist.tar.gz"

# Pinned rather than "latest": this is only the fallback path for building the
# UI when the release carries no prebuilt one, and a version that moves under
# the script is a build that breaks on a Tuesday for no reason anyone can see.
NODE_VERSION="v22.20.0"

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

info() {
    printf '%s\n' "$*"
}

have() {
    command -v "$1" >/dev/null 2>&1
}

OS=$(uname -s)
case "$OS" in
    Linux | Darwin) ;;
    *) die "this installer is for Linux and macOS; on Windows use install-server.ps1" ;;
esac

ARCH=$(uname -m)
case "$ARCH" in
    x86_64 | amd64) UV_ARCH=x86_64; NODE_ARCH=x64 ;;
    arm64 | aarch64) UV_ARCH=aarch64; NODE_ARCH=arm64 ;;
    *) die "unsupported architecture: $ARCH (x86_64 and arm64 are the ones with wheels)" ;;
esac

if [ "$OS" = Darwin ]; then
    UV_TRIPLE="${UV_ARCH}-apple-darwin"
    NODE_PLATFORM=darwin
else
    UV_TRIPLE="${UV_ARCH}-unknown-linux-gnu"
    NODE_PLATFORM=linux
fi

have curl || die "curl not found; install it first (this script was fetched with it)"
have tar || die "tar not found"

# Never reuse the name TMPDIR here either: this directory is deleted on exit,
# and anything that inherited it as TMPDIR then dies writing temp files. The
# same trap bit install.sh once; see its comment.
WORK=$(mktemp -d "${TMPDIR:-/tmp}/mambodubb-server.XXXXXX")
cleanup() {
    rm -rf "$WORK"
}
trap cleanup EXIT INT HUP TERM

# ---------------------------------------------------------------------------
# System packages: git to fetch the source, ffmpeg and sox to run a dub.
# ---------------------------------------------------------------------------
#
# dubbing/tools.py already decides this question for ffmpeg and sox and the
# answer here has to match it, or the installer and the app's Setup screen
# disagree about the same machine. Its rule: `brew` finishes on its own, and
# `sudo apt-get` does not, because it wants a password on a terminal a piped
# script does not have, and a hidden password prompt behind a pipe is a hang
# rather than an install. So a package manager runs unattended here only when it
# needs no password at all, which means root or a sudo that already answers
# `sudo -n true`. Every other case gets the exact line to paste, and the install
# continues: a missing ffmpeg does not stop the editor from opening, it stops a
# dub from finishing, and Setup says so in the app.

PM_LINE=""
PM_ARGV=""
if [ "$OS" = Darwin ]; then
    if have brew; then
        PM_LINE="brew install"
        PM_ARGV="brew install"
    fi
elif have apt-get; then
    PM_LINE="sudo apt-get install -y"
    PM_ARGV="apt-get install -y"
elif have dnf; then
    PM_LINE="sudo dnf install -y"
    PM_ARGV="dnf install -y"
elif have pacman; then
    PM_LINE="sudo pacman -S --noconfirm"
    PM_ARGV="pacman -S --noconfirm"
elif have zypper; then
    PM_LINE="sudo zypper install -y"
    PM_ARGV="zypper install -y"
elif have apk; then
    PM_LINE="sudo apk add"
    PM_ARGV="apk add"
fi

# Homebrew refuses to run as root, so it never takes the sudo prefix.
SUDO=""
CAN_INSTALL=no
if [ -n "$PM_ARGV" ]; then
    # Written as a full `if` and not `[ ... ] && CAN_INSTALL=yes`, because under
    # `set -e` a false test at the end of a branch ends the script, and root on a
    # Mac is a reason to skip brew, not a reason to stop installing.
    if [ "$OS" = Darwin ]; then
        if [ "$(id -u)" != 0 ]; then
            CAN_INSTALL=yes
        fi
    elif [ "$(id -u)" = 0 ]; then
        CAN_INSTALL=yes
    elif have sudo && sudo -n true >/dev/null 2>&1; then
        SUDO=sudo
        CAN_INSTALL=yes
    fi
fi

MISSING=""
for tool in git ffmpeg sox; do
    have "$tool" || MISSING="$MISSING $tool"
done

if [ -n "$MISSING" ]; then
    # apt's package list in a fresh container or a long-idle machine is stale,
    # and `apt-get install` on a stale list fails with 404s on every package.
    if [ "$CAN_INSTALL" = yes ]; then
        info "Installing:$MISSING"
        if have apt-get; then
            ${SUDO:+$SUDO} apt-get update -qq || true
        fi
        # shellcheck disable=SC2086  # word splitting is the point: argv, not one string
        if ! ${SUDO:+$SUDO} $PM_ARGV $MISSING; then
            info "warning: '$PM_ARGV$MISSING' failed; continuing"
        fi
    elif [ -n "$PM_LINE" ]; then
        info ""
        info "These are missing, and installing them here would need a password"
        info "this script has no terminal to ask on:"
        info "   $PM_LINE$MISSING"
        info "Run that line in another terminal. ffmpeg and sox are only needed once"
        info "you start a dub, so the editor still opens without them."
        info ""
    else
        info "warning: no package manager found; install$MISSING yourself"
    fi
fi

# git is the one that cannot wait: pyproject pins third_party/Qwen3-TTS as a
# path dependency, so a source tree without that submodule can never `uv sync`,
# and a tarball download has no submodule in it.
have git || die "git is required to fetch the source with its submodule; install it and rerun"

# ---------------------------------------------------------------------------
# The source, at the tag the latest release was cut from.
# ---------------------------------------------------------------------------
#
# A tag rather than main, because the prebuilt UI below is a release asset: tying
# both to the same tag is what keeps the web UI and the API it talks to in step.
REF="${MAMBODUBB_REF:-}"
if [ -z "$REF" ]; then
    REF=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" 2>/dev/null \
        | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -n 1) || REF=""
fi
[ -n "$REF" ] || REF=main

if [ -d "$DIR/.git" ]; then
    info "Updating the checkout in $DIR to $REF"
    git -C "$DIR" fetch --depth 1 origin "$REF" || die "could not fetch $REF"
    git -C "$DIR" checkout --detach FETCH_HEAD >/dev/null 2>&1 \
        || die "could not check out $REF; move $DIR aside and rerun"
    git -C "$DIR" submodule update --init --recursive || die "submodule checkout failed"
elif [ -e "$DIR" ]; then
    die "$DIR exists and is not a git checkout; set MAMBODUBB_DIR to somewhere else"
else
    info "Cloning $REPO ($REF) into $DIR"
    git clone --depth 1 --branch "$REF" --recurse-submodules \
        "https://github.com/${REPO}.git" "$DIR" || die "clone failed"
fi

[ -f "$DIR/third_party/Qwen3-TTS/pyproject.toml" ] \
    || die "third_party/Qwen3-TTS is empty; run: git -C $DIR submodule update --init --recursive"

cd "$DIR"

# ---------------------------------------------------------------------------
# uv, which brings its own Python.
# ---------------------------------------------------------------------------

find_uv() {
    if have uv; then
        command -v uv
        return 0
    fi
    for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        if [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

sha256_of() {
    if have sha256sum; then
        sha256sum "$1" | cut -d' ' -f1
    elif have shasum; then
        shasum -a 256 "$1" | cut -d' ' -f1
    else
        printf ''
    fi
}

UV=$(find_uv) || UV=""
if [ -z "$UV" ]; then
    info "Installing uv (${UV_TRIPLE})"
    base="https://github.com/astral-sh/uv/releases/latest/download/uv-${UV_TRIPLE}.tar.gz"
    curl -fsSL -o "$WORK/uv.tar.gz" "$base" || die "could not download uv"
    # The published .sha256 travels from the same host as the archive, so it
    # proves the transfer and not the publisher. That is exactly the check
    # astral's own install script makes, and it is strictly better than piping
    # an unpinned script straight into a shell, which is the alternative.
    if curl -fsSL -o "$WORK/uv.sha256" "${base}.sha256"; then
        want=$(cut -d' ' -f1 < "$WORK/uv.sha256")
        got=$(sha256_of "$WORK/uv.tar.gz")
        if [ -n "$got" ] && [ "$want" != "$got" ]; then
            die "uv download failed its checksum; refusing to install it"
        fi
        [ -n "$got" ] || info "warning: no sha256 tool on this machine; uv checksum unverified"
    else
        info "warning: could not fetch uv's published checksum; continuing"
    fi
    mkdir -p "$WORK/uv" "$HOME/.local/bin"
    tar -xzf "$WORK/uv.tar.gz" -C "$WORK/uv"
    extracted=$(find "$WORK/uv" -type f -name uv -perm -u+x | head -n 1)
    [ -n "$extracted" ] || die "the uv archive did not contain a uv binary"
    install -m 0755 "$extracted" "$HOME/.local/bin/uv"
    UV="$HOME/.local/bin/uv"
fi
info "Using uv at $UV"

# `--extra app` rather than a plain `uv sync`. fastapi, uvicorn, httpx and
# python-multipart do land today either way, as transitive dependencies of the
# gradio that sits under qwen-tts, but that is an accident of somebody else's
# dependency tree and not a promise. The `app` extra is where this server's own
# dependencies are declared, and asking for it by name is what keeps the install
# working on the day gradio drops one of them.
info "Resolving Python dependencies (several GB the first time)"
"$UV" sync --extra app || die "uv sync failed"

# ---------------------------------------------------------------------------
# The web UI.
# ---------------------------------------------------------------------------
#
# The server serves app/ui/dist by default, so getting the built UI into that
# directory is the whole job. Downloading it is the fast path and the reason
# Node is not a requirement of this install; building it is the fallback for a
# release that predates the asset, and that fallback provisions its own Node
# rather than sending the user away to install one.

DIST="$DIR/app/ui/dist"

fetch_ui() {
    archive="$WORK/ui.tar.gz"
    if [ -n "${MAMBODUBB_UI_TARBALL:-}" ]; then
        [ -f "$MAMBODUBB_UI_TARBALL" ] || die "no such file: $MAMBODUBB_UI_TARBALL"
        cp "$MAMBODUBB_UI_TARBALL" "$archive"
    else
        # -f without -S: a release cut before this asset existed answers 404,
        # and that is a normal fallback into the source build, not an error
        # worth printing curl's own complaint about.
        url="https://github.com/${REPO}/releases/download/${REF}/${UI_ASSET}"
        curl -fsL -o "$archive" "$url" || return 1
    fi
    mkdir -p "$DIST"
    tar -xzf "$archive" -C "$DIST" || return 1
    [ -f "$DIST/index.html" ]
}

ensure_node() {
    # A Node already on PATH is used as-is; corepack ships with every Node 16+
    # and is how pnpm is meant to arrive, so it comes before any npm -g install.
    if have node && have pnpm; then
        return 0
    fi
    # Nested in an `if` condition and not run as a statement, because `set -e`
    # kills the script on the last command of an && list that fails, and a
    # `corepack enable` refused by a read-only prefix is a fallback to the
    # private Node below, not the end of the install.
    if have node && have corepack; then
        if corepack enable pnpm >/dev/null 2>&1 && have pnpm; then
            return 0
        fi
    fi
    node_dir="$DIR/.tools/node-${NODE_VERSION}-${NODE_PLATFORM}-${NODE_ARCH}"
    if [ ! -x "$node_dir/bin/pnpm" ]; then
        name="node-${NODE_VERSION}-${NODE_PLATFORM}-${NODE_ARCH}"
        # .tar.gz and not the smaller .tar.xz: xz is not installed on every
        # minimal Linux image, and a missing decompressor here would be a
        # confusing failure a long way from its cause.
        info "Installing Node ${NODE_VERSION} under $DIR/.tools (only to build the UI)"
        curl -fsSL -o "$WORK/node.tar.gz" \
            "https://nodejs.org/dist/${NODE_VERSION}/${name}.tar.gz" || return 1
        if curl -fsSL -o "$WORK/node.sums" \
            "https://nodejs.org/dist/${NODE_VERSION}/SHASUMS256.txt"; then
            want=$(grep " ${name}.tar.gz\$" "$WORK/node.sums" | cut -d' ' -f1)
            got=$(sha256_of "$WORK/node.tar.gz")
            if [ -n "$want" ] && [ -n "$got" ] && [ "$want" != "$got" ]; then
                info "error: Node download failed its checksum"
                return 1
            fi
        fi
        mkdir -p "$DIR/.tools"
        rm -rf "$node_dir"
        tar -xzf "$WORK/node.tar.gz" -C "$DIR/.tools" || return 1
        # PATH first, then corepack. The shipped `corepack` is a
        # `#!/usr/bin/env node` script, so on a machine with no system Node it
        # exits 127 with "env: 'node': No such file or directory", which reads
        # like a missing Node rather than a Node that is right there unlisted.
        PATH="$node_dir/bin:$PATH"
        export PATH
        corepack enable pnpm --install-directory "$node_dir/bin" || return 1
    fi
    PATH="$node_dir/bin:$PATH"
    export PATH
    have pnpm
}

build_ui() {
    info "No prebuilt web UI on the ${REF} release; building it from source."
    ensure_node || return 1
    info "Building the web UI from source"
    ( cd "$DIR/app/ui" && pnpm install --frozen-lockfile && pnpm build ) || return 1
    [ -f "$DIST/index.html" ]
}

if [ -f "$DIST/index.html" ]; then
    info "Web UI already built."
elif fetch_ui; then
    if [ -n "${MAMBODUBB_UI_TARBALL:-}" ]; then
        info "Web UI unpacked from $MAMBODUBB_UI_TARBALL."
    else
        info "Web UI installed from the ${REF} release."
    fi
elif build_ui; then
    info "Web UI built from source."
else
    die "could not install the web UI; see docs/SERVER.md for the manual build"
fi

# ---------------------------------------------------------------------------
# Serve.
# ---------------------------------------------------------------------------

info ""
info "Installed in $DIR"
info "Start it again later with:"
info "    cd $DIR && $UV run mambodubb --port $PORT"
info ""
info "First run: open Setup in the editor and press Install everything. That is"
info "about 25 GB of models, and it resumes where it left off if you interrupt it."
info ""

if [ "${MAMBODUBB_START:-1}" = 0 ]; then
    info "MAMBODUBB_START=0, so the server was not started."
    exit 0
fi

info "Starting the studio on http://127.0.0.1:${PORT} (Ctrl-C to stop)"
exec "$UV" run mambodubb --port "$PORT"
