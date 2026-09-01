"""The official `uv` release archive: which one this machine wants, and how to
get the binary out of it with its checksum checked.

One copy, because there were two and they had already drifted. `dubbing_app/
install.py` (the Setup screen's uv button) and `scripts/stage_desktop_payload.py`
(the desktop bundle's sidecar) both need the same six steps: name the target
triple, build the asset name, fetch the archive, fetch the `.sha256` beside it,
compare, and pull one file out of a zip or a tar. Both had written those six
steps out, and by the time anybody looked they no longer agreed about the first
one: the staging script refused an unknown triple with `SystemExit` and the app
quietly returned `None`, which is a different answer to a different question
asked of the same table. A table that exists twice is a table that is wrong once.

The two callers still differ where they should, and only there. `host_triple()`
answers `None` for a machine astral publishes nothing for, and what that means is
the caller's business: the app turns its button off, the staging script refuses
the build. Everything above that line is here.

Stdlib only, and no imports from anywhere else in this repo, because
`scripts/stage_desktop_payload.py` runs as a `uv run --script` with no
dependencies at all, before the workspace venv it is staging necessarily exists.
"""

from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

# Where the official builds live. The asset names *are* Rust target triples,
# which is the happy accident that makes this a one-line mapping to what Tauri's
# `externalBin` wants.
UV_RELEASES = "https://github.com/astral-sh/uv/releases"

# The triples there is both an official uv asset and a Tauri bundle target for,
# mapped to the extension of that asset. A triple that is not a key here is one
# neither caller can honour, which is what makes this table the single place the
# question "can this machine have uv?" is answered.
SUPPORTED_TRIPLES: dict[str, str] = {
    "aarch64-apple-darwin": ".tar.gz",
    "x86_64-apple-darwin": ".tar.gz",
    "x86_64-unknown-linux-gnu": ".tar.gz",
    "aarch64-unknown-linux-gnu": ".tar.gz",
    "x86_64-pc-windows-msvc": ".zip",
    "aarch64-pc-windows-msvc": ".zip",
}

# Machine names, as the two libraries that report them spell them, folded to the
# two architectures astral builds for. Anything else (i686, riscv64, ppc64le) is
# a machine with no uv release, and `host_triple` says so by answering None
# rather than by guessing at the nearest triple and 404ing later.
_ARM = frozenset({"arm64", "aarch64"})
_X86 = frozenset({"x86_64", "amd64"})


def host_triple(platform: str | None = None, machine: str | None = None) -> str | None:
    """This machine's uv release triple, or None when astral publishes none.

    `None` rather than an exception, because the two callers disagree about what
    to do with it and both readings are right: the Setup screen's button turns
    itself off (a button that downloads a 404 is worse than no button), and the
    staging script stops the build. Neither of those belongs in a lookup.

    Both arguments are injectable so a test can ask about a machine it is not
    running on, which is the only way any of this gets tested off Windows.
    """
    name = (platform if platform is not None else sys.platform).lower()
    if machine is None:
        import platform as platform_module

        machine = platform_module.machine()
    machine = machine.lower()
    if machine in _ARM:
        arch = "aarch64"
    elif machine in _X86:
        arch = "x86_64"
    else:
        return None
    if name == "darwin":
        return f"{arch}-apple-darwin"
    if name.startswith("win"):
        return f"{arch}-pc-windows-msvc"
    # Everything else is treated as Linux, the same fold `dubbing.tools.
    # platform_key` performs and for the same reason: a POSIX box we do not
    # drive a package manager on is honestly described as Linux, and astral's
    # `-unknown-linux-gnu` build is a static musl-free binary that runs there.
    return f"{arch}-unknown-linux-gnu"


def asset_name(triple: str) -> str:
    """The release asset for `triple`. Raises for a triple this cannot serve, so
    that a caller that skipped `host_triple` cannot build a URL out of a guess."""
    try:
        return f"uv-{triple}{SUPPORTED_TRIPLES[triple]}"
    except KeyError:
        raise RuntimeError(
            f"no uv release is published for {triple}; known targets: "
            f"{', '.join(sorted(SUPPORTED_TRIPLES))}") from None


def release_base(version: str = "latest") -> str:
    """The directory the asset and its checksum both live in."""
    return (f"{UV_RELEASES}/latest/download" if version == "latest"
            else f"{UV_RELEASES}/download/{version}")


def download(url: str, timeout: float = 180.0) -> bytes:
    """Fetch `url`. Module-level so a test replaces the whole thing and no bytes
    cross the network."""
    import urllib.request

    # Always an https://github.com literal built from a triple validated against
    # SUPPORTED_TRIPLES above; nothing here comes from a request or an argv.
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        return response.read()


def fetch_uv(triple: str, version: str = "latest",
             log=None, fetch=None) -> bytes:
    """The `uv` binary for `triple`, verified, as bytes.

    The archive and deliberately not `curl -LsSf https://astral.sh/uv/install.sh
    | sh` (or the `irm … | iex` spelling). A piped script is whatever the URL
    served at the moment it was read and nothing downstream can check it; an
    archive has a checksum published beside it, so a corrupted or truncated
    transfer is caught before a binary is written rather than after it fails to
    run. The checksum travels from the same host as the archive, so it proves the
    transfer and not the publisher, which is exactly the guarantee astral's own
    installer gives itself. And there is no shell to pipe into: one caller is a
    GUI app with no terminal, and on Windows the official line spawns a
    policy-bypassing PowerShell, which some managed machines simply refuse.

    `fetch` is injectable for the same reason `download` is module level: the
    tests must never reach the network, and one seam is easier to keep honest
    than two.
    """
    asset = asset_name(triple)
    base = release_base(version)
    get = fetch if fetch is not None else download
    if log is not None:
        log(f"downloading {base}/{asset}")
    blob = get(f"{base}/{asset}")
    published = get(f"{base}/{asset}.sha256").decode("utf-8", "replace").split()
    digest = hashlib.sha256(blob).hexdigest()
    if not published or published[0].lower() != digest:
        raise RuntimeError(
            f"{asset} does not match its published checksum (got {digest}, "
            f"expected {published[0] if published else '<none>'}), so it was not "
            "installed")
    if log is not None:
        log(f"checksum verified against {asset}.sha256")
    return extract(blob, triple, asset)


def binary_name(triple: str) -> str:
    return "uv.exe" if triple.endswith("windows-msvc") else "uv"


def extract(blob: bytes, triple: str, asset: str | None = None) -> bytes:
    """The one binary out of the release archive, by basename.

    By basename rather than by full path because astral has moved it between a
    flat archive and a `uv-<triple>/` directory before now, and a hardcoded path
    would turn that into an installer that stopped working.
    """
    asset = asset or asset_name(triple)
    wanted = binary_name(triple)
    if SUPPORTED_TRIPLES.get(triple, ".tar.gz") == ".zip":
        import zipfile

        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            member = next((n for n in archive.namelist()
                           if n.rsplit("/", 1)[-1] == wanted), None)
            if member is None:
                raise RuntimeError(f"no {wanted} inside {asset}")
            return archive.read(member)
    import tarfile

    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        member = next((m for m in archive.getmembers()
                       if m.isfile() and m.name.rsplit("/", 1)[-1] == wanted), None)
        if member is None:
            raise RuntimeError(f"no {wanted} inside {asset}")
        extracted = archive.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"could not read {wanted} out of {asset}")
        return extracted.read()


def write_binary(payload: bytes, target: Path) -> Path:
    """Write `payload` to `target` through an `.incoming` sibling.

    Renamed into place only once it is whole, so an interrupted download leaves
    the old uv (or no uv) rather than a truncated one every later run would try
    to execute.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(target.name + ".incoming")
    staging.write_bytes(payload)
    staging.chmod(staging.stat().st_mode | 0o755)
    staging.replace(target)
    return target


__all__ = ["SUPPORTED_TRIPLES", "UV_RELEASES", "asset_name", "binary_name",
           "download", "extract", "fetch_uv", "host_triple", "release_base",
           "write_binary"]
