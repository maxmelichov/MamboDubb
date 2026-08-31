#!/usr/bin/env python3
"""Preflight gate. Run this before every push; nothing ships on a red.

    uv run scripts/preflight.py

Stdlib only, so `python3 scripts/preflight.py` works identically. Every check
here was earned by something that actually shipped broken. See
docs/SHIPPING.md for the check-to-incident table.

Exit codes: 0 if every check passed (WARN is not fatal), 1 otherwise.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EM_DASH = "—"

# The interpreter that has the test dependencies installed. Overridable so the
# gate is not welded to one machine.
TEST_PYTHON = Path(
    os.environ.get(
        "MAMBODUBB_TEST_PYTHON",
        "/Users/maxm/Documents/MamboRambo-folder/DubbingQwen/.venv/bin/python",
    )
)

# Tests that assume no local weights are on disk. They fail in any checkout
# that has a `models/` tree, which is every dev machine that has ever dubbed
# anything. Skipped by name, never silently.
MODELS_SENSITIVE_TESTS = (
    "tests/test_dubbing.py::test_low_vram_swaps_the_mlx_build_and_nothing_else",
)

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"

_results: list[tuple[str, str, str]] = []


def record(status: str, name: str, reason: str) -> None:
    _results.append((status, name, reason))
    print(f"{status:4}  {name:<28}  {reason}", flush=True)


def run(cmd: list[str], cwd: Path | None = None,
        env: dict | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        env=env,
        capture_output=True,
        text=True,
    )


def git(*args: str) -> str:
    proc = run(["git", *args])
    return proc.stdout if proc.returncode == 0 else ""


# --------------------------------------------------------------------------
# a. Version declarations agree across all five places.
# --------------------------------------------------------------------------

VERSION_SOURCES = (
    ("pyproject.toml", r'(?m)^\s*version\s*=\s*"([^"]+)"'),
    ("dubbing_app/__init__.py", r'(?m)^__version__\s*=\s*"([^"]+)"'),
    ("app/desktop/package.json", r'"version"\s*:\s*"([^"]+)"'),
    ("app/desktop/src-tauri/Cargo.toml", r'(?m)^version\s*=\s*"([^"]+)"'),
    ("app/desktop/src-tauri/tauri.conf.json", r'"version"\s*:\s*"([^"]+)"'),
)


def check_versions() -> None:
    found: dict[str, str] = {}
    missing: list[str] = []
    for rel, pattern in VERSION_SOURCES:
        path = ROOT / rel
        if not path.exists():
            missing.append(rel)
            continue
        match = re.search(pattern, path.read_text(encoding="utf-8"))
        if not match:
            missing.append(rel)
            continue
        found[rel] = match.group(1)
    if missing:
        record(FAIL, "versions", f"no version found in: {', '.join(missing)}")
        return
    distinct = sorted(set(found.values()))
    if len(distinct) != 1:
        drift = ", ".join(f"{rel}={ver}" for rel, ver in found.items())
        record(FAIL, "versions", f"drifted across {len(distinct)} values: {drift}")
        return
    record(PASS, "versions", f"all five declare {distinct[0]}")


# --------------------------------------------------------------------------
# b. No tracked symlinks sweeping local bulk into the repo.
# --------------------------------------------------------------------------

SUBMODULE = "third_party/Qwen3-TTS"
FORBIDDEN_LINK_NAMES = {"models", "outputs"}


def check_tracked_links() -> None:
    listing = git("ls-files", "-s")
    if not listing:
        record(WARN, "tracked-links", "git ls-files produced nothing; not a checkout?")
        return
    symlinks: list[str] = []
    submodule_mode = None
    for line in listing.splitlines():
        head, _, path = line.partition("\t")
        mode = head.split()[0]
        if path == SUBMODULE:
            submodule_mode = mode
        if mode == "120000":
            symlinks.append(path)
    offenders = [p for p in symlinks if Path(p).name in FORBIDDEN_LINK_NAMES]
    if offenders:
        record(FAIL, "tracked-links", f"tracked symlink(s) committed: {', '.join(offenders)}")
        return
    if submodule_mode is None:
        record(FAIL, "tracked-links", f"{SUBMODULE} is not tracked at all")
        return
    if submodule_mode != "160000":
        record(
            FAIL,
            "tracked-links",
            f"{SUBMODULE} is mode {submodule_mode}, expected gitlink 160000",
        )
        return
    extra = f"; {len(symlinks)} unrelated tracked symlink(s)" if symlinks else ""
    record(PASS, "tracked-links", f"no models/outputs symlinks, submodule is a gitlink{extra}")


# --------------------------------------------------------------------------
# c. Payload completeness: nested TTS source survives the staging exclusions.
# --------------------------------------------------------------------------

NESTED_TTS = Path("third_party/Qwen3-TTS/qwen_tts/core/models")
PAYLOAD_WORKSPACE = Path("app/desktop/src-tauri/payload/workspace")


def check_payload() -> None:
    source = ROOT / NESTED_TTS
    if not source.is_dir():
        record(
            FAIL,
            "payload",
            f"{NESTED_TTS} missing from the checkout; submodule not initialised?",
        )
        return
    staged = ROOT / PAYLOAD_WORKSPACE
    if not staged.is_dir():
        record(PASS, "payload", f"{NESTED_TTS} present; no staged payload to compare")
        return
    mirrored = staged / NESTED_TTS
    if not mirrored.is_dir():
        record(
            FAIL,
            "payload",
            f"staged payload dropped {NESTED_TTS}; a name exclusion ate the TTS source",
        )
        return
    count = len(list(mirrored.glob("*.py")))
    if count == 0:
        record(FAIL, "payload", f"staged {NESTED_TTS} exists but holds no .py files")
        return
    record(PASS, "payload", f"staged payload keeps {NESTED_TTS} ({count} modules)")


# --------------------------------------------------------------------------
# d. install.sh sanity.
# --------------------------------------------------------------------------


def check_installer() -> None:
    path = ROOT / "install.sh"
    if not path.exists():
        record(FAIL, "install.sh", "file missing")
        return
    text = path.read_text(encoding="utf-8")
    proc = run(["sh", "-n", str(path)])
    if proc.returncode != 0:
        first = (proc.stderr.strip().splitlines() or ["syntax error"])[0]
        record(FAIL, "install.sh", f"sh -n failed: {first}")
        return
    assigns = [
        line.strip()
        for line in text.splitlines()
        if "TMPDIR=$(" in line and not line.lstrip().startswith("#")
    ]
    if assigns:
        record(
            FAIL,
            "install.sh",
            "assigns TMPDIR=$(...); the launched app inherits it and dies on temp writes",
        )
        return
    if "-mountpoint" not in text:
        record(
            FAIL,
            "install.sh",
            "hdiutil attach has no -mountpoint; parsing the mount path breaks on spaces",
        )
        return
    record(PASS, "install.sh", "sh -n clean, TMPDIR untouched, mounts at a chosen -mountpoint")


# --------------------------------------------------------------------------
# d2. The two from-source installers.
# --------------------------------------------------------------------------

POSIX_INSTALLER = "install-server.sh"
PS_INSTALLER = "install-server.ps1"
# The release asset both installers download, and the one release_dmg.py packs.
# A rename in either place is silent: the installer just 404s and falls back to
# building the UI from source, which works, so nobody notices until someone
# wonders why every install takes an extra three minutes.
UI_ASSET_NAME = "mambodubb-ui-dist.tar.gz"


def check_source_installers() -> None:
    """Windows and Linux install by pasting one of these into a shell.

    Both are consumed through a pipe, which rules out one thing above all others:
    reading stdin. A `read` in the POSIX one eats the rest of its own script, and
    a `Read-Host` in the PowerShell one waits forever on a console that is busy
    feeding it. `sh -n` catches the rest of what a syntax check can catch. The
    PowerShell side gets no parser here, because this machine has no pwsh, so it
    is checked for the two settings that turn a silent half-install into a stop.
    """
    posix = ROOT / POSIX_INSTALLER
    powershell = ROOT / PS_INSTALLER
    for path in (posix, powershell):
        if not path.exists():
            record(FAIL, "source-installers", f"{path.name} missing")
            return

    proc = run(["sh", "-n", str(posix)])
    if proc.returncode != 0:
        first = (proc.stderr.strip().splitlines() or ["syntax error"])[0]
        record(FAIL, "source-installers", f"sh -n {POSIX_INSTALLER}: {first}")
        return

    sh_text = posix.read_text(encoding="utf-8")
    reads = [
        line.strip()
        for line in sh_text.splitlines()
        if re.match(r"^\s*read\s", line) and not line.lstrip().startswith("#")
    ]
    if reads:
        record(FAIL, "source-installers", f"{POSIX_INSTALLER} reads stdin: {reads[0]!r}")
        return

    ps_text = powershell.read_text(encoding="utf-8")
    if "Read-Host" in ps_text:
        record(FAIL, "source-installers", f"{PS_INSTALLER} calls Read-Host; it is piped into iex")
        return
    for needed in ("Set-StrictMode", "$ErrorActionPreference = 'Stop'"):
        if needed not in ps_text:
            record(FAIL, "source-installers", f"{PS_INSTALLER} does not set {needed}")
            return

    release = (ROOT / "scripts" / "release_dmg.py").read_text(encoding="utf-8")
    for rel, text in ((POSIX_INSTALLER, sh_text), (PS_INSTALLER, ps_text),
                      ("scripts/release_dmg.py", release)):
        if UI_ASSET_NAME not in text:
            record(FAIL, "source-installers",
                   f"{rel} no longer names {UI_ASSET_NAME}; the prebuilt UI asset is orphaned")
            return

    record(PASS, "source-installers",
           f"sh -n clean, neither reads stdin, both agree on {UI_ASSET_NAME}")


# --------------------------------------------------------------------------
# e. No em dashes in user-facing text.
# --------------------------------------------------------------------------
#
# Scope: README.md, docs/*.md, install.sh, app/ui/src TS/TSX, and the Python
# that writes user-visible strings (dubbing_app/, dubbing/).
#
# Two honest caveats, both documented in docs/SHIPPING.md:
#   1. Comment lines in .ts/.tsx/.py/.css are stripped by a simple prefix test
#      before scanning. Prose inside a block comment can hide an em dash from
#      this check. That is deliberate: those bytes never reach a user.
#   2. New content FAILS; pre-existing occurrences WARN. A push gate exists to
#      stop new em dashes, not to demand a repo-wide rewrite before the next
#      push. Clearing the WARN backlog is its own task.

# .sh and .ps1 are scanned whole, comments included, the way install.sh always
# has been: an installer is read by the people it fails on, so its prose counts.
SCOPE_SUFFIXES = (".md", ".sh", ".ps1", ".ts", ".tsx", ".py", ".css")
SCOPE_ROOTS = ("README.md", "install.sh", "install-server.sh", "install-server.ps1",
               "docs", "app/ui/src", "dubbing_app", "dubbing")
COMMENT_PREFIXES = ("//", "*", "/*", "#")
CODE_LIKE = (".ts", ".tsx", ".py", ".css")
# An em dash alone in a string is a placeholder glyph for an empty value, not
# prose. `{progress != null ? "12%" : "—"}` is typography, so it is allowed.
GLYPH_ONLY = re.compile(r'^[\s{}()?:,;]*["\'`]\s*' + EM_DASH + r'\s*["\'`][\s{}()?:,;]*$')


def in_scope(rel: str) -> bool:
    if not rel.endswith(SCOPE_SUFFIXES):
        return False
    return any(rel == root or rel.startswith(root + "/") for root in SCOPE_ROOTS)


def offending_lines(rel: str, lines: list[str]) -> list[int]:
    code_like = rel.endswith(CODE_LIKE)
    hits: list[int] = []
    for number, line in enumerate(lines, start=1):
        if EM_DASH not in line:
            continue
        stripped = line.strip()
        if code_like and stripped.startswith(COMMENT_PREFIXES):
            continue
        if GLYPH_ONLY.match(stripped):
            continue
        hits.append(number)
    return hits


def _added_lines_by_file() -> dict[str, list[str]]:
    """Lines this push would add: working tree vs HEAD, plus untracked files."""
    added: dict[str, list[str]] = {}
    diff = git("diff", "HEAD", "--unified=0", "--no-color")
    current: str | None = None
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
        elif line.startswith("+++ "):
            current = None
        elif line.startswith("+") and not line.startswith("+++") and current:
            added.setdefault(current, []).append(line[1:])
    for rel in git("ls-files", "--others", "--exclude-standard").splitlines():
        path = ROOT / rel
        if not path.is_file():
            continue
        try:
            added.setdefault(rel, []).extend(path.read_text(encoding="utf-8").splitlines())
        except (UnicodeDecodeError, OSError):
            continue
    return added


def check_em_dashes() -> None:
    new_offenders: list[str] = []
    for rel, lines in _added_lines_by_file().items():
        if not in_scope(rel):
            continue
        hits = offending_lines(rel, lines)
        if hits:
            new_offenders.append(f"{rel} ({len(hits)})")

    backlog = 0
    backlog_files: list[str] = []
    for rel in git("ls-files").splitlines():
        if not in_scope(rel):
            continue
        path = ROOT / rel
        if not path.is_file():
            continue
        try:
            hits = offending_lines(rel, path.read_text(encoding="utf-8").splitlines())
        except (UnicodeDecodeError, OSError):
            continue
        if hits:
            backlog += len(hits)
            backlog_files.append(rel)

    if new_offenders:
        record(
            FAIL,
            "em-dashes",
            "new em dashes in " + ", ".join(sorted(new_offenders)) + "; rephrase, do not delete",
        )
        return
    if backlog:
        sample = sorted(backlog_files)
        shown = ", ".join(sample[:4])
        rest = f", +{len(sample) - 4} more" if len(sample) > 4 else ""
        record(
            WARN,
            "em-dashes",
            f"nothing new, but {backlog} pre-existing in {len(sample)} file(s): {shown}{rest}",
        )
        return
    record(PASS, "em-dashes", "none in README, docs, installer, UI strings or app prose")


# --------------------------------------------------------------------------
# f. Fast tests, and the UI typecheck.
# --------------------------------------------------------------------------

FAST_SUITE = "tests/test_dubbing.py"


def check_tests() -> None:
    if not TEST_PYTHON.exists():
        record(SKIP, "tests", f"interpreter not found at {TEST_PYTHON}")
        return
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    cmd = [str(TEST_PYTHON), "-m", "pytest", FAST_SUITE, "-q"]
    caveat = ""
    if (ROOT / "models").exists():
        for test in MODELS_SENSITIVE_TESTS:
            cmd += ["--deselect", test]
        caveat = f"; {len(MODELS_SENSITIVE_TESTS)} weights-sensitive test(s) deselected (models/ exists)"
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    summary = ""
    for line in reversed((proc.stdout or "").splitlines()):
        if " passed" in line or " failed" in line or " error" in line:
            summary = line.strip()
            break
    if proc.returncode != 0:
        record(FAIL, "tests", f"{FAST_SUITE} red: {summary or 'see pytest output'}")
        print(proc.stdout[-4000:], file=sys.stderr)
        return
    record(PASS, "tests", f"{summary or FAST_SUITE + ' green'}{caveat}")


def check_typecheck() -> None:
    ui = ROOT / "app" / "ui"
    if not (ui / "node_modules").is_dir():
        record(SKIP, "typecheck", "app/ui/node_modules absent; run pnpm install to enable")
        return
    if not shutil.which("pnpm"):
        record(SKIP, "typecheck", "pnpm not on PATH")
        return
    proc = run(["pnpm", "exec", "tsc", "--noEmit"], cwd=ui)
    if proc.returncode != 0:
        errors = [
            line for line in (proc.stdout + proc.stderr).splitlines() if "error TS" in line
        ]
        first = errors[0].strip() if errors else "tsc failed"
        record(FAIL, "typecheck", f"{len(errors) or '?'} error(s), first: {first}")
        return
    record(PASS, "typecheck", "pnpm exec tsc --noEmit clean")


# --------------------------------------------------------------------------
# g. Release asset freshness. Advisory only.
# --------------------------------------------------------------------------

DMG_DIR = Path("app/desktop/src-tauri/target/release/bundle/dmg")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_dmg_helper() -> None:
    """The Install MamboDubb.command inside a built .dmg must be the repo's
    install.sh, byte for byte. The release-assets digest check cannot see this:
    a stale helper baked into the image matches its own upload perfectly. A DMG
    once shipped carrying the TMPDIR-clobbering installer a full commit after
    the script itself was fixed, because the finish step had run against an
    older tree. Comparing the embedded file to install.sh catches that class."""
    dmgs = sorted((ROOT / DMG_DIR).glob("MamboDubb_*.dmg"))
    if not dmgs:
        record(SKIP, "dmg-helper", "no built .dmg to inspect")
        return
    if sys.platform != "darwin" or not shutil.which("hdiutil"):
        record(SKIP, "dmg-helper", "hdiutil unavailable; cannot open the image")
        return
    dmg = dmgs[-1]
    want = sha256(ROOT / "install.sh")
    proc = run(["hdiutil", "attach", "-nobrowse", "-readonly", str(dmg)])
    if proc.returncode != 0:
        record(FAIL, "dmg-helper", f"could not mount {dmg.name}")
        return
    mount = None
    for line in proc.stdout.splitlines():
        if "/Volumes/" in line:
            mount = Path(line.split("\t")[-1].strip())
    try:
        helper = (mount / "Install MamboDubb.command") if mount else None
        if helper is None or not helper.is_file():
            record(FAIL, "dmg-helper", f"{dmg.name} has no Install MamboDubb.command")
        elif sha256(helper) != want:
            record(FAIL, "dmg-helper",
                   f"{dmg.name} embeds a stale installer; rerun release_dmg.py --finish-unsigned")
        else:
            record(PASS, "dmg-helper", f"{dmg.name} embeds the current install.sh")
    finally:
        if mount:
            run(["hdiutil", "detach", "-quiet", str(mount)])


def check_release_assets() -> None:
    if not shutil.which("gh"):
        record(SKIP, "release-assets", "gh not on PATH; cannot compare with the latest release")
        return
    local_dmgs = sorted((ROOT / DMG_DIR).glob("MamboDubb_*.dmg")) if (ROOT / DMG_DIR).is_dir() else []
    if not local_dmgs:
        record(SKIP, "release-assets", "no local .dmg built; nothing to compare")
        return
    proc = run(["gh", "release", "view", "--json", "tagName,assets"])
    if proc.returncode != 0:
        record(WARN, "release-assets", "gh release view failed (offline or unauthenticated)")
        return
    try:
        release = json.loads(proc.stdout)
    except json.JSONDecodeError:
        record(WARN, "release-assets", "could not parse gh release output")
        return
    tag = release.get("tagName", "?")
    digests = {
        asset["name"]: (asset.get("digest") or "").removeprefix("sha256:")
        for asset in release.get("assets", [])
    }
    notes: list[str] = []
    for dmg in local_dmgs:
        remote = digests.get(dmg.name)
        if not remote:
            notes.append(f"{dmg.name} is not on {tag}")
        elif remote != sha256(dmg):
            notes.append(f"{dmg.name} differs from the {tag} asset")
    installer = ROOT / "install.sh"
    remote_sh = digests.get("install.sh")
    if installer.exists() and remote_sh and remote_sh != sha256(installer):
        notes.append(f"install.sh differs from the {tag} asset")
    # The prebuilt UI. Its absence is not fatal to anything (both source
    # installers fall back to building it), so this is a note and not a red, but
    # a release without it makes every Windows and Linux install minutes slower.
    local_ui = (ROOT / DMG_DIR / UI_ASSET_NAME)
    if local_ui.is_file():
        remote_ui = digests.get(UI_ASSET_NAME)
        if not remote_ui:
            notes.append(f"{UI_ASSET_NAME} is not on {tag}")
        elif remote_ui != sha256(local_ui):
            notes.append(f"{UI_ASSET_NAME} differs from the {tag} asset")
    if notes:
        record(
            WARN,
            "release-assets",
            "; ".join(notes) + "; did you forget to upload?",
        )
        return
    record(PASS, "release-assets", f"local artifacts match the {tag} assets")


# --------------------------------------------------------------------------
# h. Every relative link in README.md resolves on disk.
# --------------------------------------------------------------------------

LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)")


def check_readme_links() -> None:
    readme = ROOT / "README.md"
    if not readme.exists():
        record(FAIL, "readme-links", "README.md missing")
        return
    broken: list[str] = []
    checked = 0
    for target in LINK.findall(readme.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        checked += 1
        path = (ROOT / target.split("#", 1)[0]).resolve()
        if not path.exists():
            broken.append(target)
    if broken:
        record(FAIL, "readme-links", f"{len(broken)} dead relative link(s): {', '.join(broken)}")
        return
    record(PASS, "readme-links", f"all {checked} relative link(s) resolve")


CHECKS = (
    check_versions,
    check_tracked_links,
    check_payload,
    check_installer,
    check_source_installers,
    check_em_dashes,
    check_tests,
    check_typecheck,
    check_dmg_helper,
    check_release_assets,
    check_readme_links,
)


def main() -> int:
    print(f"preflight  {ROOT}\n")
    for check in CHECKS:
        try:
            check()
        except Exception as exc:  # a broken check is a red gate, not a crash
            record(FAIL, check.__name__.removeprefix("check_"), f"check raised {exc!r}")
    failed = [name for status, name, _ in _results if status == FAIL]
    warned = [name for status, name, _ in _results if status == WARN]
    print()
    if failed:
        print(f"FAIL  {len(failed)} check(s) red: {', '.join(failed)}. Nothing ships on a red.")
        return 1
    tail = f" ({len(warned)} warning: {', '.join(warned)})" if warned else ""
    print(f"PASS  every check green{tail}. Clear to push.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
