# Shipping

**Run `uv run scripts/preflight.py` before every push. Nothing ships on a red.**

The script is stdlib only, so `python3 scripts/preflight.py` and
`uv run --no-project scripts/preflight.py` are the same run and neither touches
`uv.lock`. Plain `uv run` syncs the workspace first, which is slower and can
rewrite one line of the lockfile; use `--no-project` when you want the gate and
nothing else. It
prints PASS, FAIL, WARN or SKIP per check with one line of reason, and exits
nonzero if anything is red. A WARN is advisory and does not block; a SKIP means
the check could not run here and says why.

## The ship ritual, in order

Each step is one line. Do not reorder them; several later steps assume an
earlier one already happened.

```bash
# 1. The gate. Red means stop.
uv run scripts/preflight.py

# 2. Bump the version in all five places at once (pyproject.toml, dubbing_app/__init__.py,
#    app/desktop/package.json, src-tauri/Cargo.toml, src-tauri/tauri.conf.json).
#    A version bump is also the ONLY supported way to force a workspace refresh.

# 3. Stage the Python workspace into the app bundle.
uv run --script scripts/stage_desktop_payload.py

# 4. Build the desktop app.
cd app/desktop && pnpm tauri build

# 5. Ad-hoc-sign the bundle and inject "Install MamboDubb.command" into the .dmg.
uv run --script scripts/release_dmg.py --finish-unsigned

# 6. Swap the installed copy so you are testing what users get.
rm -rf /Applications/MamboDubb.app && cp -R app/desktop/src-tauri/target/release/bundle/macos/MamboDubb.app /Applications/

# 7. Re-run the gate: it now compares the fresh .dmg and install.sh against the last release.
uv run scripts/preflight.py

# 8. Cut the release, uploading BOTH assets whenever the installer changed.
gh release create v0.4.0 app/desktop/src-tauri/target/release/bundle/dmg/MamboDubb_0.4.0_aarch64.dmg install.sh
```

Signed and notarized releases run `uv run --script scripts/release_dmg.py` at
step 5 instead. See [RELEASING.md](RELEASING.md) for the certificate setup.

### Two rules that are not optional

**Force a workspace refresh only by bumping the version.** The app stamps a
`.mambodubb-payload` marker at the provisioned workspace root with the version
it copied, and refreshes payload-owned entries when that marker is stale. Hand
editing or deleting the marker to trigger a refresh is the trick that pushed a
truncated TTS payload into a live workspace: the marker said "refresh from this
payload" and the payload was the one missing `qwen_tts/core/models`. A version
bump makes the refresh follow a build you actually staged and checked.

**Upload the .dmg and install.sh together on any installer change.** They are
separate release assets and they have drifted before, leaving the published
one-liner installing an app the .dmg no longer matched. Preflight compares both
local files against the latest release digests and warns on a mismatch.

## Why each check exists

| Check | What it asserts | The incident that earned it |
| --- | --- | --- |
| `versions` | The same version string in pyproject.toml, `dubbing_app/__init__.py`, package.json, Cargo.toml and tauri.conf.json | The five declarations drifted, so the .dmg name, the bundle version and the payload marker disagreed about what shipped |
| `tracked-links` | No tracked symlink named `models` or `outputs`, and `third_party/Qwen3-TTS` is a gitlink (mode 160000) | A release commit swept the local `models` and `outputs` symlinks into main, committing paths that exist only on one machine |
| `payload` | `third_party/Qwen3-TTS/qwen_tts/core/models` exists in the checkout, and in the staged payload when one is present | Issue #14: a `models` name exclusion in the staging script matched at any depth and deleted the TTS source from the payload |
| `install.sh` | `sh -n` is clean, no `TMPDIR=$(` assignment, `hdiutil attach` uses `-mountpoint` | The installer overwrote the exported `TMPDIR`, the launched app inherited it and died writing temp files (fixed in 36d4989). Separately, issue #13: parsing the mount path out of `hdiutil` with `awk $NF` broke on `/Volumes/MamboDubb 2` |
| `em-dashes` | No new em dashes in user-facing text | A standing style rule that keeps getting reintroduced by edits to prose |
| `tests` | `tests/test_dubbing.py` is green | The fast pure-logic suite, run because it costs seconds and catches the pipeline regressions that model-free logic can catch |
| `typecheck` | `pnpm exec tsc --noEmit` is clean in app/ui | Type errors do not stop a Vite build, so they reach the shipped bundle |
| `release-assets` | The local .dmg and install.sh match the latest release digests | The .dmg and the installer drifted apart across two uploads |
| `readme-links` | Every relative link target in README.md exists on disk | A README rewrite pointed at docs files that only got created afterwards |

## Caveats worth knowing

**The em dash check FAILs on new content and WARNs on old.** New means lines
this push would add: the working tree diff against HEAD, plus untracked files.
A push gate exists to stop new em dashes reaching users, not to demand a
repo-wide rewrite before the next push. Clearing the WARN backlog is its own
task. Scope is README.md, `docs/`, `install.sh`, `app/ui/src` TS, TSX and CSS,
and `dubbing/` and `dubbing_app/` Python. In those code files, lines whose
first characters are `//`, `*`, `/*` or `#` are stripped before scanning, since
those bytes never reach a user; an em dash inside a block comment can therefore
hide from this check. A string that is only an em dash, the placeholder glyph
for an empty value, is allowed.

**Some tests assume no local weights.** `test_low_vram_swaps_the_mlx_build_and_nothing_else`
resolves a model path and expects the Hugging Face hub id, which is wrong in
any checkout that has a `models/` tree. Preflight deselects it by name when
`models/` exists and says so in the PASS line, rather than failing a gate over
a development machine having weights on it.

**A dev tree with local symlinks is fine.** `models` and `outputs` are symlinks
into the real working directory here, and `.env` exists. Preflight only reads
`git ls-files`, so untracked local symlinks are invisible to it. Only a tracked
one is a failure.

**The interpreter for tests is overridable.** Preflight uses the checkout's own
`.venv/bin/python` by default, the one `uv sync` makes, and skips the check with
a reason if that path is absent. Set `MAMBODUBB_TEST_PYTHON` to point it
somewhere else.

**Release freshness needs `gh` and network.** Without either, that check prints
SKIP or WARN and never blocks the push.
