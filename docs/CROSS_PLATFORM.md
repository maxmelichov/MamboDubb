# MamboDubb on macOS, Windows and Linux

**The short version: the only *bundle* that exists is the macOS one, and `install.sh`
is how you use it. Windows and Linux have no downloadable `.exe`, `.msi`, `.deb` or
AppImage, and cannot get one until the account-level Actions block below is lifted.
What they have instead is a one-command install of the source: `install-server.sh` on
Linux, `install-server.ps1` on Windows. Both end with the full studio serving on
`http://127.0.0.1:4400`, which is the same editor the Mac app wraps.**

| Platform | The one command | What you get |
|---|---|---|
| macOS (arm64) | `curl -fsSL https://github.com/maxmelichov/MamboDubb/releases/latest/download/install.sh \| sh` | the desktop app in `/Applications` |
| Linux (and macOS) | `curl -fsSL https://raw.githubusercontent.com/maxmelichov/MamboDubb/main/install-server.sh \| sh` | the editor at 127.0.0.1:4400 |
| Windows | `powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/maxmelichov/MamboDubb/main/install-server.ps1 \| iex"` | the editor at 127.0.0.1:4400 |

The two source installers are fetched from `raw.githubusercontent.com` and not from a
release asset, deliberately: they install the *source*, so pinning them to a release
would mean uploading a new copy of each on every tag for no benefit. What they install
is pinned, to the tag of the latest release.

The desktop shell is not Mac-only *in the code*. The Rust shell, the payload staging
script and `tauri.conf.json` are per-platform correct, and
[`.github/workflows/build-desktop.yml`](../.github/workflows/build-desktop.yml) is
written to build installers for all three on GitHub's runners.

That workflow has never run. GitHub Actions is disabled at the account level, not per
repository: a manual `gh workflow run` is refused with HTTP 422 "Actions has been
disabled for this user", and the account has zero workflow runs of any kind. The
workflow file is present and valid in the `v0.5.0` tag, and the repository's own Actions
permissions report as enabled, so nothing in this repo needs changing. Only the account
owner can lift the block. Until that happens no `.exe`, `.msi`, `.deb` or `.AppImage`
is produced or attached to any release, and this document describes what those bundles
*would* be, not something you can download.

**Read the status table before you trust anything below it.** Everything here was
written and reviewed on an Apple Silicon Mac. What that machine can prove is that the
code compiles, that its tests pass, and that the Windows/Linux code paths are the right
ones by inspection. What it cannot prove is that a `.exe` installs, or that an AppImage
launches. Those need a real machine, and this document says so where it applies rather
than pretending otherwise.

## Status

| | macOS (arm64) | Windows (x64) | Linux (x64) |
|---|---|---|---|
| Rust shell compiles | ✅ locally | ❌ never built | ❌ never built |
| Unit tests pass | ✅ locally (37) | ❌ never run | ❌ never run |
| Installer produced | ✅ `.dmg` (+ `.app`), on releases | ❌ CI has never run | ❌ CI has never run |
| Installer **launches** | ✅ verified by hand | ❌ nothing to launch | ❌ nothing to launch |
| First-run provisioning | ✅ verified by hand | ❓ unverified | ❓ unverified |
| `uv sync` in the provisioned workspace | ✅ verified | ❓ unverified | ❓ unverified |
| Headless pipeline and studio from source | ✅ | ✅ this is the path, see [WINDOWS.md](WINDOWS.md) | ✅ this is the path (CUDA) |
| **One-command source install** | ✅ `install-server.sh` | ⚠️ `install-server.ps1`, reviewed but never run on Windows | ✅ `install-server.sh`, run end to end in `ubuntu:22.04` |
| Signed / notarized | ⚠️ ad-hoc signed only, see [RELEASING.md](RELEASING.md) | n/a | n/a |

✅ verified · ❓ needs a real machine and an artifact to test · ❌ has not happened

Intel Macs are **not** in the matrix: `macos-latest` is Apple Silicon, so the `.dmg` is
arm64-only. Staging supports `--target x86_64-apple-darwin`, so adding an Intel (or
universal) row is a matrix entry away when someone has a machine to test it on.

## What the release actually carries

The [v0.5.0 release](https://github.com/maxmelichov/MamboDubb/releases/tag/v0.5.0) has
exactly two assets, both built by hand on a Mac:

| Asset | For |
|---|---|
| `install.sh` | The macOS install. Fetches the `.dmg` below and does the rest. |
| `MamboDubb_0.5.0_aarch64.dmg` | The macOS bundle `install.sh` installs. |

The `.dmg` is attached because `install.sh` downloads it; it is not meant to be
downloaded by hand, and the README no longer offers it that way. Windows and Linux
users run from source: [SERVER.md](SERVER.md), [WINDOWS.md](WINDOWS.md).

**From the next release there is a third asset, `mambodubb-ui-dist.tar.gz`**: the
built `app/ui/dist`, so the two source installers can put a working editor in place
without asking anyone to install Node and pnpm. `scripts/release_dmg.py` packs it
next to the `.dmg` on every run, including `--finish-unsigned`, so it cannot be
forgotten; upload it alongside the other two. v0.5.0 predates it, and both
installers handle that by building the UI from a private Node they download into
`.tools/` inside the checkout, which is slower but needs nothing on the machine.

Were Actions enabled, pushing a `v*` tag would run `build-desktop.yml`, whose last step
uploads every bundle it produces to that tag's release with `gh release upload
--clobber`, and a `workflow_dispatch` run on a branch would attach
`mambodubb-macos-aarch64`, `mambodubb-windows-x86_64` and `mambodubb-linux-x86_64` as
30-day `.zip` artifacts. Neither has ever happened.

### macOS

Install with:

```bash
curl -fsSL https://github.com/maxmelichov/MamboDubb/releases/latest/download/install.sh | sh
```

That is the only path with no Gatekeeper dialogs, because `curl` sets no quarantine
bit. Everything a browser downloads is quarantined, and the bundle is ad-hoc-signed
rather than notarized, so on a downloaded `.dmg` **both** of the obvious moves are
refused: dragging `MamboDubb.app` to Applications, and double-clicking **Install
MamboDubb** inside the volume. macOS 26 shows "Apple could not verify..." with only
**Move to Trash** and **Done**, no Open Anyway button in the dialog itself.

Two ways past it, for a `.dmg` already on disk:

```bash
sh install.sh /path/to/MamboDubb_*.dmg          # from the attached install.sh
sh "/Volumes/MamboDubb/Install MamboDubb.command"   # the same script, inside the .dmg
```

or take the refusal once and then press **Open Anyway** in System Settings >
Privacy & Security. The script copies the app to `/Applications`, clears quarantine,
and ad-hoc-signs the bundle so it launches. A release where dragging to Applications
just works needs Developer ID signing and notarization through
`scripts/release_dmg.py`; see [RELEASING.md](RELEASING.md).

### Windows

There is no bundle to download. The install is the source, in one command:

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/maxmelichov/MamboDubb/main/install-server.ps1 | iex"
```

`install-server.ps1` installs `git`, `ffmpeg` and `sox` through the winget ids
`dubbing/tools.py` names, clones the latest release tag with its submodule into
`%USERPROFILE%\MamboDubb`, downloads `uv` from GitHub with its published SHA-256
verified, runs `uv sync --extra app` (plus `--extra cuda` when `nvidia-smi` says
there is a card), puts the built web UI in `app\ui\dist`, and serves on 4400. It
reads nothing from the keyboard. It used to print the CUDA `torch` swap as advice
and leave it to you; that advice named an index with no build of this project's
torch, and `uv pip install` was undone by the next `uv run` anyway, so the CUDA
wheels are a declared extra now ([WINDOWS.md](WINDOWS.md#cuda-wheels)).

**That script has never been run on Windows.** No machine here can run it, and this
document does not pretend otherwise. What has been done: it parses clean under
PowerShell 7.6 and PSScriptAnalyzer reports nothing above informational, and every
step of it except the winget block, the `uv` download and the Node download has been
executed on a POSIX host with `$env:OS` spoofed to `Windows_NT`, which covers the
release-tag lookup, the clone with submodule, `uv sync --extra app`, the prebuilt-UI
unpack, and the fallback UI build. The untested parts are the Windows-only ones.
[WINDOWS.md](WINDOWS.md) has the same steps by hand for anyone who would rather
watch them happen.

If the bundle build ever runs, it produces an NSIS `-setup.exe` (per-user, no admin prompt) and
an `.msi`, both unsigned, so **SmartScreen** would show "Windows protected your PC" on
first run: click **More info → Run anyway**. Some corporate policies block unsigned
installers outright, and there is no workaround short of code signing (open item below).

WebView2 is present on every Windows 10 1803+ and Windows 11 machine. The installer is
configured with `downloadBootstrapper`, so on the rare machine without it, it is fetched
during install, which means the install needs a network connection.

The provisioned workspace lives in `%LOCALAPPDATA%\MamboDubb\workspace`, deliberately
**not** `%APPDATA%`: that directory grows a ~10 GB `.venv` plus model weights, and
`%APPDATA%` roams to the domain server on a managed machine.

For the GPU story, `ffmpeg`/`sox`, and running from source, [WINDOWS.md](WINDOWS.md) is
the detailed guide and it applies to the packaged app too.

### Linux

There is no `.deb` and no AppImage to download. The install is the source, in one
command:

```bash
curl -fsSL https://raw.githubusercontent.com/maxmelichov/MamboDubb/main/install-server.sh | sh
```

`install-server.sh` detects the distro's package manager, clones the latest release
tag with its submodule into `~/MamboDubb`, downloads `uv` from GitHub with its
published SHA-256 verified, runs `uv sync --extra app`, puts the built web UI in
`app/ui/dist`, and serves on 4400. It never reads stdin, so it is safe in a pipe.

`ffmpeg`, `sox` and `git` are the one place it has to be careful. It installs them
itself only where the package manager needs no password, which means root or a
`sudo` that already answers `sudo -n true`; otherwise it prints the exact line for
that distro and carries on, because a hidden password prompt behind a pipe is a
hang and not an install. That is the same rule `dubbing/tools.py` applies to the
app's Setup button, and it has to be the same rule, or the installer and the app
disagree about the same machine. `git` is the exception that stops the run: without
it there is no way to get `third_party/Qwen3-TTS`, and `pyproject.toml` pins that
submodule as a path dependency, so a checkout without it can never `uv sync`.

This has been run end to end in a clean `ubuntu:22.04` container: package install,
clone with submodule, `uv` install with its checksum verified, `uv sync --extra app`,
both routes to a built UI, and the server answering `/health` and serving
`index.html` on 4400. Models were not downloaded; that is Setup's 25 GB and no part
of the install.

The rest of this section describes what the *bundle* build would produce if it ran.

The `.deb` would be the better install on Debian/Ubuntu:

```bash
sudo apt install ./MamboDubb_*_amd64.deb
```

The AppImage needs the executable bit, which the download does not preserve:

```bash
chmod +x MamboDubb_*_amd64.AppImage
./MamboDubb_*_amd64.AppImage
```

Built on Ubuntu 22.04, so glibc 2.35 or newer. Both bundles need
`libwebkit2gtk-4.1-0` (the `.deb` declares it; for the AppImage, install
`libwebkit2gtk-4.1-0` yourself if the window never appears). "Reveal in file manager"
goes through the freedesktop `FileManager1` D-Bus interface, falling back to opening the
containing directory with `xdg-open`; on a session with neither, the button does
nothing.

The provisioned workspace lives in `~/.local/share/MamboDubb/workspace`.

## What is per-platform in the code

- **`workspace.rs`**: `uv` is found via `DUBSTUDIO_UV_PATH`, then the bundled sidecar
  (`uv` / `uv.exe` next to the app binary), then platform install paths (Homebrew on
  macOS, `/usr/{local/,}bin` and linuxbrew on Linux, nothing on Windows), then `PATH`
  (`.exe` first there), then `~/.local/bin` and `~/.cargo/bin`. The home directory is
  `HOME` **or `USERPROFILE`**, because a Windows GUI process only ever gets the latter.
- **`dubbing_app/setup.py`**: `find_uv()` is the same chain, minus the sidecar rung,
  which it cannot resolve (its `sys.executable` is the venv's Python, nowhere near the
  shell) and does not need to: `runner/process.rs` hands the resolved sidecar down as
  `DUBSTUDIO_UV_PATH`, which is rung one. The two are held in step by a test that reads
  the constants straight out of `workspace.rs`
  (`tests/test_windows_portability.py::test_the_python_uv_chain_still_matches_the_rust_one_it_claims_to`).
  The uv row is graded per machine (`setup.uv_row`), because the answer genuinely is.
  On a Mac the MLX backend loads the translator in the server's own process and no
  stage of a dub shells out to uv, so the row is `optional` and a missing uv is never
  a reason that machine cannot dub. On Windows and on Linux the translate stage is
  `uv run --project translator`, so the row is `blocking` and names the stage it
  kills: without uv, a dub fetches, separates and transcribes and then dies at
  translate. `dubbing/tools.py` owns the lookup so both the pipeline and the app can
  reach it; `setup.find_uv` is a re-export of that one function. Its Install button is `brew` or `winget` where
  either exists, and otherwise the official `astral-sh/uv` release archive with its
  published SHA-256 verified, unpacked into `~/.local/bin`, which is the same route and
  the same
  destination as `install-server.sh` and `install-server.ps1`, so a machine ends up with
  one uv however it was installed.
- **The caches**: `hf_hub_cache()` and the Demucs probe resolve `XDG_CACHE_HOME` before
  `~/.cache`, because `huggingface_hub` and `torch.hub` both do, on every platform.
- **`provision.rs`**: the writable workspace copy goes under the platform's *local*
  data dir (see the Windows note above). Copies force owner-write on every file,
  because the source sits inside a read-only bundle.
- **`runner/process.rs`**: spawns `uv run --project <workspace> …` with
  `CREATE_NO_WINDOW` on Windows so no console flashes up, and holds the child's stdin
  open as the liveness channel (`--exit-on-stdin-close`) on all three.
- **`files.rs`**: reveal/open dialogs go through `tauri-plugin-opener` and
  `tauri-plugin-dialog`, both of which have all three backends.
- **`scripts/stage_desktop_payload.py --target <triple>`**: copies a locally installed
  `uv` when the target is this machine (the fast path for local rebuilds), and otherwise
  downloads the official `astral-sh/uv` build for that triple and verifies its published
  SHA-256 before writing `binaries/uv-<triple>[.exe]`.
- **`tauri.conf.json`**: `bundle.targets` lists all six formats; Tauri intersects that
  list with what the host can build, so each runner produces only its own. `rpm` is
  deliberately absent: nobody has tested it.
- **The Python pipeline**: already portable. `dubbing/tools.py` names the right install
  command per OS (`brew` / `winget` / `apt-get`), resolves tools with a `.exe` suffix on
  Windows, and forces UTF-8 stdio there. `pyproject.toml` markers keep `mlx-lm` on macOS
  only. Device selection is `cuda` → `mps` → `cpu` everywhere it appears
  (`dubbing/segments.py`, `dubbing/tts.py`); there is no hard MPS assumption. Loading
  the CUDA libraries is `dubbing/nvlibs.py`, which is one module for three platforms:
  Linux preloads the wheel's cuDNN by absolute path so a mismatched system copy cannot
  answer, and Windows hands the wheel directories to `os.add_dll_directory` because
  since Python 3.8 the loader does not search `PATH` for an extension module's
  dependent DLLs. That second one is why a Windows box with CUDA installed system wide
  still reported `Library cublas64_12.dll is not found or cannot be loaded`.

## Open items

1. **No code signing anywhere.** macOS has a documented signing path
   ([RELEASING.md](RELEASING.md)) that CI does not run. The workflow has a loud
   `TODO(signing)` where the env vars go. Windows has no certificate at all, so
   SmartScreen will warn on every build until someone buys an OV/EV cert. Linux
   packages are conventionally unsigned; the `.deb`/`.AppImage` would be fine as-is.
2. **Inference speed off a GPU.** The pipeline picks CUDA if it is there, then MPS, then
   CPU. A Windows or Linux box with an NVIDIA card is the good case, but PyPI's default
   `torch` wheel is **CPU-only on Windows**, so the CUDA build comes from the `cuda`
   extra ([WINDOWS.md](WINDOWS.md#cuda-wheels)) and without it every model runs on the
   CPU: one real run spent sixteen hours on stem separation alone. On a CPU-only machine
   of any OS the app *works* and is unusably slow for anything but a short clip. The
   silence is fixed in three places now: `install-server.ps1` syncs the extra when
   `nvidia-smi` is present and says whether CUDA came back available, the run prints a
   warning at startup and again at the stem stage when the driver is there and torch is
   a CPU build, and Setup carries a **GPU acceleration** row on machines with a driver.
3. **No Windows or Linux *bundle* exists, because CI has never run.** The source
   installers above close the gap for users, but a `.exe` and a `.deb` are still the
   thing most people expect, and lifting the account-level Actions block is what
   unblocks everything else here. After that they
   still need a real machine to verify: the installer completes; the app
   window opens (WebView2 / WebKitGTK); first-run provisioning writes the workspace to
   the local data dir; `uv sync` completes there (~10 GB, and the Qwen3-TTS editable
   install is the fragile part); `ffmpeg`/`sox` detection and the Setup screen's install
   button; the file dialogs and "reveal in file manager".
4. **The AppImage bundles no `libwebkit2gtk`.** It is an AppImage in name only, and still
   depends on the host's WebKitGTK.
5. **`install-server.ps1` has never run on Windows.** It is reviewed, parsed and
   analyzer-clean, and most of it has been executed on a spoofed host, but the
   winget block, the `uv` zip download and the Node zip download are Windows-only
   code paths that nothing here can exercise. First Windows machine to touch this
   repo should run it and report back.
6. **`translator/` is a separate venv** with its own CUDA story on Linux/Windows; the
   desktop shell does not provision or manage it.
7. **Bundle size.** The uv sidecar is 35 to 59 MB depending on target, and the payload adds
   ~44 MB: ~11 MB of source and 31 MB of bundled diarization weights (`third_party/
   pyannote-speaker-diarization-community-1`, CC-BY-4.0, shipped so a fresh install
   tells speakers apart with no Hugging Face account). Nothing is stripped.
