# MamboDubb on macOS, Windows and Linux

**The short version: macOS is the only OS with an installer. On Windows and Linux you
run from source, and that path works today.**

The desktop shell is not Mac-only *in the code*. The Rust shell, the payload staging
script and `tauri.conf.json` are per-platform correct, and
[`.github/workflows/build-desktop.yml`](../.github/workflows/build-desktop.yml) is
written to build installers for all three on GitHub's runners. That workflow has never
run: GitHub Actions is disabled on the account, so no Windows or Linux artifact has ever
been produced. The releases page carries the Mac `.dmg` and `install.sh`, and nothing
else. There is no date for that changing.

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
| Installer produced | ✅ `.dmg` (+ `.app`), on releases | ❌ none exists | ❌ none exists |
| Installer **launches** | ✅ verified by hand | ❓ nothing to launch | ❓ nothing to launch |
| First-run provisioning | ✅ verified by hand | ❓ unverified | ❓ unverified |
| `uv sync` in the provisioned workspace | ✅ verified | ❓ unverified | ❓ unverified |
| Headless pipeline and studio from source | ✅ | ✅ this is the path, see [WINDOWS.md](WINDOWS.md) | ✅ this is the path (CUDA) |
| Signed / notarized | ⚠️ ad-hoc signed only, see [RELEASING.md](RELEASING.md) | n/a | n/a |

✅ verified · ❓ needs a real machine and an artifact to test · ❌ has not happened

Intel Macs are **not** in the matrix: `macos-latest` is Apple Silicon, so the `.dmg` is
arm64-only. Staging supports `--target x86_64-apple-darwin`, so adding an Intel (or
universal) row is a matrix entry away when someone has a machine to test it on.

## Getting a build

**macOS:** the [latest release](https://github.com/maxmelichov/MamboDubb/releases/latest)
carries `MamboDubb_<version>_aarch64.dmg` and `install.sh`. Use the one-liner below.

**Windows and Linux:** there is nothing to download. Run from source:

```bash
git clone --recurse-submodules https://github.com/maxmelichov/MamboDubb.git
cd MamboDubb && uv sync
cd app/ui && pnpm install && pnpm build && cd ../..
uv run mambodubb          # the full editor at http://127.0.0.1:4400
```

That is the same server the Mac app wraps, so you get the whole studio in a browser.
Windows has extra steps, mostly around CUDA wheels: [WINDOWS.md](WINDOWS.md), and WSL2
is the easier road. An NVIDIA GPU is what you want on either OS; `translate.load()`
switches from MLX to the CUDA worker in `translator/` on its own.

If someone re-enables Actions, `build-desktop.yml` is ready: push a `v*` tag or run the
workflow by hand, and each job uploads `mambodubb-macos-aarch64`,
`mambodubb-windows-x86_64` (`-setup.exe`, `.msi`) and `mambodubb-linux-x86_64` (`.deb`,
`.AppImage`) as a `.zip` artifact that expires after 30 days. Nobody has done that yet,
so the two sections below are what the code intends, not what anyone has seen work.

## Per-OS install notes

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

No installer exists. Run from source, as above and in [WINDOWS.md](WINDOWS.md). The rest
of this section describes a packaged build nobody has produced yet.

It would install from the NSIS `-setup.exe` (per-user, no admin prompt) or the `.msi`.
Both would be unsigned, so **SmartScreen** shows "Windows protected your PC" on first
run: click **More info → Run anyway**. Some corporate policies block unsigned installers
outright, and there is no workaround short of code signing (open item below).

WebView2 is present on every Windows 10 1803+ and Windows 11 machine. The installer is
configured with `downloadBootstrapper`, so on the rare machine without it, it is fetched
during install, which means the install needs a network connection.

The provisioned workspace lives in `%LOCALAPPDATA%\MamboDubb\workspace`, deliberately
**not** `%APPDATA%`: that directory grows a ~10 GB `.venv` plus model weights, and
`%APPDATA%` roams to the domain server on a managed machine.

For the GPU story, `ffmpeg`/`sox`, and running from source, [WINDOWS.md](WINDOWS.md) is
the detailed guide and it applies to the packaged app too.

### Linux

No `.deb` and no AppImage exist. Run from source, as above. The rest of this section
describes a packaged build nobody has produced yet.

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
  (`dubbing/segments.py`, `dubbing/tts.py`); there is no hard MPS assumption.

## Open items

1. **No code signing anywhere.** macOS has a documented signing path
   ([RELEASING.md](RELEASING.md)) that CI does not run. The workflow has a loud
   `TODO(signing)` where the env vars go. Windows has no certificate at all, so
   SmartScreen will warn on every build until someone buys an OV/EV cert. Linux
   packages are conventionally unsigned; the `.deb`/`.AppImage` are fine as-is.
2. **Inference speed off a GPU.** The pipeline picks CUDA if it is there, then MPS, then
   CPU. A Windows or Linux box with an NVIDIA card is the good case, but PyPI's default
   `torch` wheel is **CPU-only on Windows**, so a Windows user must install the CUDA
   build explicitly ([WINDOWS.md](WINDOWS.md#cuda-wheels)) or every model runs on the CPU
   and a five-minute video takes hours. On a CPU-only machine of any OS the app *works*
   and is unusably slow for anything but a short clip. The app does not currently warn
   about this at setup time.
3. **Nobody has built the Windows or Linux installers, let alone run them.** Actions is
   disabled on the account, so the first task is a machine that can build them at all.
   Then the ❓ rows above: the installer completes; the app
   window opens (WebView2 / WebKitGTK); first-run provisioning writes the workspace to
   the local data dir; `uv sync` completes there (~10 GB, and the Qwen3-TTS editable
   install is the fragile part); `ffmpeg`/`sox` detection and the Setup screen's install
   button; the file dialogs and "reveal in file manager".
4. **The AppImage bundles no `libwebkit2gtk`.** It is an AppImage in name only, and still
   depends on the host's WebKitGTK.
5. **`translator/` is a separate venv** with its own CUDA story on Linux/Windows; the
   desktop shell does not provision or manage it.
6. **Bundle size.** The uv sidecar is 35 to 59 MB depending on target, and the payload adds
   ~44 MB: ~11 MB of source and 31 MB of bundled diarization weights (`third_party/
   pyannote-speaker-diarization-community-1`, CC-BY-4.0, shipped so a fresh install
   tells speakers apart with no Hugging Face account). Nothing is stripped.
