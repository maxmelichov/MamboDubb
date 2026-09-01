# Running MamboDubb on Windows

Both halves, the headless pipeline (`python -m dubbing`) and the studio server
(`mambodubb`), run on native Windows 10/11 with an NVIDIA GPU.

This page is the **run-from-source** guide, and on Windows that is the only way
in: the releases page carries no `-setup.exe` and no `.msi`. The workflow that
would build them has never run, because Actions is disabled on the account
(details in [CROSS_PLATFORM.md](CROSS_PLATFORM.md)). If that changes, everything
below still applies to the packaged app, because it provisions and runs this
same source tree.

You end up with the full studio in a browser at `http://127.0.0.1:4400`, the same
editor the Mac app wraps.

**The short version:** if you have WSL2, use it (see the last section). Native
Windows works and is supported, but the translator runs on a plain transformers
worker there instead of vLLM, which is slower per line.

---

## 0. The one command

Everything in sections 1 to 3 is what `install-server.ps1` does for you:

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/maxmelichov/MamboDubb/main/install-server.ps1 | iex"
```

It installs `git`, `ffmpeg` and `sox` through winget (the same three ids as
below, because `dubbing/tools.py` owns that table and the Setup screen reads the
same one), clones the latest release tag with its submodule into
`%USERPROFILE%\MamboDubb`, downloads `uv` from GitHub with its published SHA-256
checked, runs `uv sync --extra app`, puts the built web UI in `app\ui\dist`
without needing Node on the machine, and starts the server. Rerunning it updates
an existing checkout in place. It reads nothing from the keyboard, so it is safe
to leave alone.

If `nvidia-smi` is on the machine it also syncs the `cuda` extra, which is the
CUDA build of `torch` plus the cuBLAS and cuDNN wheels CTranslate2 needs. Without
that, every model runs on the CPU and nothing says so (see [CUDA
wheels](#cuda-wheels)). The script prints whether `torch.cuda.is_available()`
came back true, so a GPU install that did not take is visible at the end of the
install rather than at the end of a sixteen-hour dub.

Nothing above downloads a model. That is **Setup → Install everything** on first
run, about 25 GB, and it resumes if you interrupt it.

The rest of this page is what to do when you want the steps by hand, and the
reference for everything that behaves differently on Windows.

---

## 1. Prerequisites

```powershell
winget install --id astral-sh.uv -e
winget install --id Git.Git -e
winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
winget install --id ChrisBagwell.SoX -e --accept-source-agreements --accept-package-agreements
```

Open a **new** terminal afterwards: winget puts these on PATH for processes
started after it, and the Setup screen in the app re-probes `PATH` at startup, so
a server launched before the install still cannot see `ffmpeg.exe`.

The Setup screen offers an **Install** button for `ffmpeg` and `sox` on Windows;
it runs exactly the two `winget` lines above and nothing else.

Python comes from `uv` (3.12), so no separate Python install is needed.

## 2. The checkout

```powershell
git clone --recurse-submodules https://github.com/maxmelichov/MamboDubb.git
cd MamboDubb
uv sync --extra app --extra cuda      # drop `--extra cuda` if there is no NVIDIA card
copy .env.example .env      # nothing in it is required; diarization ships with the app
```

`third_party/Qwen3-TTS` is a submodule; `uv sync` fails with "does not appear to
be a Python project" if you cloned without `--recurse-submodules`
(`git submodule update --init` fixes it).

### CUDA wheels

PyPI's default `torch` wheel is **CPU-only on Windows** (on Linux it is a CUDA
build). Nothing about that is an error: the pipeline runs, every model loads,
and a video that should take twenty minutes takes most of a day. One real run
spent 57576 seconds, which is sixteen hours, on stem separation alone.

The `cuda` extra is the fix, and it is one flag:

```powershell
uv sync --extra app --extra cuda
```

It installs two things that fail for two different reasons:

* **The CUDA build of `torch`** (cu126, from `download.pytorch.org`), which is
  what Demucs, pyannote, the speaker-embedding model and Qwen3-TTS all run on.
  This is the one that costs the sixteen hours.
* **`nvidia-cublas-cu12` and `nvidia-cudnn-cu12`**, which are for
  faster-whisper. It runs on CTranslate2, whose Windows wheel bundles neither:
  it loads `cublas64_12.dll` by name at the first forward pass, and without
  these wheels the ASR reports `cuda unusable (Library cublas64_12.dll is not
  found or cannot be loaded)` and drops to the CPU. Installing CUDA system wide
  does **not** fix that, because since Python 3.8 the loader no longer searches
  `PATH` for an extension module's dependent DLLs; `dubbing/nvlibs.py` is what
  registers the wheel directories with `os.add_dll_directory` at startup.

Keep `--extra app --extra cuda` on every `uv sync` **and every `uv run`**. A bare
one syncs to the lockfile as if the extra had never been asked for, and removes
what it brought: the CUDA `torchaudio` and the cuBLAS and cuDNN wheels the speech
recogniser loads. For the same reason, do not add anything with `uv pip install`
and expect it to last: `uv run` re-syncs the environment to the lockfile before
it runs anything, and whatever the lock does not name is removed at the next
launch. (This page used to recommend exactly that for torch, with a `cu124` index
that has no build of this project's torch.)

The translator has its own venv (`translator/`) with its own torch. Its lockfile
sends Windows to the CUDA build of torch on its own, with no flag to remember:
the installer syncs it, and a bare `uv run --project translator` gets the card.
By hand:

```powershell
uv sync --project translator
```

Check with `uv run --no-sync python -c "import torch; print(torch.cuda.is_available())"`,
or open **Setup** in the studio: the **GPU acceleration** row says which torch
build is installed, and appears only on machines that have an NVIDIA driver.

## 3. Run it

```powershell
uv run python -m dubbing "https://www.youtube.com/watch?v=VIDEO_ID"
uv run python -m dubbing input.mp4 --duration 300
```

The studio server, with the UI built once (Node 20+ and pnpm, via `winget install
--id OpenJS.NodeJS.LTS -e`, then `npm i -g pnpm`):

```powershell
cd app\ui; pnpm install; pnpm build; cd ..\..
uv run --extra app --extra cuda mambodubb --port 4400   # → http://127.0.0.1:4400
```

Node is only needed for that first line, and only when the release you installed
carries no prebuilt `mambodubb-ui-dist.tar.gz`. Section 0's script handles both
cases without touching your PATH: it unpacks the release asset when there is one,
and otherwise puts a private Node under `.tools\` inside the checkout and builds
with that.

## 4. What is different on Windows

| | Behaviour |
|---|---|
| **Translation backend** | No MLX (Apple-only) and no vLLM (no native Windows build), so `translate.load()` starts the transformers CUDA worker in `translator/` automatically. Nothing to configure. |
| **Multi-GPU** | Unchanged: with two or more CUDA devices the translator pins itself to the last one via `CUDA_VISIBLE_DEVICES` and stays resident. |
| **Cancelling a job** | The job child gets its own process group; cancel sends Ctrl-Break and then `taskkill /F /T`, so the ffmpeg and yt-dlp it spawned die with it. |
| **Server lifetime** | Windows does not reparent orphans, so the ppid watchdog is disabled there. A launcher that wants the server to die with it should hold the server's stdin and pass `--exit-on-stdin-close`. |
| **Console encoding** | The CLI, the server and both workers force UTF-8 on stdio; a `chcp 1252` console cannot otherwise print a Hebrew line. |
| **Tool lookup** | `ffmpeg`/`ffprobe`/`sox` are resolved with `shutil.which`, which honours `PATHEXT`, so the `.exe` suffix is never assumed or spelled out. |

Everything else is identical: the manifest, the caching, every pipeline
invariant (see [APP_ARCHITECTURE.md](APP_ARCHITECTURE.md), Non-negotiables). Run directories are portable between
platforms except for absolute paths recorded in `m["files"]["video"]` for runs
whose input lived outside the run directory.

## 5. Known rough edges

- **Long paths.** Deep run directories under a long checkout path can exceed the
  legacy 260-character limit. Enable long paths
  (`Computer\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled = 1`)
  or keep the checkout near the drive root.
- **Antivirus.** Real-time scanning of `outputs/` slows the mix stage noticeably;
  excluding the run directory is worth it.
- **`sox` on PATH.** Qwen3-TTS shells out to it during text normalization; a
  missing `sox.exe` fails inside the tts stage, not at startup. The Setup screen
  says so before you start.

## 6. WSL2, the recommended path

WSL2 with an NVIDIA driver gives you the Linux setup exactly: vLLM is available
(much faster translation), the cuDNN handling in `dubbing/nvlibs.py` applies, and
the whole of [docs/setup.md](setup.md) reads literally.

```powershell
wsl --install -d Ubuntu
```

Then inside Ubuntu, the Linux one-liner does the rest, and its default install
directory is already the right one:

```bash
curl -fsSL https://raw.githubusercontent.com/maxmelichov/MamboDubb/main/install-server.sh | sh
```

Keep the checkout **in the Linux filesystem** (`~/MamboDubb`, which is where that
script puts it, and not `/mnt/c/...`, because the 9p mount makes every stage
several times slower). The server binds inside WSL; open `http://127.0.0.1:4400`
from Windows as usual (WSL2 forwards loopback).
