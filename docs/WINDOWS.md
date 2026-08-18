# Running MamboDubb on Windows

Both halves — the headless pipeline (`python -m dubbing`) and the studio server
(`mambodubb`) — run on native Windows 10/11 with an NVIDIA GPU. There is no
desktop `.exe` bundle; you run from source and use the editor in a browser.

**The short version:** if you have WSL2, use it (see the last section). Native
Windows works and is supported, but the translator runs on a plain transformers
worker there instead of vLLM, which is slower per line.

---

## 1. Prerequisites

```powershell
winget install --id astral-sh.uv -e
winget install --id Git.Git -e
winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
winget install --id ChrisBagwell.SoX -e --accept-source-agreements --accept-package-agreements
```

Open a **new** terminal afterwards: winget puts these on PATH for processes
started after it, and the Setup screen in the app re-probes `PATH` at startup —
a server launched before the install still cannot see `ffmpeg.exe`.

The Setup screen offers an **Install** button for `ffmpeg` and `sox` on Windows;
it runs exactly the two `winget` lines above and nothing else.

Python comes from `uv` (3.12), so no separate Python install is needed.

## 2. The checkout

```powershell
git clone --recurse-submodules https://github.com/maxmelichov/MamboDubb.git
cd MamboDubb
uv sync
copy .env.example .env      # HF_TOKEN for speaker diarization (optional)
```

`third_party/Qwen3-TTS` is a submodule; `uv sync` fails with "does not appear to
be a Python project" if you cloned without `--recurse-submodules`
(`git submodule update --init` fixes it).

### CUDA wheels

PyPI's default `torch` wheel is **CPU-only on Windows** (on Linux it is a CUDA
build). Install the CUDA build explicitly, or every model runs on the CPU and a
five-minute video takes hours:

```powershell
uv pip install --index-url https://download.pytorch.org/whl/cu124 torch torchaudio
```

Do the same inside the translator's own venv (`translator/`), which holds the
12B translation model:

```powershell
uv pip install --project translator --index-url https://download.pytorch.org/whl/cu124 torch
```

Check with `uv run python -c "import torch; print(torch.cuda.is_available())"`.

## 3. Run it

```powershell
uv run python -m dubbing "https://www.youtube.com/watch?v=VIDEO_ID"
uv run python -m dubbing input.mp4 --duration 300
```

The studio server, with the UI built once (Node 20+ and pnpm — `winget install
--id OpenJS.NodeJS.LTS -e`, then `npm i -g pnpm`):

```powershell
cd app\ui; pnpm install; pnpm build; cd ..\..
uv run mambodubb --port 4400        # → http://127.0.0.1:4400
```

## 4. What is different on Windows

| | Behaviour |
|---|---|
| **Translation backend** | No MLX (Apple-only) and no vLLM (no native Windows build), so `translate.load()` starts the transformers CUDA worker in `translator/` automatically. Nothing to configure. |
| **Multi-GPU** | Unchanged: with two or more CUDA devices the translator pins itself to the last one via `CUDA_VISIBLE_DEVICES` and stays resident. |
| **Cancelling a job** | The job child gets its own process group; cancel sends Ctrl-Break and then `taskkill /F /T`, so the ffmpeg and yt-dlp it spawned die with it. |
| **Server lifetime** | Windows does not reparent orphans, so the ppid watchdog is disabled there. A launcher that wants the server to die with it should hold the server's stdin and pass `--exit-on-stdin-close`. |
| **Console encoding** | The CLI, the server and both workers force UTF-8 on stdio; a `chcp 1252` console cannot otherwise print a Hebrew line. |
| **Tool lookup** | `ffmpeg`/`ffprobe`/`sox` are resolved with `shutil.which`, which honours `PATHEXT` — the `.exe` suffix is never assumed or spelled out. |

Everything else — the manifest, the caching, every pipeline invariant
(see [APP_ARCHITECTURE.md](APP_ARCHITECTURE.md), Non-negotiables) — is
identical. Run directories are portable between
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
  missing `sox.exe` fails inside the tts stage, not at startup — the Setup screen
  says so before you start.

## 6. WSL2 — the recommended path

WSL2 with an NVIDIA driver gives you the Linux setup exactly: vLLM is available
(much faster translation), the cuDNN handling in `dubbing/nvlibs.py` applies, and
the whole of [docs/setup.md](setup.md) reads literally.

```powershell
wsl --install -d Ubuntu
```

Then inside Ubuntu: install `uv`, `ffmpeg`, `sox`, clone the repo **into the
Linux filesystem** (`~/MamboDubb`, not `/mnt/c/...` — the 9p mount makes every
stage several times slower), and follow the Linux instructions. The server binds
inside WSL; open `http://127.0.0.1:4400` from Windows as usual (WSL2 forwards
loopback).
