<p align="center">
  <a href="https://github.com/maxmelichov/MamboDubb/releases/latest">
    <img src="docs/assets/mambodubb-logo.png" alt="MamboDubb" width="300">
  </a>
</p>

<h3 align="center">Dub any video into another language. Everything runs on your own machine.</h3>

<p align="center">
  <a href="https://github.com/maxmelichov/MamboDubb/releases/latest"><b>Download</b></a>
</p>

---

Give it a video file or a YouTube link. It transcribes, translates, speaks every line **in the original speaker's own voice**, and mixes the new speech back over the original music at the original timing. Nothing is uploaded anywhere. No account, no API key.

## Install

**Mac (Apple Silicon):**

```bash
curl -fsSL https://github.com/maxmelichov/MamboDubb/releases/latest/download/install.sh | sh
```

That one line is the install, and it is the only route with no Gatekeeper dialog anywhere in it: nothing here is notarized, and `curl` is what sets no quarantine bit. The script fetches the app from the latest release, clears quarantine, ad-hoc-signs the bundle so macOS will open it, and puts it in `/Applications`.

**Windows, Linux, or a Mac that would rather run the source:**

There is no installer to download for Windows or Linux. Building one needs a CI pipeline that is not running on this account, so rather than point you at a file that is not there, here is the route that works today on all three:

```bash
git clone --recurse-submodules https://github.com/maxmelichov/MamboDubb.git
cd MamboDubb && uv sync
cd app/ui && pnpm install && pnpm build && cd ../..
uv run mambodubb --port 4400          # editor at http://127.0.0.1:4400
```

That is the whole studio, not a reduced version of it: the desktop app is a thin shell over this same server, so a browser gets the identical editor. Build the UI once; after that the last line is the whole startup.

On Windows, install the CUDA build of PyTorch explicitly, because PyPI's default wheel is CPU-only there and every model would otherwise run on the CPU:

```powershell
uv pip install --index-url https://download.pytorch.org/whl/cu124 torch torchaudio
```

An NVIDIA GPU is what you want on either OS, and WSL2 is the smoother path on Windows if you have it. Details: [docs/SERVER.md](docs/SERVER.md), [docs/WINDOWS.md](docs/WINDOWS.md), [docs/CROSS_PLATFORM.md](docs/CROSS_PLATFORM.md).

Then open **Setup** in the app and press **Install everything**. It downloads every model the app can fetch, the optional extras included, and every row goes green. The button carries the total in GB before you press it, and each row keeps its own button if you would rather take one thing at a time.

## What you need

| | RAM / VRAM | Disk |
|---|---|---|
| **Mac** (M-series) | 16 GB minimum, 24 GB comfortable | 25 GB |
| **NVIDIA** | 12 GB with [low-VRAM mode](docs/LOW_VRAM.md), 32 GB full | 25 GB |

A 3 minute clip takes 20 to 40 minutes. Fixing one line afterwards only re-runs that line.

## Languages

**From:** Hebrew, English, Arabic, Russian, French, Spanish, German, Italian, Portuguese, Chinese, Japanese, Korean.
**Into:** all of those except Arabic.

## The editor

Every line shows the original and the translation side by side. Edit a translation, re-voice one line, keep the original audio on another, play original vs dub back to back. Your hand edits are never overwritten.

## CLI

```bash
uv run python -m dubbing "https://www.youtube.com/watch?v=VIDEO_ID"
```

Result lands in `outputs/<run>/preview.mp4`.

## More

- [How it works and architecture](docs/APP_ARCHITECTURE.md)
- [Low-VRAM mode](docs/LOW_VRAM.md)
- [Run as a server / use from a browser](docs/SERVER.md)
- [Building the desktop app](app/desktop/README.md)

## Credits

MamboDubb is a pipeline. The intelligence is the work of the teams behind [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) (voice cloning), [Gemma](https://huggingface.co/mlx-community/gemma-4-12B-it-6bit) (translation), [Whisper](https://github.com/SYSTRAN/faster-whisper) via [ivrit.ai](https://huggingface.co/ivrit-ai/whisper-large-v3-turbo-ct2) and deepdml (transcription), [Demucs](https://github.com/adefossez/demucs) (music separation), [pyannote](https://huggingface.co/pyannote/speaker-diarization-community-1) (speakers, CC-BY-4.0, redistributed in `third_party/`), [SpeechBrain](https://huggingface.co/speechbrain) (speaker and language ID), [Silero VAD](https://github.com/snakers4/silero-vad), and the Hebrew stack: [QwenTTS-he](https://huggingface.co/notmax123/QwenTTS-he-1.7B) + [RenikudPlus](https://github.com/maxmelichov/RenikudPlus). Full table with roles: [docs/CREDITS.md](docs/CREDITS.md).
