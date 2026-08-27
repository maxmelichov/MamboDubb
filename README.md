<p align="center">
  <a href="https://github.com/maxmelichov/MamboDubb/releases/latest">
    <img src="docs/assets/mambodubb-banner.png" alt="MamboDubb" width="760">
  </a>
</p>

<h3 align="center">Dub any video into another language entirely on your own machine.</h3>

<p align="center">
  <a href="https://github.com/maxmelichov/MamboDubb/releases/latest"><b>Download</b></a>
  ·
  <a href="#install">Mac · Windows · Linux</a>
  ·
  <a href="#setup-installs-the-models">Setup</a>
  ·
  <a href="#quick-start-cli">Run from source</a>
  ·
  <a href="docs/APP_ARCHITECTURE.md">Architecture</a>
</p>

<p align="center">
  <sub>
    Latest release:
    <a href="https://github.com/maxmelichov/MamboDubb/releases/tag/v0.4.0"><b>v0.4.0</b></a>,
    where Setup installs every model in one click. No account, no API key.
  </sub>
</p>

---

MamboDubb takes a video (a file or a YouTube link), transcribes it, translates the
transcript with context, speaks every line **in the original speaker's own voice**
using zero-shot cloning, and mixes the new speech back over the original music at the
original timing. Nothing is uploaded anywhere: every model runs locally, from ASR
and translation to voice synthesis, source separation and diarization.

## What it needs

Everything runs locally, so the hardware is the whole cost. Models load one at a
time and never sit in memory together, so the peak is whichever model is largest:
the 12B translator.

| | Minimum | Comfortable |
|---|---|---|
| **Apple Silicon** (unified memory) | 16 GB (low-VRAM mode) | 24 to 32 GB |
| **NVIDIA** (VRAM, bfloat16 translator) | 32 GB | 32 GB or more |
| **NVIDIA** (VRAM, low-VRAM mode) | 12 GB | 16 GB or more |
| **Disk** | 25 GB | 40 GB or more |

**On a Mac**, the translator is a 6-bit MLX build and peaks around 10 GB, and a
whole run sits near 13 GB resident. An M-series chip is required: the app has no
Intel build.

**On NVIDIA**, the translator runs in `translator/` as bfloat16, which is about
24 GB of weights before the KV cache: a 24 GB card cannot actually hold it, so
32 GB is the real floor for the default weights and low-VRAM mode is what a 3090
or a 4090 wants. Everything except translation is small: TTS is 4.2 GB, ASR
1.5 GB, source separation 0.4 GB.

### Low-VRAM mode

The 12B translator, quantised harder, so an ordinary card can run it. Same model,
same prompts, same everything else: only the weights are smaller.

| | Default | Low VRAM |
|---|---|---|
| **NVIDIA** | bfloat16, ~24 GB of weights | 4-bit NF4 through bitsandbytes, ~7 GB (calculated, not yet measured on a card) |
| **Apple Silicon** | 6-bit MLX, 9.7 GB on disk, peaks near 10 GB resident | mxfp4 MLX, 6.4 GB on disk (measured) |

It costs translation quality, and the honest number is on the Mac side, where it
has been measured. Thirty-one real Hebrew lines through both builds: mxfp4 was
1.6x faster and differed from the 6-bit output on 24 of them, most of that
harmless rewording. Six were real losses (an idiom read literally, a hallucinated
name pulled out of the run's context, "wounds" becoming "quills", a name spelled
two ways in one run), and one was a fix. The 4-bit CUDA figures are calculated
from the quantisation, not measured: nobody has run that path on a card yet.

Three ways to turn it on, and an explicit choice always beats the detection:

- **CLI**: `--low-vram`, or `--no-low-vram` to refuse the autodetection.
- **Env**: `DUBBING_LOW_VRAM=1` (`0` to force it off, `auto` or unset to detect).
- **App**: the "Low-VRAM translator" row on the Setup screen, which writes the
  same variable to the workspace `.env`. It applies to the next run.

It is detected automatically under 28 GiB of VRAM or under 20 GiB of unified
memory, and the run says so in its log rather than running out of memory. The
setting belongs to the machine and is never recorded in a project, so carrying a
run to a bigger card does not carry the small weights with it. On CUDA the mode
needs bitsandbytes, which the launch installs into the translator venv
(`uv sync --project translator --extra lowvram` does it by hand); where it cannot
be installed the run says so and loads the full weights instead of crashing.

**Disk**: about 16 GB of model weights, downloaded once from inside the app, plus
roughly 4 GB per 20 minutes of finished video. The setup screen warns below 10 GB
free.

**CPU only** works and is exercised by the test suite, but a run takes many times
real time. Treat it as a fallback, not a configuration.

**Speed**, measured on an M-series laptop: a 3 minute clip takes 20 to 40 minutes
end to end, most of it in translation and voice synthesis. Longer videos scale
close to linearly. Re-running is incremental, so a fix costs one line, not one run.

## The app

A desktop editor built around the script, not the timeline: every line shows the
original text and its translation together. Fix a translation in place, choose per
line whether it is dubbed or keeps the original audio, re-voice a single line in
seconds, and watch the source and output waveforms side by side.

- **Script-first editing**: original + translation on every row, inline edits,
  RTL-correct Hebrew alongside English.
- **Per-line control**: dub / keep original, re-translate, re-voice, split, merge,
  speaker reassignment, per-segment voice options.
- **Honest state**: every line says whether it is dubbed, kept, waiting, or failed,
  and one click fixes the ones that are waiting.
- **Hand edits are sacred**: a corrected line is locked, and no re-run ever
  overwrites it.
- **Compare everything**: the Orig/Dub buttons play the original span and the
  dubbed take of any line, back to back.

### Install

The app carries the pipeline source and the models it cannot fetch, and sets
itself up on first run: no terminal, and **no account or API key for anything**
(the one model behind a sign-up form is CC-BY-4.0 and ships inside the app).

**macOS (Apple Silicon).** The build is not yet notarized, so dragging the `.dmg`
into Applications makes macOS say the app is damaged and should go to the Trash.
That is Gatekeeper, not a bad download. Install with:

```bash
curl -fsSL https://github.com/maxmelichov/MamboDubb/releases/latest/download/install.sh | sh
```

That fetches the latest `.dmg`, copies MamboDubb to Applications, clears the
quarantine bit, signs the app on your Mac, and launches it. Apple Silicon only:
the translator runs on MLX, which needs an M-series chip. No Intel or iOS build.

If you already have the `.dmg`, either double-click **Install MamboDubb** inside
it, or run `sh install.sh /path/to/MamboDubb_*.dmg`.

**Windows and Linux.** The desktop shell builds for both:
`.github/workflows/build-desktop.yml` produces an NSIS installer and an `.msi` for
Windows and a `.deb` and an `.AppImage` for Linux, alongside the Mac `.dmg`. Status
is honest in [docs/CROSS_PLATFORM.md](docs/CROSS_PLATFORM.md): the code is
platform-correct and CI-built, but **no one has yet run those installers on real
Windows or Linux hardware**, and they are unsigned. On either platform the
translator moves from MLX to a CUDA worker in `translator/` automatically, so an
NVIDIA GPU is what you want; everything else is unchanged.

Running from source works everywhere today. On Windows:

```powershell
winget install --id astral-sh.uv -e
winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
winget install --id ChrisBagwell.SoX -e --accept-source-agreements --accept-package-agreements
git clone --recurse-submodules https://github.com/maxmelichov/MamboDubb.git
cd MamboDubb; uv sync
```

Windows needs one extra step (PyPI's default `torch` wheel is CPU-only there)
and has a few platform notes worth reading first: **[docs/WINDOWS.md](docs/WINDOWS.md)**.
WSL2 is the easier road if you have it, and the faster one.

### Setup installs the models

Open **Setup** on first run and press **Install everything**. The button prices
itself before you press it (the total download, in GB) and then pulls the whole
set: the translator, Qwen3-TTS, every ASR model, spoken-language ID, the Hebrew
adapter, and speaker diarization. Rows go green as they land. There is no Hugging
Face account or API key at any point, and nothing in the queue can stop half way
to ask for one.

- Every missing row also has its own **Download** button, labelled with that
  model's size, so you can take the 9.7 GB translator now and leave the rest.
- Diarization needs no download at all: the pyannote community-1 weights
  (CC-BY-4.0) ship inside the app in `third_party/`, and Setup restores them
  from that copy, checksum verified, with no network.
- A model that already sits in your Hugging Face cache counts as installed,
  but only where the loader actually reads the cache. No row goes green on a
  file the run cannot open.
- Prefer the terminal? Every missing row carries the exact command that fixes
  it, ready to copy, with absolute paths, so it works from any directory.
- Optional extras are not in "everything". Blocking checks are installed first,
  then the ones that only degrade quality, so the shortest usable install comes
  first on a slow connection.

## Quick start (CLI)

The whole pipeline is also a single headless command:

```bash
uv run python -m dubbing "https://www.youtube.com/watch?v=VIDEO_ID"
uv run python -m dubbing input.mp4 --duration 300        # just the first 5 minutes
```

The result lands in `outputs/<run>/preview.mp4`, next to a `report.json` that
accounts for every second of audio. Re-running is incremental: finished stages are
cached, and only what changed is redone.

## Run as a server (no desktop app)

The desktop app is a thin shell over a local Python server, so you can run the
server directly and use the full editor in any browser:

```bash
cd app/ui && pnpm install && pnpm build && cd ../..   # build the UI once
uv run mambodubb --port 4400                          # → http://127.0.0.1:4400
```

`--outputs` picks the run directory root (default `outputs/`); pass
`--ui-dir ""` to serve the API alone.

On the default loopback bind no login is needed; requests with a non-local
`Host` header are refused (DNS-rebinding guard). Binding any other address
(`--host 0.0.0.0` for LAN use) **requires a token**: pass `--token`, or let the
server generate one. It prints a one-click `?token=…` link that sets a cookie
for the rest of the session. The traffic is plain HTTP; treat the LAN mode as
"trusted home network", not "the internet".

## Languages

| | |
|---|---|
| **Spoken language (source)** | Hebrew, English, Arabic, Russian, French, Spanish, German, Italian, Portuguese, Chinese, Japanese, Korean |
| **Dub into (target)** | English, Russian, French, Spanish, German, Italian, Portuguese, Chinese, Japanese, Korean, **Hebrew** |

Targets are bounded by what Qwen3-TTS can speak. Hebrew is not one of its ten
languages; it is added by a LoRA over the same checkpoint (one extra download,
which Setup fetches with the rest, see [docs/setup.md](docs/setup.md)), and it
is switched off again for the other ten, so there is no second synthesiser and
nothing about them changes. Every
target can also be a source; Arabic is the one language that goes the other way
only: it can be heard and translated, never spoken, so it is still source-only.

Source and target may be the same language (`--src he --tgt he`). That is a dub,
not a no-op: every line is re-voiced in the speaker's cloned voice, with no
translation step and no translator loaded.

## How it works

```
fetch → stems → transcript → segments → translate → tts → timeline → mix → report
```

One module per stage, one `manifest.json` per run as the single source of truth.
The desktop app is a thin Tauri shell over a local FastAPI server that calls the
same pipeline, so the app is a second front end and never a fork. The models:
faster-whisper (ASR + verification), Gemma (translation, via MLX), Qwen3-TTS
(voice cloning), Demucs (music/voice separation), Pyannote (speakers). They load
sequentially and never co-exist in memory (see [What it needs](#what-it-needs)).

Guarantees the pipeline enforces on itself: no second of audible speech is ever
silently dropped, dubbed lines never overlap, transcription only ever listens to
the separated voice track so background music can never become invented lines,
and every synthesized take is verified by a second ASR pass before it is
accepted. Anything the pipeline could not account for is named in the run's
`report.json` instead of glossed over.

## Credits

MamboDubb is a pipeline, not a model: all of the intelligence below is the work
of the teams that built and released these models openly. Full credit to them:

| Model | By | Used for |
|---|---|---|
| [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | Qwen team, Alibaba Cloud | Speech synthesis + zero-shot voice cloning |
| [QwenTTS-he-1.7B](https://huggingface.co/notmax123/QwenTTS-he-1.7B) | Maxim Melichov | Hebrew LoRA over the same Qwen3-TTS checkpoint |
| [RenikudPlus](https://github.com/maxmelichov/RenikudPlus) | Maxim Melichov, Yakov Kolani, Morris Alper | Hebrew grapheme→stressed-IPA G2P feeding the Hebrew TTS |
| [Gemma 4 12B it](https://huggingface.co/mlx-community/gemma-4-12B-it-6bit) | Google DeepMind, 6-bit MLX quant by the mlx-community | Context-aware translation |
| [whisper-large-v3-turbo-ct2](https://huggingface.co/ivrit-ai/whisper-large-v3-turbo-ct2) | ivrit.ai, fine-tuning OpenAI Whisper | Hebrew transcription |
| [faster-whisper-large-v3-turbo-ct2](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2) | OpenAI Whisper, CT2 conversion by deepdml | Transcription for the other source languages |
| [faster-whisper base / base.en / tiny.en](https://huggingface.co/Systran) | OpenAI Whisper, converted by SYSTRAN | Verifying every synthesized take by ear |
| [Demucs (htdemucs_ft)](https://github.com/adefossez/demucs) | Alexandre Défossez et al., Meta AI | Separating voices from music and effects |
| [speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) | pyannote.audio, Hervé Bredin | Who speaks when. CC-BY-4.0, redistributed unmodified in `third_party/pyannote-speaker-diarization-community-1` (see its `NOTICE.md`), so no HF account is needed |
| [spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | SpeechBrain | Speaker embeddings for voice-consistent cloning |
| [lang-id-voxlingua107-ecapa](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa) | SpeechBrain | Spoken-language identification |
| [Silero VAD](https://github.com/snakers4/silero-vad) | Silero Team | Voice activity detection |

Running on [MLX](https://github.com/ml-explore/mlx) + [mlx-lm](https://github.com/ml-explore/mlx-lm) (Apple),
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) / CTranslate2, PyTorch,
[qwen-tts](https://github.com/QwenLM/Qwen3-TTS), FFmpeg, SoX and yt-dlp.

## Building the app from source

```bash
uv run scripts/build_desktop.py check   # toolchain sanity
uv run scripts/build_desktop.py build   # → app/desktop/src-tauri/target/release/bundle/
```
