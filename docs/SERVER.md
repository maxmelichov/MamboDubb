# Run as a server (no desktop app)

The desktop app is a thin shell over a local Python server, so you can run the server directly and use the full editor in any browser:

```bash
cd app/ui && pnpm install && pnpm build && cd ../..   # build the UI once
uv run mambodubb --port 4400                          # -> http://127.0.0.1:4400
```

`--outputs` picks the run directory root (default `outputs/`); `--ui-dir` picks the
directory the built UI is served from (default `app/ui/dist`), and passing it an
empty string serves the API alone.

On Windows and Linux this is not just an option, it is the way in: no installer is
built for either OS. You do not have to do the steps above by hand, though. One
command does the whole of it, Node and pnpm included:

```bash
curl -fsSL https://raw.githubusercontent.com/maxmelichov/MamboDubb/main/install-server.sh | sh
```

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/maxmelichov/MamboDubb/main/install-server.ps1 | iex"
```

Both fetch the source at the latest release tag with its submodule, install `uv`,
run `uv sync --extra app`, put the built web UI in `app/ui/dist`, and serve on port
4400. The web UI comes off the release as `mambodubb-ui-dist.tar.gz` where that
asset exists, and is built from a Node the installer provisions under `.tools/`
where it does not, so neither script needs Node on the machine beforehand.

By hand, the same thing is: clone with `--recurse-submodules`, `uv sync --extra app`,
then the two lines above. The extra is belt and braces rather than strictly
required today: `fastapi`, `uvicorn`, `httpx` and `python-multipart` currently
arrive as transitive dependencies of `gradio` underneath `qwen-tts`, which is not
a promise anyone made. Windows has extra steps around CUDA wheels in
[WINDOWS.md](WINDOWS.md); the state of packaging is in
[CROSS_PLATFORM.md](CROSS_PLATFORM.md).

## Security model

On the default loopback bind no login is needed; requests with a non-local `Host` header are refused (DNS-rebinding guard). Binding any other address (`--host 0.0.0.0` for LAN use) **requires a token**: pass `--token`, or let the server generate one. It prints a one-click `?token=...` link that sets a cookie for the rest of the session. The traffic is plain HTTP; treat the LAN mode as "trusted home network", not "the internet".

## Setup installs the models

Open **Setup** on first run and press **Install everything**. The button shows the total download in GB before you press it, then pulls the whole set: the translator, Qwen3-TTS, every ASR model, spoken-language ID, the Hebrew adapter, and speaker diarization. Rows go green as they land. There is no Hugging Face account or API key at any point, and nothing in the queue can stop half way to ask for one.

- Every missing row also has its own **Download** button, labelled with that model's size, so you can take the 9.7 GB translator now and leave the rest.
- Diarization needs no download at all: the pyannote community-1 weights (CC-BY-4.0) ship inside the app in `third_party/`, and Setup restores them from that copy, checksum verified, with no network.
- A model that already sits in your Hugging Face cache counts as installed, but only where the loader actually reads the cache. No row goes green on a file the run cannot open.
- For the terminal, every missing row carries the exact command that fixes it, ready to copy, with absolute paths, so it works from any directory.
- "Everything" means everything: the optional extras are in it too, including the Demucs stem-separation cache, which otherwise downloads itself in the middle of your first dub. The order is what protects a slow connection blocking checks first, then the ones that only degrade quality, then the extras so the machine is usable long before the queue ends, and Cancel stops it after whatever is in flight.
- Press it on a machine with nothing missing and it says so: the button goes quiet and reads "Everything is installed". It used to show you the last install's success instead.

## Same-language dubbing

Source and target may be the same language (`--src he --tgt he`). That still runs a full dub: every line is re-voiced in the speaker's cloned voice, with no translation step and no translator loaded.
