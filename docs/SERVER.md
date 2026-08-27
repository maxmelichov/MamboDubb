# Run as a server (no desktop app)

The desktop app is a thin shell over a local Python server, so you can run the server directly and use the full editor in any browser:

```bash
cd app/ui && pnpm install && pnpm build && cd ../..   # build the UI once
uv run mambodubb --port 4400                          # -> http://127.0.0.1:4400
```

`--outputs` picks the run directory root (default `outputs/`); pass `--ui-dir ""` to serve the API alone.

## Security model

On the default loopback bind no login is needed; requests with a non-local `Host` header are refused (DNS-rebinding guard). Binding any other address (`--host 0.0.0.0` for LAN use) **requires a token**: pass `--token`, or let the server generate one. It prints a one-click `?token=...` link that sets a cookie for the rest of the session. The traffic is plain HTTP; treat the LAN mode as "trusted home network", not "the internet".

## Setup installs the models

Open **Setup** on first run and press **Install everything**. The button prices itself before you press it (the total download, in GB) and then pulls the whole set: the translator, Qwen3-TTS, every ASR model, spoken-language ID, the Hebrew adapter, and speaker diarization. Rows go green as they land. There is no Hugging Face account or API key at any point, and nothing in the queue can stop half way to ask for one.

- Every missing row also has its own **Download** button, labelled with that model's size, so you can take the 9.7 GB translator now and leave the rest.
- Diarization needs no download at all: the pyannote community-1 weights (CC-BY-4.0) ship inside the app in `third_party/`, and Setup restores them from that copy, checksum verified, with no network.
- A model that already sits in your Hugging Face cache counts as installed, but only where the loader actually reads the cache. No row goes green on a file the run cannot open.
- Prefer the terminal? Every missing row carries the exact command that fixes it, ready to copy, with absolute paths, so it works from any directory.
- Optional extras are not in "everything". Blocking checks are installed first, then the ones that only degrade quality, so the shortest usable install comes first on a slow connection.

## Same-language dubbing

Source and target may be the same language (`--src he --tgt he`). That is a dub, not a no-op: every line is re-voiced in the speaker's cloned voice, with no translation step and no translator loaded.
