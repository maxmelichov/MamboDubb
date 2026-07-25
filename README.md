# DubbingQwen

Dub a video into another language, entirely on your own machine.

It takes the video and its transcript, translates the transcript with context,
speaks each line in the original speaker's voice using zero-shot cloning, and mixes
the result back over the original music at the original timing.

```bash
uv sync
cp .env.example .env      # set HF_TOKEN for Pyannote diarization

uv run python -m dubbing "https://www.youtube.com/watch?v=VIDEO_ID"
uv run python -m dubbing input.mp4 --captions captions.json3
```

The result lands in `outputs/<run>/preview.mp4`, alongside `report.json` describing
what happened to every second of audio.

Iterating on a long video? `--duration 300` dubs just the opening, and re-running any
command resumes from cached work instead of redoing it.

## How it works

`fetch → stems → transcript → segments → translate → tts → timeline → mix → report`

- **Transcript** comes from local ASR, not the video's captions: auto-captions mangle
  exactly the words that matter most (on the test video `שייחה מוזה … בקטאר` — "Sheikha
  Moza … in Qatar" — is captioned as a broken name plus a non-word), and a translator
  cannot recover from that. Captions are still used to locate passages already spoken in
  the target language, which play as original audio.
- **Segments** are built from word timings, split at pauses and sentence ends. Pyannote
  labels who is speaking; when the transcript marks speaker changes itself, those markers
  win, since diarization edges land a word or two off.
- **Translation** runs twice per segment — once with neighbouring segments as context,
  once standalone — and keeps the context version only when the two agree, which is
  what stops a line from picking up its neighbour's content.
- **TTS** is one Qwen3-TTS call per segment, cloned from the cleanest few seconds of
  that speaker's own voice, then verified by transcribing it back. A segment that
  cannot be verified plays its original audio rather than going quiet.
- **Timeline** places each clip at its original onset, or as soon after as the previous
  clip allows. Nothing is ever trimmed to fit: overlong lines are gently sped up, and
  any remaining lateness is absorbed at the next natural pause.
- **Mix** ducks the music under speech and adds the clips in — never overwriting, which
  is guaranteed by the timeline's non-overlap invariant.

Details and the invariants that hold it together: [AGENTS.md](AGENTS.md).
Install and model downloads: [docs/setup.md](docs/setup.md).

## Layout

```text
DubbingQwen/
├── dubbing/            # the pipeline — one module per stage
├── tests/              # pure-logic tests, no models needed
├── models/             # local checkpoints (gitignored)
├── outputs/            # run artifacts (gitignored)
└── third_party/Qwen3-TTS/
```

## Requirements

macOS on Apple Silicon, Python 3.12 via `uv`, `ffmpeg` and `sox`. Models are
Gemma 4 12B (MLX), Qwen3-TTS-1.7B, Demucs, Pyannote and faster-whisper — all local.
