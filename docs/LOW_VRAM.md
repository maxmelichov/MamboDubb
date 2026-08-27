# Low-VRAM mode

The 12B translator, quantised harder, so an ordinary card can run it. Same model, same prompts, same everything else: only the weights are smaller.

| | Default | Low VRAM |
|---|---|---|
| **NVIDIA** | bfloat16, ~24 GB of weights | 4-bit NF4 through bitsandbytes, ~7 GB (calculated, not yet measured on a card) |
| **Apple Silicon** | 6-bit MLX, 9.7 GB on disk, peaks near 10 GB resident | mxfp4 MLX, 6.4 GB on disk (measured) |

## Hardware, in full

Models load one at a time and never sit in memory together, so the peak is whichever model is largest: the translator.

| | Minimum | Comfortable |
|---|---|---|
| **Apple Silicon** (unified memory) | 16 GB (low-VRAM mode) | 24 to 32 GB |
| **NVIDIA** (VRAM, bfloat16 translator) | 32 GB | 32 GB or more |
| **NVIDIA** (VRAM, low-VRAM mode) | 12 GB | 16 GB or more |
| **Disk** | 25 GB | 40 GB or more |

**On a Mac**, the translator is a 6-bit MLX build and peaks around 10 GB, and a whole run sits near 13 GB resident. An M-series chip is required: there is no Intel build.

**On NVIDIA**, the translator runs in `translator/` as bfloat16, which is about 24 GB of weights before the KV cache: a 24 GB card cannot actually hold it, so 32 GB is the real floor for the default weights and low-VRAM mode is what a 3090 or a 4090 wants. Everything except translation is small: TTS is 4.2 GB, ASR 1.5 GB, source separation 0.4 GB.

**CPU only** works and is exercised by the test suite, but a run takes many times real time. Treat it as a fallback, not a configuration.

## What it costs

Translation quality, a little. The honest number is on the Mac side, where it has been measured. Thirty-one real Hebrew lines through both builds: mxfp4 was 1.6x faster and differed from the 6-bit output on 24 of them, most of that harmless rewording. Six were real losses (an idiom read literally, a hallucinated name pulled out of the run's context, "wounds" becoming "quills", a name spelled two ways in one run), and one was a fix. The 4-bit CUDA figures are calculated from the quantisation, not measured: nobody has run that path on a card yet.

## Turning it on

Three ways, and an explicit choice always beats the detection:

- **CLI**: `--low-vram`, or `--no-low-vram` to refuse the autodetection.
- **Env**: `DUBBING_LOW_VRAM=1` (`0` to force it off, `auto` or unset to detect).
- **App**: the "Low-VRAM translator" row on the Setup screen, which writes the same variable to the workspace `.env`. It applies to the next run.

It is detected automatically under 28 GiB of VRAM or under 20 GiB of unified memory, and the run says so in its log rather than running out of memory. The setting belongs to the machine and is never recorded in a project, so carrying a run to a bigger card does not carry the small weights with it. On CUDA the mode needs bitsandbytes, which the launch installs into the translator venv (`uv sync --project translator --extra lowvram` does it by hand); where it cannot be installed the run says so and loads the full weights instead of crashing.
