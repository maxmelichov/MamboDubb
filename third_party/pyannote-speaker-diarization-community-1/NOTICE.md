# speaker-diarization-community-1 — bundled, unmodified

These are the weights of the pyannote speaker-diarization pipeline, checked into
this repository and shipped inside the app. `dubbing/segments.py` loads them from
here (`segments.DIARIZATION_DIR`), which is why telling one speaker from another
costs a MamboDubb user no account, no token and no download.

**This is not a fork.** Same revision, same files, same bytes — verified against
the upstream repository, and `SHA256SUMS` records what was copied so the claim
can be checked rather than believed.

## Credit

- **Model** — `pyannote/speaker-diarization-community-1`, a speaker diarization
  pipeline (segmentation + speaker embedding + VBx/PLDA clustering).
- **Authors** — Hervé Bredin and the pyannote / pyannoteAI team.
- **Source** — <https://huggingface.co/pyannote/speaker-diarization-community-1>
- **Revision bundled** — `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`
  ([tree](https://huggingface.co/pyannote/speaker-diarization-community-1/tree/3533c8cf8e369892e6b79ff1bf80f7b0286a54ee))
- **License** — [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/), the
  full text of which sits next to these files as `LICENSE` (and in the repo's
  own list at `third_party/licenses/CC-BY-4.0.txt`). The upstream authors state
  the pipeline "is released under the CC-BY-4.0 license and will always remain
  freely accessible".

If you use this pipeline, cite the upstream work — not MamboDubb:

```bibtex
@inproceedings{Plaquet23,
  author={Alexis Plaquet and Hervé Bredin},
  title={{Powerset multi-class cross entropy loss for neural speaker diarization}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
}
@inproceedings{Bredin23,
  author={Hervé Bredin},
  title={{pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
}
```

## Why the bytes are here and not a download

Every other model MamboDubb opens is a public snapshot it can fetch on demand.
This one sat behind Hugging Face's gate — a contact-info form and a read token —
and the cost of not having an account was silent: diarization simply failed, and
every character in the video was dubbed in one voice. The gate is a form, not a
license restriction; CC-BY-4.0 permits redistribution with attribution, which is
what this directory is. 31 MB in the installer buys a fresh machine working
per-speaker voices with no sign-in.

Prefer upstream where you can. The pyannote team's gate is how they hear from
the people using their work, and `DUB_DIARIZATION_HUB=pyannote/speaker-diarization-community-1`
(with an `HF_TOKEN`) fetches the canonical repo instead of reading these files.
