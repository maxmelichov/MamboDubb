#!/usr/bin/env python3
"""Publish an ungated mirror of the pyannote diarization pipeline.

Why this script exists
----------------------
`pyannote/speaker-diarization-community-1` is the one model MamboDubb opens that
is **gated**. Not paid, not restricted the model card says the pipeline "is
released under the CC-BY-4.0 license and will always remain freely accessible"
but Hugging Face still puts an accept-the-terms click and a read token in front
of the files. On a machine that has never signed in, diarization failed, every
speaker in the video collapsed into one voice, and the app had to ask a user who
wanted to dub a video for an account on a model hub.

CC-BY-4.0 permits redistribution with attribution, so this publishes the *same
bytes* under a repo of ours that is not gated. Nothing is retrained, converted,
quantized or renamed: the files are copied at revision
`segments.DIARIZATION_REVISION`, and `--verify` re-hashes them against the
upstream LFS oids so "the same weights" is a checkable claim rather than a
promise. Attribution travels with them, in the README this writes.

Running it
----------
You need a Hugging Face **write** token for the namespace you are publishing to
(`hf auth login`, or `HF_TOKEN=hf_...`). It prints the exact plan and stops;
nothing is created until you pass `--yes`::

    uv run python scripts/upload_diarization_mirror.py --repo-prefix notmax123
    uv run python scripts/upload_diarization_mirror.py --repo-prefix notmax123 --yes

The default prefix is the owner of `segments.DIARIZATION_MIRROR`, so the default
run publishes exactly the repo the app already looks for. Publish somewhere else
and the app needs to be told:  `DUB_DIARIZATION_HUB=you/speaker-diarization-community-1`.

Where the files come from
-------------------------
The local HF cache first (a machine that has ever run diarization already has
them, and this then needs no read access to the gated repo at all), otherwise a
`snapshot_download` of the gated original which does need a token that has
accepted its terms. Either way they land in a staging directory you can inspect
before `--yes`.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dubbing import segments  # noqa: E402  (after sys.path)

# The pipeline is one self-contained repo: a config.yaml whose `$model/...`
# entries resolve against the repo itself. These five files are the whole thing —
# the rest of the upstream repo is a README and a GIF, and the README is not ours
# to copy verbatim (it is the upstream card, with their gating blurb in it).
FILES = ("config.yaml",
         "segmentation/pytorch_model.bin",
         "embedding/pytorch_model.bin",
         "plda/plda.npz",
         "plda/xvec_transform.npz")

LICENSE_TXT = REPO_ROOT / "third_party" / "licenses" / "CC-BY-4.0.txt"

README = """\
---
license: cc-by-4.0
library_name: pyannote-audio
tags:
  - pyannote
  - pyannote-audio
  - pyannote-audio-pipeline
  - speaker-diarization
  - mirror
---

# speaker-diarization-community-1 (ungated mirror)

An unmodified mirror of
[`{upstream}`](https://huggingface.co/{upstream}) at revision
[`{revision}`](https://huggingface.co/{upstream}/tree/{revision}).

**This is not a new model.** Same files, same bytes, same behaviour. The only
difference is that this copy is not gated, so software can fetch it on a machine
that has no Hugging Face account.

## Credit

- **Model:** `{upstream}` speaker diarization pipeline (segmentation +
  speaker embedding + VBx/PLDA clustering)
- **Authors:** Hervé Bredin and the pyannote / pyannoteAI team
- **Source:** <https://huggingface.co/{upstream}>
- **Revision mirrored:** `{revision}`
- **License:** [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) (see
  `LICENSE`), under which the upstream authors state the pipeline "is released
  under the CC-BY-4.0 license and will always remain freely accessible"

If you use this pipeline, cite the upstream work, not the mirror:

```bibtex
@inproceedings{{Plaquet23,
  author={{Alexis Plaquet and Hervé Bredin}},
  title={{{{Powerset multi-class cross entropy loss for neural speaker diarization}}}},
  year=2023,
  booktitle={{Proc. INTERSPEECH 2023}},
}}
@inproceedings{{Bredin23,
  author={{Hervé Bredin}},
  title={{{{pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe}}}},
  year=2023,
  booktitle={{Proc. INTERSPEECH 2023}},
}}
```

## Why it exists

[MamboDubb](https://github.com/) dubs video locally. Every other model it opens
is a public snapshot; this one asked for an account, and the cost of not having
one was silent — diarization simply failed and every character in the video was
dubbed in a single voice. Mirroring what the licence already permits removes the
account from the install.

Prefer upstream where you can: this mirror is a convenience, and the pyannote
team's gate is how they hear from the people using their work.
"""


def staged(stage: Path, source: Path, *, verify: bool) -> None:
    """Fill `stage` with the mirror's contents. Refuses on a missing or odd file."""
    if stage.exists():
        shutil.rmtree(stage)
    for rel in FILES:
        src = source / rel
        if not src.is_file():
            raise SystemExit(f"missing from {source}: {rel}")
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    shutil.copyfile(LICENSE_TXT, stage / "LICENSE")
    (stage / "README.md").write_text(
        README.format(upstream=segments.DIARIZATION_MODEL,
                      revision=segments.DIARIZATION_REVISION), encoding="utf-8")
    if verify:
        _verify(stage)


def _verify(stage: Path) -> None:
    """Re-hash the staged weights against the upstream repo's recorded oids.

    The hub's LFS oid for these files is their sha256, so this compares the bytes
    we are about to publish with the bytes upstream serves — no token needed, the
    metadata API answers for a gated repo. A mismatch is fatal: publishing
    something that is not the upstream weights while claiming it is would be the
    one failure this whole script exists to make impossible.
    """
    from huggingface_hub import HfApi

    info = HfApi().model_info(segments.DIARIZATION_MODEL,
                              revision=segments.DIARIZATION_REVISION, files_metadata=True)
    upstream = {s.rfilename: getattr(s, "lfs", None) for s in info.siblings}
    for rel in FILES:
        lfs = upstream.get(rel)
        oid = getattr(lfs, "sha256", None) if lfs else None
        if not oid:
            print(f"  {rel}: upstream reports no LFS hash — not verified")
            continue
        h = hashlib.sha256((stage / rel).read_bytes()).hexdigest()
        if h != oid:
            raise SystemExit(f"{rel}: staged {h} != upstream {oid}")
        print(f"  {rel}: sha256 matches upstream")


def find_source(token: str | None) -> Path:
    """The cached snapshot at the mirrored revision, downloading it if absent.

    Cache first and *offline*, so a machine that has already run diarization
    needs no read access to the gated repo to publish the mirror — and so a
    dry-run does not 401 on a checkout whose token was never set.
    """
    from huggingface_hub import snapshot_download

    kw = dict(revision=segments.DIARIZATION_REVISION, allow_patterns=list(FILES))
    try:
        path = Path(snapshot_download(segments.DIARIZATION_MODEL,
                                      local_files_only=True, **kw))
        if all((path / rel).is_file() for rel in FILES):
            print(f"using cached snapshot {path}")
            return path
    except Exception:
        pass
    print("not in the local cache — downloading from the gated repo "
          "(needs a token that has accepted its terms)")
    return Path(snapshot_download(segments.DIARIZATION_MODEL, token=token, **kw))


def main(argv: list[str] | None = None) -> int:
    default_prefix = segments.DIARIZATION_MIRROR.split("/")[0]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-prefix", default=default_prefix,
                   help=f"user or org to publish under (default: {default_prefix})")
    p.add_argument("--stage", type=Path,
                   default=REPO_ROOT / "build" / "diarization-mirror",
                   help="where the repo is assembled before upload")
    p.add_argument("--token", default=None,
                   help="HF write token; defaults to your logged-in credentials")
    p.add_argument("--no-verify", action="store_true",
                   help="skip re-hashing the staged files against upstream")
    p.add_argument("--yes", action="store_true",
                   help="actually create and upload. Without it, this only prints the plan")
    args = p.parse_args(argv)

    name = segments.DIARIZATION_MIRROR.split("/")[-1]
    repo_id = f"{args.repo_prefix}/{name}"

    print("This will publish an ungated CC-BY-4.0 mirror of")
    print(f"  {segments.DIARIZATION_MODEL} @ {segments.DIARIZATION_REVISION}")
    print(f"to  https://huggingface.co/{repo_id}   (public, NOT gated)")
    print("containing exactly:")
    for rel in (*FILES, "LICENSE", "README.md"):
        print(f"  {rel}")
    if repo_id != segments.DIARIZATION_MIRROR:
        print(f"\nNOTE: the app looks for {segments.DIARIZATION_MIRROR}. Publishing to")
        print(f"      {repo_id} instead means every machine needs")
        print(f"      {segments.DIARIZATION_HUB_ENV}={repo_id} (or edit segments.DIARIZATION_MIRROR).")

    print(f"\nstaging into {args.stage}")
    staged(args.stage, find_source(args.token), verify=not args.no_verify)
    print("staged.")

    if not args.yes:
        print("\nNothing was created. Re-run with --yes to publish.")
        return 0

    from huggingface_hub import HfApi

    api = HfApi(token=args.token)
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=str(args.stage), repo_type="model",
                      commit_message=f"Mirror {segments.DIARIZATION_MODEL} @ "
                                     f"{segments.DIARIZATION_REVISION} (CC-BY-4.0)")
    print(f"\npublished https://huggingface.co/{repo_id}")
    print("Check it is NOT gated (Settings → Gated access: off) — a gated mirror")
    print("solves nothing. Then a tokenless machine can diarize.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
