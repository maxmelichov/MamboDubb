"""Projects: a run directory under `outputs/`, read and written only via `dubbing.manifest`.

The server holds no database. `manifest.json` is the source of truth, so every
read goes to disk and every write goes through `manifest.save` which enforces
the `SEGMENT_KEYS` whitelist, the reason nothing here ever writes JSON itself.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from dubbing import STAGES, manifest

from . import ops
from .errors import invalid, not_found

# A project name is a single directory name under the outputs root. Anything with
# a separator, a drive letter or a dot-segment is refused before it can become a
# path the same check `media` relies on.
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def valid_name(name: str) -> bool:
    return bool(NAME_RE.match(name or "")) and name not in (".", "..") and ".." not in name


def slugify(source: str) -> str:
    stem = Path(source).stem if not source.startswith(("http://", "https://")) else source
    slug = re.sub(r"\W+", "_", stem).strip("_")[:48]
    return slug or "run"


class Projects:
    """Everything the HTTP layer needs to know about the outputs directory."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    # -- paths -------------------------------------------------------------

    def dir_for(self, name: str) -> Path:
        if not valid_name(name):
            raise invalid(f"invalid project name {name!r}")
        return self.root / name

    def require_dir(self, name: str) -> Path:
        workdir = self.dir_for(name)
        if not workdir.is_dir():
            raise not_found(f"no project {name!r}")
        return workdir

    def unique_name(self, wanted: str) -> str:
        base = wanted if valid_name(wanted) else slugify(wanted)
        if not valid_name(base):
            raise invalid(f"cannot derive a project name from {wanted!r}")
        name, n = base, 2
        while (self.root / name).exists():
            name, n = f"{base}_{n}", n + 1
        return name

    # -- manifest ----------------------------------------------------------

    def load(self, name: str) -> dict[str, Any]:
        workdir = self.require_dir(name)
        m = manifest.load(workdir)
        if m is None:
            raise not_found(f"project {name!r} has no manifest yet")
        if ops.ensure_uids(m):
            manifest.save(workdir, m)
        return m

    def save(self, name: str, m: dict[str, Any]) -> None:
        manifest.save(self.dir_for(name), m)

    def report(self, name: str, m: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """`report.json`, plus whether it is still about the manifest on disk.

        The file is served whatever the answer the numbers in a stale report are
        still the numbers of the run it described, and the UI decides how to show
        that. What it may not do is present them as current: an edit changes no
        stage parameter, so nothing in the file itself said the manifest had moved
        on, and a report from before a dozen corrections read exactly like a fresh
        one. A report written before the stamp existed cannot prove it is current,
        so it is not shown as if it were.
        """
        workdir = self.require_dir(name)
        path = workdir / "report.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(data, dict):
            return data
        if m is None:
            m = manifest.load(workdir)
        stamp = data.get("manifest")
        data["stale"] = not (stamp and m is not None
                             and stamp == manifest.content_fingerprint(m))
        return data

    # -- derived views -----------------------------------------------------

    @staticmethod
    def render_state(m: dict[str, Any] | None) -> dict[str, Any]:
        """What `preview.mp4` is a render of: `{at, stale, changed}`.

        `report.stale` says the *numbers* moved on; this says the *video* did, which
        is the one the user is about to watch. `mix.run` stamps `m["render"]` with the
        content fingerprint and a per-uid digest at the moment it wrote the file, so:

        * `at`   epoch seconds of that render, or `None` if the run predates the
          stamp or was never mixed;
        * `stale` the fingerprint no longer matches the manifest. Missing stamp
          counts as stale, on the same rule as `report.stale`: a file that cannot
          prove it is current is not shown as if it were;
        * `changed` how many lines differ from the ones that render was made of, by
          `manifest.digest_delta`. Exact, not an estimate of it; `0` when there is no
          stamp to compare against, which is why the UI must not phrase "N lines
          changed" off a render with no `at`.
        """
        stamp = (m or {}).get("render") or {}
        at = stamp.get("at")
        fp = stamp.get("fp")
        if m is None:
            return {"at": None, "stale": True, "changed": 0}
        stale = not (fp and fp == manifest.content_fingerprint(m))
        changed = manifest.digest_delta(stamp.get("segments"),
                                        manifest.segment_digests(m)) if stale else 0
        return {"at": at if isinstance(at, (int, float)) else None,
                "stale": stale, "changed": changed}

    @staticmethod
    def stage_status(m: dict[str, Any],
                     jobs: list[dict[str, Any]] | None = None) -> dict[str, str]:
        """What each pipeline stage is doing: `done`, `pending`, `running`, `failed`.

        Derived, never stored: the manifest's `stages` map is the only truth about
        what has *finished*, and re-deriving it keeps the server from holding a
        second copy that can drift. But the manifest cannot say what is happening
        right now or what died an unfinished stage looks exactly like one that
        was never reached so the project's jobs (as `Job.to_dict`, newest last)
        are folded in on top:

        * the stage a **running** job is in is `running`;
        * the stage the most recent **failed** job died in is `failed`, unless a
          job has succeeded since that later success is the answer about whether
          the run gets past that stage.

        Without this the whole failure treatment in the UI the red dot, "stopped
        at transcript" was unreachable against the real server, which only ever
        said done or pending.
        """
        recorded = m.get("stages") or {}
        status = {stage: ("done" if stage in recorded else "pending") for stage in STAGES}
        died: str | None = None
        for job in jobs or []:
            if job.get("status") == "done":
                died = None
            elif job.get("status") == "failed" and job.get("stage") in status:
                died = job["stage"]
        if died:
            status[died] = "failed"
        for job in jobs or []:
            if job.get("status") == "running" and job.get("stage") in status:
                status[job["stage"]] = "running"
        return status

    def summary(self, name: str, jobs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        workdir = self.root / name
        m = manifest.load(workdir) or {}
        src = m.get("source") or {}
        stages = self.stage_status(m, jobs) if m else {s: "pending" for s in STAGES}
        done = [s for s in STAGES if stages[s] == "done"]
        return {
            "name": name,
            "title": src.get("title") or src.get("input") or name,
            "input": src.get("input"),
            "src_lang": src.get("src_lang"),
            "tgt_lang": src.get("tgt_lang"),
            "duration": src.get("duration"),
            "segments": len(m.get("segments") or []),
            "stages": stages,
            "stage": done[-1] if done else None,
            "complete": bool(done) and done[-1] == STAGES[-1],
            "mtime": (workdir / "manifest.json").stat().st_mtime
            if (workdir / "manifest.json").is_file() else workdir.stat().st_mtime,
        }

    def list(self) -> list[dict[str, Any]]:
        if not self.root.is_dir():
            return []
        # A directory without a manifest is not a project: `load()` 404s on it,
        # so listing it hands the UI a row that can only dead-end (and its
        # events stream a 404 the client keeps retrying). Abandoned run dirs
        # exist on real machines; skip them rather than dressing them up.
        names = sorted(p.name for p in self.root.iterdir()
                       if p.is_dir() and valid_name(p.name)
                       and (p / "manifest.json").is_file())
        out = [self.summary(n) for n in names]
        out.sort(key=lambda p: p["mtime"], reverse=True)
        return out

    # -- segments ----------------------------------------------------------

    def enrich(self, name: str, m: dict[str, Any], seg: dict[str, Any]) -> dict[str, Any]:
        """A manifest segment plus what the UI needs and the manifest does not store.

        Three additions, all derived from disk:

        * `media.play` `place.clip`, the file the mix actually uses. This is
          NOT `tts.clip`: a segment that had to be sped up to fit plays
          `clips/fit_<hash>_1.300.wav`, and playing the raw clip would show the
          user audio that is not in the video.
        * `media.source` the segment's own window of `source.wav`, for A/B
          against the original.
        * `verify` the verdict `tts` already wrote to `clips/<hash>.json`
          (`heard`, `overlap`): the ready-made "did the clone say the right
          words" signal, keyed off `tts.clip`, since the fitted clip has no
          verdict of its own.
        """
        workdir = self.root / name
        out = dict(seg)
        tts = seg.get("tts") or {}
        place = seg.get("place") or {}
        start, end = float(seg.get("start") or 0.0), float(seg.get("end") or 0.0)
        source_wav = (m.get("files") or {}).get("source_wav") or "source.wav"
        out["media"] = {
            "play": self.media_url(name, place.get("clip"), workdir),
            "tts": self.media_url(name, tts.get("clip"), workdir),
            "source": f"{media_path(name, source_wav)}#t={start:.3f},{end:.3f}",
            "source_window": [start, end],
            # A dub-wanted line can wear an original-audio slice: tts gives an
            # unspeakable dub (no translation, failed synthesis) the universal
            # fallback rather than silence. Its "dub" then sounds exactly like
            # the original true, and baffling unless the UI says so. Keep
            # slices are span-keyed `keep_*.wav` files, which is the seam this
            # reads; a kept line's slice is not a fallback, it is the verdict.
            # `fit_keep_` is the same slice after `timeline` rate-fitted it —
            # timelines placed before timeline/v15 (which stopped stretching
            # fallback slices) still carry that name, and it is still original
            # audio wearing a dub's verdict.
            "fallback": bool(not seg.get("keep")
                             and str(place.get("clip") or "").split("/")[-1]
                             .startswith(("keep_", "fit_keep_"))),
        }
        out["verify"] = self.verdict(workdir, tts.get("clip"))
        return out

    def enrich_all(self, name: str, m: dict[str, Any]) -> list[dict[str, Any]]:
        return [self.enrich(name, m, seg) for seg in (m.get("segments") or [])]

    @staticmethod
    def media_url(name: str, rel: str | None, workdir: Path) -> str | None:
        if not rel:
            return None
        if not (workdir / rel).is_file():
            return None
        return media_path(name, rel)

    @staticmethod
    def verdict(workdir: Path, clip: str | None) -> dict[str, Any] | None:
        """`clips/<hash>.json`, written by `tts._verify_and_store`."""
        if not clip:
            return None
        meta = (workdir / clip).with_suffix(".json")
        if not meta.is_file():
            return None
        try:
            return json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            return None


def media_path(name: str, rel: str) -> str:
    return f"/media/{name}/{str(rel).lstrip('/')}"
