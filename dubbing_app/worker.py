"""The job child process JSON spec on stdin, NDJSON progress on stdout, logs on stderr.

Same shape as `translator/worker.py`: structured lines on stdout, human noise on
stderr, one line per message, flushed. Nothing here decides anything; it is the
process boundary that keeps every model out of the server.

    parent → child   {"kind":"render","workdir":"...","payload":{...}}   one line
    child  → parent  {"type":"stage"|"segment"|"log", ...}               many lines
    child  → parent  {"type":"result","ok":true,"data":...}              once, last

Exit status is still authoritative: a non-zero exit is a failed job whatever was
printed.
"""

from __future__ import annotations

import copy
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from dubbing import manifest, tools

from . import ops


def emit(event: dict[str, Any]) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


def log(message: str, level: str = "info") -> None:
    emit({"type": "log", "level": level, "message": message})


_ABSENT = object()


class Journal:
    """The manifest write path for a job child, edit-safe.

    The child loads the manifest once and then works on its own copy for minutes.
    The server keeps answering `PATCH /segments/{uid}` against the same file the
    whole time that is the contract, no-model edits never wait for a job so a
    plain `manifest.save` at the end silently throws away every edit made while
    the job ran.

    Each save therefore re-reads the file first and re-applies whatever changed on
    disk since this child last wrote: a hand-edit outranks the machine's work,
    which is the same rule `manifest.reset_stage` follows for `locked` fields.
    Comparing against a snapshot rather than against `locked` alone is what makes
    a *release* (`PATCH {"locked": {}}`, which deletes keys) replay too.

    Matching by `uid` is enough for field edits, which is what the server allows
    while a job runs. A *structural* edit is a different thing a split retires
    the uid this child holds a clip for and makes two it has never seen and the
    merge cannot re-apply one: it would silently keep its own list and undo the
    split. `guard_structural` refuses those edits while a job runs, but the check
    and the job's start are not one atomic act, so `conflict` says what it found
    in the job log rather than resolving it quietly.
    """

    # `id` is positional and belongs to whoever renumbered last; `uid` is identity.
    SKIP = frozenset({"id", "uid"})

    def __init__(self, workdir: Path, m: dict[str, Any]) -> None:
        self.workdir = workdir
        self.m = m
        self.base = self._snapshot(m)

    @staticmethod
    def _snapshot(m: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {seg["uid"]: copy.deepcopy(seg)
                for seg in (m.get("segments") or []) if seg.get("uid")}

    def conflict(self, disk: dict[str, Any]) -> list[str]:
        """Segments that appeared or vanished on disk while this job worked.

        Field edits merge cleanly; a *structural* edit does not. A split makes two
        segments this child has never seen and retires the one it is holding a
        clip for, and `merge` matches by uid so it silently keeps its own list
        and the save undoes the split without a word. `guard_structural` refuses
        those edits while a job runs, but it cannot be atomic against a job that
        starts in the same instant, so the last line of defence is to say so
        rather than to resolve it quietly.
        """
        seen = {seg.get("uid") for seg in disk.get("segments") or []} - {None}
        return sorted(seen.symmetric_difference(self.base))

    def merge(self, disk: dict[str, Any] | None) -> list[str]:
        """Re-apply onto our copy the edits `disk` gained since our last write."""
        if not disk:
            return []
        clash = self.conflict(disk)
        if clash:
            log(f"the segment list changed on disk while this job ran "
                f"({len(clash)} segment(s): {', '.join(clash[:6])}"
                f"{'…' if len(clash) > 6 else ''}). This job is working from the "
                f"list as it was, so that structural edit is not in what it is "
                f"about to write. Redo it, or re-run the affected segments.",
                "error")
        mine = {seg["uid"]: seg for seg in (self.m.get("segments") or []) if seg.get("uid")}
        touched: list[str] = []
        for seg in disk.get("segments") or []:
            uid = seg.get("uid")
            was, ours = self.base.get(uid), mine.get(uid)
            if was is None or ours is None:
                continue
            # A verdict flip on disk owns the audio keys outright. Snapshot
            # comparison alone misses one case: the segment had no clip when
            # the job started (`tts` absent in `was`), the user flipped `keep`
            # mid-job (set_keep deletes `tts`/`place` — still absent on disk),
            # and the job then wrote a fresh clip. absent == absent, so the
            # key-by-key merge kept the job's clip alongside the re-applied
            # `keep=true` — a kept line with a dub placed, which the mix plays
            # and the UI (deriving from `keep`) never exposes.
            flip = was.get("keep") != seg.get("keep")
            forced = {"tts", "place"} if flip else frozenset()
            for key in manifest.SEGMENT_KEYS - self.SKIP:
                old, new = was.get(key, _ABSENT), seg.get(key, _ABSENT)
                if old == new and key not in forced:
                    continue                      # nobody touched it while we worked
                if new is _ABSENT:
                    ours.pop(key, None)
                else:
                    ours[key] = copy.deepcopy(new)
                if uid not in touched:
                    touched.append(uid)
            if flip:
                log(f"segment {uid}: keep flipped while this job ran; "
                    f"the flip wins over the job's clip", "info")
        return touched

    def save(self) -> None:
        merged = self.merge(manifest.load(self.workdir))
        if merged:
            log(f"kept {len(merged)} segment edit(s) made while this job ran")
        manifest.save(self.workdir, self.m)
        self.base = self._snapshot(self.m)


def execute(spec: dict[str, Any]) -> Any:
    kind = spec.get("kind")
    workdir = Path(spec["workdir"])
    payload = spec.get("payload") or {}

    if kind == "run":
        code = ops.full_run(workdir, payload.get("source") or {}, progress=emit)
        if code != 0:
            # `report` exits 1 on unaccounted audio: the run finished but the
            # result is not trustworthy. Surface it as a failure, not a success.
            raise RuntimeError(f"pipeline exited {code} (see report.json warnings)")
        return {"workdir": str(workdir)}

    m = manifest.load(workdir)
    if m is None:
        raise RuntimeError(f"no manifest in {workdir}")
    ops.ensure_uids(m)
    uids = list(payload.get("uids") or [])
    journal = Journal(workdir, m)

    try:
        if kind == "retranslate":
            result = ops.retranslate(m, workdir, uids, progress=emit)
        elif kind == "resynthesize":
            result = ops.resynthesize(m, workdir, uids, progress=emit)
        elif kind == "render":
            # `rebuild` saves after every stage, and each of those writes is a
            # chance to clobber a live edit so it gets the journal, not `save`.
            ops.rebuild(m, workdir, from_stage="timeline", progress=emit,
                        save=journal.save)
            result = {"preview": "preview.mp4"}
        else:
            raise RuntimeError(f"unknown job kind {kind!r}")
    finally:
        # Whatever happened, persist what got done the pipeline resumes from the
        # manifest, so a half-finished job must not throw its work away.
        journal.save()
    return result


def main(argv: list[str] | None = None) -> int:
    # Before the first read or emit: the spec on stdin and the NDJSON on stdout
    # are UTF-8 whatever this console thinks (the parent sets PYTHONIOENCODING
    # too; a worker started by hand on Windows gets it from here).
    tools.utf8_stdio((sys.stdin, sys.stdout, sys.stderr))
    raw = (argv[0] if argv else None) or sys.stdin.readline()
    try:
        spec = json.loads(raw)
    except Exception as exc:
        log(f"bad job spec: {exc}", "error")
        return 2
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except Exception:
        pass
    try:
        data = execute(spec)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        emit({"type": "result", "ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1
    emit({"type": "result", "ok": True, "data": data})
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
