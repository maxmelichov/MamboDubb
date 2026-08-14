"""The job child process — JSON spec on stdin, NDJSON progress on stdout, logs on stderr.

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

import json
import sys
import traceback
from pathlib import Path
from typing import Any

from dubbing import manifest

from . import ops


def emit(event: dict[str, Any]) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


def log(message: str, level: str = "info") -> None:
    emit({"type": "log", "level": level, "message": message})


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

    try:
        if kind == "retranslate":
            result = ops.retranslate(m, workdir, uids, progress=emit)
        elif kind == "resynthesize":
            result = ops.resynthesize(m, workdir, uids, progress=emit)
        elif kind == "render":
            ops.rebuild(m, workdir, from_stage="timeline", progress=emit)
            result = {"preview": "preview.mp4"}
        else:
            raise RuntimeError(f"unknown job kind {kind!r}")
    finally:
        # Whatever happened, persist what got done — the pipeline resumes from the
        # manifest, so a half-finished job must not throw its work away.
        manifest.save(workdir, m)
    return result


def main(argv: list[str] | None = None) -> int:
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
