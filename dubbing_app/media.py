"""Range-capable file serving out of a run directory.

The UI seeks in `preview.mp4` and scrubs clip wavs, so a plain whole-file
response is not enough: `<video>` will not seek without `Accept-Ranges`, and
Safari refuses to start playback at all unless the first request is answered with
a 206. Everything served is resolved and checked against the run directory first
a project name and a relative path both arrive from the network.
"""

from __future__ import annotations

import mimetypes
import re
from pathlib import Path
from typing import Iterator

from .errors import invalid, not_found

CHUNK = 64 * 1024

_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")

TYPES = {
    ".wav": "audio/wav",
    ".mp4": "video/mp4",
    ".m4a": "audio/mp4",
    ".json": "application/json",
    ".srt": "application/x-subrip",
    ".txt": "text/plain; charset=utf-8",
}


def resolve(workdir: Path, rel: str) -> Path:
    """The file `rel` names inside `workdir`, or an error. Never escapes."""
    if not rel or rel.startswith("/") or "\x00" in rel:
        raise invalid("invalid media path")
    root = workdir.resolve()
    # `strict=False`: resolve symlinks and `..` before comparing, so neither a
    # traversal (`../../.env`) nor a symlink planted in the run dir can point out.
    path = (root / rel).resolve()
    if path != root and root not in path.parents:
        raise not_found("no such media file")
    if not path.is_file():
        raise not_found("no such media file")
    return path


def content_type(path: Path) -> str:
    return TYPES.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0] \
        or "application/octet-stream"


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """(start, end) inclusive, or None for "serve the whole thing".

    A syntactically odd header, or more than one range, is answered with the
    whole file rather than an error that is what RFC 9110 allows and what keeps
    an exotic client playing instead of failing.
    """
    if not header:
        return None
    match = _RANGE_RE.match(header.strip())
    if not match:
        return None
    first, last = match.group(1), match.group(2)
    if not first and not last:
        return None
    if not first:                       # bytes=-500 → the final 500 bytes
        length = int(last)
        if length <= 0:
            raise Unsatisfiable(size)
        return max(0, size - length), size - 1
    start = int(first)
    end = int(last) if last else size - 1
    if start >= size or end < start:
        raise Unsatisfiable(size)
    return start, min(end, size - 1)


class Unsatisfiable(Exception):
    def __init__(self, size: int):
        super().__init__("range not satisfiable")
        self.size = size


def read_range(path: Path, start: int, end: int) -> Iterator[bytes]:
    remaining = end - start + 1
    with path.open("rb") as fh:
        fh.seek(start)
        while remaining > 0:
            chunk = fh.read(min(CHUNK, remaining))
            if not chunk:
                return
            remaining -= len(chunk)
            yield chunk


def serve(path: Path, range_header: str | None, *, head: bool = False):
    """A 200, a 206 or a 416 for `path`, honouring `Range`."""
    from fastapi.responses import JSONResponse, Response, StreamingResponse

    from .errors import envelope

    size = path.stat().st_size
    try:
        rng = parse_range(range_header, size)
    except Unsatisfiable:
        resp = JSONResponse(status_code=416,
                            content=envelope("invalid_request", "range not satisfiable"))
        resp.headers["Content-Range"] = f"bytes */{size}"
        resp.headers["Accept-Ranges"] = "bytes"
        return resp

    headers = {"Accept-Ranges": "bytes", "Cache-Control": "no-cache"}
    status = 200
    if rng is None:
        headers["Content-Length"] = str(size)
        body = read_range(path, 0, size - 1) if size else iter(())
    else:
        start, end = rng
        status = 206
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        headers["Content-Length"] = str(end - start + 1)
        body = read_range(path, start, end)
    if head:
        return Response(status_code=status, headers=headers, media_type=content_type(path))
    return StreamingResponse(body, status_code=status, media_type=content_type(path),
                             headers=headers)
