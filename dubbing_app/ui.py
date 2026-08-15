"""Serving the built React app out of the server process.

In development the UI is a Vite dev server on its own origin. Packaged, there is
no second process to run: the desktop shell starts this server and points a
webview at it, so the same process that answers `/api/...` must also hand out
`index.html` and the hashed bundles.

Two properties this has to get right:

* **The editor's routes exist only in the browser.** `/editor/whatsapp_0809` is a
  React-router path with no file behind it, so anything that is not a real file
  falls back to `index.html` and lets the router resolve it. That is what a
  hard reload on an editor URL needs.
* **The fallback must never eat the API.** It is registered last, and `/health`,
  `/api/...` and `/media/...` are refused explicitly rather than relying on route
  order order alone would still hand `index.html` to a *misspelled* API path,
  turning "404 no such project" into an HTML page the UI cannot parse.

Path handling follows `media.resolve`: resolve first, compare against the root
after, so neither `../` nor a symlink planted in `dist/` can serve a file from
outside the directory.
"""

from __future__ import annotations

from pathlib import Path

from .errors import not_found

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UI_DIR = REPO_ROOT / "app" / "ui" / "dist"

# Prefixes the fallback must never answer for. `health` is exact; the others are
# whole path segments, so a project literally named "apiary" is unaffected.
RESERVED = ("api", "media", "health")

# The built bundles are content-hashed, so they are immutable; index.html is not
# and must be revalidated or a released app serves yesterday's asset manifest.
IMMUTABLE = "public, max-age=31536000, immutable"
NO_CACHE = "no-cache"


def resolve_dir(ui_dir: str | Path | None) -> Path | None:
    """The directory to serve, or None for "do not serve a UI".

    `None` means the default location (a normal checkout or a packaged app), the
    empty string means explicitly disabled, anything else is taken literally.
    """
    if ui_dir is None:
        return DEFAULT_UI_DIR
    if isinstance(ui_dir, str) and not ui_dir.strip():
        return None
    return Path(ui_dir).expanduser()


def is_built(ui_dir: Path | None) -> bool:
    return bool(ui_dir) and (ui_dir / "index.html").is_file()


def reserved(path: str) -> bool:
    return path.split("/", 1)[0] in RESERVED


def resolve_file(root: Path, rel: str) -> Path | None:
    """The file `rel` names inside `root`, or None when there is none.

    Raises `not_found` rather than returning None, which would fall back to
    `index.html` when the path tries to leave the directory. A traversal is not
    a missing page; answering it with the app would be a confusing lie.
    """
    if not rel:
        return None
    if rel.startswith("/") or "\x00" in rel:
        raise not_found("no such file")
    root = root.resolve()
    path = (root / rel).resolve()
    if path != root and root not in path.parents:
        raise not_found("no such file")
    return path if path.is_file() else None


def install(app, ui_dir: Path | None) -> bool:
    """Mount `ui_dir` on `app`. Returns whether anything was mounted.

    **Call this last.** Everything registered before it keeps its routes; the
    catch-all only ever sees what nothing else claimed.
    """
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    if not is_built(ui_dir):
        return False
    root = ui_dir.resolve()
    index = root / "index.html"

    assets = root / "assets"
    if assets.is_dir():
        # The bundles get a real static mount rather than the fallback: it brings
        # its own traversal guard, and ETag/Last-Modified conditional responses
        # that a plain FileResponse would not send.
        app.mount("/assets", StaticFiles(directory=str(assets)), name="ui-assets")

    def send(path: Path, *, cache: str) -> FileResponse:
        return FileResponse(path, headers={"Cache-Control": cache})

    @app.get("/{ui_path:path}", include_in_schema=False)
    def ui(ui_path: str = ""):
        if reserved(ui_path):
            raise not_found("no such endpoint")
        found = resolve_file(root, ui_path)
        if found is not None and found != index:
            return send(found, cache=IMMUTABLE if ui_path.startswith("assets/") else NO_CACHE)
        return send(index, cache=NO_CACHE)

    app.state.ui_dir = str(root)
    return True


__all__ = ["DEFAULT_UI_DIR", "install", "is_built", "resolve_dir", "resolve_file"]
