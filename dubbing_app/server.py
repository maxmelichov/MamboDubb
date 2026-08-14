"""Process entry point.

    uv run python -m dubbing_app.server --host 127.0.0.1 --port 0 \
        [--outputs DIR] [--ui-dir DIR]

The handshake, deliberately identical to `translator/worker.py`'s and to what a
Tauri shell would parse:

1. Bind the port (0 = OS-assigned), so the number printed is the number served.
2. Print **exactly one line** of JSON to stdout and flush:
   `{"status":"ready","port":54321,"version":"0.1.0"}`
3. Serve. Every log goes to stderr — stdout is the handshake channel and nothing
   else may write to it, which is why uvicorn's logging is redirected below.
4. Watchdog: poll `os.getppid()` once a second and exit(0) when it changes, so a
   crashed parent never leaves a pipeline running. Skipped when the parent is
   already init (pid 1) — there is nothing to outlive.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import threading
import time
from pathlib import Path

from . import __version__

REPO_ROOT = Path(__file__).resolve().parents[1]
WATCHDOG_INTERVAL = 1.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m dubbing_app.server",
                                description="Dubbing Studio local server.")
    p.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=0, help="bind port, 0 = OS-assigned")
    p.add_argument("--outputs", type=Path, default=REPO_ROOT / "outputs",
                   help="run directory root (default: outputs/)")
    p.add_argument("--ui-dir", default=None,
                   help="directory of the built UI (default: app/ui/dist; "
                        "pass an empty string to serve the API only)")
    p.add_argument("--no-watchdog", action="store_true",
                   help="do not exit when the parent process goes away")
    return p.parse_args(argv)


def bind(host: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(128)
    sock.set_inheritable(True)
    return sock


def announce(port: int, stream=None) -> None:
    """The one line on stdout. Nothing else ever writes there."""
    stream = stream or sys.stdout
    stream.write(json.dumps({"status": "ready", "port": port, "version": __version__}) + "\n")
    stream.flush()


def watchdog(on_orphan=None, interval: float = WATCHDOG_INTERVAL) -> threading.Thread | None:
    parent = os.getppid()
    if parent <= 1:
        return None                      # already reparented to init; nothing to watch
    exit_ = on_orphan or (lambda: os._exit(0))

    def loop() -> None:
        while True:
            time.sleep(interval)
            if os.getppid() != parent:
                print("parent exited; shutting down", file=sys.stderr, flush=True)
                exit_()
                return

    thread = threading.Thread(target=loop, name="ppid-watchdog", daemon=True)
    thread.start()
    return thread


def configure_logging() -> dict:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(levelname)s %(name)s: %(message)s")
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "handlers": {"stderr": {"class": "logging.StreamHandler", "stream": "ext://sys.stderr"}},
        "root": {"handlers": ["stderr"], "level": "INFO"},
        "loggers": {"uvicorn": {"handlers": ["stderr"], "level": "INFO", "propagate": False},
                    "uvicorn.error": {"handlers": ["stderr"], "level": "INFO",
                                      "propagate": False},
                    "uvicorn.access": {"handlers": ["stderr"], "level": "WARNING",
                                       "propagate": False}},
    }


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    from .app import create_app

    args = parse_args(argv)
    log_config = configure_logging()

    outputs = args.outputs.resolve()
    outputs.mkdir(parents=True, exist_ok=True)
    app = create_app(outputs, ui_dir=args.ui_dir)

    sock = bind(args.host, args.port)
    port = sock.getsockname()[1]

    if not args.no_watchdog:
        watchdog()

    announce(port)
    served = getattr(app.state, "ui_dir", None)
    print(f"dubbing studio on http://{args.host}:{port} (outputs {outputs}; "
          f"ui {served or 'not served — API only'})", file=sys.stderr, flush=True)

    server = uvicorn.Server(uvicorn.Config(app, log_config=log_config, access_log=False))
    server.run(sockets=[sock])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
