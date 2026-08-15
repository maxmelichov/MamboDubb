"""Process entry point.

    uv run python -m dubbing_app.server --host 127.0.0.1 --port 0 \
        [--outputs DIR] [--ui-dir DIR]

The handshake, deliberately identical to `translator/worker.py`'s and to what a
Tauri shell would parse:

1. Bind the port (0 = OS-assigned), so the number printed is the number served.
2. Print **exactly one line** of JSON to stdout and flush:
   `{"status":"ready","port":54321,"version":"0.1.0"}`
3. Serve. Every log goes to stderr stdout is the handshake channel and nothing
   else may write to it, which is why uvicorn's logging is redirected below.
4. Watchdog: poll `os.getppid()` once a second and exit(0) when it changes, so a
   crashed parent never leaves a pipeline running. Skipped when the parent is
   already init (pid 1) there is nothing to outlive.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import socket
import sys
import threading
import time
from pathlib import Path

from . import __version__

REPO_ROOT = Path(__file__).resolve().parents[1]
WATCHDOG_INTERVAL = 1.0

# Where command-line tools actually live on a Mac. A GUI-launched app inherits
# launchd's minimal PATH (/usr/bin:/bin:...), not the shell profile's so the
# same checkout that works from a terminal reports ffmpeg/sox MISSING inside the
# installed app, and every stage's shell-out would fail the same way. The server
# is the ancestor of every job child, so widening PATH once here fixes them all.
TOOL_DIRS = ("/opt/homebrew/bin", "/opt/homebrew/sbin", "/usr/local/bin")


def widen_path(env: dict | None = None) -> str:
    env = env if env is not None else os.environ
    parts = [p for p in env.get("PATH", "").split(os.pathsep) if p]
    for d in TOOL_DIRS:
        if d not in parts and Path(d).is_dir():
            parts.append(d)
    env["PATH"] = os.pathsep.join(parts)
    return env["PATH"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m dubbing_app.server",
                                description="MamboDubb local server.")
    p.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=0, help="bind port, 0 = OS-assigned")
    p.add_argument("--outputs", type=Path, default=REPO_ROOT / "outputs",
                   help="run directory root (default: outputs/)")
    p.add_argument("--ui-dir", default=None,
                   help="directory of the built UI (default: app/ui/dist; "
                        "pass an empty string to serve the API only)")
    p.add_argument("--token", default=None,
                   help="bearer token required on every request; auto-generated "
                        "(and printed) whenever --host is not loopback")
    p.add_argument("--no-watchdog", action="store_true",
                   help="do not exit when the parent process goes away")
    p.add_argument("--exit-on-stdin-close", action="store_true",
                   help="exit when stdin reaches EOF (the desktop shell holds the "
                        "other end of the pipe, so EOF means the shell is gone)")
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


def stdin_watchdog(stream=None, on_close=None) -> threading.Thread:
    """Exit when stdin closes.

    The ppid watchdog cannot see through `uv run`: killing the shell kills the
    wrapper, but a wrapper that survives (or a shell that was SIGKILLed before
    its Drop ran) leaves this server's parent chain alive and a stray server
    listening forever that is exactly the leak that produced five orphaned
    servers in one afternoon of app relaunches. The pipe does not lie: the shell
    holds the write end of stdin for the process's whole life, and EOF here
    means it is gone, no matter what happened to the pids in between. Blocking
    `read()` costs nothing and needs no polling.
    """
    stream = stream if stream is not None else sys.stdin.buffer
    exit_ = on_close or (lambda: os._exit(0))

    def loop() -> None:
        try:
            while stream.read(4096):
                pass  # the shell never writes; drain defensively anyway
        except Exception:
            pass  # a broken pipe is the same news as EOF
        print("stdin closed; shutting down", file=sys.stderr, flush=True)
        exit_()

    thread = threading.Thread(target=loop, name="stdin-watchdog", daemon=True)
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
    widen_path()

    outputs = args.outputs.resolve()
    outputs.mkdir(parents=True, exist_ok=True)

    # Loopback binds are gated by the Host header; anything else is reachable
    # from the network and REQUIRES a token — generated here when the flag
    # didn't supply one, never skipped. The token has to outrank the bind
    # address because the server can read the filesystem and run the pipeline.
    loopback = args.host in ("127.0.0.1", "localhost", "::1")
    token = args.token or (None if loopback else secrets.token_urlsafe(24))
    app = create_app(outputs, ui_dir=args.ui_dir, token=token)

    sock = bind(args.host, args.port)
    port = sock.getsockname()[1]

    if not args.no_watchdog:
        watchdog()
    if args.exit_on_stdin_close:
        stdin_watchdog()

    announce(port)
    served = getattr(app.state, "ui_dir", None)
    print(f"mambodubb on http://{args.host}:{port} (outputs {outputs}; "
          f"ui {served or 'not served API only'})", file=sys.stderr, flush=True)
    if token:
        # The one line the user needs: opening this URL once sets the cookie.
        print(f"  open http://{args.host}:{port}/?token={token}",
              file=sys.stderr, flush=True)

    server = uvicorn.Server(uvicorn.Config(app, log_config=log_config, access_log=False))
    server.run(sockets=[sock])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
