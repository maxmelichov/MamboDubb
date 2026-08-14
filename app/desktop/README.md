# Dubbing Studio — desktop shell

A Tauri 2 shell around the studio server. It is the third layer of the stack in
[docs/APP_ARCHITECTURE.md](../../docs/APP_ARCHITECTURE.md), and it changes nothing about
the other two: the UI still speaks HTTP to the server, the server still owns the pipeline.

## Why there is no bundled sidecar

MamboRambo ships its engine as a compiled binary inside the .app. Ours cannot: the
pipeline is a ~10 GB Python environment plus tens of gigabytes of model weights. So the
shell holds a **workspace** — the path to a DubbingQwen checkout — and runs the server
out of it:

```
uv run --project <workspace> python -m dubbing_app.server \
    --host 127.0.0.1 --port 0 --outputs <workspace>/outputs
```

then blocks on the single stdout ready line `{"status":"ready","port":N,...}`, exactly
as the process contract specifies. stderr is pumped into a 300-line ring buffer from the
moment of spawn (a first run is a multi-minute `uv sync`, and that is the only sign of
life while we wait). The child is killed on drop and on app exit; the server's own
parent-pid watchdog is the second belt.

`uv` is found via `DUBSTUDIO_UV_PATH`, then `/opt/homebrew/bin/uv`, `/usr/local/bin/uv`,
then `PATH`, then `~/.local/bin/uv`. The literal Homebrew path matters: an .app launched
from Finder inherits almost none of the user's shell `PATH`.

## Commands (all `Result<T, String>`)

| command | returns | notes |
|---|---|---|
| `get_workspace()` | `WorkspaceReport` | the stored workspace, or the default |
| `check_workspace(path)` | `WorkspaceReport` | inspect without storing |
| `set_workspace(path)` | `WorkspaceReport` | refuses a directory that is not a checkout |
| `start_server()` | `ServerInfo` | idempotent; starts or returns the running one |
| `stop_server()` | `()` | |
| `get_server_url()` | `ServerInfo \| null` | never starts one — poll this to pick a screen |
| `get_server_log()` | `string` | tail of the server's stderr, for the error panel |
| `reveal_path(path)` | `()` | Finder / Explorer |
| `pick_video_file()` | `string \| null` | native open dialog; `null` is a cancel |
| `pick_workspace_dir()` | `string \| null` | folder picker for the setup screen |

```ts
type WorkspaceReport = {
  path: string;
  exists: boolean;      // the directory is there
  has_project: boolean; // pyproject.toml + dubbing_app/
  has_venv: boolean;    // .venv/bin/python — absent is only a slow first run
  uv_found: boolean;
  uv_path: string | null;
  ready: boolean;       // has_project && uv_found — start_server can work
};

type ServerInfo = {
  base_url: string;   // http://127.0.0.1:<port>
  port: number;
  version: string | null;
  workspace: string;
};
```

`pick_video_file` is the one capability the webview cannot have on its own: a browser
`<input type=file>` yields a blob, and the pipeline needs an absolute path.

## Which frontend the window loads

The window loads the **bundled UI** (Tauri `frontendDist`), not the server's own
`--ui-dir`. The server is a subprocess that may take minutes to come up on a first run,
may fail, or may have no workspace to run from at all — a window pointed at its URL would
be blank in every one of those cases, with nowhere to render the setup screen. Loading
the bundled assets means the UI is up immediately and the server is something it reports
on. The UI gets the base URL from `get_server_url()` / `start_server()` and does plain
`fetch` against it; the server sets `allow_origins=["*"]`, so `tauri://localhost` is fine,
and `@tauri-apps/plugin-http` is available as a fallback. Either way the capability
allowlist is locked to `http://127.0.0.1:*` and `http://localhost:*`.

## Build and dev

Rust needs `export PATH="$HOME/.cargo/bin:$PATH"`.

```bash
uv run --script scripts/build_desktop.py check     # toolchain + a frontendDist to point at
uv run --script scripts/build_desktop.py ui        # pnpm build in app/ui, copy dist here
uv run --script scripts/build_desktop.py dev       # == pnpm tauri dev, Vite on :1430
uv run --script scripts/build_desktop.py build --debug   # .app + .dmg under src-tauri/target

cd app/desktop && pnpm install && pnpm tauri dev     # the same, by hand
cd app/desktop && pnpm tauri build --debug
cd app/desktop/src-tauri && cargo test               # the pure pieces
```

`pretauri` runs `build_desktop.py check` before every `pnpm tauri …`; `beforeBuildCommand`
runs `build_desktop.py ui`, which builds `app/ui` and copies `dist/` to `app/desktop/dist`.
If `app/ui` has never been built, `check` writes a placeholder `index.html` there so the
Tauri CLI has something to bundle.
