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

## The app icon

`src-tauri/icons/icon.svg` is the source of truth; **every other file in that directory
is generated** and should never be hand-edited:

```bash
uv run --script scripts/make_icons.py            # rewrite the ladder from the SVG
uv run --script scripts/make_icons.py --check    # non-zero if the rasters are stale
```

The mark is two overlapping speech bubbles on a deep indigo-violet tile — the faint one
behind is the original track, the solid one in front is the dub and it carries the
waveform. Violet rather than blue because a Dock is mostly blue. The tile sits on the
macOS icon grid (824×824 inset in a 1024 canvas, corner radius 185) so the Dock draws it
the same size as a system app instead of a hair too large.

The generator uses `qlmanage -t -s 1024` for the master, because it is the only SVG
rasteriser a stock Mac has: `sips` cannot read SVG, and `rsvg-convert` / Inkscape /
ImageMagick / `cairosvg` are all extra installs (`cairosvg` additionally needs libcairo
from Homebrew). QuickLook's one flaw is that it composites onto opaque white and drops
the alpha channel, so the script re-cuts the alpha from the tile silhouette — which it
knows exactly, the tile being a plain rounded rect — and the SVG paints 4 px of bleed
past that silhouette so the white-blended edge pixels land outside the cut. Everything
below 1024 is a Pillow/Lanczos downscale, which is sharper at 16–32 px than
re-rasterising would be; `iconutil -c icns` assembles the `.icns` from a standard
`.iconset`, and Pillow writes the multi-resolution `.ico`.

If you ever install `rsvg-convert`, swapping it in is a two-line change to `rasterize()`
and the alpha re-cut can go away with it.

## Window chrome, and a note for whoever styles the UI

The window ships a **standard macOS title bar** (`titleBarStyle: "Visible"`), 1280×820,
centred, floored at 1024×680.

The prettier option is the frameless one — `titleBarStyle: "Overlay"` with
`hiddenTitle: true`, which drops the traffic lights straight onto the web content and
lets the app's own header act as the title bar. It is deliberately **not** enabled,
because with the header as it stands today the traffic lights would land on top of the
"Dubbing Studio" wordmark: `AppHeader` renders `h-10 … px-3` with the wordmark first, so
the first ~78 px of that row is exactly where macOS wants to draw its buttons. A title
bar that overlaps the app name is worse than a plain one.

To turn it on, both halves have to land together. The shell half:

```jsonc
// src-tauri/tauri.conf.json → app.windows[0]
"titleBarStyle": "Overlay",
"hiddenTitle": true,
"trafficLightPosition": { "x": 16, "y": 14 }   // centres 12 px dots in a 40 px header
```

The UI half — **not made here; `app/ui` belongs to someone else** — is:

1. Stamp the platform on the root element once at boot (`app/ui/src/main.tsx`), e.g.
   `document.documentElement.dataset.os = await platform();` from
   `@tauri-apps/plugin-os`, or `"macos"` inferred from `navigator.userAgent` to avoid the
   extra plugin.
2. Declare the inset as a custom property, so the padding is zero everywhere else:

   ```css
   :root            { --titlebar-inset: 0px; }
   [data-os="macos"] { --titlebar-inset: 78px; }
   ```

3. In `AppHeader`, spend it and make the row draggable:

   ```tsx
   <header
     data-tauri-drag-region
     className="flex h-10 shrink-0 items-center gap-3 border-b border-border
                bg-surface pl-[var(--titlebar-inset)] pr-3"
   >
   ```

   `data-tauri-drag-region` is not optional: with the native bar hidden, that row is the
   only thing left to drag the window by. Interactive children (the links, the theme
   button) must not carry the attribute, or they stop being clickable.

78 px is the measured extent of the three buttons plus a comfortable gap; if the header
grows taller than 40 px, raise `trafficLightPosition.y` to `(height - 12) / 2` to keep
them optically centred.

## Bundle metadata

`bundle.macOS.dmg` fixes the installer window at 700×460 with the .app at (180, 220) and
the Applications alias at (520, 220) — symmetric about the centre, with room above for
the volume name — so the drag reads as an instruction rather than two icons that happened
to land somewhere. There is deliberately no `background` image: Tauri only accepts a
single PNG/JPG/GIF for it, with no way to supply a `@2x` representation, so any type or
edge in it is visibly soft on a Retina display. A clean window with deliberate positions
beats a blurry one.

The version number lives in **three** files and they have to move together, exactly as
MamboRambo keeps them:

| file | field |
|---|---|
| `src-tauri/tauri.conf.json` | `version` |
| `src-tauri/Cargo.toml` | `[package] version` |
| `package.json` | `version` |

`tauri.conf.json` is the one that reaches the bundle (`CFBundleShortVersionString`, the
`.dmg` filename); `Cargo.toml` is what `app_version()` would report from Rust; the
`package.json` copy is inert but drifts silently, which is worse. All three are `0.1.0`.
`productName` is "Dubbing Studio" and the identifier is `com.maxmelichov.dubbingstudio`.

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

uv run --script scripts/make_icons.py --check        # icons still match icon.svg
```

`pretauri` runs `build_desktop.py check` before every `pnpm tauri …`; `beforeBuildCommand`
runs `build_desktop.py ui`, which builds `app/ui` and copies `dist/` to `app/desktop/dist`.
If `app/ui` has never been built, `check` writes a placeholder `index.html` there so the
Tauri CLI has something to bundle.
