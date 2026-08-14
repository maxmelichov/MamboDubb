/**
 * The Tauri seam — the only file in the app that knows a desktop shell exists.
 *
 * Everything here is best-effort by construction: in a plain browser tab there
 * is no `window.__TAURI__` and no `@tauri-apps/api` on disk, so every call falls
 * back to the browser behaviour (a null path, a null base URL, a no-op reveal)
 * and the web build stays fully functional. That is the contract: callers may
 * import this module unconditionally and never branch on the platform except to
 * offer a *better* affordance when `isDesktop()` is true.
 *
 * The Tauri API package is deliberately not a dependency. It is reached through
 * the `withGlobalTauri` global when present, and otherwise through a dynamic
 * import whose specifier is assembled at runtime so the bundler leaves it alone
 * — the browser bundle must not require a package that only the shell ships.
 *
 * Commands, as the desktop shell (`app/desktop`, branch desk/tauri-shell)
 * exposes them:
 *
 *   get_server_url()            -> ServerInfo { base_url, port, version, workspace }
 *                                  (never starts a server; null-ish when none runs)
 *   start_server()              -> ServerInfo — idempotent, blocks on the ready line
 *   pick_video_file()           -> string | null   native open dialog
 *   reveal_path({ path })       -> void     show the file in Finder
 *   get_workspace()             -> string
 *   set_workspace({ path })     -> void
 *   check_workspace({ path })   -> unknown
 *   start_server() / stop_server()
 *
 * Only the four the UI actually needs are wrapped below; the rest are listed so
 * the next person does not have to go and read the Rust.
 */

type Invoke = <T>(cmd: string, args?: Record<string, unknown>) => Promise<T>;

type TauriGlobal = {
  core?: { invoke?: Invoke };
  invoke?: Invoke;
};

declare global {
  interface Window {
    __TAURI__?: TauriGlobal;
    __TAURI_INTERNALS__?: unknown;
  }
}

/**
 * True when the page is running inside the desktop shell. Synchronous on
 * purpose: a component may not render two different buttons one tick apart.
 * Tauri 2 installs `__TAURI_INTERNALS__` before any app code runs, and
 * `__TAURI__` too when `withGlobalTauri` is on, so a global sniff is enough.
 */
export function isDesktop(): boolean {
  if (typeof window === "undefined") return false;
  return "__TAURI__" in window || "__TAURI_INTERNALS__" in window;
}

let invokePromise: Promise<Invoke | null> | null = null;

/** Resolve an `invoke`, from the global if it is there and the module if not. */
function getInvoke(): Promise<Invoke | null> {
  if (invokePromise) return invokePromise;
  invokePromise = (async () => {
    if (typeof window === "undefined") return null;
    const global = window.__TAURI__;
    const fromGlobal = global?.core?.invoke ?? global?.invoke;
    if (typeof fromGlobal === "function") return fromGlobal;
    if (!isDesktop()) return null;
    try {
      // Assembled at runtime: an analyzable literal would make the bundler
      // resolve (and fail on) a package the browser build does not have.
      const specifier = ["@tauri-apps", "api", "core"].join("/");
      const mod = (await import(/* @vite-ignore */ specifier)) as { invoke?: Invoke };
      return typeof mod.invoke === "function" ? mod.invoke : null;
    } catch {
      return null;
    }
  })();
  return invokePromise;
}

/**
 * Call a shell command. Returns `fallback` in a browser, and also when the
 * command is missing or throws — the shell is built in parallel with this file,
 * so "the command is not there yet" must degrade, not crash the page.
 */
async function call<T>(cmd: string, args: Record<string, unknown> | undefined, fallback: T): Promise<T> {
  const invoke = await getInvoke();
  if (!invoke) return fallback;
  try {
    return await invoke<T>(cmd, args);
  } catch (error) {
    console.warn(`desktop: ${cmd} failed`, error);
    return fallback;
  }
}

/**
 * `call` for the commands that return nothing, where the only interesting
 * answer is whether it worked — `call` cannot say, because a command that
 * resolves with `null` and a command that threw both come back as the
 * fallback. The caller needs the difference: "the shell revealed the file" and
 * "the shell could not" lead to different affordances.
 */
async function callOk(cmd: string, args?: Record<string, unknown>): Promise<boolean> {
  const invoke = await getInvoke();
  if (!invoke) return false;
  try {
    await invoke<unknown>(cmd, args);
    return true;
  } catch (error) {
    console.warn(`desktop: ${cmd} failed`, error);
    return false;
  }
}

let serverInfoPromise: Promise<unknown> | null = null;

/**
 * Where the studio server is listening, or null when the app should just use
 * relative URLs. In the shell the webview's origin is the Tauri asset protocol
 * — not the sidecar — so every request has to be prefixed with this; in a
 * browser the server serves the UI itself and relative URLs are already right.
 *
 * Cached: the sidecar's port does not change while the app is open.
 */
/** The shell answers with `ServerInfo { base_url, port, ... }`; older sketches
 *  of the contract said a bare string. Accept both — the seam exists precisely
 *  so a shape drift lands here and nowhere else. */
function baseUrlOf(info: unknown): string | null {
  if (typeof info === "string") return info.replace(/\/+$/, "") || null;
  if (info && typeof info === "object" && "base_url" in info) {
    const url = (info as { base_url?: unknown }).base_url;
    if (typeof url === "string" && url) return url.replace(/\/+$/, "");
  }
  return null;
}

/**
 * The whole `ServerInfo`, cached — not just its URL.
 *
 * `workspace` comes back in the same handshake and is the only way the UI can
 * turn a run-relative path (`preview.mp4`, which is all the manifest stores)
 * into the absolute one `reveal_path` insists on. Asking for it separately
 * would be a second command for something already in hand.
 */
function serverInfo(): Promise<unknown> {
  if (!serverInfoPromise) {
    serverInfoPromise = (async () => {
      if (!isDesktop()) return null;
      // `get_server_url` never starts a server (that is its contract); on a
      // fresh launch there is nothing running yet, so fall through to
      // `start_server`, which is idempotent and blocks on the ready line.
      const running = await call<unknown>("get_server_url", undefined, null);
      if (baseUrlOf(running)) return running;
      return call<unknown>("start_server", undefined, null);
    })();
  }
  return serverInfoPromise;
}

export async function serverBaseUrl(): Promise<string | null> {
  return baseUrlOf(await serverInfo());
}

/** The checkout the shell is pointed at; its `outputs/` holds every run. */
export async function workspaceDir(): Promise<string | null> {
  const info = await serverInfo();
  if (info && typeof info === "object" && "workspace" in info) {
    const dir = (info as { workspace?: unknown }).workspace;
    if (typeof dir === "string" && dir) return dir.replace(/\/+$/, "");
  }
  return null;
}

/**
 * A native open dialog, returning an **absolute path** — the thing a browser
 * `<input type=file>` cannot give us and the pipeline actually needs. Null when
 * the user cancelled, and null in a browser so the caller falls back.
 */
export function pickVideoFile(): Promise<string | null> {
  if (!isDesktop()) return Promise.resolve(null);
  return call<string | null>("pick_video_file", undefined, null);
}

/**
 * Show a produced file in Finder. False in a browser, and false when the shell
 * refused — `reveal_path` errors on a path that does not exist, which is the
 * one failure the caller must not paper over: it means the file the button
 * offered is not there.
 */
export function revealPath(path: string): Promise<boolean> {
  if (!isDesktop() || !path) return Promise.resolve(false);
  return callOk("reveal_path", { path });
}

/**
 * Reveal a file *inside a run*, given the run-relative path the manifest
 * stores (`preview.mp4`, `preview_en.srt`).
 *
 * The manifest never records absolute paths — it is copied between machines —
 * and `reveal_path` takes nothing else, so the composition has to happen
 * somewhere. Here, because this is the file that already knows both halves:
 * the workspace comes from the shell's own handshake and every run lives at
 * `<workspace>/outputs/<name>/`, which is the layout the shell itself starts
 * the server with (`--outputs <workspace>/outputs`).
 *
 * False means "could not" — no shell, no workspace, or no such file — and the
 * caller should fall back to opening the URL instead of failing silently.
 */
export async function revealRunFile(project: string, relPath: string): Promise<boolean> {
  if (!isDesktop() || !project || !relPath) return false;
  const workspace = await workspaceDir();
  if (!workspace) return false;
  return revealPath(`${workspace}/outputs/${project}/${relPath}`);
}
