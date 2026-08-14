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
 *   get_server_url()            -> string   "http://127.0.0.1:<port>"
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

let baseUrlPromise: Promise<string | null> | null = null;

/**
 * Where the studio server is listening, or null when the app should just use
 * relative URLs. In the shell the webview's origin is the Tauri asset protocol
 * — not the sidecar — so every request has to be prefixed with this; in a
 * browser the server serves the UI itself and relative URLs are already right.
 *
 * Cached: the sidecar's port does not change while the app is open.
 */
export function serverBaseUrl(): Promise<string | null> {
  if (!baseUrlPromise) {
    baseUrlPromise = (async () => {
      if (!isDesktop()) return null;
      const url = await call<string | null>("get_server_url", undefined, null);
      if (!url || typeof url !== "string") return null;
      return url.replace(/\/+$/, "");
    })();
  }
  return baseUrlPromise;
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

/** Show a produced file in Finder. A no-op in a browser. */
export function revealPath(path: string): Promise<void> {
  if (!isDesktop() || !path) return Promise.resolve();
  return call<void>("reveal_path", { path }, undefined as void);
}
