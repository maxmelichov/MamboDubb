/**
 * The theme choice: one class, one localStorage key, no OS preference.
 *
 * `prefers-color-scheme` is deliberately absent from this file and from
 * App.css. The user picks, once, in the header, and the app remembers — a
 * video editor whose surfaces change colour when the OS crosses sunset is
 * changing the ground the user is judging a picture against.
 *
 * **Dark is the default.** Only an explicit stored `"light"` makes it light, so
 * a first-ever launch, a cleared profile and a private window all open dark.
 *
 * Three things apply the theme, and they must agree:
 *   1. the inline boot script in `index.html`, which runs before first paint;
 *   2. `applyTheme` here, called from `main.tsx` before render, which is what
 *      makes the app self-correcting if the boot script never ran (jsdom in
 *      the smoke test does not execute inline scripts);
 *   3. `setTheme`, from the header toggle.
 * The key and the class name are duplicated in index.html on purpose — that
 * script cannot import. If you rename either, rename it in both places.
 */

export type Theme = "light" | "dark";

export const THEME_KEY = "dubbing-studio.theme";
export const DARK_CLASS = "theme-dark";

/** The plane colour of each theme, for the pre-paint canvas and `theme-color`. */
export const THEME_PLANE: Record<Theme, string> = {
  light: "#f7f6f2",
  dark: "#110e16",
};

/** What the user last chose, or `null` if they never have. */
export function storedTheme(): Theme | null {
  try {
    const value = window.localStorage.getItem(THEME_KEY);
    return value === "light" || value === "dark" ? value : null;
  } catch {
    // Private windows and locked-down profiles throw on access, not on write.
    return null;
  }
}

/** The theme to render: the stored choice, else dark. */
export function currentTheme(): Theme {
  return storedTheme() ?? "dark";
}

/**
 * Stamp the theme on `<html>`. Also moves `theme-color`, which is what the
 * desktop shell's title bar and the browser's own chrome sample — leaving it
 * on the other theme's plane is the one seam a class cannot cover.
 */
export function applyTheme(theme: Theme): void {
  const root = document.documentElement;
  root.classList.toggle(DARK_CLASS, theme === "dark");
  document
    .querySelector('meta[name="theme-color"]')
    ?.setAttribute("content", THEME_PLANE[theme]);
}

const listeners = new Set<(theme: Theme) => void>();

export function setTheme(theme: Theme): void {
  try {
    window.localStorage.setItem(THEME_KEY, theme);
  } catch {
    // A theme that cannot be persisted is still worth applying for this session.
  }
  applyTheme(theme);
  for (const listener of listeners) listener(theme);
}

/** Subscribe to changes so every mounted toggle agrees. */
export function onThemeChange(listener: (theme: Theme) => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
