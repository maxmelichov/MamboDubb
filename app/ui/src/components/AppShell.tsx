/**
 * The two shells.
 *
 * `PageShell` is the roomy one — a brand header, a two-line display title and a
 * column of cards on the plane. Import and Setup use it, because both are
 * "arrive, read, decide" screens where breathing room *is* the affordance.
 *
 * `AppHeader` is the workspace bar: 56px, bordered, dense, and it never grows,
 * because the editor's job is to give every remaining pixel to the timeline.
 *
 * Both carry the same right-hand tools (fixtures badge, Setup, theme), so the
 * app never loses its chrome when the layout changes underneath it.
 */

import { useEffect, useState, type ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";
import { Moon, SlidersHorizontal, Sun } from "lucide-react";
import { USE_FIXTURES } from "../lib/api";
import { cn } from "../lib/classNames";
import { Badge, Brand } from "./ui";

export function PageShell({
  title,
  accent,
  lede,
  children,
}: {
  title: string;
  /** The second line of the display title, set quieter and italic. */
  accent?: string;
  lede?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="min-h-screen bg-plane">
      <div className="mx-auto w-full max-w-4xl px-5 pb-20 pt-5 sm:px-8 sm:pt-7">
        <header className="flex h-14 items-center justify-between gap-4">
          <Link to="/" className="rounded-xl" aria-label="Dubbing Studio — projects">
            <Brand />
          </Link>
          <HeaderTools />
        </header>

        <div className="rise">
          <h1 className="mt-9 text-[2.25rem] leading-[1.1] font-semibold tracking-tight sm:text-[2.75rem]">
            {title}
            {accent ? (
              <>
                <br />
                <span className="font-normal italic text-muted/70">{accent}</span>
              </>
            ) : null}
          </h1>
          {lede ? (
            <p className="mt-5 max-w-xl text-[15px] leading-relaxed text-secondary">{lede}</p>
          ) : null}

          <div className="mt-9 flex flex-col gap-6">{children}</div>
        </div>
      </div>
    </div>
  );
}

export function AppHeader({
  children,
  actions,
}: {
  /** The identity area: what this screen is looking at. */
  children?: ReactNode;
  /** Screen-specific actions, sitting left of the permanent tools. */
  actions?: ReactNode;
}) {
  return (
    <header className="flex h-14 shrink-0 items-center gap-3 border-b border-border bg-surface px-4">
      <Link
        to="/"
        className="rounded-md text-primary transition-opacity hover:opacity-70"
        aria-label="Dubbing Studio — projects"
      >
        <Brand compact />
      </Link>
      <span className="h-5 w-px shrink-0 bg-border" aria-hidden />
      <div className="flex min-w-0 flex-1 items-center gap-3">{children}</div>
      {actions ? <div className="flex shrink-0 items-center gap-2">{actions}</div> : null}
      <span className="h-5 w-px shrink-0 bg-border" aria-hidden />
      <HeaderTools />
    </header>
  );
}

/** Fixtures badge, the Setup link and the theme toggle — the permanent tools. */
function HeaderTools() {
  const onSetup = useLocation().pathname === "/setup";
  return (
    <div className="flex shrink-0 items-center gap-2">
      {USE_FIXTURES ? (
        <Badge title="VITE_USE_FIXTURES=1 — sample data, no server, no models">fixtures</Badge>
      ) : null}
      <Link
        to="/setup"
        title="What this machine has installed"
        className={cn(
          "inline-flex h-8 items-center gap-1.5 rounded-lg border px-2.5 transition-colors",
          "text-[10px] font-bold uppercase tracking-[0.14em]",
          onSetup
            ? "border-transparent bg-primary text-on-primary"
            : "border-border bg-raised text-muted hover:border-axis hover:text-primary",
        )}
      >
        <SlidersHorizontal className="h-3.5 w-3.5" aria-hidden />
        Setup
      </Link>
      <ThemeToggle />
    </div>
  );
}

/**
 * Stamps data-theme on <html>. The CSS declares the dark steps under that
 * scope as well as prefers-color-scheme, so the toggle wins in both directions.
 */
function ThemeToggle() {
  const [theme, setTheme] = useState<"light" | "dark">(() => {
    const saved = localStorage.getItem("theme");
    if (saved === "light" || saved === "dark") return saved;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("theme", theme);
  }, [theme]);

  return (
    <button
      type="button"
      aria-label={theme === "dark" ? "Switch to light" : "Switch to dark"}
      onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
      className="inline-flex h-8 w-8 items-center justify-center rounded-lg border border-border bg-raised text-muted transition-colors hover:border-axis hover:text-primary"
    >
      {theme === "dark" ? <Sun className="h-3.5 w-3.5" /> : <Moon className="h-3.5 w-3.5" />}
    </button>
  );
}
