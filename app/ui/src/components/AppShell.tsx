/**
 * The two shells.
 *
 * `PageShell` is the roomy one — a brand header, a two-line display title and a
 * column of cards on the plane. Import and Setup use it, because both are
 * "arrive, read, decide" screens where breathing room *is* the affordance.
 *
 * `AppHeader` is the workspace bar: 44px, bordered, dense, and it never grows,
 * because the editor's job is to give every remaining pixel to the script.
 *
 * Both carry the same right-hand tools (theme toggle, fixtures badge, Setup),
 * so the app never loses its chrome when the layout changes underneath it.
 */

import { useEffect, useState, type ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";
import { ChevronLeft, Moon, SlidersHorizontal, Sun } from "lucide-react";
import { USE_FIXTURES } from "../lib/api";
import { cn } from "../lib/classNames";
import { currentTheme, onThemeChange, setTheme, type Theme } from "../lib/theme";
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
          <Link to="/" className="rounded-xl" aria-label="MamboDubb — projects">
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

/**
 * The workspace bar. 44px, and the brand tile is a back link rather than a
 * logo: on the editor the one navigation a user wants from the top-left is
 * "out of this run", and a wordmark that happens to be clickable does not say
 * so. It says "Runs", with the arrow, because that is where it goes.
 */
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
    <header className="flex h-11 shrink-0 items-center gap-2.5 border-b border-border bg-surface px-3">
      <Link
        to="/"
        className={cn(
          "inline-flex h-7 shrink-0 items-center gap-1 rounded-md pr-1.5 text-[12.5px] font-semibold",
          "text-secondary transition-colors hover:text-primary",
        )}
        title="Back to every run in the workspace"
      >
        <ChevronLeft className="h-3.5 w-3.5" aria-hidden />
        Runs
      </Link>
      <span className="h-4 w-px shrink-0 bg-border" aria-hidden />
      <div className="flex min-w-0 flex-1 items-center gap-2.5">{children}</div>
      {actions ? <div className="flex shrink-0 items-center gap-1.5">{actions}</div> : null}
      <span className="h-4 w-px shrink-0 bg-border" aria-hidden />
      <HeaderTools />
    </header>
  );
}

/**
 * The theme control.
 *
 * A segmented pair rather than one mystery icon: a single button that swaps
 * between a sun and a moon never says whether the glyph is the state or the
 * destination, and the answer differs by app. Two cells with `aria-pressed`
 * say both at once — which one you are in, and what the other one is.
 *
 * There is no "System" third cell on purpose. The OS preference is not
 * consulted anywhere in this app (see lib/theme.ts), so offering it would be
 * a control that does nothing.
 */
function ThemeToggle() {
  const [theme, setLocal] = useState<Theme>(currentTheme);

  // Both shells mount their own copy; whichever one is clicked, all of them
  // move. Also picks up the value written before React mounted.
  useEffect(() => {
    setLocal(currentTheme());
    return onThemeChange(setLocal);
  }, []);

  const cell = (value: Theme, Icon: typeof Sun, label: string) => (
    <button
      type="button"
      onClick={() => setTheme(value)}
      aria-pressed={theme === value}
      aria-label={`Switch to ${label} theme`}
      title={`${label[0].toUpperCase()}${label.slice(1)} theme`}
      className={cn(
        "grid h-6 w-7 place-items-center rounded-md transition-colors",
        theme === value
          ? "bg-primary text-on-primary"
          : "text-muted hover:text-primary",
      )}
    >
      <Icon className="h-3.5 w-3.5" aria-hidden />
    </button>
  );

  return (
    <div
      role="group"
      aria-label="Theme"
      className="flex shrink-0 items-center gap-0.5 rounded-lg border border-border bg-raised p-0.5"
    >
      {cell("light", Sun, "light")}
      {cell("dark", Moon, "dark")}
    </div>
  );
}

/** The theme toggle, the fixtures badge and the Setup link — permanent tools. */
function HeaderTools() {
  const onSetup = useLocation().pathname === "/setup";
  return (
    <div className="flex shrink-0 items-center gap-2">
      {USE_FIXTURES ? (
        <Badge title="VITE_USE_FIXTURES=1 — sample data, no server, no models">fixtures</Badge>
      ) : null}
      <ThemeToggle />
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
    </div>
  );
}
