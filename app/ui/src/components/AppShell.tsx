/**
 * The two shells.
 *
 * `PageShell` is the roomy one a brand chip, a segmented nav, a display title
 * and cards on the plane. Import and Setup use it, because both are "arrive,
 * read, decide" screens where breathing room *is* the affordance.
 *
 * It comes in two widths. `default` is the reading column Setup is written for:
 * a checklist is a list of sentences and a sentence has a comfortable measure.
 * `wide` is the composition width the import screen lays a primary card, an
 * options rail and a full-width runs region across it, and on a maximised
 * window the old 56rem column left two thirds of the display empty with a
 * phone-shaped form marooned in the middle of it. The width is a prop rather
 * than a breakpoint because it is a property of the *screen*, not the viewport.
 *
 * `AppHeader` is the workspace bar: 44px, bordered, dense, and it never grows,
 * because the editor's job is to give every remaining pixel to the script.
 *
 * Both carry the same right-hand tools (theme toggle, fixtures badge, Setup),
 * so the app never loses its chrome when the layout changes underneath it.
 */

import { useEffect, useState, type ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";
import { ChevronLeft, Moon, Plus, SlidersHorizontal, Sun } from "lucide-react";
import { USE_FIXTURES } from "../lib/api";
import { cn } from "../lib/classNames";
import { currentTheme, onThemeChange, setTheme, type Theme } from "../lib/theme";
import { Badge, BrandChip, Segmented, segmentedCell } from "./ui";

export function PageShell({
  title,
  accent,
  lede,
  width = "default",
  children,
}: {
  /** Omit for no hero at all — the nav pill already names the place. */
  title?: string;
  /** The second line of the display title, set quieter and italic. */
  accent?: string;
  lede?: ReactNode;
  /** `wide` opens the page to the composition width; see the note above. */
  width?: "default" | "wide";
  children: ReactNode;
}) {
  const wide = width === "wide";
  return (
    <div className="min-h-screen bg-plane">
      <div
        className={cn(
          "mx-auto w-full px-5 pb-10 pt-5 sm:px-8 sm:pt-6",
          wide ? "max-w-[1440px]" : "max-w-4xl",
        )}
      >
        <header className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2.5">
            <Link to="/" className="rounded-full" aria-label="MamboDubb projects">
              <BrandChip />
            </Link>
            <ShellNav />
          </div>
          {/* Setup lives in the nav pill here, so the tool cluster does not
              carry it a second time. */}
          <HeaderTools withSetup={false} />
        </header>

        <div className="rise">
          {/* No title, no hero: the content starts right under the header. */}
          {!title && !lede ? null : wide ? (
            // One line, with the accent set as a clause rather than a second
            // display line: the wide layout wants its 90px of vertical back for
            // the composition under it.
            <div className="mt-8 flex flex-wrap items-end justify-between gap-x-12 gap-y-3">
              <h1 className="text-[2rem] leading-[1.08] font-semibold tracking-tight sm:text-[2.5rem]">
                {title}
                {accent ? (
                  <span className="font-normal italic text-muted/70"> {accent}</span>
                ) : null}
              </h1>
              {lede ? (
                <p className="max-w-xl text-[13.5px] leading-relaxed text-secondary">{lede}</p>
              ) : null}
            </div>
          ) : (
            <>
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
            </>
          )}

          <div
            className={cn(
              "flex flex-col gap-6",
              !title && !lede ? "mt-6" : wide ? "mt-7" : "mt-9",
            )}
          >
            {children}
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * The shell's navigation: two cells in a pill, one filled with ink.
 *
 * Runs had a cell for exactly one release. The list lives on the home page
 * now, under the form, so a pill cell for it was the same list twice — and
 * the user said so ("no need for the runs above"). /runs still exists as the
 * full archive; the "All runs" link under the home list is its door. The
 * editor is still not on the pill either: you get there by opening a run.
 *
 * New dub wears the ink whenever Setup does not, so an unknown path reads as
 * "you are at home" rather than as a pill with no answer.
 */
function ShellNav() {
  const path = useLocation().pathname;
  return (
    <Segmented aria-label="Sections">
      <Link
        to="/"
        className={segmentedCell(path !== "/setup")}
        title="Start a new dub"
      >
        <Plus className="h-3.5 w-3.5" aria-hidden />
        New dub
      </Link>
      <Link
        to="/setup"
        className={segmentedCell(path === "/setup")}
        title="What this machine has installed"
      >
        <SlidersHorizontal className="h-3.5 w-3.5" aria-hidden />
        Setup
      </Link>
    </Segmented>
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
        to="/runs"
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
 * The theme control. One button, two glyphs.
 *
 * It was a segmented pair two buttons, `aria-pressed` on each on the
 * argument that a lone sun-or-moon never says whether the glyph is the state or
 * the destination. The argument holds; the pair was the wrong answer to it. Two
 * radio-shaped controls for a binary is two tab stops, two things to hit, and a
 * dead cell half the time: pressing the one that is already on does nothing, and
 * a control that is sometimes a no-op is a control the keyboard has to be told
 * about. One button toggles, and the ambiguity is settled by *saying* it —
 * `aria-label` and `title` both name the destination ("Switch to light theme"),
 * which is the thing pressing it will do.
 *
 * The picture is unchanged, because the picture was never the problem: both
 * glyphs stay in the pill, and the one that is on wears the accent. That is a
 * state anyone can read at a glance, and it is now decoration over a single
 * control rather than two controls pretending to be one.
 *
 * There is no "System" option on purpose. The OS preference is not consulted
 * anywhere in this app (see lib/theme.ts), so offering it would be a control
 * that does nothing.
 */
function ThemeToggle() {
  const [theme, setLocal] = useState<Theme>(currentTheme);

  // Both shells mount their own copy; whichever one is clicked, all of them
  // move. Also picks up the value written before React mounted.
  useEffect(() => {
    setLocal(currentTheme());
    return onThemeChange(setLocal);
  }, []);

  const next: Theme = theme === "dark" ? "light" : "dark";
  const glyph = (value: Theme, Icon: typeof Sun) => (
    <span
      className={cn(
        "grid h-6 w-7 place-items-center rounded-full transition-colors",
        theme === value ? "bg-accent text-on-accent" : "text-muted",
      )}
    >
      <Icon className="h-3.5 w-3.5" aria-hidden />
    </span>
  );

  return (
    <button
      type="button"
      data-theme-toggle={theme}
      onClick={() => setTheme(next)}
      aria-label={`Switch to ${next} theme`}
      title={`Switch to ${next} theme`}
      className="flex shrink-0 items-center gap-0.5 rounded-full border border-border bg-raised p-0.5"
    >
      {glyph("light", Sun)}
      {glyph("dark", Moon)}
    </button>
  );
}

/**
 * The theme toggle, the fixtures badge and the Setup link permanent tools.
 *
 * `withSetup` is off in `PageShell`, where the nav pill beside the brand chip
 * already carries Setup and a second link to it two inches away is one control
 * too many. The editor's bar has no nav pill, so it keeps the link.
 */
function HeaderTools({ withSetup = true }: { withSetup?: boolean }) {
  const onSetup = useLocation().pathname === "/setup";
  return (
    <div className="flex shrink-0 items-center gap-2">
      {USE_FIXTURES ? (
        <Badge title="VITE_USE_FIXTURES=1 sample data, no server, no models">fixtures</Badge>
      ) : null}
      <ThemeToggle />
      {withSetup ? (
      <Link
        to="/setup"
        title="What this machine has installed"
        className={cn(
          "inline-flex h-8 items-center gap-1.5 rounded-lg border px-2.5 transition-colors",
          "text-[10px] font-bold uppercase tracking-[0.14em]",
          onSetup
            ? "border-transparent bg-accent text-on-accent"
            : "border-border bg-raised text-muted hover:border-axis hover:text-primary",
        )}
      >
        <SlidersHorizontal className="h-3.5 w-3.5" aria-hidden />
        Setup
      </Link>
      ) : null}
    </div>
  );
}
