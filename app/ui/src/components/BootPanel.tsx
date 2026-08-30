/**
 * The boot surface what the desktop shell shows while `start_server` blocks.
 *
 * On a warm launch that block is under a second and this component renders
 * nothing at all (a short grace delay swallows the flash). On a *first* launch
 * it is the single longest wait in the product a payload copy and then a
 * multi-minute `uv sync` pulling ~1.7 GB of Python and before this panel
 * existed the window spent those minutes as a bare rectangle in the boot
 * script's background colour. A user who has just installed a DMG and gets a
 * dead window for five minutes does not wait; they quit. So the panel says the
 * three things that keep them: what is happening, that it only happens once,
 * and proof of life a ticking clock and the live tail of the server's
 * stderr, which is where uv narrates its download.
 *
 * The log comes from the shell's `get_server_log` command, polled once a
 * second. The runner keeps that readable *during* the blocking start (the ring
 * outlives and predates the process see runner/process.rs), which is the
 * whole reason polling works here. The command is invoked directly rather
 * than through `lib/desktop.ts`: the seam's `call()` deliberately flattens
 * errors into fallbacks for render-path callers, and this panel is the one
 * place that wants the raw channel. The invoke resolution is a copy of the
 * seam's, for the same reason the seam does it that way an analyzable
 * `@tauri-apps/api` literal would break the browser bundle.
 *
 * The failed state exists because the seam maps a rejected `start_server` to
 * a null base URL: without it the app would proceed to an Import screen whose
 * every fetch fails with a generic "could not ask the server". Here the log
 * tail names the real cause (the runner appends `[start error] ...` to the
 * ring on rejection), and Try again re-invokes `start_server` then reloads
 * the page reloading is what resets the seam's cached null, and the
 * now-running server makes the second boot a warm one. That same `[start error]`
 * line is what `diagnoseStartFailure` reads to decide which instruction the
 * panel gives, because "try again" is only advice for some of the ways a start
 * can fail and is a dead end for the rest.
 */

import { useEffect, useRef, useState } from "react";
import { Loader2, RefreshCw, TriangleAlert } from "lucide-react";
import { BrandChip, Button, Card, CardSection, Eyebrow } from "./ui";

type Invoke = <T>(cmd: string, args?: Record<string, unknown>) => Promise<T>;

let invokePromise: Promise<Invoke | null> | null = null;

function getInvoke(): Promise<Invoke | null> {
  if (invokePromise) return invokePromise;
  invokePromise = (async () => {
    if (typeof window === "undefined") return null;
    const global = window.__TAURI__;
    const fromGlobal = global?.core?.invoke ?? global?.invoke;
    if (typeof fromGlobal === "function") return fromGlobal;
    if (!("__TAURI__" in window || "__TAURI_INTERNALS__" in window)) return null;
    try {
      const specifier = ["@tauri-apps", "api", "core"].join("/");
      const mod = (await import(/* @vite-ignore */ specifier)) as { invoke?: Invoke };
      return typeof mod.invoke === "function" ? mod.invoke : null;
    } catch {
      return null;
    }
  })();
  return invokePromise;
}

/** How long a boot may stay invisible. Warm launches finish inside this. */
const GRACE_MS = 600;

/** What the panel says under the headline, and whether Try again is any use. */
type Diagnosis = { advice: string; retryHelps: boolean };

/**
 * The runner's reason, turned into an instruction that is true of *this* failure.
 *
 * The panel used to say the same paragraph whatever had happened: "usually the
 * one-time component download stopping part way, or a workspace the app cannot
 * read or write", and under the button, unconditionally, "Nothing already
 * downloaded is lost; setup resumes where it stopped". Neither sentence is true
 * of the most common first-run failure, which is that `uv` is not installed:
 * there is no download to resume, and Try again produces the identical error
 * for as long as the user is willing to press it. Telling someone to repeat a
 * gesture that cannot work is the same fault this project has already shipped
 * once, in the install screen, and it is worth not shipping twice.
 *
 * The reason is read off the ring's last `[start error] ...` line, which is the
 * one thing `ensure_server` guarantees is there on a rejection (runner/mod.rs).
 * Matched on the phrases those errors are actually built from; anything
 * unrecognised falls through to a paragraph that promises nothing beyond
 * pointing at the log, because a wrong instruction costs more than none.
 */
export function diagnoseStartFailure(log: string): Diagnosis {
  const marker = log.lastIndexOf("[start error]");
  const reason = marker < 0 ? "" : log.slice(marker + "[start error]".length).trim();
  const head = reason.split("\n")[0] ?? "";

  if (/uv was not found/i.test(head)) {
    return {
      advice:
        "The app cannot find uv, the tool it builds its Python environment with. " +
        "Install uv (on a Mac: brew install uv), then start the app again. Try " +
        "again will give the same error until uv is there.",
      retryHelps: false,
    };
  }
  if (/workspace not found|does not look like|no bundled workspace payload|no resource dir/i.test(head)) {
    return {
      advice:
        "The app's workspace is missing or incomplete, so there is nothing to " +
        "start from. Reinstalling the app restores it. Try again will report the " +
        "same thing until it is back.",
      retryHelps: false,
    };
  }
  if (/failed to provision|failed to refresh|failed to write|failed to read payload|failed to hash/i.test(head)) {
    return {
      advice:
        "The app could not finish setting up its workspace on disk. That is " +
        "almost always a full disk or a folder it is not allowed to write to. " +
        "Free some space, then try again.",
      retryHelps: true,
    };
  }
  if (reason) {
    return {
      advice:
        "The engine started and stopped before it was ready. The last lines of " +
        "its log below say why. Try again picks the setup up where it stopped " +
        "rather than starting it over.",
      retryHelps: true,
    };
  }
  return {
    advice:
      "The engine did not start, and it did not say why. The log below is " +
      "everything that was captured; please include it in a bug report.",
    retryHelps: false,
  };
}

/** m:ss, because "247 seconds" is a number and "4:07" is a wait. */
function formatElapsed(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

export function BootPanel({ failed }: { failed: boolean }) {
  // Nothing until the grace period passes or the start has already failed
  // the panel is for the long boot, not a flash on every good one.
  const [visible, setVisible] = useState(failed);
  const [log, setLog] = useState("");
  const [seconds, setSeconds] = useState(0);
  const [retrying, setRetrying] = useState(false);
  const logRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    if (failed) {
      setVisible(true);
      return;
    }
    const timer = window.setTimeout(() => setVisible(true), GRACE_MS);
    return () => window.clearTimeout(timer);
  }, [failed]);

  // The clock and the log share a cadence; one interval drives both.
  useEffect(() => {
    let alive = true;
    const poll = async () => {
      const invoke = await getInvoke();
      if (!invoke || !alive) return;
      try {
        const text = await invoke<string>("get_server_log");
        if (alive && typeof text === "string") setLog(text);
      } catch {
        // A missing command is an older shell; the panel still shows the clock.
      }
    };
    void poll();
    const interval = window.setInterval(() => {
      setSeconds((s) => s + 1);
      void poll();
    }, 1000);
    return () => {
      alive = false;
      window.clearInterval(interval);
    };
  }, []);

  // Pinned to the newest line, which is the only one that is news. uv rewrites
  // progress in place on a TTY but is line-by-line into a pipe, so the tail
  // moves constantly and a user reading history can use the scrollbar.
  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [log, visible]);

  const retry = async () => {
    setRetrying(true);
    const invoke = await getInvoke();
    if (!invoke) {
      setRetrying(false);
      return;
    }
    try {
      await invoke("start_server");
      // Success: reload so main.tsx re-resolves the api base against the
      // now-running server. The caches being module state is what makes a
      // reload the reset, and a warm server makes the second boot instant.
      window.location.reload();
    } catch {
      // The runner has already pushed "[start error] ..." into the ring; the
      // next poll paints it. Nothing to do here but let the button recover.
      setRetrying(false);
    }
  };

  const diagnosis = diagnoseStartFailure(log);

  if (!visible) {
    // The grace window: not a black void, the theme's own plane, so a boot
    // that resolves quickly still never shows a foreign colour.
    return <div className="min-h-screen bg-plane" />;
  }

  return (
    <div className="grid min-h-screen place-items-center bg-plane px-6 py-10">
      <div className="w-full max-w-xl">
        <div className="mb-6 flex justify-center">
          <BrandChip />
        </div>
        <Card className="overflow-hidden p-0">
          <CardSection className="pb-4">
            <Eyebrow>{failed ? "Startup failed" : "Starting up"}</Eyebrow>
            {failed ? (
              <>
                <p className="mt-2 flex items-center gap-2 text-lg font-semibold tracking-tight text-primary">
                  <TriangleAlert className="h-4.5 w-4.5 text-critical" strokeWidth={2.25} />
                  The dubbing engine could not start
                </p>
                <p className="mt-1.5 max-w-lg text-[13px] leading-relaxed text-secondary">
                  {diagnosis.advice}
                </p>
              </>
            ) : (
              <>
                <p className="mt-2 flex items-center gap-2 text-lg font-semibold tracking-tight text-primary">
                  <Loader2 className="h-4.5 w-4.5 animate-spin text-muted" strokeWidth={2.25} />
                  Preparing the dubbing engine
                </p>
                <p className="mt-1.5 max-w-lg text-[13px] leading-relaxed text-secondary">
                  First launch only: the app is setting up its Python environment,
                  which downloads about 2 GB and can take several minutes.
                  Every launch after this one takes seconds.
                </p>
              </>
            )}
          </CardSection>
          <CardSection tone="sunken" className="border-t border-border pt-4">
            <div className="flex items-baseline justify-between gap-3">
              <Eyebrow>Engine log</Eyebrow>
              <span className="font-mono text-[11px] tabular-nums text-muted">
                {formatElapsed(seconds)}
              </span>
            </div>
            <pre
              ref={logRef}
              className="mt-2 h-44 overflow-y-auto whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed text-secondary"
            >
              {log || (failed ? "No log was captured." : "Waiting for the first line…")}
            </pre>
            {failed && (
              <div className="mt-4 flex items-center gap-3">
                <Button variant="primary" onClick={() => void retry()} disabled={retrying}>
                  {retrying ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <RefreshCw className="h-3.5 w-3.5" />
                  )}
                  {retrying ? "Starting…" : "Try again"}
                </Button>
                <span className="text-[12px] leading-relaxed text-muted">
                  {diagnosis.retryHelps
                    ? "Nothing already downloaded is lost; setup resumes where it stopped."
                    : "Try again once you have done the above; on its own it will report the same thing."}
                </span>
              </div>
            )}
          </CardSection>
        </Card>
      </div>
    </div>
  );
}
