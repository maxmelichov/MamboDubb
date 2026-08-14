/**
 * Setup — the first-run screen, and the place to come back to when something
 * on the machine changed underneath the app.
 *
 * `GET /api/setup` is a list of fast filesystem checks: the two binaries, the
 * token, each model directory, free disk. This screen is a checklist of them
 * and nothing else. Three rules it follows, all of them the editor's:
 *
 * - **Never colour alone.** Every row carries a glyph, the word "Ready" or
 *   "Missing", and a hue — in that order of importance. A monochrome screen
 *   reads exactly the same.
 * - **The detail line is the whole point.** "Missing" is not actionable; "run
 *   this command to fetch it (320 MB)" is. The server writes that sentence, the
 *   UI only renders it.
 * - **A row the server can fix gets a button.** `installable` comes from the
 *   server (it is the key set of `install.INSTALLERS`), never from a list kept
 *   here — a copy would drift and put a button on a row whose POST is a 400.
 *   Two binaries qualify; a ten-gigabyte model does not, and its detail line
 *   stays the whole answer.
 * - **Do not nag.** The gate in App.tsx routes here only when the server says
 *   `ok: false`. Otherwise this screen exists but is never in the way.
 *
 * Visually it borrows MamboRambo's onboarding: a summary band with a progress
 * bar over a divided list, and the actions in a sunken footer under it, so the
 * screen answers "how far off am I" before it answers "what exactly is wrong".
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ArrowRight, Check, Download, Loader2, RefreshCw, X } from "lucide-react";
import { PageShell } from "../components/AppShell";
import {
  Badge,
  Button,
  Card,
  CardSection,
  ErrorBlock,
  Eyebrow,
  Progress,
} from "../components/ui";
import { USE_FIXTURES, api } from "../lib/api";
import { cn } from "../lib/classNames";
import { isDesktop } from "../lib/desktop";
import type { SetupCheck, SetupInstall, SetupStatus } from "../lib/types";

/**
 * How often to ask where the install got to.
 *
 * There is no stream for this — setup has no project, so it has no event
 * stream — and a `brew install` runs for minutes, so two seconds is honest and
 * cheap. Fixture mode simulates the whole thing in under a second, which is the
 * only reason it polls faster: at 2 s the demo (and the smoke test) would watch
 * a spinner for one frame and miss every state in between.
 */
const POLL_MS = USE_FIXTURES ? 60 : 2000;

export function SetupPage() {
  const navigate = useNavigate();
  const [status, setStatus] = useState<SetupStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [checking, setChecking] = useState(true);
  const [install, setInstall] = useState<SetupInstall | null>(null);
  const [installError, setInstallError] = useState<string | null>(null);

  const recheck = useCallback(async () => {
    setChecking(true);
    try {
      setStatus(await api.setup());
      setError(null);
    } catch (err) {
      // An old server has no /api/setup at all. That is not a failed check, it
      // is no information — say so rather than inventing a red row.
      setStatus(null);
      setError(String(err instanceof Error ? err.message : err));
    } finally {
      setChecking(false);
    }
  }, []);

  useEffect(() => {
    void recheck();
  }, [recheck]);

  /*
   * The install poll.
   *
   * It runs only while something is running, and it is the only thing that ends
   * it: when a poll comes back `running: false` the row is redrawn from the
   * server's fresh probe *and* the whole checklist is re-run, because installing
   * ffmpeg can turn more than one row green (a stage that needed it, a tool that
   * shells out to it) and a screen that updates one row while the rest lie is
   * worse than a screen that takes another 40 ms.
   */
  const running = install?.running === true;
  const rechecking = useRef(false);
  useEffect(() => {
    if (!running) return;
    let live = true;
    const timer = setInterval(() => {
      void api
        .installStatus()
        .then((next) => {
          if (!live) return;
          setInstall(next);
          if (!next.running && !rechecking.current) {
            rechecking.current = true;
            void recheck().finally(() => {
              rechecking.current = false;
            });
          }
        })
        .catch((err: unknown) => {
          if (!live) return;
          setInstallError(String(err instanceof Error ? err.message : err));
          setInstall((prev) => (prev ? { ...prev, running: false } : prev));
        });
    }, POLL_MS);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [running, recheck]);

  const startInstall = useCallback(async (id: string) => {
    setInstallError(null);
    try {
      setInstall(await api.startInstall(id));
    } catch (err) {
      setInstallError(String(err instanceof Error ? err.message : err));
    }
  }, []);

  const checks = status?.checks ?? [];
  const failing = checks.filter((check) => !check.ok);
  const passing = checks.length - failing.length;
  const ready = status?.ok === true;

  return (
    <PageShell
      title="Setup."
      accent="What this machine has."
      lede={
        <>
          Everything runs on this machine, so the machine has to have it. These are cheap
          filesystem checks — no model is loaded to answer them.
          {isDesktop() ? null : " Re-check after installing anything."}
        </>
      }
    >
      <Card className="overflow-hidden p-0">
        <CardSection className="pb-4">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <Eyebrow>Readiness</Eyebrow>
              <p className="mt-2 text-lg font-semibold tracking-tight">
                {status ? (
                  ready ? (
                    "All checks pass"
                  ) : (
                    <>
                      {failing.length} of {checks.length} need attention
                    </>
                  )
                ) : checking ? (
                  "Checking…"
                ) : (
                  "No answer"
                )}
              </p>
            </div>
            {status ? (
              <span className="font-mono text-2xl tabular-nums text-muted">
                {passing}
                <span className="text-base">/{checks.length}</span>
              </span>
            ) : null}
          </div>
          <Progress
            className="mt-4"
            value={checks.length ? passing / checks.length : checking ? null : 0}
            tone={ready ? "var(--color-good)" : undefined}
          />
        </CardSection>

        {error && !status ? (
          <CardSection className="border-t border-border pt-5 text-[13px] leading-relaxed text-secondary">
            <p>
              Could not ask the server what is installed:{" "}
              <span className="text-primary">{error}</span>
            </p>
            <p className="mt-1.5 text-muted">
              Either it is not running yet, or it is an older build without{" "}
              <code className="font-mono text-[12px]">/api/setup</code>. The rest of the app works
              either way.
            </p>
          </CardSection>
        ) : null}

        {status ? (
          <ul className="divide-y divide-border border-t border-border">
            {status.checks.map((check) => (
              <CheckRow
                key={check.id}
                check={check}
                install={install?.id === check.id ? install : null}
                busy={running}
                onInstall={startInstall}
              />
            ))}
          </ul>
        ) : null}

        <CardSection
          tone="sunken"
          className="flex flex-wrap items-center gap-3 border-t border-border"
        >
          <Button onClick={() => void recheck()} disabled={checking}>
            {checking ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <RefreshCw className="h-3.5 w-3.5" />
            )}
            Re-check
          </Button>
          {ready ? (
            <Button variant="primary" onClick={() => navigate("/", { replace: true })}>
              Continue to projects
              <ArrowRight className="h-3.5 w-3.5" />
            </Button>
          ) : (
            <Link
              to="/"
              replace
              className="rounded-md text-[12px] font-semibold text-secondary underline underline-offset-4 transition-colors hover:text-primary"
            >
              Skip for now
            </Link>
          )}
          <span className="ml-auto max-w-xs text-[12px] leading-relaxed text-muted">
            {/* `ready` is the server's verdict on the REQUIRED checks only, so it
                can be true while optional rows sit red above this line — saying
                "nothing is missing" under a MISSING badge reads as a lie. */}
            {ready
              ? failing.length === 0
                ? "Nothing is missing."
                : `Everything required is ready; ${failing.length} optional ` +
                  `item${failing.length === 1 ? " is" : "s are"} missing for wider ` +
                  "language pairs."
              : "A missing model fails the stage that needs it, not the whole run."}
          </span>
        </CardSection>
      </Card>

      {error && status ? <ErrorBlock title="Re-check failed">{error}</ErrorBlock> : null}
      {/* A refusal — an id the server has no argv for, a second install while
          one runs, or no Homebrew at all. It carries the sentence that says what
          to do instead, so it goes where the user can read all of it. */}
      {installError ? (
        <ErrorBlock title="Install refused" onDismiss={() => setInstallError(null)}>
          {installError}
        </ErrorBlock>
      ) : null}
    </PageShell>
  );
}

/**
 * One check. The state is spelled three ways — glyph, word, hue — because the
 * palette's "kept"/"failed" pair is below the contrast gate for colour-vision
 * deficiency in light mode, and because a screenshot of a checklist gets read
 * at a glance by people who never see the hue at all.
 */
function CheckRow({
  check,
  install,
  busy,
  onInstall,
}: {
  check: SetupCheck;
  /** This row's install, or null — the page hands each row only its own. */
  install: SetupInstall | null;
  /** Any install is running. One at a time, so every other button greys out. */
  busy: boolean;
  onInstall: (id: string) => void;
}) {
  const Glyph = check.ok ? Check : X;
  const installing = install?.running === true;
  const failed = install !== null && !install.running && install.ok === false;
  // The button is offered for exactly one state: a row the server says it can
  // fix, that is currently broken. A passing row needs nothing and a model row
  // has no argv behind it — its detail line is the answer.
  const offerInstall = check.installable === true && !check.ok;
  return (
    <li
      className={cn(
        "flex items-start gap-3.5 px-6 py-4 sm:px-7",
        // Eight rows that look identical make the reader scan all eight to
        // find the two that need them. The failing ones get a wash and a rule
        // so the eye lands on them first — reinforcing the word and the glyph
        // that already say it, never replacing them.
        !check.ok && "bg-critical/[0.04] shadow-[inset_3px_0_0_var(--color-critical)]",
      )}
      data-check={check.id}
    >
      <span
        aria-hidden
        className="mt-px grid h-6 w-6 shrink-0 place-items-center rounded-md border"
        style={{
          color: check.ok ? "var(--color-good)" : "var(--color-critical)",
          borderColor: `color-mix(in srgb, ${
            check.ok ? "var(--color-good)" : "var(--color-critical)"
          } 35%, transparent)`,
          backgroundColor: `color-mix(in srgb, ${
            check.ok ? "var(--color-good)" : "var(--color-critical)"
          } 10%, transparent)`,
        }}
      >
        <Glyph className="h-3.5 w-3.5" strokeWidth={3} />
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1">
          <span className="text-[13px] font-semibold text-primary">{check.label}</span>
          <Badge tone={check.ok ? "good" : "bad"}>{check.ok ? "Ready" : "Missing"}</Badge>
        </div>
        {check.detail ? (
          <p className="mt-1 max-w-2xl text-[12px] leading-relaxed text-secondary">
            <Detail text={check.detail} />
          </p>
        ) : null}
        {/* This row's button, while it runs: one spinner, the word, and the last
            line of output. An install is minutes long and the poll is seconds
            long, so the last line is the only honest progress there is — a bar
            would have to invent the fraction. */}
        {installing ? (
          <p className="mt-1.5 flex items-center gap-2 text-[12px]">
            <Loader2 aria-hidden className="h-3 w-3 shrink-0 animate-spin text-secondary" />
            <span className="shrink-0 font-semibold text-secondary">Installing…</span>
            <span className="truncate font-mono text-[11.5px] text-muted">
              {lastLine(install) ?? "starting"}
            </span>
          </p>
        ) : null}
        {/* It failed: the reason is in the output, and it is never the first
            line. The whole tail is here rather than a summary, because the
            sentence that explains it is the package manager's, not ours. */}
        {failed ? (
          <>
            <p className="mt-1.5 text-[12px] font-semibold text-primary">
              {install.error ?? "The install did not finish."}
            </p>
            <pre className="mt-1 max-h-36 overflow-auto rounded-md bg-sunken px-2.5 py-2 font-mono text-[11px] leading-relaxed whitespace-pre-wrap text-secondary">
              {install.tail.join("\n")}
            </pre>
          </>
        ) : null}
      </div>
      {/* The button is *replaced* by the progress line above while this row is
          installing — two spinners on one row is one spinner too many. Every
          other installable row keeps its button and greys it out. */}
      {offerInstall && !installing ? (
        <Button
          size="sm"
          className="mt-px"
          // `busy` is the one-at-a-time rule, spelled where the user meets it.
          // The server enforces it with a 409 either way; disabling is how the
          // screen says so before the click rather than after.
          disabled={busy}
          onClick={() => onInstall(check.id)}
          title={busy ? "One install at a time" : undefined}
        >
          <Download className="h-3.5 w-3.5" />
          Install
        </Button>
      ) : null}
    </li>
  );
}

/** The most recent line of an install's output, if it has produced one. */
function lastLine(install: SetupInstall): string | null {
  return install.tail.length ? install.tail[install.tail.length - 1] : null;
}

/**
 * The detail sentence, with its backticked spans set as code.
 *
 * The server writes these to be read by a human and marks the parts that are
 * meant to be typed — a path, a variable, a command — the way every other tool
 * on that machine does, with backticks. Rendering them literally put stray
 * punctuation in the middle of the one line on the screen whose whole job is
 * to be copied correctly.
 */
function Detail({ text }: { text: string }) {
  return (
    <>
      {text.split(/`([^`]+)`/g).map((part, i) =>
        i % 2 === 1 ? (
          <code
            key={i}
            className="rounded bg-sunken px-1 py-0.5 font-mono text-[11.5px] text-primary"
          >
            {part}
          </code>
        ) : (
          <span key={i}>{part}</span>
        ),
      )}
    </>
  );
}
