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
 * - **A failure says what it costs.** Every missing row used to be the same red
 *   X, so a gated Hugging Face token (the run works, everyone in the video
 *   becomes one speaker) looked exactly like a missing ffmpeg (nothing runs at
 *   all), which looked exactly like a Korean TTS checkpoint a Hebrew→English run
 *   will never open. The server now grades them — `severity` is `blocking`,
 *   `degrades` or `optional` — and each grade gets its own word: REQUIRED,
 *   DEGRADES, OPTIONAL. The hue follows the word, never the other way round.
 * - **The detail line is the whole point.** "Missing" is not actionable; "run
 *   this command to fetch it (320 MB)" is. The server writes that sentence, the
 *   UI only renders it — and every command in it is one click to the clipboard,
 *   because a command that has to be retyped from a screenshot is a command
 *   that gets retyped wrong.
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

import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ArrowRight, Check, Copy, Download, Loader2, RefreshCw, X } from "lucide-react";
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
import { STAGES } from "../lib/types";
import type { SetupCheck, SetupInstall, SetupSeverity, SetupStatus } from "../lib/types";

/**
 * What a failing row actually costs, with a fallback for a server that predates
 * the field.
 *
 * That fallback reads `required` as blocking and everything else as optional,
 * which is the *old* two-value contract — the one reading that can understate a
 * failure but never overstate one. An unknown grade defaulting to "degrades"
 * would invent a claim about a check nobody has graded.
 */
function severityOf(check: SetupCheck): SetupSeverity {
  return check.severity ?? (check.required === false ? "optional" : "blocking");
}

const SEVERITY_META: Record<
  SetupSeverity,
  { word: string; token: string; tone: "bad" | "warn" | "neutral"; wash: boolean }
> = {
  // Nothing runs. Red, a wash, and the word people already read as "stop".
  blocking: { word: "Required", token: "var(--color-critical)", tone: "bad", wash: true },
  // The run works and is worse. Amber: it must not be dismissible at a glance,
  // and it must not read as broken either.
  degrades: { word: "Degrades", token: "var(--color-warning)", tone: "warn", wash: true },
  // Irrelevant until you ask for it. No hue and no wash — an optional row that
  // shouted was what taught users to ignore the whole list.
  optional: { word: "Optional", token: "var(--color-muted)", tone: "neutral", wash: false },
};

/** The earliest stage a failing blocking row would stop a run at, if any. */
function firstBlockingStage(checks: SetupCheck[]): string | null {
  const stages = checks
    .filter((check) => !check.ok && severityOf(check) === "blocking")
    .map((check) => check.stage)
    .filter((stage): stage is (typeof STAGES)[number] => Boolean(stage));
  if (stages.length === 0) return null;
  return STAGES.find((stage) => stages.includes(stage)) ?? null;
}

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

  /*
   * The three failure counts, kept apart.
   *
   * `ready` is the server's verdict on the *required* rows, and the footer used
   * to lump every other failure into "optional items missing for wider language
   * pairs" — which is a specific, wrong claim about a missing HF token. Split by
   * grade, each sentence is about the thing it names.
   */
  const blocking = failing.filter((check) => severityOf(check) === "blocking");
  const degraded = failing.filter((check) => severityOf(check) === "degrades");
  const optional = failing.filter((check) => severityOf(check) === "optional");
  const stopsAt = firstBlockingStage(checks);

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
              <p className="mt-2 text-lg font-semibold tracking-tight" data-readiness>
                {/* "All checks pass" is only true when they all do. `ready` is
                    the verdict on the BLOCKING rows, so it can be true with a
                    gated token and an un-downloaded cache still red above this
                    line — and that state is "ready to run", not "all pass". */}
                {status ? (
                  ready ? (
                    failing.length === 0 ? (
                      "All checks pass"
                    ) : (
                      "Ready to run"
                    )
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
            /*
             * Skipping is allowed and always was — the gate never traps anyone.
             * What it must not do is stay cheerful: with a blocking row red, the
             * next thing that happens after "Skip for now" is a run that dies,
             * and the link is the last place to say so before it does.
             */
            <Link
              to="/"
              replace
              data-skip
              className="rounded-md text-[12px] font-semibold text-secondary underline underline-offset-4 transition-colors hover:text-primary"
            >
              {blocking.length > 0
                ? `Skip anyway — runs will fail${stopsAt ? ` at ${stopsAt}` : ""}`
                : "Skip for now"}
            </Link>
          )}
          <span className="ml-auto max-w-xs text-[12px] leading-relaxed text-muted" data-footer>
            {/* `ready` is the server's verdict on the REQUIRED checks only, so it
                can be true while degraded and optional rows sit above this line —
                saying "nothing is missing" under a MISSING badge reads as a lie,
                and so does calling a gated token an "optional item for wider
                language pairs", which is what this used to do. */}
            {blocking.length > 0
              ? "A required tool is missing — runs will fail."
              : failing.length === 0
                ? "Nothing is missing."
                : degraded.length > 0
                  ? `Everything required is ready. ${degraded.length} ` +
                    `thing${degraded.length === 1 ? "" : "s"} above will still run — ` +
                    "just worse."
                  : `Everything required is ready; ${optional.length} optional ` +
                    `item${optional.length === 1 ? " is" : "s are"} missing for wider ` +
                    "language pairs."}
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
  const severity = severityOf(check);
  const meta = SEVERITY_META[severity];
  // A passing row is green whatever it would have cost; a failing one wears the
  // hue of what it costs. `--color-good` is the one case that is not a grade.
  const token = check.ok ? "var(--color-good)" : meta.token;
  return (
    <li
      className={cn(
        "flex items-start gap-3.5 px-6 py-4 sm:px-7",
        // Eight rows that look identical make the reader scan all eight to
        // find the two that need them. The failing ones get a wash and a rule
        // so the eye lands on them first — reinforcing the word and the glyph
        // that already say it, never replacing them. An *optional* failure gets
        // neither: it is the row the list is allowed to be quiet about, and
        // shouting it in red is what taught users to skim past the red.
        !check.ok && meta.wash && "shadow-[inset_3px_0_0_var(--wash)]",
      )}
      style={!check.ok && meta.wash
        ? ({
            "--wash": meta.token,
            backgroundColor: `color-mix(in srgb, ${meta.token} 4%, transparent)`,
          } as CSSProperties)
        : undefined}
      data-check={check.id}
      data-severity={severity}
    >
      <span
        aria-hidden
        className="mt-px grid h-6 w-6 shrink-0 place-items-center rounded-md border"
        style={{
          color: token,
          borderColor: `color-mix(in srgb, ${token} 35%, transparent)`,
          backgroundColor: `color-mix(in srgb, ${token} 10%, transparent)`,
        }}
      >
        <Glyph className="h-3.5 w-3.5" strokeWidth={3} />
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1">
          <span className="text-[13px] font-semibold text-primary">{check.label}</span>
          <Badge tone={check.ok ? "good" : "bad"}>{check.ok ? "Ready" : "Missing"}</Badge>
          {/* The consequence, spelled as a word beside the state. "Missing" is
              what is true; this is what it costs, and the two are different
              facts — a screenshot read at a glance needs both. Only on a
              failing row: what a passing check *would* have cost is trivia. */}
          {check.ok ? null : <Badge tone={meta.tone}>{meta.word}</Badge>}
          {!check.ok && severity === "blocking" && check.stage ? (
            <span className="text-[11px] text-muted">stops the run at {check.stage}</span>
          ) : null}
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
 * The detail sentence, with its backticked spans set as code — and copyable.
 *
 * The server writes these to be read by a human and marks the parts that are
 * meant to be typed — a path, a variable, a command — the way every other tool
 * on that machine does, with backticks. Rendering them literally put stray
 * punctuation in the middle of the one line on the screen whose whole job is
 * to be copied correctly.
 *
 * Setting them as code was half the fix. The other half is that these are
 * `uv run hf download …` commands and absolute `.env` paths — sixty characters
 * of exactness that a user was expected to select by dragging across a 11.5px
 * monospace span, in a *desktop shell* where there is no address bar to paste
 * into and no browser view-source to fall back on. One click is the whole
 * interaction, and a tick confirms it, because a clipboard write that says
 * nothing is indistinguishable from a click that missed.
 */
function Detail({ text }: { text: string }) {
  return (
    <>
      {text.split(/`([^`]+)`/g).map((part, i) =>
        i % 2 === 1 ? <CopyCode key={i} text={part} /> : <span key={i}>{part}</span>,
      )}
    </>
  );
}

/** How long the tick stays up. Long enough to be seen, short enough to not nag. */
const COPIED_MS = 1400;

function CopyCode({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<number | undefined>(undefined);

  useEffect(() => () => window.clearTimeout(timer.current), []);

  const copy = useCallback(() => {
    // `navigator.clipboard` is absent over plain HTTP on a non-localhost origin
    // and in jsdom. Failing silently would leave a button that looks like it
    // worked, so the tick is only shown when the write actually resolved.
    void navigator.clipboard
      ?.writeText(text)
      .then(() => {
        setCopied(true);
        window.clearTimeout(timer.current);
        timer.current = window.setTimeout(() => setCopied(false), COPIED_MS);
      })
      .catch(() => setCopied(false));
  }, [text]);

  return (
    <button
      type="button"
      onClick={copy}
      data-copy={text}
      title={copied ? "Copied" : `Copy "${text}"`}
      aria-label={`Copy ${text}`}
      className={cn(
        "inline-flex max-w-full items-center gap-1 rounded bg-sunken px-1 py-0.5 align-baseline",
        "font-mono text-[11.5px] text-primary transition-colors",
        "hover:bg-raised hover:ring-1 hover:ring-axis",
      )}
    >
      <code className="min-w-0 break-all">{text}</code>
      {copied ? (
        <Check aria-hidden className="h-3 w-3 shrink-0" style={{ color: "var(--color-good)" }} />
      ) : (
        <Copy aria-hidden className="h-3 w-3 shrink-0 text-muted" />
      )}
    </button>
  );
}
