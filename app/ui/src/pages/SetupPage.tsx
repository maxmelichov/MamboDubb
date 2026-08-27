/**
 * Setup the first-run screen, and the place to come back to when something
 * on the machine changed underneath the app.
 *
 * `GET /api/setup` is a list of fast filesystem checks: the two binaries, each
 * model directory, free disk. This screen is a checklist of them
 * and nothing else. Three rules it follows, all of them the editor's:
 *
 * - **Never colour alone.** Every row carries a glyph, a word ("Ready",
 *   "Missing", "Not installed"), and a hue in that order of importance. A
 *   monochrome screen reads exactly the same.
 * - **A provisioned machine is all green, and nothing is amber by design.**
 *   The only rows that are not green on a machine with everything installed are
 *   the ones where something genuinely is not there and the only *grade* that
 *   survives that with nothing to do about it is `optional`, which is drawn as
 *   a grey dash, the words "Not installed", and no wash. It is not counted in
 *   the "N of M need attention" headline and it cannot hold `ok` back (the
 *   server's `report` conjoins the required rows only). SoX is the row this
 *   exists for: the shipped pipeline never calls it.
 * - **A failure says what it costs.** Every missing row used to be the same red
 *   X, so absent diarization weights (the run works, everyone in the video
 *   becomes one speaker) looked exactly like a missing ffmpeg (nothing runs at
 *   all), which looked exactly like a Korean TTS checkpoint a Hebrew→English run
 *   will never open. The server now grades them `severity` is `blocking`,
 *   `degrades` or `optional` and each grade gets its own word: REQUIRED,
 *   DEGRADES, OPTIONAL. The hue follows the word, never the other way round.
 * - **The detail line is the whole point.** "Missing" is not actionable; "run
 *   this command to fetch it (320 MB)" is. The server writes that sentence, the
 *   UI only renders it and every command in it is one click to the clipboard,
 *   because a command that has to be retyped from a screenshot is a command
 *   that gets retyped wrong.
 * - **A row the server can fix gets a button.** `installable` comes from the
 *   server (the key set of `install.INSTALLERS` plus `setup.model_downloads()`),
 *   never from a list kept here a copy would drift and put a button on a row
 *   whose POST is a 400. Two binaries qualify, and so do the hub-snapshot
 *   models: a fresh install is mostly red model rows, and for a user who chose
 *   a DMG so as never to open a terminal, the Download button IS the happy
 *   path. The button carries the price ("Download · ~9.7 GB"), the poll draws
 *   a real bar (bytes on disk against the estimate), and a torn-off attempt
 *   resumes the server keeps the partial files, and the Retry says so.
 *   Where no snapshot can help the self-fetching caches, Demucs and the Hebrew
 *   G2P the detail line stays the whole answer.
 * - **Do not nag.** The gate in App.tsx routes here only when the server says
 *   `ok: false`. Otherwise this screen exists but is never in the way.
 *
 * Visually it borrows MamboRambo's onboarding: a summary band with a progress
 * bar over a divided list, and the actions in a sunken footer under it, so the
 * screen answers "how far off am I" before it answers "what exactly is wrong".
 */

import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ArrowRight, Check, Copy, Download, Loader2, Minus, RefreshCw, X } from "lucide-react";
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
import type { SetupInstallState } from "../lib/api";
import { cn } from "../lib/classNames";
import { STAGES } from "../lib/types";
import type { SetupCheck, SetupInstall, SetupSeverity, SetupStatus } from "../lib/types";

/**
 * What a failing row actually costs, with a fallback for a server that predates
 * the field.
 *
 * That fallback reads `required` as blocking and everything else as optional,
 * which is the *old* two-value contract the one reading that can understate a
 * failure but never overstate one. An unknown grade defaulting to "degrades"
 * would invent a claim about a check nobody has graded.
 */
function severityOf(check: SetupCheck): SetupSeverity {
  return check.severity ?? (check.required === false ? "optional" : "blocking");
}

const SEVERITY_META: Record<
  SetupSeverity,
  {
    word: string;
    token: string;
    tone: "bad" | "warn" | "neutral";
    wash: boolean;
    /** What a *failing* row of this grade says where a red one says "Missing". */
    state: string;
  }
> = {
  // Nothing runs. Red, a wash, and the word people already read as "stop".
  blocking: {
    word: "Required", token: "var(--color-critical)", tone: "bad", wash: true,
    state: "Missing",
  },
  // The run works and is worse. Amber: it must not be dismissible at a glance,
  // and it must not read as broken either.
  degrades: {
    word: "Degrades", token: "var(--color-warning)", tone: "warn", wash: true,
    state: "Missing",
  },
  /*
   * Irrelevant until you ask for it. No hue, no wash, and — the part that was
   * still wrong — no red MISSING badge either.
   *
   * SoX is the row this is about. Nothing the shipped pipeline runs touches it
   * (see `setup.TOOLS`), the server has graded it `optional` for exactly that
   * reason, and the screen still drew a brewless machine a red X and the word
   * "Missing" beside a second chip reading "Optional". A checklist that says
   * MISSING · OPTIONAL in the same breath is asking the reader to work out
   * which half to believe, and a user reading "the setup is not all green"
   * off a tool the app will never call is the whole failure. One quiet chip
   * that says the true thing instead, and the row is not counted as needing
   * attention anywhere on the screen.
   */
  optional: {
    word: "Optional", token: "var(--color-muted)", tone: "neutral", wash: false,
    state: "Not installed",
  },
};

/** Does this failing row want the user to do something? Optional ones do not. */
function needsAttention(check: SetupCheck): boolean {
  return !check.ok && severityOf(check) !== "optional";
}

/**
 * `9700000000` → `9.7 GB`, the same rounding as the server's `human_bytes`.
 *
 * The server formats the sizes it puts in *sentences* (`detail`), but the
 * numbers `download_bytes`, `bytes_done` arrive raw because the UI has to
 * do arithmetic on them (a percent, a sum for the header line) before it can
 * say them. One formatter, matched to the server's, so "~9.7 GB" on the button
 * and "9.7 GB" in the detail line never disagree about the same model.
 */
function humanBytes(n: number): string {
  for (const unit of ["B", "KB", "MB", "GB"] as const) {
    if (n < 1024) return unit === "B" || unit === "KB" ? `${Math.round(n)} ${unit}` : `${n.toFixed(1)} ${unit}`;
    n /= 1024;
  }
  return `${n.toFixed(1)} TB`;
}

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
 * There is no stream for this setup has no project, so it has no event
 * stream and a `brew install` runs for minutes, so two seconds is honest and
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
  const [install, setInstall] = useState<SetupInstallState | null>(null);
  const [installError, setInstallError] = useState<string | null>(null);

  const recheck = useCallback(async () => {
    setChecking(true);
    try {
      setStatus(await api.setup());
      setError(null);
    } catch (err) {
      // An old server has no /api/setup at all. That is not a failed check, it
      // is no information say so rather than inventing a red row.
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
   * Pick up an install that was already running when this screen mounted.
   *
   * A model download is tens of minutes on real bandwidth, and the user will
   * navigate away and back or reload the shell before it finishes. The slot
   * lives in the server, so the only thing lost by a reload was the *display*
   * of it and a screen that shows a Download button for a model the server
   * is half-way through fetching invites the exact double-start the 409 exists
   * to refuse. One GET on mount; a server that has never run anything answers
   * `running: false` and this is a no-op.
   */
  useEffect(() => {
    void api
      .installStatus()
      .then((current) => setInstall((prev) => prev ?? current))
      .catch(() => {
        // An older server without the endpoint: nothing was running anyway.
      });
  }, []);

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
  const queue = install?.queue ?? null;
  const queueRunning = queue?.running === true;
  /*
   * The queue polls on the same clock, and needs its own reason to keep going:
   * between two items the slot is briefly idle, and a poll that stopped there
   * would leave the screen frozen on item two of nine with nothing moving.
   */
  const polling = running || queueRunning;
  const rechecking = useRef(false);
  // Which item the queue was on at the last poll. A queue turns rows green one
  // at a time, and the checklist has to follow it down the list rather than
  // waiting for the whole thing to finish.
  const lastPos = useRef<number | null>(null);
  useEffect(() => {
    if (!polling) return;
    let live = true;
    const timer = setInterval(() => {
      void api
        .installStatus()
        .then((next) => {
          if (!live) return;
          setInstall(next);
          const advanced = next.queue != null && next.queue.pos !== lastPos.current;
          lastPos.current = next.queue?.pos ?? null;
          const settled = !next.running && next.queue?.running !== true;
          if ((settled || advanced) && !rechecking.current) {
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
  }, [polling, recheck]);

  const startInstall = useCallback(async (id: string) => {
    setInstallError(null);
    try {
      setInstall(await api.startInstall(id));
    } catch (err) {
      setInstallError(String(err instanceof Error ? err.message : err));
    }
  }, []);

  const startEverything = useCallback(async () => {
    setInstallError(null);
    try {
      setInstall(await api.startInstallAll());
    } catch (err) {
      setInstallError(String(err instanceof Error ? err.message : err));
    }
  }, []);

  const cancelEverything = useCallback(async () => {
    try {
      setInstall(await api.cancelInstallAll());
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
   * pairs" which is a specific, wrong claim about most of them. Split by
   * grade, each sentence is about the thing it names.
   */
  const blocking = failing.filter((check) => severityOf(check) === "blocking");
  const degraded = failing.filter((check) => severityOf(check) === "degrades");
  const optional = failing.filter((check) => severityOf(check) === "optional");
  // What the headline counts. A missing optional tool is not a thing to attend
  // to — it is a row saying "you did not need this", and counting it was how a
  // fully provisioned machine still met a number at the top of the screen.
  const attention = checks.filter(needsAttention);
  const stopsAt = firstBlockingStage(checks);

  /*
   * The one-time cost, added up.
   *
   * A fresh install is several red model rows, each with its own gigabytes,
   * and a user meeting that list for the first time has one question the rows
   * answer only by mental arithmetic: how much, in total, before this app
   * works? One line at the top "3 models to download, ~28 GB total" is
   * that answer, and saying it once up front is what makes each row's own
   * button read as a step rather than a fresh demand.
   */
  const toDownload = failing.filter((check) => check.installable && check.hub);
  const downloadTotal = toDownload.reduce((sum, check) => sum + (check.download_bytes ?? 0), 0);

  /*
   * What one button would install, and what it would cost.
   *
   * The same rule the server applies in `setup.install_plan` — missing, fixable
   * from here, and not graded `optional` — computed again on this side for one
   * reason only: to *price the button before it is pressed*. The list that
   * actually runs is the server's, from a fresh report at the moment of the
   * click, so the two cannot drift into a button that installs something the
   * screen did not name; the worst a stale count here can do is a label that is
   * one row out of date, which the first poll corrects.
   */
  const plan = failing.filter(
    (check) => check.installable === true && severityOf(check) !== "optional",
  );
  const planBytes = plan.reduce((sum, check) => sum + (check.download_bytes ?? 0), 0);

  return (
    // No hero — the nav pill says Setup and the Readiness card leads.
    <PageShell>
      <Card className="overflow-hidden p-0">
        <CardSection className="pb-4">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <Eyebrow>Readiness</Eyebrow>
              <p className="mt-2 text-lg font-semibold tracking-tight" data-readiness>
                {/* "All checks pass" is only true when they all do. `ready` is
                    the verdict on the BLOCKING rows, so it can be true with an
                    un-downloaded cache and a Korean checkpoint still red above
                    this line and that state is "ready to run", not "all pass". */}
                {status ? (
                  ready ? (
                    failing.length === 0 ? (
                      "All checks pass"
                    ) : (
                      "Ready to run"
                    )
                  ) : (
                    <>
                      {attention.length} of {checks.length} need attention
                    </>
                  )
                ) : checking ? (
                  "Checking…"
                ) : (
                  "No answer"
                )}
              </p>
              {toDownload.length > 0 ? (
                <p className="mt-1 text-[12px] leading-relaxed text-secondary" data-downloads>
                  {toDownload.length === 1
                    ? `1 model to download, ~${humanBytes(downloadTotal)}`
                    : `${toDownload.length} models to download, ~${humanBytes(downloadTotal)} total`}{" "}
                  <span className="text-muted">(one time, from inside the app)</span>
                </p>
              ) : null}
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
          <InstallAll
            plan={plan}
            planBytes={planBytes}
            queue={queue}
            install={install}
            busy={running && !queueRunning}
            onStart={() => void startEverything()}
            onCancel={() => void cancelEverything()}
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
                // Greyed out for the gap between two queued items as well as
                // for the install itself: the slot is idle for a moment there,
                // and a click that lands in it is a 409 the user cannot see
                // coming.
                busy={running || queueRunning}
                onInstall={startInstall}
                onRecheck={recheck}
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
             * Skipping is allowed and always was the gate never traps anyone.
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
                ? `Skip anyway: runs will fail${stopsAt ? ` at ${stopsAt}` : ""}`
                : "Skip for now"}
            </Link>
          )}
          <span className="ml-auto max-w-xs text-[12px] leading-relaxed text-muted" data-footer>
            {/* `ready` is the server's verdict on the REQUIRED checks only, so it
                can be true while degraded and optional rows sit above this line:
                saying "nothing is missing" under a MISSING badge reads as a lie,
                and so does filing a real defect under "optional items for wider
                language pairs", which is what this used to do. */}
            {blocking.length > 0
              ? "A required tool is missing: runs will fail."
              : failing.length === 0
                ? "Nothing is missing."
                : degraded.length > 0
                  ? `Everything required is ready. ${degraded.length} ` +
                    `thing${degraded.length === 1 ? "" : "s"} above will still run, ` +
                    "just worse."
                  : `Everything required is ready; ${optional.length} optional ` +
                    `item${optional.length === 1 ? " is" : "s are"} not installed, ` +
                    `and nothing needs ${optional.length === 1 ? "it" : "them"}.`}
          </span>
        </CardSection>
      </Card>

      {error && status ? <ErrorBlock title="Re-check failed">{error}</ErrorBlock> : null}
      {/* A refusal an id the server has no argv for, a second install while
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
 * The one button — and, while it runs, the only place that says where the queue
 * has got to.
 *
 * Nine red rows on a fresh machine is nine buttons, nine visits back to this
 * screen and forty minutes of babysitting a thing the app already knows how to
 * do in order. This is that order, offered once. Four rules, and each of them
 * is what stops a "do everything" button from being a blind one:
 *
 * - **It says what it costs before it is pressed.** The same total the rows
 *   already carry (`download_bytes`), added up. A button that said only
 *   "Install everything" would be the blindness the model downloads were
 *   refused over in the first place.
 * - **It never installs what nothing needs.** Blocking and degraded rows only —
 *   the ones the headline counts. A Korean checkpoint is not part of
 *   "everything" for a Hebrew→English run, and quietly fetching 40 GB of it
 *   would make the price tag a lie.
 * - **In flight it is one item, named, with the row's own progress.** "n of m",
 *   the label, and the same bar the row draws, because there is exactly one
 *   install running and two different-looking progress bars for it would be two
 *   claims to reconcile.
 * - **Cancel stops the queue, not the download.** The item in flight finishes;
 *   nothing after it starts. Said in those words while it winds down, so
 *   "Cancel" is never read as "kill this 9 GB fetch half way".
 *
 * The rows keep their own buttons throughout: this is the shortcut, not the
 * replacement, and a user who wants the translator and nothing else still has
 * the row that does exactly that.
 */
function InstallAll({
  plan,
  planBytes,
  queue,
  install,
  busy,
  onStart,
  onCancel,
}: {
  /** The rows one press would install, in the screen's order. */
  plan: SetupCheck[];
  planBytes: number;
  queue: SetupInstallState["queue"];
  install: SetupInstall | null;
  /** A single row's install holds the slot — the server refuses a queue behind it. */
  busy: boolean;
  onStart: () => void;
  onCancel: () => void;
}) {
  const running = queue?.running === true;
  // Nothing missing that this can fix: no button, no line, no leftovers from a
  // queue that already finished. An all-green screen says nothing about installs.
  if (!running && plan.length === 0) return null;

  if (running && queue) {
    const current = queue.items[Math.min(queue.pos, queue.items.length - 1)];
    return (
      <div className="mt-4 rounded-lg border border-border bg-sunken px-4 py-3" data-install-all-panel>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <span className="text-[12px] font-semibold text-primary" data-queue-position>
            Installing {Math.min(queue.pos + 1, queue.total)} of {queue.total}
            <span className="font-normal text-secondary"> · {current?.label ?? ""}</span>
          </span>
          <Button size="sm" onClick={onCancel} disabled={queue.cancelled} data-install-all-cancel>
            <X className="h-3.5 w-3.5" />
            {queue.cancelled ? "Cancelling" : "Cancel"}
          </Button>
        </div>
        {install ? <InstallProgress install={install} /> : null}
        <p className="mt-1.5 text-[11px] leading-relaxed text-muted" data-queue-remaining>
          {queue.cancelled
            ? "Stopping after this one. It finishes, nothing after it starts."
            : queue.remaining_bytes > 0
              ? `~${humanBytes(queue.remaining_bytes)} still to fetch, one at a time.`
              : "One at a time, in the order above."}
          {queue.failed.length > 0
            ? ` ${queue.failed.length} did not finish. Those rows say why.`
            : ""}
        </p>
      </div>
    );
  }

  // Idle, with things to install. The count and the price are the whole label.
  const failed = queue && !queue.running ? queue.failed.length : 0;
  return (
    <div className="mt-4 flex flex-wrap items-center gap-x-3 gap-y-2">
      <Button
        variant="primary"
        onClick={onStart}
        disabled={busy}
        data-install-all
        title={busy ? "One install at a time: wait for the running one" : undefined}
      >
        <Download className="h-3.5 w-3.5" />
        Install everything{planBytes > 0 ? ` · ~${humanBytes(planBytes)}` : ""}
      </Button>
      <span className="text-[12px] leading-relaxed text-muted" data-install-all-note>
        {plan.length === 1
          ? "The one thing missing that the app can fetch itself."
          : `${plan.length} things, one at a time, required first.`}{" "}
        {failed > 0
          ? `${failed} did not finish last time. Starting again resumes what is there.`
          : "Nothing optional is included."}
      </span>
    </div>
  );
}

/**
 * How far into the one install the server has got, in the one honest shape for
 * each kind.
 *
 * A *tool* install gets a spinner, the word, and the last line of output — an
 * install is minutes long and the poll is seconds long, so the last line is the
 * only progress there is and a bar would have to invent the fraction.
 *
 * A *model download* gets a real bar, because here the fraction is not
 * invented: the server re-walks the target directory each poll (`bytes_done`)
 * against the table's estimate (`bytes_total`). The estimate can undershoot, so
 * the fraction is clamped — a bar at 101% reads as a bug in the one minute the
 * download is finishing.
 *
 * Shared by the row and the queue header on purpose: there is exactly one
 * install running at any moment, and two differently-drawn views of it would be
 * two claims about the same bytes.
 */
function InstallProgress({ install }: { install: SetupInstall }) {
  if (install.bytes_done != null) {
    return (
      <div className="mt-2 max-w-md" data-download-progress>
        <div className="flex items-baseline justify-between gap-3 text-[12px]">
          <span className="flex items-center gap-2 font-semibold text-secondary">
            <Loader2 aria-hidden className="h-3 w-3 shrink-0 animate-spin" />
            Downloading…
          </span>
          <span className="font-mono text-[11.5px] tabular-nums text-muted">
            {humanBytes(install.bytes_done)}
            {install.bytes_total
              ? ` of ~${humanBytes(install.bytes_total)} · ${Math.min(
                  100,
                  Math.floor((install.bytes_done / install.bytes_total) * 100),
                )}%`
              : ""}
          </span>
        </div>
        <Progress
          className="mt-1.5"
          value={install.bytes_total ? Math.min(1, install.bytes_done / install.bytes_total) : null}
        />
      </div>
    );
  }
  return (
    <p className="mt-1.5 flex items-center gap-2 text-[12px]">
      <Loader2 aria-hidden className="h-3 w-3 shrink-0 animate-spin text-secondary" />
      <span className="shrink-0 font-semibold text-secondary">Installing…</span>
      <span className="truncate font-mono text-[11.5px] text-muted">
        {lastLine(install) ?? "starting"}
      </span>
    </p>
  );
}

/**
 * One check. The state is spelled three ways glyph, word, hue because the
 * palette's "kept"/"failed" pair is below the contrast gate for colour-vision
 * deficiency in light mode, and because a screenshot of a checklist gets read
 * at a glance by people who never see the hue at all.
 */
function CheckRow({
  check,
  install,
  busy,
  onInstall,
  onRecheck,
}: {
  check: SetupCheck;
  /** This row's install, or null the page hands each row only its own. */
  install: SetupInstall | null;
  /** Any install is running. One at a time, so every other button greys out. */
  busy: boolean;
  onInstall: (id: string) => void;
  /** Re-run the whole checklist, because a row's control changed the machine. */
  onRecheck: () => void;
}) {
  const severity = severityOf(check);
  const meta = SEVERITY_META[severity];
  // A cross is "this is wrong". An optional row that is simply not here is not
  // wrong, so it gets a dash the glyph a checklist uses for "not applicable".
  const Glyph = check.ok ? Check : severity === "optional" ? Minus : X;
  const installing = install?.running === true;
  const failed = install !== null && !install.running && install.ok === false;
  // The button is offered for exactly one state: a row the server says it can
  // fix, that is currently broken. A passing row needs nothing and a model row
  // has no argv behind it its detail line is the answer.
  const offerInstall = check.installable === true && !check.ok;
  // A passing row is green whatever it would have cost; a failing one wears the
  // hue of what it costs. `--color-good` is the one case that is not a grade.
  const token = check.ok ? "var(--color-good)" : meta.token;
  return (
    <li
      className={cn(
        "flex items-start gap-3.5 px-6 py-4 sm:px-7",
        // Eight rows that look identical make the reader scan all eight to
        // find the two that need them. The failing ones get a wash and a rule
        // so the eye lands on them first reinforcing the word and the glyph
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
          <Badge tone={check.ok ? "good" : meta.tone}>
            {check.ok ? "Ready" : meta.state}
          </Badge>
          {/* The consequence, spelled as a word beside the state. "Missing" is
              what is true; this is what it costs, and the two are different
              facts a screenshot read at a glance needs both. Only where the
              two are different facts: a passing check's cost is trivia, and an
              optional row's state chip already reads "Not installed", so a
              second chip saying "Optional" beside it is the same word twice. */}
          {check.ok || severity === "optional" ? null : (
            <Badge tone={meta.tone}>{meta.word}</Badge>
          )}
          {/* …which leaves the grade itself to be said in plain words, because
              a screenshot read at a glance still has to carry it. */}
          {!check.ok && severity === "optional" ? (
            <span className="text-[11px] text-muted">Optional: nothing here needs it</span>
          ) : null}
          {!check.ok && severity === "blocking" && check.stage ? (
            <span className="text-[11px] text-muted">stops the run at {check.stage}</span>
          ) : null}
        </div>
        {check.detail ? (
          <p className="mt-1 max-w-2xl text-[12px] leading-relaxed text-secondary">
            <Detail text={check.detail} />
          </p>
        ) : null}
        {/* The one row that is a choice rather than a finding. An escape
            hatch, same contract: the control changes a file on the server and
            the re-probed row is what says what changed. */}
        {check.id === "low_vram" ? <LowVramField check={check} onChanged={onRecheck} /> : null}
        {/* This row's install, while it runs — the same two shapes the queue
            header draws, from the same poll (see `InstallProgress`). The row is
            where it belongs when a row's own button started it; when the queue
            did, both are on screen and they are one install, drawn twice. */}
        {installing ? <InstallProgress install={install} /> : null}
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
            {/* Only under a torn-off download: the fear Retry has to answer is
                "do I pay the gigabytes again?", and the answer is the server's
                (`snapshot_download` keeps partial files), said where the user
                decides whether to press it. A failed brew has no partials and
                no such fear, so it gets no such sentence. */}
            {check.hub ? (
              <p className="mt-1 text-[12px] text-muted">
                Resumes where it left off: the bytes already downloaded are kept.
              </p>
            ) : null}
          </>
        ) : null}
      </div>
      {/* The button is *replaced* by the progress line above while this row is
          installing two spinners on one row is one spinner too many. Every
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
          {failed ? (
            <RefreshCw className="h-3.5 w-3.5" />
          ) : (
            <Download className="h-3.5 w-3.5" />
          )}
          {/* The label is the price tag. A tool is seconds and says "Install";
              a model is gigabytes and says so before the click, from the same
              `download_bytes` the server puts in the detail sentence. After a
              failure it says "Retry" the resume note beside the error is
              what makes that word safe to press. */}
          {failed
            ? "Retry"
            : check.hub
              ? `Download${check.download_bytes ? ` · ~${humanBytes(check.download_bytes)}` : ""}`
              : "Install"}
        </Button>
      ) : null}
    </li>
  );
}

/*
 * There is no token field on this screen any more, and no `hf_token` row for it
 * to hang off. It used to be the fix for a real problem: diarization loaded a
 * gated repo, and without a token every character in the video was dubbed in
 * one voice. Since v0.4.0 the same CC-BY-4.0 weights ship inside the app
 * (`segments.DIARIZATION_DIR`, and the app can put them back itself), so the
 * row could only ever say "optional, nothing here needs it" while showing a
 * paste box and a NOT INSTALLED badge. That is a checklist advertising a
 * problem nobody has, and the paragraph explaining why it did not matter was
 * the tell. `HF_TOKEN` and `DUB_DIARIZATION_HUB` still work in `.env` for
 * anyone who wants the gated upstream repo; they are settings, not setup.
 */

/**
 * Low-VRAM mode, as two buttons and a sentence about what it costs.
 *
 * Not a bare switch, because the honest version of this control is not "on/off"
 * but "on, off, or let the machine decide", and the third position is the
 * default a user should mostly stay in. Auto is what the row already says in
 * words, so the two buttons are the two ways to overrule it, and the one that
 * matches the current answer is the one that reads as pressed.
 *
 * The cost is named on the control itself rather than left to the docs. This is
 * the only setting in the app that trades output quality for hardware, and a
 * toggle labelled "Low-VRAM mode" with no consequence beside it is an invitation
 * to turn it on for no reason: the smaller weights lose names and idioms that
 * the default gets right, which is a thing to accept deliberately.
 *
 * Two states this cannot change, and both say so instead of failing:
 * a value exported in the server's own environment (`source === "env"`) beats
 * any file this writes, and a change lands on the next job rather than on one
 * already running.
 */
function LowVramField({ check, onChanged }: { check: SetupCheck; onChanged: () => void }) {
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const enabled = check.enabled === true;
  const fromEnv = check.source === "env";

  const choose = useCallback(
    async (next: boolean) => {
      if (saving) return;
      setSaving(true);
      setError(null);
      try {
        await api.setLowVram(next);
        onChanged();
      } catch (err) {
        setError(String(err instanceof Error ? err.message : err));
      } finally {
        setSaving(false);
      }
    },
    [saving, onChanged],
  );

  return (
    <div className="mt-2 max-w-2xl" data-low-vram>
      <p className="text-[12px] leading-relaxed text-secondary">
        Smaller translator weights, so the 12B translator fits on an ordinary graphics card (about
        7 GB instead of 24) or a 16 GB Mac. It costs translation quality: on Hebrew the smaller
        weights lose idioms, drift on the spelling of a name, and sometimes borrow a name from the
        context that the line never mentioned. Leave it off if the full-size weights fit.
      </p>
      {fromEnv ? (
        <p className="mt-1.5 text-[12px] leading-relaxed text-muted">
          Set by an environment variable, which the app cannot change from here. Unset
          DUBBING_LOW_VRAM to choose it in the app.
        </p>
      ) : (
        <>
          <div className="mt-2 flex items-center gap-2">
            <Button
              size="sm"
              data-low-vram-off
              aria-pressed={!enabled}
              disabled={saving || !enabled}
              onClick={() => void choose(false)}
            >
              Full weights
            </Button>
            <Button
              size="sm"
              data-low-vram-on
              aria-pressed={enabled}
              disabled={saving || enabled}
              onClick={() => void choose(true)}
            >
              {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}
              Low VRAM
            </Button>
          </div>
          <p className="mt-1.5 text-[12px] text-muted">
            Takes effect on the next run, not on one already going.
          </p>
        </>
      )}
      {error ? (
        <p
          className="mt-1.5 text-[12px] leading-relaxed"
          style={{ color: "var(--color-critical)" }}
          data-low-vram-error
        >
          {error}
        </p>
      ) : null}
    </div>
  );
}

/** The most recent line of an install's output, if it has produced one. */
function lastLine(install: SetupInstall): string | null {
  return install.tail.length ? install.tail[install.tail.length - 1] : null;
}

/**
 * The detail sentence, with its backticked spans set as code and copyable.
 *
 * The server writes these to be read by a human and marks the parts that are
 * meant to be typed a path, a variable, a command the way every other tool
 * on that machine does, with backticks. Rendering them literally put stray
 * punctuation in the middle of the one line on the screen whose whole job is
 * to be copied correctly.
 *
 * Setting them as code was half the fix. The other half is that these are
 * `uv run hf download …` commands and absolute `.env` paths sixty characters
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
