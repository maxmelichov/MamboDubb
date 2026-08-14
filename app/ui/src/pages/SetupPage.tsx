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
 * - **Do not nag.** The gate in App.tsx routes here only when the server says
 *   `ok: false`. Otherwise this screen exists but is never in the way.
 */

import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ArrowRight, Loader2, RefreshCw } from "lucide-react";
import { AppHeader } from "../components/AppHeader";
import { Button, Panel, PanelHeader } from "../components/ui";
import { api } from "../lib/api";
import { isDesktop } from "../lib/desktop";
import type { SetupCheck, SetupStatus } from "../lib/types";

export function SetupPage() {
  const navigate = useNavigate();
  const [status, setStatus] = useState<SetupStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [checking, setChecking] = useState(true);

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

  const failing = status?.checks.filter((check) => !check.ok) ?? [];
  const ready = status?.ok === true;

  return (
    <div className="flex h-screen flex-col">
      <AppHeader />
      <main className="mx-auto w-full max-w-3xl flex-1 overflow-y-auto p-6">
        <h1 className="text-lg font-semibold">Setup</h1>
        <p className="mt-1 max-w-prose text-[13px] text-secondary">
          Everything runs on this machine, so the machine has to have it. These are cheap
          filesystem checks — no model is loaded to answer them.
          {isDesktop() ? null : " Re-check after installing anything."}
        </p>

        <Panel className="mt-4">
          <PanelHeader>
            <span>Readiness</span>
            <span className="ml-auto font-normal normal-case tracking-normal text-[11px]">
              {status ? (
                ready ? (
                  "All checks pass"
                ) : (
                  <>
                    {failing.length} of {status.checks.length} need attention
                  </>
                )
              ) : checking ? (
                "Checking…"
              ) : (
                "No answer"
              )}
            </span>
          </PanelHeader>

          {error && !status ? (
            <div className="p-3 text-[13px] text-secondary">
              <p>
                Could not ask the server what is installed:{" "}
                <span className="text-primary">{error}</span>
              </p>
              <p className="mt-1 text-muted">
                Either it is not running yet, or it is an older build without{" "}
                <code className="font-mono text-[12px]">/api/setup</code>. The rest of the app works
                either way.
              </p>
            </div>
          ) : null}

          {status ? (
            <ul className="divide-y divide-border">
              {status.checks.map((check) => (
                <CheckRow key={check.id} check={check} />
              ))}
            </ul>
          ) : null}

          <div className="flex items-center gap-2 border-t border-border p-3">
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
                className="text-[12px] text-secondary underline underline-offset-2 hover:text-primary"
              >
                Skip for now
              </Link>
            )}
            <span className="ml-auto text-[12px] text-muted">
              {ready
                ? "Nothing is missing."
                : "A missing model fails the stage that needs it, not the whole run."}
            </span>
          </div>
        </Panel>
      </main>
    </div>
  );
}

/**
 * One check. The state is spelled three ways — glyph, word, hue — because the
 * palette's "kept"/"failed" pair is below the contrast gate for colour-vision
 * deficiency in light mode, and because a screenshot of a checklist gets read
 * at a glance by people who never see the hue at all.
 */
function CheckRow({ check }: { check: SetupCheck }) {
  const token = check.ok ? "var(--color-good)" : "var(--color-critical)";
  return (
    <li className="flex items-start gap-3 px-3 py-2.5" data-check={check.id}>
      <span
        aria-hidden
        style={{ color: token }}
        className="mt-px w-4 shrink-0 text-center text-[13px] leading-5"
      >
        {check.ok ? "✓" : "✕"}
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-baseline gap-x-2">
          <span className="text-[13px] font-medium">{check.label}</span>
          <span className="text-[11px] font-semibold uppercase tracking-[0.1em] text-muted">
            {check.ok ? "Ready" : "Missing"}
          </span>
        </div>
        {check.detail ? (
          <p className="mt-0.5 text-[12px] leading-snug text-secondary">{check.detail}</p>
        ) : null}
      </div>
    </li>
  );
}
