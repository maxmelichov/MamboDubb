import { useEffect, useState } from "react";
import { Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { BootPanel } from "./components/BootPanel";
import { EditorPage } from "./pages/EditorPage";
import { ImportPage } from "./pages/ImportPage";
import { RunsPage } from "./pages/RunsPage";
import { SetupPage } from "./pages/SetupPage";
import { api, initApiBase, USE_FIXTURES } from "./lib/api";
import { isDesktop } from "./lib/desktop";
import "./App.css";

export default function App() {
  const boot = useBoot();
  if (boot !== "ready") return <BootPanel failed={boot === "failed"} />;
  return (
    <>
      <SetupGate />
      <Routes>
        {/* Home is the runs list; the import form moved to /new. A workspace
            with zero runs still lands on "/" — RunsPage's empty state is the
            invitation to start one, so no route-level redirect is needed. */}
        <Route path="/" element={<RunsPage />} />
        <Route path="/new" element={<ImportPage />} />
        <Route path="/setup" element={<SetupPage />} />
        <Route path="/editor/:name" element={<EditorPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </>
  );
}

type BootPhase = "starting" | "ready" | "failed";

/**
 * Resolve the api base before any route exists `mediaUrl` is called during
 * render and a <video src> cannot await, so the routes must not mount until
 * `initApiBase` has answered.
 *
 * In a browser the answer is "" (same-origin) after one microtask; the gate is
 * pre-resolved to `ready` and no panel ever flashes. Only in the desktop shell
 * is there a real wait: `initApiBase` reaches `start_server`, and on a fresh
 * install that command hides a payload copy plus a multi-minute `uv sync`.
 * Gating here rather than in main.tsx (which used to await before the first
 * render) is the whole fix a window now exists during the wait, and
 * BootPanel fills it with the first-run explanation and the live server log.
 *
 * `failed` is an inference, not an error object: the desktop seam maps a
 * rejected `start_server` to a null base URL, so in the shell an empty base
 * *is* the failure signal. The reason lives in the server log, which the
 * runner tops up on rejection and BootPanel shows.
 */
function useBoot(): BootPhase {
  const gated = isDesktop() && !USE_FIXTURES;
  const [phase, setPhase] = useState<BootPhase>(gated ? "starting" : "ready");

  useEffect(() => {
    if (!gated) {
      // Still resolves the base ("" in a browser) so api.ts is initialized on
      // every path, not just the gated one.
      void initApiBase();
      return;
    }
    let cancelled = false;
    void initApiBase().then((base) => {
      if (!cancelled) setPhase(base ? "ready" : "failed");
    });
    return () => {
      cancelled = true;
    };
    // `gated` cannot change within a page load: it is a platform sniff.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return phase;
}

/** Asked once per app load, not once per mount StrictMode double-mounts. */
let gateChecked = false;

/**
 * Route to /setup on the first load if and only if the server says the
 * machine is not ready.
 *
 * Three things it deliberately does not do. It does not block the first paint:
 * the import screen renders immediately and this replaces it a moment later if
 * it has to, which is better than a spinner on every boot for the 99% of loads
 * where everything is fine. It does not treat an error as a failure: an old
 * server without /api/setup, or one still starting, tells us nothing, and
 * nothing is not a reason to interrupt. And it does not run in fixture mode,
 * where the checklist is a demo with deliberate failures in it.
 */
function SetupGate() {
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    if (USE_FIXTURES || gateChecked) return;
    gateChecked = true;
    let cancelled = false;
    void api
      .setup()
      .then((status) => {
        if (cancelled || status.ok) return;
        if (location.pathname === "/setup") return;
        navigate("/setup", { replace: true });
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
    // Deliberately once, on mount: this is a boot decision, not a route guard.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return null;
}
