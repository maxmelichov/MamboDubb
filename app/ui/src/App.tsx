import { useEffect } from "react";
import { Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { EditorPage } from "./pages/EditorPage";
import { ImportPage } from "./pages/ImportPage";
import { SetupPage } from "./pages/SetupPage";
import { api, USE_FIXTURES } from "./lib/api";
import "./App.css";

export default function App() {
  return (
    <>
      <SetupGate />
      <Routes>
        <Route path="/" element={<ImportPage />} />
        <Route path="/setup" element={<SetupPage />} />
        <Route path="/editor/:name" element={<EditorPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </>
  );
}

/** Asked once per app load, not once per mount — StrictMode double-mounts. */
let gateChecked = false;

/**
 * Route to /setup on the first load if — and only if — the server says the
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
