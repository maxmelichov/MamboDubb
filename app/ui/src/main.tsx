import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { applyTheme, currentTheme } from "./lib/theme";

// Before anything renders. index.html's boot script has normally already done
// this but it is an inline script, and an inline script is exactly the thing
// a CSP, an extension or a headless renderer can decline to run. Applying it
// again here costs one class toggle and removes the whole failure mode.
applyTheme(currentTheme());

const root = ReactDOM.createRoot(document.getElementById("root") as HTMLElement);

// Render immediately; App's boot gate (`useBoot`) resolves the api base and
// keeps the routes unmounted until it has. It used to be an await here, which
// was correct about ordering (`mediaUrl` is called during render and a
// <video src> cannot await) but meant *nothing* rendered while the desktop
// shell's start_server ran and on a fresh install that is a payload copy
// plus a multi-minute `uv sync`, spent as a blank window. The gate keeps the
// ordering and gives the wait a surface (components/BootPanel.tsx).
root.render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>,
);
