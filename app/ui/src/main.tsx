import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { initApiBase } from "./lib/api";
import { applyTheme, currentTheme } from "./lib/theme";

// Before anything renders. index.html's boot script has normally already done
// this but it is an inline script, and an inline script is exactly the thing
// a CSP, an extension or a headless renderer can decline to run. Applying it
// again here costs one class toggle and removes the whole failure mode.
applyTheme(currentTheme());

const root = ReactDOM.createRoot(document.getElementById("root") as HTMLElement);

const render = () =>
  root.render(
    <React.StrictMode>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </React.StrictMode>,
  );

// Where the server lives has to be known before the first render: `mediaUrl` is
// called during render and a <video src> cannot await. In a browser this
// resolves to "" on the spot (same-origin); only in the desktop shell is there
// an actual round trip, to ask the shell which port the sidecar got.
void initApiBase()
  .catch(() => undefined)
  .then(render);
