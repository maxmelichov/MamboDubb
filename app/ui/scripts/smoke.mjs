/**
 * Render the production bundle in jsdom and assert the two screens actually
 * come up. `pnpm build` proves the types; this proves the app does not throw on
 * first paint, which is the failure a type checker cannot see.
 *
 *   pnpm build && node scripts/smoke.mjs
 */

import { readFileSync, readdirSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { JSDOM } from "jsdom";

const assetDir = new URL("../dist/assets/", import.meta.url);
const bundle = readdirSync(assetDir).find((f) => f.endsWith(".js"));
if (!bundle) throw new Error("no bundle in dist/assets — run pnpm build first");

const dom = new JSDOM(readFileSync(new URL("../dist/index.html", import.meta.url), "utf8"), {
  url: "http://localhost/",
  pretendToBeVisual: true,
});

// Hoist every browser global jsdom provides that Node does not already have.
for (const key of Object.getOwnPropertyNames(dom.window)) {
  if (key in globalThis) continue;
  try {
    globalThis[key] = dom.window[key];
  } catch {
    // Some jsdom accessors throw when read off-window; none of them matter here.
  }
}
// Node 25 ships its own experimental globals for some of these; jsdom's must win.
for (const key of ["window", "document", "navigator", "localStorage", "sessionStorage"]) {
  Object.defineProperty(globalThis, key, {
    value: dom.window[key],
    configurable: true,
    writable: true,
  });
}
globalThis.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
dom.window.matchMedia = globalThis.matchMedia;
// jsdom implements no layout, so these exist in every browser but not here.
dom.window.Element.prototype.scrollTo = function scrollTo() {};
dom.window.Element.prototype.scrollIntoView = function scrollIntoView() {};
dom.window.HTMLMediaElement.prototype.play = function play() {
  return Promise.resolve();
};
dom.window.HTMLMediaElement.prototype.pause = function pause() {};
dom.window.URL.createObjectURL = () => "blob:fixture";
globalThis.URL.createObjectURL = dom.window.URL.createObjectURL;

const errors = [];
const realError = console.error;
console.error = (...args) => {
  errors.push(args.map(String).join(" "));
  realError(...args);
};

await import(pathToFileURL(new URL(bundle, assetDir).pathname).href);
await new Promise((resolve) => setTimeout(resolve, 400));

const check = (label, ok) => {
  if (!ok) throw new Error(`smoke: ${label}`);
  console.log(`  ok  ${label}`);
};

/*
 * The theme lock. This app is black-on-white and has no other state, which is
 * a promise about the *built* artifact, not about the source — so these read
 * the bundle. A `prefers-color-scheme` block or a `color-scheme: dark` in the
 * shipped CSS is exactly the regression that puts a dark app back on a user's
 * screen without anyone editing a component.
 */
const cssFile = readdirSync(assetDir).find((f) => f.endsWith(".css"));
const css = readFileSync(new URL(cssFile, assetDir), "utf8");
const html = readFileSync(new URL("../dist/index.html", import.meta.url), "utf8");
check("no dark colour-scheme ships", !/color-scheme:\s*dark/.test(css));
check("the OS preference is never consulted", !/prefers-color-scheme/.test(css));
check("the light colour-scheme is declared", /color-scheme:\s*light/.test(css));
check("the page declares a light theme-color", /theme-color[^>]*#f7f6f2/.test(html));
check("the pre-paint canvas is light", /background:\s*#f7f6f2/.test(html));

const root = document.getElementById("root");
check("import screen renders", /New dub/.test(root.textContent));
check("no theme toggle", ![...document.querySelectorAll("button")].some((b) =>
  /Switch to (light|dark)/.test(b.getAttribute("aria-label") ?? ""),
));
check("nothing stamps a theme on <html>", !document.documentElement.hasAttribute("data-theme"));
check("context field is present", /Context/.test(root.textContent));
check("setup is reachable from the header", /Setup/.test(root.textContent));

const go = async (path, ms) => {
  dom.window.history.pushState({}, "", path);
  dom.window.dispatchEvent(new dom.window.PopStateEvent("popstate"));
  await new Promise((resolve) => setTimeout(resolve, ms));
};

// The setup checklist: every row must say its state in words, not just colour,
// and a failing row must carry the sentence that says what to do about it.
await go("/setup", 300);
const setup = root.textContent;
check("setup screen renders", /Readiness/.test(setup));
check("setup lists every check", document.querySelectorAll("[data-check]").length === 8);
check("passing checks say Ready", /Ready/.test(setup));
check("failing checks say Missing", /Missing/.test(setup));
check("state is never colour alone", [...document.querySelectorAll("[data-check]")].every((row) =>
  /Ready|Missing/.test(row.textContent),
));
check("failing checks explain themselves", /HF_TOKEN/.test(setup) && /htdemucs|Demucs/.test(setup));
check("model sizes are shown", /GB/.test(setup));
check("a mixed result is counted", /2 of 8 need attention/.test(setup));
check("no continue while something is missing", ![...document.querySelectorAll("button")].some((b) =>
  b.textContent.includes("Continue to projects"),
));

const recheck = [...document.querySelectorAll("button")].find((b) =>
  b.textContent.includes("Re-check"),
);
if (!recheck) throw new Error("smoke: no Re-check button");
recheck.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
await new Promise((resolve) => setTimeout(resolve, 300));
check("re-check re-renders the list", document.querySelectorAll("[data-check]").length === 8);

// The gate must not strand the user here: fixture mode never auto-routes, and
// the import screen is one link away.
await go("/", 200);
check("setup does not trap navigation", /New dub/.test(root.textContent));

// Navigate to the editor and let the fixture load.
dom.window.history.pushState({}, "", "/editor/kan11_v3");
dom.window.dispatchEvent(new dom.window.PopStateEvent("popstate"));
await new Promise((resolve) => setTimeout(resolve, 600));

const editor = root.textContent;
check("editor renders", /Render preview/.test(editor));
check("legend names every state", ["Dubbed", "Kept original", "Failed", "Unclaimed time"].every((s) => editor.includes(s)));
check("segments loaded", document.querySelectorAll("tbody tr").length > 40);
check("timeline drew marks", document.querySelectorAll('[aria-label^="Segment "]').length > 40);

// Select a segment and confirm the inspector fills in.
document.querySelectorAll("tbody tr")[1].dispatchEvent(
  new dom.window.MouseEvent("click", { bubbles: true }),
);
await new Promise((resolve) => setTimeout(resolve, 200));
const withInspector = root.textContent;
check("inspector opens on selection", /Verification/.test(withInspector));
check("inspector has A\\/B playback", /Original/.test(withInspector) && /Kept \(same audio\)|Dubbed/.test(withInspector));
const click = (label) => {
  const button = [...document.querySelectorAll("button")].find((b) =>
    b.textContent.includes(label),
  );
  if (!button) throw new Error(`smoke: no button "${label}"`);
  button.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
};
const settle = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// A no-model edit must apply immediately, without a job.
const keptBefore = (root.textContent.match(/Kept original/g) ?? []).length;
click("Keep original");
await settle(200);
check(
  "keep toggle applies immediately",
  (root.textContent.match(/Kept original/g) ?? []).length > keptBefore,
);

// A model action must queue and report progress instead of blocking the UI.
click("Re-translate");
await settle(150);
check("model action queues and reports", /Re-translating/.test(root.textContent));
check("editor still interactive during a job", document.querySelectorAll("tbody tr").length > 40);

// …and the finished job writes its result back through the event stream.
await settle(900);
check("job clears when done", !/Re-translating/.test(root.textContent));
check("re-translation landed", /\[re-translated]/.test(root.textContent));

check("no console errors", errors.filter((e) => !/not implemented|Not implemented/.test(e)).length === 0);

console.log("smoke: all checks passed");
process.exit(0);
