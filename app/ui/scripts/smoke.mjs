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
 * The theme system. Two themes, both shipped, the user picks — and the promise
 * is about the *built* artifact, not the source, so the first group reads the
 * bundle. The one thing that must never come back is the OS preference: a
 * `prefers-color-scheme` block is how a user's choice silently stops being
 * theirs.
 */
const cssFile = readdirSync(assetDir).find((f) => f.endsWith(".css"));
const css = readFileSync(new URL(cssFile, assetDir), "utf8");
const html = readFileSync(new URL("../dist/index.html", import.meta.url), "utf8");
check("the OS preference is never consulted", !/prefers-color-scheme/.test(css + html));
check("both colour-schemes ship", /color-scheme:\s*light/.test(css) && /color-scheme:\s*dark/.test(css));
check("the dark theme is a class, not a media query", /\.theme-dark/.test(css));
check("the pre-paint canvas covers both themes", /theme-dark[^}]*#0e0e0d/.test(html) && /#f7f6f2/.test(html));
check("native number spinners are suppressed", /-webkit-inner-spin-button/.test(css));

const root = document.getElementById("root");
check("import screen renders", /New dub/.test(root.textContent));

/*
 * The toggle itself, driven through the DOM. jsdom does not run index.html's
 * inline boot script, so what this proves is the second half of the contract:
 * the bundle applies the stored choice on mount, defaulting to dark.
 */
const themeButton = (which) =>
  [...document.querySelectorAll("button")].find(
    (b) => b.getAttribute("aria-label") === `Switch to ${which} theme`,
  );
check("the header carries a theme toggle", themeButton("light") != null && themeButton("dark") != null);
check("dark is the default with nothing stored", document.documentElement.classList.contains("theme-dark"));
check("the toggle says which theme is on", themeButton("dark").getAttribute("aria-pressed") === "true");

/*
 * "Each theme paints its own ground" is the one claim worth checking against a
 * real cascade rather than a regex, because the failure it guards — a theme
 * that flips the tokens but leaves the canvas on the other theme's colour — is
 * invisible to a source grep. jsdom does not fetch the linked stylesheet, so
 * the two `html` rules are lifted out of the shipped CSS and injected.
 */
const canvasRules = css.match(/html\{[^}]*\}html\.theme-dark\{[^}]*\}/)?.[0];
if (!canvasRules) throw new Error("smoke: no html/html.theme-dark rules in the shipped CSS");
const injected = document.createElement("style");
injected.textContent = canvasRules;
document.head.append(injected);

const canvas = () => dom.window.getComputedStyle(document.documentElement).backgroundColor;
check("dark paints its own canvas", canvas() === "rgb(14, 14, 13)");

themeButton("light").dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
await new Promise((resolve) => setTimeout(resolve, 120));
check("toggling drops the dark class", !document.documentElement.classList.contains("theme-dark"));
check("the choice is persisted", dom.window.localStorage.getItem("dubbing-studio.theme") === "light");
check("light paints its own canvas", canvas() === "rgb(247, 246, 242)");
check(
  "the toggle follows the choice",
  themeButton("light").getAttribute("aria-pressed") === "true" &&
    themeButton("dark").getAttribute("aria-pressed") === "false",
);
check(
  "theme-color follows the theme",
  document.querySelector('meta[name="theme-color"]').getAttribute("content") === "#f7f6f2",
);

themeButton("dark").dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
await new Promise((resolve) => setTimeout(resolve, 120));
check("toggling back restores dark", document.documentElement.classList.contains("theme-dark"));
check("dark is persisted too", dom.window.localStorage.getItem("dubbing-studio.theme") === "dark");

// An existing run's row has to answer "where did this get to" without being
// opened — the whole point of listing it.
await new Promise((resolve) => setTimeout(resolve, 200));
const runs = root.textContent;
check("runs say how far they got", /Complete/.test(runs) && /Running translate/.test(runs));
check("a failed run names the stage", /Failed at transcript/.test(runs));
check("runs say when they last moved", /just now|hours? ago|days? ago/.test(runs));
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
// The detail lines mark the parts meant to be typed with backticks; rendering
// them literally puts punctuation in the middle of a command to be copied.
check("commands render as code, not backticks", !setup.includes("`"));
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

// No panel in the editor may be blank. With nothing selected the rail is the
// run's report, and the preview stage says why there is no picture rather than
// showing an empty rectangle.
check("the idle rail summarises the run", /This run/.test(editor) && /Coverage/.test(editor));
check("uncovered speech is surfaced", /Audible, uncovered/.test(editor));
check("the empty preview explains itself", /there is no video/.test(editor));
check("the preview shows the pipeline position", /stages done/.test(editor));

// The table is one tab stop with a roving tabindex, not fifty-eight, and it is
// a grid so `aria-selected` on a row means something.
check("the segment table is reachable by keyboard", document.querySelector('tbody tr[tabindex="0"]') != null);
check("only one row is a tab stop", document.querySelectorAll('tbody tr[tabindex="0"]').length === 1);

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
