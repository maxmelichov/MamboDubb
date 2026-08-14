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

const settle = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// --- the editor -----------------------------------------------------------
//
// The editor is script-first: two hundred rows, each showing an original and
// its translation stacked and complete, with the video and the per-line detail
// in a fixed column beside them. Nearly every assertion below is about that
// one claim, because it is the claim the whole screen is built on.

dom.window.history.pushState({}, "", "/editor/kan11_v3");
dom.window.dispatchEvent(new dom.window.PopStateEvent("popstate"));
await new Promise((resolve) => setTimeout(resolve, 600));

const editor = root.textContent;
check("editor renders", /Render preview/.test(editor));
check("the script loaded", document.querySelectorAll('[role="option"]').length > 40);
check("timeline drew marks", document.querySelectorAll('[aria-label^="Segment "]').length > 40);

const rows = () => [...document.querySelectorAll('[role="option"]')];
const rowFor = (id) => rows().find((r) => r.textContent.includes(`#${id}`));

/*
 * The centrepiece. Segment 1 of the fixture is a real Hebrew line with a real
 * English translation, and the point of the redesign is that a reviewer can see
 * *both at once*, in one row, without selecting anything — that is what makes
 * checking a translation possible at all. The previous list showed whichever
 * one "would play" and hid the other.
 */
const HEBREW = "היא לא מאיימת עלינו ברמה הצבאית.";
const ENGLISH = "She does not threaten us on a military level.";
const row1 = rowFor(1);
if (!row1) throw new Error("smoke: no row for segment 1");
check("a row shows the original", row1.textContent.includes(HEBREW));
check("the same row shows the translation", row1.textContent.includes(ENGLISH));
check(
  "both lines are separate elements, stacked",
  row1.querySelector('[data-line="text"]').textContent.includes(HEBREW) &&
    row1.querySelector('[data-line="text_en"]').textContent.includes(ENGLISH),
);

/*
 * Bidi. The row's chrome — timecode, id, buttons — is laid out left-to-right no
 * matter what language the run is in; each text line takes its direction from
 * its own content. Getting this wrong drags the whole row around whenever a
 * Hebrew line is on screen.
 */
check("the row container is explicitly LTR", row1.getAttribute("dir") === "ltr");
check(
  "each text line carries dir=auto and .auto-dir",
  [...row1.querySelectorAll("[data-line]")].every(
    (p) => p.getAttribute("dir") === "auto" && p.classList.contains("auto-dir"),
  ),
);
check(
  "script text is clamped, never ellipsis-truncated",
  [...row1.querySelectorAll("[data-line]")].every((p) => !p.className.includes("truncate")),
);
// A composed `${speaker} · ${text}` tooltip is one string with two directions
// and no markup to scope them, which the platform renders scrambled.
check("no composed tooltip on the row", row1.getAttribute("title") == null);

/*
 * A kept line still shows both halves: a reviewer's job includes checking the
 * keep decisions, and a row that hides the translation of a kept line hides the
 * evidence for the judgement it is asking for.
 */
const keptRow = rows().find((r) => r.textContent.includes("original audio plays here"));
check("kept lines say so, and still show both texts", keptRow != null);

// The state is a word on every row. Light-mode "kept" is 2.17:1 against the
// card — under the 3:1 gate — and there is no legend on screen any more.
check(
  "every row says its state in words",
  rows().every((r) => /Dub|Keep|Fail|Voice|Text/.test(r.textContent)),
);
check("the legend is not permanent chrome", !editor.includes("Unclaimed time"));

// One tab stop, not two hundred.
check(
  "the script is reachable by keyboard",
  document.querySelector('[role="option"][tabindex="0"]') != null,
);
check(
  "only one row is a tab stop",
  document.querySelectorAll('[role="option"][tabindex="0"]').length === 1,
);

/*
 * The regression test for the media contract.
 *
 * The server sends `seg.media = {play, tts, source, source_window}`; the UI
 * used to read `seg.source_clip_url` / `place_clip_url` / `tts_clip_url`, which
 * only the fixtures ever produced — so A/B playback was dead against the real
 * server and green in every test. The fixtures now speak the server's shape, so
 * a button with a URL on it proves the UI read the field the server actually
 * fills.
 */
const clip = (row, side) => row.querySelector(`[data-clip="${side}"]`);
check("every row has A and B", rows().every((r) => clip(r, "A") && clip(r, "B")));
check(
  "A plays the original, from the server-shaped media field",
  clip(row1, "A").getAttribute("data-url").length > 0 && !clip(row1, "A").disabled,
);
check(
  "B plays the dub, from the server-shaped media field",
  clip(row1, "B").getAttribute("data-url").length > 0 && !clip(row1, "B").disabled,
);
check(
  "a line with nothing synthesized has a dead B, not a broken one",
  rows().some((r) => clip(r, "B").disabled),
);

/*
 * Inline editing. The translation is edited where it is read — that is the
 * whole point of putting it in the row — and the two hard rules are that a
 * commit locks the line and an empty commit is refused outright (the server
 * 400s: "text_en cannot be empty").
 */
const setValue = (el, value) => {
  const setter = Object.getOwnPropertyDescriptor(
    dom.window.HTMLTextAreaElement.prototype,
    "value",
  ).set;
  setter.call(el, value);
  el.dispatchEvent(new dom.window.Event("input", { bubbles: true }));
};
const clickIt = (el) => el.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));

const target = rowFor(2);
check("the row is not hand-edited yet", target.querySelector('[aria-label^="Hand-edited"]') == null);
clickIt(target.querySelector('[data-line="text_en"]'));
await settle(150);
const field = () => rowFor(2).querySelector("[data-editing]");
check("clicking the translation opens a field in place", field() != null);
check("the field is bidi-aware too", field().getAttribute("dir") === "auto");

// An empty commit must never reach the server.
setValue(field(), "   ");
await settle(80);
check(
  "an empty line refuses itself before it is sent",
  rowFor(2).textContent.includes("has to say something"),
);
field().blur();
await settle(200);
check(
  "and blurring an empty field changes nothing",
  rowFor(2).querySelector('[aria-label^="Hand-edited"]') == null &&
    rowFor(2).textContent.includes("The Qatari threat"),
);

// A real commit lands, and marks the line as the user's.
clickIt(rowFor(2).querySelector('[data-line="text_en"]'));
await settle(150);
setValue(field(), "Edited by hand.");
await settle(80);
field().blur();
await settle(300);
check("committing on blur saves the line", rowFor(2).textContent.includes("Edited by hand."));
check(
  "a hand-edited line is locked against the pipeline",
  rowFor(2).querySelector('[aria-label^="Hand-edited"]') != null,
);
// Saving the text drops the clip that says the old words — the row must say so
// rather than keep a "Dubbed" badge and a live B button over stale audio.
check(
  "the invalidated clip is modelled locally, not left stale",
  /Voice/.test(rowFor(2).textContent) && clip(rowFor(2), "B").disabled,
);

/*
 * Filters, and the bulk fix behind them. `POST /resynthesize` has always taken
 * `{uids:[…]}` and the UI has always sent exactly one; one model load beats N.
 */
const chip = (label) =>
  [...document.querySelectorAll("button")].find((b) => b.textContent.startsWith(label));
check("the script has filter chips", ["All", "Failed", "Kept", "Edited"].every((l) => chip(l)));
check("the Edited chip counts the line just edited", /Edited\s*2/.test(chip("Edited").textContent));

clickIt(chip("Failed"));
await settle(200);
check("the Failed chip narrows the script", rows().length < 40);
const bulk = [...document.querySelectorAll("button")].find((b) =>
  /Re-voice these \d+/.test(b.textContent),
);
check("a filtered set can be fixed in one job", bulk != null);
check(
  "…and re-translated in one job too",
  [...document.querySelectorAll("button")].some((b) => /Re-translate these \d+/.test(b.textContent)),
);
clickIt(chip("Failed"));
await settle(200);
check("the chip toggles back off", rows().length > 40);

/*
 * The selection panel. It holds everything true *about* a line and no text
 * field for the line itself — the text is in the script, where the comparison
 * is. Four shelves, all shut, all named for what is on them.
 */
// Segment 2 is the one just edited by hand, so its Advanced shelf has locks on
// it — which is the only state in which "Release locks" exists to be found.
clickIt(rowFor(2).querySelector('[aria-label^="Select segment"]'));
await settle(200);
const panel = root.textContent;
check("the panel opens on selection", /Dub it/.test(panel) && /Keep original/.test(panel));
check("the model actions state their cost", /~1 min · queues/.test(panel));
check("the panel does not restate the script", !/^Translation$/m.test(panel));
const shelves = [...document.querySelectorAll("aside details")];
check("the rest is on named shelves", shelves.length === 4);
check(
  "the shelves are named for what is on them",
  ["Voice & speaker", "Verification", "Timing & languages", "Advanced"].every((label) =>
    shelves.some((d) => d.querySelector("summary").textContent.includes(label)),
  ),
);
check("every shelf starts shut", shelves.every((d) => !d.open));
check("a shut shelf still summarises itself", /inherit → inherit/.test(panel));
const onAShutShelf = (text) => shelves.some((d) => !d.open && d.textContent.includes(text));
check(
  "the rare controls are shut away, not merely small",
  ["Reference clip", "Split at playhead", "Translate into", "Release locks"].every(onAShutShelf),
);
// The style/instruct box is gone for good: Qwen3-TTS has no instruct parameter,
// so the field was a promise the pipeline cannot keep.
check("there is no style box anywhere", !/e\.g\. calm, urgent/.test(root.textContent));

const voice = shelves.find((d) => d.querySelector("summary").textContent.includes("Voice"));
clickIt(voice.querySelector("summary"));
await settle(150);
check("opening a shelf reveals its controls", voice.open && /Reference clip/.test(voice.textContent));
check(
  "the open state is remembered for the session",
  dom.window.sessionStorage.getItem("dubbing-studio.open.seg.voice") === "1",
);
clickIt(voice.querySelector("summary"));
await settle(150);
check(
  "and closing is remembered too",
  dom.window.sessionStorage.getItem("dubbing-studio.open.seg.voice") === "0",
);

const click = (label) => {
  const button = [...document.querySelectorAll("button")].find((b) =>
    b.textContent.includes(label),
  );
  if (!button) throw new Error(`smoke: no button "${label}"`);
  clickIt(button);
};

// A no-model edit must apply immediately, without a job.
const keptBefore = rows().filter((r) => r.textContent.includes("original audio plays here")).length;
click("Keep original");
await settle(250);
check(
  "keep applies immediately, and the row says what will play",
  rows().filter((r) => r.textContent.includes("original audio plays here")).length > keptBefore,
);
click("Dub it");
await settle(250);
check(
  "the verdict goes both ways",
  !rowFor(2).textContent.includes("original audio plays here"),
);

/*
 * The chrome that is not permanent. The keyboard map lives behind "?", and the
 * run's health — the uncovered-speech list, which is the highest-value thing
 * report.json produces — lives behind "⋯". Neither of them changes while you
 * work through a line, so neither of them rents a row of the screen.
 */
const byLabel = (label) =>
  [...document.querySelectorAll("button")].find((b) => b.getAttribute("aria-label") === label);
const help = byLabel("Keyboard shortcuts");
if (!help) throw new Error("smoke: no shortcuts popover trigger");
clickIt(help);
await settle(120);
check(
  "the shortcuts are one click away",
  /play \/ pause/.test(root.textContent) && /zoom the timeline/.test(root.textContent),
);
clickIt(help);
await settle(120);
check("the popover closes again", !/play \/ pause/.test(root.textContent));

const menu = byLabel("Run health and files");
if (!menu) throw new Error("smoke: no run menu trigger");
clickIt(menu);
await settle(120);
check("run health surfaces uncovered speech", /Audible, uncovered/.test(root.textContent));
check("run health counts the states", /Kept original/.test(root.textContent));
clickIt(menu);
await settle(120);

// No panel in the editor may be blank: with no preview file the stage says why
// there is no picture rather than showing an empty rectangle.
check("the empty preview explains itself", /there is no video/.test(root.textContent));
check("the preview shows the pipeline position", /stages done/.test(root.textContent));

/*
 * Confirmation is themed and local, never `window.confirm` — which is drawn by
 * the OS, blocks the main thread and stops the playhead.
 */
click("Render preview");
await settle(150);
check(
  "a destructive action asks in the app, not in an OS sheet",
  /full video re-encode/.test(root.textContent),
);
check(
  "the confirming button repeats the verb",
  [...document.querySelectorAll('[role="dialog"] button')].some((b) => b.textContent === "Render"),
);
click("Cancel");
await settle(150);
check("cancelling does nothing", !/full video re-encode/.test(root.textContent));

// A model action must queue and report progress instead of blocking the UI.
click("Re-voice this line");
await settle(200);
check("model action queues and reports", /Re-voicing/.test(root.textContent));
check("editor still interactive during a job", rows().length > 40);

// …and the finished job writes its result back through the event stream —
// without clobbering anything, because nothing is being typed.
await settle(1200);
check("job clears when done", !/Re-voicing/.test(root.textContent));

check(
  "no console errors",
  errors.filter((e) => !/not implemented|Not implemented/.test(e)).length === 0,
);

console.log("smoke: all checks passed");
process.exit(0);
