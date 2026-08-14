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
check("the pre-paint canvas covers both themes", /theme-dark[^}]*#110e16/.test(html) && /#f7f6f2/.test(html));
check("native number spinners are suppressed", /-webkit-inner-spin-button/.test(css));

/*
 * The one claim about `request()` this mode can make.
 *
 * `VITE_USE_FIXTURES=1` is inlined as a literal, so every `if (USE_FIXTURES)
 * return fixtures.x()` makes the HTTP path dead code and the bundler removes
 * `request` entirely — it is not in the artifact to assert against, and there
 * is no fetch in this mode to drive it with. The guard is still worth pinning,
 * because what it replaced was a sentence the user had to read: a 200 whose
 * body is not JSON (an SPA fallback, a captive portal, the wrong port) was cast
 * straight to the response type, and `getProject` surfaced "TypeError: Cannot
 * read properties of null (reading 'name')" in the editor's error bar. So this
 * one reads the source, and says so.
 */
const apiSource = readFileSync(new URL("../src/lib/api.ts", import.meta.url), "utf8");
check(
  "api.request names a 200 that is not JSON instead of casting it",
  /if \(text && body === null\)/.test(apiSource) && /is not JSON/.test(apiSource),
);

/*
 * The state palette, checked structurally rather than by hex.
 *
 * Pinning "kept is #306357" in a test is pinning a *decision* that is allowed
 * to change; what is not allowed to change is the shape of the decision. Three
 * things hold:
 *
 * 1. Every state hue is restated in the dark theme. A hue declared only in
 *    `@theme` inherits into dark unnoticed, which is how a light-mode green
 *    ends up at 2:1 on a near-black plane.
 * 2. The resolved state and the waiting state are different hues. They were
 *    the same one — everything a run kept wore the warning colour, so an
 *    already-English video rendered as a wall of amber with nothing wrong in
 *    it — and "kept is not the warn colour" is the whole fix, in one line.
 * 3. The semantic tones alias the state hues instead of adding more. A second
 *    green for `good` or a second amber for `warn` is how five hues quietly
 *    become nine, none of them measured against the other four.
 */
const hue = (name, block = css) =>
  block.match(new RegExp(`--color-${name}:\\s*([^;}]+)`))?.[1]?.trim() ?? null;
const darkBlock = css.match(/:root\.theme-dark\{[^}]*\}/)?.[0] ?? "";
const STATES = ["dubbed", "kept", "failed", "pending", "unclaimed"];
check("every state hue ships in both themes", STATES.every((s) => hue(s) && hue(s, darkBlock)));
check(
  "the two themes pick different values for every state hue",
  STATES.every((s) => hue(s) !== hue(s, darkBlock)),
);
check("resolved and waiting are not the same hue", hue("kept") !== hue("pending"));
check("…in the dark theme too", hue("kept", darkBlock) !== hue("pending", darkBlock));
check(
  "the semantic tones alias the state hues rather than adding more",
  hue("good") === "var(--color-kept)" &&
    hue("warning") === "var(--color-pending)" &&
    hue("critical") === "var(--color-failed)",
);
check("the kept hue is not the warn hue", hue("kept") !== hue("warning"));
/* The timeline's mark wash is a theme token, because how much hue a fill can
   take is a property of the ground it sits on — 18% of a dark hue on a white
   lane and 26% of a light one over a waveform on near-black. */
check("the mark wash is a theme token, per theme", /--mark-wash:/.test(css) && /--mark-wash:/.test(darkBlock));

/*
 * The interaction accent.
 *
 * "The dark mode is too dim — the colours are dead" was a theme in which the
 * only saturated pixels on screen were the state hues, and the thing to press
 * was a near-white rectangle. The fix is one token, and the three things that
 * hold about it are structural rather than a hex:
 *
 * 1. It ships in both themes. A `--color-accent` declared only in `@theme`
 *    would make dark's primary button near-white again the moment somebody
 *    edits the light value, silently.
 * 2. In light it *is* ink. That is not a placeholder — it is the promise that
 *    naming the accent changed nothing about the theme nobody complained
 *    about — and it is why every `bg-accent` below is safe to have replaced a
 *    `bg-primary`.
 * 3. In dark it is not ink. A dark accent that resolves back to the near-white
 *    is the bug this whole pass exists to fix, and it would pass every other
 *    check in this file.
 */
check("the accent ships in both themes", hue("accent") != null && hue("accent", darkBlock) != null);
check("light's accent is ink, so light is unchanged", hue("accent") === "var(--color-primary)");
check("…and its label is ink's label", hue("on-accent") === "var(--color-on-primary)");
check(
  "dark picks a colour for the accent rather than inheriting the near-white ink",
  /^#[0-9a-f]{6}$/i.test(hue("accent", darkBlock) ?? "") &&
    hue("accent", darkBlock) !== hue("primary", darkBlock),
);
check("…with a label of its own to sit on it", hue("on-accent", darkBlock) != null);
/* The accent is a fill, a ring and a rule, and it is measured against the 3:1
   non-text gate only. There is no assertion here that it is never set as text,
   because writing one puts the class name in a file Tailwind scans and mints
   the utility it was checking for — the check would fail on itself. */

/*
 * The desktop shell paints its window before the webview exists, so its
 * `backgroundColor` has to be the dark plane exactly — any drift is a flash of
 * the wrong near-black on every launch. Three copies of that value ship (the
 * CSS token, the pre-paint canvas in index.html, the Tauri config) and this is
 * the one seam a class cannot cover, so it is checked rather than trusted.
 */
const darkPlane = hue("plane", darkBlock);
const tauri = JSON.parse(
  readFileSync(new URL("../../desktop/src-tauri/tauri.conf.json", import.meta.url), "utf8"),
);
check(
  "the shell's window colour is the dark plane",
  tauri.app.windows[0].backgroundColor.toLowerCase() === darkPlane,
);
check("…and so is the canvas painted before the bundle parses", html.includes(darkPlane));

/*
 * And the palette stays in one file. A literal `#3b7f5c` in a component is a
 * colour that no theme can restate and no measurement covers — it is how the
 * dark theme grows a light-mode hue nobody notices until a screenshot. The one
 * pair allowed outside App.css is the canvas colour in lib/theme.ts, which has
 * to be a literal because it is written into a `<meta name="theme-color">`.
 */
const srcDir = new URL("../src/", import.meta.url);
const sources = [];
const walk = (dir) => {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const at = new URL(entry.name + (entry.isDirectory() ? "/" : ""), dir);
    if (entry.isDirectory()) walk(at);
    else if (/\.tsx?$/.test(entry.name)) sources.push([entry.name, readFileSync(at, "utf8")]);
  }
};
walk(srcDir);
const literalColour = sources.filter(
  ([name, text]) => name !== "theme.ts" && /#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?\b/.test(text),
);
check(
  `no component picks its own colour (${sources.length} files scanned)`,
  literalColour.length === 0,
);

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
check("dark paints its own canvas", canvas() === "rgb(17, 14, 22)");

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

/*
 * Nothing is selected when a run is first opened, which is exactly when "what
 * happened here" is the only question there is. The rail answers it instead of
 * printing "No line selected" in the middle of two fifths of the screen.
 */
check("with nothing selected the rail summarises the run", /This run/.test(editor));
check("…counting the lines by state", /Dubbed/.test(editor) && /Kept original/.test(editor));
check("…saying why the kept ones were kept", /Kept because/.test(editor));
check("…and where speech is uncovered", /Audible, uncovered/.test(editor));
check("the empty rail is not an apology", !/No line selected/.test(editor));
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
 * The other half of that claim: two lines are for a *comparison*, and there is
 * none when the two halves are the same string. A run over a video that already
 * speaks the target language passes the text through untouched, so `text_en ===
 * text` on most of it — and the row used to print the identical sentence twice,
 * at two weights, on every one of them. Nothing in this list may ever do that.
 */
check(
  "no row prints the same sentence twice",
  rows().every((r) => {
    const lines = [...r.querySelectorAll("[data-line]")].map((p) => p.textContent.trim());
    return !(lines.length === 2 && lines[0] === lines[1]);
  }),
);
const echoRow = rows().find((r) => r.getAttribute("data-lines") === "1");
check("a passthrough line is drawn as one line", echoRow != null);
check(
  "…and it is the original that survives, not a copy of it",
  echoRow.querySelectorAll("[data-line]").length === 1 &&
    echoRow.querySelector('[data-line="text"]') != null,
);

/*
 * A kept line says what will play — once, as a clause on the meta line. It used
 * to be a paragraph of its own under the two texts, which on an all-kept run is
 * seventy-three identical sentences down the page.
 */
const keptRow = rows().find((r) => r.textContent.includes("original audio plays here"));
check("kept lines say what plays", keptRow != null);
const keptMeta = [...keptRow.querySelectorAll("button")].find((b) => /Keep/.test(b.textContent));
check(
  "…as a clause on the row's meta line, not a fourth line of its own",
  keptMeta != null && keptMeta.textContent.includes("original audio plays here"),
);

/*
 * The encoding is a drawn shape now. Two of the Unicode glyphs it used to set —
 * U+25A3 for "kept" above all — are outside the UI font, so the browser fell
 * back per character and drew a lumpy square on every kept row.
 */
check("the state shape is drawn, not set in a font", !editor.includes("▣") && !editor.includes("◆"));
check("every row carries its state shape", rows().every((r) => r.querySelector("svg") != null));

// The state is a word on every row. Light-mode "kept" is 2.17:1 against the
// card — under the 3:1 gate — and there is no legend on screen any more.
check(
  "every row says its state in words",
  rows().every((r) => /Dub|Keep|Fail|Voice|Text/.test(r.textContent)),
);
check("the legend is not permanent chrome", !editor.includes("Unclaimed time"));

/*
 * The list scrolls to rows — the playhead's, the selection's, ↑/↓'s — and every
 * one of those landed the row flush under the filter bar, reading as text
 * sliced off by it. `scroll-padding` is the platform's inset for exactly that,
 * and the bar above has to be opaque or the row passing under shows through.
 */
const scroller = document.querySelector("[data-script-scroll]");
const scriptHeader = document.querySelector("[data-script-header]");
check("the list keeps its rows clear of the filter bar", /scroll-pt-3/.test(scroller.className));
check(
  "the filter bar is opaque, and hairlined off the list",
  / bg-sunken |^bg-sunken /.test(` ${scriptHeader.className} `) &&
    !/bg-sunken\//.test(scriptHeader.className) &&
    /border-b/.test(scriptHeader.className),
);

/*
 * The timeline's track headers live in a gutter beside the lanes, not floating
 * over the first few seconds of them — which is where every run's first mark
 * is, and where a reviewer starts reading.
 */
const laneLabels = [...document.querySelectorAll("[data-lane-label]")];
check(
  "both lanes are labelled",
  laneLabels.length === 2 &&
    /Source/i.test(laneLabels[0].textContent) &&
    /Output/i.test(laneLabels[1].textContent),
);
check(
  "the lane labels sit in a gutter, never over the marks",
  laneLabels.every(
    (label) =>
      label.closest("[data-mark]") == null &&
      label.parentElement.querySelector("[data-mark]") == null,
  ),
);
check(
  "the playhead is drawn above both lanes",
  document.querySelector("[data-playhead]") != null,
);

/*
 * The lanes draw the run's actual audio behind the marks — "make the audio look
 * like audio". Two claims worth pinning: the picture is there for both lanes,
 * and it is scaled by the *audio's* duration rather than the timeline's, which
 * is the difference between a waveform that lines up with the marks and one
 * that slides against them.
 */
const waves = [...document.querySelectorAll("[data-waveform]")];
check(
  "both lanes draw their audio",
  waves.length === 2 &&
    waves.map((w) => w.getAttribute("data-waveform")).join() === "source,dub",
);
check(
  "the waveform is a path with real values in it",
  waves.every((w) => (w.querySelector("path")?.getAttribute("d") ?? "").split("L").length > 100),
);
check(
  "the picture is scaled by the audio's own duration",
  // 320s of fixture audio at the default 4px/s.
  waves.every((w) => /width:\s*1280px/.test(w.getAttribute("style") ?? "")),
);
check(
  "the buckets stay inside the server's clamp",
  waves.every((w) => {
    const n = Number(w.getAttribute("viewBox").split(" ")[2]);
    return n >= 16 && n <= 4000;
  }),
);
check(
  "the marks stay on top of it, and stay clickable",
  document.querySelectorAll("[data-mark]").length > 40 &&
    waves.every((w) => w.getAttribute("aria-hidden") === "true"),
);

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
 * …and pressing one has to hand the element something a browser can load.
 *
 * `seg.media.*` is the *server's* name for the clip, which in fixture mode is
 * `fixture:tone?hz=…`. Assigning that to `<audio src>` is ERR_UNKNOWN_URL_SCHEME
 * and the button snaps straight back out of its pressed state — which is what
 * every A/B press did, because `api.audioUrl` (the seam that resolves it) had
 * no callers at all. The claim is narrow and it is the one that broke: the URL
 * that reaches the element went through the seam, and it is not the raw one.
 */
const played = [];
dom.window.HTMLMediaElement.prototype.play = function play() {
  played.push(this.src);
  return Promise.resolve();
};
const press = (el) => el.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
press(clip(rowFor(1), "A"));
await settle(120);
check("pressing A puts a clip on the shared element", played.length === 1);
check(
  "…resolved through the audio seam, not the server's own name for it",
  played.length === 1 && !played[0].startsWith("fixture:") && played[0].length > 0,
);
check(
  "…and the button says which side is sounding",
  clip(rowFor(1), "A").getAttribute("aria-pressed") === "true",
);
press(clip(rowFor(1), "B"));
await settle(120);
check(
  "pressing B stops A — one element, one clip",
  played.length === 2 &&
    clip(rowFor(1), "A").getAttribute("aria-pressed") === "false" &&
    clip(rowFor(1), "B").getAttribute("aria-pressed") === "true",
);
press(clip(rowFor(1), "B"));
await settle(120);
check("pressing the sounding side again stops it", clip(rowFor(1), "B").getAttribute("aria-pressed") === "false");

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
 * Closing a field you only opened to read must cost nothing.
 *
 * `edit.set_text` invalidates the clip and the placement and stamps a lock, so
 * a PATCH that carries the text back unchanged is a re-voice queued against a
 * line nobody edited — which is exactly what "I don't want the retranscription
 * to happen without any change" is about. Fixture mode counts its own calls
 * because a no-op save is invisible from the DOM: it looks the same as no save.
 */
// The bundle is imported into *this* module's realm, so its `globalThis` is
// Node's, not jsdom's `window` — the counter lands on whichever one the fixture
// module saw.
const calls = () => globalThis.__DUBBING_FIXTURE_CALLS__ ?? dom.window.__DUBBING_FIXTURE_CALLS__;
check("fixture mode counts its own round trips", calls() != null);
const patchesBefore = calls().patch;
clickIt(rowFor(3).querySelector('[data-line="text_en"]'));
await settle(150);
check("a field opens on the line it was clicked", rowFor(3).querySelector("[data-editing]") != null);
rowFor(3).querySelector("[data-editing]").blur();
await settle(250);
check("closing an unchanged field saves nothing", calls().patch === patchesBefore);
check(
  "…and leaves the line the pipeline's",
  rowFor(3).querySelector('[aria-label^="Hand-edited"]') == null,
);

// Whitespace is not an edit either — the editor commits a trimmed draft, so
// the comparison it is refused by has to be trimmed too.
clickIt(rowFor(3).querySelector('[data-line="text_en"]'));
await settle(150);
const padded = rowFor(3).querySelector("[data-editing]");
setValue(padded, `   ${padded.value}  `);
await settle(80);
padded.blur();
await settle(250);
check("re-spacing a line is not an edit", calls().patch === patchesBefore);

/*
 * Filters, and the bulk fix behind them. `POST /resynthesize` has always taken
 * `{uids:[…]}` and the UI has always sent exactly one; one model load beats N.
 */
const chip = (label) =>
  [...document.querySelectorAll("button")].find((b) => b.textContent.startsWith(label));
check(
  "the script has filter chips",
  ["All", "Failed", "Unfinished", "Kept", "Edited"].every((l) => chip(l)),
);
// Two of the three locked lines are the fixture's own: the one the snapshot
// carries, and the stranded line, whose `locked:{keep:true}` is what a verdict
// flip stamps. The third is the one edited by hand a few checks above.
check("the Edited chip counts the line just edited", /Edited\s*3/.test(chip("Edited").textContent));

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
 * Limbo, and the way out of it.
 *
 * `PATCH {keep:false}` invalidates the translate stage, so a "Dub it" that
 * queued nothing left the line with no translation, no clip and no job coming
 * — invisible in a list of two hundred rows and unreachable by every other
 * chip. The fixture carries one of each unfinished shape (no translation, no
 * voice) and the chip has to find both and offer the one click that fixes them.
 */
clickIt(chip("Unfinished"));
await settle(200);
const stuck = rows();
// Two from the fixture — one stranded by a verdict flip, one never voiced —
// plus the line whose translation was rewritten by hand a few checks above,
// which dropped the clip that said the old words.
check("the Unfinished chip finds the stranded lines", stuck.length === 3);
const stranded = stuck.find((r) => r.textContent.includes("not translated yet"));
check("…the one a Dub it left with nothing to say", stranded != null);
check(
  "…still carrying the keep lock that flip stamped on it",
  stranded.querySelector('[aria-label^="Hand-edited"]') != null,
);
check(
  "…and the ones that only need a voice",
  stuck.filter((r) => r !== stranded).every((r) => /Voice/.test(r.textContent)),
);
check(
  "…fixable in one click, not two hundred",
  [...document.querySelectorAll("button")].some((b) =>
    /Translate & voice these 3/.test(b.textContent),
  ),
);
clickIt(chip("Unfinished"));
await settle(200);
check("the Unfinished chip toggles back off too", rows().length > 40);

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

/*
 * The verdict, and what it costs.
 *
 * "Keep original" is the direction that needs no work — the original audio is
 * already on disk — so it applies instantly and queues nothing. "Dub it" is the
 * direction that needs all of it: `edit.set_keep` invalidates the translate
 * stage, so the line loses its subtitle and its clip on the way through, and a
 * flip that sent the PATCH and stopped there left the segment with nothing to
 * say and nothing coming to fix it. That was the bug. Both claims are counted
 * rather than watched: a job that was never enqueued looks exactly like a job
 * that has not started.
 */
const keptBefore = rows().filter((r) => r.textContent.includes("original audio plays here")).length;
const sinceKeep = calls().log.length;
click("Keep original");
await settle(250);
check(
  "keep applies immediately, and the row says what will play",
  rows().filter((r) => r.textContent.includes("original audio plays here")).length > keptBefore,
);
check("keeping the original queues nothing", calls().log.slice(sinceKeep).join() === "patch");
// Segment 2's translation was typed by hand a few checks above, and a kept line
// with a hand-written translation is the case the row used to render as a dub:
// an English sentence under a Hebrew one, in the place a spoken line goes.
check(
  "a translation on a kept line says it is only a subtitle",
  rowFor(2).textContent.includes("the edit is a subtitle"),
);

// …and back. The hand-written line is locked, so it survives the flip and only
// the voice has to be queued.
click("Dub it");
await settle(350);
check(
  "the verdict goes both ways",
  !rowFor(2).textContent.includes("original audio plays here"),
);
check(
  "a dub whose translation survived queues only the voice",
  calls().log.slice(sinceKeep).join() === "patch,patch,resynthesize",
);

// The general case: a kept line whose translation the pipeline wrote loses it
// on the flip, so the translator has to run before the voice does — and the
// one-job queue is FIFO, so enqueueing in that order *is* running in it.
const sinceDub = calls().log.length;
clickIt(rowFor(0).querySelector('[aria-label^="Select segment"]'));
await settle(200);
check(
  "the panel says what Dub it will cost before it is pressed",
  /queues translate \+ voice for this line/.test(root.textContent),
);
click("Dub it");
await settle(400);
check(
  "flipping a kept line to Dub queues the work the flip invalidated",
  calls().log.slice(sinceDub).join() === "patch,retranslate,resynthesize",
);

// Let the three jobs those flips queued drain, so the next assertions are about
// the job they ask for and not about one of these.
await settle(2000);
check("the queue drains", document.querySelector("[data-job-strip]") == null);

/*
 * A mark on the strip is a question about a line — what does it say, what did
 * it become, what does it sound like — and all three answers are in the script
 * row. So the click has to *bring that row to the reviewer*, centred, whether
 * or not the video is playing. Recorded rather than measured because jsdom has
 * no layout: what matters is that the list was told to centre the right row.
 */
const scrolls = [];
dom.window.Element.prototype.scrollIntoView = function scrollIntoView(options) {
  scrolls.push({ uid: this.getAttribute?.("data-uid") ?? null, options });
};
const someMark = [...document.querySelectorAll("[data-mark]")].find((m) =>
  /^Segment 9,/.test(m.getAttribute("aria-label") ?? ""),
);
if (!someMark) throw new Error("smoke: no mark for segment 9");
clickIt(someMark);
await settle(250);
const centred = scrolls.filter((s) => s.options?.block === "center").pop();
check("clicking a mark selects its line", rowFor(9).getAttribute("aria-selected") === "true");
check(
  "…and scrolls the script to it, centred",
  centred != null && centred.uid === rowFor(9).getAttribute("data-uid"),
);
check(
  "…with its A and B one click away",
  rowFor(9).querySelector('[data-clip="A"]') != null &&
    rowFor(9).querySelector('[data-clip="B"]') != null,
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

/*
 * "I can't easily open the ready file." Both of the things a finished run is
 * for are one click away and named: the video in the header, the subtitles in
 * this menu. Each one goes through `openRunFile`, which reveals the file in
 * Finder inside the shell and opens the served URL in a browser — the old
 * button was desktop-only *and* handed the shell a run-relative path it always
 * refused.
 */
const fileRows = [...document.querySelectorAll("[data-run-file]")];
check(
  "the run's files are listed by name",
  fileRows.map((b) => b.getAttribute("data-run-file")).join() === "preview.mp4,preview_en.srt",
);
check("the subtitles are reachable at last", /Subtitles \(\.srt\)/.test(root.textContent));
check(
  "each file says what the click will do",
  fileRows.every((b) => /Open .* in a new tab/.test(b.getAttribute("title") ?? "")),
);
clickIt(menu);
await settle(120);

const openPreview = [...document.querySelectorAll("header button")].find((b) =>
  /Open preview|Show in Finder/.test(b.textContent),
);
check("the finished video is one click from the header", openPreview != null);
check(
  "…and the header stays at its height",
  document.querySelector("header").className.includes("h-11"),
);

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

/*
 * The strip is one line high, always. With `flex-wrap` a long stage message
 * wrapped to a second row and moved the whole editor down mid-job — the script
 * under the cursor jumping while you read it. And the bar and the number are
 * one value read once: they used to disagree, the bar moving while the
 * percentage sat on "—".
 */
const strip = document.querySelector("[data-job-strip]");
check("the job strip is one line, always", strip != null && / h-8 /.test(` ${strip.className} `));
const bar = strip.querySelector('[role="progressbar"]');
const shown = strip.textContent.match(/(\d+)%/);
check(
  "the bar and the percentage are the same number",
  shown == null ? bar.getAttribute("aria-valuenow") == null : bar.getAttribute("aria-valuenow") === shown[1],
);

// …and the finished job writes its result back through the event stream —
// without clobbering anything, because nothing is being typed.
await settle(1200);
check("job clears when done", !/Re-voicing/.test(root.textContent));

/*
 * A subscriber that joins late still learns what is running.
 *
 * The editor mounts and subscribes *after* the job is created — that is the
 * order "Start dubbing" happens in, project first, editor second — so a stream
 * that only forwards frames from the moment you connect tells a freshly created
 * run nothing at all: no job strip, and a preview stage sitting on "Nothing has
 * run yet" for the whole run. The server opens every stream with a prelude
 * (`app.py::project_events`: a log line, a stage frame per stage, and every job
 * that has not finished) for exactly this reason, and the fixtures have to do
 * the same or the flow is green here and dead in the app.
 */
click("Render preview");
await settle(150);
[...document.querySelectorAll('[role="dialog"] button')]
  .find((b) => b.textContent === "Render")
  .dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
await settle(400);
check("a render queues and the strip says so", /Rendering preview/.test(root.textContent));
await go("/", 300);
check("leaving the run leaves its strip behind", document.querySelector("[data-job-strip]") == null);
await go("/editor/kan11_v3", 700);
check(
  "re-opening a run mid-job finds the job again — the stream replays it",
  /Rendering preview/.test(root.textContent),
);
click("Cancel");
await settle(500);
check("…and it is the same job, cancellable from here", !/Rendering preview/.test(root.textContent));

check(
  "no console errors",
  errors.filter((e) => !/not implemented|Not implemented/.test(e)).length === 0,
);

console.log("smoke: all checks passed");
process.exit(0);
