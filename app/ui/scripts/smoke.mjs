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
if (!bundle) throw new Error("no bundle in dist/assets run pnpm build first");

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
// jsdom ships no clipboard, and the setup screen's copy buttons only show their
// tick when the write actually resolves so without this the one interaction
// they exist for is untestable in the only mode that is ever tested.
const clipboard = { text: null, writes: 0 };
Object.defineProperty(dom.window.navigator, "clipboard", {
  configurable: true,
  value: {
    writeText(text) {
      clipboard.text = text;
      clipboard.writes += 1;
      return Promise.resolve();
    },
  },
});

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
 * The theme system. Two themes, both shipped, the user picks and the promise
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
 * The quiet ink has to clear the text gate.
 *
 * `--color-muted` is what every 9–11px label in the app is set in the
 * eyebrows, the meta line, the tallies, the hints under a field which is
 * exactly the size that cannot afford to be under 4.5:1. Light's was #7d7a71:
 * 4.29:1 on the card and 3.86:1 on the sunken tone, both short. Measured here
 * rather than asserted as a hex, so the check survives the next re-pick.
 */
const srgb = (c) => (c / 255 <= 0.04045 ? c / 255 / 12.92 : ((c / 255 + 0.055) / 1.055) ** 2.4);
const luminance = (hex) => {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  return 0.2126 * srgb(r) + 0.7152 * srgb(g) + 0.0722 * srgb(b);
};
const contrast = (a, b) => {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
};
/* First declaration wins: `@theme` (light) is emitted before `:root.theme-dark`.
   The minifier shortens `#ffffff` to `#fff`, hence the three-digit branch. */
const token = (name, block = css) => {
  const hex = new RegExp(`--color-${name}:\\s*(#[0-9a-f]{3,6})`, "i").exec(block)?.[1];
  return hex?.length === 4 ? `#${[...hex.slice(1)].map((c) => c + c).join("")}` : hex;
};
const lightMuted = token("muted");
/* The three grounds muted text is actually set on. */
check(
  `the quiet ink clears 4.5:1 on every light ground (${lightMuted})`,
  ["surface", "plane", "sunken"].every((ground) => contrast(lightMuted, token(ground)) >= 4.5),
);

/*
 * The one claim about `request()` this mode can make.
 *
 * `VITE_USE_FIXTURES=1` is inlined as a literal, so every `if (USE_FIXTURES)
 * return fixtures.x()` makes the HTTP path dead code and the bundler removes
 * `request` entirely it is not in the artifact to assert against, and there
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
 * One contract, one error type, one 404 branch.
 *
 * `peaks` returns null when the file is not there yet `dub.wav` does not
 * exist until the mix stage has run and the fixture call was routed *around*
 * the catch that does it, so the one branch this method exists for behaved one
 * way against the server and threw in the only mode that is ever tested. It is
 * the same bug the fixtures' plain `new Error(...)` rejections cause everywhere
 * else: an error with no code and no status cannot be branched on at all.
 */
const fixtureSource = readFileSync(new URL("../src/lib/fixtures.ts", import.meta.url), "utf8");
check(
  "the fixture's 404 goes through the same catch as the server's",
  /try \{\s*if \(USE_FIXTURES\) return await fixtures\.peaks/.test(apiSource),
);
check(
  "every fixture rejection is an ApiError, not a bare Error",
  !/new Error\(/.test(fixtureSource) && /new ApiError\(/.test(fixtureSource),
);
/* And the invented busy sentence is gone from the artifact: 409/busy is raised
   for edits the server *refuses*, which are never queued behind anything. */
const bundleSource = readFileSync(new URL(bundle, assetDir), "utf8");
check("a refusal is never reported as a queued job", !/queued behind it/.test(bundleSource));
/* One table for what a PATCH does, imported by both sides of the seam. The
   fixture used to carry its own copy, which modelled three of the seven
   patchable fields a speaker change kept the clip in the old voice. */
check(
  "the fixture applies patches through the shared prediction, not a copy",
  /import \{ applyPatch \} from "\.\/patch"/.test(fixtureSource) &&
    /Object\.assign\(seg, applyPatch\(seg, patch\)\)/.test(fixtureSource) &&
    !/locked\.keep = true/.test(fixtureSource),
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
 *    the same one everything a run kept wore the warning colour, so an
 *    already-English video rendered as a wall of amber with nothing wrong in
 *    it and "kept is not the warn colour" is the whole fix, in one line.
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
   take is a property of the ground it sits on 18% of a dark hue on a white
   lane and 26% of a light one over a waveform on near-black. */
check("the mark wash is a theme token, per theme", /--mark-wash:/.test(css) && /--mark-wash:/.test(darkBlock));

/*
 * The interaction accent.
 *
 * "The dark mode is too dim the colours are dead" was a theme in which the
 * only saturated pixels on screen were the state hues, and the thing to press
 * was a near-white rectangle. The fix is one token, and the three things that
 * hold about it are structural rather than a hex:
 *
 * 1. It ships in both themes. A `--color-accent` declared only in `@theme`
 *    would make dark's primary button near-white again the moment somebody
 *    edits the light value, silently.
 * 2. In light it *is* ink. That is not a placeholder it is the promise that
 *    naming the accent changed nothing about the theme nobody complained
 *    about and it is why every `bg-accent` below is safe to have replaced a
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
   the utility it was checking for the check would fail on itself. */

/*
 * The desktop shell paints its window before the webview exists, so its
 * `backgroundColor` has to be the dark plane exactly any drift is a flash of
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
 * colour that no theme can restate and no measurement covers it is how the
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
/* The hero is gone on purpose: the nav pill says Runs, the card says what it
   does, and a display title with a lede over both was the page explaining
   itself to people already using it. */
check(
  "home screen renders without hero copy",
  root.textContent.length > 0 &&
    !/Entirely on this machine/.test(root.textContent) &&
    !/Point it at a video/.test(root.textContent),
);
check("…and still carries the new-dub card", root.querySelector('[data-region="new-dub"]') != null);
check("…whose action still says what it starts", /Start dubbing/.test(root.textContent));
/* Context is the one optional field sitting under a required one of the same
   size, and it says so in the label rather than three lines below it. */
check("the context field leads with Optional", /Optional Context/.test(root.textContent));

/*
 * The toggle itself, driven through the DOM. jsdom does not run index.html's
 * inline boot script, so what this proves is the second half of the contract:
 * the bundle applies the stored choice on mount, defaulting to dark.
 */
/*
 * One button, not two. A binary does not need a radio group: the pair meant two
 * tab stops for one decision and a dead cell half the time, because pressing the
 * theme you are already in does nothing. What the single button owes instead is
 * to *say* which way it goes the ambiguity a lone sun-or-moon has so the
 * accessible name is the destination, and that name is what these read.
 */
const themeToggle = () => document.querySelector("[data-theme-toggle]");
const themeGoesTo = () =>
  /^Switch to (light|dark) theme$/.exec(themeToggle()?.getAttribute("aria-label") ?? "")?.[1];
check("the header carries a theme toggle", themeToggle() != null);
check("…and it is one control, not a pair", document.querySelectorAll("[data-theme-toggle]").length === 1);
check("dark is the default with nothing stored", document.documentElement.classList.contains("theme-dark"));
check("the toggle says which theme is on", themeToggle().getAttribute("data-theme-toggle") === "dark");
check("…and names the one it would switch to", themeGoesTo() === "light");

/*
 * "Each theme paints its own ground" is the one claim worth checking against a
 * real cascade rather than a regex, because the failure it guards a theme
 * that flips the tokens but leaves the canvas on the other theme's colour is
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

themeToggle().dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
await new Promise((resolve) => setTimeout(resolve, 120));
check("toggling drops the dark class", !document.documentElement.classList.contains("theme-dark"));
check("the choice is persisted", dom.window.localStorage.getItem("dubbing-studio.theme") === "light");
check("light paints its own canvas", canvas() === "rgb(247, 246, 242)");
check(
  "the toggle follows the choice",
  themeToggle().getAttribute("data-theme-toggle") === "light" && themeGoesTo() === "dark",
);
check(
  "theme-color follows the theme",
  document.querySelector('meta[name="theme-color"]').getAttribute("content") === "#f7f6f2",
);

themeToggle().dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
await new Promise((resolve) => setTimeout(resolve, 120));
check("toggling back restores dark", document.documentElement.classList.contains("theme-dark"));
check("dark is persisted too", dom.window.localStorage.getItem("dubbing-studio.theme") === "dark");
check("…and one button did both directions", themeGoesTo() === "light");

// An existing run's row has to answer "where did this get to" without being
// opened the whole point of listing it.
await new Promise((resolve) => setTimeout(resolve, 200));
const runs = root.textContent;
check("runs say how far they got", /Complete/.test(runs) && /Running translate/.test(runs));
check("a failed run names the stage", /Failed at fetch/.test(runs));
check("runs say when they last moved", /just now|hours? ago|days? ago/.test(runs));
check("context field is present", /Context/.test(root.textContent));
check("setup is reachable from the header", /Setup/.test(root.textContent));

/*
 * The import screen's composition, pinned structurally.
 *
 * The screen is three regions and not one column: the primary "new dub" card,
 * the options rail beside it, and existing runs full width underneath. jsdom
 * has no layout, so what can honestly be asserted is that all three exist and
 * hold what they claim to a redesign that quietly drops the rail, or houses
 * the runs inside the form card, fails here rather than in a screenshot.
 */
check("the brand chip carries the drawn mark", document.querySelector("[data-brand] img") != null);
check(
  "the form is a card and a rail, not one column",
  document.querySelector('[data-region="new-dub"]') != null &&
    document.querySelector('[data-region="options"]') != null,
);
check("existing runs are a region of their own", document.querySelector('[data-region="runs"]') != null);
check(
  "…with one card per run in outputs/",
  document.querySelectorAll('[data-region="runs"] li').length === 3,
);

/*
 * The two language lists are *different lists*, and that is not a typo to be
 * tidied away: the ASR and the translator read more languages than Qwen3-TTS
 * can speak, so Hebrew and Arabic are source-only. Offering Hebrew as a dub
 * target creates a run whose tts stage can only fail.
 */
const selectFor = (label) => document.querySelector(`select[aria-label="${label}"]`);
const optionsOf = (select) => [...select.options].map((o) => o.value);
const srcLangSelect = selectFor("Spoken language");
const dubInto = selectFor("Dub into");
check("both language selects are on the screen", srcLangSelect != null && dubInto != null);
check(
  "spoken and dub-into are different lists",
  optionsOf(srcLangSelect).join() !== optionsOf(dubInto).join(),
);
check(
  "…and only the source list offers a language the voice cannot speak",
  // Arabic is the source-only language now; Hebrew became a target via the
  // LoRA adapter, and the same-language pair (he→he) must be expressible.
  optionsOf(srcLangSelect).includes("ar") && !optionsOf(dubInto).includes("ar"),
);
check(
  "Hebrew is a dub target, so a same-language pair is expressible",
  optionsOf(dubInto).includes("he"),
);

/*
 * Where the transcript comes from.
 *
 * `--transcript auto|captions|asr` was accepted by the server from the first day
 * and reachable only from the CLI, so a user who could hear that the
 * auto-captions were mangled the case AGENTS.md's invariant 4 exists for had
 * no way to say so from the screen that starts the run. `auto` is the pipeline's
 * own answer and stays the default; the point is that the other two exist.
 *
 * `captions` (a *path* to a caption file) stays server-side: it is a local path
 * a browser cannot produce, and offering a text box for it would be a control
 * that only works in the desktop shell.
 */
const transcript = selectFor("Transcript source");
check("the transcript source is pickable at last", transcript != null);
check(
  "…offering exactly the CLI's three answers",
  optionsOf(transcript).join() === "auto,captions,asr",
);
check("…defaulting to the pipeline's own", transcript.value === "auto");
check(
  "…with the one sentence that says when to override it",
  /auto-generated and mangled/i.test(root.textContent),
);

/*
 * What this screen commits you to, and what it does not.
 *
 * Three sentences, each one an answer to a question the screen used to leave
 * open. They are not decoration: the observed behaviour is a user who stalls on
 * the genre radio because starting a run feels irreversible, and a user who
 * discovers the source field was mandatory by pressing the primary button.
 */
check(
  "the rail says which of these choices are final",
  /Genre, register and context can be changed later; the source and languages cannot\./.test(
    document.querySelector("[data-rail-note]")?.textContent ?? "",
  ),
);
check(
  "the one mandatory field says so before the refusal does",
  document.querySelector("[data-required]") != null &&
    document.querySelector('[aria-label="Source"]').getAttribute("aria-required") === "true",
);
check(
  "a third language kept now can still be dubbed line by line later",
  /individual lines can still be switched to dubbed later, in the editor/.test(root.textContent),
);
check(
  "the scope placeholders read as examples, not as values already set",
  document.querySelector('[aria-label="Duration cap in seconds"]').placeholder === "e.g. 320" &&
    document.querySelector('[aria-label="Run name"]').placeholder === "e.g. my_first_dub",
);

/*
 * A third language is the case the two selects cannot express: English inside a
 * Hebrew→German run is neither the spoken language nor the dub's, and the
 * pipeline KEEPS it played as recorded, subtitled unless the run opts in.
 * That opt-in was reachable only from the CLI, so a run started here could
 * never ask for it. It is off by default, because off is what the pipeline does
 * when nobody says otherwise, and the clause under it is not decoration: the
 * whole difference between the two states is what happens to speech the user
 * has not thought about yet.
 */
const foreignBox = () => document.querySelector("[data-dub-foreign] input[type=checkbox]");
check("a third language can be dubbed rather than kept", foreignBox() != null);
check("…and the default is the pipeline's, which is off", foreignBox().checked === false);
check(
  "…with the consequence of leaving it off spelled out",
  /Dub foreign speech/.test(root.textContent) &&
    /plays as recorded, subtitled/.test(root.textContent),
);

// Genre and register are two-way choices with a clause each, so they are rows
// and a pill rather than two more dropdowns but they are still one-of-N, and
// one of each pair is always on.
const radios = () => [...document.querySelectorAll('[role="radio"]')];
check("genre and register are pickable without a dropdown", radios().length === 4);
check(
  "…and each pair has exactly one answer",
  radios().filter((r) => r.getAttribute("aria-checked") === "true").length === 2,
);
check(
  "…named with the clause that makes them mean something",
  /Narrated over pictures/.test(root.textContent) && /Scripted, spoken in scene/.test(root.textContent),
);

// The primary action, and its refusal: a run with no source is the one thing
// this screen must not send.
const startButton = () =>
  [...document.querySelectorAll("button")].find((b) => b.textContent.includes("Start dubbing"));
check("the primary action is on the card", startButton() != null);
startButton().dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
await new Promise((resolve) => setTimeout(resolve, 150));
check("starting with no source is refused, in words", /Give it a video/.test(root.textContent));
[...document.querySelectorAll("button")]
  .find((b) => b.textContent === "Dismiss")
  .dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
await new Promise((resolve) => setTimeout(resolve, 120));
check("…and the refusal can be dismissed", !/Give it a video/.test(root.textContent));

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
check("a mixed result is counted", /4 of 8 need attention/.test(setup));
check("no continue while something is missing", ![...document.querySelectorAll("button")].some((b) =>
  b.textContent.includes("Continue to projects"),
));

/*
 * What a failure COSTS, not just that there is one.
 *
 * Every missing row used to be the same red X and the same word. So a gated
 * Hugging Face token the run works, everybody in the video becomes one
 * speaker was drawn identically to a missing ffmpeg, which was drawn
 * identically to a Demucs cache that downloads itself on first use. Three very
 * different situations, one alarm, and a list that teaches you to ignore it.
 *
 * The server grades them (`severity`) and the screen spells the grade as a word
 * beside the state, because a hue is not a grade to a colour-blind reader or in
 * a screenshot. `data-severity` is that grade in the DOM.
 */
const severityOf = (id) => rowSeverity(id);
function rowSeverity(id) {
  return document.querySelector(`[data-check="${id}"]`)?.getAttribute("data-severity") ?? null;
}
check(
  "every row is graded, and the grade is the server's",
  [...document.querySelectorAll("[data-check]")].every((row) =>
    ["blocking", "degrades", "optional"].includes(row.getAttribute("data-severity")),
  ),
);
check(
  "the three grades are three different situations",
  severityOf("ffmpeg") === "blocking" &&
    severityOf("hf_token") === "degrades" &&
    severityOf("model_stems") === "optional",
);
check(
  "…and each says its grade as a word, never as a hue alone",
  /Required/.test(document.querySelector('[data-check="ffmpeg"]').textContent) &&
    /Degrades/.test(document.querySelector('[data-check="hf_token"]').textContent) &&
    /Optional/.test(document.querySelector('[data-check="model_stems"]').textContent),
);
check(
  "a blocking row names the stage it stops",
  /stops the run at fetch/.test(document.querySelector('[data-check="ffmpeg"]').textContent),
);
check(
  "the footer is honest while something required is missing",
  /A required tool is missing runs will fail\./.test(
    document.querySelector("[data-footer]").textContent,
  ),
);
check(
  "…and skipping says what skipping costs",
  /Skip anyway runs will fail at fetch/.test(document.querySelector("[data-skip]").textContent),
);

/*
 * The commands are copyable.
 *
 * These are `uv run hf download …` lines and absolute `.env` paths sixty
 * characters of exactness, in an 11.5px monospace span, inside a desktop shell
 * with no address bar to paste into. Selecting one by dragging is not an
 * interaction, it is a transcription error waiting to happen.
 */
const copyButtons = [...document.querySelectorAll("[data-copy]")];
check("the parts meant to be typed are one click to the clipboard", copyButtons.length > 0);
const tokenCopy = [...document.querySelectorAll('[data-check="hf_token"] [data-copy]')].find((b) =>
  b.getAttribute("data-copy").endsWith(".env"),
);
check("…including the absolute path of the .env the server actually reads", tokenCopy != null);
tokenCopy.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
await new Promise((resolve) => setTimeout(resolve, 80));
check("clicking one copies it exactly", clipboard.text === tokenCopy.getAttribute("data-copy"));
check("…and says it did", /Copied/.test(tokenCopy.getAttribute("title")));

/*
 * Install from the app.
 *
 * The button exists for one state and no other: a row the *server* says it has
 * a command for (`installable`), that is currently missing. A model row is
 * missing and has no button, which is the assertion that matters most the
 * alternative is a button that posts an id the server refuses, and the user
 * finds out by reading a 400.
 */
const rowOf = (id) => document.querySelector(`[data-check="${id}"]`);
const installButton = (id) =>
  [...(rowOf(id)?.querySelectorAll("button") ?? [])].find((b) => /Install/.test(b.textContent));

check("a missing installable row offers to install itself",
  Boolean(installButton("ffmpeg")) && Boolean(installButton("sox")));
check("a row nothing can install offers no button",
  !installButton("hf_token") && !installButton("model_stems") && !installButton("model_translate"));
check("a passing row offers no button", !installButton("disk"));

const fixtureCalls = globalThis.__DUBBING_FIXTURE_CALLS__;
const before = fixtureCalls.install;
installButton("ffmpeg").dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
await new Promise((resolve) => setTimeout(resolve, 150));
check("clicking starts exactly one install", fixtureCalls.install === before + 1);
check("the row it started says so", /Installing/.test(rowOf("ffmpeg").textContent));
// The button is replaced by the progress line, not doubled up beside it, and
// the line carries the installer's own last words the only honest progress a
// poll can show for something that takes minutes.
check("…with its button swapped for the spinner", !installButton("ffmpeg"));
await new Promise((resolve) => setTimeout(resolve, 250));
check("…and the last line of output", /Fetching ffmpeg|Downloading|Pouring/.test(
  rowOf("ffmpeg").textContent,
));
// One at a time: the server 409s a second install either way, but the screen
// has to say so before the click, not after.
check("the other install button is disabled while one runs", installButton("sox").disabled === true);
installButton("sox").dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
check("…and a click on it starts nothing", fixtureCalls.install === before + 1);

// The simulated install ends, the server re-probes, and the page re-runs the
// whole checklist one row turning green is not the same claim as the machine
// being one step readier.
await new Promise((resolve) => setTimeout(resolve, 1400));
check("the installed row turns Ready", /Ready/.test(rowOf("ffmpeg").textContent));
check("…and drops its Install button", !installButton("ffmpeg"));
check("…and the count comes down", /3 of 8 need attention/.test(root.textContent));
check("the other row can be installed again", installButton("sox").disabled === false);

/*
 * The state the footer's kind branch was written for, reachable at last.
 *
 * `ok` is the conjunction of the BLOCKING rows only that is the server's
 * contract and the fixture used to compute it as "every row passes", which is
 * stricter than the server and made this whole branch dead code: no fixture
 * state could ever have both `ok: true` and a failing row. With both tools
 * installed the machine IS ready, with a token still missing and a Demucs cache
 * still un-downloaded, and the screen has to say both halves of that.
 */
installButton("sox").dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
await new Promise((resolve) => setTimeout(resolve, 1400));
check(
  "with both tools in, the machine is ready and does not claim all checks pass",
  document.querySelector("[data-readiness]").textContent === "Ready to run",
);
check("…with the two that are still red still counted", /6\/8/.test(root.textContent));
check(
  "…so the way forward appears",
  [...document.querySelectorAll("button")].some((b) =>
    b.textContent.includes("Continue to projects"),
  ),
);
check(
  "…and the footer stops calling a gated token an optional item for wider language pairs",
  /Everything required is ready\..*will still run just worse\./.test(
    document.querySelector("[data-footer]").textContent,
  ),
);
check(
  "…while the two rows that are still red keep their own grades",
  rowSeverity("hf_token") === "degrades" && rowSeverity("model_stems") === "optional",
);

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
check("setup does not trap navigation", /Start dubbing/.test(root.textContent));

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
/*
 * …in words, not in manifest tokens.
 *
 * `keep_reason` is a pipeline enum (`latin`, `speaker_en`, `tts_failed`, and —
 * after a headless re-run of a keep made right here `user`), and it was
 * printed raw. So the app told the user "Kept because user" about a button they
 * had pressed themselves, while special-casing exactly two of the tokens
 * somewhere else entirely. One mapping, used by the rail and the panel both.
 */
check("…in a phrase rather than a manifest token", /already in the target language/.test(editor));
check(
  "…with no raw enum left on screen",
  !/\blatin\b/.test(editor) && !/speaker_en/.test(editor) && !/tts_failed/.test(editor),
);
check("…and where speech is uncovered", /Audible, uncovered/.test(editor));
check("the empty rail is not an apology", !/No line selected/.test(editor));

/*
 * …and the ⋯ menu does not say it a second time.
 *
 * The menu's first half is the run's health the same tally and the same gap
 * list the rail is showing right now, permanently, three inches to the left. So
 * with nothing selected the menu drops that half and keeps the half the rail has
 * never carried. Select a line and the rail becomes that line's, and the menu
 * carries the lot again (checked further down, where a line is selected).
 */
const runMenuTrigger = () =>
  [...document.querySelectorAll("button")].find((b) =>
    /^Run (health and files|files and options)$/.test(b.getAttribute("aria-label") ?? ""),
  );
check(
  "with the rail already summarising the run, the menu is not a second copy",
  runMenuTrigger()?.getAttribute("aria-label") === "Run files and options",
);
runMenuTrigger().dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
await new Promise((resolve) => setTimeout(resolve, 150));
check("…so it opens without a run-health section", document.querySelector("[data-run-health]") == null);
check(
  "…and keeps the files, the metadata and the run options",
  /run dir/.test(root.textContent) && /Run options/.test(root.textContent),
);
/* …without saying its own name twice: with the health half gone, the panel's
   title *is* "This run", so the section label under it would be the same two
   words four pixels lower. */
check(
  "…and heads that metadata once, not twice",
  (document.querySelector('[role="dialog"]').textContent.match(/This run/g) ?? []).length === 1,
);
runMenuTrigger().dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
await new Promise((resolve) => setTimeout(resolve, 150));
check("the script loaded", document.querySelectorAll('[role="option"]').length > 40);
check("timeline drew marks", document.querySelectorAll('[aria-label^="Segment "]').length > 40);

const rows = () => [...document.querySelectorAll('[role="option"]')];
const rowFor = (id) => rows().find((r) => r.textContent.includes(`#${id}`));

/*
 * The centrepiece. Segment 1 of the fixture is a real Hebrew line with a real
 * English translation, and the point of the redesign is that a reviewer can see
 * *both at once*, in one row, without selecting anything that is what makes
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
 * Bidi. The row's chrome timecode, id, buttons is laid out left-to-right no
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
 * text` on most of it and the row used to print the identical sentence twice,
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
 * A kept line says WHY it was kept once, as a clause on the meta line. It used
 * to be a paragraph of its own under the two texts, which on an all-kept run is
 * seventy-three identical sentences down the page; then it was one clause saying
 * "original audio plays here", which is the *consequence*, and the consequence
 * is already spelled two spans to the left by the word "Keep".
 *
 * The question a reviewer checking keeps actually has is why this one was kept,
 * and until now it took selecting the row and reading the panel to answer it —
 * seventy-three times. `keepReason` is the same mapping the panel and the run
 * summary use, so the three cannot drift.
 */
const keptRow = rows().find((r) => r.textContent.includes("already in the target language"));
check("kept lines say why they were kept, on the row", keptRow != null);
const keptMeta = [...keptRow.querySelectorAll("button")].find((b) => /Keep/.test(b.textContent));
check(
  "…as a clause on the row's meta line, not a fourth line of its own",
  keptMeta != null && keptMeta.textContent.includes("already in the target language"),
);
// The reason and nothing else: the fixture's keeps all have a phrase, so no row
// falls back to restating what the badge says.
check(
  "…in place of the sentence the state badge already says",
  !/original audio plays here/.test(root.textContent),
);
// And the span nobody transcribed says so in words: `uncovered` had no phrase
// in the map, so the row would have printed the manifest's own token.
check(
  "…including the spans the transcript never claimed",
  rows().some((r) => r.textContent.includes("nothing was transcribed here")),
);

/*
 * The encoding is a drawn shape now. Two of the Unicode glyphs it used to set —
 * U+25A3 for "kept" above all are outside the UI font, so the browser fell
 * back per character and drew a lumpy square on every kept row.
 */
check("the state shape is drawn, not set in a font", !editor.includes("▣") && !editor.includes("◆"));
check("every row carries its state shape", rows().every((r) => r.querySelector("svg") != null));

// The state is a word on every row. Light-mode "kept" is 2.17:1 against the
// card under the 3:1 gate and there is no legend on screen any more.
check(
  "every row says its state in words",
  rows().every((r) => /Dubbed|Keep|Fail|Render|Needs voice|Needs translation/.test(r.textContent)),
);
/*
 * …in words that are the state, not abbreviations of it.
 *
 * "Voice" and "Text" read as nouns a column saying "Text" beside a line of
 * text says nothing and "Dub" was the same word as the verb on the row's own
 * button, so the state and the action shared a label. The three that lost
 * something got it back; "Keep", "Fail" and "Render" are unambiguous short and
 * stay short.
 */
const segSource = readFileSync(new URL("../src/lib/segments.ts", import.meta.url), "utf8");
check(
  "the state words are the states, not abbreviations of them",
  !/short: "(Voice|Text|Dub)"/.test(segSource) &&
    /short: "Needs voice"/.test(segSource) &&
    /short: "Needs translation"/.test(segSource) &&
    /short: "Dubbed"/.test(segSource),
);
check("the legend is not permanent chrome", !editor.includes("Unclaimed time"));

/*
 * The list scrolls to rows the playhead's, the selection's, ↑/↓'s and every
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
 * over the first few seconds of them which is where every run's first mark
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
 * The lanes draw the run's actual audio behind the marks "make the audio look
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

/*
 * The hatch, named.
 *
 * It is the one mark on this strip with no row, no chip and no word anywhere in
 * the app a 135° hatch that means "unclaimed" to whoever drew it and nothing
 * to the reviewer looking at it. Every span carries the sentence, and the
 * shortcuts popover carries the same sentence for the reader who goes looking
 * rather than pointing.
 */
const hatches = [...document.querySelectorAll("[data-hatch]")];
check("the timeline draws unclaimed time", hatches.length > 0);
check(
  "…and every hatch says what it is and which seconds it covers",
  hatches.every((h) => /^Unclaimed no segment covers \d+:\d\d–\d+:\d\d$/.test(h.getAttribute("aria-label") ?? "")),
);
check(
  "…in a tooltip as well, for the mouse",
  hatches.every((h) => h.getAttribute("title") === h.getAttribute("aria-label")),
);
/*
 * …and the rail's list of them points at the map.
 *
 * "Audible, uncovered 0:52" is half an answer: a timecode does not say whether
 * that is early, late, or in the middle of the one stretch that is already fine.
 * Pointing at a row lights the hatch it is inside by focus as well as hover,
 * because a keyboard user gets the same map. The report's bounds and the strip's
 * are two measurements of the same silence, so they are matched by overlap.
 */
const litHatch = () => document.querySelector('[data-hatch="lit"]');
const gapRow = document.querySelector("[data-gap]");
check("the rail lists the uncovered spans", gapRow != null);
check("nothing is lit before anyone points at it", litHatch() == null);
gapRow.focus();
await new Promise((resolve) => setTimeout(resolve, 120));
check("…and focusing one lights the hatch it is inside", litHatch() != null);
check(
  "…the one that actually covers it",
  Number(gapRow.getAttribute("data-gap")) >= 0 &&
    /Unclaimed no segment covers/.test(litHatch().getAttribute("aria-label")),
);
gapRow.blur();
await new Promise((resolve) => setTimeout(resolve, 120));
check("…and it goes out again", litHatch() == null);

/*
 * Zoom has a floor and a way back to it.
 *
 * Every − used to halve the scale again with nothing stopping it, so a run went
 * to a smear against the left edge of an empty strip and the only route back was
 * pressing + and counting. Fit is that route, and it sits between the − it
 * rescues and the readout it changes.
 */
const zoomFit = document.querySelector("[data-zoom-fit]");
check("the timeline offers Fit", zoomFit != null && /Fit/.test(zoomFit.textContent));
check(
  "…between the zoom-out and the readout",
  zoomFit.previousElementSibling?.getAttribute("aria-label") === "Zoom out" &&
    /px\/s/.test(zoomFit.nextElementSibling?.textContent ?? ""),
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
 * only the fixtures ever produced so A/B playback was dead against the real
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
 * and the button snaps straight back out of its pressed state which is what
 * every A/B press did, because `api.audioUrl` (the seam that resolves it) had
 * no callers at all. The claim is narrow and it is the one that broke: the URL
 * that reaches the element went through the seam, and it is not the raw one.
 */
const played = [];
let sounding = null;
dom.window.HTMLMediaElement.prototype.play = function play() {
  played.push(this.src);
  sounding = this;
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
  "pressing B stops A one element, one clip",
  played.length === 2 &&
    clip(rowFor(1), "A").getAttribute("aria-pressed") === "false" &&
    clip(rowFor(1), "B").getAttribute("aria-pressed") === "true",
);
press(clip(rowFor(1), "B"));
await settle(120);
check("pressing the sounding side again stops it", clip(rowFor(1), "B").getAttribute("aria-pressed") === "false");

/*
 * A is a *window* of the source track, and the window has two edges.
 *
 * `media.source_window` has been on the wire since `Projects.enrich` was
 * written and had no consumers at all: the element got the URL, then
 * `currentTime = 0`, which fights the `#t=start,end` fragment's own seek so
 * pressing A on a line an hour in could play the top of the video and nothing
 * anywhere enforced the fragment's *end*, so it then ran on into the next
 * lines. Both edges are enforced from the numbers now, which is also the only
 * way they can be checked: jsdom loads no media, so a fragment would do nothing
 * here and a browser's handling of it is not the app's to rely on.
 */
const row9 = rowFor(9);
const window9 = JSON.parse(row9.querySelector('[data-clip="A"]').getAttribute("data-window"));
press(clip(row9, "A"));
await settle(120);
check("A knows which span of the source it is", window9[1] > window9[0]);
check("pressing A seeks to the segment's start, not to 0:00", sounding.currentTime === window9[0]);
// The element reports its own position as it plays; the span's end has to stop
// it, or A runs into the lines after it.
sounding.currentTime = window9[1] + 0.01;
sounding.dispatchEvent(new dom.window.Event("timeupdate"));
await settle(120);
check(
  "…and stops at its end instead of running on",
  clip(rowFor(9), "A").getAttribute("aria-pressed") === "false",
);
// The dub side is a clip file that is already exactly the segment, so it has no
// window to confine it and must not inherit the last one's.
press(clip(rowFor(9), "B"));
await settle(120);
sounding.currentTime = window9[1] + 5;
sounding.dispatchEvent(new dom.window.Event("timeupdate"));
await settle(120);
check(
  "the dub side plays the whole clip it is not a window",
  clip(rowFor(9), "B").getAttribute("aria-pressed") === "true",
);
press(clip(rowFor(9), "B"));
await settle(120);

/*
 * Inline editing. The translation is edited where it is read that is the
 * whole point of putting it in the row and the two hard rules are that a
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
/*
 * Opening a field brings the row to the middle.
 *
 * Selection scrolls with `block: "nearest"`, which is right for selecting and
 * wrong for editing: a row at the bottom of the list is already "nearest" where
 * it is, so ↵ on it opened a textarea in the last thirty pixels of the pane and
 * the field then grew down under the timeline strip with the caret in the
 * covered part. jsdom has no layout, so what is checked is the instruction.
 */
const editScrolls = [];
const priorScrollIntoView = dom.window.Element.prototype.scrollIntoView;
dom.window.Element.prototype.scrollIntoView = function scrollIntoView(options) {
  editScrolls.push({ uid: this.getAttribute?.("data-uid") ?? null, options });
};
clickIt(target.querySelector('[data-line="text_en"]'));
await settle(150);
const centredOnEdit = editScrolls.filter((s) => s.options?.block === "center").pop();
check(
  "opening a field centres its row",
  centredOnEdit != null && centredOnEdit.uid === target.getAttribute("data-uid"),
);
dom.window.Element.prototype.scrollIntoView = priorScrollIntoView;
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
// Saving the text drops the clip that says the old words the row must say so
// rather than keep a "Dubbed" badge and a live B button over stale audio.
check(
  "the invalidated clip is modelled locally, not left stale",
  /Needs voice/.test(rowFor(2).textContent) && clip(rowFor(2), "B").disabled,
);

/*
 * Closing a field you only opened to read must cost nothing.
 *
 * `edit.set_text` invalidates the clip and the placement and stamps a lock, so
 * a PATCH that carries the text back unchanged is a re-voice queued against a
 * line nobody edited which is exactly what "I don't want the retranscription
 * to happen without any change" is about. Fixture mode counts its own calls
 * because a no-op save is invisible from the DOM: it looks the same as no save.
 */
// The bundle is imported into *this* module's realm, so its `globalThis` is
// Node's, not jsdom's `window` the counter lands on whichever one the fixture
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

// Whitespace is not an edit either the editor commits a trimmed draft, so
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
/** The confirm panel's own buttons every bulk action opens one now. */
const dialogButton = (label) =>
  [...document.querySelectorAll('[role="dialog"] button')].find((b) => b.textContent === label);
check(
  "the script has filter chips",
  ["All", "Failed", "Unfinished", "Kept", "Edited"].every((l) => chip(l)),
);
// Two of the three locked lines are the fixture's own: the one the snapshot
// carries, and the stranded line, whose `locked:{keep:true}` is what a verdict
// flip stamps. The third is the one edited by hand a few checks above.
check("the Edited chip counts the line just edited", /Edited\s*3/.test(chip("Edited").textContent));

/*
 * What "failed" is.
 *
 * It used to be `tts.verify === "failed"` a value `dubbing/tts.py` has never
 * written; its verdicts are ok / soft / keep. A real failure is a keep the
 * pipeline decided *against itself*: the voice could not say the line
 * (`tts_failed`) or the translator could not produce the target language
 * (`mt_failed`), and either way the segment is stored as a keep with a slice of
 * the original attached so the mix is never silent. Which the UI drew as a calm
 * green "Kept original" the pipeline's own losses, reported as resolved, with
 * a Failed chip that could only ever count zero.
 */
clickIt(chip("Failed"));
await settle(200);
check("the Failed chip narrows the script", rows().length < 40 && rows().length > 0);
check("…and it finds the keeps the pipeline decided against itself", rows().length === 2);
check(
  "…which are keeps, and name the failure as the reason on the row",
  rows().every((r) => /voice failed|translation failed/.test(r.textContent)),
);
const bulk = [...document.querySelectorAll("button")].find((b) =>
  /Re-voice (these \d+|this line)/.test(b.textContent),
);
check("a filtered set can be fixed in one job", bulk != null);
// The bulk set used to be `visible.filter(seg => !seg.keep)`, which is every
// row on this screen removed: both buttons offered a job for nobody. And the
// two buttons are not two ways to do one thing an `mt_failed` line's
// `text_en` is the source line the translator copied in when it gave up, so
// re-voicing it would synthesize the wrong language.
check("…over the lines the chip actually found", /Re-voice this line/.test(bulk.textContent));
check(
  "…and the one the translator lost is re-translated instead",
  [...document.querySelectorAll("button")].some((b) => /Re-translate this line/.test(b.textContent)),
);
/*
 * n=1, in the copy. "Re-voice these 1" and "One job, not 1." are the same
 * mistake in two parts of speech, and this is the set that produces it: the
 * Failed chip on a healthy run finds one of each.
 */
check("…without saying “these 1”", !/these 1\b/.test(root.textContent));
check("…or promising one job instead of one job", !/One job, not 1\./.test(root.textContent));

/*
 * And the price, before the click.
 *
 * Every one of these buttons is minutes of model time per line "Re-voice these
 * 27" is half an hour of GPU and each was one unguarded click that asked for
 * the work and started it in the same gesture.
 */
const beforeAsking = calls().log.length;
clickIt(bulk);
await settle(150);
check(
  "a bulk model action asks first, and quotes what it costs",
  /min of model time/.test(document.querySelector('[role="dialog"]').textContent),
);
/*
 * …and the question has to be on screen to be answered.
 *
 * Every confirm hung its panel under its trigger unconditionally, which is fine
 * until the trigger is thirty pixels off the bottom of the window the
 * timeline's Split, whose panel opened 81px below the fold, so the strip's one
 * destructive gesture could be armed and never confirmed with a mouse. The panel
 * measures itself and flips above the trigger when it must; the side it chose is
 * on the panel, so it is checkable from outside. (jsdom has no layout, so what
 * this proves is that the decision is made and recorded, not which way it went.)
 */
check(
  "…and the question records which side it opened on",
  ["above", "below"].includes(
    document.querySelector('[role="dialog"]').getAttribute("data-confirm-side"),
  ),
);
check("…and sends nothing while it is asking", calls().log.length === beforeAsking);
clickIt(dialogButton("Cancel"));
await settle(150);
check("…so cancelling queues no model time", calls().log.length === beforeAsking);
clickIt(chip("Failed"));
await settle(200);
check("the chip toggles back off", rows().length > 40);

// The panel says which failure it is, because the two have different fixes:
// `edit.resynthesize` reopens a tts_failed keep on the way in, and a mt_failed
// one needs the translator first.
clickIt(chip("Failed"));
await settle(200);
clickIt(rows()[0].querySelector('[aria-label^="Select segment"]'));
await settle(250);
check(
  "a failed line names its failure",
  /Synthesis failed for this line|Translation failed for this line/.test(root.textContent),
);
const revoice = [...document.querySelectorAll("aside button")].find((b) =>
  b.textContent.includes("Re-voice this line"),
);
check(
  "…and the way out of it is not disabled because it is a keep",
  revoice != null && !revoice.disabled,
);
clickIt(chip("Failed"));
await settle(200);

/*
 * Limbo, and the way out of it.
 *
 * `PATCH {keep:false}` invalidates the translate stage, so a "Dub it" that
 * queued nothing left the line with no translation, no clip and no job coming
 * invisible in a list of two hundred rows and unreachable by every other
 * chip. The fixture carries one of each unfinished shape (no translation, no
 * voice) and the chip has to find both and offer the one click that fixes them.
 */
clickIt(chip("Unfinished"));
await settle(200);
const stuck = rows();
// Three from the fixture one stranded by a verdict flip, one never voiced,
// one voiced and never placed plus the line whose translation was rewritten
// by hand a few checks above, which dropped the clip that said the old words.
check("the Unfinished chip finds the stranded lines", stuck.length === 4);
const stranded = stuck.find((r) => r.textContent.includes("not translated yet"));
check("…the one a Dub it left with nothing to say", stranded != null);
check(
  "…still carrying the keep lock that flip stamped on it",
  stranded.querySelector('[aria-label^="Hand-edited"]') != null,
);
/*
 * …and the one that is *not* waiting on a model.
 *
 * A line with a clip and no placement is the ordinary state of a segment
 * between a re-voice and the render that lays it down. It read as "Needs
 * voice", so the one-click fix swept it up and queued a minute of synthesis to
 * reproduce a clip that already existed and left it exactly as unplaced as it
 * found it, because only `timeline.place` can finish it.
 */
const unplaced = stuck.filter((r) => /Render/.test(r.textContent));
check("…the one that is synthesized and waiting for a render", unplaced.length === 1);
check(
  "…and the ones that only need a voice",
  stuck.filter((r) => r !== stranded && !unplaced.includes(r)).every((r) =>
    /Needs voice/.test(r.textContent),
  ),
);
check(
  "…fixable in one click, not two hundred",
  [...document.querySelectorAll("button")].some((b) =>
    /Translate & voice these 3/.test(b.textContent),
  ),
);
// …and that click costs three model loads, which it now says before it is made.
clickIt(
  [...document.querySelectorAll("button")].find((b) =>
    /Translate & voice these 3/.test(b.textContent),
  ),
);
await settle(150);
check(
  "the one-click fix quotes its model time first",
  /min of model time/.test(document.querySelector('[role="dialog"]').textContent),
);
clickIt(dialogButton("Cancel"));
await settle(150);
// Four rows, three of them model work: the render one is offered a render.
check(
  "the line that only needs a render is offered one, not a re-voice",
  [...document.querySelectorAll("button")].some((b) => /^Render 1$/.test(b.textContent.trim())),
);
clickIt(chip("Unfinished"));
await settle(200);
check("the Unfinished chip toggles back off too", rows().length > 40);

/*
 * The Kept chip, and the way out of a whole video kept by mistake.
 *
 * The keep rule read a video of English inside a he→de run as "already the
 * target" and kept every line of it, so the German dub was audibly the
 * original. The rule is fixed, but a run already on disk is still fifty-eight
 * verdicts, and the only way to change them was one line at a time. So the
 * chip that selects them offers the flip over the set it has selected.
 *
 * Four claims, and the third is the one that makes it safe to press: the set is
 * *what is on screen*, so the search box narrows it and the button says so; the
 * rewrite is confirmed before it fires; and however many lines it takes, it
 * costs one translate and one voice, not 2N.
 */
const setInput = (el, value) => {
  Object.getOwnPropertyDescriptor(dom.window.HTMLInputElement.prototype, "value").set.call(
    el,
    value,
  );
  el.dispatchEvent(new dom.window.Event("input", { bubbles: true }));
};
const bulkBar = () => document.querySelector('[data-bulk="kept"]');
const dubTrigger = () =>
  [...(bulkBar()?.querySelectorAll("button") ?? [])].find((b) => b.hasAttribute("aria-expanded"));

clickIt(chip("Kept"));
await settle(200);
check("the Kept chip narrows to the lines that play as recorded", rows().length === 29);
check(
  "…over the same count the chip is showing",
  /Kept\s*29/.test(chip("Kept").textContent),
);
/*
 * …but the button offers 25, not 29.
 *
 * Four of those keeps are spans `fill_uncovered_audible` wrote with `text: ""` —
 * audible stretches the transcript never claimed, kept so the original at least
 * plays. Dubbing one asks the translator to translate an empty string and the
 * voice to say the result: four minutes of GPU spent replacing correct original
 * audio with garbage. They are out of the work set, and the confirm says how
 * many and why rather than dropping them quietly.
 */
check(
  "…and offers to dub the ones there is something to dub",
  /Dub these 25/.test(dubTrigger().textContent),
);
check(
  "…with the bar itself accounting for the four it will not touch",
  /29 lines play as recorded, subtitled 4 with no transcript to translate/.test(
    bulkBar().textContent,
  ),
);

const beforeBulk = calls().log.length;
clickIt(dubTrigger());
await settle(150);
const dubDialog = document.querySelector('[role="dialog"]').textContent;
check(
  "rewriting 25 verdicts asks first, and says what it costs in model time",
  /25 lines/.test(dubDialog) && /min of model time/.test(dubDialog),
);
check(
  "…and names the four it is leaving out, and why",
  /\(4 skipped no transcript to translate\)/.test(dubDialog),
);
check(
  "…and still says what the flip does",
  /translate \+ voice queue behind any running job/.test(dubDialog),
);
check("…and sends nothing while it is asking", calls().log.length === beforeBulk);
clickIt(dialogButton("Cancel"));
await settle(150);
check(
  "…so cancelling changes no verdict",
  calls().log.length === beforeBulk && rows().length === 29,
);

// The search box is a filter like any other, and a bulk button that ignored it
// would act on two hundred lines while eleven are on screen.
const searchBox = document.querySelector('[aria-label="Search the script"]');
setInput(searchBox, "Qatari");
await settle(200);
check("a search narrows the kept set too", rows().length === 2);
check("…and the button counts the rows on screen", /Dub these 2/.test(dubTrigger().textContent));

/*
 * The search says what it found, in three places.
 *
 * It used to say nothing at all: you typed, rows disappeared, and the only way
 * to learn whether the word matched eleven lines or none was to scroll and
 * count with the All chip two inches away still confidently reading 73. The
 * chip counts the hits against the run, every hit is marked inside its own line,
 * and the timeline drops the marks that are not in the set.
 */
const allChip = () => document.querySelector('[data-chip="all"]').textContent;
check("the All chip counts what the search found, out of the run", /^All\d+ \/ \d+$/.test(allChip()));
const hits = [...document.querySelectorAll("[data-line] mark[data-hit]")];
check("every hit is marked where it stands", hits.length > 0);
check("…with the text's own case, not the query's", hits.every((m) => /qatari/i.test(m.textContent)));
/* The mark goes *inside* the `dir="auto"` paragraph. Wrapping or splitting the
   line instead would give each fragment its own bidi context, which reorders a
   Hebrew sentence on screen. */
check(
  "…inside the bidi context, never splitting it",
  hits.every((m) => m.closest("[data-line]")?.getAttribute("dir") === "auto"),
);
const dimmed = document.querySelectorAll("[data-mark][data-dim]");
check("the timeline drops the marks the search is not about", dimmed.length > 0);
check(
  "…and leaves the matches at full strength",
  document.querySelectorAll("[data-mark]").length > dimmed.length,
);
check(
  "…saying out loud that it is the search's set, not the run's",
  /not every kept line in the run/.test(bulkBar().textContent),
);
/*
 * …and every other bulk bar says it the same way.
 *
 * The Kept bar has carried this sentence since it was written; the Failed and
 * Unfinished bars had exactly the same property and never mentioned it. Which
 * bar is on screen is a filter away, so the claim is checked at the source: one
 * sentence, one definition, every bar and every confirm using it.
 */
const paneSource = readFileSync(new URL("../src/components/ScriptPane.tsx", import.meta.url), "utf8");
check(
  "one sentence names a searched-down set, and every bulk bar uses it",
  (paneSource.match(/not every \$\{noun\} line in the run/g) ?? []).length === 1 &&
    (paneSource.match(/searchNote\("(failed|unfinished|kept)"\)/g) ?? []).length >= 6,
);

clickIt(dubTrigger());
await settle(150);
clickIt(dialogButton("Dub these 2"));

await settle(700);
check(
  "the bulk flip is one PATCH per line, then one translate and one voice for the lot",
  calls().log.slice(beforeBulk).join() === "patch,patch,retranslate,resynthesize",
);
/*
 * …and the screen follows the lines instead of emptying under the user.
 *
 * The gesture only exists on the Kept filter and it empties it every line it
 * flipped has left. The user was left reading "Nothing matches" at the exact
 * moment their lines started being worked on. Unfinished is where they now are.
 */
check(
  "the filter follows the lines it just flipped",
  chip("Unfinished").getAttribute("aria-pressed") === "true",
);
check("…so the flipped lines are on screen, not an empty state", rows().length >= 2 && !/Nothing matches/.test(root.textContent));

/*
 * The queue, while it actually has something in it.
 *
 * Both halves of the gesture are live here: a re-translate running and a
 * re-voice queued behind it, sharing a batch. Everything the strip could not
 * say before is asserted against that state.
 */
const stripText = () => document.querySelector("[data-job-strip]")?.textContent ?? "";
check("the running job names its work and its size", /Re-translating 2 lines/.test(stripText()));
check("the tail names what is next, not a bare number", /then Re-voicing 2 lines/.test(stripText()));
check("…and never the old count-with-no-name", !/\+\d+ queued/.test(stripText()));

/*
 * The rows pulse for as long as the work does.
 *
 * `busyUids` was a local list wiped by any job's completion and by every
 * `segment` frame measured lifetime about 100 ms against a job that runs for
 * a minute a line. Derived from the queue, the mark lasts exactly as long as
 * the job naming those uids does.
 */
check(
  "the lines a job names are marked busy",
  rows().filter((r) => /animate-pulse/.test(r.className)).length >= 2,
);

/* The stage strip is not still showing the last job's final frame. */
check(
  "a new job does not inherit the finished one's progress",
  !/Running report/.test(root.textContent),
);

// The queue opens from the strip that raised the question.
clickIt(document.querySelector("[data-queue-trigger]"));
await settle(150);
const queuePanel = () => document.querySelector("[data-queue-panel]");
check("clicking the strip opens the queue", queuePanel() != null);
check(
  "…listing every job that has not finished, running and queued alike",
  queuePanel().querySelectorAll("[data-queue-job]").length === 2,
);
check(
  "…each with its own Cancel, so the queued half is reachable",
  [...queuePanel().querySelectorAll("[data-queue-job]")].every((li) =>
    [...li.querySelectorAll("button")].some((b) => b.textContent.trim() === "Cancel"),
  ),
);
check(
  "…and the three costs are written down where the queue is",
  /about 20 seconds a line/.test(queuePanel().textContent) &&
    /about a minute a line/.test(queuePanel().textContent) &&
    /re-encodes the whole file/.test(queuePanel().textContent),
);
/* `state.log` has been collected since the first version and never rendered —
   a failure's actual message went to a variable nobody could read. */
check("…and the log finally has somewhere to be read", queuePanel().querySelector("[data-queue-log]") != null);
clickIt(document.querySelector("[data-queue-trigger]"));
await settle(120);
check("the queue closes again", queuePanel() == null);

clickIt(chip("Kept"));
await settle(200);
setInput(searchBox, "");
await settle(200);
check("the rest of the run is still kept", rows().length === 27);
clickIt(chip("Kept"));
await settle(200);
check("the Kept chip toggles back off as well", rows().length > 40);
await settle(2400);
/*
 * The strip used to vanish on the frame the last job ended, which made a
 * successful job's only report its own disappearance. It holds for six seconds
 * in a still, quiet version of itself same height, same trigger, so the queue
 * is still reachable for as long as the question "what did that just do" is
 * live and then goes.
 */
check(
  "the two jobs it queued drain",
  document.querySelector("[data-job-strip]")?.getAttribute("data-job-state") === "done",
);
check(
  "…and the strip says what finished rather than just leaving",
  /Re-voicing finished/.test(document.querySelector("[data-job-strip]").textContent),
);
await settle(6400);
check("…then it goes on its own", document.querySelector("[data-job-strip]") == null);

/*
 * Cancelling the gesture, not the step the audit's disaster, made unreachable.
 *
 * Cancel the running re-translate and the re-voice queued behind it used to run
 * anyway, on lines whose translation had just been abandoned: 27 lines
 * synthesised from nothing, each landing as a `tts_failed` keep. One click, and
 * the user had said "stop dubbing these". The strip's Cancel could only ever
 * reach the running job, so the second half was not merely missed it was
 * unreachable.
 */
clickIt(chip("Kept"));
await settle(200);
/* The last kept line that there is something to dub. The four no-transcript
   lines sort to the end of this filter and "Dub it" on one of them is a verdict
   flip and nothing else correctly, there is no source text to translate so
   taking the last row outright would be asserting a batch on a line that cannot
   have one. Same marker the no-transcript check above reads. */
const batchRow = rows()
  .filter((r) => !r.textContent.includes("nothing was transcribed here"))
  .at(-1);
clickIt(batchRow.querySelector('[aria-label^="Select segment"]'));
await settle(200);
const beforeCancel = calls().log.length;
clickIt([...document.querySelectorAll("aside button")].find((b) => /Dub it/.test(b.textContent)));
await settle(250);
check(
  "one flip queues both halves of the work",
  calls().log.slice(beforeCancel).join() === "patch,retranslate,resynthesize",
);
check("…and both are in the queue", /then Re-voicing 1 line/.test(stripText()));

// Cancel does not act until it knows which was meant.
clickIt([...document.querySelectorAll("[data-job-strip] button")].find((b) => b.textContent.trim() === "Cancel"));
await settle(150);
check(
  "cancelling a batched job asks which was meant",
  /Cancel just this, or the whole batch \(1 line\)\?/.test(root.textContent),
);
check(
  "…in the app, never an OS sheet",
  document.querySelector('[role="dialog"][aria-label="Cancel"]') != null,
);
clickIt(dialogButton("The whole batch"));
await settle(600);
check(
  "…and the whole batch stops: nothing is left to voice untranslated lines",
  document.querySelector("[data-job-strip]")?.getAttribute("data-job-state") === "done",
);
check(
  "…and the strip says it was cancelled rather than silently emptying",
  /cancelled/.test(document.querySelector("[data-job-strip]").textContent),
);
/* The proof the disaster is gone: the re-voice never ran, so no line was
   synthesised from a translation that had just been abandoned. */
check(
  "…the queued re-voice never reached the model",
  calls().log.slice(beforeCancel).join() === "patch,retranslate,resynthesize",
);
// Back to the whole run, for the sections below that read it unfiltered.
clickIt(chip("Kept"));
await settle(200);

/*
 * A soft verification is a concern, and it was drawn as furniture.
 *
 * Segment 17's clone said 78% of its words. The row's triangle was the same
 * muted grey as the timecode next to it and the shelf's summary was the same
 * grey as a line with nothing wrong, so a clip that said four fifths of its
 * sentence looked exactly like one that said all of it. Both join the pending
 * family amber, this app's hue for "still outstanding" with red kept for
 * the hard failure. All of it is icons and a bar, never text: `--color-pending`
 * is 3.70:1 in light, which clears the 3:1 non-text gate and not the text one.
 */
const softRow = rowFor(17);
const softFlag = softRow.querySelector("[data-concern]");
check("a soft verification is flagged on the row", softFlag?.getAttribute("data-concern") === "soft");
// `.className` on an SVG element is an SVGAnimatedString, not a string.
const softFlagClass = softFlag.getAttribute("class");
check("…in the pending family, not as furniture", /text-warning/.test(softFlagClass));
check(
  "…and it is not the same mark as a hard failure",
  !/text-critical/.test(softFlagClass) && !/text-muted/.test(softFlagClass),
);
clickIt(softRow.querySelector('[aria-label^="Select segment"]'));
await settle(200);
const verifyShelf = [...document.querySelectorAll("aside details")].find((d) =>
  d.querySelector("summary").textContent.includes("Verification"),
);
check("…and the shelf says so while it is still shut", verifyShelf.getAttribute("data-tone") === "caution");
clickIt(verifyShelf.querySelector("summary"));
await settle(150);
check(
  "…with an amber overlap bar rather than the blue one that meant “fine”",
  verifyShelf.querySelector("[data-overlap-bar]")?.getAttribute("data-overlap-bar") === "soft" &&
    /var\(--color-warning\)/.test(verifyShelf.querySelector("[data-overlap-bar]").getAttribute("style")),
);
clickIt(verifyShelf.querySelector("summary"));
await settle(150);

/*
 * The selection panel. It holds everything true *about* a line and no text
 * field for the line itself the text is in the script, where the comparison
 * is. Four shelves, all shut, all named for what is on them.
 */
// Segment 2 is the one just edited by hand, so its Advanced shelf has locks on
// it which is the only state in which "Release locks" exists to be found.
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
 * "Keep original" is the direction that needs no work the original audio is
 * already on disk so it applies instantly and queues nothing. "Dub it" is the
 * direction that needs all of it: `edit.set_keep` invalidates the translate
 * stage, so the line loses its subtitle and its clip on the way through, and a
 * flip that sent the PATCH and stopped there left the segment with nothing to
 * say and nothing coming to fix it. That was the bug. Both claims are counted
 * rather than watched: a job that was never enqueued looks exactly like a job
 * that has not started.
 */
/*
 * …and the direction that had neither a cost nor a guard.
 *
 * `edit.set_keep` invalidates translate in BOTH directions, so pressing Keep
 * throws the line's translation away and Keep is a button, a menu item, and
 * `k`, a bare keystroke. It was the only unguarded destructive action in the
 * app, and it said nothing at all about what it destroyed.
 */
clickIt(rowFor(1).querySelector('[aria-label^="Select segment"]'));
await settle(250);
check(
  "the panel says what Keep will cost before it is pressed",
  /Switching to “Keep original” discards this line’s translation/.test(root.textContent),
);
// The lead sentence describes the CURRENT verdict. A dubbed line that opened
// with "the source audio plays untouched" read as the opposite of the button
// that was lit the exact inversion the user kept reporting.
check(
  "a dubbed line's panel says the dub replaces the source audio",
  /The dubbed voice replaces the source audio/.test(root.textContent),
);

const keptBefore = rows().filter((r) => r.textContent.includes("you chose this")).length;
clickIt(rowFor(2).querySelector('[aria-label^="Select segment"]'));
await settle(250);
// Segment 2's translation was typed by hand a few checks above, and a lock is
// honoured by `invalidate` so this one survives the flip, and warning about a
// loss that cannot happen is the same failure as staying silent about one that
// can.
check(
  "…and does not claim a loss a locked line cannot suffer",
  /Switching to “Keep original” leaves your translation in place/.test(root.textContent),
);

const sinceKeep = calls().log.length;
click("Keep original");
await settle(250);
check(
  "keep applies immediately, and the row says why it is kept",
  rows().filter((r) => r.textContent.includes("you chose this")).length > keptBefore,
);
check("keeping the original queues nothing", calls().log.slice(sinceKeep).join() === "patch");
// The whole guard: no dialog judging a run is a hundred of these but an
// undo, for as long as anyone would notice they had pressed the wrong thing.
const keepToast = document.querySelector("[data-undo-toast]");
check("…but it leaves an undo behind, naming the line", keepToast != null &&
  /Kept #2\b/.test(keepToast.textContent) && /Undo/.test(keepToast.textContent));
check(
  "…and says that a restored line comes back hand-written",
  /counts as hand-written/.test(keepToast.getAttribute("title") ?? ""),
);
/*
 * …and it is said out loud as well.
 *
 * Almost everything this screen says it says by *changing* a badge flips, a
 * strip appears, the marks get wider and a change with no text is a change a
 * screen reader cannot report. One polite region carries the three events that
 * happen without a sentence anywhere: the verdict, the zoom, and a job starting
 * or ending.
 */
const liveRegion = () => document.querySelector('[role="status"][aria-live="polite"]');
check("the editor has one polite live region", document.querySelectorAll('[role="status"][aria-live="polite"]').length === 1);
check("…and the verdict is announced in it", /Kept #2\b/.test(liveRegion().textContent));
const zoomOutButton = [...document.querySelectorAll("button")].find(
  (b) => b.getAttribute("aria-label") === "Zoom out",
);
clickIt(zoomOutButton);
await settle(150);
check("…and so is a zoom change", /Timeline zoom \d/.test(liveRegion().textContent));
clickIt([...document.querySelectorAll("button")].find((b) => b.getAttribute("aria-label") === "Zoom in"));
await settle(150);
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
check("the verdict goes both ways", !rowFor(2).textContent.includes("you chose this"));
check(
  "a dub whose translation survived queues only the voice",
  calls().log.slice(sinceKeep).join() === "patch,patch,resynthesize",
);

// The general case: a kept line whose translation the pipeline wrote loses it
// on the flip, so the translator has to run before the voice does and the
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

/*
 * And the undo, driven from the key that makes the guard necessary.
 *
 * `k` is one keystroke with no modifier and it destroys a translation. The undo
 * has to put back exactly the sentence that was there nothing else on the
 * client remembers it once the server has answered and it has to put the
 * verdict back with it, and queue the voice the flip destroyed, or the line is
 * left in the same limbo a "Dub it" used to create.
 */
const undoTarget = rowFor(1);
const priorLine = undoTarget.querySelector('[data-line="text_en"]').textContent.trim();
clickIt(undoTarget.querySelector('[aria-label^="Select segment"]'));
await settle(250);
const sinceUndo = calls().log.length;
dom.window.dispatchEvent(new dom.window.KeyboardEvent("keydown", { key: "k", bubbles: true }));
await settle(350);
check("k flips the verdict, in one keystroke", rowFor(1).textContent.includes("you chose this"));
check("…discarding the translation, as it always did", !rowFor(1).textContent.includes(priorLine));
const undoStrip = document.querySelector("[data-undo-toast]");
check("…and the strip is the only thing that still knows the line", undoStrip != null);
clickIt(undoStrip.querySelector("[data-undo-keep]"));
await settle(400);
check(
  "undo puts back the exact sentence that was there",
  rowFor(1).querySelector('[data-line="text_en"]').textContent.trim() === priorLine,
);
check("…and the verdict with it", !rowFor(1).textContent.includes("you chose this"));
check(
  "…in one PATCH, and queues the voice the flip destroyed never the limbo again",
  calls().log.slice(sinceUndo).join() === "patch,patch,resynthesize",
);
check("…and the strip goes when it is used", document.querySelector("[data-undo-toast]") == null);

// Let the jobs those flips queued drain and then let the finished strip time
// itself out, so the next assertions are about the job they ask for and not
// about the tail of one of these.
await settle(3500);
check(
  "the queue drains",
  document.querySelector("[data-job-strip]")?.getAttribute("data-job-state") === "done",
);
await settle(6400);
check("…and the finished strip lets go", document.querySelector("[data-job-strip]") == null);

/*
 * A mark on the strip is a question about a line what does it say, what did
 * it become, what does it sound like and all three answers are in the script
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
/* The keys stayed A and B; the buttons say what the sides are. "A" and "B" on
   a row a reviewer is meeting for the first time is a convention they have to
   be told, and nothing on the screen was telling them. */
check(
  "…labelled Orig and Dub, not A and B",
  clip(rowFor(9), "A").textContent.includes("Orig") &&
    clip(rowFor(9), "B").textContent.includes("Dub"),
);

/*
 * The chrome that is not permanent. The keyboard map lives behind "?", and the
 * run's health the uncovered-speech list, which is the highest-value thing
 * report.json produces lives behind "⋯". Neither of them changes while you
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
/* One vocabulary: the buttons say Orig and Dub, so the help says Orig and Dub
   about the keys that press them. */
check(
  "…naming the buttons a and b press",
  /play Orig/.test(root.textContent) && /play Dub/.test(root.textContent),
);
/* And the one mark on the timeline that has no row, chip or word anywhere else
   gets its sentence here, where somebody goes to look things up. */
check(
  "…and the hatch is explained where explanations live",
  /no segment covers them, so the dub plays the original there/.test(
    document.querySelector("[data-hatch-note]")?.textContent ?? "",
  ),
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
 * A gap list is a finding, and by now the script has been edited past the
 * report that found them: splitting a segment closes a gap and re-segmenting
 * opens new ones, and neither shows here until a render re-runs the report.
 */
check(
  "the gap list says which render found them",
  /Audible, uncovered \d+ · from the last render/.test(root.textContent),
);

/*
 * "I can't easily open the ready file." Both of the things a finished run is
 * for are one click away and named: the video in the header, the subtitles in
 * this menu. Each one goes through `openRunFile`, which reveals the file in
 * Finder inside the shell and opens the served URL in a browser the old
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

/*
 * The run options that are still a decision, editable where the run is read.
 *
 * Genre, register and context were picked once on the import screen in ten
 * seconds, before a single line had been read and then became unreachable for
 * the life of the project. Nothing about them is structural: all three are
 * inputs to the translator, which is exactly why they can change and the source
 * and the language pair cannot.
 *
 * Everything asserted here is about the *request*, because the screen shows no
 * consequence of any of them until translation runs again the same reason the
 * import screen's switches are judged on `calls.created`.
 */
check("the run's options are on the screen at last", document.querySelector("[data-run-options]") != null);
const optionSelect = (label) =>
  document.querySelector(`[data-run-options] select[aria-label="${label}"]`);
check(
  "…as the two the pipeline actually takes",
  optionSelect("Genre") != null && optionSelect("Register") != null,
);
check(
  "…and they say when they take effect, since nothing is queued",
  /Applies to the next translate or render/.test(root.textContent),
);

const setSelect = (el, value) => {
  Object.getOwnPropertyDescriptor(dom.window.HTMLSelectElement.prototype, "value").set.call(
    el,
    value,
  );
  el.dispatchEvent(new dom.window.Event("change", { bubbles: true }));
};
const updatesBefore = calls().updated.length;
setSelect(optionSelect("Genre"), "movie");
await settle(250);
check("changing one saves it", calls().updated.length === updatesBefore + 1);
check("…carrying only what changed", JSON.stringify(calls().updated.at(-1)) === '{"genre":"movie"}');

// Context is prose, so it commits explicitly: a textarea that saved on blur
// would fire on every stray click out of a note somebody was still writing.
const editContext = document.querySelector("[data-edit-context]");
check("the context note is editable in place", editContext != null);
clickIt(editContext);
await settle(120);
const contextField = document.querySelector('[data-run-options] textarea[aria-label="Context"]');
check("…in a field of its own", contextField != null);
setValue(contextField, "  the host is Dana; the guest is Prof. Ronen Levi  ");
await settle(80);
clickIt(document.querySelector("[data-save-context]"));
await settle(220);
check(
  "…and saving sends the note",
  calls().updated.at(-1).context === "the host is Dana; the guest is Prof. Ronen Levi",
);
check("…closing the field", document.querySelector('[data-run-options] textarea') == null);

clickIt(menu);
await settle(120);

/*
 * One source of truth about whether there is a video.
 *
 * The header offered "Open preview" whenever the *manifest* named one, while
 * the panel three inches below it decides on whether the file can actually be
 * reached so this run, whose preview.mp4 no fixture can serve, showed a
 * button offering to open a video directly beside a panel saying there is no
 * video. Both now read the same fact, which is the URL and not the manifest
 * key: no reachable file, no offer to open one.
 */
const openPreview = () =>
  [...document.querySelectorAll("header button")].find((b) =>
    /Open preview|Show in Finder/.test(b.textContent),
  );
check("no offer to open a preview that cannot be reached", openPreview() == null);
check(
  "…and the header stays at its height",
  document.querySelector("header").className.includes("h-11"),
);

// No panel in the editor may be blank: with no preview file the stage says why
// there is no picture rather than showing an empty rectangle.
check("the empty preview explains itself", /there is no video/.test(root.textContent));
check("the preview shows the pipeline position", /stages done/.test(root.textContent));

/*
 * What the transport is on, said in the DOM.
 *
 * Three modes the preview, the run's original audio, nothing and the chip
 * that names the middle one must not appear in the other two, because a label
 * reading "Original audio" over the finished dub is worse than no label. This
 * run's manifest names a preview.mp4 (which is what the mode is decided on;
 * fixture mode cannot serve the file itself, hence the status board below it),
 * so it is the "a preview exists" case and the chip is absent. The other two
 * modes are exercised on the two runs at the very bottom of this file.
 */
const transportBar = () => document.querySelector("[data-transport]");
check("the transport says what it is on", transportBar()?.getAttribute("data-transport") === "preview");
check(
  "no original-audio chip on a run that has a preview",
  document.querySelector("[data-transport-note]") == null,
);
/*
 * …and a preview that is named is not a preview that plays.
 *
 * The silent guard was `mode === "none"` alone, so this run preview mode, no
 * reachable file kept a live play button, and pressing it started the
 * transport's fallback clock: a playhead sweeping the strip with no audio
 * anywhere, which is the exact lie the dead play button was written to stop.
 * Nothing attached is nothing to play, whatever the mode says.
 */
const previewPlay = [...document.querySelectorAll("button")].find((b) =>
  ["Play", "Pause"].includes(b.getAttribute("aria-label")),
);
check("a named-but-unreachable preview does not get a live play button", previewPlay.disabled);
check(
  "…and says which of the two reasons it is",
  /could not be loaded/.test(previewPlay.getAttribute("title")),
);

/*
 * The header says whether the video is still about the script.
 *
 * By this point the run has been edited a dozen times, so `render.stale` is
 * true and the button is the loud one, naming the number it would re-render.
 * It is the whole point of the staleness work: a permanently-accented "Render
 * preview" could not tell "there is work waiting" from "you are up to date".
 */
const renderButton = () =>
  [...document.querySelectorAll("header button")].find((b) => /Render preview|Update the video/.test(b.textContent));
check("the header offers to update the video, and says how much", /Update the video · \d+ lines? changed/.test(renderButton().textContent));
check(
  "…as the accented button, not the quiet one",
  /bg-accent/.test(renderButton().className),
);
check(
  "the video area says the same thing in words",
  /Mixed before your last \d+ changes Update the video to hear them/.test(root.textContent),
);
/*
 * The three stages a render re-runs are drawn hollow done, but done about
 * something else. A ring rather than a lighter fill, so the difference is shape
 * and survives greyscale and colour-blindness.
 */
check(
  "timeline, mix and report are drawn hollow, not simply done",
  [...document.querySelectorAll('[title$="from the last render"]')].length === 3,
);
check(
  "the output lane is labelled as the last render",
  [...document.querySelectorAll("[data-lane-label]")].some((el) => /Output\s*last render/.test(el.textContent)),
);
check(
  "…and its waveform is drawn back, while the marks stay live",
  document.querySelector('[data-waveform="dub"]')?.hasAttribute("data-faded") === true,
);

/*
 * Confirmation is themed and local, never `window.confirm` which is drawn by
 * the OS, blocks the main thread and stops the playhead.
 */
click("Update the video");
await settle(150);
check(
  "a destructive action asks in the app, not in an OS sheet",
  /no clip yet|full video re-encode/.test(root.textContent),
);
/*
 * …and the sentence is about *this* run.
 *
 * "A full video re-encode, typically a few minutes" is true of a finished run
 * and a lie about this one: `edit.start_stage` deliberately backs a render up to
 * tts or translate to make the missing clips, so on a run with unfinished lines
 * the button loads two models. The fixture has such lines (`seedOverrides`),
 * which is the only reason this branch is reachable here at all.
 */
check(
  "…and it says what this render will actually cost",
  /model time, not the few minutes a re-encode takes/.test(root.textContent),
);
check(
  "the confirming button repeats the verb",
  [...document.querySelectorAll('[role="dialog"] button')].some((b) => b.textContent === "Render"),
);
click("Cancel");
await settle(150);
check("cancelling does nothing", !/model time, not the few minutes/.test(root.textContent));

// A model action must queue and report progress instead of blocking the UI.
click("Re-voice this line");
await settle(200);
check("model action queues and reports", /Re-voicing/.test(root.textContent));
check("editor still interactive during a job", rows().length > 40);

/*
 * The strip is one line high, always. With `flex-wrap` a long stage message
 * wrapped to a second row and moved the whole editor down mid-job the script
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
check("job stops running when done", !/Re-voicing 1 line · /.test(root.textContent));
check(
  "…and reports it, for six seconds, before clearing",
  /Re-voicing finished/.test(root.textContent),
);
await settle(6400);
check("job clears when done", !/Re-voicing/.test(root.textContent));

/*
 * A subscriber that joins late still learns what is running.
 *
 * The editor mounts and subscribes *after* the job is created that is the
 * order "Start dubbing" happens in, project first, editor second so a stream
 * that only forwards frames from the moment you connect tells a freshly created
 * run nothing at all: no job strip, and a preview stage sitting on "Nothing has
 * run yet" for the whole run. The server opens every stream with a prelude
 * (`app.py::project_events`: a log line, a stage frame per stage, and every job
 * that has not finished) for exactly this reason, and the fixtures have to do
 * the same or the flow is green here and dead in the app.
 */
click("Update the video");
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
  "re-opening a run mid-job finds the job again the stream replays it",
  /Rendering preview/.test(root.textContent),
);
/*
 * Cancelling is an ending, not a pause.
 *
 * The worker's journal saves whatever the job finished before it stopped, so a
 * cancelled run has real results on disk and the editor only treated
 * done|failed as terminal, so the stage strip and every per-segment spinner
 * stayed up and the partial work was never re-read. The refetch is the claim
 * worth counting: it is invisible in the DOM (the segments come back looking
 * the same) and it is the whole difference.
 */
const readsBeforeCancel = calls().segments;
click("Cancel");
await settle(500);
check(
  "…and it is the same job, cancellable from here",
  /Rendering preview cancelled/.test(root.textContent),
);
check("a cancel is terminal the partial work is read back", calls().segments > readsBeforeCancel);

/*
 * "Inherit" has to be able to undo an override.
 *
 * The per-segment language pair is the one field in the patch body where the
 * empty string means *clear* `null` is "not supplied, leave it alone" for
 * everything (`app.py::PatchSegment`), and the select was sending `value ||
 * null`. So picking "inherit" sent a patch the server correctly ignored: an
 * override could be set and never taken off. And clearing it is a translate
 * invalidation like any other the line was translated under the old pair.
 */
clickIt(rowFor(23).querySelector('[aria-label^="Select segment"]'));
await settle(250);
const timing = [...document.querySelectorAll("aside details")].find((d) =>
  d.querySelector("summary").textContent.includes("Timing & languages"),
);
check("the segment carries a language override", /ar → en/.test(timing.textContent));
if (!timing.open) clickIt(timing.querySelector("summary"));
await settle(150);
const spoken = [...timing.querySelectorAll("select")].find((s) => s.value === "ar");
if (!spoken) throw new Error("smoke: no spoken-language select on the override segment");
const selectValue = Object.getOwnPropertyDescriptor(
  dom.window.HTMLSelectElement.prototype,
  "value",
).set;
selectValue.call(spoken, "");
spoken.dispatchEvent(new dom.window.Event("change", { bubbles: true }));
await settle(300);
check("picking inherit clears the override", /inherit → en/.test(timing.textContent));
check(
  "…and the translation made under the old pair goes with it",
  rowFor(23).textContent.includes("not translated yet"),
);

/*
 * Claiming a span, and the sentence the composer must not eat.
 *
 * `POST /segments` refuses an overlap outright rather than clamping it, and the
 * composer cleared its textarea in the same gesture that sent the request: the
 * refusal took the typed line with it and left an error bar asking for a span
 * with nothing to put in it. The other half is the acknowledgement a segment
 * that lands mid-list, unselected and unscrolled, is announced by nothing but a
 * count on a chip.
 *
 * Adding lives on the rail's run summary, which is up exactly when nothing is
 * selected, so the run is re-opened rather than deselected there is no key
 * for "select nothing" and there should not be one.
 */
await go("/", 300);
await go("/editor/kan11_v3", 700);
const addTrigger = () =>
  [...document.querySelectorAll("button")].find((b) =>
    /^Add a segment at /.test(b.getAttribute("aria-label") ?? ""),
  );
check("an uncovered span can be claimed from the rail", addTrigger() != null);
clickIt(addTrigger());
await settle(150);
const composer = () => document.querySelector('[role="dialog"][aria-label="New segment"]');
const composerText = () => composer()?.querySelector("textarea") ?? null;
const composerEnd = () => composer().querySelectorAll("input")[1];
const addButton = () =>
  [...composer().querySelectorAll("button")].find((b) => /Add segment/.test(b.textContent));
check("…in a composer, because a segment needs words", composerText() != null);

const SPOKEN = "the line the transcript never claimed";
setValue(composerText(), SPOKEN);
// The gap's own bounds are free; stretching the end into the next line is the
// one refusal the server exists to make here.
setInput(composerEnd(), "46.0");
await settle(120);
const rowsAtStart = rows().length;
clickIt(addButton());
await settle(400);
check(
  "an overlapping span is refused, in the server's own words",
  /overlaps segment \d+/.test(document.querySelector('[role="alert"]')?.textContent ?? ""),
);
check("…adding nothing", rows().length === rowsAtStart);
check("…and leaving the typed line where it was typed", composerText().value === SPOKEN);

// The same words over a span that is actually free are taken, and the screen
// answers with the line rather than with an arithmetic difference on a chip.
setInput(composerEnd(), "44.80");
await settle(120);
const scrollsAt = scrolls.length;
clickIt(addButton());
await settle(400);
const addedRow = () => rows().find((r) => r.textContent.includes(SPOKEN));
check("a free span becomes a row", rows().length === rowsAtStart + 1);
check("…carrying the words that were typed", addedRow() != null);
check(
  "…selected, so the rail is about the line that was just made",
  addedRow().getAttribute("aria-selected") === "true",
);
check(
  "…and scrolled to, rather than left somewhere in two hundred rows",
  scrolls.slice(scrollsAt).some((s) => s.uid === addedRow().getAttribute("data-uid")),
);
check("…which is what closes the composer", composer() == null);

/*
 * A dub-wanted line whose mix plays the original.
 *
 * `tts` falls back to the keep slice rather than leaving a span silent, so the
 * verdict says dub and the audio is the source. `media.fallback` is the server's
 * word for exactly that (`Projects.enrich`, `keep_`/`fit_keep_`), and the panel
 * used to promise "The dubbed voice replaces the source audio" over a line where
 * no dubbed voice exists.
 */
const fallbackRow = rows().find((r) => /No dub yet/.test(clip(r, "B").getAttribute("title") ?? ""));
check("the run carries a line whose dub fell back to the original", fallbackRow != null);
check("…with a dead Dub side, because a fallback is not a dub", clip(fallbackRow, "B").disabled);
clickIt(fallbackRow.querySelector('[aria-label^="Select segment"]'));
await settle(250);
const panelText = () => document.querySelector("aside").textContent;
check("…and a panel that says the mix plays the original here", /No dub yet/.test(panelText()));
check(
  "…never that a dubbed voice replaces a source audio it never replaced",
  !/The dubbed voice replaces the source audio/.test(panelText()),
);

/*
 * Orig is a window of the source track, so the window has to be inside the file.
 *
 * The fixture served a tone as long as the *segment* under a window in
 * whole-track coordinates two coordinate systems in one pair of fields. The
 * seek then landed past the end of the blob and playback ended on the spot, so
 * every Orig button in demo mode was dead while the live server's were fine.
 */
const toneDur = (url) =>
  Number(new URLSearchParams(url.split("#")[0].slice("fixture:tone?".length)).get("dur"));
check(
  "every Orig window lies inside the track it is a window of",
  rows().every((r) => {
    const button = clip(r, "A");
    const span = JSON.parse(button.getAttribute("data-window") || "null");
    return span != null && span[1] <= toneDur(button.getAttribute("data-url")) + 0.001;
  }),
);
press(clip(rowFor(4), "A"));
await settle(150);
check("…so pressing Orig sounds, and stays sounding", clip(rowFor(4), "A").getAttribute("aria-pressed") === "true");
press(clip(rowFor(4), "A"));
await settle(120);

/*
 * The pipeline's own loss, settled by hand.
 *
 * A `tts_failed` keep is stored `keep=true`, so "Keep original" read as the
 * verdict already in force and did nothing at all: the guard returned on the
 * boolean, the control lit neither half, and the line stayed red whatever the
 * reviewer pressed. The verdict is (keep, keep_reason), and pressing Keep here
 * is a real change the line becomes the user's own keep, which a re-run
 * honours instead of retrying.
 */
clickIt(chip("Failed"));
await settle(250);
const rowByUid = (uid) => rows().find((r) => r.getAttribute("data-uid") === uid);
const lost = rows().find((r) => /voice failed/.test(r.textContent));
if (!lost) throw new Error("smoke: no tts_failed row");
const lostUid = lost.getAttribute("data-uid");
const lostId = Number(
  lost.querySelector('[aria-label^="Select segment"]').getAttribute("aria-label").match(/\d+/)[0],
);
clickIt(lost.querySelector('[aria-label^="More actions"]'));
await settle(150);
const menuLabels = [...document.querySelectorAll('[role="menu"] button')].map((b) => b.textContent);
check(
  "a failed line's row menu offers to settle it as a keep",
  menuLabels.includes("Keep original audio"),
);
check(
  "…beside the flip, which on a keep=true line can only offer the other direction",
  menuLabels.includes("Dub this line"),
);
clickIt(rowByUid(lostUid).querySelector('[aria-label^="More actions"]'));
await settle(200);

const choice = (label) =>
  [...document.querySelectorAll("aside button")].find((b) => b.textContent.includes(label));
check(
  "neither half is lit while the keep is the pipeline's and not the reviewer's",
  choice("Dub it").getAttribute("aria-pressed") === "false" &&
    choice("Keep original").getAttribute("aria-pressed") === "false",
);
const sinceSettle = calls().log.length;
clickIt(choice("Keep original"));
await settle(350);
check(
  "pressing Keep on a failed line is heard, in one PATCH and no job",
  calls().log.slice(sinceSettle).join() === "patch",
);
check(
  "…and the control lights the verdict it just settled",
  choice("Keep original").getAttribute("aria-pressed") === "true",
);
check("…so the Failed chip stops counting it", rows().length === 1);
clickIt(chip("Failed"));
await settle(250);
check(
  "…and the row is a keep the reviewer chose, not the failure it was",
  /you chose this/.test(rowByUid(lostUid).textContent) &&
    !/voice failed/.test(rowByUid(lostUid).textContent),
);

/*
 * …and the undo it leaves behind is about a *line*, not about a number.
 *
 * Every id is positional and renumbers on any structural edit, and the strip
 * was rendered from the id captured at the moment of the flip so removing a
 * line above it left a toast offering to undo a segment that is now somebody
 * else's number entirely.
 */
const toast = () => document.querySelector("[data-undo-toast]");
check(
  "settling a failed line leaves the same undo behind, naming the line",
  new RegExp(`Kept #${lostId}\\b`).test(toast()?.textContent ?? ""),
);
clickIt(addedRow().querySelector('[aria-label^="Select segment"]'));
await settle(250);
const timingShelf = [...document.querySelectorAll("aside details")].find((d) =>
  d.querySelector("summary").textContent.includes("Timing & languages"),
);
if (!timingShelf.open) clickIt(timingShelf.querySelector("summary"));
await settle(150);
clickIt([...timingShelf.querySelectorAll("button")].find((b) => /Remove segment/.test(b.textContent)));
await settle(150);
clickIt(dialogButton("Remove"));
await settle(400);
check("a claimed span can be given back", addedRow() == null);
check("…and the script is the length it was", rows().length === rowsAtStart);
check(
  "the undo names the line's live id, not the one it had when it was kept",
  new RegExp(`Kept #${lostId - 1}\\b`).test(toast()?.textContent ?? ""),
);

/*
 * Bounds move together, and an empty field is not a number.
 *
 * `Number("")` is 0 and 0 is finite, so clearing Start left Move enabled over
 * `start: 0` a span reaching back to the top of the video, across every line
 * in between. `edit.set_bounds` refuses it; the button must not offer it.
 */
clickIt(rowFor(5).querySelector('[aria-label^="Select segment"]'));
await settle(250);
const boundsShelf = [...document.querySelectorAll("aside details")].find((d) =>
  d.querySelector("summary").textContent.includes("Timing & languages"),
);
if (!boundsShelf.open) clickIt(boundsShelf.querySelector("summary"));
await settle(150);
const moveButton = () =>
  [...boundsShelf.querySelectorAll("button")].find((b) => b.textContent.trim() === "Move");
const startField = boundsShelf.querySelectorAll("input")[0];
check("the span is editable as two numbers", startField != null && moveButton() != null);
const patchesAtMove = calls().patch;
setInput(startField, "");
await settle(150);
check("an emptied bound is not a move to 0:00", moveButton().disabled === true);
setInput(startField, "26.00");
await settle(150);
check("…while a real number still moves it", moveButton().disabled === false);
check("…and nothing was sent in between", calls().patch === patchesAtMove);

/*
 * Play before render.
 *
 * "i can start playing. but because it still not ready it shown nothing": for
 * the whole hour between `fetch` and `mix` the play button was live and there
 * was no preview.mp4 behind it. `source.wav` is on disk from the very first
 * stage, so that is what the transport plays until the preview exists and
 * when even that is missing the button is honestly dead rather than a clock
 * ticking over silence.
 *
 * jsdom plays nothing, so every claim here is structural: which file is in the
 * element, whether the element is there at all, and what the controls say.
 */
const transportMode = () => transportBar()?.getAttribute("data-transport") ?? null;
const transportMedia = () => document.querySelector("[data-transport-media]");
const playButton = () =>
  [...document.querySelectorAll("button")].find((b) =>
    ["Play", "Pause"].includes(b.getAttribute("aria-label")),
  );
const clock = () => document.querySelector("[data-timecode]").textContent;
const pressSpace = () =>
  dom.window.dispatchEvent(new dom.window.KeyboardEvent("keydown", { key: " ", bubbles: true }));

// A run stopped mid-pipeline: past fetch, nowhere near mix.
await go("/editor/doha_panel_v2", 700);
check("a run with no preview plays its original audio", transportMode() === "source");
const audio = transportMedia();
check("…out of a media element with a real file in it", audio != null && audio.getAttribute("src").length > 0);
check(
  "…composed through the media seam, not the fixture's own name for the clip",
  !audio.getAttribute("src").startsWith("fixture:"),
);
check(
  "…with the picture hidden, so the panel stays the run's status board",
  /\bhidden\b/.test(audio.className) && /mix stage/.test(root.textContent),
);
check(
  "…and one chip says which of the two it is",
  document.querySelectorAll("[data-transport-note]").length === 1 &&
    /Original audio \(no preview yet\)/.test(transportBar().textContent),
);
check("…behind a live play button", playButton() != null && !playButton().disabled);

/* The YouTube grammar: a scrubber that drives the one shared transport, so the
   timeline playhead and the script's follow-along ride the same seek. */
const scrubber = () => document.querySelector("[data-scrubber]");
check("the player carries a seek bar", scrubber() != null);
const scrubAt = () => Number(scrubber().getAttribute("aria-valuenow"));
const scrubFrom = scrubAt();
scrubber().dispatchEvent(
  new dom.window.KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true }),
);
await settle(150);
check("keyboard-scrubbing moves the shared clock", scrubAt() >= scrubFrom + 4);
scrubber().dispatchEvent(
  new dom.window.KeyboardEvent("keydown", { key: "ArrowLeft", bubbles: true }),
);
await settle(150);
check("…and scrubs back", Math.abs(scrubAt() - scrubFrom) <= 1);
check(
  "volume and fullscreen sit in the bar",
  document.querySelector("[data-volume]") != null &&
    [...document.querySelectorAll("button")].some((b) =>
      ["Fullscreen", "Exit fullscreen"].includes(b.getAttribute("aria-label")),
    ),
);
/*
 * …in a band, not on a stage.
 *
 * With no video the panel kept the video's own 16:9, which on a wide window is
 * 270 vertical pixels of the right-hand rail spent on four short sentences —
 * while the selection panel under it scrolled a thirty-control inspector through
 * what was left. The picture gets a picture's room; a status board gets a band.
 */
const stage = transportBar().parentElement;
check(
  "with no picture the viewer collapses to a band",
  !/aspect-video/.test(stage.className) && !/\bh-40\b/.test(stage.className),
);
check(
  "…still carrying the eyebrow, a sentence and the stage track",
  document.querySelector("[data-preview-placeholder]") != null &&
    /stages done/.test(root.textContent),
);
const running = clock();
pressSpace();
await settle(250);
check("space drives it", clock() !== running);

// A run that fell over on its first stage: no source.wav, nothing to play.
await go("/editor/archive_reel", 700);
check("a run that never fetched has nothing to play", transportMode() === "none");
check("…so no media element is mounted at all", transportMedia() == null);
check(
  "…the play button is dead, and its tooltip says why",
  playButton().disabled && /fetch stage hasn't run/.test(playButton().getAttribute("title")),
);
check("…and no chip claims otherwise", document.querySelector("[data-transport-note]") == null);
const stopped = clock();
pressSpace();
await settle(250);
check("…space does nothing either the key is the same control", clock() === stopped);
// …and the run is its own: opening it used to show the finished snapshot's
// fifty-eight lines, which is a kinder server than the real one.
check(
  "a run with no segments stage has no script, and says so",
  document.querySelectorAll('[role="option"]').length === 0 && /No segments yet/.test(root.textContent),
);

/*
 * …and it is not a dead end any more.
 *
 * This screen was the end of the road: a run that fell over at fetch had an
 * empty script, a dead play button, a Render button that could only produce a
 * second failure, and nothing anywhere in the app that could start the pipeline
 * again. Two things fix it, and they are two halves of one question what
 * happened, and what do I press.
 *
 * The error is the *pipeline's own last words*, read off the project's job list
 * (`GET /api/projects/{name}` → `jobs`). It is the only copy: the event stream
 * deliberately replays nothing terminal, so after a reload there is nowhere else
 * for it to come from. Paraphrasing it would collapse a dead URL, a private
 * video and a full disk into one sentence, and only one of the three is worth
 * retrying unchanged.
 */
check(
  "a run that died says what killed it, in the pipeline's own words",
  /Video unavailable/.test(document.querySelector("[data-failure]")?.textContent ?? ""),
);
const resume = document.querySelector("[data-resume]");
check("…and offers the way back in", resume != null);
check("…named for the stage it will retry", resume.textContent.includes("Retry from fetch"));
check(
  "…promising a resume, not a restart",
  /picks up where the run stopped rather than starting over/.test(root.textContent),
);
/* Render, meanwhile, is refused with its reason instead of enqueueing a job that
   dies in the mix: there is no source.wav to lay speech over and no duration to
   lay it against, because fetch never finished.

   `renderButton()` is the staleness section's helper, re-queried rather than
   captured: this is a different project on a different screen, and on a run that
   never mixed the label is the quiet "Render preview" a run with no render
   cannot be behind one. */
check("Render is refused before the click, not by a job that dies",
      renderButton().disabled === true);
check(
  "…and says why",
  /fetch has not finished/.test(renderButton().getAttribute("title") ?? ""),
);

const logBefore = calls().log.length;
clickIt(resume);
await settle(250);
check("resuming enqueues the run job that IS the resume", calls().log.at(-1) === "resume");
check("…exactly one of them", calls().log.length === logBefore + 1);
check(
  "…and the strip picks it up, so the dead end is now a running run",
  document.querySelector("[data-job-strip]") != null,
);
/*
 * …naming the stage, and where it sits in the nine.
 *
 * A whole-run job's progress bar is the *stage's* fraction the only one the
 * pipeline reports so "62%" over a nine-stage run meant "62% through
 * transcript" and read as "62% through the dub". It says which stage and which
 * of nine, so the percentage is understood as the thing it is. Edit jobs keep
 * their naming: one is one stage by construction.
 */
await settle(700);
check(
  "a whole-run job says which stage it is on, and which of nine",
  /^\w+ stage [1-9] of 9$/.test(
    document.querySelector("[data-job-stage]")?.textContent?.trim() ?? "",
  ),
);

/*
 * The import screen's switches only exist to be *sent*: the screen shows no
 * consequence of any of them, so the request body is the only place their
 * correctness is observable. "Dub foreign speech" is the one that had no way to
 * reach a run at all the pipeline takes `--dub-foreign`, the server's
 * CreateProject has always accepted it, and the screen never put it in the
 * body, so a run started here could only ever keep a third language.
 *
 * Last in the file because starting a run is the one action on that screen that
 * mutates the fixture store and occupies the one-job queue for the length of a
 * whole pipeline.
 */
await go("/", 300);
const SOURCE = "https://example.com/watch?v=three_languages";
check("the import screen still carries the switch", foreignBox() != null);
setInput(document.querySelector('[aria-label="Source"]'), SOURCE);
await settle(80);
clickIt(foreignBox());
await settle(120);
check("the switch flips", foreignBox().checked === true);
// The other option that could not reach a run: forcing local transcription when
// the video's auto-captions are the mangled kind AGENTS.md's invariant 4 is about.
setSelect(document.querySelector('select[aria-label="Transcript source"]'), "asr");
await settle(120);

const createdBefore = calls().created.length;
[...document.querySelectorAll("button")]
  .find((b) => b.textContent.includes("Start dubbing"))
  .dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
await settle(400);
check("starting sends exactly one create", calls().created.length === createdBefore + 1);
const created = calls().created.at(-1);
check("…carrying the source that was typed", created.source === SOURCE);
check(
  "…and dub_foreign, which no run started here could ask for before",
  created.dub_foreign === true,
);
check(
  "…and the transcript source, which was the other one",
  created.transcript === "asr",
);

check(
  "no console errors",
  errors.filter((e) => !/not implemented|Not implemented/.test(e)).length === 0,
);

console.log("smoke: all checks passed");
process.exit(0);
