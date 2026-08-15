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
// jsdom ships no clipboard, and the setup screen's copy buttons only show their
// tick when the write actually resolves — so without this the one interaction
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
 * One contract, one error type, one 404 branch.
 *
 * `peaks` returns null when the file is not there yet — `dub.wav` does not
 * exist until the mix stage has run — and the fixture call was routed *around*
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
   patchable fields — a speaker change kept the clip in the old voice. */
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
 * hold what they claim to — a redesign that quietly drops the rail, or houses
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
 * auto-captions were mangled — the case AGENTS.md's invariant 4 exists for — had
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
 * pipeline KEEPS it — played as recorded, subtitled — unless the run opts in.
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
// and a pill rather than two more dropdowns — but they are still one-of-N, and
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
 * Hugging Face token — the run works, everybody in the video becomes one
 * speaker — was drawn identically to a missing ffmpeg, which was drawn
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
  /A required tool is missing — runs will fail\./.test(
    document.querySelector("[data-footer]").textContent,
  ),
);
check(
  "…and skipping says what skipping costs",
  /Skip anyway — runs will fail at fetch/.test(document.querySelector("[data-skip]").textContent),
);

/*
 * The commands are copyable.
 *
 * These are `uv run hf download …` lines and absolute `.env` paths — sixty
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
 * missing and has no button, which is the assertion that matters most — the
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
// the line carries the installer's own last words — the only honest progress a
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
// whole checklist — one row turning green is not the same claim as the machine
// being one step readier.
await new Promise((resolve) => setTimeout(resolve, 1400));
check("the installed row turns Ready", /Ready/.test(rowOf("ffmpeg").textContent));
check("…and drops its Install button", !installButton("ffmpeg"));
check("…and the count comes down", /3 of 8 need attention/.test(root.textContent));
check("the other row can be installed again", installButton("sox").disabled === false);

/*
 * The state the footer's kind branch was written for, reachable at last.
 *
 * `ok` is the conjunction of the BLOCKING rows only — that is the server's
 * contract — and the fixture used to compute it as "every row passes", which is
 * stricter than the server and made this whole branch dead code: no fixture
 * state could ever have both `ok: true` and a failing row. With both tools
 * installed the machine IS ready, with a token still missing and a Demucs cache
 * still un-downloaded, and the screen has to say both halves of that.
 */
installButton("sox").dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
await new Promise((resolve) => setTimeout(resolve, 1400));
check(
  "with both tools in, the machine is ready — and does not claim all checks pass",
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
  /Everything required is ready\..*will still run — just worse\./.test(
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
/*
 * …in words, not in manifest tokens.
 *
 * `keep_reason` is a pipeline enum (`latin`, `speaker_en`, `tts_failed`, and —
 * after a headless re-run of a keep made right here — `user`), and it was
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
  rows().every((r) => /Dub|Keep|Fail|Render|Voice|Text/.test(r.textContent)),
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
  "pressing B stops A — one element, one clip",
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
 * `currentTime = 0`, which fights the `#t=start,end` fragment's own seek — so
 * pressing A on a line an hour in could play the top of the video — and nothing
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
  "the dub side plays the whole clip — it is not a window",
  clip(rowFor(9), "B").getAttribute("aria-pressed") === "true",
);
press(clip(rowFor(9), "B"));
await settle(120);

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

/*
 * What "failed" is.
 *
 * It used to be `tts.verify === "failed"` — a value `dubbing/tts.py` has never
 * written; its verdicts are ok / soft / keep. A real failure is a keep the
 * pipeline decided *against itself*: the voice could not say the line
 * (`tts_failed`) or the translator could not produce the target language
 * (`mt_failed`), and either way the segment is stored as a keep with a slice of
 * the original attached so the mix is never silent. Which the UI drew as a calm
 * green "Kept original" — the pipeline's own losses, reported as resolved, with
 * a Failed chip that could only ever count zero.
 */
clickIt(chip("Failed"));
await settle(200);
check("the Failed chip narrows the script", rows().length < 40 && rows().length > 0);
check("…and it finds the keeps the pipeline decided against itself", rows().length === 2);
check(
  "…which are keeps, and still say so",
  rows().every((r) => r.textContent.includes("original audio plays here")),
);
const bulk = [...document.querySelectorAll("button")].find((b) =>
  /Re-voice these \d+/.test(b.textContent),
);
check("a filtered set can be fixed in one job", bulk != null);
// The bulk set used to be `visible.filter(seg => !seg.keep)`, which is every
// row on this screen removed: both buttons offered a job for nobody. And the
// two buttons are not two ways to do one thing — an `mt_failed` line's
// `text_en` is the source line the translator copied in when it gave up, so
// re-voicing it would synthesize the wrong language.
check("…over the lines the chip actually found", /Re-voice these 1/.test(bulk.textContent));
check(
  "…and the one the translator lost is re-translated instead",
  [...document.querySelectorAll("button")].some((b) => /Re-translate these 1/.test(b.textContent)),
);
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
 * — invisible in a list of two hundred rows and unreachable by every other
 * chip. The fixture carries one of each unfinished shape (no translation, no
 * voice) and the chip has to find both and offer the one click that fixes them.
 */
clickIt(chip("Unfinished"));
await settle(200);
const stuck = rows();
// Three from the fixture — one stranded by a verdict flip, one never voiced,
// one voiced and never placed — plus the line whose translation was rewritten
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
 * reproduce a clip that already existed — and left it exactly as unplaced as it
 * found it, because only `timeline.place` can finish it.
 */
const unplaced = stuck.filter((r) => /Render/.test(r.textContent));
check("…the one that is synthesized and waiting for a render", unplaced.length === 1);
check(
  "…and the ones that only need a voice",
  stuck.filter((r) => r !== stranded && !unplaced.includes(r)).every((r) =>
    /Voice/.test(r.textContent),
  ),
);
check(
  "…fixable in one click, not two hundred",
  [...document.querySelectorAll("button")].some((b) =>
    /Translate & voice these 3/.test(b.textContent),
  ),
);
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
const dialogButton = (label) =>
  [...document.querySelectorAll('[role="dialog"] button')].find((b) => b.textContent === label);

clickIt(chip("Kept"));
await settle(200);
check("the Kept chip narrows to the lines that play as recorded", rows().length === 29);
check("…and offers to dub the lot in one gesture", /Dub these 29/.test(dubTrigger().textContent));
check(
  "…over the same count the chip is showing",
  /Kept\s*29/.test(chip("Kept").textContent),
);

const beforeBulk = calls().log.length;
clickIt(dubTrigger());
await settle(150);
check(
  "rewriting 29 verdicts asks first, and says what it costs",
  /29 lines switch to dubbed/.test(document.querySelector('[role="dialog"]').textContent) &&
    /translate \+ voice queue behind any running job/.test(root.textContent),
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
check(
  "…saying out loud that it is the search's set, not the run's",
  /not every kept line in the run/.test(bulkBar().textContent),
);

clickIt(dubTrigger());
await settle(150);
clickIt(dialogButton("Dub these 2"));
await settle(700);
check(
  "the bulk flip is one PATCH per line, then one translate and one voice for the lot",
  calls().log.slice(beforeBulk).join() === "patch,patch,retranslate,resynthesize",
);
check("…and the lines it flipped have left the kept filter", rows().length === 0);

setInput(searchBox, "");
await settle(200);
check("the rest of the run is still kept", rows().length === 27);
clickIt(chip("Kept"));
await settle(200);
check("the Kept chip toggles back off as well", rows().length > 40);
await settle(2400);
check("the two jobs it queued drain", document.querySelector("[data-job-strip]") == null);

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

/*
 * The run options that are still a decision, editable where the run is read.
 *
 * Genre, register and context were picked once on the import screen — in ten
 * seconds, before a single line had been read — and then became unreachable for
 * the life of the project. Nothing about them is structural: all three are
 * inputs to the translator, which is exactly why they can change and the source
 * and the language pair cannot.
 *
 * Everything asserted here is about the *request*, because the screen shows no
 * consequence of any of them until translation runs again — the same reason the
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
 * What the transport is on, said in the DOM.
 *
 * Three modes — the preview, the run's original audio, nothing — and the chip
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
 * Confirmation is themed and local, never `window.confirm` — which is drawn by
 * the OS, blocks the main thread and stops the playhead.
 */
click("Render preview");
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
/*
 * Cancelling is an ending, not a pause.
 *
 * The worker's journal saves whatever the job finished before it stopped, so a
 * cancelled run has real results on disk — and the editor only treated
 * done|failed as terminal, so the stage strip and every per-segment spinner
 * stayed up and the partial work was never re-read. The refetch is the claim
 * worth counting: it is invisible in the DOM (the segments come back looking
 * the same) and it is the whole difference.
 */
const readsBeforeCancel = calls().segments;
click("Cancel");
await settle(500);
check("…and it is the same job, cancellable from here", !/Rendering preview/.test(root.textContent));
check("a cancel is terminal — the partial work is read back", calls().segments > readsBeforeCancel);

/*
 * "Inherit" has to be able to undo an override.
 *
 * The per-segment language pair is the one field in the patch body where the
 * empty string means *clear* — `null` is "not supplied, leave it alone" for
 * everything (`app.py::PatchSegment`), and the select was sending `value ||
 * null`. So picking "inherit" sent a patch the server correctly ignored: an
 * override could be set and never taken off. And clearing it is a translate
 * invalidation like any other — the line was translated under the old pair.
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
 * Play before render.
 *
 * "i can start playing. but because it still not ready it shown nothing": for
 * the whole hour between `fetch` and `mix` the play button was live and there
 * was no preview.mp4 behind it. `source.wav` is on disk from the very first
 * stage, so that is what the transport plays until the preview exists — and
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
check("…space does nothing either — the key is the same control", clock() === stopped);
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
 * again. Two things fix it, and they are two halves of one question — what
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
   lay it against, because fetch never finished. */
const renderButton = [...document.querySelectorAll("header button")].find((b) =>
  b.textContent.includes("Render preview"),
);
check("Render is refused before the click, not by a job that dies", renderButton.disabled === true);
check(
  "…and says why",
  /fetch has not finished/.test(renderButton.getAttribute("title") ?? ""),
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
 * The import screen's switches only exist to be *sent*: the screen shows no
 * consequence of any of them, so the request body is the only place their
 * correctness is observable. "Dub foreign speech" is the one that had no way to
 * reach a run at all — the pipeline takes `--dub-foreign`, the server's
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
