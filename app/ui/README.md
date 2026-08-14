# Dubbing Studio — editor UI

The web front end for the dubbing pipeline: import a video, then review and correct the
dub segment by segment. React 19 + Vite 7 + Tailwind 4, pnpm, no state library and no
component library — the same plainness as `MamboRambo/mamborambo-desktop`.

`docs/APP_ARCHITECTURE.md` is the contract. This app never imports the pipeline; it only
speaks HTTP to the studio server.

## Run it

```bash
pnpm install
pnpm dev:fixtures     # http://localhost:1430 — sample data, no server needed
pnpm dev              # against a real studio server (proxied, see below)
pnpm build            # tsc + vite build — the acceptance gate
pnpm smoke            # render the production bundle in jsdom and click through it
pnpm check            # both
```

`pnpm smoke` is the test the type checker cannot be: it boots the real bundle in jsdom
against the fixture, then asserts both screens render, the timeline draws its marks, a keep
toggle applies with no job, and a re-translate queues, reports progress and writes its
result back through the event stream.

`pnpm dev` proxies `/api`, `/media` and `/health` to `http://127.0.0.1:8756`. The server
binds an OS-assigned port and prints `{"status":"ready","port":N}` on stdout, so set
`VITE_SERVER_URL` to whatever it printed. See `.env.example`.

## Fixture mode

`VITE_USE_FIXTURES=1` swaps the backend inside `src/lib/api.ts` — nothing above that file
changes. The fixture serves a snapshot of a real 320 s Hebrew→English run: 58 segments
with their actual translations, drift, placement rates and ASR verification transcripts,
regenerated with

```bash
python3 scripts/make_fixture.py outputs/<run> src/lib/fixture-data.json
```

It implements the contract, not a mock of it: the one-job rule, the NDJSON event stream
with stage/segment/job/log frames, optimistic no-model edits, and queued model actions
that actually mutate the segments when they finish. A/B playback works — the clips are
synthesized buzz at the right durations, not silence — so the transport is real even
though there is no `preview.mp4` to play.

## Setup screen

`/setup` renders `GET /api/setup` — the server's fast filesystem checks (ffmpeg, sox, the
HF token, each model directory with its size, free disk). Every row states its verdict as a
glyph, the word *Ready* or *Missing*, and a hue, in that order: the screen reads the same in
monochrome. The detail sentence is the server's, and it is the point of the row — it says
what to install, not merely that something is absent.

On boot the app asks once. It routes to `/setup` **only** when the server answers `ok:
false`; an error, a missing endpoint or a server still starting says nothing, and nothing is
not a reason to interrupt. Otherwise the screen is a small link in the header. Fixture mode
never auto-routes, because its checklist deliberately fails two checks so the screen is
demoable.

## Desktop shell

`src/lib/desktop.ts` is the only file that knows Tauri exists, and every call in it degrades
to the browser behaviour: `isDesktop()` is false, `pickVideoFile()` and `serverBaseUrl()`
resolve to null, `revealPath()` is a no-op. The Tauri API package is not a dependency — the
module reads `window.__TAURI__` when the shell injects it and otherwise dynamic-imports a
specifier assembled at runtime, so the browser bundle never requires it.

Two things change inside the shell. *Choose file* opens a native dialog and fills in a real
absolute path (a browser `<input type=file>` can only ever report a name). And the webview's
origin is the Tauri asset protocol rather than the sidecar, so `initApiBase()` — awaited in
`main.tsx` before the first render, because `mediaUrl` is called during render — resolves
`http://127.0.0.1:<port>` once and every path in `api.ts` goes through it. In a browser that
prefix is `""` and the URLs stay relative, exactly as before.

## Design system

The look is MamboRambo's, refitted: a warm neutral ramp (the plane, the borders and the ink
all sit slightly yellow of neutral), 2xl-radius cards on a big soft shadow, and tiny
uppercase widely-tracked "eyebrow" labels doing all the section titling.

Two rules are worth knowing before you touch it.

**Ink is the accent, not colour.** Primary buttons, the active nav pill, the selected table
row and the focus ring are all `primary` — near-black on light, near-white on dark. That is
the house style, and here it also buys correctness: it leaves the blue free to mean exactly
one thing, "dubbed". Anything painted `--color-brand` or `--color-dubbed` is data, never
chrome.

**Everything is a primitive.** `components/ui.tsx` holds Button, Card/CardSection, Panel,
Field, TextInput/TextArea/Select, Badge, StatePill, Progress, Kbd, ErrorBlock/ErrorBar,
Eyebrow/SectionLabel and Brand/LogoMark. There are no ad-hoc styled containers above that
file — a new screen composes these or the system stops being one.

`src/App.css` is the whole token layer: surfaces, ink, elevation and the segment-state hues,
declared once in `@theme` and restated for dark under `:root[data-theme="dark"]`. Element
resets live inside `@layer base`; an unlayered `button { color: inherit }` beats every
Tailwind `text-*` utility no matter the specificity, which is a bug that has already been
written here once.

## Layout

```
src/
  lib/
    types.ts        every wire shape, transcribed from the contract's API table
    api.ts          the only place the app touches the network
    fixtures.ts     the contract, implemented in memory
    ndjson.ts       partial-line-tolerant NDJSON reader with reconnect
    useProject.ts   the project store: one state object + actions
    useTransport.ts one clock, whether or not there is a video
    segments.ts     derived state — what "dubbed"/"kept"/"failed" mean
  components/       Timeline, SegmentList, SegmentInspector, ABPlayer, VideoPlayer …
  pages/            ImportPage, EditorPage
```

## Keyboard

| key | does |
|---|---|
| `space` | play / pause |
| `←` `→` | previous / next segment |
| `shift` + `←` `→` | nudge the playhead one second |
| `a` / `b` | play the original / the dubbed audio for the selection |
| `k` | toggle keep-original |
| `+` `-` | zoom the timeline |

## Colour

Four segment states, each with a hue **and** a glyph **and** a word — colour never carries
meaning alone. The hues are validated all-pairs for colour-vision deficiency in both light
and dark (`dataviz` skill, `scripts/validate_palette.js`); light-mode "kept" sits below the
3:1 contrast gate (2.17:1 on the card surface), which is why the segment table repeats every
state as text. Unclaimed time gets no hue: it is an absence, so it is a neutral hatch.

The restyle moved the surfaces *under* these fills and left the fills themselves untouched,
so the validated pair separations still hold. If you ever change one of the three hues,
re-run the validation rather than eyeballing it.
