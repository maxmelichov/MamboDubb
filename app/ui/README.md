# MamboDubb editor UI

The web front end for the dubbing pipeline: import a video, then review and correct the
dub segment by segment. React 19 + Vite 7 + Tailwind 4, pnpm, no state library and no
component library the same plainness as `MamboRambo/mamborambo-desktop`.

`docs/APP_ARCHITECTURE.md` is the contract. This app never imports the pipeline; it only
speaks HTTP to the studio server.

## Run it

```bash
pnpm install
pnpm dev:fixtures     # http://localhost:1430 sample data, no server needed
pnpm dev              # against a real studio server (proxied, see below)
pnpm build            # tsc + vite build the acceptance gate
pnpm smoke            # render the production bundle in jsdom and click through it
pnpm check            # both
```

`pnpm smoke` is the test the type checker cannot be: it boots the real bundle in jsdom
against the fixture, then asserts every screen renders, the theme toggle defaults to light and
persists a choice, the timeline draws its marks, the inspector's shelves start shut and
remember being opened, a keep toggle applies with no job, and a re-translate queues, reports
progress and writes its result back through the event stream.

`pnpm dev` proxies `/api`, `/media` and `/health` to `http://127.0.0.1:8756`. The server
binds an OS-assigned port and prints `{"status":"ready","port":N}` on stdout, so set
`VITE_SERVER_URL` to whatever it printed. See `.env.example`.

## Fixture mode

`VITE_USE_FIXTURES=1` swaps the backend inside `src/lib/api.ts` nothing above that file
changes. The fixture serves a snapshot of a real 320 s Hebrew→English run: 58 segments
with their actual translations, drift, placement rates and ASR verification transcripts,
regenerated with

```bash
python3 scripts/make_fixture.py outputs/<run> src/lib/fixture-data.json
```

It implements the contract, not a mock of it: the one-job rule, the NDJSON event stream
with stage/segment/job/log frames, optimistic no-model edits, and queued model actions
that actually mutate the segments when they finish. A/B playback works the clips are
synthesized buzz at the right durations, not silence so the transport is real even
though there is no `preview.mp4` to play.

The list has two more runs beside the snapshot, and opening either gives that run's own
state rather than the snapshot's: one stopped mid-pipeline, which is past `fetch` and so
plays its `source.wav` under the "Original audio" chip, and one that failed at `fetch`,
which has nothing to play and a dead play button that says why. Those are the editor's
three transport modes, and the only place they can all be seen at once.

## Editor screen

Three panes and a rail. The preview is the pane that **grows** it and the timeline are the
workspace, and everything else is sized around them. Under the timeline is the segment
**navigator**, a fixed strip: position, state, time, one line of text, and nothing else. It
is a `listbox`, one tab stop with a roving tabindex, and the line of text follows what will
actually play a kept segment shows its source, a dubbed one its translation.

The right rail is ordered by how often a reviewer touches a thing, which is not evenly:

| | on the surface |
|---|---|
| 1 | the original, then the translation as the one editable field |
| 2 | A/B the two versions |
| 3 | dub-or-keep, two named states with the consequence written under them |
| 4 | re-translate / re-voice this line, each with its cost on its face |

Everything else speaker and TTS overrides and how the last take verified; the per-segment
language tags; the transcript text, split/merge and the locks is on one of three named
shelves (`Disclosure`) that start **shut** and remember otherwise in `sessionStorage` for the
session. A shut shelf still shows a one-line summary of its current values, so "nothing
overridden here" costs no click, and the voice shelf's summary carries the verification
overlap so a bad take is visible without opening it. A clone that said the wrong words is the
one thing that jumps to the top of the rail regardless.

Chrome that is not permanent: the job strip renders only while a job runs or the event stream
is down, and the legend and keyboard shortcuts are behind the timeline's "?" popover both
are read on your first day and the day you forget, and neither ever changes. What stays in
the toolbar is a census of *this run*, which does.

## Setup screen

`/setup` renders `GET /api/setup` the server's fast filesystem checks (ffmpeg, sox, the
HF token, each model directory with its size, free disk). Every row states its verdict as a
glyph, a word (*Ready*, *Missing*, *Not installed*, *Incomplete*), and a hue, in that order:
the screen reads the same in monochrome. The detail sentence is the server's, and it is the
point of the row it says what to install, not merely that something is absent.

A provisioned machine reads **all green**, and the one grade that survives being absent with
nothing to do about it is `optional` sox, whose only caller is a tokenizer this pipeline
never loads and it is drawn as a grey dash and the words *Not installed*: no wash, no red,
not counted in the "N of M need attention" headline, and unable to hold `ok` back (the server
conjoins the required rows only).

The one amber thing on the list is not a grade at all. `state: "incomplete"` is a model
directory that is part way downloaded the first-run fetch writes the config and the shard
index in a second and the weights over the next several minutes and the row draws it with a
bar (`bytes` against `download_bytes`), a *Resume* button, and *Downloading* or *Incomplete*
depending on whether the server can still see a live fetch (`downloading`). While one says it
is live the screen re-runs the checklist on its own clock, so the row turns green without
anyone pressing Re-check.

On boot the app asks once. It routes to `/setup` **only** when the server answers `ok:
false`; an error, a missing endpoint or a server still starting says nothing, and nothing is
not a reason to interrupt. Otherwise the screen is a small link in the header. Fixture mode
never auto-routes, because its checklist deliberately fails several checks so the screen is
demoable and `/setup?ready=1` serves the opposite board, every row green, for the demo
that wants to show a machine that is set up (see `fixtures.setup`).

## Desktop shell

`src/lib/desktop.ts` is the only file that knows Tauri exists, and every call in it degrades
to the browser behaviour: `isDesktop()` is false, `pickVideoFile()` and `serverBaseUrl()`
resolve to null, `revealPath()` is a no-op. The Tauri API package is not a dependency the
module reads `window.__TAURI__` when the shell injects it and otherwise dynamic-imports a
specifier assembled at runtime, so the browser bundle never requires it.

Two things change inside the shell. *Choose file* opens a native dialog and fills in a real
absolute path (a browser `<input type=file>` can only ever report a name). And the webview's
origin is the Tauri asset protocol rather than the sidecar, so `initApiBase()` awaited in
`main.tsx` before the first render, because `mediaUrl` is called during render resolves
`http://127.0.0.1:<port>` once and every path in `api.ts` goes through it. In a browser that
prefix is `""` and the URLs stay relative, exactly as before.

## Design system

The look is MamboRambo's, refitted: a warm neutral ramp (the plane, the borders and the ink
all sit slightly yellow of neutral), 2xl-radius cards on a big soft shadow, and tiny
uppercase widely-tracked "eyebrow" labels doing all the section titling.

Two rules are worth knowing before you touch it.

**One accent, and it is teal.** Primary buttons, the active nav cell, the selected script row
and the focus ring are all `--color-accent`, the brand badge's teal: `#0f766e` on light,
`#2dd4bf` on dark. It is a fill, a ring and a rule, never text, and it is spent on a closed
list App.css keeps beside the token. It was ink in light for a while; a near-black button on
a near-white card is correct and colourless, and colourless is what the redesign answered.
Everything else stays neutral, which is what leaves the blue free to mean exactly one thing,
"dubbed": anything painted `--color-dubbed` is data, never chrome.

**Everything is a primitive.** `components/ui.tsx` holds Button, Card/CardSection, Panel,
Field, TextInput/TextArea/NumberInput/Select, Badge, StatePill, Progress, Kbd, Empty,
Disclosure, Popover, ErrorBlock/ErrorBar, Eyebrow/SectionLabel and Brand/LogoMark. There are
no ad-hoc styled containers above that file a new screen composes these or the system stops
being one. `Disclosure` and `Popover` are the two ways something leaves the screen without
leaving the app, and between them they are why the editor fits.

`src/App.css` is the whole token layer: surfaces, ink, elevation and the segment-state hues,
declared once in `@theme` and restated for dark under `:root.theme-dark`. Element resets live
inside `@layer base`; an unlayered `button { color: inherit }` beats every Tailwind `text-*`
utility no matter the specificity, which is a bug that has already been written here once.

**Two themes, and the OS does not get a vote.** Light is the default; dark is what you get by
asking. The choice is one class, `.theme-dark`, on `<html>`, backed by `localStorage` under
`dubbing-studio.theme` and `prefers-color-scheme` appears nowhere in the app, which the
smoke test asserts against the *built* bundle. Three places apply it and must agree: the
inline boot script in `index.html` (pre-paint, so there is no flash), `applyTheme()` in
`lib/theme.ts` called from `main.tsx` (the belt to that braces), and the header toggle. Dark
is not inverted light: the neutrals are re-picked for a near-black plane, elevation swaps
from a cast shadow to an inset top highlight, and the state hues are a separately validated
triple. The desktop shell's window `backgroundColor` has to match the *default* theme's plane,
`#f2f1ec`, and the smoke test reads `tauri.conf.json` to check that it does.

**One accent, and it is the brand's.** `--color-accent` is the badge's teal in both themes,
at the lightness each ground can hold: `#0f766e` on paper (5.47:1 under white text, so the
primary button's label is a real label) and `#2dd4bf` on near-black. It is spent on a closed
list: the primary button, the active cell of a nav or filter
group, the selected script row and timeline mark, the focus ring, and the brand chip. The
playhead, the transport and the stage track stay ink on purpose. App.css lists the whole set
next to the token.

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
    segments.ts     derived state what "dubbed"/"kept"/"failed" mean
    languages.ts    which languages can be heard, and which can be spoken
    theme.ts        the light/dark choice: one class, one key, no OS preference
  components/       ui.tsx, Timeline, ScriptPane/ScriptRow, SelectionPanel,
                    VideoPlayer, JobBar, and the editor's own pieces
                    (RunSummary, RunMenu, PreviewPlaceholder, SegmentShelves …)
  pages/            ImportPage, SetupPage, EditorPage, RunsPage
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

Four segment states, each with a hue **and** a glyph **and** a word colour never carries
meaning alone. The hues are validated all-pairs for colour-vision deficiency in both light
and dark (`dataviz` skill, `scripts/validate_palette.js`); light-mode "kept" sits below the
3:1 contrast gate (2.17:1 on the card surface), which is why the segment navigator repeats
every state as a word. The dark triple clears 3:1 on all three, and the word stays anyway —
one encoding, both themes. Unclaimed time gets no hue: it is an absence, so it is a hatch.

The restyle moved the surfaces *under* these fills and left the fills themselves untouched,
so the validated pair separations still hold. If you ever change one of the three hues,
re-run the validation rather than eyeballing it.
