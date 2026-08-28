/**
 * Fixture backend `VITE_USE_FIXTURES=1`.
 *
 * The studio server is being written in parallel against the same contract, so
 * the UI ships its own implementation of that contract: same shapes, same event
 * stream, same one-job rule. The data is a snapshot of a real run
 * (`scripts/make_fixture.py`), which matters 58 segments of real Hebrew with
 * real translations, real drift and real verification transcripts exercise
 * bidi text, long lines and low-overlap warnings that invented data would not.
 *
 * Everything here is in-memory: edits survive navigation, not a reload.
 */

import data from "./fixture-data.json";
import { ApiError } from "./apiError";
// Type-only, so the cycle (`api` imports this module) is erased at build: the
// install slot's response shape grew a queue block, and it is declared where
// the endpoint that sends it is.
import type { SetupInstallState } from "./api";
import { applyPatch } from "./patch";
import { placedSpan } from "./segments";
import { isPending } from "./types";
import { toneUrl } from "./tone";
import type {
  CreateProjectRequest,
  CreateProjectResponse,
  Health,
  Job,
  JobKind,
  NewSegment,
  Peaks,
  PeaksFile,
  ProjectDetail,
  ProjectOptionsPatch,
  ProjectSource,
  ProjectSummary,
  RenderState,
  Segment,
  SegmentPatch,
  SetupCheck,
  SetupInstall,
  SetupSeverity,
  SetupStatus,
  Stage,
  StudioEvent,
} from "./types";

const LATENCY_MS = 45;

type Store = { project: ProjectDetail; segments: Segment[] };

const store: Store = structuredClone(data) as unknown as Store;

/**
 * States the snapshot happens not to contain.
 *
 * `fixture-data.json` is a faithful dump of a run that finished, and a run that
 * finished has no per-segment TTS override, no language tag, and the reason
 * this matters most no line left unfinished. So two of the inspector's
 * shelves would demo as permanently empty and the script's Unfinished chip
 * would demo as permanently zero, which are exactly the states that hide a
 * bug. This adds one of each, by position rather than by content, and it is the
 * only place fixture mode invents anything.
 *
 * The rest are shapes a user was actually left in, and every one of them was a
 * state the UI got wrong: a kept line flipped to "Dub it" by a UI that sent
 * `PATCH {keep:false}` and queued nothing, so the server dropped the subtitle
 * and the clip and no job ever replaced them; a line that has its translation
 * and never got a voice; a line that has its voice and never got placed; and
 * the two keeps the pipeline decides against itself when the voice or the
 * translator fails. None of them is reachable except through a chip.
 */
function seedOverrides(): void {
  const withOpts = store.segments.find((seg) => !seg.keep && seg.tts);
  if (withOpts) withOpts.tts_opts = { seed: 4711, greedy: true };
  const withLang = store.segments.find((seg) => seg.src_lang);
  if (withLang) withLang.tgt_lang = "en";

  // From the back of the run, so the low ids the smoke test reads stay as the
  // snapshot wrote them.
  const stranded = [...store.segments].reverse().find((seg) => seg.keep);
  if (stranded) {
    Object.assign(stranded, {
      keep: false,
      keep_reason: null,
      locked: { keep: true },
      text_en: null,
      tts: null,
      place: null,
      verify: null,
    });
  }
  const voiceless = [...store.segments].reverse().find((seg) => !seg.keep && seg.place);
  if (voiceless) Object.assign(voiceless, { tts: null, place: null, verify: null });

  /*
   * A line that has its clip and no placement the ordinary state of a studio
   * segment between a re-voice and the render that lays it down. It is here
   * because it is the state that was *mislabelled*: with no distinct state for
   * it the row said "Needs voice", and the Unfinished chip's one-click fix
   * queued a minute of synthesis to reproduce a clip that already existed.
   */
  const unplaced = [...store.segments]
    .reverse()
    .find((seg) => !seg.keep && seg.place && seg.tts && seg !== voiceless);
  if (unplaced) Object.assign(unplaced, { place: null });

  /*
   * And a real failure, in the shape the pipeline actually writes.
   *
   * The snapshot carried `tts.verify = "failed"`, which `dubbing/tts.py` never
   * writes the record's verdict is "ok", "soft" or "keep" and nothing else.
   * That invented value was the *only* thing the Failed chip counted, so the UI
   * had a failure state that could only be reached by a fixture. What a failed
   * synthesis really leaves behind is a keep the pipeline decided against
   * itself: `keep_reason="tts_failed"`, with a slice of the original audio
   * attached (`Engine.keep_clip`) so the mix is never silent.
   */
  const failed = store.segments.find(
    (seg) => seg.keep_reason === "tts_failed" || seg.tts?.verify === "failed",
  );
  if (failed) Object.assign(failed, keptByFailure(failed, "tts_failed"));

  // Its sibling, and the other half of the Failed filter's offer: a line the
  // *translator* could not put into the target language (`translate.run` →
  // `keep_reason="mt_failed"`, with the source line copied into `text_en`).
  // Its fix is the re-translate button, not the re-voice one.
  const untranslatable = [...store.segments]
    .reverse()
    .find((seg) => !seg.keep && seg.place && seg !== unplaced && seg !== voiceless);
  if (untranslatable) {
    Object.assign(untranslatable, keptByFailure(untranslatable, "mt_failed"), {
      text_en: untranslatable.text,
    });
  }

  /*
   * And the state the server has a rule for and no fixture ever produced: a
   * dub-wanted line whose *mix* plays the original audio. `tts` falls back to
   * the keep slice rather than leaving the span silent and `timeline` fits it
   * to the span (`fit_keep_`), so the verdict says dub, the file says original,
   * and only `media.fallback` tells them apart. From the back, past the three
   * shapes above, which no longer match: they are keeps or have no placement.
   */
  const fellBack = [...store.segments]
    .reverse()
    .find((seg) => !seg.keep && seg.place && seg.tts);
  if (fellBack?.place && fellBack.tts) {
    const clip = `clips/fit_keep_${fellBack.uid}.wav`;
    Object.assign(fellBack, {
      tts: { ...fellBack.tts, clip },
      place: { ...fellBack.place, clip },
    });
  }
}

/**
 * The record a pipeline failure leaves behind: a keep it decided against
 * itself, with a slice of the original audio attached so the mix is never
 * silent (`tts.Engine.keep_clip`). `verify` stays as it was the low overlap
 * from the attempt that failed is the evidence for the verdict.
 */
function keptByFailure(seg: Segment, reason: "tts_failed" | "mt_failed"): Partial<Segment> {
  const clip = `clips/${seg.uid}_keep.wav`;
  return {
    keep: true,
    keep_reason: reason,
    tts: { clip, dur: Number((seg.end - seg.start).toFixed(2)), tries: 3, overlap: 1,
           verify: "keep" },
    place: { start: seg.start, end: seg.end, rate: 1, drift: 0, clip },
  };
}
seedOverrides();

const jobs: Job[] = [];
const listeners = new Set<(event: StudioEvent) => void>();
let jobSeq = 0;
let running = false;

/**
 * How many round trips the editor has made, by kind.
 *
 * Fixture mode is the only mode the smoke test runs in and it never touches
 * `fetch`, so "did that interaction hit the server" has to be counted here or
 * it cannot be asserted at all. It exists for one claim in particular that
 * closing an editor without changing the text saves nothing which is
 * invisible to a DOM assertion because a no-op PATCH looks exactly like no
 * PATCH from the outside.
 *
 * `log` is the same information with the *order* kept, which is what a flip to
 * "Dub it" has to be judged on: the PATCH clears the line and the clip, so the
 * translate job must be enqueued after it and the voice job after that. A pair
 * of counters cannot tell that apart from the same two calls in the wrong
 * order. Only the calls that queue work are logged a background refetch of
 * the segment list is not part of any sequence worth asserting.
 */
export const calls = {
  patch: 0,
  segments: 0,
  peaks: 0,
  retranslate: 0,
  resynthesize: 0,
  // Setup's one-at-a-time slot is only observable as a count: a second click
  // while an install runs must produce no second install, and a disabled button
  // that still fires would look identical from the DOM.
  install: 0,
  // Same argument for the one-button queue: "pressing it again while it runs
  // starts nothing new" is a claim about requests, and the DOM cannot see it.
  installAll: 0,
  log: [] as string[],
  /**
   * Every body the import screen has sent, kept whole.
   *
   * A counter cannot answer the question this one is asked: whether a switch on
   * that screen reached the request. The options rail is a pile of fields that
   * only exist to be *sent* the screen shows no consequence of any of them —
   * so the sent body is the only place their correctness is observable.
   */
  created: [] as CreateProjectRequest[],
  /**
   * Every run-option patch the editor has sent. Same argument as `created`: the
   * three options live on the server, the screen re-reads them, and the only way
   * to tell "the control saved" from "the control redrew itself" is the body.
   */
  updated: [] as ProjectOptionsPatch[],
};
(globalThis as { __DUBBING_FIXTURE_CALLS__?: typeof calls }).__DUBBING_FIXTURE_CALLS__ = calls;

// --- enrichment -----------------------------------------------------------

/** Speaker label → a stable pitch, so A/B tells two voices apart by ear. */
function speakerHz(speaker: string, offset: number): number {
  let hash = 0;
  for (let i = 0; i < speaker.length; i += 1) hash = (hash * 31 + speaker.charCodeAt(i)) % 997;
  return 110 + (hash % 9) * 14 + offset;
}

/**
 * The run's source track, as one file per voice.
 *
 * `source.wav` is a single file on the server and the fixture cannot be quite
 * that faithful and stay useful: pitch is what tells two speakers apart by ear
 * in demo mode, and that has to survive on the side the comparison starts from.
 * What matters is that it is a *track* long enough to hold every window cut
 * from it, and that it is one blob rather than one per line.
 */
function sourceTrackUrl(speaker: string): string {
  const duration = Math.max(
    store.project.source.duration ?? 0,
    ...store.segments.map((seg) => seg.end),
    1,
  );
  return toneUrl({ hz: speakerHz(speaker, 0), dur: duration, seed: seedOf(speaker) });
}

function seedOf(uid: string): number {
  let hash = 7;
  for (let i = 0; i < uid.length; i += 1) hash = (hash * 33 + uid.charCodeAt(i)) % 100003;
  return hash;
}

/**
 * Add the fields `GET /segments` promises and the manifest does not store.
 *
 * The shape is `Projects.enrich`'s, key for key `media.play`, `media.tts`,
 * `media.source`, `media.source_window`. It has to be: these fixtures are the
 * only implementation of the contract that the smoke test ever exercises, so a
 * fixture that invents its own field names is not a stand-in for the server,
 * it is a second, divergent server that hides the first one's bugs. That is
 * literally how A/B playback shipped broken.
 */
function enrich(seg: Segment): Segment {
  const seed = seedOf(seg.uid);
  const sourceDur = Math.max(0.4, seg.end - seg.start);
  const placed = (seg.place?.clip ?? "").split("/").pop() ?? "";
  return {
    ...seg,
    media: {
      play: seg.place?.clip
        ? toneUrl({
            hz: speakerHz(seg.speaker, 26),
            dur: Math.max(0.4, seg.place.end - seg.place.start),
            seed: seed + 1,
          })
        : null,
      tts: seg.tts?.clip
        ? toneUrl({ hz: speakerHz(seg.speaker, 26), dur: seg.tts.dur || sourceDur, seed: seed + 1 })
        : null,
      /*
       * One file, many windows exactly the shape `Projects.enrich` sends:
       * `media.source` is the whole source track with a `#t=start,end` fragment
       * naming this line's place in it, and `source_window` states the same two
       * numbers outright.
       *
       * It used to be a tone as long as the *segment* under a window in
       * whole-track coordinates, which is two coordinate systems in one pair of
       * fields: `toggleClip` seeks to 219s in a 2.4s clip, playback ends on the
       * spot, and every Orig button in the demo was dead while the live server's
       * were fine. The fragment keeps the URL per-segment, which is what the row
       * compares to decide whether *this* line's A is the one sounding.
       */
      source: `${sourceTrackUrl(seg.speaker)}#t=${seg.start.toFixed(3)},${seg.end.toFixed(3)}`,
      source_window: [seg.start, seg.end],
      // Server rule verbatim: a dub-wanted line wearing a keep-slice is a
      // fallback, not a dub (dubbing_app/projects.py). Both names, because the
      // server reads both: `fit_keep_` is the same slice after `timeline`
      // rate-fitted it, and a fitted original is still not a dub.
      fallback: !seg.keep && (placed.startsWith("keep_") || placed.startsWith("fit_keep_")),
    },
  };
}

const delay = <T>(value: T): Promise<T> =>
  new Promise((resolve) => setTimeout(() => resolve(value), LATENCY_MS));

/**
 * Every failure here is an `ApiError`, never a bare `Error`.
 *
 * The server speaks one error envelope `{"error":{code,message}}` and the
 * app branches on it: a 404 from `peaks` is "no dub.wav yet" and resolves to
 * null, a 409 is the one-job rule refusing an edit. A fixture that rejects with
 * a plain `Error` has no code and no status, so every one of those branches is
 * untested in the only mode the smoke test runs in, and one of them (`peaks`)
 * was wrong in exactly that way.
 */
function find(uid: string): Segment {
  const seg = store.segments.find((s) => s.uid === uid);
  if (!seg) throw new ApiError("not_found", `no segment '${uid}'`, 404);
  return seg;
}

function emit(event: StudioEvent): void {
  for (const listener of listeners) listener(event);
}

// --- reads ----------------------------------------------------------------

export function health(): Promise<Health> {
  return delay({ status: "ok", version: "0.1.0-fixtures" } as Health);
}

/**
 * `GET /api/setup`, deliberately mixed: a fresh machine with neither
 * command-line tool and the models in every state the screen has
 * to draw. A checklist where everything passes demos nothing, and the failure
 * rows are where the copy has to earn its keep each one says what to do, not
 * just that it is missing.
 *
 * The tool rows are one `brew install` away, so the app offers to run it. The
 * model rows cover the whole surface on purpose: one already on disk (a size,
 * no button), one downloadable and idle (the "Download · ~9.7 GB" price tag),
 * one *mid-download when the page mounts* (seeded below the state a user who
 * reloaded during an 81 GB fetch comes back to, and the one that proves the
 * page picks a running install back up), and one no button can fix (Demucs
 * fetches its own cache; the detail line's command is the whole answer).
 *
 * ## `?ready=1` the provisioned machine
 *
 * The mixed board above is the *default* and stays the default: the smoke test
 * drives the install buttons, the progress bar and every failure sentence
 * through it, and a fixture where nothing is wrong would test none of that.
 *
 * But the other screenshot matters too the machine that is set up, which is
 * what a demo of the finished product shows and it cannot be reached by
 * clicking, because two of these rows have no button. So any URL carrying
 * `?ready=1` (`/setup?ready=1`) serves the all-green board instead: every tool
 * found, every model on disk, nothing downloading. It is read
 * from the live URL on each call rather than captured once, so flipping it on
 * or off is a navigation and not a reload.
 */
function readyDemo(): boolean {
  try {
    return new URLSearchParams(window.location.search).get("ready") === "1";
  } catch {
    // No `window` (a test harness), no query string, no demo mode.
    return false;
  }
}

export function setup(): Promise<SetupStatus> {
  if (readyDemo()) return delay({ ok: true, checks: readyChecks() });
  seedDownload();
  const checks = setupChecks();
  // `ok` is the conjunction of the BLOCKING checks only, exactly as
  // `setup.report` computes it. It used to be `every(c => c.ok)`, which made the
  // fixture stricter than the server: a machine with an un-downloaded Demucs
  // cache is ready to run, and this said it was not so
  // the whole "everything required is ready, N optional things are missing"
  // branch of the footer could not be reached in the only mode that is tested.
  return delay({
    ok: checks.filter((c) => c.severity === "blocking").every((c) => c.ok),
    checks,
  });
}

/** Which tools this fake machine has had installed during the session. */
const installed = new Set<string>();

/**
 * The workspace `.env` this fake machine reads, for any row whose detail names
 * a file path.
 *
 * There is no `hf_token` row here any more, and the fixture is where that is
 * easiest to see: a demo that painted a credential row for a token nothing
 * needs would be demoing a wall the app removed. The diarization weights ship
 * with the app, so the checklist below has no token in it and no
 * `saveHfToken`/`clearHfToken` fakes to go with one.
 */
const FIXTURE_ENV = "/Users/you/DubbingQwen/.env";

/**
 * The low-VRAM row, and the fake of `POST /api/setup/low_vram`.
 *
 * Modelled on a machine that has enough memory and has not been told
 * otherwise, because that is the state the switch is interesting in: `auto`
 * and off, so the demo can turn it on and watch the row say what it now costs.
 * The row is always `ok` and always `optional` — it is a setting, and a setting
 * is never a missing thing. Turning it on here also flips `source` from "auto"
 * to "env_file", which is the real server's answer once the value is written to
 * a file, and the difference the row's sentence exists to show.
 */
let lowVramOn: boolean | null = null;      // null: nobody has said, so auto decides

function lowVramRow(): SetupCheck {
  const enabled = lowVramOn ?? false;
  const weights = enabled
    ? "the mxfp4 MLX build, about 6.4 GB, and translates less well"
    : "the 6-bit MLX build, 9.7 GB";
  const where = lowVramOn === null
    ? " (the default weights fit)"
    : `, set in \`${FIXTURE_ENV}\``;
  return {
    id: "low_vram",
    label: "Low-VRAM translator",
    ok: true,
    severity: "optional",
    enabled,
    source: lowVramOn === null ? "auto" : "env_file",
    path: FIXTURE_ENV,
    detail: `${enabled ? "on" : "off"}: the translator loads ${weights}${where}`,
  };
}

export function setLowVram(enabled: boolean): Promise<SetupCheck> {
  lowVramOn = enabled;
  return delay(lowVramRow());
}

/**
 * The two command-line tools, graded the way `setup.TOOLS` grades them.
 *
 * ffmpeg blocks every stage shells out to it. SoX does not, and saying it
 * did was the fixture's own invention: the only sox caller in the tree is
 * qwen_tts's 25Hz tokenizer and this pipeline loads 12Hz checkpoints, which is
 * why the server calls the row `optional` and attaches no stage to it. A demo
 * that paints a red REQUIRED row for a tool nothing runs is a demo of a machine
 * that is broken when it is not.
 */
const TOOL_ROWS: Record<
  string,
  { label: string; here: string; missing: string; severity: SetupSeverity; stage?: Stage }
> = {
  ffmpeg: {
    label: "ffmpeg",
    severity: "blocking",
    stage: "fetch",
    here: "7.1.1 at /opt/homebrew/bin/ffmpeg",
    missing:
      "ffmpeg not on PATH: every stage shells out to it for audio and video. " +
      "Install it with `brew install ffmpeg`.",
  },
  sox: {
    label: "SoX",
    severity: "optional",
    here: "14.4.2 at /opt/homebrew/bin/sox",
    missing:
      "sox not on PATH: nothing the shipped pipeline runs needs it. Only " +
      "qwen_tts's 25Hz tokenizer would, and this pipeline loads 12Hz checkpoints. " +
      "Install it with `brew install sox`.",
  },
};

/** `ready` is the provisioned-machine demo: every tool found, whatever this
    session installed. See `setup()` for the flag that turns it on. */
function toolRow(id: string, ready = false): SetupCheck {
  const row = TOOL_ROWS[id];
  const ok = ready || installed.has(id);
  return {
    id,
    label: row.label,
    ok,
    installable: true,
    severity: row.severity,
    required: row.severity === "blocking",
    // Only ever on a blocking row, exactly as the server attaches it.
    ...(row.severity === "blocking" && row.stage ? { stage: row.stage } : {}),
    detail: ok ? row.here : row.missing,
  };
}

/**
 * The downloadable models, mirroring `setup.model_downloads()`: the id maps to
 * a hub repo, a directory and an approximate size, and `installable: true` is
 * what puts the Download button on the row. Two of them, in the two states the
 * button has: idle with its price tag, and mid-download (see `seedDownload`).
 *
 * `severity` defaults to blocking, and one row is here precisely because it is
 * not. The screen has three grades and the fixture used to reach the middle one
 * only through the HF-token row, which does not exist any more (diarization
 * needs no credential), so language ID carries `degrades` here as it does
 * on the server. Diarization is not in this table at all: its weights ship with
 * the app, so it has no hub id and no size, and the button its row can have is a
 * restore rather than a download. See `setupChecks`.
 */
const MODEL_ROWS: Record<
  string,
  {
    label: string;
    stage?: Stage;
    hub: string;
    dir: string;
    bytes: number;
    here: string;
    severity?: SetupSeverity;
    missing?: string;
  }
> = {
  "model.translate": {
    label: "Translation model (Gemma 3 12B)",
    stage: "translate",
    hub: "mlx-community/gemma-3-12b-it-qat-4bit",
    dir: "gemma-3-12b-it-qat-4bit",
    bytes: 9_700_000_000,
    here: "9.0 GB in models/gemma-3-12b-it-qat-4bit",
  },
  "model.tts.1.7b": {
    label: "TTS checkpoint 1.7b",
    stage: "tts",
    hub: "mlx-community/Qwen3-TTS-1.7B-4bit",
    dir: "qwen3-tts-1.7b",
    bytes: 4_500_000_000,
    here: "4.2 GB in models/qwen3-tts-1.7b",
  },
  // The third blocking download the server offers (`setup.model_checks`), and
  // the smallest — which is exactly why it is here: a queue of one item cannot
  // show "2 of 3", and a board where every download costs gigabytes cannot show
  // that the one-button install works through them in order and at different
  // speeds. Without it verification silently stops happening.
  "model.asr.en": {
    label: "Target ASR: English (clip verification)",
    stage: "tts",
    hub: "Systran/faster-whisper-base.en",
    dir: "faster-whisper-base.en",
    bytes: 147_000_000,
    here: "141 MB in models/faster-whisper-base.en",
  },
  // The sizes on these two are the server's own, measured off a complete
  // install rather than rounded to something memorable. The rounded ones used
  // to live here too, and a rounded one is what failed a working model: the
  // server's size floor compared a real 82.4 MB directory against a written
  // down 100 MB and called it a partial download.
  "model.lid": {
    label: "Language ID (VoxLingua107)",
    severity: "degrades",
    hub: "speechbrain/lang-id-voxlingua107-ecapa",
    dir: "lang-id-voxlingua107-ecapa",
    bytes: 86_400_000,
    here: "82.4 MB in models/lang-id-voxlingua107-ecapa",
    missing: "without it foreign-speech detection is skipped",
  },
};

/**
 * The row that is neither there nor absent, and the state the fixture board was
 * missing entirely.
 *
 * A first-run fetch writes the config and the shard index in the first second
 * and the weights over the next several minutes, and the server used to call
 * that directory READY at 1%, certifying a model that cannot
 * load. It reports `incomplete` now, and the demo needs one so the amber row,
 * the bar drawn off `bytes`/`download_bytes` and the Resume button are on screen
 * without anyone having to interrupt a download to see them. The translator is
 * the row for it: the biggest download on the board, so it is the one where
 * "start over" versus "resume" is worth gigabytes.
 */
const PARTIAL_ID = "model.translate";
const PARTIAL_SHARE = 0.32;

/** How much of `id` is already on disk before anything in this session ran. */
function partialBytes(id: string): number {
  return id === PARTIAL_ID ? Math.round(MODEL_ROWS[id].bytes * PARTIAL_SHARE) : 0;
}

/** One downloadable model row, in whichever state this fake session has it. */
function modelRow(id: string, ready = false): SetupCheck {
  const row = MODEL_ROWS[id];
  const ok = ready || installed.has(id);
  const approx = fixtureBytes(row.bytes);
  const severity = row.severity ?? "blocking";
  const partial = !ok && partialBytes(id) > 0;
  if (partial) {
    const done = partialBytes(id);
    return {
      id,
      label: row.label,
      ok: false,
      state: "incomplete",
      installable: true,
      severity,
      required: severity === "blocking",
      ...(severity === "blocking" && row.stage ? { stage: row.stage } : {}),
      hub: row.hub,
      download_bytes: row.bytes,
      bytes: done,
      // Stalled rather than live: a demo board that polled itself forever would
      // never sit still, and "press Download to finish" is the half of this
      // state that has a gesture attached to it.
      detail:
        `incomplete: ${fixtureBytes(done)} in models/${row.dir}: partial download, ` +
        `press Download to finish. It resumes: ` +
        `\`uv run hf download ${row.hub} --local-dir models/${row.dir}\``,
    };
  }
  return {
    id,
    label: row.label,
    ok,
    installable: true,
    severity,
    required: severity === "blocking",
    // Only ever on a blocking row, exactly as the server attaches it.
    ...(severity === "blocking" && row.stage ? { stage: row.stage } : {}),
    hub: row.hub,
    download_bytes: row.bytes,
    detail: ok
      ? row.here
      : `missing: models/${row.dir} (${row.missing ?? `downloads from ${row.hub} on first use`}). ` +
        `Fetch it (~${approx}): \`uv run hf download ${row.hub} --local-dir models/${row.dir}\``,
  };
}

/** `4500000000` → `4.5 GB`, the server's `human_bytes` rounding. */
function fixtureBytes(n: number): string {
  for (const unit of ["B", "KB", "MB", "GB"] as const) {
    if (n < 1024) return unit === "B" || unit === "KB" ? `${Math.round(n)} ${unit}` : `${n.toFixed(1)} ${unit}`;
    n /= 1024;
  }
  return `${n.toFixed(1)} TB`;
}

/**
 * The stem-separation cache, and the row that used to have no gesture at all.
 *
 * It is `optional` and it downloads itself on the first stems run, and that was
 * read as "there is nothing to do here": no button, no place in the queue, and a
 * detail line whose only offer was to start a dub and sit through a silent
 * mid-run fetch. The server installs it now (`install.DEMUCS_ID`), through a
 * route that is not a hub snapshot, which is why this row still carries no `hub`
 * and no size: the button says "Install", not "Download · ~320 MB", because the
 * bytes are demucs's business and a price tag here would be invented.
 */
const DEMUCS_ID = "model.demucs";

function demucsRow(ready = false): SetupCheck {
  const ok = ready || installed.has(DEMUCS_ID);
  return {
    id: DEMUCS_ID,
    label: "Stem separation: Demucs htdemucs",
    ok,
    severity: "optional",
    installable: true,
    detail: ok
      ? "htdemucs_ft cache: 320 MB in ~/.cache/torch/hub"
      : "htdemucs_ft not downloaded yet: it is fetched on the first stems run. " +
        "The Install button installs it by fetching the htdemucs_ft weights now, " +
        "into the same cache the first stems run would fill anyway",
  };
}

function setupChecks(ready = false): SetupCheck[] {
  const rows: SetupCheck[] = [
    toolRow("ffmpeg", ready),
    toolRow("sox", ready),
    modelRow("model.translate", ready),
    modelRow("model.tts.1.7b", ready),
    modelRow("model.asr.en", ready),
    modelRow("model.lid", ready),
    // Green before the user has done anything, and that is the demo. The
    // CC-BY-4.0 weights ride in the app bundle (`segments.DIARIZATION_DIR`), so
    // there is no hub id and no size on this row: a fresh machine already tells
    // speakers apart. It stays on the list because a checklist that silently
    // omits a model reads as an omission, not a fact. `degrades` rather than
    // `optional`, as the server grades it: red here means every character in
    // the video dubbed in one voice, and the app can now put the weights back
    // itself (`install.restore_diarization`) with no account and no network.
    {
      id: "model.diarization",
      label: "Speaker diarization (pyannote community-1)",
      ok: true,
      severity: "degrades",
      detail: "31 MB in third_party/pyannote-speaker-diarization-community-1 (ships with the app)",
    },
    demucsRow(ready),
    {
      id: "disk",
      label: "Free disk space",
      ok: true,
      severity: "optional",
      detail: "184 GB free: a 20-minute run writes about 4 GB under outputs/",
    },
    // Last, as the server appends it, and the same row on both boards: a
    // machine being "ready" has nothing to do with which translator weights it
    // chose, and `?ready=1` flipping this one would demo the wrong thing.
    lowVramRow(),
  ];
  // `required` is the server's derived view of `severity`, and the fixture has to
  // derive it the same way a row that disagreed with itself would let the
  // screen read one field on the server and the other here.
  return rows.map((row) => ({
    installable: false,
    ...row,
    required: row.severity === "blocking",
  }));
}

/** The provisioned machine every row green. See `setup()` for `?ready=1`. */
function readyChecks(): SetupCheck[] {
  return setupChecks(true);
}

/**
 * `POST|GET /api/setup/install`, compressed.
 *
 * The real thing is a `brew install` and minutes of output; this is the same
 * contract at a speed a smoke test can watch a handful of lines, a few poll
 * cycles, then the check flips to Ready. It has to refuse the same two things
 * the server refuses, or the button's disabled state and the page's error path
 * are untested in the only mode that ever runs.
 */
const INSTALL_TICK_MS = 120;

let installState: SetupInstall = {
  running: false,
  id: null,
  ok: null,
  error: null,
  tail: [],
  check: null,
};

export function startInstall(id: string): Promise<SetupInstallState> {
  calls.install += 1;
  calls.log.push(`install:${id}`);
  // A row's own button ends the last queue's story, exactly as the server's
  // `POST /api/setup/install` does before it claims the slot.
  if (!queueState?.running) queueState = null;
  if (installState.running) {
    return Promise.reject(
      new ApiError("busy", `an install is already running (${installState.id}); one at a time`, 409),
    );
  }
  if (id in MODEL_ROWS) {
    void startDownload(id);
    return delay(installBody());
  }
  if (id === DEMUCS_ID) {
    void startDemucs();
    return delay(installBody());
  }
  if (!(id in TOOL_ROWS)) {
    return Promise.reject(
      new ApiError(
        "invalid_request",
        `'${id}' cannot be installed from the app. The only installs it runs are ` +
          "`brew install ffmpeg`, `brew install sox` and the hub-snapshot model " +
          "downloads. Everything else is by hand: the command is in that " +
          "check's detail line.",
        400,
      ),
    );
  }
  void startTool(id);
  return delay(installBody());
}

export function installStatus(): Promise<SetupInstallState> {
  // Not in the provisioned-machine demo: the seeded mid-flight download is the
  // one thing that would still be moving on a screen whose whole claim is that
  // there is nothing left to do.
  if (!readyDemo()) seedDownload();
  return delay(installBody());
}

/**
 * `POST|DELETE /api/setup/install_all` — the one button, faked end to end.
 *
 * The server computes the list from a fresh report (`setup.install_plan`) and
 * runs it through the single slot; this does the same over the fixture board,
 * so the queue the smoke test drives advances deterministically: the same
 * ordering rule (blocking before degrades, nothing optional), the same one
 * install in flight at a time, the same cancel boundary — the item running
 * finishes, nothing after it starts.
 */
type FixtureQueue = {
  items: { id: string; label: string; bytes: number }[];
  pos: number;
  running: boolean;
  cancelled: boolean;
  failed: string[];
  started: number;
  finished: number | null;
};

let queueState: FixtureQueue | null = null;

/**
 * The fixture's `setup.install_plan`: everything missing that the app can fix,
 * in grade order.
 *
 * Optional rows are in it, as they are on the server. Filtering them out was how
 * a machine whose only red rows were optional got a button that installed
 * nothing and reported the previous install's success.
 */
function installPlan(): FixtureQueue["items"] {
  const rank: Record<SetupSeverity, number> = { blocking: 0, degrades: 1, optional: 2 };
  return setupChecks()
    .filter((row) => !row.ok && row.installable)
    .sort((a, b) => rank[a.severity ?? "blocking"] - rank[b.severity ?? "blocking"])
    .map((row) => ({ id: row.id, label: row.label, bytes: row.download_bytes ?? 0 }));
}

export function startInstallAll(): Promise<SetupInstallState> {
  calls.installAll += 1;
  calls.log.push("install_all");
  // Idempotent, like the server: pressing it again while it runs answers the
  // running queue. A 409 here would teach the user that the button they just
  // pressed did something wrong.
  if (queueState?.running) return delay(installBody());
  if (installState.running) {
    return Promise.reject(
      new ApiError("busy", "an install is already running; wait for it to finish, " +
        "then install the rest in one go", 409),
    );
  }
  // The slot's leftovers are the last gesture's story, and the server drops them
  // here too (`Installer.reset`): without that, "install everything" on a board
  // with nothing missing answered with the previous install's `ok: true`, which
  // is how "nothing happened" came to look like "something succeeded".
  installState = { running: false, id: null, ok: null, error: null, tail: [], check: null };
  const items = installPlan();
  queueState = {
    items,
    pos: 0,
    running: items.length > 0,
    cancelled: false,
    failed: [],
    started: Date.now() / 1000,
    finished: items.length ? null : Date.now() / 1000,
  };
  if (queueState.running) void runQueue();
  return delay(installBody());
}

export function cancelInstallAll(): Promise<SetupInstallState> {
  if (queueState?.running) queueState = { ...queueState, cancelled: true };
  return delay(installBody());
}

async function runQueue(): Promise<void> {
  for (;;) {
    const queue = queueState;
    if (!queue || queue.cancelled || queue.pos >= queue.items.length) break;
    const { id } = queue.items[queue.pos];
    await (id in MODEL_ROWS
      ? startDownload(id)
      : id === DEMUCS_ID
        ? startDemucs()
        : startTool(id));
    if (queueState === null) return;                 // a row's button took over
    queueState = {
      ...queueState,
      pos: queueState.pos + 1,
      failed: installState.ok ? queueState.failed : [...queueState.failed, id],
    };
  }
  if (queueState) queueState = { ...queueState, running: false, finished: Date.now() / 1000 };
}

/** The slot's status with the queue block beside it — one body, as the server
    sends it, so the header and the row read the same poll. */
function installBody(): SetupInstallState {
  const body: SetupInstallState = structuredClone(installState);
  if (!queueState) return body;
  const done = queueState.running ? (installState.bytes_done ?? 0) : 0;
  const left = queueState.items.slice(queueState.pos).reduce((sum, item) => sum + item.bytes, 0);
  body.queue = {
    running: queueState.running,
    cancelled: queueState.cancelled,
    items: structuredClone(queueState.items),
    pos: queueState.pos,
    total: queueState.items.length,
    failed: [...queueState.failed],
    remaining_bytes: Math.max(0, left - done),
    started: queueState.started,
    finished: queueState.finished,
  };
  return body;
}

/**
 * A model download, at fixture speed.
 *
 * Same contract as the server's: the slot goes `running` with `bytes_done`
 * against `bytes_total`, the poll watches the number climb (the server
 * re-walks the directory; this ticks a counter), and the finish re-probes the
 * row. Compressed to a few seconds so the smoke test and the demo can watch
 * the bar actually move and then the row turn Ready.
 */
function startDownload(id: string, from = partialBytes(id)): Promise<void> {
  const row = MODEL_ROWS[id];
  installState = {
    running: true,
    id,
    ok: null,
    error: null,
    tail: [
      `$ snapshot_download('${row.hub}', local_dir='models/${row.dir}')`,
      `downloading ~${fixtureBytes(row.bytes)}; partial files are kept, so an ` +
        "interrupted download resumes where it stopped",
    ],
    check: null,
    bytes_done: from,
    bytes_total: row.bytes,
  };
  // Returned rather than fired and forgotten, so the queue can wait for one
  // item before claiming the slot for the next — the server's one-at-a-time
  // rule, which the queue is the only caller that has to honour by hand.
  return runDownload(id);
}

/** The tool half of the same thing: claim the slot, hand back the promise. */
function startTool(id: string): Promise<void> {
  installState = {
    running: true,
    id,
    ok: null,
    error: null,
    tail: [`$ brew install ${id}`],
    check: null,
  };
  return runInstall(id, [
    `==> Fetching ${id}`,
    `==> Downloading https://ghcr.io/v2/homebrew/core/${id}/blobs/sha256:…`,
    `==> Pouring ${id}.arm64_sequoia.bottle.tar.gz`,
    `🍺  /opt/homebrew/Cellar/${id}: 1 file`,
  ], () => toolRow(id));
}

/**
 * The Demucs cache, installed on purpose instead of in the middle of a dub.
 *
 * Neither a brew nor a snapshot: the server runs demucs's own `get_model` in a
 * subprocess, so what the screen gets is a spinner and the fetch's last line,
 * exactly like a tool install and unlike a download with a denominator.
 */
function startDemucs(): Promise<void> {
  installState = {
    running: true,
    id: DEMUCS_ID,
    ok: null,
    error: null,
    tail: ["$ python -c \"from demucs.pretrained import get_model; get_model('htdemucs_ft')\""],
    check: null,
  };
  return runInstall(DEMUCS_ID, [
    'Downloading: "https://dl.fbaipublicfiles.com/demucs/htdemucs_ft.th"',
    "demucs weights ready",
  ], () => demucsRow());
}

async function runDownload(id: string): Promise<void> {
  const row = MODEL_ROWS[id];
  // ~3% per tick: enough polls that the bar visibly climbs, done in seconds.
  const step = Math.round(row.bytes * 0.03);
  while (installState.running && installState.id === id) {
    await sleep(INSTALL_TICK_MS);
    if (!installState.running || installState.id !== id) return;
    const done = Math.min(row.bytes, (installState.bytes_done ?? 0) + step);
    installState = { ...installState, bytes_done: done };
    if (done >= row.bytes) break;
  }
  if (installState.id !== id) return;
  installed.add(id);
  installState = {
    ...installState,
    running: false,
    ok: true,
    tail: [...installState.tail, "download complete"],
    check: modelRow(id),
    bytes_done: row.bytes,
  };
}

/**
 * The state the page cannot reach by clicking: an install that was already
 * running when it mounted. Seeded on the first `setup()` or `installStatus()`
 * call the moment fixture mode is actually looked at as the default TTS
 * checkpoint mid-download, 40% in, so the mount-time pickup, the live bar and
 * the one-at-a-time greying of every other button are all on screen without a
 * click. It finishes on its own, which demos the last transition too.
 */
let downloadSeeded = false;

function seedDownload(): void {
  if (downloadSeeded || installState.running) return;
  downloadSeeded = true;
  const id = "model.tts.1.7b";
  void startDownload(id, Math.round(MODEL_ROWS[id].bytes * 0.4));
}

async function runInstall(id: string, lines: string[],
                          probe: () => SetupCheck): Promise<void> {
  for (const line of lines) {
    await sleep(INSTALL_TICK_MS);
    installState.tail = [...installState.tail, line];
  }
  await sleep(INSTALL_TICK_MS);
  installed.add(id);
  // The server re-probes rather than trusting the exit code, and hands the
  // fresh row back with the status; the fixture has to do the same or the
  // page's "redraw this row" path never runs here.
  installState = { ...installState, running: false, ok: true, check: probe() };
}

/**
 * The runs that are not the snapshot.
 *
 * The row's whole job is to say *where a run got to*, and a list where every
 * run is finished cannot show that it does. One is stopped mid-pipeline and one
 * fell over on its first stage and those two are not decoration for the list
 * any more, because the editor's transport now has three states that depend on
 * exactly this: a run past `fetch` has `source.wav` to play before the preview
 * exists, and a run that never got through `fetch` has nothing at all. Opening
 * either used to show the finished snapshot, which is the sort of kinder-server
 * divergence these fixtures exist not to have.
 */
const HOUR = 3600;

/**
 * A row, plus how long ago it last moved the list wants a clock time and,
 * when the run died, the job that died.
 *
 * `failure` is not decoration either. The server keeps a project's jobs and the
 * event stream deliberately replays nothing terminal, so after a reload the
 * *reason* a run stopped exists only in that list. A fixture with a failed run
 * and no failed job would demo the recovery panel with the one sentence that
 * matters missing, which is the state it was written for.
 */
type OtherRun = Omit<ProjectSummary, "mtime"> & {
  age: number;
  failure?: { stage: Stage; error: string };
};

const OTHER_RUNS: OtherRun[] = [
  {
    name: "doha_panel_v2",
    title: "Doha panel, full episode",
    src_lang: "ar",
    tgt_lang: "en",
    duration: 1840,
    stages: {
      fetch: "done",
      stems: "done",
      transcript: "done",
      segments: "done",
      translate: "running",
    },
    age: 2 * HOUR,
  },
  {
    name: "archive_reel",
    title: "Archive reel 1994",
    src_lang: "he",
    tgt_lang: "en",
    duration: 415,
    // The download itself failed a dead URL, the commonest first-stage
    // failure there is so this run has no source.wav and nothing to play.
    stages: { fetch: "failed" },
    failure: {
      stage: "fetch",
      error:
        "RuntimeError: yt-dlp exited 1. ERROR: [youtube] 8pQ2mAy: Video unavailable. " +
        "This video is no longer available.",
    },
    age: 39 * HOUR,
  },
];

export function listProjects(): Promise<ProjectSummary[]> {
  const p = store.project;
  const now = Date.now() / 1000;
  return delay([
    {
      name: p.name,
      title: p.source.title,
      src_lang: p.source.src_lang,
      tgt_lang: p.source.tgt_lang,
      duration: p.source.duration,
      stages: p.stages,
      mtime: now,
    },
    ...OTHER_RUNS.map(({ age, ...run }) => ({ ...run, mtime: now - age })),
  ]);
}

/** The detail for a listed run that is not the snapshot: its row, unpacked. */
function detailOf(run: OtherRun): ProjectDetail {
  return {
    name: run.name,
    source: {
      input: `https://example.com/${run.name}`,
      src_lang: run.src_lang,
      tgt_lang: run.tgt_lang,
      duration_limit: null,
      title: run.title,
      duration: run.duration,
      transcript_origin: "asr",
      genre: "documentary",
      register: "narration",
      context: null,
    },
    speakers: {},
    stages: { ...run.stages },
    // Nothing has been produced: `outputs` is written per file, as each stage
    // writes one, and neither of these runs has reached one that does.
    outputs: {},
    report: null,
    jobs: [],
    render: { at: null, stale: true, changed: 0 },
  };
}

/**
 * A project's jobs, as `GET /api/projects/{name}` serves them: oldest first,
 * this project's only.
 *
 * The dead job of a run that failed goes first, because it happened first and
 * because it is the whole point of the list. Everything the session has since
 * enqueued follows it, so a resume shows up here the moment it is submitted.
 */
function jobsOf(name: string): Job[] {
  const failure = OTHER_RUNS.find((run) => run.name === name)?.failure;
  const dead: Job[] = failure
    ? [{
        id: `job_dead_${name}`,
        project: name,
        kind: "run",
        status: "failed",
        progress: 0.12,
        stage: failure.stage,
        message: null,
        error: failure.error,
        // A whole-run job names no lines, and was its own gesture.
        uids: [],
        batch: null,
      }]
    : [];
  return [...dead, ...jobs.filter((job) => job.project === name)].map((job) =>
    structuredClone(job),
  );
}

/**
 * What the last fixture render was made of: uid -> a digest of that line.
 *
 * The server keeps this in `m["render"].segments`, stamped by `mix.run`, and
 * derives `{stale, changed}` from it on every GET. The fixture does the same
 * rather than hardcoding a flag, so an edit in the demo goes stale for the real
 * reason and the count is the real count which is the whole thing the header
 * is about to claim.
 */
let rendered: Map<string, string> | null = null;

/** `id` is left out for the reason the server leaves it out: splits renumber it. */
function digestOf(seg: Segment): string {
  const { id: _id, media: _media, verify: _verify, ...rest } = seg as Segment &
    Record<string, unknown>;
  return JSON.stringify(rest, Object.keys(rest).sort());
}

function renderState(): RenderState {
  if (!rendered) return { at: null, stale: true, changed: 0 };
  const now = new Map(store.segments.map((seg) => [seg.uid, digestOf(seg)]));
  const uids = new Set([...rendered.keys(), ...now.keys()]);
  let changed = 0;
  for (const uid of uids) if (rendered.get(uid) !== now.get(uid)) changed += 1;
  return { at: RENDERED_AT, stale: changed > 0, changed };
}

function stampRender(): void {
  rendered = new Map(store.segments.map((seg) => [seg.uid, digestOf(seg)]));
}

/** Fixed so the demo does not drift: 2024-05-01T09:00:00Z. */
const RENDERED_AT = 1714554000;

// The snapshot is a run that finished, so it opens with a video that is current.
// Stamped after `seedOverrides`, or the demo would open already stale and the
// header would show its loudest state before the user had done anything.
stampRender();

function projectOf(name: string): ProjectDetail {
  const other = OTHER_RUNS.find((run) => run.name === name);
  // Both runs get their jobs: the failed one's dead job is the only record of
  // why it stopped, and it is exactly the run that has one.
  if (other) return { ...detailOf(other), jobs: jobsOf(name) };
  const render = renderState();
  const project = structuredClone(store.project);
  return {
    ...project,
    jobs: jobsOf(name),
    render,
    // `report.run` is called by the same job that ran `mix`, so the numbers and
    // the video go out of date together and on the same cause. The server works
    // it out from the report's own stamp (`Projects.report`); here there is one
    // stamp for both, which is the same answer by a shorter road.
    report: project.report ? { ...project.report, stale: render.stale } : null,
  };
}

export function getProject(name: string): Promise<ProjectDetail> {
  return delay(projectOf(name));
}

/**
 * `PATCH /api/projects/{name}` the three run options that are still a decision.
 *
 * No job, no stage invalidation: the server writes them onto `m["source"]` and
 * they reach the translator the next time it runs. Only the snapshot project has
 * a mutable source here; the two synthetic runs answer with what they would
 * become, which is enough for a screen that re-reads the project afterwards.
 */
export function updateProject(
  name: string,
  patch: ProjectOptionsPatch,
): Promise<ProjectSource> {
  calls.log.push("updateProject");
  calls.updated.push(structuredClone(patch));
  if (activeJobsFor(name).length > 0) {
    return Promise.reject(
      new ApiError(
        "busy",
        `a job is running on '${name}': run options are read when a job starts, ` +
          "not while it runs",
        409,
      ),
    );
  }
  const next: ProjectSource = {
    ...projectOf(name).source,
    ...(patch.genre ? { genre: patch.genre } : {}),
    ...(patch.register ? { register: patch.register } : {}),
    // "" is how the note is cleared the one field with a way out.
    ...(patch.context === undefined ? {} : { context: patch.context.trim() || null }),
  };
  if (name === store.project.name) store.project.source = next;
  return delay(structuredClone(next));
}

/**
 * `POST /api/projects/{name}/run` run it again, which on a stopped run is a
 * resume. Refused while anything is already in flight, exactly as the server
 * refuses it: the pipeline would otherwise re-run the stages it is inside.
 */
export function resume(name: string): Promise<Job> {
  calls.log.push("resume");
  if (activeJobsFor(name).length > 0) {
    return Promise.reject(
      new ApiError("busy", `a job is already going on '${name}'`, 409),
    );
  }
  return delay(structuredClone(makeJob("run", [], undefined, name)));
}

function activeJobsFor(name: string): Job[] {
  return jobs.filter(
    (job) => job.project === name && (job.status === "queued" || job.status === "running"),
  );
}

/**
 * The script and only for a run that has one.
 *
 * `segments` is what fills this list, so a run stopped before it has no lines,
 * not the snapshot's fifty-eight. That is the difference between the editor's
 * empty state being demoable and being a screen nobody has ever seen.
 */
export function getSegments(name: string): Promise<Segment[]> {
  calls.segments += 1;
  if (projectOf(name).stages.segments !== "done") return delay([]);
  return delay(store.segments.map(enrich));
}

/**
 * A waveform that agrees with the marks.
 *
 * The point of a fixture waveform is not that it looks like this particular
 * video it is that the picture and the segment list tell the same story, so
 * a lane drawn against the wrong time base, or a SOURCE envelope accidentally
 * fed the OUTPUT spans, is visible rather than plausible. So the envelope is
 * built *from the segments*: speech inside a span, near-silence between them,
 * a syllable-rate wobble on top and a per-bucket jitter from a hash of the
 * index. Deterministic by construction no `Math.random`, so two calls draw
 * the same picture and a screenshot diff means something.
 */
export function peaks(name: string, file: PeaksFile, n: number): Promise<Peaks> {
  calls.peaks += 1;
  const buckets = Math.max(16, Math.min(Math.round(n), 4000));
// Each lane's file is written by one stage `source.wav` by fetch, `dub.wav`
  // by mix and before that stage has run the server 404s and the lane falls
  // back to marks alone. The fixture has to refuse in the same cases, in the
  // server's own error shape, or that fallback is never exercised anywhere.
  const stages = projectOf(name).stages;
  if (stages[file === "dub" ? "mix" : "fetch"] !== "done") {
    return Promise.reject(
      new ApiError("not_found", `${file}.wav does not exist yet for this run`, 404),
    );
  }

  const spans = store.segments.map((seg) =>
    file === "dub" ? placedSpan(seg) : { start: seg.start, end: seg.end },
  );
  const duration = Math.max(
    store.project.source.duration ?? 0,
    ...spans.map((s) => s.end),
    1,
  );

  const values = new Array<number>(buckets);
  for (let i = 0; i < buckets; i += 1) {
    const t = ((i + 0.5) * duration) / buckets;
    const span = spans.find((s) => t >= s.start && t < s.end);
    if (!span) {
      values[i] = 0.012;
      continue;
    }
    // Fast attack, long sustain, soft release across the span…
    const u = (t - span.start) / Math.max(0.05, span.end - span.start);
    const shape = Math.sin(Math.PI * Math.min(1, Math.max(0, u))) ** 0.7;
    // …syllables, with real troughs between them (a `|sin|` floor of 0.6 draws
    // a sausage, and speech is not a sausage)…
    const syllables = 0.3 + 0.7 * Math.abs(Math.sin(t * 8.5)) ** 1.4;
    // …and grain, so no two buckets are identical.
    const grain = 0.72 + 0.28 * hash01(i);
    values[i] = Number(Math.min(1, shape * syllables * grain).toFixed(4));
  }
  return delay({ file, duration, peaks: values });
}

/** A stable [0, 1) from an integer. Knuth's multiplicative hash, 32-bit. */
function hash01(i: number): number {
  return ((Math.imul(i + 1, 2654435761) >>> 0) % 10007) / 10007;
}

// --- no-model edits: instant, allowed while a job runs ---------------------

export function patchSegment(_name: string, uid: string, patch: SegmentPatch): Promise<Segment> {
  calls.patch += 1;
  calls.log.push("patch");
  const seg = find(uid);

  // `dubbing/edit.py` refuses both of these outright, with a 400. The editor
  // refuses them before they are ever sent but the fixture has to refuse
  // them too, or the guard above it is untested in the only mode the smoke
  // test runs in.
  if ("text_en" in patch && !patch.text_en?.trim()) {
    return Promise.reject(
      new ApiError(
        "invalid_request",
        "text_en cannot be empty: a dubbed segment must say something",
        400,
      ),
    );
  }
  if ("text" in patch && !patch.text?.trim()) {
    return Promise.reject(new ApiError("invalid_request", "text cannot be empty", 400));
  }

  /*
   * The patch's effects come from `applyPatch` the same table the editor
   * predicts with, which is the same table `dubbing/edit.py` implements.
   *
   * What was here was a third copy: it spread the patch over the segment and
   * then invalidated for `text`, `text_en` and `keep`. Everything else went
   * through untouched a speaker change kept the clip in the old voice, a
   * moved span kept a clip cut from the old one, and a language override was
   * written on as a value with nothing dropped. Three implementations of one
   * contract is how a fixture stops being a stand-in for the server and starts
   * being a second, kinder one that hides the first one's behaviour.
   */
  Object.assign(seg, applyPatch(seg, patch));
  // `media` is server-side enrichment, not manifest state: `enrich` below is
  // the only thing allowed to write it, exactly as `Projects.enrich` is.
  delete seg.media;

  return delay(enrich(seg));
}

export function splitSegment(_name: string, uid: string, at: number): Promise<Segment[]> {
  const seg = find(uid);
  const index = store.segments.indexOf(seg);
  const point = Math.min(Math.max(at, seg.start + 0.05), seg.end - 0.05);

  // Splitting invalidates everything downstream of segmentation for both halves:
  // the translation was written for the whole line and the clip was synthesized
  // from it, so neither half may keep them.
  const head: Segment = {
    ...seg,
    uid: `${seg.uid}a`,
    end: point,
    text_en: null,
    tts: null,
    place: null,
    verify: null,
  };
  const tail: Segment = {
    ...seg,
    uid: `${seg.uid}b`,
    start: point,
    text: "",
    text_en: null,
    tts: null,
    place: null,
    verify: null,
    locked: null,
  };
  store.segments.splice(index, 1, head, tail);
  renumber();
  return delay(store.segments.map(enrich));
}

export function mergeSegments(_name: string, uid: string, uidB: string): Promise<Segment[]> {
  const a = find(uid);
  const b = find(uidB);
  const [first, second] = a.start <= b.start ? [a, b] : [b, a];
  const index = store.segments.indexOf(first);

  const merged: Segment = {
    ...first,
    end: second.end,
    text: [first.text, second.text].filter(Boolean).join(" "),
    text_en: null,
    tts: null,
    place: null,
    verify: null,
  };
  store.segments.splice(index, 2, merged);
  renumber();
  return delay(store.segments.map(enrich));
}

/**
 * `POST /segments` claim an uncovered span.
 *
 * Every refusal the server makes is made here too, with the same code, because
 * this is the only implementation of the contract the smoke test runs against:
 * no text, a span shorter than the pipeline's `MIN_SEG_SEC`, and above all an
 * overlap, which the server *refuses* rather than clamping (`timeline.place` is
 * the sole authority on where audio goes). A fixture that quietly clamped would
 * make the one error the composer has to show unreachable in demo mode.
 */
export function addSegment(_name: string, body: NewSegment): Promise<Segment[]> {
  const text = body.text.trim();
  if (!text) throw new ApiError("invalid_request", "a new segment needs its text", 400);
  if (body.end - body.start < MIN_SEG_SEC) {
    throw new ApiError(
      "invalid_request",
      `a segment shorter than ${MIN_SEG_SEC}s cannot be synthesized reliably`,
      400,
    );
  }
  const clash = store.segments.find((s) => s.start < body.end && s.end > body.start);
  if (clash) {
    throw new ApiError(
      "invalid_request",
      `[${body.start}, ${body.end}] overlaps segment ${clash.id}`,
      400,
    );
  }
  const nearest = [...store.segments].sort(
    (a, b) => gapTo(a, body) - gapTo(b, body),
  )[0];
  const seg = {
    ...blankSegment(),
    uid: `added-${Math.round(body.start * 1000)}`,
    start: body.start,
    end: body.end,
    speaker: body.speaker || nearest?.speaker || "SPEAKER_00",
    text,
    // Exactly what `edit.add` writes: the words are the user's, and so is the
    // verdict that this span is a dub rather than a keep waiting to happen.
    locked: { text: true },
    passthrough: false,
  } as Segment;
  store.segments.push(seg);
  renumber();
  return delay(store.segments.map(enrich));
}

/** `DELETE /segments/{uid}` the record goes; the span plays the original. */
export function removeSegment(_name: string, uid: string): Promise<Segment[]> {
  const seg = find(uid);
  store.segments.splice(store.segments.indexOf(seg), 1);
  renumber();
  return delay(store.segments.map(enrich));
}

/** `dubbing.segments.MIN_SEG_SEC` restated, because the check is server-side. */
const MIN_SEG_SEC = 0.9;

const gapTo = (seg: Segment, span: { start: number; end: number }): number =>
  Math.max(seg.start - span.end, span.start - seg.end, 0);

/** A segment with nothing generated for it, which is what a new one is. */
function blankSegment(): Segment {
  const template = store.segments[0];
  return {
    ...template,
    keep: false,
    keep_reason: null,
    text_en: null,
    text_mid: null,
    tts: null,
    place: null,
    verify: null,
    tts_opts: null,
    locked: null,
    src_lang: null,
    tgt_lang: null,
    detected_lang: null,
    passthrough: null,
  };
}

/** `id` is positional and renumbers on every structural edit; `uid` never does. */
function renumber(): void {
  store.segments.sort((x, y) => x.start - y.start);
  store.segments.forEach((seg, i) => {
    seg.id = i;
  });
}

// --- jobs: the one-job rule ------------------------------------------------

export function listJobs(): Promise<Job[]> {
  return delay(structuredClone(jobs));
}

/**
 * `batch` mirrors `JobQueue.cancel_batch`, order included.
 *
 * The queued members go first, while the running one still holds the worker —
 * cancelling the running job first would let `pump` start the very job this call
 * is stopping, which is the bug the server-side ordering exists to prevent and
 * which the fixture must be able to reproduce or the smoke test proves nothing.
 */
export function cancelJob(id: string, batch = false): Promise<void> {
  const target = jobs.find((j) => j.id === id);
  if (!target) return delay(undefined);
  const doomed =
    batch && target.batch
      ? [...jobs.filter((j) => j.batch === target.batch && j.status === "queued"),
         ...jobs.filter((j) => j.batch === target.batch && j.status === "running")]
      : [target];
  for (const job of doomed) {
    if (job.status !== "queued" && job.status !== "running") continue;
    job.status = "cancelled";
    emit({ type: "job", id: job.id, status: "cancelled", kind: job.kind,
           uids: job.uids, batch: job.batch });
  }
  return delay(undefined);
}

export function createProject(body: CreateProjectRequest): Promise<CreateProjectResponse> {
  calls.created.push(structuredClone(body));
  calls.log.push("createProject");
  store.project.source = {
    ...store.project.source,
    input: body.source,
    src_lang: body.src_lang,
    tgt_lang: body.tgt_lang,
    duration_limit: body.duration ?? null,
    context: body.context ?? null,
    genre: body.genre ?? null,
    register: body.register ?? null,
    transcript_origin: body.transcript && body.transcript !== "auto"
      ? body.transcript
      : store.project.source.transcript_origin,
  };
  const job = makeJob("run", []);
  return delay({ name: store.project.name, job: structuredClone(job) });
}

export function enqueue(
  _name: string,
  kind: JobKind,
  uids: string[],
  batch?: string,
): Promise<Job> {
  if (kind === "retranslate" || kind === "resynthesize") calls[kind] += 1;
  calls.log.push(kind);
  const job = makeJob(kind, uids, batch);
  return delay(structuredClone(job));
}

function makeJob(
  kind: JobKind,
  uids: string[],
  batch?: string,
  project = store.project.name,
): Job {
  jobSeq += 1;
  const job: Job = {
    id: `job_${jobSeq}`,
    project,
    kind,
    status: "queued",
    progress: null,
    stage: null,
    message: null,
    error: null,
    uids,
    batch: batch ?? null,
  };
  jobs.push(job);
  emit({ type: "job", id: job.id, status: "queued", kind, uids, batch: job.batch });
  void pump();
  return job;
}

/** One job at a time, process-wide the models are never co-resident. */
async function pump(): Promise<void> {
  if (running) return;
  running = true;
  try {
    for (;;) {
      const job = jobs.find((j) => j.status === "queued");
      if (!job) break;
      await runJob(job);
    }
  } finally {
    running = false;
  }
}

const RUN_STAGES: Stage[] = [
  "fetch",
  "stems",
  "transcript",
  "segments",
  "translate",
  "tts",
  "timeline",
  "mix",
  "report",
];

async function runJob(job: Job): Promise<void> {
  job.status = "running";
  emit({ type: "job", id: job.id, status: "running", kind: job.kind,
         uids: job.uids, batch: job.batch });

  const stages: Stage[] =
    job.kind === "run"
      ? RUN_STAGES
      : job.kind === "render"
        ? ["timeline", "mix", "report"]
        : job.kind === "retranslate"
          ? ["translate"]
          : ["tts"];

  /*
   * A rebuild goes dark before it goes forward.
   *
   * `edit.rebuild` drops the "done" mark of every stage it is about to redo
   * (`manifest.clear_stage`) before it redoes any of them, so a render really
   * does take five of the nine dots off the track the moment it starts. A
   * fixture that skipped that never showed the state the UI has to stay honest
   * about the state a user read as "it removed all my renders".
   */
  const owner = OTHER_RUNS.find((run) => run.name === job.project);
  if (!owner) {
    for (const stage of stages) {
      store.project.stages = { ...store.project.stages, [stage]: "pending" };
    }
  }

  for (const stage of stages) {
    if ((job.status as string) === "cancelled") return;
    const steps = job.uids?.length ? job.uids.length : 6;
    for (let i = 0; i < steps; i += 1) {
      if ((job.status as string) === "cancelled") return;
      await sleep(job.kind === "run" ? 260 : 420);
      job.stage = stage;
      job.progress = (i + 1) / steps;
      job.message = `${stage} ${i + 1}/${steps}`;
      emit({
        type: "stage",
        stage,
        status: "running",
        progress: job.progress,
        message: job.message,
      });
      const uid = job.uids?.[i];
      if (uid) {
        applyModelResult(job.kind, uid);
        emit({
          type: "segment",
          uid,
          field: job.kind === "retranslate" ? "text_en" : "tts",
          status: "done",
        });
      }
    }
    // The *job's* project, not always the snapshot: a resume on a run that fell
    // over at fetch has to move that run's stages, or the recovery panel it was
    // started from stays stuck on the failure it just fixed.
    const other = OTHER_RUNS.find((run) => run.name === job.project);
    if (other) {
      other.stages = { ...other.stages, [stage]: "done" };
    } else {
      store.project.stages = { ...store.project.stages, [stage]: "done" };
      // `mix.run` is what writes preview.mp4 and what stamps the manifest with
      // the segments it was made from, so the video stops being stale exactly
      // here not when the job ends, and not when `report` finishes after it.
      // Only the snapshot has segments to stamp; `rendered` is its map alone.
      if (stage === "mix") stampRender();
    }
    emit({ type: "stage", stage, status: "done", progress: 1 });
    emit({ type: "log", level: "info", message: `${stage} done` });
  }

  job.status = "done";
  job.progress = 1;
  emit({ type: "job", id: job.id, status: "done", progress: 1, kind: job.kind,
         uids: job.uids, batch: job.batch });
}

/** What a model action actually changes on the segment. */
function applyModelResult(kind: JobKind, uid: string): void {
  const seg = store.segments.find((s) => s.uid === uid);
  if (!seg) return;
  /*
   * Asking for the work undoes the verdict the failure of that work caused.
   *
   * `edit.retranslate`/`resynthesize` both start with `invalidate`, whose undo
   * table reopens a `mt_failed` keep on a translate and a `tts_failed` keep on
   * either the segment gets another go rather than being stuck on a verdict
   * about a clip that no longer exists. It is what makes the Failed filter's
   * two buttons the right two buttons, so the fixture has to do it too.
   */
  const undo = kind === "retranslate" ? ["mt_failed", "tts_failed"] : ["tts_failed"];
  if (!seg.locked?.keep && undo.includes(seg.keep_reason ?? "")) {
    seg.keep = false;
    seg.keep_reason = null;
  }
  if (kind === "retranslate") {
    // Locked fields are never overwritten by a re-run.
    if (seg.locked?.text_en) return;
    seg.text_en = (seg.text_en ?? seg.text).replace(/\s*\[re-translated]$/, "") + " [re-translated]";
    seg.tts = null;
    seg.place = null;
    seg.verify = null;
  }
  if (kind === "resynthesize") {
    const dur = Math.max(0.6, (seg.end - seg.start) * 0.95);
    seg.tts = {
      clip: `clips/${seg.uid}.wav`,
      dur: Number(dur.toFixed(2)),
      tries: 1,
      overlap: 1,
      verify: "ok",
    };
    seg.place = {
      start: seg.start,
      end: seg.start + dur,
      rate: 1,
      drift: 0,
      clip: `clips/${seg.uid}.wav`,
    };
    seg.verify = { ok: true, overlap: 1, heard: seg.text_en ?? "" };
  }
}

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

// --- media & events --------------------------------------------------------

/**
 * There is no video in fixture mode. Returning "" is honest and lets the player
 * fall back to a virtual transport, so the timeline and playhead stay demoable.
 *
 * Audio is a different matter: `source.wav` is what the transport plays before
 * a run has a preview, and a fixture that answered "" for it would leave the
 * one mode this file exists to demo untestable. It is a tone, like every other
 * fixture clip long enough to hear the playhead move, and capped by
 * `resolveToneUrl` either way.
 */
export function mediaUrl(_name: string, path: string): string {
  if (/\.(mp4|mkv|webm|mov)$/i.test(path)) return "";
  if (path === "source.wav") return toneUrl({ hz: 150, dur: 20, seed: 5 });
  return toneUrl({ hz: 150, dur: 4, seed: path.length });
}

/**
 * The prelude the server opens every stream with (`app.py::project_events`):
 * one log line, one stage frame per stage, and one job frame for every job that
 * has not finished.
 *
 * It is not decoration it is the only reason "Start dubbing" works. The
 * project is created, the job is enqueued, and *then* the editor mounts and
 * subscribes; without a replay the queued/running frames were already gone, so
 * the run's own job strip never appeared and the preview stage sat on "Nothing
 * has run yet" for the whole run. Against the real server that is not what
 * happens, which is exactly the kind of divergence these fixtures exist to not
 * have a second, kinder server that hides the first one's behaviour.
 */
function prelude(name: string): StudioEvent[] {
  const stages = projectOf(name).stages;
  return [
    { type: "log", level: "info", message: `watching ${name}` },
    // `replay: true` on every prelude frame and on nothing else the nine stage
    // frames are a snapshot of all nine at once, not a progression, and a client
    // that pinned its display to the last one read the ninth as the stage the run
    // was in. The fixture has to say so too, or the smoke test cannot tell the
    // difference between a client that reads the flag and one that ignores it.
    ...RUN_STAGES.map((stage) => ({
      type: "stage" as const,
      stage,
      status: stages[stage] ?? ("pending" as const),
      ...(stages[stage] === "done" ? { progress: 1 } : {}),
      replay: true,
    })),
    ...jobs
      // This run's unfinished jobs, not the process's: a stream is opened per
      // project and replaying somebody else's job puts a strip on the wrong
      // editor. `app.py::project_events` filters the same way and terminal
      // means done, failed *or* cancelled: a failed job resent on every
      // reconnect resurrects an error the user dismissed an hour ago.
      .filter((job) => job.project === name && isPending(job))
      .map((job) => ({
        type: "job" as const,
        id: job.id,
        status: job.status,
        kind: job.kind,
        progress: job.progress,
        error: job.error,
        uids: job.uids,
        batch: job.batch,
        replay: true,
      })),
  ];
}

export function events(
  name: string,
  signal: AbortSignal,
  onMessage: (event: StudioEvent) => void,
  onConnectionChange?: (open: boolean) => void,
): Promise<void> {
  listeners.add(onMessage);
  onConnectionChange?.(true);
  for (const event of prelude(name)) onMessage(event);
  const beat = setInterval(() => onMessage({ type: "heartbeat", t: Date.now() }), 15000);

  return new Promise((resolve) => {
    const stop = () => {
      clearInterval(beat);
      listeners.delete(onMessage);
      onConnectionChange?.(false);
      resolve();
    };
    if (signal.aborted) stop();
    else signal.addEventListener("abort", stop, { once: true });
  });
}

/** The run directory name the fixture serves, so routes can link straight to it. */
export const FIXTURE_PROJECT = store.project.name;
