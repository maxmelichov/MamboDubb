/**
 * Fixture backend — `VITE_USE_FIXTURES=1`.
 *
 * The studio server is being written in parallel against the same contract, so
 * the UI ships its own implementation of that contract: same shapes, same event
 * stream, same one-job rule. The data is a snapshot of a real run
 * (`scripts/make_fixture.py`), which matters — 58 segments of real Hebrew with
 * real translations, real drift and real verification transcripts exercise
 * bidi text, long lines and low-overlap warnings that invented data would not.
 *
 * Everything here is in-memory: edits survive navigation, not a reload.
 */

import data from "./fixture-data.json";
import { ApiError } from "./apiError";
import { applyPatch } from "./patch";
import { placedSpan } from "./segments";
import { toneUrl } from "./tone";
import type {
  CreateProjectRequest,
  CreateProjectResponse,
  Health,
  Job,
  JobKind,
  Peaks,
  PeaksFile,
  ProjectDetail,
  ProjectSummary,
  Segment,
  SegmentPatch,
  SetupCheck,
  SetupInstall,
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
 * finished has no per-segment TTS override, no language tag, and — the reason
 * this matters most — no line left unfinished. So two of the inspector's
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
   * A line that has its clip and no placement — the ordinary state of a studio
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
   * writes — the record's verdict is "ok", "soft" or "keep" and nothing else.
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
}

/**
 * The record a pipeline failure leaves behind: a keep it decided against
 * itself, with a slice of the original audio attached so the mix is never
 * silent (`tts.Engine.keep_clip`). `verify` stays as it was — the low overlap
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
 * it cannot be asserted at all. It exists for one claim in particular — that
 * closing an editor without changing the text saves nothing — which is
 * invisible to a DOM assertion because a no-op PATCH looks exactly like no
 * PATCH from the outside.
 *
 * `log` is the same information with the *order* kept, which is what a flip to
 * "Dub it" has to be judged on: the PATCH clears the line and the clip, so the
 * translate job must be enqueued after it and the voice job after that. A pair
 * of counters cannot tell that apart from the same two calls in the wrong
 * order. Only the calls that queue work are logged — a background refetch of
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
  log: [] as string[],
  /**
   * Every body the import screen has sent, kept whole.
   *
   * A counter cannot answer the question this one is asked: whether a switch on
   * that screen reached the request. The options rail is a pile of fields that
   * only exist to be *sent* — the screen shows no consequence of any of them —
   * so the sent body is the only place their correctness is observable.
   */
  created: [] as CreateProjectRequest[],
};
(globalThis as { __DUBBING_FIXTURE_CALLS__?: typeof calls }).__DUBBING_FIXTURE_CALLS__ = calls;

// --- enrichment -----------------------------------------------------------

/** Speaker label → a stable pitch, so A/B tells two voices apart by ear. */
function speakerHz(speaker: string, offset: number): number {
  let hash = 0;
  for (let i = 0; i < speaker.length; i += 1) hash = (hash * 31 + speaker.charCodeAt(i)) % 997;
  return 110 + (hash % 9) * 14 + offset;
}

function seedOf(uid: string): number {
  let hash = 7;
  for (let i = 0; i < uid.length; i += 1) hash = (hash * 33 + uid.charCodeAt(i)) % 100003;
  return hash;
}

/**
 * Add the fields `GET /segments` promises and the manifest does not store.
 *
 * The shape is `Projects.enrich`'s, key for key — `media.play`, `media.tts`,
 * `media.source`, `media.source_window`. It has to be: these fixtures are the
 * only implementation of the contract that the smoke test ever exercises, so a
 * fixture that invents its own field names is not a stand-in for the server,
 * it is a second, divergent server that hides the first one's bugs. That is
 * literally how A/B playback shipped broken.
 */
function enrich(seg: Segment): Segment {
  const seed = seedOf(seg.uid);
  const sourceDur = Math.max(0.4, seg.end - seg.start);
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
      source: toneUrl({ hz: speakerHz(seg.speaker, 0), dur: sourceDur, seed }),
      source_window: [seg.start, seg.end],
    },
  };
}

const delay = <T>(value: T): Promise<T> =>
  new Promise((resolve) => setTimeout(() => resolve(value), LATENCY_MS));

/**
 * Every failure here is an `ApiError`, never a bare `Error`.
 *
 * The server speaks one error envelope — `{"error":{code,message}}` — and the
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
 * `GET /api/setup`, deliberately mixed: a fresh machine with the big models
 * already fetched but neither command-line tool, no HF token and no Demucs
 * weights. A checklist where everything passes demos nothing, and the failure
 * rows are where the copy has to earn its keep — each one says what to do, not
 * just that it is missing.
 *
 * The two tool rows are the *installable* ones, which is the state the Install
 * button only exists for: `ffmpeg` and `sox` are one `brew install` away, so the
 * app offers to run it. Everything else below is missing in a way no button can
 * fix — a gated token, a download the pipeline does for itself — and must
 * therefore show no button at all.
 */
export function setup(): Promise<SetupStatus> {
  const checks = setupChecks();
  return delay({ ok: checks.every((c) => c.ok), checks });
}

/** Which tools this fake machine has had installed during the session. */
const installed = new Set<string>();

const TOOL_ROWS: Record<string, { label: string; here: string; missing: string }> = {
  ffmpeg: {
    label: "ffmpeg",
    here: "7.1.1 — /opt/homebrew/bin/ffmpeg",
    missing:
      "ffmpeg not on PATH — every stage shells out to it for audio and video. " +
      "Install it with `brew install ffmpeg`.",
  },
  sox: {
    label: "SoX",
    here: "14.4.2 — /opt/homebrew/bin/sox",
    missing:
      "sox not on PATH — Qwen3-TTS text normalization needs it. " +
      "Install it with `brew install sox`.",
  },
};

function toolRow(id: string): SetupCheck {
  const row = TOOL_ROWS[id];
  const ok = installed.has(id);
  return { id, label: row.label, ok, installable: true, detail: ok ? row.here : row.missing };
}

function setupChecks(): SetupCheck[] {
  return [
    toolRow("ffmpeg"),
    toolRow("sox"),
    {
      id: "hf_token",
      label: "Hugging Face token",
      ok: false,
      detail:
        "No HF_TOKEN in .env. Pyannote's diarization models are gated; without a token every " +
        "segment is attributed to one speaker. Accept the model terms, then put HF_TOKEN=… in .env.",
    },
    {
      id: "model_translate",
      label: "Translation model — Gemma 3 12B (MLX, 4-bit)",
      ok: true,
      detail: "6.9 GB — models/gemma-3-12b-it-qat-4bit",
    },
    {
      id: "model_tts",
      label: "Speech model — Qwen3-TTS 1.7B",
      ok: true,
      detail: "3.4 GB — models/qwen3-tts-1.7b",
    },
    {
      id: "model_asr",
      label: "Transcription model — faster-whisper large-v3",
      ok: true,
      detail: "3.1 GB — models/faster-whisper-large-v3",
    },
    {
      id: "model_stems",
      label: "Stem separation — Demucs htdemucs",
      ok: false,
      detail:
        "Not in models/demucs. Needed to lift the speech off the music before transcription. " +
        "Run `uv run python -m dubbing.stems --download` to fetch it (320 MB).",
    },
    {
      id: "disk",
      label: "Free disk space",
      ok: true,
      detail: "184 GB free — a 20-minute run writes about 4 GB under outputs/",
    },
  ].map((row) => ({ installable: false, ...row }));
}

/**
 * `POST|GET /api/setup/install`, compressed.
 *
 * The real thing is a `brew install` and minutes of output; this is the same
 * contract at a speed a smoke test can watch — a handful of lines, a few poll
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

export function startInstall(id: string): Promise<SetupInstall> {
  calls.install += 1;
  calls.log.push(`install:${id}`);
  if (installState.running) {
    return Promise.reject(
      new ApiError("busy", `an install is already running (${installState.id}); one at a time`, 409),
    );
  }
  if (!(id in TOOL_ROWS)) {
    return Promise.reject(
      new ApiError(
        "invalid_request",
        `'${id}' cannot be installed from the app. The only installs it runs are ` +
          "`brew install ffmpeg`, `brew install sox`. Everything else is by hand — " +
          "the command is in that check's detail line.",
        400,
      ),
    );
  }
  installState = {
    running: true,
    id,
    ok: null,
    error: null,
    tail: [`$ brew install ${id}`],
    check: null,
  };
  void runInstall(id);
  return delay(structuredClone(installState));
}

export function installStatus(): Promise<SetupInstall> {
  return delay(structuredClone(installState));
}

async function runInstall(id: string): Promise<void> {
  for (const line of [
    `==> Fetching ${id}`,
    `==> Downloading https://ghcr.io/v2/homebrew/core/${id}/blobs/sha256:…`,
    `==> Pouring ${id}.arm64_sequoia.bottle.tar.gz`,
  ]) {
    await sleep(INSTALL_TICK_MS);
    installState.tail = [...installState.tail, line];
  }
  await sleep(INSTALL_TICK_MS);
  installed.add(id);
  // The server re-probes rather than trusting the exit code, and hands the
  // fresh row back with the status; the fixture has to do the same or the
  // page's "redraw this row" path never runs here.
  installState = {
    ...installState,
    running: false,
    ok: true,
    tail: [...installState.tail, `🍺  /opt/homebrew/Cellar/${id}: 1 file`],
    check: toolRow(id),
  };
}

/**
 * The runs that are not the snapshot.
 *
 * The row's whole job is to say *where a run got to*, and a list where every
 * run is finished cannot show that it does. One is stopped mid-pipeline and one
 * fell over on its first stage — and those two are not decoration for the list
 * any more, because the editor's transport now has three states that depend on
 * exactly this: a run past `fetch` has `source.wav` to play before the preview
 * exists, and a run that never got through `fetch` has nothing at all. Opening
 * either used to show the finished snapshot, which is the sort of kinder-server
 * divergence these fixtures exist not to have.
 */
const HOUR = 3600;

/** A row, plus how long ago it last moved — the list wants a clock time. */
type OtherRun = Omit<ProjectSummary, "mtime"> & { age: number };

const OTHER_RUNS: OtherRun[] = [
  {
    name: "doha_panel_v2",
    title: "Doha panel — full episode",
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
    // The download itself failed — a dead URL, the commonest first-stage
    // failure there is — so this run has no source.wav and nothing to play.
    stages: { fetch: "failed" },
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
    },
    speakers: {},
    stages: { ...run.stages },
    // Nothing has been produced: `outputs` is written per file, as each stage
    // writes one, and neither of these runs has reached one that does.
    outputs: {},
    report: null,
  };
}

function projectOf(name: string): ProjectDetail {
  const other = OTHER_RUNS.find((run) => run.name === name);
  return other ? detailOf(other) : structuredClone(store.project);
}

export function getProject(name: string): Promise<ProjectDetail> {
  return delay(projectOf(name));
}

/**
 * The script — and only for a run that has one.
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
 * video — it is that the picture and the segment list tell the same story, so
 * a lane drawn against the wrong time base, or a SOURCE envelope accidentally
 * fed the OUTPUT spans, is visible rather than plausible. So the envelope is
 * built *from the segments*: speech inside a span, near-silence between them,
 * a syllable-rate wobble on top and a per-bucket jitter from a hash of the
 * index. Deterministic by construction — no `Math.random`, so two calls draw
 * the same picture and a screenshot diff means something.
 */
export function peaks(name: string, file: PeaksFile, n: number): Promise<Peaks> {
  calls.peaks += 1;
  const buckets = Math.max(16, Math.min(Math.round(n), 4000));
// Each lane's file is written by one stage — `source.wav` by fetch, `dub.wav`
  // by mix — and before that stage has run the server 404s and the lane falls
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
  // refuses them before they are ever sent — but the fixture has to refuse
  // them too, or the guard above it is untested in the only mode the smoke
  // test runs in.
  if ("text_en" in patch && !patch.text_en?.trim()) {
    return Promise.reject(
      new ApiError(
        "invalid_request",
        "text_en cannot be empty — a dubbed segment must say something",
        400,
      ),
    );
  }
  if ("text" in patch && !patch.text?.trim()) {
    return Promise.reject(new ApiError("invalid_request", "text cannot be empty", 400));
  }

  /*
   * The patch's effects come from `applyPatch` — the same table the editor
   * predicts with, which is the same table `dubbing/edit.py` implements.
   *
   * What was here was a third copy: it spread the patch over the segment and
   * then invalidated for `text`, `text_en` and `keep`. Everything else went
   * through untouched — a speaker change kept the clip in the old voice, a
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

export function cancelJob(id: string): Promise<void> {
  const job = jobs.find((j) => j.id === id);
  if (job && (job.status === "queued" || job.status === "running")) {
    job.status = "cancelled";
    emit({ type: "job", id: job.id, status: "cancelled" });
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
  };
  const job = makeJob("run", []);
  return delay({ name: store.project.name, job: structuredClone(job) });
}

export function enqueue(_name: string, kind: JobKind, uids: string[]): Promise<Job> {
  if (kind === "retranslate" || kind === "resynthesize") calls[kind] += 1;
  calls.log.push(kind);
  const job = makeJob(kind, uids);
  return delay(structuredClone(job));
}

function makeJob(kind: JobKind, uids: string[]): Job {
  jobSeq += 1;
  const job: Job = {
    id: `job_${jobSeq}`,
    project: store.project.name,
    kind,
    status: "queued",
    progress: null,
    stage: null,
    message: null,
    error: null,
    uids,
  };
  jobs.push(job);
  emit({ type: "job", id: job.id, status: "queued", kind });
  void pump();
  return job;
}

/** One job at a time, process-wide — the models are never co-resident. */
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
  emit({ type: "job", id: job.id, status: "running", kind: job.kind });

  const stages: Stage[] =
    job.kind === "run"
      ? RUN_STAGES
      : job.kind === "render"
        ? ["timeline", "mix", "report"]
        : job.kind === "retranslate"
          ? ["translate"]
          : ["tts"];

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
    store.project.stages = { ...store.project.stages, [stage]: "done" };
    emit({ type: "stage", stage, status: "done", progress: 1 });
    emit({ type: "log", level: "info", message: `${stage} done` });
  }

  job.status = "done";
  job.progress = 1;
  emit({ type: "job", id: job.id, status: "done", progress: 1 });
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
   * either — the segment gets another go rather than being stuck on a verdict
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
 * fixture clip — long enough to hear the playhead move, and capped by
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
 * It is not decoration — it is the only reason "Start dubbing" works. The
 * project is created, the job is enqueued, and *then* the editor mounts and
 * subscribes; without a replay the queued/running frames were already gone, so
 * the run's own job strip never appeared and the preview stage sat on "Nothing
 * has run yet" for the whole run. Against the real server that is not what
 * happens, which is exactly the kind of divergence these fixtures exist to not
 * have — a second, kinder server that hides the first one's behaviour.
 */
function prelude(name: string): StudioEvent[] {
  const stages = projectOf(name).stages;
  return [
    { type: "log", level: "info", message: `watching ${name}` },
    ...RUN_STAGES.map((stage) => ({
      type: "stage" as const,
      stage,
      status: stages[stage] ?? ("pending" as const),
      ...(stages[stage] === "done" ? { progress: 1 } : {}),
    })),
    ...jobs
      // This run's unfinished jobs, not the process's: a stream is opened per
      // project and replaying somebody else's job puts a strip on the wrong
      // editor. `app.py::project_events` filters the same way.
      .filter((job) => job.project === name && job.status !== "done")
      .map((job) => ({
        type: "job" as const,
        id: job.id,
        status: job.status,
        kind: job.kind,
        progress: job.progress,
        error: job.error,
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
