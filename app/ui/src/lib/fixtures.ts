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
  SetupStatus,
  Stage,
  StudioEvent,
} from "./types";

const LATENCY_MS = 45;

type Store = { project: ProjectDetail; segments: Segment[] };

const store: Store = structuredClone(data) as unknown as Store;

/**
 * Three fields the snapshot happens not to contain.
 *
 * `fixture-data.json` is a faithful dump of a real run, and that run never had
 * a per-segment TTS override or a language tag on it — so two of the
 * inspector's three shelves would demo as permanently empty, which is exactly
 * the state that hides a layout bug. This adds one of each, by position rather
 * than by content, and it is the only place fixture mode invents anything.
 */
function seedOverrides(): void {
  const withOpts = store.segments.find((seg) => !seg.keep && seg.tts);
  if (withOpts) withOpts.tts_opts = { seed: 4711, greedy: true };
  const withLang = store.segments.find((seg) => seg.src_lang);
  if (withLang) withLang.tgt_lang = "en";
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
 */
export const calls = { patch: 0, segments: 0, peaks: 0 };
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

function find(uid: string): Segment {
  const seg = store.segments.find((s) => s.uid === uid);
  if (!seg) throw new Error(`no such segment: ${uid}`);
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
 * `GET /api/setup`, deliberately mixed: a machine with the tools and the big
 * models in place but no HF token and no Demucs weights. A checklist where
 * everything passes demos nothing, and the failure rows are where the copy has
 * to earn its keep — each one says what to do, not just that it is missing.
 */
export function setup(): Promise<SetupStatus> {
  const checks = [
    { id: "ffmpeg", label: "ffmpeg", ok: true, detail: "7.1.1 — /opt/homebrew/bin/ffmpeg" },
    { id: "sox", label: "sox", ok: true, detail: "14.4.2 — /opt/homebrew/bin/sox" },
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
  ];
  return delay({ ok: checks.every((c) => c.ok), checks });
}

/**
 * The projects list, with three runs in three different states.
 *
 * Only the first is real — it is the snapshot everything else here serves. The
 * other two exist because the row's whole job is to say *where a run got to*,
 * and a list where every run is finished cannot show that it does: one is
 * stopped mid-pipeline and one failed, which are the two rows a user needs to
 * be able to tell apart at a glance without opening either.
 */
export function listProjects(): Promise<ProjectSummary[]> {
  const p = store.project;
  const hour = 3600;
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
      mtime: now - 2 * hour,
    },
    {
      name: "archive_reel",
      title: "Archive reel 1994",
      src_lang: "he",
      tgt_lang: "en",
      duration: 415,
      stages: { fetch: "done", stems: "done", transcript: "failed" },
      mtime: now - 39 * hour,
    },
  ]);
}

export function getProject(_name: string): Promise<ProjectDetail> {
  return delay(structuredClone(store.project));
}

export function getSegments(_name: string): Promise<Segment[]> {
  calls.segments += 1;
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
export function peaks(_name: string, file: PeaksFile, n: number): Promise<Peaks> {
  calls.peaks += 1;
  const buckets = Math.max(16, Math.min(Math.round(n), 4000));
  // `dub.wav` is written by the mix stage; before it has run the server 404s
  // and the lane falls back to marks alone. The fixture has to refuse it in
  // the same case or that fallback is never exercised anywhere.
  if (file === "dub" && store.project.stages.mix !== "done") {
    return Promise.reject(new Error("dub.wav does not exist yet for this run"));
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
  const seg = find(uid);

  // `dubbing/edit.py` refuses both of these outright, with a 400. The editor
  // refuses them before they are ever sent — but the fixture has to refuse
  // them too, or the guard above it is untested in the only mode the smoke
  // test runs in.
  if ("text_en" in patch && !patch.text_en?.trim()) {
    return Promise.reject(
      new Error("text_en cannot be empty — a dubbed segment must say something"),
    );
  }
  if ("text" in patch && !patch.text?.trim()) {
    return Promise.reject(new Error("text cannot be empty"));
  }

  Object.assign(seg, patch);

  // Editing the line invalidates everything made *from* the line: the clip
  // provably says the old words. `edit.set_text` calls `invalidate(translate)`
  // for exactly this, and a fixture that quietly kept a stale clip would make
  // the editor's optimistic merge look correct when it is not.
  if ("text_en" in patch || "text" in patch) {
    seg.tts = null;
    seg.place = null;
    seg.verify = null;
  }

  // The user's edits outrank the pipeline: a hand-edited field is locked so a
  // re-run cannot overwrite it (docs/APP_ARCHITECTURE.md, non-negotiables).
  const locked = { ...(seg.locked ?? {}) };
  for (const field of ["text", "text_en", "speaker"] as const) {
    if (field in patch) locked[field] = true;
  }
  if ("keep" in patch) {
    locked.keep = true;
    seg.keep_reason = patch.keep ? (patch.keep_reason ?? "manual") : null;
  }
  // `locked` in the patch is the user's final word, applied last, and it is a
  // replace rather than a merge — `{}` is how "hand this line back to the
  // pipeline" is expressed (edit.set_locked).
  const final = patch.locked ? { ...patch.locked } : locked;
  const on = Object.fromEntries(Object.entries(final).filter(([, v]) => v));
  seg.locked = Object.keys(on).length ? on : null;

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
 */
export function mediaUrl(_name: string, path: string): string {
  if (/\.(mp4|mkv|webm|mov)$/i.test(path)) return "";
  return toneUrl({ hz: 150, dur: 4, seed: path.length });
}

export function events(
  _name: string,
  signal: AbortSignal,
  onMessage: (event: StudioEvent) => void,
  onConnectionChange?: (open: boolean) => void,
): Promise<void> {
  listeners.add(onMessage);
  onConnectionChange?.(true);
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
