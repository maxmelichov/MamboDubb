/**
 * Wire types for the studio server, transcribed from the HTTP API table in
 * docs/APP_ARCHITECTURE.md. This file is the UI's half of that contract: if the
 * server disagrees with something here, one of the two is a bug and the doc says
 * which. Nothing else in the app may declare a wire shape.
 */

export const STAGES = [
  "fetch",
  "stems",
  "transcript",
  "segments",
  "translate",
  "tts",
  "timeline",
  "mix",
  "report",
] as const;

export type Stage = (typeof STAGES)[number];

export type StageStatus = "pending" | "running" | "done" | "failed";

// --- errors ---------------------------------------------------------------

export type ErrorCode = "invalid_request" | "not_found" | "busy" | "internal_error";

export type ApiErrorBody = { error: { code: ErrorCode; message: string } };

// --- segments -------------------------------------------------------------

/** `tts` as the manifest stores it (dubbing/tts.py). */
export type SegmentTts = {
  clip: string;
  dur: number;
  tries: number;
  /** word overlap between what was asked for and what the ASR heard, 0..1 */
  overlap: number;
  verify: "ok" | "soft" | "keep" | "failed" | string;
};

/** `place` as the manifest stores it (dubbing/timeline.py). */
export type SegmentPlace = {
  start: number;
  end: number;
  /** time-stretch factor actually applied */
  rate: number;
  /** seconds the placement slipped from the original start */
  drift: number;
  clip: string;
};

/** Per-segment synthesis overrides — `feat/tts-controls` owns the field list. */
export type TtsOpts = {
  seed?: number | null;
  greedy?: boolean | null;
  /** run-relative path to a reference wav that overrides the speaker's */
  ref?: string | null;
  style?: string | null;
};

/**
 * The playable audio for one segment, exactly as `Projects.enrich` writes it
 * (dubbing_app/projects.py). This is the server's shape and the only shape the
 * UI knows: an earlier draft of this file invented `place_clip_url` /
 * `tts_clip_url` / `source_clip_url`, the fixtures implemented *those*, and so
 * every fixture-backed test passed while A/B playback was dead against the
 * real server. One name, both sides.
 */
export type SegmentMedia = {
  /** `place.clip` — the file the mix actually uses, after time-fitting */
  play: string | null;
  /** `tts.clip` — the raw synthesis, before time-fitting */
  tts: string | null;
  /** this segment's window of the source audio, as a `#t=` media fragment */
  source: string | null;
  source_window?: [number, number] | null;
};

/**
 * A segment as `GET /segments` returns it: the manifest record plus the
 * enrichment the UI needs and the manifest deliberately does not store.
 */
export type Segment = {
  /** stable across re-segmentation — every API keys on this, never on `id` */
  uid: string;
  /** positional, renumbered on every re-segmentation; ordering and display only */
  id: number;
  start: number;
  end: number;
  speaker: string;
  /** source-language text */
  text: string;
  /** target-language text */
  text_en: string | null;
  /** pivot runs only: the English intermediate text_en was made from */
  text_mid?: string | null;
  keep: boolean;
  keep_reason: string | null;
  /**
   * Third-language keeps: what the span's speech actually is.
   *
   * Optional, because it is absent from 98% of the segments a real run serves —
   * `manifest.SEGMENT_KEYS` keeps it only where the classifier had something to
   * say. Declared non-optional, it was a guarantee the server does not make and
   * a `seg.lang` read that TypeScript swore could not be `undefined`.
   */
  lang?: string | null;
  /**
   * Advisory: what the language classifier heard over this span. It decides
   * nothing — it is what an editor reads to *suggest* a passthrough.
   */
  detected_lang?: string | null;
  /**
   * The user's own verdict about the span, as the pipeline stores it: true =
   * play the original, false = dub it, absent = decide automatically. Written
   * by `edit.set_keep` for a manual keep and honoured by a headless re-run
   * (`segments.apply_passthrough`), which is why a keep made here survives one.
   */
  passthrough?: boolean | null;
  src_lang?: string | null;
  tgt_lang?: string | null;
  tts_opts?: TtsOpts | null;
  /** fields the user edited by hand that a re-run must not overwrite */
  locked?: Record<string, boolean> | null;
  tts: SegmentTts | null;
  place: SegmentPlace | null;

  // --- server-side enrichment, not in the manifest ---
  /** every playable URL for this segment; see SegmentMedia */
  media?: SegmentMedia | null;
  /** the verification verdict from clips/<hash>.json */
  verify?: SegmentVerify | null;
};

export type SegmentVerify = {
  /** what the ASR heard back from the generated clip */
  heard: string;
  /** word overlap with the requested text, 0..1 */
  overlap: number;
  ok: boolean;
  tries?: number;
};

/**
 * Body of `PATCH /segments/{uid}` — mirrors the `dubbing/edit.py` setters.
 *
 * `null` means "not supplied", never "clear" (`app.py::PatchSegment`), which is
 * why nothing here is nullable: an omitted key is how a field is left alone.
 * The one way to *remove* a value is `src_lang`/`tgt_lang`, where the empty
 * string clears the override — a language tag has no other way out, and sending
 * `null` for it was a dead control.
 */
export type SegmentPatch = {
  text?: string;
  text_en?: string;
  keep?: boolean;
  /** Only read alongside `keep`; the server ignores it on its own. */
  keep_reason?: string;
  speaker?: string;
  start?: number;
  end?: number;
  /** "" clears the override and falls back to the run's language. */
  src_lang?: string;
  /** "" clears the override and falls back to the run's language. */
  tgt_lang?: string;
  tts_opts?: TtsOpts;
  locked?: Record<string, boolean>;
};

// --- projects -------------------------------------------------------------

export type ProjectSummary = {
  /** the run directory name — the project's id in every path */
  name: string;
  title: string;
  src_lang: string;
  tgt_lang: string;
  duration: number | null;
  stages: Partial<Record<Stage, StageStatus>>;
  mtime: number;
};

export type ProjectSource = {
  input: string;
  src_lang: string;
  tgt_lang: string;
  duration_limit: number | null;
  title: string;
  duration: number | null;
  transcript_origin: string;
  context?: string | null;
  genre?: string | null;
  register?: string | null;
};

export type SpeakerInfo = {
  dur: number;
  ref?: string | null;
  ref_span?: [number, number] | null;
  ref_noise?: number | null;
};

export type Report = {
  segments: number;
  dubbed: number;
  kept: number;
  keep_reasons: Record<string, number>;
  unaccounted: unknown[];
  verify: Record<string, number>;
  shortened: unknown[];
  drift: { max: number; mean: number; over_soft: number };
  speed: { max: number; compressed: number };
  uncovered_audible: { start: number; end: number; duration: number; rms: number }[];
};

export type ProjectDetail = {
  name: string;
  source: ProjectSource;
  speakers: Record<string, SpeakerInfo>;
  stages: Partial<Record<Stage, StageStatus>>;
  outputs: { preview?: string; dub_wav?: string; srt?: string } & Record<string, string>;
  report: Report | null;
};

/**
 * A waveform overview for one timeline lane.
 *
 * `peaks` is normalized to [0, 1] and `duration` is the audio's own length in
 * seconds — which is what makes the picture time-alignable, and why the lane
 * must scale it by `duration` rather than assuming it covers the run.
 */
export type PeaksFile = "source" | "dub";

export type Peaks = {
  file: PeaksFile;
  duration: number;
  peaks: number[];
};

export type CreateProjectRequest = {
  /** a local file path or a URL */
  source: string;
  src_lang: string;
  tgt_lang: string;
  duration?: number | null;
  name?: string | null;
  context?: string | null;
  genre?: "documentary" | "movie" | null;
  register?: "narration" | "dialogue" | null;
};

export type CreateProjectResponse = { name: string; job: Job };

// --- jobs -----------------------------------------------------------------

export type JobStatus = "queued" | "running" | "done" | "failed" | "cancelled";

export type JobKind = "run" | "retranslate" | "resynthesize" | "render";

export type Job = {
  id: string;
  project: string;
  kind: JobKind;
  status: JobStatus;
  /** 0..1, or null when the job has not reported any yet */
  progress: number | null;
  stage: Stage | null;
  message: string | null;
  error: string | null;
  /**
   * FIXTURE-ONLY. `jobs.Job.to_dict` serializes the uid list under `payload`,
   * never as a top-level `uids`, so against the real server this is `undefined`
   * every time and nothing in the app may branch on it. The fixture backend is
   * its only reader — it needs to know which segments to advance while it fakes
   * a run. Whether the server should promote it (or the UI read
   * `payload.uids`) is a server-side decision, not a gap to paper over here.
   */
  uids?: string[];
};

// --- the NDJSON event stream ---------------------------------------------

export type StageEvent = {
  type: "stage";
  stage: Stage;
  status: StageStatus;
  progress?: number;
  message?: string;
};

export type SegmentEvent = {
  type: "segment";
  uid: string;
  field: "text_en" | "tts" | "place" | string;
  status: StageStatus;
};

export type JobEvent = {
  type: "job";
  id: string;
  status: JobStatus;
  error?: string | null;
  kind?: JobKind;
  progress?: number | null;
};

export type LogEvent = {
  type: "log";
  level: "debug" | "info" | "warn" | "error";
  message: string;
};

/** A heartbeat keeps proxies and sleeping laptops from dropping the stream. */
export type HeartbeatEvent = { type: "heartbeat"; t?: number };

export type StudioEvent =
  | StageEvent
  | SegmentEvent
  | JobEvent
  | LogEvent
  | HeartbeatEvent;

export type Health = { status: "ok"; version: string };

// --- first-run readiness --------------------------------------------------

/**
 * One row of `GET /api/setup`: a fast filesystem check the server can answer
 * without loading anything — ffmpeg, sox, the HF token, each model directory,
 * free disk. `detail` is the sentence shown under the label: a size when the
 * check passes, what to install when it does not.
 */
export type SetupCheck = {
  id: string;
  label: string;
  ok: boolean;
  detail: string;
};

/** `ok` is the whole checklist's verdict — false means the app is not usable. */
export type SetupStatus = { ok: boolean; checks: SetupCheck[] };
