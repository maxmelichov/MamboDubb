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
  /** third-language keeps: what the span's speech actually is */
  lang: string | null;
  src_lang?: string | null;
  tgt_lang?: string | null;
  tts_opts?: TtsOpts | null;
  /** fields the user edited by hand that a re-run must not overwrite */
  locked?: Record<string, boolean> | null;
  tts: SegmentTts | null;
  place: SegmentPlace | null;

  // --- server-side enrichment, not in the manifest ---
  /** absolute URL for `place.clip` — the audio that is actually in the mix */
  place_clip_url?: string | null;
  /** absolute URL for `tts.clip` — the raw synthesis before time-fitting */
  tts_clip_url?: string | null;
  /** absolute URL for this segment's window of the original source audio */
  source_clip_url?: string | null;
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

/** Body of `PATCH /segments/{uid}` — mirrors the `dubbing/edit.py` setters. */
export type SegmentPatch = {
  text?: string;
  text_en?: string;
  keep?: boolean;
  keep_reason?: string;
  speaker?: string;
  start?: number;
  end?: number;
  src_lang?: string | null;
  tgt_lang?: string | null;
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
  /** for retranslate / resynthesize */
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
