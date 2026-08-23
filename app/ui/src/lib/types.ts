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
  /**
   * "accepted" sits between "ok" and "soft": the clone cleared the accept floor
   * but never reached `tts.CLONE_GOOD_OVERLAP`, and every clone reference had
   * already been tried. A real dub, and not a clean one.
   */
  verify: "ok" | "accepted" | "soft" | "keep" | "failed" | string;
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

/** Per-segment synthesis overrides `feat/tts-controls` owns the field list. */
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
  /** `place.clip` the file the mix actually uses, after time-fitting */
  play: string | null;
  /** `tts.clip` the raw synthesis, before time-fitting */
  tts: string | null;
  /** this segment's window of the source audio, as a `#t=` media fragment */
  source: string | null;
  source_window?: [number, number] | null;
  /**
   * True when a dub-wanted line's placed clip is the original-audio slice —
   * tts's universal fallback for a line it could not voice (no translation,
   * failed synthesis). Its "dub" then IS the original, and the row must say
   * so instead of letting Orig and Dub play identical audio unexplained.
   */
  fallback?: boolean;
};

/**
 * A segment as `GET /segments` returns it: the manifest record plus the
 * enrichment the UI needs and the manifest deliberately does not store.
 */
export type Segment = {
  /** stable across re-segmentation every API keys on this, never on `id` */
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
   * nothing it is what an editor reads to *suggest* a passthrough.
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
 * Body of `PATCH /segments/{uid}` mirrors the `dubbing/edit.py` setters.
 *
 * `null` means "not supplied", never "clear" (`app.py::PatchSegment`), which is
 * why nothing here is nullable: an omitted key is how a field is left alone.
 * The one way to *remove* a value is `src_lang`/`tgt_lang`, where the empty
 * string clears the override a language tag has no other way out, and sending
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

/**
 * `POST /api/projects/{name}/segments` a span the pipeline missed, claimed by
 * hand.
 *
 * Not a `SegmentPatch`: every field here is required to exist (the server
 * refuses a segment with no text there is nothing to translate or speak) and
 * the span must be *free*, so an overlap with a neighbour comes back as a 400
 * rather than being clamped. `speaker` omitted inherits the nearest segment's,
 * which is right for a line dropped out of the middle of a passage.
 */
export type NewSegment = {
  start: number;
  end: number;
  text: string;
  speaker?: string;
};

// --- projects -------------------------------------------------------------

export type ProjectSummary = {
  /** the run directory name the project's id in every path */
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
  /**
   * The manifest has moved on since these numbers were counted.
   *
   * Added by the server on read (`Projects.report`), never stored: an edit
   * changes no stage parameter, so nothing in report.json itself could tell a
   * current report from one the manifest passed hours ago. The numbers are
   * served either way they were true of the run they described and this is
   * what lets the UI caption them instead of presenting them as now.
   */
  stale?: boolean;
};

export type ProjectDetail = {
  name: string;
  source: ProjectSource;
  speakers: Record<string, SpeakerInfo>;
  stages: Partial<Record<Stage, StageStatus>>;
  outputs: { preview?: string; dub_wav?: string; srt?: string } & Record<string, string>;
  report: Report | null;
  /**
   * This project's jobs, oldest first the server has always sent them and the
   * client threw them away.
   *
   * They are the only record of *why* a run stopped. The event stream replays
   * nothing terminal (`app.py::project_events`, deliberately a failure
   * replayed on every reconnect resurrects an error bar dismissed an hour ago),
   * so after a reload the most recent failed job's `error` and `stage` exist
   * here and nowhere else. Without them "The run stopped at fetch" is all the
   * editor can say, and the sentence the user needs is the one after it.
   */
  jobs: Job[];
  render: RenderState;
};

/**
 * Body of `PATCH /api/projects/{name}` the run options that are still a
 * decision after the run has started.
 *
 * These three are inputs to the *translator*, so nothing already fetched,
 * transcribed or segmented depends on them and they take effect the next time
 * translation runs. The source and the language pair are deliberately not here:
 * changing either invalidates fetch and everything after it, which is a new
 * project wearing an old project's name.
 *
 * `context: ""` clears the note the one field with a way to be removed.
 */
export type ProjectOptionsPatch = {
  context?: string;
  genre?: "documentary" | "movie";
  register?: "narration" | "dialogue";
};

/**
 * What `preview.mp4` is a render of.
 *
 * `report.stale` says the *numbers* moved on; this says the *video* did, and the
 * video is what the user is about to watch. Without it a mix from before a dozen
 * corrections looked exactly like a fresh one, and the only way to find out was
 * to watch five minutes of it.
 */
export type RenderState = {
  /** epoch seconds, or null when the run predates the stamp or was never mixed */
  at: number | null;
  stale: boolean;
  /**
   * How many lines differ from the ones that render was made of. Exact, not an
   * estimate but `0` when there is no stamp to compare against, so never phrase
   * "N lines changed" off a render whose `at` is null.
   */
  changed: number;
};

/**
 * A waveform overview for one timeline lane.
 *
 * `peaks` is normalized to [0, 1] and `duration` is the audio's own length in
 * seconds which is what makes the picture time-alignable, and why the lane
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
  /**
   * Where the transcript comes from. `auto` (the pipeline's default) uses the
   * captions when the fetch found some and ASR otherwise; `captions` and `asr`
   * force one. It was accepted by the server and unreachable from the screen,
   * so a user who knew the auto-captions were garbage had no way to say so.
   *
   * `file` is the fourth answer, for the user who already has a transcript: the
   * words come from `captions` below and nowhere else, and a file that will not
   * parse is a refusal rather than a silent fall back to the ASR.
   */
  transcript?: "auto" | "captions" | "asr" | "file" | null;
  /**
   * An absolute path to a transcript the user already has .srt, .vtt or .json3,
   * the three formats that carry timestamps. There is no forced aligner in the
   * pipeline, so plain text cannot be placed in time and is refused.
   *
   * A path, not an upload: the desktop shell's picker returns real paths and the
   * server reads the file itself, exactly as it does for `source`. A browser tab
   * can only report a file's *name*, which is why the screen tells the user to
   * paste the full path there.
   */
  captions?: string | null;
  /**
   * Translate and voice speech that is in neither the source nor the target
   * language. Off by default, which is the pipeline's own default: a third
   * language is KEPT played as recorded, subtitled unless the run opts in.
   */
  dub_foreign?: boolean;
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
   * The segments this job is about; `[]` for a whole-run job.
   *
   * Real now: `jobs.Job.to_dict` hoists it out of the payload and the job frames
   * carry it too, live and replayed. It used to be fixture-only, which is why the
   * app marked busy rows from a local list that unrelated completions wiped after
   * about 100 ms. `pendingUids` is derived from these instead see `useProject`.
   */
  uids: string[];
  /**
   * The gesture this job came from, shared with the jobs enqueued beside it.
   *
   * "Dub these 27" is one decision that becomes a retranslate and a resynthesize.
   * `null` for a job that was its own gesture. What the Cancel dialog offers
   * "the whole batch" for, and what `DELETE /api/jobs/{id}?batch=1` acts on.
   */
  batch: string | null;
};

/** A job the user can still stop: it has not finished, failed, or been cancelled. */
export function isPending(job: Job): boolean {
  return job.status === "queued" || job.status === "running";
}

// --- the NDJSON event stream ---------------------------------------------

/**
 * `true` on prelude frames and on nothing else.
 *
 * The nine replayed stage frames are a snapshot of all nine stages at once, not a
 * progression, so a client that pins its display to the last stage frame it saw
 * reads the ninth as the stage the run is in which is how the editor came to
 * announce "Running report · 100%" the instant a re-voice started.
 */
type Replayed = { replay?: boolean };

export type StageEvent = Replayed & {
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

export type JobEvent = Replayed & {
  type: "job";
  id: string;
  status: JobStatus;
  error?: string | null;
  kind?: JobKind;
  progress?: number | null;
  uids?: string[];
  batch?: string | null;
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
 * without loading anything ffmpeg, sox, the HF token, each model directory,
 * free disk. `detail` is the sentence shown under the label: a size when the
 * check passes, what to install when it does not.
 */
export type SetupSeverity = "blocking" | "degrades" | "optional";

export type SetupCheck = {
  id: string;
  label: string;
  ok: boolean;
  detail: string;
  /**
   * What a failure of this check actually costs `blocking` (the run fails),
   * `degrades` (the run works and is worse), `optional` (irrelevant until you
   * ask for it). The screen showed a single red MISSING for all three, which
   * made a gated HF token look like a broken install and a Korean checkpoint
   * look like a reason not to start.
   *
   * Optional on the type, because a server older than this field sends rows
   * without it; `severityOf` treats those as blocking-if-required, which is the
   * only reading that cannot understate a failure.
   */
  severity?: SetupSeverity;
  /**
   * Blocking rows only: the pipeline stage this check's absence kills, so the
   * screen can say *where* a run would stop rather than that it would.
   */
  stage?: Stage | null;
  /** `severity === "blocking"`, as the server has always sent it. */
  required?: boolean;
  /**
   * The server can fix this one itself, so the row may offer a button. It is a
   * server flag rather than a list in the UI on purpose: a hardcoded list here
   * drifts from `install.INSTALLERS` (tools) and `setup.model_downloads()`
   * (hub snapshots) there, and the first symptom is a button whose POST is a
   * 400.
   */
  installable?: boolean;
  /** Where this check looked a binary's path, a model's directory. */
  path?: string | null;
  /** Model rows: bytes on disk when present, 0 when missing. */
  bytes?: number;
  /**
   * Downloadable model rows only: the hub repo the app would fetch. Its
   * presence is what turns the row's button from "Install" into "Download"
   * a tool is seconds and a model is gigabytes, and the label has to say
   * which before the click.
   */
  hub?: string;
  /**
   * Downloadable model rows only: the download's size, measured from real
   * installs. Approximate on purpose good for a button label ("~9.7 GB")
   * and a progress denominator, never for accounting.
   */
  download_bytes?: number;
  /**
   * The `low_vram` row only: whether this machine will load the small
   * translator weights. Not the same fact as `ok` that row is a setting, and
   * a setting is never a failure, so it is always green in either position.
   */
  enabled?: boolean;
  /**
   * Where a setting-shaped row got its answer. `"env"` is a variable exported
   * in the server's own environment, which the app cannot rewrite; `"env_file"`
   * is the workspace `.env`, which it can; `"auto"` is nobody having said, so
   * the machine's memory decided. The control has to know: offering a switch
   * against an `"env"` value would be offering a change that does not happen.
   */
  source?: "env" | "env_file" | "auto" | null;
};

/** `ok` is the whole checklist's verdict false means the app is not usable. */
export type SetupStatus = { ok: boolean; checks: SetupCheck[] };

/**
 * `GET|POST /api/setup/install` the one install slot.
 *
 * Polled, not streamed: setup has no project and therefore no event stream, and
 * a `brew install` is minutes long, so the last lines of output plus a 2 s poll
 * is the entire progress design. `check` is a freshly probed `SetupCheck`, filled
 * in when the process exits the exit code is a claim about the package
 * manager, not about this machine's PATH.
 */
export type SetupInstall = {
  running: boolean;
  id: string | null;
  ok: boolean | null;
  error: string | null;
  tail: string[];
  check: SetupCheck | null;
  /** Epoch seconds, server clock. Null before anything has ever run. */
  started?: number | null;
  finished?: number | null;
  /**
   * Model downloads only: the two numbers behind a real progress bar.
   * `bytes_done` is the target directory's size on disk, re-walked each poll
   * the one progress source that cannot disagree with reality and
   * `bytes_total` is the table's estimate, so done/total can overshoot 100%
   * by a little. A bar clamps; accounting never asks. Absent entirely while
   * the slot holds a tool install, which is how the row knows to show the
   * tail line instead.
   */
  bytes_done?: number;
  bytes_total?: number | null;
};
