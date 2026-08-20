/**
 * The one place the UI touches the network. Everything above this file works in
 * terms of the types in ./types, so the server can be swapped for fixtures by
 * flipping VITE_USE_FIXTURES=1 and nothing else changes.
 */

import { readNdjson } from "./ndjson";
import { ApiError } from "./apiError";
import * as fixtures from "./fixtures";
import { serverBaseUrl } from "./desktop";
import { isToneUrl, resolveToneUrl } from "./tone";
import type {
  ApiErrorBody,
  CreateProjectRequest,
  CreateProjectResponse,
  Health,
  Job,
  NewSegment,
  Peaks,
  PeaksFile,
  ProjectDetail,
  ProjectOptionsPatch,
  ProjectSource,
  ProjectSummary,
  Segment,
  SegmentMedia,
  SegmentPatch,
  SetupInstall,
  SetupStatus,
  StudioEvent,
} from "./types";

export const USE_FIXTURES = import.meta.env.VITE_USE_FIXTURES === "1";

// --- where the server is --------------------------------------------------

/**
 * Every path this file builds goes through `url()`, which is empty-prefixed in
 * a browser (the server serves the UI, so /api and /media are same-origin) and
 * prefixed with the sidecar's `http://127.0.0.1:<port>` inside the desktop
 * shell, whose origin is the Tauri asset protocol instead.
 *
 * It is a plain string rather than a promise because `mediaUrl` is called
 * during render and a `<video src>` cannot await. `initApiBase()` resolves it
 * once at boot, before anything is rendered.
 */
let base = "";

/** Resolve the server's base URL. Call once, before the first render. */
export async function initApiBase(): Promise<string> {
  if (USE_FIXTURES) return base;
  base = (await serverBaseUrl()) ?? "";
  return base;
}

/** The prefix in force, for display ("" means same-origin). */
export function apiBase(): string {
  return base;
}

const url = (path: string): string => base + path;

/**
 * The server's uniform error body, as a throwable. Defined in `./apiError` so
 * the fixture backend can throw the same class without importing this module
 * (which imports it) re-exported here because this is where callers look.
 */
export { ApiError };

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url(path), {
      ...init,
      headers: {
        ...(init?.body ? { "content-type": "application/json" } : {}),
        accept: "application/json",
        ...init?.headers,
      },
    });
  } catch (error) {
    throw new ApiError("internal_error", `cannot reach the studio server: ${String(error)}`, 0);
  }

  if (response.status === 204) return undefined as T;

  const text = await response.text();
  const body: unknown = text ? safeParse(text) : null;

  if (!response.ok) {
    const err = (body as ApiErrorBody | null)?.error;
    throw new ApiError(
      err?.code ?? "internal_error",
      err?.message ?? `${response.status} ${response.statusText}`,
      response.status,
    );
  }
  /*
   * A 200 whose body is not JSON is somebody else answering.
   *
   * It happens: the SPA fallback of a static host, a captive portal, a stale
   * service worker anything that hands back `index.html` for `/api/…`. The
   * cast below is a lie in that case, and the lie surfaced three layers up as
   * "TypeError: Cannot read properties of null (reading 'name')" in the
   * editor's error bar, which tells the user nothing about what to do. Named
   * here, once, where the response is still in hand.
   */
  if (text && body === null) {
    throw new ApiError(
      "internal_error",
      `the studio server answered ${path} with something that is not JSON ` +
        "is the port serving the studio server, or something else?",
      response.status,
    );
  }
  return body as T;
}

function safeParse(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  body: JSON.stringify(body),
});

// --- the segment seam -----------------------------------------------------

const NO_MEDIA: SegmentMedia = { play: null, tts: null, source: null, source_window: null };

/**
 * Every segment enters the app through here.
 *
 * The one job is `media`: the server always sends it (`Projects.enrich`), but a
 * segment that arrives from an older server, a hand-written test or a partially
 * enriched response must still be a legal `Segment` rather than a crash inside
 * a play button. Filling the absence in *once*, at the seam, is what lets every
 * component above read `seg.media.play` without a guard and is where a future
 * rename of the server's field belongs, so it is one edit and not thirty.
 */
/**
 * A media path arrives server-relative (`/media/<run>/clip.wav#t=a,b`), which
 * only resolves in a browser where the server also serves the page. The desktop
 * shell's origin is the Tauri asset protocol, so a raw path handed to
 * `new Audio()` resolves against the wrong origin and every play button dies
 * silently. The base prefix is applied here, once, for the same reason `media`
 * itself is normalized here so no component ever has to know.
 */
function absMedia(path: string | null | undefined): string | null {
  if (!path) return null;
  return path.startsWith("/") ? base + path : path;
}

function adopt(seg: Segment): Segment {
  const m = seg.media;
  if (!m) return { ...seg, media: NO_MEDIA };
  return {
    ...seg,
    media: { ...m, play: absMedia(m.play), tts: absMedia(m.tts), source: absMedia(m.source) },
  };
}

const adoptAll = (segs: Segment[]): Segment[] => segs.map(adopt);

// --- the surface ----------------------------------------------------------

export const api = {
  health(): Promise<Health> {
    if (USE_FIXTURES) return fixtures.health();
    return request<Health>("/health");
  },

  /**
   * First-run readiness. Fast filesystem checks only the server never loads a
   * model to answer this, so it is cheap enough to call on every boot.
   */
  setup(): Promise<SetupStatus> {
    if (USE_FIXTURES) return fixtures.setup();
    return request<SetupStatus>("/api/setup");
  },

  /**
   * Install one missing tool or download one missing model. The body is an
   * *id* from the checklist, never a command the server maps it to a
   * hardcoded argv (`install.INSTALLERS`) or a hub snapshot
   * (`setup.model_downloads()`) and 400s anything else, so the worst a bad
   * call can do is fail.
   */
  startInstall(id: string): Promise<SetupInstall> {
    if (USE_FIXTURES) return fixtures.startInstall(id);
    return request<SetupInstall>("/api/setup/install", json({ id }));
  },

  /** Where that install got to. Polled; see `SetupInstall`. */
  installStatus(): Promise<SetupInstall> {
    if (USE_FIXTURES) return fixtures.installStatus();
    return request<SetupInstall>("/api/setup/install");
  },

  // The server speaks in envelopes ({"projects": [...]}, {"segment": {...}});
  // the components speak in values. The unwrapping lives here and only here —
  // the fixtures return bare values, which is how a mismatch stayed invisible
  // to every fixture-backed test and black-screened the first real launch.

  listProjects(): Promise<ProjectSummary[]> {
    if (USE_FIXTURES) return fixtures.listProjects();
    return request<{ projects: ProjectSummary[] }>("/api/projects").then((r) => r.projects);
  },

  createProject(body: CreateProjectRequest): Promise<CreateProjectResponse> {
    if (USE_FIXTURES) return fixtures.createProject(body);
    return request<{ project: { name: string }; job: Job }>("/api/projects", json(body))
      .then((r) => ({ name: r.project.name, job: r.job }));
  },

  getProject(name: string): Promise<ProjectDetail> {
    if (USE_FIXTURES) return fixtures.getProject(name);
    type Wire = {
      name: string;
      manifest: { source: ProjectDetail["source"]; speakers: ProjectDetail["speakers"];
                  outputs: ProjectDetail["outputs"] };
      stages: ProjectDetail["stages"];
      report: ProjectDetail["report"];
      jobs?: Job[];
      render?: ProjectDetail["render"];
    };
    return request<Wire>(`/api/projects/${encodeURIComponent(name)}`).then((r) => ({
      name: r.name,
      source: r.manifest.source,
      speakers: r.manifest.speakers ?? {},
      stages: r.stages,
      outputs: r.manifest.outputs ?? {},
      report: r.report,
      // Dropped on the floor until now, which is why a failed run could only be
      // described as "stopped at fetch": the stream replays nothing terminal, so
      // after a reload the error that stopped it is in this list and nowhere else.
      jobs: r.jobs ?? [],
      // A server too old to send it cannot prove the video is current, which is
      // the same answer `render_state` gives for a run with no stamp.
      render: r.render ?? { at: null, stale: true, changed: 0 },
    }));
  },

  /**
   * Change the run options that are still a decision: genre, register, context.
   *
   * Not a job and not a stage invalidation the server writes them and says so.
   * They reach the translator the next time it runs, which is the sentence the
   * editor puts under the control.
   */
  updateProject(name: string, patch: ProjectOptionsPatch): Promise<ProjectSource> {
    if (USE_FIXTURES) return fixtures.updateProject(name, patch);
    return request<{ source: ProjectSource }>(`/api/projects/${encodeURIComponent(name)}`, {
      method: "PATCH",
      body: JSON.stringify(patch),
    }).then((r) => r.source);
  },

  /**
   * Run this project a resume on one that has already run.
   *
   * The pipeline is stage-cached, so re-running is how it picks up where it
   * stopped; there is no second code path and there must not be. 409 while
   * anything is already in flight on the project.
   */
  resume(name: string): Promise<Job> {
    if (USE_FIXTURES) return fixtures.resume(name);
    return request<{ job: Job }>(
      `/api/projects/${encodeURIComponent(name)}/run`, json({}),
    ).then((r) => r.job);
  },

  getSegments(name: string): Promise<Segment[]> {
    if (USE_FIXTURES) return fixtures.getSegments(name).then(adoptAll);
    return request<{ segments: Segment[] }>(
      `/api/projects/${encodeURIComponent(name)}/segments`,
    ).then((r) => adoptAll(r.segments));
  },

  /** Every no-model edit. Must stay responsive while a job runs. */
  patchSegment(name: string, uid: string, patch: SegmentPatch): Promise<Segment> {
    if (USE_FIXTURES) return fixtures.patchSegment(name, uid, patch).then(adopt);
    return request<{ segment: Segment }>(
      `/api/projects/${encodeURIComponent(name)}/segments/${encodeURIComponent(uid)}`,
      { method: "PATCH", body: JSON.stringify(patch) },
    ).then((r) => adopt(r.segment));
  },

  /**
   * Returns the whole list because `id` renumbers on every structural edit.
   * The server only returns the affected segments, so re-fetch: correctness
   * over one round trip on an explicit, rare user action.
   */
  splitSegment(name: string, uid: string, at: number): Promise<Segment[]> {
    if (USE_FIXTURES) return fixtures.splitSegment(name, uid, at).then(adoptAll);
    return request<unknown>(
      `/api/projects/${encodeURIComponent(name)}/segments/${encodeURIComponent(uid)}/split`,
      json({ at }),
    ).then(() => api.getSegments(name));
  },

  /**
   * Claim an uncovered span as a new segment. Returns the whole list, for the
   * reason `splitSegment` does: `id` is positional and every one after the new
   * segment has just moved.
   *
   * The server refuses an overlap rather than clamping it, so a 400 here is a
   * real answer to show the user, not a validation the client should have done.
   */
  addSegment(name: string, body: NewSegment): Promise<Segment[]> {
    if (USE_FIXTURES) return fixtures.addSegment(name, body).then(adoptAll);
    return request<unknown>(
      `/api/projects/${encodeURIComponent(name)}/segments`, json(body),
    ).then(() => api.getSegments(name));
  },

  /**
   * Take a segment out of the dub entirely.
   *
   * Not "mute it": the mix plays the original vocals in every span no placement
   * claims, so the span goes back to sounding like a passage the pipeline never
   * detected. Deliberately playing the original *as a decision* is the keep
   * verdict, which is a `patchSegment({keep: true})` and keeps the segment.
   */
  removeSegment(name: string, uid: string): Promise<Segment[]> {
    if (USE_FIXTURES) return fixtures.removeSegment(name, uid).then(adoptAll);
    return request<unknown>(
      `/api/projects/${encodeURIComponent(name)}/segments/${encodeURIComponent(uid)}`,
      { method: "DELETE" },
    ).then(() => api.getSegments(name));
  },

  mergeSegments(name: string, uid: string, uidB: string): Promise<Segment[]> {
    if (USE_FIXTURES) return fixtures.mergeSegments(name, uid, uidB).then(adoptAll);
    return request<unknown>(
      `/api/projects/${encodeURIComponent(name)}/segments/${encodeURIComponent(uid)}/merge`,
      json({ uid: uidB }),
    ).then(() => api.getSegments(name));
  },

  /**
   * `batch` ties this job to the others the same gesture created, so one Cancel
   * can stop the decision instead of half of it. Only the client knows the two
   * POSTs came from one click, so the id is minted here and sent on both.
   */
  retranslate(name: string, uids: string[], batch?: string): Promise<Job> {
    if (USE_FIXTURES) return fixtures.enqueue(name, "retranslate", uids, batch);
    return request<{ job: Job }>(
      `/api/projects/${encodeURIComponent(name)}/retranslate`, json({ uids, batch }),
    ).then((r) => r.job);
  },

  resynthesize(name: string, uids: string[], batch?: string): Promise<Job> {
    if (USE_FIXTURES) return fixtures.enqueue(name, "resynthesize", uids, batch);
    return request<{ job: Job }>(
      `/api/projects/${encodeURIComponent(name)}/resynthesize`, json({ uids, batch }),
    ).then((r) => r.job);
  },

  /** timeline → mix → report. A full libx264 re-encode, so never an autosave. */
  render(name: string): Promise<Job> {
    if (USE_FIXTURES) return fixtures.enqueue(name, "render", []);
    return request<{ job: Job }>(
      `/api/projects/${encodeURIComponent(name)}/render`, json({}),
    ).then((r) => r.job);
  },

  /**
   * The waveform behind a timeline lane.
   *
   * `null` rather than a throw when the file is not there yet: `dub.wav` does
   * not exist until the mix stage has run, and a lane with no waveform is a
   * *state* of a young run, not a failure the same reasoning that keeps the
   * event stream from rejecting on a dropped connection. Everything else still
   * throws, because a 500 here is a bug and should not be silently drawn as
   * silence.
   *
   * `n` is buckets, clamped server-side to [16, 4000].
   *
   * The fixture call goes through the *same* catch, deliberately: it was routed
   * around it, so a missing `dub.wav` resolved to null against the server and
   * rejected in fixture mode the one branch this method exists for, tested in
   * the one mode the smoke test runs, backwards.
   */
  async peaks(name: string, file: PeaksFile, n: number): Promise<Peaks | null> {
    try {
      if (USE_FIXTURES) return await fixtures.peaks(name, file, n);
      return await request<Peaks>(
        `/api/projects/${encodeURIComponent(name)}/peaks?file=${file}&n=${Math.round(n)}`,
      );
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) return null;
      throw error;
    }
  },

  listJobs(): Promise<Job[]> {
    if (USE_FIXTURES) return fixtures.listJobs();
    return request<{ jobs: Job[] }>("/api/jobs").then((r) => r.jobs);
  },

  /** `batch: true` stops every job the same gesture queued, not just this one. */
  cancelJob(id: string, batch = false): Promise<void> {
    if (USE_FIXTURES) return fixtures.cancelJob(id, batch);
    return request<void>(
      `/api/jobs/${encodeURIComponent(id)}${batch ? "?batch=1" : ""}`,
      { method: "DELETE" },
    );
  },

  /**
   * Make a URL from `GET /segments` playable. Identity against the real server;
   * in fixture mode it materializes the synthetic clip on first use.
   */
  audioUrl(raw: string | null | undefined): string | null {
    if (!raw) return null;
    if (isToneUrl(raw)) return resolveToneUrl(raw);
    // The server calls these "absolute URLs"; a root-relative one still needs
    // the sidecar's origin in the shell, where the page is not on it.
    return raw.startsWith("/") ? url(raw) : raw;
  },

  /** Range-capable file serving from the run dir. */
  mediaUrl(name: string, path: string): string {
    if (USE_FIXTURES) return fixtures.mediaUrl(name, path);
    return url(
      `/media/${encodeURIComponent(name)}/${path.split("/").map(encodeURIComponent).join("/")}`,
    );
  },

  /**
   * Progress. Runs until `signal` aborts; reconnects on its own.
   * Never rejects a dropped stream is a UI state, not an exception.
   */
  events(
    name: string,
    signal: AbortSignal,
    onMessage: (event: StudioEvent) => void,
    onConnectionChange?: (open: boolean) => void,
  ): Promise<void> {
    if (USE_FIXTURES) return fixtures.events(name, signal, onMessage, onConnectionChange);
    return readNdjson<StudioEvent>(url(`/api/projects/${encodeURIComponent(name)}/events`), signal, {
      onMessage,
      onOpen: () => onConnectionChange?.(true),
      onClose: () => onConnectionChange?.(false),
    });
  },
};

export type Api = typeof api;
