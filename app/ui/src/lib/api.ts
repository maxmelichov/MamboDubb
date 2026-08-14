/**
 * The one place the UI touches the network. Everything above this file works in
 * terms of the types in ./types, so the server can be swapped for fixtures by
 * flipping VITE_USE_FIXTURES=1 and nothing else changes.
 */

import { readNdjson } from "./ndjson";
import * as fixtures from "./fixtures";
import { serverBaseUrl } from "./desktop";
import { isToneUrl, resolveToneUrl } from "./tone";
import type {
  ApiErrorBody,
  CreateProjectRequest,
  CreateProjectResponse,
  ErrorCode,
  Health,
  Job,
  ProjectDetail,
  ProjectSummary,
  Segment,
  SegmentPatch,
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

/** The server's uniform error body, as a throwable. */
export class ApiError extends Error {
  readonly code: ErrorCode;
  readonly status: number;

  constructor(code: ErrorCode, message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
  }

  /** `busy` means the one job slot is taken — the caller can retry, not fail. */
  get isBusy(): boolean {
    return this.code === "busy";
  }
}

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

// --- the surface ----------------------------------------------------------

export const api = {
  health(): Promise<Health> {
    if (USE_FIXTURES) return fixtures.health();
    return request<Health>("/health");
  },

  /**
   * First-run readiness. Fast filesystem checks only — the server never loads a
   * model to answer this, so it is cheap enough to call on every boot.
   */
  setup(): Promise<SetupStatus> {
    if (USE_FIXTURES) return fixtures.setup();
    return request<SetupStatus>("/api/setup");
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
    };
    return request<Wire>(`/api/projects/${encodeURIComponent(name)}`).then((r) => ({
      name: r.name,
      source: r.manifest.source,
      speakers: r.manifest.speakers ?? {},
      stages: r.stages,
      outputs: r.manifest.outputs ?? {},
      report: r.report,
    }));
  },

  getSegments(name: string): Promise<Segment[]> {
    if (USE_FIXTURES) return fixtures.getSegments(name);
    return request<{ segments: Segment[] }>(
      `/api/projects/${encodeURIComponent(name)}/segments`,
    ).then((r) => r.segments);
  },

  /** Every no-model edit. Must stay responsive while a job runs. */
  patchSegment(name: string, uid: string, patch: SegmentPatch): Promise<Segment> {
    if (USE_FIXTURES) return fixtures.patchSegment(name, uid, patch);
    return request<{ segment: Segment }>(
      `/api/projects/${encodeURIComponent(name)}/segments/${encodeURIComponent(uid)}`,
      { method: "PATCH", body: JSON.stringify(patch) },
    ).then((r) => r.segment);
  },

  /**
   * Returns the whole list because `id` renumbers on every structural edit.
   * The server only returns the affected segments, so re-fetch: correctness
   * over one round trip on an explicit, rare user action.
   */
  splitSegment(name: string, uid: string, at: number): Promise<Segment[]> {
    if (USE_FIXTURES) return fixtures.splitSegment(name, uid, at);
    return request<unknown>(
      `/api/projects/${encodeURIComponent(name)}/segments/${encodeURIComponent(uid)}/split`,
      json({ at }),
    ).then(() => api.getSegments(name));
  },

  mergeSegments(name: string, uid: string, uidB: string): Promise<Segment[]> {
    if (USE_FIXTURES) return fixtures.mergeSegments(name, uid, uidB);
    return request<unknown>(
      `/api/projects/${encodeURIComponent(name)}/segments/${encodeURIComponent(uid)}/merge`,
      json({ uid: uidB }),
    ).then(() => api.getSegments(name));
  },

  retranslate(name: string, uids: string[]): Promise<Job> {
    if (USE_FIXTURES) return fixtures.enqueue(name, "retranslate", uids);
    return request<{ job: Job }>(
      `/api/projects/${encodeURIComponent(name)}/retranslate`, json({ uids }),
    ).then((r) => r.job);
  },

  resynthesize(name: string, uids: string[]): Promise<Job> {
    if (USE_FIXTURES) return fixtures.enqueue(name, "resynthesize", uids);
    return request<{ job: Job }>(
      `/api/projects/${encodeURIComponent(name)}/resynthesize`, json({ uids }),
    ).then((r) => r.job);
  },

  /** timeline → mix → report. A full libx264 re-encode, so never an autosave. */
  render(name: string): Promise<Job> {
    if (USE_FIXTURES) return fixtures.enqueue(name, "render", []);
    return request<{ job: Job }>(
      `/api/projects/${encodeURIComponent(name)}/render`, json({}),
    ).then((r) => r.job);
  },

  listJobs(): Promise<Job[]> {
    if (USE_FIXTURES) return fixtures.listJobs();
    return request<{ jobs: Job[] }>("/api/jobs").then((r) => r.jobs);
  },

  cancelJob(id: string): Promise<void> {
    if (USE_FIXTURES) return fixtures.cancelJob(id);
    return request<void>(`/api/jobs/${encodeURIComponent(id)}`, { method: "DELETE" });
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
   * Never rejects — a dropped stream is a UI state, not an exception.
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
