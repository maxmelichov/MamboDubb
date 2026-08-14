/**
 * The one place the UI touches the network. Everything above this file works in
 * terms of the types in ./types, so the server can be swapped for fixtures by
 * flipping VITE_USE_FIXTURES=1 and nothing else changes.
 */

import { readNdjson } from "./ndjson";
import * as fixtures from "./fixtures";
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
  StudioEvent,
} from "./types";

export const USE_FIXTURES = import.meta.env.VITE_USE_FIXTURES === "1";

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
    response = await fetch(path, {
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

  listProjects(): Promise<ProjectSummary[]> {
    if (USE_FIXTURES) return fixtures.listProjects();
    return request<ProjectSummary[]>("/api/projects");
  },

  createProject(body: CreateProjectRequest): Promise<CreateProjectResponse> {
    if (USE_FIXTURES) return fixtures.createProject(body);
    return request<CreateProjectResponse>("/api/projects", json(body));
  },

  getProject(name: string): Promise<ProjectDetail> {
    if (USE_FIXTURES) return fixtures.getProject(name);
    return request<ProjectDetail>(`/api/projects/${encodeURIComponent(name)}`);
  },

  getSegments(name: string): Promise<Segment[]> {
    if (USE_FIXTURES) return fixtures.getSegments(name);
    return request<Segment[]>(`/api/projects/${encodeURIComponent(name)}/segments`);
  },

  /** Every no-model edit. Must stay responsive while a job runs. */
  patchSegment(name: string, uid: string, patch: SegmentPatch): Promise<Segment> {
    if (USE_FIXTURES) return fixtures.patchSegment(name, uid, patch);
    return request<Segment>(
      `/api/projects/${encodeURIComponent(name)}/segments/${encodeURIComponent(uid)}`,
      { method: "PATCH", body: JSON.stringify(patch) },
    );
  },

  /** Returns the whole list because `id` renumbers on every structural edit. */
  splitSegment(name: string, uid: string, at: number): Promise<Segment[]> {
    if (USE_FIXTURES) return fixtures.splitSegment(name, uid, at);
    return request<Segment[]>(
      `/api/projects/${encodeURIComponent(name)}/segments/${encodeURIComponent(uid)}/split`,
      json({ at }),
    );
  },

  mergeSegments(name: string, uid: string, uidB: string): Promise<Segment[]> {
    if (USE_FIXTURES) return fixtures.mergeSegments(name, uid, uidB);
    return request<Segment[]>(
      `/api/projects/${encodeURIComponent(name)}/segments/${encodeURIComponent(uid)}/merge`,
      json({ uid_b: uidB }),
    );
  },

  retranslate(name: string, uids: string[]): Promise<Job> {
    if (USE_FIXTURES) return fixtures.enqueue(name, "retranslate", uids);
    return request<Job>(`/api/projects/${encodeURIComponent(name)}/retranslate`, json({ uids }));
  },

  resynthesize(name: string, uids: string[]): Promise<Job> {
    if (USE_FIXTURES) return fixtures.enqueue(name, "resynthesize", uids);
    return request<Job>(`/api/projects/${encodeURIComponent(name)}/resynthesize`, json({ uids }));
  },

  /** timeline → mix → report. A full libx264 re-encode, so never an autosave. */
  render(name: string): Promise<Job> {
    if (USE_FIXTURES) return fixtures.enqueue(name, "render", []);
    return request<Job>(`/api/projects/${encodeURIComponent(name)}/render`, json({}));
  },

  listJobs(): Promise<Job[]> {
    if (USE_FIXTURES) return fixtures.listJobs();
    return request<Job[]>("/api/jobs");
  },

  cancelJob(id: string): Promise<void> {
    if (USE_FIXTURES) return fixtures.cancelJob(id);
    return request<void>(`/api/jobs/${encodeURIComponent(id)}`, { method: "DELETE" });
  },

  /**
   * Make a URL from `GET /segments` playable. Identity against the real server;
   * in fixture mode it materializes the synthetic clip on first use.
   */
  audioUrl(url: string | null | undefined): string | null {
    if (!url) return null;
    if (isToneUrl(url)) return resolveToneUrl(url);
    return url;
  },

  /** Range-capable file serving from the run dir. */
  mediaUrl(name: string, path: string): string {
    if (USE_FIXTURES) return fixtures.mediaUrl(name, path);
    return `/media/${encodeURIComponent(name)}/${path.split("/").map(encodeURIComponent).join("/")}`;
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
    return readNdjson<StudioEvent>(`/api/projects/${encodeURIComponent(name)}/events`, signal, {
      onMessage,
      onOpen: () => onConnectionChange?.(true),
      onClose: () => onConnectionChange?.(false),
    });
  },
};

export type Api = typeof api;
