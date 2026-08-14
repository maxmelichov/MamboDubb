/**
 * The project store: one state object and a set of actions, following
 * MamboRambo's deliberately plain approach. No reducer, no state library.
 *
 * The split that matters is in the actions, not the state. No-model edits
 * (`patch`) apply to local state first and are saved in the background, so the
 * editor stays responsive while a job runs. Model actions (`retranslate`,
 * `resynthesize`, `render`) return a job and are reported through the event
 * stream, because only one can run at a time.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, api } from "./api";
import type {
  Job,
  LogEvent,
  ProjectDetail,
  Segment,
  SegmentPatch,
  Stage,
  StageStatus,
} from "./types";

export type StageProgress = {
  stage: Stage;
  status: StageStatus;
  progress: number | null;
  message: string | null;
};

export type ProjectState = {
  name: string;
  project: ProjectDetail | null;
  segments: Segment[];
  selectedUid: string | null;
  loading: boolean;
  error: string | null;
  /** true while the event stream is connected; false shows the reconnect strip */
  connected: boolean;
  jobs: Job[];
  stage: StageProgress | null;
  /** uids with a model action in flight, so the inspector can show a spinner */
  busyUids: string[];
  log: LogEvent[];
};

export type ProjectActions = {
  select: (uid: string | null) => void;
  patch: (uid: string, patch: SegmentPatch) => Promise<void>;
  split: (uid: string, at: number) => Promise<void>;
  merge: (uid: string, uidB: string) => Promise<void>;
  retranslate: (uids: string[]) => Promise<void>;
  resynthesize: (uids: string[]) => Promise<void>;
  render: () => Promise<void>;
  cancel: (id: string) => Promise<void>;
  reload: () => Promise<void>;
  dismissError: () => void;
};

const MAX_LOG = 200;

export function useProject(name: string): [ProjectState, ProjectActions] {
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [selectedUid, setSelectedUid] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [stage, setStage] = useState<StageProgress | null>(null);
  const [busyUids, setBusyUids] = useState<string[]>([]);
  const [log, setLog] = useState<LogEvent[]>([]);

  // Refs so the event handler stays stable across renders.
  const nameRef = useRef(name);
  nameRef.current = name;

  const load = useCallback(async () => {
    try {
      const [detail, segs] = await Promise.all([
        api.getProject(nameRef.current),
        api.getSegments(nameRef.current),
      ]);
      setProject(detail);
      setSegments(segs);
      setError(null);
    } catch (err) {
      setError(describe(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    setLoading(true);
    void load();
  }, [load, name]);

  // The event stream. It reconnects on its own; the promise settles on abort.
  useEffect(() => {
    const controller = new AbortController();
    let refreshTimer: number | undefined;

    /** Segment-level frames say *that* something changed, not what to. */
    const scheduleRefresh = () => {
      window.clearTimeout(refreshTimer);
      refreshTimer = window.setTimeout(() => void load(), 250);
    };

    void api.events(
      name,
      controller.signal,
      (event) => {
        switch (event.type) {
          case "stage":
            setStage({
              stage: event.stage,
              status: event.status,
              progress: event.progress ?? null,
              message: event.message ?? null,
            });
            if (event.status === "done" || event.status === "failed") scheduleRefresh();
            break;
          case "segment":
            setBusyUids((current) => current.filter((uid) => uid !== event.uid));
            scheduleRefresh();
            break;
          case "job":
            setJobs((current) => {
              const next = current.map((job) =>
                job.id === event.id
                  ? { ...job, status: event.status, error: event.error ?? null }
                  : job,
              );
              return next.some((job) => job.id === event.id)
                ? next
                : [
                    ...next,
                    {
                      id: event.id,
                      project: name,
                      kind: event.kind ?? "run",
                      status: event.status,
                      progress: event.progress ?? null,
                      stage: null,
                      message: null,
                      error: event.error ?? null,
                    } satisfies Job,
                  ];
            });
            if (event.status === "done" || event.status === "failed") {
              setStage(null);
              setBusyUids([]);
              scheduleRefresh();
            }
            if (event.status === "failed" && event.error) setError(event.error);
            break;
          case "log":
            setLog((current) => [...current, event].slice(-MAX_LOG));
            break;
          case "heartbeat":
            break;
        }
      },
      setConnected,
    );

    return () => {
      window.clearTimeout(refreshTimer);
      controller.abort();
    };
  }, [load, name]);

  const patch = useCallback(
    async (uid: string, body: SegmentPatch) => {
      // Optimistic: the point of a no-model edit is that it is instant.
      const before = segments;
      setSegments((current) =>
        current.map((seg) => (seg.uid === uid ? { ...seg, ...body } : seg)),
      );
      try {
        const updated = await api.patchSegment(name, uid, body);
        setSegments((current) => current.map((seg) => (seg.uid === uid ? updated : seg)));
      } catch (err) {
        setSegments(before);
        setError(describe(err));
      }
    },
    [name, segments],
  );

  const structural = useCallback(
    async (run: () => Promise<Segment[]>) => {
      try {
        setSegments(await run());
      } catch (err) {
        setError(describe(err));
      }
    },
    [],
  );

  const split = useCallback(
    (uid: string, at: number) => structural(() => api.splitSegment(name, uid, at)),
    [name, structural],
  );

  const merge = useCallback(
    (uid: string, uidB: string) => structural(() => api.mergeSegments(name, uid, uidB)),
    [name, structural],
  );

  const submit = useCallback(
    async (uids: string[], run: () => Promise<Job>) => {
      setBusyUids((current) => [...new Set([...current, ...uids])]);
      try {
        const job = await run();
        setJobs((current) => [...current.filter((j) => j.id !== job.id), job]);
      } catch (err) {
        setBusyUids((current) => current.filter((uid) => !uids.includes(uid)));
        setError(describe(err));
      }
    },
    [],
  );

  const actions = useMemo<ProjectActions>(
    () => ({
      select: setSelectedUid,
      patch,
      split,
      merge,
      retranslate: (uids) => submit(uids, () => api.retranslate(name, uids)),
      resynthesize: (uids) => submit(uids, () => api.resynthesize(name, uids)),
      render: () => submit([], () => api.render(name)),
      cancel: async (id) => {
        try {
          await api.cancelJob(id);
          setJobs((current) =>
            current.map((job) => (job.id === id ? { ...job, status: "cancelled" } : job)),
          );
        } catch (err) {
          setError(describe(err));
        }
      },
      reload: load,
      dismissError: () => setError(null),
    }),
    [load, merge, name, patch, split, submit],
  );

  const state: ProjectState = {
    name,
    project,
    segments,
    selectedUid,
    loading,
    error,
    connected,
    jobs,
    stage,
    busyUids,
    log,
  };

  return [state, actions];
}

export function activeJob(jobs: Job[]): Job | null {
  return jobs.find((job) => job.status === "running") ?? jobs.find((job) => job.status === "queued") ?? null;
}

function describe(error: unknown): string {
  if (error instanceof ApiError) {
    return error.isBusy
      ? "A job is already running — this one is queued behind it."
      : error.message;
  }
  return String(error);
}
