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
import { applyPatch } from "./patch";
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
  /**
   * Returns the saved segment, or null if the save failed.
   *
   * The return value is not decoration: a verdict flip is a PATCH *and then* a
   * decision about what to queue, and the only honest input to that decision is
   * what the server says the segment became — `edit.set_keep` drops the line and
   * the clip on the way through, so "does this segment still have a translation"
   * cannot be answered from the patch that was sent.
   */
  patch: (uid: string, patch: SegmentPatch) => Promise<Segment | null>;
  split: (uid: string, at: number) => Promise<void>;
  merge: (uid: string, uidB: string) => Promise<void>;
  retranslate: (uids: string[]) => Promise<void>;
  resynthesize: (uids: string[]) => Promise<void>;
  render: () => Promise<void>;
  /**
   * Run the pipeline on this project again — a resume, on one that stopped.
   *
   * Goes through `submit` like every other model action, which is what puts the
   * refusal in the error bar: the server 409s while anything is already in
   * flight, and a resume that silently did nothing would look exactly like a
   * resume that worked.
   */
  resume: () => Promise<void>;
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
            /*
             * Cancelled is terminal, and it is terminal in the way that
             * matters most: the worker saves the partial work it did before it
             * stopped (`jobs`' journal), so a cancelled resynthesis leaves real
             * new clips on disk. Treating only done|failed as the end left the
             * spinners turning on every uid in the job and never re-read the
             * segments — the results were there and the screen never showed
             * them until the run was re-opened.
             */
            if (event.status === "done" || event.status === "failed" ||
                event.status === "cancelled") {
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
    async (uid: string, body: SegmentPatch): Promise<Segment | null> => {
      /*
       * Optimistic: the point of a no-model edit is that it is instant.
       *
       * `applyPatch` is `dubbing/edit.py`'s table, shared with the fixture
       * backend so there is one model of what a PATCH does and not three. The
       * server's answer overwrites this a moment later; this is only what the
       * row shows in between — and what it showed before was a ◆ Dubbed badge
       * over audio of a sentence the user had just replaced.
       */
      const before = segments;
      setSegments((current) =>
        current.map((seg) => (seg.uid === uid ? applyPatch(seg, body) : seg)),
      );
      try {
        const updated = await api.patchSegment(name, uid, body);
        setSegments((current) => current.map((seg) => (seg.uid === uid ? updated : seg)));
        return updated;
      } catch (err) {
        setSegments(before);
        setError(describe(err));
        return null;
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
      resume: () => submit([], () => api.resume(name)),
      // Optimistic in the same shape the event handler is: a cancel ends the
      // job, so the stage strip and the per-segment spinners end with it, and
      // whatever the worker saved on its way out is re-read. The `cancelled`
      // frame will say the same thing a moment later; this is for the moment
      // before it arrives, and for a stream that has dropped.
      cancel: async (id) => {
        try {
          await api.cancelJob(id);
          setJobs((current) =>
            current.map((job) => (job.id === id ? { ...job, status: "cancelled" } : job)),
          );
          setStage(null);
          setBusyUids([]);
          void load();
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

/**
 * The server's own words, never ours.
 *
 * A `busy` used to be rewritten as "A job is already running — this one is
 * queued behind it.", which is a sentence about something that never happened:
 * the model actions *do* queue and answer 202, and 409/busy is raised only for
 * the edits the server refuses outright while a job runs (a split, a merge, a
 * bounds change, a second install). Telling a user their split is queued when
 * it was rejected is worse than saying nothing — and the server's message
 * already names which edit and why.
 */
function describe(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  return String(error);
}
