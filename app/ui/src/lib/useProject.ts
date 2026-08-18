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
import { isPending } from "./types";
import type {
  Job,
  LogEvent,
  NewSegment,
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
  /**
   * Lines a model job is working on right now, derived from `jobs`.
   *
   * It used to be its own `busyUids` list, set when a request was sent and
   * cleared by two paths that both fired far too early: any `segment` frame
   * dropped that uid, and *any* job reaching a terminal state emptied the whole
   * list including a job about other lines entirely. Measured lifetime was
   * about 100 ms against a re-voice that runs for a minute per line, and a
   * reload cleared it outright. Derived, it lasts exactly as long as the work
   * does and survives reconnecting, because the jobs do.
   */
  pendingUids: string[];
  /** True once this session has made any edit (patch or structural). */
  edited: boolean;
  log: LogEvent[];
};

export type ProjectActions = {
  select: (uid: string | null) => void;
  /**
   * Returns the saved segment, or null if the save failed.
   *
   * The return value is not decoration: a verdict flip is a PATCH *and then* a
   * decision about what to queue, and the only honest input to that decision is
   * what the server says the segment became `edit.set_keep` drops the line and
   * the clip on the way through, so "does this segment still have a translation"
   * cannot be answered from the patch that was sent.
   */
  patch: (uid: string, patch: SegmentPatch) => Promise<Segment | null>;
  split: (uid: string, at: number) => Promise<void>;
  merge: (uid: string, uidB: string) => Promise<void>;
  /**
   * Claim an uncovered span as a new segment. Overlaps come back as an error.
   *
   * Resolves to whether the server took it. The boolean is not a second error
   * channel the error bar still carries every refusal it is the one thing
   * the caller cannot read off `segments`, because a refused add leaves the list
   * exactly as it was, and the composer must not clear the sentence the user
   * typed until there is a segment holding it.
   */
  add: (segment: NewSegment) => Promise<boolean>;
  /**
   * Take a segment out of the dub. The selection goes with it: a panel pointing
   * at a uid the server has just retired is the same bug `split` avoids by
   * minting new ones.
   */
  remove: (uid: string) => Promise<void>;
  retranslate: (uids: string[], batch?: string) => Promise<void>;
  resynthesize: (uids: string[], batch?: string) => Promise<void>;
  render: () => Promise<void>;
  /**
   * Run the pipeline on this project again a resume, on one that stopped.
   *
   * Goes through `submit` like every other model action, which is what puts the
   * refusal in the error bar: the server 409s while anything is already in
   * flight, and a resume that silently did nothing would look exactly like a
   * resume that worked.
   */
  resume: () => Promise<void>;
  /** `batch` stops every job the same gesture queued, not just this one. */
  cancel: (id: string, batch?: boolean) => Promise<void>;
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
  // Has *this session* edited anything? A run mixed before the render stamp
  // existed is `stale` with `changed=0` forever; the header must not nag on it,
  // but it also must not stay silent after a real edit. This is the tiebreak.
  const [edited, setEdited] = useState(false);
  // Mirror for reads inside stable callbacks (see `patch`'s rollback).
  const segmentsRef = useRef<Segment[]>(segments);
  useEffect(() => {
    segmentsRef.current = segments;
  }, [segments]);
  const [stage, setStage] = useState<StageProgress | null>(null);
  const [log, setLog] = useState<LogEvent[]>([]);

  // Refs so the event handler stays stable across renders.
  const nameRef = useRef(name);
  nameRef.current = name;
  /** Whose progress `stage` is describing, so a new job's start can clear it. */
  const stageJobRef = useRef<string | null>(null);

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

  /*
   * An edit changes what the render is behind by, and only the server can say
   * by how much.
   *
   * `render.stale` and `render.changed` are derived on the server from the
   * manifest, so they are as old as the last `GET /api/projects`. Nothing
   * refetched that after a no-model edit `patch` deliberately updates only the
   * one segment so the header sat quiet on "Render preview" until some job
   * happened to finish and pull the project in behind it. The user's own edit is
   * the one thing that should light it up.
   *
   * The project payload only, never the segments: the segment list is already
   * authoritative from the PATCH's own response, and re-reading it here would
   * throw away the optimistic state that makes an edit feel instant. Debounced,
   * because a bulk flip is a hundred PATCHes and this question has one answer.
   */
  const projectTimer = useRef<number | undefined>(undefined);
  const refreshProject = useCallback(() => {
    window.clearTimeout(projectTimer.current);
    projectTimer.current = window.setTimeout(() => {
      void api.getProject(nameRef.current).then(setProject).catch(() => {
        // A failed refresh leaves the previous answer on screen, which is the
        // right fallback: it is stale by at most one edit, and the PATCH that
        // triggered this has already reported its own failure if it had one.
      });
    }, 300);
  }, []);

  useEffect(() => () => window.clearTimeout(projectTimer.current), []);

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
            /*
             * A replayed frame is history, not news. The prelude sends all nine
             * stages at once as a snapshot, so pinning the display to "the last
             * stage frame seen" read the ninth as the stage the run is in the
             * editor announced "Running report · 100%" the moment a re-voice
             * started, over a video that was minutes out of date. They still
             * count as a reason to re-read the project; they are not a progress
             * report about anything happening now.
             */
            if (!event.replay) {
              setStage({
                stage: event.stage,
                status: event.status,
                progress: event.progress ?? null,
                message: event.message ?? null,
              });
            }
            if (event.status === "done" || event.status === "failed") scheduleRefresh();
            break;
          case "segment":
            scheduleRefresh();
            break;
          case "job":
            setJobs((current) => {
              const next = current.map((job) =>
                job.id === event.id
                  ? {
                      ...job,
                      status: event.status,
                      error: event.error ?? null,
                      // A frame knows the job's lines and gesture; an entry made
                      // from an earlier frame may not. Never unlearn them.
                      uids: event.uids ?? job.uids,
                      batch: event.batch ?? job.batch,
                    }
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
                      uids: event.uids ?? [],
                      batch: event.batch ?? null,
                    } satisfies Job,
                  ];
            });
            /*
             * A new job is not the old job's last frame. The stage strip kept
             * whatever the previous job left behind "Running report", 100% —
             * until the new one happened to emit its first frame, which for a
             * re-voice is after the model loads. Clearing on the transition
             * means the strip says nothing until the new job says something.
             */
            if (event.status === "running" && !event.replay &&
                stageJobRef.current !== event.id) {
              stageJobRef.current = event.id;
              setStage(null);
            }
            /*
             * Cancelled is terminal, and it is terminal in the way that
             * matters most: the worker saves the partial work it did before it
             * stopped (`jobs`' journal), so a cancelled resynthesis leaves real
             * new clips on disk. Treating only done|failed as the end left the
             * segments unread the results were there and the screen never
             * showed them until the run was re-opened.
             */
            if (event.status === "done" || event.status === "failed" ||
                event.status === "cancelled") {
              if (stageJobRef.current === event.id) {
                stageJobRef.current = null;
                setStage(null);
              }
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
       * row shows in between and what it showed before was a ◆ Dubbed badge
       * over audio of a sentence the user had just replaced.
       */
      /*
       * Rollback is per-uid, never the whole array. In a bulk flip this
       * callback closes over an array from before the loop started, so
       * restoring it wholesale visually undid every *successful* PATCH made
       * before the one that failed. The one record this call touched is the
       * only thing a failure may revert — and reading it through a ref keeps
       * `segments` out of the deps, so `actions` (and every row callback fed
       * from it) keeps its identity across segment updates.
       */
      const before = segmentsRef.current.find((seg) => seg.uid === uid);
      setSegments((current) =>
        current.map((seg) => (seg.uid === uid ? applyPatch(seg, body) : seg)),
      );
      try {
        const updated = await api.patchSegment(name, uid, body);
        setSegments((current) => current.map((seg) => (seg.uid === uid ? updated : seg)));
        setEdited(true);
        refreshProject();
        return updated;
      } catch (err) {
        setSegments((current) =>
          current.map((seg) => (seg.uid === uid && before ? before : seg)),
        );
        setError(describe(err));
        return null;
      }
    },
    [name, refreshProject],
  );

  /**
   * Returns whether the edit landed; the error bar still gets every failure.
   *
   * `land` names the line to select afterwards. Every one of these edits leaves
   * the reviewer looking at a list that has moved under them, and the selection
   * was simply dropped: a split collapsed the inspector back to the run board,
   * and an added segment arrived mid-list, unselected and unscrolled the only
   * sign it had worked being the count on the All chip.
   */
  const structural = useCallback(
    async (
      run: () => Promise<Segment[]>,
      land?: (before: Segment[], after: Segment[]) => string | null,
    ): Promise<boolean> => {
      const before = segmentsRef.current;
      try {
        const after = await run();
        setSegments(after);
        // A split or a merge is the largest edit there is it mints new uids,
        // so every count the render was made against moves.
        setEdited(true);
        refreshProject();
        const uid = land?.(before, after) ?? null;
        if (uid) setSelectedUid(uid);
        return true;
      } catch (err) {
        setError(describe(err));
        return false;
      }
    },
    [refreshProject],
  );

  const split = useCallback(
    async (uid: string, at: number) => {
      await structural(
        () => api.splitSegment(name, uid, at),
        (before, after) => landed(before, after, uid),
      );
    },
    [name, structural],
  );

  const merge = useCallback(
    async (uid: string, uidB: string) => {
      await structural(
        () => api.mergeSegments(name, uid, uidB),
        (before, after) => landed(before, after, uid),
      );
    },
    [name, structural],
  );

  const add = useCallback(
    (segment: NewSegment) => structural(() => api.addSegment(name, segment), landed),
    [name, structural],
  );

  const remove = useCallback(
    async (uid: string) => {
      await structural(() => api.removeSegment(name, uid));
      // Whether or not the call succeeded, the panel must not be left pointing
      // at a uid that no longer names anything. `setSelectedUid` is a no-op when
      // something else was selected.
      setSelectedUid((current) => (current === uid ? null : current));
    },
    [name, structural],
  );

  /*
   * The job the server just accepted is the record that the work exists the
   * rows it names start pulsing because it is in `jobs`, not because a separate
   * list was primed here. One source, so nothing can clear it early.
   */
  const submit = useCallback(
    async (run: () => Promise<Job>) => {
      try {
        const job = await run();
        setJobs((current) => [...current.filter((j) => j.id !== job.id), job]);
      } catch (err) {
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
      add,
      remove,
      retranslate: (uids, batch) => submit(() => api.retranslate(name, uids, batch)),
      resynthesize: (uids, batch) => submit(() => api.resynthesize(name, uids, batch)),
      render: () => submit(() => api.render(name)),
      resume: () => submit(() => api.resume(name)),
      // Optimistic in the same shape the event handler is: a cancel ends the
      // job, so the stage strip ends with it and whatever the worker saved on
      // its way out is re-read. The `cancelled` frame will say the same thing a
      // moment later; this is for the moment before it arrives, and for a
      // stream that has dropped. With `batch`, the mates it stopped are marked
      // too they are cancelled server-side whether or not a frame arrives.
      cancel: async (id, batch = false) => {
        try {
          await api.cancelJob(id, batch);
          setJobs((current) => {
            const gesture = current.find((job) => job.id === id)?.batch ?? null;
            return current.map((job) =>
              isPending(job) &&
              (job.id === id || (batch && gesture !== null && job.batch === gesture))
                ? { ...job, status: "cancelled" as const }
                : job,
            );
          });
          setStage(null);
          stageJobRef.current = null;
          void load();
        } catch (err) {
          setError(describe(err));
        }
      },
      reload: load,
      dismissError: () => setError(null),
    }),
    [add, load, merge, name, patch, remove, split, submit],
  );

  /*
   * Which lines have work in flight, straight off the queue.
   *
   * Only `retranslate` and `resynthesize` name lines. A `render` re-lays the
   * whole timeline and a `run` rebuilds everything, so neither marks particular
   * rows busy every row would pulse, which says nothing.
   */
  const pendingUids = useMemo(() => {
    const out = new Set<string>();
    for (const job of jobs) {
      if (!isPending(job)) continue;
      if (job.kind !== "retranslate" && job.kind !== "resynthesize") continue;
      for (const uid of job.uids) out.add(uid);
    }
    return [...out];
  }, [jobs]);

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
    pendingUids,
    edited,
    log,
  };

  return [state, actions];
}

/**
 * Which line a structural edit leaves the reviewer on.
 *
 * Which uid the server mints is the server's business, so the rule is
 * positional: the record that was not in the list before, earliest first a
 * split mints two halves and the first is the one being read. `survivor` is the
 * fallback for an edit that mints nothing, as a merge keeping the first line's
 * uid does; a uid the answer no longer carries selects nothing rather than
 * pointing the panel at a retired record.
 */
function landed(before: Segment[], after: Segment[], survivor?: string): string | null {
  const had = new Set(before.map((seg) => seg.uid));
  const minted = after.filter((seg) => !had.has(seg.uid));
  if (minted.length > 0) return minted.reduce((a, b) => (a.start <= b.start ? a : b)).uid;
  return survivor && after.some((seg) => seg.uid === survivor) ? survivor : null;
}

export function activeJob(jobs: Job[]): Job | null {
  return jobs.find((job) => job.status === "running") ?? jobs.find((job) => job.status === "queued") ?? null;
}

/**
 * The server's own words, never ours.
 *
 * A `busy` used to be rewritten as "A job is already running this one is
 * queued behind it.", which is a sentence about something that never happened:
 * the model actions *do* queue and answer 202, and 409/busy is raised only for
 * the edits the server refuses outright while a job runs (a split, a merge, a
 * bounds change, a second install). Telling a user their split is queued when
 * it was rejected is worse than saying nothing and the server's message
 * already names which edit and why.
 */
function describe(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  return String(error);
}
