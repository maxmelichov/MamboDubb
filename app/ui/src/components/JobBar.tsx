/**
 * The queue, as one line and as a panel when you ask.
 *
 * At most one pipeline job runs process-wide, so the resting state is a single
 * strip rather than a queue panel. But "one job runs" is not "one job exists":
 * a verdict flip queues a translate *and* a voice, and the strip used to say
 * "+1 queued" about the second one a number, with no name, no size, and no
 * way to cancel it. Everything the queue knows is now sayable:
 *
 * * the running job names its work "Re-voicing 27 lines · 4 of 27";
 * * the tail names the next one "then Re-translating 12 lines";
 * * clicking the strip opens the queue, where every waiting job has its own
 *   Cancel, the log has its first home on screen, and the three costs are
 *   written down.
 *
 * It sits on the sunken tone so it reads as a status rail against the surfaces
 * above and below it, and it is the one place in the workspace that animates.
 *
 * Two things it used to do and no longer does. It drew a StageTrack, which the
 * preview placeholder was already drawing two inches below it the same nine
 * dots, twice on one screen. And the bar and the percentage disagreed: the bar
 * read `stage?.progress ?? job.progress` while the number read only
 * `stage?.progress`, so a job reporting job-level progress showed a bar that
 * moved next to a percentage stuck on "—". One value, read once.
 *
 * Two things it does that it did not. A whole-run job names the stage it is in
 * and where that stage sits in the nine ("transcript stage 3 of 9"), because
 * the one percentage this strip has is the *stage's* and was being read as the
 * run's. And a job that ends holds the strip for six seconds in a still, quiet
 * version of itself rather than vanishing on the frame it finished: the strip
 * disappearing was the only report a successful job ever made.
 */

import { useEffect, useRef, useState } from "react";
import { Check, ChevronUp, Loader2, WifiOff } from "lucide-react";
import { Button, Eyebrow, Progress, useDismissable } from "./ui";
import { cn } from "../lib/classNames";
import { lineCount } from "../lib/format";
import { STAGES, isPending, type Job, type LogEvent } from "../lib/types";
import type { StageProgress } from "../lib/useProject";

const KIND_LABEL: Record<Job["kind"], string> = {
  run: "Dubbing",
  retranslate: "Re-translating",
  resynthesize: "Re-voicing",
  render: "Rendering preview",
};

/** How many log lines the queue panel keeps on screen. */
const LOG_TAIL = 15;

/**
 * How long a finished job's strip stays up.
 *
 * The strip used to vanish on the frame the job ended, which meant the *only*
 * report a successful job ever made was its own disappearance a reviewer who
 * looked away for five seconds came back to a screen that had never mentioned
 * the twenty-seven lines it had just re-voiced. Long enough to read one
 * sentence, and matched to the undo strip's six seconds because they are the
 * same promise: this is over, here is what it was.
 */
const DONE_MS = 6000;

/**
 * What a job is, in words, including its size.
 *
 * A whole-run job names no lines (`uids` is `[]` for `run` and `render`), and
 * "Rendering preview 0 lines" would be worse than saying nothing.
 *
 * Exported because the editor's live region announces the same events this
 * strip draws, and two spellings of "Re-voicing 27 lines" is how the spoken app
 * and the drawn one start describing different things.
 */
export function describeJob(job: Job): string {
  const label = KIND_LABEL[job.kind];
  return job.uids.length > 0 ? `${label} ${lineCount(job.uids.length)}` : label;
}

/** How a finished job reads once it has stopped moving. */
function describeEnd(job: Job): string {
  const verb =
    job.status === "failed" ? "failed" : job.status === "cancelled" ? "cancelled" : "finished";
  return job.uids.length > 0
    ? `${KIND_LABEL[job.kind]} ${verb}: ${lineCount(job.uids.length)}`
    : `${KIND_LABEL[job.kind]} ${verb}`;
}

export function JobBar({
  jobs,
  stage,
  connected,
  log,
  onCancel,
  onAnnounce,
}: {
  jobs: Job[];
  stage: StageProgress | null;
  connected: boolean;
  log: LogEvent[];
  /** `batch` stops every job the same gesture queued, not just this one. */
  onCancel: (id: string, batch: boolean) => void;
  /** Said in the editor's one live region see EditorPage's `say`. */
  onAnnounce?: (message: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const wrap = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  useDismissable(open, setOpen, wrap, trigger);

  const pending = jobs.filter(isPending);
  const job = pending.find((j) => j.status === "running") ?? pending[0] ?? null;
  const waiting = pending.filter((j) => j.id !== job?.id);

  /*
   * The job that just ended, held for a beat.
   *
   * It is set by a *transition* and not by "the newest terminal job in the
   * list": on mount the list is full of jobs that finished hours ago, and
   * announcing one of those as news would be a strip that appears every time a
   * run is opened. `seen` is the id the strip was showing last frame; when that
   * id leaves the pending set, whatever it became is the thing to say.
   */
  const [ended, setEnded] = useState<Job | null>(null);
  const seen = useRef<string | null>(null);
  const activeId = job?.id ?? null;
  useEffect(() => {
    const previous = seen.current;
    seen.current = activeId;
    if (!previous || previous === activeId) return;
    const done = jobs.find((j) => j.id === previous && !isPending(j));
    if (done) setEnded(done);
  }, [activeId, jobs]);

  /*
   * …and let go of it. The countdown is paused while the queue panel is open,
   * because the panel is hung off this strip: unmounting under a user who is
   * reading the log would close the thing they opened. It restarts when they
   * close it.
   */
  useEffect(() => {
    if (!ended || open) return;
    const timer = window.setTimeout(() => setEnded(null), DONE_MS);
    return () => window.clearTimeout(timer);
  }, [ended, open]);

  /* Start and end, spoken once each. The deps are the ids, so a progress frame
     four times a second does not re-announce anything. */
  useEffect(() => {
    if (job && onAnnounce) onAnnounce(`${describeJob(job)} started`);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job?.id]);
  useEffect(() => {
    if (ended && onAnnounce) onAnnounce(describeEnd(ended));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ended?.id]);

  if (!job && !ended && connected) return null;

  const progress = stage?.progress ?? job?.progress ?? null;
  /*
   * Where a whole-run job is, in the pipeline's own numbering.
   *
   * A `run` job's progress bar is the *stage's* fraction the only one the
   * pipeline reports so "62%" on a nine-stage run meant "62% through
   * transcript", and it read as "62% through the dub". It names the stage and
   * its position, which is the fact the percentage was being mistaken for. Only
   * for `run`: an edit job is one stage by construction, and "tts stage 6 of
   * 9" over a re-voice would be inventing a pipeline it is not in.
   */
  const runStage = job?.kind === "run" ? (stage?.stage ?? null) : null;
  const stageAt = runStage ? STAGES.indexOf(runStage) : -1;
  const stageNote =
    runStage && stageAt >= 0 ? `${runStage}: stage ${stageAt + 1} of ${STAGES.length}` : null;
  /*
   * "4 of 27" is the fraction restated over the work the user asked for. The
   * stage frames report a 0..1 fraction and the job knows how many lines it was
   * given, so the count is derived rather than reported which is honest at
   * both ends (0 of 27 before the first line, 27 of 27 at the end) and is the
   * form the question is actually asked in: not "what percent", but "how many
   * of my twenty-seven".
   */
  const done =
    job && progress != null && job.uids.length > 0
      ? `${Math.round(progress * job.uids.length)} of ${job.uids.length}`
      : null;

  return (
    /*
      One line, always: `flex-wrap` let a long stage message or a dropped
      stream wrap to a second row, which moved the whole editor down by 28px
      mid-job the script under the cursor jumping while you read it. Both
      variable-length parts truncate instead, and the strip has a fixed height
      so its appearance is the only layout change it can cause. `relative` is
      for the queue panel, which hangs off the strip and adds no height.
    */
    <div
      data-job-strip
      data-job-state={job ? "running" : ended ? "done" : "offline"}
      ref={wrap}
      className="relative flex h-8 shrink-0 items-center gap-3 border-b border-border bg-sunken px-4 text-[12.5px]"
    >
      {/*
        A dropped stream is not a dead app every edit still goes over plain
        HTTP and still saves. The only thing that stops is progress arriving on
        its own, so the strip says exactly that. It used to offer a "Refresh
        now" button, which was the third copy of the header's Reload; the
        stream reconnects by itself and any edit re-reads the run anyway.
      */}
      {!connected ? (
        <span className="flex min-w-0 flex-1 items-center gap-1.5 font-medium text-secondary">
          <WifiOff className="h-3.5 w-3.5 shrink-0" aria-hidden />
          <span className="min-w-0 truncate">
            Progress stream disconnected, retrying. Edits still save; progress will not
            update on its own.
          </span>
        </span>
      ) : null}

      {job ? (
        <>
          {/*
            The strip is the trigger. The queue is what a user wants the moment
            they read "then Re-translating 12 lines" and think "not those" —
            so it opens from the sentence that raised the question, not from a
            menu somewhere else.
          */}
          <button
            ref={trigger}
            type="button"
            data-queue-trigger
            aria-expanded={open}
            aria-label="Show the queue"
            onClick={() => setOpen((v) => !v)}
            className={cn(
              "inline-flex shrink-0 items-center gap-1.5 rounded text-[11px] font-bold",
              "uppercase tracking-[0.14em] text-primary hover:opacity-80",
            )}
          >
            <Loader2 className="h-3 w-3 animate-spin" aria-hidden />
            {describeJob(job)}
            <ChevronUp
              className={cn("h-3 w-3 transition-transform", !open && "rotate-180")}
              aria-hidden
            />
          </button>

          {done ? (
            <span className="shrink-0 font-mono text-[11px] tabular-nums text-secondary">
              {done}
            </span>
          ) : null}
          {stageNote ? (
            <span data-job-stage className="shrink-0 text-[11px] text-secondary">
              {stageNote}
            </span>
          ) : null}

          <span className="min-w-0 flex-1 truncate text-secondary">
            {stage?.message ?? job.message ?? ""}
          </span>
          <Progress value={progress} className="w-40 shrink-0" />
          <span className="w-9 shrink-0 text-right font-mono text-[11px] tabular-nums text-muted">
            {progress != null ? `${Math.round(progress * 100)}%` : "—"}
          </span>

          {/*
            The tail says what is waiting, by name. "+1 queued" was a number
            about an unnamed thing, and the unnamed thing was routinely the
            minute-per-line half of the pair.
          */}
          {waiting.length > 0 ? (
            <span data-queue-tail className="shrink-0 truncate text-[11px] text-muted">
              then {describeJob(waiting[0])}
              {waiting.length > 1 ? ` +${waiting.length - 1} more` : ""}
            </span>
          ) : null}

          <CancelControl job={job} jobs={jobs} onCancel={onCancel} />
        </>
      ) : null}

      {/*
        The same strip, standing still.
        Same height, same ground, same trigger so the queue is still one click
        away for the six seconds a user has to ask "what did that just do", and
        the layout does not move on the way out. It is deliberately quiet: a
        tick rather than a spinner, no bar, no percentage, and the tail is gone
        because there is nothing behind it.
      */}
      {!job && ended ? (
        <>
          <button
            ref={trigger}
            type="button"
            data-queue-trigger
            aria-expanded={open}
            aria-label="Show the queue"
            onClick={() => setOpen((v) => !v)}
            className={cn(
              "inline-flex shrink-0 items-center gap-1.5 rounded text-[11px] font-bold",
              "uppercase tracking-[0.14em] hover:opacity-80",
              ended.status === "failed" ? "text-critical" : "text-secondary",
            )}
          >
            <Check className="h-3 w-3" aria-hidden />
            {describeEnd(ended)}
            <ChevronUp
              className={cn("h-3 w-3 transition-transform", !open && "rotate-180")}
              aria-hidden
            />
          </button>
          <span className="min-w-0 flex-1 truncate text-muted">{ended.error ?? ""}</span>
        </>
      ) : null}

      {open ? (
        <QueuePanel
          jobs={jobs}
          pending={pending}
          log={log}
          onCancel={(id, batch) => {
            onCancel(id, batch);
            setOpen(false);
          }}
        />
      ) : null}
    </div>
  );
}

/**
 * Cancel, and the question it has to ask first.
 *
 * The audit's disaster in one gesture: cancel the running re-translate, and the
 * re-voice queued behind it runs anyway on lines whose translation was just
 * abandoned 27 lines synthesised from nothing, each landing as a `tts_failed`
 * keep. The user had cancelled "dubbing these 27"; the button cancelled a step.
 *
 * So when the running job has queued mates from the same gesture, the button
 * asks which was meant. No default is safe enough to skip the question: "just
 * this" is what someone re-thinking one stage wants, and "the whole batch" is
 * what someone who has changed their mind wants, and they are one click apart.
 */
function CancelControl({
  job,
  jobs,
  onCancel,
}: {
  job: Job;
  jobs: Job[];
  onCancel: (id: string, batch: boolean) => void;
}) {
  const [asking, setAsking] = useState(false);
  const wrap = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  useDismissable(asking, setAsking, wrap, trigger);

  const mates = job.batch
    ? jobs.filter((j) => j.batch === job.batch && j.id !== job.id && isPending(j))
    : [];
  const batchLines = new Set(
    [job, ...mates].flatMap((j) => j.uids),
  ).size;

  if (mates.length === 0) {
    return (
      <Button variant="ghost" size="xs" onClick={() => onCancel(job.id, false)}>
        Cancel
      </Button>
    );
  }

  return (
    <div className="relative shrink-0" ref={wrap}>
      <Button
        ref={trigger}
        variant="ghost"
        size="xs"
        aria-expanded={asking}
        onClick={() => setAsking((v) => !v)}
      >
        Cancel
      </Button>
      {asking ? (
        <div
          role="dialog"
          aria-label="Cancel"
          className="absolute right-0 top-full z-50 mt-1.5 w-72 rounded-xl border border-border bg-raised p-3.5 shadow-pop"
        >
          <p className="text-[12.5px] leading-relaxed text-primary">
            Cancel just this, or the whole batch ({lineCount(batchLines)})?
          </p>
          <p className="mt-1.5 text-[11px] leading-relaxed text-muted">
            {describeJob(mates[0])} is waiting, from the same request. Stopping only
            this one leaves it to run.
          </p>
          <div className="mt-3 flex justify-end gap-2">
            <Button variant="ghost" size="xs" onClick={() => setAsking(false)}>
              Keep going
            </Button>
            <Button variant="secondary" size="xs" onClick={() => onCancel(job.id, false)}>
              Just this
            </Button>
            <Button variant="danger" size="xs" onClick={() => onCancel(job.id, true)}>
              The whole batch
            </Button>
          </div>
        </div>
      ) : null}
    </div>
  );
}

/**
 * Everything the queue knows, on one panel.
 *
 * Three things live here that had nowhere else to be. Every waiting job, with
 * its own Cancel the strip could only ever cancel the running one, so the
 * queued half of a pair was unreachable. The log, which the client has been
 * collecting into `state.log` since the first version and never rendered, so a
 * failure's actual message went to a variable nobody could read. And the cost
 * of the three actions, written down where the queue is: the difference
 * between twenty seconds and several minutes is the whole basis for deciding
 * what to queue, and it was learned by waiting.
 */
function QueuePanel({
  jobs,
  pending,
  log,
  onCancel,
}: {
  jobs: Job[];
  pending: Job[];
  log: LogEvent[];
  onCancel: (id: string, batch: boolean) => void;
}) {
  // The most recent job that ended, so a failure has somewhere to be read.
  // Terminal jobs are never replayed on the stream, which is right but it
  // left a failed job's error with no home at all once the strip cleared.
  const last = [...jobs].reverse().find((job) => !isPending(job)) ?? null;
  const tail = log.slice(-LOG_TAIL);

  return (
    <div
      role="dialog"
      aria-label="Queue"
      data-queue-panel
      className="absolute left-3 top-full z-50 mt-1 w-[26rem] rounded-xl border border-border bg-raised p-3.5 shadow-pop"
    >
      <Eyebrow className="mb-2">Queue: {pending.length}</Eyebrow>
      <ul className="flex flex-col gap-1">
        {pending.map((job) => (
          <li
            key={job.id}
            data-queue-job
            className="flex items-center gap-2 rounded-md border border-border bg-sunken px-2 py-1.5"
          >
            <span className="min-w-0 flex-1 truncate text-[12.5px] text-primary">
              {describeJob(job)}
            </span>
            <span className="shrink-0 text-[11px] text-muted">
              {job.status === "running" ? "running" : "queued"}
            </span>
            <Button variant="ghost" size="xs" onClick={() => onCancel(job.id, false)}>
              Cancel
            </Button>
          </li>
        ))}
      </ul>

      {last ? (
        <p
          data-queue-last
          className={cn(
            "mt-2 text-[11px] leading-relaxed",
            last.status === "failed" ? "text-critical" : "text-muted",
          )}
        >
          Last: {describeJob(last)} ({last.status})
          {last.error ? `. ${last.error}` : ""}
        </p>
      ) : null}

      {/*
        Three lines, in the order a user meets them, because the ratio is what
        decides whether to fix one line or thirty. It is not a warning and does
        not need to be read twice it sits under the queue it is about.
      */}
      <div className="mt-3 border-t border-border pt-2.5 text-[11px] leading-relaxed text-muted">
        <p>Re-translate: about 20 seconds a line.</p>
        <p>Re-voice: about a minute a line, and it re-places the whole timeline.</p>
        <p>Update the video: minutes, because it re-encodes the whole file.</p>
      </div>

      {tail.length > 0 ? (
        <>
          <Eyebrow className="mb-1.5 mt-3">Log</Eyebrow>
          <ul
            data-queue-log
            className="flex max-h-40 flex-col gap-0.5 overflow-y-auto font-mono text-[10.5px] leading-relaxed"
          >
            {tail.map((line, i) => (
              <li
                key={i}
                className={cn(
                  "truncate",
                  line.level === "error"
                    ? "text-critical"
                    : line.level === "warn"
                      ? "text-warning"
                      : "text-muted",
                )}
                title={line.message}
              >
                {line.message}
              </li>
            ))}
          </ul>
        </>
      ) : null}
    </div>
  );
}
