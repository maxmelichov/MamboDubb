/**
 * The viewer's status board while there is no `preview.mp4` to show, and the
 * rule for which failure is still the answer about a run.
 */

import { RotateCcw } from "lucide-react";
import { Button, Eyebrow } from "./ui";
import { StageTrack } from "./StageTrack";
import type { TransportMode } from "./VideoPlayer";
import { summarizeStages } from "../lib/stages";
import type { Job, ProjectDetail } from "../lib/types";
import type { StageProgress } from "../lib/useProject";

/**
 * The most recent failure that is still the answer about this run.
 *
 * Same rule the server applies to stages (`Projects.stage_status`): a job that
 * succeeded *after* a failure is the newer answer, so the failure is cleared
 * rather than kept as a permanent scar. Jobs arrive oldest first.
 */
function lastFailure(jobs: Job[]): Job | null {
  let dead: Job | null = null;
  for (const job of jobs) {
    if (job.status === "done") dead = null;
    else if (job.status === "failed") dead = job;
  }
  return dead;
}

/**
 * The preview stage when there is no `preview.mp4` to show.
 *
 * This is the screen a user stares at for the entire length of a run, so it has
 * to be the run's status board rather than an apology. It no longer carries its
 * own Render button or its own progress bar: the header has the first and the
 * job strip has the second, both of them three inches away and both of them
 * permanent. What it keeps that nothing else has is the stage track which of
 * the nine stages this run has got through.
 *
 * It is a band and not a stage. It used to be laid out as a centred column
 * inside the video's own 16:9, which on a wide window is 270 vertical pixels of
 * the right-hand rail spent on four short sentences while the selection panel
 * beneath it, which is where the actual work is done, scrolled thirty controls
 * through what was left. Eyebrow, one sentence, the stage track, left-aligned
 * and about 140px. The two cases that are worth more than that take more: a
 * stopped run adds the pipeline's own error and the Resume button, and a run
 * behind its script adds the stale sentence.
 *
 * And, since a run could stop, the way back in.
 *
 * A failed or abandoned run was a dead end: the app could create a project and
 * edit one, and nothing anywhere in it could start the pipeline again. The
 * screen said "The run stopped at fetch" and left the user with a script pane of
 * nothing, a dead play button and a Render button that would produce another
 * failure twenty minutes later. Both halves of the fix are here because both
 * halves are the same question *what happened, and what do I press*:
 *
 * * **What happened** is the failed job's own error. It is only on the project
 *   (`GET /api/projects/{name}` → `jobs`); the event stream deliberately replays
 *   nothing terminal, so after a reload this is the only copy of it.
 * * **What to press** is the same `run` job that created the project. Every
 *   stage is skipped when its inputs are unchanged, so re-running *is* resuming
 *   which is why the button is honest about naming the stage it will retry.
 */
export function PreviewPlaceholder({
  project,
  job,
  stage,
  mode,
  onResume,
  stale = false,
  staleBand = null,
}: {
  project: ProjectDetail | null;
  job: Job | null;
  stage: StageProgress | null;
  /** What the transport is on, so the panel does not promise a clock that is not one. */
  mode: TransportMode;
  onResume: () => void;
  /** The render is behind the script: the last three stage dots go hollow. */
  stale?: boolean;
  /** The sentence, when this panel is the one carrying it else null. */
  staleBand?: string | null;
}) {
  const summary = summarizeStages(project?.stages);
  const working = job != null && (job.status === "running" || job.status === "queued");
  const dead = lastFailure(project?.jobs ?? []);
  /*
   * A run that is not finished and has nothing in flight is stopped, whether or
   * not a stage says "failed". A job cancelled mid-fetch, a server restarted
   * under a run, a laptop that slept none of them leave a failed stage behind,
   * and all of them leave a project that will never move again on its own.
   */
  const stopped = project != null && !working && !summary.complete;

  return (
    <div className="px-3 py-2.5" data-preview-placeholder>
      <div className="w-full">
        <Eyebrow>
          {working
            ? "Working"
            : summary.failed || dead
              ? "Stopped"
              : summary.complete
                ? "No preview file"
                : "Preview"}
        </Eyebrow>

        <p className="mt-1 text-[14px] font-semibold leading-snug text-primary">
          {working
            ? `Running ${stage?.stage ?? summary.current ?? "the pipeline"}`
            : summary.failed
              ? `The run stopped at ${summary.failed}`
              : summary.complete
                ? "The run finished, but there is no video"
                : summary.started
                  ? `Stopped after ${summary.done} of ${summary.total} stages`
                  : "Nothing has run yet"}
        </p>

        {/*
          The pipeline's own last words, verbatim.
          Never paraphrased and never truncated to a category: "the run stopped
          at fetch" is the same sentence for a dead URL, a private video and a
          full disk, and only one of the three is worth retrying unchanged.
        */}
        {!working && dead?.error ? (
          <p
            data-failure
            className="mt-2 max-h-24 overflow-auto rounded-lg border border-critical/35 bg-critical/[0.06] px-2.5 py-2 text-left font-mono text-[11px] leading-relaxed break-words text-secondary"
          >
            {dead.error}
          </p>
        ) : null}

        {/*
          One sentence, not three.
          It was a paragraph explaining where preview.mp4 comes from, in a panel
          whose eyebrow and headline had already said which stage the run is on
          read once on the first run and then re-read on every screen for the
          rest of the project's life. What survives is the part that is news:
          what you can do *now*, which differs by mode.
        */}
        <p className="mt-1 text-[11px] leading-relaxed text-muted">
          {working
            ? (stage?.message ?? "preview.mp4 is written by the mix stage, at the end.")
            : summary.complete
              ? "Render preview re-runs timeline, mix and report and writes preview.mp4."
              : mode === "source"
                ? "Press play for the run's original audio, in step with the script. " +
                  "preview.mp4 arrives with the mix stage."
                : "Nothing to play until fetch writes source.wav; the timeline and the script " +
                  "still work."}
        </p>

        {/*
          The way back in.
          The rationale sits under the button rather than above it because it is
          the answer to the question the button raises "will this start over?"
          and because a user who already knows presses it without reading.
        */}
        {stopped ? (
          <>
            <Button
              variant="primary"
              size="sm"
              className="mt-2"
              data-resume
              onClick={onResume}
              title={
                summary.failed
                  ? `Run the pipeline again from ${summary.failed}`
                  : "Run the pipeline again; finished stages are skipped"
              }
            >
              <RotateCcw className="h-3.5 w-3.5" aria-hidden />
              {summary.failed ? `Retry from ${summary.failed}` : "Resume the run"}
            </Button>
            <p className="mt-1.5 text-[11px] leading-relaxed text-muted">
              Every stage is skipped when its inputs have not changed, so this picks up where the
              run stopped rather than starting over.
            </p>
          </>
        ) : null}

        <div className="mt-2 flex">
          <StageTrack
            stages={project?.stages}
            current={stage?.stage ?? null}
            stale={stale}
            /*
              A render, a re-voice or a re-translate rebuilds the end of a run
              that already reached the end. It clears the marks of the stages it
              is about to redo before it redoes them, so those segments empty
              out at once, which is the same picture as a run that never got
              that far. A first `run` is the one job for which the empty end of
              the bar is the truth.
            */
            rebuilding={working && job?.kind !== "run"}
          />
        </div>

        {/*
          The same sentence the video area carries, for the case where there is
          no video to carry it: a run whose preview was deleted, or one still
          mid-render, is exactly as far behind the script and has nothing else
          on screen that says so.
        */}
        {staleBand && !working ? (
          <p data-stale-band className="mt-2 text-[11px] leading-relaxed text-secondary">
            {staleBand}
          </p>
        ) : null}
      </div>
    </div>
  );
}
