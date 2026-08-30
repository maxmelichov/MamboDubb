/**
 * Where a run is in the pipeline, as one bar cut into nine segments.
 *
 * The pipeline is fixed and ordered, which is the whole reason this reads at a
 * glance: the shape never changes, so "how far along" is a position, not a
 * number you have to parse. Filled means done, a lighter fill means in flight,
 * a hatched segment means the result behind it is not about this cut, and a
 * failure is the one thing that gets a hue.
 *
 * It used to be nine loose dots. Nine separated marks read as nine unrelated
 * facts; the run is one thing moving through one pipeline, so it is drawn as
 * one rounded tube filled from the left, with hairline separators in the chip's
 * own background so the nine are still countable. The states are unchanged, and
 * so are the reasons for them, written on each branch below.
 *
 * Three screens use it the job bar, the projects list and the preview
 * placeholder and they share it so a run cannot look 6/9 in one place and
 * "mix" in another.
 */

import { cn } from "../lib/classNames";
import { summarizeStages } from "../lib/stages";
import { STAGES, type Stage, type StageStatus } from "../lib/types";

/**
 * The three stages a render re-runs, and so the three a manifest edit outdates.
 * Everything before `timeline` is about the source media and is still true.
 */
const RENDER_STAGES: Stage[] = ["timeline", "mix", "report"];

export function StageTrack({
  stages,
  current,
  showLabel = true,
  stale = false,
  rebuilding = false,
  className,
}: {
  stages?: Partial<Record<Stage, StageStatus>>;
  /** Overrides the stage derived from `stages` the live one from the stream. */
  current?: Stage | null;
  showLabel?: boolean;
  /**
   * The job in flight is rebuilding the end of a run that already got there
   * a render, a re-voice, a re-translate rather than running it for the first
   * time. It matters because a rebuild drops the "done" mark of every stage it
   * is about to redo *before* it redoes them, so five of the nine dots go dark
   * the moment the job starts. That is the same picture as a run that has never
   * got past four, and it is how correcting one line and pressing Render read
   * as "it removed all my renders and started from zero". The stages this job
   * has still to reach are drawn hatched there is a result behind them,
   * being replaced instead of as the empty track of work never done.
   */
  rebuilding?: boolean;
  /**
   * The segments have moved on since the last render, so `timeline`, `mix` and
   * `report` are done *about something else*. They are drawn hatched a
   * striped segment where the solid fill would be because "done" and "done,
   * but not about this" are different facts and the track only had one mark
   * for both. The stripe is chosen for the reason the ring was chosen before
   * it: the difference from a filled segment is texture, not shade.
   */
  stale?: boolean;
  className?: string;
}) {
  const summary = summarizeStages(stages);
  const active = current ?? summary.current;
  const activeIndex = active ? STAGES.indexOf(active) : STAGES.length;
  // The stages this rebuild has yet to reach. Counted here so the sentence and
  // the dots cannot disagree about how much of the run is being replaced.
  const waiting =
    rebuilding && active
      ? STAGES.filter((stage, i) => i > activeIndex && stages?.[stage] !== "done")
      : [];
  const rebuildNote = waiting.length
    ? ` · rebuilding ${waiting.length + 1} of ${STAGES.length} stages; the rest stand`
    : "";

  return (
    <span
      className={cn(
        "inline-flex shrink-0 items-center gap-1 rounded-md border border-border bg-raised px-2 py-1",
        className,
      )}
      title={
        `${summary.done} of ${summary.total} stages done${active ? ` · at ${active}` : ""}` +
        (stale ? " · timeline, mix and report are about an older cut" : "") +
        rebuildNote
      }
    >
      {/*
        * One tube, not nine dots. `overflow-hidden` plus `rounded-full` is what
        * rounds the two ends of the bar and leaves the eight inner joins square,
        * so the segments meet edge to edge and the fill reads as one length.
        * The bar carries the empty colour itself, so a run with nothing done is
        * still a countable track rather than a hole in the chip.
        */}
      <span
        data-stage-track
        className="flex h-2 w-18 shrink-0 overflow-hidden rounded-full bg-axis"
      >
        {STAGES.map((stage, i) => {
          const status = stages?.[stage];
          const done = status === "done" || (!stages && i < activeIndex);
          const failedHere = status === "failed" || (summary.failed === stage);
          const isActive = stage === active;
          // Hatched only where the segment would otherwise claim to be
          // finished: a stage that never ran, or is running now, has its own
          // mark already.
          const hollow = stale && done && !isActive && !failedHere &&
            RENDER_STAGES.includes(stage);
          // Waiting its turn in this rebuild, not never run. Same stripe as
          // `hollow` and for the same reason there is a previous result here
          // at a weight that says the job has not got to it yet.
          const pending = waiting.includes(stage) && !isActive && !failedHere;
          return (
            <span
              key={stage}
              data-stage-segment
              aria-hidden
              title={
                hollow
                  ? `${stage}: from the last render`
                  : pending
                    ? `${stage}: waiting its turn in this rebuild`
                    : stage
              }
              className={cn(
                // Equal ninths, joined. The separator is a hairline of the chip's
                // own background rather than a drawn line: it is the gap the nine
                // dots used to leave, only one pixel of it, so the segments are
                // countable without a tenth colour to measure.
                "h-full flex-1 transition-colors",
                i > 0 && "border-l border-raised",
                failedHere
                  ? "bg-critical"
                  : isActive
                    // Lighter than done, at the one place on the bar where the
                    // fill stops, so "in flight" is a weight and a position
                    // rather than a hue. Not a pulse: it sits in a row of run
                    // cards, and nine of these breathing is a busy list.
                    ? "bg-primary/55"
                    : hollow
                      // A stripe over the track's own background, so the
                      // difference from a filled segment is texture rather than
                      // shade it survives greyscale, and colour-blindness,
                      // which a lighter fill would not. A ring did this job
                      // when the stages were dots; inside a joined bar there is
                      // no outside for a ring to sit on.
                      ? "stage-hatch [--hatch-ink:var(--color-primary)]"
                      : pending
                        // The same texture, in the quieter ink: same fact (a
                        // result is behind this), less of a claim on the eye,
                        // and the two never appear on one track anyway.
                        ? "stage-hatch [--hatch-ink:var(--color-secondary)]"
                        : done
                            ? "bg-primary"
                          // `axis`, not `border`: a border-coloured segment is a
                          // 1.1:1 difference from the chip it sits on in light and
                          // 1.26:1 in dark, which is a track you cannot count.
                          // Inherited from the bar, so nothing to paint here.
                          : "",
              )}
            />
          );
        })}
      </span>
      {showLabel ? (
        <span className="ml-1 text-[11px] font-semibold text-secondary">
          {active ?? "done"}
        </span>
      ) : null}
      <span className="sr-only">
        {summary.done} of {summary.total} stages done
        {active ? `, at ${active}` : ""}
        {stale ? ", timeline, mix and report are about an older cut" : ""}
        {rebuildNote ? `, rebuilding the last ${waiting.length + 1} stages` : ""}
      </span>
    </span>
  );
}
