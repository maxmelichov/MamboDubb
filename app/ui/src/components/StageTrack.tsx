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
 * one rounded tube filled from the left. The states are unchanged, and so are
 * the reasons for them, written on each branch below.
 *
 * The tube carried a hairline separator on each of the eight inner joins for a
 * while, drawn in the chip's own background so the nine were countable. At the
 * size this actually ships at, on a dark theme, a hairline of ground colour
 * across a solid fill is not a hairline: it is a gap, and a complete run read
 * as roughly ten white blocks in a rounded outline, which is the segmented
 * battery gauge the dots were replaced to stop being. Connectedness is what was
 * asked for, twice. Counting to nine by eye never was, and the count is already
 * written out in words beside the bar ("Complete", "Stopped after 4 of 9"),
 * which answers "how far along" better than counting blocks does. So adjacent
 * segments in the same state are one unbroken length of fill, and the only
 * edges left in the bar are the ones where the state actually changes, which
 * are the joins that carry information.
 *
 * Three screens use it the job bar, the projects list and the preview
 * placeholder and they share it so a run cannot look 6/9 in one place and
 * "mix" in another.
 *
 * The chip carries the words too. It used to sit next to a separate badge
 * holding the run's phrase, two borders and a gap for one fact, which is what
 * a user meant by "the progress and the dots need to be more connected". With
 * `label="state"` the phrase lives inside this border, beside the fill it
 * describes, and the chip takes the phrase's tone.
 */

import { badgeTone } from "./ui";
import { cn } from "../lib/classNames";
import { stageTone, summarizeStages } from "../lib/stages";
import { STAGES, type Stage, type StageStatus } from "../lib/types";

/**
 * What the chip says next to the bar.
 *
 * `"state"` is the merged control. The run's position used to be drawn as two
 * bordered chips with a gap between them, a badge reading "Stopped after 4 of
 * 9" and, beside it, a track of the same nine stages: one fact, stated twice,
 * in two containers, which is exactly how a user reads them as two unrelated
 * widgets. In `"state"` the phrase and the bar share one border, so the number
 * and the fill are visibly the same sentence, and the chip wears the state's
 * tone the way the badge did.
 *
 * `"stage"` is the older, quieter label: just the stage being worked on, for
 * the panel that has already said the whole sentence in its headline.
 */
type TrackLabel = "state" | "stage" | "none";

/**
 * The three stages a render re-runs, and so the three a manifest edit outdates.
 * Everything before `timeline` is about the source media and is still true.
 */
const RENDER_STAGES: Stage[] = ["timeline", "mix", "report"];

export function StageTrack({
  stages,
  current,
  label = "stage",
  stale = false,
  rebuilding = false,
  className,
}: {
  stages?: Partial<Record<Stage, StageStatus>>;
  /** Overrides the stage derived from `stages` the live one from the stream. */
  current?: Stage | null;
  label?: TrackLabel;
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
  // The phrase is the message and the tone is reinforcement, the same contract
  // the badge held before the two chips became one.
  const stated = label === "state";

  return (
    <span
      data-stage-chip
      className={cn(
        "inline-flex shrink-0 items-center gap-1 rounded-md border px-2 py-1",
        stated ? badgeTone[stageTone(summary)] : "border-border bg-raised",
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
        * No segment carries a radius of its own: the capsule is the parent's
        * shape, clipped, and a rounded child inside it is how a bar turns back
        * into a row of beads. The bar carries the empty colour itself, so a run
        * with nothing done is still a visible track rather than a hole in the
        * chip. `h-2.5` is the height it reads as a capsule at: at 8px the 4px
        * radius is most of the bar's own height, so the ends look clipped
        * rather than rounded.
        */}
      <span
        data-stage-track
        className="flex h-2.5 w-18 shrink-0 overflow-hidden rounded-full bg-axis"
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
                // Equal ninths, and nothing drawn between them. The separator
                // that used to sit on each inner join was a hairline of the
                // chip's own ground, which against a solid fill is read as a
                // gap and turned the tube back into beads. The joins that mean
                // something are the ones where the state changes, and those
                // draw themselves: the fill colour changes there. Everywhere
                // else the neighbours share a colour and now share an edge, so
                // a finished run is one length of fill from end to end.
                "h-full flex-1 transition-colors",
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
      {stated ? (
        // `summary.label` verbatim, because "Stopped after 4 of 9" and "4 of 9
        // done" are different facts: the first says the run will not move again
        // on its own. Merging the chips must not quietly trade that away, so the
        // phrase the badge carried is the phrase the merged chip carries, at the
        // badge's own size and weight.
        <span className="ml-1 whitespace-nowrap text-[10px] font-bold uppercase tracking-[0.12em]">
          {summary.label}
        </span>
      ) : label === "stage" ? (
        <span className="ml-1 text-[11px] font-semibold text-secondary">
          {active ?? "done"}
        </span>
      ) : null}
      {/* Said once. With the phrase on screen the count is already spoken, and a
          screen reader that hears "Stopped after 4 of 9" and then "4 of 9 stages
          done" is being told one thing twice; the hover tooltip above keeps the
          detail the phrase leaves out. Without a visible phrase the bar is a
          picture with no words, so it says the sentence itself. */}
      {stated ? (
        <span className="sr-only">
          {stale ? ", timeline, mix and report are about an older cut" : ""}
          {rebuildNote ? `, rebuilding the last ${waiting.length + 1} stages` : ""}
        </span>
      ) : (
        <span className="sr-only">
          {summary.done} of {summary.total} stages done
          {active ? `, at ${active}` : ""}
          {stale ? ", timeline, mix and report are about an older cut" : ""}
          {rebuildNote ? `, rebuilding the last ${waiting.length + 1} stages` : ""}
        </span>
      )}
    </span>
  );
}
