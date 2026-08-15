/**
 * Where a run is in the pipeline, as nine dots.
 *
 * The pipeline is fixed and ordered, which is the whole reason this reads at a
 * glance: the shape never changes, so "how far along" is a position, not a
 * number you have to parse. Filled means done, ringed means in flight, hollow
 * means not yet, and a failure is the one thing that gets a hue.
 *
 * Three screens use it — the job bar, the projects list and the preview
 * placeholder — and they share it so a run cannot look 6/9 in one place and
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
  className,
}: {
  stages?: Partial<Record<Stage, StageStatus>>;
  /** Overrides the stage derived from `stages` — the live one from the stream. */
  current?: Stage | null;
  showLabel?: boolean;
  /**
   * The segments have moved on since the last render, so `timeline`, `mix` and
   * `report` are done *about something else*. They are drawn hollow — a ring
   * where the filled dot would be — because "done" and "done, but not about
   * this" are different facts and the track only had one mark for both.
   */
  stale?: boolean;
  className?: string;
}) {
  const summary = summarizeStages(stages);
  const active = current ?? summary.current;
  const activeIndex = active ? STAGES.indexOf(active) : STAGES.length;

  return (
    <span
      className={cn(
        "inline-flex shrink-0 items-center gap-1 rounded-md border border-border bg-raised px-2 py-1",
        className,
      )}
      title={
        `${summary.done} of ${summary.total} stages done${active ? ` · at ${active}` : ""}` +
        (stale ? " · timeline, mix and report are about an older cut" : "")
      }
    >
      {STAGES.map((stage, i) => {
        const status = stages?.[stage];
        const done = status === "done" || (!stages && i < activeIndex);
        const failedHere = status === "failed" || (summary.failed === stage);
        const isActive = stage === active;
        // Hollow only where the dot would otherwise claim to be finished: a
        // stage that never ran, or is running now, has its own mark already.
        const hollow = stale && done && !isActive && !failedHere &&
          RENDER_STAGES.includes(stage);
        return (
          <span
            key={stage}
            aria-hidden
            title={hollow ? `${stage} — from the last render` : stage}
            className={cn(
              "h-1.5 w-1.5 rounded-full transition-colors",
              failedHere
                ? "bg-critical ring-2 ring-critical/25"
                : isActive
                  ? "bg-primary ring-2 ring-primary/25"
                  : hollow
                    // A ring around the track's own background, so the
                    // difference from a filled dot is shape rather than
                    // shade — it survives greyscale, and colour-blindness,
                    // which a lighter fill would not.
                    ? "bg-raised ring-[1.5px] ring-primary"
                    : done
                      ? "bg-primary"
                      // `axis`, not `border`: a border-coloured dot is a 1.1:1
                      // difference from the chip it sits on in light and 1.26:1
                      // in dark, which is a dot you cannot count.
                      : "bg-axis",
            )}
          />
        );
      })}
      {showLabel ? (
        <span className="ml-1 text-[11px] font-semibold text-secondary">
          {active ?? "done"}
        </span>
      ) : null}
      <span className="sr-only">
        {summary.done} of {summary.total} stages done
        {active ? `, at ${active}` : ""}
        {stale ? ", timeline, mix and report are about an older cut" : ""}
      </span>
    </span>
  );
}
