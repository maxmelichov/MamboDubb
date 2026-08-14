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

export function StageTrack({
  stages,
  current,
  showLabel = true,
  className,
}: {
  stages?: Partial<Record<Stage, StageStatus>>;
  /** Overrides the stage derived from `stages` — the live one from the stream. */
  current?: Stage | null;
  showLabel?: boolean;
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
      title={`${summary.done} of ${summary.total} stages done${active ? ` · at ${active}` : ""}`}
    >
      {STAGES.map((stage, i) => {
        const status = stages?.[stage];
        const done = status === "done" || (!stages && i < activeIndex);
        const failedHere = status === "failed" || (summary.failed === stage);
        const isActive = stage === active;
        return (
          <span
            key={stage}
            aria-hidden
            title={stage}
            className={cn(
              "h-1.5 w-1.5 rounded-full transition-colors",
              failedHere
                ? "bg-critical ring-2 ring-critical/25"
                : isActive
                  ? "bg-primary ring-2 ring-primary/25"
                  : done
                    ? "bg-primary"
                    : "bg-border",
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
      </span>
    </span>
  );
}
