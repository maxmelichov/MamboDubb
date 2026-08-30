/**
 * One existing run, as one generous row and the ordering every list of runs
 * uses.
 *
 * It lived inside RunsPage until the home screen grew a runs region of its own
 * under the new-dub card. Two lists showing the same object have to show it the
 * same way — a second, slightly different row is how "the run I started" starts
 * looking like a different kind of thing depending on which screen found it —
 * so the row is a component and the two pages differ only in how many of them
 * they draw.
 *
 * The row answers the question a list exists for — *where did this run get to*
 * — without being opened: a state stripe down the left edge (the first thing a
 * scanning eye meets, and readable from across the room), the same phrase as a
 * chip ("Running translate", "Failed at fetch" — the stripe is reinforcement,
 * never the message), and the nine-dot track.
 *
 * The title gets the width. It is the field a user recognises a run by and it
 * is frequently Hebrew, so it carries `dir="auto"` and `.auto-dir`: an RTL title
 * lays out right-to-left inside its own span while the row's chrome — name,
 * langs, track, Open — stays put.
 *
 * The whole row is the button, and Open at the right edge is the label for that
 * fact rather than a second target: it brightens with the row's hover lift so
 * the affordance reads before the click, but clicking anywhere works.
 *
 * `compact` is the home screen's variant: the same row, one step tighter, for a
 * list that is a glance at the workspace rather than the workspace itself.
 */

import { ChevronRight } from "lucide-react";
import { StageTrack } from "./StageTrack";
import { Badge } from "./ui";
import { cn } from "../lib/classNames";
import { timecode } from "../lib/format";
import { ago, stageTone, summarizeStages } from "../lib/stages";
import type { ProjectSummary } from "../lib/types";

/** A run with any stage in flight is the one the user came back to watch. */
export function isRunning(project: ProjectSummary): boolean {
  return Object.values(project.stages ?? {}).includes("running");
}

/**
 * Anything currently running first, then most recently touched. A run in flight
 * is the one a user comes back to check on, and "most recent" is how humans
 * actually address the rest ("the one from yesterday").
 */
export function orderRuns(projects: ProjectSummary[]): ProjectSummary[] {
  return [...projects].sort(
    (a, b) => Number(isRunning(b)) - Number(isRunning(a)) || b.mtime - a.mtime,
  );
}

export function RunRow({
  project,
  onOpen,
  compact = false,
}: {
  project: ProjectSummary;
  onOpen: () => void;
  compact?: boolean;
}) {
  const summary = summarizeStages(project.stages);
  const tone = stageTone(summary);
  return (
    <li>
      <button
        type="button"
        onClick={onOpen}
        className={cn(
          "group flex w-full items-stretch overflow-hidden rounded-2xl border border-border bg-surface text-left",
          "shadow-card transition-all hover:-translate-y-0.5 hover:border-axis hover:shadow-lift",
        )}
      >
        {/* The state, before any words: a stripe, full height, in the state's
            hue. `overflow-hidden` on the button clips it to the corner radius. */}
        <span
          aria-hidden
          className={cn(
            "w-1 shrink-0 self-stretch",
            tone === "good" && "bg-good",
            tone === "warn" && "bg-warning",
            tone === "bad" && "bg-critical",
            tone === "neutral" && "bg-axis",
          )}
        />

        <span
          className={cn(
            "flex min-w-0 flex-1 flex-wrap items-center gap-y-2.5",
            // The compact row lives in the import screen's left column, which
            // is a third narrower than the archive's measure: at the archive's
            // gaps the widest chip ("Running translate") pushed the recency and
            // Open onto a second line, so one row in a list of four was a
            // different height than the rest.
            compact ? "gap-x-4 px-4 py-3" : "gap-x-6 px-5 py-4",
          )}
        >
          {/* Identity: the title at reading size, the run's directory name and
              the pair-and-duration meta under it. This block flexes; everything
              after it holds its shape and wraps under on a narrow window. */}
          <span className="min-w-0 flex-1 basis-60">
            <span
              dir="auto"
              className={cn(
                "auto-dir block truncate font-semibold leading-snug text-primary",
                compact ? "text-[13.5px]" : "text-[14.5px]",
              )}
            >
              {project.title}
            </span>
            <span className="mt-1 flex min-w-0 items-center gap-2 text-[11px] text-muted">
              <span className="truncate font-mono">{project.name}</span>
              <span aria-hidden>·</span>
              <span className="shrink-0 font-bold uppercase tracking-[0.14em]">
                {project.src_lang} → {project.tgt_lang}
              </span>
              <span aria-hidden>·</span>
              <span className="shrink-0 tabular-nums">
                {project.duration ? timecode(project.duration, 0) : "—"}
              </span>
            </span>
          </span>

          {/* Position: the phrase, then the map. The chip is the sentence
              ("Running translate") and the track is the same fact as a shape,
              its bar a step up from the card-grid size now that a row has the
              room, which is the whole point of switching to rows. */}
          <span className="flex shrink-0 flex-wrap items-center gap-3">
            <Badge tone={tone}>{summary.label}</Badge>
            <StageTrack
              stages={project.stages}
              showLabel={false}
              className="gap-1.5 px-2.5 py-1.5 [&>[data-stage-track]]:h-2.5 [&>[data-stage-track]]:w-24"
            />
          </span>

          {/* Recency and the way in. "Open" is styled as a quiet control so the
              row advertises that it goes somewhere, but the entire row is the
              button — this span is a label, not a nested target. */}
          <span className={cn("flex shrink-0 items-center", compact ? "gap-3" : "gap-4")}>
            <span
              className={cn(
                "text-right text-[11px] tabular-nums text-muted max-sm:w-auto",
                compact ? "w-auto" : "w-[7.5rem]",
              )}
            >
              {ago(project.mtime)}
            </span>
            <span
              className={cn(
                "flex items-center gap-1 rounded-lg border border-border px-2.5 py-1",
                "text-[11px] font-semibold text-muted transition-colors",
                "group-hover:border-axis group-hover:bg-sunken group-hover:text-primary",
              )}
            >
              Open
              <ChevronRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5" />
            </span>
          </span>
        </span>
      </button>
    </li>
  );
}
