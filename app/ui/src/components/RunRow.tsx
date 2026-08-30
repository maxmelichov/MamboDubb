/**
 * One existing run, as one full-width line, and the ordering every list of runs
 * uses.
 *
 * It lived inside RunsPage until the home screen grew a runs region of its own
 * under the new-dub card. Two lists showing the same object have to show it the
 * same way, because a second, slightly different row is how "the run I started"
 * starts looking like a different kind of thing depending on which screen found
 * it. So there is one row, both pages draw it, and they differ only in how many
 * of them they draw and in `compact`, which is a density and not a second
 * design.
 *
 * Home briefly drew its runs as a sideways strip of tab cards instead. The user
 * saw it against a real workspace and rejected the orientation outright, so the
 * list is a vertical column again: one run per line, stacked down the page. The
 * strip's content was never the problem and it is kept here. What a line has to
 * carry is what tells two runs apart, and two dubs of one YouTube video share a
 * title, a directory prefix and often a phrase, so the fields that actually
 * separate them (how long the cut is, how long ago it was touched) sit together
 * as one adjacent pair rather than at opposite ends of the line.
 *
 * The line is one line on purpose. Stacking the title over a meta row spends
 * height, which is the scarce axis here: home's list sits under the form and
 * every extra band of it pushes Start dubbing closer to the fold. Laid out
 * across, the title takes the slack and everything else holds its own width, so
 * the column is full rather than half empty and eight runs cost eight lines.
 *
 * The title gets that slack. It is the field a user recognises a run by and it
 * is frequently Hebrew, so it carries `dir="auto"` and `.auto-dir`: an RTL title
 * lays out right-to-left inside its own span while the chrome around it (name,
 * langs, timings, chip, Open) stays put. It truncates, and the whole of it
 * survives in the line's `title` tooltip along with every field the line drops.
 *
 * The row is a link, not a button with a handler. It goes to /editor/<name>,
 * which is a place, so it is an anchor: middle-click and open-in-new-tab work,
 * and the destination is in the markup rather than in a callback each page
 * passes separately.
 */

import { ChevronRight } from "lucide-react";
import { Link } from "react-router-dom";
import { StageTrack } from "./StageTrack";
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

/** The state's hue, for the stripe that reinforces the phrase. */
function stripeClass(tone: ReturnType<typeof stageTone>): string {
  return tone === "good"
    ? "bg-good"
    : tone === "warn"
      ? "bg-warning"
      : tone === "bad"
        ? "bg-critical"
        : "bg-axis";
}

/** The run's whole identity, for a `title` attribute. */
function fullDescription(project: ProjectSummary): string {
  const summary = summarizeStages(project.stages);
  return [
    project.title,
    project.name,
    `${project.src_lang} to ${project.tgt_lang}`,
    project.duration ? timecode(project.duration, 0) : "no duration yet",
    summary.label,
    ago(project.mtime),
  ].join(" · ");
}

/**
 * The row both lists draw.
 *
 * `compact` is the home screen's density: the same line, tighter, and without
 * the directory name and the language pair. Those two are dropped there for the
 * reason home's list exists at all, which is width: the column under the form
 * is about half the archive's measure, and a line that wraps is a line that
 * makes one run in a stack of eight taller than the rest. They are also exactly
 * the fields that are equal between two dubs of one source, so they are the
 * cheapest thing on the line to move into the tooltip.
 */
export function RunRow({
  project,
  compact = false,
}: {
  project: ProjectSummary;
  compact?: boolean;
}) {
  const tone = stageTone(summarizeStages(project.stages));
  return (
    <li>
      <Link
        to={`/editor/${encodeURIComponent(project.name)}`}
        title={fullDescription(project)}
        className={cn(
          "group flex w-full items-stretch overflow-hidden rounded-2xl border border-border bg-surface text-left",
          "shadow-card transition-all hover:-translate-y-0.5 hover:border-axis hover:shadow-lift",
        )}
      >
        {/* The state, before any words: a stripe down the left edge, full
            height, in the state's hue. It is the first thing a scanning eye
            meets and it is readable from across the room, and it is the left
            edge in both lists, because one convention for "where did this get
            to" is the entire value of having one. `overflow-hidden` on the link
            clips it to the corner radius. */}
        <span aria-hidden className={cn("w-1 shrink-0 self-stretch", stripeClass(tone))} />

        <span
          className={cn(
            "flex min-w-0 flex-1 flex-wrap items-center gap-y-2",
            compact ? "gap-x-3 px-3.5 py-2.5" : "gap-x-5 px-5 py-3",
          )}
        >
          {/* The title takes the slack. Everything after it holds its shape and
              wraps under only when the window leaves it nowhere to go. */}
          <span
            dir="auto"
            className={cn(
              "auto-dir min-w-0 flex-1 basis-52 truncate font-semibold leading-snug text-primary",
              compact ? "text-[13px]" : "text-[14.5px]",
            )}
          >
            {project.title}
          </span>

          {/* Where it came from, at archive width only. */}
          {compact ? null : (
            <span className="flex min-w-0 shrink items-center gap-2 text-[11px] text-muted">
              <span className="truncate font-mono">{project.name}</span>
              <span aria-hidden>·</span>
              <span className="shrink-0 font-bold uppercase tracking-[0.14em]">
                {project.src_lang} → {project.tgt_lang}
              </span>
            </span>
          )}

          {/* The disambiguating pair, kept adjacent: how long the cut is and
              how long ago it moved. Two dubs of one source differ here and
              almost nowhere else, so reading them as one phrase ("0:30 · 2 days
              ago") is worth more than filing each under a column of its own. */}
          <span className="flex shrink-0 items-center gap-1.5 text-[11px] tabular-nums text-muted">
            {/* A run whose length is not known yet drops the field rather than
                standing a placeholder glyph in for it: the pair is read as a
                phrase, and a phrase with a dash where a number goes asks the
                reader to work out what is missing. */}
            {project.duration ? (
              <>
                <span>{timecode(project.duration, 0)}</span>
                <span aria-hidden>·</span>
              </>
            ) : null}
            <span>{ago(project.mtime)}</span>
          </span>

          {/* Position: the phrase and the map, in one chip. They were two, a
              badge saying "Stopped after 4 of 9" and, past a gap, a bar of the
              same nine stages, and a bordered pill beside a bordered pill reads
              as two widgets that happen to be neighbours rather than as one
              sentence and its picture. */}
          <StageTrack
            stages={project.stages}
            label="state"
            className={cn(
              "shrink-0 gap-2 px-2.5 py-1",
              compact
                ? "[&>[data-stage-track]]:w-20"
                : "[&>[data-stage-track]]:w-24",
            )}
          />

          {/* The way in. "Open" is styled as a quiet control so the line
              advertises that it goes somewhere, but the entire line is the
              link, so this span is a label and not a nested target. */}
          <span
            className={cn(
              "flex shrink-0 items-center gap-1 rounded-lg border border-border px-2.5 py-1",
              "text-[11px] font-semibold text-muted transition-colors",
              "group-hover:border-axis group-hover:bg-sunken group-hover:text-primary",
            )}
          >
            Open
            <ChevronRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5" />
          </span>
        </span>
      </Link>
    </li>
  );
}
