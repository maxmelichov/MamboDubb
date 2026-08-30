/**
 * One existing run, in the two shapes the app shows it in, and the ordering
 * every list of runs uses.
 *
 * It lived inside RunsPage until the home screen grew a runs region of its own
 * under the new-dub card. Two lists showing the same object have to show it the
 * same way, because a second, slightly different row is how "the run I started"
 * starts looking like a different kind of thing depending on which screen found
 * it. So the presentation is a component and the pages differ only in how they
 * arrange it.
 *
 * `RunRow` is the archive's shape: a full-width row, one per run, down a
 * reading column. Unopened, it answers the question a list exists for, which is
 * *where did this run get to*: a state stripe down the left edge (the first
 * thing a scanning eye meets, and readable from across the room) and the stage
 * chip, which carries the phrase and the tube bar inside one border because
 * they are one fact.
 *
 * `RunTab` is home's shape, and the reason it exists is width. Two runs of the
 * same source drew two full-width rows carrying the same title, the same pair
 * and the same phrase, stacked down a column with the whole right half of the
 * screen empty: a stack that says "these are two different things" while
 * showing nothing different. A tab is small, so it can only carry what differs,
 * which is the point.
 *
 * The title gets the width in both. It is the field a user recognises a run by
 * and it is frequently Hebrew, so it carries `dir="auto"` and `.auto-dir`: an
 * RTL title lays out right-to-left inside its own span while the chrome around
 * it (name, langs, track, Open) stays put.
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
 * The archive's row. The whole row is the button, and Open at the right edge is
 * the label for that fact rather than a second target: it brightens with the
 * row's hover lift so the affordance reads before the click, but clicking
 * anywhere works.
 */
export function RunRow({ project, onOpen }: { project: ProjectSummary; onOpen: () => void }) {
  const tone = stageTone(summarizeStages(project.stages));
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
        <span aria-hidden className={cn("w-1 shrink-0 self-stretch", stripeClass(tone))} />

        <span className="flex min-w-0 flex-1 flex-wrap items-center gap-x-6 gap-y-2.5 px-5 py-4">
          {/* Identity: the title at reading size, the run's directory name and
              the pair-and-duration meta under it. This block flexes; everything
              after it holds its shape and wraps under on a narrow window. */}
          <span className="min-w-0 flex-1 basis-60">
            <span
              dir="auto"
              className="auto-dir block truncate text-[14.5px] font-semibold leading-snug text-primary"
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

          {/* Position: the phrase and the map, in one chip. They were two, a
              badge saying "Stopped after 4 of 9" and, past a gap, a bar of the
              same nine stages, and a bordered pill beside a bordered pill reads
              as two widgets that happen to be neighbours rather than as one
              sentence and its picture. */}
          <StageTrack
            stages={project.stages}
            label="state"
            className="shrink-0 gap-2 px-2.5 py-1.5 [&>[data-stage-track]]:h-2.5 [&>[data-stage-track]]:w-24"
          />

          {/* Recency and the way in. "Open" is styled as a quiet control so the
              row advertises that it goes somewhere, but the entire row is the
              button, so this span is a label and not a nested target. */}
          <span className="flex shrink-0 items-center gap-4">
            <span className="w-[7.5rem] text-right text-[11px] tabular-nums text-muted max-sm:w-auto">
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

/**
 * Home's shape: one run as a tab in a strip.
 *
 * A link, not `role="tab"`. It looks like a tab and it is seated on a rail like
 * a tab, but pressing it leaves this page for the editor, and a tab that
 * navigates is a lie told to a screen reader: `role="tab"` promises a panel on
 * this page that will become the shown one, and announces a selected state that
 * never arrives because the screen is gone. So the roles stay off, the element
 * is the anchor it behaves like, and the tab look is styling. It is also why no
 * tab is drawn "current": none of them is.
 *
 * What a tab carries is what differs. Two dubs of one YouTube video share a
 * title, a language pair and often a phrase, so a strip of tabs labelled by
 * title alone would be worse than the rows it replaced. The visible text is the
 * title, then the two fields that separate two runs of one source (how long the
 * cut is and how long ago it was touched), then the state chip. The
 * directory name and the language pair are the fields that are equal in that
 * case, and they are the ones that move to the tooltip along with the untruncated
 * title.
 */
export function RunTab({ project }: { project: ProjectSummary }) {
  const tone = stageTone(summarizeStages(project.stages));
  return (
    // `-mb-px` seats the tab on the strip's rail, so the tab's open bottom edge
    // and the rail are one line rather than two.
    <li className="-mb-px flex shrink-0">
      <Link
        to={`/editor/${encodeURIComponent(project.name)}`}
        title={fullDescription(project)}
        className={cn(
          "group flex w-64 flex-col overflow-hidden rounded-t-xl border border-b-0 border-border bg-surface",
          "transition-colors hover:border-axis hover:bg-sunken",
        )}
      >
        {/* The same stripe the row wears, along the top edge where a tab's
            own accent belongs. */}
        <span aria-hidden className={cn("h-1 w-full", stripeClass(tone))} />
        <span className="flex min-w-0 flex-col gap-1.5 px-3 py-2.5">
          <span
            dir="auto"
            className="auto-dir block truncate text-[13px] font-semibold leading-snug text-primary"
          >
            {project.title}
          </span>
          {/* The disambiguating line. Two runs of one source differ here and
              almost nowhere else, so it is the line that gets a whole row of
              the tab to itself. */}
          <span className="flex min-w-0 items-center gap-1.5 text-[11px] tabular-nums text-muted">
            <span className="shrink-0">
              {project.duration ? timecode(project.duration, 0) : "—"}
            </span>
            <span aria-hidden>·</span>
            <span className="truncate">{ago(project.mtime)}</span>
          </span>
          {/* The bar takes whatever width the phrase leaves, so a short phrase
              buys a longer picture instead of a wider gap. */}
          <StageTrack
            stages={project.stages}
            label="state"
            className="w-full gap-2 [&>[data-stage-track]]:w-auto [&>[data-stage-track]]:min-w-8 [&>[data-stage-track]]:flex-1"
          />
        </span>
      </Link>
    </li>
  );
}
