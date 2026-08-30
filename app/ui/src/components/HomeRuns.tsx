/**
 * The newest runs, under the import form.
 */

import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Clapperboard } from "lucide-react";
import { RunTab, orderRuns } from "./RunRow";
import { SectionLabel } from "./ui";
import { api } from "../lib/api";
import type { ProjectSummary } from "../lib/types";

/**
 * How many tabs the glance shows before it defers to the archive.
 *
 * The cap stays, and the strip scrolls sideways rather than wrapping. Wrapping
 * is the option that costs the form: every wrapped line is another band of
 * height under the card, and on a short window the fifth run is what pushes
 * Start dubbing off the screen. A sideways scroll spends a fixed strip of
 * height no matter how many runs there are, and the cap keeps that scroll short
 * enough to be a glance rather than a second archive. The count and "All runs"
 * above the strip are what say there is more, because a scroll box that runs
 * past its right edge does not announce itself.
 */
const HOME_RUNS = 8;

/**
 * The workspace, under the form.
 *
 * Home is the new-dub form and stays the new-dub form, but a screen that only
 * ever offers to start something implies there is nothing already started, and
 * the run people actually want on a second visit is usually the one still in
 * flight. So the newest few sit under the card, running-first, with "All runs"
 * as the way to the whole list.
 *
 * They are tabs rather than rows because of what home's runs actually look
 * like: dub one video twice and the two rows carry the same title, the same
 * pair and the same phrase, full width, stacked, with the right half of the
 * screen empty. The archive keeps the rows, since a row is right when the list
 * is the page. Here the list is a shortcut, so it is a strip along the top of
 * the region and the tabs carry the fields that differ.
 *
 * It is quiet about its failures on purpose. A server that did not answer is
 * worth a sentence here and a red card *there*, because this region is a
 * shortcut, and a shortcut that shouts about a list you did not ask for is a
 * worse home screen than one that simply has no shortcut in it. Zero runs draws
 * nothing at all: the empty state for "no runs yet" is the form the user is
 * looking at.
 */
export function HomeRuns() {
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    void api
      .listProjects()
      .then((list) => {
        if (!cancelled) setProjects(list);
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Vanishing here leaves the column a dead plane under the form — the exact
  // hole the user pointed at. While loading, on failure, and with zero runs
  // alike, the region keeps its place with a quiet card instead.
  if (failed || projects == null || projects.length === 0) {
    return (
      <section data-region="home-runs" className="flex min-h-0 flex-1 flex-col gap-2.5">
        <div className="px-1">
          <SectionLabel icon={Clapperboard}>Your runs</SectionLabel>
        </div>
        <div className="flex min-h-40 flex-1 items-center justify-center rounded-2xl border border-dashed border-border px-6 py-10">
          <p className="max-w-sm text-center text-[13px] leading-relaxed text-muted">
            {projects == null && !failed
              ? "Looking for runs…"
              : "Nothing here yet. Every dub you start lands in this list, live while it runs."}
          </p>
        </div>
      </section>
    );
  }
  const ordered = orderRuns(projects);
  const shown = ordered.slice(0, HOME_RUNS);

  return (
    <section data-region="home-runs" className="flex min-h-0 flex-1 flex-col gap-2.5">
      <div className="flex flex-wrap items-center justify-between gap-3 px-1">
        <div className="flex items-baseline gap-3">
          <SectionLabel icon={Clapperboard}>Your runs</SectionLabel>
          <span className="text-[11px] tabular-nums text-muted">
            {projects.length} {projects.length === 1 ? "run" : "runs"} in outputs/
          </span>
        </div>
        {/* A link, not a button: the page's one *button* is Start dubbing, and
            a second filled control under it would compete with it. */}
        <Link
          to="/runs"
          data-all-runs
          className="rounded-md text-[11px] font-bold uppercase tracking-[0.14em] text-muted transition-colors hover:text-primary"
        >
          All runs →
        </Link>
      </div>
      {/* The rail the tabs are seated on. `overflow-x-auto` keeps the scroll
          inside this box, so a ninth run cannot put a scrollbar on the page
          body, and `min-w-0` is what lets a flex column's child be narrower
          than its content in the first place. */}
      <ul
        data-runs-strip
        className="flex w-full min-w-0 shrink-0 items-stretch gap-2 overflow-x-auto overflow-y-hidden border-b border-border px-0.5 pt-0.5"
      >
        {shown.map((project) => (
          <RunTab key={project.name} project={project} />
        ))}
      </ul>
      {ordered.length > shown.length ? (
        <p className="px-1 text-[11px] text-muted">
          {ordered.length - shown.length} more on the runs page.
        </p>
      ) : null}
    </section>
  );
}
