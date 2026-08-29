/**
 * The newest runs, under the import form.
 */

import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Clapperboard } from "lucide-react";
import { RunRow, orderRuns } from "./RunRow";
import { SectionLabel } from "./ui";
import { api } from "../lib/api";
import type { ProjectSummary } from "../lib/types";

/** How many rows the glance shows before it defers to the archive. */
const HOME_RUNS = 8;

/**
 * The workspace, under the form.
 *
 * Home is the new-dub form and stays the new-dub form — but a screen that only
 * ever offers to start something implies there is nothing already started, and
 * the run people actually want on a second visit is usually the one still in
 * flight. So the newest few rows sit under the card, running-first, in the same
 * rows /runs draws (`components/RunRow`), with "All runs" as the way to the
 * whole list. Four of them: enough to hold the one that is running and the ones
 * from this morning, few enough that the form is still what this page is.
 *
 * It is quiet about its failures on purpose. A server that did not answer is
 * worth a sentence here and a red card *there* — this region is a shortcut, and
 * a shortcut that shouts about a list you did not ask for is a worse home
 * screen than one that simply has no shortcut in it. Zero runs draws nothing at
 * all: the empty state for "no runs yet" is the form the user is looking at.
 */
export function HomeRuns() {
  const navigate = useNavigate();
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
      {/* A strip of padding, so the hover lift is not shaved off at the edges. */}
      <ul className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto p-0.5">
        {shown.map((project) => (
          <RunRow
            key={project.name}
            project={project}
            compact
            onOpen={() => navigate(`/editor/${encodeURIComponent(project.name)}`)}
          />
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
