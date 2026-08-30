/**
 * The newest runs, under the import form.
 */

import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Clapperboard } from "lucide-react";
import { RunRow, orderRuns } from "./RunRow";
import { SectionLabel } from "./ui";
import { api } from "../lib/api";
import type { ProjectSummary } from "../lib/types";

/**
 * How many lines the glance shows before it defers to the archive.
 *
 * The cap stays at eight, and stays for a different reason than it used to.
 * It no longer buys height back: the list is its own scroll box inside a
 * `min-h-0 flex-1` section, so the region is the height the column leaves it
 * whether it holds four runs or forty, and the form above it never moves. What
 * the cap buys is that this stays a glance. A workspace with nineteen runs
 * scrolled whole under the new-dub form is a second archive with a worse header,
 * and the archive is one place. Eight is about what the region shows without
 * scrolling at a normal window height, so the common case is the whole list at
 * once and the count plus "N more on the runs page" says where the rest is.
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
 * A vertical list, one run per line, scrolling inside its own box. It was a
 * sideways strip of cards for one commit, on the theory that a narrow card can
 * only carry what differs between two dubs of one source. Seen against a real
 * workspace the theory lost to the shape: a list of runs reads down, a sideways
 * scroll hides most of itself off the right edge, and the user said so plainly.
 * What the strip was right about is kept in the line instead, so each run still
 * carries the timing pair that tells two dubs of one video apart. It is the
 * archive's own row (`components/RunRow`), one density tighter, because two
 * lists of the same object drifting apart is the thing having a shared
 * component prevents.
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
      {/* The column. `min-h-0` with `flex-1` is what keeps the scroll inside
          this box: without it a flex child refuses to be shorter than its
          content, the section grows to whatever nineteen runs need, and the
          form above it is pushed off a short window. A strip of padding, so the
          hover lift is not shaved off at the edges. */}
      <ul
        data-runs-list
        className="flex min-h-0 w-full min-w-0 flex-1 flex-col gap-2 overflow-y-auto p-0.5"
      >
        {shown.map((project) => (
          <RunRow key={project.name} project={project} compact />
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
