/**
 * Runs: every dub in the workspace, and the door to starting another one.
 *
 * This is the whole archive, and it is a *list*, on purpose. It used to be a
 * column bolted onto the import form, which meant the nav pill said "Runs" and
 * the page answered with "start something new" — and users who came back for
 * yesterday's dub read the screen as "there are no old runs" and started it
 * again. The switch works the way the user asked for: Runs is a place, New dub
 * is a place, and the pill up top moves between them.
 *
 * Home keeps a *glance* at the same list — the newest few rows under the form,
 * with "All runs" pointing here. That is a shortcut, not a second archive: the
 * rows are literally this page's rows (`components/RunRow`), and everything
 * about ordering, counting and the empty state stays here.
 *
 * The composition is a centred column of full-width rows, one per run. It was
 * a four-across card grid, and the grid had a failure mode the card design
 * could not fix: the layout only composed at the run count it was drawn for.
 * Three runs at 1440px was three small tiles huddled in the top-left corner of
 * an empty plane — a screen that read as broken, not as short — and each tile
 * paid for its width by truncating the one field a user recognises a run by.
 * A row spends the width on the title and the column spends the emptiness on
 * margin, which is the difference between "not much here yet" and "something
 * failed to load". One run, three, or twenty: the shape is the same shape.
 *
 * A header row names the place and counts what is in it; the primary action —
 * New dub — sits at its right edge, because "start another" is the only thing
 * you can do here that is not opening a run.
 *
 * Order: anything currently running first, then most recently touched. A run
 * in flight is the one a user comes back to check on, and "most recent" is
 * how humans actually address the rest ("the one from yesterday").
 *
 * The empty state is the invitation, not a redirect. Zero runs on first load
 * still lands here — a centred card that says what will fill the page and
 * offers the one button that starts it. Auto-routing to /new would teach the
 * user that "/" is the form, which is exactly the confusion this page ends.
 */

import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Clapperboard, Loader2, Plus, PlugZap } from "lucide-react";
import { PageShell } from "../components/AppShell";
import { RunRow, orderRuns } from "../components/RunRow";
import { Button, Card, Empty, SectionLabel } from "../components/ui";
import { api } from "../lib/api";
import type { ProjectSummary } from "../lib/types";

export function RunsPage() {
  const navigate = useNavigate();
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);
  // Distinct from "no projects": one means the list worked and there is
  // nothing to open, the other means the server never answered, and telling a
  // user the first when it was the second is how they end up re-running a
  // dub they already have.
  const [listError, setListError] = useState<string | null>(null);

  const [reloads, setReloads] = useState(0);
  useEffect(() => {
    let cancelled = false;
    void api
      .listProjects()
      .then((list) => {
        if (cancelled) return;
        setProjects(list);
        setListError(null);
      })
      .catch((err) => {
        if (cancelled) return;
        setProjects([]);
        setListError(String(err instanceof Error ? err.message : err));
      });
    return () => {
      cancelled = true;
    };
  }, [reloads]);

  const ordered = projects == null ? null : orderRuns(projects);

  return (
    // No hero: the nav pill says Runs, the header row below says it again with
    // a count, and a display title over a list of rows would be the page
    // explaining itself to people already using it.
    //
    // The shell stays at the composition width so the chrome lines up with the
    // import screen next door; the column below is the page's own measure. A
    // reading column, because rows are read: 64rem holds a long title, the
    // nine-dot track and the meta in one line without the title paying for it,
    // and it keeps twenty rows from becoming twenty 1400px hairlines.
    <PageShell width="wide">
      <section
        data-region="runs"
        className="mx-auto flex w-full max-w-[64rem] flex-col gap-3.5"
      >
        <div className="flex flex-wrap items-center justify-between gap-3 px-1">
          <div className="flex items-baseline gap-3">
            <SectionLabel icon={Clapperboard}>Your runs</SectionLabel>
            <span className="text-[11px] tabular-nums text-muted">
              {projects && projects.length > 0
                ? `${projects.length} ${projects.length === 1 ? "run" : "runs"} in outputs/`
                : ""}
            </span>
          </div>
          {/* A button rather than a styled link: it is this page's one primary
              action, and it reads (and tabs) like the other primaries in the
              app. The nav pill carries the same destination for every other
              screen. */}
          <Button variant="primary" onClick={() => navigate("/")}>
            <Plus className="h-4 w-4" />
            New dub
          </Button>
        </div>

        {ordered == null ? (
          <Card className="rounded-3xl">
            <div className="flex items-center gap-3 px-6 py-7 text-[13px] text-muted">
              <Loader2 className="h-4 w-4 shrink-0 animate-spin" aria-hidden />
              Reading outputs/…
            </div>
          </Card>
        ) : listError ? (
          <Card className="rounded-3xl">
            <Empty
              className="py-9"
              icon={PlugZap}
              title="Can't reach the studio server"
              action={
                <Button size="sm" onClick={() => setReloads((n) => n + 1)}>
                  Try again
                </Button>
              }
            >
              Existing runs live on the server, and it did not answer:{" "}
              <span className="text-secondary">{listError}</span>. Start it with{" "}
              <code className="font-mono text-secondary">uv run python -m dubbing_app.server</code>,
              then try again.
            </Empty>
          </Card>
        ) : ordered.length === 0 ? (
          // The zero state is the front door, not a dead end: it says what the
          // page will hold and hands over the one button that fills it.
          <Card className="rounded-3xl">
            <Empty
              className="py-14"
              icon={Clapperboard}
              title="No runs yet"
              action={
                <Button variant="primary" onClick={() => navigate("/")}>
                  <Plus className="h-4 w-4" />
                  Start your first dub
                </Button>
              }
            >
              Every dub you start lands here, resumable. Point the studio at a video and it
              takes it from there.
            </Empty>
          </Card>
        ) : (
          // A strip of padding on every side, so the hover lift and the row
          // shadows are not shaved off at the column's edges.
          <ul className="flex flex-col gap-2.5 p-0.5">
            {ordered.map((project) => (
              <RunRow
                key={project.name}
                project={project}
                onOpen={() => navigate(`/editor/${encodeURIComponent(project.name)}`)}
              />
            ))}
          </ul>
        )}
      </section>
    </PageShell>
  );
}
