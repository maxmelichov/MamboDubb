/**
 * Runs: every dub in the workspace, and the door to starting another one.
 *
 * This is the home screen, and it is a *list*, on purpose. It used to be a
 * column bolted onto the import form, which meant the nav pill said "Runs" and
 * the page answered with "start something new" — and users who came back for
 * yesterday's dub read the screen as "there are no old runs" and started it
 * again. The switch works the way the user asked for: Runs is a place, New dub
 * is a place, and the pill up top moves between them.
 *
 * The composition is one region, full width. A header row names the place and
 * counts what is in it; the primary action — New dub — sits at its right edge,
 * because "start another" is the only thing you can do here that is not
 * opening a run. The cards themselves are unchanged from their column days:
 * each one answers "where did this run get to" without being opened.
 *
 * The empty state is the invitation, not a redirect. Zero runs on first load
 * still lands here — a centred card that says what will fill the page and
 * offers the one button that starts it. Auto-routing to /new would teach the
 * user that "/" is the form, which is exactly the confusion this page ends.
 */

import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ChevronRight, Clapperboard, Loader2, Plus, PlugZap } from "lucide-react";
import { PageShell } from "../components/AppShell";
import { StageTrack } from "../components/StageTrack";
import { Badge, Button, Card, Empty, LogoMark, SectionLabel } from "../components/ui";
import { api } from "../lib/api";
import { cn } from "../lib/classNames";
import { timecode } from "../lib/format";
import { ago, stageTone, summarizeStages } from "../lib/stages";
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

  return (
    // No hero: the nav pill says Runs, the header row below says it again with
    // a count, and a display title over a list of cards would be the page
    // explaining itself to people already using it.
    <PageShell width="wide">
      <section data-region="runs" className="flex flex-col gap-3.5">
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
          <Button variant="primary" onClick={() => navigate("/new")}>
            <Plus className="h-4 w-4" />
            New dub
          </Button>
        </div>

        {projects == null ? (
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
        ) : projects.length === 0 ? (
          // The zero state is the front door, not a dead end: it says what the
          // page will hold and hands over the one button that fills it.
          <Card className="rounded-3xl">
            <Empty
              className="py-14"
              icon={Clapperboard}
              title="No runs yet"
              action={
                <Button variant="primary" onClick={() => navigate("/new")}>
                  <Plus className="h-4 w-4" />
                  Start your first dub
                </Button>
              }
            >
              Every dub you start lands here, resumable — point the studio at a video and it
              takes it from there.
            </Empty>
          </Card>
        ) : (
          // A strip of padding on every side, so the hover lift and the card
          // shadows are not shaved off at the grid's edges.
          <ul className="grid gap-4 p-0.5 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {projects.map((project) => (
              <ProjectCard
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

/**
 * One existing run.
 *
 * The row this replaces said the title, the run name, the language pair and the
 * duration everything except the thing you actually came to find out, which
 * is *where this run got to*. A half-finished run and a finished one looked
 * identical, so the only way to tell them apart was to open both. The card
 * carries the pipeline position (nine dots, the shared track), a word for it,
 * and when it last moved.
 *
 * A card rather than a row because the page is wide: four runs across at
 * 1440px is a composition, and the same four as full-width rows is four
 * hairlines with eleven hundred pixels of nothing in the middle of them.
 */
function ProjectCard({ project, onOpen }: { project: ProjectSummary; onOpen: () => void }) {
  const summary = summarizeStages(project.stages);
  return (
    <li className="flex">
      <button
        type="button"
        onClick={onOpen}
        className={cn(
          "group flex w-full flex-col gap-3 rounded-2xl border border-border bg-surface p-4 text-left",
          "shadow-card transition-all hover:-translate-y-0.5 hover:border-axis hover:shadow-lift",
        )}
      >
        <span className="flex items-start gap-3">
          <span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl border border-border bg-sunken text-muted transition-colors group-hover:border-axis group-hover:text-primary">
            <LogoMark className="h-4 w-4" />
          </span>
          <span className="min-w-0 flex-1">
            <span className="block truncate text-[13.5px] font-semibold text-primary">
              {project.title}
            </span>
            <span className="mt-0.5 block truncate font-mono text-[11px] text-muted">
              {project.name}
            </span>
          </span>
          <ChevronRight className="mt-1 h-4 w-4 shrink-0 text-muted transition-transform group-hover:translate-x-0.5 group-hover:text-primary" />
        </span>

        <span className="flex flex-wrap items-center gap-2">
          <Badge tone={stageTone(summary)}>{summary.label}</Badge>
          <span className="text-[11px] text-muted">{ago(project.mtime)}</span>
        </span>

        <span className="mt-auto flex items-center justify-between gap-2 border-t border-border pt-3">
          <StageTrack stages={project.stages} showLabel={false} />
          <span className="flex shrink-0 items-center gap-1.5 text-[10px] font-bold uppercase tracking-[0.14em] text-muted">
            {project.src_lang} → {project.tgt_lang}
            <span aria-hidden>·</span>
            <span className="tabular-nums">
              {project.duration ? timecode(project.duration, 0) : "—"}
            </span>
          </span>
        </span>
      </button>
    </li>
  );
}
