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
import { ChevronRight, Clapperboard, Loader2, Plus, PlugZap } from "lucide-react";
import { PageShell } from "../components/AppShell";
import { StageTrack } from "../components/StageTrack";
import { Badge, Button, Card, Empty, SectionLabel } from "../components/ui";
import { api } from "../lib/api";
import { cn } from "../lib/classNames";
import { timecode } from "../lib/format";
import { ago, stageTone, summarizeStages } from "../lib/stages";
import type { ProjectSummary } from "../lib/types";

/** A run with any stage in flight is the one the user came back to watch. */
function isRunning(project: ProjectSummary): boolean {
  return Object.values(project.stages ?? {}).includes("running");
}

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

  const ordered =
    projects == null
      ? null
      : [...projects].sort(
          (a, b) => Number(isRunning(b)) - Number(isRunning(a)) || b.mtime - a.mtime,
        );

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
          <Button variant="primary" onClick={() => navigate("/new")}>
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

/**
 * One existing run, as one generous row.
 *
 * The row answers the question the list exists for — *where did this run get
 * to* — without being opened: a state stripe down the left edge (the first
 * thing a scanning eye meets, and readable from across the room), the same
 * phrase as a chip ("Running translate", "Failed at fetch" — the stripe is
 * reinforcement, never the message), and the nine-dot track a step wider than
 * the card grid could afford it.
 *
 * The title gets the width the cards never had. It is the field a user
 * recognises a run by and it is frequently Hebrew, so it carries `dir="auto"`
 * and `.auto-dir`: an RTL title lays out right-to-left inside its own span
 * while the row's chrome — name, langs, track, Open — stays put.
 *
 * The whole row is the button, and Open at the right edge is the label for
 * that fact rather than a second target: it brightens with the row's hover
 * lift so the affordance reads before the click, but clicking anywhere works.
 */
function RunRow({ project, onOpen }: { project: ProjectSummary; onOpen: () => void }) {
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

          {/* Position: the phrase, then the map. The chip is the sentence
              ("Running translate") and the track is the same fact as a shape,
              its dots a step up from the card-grid size now that a row has the
              room — the whole point of switching to rows. */}
          <span className="flex shrink-0 flex-wrap items-center gap-3">
            <Badge tone={tone}>{summary.label}</Badge>
            <StageTrack
              stages={project.stages}
              showLabel={false}
              className="gap-1.5 px-2.5 py-1.5 [&>span[aria-hidden]]:h-2 [&>span[aria-hidden]]:w-2"
            />
          </span>

          {/* Recency and the way in. "Open" is styled as a quiet control so the
              row advertises that it goes somewhere, but the entire row is the
              button — this span is a label, not a nested target. */}
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
