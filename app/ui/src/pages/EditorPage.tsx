/**
 * The editor.
 *
 * Premiere-shaped: preview and timeline stack on the left in playback order,
 * the inspector holds the right rail so the selection never moves under the
 * cursor, and the segment navigator sits under the timeline.
 *
 * What changed, and why, after watching it in use:
 *
 * **The picture gets the room.** The preview used to be pinned at 38% of the
 * height while the segment table took everything left over — so the largest
 * pane on a video editor's screen was a spreadsheet, and the video was a
 * postage stamp. Now the preview is the growing pane and the navigator is a
 * fixed strip: the workspace is the video and the timeline, and the list is
 * how you get around them.
 *
 * **Only one thing is on screen at a time.** The job strip renders only when
 * there is a job (or the stream has dropped); the legend and the shortcut
 * hints, which were two permanent rows of chrome across the top of the
 * timeline, are behind one "?" button. What replaces them in the toolbar is a
 * single line of counts — that is information about *this run*, not a key to
 * be re-read every minute.
 *
 * The whole screen stays usable while a job runs. Nothing here disables itself
 * on a running job except the two model actions, which queue.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { Film, HelpCircle, Loader2, Minus, Plus, RefreshCw } from "lucide-react";
import { AppHeader } from "../components/AppShell";
import { JobBar } from "../components/JobBar";
import { RunSummary } from "../components/RunSummary";
import { SegmentInspector } from "../components/SegmentInspector";
import { SegmentList } from "../components/SegmentList";
import { StageTrack } from "../components/StageTrack";
import { Timeline, TimelineLegend } from "../components/Timeline";
import { VideoPlayer } from "../components/VideoPlayer";
import {
  Badge,
  Button,
  ButtonGroup,
  Empty,
  ErrorBar,
  Eyebrow,
  Kbd,
  Popover,
  Progress,
} from "../components/ui";
import { api } from "../lib/api";
import { FIXTURE_PROJECT } from "../lib/fixtures";
import { STATE_META, segmentState, totalDuration, type SegmentState } from "../lib/segments";
import { summarizeStages } from "../lib/stages";
import { activeJob, useProject } from "../lib/useProject";
import { useTransport } from "../lib/useTransport";
import type { Job, ProjectDetail } from "../lib/types";
import type { StageProgress } from "../lib/useProject";

const ZOOM_STEPS = [0.5, 1, 2, 4, 8, 16, 32];

/** Every shortcut the editor binds, in the order they are learned. */
const SHORTCUTS: [string[], string][] = [
  [["space"], "play / pause"],
  [["←", "→"], "previous / next segment"],
  [["shift", "←", "→"], "nudge the playhead one second"],
  [["a"], "play the original for the selection"],
  [["b"], "play the dub for the selection"],
  [["k"], "switch between dub and keep"],
  [["+", "−"], "zoom the timeline"],
];

export function EditorPage() {
  const params = useParams<{ name: string }>();
  const name = params.name ?? FIXTURE_PROJECT;
  const [state, actions] = useProject(name);
  const { segments, project, selectedUid } = state;

  const total = useMemo(
    () => totalDuration(segments, project?.source.duration ?? null),
    [project, segments],
  );
  const transport = useTransport(total);
  const [zoom, setZoom] = useState(4);
  const timelineRef = useRef<HTMLDivElement>(null);

  const selected = useMemo(
    () => segments.find((seg) => seg.uid === selectedUid) ?? null,
    [segments, selectedUid],
  );

  const previewUrl = project?.outputs.preview
    ? api.mediaUrl(name, project.outputs.preview)
    : null;

  const counts = useMemo(() => {
    const out: Record<SegmentState, number> = { dubbed: 0, kept: 0, failed: 0, pending: 0 };
    for (const seg of segments) out[segmentState(seg)] += 1;
    return out;
  }, [segments]);

  /** Selecting a segment also moves the playhead to it — they are one gesture. */
  const selectAndSeek = useCallback(
    (uid: string) => {
      actions.select(uid);
      const seg = segments.find((s) => s.uid === uid);
      if (seg) transport.seek(seg.start);
    },
    [actions, segments, transport],
  );

  const step = useCallback(
    (delta: number) => {
      if (segments.length === 0) return;
      const index = segments.findIndex((seg) => seg.uid === selectedUid);
      const next = Math.max(0, Math.min(segments.length - 1, (index < 0 ? 0 : index) + delta));
      selectAndSeek(segments[next].uid);
    },
    [segments, selectAndSeek, selectedUid],
  );

  const zoomOut = useCallback(
    () => setZoom((z) => ZOOM_STEPS[Math.max(0, ZOOM_STEPS.indexOf(z) - 1)] ?? z),
    [],
  );
  const zoomIn = useCallback(
    () =>
      setZoom(
        (z) => ZOOM_STEPS[Math.min(ZOOM_STEPS.length - 1, ZOOM_STEPS.indexOf(z) + 1)] ?? z,
      ),
    [],
  );

  // Keyboard. Never while a field has focus — an editor that eats the space bar
  // mid-sentence is worse than one with no shortcuts.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && (/^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName) || target.isContentEditable)) {
        return;
      }
      if (event.metaKey || event.ctrlKey) return;

      switch (event.key) {
        case " ":
          event.preventDefault();
          transport.toggle();
          break;
        case "ArrowRight":
          event.preventDefault();
          if (event.shiftKey) transport.nudge(1);
          else step(1);
          break;
        case "ArrowLeft":
          event.preventDefault();
          if (event.shiftKey) transport.nudge(-1);
          else step(-1);
          break;
        case "k":
          if (selected) {
            event.preventDefault();
            void actions.patch(selected.uid, {
              keep: !selected.keep,
              keep_reason: selected.keep ? undefined : "manual",
            });
          }
          break;
        case "+":
        case "=":
          event.preventDefault();
          zoomIn();
          break;
        case "-":
          event.preventDefault();
          zoomOut();
          break;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [actions, selected, step, transport, zoomIn, zoomOut]);

  const job = activeJob(state.jobs);
  const queued = state.jobs.filter((j) => j.status === "queued" && j.id !== job?.id).length;

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-plane">
      <AppHeader
        actions={
          <>
            <Button
              onClick={() => void actions.reload()}
              title="Read this run's manifest from disk again"
              size="sm"
            >
              <RefreshCw className="h-3.5 w-3.5" />
              Reload
            </Button>
            <Button
              variant="primary"
              size="sm"
              title="Re-runs timeline, mix and report and replaces preview.mp4"
              onClick={() => {
                if (
                  window.confirm(
                    "Re-render the preview? This re-runs timeline, mix and report and replaces " +
                      "preview.mp4 — a full video re-encode, typically a few minutes.",
                  )
                ) {
                  void actions.render();
                }
              }}
            >
              <Film className="h-3.5 w-3.5" />
              Render preview
            </Button>
          </>
        }
      >
        <span
          className="min-w-0 flex-1 truncate text-[13px] font-semibold text-primary"
          title={project?.source.title}
        >
          {project?.source.title ?? name}
        </span>
        {project ? (
          <Badge>
            {project.source.src_lang} → {project.source.tgt_lang}
          </Badge>
        ) : null}
      </AppHeader>

      {/* Renders nothing at all unless there is a job or the stream dropped. */}
      <JobBar
        job={job}
        queued={queued}
        stage={state.stage}
        connected={state.connected}
        onCancel={(id) => void actions.cancel(id)}
        onReload={() => void actions.reload()}
      />
      {state.error ? <ErrorBar message={state.error} onDismiss={actions.dismissError} /> : null}

      {state.loading ? (
        <Empty title="Opening the run">
          <Loader2 className="mx-auto mt-2 h-5 w-5 animate-spin" aria-label="Loading" />
        </Empty>
      ) : (
        <div className="flex min-h-0 flex-1">
          <div className="flex min-w-0 flex-1 flex-col">
            {/* The workspace. This is the pane that grows. */}
            <div className="min-h-40 flex-1 border-b border-border">
              <VideoPlayer
                src={previewUrl}
                transport={transport}
                duration={total}
                title={project?.source.title ?? name}
                placeholder={
                  <PreviewPlaceholder
                    project={project}
                    job={job}
                    stage={state.stage}
                    onRender={() => void actions.render()}
                  />
                }
              />
            </div>

            <section ref={timelineRef} className="shrink-0 border-b border-border bg-surface">
              <div className="flex items-center gap-3 border-b border-border bg-sunken px-3 py-1.5">
                <Eyebrow className="shrink-0">Timeline</Eyebrow>
                <StateCounts counts={counts} />
                <div className="ml-auto flex shrink-0 items-center gap-2">
                  <ButtonGroup className="h-7">
                    <Button variant="ghost" size="xs" onClick={zoomOut} aria-label="Zoom out">
                      <Minus className="h-3 w-3" />
                    </Button>
                    <span className="flex w-16 items-center justify-center bg-raised font-mono text-[11px] tabular-nums text-muted">
                      {zoom}px/s
                    </span>
                    <Button variant="ghost" size="xs" onClick={zoomIn} aria-label="Zoom in">
                      <Plus className="h-3 w-3" />
                    </Button>
                  </ButtonGroup>
                  <EditorHelp counts={counts} />
                </div>
              </div>
              <Timeline
                segments={segments}
                total={total}
                currentTime={transport.currentTime}
                selectedUid={selectedUid}
                busyUids={state.busyUids}
                pxPerSecond={zoom}
                onSelect={selectAndSeek}
                onSeek={transport.seek}
              />
            </section>

            {/*
              A fixed strip, not the leftover space. Eight or nine lines is
              enough to see where you are and what is around you, which is what
              a navigator is for; the segment being worked on is in the rail.
            */}
            <section className="flex h-[15rem] shrink-0 flex-col bg-surface">
              <div className="flex shrink-0 items-center gap-2 border-b border-border bg-sunken px-3 py-1.5">
                <Eyebrow className="shrink-0">Segments</Eyebrow>
                <span className="text-[11px] text-muted">
                  {segments.length} line{segments.length === 1 ? "" : "s"}
                </span>
                <span className="ml-auto hidden shrink-0 items-center gap-1.5 text-[11px] text-muted sm:flex">
                  <Kbd>←</Kbd>
                  <Kbd>→</Kbd> to step
                </span>
              </div>
              <div className="min-h-0 flex-1">
                <SegmentList
                  segments={segments}
                  selectedUid={selectedUid}
                  currentTime={transport.currentTime}
                  busyUids={state.busyUids}
                  onSelect={selectAndSeek}
                />
              </div>
            </section>
          </div>

          <aside className="w-[27rem] shrink-0 border-l border-border bg-surface xl:w-[29rem]">
            {selected ? (
              <SegmentInspector
                seg={selected}
                segments={segments}
                speakers={Object.keys(project?.speakers ?? {})}
                busy={state.busyUids.includes(selected.uid)}
                playhead={transport.currentTime}
                onPatch={(patch) => void actions.patch(selected.uid, patch)}
                onSplit={(at) => void actions.split(selected.uid, at)}
                onMerge={(uidB) => void actions.merge(selected.uid, uidB)}
                onRetranslate={() => void actions.retranslate([selected.uid])}
                onResynthesize={() => void actions.resynthesize([selected.uid])}
              />
            ) : (
              <RunSummary
                project={project}
                counts={counts}
                total={total}
                onSeek={transport.seek}
              />
            )}
          </aside>
        </div>
      )}
    </div>
  );
}

/**
 * What this run is made of, in one line.
 *
 * This is not the legend — it is a census, and it belongs on screen for the
 * same reason the legend does not: it changes as you work, and the legend
 * never does. States with nothing in them are omitted rather than greyed;
 * "0 failed" is a fact nobody needs printed at them all day.
 */
function StateCounts({ counts }: { counts: Record<SegmentState, number> }) {
  const present = (Object.keys(STATE_META) as SegmentState[]).filter((s) => counts[s] > 0);
  if (present.length === 0) return null;
  return (
    <span className="flex min-w-0 flex-wrap items-center gap-x-2.5 gap-y-0.5 text-[11px] text-muted">
      {present.map((state) => (
        <span key={state} className="inline-flex items-center gap-1 whitespace-nowrap">
          <span aria-hidden style={{ color: STATE_META[state].token }}>
            {STATE_META[state].glyph}
          </span>
          <span className="font-mono tabular-nums text-secondary">{counts[state]}</span>
          {STATE_META[state].label.toLowerCase()}
        </span>
      ))}
    </span>
  );
}

/**
 * The reference material, behind one button.
 *
 * The legend and the shortcut list are both things you read on your first day
 * and on the day you forget — and neither changes, ever. They used to occupy
 * two rows across the top of the timeline on every frame of every session,
 * which is a lot of rent for something read twice.
 */
function EditorHelp({ counts }: { counts: Record<SegmentState, number> }) {
  return (
    <Popover
      label="Legend and keyboard shortcuts"
      title="What the colours mean"
      trigger={<HelpCircle className="h-3.5 w-3.5" />}
      className="w-[19rem]"
    >
      <TimelineLegend counts={counts} />
      <Eyebrow className="mt-3.5 mb-2">Keyboard</Eyebrow>
      <dl className="flex flex-col gap-1.5 text-[11px]">
        {SHORTCUTS.map(([keys, does]) => (
          <div key={does} className="flex items-baseline gap-2">
            <dt className="flex shrink-0 items-center gap-1">
              {keys.map((key) => (
                <Kbd key={key}>{key}</Kbd>
              ))}
            </dt>
            <dd className="min-w-0 flex-1 text-right text-muted">{does}</dd>
          </div>
        ))}
      </dl>
    </Popover>
  );
}

/**
 * The preview stage when there is no `preview.mp4` to show.
 *
 * This is the screen a user stares at for the entire length of a run, so it
 * has to be the run's status board rather than an apology. Which of the three
 * things it says depends on why the file is missing, and they are genuinely
 * different situations: the run is still working (wait, here is how far), the
 * run finished but never muxed a video (press the button), or nothing has run
 * at all (the editor still works on the segments).
 */
function PreviewPlaceholder({
  project,
  job,
  stage,
  onRender,
}: {
  project: ProjectDetail | null;
  job: Job | null;
  stage: StageProgress | null;
  onRender: () => void;
}) {
  const summary = summarizeStages(project?.stages);
  const working = job != null && (job.status === "running" || job.status === "queued");
  const progress = stage?.progress ?? job?.progress ?? null;

  return (
    <div className="grid h-full place-items-center px-6 py-5">
      <div className="w-full max-w-md text-center">
        <Eyebrow>{working ? "Working" : summary.complete ? "No preview file" : "Preview"}</Eyebrow>

        <p className="mt-2 text-[15px] font-semibold text-primary">
          {working
            ? `Running ${stage?.stage ?? summary.current ?? "the pipeline"}`
            : summary.failed
              ? `The run stopped at ${summary.failed}`
              : summary.complete
                ? "The run finished, but there is no video"
                : summary.started
                  ? `Stopped after ${summary.done} of ${summary.total} stages`
                  : "Nothing has run yet"}
        </p>

        <p className="mt-1.5 text-[12px] leading-relaxed text-muted">
          {working
            ? (stage?.message ?? "preview.mp4 is written by the mix stage, at the end.")
            : summary.complete
              ? "Render preview re-runs timeline, mix and report and writes preview.mp4."
              : "preview.mp4 is written by the mix stage. Until then the transport below is a virtual clock, so the timeline and the segment list still work."}
        </p>

        <div className="mt-4 flex flex-col items-center gap-2.5">
          <StageTrack stages={project?.stages} current={stage?.stage ?? null} />
          {working ? <Progress value={progress} className="w-56" /> : null}
          {!working && (summary.complete || summary.started) ? (
            <Button size="sm" onClick={onRender}>
              <Film className="h-3.5 w-3.5" />
              Render preview
            </Button>
          ) : null}
        </div>
      </div>
    </div>
  );
}
