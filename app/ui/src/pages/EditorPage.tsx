/**
 * The editor.
 *
 * Three regions, and the proportions are the argument:
 *
 *   header 44px  — where you are, and the one expensive button
 *   main row     — SCRIPT (58%, and it grows) | VIEWER (42%, fixed column)
 *   timeline     — 112px, full width, at the very bottom
 *
 * The previous shape had four stacked regions with the *video* as the pane that
 * grew and the text as a fifteen-rem drawer under the timeline. That is the
 * layout of a tool for cutting picture. This is a tool for checking a
 * translation against its original, and the artefact being checked is text — so
 * the text takes the room, the picture is the reference beside it, and the
 * timeline is a map along the bottom. Final Cut's grammar (viewer top-right,
 * index left, timeline across the bottom), with the index promoted to the work
 * surface because here it is the work.
 *
 * Below ~1200px the viewer column collapses: the video becomes a compact strip
 * over the selection panel rather than a column of its own, because a 300px
 * video next to a 300px script is two things too small to use.
 *
 * ## Three type sizes
 *
 * 11 / 12.5 / 14, and nothing else, across every component this screen owns:
 * 11 for meta and hints, 12.5 for body and controls, 14 for the two things that
 * are the point — a translation line and a panel heading. The only exception is
 * the 9px on the timeline's axis ticks, which are chart furniture and were
 * already there. A dense tool drifts into eight sizes one component at a time;
 * three is what makes the hierarchy readable rather than merely present.
 *
 * ## One keyboard listener
 *
 * There used to be two — this page's, and a second inside ABPlayer whose guard
 * only checked `INPUT|TEXTAREA|SELECT`, so typing "a" or "b" into any
 * contentEditable started playing audio at the user. There is exactly one now,
 * with exactly one guard, and every binding goes through it.
 *
 * The whole screen stays usable while a job runs. Nothing here disables itself
 * on a running job except the two model actions, which queue.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import {
  Film,
  FolderOpen,
  HelpCircle,
  Loader2,
  MoreHorizontal,
  TriangleAlert,
} from "lucide-react";
import { AppHeader } from "../components/AppShell";
import { JobBar } from "../components/JobBar";
import { ScriptPane, filterSegments, type ScriptFilter } from "../components/ScriptPane";
import { SelectionPanel } from "../components/SelectionPanel";
import { StageTrack } from "../components/StageTrack";
import { Timeline } from "../components/Timeline";
import { VideoPlayer } from "../components/VideoPlayer";
import type { EditTarget } from "../components/ScriptRow";
import { Badge, Button, ConfirmButton, Empty, ErrorBar, Eyebrow, Kbd, Popover } from "../components/ui";
import { api } from "../lib/api";
import { stopClip, toggleClip, useClipPlayback } from "../lib/clipAudio";
import { isDesktop, revealPath } from "../lib/desktop";
import { FIXTURE_PROJECT } from "../lib/fixtures";
import { timecode } from "../lib/format";
import { STATE_META, segmentState, totalDuration, type SegmentState } from "../lib/segments";
import { summarizeStages } from "../lib/stages";
import { activeJob, useProject } from "../lib/useProject";
import { useTransport } from "../lib/useTransport";
import type { Job, ProjectDetail, Segment } from "../lib/types";
import type { StageProgress } from "../lib/useProject";

const ZOOM_STEPS = [0.5, 1, 2, 4, 8, 16, 32];

/** Every shortcut the editor binds, in the order they are learned. */
const SHORTCUTS: [string[], string][] = [
  [["space"], "play / pause"],
  [["↑", "↓"], "previous / next line"],
  [["↵"], "edit the selected translation"],
  [["esc"], "leave the field without saving"],
  [["a"], "play the original for the selected line"],
  [["b"], "play the dub for the selected line"],
  [["k"], "switch between dub and keep"],
  [["s"], "split the selection at the playhead"],
  [["+", "−"], "zoom the timeline"],
  [["⌘", "f"], "search the script"],
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
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<ScriptFilter>("all");
  const [editing, setEditing] = useState<EditTarget>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const playingUrl = useClipPlayback();

  const selected = useMemo(
    () => segments.find((seg) => seg.uid === selectedUid) ?? null,
    [segments, selectedUid],
  );

  const visible = useMemo(
    () => filterSegments(segments, query, filter),
    [filter, query, segments],
  );

  const previewUrl = project?.outputs.preview
    ? api.mediaUrl(name, project.outputs.preview)
    : null;

  const counts = useMemo(() => {
    const out: Record<SegmentState, number> = {
      dubbed: 0,
      kept: 0,
      failed: 0,
      unvoiced: 0,
      untranslated: 0,
    };
    for (const seg of segments) out[segmentState(seg)] += 1;
    return out;
  }, [segments]);

  /**
   * Selecting a segment also moves the playhead to it — they are one gesture.
   *
   * Depends on `transport.seek` rather than on `transport`: the transport hook
   * returns a fresh object literal every render, so closing over the whole
   * thing would rebuild this callback four times a second during playback and
   * re-render all two hundred memoized rows with it. `seek` itself is stable.
   */
  const seek = transport.seek;
  const selectAndSeek = useCallback(
    (uid: string) => {
      actions.select(uid);
      const seg = segments.find((s) => s.uid === uid);
      if (seg) seek(seg.start);
    },
    [actions, seek, segments],
  );

  /** ↑/↓ walk what is on screen, not what is in the run. */
  const step = useCallback(
    (delta: number) => {
      if (visible.length === 0) return;
      const index = visible.findIndex((seg) => seg.uid === selectedUid);
      const next = Math.max(0, Math.min(visible.length - 1, (index < 0 ? 0 : index) + delta));
      selectAndSeek(visible[next].uid);
    },
    [selectAndSeek, selectedUid, visible],
  );

  const zoomOut = useCallback(
    () => setZoom((z) => ZOOM_STEPS[Math.max(0, ZOOM_STEPS.indexOf(z) - 1)] ?? z),
    [],
  );
  const zoomIn = useCallback(
    () => setZoom((z) => ZOOM_STEPS[Math.min(ZOOM_STEPS.length - 1, ZOOM_STEPS.indexOf(z) + 1)] ?? z),
    [],
  );

  const toggleKeep = useCallback(
    (seg: Segment) =>
      void actions.patch(seg.uid, {
        keep: !seg.keep,
        keep_reason: seg.keep ? undefined : "manual",
      }),
    [actions],
  );

  /** A committed edit is a PATCH; an empty one never gets this far (ScriptRow). */
  const commit = useCallback(
    (uid: string, field: "text" | "text_en", value: string) => {
      setEditing(null);
      const seg = segments.find((s) => s.uid === uid);
      if (!seg) return;
      if (value === (field === "text" ? seg.text : (seg.text_en ?? ""))) return;
      void actions.patch(uid, { [field]: value });
    },
    [actions, segments],
  );

  const splitAt =
    selected && transport.currentTime > selected.start + 0.05 &&
    transport.currentTime < selected.end - 0.05
      ? transport.currentTime
      : null;

  // Playback of a clip is per-selection: moving on must not leave the previous
  // line's audio sounding under the new one.
  useEffect(() => stopClip, []);
  useEffect(() => {
    stopClip();
  }, [selectedUid]);

  /**
   * The editor's only keydown listener, with the editor's only guard.
   *
   * ⌘F is checked before the modifier bail, because it is the one binding that
   * *is* a modifier chord; everything after it is a bare key and a bare key
   * pressed while a field has focus belongs to the field.
   */
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key === "f") {
        event.preventDefault();
        searchRef.current?.focus();
        searchRef.current?.select();
        return;
      }
      const target = event.target as HTMLElement | null;
      if (
        target &&
        (/^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName) || target.isContentEditable)
      ) {
        return;
      }
      if (event.metaKey || event.ctrlKey || event.altKey) return;

      switch (event.key) {
        case " ":
          event.preventDefault();
          transport.toggle();
          break;
        case "ArrowDown":
          event.preventDefault();
          step(1);
          break;
        case "ArrowUp":
          event.preventDefault();
          step(-1);
          break;
        case "Enter":
          if (selected) {
            event.preventDefault();
            setEditing({ uid: selected.uid, field: "text_en" });
          }
          break;
        case "Escape":
          setEditing(null);
          stopClip();
          break;
        case "a":
          if (selected) {
            event.preventDefault();
            toggleClip(selected.media?.source ?? null);
          }
          break;
        case "b":
          if (selected) {
            event.preventDefault();
            toggleClip(selected.media?.play ?? selected.media?.tts ?? null);
          }
          break;
        case "k":
          if (selected) {
            event.preventDefault();
            toggleKeep(selected);
          }
          break;
        case "s":
          if (selected && splitAt != null) {
            event.preventDefault();
            void actions.split(selected.uid, splitAt);
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
  }, [actions, selected, splitAt, step, toggleKeep, transport, zoomIn, zoomOut]);

  const job = activeJob(state.jobs);
  const queued = state.jobs.filter((j) => j.status === "queued" && j.id !== job?.id).length;

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-plane">
      <AppHeader
        actions={
          <>
            <ConfirmButton
              variant="primary"
              size="sm"
              confirmLabel="Render"
              message="Re-render the preview. This re-runs timeline, mix and report and replaces preview.mp4 — a full video re-encode, typically a few minutes."
              onConfirm={() => void actions.render()}
            >
              <Film className="h-3.5 w-3.5" />
              Render preview
            </ConfirmButton>
            <RunMenu
              project={project}
              name={name}
              counts={counts}
              onSeek={transport.seek}
            />
            <ShortcutHelp />
          </>
        }
      >
        <span
          className="min-w-0 flex-1 truncate text-[12.5px] font-semibold text-primary"
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
      />
      {state.error ? <ErrorBar message={state.error} onDismiss={actions.dismissError} /> : null}

      {state.loading ? (
        <Empty title="Opening the run">
          <Loader2 className="mx-auto mt-2 h-5 w-5 animate-spin" aria-label="Loading" />
        </Empty>
      ) : (
        <>
          <div className="flex min-h-0 flex-1 flex-col xl:flex-row">
            <ScriptPane
              segments={segments}
              selectedUid={selectedUid}
              currentTime={transport.currentTime}
              busyUids={state.busyUids}
              editing={editing}
              playingUrl={playingUrl}
              query={query}
              filter={filter}
              searchRef={searchRef}
              onQuery={setQuery}
              onFilter={setFilter}
              onSelect={selectAndSeek}
              onEdit={setEditing}
              onCommit={commit}
              onPlay={toggleClip}
              onToggleKeep={toggleKeep}
              onRetranslateMany={(uids) => void actions.retranslate(uids)}
              onResynthesizeMany={(uids) => void actions.resynthesize(uids)}
            />

            {/*
              The viewer column. Fixed width and never the pane that grows —
              past 640px a talking-head preview is not telling the reviewer
              anything the 480px one did not.
            */}
            <aside className="flex min-h-0 max-h-[45%] shrink-0 flex-col bg-surface xl:max-h-none xl:w-[42%] xl:min-w-[26rem] xl:max-w-[40rem]">
              <VideoPlayer
                src={previewUrl}
                transport={transport}
                duration={total}
                className="h-40 shrink-0 xl:h-auto xl:aspect-video"
                placeholder={
                  <PreviewPlaceholder project={project} job={job} stage={state.stage} />
                }
              />
              {selected ? (
                <SelectionPanel
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
                <Empty title="No line selected" className="min-h-0 flex-1">
                  Pick a line in the script — everything that is true about it, and cannot fit
                  on a row, is here.
                </Empty>
              )}
            </aside>
          </div>

          <Timeline
            segments={segments}
            total={total}
            currentTime={transport.currentTime}
            selectedUid={selectedUid}
            busyUids={state.busyUids}
            pxPerSecond={zoom}
            splitAt={splitAt}
            onSelect={selectAndSeek}
            onSeek={transport.seek}
            onZoomIn={zoomIn}
            onZoomOut={zoomOut}
            onSplit={(at) => selected && void actions.split(selected.uid, at)}
          />
        </>
      )}
    </div>
  );
}

/**
 * The run, behind one button.
 *
 * Three things that are true about a run and are not per-line: how many
 * segments are in what state, where there is audible speech that nothing
 * covers, and where the files are. None of them changes while you work through
 * a line, and none of them earns permanent chrome — but the middle one is the
 * highest-value readout the report produces, so every gap is a button that
 * seeks to it.
 *
 * This is what is left of the 194-line RunSummary that used to fill the rail
 * whenever nothing was selected. The rest of it — the coverage bars, the drift
 * and speed stats, the keep-reason breakdown — was a report rendered twice,
 * once here and once in `report.json`, and it was never the thing a reviewer
 * was looking at the screen to find.
 */
function RunMenu({
  project,
  name,
  counts,
  onSeek,
}: {
  project: ProjectDetail | null;
  name: string;
  counts: Record<SegmentState, number>;
  onSeek: (time: number) => void;
}) {
  const gaps = project?.report?.uncovered_audible ?? [];
  const preview = project?.outputs.preview;

  return (
    <Popover
      label="Run health and files"
      title="Run health"
      trigger={<MoreHorizontal className="h-3.5 w-3.5" />}
      className="w-[21rem]"
    >
      <dl className="flex flex-col gap-1 text-[12.5px]">
        {(Object.keys(STATE_META) as SegmentState[])
          .filter((s) => counts[s] > 0)
          .map((s) => (
            <div key={s} className="flex items-baseline gap-2">
              <dt className="flex items-center gap-1.5 text-secondary">
                <span aria-hidden style={{ color: STATE_META[s].token }}>
                  {STATE_META[s].glyph}
                </span>
                {STATE_META[s].label}
              </dt>
              <span className="h-px flex-1 bg-border" aria-hidden />
              <dd className="font-mono tabular-nums text-primary">{counts[s]}</dd>
            </div>
          ))}
      </dl>

      {gaps.length > 0 ? (
        <>
          <Eyebrow className="mt-3.5 mb-1.5">Audible, uncovered — {gaps.length}</Eyebrow>
          <p className="mb-1.5 text-[11px] leading-relaxed text-muted">
            Speech here is in the source but no segment claims it, so the dub plays the
            original. Jump to one and listen.
          </p>
          <ul className="flex max-h-48 flex-col gap-1 overflow-y-auto">
            {gaps.map((gap) => (
              <li key={`${gap.start}-${gap.end}`}>
                <button
                  type="button"
                  onClick={() => onSeek(gap.start)}
                  className="flex w-full items-center gap-2 rounded-lg border border-border bg-raised px-2 py-1 text-left text-[12.5px] transition-colors hover:border-axis hover:bg-sunken"
                >
                  <TriangleAlert className="h-3 w-3 shrink-0 text-muted" aria-hidden />
                  <span className="font-mono tabular-nums text-primary">
                    {timecode(gap.start)}
                  </span>
                  <span className="ml-auto text-muted">{gap.duration.toFixed(1)}s</span>
                </button>
              </li>
            ))}
          </ul>
        </>
      ) : null}

      <Eyebrow className="mt-3.5 mb-1.5">This run</Eyebrow>
      <dl className="flex flex-col gap-0.5 text-[11px] text-muted">
        <div className="flex gap-2">
          <dt className="shrink-0">languages</dt>
          <dd className="ml-auto font-mono text-secondary">
            {project?.source.src_lang} → {project?.source.tgt_lang}
          </dd>
        </div>
        <div className="flex gap-2">
          <dt className="shrink-0">run dir</dt>
          <dd className="ml-auto truncate font-mono text-secondary">{name}</dd>
        </div>
        {project?.source.transcript_origin ? (
          <div className="flex gap-2">
            <dt className="shrink-0">transcript</dt>
            <dd className="ml-auto font-mono text-secondary">
              {project.source.transcript_origin}
            </dd>
          </div>
        ) : null}
      </dl>

      {/* Only inside the desktop shell: in a browser tab there is no Finder to
          reveal anything in, and a button that silently does nothing is worse
          than one that is not there. */}
      {isDesktop() && preview ? (
        <Button
          size="sm"
          className="mt-3 w-full"
          onClick={() => void revealPath(preview)}
        >
          <FolderOpen className="h-3.5 w-3.5" />
          Show preview.mp4 in Finder
        </Button>
      ) : null}
    </Popover>
  );
}

/**
 * The keyboard, behind one button.
 *
 * It used to share this popover with a colour legend. The legend is gone: every
 * script row now carries its state as a word, which is the same information in
 * the place it is needed, and a key to an encoding that is already spelled out
 * is a key to nothing.
 */
function ShortcutHelp() {
  return (
    <Popover
      label="Keyboard shortcuts"
      title="Keyboard"
      trigger={<HelpCircle className="h-3.5 w-3.5" />}
      className="w-[19rem]"
    >
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
 * This is the screen a user stares at for the entire length of a run, so it has
 * to be the run's status board rather than an apology. It no longer carries its
 * own Render button or its own progress bar: the header has the first and the
 * job strip has the second, both of them three inches away and both of them
 * permanent. What it keeps that nothing else has is the stage track — which of
 * the nine stages this run has got through.
 */
function PreviewPlaceholder({
  project,
  job,
  stage,
}: {
  project: ProjectDetail | null;
  job: Job | null;
  stage: StageProgress | null;
}) {
  const summary = summarizeStages(project?.stages);
  const working = job != null && (job.status === "running" || job.status === "queued");

  return (
    <div className="grid h-full place-items-center px-4 py-3">
      <div className="w-full max-w-sm text-center">
        <Eyebrow>{working ? "Working" : summary.complete ? "No preview file" : "Preview"}</Eyebrow>

        <p className="mt-1.5 text-[14px] font-semibold text-primary">
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

        <p className="mt-1 text-[11px] leading-relaxed text-muted">
          {working
            ? (stage?.message ?? "preview.mp4 is written by the mix stage, at the end.")
            : summary.complete
              ? "Render preview re-runs timeline, mix and report and writes preview.mp4."
              : "preview.mp4 is written by the mix stage. Until then the transport is a virtual clock, so the timeline and the script still work."}
        </p>

        <div className="mt-3 flex justify-center">
          <StageTrack stages={project?.stages} current={stage?.stage ?? null} />
        </div>
      </div>
    </div>
  );
}
