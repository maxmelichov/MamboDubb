/**
 * The editor.
 *
 * Layout is Premiere-shaped on purpose: preview and timeline stack on the left
 * in playback order, the inspector holds the right rail so the selection never
 * moves under the cursor, and the segment table sits under the timeline as the
 * dense, sortable-by-eye view of the same data.
 *
 * The whole screen stays usable while a job runs. Nothing here disables itself
 * on a running job except the two model actions, which queue.
 *
 * Chrome is the same system as the two page screens — same tokens, same
 * primitives, same eyebrow labels — but tuned for density: every horizontal
 * band that is not the timeline or the table is a 2-line toolbar on the sunken
 * tone, so the eye can tell "controls" from "content" without reading.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { Film, Loader2, Minus, Plus, RefreshCw } from "lucide-react";
import { AppHeader } from "../components/AppShell";
import { JobBar } from "../components/JobBar";
import { SegmentInspector } from "../components/SegmentInspector";
import { SegmentList } from "../components/SegmentList";
import { Timeline, TimelineLegend } from "../components/Timeline";
import { VideoPlayer } from "../components/VideoPlayer";
import { Badge, Button, ButtonGroup, Empty, ErrorBar, Eyebrow, Kbd } from "../components/ui";
import { api } from "../lib/api";
import { FIXTURE_PROJECT } from "../lib/fixtures";
import { segmentState, totalDuration, type SegmentState } from "../lib/segments";
import { activeJob, useProject } from "../lib/useProject";
import { useTransport } from "../lib/useTransport";

const ZOOM_STEPS = [0.5, 1, 2, 4, 8, 16, 32];

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
            <Button onClick={() => void actions.reload()} aria-label="Reload" size="sm">
              <RefreshCw className="h-3.5 w-3.5" />
            </Button>
            <Button
              variant="primary"
              size="sm"
              onClick={() => {
                if (
                  window.confirm(
                    "Re-render the preview? This re-runs timeline, mix and report and replaces preview.mp4 — a full video re-encode.",
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

      <JobBar
        job={job}
        queued={queued}
        stage={state.stage}
        connected={state.connected}
        onCancel={(id) => void actions.cancel(id)}
      />
      {state.error ? <ErrorBar message={state.error} onDismiss={actions.dismissError} /> : null}

      {state.loading ? (
        <Empty>
          <Loader2 className="mx-auto h-5 w-5 animate-spin" aria-label="Loading" />
        </Empty>
      ) : (
        <div className="flex min-h-0 flex-1">
          <div className="flex min-w-0 flex-1 flex-col">
            <div className="h-[38%] min-h-40 shrink-0 border-b border-border">
              <VideoPlayer
                src={previewUrl}
                transport={transport}
                duration={total}
                title={project?.source.title ?? name}
              />
            </div>

            <section ref={timelineRef} className="shrink-0 border-b border-border bg-surface">
              {/* The legend wraps inside its own cell rather than pushing the
                  zoom control onto a second row of its own. */}
              <div className="flex items-start gap-3 border-b border-border bg-sunken px-3 py-2">
                <Eyebrow className="mt-1.5 shrink-0">Timeline</Eyebrow>
                <div className="min-w-0 flex-1">
                  <TimelineLegend counts={counts} />
                </div>
                <div className="flex shrink-0 items-center gap-3 pt-0.5">
                  <span className="hidden items-center gap-1.5 text-[11px] text-muted lg:flex">
                    <Kbd>←</Kbd>
                    <Kbd>→</Kbd> segment
                    <Kbd>space</Kbd> play
                    <Kbd>k</Kbd> keep
                    <Kbd>a</Kbd>
                    <Kbd>b</Kbd> compare
                  </span>
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

            <div className="min-h-0 flex-1 bg-surface">
              <SegmentList
                segments={segments}
                selectedUid={selectedUid}
                currentTime={transport.currentTime}
                busyUids={state.busyUids}
                onSelect={selectAndSeek}
              />
            </div>
          </div>

          <aside className="w-[26rem] shrink-0 border-l border-border bg-surface">
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
              <Empty>
                Pick a segment in the timeline or the list.
                <br />
                <span className="mt-2 inline-flex items-center gap-1.5 text-[12px]">
                  <Kbd>←</Kbd> <Kbd>→</Kbd> step through them.
                </span>
              </Empty>
            )}
          </aside>
        </div>
      )}
    </div>
  );
}
