/**
 * The segment timeline — a map, not an edit surface.
 *
 * Two lanes on one shared time axis: SOURCE is where the speech is in the
 * original, OUTPUT is where the dubbed audio actually landed. They are drawn
 * separately because the gap between them *is* drift — the thing a reviewer is
 * looking for — and a single lane would hide it. Unclaimed time gets no hue at
 * all: it is an absence, so it is a neutral 135° hatch.
 *
 * It is a 112px strip across the bottom now rather than a third of the screen,
 * and it lost three things in the move. The per-mark hover tooltip is gone: it
 * repeated the two lines of text that the script pane now shows in full, three
 * inches away, permanently. The legend is gone: four states, each already
 * carrying its word on every script row. And the "Timeline" eyebrow is gone —
 * a strip of clips on a time axis at the bottom of a video editor does not need
 * to be captioned.
 *
 * What it gained is its own controls (zoom, split at the playhead) at the right
 * edge, where they belong, and drag-to-scrub across the whole strip. What it
 * still refuses to do is move anything: `timeline.place()` is the sole authority
 * on where audio goes, so there is no dragging a clip and no trim handle. This
 * is a picture of the placement, and the way to change it is to change what is
 * placed.
 */

import { useCallback, useEffect, useMemo, useRef } from "react";
import { Minus, Plus, Scissors } from "lucide-react";
import { cn } from "../lib/classNames";
import { timecode } from "../lib/format";
import { Button, ButtonGroup, ConfirmButton } from "./ui";
import { STATE_META, placedSpan, segmentState, unclaimedSpans } from "../lib/segments";
import type { Segment } from "../lib/types";

const TICK_STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
const MIN_MARK_PX = 3;

export function Timeline({
  segments,
  total,
  currentTime,
  selectedUid,
  busyUids,
  pxPerSecond,
  splitAt,
  onSelect,
  onSeek,
  onZoomIn,
  onZoomOut,
  onSplit,
}: {
  segments: Segment[];
  total: number;
  currentTime: number;
  selectedUid: string | null;
  busyUids: string[];
  pxPerSecond: number;
  /** The playhead, when it is inside the selected segment — else null. */
  splitAt: number | null;
  onSelect: (uid: string) => void;
  onSeek: (time: number) => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onSplit: (at: number) => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const laneRef = useRef<HTMLDivElement>(null);

  const width = Math.max(320, total * pxPerSecond);
  const gaps = useMemo(() => unclaimedSpans(segments, total), [segments, total]);
  const tickStep = useMemo(
    () => TICK_STEPS.find((step) => step * pxPerSecond >= 64) ?? TICK_STEPS[TICK_STEPS.length - 1],
    [pxPerSecond],
  );

  // The timeline follows playback: keep the playhead on screen without
  // fighting a user who is scrolling somewhere else.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const x = currentTime * pxPerSecond;
    const left = el.scrollLeft;
    const right = left + el.clientWidth;
    if (x < left + 40 || x > right - 80) {
      el.scrollTo({ left: Math.max(0, x - el.clientWidth * 0.4), behavior: "smooth" });
    }
  }, [currentTime, pxPerSecond]);

  const seekFromClientX = useCallback(
    (clientX: number) => {
      const rect = laneRef.current?.getBoundingClientRect();
      if (!rect) return;
      onSeek(Math.max(0, (clientX - rect.left) / pxPerSecond));
    },
    [onSeek, pxPerSecond],
  );

  /**
   * Drag-to-scrub. Pointer capture rather than window listeners so a drag that
   * leaves the strip — which is most of them, the strip is 112px tall — keeps
   * scrubbing instead of stopping at the edge.
   *
   * A press that starts on a mark is that mark's click: selecting a segment is
   * the more common gesture and it must not be swallowed by the scrubber.
   */
  const onPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    if ((event.target as HTMLElement).closest("[data-mark]")) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    seekFromClientX(event.clientX);
  };

  return (
    <section className="relative flex h-28 shrink-0 flex-col border-t border-border bg-surface">
      <div ref={scrollRef} className="min-h-0 flex-1 overflow-x-auto overflow-y-hidden">
        <div
          ref={laneRef}
          className="relative h-full cursor-col-resize select-none"
          style={{ width }}
          onPointerDown={onPointerDown}
          onPointerMove={(event) => {
            if (event.currentTarget.hasPointerCapture(event.pointerId)) {
              seekFromClientX(event.clientX);
            }
          }}
          onPointerUp={(event) => event.currentTarget.releasePointerCapture(event.pointerId)}
        >
          {/* Axis. Recessive: hairline ticks, muted labels, tabular figures. */}
          <div className="relative h-4 border-b border-grid bg-sunken">
            {ticks(total, tickStep).map((t) => (
              <div key={t} className="absolute top-0 h-full" style={{ left: t * pxPerSecond }}>
                <div className="h-1.5 w-px bg-axis" />
                <span className="absolute left-1 top-0 font-mono text-[9px] tabular-nums text-muted">
                  {timecode(t, 0)}
                </span>
              </div>
            ))}
          </div>

          <Lane label="Source">
            {gaps.map((gap) => (
              <div
                key={`gap-${gap.start}`}
                className="hatch-unclaimed absolute inset-y-1.5 rounded-[3px] border border-dashed"
                style={{
                  left: gap.start * pxPerSecond + 1,
                  width: Math.max(MIN_MARK_PX, (gap.end - gap.start) * pxPerSecond - 2),
                  borderColor: "color-mix(in srgb, var(--color-unclaimed) 45%, transparent)",
                }}
              />
            ))}
            {segments.map((seg) => (
              <Mark
                key={seg.uid}
                seg={seg}
                start={seg.start}
                end={seg.end}
                pxPerSecond={pxPerSecond}
                selected={seg.uid === selectedUid}
                busy={busyUids.includes(seg.uid)}
                onSelect={onSelect}
              />
            ))}
          </Lane>

          <Lane label="Output">
            {segments.map((seg) => {
              const span = placedSpan(seg);
              return (
                <Mark
                  key={seg.uid}
                  seg={seg}
                  start={span.start}
                  end={span.end}
                  muted={!seg.place}
                  pxPerSecond={pxPerSecond}
                  selected={seg.uid === selectedUid}
                  busy={busyUids.includes(seg.uid)}
                  onSelect={onSelect}
                />
              );
            })}
          </Lane>

          {/* Playhead sits above both lanes so the drift between them is legible. */}
          <div
            className="pointer-events-none absolute top-0 bottom-0 z-20 w-px bg-primary"
            style={{ left: currentTime * pxPerSecond }}
          >
            <div className="absolute -left-[3.5px] top-0 h-2 w-2 rounded-b-full bg-primary" />
          </div>
        </div>
      </div>

      {/*
        The strip's own controls, floated over its right edge rather than given
        a toolbar row of their own — 112px is the whole budget and a header
        would take a quarter of it. They stop the scrub because they sit above
        the lane, which is correct: nobody means "seek" by "zoom".
      */}
      <div className="pointer-events-none absolute inset-y-0 right-0 z-30 flex items-center gap-1.5 bg-gradient-to-l from-surface via-surface/95 to-transparent pl-8 pr-2">
        <div className="pointer-events-auto flex items-center gap-1.5">
          <ConfirmButton
            size="xs"
            disabled={splitAt == null}
            title={
              splitAt == null
                ? "Select a segment and put the playhead inside it to split"
                : `Split at ${timecode(splitAt)}`
            }
            confirmLabel="Split"
            message={
              splitAt == null
                ? ""
                : `Split at ${timecode(splitAt)}. Both halves lose their translation and their clip — the line was written for the whole span.`
            }
            onConfirm={() => splitAt != null && onSplit(splitAt)}
          >
            <Scissors className="h-3 w-3" />
            Split
          </ConfirmButton>
          <ButtonGroup className="h-6">
            <Button variant="ghost" size="xs" onClick={onZoomOut} aria-label="Zoom out">
              <Minus className="h-3 w-3" />
            </Button>
            <span className="flex w-11 items-center justify-center bg-raised font-mono text-[11px] tabular-nums text-muted">
              {pxPerSecond}px/s
            </span>
            <Button variant="ghost" size="xs" onClick={onZoomIn} aria-label="Zoom in">
              <Plus className="h-3 w-3" />
            </Button>
          </ButtonGroup>
        </div>
      </div>
    </section>
  );
}

function Lane({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="relative h-[calc(50%-0.5rem)] border-b border-grid last:border-b-0">
      {/* The label rides the scroll container, so it needs its own backing to
          stay legible when a mark passes underneath it. */}
      <span className="pointer-events-none absolute left-1 top-0.5 z-10 rounded bg-surface/85 px-1 text-[9px] font-bold uppercase tracking-[0.16em] text-muted">
        {label}
      </span>
      {children}
    </div>
  );
}

function Mark({
  seg,
  start,
  end,
  pxPerSecond,
  selected,
  busy,
  muted,
  onSelect,
}: {
  seg: Segment;
  start: number;
  end: number;
  pxPerSecond: number;
  selected: boolean;
  busy: boolean;
  muted?: boolean;
  onSelect: (uid: string) => void;
}) {
  const state = segmentState(seg);
  const meta = STATE_META[state];
  // 1px inset each side = the 2px surface gap that keeps adjacent fills from
  // reading as one block.
  const width = Math.max(MIN_MARK_PX, (end - start) * pxPerSecond - 2);
  const showGlyph = width > 16;

  return (
    <button
      type="button"
      data-mark
      onClick={() => onSelect(seg.uid)}
      aria-label={`Segment ${seg.id}, ${meta.label}, ${timecode(start)} to ${timecode(end)}`}
      aria-pressed={selected}
      className={cn(
        "absolute inset-y-1.5 overflow-hidden rounded-[3px] text-left transition-all",
        // Hover is a real affordance here: the marks are 3–40px wide and the
        // only other cue that they are clickable is the cursor.
        "hover:inset-y-1 hover:ring-2 hover:ring-primary/45",
        selected && "ring-2 ring-primary ring-offset-1 ring-offset-surface",
        busy && "animate-pulse",
        muted && "opacity-40",
      )}
      style={{ left: start * pxPerSecond + 1, width, backgroundColor: meta.token }}
    >
      {showGlyph ? (
        <span
          aria-hidden
          className="pointer-events-none absolute inset-y-0 left-0.5 flex items-center text-[9px] leading-none"
          style={{ color: meta.onFill }}
        >
          {meta.glyph}
        </span>
      ) : null}
    </button>
  );
}

function ticks(total: number, step: number): number[] {
  const out: number[] = [];
  for (let t = 0; t <= total; t += step) out.push(t);
  return out;
}
