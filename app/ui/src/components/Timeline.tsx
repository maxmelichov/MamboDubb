/**
 * The segment timeline.
 *
 * Two lanes on one shared time axis: SOURCE is where the speech is in the
 * original, OUTPUT is where the dubbed audio actually landed. They are drawn
 * separately because the gap between them *is* drift — the thing a reviewer is
 * looking for — and a single lane would hide it.
 *
 * Encoding discipline (dataviz): four states, each with a validated hue, a
 * glyph and a word. Colour never carries meaning alone; the hue is the fast
 * channel and the glyph is the correct one. Unclaimed time gets no hue at all —
 * it is an absence, so it is a neutral 135° hatch.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { cn } from "../lib/classNames";
import { timecode } from "../lib/format";
import {
  STATE_META,
  UNCLAIMED_META,
  placedSpan,
  segmentState,
  unclaimedSpans,
  type SegmentState,
} from "../lib/segments";
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
  onSelect,
  onSeek,
}: {
  segments: Segment[];
  total: number;
  currentTime: number;
  selectedUid: string | null;
  busyUids: string[];
  pxPerSecond: number;
  onSelect: (uid: string) => void;
  onSeek: (time: number) => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<{ seg: Segment; x: number } | null>(null);

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

  const seekFromEvent = (event: React.MouseEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    onSeek(Math.max(0, (event.clientX - rect.left) / pxPerSecond));
  };

  return (
    <div className="relative flex min-h-0 flex-col">
      <div ref={scrollRef} className="overflow-x-auto overflow-y-hidden">
        <div className="relative select-none" style={{ width }}>
          {/* Axis. Recessive: hairline ticks, muted labels, tabular figures. */}
          <div
            className="relative h-5 cursor-pointer border-b border-grid"
            onClick={seekFromEvent}
          >
            {ticks(total, tickStep).map((t) => (
              <div key={t} className="absolute top-0 h-full" style={{ left: t * pxPerSecond }}>
                <div className="h-2 w-px bg-axis" />
                <span className="absolute left-1 top-1 text-[10px] tabular-nums text-muted">
                  {timecode(t, 0)}
                </span>
              </div>
            ))}
          </div>

          <Lane label="Source">
            {gaps.map((gap) => (
              <div
                key={`gap-${gap.start}`}
                className="hatch-unclaimed absolute inset-y-1 rounded-sm border border-dashed"
                style={{
                  left: gap.start * pxPerSecond + 1,
                  width: Math.max(MIN_MARK_PX, (gap.end - gap.start) * pxPerSecond - 2),
                  borderColor: "color-mix(in srgb, var(--color-unclaimed) 45%, transparent)",
                }}
                title={`Unclaimed ${timecode(gap.start)} – ${timecode(gap.end)}`}
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
                onHover={setHover}
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
                  onHover={setHover}
                />
              );
            })}
          </Lane>

          {/* Playhead sits above both lanes so the drift between them is legible. */}
          <div
            className="pointer-events-none absolute top-0 bottom-0 w-px bg-primary"
            style={{ left: currentTime * pxPerSecond }}
          >
            <div className="absolute -left-[3px] top-0 h-1.5 w-[7px] rounded-b-sm bg-primary" />
          </div>
        </div>
      </div>

      {hover ? <Tooltip seg={hover.seg} x={hover.x} /> : null}
    </div>
  );
}

function Lane({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="relative h-9 border-b border-grid last:border-b-0">
      <span className="pointer-events-none absolute left-1 top-0.5 z-10 text-[9px] font-semibold uppercase tracking-[0.14em] text-muted">
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
  onHover,
}: {
  seg: Segment;
  start: number;
  end: number;
  pxPerSecond: number;
  selected: boolean;
  busy: boolean;
  muted?: boolean;
  onSelect: (uid: string) => void;
  onHover: (value: { seg: Segment; x: number } | null) => void;
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
      onClick={() => onSelect(seg.uid)}
      onMouseEnter={(event) =>
        onHover({ seg, x: event.currentTarget.getBoundingClientRect().left })
      }
      onMouseLeave={() => onHover(null)}
      aria-label={`Segment ${seg.id}, ${meta.label}, ${timecode(start)} to ${timecode(end)}`}
      aria-pressed={selected}
      className={cn(
        "absolute inset-y-1.5 overflow-hidden rounded-sm text-left",
        "focus:outline-none focus-visible:ring-2 focus-visible:ring-primary",
        selected && "ring-2 ring-primary",
        busy && "animate-pulse",
        muted && "opacity-40",
      )}
      style={{ left: start * pxPerSecond + 1, width, backgroundColor: meta.token }}
    >
      {showGlyph ? (
        <span
          aria-hidden
          className="pointer-events-none absolute inset-y-0 left-0.5 flex items-center text-[9px] leading-none text-white drop-shadow-[0_0_1px_rgba(0,0,0,0.6)]"
        >
          {meta.glyph}
        </span>
      ) : null}
    </button>
  );
}

function Tooltip({ seg, x }: { seg: Segment; x: number }) {
  const meta = STATE_META[segmentState(seg)];
  return (
    <div
      className="pointer-events-none fixed z-30 max-w-[22rem] rounded-md border border-border bg-raised p-2 text-[12px] shadow-lg"
      style={{ left: Math.min(x, window.innerWidth - 360), top: undefined, bottom: 12 }}
    >
      <div className="flex items-center gap-2 text-muted">
        <span aria-hidden style={{ color: meta.token }}>
          {meta.glyph}
        </span>
        <span className="font-semibold text-primary">#{seg.id}</span>
        <span>{meta.label}</span>
        <span className="tabular-nums">
          {timecode(seg.start)} – {timecode(seg.end)}
        </span>
      </div>
      <p className="auto-dir mt-1 line-clamp-2 text-secondary">{seg.text}</p>
      {seg.text_en ? <p className="mt-0.5 line-clamp-2 text-primary">{seg.text_en}</p> : null}
    </div>
  );
}

/** Identity is never colour-alone: the legend is always present. */
export function TimelineLegend({ counts }: { counts: Record<SegmentState, number> }) {
  const entries: { key: string; token: string; glyph: string; label: string; count?: number }[] = (
    Object.keys(STATE_META) as SegmentState[]
  ).map((state) => ({
    key: state,
    token: STATE_META[state].token,
    glyph: STATE_META[state].glyph,
    label: STATE_META[state].label,
    count: counts[state],
  }));
  entries.push({ key: "unclaimed", ...UNCLAIMED_META });

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-secondary">
      {entries.map((entry) => (
        <span key={entry.key} className="inline-flex items-center gap-1.5">
          <span
            aria-hidden
            className={cn("h-2.5 w-4 rounded-sm", entry.key === "unclaimed" && "hatch-unclaimed")}
            style={{
              backgroundColor: entry.key === "unclaimed" ? "transparent" : entry.token,
              border: entry.key === "unclaimed" ? "1px dashed var(--color-unclaimed)" : undefined,
            }}
          />
          <span aria-hidden style={{ color: entry.token }}>
            {entry.glyph}
          </span>
          {entry.label}
          {entry.count != null ? (
            <span className="tabular-nums text-muted">{entry.count}</span>
          ) : null}
        </span>
      ))}
    </div>
  );
}

function ticks(total: number, step: number): number[] {
  const out: number[] = [];
  for (let t = 0; t <= total; t += step) out.push(t);
  return out;
}
