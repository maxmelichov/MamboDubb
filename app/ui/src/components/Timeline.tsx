/**
 * The segment timeline — a map, not an edit surface.
 *
 * Two lanes on one shared time axis: SOURCE is where the speech is in the
 * original, OUTPUT is where the dubbed audio actually landed. They are drawn
 * separately because the gap between them *is* drift — the thing a reviewer is
 * looking for — and a single lane would hide it. Unclaimed time gets no hue at
 * all: it is an absence, so it is a neutral 135° hatch.
 *
 * Behind the marks, each lane draws its actual audio: `GET …/peaks` returns a
 * few thousand normalized amplitudes and the lane draws them mirrored about its
 * centre line. That is not decoration. A mark says a segment claims 12.4s to
 * 15.1s; the waveform says whether anybody is *talking* in it — so a mark that
 * has drifted off its speech, a kept span that is actually silent, and a gap
 * the hatch calls empty but the ear does not are all visible without pressing
 * play. The picture is scaled by the audio's own `duration`, never by the
 * timeline's, so it stays time-aligned with the marks on top of it.
 *
 * It is a 128px strip across the bottom now rather than a third of the screen,
 * and it lost three things in the move. The per-mark hover tooltip is gone: it
 * repeated the two lines of text that the script pane now shows in full, three
 * inches away, permanently. The legend is gone: four states, each already
 * carrying its word on every script row. And the "Timeline" eyebrow is gone —
 * a strip of clips on a time axis at the bottom of a video editor does not need
 * to be captioned.
 *
 * ## The three things a strip of marks has to get right
 *
 * 1. **It has to fit.** The lanes used to be `calc(50% - 0.5rem)` of the strip,
 *    which added up to exactly the strip — so the second lane ended on the
 *    window's last pixel and its marks were shaved off by the screen edge. The
 *    height is a flex column now, ruler and two lanes, with padding at the
 *    bottom that belongs to the strip rather than to the lane.
 * 2. **The labels need somewhere to be.** SOURCE and OUTPUT used to float over
 *    the first few seconds of their own lane on a translucent chip, which put
 *    them on top of the marks — the ones at 0:00, which is where a reviewer
 *    starts. They sit in a fixed gutter to the left of the scrolling area now,
 *    the way a track header does in every editor that has ever had one.
 * 3. **A mark is a shape, not a slab.** Filled at full strength, a run where
 *    every line is kept is a solid amber wall a hundred marks long, and the
 *    playhead — the one thing that moves — is lost in it. The fill is a wash
 *    and the state hue is spent on the *edge*, which is what makes a boundary
 *    legible; hover and selection bring the fill up. Then the two brightest
 *    things on the strip are the playhead and whatever you picked, which is
 *    the correct answer to "what am I looking at".
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
import { Button, ButtonGroup, ConfirmButton, StateIcon } from "./ui";
import { STATE_META, placedSpan, segmentState, unclaimedSpans } from "../lib/segments";
import type { Peaks, Segment } from "../lib/types";

const TICK_STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
const MIN_MARK_PX = 3;
/** The lane-label gutter — wide enough for "OUTPUT" at the tracked-out size. */
const GUTTER = "w-16";

export function Timeline({
  segments,
  total,
  currentTime,
  selectedUid,
  busyUids,
  pxPerSecond,
  sourcePeaks,
  dubPeaks,
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
  /** The original audio's envelope, or null while there is none to draw. */
  sourcePeaks: Peaks | null;
  /** The finished mix's envelope — null until the mix stage has run. */
  dubPeaks: Peaks | null;
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
   * leaves the strip — which is most of them, the strip is 128px tall — keeps
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
    <section className="relative flex h-32 shrink-0 border-t border-border bg-surface pb-1.5">
      {/*
        The track headers. Outside the scroll container on purpose: a label
        inside it either scrolls away from the lane it names or is pinned over
        the marks it is supposed to be labelling. Height-matched to the lanes
        by mirroring their flex structure — one spacer for the ruler, then one
        `flex-1` per lane.
      */}
      <div
        className={cn(
          "flex shrink-0 flex-col border-e border-grid bg-sunken text-[9px] font-bold",
          "uppercase tracking-[0.12em] text-muted",
          GUTTER,
        )}
      >
        <div className="h-5 shrink-0 border-b border-grid" aria-hidden />
        <span data-lane-label className="flex min-h-0 flex-1 items-center border-b border-grid px-2">
          Source
        </span>
        <span data-lane-label className="flex min-h-0 flex-1 items-center px-2">
          Output
        </span>
      </div>

      <div ref={scrollRef} className="min-h-0 min-w-0 flex-1 overflow-x-auto overflow-y-hidden">
        <div
          ref={laneRef}
          className="relative flex h-full cursor-col-resize select-none flex-col"
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
          <div className="relative h-5 shrink-0 border-b border-grid bg-sunken">
            {ticks(total, tickStep).map((t) => (
              <div key={t} className="absolute top-0 h-full" style={{ left: t * pxPerSecond }}>
                <div className="h-1.5 w-px bg-axis" />
                <span className="absolute left-1 top-0.5 font-mono text-[9px] tabular-nums text-muted">
                  {timecode(t, 0)}
                </span>
              </div>
            ))}
          </div>

          <Lane>
            <Waveform peaks={sourcePeaks} lane="source" pxPerSecond={pxPerSecond} />
            {gaps.map((gap) => (
              <div
                key={`gap-${gap.start}`}
                className="hatch-unclaimed absolute inset-y-2 rounded-[3px] border border-dashed"
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

          <Lane last>
            <Waveform peaks={dubPeaks} lane="dub" pxPerSecond={pxPerSecond} />
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

          {/* Playhead sits above both lanes so the drift between them is
              legible, and it is the most salient thing on the strip: full ink
              against washed marks, with a head wide enough to grab by eye. */}
          <div
            data-playhead
            className="pointer-events-none absolute top-0 bottom-0 z-20 w-[1.5px] -translate-x-[0.75px] bg-primary"
            style={{ left: currentTime * pxPerSecond }}
          >
            <div className="absolute -left-[3.25px] top-0 h-2.5 w-2 rounded-b-[3px] bg-primary" />
          </div>
        </div>
      </div>

      {/*
        The strip's own controls, floated over its right edge rather than given
        a toolbar row of their own — 128px is the whole budget and a header
        would take a fifth of it. They stop the scrub because they sit above
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

/**
 * One lane's plotting area. It carries no label — the gutter does that — so
 * every pixel of it belongs to the marks, including the ones at 0:00.
 */
function Lane({ last, children }: { last?: boolean; children: React.ReactNode }) {
  return (
    <div className={cn("relative min-h-0 flex-1", !last && "border-b border-grid")}>
      {children}
    </div>
  );
}

/**
 * One lane's audio, mirrored about its centre.
 *
 * SVG rather than canvas, and stretched rather than sampled per pixel. A lane
 * at 32px/s over a twenty-minute run is 38,000px wide — past the maximum canvas
 * dimension on Safari and a lot of memory everywhere else — while a path with a
 * `viewBox` of `0 0 n 100` and `preserveAspectRatio="none"` draws once and
 * scales for free at every zoom. The width is `duration * pxPerSecond` because
 * the audio's length is the thing that maps the buckets onto the time axis; use
 * the timeline's `total` (which is the max of the audio, the last segment and
 * the last placement) and the picture slides against the marks.
 *
 * `aria-hidden` and `pointer-events-none`: it is a backdrop for the marks,
 * which stay the clickable thing in the lane.
 */
function Waveform({
  peaks,
  lane,
  pxPerSecond,
}: {
  peaks: Peaks | null;
  lane: "source" | "dub";
  pxPerSecond: number;
}) {
  const path = useMemo(() => {
    if (!peaks || peaks.duration <= 0 || peaks.peaks.length === 0) return null;
    const n = peaks.peaks.length;
    const top: string[] = [];
    const bottom: string[] = [];
    for (let i = 0; i < n; i += 1) {
      // A floor of 0.6 units keeps a silent stretch as a hairline rather than
      // as a hole: the eye reads the continuous line as "audio, quiet here".
      const amp = Math.max(0.6, Math.min(1, peaks.peaks[i]) * 48);
      top.push(`${i},${(50 - amp).toFixed(1)}`);
      bottom.push(`${i},${(50 + amp).toFixed(1)}`);
    }
    bottom.reverse();
    return `M${top.join("L")}L${bottom.join("L")}Z`;
  }, [peaks]);

  if (!path || !peaks) return null;

  return (
    <svg
      aria-hidden
      data-waveform={lane}
      viewBox={`0 0 ${peaks.peaks.length} 100`}
      preserveAspectRatio="none"
      /*
       * The height is stated, not inherited. An absolutely-positioned SVG with
       * a width, a `top`, a `bottom` and `height: auto` is over-constrained, so
       * the browser drops `bottom` and takes the height from the viewBox ratio
       * — which at 4px/s happened to land within a pixel of the lane and at
       * 32px/s was 338px tall, drawing the top eighth of the waveform in this
       * lane and the rest over the one below.
       */
      /* Inset to exactly the marks' box, so the audio reads as being *in* the
         clip rather than passing behind it. */
      className="pointer-events-none absolute left-0 top-2 h-[calc(100%-1rem)] text-muted"
      style={{ width: Math.max(1, peaks.duration * pxPerSecond) }}
    >
      <path d={path} fill="currentColor" fillOpacity={0.42} />
    </svg>
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
  const showIcon = width > 18;

  return (
    <button
      type="button"
      data-mark
      onClick={() => onSelect(seg.uid)}
      aria-label={`Segment ${seg.id}, ${meta.label}, ${timecode(start)} to ${timecode(end)}`}
      aria-pressed={selected}
      className={cn(
        "group/mark absolute inset-y-2 overflow-hidden rounded-[3px] border text-left transition-all",
        // Hover is a real affordance here: the marks are 3–40px wide and the
        // only other cue that they are clickable is the cursor.
        "hover:inset-y-1.5 hover:ring-2 hover:ring-primary/45",
        selected && "z-10 ring-2 ring-primary ring-offset-1 ring-offset-surface",
        busy && "animate-pulse",
        // A mark in the OUTPUT lane with nothing placed under it is a
        // *prediction*, not a placement, so it is drawn as one.
        muted && "border-dashed opacity-55",
      )}
      style={{
        left: start * pxPerSecond + 1,
        width,
        borderColor: meta.token,
        // The hue lives on the edge; the fill is a wash of it. A hundred
        // adjacent marks then read as a hundred marks rather than as one
        // coloured band, and the playhead stays the brightest thing here.
        backgroundColor: `color-mix(in srgb, ${meta.token} ${selected ? 55 : 22}%, transparent)`,
      }}
    >
      {showIcon ? (
        <StateIcon
          state={state}
          className="pointer-events-none absolute left-[3px] top-1/2 h-2 w-2 -translate-y-1/2 opacity-90"
        />
      ) : null}
    </button>
  );
}

function ticks(total: number, step: number): number[] {
  const out: number[] = [];
  for (let t = 0; t <= total; t += step) out.push(t);
  return out;
}
