/**
 * The segment timeline a map, not an edit surface.
 *
 * Two lanes on one shared time axis: SOURCE is where the speech is in the
 * original, OUTPUT is where the dubbed audio actually landed. They are drawn
 * separately because the gap between them *is* drift the thing a reviewer is
 * looking for and a single lane would hide it. Unclaimed time gets no hue at
 * all: it is an absence, so it is a neutral 135° hatch.
 *
 * Behind the marks, each lane draws its actual audio: `GET …/peaks` returns a
 * few thousand normalized amplitudes and the lane draws them mirrored about its
 * centre line. That is not decoration. A mark says a segment claims 12.4s to
 * 15.1s; the waveform says whether anybody is *talking* in it so a mark that
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
 *    which added up to exactly the strip so the second lane ended on the
 *    window's last pixel and its marks were shaved off by the screen edge. The
 *    height is a flex column now, ruler and two lanes, with padding at the
 *    bottom that belongs to the strip rather than to the lane.
 * 2. **The labels need somewhere to be.** SOURCE and OUTPUT used to float over
 *    the first few seconds of their own lane on a translucent chip, which put
 *    them on top of the marks the ones at 0:00, which is where a reviewer
 *    starts. They sit in a fixed gutter to the left of the scrolling area now,
 *    the way a track header does in every editor that has ever had one.
 * 3. **A mark is a shape, not a slab.** Filled at full strength, a run where
 *    every line is kept is a solid wall a hundred marks long, and the playhead
 *    the one thing that moves is lost in it. The fill is a wash and the
 *    state hue is spent on the *edge*, which is what makes a boundary legible;
 *    hover and selection bring the fill up. Then the two brightest things on
 *    the strip are the playhead and whatever you picked, which is the correct
 *    answer to "what am I looking at".
 *
 *    The hue itself does the rest of the work, and only since it stopped being
 *    amber for the commonest state: a lane now reads as a quiet green run with
 *    the blue dubs, the amber waiting lines and the red failures standing out
 *    of it, rather than as one warning-coloured barcode with the exceptions
 *    hidden inside it.
 *
 * What it gained is its own controls (zoom, Fit, split at the playhead) at the
 * right edge, where they belong, and drag-to-scrub across the whole strip.
 *
 * The SOURCE lane's marks move now, because they are the one thing here the
 * user actually owns: a mark there is the segment's *claim* on the source
 * audio, and dragging it (body to move, edges to trim) sends the same
 * `PATCH {start, end}` the panel's bounds fields send with the same cost, the
 * clip and nothing else (`lib/patch.ts`). The neighbours are hard walls the
 * drag cannot cross overlap is refused by `timeline.place`'s assertion, so it
 * must be impossible to draw, not merely rejected later and the edges snap to
 * them and to the playhead, which are the three positions a trim is usually
 * aiming for. What still refuses to move is the OUTPUT lane: `timeline.place()`
 * is the sole authority on where audio *lands*, so that lane stays a picture of
 * the placement, and the way to change it is to change what is placed.
 *
 * Two later additions, both of which make the strip answer a question that was
 * being asked somewhere else. Zoom has a floor the scale at which the run is
 * exactly as wide as the strip and a Fit button that goes straight to it,
 * because an unbounded zoom-out ends in an empty strip with no way back but
 * counting presses. And while a search is running, the marks outside the result
 * set drop to 35%: this is the only view that shows all two hundred lines at
 * once, so it is the only one that can say *where* the eleven matches are.
 */

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Minus, Plus, Scissors } from "lucide-react";
import { cn } from "../lib/classNames";
import { timecode } from "../lib/format";
import { Button, ButtonGroup, ConfirmButton, StateIcon } from "./ui";
import {
  STATE_META,
  placedSpan,
  segmentState,
  unclaimedSpans,
  type Span,
} from "../lib/segments";
import type { Peaks, Segment } from "../lib/types";

const TICK_STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
const MIN_MARK_PX = 3;
/** The shortest span a trim can leave behind, in seconds see `edit.set_bounds`. */
const MIN_SPAN = 0.1;
/** How far a press travels before it is a drag and no longer a click, in px. */
const DRAG_SLOP_PX = 3;

/** Which part of a source mark the pointer grabbed. */
type DragMode = "move" | "start" | "end";

/** One retiming drag, from pointer-down to pointer-up. */
type DragSession = {
  uid: string;
  mode: DragMode;
  /** The span as it was when the pointer went down. */
  originStart: number;
  originEnd: number;
  grabX: number;
  /** The free window the drag may not leave: prev.end to next.start. */
  min: number;
  max: number;
  /** Crossed the slop threshold at least once so releasing is a save. */
  moved: boolean;
};
/** The lane-label gutter wide enough for "OUTPUT" at the tracked-out size. */
const GUTTER = "w-16";

/**
 * The zoom readout, at both ends of the range.
 *
 * "Fit" lands on `containerWidth / total`, which is a fraction and `3.4523px/s`
 * in an eleven-pixel monospace cell is not a number anybody reads. One decimal
 * below ten, whole numbers above, and never a trailing `.0`.
 */
function zoomLabel(pxPerSecond: number): string {
  if (pxPerSecond >= 10) return String(Math.round(pxPerSecond));
  const one = Math.round(pxPerSecond * 10) / 10;
  return Number.isInteger(one) ? String(one) : one.toFixed(1);
}

export function Timeline({
  segments,
  total,
  currentTime,
  selectedUid,
  busyUids,
  pxPerSecond,
  fitPxPerSecond,
  matchedUids,
  highlightGap,
  sourcePeaks,
  dubPeaks,
  stale = false,
  splitAt,
  onSelect,
  onSeek,
  onRetime,
  onViewport,
  onZoomIn,
  onZoomOut,
  onZoomBy,
  onFit,
  onSplit,
}: {
  segments: Segment[];
  total: number;
  currentTime: number;
  selectedUid: string | null;
  busyUids: string[];
  pxPerSecond: number;
  /**
   * The zoom at which the whole run is exactly as wide as the strip.
   *
   * It is the *floor*: below it the run stops filling the window and the strip
   * is empty space with a run drawn in the left of it, which is the one zoom
   * level that answers no question at all. Null until the strip has been
   * measured (the first frame, and any run with no length).
   */
  fitPxPerSecond: number | null;
  /**
   * The uids the search matches, or null when nothing is being searched for.
   *
   * The strip is the one view of a run that shows every line at once, so it is
   * where "which of these two hundred is my search talking about" is actually
   * answerable the marks outside the set drop back rather than disappearing,
   * because where the matches are *relative to the rest* is the answer.
   */
  matchedUids: Set<string> | null;
  /** A gap the reviewer is pointing at in the rail, drawn brighter here. */
  highlightGap: Span | null;
  /** The original audio's envelope, or null while there is none to draw. */
  sourcePeaks: Peaks | null;
  /** The finished mix's envelope null until the mix stage has run. */
  dubPeaks: Peaks | null;
  /**
   * The segments have moved on since that envelope was drawn.
   *
   * Only the waveform is affected. The marks over it come from the segments and
   * are as current as the script, which is why the lane dims one and not both —
   * dimming the whole lane would hide the very edits that made it stale.
   */
  stale?: boolean;
  /** The playhead, when it is inside the selected segment else null. */
  splitAt: number | null;
  onSelect: (uid: string) => void;
  onSeek: (time: number) => void;
  /**
   * A source mark was dragged somewhere new same contract as the panel's
   * bounds fields: both ends, already clamped against the neighbours here,
   * because the server refuses an overlap rather than fixing one.
   */
  onRetime: (uid: string, start: number, end: number) => void;
  /** How wide the scrolling area is, so the page can work out the fit zoom. */
  onViewport: (width: number) => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  /** Continuous zoom by a multiplicative factor — the pinch / ⌘-scroll path. */
  onZoomBy: (factor: number) => void;
  onFit: () => void;
  onSplit: (at: number) => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const laneRef = useRef<HTMLDivElement>(null);

  // Trailing room equal to the floating Split/zoom cluster, so the end of the
  // run exactly where drift accumulates can always be scrolled out from
  // under it. Without this, at a zoom where the run fits the strip there is no
  // scroll range at all and the last marks are permanently covered.
  const width = Math.max(320, total * pxPerSecond) + 208;
  const gaps = useMemo(() => unclaimedSpans(segments, total), [segments, total]);
  // A hair of tolerance: `fit` is a float and the state it was written into is
  // the same float, but a resize can move it by a fraction of a pixel and a −
  // that greys itself out one pixel early reads as broken.
  const atFloor = fitPxPerSecond != null && pxPerSecond <= fitPxPerSecond + 1e-6;
  const tickStep = useMemo(
    () => TICK_STEPS.find((step) => step * pxPerSecond >= 64) ?? TICK_STEPS[TICK_STEPS.length - 1],
    [pxPerSecond],
  );

  /*
   * The strip's own width, reported upward.
   *
   * "Fit" and the zoom-out floor are both `containerWidth / total`, and the
   * container is this component's but the zoom itself belongs to the page,
   * which owns the keyboard's `+`/`−` as well as these buttons. So the
   * measurement travels up rather than the zoom coming down. A ResizeObserver
   * and not one read on mount: the window is resizable and the fit zoom is a
   * fact about the window.
   */
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    onViewport(el.clientWidth);
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() => onViewport(el.clientWidth));
    observer.observe(el);
    return () => observer.disconnect();
  }, [onViewport]);

  /*
   * Pinch and ⌘/Ctrl-scroll zoom, anchored under the cursor.
   *
   * A trackpad pinch arrives as a wheel event with `ctrlKey` set — that is the
   * browser convention, not a modifier the user held — so one handler covers
   * both gestures. The zoom itself lives in the page (it owns the +/− ladder
   * too); what this component owns is the anchor: the moment the gesture
   * fires, remember which second of the run is under the pointer and where on
   * screen it sits, and when the new scale comes back down as a prop, put that
   * second back under the pointer. Without the anchor every zoom recentres on
   * 0:00 and zooming into minute 14 means finding minute 14 again.
   *
   * A native non-passive listener, not React's `onWheel`: the default has to
   * be prevented (a ctrl-scroll zooms the whole page otherwise) and passive
   * listeners cannot do that.
   */
  const pxRef = useRef(pxPerSecond);
  pxRef.current = pxPerSecond;
  const zoomAnchor = useRef<{ time: number; x: number } | null>(null);
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const onWheel = (event: WheelEvent) => {
      if (!event.ctrlKey && !event.metaKey) return;
      event.preventDefault();
      const x = event.clientX - el.getBoundingClientRect().left;
      zoomAnchor.current = { time: (el.scrollLeft + x) / pxRef.current, x };
      // Exponential: equal wheel travel is an equal zoom *ratio* at any scale.
      onZoomBy(Math.exp(-event.deltaY * 0.0015));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [onZoomBy]);
  useLayoutEffect(() => {
    const el = scrollRef.current;
    const anchor = zoomAnchor.current;
    if (!el || !anchor) return;
    zoomAnchor.current = null;
    el.scrollLeft = Math.max(0, anchor.time * pxPerSecond - anchor.x);
  }, [pxPerSecond]);

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
      // Clamp to the run: the lane carries trailing blank room (see `width`),
      // and a click there means "the end", not a time past the media.
      onSeek(Math.min(total, Math.max(0, (clientX - rect.left) / pxPerSecond)));
    },
    [onSeek, pxPerSecond, total],
  );

  /**
   * Drag-to-scrub. Pointer capture rather than window listeners so a drag that
   * leaves the strip which is most of them, the strip is 128px tall keeps
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

  /*
   * Retiming a source mark. The preview span lives in state so the mark under
   * the pointer redraws; everything else about the drag lives in a ref because
   * nothing needs to render it. Window listeners rather than pointer capture,
   * unlike the scrubber above: the grab starts on a <button> whose click must
   * still fire when the press turns out to be a selection, and capturing on it
   * re-targets the lane's own events for no gain the listeners are attached
   * for exactly one drag and removed on release.
   */
  const [preview, setPreview] = useState<{ uid: string } & Span | null>(null);
  const dragRef = useRef<DragSession | null>(null);

  const beginDrag = (seg: Segment, mode: DragMode, event: React.PointerEvent) => {
    if (event.button !== 0 || busyUids.includes(seg.uid)) return;
    // The free window: the neighbours by *time*, not by list order the drag
    // must never draw an overlap, because the server refuses one outright.
    let min = 0;
    let max = total;
    for (const other of segments) {
      if (other.uid === seg.uid) continue;
      if (other.end <= seg.start) min = Math.max(min, other.end);
      if (other.start >= seg.end) max = Math.min(max, other.start);
    }
    const session: DragSession = {
      uid: seg.uid,
      mode,
      originStart: seg.start,
      originEnd: seg.end,
      grabX: event.clientX,
      min,
      max,
      moved: false,
    };
    dragRef.current = session;
    onSelect(seg.uid);

    /** Where the span is with the pointer at `clientX` snapped, then clamped. */
    const spanAt = (clientX: number): Span => {
      const dt = (clientX - session.grabX) / pxPerSecond;
      // Snap when within a grab's width of a neighbour's edge or the playhead —
      // the three positions a retime is usually aiming for. Otherwise round to
      // 10ms: nobody drags to 12.3456s on purpose. Clamping comes *after*, so a
      // snap or a rounding can never push an edge across a neighbour.
      const snap = (t: number, edges: number[]): number => {
        const tolerance = 8 / pxPerSecond;
        for (const edge of edges) if (Math.abs(t - edge) <= tolerance) return edge;
        return Math.round(t * 100) / 100;
      };
      const clamp = (t: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, t));
      const { originStart, originEnd, min, max } = session;
      if (session.mode === "move") {
        const span = originEnd - originStart;
        const start = clamp(
          snap(originStart + dt, [min, max - span, currentTime]),
          min,
          max - span,
        );
        return { start, end: start + span };
      }
      if (session.mode === "start") {
        const start = clamp(
          snap(originStart + dt, [min, currentTime]),
          min,
          originEnd - MIN_SPAN,
        );
        return { start, end: originEnd };
      }
      const end = clamp(snap(originEnd + dt, [max, currentTime]), originStart + MIN_SPAN, max);
      return { start: originStart, end };
    };

    const onMove = (ev: PointerEvent) => {
      if (Math.abs(ev.clientX - session.grabX) > DRAG_SLOP_PX) session.moved = true;
      if (session.moved) setPreview({ uid: session.uid, ...spanAt(ev.clientX) });
    };
    const onUp = (ev: PointerEvent) => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      dragRef.current = null;
      setPreview(null);
      // A press that never travelled is the mark's click selection, already
      // done above and saving it would stamp a bounds edit on nothing.
      if (!session.moved) return;
      const span = spanAt(ev.clientX);
      if (span.start !== session.originStart || span.end !== session.originEnd) {
        onRetime(session.uid, span.start, span.end);
      }
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  };

  return (
    <section className="relative flex h-32 shrink-0 border-t border-border bg-surface pb-1.5">
      {/*
        The track headers. Outside the scroll container on purpose: a label
        inside it either scrolls away from the lane it names or is pinned over
        the marks it is supposed to be labelling. Height-matched to the lanes
        by mirroring their flex structure one spacer for the ruler, then one
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
        {/*
          The lane's waveform is the *rendered* audio, so when the render is
          behind the script the picture is of an older cut. The label says which
          the marks on top of it stay live, because they are drawn from the
          segments and are current.
        */}
        <span
          data-lane-label
          title={stale ? "Output · last render" : undefined}
          className="flex min-h-0 flex-1 flex-col justify-center px-2"
        >
          <span>Output</span>
          {/* Stacked, not inline: the gutter is 64px and an inline qualifier
              wraps into the lane it is labelling. Two deliberate lines fit. */}
          {stale ? (
            <span className="mt-0.5 text-[8px] font-semibold leading-none text-muted">
              last render
            </span>
          ) : null}
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
            {gaps.map((gap) => {
              /*
               * The hatch, named.
               *
               * It was the one mark on the strip with no explanation anywhere:
               * a 135° hatch means "unclaimed" to whoever wrote it and nothing
               * at all to the reviewer looking at it, who has no legend, no
               * tooltip and no row in the script to click. It says what it is
               * and which seconds it covers, in the same sentence a mark's
               * `aria-label` uses, and the shortcuts popover carries the
               * one-line version of the same fact.
               */
              const lit =
                highlightGap != null &&
                gap.start < highlightGap.end &&
                gap.end > highlightGap.start;
              const label = `Unclaimed no segment covers ${timecode(gap.start, 0)}–${timecode(
                gap.end,
                0,
              )}`;
              return (
                <div
                  key={`gap-${gap.start}`}
                  role="img"
                  data-hatch={lit ? "lit" : ""}
                  aria-label={label}
                  title={label}
                  className={cn(
                    "hatch-unclaimed absolute inset-y-2 rounded-[3px] border border-dashed",
                    "transition-all",
                    // Pointing at a gap in the rail's list lights the one it is
                    // about: the list is timecodes, and a timecode is only an
                    // answer once you can see where in the run it lands.
                    lit && "inset-y-1 ring-2 ring-accent",
                  )}
                  style={{
                    left: gap.start * pxPerSecond + 1,
                    width: Math.max(MIN_MARK_PX, (gap.end - gap.start) * pxPerSecond - 2),
                    borderColor: `color-mix(in srgb, var(--color-unclaimed) ${
                      lit ? "100%" : "45%"
                    }, transparent)`,
                  }}
                />
              );
            })}
            {segments.map((seg) => (
              <Mark
                key={seg.uid}
                seg={seg}
                start={preview?.uid === seg.uid ? preview.start : seg.start}
                end={preview?.uid === seg.uid ? preview.end : seg.end}
                pxPerSecond={pxPerSecond}
                selected={seg.uid === selectedUid}
                busy={busyUids.includes(seg.uid)}
                dim={matchedUids != null && !matchedUids.has(seg.uid)}
                dragging={preview?.uid === seg.uid}
                onSelect={onSelect}
                onDrag={(mode, event) => beginDrag(seg, mode, event)}
              />
            ))}
          </Lane>

          <Lane last>
            <Waveform peaks={dubPeaks} lane="dub" pxPerSecond={pxPerSecond} faded={stale} />
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
                  dim={matchedUids != null && !matchedUids.has(seg.uid)}
                  onSelect={onSelect}
                />
              );
            })}
          </Lane>

          {/* Playhead sits above both lanes so the drift between them is
              legible, and it is the most salient thing on the strip: full ink
              against washed marks, with a faint halo so it survives crossing a
              solid run of selected fills. The head is a real grab handle now —
              "where is the video" was being answered by a hairline, and a
              hairline is what people told us they could not find. It takes the
              pointer (the line stays transparent to it, so marks under the
              wire stay clickable) and a press on it falls through to the
              lane's scrubber, which is exactly what grabbing a playhead means. */}
          <div
            data-playhead
            className="pointer-events-none absolute top-0 bottom-0 z-20 w-[2px] -translate-x-[1px] bg-primary shadow-[0_0_0_1px_var(--color-surface),0_0_8px_var(--color-primary)]"
            style={{ left: currentTime * pxPerSecond }}
          >
            <div
              title={`Playhead ${timecode(currentTime)} drag to scrub`}
              className="pointer-events-auto absolute -left-[5px] top-0 h-3.5 w-3 cursor-ew-resize rounded-b-[4px] bg-primary shadow-[0_1px_3px_rgba(0,0,0,0.35)]"
            />
          </div>
        </div>
      </div>

      {/*
        The strip's own controls, floated over its right edge rather than given
        a toolbar row of their own 128px is the whole budget and a header
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
                : `Split at ${timecode(splitAt)}. Both halves lose their translation and their clip the line was written for the whole span.`
            }
            onConfirm={() => splitAt != null && onSplit(splitAt)}
          >
            <Scissors className="h-3 w-3" />
            Split
          </ConfirmButton>
          {/*
            Zoom out had no floor. Every press divided the scale again, so a
            twelve-minute run went to a two-inch smear against the left edge of
            a strip that was otherwise empty and nothing in the cluster could
            get back to a useful scale except pressing + the same number of
            times and counting. The floor is the zoom at which the run is
            exactly as wide as the strip, and Fit is the one press that goes
            there: between − and the readout, because that is the direction it
            travels.
          */}
          <ButtonGroup className="h-6">
            <Button
              variant="ghost"
              size="xs"
              onClick={onZoomOut}
              disabled={atFloor}
              title={atFloor ? "The whole run already fits the strip" : "Zoom out"}
              aria-label="Zoom out"
            >
              <Minus className="h-3 w-3" />
            </Button>
            <Button
              variant="ghost"
              size="xs"
              data-zoom-fit
              onClick={onFit}
              disabled={fitPxPerSecond == null}
              title="Zoom so the whole run fits the strip"
              className="px-2 text-[10px] font-bold uppercase tracking-[0.12em]"
            >
              Fit
            </Button>
            <span className="flex w-12 items-center justify-center bg-raised font-mono text-[11px] tabular-nums text-muted">
              {zoomLabel(pxPerSecond)}px/s
            </span>
            <Button
              variant="ghost"
              size="xs"
              onClick={onZoomIn}
              disabled={pxPerSecond >= 256 - 1e-6}
              title="Zoom in — or pinch / ⌘-scroll the strip for any scale"
              aria-label="Zoom in"
            >
              <Plus className="h-3 w-3" />
            </Button>
          </ButtonGroup>
        </div>
      </div>
    </section>
  );
}

/**
 * One lane's plotting area. It carries no label the gutter does that so
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
 * at 32px/s over a twenty-minute run is 38,000px wide past the maximum canvas
 * dimension on Safari and a lot of memory everywhere else while a path with a
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
  faded = false,
}: {
  peaks: Peaks | null;
  lane: "source" | "dub";
  pxPerSecond: number;
  /** Drawn from audio the script has moved past see `Timeline`'s `stale`. */
  faded?: boolean;
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
      data-faded={faded ? "" : undefined}
      viewBox={`0 0 ${peaks.peaks.length} 100`}
      preserveAspectRatio="none"
      /*
       * The height is stated, not inherited. An absolutely-positioned SVG with
       * a width, a `top`, a `bottom` and `height: auto` is over-constrained, so
       * the browser drops `bottom` and takes the height from the viewBox ratio
       * which at 4px/s happened to land within a pixel of the lane and at
       * 32px/s was 338px tall, drawing the top eighth of the waveform in this
       * lane and the rest over the one below.
       */
      /* Inset to exactly the marks' box, so the audio reads as being *in* the
         clip rather than passing behind it. */
      className="pointer-events-none absolute left-0 top-2 h-[calc(100%-1rem)] text-muted"
      style={{ width: Math.max(1, peaks.duration * pxPerSecond) }}
    >
      <path d={path} fill="currentColor" fillOpacity={faded ? 0.18 : 0.42} />
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
  dim,
  dragging,
  onSelect,
  onDrag,
}: {
  seg: Segment;
  start: number;
  end: number;
  pxPerSecond: number;
  selected: boolean;
  busy: boolean;
  muted?: boolean;
  /** A search is running and this line is not in it. */
  dim?: boolean;
  /** This mark is mid-retime and `start`/`end` are the preview span. */
  dragging?: boolean;
  onSelect: (uid: string) => void;
  /**
   * Source lane only the OUTPUT lane's marks stay a picture, so they never
   * pass this and never grow handles.
   */
  onDrag?: (mode: DragMode, event: React.PointerEvent) => void;
}) {
  const state = segmentState(seg);
  const meta = STATE_META[state];
  // 1px inset each side = the 2px surface gap that keeps adjacent fills from
  // reading as one block.
  const width = Math.max(MIN_MARK_PX, (end - start) * pxPerSecond - 2);
  const showIcon = width > 18;
  // No handles on a sliver: at 14px the two 5px grips leave 4px of body, and a
  // grab meant as a move becomes a trim. The whole mark still moves.
  const showHandles = onDrag != null && !busy && width > 14;

  return (
    <button
      type="button"
      data-mark
      data-dragging={dragging ? "" : undefined}
      onClick={() => onSelect(seg.uid)}
      onPointerDown={onDrag ? (event) => onDrag("move", event) : undefined}
      aria-label={`Segment ${seg.id}, ${meta.label}, ${timecode(start)} to ${timecode(end)}`}
      aria-pressed={selected}
      className={cn(
        "group/mark absolute inset-y-2 overflow-hidden rounded-[3px] border text-left transition-all",
        // Hover is a real affordance here: the marks are 3–40px wide and the
        // only other cue that they are clickable is the cursor.
        // Hover and selection ring in the accent, not in ink: the playhead is
        // the ink on this strip and it has to stay the brightest thing on it.
        "hover:inset-y-1.5 hover:ring-2 hover:ring-accent/45",
        selected && "z-10 ring-2 ring-accent ring-offset-1 ring-offset-surface",
        busy && "animate-pulse",
        // Draggable marks say so with the cursor; while a job holds the line
        // the grab is refused (`beginDrag`), so the hand would be a lie.
        onDrag && !busy && (dragging ? "cursor-grabbing" : "cursor-grab"),
        dragging && "z-10 ring-2 ring-accent",
        // A mark in the OUTPUT lane with nothing placed under it is a
        // *prediction*, not a placement, so it is drawn as one.
        muted && "border-dashed opacity-55",
        // Last, so it wins over `muted`: while a search is running, "is this
        // one of my results" outranks every other thing a mark is saying.
        dim && "opacity-35",
      )}
      data-dim={dim ? "" : undefined}
      style={{
        left: start * pxPerSecond + 1,
        width,
        borderColor: meta.token,
        // The hue lives on the edge; the fill is a wash of it. A hundred
        // adjacent marks then read as a hundred marks rather than as one
        // coloured band, and the playhead stays the brightest thing here.
        //
        // How strong the wash is belongs to the theme, not to this file: the
        // light hues are dark on a white lane, the dark ones are light on
        // near-black *and* sit over the lane's waveform. Both numbers are in
        // App.css next to the hues they are tuned against.
        backgroundColor: `color-mix(in srgb, ${meta.token} var(${
          selected ? "--mark-wash-selected" : "--mark-wash"
        }), transparent)`,
      }}
    >
      {showIcon ? (
        <StateIcon
          state={state}
          className="pointer-events-none absolute left-[3px] top-1/2 h-2 w-2 -translate-y-1/2 opacity-90"
        />
      ) : null}
      {/*
        The trim grips. Spans, not buttons a button inside a button is
        invalid HTML and the keyboard path to a precise trim is the panel's
        bounds fields anyway. `stopPropagation` keeps a grab on the edge from
        also being a grab on the body, which would turn every trim into a move.
      */}
      {showHandles ? (
        <>
          <span
            data-handle="start"
            aria-hidden
            onPointerDown={(event) => {
              event.stopPropagation();
              onDrag?.("start", event);
            }}
            className={cn(
              "absolute inset-y-0 left-0 w-[5px] cursor-ew-resize opacity-0 transition-opacity",
              "group-hover/mark:opacity-100",
              (selected || dragging) && "opacity-100",
            )}
            style={{ backgroundColor: `color-mix(in srgb, ${meta.token} 55%, transparent)` }}
          />
          <span
            data-handle="end"
            aria-hidden
            onPointerDown={(event) => {
              event.stopPropagation();
              onDrag?.("end", event);
            }}
            className={cn(
              "absolute inset-y-0 right-0 w-[5px] cursor-ew-resize opacity-0 transition-opacity",
              "group-hover/mark:opacity-100",
              (selected || dragging) && "opacity-100",
            )}
            style={{ backgroundColor: `color-mix(in srgb, ${meta.token} 55%, transparent)` }}
          />
        </>
      ) : null}
    </button>
  );
}

function ticks(total: number, step: number): number[] {
  const out: number[] = [];
  for (let t = 0; t <= total; t += step) out.push(t);
  return out;
}
