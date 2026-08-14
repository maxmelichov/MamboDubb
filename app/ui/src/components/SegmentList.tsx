/**
 * The segment navigator.
 *
 * This was a seven-column table — id, state, start, speaker, source, target,
 * verify — and it was the wrong shape for what it is used for. Nobody reads a
 * run in a spreadsheet; they find the line they want and open it. Every column
 * that is not "which line is this" was competing for width with the one that
 * is, and the two text columns each got a truncated half of the space, so
 * neither was readable.
 *
 * So it is a list now: position, state, time, one line of text. Everything
 * that was in the other columns is in the inspector, at full width, for the
 * one segment being worked on. What stays here is exactly what you scan.
 *
 * Two things are load-bearing and are not decoration:
 *
 * - **The state is a word, not just a hue.** Light-mode "kept" sits at 2.17:1
 *   against the card, under the 3:1 gate, and with the legend now behind a
 *   popover this list is where the encoding is spelled out. The short forms
 *   (Dub / Keep / Fail / Wait) exist for exactly this column.
 * - **The line of text follows what will play.** A kept segment shows its
 *   source, because the source is what the viewer hears; a dubbed one shows
 *   the translation, because that is the thing being reviewed. Showing the
 *   translation of a line that will never be spoken is a quiet lie.
 */

import { useEffect, useRef } from "react";
import { Lock, ListX, TriangleAlert } from "lucide-react";
import { cn } from "../lib/classNames";
import { speakerLabel, timecode } from "../lib/format";
import { STATE_META, segmentState, verifyConcern } from "../lib/segments";
import { Empty } from "./ui";
import type { Segment } from "../lib/types";

export function SegmentList({
  segments,
  selectedUid,
  currentTime,
  busyUids,
  onSelect,
}: {
  segments: Segment[];
  selectedUid: string | null;
  currentTime: number;
  busyUids: string[];
  onSelect: (uid: string) => void;
}) {
  const selectedRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    selectedRef.current?.scrollIntoView({ block: "nearest" });
  }, [selectedUid]);

  if (segments.length === 0) {
    return (
      <Empty icon={ListX} title="No segments yet">
        Segments appear once the <code className="font-mono">segments</code> stage has run —
        it is what turns the transcript's words into the lines you review here.
      </Empty>
    );
  }

  return (
    /*
     * `listbox`/`option` rather than `grid`/`row`: this is a single-select
     * picker now, not a table, and that is the role that makes `aria-selected`
     * mean "this is the one open in the inspector".
     */
    <div role="listbox" aria-label="Segments" className="h-full overflow-auto">
      {segments.map((seg) => {
        const state = segmentState(seg);
        const meta = STATE_META[state];
        const concern = verifyConcern(seg);
        const selected = seg.uid === selectedUid;
        const playing = currentTime >= seg.start && currentTime < seg.end;
        // See the header note: what plays is what is shown.
        const line = seg.keep ? seg.text : (seg.text_en ?? seg.text);
        const untranslated = !seg.keep && !seg.text_en;

        return (
          <div
            key={seg.uid}
            role="option"
            ref={selected ? selectedRef : undefined}
            onClick={() => onSelect(seg.uid)}
            /*
             * Roving tabindex: the list is one tab stop, not fifty-eight. Tab
             * lands on the current row, Enter commits it, and ←/→ (handled
             * globally by the editor) move the selection.
             */
            tabIndex={selected || (!selectedUid && seg.uid === segments[0].uid) ? 0 : -1}
            aria-selected={selected}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                onSelect(seg.uid);
              }
            }}
            title={`${speakerLabel(seg.speaker)} · ${seg.text}`}
            className={cn(
              "flex cursor-pointer items-center gap-2.5 border-b border-grid px-3 py-1.5 text-[12px]",
              "transition-colors",
              // Selection is ink, not the brand blue — blue already means
              // "dubbed" two cells to the left. The rule down the left edge is
              // the part that actually reads at a glance; the wash alone is
              // too faint on either theme.
              selected
                ? "bg-primary/[0.07] shadow-[inset_3px_0_0_var(--color-primary)]"
                : "hover:bg-sunken",
              // "Where the audio is" and "what I am editing" are different
              // questions, and they drift apart the moment you scrub.
              playing &&
                !selected &&
                "bg-sunken shadow-[inset_3px_0_0_color-mix(in_srgb,var(--color-primary)_28%,transparent)]",
              busyUids.includes(seg.uid) && "animate-pulse",
            )}
          >
            <span className="w-6 shrink-0 text-right font-mono tabular-nums text-muted">
              {seg.id}
            </span>

            <span className="inline-flex w-[3.6rem] shrink-0 items-center gap-1 text-[11px] text-secondary">
              <span aria-hidden style={{ color: meta.token }}>
                {meta.glyph}
              </span>
              {meta.short}
            </span>

            <span className="w-11 shrink-0 font-mono tabular-nums text-muted">
              {timecode(seg.start)}
            </span>

            <span
              dir="auto"
              className={cn(
                "auto-dir min-w-0 flex-1 truncate",
                untranslated ? "italic text-muted" : "text-secondary",
              )}
            >
              {untranslated ? seg.text : line}
            </span>

            {/* The two flags worth a mark in a scan: this line was written by
                hand, and this line's clone did not say it. */}
            {seg.locked ? (
              <Lock className="h-3 w-3 shrink-0 text-muted" aria-label="Hand-edited" />
            ) : null}
            {concern !== "none" ? (
              <TriangleAlert
                className={cn(
                  "h-3 w-3 shrink-0",
                  concern === "bad" ? "text-critical" : "text-muted",
                )}
                aria-label={concern === "bad" ? "Verification failed" : "Verification is low"}
              />
            ) : null}
          </div>
        );
      })}
    </div>
  );
}
