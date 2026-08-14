/**
 * The table view of the same data the timeline draws.
 *
 * It is not a convenience: two of the state hues sit below 3:1 on the light
 * surface, and the relief for that is a view where every state is a word. It
 * also carries what a 4px-wide mark cannot — the text, the speaker, and the
 * verification overlap.
 */

import { useEffect, useRef } from "react";
import { Lock, TriangleAlert } from "lucide-react";
import { cn } from "../lib/classNames";
import { percent, speakerLabel, timecode } from "../lib/format";
import { STATE_META, segmentState, verifyConcern } from "../lib/segments";
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
  const selectedRef = useRef<HTMLTableRowElement>(null);

  useEffect(() => {
    selectedRef.current?.scrollIntoView({ block: "nearest" });
  }, [selectedUid]);

  return (
    <div className="h-full overflow-auto">
      <table className="w-full border-collapse text-[12px]">
        <thead className="sticky top-0 z-10 bg-surface text-left text-[10px] uppercase tracking-[0.1em] text-muted">
          <tr className="border-b border-border">
            <th className="w-10 px-2 py-1.5 font-semibold">#</th>
            <th className="w-24 px-2 py-1.5 font-semibold">State</th>
            <th className="w-20 px-2 py-1.5 font-semibold">Start</th>
            <th className="w-12 px-2 py-1.5 font-semibold">Spk</th>
            <th className="px-2 py-1.5 font-semibold">Source</th>
            <th className="px-2 py-1.5 font-semibold">Target</th>
            <th className="w-16 px-2 py-1.5 text-right font-semibold">Verify</th>
          </tr>
        </thead>
        <tbody>
          {segments.map((seg) => {
            const state = segmentState(seg);
            const meta = STATE_META[state];
            const concern = verifyConcern(seg);
            const selected = seg.uid === selectedUid;
            const playing = currentTime >= seg.start && currentTime < seg.end;
            return (
              <tr
                key={seg.uid}
                ref={selected ? selectedRef : undefined}
                onClick={() => onSelect(seg.uid)}
                className={cn(
                  "cursor-pointer border-b border-grid align-top",
                  selected ? "bg-brand/10" : "hover:bg-border/40",
                  playing && !selected && "bg-border/25",
                  busyUids.includes(seg.uid) && "animate-pulse",
                )}
              >
                <td className="px-2 py-1.5 tabular-nums text-muted">{seg.id}</td>
                <td className="px-2 py-1.5">
                  <span className="inline-flex items-center gap-1.5 whitespace-nowrap text-secondary">
                    <span aria-hidden style={{ color: meta.token }}>
                      {meta.glyph}
                    </span>
                    {meta.label}
                  </span>
                </td>
                <td className="px-2 py-1.5 tabular-nums text-secondary">{timecode(seg.start)}</td>
                <td className="px-2 py-1.5 text-secondary">{speakerLabel(seg.speaker)}</td>
                <td className="auto-dir max-w-0 truncate px-2 py-1.5 text-secondary">
                  {seg.text}
                </td>
                <td className="max-w-0 truncate px-2 py-1.5">
                  <span className="inline-flex max-w-full items-center gap-1">
                    {seg.locked ? (
                      <Lock className="h-3 w-3 shrink-0 text-muted" aria-label="Hand-edited" />
                    ) : null}
                    <span className="truncate">{seg.text_en ?? "—"}</span>
                  </span>
                </td>
                <td className="px-2 py-1.5 text-right tabular-nums">
                  <span
                    className={cn(
                      "inline-flex items-center gap-1",
                      concern === "bad" ? "text-critical" : "text-muted",
                    )}
                  >
                    {concern !== "none" ? <TriangleAlert className="h-3 w-3" /> : null}
                    {percent(seg.verify?.overlap ?? seg.tts?.overlap)}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
