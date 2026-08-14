/**
 * The script.
 *
 * This is the editor's largest region — 58% of the width and all of the height
 * between the header and the timeline strip — which is the whole argument of
 * the redesign in one number. The video used to be the pane that grew and the
 * text was a fifteen-rem drawer at the bottom; but nobody reviews a dub by
 * watching it, they review it by *reading* it, with the video as the reference
 * they glance at. So the text grows and the picture is fixed.
 *
 * Above the rows sit the two things that make two hundred rows navigable: a
 * search box and four filter chips. They are not decoration — "show me the
 * eleven that failed" is the single most common thing a reviewer wants from a
 * run, and it used to require scrolling the whole list looking for a hue.
 *
 * And once a filter has selected a set, fixing that set is one job rather than
 * eleven: `POST /resynthesize` has always taken `{uids:[…]}` and the UI has
 * always sent exactly one. Eleven separate jobs is eleven model loads, in a
 * process where only one model may be resident at a time — so it is not a
 * convenience, it is the difference between one minute and twenty.
 */

import { useEffect, useMemo, useRef } from "react";
import { Languages, ListX, Search, Volume2 } from "lucide-react";
import { cn } from "../lib/classNames";
import { hasLocks, segmentState } from "../lib/segments";
import { Button, Empty } from "./ui";
import { ScriptRow, type EditTarget } from "./ScriptRow";
import type { Segment } from "../lib/types";

export type ScriptFilter = "all" | "failed" | "kept" | "edited";

/** Past this many rows the list opts into browser-side render skipping. */
const VIRTUALIZE_ABOVE = 300;

/** Keep this in step with `scroll-pt-3` / `scroll-pb-3` on the list. */
const SCROLL_INSET = 12;

/** A uid is `s_<hex>` today, but a selector built from data must still be safe. */
const escapeId = (value: string): string =>
  typeof CSS !== "undefined" && CSS.escape ? CSS.escape(value) : value.replace(/["\\]/g, "\\$&");

/**
 * Which lines the search box and the chips leave on screen.
 *
 * Exported and pure because the editor's ↑/↓ have to step through *what is
 * visible*, not through the whole run — an arrow key that jumps to a row the
 * filter is hiding is a selection the user cannot see.
 */
export function filterSegments(
  segments: Segment[],
  query: string,
  filter: ScriptFilter,
): Segment[] {
  const needle = query.trim().toLowerCase();
  return segments.filter((seg) => {
    if (filter === "failed" && segmentState(seg) !== "failed") return false;
    if (filter === "kept" && !seg.keep) return false;
    if (filter === "edited" && !hasLocks(seg)) return false;
    if (!needle) return true;
    return (
      seg.text.toLowerCase().includes(needle) ||
      (seg.text_en ?? "").toLowerCase().includes(needle) ||
      seg.speaker.toLowerCase().includes(needle) ||
      String(seg.id) === needle
    );
  });
}

export function ScriptPane({
  segments,
  selectedUid,
  currentTime,
  busyUids,
  editing,
  playingUrl,
  query,
  filter,
  searchRef,
  onQuery,
  onFilter,
  onSelect,
  onEdit,
  onCommit,
  onPlay,
  onToggleKeep,
  onRetranslateMany,
  onResynthesizeMany,
}: {
  segments: Segment[];
  selectedUid: string | null;
  currentTime: number;
  busyUids: string[];
  editing: EditTarget;
  playingUrl: string | null;
  query: string;
  filter: ScriptFilter;
  searchRef: React.RefObject<HTMLInputElement | null>;
  onQuery: (query: string) => void;
  onFilter: (filter: ScriptFilter) => void;
  onSelect: (uid: string) => void;
  onEdit: (target: EditTarget) => void;
  onCommit: (uid: string, field: "text" | "text_en", value: string) => void;
  onPlay: (url: string | null) => void;
  onToggleKeep: (seg: Segment) => void;
  onRetranslateMany: (uids: string[]) => void;
  onResynthesizeMany: (uids: string[]) => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);

  const counts = useMemo(
    () => ({
      all: segments.length,
      failed: segments.filter((seg) => segmentState(seg) === "failed").length,
      kept: segments.filter((seg) => seg.keep).length,
      edited: segments.filter(hasLocks).length,
    }),
    [segments],
  );

  const visible = useMemo(
    () => filterSegments(segments, query, filter),
    [filter, query, segments],
  );

  const nowUid = useMemo(
    () => segments.find((seg) => currentTime >= seg.start && currentTime < seg.end)?.uid ?? null,
    [currentTime, segments],
  );

  /**
   * Follow the playhead, Final Cut's timeline-index behaviour: the row under
   * the playhead centres itself as playback moves through it.
   *
   * The guard is the same one the timeline strip uses — only scroll when the
   * row has actually left the visible band. Anything more eager fights a user
   * who has scrolled somewhere else to look at something, which is the one
   * thing an auto-scrolling list must never do.
   */
  useEffect(() => {
    const box = scrollRef.current;
    if (!box || !nowUid) return;
    const row = box.querySelector<HTMLElement>(`[data-uid="${escapeId(nowUid)}"]`);
    if (!row) return;
    const top = row.offsetTop;
    const bottom = top + row.offsetHeight;
    if (
      top < box.scrollTop + SCROLL_INSET ||
      bottom > box.scrollTop + box.clientHeight - SCROLL_INSET
    ) {
      box.scrollTo({ top: Math.max(0, top - box.clientHeight / 2), behavior: "smooth" });
    }
  }, [nowUid]);

  // Selection is a deliberate act, so it is allowed to be less polite than the
  // playhead: `nearest` keeps the row on screen without recentring the list.
  useEffect(() => {
    if (!selectedUid) return;
    const box = scrollRef.current;
    box
      ?.querySelector<HTMLElement>(`[data-uid="${escapeId(selectedUid)}"]`)
      ?.scrollIntoView({ block: "nearest" });
  }, [selectedUid]);

  const tabStopUid = selectedUid ?? visible[0]?.uid ?? null;

  const chip = (key: ScriptFilter, label: string, count: number) => (
    <button
      key={key}
      type="button"
      aria-pressed={filter === key}
      onClick={() => onFilter(filter === key ? "all" : key)}
      className={cn(
        "inline-flex h-6 shrink-0 items-center gap-1 rounded-md border px-2 text-[11px] font-semibold transition-colors",
        filter === key
          ? "border-transparent bg-primary text-on-primary"
          : "border-border bg-raised text-secondary hover:border-axis hover:text-primary",
        count === 0 && filter !== key && "opacity-45",
      )}
    >
      {label}
      <span className="font-mono tabular-nums">{count}</span>
    </button>
  );

  // The bulk offer is tied to the Failed chip because that is the set where
  // "do all of these" is unambiguous: every one of them is a line the pipeline
  // did not manage to say.
  const bulkUids = visible.filter((seg) => !seg.keep).map((seg) => seg.uid);
  const showBulk = filter === "failed" && bulkUids.length > 0;

  return (
    <section
      className={cn(
        "flex min-h-0 min-w-0 flex-1 flex-col bg-surface",
        // The divider follows the axis the two regions are stacked on: a
        // bottom rule while the viewer sits underneath, an inline-end rule once
        // it moves beside.
        "border-b border-border xl:border-b-0 xl:border-e",
      )}
    >
      {/*
        The filter bar is not inside the scroll container and never scrolls, so
        it needs an opaque ground and a hairline under it — with a wash or a
        translucent tone the row passing beneath shows through and the two sets
        of letters overprint. `bg-sunken` is a solid token in both themes.
      */}
      <div
        data-script-header
        className="flex shrink-0 flex-wrap items-center gap-2 border-b border-border bg-sunken px-3 py-1.5"
      >
        <label className="relative flex h-7 min-w-[11rem] flex-1 items-center">
          <Search className="pointer-events-none absolute left-2 h-3.5 w-3.5 text-muted" aria-hidden />
          <input
            ref={searchRef}
            type="search"
            value={query}
            onChange={(event) => onQuery(event.currentTarget.value)}
            placeholder="Search the script"
            aria-label="Search the script"
            className="h-7 w-full rounded-md border border-border bg-raised pl-7 pr-2 text-[12.5px] text-primary outline-none transition-colors placeholder:text-muted/70 hover:border-axis focus:border-primary"
          />
        </label>
        {chip("all", "All", counts.all)}
        {chip("failed", "Failed", counts.failed)}
        {chip("kept", "Kept", counts.kept)}
        {chip("edited", "Edited", counts.edited)}
      </div>

      {showBulk ? (
        <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-border bg-critical/[0.06] px-3 py-1.5">
          <span className="min-w-0 flex-1 text-[11px] text-secondary">
            {bulkUids.length} line{bulkUids.length === 1 ? "" : "s"} the pipeline could not say.
            One job, not {bulkUids.length}.
          </span>
          <Button size="xs" onClick={() => onResynthesizeMany(bulkUids)}>
            <Volume2 className="h-3 w-3" />
            Re-voice these {bulkUids.length}
          </Button>
          <Button size="xs" onClick={() => onRetranslateMany(bulkUids)}>
            <Languages className="h-3 w-3" />
            Re-translate these {bulkUids.length}
          </Button>
        </div>
      ) : null}

      <div
        ref={scrollRef}
        role="listbox"
        aria-label="Script"
        data-script-scroll
        className={cn(
          "min-h-0 flex-1 overflow-y-auto",
          // Every scroll this list performs is a scroll *to a row* — the
          // playhead's, the selection's, ↑/↓'s — and a row landed flush under
          // the filter bar reads as text sliced off by it. `scroll-padding`
          // is the platform's answer: it is the inset every one of those
          // scrolls, including `scrollIntoView`, aligns to.
          "scroll-pt-3 scroll-pb-3",
          // Past a few hundred rows the browser skips the layout and paint of
          // everything off screen. No windowing library, no measured
          // virtualiser, no new dependency — the platform has done this since
          // 2021 and `contain-intrinsic-size` keeps the scrollbar honest.
          segments.length > VIRTUALIZE_ABOVE && "script-virtual",
        )}
      >
        {visible.length === 0 ? (
          <Empty icon={ListX} title={segments.length === 0 ? "No segments yet" : "Nothing matches"}>
            {segments.length === 0 ? (
              <>
                Segments appear once the <code className="font-mono">segments</code> stage has
                run — it is what turns the transcript's words into the lines you review here.
              </>
            ) : (
              "No line in this run matches the search and the filter together."
            )}
          </Empty>
        ) : (
          visible.map((seg) => (
            <ScriptRow
              key={seg.uid}
              seg={seg}
              selected={seg.uid === selectedUid}
              now={seg.uid === nowUid}
              busy={busyUids.includes(seg.uid)}
              tabStop={seg.uid === tabStopUid}
              editing={editing?.uid === seg.uid ? editing.field : null}
              playingUrl={playingUrl}
              onSelect={onSelect}
              onEdit={onEdit}
              onCommit={onCommit}
              onPlay={onPlay}
              onToggleKeep={onToggleKeep}
            />
          ))
        )}
      </div>
    </section>
  );
}

