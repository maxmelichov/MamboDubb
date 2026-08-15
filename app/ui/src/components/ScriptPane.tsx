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
 * search box and five filter chips. They are not decoration — "show me the
 * eleven that failed" is the single most common thing a reviewer wants from a
 * run, and it used to require scrolling the whole list looking for a hue.
 * "Unfinished" is the second most common and used to be unaskable: a line with
 * no translation or no clip is invisible in a list of two hundred, and it is
 * exactly what a verdict flip leaves behind.
 *
 * And once a filter has selected a set, fixing that set is one job rather than
 * eleven: `POST /resynthesize` has always taken `{uids:[…]}` and the UI has
 * always sent exactly one. Eleven separate jobs is eleven model loads, in a
 * process where only one model may be resident at a time — so it is not a
 * convenience, it is the difference between one minute and twenty.
 *
 * Kept is the third chip with a bulk offer and the only one whose button
 * changes a *verdict* rather than re-running a stage, which is why it is the
 * only one that asks first.
 */

import { useEffect, useMemo, useRef } from "react";
import { Film, Languages, ListX, Mic2, Search, Volume2 } from "lucide-react";
import { cn } from "../lib/classNames";
import { hasLocks, needsModelWork, segmentState, unfinished } from "../lib/segments";
import { Button, ConfirmButton, Empty } from "./ui";
import { ScriptRow, type EditTarget } from "./ScriptRow";
import type { Segment } from "../lib/types";

export type ScriptFilter = "all" | "failed" | "unfinished" | "kept" | "edited";

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
    if (filter === "unfinished" && !unfinished(seg)) return false;
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
  reveal,
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
  onFixMany,
  onDubMany,
  onRender,
}: {
  segments: Segment[];
  selectedUid: string | null;
  currentTime: number;
  busyUids: string[];
  editing: EditTarget;
  playingUrl: string | null;
  query: string;
  filter: ScriptFilter;
  /** "Put this row in the middle", bumped each time it is asked for. */
  reveal: { uid: string; n: number } | null;
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
  /** Translate whatever has no line, then voice all of them — in that order. */
  onFixMany: (uids: string[]) => void;
  /** Flip every one of these from keep to dub, then queue the work once. */
  onDubMany: (uids: string[]) => void;
  /** timeline → mix → report, for the lines that only lack a placement. */
  onRender: () => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);

  const counts = useMemo(
    () => ({
      all: segments.length,
      failed: segments.filter((seg) => segmentState(seg) === "failed").length,
      unfinished: segments.filter(unfinished).length,
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

  /**
   * A row asked for by name — today, by a click on its mark in the timeline.
   *
   * Centred rather than `nearest`, and unconditional rather than only-when-off
   * screen: this is not the list keeping up with something, it is the list
   * being *sent* somewhere, and the answer to "which line is that mark" should
   * land in the middle of the reading area, not scraped against the bottom
   * edge. Declared after the two effects above so it wins when they disagree.
   */
  useEffect(() => {
    if (!reveal) return;
    scrollRef.current
      ?.querySelector<HTMLElement>(`[data-uid="${escapeId(reveal.uid)}"]`)
      ?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [reveal]);

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
          ? "border-transparent bg-accent text-on-accent"
          : "border-border bg-raised text-secondary hover:border-axis hover:text-primary",
        count === 0 && filter !== key && "opacity-45",
        /*
          One chip is allowed a hue, and it is the one whose count is news.
          "Failed 3" in the same grey as "All 73" is a number you have to read
          to notice; in the failed hue it is the thing your eye lands on when
          the bar comes up. The rest stay quiet — a filter bar where every chip
          is coloured is a filter bar with no emphasis in it — and the hue is
          the state's own token, so this chip and the rows it selects are the
          same red. It drops the hue when it is the active filter, because then
          the accent pill *is* the emphasis, and two loud colours on one chip is
          one too many.
        */
        key === "failed" &&
          count > 0 &&
          filter !== key &&
          "border-critical/45 bg-critical/[0.07] text-critical hover:border-critical hover:text-critical",
      )}
    >
      {label}
      <span className="font-mono tabular-nums">{count}</span>
    </button>
  );

  /*
   * The bulk offer is tied to the two chips where "do all of these" is
   * unambiguous: Failed, every one of which is a line the pipeline could not
   * say, and Unfinished, every one of which is a line with nothing to play.
   *
   * Unfinished is the one that gets a run's stragglers back — including the
   * lines a "Dub it" used to strand, which had no translation, no clip and no
   * job on the way. Its button is one click and at most two jobs: translate
   * whatever has no line, then voice the lot.
   *
   * The work set is not "everything on screen that is not a keep", though, and
   * it was wrong in both directions. A line the *pipeline* gave up on is stored
   * as a keep (`keep_reason` tts_failed / mt_failed) with the original audio
   * attached — so excluding keeps excluded every member of the Failed filter,
   * and its two buttons enqueued jobs for nobody. And a line that has a clip
   * but no placement needs a render, not a model: re-voicing it is a minute of
   * GPU to arrive back exactly where it started.
   */
  const bulkSegs = visible.filter(
    (seg) => needsModelWork(seg) && (!seg.keep || segmentState(seg) === "failed"),
  );
  const bulkUids = bulkSegs.map((seg) => seg.uid);
  const renderUids = visible.filter((seg) => segmentState(seg) === "unplaced").map((s) => s.uid);
  const needText = bulkSegs.filter((seg) => !(seg.text_en ?? "").trim()).length;
  /*
   * Within a failed set the two buttons are not two ways to do one thing: a
   * `tts_failed` line has a translation the voice could not say, and an
   * `mt_failed` line's `text_en` is the *source* line the translator copied in
   * when it gave up. Re-voicing that one would synthesize the wrong language,
   * so each button asks for exactly the lines it fixes.
   */
  const voiceUids = bulkSegs.filter((s) => s.keep_reason !== "mt_failed").map((s) => s.uid);
  const textUids = bulkSegs
    .filter((s) => s.keep_reason === "mt_failed" || !(s.text_en ?? "").trim())
    .map((s) => s.uid);
  const showBulk =
    (filter === "failed" || filter === "unfinished") && bulkUids.length + renderUids.length > 0;
  const lines = `${bulkUids.length} line${bulkUids.length === 1 ? "" : "s"}`;

  /*
   * The third bulk offer, and the only one that changes a verdict rather than
   * re-running a stage.
   *
   * A kept line plays as recorded. When the keep rule was wrong about a whole
   * video — a third language read as "already the target", which is what a
   * Hebrew→German run full of English used to be — every one of those lines is
   * a keep the reviewer disagrees with, and the only way out was the per-line
   * flip, N times. This is that flip over the set on screen.
   *
   * It is the set *on screen* and not "every kept line in the run": the search
   * box is a filter like any other, and a button that quietly acted on two
   * hundred lines when eleven are showing would be the worst kind of bulk
   * action. It says which it is, in the sentence beside it.
   */
  const keptUids = filter === "kept" ? visible.map((seg) => seg.uid) : [];
  const searching = query.trim().length > 0;
  const showDubBulk = filter === "kept" && keptUids.length > 0;
  const keptLines = `${keptUids.length} line${keptUids.length === 1 ? "" : "s"}`;

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
            className="h-7 w-full rounded-md border border-border bg-raised pl-7 pr-2 text-[12.5px] text-primary outline-none transition-colors placeholder:text-muted/70 hover:border-axis focus:border-accent"
          />
        </label>
        {chip("all", "All", counts.all)}
        {chip("failed", "Failed", counts.failed)}
        {chip("unfinished", "Unfinished", counts.unfinished)}
        {chip("kept", "Kept", counts.kept)}
        {chip("edited", "Edited", counts.edited)}
      </div>

      {showBulk ? (
        <div
          className={cn(
            "flex shrink-0 flex-wrap items-center gap-2 border-b border-border px-3 py-1.5",
            filter === "failed" ? "bg-critical/[0.06]" : "bg-sunken",
          )}
        >
          <span className="min-w-0 flex-1 text-[11px] text-secondary">
            {filter === "failed" ? (
              <>{lines} the pipeline could not say. One job, not {bulkUids.length}.</>
            ) : (
              <>
                {lines} with nothing to play
                {needText > 0 ? ` — ${needText} of them have no translation yet` : null}
                {renderUids.length > 0
                  ? `${bulkUids.length > 0 ? ", and " : ""}${renderUids.length} already voiced and waiting for a render`
                  : null}
                . One click, at most two jobs.
              </>
            )}
          </span>
          {/* A voiced-but-unplaced line's button is Render, and it is a
              different button on purpose: the model has already done its work
              on this line, and only `timeline.place` can finish it. */}
          {renderUids.length > 0 && filter === "unfinished" ? (
            <Button size="xs" onClick={onRender}>
              <Film className="h-3 w-3" />
              Render {renderUids.length}
            </Button>
          ) : null}
          {bulkUids.length === 0 ? null : filter === "unfinished" ? (
            <Button size="xs" onClick={() => onFixMany(bulkUids)}>
              {needText > 0 ? (
                <>
                  <Languages className="h-3 w-3" />
                  Translate &amp; voice these {bulkUids.length}
                </>
              ) : (
                <>
                  <Volume2 className="h-3 w-3" />
                  Re-voice these {bulkUids.length}
                </>
              )}
            </Button>
          ) : (
            <>
              {voiceUids.length > 0 ? (
                <Button size="xs" onClick={() => onResynthesizeMany(voiceUids)}>
                  <Volume2 className="h-3 w-3" />
                  Re-voice these {voiceUids.length}
                </Button>
              ) : null}
              {textUids.length > 0 ? (
                <Button size="xs" onClick={() => onRetranslateMany(textUids)}>
                  <Languages className="h-3 w-3" />
                  Re-translate these {textUids.length}
                </Button>
              ) : null}
            </>
          )}
        </div>
      ) : null}

      {showDubBulk ? (
        <div
          data-bulk="kept"
          className="flex shrink-0 flex-wrap items-center gap-2 border-b border-border bg-sunken px-3 py-1.5"
        >
          <span className="min-w-0 flex-1 text-[11px] text-secondary">
            {searching ? (
              <>
                {keptLines} play as recorded — the ones this search leaves on screen, not every
                kept line in the run.
              </>
            ) : (
              <>{keptLines} play as recorded, subtitled. One click, at most two jobs.</>
            )}
          </span>
          <ConfirmButton
            size="xs"
            confirmLabel={`Dub these ${keptUids.length}`}
            message={
              <>
                {keptUids.length} line{keptUids.length === 1 ? "" : "s"} switch to dubbed
                {searching ? " — the ones this search leaves on screen" : ""}; translate + voice
                queue behind any running job.
              </>
            }
            onConfirm={() => onDubMany(keptUids)}
          >
            <Mic2 className="h-3 w-3" />
            Dub these {keptUids.length}
          </ConfirmButton>
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

