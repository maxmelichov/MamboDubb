/**
 * The script.
 *
 * This is the editor's largest region 58% of the width and all of the height
 * between the header and the timeline strip which is the whole argument of
 * the redesign in one number. The video used to be the pane that grew and the
 * text was a fifteen-rem drawer at the bottom; but nobody reviews a dub by
 * watching it, they review it by *reading* it, with the video as the reference
 * they glance at. So the text grows and the picture is fixed.
 *
 * Above the rows sit the two things that make two hundred rows navigable: a
 * search box and five filter chips. They are not decoration "show me the
 * eleven that failed" is the single most common thing a reviewer wants from a
 * run, and it used to require scrolling the whole list looking for a hue.
 * "Unfinished" is the second most common and used to be unaskable: a line with
 * no translation or no clip is invisible in a list of two hundred, and it is
 * exactly what a verdict flip leaves behind.
 *
 * And once a filter has selected a set, fixing that set is one job rather than
 * eleven: `POST /resynthesize` has always taken `{uids:[…]}` and the UI has
 * always sent exactly one. Eleven separate jobs is eleven model loads, in a
 * process where only one model may be resident at a time so it is not a
 * convenience, it is the difference between one minute and twenty.
 *
 * Kept is the third chip with a bulk offer and the only one whose button
 * changes a *verdict* rather than re-running a stage, which is why it is the
 * only one that asks first.
 *
 * The search says what it found, in three places, because a search that only
 * removes rows is a search you have to audit by scrolling: the All chip counts
 * the hits against the run ("11 / 73"), every hit is marked inside its own line,
 * and the timeline's marks for everything else drop back. Between them they
 * answer how many, which words, and where.
 */

import { useEffect, useMemo, useRef } from "react";
import { Film, Languages, ListX, Mic2, Search, Volume2 } from "lucide-react";
import { cn } from "../lib/classNames";
import { lineCount, theseLines } from "../lib/format";
import {
  hasLocks,
  hasTranscript,
  modelCost,
  needsModelWork,
  segmentState,
  unfinished,
} from "../lib/segments";
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
 * visible*, not through the whole run an arrow key that jumps to a row the
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
  onSettleKeep,
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
  onSettleKeep: (seg: Segment) => void;
  onRetranslateMany: (uids: string[]) => void;
  onResynthesizeMany: (uids: string[]) => void;
  /** Translate whatever has no line, then voice all of them in that order. */
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
   * The guard is the same one the timeline strip uses only scroll when the
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
   * A row asked for by name today, by a click on its mark in the timeline.
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

  const needle = query.trim().toLowerCase();

  /*
   * How many lines the search found, out of how many there are.
   *
   * The search box gave no feedback at all: you typed, rows disappeared, and
   * the only way to learn whether "moza" matched eleven lines or none was to
   * scroll to the bottom and count with the All chip, two inches away, still
   * confidently reading 73. It is the chip that means "no filter", so it is the
   * one that owes an answer about the *other* filter that is running.
   */
  const found = useMemo(
    () => (needle ? filterSegments(segments, query, "all").length : null),
    [needle, query, segments],
  );

  const chip = (key: ScriptFilter, label: string, count: number, text?: string) => (
    <button
      key={key}
      type="button"
      data-chip={key}
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
          the bar comes up. The rest stay quiet a filter bar where every chip
          is coloured is a filter bar with no emphasis in it and the hue is
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
      <span className="font-mono tabular-nums">{text ?? count}</span>
    </button>
  );

  /*
   * The bulk offer is tied to the two chips where "do all of these" is
   * unambiguous: Failed, every one of which is a line the pipeline could not
   * say, and Unfinished, every one of which is a line with nothing to play.
   *
   * Unfinished is the one that gets a run's stragglers back including the
   * lines a "Dub it" used to strand, which had no translation, no clip and no
   * job on the way. Its button is one click and at most two jobs: translate
   * whatever has no line, then voice the lot.
   *
   * The work set is not "everything on screen that is not a keep", though, and
   * it was wrong in both directions. A line the *pipeline* gave up on is stored
   * as a keep (`keep_reason` tts_failed / mt_failed) with the original audio
   * attached so excluding keeps excluded every member of the Failed filter,
   * and its two buttons enqueued jobs for nobody. And a line that has a clip
   * but no placement needs a render, not a model: re-voicing it is a minute of
   * GPU to arrive back exactly where it started.
   */
  const bulkCandidates = visible.filter(
    (seg) => needsModelWork(seg) && (!seg.keep || segmentState(seg) === "failed"),
  );
  /*
   * …and a line with nothing written on it is not model work either.
   *
   * `segments.fill_uncovered_audible` writes spans with `text: ""` audible
   * stretches the transcript never claimed and a flip to "Dub it" puts one in
   * this set. Translating an empty string and voicing the result is a minute of
   * GPU spent replacing correct original audio with garbage, so they are skipped
   * and the count says how many, rather than being quietly folded in.
   */
  const bulkSegs = bulkCandidates.filter(hasTranscript);
  const bulkSkipped = bulkCandidates.length - bulkSegs.length;
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
  const lines = lineCount(bulkUids.length);

  const searching = query.trim().length > 0;
  /*
   * Every bulk bar names its set precisely once a search has narrowed it.
   *
   * The Kept bar has said this since it was written the search box is a filter
   * like any other, and a button that quietly acted on two hundred lines while
   * eleven are on screen is the worst kind of bulk action. The Failed and
   * Unfinished bars had the same property and never said so; it is one sentence,
   * so they say it the same way rather than each in its own words.
   */
  const searchNote = (noun: string) =>
    searching ? ` (the ones this search leaves on screen, not every ${noun} line in the run)` : "";
  /** What a work set left behind, and why. Reads as nothing when it left nothing. */
  const skipNote = (n: number) => (n > 0 ? ` (${n} skipped: no transcript to translate)` : "");

  /*
   * The third bulk offer, and the only one that changes a verdict rather than
   * re-running a stage.
   *
   * A kept line plays as recorded. When the keep rule was wrong about a whole
   * video a third language read as "already the target", which is what a
   * Hebrew→German run full of English used to be every one of those lines is
   * a keep the reviewer disagrees with, and the only way out was the per-line
   * flip, N times. This is that flip over the set on screen.
   *
   * It is the set *on screen* and not "every kept line in the run": the search
   * box is a filter like any other, and a button that quietly acted on two
   * hundred lines when eleven are showing would be the worst kind of bulk
   * action. It says which it is, in the sentence beside it.
   */
  const keptSegs = filter === "kept" ? visible.filter(hasTranscript) : [];
  const keptUids = keptSegs.map((seg) => seg.uid);
  const keptSkipped = filter === "kept" ? visible.length - keptSegs.length : 0;
  // What the flip will cost: every line is voiced, and every line whose
  // translation is the pipeline's is translated again first `set_keep`
  // discards it on the way through, and only a hand-written one is locked
  // against that.
  const keptNeedText = keptSegs.filter(
    (seg) => !((seg.text_en ?? "").trim() && seg.locked?.text_en),
  ).length;
  const showDubBulk = filter === "kept" && keptUids.length > 0;
  const keptLines = lineCount(keptUids.length);

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
        it needs an opaque ground and a hairline under it with a wash or a
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
        {chip("all", "All", counts.all, found != null ? `${found} / ${counts.all}` : undefined)}
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
              <>
                {lines} the pipeline could not say{searchNote("failed")}.
                {/* "One job, not 1." is a promise about nothing. */}
                {bulkUids.length > 1 ? ` One job, not ${bulkUids.length}.` : null}
              </>
            ) : (
              <>
                {lines} with nothing to play
                {needText > 0 ? `: ${needText} of them have no translation yet` : null}
                {renderUids.length > 0
                  ? `${bulkUids.length > 0 ? ", and " : ""}${renderUids.length} already voiced and waiting for a render`
                  : null}
                {bulkSkipped > 0
                  ? `${bulkUids.length + renderUids.length > 0 ? ", and " : ""}${bulkSkipped} with no transcript to translate`
                  : null}
                {searchNote("unfinished")}. One click, at most two jobs.
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
          {/*
            Every one of these is minutes of model time per line, and until now
            each was a single unguarded click: "Re-voice these 27" is half an
            hour of GPU, asked for and started in the same gesture. They ask
            first, and what they ask with is the price the same ~20s / ~1min
            the selection panel already quotes for one line, multiplied by the
            work set each button actually sends.
          */}
          {bulkUids.length === 0 ? null : filter === "unfinished" ? (
            <ConfirmButton
              size="xs"
              confirmLabel={
                needText > 0
                  ? `Translate & voice ${theseLines(bulkUids.length)}`
                  : `Re-voice ${theseLines(bulkUids.length)}`
              }
              message={
                <>
                  {lines}
                  {skipNote(bulkSkipped)} ·{" "}
                  {modelCost({ translate: needText, voice: bulkUids.length })}
                  {searchNote("unfinished")}. Whatever has no line is translated first, then
                  the lot is voiced: two jobs, in that order.
                </>
              }
              onConfirm={() => onFixMany(bulkUids)}
            >
              {needText > 0 ? (
                <>
                  <Languages className="h-3 w-3" />
                  Translate &amp; voice {theseLines(bulkUids.length)}
                </>
              ) : (
                <>
                  <Volume2 className="h-3 w-3" />
                  Re-voice {theseLines(bulkUids.length)}
                </>
              )}
            </ConfirmButton>
          ) : (
            <>
              {voiceUids.length > 0 ? (
                <ConfirmButton
                  size="xs"
                  confirmLabel={`Re-voice ${theseLines(voiceUids.length)}`}
                  message={
                    <>
                      {lineCount(voiceUids.length)} · {modelCost({ voice: voiceUids.length })}
                      {searchNote("failed")}. Each one is synthesized again and its old clip
                      replaced.
                    </>
                  }
                  onConfirm={() => onResynthesizeMany(voiceUids)}
                >
                  <Volume2 className="h-3 w-3" />
                  Re-voice {theseLines(voiceUids.length)}
                </ConfirmButton>
              ) : null}
              {textUids.length > 0 ? (
                <ConfirmButton
                  size="xs"
                  confirmLabel={`Re-translate ${theseLines(textUids.length)}`}
                  message={
                    <>
                      {lineCount(textUids.length)} · {modelCost({ translate: textUids.length })}
                      {searchNote("failed")}. The model's line replaces what is there now,
                      including anything written by hand.
                    </>
                  }
                  onConfirm={() => onRetranslateMany(textUids)}
                >
                  <Languages className="h-3 w-3" />
                  Re-translate {theseLines(textUids.length)}
                </ConfirmButton>
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
          {/* The sentence counts what is on screen; the button counts what it
              will actually send, which is not the same number once the spans
              with no transcript are out of the work set. */}
          <span className="min-w-0 flex-1 text-[11px] text-secondary">
            {lineCount(visible.length)} play as recorded
            {searching ? null : ", subtitled"}
            {keptSkipped > 0 ? `: ${keptSkipped} with no transcript to translate` : null}
            {searchNote("kept")}. One click, at most two jobs.
          </span>
          <ConfirmButton
            size="xs"
            confirmLabel={`Dub ${theseLines(keptUids.length)}`}
            message={
              <>
                {keptLines}
                {skipNote(keptSkipped)} ·{" "}
                {modelCost({ translate: keptNeedText, voice: keptUids.length })}
                {searchNote("kept")}. {keptUids.length === 1 ? "It switches" : "They switch"} to
                dubbed; translate + voice queue behind any running job.
              </>
            }
            onConfirm={() => onDubMany(keptUids)}
          >
            <Mic2 className="h-3 w-3" />
            Dub {theseLines(keptUids.length)}
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
          // Every scroll this list performs is a scroll *to a row* the
          // playhead's, the selection's, ↑/↓'s and a row landed flush under
          // the filter bar reads as text sliced off by it. `scroll-padding`
          // is the platform's answer: it is the inset every one of those
          // scrolls, including `scrollIntoView`, aligns to.
          "scroll-pt-3 scroll-pb-3",
          // Past a few hundred rows the browser skips the layout and paint of
          // everything off screen. No windowing library, no measured
          // virtualiser, no new dependency the platform has done this since
          // 2021 and `contain-intrinsic-size` keeps the scrollbar honest.
          segments.length > VIRTUALIZE_ABOVE && "script-virtual",
        )}
      >
        {visible.length === 0 ? (
          <Empty icon={ListX} title={segments.length === 0 ? "No segments yet" : "Nothing matches"}>
            {segments.length === 0 ? (
              <>
                Segments appear once the <code className="font-mono">segments</code> stage has
                run. It is what turns the transcript's words into the lines you review here.
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
              query={needle}
              onSelect={onSelect}
              onEdit={onEdit}
              onCommit={onCommit}
              onPlay={onPlay}
              onToggleKeep={onToggleKeep}
              onSettleKeep={onSettleKeep}
            />
          ))
        )}
      </div>
    </section>
  );
}

