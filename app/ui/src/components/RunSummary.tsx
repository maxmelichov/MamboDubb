/**
 * The editor rail when no line is selected: what happened to this run, and
 * where there is speech nothing covers. Its tally and its gap list are shared
 * with the run menu, which shows the same answers from behind a button.
 */

import { useMemo, useState } from "react";
import { Plus, TriangleAlert } from "lucide-react";
import {
  Button,
  Eyebrow,
  Field,
  NumberInput,
  Popover,
  Select,
  StateIcon,
  TextArea,
} from "./ui";
import { languageName, timecode } from "../lib/format";
import {
  STATE_META,
  keepReason,
  keptAsTargetLanguage,
  unclaimedSpans,
  type SegmentState,
  type Span,
} from "../lib/segments";
import type { NewSegment, ProjectDetail, Segment } from "../lib/types";

/**
 * What the rail says when nothing is selected.
 *
 * It used to say "No line selected", centred, in a column six hundred pixels
 * tall an empty state that filled two fifths of the screen with an
 * instruction the user had already worked out. The space is worth more than
 * that: nothing is selected exactly when a run has just been opened, which is
 * the moment "what happened here" is the only question there is.
 *
 * So it answers that question and no other: how the lines came out, why the
 * kept ones were kept, and where there is speech nothing covers. All of it is
 * derived from segments already on the client no request, no report file —
 * and it is set as prose and a tally rather than as a dashboard, because it is
 * read once per run and then replaced by the panel for a line.
 */
export function RunSummary({
  segments,
  project,
  counts,
  total,
  claimGap,
  onSeek,
  onAdd,
  onHighlightGap,
}: {
  segments: Segment[];
  project: ProjectDetail | null;
  counts: Record<SegmentState, number>;
  /** The run's length, so the live coverage below has an end to measure to. */
  total: number;
  /** A hatch the reviewer clicked on the strip pinned onto the gap list. */
  claimGap?: Span | null;
  onSeek: (time: number) => void;
  /**
   * Adding lives here and not in the run menu behind the header.
   *
   * The rail is the surface that is *up* whenever no line is selected, which is
   * exactly when a reviewer is working through the uncovered spans, and a
   * composer nested inside the menu's popover would be a dialog inside a dialog
   * in a 21rem panel. The menu keeps the same list, read-only.
   *
   * It answers whether the server took the segment the composer below cannot
   * clear itself until it knows.
   */
  onAdd?: (segment: NewSegment) => Promise<boolean>;
  onHighlightGap?: (span: Span | null) => void;
}) {
  const count = segments.length;

  /*
   * The report's gaps, clipped to what is still uncovered *now*.
   *
   * `uncovered_audible` is a finding from the last render, and the script has
   * usually moved since: a span the reviewer just claimed stayed on this list
   * offering a + whose only possible answer was the server's overlap refusal,
   * until a render happened to re-run the report. The report still decides
   * which silences are worth listing it is the only party that has *heard*
   * them but the live coverage decides how much of each is still open, so a
   * claimed gap drops off the moment the segment holding it lands.
   */
  const gaps = useMemo(() => {
    const live = unclaimedSpans(segments, total);
    return (project?.report?.uncovered_audible ?? []).flatMap((gap) =>
      live
        .map((span) => ({ start: Math.max(gap.start, span.start), end: Math.min(gap.end, span.end) }))
        .filter((span) => span.end - span.start > 0.05)
        .map((span) => ({ ...span, duration: span.end - span.start })),
    );
  }, [project, segments, total]);

  /*
   * Why the kept lines were kept, most common first.
   *
   * The reasons arrive as the pipeline's own tokens (`latin`, `speaker_en`,
   * `user`, `tts_failed`, …) and were printed raw, which put "Kept because
   * user" on the screen of the user who pressed the button. `keepReason` is the
   * one mapping, shared with the selection panel; a token it has never seen
   * still falls through to itself rather than being given an invented meaning.
   */
  const reasons = useMemo(() => {
    const tally = new Map<string, number>();
    for (const seg of segments) {
      if (!seg.keep) continue;
      const reason = keepReason(seg.keep_reason);
      tally.set(reason, (tally.get(reason) ?? 0) + 1);
    }
    return [...tally.entries()].sort((a, b) => b[1] - a[1]);
  }, [segments]);

  // "Already in the target language" is a claim about *why*, so it is only made
  // when the reasons say so a run kept by hand or by a failure is a different
  // story with the same count.
  const passthrough = segments.filter(
    (seg) => seg.keep && keptAsTargetLanguage(seg.keep_reason),
  ).length;
  const mostlyKept = count > 0 && counts.kept / count >= 0.6;

  const lead =
    count === 0
      ? "No lines yet: the segments stage is what fills this list."
      : mostlyKept && passthrough >= counts.kept / 2
        ? `This video mostly speaks ${languageName(project?.source.tgt_lang)} already: ` +
          `${counts.kept} of ${count} lines keep their original audio.`
        : mostlyKept
          ? `${counts.kept} of ${count} lines keep their original audio.`
          : `${counts.dubbed} of ${count} lines are dubbed.`;

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3.5 overflow-y-auto px-3 py-3">
      <div>
        <Eyebrow>This run</Eyebrow>
        <p className="mt-1.5 text-[14px] leading-snug text-primary">{lead}</p>
      </div>

      <StateTally counts={counts} />

      {reasons.length > 0 ? (
        <div>
          <Eyebrow className="mb-1.5">Kept because</Eyebrow>
          <p className="text-[12.5px] leading-relaxed text-muted">
            {reasons.map(([reason, n], i) => (
              <span key={reason}>
                {i > 0 ? " · " : null}
                <span className="text-secondary">{reason}</span> {n}
              </span>
            ))}
          </p>
        </div>
      ) : null}

      <GapList
        gaps={gaps}
        pinned={claimGap}
        speakers={Object.keys(project?.speakers ?? {})}
        onSeek={onSeek}
        onAdd={onAdd}
        onHighlight={onHighlightGap}
        stale={project?.report?.stale}
      />

      <p className="mt-auto pt-2 text-[11px] leading-relaxed text-muted">
        Pick a line in the script: everything that is true about it, and cannot fit on a row,
        is here.
      </p>
    </div>
  );
}

/** The five states and their counts, as a tally with leader lines. */
export function StateTally({ counts }: { counts: Record<SegmentState, number> }) {
  const states = (Object.keys(STATE_META) as SegmentState[]).filter((s) => counts[s] > 0);
  return (
    <dl className="flex flex-col gap-1 text-[12.5px]">
      {states.map((s) => (
        <div key={s} className="flex items-baseline gap-2">
          <dt className="flex items-center gap-1.5 text-secondary">
            <StateIcon state={s} className="h-2.5 w-2.5" />
            {STATE_META[s].label}
          </dt>
          <span className="h-px flex-1 bg-border" aria-hidden />
          <dd className="font-mono tabular-nums text-primary">{counts[s]}</dd>
        </div>
      ))}
    </dl>
  );
}

/**
 * Speech in the source that no segment claims, so the dub plays the original
 * over it. The highest-value readout `report.json` produces, and every row of
 * it is a button that seeks there a list of timecodes you cannot jump to is
 * a list of homework.
 */
export function GapList({
  gaps,
  pinned = null,
  speakers = [],
  onSeek,
  onAdd,
  onHighlight,
  stale = false,
  className,
}: {
  gaps: { start: number; end: number; duration: number }[];
  /**
   * A span picked on the strip itself, listed whether or not the report knew
   * it. The hatches are drawn from the live segments, so a gap opened since
   * the last render exists there and nowhere in `gaps`; clicking one pins it
   * here, above the list, under its own eyebrow because the list's is a
   * claim about *audible* speech and the strip only knows about time. When a
   * listed gap already covers the span, the list row is the control and the
   * pin stays quiet.
   */
  pinned?: Span | null;
  /** The run's voices, so a new line can be attributed to one of them. */
  speakers?: string[];
  onSeek: (time: number) => void;
  /**
   * Claim one of these spans as a segment.
   *
   * This is where adding belongs, and the only place it does: a segment needs a
   * span, and these spans are precisely the ones the pipeline left uncovered —
   * the hallucination-dropped intro, the line under music the ASR skipped. A
   * free-floating "New segment" button would have to invent a span, and every
   * span it could invent either overlaps a neighbour (which the server refuses)
   * or is one of these.
   *
   * Resolves to whether the span was taken; a refusal is the composer's to keep.
   */
  onAdd?: (segment: NewSegment) => Promise<boolean>;
  /**
   * Point at one and the timeline lights the hatch it is inside.
   *
   * Hover *and* focus, because half the reason this list exists is that it is
   * reachable a keyboard user tabbing down it gets the same map the mouse
   * does. The bounds here are the report's, and the hatch's are derived from
   * the segments on screen, so the timeline matches them by overlap rather than
   * by equality; they are two measurements of the same silence.
   */
  onHighlight?: (span: Span | null) => void;
  /** These spans were found by the last report, not by the script on screen. */
  stale?: boolean;
  className?: string;
}) {
  const extra =
    pinned && !gaps.some((gap) => gap.start < pinned.end && gap.end > pinned.start)
      ? { start: pinned.start, end: pinned.end, duration: pinned.end - pinned.start }
      : null;
  if (gaps.length === 0 && !extra) return null;
  return (
    <div className={className}>
      {extra ? (
        <div className="mb-3.5" data-pinned-gap>
          <Eyebrow className="mb-1.5">From the timeline</Eyebrow>
          <p className="mb-1.5 text-[11px] leading-relaxed text-muted">
            The span you picked on the strip. No segment claims it, so the dub plays the
            original there{onAdd ? "; claim it if it should be dubbed" : ""}.
          </p>
          {/* Keyed on the span: picking a second hatch has to reseed the
              composer, not hand it the first hatch's numbers. */}
          <ul className="flex flex-col gap-1">
            <GapRow
              key={`${extra.start}-${extra.end}`}
              gap={extra}
              speakers={speakers}
              onSeek={onSeek}
              onAdd={onAdd}
              onHighlight={onHighlight}
            />
          </ul>
        </div>
      ) : null}
      {gaps.length > 0 ? (
        <>
          {/*
            A gap list is a finding, and a finding has a date. The rows are
            clipped live against the segments on screen (see `RunSummary`), so
            a claimed span leaves at once but a gap the report never *found*
            still cannot appear until a render re-runs it, which is what the
            eyebrow admits when the report is behind.
          */}
          <Eyebrow className="mb-1.5">
            Audible, uncovered: {gaps.length}
            {stale ? " · from the last render" : ""}
          </Eyebrow>
          <p className="mb-1.5 text-[11px] leading-relaxed text-muted">
            Speech here is in the source but no segment claims it, so the dub plays the original.
            Jump to one and listen{onAdd ? ", and claim it if it should be dubbed" : ""}.
          </p>
          <ul className="flex max-h-48 flex-col gap-1 overflow-y-auto">
            {gaps.map((gap) => (
              <GapRow
                key={`${gap.start}-${gap.end}`}
                gap={gap}
                speakers={speakers}
                onSeek={onSeek}
                onAdd={onAdd}
                onHighlight={onHighlight}
              />
            ))}
          </ul>
        </>
      ) : null}
    </div>
  );
}

/** One uncovered span: a seek button, and the + composer when adding is wired. */
function GapRow({
  gap,
  speakers,
  onSeek,
  onAdd,
  onHighlight,
}: {
  gap: { start: number; end: number; duration: number };
  speakers: string[];
  onSeek: (time: number) => void;
  onAdd?: (segment: NewSegment) => Promise<boolean>;
  onHighlight?: (span: Span | null) => void;
}) {
  return (
    <li
      className="flex items-stretch gap-1"
      onMouseEnter={() => onHighlight?.({ start: gap.start, end: gap.end })}
      onMouseLeave={() => onHighlight?.(null)}
    >
      <button
        type="button"
        data-gap={gap.start}
        onClick={() => onSeek(gap.start)}
        onFocus={() => onHighlight?.({ start: gap.start, end: gap.end })}
        onBlur={() => onHighlight?.(null)}
        className="flex min-w-0 flex-1 items-center gap-2 rounded-lg border border-border bg-raised px-2 py-1 text-left text-[12.5px] transition-colors hover:border-axis hover:bg-sunken"
      >
        <TriangleAlert className="h-3 w-3 shrink-0 text-muted" aria-hidden />
        <span className="font-mono tabular-nums text-primary">{timecode(gap.start)}</span>
        <span className="ml-auto text-muted">{gap.duration.toFixed(1)}s</span>
      </button>
      {onAdd ? <AddSegmentButton gap={gap} speakers={speakers} onAdd={onAdd} /> : null}
    </li>
  );
}

/** Below this the synthesiser reliably fails a line `segments.MIN_SEG_SEC`. */
const MIN_SEGMENT_SEC = 0.9;

/**
 * Claim an uncovered span as a segment.
 *
 * A composer, not a one-click "add": a segment needs *words*. The server
 * refuses one without them and it is right to — `text` is what the translator
 * reads and the voice speaks, and there is no transcript for a span the
 * pipeline never segmented, which is exactly why the span is on this list.
 *
 * The span itself is the gap's, editable, because the report's bounds are where
 * *audible* speech was found and a sentence usually starts a beat before that.
 * They are two number inputs rather than drag handles for the reason the strip
 * has no trim handles at all: `timeline.place()` is the sole authority on where
 * audio goes, so a span is a value the user states, not a thing they nudge.
 *
 * Nothing here validates the overlap. The server does, and refuses rather than
 * clamping; duplicating that rule in the client would put a second, drifting
 * copy of "where is there room" in the one place that cannot see the manifest.
 */
function AddSegmentButton({
  gap,
  speakers,
  onAdd,
}: {
  gap: { start: number; end: number };
  speakers: string[];
  onAdd: (segment: NewSegment) => Promise<boolean>;
}) {
  const [text, setText] = useState("");
  const [speaker, setSpeaker] = useState("");
  const [start, setStart] = useState(gap.start.toFixed(2));
  const [end, setEnd] = useState(gap.end.toFixed(2));
  const span = Number(end) - Number(start);
  const ready = text.trim().length > 0 && Number.isFinite(span) && span >= MIN_SEGMENT_SEC;

  return (
    <Popover
      label={`Add a segment at ${timecode(gap.start)}`}
      title="New segment"
      align="right"
      className="w-72"
      trigger={<Plus className="h-3.5 w-3.5" />}
    >
      <div className="flex flex-col gap-2">
        <p className="text-[11px] leading-relaxed text-muted">
          Type what is said here. It goes through translate and voice on the next run, like
          any other line, and the words are kept as yours.
        </p>
        <TextArea
          rows={3}
          dir="auto"
          className="auto-dir"
          value={text}
          placeholder="what is spoken in this span"
          onChange={(event) => setText(event.currentTarget.value)}
        />
        <div className="flex items-end gap-2">
          <Field label="Start" className="flex-1">
            <NumberInput
              step={0.05}
              suffix="s"
              value={start}
              onChange={(event) => setStart(event.currentTarget.value)}
            />
          </Field>
          <Field label="End" className="flex-1">
            <NumberInput
              step={0.05}
              suffix="s"
              value={end}
              onChange={(event) => setEnd(event.currentTarget.value)}
            />
          </Field>
        </div>
        <Field label="Speaker" hint="blank inherits the nearest line's voice">
          <Select value={speaker} onChange={(event) => setSpeaker(event.currentTarget.value)}>
            <option value="">nearest</option>
            {speakers.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </Select>
        </Field>
        <Button
          size="md"
          className="self-start"
          disabled={!ready}
          title={
            ready
              ? "Add this segment"
              : text.trim()
                ? `A segment has to be at least ${MIN_SEGMENT_SEC}s long to be voiced`
                : "A new segment needs its text"
          }
          onClick={async () => {
            // The sentence stays in the box until a segment is holding it. The
            // server refuses an overlap outright, and clearing on the way out
            // deleted the very text the refusal is asking the user to re-place.
            const added = await onAdd({
              start: Number(start),
              end: Number(end),
              text: text.trim(),
              ...(speaker ? { speaker } : {}),
            });
            if (added) setText("");
          }}
        >
          <Plus className="h-3.5 w-3.5" />
          Add segment
        </Button>
      </div>
    </Popover>
  );
}
