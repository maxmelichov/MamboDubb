/**
 * What the inspector rail shows when no segment is selected.
 *
 * It used to show one sentence — "pick a segment" — in the middle of a 26rem
 * column of white, which is the largest piece of dead space in the app and
 * also the moment the user has the least idea what to look at. The run's own
 * report answers that: which segments failed, how far the placement drifted,
 * how hard the time-fit had to squeeze, and where there is audible speech that
 * nothing covers. That last one is the highest-value thing on the screen, so
 * every gap is a button that seeks the playhead to it.
 *
 * Everything here is read from `report.json` as the pipeline wrote it. Nothing
 * is recomputed and nothing is editorialised beyond the thresholds already in
 * `lib/segments.ts`.
 */

import { ArrowUpRight, ListChecks, TriangleAlert } from "lucide-react";
import { cn } from "../lib/classNames";
import { timecode } from "../lib/format";
import { STATE_META, type SegmentState } from "../lib/segments";
import { Disclosure, Eyebrow, SectionLabel } from "./ui";
import type { ProjectDetail } from "../lib/types";

export function RunSummary({
  project,
  counts,
  total,
  onSeek,
}: {
  project: ProjectDetail | null;
  counts: Record<SegmentState, number>;
  total: number;
  onSeek: (time: number) => void;
}) {
  const report = project?.report ?? null;
  const segments = Object.values(counts).reduce((sum, n) => sum + n, 0);
  const gaps = report?.uncovered_audible ?? [];

  return (
    <div className="flex h-full flex-col overflow-y-auto">
      <header className="sticky top-0 z-10 border-b border-border bg-sunken px-4 py-2.5">
        <Eyebrow>This run</Eyebrow>
        <p className="mt-1 text-[13px] font-semibold text-primary">
          {segments} segment{segments === 1 ? "" : "s"} over {timecode(total, 0)}
        </p>
      </header>

      <div className="flex flex-col gap-3.5 px-4 py-3.5">
        <section className="flex flex-col gap-1.5">
          <SectionLabel icon={ListChecks}>Coverage</SectionLabel>
          {(Object.keys(STATE_META) as SegmentState[]).map((state) => (
            <StateRow
              key={state}
              state={state}
              count={counts[state]}
              of={Math.max(1, segments)}
            />
          ))}
        </section>

        {report ? (
          <>
            <section className="grid grid-cols-2 gap-2">
              <Stat
                label="Max drift"
                value={`${report.drift.max.toFixed(2)}s`}
                hint={`mean ${report.drift.mean.toFixed(2)}s`}
                concern={report.drift.max > 0.5}
              />
              <Stat
                label="Fastest fit"
                value={`${report.speed.max.toFixed(2)}×`}
                hint={`${report.speed.compressed} compressed`}
                concern={report.speed.max > 1.25}
              />
            </section>

            {gaps.length > 0 ? (
              <section className="flex flex-col gap-1.5">
                <SectionLabel icon={TriangleAlert}>
                  Audible, uncovered — {gaps.length}
                </SectionLabel>
                <p className="text-[12px] leading-relaxed text-muted">
                  Speech here is in the source but no segment claims it, so the dub plays the
                  original. Jump to one and listen.
                </p>
                <ul className="flex flex-col gap-1">
                  {gaps.map((gap) => (
                    <li key={`${gap.start}-${gap.end}`}>
                      <button
                        type="button"
                        onClick={() => onSeek(gap.start)}
                        className="flex w-full items-center gap-2 rounded-lg border border-border bg-raised px-2.5 py-1.5 text-left text-[12px] transition-colors hover:border-axis hover:bg-sunken"
                      >
                        <span className="font-mono tabular-nums text-primary">
                          {timecode(gap.start)}
                        </span>
                        <span className="text-muted">
                          {gap.duration.toFixed(1)}s
                        </span>
                        <ArrowUpRight className="ml-auto h-3.5 w-3.5 shrink-0 text-muted" aria-hidden />
                      </button>
                    </li>
                  ))}
                </ul>
              </section>
            ) : null}

            {/* A breakdown by machine-readable reason code is diagnostic, not
                a headline — it goes on a shelf like everything else of that
                kind, with the total on the outside. */}
            {Object.keys(report.keep_reasons).length > 0 ? (
              <Disclosure
                id="run.keepReasons"
                label="Why segments were kept"
                summary={`${Object.keys(report.keep_reasons).length} reasons`}
              >
                <ul className="flex flex-col gap-1 text-[12px]">
                  {Object.entries(report.keep_reasons)
                    .sort((a, b) => b[1] - a[1])
                    .map(([reason, count]) => (
                      <li key={reason} className="flex items-baseline gap-2 text-secondary">
                        <code className="font-mono text-[11px]">{reason}</code>
                        <span className="h-px flex-1 bg-border" aria-hidden />
                        <span className="font-mono tabular-nums text-muted">{count}</span>
                      </li>
                    ))}
                </ul>
              </Disclosure>
            ) : null}
          </>
        ) : (
          <p className="text-[12px] leading-relaxed text-muted">
            No report yet — it is written by the last stage of the run, and it is where drift,
            time-fit and uncovered speech get summarised.
          </p>
        )}

        {/* The shortcuts that used to be repeated here are in the timeline's
            "?" popover now, which is also where the legend went. */}
        <p className="border-t border-border pt-3.5 text-[12px] leading-relaxed text-muted">
          Pick a segment in the timeline or the list below to edit it.
        </p>
      </div>
    </div>
  );
}

/** One state and its share of the run — the bar is the count, drawn. */
function StateRow({ state, count, of }: { state: SegmentState; count: number; of: number }) {
  const meta = STATE_META[state];
  return (
    <div className={cn("flex items-center gap-2 text-[12px]", count === 0 && "opacity-45")}>
      <span aria-hidden style={{ color: meta.token }} className="w-3 text-center">
        {meta.glyph}
      </span>
      <span className="w-24 shrink-0 text-secondary">{meta.label}</span>
      <span className="h-1.5 min-w-0 flex-1 overflow-hidden rounded-full bg-border">
        <span
          className="block h-full rounded-full"
          style={{ width: `${(count / of) * 100}%`, backgroundColor: meta.token }}
        />
      </span>
      <span className="w-6 shrink-0 text-right font-mono tabular-nums text-primary">{count}</span>
    </div>
  );
}

function Stat({
  label,
  value,
  hint,
  concern,
}: {
  label: string;
  value: string;
  hint: string;
  concern?: boolean;
}) {
  return (
    <div className="rounded-xl border border-border bg-sunken px-3 py-2.5">
      <Eyebrow>{label}</Eyebrow>
      <p
        className={cn(
          "mt-1 font-mono text-[15px] tabular-nums",
          concern ? "text-critical" : "text-primary",
        )}
      >
        {value}
      </p>
      <p className="text-[11px] text-muted">{hint}</p>
    </div>
  );
}
