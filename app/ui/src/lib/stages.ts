/**
 * Reading a run's position in the pipeline.
 *
 * `stages` arrives as a sparse map the server only writes a key once a stage
 * has been attempted so "not in the map" and "pending" mean the same thing,
 * and the count of `done` keys is the only honest measure of how far along a
 * run is. Three screens ask that question (the projects list, the job bar and
 * the preview placeholder) and they must not answer it three different ways.
 */

import { STAGES, type Stage, type StageStatus } from "./types";

export type StageSummary = {
  /** stages reported `done` */
  done: number;
  /** always STAGES.length the pipeline is fixed */
  total: number;
  /** the stage being worked on, or the first one not yet done */
  current: Stage | null;
  /** the first stage that reported `failed`, if any */
  failed: Stage | null;
  complete: boolean;
  started: boolean;
  /** one phrase for the row: "Complete", "Failed at tts", "Running translate" */
  label: string;
};

export function summarizeStages(stages: Partial<Record<Stage, StageStatus>> | undefined): StageSummary {
  const map = stages ?? {};
  const done = STAGES.filter((stage) => map[stage] === "done").length;
  const failed = STAGES.find((stage) => map[stage] === "failed") ?? null;
  const running = STAGES.find((stage) => map[stage] === "running") ?? null;
  const next = STAGES.find((stage) => map[stage] !== "done") ?? null;
  const complete = done === STAGES.length;
  const started = done > 0 || running != null || failed != null;

  return {
    done,
    total: STAGES.length,
    current: running ?? (complete ? null : next),
    failed,
    complete,
    started,
    label: failed
      ? `Failed at ${failed}`
      : complete
        ? "Complete"
        : running
          ? `Running ${running}`
          : started
            ? `Stopped after ${done} of ${STAGES.length}`
            : "Not started",
  };
}

/** The tone a badge should wear for that phrase. */
export function stageTone(summary: StageSummary): "good" | "warn" | "bad" | "neutral" {
  if (summary.failed) return "bad";
  if (summary.complete) return "good";
  if (summary.started) return "warn";
  return "neutral";
}

/** "3 minutes ago" runs are compared by recency far more often than by date. */
export function ago(epochSeconds: number): string {
  const seconds = Math.max(0, Date.now() / 1000 - epochSeconds);
  if (seconds < 45) return "just now";

  const scale: [number, string][] = [
    [60, "minute"],
    [60, "hour"],
    [24, "day"],
    [7, "week"],
    [4.35, "month"],
    [12, "year"],
  ];

  let value = seconds;
  let unit = "second";
  for (const [divisor, name] of scale) {
    if (value < divisor) break;
    value /= divisor;
    unit = name;
  }

  const rounded = Math.round(value);
  return `${rounded} ${unit}${rounded === 1 ? "" : "s"} ago`;
}
