/**
 * Derived segment state. Everything the timeline, the list and the inspector
 * colour or label comes from here, so there is exactly one definition of what
 * "failed" means.
 */

import type { Segment } from "./types";

export type SegmentState = "dubbed" | "kept" | "failed" | "pending";

/**
 * Invariant 1 (AGENTS.md) says every segment is either dubbed or plays its
 * original audio — so "failed" is not a normal outcome, it is the pipeline
 * having lost a segment, and it is the thing the reviewer must see first.
 */
export function segmentState(seg: Segment): SegmentState {
  if (seg.tts?.verify === "failed") return "failed";
  if (seg.keep) return "kept";
  if (seg.place) return "dubbed";
  // Not kept and not placed: either still being worked on, or dropped.
  return seg.text_en ? "pending" : "pending";
}

/** Colour is never the only channel: every state carries a glyph and a word. */
export const STATE_META: Record<
  SegmentState,
  { label: string; short: string; glyph: string; token: string }
> = {
  dubbed: { label: "Dubbed", short: "Dub", glyph: "◆", token: "var(--color-dubbed)" },
  kept: { label: "Kept original", short: "Keep", glyph: "▣", token: "var(--color-kept)" },
  failed: { label: "Failed", short: "Fail", glyph: "✕", token: "var(--color-failed)" },
  pending: { label: "Not yet dubbed", short: "Wait", glyph: "◇", token: "var(--color-unclaimed)" },
};

export const UNCLAIMED_META = {
  label: "Unclaimed time",
  glyph: "╱",
  token: "var(--color-unclaimed)",
};

export type Span = { start: number; end: number };

/**
 * Time no segment covers. `report.uncovered_audible` flags the dangerous
 * subset (gaps with speech in them); this is the whole picture, so silence
 * between lines reads as gap-not-problem.
 */
export function unclaimedSpans(segments: Segment[], total: number, minGap = 0.35): Span[] {
  const gaps: Span[] = [];
  let cursor = 0;
  for (const seg of [...segments].sort((a, b) => a.start - b.start)) {
    if (seg.start - cursor >= minGap) gaps.push({ start: cursor, end: seg.start });
    cursor = Math.max(cursor, seg.end);
  }
  if (total - cursor >= minGap) gaps.push({ start: cursor, end: total });
  return gaps;
}

/** Where a segment's audio actually sits in the output, which is not `start`. */
export function placedSpan(seg: Segment): Span {
  return seg.place ? { start: seg.place.start, end: seg.place.end } : { start: seg.start, end: seg.end };
}

export function totalDuration(segments: Segment[], hint: number | null): number {
  const last = segments.reduce((max, s) => Math.max(max, s.end, s.place?.end ?? 0), 0);
  return Math.max(hint ?? 0, last, 1);
}

/** Verification worth surfacing: a clone that did not say the right words. */
export function verifyConcern(seg: Segment): "none" | "soft" | "bad" {
  if (seg.keep) return "none";
  const overlap = seg.verify?.overlap ?? seg.tts?.overlap;
  if (overlap == null) return "none";
  if (overlap < 0.6) return "bad";
  if (overlap < 0.9) return "soft";
  return "none";
}

/** Placement problems the reviewer should hear before trusting the mix. */
export function placementConcern(seg: Segment): string[] {
  const notes: string[] = [];
  if (!seg.place) return notes;
  if (seg.place.rate > 1.25) notes.push(`sped up ${seg.place.rate.toFixed(2)}x`);
  if (seg.place.rate < 0.85) notes.push(`slowed to ${seg.place.rate.toFixed(2)}x`);
  if (Math.abs(seg.place.drift) > 0.4) notes.push(`drifts ${seg.place.drift.toFixed(2)}s`);
  return notes;
}

export function findAt(segments: Segment[], time: number): Segment | null {
  return segments.find((s) => time >= s.start && time < s.end) ?? null;
}

export function neighbours(
  segments: Segment[],
  uid: string,
): { index: number; prev: Segment | null; next: Segment | null } {
  const index = segments.findIndex((s) => s.uid === uid);
  return {
    index,
    prev: index > 0 ? segments[index - 1] : null,
    next: index >= 0 && index < segments.length - 1 ? segments[index + 1] : null,
  };
}
