/**
 * Derived segment state. Everything the timeline, the list and the inspector
 * colour or label comes from here, so there is exactly one definition of what
 * "failed" means.
 */

import type { Segment } from "./types";

export type SegmentState = "dubbed" | "kept" | "failed" | "unvoiced" | "untranslated";

/**
 * Invariant 1 (AGENTS.md) says every segment is either dubbed or plays its
 * original audio — so "failed" is not a normal outcome, it is the pipeline
 * having lost a segment, and it is the thing the reviewer must see first.
 *
 * The last two used to be one state called "pending", returned by a ternary
 * whose two arms were the same word. They are not the same situation and they
 * do not have the same fix: a line with a translation and no clip needs the
 * *voice* run over it, and a line with no translation needs the *translator*.
 * Telling a reviewer "Wait" for both is the row refusing to say which button
 * to press.
 */
export function segmentState(seg: Segment): SegmentState {
  if (seg.tts?.verify === "failed") return "failed";
  if (seg.keep) return "kept";
  if (seg.place) return "dubbed";
  return seg.text_en ? "unvoiced" : "untranslated";
}

/**
 * Has this segment been hand-edited?
 *
 * `seg.locked` is an object, and `{}` is truthy — so `seg.locked ? …` painted a
 * padlock on every segment the server had ever sent an empty lock map for. The
 * question is always "are any locks *on*", never "is there a lock map".
 */
export function lockedFields(seg: Segment): string[] {
  return Object.entries(seg.locked ?? {})
    .filter(([, on]) => on)
    .map(([field]) => field);
}

export function hasLocks(seg: Segment): boolean {
  return lockedFields(seg).length > 0;
}

/**
 * Colour is never the only channel: every state carries a glyph and a word.
 *
 * `onFill` is which ink the glyph takes when it is drawn *inside* the fill, on
 * a timeline mark. It is not decoration: white on the amber keep-fill is
 * 2.1:1, which is the glyph — the accessible channel — being illegible on the
 * one state where it matters most.
 *
 * These are literals rather than tokens because they are chosen against the
 * *fill*, not against the page, and both themes' fills are close enough in
 * lightness that one choice serves both: on the dark triple white sits at
 * 3.6:1 over the blue and 4.8:1 over the red, and near-black at 6.1:1 over the
 * amber. If a state hue ever moves, re-check these three, not the tokens.
 */
export const STATE_META: Record<
  SegmentState,
  { label: string; short: string; glyph: string; token: string; onFill: string }
> = {
  dubbed: {
    label: "Dubbed",
    short: "Dub",
    glyph: "◆",
    token: "var(--color-dubbed)",
    onFill: "#ffffff",
  },
  kept: {
    label: "Kept original",
    short: "Keep",
    glyph: "▣",
    token: "var(--color-kept)",
    onFill: "#14130f",
  },
  failed: {
    label: "Failed",
    short: "Fail",
    glyph: "✕",
    token: "var(--color-failed)",
    onFill: "#ffffff",
  },
  unvoiced: {
    label: "Needs voice",
    short: "Voice",
    glyph: "◇",
    token: "var(--color-unclaimed)",
    onFill: "#ffffff",
  },
  untranslated: {
    label: "Needs translation",
    short: "Text",
    glyph: "○",
    token: "var(--color-unclaimed)",
    onFill: "#ffffff",
  },
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
