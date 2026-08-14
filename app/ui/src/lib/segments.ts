/**
 * Derived segment state. Everything the timeline, the list and the inspector
 * colour or label comes from here, so there is exactly one definition of what
 * "failed" means.
 */

import { Circle, CircleDashed, Diamond, Square, X } from "lucide-react";
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
 * Colour is never the only channel: every state carries a shape and a word.
 *
 * The shape used to be a Unicode glyph — `◆ ▣ ✕ ◇ ○` — set in the UI font.
 * Two of those (U+25A3 especially) are outside Inter's coverage, so the browser
 * fell back per-character and drew "Keep" with a lumpy filled square that read,
 * seventy-three rows in a row, as a rendering fault. A drawn icon has no
 * coverage question: it is a path, it is `currentColor`, and it is the same
 * shape at 10px and at 14px in both themes.
 *
 * `filled` is the second, non-colour channel *within* the shapes: the two
 * finished states are solid, the three unfinished ones are outlines. So "not
 * done yet" is legible in greyscale, before any hue is involved.
 */
export const STATE_META: Record<
  SegmentState,
  { label: string; short: string; icon: typeof Square; filled: boolean; token: string }
> = {
  dubbed: {
    label: "Dubbed",
    short: "Dub",
    icon: Diamond,
    filled: true,
    token: "var(--color-dubbed)",
  },
  kept: {
    label: "Kept original",
    short: "Keep",
    icon: Square,
    filled: true,
    token: "var(--color-kept)",
  },
  failed: {
    label: "Failed",
    short: "Fail",
    icon: X,
    filled: false,
    token: "var(--color-failed)",
  },
  unvoiced: {
    label: "Needs voice",
    short: "Voice",
    icon: Circle,
    filled: false,
    token: "var(--color-unclaimed)",
  },
  untranslated: {
    label: "Needs translation",
    short: "Text",
    icon: CircleDashed,
    filled: false,
    token: "var(--color-unclaimed)",
  },
};

export const UNCLAIMED_META = {
  label: "Unclaimed time",
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
