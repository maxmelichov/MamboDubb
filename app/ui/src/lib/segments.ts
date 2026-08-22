/**
 * Derived segment state. Everything the timeline, the list and the inspector
 * colour or label comes from here, so there is exactly one definition of what
 * "failed" means.
 */

import { Circle, CircleDashed, Diamond, Square, X } from "lucide-react";
import type { Segment } from "./types";

export type SegmentState =
  | "dubbed"
  | "kept"
  | "failed"
  | "unplaced"
  | "unvoiced"
  | "untranslated";

/**
 * The two things the pipeline calls a failure, and it never calls them that.
 *
 * "failed" used to be `tts.verify === "failed"` a value `dubbing/tts.py` does
 * not write and never has: the record's `verify` is `"ok"`, `"accepted"`,
 * `"soft"` or `"keep"`, and nothing else is in the corpus of a real run
 * ("accepted" is a dub the retry ladder settled for it is a concern the
 * overlap already carries, not a failure). So the Failed chip
 * counted zero on a run with real failures in it.
 *
 * What a failure actually looks like is a *keep the pipeline decided against
 * itself*: `tts.run` cannot get a usable clip and falls back to the original
 * audio (`keep_reason="tts_failed"`), or the translator hands back something
 * that is not the target language (`keep_reason="mt_failed"`). Both are stored
 * as `keep=true` with the original-audio slice attached which the UI drew as
 * a calm green "Kept original", indistinguishable from a line the reviewer
 * chose to keep. The pipeline's own losses, reported as resolved.
 *
 * `edit.invalidate` undoes exactly these two reasons when the stage that set
 * them is re-run, which is the other half of the argument: the pipeline itself
 * treats them as "not settled", so the screen must not say they are.
 */
export const FAILURE_REASONS = ["tts_failed", "mt_failed"] as const;

export function pipelineFailed(seg: Segment): boolean {
  return FAILURE_REASONS.includes((seg.keep_reason ?? "") as (typeof FAILURE_REASONS)[number]);
}

/**
 * Invariant 1 (AGENTS.md) says every segment is either dubbed or plays its
 * original audio so "failed" is not a normal outcome, it is the pipeline
 * having lost a segment, and it is the thing the reviewer must see first.
 *
 * Two of the last three used to be one state called "pending", returned by a
 * ternary whose two arms were the same word, and the third had no name at all.
 * They are not one situation and they do not have one fix: a line with no
 * translation needs the *translator*,
 * a line with a translation and no clip needs the *voice*, and a line with a
 * clip and no placement needs neither it needs a *render*. That last one is
 * the ordinary state of a studio segment between a re-voice and the render
 * that lays it down, and calling it "Needs voice" sent the bulk button off to
 * re-synthesize work that was already done.
 */
export function segmentState(seg: Segment): SegmentState {
  if (pipelineFailed(seg)) return "failed";
  if (seg.keep) return "kept";
  if (seg.place) return "dubbed";
  if (seg.tts) return "unplaced";
  return seg.text_en ? "unvoiced" : "untranslated";
}

/**
 * Has this segment been hand-edited?
 *
 * `seg.locked` is an object, and `{}` is truthy so `seg.locked ? …` painted a
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
 * A line that is meant to be dubbed and has nothing in the mix.
 *
 * The three unfinished states, together, because they are one situation from
 * the reviewer's side "this line will be silent in the output" and they are
 * found the same way. It is also the shape a segment is left in by a verdict
 * flip that queued no work: `keep=false`, no `text_en`, no clip, waiting for a
 * job nobody enqueued.
 *
 * They do *not* share a fix, which is why `needsModelWork` exists below: two of
 * them want a model and the third only wants the timeline re-laid.
 */
export function unfinished(seg: Segment): boolean {
  const state = segmentState(seg);
  return state === "untranslated" || state === "unvoiced" || state === "unplaced";
}

/**
 * Would translating and voicing this line change anything?
 *
 * The bulk fix's work set. A synthesized-but-unplaced line answers no: its clip
 * exists and says the right words, and re-voicing it is a minute of model time
 * to arrive back where it started with the placement still missing, because
 * only a render lays it down. It is the ordinary state after a re-voice with
 * the render deferred, and it used to be swept into "Needs voice" and redone.
 */
export function needsModelWork(seg: Segment): boolean {
  return segmentState(seg) !== "unplaced";
}

/**
 * Is there anything here to translate or to say?
 *
 * `segments.fill_uncovered_audible` writes spans with `text: ""` audible
 * stretches the transcript never claimed, kept so the original at least plays.
 * They are keeps with nothing in them, and a bulk "dub these" that swept them up
 * asked the translator to translate an empty string and the voice to say the
 * result. Voicing nothing produces garbage, over the top of audio that was
 * already correct.
 */
export function hasTranscript(seg: Segment): boolean {
  return seg.text.trim().length > 0;
}

/**
 * What a bulk action costs, in the only unit that matters here.
 *
 * The numbers are the ones the selection panel already prints for a single line
 * `~20 s` for a translate, `~1 min` for a voice so a bar that offers to do
 * twenty-seven of them says half an hour rather than "queues behind any running
 * job", which is what "Dub these 27" used to say about thirty minutes of GPU.
 */
const TRANSLATE_SECONDS = 20;
const VOICE_SECONDS = 60;

export function modelCost(work: { translate?: number; voice?: number }): string {
  const seconds = (work.translate ?? 0) * TRANSLATE_SECONDS + (work.voice ?? 0) * VOICE_SECONDS;
  // Never "0 min": the cheapest thing on offer is still a model load, and a
  // quoted zero is the one number a user would read as free.
  return `roughly ${Math.max(1, Math.round(seconds / 60))} min of model time`;
}

/**
 * A kept line carrying a translation the user typed.
 *
 * `edit.set_text` stores it and does *not* reopen the verdict a keep the user
 * or the span decided is not overturned by a subtitle so the line still plays
 * its original audio. True on screen, invisible on screen: the row shows an
 * English sentence under a Hebrew one and nothing says it will never be spoken.
 */
export function subtitleOnly(seg: Segment): boolean {
  const en = (seg.text_en ?? "").trim();
  return Boolean(seg.keep && seg.locked?.text_en && en && en !== seg.text.trim());
}

/**
 * Colour is never the only channel: every state carries a shape and a word.
 *
 * The shape used to be a Unicode glyph `◆ ▣ ✕ ◇ ○` set in the UI font.
 * Two of those (U+25A3 especially) are outside Inter's coverage, so the browser
 * fell back per-character and drew "Keep" with a lumpy filled square that read,
 * seventy-three rows in a row, as a rendering fault. A drawn icon has no
 * coverage question: it is a path, it is `currentColor`, and it is the same
 * shape at 10px and at 14px in both themes.
 *
 * `filled` is the second, non-colour channel *within* the shapes: the two
 * finished states are solid, the three unfinished ones are outlines. So "not
 * done yet" is legible in greyscale, before any hue is involved.
 *
 * The hues are grouped by *what is left to do*, which is the only question the
 * colour is there to answer (see the palette note in App.css):
 *
 *   kept                     resolved the original plays   green
 *   dubbed                   done there is a clip to check  blue
 *   unvoiced, untranslated   waiting on a model               amber
 *   failed                   the pipeline lost the line       red
 *
 * The two waiting states used to be grey, borrowing "unclaimed time"'s hue —
 * which is the colour of *nothing here*, and these are lines with something
 * outstanding. They take the warning hue that "kept" used to wear, and "kept"
 * takes the resolved green, so a run where every line is correctly kept is a
 * calm screen instead of two hundred rows of amber.
 */
/**
 * `short` is the word a script row prints, and it is only allowed to be shorter
 * than `label` where the shortening loses nothing.
 *
 * Three of them lost the sentence. "Voice" and "Text" were abbreviations of
 * "Needs voice" and "Needs translation" that read as nouns a column that says
 * "Text" next to a line of text is saying nothing and "Dub" was the same word
 * as the *verb* on the row's own button, so the state and the action shared a
 * label. The full phrases cost a few pixels on a row that has room for them.
 * "Keep", "Fail" and "Render" stay short: each is unambiguous on its own.
 */
export const STATE_META: Record<
  SegmentState,
  { label: string; short: string; icon: typeof Square; filled: boolean; token: string }
> = {
  dubbed: {
    label: "Dubbed",
    short: "Dubbed",
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
  // Waiting on the *timeline*, not on a model: the clip exists and says the
  // right words, and the only thing between it and the mix is a render. Same
  // pending hue as the other two what is outstanding is still outstanding —
  // and a distinct word, because the button is a different button.
  unplaced: {
    label: "Needs render",
    short: "Render",
    icon: Circle,
    filled: false,
    token: "var(--color-pending)",
  },
  unvoiced: {
    label: "Needs voice",
    short: "Needs voice",
    icon: Circle,
    filled: false,
    token: "var(--color-pending)",
  },
  untranslated: {
    label: "Needs translation",
    short: "Needs translation",
    icon: CircleDashed,
    filled: false,
    token: "var(--color-pending)",
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

/**
 * Verification worth surfacing: a clone that did not say the right words.
 *
 * A keep has nothing to verify its "clip" is a slice of the original so the
 * check short-circuits on it. Except for the two keeps the pipeline decided
 * against itself: a `tts_failed` line's low overlap is *why* it is a keep, and
 * suppressing it hid the evidence for the one verdict a reviewer most needs to
 * check.
 */
export function verifyConcern(seg: Segment): "none" | "soft" | "bad" {
  if (seg.keep && !pipelineFailed(seg)) return "none";
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

/**
 * Why a line keeps its original audio, in words a reviewer can read.
 *
 * The reasons are `keep_reason` as the pipeline writes it, and the UI used to
 * print them raw "Kept because `speaker_en`" while special-casing exactly
 * two of them somewhere else. The one that reads worst is the user's own: a
 * keep made in this app is stored as `manual`, and after a headless re-run the
 * same decision comes back as `user` (`PASSTHROUGH_REASON` one concept, one
 * manifest key), so the app told a user "Kept because user" about a button they
 * had pressed themselves. Both spellings mean the same thing and it is said
 * once, here, so the panel and the run summary cannot drift apart again.
 *
 * An unknown reason falls through to itself. Inventing a phrase for a value
 * this file has never seen would be inventing a vocabulary the run's own report
 * does not use the raw token is at least searchable.
 */
const KEEP_REASONS: Record<string, string> = {
  // The user's verdict, in both spellings the manifest can hold.
  user: "you chose this",
  manual: "you chose this",
  // The pipeline failing, said plainly. These are the two the Failed state is
  // derived from, so the phrase has to agree with the chip that found them.
  tts_failed: "voice failed",
  mt_failed: "translation failed",
  // Nothing to dub.
  foreign: "another language",
  no_text: "no speech",
  interjection: "just a sound",
  uncovered: "nothing was transcribed here",
  // Already the target language, by four different routes.
  latin: "already in the target language",
  target_lang: "already in the target language",
  spoken_target: "already in the target language",
  speaker_en: "this speaker already speaks it",
};

export function keepReason(reason: string | null | undefined): string {
  const key = (reason ?? "").trim();
  return KEEP_REASONS[key] || key || "no reason recorded";
}

/**
 * Does this file have a real phrase for the reason, or is it falling through?
 *
 * The row's meta line prints the reason where it used to print "original audio
 * plays here", so the reviewer judging a keep can see *why* without opening the
 * panel. But the fallthrough is a raw manifest token or "no reason recorded",
 * and neither of those is worth the space the sentence they replaced was
 * earning so those rows keep the sentence.
 */
export function hasKeepPhrase(reason: string | null | undefined): boolean {
  return Boolean(KEEP_REASONS[(reason ?? "").trim()]);
}

/** True when the keep is "this is already the target language". */
export function keptAsTargetLanguage(reason: string | null | undefined): boolean {
  return KEEP_REASONS[(reason ?? "").trim()] === KEEP_REASONS.latin;
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
