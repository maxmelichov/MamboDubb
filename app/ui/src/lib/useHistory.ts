/**
 * The editor's undo history: what an edit *was*, held so ⌘Z can send its
 * inverse.
 *
 * Not a snapshot store. The manifest lives on the server and jobs rewrite it
 * while the editor is open, so "put the whole segment list back" would undo the
 * user's edit and a finished re-voice with it. Every entry here is instead one
 * more request of a kind the server already accepts — a PATCH carrying the
 * previous field values, a POST re-claiming a deleted span, a DELETE of a line
 * an undo re-created — applied through the same optimistic actions every other
 * gesture goes through. Undo is an edit like any other; the only special thing
 * about it is where its field values come from.
 *
 * Two stacks, standard shape: `done` is what ⌘Z walks, `undone` is what ⌘⇧Z
 * walks, and a fresh edit empties `undone` because the branch it described no
 * longer leads anywhere. The stacks live in a ref, not state — nothing renders
 * from them, and an undo history that re-rendered the editor on every
 * keystroke's push would be paying for a picture nobody is shown.
 *
 * ## What is honestly reversible, and what is not
 *
 * A field edit is: the previous value is right there and PATCH sets it back. A
 * delete is: the whole of what `POST /segments` accepts (start, end, text,
 * speaker) is captured before the DELETE, and re-creating the span is exactly
 * the add composer's request. An add is: the minted uid is deletable.
 *
 * A split or a merge is NOT. The server mints new uids and renumbers the run,
 * and there is no inverse request — "merge these halves back" is itself a
 * merge, with its own new uids and its own information loss. Skipping them
 * silently would be worse than refusing: every entry beneath one describes a
 * run that no longer exists in that shape, so an undo that hopped over the
 * split would patch old values onto the wrong lines. So they go on the stack as
 * {@link BarrierEntry} — a full stop that, when reached, says why instead of
 * doing something wrong. The barrier is never popped; history before it is
 * simply over, the same way it would be after closing the tab.
 *
 * ## Staleness
 *
 * The stack can outlive its own truth: a translate job, another window, or a
 * re-run may rewrite a line after the entry describing it was pushed. Each
 * PATCH entry therefore carries `expect` — the values its fields held right
 * after the edit it inverts, straight from the server's own response — and
 * {@link matches} checks them against the live segment at undo time. A
 * mismatch means the edit being undone is no longer the newest fact about
 * those fields, and undoing it would overwrite whatever *is*; the entry is
 * dropped with a notice instead. uid + fields, not deep equality of the whole
 * segment: a re-voice changes `tts` without touching the text, and undo of a
 * text edit is still valid across it.
 */

import type { NewSegment, Segment, SegmentPatch } from "./types";

/** Undo depth. Enough for a session of judging; small enough to never matter. */
export const HISTORY_LIMIT = 50;

/**
 * The fields whose previous value a PATCH can state. `tts_opts` and `locked`
 * are deliberately absent: their patches are merged server-side over state this
 * client cannot fully see (`mergeTtsOpts` models the half it can), so an
 * "inverse" would be a guess — and neither invalidates the fields other
 * entries target, so leaving them unrecorded strands nothing beneath them.
 */
const INVERTIBLE = [
  "text",
  "text_en",
  "keep",
  "speaker",
  "start",
  "end",
  "src_lang",
  "tgt_lang",
] as const;

type InvertibleKey = (typeof INVERTIBLE)[number];

export type PatchEntry = {
  kind: "patch";
  uid: string;
  /** the inverse request, ready to send */
  body: SegmentPatch;
  /**
   * What `body`'s fields held immediately *after* the edit this entry inverts,
   * from the server's own response — the staleness check compares the live
   * segment against these at undo time.
   */
  expect: Partial<Record<InvertibleKey, unknown>>;
};

/** Undo of a delete: claim the span again, exactly as the composer would. */
export type CreateEntry = { kind: "create"; segment: NewSegment };

/** Undo of an add (or redo of a delete): retire the line again. */
export type DeleteEntry = { kind: "delete"; uid: string };

/**
 * A step with no inverse. Reached, it explains itself and stays put — see the
 * module comment for why skipping it would corrupt everything beneath.
 */
export type BarrierEntry = { kind: "barrier"; op: "split" | "merge" | "edit" };

export type HistoryEntry = PatchEntry | CreateEntry | DeleteEntry | BarrierEntry;

export type HistoryStacks = { done: HistoryEntry[]; undone: HistoryEntry[] };

export function emptyHistory(): HistoryStacks {
  return { done: [], undone: [] };
}

/**
 * A new edit goes on `done`, empties `undone` (the redone branch is gone), and
 * evicts the oldest entry past the cap. Null is "nothing worth remembering" —
 * a tts_opts tweak — and costs nothing.
 */
export function remember(stacks: HistoryStacks, entry: HistoryEntry | null): void {
  if (!entry) return;
  stacks.undone.length = 0;
  stacks.done.push(entry);
  if (stacks.done.length > HISTORY_LIMIT) {
    stacks.done.splice(0, stacks.done.length - HISTORY_LIMIT);
  }
}

/**
 * Build the entry that inverts `body`, from the segment as it was (`before`)
 * and as the server answered (`after`).
 *
 * This is also how an applied entry mints its own inverse for the other stack:
 * undoing a PATCH is a PATCH, so capturing *that* against the segment it hit
 * yields the redo, with no second code path to disagree.
 *
 * Two rules beyond straight field copying, both consequences of
 * `dubbing/edit.py`'s invalidation table (see `patch.ts`):
 *
 * - An edit to `text`, `keep`, or a language override drops the translation on
 *   the way through, so the inverse carries `text_en` back too — except when
 *   the inverse itself sets keep:true, which would drop it again anyway.
 * - A `text_en` whose previous value was empty cannot come back: PATCH has no
 *   null, and the server refuses an empty string outright. That one edit is a
 *   barrier — said, not skipped, for the same reason splits are.
 */
export function capture(
  before: Segment,
  after: Segment,
  body: SegmentPatch,
): HistoryEntry | null {
  const keys = INVERTIBLE.filter((key) => body[key] !== undefined);
  if (keys.length === 0) return null;

  const inverse: SegmentPatch = {};
  for (const key of keys) {
    switch (key) {
      case "text":
        inverse.text = before.text;
        break;
      case "text_en": {
        if (!(before.text_en ?? "").trim()) return { kind: "barrier", op: "edit" };
        inverse.text_en = before.text_en!;
        break;
      }
      case "keep":
        inverse.keep = before.keep;
        // Only read alongside keep. A pipeline reason ("no_speech") survives
        // the round trip this way; absent one, the server stamps "manual",
        // which is true — the undo is the user's verdict.
        if (before.keep && before.keep_reason) inverse.keep_reason = before.keep_reason;
        break;
      case "speaker":
        inverse.speaker = before.speaker;
        break;
      case "start":
        inverse.start = before.start;
        break;
      case "end":
        inverse.end = before.end;
        break;
      case "src_lang":
        // "" clears the override — the one field where empty is the inverse.
        inverse.src_lang = before.src_lang ?? "";
        break;
      case "tgt_lang":
        inverse.tgt_lang = before.tgt_lang ?? "";
        break;
    }
  }

  const dropsTranslation = keys.some(
    (key) => key === "text" || key === "keep" || key === "src_lang" || key === "tgt_lang",
  );
  if (
    dropsTranslation &&
    inverse.keep !== true &&
    inverse.text_en === undefined &&
    (before.text_en ?? "").trim()
  ) {
    inverse.text_en = before.text_en!;
  }

  const expect: PatchEntry["expect"] = {};
  for (const key of Object.keys(inverse)) {
    if (key === "keep_reason") continue; // rides with `keep`; compared through it
    expect[key as InvertibleKey] = after[key as InvertibleKey] ?? null;
  }
  return { kind: "patch", uid: before.uid, body: inverse, expect };
}

/**
 * Is the live segment still the one this entry knows how to invert?
 *
 * null and undefined are the same absence (the wire drops absent keys), and
 * numbers get an epsilon because the fixture backend and the server round
 * differently — a retime undo must not be declared stale by a float.
 */
export function matches(seg: Segment, expect: PatchEntry["expect"]): boolean {
  return Object.entries(expect).every(([key, want]) => {
    const have = seg[key as InvertibleKey] ?? null;
    const wanted = want ?? null;
    if (typeof have === "number" && typeof wanted === "number") {
      return Math.abs(have - wanted) < 1e-6;
    }
    return have === wanted;
  });
}
