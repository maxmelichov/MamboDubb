/**
 * What a `PATCH /segments/{uid}` does to the segment once, for everyone.
 *
 * `dubbing/edit.py` is the truth: every setter both writes a field *and* drops
 * what was built from it, because redo is deletion (`invalidate`). Three copies
 * of that table lived in this app the editor's optimistic merge, the fixture
 * backend's "server", and whatever a component assumed and each one modelled a
 * different subset. The editor knew about `text` and `keep`; the fixtures knew
 * about `text`, `text_en` and `keep`; neither knew that a speaker change drops
 * the clip, that moving the bounds drops it too, or that a language override
 * throws the translation away. A patch nobody modelled left the row showing a
 * ◆ Dubbed badge over audio of a sentence that no longer exists.
 *
 * So this is the table, and both callers import it:
 *
 *   field            drops                          locks        unlocks
 *   text             text_en, text_mid, tts, place  text
 *   text_en          text_mid, tts, place           text_en      tts
 *   keep             text_en, text_mid, tts, place  keep         tts
 *   speaker          tts, place                     speaker      tts
 *   start / end      tts, place                     bounds       tts
 *   src/tgt_lang     text_en, text_mid, tts, place             text_en
 *   tts_opts         tts, place                                tts
 *   locked                                        (replaces every lock)
 *
 * Locks are honoured on the way through, as `invalidate` honours them: a
 * translation the user wrote is not the translator's to discard, so a `text`
 * edit under a `text_en` lock keeps the line and drops only the clip.
 *
 * The result is the *predicted* segment: the server's answer overwrites it a
 * moment later in the editor, and in fixture mode this IS the answer. Nothing
 * here is speculative every line of it is a line of `edit.py`.
 */

import type { Segment, SegmentPatch, TtsOpts } from "./types";

/** The stages a field's edit invalidates, downstream included. */
type Drop = { text_en?: boolean; tts?: boolean; place?: boolean };

const NO_MEDIA = { play: null, tts: null, source: null, source_window: null };

/**
 * `ttsopts.merge`, to the extent this app can see it: an option set to null is
 * removed, and a set that only restates the defaults removes the whole record.
 * The server merges the patch over what is stored and strips the defaults, so a
 * predictor that wrote the raw patch as the whole object would report an
 * override the segment does not have and would drop the seed the user set
 * last week the moment they ticked "greedy".
 */
export function mergeTtsOpts(current: TtsOpts | null | undefined, patch: TtsOpts): TtsOpts | null {
  const merged: Record<string, unknown> = { ...(current ?? {}), ...patch };
  for (const [key, value] of Object.entries(merged)) {
    // null is "reset this to its default"; `greedy: false` and an empty
    // reference path *are* the defaults, and a default is not an override.
    if (value == null || value === false || value === "") delete merged[key];
  }
  return Object.keys(merged).length ? (merged as TtsOpts) : null;
}

/** Normalize a language override the way `edit.set_langs` does. */
const normalizeLang = (value: string): string => value.trim().toLowerCase();

/**
 * The segment as it will be once this patch has been applied.
 *
 * Pure: it never mutates `seg`, so the editor can use it as an optimistic
 * render and the fixture store can assign the result over its own record.
 */
export function applyPatch(seg: Segment, patch: SegmentPatch): Segment {
  const next: Segment = { ...seg };
  const locked: Record<string, boolean> = { ...(seg.locked ?? {}) };
  const drop: Drop = {};

  if (patch.text != null) {
    next.text = patch.text.trim();
    locked.text = true;
    // The translation was made from the line that just changed.
    drop.text_en = true;
    drop.tts = true;
    drop.place = true;
  }
  if (patch.text_en != null) {
    // Invalidate first, then write: `set_text` drops the clip *and* the pivot
    // intermediate before storing the user's line, or this call's own lock
    // would protect the line from the write it is making.
    drop.tts = true;
    drop.place = true;
    delete locked.tts;
    next.text_en = patch.text_en.trim();
    next.text_mid = null;
    locked.text_en = true;
  }
  if (patch.keep != null) {
    next.keep = patch.keep;
    next.keep_reason = patch.keep ? (patch.keep_reason ?? "manual") : null;
    // One concept, one key: a manual keep is also the pipeline's passthrough
    // override, which is how it survives a headless re-run.
    if (!patch.keep_reason || patch.keep_reason === "manual") next.passthrough = patch.keep;
    locked.keep = true;
    delete locked.tts;
    // A verdict flip invalidates translate in both directions: a keep's line is
    // a subtitle and a dub's is what the voice says, so neither survives the
    // other, and the clip is the wrong *kind* either way.
    drop.text_en = true;
    drop.tts = true;
    drop.place = true;
  }
  if (patch.speaker != null) {
    if (patch.speaker !== seg.speaker) {
      next.speaker = patch.speaker;
      delete locked.tts; // the old clip is in the old voice
      drop.tts = true;
      drop.place = true;
    }
    locked.speaker = true;
  }
  if (patch.start != null && patch.end != null) {
    if (patch.start !== seg.start || patch.end !== seg.end) {
      next.start = patch.start;
      next.end = patch.end;
      delete locked.tts; // a keep's clip is cut from the span itself
      drop.tts = true;
      drop.place = true;
    }
    locked.bounds = true;
  }
  if (patch.src_lang != null || patch.tgt_lang != null) {
    let changed = false;
    for (const key of ["src_lang", "tgt_lang"] as const) {
      const raw = patch[key];
      if (raw == null) continue;
      const value = normalizeLang(raw);
      // "" clears the override the only way a language tag can be removed.
      if (!value) changed = (next[key] ?? null) !== null || changed;
      else changed = next[key] !== value || changed;
      next[key] = value || null;
    }
    if (changed) {
      delete locked.text_en; // the locked line is in the wrong language now
      drop.text_en = true;
      drop.tts = true;
      drop.place = true;
    }
  }
  if (patch.tts_opts != null) {
    const merged = mergeTtsOpts(seg.tts_opts, patch.tts_opts);
    if (JSON.stringify(merged ?? {}) !== JSON.stringify(seg.tts_opts ?? {})) {
      next.tts_opts = merged;
      delete locked.tts; // the clip predates these options
      drop.tts = true;
      drop.place = true;
    }
  }

  // --- the invalidation, honouring the locks the patch leaves standing ------
  if (drop.text_en && !locked.text_en) {
    next.text_en = null;
    next.text_mid = null;
  }
  if (drop.tts && !locked.tts) {
    next.tts = null;
    next.verify = null;
    drop.place = true;
  }
  if (drop.place && !locked.place) next.place = null;
  if (next.tts !== seg.tts || next.place !== seg.place) {
    next.media = {
      ...(seg.media ?? NO_MEDIA),
      tts: next.tts ? (seg.media?.tts ?? null) : null,
      play: next.place ? (seg.media?.play ?? null) : null,
    };
  }

  // An explicit `locked` in the patch is the user's final word, applied last —
  // a replace, not a merge, exactly as `edit.set_locked` treats it. `{}` is how
  // "hand this line back to the pipeline" is expressed.
  const final = patch.locked ?? locked;
  const on = Object.fromEntries(Object.entries(final).filter(([, value]) => value));
  next.locked = Object.keys(on).length ? on : null;

  return next;
}
