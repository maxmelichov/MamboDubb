/**
 * One `<audio>` element for the whole app.
 *
 * Judging a dub is a comparison, not an audition: A and B have to be mutually
 * exclusive, and switching sides mid-listen has to be one click. The old
 * ABPlayer got that right for one segment by sharing a single element inside
 * itself — but the A/B buttons now live on two hundred script rows, and two
 * hundred `<audio>` elements each holding their own idea of "playing" is two
 * hundred things that can talk over each other.
 *
 * So the element is a module singleton and the playing clip is module state.
 * Starting anything stops everything else by construction, which is the actual
 * requirement; components subscribe with `useClipPlayback()` and get back the
 * one URL that is sounding.
 *
 * Created lazily rather than at import: `new Audio()` at module scope would run
 * during SSR-less-but-still-headless test boots before jsdom's stubs are in.
 *
 * ## The URL that plays is not the URL the caller holds
 *
 * A segment's `media.play` / `media.source` is what the *server* calls the clip,
 * and it is not always something `<audio src>` can load: fixture mode hands out
 * `fixture:tone?hz=…`, which a browser answers with ERR_UNKNOWN_URL_SCHEME.
 * `api.audioUrl` is the seam that turns one into the other — and it had no
 * callers at all, because every A/B path (the row buttons, the `a`/`b` keys)
 * came straight here and assigned the raw string. So every clip button in
 * fixture mode fired a network error and snapped back out of its pressed state.
 *
 * Resolving here rather than at each call site is what makes that unrepeatable:
 * there is one assignment to `element.src` in the app and it goes through the
 * seam. `playing` deliberately stays the caller's URL — the A/B buttons compare
 * it against `seg.media.*` to decide which side is lit.
 *
 * ## A is a window, not a file
 *
 * The dub side is a clip file: it starts where it starts and ends where it
 * ends. The source side is not — it is one segment's *window* of the whole
 * source track, which the server names with a `#t=start,end` media fragment and
 * also states outright as `media.source_window`. Neither of them was used. The
 * element got `el.src = url` and then `el.currentTime = 0`, which fights the
 * fragment's own seek and wins often enough to start A at 0:00; and nothing
 * anywhere enforced the fragment's *end*, which browsers honour inconsistently
 * and jsdom not at all. Pressing A on segment 140 could play the top of the
 * video and then keep going.
 *
 * So the window is enforced here, from the numbers: seek to its start once the
 * metadata says seeking is possible, and pause at its end on `timeupdate`. The
 * fragment stays on the URL and is now merely a hint.
 */

import { useSyncExternalStore } from "react";
import { api } from "./api";

let element: HTMLAudioElement | null = null;
let playing: string | null = null;
/** The window `playing` is confined to, or null for a whole-file clip. */
let window_: [number, number] | null = null;

const listeners = new Set<() => void>();

function emit(): void {
  for (const listener of listeners) listener();
}

function audio(): HTMLAudioElement | null {
  if (element) return element;
  if (typeof Audio === "undefined") return null;
  element = new Audio();
  element.addEventListener("ended", () => {
    playing = null;
    emit();
  });
  element.addEventListener("error", () => {
    playing = null;
    emit();
  });
  // The window's two edges. Registered once, on the one element, and read from
  // module state — a listener per press would leak one per press.
  element.addEventListener("loadedmetadata", () => {
    if (window_ && element) element.currentTime = window_[0];
  });
  element.addEventListener("timeupdate", () => {
    if (window_ && element && element.currentTime >= window_[1]) stopClip();
  });
  return element;
}

/** Stop whatever is sounding. Safe to call when nothing is. */
export function stopClip(): void {
  element?.pause();
  window_ = null;
  if (playing !== null) {
    playing = null;
    emit();
  }
}

/**
 * Play `url`, or stop if it is already the one playing.
 *
 * `window` is the span of the file this clip *is* — `media.source_window` for
 * the source side, absent for a clip file that is already exactly the segment.
 *
 * A null url is a no-op rather than an error: the callers are buttons whose
 * clip may not exist yet, and they render disabled — this is the belt to that
 * pair of braces.
 */
export function toggleClip(url: string | null, window?: [number, number] | null): void {
  if (!url) return;
  const el = audio();
  if (!el) return;
  if (playing === url) {
    stopClip();
    return;
  }
  // The seam: the caller's URL is the server's name for the clip, `src` is the
  // one a browser can actually fetch. See the note at the top of this file.
  const src = api.audioUrl(url);
  if (!src) return;
  window_ = window && window[1] > window[0] ? window : null;
  el.src = src;
  // The window's start, not zero: `src` may already be seekable (the same file
  // as last time), in which case no `loadedmetadata` is coming to do it.
  el.currentTime = window_ ? window_[0] : 0;
  playing = url;
  emit();
  const started = el.play();
  // jsdom's stub returns undefined; a real refusal (no gesture, bad codec)
  // must not leave a button stuck in the pressed state.
  if (started && typeof started.catch === "function") {
    void started.catch(() => {
      if (playing === url) {
        playing = null;
        emit();
      }
    });
  }
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

const snapshot = (): string | null => playing;

/** The URL currently sounding, or null. Re-renders the subscriber when it changes. */
export function useClipPlayback(): string | null {
  return useSyncExternalStore(subscribe, snapshot, snapshot);
}
