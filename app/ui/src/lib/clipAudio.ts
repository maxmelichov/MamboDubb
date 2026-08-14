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
 */

import { useSyncExternalStore } from "react";

let element: HTMLAudioElement | null = null;
let playing: string | null = null;

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
  return element;
}

/** Stop whatever is sounding. Safe to call when nothing is. */
export function stopClip(): void {
  element?.pause();
  if (playing !== null) {
    playing = null;
    emit();
  }
}

/**
 * Play `url`, or stop if it is already the one playing.
 *
 * A null url is a no-op rather than an error: the callers are buttons whose
 * clip may not exist yet, and they render disabled — this is the belt to that
 * pair of braces.
 */
export function toggleClip(url: string | null): void {
  if (!url) return;
  const el = audio();
  if (!el) return;
  if (playing === url) {
    stopClip();
    return;
  }
  el.src = url;
  el.currentTime = 0;
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
