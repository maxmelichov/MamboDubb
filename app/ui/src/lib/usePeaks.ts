/**
 * The waveform behind a timeline lane.
 *
 * One request per lane per run, and deliberately not part of `useProject`: the
 * picture is not state the editor edits, nothing upstream depends on it, and a
 * lane that never gets one still works. Keeping it out of the store is what
 * lets a 404 `dub.wav` does not exist until the mix stage has run be
 * nothing at all rather than an error banner over a working editor.
 *
 * `revision` is the caller's "go and look again" signal. The OUTPUT lane passes
 * the count of finished jobs, because a re-voice or a render is precisely when
 * the mix on disk stops matching the picture on screen; the SOURCE lane passes
 * nothing, because `source.wav` is written once by the fetch stage and never
 * again.
 */

import { useEffect, useState } from "react";
import { api } from "./api";
import type { Peaks, PeaksFile } from "./types";

/**
 * Buckets to ask for, from a pixel width: about two values per pixel, which is
 * one per physical pixel on the retina displays this app is used on, inside the
 * server's own [16, 4000] clamp.
 */
export function bucketsFor(pixelWidth: number): number {
  return Math.max(16, Math.min(4000, Math.round(pixelWidth * 2)));
}

export function usePeaks(
  name: string,
  file: PeaksFile,
  buckets: number,
  revision = 0,
): Peaks | null {
  const [peaks, setPeaks] = useState<Peaks | null>(null);

  useEffect(() => {
    let live = true;
    void (async () => {
      try {
        const next = await api.peaks(name, file, buckets);
        if (live) setPeaks(next);
      } catch {
        // A lane with no waveform is the fallback, and it is a good one: the
        // marks are still there and still clickable. Nothing about a picture
        // that failed to load is worth interrupting a reviewer for.
        if (live) setPeaks(null);
      }
    })();
    return () => {
      live = false;
    };
  }, [buckets, file, name, revision]);

  return peaks;
}
