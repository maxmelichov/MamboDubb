/**
 * One transport for the whole editor, whether or not there is a video to play.
 *
 * Before `mix` has run there is no preview.mp4 — and in fixture mode there
 * never is — but the timeline, the playhead and "space to play" still have to
 * work, so the transport falls back to a virtual clock. Same interface either
 * way, which is what keeps the timeline free of "is there a video" branches.
 *
 * ## One element, whatever is in it
 *
 * A run that has not reached `mix` still has `source.wav` on disk from `fetch`,
 * and that is what the transport plays until the preview exists (see
 * EditorPage's transport mode). It is handed to the *same* media element — a
 * `<video>` with a wav in it is an audio player that happens to draw nothing —
 * so there is exactly one clock, one set of listeners and one `hasMedia()` here
 * rather than a parallel audio path that would drift out of step with this one.
 * It also leaves the A/B clip player (lib/clipAudio.ts, its own `<audio>`
 * singleton) exactly as it was: the transport has never stopped a clip and a
 * clip has never stopped the transport, and swapping what is inside this
 * element does not change that.
 */

import { useCallback, useEffect, useRef, useState } from "react";

export type Transport = {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  currentTime: number;
  playing: boolean;
  /** true when the clock is synthetic because no media is attached */
  virtual: boolean;
  play: () => void;
  pause: () => void;
  toggle: () => void;
  seek: (time: number) => void;
  nudge: (delta: number) => void;
};

export function useTransport(duration: number): Transport {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [virtual, setVirtual] = useState(true);

  const rafRef = useRef(0);
  const clockRef = useRef({ base: 0, since: 0 });
  const durationRef = useRef(duration);
  durationRef.current = duration;
  // Where the transport is, readable from an event handler without making the
  // listener effect depend on it. Same reason `durationRef` exists.
  const timeRef = useRef(0);
  timeRef.current = currentTime;

  const hasMedia = useCallback(() => {
    const el = videoRef.current;
    return Boolean(el && el.currentSrc);
  }, []);

  // Mirror the media element's own clock when there is one.
  useEffect(() => {
    const el = videoRef.current;
    if (!el) return;
    const onTime = () => setCurrentTime(el.currentTime);
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    /**
     * A new source starts at zero. The transport does not.
     *
     * Two moments land here: `source.wav` replaced by `preview.mp4` when a
     * render finishes, and the first file arriving under a clock that has been
     * synthetic until now. Both must leave the playhead where the reviewer put
     * it — being thrown back to 00:00 by a background job is the same bug as
     * having nothing to play, one screen later. Clamped to whichever length is
     * known, because the new file may be shorter than the old one.
     */
    const onLoaded = () => {
      setVirtual(!el.currentSrc);
      const limit = Number.isFinite(el.duration) ? el.duration : durationRef.current;
      const want = Math.min(timeRef.current, limit);
      if (want > 0.05 && Math.abs(el.currentTime - want) > 0.05) {
        el.currentTime = want;
        setCurrentTime(want);
      }
      // A synthetic clock was standing in for this file — either because the
      // element had not picked its source up yet when play was pressed, or
      // because there was no file at all until now. Hand playback over rather
      // than leave two clocks running or drop the playback that was asked for.
      if (rafRef.current) {
        stopClock();
        void el.play().catch(() => setPlaying(false));
      }
    };
    el.addEventListener("timeupdate", onTime);
    el.addEventListener("seeked", onTime);
    el.addEventListener("play", onPlay);
    el.addEventListener("pause", onPause);
    el.addEventListener("loadedmetadata", onLoaded);
    setVirtual(!el.currentSrc);
    return () => {
      el.removeEventListener("timeupdate", onTime);
      el.removeEventListener("seeked", onTime);
      el.removeEventListener("play", onPlay);
      el.removeEventListener("pause", onPause);
      el.removeEventListener("loadedmetadata", onLoaded);
    };
  });

  const stopClock = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    rafRef.current = 0;
  }, []);

  const startClock = useCallback(
    (from: number) => {
      clockRef.current = { base: from, since: performance.now() };
      const tick = () => {
        const elapsed = (performance.now() - clockRef.current.since) / 1000;
        const next = clockRef.current.base + elapsed;
        if (next >= durationRef.current) {
          setCurrentTime(durationRef.current);
          setPlaying(false);
          stopClock();
          return;
        }
        setCurrentTime(next);
        rafRef.current = requestAnimationFrame(tick);
      };
      rafRef.current = requestAnimationFrame(tick);
    },
    [stopClock],
  );

  useEffect(() => stopClock, [stopClock]);

  const play = useCallback(() => {
    if (hasMedia()) {
      void videoRef.current?.play().catch(() => setPlaying(false));
      return;
    }
    setPlaying(true);
    startClock(currentTime >= durationRef.current ? 0 : currentTime);
  }, [currentTime, hasMedia, startClock]);

  const pause = useCallback(() => {
    if (hasMedia()) {
      videoRef.current?.pause();
      return;
    }
    stopClock();
    setPlaying(false);
  }, [hasMedia, stopClock]);

  const toggle = useCallback(() => (playing ? pause() : play()), [pause, play, playing]);

  const seek = useCallback(
    (time: number) => {
      const clamped = Math.max(0, Math.min(time, durationRef.current));
      if (hasMedia() && videoRef.current) {
        videoRef.current.currentTime = clamped;
        setCurrentTime(clamped);
        return;
      }
      setCurrentTime(clamped);
      if (rafRef.current) startClock(clamped);
    },
    [hasMedia, startClock],
  );

  const nudge = useCallback((delta: number) => seek(currentTime + delta), [currentTime, seek]);

  return { videoRef, currentTime, playing, virtual, play, pause, toggle, seek, nudge };
}
