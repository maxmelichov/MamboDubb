/**
 * One transport for the whole editor, whether or not there is a video to play.
 *
 * Before `mix` has run there is no preview.mp4 — and in fixture mode there
 * never is — but the timeline, the playhead and "space to play" still have to
 * work, so the transport falls back to a virtual clock. Same interface either
 * way, which is what keeps the timeline free of "is there a video" branches.
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
    const onLoaded = () => setVirtual(!el.currentSrc);
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
