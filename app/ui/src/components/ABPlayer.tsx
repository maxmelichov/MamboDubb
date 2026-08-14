/**
 * A/B playback for one segment.
 *
 * Judging a dub is a comparison, not an audition, so the two clips share one
 * <audio> element: only one can be playing, and switching sides mid-listen is a
 * single click. "A" is the original span of source audio; "B" is what actually
 * went into the mix (`place.clip`, after time-fitting) — not the raw synthesis,
 * because the time-fit is exactly what can ruin a good clone.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { Pause, Play, Square } from "lucide-react";
import { api } from "../lib/api";
import { cn } from "../lib/classNames";
import { duration as fmtDuration } from "../lib/format";
import { Kbd } from "./ui";
import type { Segment } from "../lib/types";

type Side = "a" | "b";

export function ABPlayer({ seg }: { seg: Segment }) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [side, setSide] = useState<Side | null>(null);
  const [progress, setProgress] = useState(0);

  const sources: Record<Side, string | null> = {
    a: api.audioUrl(seg.source_clip_url),
    b: api.audioUrl(seg.place_clip_url ?? seg.tts_clip_url),
  };

  const stop = useCallback(() => {
    audioRef.current?.pause();
    setSide(null);
    setProgress(0);
  }, []);

  // A new selection must not keep playing the old segment's audio.
  useEffect(() => {
    stop();
  }, [seg.uid, stop]);

  const playSide = useCallback(
    (next: Side) => {
      const url = sources[next];
      const el = audioRef.current;
      if (!url || !el) return;
      if (side === next && !el.paused) {
        el.pause();
        setSide(null);
        return;
      }
      el.src = url;
      el.currentTime = 0;
      setSide(next);
      void el.play().catch(() => setSide(null));
    },
    [side, sources],
  );

  // a / b are the hot keys for the comparison itself.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      const target = event.target as HTMLElement | null;
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
      if (event.key === "a") {
        event.preventDefault();
        playSide("a");
      }
      if (event.key === "b") {
        event.preventDefault();
        playSide("b");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [playSide]);

  const sourceDur = seg.end - seg.start;
  const dubbedDur = seg.place ? seg.place.end - seg.place.start : (seg.tts?.dur ?? null);
  // The number a reviewer is actually after: how much longer the dub runs than
  // the span it has to fit in. It is what the time-fit had to absorb.
  const delta = dubbedDur == null ? null : dubbedDur - sourceDur;

  return (
    <div className="flex flex-col gap-2">
      {/* Naming the axis is the whole affordance: without it "A" and "B" are
          two buttons, with it they are one comparison. */}
      <p className="flex items-baseline gap-1.5 text-[11px] text-muted">
        <span>Same span, both ways</span>
        {delta != null && Math.abs(delta) >= 0.05 ? (
          <span className="ml-auto font-mono tabular-nums">
            B is {delta > 0 ? "+" : "−"}
            {Math.abs(delta).toFixed(2)}s
          </span>
        ) : null}
      </p>

      <div className="flex items-stretch gap-1.5">
        <SideButton
          active={side === "a"}
          disabled={!sources.a}
          onClick={() => playSide("a")}
          badge="A"
          label="Original"
          meta={fmtDuration(sourceDur)}
          hotkey="a"
          title="The source audio for this span, exactly as recorded"
          disabledReason="No source audio for this span."
        />
        <SideButton
          active={side === "b"}
          disabled={!sources.b}
          onClick={() => playSide("b")}
          badge="B"
          label={seg.keep ? "Kept (same audio)" : "Dubbed"}
          meta={dubbedDur != null ? fmtDuration(dubbedDur) : "no clip"}
          hotkey="b"
          title={
            seg.keep
              ? "This segment is kept, so B is the original audio too"
              : "What actually went into the mix, after time-fitting"
          }
          disabledReason="Nothing synthesized yet — re-voice this segment first."
        />
        <button
          type="button"
          onClick={stop}
          disabled={side == null}
          aria-label="Stop playback"
          title="Stop playback"
          className="grid w-9 shrink-0 place-items-center rounded-lg border border-border bg-raised text-muted transition-colors hover:border-axis hover:text-primary disabled:pointer-events-none disabled:opacity-40"
        >
          <Square className="h-3 w-3 fill-current" />
        </button>
      </div>

      <div
        className="h-1 overflow-hidden rounded-full bg-border"
        role="progressbar"
        aria-label="Clip position"
        aria-valuenow={Math.round(progress * 100)}
      >
        <div
          className="h-full rounded-full bg-primary transition-[width] duration-100"
          style={{ width: `${progress * 100}%` }}
        />
      </div>

      <audio
        ref={audioRef}
        onTimeUpdate={(event) => {
          const el = event.currentTarget;
          setProgress(el.duration ? el.currentTime / el.duration : 0);
        }}
        onEnded={() => {
          setSide(null);
          setProgress(0);
        }}
      />
    </div>
  );
}

function SideButton({
  active,
  disabled,
  onClick,
  badge,
  label,
  meta,
  hotkey,
  title,
  disabledReason,
}: {
  active: boolean;
  disabled: boolean;
  onClick: () => void;
  badge: string;
  label: string;
  meta: string;
  hotkey: string;
  title: string;
  /** Why it cannot be played — a dead button that says nothing is a bug. */
  disabledReason: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-pressed={active}
      title={disabled ? disabledReason : title}
      className={cn(
        "flex flex-1 items-center gap-2 rounded-lg border px-2 py-2 text-left transition-all active:scale-[0.99]",
        active
          ? "border-primary bg-primary/[0.06] shadow-card"
          : "border-border bg-raised hover:border-axis",
        // Not `pointer-events-none`: a disabled control that will not even
        // show its own tooltip is the reason people think the app is broken.
        disabled && "cursor-not-allowed opacity-45",
      )}
    >
      <span
        className={cn(
          "grid h-6 w-6 shrink-0 place-items-center rounded-md text-[11px] font-black transition-colors",
          active ? "bg-primary text-on-primary" : "bg-border text-secondary",
        )}
      >
        {badge}
      </span>
      {active ? (
        <Pause className="h-3.5 w-3.5 shrink-0 fill-current" />
      ) : (
        <Play className="h-3.5 w-3.5 shrink-0 fill-current" />
      )}
      <span className="min-w-0 flex-1">
        <span className="block truncate text-[12px] font-semibold">{label}</span>
        <span className="block font-mono text-[10px] tabular-nums text-muted">{meta}</span>
      </span>
      <Kbd>{hotkey}</Kbd>
    </button>
  );
}
