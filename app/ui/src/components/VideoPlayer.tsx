/**
 * The preview stage.
 *
 * This used to be the app's one dark island — a black surround with white
 * transport controls, on the reasoning that video wants black around it. That
 * reasoning does not survive contact with either theme: in light the stage sat
 * in the top-left of every editor screen as a slab, and in dark a *second*,
 * differently-black rectangle inside a near-black app reads as a hole. So the
 * surround is the app's own sunken tone, whichever theme that resolves to, and
 * the transport is the house style.
 *
 * It is no longer the pane that grows. The script is. The picture takes the top
 * of a fixed right-hand column, at the video's own aspect ratio, and the
 * transport sits directly under it — which is also why the footer lost the
 * project title (the header already says it, forty pixels above) and the
 * `space` hint (a keyboard hint printed permanently is a legend, and the
 * legends went behind the "?").
 */

import { Pause, Play, SkipBack, SkipForward } from "lucide-react";
import { cn } from "../lib/classNames";
import { timecode } from "../lib/format";
import type { Transport } from "../lib/useTransport";
import type { ReactNode } from "react";

export function VideoPlayer({
  src,
  transport,
  duration,
  placeholder,
  className,
}: {
  src: string | null;
  transport: Transport;
  duration: number;
  /** What to show instead of the picture when there is no preview file yet. */
  placeholder?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex flex-col bg-sunken", className)}>
      <div className="relative min-h-0 flex-1">
        {src ? (
          <video
            ref={transport.videoRef}
            src={src}
            className="h-full w-full object-contain"
            preload="metadata"
          />
        ) : (
          placeholder
        )}
      </div>

      <div className="flex shrink-0 items-center gap-1.5 border-y border-border bg-surface px-3 py-1.5">
        <TransportButton onClick={() => transport.nudge(-5)} label="Back 5 seconds">
          <SkipBack className="h-3.5 w-3.5" />
        </TransportButton>
        <TransportButton
          onClick={transport.toggle}
          label={transport.playing ? "Pause" : "Play"}
          primary
        >
          {transport.playing ? (
            <Pause className="h-4 w-4 fill-current" />
          ) : (
            <Play className="h-4 w-4 fill-current" />
          )}
        </TransportButton>
        <TransportButton onClick={() => transport.nudge(5)} label="Forward 5 seconds">
          <SkipForward className="h-3.5 w-3.5" />
        </TransportButton>

        <span className="ml-auto font-mono text-[12.5px] tabular-nums text-primary">
          {timecode(transport.currentTime)}
          <span className="text-muted"> / {timecode(duration, 0)}</span>
        </span>
      </div>
    </div>
  );
}

function TransportButton({
  onClick,
  label,
  primary,
  children,
}: {
  onClick: () => void;
  label: string;
  primary?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      title={label}
      className={cn(
        "inline-flex h-7 w-7 items-center justify-center rounded-full transition-colors active:scale-[0.96]",
        primary
          ? "bg-primary text-on-primary hover:opacity-90"
          : "text-secondary hover:bg-border/60 hover:text-primary",
      )}
    >
      {children}
    </button>
  );
}
