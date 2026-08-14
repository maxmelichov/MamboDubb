/**
 * The preview stage.
 *
 * This used to be the app's one dark island — a black surround with white
 * transport controls, on the reasoning that video wants black around it. That
 * reasoning does not survive contact with either theme: in light the stage sat
 * in the top-left of every editor screen as a slab, and in dark a *second*,
 * differently-black rectangle inside a near-black app reads as a hole. Either
 * way the "no preview yet" state was a black rectangle, which is the single
 * worst thing a screen can show a user who is waiting for something.
 *
 * So the surround is the app's own sunken tone, whichever theme that resolves
 * to, and the transport is the house style. The picture keeps its own edges —
 * `object-contain` on a neutral ground, with a hairline frame — which is all
 * the separation a video actually needs; the letterbox bars are the stage, not
 * the video.
 */

import { Pause, Play, SkipBack, SkipForward } from "lucide-react";
import { cn } from "../lib/classNames";
import { timecode } from "../lib/format";
import { Kbd } from "./ui";
import type { Transport } from "../lib/useTransport";
import type { ReactNode } from "react";

export function VideoPlayer({
  src,
  transport,
  duration,
  title,
  placeholder,
}: {
  src: string | null;
  transport: Transport;
  duration: number;
  title: string;
  /** What to show instead of the picture when there is no preview file yet. */
  placeholder?: ReactNode;
}) {
  return (
    <div className="flex h-full flex-col bg-sunken">
      <div className="relative min-h-0 flex-1">
        {src ? (
          <video
            ref={transport.videoRef}
            src={src}
            className="h-full w-full object-contain"
            preload="metadata"
          />
        ) : (
          placeholder ?? <NoPreview />
        )}
      </div>

      <div className="flex shrink-0 items-center gap-1.5 border-t border-border bg-surface px-3 py-2">
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

        <span className="ml-2 font-mono text-[12px] tabular-nums text-primary">
          {timecode(transport.currentTime)}
          <span className="text-muted"> / {timecode(duration, 0)}</span>
        </span>

        <span className="ml-auto flex min-w-0 items-center gap-2 text-[11px] text-muted">
          <Kbd>space</Kbd>
          <span className="truncate" title={title}>
            {title}
          </span>
        </span>
      </div>
    </div>
  );
}

/** The fallback when the editor did not hand us a run-aware placeholder. */
function NoPreview() {
  return (
    <div className="grid h-full place-items-center px-6 text-center text-[12px] leading-relaxed text-muted">
      <div className="max-w-md">
        <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-secondary">
          No preview yet
        </p>
        <p className="mt-2">
          <code className="font-mono text-secondary">preview.mp4</code> appears after{" "}
          <code className="font-mono text-secondary">mix</code> runs. The transport below is a
          virtual clock, so the timeline still scrubs.
        </p>
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
        "inline-flex h-8 w-8 items-center justify-center rounded-full transition-colors active:scale-[0.96]",
        primary
          ? "bg-primary text-on-primary hover:opacity-90"
          : "text-secondary hover:bg-border/60 hover:text-primary",
      )}
    >
      {children}
    </button>
  );
}
