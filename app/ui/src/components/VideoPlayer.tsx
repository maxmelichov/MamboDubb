/**
 * The preview stage.
 *
 * The one place in the app that is deliberately not on the warm neutral ramp:
 * video wants a black surround, and the transport that hangs off it inherits
 * that rather than the theme. Everything else — radii, type sizes, the Kbd
 * chip — is the house style, so it still reads as the same product.
 */

import { Pause, Play, SkipBack, SkipForward } from "lucide-react";
import { cn } from "../lib/classNames";
import { timecode } from "../lib/format";
import type { Transport } from "../lib/useTransport";

export function VideoPlayer({
  src,
  transport,
  duration,
  title,
}: {
  src: string | null;
  transport: Transport;
  duration: number;
  title: string;
}) {
  return (
    <div className="flex h-full flex-col bg-[#0a0a0a]">
      <div className="relative min-h-0 flex-1">
        {src ? (
          <video
            ref={transport.videoRef}
            src={src}
            className="h-full w-full object-contain"
            preload="metadata"
          />
        ) : (
          <div className="grid h-full place-items-center px-6 text-center text-[12px] leading-relaxed text-white/50">
            <div className="max-w-md">
              <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-white/70">
                No preview yet
              </p>
              <p className="mt-2">
                <code className="font-mono text-white/70">preview.mp4</code> appears after{" "}
                <code className="font-mono text-white/70">mix</code> runs. The transport below is
                a virtual clock, so the timeline still scrubs.
              </p>
            </div>
          </div>
        )}
      </div>

      <div className="flex shrink-0 items-center gap-1.5 border-t border-white/10 bg-black px-3 py-2 text-white">
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

        <span className="ml-2 font-mono text-[12px] tabular-nums text-white/75">
          {timecode(transport.currentTime)}
          <span className="text-white/35"> / {timecode(duration, 0)}</span>
        </span>

        <span className="ml-auto flex min-w-0 items-center gap-2 text-[11px] text-white/45">
          <span className="inline-flex h-4 min-w-4 items-center justify-center rounded border border-white/20 px-1 text-[10px] font-semibold text-white/60">
            space
          </span>
          <span className="truncate">{title}</span>
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
      className={cn(
        "inline-flex items-center justify-center rounded-full transition-colors active:scale-[0.96]",
        primary
          ? "h-8 w-8 bg-white text-black hover:bg-white/85"
          : "h-8 w-8 text-white/70 hover:bg-white/10 hover:text-white",
      )}
    >
      {children}
    </button>
  );
}
