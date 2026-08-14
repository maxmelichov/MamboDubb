import { Pause, Play, SkipBack, SkipForward } from "lucide-react";
import { timecode } from "../lib/format";
import type { Transport } from "../lib/useTransport";
import { Button, Kbd } from "./ui";

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
    <div className="flex h-full flex-col bg-black/90">
      <div className="relative min-h-0 flex-1">
        {src ? (
          <video
            ref={transport.videoRef}
            src={src}
            className="h-full w-full object-contain"
            preload="metadata"
          />
        ) : (
          <div className="grid h-full place-items-center px-6 text-center text-[13px] text-white/55">
            <div>
              <p className="font-medium text-white/80">No preview yet</p>
              <p className="mt-1">
                <code>preview.mp4</code> appears after <code>mix</code> runs. The transport below
                is a virtual clock, so the timeline still scrubs.
              </p>
            </div>
          </div>
        )}
      </div>

      <div className="flex items-center gap-2 border-t border-white/10 bg-black px-2 py-1.5 text-white">
        <Button
          variant="ghost"
          className="text-white hover:bg-white/10"
          onClick={() => transport.nudge(-5)}
          aria-label="Back 5 seconds"
        >
          <SkipBack className="h-4 w-4" />
        </Button>
        <Button
          variant="ghost"
          className="text-white hover:bg-white/10"
          onClick={transport.toggle}
          aria-label={transport.playing ? "Pause" : "Play"}
        >
          {transport.playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
        </Button>
        <Button
          variant="ghost"
          className="text-white hover:bg-white/10"
          onClick={() => transport.nudge(5)}
          aria-label="Forward 5 seconds"
        >
          <SkipForward className="h-4 w-4" />
        </Button>

        <span className="ml-1 text-[12px] tabular-nums text-white/70">
          {timecode(transport.currentTime)} / {timecode(duration, 0)}
        </span>

        <span className="ml-auto flex items-center gap-1.5 truncate text-[11px] text-white/45">
          <Kbd>space</Kbd>
          <span className="truncate">{title}</span>
        </span>
      </div>
    </div>
  );
}
