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
 *
 * ## Three modes, one transport
 *
 * `preview` is the finished thing. `source` is the run's own audio, which
 * `fetch` writes long before `mix` does — it is loaded into the same element
 * with the picture hidden, so play, seek, the timecode, the playhead and the
 * script's follow-along all drive the file rather than a synthetic clock, and
 * the panel keeps showing the run's status board instead of a black rectangle.
 * `none` is a run that has not fetched anything yet: there is nothing to play,
 * so the play button says so and refuses, rather than running a clock over
 * silence — which is what "i can start playing. but because it still not ready
 * it shown nothing" was.
 *
 * The label is only shown in `source`, because it is only true there: a chip
 * that says "original audio" over the finished dub would be worse than no chip.
 */

import { Pause, Play, SkipBack, SkipForward } from "lucide-react";
import { cn } from "../lib/classNames";
import { timecode } from "../lib/format";
import type { Transport } from "../lib/useTransport";
import type { ReactNode } from "react";

/** What the transport is on. EditorPage decides; this draws it. */
export type TransportMode = "preview" | "source" | "none";

/** Said in the chip, and nowhere else — see the note above. */
const ORIGINAL_AUDIO = "Original audio (no preview yet)";

/** Why the play button is dead. Reads as a state of the run, not a failure. */
const NOTHING_TO_PLAY = "Nothing to play yet — the fetch stage hasn't run";

export function VideoPlayer({
  src,
  mode,
  transport,
  duration,
  placeholder,
  className,
}: {
  /** The file in the element: the preview, the original audio, or neither. */
  src: string | null;
  mode: TransportMode;
  transport: Transport;
  duration: number;
  /** What to show instead of the picture when there is no preview file yet. */
  placeholder?: ReactNode;
  className?: string;
}) {
  // The picture needs both a preview and a URL for it. They come apart in
  // fixture mode, where the manifest names a preview.mp4 that no fixture can
  // serve — and the panel is a status board rather than a blank rectangle in
  // exactly that case, which is the case it was written for.
  const picture = mode === "preview" && src != null;
  const silent = mode === "none";

  return (
    <div className={cn("flex flex-col bg-sunken", className)}>
      <div className="relative min-h-0 flex-1">
        {/* One element for both files. Hidden rather than unmounted in `source`
            mode: a display:none media element still plays, and unmounting it
            would hand the transport back its synthetic clock. */}
        {src ? (
          <video
            ref={transport.videoRef}
            src={src}
            data-transport-media={mode}
            className={cn("h-full w-full object-contain", !picture && "hidden")}
            preload="metadata"
          />
        ) : null}
        {picture ? null : placeholder}
      </div>

      <div
        data-transport={mode}
        className="flex shrink-0 items-center gap-1.5 border-y border-border bg-surface px-3 py-1.5"
      >
        <TransportButton onClick={() => transport.nudge(-5)} label="Back 5 seconds">
          <SkipBack className="h-3.5 w-3.5" />
        </TransportButton>
        <TransportButton
          onClick={transport.toggle}
          label={transport.playing ? "Pause" : "Play"}
          title={silent ? NOTHING_TO_PLAY : undefined}
          disabled={silent}
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

        {mode === "source" ? (
          <span
            data-transport-note
            title={ORIGINAL_AUDIO}
            className="min-w-0 truncate rounded-md border border-border px-1.5 py-0.5 text-[11px] text-muted"
          >
            {ORIGINAL_AUDIO}
          </span>
        ) : null}

        <span
          data-timecode
          className="ml-auto shrink-0 font-mono text-[12.5px] tabular-nums text-primary"
        >
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
  title,
  disabled,
  primary,
  children,
}: {
  onClick: () => void;
  label: string;
  /** The tooltip, when it has more to say than the label — a reason, usually. */
  title?: string;
  disabled?: boolean;
  primary?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      title={title ?? label}
      className={cn(
        "inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-full transition-colors active:scale-[0.96]",
        primary
          ? "bg-primary text-on-primary hover:opacity-90"
          : "text-secondary hover:bg-border/60 hover:text-primary",
        // Not `pointer-events-none`: the reason lives in the tooltip, and a
        // control you cannot hover cannot tell you why it is dead.
        "disabled:cursor-not-allowed disabled:opacity-40",
      )}
    >
      {children}
    </button>
  );
}
