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

import { useEffect, useState } from "react";
import { Pause, Play, SkipBack, SkipForward, TriangleAlert } from "lucide-react";
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

/**
 * Why the play button is dead when the run *says* it has a file.
 *
 * The manifest naming `preview.mp4` and the browser being able to load it are
 * two different facts, and they come apart routinely: fixture mode, a file
 * deleted out from under a run, a mid-render truncated mp4. "The fetch stage
 * hasn't run" is the wrong sentence for every one of those.
 */
const NO_MEDIA = "This run's video could not be loaded — there is nothing to play";

export function VideoPlayer({
  src,
  srcLabel,
  mode,
  transport,
  duration,
  placeholder,
  className,
}: {
  /** The file in the element: the preview, the original audio, or neither. */
  src: string | null;
  /** What that file is called, for the sentence when it will not load. */
  srcLabel?: string | null;
  mode: TransportMode;
  transport: Transport;
  duration: number;
  /** What to show instead of the picture when there is no preview file yet. */
  placeholder?: ReactNode;
  className?: string;
}) {
  /*
   * The element said it could not load the file.
   *
   * Without this, a `<video>` whose `src` 404s is a black rectangle with a live
   * transport under it — the app claiming to be playing a video it does not
   * have. The state is keyed to the source: a render finishing swaps the URL,
   * and the new file deserves its own attempt rather than inheriting the old
   * one's failure.
   */
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [src]);

  // The picture needs both a preview and a URL for it, and the URL has to load.
  // They come apart in fixture mode, where the manifest names a preview.mp4 that
  // no fixture can serve — and the panel is a status board rather than a blank
  // rectangle in exactly that case, which is the case it was written for.
  const picture = mode === "preview" && src != null && !failed;
  /*
   * Nothing attached is nothing to play, whatever the mode says.
   *
   * `mode === "none"` alone left the button live over a run whose preview is
   * named but unreachable, and the transport's fallback clock then swept a
   * playhead across the timeline with no audio anywhere — the same lie the dead
   * play button was written to stop, one branch further along.
   */
  const silent = mode === "none" || src == null || failed;

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
            onError={() => setFailed(true)}
          />
        ) : null}
        {picture ? null : failed ? <MediaError label={srcLabel} /> : placeholder}
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
          title={silent ? (mode === "none" ? NOTHING_TO_PLAY : NO_MEDIA) : undefined}
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

/**
 * The file is named and the file will not load.
 *
 * It replaces the status board rather than sitting beside it, because the board
 * would be answering a question this run has already moved past: the stages all
 * say `done`, and the thing that is wrong is one file. Named, because "could not
 * be loaded" about an unnamed thing is not something anyone can act on.
 */
function MediaError({ label }: { label?: string | null }) {
  return (
    <div className="grid h-full place-items-center px-4 py-3" data-media-error>
      <div className="max-w-sm text-center">
        <TriangleAlert
          aria-hidden
          className="mx-auto h-4 w-4"
          style={{ color: "var(--color-critical)" }}
        />
        <p className="mt-2 text-[14px] font-semibold text-primary">
          {label ?? "The video"} could not be loaded
        </p>
        <p className="mt-1.5 text-[11px] leading-relaxed text-muted">
          The run names the file but the player could not open it — it may have been moved or
          deleted, or the render may not have finished writing it. Render the preview again to
          make a new one.
        </p>
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
