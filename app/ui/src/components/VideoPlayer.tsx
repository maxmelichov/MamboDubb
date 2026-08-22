/**
 * The preview stage.
 *
 * This used to be the app's one dark island a black surround with white
 * transport controls, on the reasoning that video wants black around it. That
 * reasoning does not survive contact with either theme: in light the stage sat
 * in the top-left of every editor screen as a slab, and in dark a *second*,
 * differently-black rectangle inside a near-black app reads as a hole. So the
 * surround is the app's own sunken tone, whichever theme that resolves to, and
 * the transport is the house style.
 *
 * It is no longer the pane that grows. The script is. The picture takes the top
 * of a fixed right-hand column, at the video's own aspect ratio, and the
 * transport sits directly under it which is also why the footer lost the
 * project title (the header already says it, forty pixels above) and the
 * `space` hint (a keyboard hint printed permanently is a legend, and the
 * legends went behind the "?").
 *
 * ## Three modes, one transport
 *
 * `preview` is the finished thing. `source` is the run's own audio, which
 * `fetch` writes long before `mix` does it is loaded into the same element
 * with the picture hidden, so play, seek, the timecode, the playhead and the
 * script's follow-along all drive the file rather than a synthetic clock, and
 * the panel keeps showing the run's status board instead of a black rectangle.
 * `none` is a run that has not fetched anything yet: there is nothing to play,
 * so the play button says so and refuses, rather than running a clock over
 * silence which is what "i can start playing. but because it still not ready
 * it shown nothing" was.
 *
 * The label is only shown in `source`, because it is only true there: a chip
 * that says "original audio" over the finished dub would be worse than no chip.
 */

import { useEffect, useRef, useState } from "react";
import {
  Maximize,
  Minimize,
  Pause,
  Play,
  SkipBack,
  SkipForward,
  TriangleAlert,
  Volume2,
  VolumeX,
} from "lucide-react";
import { cn } from "../lib/classNames";
import { timecode } from "../lib/format";
import type { Transport } from "../lib/useTransport";
import type { ReactNode } from "react";

/** What the transport is on. EditorPage decides; this draws it. */
export type TransportMode = "preview" | "source" | "none";

/** Said in the chip, and nowhere else see the note above. */
const ORIGINAL_AUDIO = "Original audio (no preview yet)";

/** Why the play button is dead. Reads as a state of the run, not a failure. */
const NOTHING_TO_PLAY = "Nothing to play yet: the fetch stage hasn't run";

/**
 * Why the play button is dead when the run *says* it has a file.
 *
 * The manifest naming `preview.mp4` and the browser being able to load it are
 * two different facts, and they come apart routinely: fixture mode, a file
 * deleted out from under a run, a mid-render truncated mp4. "The fetch stage
 * hasn't run" is the wrong sentence for every one of those.
 */
const NO_MEDIA = "This run's video could not be loaded: there is nothing to play";

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
   * transport under it the app claiming to be playing a video it does not
   * have. The state is keyed to the source: a render finishing swaps the URL,
   * and the new file deserves its own attempt rather than inheriting the old
   * one's failure.
   */
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [src]);

  /*
   * The YouTube grammar, because that is the grammar everyone's hands know:
   * click the picture to play/pause, double-click it for fullscreen, scrub on
   * a bar, mute where you expect mute. All of it drives the one shared
   * transport — the timeline playhead and the script's follow-along move with
   * the scrub exactly as they move with the timeline's own seek.
   */
  const rootRef = useRef<HTMLDivElement | null>(null);
  const [fullscreen, setFullscreen] = useState(false);
  useEffect(() => {
    const onChange = () => setFullscreen(document.fullscreenElement != null);
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, []);
  const toggleFullscreen = () => {
    if (document.fullscreenElement) void document.exitFullscreen();
    else void rootRef.current?.requestFullscreen?.().catch(() => {});
  };

  // Volume lives on the element, but the element is remounted per source —
  // state here is what keeps a chosen volume across the source→preview swap.
  const [muted, setMuted] = useState(false);
  const [volume, setVolume] = useState(1);
  useEffect(() => {
    const el = transport.videoRef.current;
    if (el) {
      el.muted = muted;
      el.volume = volume;
    }
  }, [muted, volume, src, transport.videoRef]);

  // The picture needs both a preview and a URL for it, and the URL has to load.
  // They come apart in fixture mode, where the manifest names a preview.mp4 that
  // no fixture can serve and the panel is a status board rather than a blank
  // rectangle in exactly that case, which is the case it was written for.
  const picture = mode === "preview" && src != null && !failed;
  /*
   * Nothing attached is nothing to play, whatever the mode says.
   *
   * `mode === "none"` alone left the button live over a run whose preview is
   * named but unreachable, and the transport's fallback clock then swept a
   * playhead across the timeline with no audio anywhere the same lie the dead
   * play button was written to stop, one branch further along.
   */
  const silent = mode === "none" || src == null || failed;

  return (
    <div ref={rootRef} className={cn("flex flex-col bg-sunken", className)}>
      <div
        className={cn("relative min-h-0 flex-1", picture && "cursor-pointer")}
        // On the PICTURE only: with the placeholder up, a click is aimed at
        // whatever the placeholder offers (a retry button, a resume), and a
        // toggle stolen from under it would play audio nobody asked for.
        onClick={picture ? transport.toggle : undefined}
        onDoubleClick={picture ? toggleFullscreen : undefined}
      >
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

      <Scrubber current={transport.currentTime} duration={duration} seek={transport.seek} />

      <div
        data-transport={mode}
        className="flex shrink-0 items-center gap-1.5 border-b border-border bg-surface px-3 py-1.5"
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

        <TransportButton
          onClick={() => setMuted((m) => !m)}
          label={muted ? "Unmute" : "Mute"}
          title={silent ? "No audio attached to mute" : undefined}
          disabled={silent}
        >
          {muted || volume === 0 ? (
            <VolumeX className="h-3.5 w-3.5" />
          ) : (
            <Volume2 className="h-3.5 w-3.5" />
          )}
        </TransportButton>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={muted ? 0 : volume}
          disabled={silent}
          aria-label="Volume"
          data-volume
          onChange={(event) => {
            setMuted(false);
            setVolume(Number(event.currentTarget.value));
          }}
          className="w-14 shrink-0 accent-[var(--color-primary)] disabled:opacity-40"
        />

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

        <TransportButton
          onClick={toggleFullscreen}
          label={fullscreen ? "Exit fullscreen" : "Fullscreen"}
          title={picture ? undefined : "Fullscreen needs a picture to fill"}
          disabled={!picture && !fullscreen}
        >
          {fullscreen ? <Minimize className="h-3.5 w-3.5" /> : <Maximize className="h-3.5 w-3.5" />}
        </TransportButton>
      </div>
    </div>
  );
}

/**
 * The seek bar — YouTube's, structurally: a full-width strip whose fill is the
 * played fraction, click or drag anywhere on it to go there. It drives
 * `transport.seek`, so it works over the preview, the source audio and the
 * synthetic clock alike, and the timeline playhead follows it for free.
 */
function Scrubber({
  current,
  duration,
  seek,
}: {
  current: number;
  duration: number;
  seek: (time: number) => void;
}) {
  const barRef = useRef<HTMLDivElement | null>(null);
  const disabled = duration <= 0;
  const frac = disabled ? 0 : Math.max(0, Math.min(1, current / duration));

  const seekAt = (clientX: number) => {
    const rect = barRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0) return;
    const f = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    seek(f * duration);
  };

  return (
    <div
      ref={barRef}
      role="slider"
      aria-label="Seek"
      aria-valuemin={0}
      aria-valuemax={Math.max(0, Math.round(duration))}
      aria-valuenow={Math.round(current)}
      aria-disabled={disabled || undefined}
      tabIndex={disabled ? -1 : 0}
      data-scrubber
      onPointerDown={(event) => {
        if (disabled) return;
        // Capture so a drag keeps scrubbing past the strip's own edge —
        // guarded, because the smoke DOM has no pointer capture.
        event.currentTarget.setPointerCapture?.(event.pointerId);
        seekAt(event.clientX);
      }}
      onPointerMove={(event) => {
        if (disabled) return;
        if (event.currentTarget.hasPointerCapture?.(event.pointerId)) seekAt(event.clientX);
      }}
      onKeyDown={(event) => {
        if (disabled) return;
        if (event.key === "ArrowRight") {
          event.preventDefault();
          seek(current + 5);
        } else if (event.key === "ArrowLeft") {
          event.preventDefault();
          seek(current - 5);
        }
      }}
      className={cn(
        "group/scrub flex h-3 shrink-0 touch-none items-center border-t border-border bg-surface px-3",
        disabled ? "cursor-default" : "cursor-pointer",
      )}
    >
      <div className="relative h-[3px] w-full rounded-full bg-border transition-[height] group-hover/scrub:h-[5px]">
        <div
          data-scrubber-fill
          className="absolute inset-y-0 left-0 rounded-full"
          style={{ width: `${frac * 100}%`, background: "var(--color-primary)" }}
        />
        <div
          aria-hidden
          className="absolute top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full opacity-0 transition-opacity group-hover/scrub:opacity-100"
          style={{ left: `${frac * 100}%`, background: "var(--color-primary)" }}
        />
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
          The run names the file but the player could not open it. It may have been moved or
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
  /** The tooltip, when it has more to say than the label a reason, usually. */
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
