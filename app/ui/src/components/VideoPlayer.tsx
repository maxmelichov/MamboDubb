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
import { isDesktop, setWindowFullscreen } from "../lib/desktop";
import { timecode } from "../lib/format";
import { ErrorBar } from "./ui";
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

/*
 * The two ways fullscreen can end badly, said out loud.
 *
 * Neither of these is "the button did nothing", which is the state this pair
 * exists to make unreachable. The first is the fallback landing half-way: the
 * webview refused the Fullscreen API *and* the shell would not take the window
 * with it, so the picture fills the app window and the app window is whatever
 * size it was. That is a real result and a worse one than asked for, so it is
 * worth a sentence. The second is the genuinely bad one, where the browser will
 * not let go, and the sentence has to name the way out.
 */
const FULLSCREEN_WINDOW_ONLY =
  "This build's webview refused fullscreen and the window would not follow, so the picture fills the app window instead.";

const FULLSCREEN_STUCK = "The browser refused to leave fullscreen. Press Escape to get out.";

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

  /*
   * Fullscreen, and what to do when the platform says no.
   *
   * This was one line that asked the element to go fullscreen and swallowed the
   * rejection in an empty `catch`, which is how "full screen dont work" got to
   * be a bug report with no evidence attached: in the desktop shell the press
   * did nothing, said nothing, and left no trace anywhere.
   *
   * The reason is not the button and not the page. The shell is a WKWebView,
   * and WebKit only honours the Fullscreen API when the `fullScreenEnabled`
   * preference is set on the webview before it is created. wry does set it, but
   * only under its `fullscreen` Cargo feature, which Tauri re-exports as
   * `macos-private-api`, and the shell did not ask for it. That switch is now
   * on (see `app/desktop/src-tauri/Cargo.toml`), so the desktop path and the
   * browser path are the same path: `requestFullscreen` on the stage, with the
   * platform's own escape key and menu-bar behaviour for free.
   *
   * The fallback below is not there to paper over that fix. It is there because
   * a private WebKit preference is exactly the sort of thing that stops working
   * one macOS release later, and because this bundle also has to survive a
   * WebKitGTK or WebView2 that refuses for its own reasons. When the request is
   * refused we fill instead: the stage goes `fixed inset-0` over the page, and
   * on the desktop we additionally ask the *window* to go fullscreen, so what
   * the user gets is still a picture that covers the screen rather than nothing
   * at all. The rejection reason goes to the console either way, because the
   * whole point of the old empty `catch` being wrong is that nobody could see
   * the refusal, and if the window would not come with us we say so out loud.
   *
   * Two states, because the icon must never claim something the screen is not
   * doing. `native` is the browser's, and only the browser can set or clear it,
   * so it is read back off `fullscreenchange` and compared against our own
   * element rather than any element. `filling` is ours, and is the truth for
   * the fallback, which fires no `fullscreenchange` at all. The button and the
   * double-click read the two together.
   */
  const [native, setNative] = useState(false);
  const [filling, setFilling] = useState(false);
  const [refused, setRefused] = useState<string | null>(null);
  const fullscreen = native || filling;

  useEffect(() => {
    const onChange = () =>
      setNative(rootRef.current != null && document.fullscreenElement === rootRef.current);
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, []);

  const leaveFill = () => {
    setFilling(false);
    setRefused(null);
    void setWindowFullscreen(false);
  };

  /*
   * Escape, by hand, because the fill is not a real fullscreen and nothing else
   * will do it. Capture phase and `stopPropagation`, so the one press that
   * leaves the fill does not also reach the editor's own Escape and clear the
   * selection underneath it.
   */
  useEffect(() => {
    if (!filling) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.stopPropagation();
      leaveFill();
    };
    document.addEventListener("keydown", onKey, true);
    return () => document.removeEventListener("keydown", onKey, true);
  }, [filling]);

  const enterFullscreen = async () => {
    const root = rootRef.current;
    if (!root) return;
    setRefused(null);

    const request = root.requestFullscreen?.bind(root);
    if (request) {
      try {
        await request();
        // `fullscreenchange` takes it from here; nothing to fill.
        return;
      } catch (error) {
        console.warn("VideoPlayer: the platform refused element fullscreen", error);
      }
    } else {
      console.warn("VideoPlayer: no Fullscreen API on this element");
    }

    setFilling(true);
    // False in a browser, where filling the viewport is already the whole of
    // what we can do and nothing is wrong. Only the shell owes us a window.
    const windowed = await setWindowFullscreen(true);
    if (isDesktop() && !windowed) setRefused(FULLSCREEN_WINDOW_ONLY);
  };

  const exitFullscreen = async () => {
    if (document.fullscreenElement) {
      try {
        await document.exitFullscreen();
      } catch (error) {
        console.warn("VideoPlayer: the platform refused to exit fullscreen", error);
        setRefused(FULLSCREEN_STUCK);
        return;
      }
    }
    leaveFill();
  };

  const toggleFullscreen = () => {
    void (fullscreen ? exitFullscreen() : enterFullscreen());
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
    <div
      ref={rootRef}
      // `fill` drops the caller's own sizing on purpose: `className` is what
      // parks this in the editor's right-hand column, and a fullscreen that
      // still obeys a 40rem max-width is not a fullscreen.
      data-fullscreen={fullscreen ? (filling ? "fill" : "native") : undefined}
      className={cn("flex flex-col bg-sunken", filling ? "fixed inset-0 z-50" : className)}
    >
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

      {/* The house error strip, in the player rather than the page: the press
          happened here, and in the fill the page's own chrome is behind us. */}
      {refused ? <ErrorBar message={refused} onDismiss={() => setRefused(null)} /> : null}

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
