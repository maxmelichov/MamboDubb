/**
 * One line of the script.
 *
 * This is the object the whole editor is now built around, so it is worth
 * saying what it is for: a reviewer is checking a *translation against its
 * original*, and that is a comparison between two pieces of text. Every earlier
 * shape of this list made that comparison impossible. The seven-column table
 * gave each half a truncated third of the width. The one-line list showed
 * whichever half "would play" and hid the other, so checking a translation
 * meant selecting the row and reading the rail. Both were asking the user to
 * hold one sentence in their head while they went and found the other.
 *
 * So the two lines are stacked, always, both visible, both complete:
 *
 *   ORIGINAL      muted, 12.5px  — the reference
 *   TRANSLATION   ink,   14px    — the work
 *
 * and nothing here is ever ellipsis-truncated. A collapsed row clamps at two
 * lines; the selected row shows everything. Truncating the text is truncating
 * the only thing on the screen the user came for.
 *
 * ## When the two lines are one line
 *
 * "Both visible, always" is a rule about a *comparison*, and there is no
 * comparison when the two halves are the same string. A run over a video that
 * already speaks the target language keeps every line and passes the text
 * through untouched, so `text_en === text` on all seventy-three of them — and
 * the row printed the identical sentence twice, at two weights, seventy-three
 * times. That does not read as "kept", it reads as a broken renderer.
 *
 * So: identical text is one line, and a kept line with no subtitle of its own
 * is one line. Two lines are drawn only when they actually differ, and on a
 * kept row the second one is dimmed, because a subtitle on a kept line is not
 * what will be spoken. Which state the row is in is said in words on the meta
 * line either way, where it always was.
 *
 * ## Direction
 *
 * The row container is explicitly `dir="ltr"` and the two text lines are
 * `dir="auto"` + `.auto-dir`. That split is the whole bidi story: the timecode,
 * the id and the buttons are *chrome*, laid out left-to-right no matter what
 * language the run is in, while each text line takes its direction from its own
 * first strong character — so a Hebrew original renders right-to-left directly
 * above its left-to-right English translation, in one row, correctly, without
 * either of them dragging the layout around.
 *
 * The per-row state rule uses `border-inline-start` rather than `border-left`
 * so it stays on the reading edge if the *chrome* is ever mirrored.
 *
 * There is deliberately no `title` on the row. A composed tooltip
 * (`${speaker} · ${text}`) is a single string with mixed directions and no
 * markup to scope them, which the platform renders as a scrambled mess — and it
 * was duplicating text that is now permanently on screen anyway.
 */

import { memo, useEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { Lock, MoreHorizontal, Pause, Play, TriangleAlert } from "lucide-react";
import { cn } from "../lib/classNames";
import { speakerLabel, timecode } from "../lib/format";
import {
  STATE_META,
  hasKeepPhrase,
  hasLocks,
  keepReason,
  segmentState,
  subtitleOnly,
  verifyConcern,
} from "../lib/segments";
import { StateIcon, TextArea } from "./ui";
import type { Segment } from "../lib/types";

export type EditTarget = { uid: string; field: "text" | "text_en" } | null;

export type ScriptRowProps = {
  seg: Segment;
  selected: boolean;
  /** the playhead is inside this segment */
  now: boolean;
  busy: boolean;
  /** this row is the list's single tab stop */
  tabStop: boolean;
  /** which field of this row is open for editing, if any */
  editing: "text" | "text_en" | null;
  /** the clip URL currently sounding, so Orig/Dub can show which side is playing */
  playingUrl: string | null;
  /** what the search box holds, lower-cased, so the hits can be marked */
  query: string;
  onSelect: (uid: string) => void;
  onEdit: (target: EditTarget) => void;
  onCommit: (uid: string, field: "text" | "text_en", value: string) => void;
  /** `window` confines playback to a span of the file — see `clipAudio`. */
  onPlay: (url: string | null, window?: [number, number] | null) => void;
  onToggleKeep: (seg: Segment) => void;
};

function Row({
  seg,
  selected,
  now,
  busy,
  tabStop,
  editing,
  playingUrl,
  query,
  onSelect,
  onEdit,
  onCommit,
  onPlay,
  onToggleKeep,
}: ScriptRowProps) {
  const state = segmentState(seg);
  const meta = STATE_META[state];
  const concern = verifyConcern(seg);
  const [menuOpen, setMenuOpen] = useState(false);

  const sourceUrl = seg.media?.source ?? null;
  const dubUrl = seg.media?.play ?? seg.media?.tts ?? null;

  const translation = seg.text_en ?? "";
  // An echo is a translation that is character-for-character the original —
  // what a passthrough keep produces. There is nothing to compare, so there is
  // one line. A kept line with no subtitle at all is one line for the same
  // reason: "no subtitle line for this span" is not news on a row that already
  // says the original audio plays.
  const echo = translation.trim() !== "" && translation.trim() === seg.text.trim();
  // …but ↵ still opens the field on a collapsed row: the line the reviewer
  // wants to write is precisely the one that is not there yet.
  const showTranslation =
    editing === "text_en" || (!echo && !(seg.keep && translation.trim() === ""));

  return (
    <div
      /*
       * `listbox`/`option`: this is a single-select picker, and that is the
       * role that makes `aria-selected` mean "this is the one in the panel".
       */
      role="option"
      dir="ltr"
      data-uid={seg.uid}
      data-lines={showTranslation ? 2 : 1}
      data-now={now ? "" : undefined}
      aria-selected={selected}
      // Roving tabindex: the list is one tab stop, not two hundred.
      tabIndex={tabStop ? 0 : -1}
      onKeyDown={(event) => {
        if (event.key === "Enter" && event.target === event.currentTarget) {
          event.preventDefault();
          onSelect(seg.uid);
          onEdit({ uid: seg.uid, field: "text_en" });
        }
      }}
      /*
       * Three states, three strengths, and they have to be told apart at a
       * glance because two of them are usually the same row: selecting a line
       * seeks to it, so the selection *is* the playhead until playback moves
       * on. Before, both were a faint ink wash over an inline-start rule that
       * changed hue between them, which is a difference nobody can see.
       *
       *   selected     accent wash + inset ring + the rule goes accent (strongest)
       *   now playing  a fainter ink wash + a short ink tick on the rule (medium)
       *   hover        the sunken tone, and nothing else (faintest)
       *
       * The two strongest are deliberately different *colours* and not two
       * strengths of one: the selection is the accent (`primary` in light, the
       * brand violet in dark) and the playhead stays ink, so on the row that is
       * both — which is most of them, since selecting a line seeks to it — the
       * violet rule and the white tick are still two readable facts.
       *
       * The rule itself is the state's hue at every other time — a quiet stripe
       * down the reading edge that says dub/keep/fail without a second glance.
       */
      className={cn(
        "group/row relative flex gap-2 border-b border-grid px-2 py-1.5",
        "transition-colors",
        selected
          ? // The wash and the ring are stronger in dark for the same reason
            // `--mark-wash` is: how much of a hue a fill needs is a property of
            // the ground. Light's accent is ink and 7% of a near-black already
            // reads as a tint; dark's is a mid violet, which at 7% over a
            // near-black plane lightens the row less than the near-white it
            // replaced and loses the selection. Light's two numbers are the
            // ones that shipped and are untouched.
            "bg-accent/[0.07] ring-1 ring-inset ring-accent/25 dark:bg-accent/10 dark:ring-accent/45"
          : now
            ? "bg-primary/[0.045]"
            : "hover:bg-sunken",
        busy && "animate-pulse",
      )}
      style={{
        borderInlineStartWidth: 3,
        borderInlineStartStyle: "solid",
        borderInlineStartColor: selected
          ? "var(--color-accent)"
          : `color-mix(in srgb, ${meta.token} var(--state-rule), transparent)`,
      }}
    >
      {/* The playhead's tick, painted over the state stripe: short, so it does
          not read as the selection's full-height rule. */}
      {now && !selected ? (
        <span
          aria-hidden
          className="pointer-events-none absolute top-1/2 h-6 w-[3px] -translate-y-1/2 rounded-e-sm bg-primary"
          style={{ insetInlineStart: -3 }}
        />
      ) : null}

      {/* --- gutter: where this line is, and whether a hand has been on it --- */}
      <button
        type="button"
        onClick={() => onSelect(seg.uid)}
        aria-label={`Select segment ${seg.id} at ${timecode(seg.start)}`}
        className="flex w-16 shrink-0 cursor-pointer flex-col items-start gap-px pt-px text-left"
      >
        <span
          className={cn(
            "font-mono text-[12.5px] tabular-nums",
            selected || now ? "font-semibold text-primary" : "text-secondary",
          )}
        >
          {timecode(seg.start, 0)}
        </span>
        <span className="flex items-center gap-1 font-mono text-[11px] tabular-nums text-muted">
          #{seg.id}
          {hasLocks(seg) ? (
            <Lock className="h-2.5 w-2.5" aria-label="Hand-edited — a re-run will not overwrite it" />
          ) : null}
        </span>
      </button>

      {/* --- body: the meta line, then the two texts --- */}
      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
        <button
          type="button"
          onClick={() => onSelect(seg.uid)}
          className="flex min-w-0 items-center gap-1.5 text-left text-[11px] text-muted"
        >
          <span className="shrink-0 font-semibold uppercase tracking-[0.08em]">
            {speakerLabel(seg.speaker)}
          </span>
          <span aria-hidden>·</span>
          <StateIcon state={state} className="h-2.5 w-2.5" />
          {/* The state as a *word*, on every row, in ink rather than in the
              state's hue: at 11px this is small text, and light-mode "pending"
              is 3.70:1 against the card — fine for a stripe, under the 4.5:1
              gate for a label. There is no legend on screen any more, so this
              is the encoding's only spelling out. */}
          <span className="shrink-0 text-secondary">{meta.short}</span>
          {/*
            A soft verification is a *concern*, and it was drawn in the colour
            of furniture: the same muted grey as the timecode beside it, on a
            row that otherwise looks finished. It joins the pending family —
            amber, the hue this app already spends on "something is still
            outstanding here" — while a hard failure keeps the red. Both are
            icons, where the 3:1 gate applies; neither is coloured text.
          */}
          {concern !== "none" ? (
            <TriangleAlert
              data-concern={concern}
              className={cn(
                "h-3 w-3 shrink-0",
                concern === "bad" ? "text-critical" : "text-warning",
              )}
              aria-label={concern === "bad" ? "Verification failed" : "Verification is low"}
            />
          ) : null}
          {/*
            What a kept row does instead of a fourth line. It used to be a
            paragraph of its own under the two texts — on all seventy-three
            rows of an all-kept run, which is not information, it is a
            drumbeat. Now it is a clause on the meta line, and it fades in on
            the row under the cursor or under the selection: the *state* is on
            every row, in a word and a shape, and the sentence spelling out
            what that state means is worth reading once, for the row you are
            actually looking at. It stays in the DOM either way, so it is
            always there for a screen reader and for find-in-page.

            And what it says is the *reason*, not the consequence. "Original
            audio plays here" is already spelled by the Keep badge two spans to
            the left; the question a reviewer checking keeps actually has is
            why this one was kept, and it was only answerable by selecting the
            row and reading the panel. `keepReason` is the same mapping the
            panel and the run summary use, so the three cannot drift — and a
            reason it has no phrase for falls back to the old sentence rather
            than printing a manifest token on seventy-three rows.
          */}
          {seg.keep ? (
            <span
              className={cn(
                "min-w-0 truncate transition-opacity",
                selected ? "opacity-100" : "opacity-0 group-hover/row:opacity-100",
              )}
            >
              — {hasKeepPhrase(seg.keep_reason)
                ? keepReason(seg.keep_reason)
                : "original audio plays here"}
            </span>
          ) : null}
        </button>

        <Line
          seg={seg}
          field="text"
          value={seg.text}
          query={query}
          editing={editing === "text"}
          placeholder="no transcript for this span"
          /* When it is the only line it is also the line that plays, so it
             takes the reading size; under a translation it is the reference. */
          className={
            showTranslation
              ? "text-[12.5px] text-muted"
              : "text-[14px] leading-snug text-secondary"
          }
          onEdit={onEdit}
          onCommit={onCommit}
          selected={selected}
          /* The original is reference, not the work: it opens for editing only
             through the ⋯ menu ("Correct transcript"), never by a stray click
             on the line a reviewer is reading. */
          readOnly
        />

        {/*
          A kept line that has a subtitle of its own still shows it — a
          reviewer's job includes checking the keep decisions, and hiding the
          translation of a kept line hides the evidence for the judgement it is
          asking for. It is drawn dimmer than the original above it, because it
          is not what will be spoken.
        */}
        {showTranslation ? (
          <Line
            seg={seg}
            field="text_en"
            value={translation}
            query={query}
            editing={editing === "text_en"}
            placeholder="not translated yet"
            className={cn(
              "text-[14px] leading-snug",
              seg.keep ? "text-muted" : "font-medium text-primary",
            )}
            /* Dimmed as a *reading* line only: a field you are typing into is
               the one thing on the row that must be at full strength. */
            dimmed={seg.keep}
            onEdit={onEdit}
            onCommit={onCommit}
            selected={selected}
          />
        ) : null}

        {/*
          The one thing a dimmed line cannot say on its own.

          Typing a translation into a kept row is allowed and it is saved — but
          `edit.set_text` does not reopen a verdict the user or the span
          decided, so the line is a *subtitle* and the original audio still
          plays. Nothing on the row said so: it showed an English sentence
          under a Hebrew one, in the same place the dubbed rows show what will
          be spoken, and the reasonable reading is that it will be spoken.
          Flipping the verdict here instead would be worse — it would overturn
          a decision the user made, on the strength of them fixing a subtitle.
        */}
        {subtitleOnly(seg) ? (
          <p className="px-2 text-[11px] leading-snug text-muted">
            This line keeps its original audio — the edit is a subtitle. Switch to “Dub it” to
            voice it.
          </p>
        ) : null}
      </div>

      {/*
        --- actions ---
        Rendered always (so the row's height never changes under the cursor)
        but invisible until the row is hovered, selected, or has its menu open.
        Two hundred rows must not be six hundred buttons competing for the eye.
      */}
      <div
        className={cn(
          // Wide enough for the two words and the menu. It was 88px when the
          // buttons said "A" and "B"; the 36 the labels cost come out of the
          // reading column, which is the trade the words are worth.
          "flex w-[124px] shrink-0 items-start gap-0.5 pt-0.5",
          "invisible group-hover/row:visible",
          (selected || menuOpen) && "visible",
        )}
      >
        {/* Orig is a *window* of the source track, not a file of its own: the
            server says which one in `media.source_window`, and without it the
            press starts wherever the browser felt like starting and never
            stopped at the segment's end.

            "A" and "B" were the keyboard's names for these, printed on the
            buttons — a convention from comparison tools, which this is, but one
            that tells a reviewer meeting the row nothing about which side is
            which. The words say it; `a` and `b` still press them, and the
            shortcut list names both. `data-clip` stays A/B: it is the seam the
            smoke test reads, and the sides have not changed. */}
        <ClipButton
          id="A"
          label="Orig"
          title="Play the original audio for this span"
          url={sourceUrl}
          window={seg.media?.source_window ?? null}
          playing={playingUrl != null && playingUrl === sourceUrl}
          onPlay={onPlay}
        />
        <ClipButton
          id="B"
          label="Dub"
          title={
            seg.keep
              ? "No dub exists — this line keeps the original, so A and B would be the same audio"
              : "Play what actually went into the mix, after time-fitting"
          }
          // A kept line's placed clip IS the original slice; letting B play it
          // sounds identical to A, which a user reads as "the buttons are
          // swapped". No dub means no B.
          emptyTitle={
            seg.keep
              ? "No dub exists — this line keeps the original, so A and B would be the same audio"
              : undefined
          }
          url={seg.keep ? null : dubUrl}
          playing={playingUrl != null && !seg.keep && playingUrl === dubUrl}
          onPlay={onPlay}
        />
        <RowMenu
          seg={seg}
          open={menuOpen}
          setOpen={(open) => {
            if (open) onSelect(seg.uid);
            setMenuOpen(open);
          }}
          onCorrect={() => onEdit({ uid: seg.uid, field: "text" })}
          onToggleKeep={() => onToggleKeep(seg)}
        />
      </div>
    </div>
  );
}

/**
 * Memoized on purpose, and the callbacks above it are all stable: a two-hundred
 * row list re-rendering every row on every `timeupdate` (four times a second,
 * while playing) is the difference between a scrub that tracks the audio and
 * one that stutters.
 */
export const ScriptRow = memo(Row);

/**
 * One line of text that turns into a field where it stands.
 *
 * "Where it stands" is the requirement and it is why the static paragraph
 * carries the same padding, radius and border box as the textarea, with the
 * border merely transparent: a click that shifts the text by three pixels
 * makes the user re-find the word they were aiming at.
 */
/**
 * The search's hits, wrapped where they are.
 *
 * Two rules, and the second is the whole reason this is a function rather than
 * a `dangerouslySetInnerHTML`. The `<mark>`s go *inside* the `dir="auto"`
 * paragraph, never around it and never splitting it into fragments with their
 * own direction: the paragraph is the bidi context, and a Hebrew line chopped
 * into three siblings is three lines each laid out from its own first strong
 * character — which reorders the sentence on screen. And the match is found on
 * a lower-cased copy while the slices come from the original, so "Moza" typed
 * as "moza" still shows the capital it actually has.
 */
function marked(value: string, needle: string): ReactNode {
  if (!needle) return value;
  const hay = value.toLowerCase();
  const out: ReactNode[] = [];
  let cursor = 0;
  for (let at = hay.indexOf(needle); at !== -1; at = hay.indexOf(needle, cursor)) {
    if (at > cursor) out.push(value.slice(cursor, at));
    out.push(
      <mark key={at} data-hit>
        {value.slice(at, at + needle.length)}
      </mark>,
    );
    cursor = at + needle.length;
  }
  if (cursor === 0) return value;
  if (cursor < value.length) out.push(value.slice(cursor));
  return out;
}

function Line({
  seg,
  field,
  value,
  query,
  editing,
  placeholder,
  className,
  dimmed,
  selected,
  readOnly,
  onEdit,
  onCommit,
}: {
  seg: Segment;
  field: "text" | "text_en";
  value: string;
  /** The search, lower-cased. Its hits are marked in this line. */
  query: string;
  editing: boolean;
  placeholder: string;
  className?: string;
  /** Held back from the line above it — never from the field. */
  dimmed?: boolean;
  selected: boolean;
  readOnly?: boolean;
  onEdit: (target: EditTarget) => void;
  onCommit: (uid: string, field: "text" | "text_en", value: string) => void;
}) {
  if (editing) {
    return (
      <RowEditor
        initial={value}
        className={className}
        onCancel={() => onEdit(null)}
        onCommit={(next) => onCommit(seg.uid, field, next)}
      />
    );
  }

  return (
    <p
      dir="auto"
      data-line={field}
      onClick={readOnly ? undefined : () => onEdit({ uid: seg.uid, field })}
      className={cn(
        "auto-dir rounded-lg border border-transparent px-2 py-1",
        // Never an ellipsis: a clamp hides *lines*, a truncate hides the end of
        // a sentence — and the end of a translated sentence is exactly where
        // the mistakes are. The selected row is uncapped.
        selected ? "whitespace-pre-wrap" : "line-clamp-2",
        dimmed && "opacity-70",
        !readOnly && "cursor-text hover:border-border",
        !value && "italic text-muted",
        className,
      )}
    >
      {value ? marked(value, query) : placeholder}
    </p>
  );
}

/**
 * The field itself.
 *
 * Three rules, each of which was a bug:
 *
 * 1. **The draft is seeded once, on mount, and never re-seeded.** The old
 *    inspector re-synced its draft from props on every change of `seg.text_en`
 *    — so a background job completing, or the 250ms refetch the event stream
 *    schedules, silently threw away whatever the user was in the middle of
 *    typing. Mounting the editor *with* the value and never looking at props
 *    again makes a refetch structurally unable to reach it.
 * 2. **An empty commit is refused, never sent.** `edit.set_text` 400s on it —
 *    "a dubbed segment must say something" — and a user who wants the line gone
 *    wants Keep original, which is a different button with a different meaning.
 * 3. **Blur commits, Escape reverts**, and Escape has to win: it blurs the
 *    field on its way out, so the blur handler has to know it was cancelled.
 */
function RowEditor({
  initial,
  className,
  onCommit,
  onCancel,
}: {
  initial: string;
  className?: string;
  onCommit: (value: string) => void;
  onCancel: () => void;
}) {
  const [draft, setDraft] = useState(initial);
  const cancelled = useRef(false);
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    ref.current?.focus();
    ref.current?.setSelectionRange(initial.length, initial.length);
    // Once, on mount. Re-running this on `initial` would re-seed the draft,
    // which is precisely rule 1 above.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const commit = () => {
    if (cancelled.current) return;
    // Rule 2. Refused before it is ever sent — and refused by *reverting*,
    // because a field that will not close until you type something is a trap:
    // the user who emptied the line meant Keep original, and they can still
    // reach it with the line intact.
    if (!draft.trim()) {
      onCancel();
      return;
    }
    onCommit(draft.trim());
  };

  return (
    <span className="flex flex-col">
      {/*
        `dir="auto"` and not just the `.auto-dir` class: the class sets
        `unicode-bidi: plaintext`, which is enough to *render* a Hebrew line
        right-to-left, but a textarea also has a caret, a selection and a
        home/end key, and those follow the element's direction. A Hebrew
        line typed into an LTR box puts the cursor in the wrong place on
        every keystroke.
      */}
      <TextArea
        ref={ref}
        dir="auto"
        data-editing
        autoGrow
        value={draft}
        onChange={(event) => setDraft(event.currentTarget.value)}
        onBlur={commit}
        onKeyDown={(event) => {
          if (event.key === "Escape") {
            event.preventDefault();
            event.stopPropagation();
            cancelled.current = true;
            onCancel();
          }
          if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
            event.preventDefault();
            commit();
          }
          // The list's own ↑/↓/space/a/b bindings must not fire mid-word.
          event.stopPropagation();
        }}
        className={cn("auto-dir min-h-0 rounded-lg px-2 py-1", className)}
      />
      {!draft.trim() ? (
        <span className="px-2 pt-0.5 text-[11px] text-critical">
          A dubbed line has to say something — switch it to Keep original instead.
        </span>
      ) : (
        <span className="px-2 pt-0.5 text-[11px] text-muted">⌘↵ or click away to save</span>
      )}
    </span>
  );
}

/** Orig/Dub, one clip each, on the app's single shared audio element. */
function ClipButton({
  id,
  label,
  title,
  url,
  window,
  playing,
  onPlay,
  emptyTitle,
}: {
  /** The side, A or B — the keybinding's name and the smoke test's handle. */
  id: "A" | "B";
  /** The word on the button. */
  label: string;
  title: string;
  url: string | null;
  /** The span of `url` this side is: null when the file is already the clip. */
  window?: [number, number] | null;
  playing: boolean;
  onPlay: (url: string | null, window?: [number, number] | null) => void;
  emptyTitle?: string;
}) {
  return (
    <button
      type="button"
      data-clip={id}
      data-url={url ?? ""}
      // The span this side is confined to, for the same reason `data-url` is
      // here: it is the contract the smoke test can see from outside.
      data-window={window ? JSON.stringify(window) : ""}
      disabled={!url}
      aria-pressed={playing}
      title={url ? title : (emptyTitle ?? "No audio for this side yet")}
      onClick={() => onPlay(url, window)}
      className={cn(
        "inline-flex h-6 items-center gap-0.5 rounded-md border px-1.5 text-[11px] font-bold",
        "transition-colors",
        playing
          ? "border-primary bg-primary text-on-primary"
          : "border-border bg-raised text-secondary hover:border-axis hover:text-primary",
        !url && "cursor-not-allowed opacity-40",
      )}
    >
      {playing ? (
        <Pause className="h-2.5 w-2.5 fill-current" aria-hidden />
      ) : (
        <Play className="h-2.5 w-2.5 fill-current" aria-hidden />
      )}
      {label}
    </button>
  );
}

/**
 * The row's overflow menu.
 *
 * Portalled to `<body>` and positioned from the trigger's own rect, because the
 * script list is an `overflow-auto` container: an absolutely-positioned panel
 * inside it is clipped at the container's edge, which means the menu on the
 * last visible row — the one you are most likely to be reaching for — would
 * open into nothing.
 */
function RowMenu({
  seg,
  open,
  setOpen,
  onCorrect,
  onToggleKeep,
}: {
  seg: Segment;
  open: boolean;
  setOpen: (open: boolean) => void;
  onCorrect: () => void;
  onToggleKeep: () => void;
}) {
  const trigger = useRef<HTMLButtonElement>(null);
  const [at, setAt] = useState<{ top: number; left: number } | null>(null);

  useEffect(() => {
    if (!open) return;
    const rect = trigger.current?.getBoundingClientRect();
    if (rect) setAt({ top: rect.bottom + 4, left: Math.max(8, rect.right - 200) });
    const close = () => setOpen(false);
    // Any scroll moves the trigger out from under the panel, so the panel goes.
    document.addEventListener("mousedown", close);
    window.addEventListener("scroll", close, true);
    document.addEventListener("keydown", onEscape);
    function onEscape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        event.stopPropagation();
        setOpen(false);
      }
    }
    return () => {
      document.removeEventListener("mousedown", close);
      window.removeEventListener("scroll", close, true);
      document.removeEventListener("keydown", onEscape);
    };
  }, [open, setOpen]);

  const item = (label: string, run: () => void) => (
    <button
      type="button"
      onClick={() => {
        setOpen(false);
        run();
      }}
      className="w-full rounded-md px-2 py-1.5 text-left text-[12.5px] text-secondary transition-colors hover:bg-sunken hover:text-primary"
    >
      {label}
    </button>
  );

  return (
    <>
      <button
        ref={trigger}
        type="button"
        aria-label={`More actions for segment ${seg.id}`}
        aria-expanded={open}
        onClick={(event) => {
          event.stopPropagation();
          setOpen(!open);
        }}
        className="inline-flex h-6 w-6 items-center justify-center rounded-md border border-border bg-raised text-secondary transition-colors hover:border-axis hover:text-primary"
      >
        <MoreHorizontal className="h-3 w-3" aria-hidden />
      </button>
      {open && typeof document !== "undefined"
        ? createPortal(
            <div
              role="menu"
              onMouseDown={(event) => event.stopPropagation()}
              style={{ top: at?.top ?? 0, left: at?.left ?? 0 }}
              className="fixed z-50 w-50 rounded-xl border border-border bg-raised p-1 shadow-pop"
            >
              {item("Correct transcript", onCorrect)}
              {item(seg.keep ? "Dub this line" : "Keep original audio", onToggleKeep)}
            </div>,
            document.body,
          )
        : null}
    </>
  );
}
