/**
 * The primitives. Every surface, control and label in the app is one of these —
 * there are no ad-hoc styled divs above this file, which is what keeps three
 * very differently-shaped screens looking like one product.
 *
 * The vocabulary is MamboRambo's, refitted:
 *
 * - **Card** is the page-level container: 2xl radius, hairline border, one big
 *   soft shadow. The editor does not use it a Final-Cut-shaped screen made of
 *   floating cards is unreadable, so its regions are plain bordered panes.
 * - **Eyebrow** is the section label everywhere: tiny, black, uppercase, widely
 *   tracked. It is the single strongest signature of the house style.
 * - **Button**'s primary variant is *ink*, not colour. See App.css for why.
 * - Controls share one recipe (`controlBase`) so an input, a select and a
 *   button line up on the same baseline at the same height.
 * - **Disclosure** and **Popover** are the two ways something leaves the
 *   screen without leaving the app: a named shelf that remembers whether it is
 *   open, and a panel hung off a button for reference material. Between them
 *   they are why the editor fits.
 */

import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ComponentProps,
  type ComponentPropsWithRef,
  type ReactNode,
  type RefObject,
} from "react";
import { ChevronDown, TriangleAlert } from "lucide-react";
import { cn } from "../lib/classNames";
import { STATE_META, type SegmentState } from "../lib/segments";
import badgeUrl from "../assets/mambodubb-badge.svg";

/* ------------------------------------------------------------------ brand */

/**
 * The brand chip the painted mark in a pill, top-left of every page shell.
 *
 * The artwork is the MamboDubb badge SVG a self-contained teal tile with its
 * own rounded corners, so it sits on either theme's pill without keying or
 * blending; the pill stays light in both themes purely so the wordmark beside
 * it keeps one treatment (dark's `--color-primary` is the near-white ink).
 *
 * In dark the chip's hairline takes a wash of the accent (`/25`, deliberately —
 * a saturated 1px outline reads as a focus ring, and focus rings are this same
 * violet). The artwork already carries the brand's colour; the border just
 * agrees with it.
 */
export function BrandChip({ className }: { className?: string }) {
  return (
    <span
      data-brand
      className={cn(
        "inline-flex h-11 items-center gap-2.5 whitespace-nowrap rounded-full border px-3.5 shadow-card",
        "border-border bg-surface dark:border-accent/25 dark:bg-primary",
        className,
      )}
    >
      <img
        src={badgeUrl}
        alt=""
        aria-hidden
        width={22}
        height={22}
        className="h-[22px] w-[22px] rounded-[5px]"
      />
      <span className="text-[11px] font-black uppercase tracking-[0.2em] text-primary dark:text-on-primary">
        MamboDubb
      </span>
    </span>
  );
}

/* --------------------------------------------------------------- surfaces */

/** The page-level container. Roomy radius, hairline border, soft elevation. */
export function Card({ className, ...props }: ComponentProps<"div">) {
  return (
    <div
      className={cn("rounded-2xl border border-border bg-surface shadow-card", className)}
      {...props}
    />
  );
}

/**
 * A band inside a Card. `tone="sunken"` is the footer/summary treatment —
 * MamboRambo's trick of washing the plane colour back over the bottom of a
 * card so the action in it reads as the card's conclusion.
 */
export function CardSection({
  tone = "plain",
  className,
  ...props
}: ComponentProps<"div"> & { tone?: "plain" | "sunken" }) {
  return (
    <div
      className={cn(
        "px-6 py-5 sm:px-7",
        tone === "sunken" && "bg-sunken",
        className,
      )}
      {...props}
    />
  );
}

/** The section label. Tiny, black, uppercase, widely tracked. */
export function Eyebrow({ className, children }: { className?: string; children: ReactNode }) {
  return (
    <span
      className={cn(
        "block text-[10px] font-bold uppercase tracking-[0.2em] text-muted",
        className,
      )}
    >
      {children}
    </span>
  );
}

/** A section heading with an optional icon, used inside cards and the inspector. */
export function SectionLabel({
  icon: Icon,
  children,
  className,
  ...props
}: ComponentProps<"div"> & { icon?: typeof TriangleAlert; children: ReactNode }) {
  return (
    <div className={cn("flex items-center gap-2", className)} {...props}>
      {Icon ? <Icon className="h-3.5 w-3.5 shrink-0 text-muted" aria-hidden /> : null}
      <Eyebrow>{children}</Eyebrow>
    </div>
  );
}

export function Divider({ className }: { className?: string }) {
  return <div className={cn("h-px w-full bg-border", className)} aria-hidden />;
}

/* ---------------------------------------------------------------- buttons */

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger" | "outline";
type ControlSize = "xs" | "sm" | "md" | "lg";

const buttonSize: Record<ControlSize, string> = {
  xs: "h-6 gap-1 rounded-md px-1.5 text-[11px]",
  sm: "h-7 gap-1.5 rounded-md px-2.5 text-[12px]",
  md: "h-9 gap-1.5 rounded-lg px-3 text-[13px]",
  lg: "h-11 gap-2 rounded-lg px-5 text-sm",
};

export function Button({
  variant = "secondary",
  size = "md",
  className,
  ...props
}: ComponentPropsWithRef<"button"> & { variant?: ButtonVariant; size?: ControlSize }) {
  return (
    <button
      type="button"
      className={cn(
        "inline-flex shrink-0 items-center justify-center font-semibold transition-all",
        "active:scale-[0.98] disabled:pointer-events-none disabled:opacity-45",
        buttonSize[size],
        // The one saturated control in the app. `accent` is ink in light the
        // near-black button that shipped and the brand's violet in dark, where
        // a near-white button was the whole reason the theme read as dead.
        variant === "primary" && "bg-accent text-on-accent shadow-card hover:opacity-90",
        variant === "secondary" &&
          "border border-border bg-raised text-primary shadow-card hover:border-axis",
        variant === "outline" &&
          "border border-border bg-transparent text-secondary hover:border-primary hover:text-primary",
        variant === "ghost" && "text-secondary hover:bg-border/60 hover:text-primary",
        variant === "danger" &&
          "border border-critical/40 bg-critical/5 text-critical hover:bg-critical/12",
        className,
      )}
      {...props}
    />
  );
}

/**
 * A segmented row of buttons the zoom stepper, the dub/keep toggle. One
 * border around the group, hairlines between the children.
 */
export function ButtonGroup({ className, ...props }: ComponentProps<"div">) {
  return (
    <div
      className={cn(
        "inline-flex items-stretch overflow-hidden rounded-lg border border-border bg-raised",
        "[&>*]:rounded-none [&>*+*]:border-l [&>*+*]:border-border [&>*]:shadow-none",
        className,
      )}
      {...props}
    />
  );
}

/* ------------------------------------------------------------- selections */

/**
 * The pill. A rounded track holding two or three cells, of which exactly one is
 * filled with ink the shell's nav and the small either/or choices on the
 * import screen are the same control at two sizes.
 *
 * The cell is a className rather than a component because half the callers are
 * `<Link>`s and half are `<button>`s, and a polymorphic `as` prop to paper over
 * that costs more than the one function does.
 */
export function Segmented({ className, ...props }: ComponentProps<"div">) {
  return (
    <div
      className={cn(
        "inline-flex items-center gap-1 rounded-full border border-border bg-surface p-1 shadow-card",
        className,
      )}
      {...props}
    />
  );
}

export function segmentedCell(active: boolean, className?: string): string {
  return cn(
    "inline-flex h-9 items-center gap-1.5 rounded-full px-3.5 transition-colors",
    "text-[11px] font-bold uppercase tracking-[0.14em]",
    // The filled cell is the accent, not ink: "which section am I in" is an
    // active state, and active states are the accent's job in both themes.
    active ? "bg-accent text-on-accent" : "text-muted hover:text-primary",
    className,
  );
}

/**
 * A choice made of rows rather than a `<select>`.
 *
 * Worth the vertical space exactly when the option has a *second line* a
 * sentence saying what picking it does. "Documentary / narrated, factual" is a
 * decision; "Documentary" in a dropdown is a word you have to already know.
 * Selection fills the row with the accent, which is the same "this one is on"
 * the nav pill's filled cell uses one answer to that question, everywhere.
 */
export function OptionList({
  label,
  className,
  ...props
}: ComponentProps<"div"> & { label: string }) {
  return (
    <div role="radiogroup" aria-label={label} className={cn("flex flex-col gap-1.5", className)} {...props} />
  );
}

export function OptionRow({
  icon: Icon,
  label,
  hint,
  selected,
  className,
  ...props
}: ComponentPropsWithRef<"button"> & {
  icon?: typeof TriangleAlert;
  label: string;
  hint?: ReactNode;
  selected: boolean;
}) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={selected}
      className={cn(
        "flex w-full items-center gap-2.5 rounded-xl border px-3 py-2 text-left transition-all",
        "active:scale-[0.99]",
        selected
          ? "border-transparent bg-accent text-on-accent shadow-card"
          : "border-border bg-raised text-primary hover:border-axis",
        className,
      )}
      {...props}
    >
      {Icon ? (
        <Icon
          aria-hidden
          className={cn("h-4 w-4 shrink-0", selected ? "opacity-70" : "text-muted")}
        />
      ) : null}
      <span className="min-w-0 flex-1">
        <span className="block text-[13px] font-semibold leading-tight">{label}</span>
        {hint ? (
          <span
            className={cn(
              "mt-0.5 block text-[11px] leading-tight",
              selected ? "opacity-70" : "text-muted",
            )}
          >
            {hint}
          </span>
        ) : null}
      </span>
    </button>
  );
}

/* ----------------------------------------------------------------- fields */

export function Field({
  label,
  hint,
  children,
  className,
}: {
  label: string;
  hint?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <label className={cn("flex min-w-0 flex-col gap-1.5", className)}>
      <Eyebrow>{label}</Eyebrow>
      {children}
      {hint ? (
        <span className="text-[11px] leading-snug text-muted">{hint}</span>
      ) : null}
    </label>
  );
}

const controlBase =
  "w-full rounded-lg border border-border bg-raised px-3 text-[13px] text-primary outline-none " +
  "transition-colors placeholder:text-muted/70 hover:border-axis " +
  "focus:border-accent disabled:opacity-50";

export function TextInput({ className, ...props }: ComponentProps<"input">) {
  return <input className={cn(controlBase, "h-9", className)} {...props} />;
}

/**
 * A number field that looks like every other field.
 *
 * `<input type=number>` is the one control the platform insists on decorating:
 * it hangs its own spin buttons off the right edge, drawn at the OS's size in
 * the OS's colours, which on a themed form reads as damage. The arrows are
 * suppressed in App.css and the unit goes where they were a unit is what the
 * user actually needed there, and ↑/↓ and the scroll wheel still step the
 * value for anyone who wanted the arrows.
 */
export function NumberInput({
  className,
  suffix,
  ...props
}: ComponentProps<"input"> & { suffix?: string }) {
  return (
    <span className="relative flex min-w-0 items-center">
      <input
        type="number"
        inputMode="numeric"
        className={cn(controlBase, "h-9 tabular-nums", suffix && "pr-9", className)}
        {...props}
      />
      {suffix ? (
        <span className="pointer-events-none absolute right-3 text-[12px] text-muted">
          {suffix}
        </span>
      ) : null}
    </span>
  );
}

/**
 * `autoGrow` is for the fields holding a translated line: they are read far
 * more often than they are typed in, and a line clipped at three rows with the
 * rest behind a scrollbar is the one thing a review surface must not do.
 */
export function TextArea({
  className,
  autoGrow,
  ref,
  ...props
}: ComponentPropsWithRef<"textarea"> & { autoGrow?: boolean }) {
  const fit = (el: HTMLTextAreaElement | null) => {
    if (el && autoGrow) {
      el.style.height = "auto";
      el.style.height = `${el.scrollHeight + 2}px`;
    }
  };
  // The auto-grow ref is the component's own, so a caller that also wants the
  // node (to focus it, to place a caret) has to be merged in rather than
  // allowed to overwrite it `{...props}` used to do exactly that silently.
  const attach = (el: HTMLTextAreaElement | null) => {
    fit(el);
    if (typeof ref === "function") ref(el);
    else if (ref) ref.current = el;
  };
  return (
    <textarea
      ref={attach}
      onInput={(event) => fit(event.currentTarget)}
      className={cn(
        controlBase,
        "resize-y py-2 leading-relaxed",
        autoGrow && "overflow-hidden",
        className,
      )}
      {...props}
    />
  );
}

/**
 * The native select, with the platform arrow suppressed and ours drawn in its
 * place a bare `<select>` is the one control that will not match anything
 * else on the screen.
 */
export function Select({ className, ...props }: ComponentProps<"select">) {
  return (
    <span className="relative flex min-w-0 items-center">
      <select
        className={cn(controlBase, "h-9 cursor-pointer appearance-none pr-8", className)}
        {...props}
      />
      <ChevronDown
        aria-hidden
        className="pointer-events-none absolute right-2.5 h-3.5 w-3.5 text-muted"
      />
    </span>
  );
}

export function Checkbox({ className, ...props }: ComponentProps<"input">) {
  return (
    <input
      type="checkbox"
      className={cn("h-3.5 w-3.5 shrink-0 accent-[var(--color-accent)]", className)}
      {...props}
    />
  );
}

/* ------------------------------------------------------- progressive detail */

/**
 * Where the rest of it lives.
 *
 * The inspector holds about thirty controls and a reviewer touches four of
 * them on a normal segment. The other twenty-six are not unimportant, they are
 * *infrequent* and the fix for infrequent is a shelf with its name on it, not
 * a smaller font.
 *
 * Three rules make a shelf honest, and they are all enforced here rather than
 * left to each caller:
 *
 * 1. **The label is a noun phrase for what is inside**, never "More" or an
 *    ellipsis. If you cannot name the shelf, the things on it do not belong
 *    together.
 * 2. **A shut shelf still says what is on it.** `summary` is one line of the
 *    current values, so the common case "nothing overridden here" is
 *    answered without a click. A shelf that reveals nothing until opened is
 *    just a hidden control.
 * 3. **Closed is the default, and the session remembers otherwise.** Someone
 *    working through fifty segments tuning voices opens "Voice & speaker" once
 *    and it stays open for the rest of the sitting; someone who never touches
 *    it never sees it. `sessionStorage` and not `localStorage` on purpose —
 *    the memory should last exactly as long as the task does.
 */
const DISCLOSURE_KEY = "dubbing-studio.open.";

function readOpen(id: string, fallback: boolean): boolean {
  try {
    const value = window.sessionStorage.getItem(DISCLOSURE_KEY + id);
    return value == null ? fallback : value === "1";
  } catch {
    return fallback;
  }
}

function writeOpen(id: string, open: boolean): void {
  try {
    window.sessionStorage.setItem(DISCLOSURE_KEY + id, open ? "1" : "0");
  } catch {
    // Remembering is a nicety; not remembering is not a failure.
  }
}

export function Disclosure({
  id,
  icon: Icon,
  label,
  summary,
  tone = "neutral",
  defaultOpen = false,
  className,
  children,
}: {
  /** Stable across mounts it is the sessionStorage key. */
  id: string;
  icon?: typeof TriangleAlert;
  label: string;
  /** One line of the current values, shown only while shut. */
  summary?: ReactNode;
  /**
   * How loudly a shut shelf says there is something on it.
   *
   * `warn` is the red one: a failure, and the summary itself goes critical —
   * which is legible as small text, because red measures 4.80:1 on the card.
   * `caution` is the amber one, for a *soft* problem, and it deliberately does
   * not colour the text: `--color-pending` is 3.70:1 in light, fine for a
   * shape and under the gate for an 11px label. So the amber goes on the shelf's
   * icon, where the 3:1 non-text gate applies, and the summary merely takes the
   * weight. Same rule the script row follows for the same concern.
   */
  tone?: "neutral" | "warn" | "caution";
  defaultOpen?: boolean;
  className?: string;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(() => readOpen(id, defaultOpen));

  return (
    <details
      open={open}
      data-tone={tone === "neutral" ? undefined : tone}
      onToggle={(event) => {
        const next = event.currentTarget.open;
        setOpen(next);
        writeOpen(id, next);
      }}
      className={cn("group border-t border-border", className)}
    >
      <summary
        className={cn(
          "flex cursor-pointer list-none items-center gap-2 py-2.5 transition-colors",
          "hover:text-primary [&::-webkit-details-marker]:hidden",
        )}
      >
        {Icon ? (
          <Icon
            className={cn(
              "h-3.5 w-3.5 shrink-0",
              tone === "warn"
                ? "text-critical"
                : tone === "caution"
                  ? "text-warning"
                  : "text-muted",
            )}
            aria-hidden
          />
        ) : null}
        <span className="shrink-0 text-[11px] font-bold uppercase tracking-[0.14em] text-secondary">
          {label}
        </span>
        {summary ? (
          <span
            className={cn(
              "min-w-0 flex-1 truncate text-right text-[11px] group-open:hidden",
              tone === "warn"
                ? "font-semibold text-critical"
                : tone === "caution"
                  ? "font-semibold text-secondary"
                  : "text-muted",
            )}
          >
            {summary}
          </span>
        ) : (
          <span className="flex-1" />
        )}
        <ChevronDown
          aria-hidden
          className="h-3.5 w-3.5 shrink-0 text-muted transition-transform group-open:rotate-180"
        />
      </summary>
      <div className="flex flex-col gap-3 pb-4">{children}</div>
    </details>
  );
}

/**
 * A small panel hung off a button the reference material that used to be
 * printed permanently across the top of the timeline.
 *
 * A legend and a shortcut list are read twice: once on the first day, and once
 * on the day you forget. Leaving them on screen for every other minute of
 * every other day is how a workspace ends up with no room in it. The trigger
 * keeps them one keystroke away and costs one button.
 */
/**
 * Outside-click and Escape close an open panel, and Escape hands focus back.
 *
 * Shared rather than copied because the second panel to want it the queue —
 * has a trigger that cannot be a `Button`, so it could not use `Popover` and
 * would otherwise have reimplemented these two listeners slightly differently.
 * The `stopPropagation` matters: without it Escape also reaches the editor's
 * one keyboard handler, which reads it as "leave the field".
 */
export function useDismissable(
  open: boolean,
  setOpen: (open: boolean) => void,
  wrap: RefObject<HTMLElement | null>,
  trigger?: RefObject<HTMLElement | null>,
): void {
  useEffect(() => {
    if (!open) return;
    const onDown = (event: MouseEvent) => {
      if (!wrap.current?.contains(event.target as Node)) setOpen(false);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.stopPropagation();
      setOpen(false);
      trigger?.current?.focus();
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, setOpen, trigger, wrap]);
}

export function Popover({
  label,
  trigger,
  title,
  align = "right",
  className,
  children,
}: {
  /** The trigger's accessible name. */
  label: string;
  trigger: ReactNode;
  /** Heading inside the panel, and the panel's accessible name. */
  title: string;
  align?: "left" | "right";
  className?: string;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const wrap = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  useDismissable(open, setOpen, wrap, triggerRef);

  return (
    <div className="relative" ref={wrap}>
      <Button
        ref={triggerRef}
        size="sm"
        aria-label={label}
        aria-expanded={open}
        title={label}
        onClick={() => setOpen((v) => !v)}
        className={open ? "border-primary" : undefined}
      >
        {trigger}
      </Button>
      {open ? (
        <div
          role="dialog"
          aria-label={title}
          className={cn(
            "absolute top-full z-40 mt-1.5 w-80 rounded-xl border border-border bg-raised p-3.5 shadow-pop",
            align === "right" ? "right-0" : "left-0",
            className,
          )}
        >
          <Eyebrow className="mb-2.5">{title}</Eyebrow>
          {children}
        </div>
      ) : null}
    </div>
  );
}

/**
 * A button that asks first.
 *
 * `window.confirm` is the wrong control for this app in three separate ways: it
 * is drawn by the OS in the OS's colours, so a themed dark editor pops a white
 * system sheet; it blocks the main thread, so the playhead stops; and it is
 * modal over the *window*, which is far more interruption than "this discards
 * the translation" deserves. What the question actually needs is to be asked
 * next to the thing being asked about.
 *
 * So it is a small panel hung off the button itself, with the consequence
 * spelled out and the confirming verb repeated on the confirm button never
 * "OK", which tells you nothing about what you are agreeing to. Escape and a
 * click outside both mean no, which is the default a destructive question
 * should have.
 *
 * Reserved for the genuinely destructive: re-rendering the preview, splitting
 * and merging (both discard a translation and a clip), and re-translating over
 * a line the user wrote by hand. Cheap reversible things keep/dub, a text
 * edit are just done.
 *
 * ## Which side it opens on
 *
 * Downward, until downward does not fit. The panel is 72 rem-ish of prose hung
 * under a button, and the button that most needs it Split, in the timeline's
 * control cluster sits about thirty pixels off the bottom of the window: the
 * question opened 81px below the fold, so the only destructive gesture in the
 * strip could be armed and never confirmed with a mouse. `side="auto"` (the
 * default, so every other caller gets it for free) measures the panel once it
 * is mounted and flips it above the trigger when there is not room under it and
 * there is more room over it. The measurement is in a layout effect, so the
 * flip happens before the frame is painted rather than as a visible jump.
 */
export function ConfirmButton({
  message,
  confirmLabel,
  onConfirm,
  align = "right",
  side = "auto",
  children,
  ...props
}: ComponentProps<"button"> & {
  variant?: ButtonVariant;
  size?: ControlSize;
  /** What will happen. One sentence, in the indicative. */
  message: ReactNode;
  /** The verb, repeated. Never "OK". */
  confirmLabel: string;
  onConfirm: () => void;
  align?: "left" | "right";
  /** `below` pins it under the trigger; `auto` flips it up when it must. */
  side?: "below" | "auto";
}) {
  const [open, setOpen] = useState(false);
  const [above, setAbove] = useState(false);
  const wrap = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const panel = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (event: MouseEvent) => {
      if (!wrap.current?.contains(event.target as Node)) setOpen(false);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.stopPropagation();
      setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  /*
   * Measured, not guessed. The panel's height depends on the message, which is
   * a sentence a caller composes so a fixed "flip below 200px from the bottom"
   * rule would be wrong for half of them. `useLayoutEffect` runs after the panel
   * is in the DOM and before the browser paints it, which is the whole window in
   * which a position can change without the user seeing it move.
   */
  useLayoutEffect(() => {
    if (!open || side !== "auto") {
      setAbove(false);
      return;
    }
    const rect = trigger.current?.getBoundingClientRect();
    const height = panel.current?.offsetHeight ?? 0;
    if (!rect || height === 0) return;
    const GAP = 6;
    const below = window.innerHeight - rect.bottom;
    setAbove(height + GAP > below && rect.top > below);
  }, [open, side, message]);

  return (
    <div className="relative" ref={wrap}>
      <Button ref={trigger} {...props} aria-expanded={open} onClick={() => setOpen((v) => !v)}>
        {children}
      </Button>
      {open ? (
        <div
          ref={panel}
          role="dialog"
          aria-label={confirmLabel}
          data-confirm-side={above ? "above" : "below"}
          className={cn(
            "absolute z-50 w-72 rounded-xl border border-border bg-raised p-3 shadow-pop",
            above ? "bottom-full mb-1.5" : "top-full mt-1.5",
            align === "right" ? "right-0" : "left-0",
          )}
        >
          <p className="text-[12.5px] leading-relaxed text-secondary">{message}</p>
          <div className="mt-3 flex justify-end gap-2">
            <Button size="sm" variant="ghost" onClick={() => setOpen(false)}>
              Cancel
            </Button>
            <Button
              size="sm"
              variant="primary"
              onClick={() => {
                setOpen(false);
                onConfirm();
              }}
            >
              {confirmLabel}
            </Button>
          </div>
        </div>
      ) : null}
    </div>
  );
}

/* ------------------------------------------------------------- indicators */

type BadgeTone = "neutral" | "good" | "warn" | "bad" | "accent";

const badgeTone: Record<BadgeTone, string> = {
  neutral: "border-border bg-raised text-secondary",
  good: "border-good/35 bg-good/10 text-primary",
  warn: "border-warning/45 bg-warning/12 text-primary",
  bad: "border-critical/35 bg-critical/10 text-primary",
  accent: "border-transparent bg-accent text-on-accent",
};

/**
 * A small labelled chip. Always carries a word the tone is a reinforcement,
 * never the message.
 */
export function Badge({
  tone = "neutral",
  className,
  ...props
}: ComponentProps<"span"> & { tone?: BadgeTone }) {
  return (
    <span
      className={cn(
        "inline-flex shrink-0 items-center gap-1 rounded-md border px-1.5 py-0.5",
        "text-[10px] font-bold uppercase tracking-[0.12em]",
        badgeTone[tone],
        className,
      )}
      {...props}
    />
  );
}

/**
 * A segment state's shape, in its hue.
 *
 * One component for all five places the encoding appears the script row's
 * meta line, the timeline mark, the dub/keep control and the run summary so
 * the shape a reviewer learns on a row is the same shape they meet on the
 * strip. It draws at the size the caller asks for; below about 10px the two
 * outline states stop separating, so 10 is the floor.
 */
export function StateIcon({ state, className }: { state: SegmentState; className?: string }) {
  const meta = STATE_META[state];
  const Icon = meta.icon;
  return (
    // Always decoration: every place it appears the row, the mark, the
    // control, the tally already carries the state as a word or an
    // `aria-label`, so a second announcement of it is noise.
    <Icon
      aria-hidden
      className={cn("shrink-0", meta.filled && "fill-current", className ?? "h-3 w-3")}
      style={{ color: meta.token }}
    />
  );
}

export function Progress({
  value,
  className,
  tone,
}: {
  value: number | null;
  className?: string;
  tone?: string;
}) {
  const indeterminate = value == null;
  return (
    <div
      className={cn("h-1.5 w-full overflow-hidden rounded-full bg-border", className)}
      role="progressbar"
      aria-valuenow={indeterminate ? undefined : Math.round(value * 100)}
    >
      <div
        className="h-full rounded-full transition-[width] duration-300 ease-out"
        style={{
          width: indeterminate ? "35%" : `${Math.min(100, Math.max(0, value * 100))}%`,
          backgroundColor: tone ?? "var(--color-primary)",
        }}
      />
    </div>
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="inline-flex h-4 min-w-4 items-center justify-center rounded border border-border bg-raised px-1 font-sans text-[10px] font-semibold text-muted">
      {children}
    </kbd>
  );
}

/* ---------------------------------------------------------------- failure */

/**
 * The page-level failure card. Big enough to hold a stack trace and quiet
 * enough not to shout a red rule and a tinted wash, not a red slab.
 */
export function ErrorBlock({
  title = "Something went wrong",
  children,
  onDismiss,
  className,
}: {
  title?: string;
  children: ReactNode;
  onDismiss?: () => void;
  className?: string;
}) {
  return (
    <div
      role="alert"
      className={cn(
        "flex gap-3 rounded-2xl border border-critical/30 bg-critical/[0.06] p-5 text-[13px] text-primary",
        className,
      )}
    >
      <TriangleAlert
        aria-hidden
        className="mt-0.5 h-4 w-4 shrink-0"
        style={{ color: "var(--color-critical)" }}
      />
      <div className="min-w-0 flex-1 overflow-auto">
        <p className="text-[11px] font-bold uppercase tracking-[0.16em]">{title}</p>
        <pre className="mt-1.5 font-sans text-[13px] leading-relaxed whitespace-pre-wrap break-words text-secondary">
          {children}
        </pre>
      </div>
      {onDismiss ? (
        <Button variant="ghost" size="sm" onClick={onDismiss}>
          Dismiss
        </Button>
      ) : null}
    </div>
  );
}

/** The workspace's inline failure strip full-bleed, one line, dismissible. */
export function ErrorBar({ message, onDismiss }: { message: string; onDismiss: () => void }) {
  return (
    <div
      role="alert"
      className="flex shrink-0 items-start gap-2 border-b border-critical/35 bg-critical/10 px-4 py-2 text-[13px] text-primary"
    >
      <TriangleAlert
        aria-hidden
        className="mt-0.5 h-3.5 w-3.5 shrink-0"
        style={{ color: "var(--color-critical)" }}
      />
      <span className="min-w-0 flex-1">{message}</span>
      <Button variant="ghost" size="xs" onClick={onDismiss}>
        Dismiss
      </Button>
    </div>
  );
}

/**
 * The empty state.
 *
 * An empty panel is a bug report the user has to write themselves. Every one
 * of these answers three things in order: what this space is, why there is
 * nothing in it, and what to do next the last being the only part that is
 * ever optional, and only when there is genuinely nothing to do but wait.
 */
export function Empty({
  icon: Icon,
  title,
  action,
  className,
  children,
}: {
  icon?: typeof TriangleAlert;
  title?: string;
  action?: ReactNode;
  className?: string;
  children?: ReactNode;
}) {
  return (
    <div
      className={cn(
        "grid h-full place-items-center p-8 text-center text-[13px] leading-relaxed text-muted",
        className,
      )}
    >
      <div className="max-w-72">
        {Icon ? (
          <span className="mx-auto mb-3 grid h-9 w-9 place-items-center rounded-xl border border-border bg-sunken text-muted">
            <Icon className="h-4 w-4" aria-hidden />
          </span>
        ) : null}
        {title ? (
          <p className="text-[13px] font-semibold text-primary">{title}</p>
        ) : null}
        {children ? <div className={cn(title && "mt-1.5")}>{children}</div> : null}
        {action ? <div className="mt-4 flex justify-center gap-2">{action}</div> : null}
      </div>
    </div>
  );
}
