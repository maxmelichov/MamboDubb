/**
 * The primitives. Every surface, control and label in the app is one of these —
 * there are no ad-hoc styled divs above this file, which is what keeps three
 * very differently-shaped screens looking like one product.
 *
 * The vocabulary is MamboRambo's, refitted:
 *
 * - **Card** is the page-level container: 2xl radius, hairline border, one big
 *   soft shadow. **Panel** is its dense cousin for the editor's work surfaces —
 *   same border and surface tokens, tighter radius, no shadow, because a
 *   Premiere-shaped screen made of floating cards is unreadable.
 * - **Eyebrow** is the section label everywhere: tiny, black, uppercase, widely
 *   tracked. It is the single strongest signature of the house style.
 * - **Button**'s primary variant is *ink*, not colour. See App.css for why.
 * - Controls share one recipe (`controlBase`) so an input, a select and a
 *   button line up on the same baseline at the same height.
 */

import { useId, type ComponentProps, type ReactNode } from "react";
import { ChevronDown, TriangleAlert } from "lucide-react";
import { cn } from "../lib/classNames";

/* ------------------------------------------------------------------ brand */

/**
 * The mark: a speech tile with a waveform knocked out of it. One path, drawn
 * in `currentColor`, so it inverts with the theme and needs no asset.
 */
export function LogoMark({ className }: { className?: string }) {
  // Rendered more than once per page (header, boot screen), and an SVG <mask>
  // is referenced by id — so the id has to be per-instance.
  const maskId = useId();
  return (
    <svg
      viewBox="0 0 24 24"
      aria-hidden
      focusable="false"
      className={cn("h-5 w-5 shrink-0", className)}
    >
      <mask id={maskId}>
        <rect width="24" height="24" fill="black" />
        <path
          d="M6 2 H18 A4 4 0 0 1 22 6 V15 A4 4 0 0 1 18 19 H12.5 L8.5 22.4 L9 19 H6 A4 4 0 0 1 2 15 V6 A4 4 0 0 1 6 2 Z"
          fill="white"
        />
        <g fill="black">
          <rect x="6.2" y="8" width="2.4" height="5" rx="1.2" />
          <rect x="9.4" y="6" width="2.4" height="9" rx="1.2" />
          <rect x="12.6" y="4" width="2.4" height="13" rx="1.2" />
          <rect x="15.8" y="7" width="2.4" height="7" rx="1.2" />
        </g>
      </mask>
      <rect width="24" height="24" fill="currentColor" mask={`url(#${maskId})`} />
    </svg>
  );
}

/**
 * The wordmark chip. `compact` drops the border and the box for the workspace
 * header, where the brand has to share a 56px bar with a project title.
 */
export function Brand({ className, compact }: { className?: string; compact?: boolean }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-2.5 whitespace-nowrap font-black uppercase text-primary",
        compact
          ? "text-[10px] tracking-[0.16em]"
          : "h-10 rounded-xl border border-border bg-surface px-3.5 text-[11px] tracking-[0.2em] shadow-card",
        className,
      )}
    >
      <LogoMark className={compact ? "h-4 w-4" : "h-[18px] w-[18px]"} />
      Dubbing Studio
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

/** The dense work surface: same tokens as Card, no shadow, tighter radius. */
export function Panel({ className, ...props }: ComponentProps<"div">) {
  return (
    <div className={cn("rounded-xl border border-border bg-surface", className)} {...props} />
  );
}

export function PanelHeader({ className, ...props }: ComponentProps<"div">) {
  return (
    <div
      className={cn(
        "flex items-center gap-2 border-b border-border bg-sunken px-3 py-2",
        "text-[10px] font-bold uppercase tracking-[0.18em] text-muted",
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
}: ComponentProps<"button"> & { variant?: ButtonVariant; size?: ControlSize }) {
  return (
    <button
      type="button"
      className={cn(
        "inline-flex shrink-0 items-center justify-center font-semibold transition-all",
        "active:scale-[0.98] disabled:pointer-events-none disabled:opacity-45",
        buttonSize[size],
        variant === "primary" && "bg-primary text-on-primary shadow-card hover:opacity-90",
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
 * A segmented row of buttons — the zoom stepper, the dub/keep toggle. One
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
  "focus:border-primary disabled:opacity-50";

export function TextInput({ className, ...props }: ComponentProps<"input">) {
  return <input className={cn(controlBase, "h-9", className)} {...props} />;
}

export function TextArea({ className, ...props }: ComponentProps<"textarea">) {
  return (
    <textarea className={cn(controlBase, "resize-y py-2 leading-relaxed", className)} {...props} />
  );
}

/**
 * The native select, with the platform arrow suppressed and ours drawn in its
 * place — a bare `<select>` is the one control that will not match anything
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
      className={cn("h-3.5 w-3.5 shrink-0 accent-[var(--color-primary)]", className)}
      {...props}
    />
  );
}

/* ------------------------------------------------------------- indicators */

type BadgeTone = "neutral" | "good" | "warn" | "bad" | "accent";

const badgeTone: Record<BadgeTone, string> = {
  neutral: "border-border bg-raised text-secondary",
  good: "border-good/35 bg-good/10 text-primary",
  warn: "border-warning/45 bg-warning/12 text-primary",
  bad: "border-critical/35 bg-critical/10 text-primary",
  accent: "border-transparent bg-primary text-on-primary",
};

/**
 * A small labelled chip. Always carries a word — the tone is a reinforcement,
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
 * A state chip. The colour is a swatch beside the word, never the word's own
 * colour — text always wears text tokens, and the label carries the meaning.
 */
export function StatePill({
  token,
  glyph,
  label,
  className,
}: {
  token: string;
  glyph: string;
  label: string;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex shrink-0 items-center gap-1.5 rounded-md border border-border bg-raised px-1.5 py-0.5",
        "text-[11px] font-medium text-secondary",
        className,
      )}
    >
      <span aria-hidden style={{ color: token }} className="text-[11px] leading-none">
        {glyph}
      </span>
      {label}
    </span>
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
 * enough not to shout — a red rule and a tinted wash, not a red slab.
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

/** The workspace's inline failure strip — full-bleed, one line, dismissible. */
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

export function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="grid h-full place-items-center p-8 text-center text-[13px] leading-relaxed text-muted">
      <div className="max-w-64">{children}</div>
    </div>
  );
}
