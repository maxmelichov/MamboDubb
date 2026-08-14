import type { ComponentProps, ReactNode } from "react";
import { cn } from "../lib/classNames";

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";

export function Button({
  variant = "secondary",
  className,
  ...props
}: ComponentProps<"button"> & { variant?: ButtonVariant }) {
  return (
    <button
      type="button"
      className={cn(
        "inline-flex items-center justify-center gap-1.5 rounded-md px-2.5 py-1.5 text-[13px] font-medium transition-colors",
        "disabled:cursor-not-allowed disabled:opacity-45",
        variant === "primary" && "bg-brand text-white hover:brightness-110",
        variant === "secondary" &&
          "border border-border bg-raised text-primary hover:border-axis",
        variant === "ghost" && "text-secondary hover:bg-border/50 hover:text-primary",
        variant === "danger" && "border border-critical/40 text-critical hover:bg-critical/10",
        className,
      )}
      {...props}
    />
  );
}

export function Panel({ className, ...props }: ComponentProps<"div">) {
  return (
    <div
      className={cn("rounded-lg border border-border bg-surface", className)}
      {...props}
    />
  );
}

export function PanelHeader({ className, ...props }: ComponentProps<"div">) {
  return (
    <div
      className={cn(
        "flex items-center gap-2 border-b border-border px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.12em] text-muted",
        className,
      )}
      {...props}
    />
  );
}

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
    <label className={cn("flex flex-col gap-1", className)}>
      <span className="text-[11px] font-semibold uppercase tracking-[0.1em] text-muted">
        {label}
      </span>
      {children}
      {hint ? <span className="text-[11px] leading-snug text-muted">{hint}</span> : null}
    </label>
  );
}

const fieldBase =
  "w-full rounded-md border border-border bg-raised px-2 py-1.5 text-[13px] outline-none transition-colors placeholder:text-muted focus:border-brand";

export function TextInput({ className, ...props }: ComponentProps<"input">) {
  return <input className={cn(fieldBase, className)} {...props} />;
}

export function TextArea({ className, ...props }: ComponentProps<"textarea">) {
  return <textarea className={cn(fieldBase, "resize-y leading-relaxed", className)} {...props} />;
}

export function Select({ className, ...props }: ComponentProps<"select">) {
  return <select className={cn(fieldBase, "pr-6", className)} {...props} />;
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
        "inline-flex items-center gap-1.5 rounded border border-border bg-raised px-1.5 py-0.5 text-[11px] font-medium text-secondary",
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

export function Progress({ value, className }: { value: number | null; className?: string }) {
  const indeterminate = value == null;
  return (
    <div
      className={cn("h-1 w-full overflow-hidden rounded-full bg-border", className)}
      role="progressbar"
      aria-valuenow={indeterminate ? undefined : Math.round(value * 100)}
    >
      <div
        className={cn("h-full rounded-full bg-brand transition-[width] duration-200")}
        style={{ width: indeterminate ? "35%" : `${Math.min(100, value * 100)}%` }}
      />
    </div>
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="rounded border border-border bg-raised px-1 py-px font-sans text-[10px] text-muted">
      {children}
    </kbd>
  );
}

export function ErrorBar({ message, onDismiss }: { message: string; onDismiss: () => void }) {
  return (
    <div
      role="alert"
      className="flex items-start gap-2 border-b border-critical/40 bg-critical/10 px-3 py-2 text-[13px] text-primary"
    >
      <span aria-hidden style={{ color: "var(--color-critical)" }}>
        ✕
      </span>
      <span className="flex-1">{message}</span>
      <Button variant="ghost" className="px-1.5 py-0.5 text-[11px]" onClick={onDismiss}>
        Dismiss
      </Button>
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="grid h-full place-items-center p-8 text-center text-[13px] text-muted">
      {children}
    </div>
  );
}
