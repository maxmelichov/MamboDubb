/**
 * The one-job strip. At most one pipeline job runs process-wide, so this is a
 * single line rather than a queue panel — with the queue depth beside it when
 * more are waiting.
 *
 * It sits on the sunken tone so it reads as a status rail against the surfaces
 * above and below it, and it is the one place in the workspace that animates.
 */

import { Loader2, WifiOff } from "lucide-react";
import { Button, Progress } from "./ui";
import { STAGES } from "../lib/types";
import type { Job, Stage } from "../lib/types";
import type { StageProgress } from "../lib/useProject";
import { cn } from "../lib/classNames";

const KIND_LABEL: Record<Job["kind"], string> = {
  run: "Dubbing",
  retranslate: "Re-translating",
  resynthesize: "Re-voicing",
  render: "Rendering preview",
};

export function JobBar({
  job,
  queued,
  stage,
  connected,
  onCancel,
}: {
  job: Job | null;
  queued: number;
  stage: StageProgress | null;
  connected: boolean;
  onCancel: (id: string) => void;
}) {
  if (!job && connected) return null;

  return (
    <div className="flex shrink-0 items-center gap-3 border-b border-border bg-sunken px-4 py-2 text-[12px]">
      {!connected ? (
        <span className="inline-flex items-center gap-1.5 font-medium text-secondary">
          <WifiOff className="h-3.5 w-3.5" aria-hidden />
          Progress stream disconnected — reconnecting
        </span>
      ) : null}

      {job ? (
        <>
          <span className="inline-flex shrink-0 items-center gap-1.5 text-[10px] font-bold uppercase tracking-[0.14em] text-primary">
            <Loader2 className="h-3 w-3 animate-spin" aria-hidden />
            {KIND_LABEL[job.kind]}
          </span>
          {stage ? <StageTrack current={stage.stage} /> : null}
          <span className="min-w-0 flex-1 truncate text-secondary">
            {stage?.message ?? job.message ?? ""}
          </span>
          <Progress value={stage?.progress ?? job.progress} className="w-40 shrink-0" />
          <span className="w-9 shrink-0 text-right font-mono tabular-nums text-muted">
            {stage?.progress != null ? `${Math.round(stage.progress * 100)}%` : "—"}
          </span>
          {queued > 0 ? (
            <span className="shrink-0 text-[11px] text-muted">+{queued} queued</span>
          ) : null}
          <Button variant="ghost" size="xs" onClick={() => onCancel(job.id)}>
            Cancel
          </Button>
        </>
      ) : null}
    </div>
  );
}

/** Stage position, as a row of dots — the pipeline is fixed and ordered. */
function StageTrack({ current }: { current: Stage }) {
  const index = STAGES.indexOf(current);
  return (
    <span
      className="flex shrink-0 items-center gap-1 rounded-md border border-border bg-raised px-2 py-1"
      title={`stage ${current}`}
    >
      {STAGES.map((stage, i) => (
        <span
          key={stage}
          aria-hidden
          className={cn(
            "h-1.5 w-1.5 rounded-full transition-colors",
            i < index
              ? "bg-primary"
              : i === index
                ? "bg-primary ring-2 ring-primary/25"
                : "bg-border",
          )}
        />
      ))}
      <span className="ml-1 text-[11px] font-semibold text-secondary">{current}</span>
    </span>
  );
}
