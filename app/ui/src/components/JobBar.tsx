/**
 * The one-job strip. At most one pipeline job runs process-wide, so this is a
 * single line rather than a queue panel — with the queue depth beside it when
 * more are waiting.
 */

import { WifiOff } from "lucide-react";
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
    <div className="flex items-center gap-3 border-b border-border bg-surface px-3 py-1.5 text-[12px]">
      {!connected ? (
        <span className="inline-flex items-center gap-1.5 text-secondary">
          <WifiOff className="h-3.5 w-3.5" />
          Progress stream disconnected — reconnecting
        </span>
      ) : null}

      {job ? (
        <>
          <span className="font-medium">{KIND_LABEL[job.kind]}</span>
          {stage ? <StageTrack current={stage.stage} /> : null}
          <span className="truncate text-secondary">{stage?.message ?? job.message ?? ""}</span>
          <Progress value={stage?.progress ?? job.progress} className="ml-auto w-40" />
          <span className="w-9 text-right tabular-nums text-muted">
            {stage?.progress != null ? `${Math.round(stage.progress * 100)}%` : "—"}
          </span>
          {queued > 0 ? <span className="text-muted">+{queued} queued</span> : null}
          <Button variant="ghost" className="px-1.5 py-0.5" onClick={() => onCancel(job.id)}>
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
    <span className="flex items-center gap-1" title={`stage ${current}`}>
      {STAGES.map((stage, i) => (
        <span
          key={stage}
          aria-hidden
          className={cn(
            "h-1.5 w-1.5 rounded-full",
            i < index ? "bg-brand" : i === index ? "bg-brand ring-2 ring-brand/30" : "bg-border",
          )}
        />
      ))}
      <span className="ml-1 text-secondary">{current}</span>
    </span>
  );
}
