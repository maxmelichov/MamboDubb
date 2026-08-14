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
import { StageTrack } from "./StageTrack";
import type { Job } from "../lib/types";
import type { StageProgress } from "../lib/useProject";

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
  onReload,
}: {
  job: Job | null;
  queued: number;
  stage: StageProgress | null;
  connected: boolean;
  onCancel: (id: string) => void;
  onReload?: () => void;
}) {
  if (!job && connected) return null;

  return (
    <div className="flex shrink-0 flex-wrap items-center gap-3 border-b border-border bg-sunken px-4 py-2 text-[12px]">
      {/*
        A dropped stream is not a dead app — every edit still goes over plain
        HTTP and still saves. The only thing that stops is progress arriving on
        its own, so the strip says exactly that and offers the manual version
        of what the stream would have done.
      */}
      {!connected ? (
        <span className="inline-flex items-center gap-1.5 font-medium text-secondary">
          <WifiOff className="h-3.5 w-3.5 shrink-0" aria-hidden />
          Progress stream disconnected — retrying. Edits still save; progress will not
          update on its own.
          {onReload ? (
            <Button variant="ghost" size="xs" onClick={onReload}>
              Refresh now
            </Button>
          ) : null}
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
