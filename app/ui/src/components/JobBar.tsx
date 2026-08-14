/**
 * The one-job strip. At most one pipeline job runs process-wide, so this is a
 * single line rather than a queue panel — with the queue depth beside it when
 * more are waiting.
 *
 * It sits on the sunken tone so it reads as a status rail against the surfaces
 * above and below it, and it is the one place in the workspace that animates.
 *
 * Two things it used to do and no longer does. It drew a StageTrack, which the
 * preview placeholder was already drawing two inches below it — the same nine
 * dots, twice on one screen. And the bar and the percentage disagreed: the bar
 * read `stage?.progress ?? job.progress` while the number read only
 * `stage?.progress`, so a job reporting job-level progress showed a bar that
 * moved next to a percentage stuck on "—". One value, read once.
 */

import { Loader2, WifiOff } from "lucide-react";
import { Button, Progress } from "./ui";
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
}: {
  job: Job | null;
  queued: number;
  stage: StageProgress | null;
  connected: boolean;
  onCancel: (id: string) => void;
}) {
  if (!job && connected) return null;

  const progress = stage?.progress ?? job?.progress ?? null;

  return (
    /*
      One line, always: `flex-wrap` let a long stage message or a dropped
      stream wrap to a second row, which moved the whole editor down by 28px
      mid-job — the script under the cursor jumping while you read it. Both
      variable-length parts truncate instead, and the strip has a fixed height
      so its appearance is the only layout change it can cause.
    */
    <div
      data-job-strip
      className="flex h-8 shrink-0 items-center gap-3 border-b border-border bg-sunken px-4 text-[12.5px]"
    >
      {/*
        A dropped stream is not a dead app — every edit still goes over plain
        HTTP and still saves. The only thing that stops is progress arriving on
        its own, so the strip says exactly that. It used to offer a "Refresh
        now" button, which was the third copy of the header's Reload; the
        stream reconnects by itself and any edit re-reads the run anyway.
      */}
      {!connected ? (
        <span className="flex min-w-0 flex-1 items-center gap-1.5 font-medium text-secondary">
          <WifiOff className="h-3.5 w-3.5 shrink-0" aria-hidden />
          <span className="min-w-0 truncate">
            Progress stream disconnected — retrying. Edits still save; progress will not
            update on its own.
          </span>
        </span>
      ) : null}

      {job ? (
        <>
          <span className="inline-flex shrink-0 items-center gap-1.5 text-[11px] font-bold uppercase tracking-[0.14em] text-primary">
            <Loader2 className="h-3 w-3 animate-spin" aria-hidden />
            {KIND_LABEL[job.kind]}
          </span>

          <span className="min-w-0 flex-1 truncate text-secondary">
            {stage?.message ?? job.message ?? ""}
          </span>
          <Progress value={progress} className="w-40 shrink-0" />
          <span className="w-9 shrink-0 text-right font-mono text-[11px] tabular-nums text-muted">
            {progress != null ? `${Math.round(progress * 100)}%` : "—"}
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
