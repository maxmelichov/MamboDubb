/**
 * The run, behind one button: its health when the rail is busy with a line,
 * its metadata, its editable options and its files.
 */

import { useState } from "react";
import { ExternalLink, FolderOpen, MoreHorizontal } from "lucide-react";
import { Button, Eyebrow, Popover, Select, TextArea } from "./ui";
import { GapList, StateTally } from "./RunSummary";
import { openRunFile } from "./runFiles";
import { isDesktop } from "../lib/desktop";
import type { SegmentState, Span } from "../lib/segments";
import type { ProjectDetail, ProjectOptionsPatch } from "../lib/types";

/**
 * The run, behind one button.
 *
 * Three things that are true about a run and are not per-line: how many
 * segments are in what state, where there is audible speech that nothing
 * covers, and where the files are. None of them changes while you work through
 * a line, and none of them earns permanent chrome but the middle one is the
 * highest-value readout the report produces, so every gap is a button that
 * seeks to it.
 *
 * It shares its tally and its gap list with the rail's own summary same
 * answers, one for the reviewer who has a line open and one for the reviewer
 * who does not and that sharing is why it now *drops* them when nothing is
 * selected. The rail is showing the run summary at exactly that moment, three
 * inches to the left and permanently, so the menu was a second copy of a panel
 * already on screen: opening it moved nothing forward. When a line is selected
 * the rail belongs to that line, and the menu is the only way back to the run,
 * so it carries the lot.
 *
 * What neither of them brought back from the 194-line original are the coverage
 * bars and the drift and speed stats: those were a report rendered twice, once
 * here and once in `report.json`.
 */
export function RunMenu({
  project,
  name,
  counts,
  showHealth,
  onSeek,
  onHighlightGap,
  onSaveOptions,
}: {
  project: ProjectDetail | null;
  name: string;
  counts: Record<SegmentState, number>;
  /** False when the rail is already showing the run summary. */
  showHealth: boolean;
  onSeek: (time: number) => void;
  onHighlightGap?: (span: Span | null) => void;
  onSaveOptions: (patch: ProjectOptionsPatch) => Promise<void>;
}) {
  const gaps = project?.report?.uncovered_audible ?? [];
  const preview = project?.outputs.preview;
  const srt = project?.outputs.srt;

  return (
    <Popover
      label={showHealth ? "Run health and files" : "Run files and options"}
      title={showHealth ? "Run health" : "This run"}
      trigger={<MoreHorizontal className="h-3.5 w-3.5" />}
      className="w-[21rem]"
    >
      {showHealth ? (
        <div data-run-health>
          <StateTally counts={counts} />
          <GapList
            gaps={gaps}
            onSeek={onSeek}
            onHighlight={onHighlightGap}
            stale={project?.report?.stale}
            className="mt-3.5"
          />
        </div>
      ) : null}

      {/* The panel's own title is "This run" when the health half is gone, so
          the section label would be the same words twice, four pixels apart. */}
      {showHealth ? <Eyebrow className="mt-3.5 mb-1.5">This run</Eyebrow> : null}
      <dl className="flex flex-col gap-0.5 text-[11px] text-muted">
        <div className="flex gap-2">
          <dt className="shrink-0">languages</dt>
          <dd className="ml-auto font-mono text-secondary">
            {project?.source.src_lang} → {project?.source.tgt_lang}
          </dd>
        </div>
        <div className="flex gap-2">
          <dt className="shrink-0">run dir</dt>
          <dd className="ml-auto truncate font-mono text-secondary">{name}</dd>
        </div>
        {project?.source.transcript_origin ? (
          <div className="flex gap-2">
            <dt className="shrink-0">transcript</dt>
            <dd className="ml-auto font-mono text-secondary">
              {project.source.transcript_origin}
            </dd>
          </div>
        ) : null}
      </dl>

      <RunOptions source={project?.source ?? null} onSave={onSaveOptions} />

      {/*
        What the run produced, by name, each one a click away.
        It used to be a single desktop-only "Show preview.mp4 in Finder" button
        that handed `reveal_path` the manifest's *run-relative* path which
        the shell resolves against its own working directory and refuses,
        because nothing is there. Both rows go through `openRunFile`, which
        composes the absolute path in the shell and opens the served URL in a
        browser, so the list is useful in both and the subtitles the other
        thing a finished run is for are no longer unreachable.
      */}
      {preview || srt ? (
        <>
          <Eyebrow className="mt-3.5 mb-1.5">Files</Eyebrow>
          <div className="flex flex-col gap-1">
            {preview ? <FileRow name={name} path={preview} label="Preview video" /> : null}
            {srt ? <FileRow name={name} path={srt} label="Subtitles (.srt)" /> : null}
          </div>
        </>
      ) : null}
    </Popover>
  );
}

/**
 * The three run options that are still a decision, editable in place.
 *
 * Genre, register and context were chosen once on the import screen and then
 * became unreachable a `--genre documentary` picked in ten seconds before the
 * first line had been read, binding every re-translate for the rest of the
 * project's life. Nothing about them is structural: all three are inputs to the
 * *translator*, so nothing already fetched, transcribed or segmented depends on
 * them, which is exactly why they can be changed and the source and the language
 * pair cannot (the import screen now says so, in those words).
 *
 * Saving enqueues nothing. `PATCH /api/projects/{name}` writes them and returns;
 * silently re-translating two hundred lines because a dropdown moved would be a
 * worse surprise than the wait. The note under the group is therefore not a
 * disclaimer, it is the contract: they take effect the next time translation
 * runs, and the buttons that run it are on this screen already.
 */
function RunOptions({
  source,
  onSave,
}: {
  source: ProjectDetail["source"] | null;
  onSave: (patch: ProjectOptionsPatch) => Promise<void>;
}) {
  const [editingContext, setEditingContext] = useState(false);
  const [draft, setDraft] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const context = (source?.context ?? "").trim();

  /*
   * The refusal is shown here, not swallowed and not thrown at the window.
   *
   * The server 409s while a job runs run options are read when a job starts,
   * so changing one mid-render would be a setting the user then watches not
   * happen. A dropdown that snapped back with no sentence would be the same
   * lie in the other direction, and the field stays open on failure so the
   * note nobody managed to save is still there to try again with.
   */
  const save = async (patch: ProjectOptionsPatch) => {
    setSaving(true);
    setError(null);
    try {
      await onSave(patch);
      setEditingContext(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  if (!source) return null;

  return (
    <div data-run-options>
      <Eyebrow className="mt-3.5 mb-1.5">Run options</Eyebrow>

      <div className="flex flex-col gap-1.5">
        {/*
          Two-value choices, so the select IS the edit: an Edit button in front
          of a two-option dropdown is a click spent to reach a click.
        */}
        <OptionSelect
          label="Genre"
          value={source.genre ?? "documentary"}
          disabled={saving}
          options={[
            ["documentary", "Documentary"],
            ["movie", "Movie"],
          ]}
          onChange={(value) => void save({ genre: value as ProjectOptionsPatch["genre"] })}
        />
        <OptionSelect
          label="Register"
          value={source.register ?? "narration"}
          disabled={saving}
          options={[
            ["narration", "Narration"],
            ["dialogue", "Dialogue"],
          ]}
          onChange={(value) => void save({ register: value as ProjectOptionsPatch["register"] })}
        />
      </div>

      {/* Context is prose, so it gets a field and an explicit commit a
          textarea that saved on blur would fire on every accidental click out
          of a note somebody was still writing. */}
      <div className="mt-2">
        <div className="flex items-baseline gap-2">
          <span className="shrink-0 text-[11px] text-muted">context</span>
          {!editingContext ? (
            <button
              type="button"
              data-edit-context
              disabled={saving}
              onClick={() => {
                setDraft(context);
                setEditingContext(true);
              }}
              className="ml-auto rounded text-[11px] font-semibold text-secondary underline underline-offset-2 transition-colors hover:text-primary disabled:opacity-50"
            >
              {context ? "Edit" : "Add"}
            </button>
          ) : null}
        </div>

        {editingContext ? (
          <>
            <TextArea
              autoFocus
              rows={4}
              aria-label="Context"
              className="mt-1 text-[12px]"
              value={draft}
              disabled={saving}
              onChange={(event) => setDraft(event.currentTarget.value)}
            />
            <div className="mt-1.5 flex justify-end gap-2">
              <Button size="sm" variant="ghost" onClick={() => setEditingContext(false)}>
                Cancel
              </Button>
              <Button
                size="sm"
                variant="primary"
                data-save-context
                disabled={saving}
                // The empty string is how the note is *removed*; the server
                // reads it as a clear, which is why the draft is sent as typed
                // rather than skipped when it is blank.
                onClick={() => void save({ context: draft.trim() })}
              >
                Save
              </Button>
            </div>
          </>
        ) : (
          <p className="mt-0.5 text-[11.5px] leading-relaxed text-secondary">
            {context || <span className="text-muted">none: names and spellings go here</span>}
          </p>
        )}
      </div>

      {error ? (
        <p
          data-options-error
          className="mt-2 rounded-lg border border-critical/35 bg-critical/[0.06] px-2 py-1.5 text-[11px] leading-relaxed text-secondary"
        >
          {error}
        </p>
      ) : null}

      <p className="mt-2 text-[11px] leading-relaxed text-muted">
        Applies to the next translate or render. Nothing already translated changes on its own.
      </p>
    </div>
  );
}

/** One labelled two-value run option. The select is the whole control. */
function OptionSelect({
  label,
  value,
  options,
  disabled,
  onChange,
}: {
  label: string;
  value: string;
  options: [string, string][];
  disabled: boolean;
  onChange: (value: string) => void;
}) {
  return (
    <label className="flex items-center gap-2">
      <span className="shrink-0 text-[11px] text-muted">{label.toLowerCase()}</span>
      <Select
        aria-label={label}
        value={value}
        disabled={disabled}
        className="ml-auto h-7 w-[9.5rem] text-[12px]"
        onChange={(event) => onChange(event.currentTarget.value)}
      >
        {options.map(([key, text]) => (
          <option key={key} value={key}>
            {text}
          </option>
        ))}
      </Select>
    </label>
  );
}

function FileRow({ name, path, label }: { name: string; path: string; label: string }) {
  return (
    <button
      type="button"
      data-run-file={path}
      onClick={() => void openRunFile(name, path)}
      title={isDesktop() ? `Show ${path} in Finder` : `Open ${path} in a new tab`}
      className="flex w-full items-center gap-2 rounded-lg border border-border bg-raised px-2 py-1.5 text-left text-[12.5px] text-primary transition-colors hover:border-axis hover:bg-sunken"
    >
      {isDesktop() ? (
        <FolderOpen className="h-3.5 w-3.5 shrink-0 text-muted" aria-hidden />
      ) : (
        <ExternalLink className="h-3.5 w-3.5 shrink-0 text-muted" aria-hidden />
      )}
      {label}
      <span className="ml-auto truncate font-mono text-[11px] text-muted">{path}</span>
    </button>
  );
}
