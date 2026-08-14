/**
 * Everything about one segment, in the order a reviewer works through it:
 * hear it, read it, fix it, re-run it.
 *
 * The two classes of control are deliberately not interleaved. Text, speaker,
 * keep and the language tags are no-model edits and save immediately, even
 * while a job is running. Re-translate and re-voice need the single job slot,
 * so they sit together at the bottom behind a queue.
 */

import { useEffect, useState } from "react";
import { Languages, Lock, Merge, Scissors, TriangleAlert, Volume2 } from "lucide-react";
import { ABPlayer } from "./ABPlayer";
import { Button, Field, Kbd, Select, StatePill, TextArea, TextInput } from "./ui";
import { cn } from "../lib/classNames";
import { duration as fmtDuration, percent, timecode } from "../lib/format";
import {
  STATE_META,
  neighbours,
  placementConcern,
  segmentState,
  verifyConcern,
} from "../lib/segments";
import type { Segment, SegmentPatch, TtsOpts } from "../lib/types";

const LANGS = ["", "he", "en", "ar", "ru", "fr", "es", "de"];

export function SegmentInspector({
  seg,
  segments,
  speakers,
  busy,
  playhead,
  onPatch,
  onSplit,
  onMerge,
  onRetranslate,
  onResynthesize,
}: {
  seg: Segment;
  segments: Segment[];
  speakers: string[];
  busy: boolean;
  playhead: number;
  onPatch: (patch: SegmentPatch) => void;
  onSplit: (at: number) => void;
  onMerge: (uidB: string) => void;
  onRetranslate: () => void;
  onResynthesize: () => void;
}) {
  const state = segmentState(seg);
  const meta = STATE_META[state];
  const { prev, next } = neighbours(segments, seg.uid);
  const concern = verifyConcern(seg);
  const placement = placementConcern(seg);

  // Text is drafted locally and committed on blur; a PATCH per keystroke would
  // rewrite the manifest for every letter.
  const [source, setSource] = useState(seg.text);
  const [target, setTarget] = useState(seg.text_en ?? "");
  useEffect(() => {
    setSource(seg.text);
    setTarget(seg.text_en ?? "");
  }, [seg.uid, seg.text, seg.text_en]);

  const splitPoint = playhead > seg.start + 0.05 && playhead < seg.end - 0.05 ? playhead : null;

  return (
    <div className="flex h-full flex-col overflow-y-auto">
      <header className="sticky top-0 z-10 flex flex-wrap items-center gap-2 border-b border-border bg-surface px-3 py-2">
        <span className="text-[13px] font-semibold tabular-nums">#{seg.id}</span>
        <StatePill token={meta.token} glyph={meta.glyph} label={meta.label} />
        <span className="text-[12px] tabular-nums text-secondary">
          {timecode(seg.start)} – {timecode(seg.end)}
        </span>
        <span className="text-[11px] tabular-nums text-muted">
          {fmtDuration(seg.end - seg.start)}
        </span>
        {seg.locked ? (
          <span className="inline-flex items-center gap-1 text-[11px] text-muted">
            <Lock className="h-3 w-3" />
            hand-edited
          </span>
        ) : null}
        <span className="ml-auto font-mono text-[10px] text-muted">{seg.uid}</span>
      </header>

      <div className="flex flex-col gap-3 p-3">
        <ABPlayer seg={seg} />

        <div className="grid grid-cols-2 gap-2">
          <Field label="Speaker">
            <div className="flex gap-1">
              <Select
                value={speakers.includes(seg.speaker) ? seg.speaker : ""}
                onChange={(event) => onPatch({ speaker: event.currentTarget.value })}
              >
                {!speakers.includes(seg.speaker) ? <option value="">{seg.speaker}</option> : null}
                {speakers.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </Select>
            </div>
          </Field>

          <Field label="This segment">
            {/*
              Keep is a manual decision once the user makes it — `mark_keep`
              must never re-decide it, which is what the `locked` flag on the
              PATCH is for.
            */}
            <div className="flex overflow-hidden rounded-md border border-border">
              <Toggle
                active={!seg.keep}
                onClick={() => onPatch({ keep: false })}
                label="Dub it"
                glyph={STATE_META.dubbed.glyph}
                token={STATE_META.dubbed.token}
              />
              <Toggle
                active={seg.keep}
                onClick={() => onPatch({ keep: true, keep_reason: "manual" })}
                label="Keep original"
                glyph={STATE_META.kept.glyph}
                token={STATE_META.kept.token}
              />
            </div>
          </Field>
        </div>

        <Field
          label="Source text"
          hint={
            seg.keep_reason ? (
              <>
                kept because <code>{seg.keep_reason}</code>
              </>
            ) : null
          }
        >
          <TextArea
            className="auto-dir min-h-20"
            value={source}
            onChange={(event) => setSource(event.currentTarget.value)}
            onBlur={() => source !== seg.text && onPatch({ text: source })}
          />
        </Field>

        <Field
          label="Target text"
          hint={
            <span className="flex items-center gap-2">
              <Kbd>tab</Kbd> to save
              {seg.locked?.text_en ? " · locked, a re-run will not overwrite it" : null}
            </span>
          }
        >
          <TextArea
            className="auto-dir min-h-20"
            value={target}
            placeholder="not translated yet"
            onChange={(event) => setTarget(event.currentTarget.value)}
            onBlur={() => target !== (seg.text_en ?? "") && onPatch({ text_en: target })}
          />
        </Field>

        {/*
          Per-segment language tags: "this bit is Arabic, translate it to
          English". Empty means "inherit the run's languages".
        */}
        <div className="grid grid-cols-2 gap-2">
          <Field label="Spoken language" hint="overrides the run's source language">
            <Select
              value={seg.src_lang ?? ""}
              onChange={(event) => onPatch({ src_lang: event.currentTarget.value || null })}
            >
              {LANGS.map((lang) => (
                <option key={lang || "inherit"} value={lang}>
                  {lang || "inherit"}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="Translate into" hint="overrides the run's target language">
            <Select
              value={seg.tgt_lang ?? ""}
              onChange={(event) => onPatch({ tgt_lang: event.currentTarget.value || null })}
            >
              {LANGS.map((lang) => (
                <option key={lang || "inherit"} value={lang}>
                  {lang || "inherit"}
                </option>
              ))}
            </Select>
          </Field>
        </div>

        <TtsOptions
          opts={seg.tts_opts ?? null}
          onChange={(tts_opts) => onPatch({ tts_opts })}
        />

        <Verification seg={seg} concern={concern} placement={placement} />

        <div className="flex flex-col gap-2 border-t border-border pt-3">
          <span className="text-[11px] font-semibold uppercase tracking-[0.1em] text-muted">
            Model actions — these queue
          </span>
          <div className="flex flex-wrap gap-1.5">
            <Button
              onClick={() => {
                if (
                  seg.locked?.text_en &&
                  !window.confirm("This line was edited by hand. Re-translate and discard it?")
                ) {
                  return;
                }
                onRetranslate();
              }}
              disabled={busy}
            >
              <Languages className="h-3.5 w-3.5" />
              Re-translate
            </Button>
            <Button onClick={onResynthesize} disabled={busy || seg.keep}>
              <Volume2 className="h-3.5 w-3.5" />
              Re-voice
            </Button>
          </div>

          <span className="mt-1 text-[11px] font-semibold uppercase tracking-[0.1em] text-muted">
            Structure
          </span>
          <div className="flex flex-wrap gap-1.5">
            <Button
              disabled={splitPoint == null}
              title={
                splitPoint == null
                  ? "Move the playhead inside this segment to split it"
                  : `Split at ${timecode(splitPoint)}`
              }
              onClick={() => {
                if (splitPoint == null) return;
                if (!window.confirm(`Split at ${timecode(splitPoint)}? Both halves lose their translation and clip.`)) return;
                onSplit(splitPoint);
              }}
            >
              <Scissors className="h-3.5 w-3.5" />
              Split at playhead
            </Button>
            <Button
              disabled={!prev || prev.speaker !== seg.speaker}
              title={
                prev && prev.speaker !== seg.speaker
                  ? "Only adjacent segments with the same speaker can merge"
                  : undefined
              }
              onClick={() => {
                if (!prev) return;
                if (!window.confirm("Merge with the previous segment? The merged line loses its translation and clip.")) return;
                onMerge(prev.uid);
              }}
            >
              <Merge className="h-3.5 w-3.5" />
              Merge ←
            </Button>
            <Button
              disabled={!next || next.speaker !== seg.speaker}
              title={
                next && next.speaker !== seg.speaker
                  ? "Only adjacent segments with the same speaker can merge"
                  : undefined
              }
              onClick={() => {
                if (!next) return;
                if (!window.confirm("Merge with the next segment? The merged line loses its translation and clip.")) return;
                onMerge(next.uid);
              }}
            >
              <Merge className="h-3.5 w-3.5" />
              Merge →
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

function Toggle({
  active,
  onClick,
  label,
  glyph,
  token,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
  glyph: string;
  token: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={cn(
        "flex flex-1 items-center justify-center gap-1.5 px-2 py-1.5 text-[12px] transition-colors",
        active ? "bg-brand/15 font-medium text-primary" : "bg-raised text-secondary hover:bg-border/50",
      )}
    >
      <span aria-hidden style={{ color: token }}>
        {glyph}
      </span>
      {label}
    </button>
  );
}

/** Per-segment synthesis overrides; empty fields inherit the run's defaults. */
function TtsOptions({
  opts,
  onChange,
}: {
  opts: TtsOpts | null;
  onChange: (opts: TtsOpts) => void;
}) {
  const set = (patch: TtsOpts) => onChange({ ...(opts ?? {}), ...patch });
  return (
    <details className="rounded-md border border-border bg-raised px-2 py-1.5" open={Boolean(opts)}>
      <summary className="cursor-pointer text-[11px] font-semibold uppercase tracking-[0.1em] text-muted">
        TTS options
      </summary>
      <div className="mt-2 grid grid-cols-2 gap-2">
        <Field label="Seed" hint="blank = the run's default">
          <TextInput
            type="number"
            value={opts?.seed ?? ""}
            onChange={(event) =>
              set({ seed: event.currentTarget.value === "" ? null : Number(event.currentTarget.value) })
            }
          />
        </Field>
        <Field label="Style">
          <TextInput
            value={opts?.style ?? ""}
            placeholder="e.g. calm, urgent"
            onChange={(event) => set({ style: event.currentTarget.value || null })}
          />
        </Field>
        <Field label="Reference clip" className="col-span-2" hint="run-relative path, overrides the speaker's reference">
          <TextInput
            value={opts?.ref ?? ""}
            placeholder="refs/SPEAKER_04.wav"
            onChange={(event) => set({ ref: event.currentTarget.value || null })}
          />
        </Field>
        <label className="col-span-2 flex items-center gap-2 text-[12px] text-secondary">
          <input
            type="checkbox"
            checked={Boolean(opts?.greedy)}
            onChange={(event) => set({ greedy: event.currentTarget.checked })}
          />
          Greedy decoding — repeatable, flatter
        </label>
      </div>
    </details>
  );
}

/**
 * The "did the clone actually say the right words" panel: the ASR transcript of
 * the generated clip next to what it was asked to say, and the overlap between
 * them. A low overlap with a plausible-sounding clip is the failure mode that
 * listening alone misses.
 */
function Verification({
  seg,
  concern,
  placement,
}: {
  seg: Segment;
  concern: "none" | "soft" | "bad";
  placement: string[];
}) {
  const overlap = seg.verify?.overlap ?? seg.tts?.overlap ?? null;
  const heard = seg.verify?.heard;

  return (
    <div className="rounded-md border border-border bg-raised p-2">
      <div className="flex items-center gap-2">
        <span className="text-[11px] font-semibold uppercase tracking-[0.1em] text-muted">
          Verification
        </span>
        {seg.tts?.verify ? (
          <span className="text-[11px] text-secondary">
            {seg.tts.verify}
            {seg.tts.tries ? ` · ${seg.tts.tries} ${seg.tts.tries === 1 ? "try" : "tries"}` : null}
          </span>
        ) : null}
        <span
          className={cn(
            "ml-auto inline-flex items-center gap-1 text-[12px] tabular-nums",
            concern === "bad" ? "text-critical" : concern === "soft" ? "text-secondary" : "text-muted",
          )}
        >
          {concern !== "none" ? <TriangleAlert className="h-3.5 w-3.5" /> : null}
          overlap {percent(overlap)}
        </span>
      </div>

      {overlap != null ? (
        <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-border">
          <div
            className="h-full rounded-full"
            style={{
              width: `${Math.max(0, Math.min(1, overlap)) * 100}%`,
              backgroundColor:
                concern === "bad" ? "var(--color-critical)" : "var(--color-dubbed)",
            }}
          />
        </div>
      ) : null}

      {heard ? (
        <p className="auto-dir mt-2 text-[12px] leading-relaxed text-secondary">
          <span className="text-muted">heard: </span>
          {heard}
        </p>
      ) : (
        <p className="mt-2 text-[12px] text-muted">
          {seg.keep ? "Kept segments are not synthesized, so nothing to verify." : "No clip yet."}
        </p>
      )}

      {seg.place ? (
        <p className="mt-2 text-[11px] tabular-nums text-muted">
          placed {timecode(seg.place.start)} · rate {seg.place.rate.toFixed(3)} · drift{" "}
          {seg.place.drift.toFixed(2)}s
          {placement.length ? (
            <span className="text-secondary"> — {placement.join(", ")}</span>
          ) : null}
        </p>
      ) : null}
    </div>
  );
}
