/**
 * Everything about one segment, in the order a reviewer works through it:
 * hear it, read it, fix it, re-run it.
 *
 * The two classes of control are deliberately not interleaved. Text, speaker,
 * keep and the language tags are no-model edits and save immediately, even
 * while a job is running. Re-translate and re-voice need the single job slot,
 * so they sit together at the bottom behind a queue.
 *
 * Structurally it is MamboRambo's settings card: an eyebrow-labelled stack,
 * hairline dividers between concerns, and no box inside a box. The rail is
 * dense — it holds a dozen controls — so the rhythm is tighter than a page
 * card's, but the vocabulary is identical.
 */

import { useEffect, useState } from "react";
import {
  AudioLines,
  ChevronDown,
  Languages,
  ListTree,
  Lock,
  Merge,
  Scissors,
  Settings2,
  ShieldCheck,
  Sparkles,
  TriangleAlert,
  Type,
  Volume2,
} from "lucide-react";
import { ABPlayer } from "./ABPlayer";
import {
  Button,
  Checkbox,
  Divider,
  Field,
  Kbd,
  NumberInput,
  SectionLabel,
  Select,
  StatePill,
  TextArea,
  TextInput,
} from "./ui";
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
      <header className="sticky top-0 z-10 border-b border-border bg-sunken px-4 py-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-mono text-[13px] font-bold tabular-nums text-primary">
            #{seg.id}
          </span>
          <StatePill token={meta.token} glyph={meta.glyph} label={meta.label} />
          {seg.locked ? (
            <span className="inline-flex items-center gap-1 text-[11px] text-muted">
              <Lock className="h-3 w-3" />
              hand-edited
            </span>
          ) : null}
          <span className="ml-auto truncate font-mono text-[10px] text-muted">{seg.uid}</span>
        </div>
        <div className="mt-1.5 flex items-baseline gap-2 font-mono text-[11px] tabular-nums text-secondary">
          {timecode(seg.start)} – {timecode(seg.end)}
          <span className="text-muted">{fmtDuration(seg.end - seg.start)}</span>
        </div>
      </header>

      <div className="flex flex-col gap-4 p-4">
        <section className="flex flex-col gap-2.5">
          <SectionLabel icon={AudioLines}>Compare</SectionLabel>
          <ABPlayer seg={seg} />
        </section>

        <Divider />

        <section className="flex flex-col gap-2.5">
          <SectionLabel icon={Type}>Text</SectionLabel>

          {/* 2:3, not 1:1 — "Keep original" needs the wider cell or it wraps
              inside its own toggle. */}
          <div className="grid grid-cols-5 gap-3">
            <Field label="Speaker" className="col-span-2">
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
            </Field>

            <Field label="This segment" className="col-span-3">
              {/*
                Keep is a manual decision once the user makes it — `mark_keep`
                must never re-decide it, which is what the `locked` flag on the
                PATCH is for.
              */}
              <div className="flex h-9 overflow-hidden rounded-lg border border-border bg-raised">
                <Toggle
                  active={!seg.keep}
                  onClick={() => onPatch({ keep: false })}
                  label="Dub it"
                  glyph={STATE_META.dubbed.glyph}
                  token={STATE_META.dubbed.token}
                />
                <span className="w-px shrink-0 bg-border" aria-hidden />
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
                  kept because <code className="font-mono">{seg.keep_reason}</code>
                </>
              ) : null
            }
          >
            {/*
              `dir="auto"` and not just the `.auto-dir` class: the class sets
              `unicode-bidi: plaintext`, which is enough to *render* a Hebrew
              line right-to-left, but a textarea also has a caret, a selection
              and a home/end key, and those follow the element's direction. A
              Hebrew line typed into an LTR box puts the cursor in the wrong
              place on every keystroke.
            */}
            <TextArea
              dir="auto"
              autoGrow
              className="auto-dir min-h-20"
              value={source}
              onChange={(event) => setSource(event.currentTarget.value)}
              onBlur={() => source !== seg.text && onPatch({ text: source })}
            />
          </Field>

          <Field
            label="Target text"
            hint={
              <span className="inline-flex flex-wrap items-center gap-1.5">
                <Kbd>tab</Kbd> to save
                {seg.locked?.text_en ? " · locked, a re-run will not overwrite it" : null}
              </span>
            }
          >
            <TextArea
              dir="auto"
              autoGrow
              className="auto-dir min-h-20"
              value={target}
              placeholder="not translated yet"
              onChange={(event) => setTarget(event.currentTarget.value)}
              onBlur={() => target !== (seg.text_en ?? "") && onPatch({ text_en: target })}
            />
          </Field>
        </section>

        <Divider />

        {/*
          Per-segment language tags: "this bit is Arabic, translate it to
          English". Empty means "inherit the run's languages".
        */}
        <section className="flex flex-col gap-2.5">
          <SectionLabel icon={Languages}>Language overrides</SectionLabel>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Spoken">
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
            <Field label="Translate into">
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
          {/* One hint under the pair, not the same sentence twice — the two
              fields differ by one word and the labels already carry it. */}
          <p className="text-[11px] leading-snug text-muted">
            <em>inherit</em> uses the run's languages. Set these when one segment is spoken in
            a different language than the rest.
          </p>
        </section>

        <Divider />

        <TtsOptions opts={seg.tts_opts ?? null} onChange={(tts_opts) => onPatch({ tts_opts })} />

        <Divider />

        <Verification seg={seg} concern={concern} placement={placement} />

        <Divider />

        <section className="flex flex-col gap-2.5">
          <SectionLabel icon={Sparkles}>Model actions</SectionLabel>
          <p className="-mt-1 text-[11px] leading-relaxed text-muted">
            These load a model, so they queue behind whatever is running. Everything above
            saves immediately.
          </p>
          <div className="flex flex-wrap gap-2">
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
              size="sm"
            >
              <Languages className="h-3.5 w-3.5" />
              Re-translate
            </Button>
            <Button onClick={onResynthesize} disabled={busy || seg.keep} size="sm">
              <Volume2 className="h-3.5 w-3.5" />
              Re-voice
            </Button>
          </div>
        </section>

        <section className="flex flex-col gap-2.5">
          <SectionLabel icon={ListTree}>Structure</SectionLabel>
          <div className="flex flex-wrap gap-2">
            <Button
              size="sm"
              disabled={splitPoint == null}
              title={
                splitPoint == null
                  ? "Move the playhead inside this segment to split it"
                  : `Split at ${timecode(splitPoint)}`
              }
              onClick={() => {
                if (splitPoint == null) return;
                if (
                  !window.confirm(
                    `Split at ${timecode(splitPoint)}? Both halves lose their translation and clip.`,
                  )
                )
                  return;
                onSplit(splitPoint);
              }}
            >
              <Scissors className="h-3.5 w-3.5" />
              Split at playhead
            </Button>
            <Button
              size="sm"
              disabled={!prev || prev.speaker !== seg.speaker}
              title={
                prev && prev.speaker !== seg.speaker
                  ? "Only adjacent segments with the same speaker can merge"
                  : undefined
              }
              onClick={() => {
                if (!prev) return;
                if (
                  !window.confirm(
                    "Merge with the previous segment? The merged line loses its translation and clip.",
                  )
                )
                  return;
                onMerge(prev.uid);
              }}
            >
              <Merge className="h-3.5 w-3.5" />
              Merge ←
            </Button>
            <Button
              size="sm"
              disabled={!next || next.speaker !== seg.speaker}
              title={
                next && next.speaker !== seg.speaker
                  ? "Only adjacent segments with the same speaker can merge"
                  : undefined
              }
              onClick={() => {
                if (!next) return;
                if (
                  !window.confirm(
                    "Merge with the next segment? The merged line loses its translation and clip.",
                  )
                )
                  return;
                onMerge(next.uid);
              }}
            >
              <Merge className="h-3.5 w-3.5" />
              Merge →
            </Button>
          </div>
        </section>
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
        "flex flex-1 items-center justify-center gap-1.5 whitespace-nowrap px-2 text-[12px] transition-colors",
        active
          ? "bg-primary/[0.08] font-semibold text-primary"
          : "text-secondary hover:bg-border/50",
      )}
    >
      <span aria-hidden style={{ color: token }}>
        {glyph}
      </span>
      {label}
    </button>
  );
}

/** What the collapsed disclosure says it is hiding. */
function summarizeOpts(opts: TtsOpts | null): string {
  const set = [
    opts?.seed != null ? `seed ${opts.seed}` : null,
    opts?.style ? opts.style : null,
    opts?.ref ? "custom reference" : null,
    opts?.greedy ? "greedy" : null,
  ].filter(Boolean);
  return set.length ? set.join(" · ") : "seed, style, reference clip — all inherited";
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
    <details className="group" open={Boolean(opts)}>
      {/* The summary is the only control in the rail that is not obviously a
          control, so it gets the hover wash, a real chevron and — when it is
          shut — a line saying what is inside it. */}
      <summary className="-mx-1.5 flex cursor-pointer list-none items-center gap-2 rounded-md px-1.5 py-1 transition-colors hover:bg-sunken [&::-webkit-details-marker]:hidden">
        <SectionLabel icon={Settings2} className="shrink-0">
          TTS options
        </SectionLabel>
        <span className="min-w-0 flex-1 truncate text-[11px] text-muted group-open:hidden">
          {summarizeOpts(opts)}
        </span>
        <ChevronDown
          aria-hidden
          className="h-3.5 w-3.5 shrink-0 text-muted transition-transform group-open:rotate-180"
        />
      </summary>
      <div className="mt-3 grid grid-cols-2 gap-3">
        <Field label="Seed" hint="blank = the run's default">
          <NumberInput
            value={opts?.seed ?? ""}
            onChange={(event) =>
              set({
                seed: event.currentTarget.value === "" ? null : Number(event.currentTarget.value),
              })
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
        <Field
          label="Reference clip"
          className="col-span-2"
          hint="run-relative path, overrides the speaker's reference"
        >
          <TextInput
            value={opts?.ref ?? ""}
            placeholder="refs/SPEAKER_04.wav"
            onChange={(event) => set({ ref: event.currentTarget.value || null })}
          />
        </Field>
        <label className="col-span-2 flex items-center gap-2 text-[12px] text-secondary">
          <Checkbox
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
    <section className="flex flex-col gap-2.5">
      <div className="flex items-center gap-2">
        <SectionLabel icon={ShieldCheck} className="flex-1">
          Verification
        </SectionLabel>
        <span
          className={cn(
            "inline-flex items-center gap-1 font-mono text-[11px] tabular-nums",
            concern === "bad" ? "text-critical" : concern === "soft" ? "text-secondary" : "text-muted",
          )}
        >
          {concern !== "none" ? <TriangleAlert className="h-3.5 w-3.5" /> : null}
          overlap {percent(overlap)}
        </span>
      </div>

      {seg.tts?.verify ? (
        <p className="text-[11px] text-secondary">
          {seg.tts.verify}
          {seg.tts.tries ? ` · ${seg.tts.tries} ${seg.tts.tries === 1 ? "try" : "tries"}` : null}
        </p>
      ) : null}

      {overlap != null ? (
        <div className="h-1 overflow-hidden rounded-full bg-border">
          <div
            className="h-full rounded-full transition-[width] duration-300"
            style={{
              width: `${Math.max(0, Math.min(1, overlap)) * 100}%`,
              backgroundColor: concern === "bad" ? "var(--color-critical)" : "var(--color-dubbed)",
            }}
          />
        </div>
      ) : null}

      {heard ? (
        <p
          dir="auto"
          className="auto-dir rounded-lg border border-border bg-raised p-2.5 text-[12px] leading-relaxed text-secondary"
        >
          <span className="text-[10px] font-bold uppercase tracking-[0.14em] text-muted">
            heard{" "}
          </span>
          {heard}
        </p>
      ) : (
        <p className="text-[12px] text-muted">
          {seg.keep ? "Kept segments are not synthesized, so nothing to verify." : "No clip yet."}
        </p>
      )}

      {seg.place ? (
        <p className="font-mono text-[11px] leading-relaxed tabular-nums text-muted">
          placed {timecode(seg.place.start)} · rate {seg.place.rate.toFixed(3)} · drift{" "}
          {seg.place.drift.toFixed(2)}s
          {placement.length ? <span className="text-secondary"> — {placement.join(", ")}</span> : null}
        </p>
      ) : null}
    </section>
  );
}
