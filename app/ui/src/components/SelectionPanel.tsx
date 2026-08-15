/**
 * Everything about the selected line except the line itself.
 *
 * That exception is the design. The old inspector was 704 lines and its top
 * third was a copy of the text — an "Original" paragraph and a "Translation"
 * textarea — which meant the two places a user could read a translation
 * disagreed about which was authoritative, and the one they were looking at
 * (the list) was the one they could not type in. Now the text lives in the
 * script, where the comparison is, and it is edited in place. This panel holds
 * what is *true about* the line and cannot be said in a row.
 *
 * Order is frequency, as before, and the frequencies are unchanged: the verdict
 * (dub or keep) and the two model actions are what a reviewer reaches for on a
 * normal segment; the speaker, the seeds, the verification detail, the bounds
 * and the locks are what they reach for when something is wrong. So the first
 * three are on the surface and the rest are four named shelves that start shut
 * and remember, for the session, whether you opened them.
 *
 * One control was deleted rather than moved: the "Style" text input. Qwen3-TTS
 * has no instruct parameter — there is nowhere for "calm, urgent" to go — so
 * the field was a promise the pipeline cannot keep, which is worse than a
 * missing feature.
 */

import { useState } from "react";
import {
  Check,
  Clock,
  Languages,
  ListTree,
  Merge,
  ShieldCheck,
  TriangleAlert,
  Unlock,
  Volume2,
} from "lucide-react";
import {
  Button,
  Checkbox,
  ConfirmButton,
  Disclosure,
  Eyebrow,
  Field,
  NumberInput,
  Select,
  StateIcon,
  TextInput,
} from "./ui";
import { cn } from "../lib/classNames";
import { duration as fmtDuration, percent, speakerLabel, timecode } from "../lib/format";
import {
  lockedFields,
  neighbours,
  placementConcern,
  segmentState,
  verifyConcern,
  type SegmentState,
} from "../lib/segments";
import type { Segment, SegmentPatch, TtsOpts } from "../lib/types";

const LANGS = ["", "he", "en", "ar", "ru", "fr", "es", "de"];

export function SelectionPanel({
  seg,
  segments,
  speakers,
  busy,
  playhead,
  onPatch,
  onVerdict,
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
  /** The verdict is not a patch: flipping to a dub also queues the work. */
  onVerdict: (keep: boolean) => void;
  onSplit: (at: number) => void;
  onMerge: (uidB: string) => void;
  onRetranslate: () => void;
  onResynthesize: () => void;
}) {
  const state = segmentState(seg);
  const concern = verifyConcern(seg);
  const locked = lockedFields(seg);
  const { prev, next } = neighbours(segments, seg.uid);
  const splitPoint = playhead > seg.start + 0.05 && playhead < seg.end - 0.05 ? playhead : null;

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-y-auto">
      <header className="sticky top-0 z-10 flex items-center gap-2 border-b border-border bg-sunken px-3 py-1.5">
        <span className="font-mono text-[14px] font-bold tabular-nums text-primary">#{seg.id}</span>
        <span className="font-mono text-[11px] tabular-nums text-secondary">
          {timecode(seg.start)}–{timecode(seg.end)}
        </span>
        <span className="ml-auto font-mono text-[11px] tabular-nums text-muted">
          {fmtDuration(seg.end - seg.start)}
        </span>
      </header>

      <div className="flex flex-col gap-3 px-3 pt-3">
        {/*
          The one thing that jumps the queue: a clone that said the wrong words
          is the reason this segment is worth opening at all.
        */}
        {concern === "bad" || state === "failed" ? (
          <p
            role="status"
            className="flex gap-2 rounded-lg border border-critical/40 bg-critical/10 px-2.5 py-2 text-[12.5px] leading-relaxed text-primary"
          >
            <TriangleAlert
              aria-hidden
              className="mt-0.5 h-3.5 w-3.5 shrink-0"
              style={{ color: "var(--color-critical)" }}
            />
            <span>
              {state === "failed"
                ? "Synthesis failed for this line — the dub falls back to the original audio."
                : `The clone said ${percent(seg.verify?.overlap ?? seg.tts?.overlap)} of it.`}{" "}
              Re-voice it, or keep the original.
            </span>
          </p>
        ) : null}

        {/* 1 — the verdict. Two states, both named, neither of them jargon. */}
        <section className="flex flex-col gap-1.5">
          <div className="flex overflow-hidden rounded-lg border border-border bg-raised">
            {/* Each half only acts when it is *not* already the verdict:
                pressing the one that is already on is a no-op the user can
                see, and it must be a no-op the server never hears — it would
                otherwise stamp a `keep` lock on a line nobody changed. */}
            <Choice
              active={!seg.keep}
              onClick={() => seg.keep && onVerdict(false)}
              label="Dub it"
              state="dubbed"
            />
            <span className="w-px shrink-0 bg-border" aria-hidden />
            <Choice
              active={seg.keep}
              onClick={() => !seg.keep && onVerdict(true)}
              label="Keep original"
              state="kept"
            />
          </div>
          {/*
            What the other half of the control will *do*, not just what it will
            mean. Flipping to a dub is not a free relabelling: `edit.set_keep`
            drops the subtitle and the clip, so the line has to be translated
            and voiced again — two jobs, or one when the user's own translation
            is locked and survives the flip. Saying "synthesized speech
            replaces the source audio" and then queueing nothing is how two
            lines ended up dubbed in name only.
          */}
          <p className="text-[11px] leading-snug text-muted">
            {seg.keep ? (
              <>
                The source audio plays untouched. “Dub it”{" "}
                {(seg.text_en ?? "").trim() && seg.locked?.text_en
                  ? "queues voice for this line — your translation is kept."
                  : "queues translate + voice for this line."}
              </>
            ) : (
              "Synthesized speech replaces the source audio."
            )}
            {seg.keep && seg.keep_reason ? (
              <>
                {" "}
                Kept because <code className="font-mono">{seg.keep_reason}</code>.
              </>
            ) : null}
          </p>
        </section>

        {/* 2 — the two model actions. Both cost a model load and a place in the
            queue, and both say so before you press them rather than after. */}
        <section className="flex flex-col gap-2">
          {seg.locked?.text_en ? (
            <ConfirmButton
              className="w-full"
              size="md"
              disabled={busy}
              align="left"
              confirmLabel="Re-translate"
              message="This translation was written by hand. Re-translating discards it and replaces it with the model's."
              onConfirm={onRetranslate}
            >
              <ModelActionFace
                icon={Languages}
                label="Re-translate this line"
                cost="~20 s · queues behind the running job"
              />
            </ConfirmButton>
          ) : (
            <ModelAction
              icon={Languages}
              label="Re-translate this line"
              cost="~20 s · queues behind the running job"
              disabled={busy}
              disabledReason="Already working on this segment."
              onClick={onRetranslate}
            />
          )}
          <ModelAction
            icon={Volume2}
            label="Re-voice this line"
            cost="~1 min · queues behind the running job"
            disabled={busy || seg.keep}
            disabledReason={
              seg.keep
                ? "This segment keeps its original audio, so there is nothing to voice. Switch it to “Dub it” first."
                : "Already working on this segment."
            }
            onClick={onResynthesize}
          />
        </section>
      </div>

      {/* Everything below here starts shut. */}
      <div className="mt-3 flex flex-col px-3 pb-4 [&>details:last-of-type]:border-b">
        <VoiceShelf seg={seg} speakers={speakers} onPatch={onPatch} />
        <VerificationShelf seg={seg} concern={concern} />

        <Disclosure
          id="seg.timing"
          icon={Clock}
          label="Timing & languages"
          summary={`${fmtDuration(seg.end - seg.start)} · ${seg.src_lang ?? "inherit"} → ${
            seg.tgt_lang ?? "inherit"
          }`}
        >
          {/* Keyed on the segment: the drafts below are seeded from props on
              mount, so a new selection has to be a new component or it would
              show the previous line's numbers. */}
          <Bounds key={seg.uid} seg={seg} onPatch={onPatch} />

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
          <p className="-mt-1 text-[11px] leading-snug text-muted">
            <em>inherit</em> uses the run's languages. Set these when one segment is spoken in a
            different language than the rest.
          </p>

          <div className="flex flex-col gap-1.5 border-t border-border pt-3">
            <Eyebrow>Change the span</Eyebrow>
            <p className="text-[11px] leading-snug text-muted">
              Both of these discard the translation and the clip for every segment they touch —
              the line was written for the old span.
            </p>
            <div className="flex flex-wrap gap-2">
              <ConfirmButton
                size="sm"
                align="left"
                disabled={splitPoint == null}
                title={
                  splitPoint == null
                    ? "Move the playhead inside this segment to split it"
                    : `Split at ${timecode(splitPoint)}`
                }
                confirmLabel="Split"
                message={
                  splitPoint == null
                    ? ""
                    : `Split at ${timecode(splitPoint)}. Both halves lose their translation and clip.`
                }
                onConfirm={() => splitPoint != null && onSplit(splitPoint)}
              >
                Split at playhead
              </ConfirmButton>
              <ConfirmButton
                size="sm"
                align="left"
                disabled={!prev || prev.speaker !== seg.speaker}
                title={
                  prev && prev.speaker !== seg.speaker
                    ? "Only adjacent segments with the same speaker can merge"
                    : "Merge this segment into the one before it"
                }
                confirmLabel="Merge"
                message="Merge with the previous segment. The merged line loses its translation and clip."
                onConfirm={() => prev && onMerge(prev.uid)}
              >
                <Merge className="h-3.5 w-3.5" />
                Previous
              </ConfirmButton>
              <ConfirmButton
                size="sm"
                align="left"
                disabled={!next || next.speaker !== seg.speaker}
                title={
                  next && next.speaker !== seg.speaker
                    ? "Only adjacent segments with the same speaker can merge"
                    : "Merge the next segment into this one"
                }
                confirmLabel="Merge"
                message="Merge with the next segment. The merged line loses its translation and clip."
                onConfirm={() => next && onMerge(next.uid)}
              >
                <Merge className="h-3.5 w-3.5" />
                Next
              </ConfirmButton>
            </div>
          </div>
        </Disclosure>

        <Disclosure
          id="seg.advanced"
          icon={ListTree}
          label="Advanced"
          summary={locked.length ? `${locked.length} locked` : "nothing hand-edited"}
        >
          {/*
            Locks are the answer to "why did my edit survive a re-run" and to
            "why did the pipeline not fix this". Releasing them is the only way
            to hand a line back to the pipeline after changing your mind —
            `PATCH {locked:{}}` has existed on the server since the edit API
            landed and nothing in the UI had ever called it.
          */}
          <div className="flex flex-col gap-2">
            <Eyebrow>Hand-edited fields</Eyebrow>
            {locked.length ? (
              <>
                <p className="text-[11px] leading-relaxed text-muted">
                  <code className="font-mono text-secondary">{locked.join(", ")}</code> — edited
                  here, so a re-run of the pipeline will not overwrite them.
                </p>
                <ConfirmButton
                  size="sm"
                  align="left"
                  className="self-start"
                  confirmLabel="Release"
                  message="Release every lock on this line. A re-run of the pipeline may then replace any of these fields with its own."
                  onConfirm={() => onPatch({ locked: {} })}
                >
                  <Unlock className="h-3.5 w-3.5" />
                  Release locks
                </ConfirmButton>
              </>
            ) : (
              <p className="text-[11px] leading-relaxed text-muted">
                Nothing on this segment has been hand-edited. Every field is still the
                pipeline's, and a re-run may replace any of them.
              </p>
            )}
            <p className="font-mono text-[11px] text-muted">uid {seg.uid}</p>
          </div>
        </Disclosure>
      </div>
    </div>
  );
}

/**
 * The bounds.
 *
 * `PATCH` insists start and end move together (the server 400s otherwise,
 * because a half-moved span is not a span), so they are drafted locally and
 * committed as one — and the commit is guarded, because sending the values back
 * unchanged would re-place the segment for nothing.
 */
function Bounds({ seg, onPatch }: { seg: Segment; onPatch: (patch: SegmentPatch) => void }) {
  const [start, setStart] = useState(String(seg.start));
  const [end, setEnd] = useState(String(seg.end));
  const dirty = Number(start) !== seg.start || Number(end) !== seg.end;
  const valid = Number.isFinite(Number(start)) && Number.isFinite(Number(end)) &&
    Number(end) > Number(start);

  return (
    <div className="flex items-end gap-2">
      <Field label="Start" className="flex-1">
        <NumberInput
          step={0.05}
          suffix="s"
          value={start}
          onChange={(event) => setStart(event.currentTarget.value)}
        />
      </Field>
      <Field label="End" className="flex-1">
        <NumberInput
          step={0.05}
          suffix="s"
          value={end}
          onChange={(event) => setEnd(event.currentTarget.value)}
        />
      </Field>
      <Button
        size="md"
        disabled={!dirty || !valid}
        title={valid ? "Move this segment's span" : "The end has to come after the start"}
        onClick={() => onPatch({ start: Number(start), end: Number(end) })}
      >
        Move
      </Button>
    </div>
  );
}

/** Who this line is spoken by, and how it is synthesized. */
function VoiceShelf({
  seg,
  speakers,
  onPatch,
}: {
  seg: Segment;
  speakers: string[];
  onPatch: (patch: SegmentPatch) => void;
}) {
  const opts = seg.tts_opts ?? null;
  /** Same discipline as the text fields: a value that did not move is not a save. */
  const set = (patch: TtsOpts) => {
    const next = { ...(opts ?? {}), ...patch };
    if (JSON.stringify(next) === JSON.stringify(opts ?? {})) return;
    onPatch({ tts_opts: next });
  };

  return (
    <Disclosure
      id="seg.voice"
      label="Voice & speaker"
      summary={`${speakerLabel(seg.speaker)}${opts ? " · overridden" : ""}`}
    >
      <Field label="Speaker" hint="Which reference voice this line is cloned from.">
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

      <div className="grid grid-cols-2 gap-3">
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
        <label className="flex items-end gap-2 pb-2.5 text-[12.5px] text-secondary">
          <Checkbox
            checked={Boolean(opts?.greedy)}
            onChange={(event) => set({ greedy: event.currentTarget.checked })}
          />
          Greedy — repeatable, flatter
        </label>
        <Field
          label="Reference clip"
          className="col-span-2"
          hint="run-relative path, overrides the speaker's"
        >
          <TextInput
            value={opts?.ref ?? ""}
            placeholder="refs/SPEAKER_04.wav"
            onChange={(event) => set({ ref: event.currentTarget.value || null })}
          />
        </Field>
      </div>
      <p className="-mt-1 text-[11px] leading-snug text-muted">
        These take effect the next time this line is voiced.
      </p>
    </Disclosure>
  );
}

/**
 * Did the clone say the right words, and did the fit ruin it?
 *
 * A shelf of its own now rather than a footnote under the voice controls: it is
 * the answer the reviewer wants when a row is flagged, and a low overlap on a
 * clip that *sounds* plausible is the failure mode listening alone misses.
 */
function VerificationShelf({ seg, concern }: { seg: Segment; concern: "none" | "soft" | "bad" }) {
  const overlap = seg.verify?.overlap ?? seg.tts?.overlap ?? null;
  const heard = seg.verify?.heard;
  const placement = placementConcern(seg);
  const tries = seg.verify?.tries ?? seg.tts?.tries ?? null;

  return (
    <Disclosure
      id="seg.verify"
      icon={ShieldCheck}
      label="Verification"
      tone={concern === "bad" ? "warn" : "neutral"}
      summary={
        seg.keep ? "not synthesized" : overlap == null ? "no clip yet" : `heard ${percent(overlap)}`
      }
    >
      <div className="flex items-center gap-2">
        <Eyebrow className="flex-1">Word overlap</Eyebrow>
        <span
          className={cn(
            "font-mono text-[11px] tabular-nums",
            concern === "bad" ? "text-critical" : "text-secondary",
          )}
        >
          {percent(overlap)}
          {tries ? ` · ${tries} ${tries === 1 ? "try" : "tries"}` : null}
        </span>
      </div>

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
          className="auto-dir rounded-lg border border-border bg-raised p-2 text-[12.5px] leading-relaxed text-secondary"
        >
          <span className="text-[11px] font-bold uppercase tracking-[0.14em] text-muted">
            heard{" "}
          </span>
          {heard}
        </p>
      ) : (
        <p className="text-[12.5px] text-muted">
          {seg.keep
            ? "Kept segments are not synthesized, so there is nothing to verify."
            : "No clip yet."}
        </p>
      )}

      {seg.place ? (
        <p className="font-mono text-[11px] leading-relaxed tabular-nums text-muted">
          placed {timecode(seg.place.start)} · rate {seg.place.rate.toFixed(3)} · drift{" "}
          {seg.place.drift.toFixed(2)}s
          {placement.length ? (
            <span className="text-secondary"> — {placement.join(", ")}</span>
          ) : null}
        </p>
      ) : null}
    </Disclosure>
  );
}

/** One half of the dub-or-keep control. */
function Choice({
  active,
  onClick,
  label,
  state,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
  state: SegmentState;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={cn(
        "flex h-8 flex-1 items-center justify-center gap-1.5 whitespace-nowrap px-2 text-[12.5px] transition-colors",
        // The chosen half must dominate at a glance: a user who has to weigh two
        // washes reads the control backwards, and then reads everything the row
        // says through that mistake (a real support case, not a hypothetical).
        // Ink-on-inverted is unambiguous in both themes; the state icon stays as
        // the hue but only the active side gets the checkmark's weight.
        active
          ? "bg-primary font-semibold text-on-primary"
          : "text-muted hover:bg-border/50 hover:text-secondary",
      )}
    >
      <StateIcon state={state} className={cn("h-2.5 w-2.5", !active && "opacity-50")} />
      {label}
      {active ? <Check className="h-3 w-3" strokeWidth={3} aria-hidden /> : null}
    </button>
  );
}

/**
 * A button that loads a model.
 *
 * Two lines, always: the verb on top and the price underneath. The price is not
 * a tooltip because the decision it informs — "is it worth waiting a minute for
 * this line" — is made *before* the click, and a tooltip arrives after you have
 * already committed to hovering the thing you were going to press anyway.
 */
function ModelActionFace({
  icon: Icon,
  label,
  cost,
}: {
  icon: typeof Volume2;
  label: string;
  cost: string;
}) {
  return (
    <>
      <Icon className="h-3.5 w-3.5 shrink-0 text-secondary" aria-hidden />
      <span className="min-w-0 flex-1 text-left">
        <span className="block truncate text-[12.5px] font-semibold text-primary">{label}</span>
        <span className="block truncate text-[11px] font-normal text-muted">{cost}</span>
      </span>
    </>
  );
}

function ModelAction({
  icon,
  label,
  cost,
  disabled,
  disabledReason,
  onClick,
}: {
  icon: typeof Volume2;
  label: string;
  cost: string;
  disabled: boolean;
  /** A dead button that will not say why is the reason people file bugs. */
  disabledReason: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={disabled ? disabledReason : undefined}
      className={cn(
        "flex items-center gap-2.5 rounded-lg border border-border bg-raised px-2.5 py-2 text-left",
        "shadow-card transition-all active:scale-[0.99] hover:border-axis",
        // Not `pointer-events-none`: the tooltip above is the whole point.
        disabled && "cursor-not-allowed opacity-45 hover:border-border",
      )}
    >
      <ModelActionFace icon={icon} label={label} cost={cost} />
    </button>
  );
}
