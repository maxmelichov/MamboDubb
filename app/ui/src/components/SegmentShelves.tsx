/**
 * The two shelves of the selection panel that hold a line's *machinery*: who
 * voices it and how, and whether the clone actually said it. Plus the bounds
 * editor the timing shelf opens with.
 *
 * They start shut (`Disclosure`), so this is the part of the panel a reviewer
 * only reads when something is wrong, which is why it is its own file.
 */

import { useState } from "react";
import { ShieldCheck } from "lucide-react";
import {
  Button,
  Checkbox,
  Disclosure,
  Eyebrow,
  Field,
  NumberInput,
  Select,
  TextInput,
} from "./ui";
import { cn } from "../lib/classNames";
import { percent, speakerLabel, timecode } from "../lib/format";
import { placementConcern, pipelineFailed } from "../lib/segments";
import type { Segment, SegmentPatch, TtsOpts } from "../lib/types";

/**
 * The bounds.
 *
 * `PATCH` insists start and end move together (the server 400s otherwise,
 * because a half-moved span is not a span), so they are drafted locally and
 * committed as one and the commit is guarded, because sending the values back
 * unchanged would re-place the segment for nothing.
 */
export function Bounds({ seg, onPatch }: { seg: Segment; onPatch: (patch: SegmentPatch) => void }) {
  const [start, setStart] = useState(String(seg.start));
  const [end, setEnd] = useState(String(seg.end));
  // An empty field is not a number, and `Number("")` is 0: clearing Start left
  // Move enabled and PATCHed `start: 0`, which is a span reaching back to the
  // top of the video over every line in between. The server refuses it
  // (`edit.set_bounds`) and the client must not offer it.
  const value = (text: string): number => (text.trim() === "" ? NaN : Number(text));
  const dirty = value(start) !== seg.start || value(end) !== seg.end;
  const valid =
    Number.isFinite(value(start)) && Number.isFinite(value(end)) && value(end) > value(start);

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
        title={
          valid
            ? "Move this segment's span"
            : "Both bounds are numbers, and the end comes after the start"
        }
        onClick={() => onPatch({ start: value(start), end: value(end) })}
      >
        Move
      </Button>
    </div>
  );
}

/** Who this line is spoken by, and how it is synthesized. */
export function VoiceShelf({
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
          Greedy: repeatable, flatter
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
export function VerificationShelf({ seg, concern }: { seg: Segment; concern: "none" | "soft" | "bad" }) {
  const overlap = seg.verify?.overlap ?? seg.tts?.overlap ?? null;
  const heard = seg.verify?.heard;
  const placement = placementConcern(seg);
  const tries = seg.verify?.tries ?? seg.tts?.tries ?? null;

  return (
    <Disclosure
      id="seg.verify"
      icon={ShieldCheck}
      label="Verification"
      /*
       * A soft concern is a concern. The shelf read `neutral` for it same
       * grey shield, same muted summary as a line with nothing wrong so a
       * clone that said 70% of its words was indistinguishable, shut, from one
       * that said all of them. `caution` is the pending family: the shield goes
       * amber and the summary takes the weight, without putting a 3.70:1 hue
       * on eleven-pixel text.
       */
      tone={concern === "bad" ? "warn" : concern === "soft" ? "caution" : "neutral"}
      summary={
        seg.keep && !pipelineFailed(seg)
          ? "not synthesized"
          : overlap == null
            ? "no clip yet"
            : `heard ${percent(overlap)}`
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

      {/*
        The bar carries the verdict too. It was blue at 70% and blue at 100% —
        blue being the "done, there is a clip to check" hue so the one graphic
        in this shelf said "fine" about the number printed above it. Three
        states, three hues, all of them the app's own: red below 0.6, amber
        below 0.9, blue above it.
      */}
      {overlap != null ? (
        <div className="h-1 overflow-hidden rounded-full bg-border">
          <div
            data-overlap-bar={concern}
            className="h-full rounded-full transition-[width] duration-300"
            style={{
              width: `${Math.max(0, Math.min(1, overlap)) * 100}%`,
              backgroundColor:
                concern === "bad"
                  ? "var(--color-critical)"
                  : concern === "soft"
                    ? "var(--color-warning)"
                    : "var(--color-dubbed)",
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
            <span className="text-secondary"> · {placement.join(", ")}</span>
          ) : null}
        </p>
      ) : null}
    </Disclosure>
  );
}
