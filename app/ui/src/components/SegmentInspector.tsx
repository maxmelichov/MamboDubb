/**
 * Everything about one segment — but not all at once.
 *
 * The old rail put thirteen controls in a flat column with a hairline between
 * every pair, which made "change the speaker" and "read the translation" look
 * like the same size of decision. They are not. Watch someone review a dub and
 * the frequencies are wildly uneven:
 *
 *   1. read the translation, and fix it            — almost every segment
 *   2. hear the two versions back to back          — almost every segment
 *   3. decide dub-or-keep                          — often
 *   4. re-translate or re-voice this one line      — sometimes
 *   5. speaker, languages, split/merge, TTS seeds  — rarely
 *   6. which fields are locked, the uid            — when something is wrong
 *
 * So 1–4 are the rail, unlabelled by any container, in that order and nothing
 * else. 5 and 6 are three named shelves that start shut. The shelves are not a
 * demotion — they are what makes the four things above them legible.
 *
 * The two classes of edit are still kept apart, because the distinction is
 * real: text, keep, speaker and the language tags are no-model edits that save
 * immediately even mid-job, while re-translate and re-voice need the single
 * model slot and queue. The model actions say so on their faces now — a button
 * that will take a minute and get in line behind something else should not
 * look like a checkbox.
 */

import { useEffect, useState } from "react";
import {
  Languages,
  ListTree,
  Lock,
  Merge,
  Scissors,
  Settings2,
  TriangleAlert,
  Volume2,
} from "lucide-react";
import { ABPlayer } from "./ABPlayer";
import {
  Button,
  Checkbox,
  Disclosure,
  Eyebrow,
  Field,
  Kbd,
  NumberInput,
  Select,
  StatePill,
  TextArea,
  TextInput,
} from "./ui";
import { cn } from "../lib/classNames";
import { duration as fmtDuration, percent, speakerLabel, timecode } from "../lib/format";
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
  const locked = Object.entries(seg.locked ?? {})
    .filter(([, on]) => on)
    .map(([field]) => field);

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
      <header className="sticky top-0 z-10 border-b border-border bg-sunken px-4 py-2.5">
        <div className="flex items-center gap-2">
          <span className="font-mono text-[13px] font-bold tabular-nums text-primary">
            #{seg.id}
          </span>
          <StatePill token={meta.token} glyph={meta.glyph} label={meta.label} />
          {locked.length ? (
            <span
              className="inline-flex items-center gap-1 text-[11px] text-muted"
              title={`Hand-edited: ${locked.join(", ")}. A re-run will not overwrite these.`}
            >
              <Lock className="h-3 w-3" aria-hidden />
              hand-edited
            </span>
          ) : null}
          <span className="ml-auto shrink-0 font-mono text-[11px] tabular-nums text-secondary">
            {timecode(seg.start)}–{timecode(seg.end)}
            <span className="ml-1.5 text-muted">{fmtDuration(seg.end - seg.start)}</span>
          </span>
        </div>
      </header>

      <div className="flex flex-col gap-4 px-4 pt-4">
        {/*
          The one thing that jumps the queue. A clone that said the wrong words
          is not a "detail" to be found behind a shelf — it is the reason this
          segment is worth opening at all, so it sits above the text it
          contradicts and names the fix.
        */}
        {concern === "bad" || state === "failed" ? (
          <p
            role="status"
            className="flex gap-2 rounded-lg border border-critical/40 bg-critical/10 px-2.5 py-2 text-[12px] leading-relaxed text-primary"
          >
            <TriangleAlert
              aria-hidden
              className="mt-0.5 h-3.5 w-3.5 shrink-0"
              style={{ color: "var(--color-critical)" }}
            />
            <span>
              {state === "failed"
                ? "Synthesis failed for this line — the dub falls back to the original audio."
                : `The clone did not say this line: the check heard ${percent(
                    seg.verify?.overlap ?? seg.tts?.overlap,
                  )} of it.`}{" "}
              Re-voice it below, or keep the original.
            </span>
          </p>
        ) : null}

        {/* 1 — the text. The original is reference and the translation is the
            work, so only one of them is a field. Correcting the transcript
            itself is real but rare; it lives in Advanced. */}
        <section className="flex flex-col gap-1.5">
          <Eyebrow>Original</Eyebrow>
          <p
            dir="auto"
            className="auto-dir rounded-lg border border-border bg-sunken px-2.5 py-2 text-[12px] leading-relaxed text-secondary"
          >
            {seg.text || <span className="italic text-muted">no transcript for this span</span>}
          </p>
        </section>

        <Field
          label="Translation"
          hint={
            <span className="inline-flex flex-wrap items-center gap-1.5">
              <Kbd>tab</Kbd> to save
              {seg.locked?.text_en ? " · locked, a re-run will not overwrite it" : null}
            </span>
          }
        >
          {/*
            `dir="auto"` and not just the `.auto-dir` class: the class sets
            `unicode-bidi: plaintext`, which is enough to *render* a Hebrew line
            right-to-left, but a textarea also has a caret, a selection and a
            home/end key, and those follow the element's direction. A Hebrew
            line typed into an LTR box puts the cursor in the wrong place on
            every keystroke.
          */}
          <TextArea
            dir="auto"
            autoGrow
            className="auto-dir min-h-24 text-[13px]"
            value={target}
            placeholder="not translated yet"
            onChange={(event) => setTarget(event.currentTarget.value)}
            onBlur={() => target !== (seg.text_en ?? "") && onPatch({ text_en: target })}
          />
        </Field>

        {/* 2 — the comparison. */}
        <ABPlayer seg={seg} />

        {/* 3 — the verdict. Two states, both named, neither of them jargon.
            This was a checkbox labelled "passthrough" once. */}
        <section className="flex flex-col gap-1.5">
          <Eyebrow>What plays here</Eyebrow>
          <div className="flex overflow-hidden rounded-lg border border-border bg-raised">
            <Choice
              active={!seg.keep}
              onClick={() => onPatch({ keep: false })}
              label="Dub it"
              glyph={STATE_META.dubbed.glyph}
              token={STATE_META.dubbed.token}
            />
            <span className="w-px shrink-0 bg-border" aria-hidden />
            <Choice
              active={seg.keep}
              onClick={() => onPatch({ keep: true, keep_reason: "manual" })}
              label="Keep original"
              glyph={STATE_META.kept.glyph}
              token={STATE_META.kept.token}
            />
          </div>
          <p className="text-[11px] leading-snug text-muted">
            {seg.keep
              ? "The source audio plays untouched. Nothing is synthesized for this span."
              : "Synthesized speech replaces the source audio for this span."}
            {seg.keep && seg.keep_reason ? (
              <>
                {" "}
                Kept because <code className="font-mono">{seg.keep_reason}</code>.
              </>
            ) : null}
            <span className="ml-1 whitespace-nowrap">
              <Kbd>k</Kbd> toggles.
            </span>
          </p>
        </section>

        {/* 4 — the two model actions. Both cost a model load and a place in the
            queue, and both say so before you press them rather than after. */}
        <section className="flex flex-col gap-2">
          <Eyebrow>Redo this line</Eyebrow>
          <ModelAction
            icon={Languages}
            label="Re-translate this line"
            cost="~20 s · queues behind the running job"
            disabled={busy}
            disabledReason="Already working on this segment."
            onClick={() => {
              if (
                seg.locked?.text_en &&
                !window.confirm(
                  "This translation was written by hand. Re-translate and discard it?",
                )
              ) {
                return;
              }
              onRetranslate();
            }}
          />
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

      {/* Everything below here starts shut. See the file header for why. */}
      <div className="mt-4 flex flex-col px-4 pb-4">
        <VoiceShelf
          seg={seg}
          speakers={speakers}
          concern={concern}
          placement={placement}
          onPatch={onPatch}
        />

        <Disclosure
          id="seg.languages"
          icon={Languages}
          label="Languages"
          summary={`${seg.src_lang ?? "inherit"} → ${seg.tgt_lang ?? "inherit"}`}
        >
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
          <p className="text-[11px] leading-snug text-muted">
            <em>inherit</em> uses the run's languages. Set these when one segment is spoken in a
            different language than the rest.
          </p>
        </Disclosure>

        <Disclosure
          id="seg.advanced"
          icon={ListTree}
          label="Advanced"
          summary="transcript, split & merge, locks"
        >
          <Field
            label="Correct the transcript"
            hint="What the ASR heard. Changing it does not re-translate on its own."
          >
            <TextArea
              dir="auto"
              autoGrow
              className="auto-dir min-h-16"
              value={source}
              onChange={(event) => setSource(event.currentTarget.value)}
              onBlur={() => source !== seg.text && onPatch({ text: source })}
            />
          </Field>

          <div className="flex flex-col gap-1.5">
            <Eyebrow>Change the span</Eyebrow>
            <p className="text-[11px] leading-snug text-muted">
              Both of these discard the translation and the clip for every segment they touch —
              the line was written for the old span.
            </p>
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
                    : "Merge this segment into the one before it"
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
                Merge with previous
              </Button>
              <Button
                size="sm"
                disabled={!next || next.speaker !== seg.speaker}
                title={
                  next && next.speaker !== seg.speaker
                    ? "Only adjacent segments with the same speaker can merge"
                    : "Merge the next segment into this one"
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
                Merge with next
              </Button>
            </div>
          </div>

          {/*
            Locks are the answer to "why did my edit survive a re-run" and to
            "why did the pipeline not fix this". They are read-only here: a
            lock is set by editing the field, and the honest way to clear one
            is to put the pipeline's own value back.
          */}
          <div className="flex flex-col gap-1.5">
            <Eyebrow>Hand-edited fields</Eyebrow>
            {locked.length ? (
              <p className="text-[11px] leading-relaxed text-muted">
                <code className="font-mono text-secondary">{locked.join(", ")}</code> — edited
                here, so a re-run of the pipeline will not overwrite them.
              </p>
            ) : (
              <p className="text-[11px] leading-relaxed text-muted">
                Nothing on this segment has been hand-edited. Every field is still the
                pipeline's, and a re-run may replace any of them.
              </p>
            )}
            <p className="font-mono text-[10px] text-muted">uid {seg.uid}</p>
          </div>
        </Disclosure>
      </div>
    </div>
  );
}

/**
 * The "Voice & speaker" shelf: who this line is spoken by, how it was
 * synthesized, and how the last take checked out.
 *
 * Verification lives here rather than in a shelf of its own because it is the
 * *result* of everything else in the shelf — the seed, the reference clip and
 * the speaker are the inputs, and the overlap is what they produced. The shut
 * summary carries the result, so a bad take is visible without opening.
 */
function VoiceShelf({
  seg,
  speakers,
  concern,
  placement,
  onPatch,
}: {
  seg: Segment;
  speakers: string[];
  concern: "none" | "soft" | "bad";
  placement: string[];
  onPatch: (patch: SegmentPatch) => void;
}) {
  const opts = seg.tts_opts ?? null;
  const overlap = seg.verify?.overlap ?? seg.tts?.overlap ?? null;
  const heard = seg.verify?.heard;

  const verdict = seg.keep
    ? "not synthesized"
    : overlap == null
      ? "no clip yet"
      : `heard ${percent(overlap)}`;

  const set = (patch: TtsOpts) => onPatch({ tts_opts: { ...(opts ?? {}), ...patch } });

  return (
    <Disclosure
      id="seg.voice"
      icon={Settings2}
      label="Voice & speaker"
      tone={concern === "bad" ? "warn" : "neutral"}
      summary={`${speakerLabel(seg.speaker)} · ${verdict}${opts ? " · overridden" : ""}`}
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
          hint="run-relative path, overrides the speaker's"
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
      <p className="-mt-1 text-[11px] leading-snug text-muted">
        These take effect the next time this line is voiced.
      </p>

      {/*
        The "did the clone actually say the right words" panel: the ASR
        transcript of the generated clip next to what it was asked to say. A low
        overlap with a plausible-sounding clip is the failure mode that
        listening alone misses.
      */}
      <div className="flex flex-col gap-2 border-t border-border pt-3">
        <div className="flex items-center gap-2">
          <Eyebrow className="flex-1">Last take</Eyebrow>
          <span
            className={cn(
              "inline-flex items-center gap-1 font-mono text-[11px] tabular-nums",
              concern === "bad"
                ? "text-critical"
                : concern === "soft"
                  ? "text-secondary"
                  : "text-muted",
            )}
          >
            {concern !== "none" ? <TriangleAlert className="h-3.5 w-3.5" aria-hidden /> : null}
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
                backgroundColor:
                  concern === "bad" ? "var(--color-critical)" : "var(--color-dubbed)",
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
      </div>
    </Disclosure>
  );
}

/** One half of the dub-or-keep control. */
function Choice({
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
        "flex h-9 flex-1 items-center justify-center gap-1.5 whitespace-nowrap px-2 text-[12px] transition-colors",
        active
          ? "bg-primary/[0.09] font-semibold text-primary"
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

/**
 * A button that loads a model.
 *
 * Two lines, always: the verb on top and the price underneath. The price is
 * not a tooltip because the decision it informs — "is it worth waiting a
 * minute for this line" — is made *before* the click, and a tooltip arrives
 * after you have already committed to hovering the thing you were going to
 * press anyway.
 */
function ModelAction({
  icon: Icon,
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
      <Icon className="h-3.5 w-3.5 shrink-0 text-secondary" aria-hidden />
      <span className="min-w-0 flex-1">
        <span className="block truncate text-[12px] font-semibold text-primary">{label}</span>
        <span className="block truncate text-[11px] text-muted">{cost}</span>
      </span>
    </button>
  );
}
