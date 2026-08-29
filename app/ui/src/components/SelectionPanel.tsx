/**
 * Everything about the selected line except the line itself.
 *
 * That exception is the design. The old inspector was 704 lines and its top
 * third was a copy of the text an "Original" paragraph and a "Translation"
 * textarea which meant the two places a user could read a translation
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
 * has no instruct parameter there is nowhere for "calm, urgent" to go so
 * the field was a promise the pipeline cannot keep, which is worse than a
 * missing feature.
 */

import {
  Check,
  Clock,
  Languages,
  ListTree,
  Merge,
  Trash2,
  TriangleAlert,
  Unlock,
  Volume2,
} from "lucide-react";
import {
  ConfirmButton,
  Disclosure,
  Eyebrow,
  Field,
  Select,
  StateIcon,
} from "./ui";
import { Bounds, VerificationShelf, VoiceShelf } from "./SegmentShelves";
import { cn } from "../lib/classNames";
import {
  duration as fmtDuration,
  languageName,
  percent,
  timecode,
} from "../lib/format";
import {
  keepReason,
  lockedFields,
  neighbours,
  pipelineFailed,
  segmentState,
  verifyConcern,
  type SegmentState,
} from "../lib/segments";
import { SOURCE_LANG_CODES, TARGET_LANG_CODES } from "../lib/languages";
import type { Segment, SegmentPatch } from "../lib/types";

/*
 * The run's own two lists, which differ because what the ASR and the translator
 * can READ is Arabic plus every dub target, while what Qwen3-TTS can SPEAK is
 * that set minus Arabic (see `lib/languages`). `""` leads both here — it is
 * "inherit", and the empty string is what *clears* an override (see below).
 */
const SRC_LANGS = ["", ...SOURCE_LANG_CODES];
const TGT_LANGS = ["", ...TARGET_LANG_CODES];

/** The picker's own option list — codes in the value, names on the screen. */
function LangOptions({ langs }: { langs: string[] }) {
  return (
    <>
      {langs.map((lang) => (
        <option key={lang || "inherit"} value={lang}>
          {lang ? languageName(lang) : "inherit"}
        </option>
      ))}
    </>
  );
}

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
  onRemove,
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
  /** Take this segment out of the dub entirely see the panel's last section. */
  onRemove: () => void;
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
          {timecode(seg.start)}-{timecode(seg.end)}
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
            {/* Two different failures with two different fixes: the translator
                could not produce the target language (re-translate, then the
                voice), or the voice could not say it (re-voice). Both are
                stored as a keep with the original audio attached, which is why
                the row's badge alone cannot be trusted to say so. */}
            <span>
              {seg.keep_reason === "mt_failed"
                ? "Translation failed for this line: the dub falls back to the original audio. " +
                  "Re-translate it, then re-voice it."
                : state === "failed"
                  ? "Synthesis failed for this line: the dub falls back to the original audio. " +
                    "Re-voice it, or keep the original."
                  : `The clone said ${percent(seg.verify?.overlap ?? seg.tts?.overlap)} of it. ` +
                    "Re-voice it, or keep the original."}
            </span>
          </p>
        ) : null}

        {/* 1 the verdict. Two states, both named, neither of them jargon. */}
        <section className="flex flex-col gap-1.5">
          <div className="flex overflow-hidden rounded-lg border border-border bg-raised">
            {/* Each half only acts when it is *not* already the verdict:
                pressing the one that is already on is a no-op the user can
                see, and it must be a no-op the server never hears it would
                otherwise stamp a `keep` lock on a line nobody changed. */}
            <Choice
              active={!seg.keep}
              onClick={() => seg.keep && onVerdict(false)}
              label="Dub it"
              state="dubbed"
            />
            <span className="w-px shrink-0 bg-border" aria-hidden />
            {/* A pipeline-failed line is stored keep=true, but that is the
                pipeline's loss, not a verdict — so neither half renders
                active, and pressing Keep settles it as the user's own
                (keep_reason "manual"), which IS a change the server hears. */}
            <Choice
              active={seg.keep && !pipelineFailed(seg)}
              onClick={() => (!seg.keep || pipelineFailed(seg)) && onVerdict(true)}
              label="Keep original"
              state="kept"
            />
          </div>
          {/*
            What the other half of the control will *do*, not just what it will
            mean. Flipping to a dub is not a free relabelling: `edit.set_keep`
            drops the subtitle and the clip, so the line has to be translated
            and voiced again two jobs, or one when the user's own translation
            is locked and survives the flip. Saying "synthesized speech
            replaces the source audio" and then queueing nothing is how two
            lines ended up dubbed in name only.

            The other direction cost nothing to say and said nothing about its
            cost. `set_keep` invalidates translate in *both* directions, so
            pressing Keep throws the line's translation away the one
            destructive action in this app that neither asked nor warned, and
            the one `k` fires on a single keystroke. It says so now, and the
            flip itself leaves an undo behind.
          */}
          <p className="text-[11px] leading-snug text-muted">
            {seg.keep ? (
              <>
                The source audio plays untouched. “Dub it”{" "}
                {(seg.text_en ?? "").trim() && seg.locked?.text_en
                  ? "queues voice for this line: your translation is kept."
                  : "queues translate + voice for this line."}
              </>
            ) : seg.media?.fallback ? (
              // No dub exists yet — the mix already plays the original here.
              // Promising "the dubbed voice replaces the source audio" on this
              // line is false, and pressing Keep changes nothing audible; say
              // both, and say what Keep *does* do (settle the verdict).
              <>
                No dub yet: the mix plays the original audio here. “Keep original”
                makes that the verdict, so a re-run stops retrying this line; nothing
                you hear changes.
              </>
            ) : (seg.text_en ?? "").trim() && seg.locked?.text_en ? (
              // A hand-written line is locked, and `invalidate` honours locks:
              // it survives the flip as a subtitle. Warning about a loss that
              // cannot happen is the same failure as staying silent about one
              // that can.
              <>
                The dubbed voice replaces the source audio. Switching to “Keep original”
                leaves your translation in place, as a subtitle.
              </>
            ) : (
              <>
                The dubbed voice replaces the source audio. Switching to “Keep original”
                discards this line’s translation: a later re-run makes a new one.
              </>
            )}
            {/* The reason in words. `keep_reason` is a manifest token and
                the user's own verdict comes back from a headless re-run spelled
                `user`, so the panel used to tell someone "Kept because user"
                about a button they had pressed themselves. */}
            {seg.keep && seg.keep_reason ? <> Kept because {keepReason(seg.keep_reason)}.</> : null}
          </p>
        </section>

        {/* 2 the two model actions. Both cost a model load and a place in the
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
          {/*
            A keep has nothing to voice except the keeps the pipeline made
            because the voice failed. `edit.resynthesize` reopens those on the
            way in (`invalidate` undoes a `tts_failed` verdict), so asking for
            the voice is exactly the right button and disabling it left the
            failure with no way out but flipping the verdict by hand.
          */}
          <ModelAction
            icon={Volume2}
            label="Re-voice this line"
            cost="~1 min · queues behind the running job"
            disabled={busy || (seg.keep && !pipelineFailed(seg))}
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
            {/*
              "inherit" is the empty string, and the empty string is what
              *clears* the override `edit.set_langs` pops the key on a blank
              value, while `null` is "not supplied, leave it alone" for every
              field in the patch body (`app.py::PatchSegment`). Sending null was
              a control that looked like it worked and could never undo itself.
            */}
            <Field label="Spoken">
              <Select
                aria-label="Spoken in this line"
                value={seg.src_lang ?? ""}
                onChange={(event) => onPatch({ src_lang: event.currentTarget.value })}
              >
                <LangOptions langs={SRC_LANGS} />
              </Select>
            </Field>
            <Field label="Translate into">
              <Select
                aria-label="Translate this line into"
                value={seg.tgt_lang ?? ""}
                onChange={(event) => onPatch({ tgt_lang: event.currentTarget.value })}
              >
                <LangOptions langs={TGT_LANGS} />
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
              Both of these discard the translation and the clip for every segment they touch:
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
                Merge with previous
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
                Merge with next
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
                  <code className="font-mono text-secondary">{locked.join(", ")}</code>: edited
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

        {/*
          Remove, on the surface and last and deliberately not next to
          "Keep original". They read as the same thing and are not: a keep is a
          *verdict* about a passage the pipeline found the segment stays, it
          is placed, and the mix ducks the bed away under its original audio.
          Removing says there is no segment here at all (transcript noise, a
          hallucinated line, a fragment belonging to nobody), and the record
          goes.

          It used to live inside the "Timing & languages" shelf, which is a
          label that does not suggest deletion two levels down from a line the
          reviewer has already judged as noise. The bottom of the panel is
          where a destructive control belongs: visible without opening
          anything, and past everything that fixes a line rather than ends it.
          ⌫ is the same action without the confirm, guarded by ⌘Z instead.

          What the viewer hears afterwards is the sentence that matters, and
          it is not silence: `dubbing/mix.py` adds the original vocals into
          every span no placement claims, so a removed span sounds exactly
          like a passage the pipeline never detected. Saying so here is the
          difference between an informed delete and a surprise.
        */}
        <div className="mt-3 flex flex-col gap-1.5" data-remove-section>
          <Eyebrow>Remove from the dub</Eyebrow>
          <p className="text-[11px] leading-snug text-muted">
            Deletes this segment. The span is not muted: the mix plays the original
            audio wherever no segment claims it, the same as a passage the pipeline
            never found. To play the original <em>deliberately</em>, use “Keep original”
            above, which keeps the line and its subtitle. ⌫ removes without asking;
            ⌘Z restores a removed line.
          </p>
          <ConfirmButton
            size="sm"
            align="left"
            className="self-start"
            confirmLabel="Remove"
            message="Remove this segment. Its translation and clip go with it, and the span plays the original audio. Re-running the segments stage would bring it back."
            onConfirm={onRemove}
          >
            <Trash2 className="h-3.5 w-3.5" />
            Remove segment
          </ConfirmButton>
        </div>
      </div>
    </div>
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
 * a tooltip because the decision it informs "is it worth waiting a minute for
 * this line" is made *before* the click, and a tooltip arrives after you have
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
