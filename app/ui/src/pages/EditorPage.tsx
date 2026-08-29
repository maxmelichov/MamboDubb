/**
 * The editor.
 *
 * Three regions, and the proportions are the argument:
 *
 *   header 44px  where you are, and the one expensive button
 *   main row     SCRIPT (58%, and it grows) | VIEWER (42%, fixed column)
 *   timeline     112px, full width, at the very bottom
 *
 * The previous shape had four stacked regions with the *video* as the pane that
 * grew and the text as a fifteen-rem drawer under the timeline. That is the
 * layout of a tool for cutting picture. This is a tool for checking a
 * translation against its original, and the artefact being checked is text so
 * the text takes the room, the picture is the reference beside it, and the
 * timeline is a map along the bottom. Final Cut's grammar (viewer top-right,
 * index left, timeline across the bottom), with the index promoted to the work
 * surface because here it is the work.
 *
 * Below ~1200px the viewer column collapses: the video becomes a compact strip
 * over the selection panel rather than a column of its own, because a 300px
 * video next to a 300px script is two things too small to use.
 *
 * ## Three type sizes
 *
 * 11 / 12.5 / 14, and nothing else, across every component this screen owns:
 * 11 for meta and hints, 12.5 for body and controls, 14 for the two things that
 * are the point a translation line and a panel heading. The only exception is
 * the 9px on the timeline's axis ticks, which are chart furniture and were
 * already there. A dense tool drifts into eight sizes one component at a time;
 * three is what makes the hierarchy readable rather than merely present.
 *
 * ## One keyboard listener
 *
 * There used to be two this page's, and a second inside ABPlayer whose guard
 * only checked `INPUT|TEXTAREA|SELECT`, so typing "a" or "b" into any
 * contentEditable started playing audio at the user. There is exactly one now,
 * with exactly one guard, and every binding goes through it.
 *
 * The whole screen stays usable while a job runs. Nothing here disables itself
 * on a running job except the two model actions, which queue.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { Film, Loader2 } from "lucide-react";
import { AppHeader } from "../components/AppShell";
import { JobBar } from "../components/JobBar";
import { PreviewPlaceholder } from "../components/PreviewPlaceholder";
import { RunMenu } from "../components/RunMenu";
import { RunSummary } from "../components/RunSummary";
import { ScriptPane, filterSegments, type ScriptFilter } from "../components/ScriptPane";
import { SelectionPanel } from "../components/SelectionPanel";
import { ShortcutHelp } from "../components/ShortcutHelp";
import { Timeline } from "../components/Timeline";
import { UndoKeepStrip, type KeptUndo } from "../components/UndoKeepStrip";
import { VideoPlayer, type TransportMode } from "../components/VideoPlayer";
import { OpenFileButton } from "../components/runFiles";
import type { EditTarget } from "../components/ScriptRow";
import { Badge, ConfirmButton, Empty, ErrorBar } from "../components/ui";
import { api } from "../lib/api";
import { cn } from "../lib/classNames";
import { stopClip, toggleClip, useClipPlayback } from "../lib/clipAudio";
import { FIXTURE_PROJECT } from "../lib/fixtures";
import { lineCount } from "../lib/format";
import {
  hasTranscript,
  needsModelWork,
  pipelineFailed,
  segmentState,
  totalDuration,
  type SegmentState,
  type Span,
} from "../lib/segments";
import { bucketsFor, usePeaks } from "../lib/usePeaks";
import { activeJob, useProject } from "../lib/useProject";
import { useTransport } from "../lib/useTransport";
import type { ProjectOptionsPatch, Segment } from "../lib/types";

// The button ladder. 32 px/s was the old ceiling and it made fine retiming a
// guess — a 100 ms nudge was three pixels. 256 px/s puts a syllable on screen.
// The ladder is what +/− walk; the pinch/⌘-scroll zoom moves freely between
// the fit floor and ZOOM_MAX and is not snapped to it.
const ZOOM_STEPS = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256];
const ZOOM_MAX = ZOOM_STEPS[ZOOM_STEPS.length - 1];

/** The run's own audio, written by `fetch` (dubbing/fetch.py) long before `mix`. */
const SOURCE_AUDIO = "source.wav";

/**
 * How long the undo behind a keep stays reachable.
 *
 * Long enough to read the row you just changed and notice it was the wrong one;
 * short enough that it is gone before the next decision, so the strip is never
 * something to dismiss.
 */
const UNDO_MS = 6000;

/** How long an announcement stays in the live region before it empties. */
const ANNOUNCE_MS = 4000;

export function EditorPage() {
  const params = useParams<{ name: string }>();
  const name = params.name ?? FIXTURE_PROJECT;
  const [state, actions] = useProject(name);
  const { segments, project, selectedUid } = state;

  const total = useMemo(
    () => totalDuration(segments, project?.source.duration ?? null),
    [project, segments],
  );
  const transport = useTransport(total);
  const [zoom, setZoom] = useState(4);

  /**
   * The two lane waveforms.
   *
   * Buckets come from the window's width because the strip is full-bleed, and
   * they are fixed for the session: the picture is an SVG that stretches, so
   * zooming does not need a new one. The OUTPUT lane re-reads when a job
   * finishes a re-voice or a render is exactly when the mix on disk stops
   * matching what is drawn while the SOURCE lane never does, because
   * `source.wav` is written once by the fetch stage.
   */
  const [buckets] = useState(() =>
    bucketsFor(typeof window === "undefined" ? 1400 : window.innerWidth),
  );
  const finishedJobs = state.jobs.filter((job) => job.status === "done").length;
  const sourcePeaks = usePeaks(name, "source", buckets);
  const dubPeaks = usePeaks(name, "dub", buckets, finishedJobs);

  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<ScriptFilter>("all");
  const [editing, setEditing] = useState<EditTarget>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const playingUrl = useClipPlayback();

  const selected = useMemo(
    () => segments.find((seg) => seg.uid === selectedUid) ?? null,
    [segments, selectedUid],
  );

  const visible = useMemo(
    () => filterSegments(segments, query, filter),
    [filter, query, segments],
  );

  /**
   * Which lines the search matches, for the strip.
   *
   * The *search* only, never the chips: a filter already empties the list it
   * governs, and dimming the timeline for it as well would leave a strip with
   * eleven marks on it and no picture of the run they came out of. Null when
   * nothing is being searched for, which is how the timeline knows to draw
   * every mark at full strength rather than being handed a set that happens to
   * contain all of them.
   */
  const matchedUids = useMemo(
    () =>
      query.trim()
        ? new Set(filterSegments(segments, query, "all").map((seg) => seg.uid))
        : null,
    [query, segments],
  );

  /**
   * What the transport plays, and why.
   *
   * `preview.mp4` is written by `mix`, at the very end of the pipeline but
   * `source.wav` is written by `fetch`, at the very start, and for the whole
   * hour in between the transport had a live play button attached to nothing:
   * "i can start playing. but because it still not ready it shown nothing". So
   * the original audio stands in until the preview exists, and when it does not
   * exist either the play button is honestly dead.
   *
   * Both URLs are composed through the api seam rather than as `/media/…`
   * strings: the desktop shell's origin is the Tauri asset protocol, so a
   * root-relative path handed to a media element there resolves against the
   * wrong origin and plays nothing which is exactly the bug that killed every
   * A/B button once already. `audioUrl` is the second half of that seam, and
   * the half that makes fixture mode's synthetic clip loadable.
   *
   * `outputs.preview` and not `previewUrl` decides the mode, because the
   * manifest is what knows whether the file exists; the URL is only how it is
   * reached, and in fixture mode there is no video to reach.
   */
  const previewPath = project?.outputs.preview ?? null;
  const previewUrl = previewPath ? api.mediaUrl(name, previewPath) || null : null;
  const fetched = project?.stages.fetch === "done";
  const sourceUrl = useMemo(
    () => (!previewPath && fetched ? api.audioUrl(api.mediaUrl(name, SOURCE_AUDIO)) : null),
    [fetched, name, previewPath],
  );
  const transportMode: TransportMode = previewPath ? "preview" : sourceUrl ? "source" : "none";
  const mediaUrl = previewUrl ?? sourceUrl;

  /**
   * Nothing to play means nothing playing.
   *
   * The mode changes under a running transport more often than it looks:
   * `/editor/a` → `/editor/b` is the same route, so this component is not
   * remounted and the clock carries across. A playhead sweeping a run that has
   * no audio at all is the same lie the live play button was telling.
   *
   * `mode` alone was not the whole test. A run whose manifest names a
   * `preview.mp4` is in `preview` mode whether or not that file can actually be
   * reached fixture mode never can, and a deleted or half-written render is
   * the same shape so the clock swept a playhead across an empty strip with a
   * live play button under it. The transport is silent when there is no URL,
   * and `space` is the same control as the button, so it refuses in the same
   * case rather than being a second opinion about it.
   */
  const playable = transportMode !== "none" && mediaUrl != null;
  const pause = transport.pause;
  useEffect(() => {
    if (!playable) pause();
  }, [pause, playable]);

  const counts = useMemo(() => {
    const out: Record<SegmentState, number> = {
      dubbed: 0,
      kept: 0,
      failed: 0,
      unplaced: 0,
      unvoiced: 0,
      untranslated: 0,
    };
    for (const seg of segments) out[segmentState(seg)] += 1;
    return out;
  }, [segments]);

  /**
   * Selecting a segment also moves the playhead to it they are one gesture.
   *
   * Depends on `transport.seek` rather than on `transport`: the transport hook
   * returns a fresh object literal every render, so closing over the whole
   * thing would rebuild this callback four times a second during playback and
   * re-render all two hundred memoized rows with it. `seek` itself is stable.
   */
  const seek = transport.seek;
  const selectAndSeek = useCallback(
    (uid: string) => {
      actions.select(uid);
      const seg = segments.find((s) => s.uid === uid);
      if (seg) seek(seg.start);
    },
    [actions, seek, segments],
  );

  /**
   * Picking a mark on the strip is a question about a *line*: what does it say,
   * what did it become, what does it sound like. All three answers are in the
   * script row its two texts and its A/B buttons so the click has to put
   * that row in front of the reviewer, in the middle of the list, rather than
   * leave them to find a row seven hundred pixels up.
   *
   * It is a counter and not just a uid because clicking the same mark twice
   * must scroll twice: the reviewer has scrolled away in between, which is
   * exactly why they clicked it again.
   */
  const [reveal, setReveal] = useState<{ uid: string; n: number } | null>(null);
  const revealRow = useCallback((uid: string) => {
    setReveal((current) => ({ uid, n: (current?.n ?? 0) + 1 }));
  }, []);
  const selectFromTimeline = useCallback(
    (uid: string) => {
      selectAndSeek(uid);
      revealRow(uid);
    },
    [revealRow, selectAndSeek],
  );

  /**
   * Opening a field is asking for the row, so the row comes to the middle.
   *
   * Selection scrolls with `block: "nearest"`, which is right for selecting: it
   * moves the list as little as it can. It is wrong for *editing*. A row at the
   * bottom of the list is "nearest" where it already is, so pressing ↵ on it
   * opened a textarea in the last thirty pixels of the pane and the field then
   * grew downward, under the timeline strip, with the caret in the part that was
   * covered. Every route into a field goes through here (↵, a click on a
   * translation, the row menu's Correct transcript) and every one of them means
   * the same thing, so the centring is at the seam rather than at three of them.
   */
  const startEditing = useCallback(
    (target: EditTarget) => {
      setEditing(target);
      if (target) revealRow(target.uid);
    },
    [revealRow],
  );

  /** ↑/↓ walk what is on screen, not what is in the run. */
  const step = useCallback(
    (delta: number) => {
      if (visible.length === 0) return;
      const index = visible.findIndex((seg) => seg.uid === selectedUid);
      const next = Math.max(0, Math.min(visible.length - 1, (index < 0 ? 0 : index) + delta));
      selectAndSeek(visible[next].uid);
    },
    [selectAndSeek, selectedUid, visible],
  );

  /**
   * Zoom, with a floor, and one press that goes to it.
   *
   * The steps are the ladder; the floor is a measurement. `fit` is the scale at
   * which the run is exactly as wide as the strip below it the marks are a
   * smear against the left edge of an otherwise empty lane, which is the one
   * zoom level that cannot answer a question, and there was no way back to a
   * useful scale except pressing + and counting. So zoom-out lands on the next
   * step down *or* on the fit scale, whichever is larger, and Fit goes straight
   * there. Zoom-in walks the ladder from wherever it is, which is what lets it
   * climb off a fractional fit value.
   *
   * The measurement comes up from the timeline (it owns the container) and is
   * kept in a ref as well as in state: the ref is what the zoom callbacks read,
   * so they stay stable and the one keyboard listener that closes over them is
   * not rebuilt on every resize.
   */
  const [fitZoom, setFitZoom] = useState<number | null>(null);
  const fitRef = useRef<number | null>(null);
  fitRef.current = fitZoom;
  const onViewport = useCallback(
    (width: number) => {
      setFitZoom(width > 0 && total > 0 ? width / total : null);
    },
    [total],
  );

  const zoomOut = useCallback(() => {
    setZoom((z) => {
      const step = [...ZOOM_STEPS].reverse().find((s) => s < z - 1e-6) ?? z;
      const floor = fitRef.current;
      return floor != null ? Math.max(step, Math.min(z, floor)) : step;
    });
  }, []);
  const zoomIn = useCallback(() => {
    setZoom((z) => ZOOM_STEPS.find((s) => s > z + 1e-6) ?? z);
  }, []);
  const zoomFit = useCallback(() => {
    const floor = fitRef.current;
    if (floor != null) setZoom(floor);
  }, []);
  // The continuous zoom: a multiplicative factor from a pinch or ⌘/Ctrl-scroll
  // on the strip, clamped to the same floor the − button lands on and to
  // ZOOM_MAX. Multiplicative because zoom is perceptually a ratio — the same
  // gesture should feel the same at 2 px/s and at 200.
  const zoomBy = useCallback((factor: number) => {
    setZoom((z) => {
      const floor = fitRef.current;
      const lo = floor != null ? Math.min(floor, ZOOM_MAX) : ZOOM_STEPS[0];
      return Math.min(ZOOM_MAX, Math.max(lo, z * factor));
    });
  }, []);

  /**
   * The gap the reviewer is pointing at, lit on the strip.
   *
   * The run summary lists audible speech nothing covers as a column of
   * timecodes, and a timecode is only half an answer: "0:52" does not say
   * whether that is early, late, next to a run of failures, or in the middle of
   * the one stretch that is already fine. Pointing at a row lights the matching
   * hatch, so the list and the map are one thing.
   */
  const [highlightGap, setHighlightGap] = useState<Span | null>(null);

  /**
   * The unclaimed span the reviewer clicked on the strip, waiting to be claimed.
   *
   * The rail's gap list is where adding lives, and it is fed by the report —
   * which is a finding from the last render, so a gap opened *since* (a split,
   * a retime, a remove) has a hatch on the strip and no row in the list.
   * Clicking the hatch is the answer: the click clears the selection (the run
   * summary is the surface that carries the composer), seeks there, and pins
   * the span at the top of the list so it has an add control whether or not
   * the report ever heard of it. It clears itself the moment a line is
   * selected again which includes the line the add itself mints.
   */
  const [claimGap, setClaimGap] = useState<Span | null>(null);
  const claimFromTimeline = useCallback(
    (span: Span) => {
      actions.select(null);
      seek(span.start);
      setClaimGap(span);
    },
    [actions, seek],
  );
  useEffect(() => {
    if (selectedUid) setClaimGap(null);
  }, [selectedUid]);
  useEffect(() => setClaimGap(null), [name]);

  /**
   * Make these lines playable: translate the ones with nothing to say, then
   * voice all of them.
   *
   * Two jobs at most, never 2N the one-job queue runs them FIFO, so the
   * translate lands before the voice that needs it, and only one model is
   * resident at a time either way. A line that already has a translation skips
   * the translator: it is either the pipeline's, in which case re-running it
   * changes nothing, or the user's, in which case re-running it would throw
   * their sentence away.
   */
  const queueDubWork = useCallback(
    async (segs: Segment[]) => {
      // A line that has its clip and only lacks a placement is not model work:
      // re-voicing it costs a model load to produce the clip it already has,
      // and leaves it exactly as unplaced as before. Render is its button.
      //
      // Nor is a span with nothing written on it: `fill_uncovered_audible`
      // writes `text: ""` for audible stretches the transcript never claimed,
      // and translating an empty string and voicing the result replaces correct
      // original audio with garbage. The bulk bars filter these out of the sets
      // they offer; this is the same rule at the seam every caller goes through,
      // including the single-line flip.
      const work = segs.filter((seg) => needsModelWork(seg) && hasTranscript(seg));
      const uids = work.map((seg) => seg.uid);
      if (uids.length === 0) return;
      /*
       * Both jobs wear the same batch id, because they are one decision.
       *
       * Only this function knows that: the server sees two POSTs on two routes
       * and has no way to tell a pair from two unrelated requests a second
       * apart. Without the id, cancelling the translate left the voice job to
       * run on lines whose translation had just been abandoned 27 lines
       * synthesised from nothing. With it, one DELETE reaches both.
       */
      const batch = `b${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
      const untranslated = work.filter((seg) => !(seg.text_en ?? "").trim());
      if (untranslated.length > 0) {
        await actions.retranslate(untranslated.map((seg) => seg.uid), batch);
      }
      await actions.resynthesize(uids, batch);
    },
    [actions],
  );

  /**
   * The editor's one spoken channel.
   *
   * Almost everything this screen says, it says by *changing* a row's badge
   * flips, the strip appears, the marks get wider and a change with no text is
   * a change a screen reader cannot report. Three kinds of event were silent and
   * are the three worth saying: the verdict, because `k` is a single keystroke
   * that rewrites a line's future; the zoom, because its only readout is a
   * number in the corner of a strip; and a job starting or ending, because those
   * happen without being asked for. One region, polite, so they queue behind
   * whatever the user is reading rather than interrupting it.
   */
  const [spoken, setSpoken] = useState("");
  const say = useCallback((message: string) => setSpoken(message), []);

  /*
   * And empties itself again.
   *
   * A live region that keeps its last sentence is a sentence still on the page:
   * it is read again by anything that walks the document, and it makes the same
   * announcement twice in a row impossible the region only speaks when its
   * contents *change*, so a second "Kept #17" after a first would be silent.
   * Going back to empty after a beat fixes both.
   */
  useEffect(() => {
    if (!spoken) return;
    const timer = window.setTimeout(() => setSpoken(""), ANNOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [spoken]);

  /* Skips the mount, which is not a change otherwise every run announces its
     own default zoom on open. */
  const firstZoom = useRef(true);
  useEffect(() => {
    if (firstZoom.current) {
      firstZoom.current = false;
      return;
    }
    say(`Timeline zoom ${zoom >= 10 ? Math.round(zoom) : Math.round(zoom * 10) / 10} pixels per second`);
  }, [say, zoom]);

  /**
   * What the last ⌘Z / ⌘⇧Z did, said out loud.
   *
   * Undo has no cursor: the row it changed may be filtered out, scrolled away,
   * or — when the entry was a barrier or had gone stale (`useHistory.ts`) —
   * not changed at all, and those three must not feel identical. The sentence
   * comes from `useProject`, which is the layer that knows what happened; this
   * is only the strip it appears on. Same shape as the keep toast (transient,
   * bottom-centre, `role="status"` so it is its own announcement), and a
   * counter rather than the string alone because pressing ⌘Z twice on an empty
   * stack must re-arm the timer even though the words did not change.
   */
  const [notice, setNotice] = useState<{ text: string; n: number } | null>(null);
  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(null), ANNOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [notice]);
  const nudgeHistory = useCallback(
    (direction: "undo" | "redo") => {
      void actions[direction]().then((text) => {
        setNotice((current) => ({ text, n: (current?.n ?? 0) + 1 }));
      });
    },
    [actions],
  );

  /**
   * ⌫ on a selected line the keyboard path to the panel's Remove.
   *
   * Instant, like `k`, and for the same reason: pruning transcript noise out
   * of a run is dozens of these, and a confirm on each is a tool nobody uses.
   * What makes instant honest is the same pair `k` leans on: the removal is on
   * the undo stack (`useProject.remove` records its own inverse), and the
   * receipt says so in the strip and out loud. The receipt waits for the
   * server the one refusal here is a 409 while a job runs, and announcing a
   * removal beside an error bar denying it would be two answers to one press.
   */
  const removeSelected = useCallback(
    (seg: Segment) => {
      void actions.remove(seg.uid).then((removed) => {
        if (!removed) return;
        say(`Removed #${seg.id}`);
        setNotice((current) => ({
          text: `Removed #${seg.id}. ⌘Z restores the line`,
          n: (current?.n ?? 0) + 1,
        }));
      });
    },
    [actions, say],
  );

  /** The undo behind the one destructive verdict see `setVerdict` below. */
  const [undoKeep, setUndoKeep] = useState<KeptUndo | null>(null);
  useEffect(() => {
    if (!undoKeep) return;
    const timer = window.setTimeout(() => setUndoKeep(null), UNDO_MS);
    return () => window.clearTimeout(timer);
  }, [undoKeep]);
  // The kept line the undo toast is about, resolved live (see the toast's
  // render-site comment).
  const undoKept = useMemo(
    () => (undoKeep ? (segments.find((seg) => seg.uid === undoKeep.uid) ?? null) : null),
    [segments, undoKeep],
  );

  /**
   * The verdict, and everything it implies.
   *
   * `PATCH {keep:false}` on its own is what the bug was: `edit.set_keep`
   * invalidates the translate stage in both directions, so flipping a kept line
   * to "Dub it" leaves it with no subtitle and no clip and nothing queued to
   * make either. The line then sits in the run forever, dubbed in name only.
   * So the flip *is* the PATCH plus the work it creates, in that order, and the
   * server's answer decides which work: a hand-written translation is locked
   * and survives the flip, and re-translating over it would discard the very
   * thing the user typed.
   *
   * The other direction queues nothing the original audio is already on disk
   * and cancels nothing either: a running job belongs to whatever asked for
   * it, not to this line.
   *
   * `queue` is how a caller flipping *many* lines borrows this without
   * borrowing 2N jobs: it takes the PATCH, the lock and the optimistic predict
   * exactly as they are, and queues the work itself, once, over the lot. The
   * saved segment comes back for that reason the server's answer is what says
   * whether the translation survived the flip.
   *
   * The *other* other direction is the one nothing guarded. `set_keep`
   * invalidates translate both ways, so flipping a line TO keep throws its
   * translation away and that flip is a button, a menu item, and `k`, a single
   * keystroke with no modifier. A confirm on it would be the wrong control:
   * judging a run is a hundred of these decisions and a dialog on each is a tool
   * nobody uses. So the flip stays instant and leaves an undo behind, holding
   * the *whole* of what was lost the server drops the line a moment later and
   * nothing else on the client remembers it.
   */
  const setVerdict = useCallback(
    async (seg: Segment, keep: boolean, queue = true): Promise<Segment | null> => {
      // The verdict is (keep, keep_reason), not the boolean alone: a line the
      // pipeline gave up on is stored keep=true with a *failure* reason, and
      // pressing "Keep original" on it is not a no-op — it settles the line as
      // the user's own verdict (reason "manual", passthrough stamped), turning
      // a red Failed into a green Keep that re-runs honour.
      if (seg.keep === keep && !(keep && pipelineFailed(seg))) return null;
      const lost = (seg.text_en ?? "").trim() || null;
      const saved = await actions.patch(
        seg.uid,
        keep ? { keep: true, keep_reason: "manual" } : { keep: false },
      );
      if (!saved) return null;
      if (keep) {
        setUndoKeep({ uid: seg.uid, id: seg.id, text_en: lost });
        say(`Kept #${seg.id}`);
        return saved;
      }
      say(`Dubbing #${seg.id}`);
      if (queue) await queueDubWork([saved]);
      return saved;
    },
    [actions, queueDubWork, say],
  );

  /**
   * Undo: back to a dub, with the sentence that was there.
   *
   * One PATCH and not two, because `_apply_patch` runs `set_text` before
   * `set_keep` and `invalidate` honours locks so the restored line is written,
   * locked, and then survives the very flip that is being undone. It is locked
   * afterwards, which is a real difference from the state before the keep and is
   * why the strip says so: a restored line is now the user's, and a re-run will
   * not replace it.
   *
   * The work is queued for the same reason the ordinary "Dub it" queues it: the
   * clip is gone and only a job can make another one. A line handed back with no
   * audio and nothing coming is exactly the limbo this flip used to create.
   */
  const undoLastKeep = useCallback(() => {
    const target = undoKeep;
    if (!target) return;
    setUndoKeep(null);
    void (async () => {
      const saved = await actions.patch(
        target.uid,
        target.text_en ? { keep: false, text_en: target.text_en } : { keep: false },
      );
      if (saved) await queueDubWork([saved]);
    })();
  }, [actions, queueDubWork, undoKeep]);

  const toggleKeep = useCallback(
    (seg: Segment) => void setVerdict(seg, !seg.keep),
    [setVerdict],
  );

  /** Row-menu path for a pipeline-failed line: keep, as the user's verdict. */
  const settleKeep = useCallback((seg: Segment) => void setVerdict(seg, true), [setVerdict]);

  /** The script pane's bulk fix, over whatever the filter has selected. */
  const fixMany = useCallback(
    (uids: string[]) => {
      const wanted = new Set(uids);
      void queueDubWork(segments.filter((seg) => wanted.has(seg.uid)));
    },
    [queueDubWork, segments],
  );

  /**
   * The same flip, over a filtered set: "dub these too".
   *
   * The case is a run the keep rule got wrong a video of a third language
   * read as "already the target", kept whole, and audibly the original. The
   * only way out was one line at a time, N times, and a run is two hundred
   * lines long.
   *
   * The PATCHes go one at a time and in order, because each one is a distinct
   * lock and the server answers each with the segment it produced; the work
   * goes at the end, over every segment the server gave back, so a hundred
   * flips still cost at most one translate and one voice. Sequential and not
   * `Promise.all` on purpose: the flips must all land before the translate is
   * enqueued, or the job runs against lines the server has not written yet.
   */
  const dubMany = useCallback(
    (uids: string[]) => {
      const wanted = new Set(uids);
      void (async () => {
        const flipped: Segment[] = [];
        for (const seg of segments) {
          if (!wanted.has(seg.uid) || !seg.keep) continue;
          const saved = await setVerdict(seg, false, false);
          if (saved) flipped.push(saved);
        }
        /*
         * Follow the lines, do not strand the user where they were.
         *
         * The gesture only exists on the Kept filter, and it empties it every
         * line it flipped has left. So the screen the user was reading became
         * "Nothing matches" at the exact moment their twenty-seven lines
         * started being worked on, with no way to watch it happen. Unfinished
         * is where those lines now are, until the jobs finish and they leave
         * that too.
         */
        if (flipped.length > 0) setFilter("unfinished");
        await queueDubWork(flipped);
      })();
    },
    [queueDubWork, segments, setVerdict],
  );

  /**
   * A committed edit is a PATCH unless nothing was edited.
   *
   * Every way out of the field goes through here (blur, ⌘↵, the ⋯ menu's
   * Correct transcript), and closing a field you only opened to *read* must
   * cost nothing: no request, no lock stamped on the line, and above all no
   * invalidation `edit.set_text` drops the clip and the placement, so a
   * no-op save is a re-voice queued for a line nobody changed. The comparison
   * is trimmed on both sides because the editor commits a trimmed draft, and
   * `"a line "` and `"a line"` are not a translation the user rewrote.
   */
  const commit = useCallback(
    (uid: string, field: "text" | "text_en", value: string) => {
      setEditing(null);
      const seg = segments.find((s) => s.uid === uid);
      if (!seg) return;
      const current = (field === "text" ? seg.text : (seg.text_en ?? "")).trim();
      if (value.trim() === current) return;
      void actions.patch(uid, { [field]: value });
    },
    [actions, segments],
  );

  const splitAt =
    selected && transport.currentTime > selected.start + 0.05 &&
    transport.currentTime < selected.end - 0.05
      ? transport.currentTime
      : null;

  // Playback of a clip is per-selection: moving on must not leave the previous
  // line's audio sounding under the new one.
  useEffect(() => stopClip, []);
  useEffect(() => {
    stopClip();
  }, [selectedUid]);

  /**
   * The editor's only keydown listener, with the editor's only guard.
   *
   * ⌘F is checked before the modifier bail, because it is the one binding that
   * *is* a modifier chord; everything after it is a bare key and a bare key
   * pressed while a field has focus belongs to the field.
   */
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key === "f") {
        event.preventDefault();
        searchRef.current?.focus();
        searchRef.current?.select();
        return;
      }
      const target = event.target as HTMLElement | null;
      if (
        target &&
        (/^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName) || target.isContentEditable)
      ) {
        return;
      }
      /*
       * ⌘Z / ⌘⇧Z (and Ctrl+Y, for hands that learned redo there). After the
       * field guard on purpose, unlike ⌘F: a field has its own undo — the
       * browser's, over keystrokes — and hijacking it mid-sentence to revert
       * some other line's PATCH would be two histories fighting over one key.
       * Outside a field, the editor's history is the only one there is.
       */
      if ((event.metaKey || event.ctrlKey) && !event.altKey) {
        const key = event.key.toLowerCase();
        if (key === "z" || key === "y") {
          event.preventDefault();
          nudgeHistory(key === "y" || event.shiftKey ? "redo" : "undo");
          return;
        }
      }
      if (event.metaKey || event.ctrlKey || event.altKey) return;

      switch (event.key) {
        case " ":
          event.preventDefault();
          // Nothing on disk to play: the button is disabled and the key is the
          // same control, so it refuses in the same case rather than starting a
          // clock over silence.
          if (playable) transport.toggle();
          break;
        case "ArrowDown":
          event.preventDefault();
          step(1);
          break;
        case "ArrowUp":
          event.preventDefault();
          step(-1);
          break;
        case "Enter":
          if (selected) {
            event.preventDefault();
            startEditing({ uid: selected.uid, field: "text_en" });
          }
          break;
        case "Escape":
          setEditing(null);
          stopClip();
          break;
        case "a":
          if (selected) {
            event.preventDefault();
            // The source side is a window of the whole track same contract
            // as the row's A button, and the same reason.
            toggleClip(selected.media?.source ?? null, selected.media?.source_window ?? null);
          }
          break;
        case "b":
          if (selected) {
            event.preventDefault();
            toggleClip(selected.media?.play ?? selected.media?.tts ?? null);
          }
          break;
        case "k":
          if (selected) {
            event.preventDefault();
            toggleKeep(selected);
          }
          break;
        case "s":
          if (selected && splitAt != null) {
            event.preventDefault();
            void actions.split(selected.uid, splitAt);
          }
          break;
        case "Delete":
        case "Backspace":
          if (selected) {
            event.preventDefault();
            removeSelected(selected);
          }
          break;
        case "+":
        case "=":
          event.preventDefault();
          zoomIn();
          break;
        case "-":
          event.preventDefault();
          zoomOut();
          break;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [
    actions,
    nudgeHistory,
    playable,
    removeSelected,
    selected,
    splitAt,
    startEditing,
    step,
    toggleKeep,
    transport,
    zoomIn,
    zoomOut,
  ]);

  const job = activeJob(state.jobs);

  /*
   * What the video on screen is a render of.
   *
   * `render.at === null` is a run with no stamp old, or never mixed where
   * `changed` is 0 because there is nothing to compare against. That is why the
   * "N lines changed" phrasing is guarded on the count rather than on `stale`:
   * saying "0 lines changed" next to a Render button would be worse than quiet.
   */
  const render = project?.render ?? { at: null, stale: false, changed: 0 };
  const stale = render.stale;
  const changed = render.changed;
  // The button counts *lines*, because that is what it would re-render. These
  // two count *changes*, because they are about what the user did and "your
  // last 1 line" is not a phrase anybody says.
  const edits = changed === 1 ? "change" : `${changed} changes`;
  // `changed` needs a render-time baseline; a run mixed before the stamp
  // existed reports stale with changed=0 *forever*, so `changed > 0` alone
  // hid the staleness of exactly the runs that cannot count it — an edit on
  // such a run gave no header feedback at all. `state.edited` is the
  // tiebreak: this session touched something, so the staleness is real to
  // this user, even if the server cannot say how many lines it spans.
  const staleKnown = stale && (changed > 0 || state.edited);
  const staleNote = staleKnown
    ? changed > 0
      ? `rendered before your last ${edits}`
      : "rendered before your recent edits"
    : null;
  const staleBand = staleKnown
    ? changed > 0
      ? `Mixed before your last ${edits}. Update the video to hear them`
      : "Mixed before your recent edits. Update the video to hear them"
    : null;

  /*
   * When Render has nothing to work with, and when it is not the button it says.
   *
   * `mix` lays speech over `source.wav` for the length the manifest recorded, so
   * on a run that never got through `fetch` there is no bed and no duration —
   * and the button sat there, primary and inviting, on the one screen where it
   * could only produce a second failure twenty minutes later. That is the case
   * that is refused *here*, with the reason, rather than by a job that dies.
   *
   * Unfinished lines are a different thing and must NOT be refused:
   * `edit.start_stage` deliberately backs a render up to tts or translate to
   * make the missing work, because "refusing would leave the user with a button
   * that only works on runs that did not need it". What was wrong there was the
   * sentence "typically a few minutes" over a job that loads two models so
   * that is what changes, not the button's availability.
   */
  const renderBlocked =
    project == null
      ? null
      : project.stages.fetch !== "done"
        ? "Nothing to render yet: the mix lays speech over the audio the fetch stage " +
          "writes, and fetch has not finished."
        : segments.length === 0
          ? "Nothing to render yet: this run has no lines. The segments stage makes them."
          : null;
  const unfinishedLines = counts.untranslated + counts.unvoiced;

  /*
   * A run option changed here is written by the server and read back, never
   * predicted: `PATCH /api/projects/{name}` normalizes the context (trimmed, and
   * cleared when empty) and the reload is what puts the server's version on the
   * screen. Optimism would show a note that the server stored differently.
   */
  const reload = actions.reload;
  const saveOptions = useCallback(
    async (patch: ProjectOptionsPatch) => {
      await api.updateProject(name, patch);
      await reload();
    },
    [name, reload],
  );

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-plane">
      <AppHeader
        actions={
          <>
            {/*
              The finished file, one click from the run it came out of. It only
              exists when the file does and "the file does" is `previewUrl`,
              not `outputs.preview`. The manifest naming a preview and the app
              being able to reach one are two different facts, and while the
              button hung off the first of them it sat there offering to open a
              video three inches from a panel explaining that there is no video.
              One source of truth: if the panel cannot show it, the header does
              not offer it.
            */}
            {previewPath && previewUrl ? (
              <OpenFileButton name={name} path={previewPath} title={staleNote ?? undefined} />
            ) : null}
            {/*
              Quiet when the video is current, loud when it is not.
              The button was permanently accented, which made "there is work
              waiting" indistinguishable from "everything is up to date" the
              one thing this button is in a position to say. It names the
              number, because "some edits" is not a reason to spend minutes on a
              re-encode and "27 lines" is.
            */}
            <ConfirmButton
              variant={stale ? "primary" : "secondary"}
              size="sm"
              confirmLabel="Render"
              disabled={renderBlocked != null}
              title={renderBlocked ?? undefined}
              message={
                unfinishedLines > 0
                  ? `${unfinishedLines} line${unfinishedLines === 1 ? " has" : "s have"} no ` +
                    "clip yet, so this render backs up and makes them first: it loads the " +
                    "translator and the voice, which is model time, not the few minutes a " +
                    "re-encode takes."
                  : "Re-render the preview. This re-runs timeline, mix and report and replaces preview.mp4: a full video re-encode, typically a few minutes."
              }
              onConfirm={() => void actions.render()}
            >
              <Film className="h-3.5 w-3.5" />
              {stale && changed > 0
                ? `Update the video · ${lineCount(changed)} changed`
                : stale && render.at
                  ? "Update the video"
                  : "Render preview"}
            </ConfirmButton>
            <RunMenu
              project={project}
              name={name}
              counts={counts}
              /*
                The rail behind this menu is *already* the run summary whenever
                nothing is selected same tally, same gap list, permanently on
                screen, three inches to the left. So the menu drops that half of
                itself exactly then, and keeps the half the rail has never
                carried: the files, the metadata and the run options.
              */
              showHealth={selected != null}
              onSeek={transport.seek}
              onHighlightGap={setHighlightGap}
              onSaveOptions={saveOptions}
            />
            <ShortcutHelp />
          </>
        }
      >
        <span
          className="min-w-0 flex-1 truncate text-[12.5px] font-semibold text-primary"
          title={project?.source.title}
        >
          {project?.source.title ?? name}
        </span>
        {project ? (
          <Badge>
            {project.source.src_lang} → {project.source.tgt_lang}
          </Badge>
        ) : null}
      </AppHeader>

      {/* Renders nothing at all unless there is a job or the stream dropped. */}
      <JobBar
        jobs={state.jobs}
        stage={state.stage}
        connected={state.connected}
        log={state.log}
        onCancel={(id, batch) => void actions.cancel(id, batch)}
        onAnnounce={say}
      />
      {/*
        The one place this screen speaks. Off-screen, polite, never given
        anything a sighted user is not also being shown see `say`. It sits
        beside the strip because that is where two of the three announcements
        come from, and early in the document because a live region has to be in
        the tree before it is written to for the first write to be announced.
      */}
      <div role="status" aria-live="polite" aria-atomic="true" className="sr-only">
        {spoken}
      </div>
      {state.error ? <ErrorBar message={state.error} onDismiss={actions.dismissError} /> : null}

      {state.loading ? (
        <Empty title="Opening the run">
          <Loader2 className="mx-auto mt-2 h-5 w-5 animate-spin" aria-label="Loading" />
        </Empty>
      ) : (
        <>
          <div className="flex min-h-0 flex-1 flex-col xl:flex-row">
            <ScriptPane
              segments={segments}
              selectedUid={selectedUid}
              currentTime={transport.currentTime}
              busyUids={state.pendingUids}
              editing={editing}
              playingUrl={playingUrl}
              query={query}
              filter={filter}
              reveal={reveal}
              searchRef={searchRef}
              onQuery={setQuery}
              onFilter={setFilter}
              onSelect={selectAndSeek}
              onEdit={startEditing}
              onCommit={commit}
              onPlay={toggleClip}
              onToggleKeep={toggleKeep}
              onSettleKeep={settleKeep}
              onRetranslateMany={(uids) => void actions.retranslate(uids)}
              onResynthesizeMany={(uids) => void actions.resynthesize(uids)}
              onFixMany={fixMany}
              onDubMany={dubMany}
              onRender={() => void actions.render()}
            />

            {/*
              The viewer column. Fixed width and never the pane that grows —
              past 640px a talking-head preview is not telling the reviewer
              anything the 480px one did not.
            */}
            <aside className="flex min-h-0 max-h-[45%] shrink-0 flex-col bg-surface xl:max-h-none xl:w-[42%] xl:min-w-[26rem] xl:max-w-[40rem]">
              <VideoPlayer
                src={mediaUrl}
                srcLabel={previewPath ?? SOURCE_AUDIO}
                mode={transportMode}
                transport={transport}
                duration={total}
                /*
                  The picture gets a picture's worth of room. Anything else gets
                  a band.

                  There is no video until `mix` has run, which is most of the
                  time a run exists and the empty stage kept the video's own
                  16:9, so on a 1512px window a status board four sentences long
                  sat in 270 vertical pixels of nothing while the selection panel
                  under it scrolled a thirty-control inspector through the
                  remainder. The placeholder is laid out for a band and the
                  column stops reserving a stage for a file that is not there;
                  a stopped run grows it back, because its failure text and its
                  Resume button are worth the pixels the idle case is not.

                  The test is a preview that is actually *attached*, not one the
                  manifest names: a run whose preview.mp4 cannot be reached shows
                  the same status board as a run that has not mixed yet, and it
                  has earned the same band rather than the same empty 16:9.
                */
                className={cn(
                  "shrink-0",
                  transportMode === "preview" && previewUrl
                    ? "h-40 xl:h-auto xl:aspect-video"
                    : "h-auto",
                )}
                placeholder={
                  <PreviewPlaceholder
                    project={project}
                    job={job}
                    stage={state.stage}
                    mode={transportMode}
                    onResume={() => void actions.resume()}
                    // Two separate questions. The dots are hollow whenever the
                    // render is behind; the *sentence* appears exactly once —
                    // the strip below carries it whenever there is a preview to
                    // be behind, and the placeholder covers the case where
                    // there is no video to put a strip under.
                    stale={stale && render.at != null}
                    staleBand={transportMode === "preview" ? null : staleBand}
                  />
                }
              />
              {/*
                One quiet line under the video, where the thing it is about is.
                Not a warning colour and not dismissable: it is a fact about the
                file being played, and it stops being true the moment a render
                makes it stop being true.
              */}
              {staleBand && transportMode === "preview" ? (
                <p
                  data-stale-band
                  className="shrink-0 border-b border-border bg-sunken px-3 py-1.5 text-[11px] text-secondary"
                >
                  {staleBand}
                </p>
              ) : null}
              {selected ? (
                <SelectionPanel
                  seg={selected}
                  segments={segments}
                  speakers={Object.keys(project?.speakers ?? {})}
                  busy={state.pendingUids.includes(selected.uid)}
                  playhead={transport.currentTime}
                  onPatch={(patch) => void actions.patch(selected.uid, patch)}
                  onVerdict={(keep) => void setVerdict(selected, keep)}
                  onSplit={(at) => void actions.split(selected.uid, at)}
                  onMerge={(uidB) => void actions.merge(selected.uid, uidB)}
                  onRemove={() => void actions.remove(selected.uid)}
                  onRetranslate={() => void actions.retranslate([selected.uid])}
                  onResynthesize={() => void actions.resynthesize([selected.uid])}
                />
              ) : (
                <RunSummary
                  segments={segments}
                  project={project}
                  counts={counts}
                  total={total}
                  claimGap={claimGap}
                  onSeek={transport.seek}
                  onAdd={actions.add}
                  onHighlightGap={setHighlightGap}
                />
              )}
            </aside>
          </div>

          <Timeline
            segments={segments}
            total={total}
            currentTime={transport.currentTime}
            selectedUid={selectedUid}
            busyUids={state.pendingUids}
            pxPerSecond={zoom}
            fitPxPerSecond={fitZoom}
            matchedUids={matchedUids}
            highlightGap={highlightGap}
            sourcePeaks={sourcePeaks}
            dubPeaks={dubPeaks}
            stale={stale && render.at != null}
            splitAt={splitAt}
            onSelect={selectFromTimeline}
            onSeek={transport.seek}
            // The strip's drag is the panel's Move button by another gesture:
            // the same PATCH, optimistic with the same rollback (`useProject`).
            onRetime={(uid, start, end) => void actions.patch(uid, { start, end })}
            onClaimGap={claimFromTimeline}
            onViewport={onViewport}
            onZoomIn={zoomIn}
            onZoomOut={zoomOut}
            onZoomBy={zoomBy}
            onFit={zoomFit}
            onSplit={(at) => selected && void actions.split(selected.uid, at)}
          />
        </>
      )}

      {/* Rendered from the LIVE segment: a structural edit renumbers every id
          (so the captured number can name the wrong line) and can retire the
          uid outright (so Undo would PATCH a segment that no longer exists).
          No live segment, no toast. */}
      {undoKept ? (
        <UndoKeepStrip undo={{ ...undoKeep!, id: undoKept.id }} onUndo={undoLastKeep} />
      ) : null}

      {/* The ⌘Z receipt — same strip grammar as the keep toast, hoisted above
          it when both are up so neither covers the other. Its own live region:
          "Nothing to undo" said only by a row that did not change is silence
          to a screen reader, and to everyone else. */}
      {notice ? (
        <div
          role="status"
          data-history-notice
          className={cn(
            "fixed left-1/2 z-50 -translate-x-1/2",
            undoKept ? "bottom-[11.25rem]" : "bottom-[8.5rem]",
            "flex items-center gap-2 rounded-full border border-border bg-raised px-3 py-1.5",
            "text-[12.5px] text-primary shadow-pop",
          )}
        >
          {notice.text}
        </div>
      ) : null}
    </div>
  );
}
