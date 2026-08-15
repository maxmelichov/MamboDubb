/**
 * The editor.
 *
 * Three regions, and the proportions are the argument:
 *
 *   header 44px  — where you are, and the one expensive button
 *   main row     — SCRIPT (58%, and it grows) | VIEWER (42%, fixed column)
 *   timeline     — 112px, full width, at the very bottom
 *
 * The previous shape had four stacked regions with the *video* as the pane that
 * grew and the text as a fifteen-rem drawer under the timeline. That is the
 * layout of a tool for cutting picture. This is a tool for checking a
 * translation against its original, and the artefact being checked is text — so
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
 * are the point — a translation line and a panel heading. The only exception is
 * the 9px on the timeline's axis ticks, which are chart furniture and were
 * already there. A dense tool drifts into eight sizes one component at a time;
 * three is what makes the hierarchy readable rather than merely present.
 *
 * ## One keyboard listener
 *
 * There used to be two — this page's, and a second inside ABPlayer whose guard
 * only checked `INPUT|TEXTAREA|SELECT`, so typing "a" or "b" into any
 * contentEditable started playing audio at the user. There is exactly one now,
 * with exactly one guard, and every binding goes through it.
 *
 * The whole screen stays usable while a job runs. Nothing here disables itself
 * on a running job except the two model actions, which queue.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import {
  ExternalLink,
  Film,
  FolderOpen,
  HelpCircle,
  Loader2,
  MoreHorizontal,
  RotateCcw,
  TriangleAlert,
} from "lucide-react";
import { AppHeader } from "../components/AppShell";
import { JobBar } from "../components/JobBar";
import { ScriptPane, filterSegments, type ScriptFilter } from "../components/ScriptPane";
import { SelectionPanel } from "../components/SelectionPanel";
import { StageTrack } from "../components/StageTrack";
import { Timeline } from "../components/Timeline";
import { VideoPlayer, type TransportMode } from "../components/VideoPlayer";
import type { EditTarget } from "../components/ScriptRow";
import {
  Badge,
  Button,
  ConfirmButton,
  Empty,
  ErrorBar,
  Eyebrow,
  Kbd,
  Popover,
  Select,
  StateIcon,
  TextArea,
} from "../components/ui";
import { api } from "../lib/api";
import { cn } from "../lib/classNames";
import { stopClip, toggleClip, useClipPlayback } from "../lib/clipAudio";
import { isDesktop, revealRunFile } from "../lib/desktop";
import { FIXTURE_PROJECT } from "../lib/fixtures";
import { languageName, lines, timecode } from "../lib/format";
import {
  STATE_META,
  hasTranscript,
  keepReason,
  keptAsTargetLanguage,
  needsModelWork,
  segmentState,
  totalDuration,
  type SegmentState,
} from "../lib/segments";
import { summarizeStages } from "../lib/stages";
import { bucketsFor, usePeaks } from "../lib/usePeaks";
import { activeJob, useProject } from "../lib/useProject";
import { useTransport } from "../lib/useTransport";
import type { Job, ProjectDetail, ProjectOptionsPatch, Segment } from "../lib/types";
import type { StageProgress } from "../lib/useProject";

const ZOOM_STEPS = [0.5, 1, 2, 4, 8, 16, 32];

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

/**
 * The whole of what a flip to keep destroys, held for {@link UNDO_MS}.
 *
 * A fresh object every time, so a second keep re-arms the timer rather than
 * inheriting whatever was left of the first one's.
 */
type KeptUndo = { uid: string; id: number; text_en: string | null };

/**
 * Every shortcut the editor binds, in the order they are learned.
 *
 * `a` and `b` name the buttons they press, which now say Orig and Dub on their
 * faces. The keys did not change — a comparison tool's A and B are worth keeping
 * as *keys* — but a help list that says "play the original" next to a button
 * labelled "Orig" is one vocabulary, and it used to be two.
 */
const SHORTCUTS: [string[], string][] = [
  [["space"], "play / pause"],
  [["↑", "↓"], "previous / next line"],
  [["↵"], "edit the selected translation"],
  [["esc"], "leave the field without saving"],
  [["a"], "play Orig — the original for this line"],
  [["b"], "play Dub — the dubbed clip for this line"],
  [["k"], "switch between dub and keep"],
  [["s"], "split the selection at the playhead"],
  [["+", "−"], "zoom the timeline"],
  [["⌘", "f"], "search the script"],
];

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
   * finishes — a re-voice or a render is exactly when the mix on disk stops
   * matching what is drawn — while the SOURCE lane never does, because
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
   * What the transport plays, and why.
   *
   * `preview.mp4` is written by `mix`, at the very end of the pipeline — but
   * `source.wav` is written by `fetch`, at the very start, and for the whole
   * hour in between the transport had a live play button attached to nothing:
   * "i can start playing. but because it still not ready it shown nothing". So
   * the original audio stands in until the preview exists, and when it does not
   * exist either the play button is honestly dead.
   *
   * Both URLs are composed through the api seam rather than as `/media/…`
   * strings: the desktop shell's origin is the Tauri asset protocol, so a
   * root-relative path handed to a media element there resolves against the
   * wrong origin and plays nothing — which is exactly the bug that killed every
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

  /**
   * Nothing to play means nothing playing.
   *
   * The mode changes under a running transport more often than it looks:
   * `/editor/a` → `/editor/b` is the same route, so this component is not
   * remounted and the clock carries across. A playhead sweeping a run that has
   * no audio at all is the same lie the live play button was telling.
   */
  const pause = transport.pause;
  useEffect(() => {
    if (transportMode === "none") pause();
  }, [pause, transportMode]);

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
   * Selecting a segment also moves the playhead to it — they are one gesture.
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
   * script row — its two texts and its A/B buttons — so the click has to put
   * that row in front of the reviewer, in the middle of the list, rather than
   * leave them to find a row seven hundred pixels up.
   *
   * It is a counter and not just a uid because clicking the same mark twice
   * must scroll twice: the reviewer has scrolled away in between, which is
   * exactly why they clicked it again.
   */
  const [reveal, setReveal] = useState<{ uid: string; n: number } | null>(null);
  const selectFromTimeline = useCallback(
    (uid: string) => {
      selectAndSeek(uid);
      setReveal((current) => ({ uid, n: (current?.n ?? 0) + 1 }));
    },
    [selectAndSeek],
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

  const zoomOut = useCallback(
    () => setZoom((z) => ZOOM_STEPS[Math.max(0, ZOOM_STEPS.indexOf(z) - 1)] ?? z),
    [],
  );
  const zoomIn = useCallback(
    () => setZoom((z) => ZOOM_STEPS[Math.min(ZOOM_STEPS.length - 1, ZOOM_STEPS.indexOf(z) + 1)] ?? z),
    [],
  );

  /**
   * Make these lines playable: translate the ones with nothing to say, then
   * voice all of them.
   *
   * Two jobs at most, never 2N — the one-job queue runs them FIFO, so the
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
       * run on lines whose translation had just been abandoned — 27 lines
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
   * Almost everything this screen says, it says by *changing* — a row's badge
   * flips, the strip appears, the marks get wider — and a change with no text is
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
   * announcement twice in a row impossible — the region only speaks when its
   * contents *change*, so a second "Kept #17" after a first would be silent.
   * Going back to empty after a beat fixes both.
   */
  useEffect(() => {
    if (!spoken) return;
    const timer = window.setTimeout(() => setSpoken(""), ANNOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [spoken]);

  /* Skips the mount, which is not a change — otherwise every run announces its
     own default zoom on open. */
  const firstZoom = useRef(true);
  useEffect(() => {
    if (firstZoom.current) {
      firstZoom.current = false;
      return;
    }
    say(`Timeline zoom ${zoom >= 10 ? Math.round(zoom) : Math.round(zoom * 10) / 10} pixels per second`);
  }, [say, zoom]);

  /** The undo behind the one destructive verdict — see `setVerdict` below. */
  const [undoKeep, setUndoKeep] = useState<KeptUndo | null>(null);
  useEffect(() => {
    if (!undoKeep) return;
    const timer = window.setTimeout(() => setUndoKeep(null), UNDO_MS);
    return () => window.clearTimeout(timer);
  }, [undoKeep]);

  /**
   * The verdict, and everything it implies.
   *
   * `PATCH {keep:false}` on its own is what the bug was: `edit.set_keep`
   * invalidates the translate stage in both directions, so flipping a kept line
   * to "Dub it" leaves it with no subtitle and no clip — and nothing queued to
   * make either. The line then sits in the run forever, dubbed in name only.
   * So the flip *is* the PATCH plus the work it creates, in that order, and the
   * server's answer decides which work: a hand-written translation is locked
   * and survives the flip, and re-translating over it would discard the very
   * thing the user typed.
   *
   * The other direction queues nothing — the original audio is already on disk
   * — and cancels nothing either: a running job belongs to whatever asked for
   * it, not to this line.
   *
   * `queue` is how a caller flipping *many* lines borrows this without
   * borrowing 2N jobs: it takes the PATCH, the lock and the optimistic predict
   * exactly as they are, and queues the work itself, once, over the lot. The
   * saved segment comes back for that reason — the server's answer is what says
   * whether the translation survived the flip.
   *
   * The *other* other direction is the one nothing guarded. `set_keep`
   * invalidates translate both ways, so flipping a line TO keep throws its
   * translation away — and that flip is a button, a menu item, and `k`, a single
   * keystroke with no modifier. A confirm on it would be the wrong control:
   * judging a run is a hundred of these decisions and a dialog on each is a tool
   * nobody uses. So the flip stays instant and leaves an undo behind, holding
   * the *whole* of what was lost — the server drops the line a moment later and
   * nothing else on the client remembers it.
   */
  const setVerdict = useCallback(
    async (seg: Segment, keep: boolean, queue = true): Promise<Segment | null> => {
      if (seg.keep === keep) return null;
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
   * `set_keep` and `invalidate` honours locks — so the restored line is written,
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
   * The case is a run the keep rule got wrong — a video of a third language
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
         * The gesture only exists on the Kept filter, and it empties it — every
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
   * A committed edit is a PATCH — unless nothing was edited.
   *
   * Every way out of the field goes through here (blur, ⌘↵, the ⋯ menu's
   * Correct transcript), and closing a field you only opened to *read* must
   * cost nothing: no request, no lock stamped on the line, and above all no
   * invalidation — `edit.set_text` drops the clip and the placement, so a
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
      if (event.metaKey || event.ctrlKey || event.altKey) return;

      switch (event.key) {
        case " ":
          event.preventDefault();
          // Nothing on disk to play: the button is disabled and the key is the
          // same control, so it refuses in the same case rather than starting a
          // clock over silence.
          if (transportMode !== "none") transport.toggle();
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
            setEditing({ uid: selected.uid, field: "text_en" });
          }
          break;
        case "Escape":
          setEditing(null);
          stopClip();
          break;
        case "a":
          if (selected) {
            event.preventDefault();
            // The source side is a window of the whole track — same contract
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
  }, [actions, selected, splitAt, step, toggleKeep, transport, transportMode, zoomIn, zoomOut]);

  const job = activeJob(state.jobs);

  /*
   * What the video on screen is a render of.
   *
   * `render.at === null` is a run with no stamp — old, or never mixed — where
   * `changed` is 0 because there is nothing to compare against. That is why the
   * "N lines changed" phrasing is guarded on the count rather than on `stale`:
   * saying "0 lines changed" next to a Render button would be worse than quiet.
   */
  const render = project?.render ?? { at: null, stale: false, changed: 0 };
  const stale = render.stale;
  const changed = render.changed;
  // The button counts *lines*, because that is what it would re-render. These
  // two count *changes*, because they are about what the user did — and "your
  // last 1 line" is not a phrase anybody says.
  const edits = changed === 1 ? "change" : `${changed} changes`;
  const staleNote = stale && changed > 0 ? `rendered before your last ${edits}` : null;
  const staleBand =
    stale && changed > 0
      ? `Mixed before your last ${edits} — Update the video to hear them`
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
   * sentence — "typically a few minutes" over a job that loads two models — so
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
            {/* The finished file, one click from the run it came out of. It
                only exists when the file does. */}
            {project?.outputs.preview ? (
              <OpenFileButton
                name={name}
                path={project.outputs.preview}
                title={staleNote ?? undefined}
              />
            ) : null}
            {/*
              Quiet when the video is current, loud when it is not.
              The button was permanently accented, which made "there is work
              waiting" indistinguishable from "everything is up to date" — the
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
                  : "Re-render the preview. This re-runs timeline, mix and report and replaces preview.mp4 — a full video re-encode, typically a few minutes."
              }
              onConfirm={() => void actions.render()}
            >
              <Film className="h-3.5 w-3.5" />
              {stale && changed > 0
                ? `Update the video · ${lines(changed)} changed`
                : stale && render.at
                  ? "Update the video"
                  : "Render preview"}
            </ConfirmButton>
            <RunMenu
              project={project}
              name={name}
              counts={counts}
              onSeek={transport.seek}
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
        anything a sighted user is not also being shown — see `say`. It sits
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
              onEdit={setEditing}
              onCommit={commit}
              onPlay={toggleClip}
              onToggleKeep={toggleKeep}
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
                src={previewUrl ?? sourceUrl}
                mode={transportMode}
                transport={transport}
                duration={total}
                className="h-40 shrink-0 xl:h-auto xl:aspect-video"
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
                  onRetranslate={() => void actions.retranslate([selected.uid])}
                  onResynthesize={() => void actions.resynthesize([selected.uid])}
                />
              ) : (
                <RunSummary
                  segments={segments}
                  project={project}
                  counts={counts}
                  onSeek={transport.seek}
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
            sourcePeaks={sourcePeaks}
            dubPeaks={dubPeaks}
            stale={stale && render.at != null}
            splitAt={splitAt}
            onSelect={selectFromTimeline}
            onSeek={transport.seek}
            onZoomIn={zoomIn}
            onZoomOut={zoomOut}
            onSplit={(at) => selected && void actions.split(selected.uid, at)}
          />
        </>
      )}

      {undoKeep ? <UndoKeepStrip undo={undoKeep} onUndo={undoLastKeep} /> : null}
    </div>
  );
}

/**
 * "Kept #17 — Undo."
 *
 * The whole guard on the one destructive verdict, and deliberately not a dialog:
 * the flip has to stay a keystroke, because checking a run is a hundred of them.
 * A strip that costs nothing to ignore and one click to reverse is the trade a
 * confirm cannot make.
 *
 * The title says the part the strip has no room for and the user has no way to
 * guess: the restored translation comes back *locked*, because restoring it is
 * writing it, and writing a line is what makes it the user's. That is a better
 * outcome than losing it and it is not the same state as before, so it is said
 * rather than glossed.
 */
function UndoKeepStrip({ undo, onUndo }: { undo: KeptUndo; onUndo: () => void }) {
  return (
    <div
      role="status"
      data-undo-toast
      title={
        undo.text_en
          ? "Undo switches this line back to Dub and puts its translation back. A restored " +
            "line counts as hand-written from then on, so a re-run will not replace it."
          : "Undo switches this line back to Dub and queues the work it needs."
      }
      className={cn(
        // Above the 7rem timeline strip, centred, out of the script's way.
        "fixed bottom-[8.5rem] left-1/2 z-50 -translate-x-1/2",
        "flex items-center gap-2 rounded-full border border-border bg-raised px-3 py-1.5",
        "text-[12.5px] text-primary shadow-pop",
      )}
    >
      <StateIcon state="kept" className="h-2.5 w-2.5" />
      <span className="font-mono tabular-nums">Kept #{undo.id}</span>
      <span aria-hidden className="text-muted">
        —
      </span>
      <Button size="xs" variant="ghost" data-undo-keep onClick={onUndo}>
        Undo
      </Button>
    </div>
  );
}

/**
 * Open a file the run produced.
 *
 * Two environments, one intent. In the desktop shell the useful thing is the
 * file itself — the user wants to play it in QuickTime, drop it into a chat,
 * put it somewhere — so it is revealed in Finder. In a browser tab there is no
 * Finder and the server is already serving the run directory, so the URL opens
 * in a new tab. The shell falls back to the tab if the reveal could not
 * happen, which covers an older shell with no `workspace` in its handshake.
 *
 * It is a function rather than a hook because both callers — the header button
 * and the run menu's file list — want the same three lines and neither wants
 * to think about which environment it is in.
 */
async function openRunFile(name: string, relPath: string): Promise<void> {
  if (await revealRunFile(name, relPath)) return;
  const href = api.mediaUrl(name, relPath);
  if (href) window.open(href, "_blank", "noopener");
}

/**
 * "Where is the file?" as a button.
 *
 * The single most-asked question at the end of a run, and until now the answer
 * was a Finder window and a memory of the run directory's name. Header-sized
 * and labelled for the environment it is in — "Show in Finder" is a promise a
 * browser tab cannot keep, and "Open" is a weaker one than the shell can.
 */
function OpenFileButton(
  { name, path, title }: { name: string; path: string; title?: string },
) {
  const desktop = isDesktop();
  return (
    <Button
      size="sm"
      // When the file is behind the edits, say so on the button that opens it —
      // this is the last moment before the user watches five minutes of a video
      // that does not contain the correction they just made.
      title={
        title
          ? `${desktop ? `Show ${path} in Finder` : `Open ${path} in a new tab`} — ${title}`
          : desktop
            ? `Show ${path} in Finder`
            : `Open ${path} in a new tab`
      }
      onClick={() => void openRunFile(name, path)}
    >
      {desktop ? (
        <FolderOpen className="h-3.5 w-3.5" aria-hidden />
      ) : (
        <ExternalLink className="h-3.5 w-3.5" aria-hidden />
      )}
      {desktop ? "Show in Finder" : "Open preview"}
    </Button>
  );
}

/**
 * What the rail says when nothing is selected.
 *
 * It used to say "No line selected", centred, in a column six hundred pixels
 * tall — an empty state that filled two fifths of the screen with an
 * instruction the user had already worked out. The space is worth more than
 * that: nothing is selected exactly when a run has just been opened, which is
 * the moment "what happened here" is the only question there is.
 *
 * So it answers that question and no other: how the lines came out, why the
 * kept ones were kept, and where there is speech nothing covers. All of it is
 * derived from segments already on the client — no request, no report file —
 * and it is set as prose and a tally rather than as a dashboard, because it is
 * read once per run and then replaced by the panel for a line.
 */
function RunSummary({
  segments,
  project,
  counts,
  onSeek,
}: {
  segments: Segment[];
  project: ProjectDetail | null;
  counts: Record<SegmentState, number>;
  onSeek: (time: number) => void;
}) {
  const total = segments.length;
  const gaps = project?.report?.uncovered_audible ?? [];

  /*
   * Why the kept lines were kept, most common first.
   *
   * The reasons arrive as the pipeline's own tokens (`latin`, `speaker_en`,
   * `user`, `tts_failed`, …) and were printed raw, which put "Kept because
   * user" on the screen of the user who pressed the button. `keepReason` is the
   * one mapping, shared with the selection panel; a token it has never seen
   * still falls through to itself rather than being given an invented meaning.
   */
  const reasons = useMemo(() => {
    const tally = new Map<string, number>();
    for (const seg of segments) {
      if (!seg.keep) continue;
      const reason = keepReason(seg.keep_reason);
      tally.set(reason, (tally.get(reason) ?? 0) + 1);
    }
    return [...tally.entries()].sort((a, b) => b[1] - a[1]);
  }, [segments]);

  // "Already in the target language" is a claim about *why*, so it is only made
  // when the reasons say so — a run kept by hand or by a failure is a different
  // story with the same count.
  const passthrough = segments.filter(
    (seg) => seg.keep && keptAsTargetLanguage(seg.keep_reason),
  ).length;
  const mostlyKept = total > 0 && counts.kept / total >= 0.6;

  const lead =
    total === 0
      ? "No lines yet — the segments stage is what fills this list."
      : mostlyKept && passthrough >= counts.kept / 2
        ? `This video mostly speaks ${languageName(project?.source.tgt_lang)} already — ` +
          `${counts.kept} of ${total} lines keep their original audio.`
        : mostlyKept
          ? `${counts.kept} of ${total} lines keep their original audio.`
          : `${counts.dubbed} of ${total} lines are dubbed.`;

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3.5 overflow-y-auto px-3 py-3">
      <div>
        <Eyebrow>This run</Eyebrow>
        <p className="mt-1.5 text-[14px] leading-snug text-primary">{lead}</p>
      </div>

      <StateTally counts={counts} />

      {reasons.length > 0 ? (
        <div>
          <Eyebrow className="mb-1.5">Kept because</Eyebrow>
          <p className="text-[12.5px] leading-relaxed text-muted">
            {reasons.map(([reason, n], i) => (
              <span key={reason}>
                {i > 0 ? " · " : null}
                <span className="text-secondary">{reason}</span> {n}
              </span>
            ))}
          </p>
        </div>
      ) : null}

      <GapList gaps={gaps} onSeek={onSeek} stale={project?.report?.stale} />

      <p className="mt-auto pt-2 text-[11px] leading-relaxed text-muted">
        Pick a line in the script — everything that is true about it, and cannot fit on a row,
        is here.
      </p>
    </div>
  );
}

/** The five states and their counts, as a tally with leader lines. */
function StateTally({ counts }: { counts: Record<SegmentState, number> }) {
  const states = (Object.keys(STATE_META) as SegmentState[]).filter((s) => counts[s] > 0);
  return (
    <dl className="flex flex-col gap-1 text-[12.5px]">
      {states.map((s) => (
        <div key={s} className="flex items-baseline gap-2">
          <dt className="flex items-center gap-1.5 text-secondary">
            <StateIcon state={s} className="h-2.5 w-2.5" />
            {STATE_META[s].label}
          </dt>
          <span className="h-px flex-1 bg-border" aria-hidden />
          <dd className="font-mono tabular-nums text-primary">{counts[s]}</dd>
        </div>
      ))}
    </dl>
  );
}

/**
 * Speech in the source that no segment claims, so the dub plays the original
 * over it. The highest-value readout `report.json` produces, and every row of
 * it is a button that seeks there — a list of timecodes you cannot jump to is
 * a list of homework.
 */
function GapList({
  gaps,
  onSeek,
  stale = false,
  className,
}: {
  gaps: { start: number; end: number; duration: number }[];
  onSeek: (time: number) => void;
  /** These spans were found by the last report, not by the script on screen. */
  stale?: boolean;
  className?: string;
}) {
  if (gaps.length === 0) return null;
  return (
    <div className={className}>
      {/*
        A gap list is a finding, and a finding has a date. Splitting a segment
        closes a gap and re-segmenting opens new ones, and neither shows here
        until a render re-runs the report — so when the report is behind, the
        eyebrow says which report these are from rather than implying "now".
      */}
      <Eyebrow className="mb-1.5">
        Audible, uncovered — {gaps.length}
        {stale ? " · from the last render" : ""}
      </Eyebrow>
      <p className="mb-1.5 text-[11px] leading-relaxed text-muted">
        Speech here is in the source but no segment claims it, so the dub plays the original.
        Jump to one and listen.
      </p>
      <ul className="flex max-h-48 flex-col gap-1 overflow-y-auto">
        {gaps.map((gap) => (
          <li key={`${gap.start}-${gap.end}`}>
            <button
              type="button"
              onClick={() => onSeek(gap.start)}
              className="flex w-full items-center gap-2 rounded-lg border border-border bg-raised px-2 py-1 text-left text-[12.5px] transition-colors hover:border-axis hover:bg-sunken"
            >
              <TriangleAlert className="h-3 w-3 shrink-0 text-muted" aria-hidden />
              <span className="font-mono tabular-nums text-primary">{timecode(gap.start)}</span>
              <span className="ml-auto text-muted">{gap.duration.toFixed(1)}s</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

/**
 * The run, behind one button.
 *
 * Three things that are true about a run and are not per-line: how many
 * segments are in what state, where there is audible speech that nothing
 * covers, and where the files are. None of them changes while you work through
 * a line, and none of them earns permanent chrome — but the middle one is the
 * highest-value readout the report produces, so every gap is a button that
 * seeks to it.
 *
 * It shares its tally and its gap list with the rail's own summary above —
 * same three answers, one for the reviewer who has a line open and one for the
 * reviewer who does not. What neither of them brought back from the 194-line
 * original are the coverage bars and the drift and speed stats: those were a
 * report rendered twice, once here and once in `report.json`.
 */
function RunMenu({
  project,
  name,
  counts,
  onSeek,
  onSaveOptions,
}: {
  project: ProjectDetail | null;
  name: string;
  counts: Record<SegmentState, number>;
  onSeek: (time: number) => void;
  onSaveOptions: (patch: ProjectOptionsPatch) => Promise<void>;
}) {
  const gaps = project?.report?.uncovered_audible ?? [];
  const preview = project?.outputs.preview;
  const srt = project?.outputs.srt;

  return (
    <Popover
      label="Run health and files"
      title="Run health"
      trigger={<MoreHorizontal className="h-3.5 w-3.5" />}
      className="w-[21rem]"
    >
      <StateTally counts={counts} />

      <GapList gaps={gaps} onSeek={onSeek} stale={project?.report?.stale} className="mt-3.5" />

      <Eyebrow className="mt-3.5 mb-1.5">This run</Eyebrow>
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
        that handed `reveal_path` the manifest's *run-relative* path — which
        the shell resolves against its own working directory and refuses,
        because nothing is there. Both rows go through `openRunFile`, which
        composes the absolute path in the shell and opens the served URL in a
        browser, so the list is useful in both and the subtitles — the other
        thing a finished run is for — are no longer unreachable.
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
 * became unreachable — a `--genre documentary` picked in ten seconds before the
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
   * The server 409s while a job runs — run options are read when a job starts,
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

      {/* Context is prose, so it gets a field and an explicit commit — a
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
            {context || <span className="text-muted">none — names and spellings go here</span>}
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

/**
 * The keyboard, behind one button.
 *
 * It used to share this popover with a colour legend. The legend is gone: every
 * script row now carries its state as a word, which is the same information in
 * the place it is needed, and a key to an encoding that is already spelled out
 * is a key to nothing.
 */
function ShortcutHelp() {
  return (
    <Popover
      label="Keyboard shortcuts"
      title="Keyboard"
      trigger={<HelpCircle className="h-3.5 w-3.5" />}
      className="w-[19rem]"
    >
      <dl className="flex flex-col gap-1.5 text-[11px]">
        {SHORTCUTS.map(([keys, does]) => (
          <div key={does} className="flex items-baseline gap-2">
            <dt className="flex shrink-0 items-center gap-1">
              {keys.map((key) => (
                <Kbd key={key}>{key}</Kbd>
              ))}
            </dt>
            <dd className="min-w-0 flex-1 text-right text-muted">{does}</dd>
          </div>
        ))}
      </dl>
    </Popover>
  );
}

/**
 * The most recent failure that is still the answer about this run.
 *
 * Same rule the server applies to stages (`Projects.stage_status`): a job that
 * succeeded *after* a failure is the newer answer, so the failure is cleared
 * rather than kept as a permanent scar. Jobs arrive oldest first.
 */
function lastFailure(jobs: Job[]): Job | null {
  let dead: Job | null = null;
  for (const job of jobs) {
    if (job.status === "done") dead = null;
    else if (job.status === "failed") dead = job;
  }
  return dead;
}

/**
 * The preview stage when there is no `preview.mp4` to show.
 *
 * This is the screen a user stares at for the entire length of a run, so it has
 * to be the run's status board rather than an apology. It no longer carries its
 * own Render button or its own progress bar: the header has the first and the
 * job strip has the second, both of them three inches away and both of them
 * permanent. What it keeps that nothing else has is the stage track — which of
 * the nine stages this run has got through.
 *
 * And, since a run could stop, the way back in.
 *
 * A failed or abandoned run was a dead end: the app could create a project and
 * edit one, and nothing anywhere in it could start the pipeline again. The
 * screen said "The run stopped at fetch" and left the user with a script pane of
 * nothing, a dead play button and a Render button that would produce another
 * failure twenty minutes later. Both halves of the fix are here because both
 * halves are the same question — *what happened, and what do I press*:
 *
 * * **What happened** is the failed job's own error. It is only on the project
 *   (`GET /api/projects/{name}` → `jobs`); the event stream deliberately replays
 *   nothing terminal, so after a reload this is the only copy of it.
 * * **What to press** is the same `run` job that created the project. Every
 *   stage is skipped when its inputs are unchanged, so re-running *is* resuming
 *   — which is why the button is honest about naming the stage it will retry.
 */
function PreviewPlaceholder({
  project,
  job,
  stage,
  mode,
  onResume,
  stale = false,
  staleBand = null,
}: {
  project: ProjectDetail | null;
  job: Job | null;
  stage: StageProgress | null;
  /** What the transport is on, so the panel does not promise a clock that is not one. */
  mode: TransportMode;
  onResume: () => void;
  /** The render is behind the script: the last three stage dots go hollow. */
  stale?: boolean;
  /** The sentence, when this panel is the one carrying it — else null. */
  staleBand?: string | null;
}) {
  const summary = summarizeStages(project?.stages);
  const working = job != null && (job.status === "running" || job.status === "queued");
  const dead = lastFailure(project?.jobs ?? []);
  /*
   * A run that is not finished and has nothing in flight is stopped, whether or
   * not a stage says "failed". A job cancelled mid-fetch, a server restarted
   * under a run, a laptop that slept — none of them leave a failed stage behind,
   * and all of them leave a project that will never move again on its own.
   */
  const stopped = project != null && !working && !summary.complete;

  return (
    <div className="grid h-full place-items-center px-4 py-3">
      <div className="w-full max-w-sm text-center">
        <Eyebrow>
          {working
            ? "Working"
            : summary.failed || dead
              ? "Stopped"
              : summary.complete
                ? "No preview file"
                : "Preview"}
        </Eyebrow>

        <p className="mt-1.5 text-[14px] font-semibold text-primary">
          {working
            ? `Running ${stage?.stage ?? summary.current ?? "the pipeline"}`
            : summary.failed
              ? `The run stopped at ${summary.failed}`
              : summary.complete
                ? "The run finished, but there is no video"
                : summary.started
                  ? `Stopped after ${summary.done} of ${summary.total} stages`
                  : "Nothing has run yet"}
        </p>

        {/*
          The pipeline's own last words, verbatim.
          Never paraphrased and never truncated to a category: "the run stopped
          at fetch" is the same sentence for a dead URL, a private video and a
          full disk, and only one of the three is worth retrying unchanged.
        */}
        {!working && dead?.error ? (
          <p
            data-failure
            className="mt-2.5 max-h-24 overflow-auto rounded-lg border border-critical/35 bg-critical/[0.06] px-2.5 py-2 text-left font-mono text-[11px] leading-relaxed break-words text-secondary"
          >
            {dead.error}
          </p>
        ) : null}

        <p className="mt-2 text-[11px] leading-relaxed text-muted">
          {working
            ? (stage?.message ?? "preview.mp4 is written by the mix stage, at the end.")
            : summary.complete
              ? "Render preview re-runs timeline, mix and report and writes preview.mp4."
              : mode === "source"
                ? "preview.mp4 is written by the mix stage, at the end. Until then press play " +
                  "and the transport plays the run's original audio, in step with the script."
                : "preview.mp4 is written by the mix stage. There is nothing to play until " +
                  "fetch has written source.wav, but the timeline and the script still work."}
        </p>

        {/*
          The way back in.
          The rationale sits under the button rather than above it because it is
          the answer to the question the button raises — "will this start over?"
          — and because a user who already knows presses it without reading.
        */}
        {stopped ? (
          <>
            <Button
              variant="primary"
              size="sm"
              className="mt-3"
              data-resume
              onClick={onResume}
              title={
                summary.failed
                  ? `Run the pipeline again from ${summary.failed}`
                  : "Run the pipeline again; finished stages are skipped"
              }
            >
              <RotateCcw className="h-3.5 w-3.5" aria-hidden />
              {summary.failed ? `Retry from ${summary.failed}` : "Resume the run"}
            </Button>
            <p className="mt-1.5 text-[11px] leading-relaxed text-muted">
              Every stage is skipped when its inputs have not changed, so this picks up where the
              run stopped rather than starting over.
            </p>
          </>
        ) : null}

        <div className="mt-3 flex justify-center">
          <StageTrack
            stages={project?.stages}
            current={stage?.stage ?? null}
            stale={stale}
          />
        </div>

        {/*
          The same sentence the video area carries, for the case where there is
          no video to carry it: a run whose preview was deleted, or one still
          mid-render, is exactly as far behind the script and has nothing else
          on screen that says so.
        */}
        {staleBand && !working ? (
          <p data-stale-band className="mt-2.5 text-[11px] leading-relaxed text-secondary">
            {staleBand}
          </p>
        ) : null}
      </div>
    </div>
  );
}
