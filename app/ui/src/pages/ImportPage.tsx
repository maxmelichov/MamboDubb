/**
 * Import: pick a source, say what language it is and what language it should
 * become, then start a run.
 *
 * The shape is a studio composition, not a form in a column. Two regions:
 *
 * - **New dub** — the card. It holds the two things you *type*: the source, and
 *   the context note. The single primary action sits in a sunken band directly
 *   under them, bottom-right, so it reads as the card's conclusion — and the
 *   card ends there. It used to stretch to match the rail's height, which put
 *   a void under the context field and the Start button below the fold at
 *   1280×800: the screen's one action was the thing you had to scroll for.
 *   Now the grid is `items-start` and the card is exactly as tall as its
 *   content.
 * - **Options** — the rail beside it. Five labelled groups, hairlined apart:
 *   languages, genre, register, transcript, scope. Nothing here is typed prose;
 *   it is all picking, which is why it is a rail and not a second column of
 *   paragraphs. It ends with the sentence that says which of these choices are
 *   final because two of them are, and a screen that does not say so is asking
 *   people to guess whether a run is a commitment.
 *
 * The runs used to be a third *column* here, competing with the form for the
 * width. They are a page of their own now (RunsPage, at /runs) and this form is
 * home at "/" — but the newest few of them close the page (`HomeRuns`, at the
 * foot of this file), below the form *and* the rail. Below, not beside — asked
 * for twice, so it is the layout: the form keeps the whole top of the screen
 * and Start dubbing stays above the fold, while "is yesterday's dub still
 * here?" is answered by scrolling rather than by finding the nav pill.
 *
 * The context note is not decoration. Translation quality moves measurably with
 * a sentence about who and what the video is about and how names are spelled —
 * it is the difference between one consistent spelling of a name and three
 * different manglings. It is *one sentence*, though, which is why the field is
 * two rows tall and grows: the paragraph-sized box it used to be spent the
 * card's whole lower half on white space and pushed everything under it down.
 *
 * Genre and register used to be two more dropdowns. They are two-way choices
 * whose options need a clause to be meaningful ("Documentary narrated,
 * factual"), and a clause does not fit in an `<option>`; the genre is rows that
 * fill with the accent when picked, the register is a pill.
 */

import { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  ArrowRight,
  Captions,
  Clapperboard,
  Film,
  FileVideo,
  FolderOpen,
  Languages,
  Loader2,
  MessagesSquare,
  Mic2,
  PencilLine,
  Timer,
} from "lucide-react";
import { PageShell } from "../components/AppShell";
import { RunRow, orderRuns } from "../components/RunRow";
import {
  Button,
  Card,
  CardSection,
  Checkbox,
  Divider,
  ErrorBlock,
  Field,
  NumberInput,
  OptionList,
  OptionRow,
  SectionLabel,
  Segmented,
  segmentedCell,
  Select,
  TextArea,
  TextInput,
} from "../components/ui";
import { api } from "../lib/api";
import { isDesktop, pickTranscriptFile, pickVideoFile } from "../lib/desktop";
import type { CreateProjectRequest, ProjectSummary } from "../lib/types";

// What can be HEARD is broader than what can be SPOKEN: the ASR + translator
// handle these sources, but the synthesizer voices Qwen3-TTS's ten languages
// plus Hebrew (a LoRA over the same checkpoint; the server refuses a Hebrew
// target with the download command if the adapter isn't installed). Arabic is
// the one language left on this list and off the other offering it as a
// target would create a project whose tts stage can only fail, so the two lists
// stay deliberately different.
const SRC_LANGS = [
  ["he", "Hebrew"],
  ["en", "English"],
  ["ar", "Arabic"],
  ["ru", "Russian"],
  ["fr", "French"],
  ["es", "Spanish"],
  ["de", "German"],
  ["it", "Italian"],
  ["pt", "Portuguese"],
  ["zh", "Chinese"],
  ["ja", "Japanese"],
  ["ko", "Korean"],
] as const;

const TGT_LANGS = [
  ["en", "English"],
  ["he", "Hebrew"],
  ["ru", "Russian"],
  ["fr", "French"],
  ["es", "Spanish"],
  ["de", "German"],
  ["it", "Italian"],
  ["pt", "Portuguese"],
  ["zh", "Chinese"],
  ["ja", "Japanese"],
  ["ko", "Korean"],
] as const;

export function ImportPage() {
  const navigate = useNavigate();
  const fileRef = useRef<HTMLInputElement>(null);
  const transcriptRef = useRef<HTMLInputElement>(null);
  // Computed once: the platform does not change while the page is open, and a
  // button must not swap its behaviour a tick after the user reads it.
  const desktop = isDesktop();

  const [form, setForm] = useState<CreateProjectRequest>({
    source: "",
    src_lang: "he",
    tgt_lang: "en",
    duration: null,
    name: null,
    context: "",
    genre: "documentary",
    register: "narration",
    // The pipeline's own default: captions when the fetch found some, ASR
    // otherwise. Accepted by the server since the first day and unreachable from
    // this screen until now, so a user who knew the auto-captions were garbage
    // had no way to say so.
    transcript: "auto",
    // The transcript the user brought, if they brought one. Only read when
    // `transcript` is "file"; kept across a change of mind so switching away
    // and back does not make them find the file again.
    captions: null,
    // Off, because off is what the pipeline does when nobody says otherwise.
    dub_foreign: false,
  });
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const update = (patch: Partial<CreateProjectRequest>) =>
    setForm((current) => ({ ...current, ...patch }));

  /**
   * In the shell, a native dialog gives back the absolute path the pipeline
   * needs. In a browser the same button opens `<input type=file>`, which can
   * only ever report a name — hence the hint telling the user to paste a path.
   */
  const chooseFile = async () => {
    if (!desktop) {
      fileRef.current?.click();
      return;
    }
    const path = await pickVideoFile();
    if (path) update({ source: path });
  };

  /** The same seam for the transcript file, with the same browser caveat. */
  const chooseTranscript = async () => {
    if (!desktop) {
      transcriptRef.current?.click();
      return;
    }
    const path = await pickTranscriptFile();
    if (path) update({ captions: path });
  };

  const start = async () => {
    if (!form.source.trim()) {
      setError("Give it a video: a URL, or the full path to a local file.");
      return;
    }
    // The second field that can be mandatory, and only ever conditionally: the
    // server refuses this combination too, but a round trip to be told what the
    // screen already knows is a worse way to learn it.
    if (form.transcript === "file" && !form.captions?.trim()) {
      setError("Give it the transcript: the full path to an .srt, .vtt or .json3 file.");
      return;
    }
    setStarting(true);
    setError(null);
    try {
      const created = await api.createProject({
        ...form,
        context: form.context?.trim() || null,
        // Only sent when it is the answer. A path left over from a change of
        // mind would otherwise become the run's captions on an "Automatic" run,
        // which is a setting the user backed out of.
        captions: form.transcript === "file" ? form.captions?.trim() || null : null,
      });
      navigate(`/editor/${encodeURIComponent(created.name)}`);
    } catch (err) {
      setError(String(err instanceof Error ? err.message : err));
      setStarting(false);
    }
  };

  return (
    // No hero: the nav pill says New dub, the card says what it does, and the
    // button says Start dubbing — a display title over all three was the page
    // explaining itself to people already using it.
    <PageShell width="wide">
      {/* Two regions in one grid: at `lg` the card and the rail share the row.
          The rail gets its full 24rem back — the runs column it used to pay
          for is a page of its own now. `items-start` is what keeps the Start
          button above the fold: without it the grid stretches the short card
          to the tall rail's height, and the slack lands as a void between the
          context field and the action band. */}
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_24rem]">
        {/* ------------------------------------------------- the left column */}
        <div className="flex min-w-0 flex-col gap-5">
          <Card
            data-region="new-dub"
            className="flex flex-col overflow-hidden rounded-3xl p-0 shadow-lift"
          >
            <CardSection className="pt-6">
              {/*
                The only field on this screen that must be filled in, and the only
                one that said nothing about it. Every other control has a default,
                so the screen read as "all optional" right up to the moment Start
                dubbing answered with a refusal — a rule learned by breaking it.
              */}
              <div className="flex items-baseline gap-2">
                <SectionLabel icon={FileVideo}>Source</SectionLabel>
                <span
                  data-required
                  className="text-[10px] font-bold uppercase tracking-[0.14em]"
                  style={{ color: "var(--color-critical)" }}
                >
                  * required
                </span>
              </div>
              <div className="mt-3 flex flex-col gap-2 sm:flex-row">
                <TextInput
                  className="h-12 flex-1 rounded-xl px-4 text-[14px]"
                  value={form.source}
                  required
                  aria-required="true"
                  aria-label="Source"
                  placeholder="https://www.youtube.com/watch?v=… or /Users/you/clip.mp4"
                  onChange={(event) => update({ source: event.currentTarget.value })}
                />
                <Button size="lg" className="h-12 rounded-xl" onClick={() => void chooseFile()}>
                  <FolderOpen className="h-4 w-4" />
                  Choose file
                </Button>
                <input
                  ref={fileRef}
                  type="file"
                  accept="video/*,audio/*"
                  className="hidden"
                  onChange={(event) => {
                    const file = event.currentTarget.files?.[0];
                    if (file) update({ source: file.name });
                  }}
                />
              </div>
              <p className="mt-2.5 max-w-2xl text-[12px] leading-relaxed text-muted">
                {desktop ? (
                  <>
                    A URL, or a local file — <em>Choose file</em> opens a real file dialog and
                    fills in the full path.
                  </>
                ) : (
                  <>
                    A URL, or an absolute path to a local file. The browser cannot read a file's real
                    path, so <em>Choose file</em> only fills in the name — paste the full path, or
                    use the desktop app.
                  </>
                )}
              </p>
            </CardSection>

            <Divider />

            {/* A bordered field like every other input on the page. It was
                borderless when it filled the whole card, but at three rows an
                edge is what says "type here" — without one it read as a caption
                under the section label. */}
            <CardSection className="pb-6">
              {/*
                The label leads with the word that answers the question the field
                raises. Source is marked required in red beside its own label, so
                a second unmarked field of the same size directly under it reads as
                the second half of one form and a user who does not know what to
                write in it stops there. "Optional" first, because that is the
                part that unblocks them; the sentence under the field still says
                it is the cheapest thing on the screen.
              */}
              <SectionLabel icon={PencilLine}>Optional Context</SectionLabel>
              {/* Two rows, not three-and-a-bit. It is one sentence people type,
                  and the box was sized for the paragraph nobody writes — half
                  the card's lower half spent on empty field, with the runs
                  below it pushed that much further down. `autoGrow` means the
                  rare long note still shows all of itself; the small box is the
                  resting state, not a cap. */}
              <TextArea
                autoGrow
                rows={2}
                className="mt-2 min-h-12 resize-none rounded-xl text-[13.5px]"
                aria-label="Context"
                value={form.context ?? ""}
                placeholder="Who and what this is about, and how names are spelled — e.g. a news interview about the housing market; the host is Dana (she), the guest is Prof. Ronen Levi (he)."
                onChange={(event) => update({ context: event.currentTarget.value })}
              />
              <p className="mt-3 text-[11.5px] leading-relaxed text-muted">
                The single cheapest thing on this screen: a sentence of context materially
                improves the translation.
              </p>
            </CardSection>

            <CardSection
              tone="sunken"
              className="flex flex-col gap-4 border-t border-border sm:flex-row sm:items-center sm:justify-between"
            >
              <p className="max-w-sm text-[12px] leading-relaxed text-muted">
                One job runs at a time; the editor opens straight away with live progress.
              </p>
              <Button
                variant="primary"
                size="lg"
                className="rounded-xl px-6"
                onClick={start}
                disabled={starting}
              >
                {starting ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Starting…
                  </>
                ) : (
                  <>
                    Start dubbing
                    <ArrowRight className="h-4 w-4" />
                  </>
                )}
              </Button>
            </CardSection>
          </Card>

          {/* The runs, in the room under the text box — beside the settings
              rail, where the column would otherwise sit empty. */}
          <HomeRuns />
        </div>

        {/* ------------------------------------------------------- the rail */}
        {/* The rail is the tall region now that the card stopped matching it.
            It used to cap itself to the viewport and scroll its own overflow,
            which turned the bottom of the card into a guillotine: the
            Transcript hint sliced mid-line, Scope and the immutability note
            hidden in a scroll nothing advertised. The rail flows with the
            page instead — Start stays above the fold in the left column, and
            whatever the rail hangs past it is reachable the way everything on
            a page is, by scrolling the page. */}
        {/* `self-start`: the grid stretches its items so the runs column can
            fill, but the rail must not come along for the ride — stretched, it
            parks its slack between Scope and the mt-auto footer note as a void
            inside the card. Natural height, always. */}
        <Card data-region="options" className="flex flex-col self-start overflow-hidden rounded-3xl p-0">
          <CardSection className="px-5 pt-6 sm:px-5">
            <SectionLabel icon={Languages}>Languages</SectionLabel>
            <div className="mt-3 grid grid-cols-2 gap-3">
              <Field label="Spoken language">
                <Select
                  aria-label="Spoken language"
                  value={form.src_lang}
                  onChange={(event) => update({ src_lang: event.currentTarget.value })}
                >
                  {SRC_LANGS.map(([code, label]) => (
                    <option key={code} value={code}>
                      {label}
                    </option>
                  ))}
                </Select>
              </Field>
              <Field label="Dub into">
                <Select
                  aria-label="Dub into"
                  value={form.tgt_lang}
                  onChange={(event) => update({ tgt_lang: event.currentTarget.value })}
                >
                  {TGT_LANGS.map(([code, label]) => (
                    <option key={code} value={code}>
                      {label}
                    </option>
                  ))}
                </Select>
              </Field>
            </div>
            {form.src_lang === form.tgt_lang ? (
              <p className="mt-2 text-[12px] leading-relaxed text-muted" data-same-lang-note>
                Same language: every line is re-voiced in the speaker's cloned voice —
                no translation happens.
              </p>
            ) : null}

            {/*
              A third language is the case the two selects above cannot express:
              English inside a Hebrew→German run is neither the spoken language
              nor the dub's. The pipeline keeps such a line by default — it plays
              as recorded, with a subtitle — and this is the one place a run can
              say otherwise before it starts. It lives under Languages because it
              is a question about languages, not about scope.
            */}
            <label className="mt-3 flex items-start gap-2.5" data-dub-foreign>
              <Checkbox
                className="mt-[3px]"
                checked={Boolean(form.dub_foreign)}
                onChange={(event) => update({ dub_foreign: event.currentTarget.checked })}
              />
              <span className="min-w-0">
                <span className="block text-[12.5px] text-secondary">Dub foreign speech</span>
                <span className="mt-0.5 block text-[12px] leading-relaxed text-muted">
                  A third language inside the video is translated and voiced too — otherwise it
                  plays as recorded, subtitled. Individual lines can still be switched to dubbed
                  later, in the editor.
                </span>
              </span>
            </label>
          </CardSection>

          <Divider />

          <CardSection className="px-5 sm:px-5">
            <SectionLabel icon={Clapperboard}>Genre</SectionLabel>
            <OptionList label="Genre" className="mt-3">
              <OptionRow
                icon={Mic2}
                label="Documentary"
                hint="Narrated over pictures"
                selected={form.genre === "documentary"}
                onClick={() => update({ genre: "documentary" })}
              />
              <OptionRow
                icon={Film}
                label="Movie"
                hint="Scripted, spoken in scene"
                selected={form.genre === "movie"}
                onClick={() => update({ genre: "movie" })}
              />
            </OptionList>
          </CardSection>

          <Divider />

          <CardSection className="px-5 sm:px-5">
            <SectionLabel icon={MessagesSquare}>Register</SectionLabel>
            <Segmented className="mt-3 flex w-full shadow-none" role="radiogroup" aria-label="Register">
              {(
                [
                  ["narration", "Narration"],
                  ["dialogue", "Dialogue"],
                ] as const
              ).map(([value, label]) => (
                <button
                  key={value}
                  type="button"
                  role="radio"
                  aria-checked={form.register === value}
                  onClick={() => update({ register: value })}
                  className={segmentedCell(form.register === value, "flex-1 justify-center")}
                >
                  {label}
                </button>
              ))}
            </Segmented>
          </CardSection>

          <Divider />

          {/*
            Where the words come from.
            The pipeline takes `--transcript auto|captions|asr|file` and this
            screen had no way to say it, so a run whose auto-captions were
            mangled — the case invariant 4 in AGENTS.md exists for — could only be
            fixed from the CLI. `auto` is the pipeline's own answer and stays the
            default; the other three are for the user who has already heard the
            result. The fourth is the strongest of them: somebody who *has* the
            transcript does not need a better machine to guess at it, and until
            now the only way to hand one over was `--captions` on the CLI.
          */}
          <CardSection className="px-5 sm:px-5">
            <SectionLabel icon={Captions}>Transcript</SectionLabel>
            <Field className="mt-3" label="Where the words come from">
              <Select
                aria-label="Transcript source"
                value={form.transcript ?? "auto"}
                onChange={(event) =>
                  update({
                    transcript: event.currentTarget
                      .value as CreateProjectRequest["transcript"],
                  })
                }
              >
                <option value="auto">Automatic</option>
                <option value="captions">The video's captions</option>
                <option value="asr">Transcribe it here</option>
                <option value="file">A transcript I have</option>
              </Select>
            </Field>
            {/*
              The file row exists only for the answer that needs it. A path box
              standing open under a dropdown set to "Automatic" is a control that
              does nothing, which is the same lie as a disabled one with no
              reason on it — so it appears with the choice and goes with it.

              Stacked, not the Source row's field-then-button: the rail is 24rem
              and a path is long, so side by side leaves the box too narrow to
              read what is in it.
            */}
            {form.transcript === "file" ? (
              <div data-transcript-file className="mt-3 flex flex-col items-start gap-2">
                <TextInput
                  className="w-full min-w-0"
                  value={form.captions ?? ""}
                  aria-label="Transcript file"
                  placeholder="/Users/you/episode.srt"
                  onChange={(event) => update({ captions: event.currentTarget.value })}
                />
                <Button onClick={() => void chooseTranscript()}>
                  <FolderOpen className="h-4 w-4" />
                  Choose file
                </Button>
                <input
                  ref={transcriptRef}
                  type="file"
                  accept=".srt,.vtt,.json3"
                  className="hidden"
                  onChange={(event) => {
                    const file = event.currentTarget.files?.[0];
                    if (file) update({ captions: file.name });
                  }}
                />
              </div>
            ) : null}
            <p className="mt-2 text-[12px] leading-relaxed text-muted">
              {form.transcript === "file" ? (
                <>
                  An .srt, .vtt or .json3 — the words are taken from it as they are, and
                  nothing is transcribed. It has to carry timestamps: they are what places
                  each line, and plain text cannot be lined up with the audio.
                  {desktop ? null : (
                    <>
                      {" "}
                      The browser cannot read a file's real path, so <em>Choose file</em> only
                      fills in the name — paste the full path, or use the desktop app.
                    </>
                  )}
                </>
              ) : (
                <>
                  Automatic uses the downloaded captions when there are any and transcribes
                  locally otherwise. Force transcription when the captions are auto-generated
                  and mangled.
                </>
              )}
            </p>
          </CardSection>

          <Divider />

          <CardSection className="px-5 sm:px-5">
            <SectionLabel icon={Timer}>Scope</SectionLabel>
            <div className="mt-3 grid grid-cols-2 gap-3">
              <Field label="Duration cap" hint="blank = all of it">
                <NumberInput
                  min={0}
                  step={10}
                  suffix="sec"
                  value={form.duration ?? ""}
                  // An example, not an instruction. A bare "320" in an empty
                  // field reads as a value that is already set which is
                  // exactly the wrong thing to think about the one control that
                  // decides whether this run is four minutes or two hours.
                  placeholder="e.g. 320"
                  aria-label="Duration cap in seconds"
                  onChange={(event) =>
                    update({
                      duration:
                        event.currentTarget.value === "" ? null : Number(event.currentTarget.value),
                    })
                  }
                />
              </Field>
              <Field label="Run name" hint="blank = from the title">
                <TextInput
                  value={form.name ?? ""}
                  placeholder="e.g. my_first_dub"
                  aria-label="Run name"
                  onChange={(event) => update({ name: event.currentTarget.value || null })}
                />
              </Field>
            </div>
          </CardSection>

          {/*
            What is still a decision after this button, and what is not.
            Genre, register and context are inputs to the translator and can be
            corrected from the editor's run menu at any time. The source and the
            language pair are not: changing either invalidates the fetch and
            every stage after it, which is a new project wearing an old
            project's name. Saying so here costs one line and saves the run
            somebody would otherwise abandon rather than "risk" starting.
          */}
          <CardSection
            tone="sunken"
            className="mt-auto border-t border-border px-5 py-4 sm:px-5"
          >
            <p className="text-[12px] leading-relaxed text-muted" data-rail-note>
              Genre, register and context can be changed later; the source and languages cannot.
            </p>
          </CardSection>
        </Card>
      </div>

      {error ? (
        <ErrorBlock title="Could not start" onDismiss={() => setError(null)}>
          {error}
        </ErrorBlock>
      ) : null}
    </PageShell>
  );
}

/** How many rows the glance shows before it defers to the archive. */
const HOME_RUNS = 8;

/**
 * The workspace, under the form.
 *
 * Home is the new-dub form and stays the new-dub form — but a screen that only
 * ever offers to start something implies there is nothing already started, and
 * the run people actually want on a second visit is usually the one still in
 * flight. So the newest few rows sit under the card, running-first, in the same
 * rows /runs draws (`components/RunRow`), with "All runs" as the way to the
 * whole list. Four of them: enough to hold the one that is running and the ones
 * from this morning, few enough that the form is still what this page is.
 *
 * It is quiet about its failures on purpose. A server that did not answer is
 * worth a sentence here and a red card *there* — this region is a shortcut, and
 * a shortcut that shouts about a list you did not ask for is a worse home
 * screen than one that simply has no shortcut in it. Zero runs draws nothing at
 * all: the empty state for "no runs yet" is the form the user is looking at.
 */
function HomeRuns() {
  const navigate = useNavigate();
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    void api
      .listProjects()
      .then((list) => {
        if (!cancelled) setProjects(list);
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Vanishing here leaves the column a dead plane under the form — the exact
  // hole the user pointed at. While loading, on failure, and with zero runs
  // alike, the region keeps its place with a quiet card instead.
  if (failed || projects == null || projects.length === 0) {
    return (
      <section data-region="home-runs" className="flex min-h-0 flex-1 flex-col gap-2.5">
        <div className="px-1">
          <SectionLabel icon={Clapperboard}>Your runs</SectionLabel>
        </div>
        <div className="flex min-h-40 flex-1 items-center justify-center rounded-2xl border border-dashed border-border px-6 py-10">
          <p className="max-w-sm text-center text-[13px] leading-relaxed text-muted">
            {projects == null && !failed
              ? "Looking for runs…"
              : "Nothing here yet — every dub you start lands in this list, live while it runs."}
          </p>
        </div>
      </section>
    );
  }
  const ordered = orderRuns(projects);
  const shown = ordered.slice(0, HOME_RUNS);

  return (
    <section data-region="home-runs" className="flex min-h-0 flex-1 flex-col gap-2.5">
      <div className="flex flex-wrap items-center justify-between gap-3 px-1">
        <div className="flex items-baseline gap-3">
          <SectionLabel icon={Clapperboard}>Your runs</SectionLabel>
          <span className="text-[11px] tabular-nums text-muted">
            {projects.length} {projects.length === 1 ? "run" : "runs"} in outputs/
          </span>
        </div>
        {/* A link, not a button: the page's one *button* is Start dubbing, and
            a second filled control under it would compete with it. */}
        <Link
          to="/runs"
          data-all-runs
          className="rounded-md text-[11px] font-bold uppercase tracking-[0.14em] text-muted transition-colors hover:text-primary"
        >
          All runs →
        </Link>
      </div>
      {/* A strip of padding, so the hover lift is not shaved off at the edges. */}
      <ul className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto p-0.5">
        {shown.map((project) => (
          <RunRow
            key={project.name}
            project={project}
            compact
            onOpen={() => navigate(`/editor/${encodeURIComponent(project.name)}`)}
          />
        ))}
      </ul>
      {ordered.length > shown.length ? (
        <p className="px-1 text-[11px] text-muted">
          {ordered.length - shown.length} more on the runs page.
        </p>
      ) : null}
    </section>
  );
}
