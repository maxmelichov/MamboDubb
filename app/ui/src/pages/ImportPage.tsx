/**
 * Import: pick a source, say what language it is and what language it should
 * become, then start a run.
 *
 * The shape is a studio composition, not a form in a column. Three regions:
 *
 * - **New dub** — the generous card. It holds the two things you *type*: the
 *   source, and the context note. The single primary action sits in a sunken
 *   band at its foot, bottom-right, so it reads as the card's conclusion.
 * - **Options** — the rail beside it. Five labelled groups, hairlined apart:
 *   languages, genre, register, transcript, scope. Nothing here is typed prose;
 *   it is all picking, which is why it is a rail and not a second column of
 *   paragraphs. It ends with the sentence that says which of these choices are
 *   final — because two of them are, and a screen that does not say so is asking
 *   people to guess whether a run is a commitment.
 * - **Existing runs** — full width underneath, as cards. The only other thing
 *   you can do from this screen is re-open one.
 *
 * The context note is not decoration. Translation quality moves measurably with
 * a sentence about who and what the video is about and how names are spelled —
 * it is the difference between one consistent spelling of a name and three
 * different manglings. That is why it gets the whole lower half of the primary
 * card rather than a corner of the form.
 *
 * Genre and register used to be two more dropdowns. They are two-way choices
 * whose options need a clause to be meaningful ("Documentary — narrated,
 * factual"), and a clause does not fit in an `<option>`; the genre is rows that
 * invert to ink when picked, the register is a pill.
 */

import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  ArrowRight,
  Captions,
  ChevronRight,
  Clapperboard,
  Film,
  FileVideo,
  FolderOpen,
  Languages,
  Loader2,
  MessagesSquare,
  Mic2,
  PencilLine,
  PlugZap,
  Timer,
} from "lucide-react";
import { PageShell } from "../components/AppShell";
import { StageTrack } from "../components/StageTrack";
import {
  Badge,
  Button,
  Card,
  CardSection,
  Checkbox,
  Divider,
  Empty,
  ErrorBlock,
  Field,
  LogoMark,
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
import { cn } from "../lib/classNames";
import { isDesktop, pickVideoFile } from "../lib/desktop";
import { timecode } from "../lib/format";
import { ago, stageTone, summarizeStages } from "../lib/stages";
import type { CreateProjectRequest, ProjectSummary } from "../lib/types";

// What can be HEARD is broader than what can be SPOKEN: the ASR + translator
// handle these sources, but the synthesizer voices Qwen3-TTS's ten languages
// plus Hebrew (a LoRA over the same checkpoint; the server refuses a Hebrew
// target with the download command if the adapter isn't installed). Arabic is
// still source-only — offering it as a target would create a project whose tts
// stage can only fail, so the two lists stay deliberately different.
const SRC_LANGS = [
  ["he", "Hebrew"],
  ["en", "English"],
  ["ar", "Arabic"],
  ["ru", "Russian"],
  ["fr", "French"],
  ["es", "Spanish"],
  ["de", "German"],
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
    // Off, because off is what the pipeline does when nobody says otherwise.
    dub_foreign: false,
  });
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);
  // Distinct from "no projects": one means the form worked and there is
  // nothing to open, the other means the server never answered, and telling a
  // user the first when it was the second is how they end up re-running a
  // dub they already have.
  const [listError, setListError] = useState<string | null>(null);

  const update = (patch: Partial<CreateProjectRequest>) =>
    setForm((current) => ({ ...current, ...patch }));

  const [reloads, setReloads] = useState(0);
  useEffect(() => {
    let cancelled = false;
    void api
      .listProjects()
      .then((list) => {
        if (cancelled) return;
        setProjects(list);
        setListError(null);
      })
      .catch((err) => {
        if (cancelled) return;
        setProjects([]);
        setListError(String(err instanceof Error ? err.message : err));
      });
    return () => {
      cancelled = true;
    };
  }, [reloads]);

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

  const start = async () => {
    if (!form.source.trim()) {
      setError("Give it a video: a URL, or the full path to a local file.");
      return;
    }
    setStarting(true);
    setError(null);
    try {
      const created = await api.createProject({ ...form, context: form.context?.trim() || null });
      navigate(`/editor/${encodeURIComponent(created.name)}`);
    } catch (err) {
      setError(String(err instanceof Error ? err.message : err));
      setStarting(false);
    }
  };

  return (
    <PageShell
      width="wide"
      title="New dub."
      accent="Entirely on this machine."
      lede={
        <>
          Point it at a video, say which way to translate, and it runs the whole pipeline
          locally. A full run takes a while — cap the duration while you are iterating.
        </>
      }
    >
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_21rem] xl:grid-cols-[minmax(0,1fr)_24rem]">
        {/* ------------------------------------------------------- the card */}
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
                  A URL, or a local file — <em>Choose file</em> opens a real file dialog and fills
                  in the full path.
                </>
              ) : (
                <>
                  A URL, or an absolute path to a local file. The browser cannot read a file's real
                  path, so <em>Choose file</em> only fills in the name — paste the full path, or use
                  the desktop app.
                </>
              )}
            </p>
          </CardSection>

          <Divider />

          {/* The card's own big text area. Borderless on purpose: inside a card
              this generous, a second boxed field is a box in a box, and the
              focus ring in App.css is affordance enough once the caret is in. */}
          <CardSection className="flex flex-1 flex-col pb-6">
            <SectionLabel icon={PencilLine}>Context</SectionLabel>
            <TextArea
              className={cn(
                "mt-2 min-h-40 flex-1 resize-none border-transparent bg-transparent px-0 text-[13.5px]",
                "hover:border-transparent focus:border-transparent",
              )}
              aria-label="Context"
              value={form.context ?? ""}
              placeholder="Who and what this is about, and how names are spelled. For example: a news interview about the housing market; the host is Dana (she), the guest is Prof. Ronen Levi (he) — keep these spellings."
              onChange={(event) => update({ context: event.currentTarget.value })}
            />
            <p className="mt-3 text-[11.5px] leading-relaxed text-muted">
              Optional, and the single cheapest thing on this screen: a sentence of context
              materially improves the translation.
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

        {/* ------------------------------------------------------- the rail */}
        <Card data-region="options" className="flex flex-col overflow-hidden rounded-3xl p-0">
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
                  plays as recorded, subtitled — individual lines can still be switched to dubbed
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
            The pipeline takes `--transcript auto|captions|asr` and this screen
            had no way to say it, so a run whose auto-captions were mangled — the
            case invariant 4 in AGENTS.md exists for — could only be fixed from
            the CLI. `auto` is the pipeline's own answer and stays the default;
            the other two are for the user who has already heard the result.
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
              </Select>
            </Field>
            <p className="mt-2 text-[12px] leading-relaxed text-muted">
              Automatic uses the downloaded captions when there are any and transcribes locally
              otherwise. Force transcription when the captions are auto-generated and mangled.
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
                  // field reads as a value that is already set — which is
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

      {/* ------------------------------------------------------------ runs */}
      <section data-region="runs" className="flex flex-col gap-3.5">
        <div className="flex items-baseline justify-between gap-3 px-1">
          <SectionLabel icon={Clapperboard}>Existing runs</SectionLabel>
          <span className="text-[11px] tabular-nums text-muted">
            {projects && projects.length > 0
              ? `${projects.length} ${projects.length === 1 ? "run" : "runs"} in outputs/`
              : ""}
          </span>
        </div>

        {projects == null ? (
          <Card className="rounded-3xl">
            <div className="flex items-center gap-3 px-6 py-7 text-[13px] text-muted">
              <Loader2 className="h-4 w-4 shrink-0 animate-spin" aria-hidden />
              Reading outputs/…
            </div>
          </Card>
        ) : listError ? (
          <Card className="rounded-3xl">
            <Empty
              className="py-9"
              icon={PlugZap}
              title="Can't reach the studio server"
              action={
                <Button size="sm" onClick={() => setReloads((n) => n + 1)}>
                  Try again
                </Button>
              }
            >
              Existing runs live on the server, and it did not answer:{" "}
              <span className="text-secondary">{listError}</span>. Start it with{" "}
              <code className="font-mono text-secondary">uv run python -m dubbing_app.server</code>,
              then try again.
            </Empty>
          </Card>
        ) : projects.length === 0 ? (
          <Card className="rounded-3xl">
            <Empty className="py-9" icon={Clapperboard} title="No runs yet">
              Every dub you start lands here, resumable. Fill in the card above and press{" "}
              <em>Start dubbing</em> to make the first one.
            </Empty>
          </Card>
        ) : (
          <ul className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {projects.map((project) => (
              <ProjectCard
                key={project.name}
                project={project}
                onOpen={() => navigate(`/editor/${encodeURIComponent(project.name)}`)}
              />
            ))}
          </ul>
        )}
      </section>
    </PageShell>
  );
}

/**
 * One existing run.
 *
 * The row this replaces said the title, the run name, the language pair and the
 * duration — everything except the thing you actually came to find out, which
 * is *where this run got to*. A half-finished run and a finished one looked
 * identical, so the only way to tell them apart was to open both. The card
 * carries the pipeline position (nine dots, the shared track), a word for it,
 * and when it last moved.
 *
 * A card rather than a row because the page is wide now: three runs across at
 * 1440px is a composition, and the same three as full-width rows is three
 * hairlines with eleven hundred pixels of nothing in the middle of them.
 */
function ProjectCard({ project, onOpen }: { project: ProjectSummary; onOpen: () => void }) {
  const summary = summarizeStages(project.stages);
  return (
    <li className="flex">
      <button
        type="button"
        onClick={onOpen}
        className={cn(
          "group flex w-full flex-col gap-3 rounded-2xl border border-border bg-surface p-4 text-left",
          "shadow-card transition-all hover:-translate-y-0.5 hover:border-axis hover:shadow-lift",
        )}
      >
        <span className="flex items-start gap-3">
          <span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl border border-border bg-sunken text-muted transition-colors group-hover:border-axis group-hover:text-primary">
            <LogoMark className="h-4 w-4" />
          </span>
          <span className="min-w-0 flex-1">
            <span className="block truncate text-[13.5px] font-semibold text-primary">
              {project.title}
            </span>
            <span className="mt-0.5 block truncate font-mono text-[11px] text-muted">
              {project.name}
            </span>
          </span>
          <ChevronRight className="mt-1 h-4 w-4 shrink-0 text-muted transition-transform group-hover:translate-x-0.5 group-hover:text-primary" />
        </span>

        <span className="flex flex-wrap items-center gap-2">
          <Badge tone={stageTone(summary)}>{summary.label}</Badge>
          <span className="text-[11px] text-muted">{ago(project.mtime)}</span>
        </span>

        <span className="mt-auto flex items-center justify-between gap-2 border-t border-border pt-3">
          <StageTrack stages={project.stages} showLabel={false} />
          <span className="flex shrink-0 items-center gap-1.5 text-[10px] font-bold uppercase tracking-[0.14em] text-muted">
            {project.src_lang} → {project.tgt_lang}
            <span aria-hidden>·</span>
            <span className="tabular-nums">
              {project.duration ? timecode(project.duration, 0) : "—"}
            </span>
          </span>
        </span>
      </button>
    </li>
  );
}
