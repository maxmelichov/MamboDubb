/**
 * Import: pick a source, say what language it is and what language it should
 * become, then start a run.
 *
 * The context note is not decoration. Translation quality moves measurably with
 * a sentence about who and what the video is about and how names are spelled —
 * it is the difference between "Sheikha Moza" and three different manglings.
 * That is why it gets a section of its own rather than a corner of the form.
 *
 * Shape: one card, three labelled sections separated by hairlines, and a sunken
 * footer holding the single primary action. Existing runs are a second card of
 * rows below it — the only other thing you can do from this screen.
 */

import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  ArrowRight,
  ChevronRight,
  Clapperboard,
  FileVideo,
  FolderOpen,
  Languages,
  Loader2,
  PencilLine,
  PlugZap,
} from "lucide-react";
import { PageShell } from "../components/AppShell";
import { StageTrack } from "../components/StageTrack";
import {
  Badge,
  Button,
  Card,
  CardSection,
  Divider,
  Empty,
  ErrorBlock,
  Eyebrow,
  Field,
  LogoMark,
  NumberInput,
  SectionLabel,
  Select,
  TextArea,
  TextInput,
} from "../components/ui";
import { api } from "../lib/api";
import { isDesktop, pickVideoFile } from "../lib/desktop";
import { timecode } from "../lib/format";
import { ago, stageTone, summarizeStages } from "../lib/stages";
import type { CreateProjectRequest, ProjectSummary } from "../lib/types";

// What can be HEARD is broader than what can be SPOKEN: the ASR + translator
// handle these sources, but Qwen3-TTS voices exactly ten languages — Hebrew and
// Arabic are source-only. Offering them as dub targets would create a project
// whose tts stage can only fail, so the two lists are deliberately different.
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
      title="New dub."
      accent="Entirely on this machine."
      lede={
        <>
          Point it at a video, say which way to translate, and it runs the whole pipeline
          locally. A full run takes a while — cap the duration while you are iterating.
        </>
      }
    >
      <Card className="overflow-hidden p-0">
        <CardSection>
          <SectionLabel icon={FileVideo}>Source</SectionLabel>
          <div className="mt-3 flex flex-col gap-2 sm:flex-row">
            <TextInput
              className="h-11 flex-1 text-sm"
              value={form.source}
              aria-label="Source"
              placeholder="https://www.youtube.com/watch?v=… or /Users/you/clip.mp4"
              onChange={(event) => update({ source: event.currentTarget.value })}
            />
            <Button size="lg" onClick={() => void chooseFile()}>
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

        <CardSection>
          <SectionLabel icon={Languages}>Languages &amp; scope</SectionLabel>
          <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-4 sm:grid-cols-4">
            <Field label="Spoken language">
              <Select
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
            <Field label="Duration cap" hint="blank = the whole video">
              <NumberInput
                min={0}
                step={10}
                suffix="sec"
                value={form.duration ?? ""}
                placeholder="320"
                aria-label="Duration cap in seconds"
                onChange={(event) =>
                  update({
                    duration:
                      event.currentTarget.value === "" ? null : Number(event.currentTarget.value),
                  })
                }
              />
            </Field>
            <Field label="Run name" hint="blank to derive one from the title">
              <TextInput
                value={form.name ?? ""}
                placeholder="kan11_v4"
                onChange={(event) => update({ name: event.currentTarget.value || null })}
              />
            </Field>
          </div>
        </CardSection>

        <Divider />

        <CardSection>
          <SectionLabel icon={PencilLine}>Voice &amp; context</SectionLabel>
          <div className="mt-3 grid grid-cols-2 gap-4">
            <Field label="Genre">
              <Select
                value={form.genre ?? ""}
                onChange={(event) =>
                  update({
                    genre: (event.currentTarget.value || null) as CreateProjectRequest["genre"],
                  })
                }
              >
                <option value="documentary">Documentary</option>
                <option value="movie">Movie</option>
              </Select>
            </Field>
            <Field label="Register">
              <Select
                value={form.register ?? ""}
                onChange={(event) =>
                  update({
                    register: (event.currentTarget.value ||
                      null) as CreateProjectRequest["register"],
                  })
                }
              >
                <option value="narration">Narration</option>
                <option value="dialogue">Dialogue</option>
              </Select>
            </Field>
          </div>

          <Field
            className="mt-4"
            label="Context"
            hint="Who and what this is about, and how names are spelled. This materially improves the translation."
          >
            <TextArea
              className="min-h-28 text-[13px]"
              value={form.context ?? ""}
              placeholder="An Israeli documentary about Qatar and Sheikha Moza (Hebrew שייח'ה מוזה — the ASR often mangles it); her son Emir Tamim; the Muslim Brotherhood; Yusuf al-Qaradawi. Use these English spellings and respect grammatical gender."
              onChange={(event) => update({ context: event.currentTarget.value })}
            />
          </Field>
        </CardSection>

        <CardSection
          tone="sunken"
          className="flex flex-col gap-4 border-t border-border sm:flex-row sm:items-center sm:justify-between"
        >
          <p className="max-w-sm text-[12px] leading-relaxed text-muted">
            One job runs at a time; the editor opens straight away with live progress.
          </p>
          <Button variant="primary" size="lg" onClick={start} disabled={starting}>
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

      {error ? <ErrorBlock title="Could not start" onDismiss={() => setError(null)}>{error}</ErrorBlock> : null}

      <section className="flex flex-col gap-3">
        <div className="flex items-baseline justify-between gap-3 px-1">
          <Eyebrow>Existing runs</Eyebrow>
          <span className="text-[11px] tabular-nums text-muted">
            {projects && projects.length > 0
              ? `${projects.length} ${projects.length === 1 ? "run" : "runs"} in outputs/`
              : ""}
          </span>
        </div>

        <Card className="overflow-hidden p-0">
          {projects == null ? (
            <div className="flex items-center gap-3 px-6 py-7 text-[13px] text-muted">
              <Loader2 className="h-4 w-4 shrink-0 animate-spin" aria-hidden />
              Reading outputs/…
            </div>
          ) : listError ? (
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
              <code className="font-mono text-secondary">
                uv run python -m dubbing_app.server
              </code>
              , then try again.
            </Empty>
          ) : projects.length === 0 ? (
            <Empty className="py-9" icon={Clapperboard} title="No runs yet">
              Every dub you start lands here, resumable. Fill in the form above and press{" "}
              <em>Start dubbing</em> to make the first one.
            </Empty>
          ) : (
            <ul className="divide-y divide-border">
              {projects.map((project) => (
                <ProjectRow
                  key={project.name}
                  project={project}
                  onOpen={() => navigate(`/editor/${encodeURIComponent(project.name)}`)}
                />
              ))}
            </ul>
          )}
        </Card>
      </section>
    </PageShell>
  );
}

/**
 * One existing run.
 *
 * The row used to say the title, the run name, the language pair and the
 * duration — everything except the thing you actually came to find out, which
 * is *where this run got to*. A half-finished run and a finished one looked
 * identical, so the only way to tell them apart was to open both. Now the row
 * carries the pipeline position (nine dots, the shared track), a word for it,
 * and when it last moved.
 */
function ProjectRow({ project, onOpen }: { project: ProjectSummary; onOpen: () => void }) {
  const summary = summarizeStages(project.stages);
  return (
    <li>
      <button
        type="button"
        onClick={onOpen}
        className="group flex w-full items-center gap-4 px-5 py-3.5 text-left transition-colors hover:bg-sunken"
      >
        <span className="grid h-9 w-9 shrink-0 place-items-center rounded-lg border border-border bg-sunken text-muted transition-colors group-hover:border-axis group-hover:text-primary">
          <LogoMark className="h-4 w-4" />
        </span>

        <span className="min-w-0 flex-1">
          <span className="flex min-w-0 items-center gap-2">
            <span className="truncate text-[13px] font-semibold text-primary">{project.title}</span>
            <Badge tone={stageTone(summary)}>{summary.label}</Badge>
          </span>
          <span className="mt-0.5 flex min-w-0 items-center gap-2 text-[11px] text-muted">
            <span className="truncate font-mono">{project.name}</span>
            <span aria-hidden>·</span>
            <span className="shrink-0 whitespace-nowrap">{ago(project.mtime)}</span>
          </span>
        </span>

        <StageTrack
          stages={project.stages}
          showLabel={false}
          className="hidden shrink-0 md:inline-flex"
        />

        <span className="hidden shrink-0 text-[10px] font-bold uppercase tracking-[0.14em] text-muted sm:block">
          {project.src_lang} → {project.tgt_lang}
        </span>
        <span className="w-12 shrink-0 text-right text-[11px] tabular-nums text-muted">
          {project.duration ? timecode(project.duration, 0) : "—"}
        </span>
        <ChevronRight className="h-4 w-4 shrink-0 text-muted transition-transform group-hover:translate-x-0.5 group-hover:text-primary" />
      </button>
    </li>
  );
}
