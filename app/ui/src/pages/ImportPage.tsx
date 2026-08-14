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
} from "lucide-react";
import { PageShell } from "../components/AppShell";
import {
  Button,
  Card,
  CardSection,
  Divider,
  ErrorBlock,
  Eyebrow,
  Field,
  LogoMark,
  SectionLabel,
  Select,
  TextArea,
  TextInput,
} from "../components/ui";
import { api } from "../lib/api";
import { isDesktop, pickVideoFile } from "../lib/desktop";
import { timecode } from "../lib/format";
import type { CreateProjectRequest, ProjectSummary } from "../lib/types";

const LANGS = [
  ["he", "Hebrew"],
  ["en", "English"],
  ["ar", "Arabic"],
  ["ru", "Russian"],
  ["fr", "French"],
  ["es", "Spanish"],
  ["de", "German"],
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
  const [projects, setProjects] = useState<ProjectSummary[]>([]);

  const update = (patch: Partial<CreateProjectRequest>) =>
    setForm((current) => ({ ...current, ...patch }));

  useEffect(() => {
    let cancelled = false;
    void api
      .listProjects()
      .then((list) => !cancelled && setProjects(list))
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

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
                {LANGS.map(([code, label]) => (
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
                {LANGS.map(([code, label]) => (
                  <option key={code} value={code}>
                    {label}
                  </option>
                ))}
              </Select>
            </Field>
            <Field label="Duration cap" hint="seconds, blank for the whole video">
              <TextInput
                type="number"
                min={0}
                value={form.duration ?? ""}
                placeholder="320"
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
            {projects.length === 0
              ? ""
              : `${projects.length} ${projects.length === 1 ? "run" : "runs"} in outputs/`}
          </span>
        </div>

        <Card className="overflow-hidden p-0">
          {projects.length === 0 ? (
            <div className="flex items-center gap-3 px-6 py-7 text-[13px] text-muted">
              <Clapperboard className="h-4 w-4 shrink-0" aria-hidden />
              Nothing under outputs/ yet. Your first run will show up here.
            </div>
          ) : (
            <ul className="divide-y divide-border">
              {projects.map((project) => (
                <li key={project.name}>
                  <button
                    type="button"
                    onClick={() => navigate(`/editor/${encodeURIComponent(project.name)}`)}
                    className="group flex w-full items-center gap-4 px-5 py-3.5 text-left transition-colors hover:bg-sunken"
                  >
                    <span className="grid h-9 w-9 shrink-0 place-items-center rounded-lg border border-border bg-sunken text-muted transition-colors group-hover:border-axis group-hover:text-primary">
                      <LogoMark className="h-4 w-4" />
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-[13px] font-semibold text-primary">
                        {project.title}
                      </span>
                      <span className="mt-0.5 block truncate font-mono text-[11px] text-muted">
                        {project.name}
                      </span>
                    </span>
                    <span className="hidden shrink-0 text-[10px] font-bold uppercase tracking-[0.14em] text-muted sm:block">
                      {project.src_lang} → {project.tgt_lang}
                    </span>
                    <span className="w-12 shrink-0 text-right text-[11px] tabular-nums text-muted">
                      {project.duration ? timecode(project.duration, 0) : "—"}
                    </span>
                    <ChevronRight className="h-4 w-4 shrink-0 text-muted opacity-0 transition-opacity group-hover:opacity-100" />
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </section>
    </PageShell>
  );
}
