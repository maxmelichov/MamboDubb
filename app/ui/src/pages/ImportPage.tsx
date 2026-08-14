/**
 * Import: pick a source, say what language it is and what language it should
 * become, then start a run.
 *
 * The context note is not decoration. Translation quality moves measurably with
 * a sentence about who and what the video is about and how names are spelled —
 * it is the difference between "Sheikha Moza" and three different manglings.
 */

import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { FileVideo, Loader2 } from "lucide-react";
import { AppHeader } from "../components/AppHeader";
import { Button, ErrorBar, Field, Panel, PanelHeader, Select, TextArea, TextInput } from "../components/ui";
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
    <div className="flex h-screen flex-col">
      <AppHeader />
      {error ? <ErrorBar message={error} onDismiss={() => setError(null)} /> : null}

      <main className="mx-auto w-full max-w-4xl flex-1 overflow-y-auto p-6">
        <h1 className="text-lg font-semibold">New dub</h1>
        <p className="mt-1 text-[13px] text-secondary">
          Everything runs on this machine. A full run takes a while, so cap the duration while
          you are iterating.
        </p>

        <Panel className="mt-4 p-3">
          <Field
            label="Source"
            hint={
              desktop ? (
                <>
                  A URL, or a local file — <em>Choose file</em> opens a real file dialog and fills
                  in the full path.
                </>
              ) : (
                <>
                  A URL, or an absolute path to a local file. The browser cannot read a file's real
                  path, so <em>Choose file</em> only fills in the name — paste the full path, or
                  use the desktop app.
                </>
              )
            }
          >
            <div className="flex gap-1.5">
              <TextInput
                value={form.source}
                placeholder="https://www.youtube.com/watch?v=… or /Users/you/clip.mp4"
                onChange={(event) => update({ source: event.currentTarget.value })}
              />
              <Button onClick={() => void chooseFile()}>
                <FileVideo className="h-3.5 w-3.5" />
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
          </Field>

          <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
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
                    duration: event.currentTarget.value === "" ? null : Number(event.currentTarget.value),
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

          <div className="mt-3 grid grid-cols-2 gap-3">
            <Field label="Genre">
              <Select
                value={form.genre ?? ""}
                onChange={(event) =>
                  update({ genre: (event.currentTarget.value || null) as CreateProjectRequest["genre"] })
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
                    register: (event.currentTarget.value || null) as CreateProjectRequest["register"],
                  })
                }
              >
                <option value="narration">Narration</option>
                <option value="dialogue">Dialogue</option>
              </Select>
            </Field>
          </div>

          <Field
            className="mt-3"
            label="Context"
            hint="Who and what this is about, and how names are spelled. This materially improves the translation."
          >
            <TextArea
              className="min-h-24"
              value={form.context ?? ""}
              placeholder="An Israeli documentary about Qatar and Sheikha Moza (Hebrew שייח'ה מוזה — the ASR often mangles it); her son Emir Tamim; the Muslim Brotherhood; Yusuf al-Qaradawi. Use these English spellings and respect grammatical gender."
              onChange={(event) => update({ context: event.currentTarget.value })}
            />
          </Field>

          <div className="mt-4 flex items-center gap-2">
            <Button variant="primary" onClick={start} disabled={starting}>
              {starting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}
              Start dubbing
            </Button>
            <span className="text-[12px] text-muted">
              One job runs at a time; the editor opens straight away with live progress.
            </span>
          </div>
        </Panel>

        <Panel className="mt-6">
          <PanelHeader>Existing runs</PanelHeader>
          {projects.length === 0 ? (
            <p className="p-3 text-[13px] text-muted">Nothing under outputs/ yet.</p>
          ) : (
            <ul className="divide-y divide-border">
              {projects.map((project) => (
                <li key={project.name}>
                  <button
                    type="button"
                    onClick={() => navigate(`/editor/${encodeURIComponent(project.name)}`)}
                    className="flex w-full items-center gap-3 px-3 py-2 text-left hover:bg-border/40"
                  >
                    <span className="font-mono text-[12px]">{project.name}</span>
                    <span className="min-w-0 flex-1 truncate text-[13px] text-secondary">
                      {project.title}
                    </span>
                    <span className="text-[11px] uppercase tracking-[0.08em] text-muted">
                      {project.src_lang} → {project.tgt_lang}
                    </span>
                    <span className="text-[11px] tabular-nums text-muted">
                      {project.duration ? timecode(project.duration, 0) : "—"}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </main>
    </div>
  );
}
