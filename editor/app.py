"""FastAPI app: a small editor over the pipeline's run directories.

    uv run --extra app uvicorn editor.app:app --reload

The server never loads a model. It reads and patches `outputs/<run>/manifest.json`
through `dubbing.manifest`, serves the run's media, and shells out to
`python -m dubbing …` for anything that actually computes.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dubbing import STAGES

from . import jobs, runs
from .edits import EDITABLE, apply_edits, earliest

app = FastAPI(title="Dubbing Editor")

STATIC = Path(__file__).parent / "static"


def _run_or_404(fn, *args):
    try:
        return fn(*args)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from None


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "outputs": str(runs.OUTPUTS), "stages": list(STAGES),
            "editable": sorted(EDITABLE)}


@app.get("/api/runs")
def api_runs() -> dict[str, Any]:
    return {"runs": runs.list_runs()}


@app.get("/api/runs/{run}")
def api_run(run: str) -> dict[str, Any]:
    return _run_or_404(runs.view, run)


class Edit(BaseModel):
    id: int
    fields: dict[str, Any]


class PatchBody(BaseModel):
    edits: list[Edit]


@app.patch("/api/runs/{run}/segments")
def api_patch(run: str, body: PatchBody) -> dict[str, Any]:
    m = _run_or_404(runs.load, run)
    try:
        result = apply_edits(m, [e.model_dump() for e in body.edits])
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    if result["changed"]:
        runs.save(run, m)
    return result


class RerunBody(BaseModel):
    force: str | None = None
    extra_args: list[str] = []


@app.post("/api/runs/{run}/rerun")
def api_rerun(run: str, body: RerunBody) -> dict[str, Any]:
    m = _run_or_404(runs.load, run)
    src = m.get("source") or {}
    if body.force and body.force not in STAGES and body.force != "all":
        raise HTTPException(400, f"unknown stage {body.force!r}")
    if not src.get("input"):
        raise HTTPException(400, "run has no recorded input to re-run from")
    cmd = jobs.dub_command(
        src["input"], runs.workdir(run), src=src.get("src_lang") or "he",
        tgt=src.get("tgt_lang") or "en", duration=src.get("duration_limit"),
        force=body.force, opts=runs.recorded_opts(m), extra=body.extra_args,
    )
    if src.get("context"):
        cmd += ["--context", src["context"]]
    job = jobs.launch(cmd, run, f"rerun {run}" + (f" --force {body.force}" if body.force else ""))
    return job.state()


@app.post("/api/import")
async def api_import(
    url: str | None = Form(None),
    file: UploadFile | None = None,
    src: str = Form("he"),
    tgt: str = Form("en"),
    duration: float | None = Form(None),
) -> dict[str, Any]:
    """Start a pipeline run from a URL or an uploaded video file."""
    if bool(url) == bool(file):
        raise HTTPException(400, "provide exactly one of url or file")
    if url:
        source: str = url.strip()
    else:
        assert file is not None
        runs.UPLOADS.mkdir(parents=True, exist_ok=True)
        dest = runs.UPLOADS / Path(file.filename or "upload.mp4").name
        with dest.open("wb") as fh:
            shutil.copyfileobj(file.file, fh)
        source = str(dest)
    workdir = runs.default_workdir(source)
    cmd = jobs.dub_command(source, workdir, src=src, tgt=tgt, duration=duration)
    job = jobs.launch(cmd, workdir.name, f"import {workdir.name}")
    return job.state()


@app.get("/api/jobs")
def api_jobs() -> dict[str, Any]:
    return {"jobs": jobs.listing()}


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str) -> dict[str, Any]:
    try:
        return jobs.get(job_id).state()
    except KeyError:
        raise HTTPException(404, "no such job") from None


@app.post("/api/jobs/{job_id}/cancel")
def api_job_cancel(job_id: str) -> dict[str, Any]:
    try:
        job = jobs.get(job_id)
    except KeyError:
        raise HTTPException(404, "no such job") from None
    job.cancel()
    return job.state()


@app.get("/api/runs/{run}/file/{path:path}")
def api_file(run: str, path: str) -> FileResponse:
    return FileResponse(_run_or_404(runs.resolve_in_run, run, path))


@app.get("/api/runs/{run}/segments/{seg_id}/dubbed")
def api_dubbed(run: str, seg_id: int) -> FileResponse:
    m = _run_or_404(runs.load, run)
    seg = next((s for s in m.get("segments") or [] if s["id"] == seg_id), None)
    if seg is None:
        raise HTTPException(404, "no such segment")
    try:
        return FileResponse(runs.dubbed_clip(run, seg))
    except KeyError:
        raise HTTPException(404, "segment has no synthesised clip yet") from None


@app.get("/api/runs/{run}/segments/{seg_id}/original")
def api_original(run: str, seg_id: int) -> FileResponse:
    m = _run_or_404(runs.load, run)
    seg = next((s for s in m.get("segments") or [] if s["id"] == seg_id), None)
    if seg is None:
        raise HTTPException(404, "no such segment")
    try:
        return FileResponse(runs.original_slice(run, seg))
    except KeyError:
        raise HTTPException(404, "run has no source.wav yet") from None


@app.get("/api/stage-for-fields")
def api_stage_for_fields(fields: str = "") -> dict[str, Any]:
    """Which stage a set of edited fields would force — used by the UI hint."""
    names = [f for f in fields.split(",") if f]
    unknown = [f for f in names if f not in EDITABLE]
    if unknown:
        raise HTTPException(400, f"not editable: {', '.join(unknown)}")
    return {"force": earliest([EDITABLE[f][1] for f in names])}


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
