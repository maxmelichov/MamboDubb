"""Studio server tests routing, the error envelope, media, jobs, NDJSON.

No models and no real pipeline: the run directory is built by hand
(`conftest_app.make_project`) and every job runs through an injected fake runner,
so these are as fast and as deterministic as the rest of the suite.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
pytest.importorskip("uvicorn")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from dubbing import manifest  # noqa: E402
from dubbing_app import events, install as install_mod, media, ops  # noqa: E402
from dubbing_app import runner as runner_mod, server  # noqa: E402
from dubbing_app import app as app_mod  # noqa: E402
from dubbing_app.app import create_app  # noqa: E402
from dubbing_app.jobs import JobQueue  # noqa: E402
from tests.conftest_app import make_project  # noqa: E402

NAME = "whatsapp_0809"


class FakeRunner:
    """Records what ran, and (optionally) blocks so concurrency can be observed."""

    def __init__(self, hold: threading.Event | None = None):
        self.hold = hold
        self.calls: list[tuple[str, str, dict]] = []
        self.concurrent = 0
        self.max_concurrent = 0
        self.cancelled: list[str] = []
        self._lock = threading.Lock()

    def run(self, job, emit):
        with self._lock:
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
            self.calls.append((job.kind, job.project, dict(job.payload)))
        try:
            emit(events.stage_event("tts", "running", 0.5, "1/2"))
            if self.hold is not None:
                self.hold.wait(5.0)
            return {"kind": job.kind}
        finally:
            with self._lock:
                self.concurrent -= 1

    def cancel(self, job):
        self.cancelled.append(job.id)
        if self.hold is not None:
            self.hold.set()


@pytest.fixture()
def outputs(tmp_path):
    root = tmp_path / "outputs"
    make_project(root, NAME)
    return root


@pytest.fixture()
def fake():
    return FakeRunner()


@pytest.fixture()
def client(outputs, fake):
    # ui_dir="" keeps the SPA catch-all out of API tests: with a built dist on
    # the machine it would answer any unmatched path with index.html, turning
    # asserted 404s into 200s depending on whether `pnpm build` ever ran here.
    with TestClient(create_app(outputs, runner=fake, ui_dir="")) as c:
        yield c


@contextmanager
def live_server(app):
    """A real uvicorn on a real loopback socket.

    The NDJSON stream never ends, and both `TestClient` and `httpx.ASGITransport`
    buffer a response to completion before handing it back so an endless body
    deadlocks them. Only a real socket can be read frame by frame, which is
    exactly the property being tested.
    """
    import uvicorn

    sock = server.bind("127.0.0.1", 0)
    port = sock.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app, log_config=None, access_log=False))
    thread = threading.Thread(target=srv.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        assert wait_until(lambda: srv.started, 10.0), "server did not start"
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.should_exit = True
        thread.join(10.0)


@pytest.fixture()
def live(outputs, fake):
    with live_server(create_app(outputs, runner=fake, ui_dir="")) as base:
        yield base


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------

def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["version"]


def test_list_projects(client):
    body = client.get("/api/projects").json()
    assert [p["name"] for p in body["projects"]] == [NAME]
    project = body["projects"][0]
    assert project["src_lang"] == "he" and project["tgt_lang"] == "en"
    assert project["segments"] == 5 and project["complete"] is True


def test_list_skips_a_directory_with_no_manifest(client, outputs):
    # An abandoned run dir is not a project: get_project 404s on it, so a listed
    # row could only dead-end in the editor (with its events stream retrying a
    # permanent 404 underneath).
    (outputs / "abandoned_run").mkdir()
    body = client.get("/api/projects").json()
    assert [p["name"] for p in body["projects"]] == [NAME]


def test_create_project_reaches_the_cli_with_every_option(client, outputs, fake):
    """The whole creation path on the wire: body → `source` record → job payload →
    the argv `dubbing.cli` actually parses. A flag that drifted here is an option
    the user picked in the UI and silently did not get."""
    from dubbing import cli

    # A real caption file: the server opens what it is handed (see
    # `test_create_project_refuses_a_transcript_file_it_cannot_read`), so a
    # made-up path here would be testing the refusal instead of the plumbing.
    caps = outputs / "caps.json3"
    caps.write_text('{"events": [{"tStartMs": 0, "segs": [{"utf8": "hi"}]}]}',
                    encoding="utf-8")
    body = {"source": "https://youtu.be/abc", "tgt_lang": "ru", "duration": 60,
            "name": "movieproj", "context": "a note", "genre": "movie",
            "register": "dialogue", "transcript": "asr", "tts_model": "1.7b",
            "dub_foreign": True, "captions": str(caps)}
    r = client.post("/api/projects", json=body)
    assert r.status_code == 201 and r.json()["project"]["name"] == "movieproj"
    assert wait_until(lambda: fake.calls, 5.0)

    kind, project, payload = fake.calls[0]
    assert (kind, project) == ("run", "movieproj")
    args = cli.parse_args(ops.full_run_argv(Path(payload["workdir"]), payload["source"]))
    assert (args.source, args.tgt, args.duration) == ("https://youtu.be/abc", "ru", 60)
    assert (args.genre, args.register, args.transcript) == ("movie", "dialogue", "asr")
    assert args.tts_model == "1.7b" and args.dub_foreign is True
    assert args.context == "a note" and str(args.captions) == str(caps)
    # …and the options are on the manifest too, for every later edit job.
    stored = manifest.load(outputs / "movieproj")["source"]["app_opts"]
    assert stored["genre"] == "movie" and stored["tts_model"] == "1.7b"


SRT_BODY = "1\n00:00:01,000 --> 00:00:03,000\nHello there\n"


def test_create_project_takes_a_transcript_the_user_already_has(client, outputs, fake):
    """The whole point of the feature, end to end on the wire: a .srt path in the
    body becomes `--transcript file --captions <path>` in the argv the CLI parses,
    and the run reads those words instead of downloading or transcribing any."""
    from dubbing import cli

    caps = outputs / "mine.srt"
    caps.write_text(SRT_BODY, encoding="utf-8")
    r = client.post("/api/projects", json={"source": "https://youtu.be/abc",
                                           "name": "hastranscript",
                                           "transcript": "file", "captions": str(caps)})
    assert r.status_code == 201
    assert wait_until(lambda: fake.calls, 5.0)
    args = cli.parse_args(ops.full_run_argv(Path(fake.calls[0][2]["workdir"]),
                                            fake.calls[0][2]["source"]))
    assert args.transcript == "file" and str(args.captions) == str(caps)
    stored = manifest.load(outputs / "hastranscript")["source"]["app_opts"]
    assert stored["captions"] == str(caps) and stored["transcript"] == "file"


@pytest.mark.parametrize("name,body,message", [
    ("missing.srt", SRT_BODY, "not found"),          # …written to a different name
    ("notes.txt", "no timestamps here", "Plain text"),
    ("empty.srt", "nothing that is a cue\n", "no timed words"),
])
def test_create_project_refuses_a_transcript_file_it_cannot_read(client, outputs,
                                                                 name, body, message):
    """Read at the door, not at the transcript stage. A typo'd path or the wrong
    file is a sentence here; there it is a job that dies after a download and a
    Demucs separation, with an error about a file nobody remembers naming."""
    path = outputs / name
    if name != "missing.srt":
        path.write_text(body, encoding="utf-8")
    r = client.post("/api/projects", json={"source": "https://youtu.be/abc",
                                           "name": "badtranscript",
                                           "transcript": "file", "captions": str(path)})
    assert r.status_code == 400
    env = envelope_of(r)
    assert env["code"] == "invalid_request" and message in env["message"]
    assert not (outputs / "badtranscript").exists()


def test_create_project_refuses_the_file_mode_with_no_file(client, outputs):
    r = client.post("/api/projects", json={"source": "https://youtu.be/abc",
                                           "name": "nofile", "transcript": "file"})
    assert r.status_code == 400 and "nothing to read" in envelope_of(r)["message"]
    assert not (outputs / "nofile").exists()


def test_create_project_refuses_an_option_the_cli_cannot_take(client, outputs):
    """argparse would reject it, but only in the job child, minutes later, as a
    usage dump after the project directory and its manifest already exist. There
    is no way back from that project: its one job can never succeed."""
    r = client.post("/api/projects", json={"source": "x.mp4", "genre": "banana"})
    assert r.status_code == 400 and envelope_of(r)["code"] == "invalid_request"
    assert sorted(p.name for p in outputs.iterdir()) == [NAME]


def test_create_project_refuses_a_language_the_pipeline_cannot_dub(client, outputs):
    """A bad language code is worse than a bad genre: `script.script_for` answers
    "latin" for anything it does not know, so a project created with "jp" would
    not fail — it would run to the end with every script-derived verdict (keep vs
    dub, gloss matching, source-leak detection) quietly built on a lie."""
    url = "https://youtu.be/abc"
    for field, bad in (("src_lang", "jp"),          # a typo for ja
                       ("src_lang", "klingon"),
                       ("tgt_lang", "xx"),
                       ("tgt_lang", "ar")):         # Arabic reads, but has no voice
        r = client.post("/api/projects", json={"source": url, field: bad})
        assert r.status_code == 400, (field, bad)
        env = envelope_of(r)
        assert env["code"] == "invalid_request" and field in env["message"]
    assert sorted(p.name for p in outputs.iterdir()) == [NAME]


def test_create_project_accepts_the_languages_that_opened(client, outputs, fake):
    from dubbing import cli

    body = {"source": "https://youtu.be/abc", "src_lang": "ja",
            "tgt_lang": "pt", "name": "japt"}
    r = client.post("/api/projects", json=body)
    assert r.status_code == 201
    assert wait_until(lambda: fake.calls, 5.0)
    args = cli.parse_args(ops.full_run_argv(Path(fake.calls[0][2]["workdir"]),
                                            fake.calls[0][2]["source"]))
    assert (args.src, args.tgt) == ("ja", "pt")
    assert manifest.load(outputs / "japt")["source"]["src_lang"] == "ja"


@pytest.mark.parametrize("flag,dest,literal", [
    ("--genre", "genre", app_mod.Genre),
    ("--register", "register", app_mod.Register),
    ("--transcript", "transcript", app_mod.Transcript),
    ("--tts-model", "tts_model", app_mod.TtsModel),
    ("--src", "src", app_mod.SrcLang),
    ("--tgt", "tgt", app_mod.TgtLang),
])
def test_create_project_options_are_exactly_the_cli_choices(flag, dest, literal, capsys):
    """`dubbing.cli` cannot be imported here it drags in torch so the choice
    lists are restated in `dubbing_app.app`. This is what keeps the copy honest."""
    import re
    from typing import get_args

    from dubbing import cli

    for value in get_args(literal):
        assert getattr(cli.parse_args(["src", flag, value]), dest) == value
    with pytest.raises(SystemExit):
        cli.parse_args(["src", flag, "__not_a_choice__"])
    tail = capsys.readouterr().err.split("choose from")[-1]
    # Superset, not equality: the CLI may accept retired values (0.6b) so an old
    # run's recorded options still re-run, but the API must never OFFER a value
    # the CLI would reject that direction makes an unrunnable project.
    assert set(re.findall(r"[\w.\-]+", tail)) >= set(get_args(literal))


def test_ui_language_lists_match_the_cli():
    """The UI restates the language lists a third time — ImportPage for the run
    pair, SelectionPanel for per-segment overrides — with no fixture or build
    step tying them to `cli.SRC_LANGS`/`cli.TGT_LANGS`. The Literal↔CLI leg is
    pinned above; this reads the .tsx source so the UI leg cannot drift either
    (a UI-only code would create projects the pipeline refuses; a missing one
    hides a language that works)."""
    import re

    from dubbing import cli

    ui = Path(__file__).resolve().parents[1] / "app" / "ui" / "src"

    def codes(path: Path, name: str) -> set[str]:
        src = path.read_text(encoding="utf-8")
        # The array literal, whether one line of codes (SelectionPanel) or
        # [code, label] pairs (ImportPage); labels are capitalised, so a bare
        # two-lowercase-letter string is a code. "" (inherit) never matches.
        block = re.search(rf"const {name}\b[^=]*=\s*\[(.*?)\](?=\s*(?:as const\s*)?;)",
                          src, re.S)
        assert block, f"{name} not found in {path.name}"
        return set(re.findall(r'"([a-z]{2})"', block.group(1)))

    imp = ui / "pages" / "ImportPage.tsx"
    sel = ui / "components" / "SelectionPanel.tsx"
    assert codes(imp, "SRC_LANGS") == set(cli.SRC_LANGS)
    assert codes(imp, "TGT_LANGS") == set(cli.TGT_LANGS)
    assert codes(sel, "SRC_LANGS") == set(cli.SRC_LANGS)
    assert codes(sel, "TGT_LANGS") == set(cli.TGT_LANGS)


def test_get_project(client):
    body = client.get(f"/api/projects/{NAME}").json()
    assert body["stages"]["mix"] == "done"
    assert body["report"]["segments"] == 5
    assert len(body["manifest"]["segments"]) == 5


# ---------------------------------------------------------------------------
# resuming a run, and editing the options it runs with
# ---------------------------------------------------------------------------

def test_resume_reruns_the_project_with_what_it_recorded(client, outputs, fake):
    """A stopped run had no way back in: the app could create a project and edit
    one, and nothing in it could start the pipeline again. There is no separate
    resume machinery every stage is skipped when its inputs are unchanged, so
    the same `run` job *is* the resume, and its payload is the manifest's own
    source record read back."""
    from dubbing import cli

    r = client.post(f"/api/projects/{NAME}/run", json={})
    assert r.status_code == 202 and r.json()["job"]["kind"] == "run"
    assert wait_until(lambda: fake.calls, 5.0)

    kind, project, payload = fake.calls[0]
    assert (kind, project) == ("run", NAME)
    args = cli.parse_args(ops.full_run_argv(Path(payload["workdir"]), payload["source"]))
    assert args.source == "inputs/whatsapp.mp4"
    assert (args.src, args.tgt) == ("he", "en")
    assert Path(payload["workdir"]) == outputs / NAME


def test_resume_is_refused_while_a_job_is_already_going(outputs):
    hold = threading.Event()
    fake = FakeRunner(hold=hold)
    with TestClient(create_app(outputs, runner=fake, ui_dir="")) as c:
        assert c.post(f"/api/projects/{NAME}/run", json={}).status_code == 202
        r = c.post(f"/api/projects/{NAME}/run", json={})
        assert r.status_code == 409 and envelope_of(r)["code"] == "busy"
        assert "already going" in envelope_of(r)["message"]
        hold.set()


def test_resume_of_an_unknown_project_is_not_found(client):
    r = client.post("/api/projects/nope/run", json={})
    assert r.status_code == 404 and envelope_of(r)["code"] == "not_found"


def test_resume_refuses_a_project_that_does_not_say_what_it_was_made_from(client, outputs):
    m = manifest.load(outputs / NAME)
    m["source"].pop("input")
    manifest.save(outputs / NAME, m)
    r = client.post(f"/api/projects/{NAME}/run", json={})
    assert r.status_code == 400 and "nothing to re-run" in envelope_of(r)["message"]


def test_a_failed_run_is_served_with_the_error_that_stopped_it(outputs):
    """The other half of recovery: the placeholder can only offer "retry from
    fetch" if the server says which stage died and why. Both are on the project."""
    class Boom:
        def run(self, job, emit):
            emit(events.stage_event("fetch", "running", 0.1, "downloading"))
            raise RuntimeError("HTTP Error 410: Gone")

        def cancel(self, job):
            pass

    with TestClient(create_app(outputs, runner=Boom(), ui_dir="")) as c:
        job = c.post(f"/api/projects/{NAME}/run", json={}).json()["job"]
        assert wait_until(lambda: c.get(f"/api/jobs/{job['id']}").json()["job"]["status"]
                          == "failed")
        body = c.get(f"/api/projects/{NAME}").json()
    assert body["stages"]["fetch"] == "failed"
    dead = [j for j in body["jobs"] if j["status"] == "failed"]
    assert dead and dead[-1]["stage"] == "fetch"
    assert "HTTP Error 410" in dead[-1]["error"]
    # …and the project is runnable again the moment nothing is in flight.
    assert dead[-1]["kind"] == "run"


def test_patch_project_writes_the_spelling_the_pipeline_reads_first(client, outputs):
    """Flat keys on `m["source"]`, not `app_opts`: `dubbing.edit._args` resolves
    the pair flat-first, so this is the only spelling that reaches every re-run."""
    from dubbing import edit as edit_mod

    r = client.patch(f"/api/projects/{NAME}",
                     json={"genre": "movie", "register": "dialogue", "context": " a note "})
    assert r.status_code == 200
    source = r.json()["source"]
    assert (source["genre"], source["register"]) == ("movie", "dialogue")
    assert source["context"] == "a note"                 # stripped, not stored raw

    m = manifest.load(outputs / NAME)
    assert m["source"]["genre"] == "movie"
    args = edit_mod._args(m)
    assert (args.genre, args.register, args.context) == ("movie", "dialogue", "a note")


def test_patch_project_beats_the_app_opts_recorded_at_creation(client, outputs, fake):
    """The bug this closes: `app_opts` was the only place `ops` looked, so an
    option changed here was honoured by a per-line re-translate and ignored by a
    resume the same project rendered two ways depending on the button."""
    from dubbing import cli

    m = manifest.load(outputs / NAME)
    m["source"]["app_opts"] = {"genre": "documentary", "register": "narration"}
    manifest.save(outputs / NAME, m)

    assert client.patch(f"/api/projects/{NAME}", json={"genre": "movie"}).status_code == 200
    assert client.post(f"/api/projects/{NAME}/run", json={}).status_code == 202
    assert wait_until(lambda: fake.calls, 5.0)
    args = cli.parse_args(ops.full_run_argv(Path(fake.calls[0][2]["workdir"]),
                                            fake.calls[0][2]["source"]))
    assert args.genre == "movie"
    assert ops.pipeline_overrides(manifest.load(outputs / NAME))["genre"] == "movie"


def test_patch_project_clears_a_context_with_the_empty_string(client, outputs):
    client.patch(f"/api/projects/{NAME}", json={"context": "wrong note"})
    r = client.patch(f"/api/projects/{NAME}", json={"context": ""})
    assert r.status_code == 200 and r.json()["source"]["context"] is None
    assert manifest.load(outputs / NAME)["source"]["context"] is None


def test_patch_project_refuses_what_the_cli_cannot_take(client):
    for body in ({"genre": "banana"}, {"register": "shouting"}, {"src_lang": "ru"},
                 {"source": "other.mp4"}):
        r = client.patch(f"/api/projects/{NAME}", json=body)
        assert r.status_code == 400, body
        assert envelope_of(r)["code"] == "invalid_request"


def test_patch_project_empty_body_is_invalid(client):
    r = client.patch(f"/api/projects/{NAME}", json={})
    assert r.status_code == 400 and "at least one" in envelope_of(r)["message"]


def test_patch_project_is_refused_while_a_job_runs(outputs):
    hold = threading.Event()
    fake = FakeRunner(hold=hold)
    with TestClient(create_app(outputs, runner=fake, ui_dir="")) as c:
        c.post(f"/api/projects/{NAME}/render", json={})
        assert wait_until(lambda: fake.calls, 5.0)
        r = c.patch(f"/api/projects/{NAME}", json={"genre": "movie"})
        assert r.status_code == 409 and envelope_of(r)["code"] == "busy"
        hold.set()
    # …and nothing was written on the way to the refusal.
    assert manifest.load(outputs / NAME)["source"].get("genre") is None


def test_stage_status_folds_in_the_live_queue(outputs):
    """`done`/`pending` is all the manifest can say. Running and failed live in the
    queue, and without them the UI's whole failure treatment is unreachable."""
    from dubbing_app.projects import Projects

    m = manifest.load(outputs / NAME)
    m["stages"].pop("mix"), m["stages"].pop("report")
    assert Projects.stage_status(m)["mix"] == "pending"

    running = {"id": "a", "status": "running", "stage": "mix", "error": None}
    assert Projects.stage_status(m, [running])["mix"] == "running"

    failed = {"id": "b", "status": "failed", "stage": "mix", "error": "boom"}
    assert Projects.stage_status(m, [failed])["mix"] == "failed"
    # A job that succeeded afterwards is the newer answer about that stage.
    later = {"id": "c", "status": "done", "stage": "report", "error": None}
    assert Projects.stage_status(m, [failed, later])["mix"] == "pending"
    # …and a job running now outranks an older failure on the same stage.
    assert Projects.stage_status(m, [failed, running])["mix"] == "running"


def test_a_failed_job_is_reported_on_the_stage_it_died_in(outputs):
    class Boom:
        def run(self, job, emit):
            emit(events.stage_event("mix", "running", 0.2, "muxing"))
            raise RuntimeError("mix died")

        def cancel(self, job):
            pass

    with TestClient(create_app(outputs, runner=Boom(), ui_dir="")) as c:
        job = c.post(f"/api/projects/{NAME}/render", json={}).json()["job"]
        assert wait_until(lambda: c.get(f"/api/jobs/{job['id']}").json()["job"]["status"]
                          == "failed")
        body = c.get(f"/api/projects/{NAME}").json()
    assert body["stages"]["mix"] == "failed"
    assert body["summary"]["stages"]["mix"] == "failed"


def stamp_report(workdir):
    """Re-stamp the fixture's report.json the way `report.run` does."""
    m = manifest.load(workdir)
    path = workdir / "report.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["manifest"] = manifest.content_fingerprint(m)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_a_report_that_predates_the_stamp_is_not_served_as_current(client):
    # It cannot prove it is about this manifest, and it was being shown as if it
    # were: nothing in report.json referred to the segments it counted.
    assert client.get(f"/api/projects/{NAME}").json()["report"]["stale"] is True


def test_a_report_goes_stale_the_moment_an_edit_lands(client, outputs):
    stamp_report(outputs / NAME)
    body = client.get(f"/api/projects/{NAME}").json()
    assert body["report"]["stale"] is False and body["report"]["segments"] == 5

    uid = body["manifest"]["segments"][1]["uid"]
    assert client.patch(f"/api/projects/{NAME}/segments/{uid}",
                        json={"text_en": "A better line"}).status_code == 200
    after = client.get(f"/api/projects/{NAME}").json()["report"]
    # Still served the numbers were true of the render that produced them but
    # no longer claimed to be about the manifest the editor is showing.
    assert after["stale"] is True and after["segments"] == 5


# ------------------------------------------------- what the video is a render of

def stamp_render(workdir):
    """Stamp `m["render"]` the way `mix.run` does, without running ffmpeg."""
    m = manifest.load(workdir)
    m["render"] = {"at": 1_700_000_000,
                   "fp": manifest.content_fingerprint(m),
                   "segments": manifest.segment_digests(m)}
    manifest.save(workdir, m)


def test_a_run_with_no_render_stamp_cannot_claim_to_be_current(client):
    render = client.get(f"/api/projects/{NAME}").json()["render"]
    assert render == {"at": None, "stale": True, "changed": 0}
    # `changed` is 0, not 5: there is no stamp to compare against, so the honest
    # answer is "unknown", and the UI must not say "5 lines changed" off it.


def test_a_fresh_render_is_not_stale_and_counts_nothing_changed(client, outputs):
    stamp_render(outputs / NAME)
    render = client.get(f"/api/projects/{NAME}").json()["render"]
    assert render["stale"] is False and render["changed"] == 0
    assert render["at"] == 1_700_000_000


def test_the_render_counts_the_lines_that_changed_since_it(client, outputs):
    stamp_render(outputs / NAME)
    segs = uids(client)
    for uid in segs[:2]:
        assert client.patch(f"/api/projects/{NAME}/segments/{uid}",
                            json={"text_en": f"new words for {uid}"}).status_code == 200
    render = client.get(f"/api/projects/{NAME}").json()["render"]
    # Two lines were edited, so two lines differ not "every unfinished line", and
    # not the whole list. The count is what the header offers to re-render.
    assert render["stale"] is True and render["changed"] == 2


def test_a_render_stamp_survives_the_manifest_whitelist(outputs):
    # `manifest.save` strips unknown keys from *segments*; `render` is run-level and
    # must come back verbatim, or the stamp would be erased by the next edit.
    stamp_render(outputs / NAME)
    m = manifest.load(outputs / NAME)
    assert set(m["render"]) == {"at", "fp", "segments"}
    manifest.save(outputs / NAME, m)
    assert manifest.load(outputs / NAME)["render"] == m["render"]


def test_renumbering_a_segment_does_not_count_as_changing_it(outputs):
    # `id` is positional: a split renumbers every line after it. Counting `id` would
    # report the whole tail as changed and turn "3 lines changed" into "40".
    m = manifest.load(outputs / NAME)
    before = manifest.segment_digests(m)
    for seg in m["segments"]:
        seg["id"] += 10
    assert manifest.digest_delta(before, manifest.segment_digests(m)) == 0


def test_uids_are_minted_once_and_persisted(client, outputs):
    first = [s["uid"] for s in client.get(f"/api/projects/{NAME}/segments").json()["segments"]]
    second = [s["uid"] for s in client.get(f"/api/projects/{NAME}/segments").json()["segments"]]
    assert first == second and len(set(first)) == 5
    on_disk = manifest.load(outputs / NAME)
    assert [s["uid"] for s in on_disk["segments"]] == first


def test_segments_are_enriched(client):
    segs = client.get(f"/api/projects/{NAME}/segments").json()["segments"]
    dubbed = segs[1]
    # `place.clip` is what plays the fitted clip, not the raw tts clip.
    assert dubbed["media"]["play"].startswith(f"/media/{NAME}/clips/fit_")
    assert dubbed["media"]["tts"] != dubbed["media"]["play"]
    assert dubbed["media"]["source"] == f"/media/{NAME}/source.wav#t=0.340,2.200"
    assert dubbed["media"]["source_window"] == [0.34, 2.2]
    # the QA signal, straight out of clips/<hash>.json
    assert dubbed["verify"]["heard"] == "Anti-Zionists."
    assert dubbed["verify"]["overlap"] == 0.94


def test_enrichment_drops_urls_for_missing_files(client, outputs):
    (outputs / NAME / "clips" / "fit_0000000000000001_1.300.wav").unlink()
    segs = client.get(f"/api/projects/{NAME}/segments").json()["segments"]
    assert segs[1]["media"]["play"] is None
    assert segs[1]["media"]["tts"] is not None


# ---------------------------------------------------------------------------
# the error envelope
# ---------------------------------------------------------------------------

def envelope_of(response):
    body = response.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message"}
    assert body["error"]["code"] in ("invalid_request", "not_found", "busy", "internal_error")
    return body["error"]


def test_not_found_envelope(client):
    r = client.get("/api/projects/does_not_exist")
    assert r.status_code == 404
    assert envelope_of(r)["code"] == "not_found"


def test_invalid_request_envelope_from_validation(client):
    uid = client.get(f"/api/projects/{NAME}/segments").json()["segments"][0]["uid"]
    r = client.patch(f"/api/projects/{NAME}/segments/{uid}", json={"nonsense": 1})
    assert r.status_code == 400
    assert envelope_of(r)["code"] == "invalid_request"


def test_invalid_project_name_envelope(client):
    r = client.get("/api/projects/has%20space/segments")
    assert r.status_code == 400
    assert envelope_of(r)["code"] == "invalid_request"


def test_internal_error_envelope(outputs, fake):
    app = create_app(outputs, runner=fake, ui_dir="")

    @app.get("/boom")
    def boom():
        raise ZeroDivisionError("nope")

    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/boom")
    assert r.status_code == 500
    assert envelope_of(r)["code"] == "internal_error"


def test_unknown_job_is_not_found(client):
    r = client.get("/api/jobs/deadbeef")
    assert r.status_code == 404 and envelope_of(r)["code"] == "not_found"


# ---------------------------------------------------------------------------
# no-model edits
# ---------------------------------------------------------------------------

def uids(client):
    return [s["uid"] for s in client.get(f"/api/projects/{NAME}/segments").json()["segments"]]


def test_patch_text_locks_the_field(client, outputs):
    uid = uids(client)[1]
    r = client.patch(f"/api/projects/{NAME}/segments/{uid}", json={"text_en": "Corrected."})
    assert r.status_code == 200
    assert r.json()["segment"]["text_en"] == "Corrected."
    seg = ops.find(manifest.load(outputs / NAME), uid)
    assert seg["text_en"] == "Corrected."
    assert seg["locked"]["text_en"] is True          # a re-run must not overwrite it


def test_patch_keep_flip(client, outputs):
    uid = uids(client)[1]
    client.patch(f"/api/projects/{NAME}/segments/{uid}", json={"keep": True})
    seg = ops.find(manifest.load(outputs / NAME), uid)
    assert seg["keep"] is True and seg["keep_reason"] == "manual"
    assert seg["locked"]["keep"] is True


def test_patch_speaker_and_langs(client, outputs):
    uid = uids(client)[2]
    client.patch(f"/api/projects/{NAME}/segments/{uid}",
                 json={"speaker": "SPEAKER_01", "tgt_lang": "ru"})
    seg = ops.find(manifest.load(outputs / NAME), uid)
    assert seg["speaker"] == "SPEAKER_01" and seg["tgt_lang"] == "ru"


def test_patch_tts_opts_stores_a_valid_override(client, outputs):
    uid = uids(client)[1]
    r = client.patch(f"/api/projects/{NAME}/segments/{uid}",
                     json={"tts_opts": {"seed": 7, "speed": 1.25}})
    assert r.status_code == 200
    seg = ops.find(manifest.load(outputs / NAME), uid)
    assert seg["tts_opts"] == {"seed": 7, "speed": 1.25}
    # `None` clears one option and leaves the rest.
    client.patch(f"/api/projects/{NAME}/segments/{uid}", json={"tts_opts": {"speed": None}})
    assert ops.find(manifest.load(outputs / NAME), uid)["tts_opts"] == {"seed": 7}


@pytest.mark.parametrize("opts,expected", [
    ({"seed": "banana"}, "seed"),
    ({"nonsense": 1}, "unknown option"),
    ({"speed": 99}, "speed"),
    ({"ref": "../../.ssh/id_rsa"}, "escape"),
    ({"ref_text": "hello"}, "ref"),
    ({"greedy": True, "temperature": 0.7}, "greedy"),
])
def test_patch_tts_opts_refuses_what_the_synthesiser_cannot_use(client, outputs, opts,
                                                                expected):
    """`dubbing.ttsopts` is loud, but `tts.run` is the only thing that reads it —
    minutes into a job. Storing an unusable override made every future run of the
    tts stage raise on a manifest nobody is allowed to hand-edit back."""
    uid = uids(client)[1]
    r = client.patch(f"/api/projects/{NAME}/segments/{uid}", json={"tts_opts": opts})
    assert r.status_code == 400, r.text
    assert expected in envelope_of(r)["message"]
    assert "tts_opts" not in ops.find(manifest.load(outputs / NAME), uid)


def test_patch_bounds_rejects_overlap(client):
    uid = uids(client)[2]
    r = client.patch(f"/api/projects/{NAME}/segments/{uid}", json={"start": 0.5, "end": 11.2})
    assert r.status_code == 400
    assert "overlap" in envelope_of(r)["message"]


def test_patch_bounds_requires_both(client):
    uid = uids(client)[2]
    r = client.patch(f"/api/projects/{NAME}/segments/{uid}", json={"start": 3.0})
    assert r.status_code == 400


def test_patch_empty_body_is_invalid(client):
    r = client.patch(f"/api/projects/{NAME}/segments/{uids(client)[0]}", json={})
    assert r.status_code == 400


def test_patch_unknown_uid(client):
    r = client.patch(f"/api/projects/{NAME}/segments/nope", json={"text": "x"})
    assert r.status_code == 404


def test_split_and_merge(client, outputs):
    before = uids(client)
    r = client.post(f"/api/projects/{NAME}/segments/{before[2]}/split", json={"at": 6.0})
    assert r.status_code == 200
    a, b = r.json()["uids"]
    after = uids(client)
    assert len(after) == 6 and a in after and b in after
    segs = manifest.load(outputs / NAME)["segments"]
    assert [s["id"] for s in segs] == list(range(6))     # renumbered, contiguous
    # the halves are no longer translated or voiced
    assert not (ops.find({"segments": segs}, b).get("text_en") or "")
    assert ops.find({"segments": segs}, b).get("tts") is None

    r = client.post(f"/api/projects/{NAME}/segments/{a}/merge", json={"with": b})
    assert r.status_code == 200
    assert len(uids(client)) == 5


def test_split_outside_bounds(client):
    r = client.post(f"/api/projects/{NAME}/segments/{uids(client)[1]}/split", json={"at": 99.0})
    assert r.status_code == 400


def test_merge_refuses_different_speakers(client):
    ids = uids(client)
    r = client.post(f"/api/projects/{NAME}/segments/{ids[0]}/merge", json={"with": ids[1]})
    assert r.status_code == 400
    assert "speaker" in envelope_of(r)["message"]


def test_merge_refuses_non_adjacent(client):
    ids = uids(client)
    r = client.post(f"/api/projects/{NAME}/segments/{ids[1]}/merge", json={"with": ids[3]})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# segments: adding and removing
# ---------------------------------------------------------------------------

def test_add_segment_claims_a_free_span(client, outputs):
    """The fixture run's segments stop at 20 s and the media is 38 s long, so
    everything after that is uncovered — the state a hallucination-dropped line
    leaves behind."""
    r = client.post(f"/api/projects/{NAME}/segments",
                    json={"start": 22.0, "end": 24.5, "text": "השורה שנשמטה"})
    assert r.status_code == 201
    uid = r.json()["uid"]
    segs = manifest.load(outputs / NAME)["segments"]
    added = ops.find({"segments": segs}, uid)
    assert (added["start"], added["end"]) == (22.0, 24.5)
    assert added["text"] == "השורה שנשמטה" and added["keep"] is False
    # Nothing generated for it, the user's words locked, and the user's word that
    # this span is a dub — so a failure downstream stays visibly unfinished
    # instead of answering with the original audio it was added to replace.
    assert "text_en" not in added and "tts" not in added
    assert added["locked"] == {"text": True} and added["passthrough"] is False
    assert [s["id"] for s in segs] == list(range(6))
    # It is a real segment to every other route from now on.
    assert uid in uids(client)
    # And the run knows there is work to do again: translate onward is reopened,
    # while the segmentation is not (re-running it would rebuild the list from
    # the words, which this segment is not in).
    stages = client.get(f"/api/projects/{NAME}").json()["stages"]
    assert stages["segments"] == "done" and stages["translate"] == "pending"
    assert stages["mix"] == "pending"


def test_add_segment_inherits_the_nearest_speaker_and_takes_an_explicit_one(client):
    near = client.post(f"/api/projects/{NAME}/segments",
                       json={"start": 21.0, "end": 23.0, "text": "aa"}).json()["segment"]
    assert near["speaker"] == "SPEAKER_01"          # the segment ending at 20.0
    named = client.post(f"/api/projects/{NAME}/segments",
                        json={"start": 25.0, "end": 27.0, "text": "bb",
                              "speaker": "SPEAKER_00"}).json()["segment"]
    assert named["speaker"] == "SPEAKER_00"


@pytest.mark.parametrize("body,because", [
    ({"start": 10.0, "end": 22.0, "text": "aa"}, "overlaps"),
    ({"start": 30.0, "end": 45.0, "text": "aa"}, "past the end of the media"),
    ({"start": 22.0, "end": 22.3, "text": "aa"}, "cannot be synthesized reliably"),
    ({"start": 22.0, "end": 24.0, "text": "   "}, "needs its text"),
    ({"start": -2.0, "end": 2.0, "text": "aa"}, "negative"),
])
def test_add_segment_refuses_what_the_timeline_could_not_hold(client, body, because):
    r = client.post(f"/api/projects/{NAME}/segments", json=body)
    assert r.status_code == 400
    assert because in envelope_of(r)["message"]
    assert len(uids(client)) == 5            # and nothing was written


def test_remove_segment_takes_it_out_of_the_manifest(client, outputs):
    before = uids(client)
    r = client.delete(f"/api/projects/{NAME}/segments/{before[2]}")
    assert r.status_code == 200
    # The deleted record comes back, so a client can offer an undo without
    # having kept its own copy.
    assert r.json()["removed"]["text"] == "ועם כל הגיא."
    segs = manifest.load(outputs / NAME)["segments"]
    assert [s["uid"] for s in segs] == [u for u in before if u != before[2]]
    assert [s["id"] for s in segs] == list(range(4))
    # The survivors keep every bit of their own work: nothing they hold was made
    # from the segment that went.
    assert all(s.get("text_en") for s in segs)
    # What did change is the placement — the span is free now — and everything
    # after it. Translate and tts are untouched.
    stages = client.get(f"/api/projects/{NAME}").json()["stages"]
    assert stages["translate"] == "done" and stages["tts"] == "done"
    assert stages["timeline"] == "pending" and stages["mix"] == "pending"


def test_remove_unknown_uid(client):
    r = client.delete(f"/api/projects/{NAME}/segments/nope")
    assert r.status_code == 404 and envelope_of(r)["code"] == "not_found"


def test_add_and_remove_are_structural_and_wait_for_no_running_job(outputs):
    """Both renumber `seg["id"]`, which a running stage is iterating over — the
    same reason `split` is refused rather than queued."""
    hold = threading.Event()
    fake = FakeRunner(hold=hold)
    with TestClient(create_app(outputs, runner=fake, ui_dir="")) as c:
        job = c.post(f"/api/projects/{NAME}/render", json={}).json()["job"]
        assert wait_until(lambda: c.get(f"/api/jobs/{job['id']}").json()["job"]["status"]
                          == "running")
        uid = c.get(f"/api/projects/{NAME}/segments").json()["segments"][1]["uid"]
        add = c.post(f"/api/projects/{NAME}/segments",
                     json={"start": 22.0, "end": 24.0, "text": "aa"})
        assert add.status_code == 409 and envelope_of(add)["code"] == "busy"
        gone = c.delete(f"/api/projects/{NAME}/segments/{uid}")
        assert gone.status_code == 409 and envelope_of(gone)["code"] == "busy"
        hold.set()


def test_add_and_remove_announce_themselves_on_the_stream(client, monkeypatch):
    """A client rebuilds its list from `GET /segments` when a segment frame
    arrives, and a structural edit is the one it most needs to hear about."""
    seen: list[dict] = []
    bus = client.app.state.bus
    monkeypatch.setattr(bus, "publish", lambda project, event: seen.append(event))
    uid = client.post(f"/api/projects/{NAME}/segments",
                      json={"start": 22.0, "end": 24.0, "text": "aa"}).json()["uid"]
    assert client.delete(f"/api/projects/{NAME}/segments/{uid}").status_code == 200
    assert [(e["type"], e["uid"], e["field"]) for e in seen] == [
        ("segment", uid, "add"), ("segment", uid, "remove")]


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------

def wait_until(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_retranslate_enqueues_a_job(client, fake):
    uid = uids(client)[1]
    r = client.post(f"/api/projects/{NAME}/retranslate", json={"uids": [uid]})
    assert r.status_code == 202
    job = r.json()["job"]
    assert job["kind"] == "retranslate" and job["project"] == NAME
    assert wait_until(lambda: client.get(f"/api/jobs/{job['id']}").json()["job"]["status"]
                      == "done")
    assert fake.calls[0][0] == "retranslate"
    assert fake.calls[0][2]["uids"] == [uid]


def test_retranslate_rejects_unknown_uid(client):
    r = client.post(f"/api/projects/{NAME}/retranslate", json={"uids": ["nope"]})
    assert r.status_code == 404


def test_retranslate_rejects_empty_uids(client):
    r = client.post(f"/api/projects/{NAME}/retranslate", json={"uids": []})
    assert r.status_code == 400


def test_render_enqueues_a_job(client, fake):
    r = client.post(f"/api/projects/{NAME}/render", json={})
    assert r.status_code == 202
    assert wait_until(lambda: any(c[0] == "render" for c in fake.calls))


def test_one_job_at_a_time(outputs):
    """The hard constraint: never two model jobs in flight, however many arrive."""
    bus = events.EventBus()
    fake = FakeRunner()
    queue = JobQueue(fake, bus)
    queue.start()
    try:
        jobs = [queue.submit("retranslate", NAME, {"uids": [str(i)]}) for i in range(8)]
        assert wait_until(lambda: all(j.status in ("done", "failed") for j in jobs))
    finally:
        queue.stop()
    assert fake.max_concurrent == 1
    assert [j.status for j in jobs] == ["done"] * 8
    # and in the order submitted
    assert [c[2]["uids"][0] for c in fake.calls] == [str(i) for i in range(8)]


def test_second_job_queues_behind_the_first(outputs):
    hold = threading.Event()
    fake = FakeRunner(hold=hold)
    queue = JobQueue(fake, events.EventBus())
    queue.start()
    try:
        first = queue.submit("render", NAME, {})
        second = queue.submit("render", NAME, {})
        assert wait_until(lambda: first.status == "running")
        time.sleep(0.05)
        assert second.status == "queued"        # not started, not refused
        hold.set()
        assert wait_until(lambda: second.status == "done")
    finally:
        queue.stop()


def test_cancel_queued_job(outputs):
    hold = threading.Event()
    fake = FakeRunner(hold=hold)
    queue = JobQueue(fake, events.EventBus())
    queue.start()
    try:
        first = queue.submit("render", NAME, {})
        second = queue.submit("render", NAME, {})
        assert wait_until(lambda: first.status == "running")
        queue.cancel(second.id)
        assert second.status == "cancelled"
        hold.set()
        assert wait_until(lambda: first.status == "done")
    finally:
        queue.stop()
    assert [c[0] for c in fake.calls] == ["render"]      # the cancelled one never ran


def test_cancel_running_job_via_http(client, outputs):
    hold = threading.Event()
    fake = FakeRunner(hold=hold)
    with TestClient(create_app(outputs, runner=fake)) as c:
        job = c.post(f"/api/projects/{NAME}/render", json={}).json()["job"]
        assert wait_until(lambda: c.get(f"/api/jobs/{job['id']}").json()["job"]["status"]
                          == "running")
        r = c.delete(f"/api/jobs/{job['id']}")
        assert r.status_code == 200
        assert wait_until(lambda: c.get(f"/api/jobs/{job['id']}").json()["job"]["status"]
                          == "cancelled")
    assert fake.cancelled == [job["id"]]


def test_cancel_finished_job_is_invalid(client):
    job = client.post(f"/api/projects/{NAME}/render", json={}).json()["job"]
    assert wait_until(lambda: client.get(f"/api/jobs/{job['id']}").json()["job"]["status"]
                      == "done")
    r = client.delete(f"/api/jobs/{job['id']}")
    assert r.status_code == 400 and envelope_of(r)["code"] == "invalid_request"


# --------------------------------------------- one gesture, one thing to cancel

def test_a_batch_id_travels_from_the_request_onto_both_jobs(client):
    some = uids(client)
    a = client.post(f"/api/projects/{NAME}/retranslate",
                    json={"uids": some[:2], "batch": "g1"}).json()["job"]
    b = client.post(f"/api/projects/{NAME}/resynthesize",
                    json={"uids": some[:2], "batch": "g1"}).json()["job"]
    assert a["batch"] == b["batch"] == "g1"
    assert a["id"] != b["id"]


def test_a_lone_job_has_no_batch(client):
    job = client.post(f"/api/projects/{NAME}/resynthesize",
                      json={"uids": uids(client)[:1]}).json()["job"]
    assert job["batch"] is None


def test_cancelling_a_batch_stops_the_voice_job_the_translate_was_feeding(outputs):
    """The audit's disaster, made impossible.

    Cancel the running re-translate and the queued re-voice used to run anyway, on
    lines whose translation had just been abandoned 27 `tts_failed` keeps from one
    click. `?batch=1` cancels the decision, not the step.
    """
    hold = threading.Event()
    fake = FakeRunner(hold=hold)
    with TestClient(create_app(outputs, runner=fake)) as c:
        some = uids(c)
        first = c.post(f"/api/projects/{NAME}/retranslate",
                       json={"uids": some, "batch": "g1"}).json()["job"]
        second = c.post(f"/api/projects/{NAME}/resynthesize",
                        json={"uids": some, "batch": "g1"}).json()["job"]
        assert wait_until(lambda: c.get(f"/api/jobs/{first['id']}").json()["job"]["status"]
                          == "running")
        r = c.delete(f"/api/jobs/{first['id']}?batch=1")
        assert r.status_code == 200
        assert {j["id"] for j in r.json()["cancelled"]} == {first["id"], second["id"]}
        hold.set()
        assert wait_until(lambda: c.get(f"/api/jobs/{second['id']}").json()["job"]["status"]
                          == "cancelled")
        assert wait_until(lambda: c.get(f"/api/jobs/{first['id']}").json()["job"]["status"]
                          == "cancelled")
    # The queued voice job never reached the runner at all.
    assert [call[0] for call in fake.calls] == ["retranslate"]


def test_cancelling_one_job_leaves_its_batch_mate_alone(outputs):
    # The other half of the choice the UI offers: "just this one" must still mean
    # just this one, or the dialog is lying.
    hold = threading.Event()
    fake = FakeRunner(hold=hold)
    with TestClient(create_app(outputs, runner=fake)) as c:
        some = uids(c)
        first = c.post(f"/api/projects/{NAME}/retranslate",
                       json={"uids": some, "batch": "g1"}).json()["job"]
        second = c.post(f"/api/projects/{NAME}/resynthesize",
                        json={"uids": some, "batch": "g1"}).json()["job"]
        assert wait_until(lambda: c.get(f"/api/jobs/{first['id']}").json()["job"]["status"]
                          == "running")
        assert c.delete(f"/api/jobs/{first['id']}").json()["cancelled"] is None
        hold.set()
        assert wait_until(lambda: c.get(f"/api/jobs/{second['id']}").json()["job"]["status"]
                          == "done")
    assert [call[0] for call in fake.calls] == ["retranslate", "resynthesize"]


def test_batch_cancel_on_a_job_with_no_batch_cancels_only_it(outputs):
    hold = threading.Event()
    fake = FakeRunner(hold=hold)
    with TestClient(create_app(outputs, runner=fake)) as c:
        first = c.post(f"/api/projects/{NAME}/render", json={}).json()["job"]
        second = c.post(f"/api/projects/{NAME}/render", json={}).json()["job"]
        assert wait_until(lambda: c.get(f"/api/jobs/{first['id']}").json()["job"]["status"]
                          == "running")
        r = c.delete(f"/api/jobs/{first['id']}?batch=1")
        assert [j["id"] for j in r.json()["cancelled"]] == [first["id"]]
        hold.set()
        assert wait_until(lambda: c.get(f"/api/jobs/{second['id']}").json()["job"]["status"]
                          == "done")


def test_a_job_frame_says_which_lines_and_which_batch(client):
    # The UI derives its busy rows from job frames, so a frame that omitted `uids`
    # would leave the rows unmarked until a refetch happened to land.
    some = uids(client)[:2]
    job = client.post(f"/api/projects/{NAME}/retranslate",
                      json={"uids": some, "batch": "g7"}).json()["job"]
    assert job["uids"] == some and job["batch"] == "g7"
    listed = client.get(f"/api/projects/{NAME}").json()["jobs"]
    assert [j for j in listed if j["id"] == job["id"]][0]["uids"] == some


def test_failed_job_records_the_error(outputs):
    class Boom:
        def run(self, job, emit):
            raise RuntimeError("model exploded")

        def cancel(self, job):
            pass

    queue = JobQueue(Boom(), events.EventBus())
    queue.start()
    try:
        job = queue.submit("render", NAME, {})
        assert wait_until(lambda: job.status == "failed")
    finally:
        queue.stop()
    assert "model exploded" in job.error


def test_no_model_edits_stay_responsive_while_a_job_runs(outputs):
    hold = threading.Event()
    fake = FakeRunner(hold=hold)
    with TestClient(create_app(outputs, runner=fake)) as c:
        job = c.post(f"/api/projects/{NAME}/render", json={}).json()["job"]
        assert wait_until(lambda: c.get(f"/api/jobs/{job['id']}").json()["job"]["status"]
                          == "running")
        uid = c.get(f"/api/projects/{NAME}/segments").json()["segments"][1]["uid"]
        started = time.time()
        r = c.patch(f"/api/projects/{NAME}/segments/{uid}", json={"keep": True})
        assert r.status_code == 200
        assert time.time() - started < 1.0
        # ... but a structural edit under a running job is refused, not queued.
        r = c.post(f"/api/projects/{NAME}/segments/{uid}/split", json={"at": 1.0})
        assert r.status_code == 409 and envelope_of(r)["code"] == "busy"
        hold.set()


def test_a_queued_job_does_not_block_a_structural_edit(outputs, fake):
    """Only a *running* job is renumbered under. A queued one has not read the
    manifest yet the child loads it when it starts, not when it is enqueued so
    refusing here would block edits behind a queue that may be minutes deep."""
    client = TestClient(create_app(outputs, runner=fake, ui_dir=""))   # no lifespan:
    uid = uids(client)[2]                                             # the worker
    assert client.post(f"/api/projects/{NAME}/render", json={}).status_code == 202
    assert client.get("/api/jobs").json()["jobs"][0]["status"] == "queued"
    r = client.post(f"/api/projects/{NAME}/segments/{uid}/split", json={"at": 6.0})
    assert r.status_code == 200
    assert fake.calls == []


# ---------------------------------------------------------------------------
# media: range serving and path traversal
# ---------------------------------------------------------------------------

def test_media_serves_whole_file(client, outputs):
    r = client.get(f"/media/{NAME}/preview.mp4")
    assert r.status_code == 200
    assert r.headers["accept-ranges"] == "bytes"
    assert r.headers["content-type"] == "video/mp4"
    assert len(r.content) == (outputs / NAME / "preview.mp4").stat().st_size


def test_media_range_request(client, outputs):
    whole = (outputs / NAME / "preview.mp4").read_bytes()
    r = client.get(f"/media/{NAME}/preview.mp4", headers={"Range": "bytes=10-19"})
    assert r.status_code == 206
    assert r.headers["content-range"] == f"bytes 10-19/{len(whole)}"
    assert r.headers["content-length"] == "10"
    assert r.content == whole[10:20]


def test_media_open_ended_range(client, outputs):
    whole = (outputs / NAME / "preview.mp4").read_bytes()
    r = client.get(f"/media/{NAME}/preview.mp4", headers={"Range": "bytes=100-"})
    assert r.status_code == 206 and r.content == whole[100:]


def test_media_suffix_range(client, outputs):
    whole = (outputs / NAME / "preview.mp4").read_bytes()
    r = client.get(f"/media/{NAME}/preview.mp4", headers={"Range": "bytes=-16"})
    assert r.status_code == 206 and r.content == whole[-16:]


def test_media_unsatisfiable_range(client, outputs):
    size = (outputs / NAME / "preview.mp4").stat().st_size
    r = client.get(f"/media/{NAME}/preview.mp4", headers={"Range": f"bytes={size + 10}-"})
    assert r.status_code == 416
    assert r.headers["content-range"] == f"bytes */{size}"
    assert envelope_of(r)["code"] == "invalid_request"


def test_media_range_starting_at_the_end_is_416(client, outputs):
    """The boundary an `<audio>` element hits when it seeks to the very end: the
    first byte offset equals the size, so there is nothing to send. 200 with an
    empty body would look like a truncated file to the player."""
    size = (outputs / NAME / "preview.mp4").stat().st_size
    r = client.get(f"/media/{NAME}/preview.mp4", headers={"Range": f"bytes={size}-"})
    assert r.status_code == 416
    assert r.headers["content-range"] == f"bytes */{size}"


def test_media_multi_range_is_answered_whole_not_500(client, outputs):
    """More than one range is allowed to be refused, but never with a stack trace.
    RFC 9110 lets a server answer the whole thing, which keeps the client playing."""
    whole = (outputs / NAME / "preview.mp4").read_bytes()
    r = client.get(f"/media/{NAME}/preview.mp4", headers={"Range": "bytes=0-9, 20-29"})
    assert r.status_code == 200 and r.content == whole


def test_media_head(client, outputs):
    size = (outputs / NAME / "preview.mp4").stat().st_size
    r = client.head(f"/media/{NAME}/preview.mp4")
    assert r.status_code == 200 and r.headers["content-length"] == str(size)


def test_media_head_honours_a_range(client, outputs):
    size = (outputs / NAME / "preview.mp4").stat().st_size
    r = client.head(f"/media/{NAME}/preview.mp4", headers={"Range": "bytes=10-19"})
    assert r.status_code == 206 and not r.content
    assert r.headers["content-range"] == f"bytes 10-19/{size}"
    assert r.headers["content-length"] == "10"


def test_media_source_url_keeps_its_time_fragment_unescaped(client):
    """`source.wav#t=a,b` is the A/B preview: the fragment is the browser's, so
    percent-encoding the `#` would make the player fetch a file that is not there."""
    seg = client.get(f"/api/projects/{NAME}/segments").json()["segments"][1]
    assert seg["media"]["source"] == f"/media/{NAME}/source.wav#t=0.340,2.200"
    assert "%23" not in seg["media"]["source"]


def test_media_garbage_range_serves_whole_file(client, outputs):
    size = (outputs / NAME / "preview.mp4").stat().st_size
    r = client.get(f"/media/{NAME}/preview.mp4", headers={"Range": "furlongs=1-2"})
    assert r.status_code == 200 and len(r.content) == size


@pytest.mark.parametrize("path", [
    "../../etc/passwd",
    "clips/../../../../etc/hosts",
    "/etc/passwd",
    # percent-encoded, so it survives the client's own path normalisation and
    # arrives at the route as a genuine `..` to be refused
    "..%2F..%2Fetc%2Fpasswd",
    "clips%2F..%2F..%2Fmanifest.json",
])
def test_media_refuses_traversal(client, path):
    r = client.get(f"/media/{NAME}/{path}")
    assert r.status_code in (400, 404)
    assert envelope_of(r)["code"] in ("invalid_request", "not_found")


def test_media_refuses_symlink_out_of_the_run_dir(client, outputs, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("HF_TOKEN=hunter2", encoding="utf-8")
    (outputs / NAME / "escape.wav").symlink_to(secret)
    r = client.get(f"/media/{NAME}/escape.wav")
    assert r.status_code == 404
    assert b"hunter2" not in r.content


def test_media_unknown_project(client):
    r = client.get("/media/nope/preview.mp4")
    assert r.status_code == 404


def test_media_resolve_unit(outputs):
    workdir = outputs / NAME
    assert media.resolve(workdir, "preview.mp4").name == "preview.mp4"
    for bad in ("../../etc/passwd", "/etc/passwd", ""):
        with pytest.raises(Exception):
            media.resolve(workdir, bad)


# ---------------------------------------------------------------------------
# NDJSON event stream
# ---------------------------------------------------------------------------

PRELUDE = 1 + 9          # the "watching" log, then one frame per pipeline stage


class Frames:
    """One reader over a live NDJSON body `iter_lines()` is single-use."""

    def __init__(self, response):
        self.lines = response.iter_lines()
        self.seen: list[dict] = []

    def _next(self) -> dict:
        for line in self.lines:
            if line.strip():
                frame = json.loads(line)      # one JSON object per line, by contract
                self.seen.append(frame)
                return frame
        raise AssertionError("stream ended")

    def take(self, count: int) -> list[dict]:
        return [self._next() for _ in range(count)]

    def until(self, predicate, limit: int = 200) -> dict | None:
        for _ in range(limit):
            frame = self._next()
            if predicate(frame):
                return frame
        return None


def test_events_stream_is_ndjson(live):
    with httpx.Client(base_url=live, timeout=20.0) as c:
        with c.stream("GET", f"/api/projects/{NAME}/events") as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("application/x-ndjson")
            frames = Frames(r).take(3)
    assert frames[0]["type"] == "log" and frames[0]["level"] == "info"
    assert all(isinstance(f, dict) and "type" in f for f in frames)
    assert {f["type"] for f in frames} <= {"log", "stage", "job", "segment", "heartbeat"}


def test_events_prelude_reports_stage_state(live):
    with httpx.Client(base_url=live, timeout=20.0) as c:
        with c.stream("GET", f"/api/projects/{NAME}/events") as r:
            frames = Frames(r).take(PRELUDE)
    stages = {f["stage"]: f for f in frames if f["type"] == "stage"}
    assert len(stages) == 9
    assert stages["mix"]["status"] == "done" and stages["mix"]["progress"] == 1.0
    # A snapshot of nine stages at once, not a progression: marked so a client
    # that follows "the last stage frame seen" does not read the ninth as the
    # stage this run is in.
    assert all(f["replay"] is True for f in stages.values())


def test_events_prelude_does_not_resurrect_a_finished_failure(outputs, monkeypatch):
    """The UI reopens this stream on every navigation and every wake from sleep.
    Replaying a terminal job puts the error bar the user dismissed an hour ago back
    on the screen, for a job that is long over."""
    monkeypatch.setattr(events, "HEARTBEAT_SECONDS", 0.05)

    class Boom:
        def run(self, job, emit):
            raise RuntimeError("mix died")

        def cancel(self, job):
            pass

    with live_server(create_app(outputs, runner=Boom(), ui_dir="")) as base:
        with httpx.Client(base_url=base, timeout=20.0) as c:
            job = c.post(f"/api/projects/{NAME}/render", json={}).json()["job"]
            assert wait_until(lambda: c.get(f"/api/jobs/{job['id']}").json()["job"]["status"]
                              == "failed")
            with c.stream("GET", f"/api/projects/{NAME}/events") as r:
                frames = Frames(r)
                assert frames.until(lambda f: f["type"] == "heartbeat") is not None
    assert [f for f in frames.seen if f["type"] == "job"] == []


def test_events_prelude_still_replays_work_in_flight(outputs, monkeypatch):
    """The other half: a job that has not finished is exactly what a reconnecting
    client has to be told about, and it is marked as a replay, not as news."""
    monkeypatch.setattr(events, "HEARTBEAT_SECONDS", 0.05)
    hold = threading.Event()
    with live_server(create_app(outputs, runner=FakeRunner(hold=hold), ui_dir="")) as base:
        with httpx.Client(base_url=base, timeout=20.0) as c:
            job = c.post(f"/api/projects/{NAME}/render", json={}).json()["job"]
            assert wait_until(lambda: c.get(f"/api/jobs/{job['id']}").json()["job"]["status"]
                              == "running")
            with c.stream("GET", f"/api/projects/{NAME}/events") as r:
                frames = Frames(r)
                replayed = frames.until(lambda f: f["type"] == "job")
            hold.set()
    assert replayed["id"] == job["id"] and replayed["replay"] is True


def test_events_carry_job_and_stage_progress(live):
    with httpx.Client(base_url=live, timeout=20.0) as c:
        with c.stream("GET", f"/api/projects/{NAME}/events") as r:
            frames = Frames(r)
            prelude = frames.take(PRELUDE)                # drain the prelude
            c.post(f"/api/projects/{NAME}/render", json={})
            done = frames.until(lambda f: f["type"] == "job" and f["status"] == "done")
    assert done is not None
    live_frames = frames.seen[len(prelude):]
    assert {f["type"] for f in live_frames} >= {"job", "stage"}
    stage = next(f for f in live_frames if f["type"] == "stage" and f.get("message") == "1/2")
    assert stage["progress"] == 0.5 and stage["job"]
    assert [f["status"] for f in live_frames
            if f["type"] == "job"] == ["queued", "running", "done"]


def test_events_deliver_failure_as_a_frame_not_a_status(outputs):
    """A run that dies mid-stream cannot use the status line: it is already sent."""

    class Boom:
        def run(self, job, emit):
            emit(events.log_event("about to fail", "error"))
            raise RuntimeError("mix died")

        def cancel(self, job):
            pass

    with live_server(create_app(outputs, runner=Boom())) as base:
        with httpx.Client(base_url=base, timeout=20.0) as c:
            with c.stream("GET", f"/api/projects/{NAME}/events") as r:
                assert r.status_code == 200      # the stream itself stays 200 …
                frames = Frames(r)
                frames.take(PRELUDE)
                c.post(f"/api/projects/{NAME}/render", json={})
                failure = frames.until(
                    lambda f: f["type"] == "job" and f["status"] == "failed")
    assert failure is not None and "mix died" in failure["error"]  # … the failure is a frame


def test_events_heartbeat_over_the_wire(outputs, fake, monkeypatch):
    monkeypatch.setattr(events, "HEARTBEAT_SECONDS", 0.05)
    with live_server(create_app(outputs, runner=fake)) as base:
        with httpx.Client(base_url=base, timeout=20.0) as c:
            with c.stream("GET", f"/api/projects/{NAME}/events") as r:
                frames = Frames(r)
                frames.take(PRELUDE)
                beat = frames.until(lambda f: f["type"] == "heartbeat")
    assert beat is not None


def test_events_subscription_is_released_on_every_disconnect(outputs, fake):
    """The UI reconnects the stream on every navigation and after every sleep. A
    subscription left behind per reconnect is an unbounded fan-out on a bus the job
    worker publishes to from another thread."""
    app = create_app(outputs, runner=fake, ui_dir="")
    with live_server(app) as base:
        for _ in range(3):
            with httpx.Client(base_url=base, timeout=20.0) as c:
                with c.stream("GET", f"/api/projects/{NAME}/events") as r:
                    assert next(r.iter_lines())
            assert wait_until(lambda: app.state.bus.subscriber_count(NAME) == 0, 5.0)


def test_events_unknown_project(client):
    r = client.get("/api/projects/nope/events")
    assert r.status_code == 404 and envelope_of(r)["code"] == "not_found"


def test_stream_heartbeat():
    async def run():
        bus = events.EventBus()
        sub = bus.subscribe(NAME)
        frames = []
        gen = events.stream(sub, prelude=[], heartbeat=0.02)
        async for chunk in gen:
            frames.append(json.loads(chunk))
            if len(frames) >= 2:
                break
        await gen.aclose()
        return frames

    frames = asyncio.run(run())
    assert [f["type"] for f in frames] == ["heartbeat", "heartbeat"]
    assert all("t" in f for f in frames)


def test_stream_encodes_one_object_per_line():
    payload = events.encode({"type": "log", "message": "עברית\nwith newline"})
    assert payload.endswith(b"\n") and payload.count(b"\n") == 1
    assert json.loads(payload)["message"] == "עברית\nwith newline"


def test_bus_publish_is_thread_safe():
    async def run():
        bus = events.EventBus()
        sub = bus.subscribe(NAME)
        threading.Thread(target=lambda: bus.publish(NAME, events.log_event("hi"))).start()
        event = await asyncio.wait_for(sub.queue.get(), timeout=2.0)
        return event

    assert asyncio.run(run())["message"] == "hi"


# ---------------------------------------------------------------------------
# progress derived from the pipeline's own stderr
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("line,expected", [
    ("[tts]", {"stage": "tts", "status": "running", "progress": 0.0}),
    ("[mix] up to date", {"stage": "mix", "status": "done", "progress": 1.0}),
    ("[translate] done in 91s", {"stage": "translate", "status": "done", "progress": 1.0}),
    ("  tts: 30/60", {"stage": "tts", "status": "running", "progress": 0.5}),
    ("  translate: 3/4", {"stage": "translate", "status": "running", "progress": 0.75}),
])
def test_stderr_progress_parsing(line, expected):
    event = runner_mod.parse_stderr(line)
    assert event is not None
    for key, value in expected.items():
        assert event[key] == value


@pytest.mark.parametrize("line", [
    "  tts: seg 3 unusable → keep original",
    "Preview: /x/preview.mp4",
    "  translate: loading models/gemma (mlx 4-bit)",
    "",
])
def test_stderr_non_progress_lines_pass_through(line):
    assert runner_mod.parse_stderr(line) is None


# ---------------------------------------------------------------------------
# the real subprocess boundary (no models: a stub child, or a rejected spec)
# ---------------------------------------------------------------------------

STUB_WORKER = '''
import json, sys, time
spec = json.loads(sys.stdin.readline())
print(json.dumps({"type": "stage", "stage": "tts", "status": "running",
                  "progress": 0.5}), flush=True)
print("[timeline]", file=sys.stderr, flush=True)
print("  tts: 5/10", file=sys.stderr, flush=True)
if spec["payload"].get("sleep"):
    time.sleep(60)
if spec["payload"].get("fail"):
    print(json.dumps({"type": "result", "ok": False, "error": "stub failed"}), flush=True)
    raise SystemExit(1)
print(json.dumps({"type": "result", "ok": True, "data": {"stub": True}}), flush=True)
'''


@pytest.fixture()
def stub_runner(tmp_path):
    (tmp_path / "stub_worker.py").write_text(STUB_WORKER, encoding="utf-8")
    return runner_mod.SubprocessRunner(cwd=tmp_path, module="stub_worker")


def test_subprocess_runner_streams_both_channels(stub_runner):
    from dubbing_app.jobs import Job

    seen = []
    job = Job(id="j1", kind="render", project=NAME, payload={"workdir": "/tmp"})
    data = stub_runner.run(job, seen.append)
    assert data == {"stub": True}
    stages = [f for f in seen if f["type"] == "stage"]
    # stdout NDJSON …
    assert any(f["stage"] == "tts" and f["progress"] == 0.5 for f in stages)
    # … and stderr parsed into the same event shape
    assert any(f["stage"] == "timeline" and f["status"] == "running" for f in stages)
    assert any(f["stage"] == "tts" and f["progress"] == 0.5 and f.get("message") == "5/10"
               for f in stages)


def test_subprocess_runner_reports_child_failure(stub_runner):
    from dubbing_app.jobs import Job

    job = Job(id="j2", kind="render", project=NAME, payload={"workdir": "/tmp", "fail": True})
    with pytest.raises(RuntimeError, match="stub failed"):
        stub_runner.run(job, lambda e: None)


def test_subprocess_runner_cancel_kills_the_child(stub_runner):
    from dubbing_app.jobs import Job, JobCancelled

    job = Job(id="j3", kind="render", project=NAME, payload={"workdir": "/tmp", "sleep": True})
    result: list = []

    def go():
        try:
            stub_runner.run(job, lambda e: None)
        except BaseException as exc:                 # noqa: BLE001 recorded, not swallowed
            result.append(exc)

    thread = threading.Thread(target=go, daemon=True)
    thread.start()
    assert wait_until(lambda: stub_runner._procs.get(job.id) is not None, 10.0)
    job.cancelling = True
    stub_runner.cancel(job)
    thread.join(15.0)
    assert not thread.is_alive()
    assert result and isinstance(result[0], JobCancelled)


def test_worker_rejects_an_unknown_kind(outputs):
    """The real child, end to end spec on stdin, error frame out, non-zero exit."""
    from dubbing_app.jobs import Job

    real = runner_mod.SubprocessRunner(python=sys.executable,
                                       cwd=Path(__file__).resolve().parents[1])
    job = Job(id="j4", kind="render", project=NAME, payload={"workdir": str(outputs / NAME)})
    real.spec = lambda j: {"kind": "not_a_kind", "workdir": j.payload["workdir"],
                           "payload": {}}
    with pytest.raises(RuntimeError, match="not_a_kind"):
        real.run(job, lambda e: None)


# ------------------------------------------- edits made while a job is running

def edit_on_disk(workdir, uid, **fields):
    """Exactly what `PATCH /segments/{uid}` does: load, one ops setter, save."""
    disk = manifest.load(workdir)
    if "locked" in fields:
        ops.set_locked(disk, uid, fields.pop("locked"))
    if fields:
        ops.set_text(disk, uid, **fields)
    manifest.save(workdir, disk)


def test_a_job_does_not_overwrite_edits_made_while_it_ran(outputs, monkeypatch):
    """The job child holds its own copy of the manifest for minutes while the
    server keeps answering PATCHes against the same file the contract says
    no-model edits never wait for a job. Saving that copy at the end used to throw
    every one of those edits away."""
    from dubbing_app import worker

    workdir = outputs / NAME
    segs = manifest.load(workdir)["segments"]
    target, edited = segs[1]["uid"], segs[3]["uid"]

    def fake_retranslate(m, wd, uids, *, progress=None):
        edit_on_disk(wd, edited, text_en="hand corrected")     # the user, mid-job
        for uid in uids:
            ops.find(m, uid)["text_en"] = "machine"
        return {uid: "machine" for uid in uids}

    monkeypatch.setattr(ops, "retranslate", fake_retranslate)
    worker.execute({"kind": "retranslate", "workdir": str(workdir),
                    "payload": {"uids": [target]}})

    saved = manifest.load(workdir)
    assert ops.find(saved, edited)["text_en"] == "hand corrected"
    assert ops.find(saved, edited)["locked"] == {"text_en": True}
    assert ops.find(saved, target)["text_en"] == "machine"     # the job still lands


def test_a_job_replays_a_lock_release_made_while_it_ran(outputs, monkeypatch):
    """`PATCH {"locked": {}}` deletes a key rather than changing one, so a merge
    that only looked at what is *present* on disk would silently re-lock it."""
    from dubbing_app import worker

    workdir = outputs / NAME
    edited = manifest.load(workdir)["segments"][3]["uid"]
    edit_on_disk(workdir, edited, text_en="hand corrected")
    assert ops.find(manifest.load(workdir), edited)["locked"] == {"text_en": True}

    def fake_resynthesize(m, wd, uids, *, progress=None):
        edit_on_disk(wd, edited, locked={})                    # hand it back
        return {}

    monkeypatch.setattr(ops, "resynthesize", fake_resynthesize)
    worker.execute({"kind": "resynthesize", "workdir": str(workdir), "payload": {"uids": []}})

    assert not ops.find(manifest.load(workdir), edited).get("locked")


def test_a_render_merges_at_every_stage_save(outputs, monkeypatch):
    """`dubbing.edit.rebuild` writes after each stage, so the end-of-job merge is
    not enough every one of those writes is a chance to clobber a live edit."""
    from dubbing_app import worker

    workdir = outputs / NAME
    edited = manifest.load(workdir)["segments"][3]["uid"]

    def fake_rebuild(m, wd, *, from_stage, progress=None, save=None):
        assert save is not None, "the child must own the write path"
        edit_on_disk(wd, edited, text_en="hand corrected")
        save()                                    # what rebuild does per stage
        assert ops.find(m, edited)["text_en"] == "hand corrected"
        return [from_stage]

    monkeypatch.setattr(ops, "rebuild", fake_rebuild)
    worker.execute({"kind": "render", "workdir": str(workdir), "payload": {}})
    assert ops.find(manifest.load(workdir), edited)["text_en"] == "hand corrected"


def test_a_journal_reports_a_split_that_landed_under_it(outputs, capsys):
    """`guard_structural` refuses a split while a job runs, but the check and the
    job's start are not one atomic act. When one does land, the child's list is
    stale it has never seen either half and `merge` matches by uid, so it
    would keep its own list and undo the split without a word."""
    from dubbing_app.worker import Journal

    workdir = outputs / NAME
    m = manifest.load(workdir)
    journal = Journal(workdir, m)

    disk = manifest.load(workdir)
    uid = disk["segments"][2]["uid"]
    halves = ops.split(disk, uid, 6.0)
    manifest.save(workdir, disk)

    journal.merge(manifest.load(workdir))
    frames = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]
    conflict = next(f for f in frames if f["type"] == "log" and f["level"] == "error")
    assert "segment list changed on disk" in conflict["message"]
    assert any(u in conflict["message"] for u in (uid,) + halves)


def test_a_journal_says_nothing_when_only_fields_changed(outputs, capsys):
    from dubbing_app.worker import Journal

    workdir = outputs / NAME
    journal = Journal(workdir, manifest.load(workdir))
    edit_on_disk(workdir, manifest.load(workdir)["segments"][3]["uid"],
                 text_en="hand corrected")
    journal.merge(manifest.load(workdir))
    assert "error" not in capsys.readouterr().out


def test_a_structural_edit_is_guarded_under_the_lock_it_writes_in(outputs):
    """Checked before the lock, the guard leaves a window: a job that starts in it
    is one the split is invisible to. The check belongs to the same critical
    section as the write."""
    import inspect

    source = inspect.getsource(app_mod.create_app)
    for route in ("def split_segment", "def merge_segment", "def patch_segment",
                  "def add_segment", "def remove_segment"):
        body = source.split(route, 1)[1].split("\n    @app.", 1)[0]
        assert body.index("with lock_for(name):") < body.index("guard_structural(name)")


def test_a_job_carries_the_segments_it_is_about(client, fake):
    uid = uids(client)[1]
    job = client.post(f"/api/projects/{NAME}/resynthesize",
                      json={"uids": [uid]}).json()["job"]
    assert job["uids"] == [uid]
    # A whole-run job is about the whole run, and says so as a list, not a null.
    assert client.post(f"/api/projects/{NAME}/render", json={}).json()["job"]["uids"] == []


def test_a_job_keeps_its_own_work_when_nothing_else_changed(outputs):
    """The merge must only replay what actually moved on disk re-applying an
    untouched disk copy wholesale would revert the job itself."""
    from dubbing_app.worker import Journal

    workdir = outputs / NAME
    m = manifest.load(workdir)
    journal = Journal(workdir, m)
    uid = m["segments"][1]["uid"]
    ops.find(m, uid)["text_en"] = "machine"
    assert journal.merge(manifest.load(workdir)) == []
    assert ops.find(m, uid)["text_en"] == "machine"


# ---------------------------------------------------------------------------
# process contract
# ---------------------------------------------------------------------------

def test_ready_line_is_one_json_object(capsys):
    import io

    buf = io.StringIO()
    server.announce(54321, buf)
    lines = buf.getvalue().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"status": "ready", "port": 54321,
                                    "version": server.__version__}


def test_bind_port_zero_returns_a_real_port():
    sock = server.bind("127.0.0.1", 0)
    try:
        port = sock.getsockname()[1]
        assert 1024 < port < 65536
    finally:
        sock.close()


def test_watchdog_skips_when_parent_is_init(monkeypatch):
    monkeypatch.setattr(server.os, "getppid", lambda: 1)
    assert server.watchdog() is None


def test_watchdog_fires_when_the_parent_changes(monkeypatch):
    ppids = iter([1234, 1234, 9999, 9999])
    monkeypatch.setattr(server.os, "getppid", lambda: next(ppids))
    fired = threading.Event()
    thread = server.watchdog(on_orphan=fired.set, interval=0.01)
    assert thread is not None
    assert fired.wait(2.0)


def test_stdin_watchdog_fires_on_eof():
    # The shell holds the write end for its whole life; closing it is the signal.
    read_fd, write_fd = __import__("os").pipe()
    import os as _os
    fired = threading.Event()
    server.stdin_watchdog(stream=_os.fdopen(read_fd, "rb"), on_close=fired.set)
    assert not fired.wait(0.1), "must not fire while the pipe is open"
    _os.close(write_fd)
    assert fired.wait(2.0), "EOF on stdin must shut the server down"


def test_widen_path_appends_only_missing_real_dirs():
    # A GUI-launched app gets launchd's PATH; the homebrew dir must be appended
    # exactly once, and a dir already present must not be duplicated.
    env = {"PATH": "/usr/bin:/bin:/opt/homebrew/bin"}
    server.widen_path(env)
    assert env["PATH"].count("/opt/homebrew/bin") == 1
    env2 = {"PATH": "/usr/bin:/bin"}
    server.widen_path(env2)
    for d in server.TOOL_DIRS:
        if Path(d).is_dir():
            assert d in env2["PATH"].split(":")
    assert env2["PATH"].startswith("/usr/bin:/bin")


def test_stdin_watchdog_is_opt_in():
    args = server.parse_args(["--port", "0"])
    assert args.exit_on_stdin_close is False
    args = server.parse_args(["--port", "0", "--exit-on-stdin-close"])
    assert args.exit_on_stdin_close is True


def test_server_arg_defaults():
    args = server.parse_args(["--host", "127.0.0.1", "--port", "0"])
    assert args.host == "127.0.0.1" and args.port == 0
    assert args.outputs.name == "outputs"


# ---------------------------------------------------------------------------
# ops: the pipeline seam
# ---------------------------------------------------------------------------

def test_app_segment_keys_survive_a_save(tmp_path, outputs):
    """`uid` would be dropped by manifest.save without the whitelist widening."""
    workdir = outputs / NAME
    m = manifest.load(workdir)
    ops.ensure_uids(m)
    ops.set_langs(m, m["segments"][0]["uid"], tgt_lang="ru")
    manifest.save(workdir, m)
    reloaded = manifest.load(workdir)
    assert reloaded["segments"][0]["uid"] == m["segments"][0]["uid"]
    assert reloaded["segments"][0]["tgt_lang"] == "ru"


def test_invalidate_cascades_downstream(outputs):
    m = manifest.load(outputs / NAME)
    ops.ensure_uids(m)
    uid = m["segments"][1]["uid"]
    ops.invalidate(m, uid, stages={"translate"})
    seg = ops.find(m, uid)
    assert "text_en" not in seg and "tts" not in seg and "place" not in seg


def test_invalidate_tts_keeps_the_translation(outputs):
    m = manifest.load(outputs / NAME)
    ops.ensure_uids(m)
    uid = m["segments"][1]["uid"]
    ops.invalidate(m, uid, stages={"tts"})
    seg = ops.find(m, uid)
    assert seg["text_en"] and "tts" not in seg and "place" not in seg


def test_invalidate_reopens_a_pipeline_keep_but_not_a_manual_one(outputs):
    m = manifest.load(outputs / NAME)
    ops.ensure_uids(m)
    failed = m["segments"][0]["uid"]              # keep_reason == "tts_failed"
    ops.invalidate(m, failed, stages={"tts"})
    assert ops.find(m, failed)["keep"] is False

    m = manifest.load(outputs / NAME)
    ops.ensure_uids(m)
    manual = m["segments"][1]["uid"]
    ops.set_keep(m, manual, True)
    ops.invalidate(m, manual, stages={"tts"})
    assert ops.find(m, manual)["keep"] is True    # the user's edit outranks the pipeline


def test_full_run_argv_matches_the_cli(tmp_path):
    argv = ops.full_run_argv(tmp_path / "run", {
        "input": "https://youtu.be/abc", "src_lang": "he", "tgt_lang": "ru",
        "duration_limit": 60, "context": "a note",
        "app_opts": {"genre": "movie", "register": "dialogue", "tts_model": "0.6b"}})
    from dubbing import cli

    args = cli.parse_args(argv)                   # the real parser accepts it
    assert args.source == "https://youtu.be/abc"
    assert args.tgt == "ru" and args.duration == 60 and args.genre == "movie"
    assert args.register == "dialogue" and args.tts_model == "0.6b"
    assert args.context == "a note"


def test_rebuild_rejects_stages_it_does_not_cover(outputs):
    """Only the manifest-side stages are rebuildable: everything before translate
    needs the source media and belongs to a real run. `translate` itself IS
    rebuildable the server's provisional version stopped at tts, `dubbing.edit`
    does not."""
    m = manifest.load(outputs / NAME)
    for stage in ("fetch", "stems", "transcript", "segments", "nonsense"):
        with pytest.raises(ops.EditError):
            ops.rebuild(m, outputs / NAME, from_stage=stage)


# ------------------------------------------------------ releasing a lock

def test_patch_locked_releases_a_hand_edit(client, outputs):
    """Editing locks a field; `locked: {}` hands it back to the pipeline. Without
    this a correction is permanent and can never be undone."""
    uid = uids(client)[0]
    r = client.patch(f"/api/projects/{NAME}/segments/{uid}", json={"text_en": "edited"})
    assert r.status_code == 200
    assert r.json()["segment"]["locked"] == {"text_en": True}

    r = client.patch(f"/api/projects/{NAME}/segments/{uid}", json={"locked": {}})
    assert r.status_code == 200
    assert not r.json()["segment"].get("locked")

    m = manifest.load(outputs / NAME)
    assert not (ops.find(m, uid) or {}).get("locked")


def test_patch_locked_rejects_unknown_fields(client):
    uid = uids(client)[0]
    r = client.patch(f"/api/projects/{NAME}/segments/{uid}", json={"locked": {"nope": True}})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_request"


# ------------------------------------------------- the ops -> dubbing.edit seam

MOVIE = {"genre": "movie", "register": "dialogue", "tts_model": "0.6b", "device": "cpu"}


def movie_project(outputs):
    """The manifest of a project the UI created with non-default options."""
    workdir = outputs / NAME
    m = manifest.load(workdir)
    m["source"]["app_opts"] = dict(MOVIE)
    manifest.save(workdir, m)
    return m


def test_edit_jobs_carry_the_projects_recorded_options(monkeypatch, outputs):
    """The UI's genre/register/tts-model are stored under `source.app_opts`, which
    `dubbing.edit._args` does not read so every job that re-runs a stage has to
    hand them over. Without this a `--genre movie` project is re-rendered as a
    documentary, in the default voice, and nobody is told."""
    from dubbing import edit as real_edit

    m = movie_project(outputs)
    seen: dict[str, dict] = {}

    def fake_rebuild(mm, wd, *, from_stage, progress=None, save=None, **overrides):
        seen["rebuild"] = overrides
        return [from_stage]

    def fake_retranslate(mm, wd, uids, *, progress=None, **kw):
        seen["retranslate"] = kw
        return {}

    def fake_resynthesize(mm, wd, uids, *, progress=None, **kw):
        seen["resynthesize"] = kw
        return {}

    monkeypatch.setattr(real_edit, "rebuild", fake_rebuild)
    monkeypatch.setattr(real_edit, "retranslate", fake_retranslate)
    monkeypatch.setattr(real_edit, "resynthesize", fake_resynthesize)

    ops.rebuild(m, outputs / NAME, from_stage="timeline")
    ops.retranslate(m, outputs / NAME, [])
    ops.resynthesize(m, outputs / NAME, [])

    assert seen["retranslate"] == {"register": "dialogue", "genre": "movie"}
    assert seen["resynthesize"] == {"device": "cpu", "model": "0.6b"}
    # …and the rebuild overrides really land on the argparse namespace the pipeline
    # computes its fingerprints from.
    args = real_edit._args(m, **seen["rebuild"])
    assert (args.genre, args.register, args.tts_model, args.device) == \
        ("movie", "dialogue", "0.6b", "cpu")


def test_a_render_stamps_the_fingerprints_the_full_run_would(outputs):
    """`dubbing.edit.rebuild` re-marks every stage it runs with a fingerprint, so a
    render made with the wrong options makes the next headless run redo the whole
    tail of the pipeline. The two paths must agree stage by stage."""
    from dubbing import cli
    from dubbing import edit as real_edit

    m = movie_project(outputs)
    render = cli.stage_params(real_edit._args(m, **ops.pipeline_overrides(m)), m)
    full = cli.stage_params(cli.parse_args(ops.full_run_argv(outputs / NAME, m["source"])), m)
    assert {s: render[s] for s in real_edit.REBUILDABLE} == \
        {s: full[s] for s in real_edit.REBUILDABLE}


def test_pipeline_overrides_never_invent_an_option(outputs):
    """A project that recorded nothing keeps `dubbing.edit._args`'s own fallbacks —
    overriding with a guess would be the same drift in the other direction."""
    m = manifest.load(outputs / NAME)
    assert ops.pipeline_overrides(m) == {}
    m["source"]["app_opts"] = {"genre": "movie", "register": None, "captions": "x.json3"}
    assert ops.pipeline_overrides(m) == {"genre": "movie"}


@pytest.mark.parametrize("call,stage", [
    (lambda p: ops.retranslate({"segments": []}, Path("."), [], progress=p), "translate"),
    (lambda p: ops.resynthesize({"segments": []}, Path("."), [], progress=p), "tts"),
])
def test_edit_progress_is_adapted_to_the_pipeline_signature(monkeypatch, call, stage):
    """`dubbing.edit` reports progress as (fraction, message); this module speaks
    NDJSON event dicts. The two were specified apart and passing one straight to
    the other raised `emit() takes 1 positional argument but 2 were given` on the
    first real job a path the fake runner never touched."""
    from dubbing import edit as real_edit

    seen: list[dict] = []
    captured: dict = {}

    def fake(m, workdir, uids, *, progress=None, **kw):
        captured["progress"] = progress
        progress(0.5, "halfway")          # exactly how dubbing.edit calls it
        return {}

    monkeypatch.setattr(real_edit, "retranslate", fake)
    monkeypatch.setattr(real_edit, "resynthesize", fake)
    call(seen.append)

    assert captured["progress"] is not None
    assert seen == [{"type": "stage", "stage": stage, "status": "running",
                     "progress": 0.5, "message": "halfway"}]


def test_resynthesize_leaves_every_segment_placed(monkeypatch, outputs):
    """A re-voiced segment must come back placed. Its new clip has a new length,
    so `invalidate` drops the old placement and a segment with no placement is
    absent from the mix, which is never-silent broken by the back door."""
    from dubbing import edit as real_edit
    from dubbing import tts as tts_mod

    m = manifest.load(outputs / NAME)
    uid = next(s["uid"] for s in m["segments"] if not s.get("keep"))

    # The clip a segment already owns, captured before `invalidate` drops it.
    clips = {s["uid"]: dict(s["tts"]) for s in m["segments"] if s.get("tts")}

    class FakeEngine:
        def __init__(self, *a, **k): pass
        def build_speaker_refs(self): pass
        def close(self): pass
        def keep_clip(self, seg): return dict(clips[seg["uid"]])
        def clip_for(self, seg, text): return dict(clips[seg["uid"]])

    monkeypatch.setattr(tts_mod, "Engine", FakeEngine)
    real_edit.resynthesize(m, outputs / NAME, [uid])

    unplaced = [s["uid"] for s in m["segments"] if not s.get("place")]
    assert unplaced == [], f"segments left unplaced: {unplaced}"


# ---------------------------------------------------------------------------
# built UI single-process serving (desktop packaging)
# ---------------------------------------------------------------------------

INDEX = "<!doctype html><title>Dubbing Studio</title><div id=root></div>"


def make_dist(root: Path) -> Path:
    """A stand-in for `app/ui/dist` as Vite writes it."""
    dist = root / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(INDEX, encoding="utf-8")
    (dist / "assets" / "index-abc123.js").write_text("export const x = 1;\n", encoding="utf-8")
    (dist / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    return dist


@pytest.fixture()
def dist(tmp_path):
    return make_dist(tmp_path / "ui")


@pytest.fixture()
def ui_client(outputs, fake, dist):
    with TestClient(create_app(outputs, runner=fake, ui_dir=str(dist))) as c:
        yield c


def test_ui_index_at_root(ui_client):
    resp = ui_client.get("/")
    assert resp.status_code == 200 and "Dubbing Studio" in resp.text


def test_ui_catch_all_serves_index_for_browser_routes(ui_client):
    """`/editor/<name>` exists only in the browser router a hard reload on it
    must still get the app, not a 404."""
    for path in (f"/editor/{NAME}", "/editor/x/deep", "/import"):
        resp = ui_client.get(path)
        assert resp.status_code == 200, path
        assert "Dubbing Studio" in resp.text, path


def test_ui_serves_real_files(ui_client):
    asset = ui_client.get("/assets/index-abc123.js")
    assert asset.status_code == 200 and "export const x" in asset.text
    icon = ui_client.get("/favicon.svg")
    assert icon.status_code == 200 and icon.text == "<svg/>"


def test_ui_never_shadows_the_api(ui_client):
    assert ui_client.get("/health").json()["status"] == "ok"
    assert [p["name"] for p in ui_client.get("/api/projects").json()["projects"]] == [NAME]
    # A *misspelled* API path must stay JSON: handing index.html to a fetch()
    # would turn "no such project" into an unparseable HTML body.
    for path in ("/api/nope", "/api/projects/ghost", "/media/ghost/x.wav", "/media/ghost"):
        resp = ui_client.get(path)
        assert resp.status_code == 404, path
        assert resp.json()["error"]["code"] == "not_found", path


def test_ui_refuses_traversal(ui_client, tmp_path):
    (tmp_path / "secret.txt").write_text("hf_token", encoding="utf-8")
    # Percent-encoded so the client cannot normalise it away before it is sent.
    resp = ui_client.get("/%2e%2e/secret.txt")
    assert resp.status_code == 404 and "hf_token" not in resp.text


def test_ui_resolve_file_guards(dist, tmp_path):
    from dubbing_app import ui as ui_mod
    from dubbing_app.errors import ApiError

    assert ui_mod.resolve_file(dist, "favicon.svg") == (dist / "favicon.svg").resolve()
    assert ui_mod.resolve_file(dist, "editor/whatsapp") is None      # -> index fallback
    outside = tmp_path / "secret.txt"
    outside.write_text("x", encoding="utf-8")
    (dist / "link.txt").symlink_to(outside)
    for bad in ("../secret.txt", "/etc/passwd", "a/../../secret.txt", "link.txt"):
        with pytest.raises(ApiError) as exc:
            ui_mod.resolve_file(dist, bad)
        assert exc.value.code == "not_found", bad


def test_ui_absent_dist_is_api_only(outputs, fake, tmp_path):
    with TestClient(create_app(outputs, runner=fake, ui_dir=str(tmp_path / "nope"))) as c:
        root = c.get("/")
        assert root.status_code == 404 and root.json()["error"]["code"] == "not_found"
        assert c.get("/health").json()["status"] == "ok"


def test_ui_dir_empty_string_disables(outputs, fake, dist, monkeypatch):
    from dubbing_app import ui as ui_mod

    monkeypatch.setattr(ui_mod, "DEFAULT_UI_DIR", dist)
    assert ui_mod.resolve_dir(None) == dist
    assert ui_mod.resolve_dir("") is None and ui_mod.resolve_dir("  ") is None
    with TestClient(create_app(outputs, runner=fake, ui_dir="")) as c:
        assert c.get("/").status_code == 404


def test_server_ui_dir_flag():
    assert server.parse_args([]).ui_dir is None
    assert server.parse_args(["--ui-dir", ""]).ui_dir == ""
    assert server.parse_args(["--ui-dir", "/x/dist"]).ui_dir == "/x/dist"


# ---------------------------------------------------------------------------
# setup / first-run checks
# ---------------------------------------------------------------------------

def test_setup_report_shape(client):
    body = client.get("/api/setup").json()
    assert isinstance(body["ok"], bool)
    by_id = {c["id"]: c for c in body["checks"]}
    for wanted in ("ffmpeg", "sox", "model.translate", "model.tts.1.7b",
                   "model.asr.en", "model.lid", "model.diarization", "model.demucs",
                   "disk"):
        assert wanted in by_id, wanted
    # And nothing about a credential. There is no `hf_token` row any more: the
    # diarization weights ship with the app, every model installs without an
    # account, and a checklist row that can only ever say "not set, and nothing
    # needs it" is advertising a problem no user has.
    assert "hf_token" not in by_id
    for c in body["checks"]:
        assert {"id", "label", "ok", "detail", "required"} <= set(c)
        assert isinstance(c["ok"], bool) and isinstance(c["detail"], str) and c["label"]
    # `ok` is the conjunction of the required checks only; Demucs and free space
    # are informational and must never gate first run.
    assert by_id["model.demucs"]["required"] is False
    assert by_id["disk"]["required"] is False
    assert body["ok"] == all(c["ok"] for c in body["checks"] if c["required"])


def test_setup_grades_every_check_and_required_is_derived_from_it(client):
    """`required` has two values and the question has three: absent diarization
    weights (every speaker in the video collapsed into one) were reported exactly
    as informationally as a Korean checkpoint a Hebrew→English run never opens."""
    from dubbing_app import setup as setup_mod

    checks = client.get("/api/setup").json()["checks"]
    for c in checks:
        assert c["severity"] in setup_mod.SEVERITIES, c["id"]
        # One source of truth: nothing may be blocking and not required, or the
        # reverse the old flag is now a view of the new grade.
        assert c["required"] is (c["severity"] == setup_mod.BLOCKING), c["id"]

    by_id = {c["id"]: c["severity"] for c in checks}
    from dubbing import tts

    assert by_id["ffmpeg"] == by_id["uv"] == "blocking"
    assert by_id["model.translate"] == by_id["model.asr.en"] == "blocking"
    assert by_id[f"model.tts.{tts.DEFAULT_TTS_MODEL}"] == "blocking"
    # The run works and is worse: a third language nobody noticed, or every
    # character in the video dubbed in one voice. Diarization sat in `optional`
    # while the only fix for it was a Hugging Face account, because a grade that
    # says "this matters" beside a fix the app cannot offer is only noise. The
    # app restores the bundled weights itself now, so the grade can say what a
    # missing copy actually costs.
    assert by_id["model.lid"] == by_id["model.diarization"] == "degrades"
    # Irrelevant until asked for, or self-downloading. sox is here on evidence:
    # the only sox caller in the tree is qwen_tts's 25Hz tokenizer, and the
    # pipeline loads only 12Hz checkpoints — a brewless Mac dubs without it.
    # No credential row appears in any grade: `hf_token` used to sit here, first
    # as `degrades` and then as `optional`, because diarization loaded a gated
    # repo. It does not any more, so the row is gone rather than demoted.
    for optional in ("sox", "model.asr.he",
                     "model.asr.tgt", "model.tts.he", "model.g2p.he", "model.demucs",
                     "disk"):
        assert by_id[optional] == "optional", optional


def test_setup_absent_optional_tool_changes_nothing_about_readiness(monkeypatch, tmp_path):
    """A tool the shipped pipeline never calls must not be able to make a
    provisioned machine read as anything but ready.

    sox is the row: a brewless Mac dubs without it (the only caller is
    qwen_tts's 25Hz tokenizer and this pipeline loads 12Hz checkpoints), and the
    Setup screen draws it as a grey "Not installed" that is counted nowhere. The
    screen can only be that quiet because the server is: the row is `optional`,
    it carries no `stage`, and `report`'s verdict is the conjunction of the
    *required* rows, so taking sox away moves nothing.
    """
    from dubbing import tools as tool_recipes
    from dubbing_app import setup as setup_mod

    def failing_required(report):
        return {c["id"] for c in report["checks"] if c["required"] and not c["ok"]}

    before = setup_mod.report(tmp_path)
    real = tool_recipes.resolve_tool
    monkeypatch.setattr(tool_recipes, "resolve_tool",
                        lambda name: None if name == "sox" else real(name))
    after = setup_mod.report(tmp_path)

    row = next(c for c in after["checks"] if c["id"] == "sox")
    assert row["ok"] is False
    assert row["severity"] == setup_mod.OPTIONAL and row["required"] is False
    assert "stage" not in row and setup_mod.blocking_stage("sox") is None
    assert after["ok"] == before["ok"]
    assert failing_required(after) == failing_required(before)


def test_setup_names_the_stage_a_blocking_check_would_kill(client):
    """"Runs will fail" is true and useless. The screen offers "Skip anyway —
    runs will fail at fetch", and only the server knows which stage that is."""
    checks = {c["id"]: c for c in client.get("/api/setup").json()["checks"]}
    assert checks["ffmpeg"]["stage"] == "fetch"
    assert checks["model.translate"]["stage"] == "translate"
    assert checks["model.asr.en"]["stage"] == "tts"
    # Only where it means something: a stage on an optional row reads as urgency.
    assert "stage" not in checks["disk"] and "stage" not in checks["model.demucs"]
    # And `None` is an honest answer for a blocking check with no stage to name:
    # `uv` is how this project's environment is built, but a running server
    # spawns its job child with `sys.executable` and never shells out to it, so
    # picking a stage for it would be a guess dressed as a fact.
    assert checks["uv"]["severity"] == "blocking" and checks["uv"]["stage"] is None


def test_setup_asks_for_no_credential_anywhere_and_still_diarizes(client, monkeypatch):
    """The row that made this app unshippable, and is now not a row at all.

    Diarization used to load a gated repo, so a machine with no Hugging Face
    account dubbed every character in the video in one voice, and the setup
    screen said so in the middle grade beside a paste field. The weights are
    CC-BY-4.0 and ship inside the app since v0.4.0
    (`segments.DIARIZATION_DIR`), so the honest report is silence: no row, no
    badge, no paragraph explaining why the badge does not matter.

    What is asserted here is both halves, because removing the row would be a
    lie if the second half were not true. No check mentions a token, and the
    pipeline still finds a diarization source on a machine that has never had
    one.
    """
    from dubbing import segments

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv(segments.DIARIZATION_HUB_ENV, raising=False)
    body = client.get("/api/setup").json()
    assert not any("token" in c["id"] for c in body["checks"])
    blob = json.dumps(body).lower()
    assert "hugging face token" not in blob and "hf_token" not in blob
    # And the pipeline agrees: with no token at all there is still a source, and
    # not one of the sources it would try needs one.
    sources = segments.diarization_sources()
    assert sources and not any(needs_token for _, _, needs_token in sources)
    assert segments.DIARIZATION_MODEL not in [c for c, _, _ in sources]


def test_setup_model_paths_come_from_the_pipeline(client):
    """The check reads `dubbing`'s own constants, so it cannot drift from what
    the pipeline actually opens."""
    from dubbing import transcript, translate, tts

    by_id = {c["id"]: c for c in client.get("/api/setup").json()["checks"]}
    assert by_id["model.translate"]["path"] == str(translate.MODEL_PATH)
    assert by_id["model.asr.en"]["path"] == str(transcript.EN_ASR_MODEL)
    assert by_id["model.lid"]["path"] == str(transcript.LID_MODEL)
    assert by_id["model.asr.he"]["path"] == str(transcript.WHISPER_MODEL)
    # Only the default checkpoint gets a row: 0.6b stays in tts.TTS_MODELS for
    # old-manifest re-runs but is deliberately not offered as a download.
    default_spec = tts.TTS_MODELS[tts.DEFAULT_TTS_MODEL]
    assert by_id[f"model.tts.{tts.DEFAULT_TTS_MODEL}"]["path"].endswith(default_spec["dir"])
    assert by_id[f"model.tts.{tts.DEFAULT_TTS_MODEL}"]["required"] is True
    for key in tts.TTS_MODELS:
        if key != tts.DEFAULT_TTS_MODEL:
            assert f"model.tts.{key}" not in by_id


def test_setup_never_reports_a_token_even_when_the_machine_has_one(client, monkeypatch):
    """A developer machine with HF_TOKEN exported gets the same checklist as a
    machine that has never heard of Hugging Face.

    The report used to grow a green "set via HF_TOKEN" row here, which read as
    the app noticing a credential it might use. It has no use for one, so it
    does not look, and nothing resembling the value can appear in the response
    for the simple reason that nothing reads it.
    """
    monkeypatch.setenv("HF_TOKEN", "hf_supersecret_value")
    body = client.get("/api/setup").json()
    assert "hf_supersecret_value" not in json.dumps(body)
    assert not any("token" in c["id"] for c in body["checks"])


def test_setup_loads_no_model(client):
    """Pure filesystem and env: nothing heavy may be imported, or the desktop
    shell's first-run probe would cost ten seconds and a gigabyte of RAM."""
    heavy = {"torch", "mlx", "mlx_lm", "demucs", "faster_whisper", "speechbrain",
             "transformers", "pyannote"}
    before = set(sys.modules)
    client.get("/api/setup")
    added = {n.split(".")[0] for n in set(sys.modules) - before}
    assert not (added & heavy), sorted(added & heavy)


def test_setup_disk_check_is_informational(outputs):
    from dubbing_app import setup as setup_mod

    check = setup_mod.disk_check(outputs / "does" / "not" / "exist" / "yet")
    assert check["required"] is False and check["bytes"] > 0


def test_setup_model_check_reports_size(tmp_path):
    from dubbing_app import setup as setup_mod

    d = tmp_path / "model"
    (d / "sub").mkdir(parents=True)
    (d / "sub" / "weights.bin").write_bytes(b"x" * 2048)
    check = setup_mod.model("model.x", "X", d)
    assert check["ok"] is True and check["bytes"] == 2048 and "2 KB" in check["detail"]
    missing = setup_mod.model("model.y", "Y", tmp_path / "gone", note="downloads on use")
    assert missing["ok"] is False and missing["bytes"] == 0
    assert "downloads on use" in missing["detail"]


def a_sharded_model(root: Path, *, shards=("model-00001-of-00002.safetensors",
                                           "model-00002-of-00002.safetensors"),
                    present=()) -> Path:
    """The directory `hf download` leaves within its first two seconds: the
    config, the tokenizer and the shard index, and only the shards named in
    `present` actually on disk."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text('{"model_type": "gemma3"}')
    (root / "tokenizer.json").write_text("{}")
    weight_map = {f"layer.{i}.weight": name for i, name in enumerate(shards)}
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 6_400_000_000}, "weight_map": weight_map}))
    for name in present:
        (root / name).write_bytes(b"w" * 4096)
    return root


def test_a_model_missing_the_shards_its_index_names_is_not_ready(tmp_path):
    """The bug a fresh-install tester found at 1% downloaded: `hf download`
    writes the config, the tokenizer and the index in the first second or two,
    so "the directory is not empty" called a 6.4 GB blocking model READY for the
    whole fetch, and forever if the fetch died. The index names every shard the
    loader opens, so the answer is exact and costs one JSON parse."""
    from dubbing_app import setup as setup_mod

    d = tmp_path / "gemma"
    a_sharded_model(d)
    row = setup_mod.model("model.translate", "Translation", d, hub="org/gemma",
                          hub_bytes=6_400_000_000)
    assert row["ok"] is False and row["state"] == setup_mod.INCOMPLETE
    assert row["bytes"] > 0                       # something IS there: that is the point
    assert setup_mod.model_ready(d, row["bytes"], 6_400_000_000) is False

    # One shard of two is still not a model that loads.
    a_sharded_model(d, present=("model-00001-of-00002.safetensors",))
    assert setup_mod.model("model.translate", "T", d)["state"] == setup_mod.INCOMPLETE

    # Both, and the size floor is not consulted at all: the index is exact, and
    # a rounded estimate must not be able to overrule it.
    a_sharded_model(d, present=("model-00001-of-00002.safetensors",
                                "model-00002-of-00002.safetensors"))
    whole = setup_mod.model("model.translate", "T", d, hub="org/gemma",
                            hub_bytes=6_400_000_000)
    assert whole["ok"] is True and whole["state"] == setup_mod.READY


def test_a_model_with_no_index_falls_back_to_the_size_floor(tmp_path):
    """faster-whisper and speechbrain directories carry no shard index, so the
    only evidence is how much arrived against how much was expected. Not an
    equality: the table's sizes are measured from real installs and rounded, and
    a row that went red on a working model would be the same lie backwards."""
    from dubbing_app import setup as setup_mod

    d = tmp_path / "whisper"
    d.mkdir()
    (d / "model.bin").write_bytes(b"x" * 100)
    assert setup_mod.model("model.asr.en", "ASR", d, hub_bytes=1000)["state"] == "incomplete"
    (d / "model.bin").write_bytes(b"x" * 900)     # exactly the floor
    assert setup_mod.model("model.asr.en", "ASR", d, hub_bytes=1000)["ok"] is True
    # And a model nobody measured is taken at its word: a check that cannot pass
    # is the thing this module's second rule exists to prevent.
    assert setup_mod.model("model.x", "X", d)["ok"] is True


def test_an_incomplete_row_says_which_of_wait_and_resume_it_means(tmp_path, monkeypatch):
    """"Incomplete" alone leaves the user with nothing to decide. A live fetch is
    "wait" and an abandoned one is "press Download", and the difference is
    readable off the lock files huggingface_hub keeps beside the bytes."""
    from dubbing_app import setup as setup_mod

    d = tmp_path / "gemma"
    a_sharded_model(d)
    marks = d / ".cache" / "huggingface" / "download"
    marks.mkdir(parents=True)
    lock = marks / "model-00001-of-00002.safetensors.lock"
    lock.write_text("")

    live = setup_mod.model("model.translate", "T", d, hub="org/gemma",
                           hub_bytes=1_000_000_000)
    assert live["state"] == "incomplete" and live["downloading"] is True
    assert "downloading, 0% of 953.7 MB" in live["detail"]

    # Nothing has touched it for an hour: the fetch is gone and the partial
    # files are waiting for the next attempt to resume them.
    old = time.time() - 3600
    os.utime(lock, (old, old))
    stalled = setup_mod.model("model.translate", "T", d, hub="org/gemma",
                              hub_bytes=1_000_000_000)
    assert stalled["state"] == "incomplete" and "downloading" not in stalled
    assert "partial download, press Download to finish" in stalled["detail"]
    # Either way the row carries the line that finishes it, and it is a resume:
    # `hf download` keeps what is on disk.
    assert "`uv run hf download org/gemma --local-dir " in stalled["detail"]


def test_an_incomplete_row_is_installable_and_the_plan_queues_it(client, tmp_path,
                                                                 monkeypatch):
    """A half-downloaded blocking model has to be fixable from the screen it is
    red on. `snapshot_download` resumes from the partial files, so the row's
    button and the queue both cost only the half that is not there."""
    from dubbing import translate
    from dubbing_app import setup as setup_mod

    monkeypatch.setattr(translate, "MODEL_PATH", a_sharded_model(tmp_path / "gemma"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-cache"))
    report = client.get("/api/setup").json()
    row = {c["id"]: c for c in report["checks"]}["model.translate"]
    assert row["state"] == "incomplete" and row["installable"] is True
    assert report["ok"] is False                  # and it does not certify the run
    assert "model.translate" in {i["id"] for i in setup_mod.install_plan(report)}


def test_every_row_carries_a_state_and_ok_is_derived_from_it(client):
    """`state` is to `ok` what `severity` is to `required`: one source of truth,
    so no client can be shown a row that is READY and not ok, or the reverse."""
    from dubbing_app import setup as setup_mod

    for c in client.get("/api/setup").json()["checks"]:
        assert c["state"] in setup_mod.STATES, c["id"]
        assert c["ok"] is (c["state"] == setup_mod.READY), c["id"]
    with pytest.raises(ValueError, match="disagrees"):
        setup_mod.check("x", "X", True, "", state=setup_mod.INCOMPLETE)


def test_setup_uv_probe_mirrors_the_shells_fallback_chain(tmp_path, monkeypatch):
    """The shell's `find_uv()` found uv and started this server with it, and the
    server's bare `shutil.which` then reported the tool missing and required.
    The Python probe honours the same override and the same off-PATH homes."""
    from dubbing_app import setup as setup_mod

    fake = tmp_path / "uv"
    fake.write_text("#!/bin/sh\n")
    monkeypatch.setenv("DUBSTUDIO_UV_PATH", str(fake))
    assert setup_mod.find_uv() == str(fake)

    # Off PATH entirely (a Finder-launched .app): the `~/.local/bin` that uv's
    # own installer uses still answers. A dangling override is ignored, not
    # trusted, and the literal Homebrew paths are emptied so a machine that has
    # uv installed there cannot make this pass for the wrong reason.
    monkeypatch.setenv("DUBSTUDIO_UV_PATH", str(tmp_path / "gone"))
    monkeypatch.setattr(setup_mod, "UV_FALLBACKS", ())
    monkeypatch.setattr(setup_mod.shutil, "which", lambda exe: None)
    home = tmp_path / "home"
    local = home / ".local" / "bin"
    local.mkdir(parents=True)
    (local / "uv").write_text("#!/bin/sh\n")
    monkeypatch.setattr(setup_mod.Path, "home", lambda: home)
    assert setup_mod.find_uv() == str(local / "uv")


def test_setup_demucs_passes_from_the_hf_cache(tmp_path, monkeypatch):
    """demucs 4.x fetches `htdemucs_ft` from the Hub into the HF cache, not the
    torch hub cache; probing only the latter made this a row that could never
    pass on a working install."""
    from dubbing_app import setup as setup_mod

    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))  # no `.th` anywhere
    hub = tmp_path / "hf"
    blobs = hub / "models--adefossez--HTDemucs-ft" / "blobs"
    blobs.mkdir(parents=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    # An empty `blobs/` is an interrupted fetch, not a model.
    assert setup_mod.demucs_check()["ok"] is False
    (blobs / "0a1b").write_bytes(b"x" * 1024)
    row = setup_mod.demucs_check()
    assert row["ok"] is True and row["bytes"] == 1024
    assert "HTDemucs-ft" in row["path"]

    # `HF_HOME` alone resolves the way huggingface_hub does: cache under `hub/`.
    monkeypatch.delenv("HF_HUB_CACHE")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))
    assert setup_mod.hf_hub_cache() == tmp_path / "hf_home" / "hub"


def test_a_model_row_passes_from_the_hf_cache_its_loader_downloads_into(tmp_path,
                                                                       monkeypatch):
    """The same two-location probe demucs got, generalised to every row with a
    hub. `translate.py` and `tts.py` both fall back to the hub id when
    `models/<dir>` is absent, and that download lands in the HF cache, so a row
    that stats only the local directory reads MISSING on a machine where the
    feature demonstrably works. A blocking row that cannot pass is the strongest
    possible instruction to ignore the whole screen."""
    from dubbing_app import setup as setup_mod

    local = tmp_path / "models" / "thing"
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))

    def row():
        return setup_mod.model("model.x", "X", local, hub="org/thing", hub_bytes=99,
                               hub_cached=True)

    assert row()["ok"] is False
    blobs = tmp_path / "hub" / "models--org--thing" / "blobs"
    blobs.mkdir(parents=True)
    # An empty `blobs/` is an interrupted fetch, not a model: the same reading
    # `demucs_check` makes, so the two rows cannot disagree about the same cache.
    assert row()["ok"] is False
    (blobs / "0a1b").write_bytes(b"x" * 2048)
    found = row()
    assert found["ok"] is True and found["bytes"] == 2048
    # `path` stays the place the pipeline would load from, so the row is still
    # checkable against `dubbing`'s constants; `found_at` says where it really is.
    assert found["path"] == str(local)
    assert found["found_at"] == str(blobs.parent)
    assert "Hugging Face cache" in found["detail"]
    # And the local directory still wins when it has something in it: that is
    # the path the loader prefers, so it is the path the row must report.
    local.mkdir(parents=True)
    (local / "weights.bin").write_bytes(b"y" * 512)
    here = row()
    assert here["ok"] is True and here["bytes"] == 512 and "found_at" not in here


def test_a_cached_copy_only_counts_where_the_loader_would_read_it(tmp_path, monkeypatch):
    """The other half of the two-location probe, and the half that keeps it from
    becoming the opposite lie. Language ID and the Hebrew LoRA read their local
    directory and nothing else (`transcript.load_lid` returns None without it,
    `hebrew.attach_adapter` raises), so a `models--*` in the cache does not make
    either feature work and a green row for one would send the user away from
    the thing that is actually broken."""
    from dubbing_app import setup as setup_mod

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    downloads = setup_mod.model_downloads()
    for id_ in ("model.lid", "model.tts.he"):
        hub = downloads[id_]["hub"]
        blobs = tmp_path / "hub" / f"models--{hub.replace('/', '--')}" / "blobs"
        blobs.mkdir(parents=True)
        (blobs / "0a1b").write_bytes(b"x" * 4096)
        assert downloads[id_].get("cached") is not True, id_
    rows = {c["id"]: c for c in setup_mod.model_checks()}
    for id_ in ("model.lid", "model.tts.he"):
        assert rows[id_]["ok"] is False, id_
        assert "found_at" not in rows[id_], id_


def test_every_failing_model_row_carries_the_command_that_fixes_it(client, tmp_path,
                                                                   monkeypatch):
    """`setup.model`'s docstring makes this promise, and the rows that broke it
    were both of the blocking ones: "missing" with no command is a scavenger
    hunt, and the row is the only place a user with no terminal ever looks.

    Two halves. Every failing model row has *a* backticked command, and a row
    that names a hub has exactly the download line for that hub, with an
    absolute `--local-dir` (a relative one installs the weights wherever the
    terminal happened to be, which is nowhere the app looks)."""
    import re

    from dubbing_app import setup as setup_mod

    checks = client.get("/api/setup").json()["checks"]
    model_rows = [c for c in checks if c["id"].startswith("model.")]
    assert model_rows
    for row in model_rows:
        if row["ok"]:
            continue
        commands = re.findall(r"`([^`]+)`", row["detail"])
        assert commands, f"{row['id']} says missing and names no command"
        if row.get("hub"):
            expected = (f"uv run hf download {row['hub']} "
                        f"--local-dir {row['path']}")
            assert expected in commands, row["id"]
            assert Path(row["path"]).is_absolute(), row["id"]
    # The rows that are *always* downloadable carry the command whether or not
    # this machine happens to be missing them, so the claim above is not vacuous
    # on a developer's fully-populated checkout. Both places the probe looks are
    # pointed somewhere empty, or a cached copy would make this vacuous too.
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-cache"))
    for id_, spec in setup_mod.model_downloads().items():
        built = setup_mod.model(id_, id_, Path(spec["path"]).parent / "nothing-here",
                                hub=spec["hub"], hub_bytes=spec["bytes"])
        assert f"`uv run hf download {spec['hub']} --local-dir " in built["detail"]


# ---------------------------------------------------------------------------
# installing a missing tool from the app
# ---------------------------------------------------------------------------
#
# Not one of these runs a package manager. The id → argv table is the only
# executable thing in the feature, so every test here swaps it for a shell stub
# and asserts on the plumbing around it: what is refused, what is serialised,
# and what the server believes once the process has exited.

STUB = ("/bin/sh", "-c", "echo installing; exit 0")


@pytest.fixture()
def stub_installers(monkeypatch):
    """The app's own installer, pointed at a shell stub instead of `brew`."""
    def use(argv=STUB):
        monkeypatch.setattr(install_mod, "INSTALLERS", {"ffmpeg": tuple(argv)})
    return use


def test_setup_marks_the_rows_the_app_can_install(client):
    """The UI must not carry its own copy of the whitelist: a button on a row
    whose POST is a 400 is worse than no button.

    Which rows those are is the *platform's* answer, not this file's: brew and
    winget can run unattended, `sudo apt-get` cannot (see `dubbing/tools.py`),
    so on Linux the honest answer is no buttons at all."""
    from dubbing import tools
    from dubbing_app import setup as setup_mod

    checks = client.get("/api/setup").json()["checks"]
    installable = {c["id"] for c in checks if c["installable"]}
    # Plus ffmpeg wherever the static build is the route: a machine with no
    # package manager this app may drive still has a button, and this set is
    # what the UI reads to draw it.
    static = {"ffmpeg"} if install_mod.static_route("ffmpeg") else set()
    # Plus the diarization row wherever a copy of the bundled weights is still
    # on this machine to restore from. It is in no download table and never will
    # be (the upstream repo is gated), so it is named separately here for the
    # same reason it is a separate route: it fetches nothing.
    restore = ({install_mod.DIARIZATION_ID}
               if install_mod.diarization_source() is not None else set())
    assert installable == (set(install_mod.INSTALLERS)
                           | set(setup_mod.model_downloads()) | static | restore)
    assert installable & set(tools.recipes()) == set(tools.auto_installers()) | static
    # And never the rows no table and no route covers: self-fetching caches.
    assert {"model.demucs", "model.g2p.he", "disk"}.isdisjoint(installable)
    assert all(isinstance(c["installable"], bool) for c in checks)


def test_install_refuses_an_id_it_has_no_recipe_for(client):
    """A model no snapshot satisfies (gated pyannote, self-fetching Demucs and
    G2P) still gets no button; the refusal hands the user what to do instead."""
    from dubbing import tools

    for bad in ("model.demucs", "model.g2p.he", "hf_token", "disk", "rm -rf /"):
        # `hf_token` is in this list precisely because it is no longer a row: an
        # id the report never mentions must still be refused rather than run.
        r = client.post("/api/setup/install", json={"id": bad})
        assert r.status_code == 400, bad
        message = r.json()["error"]["message"]
        assert r.json()["error"]["code"] == "invalid_request"
        # Every tool command this platform knows, whether or not it has a button.
        assert tools.command("ffmpeg") in message and tools.command("sox") in message
        assert "detail" in message                      # …where the real command is


def test_install_body_is_strict(client):
    """Nothing but `id` is read, so nothing but `id` is accepted a hopeful
    `argv` must not be quietly ignored, it must be a 400."""
    r = client.post("/api/setup/install", json={"id": "ffmpeg", "argv": ["/bin/sh"]})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request"


def test_install_says_where_to_get_homebrew(monkeypatch):
    """The manager that installs the tools cannot itself be installed from here,
    so the refusal names the one URL. Driven straight at `Installer` with a Mac's
    recipe: the table is the platform's, this rule is not. Driven with `sox` —
    the one tool that keeps this refusal, because `ffmpeg` on a brewless Mac
    takes the static route instead (the test below)."""
    monkeypatch.setattr(install_mod.shutil, "which", lambda exe, *a, **k: None)
    inst = install_mod.Installer(lambda id_: None,
                                 recipes={"sox": ("brew", "install", "sox")})
    with pytest.raises(install_mod.invalid("x").__class__) as exc:
        inst.start("sox")
    message = exc.value.message
    assert exc.value.code == "invalid_request"
    assert "https://brew.sh" in message and "sox" in message


def test_brewless_mac_installs_a_static_ffmpeg_into_the_workspace(monkeypatch, tmp_path):
    """No Homebrew is the factory state of every Mac, and 'get it from brew.sh'
    is a terminal plus an admin password — the journey the DMG exists to avoid.
    The button must fall back to the static build, land both binaries in the
    workspace tools dir, and pass the re-probe without a restart."""
    from dubbing import tools

    monkeypatch.setattr(install_mod.shutil, "which", lambda exe, *a, **k: None)
    monkeypatch.setattr(tools, "platform_key", lambda *a: "darwin")
    bin_dir = tmp_path / "tools" / "bin"
    monkeypatch.setenv(tools.TOOLS_DIR_ENV, str(bin_dir))

    fetched = tmp_path / "wheel-cache"
    fetched.mkdir()
    def fake_fetch(log):
        log("fetching (stub)")
        paths = []
        for name in ("ffmpeg", "ffprobe"):
            p = fetched / name
            p.write_text("#!/bin/sh\nexit 0\n")
            paths.append(str(p))
        return tuple(paths)
    monkeypatch.setattr(install_mod, "fetch_static_ffmpeg", fake_fetch)

    from dubbing_app import setup as setup_mod

    inst = install_mod.Installer(setup_mod.probe,
                                 recipes={"ffmpeg": ("brew", "install", "ffmpeg")})
    inst.start("ffmpeg")
    assert inst.wait(10.0)
    status = inst.status()
    assert status["ok"] is True and status["error"] is None
    # Both binaries copied — not symlinked, `uv sync` prunes the wheel — and runnable.
    for name in ("ffmpeg", "ffprobe"):
        installed = bin_dir / name
        assert installed.is_file() and not installed.is_symlink()
        assert os.access(installed, os.X_OK)
    # The re-probed row is fresh evidence: resolve_tool finds tools/bin first.
    assert status["check"]["ok"] is True
    assert status["check"]["path"] == str(bin_dir / "ffmpeg")
    assert any("static" in line for line in status["tail"])


def test_ffmpeg_has_an_install_route_on_every_platform(monkeypatch):
    """The one tool every stage shells out to must be installable from the app
    on all three platforms — not only on the Mac the feature was written on.

    Linux is the case that was missing: its recipe is `sudo apt-get`, which
    wants a password on a terminal the app does not have, so it is deliberately
    not an unattended installer and the row had *no button at all*. The static
    wheel ships a Linux and a Windows build too, and one button that installs
    everything must not quietly skip ffmpeg on two thirds of the platforms.
    """
    from dubbing import tools

    monkeypatch.setattr(install_mod.shutil, "which", lambda exe, *a, **k: None)
    for platform in ("darwin", "linux", "win32"):
        monkeypatch.setattr(tools, "platform_key", lambda *a, p=platform: p)
        assert install_mod.static_route("ffmpeg") is True, platform
        # The row's detail says what the button will do before it is pressed…
        assert "static build" in (install_mod.route("ffmpeg") or ""), platform
        # …and the flag that draws the button agrees with the code behind it.
        assert install_mod.installable("ffmpeg") is True, platform
        assert setup_mod.probe("ffmpeg")["installable"] is True, platform
        # sox gets no such fallback anywhere: no vetted static source, and
        # nothing the shipped pipeline runs needs it.
        assert install_mod.static_route("sox") is False, platform


def test_a_machine_with_no_unattended_manager_still_installs_ffmpeg(monkeypatch, tmp_path):
    """Linux, end to end through the slot: no recipe in the table at all, and
    the install still runs and lands both binaries in the workspace."""
    from dubbing import tools

    monkeypatch.setattr(install_mod.shutil, "which", lambda exe, *a, **k: None)
    monkeypatch.setattr(tools, "platform_key", lambda *a: "linux")
    bin_dir = tmp_path / "tools" / "bin"
    monkeypatch.setenv(tools.TOOLS_DIR_ENV, str(bin_dir))

    def fake_fetch(log):
        log("fetching (stub)")
        paths = []
        for name in ("ffmpeg", "ffprobe"):
            p = tmp_path / name
            p.write_text("#!/bin/sh\nexit 0\n")
            paths.append(str(p))
        return tuple(paths)

    monkeypatch.setattr(install_mod, "fetch_static_ffmpeg", fake_fetch)
    # The empty table is the point: `tools.auto_installers()` contributes no
    # rows on Linux, so this is what the real Installer holds there.
    inst = install_mod.Installer(setup_mod.probe, recipes={})
    inst.start("ffmpeg")
    assert inst.wait(10.0)
    status = inst.status()
    assert status["ok"] is True and status["error"] is None
    assert (bin_dir / "ffmpeg").is_file() and (bin_dir / "ffprobe").is_file()
    assert any("static" in line for line in status["tail"])


def test_resolve_tool_prefers_override_then_workspace_then_path(monkeypatch, tmp_path):
    """The one lookup every call site uses, in its promised order: a per-tool
    env override, the workspace tools dir, then PATH — so a static build the
    app installed beats the shell's PATH, and an explicit override beats both."""
    from dubbing import tools

    workspace = tmp_path / "tools" / "bin"
    workspace.mkdir(parents=True)
    on_path = tmp_path / "path"
    on_path.mkdir()
    for where in (workspace / "ffmpeg", on_path / "ffmpeg"):
        where.write_text("#!/bin/sh\n")
        where.chmod(0o755)
    monkeypatch.setenv("PATH", str(on_path))
    monkeypatch.delenv("DUBSTUDIO_FFMPEG", raising=False)
    # Pointed at an *empty* dir, not delenv'd: the default is the checkout's own
    # tools/bin, and this test must not care what a previous install left there.
    monkeypatch.setenv(tools.TOOLS_DIR_ENV, str(tmp_path / "empty"))

    assert tools.resolve_tool("ffmpeg") == str(on_path / "ffmpeg")   # PATH is the floor
    monkeypatch.setenv(tools.TOOLS_DIR_ENV, str(workspace))
    assert tools.resolve_tool("ffmpeg") == str(workspace / "ffmpeg")
    override = tmp_path / "custom-ffmpeg"
    override.write_text("#!/bin/sh\n")
    monkeypatch.setenv("DUBSTUDIO_FFMPEG", str(override))
    assert tools.resolve_tool("ffmpeg") == str(override)
    # A dangling override is ignored, never returned: the pipeline would exec it.
    monkeypatch.setenv("DUBSTUDIO_FFMPEG", str(tmp_path / "gone"))
    assert tools.resolve_tool("ffmpeg") == str(workspace / "ffmpeg")


def test_setup_row_names_the_route_the_button_takes(monkeypatch):
    """'Install' must say what pressing it does before it is pressed: the
    package manager when one is there, the workspace static build when not."""
    from dubbing import tools
    from dubbing_app import setup as setup_mod

    monkeypatch.setattr(tools, "resolve_tool", lambda name: None)
    monkeypatch.setattr(tools, "platform_key", lambda *a: "darwin")
    monkeypatch.setattr(install_mod, "INSTALLERS",
                        {"ffmpeg": ("brew", "install", "ffmpeg")})

    monkeypatch.setattr(install_mod.shutil, "which",
                        lambda exe, *a, **k: "/opt/homebrew/bin/brew")
    with_brew = setup_mod.tool("ffmpeg", "ffmpeg", "ffmpeg", "why")
    assert "via Homebrew" in with_brew["detail"]

    monkeypatch.setattr(install_mod.shutil, "which", lambda exe, *a, **k: None)
    without = setup_mod.tool("ffmpeg", "ffmpeg", "ffmpeg", "why")
    assert "static build into the workspace" in without["detail"]


def test_install_is_one_at_a_time(client, stub_installers):
    stub_installers(("/bin/sh", "-c", "sleep 2"))
    first = client.post("/api/setup/install", json={"id": "ffmpeg"})
    assert first.status_code == 202 and first.json()["running"] is True
    second = client.post("/api/setup/install", json={"id": "ffmpeg"})
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "busy"
    assert "already running" in second.json()["error"]["message"]
    assert client.get("/api/setup/install").json()["running"] is True
    assert client.app.state.installer.wait(10.0)


def test_install_status_carries_the_output_tail(client, stub_installers):
    stub_installers(("/bin/sh", "-c", "echo fetching bottle; echo done"))
    client.post("/api/setup/install", json={"id": "ffmpeg"})
    assert client.app.state.installer.wait(10.0)
    body = client.get("/api/setup/install").json()
    assert body["running"] is False and body["id"] == "ffmpeg"
    assert body["tail"][0].startswith("$ /bin/sh")       # the command, echoed once
    assert "fetching bottle" in body["tail"] and "done" in body["tail"]


def test_install_reprobes_the_check_when_the_process_exits():
    """`brew` exiting 0 is a claim about `brew`, not about this machine's PATH,
    so the row the UI redraws is a fresh probe and never the exit code."""
    probed: list[str] = []

    def probe(id_):
        probed.append(id_)
        return {"id": id_, "label": "ffmpeg", "ok": True, "detail": "/opt/homebrew/bin/ffmpeg",
                "required": True}

    inst = install_mod.Installer(probe, recipes={"ffmpeg": ("/bin/sh", "-c", "echo hi")})
    inst.start("ffmpeg")
    assert inst.wait(10.0)
    status = inst.status()
    assert probed == ["ffmpeg"]
    assert status["running"] is False and status["ok"] is True and status["error"] is None
    assert status["check"]["ok"] is True and status["check"]["detail"].endswith("ffmpeg")
    assert "hi" in status["tail"]


def test_probe_returns_the_same_row_shape_the_report_does(client):
    """The status response's `check` is dropped straight into the UI's list, so
    a row one key short is a row that renders differently from its neighbours."""
    from dubbing_app import setup as setup_mod

    from dubbing_app import install as install_module

    row = next(c for c in client.get("/api/setup").json()["checks"] if c["id"] == "sox")
    probed = setup_mod.probe("sox")
    assert set(probed) == set(row)
    assert probed["installable"] is ("sox" in install_module.INSTALLERS)
    assert setup_mod.probe("model.demucs") is None and setup_mod.probe("nope") is None
    # The downloadable models answer too, with the same shape as their report row.
    model_row = next(c for c in client.get("/api/setup").json()["checks"]
                     if c["id"] == "model.translate")
    model_probed = setup_mod.probe("model.translate")
    assert set(model_probed) == set(model_row)
    assert model_probed["installable"] is True


def test_install_that_exits_nonzero_is_a_failure_with_its_output():
    inst = install_mod.Installer(lambda id_: None,
                                 recipes={"ffmpeg": ("/bin/sh", "-c", "echo boom >&2; exit 3")})
    inst.start("ffmpeg")
    assert inst.wait(10.0)
    status = inst.status()
    assert status["ok"] is False and "exited 3" in status["error"]
    assert "boom" in status["tail"]                      # stderr is folded into the tail
    assert status["check"] is None


def test_install_that_succeeds_but_changes_nothing_is_not_success():
    """The bottle landed somewhere this process cannot see. Reporting Ready on
    the strength of the exit code would send the user back to a stage that still
    dies halfway."""
    inst = install_mod.Installer(lambda id_: {"id": id_, "ok": False, "label": "ffmpeg",
                                              "detail": "ffmpeg not on PATH"},
                                 recipes={"ffmpeg": ("/bin/sh", "-c", "true")})
    inst.start("ffmpeg")
    assert inst.wait(10.0)
    status = inst.status()
    assert status["ok"] is False and "still not there" in status["error"]


def test_install_status_is_empty_before_anything_runs(client):
    body = client.get("/api/setup/install").json()
    assert body == {"running": False, "id": None, "ok": None, "error": None,
                    "tail": [], "check": None, "started": None, "finished": None}


# ---------------------------------------------------------------------------
# version stamp
# ---------------------------------------------------------------------------

def test_health_carries_the_commit(client):
    commit = client.get("/health").json()["commit"]
    assert commit is None or all(ch in "0123456789abcdef" for ch in commit)


def test_git_commit_prefers_the_baked_value(monkeypatch):
    from dubbing_app import setup as setup_mod

    monkeypatch.setenv("DUBBING_STUDIO_COMMIT", "deadbee")
    assert setup_mod.git_commit(refresh=True) == "deadbee"
    monkeypatch.delenv("DUBBING_STUDIO_COMMIT")
    monkeypatch.setattr(setup_mod, "REPO_ROOT", Path("/"))
    assert setup_mod.git_commit(refresh=True) is None


def test_create_project_probes_a_local_input_before_making_anything(client, outputs, tmp_path):
    # Missing file: named at the door, not as an ffmpeg error minutes later.
    r = client.post("/api/projects", json={"source": str(tmp_path / "nope.mp4")})
    assert r.status_code == 400
    assert "input not found" in envelope_of(r)["message"]
    assert sorted(p.name for p in outputs.iterdir()) == [NAME]
    # A readable file passes the probe (and the run is created).
    ok = tmp_path / "clip.mp4"
    ok.write_bytes(b"\x00" * 64)
    r = client.post("/api/projects", json={"source": str(ok), "name": "probe_ok"})
    assert r.status_code == 201


class TestRequestGate:
    """The two gates: loopback Host names without a token, the token with one."""

    def test_foreign_host_refused(self, client):
        # DNS rebinding serves the attacker's page from a name that RESOLVES to
        # 127.0.0.1 — same-origin to the browser, foreign in the Host header.
        r = client.get("/api/projects", headers={"host": "evil.example:53943"})
        assert r.status_code == 403

    def test_loopback_hosts_pass(self, client):
        for host in ("localhost:1", "127.0.0.1:9999", "[::1]:80", "localhost"):
            assert client.get("/api/projects", headers={"host": host}).status_code == 200

    def test_the_host_the_desktop_shell_really_sends_passes(self, client):
        # Every other test rides TestClient's "testserver" default, which is on
        # the trusted list for its own reason so the gate was never exercised
        # against the Host a browser pointed at the shell's port actually sends.
        assert client.get("/health", headers={"host": "127.0.0.1:4400"}).status_code == 200
        assert client.get("/health", headers={"host": "localhost:4400"}).status_code == 200
        # A name that merely *contains* a loopback name is not one.
        for host in ("127.0.0.1.evil.example:4400", "notlocalhost:4400", ""):
            assert client.get("/health", headers={"host": host}).status_code == 403

    @staticmethod
    def _tokened(tmp_path):
        """A fresh client each time: the first accepted request sets the cookie,
        and every later assertion would then be about the cookie, not the token."""
        from dubbing_app.app import create_app

        return TestClient(create_app(tmp_path, ui_dir="", token="s3cret"))

    def test_the_bearer_scheme_is_case_insensitive(self, tmp_path):
        # RFC 7235: the scheme is case-insensitive. `removeprefix("Bearer ")`
        # is not, so a correct token spelled `bearer …` was a 401.
        for header in ("Bearer s3cret", "bearer s3cret", "BEARER s3cret",
                       "bearer  s3cret "):
            with self._tokened(tmp_path) as c:
                assert c.get("/health", headers={"Authorization": header}).status_code == 200
        # The token value itself is not: only the scheme name is folded, and a
        # scheme that is not Bearer carries no token at all.
        for header in ("Bearer S3CRET", "Basic s3cret", "s3cret", "Bearer"):
            with self._tokened(tmp_path) as c:
                assert c.get("/health", headers={"Authorization": header}).status_code == 401

    def test_a_non_ascii_token_is_a_refusal_not_a_crash(self, tmp_path):
        # `compare_digest` raises TypeError on non-ASCII *str* input, and the
        # query string is attacker-supplied: `?token=%C3%A9` answered 500 with
        # the exception text instead of 401.
        with self._tokened(tmp_path) as c:
            assert c.get("/api/projects", params={"token": "é"}).status_code == 401
            assert c.get("/api/projects", params={"token": "s3crét"}).status_code == 401
            # Headers arrive as bytes and are decoded latin-1, so they can carry
            # a non-ASCII token too the same TypeError, through two more doors.
            assert c.get("/health",
                         headers={"Authorization": b"Bearer \xe9"}).status_code == 401
            assert c.get("/health",
                         headers={"Cookie": b"mambodubb_token=\xe9"}).status_code == 401
            # …and the real token still works from the same client afterwards.
            assert c.get("/api/projects", params={"token": "s3cret"}).status_code == 200

    def test_token_mode(self, tmp_path):
        from fastapi.testclient import TestClient

        from dubbing_app.app import create_app

        app = create_app(tmp_path, ui_dir="", token="s3cret")
        with TestClient(app) as c:
            assert c.get("/api/projects").status_code == 401
            assert c.get("/api/projects?token=wrong").status_code == 401
            first = c.get("/api/projects?token=s3cret")
            assert first.status_code == 200
            # The ?token= visit set the cookie; later requests ride it bare.
            assert c.get("/api/projects").status_code == 200
            assert c.get("/health", headers={"Authorization": "Bearer s3cret"},
                         cookies={}).status_code == 200


# ---------------------------------------------------------------------------
# CORS the only cross-origin callers that exist
# ---------------------------------------------------------------------------

ALLOWED_ORIGINS = ("tauri://localhost",          # the Tauri webview
                   "http://tauri.localhost",     # …and its Windows spelling
                   "http://localhost:5173",      # the Vite dev server
                   "http://127.0.0.1:4400")      # a second instance / a plain browser


@pytest.mark.parametrize("origin", ALLOWED_ORIGINS)
def test_the_shell_and_the_dev_server_may_read_the_api(client, origin):
    r = client.get("/health", headers={"Origin": origin})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == origin


@pytest.mark.parametrize("origin", ["https://evil.com", "http://evil.com",
                                    "http://localhost.evil.com", "null"])
def test_no_other_page_may_read_the_api(client, origin):
    # `["*"]` here meant any web page open in the user's browser could read and
    # drive this API: the server reads the filesystem and runs the pipeline, so
    # the wildcard was the hole, not the port. The request still runs the browser
    # is what refuses the response, and it refuses it for want of this header.
    r = client.get("/health", headers={"Origin": origin})
    assert "access-control-allow-origin" not in r.headers


def test_a_preflight_is_answered_without_a_credential(tmp_path):
    # A CORS preflight carries neither cookie nor token by design, so the gate
    # lets OPTIONS through and the CORS layer (added after it, therefore wrapping
    # outside it) answers. Gate it and every cross-origin PATCH dies at the door.
    app = create_app(tmp_path, ui_dir="", token="s3cret")
    with TestClient(app) as c:
        r = c.options("/api/projects/x/segments/y",
                      headers={"Origin": "tauri://localhost",
                               "Access-Control-Request-Method": "PATCH",
                               "Access-Control-Request-Headers": "content-type"})
        assert r.status_code == 200
        assert r.headers["access-control-allow-origin"] == "tauri://localhost"
        assert "PATCH" in r.headers["access-control-allow-methods"]
    # …and the preflight is not a way in: the real request still needs the token.
    with TestClient(create_app(tmp_path, ui_dir="", token="s3cret")) as c:
        assert c.get("/api/projects",
                     headers={"Origin": "tauri://localhost"}).status_code == 401


# ---------------------------------------------------------------------------
# a legacy fitted fallback slice
# ---------------------------------------------------------------------------

def test_a_stretched_fallback_slice_is_still_named_a_fallback(client, outputs):
    # tts's universal fallback is a `keep_*.wav` slice of the original span on a
    # dub-wanted line. Timelines placed before timeline/v15 rate-fitted it and
    # renamed it `fit_keep_*` it is still original audio wearing a dub's
    # verdict, and the UI has to say so or the line is baffling.
    from tests.conftest_app import write_wav

    workdir = outputs / NAME
    m = manifest.load(workdir)
    fitted = "clips/fit_keep_000000000340_1.100.wav"
    write_wav(workdir / fitted, 1.6)
    m["segments"][1]["place"]["clip"] = fitted
    manifest.save(workdir, m)

    segs = client.get(f"/api/projects/{NAME}/segments").json()["segments"]
    assert segs[1]["media"]["fallback"] is True
    # A kept line's slice is not a fallback it is the verdict; and an ordinary
    # fitted synthesis is not one either.
    assert segs[0]["media"]["fallback"] is False
    assert segs[2]["media"]["fallback"] is False


# ---------------------------------------------------------------------------
# a verdict flipped while a job ran
# ---------------------------------------------------------------------------

def _clipless_line(workdir):
    """A dub-wanted line with no clip yet the state a queued job starts from."""
    m = manifest.load(workdir)
    seg = m["segments"][1]
    seg.pop("tts", None), seg.pop("place", None)
    manifest.save(workdir, m)
    return seg["uid"]


def test_a_keep_flipped_while_a_job_ran_wins_over_the_clip_it_made(outputs):
    """The segment had no clip when the job started, the user flipped `keep`
    mid-job (which deletes `tts`/`place` still absent on disk), and the job
    then wrote a fresh clip. absent == absent, so the key-by-key merge kept the
    job's clip alongside the re-applied `keep=true` a kept line with a dub
    placed, which the mix plays and the UI (deriving from `keep`) never shows."""
    from dubbing_app.worker import Journal

    workdir = outputs / NAME
    uid = _clipless_line(workdir)
    journal = Journal(workdir, manifest.load(workdir))

    # The job's own work, on its private copy.
    ops.find(journal.m, uid).update(
        {"tts": {"clip": "clips/new.wav", "dur": 1.0, "verify": "ok"},
         "place": {"start": 0.34, "end": 1.34, "rate": 1.0, "drift": 0.0,
                   "clip": "clips/new.wav"}})

    disk = manifest.load(workdir)
    ops.set_keep(disk, uid, True)
    manifest.save(workdir, disk)
    assert "tts" not in ops.find(manifest.load(workdir), uid)

    assert journal.merge(manifest.load(workdir)) == [uid]
    merged = ops.find(journal.m, uid)
    assert merged["keep"] is True
    assert "tts" not in merged and "place" not in merged


def test_a_job_keeps_its_clip_when_the_verdict_did_not_move(outputs):
    from dubbing_app.worker import Journal

    workdir = outputs / NAME
    uid = _clipless_line(workdir)
    journal = Journal(workdir, manifest.load(workdir))
    ops.find(journal.m, uid).update({"tts": {"clip": "clips/new.wav", "dur": 1.0}})

    disk = manifest.load(workdir)
    ops.set_text(disk, uid, text_en="hand corrected")   # a field edit, not a flip
    manifest.save(workdir, disk)

    journal.merge(manifest.load(workdir))
    merged = ops.find(journal.m, uid)
    assert merged["tts"] == {"clip": "clips/new.wav", "dur": 1.0}
    assert merged["text_en"] == "hand corrected"


# ---------------------------------------------------------------------------
# model downloads through the install slot
# ---------------------------------------------------------------------------
# `snapshot_download` is the only thing here that would ever move bytes, so
# every test swaps it for a fake that writes a few hundred bytes to a tmp dir.
# What is asserted is the plumbing: which ids are accepted, what the progress
# looks like on the wire, and what the server believes once the fetch returns.


@pytest.fixture()
def hub_stub(monkeypatch, tmp_path):
    """The translate model, made tiny: its path points into tmp, and the fetch
    is a fake that writes 400 bytes, waits on `gate`, then writes 600 more.
    `step` fires once the first bytes are on disk, so a test can assert on
    mid-download progress without sleeping.

    The expected size shrinks with it, and has to: `setup.model_ready` now
    measures what arrived against what the table said to expect, so a 1 KB stand
    in for a 9.7 GB model would re-probe as `incomplete` and this fixture would
    be testing the size floor instead of the download plumbing.
    """
    from dubbing import translate
    from dubbing_app import setup as setup_mod

    local = tmp_path / "hub-model"
    gate = threading.Event()
    step = threading.Event()
    real_downloads = setup_mod.model_downloads

    def small_downloads():
        table = real_downloads()
        table["model.translate"] = {**table["model.translate"], "bytes": 1000}
        return table

    monkeypatch.setattr(setup_mod, "model_downloads", small_downloads)

    def fake_download(repo_id, local_dir, **kwargs):
        assert repo_id == translate.HUB_ID           # the table's id, never the client's
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "weights.bin").write_bytes(b"x" * 400)
        step.set()
        assert gate.wait(10.0)
        (Path(local_dir) / "rest.bin").write_bytes(b"y" * 600)

    monkeypatch.setattr(translate, "MODEL_PATH", local)
    monkeypatch.setattr(install_mod, "snapshot_download", fake_download)
    return {"local": local, "gate": gate, "step": step}


def test_setup_model_rows_carry_hub_and_download_size(client):
    """"Download" without a size is the blind spinner this feature was refused
    over; the row carries the repo and the approximate cost up front."""
    from dubbing import translate
    from dubbing_app import setup as setup_mod

    checks = {c["id"]: c for c in client.get("/api/setup").json()["checks"]}
    row = checks["model.translate"]
    assert row["hub"] == translate.HUB_ID
    assert row["download_bytes"] == setup_mod.model_downloads()["model.translate"]["bytes"]
    assert row["download_bytes"] > 0 and row["installable"] is True
    # The models no snapshot satisfies carry no hub and no button.
    assert "hub" not in checks["model.demucs"]
    assert checks["model.demucs"]["installable"] is False


def test_download_reports_progress_and_reprobes_the_row(client, hub_stub):
    from dubbing_app import setup as setup_mod

    total = setup_mod.model_downloads()["model.translate"]["bytes"]
    r = client.post("/api/setup/install", json={"id": "model.translate"})
    assert r.status_code == 202 and r.json()["running"] is True
    assert r.json()["tail"][0].startswith("$ snapshot_download(")
    assert any("resumes" in line for line in r.json()["tail"])   # say it up front

    assert hub_stub["step"].wait(10.0)
    body = client.get("/api/setup/install").json()
    assert body["running"] is True
    assert body["bytes_done"] == 400                 # the directory, measured
    assert body["bytes_total"] == total              # the table's estimate

    hub_stub["gate"].set()
    assert client.app.state.installer.wait(10.0)
    body = client.get("/api/setup/install").json()
    assert body["running"] is False and body["ok"] is True and body["error"] is None
    assert body["bytes_done"] == 1000
    # The row the UI redraws is a fresh stat of the directory, not the fetch's
    # word for it — same rule as the tools.
    assert body["check"]["id"] == "model.translate" and body["check"]["ok"] is True
    assert body["check"]["installable"] is True


def test_download_holds_the_one_install_slot(client, hub_stub, stub_installers):
    """One slot for tools and models alike: two downloads share one disk and
    the screen has one progress row."""
    stub_installers()
    client.post("/api/setup/install", json={"id": "model.translate"})
    assert hub_stub["step"].wait(10.0)
    for id_ in ("ffmpeg", "model.translate"):
        r = client.post("/api/setup/install", json={"id": id_})
        assert r.status_code == 409 and r.json()["error"]["code"] == "busy", id_
    hub_stub["gate"].set()
    assert client.app.state.installer.wait(10.0)


def test_tool_install_status_carries_no_byte_fields(client, stub_installers):
    """`bytes_done` against a brew would be a number with no meaning; the keys
    exist only while the slot holds a download."""
    stub_installers()
    client.post("/api/setup/install", json={"id": "ffmpeg"})
    assert client.app.state.installer.wait(10.0)
    body = client.get("/api/setup/install").json()
    assert "bytes_done" not in body and "bytes_total" not in body


def test_download_failure_keeps_partials_and_promises_resume(monkeypatch, tmp_path):
    """A torn-off download is not a wasted download; the error says so, because
    "failed" alone reads as "start over" to a user staring at 81 GB."""
    def explode(**kwargs):
        raise RuntimeError("connection reset by peer")

    monkeypatch.setattr(install_mod, "snapshot_download", explode)
    inst = install_mod.Installer(lambda id_: None,
                                 downloads={"model.x": {"hub": "org/x",
                                                        "path": tmp_path / "x",
                                                        "bytes": 10}})
    inst.start("model.x")
    assert inst.wait(10.0)
    status = inst.status()
    assert status["ok"] is False
    assert "RuntimeError" in status["error"] and "resumes" in status["error"]


def test_download_that_completes_but_fails_its_probe_is_not_success(monkeypatch, tmp_path):
    """Same honesty rule as the tools: the fetch returning is a claim about the
    fetch, and the check is the only thing allowed to say Ready."""
    monkeypatch.setattr(install_mod, "snapshot_download", lambda **kw: None)
    inst = install_mod.Installer(lambda id_: {"id": id_, "ok": False, "label": "x",
                                              "detail": "missing"},
                                 downloads={"model.x": {"hub": "org/x",
                                                        "path": tmp_path / "x",
                                                        "bytes": 10}})
    inst.start("model.x")
    assert inst.wait(10.0)
    status = inst.status()
    assert status["ok"] is False and "still fails" in status["error"]


# ---------------------------------------------------------------------------
# restoring the bundled diarization weights (no token, no network)
# ---------------------------------------------------------------------------
# The one install here that downloads nothing. Upstream is gated, so the button
# reads the copy that shipped with the app and proves it arrived whole.


def a_weights_dir(root: Path, *, damaged: bool = False) -> Path:
    """A miniature of `third_party/pyannote-speaker-diarization-community-1`:
    a couple of files and the SHA256SUMS that vouches for them."""
    import hashlib

    root.mkdir(parents=True, exist_ok=True)
    (root / "segmentation").mkdir(exist_ok=True)
    files = {"config.yaml": b"version: 1\n",
             "segmentation/pytorch_model.bin": b"weights" * 100}
    lines = []
    for rel, blob in files.items():
        (root / rel).write_bytes(blob)
        digest = hashlib.sha256(blob if not damaged else b"other").hexdigest()
        lines.append(f"{digest}  {rel}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    return root


@pytest.fixture()
def diarization(tmp_path, monkeypatch):
    """A source to restore from and a workspace to restore into, with every
    real-machine route switched off: no /Applications payload and no checkout,
    so what a test asserts is its own fixture and not the developer's laptop."""
    target = tmp_path / "workspace" / "third_party" / "pyannote-speaker-diarization-community-1"
    monkeypatch.setattr(install_mod, "diarization_target", lambda: target)
    monkeypatch.setattr(install_mod, "DIARIZATION_BUNDLE_ROOTS", ())
    monkeypatch.delenv(install_mod.DIARIZATION_SOURCE_ENV, raising=False)
    return {"target": target, "source": tmp_path / "payload"}


def test_diarization_has_no_route_when_nothing_carries_a_copy(diarization):
    """A stripped install: no payload, no checkout, nothing to copy. The honest
    answer is no button, not a button that 400s."""
    assert install_mod.diarization_source() is None
    assert install_mod.installable(install_mod.DIARIZATION_ID) is False
    assert install_mod.diarization_command() is None
    assert install_mod.route(install_mod.DIARIZATION_ID) is None


def test_diarization_restores_from_the_bundled_copy_and_verifies_it(diarization,
                                                                    monkeypatch):
    """The whole point of the row: the weights come back with no account, no
    token and no network, and the copy is checked against the manifest that
    shipped beside it rather than believed."""
    source = a_weights_dir(diarization["source"])
    monkeypatch.setenv(install_mod.DIARIZATION_SOURCE_ENV, str(source))
    assert install_mod.diarization_source() == ("copy", source)
    assert install_mod.installable(install_mod.DIARIZATION_ID) is True
    assert "copying the weights that shipped" in install_mod.route(
        install_mod.DIARIZATION_ID)

    lines: list[str] = []
    install_mod.restore_diarization(lines.append)
    target = diarization["target"]
    assert (target / "config.yaml").read_bytes() == b"version: 1\n"
    assert (target / "segmentation" / "pytorch_model.bin").is_file()
    assert any("verified 2 files" in line for line in lines)
    # Nothing is left half-written beside it.
    assert not (target.parent / f"{target.name}.incoming").exists()


def test_a_damaged_copy_is_refused_and_the_workspace_is_left_alone(diarization,
                                                                  monkeypatch):
    """A full disk or a truncated payload must not land a broken pipeline that
    a run discovers fifty lines in as a tensor shape complaint."""
    source = a_weights_dir(diarization["source"], damaged=True)
    monkeypatch.setenv(install_mod.DIARIZATION_SOURCE_ENV, str(source))
    with pytest.raises(RuntimeError, match="does not match SHA256SUMS"):
        install_mod.restore_diarization(lambda line: None)
    assert not diarization["target"].exists()
    assert not (diarization["target"].parent
                / f"{diarization['target'].name}.incoming").exists()


def test_the_diarization_row_is_installable_and_says_what_the_button_does(client,
                                                                         diarization,
                                                                         monkeypatch):
    """The row, end to end: a button, the command as data for anyone who would
    rather type it, and no `hub`, because there is nothing to download and the
    one repo that would be is the gated one."""
    from dubbing import segments
    from dubbing_app import setup as setup_mod

    source = a_weights_dir(diarization["source"])
    monkeypatch.setenv(install_mod.DIARIZATION_SOURCE_ENV, str(source))
    monkeypatch.setattr(segments, "DIARIZATION_DIR", diarization["target"])

    row = {c["id"]: c for c in client.get("/api/setup").json()["checks"]}[
        install_mod.DIARIZATION_ID]
    assert row["ok"] is False and row["severity"] == "degrades"
    assert row["installable"] is True
    assert row["fix"] == f"cp -R {source}/. {diarization['target']}"
    assert f"`{row['fix']}`" in row["detail"]
    assert "hub" not in row
    # And "install everything" picks it up: it is missing, fixable from here and
    # not optional, which is exactly the plan's own rule.
    plan = setup_mod.install_plan(client.get("/api/setup").json())
    assert install_mod.DIARIZATION_ID in {item["id"] for item in plan}


def test_the_diarization_row_says_the_honest_alternative_when_there_is_no_copy(
        client, diarization, monkeypatch):
    """No payload and no checkout: the row must not invent a command. What it
    offers instead is the two things that do work, and neither is a login."""
    from dubbing import segments

    monkeypatch.setattr(segments, "DIARIZATION_DIR", diarization["target"])
    row = {c["id"]: c for c in client.get("/api/setup").json()["checks"]}[
        install_mod.DIARIZATION_ID]
    assert row["ok"] is False and row["installable"] is False and "fix" not in row
    assert "reinstall the app" in row["detail"]
    assert segments.DIARIZATION_HUB_ENV in row["detail"]
    # Never a demand for an account, on the row that used to be the reason for one.
    assert "token" not in row["detail"].lower()


def test_restoring_diarization_runs_through_the_one_install_slot(diarization,
                                                                monkeypatch):
    """Same slot, same re-probe, same verdict shape as a download or a brew:
    there is no second install path to keep honest."""
    from dubbing_app import setup as setup_mod

    source = a_weights_dir(diarization["source"])
    monkeypatch.setenv(install_mod.DIARIZATION_SOURCE_ENV, str(source))
    probed: list[str] = []

    def probe(id_):
        probed.append(id_)
        return {"id": id_, "label": "Diarization", "ok": True, "detail": "",
                "severity": "degrades", "required": False}

    inst = install_mod.Installer(probe, recipes={})
    inst.start(install_mod.DIARIZATION_ID)
    assert inst.wait(10.0)
    status = inst.status()
    assert status["ok"] is True and status["error"] is None
    assert probed == [install_mod.DIARIZATION_ID]
    # No byte fields: nothing was downloaded, and a progress denominator for a
    # local copy would be a spinner pretending to be a download.
    assert "bytes_total" not in status
    assert (diarization["target"] / "config.yaml").is_file()


# ---------------------------------------------------------------------------
# install everything: POST|DELETE /api/setup/install_all
# ---------------------------------------------------------------------------
# One button for a screen full of red rows. Nothing here installs anything
# either: the plan is a pure function of a report, and the queue is driven over
# an `Installer` whose recipes are shell stubs, so what is asserted is the
# order, the one-slot rule, the cancel boundary and the wire shape.


def a_report(*rows):
    """A `setup.report`-shaped body from (id, ok, severity, installable) tuples."""
    checks = []
    for id_, ok, sev, inst, *rest in rows:
        checks.append({"id": id_, "label": id_.upper(), "ok": ok, "severity": sev,
                       "required": sev == "blocking", "installable": inst,
                       "detail": "", **(rest[0] if rest else {})})
    return {"ok": all(c["ok"] for c in checks if c["required"]), "checks": checks}


def test_install_plan_queues_only_the_missing_rows_the_app_can_fix():
    plan = setup_mod.install_plan(a_report(
        ("ffmpeg", False, "blocking", True),
        ("uv", True, "blocking", True),               # already there — not re-run
        ("model.g2p.he", False, "degrades", False),   # nothing to install
        ("model.demucs", False, "optional", False),   # fetches its own cache
        ("model.tts.ko", False, "optional", True),    # a language nobody asked for
        ("model.lid", False, "degrades", True, {"download_bytes": 100}),
    ))
    assert [item["id"] for item in plan] == ["ffmpeg", "model.lid"]
    # Blocking before degrades: the run has to work before it has to be good.
    assert plan[0]["bytes"] == 0 and plan[1]["bytes"] == 100
    assert plan[1]["label"] == "MODEL.LID"


def test_install_plan_orders_required_first_and_keeps_the_screens_order():
    plan = setup_mod.install_plan(a_report(
        ("model.lid", False, "degrades", True),
        ("model.translate", False, "blocking", True),
        ("ffmpeg", False, "blocking", True),
    ))
    assert [item["id"] for item in plan] == ["model.translate", "ffmpeg", "model.lid"]


def test_install_plan_never_queues_anything_that_needs_a_credential(client):
    """No check wants a token, and no gated repo is in the download table so the
    queue can never stop half way to ask the user for something. Asserted
    against the real report, because this is a claim about the actual tables,
    not a synthetic one."""
    from dubbing import segments

    plan = setup_mod.install_plan(client.get("/api/setup").json())
    ids = {item["id"] for item in plan}
    assert not any("token" in id_ for id_ in ids)
    downloads = setup_mod.model_downloads()
    assert ids <= set(install_mod.INSTALLERS) | set(downloads)
    gated = segments.DIARIZATION_MODEL
    assert not any(spec["hub"] == gated for spec in downloads.values())
    # Diarization is not downloadable at all, and that is the whole point: the
    # weights ship with the app, so the only gated repo in the tree is never
    # something the queue could reach for.
    assert "model.diarization" not in downloads
    # Its row still has a button, and that is not a contradiction: the button
    # restores a copy this machine already carries (`install.diarization_source`),
    # verifies it against SHA256SUMS, and touches no repo, gated or otherwise.
    row = {c["id"]: c for c in client.get("/api/setup").json()["checks"]}["model.diarization"]
    assert "hub" not in row and "download_bytes" not in row
    assert row["installable"] is (install_mod.diarization_source() is not None)


@pytest.fixture()
def stub_plan(monkeypatch):
    """Point `POST /api/setup/install_all` at a plan of shell stubs.

    The ids are not real checks, which is the point: `setup.probe` answers None
    for them, so the queue's verdict is the stub's exit code and nothing about
    the developer's own machine can decide whether a test passes.
    """
    def use(*specs):
        recipes = {id_: tuple(argv) for id_, argv in specs}
        monkeypatch.setattr(install_mod, "INSTALLERS", recipes)
        plan = [{"id": id_, "label": id_, "bytes": 1000} for id_, _ in specs]
        monkeypatch.setattr(setup_mod, "install_plan", lambda report_: list(plan))
        return plan
    return use


def test_install_all_runs_the_whole_plan_through_the_one_slot(client, stub_plan, tmp_path):
    stub_plan(("tool.a", ("/bin/sh", "-c", f"echo a >> {tmp_path / 'order'}")),
              ("tool.b", ("/bin/sh", "-c", f"echo b >> {tmp_path / 'order'}")))
    r = client.post("/api/setup/install_all")
    assert r.status_code == 202
    queue = r.json()["queue"]
    assert queue["running"] is True and queue["total"] == 2
    assert [item["id"] for item in queue["items"]] == ["tool.a", "tool.b"]

    assert client.app.state.install_queue.wait(30.0)
    body = client.get("/api/setup/install").json()
    assert body["queue"]["running"] is False and body["queue"]["failed"] == []
    assert body["queue"]["pos"] == 2 and body["queue"]["remaining_bytes"] == 0
    # One at a time, in the order the plan named — the whole reason this is a
    # queue and not two POSTs the client fires off together.
    assert (tmp_path / "order").read_text().split() == ["a", "b"]


def test_install_all_is_a_no_op_answer_when_nothing_is_missing(client, monkeypatch):
    monkeypatch.setattr(setup_mod, "install_plan", lambda report_: [])
    r = client.post("/api/setup/install_all")
    assert r.status_code == 202
    # Not a 400 and not an error: "install everything" on a machine with nothing
    # missing is a queue that ran nothing, and the screen has no button anyway.
    assert r.json().get("queue") is None
    assert r.json()["running"] is False


def test_a_second_install_all_answers_the_running_queue(client, stub_plan):
    stub_plan(("tool.a", ("/bin/sh", "-c", "sleep 1")),
              ("tool.b", ("/bin/sh", "-c", "true")))
    first = client.post("/api/setup/install_all")
    second = client.post("/api/setup/install_all")
    # Idempotent: one gesture repeated is not an error, and a 409 here would
    # teach the user that the button they just pressed did something wrong.
    assert second.status_code == 202
    assert second.json()["queue"]["running"] is True
    assert second.json()["queue"]["total"] == first.json()["queue"]["total"]
    assert second.json()["queue"]["started"] == first.json()["queue"]["started"]
    assert client.app.state.install_queue.wait(30.0)


def test_install_all_waits_for_an_install_started_from_a_row(client, stub_plan):
    """The one refusal. Half a queue that dies on its first item is worse than
    being told to wait for the install already running."""
    stub_plan(("tool.a", ("/bin/sh", "-c", "sleep 1")))
    assert client.post("/api/setup/install", json={"id": "tool.a"}).status_code == 202
    r = client.post("/api/setup/install_all")
    assert r.status_code == 409 and envelope_of(r)["code"] == "busy"
    assert client.app.state.installer.wait(30.0)


def test_cancel_stops_after_the_current_item(client, stub_plan, tmp_path):
    started, never = tmp_path / "started", tmp_path / "never"
    stub_plan(("tool.a", ("/bin/sh", "-c", f"touch {started}; sleep 1")),
              ("tool.b", ("/bin/sh", "-c", f"touch {never}")))
    client.post("/api/setup/install_all")
    for _ in range(100):                      # the first item is in flight
        if started.exists():
            break
        time.sleep(0.05)
    r = client.delete("/api/setup/install_all")
    assert r.status_code == 200 and r.json()["queue"]["cancelled"] is True
    # Still running: cancelling does not kill the item in flight, because a
    # half-killed package manager is a broken prefix.
    assert r.json()["queue"]["running"] is True
    assert client.app.state.install_queue.wait(30.0)
    queue = client.get("/api/setup/install").json()["queue"]
    assert queue["running"] is False and queue["cancelled"] is True
    assert queue["pos"] == 1 and queue["total"] == 2
    # The second item was never begun, and the bytes it would have cost are
    # still counted as remaining — the header says what is left to do.
    assert not never.exists()
    assert queue["remaining_bytes"] == 1000
    # …and the first one did finish. That is what "after the current item" means.
    assert started.exists() and queue["failed"] == []


def test_cancelling_nothing_is_not_an_error(client):
    r = client.delete("/api/setup/install_all")
    assert r.status_code == 200


def test_a_failed_item_does_not_end_the_queue(client, stub_plan, tmp_path):
    """Nine downloads and one dead mirror should leave eight models on disk. The
    failure is named so the header can say so; the row itself says why."""
    stub_plan(("tool.a", ("/bin/sh", "-c", "echo mirror is down; exit 1")),
              ("tool.b", ("/bin/sh", "-c", f"touch {tmp_path / 'b'}")))
    client.post("/api/setup/install_all")
    assert client.app.state.install_queue.wait(30.0)
    queue = client.get("/api/setup/install").json()["queue"]
    assert queue["failed"] == ["tool.a"] and queue["pos"] == 2
    assert (tmp_path / "b").exists()


def test_the_queue_block_tracks_the_item_in_flight(client, monkeypatch, hub_stub):
    """The progress the header draws is the slot's own — one poll carries both
    which item (`queue.pos`) and how far into it (`bytes_done`)."""
    monkeypatch.setattr(setup_mod, "install_plan",
                        lambda report_: [{"id": "model.translate", "label": "Translation",
                                          "bytes": 1000},
                                         {"id": "model.other", "label": "Other",
                                          "bytes": 5000}])
    client.post("/api/setup/install_all")
    assert hub_stub["step"].wait(10.0)
    body = client.get("/api/setup/install").json()
    assert body["id"] == "model.translate" and body["bytes_done"] == 400
    assert body["queue"]["pos"] == 0
    # 1000 + 5000 queued, 400 of the first already on disk.
    assert body["queue"]["remaining_bytes"] == 5600
    hub_stub["gate"].set()
    client.delete("/api/setup/install_all")
    assert client.app.state.install_queue.wait(30.0)


def test_a_row_button_clears_the_finished_queue(client, stub_plan):
    """The header line describes the last queue; a hand-started install is the
    end of that story, and leaving it up would report a failure the user is in
    the middle of fixing by hand."""
    stub_plan(("tool.a", ("/bin/sh", "-c", "exit 1")))
    client.post("/api/setup/install_all")
    assert client.app.state.install_queue.wait(30.0)
    assert client.get("/api/setup/install").json()["queue"]["failed"] == ["tool.a"]
    r = client.post("/api/setup/install", json={"id": "tool.a"})
    assert r.status_code == 202 and r.json().get("queue") is None
    assert client.app.state.installer.wait(30.0)


# ---------------------------------------------------------------------------
# The workspace `.env`, written by the server
# ---------------------------------------------------------------------------
# Everything here redirects `setup.env_path()` into tmp_path by pointing the
# module's REPO_ROOT there — env_path resolves it per call, so a client built
# before the monkeypatch still writes where the test looks. The two token env
# vars are cleared first so a developer machine with HF_TOKEN exported cannot
# influence anything below.
#
# There used to be a `POST|DELETE /api/setup/hf_token` lane here, with a paste
# box and a Save on the Setup screen behind it. Both are gone: the diarization
# weights ship with the app, so the token was a credential the product had no
# use for, and an endpoint that writes one into `.env` for a row nobody is
# looking at is surface with no reader. `HF_TOKEN` and `DUB_DIARIZATION_HUB` in
# `.env` still work for anyone who wants the gated upstream repo, which is what
# `.env.example` documents and what `dubbing/segments.py` reads.

from dubbing_app import setup as setup_mod  # noqa: E402


@pytest.fixture()
def env_home(tmp_path, monkeypatch):
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(setup_mod, "REPO_ROOT", tmp_path)
    return tmp_path / ".env"


def test_no_token_endpoint_is_left_to_write_a_credential(client, env_home):
    """The save and the clear are gone, and gone means 404/405, not a quiet 200.

    A route kept "just in case" after its only caller was deleted is the way a
    removed feature comes back as an undocumented one: it writes a real
    credential to a real file with nothing on any screen saying it happened.
    Nothing else in `.env` is touched by their absence either, which is the
    second assertion: an untouched file is the proof that no handler ran.
    """
    env_home.write_text("OTHER=keep me\n", encoding="utf-8")
    for call in (lambda: client.post("/api/setup/hf_token", json={"token": "hf_abcdefgh"}),
                 lambda: client.delete("/api/setup/hf_token")):
        assert call().status_code in (404, 405)
    assert env_home.read_text() == "OTHER=keep me\n"


def test_low_vram_row_is_a_setting_not_a_finding(client, env_home, monkeypatch):
    """Green in both positions, and optional in both. The row says which
    translator weights this machine will load; a red tick beside a deliberate
    choice would send people looking for something to install."""
    monkeypatch.delenv("DUBBING_LOW_VRAM", raising=False)
    row = next(c for c in client.get("/api/setup").json()["checks"]
               if c["id"] == "low_vram")
    assert row["ok"] is True and row["severity"] == "optional"
    assert row["required"] is False and row["installable"] is False
    # Nobody has said, so the machine decided — and the row says which way.
    assert row["source"] == "auto"
    assert isinstance(row["enabled"], bool)
    assert row["detail"].startswith("on: " if row["enabled"] else "off: ")


def test_low_vram_toggle_writes_the_env_the_pipeline_reads(client, env_home, monkeypatch):
    monkeypatch.delenv("DUBBING_LOW_VRAM", raising=False)
    env_home.write_text("# mine\nOTHER=keep me\n")
    r = client.post("/api/setup/low_vram", json={"enabled": True})
    assert r.status_code == 200
    row = r.json()
    assert row["id"] == "low_vram" and row["enabled"] is True
    assert row["source"] == "env_file"
    lines = env_home.read_text().splitlines()
    assert lines[:2] == ["# mine", "OTHER=keep me"]      # hand-written lines survive
    assert lines[-1] == "DUBBING_LOW_VRAM=1"
    # Off is written, not merely unwritten: "off" and "nobody said" are
    # different answers, and only the first survives a machine the autodetect
    # would have decided the other way about.
    assert client.post("/api/setup/low_vram", json={"enabled": False}).json()["enabled"] is False
    lines = env_home.read_text().splitlines()
    assert lines.count("DUBBING_LOW_VRAM=0") == 1        # replaced, never appended
    assert "DUBBING_LOW_VRAM=1" not in lines
    # Strict body, like every other endpoint here.
    assert client.post("/api/setup/low_vram", json={"enabled": True, "x": 1}).status_code == 400


def test_low_vram_env_var_beats_the_file_and_the_row_says_so(client, env_home, monkeypatch):
    """A variable exported in the server's environment is what `load_dotenv`
    will *not* override, so a switch that claimed to have changed it would be
    lying. The row reports the value that will actually apply, and names where
    it came from; the UI reads `source` to withdraw the control."""
    client.post("/api/setup/low_vram", json={"enabled": False})
    monkeypatch.setenv("DUBBING_LOW_VRAM", "1")
    row = next(c for c in client.get("/api/setup").json()["checks"]
               if c["id"] == "low_vram")
    assert row["enabled"] is True and row["source"] == "env"
    assert "DUBBING_LOW_VRAM" in row["detail"]


def test_low_vram_save_then_probe_agree(client, env_home, monkeypatch):
    monkeypatch.delenv("DUBBING_LOW_VRAM", raising=False)
    saved = client.post("/api/setup/low_vram", json={"enabled": True}).json()
    probed = next(c for c in client.get("/api/setup").json()["checks"]
                  if c["id"] == "low_vram")
    # `installable` is `report`'s to add, and this row is never installable —
    # every other field has to match, or the switch's receipt disagrees with
    # the checklist and the user goes hunting for a difference that is not one.
    assert saved == {k: v for k, v in probed.items() if k != "installable"}
    assert probed["installable"] is False


def test_low_vram_moves_the_translator_row_to_the_weights_it_will_load(
        client, env_home, monkeypatch):
    """With the mode on, the checklist must describe the mxfp4 build. A row that
    kept reporting the 6-bit one would call the machine ready for a model the
    run never opens, and its Download button would fetch 9.7 GB the run ignores
    before quietly fetching 6.4 GB more on its own."""
    from dubbing import translate

    monkeypatch.delenv("DUBBING_LOW_VRAM", raising=False)
    row = next(c for c in client.get("/api/setup").json()["checks"]
               if c["id"] == "model.translate")
    assert row["hub"] == translate.HUB_ID

    client.post("/api/setup/low_vram", json={"enabled": True})
    row = next(c for c in client.get("/api/setup").json()["checks"]
               if c["id"] == "model.translate")
    assert row["hub"] == translate.LOW_VRAM_HUB_ID
    assert row["download_bytes"] < 9_700_000_000
    assert str(translate.LOW_VRAM_MODEL_PATH) == row["path"]


def test_setup_still_loads_no_torch(client, env_home):
    """The whole module's first rule, and the low-VRAM row is the one that was
    tempted to break it: reading VRAM through torch would put half a gigabyte
    into the server process to answer a question about a checkbox."""
    client.get("/api/setup")
    assert "torch" not in sys.modules


def test_a_token_in_the_env_file_changes_nothing_on_the_checklist(client, env_home):
    """`.env` is still where `HF_TOKEN` goes, and the checklist still has
    nothing to say about it.

    This is the line between the UI that was removed and the capability that was
    not. A user who wants the gated upstream diarization repo writes the token
    and `DUB_DIARIZATION_HUB` into this file by hand and the pipeline reads
    both; Setup neither congratulates them for it nor lists its absence as
    something to fix.
    """
    env_home.write_text("HF_TOKEN=hf_bymyownhand\n", encoding="utf-8")
    body = client.get("/api/setup").json()
    assert "hf_bymyownhand" not in json.dumps(body)
    assert not any("token" in c["id"] for c in body["checks"])
