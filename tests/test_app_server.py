"""Studio server tests routing, the error envelope, media, jobs, NDJSON.

No models and no real pipeline: the run directory is built by hand
(`conftest_app.make_project`) and every job runs through an injected fake runner,
so these are as fast and as deterministic as the rest of the suite.
"""

from __future__ import annotations

import asyncio
import json
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

    body = {"source": "https://youtu.be/abc", "tgt_lang": "ru", "duration": 60,
            "name": "movieproj", "context": "a note", "genre": "movie",
            "register": "dialogue", "transcript": "asr", "tts_model": "1.7b",
            "dub_foreign": True, "captions": "caps.json3"}
    r = client.post("/api/projects", json=body)
    assert r.status_code == 201 and r.json()["project"]["name"] == "movieproj"
    assert wait_until(lambda: fake.calls, 5.0)

    kind, project, payload = fake.calls[0]
    assert (kind, project) == ("run", "movieproj")
    args = cli.parse_args(ops.full_run_argv(Path(payload["workdir"]), payload["source"]))
    assert (args.source, args.tgt, args.duration) == ("https://youtu.be/abc", "ru", 60)
    assert (args.genre, args.register, args.transcript) == ("movie", "dialogue", "asr")
    assert args.tts_model == "1.7b" and args.dub_foreign is True
    assert args.context == "a note" and str(args.captions) == "caps.json3"
    # …and the options are on the manifest too, for every later edit job.
    stored = manifest.load(outputs / "movieproj")["source"]["app_opts"]
    assert stored["genre"] == "movie" and stored["tts_model"] == "1.7b"


def test_create_project_refuses_an_option_the_cli_cannot_take(client, outputs):
    """argparse would reject it, but only in the job child, minutes later, as a
    usage dump after the project directory and its manifest already exist. There
    is no way back from that project: its one job can never succeed."""
    r = client.post("/api/projects", json={"source": "x.mp4", "genre": "banana"})
    assert r.status_code == 400 and envelope_of(r)["code"] == "invalid_request"
    assert sorted(p.name for p in outputs.iterdir()) == [NAME]


@pytest.mark.parametrize("flag,dest,literal", [
    ("--genre", "genre", app_mod.Genre),
    ("--register", "register", app_mod.Register),
    ("--transcript", "transcript", app_mod.Transcript),
    ("--tts-model", "tts_model", app_mod.TtsModel),
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
    for route in ("def split_segment", "def merge_segment", "def patch_segment"):
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
    for wanted in ("ffmpeg", "sox", "hf_token", "model.translate", "model.tts.1.7b",
                   "model.asr.en", "model.lid", "model.demucs", "disk"):
        assert wanted in by_id, wanted
    for c in body["checks"]:
        assert {"id", "label", "ok", "detail", "required"} <= set(c)
        assert isinstance(c["ok"], bool) and isinstance(c["detail"], str) and c["label"]
    # `ok` is the conjunction of the required checks only; Demucs and free space
    # are informational and must never gate first run.
    assert by_id["model.demucs"]["required"] is False
    assert by_id["disk"]["required"] is False
    assert body["ok"] == all(c["ok"] for c in body["checks"] if c["required"])


def test_setup_grades_every_check_and_required_is_derived_from_it(client):
    """`required` has two values and the question has three: a missing HF token
    (every speaker in the video collapsed into one) was reported exactly as
    informationally as a Korean checkpoint a Hebrew→English run never opens."""
    from dubbing_app import setup as setup_mod

    checks = client.get("/api/setup").json()["checks"]
    for c in checks:
        assert c["severity"] in setup_mod.SEVERITIES, c["id"]
        # One source of truth: nothing may be blocking and not required, or the
        # reverse the old flag is now a view of the new grade.
        assert c["required"] is (c["severity"] == setup_mod.BLOCKING), c["id"]

    by_id = {c["id"]: c["severity"] for c in checks}
    from dubbing import tts

    assert by_id["ffmpeg"] == by_id["sox"] == by_id["uv"] == "blocking"
    assert by_id["model.translate"] == by_id["model.asr.en"] == "blocking"
    assert by_id[f"model.tts.{tts.DEFAULT_TTS_MODEL}"] == "blocking"
    # The run works and is worse: one voice for everybody, no third language found.
    assert by_id["hf_token"] == by_id["model.lid"] == "degrades"
    # Irrelevant until asked for, or self-downloading.
    for optional in ("model.asr.he", "model.asr.tgt", "model.tts.he", "model.g2p.he",
                     "model.demucs", "disk"):
        assert by_id[optional] == "optional", optional


def test_setup_names_the_stage_a_blocking_check_would_kill(client):
    """"Runs will fail" is true and useless. The screen offers "Skip anyway —
    runs will fail at fetch", and only the server knows which stage that is."""
    checks = {c["id"]: c for c in client.get("/api/setup").json()["checks"]}
    assert checks["ffmpeg"]["stage"] == "fetch"
    assert checks["model.translate"]["stage"] == "translate"
    assert checks["model.asr.en"]["stage"] == "tts"
    # Only where it means something: a stage on an optional row reads as urgency.
    assert "stage" not in checks["disk"] and "stage" not in checks["hf_token"]
    # And `None` is an honest answer for a blocking check with no stage to name:
    # `uv` is how this project's environment is built, but a running server
    # spawns its job child with `sys.executable` and never shells out to it, so
    # picking a stage for it would be a guess dressed as a fact.
    assert checks["uv"]["severity"] == "blocking" and checks["uv"]["stage"] is None


def test_setup_token_row_names_the_env_file_by_absolute_path(monkeypatch, tmp_path):
    """"Put HF_TOKEN in .env" is a scavenger hunt on a machine with three
    checkouts; only this process knows which `.env` it will read."""
    from dubbing_app import setup as setup_mod

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    env = tmp_path / ".env"
    row = setup_mod.hf_token_check(env)
    assert row["ok"] is False and row["severity"] == "degrades"
    assert f"`{env}`" in row["detail"]          # backticked, so the UI can copy it
    assert Path(row["path"]).is_absolute()


def test_setup_model_paths_come_from_the_pipeline(client):
    """The check reads `dubbing`'s own constants, so it cannot drift from what
    the pipeline actually opens."""
    from dubbing import transcript, translate, tts

    by_id = {c["id"]: c for c in client.get("/api/setup").json()["checks"]}
    assert by_id["model.translate"]["path"] == str(translate.MODEL_PATH)
    assert by_id["model.asr.en"]["path"] == str(transcript.EN_ASR_MODEL)
    assert by_id["model.lid"]["path"] == str(transcript.LID_MODEL)
    assert by_id["model.asr.he"]["path"] == str(transcript.WHISPER_MODEL)
    for key, spec in tts.TTS_MODELS.items():
        assert by_id[f"model.tts.{key}"]["path"].endswith(spec["dir"])
    assert by_id[f"model.tts.{tts.DEFAULT_TTS_MODEL}"]["required"] is True


def test_setup_reports_token_presence_never_the_value(client, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_supersecret_value")
    body = client.get("/api/setup").json()
    check = next(c for c in body["checks"] if c["id"] == "hf_token")
    assert check["ok"] is True and check["required"] is False
    assert "hf_supersecret_value" not in json.dumps(body)

    monkeypatch.delenv("HF_TOKEN")
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    from dubbing_app import setup as setup_mod

    monkeypatch.setattr(setup_mod, "REPO_ROOT", Path("/nonexistent"))
    check = next(c for c in client.get("/api/setup").json()["checks"] if c["id"] == "hf_token")
    assert check["ok"] is False


def test_setup_reads_token_from_env_file(tmp_path):
    from dubbing_app import setup as setup_mod

    env = tmp_path / ".env"
    env.write_text("# comment\nHF_TOKEN='hf_from_file'\n", encoding="utf-8")
    check = setup_mod.hf_token_check(env)
    assert check["ok"] is True and "hf_from_file" not in json.dumps(check)
    env.write_text("HF_TOKEN=\n", encoding="utf-8")
    assert setup_mod.hf_token_check(env)["ok"] is False
    assert setup_mod.hf_token_check(tmp_path / "missing")["ok"] is False


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

    checks = client.get("/api/setup").json()["checks"]
    installable = {c["id"] for c in checks if c["installable"]}
    assert installable == set(install_mod.INSTALLERS)
    assert installable == set(tools.auto_installers())
    assert installable <= {"ffmpeg", "sox"}
    assert all(isinstance(c["installable"], bool) for c in checks)


def test_install_refuses_an_id_it_has_no_recipe_for(client):
    """A model is gigabytes and a Hugging Face login; the refusal has to hand
    the user the command instead of pretending there is a button for it."""
    from dubbing import tools

    for bad in ("model.translate", "hf_token", "disk", "rm -rf /"):
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
    recipe: the table is the platform's, this rule is not."""
    monkeypatch.setattr(install_mod.shutil, "which", lambda exe, *a, **k: None)
    inst = install_mod.Installer(lambda id_: None,
                                 recipes={"ffmpeg": ("brew", "install", "ffmpeg")})
    with pytest.raises(install_mod.invalid("x").__class__) as exc:
        inst.start("ffmpeg")
    message = exc.value.message
    assert exc.value.code == "invalid_request"
    assert "https://brew.sh" in message and "ffmpeg" in message


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
    assert setup_mod.probe("model.translate") is None and setup_mod.probe("nope") is None


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
