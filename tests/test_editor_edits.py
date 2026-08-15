"""The lightweight editor: its manifest patching, and its HTTP surface.

The patching half is pure logic a manifest dict in, what changed out. The HTTP
half runs against the real FastAPI app over a run directory built by hand
(`conftest_app.make_project`), so the shapes asserted are the ones that go on the
wire; no model is ever loaded and no pipeline process is ever spawned.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dubbing import manifest  # noqa: E402
from editor.edits import EDITABLE, apply_edits, earliest  # noqa: E402


def a_manifest() -> dict:
    return {
        "version": 1,
        "source": {"input": "x.mp4", "src_lang": "he", "tgt_lang": "en"},
        "speakers": {"SPEAKER_00": {"dur": 5.0}, "SPEAKER_01": {"dur": 2.0}},
        "segments": [
            {"id": 0, "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00",
             "text": "שלום", "keep": False, "keep_reason": None, "text_en": "Hello"},
            {"id": 1, "start": 1.0, "end": 2.0, "speaker": "SPEAKER_00",
             "text": "Good morning", "keep": False, "keep_reason": None,
             "text_en": "Good morning"},
        ],
    }


def edit(seg_id: int, **fields) -> list[dict]:
    return [{"id": seg_id, "fields": fields}]


def test_translation_edit_forces_tts_not_translate():
    # Forcing translate would regenerate the line and throw the correction away.
    m = a_manifest()
    res = apply_edits(m, edit(0, text_en="Hi there"))
    assert m["segments"][0]["text_en"] == "Hi there"
    assert res == {"changed": [0], "fields": ["text_en"], "force": "tts"}


def test_source_text_edit_forces_translate():
    m = a_manifest()
    res = apply_edits(m, edit(0, text="שלום עולם"))
    assert res["force"] == "translate"


def test_speaker_edit_forces_tts():
    m = a_manifest()
    res = apply_edits(m, edit(0, speaker="SPEAKER_01"))
    assert m["segments"][0]["speaker"] == "SPEAKER_01"
    assert res["force"] == "tts"


def test_tts_instructions_are_rejected_with_the_reason():
    """The clone checkpoint has no instruction channel (generate_voice_clone takes
    no `instruct`), so storing the field would be a knob that silently does
    nothing. The error tells the user where per-segment control actually lives."""
    m = a_manifest()
    with pytest.raises(ValueError, match="no instruction channel"):
        apply_edits(m, edit(0, tts_instructions="calm, amused"))


def test_target_language_override_forces_translate():
    m = a_manifest()
    # `lang_override` is the old wire name; the canonical key is the studio's
    # per-segment `tgt_lang` one concept, one manifest key.
    res = apply_edits(m, edit(1, lang_override="es"))
    assert m["segments"][1]["tgt_lang"] == "es"
    assert "lang_override" not in m["segments"][1]
    assert res["force"] == "translate"
    res = apply_edits(m, edit(1, tgt_lang="fr"))
    assert m["segments"][1]["tgt_lang"] == "fr"


def test_passthrough_lands_on_the_pipelines_own_key():
    """The pipeline owns `passthrough` (segments.apply_passthrough honours it,
    carry_passthrough survives re-segmentation) the editor just writes it."""
    from dubbing import segments as seg_mod

    m = a_manifest()
    apply_edits(m, edit(1, passthrough=True))
    seg = m["segments"][1]
    assert seg["passthrough"] is True
    # The next pipeline entry turns the override into a keep.
    assert seg_mod.apply_passthrough(m["segments"]) == [1]
    assert seg["keep"] is True and seg["keep_reason"] == "user"
    res = apply_edits(m, edit(1, passthrough=False))
    assert seg["passthrough"] is False
    assert seg_mod.apply_passthrough(m["segments"]) == [1]
    assert seg["keep"] is False and seg["keep_reason"] is None
    assert res["force"] == "tts"


def test_unchanged_values_invalidate_nothing():
    m = a_manifest()
    apply_edits(m, edit(0, passthrough=False))     # the override exists now
    res = apply_edits(m, [{"id": 0, "fields": {"text_en": "Hello", "speaker": "SPEAKER_00",
                                               "passthrough": False}}])
    assert res == {"changed": [], "fields": [], "force": None}


def test_forcing_a_dub_on_a_kept_span_is_a_real_edit():
    """`passthrough` is tri-state absent means "decide automatically", False means
    "dub this span" so False is a value, not an empty field. Folding it into
    "empty" made unchecking the box on a line the pipeline kept (a foreign span the
    user wants dubbed after all) a silent no-op: the save reported "nothing
    changed" and the next run kept the original audio."""
    from dubbing import segments as seg_mod

    m = a_manifest()
    seg = m["segments"][1]
    seg["keep"], seg["keep_reason"] = True, "foreign"

    res = apply_edits(m, edit(1, passthrough=False))
    assert seg["passthrough"] is False and res["changed"] == [1]
    assert seg_mod.apply_passthrough(m["segments"]) == [1]
    assert seg["keep"] is False and seg["keep_reason"] is None


def test_mixed_edits_force_the_earliest_stage():
    m = a_manifest()
    res = apply_edits(m, edit(0, text_en="Hi") + edit(1, text="Morning, all"))
    assert res["changed"] == [0, 1]
    assert res["force"] == "translate"


def test_rejects_unknown_field_and_unknown_segment():
    m = a_manifest()
    with pytest.raises(ValueError, match="not editable"):
        apply_edits(m, edit(0, place={"start": 0}))
    with pytest.raises(ValueError, match="no segment 99"):
        apply_edits(m, edit(99, text_en="x"))


def test_rejects_bad_types_and_blank_text():
    m = a_manifest()
    with pytest.raises(ValueError, match="true or false"):
        apply_edits(m, edit(0, passthrough="yes"))
    with pytest.raises(ValueError, match="speaker cannot be empty"):
        apply_edits(m, edit(0, speaker="  "))
    with pytest.raises(ValueError, match="cannot be empty"):
        apply_edits(m, edit(0, text_en="   "))


def test_editable_fields_survive_manifest_save(tmp_path):
    m = a_manifest()
    apply_edits(m, edit(0, passthrough=True, lang_override="es", lang="he"))
    manifest.save(tmp_path, m)
    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["segments"][0]
    for field in EDITABLE:
        assert field in manifest.SEGMENT_KEYS
    assert saved["passthrough"] is True
    assert saved["tgt_lang"] == "es"


def test_hand_edits_survive_a_later_global_rerun():
    """`--force translate` calls `reset_stage("translate")`, which drops `text_en`
    on every segment it is allowed to. A correction the user typed three saves ago
    is not the translator's to reissue, so editing has to lock it the invariant
    `manifest.reset_stage` spells out and the editor never held up its end."""
    m = a_manifest()
    apply_edits(m, edit(0, text_en="Hi there"))
    assert m["segments"][0]["locked"] == {"text_en": True}

    manifest.reset_stage(m, "translate")            # what --force translate does
    assert m["segments"][0]["text_en"] == "Hi there"
    assert "text_en" not in m["segments"][1]        # untouched lines still redo


def test_every_hand_edit_locks_its_own_field():
    m = a_manifest()
    apply_edits(m, edit(0, text="שלום עולם", speaker="SPEAKER_01", passthrough=True))
    # `passthrough` is the user's verdict about the span, recorded as `keep`.
    assert m["segments"][0]["locked"] == {"text": True, "speaker": True, "keep": True}


def test_a_language_change_releases_the_translation_it_invalidates():
    """The stored line is in the wrong language now, so the lock that would stop a
    re-run replacing it has to come off same rule as `dubbing.edit.set_langs`."""
    m = a_manifest()
    apply_edits(m, edit(0, text_en="Hi there"))
    apply_edits(m, edit(0, tgt_lang="es"))
    assert not m["segments"][0].get("locked")
    manifest.reset_stage(m, "translate")
    assert "text_en" not in m["segments"][0]


def test_locks_reach_the_manifest(tmp_path):
    m = a_manifest()
    apply_edits(m, edit(0, text_en="Hi there"))
    manifest.save(tmp_path, m)
    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert saved["segments"][0]["locked"] == {"text_en": True}


def test_earliest_orders_by_pipeline_position():
    assert earliest(["tts", "translate", "mix"]) == "translate"
    assert earliest([]) is None


# ---------------------------------------------------------------------------
# the HTTP surface
# ---------------------------------------------------------------------------

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from editor import jobs, runs  # noqa: E402
from editor.app import app as editor_app  # noqa: E402
from tests.conftest_app import make_project  # noqa: E402

RUN = "whatsapp_0809"


@pytest.fixture()
def editor_client(tmp_path, monkeypatch):
    """The editor pointed at a temp `outputs/`. Nothing is ever launched."""
    root = tmp_path / "outputs"
    make_project(root, RUN)
    monkeypatch.setattr(runs, "OUTPUTS", root.resolve())
    monkeypatch.setattr(runs, "UPLOADS", root.resolve() / "_uploads")
    with TestClient(editor_app) as client:
        yield client


def launched(monkeypatch) -> list:
    """Capture the command a route would have run, instead of running it."""
    calls: list[list[str]] = []

    class Fake:
        def __init__(self, cmd):
            self.id, self.cmd = "job1", cmd

        def state(self, tail: int = 60):
            return {"id": self.id, "cmd": self.cmd, "status": "running", "log": []}

    def fake_launch(cmd, run, label):
        calls.append(cmd)
        return Fake(cmd)

    monkeypatch.setattr(jobs, "launch", fake_launch)
    return calls


def test_save_round_trips_through_the_real_routes(editor_client):
    """The whole Save button: read the run, PATCH what changed, read it back."""
    view = editor_client.get(f"/api/runs/{RUN}").json()
    assert [s["id"] for s in view["segments"]] == list(range(5))

    r = editor_client.patch(f"/api/runs/{RUN}/segments", json={"edits": [
        {"id": 1, "fields": {"text_en": "Corrected.", "passthrough": True}}]})
    assert r.status_code == 200
    assert r.json() == {"changed": [1], "fields": ["passthrough", "text_en"],
                        "force": "tts"}

    seg = editor_client.get(f"/api/runs/{RUN}").json()["segments"][1]
    assert seg["text_en"] == "Corrected."
    assert seg["passthrough"] is True                  # the pipeline's own key
    assert seg["locked"] == {"text_en": True, "keep": True}


def test_a_rerun_reproduces_the_runs_own_fingerprints(editor_client, monkeypatch):
    """A run created with non-default options has to be re-run with them. Every
    option the command drops changes that stage's fingerprint, and an upstream
    fingerprint change is not "re-run that stage": `reset_stage("segments")`
    empties `m["segments"]`, so the whole run and every edit made here is
    discarded and rebuilt from the source media."""
    from dubbing import cli

    workdir = runs.workdir(RUN)
    m = manifest.load(workdir)
    m["source"]["app_opts"] = {"genre": "movie", "register": "dialogue",
                               "tts_model": "0.6b", "transcript": "asr",
                               "dub_foreign": True}
    manifest.save(workdir, m)

    calls = launched(monkeypatch)
    r = editor_client.post(f"/api/runs/{RUN}/rerun", json={"force": "tts"})
    assert r.status_code == 200

    args = cli.parse_args(calls[0][3:])                # drop [python, -m, dubbing]
    created = cli.parse_args([str(m["source"]["input"]), "-o", str(workdir),
                              "--src", "he", "--tgt", "en", "--genre", "movie",
                              "--register", "dialogue", "--tts-model", "0.6b",
                              "--transcript", "asr", "--dub-foreign"])
    assert cli.stage_params(args, m) == cli.stage_params(created, m)
    assert args.force == "tts"


def test_a_rerun_of_a_headless_run_stays_on_the_defaults(editor_client, monkeypatch):
    """Nothing recorded, nothing invented: `python -m dubbing` stores no options
    today, so a run made by it has none to hand back."""
    calls = launched(monkeypatch)
    editor_client.post(f"/api/runs/{RUN}/rerun", json={})
    assert "--genre" not in calls[0] and "--tts-model" not in calls[0]


def test_rerun_rejects_an_unknown_stage(editor_client):
    assert editor_client.post(f"/api/runs/{RUN}/rerun",
                              json={"force": "nonsense"}).status_code == 400


def test_the_ui_reads_the_canonical_passthrough_key():
    """The one thing about the page that cannot be asserted through the API, and
    the one the schema merge broke: the checkbox was rendered from `keep &&
    keep_reason === "manual"`, while both front ends write the verdict to
    `passthrough` and `apply_passthrough` records it as "user". A saved
    passthrough therefore came back unchecked, every time, forever."""
    source = (Path(__file__).resolve().parents[1] / "editor" / "static"
              / "app.js").read_text(encoding="utf-8")
    assert 's.passthrough' in source
    assert 'keep_reason === "manual"' not in source


def test_unknown_run_is_404(editor_client):
    assert editor_client.get("/api/runs/nope").status_code == 404
    assert editor_client.patch("/api/runs/nope/segments",
                               json={"edits": []}).status_code == 404
