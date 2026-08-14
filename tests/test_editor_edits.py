"""Editor manifest patching — pure logic, no server and no models."""

from __future__ import annotations

import json

import pytest

from dubbing import manifest
from editor.edits import EDITABLE, apply_edits, earliest


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


def test_speaker_and_instructions_force_tts():
    m = a_manifest()
    res = apply_edits(m, edit(0, speaker="SPEAKER_01", tts_instructions="calm, amused"))
    assert m["segments"][0]["speaker"] == "SPEAKER_01"
    assert m["segments"][0]["tts_instructions"] == "calm, amused"
    assert res["force"] == "tts"


def test_target_language_override_forces_translate():
    m = a_manifest()
    res = apply_edits(m, edit(1, lang_override="es"))
    assert m["segments"][1]["lang_override"] == "es"
    assert res["force"] == "translate"


def test_passthrough_flag_round_trips():
    m = a_manifest()
    apply_edits(m, edit(1, passthrough=True))
    assert m["segments"][1]["passthrough"] is True
    # Clearing it drops the key rather than storing a false.
    res = apply_edits(m, edit(1, passthrough=False))
    assert "passthrough" not in m["segments"][1]
    assert res["force"] == "tts"


def test_unchanged_values_invalidate_nothing():
    m = a_manifest()
    res = apply_edits(m, [{"id": 0, "fields": {"text_en": "Hello", "speaker": "SPEAKER_00",
                                               "tts_instructions": "", "passthrough": False}}])
    assert res == {"changed": [], "fields": [], "force": None}


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
    apply_edits(m, edit(0, tts_instructions="whisper", passthrough=True, lang_override="es",
                        lang="he"))
    manifest.save(tmp_path, m)
    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["segments"][0]
    for field in EDITABLE:
        assert field in manifest.SEGMENT_KEYS
    assert saved["tts_instructions"] == "whisper"
    assert saved["passthrough"] is True
    assert saved["lang_override"] == "es"


def test_earliest_orders_by_pipeline_position():
    assert earliest(["tts", "translate", "mix"]) == "translate"
    assert earliest([]) is None
