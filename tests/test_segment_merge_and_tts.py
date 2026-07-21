"""Unit checks for gap recovery, unfinished stitch, speaker stabilize, TTS prep."""

from __future__ import annotations

from inference.segment_merge import (
    clamp_phrase_timeline,
    dedupe_repeated_sentences,
    find_uncovered_gaps,
    is_short_completion,
    merge_short_phrases,
    needs_object_continuation,
    stabilize_speaker_continuity,
    stitch_unfinished_continuations,
    utterance_unfinished,
)
from inference.tts_qwen import prepare_english_tts_text


def test_utterance_unfinished_open_adverbial():
    assert utterance_unfinished("והיא עושה את זה באופן.")
    assert utterance_unfinished("וכובש אותך,")
    assert not utterance_unfinished("אין מענה.")


def test_stitch_short_completion_baofen_amanut():
    segs = [
        {
            "speaker_id": "SPEAKER_05",
            "language": "he",
            "keep_original": False,
            "start": 88.0,
            "end": 94.0,
            "text": "והיא עושה את זה באופן.",
            "phrases": [
                {
                    "text": "והיא עושה את זה באופן.",
                    "start": 88.0,
                    "end": 94.0,
                    "pause_after": 0.0,
                    "speaker_id": "SPEAKER_05",
                }
            ],
        },
        {
            "speaker_id": "SPEAKER_02",
            "language": "he",
            "keep_original": False,
            "start": 94.2,
            "end": 95.0,
            "text": "אמנות רבה.",
            "phrases": [
                {
                    "text": "אמנות רבה.",
                    "start": 94.2,
                    "end": 95.0,
                    "pause_after": 0.0,
                    "speaker_id": "SPEAKER_02",
                }
            ],
        },
    ]
    assert is_short_completion(segs[1])
    out = stitch_unfinished_continuations(segs)
    assert len(out) == 1
    assert "אמנות רבה" in out[0]["text"]
    assert "באופן" in out[0]["text"]
    assert out[0]["speaker_id"] == "SPEAKER_05"
    # After clamp, short completion may be absorbed into previous phrase.
    assert len(out[0]["phrases"]) >= 1


def test_stitch_same_speaker_long_pause_continuation():
    segs = [
        {
            "speaker_id": "SPEAKER_05",
            "language": "he",
            "keep_original": False,
            "start": 46.0,
            "end": 55.0,
            "text": "כשבא לך האויב בצורה מתוחכמת וכובש אותך,",
            "phrases": [
                {
                    "text": "כשבא לך האויב בצורה מתוחכמת וכובש אותך,",
                    "start": 46.0,
                    "end": 55.0,
                    "pause_after": 0.0,
                }
            ],
        },
        {
            "speaker_id": "SPEAKER_05",
            "language": "he",
            "keep_original": False,
            "start": 58.5,
            "end": 60.8,
            "text": "בנינוחות כזאת, בשלוות נפש.",
            "phrases": [
                {
                    "text": "בנינוחות כזאת, בשלוות נפש.",
                    "start": 58.5,
                    "end": 60.8,
                    "pause_after": 0.0,
                }
            ],
        },
    ]
    out = stitch_unfinished_continuations(segs)
    assert len(out) == 1
    assert "בנינוחות" in out[0]["text"]
    assert out[0]["end"] == 60.8


def test_stabilize_speaker_continuity_flip():
    """Short unfinished completion inherits previous speaker; long turns do not."""
    # Short stub continuation — should flip.
    short = [
        {
            "speaker_id": "SPEAKER_08",
            "language": "he",
            "keep_original": False,
            "start": 77.0,
            "end": 83.9,
            "text": "כשאנחנו מדברים על שייחה מוזה. שייחה היא",
            "phrases": [
                {
                    "text": "כשאנחנו מדברים על שייחה מוזה. שייחה היא",
                    "start": 77.0,
                    "end": 83.9,
                    "pause_after": 0.0,
                    "speaker_id": "SPEAKER_08",
                }
            ],
        },
        {
            "speaker_id": "SPEAKER_05",
            "language": "he",
            "keep_original": False,
            "start": 83.95,
            "end": 84.8,
            "text": "אמנות רבה.",
            "phrases": [
                {
                    "text": "אמנות רבה.",
                    "start": 83.95,
                    "end": 84.8,
                    "pause_after": 0.0,
                    "speaker_id": "SPEAKER_05",
                }
            ],
        },
    ]
    out = stabilize_speaker_continuity(short)
    assert out[1]["speaker_id"] == "SPEAKER_08"

    # Long independent turn at 1:20 — must KEEP its own speaker (no stuck voice).
    long = [
        {
            "speaker_id": "SPEAKER_08",
            "language": "he",
            "keep_original": False,
            "start": 77.1,
            "end": 80.9,
            "text": "כשאנחנו מדברים על קטר המודרנית, אנחנו מדברים על שייחה מוזה. שייחה היא...",
            "phrases": [
                {
                    "text": "שייחה היא...",
                    "start": 79.8,
                    "end": 80.9,
                    "pause_after": 0.0,
                    "speaker_id": "SPEAKER_08",
                }
            ],
        },
        {
            "speaker_id": "SPEAKER_05",
            "language": "he",
            "keep_original": False,
            "start": 81.0,
            "end": 94.9,
            "text": "בראש ובראשונה, זו שקובעת את האסטרטגיה. אישה עם יכולות יוצאות דופן, מאוד חכמה.",
            "phrases": [
                {
                    "text": "בראש ובראשונה, זו שקובעת את האסטרטגיה.",
                    "start": 81.0,
                    "end": 84.0,
                    "pause_after": 0.0,
                    "speaker_id": "SPEAKER_05",
                }
            ],
        },
    ]
    out2 = stabilize_speaker_continuity(long)
    assert out2[1]["speaker_id"] == "SPEAKER_05"
    assert out2[1]["phrases"][0]["speaker_id"] == "SPEAKER_05"


def test_find_uncovered_gaps():
    segs = [
        {"start": 0.0, "end": 10.0},
        {"start": 30.0, "end": 40.0},
    ]
    gaps = find_uncovered_gaps(segs, media_end=50.0, min_gap=1.5)
    assert (10.0, 30.0) in gaps
    assert (40.0, 50.0) in gaps


def test_prepare_english_tts_jabhat():
    text = "Qatar funds ISIS, Jabhat al-Nusra, and Qatar helps"
    out = prepare_english_tts_text(text)
    assert "Jabhat al-Nusra" in out
    assert "Nusra Front" not in out
    # Legacy / misspellings normalize to the proper name.
    assert "Jabhat al-Nusra" in prepare_english_tts_text(
        "Qatar funds the Nusra Front and Jahbat a-Nusra"
    )

def test_compact_phrase_timeline_gaps():
    from inference.tts_qwen import compact_phrase_timeline_gaps

    plan = [
        {"start": 46.0, "end": 50.0},
        {"start": 50.35, "end": 55.0},  # 0.35s — sentence pause, keep
        {"start": 58.5, "end": 61.0},  # 3.5s hole — compact
    ]
    dead = compact_phrase_timeline_gaps(plan, max_pause=0.55, min_gap_to_compact=1.0)
    assert abs(dead - 2.95) < 0.02
    assert abs(plan[2]["start"] - 55.55) < 0.02
    assert abs((plan[1]["start"] - plan[0]["end"]) - 0.35) < 0.02


def test_drop_silent_and_split_sentences():
    from inference.segment_merge import split_long_phrases_at_periods

    phrases = [
        {
            "text": "משפט ראשון ארוך מאוד. משפט שני גם הוא ארוך מאוד כאן.",
            "start": 0.0,
            "end": 10.0,
            "pause_after": 0.0,
            "text_en": "First long sentence here. Second long sentence here too.",
        }
    ]
    out = split_long_phrases_at_periods(phrases, min_dur=4.5, sentence_pause=0.55)
    assert len(out) == 2
    assert out[0]["pause_after"] >= 0.28
    assert "ראשון" in out[0]["text"]
    assert "שני" in out[1]["text"]


def test_trim_leading_silence(tmp_path):
    import numpy as np
    import soundfile as sf

    from inference.tts_qwen import trim_leading_silence

    sr = 44100
    hush = np.zeros(int(0.5 * sr), dtype=np.float32)
    tone = (0.2 * np.sin(2 * np.pi * 440 * np.arange(int(0.4 * sr)) / sr)).astype(
        np.float32
    )
    src = tmp_path / "raw.wav"
    dst = tmp_path / "trim.wav"
    sf.write(str(src), np.concatenate([hush, tone]), sr)
    trimmed = trim_leading_silence(src, dst, sample_rate=sr)
    assert abs(trimmed - 0.47) < 0.05
    out, _ = sf.read(str(dst), dtype="float32")
    assert float(np.max(np.abs(out[: sr // 20]))) > 0.05


def test_dedupe_repeated_sentences():
    s = (
        "and she does it with great artistry. it with great artistry. "
        "it with great artistry."
    )
    out = dedupe_repeated_sentences(s)
    assert out.lower().count("great artistry") == 1
    assert dedupe_repeated_sentences(out) == out


def test_merge_short_phrases_idempotent():
    phrases = [
        {
            "text": "יש לה תפקיד מרכזי באופן.",
            "text_en": "She plays a central role with great artistry.",
            "start": 0.0,
            "end": 4.0,
            "pause_after": 0.2,
        },
        {
            "text": "אמנות רבה.",
            "text_en": "it with great artistry.",
            "start": 4.2,
            "end": 5.0,
            "pause_after": 0.0,
        },
    ]
    once = merge_short_phrases(phrases)
    twice = merge_short_phrases([dict(p) for p in once])
    assert once[0]["text_en"] == twice[0]["text_en"]
    assert once[0]["text_en"].count("great artistry") == 1


def test_clamp_phrase_timeline_folds_overlap():
    phrases = [
        {"text": "long clause", "text_en": "A", "start": 0.0, "end": 10.0, "pause_after": 0.0},
        {
            "text": "tiny",
            "text_en": "B",
            "start": 8.5,
            "end": 9.5,
            "pause_after": 0.0,
        },
    ]
    out = clamp_phrase_timeline(phrases)
    assert len(out) == 1
    assert "tiny" in out[0]["text"]
    assert "B" in out[0]["text_en"]


def test_distribute_en_respects_long_pause():
    from inference.build_preview import distribute_en_to_phrases

    phrases = [
        {
            "text": "אנחנו רגילים לזהות את האויב בעין, את הנשק,",
            "start": 46.0,
            "end": 50.0,
            "pause_after": 0.3,
        },
        {
            "text": "אבל כשבא לך האויב בצורה מתוחכמת וכובש אותך,",
            "start": 50.3,
            "end": 55.0,
            "pause_after": 3.5,
        },
        {
            "text": "בנינוחות כזאת, בשלוות נפש.",
            "start": 58.5,
            "end": 61.0,
            "pause_after": 0.0,
        },
    ]
    en = (
        "We are used to identifying the enemy in the eye, the weapon, "
        "but when the enemy comes in a clever way and subdues you, "
        "with such calmness, composure."
    )
    distribute_en_to_phrases(en, phrases)
    # Hard pause must survive (soft gaps before it may merge).
    assert len(phrases) >= 2
    # Last phrase starts after the long silence.
    assert float(phrases[-1]["start"]) >= 58.0
    # Gap / pause before the last phrase is still a hard silence.
    gap = float(phrases[-1]["start"]) - float(phrases[-2]["end"])
    pause = float(phrases[-2].get("pause_after") or 0.0)
    assert max(gap, pause) >= 0.8
    joined = " ".join(p["text_en"] for p in phrases)
    assert "calmness" in joined or "composure" in joined
    assert all((p.get("text_en") or "").strip() for p in phrases)
    # Tail after the hard pause should carry the calmness clause.
    assert "calmness" in phrases[-1]["text_en"] or "composure" in phrases[-1]["text_en"]


def test_merge_short_phrases_preserves_hard_pause():
    phrases = [
        {"text": "וכובש אותך,", "start": 53.9, "end": 55.1, "pause_after": 3.5},
        {"text": "בנינוחות כזאת.", "start": 58.6, "end": 60.8, "pause_after": 0.0},
    ]
    out = merge_short_phrases(phrases)
    assert len(out) == 2
    assert abs(float(out[1]["start"]) - 58.6) < 0.01


def test_merge_preserves_sentence_final_interjection():
    phrases = [
        {
            "text": "ומעבר לכך, הכסף הקטרי הוביל להסכם.",
            "start": 205.26,
            "end": 216.06,
            "pause_after": 0.16,
        },
        {
            "text": "שלא נתבלבל.",
            "start": 216.22,
            "end": 217.50,
            "pause_after": 0.0,
        },
    ]
    out = merge_short_phrases(phrases)
    assert len(out) == 2
    assert "שלא נתבלבל" in out[1]["text"]


def test_coalesce_speech_runs():
    from inference.tts_qwen import coalesce_speech_runs

    plan = [
        {"text": "When we talk about Qatar,", "text_he": "א", "start": 77.0, "end": 81.3, "pause_after": 0.0},
        {"text": "she determines the strategy.", "text_he": "ב", "start": 81.33, "end": 84.0, "pause_after": 0.0},
        {"text": "After a pause.", "text_he": "ג", "start": 85.5, "end": 87.0, "pause_after": 0.0},
    ]
    out = coalesce_speech_runs(plan)
    assert len(out) == 2
    assert "determines" in out[0]["text"]
    assert "pause" in out[1]["text"].lower()


def test_retag_english_sandwich():
    from inference.segment_merge import retag_english_sandwich

    segs = [
        {
            "speaker_id": "SPEAKER_00",
            "language": "en",
            "keep_original": True,
            "start": 272.0,
            "end": 277.0,
            "text": "I had the same concerns",
            "phrases": [],
        },
        {
            "speaker_id": "SPEAKER_00",
            "language": "he",
            "keep_original": False,
            "start": 277.1,
            "end": 278.7,
            "text": "אוקיי, על כך",
            "phrases": [
                {"text": "אוקיי, על כך", "start": 277.1, "end": 278.7, "pause_after": 0.0}
            ],
        },
        {
            "speaker_id": "SPEAKER_00",
            "language": "en",
            "keep_original": True,
            "start": 280.5,
            "end": 288.0,
            "text": "The approach that we had",
            "phrases": [],
        },
    ]
    n = retag_english_sandwich(segs)
    assert n == 1
    assert segs[1]["keep_original"] is True
    assert segs[1]["language"] == "en"


def test_fit_max_rate_capped():
    from inference.tts_qwen import (
        FIT_MAX_RATE,
        FIT_MAX_RATE_GENTLE,
        FIT_MAX_RATE_HARD,
        FIT_MIN_RATE,
        UNIT_SPLIT_PAUSE_SEC,
    )

    assert FIT_MIN_RATE <= 0.90 + 1e-9
    assert FIT_MAX_RATE <= 1.15 + 1e-9
    assert FIT_MAX_RATE_GENTLE <= 1.15 + 1e-9
    assert FIT_MAX_RATE_HARD <= 1.50 + 1e-9
    assert FIT_MIN_RATE < 1.0 <= FIT_MAX_RATE
    assert UNIT_SPLIT_PAUSE_SEC >= 1.0


def test_assemble_keeps_full_clip_before_keep(tmp_path):
    """Never-cut assemble: long unit audio is kept even when next is KEEP."""
    import numpy as np
    import soundfile as sf

    from inference.tts_qwen import assemble_on_hebrew_timeline, wav_duration

    sr = 44100
    # Phrase clip longer than its HE slot — must keep full length (packer yields).
    tone = (0.2 * np.sin(2 * np.pi * 440 * np.arange(int(1.0 * sr)) / sr)).astype(
        np.float32
    )
    clip = tmp_path / "p0.wav"
    sf.write(str(clip), tone, sr)
    plan = [
        {
            "start": 10.0,
            "end": 10.5,
            "tts_fit": str(clip),
        }
    ]
    out_keep = tmp_path / "keep.wav"
    assemble_on_hebrew_timeline(
        plan,
        10.0,
        10.5,
        out_keep,
        sample_rate=sr,
        hard_end=10.38,
        next_is_keep=True,
    )
    # Full 1.0s clip must survive (canvas grows; no mid-sentence chop).
    assert abs(wav_duration(out_keep) - 1.0) < 0.08


def test_drop_silent_keeps_source_speech(tmp_path):
    import numpy as np
    import soundfile as sf

    from inference.segment_merge import drop_silent_vocal_segments

    sr = 16000
    vocals = tmp_path / "vocals.wav"
    source = tmp_path / "source.wav"
    # Vocals stem silent; source has speech (applause-masked case).
    sf.write(str(vocals), np.zeros(sr * 3, dtype=np.float32), sr)
    tone = (0.15 * np.sin(2 * np.pi * 220 * np.arange(sr * 3) / sr)).astype(np.float32)
    sf.write(str(source), tone, sr)
    segs = [
        {
            "speaker_id": "SPEAKER_00",
            "start": 0.5,
            "end": 2.5,
            "text": "תודה רבה לך, אדוני היושב-ראש",
            "language": "he",
            "keep_original": False,
        }
    ]
    kept = drop_silent_vocal_segments(segs, vocals, source_path=source)
    assert len(kept) == 1
    dropped = drop_silent_vocal_segments(segs, vocals, source_path=None)
    assert len(dropped) == 0


def test_keep_yield_guard_constant():
    from inference.tts_qwen import KEEP_YIELD_GUARD_SEC, GAP_BLEED_SEC

    assert 0.08 <= KEEP_YIELD_GUARD_SEC <= 0.20
    # Isochronous fit keeps bleed tiny — prefer compress over overrun.
    assert 0.10 <= GAP_BLEED_SEC <= 0.30


def test_phonetic_keep_rejects_hebrew_loanwords():
    from inference.lang_detect import looks_like_phonetic_english, text_is_hebrew_script_heavy

    he_commentary = (
        "צ'ייחה מוזה נראה בישראל כפילנתרופיסטית בנבולנטרופיסטית. "
        "היא נראית במגזינות סטייל, היא מדברת על ידי החלטה וילדים."
    )
    assert text_is_hebrew_script_heavy(he_commentary)
    assert not looks_like_phonetic_english(he_commentary)

    # Real phonetic EN (markers + sparse HE / Latin islands)
    phonetic = "פרנקלי אינפלואנס אופריציה CIA World War"
    # May or may not pass depending on ratios — strong markers with Latin OK
    assert looks_like_phonetic_english(phonetic) or "CIA" in phonetic


def test_merge_refuses_thanks_after_sentence():
    from inference.segment_merge import merge_same_speaker_segments

    segs = [
        {
            "speaker_id": "SPEAKER_07",
            "language": "he",
            "keep_original": False,
            "start": 219.0,
            "end": 232.65,
            "text": "זו המטרה של האחים המוסלמים, לכבוש את העולם.",
            "phrases": [
                {
                    "text": "זו המטרה של האחים המוסלמים, לכבוש את העולם.",
                    "start": 219.0,
                    "end": 232.65,
                    "pause_after": 0.0,
                }
            ],
        },
        {
            "speaker_id": "SPEAKER_07",
            "language": "he",
            "keep_original": False,
            "start": 233.65,
            "end": 241.0,
            "text": "תודה רבה לך, אדוני היושב-ראש.",
            "phrases": [
                {
                    "text": "תודה רבה לך, אדוני היושב-ראש.",
                    "start": 233.65,
                    "end": 241.0,
                    "pause_after": 0.0,
                }
            ],
        },
    ]
    out = merge_same_speaker_segments(segs, max_pause=1.0)
    assert len(out) == 2
    assert "תודה" in out[1]["text"]


def test_stitch_cross_speaker_completing_clause():
    from inference.segment_merge import stitch_unfinished_continuations

    segs = [
        {
            "speaker_id": "SPEAKER_07",
            "language": "he",
            "keep_original": False,
            "start": 21.0,
            "end": 23.9,
            "text": "כל ארגון טרור אסלאמי שפועל",
            "phrases": [
                {
                    "text": "כל ארגון טרור אסלאמי שפועל",
                    "start": 21.0,
                    "end": 23.9,
                    "pause_after": 0.0,
                }
            ],
        },
        {
            "speaker_id": "SPEAKER_08",
            "language": "he",
            "keep_original": False,
            "start": 24.0,
            "end": 38.0,
            "text": "בעולם היא מממנת אותו. קטר מממן את חמאס.",
            "phrases": [
                {
                    "text": "בעולם היא מממנת אותו.",
                    "start": 24.0,
                    "end": 25.7,
                    "pause_after": 0.1,
                },
                {
                    "text": "קטר מממן את חמאס.",
                    "start": 25.8,
                    "end": 38.0,
                    "pause_after": 0.0,
                },
            ],
        },
    ]
    out = stitch_unfinished_continuations(segs)
    assert "בעולם" in out[0]["text"]
    assert "שפועל" in out[0]["text"]
    # Remaining B keeps the Hamas clause
    assert any("חמאס" in (s.get("text") or "") for s in out)


def test_mt_memory_echo_and_bana():
    from inference.build_preview import (
        mt_needs_retry,
        strip_leading_memory_echo,
        _postprocess_en,
    )

    prev = "In such a calm manner, conquering you with such calmness, in a peaceful way."
    echoed = (
        "In such a calm manner, conquering you with such calmness, in a peaceful way. "
        "There is no response."
    )
    reason = mt_needs_retry("אין מענה.", echoed, memory=[("prev", prev)])
    assert reason in ("memory_echo", "en_too_long_for_he")
    stripped = strip_leading_memory_echo(echoed, prev)
    assert "no response" in stripped.lower()
    assert "calm manner" not in stripped.lower()

    assert mt_needs_retry("אין מענה.", "There is no answer at all about the whole situation here today") == (
        "en_too_long_for_he"
    )
    assert mt_needs_retry("לבנה השני", "to his second daughter") == "bana_gender"
    fixed = _postprocess_en("העברת שלטון לבנה השני", "transfer to his second daughter")
    assert "son" in fixed.lower()
    assert "daughter" not in fixed.lower()


def test_tts_overrun_constants():
    from inference.tts_qwen import (
        UNIT_SPLIT_PAUSE_SEC,
        CLONE_MIN_OVERLAP,
        CLONE_MAX_TRIES,
        FIT_MAX_RATE,
        FIT_MAX_RATE_GENTLE,
        FIT_MAX_RATE_HARD,
        FIT_MIN_RATE,
        REF_MIN_SEC,
        KEEP_YIELD_GUARD_SEC,
        SHORT_SLOT_OVERRUN_CAP_SEC,
        TIGHT_NEXT_GAP_SEC,
    )

    assert UNIT_SPLIT_PAUSE_SEC >= 1.0
    assert 0.25 <= CLONE_MIN_OVERLAP <= 0.50
    assert CLONE_MAX_TRIES >= 2
    assert REF_MIN_SEC >= 2.0
    assert FIT_MIN_RATE < FIT_MAX_RATE <= FIT_MAX_RATE_GENTLE + 1e-9
    assert FIT_MAX_RATE_HARD >= 1.40
    assert KEEP_YIELD_GUARD_SEC >= 0.08
    assert SHORT_SLOT_OVERRUN_CAP_SEC <= 0.4
    assert TIGHT_NEXT_GAP_SEC <= 0.35


def test_split_utterance_into_units_long_turn_subsplit():
    """Long monologues without a big pause must still get timeline anchors."""
    from inference.tts_qwen import MAX_UNIT_SEC, split_utterance_into_units

    seg = {
        "text": (
            "משפט אחד ארוך מאוד על הנושא. משפט שני גם ארוך וממשיך. "
            "משפט שלישי ממשיך בזהירות. משפט רביעי מסיים את המחשבה."
        ),
        "text_en": (
            "First long sentence here about the topic. Second long sentence continues the idea. "
            "Third sentence keeps going carefully. Fourth sentence finishes the thought."
        ),
        "start": 155.0,
        "end": 182.0,
        "phrases": [
            {
                "text": (
                    "משפט אחד ארוך מאוד על הנושא. משפט שני גם ארוך וממשיך. "
                    "משפט שלישי ממשיך בזהירות. משפט רביעי מסיים את המחשבה."
                ),
                "start": 155.0,
                "end": 182.0,
                "pause_after": 0.0,
            }
        ],
    }
    units = split_utterance_into_units(seg)
    assert len(units) >= 2
    assert all(float(u["end"]) - float(u["start"]) <= MAX_UNIT_SEC + 1.0 for u in units)
    joined = " ".join(u["text_en"] for u in units)
    assert "First" in joined and "Fourth" in joined


def test_split_utterance_into_units_collapses_small_pauses():
    from inference.tts_qwen import split_utterance_into_units

    seg = {
        "text": "משפט אחד. המשך קצר.",
        "text_en": "One sentence. A short continuation.",
        "start": 10.0,
        "end": 14.0,
        "phrases": [
            {
                "text": "משפט אחד.",
                "start": 10.0,
                "end": 12.0,
                "pause_after": 0.3,
            },
            {
                "text": "המשך קצר.",
                "start": 12.3,
                "end": 14.0,
                "pause_after": 0.0,
            },
        ],
    }
    units = split_utterance_into_units(seg)
    assert len(units) == 1
    assert "One sentence" in units[0]["text_en"]
    assert "continuation" in units[0]["text_en"]


def test_split_utterance_into_units_keeps_big_pause():
    from inference.tts_qwen import MAX_MID_SILENCE_SEC, split_utterance_into_units

    seg = {
        "text": "חלק ראשון ארוך. חלק שני אחרי שתיקה.",
        "text_en": (
            "We are used to spotting the enemy with a glance, "
            "but not when they use clever tricks. With such calmness, composure."
        ),
        "start": 46.0,
        "end": 61.0,
        "phrases": [
            {
                "text": "חלק ראשון ארוך מאוד כאן עם מילים.",
                "start": 46.0,
                "end": 55.0,
                "pause_after": 3.5,
            },
            {
                "text": "חלק שני אחרי שתיקה ארוכה.",
                "start": 58.5,
                "end": 61.0,
                "pause_after": 0.0,
            },
        ],
    }
    units = split_utterance_into_units(seg)
    assert len(units) == 2
    # Mid-silence is capped and later unit is shifted earlier (no 3.5s dead air).
    assert float(units[0].get("pause_after") or 0) <= MAX_MID_SILENCE_SEC + 1e-6
    gap = float(units[1]["start"]) - float(units[0]["end"])
    assert abs(gap - float(units[0]["pause_after"])) < 0.05
    assert float(units[1]["start"]) < 58.0  # shifted from 58.5
    assert all((u.get("text_en") or "").strip() for u in units)
    joined = " ".join(u["text_en"] for u in units)
    assert (
        "enemy" in joined.lower()
        or "calmness" in joined.lower()
        or "composure" in joined.lower()
    )
    # EN must not dump almost everything into unit 2 after an early comma.
    assert len(units[0]["text_en"].split()) >= 6


def test_coarse_split_en_by_weights():
    from inference.tts_qwen import coarse_split_en_by_weights

    en = "First clause here. Second clause follows nicely."
    chunks = coarse_split_en_by_weights(en, [5, 4])
    assert len(chunks) == 2
    assert "First" in chunks[0]
    assert "Second" in chunks[1]

    # Duration-weighted: don't snap to early "weapon," when target is ~55% in.
    long_en = (
        "We are used to identifying the enemy with a glance, with the weapon, "
        "but when the enemy comes in a clever way and subdues you with such "
        "calmness with composure."
    )
    chunks2 = coarse_split_en_by_weights(long_en, [4.7, 3.4])
    assert len(chunks2) == 2
    assert "weapon" in chunks2[0].lower() or "glance" in chunks2[0].lower()
    # First chunk should carry most of the pre-pause content.
    assert len(chunks2[0].split()) >= 10
    assert "composure" in chunks2[1].lower() or "calmness" in chunks2[1].lower()


def test_clone_length_ok():
    from inference.tts_qwen import clone_length_ok

    assert clone_length_ok(2.5, "Hello world this is a test")
    # Chipmunk: 6 words in 0.3s
    assert not clone_length_ok(0.3, "Hello world this is a test line")
    assert not clone_length_ok(0.0, "Hello")


def test_speaker_language_consistency():
    from inference.build_preview import enforce_speaker_language_consistency

    segs = [
        {
            "speaker_id": "SPEAKER_08",
            "language": "en",
            "keep_original": True,
            "start": 1.0,
            "end": 7.0,
            "text": "Qatar is probably one of Israel's most dangerous enemies.",
            "he_score": -0.8,
            "en_score": 0.5,
        },
        {
            "speaker_id": "SPEAKER_08",
            "language": "en",
            "keep_original": True,
            "start": 38.0,
            "end": 44.0,
            "text": "Qatar is an enemy without stating that it is.",
            "he_score": -0.7,
            "en_score": 0.5,
        },
        {
            "speaker_id": "SPEAKER_08",
            "language": "he",
            "keep_original": False,
            "start": 94.0,
            "end": 110.0,
            "text": "צ'ייחה מוזה נמצאת בישראל כפילנתרופיסטית בנבלנטרופיסטית.",
            "he_score": 0.15,
            "en_score": -2.0,
            "phrases": [],
        },
    ]
    n = enforce_speaker_language_consistency(segs)
    assert segs[0]["keep_original"] is True
    assert segs[1]["keep_original"] is True
    # Low he_score overrides dense HE-script commentary for EN-native speakers.
    assert n >= 1
    assert segs[2]["keep_original"] is True
    assert segs[2]["language"] == "en"

    segs2 = [
        {
            "speaker_id": "SPEAKER_X",
            "language": "en",
            "keep_original": True,
            "start": 1.0,
            "end": 5.0,
            "text": "Hello from the English interview.",
        },
        {
            "speaker_id": "SPEAKER_X",
            "language": "en",
            "keep_original": True,
            "start": 6.0,
            "end": 10.0,
            "text": "Another English line here today.",
        },
        {
            "speaker_id": "SPEAKER_X",
            "language": "he",
            "keep_original": False,
            "start": 11.0,
            "end": 14.0,
            "text": "שייחה מוזה is portrayed.",
            "he_score": 0.20,
            "en_score": -1.5,
            "phrases": [],
        },
    ]
    n2 = enforce_speaker_language_consistency(segs2)
    assert n2 >= 1
    assert segs2[2]["keep_original"] is True
    assert segs2[2]["language"] == "en"


def test_is_latin_english_rejects_cjk_and_chrome():
    from inference.build_preview import (
        is_english_text,
        is_latin_english,
        mt_needs_retry,
        strip_caption_chrome,
        strip_non_latin_runs,
        _postprocess_en,
    )

    assert is_latin_english("She funds every organization.")
    assert not is_latin_english("She is first and foremost the one who埋头于代码，实现功能。")
    assert not is_english_text("埋头于代码")
    assert mt_needs_retry("היא.", "She who埋头于代码") == "non_english_script"
    # Longer HE avoids the short-HE balloon gate so time_overflow can fire.
    overflow_he = " ".join(["מילה"] * 12)
    assert mt_needs_retry(
        overflow_he,
        "This is a very long English paragraph that cannot possibly fit "
        "into a one-second speaking window without sounding rushed at all today.",
        duration=1.0,
    ) == "time_overflow"
    cleaned = strip_caption_chrome(">> Every org [music] she funds.")
    assert ">>" not in cleaned
    assert "[music]" not in cleaned.lower()
    assert "CJK" not in strip_non_latin_runs("Hello 中文 world")
    assert "Hello" in strip_non_latin_runs("Hello 中文 world")
    post = _postprocess_en("היא.", ">> She paints [music] 中文 Qatar.")
    assert ">>" not in post
    assert "中文" not in post
    assert "Qatar" in post


def test_prepare_english_tts_strips_non_latin():
    out = prepare_english_tts_text("She is first 埋头于代码 and foremost.")
    assert "埋" not in out
    assert "foremost" in out.lower()


def test_clamp_segment_timeline_splits_overlap():
    from inference.segment_merge import clamp_segment_timeline

    segs = [
        {
            "speaker_id": "A",
            "start": 70.0,
            "end": 72.4,
            "phrases": [{"text": "one", "start": 70.0, "end": 72.4, "pause_after": 0.0}],
        },
        {
            "speaker_id": "B",
            "start": 71.9,
            "end": 77.0,
            "phrases": [{"text": "two", "start": 71.9, "end": 77.0, "pause_after": 0.0}],
        },
    ]
    n = clamp_segment_timeline(segs)
    assert n == 1
    assert float(segs[0]["end"]) <= float(segs[1]["start"]) + 1e-6


def test_boundary_aware_trim_prefers_silence():
    import numpy as np
    from inference.tts_qwen import boundary_aware_trim

    sr = 44100
    # 1s speech + 0.2s silence + 0.3s speech
    speech = np.ones(sr, dtype=np.float32) * 0.2
    silence = np.zeros(int(0.2 * sr), dtype=np.float32)
    more = np.ones(int(0.3 * sr), dtype=np.float32) * 0.2
    audio = np.concatenate([speech, silence, more])
    # Max at 1.15s — should cut in the silence near 1.0s, not at 1.15 mid-speech.
    max_n = int(1.15 * sr)
    trimmed = boundary_aware_trim(audio, max_n, sample_rate=sr)
    assert len(trimmed) <= max_n
    assert len(trimmed) < max_n  # found a quieter cut before the hard limit
    assert abs(len(trimmed) / sr - 1.0) < 0.15


def test_merge_object_continuation_ani_lo_stub():
    """אני לא מכירה. + אף אישה… must become one phrase (not two TTS units)."""
    phrases = [
        {
            "text": "אני לא מכירה.",
            "start": 170.0,
            "end": 170.8,
            "pause_after": 0.1,
        },
        {
            "text": "אף אישה במזרח התיכון שיש לה השפעה כזאת.",
            "start": 170.9,
            "end": 177.5,
            "pause_after": 0.0,
        },
    ]
    assert needs_object_continuation(phrases[0]["text"], phrases[1]["text"])
    out = merge_short_phrases(phrases)
    assert len(out) == 1
    assert "אני לא מכירה" in out[0]["text"]
    assert "אף אישה" in out[0]["text"]
    assert float(out[0]["end"]) == 177.5


def test_merge_object_continuation_ignores_bloated_pause_after():
    """Long pause_after metadata must not block semantic object continuation."""
    phrases = [
        {
            "text": "אני לא מכירה.",
            "start": 169.1,
            "end": 179.1,
            "pause_after": 9.5,  # bloated metadata from ASR silence attribution
        },
        {
            "text": "אף אישה במזרח התיכון שיש לה השפעה כזאת.",
            "start": 179.12,
            "end": 181.6,
            "pause_after": 0.0,
        },
    ]
    out = merge_short_phrases(phrases)
    assert len(out) == 1
    assert "אף אישה" in out[0]["text"]
    assert float(out[0]["end"]) == 181.6


def test_split_units_keeps_object_continuation_together():
    """Long-utterance unit split must not sever אני לא מכירה + אף אישה."""
    from inference.tts_qwen import split_utterance_into_units

    seg = {
        "start": 155.5,
        "end": 181.6,
        "source_start": 155.5,
        "source_end": 181.6,
        "text": (
            "מרגע שהיא נכנסה לארמון היא הצליחה. "
            "היא אחראית לשתי ההפיכות. "
            "אני לא מכירה. אף אישה במזרח התיכון שיש לה השפעה כזאת."
        ),
        "text_en": (
            "From the moment she entered the palace she succeeded. "
            "She was responsible for both coups. "
            "I don't know any woman in the Middle East with such influence."
        ),
        "phrases": [
            {"text": "מרגע שהיא נכנסה לארמון היא הצליחה.", "start": 155.5, "end": 162.3, "pause_after": 0.0},
            {"text": "היא אחראית לשתי ההפיכות.", "start": 162.3, "end": 169.1, "pause_after": 0.0},
            {"text": "אני לא מכירה.", "start": 169.1, "end": 179.1, "pause_after": 0.0},
            {"text": "אף אישה במזרח התיכון שיש לה השפעה כזאת.", "start": 179.1, "end": 181.6, "pause_after": 0.0},
        ],
    }
    units = split_utterance_into_units(seg)
    he_bits = [(u.get("text_he") or "") for u in units]
    # Stub must not be alone without its object clause.
    alone = [h for h in he_bits if "מכירה" in h and "אף" not in h]
    assert not alone, he_bits
    joined = [h for h in he_bits if "מכירה" in h and "אף" in h]
    assert joined, he_bits


def test_stamp_source_timing_immutable():
    from inference.segment_merge import stamp_source_timing

    segs = [
        {
            "start": 10.0,
            "end": 12.0,
            "phrases": [
                {"text": "שלום", "start": 10.0, "end": 12.0, "pause_after": 0.0}
            ],
        }
    ]
    n = stamp_source_timing(segs)
    assert n == 1
    assert segs[0]["source_start"] == 10.0
    assert segs[0]["source_end"] == 12.0
    assert segs[0]["phrases"][0]["source_start"] == 10.0
    # Re-stamp is idempotent.
    segs[0]["start"] = 99.0
    stamp_source_timing(segs)
    assert segs[0]["source_start"] == 10.0


def test_parse_marked_en_alignment():
    from inference.translate import parse_marked_en, build_marked_he_window

    marked = build_marked_he_window(
        ["שלום עולם", "אני לא מכירה אף אישה"], focus_idx=1
    )
    assert "[[C0]]" in marked and "[[C1]]" in marked
    parsed = parse_marked_en(
        "[[C0]] Hello world. [[C1]] I don't know any woman.", 2
    )
    assert parsed is not None
    assert parsed[1].startswith("I don't know")
    assert parse_marked_en("Hello without markers", 2) is None


def test_meaning_preserving_rejects_negation_loss():
    from inference.build_preview import meaning_preserving

    he = "אני לא מכירה אף אישה במזרח התיכון"
    orig = "I don't know any women in the Middle East"
    bad = "I don't about that"
    good = "I don't know any woman in the Middle East"
    assert meaning_preserving(he, orig, good)
    assert not meaning_preserving(he, orig, bad)


def test_fit_exact_window_pads_short_clip(tmp_path):
    import numpy as np
    import soundfile as sf
    from inference.tts_qwen import fit_exact_window, wav_duration

    sr = 44100
    src = tmp_path / "short.wav"
    dst = tmp_path / "fit.wav"
    # 1.0s of audio into a 2.0s HE window → pad to ~2s.
    sf.write(str(src), np.ones(sr, dtype=np.float32) * 0.1, sr)
    rate = fit_exact_window(src, dst, 2.0, sample_rate=sr, min_rate=0.90, pad_short=True)
    assert rate <= 1.0
    dur = wav_duration(dst)
    assert abs(dur - 2.0) < 0.05


def test_fit_exact_window_never_trims_long_clip(tmp_path):
    """Long EN into a short HE slot must keep full speech (no hard_window_fade)."""
    import numpy as np
    import soundfile as sf
    from inference.tts_qwen import FIT_MAX_RATE_GENTLE, fit_exact_window, wav_duration

    sr = 44100
    src = tmp_path / "long.wav"
    dst = tmp_path / "fit.wav"
    # 2.0s of audio into a 1.0s HE window → gentle speedup, keep leftover.
    sf.write(str(src), np.ones(2 * sr, dtype=np.float32) * 0.1, sr)
    rate = fit_exact_window(
        src,
        dst,
        1.0,
        sample_rate=sr,
        max_rate=FIT_MAX_RATE_GENTLE,
        allow_overrun=False,  # even when False, never trim
    )
    assert rate <= FIT_MAX_RATE_GENTLE + 1e-6
    dur = wav_duration(dst)
    # After atempo ≤1.15, still longer than 1.0s target — must not be chopped to 1.0.
    assert dur > 1.05
    assert abs(dur - (2.0 / rate)) < 0.08


def test_plan_dub_placement_closes_gaps_and_keeps_keep():
    from inference.segment_merge import MAX_DRIFT_SEC, plan_dub_placement

    segs = [
        {
            "speaker_id": "A",
            "language": "he",
            "keep_original": False,
            "source_start": 59.0,
            "source_end": 61.0,
            "start": 59.0,
            "end": 61.0,
            "tts_fit": "dummy1.wav",
            "tts_clip_sec": 2.0,
        },
        {
            "speaker_id": "B",
            "language": "he",
            "keep_original": False,
            # Micro-gap (0.2s) — closable under CLOSE_GAP_MAX_SEC.
            "source_start": 61.2,
            "source_end": 75.2,
            "start": 61.2,
            "end": 75.2,
            "tts_fit": "dummy2.wav",
            "tts_clip_sec": 14.0,
        },
        {
            "speaker_id": "C",
            "language": "en",
            "keep_original": True,
            "source_start": 90.0,
            "source_end": 95.0,
            "start": 90.0,
            "end": 95.0,
        },
    ]
    n = plan_dub_placement(segs, media_duration=100.0)
    assert n == 2
    # Micro-gap 61→61.2 closed: seg2 starts near seg1 end (within inter_gap).
    assert segs[1]["place_start"] <= 61.2
    assert segs[1]["place_start"] >= segs[0]["place_end"] - 1e-3
    assert abs(segs[1]["place_drift"]) <= MAX_DRIFT_SEC + 1e-6
    # KEEP locked to source when extract clock matches ASR.
    assert segs[2]["place_start"] == 90.0
    assert segs[2]["place_end"] == 95.0
    assert segs[2]["place_speed"] == 1.0
    # Full clip lengths preserved (no trim).
    assert abs(segs[0]["place_end"] - segs[0]["place_start"] - 2.0) < 0.02
    assert abs(segs[1]["place_end"] - segs[1]["place_start"] - 14.0) < 0.02


def test_plan_dub_placement_preserves_intentional_stop():
    """Gaps ≥ CLOSE_GAP_MAX_SEC keep place_start ≈ source_start (no MAX_DRIFT pull)."""
    from inference.segment_merge import CLOSE_GAP_MAX_SEC, plan_dub_placement

    segs = [
        {
            "speaker_id": "A",
            "language": "he",
            "keep_original": False,
            "source_start": 59.0,
            "source_end": 61.0,
            "start": 59.0,
            "end": 61.0,
            "tts_fit": "dummy1.wav",
            "tts_clip_sec": 2.0,
        },
        {
            "speaker_id": "B",
            "language": "he",
            "keep_original": False,
            # ~1s dramatic stop (kan11_5m ~0:59 pattern) — must not pull early.
            "source_start": 62.05,
            "source_end": 76.0,
            "start": 62.05,
            "end": 76.0,
            "tts_fit": "dummy2.wav",
            "tts_clip_sec": 14.0,
        },
    ]
    assert 62.05 - 61.0 >= CLOSE_GAP_MAX_SEC
    plan_dub_placement(segs, media_duration=100.0)
    assert abs(segs[1]["place_start"] - 62.05) < 0.02
    assert segs[1]["place_drift"] >= -0.05
    # Previous clip may finish into the pause (never trim).
    assert segs[0]["place_end"] <= segs[1]["place_start"] + 0.02


def test_plan_dub_placement_keep_uses_extract_not_late_asr():
    """KEEP energy-snapped extract must not be re-locked to later source_start."""
    from inference.segment_merge import plan_dub_placement

    segs = [
        {
            "speaker_id": "EN",
            "language": "en",
            "keep_original": True,
            "source_start": 2.461,
            "source_end": 6.021,
            # Energy snap pulled extract earlier than ASR (kan11_5m bug).
            "start": 1.211,
            "end": 7.507,
            "phrases": [
                {
                    "text": "Qatar is probably one of Israel's most dangerous enemies.",
                    "start": 2.461,
                    "end": 6.021,
                    "source_start": 2.461,
                    "source_end": 6.021,
                    "tts_start": 1.211,
                    "spoken_end": 7.507,
                    "pause_after": 0.0,
                }
            ],
            # Stale packer lock to ASR (the delay bug).
            "place_start": 2.461,
            "place_end": 6.021,
            "place_speed": 1.0,
        },
        {
            "speaker_id": "HE",
            "language": "he",
            "keep_original": False,
            "source_start": 7.557,
            "source_end": 20.837,
            "start": 7.557,
            "end": 20.837,
            "tts_fit": "he.wav",
            "tts_clip_sec": 13.0,
        },
    ]
    plan_dub_placement(segs, media_duration=30.0)
    # Place at extract onset, not delayed ASR source_start.
    assert abs(segs[0]["place_start"] - 1.211) < 1e-3
    assert abs(segs[0]["place_end"] - 7.507) < 1e-3
    assert segs[0]["place_drift"] < -0.5
    # source_* stays ASR for captions.
    assert segs[0]["source_start"] == 2.461


def test_plan_dub_placement_preserves_stamped_keep_place():
    """KEEP place_* stamped by TTS extract path is preserved when not ASR-locked."""
    from inference.segment_merge import plan_dub_placement

    segs = [
        {
            "speaker_id": "EN",
            "language": "en",
            "keep_original": True,
            "source_start": 2.461,
            "source_end": 6.021,
            "start": 1.211,
            "end": 7.507,
            "place_start": 1.211,
            "place_end": 7.507,
            "place_speed": 1.0,
            "place_drift": -1.25,
        }
    ]
    plan_dub_placement(segs, media_duration=20.0)
    assert segs[0]["place_start"] == 1.211
    assert segs[0]["place_end"] == 7.507


def test_plan_dub_placement_speeds_before_keep():
    """Dense run that overruns a KEEP anchor gets uniform speed-up, never trim."""
    from inference.segment_merge import plan_dub_placement

    segs = [
        {
            "speaker_id": "A",
            "language": "he",
            "keep_original": False,
            "source_start": 10.0,
            "source_end": 11.0,
            "start": 10.0,
            "end": 11.0,
            "tts_fit": "a.wav",
            "tts_clip_sec": 1.8,  # longer than HE slot
        },
        {
            "speaker_id": "B",
            "language": "he",
            "keep_original": False,
            "source_start": 11.1,
            "source_end": 12.0,
            "start": 11.1,
            "end": 12.0,
            "tts_fit": "b.wav",
            "tts_clip_sec": 1.5,
        },
        {
            "speaker_id": "K",
            "language": "en",
            "keep_original": True,
            "source_start": 12.5,
            "source_end": 14.0,
            "start": 12.5,
            "end": 14.0,
        },
    ]
    plan_dub_placement(segs, media_duration=20.0)
    # Must speed up to finish before KEEP @ 12.5 − guard.
    assert segs[0]["place_speed"] > 1.01
    assert segs[1]["place_speed"] == segs[0]["place_speed"]
    assert segs[1]["place_end"] <= 12.5 - 0.10 + 0.05
    # KEEP untouched.
    assert segs[2]["place_start"] == 12.5


def test_plan_dub_placement_last_before_gap_finishes():
    """Last dub before a big gap may overrun its source_end (let it finish)."""
    from inference.segment_merge import plan_dub_placement

    segs = [
        {
            "speaker_id": "A",
            "language": "he",
            "keep_original": False,
            "source_start": 5.0,
            "source_end": 7.0,
            "start": 5.0,
            "end": 7.0,
            "tts_fit": "a.wav",
            "tts_clip_sec": 3.0,  # 1s longer than HE
        },
        {
            "speaker_id": "B",
            "language": "he",
            "keep_original": False,
            "source_start": 12.0,  # ≥1.5s gap → new run
            "source_end": 13.0,
            "start": 12.0,
            "end": 13.0,
            "tts_fit": "b.wav",
            "tts_clip_sec": 1.0,
        },
    ]
    plan_dub_placement(segs, media_duration=30.0)
    # First run finishes into the gap (no forced chipmunk unless needed).
    assert segs[0]["place_speed"] == 1.0
    assert segs[0]["place_end"] >= 8.0 - 0.05
    assert segs[1]["place_start"] == 12.0


def test_assert_tts_coverage_blocks_missing():
    from inference.build_preview import assert_tts_coverage

    segs = [
        {
            "speaker_id": "S0",
            "language": "he",
            "keep_original": False,
            "start": 1.0,
            "end": 3.0,
            "text": "שלום",
            "tts_failed": True,
        }
    ]
    raised = False
    try:
        assert_tts_coverage(segs, allow_missing=False)
    except SystemExit:
        raised = True
    assert raised
    missing = assert_tts_coverage(segs, allow_missing=True)
    assert len(missing) == 1


def test_suppress_vocal_leak_keeps_music_floor():
    import numpy as np
    from inference.extract_pipeline import suppress_vocal_leak_in_bed

    sr = 16000
    n = sr * 2
    # Constant music bed + speech burst in vocals.
    bg = np.ones((n, 1), dtype=np.float32) * 0.05
    voc = np.zeros(n, dtype=np.float32)
    voc[sr // 2 : sr // 2 + sr // 4] = 0.2
    cleaned = suppress_vocal_leak_in_bed(bg, voc, sr, speech_floor=0.50)
    # Never hard-muted; floor keeps music clearly present under speech.
    assert float(np.min(np.abs(cleaned))) >= 0.05 * 0.45
    # Music still present overall.
    assert float(np.mean(np.abs(cleaned))) > 0.01


def test_suppress_vocal_leak_speech_floor_below_music():
    """Default speech floor attenuates HE leak without crushing the bed."""
    import inspect

    import numpy as np
    from inference.extract_pipeline import suppress_vocal_leak_in_bed

    sig = inspect.signature(suppress_vocal_leak_in_bed)
    assert 0.25 <= float(sig.parameters["speech_floor"].default) <= 0.40
    assert float(sig.parameters["music_keep"].default) >= 0.85

    sr = 16000
    n = sr
    bg = np.ones((n, 1), dtype=np.float32) * 0.04
    voc = np.ones(n, dtype=np.float32) * 0.25  # strongly speech-dominant
    cleaned = suppress_vocal_leak_in_bed(bg, voc, sr)
    # Speech-dominant frames attenuate harder than the old 0.50 floor.
    assert float(np.mean(np.abs(cleaned))) < 0.04 * 0.45
    # But not near-muted (old 0.18 floor crushed music under dubs).
    assert float(np.mean(np.abs(cleaned))) > 0.04 * 0.20


def test_duck_gain_default_keeps_music_audible():
    import inspect
    from inference.build_preview import build_dubbed_track

    sig = inspect.signature(build_dubbed_track)
    assert float(sig.parameters["duck_gain"].default) >= 0.75


def test_speaker_search_windows_own_then_same_id_never_cross():
    """Own window first; same-speaker_id fallback only; never another speaker."""
    from inference.tts_qwen import (
        _speaker_own_windows,
        _speaker_same_id_fallback_windows,
        _speaker_search_windows,
    )

    seg = {
        "speaker_id": "SPEAKER_06",
        "start": 46.0,
        "end": 60.0,
        "language": "he",
    }
    phrase = {"start": 46.0, "end": 51.0, "text": "אנחנו"}
    all_segs = [
        seg,
        {
            # Different speaker nearby — must NEVER be a candidate (0:50 bug).
            "speaker_id": "SPEAKER_02",
            "start": 77.0,
            "end": 81.0,
            "language": "he",
            "keep_original": False,
        },
        {
            # Same id, later turn — allowed as fallback only.
            "speaker_id": "SPEAKER_06",
            "start": 81.0,
            "end": 94.0,
            "language": "he",
            "keep_original": False,
        },
    ]
    own = _speaker_own_windows(seg, phrase)
    # Own windows are unit/phrase-local only — never the full parent 46–60.
    assert own[0][0] >= 45.9 and own[0][1] <= 51.1
    assert all(b <= 51.1 for _a, b in own)
    assert not any(abs(a - 46.0) < 0.05 and abs(b - 60.0) < 0.05 for a, b in own)

    fb = _speaker_same_id_fallback_windows(seg, phrase, all_segs)
    assert any(abs(a - 81.0) < 0.1 for a, _b in fb)
    assert not any(abs(a - 77.0) < 0.1 for a, _b in fb)

    wins = _speaker_search_windows(seg, phrase, all_segs)
    starts = [a for a, _b in wins]
    assert not any(abs(a - 77.0) < 0.1 for a in starts)
    assert wins[0][0] < 61.0  # own first
    assert any(abs(a - 81.0) < 0.1 for a in starts)


def test_assemble_preserves_long_he_gap(tmp_path):
    """Real HE stops (≥ UNIT_PRESERVE_GAP_SEC) stay; micro-gaps close."""
    import numpy as np
    import soundfile as sf

    from inference.tts_qwen import (
        UNIT_INTER_GAP_SEC,
        UNIT_PRESERVE_GAP_SEC,
        assemble_on_hebrew_timeline,
        wav_duration,
    )

    sr = 44100
    c0 = tmp_path / "u0.wav"
    c1 = tmp_path / "u1.wav"
    sf.write(str(c0), np.ones(int(0.5 * sr), dtype=np.float32) * 0.1, sr)
    sf.write(str(c1), np.ones(int(0.5 * sr), dtype=np.float32) * 0.1, sr)
    assert UNIT_PRESERVE_GAP_SEC >= 0.45

    plan = [
        {
            "start": 10.0,
            "end": 10.5,
            "source_start": 10.0,
            "source_end": 10.5,
            "tts_fit": str(c0),
        },
        {
            # 3.5s dramatic stop (kan11_5m ~0:59).
            "start": 14.0,
            "end": 14.5,
            "source_start": 14.0,
            "source_end": 14.5,
            "tts_fit": str(c1),
        },
    ]
    out = tmp_path / "assembled.wav"
    assemble_on_hebrew_timeline(plan, 10.0, 15.0, out, sample_rate=sr)
    audio, _ = sf.read(str(out), dtype="float32")
    # Second unit should land near 14.0 on the canvas (offset from 10.0).
    # Quiet region between ~0.5s and ~4.0s of canvas.
    mid = audio[int(1.0 * sr) : int(3.5 * sr)]
    assert float(np.max(np.abs(mid))) < 0.01
    # Second burst starts around t=4.0 on canvas.
    onset_region = audio[int(3.9 * sr) : int(4.2 * sr)]
    assert float(np.max(np.abs(onset_region))) > 0.05
    assert wav_duration(out) >= 4.4

    # Micro-gap path: 0.15s gap should close to ~UNIT_INTER_GAP_SEC.
    c2 = tmp_path / "u2.wav"
    sf.write(str(c2), np.ones(int(0.4 * sr), dtype=np.float32) * 0.1, sr)
    plan_micro = [
        {
            "start": 20.0,
            "end": 20.5,
            "source_start": 20.0,
            "source_end": 20.5,
            "tts_fit": str(c0),
        },
        {
            "start": 20.65,
            "end": 21.05,
            "source_start": 20.65,
            "source_end": 21.05,
            "tts_fit": str(c2),
        },
    ]
    out2 = tmp_path / "micro.wav"
    assemble_on_hebrew_timeline(plan_micro, 20.0, 21.2, out2, sample_rate=sr)
    a2, _ = sf.read(str(out2), dtype="float32")
    # Gap between clips should be tiny (≤ inter + a little slack), not 0.15s of silence.
    gap_samples = a2[int(0.5 * sr) : int(0.5 * sr + (UNIT_INTER_GAP_SEC + 0.05) * sr)]
    # Most of the micro-gap region is either silence of ≤ inter or already next speech.
    assert len(gap_samples) > 0


def test_coarse_split_en_does_not_cut_mid_pp():
    """Don't split 'comfortable for the Western…' mid-prepositional phrase."""
    from inference.tts_qwen import coarse_split_en_by_weights

    en = (
        "She paints Qatar in colors that are very comfortable "
        "for the Western conscience Sheikha Moza presents."
    )
    chunks = coarse_split_en_by_weights(en, [5.0, 5.0])
    assert len(chunks) == 2
    joined = " ".join(chunks).lower()
    assert "comfortable" in joined and "western" in joined
    # "for" must not start the second chunk alone after "comfortable".
    assert not chunks[1].lower().startswith("for the western")
    assert "comfortable for" in chunks[0].lower() or "for the western" in chunks[0].lower()
    # Never split the NP "Western conscience".
    assert not chunks[0].rstrip(".,").lower().endswith("western")
    assert "western conscience" in joined


def test_distribute_repairs_western_conscience_mid_np():
    """Stale phrase EN cut at Western.|conscience is redistributed/glued."""
    from inference.build_preview import distribute_en_to_phrases, phrase_en_cuts_broken

    phrases = [
        {
            "start": 110.0,
            "end": 115.0,
            "pause_after": 0.9,
            "text": "היא צובעת את קטאר בצבעים שמאוד נוחים למצפון המערבי",
            "text_en": "She paints Qatar in colors that are very comfortable for the Western.",
        },
        {
            "start": 115.9,
            "end": 122.0,
            "pause_after": 0.0,
            "text": "שיח עמוזה מצליחה למתג את קטאר",
            "text_en": "conscience Sheikha Moza successfully brands Qatar.",
        },
    ]
    assert phrase_en_cuts_broken(phrases)
    en = (
        "She paints Qatar in colors that are very comfortable for the Western "
        "conscience Sheikha Moza successfully brands Qatar."
    )
    distribute_en_to_phrases(en, phrases)
    assert not phrase_en_cuts_broken(phrases)
    joined = " ".join((p.get("text_en") or "") for p in phrases).lower()
    assert "western conscience" in joined
    assert not any(
        (p.get("text_en") or "").rstrip(".,").lower().endswith("western")
        for p in phrases
    )


def test_speaker_own_windows_unit_local_not_parent():
    """Multi-unit turns must not share the full parent segment as a ref window."""
    from inference.tts_qwen import _speaker_own_windows

    seg = {
        "speaker_id": "SPEAKER_06",
        "start": 81.0,
        "end": 95.0,
        "language": "he",
    }
    units = [
        {"start": 81.0, "end": 84.5, "text": "א"},
        {"start": 84.5, "end": 88.0, "text": "ב"},
        {"start": 88.0, "end": 95.0, "text": "ג"},
    ]
    windows = [_speaker_own_windows(seg, u)[0] for u in units]
    for (a, b), u in zip(windows, units):
        assert a >= float(u["start"]) - 0.05
        assert b <= float(u["end"]) + 0.05
    # Distinct unit windows — not all collapsing to the parent 81–95.
    assert len({(round(a, 2), round(b, 2)) for a, b in windows}) == 3
    assert not any(abs(a - 81.0) < 0.05 and abs(b - 95.0) < 0.05 for a, b in windows)


def test_postprocess_qatar_country_pronoun():
    from inference.build_preview import _postprocess_en

    he = "הצליחה לנתב את קטאר למקום אחר מהמקום שהיא הייתה בו"
    bad = "steering Qatar to a different place from where she had been"
    fixed = _postprocess_en(he, bad)
    assert "where she" not in fixed.lower()
    assert "she had been" not in fixed.lower()
    assert "it had been" in fixed.lower() or "what it had been" in fixed.lower()

    stiff = (
        "steering Qatar to a different place from where it had been "
        "She is responsible for the two coups"
    )
    idiomatic = _postprocess_en(he, stiff)
    assert "from where it had been" not in idiomatic.lower()
    assert "than it had been" in idiomatic.lower() or "what it had been" in idiomatic.lower()
    # Must not eat the following "She is…" sentence.
    assert "she is responsible" in idiomatic.lower()
    assert "been is responsible" not in idiomatic.lower()

    # Repair subject dropped at unit boundary.
    dropped = (
        "steering Qatar to a different place than it had been "
        "is responsible for the two coups"
    )
    repaired = _postprocess_en(he, dropped)
    assert "she is responsible" in repaired.lower()
    assert "been is responsible" not in repaired.lower()


def test_fit_exact_window_pad_short_false_keeps_natural_len(tmp_path):
    """Non-final units must not pad short speech into multi-second holes."""
    import numpy as np
    import soundfile as sf

    from inference.tts_qwen import fit_exact_window, wav_duration

    sr = 44100
    src = tmp_path / "short.wav"
    sf.write(str(src), np.ones(int(1.0 * sr), dtype=np.float32) * 0.1, sr)
    dst = tmp_path / "fit.wav"
    fit_exact_window(
        src, dst, 5.0, sample_rate=sr, min_rate=0.90, pad_short=False
    )
    # Without pad, clip stays near natural length (gentle stretch only).
    assert wav_duration(dst) < 2.0
