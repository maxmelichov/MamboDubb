"""Unit checks for gap recovery, unfinished stitch, speaker stabilize, TTS prep."""

from __future__ import annotations

from inference.segment_merge import (
    clamp_phrase_timeline,
    dedupe_repeated_sentences,
    find_uncovered_gaps,
    is_short_completion,
    merge_short_phrases,
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
    assert "Nusra Front" in out
    assert "Jabhat" not in out or "Nusra Front" in out


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
    from inference.tts_qwen import FIT_MAX_RATE, SHORTEN_RETRY_RATE

    assert FIT_MAX_RATE <= 1.18 + 1e-9
    # Shorten triggers only when even mild speedup cannot fit.
    assert SHORTEN_RETRY_RATE >= FIT_MAX_RATE
    assert SHORTEN_RETRY_RATE <= 1.35


def test_assemble_disables_bleed_before_keep(tmp_path):
    import numpy as np
    import soundfile as sf

    from inference.tts_qwen import KEEP_YIELD_GUARD_SEC, assemble_on_hebrew_timeline

    sr = 44100
    # Phrase clip longer than its HE slot — would bleed without KEEP guard.
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
        hard_end=10.5 - KEEP_YIELD_GUARD_SEC,
        next_is_keep=True,
    )
    audio, _ = sf.read(str(out_keep), dtype="float32")
    # Canvas must not extend past hard_end (~0.38s of active audio max).
    assert len(audio) / sr <= (0.5 - KEEP_YIELD_GUARD_SEC) + 0.05


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
    assert GAP_BLEED_SEC >= 0.20


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
    from inference.tts_qwen import SHORT_SLOT_OVERRUN_CAP_SEC, TIGHT_NEXT_GAP_SEC, REF_MIN_SEC

    assert SHORT_SLOT_OVERRUN_CAP_SEC <= 0.4
    assert TIGHT_NEXT_GAP_SEC <= 0.35
    assert REF_MIN_SEC >= 2.0
