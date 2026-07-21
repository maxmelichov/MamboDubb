#!/usr/bin/env python3
"""Build a watchable preview from Phase 1–2 outputs.

Takes a run directory (e.g. outputs/kan11_60s) with:
  segments.json, vocals.wav, background.wav, and the source video

Produces:
  translated_segments.json
  preview_en.srt
  dubbed_audio.wav
  preview.mp4   ← video + EN TTS over ducked BGM + soft EN subs

TTS (default: Qwen3-TTS 1.7B-Base zero-shot — https://arxiv.org/abs/2601.15621):
  --tts-engine qwen        clone from each phrase's vocal ref → English
  --tts-engine f5          F5-TTS zero-shot (legacy)

No lip-sync yet (LatentSync = Phase 5).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

from inference.segment_merge import (
    apply_place_speeds,
    clamp_segment_phrases,
    clamp_segment_timeline,
    dedupe_repeated_sentences,
    dedupe_segment_text_fields,
    is_continuation_start,
    merge_short_phrases,
    needs_object_continuation,
    plan_dub_placement,
    refresh_segment_fields,
    rehydrate_missing_asr_segments,
    restore_asr_phrase_boundaries,
    retag_english_sandwich,
    stamp_source_timing,
    source_end,
    source_start,
    utterance_unfinished,
)
from inference.tts_f5 import synthesize_segments_f5, wav_duration as f5_wav_duration
from inference.tts_qwen import (
    DEFAULT_MAX_DUB_PAUSE,
    DEFAULT_MODEL as QWEN_DEFAULT_MODEL,
    synthesize_segments_qwen,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# region agent log
_DEBUG_LOG_PATH = REPO_ROOT / ".cursor" / "debug-5e8424.log"


def _debug_event(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict,
) -> None:
    try:
        payload = {
            "sessionId": "5e8424",
            "runId": os.environ.get("DUB_DEBUG_RUN", "diagnostic"),
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
# endregion

# Default: TranslateGemma 4B with neighbor-chunk context.
# mlx-lm instruct LLM available via --translator llm (A/B).
DEFAULT_MLX_TRANSLATE_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
DEFAULT_GEMMA_MODEL = REPO_ROOT / "models" / "translategemma-4b-it"
MIN_SEG_DURATION = 0.35
HE_RE = re.compile(r"[\u0590-\u05FF]")

TRANSLATE_GLOSSARY = (
    "Glossary — always use these English forms when the Hebrew mentions them:\n"
    "- קטר / קטארי / קטרי / Qatar → Qatar / Qatari (never bacterial, catalysts, cats, Kat)\n"
    "- שייחה מוזה / שייח'ה מוזא / שיח' עמוזה / שייח עמוזה / מוזה → Sheikha Moza\n"
    "- חמאס → Hamas\n"
    "- חיזבאללה → Hezbollah\n"
    "- איראן / איראנים → Iran / Iranians\n"
    "- האחים המוסלמים / אחים מוסלמים → Muslim Brotherhood\n"
    "- פר אקסלנס / פר-אקסלנס → par excellence (never 'of excellence')\n"
    "- ג'בהת אל-נוסרה → Jabhat al-Nusra\n"
    "- מצפון / למצפון / למצפון המערבי → conscience / for the Western conscience "
    "(never 'north'; צפון=north, מצפון=conscience)\n"
    "- בשלוות נפש / בנינוחות → with composure / calmly (never 'clear conscience')\n"
    "- היא לא מאיימת → It doesn't threaten (neuter/country), not 'She is not threatening'\n"
    "- בנה / לבנה / בנה השני → her son / to her son / her second son "
    "(never 'daughter' — בן is masculine)\n"
    "General Hebrew grammar for dubbing:\n"
    "- Feminine possessives (בנה / בתה / בעלה) keep the woman's perspective "
    "(her son / her daughter / her husband).\n"
    "- Idioms like שלא נתבלבל → 'let's not get confused' (not literal calques).\n"
)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def looks_hebrew(text: str) -> bool:
    return bool(HE_RE.search(text or ""))


# Letters outside Latin (CJK, Cyrillic, Arabic, Hebrew, …) — must not reach TTS.
_LATIN_LETTER_RE = re.compile(r"[A-Za-z]")
# Caption / stage chrome that must not inflate EN length or be spoken.
_CAPTION_CHROME_RE = re.compile(
    r"(?:>>+\s*)|"
    r"\[(?:music|applause|מוזיקה|שירה|קהל)[^\]]*\]|"
    r"\((?:music|applause)\)",
    flags=re.IGNORECASE,
)


def is_latin_english(text: str) -> bool:
    """True when text has Latin letters and no non-Latin letter scripts (CJK/HE/AR/…)."""
    t = (text or "").strip()
    if not t:
        return False
    if not _LATIN_LETTER_RE.search(t):
        return False
    for ch in t:
        if ch.isalpha() and not ("A" <= ch <= "Z" or "a" <= ch <= "z"):
            return False
    return True


def strip_non_latin_runs(text: str) -> str:
    """Drop non-Latin letter runs; keep Latin, digits, and punctuation."""
    out: list[str] = []
    for ch in text or "":
        if ch.isalpha() and not ("A" <= ch <= "Z" or "a" <= ch <= "z"):
            out.append(" ")
        else:
            out.append(ch)
    return re.sub(r"\s+", " ", "".join(out)).strip(" ,.;:-")


def strip_caption_chrome(text: str) -> str:
    """Remove YouTube/ASR stage marks (>>, [music], [applause], …)."""
    t = _CAPTION_CHROME_RE.sub(" ", text or "")
    return re.sub(r"\s+", " ", t).strip(" ,.;:-")


def is_english_text(text: str) -> bool:
    t = (text or "").strip()
    if not t or looks_hebrew(t):
        return False
    return is_latin_english(t)


def preserve_latin_tokens(source: str, translated: str) -> str:
    """Re-inject Latin tokens from the source if the translator dropped them."""
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9'./-]*", source or "")
    out = (translated or "").strip()
    if not out:
        return out
    # Translator sometimes echoes Hebrew or collapses to bare names — don't "fix" that.
    if looks_hebrew(out):
        return out
    if tokens and len(out.split()) <= len(tokens):
        return out
    lower = out.lower()
    missing = [t for t in tokens if t.lower() not in lower]
    if not missing:
        return out
    return (out.rstrip(" .,") + " " + " ".join(dict.fromkeys(missing))).strip()


HARD_PAUSE_SEC = 0.80  # never pack EN across pauses this long


def _split_en_across_groups(en: str, group_weights: list[int]) -> list[str]:
    """Split one EN string into N chunks by HE word-weight, preferring punctuation."""
    n = len(group_weights)
    if n <= 0:
        return []
    if n == 1 or not en.strip():
        return [en.strip()]
    total_w = float(sum(max(1, w) for w in group_weights))
    cut_candidates = [
        m.end() for m in re.finditer(r"[,;:—–]\s+|(?<=[.!?…])\s+", en)
    ]
    words = en.split()
    if not words:
        return [en.strip()] * n

    chunks: list[str] = []
    cursor_char = 0
    assigned_words = 0
    for gi in range(n - 1):
        target_frac = sum(max(1, w) for w in group_weights[: gi + 1]) / total_w
        target_char = int(round(target_frac * len(en)))
        pool = [c for c in cut_candidates if cursor_char < c < len(en) - 2]
        if pool:
            best = min(pool, key=lambda c: abs(c - target_char))
            if abs(best - target_char) <= max(12, int(0.25 * len(en))):
                chunk = en[cursor_char:best].strip().rstrip(",")
                if chunk:
                    chunks.append(chunk)
                    cursor_char = best
                    assigned_words = len(en[:cursor_char].split())
                    continue
        remaining_groups = n - gi - 1
        need = max(
            1,
            round(len(words) * max(1, group_weights[gi]) / total_w),
        )
        need = min(need, max(1, len(words) - assigned_words - remaining_groups))
        dangling = {
            "the", "a", "an", "and", "or", "but", "of", "to", "with", "for",
            "in", "on", "at", "by", "from", "as", "is", "are", "was", "were",
            # Clause openers — never end a chunk mid-"but when …"
            "when", "if", "while", "because", "though", "although", "that",
            "which", "who", "whom", "whose", "than", "then",
        }
        while (
            need > 1
            and assigned_words + need < len(words)
            and words[assigned_words + need - 1].strip(".,;:!?\"'").lower() in dangling
        ):
            need -= 1
        # If we still land on a dangling opener, push the cut forward to the
        # next comma/sentence so the clause stays intact (pause slack can absorb).
        while (
            assigned_words + need < len(words)
            and words[assigned_words + need - 1].strip(".,;:!?\"'").lower() in dangling
        ):
            need += 1
            if words[assigned_words + need - 1].endswith((".", "!", "?", "…", ",", ";", ":")):
                break
            if need > len(words) - assigned_words - remaining_groups:
                need = max(1, len(words) - assigned_words - remaining_groups)
                break
        for k in range(assigned_words + 1, len(words) - remaining_groups + 1):
            if words[k - 1].endswith((".", "!", "?", "…", ",", ";", ":")):
                if abs((k - assigned_words) - need) <= max(3, need // 2):
                    need = k - assigned_words
                    break
        chunk_words = words[assigned_words : assigned_words + need]
        chunks.append(" ".join(chunk_words).strip())
        assigned_words += need
        cursor_char = len(" ".join(words[:assigned_words]))
        if cursor_char < len(en) and en[cursor_char : cursor_char + 1] == " ":
            cursor_char += 1
    tail = en[cursor_char:].strip() if cursor_char < len(en) else ""
    if not tail:
        tail = " ".join(words[assigned_words:]).strip()
    chunks.append(tail)
    while len(chunks) < n:
        chunks.append("")
    return chunks[:n]


def distribute_en_to_phrases(en: str, phrases: list[dict]) -> None:
    """Map one utterance translation onto phrases without chopping articles mid-clause.

    Prefer 1:1 sentence alignment when sentence counts match phrase counts.
    Hard pauses (>= HARD_PAUSE_SEC) are alignment anchors — never pack phrase
    groups across them; split EN at clause punctuation instead.
    """
    en = strip_caption_chrome(strip_non_latin_runs((en or "").strip()))
    en = dedupe_repeated_sentences(en)
    if not phrases:
        return
    if len(phrases) == 1 or not en:
        phrases[0]["text_en"] = en
        return

    def _he_vav_start(text: str) -> bool:
        t = (text or "").lstrip()
        return bool(t) and t[0] == "ו"

    en_sents = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", en) if s.strip()]
    he_sents_per = []
    for p in phrases:
        he = (p.get("text") or "").strip()
        he_sents_per.append(
            max(1, len([s for s in re.split(r"(?<=[.!?…؟])\s+", he) if s.strip()]))
        )

    if len(en_sents) == len(phrases):
        for p, sent in zip(phrases, en_sents):
            p["text_en"] = sent
        return
    if len(en_sents) == sum(he_sents_per):
        idx = 0
        for p, n in zip(phrases, he_sents_per):
            chunk = en_sents[idx : idx + n]
            idx += n
            p["text_en"] = " ".join(chunk).strip()
        return

    # Prefer HE speech-duration weights when timestamps exist; else word counts.
    if all(
        p.get("start") is not None and p.get("end") is not None for p in phrases
    ):
        weights = [
            max(1, int(round(max(0.25, float(p["end"]) - float(p["start"])) * 10)))
            for p in phrases
        ]
    else:
        weights = [max(1, len((p.get("text") or "").split())) for p in phrases]
    total_w = sum(weights)

    hard_pause_idxs: list[int] = []
    for i in range(len(phrases) - 1):
        gap = float(phrases[i + 1]["start"]) - float(phrases[i]["end"])
        pause = float(phrases[i].get("pause_after") or 0.0)
        if max(gap, pause) >= HARD_PAUSE_SEC:
            hard_pause_idxs.append(i)

    # Hard pauses: split into pause-separated groups, assign EN per group.
    # Never pack/collapse phrases across a hard pause. Rebuild parent list
    # because soft packing within a group may merge soft-gap phrases.
    if hard_pause_idxs:
        bounds = [-1] + hard_pause_idxs + [len(phrases) - 1]
        groups: list[list[dict]] = []
        group_weights: list[int] = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            # Copy dicts so soft packing can mutate group lists safely.
            group = [dict(p) for p in phrases[a + 1 : b + 1]]
            groups.append(group)
            group_weights.append(
                sum(max(1, len((g.get("text") or "").split())) for g in group)
            )
        if len(en_sents) == len(groups):
            en_chunks = en_sents
        else:
            en_chunks = _split_en_across_groups(en, group_weights)
        rebuilt: list[dict] = []
        for group, chunk in zip(groups, en_chunks):
            if len(group) == 1:
                group[0]["text_en"] = chunk
                rebuilt.extend(group)
            else:
                # Soft gaps only inside the group — may merge within the group.
                distribute_en_to_phrases(chunk, group)
                rebuilt.extend(group)
        phrases.clear()
        phrases.extend(rebuilt)
        _repair_dangling_en_phrases(phrases)
        return

    # Soft packing only when there are NO hard pauses — may merge short phrases.
    # Never rewrite HE start/end/text: ASR phrase windows are immutable. Only
    # attach text_en onto the existing phrase rows.
    if 1 <= len(en_sents) < len(phrases):
        sent_w = [max(1, len(s.split())) for s in en_sents]
        sent_total = float(sum(sent_w))
        p_i = 0
        for si, sw in enumerate(sent_w):
            need = (sw / sent_total) * total_w
            got = 0.0
            p_start = p_i
            if si == len(sent_w) - 1:
                p_i = len(phrases)
            else:
                while p_i < len(phrases) and (got < need - 1e-6 or p_i == p_start):
                    got += weights[p_i]
                    p_i += 1
                    remain_phr = len(phrases) - p_i
                    remain_sent = len(sent_w) - si - 1
                    if remain_phr <= remain_sent:
                        break
            group = phrases[p_start:p_i]
            if not group:
                continue
            # Put the full sentence on the first phrase of the span; clear the
            # rest so TTS units prefer the authoritative sentence once.
            group[0]["text_en"] = en_sents[si]
            for g in group[1:]:
                g["text_en"] = ""
        _repair_dangling_en_phrases(phrases)
        return

    en_clauses = [c.strip() for c in re.split(r",\s*", en) if c.strip()]
    if len(en_clauses) == len(phrases) and len(phrases) >= 2:
        for i, (p, clause) in enumerate(zip(phrases, en_clauses)):
            p["text_en"] = clause if i == len(phrases) - 1 else clause.rstrip(",") + ","
        return

    if len(phrases) == 2 and _he_vav_start(phrases[1].get("text") or ""):
        matches = list(re.finditer(r"(?:,|;)?\s+\band\b\s+", en, flags=re.I))
        if matches:
            he1 = (phrases[1].get("text") or "").lstrip()
            prefer_qatar = he1.startswith("וקטר") or he1.startswith("וקטאר")
            target = (weights[0] / total_w) * len(en)
            best = min(matches, key=lambda m: abs(m.start() - target))
            if prefer_qatar:
                qatar_hits = [
                    m
                    for m in matches
                    if en[m.end() : m.end() + 6].lower().startswith("qatar")
                ]
                if qatar_hits:
                    best = qatar_hits[-1]
            left = en[: best.start()].rstrip(" ,;")
            right = en[best.start() :].strip()
            right = re.sub(r"^[,;]\s*", "", right)
            if right.lower().startswith("and "):
                right = "And " + right[4:]
            if left and right:
                phrases[0]["text_en"] = left
                phrases[1]["text_en"] = right
                return

    words = en.split()
    dangling = {
        "the", "a", "an", "and", "or", "but", "of", "to", "with", "for",
        "in", "on", "at", "by", "from", "as", "is", "are", "was", "were",
        "it", "its", "she", "her", "he", "his", "they", "their", "we", "our",
    }
    assigned = 0
    for i, p in enumerate(phrases):
        if i == len(phrases) - 1:
            chunk = words[assigned:]
        else:
            n = max(1, round(len(words) * weights[i] / total_w))
            remaining_phrases = len(phrases) - i - 1
            n = min(n, max(1, len(words) - assigned - remaining_phrases))
            if _he_vav_start(phrases[i + 1].get("text") or ""):
                he_next = (phrases[i + 1].get("text") or "").lstrip()
                prefer_qatar = he_next.startswith("וקטר") or he_next.startswith("וקטאר")
                for k in range(assigned + 1, len(words) - remaining_phrases + 1):
                    tok = words[k].strip(".,;:!?\"'").lower()
                    if tok != "and":
                        continue
                    if prefer_qatar and k + 1 < len(words):
                        if words[k + 1].strip(".,;:").lower().startswith("qatar"):
                            n = k - assigned
                            break
                    if abs((k - assigned) - n) <= 5:
                        n = k - assigned
                        break
            while (
                n > 1
                and assigned + n < len(words)
                and words[assigned + n - 1].strip(".,;:!?\"'").lower() in dangling
            ):
                n -= 1
            rel = {"that", "which", "who", "whom", "whose"}
            while (
                assigned + n < len(words) - remaining_phrases
                and words[assigned + n].strip(".,;:!?\"'").lower() in rel
            ):
                n += 1
                while assigned + n < len(words) - remaining_phrases:
                    prev_tok = words[assigned + n - 1]
                    if prev_tok[-1:] in ".!?":
                        break
                    nxt = words[assigned + n].strip(".,;:!?\"'").lower()
                    if nxt in dangling and nxt not in rel:
                        break
                    n += 1
            chunk = words[assigned : assigned + n]
            assigned += len(chunk)
        text = " ".join(chunk).strip().rstrip(",")
        if i > 0 and _he_vav_start(p.get("text") or "") and text.lower().startswith("and "):
            text = "And " + text[4:]
        p["text_en"] = text

    _repair_dangling_en_phrases(phrases)



_REL_CLAUSE_START = re.compile(
    r"^(that|which|who|whom|whose)\b",
    re.I,
)


_TRAILING_DANGLING_EN = re.compile(
    r"^(.*?)\b("
    r"the|a|an|and|or|but|of|to|with|for|in|on|at|by|from|as|"
    r"when|if|while|because|though|although|that|which|who|whom|whose|than|then"
    r")(?:\s*[.,;:]*)?$",
    re.I | re.S,
)


def _repair_dangling_en_phrases(phrases: list[dict]) -> None:
    """Merge phrases whose EN is only a relative-clause tail into the previous.

    Stops word-weight chops like:
      p0: "She funds every Islamic terrorist organization"
      p1: "that operates in the world."
    which drop the Hebrew clause and sound like a skipped line.

    Also repairs trailing mid-clause cuts ("… but when.") by moving the
    dangling opener onto the next phrase when a hard pause does NOT follow
    — or by pulling the next phrase's EN back when the cut left an unfinished
    clause before a pause (prefer complete talking; pause is slack).
    """
    i = 1
    while i < len(phrases):
        en = (phrases[i].get("text_en") or "").strip()
        if not en or not _REL_CLAUSE_START.match(en):
            i += 1
            continue
        prev = phrases[i - 1]
        prev_en = (prev.get("text_en") or "").strip()
        prev["text_en"] = f"{prev_en} {en}".strip() if prev_en else en
        prev["text"] = (
            f"{(prev.get('text') or '').strip()} {(phrases[i].get('text') or '').strip()}"
        ).strip()
        prev["end"] = float(phrases[i]["end"])
        prev["pause_after"] = float(phrases[i].get("pause_after") or 0.0)
        for key in ("speaker_id",):
            if key in phrases[i] and key not in prev:
                prev[key] = phrases[i][key]
        phrases.pop(i)

    # Trailing dangling mid-clause ("… but when.") — rejoin and re-split so
    # talking stays complete. Prefer a comma cut that leaves a short adverbial
    # tail on the post-pause phrase when a hard pause follows.
    i = 0
    while i < len(phrases) - 1:
        cur = phrases[i]
        nxt = phrases[i + 1]
        cur_en = (cur.get("text_en") or "").strip()
        nxt_en = (nxt.get("text_en") or "").strip()
        if not cur_en or not nxt_en:
            i += 1
            continue
        if not _TRAILING_DANGLING_EN.match(cur_en):
            i += 1
            continue
        gap = float(nxt.get("start") or 0.0) - float(cur.get("end") or 0.0)
        pause = float(cur.get("pause_after") or 0.0)
        full = f"{cur_en} {nxt_en}".strip()
        full = re.sub(r"\s+", " ", full)
        if max(gap, pause) >= HARD_PAUSE_SEC:
            # Pause is slack: keep HE windows, put the main clause before the
            # pause and a short adverbial/tail after when possible.
            cuts = [
                m.start()
                for m in re.finditer(
                    r"(?:,\s+)?\b(with such|with so|in such)\b",
                    full,
                    flags=re.I,
                )
            ]
            if not cuts:
                cuts = [
                    m.start()
                    for m in re.finditer(
                        r",\s+(with|and then)\b", full, flags=re.I
                    )
                ]
            if cuts and cuts[-1] > len(full) * 0.35:
                cut = cuts[-1]
                left = full[:cut].strip().rstrip(",") + ","
                right = full[cut:].lstrip(", ").strip()
            else:
                # Fall back: keep ~70% of words before the pause (complete talking),
                # leave a short tail for the post-pause lips.
                words = full.split()
                keep = max(4, int(round(len(words) * 0.72)))
                keep = min(keep, len(words) - 2)
                while keep < len(words) - 1 and not words[keep - 1].endswith(
                    (",", ".", "!", "?", "…", ";", ":")
                ):
                    keep += 1
                    if keep >= len(words) - 1:
                        break
                left = " ".join(words[:keep]).strip()
                right = " ".join(words[keep:]).strip()
            cur["text_en"] = left
            nxt["text_en"] = right
        else:
            # Soft gap: merge phrase rows so talking isn't chopped.
            cur["text_en"] = full
            cur["text"] = (
                f"{(cur.get('text') or '').strip()} {(nxt.get('text') or '').strip()}"
            ).strip()
            cur["end"] = float(nxt["end"])
            cur["pause_after"] = float(nxt.get("pause_after") or 0.0)
            phrases.pop(i + 1)
            continue
        i += 1

    # Absorb empty-EN soft-gap phrases into the previous (keep hard pauses).
    i = 1
    while i < len(phrases):
        if (phrases[i].get("text_en") or "").strip():
            i += 1
            continue
        prev = phrases[i - 1]
        gap = float(phrases[i].get("start") or 0.0) - float(prev.get("end") or 0.0)
        pause = float(prev.get("pause_after") or 0.0)
        if max(gap, pause) >= HARD_PAUSE_SEC:
            # Hard pause with no EN — leave the row (caller may fill later).
            i += 1
            continue
        prev["text"] = (
            f"{(prev.get('text') or '').strip()} {(phrases[i].get('text') or '').strip()}"
        ).strip()
        prev["end"] = float(phrases[i]["end"])
        prev["pause_after"] = float(phrases[i].get("pause_after") or 0.0)
        phrases.pop(i)

    # Clear stale per-phrase TTS paths after a merge.
    for p in phrases:
        for key in ("tts_fit", "tts_raw", "tts_speed_used"):
            p.pop(key, None)


def _norm_he_key(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"[\u200e\u200f]", "", t)
    t = t.replace("קטארי", "קטרי").replace("כזאתי", "כזאת")
    t = re.sub(r"\s+", "", t)
    return t


def seed_translations_from_run(segments: list[dict], seed_path: Path) -> int:
    """Copy proven text_en from a shorter reference run when Hebrew/time overlap."""
    if not seed_path.is_file():
        return 0
    seed_segs = json.loads(seed_path.read_text(encoding="utf-8")).get("segments") or []
    seeded = 0
    for seg in segments:
        if seg.get("keep_original") or (seg.get("language") or "he") != "he":
            continue
        if is_english_text(seg.get("text_en") or ""):
            continue
        he = (seg.get("text") or "").strip()
        if not he:
            continue
        s0, s1 = float(seg["start"]), float(seg["end"])
        he_key = _norm_he_key(he)
        best = None
        best_score = 0.0
        for src in seed_segs:
            if src.get("keep_original") or (src.get("language") or "he") != "he":
                continue
            en = (src.get("text_en") or "").strip()
            if not is_english_text(en):
                continue
            # Never seed truncated / unfinished reference lines onto fuller HE.
            if en.rstrip().endswith("...") or en.rstrip().endswith("…"):
                continue
            src_he_raw = (src.get("text") or "").strip()
            if src_he_raw.rstrip().endswith("...") or src_he_raw.rstrip().endswith("…"):
                continue
            if len(en.split()) + 3 < len(he.split()):
                continue
            # Don't seed a shorter HE utterance onto a stitched-longer one
            # (extra clause → incomplete EN + bad phrase distribute → slow TTS).
            if len(he.split()) > len(src_he_raw.split()) + 2:
                continue
            src_n_phr = len(src.get("phrases") or [])
            tgt_n_phr = len(seg.get("phrases") or [])
            if src_n_phr and tgt_n_phr > src_n_phr:
                continue
            t0, t1 = float(src["start"]), float(src["end"])
            overlap = max(0.0, min(s1, t1) - max(s0, t0))
            dur = max(s1 - s0, 0.1)
            src_he = _norm_he_key(src_he_raw)
            score = overlap / dur
            if he_key and src_he and (he_key[:48] == src_he[:48] or he_key in src_he or src_he in he_key):
                score += 0.75
            if score > best_score:
                best_score = score
                best = src
        if best is None or best_score < 0.55:
            continue
        en = (best.get("text_en") or "").strip()
        # Final guard: seed must not be an ellipsis fragment.
        if en.rstrip().endswith(("...", "…")):
            continue
        seg["text_en"] = en
        phrases = seg.get("phrases") or []
        if phrases:
            distribute_en_to_phrases(en, phrases)
        seeded += 1
        print(
            f"  Seeded EN from {seed_path.parent.name} "
            f"[{seg.get('speaker_id')}] {s0:.1f}-{s1:.1f}s ← {best['start']:.1f}-{best['end']:.1f}s",
            file=sys.stderr,
        )
        print(f"    → {en[:100]}", file=sys.stderr)
    return seeded


def load_mlx_translator(model_id: str):
    try:
        from mlx_lm import load
    except ImportError as exc:
        raise SystemExit(
            "mlx-lm is required for --translator llm.\n"
            "Install with: uv sync"
        ) from exc
    print(f"Loading mlx-lm translator {model_id}...", file=sys.stderr)
    model, tokenizer = load(model_id)
    return model, tokenizer


def load_gemma_translator(model_path: Path | str, device: str = "auto"):
    from inference.translate import load_translategemma, resolve_device

    return load_translategemma(Path(model_path), device=resolve_device(device))


def translate_text_gemma(
    processor,
    model,
    text: str,
    *,
    device=None,
    duration: float | None = None,
    max_tokens: int = 220,
    prev_he: str | None = None,
    next_he: str | None = None,
    next_hes: list[str] | None = None,
) -> str:
    """Translate with TranslateGemma, optionally using neighbor HE context.

    When neighbors are provided, tries marker-aligned multi-chunk translation
    and returns only the focus chunk's English. Falls back to direct MT.
    """
    from inference.translate import translate_gemma, translate_gemma_marked_chunks

    he = (text or "").strip()
    if not he:
        return ""

    # Build a small neighbor window: prev + current + up to 2 next.
    chunks: list[str] = []
    focus_idx = 0
    if prev_he and prev_he.strip() and prev_he.strip() != he:
        chunks.append(prev_he.strip())
        focus_idx = 1
    chunks.append(he)
    upcoming = list(next_hes or [])
    if next_he and next_he.strip():
        upcoming = [next_he.strip()] + upcoming
    for u in upcoming:
        if u and u != he and len(chunks) < 4:
            chunks.append(u)

    en = ""
    if len(chunks) > 1:
        parsed = translate_gemma_marked_chunks(
            processor,
            model,
            chunks,
            focus_idx=focus_idx,
            device=device,
            max_new_tokens=max_tokens,
        )
        if parsed and focus_idx in parsed:
            en = parsed[focus_idx]
            print(
                f"  gemma marked-align ok ({len(parsed)}/{len(chunks)} chunks)",
                file=sys.stderr,
            )
    if not en:
        en = translate_gemma(
            processor,
            model,
            he,
            source="he",
            target="en",
            device=device,
            max_new_tokens=max_tokens,
        )
    en = _postprocess_en(he, en)
    if duration is not None and duration >= 0.8:
        if estimate_en_speaking_sec(en) > float(duration) * EN_FIT_MAX_RATIO:
            en = shorten_en_preserving_meaning(
                he,
                en,
                duration=float(duration),
                syllable_budget=max(6, int(float(duration) * 2.6)),
            )
    if not is_english_text(en):
        # Soft failure — caller / ensure_translations can fall back to mlx-lm.
        raise RuntimeError(
            f"TranslateGemma returned empty/Hebrew for: {he[:80]!r}\nGot: {en[:80]!r}"
        )
    return en


def _mlx_chat(model, tokenizer, system: str, user: str, *, max_tokens: int = 220) -> str:
    from mlx_lm import generate

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        text_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text_prompt = f"{system}\n\n{user}"
    return generate(
        model, tokenizer, prompt=text_prompt, max_tokens=max_tokens, verbose=False
    ).strip()


def _collapse_pronoun_verb_stub(en: str) -> str:
    """Drop a tiny 'She X.' stub when the next sentence repeats the same subject.

    General pattern for join artifacts like 'She colors. She paints Qatar…'
    — not tied to any specific verb or video.
    """
    t = (en or "").strip()
    if not t:
        return t
    m = re.match(
        r"^(She|He|It|They|We)\s+\S{1,20}\.\s+(\1\s+.+)$",
        t,
        flags=re.I | re.S,
    )
    if not m:
        return t
    first = t[: m.start(2)].strip()
    # Only collapse when the first sentence is a tiny stub (≤ 3 tokens).
    if len(first.split()) <= 3:
        return m.group(2).strip()
    return t


def _postprocess_en(he: str, en: str, *, prev_en: str | None = None) -> str:
    en = strip_caption_chrome(en.strip().strip('"').strip("'"))
    en = strip_non_latin_runs(en)
    en = preserve_latin_tokens(he, en)
    en = dedupe_repeated_sentences(en)
    en = _collapse_pronoun_verb_stub(en)
    en = re.sub(r"\bbacterial\b", "Qatari", en, flags=re.I)
    en = re.sub(r"\bcatalysts\b", "Qatar", en, flags=re.I)
    en = re.sub(r"\bagenda of excellence\b", "agenda par excellence", en, flags=re.I)
    en = re.sub(r"\bIslamic Brotherhood\b", "Muslim Brotherhood", en, flags=re.I)
    if "פר אקסלנס" in he or "פר-אקסלנס" in he:
        en = re.sub(r"\bof excellence\b", "par excellence", en, flags=re.I)
    # מצפון = conscience (never north); צפון = north
    if re.search(r"מצפון", he):
        en = re.sub(
            r"\b[Ww]estern [Nn]orth\b",
            "Western conscience",
            en,
        )
        en = re.sub(
            r"\bfor the (Western )?north\b",
            r"for the \1conscience".replace("  ", " "),
            en,
            flags=re.I,
        )
        en = re.sub(
            r"\bcomfortable for the (Western )?north\b",
            r"comfortable for the \1conscience".replace("  ", " "),
            en,
            flags=re.I,
        )
        # Bare "north" after Western / conscience context.
        if re.search(r"\bnorth\b", en, re.I) and not re.search(r"\bצפון\b", he):
            en = re.sub(r"\b[Nn]orth\b", "conscience", en)
    # בנה = her son (never daughter)
    if re.search(r"\bבנה\b|לבנה", he) and re.search(r"\bdaughter\b", en, re.I):
        en = re.sub(r"\b(second )?daughter\b", r"\1son", en, flags=re.I)
        en = re.sub(r"\bher son\b", "her son", en, flags=re.I)
        en = re.sub(r"\bhis second son\b", "her second son", en, flags=re.I)
    if prev_en:
        en = strip_leading_memory_echo(en, prev_en)
    return en


# Shared threshold for strip_leading_memory_echo and mt_needs_retry.
MEMORY_ECHO_NGRAM_THRESHOLD = 0.50


def _ngram_overlap_ratio(a: str, b: str, *, n: int = 4) -> float:
    """Fraction of a's word n-grams that also appear in b."""
    wa = [w.lower().strip(".,;:!?\"'") for w in (a or "").split() if w.strip()]
    wb = [w.lower().strip(".,;:!?\"'") for w in (b or "").split() if w.strip()]
    if len(wa) < n or len(wb) < n:
        # Fall back to shared word ratio for short strings.
        if not wa:
            return 0.0
        sa, sb = set(wa), set(wb)
        return len(sa & sb) / max(1, len(sa))
    na = {" ".join(wa[i : i + n]) for i in range(len(wa) - n + 1)}
    nb = {" ".join(wb[i : i + n]) for i in range(len(wb) - n + 1)}
    if not na:
        return 0.0
    return len(na & nb) / len(na)


def strip_leading_memory_echo(en: str, prev_en: str) -> str:
    """Drop leading EN sentences that duplicate the previous memory EN."""
    en = (en or "").strip()
    prev = (prev_en or "").strip()
    if not en or not prev:
        return en
    prev_norm = re.sub(r"\s+", " ", prev.lower()).strip(" .")
    sents = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", en) if s.strip()]
    if not sents:
        return en
    kept: list[str] = []
    dropped = False
    for i, s in enumerate(sents):
        sn = re.sub(r"\s+", " ", s.lower()).strip(" .")
        # Exact / near-exact echo of previous sentence(s).
        if sn and (sn in prev_norm or prev_norm in sn):
            dropped = True
            continue
        if (
            _ngram_overlap_ratio(s, prev, n=4) >= MEMORY_ECHO_NGRAM_THRESHOLD
            and i == 0
        ):
            dropped = True
            continue
        kept.append(s)
    if not kept:
        # All sentences were echo — keep the last short one if HE-sized stub.
        return sents[-1] if sents else en
    out = " ".join(kept)
    if dropped:
        return out
    return en


def _force_shorten_en_for_he(he: str, en: str) -> str:
    """Deprecated hard truncate — prefer meaning-preserving shorten.

    Kept as last-resort only when no LLM/Gemma rewrite is available. Prefer
    clause boundaries over raw word chops.
    """
    he_words = max(1, len((he or "").split()))
    budget = max(6, he_words * 3)
    words = (en or "").split()
    if len(words) <= budget:
        return (en or "").strip()
    # Prefer cutting at a sentence/clause boundary inside the budget.
    cut_at = budget
    for k in range(budget, max(3, budget // 2), -1):
        tok = words[k - 1]
        if tok.endswith((".", "!", "?", "…", ";", ",")):
            cut_at = k
            break
    cut = " ".join(words[:cut_at]).rstrip(" ,;:")
    if cut and cut[-1] not in ".!?":
        cut += "."
    return cut


_MEANING_NEGATION = re.compile(
    r"\b(not|no|never|none|neither|don't|doesn't|didn't|isn't|aren't|won't|"
    r"can't|cannot|without)\b",
    re.I,
)
_MEANING_QUANT = re.compile(
    r"\b(any|every|all|none|most|many|few|each|both|some|no)\b",
    re.I,
)


def meaning_preserving(he: str, original_en: str, candidate_en: str) -> bool:
    """True when candidate keeps names, numbers, negation, and quantifiers."""
    orig = (original_en or "").strip()
    cand = (candidate_en or "").strip()
    if not cand or not is_english_text(cand):
        return False
    # Proper nouns / glossary tokens present in original must survive.
    for tok in re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", orig):
        if tok.lower() in {"the", "and", "for", "with", "from", "that", "this"}:
            continue
        if tok.lower() not in cand.lower():
            # Allow minor respelling only for known glossary forms.
            if tok.lower() in {"sheikha", "moza", "qatar", "hamas", "hezbollah"}:
                if tok.lower() not in cand.lower():
                    return False
            else:
                return False
    # Numbers must survive.
    for num in re.findall(r"\b\d+[\w./-]*\b", orig):
        if num not in cand:
            return False
    # Negation / quantifier polarity.
    if _MEANING_NEGATION.search(orig) and not _MEANING_NEGATION.search(cand):
        return False
    if _MEANING_QUANT.search(orig) and not _MEANING_QUANT.search(cand):
        # Soft: only require when HE also has אין/אף/כל/שום.
        if re.search(r"(אין|אף|כל|שום|לא)", he or ""):
            return False
    # Don't allow collapsing to a tiny stub relative to original.
    if len(cand.split()) < max(3, len(orig.split()) // 4):
        return False
    return True


def shorten_en_preserving_meaning(
    he: str,
    en: str,
    *,
    duration: float,
    syllable_budget: int,
    shorten_fn=None,
) -> str:
    """Compress EN for timing without dropping core meaning.

    Tries ``shorten_fn`` (LLM) first; rejects meaning-breaking rewrites.
    Never hard-truncates mid-clause when a meaning-preserving option exists.
    """
    en0 = (en or "").strip()
    if not en0:
        return en0
    if shorten_fn is not None:
        try:
            shorter = shorten_fn(
                he_text=he,
                en_text=en0,
                duration=duration,
                syllable_budget=syllable_budget,
            )
        except Exception:
            shorter = None
        if shorter and meaning_preserving(he, en0, shorter):
            return strip_caption_chrome(strip_non_latin_runs(shorter))
    # Soft clause drop from the end: remove trailing clauses until budget.
    sents = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", en0) if s.strip()]
    if len(sents) > 1:
        kept = list(sents)
        while len(kept) > 1:
            trial = " ".join(kept[:-1]).strip()
            if estimate_en_speaking_sec(trial) <= duration * EN_FIT_MAX_RATIO:
                if meaning_preserving(he, en0, trial):
                    return trial
                break
            kept.pop()
    return en0


def sanitize_mt_after_retry(
    he: str,
    en: str,
    *,
    memory: list[tuple[str, str]] | None = None,
    reason: str | None = None,
    duration: float | None = None,
) -> tuple[str, bool]:
    """Strip echo / force-shorten after a failed MT retry.

    Returns (en, ok_for_memory). Never feed echo/overlong EN into rolling memory.
    """
    en = strip_caption_chrome((en or "").strip())
    en = strip_non_latin_runs(en)
    en = dedupe_repeated_sentences(en)
    if memory:
        en = strip_leading_memory_echo(en, memory[-1][1])
    retry_reasons = ("en_too_long_for_he", "memory_echo", "non_english_script", "time_overflow")
    flagged = reason in retry_reasons or mt_needs_retry(
        he, en, memory=memory, duration=duration
    ) in retry_reasons
    if flagged:
        if memory:
            en = strip_leading_memory_echo(en, memory[-1][1])
        still = mt_needs_retry(he, en, memory=memory, duration=duration)
        if still in ("en_too_long_for_he", "memory_echo", "time_overflow"):
            en = shorten_en_preserving_meaning(
                he,
                en,
                duration=float(duration or 1.0),
                syllable_budget=max(6, int(float(duration or 1.0) * 2.6)),
            )
            if memory:
                en = strip_leading_memory_echo(en, memory[-1][1])
            # Only hard-truncate if still overflowing AND meaning check allows.
            still3 = mt_needs_retry(he, en, memory=memory, duration=duration)
            if still3 in ("en_too_long_for_he", "time_overflow"):
                truncated = _force_shorten_en_for_he(he, en)
                if meaning_preserving(he, en, truncated):
                    en = truncated
                # else keep the longer meaning-preserving EN; TTS will compress.
        if still == "non_english_script":
            en = strip_non_latin_runs(en)
        still2 = mt_needs_retry(he, en, memory=memory, duration=duration)
        if still2 in retry_reasons:
            return en, False
    return en, True


def estimate_en_speaking_sec(en: str) -> float:
    """Rough natural English speaking duration (~2.8 words/sec + small pad)."""
    words = max(1, len((en or "").split()))
    return words / 2.8 + 0.2


# Isochronic band: EN speaking time should land ~0.8–1.2× the Hebrew chunk.
EN_FIT_MIN_RATIO = 0.80
EN_FIT_MAX_RATIO = 1.20


def mt_needs_retry(
    he: str,
    en: str,
    *,
    memory: list[tuple[str, str]] | None = None,
    duration: float | None = None,
) -> str | None:
    """Return a reason string if the translation looks broken."""
    en = strip_caption_chrome((en or "").strip())
    if en and not is_latin_english(en):
        return "non_english_script"
    en_n = dedupe_repeated_sentences(en)
    if en_n != en.strip() and len(en.split()) > len(en_n.split()) + 2:
        return "repeated_clauses"
    he_words = max(1, len((he or "").split()))
    en_words = len((en or "").split())
    # Short HE must not balloon into a paragraph (memory echo symptom).
    if he_words <= 6 and en_words > max(8, he_words * 4):
        return "en_too_long_for_he"
    # Isochronic overflow: EN would need >1.2× the speaking window.
    if duration is not None and duration >= 0.8:
        est = estimate_en_speaking_sec(en)
        if est > float(duration) * EN_FIT_MAX_RATIO:
            return "time_overflow"
    if memory:
        prev_en = (memory[-1][1] or "").strip()
        if (
            prev_en
            and _ngram_overlap_ratio(en, prev_en, n=4) >= MEMORY_ECHO_NGRAM_THRESHOLD
        ):
            return "memory_echo"
    # בנה must not become daughter
    if re.search(r"\bבנה\b|לבנה", he or "") and re.search(
        r"\bdaughter\b", en or "", re.I
    ):
        return "bana_gender"
    # Dropped glossary names that appear in HE
    checks = [
        ("קטר", "qatar"),
        ("חמאס", "hamas"),
        ("חיזבאללה", "hezbollah"),
        ("מוזה", "moza"),
        ("אחים המוסלמים", "brotherhood"),
    ]
    he_l = he
    en_l = en.lower()
    for he_tok, en_tok in checks:
        if he_tok in he_l and en_tok not in en_l:
            return f"missing_{en_tok}"
    return None


def build_episode_context(model, tokenizer, segments: list[dict]) -> dict:
    """Pass 0: synopsis + auto-glossary from the full Hebrew transcript."""
    he_lines = []
    for seg in segments:
        if seg.get("keep_original") or (seg.get("language") or "he") != "he":
            continue
        t = (seg.get("text") or "").strip()
        if t:
            he_lines.append(t)
    blob = "\n".join(he_lines)[:4500]
    if not blob.strip():
        return {"synopsis": "", "glossary_extra": ""}
    print("Building episode synopsis + glossary (pass 0)…", file=sys.stderr)
    raw = _mlx_chat(
        model,
        tokenizer,
        "You extract documentary context for Hebrew→English dubbing. "
        "Output exactly two sections: SYNOPSIS: and GLOSSARY:",
        (
            "From this Hebrew news-documentary transcript, write:\n"
            "SYNOPSIS: 2–4 English sentences summarizing the episode topic and key people.\n"
            "GLOSSARY: up to 12 lines of `Hebrew → English` proper nouns / recurring terms "
            "(names, orgs, places). Prefer established English spellings "
            "(Sheikha Moza, Qatar, Hamas, Muslim Brotherhood, Hezbollah).\n\n"
            f"Transcript:\n{blob}\n"
        ),
        max_tokens=350,
    )
    synopsis = ""
    glossary_extra = ""
    if "GLOSSARY:" in raw:
        syn_part, gloss_part = raw.split("GLOSSARY:", 1)
        synopsis = syn_part.replace("SYNOPSIS:", "").strip()
        glossary_extra = gloss_part.strip()
    else:
        synopsis = raw.replace("SYNOPSIS:", "").strip()
    print(f"  synopsis: {synopsis[:160]}…", file=sys.stderr)
    return {"synopsis": synopsis, "glossary_extra": glossary_extra}


def translate_text_mlx(
    model,
    tokenizer,
    text: str,
    duration: float | None = None,
    max_tokens: int = 220,
    prev_he: str | None = None,
    next_he: str | None = None,
    *,
    synopsis: str | None = None,
    glossary_extra: str | None = None,
    memory: list[tuple[str, str]] | None = None,
    next_hes: list[str] | None = None,
    hard_syllable_budget: int | None = None,
    youtube_he: str | None = None,
) -> str:
    he = text.strip()
    if not he:
        return ""
    timing = ""
    if hard_syllable_budget is not None:
        timing = (
            f"HARD LIMIT: at most ~{hard_syllable_budget} syllables. "
            "Compress ruthlessly but keep names and core meaning.\n"
        )
    elif duration is not None and duration >= 0.8:
        # Target EN speaking time ≈ HE chunk (0.8–1.2× band).
        target_syllables = max(6, int(duration * 3.0))
        lo = max(4, int(duration * 2.4))
        hi = max(lo + 2, int(duration * 3.6))
        timing = (
            f"The Hebrew takes about {duration:.1f}s. Aim for {target_syllables} "
            f"syllables (roughly {lo}–{hi}) so English fits the same window "
            f"(about 0.8–1.2× the original length). Prefer concise phrasing; "
            "never sacrifice names or core meaning.\n"
        )
    ctx_parts: list[str] = []
    if synopsis and synopsis.strip():
        ctx_parts.append(f"Episode context:\n{synopsis.strip()[:500]}")
    gloss = TRANSLATE_GLOSSARY
    if glossary_extra and glossary_extra.strip():
        gloss = gloss + "Extra glossary:\n" + glossary_extra.strip()[:600] + "\n"
    if memory:
        mem_lines = []
        for he_m, en_m in memory[-3:]:
            mem_lines.append(f"HE: {he_m[:200]}\nEN: {en_m[:220]}")
        ctx_parts.append(
            "Recent translations (continuity only — do NOT repeat):\n"
            + "\n\n".join(mem_lines)
        )
    if prev_he and prev_he.strip() and prev_he.strip() != he and not memory:
        ctx_parts.append(
            f"Previous segment (context only — do NOT translate):\n{prev_he.strip()[:320]}"
        )
    upcoming = list(next_hes or [])
    if next_he and next_he.strip():
        upcoming = [next_he.strip()] + upcoming
    upcoming = [u for u in upcoming if u and u != he][:2]
    if upcoming:
        ctx_parts.append(
            "Upcoming Hebrew (context only — do NOT translate):\n"
            + "\n".join(u[:280] for u in upcoming)
        )
    ctx = ("\n\n".join(ctx_parts) + "\n\n") if ctx_parts else ""
    prompt = (
        "You are an expert Hebrew→English dubbing translator for a news documentary.\n"
        f"{gloss}"
        f"{timing}"
        "Use context only for pronouns, continuity, and names. "
        "Translate ONLY the current Hebrew segment below.\n"
        "The Hebrew may be noisy ASR — if a word is nonsense in context, "
        "translate the intended meaning (do not invent new facts).\n"
        "If Current YouTube caption is provided, treat it as a second ASR opinion "
        "and prefer the reading that makes sense in context.\n"
        "If the current text starts with ו- (and…), keep that subject in English "
        "(e.g. וקטר → And Qatar / Qatar also…).\n"
        "Never repeat the same clause twice. Never invent extra sentences.\n"
        f"{ctx}"
        "Provide ONLY the English translation of the current segment, "
        "with no quotes, notes, or explanations.\n\n"
        f"Current Hebrew: {he}\n"
    )
    if youtube_he and youtube_he.strip() and youtube_he.strip() != he:
        prompt += f"Current YouTube caption (second ASR): {youtube_he.strip()[:400]}\n"
    prompt += "English:"
    response = _mlx_chat(
        model,
        tokenizer,
        "You are a professional dubbing translator. Output only the raw English "
        "translation of the current segment. Preserve proper nouns from the glossary exactly.",
        prompt,
        max_tokens=max_tokens,
    )
    en = _postprocess_en(
        he,
        response,
        prev_en=(memory[-1][1] if memory else None),
    )
    if not is_english_text(en):
        raise SystemExit(
            f"Translator returned empty/Hebrew for: {he[:80]!r}\nGot: {en[:80]!r}"
        )
    return en


def shorten_english_mlx(
    model,
    tokenizer,
    *,
    he_text: str,
    en_text: str,
    duration: float,
    syllable_budget: int,
    synopsis: str | None = None,
) -> str:
    """Ask the translator for a compressed EN rewrite that fits the slot."""
    prompt = (
        "Rewrite the English dub line so it fits a short speaking window.\n"
        f"{TRANSLATE_GLOSSARY}"
        f"HARD LIMIT: about {syllable_budget} syllables (~{duration:.1f}s). "
        "Keep all proper nouns. Do not add new facts. Output ONLY the shorter English.\n"
    )
    if synopsis:
        prompt += f"Episode context: {synopsis[:300]}\n"
    prompt += (
        f"Hebrew: {he_text.strip()[:400]}\n"
        f"Current English: {en_text.strip()[:500]}\n"
        "Shorter English:"
    )
    raw = _mlx_chat(
        model,
        tokenizer,
        "You compress dubbing lines. Output only the shorter English sentence(s).",
        prompt,
        max_tokens=160,
    )
    return _postprocess_en(he_text, raw)


def retag_phonetic_english(segments: list[dict]) -> int:
    """Apply should_keep_original gate to HE rows (score-first; markers as tie-break).

    On KEEP flip, preserve HE-script ASR text — do not sanitize into empty Latin.
    """
    from inference.lang_detect import script_ratios, should_keep_original

    n = 0
    for i, seg in enumerate(segments):
        lang = seg.get("language") or "he"
        keep = bool(seg.get("keep_original", lang != "he"))
        if keep or lang != "he":
            continue
        text = (seg.get("text") or "").strip()
        neighbors: list[dict] = []
        if i > 0:
            neighbors.append(segments[i - 1])
        if i + 1 < len(segments):
            neighbors.append(segments[i + 1])
        if not should_keep_original(
            text,
            lang=lang,
            he_score=seg.get("he_score"),
            en_score=seg.get("en_score"),
            neighbors=neighbors,
        ):
            continue
        seg["language"] = "en"
        seg["keep_original"] = True
        if script_ratios(text)["en"] >= 0.35:
            seg["text_en"] = text
            for p in seg.get("phrases") or []:
                p["text_en"] = (p.get("text") or "").strip()
        n += 1
        print(
            f"  Retag KEEP [{seg.get('speaker_id')}] "
            f"{float(seg['start']):.1f}-{float(seg['end']):.1f}s | {text[:60]}",
            file=sys.stderr,
        )
    return n


def enforce_speaker_language_consistency(
    segments: list[dict],
    *,
    min_keep_frac: float = 0.45,
    min_keep_count: int = 2,
) -> int:
    """Flip low-confidence HE-dub turns to KEEP when the speaker is mostly English.

    Fixes English interviewees whose occasional turn was mis-ASR'd as Hebrew
    (e.g. SPEAKER_08 at ~1:40). Dense real-Hebrew commentary is protected by
    `_real_hebrew_commentary`.
    """
    from inference.lang_detect import _real_hebrew_commentary, script_ratios

    by_spk: dict[str, list[int]] = {}
    for i, seg in enumerate(segments):
        spk = str(seg.get("speaker_id") or "")
        if not spk:
            continue
        by_spk.setdefault(spk, []).append(i)

    n = 0
    for spk, idxs in by_spk.items():
        if len(idxs) < 2:
            continue
        keep_n = 0
        dub_n = 0
        for i in idxs:
            seg = segments[i]
            lang = seg.get("language") or "he"
            keep = bool(seg.get("keep_original", lang != "he"))
            if keep or lang == "en":
                keep_n += 1
            elif lang == "he":
                dub_n += 1
        total = keep_n + dub_n
        if total < 2 or keep_n < min_keep_count:
            continue
        if keep_n / total < min_keep_frac:
            continue
        # Speaker is mostly English — flip remaining HE dubs unless dense HE.
        for i in idxs:
            seg = segments[i]
            lang = seg.get("language") or "he"
            keep = bool(seg.get("keep_original", lang != "he"))
            if keep or lang != "he":
                continue
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            he_score = seg.get("he_score")
            en_score = seg.get("en_score")
            ratios = script_ratios(text)
            # Low Whisper HE confidence → treat as mis-tagged English even when
            # ASR text looks like dense Hebrew (common on EN natives).
            low_he_conf = he_score is not None and float(he_score) < 0.35
            competitive_en = (
                he_score is not None
                and en_score is not None
                and float(en_score) >= float(he_score) - 0.5
            )
            if _real_hebrew_commentary(text) and not low_he_conf and not competitive_en:
                continue
            weak_he = (
                low_he_conf
                or ratios["he"] < 0.70
                or competitive_en
                or (he_score is not None and float(he_score) < 0.45)
            )
            if not weak_he and ratios["he"] >= 0.85:
                continue
            seg["language"] = "en"
            seg["keep_original"] = True
            seg.pop("text_en", None)  # KEEP path will re-ASR / suppress
            for p in seg.get("phrases") or []:
                p.pop("text_en", None)
            n += 1
            print(
                f"  Speaker-consistency KEEP [{spk}] "
                f"{float(seg['start']):.1f}-{float(seg['end']):.1f}s "
                f"(speaker {keep_n}/{total} EN) | {text[:60]}",
                file=sys.stderr,
            )
    return n


def ensure_translations(
    segments: list[dict],
    model,
    tokenizer,
    *,
    episode_ctx: dict | None = None,
    translator: str = "gemma",
    gemma_device=None,
) -> None:
    """Translate Hebrew with episode synopsis + rolling HE→EN memory.

    translator='gemma' uses TranslateGemma (model=processor, tokenizer=gemma_model).
    translator='llm' uses mlx-lm (model, tokenizer).
    """
    use_gemma = translator == "gemma"

    def _seg_he(seg: dict) -> str:
        """Hebrew source for MT: video ASR is the authority.

        YouTube captions (if present) are optional context only — they must not
        replace the ASR transcript that owns the timeline.
        """
        return (seg.get("text") or "").strip()

    if episode_ctx is None and model is not None and not use_gemma:
        episode_ctx = build_episode_context(model, tokenizer, segments)
    episode_ctx = episode_ctx or {}
    synopsis = episode_ctx.get("synopsis") or ""
    glossary_extra = episode_ctx.get("glossary_extra") or ""
    memory: list[tuple[str, str]] = []
    skip_until = -1

    def _translate_one(
        he_src: str,
        *,
        slot_sec: float,
        next_hes_local: list[str] | None = None,
        youtube_he: str | None = None,
        hard_budget: int | None = None,
        max_tokens: int = 220,
        prev_he: str | None = None,
    ) -> str:
        if use_gemma:
            return translate_text_gemma(
                model,
                tokenizer,
                he_src,
                device=gemma_device,
                duration=slot_sec,
                max_tokens=max_tokens,
                prev_he=prev_he,
                next_hes=next_hes_local,
            )
        return translate_text_mlx(
            model,
            tokenizer,
            he_src,
            duration=slot_sec,
            synopsis=synopsis or None,
            glossary_extra=glossary_extra or None,
            memory=list(memory),
            next_hes=next_hes_local,
            youtube_he=youtube_he,
            hard_syllable_budget=hard_budget,
            max_tokens=max_tokens,
        )

    for i, seg in enumerate(segments):
        if i < skip_until:
            continue
        lang = seg.get("language") or "he"
        if lang == "skip" or not (seg.get("text") or "").strip():
            continue
        keep = bool(seg.get("keep_original", lang != "he"))
        phrases = seg.get("phrases") or []
        if phrases:
            phrases = merge_short_phrases(phrases)
            seg["phrases"] = phrases
            clamp_segment_phrases(seg)
            phrases = seg.get("phrases") or []
            seg["text"] = " ".join(
                (p.get("text") or "").strip() for p in phrases
            ).strip() or seg.get("text", "")
            if phrases:
                seg["start"] = float(phrases[0]["start"])
                seg["end"] = float(phrases[-1]["end"])
                seg["duration"] = round(seg["end"] - seg["start"], 3)

        if keep:
            if phrases:
                for p in phrases:
                    p["text_en"] = dedupe_repeated_sentences(
                        (p.get("text_en") or p.get("text") or "").strip()
                    )
                seg["text_en"] = dedupe_repeated_sentences(
                    " ".join(
                        (p.get("text_en") or "").strip()
                        for p in phrases
                        if p.get("text_en")
                    )
                )
            else:
                seg["text_en"] = dedupe_repeated_sentences(
                    (seg.get("text_en") or seg.get("text") or "").strip()
                )
            print(
                f"Keep original [{i+1}/{len(segments)}] {lang}: {(seg.get('text') or '')[:50]}…",
                file=sys.stderr,
            )
            continue

        he = (seg.get("_translate_text") or _seg_he(seg)).strip()
        if not he:
            continue
        if is_english_text(seg.get("text_en") or ""):
            seg["text_en"] = dedupe_repeated_sentences(
                strip_caption_chrome(strip_non_latin_runs(seg["text_en"]))
            )
            if phrases:
                distribute_en_to_phrases(seg["text_en"], phrases)
            memory.append((he, seg["text_en"]))
            seg.pop("_translate_text", None)
            continue
        if model is None:
            raise SystemExit("Translator required for Hebrew segments without English text_en")

        def _syllable_budget(slot_sec: float, *, tight: bool = False) -> int:
            # ~3.0 syl/sec natural; tight → ~2.6 so TTS atempo stays ≤1.25×.
            rate = 2.6 if tight else 3.0
            return max(6, int(float(slot_sec) * rate))

        def _finalize_mt(
            he_src: str,
            en_raw: str,
            *,
            slot_sec: float,
            next_hes_local: list[str] | None = None,
            youtube_he: str | None = None,
        ) -> tuple[str, bool]:
            """QA-retry + isochronic shorten. Returns (en, ok_for_memory)."""
            budget = _syllable_budget(slot_sec)
            en = strip_caption_chrome(strip_non_latin_runs(en_raw))
            reason = mt_needs_retry(he_src, en, memory=memory, duration=slot_sec)
            ok_for_memory = True
            if reason:
                print(f"  MT QA retry ({reason})…", file=sys.stderr)
                tight = reason in ("time_overflow", "en_too_long_for_he")
                en = _translate_one(
                    he_src,
                    slot_sec=slot_sec,
                    next_hes_local=next_hes_local,
                    youtube_he=youtube_he,
                    hard_budget=_syllable_budget(slot_sec, tight=tight),
                    max_tokens=240,
                )
                en = strip_caption_chrome(strip_non_latin_runs(en))
                en = dedupe_repeated_sentences(en)
                if memory:
                    en = strip_leading_memory_echo(en, memory[-1][1])
                reason2 = mt_needs_retry(he_src, en, memory=memory, duration=slot_sec)
                if reason2 in ("time_overflow", "en_too_long_for_he"):
                    print(f"  MT shorten ({reason2})…", file=sys.stderr)
                    if use_gemma:
                        en = shorten_en_preserving_meaning(
                            he_src,
                            en,
                            duration=slot_sec,
                            syllable_budget=_syllable_budget(slot_sec, tight=True),
                        )
                    else:
                        try:
                            shorter = shorten_english_mlx(
                                model,
                                tokenizer,
                                he_text=he_src,
                                en_text=en,
                                duration=slot_sec,
                                syllable_budget=_syllable_budget(slot_sec, tight=True),
                                synopsis=synopsis or None,
                            )
                            if shorter and meaning_preserving(he_src, en, shorter):
                                en = strip_caption_chrome(strip_non_latin_runs(shorter))
                            elif shorter and is_english_text(shorter):
                                print(
                                    "  MT shorten rejected (meaning loss)",
                                    file=sys.stderr,
                                )
                        except Exception as exc:
                            print(f"  MT shorten failed: {exc}", file=sys.stderr)
                    reason2 = mt_needs_retry(
                        he_src, en, memory=memory, duration=slot_sec
                    )
                if reason2:
                    print(f"  MT QA still flagged ({reason2})", file=sys.stderr)
                    en, ok_for_memory = sanitize_mt_after_retry(
                        he_src,
                        en,
                        memory=memory,
                        reason=reason2,
                        duration=slot_sec,
                    )
            en = strip_caption_chrome(strip_non_latin_runs(en))
            en = dedupe_repeated_sentences(en)
            return en, ok_for_memory

        # Absorb unfinished / tiny stubs INTO the next segment (one TTS unit).
        # Avoids "she paints / she paints" echo from join_only_first_clause.
        if i + 1 < len(segments):
            nxt = segments[i + 1]
            nxt_he = (nxt.get("text") or "").strip()
            nxt_keep = bool(
                nxt.get("keep_original") or (nxt.get("language") or "he") != "he"
            )
            he_words = len(he.split())
            nxt_words = len(nxt_he.split())
            seg_dur = float(seg.get("end", 0) - float(seg.get("start", 0)))
            nxt_dur = float(nxt.get("end", 0) - float(nxt.get("start", 0)))
            gap = float(nxt.get("start", 0) - float(seg.get("end", 0)))
            same_spk = str(seg.get("speaker_id")) == str(nxt.get("speaker_id"))
            # Object-seeking stub (היא צובעת.) + את … on next: absorb even across
            # speakers — diarization often flips mid-sentence here.
            object_cont = needs_object_continuation(he, nxt_he)
            short_stub = (
                he_words <= 3
                and seg_dur <= 1.2
                and (nxt_words >= 4 or object_cont)
                and not nxt_keep
                and nxt_he
                and 0 <= gap <= 1.2
                and (
                    (same_spk and utterance_unfinished(he))
                    or object_cont
                )
                and (
                    object_cont
                    or nxt_he.startswith("את ")
                    or nxt_he.startswith("ו")
                    or is_continuation_start(nxt_he)
                )
            )
            # Absorb short next only for same speaker OR object-continuation
            # (never JOIN independent cross-speaker turns).
            absorb_short_next = (
                not nxt_keep
                and nxt_he
                and utterance_unfinished(he)
                and (same_spk or object_cont)
                and (nxt_words <= 6 or nxt_dur <= 2.0 or object_cont)
                and 0 <= gap <= 1.2
                and not short_stub
            )
            if short_stub:
                # Merge stub into NEXT: prepend HE, extend start, keep next speaker.
                print(
                    f"  Absorb stub [{i+1}→{i+2}] into continuation "
                    f"({seg_dur:.1f}s + {nxt_dur:.1f}s): {he[:40]}…",
                    file=sys.stderr,
                )
                stub_phrases = list(seg.get("phrases") or [])
                nxt_phrases = list(nxt.get("phrases") or [])
                for p in stub_phrases:
                    p = dict(p)
                    p.pop("text_en", None)
                    nxt_phrases.insert(0, p)
                nxt["phrases"] = nxt_phrases
                nxt["start"] = float(seg["start"])
                nxt["text"] = f"{he} {nxt_he}".strip()
                nxt["duration"] = round(float(nxt["end"]) - float(nxt["start"]), 3)
                yt_stub = (seg.get("text_youtube") or "").strip()
                yt_nxt = (nxt.get("text_youtube") or "").strip()
                if yt_stub or yt_nxt:
                    nxt["text_youtube"] = f"{yt_stub} {yt_nxt}".strip()
                nxt.pop("text_en", None)
                for p in nxt.get("phrases") or []:
                    p.pop("text_en", None)
                refresh_segment_fields(nxt)
                # Tombstone the stub so the loop skips TTS for it.
                seg["text"] = ""
                seg["text_en"] = ""
                seg["phrases"] = []
                seg["language"] = "skip"
                seg["keep_original"] = False
                seg["_absorbed_into"] = i + 1
                continue
            if absorb_short_next:
                # Short continuation absorbed into CURRENT (existing path).
                join_he = f"{he} {nxt_he}".strip()
                dur = float(nxt["end"]) - float(seg["start"])
                next_hes = []
                for j in range(i + 2, min(i + 4, len(segments))):
                    t = (segments[j].get("text") or "").strip()
                    if t:
                        next_hes.append(t)
                print(
                    f"Translating [{i+1}+{i+2}/{len(segments)}] JOIN ({dur:.1f}s) "
                    f"{join_he[:70]}…",
                    file=sys.stderr,
                )
                yt = (seg.get("text_youtube") or nxt.get("text_youtube") or None)
                en = _translate_one(
                    join_he,
                    slot_sec=dur,
                    next_hes_local=next_hes,
                    youtube_he=yt,
                    hard_budget=_syllable_budget(dur),
                )
                en, ok_for_memory = _finalize_mt(
                    join_he, en, slot_sec=dur, next_hes_local=next_hes, youtube_he=yt
                )
                print(f"  → {en}", file=sys.stderr)
                # Map full joined EN onto current + next by HE-duration weight —
                # never duplicate the tail onto both segments (Qatar funds… bug).
                he_w = max(1, he_words)
                nxt_w = max(1, nxt_words)
                total_w = float(he_w + nxt_w)
                words = en.split()
                if len(words) >= 2:
                    cut = max(1, min(len(words) - 1, round(len(words) * he_w / total_w)))
                    # Prefer a nearby sentence/clause boundary.
                    for k in range(cut, max(1, cut - 4), -1):
                        if words[k - 1].endswith((".", "!", "?", "…", ",", ";")):
                            cut = k
                            break
                    en_head = " ".join(words[:cut]).strip().rstrip(",")
                    en_tail = " ".join(words[cut:]).strip()
                    if en_tail.lower().startswith("and "):
                        en_tail = "And " + en_tail[4:]
                else:
                    en_head = en
                    en_tail = ""
                seg["text_en"] = en_head
                if phrases:
                    distribute_en_to_phrases(en_head, phrases)
                nxt_phrases = nxt.get("phrases") or []
                if nxt_phrases and en_tail:
                    distribute_en_to_phrases(en_tail, nxt_phrases)
                    nxt["text_en"] = en_tail
                elif en_tail:
                    nxt["text_en"] = en_tail
                else:
                    # Tiny stub join: whole EN stays on current; next gets empty
                    # so repair_missing_english can fill if needed.
                    nxt.pop("text_en", None)
                if ok_for_memory:
                    memory.append((join_he, en))
                else:
                    print(
                        "  MT memory skip (echo/overlong after retry)",
                        file=sys.stderr,
                    )
                skip_until = i + 2
                continue

        next_hes = []
        for j in range(i + 1, min(i + 3, len(segments))):
            t = (segments[j].get("text") or "").strip()
            if t:
                next_hes.append(t)

        prev_he = None
        if memory:
            prev_he = memory[-1][0]
        elif i > 0:
            prev_he = (segments[i - 1].get("text") or "").strip() or None

        dur = float(seg.get("duration") or (float(seg["end"]) - float(seg["start"])))
        print(
            f"Translating [{i+1}/{len(segments)}] ({dur:.1f}s) {he[:60]}…",
            file=sys.stderr,
        )
        yt = (seg.get("text_youtube") or None)
        en = _translate_one(
            he,
            slot_sec=dur,
            next_hes_local=next_hes,
            youtube_he=yt,
            hard_budget=_syllable_budget(dur),
            prev_he=prev_he,
        )
        en, ok_for_memory = _finalize_mt(
            he, en, slot_sec=dur, next_hes_local=next_hes, youtube_he=yt
        )
        print(f"  → {en}", file=sys.stderr)

        he_src = (seg.get("_translate_text") or he).strip()
        seg["text_en"] = en
        if phrases:
            distribute_en_to_phrases(en, phrases)
        if ok_for_memory:
            memory.append((he_src, en))
        else:
            print(
                "  MT memory skip (echo/overlong after retry)",
                file=sys.stderr,
            )
        seg.pop("_translate_text", None)


def repair_missing_english(segments: list[dict], model=None, tokenizer=None) -> int:
    """Fill any HE dub rows still missing text_en (e.g. wiped by sentence-split)."""
    fixed = 0
    for i, seg in enumerate(segments):
        if seg.get("keep_original") or (seg.get("language") or "he") != "he":
            continue
        phrases = seg.get("phrases") or []
        en = (seg.get("text_en") or "").strip()
        need_phrase = bool(phrases) and any(
            not (p.get("text_en") or "").strip() for p in phrases
        )
        if is_english_text(en):
            en = dedupe_repeated_sentences(en)
            seg["text_en"] = en
            if need_phrase:
                distribute_en_to_phrases(en, phrases)
                fixed += 1
            continue
        he = (seg.get("text") or "").strip()
        if not he:
            continue
        if model is None or tokenizer is None:
            raise SystemExit(
                f"Missing English text_en for dub segment {seg.get('speaker_id')}: {he[:80]!r}"
            )
        prev_he = next_he = ""
        if i > 0:
            prev_he = (segments[i - 1].get("text") or "").strip()
        if i + 1 < len(segments):
            next_he = (segments[i + 1].get("text") or "").strip()
        dur = float(seg.get("duration") or (float(seg["end"]) - float(seg["start"])))
        print(
            f"Repair-translate [{i+1}/{len(segments)}] ({dur:.1f}s) {he[:60]}…",
            file=sys.stderr,
        )
        en = translate_text_mlx(
            model, tokenizer, he, duration=dur, prev_he=prev_he or None, next_he=next_he or None
        )
        print(f"  → {en}", file=sys.stderr)
        seg["text_en"] = en
        if phrases:
            distribute_en_to_phrases(en, phrases)
        fixed += 1
    return fixed


def srt_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - math.floor(seconds)) * 1000))
    if ms == 1000:
        s += 1
        ms = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments: list[dict], path: Path, text_key: str = "text_en") -> None:
    lines: list[str] = []
    idx = 1
    for seg in segments:
        lang = seg.get("language") or "he"
        keep = bool(seg.get("keep_original", lang != "he"))
        phrases = seg.get("phrases") or []
        if phrases:
            for p in phrases:
                text = (p.get(text_key) or p.get("text") or "").strip()
                if not text:
                    continue
                start = float(p.get("start", seg["start"]))
                end = max(float(p.get("end", seg["end"])), start + 0.4)
                speaker = seg.get("speaker_id", "")
                tag = f"{speaker}/{lang}" if keep else speaker
                lines.append(str(idx))
                lines.append(f"{srt_timestamp(start)} --> {srt_timestamp(end)}")
                lines.append(f"[{tag}] {text}" if tag else text)
                lines.append("")
                idx += 1
            continue

        text = (seg.get(text_key) or seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg["start"])
        end = max(float(seg["end"]), start + 0.4)
        speaker = seg.get("speaker_id", "")
        tag = f"{speaker}/{lang}" if keep else speaker
        lines.append(str(idx))
        lines.append(f"{srt_timestamp(start)} --> {srt_timestamp(end)}")
        lines.append(f"[{tag}] {text}" if tag else text)
        lines.append("")
        idx += 1
    path.write_text("\n".join(lines), encoding="utf-8")


def wav_duration(path: Path) -> float:
    return f5_wav_duration(path)


def _load_mono(path: Path, sample_rate: int):
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sr == sample_rate:
        return mono
    # Linear resample for rare rate mismatches (TTS / Demucs are usually 44.1k).
    n_out = int(round(len(mono) * sample_rate / sr))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=len(mono), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, mono).astype(np.float32)


def build_dubbed_track(
    segments: list[dict],
    background: Path,
    total_duration: float,
    workdir: Path,
    *,
    bg_gain: float = 0.55,
    duck_gain: float = 0.60,
    speech_gain: float = 1.2,
    speech_target_rms: float = 0.085,
    vocals: Path | None = None,
    speech_gaps: list[dict] | None = None,
) -> Path:
    """Place TTS clips on a timeline and sum with a continuous ducked bed.

    Under HE dubs: keep music/SFX audible at ``duck_gain`` with smooth
    attack/release. Residual Hebrew speech is suppressed via a leak-cleaned
    bed (spectral/energy soft attenuation) — never by hard-muting the music.
    Never mix original HE vocals under TTS.

    KEEP-original spans use the source mix clip and equal-power crossfade
    with the bed.
    """
    import numpy as np
    import soundfile as sf

    from inference.extract_pipeline import (
        write_clean_background,
        write_leak_cleaned_background,
    )

    raw_bg = workdir / "background_raw.wav"
    if not raw_bg.is_file() and background.is_file():
        import shutil

        shutil.copy2(background, raw_bg)
    if raw_bg.is_file():
        write_clean_background(raw_bg, background)
    # Soft leak suppression — keeps music continuous under speech energy.
    if vocals is not None and Path(vocals).is_file():
        leak_bg = workdir / "background_leakclean.wav"
        try:
            write_leak_cleaned_background(background, Path(vocals), leak_bg)
            background = leak_bg
        except Exception as exc:
            print(f"  leak-clean skipped: {exc}", file=sys.stderr)

    bg, sr = sf.read(str(background), dtype="float32", always_2d=True)
    n_samples = max(1, int(round(total_duration * sr)))
    if bg.shape[0] < n_samples:
        bg = np.pad(bg, ((0, n_samples - bg.shape[0]), (0, 0)))
    else:
        bg = bg[:n_samples]

    speech = np.zeros(n_samples, dtype=np.float32)
    # Continuous duck envelope: 1.0 = full bed, duck_gain under HE dubs.
    # Never forced to 0 for dubs/failures (that caused jumpy silent holes).
    duck = float(np.clip(duck_gain, 0.15, 1.0))
    bg_env = np.ones(n_samples, dtype=np.float32)

    def _apply_duck(t0: float, t1: float, level: float, *, attack=0.08, release=0.15) -> None:
        i0 = max(0, int(round(float(t0) * sr)))
        i1 = min(n_samples, int(round(float(t1) * sr)))
        if i1 <= i0:
            return
        a = max(1, int(attack * sr))
        r = max(1, int(release * sr))
        # Attack into duck level.
        a_end = min(i1, i0 + a)
        for k, idx in enumerate(range(i0, a_end)):
            t = (k + 1) / a
            target = 1.0 * (1.0 - t) + level * t
            bg_env[idx] = min(bg_env[idx], target)
        # Sustain.
        if a_end < i1 - r:
            bg_env[a_end : i1 - r] = np.minimum(bg_env[a_end : i1 - r], level)
        # Release back toward 1.0.
        r_start = max(a_end, i1 - r)
        for k, idx in enumerate(range(r_start, i1)):
            t = (k + 1) / max(1, i1 - r_start)
            target = level * (1.0 - t) + 1.0 * t
            bg_env[idx] = min(bg_env[idx], max(level, target))

    for seg in segments:
        keep = bool(seg.get("keep_original") or seg.get("keep_uses_source"))
        failed = bool(seg.get("tts_failed"))
        is_dub = (
            not keep
            and (seg.get("language") or "he") == "he"
        )
        if keep:
            a = float(seg.get("source_start") or seg.get("start") or 0.0)
            b = float(seg.get("source_end") or seg.get("end") or 0.0)
            if b <= a:
                continue
            # KEEP brings full source mix — bed silenced under the KEEP clip
            # with equal-power edges applied when placing speech.
            _apply_duck(a, b, 0.0, attack=0.04, release=0.06)
        elif is_dub:
            # Duck under the *placed* voice (may drift from ASR source window).
            a = float(
                seg.get("place_start")
                or seg.get("source_start")
                or seg.get("start")
                or 0.0
            )
            b = float(
                seg.get("place_end")
                or seg.get("spoken_end")
                or seg.get("source_end")
                or seg.get("end")
                or 0.0
            )
            b_guard = b + 0.20
            if b_guard <= a:
                continue
            # Successful or failed: keep music at duck floor (never hard mute).
            _apply_duck(a, b_guard, duck, attack=0.08, release=0.18)
        elif failed:
            a = float(seg.get("source_start") or seg.get("start") or 0.0)
            b = float(seg.get("source_end") or seg.get("end") or 0.0)
            if b + 0.35 > a:
                _apply_duck(a, b + 0.35, duck, attack=0.08, release=0.18)

    for gap in speech_gaps or []:
        # Uncovered speech energy: duck (not mute) so music stays continuous.
        _apply_duck(
            float(gap["start"]),
            float(gap["end"]) + 0.25,
            duck,
            attack=0.08,
            release=0.15,
        )

    placed = 0
    missing_tts = 0
    ordered = sorted(
        segments,
        key=lambda s: (float(s.get("start") or 0.0), 0 if s.get("keep_original") else 1),
    )
    for seg in ordered:
        if seg.get("tts_failed"):
            missing_tts += 1
            continue
        fitted = seg.get("tts_fit")
        if not fitted:
            if not seg.get("keep_original") and (seg.get("language") or "he") == "he":
                missing_tts += 1
            continue
        path = Path(fitted)
        if not path.is_file():
            print(f"  skip missing TTS clip: {path}", file=sys.stderr)
            missing_tts += 1
            continue
        clip = _load_mono(path, sr)
        keep = bool(seg.get("keep_original") or seg.get("keep_uses_source"))
        # KEEP stays locked to ASR onset; dubs use elastic place_start.
        if keep:
            i0 = int(round(float(seg.get("source_start") or seg["start"]) * sr))
        else:
            i0 = int(
                round(
                    float(
                        seg.get("place_start")
                        or seg.get("source_start")
                        or seg["start"]
                    )
                    * sr
                )
            )
        if i0 >= n_samples or len(clip) == 0:
            continue
        i1 = min(n_samples, i0 + len(clip))
        take = clip[: i1 - i0].copy()
        fade = min(int(0.015 * sr), len(take) // 4)
        if keep:
            if fade > 1:
                ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
                take[:fade] *= ramp
                take[-fade:] *= ramp[::-1]
            speech[i0:i1] = take
            bg_env[i0:i1] = 0.0
        else:
            clip_rms = float(np.sqrt(np.mean(take**2) + 1e-12))
            if clip_rms > 1e-4:
                gain = speech_target_rms / clip_rms
                take *= min(gain, 1.25)
            fade = min(int(0.012 * sr), len(take) // 4)
            if fade > 1:
                ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
                take[:fade] *= ramp
                take[-fade:] *= ramp[::-1]
            dest = speech[i0:i1]
            overlap = min(fade if fade > 1 else int(0.04 * sr), len(take), len(dest))
            if overlap > 1 and float(np.max(np.abs(dest[:overlap]))) > 1e-4:
                ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
                dest[:overlap] = dest[:overlap] * (1.0 - ramp) + take[:overlap] * ramp
                if len(take) > overlap:
                    dest[overlap:] = take[overlap:]
            else:
                dest[:] = take
            speech[i0:i1] = dest
            # Ensure duck floor under the placed EN clip.
            bg_env[i0:i1] = np.minimum(bg_env[i0:i1], duck)
        placed += 1

    # Final envelope smooth (~60 ms) so KEEP↔dub transitions don't click.
    fade_bg = max(1, int(0.06 * sr))
    gate = np.convolve(bg_env, np.ones(fade_bg, dtype=np.float32) / float(fade_bg), mode="same")
    gate = np.clip(gate, 0.0, 1.0)

    mix = bg * (bg_gain * gate)[:, None] + (speech * speech_gain)[:, None]
    # KEEP clips are already full-mix; don't boost them as hard as TTS.
    keep_mask = 1.0 - gate
    if float(np.max(keep_mask)) > 0.01 and speech_gain != 1.0:
        mix = bg * (bg_gain * gate)[:, None] + (
            speech * (speech_gain * gate + 1.0 * keep_mask)
        )[:, None]
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.98:
        mix *= 0.98 / peak

    out_wav = workdir / "dubbed_audio.wav"
    print(
        f"Mixing dubbed audio (numpy): {placed} clips, "
        f"bg_gain={bg_gain} duck={duck} (continuous leak-cleaned bed), "
        f"missing_tts={missing_tts}, speech_rms≈{speech_target_rms} → {out_wav}",
        file=sys.stderr,
    )

    # region agent log
    mix_windows = []
    bed_mono = bg.mean(axis=-1)
    for idx, seg in enumerate(segments):
        a = float(seg.get("source_start") or seg.get("start") or 0.0)
        b = float(seg.get("source_end") or seg.get("end") or 0.0)
        i0 = max(0, min(n_samples, int(round(a * sr))))
        i1 = max(i0, min(n_samples, int(round(b * sr))))
        if i1 <= i0:
            continue
        speech_slice = speech[i0:i1]
        bed_slice = bed_mono[i0:i1]
        gate_slice = gate[i0:i1]
        frame = max(1, int(0.05 * sr))
        speech_frame_rms = [
            float(np.sqrt(np.mean(speech_slice[j : j + frame] ** 2) + 1e-12))
            for j in range(0, max(0, len(speech_slice) - frame + 1), frame)
        ]
        mix_windows.append(
            {
                "index": idx,
                "speaker": seg.get("speaker_id"),
                "start": round(a, 3),
                "end": round(b, 3),
                "keep": bool(seg.get("keep_original") or seg.get("keep_uses_source")),
                "ttsFailed": bool(seg.get("tts_failed")),
                "hasTtsFit": bool(seg.get("tts_fit")),
                "speechRms": round(
                    float(np.sqrt(np.mean(speech_slice**2) + 1e-12)), 6
                ),
                "bedRms": round(
                    float(np.sqrt(np.mean(bed_slice**2) + 1e-12)), 6
                ),
                "gateMin": round(float(np.min(gate_slice)), 4),
                "gateMean": round(float(np.mean(gate_slice)), 4),
                "silentSpeechFrameFraction": round(
                    (
                        sum(value < 0.004 for value in speech_frame_rms)
                        / len(speech_frame_rms)
                    )
                    if speech_frame_rms
                    else 1.0,
                    4,
                ),
            }
        )
    _debug_event(
        "H1,H4,H5",
        "inference/build_preview.py:build_dubbed_track",
        "Measure speech coverage and bed envelope in source windows",
        {
            "placed": placed,
            "missingTts": missing_tts,
            "windows": mix_windows,
        },
    )
    # endregion

    sf.write(str(out_wav), mix.astype(np.float32), sr, subtype="PCM_16")
    return out_wav


def find_source_video(workdir: Path, payload: dict) -> Path:
    src = Path(payload.get("source") or "")
    if src.is_file():
        return src
    p = workdir / "source_video.mp4"
    if p.is_file():
        return p
    for p in workdir.glob("*.mp4"):
        if p.name != "preview.mp4":
            return p
    raise SystemExit(f"No source video found in {workdir}")


def mux_preview(
    video: Path,
    audio: Path,
    srt: Path,
    out_mp4: Path,
    duration: float,
) -> None:
    print(f"Muxing preview → {out_mp4}", file=sys.stderr)
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-i",
            str(audio),
            "-i",
            str(srt),
            "-t",
            f"{duration:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-map",
            "2:0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-c:s",
            "mov_text",
            "-metadata:s:s:0",
            "language=eng",
            "-shortest",
            "-movflags",
            "+faststart",
            str(out_mp4),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"ffmpeg mux failed (exit {result.returncode})")



def log_speech_gaps(segments: list[dict], vocals: Path, workdir: Path) -> list[dict]:
    """Log uncovered holes >=1.5s that still have vocal energy (missed ASR)."""
    from inference.segment_merge import find_uncovered_gaps, gap_has_speech_energy

    if not vocals.is_file():
        return []
    try:
        media_end = f5_wav_duration(vocals)
    except Exception:
        return []
    gaps = find_uncovered_gaps(segments, media_end, min_gap=1.5)
    hits: list[dict] = []
    for a, b in gaps:
        if not gap_has_speech_energy(vocals, a, b):
            continue
        row = {"start": a, "end": b, "duration": round(b - a, 3)}
        hits.append(row)
        print(
            f"  GAP FILL warn: speech energy in uncovered hole "
            f"{a:.1f}-{b:.1f}s ({b - a:.1f}s) — possible missed ASR",
            file=sys.stderr,
        )
    if hits:
        out = workdir / "speech_gaps.json"
        out.write_text(json.dumps({"gaps": hits}, indent=2), encoding="utf-8")
        print(f"  Wrote {out} ({len(hits)} hole(s))", file=sys.stderr)
    return hits


def assert_tts_coverage(segments: list[dict], *, allow_missing: bool = False) -> list[dict]:
    """Require every HE dub window to have a validated TTS clip before mux.

    Returns the list of failing segments. Raises SystemExit unless allow_missing.
    """
    missing: list[dict] = []
    for i, seg in enumerate(segments):
        if seg.get("keep_original") or (seg.get("language") or "he") != "he":
            continue
        if (seg.get("language") or "") == "skip":
            continue
        if not (seg.get("text") or "").strip():
            continue
        failed = bool(seg.get("tts_failed"))
        fitted = seg.get("tts_fit")
        ok_file = bool(fitted) and Path(fitted).is_file()
        if failed or not ok_file:
            missing.append(
                {
                    "index": i,
                    "speaker_id": seg.get("speaker_id"),
                    "start": float(seg.get("source_start") or seg.get("start") or 0.0),
                    "end": float(seg.get("source_end") or seg.get("end") or 0.0),
                    "tts_failed": failed,
                    "tts_fit": fitted,
                    "text": (seg.get("text") or "")[:80],
                }
            )
    if missing:
        print(
            f"TTS coverage FAIL: {len(missing)} HE window(s) without valid EN clip:",
            file=sys.stderr,
        )
        for m in missing[:12]:
            print(
                f"  [{m['speaker_id']}] {m['start']:.1f}-{m['end']:.1f}s "
                f"failed={m['tts_failed']} | {m['text']}",
                file=sys.stderr,
            )
        if not allow_missing:
            raise SystemExit(
                f"Refusing to mux: {len(missing)} Hebrew segment(s) lack validated TTS. "
                "Re-run TTS or pass --allow-missing-tts for a degraded preview."
            )
    return missing


def run_dub_qa(segments: list[dict], dubbed: Path, workdir: Path) -> Path:
    """Re-ASR dubbed speech windows and compare to text_en → qa_report.json."""
    import numpy as np
    import soundfile as sf

    report: dict = {
        "segments": [],
        "coverage": [],
        "early_ends": [],
        "truncations": [],
        "drift": [],
        "music_jumps": [],
    }
    # Coverage gate summary — honest spoken length vs clip length + drift.
    for i, seg in enumerate(segments):
        if seg.get("keep_original") or (seg.get("language") or "he") != "he":
            continue
        src_a = float(seg.get("source_start") or seg.get("start") or 0.0)
        src_b = float(seg.get("source_end") or seg.get("end") or 0.0)
        place_a = float(seg.get("place_start") or src_a)
        place_b = float(seg.get("place_end") or seg.get("spoken_end") or src_b)
        spoken = place_b
        early = max(0.0, src_b - spoken)
        drift = place_a - src_a
        fitted = seg.get("tts_fit")
        clip_sec = None
        trunc_sec = 0.0
        if fitted and Path(fitted).is_file():
            try:
                clip_sec = float(f5_wav_duration(Path(fitted)))
                # Truncation = clip longer than what the mix window can hold
                # relative to recorded place span (should be ~0 under never-cut).
                placed_span = max(0.0, place_b - place_a)
                trunc_sec = max(0.0, clip_sec - placed_span - 0.05)
            except Exception:
                clip_sec = float(seg.get("tts_clip_sec") or 0.0) or None
        row_cov = {
            "index": i,
            "speaker_id": seg.get("speaker_id"),
            "source_start": src_a,
            "source_end": src_b,
            "place_start": round(place_a, 3),
            "place_end": round(place_b, 3),
            "place_speed": seg.get("place_speed"),
            "place_drift": round(drift, 3),
            "spoken_end": spoken,
            "early_end_sec": round(early, 3),
            "clip_sec": round(clip_sec, 3) if clip_sec is not None else None,
            "truncation_sec": round(trunc_sec, 3),
            "tts_failed": bool(seg.get("tts_failed")),
            "has_tts": bool(fitted) and Path(fitted).is_file() if fitted else False,
            "tts_speed_used": seg.get("tts_speed_used"),
        }
        report["coverage"].append(row_cov)
        if early >= 0.35:
            report["early_ends"].append(row_cov)
        if trunc_sec >= 0.08:
            report["truncations"].append(row_cov)
        if abs(drift) >= 0.05:
            report["drift"].append(row_cov)

    try:
        from faster_whisper import WhisperModel
        from inference.extract_pipeline import resolve_whisper_model
        from inference.tts_f5 import extract_wav_slice
    except Exception as exc:
        report["error"] = str(exc)
        out = workdir / "qa_report.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    # Music envelope discontinuity probe on the leak-cleaned bed (not the
    # speech+bed mix — speech onsets would otherwise flood false positives).
    try:
        bed_path = workdir / "background_leakclean.wav"
        if not bed_path.is_file():
            bed_path = workdir / "background.wav"
        audio, sr = sf.read(str(bed_path), dtype="float32", always_2d=True)
        mono = audio.mean(axis=-1)
        hop = max(1, int(0.05 * sr))
        prev_rms = None
        for i0 in range(0, len(mono) - hop, hop):
            rms = float(np.sqrt(np.mean(mono[i0 : i0 + hop] ** 2) + 1e-12))
            # Near-mute collapse from an audible bed frame.
            if prev_rms is not None and prev_rms > 0.015 and rms < 0.002 and rms < prev_rms * 0.12:
                t = i0 / sr
                report["music_jumps"].append(
                    {"t": round(t, 2), "from_rms": round(prev_rms, 4), "to_rms": round(rms, 4)}
                )
            prev_rms = rms
        report["music_jumps"] = report["music_jumps"][:40]
    except Exception as exc:
        report["music_probe_error"] = str(exc)

    model_path = resolve_whisper_model("ivrit-ai/whisper-large-v3-turbo-ct2")
    model = WhisperModel(str(model_path), device="cpu", compute_type="auto")
    tmp = workdir / "tts_clips" / "_qa_probe.wav"
    tmp.parent.mkdir(parents=True, exist_ok=True)

    for i, seg in enumerate(segments):
        if seg.get("keep_original") or (seg.get("language") or "he") != "he":
            continue
        en = (seg.get("text_en") or "").strip()
        fitted = seg.get("tts_fit")
        if not en or not fitted or not Path(fitted).is_file():
            continue
        try:
            extract_wav_slice(
                Path(fitted),
                0.0,
                f5_wav_duration(Path(fitted)),
                tmp,
                sample_rate=16000,
            )
            segs, _ = model.transcribe(
                str(tmp), language="en", task="transcribe", word_timestamps=False
            )
            asr = " ".join((s.text or "").strip() for s in segs).strip()
        except Exception as exc:
            asr = f"[asr_error: {exc}]"
        from inference.tts_qwen import _word_overlap_ratio

        overlap = _word_overlap_ratio(en, asr)
        repeated = False
        words = en.lower().split()
        if len(words) >= 8:
            half = len(words) // 2
            repeated = words[:half] == words[half : half * 2]
        row = {
            "index": i,
            "speaker_id": seg.get("speaker_id"),
            "start": float(seg.get("source_start") or seg["start"]),
            "end": float(seg.get("source_end") or seg["end"]),
            "place_start": float(seg.get("place_start") or seg.get("source_start") or seg["start"]),
            "place_end": float(
                seg.get("place_end")
                or seg.get("spoken_end")
                or seg.get("source_end")
                or seg["end"]
            ),
            "place_drift": round(
                float(seg.get("place_start") or seg.get("source_start") or seg["start"])
                - float(seg.get("source_start") or seg["start"]),
                3,
            ),
            "place_speed": seg.get("place_speed"),
            "text_en": en[:200],
            "asr": asr[:200],
            "word_overlap": round(overlap, 3),
            "repeated": repeated,
            "tts_speed_used": seg.get("tts_speed_used"),
            "early_end_sec": round(
                max(
                    0.0,
                    float(seg.get("source_end") or seg["end"])
                    - float(
                        seg.get("place_end")
                        or seg.get("spoken_end")
                        or seg.get("tts_end")
                        or seg["end"]
                    ),
                ),
                3,
            ),
        }
        report["segments"].append(row)
        if overlap < 0.45 or row["repeated"]:
            print(
                f"  QA WARN seg {i} overlap={overlap:.2f} repeated={row['repeated']}: "
                f"{en[:50]}…",
                file=sys.stderr,
            )
    out = workdir / "qa_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a watchable dubbed preview from extract outputs.")
    p.add_argument(
        "workdir",
        type=Path,
        nargs="?",
        default=REPO_ROOT / "outputs" / "kan11_60s",
        help="Run directory with segments.json + stems (default: outputs/kan11_60s).",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MLX_TRANSLATE_MODEL,
        help=f"mlx-lm HF model id for HE→EN when --translator llm (default: {DEFAULT_MLX_TRANSLATE_MODEL}).",
    )
    p.add_argument(
        "--translator",
        choices=("llm", "gemma"),
        default="gemma",
        help="Translation backend: gemma (TranslateGemma-4B + neighbor context, default) "
        "or llm (mlx-lm instruct with synopsis/memory).",
    )
    p.add_argument(
        "--gemma-model",
        type=Path,
        default=DEFAULT_GEMMA_MODEL,
        help=f"Local TranslateGemma dir (default: {DEFAULT_GEMMA_MODEL}).",
    )
    p.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    p.add_argument("--skip-translate", action="store_true", help="Reuse existing text_en in JSON.")
    p.add_argument(
        "--seed-translations",
        type=Path,
        default=None,
        help="Reuse text_en from a reference run (e.g. outputs/kan11_60s/translated_segments.json) "
        "when Hebrew/time overlaps.",
    )
    p.add_argument("--skip-tts", action="store_true", help="Keep Hebrew vocals; only add EN subs.")
    p.add_argument(
        "--tts-engine",
        choices=("qwen", "f5"),
        default="qwen",
        help="TTS backend (default: qwen 1.7B-Base zero-shot clone).",
    )
    p.add_argument(
        "--qwen-model",
        type=Path,
        default=QWEN_DEFAULT_MODEL,
        help="Qwen3-TTS Base checkpoint (default: models/Qwen3-TTS-12Hz-1.7B-Base).",
    )
    p.add_argument(
        "--qwen-icl",
        action="store_true",
        help="ICL clone with Hebrew ref_text (default: x-vector-only from ref audio).",
    )
    p.add_argument(
        "--qwen-reuse-speaker-prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse one clone prompt per speaker (default: off — fresh clone per segment).",
    )
    p.add_argument(
        "--tts-speed",
        type=float,
        default=1.0,
        help="F5-TTS only: speed 0.3–2.0 (default 1.0).",
    )
    p.add_argument(
        "--tts-fit-duration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="F5 only: nudge speed toward slot length (default: on).",
    )
    p.add_argument("--fit-speed-min", type=float, default=0.85)
    p.add_argument("--fit-speed-max", type=float, default=1.25)
    p.add_argument(
        "--max-pause",
        type=float,
        default=1.0,
        help="Merge same-speaker gaps ≤ this into one utterance; TTS inserts silence for the pause.",
    )
    p.add_argument(
        "--max-dub-pause",
        type=float,
        default=DEFAULT_MAX_DUB_PAUSE,
        help="Legacy CLI; TTS units split at pauses ≥1.2s. Kept for compat.",
    )
    p.add_argument(
        "--no-clone-verify",
        action="store_true",
        help="Skip TTS clone length/ASR verification + regenerate (faster).",
    )
    p.add_argument(
        "--speaker-bank",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build/use per-speaker canonical ref bank (default: on — stable per-person voice).",
    )
    p.add_argument(
        "--duck-gain",
        type=float,
        default=0.60,
        help="Music bed gain under HE dubs (0–1). Default 0.60 keeps music clearly "
        "audible; residual HE speech still gated out via vocals energy.",
    )
    p.add_argument(
        "--qa",
        action="store_true",
        help="After mux, re-ASR dubbed speech and write qa_report.json.",
    )
    p.add_argument(
        "--allow-missing-tts",
        action="store_true",
        help="Allow mux even when some HE windows lack validated TTS (degraded).",
    )
    p.add_argument(
        "--youtube-url",
        default=None,
        help="Optional YouTube URL for caption cross-check (off by default; ASR is authority).",
    )
    p.add_argument(
        "--tts-segments",
        default=None,
        help="Only regenerate comma-separated zero-based segment indices.",
    )
    p.add_argument(
        "--reuse-tts",
        action="store_true",
        help="Reuse existing tts_fit files and only remix/mux.",
    )
    p.add_argument("--f5-model", default="F5TTS_v1_Base", help="F5-TTS model name.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    workdir = args.workdir
    seg_path = workdir / "segments.json"
    if not seg_path.is_file():
        raise SystemExit(f"Missing {seg_path}")

    asr_payload = json.loads(seg_path.read_text(encoding="utf-8"))
    payload = asr_payload
    translated_path = workdir / "translated_segments.json"
    if args.skip_translate and translated_path.is_file():
        payload = json.loads(translated_path.read_text(encoding="utf-8"))
        print(f"Loaded translations from {translated_path}", file=sys.stderr)

    asr_segments = list(asr_payload.get("segments", []))
    segments = []
    for s in payload.get("segments", []):
        dur = float(s.get("duration", s["end"] - s["start"]))
        if dur < MIN_SEG_DURATION:
            continue
        keep = bool(s.get("keep_original") or (s.get("language") or "he") != "he")
        text = (s.get("text") or "").strip()
        # KEEP/original rows must survive empty captions — blank text used to
        # drop whole English turns and leave silent holes in the mix.
        if not text and not keep:
            continue
        if keep and not text:
            s = dict(s)
            s["text"] = (s.get("text_en") or "").strip() or "[original audio]"
            s["text_en"] = (s.get("text_en") or "").strip() or s["text"]
            s["keep_original"] = True
        segments.append(s)

    # region agent log
    loaded_rows = payload.get("segments", [])
    missing_asr_rows = []
    for idx, src in enumerate(asr_segments):
        a = float(src.get("start") or 0.0)
        b = float(src.get("end") or 0.0)
        covered = any(
            max(
                0.0,
                min(b, float(row.get("source_end") or row.get("end") or 0.0))
                - max(a, float(row.get("source_start") or row.get("start") or 0.0)),
            )
            >= max(0.1, (b - a) * 0.6)
            for row in loaded_rows
        )
        if not covered:
            missing_asr_rows.append(
                {
                    "index": idx,
                    "start": round(a, 3),
                    "end": round(b, 3),
                    "speaker": src.get("speaker_id"),
                    "keep": bool(src.get("keep_original")),
                    "language": src.get("language"),
                    "textEmpty": not bool((src.get("text") or "").strip()),
                }
            )
    _debug_event(
        "H1",
        "inference/build_preview.py:main-load",
        "Compare authoritative ASR rows with loaded preview rows",
        {
            "usingTranslated": bool(args.skip_translate and translated_path.is_file()),
            "asrCount": len(asr_segments),
            "loadedCount": len(loaded_rows),
            "filteredCount": len(segments),
            "missingAsrRows": missing_asr_rows,
            "emptyLoadedRows": [
                i
                for i, row in enumerate(loaded_rows)
                if not (row.get("text") or "").strip()
            ],
        },
    )
    # endregion

    if not segments:
        raise SystemExit("No usable segments after filtering micro-turns.")

    # Repair corrupted text from earlier non-idempotent re-runs.
    for seg in segments:
        dedupe_segment_text_fields(seg)
        clamp_segment_phrases(seg)

    n_overlap = clamp_segment_timeline(segments)
    if n_overlap:
        print(
            f"Clamped {n_overlap} overlapping segment boundary(ies).",
            file=sys.stderr,
        )

    # Re-insert ASR turns missing from translated JSON (empty-KEEP drop bug).
    n_rehyd = rehydrate_missing_asr_segments(segments, asr_segments)
    if n_rehyd:
        print(f"Rehydrated {n_rehyd} missing ASR turn(s).", file=sys.stderr)

    # Restore immutable HE phrase windows from segments.json (TTS writeback
    # must not permanently corrupt ASR timing).
    n_restored = restore_asr_phrase_boundaries(segments, asr_segments)
    if n_restored:
        print(
            f"Restored ASR phrase boundaries on {n_restored} segment(s).",
            file=sys.stderr,
        )

    n_src = stamp_source_timing(segments)
    if n_src:
        print(
            f"Stamped immutable ASR source timing on {n_src} segment(s).",
            file=sys.stderr,
        )

    vocals_path = workdir / "vocals.wav"
    if vocals_path.is_file():
        from inference.segment_merge import drop_silent_vocal_segments

        before = len(segments)
        source_mix = workdir / "source.wav"
        segments = drop_silent_vocal_segments(
            segments,
            vocals_path,
            source_path=source_mix if source_mix.is_file() else None,
        )
        if len(segments) != before:
            print(
                f"Dropped {before - len(segments)} silent-vocals segment(s) "
                f"(music-bed hallucinations).",
                file=sys.stderr,
            )

    n_retag = retag_phonetic_english(segments)
    if n_retag:
        print(f"Retagged {n_retag} phonetic-English turns → keep_original.", file=sys.stderr)

    n_sandwich = retag_english_sandwich(segments)
    if n_sandwich:
        print(
            f"Retagged {n_sandwich} EN-sandwich fragment(s) → keep_original.",
            file=sys.stderr,
        )

    n_spk = enforce_speaker_language_consistency(segments)
    if n_spk:
        print(
            f"Speaker-consistency KEEP on {n_spk} turn(s).",
            file=sys.stderr,
        )

    youtube_url = args.youtube_url  # ASR is authority; captions only if explicitly requested
    if youtube_url and not args.skip_translate:
        from inference.youtube_subs import crosscheck_workdir

        print(f"YouTube caption cross-check: {youtube_url}", file=sys.stderr)
        stats = crosscheck_workdir(workdir, str(youtube_url), segments)
        payload["youtube_url"] = youtube_url
        print(
            f"  youtube_subs: annotated={stats.get('annotated', 0)} "
            f"replaced={stats.get('replaced', 0)}",
            file=sys.stderr,
        )

    # Fresh translate path: drop stale phrase-level EN so utterance MT runs.
    if not args.skip_translate:
        for seg in segments:
            if seg.get("keep_original") or (seg.get("language") or "he") != "he":
                continue
            seg.pop("text_en", None)
            for p in seg.get("phrases") or []:
                p.pop("text_en", None)

    seed_path = args.seed_translations
    if seed_path is None:
        candidate = REPO_ROOT / "outputs" / "kan11_60s" / "translated_segments.json"
        if workdir.resolve() != candidate.parent.resolve() and candidate.is_file():
            seed_path = candidate
    if seed_path is not None and not args.skip_translate:
        n_seed = seed_translations_from_run(segments, Path(seed_path))
        if n_seed:
            print(f"Seeded {n_seed} translations from {seed_path}.", file=sys.stderr)

    def _has_en(seg: dict) -> bool:
        if seg.get("keep_original") or seg.get("language", "he") != "he":
            return bool((seg.get("text") or seg.get("text_en") or "").strip())
        # Segment-level EN is enough under --skip-translate; phrase rows are
        # often wiped by ASR boundary restore and then redistributed.
        if is_english_text(seg.get("text_en") or ""):
            return True
        phrases = seg.get("phrases") or []
        if phrases:
            return all(
                is_english_text(p.get("text_en") or "")
                for p in phrases
                if (p.get("text") or "").strip()
            )
        return False

    if args.skip_translate and all(_has_en(s) for s in segments):
        print("Reusing existing English / original-language text.", file=sys.stderr)
        for seg in segments:
            dedupe_segment_text_fields(seg)
            if seg.get("keep_original") or seg.get("language", "he") != "he":
                seg["keep_original"] = True
                if not (seg.get("text_en") or "").strip():
                    seg["text_en"] = (seg.get("text") or "").strip()
                for p in seg.get("phrases") or []:
                    if not (p.get("text_en") or "").strip():
                        p["text_en"] = (p.get("text") or "").strip()
    else:
        needs_mt = [
            s
            for s in segments
            if not (s.get("keep_original") or s.get("language", "he") != "he")
            and not _has_en(s)
        ]
        model = tokenizer = None
        gemma_device = None
        if needs_mt:
            if args.translator == "gemma":
                processor, gemma_model, gemma_device = load_gemma_translator(
                    args.gemma_model, device=args.device
                )
                model, tokenizer = processor, gemma_model
            else:
                model, tokenizer = load_mlx_translator(str(args.model))
        ensure_translations(
            segments,
            model,
            tokenizer,
            translator=args.translator,
            gemma_device=gemma_device,
        )
        if model is not None:
            del model, tokenizer

    # Always sanitize EN before TTS (chrome / non-Latin / echo leftovers).
    for seg in segments:
        if seg.get("keep_original") or (seg.get("language") or "he") != "he":
            continue
        if (seg.get("text_en") or "").strip():
            en = dedupe_repeated_sentences(
                strip_caption_chrome(strip_non_latin_runs(seg["text_en"]))
            )
            he = (seg.get("text") or "").strip()
            dur = float(seg.get("duration") or (float(seg["end"]) - float(seg["start"])))
            # Skip-translate leftovers can still be overlong / non-Latin — clamp.
            # When --skip-translate, do NOT auto-shorten: that destroys hand-fixed EN.
            reason = mt_needs_retry(he, en, duration=dur)
            if reason == "non_english_script":
                en = strip_non_latin_runs(en)
            elif (
                not args.skip_translate
                and reason in ("time_overflow", "en_too_long_for_he")
            ):
                en = strip_non_latin_runs(en)
                en = shorten_en_preserving_meaning(
                    he,
                    en,
                    duration=dur,
                    syllable_budget=max(6, int(dur * 2.6)),
                )
                print(
                    f"  Pre-TTS shorten [{seg.get('speaker_id')}] "
                    f"{float(seg['start']):.1f}s ({reason})",
                    file=sys.stderr,
                )
            seg["text_en"] = en
            # Glue false sentence breaks from prior duration packing
            # ("No woman. in the Middle" → "No woman in the Middle").
            en = re.sub(r"\.\s+([a-z])", r" \1", en)
            seg["text_en"] = en
            phrases = seg.get("phrases") or []
            if phrases:
                # Re-apply semantic phrase merges (object continuations) even on
                # --skip-translate so stale JSON gets the continuity fix.
                phrases = merge_short_phrases(phrases)
                seg["phrases"] = phrases
                # Prefer existing phrase EN under --skip-translate (hand fixes).
                phrase_en_ok = all(
                    is_english_text(p.get("text_en") or "")
                    for p in phrases
                    if (p.get("text") or "").strip()
                )
                if is_english_text(en) and not (
                    args.skip_translate and phrase_en_ok
                ):
                    distribute_en_to_phrases(en, phrases)
        for p in seg.get("phrases") or []:
            if (p.get("text_en") or "").strip():
                p["text_en"] = dedupe_repeated_sentences(
                    strip_caption_chrome(strip_non_latin_runs(p["text_en"]))
                )

    # Drop stubs absorbed into continuations (language=skip / empty text).
    before_abs = len(segments)
    segments = [
        s
        for s in segments
        if (s.get("language") or "") != "skip" and (s.get("text") or "").strip()
    ]
    if len(segments) != before_abs:
        print(
            f"Dropped {before_abs - len(segments)} absorbed stub segment(s).",
            file=sys.stderr,
        )

    # Always backfill / repair phrase EN (dangling "that …" merges, etc.).
    for seg in segments:
        if seg.get("keep_original") or (seg.get("language") or "he") != "he":
            continue
        phrases = seg.get("phrases") or []
        if not phrases:
            continue
        en = (seg.get("text_en") or "").strip()
        if not is_english_text(en):
            continue
        if any(not (p.get("text_en") or "").strip() for p in phrases) or any(
            _REL_CLAUSE_START.match((p.get("text_en") or "").strip())
            for p in phrases[1:]
        ):
            distribute_en_to_phrases(en, phrases)
            refresh_segment_fields(seg)

    # Sentence-split / skip-translate can leave HE rows with no EN — repair before TTS.
    missing_seg_en = [
        s
        for s in segments
        if not (s.get("keep_original") or (s.get("language") or "he") != "he")
        and not is_english_text(s.get("text_en") or "")
    ]
    missing_phrase_en = [
        s
        for s in segments
        if not (s.get("keep_original") or (s.get("language") or "he") != "he")
        and is_english_text(s.get("text_en") or "")
        and any(
            not (p.get("text_en") or "").strip()
            for p in (s.get("phrases") or [])
            if (p.get("text") or "").strip()
        )
    ]
    if missing_phrase_en and not missing_seg_en:
        # Redistribute from existing segment EN — no MT model needed.
        n_fix = repair_missing_english(segments, None, None)
        if n_fix:
            print(f"Repaired {n_fix} segment(s) missing English.", file=sys.stderr)
    elif missing_seg_en or missing_phrase_en:
        model = tokenizer = None
        try:
            model, tokenizer = load_mlx_translator(str(args.model))
            n_fix = repair_missing_english(segments, model, tokenizer)
        finally:
            if model is not None:
                del model, tokenizer
        if n_fix:
            print(f"Repaired {n_fix} segment(s) missing English.", file=sys.stderr)

    # region agent log
    _debug_event(
        "H2,H3",
        "inference/build_preview.py:pre-write-segments",
        "Capture phrase timing and translation alignment before TTS",
        {
            "segments": [
                {
                    "index": i,
                    "speaker": seg.get("speaker_id"),
                    "start": round(float(seg.get("start") or 0.0), 3),
                    "end": round(float(seg.get("end") or 0.0), 3),
                    "sourceStart": seg.get("source_start"),
                    "sourceEnd": seg.get("source_end"),
                    "keep": bool(seg.get("keep_original")),
                    "language": seg.get("language"),
                    "phraseCount": len(seg.get("phrases") or []),
                    "phrases": [
                        {
                            "start": round(float(p.get("start") or 0.0), 3),
                            "end": round(float(p.get("end") or 0.0), 3),
                            "sourceStart": p.get("source_start"),
                            "sourceEnd": p.get("source_end"),
                            "pauseAfter": p.get("pause_after"),
                            "heWords": len((p.get("text") or "").split()),
                            "enWords": len((p.get("text_en") or "").split()),
                            "hasTtsFit": bool(p.get("tts_fit")),
                        }
                        for p in (seg.get("phrases") or [])
                    ],
                }
                for i, seg in enumerate(segments)
            ]
        },
    )
    # endregion

    out_json = workdir / "translated_segments.json"
    payload_out = {**payload, "segments": segments}
    out_json.write_text(json.dumps(payload_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}", file=sys.stderr)

    srt_path = workdir / "preview_en.srt"
    write_srt(segments, srt_path)
    print(f"Wrote {srt_path}", file=sys.stderr)

    background = Path(payload.get("background") or workdir / "background.wav")
    vocals = Path(payload.get("vocals") or workdir / "vocals.wav")
    if not background.is_file():
        raise SystemExit(f"Missing background stem: {background}")
    if not vocals.is_file():
        raise SystemExit(f"Missing vocals stem: {vocals}")

    total_duration = wav_duration(
        workdir / "source.wav" if (workdir / "source.wav").is_file() else vocals
    )

    speech_gaps = log_speech_gaps(segments, vocals, workdir)

    if args.skip_tts:
        # Preview remix without TTS: place full vocals as one "clip" at t=0.
        import numpy as np
        import soundfile as sf

        bg, sr = sf.read(str(background), dtype="float32", always_2d=True)
        voc, vsr = sf.read(str(vocals), dtype="float32", always_2d=True)
        n = max(1, int(round(total_duration * sr)))
        if bg.shape[0] < n:
            bg = np.pad(bg, ((0, n - bg.shape[0]), (0, 0)))
        else:
            bg = bg[:n]
        voc_m = voc.mean(axis=1)
        if vsr != sr and len(voc_m):
            n_out = int(round(len(voc_m) * sr / vsr))
            x_old = np.linspace(0.0, 1.0, num=len(voc_m), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
            voc_m = np.interp(x_new, x_old, voc_m).astype(np.float32)
        speech = np.zeros(n, dtype=np.float32)
        speech[: min(n, len(voc_m))] = voc_m[:n]
        mix = bg * 0.65 + speech[:, None]
        peak = float(np.max(np.abs(mix))) if mix.size else 0.0
        if peak > 0.98:
            mix *= 0.98 / peak
        dubbed = workdir / "dubbed_audio.wav"
        sf.write(str(dubbed), mix.astype(np.float32), sr, subtype="PCM_16")
    else:
        tts_device = None if args.device == "auto" else args.device
        selected = (
            {int(value) for value in args.tts_segments.split(",")}
            if args.tts_segments
            else None
        )
        if args.reuse_tts:
            print(f"Reusing existing TTS clips (engine={args.tts_engine}).", file=sys.stderr)
        elif args.tts_engine == "qwen":
            speaker_bank = None
            use_bank = bool(args.speaker_bank)
            if use_bank:
                from inference.speaker_bank import build_speaker_bank

                print("Building per-speaker voice bank…", file=sys.stderr)
                speaker_bank = build_speaker_bank(segments, vocals, workdir)
            else:
                print(
                    "Fresh per-segment clone refs (speaker bank off).",
                    file=sys.stderr,
                )

            shorten_model = shorten_tok = None
            shorten_en_fn = None
            if args.translator == "llm":
                try:
                    shorten_model, shorten_tok = load_mlx_translator(str(args.model))

                    def shorten_en_fn(**kwargs):
                        return shorten_english_mlx(shorten_model, shorten_tok, **kwargs)
                except SystemExit:
                    print(
                        "  shorten-retry unavailable (translator not loaded)",
                        file=sys.stderr,
                    )

            try:
                synthesize_segments_qwen(
                    segments,
                    vocals,
                    workdir,
                    model_path=args.qwen_model,
                    x_vector_only=not args.qwen_icl,
                    reuse_speaker_prompt=args.qwen_reuse_speaker_prompt,
                    device=tts_device,
                    selected_indices=selected,
                    merge_pauses=True,
                    max_pause=args.max_pause,
                    max_dub_pause=args.max_dub_pause,
                    shorten_en_fn=shorten_en_fn,
                    speaker_bank=speaker_bank,
                    verify_clone=not args.no_clone_verify,
                )
            finally:
                if shorten_model is not None:
                    del shorten_model, shorten_tok
        else:
            print(
                f"F5-TTS speed={args.tts_speed} fit_duration={args.tts_fit_duration}",
                file=sys.stderr,
            )
            synthesize_segments_f5(
                segments,
                vocals,
                workdir,
                speed=args.tts_speed,
                fit_duration=args.tts_fit_duration,
                fit_speed_min=args.fit_speed_min,
                fit_speed_max=args.fit_speed_max,
                model=args.f5_model,
                device=tts_device,
                selected_indices=selected,
                merge_pauses=True,
                max_pause=args.max_pause,
            )
        payload_out = {
            **payload,
            "segments": segments,
            "tts_engine": (
                "qwen3-tts-1.7b-base" if args.tts_engine == "qwen" else "f5-tts"
            ),
            "tts_speed": args.tts_speed,
            "qwen_x_vector_only": not bool(args.qwen_icl),
            "max_dub_pause": args.max_dub_pause,
            "translator": args.translator,
            "duck_gain": args.duck_gain,
        }
        out_json.write_text(json.dumps(payload_out, ensure_ascii=False, indent=2), encoding="utf-8")
        assert_tts_coverage(segments, allow_missing=bool(args.allow_missing_tts))
        # Elastic gap-closing packer: close small gaps, never cut, speed before KEEP.
        n_planned = plan_dub_placement(segments, total_duration)
        n_sped = apply_place_speeds(segments)
        print(
            f"Elastic placement: {n_planned} dub(s) planned, "
            f"{n_sped} clip(s) speed-adjusted",
            file=sys.stderr,
        )
        payload_out["segments"] = segments
        out_json.write_text(json.dumps(payload_out, ensure_ascii=False, indent=2), encoding="utf-8")
        dubbed = build_dubbed_track(
            segments,
            background,
            total_duration,
            workdir,
            vocals=vocals,
            speech_gaps=speech_gaps,
            duck_gain=args.duck_gain,
        )

    video = find_source_video(workdir, payload)
    preview = workdir / "preview.mp4"
    mux_preview(video, dubbed, srt_path, preview, total_duration)

    if args.qa and not args.skip_tts:
        qa_path = run_dub_qa(segments, dubbed, workdir)
        print(f"QA report: {qa_path}", file=sys.stderr)

    print(f"\nWatchable preview ready:\n  {preview}", file=sys.stderr)
    print(str(preview))


if __name__ == "__main__":
    main()
