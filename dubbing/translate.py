"""Stage 5 — general-purpose Gemma as a comprehension translator, one segment at a time.

A dedicated MT model (TranslateGemma) translates literally and has no world
knowledge, so it drops or garbles the references that matter — "האחים המוסלמים"
(the Muslim Brotherhood, an organisation) came back as "brotherhood among
Muslims", and "שייח'ה" (the Sheikha) as "a CEO". A general instruction model
*understands* the text: it keeps the named organisations and people, and — asked
to — every clause and detail, rather than paraphrasing them away. That is what a
dub needs: the meaning, complete, in natural English.

The model is general Gemma 4 (12B) run through MLX (Apple-native), quantised so it
fits in unified memory. Each segment is translated on its own, deterministically
(greedy); a segment never sees its neighbours, which keeps the pieces from bleeding
into each other.

Which 12B quantisation matters more than it looks. Scored on the 30 dubbed segments
of the Qatar piece against nine things the Hebrew must survive — "הגז הקטארי" as
"Qatari gas", "שתי ההפיכות" as "two coups", "לבנה השני" as "her second son",
"שלושה עשורים" as "three decades", the named organisations intact:

  6bit       9.7 GB  9/9, and the fastest of the ones that pass. The default.
  qat-4bit  10.9 GB  9/9 too, but 2.4x slower (its MLP layers are 8-bit) and bigger.
  8bit      12.7 GB  8/9 — loses "her second son". No gain for the extra 3 GB.
  4bit       6.3 GB  5/9. Same footprint as the Gemma 3 it replaces, and reads
                     fluently, which is what makes it dangerous: "the Qatari fund",
                     "two turnovers", "his son", "the following two decades".
  mxfp4      6.3 GB  6/9, about 3x faster, but "the Qatari loot" for the gas.

For reference the Gemma 3 12B 4-bit it replaces scored 7/9. Below 6 bits this model
does not hold Hebrew, so trading precision for memory is a real quality decision, not
a free one — but if 9.7 GB is too much, MODEL_PATH is the only line to change.

Those scores are for a segment translated alone. Showing the model the previous line
(see `_PRECEDING`) trades one of them — "שתי ההפיכות" comes back as "two reversals"
rather than "two coups", though "who overthrew his father" still carries it — for two
worse failures it fixes: "יגנו עליה" flipping from "defend her" to "condemn her", and
a segment losing its subject entirely ("Qatar in colors that…" for "She paints Qatar
in colors that…"). A wrong word beats a reversed meaning, so it stays.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from . import numwords
from .script import count_letters, is_script, script_for

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "gemma-4-12B-it-6bit"
HUB_ID = "mlx-community/gemma-4-12B-it-6bit"
# Off-Mac backend: the same Gemma 4 12B, bf16 transformers weights, run on CUDA in an
# isolated venv (see translator/) because the main venv's transformers 4.57.3 pin
# predates the gemma4_unified architecture.
CUDA_MODEL_PATH = REPO_ROOT / "models" / "gemma-4-12b-it-cuda"

# Words whose loss flips a sentence's meaning, per target language. `shorten`
# refuses any rewrite that drops one. Small on purpose: high-precision markers,
# not a linguistics project — a miss only means a shortening is (safely) refused.
_NEGATIONS_BY_LANG: dict[str, frozenset[str]] = {
    "en": frozenset({"not", "no", "never", "cannot", "none", "without", "nor", "n't"}),
    "ru": frozenset({"не", "нет", "никогда", "нельзя"}),
    "es": frozenset({"no", "nunca", "sin", "nada"}),
    "fr": frozenset({"ne", "pas", "jamais", "sans"}),
    "de": frozenset({"nicht", "kein", "nie", "ohne"}),
    "it": frozenset({"non", "mai", "senza"}),
    "pt": frozenset({"não", "nunca", "sem"}),
    "zh": frozenset({"不", "没"}),
    "ja": frozenset({"ない", "ません"}),
    "ko": frozenset({"안", "않", "없"}),
    "ar": frozenset({"لا", "لم", "لن", "ليس"}),
    "he": frozenset({"לא", "אין", "בלי"}),
}
_NEGATIONS = _NEGATIONS_BY_LANG["en"]

_LANG_NAMES = {"he": "Hebrew", "en": "English", "ar": "Arabic", "fr": "French",
               "es": "Spanish", "de": "German", "ru": "Russian", "it": "Italian",
               "pt": "Portuguese", "tr": "Turkish", "fa": "Persian",
               "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "uk": "Ukrainian"}


def _lang(code: str) -> str:
    return _LANG_NAMES.get(code, code)


_TURN_MARKERS = ("<end_of_turn>", "<eos>", "<start_of_turn>", "<turn|>", "<|turn>")


def _prompt(tokenizer, user_text: str):
    """One user turn, rendered by the model's own chat template.

    Gemma 4 changed the turn syntax — `<|turn>user … <turn|>` where Gemma 3 had
    `<start_of_turn>user … <end_of_turn>` — so the markers can no longer be written
    by hand. The template gets them right for either generation. Two details matter:
    it is tokenised here (the template emits `<bos>` itself, and generating from a
    string would prepend a second one), and `enable_thinking=False` makes the Gemma 4
    generation prompt pre-close the thought channel (`<|channel>thought\\n<channel|>`),
    so the model answers with the translation instead of reasoning its way there.
    """
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template([{"role": "user", "content": user_text}],
                                                 add_generation_prompt=True,
                                                 enable_thinking=False)
        except Exception as exc:                                  # template too new/old
            print(f"  translate: chat template failed ({exc}); using Gemma 3 markers",
                  file=sys.stderr)
    return f"<start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n"


def _run(tokenizer, model, user_text: str, max_new_tokens: int) -> str:
    """Greedy-decode one user turn, stripped of turn/end markers.

    The prompt is *built* identically for both backends (the caller passes the full
    user text); only the render-template-and-decode step differs. Under MLX that
    happens here; under the subprocess backend the worker applies the chat template
    itself from the raw user text. Post-processing is shared — the worker decodes
    with special tokens kept so the channel/turn-marker split below sees them.
    """
    if isinstance(model, WorkerHandle):
        out = model.request(user_text, max_new_tokens)
    else:
        import mlx_lm
        from mlx_lm.sample_utils import make_sampler

        out = mlx_lm.generate(model, tokenizer, _prompt(tokenizer, user_text),
                              max_tokens=max_new_tokens, sampler=make_sampler(temp=0.0),
                              verbose=False) or ""
    if "<channel|>" in out:              # a thought channel opened anyway — keep the answer
        out = out.split("<channel|>")[-1]
    for marker in _TURN_MARKERS:
        out = out.split(marker)[0]
    return _strip_editorial(out)


class WorkerHandle:
    """A translation model living in a subprocess, spoken to over JSON lines.

    Wraps `translator/worker.py` (its own uv venv, transformers >= 5, CUDA): one
    request line in, one response line out, ids matched so a desynced pipe is an
    error rather than a wrong answer. The worker's stderr is streamed through to
    ours (model loading takes minutes the first time — venv build plus 24 GB of
    weights) and its tail is kept for the crash message.
    """

    def __init__(self, cmd: list[str], ready_timeout: float = 3600.0,
                 env: dict[str, str] | None = None, own_gpu: bool = False):
        self.own_gpu = own_gpu
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                      env=env)
        self._stderr_tail: deque[str] = deque(maxlen=40)
        self._next_id = 0
        threading.Thread(target=self._pump_stderr, daemon=True).start()
        line = self._read_line(ready_timeout)
        try:
            ready = json.loads(line).get("ready")
        except ValueError:
            ready = None
        if ready is not True:
            self.close()
            raise RuntimeError(f"translator worker sent {line!r} instead of the ready line"
                               f"{self._stderr_note()}")

    def _pump_stderr(self) -> None:
        for line in self._proc.stderr:
            sys.stderr.write(line)
            sys.stderr.flush()
            self._stderr_tail.append(line)

    def _stderr_note(self) -> str:
        tail = "".join(self._stderr_tail).strip()
        return f"; recent worker stderr:\n{tail}" if tail else ""

    def _read_line(self, timeout: float) -> str:
        import select

        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise RuntimeError(f"translator worker: no response within {timeout:.0f}s"
                                   f"{self._stderr_note()}")
            readable, _, _ = select.select([self._proc.stdout], [], [], min(left, 1.0))
            if readable:
                line = self._proc.stdout.readline()
                if line:
                    return line
                self._proc.poll()
                raise RuntimeError(f"translator worker died (exit {self._proc.returncode})"
                                   f"{self._stderr_note()}")
            if self._proc.poll() is not None:
                raise RuntimeError(f"translator worker died (exit {self._proc.returncode})"
                                   f"{self._stderr_note()}")

    def request(self, user_text: str, max_new_tokens: int,
                timeout: float = 600.0) -> str:
        self._next_id += 1
        rid = self._next_id
        msg = json.dumps({"id": rid, "user_text": user_text,
                          "max_new_tokens": max_new_tokens}, ensure_ascii=False)
        try:
            self._proc.stdin.write(msg + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(f"translator worker pipe closed: {exc}"
                               f"{self._stderr_note()}") from exc
        reply = json.loads(self._read_line(timeout))
        if reply.get("error"):
            raise RuntimeError(f"translator worker error: {reply['error']}")
        if reply.get("id") != rid:
            raise RuntimeError(f"translator worker desynced: sent id {rid}, "
                               f"got {reply.get('id')!r}")
        return reply.get("text") or ""

    def close(self) -> None:
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except OSError:
            pass
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()


# A trailing block on its own line that opens with a mark or the word "note": a
# subtitle line never has one, and the model's asides always arrive in that shape,
# whatever wording (or target language) it dresses them in.
_NOTE = re.compile(r"\n\s*(?:[*_(\[]|note\b|n\.b\.).*\Z", re.S | re.I)
_BRACKET_ALT = re.compile(r"\[([^\[\]/]{1,40}?)(?:\s*/\s*[^\[\]]{1,40})?\]")
# A channel label alone on the first line ("thought", "analysis"): what remains when
# a thinking channel opened and its markers were stripped elsewhere (or the model
# wrote the label as plain text). Never a subtitle's first line.
_CHANNEL_LABEL = re.compile(r"\A(?:thought|thinking|analysis)\s*\n", re.I)
# Leading marks a spoken line never starts with — "/In the year 2013", ".Be careful".
# Openers that can be legitimate (quotes, parentheses, digits, ellipsis-free letters)
# are not in the set.
_LEAD_JUNK = re.compile(r"\A[\s/\\.,;:!?*\-–—|>#]+")


def _strip_editorial(text: str) -> str:
    """Drop a translator's brackets, notes and channel leaks; TTS would read them.

    Gemma 4 annotates: it brackets words it supplied ("Qatar funds ISIS, [and] Qatar
    funds …"), offers alternatives ("a [connection/link] to"), and appends a note when
    it judges the source garbled ("*(Note: the Hebrew appears to be corrupted…)*"). It
    also occasionally re-opens its thinking channel, which leaves a bare "thought"
    line, and emits stray leading punctuation ("/In the year 2013…"). The prompt asks
    for none of it and mostly gets none; this is the net underneath. On clean output
    every pattern misses and the text is returned as-is.
    """
    text = _CHANNEL_LABEL.sub("", (text or "").strip())
    text = _NOTE.sub("", text).strip()
    text = _BRACKET_ALT.sub(r"\1", text)
    text = _LEAD_JUNK.sub("", text)
    return re.sub(r"\s{2,}", " ", text).strip()


# Shown before the sentence to translate when the previous line is known. Worded to
# break a tie and nothing more: given free rein the model also starts borrowing the
# previous line's register, which cost "two coups" ("שתי ההפיכות") its meaning and
# turned it into "two shifts" after a neighbour about steering the country.
_PRECEDING = (
    "The line spoken just before was: {prev}\n"
    "It is background only. Use it solely to resolve a word that is ambiguous on its "
    "own; do not translate it, do not let its words into your answer, and do not let "
    "it change any wording the sentence below already settles.\n\n"
)


# Rolling run-level name consistency: the last N distinct proper nouns seen in this
# run's own translations, offered back to the model so a recurring person or place
# keeps one spelling. Derived entirely from the run's output — not per-video config.
MAX_ESTABLISHED_NAMES = 12

# Standard Hebrew military vocabulary, appended whenever the *source* is Hebrew.
# General language knowledge (any war footage benefits), not video content.
_HE_MILITARY_NOTE = (
    "The source may use standard Hebrew military terms and abbreviations; read them "
    "as their standard meanings: מ\"פ company commander, מ\"כ squad commander, "
    "סמ\"פ deputy company commander, בה\"ד training base, ש\"ג gate guard, "
    "כיתה squad, מחלקה platoon, פלוגה company, גדוד battalion, חטיבה brigade, "
    "אוגדה division, "
    "מוצב outpost, ציר route or road, עבור over (radio), קלטתי copy (radio), "
    "קמ\"ן intelligence officer, חגורת נפץ explosive belt (suicide belt), and in "
    "military context מודיעין means intelligence (information on the enemy), not "
    "the city Modi'in and never a person's name."
)


def update_established_names(established: list[str], translation: str,
                             target: str = "en") -> list[str]:
    """The rolling established-names list after seeing one produced translation.

    Proper nouns (via `_proper_nouns`) from `translation` move to the front in
    their original casing; the list stays distinct (case-insensitive) and is
    capped at MAX_ESTABLISHED_NAMES, most recent first. Pure — returns a new list.
    """
    nouns = _proper_nouns(translation, target)
    if not nouns:
        return list(established)
    fresh: list[str] = []
    for match in re.finditer(r"\w[\w'-]{2,}", translation or ""):
        tok = match.group(0)
        if tok.lower() in nouns and tok.lower() not in (f.lower() for f in fresh):
            fresh.append(tok)
    out = list(established)
    for name in reversed(fresh):
        out = [n for n in out if n.lower() != name.lower()]
        out.insert(0, name)
    return out[:MAX_ESTABLISHED_NAMES]


# Letter runs, any script — the tokens gloss gating compares. Digits and
# punctuation never form a token, so "2013" or a stray "«" cannot match.
_CTX_TOKEN = re.compile(r"[^\W\d_]+")


def _within_one_edit(a: str, b: str) -> bool:
    """True when `a` equals `b` or differs by one substitution/insert/delete."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = j = 0
    edited = False
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        if edited:
            return False
        edited = True
        if len(a) == len(b):
            i += 1                                     # substitution
        j += 1                                         # else: insertion into b
    return True


def relevant_context(context: str, source_text: str, src_lang: str) -> str:
    """The slice of the user's --context note that belongs in *this* segment's prompt.

    The context string is one global note, but a word gloss inside it ("זיקית means
    chameleon") is only true where that word is actually spoken: injected everywhere,
    it contaminates unrelated segments ("a bourgeois neighborhood" once came back as
    "a neighborhood of chaos" because a בלאגן gloss sat in its prompt). So the note is
    split into clauses (';' / '.' boundaries) and each clause is classified:

    - A clause with no token in the source language's script is background ("Israeli
      documentary about …") and is always included — entity context helps everywhere.
    - A clause carrying source-script tokens is a gloss, and is included only when one
      of those tokens appears in the segment's own source text — exact substring, or
      edit distance <= 1 on tokens of 4+ characters, which covers the ASR variants a
      gloss typically lists (בלאגן / מלאגן) without a per-video matching rule.

    When the source language is Latin-script the gloss/background split has no signal
    (every token is "source script") and the note passes through whole, as before.
    Pure text logic — safe to call per segment, no models involved.
    """
    ctx = (context or "").strip()
    if not ctx:
        return ""
    script = script_for(src_lang)
    if script == "latin":
        return ctx
    seg_tokens = [t.lower() for t in _CTX_TOKEN.findall(source_text or "")]
    seg_text_low = (source_text or "").lower()
    kept: list[str] = []
    dropped = False
    for clause in re.split(r"[;.]", ctx):
        clause = clause.strip()
        if not clause:
            continue
        gloss_tokens = [t for t in _CTX_TOKEN.findall(clause) if is_script(t, src_lang)]
        if not gloss_tokens:
            kept.append(clause)                        # background: always included
            continue
        hit = any(
            tok.lower() in seg_text_low
            or (len(tok) >= 4 and any(_within_one_edit(tok.lower(), st)
                                      for st in seg_tokens))
            for tok in gloss_tokens
        )
        if hit:
            kept.append(clause)
        else:
            dropped = True
    if not dropped:
        return ctx                                     # nothing gated: verbatim
    return "; ".join(kept)


def _translate_instruction(text: str, source: str, target: str, context: str = "",
                           extra: str = "", preceding: str = "",
                           register: str = "narration",
                           names: tuple[str, ...] | list[str] = (),
                           numbers_spelled: bool = False,
                           asr_source: bool = True) -> str:
    """Comprehension prompt: faithful and complete, not literal or summarised.

    `context` is an optional per-video note — who and what the video is about, and
    the established spellings of names the ASR tends to mangle. The model uses it to
    render "שייח' עמוזה" (a mistranscription) as "Sheikha Moza" rather than guessing.
    `extra` is an optional extra sentence appended on a retry (see `_repair_repeat`).

    `preceding` is the line spoken just before this one, shown as context and never
    translated. A segment on its own can be genuinely ambiguous: "יגנו עליה" is
    "they will defend her", but every model tested — Gemma 3 and 4 alike, at every
    quantisation — read it as "they will condemn her" with nothing around it, and
    every one of them got it right once it could see the previous sentence ("…they
    will do everything to protect Qatar"). One line back is enough, and it stays
    strictly context: translating a window instead lets a clause drift from one
    segment onto its neighbour, which is why the segments are separate to begin with.

    `register` selects the speaking style: "narration" (the default, unchanged
    behaviour) or "dialogue" (natural spoken register — for English that means
    contractions are welcome rather than forbidden). `names` is the rolling list of
    proper nouns already established in this run's translations (see
    `update_established_names`); when non-empty the model is told to reuse those
    exact spellings for a recurring person or place.

    `numbers_spelled` marks a hop whose *input* already carries digit-free,
    spelled-out numbers (the pivot's English intermediate after
    `numwords.spell_numbers`, or a spelled English source on a direct en→tgt
    hop). The prompt then asks only that the number-words stay words, inflected —
    the model is never asked to *convert* digits: v22 asked exactly that and the
    model turned "504" into "five zero four", which the next hop read as a year.
    Digit→word conversion is deterministic code in `numwords` (see `run`).

    `asr_source` marks a hop whose input is a raw ASR transcript (the pivot's
    first hop and every direct hop). With a non-empty `context` it adds the
    mishearing note — garbled words, names above all, should resolve to the
    phonetically-closest plausible reading and to the names the context
    establishes ("חוקרי השאר" → "al-Sharaa's researchers", not "the remaining
    researchers"; A/B'd — without context the note fixes nothing, so it is
    never emitted alone). The pivot's second hop reads clean model English and
    passes False.
    """
    src, tgt = _lang(source), _lang(target)
    hint = f"{context.strip()}\n\n" if context and context.strip() else ""
    tail = f" {extra.strip()}" if extra and extra.strip() else ""
    before = _PRECEDING.format(prev=preceding.strip()) if preceding and preceding.strip() else ""
    if register == "dialogue":
        # Dialogue: a dub of people talking should sound like people talking.
        style = (
            "Write natural spoken dialogue: contractions are welcome (\"don't\", "
            "\"we're\"), in a colloquial register matching the speakers."
            if target == "en" else
            f"Write natural conversational {tgt}, the register of spoken dialogue."
        )
    else:
        # The contraction example is English-specific; other targets get the general
        # rule ("full words") without an English aside leaking into their prompt.
        style = (
            "Write full words, no contractions (\"we are\" not "
            "\"we're\", \"do not\" not \"don't\") so the text-to-speech reads them clearly."
            if target == "en" else
            "Write full words suitable for text-to-speech."
        )
    # Numbers: digits are converted to words by code (numwords), never by the
    # model. Only a hop whose input is already spelled gets a sentence, and it
    # asks for inflection of existing words — not conversion.
    numbers = (
        f" Numbers in the source are written out as words; render them as naturally "
        f"inflected words in {tgt}, never as digits."
    ) if numbers_spelled else ""
    names_note = (
        " Names already established in this video's translation — use these exact "
        "spellings when the same person or place recurs: " + ", ".join(names) + "."
    ) if names else ""
    military = f" {_HE_MILITARY_NOTE}" if source == "he" else ""
    # Only with background context to match against: the bare instruction was
    # probed and fixed nothing (Gemma cannot do Hebrew phonetics on demand),
    # while with named entities in the context it reliably reconciles garbled
    # proper nouns to them and leaves clean lines untouched.
    garble = (
        f" The {src} text comes from automatic speech recognition and may mishear "
        f"words — especially names: a garbled word may actually be one of the names "
        f"in the background above. When a word is incoherent in its context, prefer "
        f"the phonetically-closest plausible reading and match garbled names to the "
        f"background names; never turn a garble into a new name or entity."
    ) if asr_source and hint else ""
    # The hop from clean English into a non-English target gets the guarded
    # fluency license (A/B'd twice on 87 he→ru rows: unguarded scored 33 wins but
    # coined fake proper names; guarded kept 13 wins with zero severe regressions
    # and repaired one fidelity error). Never applied when translating the noisy
    # source directly — the license presumes a coherent input.
    fluency = (
        f" Prioritize natural, idiomatic {tgt} phrasing: restructure the syntax "
        f"freely — word order, clause boundaries, the verbs and collocations a "
        f"native speaker would use — while preserving the semantics exactly: never "
        f"substitute a near-synonym that changes the claim, never re-attribute a "
        f"statement to a different speaker, and never add specifics of place, "
        f"origin or time the source does not state. Never coin, restore or "
        f"transliterate a proper name that is not literally in the source; only "
        f"when the source wording is garbled or incoherent, prefer a literal "
        f"rendering over a fluent guess. Deliberate repetition and wordplay are "
        f"kept, not smoothed away."
    ) if source == "en" and target != "en" else ""
    return (
        f"{hint}{before}"
        f"Translate the following {src} text into clear, natural {tgt} for subtitles. "
        f"Understand the meaning and translate it faithfully and completely: keep every "
        f"clause, every detail and any repeated emphasis; preserve all names, "
        f"organizations, numbers and specific references, using their established {tgt} "
        f"names and respecting grammatical gender. A word or phrase the source has "
        f"borrowed from a third language and spelled phonetically in its own alphabet "
        f"is written in the standard spelling it has in {tgt}, never transliterated "
        f"back letter by letter. Do not summarize, shorten, omit, or "
        f"translate word-for-word. {style}{numbers}{fluency}{garble}{names_note}{military}{tail} "
        f"Output only the {tgt} translation, nothing else — no notes, no comments, no "
        f"square brackets and no alternative renderings.\n\n"
        f"{src}: {text}"
    )


# Coordinators/stopwords that may legitimately repeat; a repeat of one of these is
# never the "education, education" defect and must be left alone. English-only:
# for other targets short tokens (< 4 chars) are skipped instead, which covers
# their function words without a per-language stopword project.
_REPEAT_SKIP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "is", "are", "was",
    "were", "be", "by", "for", "with", "that", "this", "it", "as", "so", "no", "not",
    "very", "really", "much", "more", "most", "too", "just", "even", "still",
}

# "and"/"or" per target — the word between the halves of an "X and X" collapse.
_COORDINATORS: dict[str, set[str]] = {
    "en": {"and", "or"}, "ru": {"и", "или"}, "es": {"y", "o"}, "fr": {"et", "ou"},
    "de": {"und", "oder"}, "it": {"e", "o"}, "pt": {"e", "ou"},
}


def _adjacent_repeat(text: str, target: str = "en") -> str | None:
    """The content word in an audible "X, X" / "X and X" collapse, or None.

    Catches a single content word repeated back-to-back (only a comma or a
    coordinator between) — the shape a translator produces when it flattens two
    distinct source words ("השכלה וחינוך") onto one word. Phrase-level emphasis
    ("Qatar funds Hamas, Qatar funds Al-Qaeda") is not matched: there the repeated
    token has other words between its occurrences, so it is left intact. CJK
    targets are never matched — without spaces `\\w+` tokens are whole runs of
    text, not words, so the check would be meaningless there.
    """
    if script_for(target) == "cjk":
        return None
    coords = _COORDINATORS.get(target, _COORDINATORS["en"])
    toks = re.findall(r"\w+", text or "")
    for i in range(len(toks) - 1):
        a = toks[i].lower()
        if len(a) < 4 or (target == "en" and a in _REPEAT_SKIP):
            continue
        if a == toks[i + 1].lower():
            return toks[i]                                   # "X, X" or "X X"
        if (i + 2 < len(toks) and toks[i + 1].lower() in coords
                and a == toks[i + 2].lower()):
            return toks[i]                                   # "X and X"
    return None


def _strip_adjacent_repeat(text: str, target: str = "en") -> str:
    """Last-resort: drop the duplicate half of an "X, X" / "X and X" collapse."""
    coord = "|".join(sorted(_COORDINATORS.get(target, _COORDINATORS["en"])))
    text = re.sub(r"\b(\w{4,})\b\s*,\s*\1\b", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(rf"\b(\w{{4,}})\b\s+(?:{coord})\s+\1\b", r"\1", text,
                  flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", text).strip()


def pivot_via_english(source: str, target: str) -> bool:
    """True when translation must route src→en→tgt instead of going direct.

    A judged A/B over 84 segments settled this: direct Gemma he→ru lost 38–7
    with disqualifying failures (entity substitution, hallucinated speaker
    labels, neighbour-segment bleed), while the pivot's errors were almost all
    inherited from the measured-good English line — so he→en improvements
    propagate to every target. Codes are normalised upstream; no aliasing here.
    """
    return source != "en" and target != "en"


def is_target_text(text: str, target: str = "en") -> bool:
    """Reject empty / untranslated (still source-script) output.

    "Did it actually translate?" — at least two letters of the target's script,
    and more of them than letters of every other script combined. Script-based,
    so it must hold whatever the source language is, and works only when source
    and target scripts differ; same-script pairs sail through, as before.
    """
    t = (text or "").strip()
    if len(t) < 2:
        return False
    tgt = count_letters(t, script_for(target))
    other = sum(1 for ch in t if ch.isalpha()) - tgt
    return tgt >= 2 and tgt > other


def _special_token_map(path: str) -> dict[str, Any]:
    """`extra_special_tokens` as the mapping transformers 4.x wants, from the model.

    Reads the model's own tokenizer_config rather than assuming which tokens it
    declares, so this keeps working for the next model that ships a list — and for one
    that lists several. A token is keyed by a name derived from itself (`<|video|>` →
    `video_token`), which is only ever used as an attribute name on the tokenizer.
    """
    config = Path(path) / "tokenizer_config.json"
    declared: Any = None
    if config.is_file():
        try:
            declared = json.loads(config.read_text(encoding="utf-8")).get("extra_special_tokens")
        except (OSError, ValueError):
            declared = None
    if isinstance(declared, dict):
        return {"extra_special_tokens": declared}          # already the shape 4.x wants
    listed = [t for t in (declared or []) if isinstance(t, str)]
    mapping = {f"{re.sub(r'[^a-z0-9]+', '_', t.lower()).strip('_') or 'extra'}_token": t
               for t in listed}
    return {"extra_special_tokens": mapping}


def _mlx_available() -> bool:
    """The MLX backend applies on a Mac, or anywhere mlx_lm happens to import."""
    if sys.platform == "darwin":
        return True
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        return False
    return True


def _worker_cmd() -> list[str]:
    """Launch line for the CUDA worker, through its own uv project venv."""
    return ["uv", "run", "--project", str(REPO_ROOT / "translator"), "python",
            str(REPO_ROOT / "translator" / "worker.py"), str(CUDA_MODEL_PATH)]


def _spare_gpu() -> str | None:
    """The CUDA device the worker can own outright, or None.

    With two or more GPUs the translator gets the last one to itself (TTS, ASR and
    Demucs all default to device 0), so it can stay resident for the whole run
    instead of reloading 24 GB of weights every time the timeline needs a shorter
    line. An externally set CUDA_VISIBLE_DEVICES is respected as-is.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return None
    try:
        import torch

        count = torch.cuda.device_count()
    except Exception:
        return None
    return str(count - 1) if count >= 2 else None


_WORKER: WorkerHandle | None = None


def exclusive_device() -> bool:
    """True when the translator and the TTS engine compete for the same memory.

    MLX (unified memory) and a single-GPU CUDA box must alternate the two models;
    a multi-GPU box runs the worker on its own device, so callers can keep both
    loaded and skip the close/reload dance.
    """
    return _mlx_available() or _spare_gpu() is None


def load(device: str | None = None):
    """Load Gemma 4 12B on whichever backend this machine has.

    On a Mac (or anywhere mlx_lm imports): the MLX quant, in-process — unchanged.
    Otherwise: the bf16 transformers weights on CUDA, in a subprocess worker with
    its own venv (the main venv's transformers 4.57.3 pin predates Gemma 4; see
    translator/). Either way the return shape is the transformers-era
    (processor, model, device) triple the call sites expect; the worker backend
    returns (None, WorkerHandle, None) and `_run`/`free` dispatch on the handle.

    A worker holding its own GPU is spawned once and reused: `free` leaves it
    running and `load` hands the live handle back.
    """
    global _WORKER
    if not _mlx_available():
        if _WORKER is not None and _WORKER._proc.poll() is None:
            return None, _WORKER, None
        gpu = _spare_gpu()
        env = None
        if gpu is not None:
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
            print(f"  translate: CUDA worker pinned to GPU {gpu}", file=sys.stderr)
        print(f"  translate: starting CUDA worker for {CUDA_MODEL_PATH.name} "
              f"(first run builds translator/ venv + loads bf16 weights)", file=sys.stderr)
        handle = WorkerHandle(_worker_cmd(), env=env, own_gpu=gpu is not None)
        if handle.own_gpu:
            _WORKER = handle
            atexit.register(handle.close)
        return None, handle, None
    return _load_mlx()


def _load_mlx():
    """Load general Gemma 4 (12B, 4-bit) through MLX.

    Returns (tokenizer, model, None) to match the transformers-era
    (processor, model, device) call sites — device is unused under MLX.
    """
    import mlx_lm

    path = str(MODEL_PATH) if MODEL_PATH.is_dir() else HUB_ID
    print(f"  translate: loading {path} (mlx 4-bit)", file=sys.stderr)
    try:
        model, tokenizer = mlx_lm.load(path)
    except AttributeError:
        # A tokenizer_config.json written for transformers 5 can list
        # extra_special_tokens, where 4.x (pinned here by qwen-tts) expects a mapping
        # and calls .keys() on it. Rebuild the mapping from whatever this model
        # actually declares — naming each token after itself — and let it override the
        # file. Harmless to retry: nothing else in load() has taken effect yet.
        model, tokenizer = mlx_lm.load(path, tokenizer_config=_special_token_map(path))
    # Gemma ends a turn with a turn marker, not <eos>. Without it in the stop set
    # generation runs to max_tokens every time — slow, and the text trails a wall
    # of end-of-turn tokens. Register both generations' markers (Gemma 4 already
    # carries <turn|> in its eos ids; the lookup is a no-op when a name is absent).
    # generation's markers; an unknown name resolves to <unk>, so round-trip it.
    for name in ("<end_of_turn>", "<turn|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(name)
            if isinstance(tid, int) and tid >= 0 and tokenizer.convert_ids_to_tokens(tid) == name:
                tokenizer.eos_token_ids.add(tid)
        except Exception:
            pass
    # Loading leaves as much again in MLX's buffer cache (a 9.7 GB model peaks near
    # 20 GB). It is reclaimable, but on a 26 GB machine there is no reason to hold it.
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass
    return tokenizer, model, None


def _echoes_source(out: str, src: str) -> bool:
    """True when the "translation" is mostly the source text handed back.

    The script gate (`is_target_text`) cannot catch an untranslated answer when
    source and target share a script: on an en→es run the model once returned
    'Source:** "The second is that…"' — its English input in a markdown wrapper —
    and Latin-vs-Latin let it straight through to a 71-second TTS clip. Token
    overlap with the source is script-blind: a real translation shares almost no
    full tokens with its source, an echo shares most of them. Short sources are
    exempt (a three-word line legitimately survives translation intact more often,
    and can do little harm).
    """
    src_toks = re.findall(r"\w+", (src or "").lower())
    out_toks = re.findall(r"\w+", (out or "").lower())
    if len(src_toks) < 4 or not out_toks:
        return False
    src_set = set(src_toks)
    return sum(1 for t in out_toks if t in src_set) / len(out_toks) > 0.65


# Markdown skeleton a subtitle line never has: bold markers, or a bulleted line.
_MARKDOWN = re.compile(r"\*\*|^\s*[-*•]\s", re.M)


def _not_a_translation(out: str, src: str) -> bool:
    """A reasoning dump, an echo, or anything else that is not a subtitle line.

    The same en→es segment that exposed `_echoes_source` later came back as the
    model's full working notes in plain markdown — 'Source:** … **Translation
    Strategy:** … **Final Version:** …', 262 tokens for a 26-token source, cut
    off at the token limit. Three shapes betray it, all impossible for a real
    subtitle: heavy token overlap with the source (echo), markdown structure,
    and gross length blow-up (a translation of a 4+-token source stays within a
    few multiples of its length; only sources with enough tokens are judged, so
    CJK sources — few `\\w+` tokens — are exempt rather than misjudged).
    """
    if _echoes_source(out, src):
        return True
    if _MARKDOWN.search(out or ""):
        return True
    src_n = len(re.findall(r"\w+", src or ""))
    out_n = len(re.findall(r"\w+", out or ""))
    return src_n >= 4 and out_n > 4 * src_n + 8


def generate(tokenizer, model, text: str, *, source: str, target: str,
             context: str = "", preceding: str = "", device=None,
             max_new_tokens: int = 400, register: str = "narration",
             names: tuple[str, ...] | list[str] = (),
             numbers_spelled: bool = False, asr_source: bool = True) -> str:
    src = (text or "").strip()
    if not src:
        return ""
    out = _run(tokenizer, model,
               _translate_instruction(src, source, target, context, preceding=preceding,
                                      register=register, names=names,
                                      numbers_spelled=numbers_spelled,
                                      asr_source=asr_source),
               max_new_tokens)
    if _not_a_translation(out, src):
        tgt = _lang(target)
        retry = _run(tokenizer, model,
                     _translate_instruction(src, source, target, context,
                                            extra=f"Never repeat or quote the source "
                                                  f"text, never explain or use any "
                                                  f"formatting; answer with the plain "
                                                  f"{tgt} translation alone.",
                                            preceding=preceding, register=register,
                                            names=names,
                                            numbers_spelled=numbers_spelled,
                                            asr_source=asr_source),
                     max_new_tokens)
        if _not_a_translation(retry, src):
            print("  translate: output is not a plain translation twice → giving up",
                  file=sys.stderr)
            return ""                      # fails is_target_text → segment keeps original
        out = retry
    out = _repair_repeat(tokenizer, model, src, out, source, target, context,
                         max_new_tokens, preceding, register, names, numbers_spelled,
                         asr_source)
    return _strip_trailing_clause_repeat(out)


def _repair_repeat(tokenizer, model, src: str, out: str, source: str, target: str,
                   context: str, max_new_tokens: int, preceding: str = "",
                   register: str = "narration",
                   names: tuple[str, ...] | list[str] = (),
                   numbers_spelled: bool = False, asr_source: bool = True) -> str:
    """Fix an "education, education" collapse if the first pass produced one.

    Greedy decoding is faithful but occasionally flattens two distinct source words
    onto one English word. Re-ask once, with an added instruction to use distinct
    words for distinct items; if the model still repeats, strip the duplicate so the
    listener never hears the same word twice. Both steps are no-ops on clean output,
    so the common case pays only the cheap `_adjacent_repeat` scan.
    """
    dup = _adjacent_repeat(out, target)
    if not dup:
        return out
    tgt = _lang(target)
    extra = (f"If the source lists several items, give each a distinct {tgt} word — "
             f"do not write the same word twice in a row or in a list.")
    retry = _run(tokenizer, model,
                 _translate_instruction(src, source, target, context, extra=extra,
                                        preceding=preceding, register=register,
                                        names=names, numbers_spelled=numbers_spelled,
                                        asr_source=asr_source),
                 max_new_tokens)
    if is_target_text(retry, target) and not _adjacent_repeat(retry, target):
        return retry
    fixed = _strip_adjacent_repeat(out, target)
    print(f"  translate: de-duplicated repeated word {dup!r}", file=sys.stderr)
    return fixed


def _strip_trailing_clause_repeat(text: str) -> str:
    """Drop a final clause that exactly duplicates an earlier one in the output.

    Greedy decoding occasionally loops at the end of a long line and re-emits a
    clause it already produced ("…may the Lord turn His face toward you and be
    gracious to you, may the Lord turn His face toward you."). Deliberate spoken
    repetition is short or non-adjacent-identical; a verbatim ≥4-word clause that
    already appeared in the same line is the decode loop, and the listener hears
    it as a stutter. Language-agnostic: clauses split on , ; : and sentence ends.
    """
    if not text:
        return text
    parts = re.split(r"([,;:.!?…]+\s*)", text.strip())
    clauses = parts[0::2]
    seps = parts[1::2] + [""] * (len(parts[0::2]) - len(parts[1::2]))
    while len(clauses) > 1:
        last = clauses[-1].strip() or (clauses[-2].strip() if not clauses[-1] else "")
        if not clauses[-1].strip():          # trailing empty piece after final period
            clauses, seps = clauses[:-1], seps[:-1]
            continue
        norm = re.sub(r"\W+", " ", clauses[-1]).strip().lower()
        if len(norm.split()) < 4:
            break
        earlier = re.sub(r"\W+", " ", " ".join(clauses[:-1])).strip().lower()
        if f" {norm} " not in f" {earlier} ":
            break
        print(f"  translate: dropped repeated trailing clause {last[:40]!r}",
              file=sys.stderr)
        clauses, seps = clauses[:-1], seps[:-1]
    out = "".join(c + s for c, s in zip(clauses, seps)).rstrip()
    if out and out[-1] not in ".!?…":
        out = out.rstrip(",;: ") + "."
    return out


def _finalize_numbers(text: str, target: str) -> str:
    """Deterministic digits→words on a hop's output (see dubbing/numwords.py).

    English output is always spelled: the model translates digits *as* digits
    into English and code converts them, so "504" can never become "five zero
    four". For other targets this is a safety net over digits the model passed
    through (its input was already spelled, so digits mean it disobeyed, or a
    direct hop leaked them) — a nominative cardinal beats a digit-with-suffix
    for TTS even when the case is imperfect.
    """
    if target == "en":
        return numwords.spell_numbers(text, "en")
    if re.search(r"\d", text or ""):
        return numwords.spell_numbers(text, target)
    return text


# Scripts with no letter case: a capitalisation-based name check cannot work
# there, so `_proper_nouns` returns nothing and the name-guard simply skips.
_CASELESS_SCRIPTS = {"hebrew", "arabic", "cjk", "hangul", "devanagari"}


def _proper_nouns(text: str, target: str = "en") -> set[str]:
    """Capitalised words that are not merely sentence-initial.

    Works for any bicameral script (Latin, Cyrillic, Greek) via `str.isupper`.
    For caseless targets it returns the empty set — no signal, so the shorten
    name-guard skips rather than rejecting every rewrite.
    """
    if script_for(target) in _CASELESS_SCRIPTS:
        return set()
    out: set[str] = set()
    for match in re.finditer(r"\w[\w'-]{2,}", text or ""):
        if not match.group(0)[0].isupper():
            continue
        before = (text[: match.start()]).rstrip()
        if not before or before[-1] in ".!?":
            continue
        out.add(match.group(0).lower())
    return out


def _digits(text: str) -> set[str]:
    return set(re.findall(r"\d+", text or ""))


def _has_negation(text: str, target: str = "en") -> bool:
    negations = _NEGATIONS_BY_LANG.get(target, _NEGATIONS_BY_LANG["en"])
    low = (text or "").lower()
    if script_for(target) == "cjk":            # no spaces — substring is the word test
        return any(neg in low for neg in negations)
    words = set(re.findall(r"\w+", low))
    if words & negations:
        return True
    return target == "en" and "n't" in low     # \w+ splits "don't" apart


def shorten(processor, model, source_text: str, current_en: str, max_words: int, *,
            source: str, target: str, context: str = "", preceding: str = "",
            device=None) -> str | None:
    """Re-translate the source more concisely. None if meaning would be at risk.

    Re-translating (rather than compressing the English) keeps the model on its
    task, and the guards below refuse any rewrite that drops a number, a name, or
    a negation. It carries `preceding` for the same reason `generate` does: this
    starts from the source again, so without it the shortened line could come back
    with the word sense the full translation had just got right.
    """
    have = len((current_en or "").split())
    want = max(3, min(max_words, have - 1))
    src, tgt = _lang(source), _lang(target)
    # Same gloss gating as `run`: a shorten re-reads the source, so only the
    # gloss clauses whose word is in *this* source text belong in its prompt.
    # (On a pivot shorten the source here is the English intermediate — Latin
    # script, so the note passes through whole; see `relevant_context`.)
    context = relevant_context(context, source_text, source)
    hint = f"{context.strip()}\n\n" if context and context.strip() else ""
    before = (f"The line spoken just before was: {preceding.strip()}\n"
              f"It is background only — do not translate it and do not let any of its "
              f"words into your answer.\n\n") if preceding and preceding.strip() else ""
    instruction = (
        f"{hint}{before}"
        f"Translate the following {src} text into {tgt} as concisely as possible, in at "
        f"most {want} words, while keeping every name, number and negation. Output only "
        f"the {tgt} translation.\n\n{src}: {source_text}"
    )
    out = _run(processor, model, instruction, max(64, want * 4 + 40))
    out = (out or "").strip().strip('"')
    if not is_target_text(out, target) or "[[C" in out or _not_a_translation(out, source_text):
        return None
    n = len(out.split())
    if n >= have or n < max(3, 0.5 * have):
        return None
    # Numbers are spelled out deterministically (numwords, via _finalize_numbers),
    # so both sides are usually digit-free and this subset check passes vacuously.
    # That is accepted: parsing number-words per language is not worth it, and the
    # name and negation guards below still stand between a shortening and a lost fact.
    if not _digits(current_en) <= _digits(out):
        return None
    keep_names = _proper_nouns(current_en, target)
    if keep_names:
        low = out.lower()
        if sum(1 for name in keep_names if name in low) < 0.8 * len(keep_names):
            return None
    if _has_negation(current_en, target) and not _has_negation(out, target):
        return None
    # Spell any digits after the guards (which compare digit sets raw): the
    # spoken line must be digit-free, exactly like a full translation's.
    return _finalize_numbers(out, target)


def run(m: dict[str, Any], workdir: Path, *, source: str, target: str, save=None,
        register: str = "narration") -> None:
    from . import manifest

    segments = m["segments"]
    # A third-language keep gets its subtitle translated below; every other keep
    # is subtitled with its own text as before.
    subs = [s for s in segments
            if s["keep"] and s.get("keep_reason") == "foreign"
            and s.get("lang") and s["lang"] != "und"
            and (s.get("text") or "").strip() not in ("", "…")
            and not (s.get("text_en") or "").strip()]
    for seg in segments:
        if seg["keep"] and seg not in subs and not (seg.get("text_en") or "").strip():
            # A third-language keep whose text never got a target rendering (an
            # "und" verdict, or the translation below fails) must not put a
            # foreign-script line in the subtitles — the placeholder is honest.
            if seg.get("lang") and not is_target_text(seg["text"], target):
                seg["text_en"] = "…"
            else:
                seg["text_en"] = seg["text"]

    dub = [s for s in segments if not s["keep"]]
    todo = [s for s in dub if not (s.get("text_en") or "").strip()]
    if not todo and not subs:
        return
    context = m["source"].get("context") or ""
    # What was said just before each segment, kept or dubbed — the line the viewer
    # has already heard. Only ever shown to the model as background (see
    # `_translate_instruction`); the segment translated is still the segment alone.
    before = {s["id"]: prev["text"] for prev, s in zip(segments, segments[1:])}
    mids: dict[int, str] = {}      # English intermediates produced this run
    # Rolling proper-noun lists per hop target, grown from this run's own output
    # (run-derived, never per-video config): each hop is told the spellings its
    # own language already established, so a recurring name stays one name.
    established: dict[str, list[str]] = {"en": [], target: []}
    processor, model, device = load()
    try:
        for n, seg in enumerate(dub, 1):
            if (seg.get("text_en") or "").strip():
                continue
            # Translate each segment on its own — standalone output is deterministic
            # and faithful, which matters more for a dub than resolving a pronoun; a
            # marked multi-segment window makes the model drift a clause onto its
            # neighbour. The per-video `context` supplies names the ASR mangles, and
            # the preceding line settles a word sense the sentence alone cannot.
            # A dubbable third-language segment (--dub-foreign) names its own source
            # language and decides the pivot for itself: an English span on a he→ru
            # run goes en→ru directly, one hop. Its neighbour spoke a different
            # language, so the preceding line carries no tie-breaking signal — pass
            # none rather than mislead the model.
            seg_src = seg.get("lang") or source
            seg_pivot = pivot_via_english(seg_src, target)
            preceding = "" if seg.get("lang") else before.get(seg["id"], "")
            # Gloss clauses in the user context apply only where their word is
            # spoken; gate against the ORIGINAL source text on both hops — the
            # second hop reads clean English, but the gloss's reason to exist
            # (or not) is still the source segment (see `relevant_context`).
            seg_ctx = relevant_context(context, seg["text"], seg_src)
            if seg_pivot:
                # src→en→tgt: the English hop is the measured-good line; the
                # direct pair substitutes entities and bleeds neighbours (A/B'd).
                mid = generate(processor, model, seg["text"], source=seg_src,
                               target="en", context=seg_ctx,
                               preceding=preceding, device=device,
                               register=register,
                               names=tuple(established["en"]))
                if not is_target_text(mid, "en"):
                    text = ""
                else:
                    # Digits → English words in code, before the intermediate is
                    # stored or handed to the target hop: the model translates
                    # digits as digits, and only code converts them (numwords).
                    mid = numwords.spell_numbers(mid.strip(), "en")
                    mids[seg["id"]] = mid
                    established["en"] = update_established_names(
                        established["en"], mid, "en")
                    # No `preceding` on the second hop: disambiguation happened on
                    # the hop that read the noisy source, and clean English needs
                    # none — with it, context pressure once swapped an entity
                    # ("Jabhat al-Nusra" became "al-Qaeda's Front"; reproduced
                    # deterministically, correct the moment preceding was dropped).
                    text = generate(processor, model, mid, source="en",
                                    target=target, context=seg_ctx,
                                    preceding="",
                                    device=device, register=register,
                                    names=tuple(established[target]),
                                    numbers_spelled=True, asr_source=False)
            else:
                # A direct en→tgt hop gets the same treatment as the pivot's
                # second hop: spell the English digits in code first, then ask
                # the model only to keep the number-words as words.
                src_text = seg["text"]
                en_direct = seg_src == "en" and target != "en"
                if en_direct:
                    src_text = numwords.spell_numbers(src_text, "en")
                text = generate(processor, model, src_text, source=seg_src,
                                target=target, context=seg_ctx,
                                preceding=preceding, device=device,
                                register=register,
                                names=tuple(established[target]),
                                numbers_spelled=en_direct)
            # target=="en": spell the final English; otherwise: safety net over
            # any digits the model passed through.
            text = _finalize_numbers(text, target)
            if is_target_text(text, target):
                seg["text_en"] = text.strip()
                established[target] = update_established_names(
                    established[target], seg["text_en"], target)
                if seg_pivot:
                    seg["text_mid"] = mids[seg["id"]]
            else:
                seg["keep"], seg["keep_reason"] = True, "mt_failed"
                seg["text_en"] = seg["text"]
                print(f"  translate: seg {seg['id']} failed → keep original",
                      file=sys.stderr)
            if n % 8 == 0:
                print(f"  translate: {n}/{len(dub)}", file=sys.stderr)
                if save:
                    save()
        for seg in subs:
            # Subtitle-only: the audio stays original, so a failure here just
            # leaves the span's own transcription as the subtitle. Gloss gating
            # applies here too, against the span's own text and language.
            seg_ctx = relevant_context(context, seg["text"], seg["lang"])
            if pivot_via_english(seg["lang"], target):
                mid = generate(processor, model, seg["text"], source=seg["lang"],
                               target="en", context=seg_ctx, device=device)
                text = "" if not is_target_text(mid, "en") else generate(
                    processor, model, numwords.spell_numbers(mid.strip(), "en"),
                    source="en", target=target, context=seg_ctx, device=device,
                    numbers_spelled=True, asr_source=False)
            else:
                text = generate(processor, model, seg["text"], source=seg["lang"],
                                target=target, context=seg_ctx, device=device)
            text = _finalize_numbers(text, target)
            seg["text_en"] = text.strip() if is_target_text(text, target) else "…"
    finally:
        free(model)
    manifest.save(workdir, m)
    # Dubbed segments must have a translation; kept segments need not (an
    # "uncovered" keep is untranscribed audio that only plays as original sound).
    missing = [s["id"] for s in segments
               if not s.get("keep") and not (s.get("text_en") or "").strip()]
    assert not missing, f"dubbed segments without text_en: {missing}"


def free(model) -> None:
    import gc

    if isinstance(model, WorkerHandle):
        if model.own_gpu and model._proc.poll() is None:
            return  # resident on its own GPU — stays hot for the next stage
        model.close()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        return
    del model
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass
