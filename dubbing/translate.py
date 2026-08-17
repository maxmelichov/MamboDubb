"""Stage 5 general-purpose Gemma as a comprehension translator, one segment at a time.

A dedicated MT model (TranslateGemma) translates literally and has no world
knowledge, so it drops or garbles the references that matter "האחים המוסלמים"
(the Muslim Brotherhood, an organisation) came back as "brotherhood among
Muslims", and "שייח'ה" (the Sheikha) as "a CEO". A general instruction model
*understands* the text: it keeps the named organisations and people, and asked
to every clause and detail, rather than paraphrasing them away. That is what a
dub needs: the meaning, complete, in natural English.

The model is general Gemma 4 (12B) run through MLX (Apple-native), quantised so it
fits in unified memory. Each segment is translated on its own, deterministically
(greedy); a segment never sees its neighbours, which keeps the pieces from bleeding
into each other.

Which 12B quantisation matters more than it looks. Scored on the 30 dubbed segments
of the Qatar piece against nine things the Hebrew must survive "הגז הקטארי" as
"Qatari gas", "שתי ההפיכות" as "two coups", "לבנה השני" as "her second son",
"שלושה עשורים" as "three decades", the named organisations intact:

  6bit       9.7 GB  9/9, and the fastest of the ones that pass. The default.
  qat-4bit  10.9 GB  9/9 too, but 2.4x slower (its MLP layers are 8-bit) and bigger.
  8bit      12.7 GB  8/9 loses "her second son". No gain for the extra 3 GB.
  4bit       6.3 GB  5/9. Same footprint as the Gemma 3 it replaces, and reads
                     fluently, which is what makes it dangerous: "the Qatari fund",
                     "two turnovers", "his son", "the following two decades".
  mxfp4      6.3 GB  6/9, about 3x faster, but "the Qatari loot" for the gas.

For reference the Gemma 3 12B 4-bit it replaces scored 7/9. Below 6 bits this model
does not hold Hebrew, so trading precision for memory is a real quality decision, not
a free one but if 9.7 GB is too much, MODEL_PATH is the only line to change.

Those scores are for a segment translated alone. Showing the model the previous line
(see `_PRECEDING`) trades one of them "שתי ההפיכות" comes back as "two reversals"
rather than "two coups", though "who overthrew his father" still carries it for two
worse failures it fixes: "יגנו עליה" flipping from "defend her" to "condemn her", and
a segment losing its subject entirely ("Qatar in colors that…" for "She paints Qatar
in colors that…"). A wrong word beats a reversed meaning, so it stays.
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import re
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Any

from . import USER_KEEP_REASONS, numwords
from .script import count_letters, is_script, same_script, script_for

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "gemma-4-12B-it-6bit"
HUB_ID = "mlx-community/gemma-4-12B-it-6bit"
# Off-Mac backends: the same Gemma 4 12B, bf16 weights, run on CUDA in an isolated
# venv (see translator/) because the main venv's transformers 4.57.3 pin predates the
# gemma4_unified architecture. Two workers live there and speak one protocol vLLM
# (throughput; Linux only) and plain transformers (the fallback, and the only CUDA
# option on Windows). See `select_backend`.
CUDA_MODEL_PATH = REPO_ROOT / "models" / "gemma-4-12b-it-cuda"

# Words whose loss flips a sentence's meaning, per target language. `shorten`
# refuses any rewrite that drops one. Small on purpose: high-precision markers,
# not a linguistics project a miss only means a shortening is (safely) refused.
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

    Gemma 4 changed the turn syntax `<|turn>user … <turn|>` where Gemma 3 had
    `<start_of_turn>user … <end_of_turn>` so the markers can no longer be written
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
    itself from the raw user text. Post-processing is shared the worker decodes
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
    if "<channel|>" in out:              # a thought channel opened anyway keep the answer
        out = out.split("<channel|>")[-1]
    for marker in _TURN_MARKERS:
        out = out.split(marker)[0]
    return _strip_editorial(out)


class WorkerHandle:
    """A translation model living in a subprocess, spoken to over JSON lines.

    Wraps `translator/worker.py` (its own uv venv, transformers >= 5, CUDA): one
    request line in, one response line out, ids matched so a desynced pipe is an
    error rather than a wrong answer. The worker's stderr is streamed through to
    ours (model loading takes minutes the first time venv build plus 24 GB of
    weights) and its tail is kept for the crash message.
    """

    def __init__(self, cmd: list[str], ready_timeout: float = 3600.0,
                 env: dict[str, str] | None = None, own_gpu: bool = False):
        self.own_gpu = own_gpu
        # Both ends of the pipe must agree on UTF-8, and only this end can be set
        # from here: a Windows child defaults its stdio to the ANSI code page,
        # which cannot write a Hebrew translation at all.
        #
        # `errors="replace"` for the same reason `dubbing_app/runner.py` and
        # `install.py` carry it: the child's stderr is not ours to guarantee — a
        # CUDA library, a progress bar, a truncated multi-byte flush — and one
        # undecodable byte raises inside the pump *thread*, which dies silently.
        # The parent then sits on `_read_line` for the whole 3600s ready_timeout
        # waiting for a worker that is already talking. A mangled log character
        # is the cheaper failure.
        env = {**(env if env is not None else os.environ), "PYTHONIOENCODING": "utf-8"}
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                      errors="replace", env=env)
        self._stderr_tail: deque[str] = deque(maxlen=40)
        self._next_id = 0
        self._lines: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._pump_stdout, daemon=True).start()
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

    def _pump_stdout(self) -> None:
        """Every reply line, into a queue; `None` marks the pipe closing.

        A thread rather than `select` on the pipe, because the two disagree about
        where a line is. `select` answers about the *file descriptor*, while
        `readline` reads from Python's decoding buffer above it and a worker that
        writes several replies at once (a batch) has them all pulled into that
        buffer by the first read. `select` then reports "nothing to read" on a
        drained fd while three answers sit one layer up, and the parent waits out
        its full timeout with the replies already in hand. Reading eagerly puts
        the line boundary in one place instead of two.
        """
        for line in self._proc.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def _pump_stderr(self) -> None:
        for line in self._proc.stderr:
            sys.stderr.write(line)
            sys.stderr.flush()
            self._stderr_tail.append(line)

    def _stderr_note(self) -> str:
        tail = "".join(self._stderr_tail).strip()
        return f"; recent worker stderr:\n{tail}" if tail else ""

    def _read_line(self, timeout: float) -> str:
        """The next reply line. Raises on timeout and on a worker that died."""
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError(f"translator worker: no response within {timeout:.0f}s"
                               f"{self._stderr_note()}") from None
        if line is None:                     # stdout closed: the worker is gone
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            self._lines.put(None)            # every later read fails the same way
            raise RuntimeError(f"translator worker died (exit {self._proc.returncode})"
                               f"{self._stderr_note()}")
        return line

    def _write(self, msg: dict) -> None:
        try:
            self._proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(f"translator worker pipe closed: {exc}"
                               f"{self._stderr_note()}") from exc

    def request(self, user_text: str, max_new_tokens: int,
                timeout: float = 600.0) -> str:
        self._next_id += 1
        rid = self._next_id
        self._write({"id": rid, "user_text": user_text,
                     "max_new_tokens": max_new_tokens})
        reply = json.loads(self._read_line(timeout))
        if reply.get("error"):
            raise RuntimeError(f"translator worker error: {reply['error']}")
        if reply.get("id") != rid:
            raise RuntimeError(f"translator worker desynced: sent id {rid}, "
                               f"got {reply.get('id')!r}")
        return reply.get("text") or ""

    def request_many(self, items: list[tuple[str, int]],
                     timeout: float = 900.0) -> list[str]:
        """Answer several independent prompts in one round trip. Order preserved.

        The whole reason the vLLM backend exists: it batches continuously, so N
        prompts submitted together cost far less than N submitted one after the
        other. The transformers worker understands the same message and simply
        decodes them in turn, so a caller never has to know which worker it got.

        Replies are matched by id and may arrive in any order a batched engine
        finishes short prompts first and reordering them back here is what keeps
        that invisible. An error on any item fails the whole call: these are
        independent prompts, and a caller silently receiving one blank of N is
        the failure mode this protocol's ids exist to prevent.
        """
        if not items:
            return []
        if len(items) == 1:
            return [self.request(items[0][0], items[0][1], timeout=timeout)]
        reqs = []
        for user_text, max_new_tokens in items:
            self._next_id += 1
            reqs.append({"id": self._next_id, "user_text": user_text,
                         "max_new_tokens": int(max_new_tokens)})
        self._write({"batch": reqs})
        pending = {r["id"] for r in reqs}
        got: dict[int, str] = {}
        while pending:
            reply = json.loads(self._read_line(timeout))
            if reply.get("error"):
                raise RuntimeError(f"translator worker error: {reply['error']}")
            rid = reply.get("id")
            if rid not in pending:
                raise RuntimeError(f"translator worker desynced: batch expected one of "
                                   f"{sorted(pending)}, got {rid!r}")
            pending.discard(rid)
            got[rid] = reply.get("text") or ""
        return [got[r["id"]] for r in reqs]

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
# Leading marks a spoken line never starts with "/In the year 2013", ".Be careful".
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
# keeps one spelling. Derived entirely from the run's output not per-video config.
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


# Capitalised tokens that are never names: the pronouns, address forms and
# interjections dialogue capitalises constantly ("His", "God", "Okay"; «Его»,
# «Это») measured self-poisoning the established-names list on a drama run
# (English) and a he→ru run (Cyrillic), where they crowded real names out of
# the rolling window. Pronouns only, small on purpose: caseless scripts yield
# no candidates at all, and a miss only costs one useless entry not a wrong
# translation. Capitalised ordinary adjectives («Свободной») cannot be listed
# without a vocabulary project; frequency canonicalisation absorbs those.
_NAME_STOP = frozenset({
    "you", "your", "yours", "him", "his", "her", "hers", "its", "they",
    "them", "their", "theirs", "our", "ours", "mine", "she", "god", "lord",
    "hey", "well", "yes", "okay", "sir", "madam", "mister", "miss",
    # Cyrillic pronouns/determiners (>= 3 letters; shorter ones never match
    # the name-token pattern anyway).
    "его", "её", "ее", "ему", "ней", "нём", "нем", "них", "ими", "оно",
    "она", "они", "это", "эта", "этот", "эти", "тот", "наш", "ваш", "мой",
    "твой", "свой", "себя", "кто", "что",
})


def _name_occurrences(text: str, target: str = "en") -> list[str]:
    """Every proper-noun occurrence in `text`, in order, original casing.

    The duplicate-preserving sibling of `update_established_names`'s extraction:
    occurrence counts are what `canonical_names` needs to pick a best-attested
    form, so unlike the rolling list this one keeps repeats. Contraction capitals
    ("I'm" any apostrophe-bearing token) and `_NAME_STOP` words are not names
    and never enter the table.
    """
    nouns = _proper_nouns(text, target)
    if not nouns:
        return []
    out = []
    for match in re.finditer(r"\w[\w'-]{2,}", text or ""):
        tok = match.group(0)
        if "'" in tok or "’" in tok:
            continue
        low = tok.lower()
        if low in nouns and low not in _NAME_STOP:
            out.append(tok)
    return out


def update_established_names(established: list[str], translation: str,
                             target: str = "en") -> list[str]:
    """The rolling established-names list after seeing one produced translation.

    Proper nouns (via `_proper_nouns`) from `translation` move to the front in
    their original casing; the list stays distinct (case-insensitive) and is
    capped at MAX_ESTABLISHED_NAMES, most recent first. Pure returns a new list.
    """
    fresh: list[str] = []
    for tok in _name_occurrences(translation, target):
        if tok.lower() not in (f.lower() for f in fresh):
            fresh.append(tok)
    if not fresh:
        return list(established)
    out = list(established)
    for name in reversed(fresh):
        out = [n for n in out if n.lower() != name.lower()]
        out.insert(0, name)
    return out[:MAX_ESTABLISHED_NAMES]


def _edit_distance_leq(a: str, b: str, k: int) -> bool:
    """True when Levenshtein(a, b) <= k. Tokens are short and k <= 2: plain DP."""
    if abs(len(a) - len(b)) > k:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > k:
            return False
        prev = cur
    return prev[-1] <= k


def _variant_names(a: str, b: str) -> bool:
    """True when `a` and `b` are plausibly two spellings of one name.

    Token-wise: the same word count, every differing token pair within a small
    edit distance 1, or 2 for tokens of 7+ letters ("Jolani"/"Julani",
    "Jabhat"/"Jablat" merge; "Golan" stays apart from both). Tokens under 5
    letters never merge: at that length one edit is a different word, not a
    misspelling ("Iran"/"Iraq"). Casefolded, so it works for any bicameral
    script Latin and Cyrillic alike.
    """
    ta, tb = a.casefold().split(), b.casefold().split()
    if len(ta) != len(tb):
        return False
    for x, y in zip(ta, tb):
        if x == y:
            continue
        short = min(len(x), len(y))
        if short < 5:
            return False
        if not _edit_distance_leq(x, y, 1 if short < 7 else 2):
            return False
    return True


def canonical_names(established: list[str] | tuple[str, ...]) -> list[str]:
    """Dedup and canonicalise a run's proper-noun list. Pure returns a new list.

    ASR noise makes one name accumulate as near-spellings across a run
    ("Jolani" / "Julani"; "Jabhat al-Nusra" / "Jablat al-Nusra"), and injecting
    the variants side by side offers the model a menu instead of an answer.
    Variants (see `_variant_names`) are grouped and each group is represented
    by its best-attested form most occurrences in the input (the list may
    carry repeats, e.g. `_name_occurrences` output), earliest first appearance
    on a tie. Output preserves order of first appearance, one entry per group.
    Derived entirely from the run's own translations never per-video config.
    """
    groups: list[dict[str, Any]] = []
    for idx, raw in enumerate(established or []):
        name = (raw or "").strip()
        if not name:
            continue
        low = name.casefold()
        home = None
        for g in groups:
            if low in g["forms"] or any(_variant_names(low, f) for f in g["forms"]):
                home = g
                break
        if home is None:
            home = {"forms": {}}
            groups.append(home)
        rec = home["forms"].setdefault(low, [0, idx, {}])
        rec[0] += 1
        rec[2][name] = rec[2].get(name, 0) + 1         # casing attestation counts
    out = []
    for g in groups:
        best = min(g["forms"].values(), key=lambda r: (-r[0], r[1]))
        # Insertion order breaks casing ties in favour of the earliest-seen form.
        out.append(max(best[2].items(), key=lambda kv: kv[1])[0])
    return out


# Letter runs, any script the tokens gloss gating compares. Digits and
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
      documentary about …") and is always included entity context helps everywhere.
    - A clause carrying source-script tokens is a gloss, and is included only when one
      of those tokens appears in the segment's own source text. Matching is gated by
      token length both limits are measured leak traps from the drama harness:
      tokens under 4 letters never participate at all (a clause glossing מ"פ matched
      by substring nearly everywhere and leaked run-wide), 4-letter tokens match only
      as an exact word (אותו reached unrelated segments through the fuzzy rule and
      perturbed them), and only 5+-letter tokens match as a substring or at edit
      distance <= 1 which still covers the ASR variants a gloss typically lists
      (בלאגן / מלאגן) without a per-video matching rule.

    When the source language is Latin-script the gloss/background split has no signal
    (every token is "source script") and the note passes through whole, as before.
    Pure text logic safe to call per segment, no models involved.
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
        hit = False
        for tok in gloss_tokens:
            low = tok.lower()
            if len(low) < 4:
                continue                   # too short to identify any word
            if len(low) == 4:
                if low in seg_tokens:      # exact word only no fuzz, no substring
                    hit = True
                    break
            elif (low in seg_text_low
                    or any(_within_one_edit(low, st) for st in seg_tokens)):
                hit = True
                break
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
                           asr_source: bool = True,
                           genre: str = "documentary") -> str:
    """Comprehension prompt: faithful and complete, not literal or summarised.

    `source` may be empty the honest answer for a line whose language nothing
    knows (see `segment_langs`). The prompt then names no source language at all
    ("Translate the following text into German") instead of asserting the run's,
    because a false claim about the input is worse than no claim: told a line of
    English is Hebrew, the model hands the English straight back, the echo guard
    correctly rejects it, and a translatable segment ends with nothing.

    `context` is an optional per-video note who and what the video is about, and
    the established spellings of names the ASR tends to mangle. The model uses it to
    render "שייח' עמוזה" (a mistranscription) as "Sheikha Moza" rather than guessing.
    `extra` is an optional extra sentence appended on a retry (see `_repair_repeat`).

    `preceding` is the line spoken just before this one, shown as context and never
    translated. A segment on its own can be genuinely ambiguous: "יגנו עליה" is
    "they will defend her", but every model tested Gemma 3 and 4 alike, at every
    quantisation read it as "they will condemn her" with nothing around it, and
    every one of them got it right once it could see the previous sentence ("…they
    will do everything to protect Qatar"). One line back is enough, and it stays
    strictly context: translating a window instead lets a clause drift from one
    segment onto its neighbour, which is why the segments are separate to begin with.

    `register` selects the speaking style: "narration" (the default, unchanged
    behaviour) or "dialogue" (natural spoken register for English that means
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
    mishearing note garbled words, names above all, should resolve to the
    phonetically-closest plausible reading and to the names the context
    establishes ("חוקרי השאר" → "al-Sharaa's researchers", not "the remaining
    researchers"; A/B'd without context the note fixes nothing, so it is
    never emitted alone). The pivot's second hop reads clean model English and
    passes False.
    """
    # An unknown source (see `segment_langs`) is named nowhere in the prompt: the
    # instruction asks about "the following text", because asserting a language
    # the pipeline cannot vouch for is the lie this exists to stop. A named source
    # renders byte-identically to before.
    src, tgt = (_lang(source) if source else ""), _lang(target)
    labelled = f"{src} " if src else ""
    hint = f"{context.strip()}\n\n" if context and context.strip() else ""
    tail = f" {extra.strip()}" if extra and extra.strip() else ""
    before = _PRECEDING.format(prev=preceding.strip()) if preceding and preceding.strip() else ""
    # genre="movie" implies the dialogue register: movie dubbing is people
    # talking, whatever register the caller left at its default.
    if genre == "movie":
        register = "dialogue"
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
    # Movie mode only: dialogue brevity plus borrowed-form greetings. Kept as a
    # separate genre block so the documentary prompt stays byte-identical.
    movie = (
        " This is dubbed movie dialogue: keep it as short and colloquial as the "
        "original an interjection stays an interjection; a greeting borrowed "
        "from another language (like 'welcome' or 'ahlan') stays in that borrowed "
        "form, not translated into formal words; never expand a short line."
    ) if genre == "movie" else ""
    # Numbers: digits are converted to words by code (numwords), never by the
    # model. Only a hop whose input is already spelled gets a sentence, and it
    # asks for inflection of existing words not conversion.
    numbers = (
        f" Numbers in the source are written out as words; render them as naturally "
        f"inflected words in {tgt}, never as digits."
    ) if numbers_spelled else ""
    names_note = (
        " Names already established in this video's translation use these exact "
        "spellings when the same person or place recurs: " + ", ".join(names) + "."
    ) if names else ""
    military = f" {_HE_MILITARY_NOTE}" if source == "he" else ""
    # Only with background context to match against: the bare instruction was
    # probed and fixed nothing (Gemma cannot do Hebrew phonetics on demand),
    # while with named entities in the context it reliably reconciles garbled
    # proper nouns to them and leaves clean lines untouched.
    garble = (
        f" The {labelled}text comes from automatic speech recognition and may mishear "
        f"words especially names: a garbled word may actually be one of the names "
        f"in the background above. When a word is incoherent in its context, prefer "
        f"the phonetically-closest plausible reading and match garbled names to the "
        f"background names; never turn a garble into a new name or entity."
    ) if asr_source and hint else ""
    # The hop from clean English into a non-English target gets the guarded
    # fluency license (A/B'd twice on 87 he→ru rows: unguarded scored 33 wins but
    # coined fake proper names; guarded kept 13 wins with zero severe regressions
    # and repaired one fidelity error). Never applied when translating the noisy
    # source directly the license presumes a coherent input.
    fluency = (
        f" Prioritize natural, idiomatic {tgt} phrasing: restructure the syntax "
        f"freely word order, clause boundaries, the verbs and collocations a "
        f"native speaker would use while preserving the semantics exactly: never "
        f"substitute a near-synonym that changes the claim, never re-attribute a "
        f"statement to a different speaker, and never add specifics of place, "
        f"origin or time the source does not state. Never coin, restore or "
        f"transliterate a proper name that is not literally in the source; only "
        f"when the source wording is garbled or incoherent, prefer a literal "
        f"rendering over a fluent guess. Deliberate repetition and wordplay are "
        f"kept, not smoothed away. Prefer the plain verb and the standard "
        f"collocation a native {tgt} speaker would use over bureaucratic or "
        f"copula-heavy phrasing; a word-for-word calque of an English "
        f"construction is wrong when {tgt} has an established way to say it."
    ) if source == "en" and target != "en" else ""
    # Same guard as the fluency license: false-friend military terms are a
    # measured en→ru failure (дивизион, an artillery battalion, for "division")
    # and the echelon rule fixed it with zero breaks on the controls.
    echelon = (
        f" Military unit types translate by echelon: a division, a brigade, a "
        f"battalion each map to the {tgt} term for the same echelon and size of "
        f"formation, never to a similar-sounding term for a different one."
    ) if source == "en" and target != "en" else ""
    return (
        f"{hint}{before}"
        f"Translate the following {labelled}text into clear, natural {tgt} for subtitles. "
        f"Understand the meaning and translate it faithfully and completely: keep every "
        f"clause, every detail and any repeated emphasis; preserve all names, "
        f"organizations, numbers and specific references, using their established {tgt} "
        f"names and respecting grammatical gender. A word or phrase the source has "
        f"borrowed from a third language and spelled phonetically in its own alphabet "
        f"is written in the standard spelling it has in {tgt}, never transliterated "
        f"back letter by letter. Do not summarize, shorten, omit, or "
        f"translate word-for-word. {style}{movie}{numbers}{fluency}{echelon}{garble}{names_note}{military}{tail} "
        f"Output only the {tgt} translation, nothing else no notes, no comments, no "
        f"square brackets and no alternative renderings.\n\n"
        f"{src or 'Text'}: {text}"
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

# "and"/"or" per target the word between the halves of an "X and X" collapse.
_COORDINATORS: dict[str, set[str]] = {
    "en": {"and", "or"}, "ru": {"и", "или"}, "es": {"y", "o"}, "fr": {"et", "ou"},
    "de": {"und", "oder"}, "it": {"e", "o"}, "pt": {"e", "ou"},
}


def _adjacent_repeat(text: str, target: str = "en") -> str | None:
    """The content word in an audible "X, X" / "X and X" collapse, or None.

    Catches a single content word repeated back-to-back (only a comma or a
    coordinator between) the shape a translator produces when it flattens two
    distinct source words ("השכלה וחינוך") onto one word. Phrase-level emphasis
    ("Qatar funds Hamas, Qatar funds Al-Qaeda") is not matched: there the repeated
    token has other words between its occurrences, so it is left intact. CJK
    targets are never matched without spaces `\\w+` tokens are whole runs of
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
    # "X. X." an ASR echo the source carried through ("65%. אחוז") lands as a
    # one-word sentence duplicating the word before it; keep the first copy and
    # its sentence punctuation. The detector is punctuation-blind, so the
    # stripper must accept every separator the detector does.
    text = re.sub(r"\b(\w{4,})([.!?…]+)\s*\1\b[.!?…]*(?=\s|$)", r"\1\2",
                  text, flags=re.IGNORECASE)
    text = re.sub(rf"\b(\w{{4,}})\b\s+(?:{coord})\s+\1\b", r"\1", text,
                  flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", text).strip()


def same_language(source: str, target: str) -> bool:
    """True when a segment's source and the target are one language.

    The pair then needs no translation at all the dub re-voices the line as it
    was spoken (see `run`). Case-folded only: codes are normalised upstream, and
    a language does not become a different one by being spelled in capitals.
    """
    return bool(source) and (source or "").lower() == (target or "").lower()


# Below this many letters a line carries no script evidence worth acting on: a
# number, an initial, a stray glyph, a one-word interjection.
MIN_SCRIPT_LETTERS = 4


def contradicts_source(text: str, source: str) -> bool:
    """True when `text` is demonstrably not written in `source`'s script.

    Script is a cheap, high-precision signal about what a line is *not*, and only
    that a Latin line inside a Hebrew run could be English, German or Turkish.
    That is still the whole answer to "is the run's `--src` true of this segment",
    which is the question worth asking before a prompt asserts it.

    Silent for a same-script pair and for a line with too few letters to judge:
    both are "no evidence", which is not the same as "no contradiction".
    """
    if sum(1 for ch in text or "" if ch.isalpha()) < MIN_SCRIPT_LETTERS:
        return False
    return not is_script(text or "", source)


def segment_langs(seg: dict[str, Any], source: str, target: str) -> tuple[str, str]:
    """This segment's (source, target) pair its own knowledge, then the run's.

    `--src` is a claim about the *video*, not about every line in it, and a video
    that mixes languages breaks the claim segment by segment. Three things know
    better, in order of authority: the editor's `src_lang` override (the user said
    so), the span witness `lang` the transcript stage left (it heard so), and the
    segment's own text, which is written in the script of whatever was spoken.

    The third can only *refute*, so where it is all there is the source comes back
    unknown (`""`) rather than guessed. Unknown is a usable answer, not a failure:
    `_translate_instruction` then asserts no source language at all and
    `pivot_via_english` sends the line straight to the target instead of through
    an English hop premised on the claim just refuted. Told a line of English is
    Hebrew, the model hands the English straight back, `_not_a_translation`
    correctly rejects the echo, and a perfectly translatable segment ends with
    nothing which is the whole bug this exists to stop.

    `tgt_lang` is the editor's target override; a target has no witness, only the
    run's word and the user's.
    """
    tgt = (seg.get("tgt_lang") or target or "").strip().lower()
    known = (seg.get("src_lang") or seg.get("lang") or "").strip().lower()
    if known:
        return known, tgt
    src = (source or "").strip().lower()
    if src and contradicts_source(seg.get("text") or "", src):
        return "", tgt
    return src, tgt


def pivot_via_english(source: str, target: str) -> bool:
    """True when translation must route src→en→tgt instead of going direct.

    A judged A/B over 84 segments settled this: direct Gemma he→ru lost 38–7
    with disqualifying failures (entity substitution, hallucinated speaker
    labels, neighbour-segment bleed), while the pivot's errors were almost all
    inherited from the measured-good English line so he→en improvements
    propagate to every target. Codes are normalised upstream; no aliasing here.

    A same-language pair pivots nowhere: there is no hop to make. Neither does an
    unknown source (`segment_langs`) the first hop would have to name a language
    nothing knows, and a hop that lies about its input is what the pivot exists to
    avoid.
    """
    if not source or same_language(source, target):
        return False
    return source != "en" and target != "en"


def mark_failed(seg: dict[str, Any]) -> bool:
    """Record that this segment has no translation. True when it now keeps.

    "Never silent" (AGENTS.md, invariant 1) says a segment the translator could
    not do plays its original audio, with its own text as the subtitle. That is
    the pipeline's answer to give but only while the verdict is still the
    pipeline's to make. `keep.user_wants_dub` names the case where it is not: the
    user asked for this line to be dubbed, and answering with a keep both
    overrules them (it restores the very audio they were replacing, so the button
    appears to have done nothing) and contradicts the manifest, which then
    flip-flops on every run.

    So the user's "dub it" stands. The segment stays a dub with no `text_en` —
    the `untranslated` state the editor already paints and offers a retry for.
    Visibly unfinished beats invisibly reverted, and the floor still holds: `tts`
    gives an unspeakable dub its original audio under its own honest reason.
    """
    from . import keep as keep_mod

    if keep_mod.user_wants_dub(seg):
        seg.pop("text_en", None)
        seg.pop("text_mid", None)
        return False
    seg["keep"], seg["keep_reason"] = True, "mt_failed"
    seg["text_en"] = seg["text"]
    return True


def untranslated(segments: list[dict[str, Any]]) -> list[int]:
    """Dub-wanted segments with no translation work a stage stamp cannot close.

    `mark_failed` honours a user-locked dub by leaving it visibly unfinished:
    keep=false, no `text_en`. Right per-segment answer, wrong stage stamp a
    manifest stamped "translate done" while these exist makes every later run
    skip the stage, and the retry the editor promises never happens on a
    re-run. The CLI's stage gate treats a non-empty answer here as not-done,
    whatever the fingerprint says.
    """
    return [s["id"] for s in segments
            if not s.get("keep") and not (s.get("text_en") or "").strip()]


def is_target_text(text: str, target: str = "en") -> bool:
    """Reject empty / untranslated (still source-script) output.

    "Did it actually translate?" at least two letters of the target's script,
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
    declares, so this keeps working for the next model that ships a list and for one
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


_VLLM_INSTALLED: bool | None = None


def _vllm_installed() -> bool:
    """True when the translator venv carries vLLM (the `vllm` extra was synced).

    Deliberately a filesystem look, not an import: vllm must never be imported
    into the main venv (it pins its own torch, which is exactly why the worker
    lives in another venv), and spawning `uv run … -c "import vllm"` to ask would
    cost a venv resolution on a question asked before every load. The answer is
    cached for the process a sync mid-run is not a case worth serving.

    Both venv layouts are checked (`lib/python*/site-packages` on POSIX,
    `Lib/site-packages` on Windows) so the probe is the same code everywhere,
    even though the answer on Windows is always False for want of wheels.
    """
    global _VLLM_INSTALLED
    if _VLLM_INSTALLED is None:
        venv = REPO_ROOT / "translator" / ".venv"
        _VLLM_INSTALLED = any(venv.glob(pattern) for pattern in
                              ("lib/python*/site-packages/vllm/__init__.py",
                               "Lib/site-packages/vllm/__init__.py"))
    return _VLLM_INSTALLED


def _cuda_count() -> int:
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _vllm_available() -> bool:
    """True when the vLLM backend can actually run here.

    Three conditions, all cheap: the extra is synced into the translator venv,
    the platform is Linux (vLLM publishes no Windows wheels the transformers
    worker is the only CUDA backend there), and a CUDA device exists at all.
    """
    return _vllm_installed() and sys.platform.startswith("linux") and _cuda_count() > 0


BACKENDS = ("mlx", "vllm", "transformers")
BACKEND_ENV = "DUBBING_TRANSLATOR_BACKEND"


def select_backend(forced: str, mlx_ok: bool, vllm_ok: bool) -> str:
    """Which translation backend to load. Pure the probes are the caller's job.

    Order, when nothing is forced: MLX (a Mac, or anywhere mlx_lm imports) → the
    vLLM worker (Linux with the extra synced and a CUDA device) → the
    transformers worker. The last one is the floor rather than a fourth "cpu"
    backend because it already *is* the CPU path: `translator/worker.py` picks
    cuda:0 when torch sees a GPU and cpu when it does not, so a box with no CUDA
    at all lands there and works, slowly.

    `DUBBING_TRANSLATOR_BACKEND` overrides the order outright and is honoured even
    when the probe says no on purpose: the reason to force a backend is usually
    that the probe is wrong about this machine, and a forced choice that silently
    fell back would hide exactly the failure the user asked to see. An
    unrecognised value is a typo, not a backend: it warns and auto-selects.
    """
    want = (forced or "").strip().lower()
    if want in BACKENDS:
        return want
    if want not in ("", "auto"):
        print(f"  translate: unknown {BACKEND_ENV}={forced!r} (expected one of "
              f"{', '.join(BACKENDS)} or auto) → choosing automatically", file=sys.stderr)
    if mlx_ok:
        return "mlx"
    if vllm_ok:
        return "vllm"
    return "transformers"


def _backend() -> str:
    """The backend this machine will use, probes and env override included."""
    return select_backend(os.environ.get(BACKEND_ENV, ""), _mlx_available(),
                          _vllm_available())


def _worker_cmd(backend: str = "transformers") -> list[str]:
    """Launch line for a CUDA worker, through the translator uv project venv.

    Both workers share one venv (see translator/pyproject.toml), and `uv run`
    re-syncs that venv to the extras its own command line names so a launch
    without `--extra vllm` would uninstall vLLM out from under the next run's
    probe. The flag therefore follows the *venv's* state, not the chosen backend.
    """
    project = REPO_ROOT / "translator"
    script = "worker_vllm.py" if backend == "vllm" else "worker.py"
    extra = ["--extra", "vllm"] if _vllm_installed() else []
    return ["uv", "run", "--project", str(project), *extra, "python",
            str(project / script), str(CUDA_MODEL_PATH)]


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
    loaded and skip the close/reload dance. True of either CUDA worker: vLLM
    reserves its KV cache up front, which makes sharing a card with TTS *worse*,
    not better.
    """
    return _backend() == "mlx" or _spare_gpu() is None


def load(device: str | None = None):
    """Load Gemma 4 12B on whichever backend this machine has.

    On a Mac (or anywhere mlx_lm imports): the MLX quant, in-process unchanged.
    Otherwise a subprocess worker with its own venv (the main venv's transformers
    4.57.3 pin predates Gemma 4; see translator/), running either vLLM
    (`worker_vllm.py`, Linux + CUDA + the `vllm` extra synced) or plain
    transformers (`worker.py`, the fallback that also covers Windows, where vLLM
    has no wheels, and CPU-only boxes). Either way the return shape is the
    transformers-era (processor, model, device) triple the call sites expect; the
    worker backends return (None, WorkerHandle, None) and `_run`/`free` dispatch
    on the handle.

    Both workers speak one protocol, so nothing above this line changes with the
    backend and neither do the prompts or the post-processing, which live in this
    module for exactly that reason. `DUBBING_TRANSLATOR_BACKEND` forces the choice.

    A worker holding its own GPU is spawned once and reused: `free` leaves it
    running and `load` hands the live handle back.
    """
    global _WORKER
    backend = _backend()
    if backend != "mlx":
        if _WORKER is not None and _WORKER._proc.poll() is None:
            return None, _WORKER, None
        gpu = _spare_gpu()
        env = None
        if gpu is not None:
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
            print(f"  translate: CUDA worker pinned to GPU {gpu}", file=sys.stderr)
        print(f"  translate: starting {backend} worker for {CUDA_MODEL_PATH.name} "
              f"(first run builds translator/ venv + loads bf16 weights)", file=sys.stderr)
        handle = WorkerHandle(_worker_cmd(backend), env=env, own_gpu=gpu is not None)
        if handle.own_gpu:
            _WORKER = handle
            atexit.register(handle.close)
        return None, handle, None
    return _load_mlx()


def _load_mlx():
    """Load general Gemma 4 (12B, 4-bit) through MLX.

    Returns (tokenizer, model, None) to match the transformers-era
    (processor, model, device) call sites device is unused under MLX.
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
        # actually declares naming each token after itself and let it override the
        # file. Harmless to retry: nothing else in load() has taken effect yet.
        model, tokenizer = mlx_lm.load(path, tokenizer_config=_special_token_map(path))
    # Gemma ends a turn with a turn marker, not <eos>. Without it in the stop set
    # generation runs to max_tokens every time slow, and the text trails a wall
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
    'Source:** "The second is that…"' its English input in a markdown wrapper —
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
    model's full working notes in plain markdown 'Source:** … **Translation
    Strategy:** … **Final Version:** …', 262 tokens for a 26-token source, cut
    off at the token limit. Three shapes betray it, all impossible for a real
    subtitle: heavy token overlap with the source (echo), markdown structure,
    and gross length blow-up (a translation of a 4+-token source stays within a
    few multiples of its length; only sources with enough tokens are judged, so
    CJK sources few `\\w+` tokens are exempt rather than misjudged).
    """
    if _echoes_source(out, src):
        return True
    if _MARKDOWN.search(out or ""):
        return True
    src_n = len(re.findall(r"\w+", src or ""))
    out_n = len(re.findall(r"\w+", out or ""))
    return src_n >= 4 and out_n > 4 * src_n + 8


def _leaks_source_script(out: str, source: str, target: str) -> bool:
    """True when a hop's output carries letters of the source language's script.

    `is_target_text` only requires a target-script *majority*, so a line that is
    mostly translated but drags a source-script word along sails through it —
    and TTS then reads the foreign word aloud in the middle of the dub. Only
    meaningful when the two scripts differ; same-script pairs have no signal
    and are never flagged. Nor is an unknown source (`segment_langs`): there is
    no source script to hunt for, and `script_for`'s "latin" default for an
    unknown code would condemn every Latin word in a Hebrew or Russian dub.
    """
    if not source or same_script(source, target):
        return False
    return count_letters(out or "", script_for(source)) > 0


def _retry_extra(out: str, src: str, source: str, target: str) -> str:
    """The retry instruction an unusable first pass earns, or "" when clean.

    Two rejection classes share the one-retry mechanism: output that is not a
    plain translation at all (echo, markdown notes, length blow-up see
    `_not_a_translation`) and output that leaks source-script letters into a
    different-script target (see `_leaks_source_script`).
    """
    tgt = _lang(target)
    if _not_a_translation(out, src):
        return (f"Never repeat or quote the source text, never explain or use any "
                f"formatting; answer with the plain {tgt} translation alone.")
    if _leaks_source_script(out, source, target):
        return (f"Write the answer entirely in {tgt}; not a single {_lang(source)} "
                f"letter may appear in it.")
    return ""


def generate(tokenizer, model, text: str, *, source: str, target: str,
             context: str = "", preceding: str = "", device=None,
             max_new_tokens: int = 400, register: str = "narration",
             names: tuple[str, ...] | list[str] = (),
             numbers_spelled: bool = False, asr_source: bool = True,
             genre: str = "documentary") -> str:
    src = (text or "").strip()
    if not src:
        return ""
    out = _run(tokenizer, model,
               _translate_instruction(src, source, target, context, preceding=preceding,
                                      register=register, names=names,
                                      numbers_spelled=numbers_spelled,
                                      asr_source=asr_source, genre=genre),
               max_new_tokens)
    extra = _retry_extra(out, src, source, target)
    if extra:
        retry = _run(tokenizer, model,
                     _translate_instruction(src, source, target, context, extra=extra,
                                            preceding=preceding, register=register,
                                            names=names,
                                            numbers_spelled=numbers_spelled,
                                            asr_source=asr_source, genre=genre),
                     max_new_tokens)
        if _retry_extra(retry, src, source, target):
            print("  translate: output unusable twice (echo/leak) → giving up",
                  file=sys.stderr)
            return ""                      # fails is_target_text → segment keeps original
        out = retry
    out = _repair_repeat(tokenizer, model, src, out, source, target, context,
                         max_new_tokens, preceding, register, names, numbers_spelled,
                         asr_source, genre)
    return _strip_trailing_clause_repeat(out)


def _repair_repeat(tokenizer, model, src: str, out: str, source: str, target: str,
                   context: str, max_new_tokens: int, preceding: str = "",
                   register: str = "narration",
                   names: tuple[str, ...] | list[str] = (),
                   numbers_spelled: bool = False, asr_source: bool = True,
                   genre: str = "documentary") -> str:
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
    extra = (f"If the source lists several items, give each a distinct {tgt} word "
             f"do not write the same word twice in a row or in a list.")
    retry = _run(tokenizer, model,
                 _translate_instruction(src, source, target, context, extra=extra,
                                        preceding=preceding, register=register,
                                        names=names, numbers_spelled=numbers_spelled,
                                        asr_source=asr_source, genre=genre),
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
    direct hop leaked them) a nominative cardinal beats a digit-with-suffix
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
    For caseless targets it returns the empty set no signal, so the shorten
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
    if script_for(target) == "cjk":            # no spaces substring is the word test
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
    # (On a pivot shorten the source here is the English intermediate Latin
    # script, so the note passes through whole; see `relevant_context`.)
    context = relevant_context(context, source_text, source)
    hint = f"{context.strip()}\n\n" if context and context.strip() else ""
    before = (f"The line spoken just before was: {preceding.strip()}\n"
              f"It is background only do not translate it and do not let any of its "
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


# Revision batching: ~25 lines per model call keeps the whole batch well inside
# one context and one parseable answer; 2 lines of already-revised overlap give
# each batch the continuity a lone opening line lacks.
REVISE_BATCH = 25
REVISE_OVERLAP = 2


def _parse_numbered(out: str, k: int) -> list[str] | None:
    """The k numbered lines of a revision reply, or None when it isn't one.

    Strict on the only thing that matters the markers 1..k must all be found,
    in order, each claiming the text up to the next but tolerant of the shapes
    `_strip_editorial` leaves behind (blank lines collapsed to single spaces, a
    stray preamble before marker 1). In-text digit markers cannot collide: the
    lines being revised are digit-free (`_finalize_numbers` spelled them).
    """
    marks = []
    want = 1
    for match in re.finditer(r"(?:^|[\n\s])(\d{1,3})[.)]\s", out or ""):
        if int(match.group(1)) == want:
            marks.append(match)
            want += 1
    if want != k + 1:
        return None
    lines = []
    for i, match in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(out)
        lines.append(out[match.end():end].strip())
    return lines


def revise_run(tokenizer, model, lines: list[str], *, target: str,
               names: list[str] | tuple[str, ...] = (),
               batch_size: int = REVISE_BATCH, overlap: int = REVISE_OVERLAP) -> list[str]:
    """One revision pass over the run's final target-language lines.

    Per-segment translation is deliberate (deterministic, no neighbour bleed),
    but it means no call ever sees the run whole so a name spelled three ways,
    or a line that is broken *as target-language text*, survives to TTS. This
    pass reads the finished script in batches and may fix exactly that: broken
    or unnatural target text, and proper-noun spelling drift against the run's
    own canonical names (see `canonical_names`). It is forbidden to touch
    claims, numbers, roles or content and it structurally cannot reintroduce
    much: every revised line must still pass `is_target_text`, digits are
    re-spelled, and any batch that comes back unparseable (or a dead model call)
    keeps its originals, logged. Returns a new list, same length and order.
    """
    revised = list(lines)
    if not revised:
        return revised
    tgt = _lang(target)
    ents = (" Use these canonical spellings, established by this video's own "
            "script: " + ", ".join(names) + "." if names else "")
    for start in range(0, len(revised), batch_size):
        ctx = revised[max(0, start - overlap):start]
        chunk = revised[start:start + batch_size]
        listed = ctx + chunk
        ctx_note = (f" The first {len(ctx)} lines are final context from the "
                    f"previous batch return them unchanged." if ctx else "")
        numbered = "\n".join(f"{i}. {line}" for i, line in enumerate(listed, 1))
        prompt = (
            f"You are revising the numbered lines of a dubbing script written in "
            f"{tgt}. Fix ONLY: (a) a line that is broken or unnatural as {tgt} "
            f"text; (b) inconsistent spellings of a proper noun;{ents} (c) an "
            f"obviously incoherent word that its context shows must be one of "
            f"those names. Never change claims, numbers, or who does what to "
            f"whom, and never add or remove content. Return every line under its "
            f"same number, each on its own line, changed or unchanged no notes, "
            f"no comments.{ctx_note}\n\n{numbered}"
        )
        try:
            out = _run(tokenizer, model, prompt,
                       max(512, sum(len(line) for line in listed)))
        except Exception as exc:
            # The translations already stand; a revision batch dying must not
            # take the run down with it.
            print(f"  translate: revise batch at line {start + 1} failed ({exc}) "
                  f"→ keeping originals", file=sys.stderr)
            continue
        parsed = _parse_numbered(out, len(listed))
        if parsed is None:
            print(f"  translate: revise batch at line {start + 1} unparseable "
                  f"→ keeping originals", file=sys.stderr)
            continue
        changed = 0
        for i, new in enumerate(parsed[len(ctx):]):
            new = _finalize_numbers(new.strip(), target)
            if not new or new == revised[start + i]:
                continue
            if not is_target_text(new, target):
                continue                   # untranslated / source-script relapse
            revised[start + i] = new
            changed += 1
        if changed:
            print(f"  translate: revise changed {changed} line(s) in batch at "
                  f"line {start + 1}", file=sys.stderr)
    return revised


def needs_translation(seg: dict[str, Any]) -> bool:
    """True when this stage should produce `text_en` for `seg`.

    Empty means "not translated yet" that is the per-segment resume mechanism,
    and the editor's per-segment redo (`edit.invalidate`) is the same deletion. A
    `locked` line is the user's own text and is never regenerated, however it got
    invalidated.
    """
    from . import manifest

    if manifest.is_locked(seg, "text_en"):
        return False
    return not (seg.get("text_en") or "").strip()


def needs_subtitle_translation(seg: dict[str, Any]) -> bool:
    """True when a keep segment's subtitle must be a translation, not its own text.

    Two kinds: a confident third-language keep (known language, real words), and
    a movie-mode interjection keep the original audio plays either way, but the
    subtitle should read in the target language.
    """
    if not seg.get("keep"):
        return False
    if (seg.get("text") or "").strip() in ("", "…"):
        return False
    if not needs_translation(seg):
        return False
    if seg.get("keep_reason") == "interjection":
        return True
    return (seg.get("keep_reason") == "foreign"
            and bool(seg.get("lang")) and seg["lang"] != "und")


def segment_source(seg: dict[str, Any], source: str) -> str:
    """The language a segment is spoken in: its own, else the run's.

    Thin wrapper over `segment_langs` for the callers that only need the source
    half (the same-language identity check). One rule, one implementation.
    """
    return segment_langs(seg, source, "")[0]


def _assert_translated(segments: list[dict[str, Any]]) -> None:
    """Dubbed segments must have a translation; kept segments need not (an
    "uncovered" keep is untranscribed audio that only plays as original sound).

    One dub may end without one: the line whose translation failed while the user
    had asked for it to be dubbed (`mark_failed`). This stage is not allowed to
    answer that with a keep, so it answers with the truth the segment is
    reported as unfinished, `tts` still gives it audio, and the editor shows it
    as untranslated with a retry.
    """
    from . import keep as keep_mod

    stranded = [s["id"] for s in segments
                if not s.get("keep") and not (s.get("text_en") or "").strip()
                and keep_mod.user_wants_dub(s)]
    if stranded:
        print(f"  translate: {len(stranded)} segment(s) the user asked to dub are "
              f"still untranslated: {stranded}", file=sys.stderr)
    missing = [s["id"] for s in segments
               if not s.get("keep") and not (s.get("text_en") or "").strip()
               and s["id"] not in stranded]
    assert not missing, f"dubbed segments without text_en: {missing}"


def run(m: dict[str, Any], workdir: Path, *, source: str, target: str, save=None,
        register: str = "narration", genre: str = "documentary") -> None:
    from . import manifest

    segments = m["segments"]
    # A third-language keep and a movie-mode interjection keep gets its
    # subtitle translated below; every other keep is subtitled with its own text
    # as before.
    subs = [s for s in segments if needs_subtitle_translation(s)]
    for seg in segments:
        if seg["keep"] and seg not in subs and needs_translation(seg):
            # A third-language keep whose text never got a target rendering (an
            # "und" verdict, or the translation below fails) must not put a
            # foreign-script line in the subtitles the placeholder is honest.
            # A user passthrough is the same case: the viewer is about to hear the
            # target language, so the source-language ASR's reading of it (which is
            # what made the user reach for the override) is not the subtitle. Both
            # of its names count the studio's `edit.set_keep` writes the same
            # verdict as MANUAL_REASON (see `dubbing.USER_KEEP_REASONS`).
            if ((seg.get("lang") or seg.get("keep_reason") in USER_KEEP_REASONS)
                    and not is_target_text(seg["text"], target)):
                seg["text_en"] = "…"
            else:
                seg["text_en"] = seg["text"]

    dub = [s for s in segments if not s["keep"]]
    todo = [s for s in dub if needs_translation(s)]
    if not todo and not subs:
        return

    # Same-language dubbing (he→he, en→en, …): the target line IS the source line.
    # There is nothing for a translator to do, so nothing loads Gemma the stage
    # completes on identity and stamps its fingerprint like any other run. Decided
    # per segment rather than per run, because a third-language span carries its own
    # `lang` and does need a real hop even on a same-language run (Arabic inside a
    # Hebrew video, dubbed into Hebrew).
    #
    # Nothing is kept on "already the target language" grounds: `segments.mark_keep`
    # voids its script and speaker evidence when the pair shares a script, so every
    # speech segment is re-voiced in the cloned voice. The keeps that remain are the
    # ones that never depended on the pair music, noise, spans with no text, a
    # confident third language, the user's own passthrough.
    #
    # `_finalize_numbers` still runs: digits are not speech, and spelling them is
    # code's job here exactly as it is on a translated line.
    identity = [s for s in todo + subs
                if same_language(segment_source(s, source), target)]
    for seg in identity:
        text = _finalize_numbers((seg.get("text") or "").strip(), target)
        if text:
            seg["text_en"] = text
    if identity:
        print(f"  translate: {len(identity)} segment(s) already in {_lang(target)} "
              "kept verbatim, no translation", file=sys.stderr)
        todo = [s for s in todo if needs_translation(s)]
        subs = [s for s in subs if needs_translation(s)]
    if not todo and not subs:
        manifest.save(workdir, m)
        _assert_translated(segments)
        return

    context = m["source"].get("context") or ""
    # What was said just before each segment, kept or dubbed the line the viewer
    # has already heard. Only ever shown to the model as background (see
    # `_translate_instruction`); the segment translated is still the segment alone.
    before = {s["id"]: prev["text"] for prev, s in zip(segments, segments[1:])}
    prev_of = {s["id"]: prev for prev, s in zip(segments, segments[1:])}
    mids: dict[int, str] = {}      # English intermediates produced this run
    # Rolling proper-noun lists per hop target, grown from this run's own output
    # (run-derived, never per-video config): each hop is told the spellings its
    # own language already established, so a recurring name stays one name.
    established: dict[str, list[str]] = {"en": [], target: []}
    processor, model, device = load()
    try:
        for n, seg in enumerate(dub, 1):
            if not needs_translation(seg):
                continue
            # Translate each segment on its own standalone output is deterministic
            # and faithful, which matters more for a dub than resolving a pronoun; a
            # marked multi-segment window makes the model drift a clause onto its
            # neighbour. The per-video `context` supplies names the ASR mangles, and
            # the preceding line settles a word sense the sentence alone cannot.
            # A segment off the run's own language pair decides its hops for
            # itself (`segment_langs`): a dubbable third-language span
            # (--dub-foreign) named by the witness, a pair the editor overrode, or
            # a line whose own script refutes `--src`. An English span on a he→ru
            # run goes en→ru directly, one hop. Its neighbour spoke a different
            # language, so the preceding line carries no tie-breaking signal pass
            # none rather than mislead the model.
            seg_src, seg_tgt = segment_langs(seg, source, target)
            own_pair = (seg_src, seg_tgt) != (source, target)
            seg_pivot = pivot_via_english(seg_src, seg_tgt)
            # The preceding line is shown in the language the hop WRITES the
            # previous English intermediate for a pivot's first hop, the
            # previous target output for a direct hop. Measured: garbled-name
            # reconciliation fires with an English preceding line and not with
            # a source-language one. Falls back to the previous SOURCE text for
            # the first segment and wherever the previous translation failed.
            preceding = ""
            prev_mid = ""
            if not own_pair:
                prev = prev_of.get(seg["id"])
                prior = ""
                if prev is not None:
                    if seg_pivot:
                        prev_mid = (mids.get(prev["id"])
                                    or prev.get("text_mid") or "").strip()
                        prior = prev_mid
                    else:
                        prior = prev.get("text_en") or ""
                        if not is_target_text(prior, seg_tgt):
                            prior = ""     # kept/failed neighbour: subtitle, not output
                preceding = prior.strip() or before.get(seg["id"], "")
            # Gloss clauses in the user context apply only where their word is
            # spoken; gate against the ORIGINAL source text on both hops the
            # second hop reads clean English, but the gloss's reason to exist
            # (or not) is still the source segment (see `relevant_context`).
            seg_ctx = relevant_context(context, seg["text"], seg_src)
            if seg_pivot:
                # src→en→tgt: the English hop is the measured-good line; the
                # direct pair substitutes entities and bleeds neighbours (A/B'd).
                mid = generate(processor, model, seg["text"], source=seg_src,
                               target="en", context=seg_ctx,
                               preceding=preceding, device=device,
                               register=register, genre=genre,
                               names=tuple(canonical_names(established["en"])))
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
                    # The second hop sees the PREVIOUS segment's English
                    # intermediate as preceding ("" for the first / after a
                    # failed hop). The documented entity swap ("Jabhat al-Nusra"
                    # → "al-Qaeda's Front") happened with a HEBREW preceding
                    # line; A/B'd on both harness sets, a coherent English one
                    # measured safe 0 semantic breaks on 17 controls including
                    # every entity-swap guard line and fixed «запись»→«въезд»,
                    # a dangling «её», and a chameleon incoherence.
                    text = generate(processor, model, mid, source="en",
                                    target=seg_tgt, context=seg_ctx,
                                    preceding=prev_mid,
                                    device=device, register=register, genre=genre,
                                    names=tuple(canonical_names(
                                        established.setdefault(seg_tgt, []))),
                                    numbers_spelled=True, asr_source=False)
            else:
                # A direct en→tgt hop gets the same treatment as the pivot's
                # second hop: spell the English digits in code first, then ask
                # the model only to keep the number-words as words.
                src_text = seg["text"]
                en_direct = seg_src == "en" and seg_tgt != "en"
                if en_direct:
                    src_text = numwords.spell_numbers(src_text, "en")
                text = generate(processor, model, src_text, source=seg_src,
                                target=seg_tgt, context=seg_ctx,
                                preceding=preceding, device=device,
                                register=register, genre=genre,
                                names=tuple(canonical_names(
                                    established.setdefault(seg_tgt, []))),
                                numbers_spelled=en_direct)
            # target=="en": spell the final English; otherwise: safety net over
            # any digits the model passed through.
            text = _finalize_numbers(text, seg_tgt)
            if is_target_text(text, seg_tgt):
                seg["text_en"] = text.strip()
                established[seg_tgt] = update_established_names(
                    established.setdefault(seg_tgt, []), seg["text_en"], seg_tgt)
                if seg_pivot:
                    seg["text_mid"] = mids[seg["id"]]
            elif mark_failed(seg):
                print(f"  translate: seg {seg['id']} failed → keep original",
                      file=sys.stderr)
            else:
                print(f"  translate: seg {seg['id']} failed → still untranslated "
                      f"(the user asked for this line to be dubbed)", file=sys.stderr)
            if n % 8 == 0:
                print(f"  translate: {n}/{len(dub)}", file=sys.stderr)
                if save:
                    save()
        for seg in subs:
            # Subtitle-only: the audio stays original, so a failure here just
            # leaves the span's own transcription as the subtitle. Gloss gating
            # applies here too, against the span's own text and language. An
            # interjection keep (movie mode) has no span language: it is source
            # speech kept for its actor's own voice, so it translates from the
            # run's source language unless its own script refutes that too
            # (`segment_langs`).
            seg_lang, seg_tgt = segment_langs(seg, source, target)
            seg_ctx = relevant_context(context, seg["text"], seg_lang)
            if pivot_via_english(seg_lang, seg_tgt):
                mid = generate(processor, model, seg["text"], source=seg_lang,
                               target="en", context=seg_ctx, device=device,
                               genre=genre)
                text = "" if not is_target_text(mid, "en") else generate(
                    processor, model, numwords.spell_numbers(mid.strip(), "en"),
                    source="en", target=seg_tgt, context=seg_ctx, device=device,
                    numbers_spelled=True, asr_source=False, genre=genre)
            else:
                text = generate(processor, model, seg["text"], source=seg_lang,
                                target=seg_tgt, context=seg_ctx, device=device,
                                genre=genre)
            text = _finalize_numbers(text, seg_tgt)
            seg["text_en"] = text.strip() if is_target_text(text, seg_tgt) else "…"
        # Revision pass over the finished dubbing script (dubbed lines only —
        # kept segments' text_en is a subtitle of audio that will play as-is).
        # The entity table is canonicalised from the script's own proper-noun
        # occurrences, so a name the run spelled three ways converges on its
        # best-attested form. Runs only when this call translated something:
        # a resumed no-op run must not re-revise an already-revised script.
        # A hand-corrected line is excluded outright: this pass rewrites the whole
        # script whenever anything was translated, which is exactly the path that
        # would silently undo a user's correction.
        rev = [s for s in segments
               if not s.get("keep") and (s.get("text_en") or "").strip()
               and not manifest.is_locked(s, "text_en")]
        if todo and rev:
            table = canonical_names(
                [n for s in rev for n in _name_occurrences(s["text_en"], target)])
            for s, text in zip(rev, revise_run(processor, model,
                                               [s["text_en"] for s in rev],
                                               target=target, names=table)):
                s["text_en"] = text.strip()
    finally:
        free(model)
    manifest.save(workdir, m)
    _assert_translated(segments)


def free(model) -> None:
    import gc

    if isinstance(model, WorkerHandle):
        if model.own_gpu and model._proc.poll() is None:
            return  # resident on its own GPU stays hot for the next stage
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
