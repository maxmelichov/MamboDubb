# Bug hunt — first 10 min of nNQhzD-T85M (Enemies s5e4, al-Sharaa/al-Julani), he→en

Run of 2026-08-07 on the Linux box. 96 segments: 94 dubbed, 2 kept
(1 uncovered, 1 latin). All 94 clips passed TTS verification; max drift 0.36s,
no shortening needed. A full manual review of every segment flagged 46.

## Not bugs

- 35.5s "uncovered audible" span at 211.9–247.4s is archival Arabic chanting +
  `[מוזיקה]` captions — playing original audio there is the designed behavior.
  Same for the other 10 short uncovered spans (music stings, intro).
- Seg 22 ("Ja.") correctly kept as `latin`.

## Fixed in this round (general causes, translate/v14)

1. **Thinking-channel leak spoken aloud** (segs 25, 72 — a literal line "thought"
   before the translation). Root cause: the CUDA worker decoded with
   `skip_special_tokens=True`, deleting Gemma's `<|channel>…<channel|>` markers so
   the shared channel-stripping in `translate._run` never saw them. Worker now keeps
   special tokens; `_strip_editorial` additionally drops a bare channel-label first
   line as a net. Tests added.
2. **Stray leading punctuation** (seg 0 "/In the year 2013…", seg 42 ".Be careful").
   `_strip_editorial` now strips leading marks a spoken line never opens with
   (quotes/parens/digits stay). Tests added.

## Open findings (need design/eval, not a blind patch)

Categorized over the 94 dubbed segments: 26 ASR-garble casualties, 13 genuine
mistranslations, 2 awkward-but-correct. Full per-segment list in the review
transcript; highlights:

- **ASR garbles dominate** (26 segs). The ivrit-ai whisper mangles names and rare
  words: Jabhat al-Nusra came out five different ways (Jabtala Nosra / Jabtonosra /
  Jabtunosa / Jabal Talusra / Jebel Taras), "חגורת הנפץ" (explosive belt) → "חגורת
  הנפש" → "safety belt" (reverses the scene), "זיקית" (chameleon) → the invented
  person "Zakit", "עלווית" (Alawite) → "he is exhausted in the evening", "בפעם
  הראשונה" → "של הראשונה".
- **One-segment-blind mistranslations** (13 segs). Lost subject across a split
  (50, 57, 87, 94), idioms taken literally ("עובד עלינו" → "working against us"
  instead of "fooling us", 63), homophone traps ("בורגנים" → "burglars" for
  bourgeois, 85; "הסלפיה הג'יהאדיסטית" → "the jihadist selfie", 68; "סורית" →
  "Sorry.", 78).
- **Entity romanization inconsistency**: Julani / Julni / Al-Jolani across the run.

### Proposed next experiments (in order)

1. **Noisy-ASR instruction**: add one prompt sentence — the source is ASR output;
   repair obvious mistranscriptions from context; never transliterate a word that
   does not parse. Must be evaluated against the Qatar 9/9 fidelity set before
   landing (the current prompt was tuned measurably; see translate.py docstring).
2. **Run-level entity consistency**: cluster fuzzy-matched proper nouns across the
   run's translations and normalize to the majority romanization (uses only the
   video's own output — no per-video glossary, invariant 7 safe).
3. **Suspicious-source detector**: adjacent duplicated tokens, mixed-script words
   (Arabic chars inside a Hebrew line, seg 21), mid-word truncations — route those
   segments through a context-heavier retry rather than translating the garble
   literally.
4. Consider whether the ASR itself can do better: large-v3-turbo beam width /
   temperature settings, or biasing via `--context` (already supported, unused in
   this run: `--context "Documentary about Ahmed al-Sharaa (al-Julani), Jabhat
   al-Nusra, Syria…"` would likely fix half the name garbles — worth an A/B).

## Round 2 — bugs exposed by the multilingual validation matrix (2026-08-07)

All three found by the en→es same-script run and fixed generally:

3. **Same-script blindness of the translation gate** (`translate/v16-v17`). For en→es the
   script-based `is_target_text` cannot tell untranslated output from translation. Gemma
   twice mishandled one segment: first echoing the English source in a markdown wrapper,
   then dumping its full reasoning notes ("Source:** … Translation Strategy:** … Final
   Version:** …", 262 tokens for a 26-token source, truncated). New `_not_a_translation`
   detector (source-token echo >65%, markdown structure, >4x token blow-up) with one
   "never explain, plain translation only" retry; a second failure keeps original audio.
4. **Runaway TTS clip destroyed the timeline** (`tts/v9`). The reasoning dump synthesized a
   71.5s clip for a 5.7s slot, passed soft-verify, and pushed every later segment up to
   49.1s late. New `clip_exceeds_slot` bound (3x slot + 8s) rejects such clips at verify
   time regardless of overlap; the segment falls back to original audio (never-silent).
5. **Stale text after downstream failure** (`manifest.reset_stage`). A segment kept by
   `tts_failed` still held the translation that failed; a translate re-run skipped it as
   "kept" (the flip is only undone when the *tts* stage later resets), so the same bad
   text went back to TTS forever. `reset_stage("translate")` now also un-flips downstream
   failure keeps and drops their text.

## Round 3 — user viewing feedback on the 10-min preview (2026-08-07)

1. **1:05 — Hebrew played undubbed** (`transcript/v30`). The 61.4–64.3s gap span sits
   under a music sting: VoxLingua answered `my`@0.34 (no opinion) and the Hebrew ASR
   hallucinated "תודה רבה" ×4 at logprob −0.55, just under FOREIGN_SRC_LOGPROB — so the
   span was kept as unnamed foreign. Judged with ±2s context the same audio is he@0.95
   with a clean read. Fix: an "und" verdict on a span < 6s is re-judged once with a
   widened window; a confidently *named* language (the real Arabic, ar@0.96) never
   retries. General rule: the least trustworthy verdict (unnamed + short) gets a second
   opinion before costing a dub.
2. **2:15 — "Modi'in" mistranslation**: ASR-garble class (the fresh transcript hears
   "בואו נביא מסמרודים", clearly garbled; מודיעין is also the intelligence/Modi'in
   homonym). Tracked with the round-1 ASR experiments (--context A/B is the next lever).
3. **3:20 — Arabic archival speech** (feature, `transcript/v30 + segments/v24 +
   translate/v18`). Confirmed third-language spans (ar@0.96) used to carry no text at
   all — the SRT showed "…" while the source show burns its own subtitles there. Now:
   the vanilla multilingual turbo transcribes the span (told the confirmed language),
   the segment carries `lang`, and the translate stage renders a target-language
   subtitle for it. Audio stays original (user's choice: keep the voice, subtitle on
   top). A failure at any step falls back to the old "…" behavior.

## Round 4 — hallucination defenses (2026-08-07, transcript/v33 translate/v19 mix/v6)

The 1:05 area exposed a family of whisper-hallucination bugs, each now guarded:
- `drop_stretched_words` (main pass): a word stretched over >2.5s with later words
  starting inside its span is a decode artifact — confidence does not catch it (the
  invented filler scored up to 0.97). The stretched word and the inventions buried
  well inside its span are dropped; the genuine re-read at the span's end survives.
- `GAP_MIN_LOGPROB` (gap recovery): a gap read below −0.5 is a hallucination (music
  sting), not recovered speech; the window stays uncovered and plays original audio.
- Span-ASR floor (−0.8) in `_read_foreign_span`: chanting made the multilingual model
  invent "اشتركوا في القناة" ("subscribe to the channel"); low-confidence span reads
  fall back to the placeholder.
- Untranslatable foreign keeps put "…" in text_en (never raw foreign script), and
  `write_srt` skips placeholder lines entirely.
Residue: a 0.5s orphan "לא"→"No" dub at 61.2s (the artifact's leading word predates
the stretched word, so the dropper keeps it) — cosmetic, tracked.

## Round 5 — phantom credit line in ru→en (2026-08-08, transcript/v34)

The Arzamas dub opened with spoken English "Subtitles by DimaTorzok" — never said in
the audio (it is the intro jingle; an isolated re-read returns only "МУЗЫКАЛЬНАЯ
ЗАСТАВКА"). Whisper injects memorised subtitler credits/outro thanks over music.
New `drop_stock_phrases` in the main pass removes the known stereotyped family
(Субтитры/DimaTorzok/subtitles by/Amara/thanks for watching/подпишись на канал/
اشتركوا في القناة…) with exact char-span matching, taking the contiguous invented
run with it. A model-artifact rule, not per-video content: these phrases are never
legitimate dialogue in the stereotyped forms matched. Verified live on the ru→en run.

## Round 6 — Russian quality + alignment, judged by agents over the full 10-min run (2026-08-08)

**Direct he→ru loses to pivot he→en→ru, 38–7 (39 ties) over 84 matched segments.**
Direct's failures are disqualifying for a dub: entity substitution ("ИГИЛ" for Jabhat
al-Nusra), hallucinated speaker labels spoken aloud, neighbor-segment bleed
(lines spoken twice), silent clause drops, name mangling ("аль-Арби" for aш-Шараа).
Pivot's errors are ~90% inherited from the English line — so he→en fixes propagate to
every target — with only ~4% genuine double-translation losses (entry/запись-въезд).
→ Decision: non-English targets translate via the English pivot (translate/v20, with
`text_mid` stored on the segment; shorten operates on the English intermediate).

**Alignment "not 100%" root cause: fill-forward stretching.** The fitter slowed 54/88
clips (most to the 0.82 floor) to fill toward the NEXT segment's start: 46/88 lines
ended >0.5s after the original speaker stopped (21 over 1s, worst +3.99s), and two
clips ran across a speaker change into the next speaker's opening. All invariants
held — the policy, not the mechanics, was wrong. → timeline/v8 anchors each clip to
its segment's OWN end: no slowing past own-end, tail spill capped (0.6s same-speaker,
none across a speaker change), drift-of-next preferred over cross-speaker overhang.

**Post-pivot review (88 rows, agent-judged):** ~74% ship-quality, 23 severe rows — 11
from ASR garbles, 12 from the en-stage (idioms: בלאגן/מודיעין/עובד עלינו/תמות נפשי;
lexical: זיקית→lizard, בורגנים→burglars), **0 from the ru-stage**. Pivot's marginal
cost is register noise only (ты/вы instability, EN-pronoun gender drift, calques).
Next levers, all upstream: --context A/B (running), idiom handling in he→en, name
romanization consistency, digit/percent normalization for TTS.

**--context A/B (2026-08-08):** context reliably repairs NAMES it lists (Algoluni→
al-Julani; the השער garble → "Al-Sharaa … revolting against his family") but does not
help non-name garbles (מסמרודים→"the monsters", בורגנים still wrong, זיקית name-ified).
Practice: pass --context with the episode's people/orgs on every run (it is run input,
not code — invariant 7 untouched). The idiom/lexical class needs the queued noisy-ASR
prompt experiment, to be A/B'd against the Qatar 9/9 fidelity set before landing.

## Round 7 — muted speaker at 1:25 in the ru dub (2026-08-08, mix/v7)

78.5–89.1s is an English-speaking interviewee — a THIRD language in a he→ru run.
The foreign-span detector was vetoed by a Hebrew fragment at the window's edge
(the source ASR read "מלחמת האזרחים?" at −0.38, above the fail bar), so no segment
existed, and unsegmented time played only the background stem: the speaker's voice
was surgically removed. Structural gap in invariant 1 — it covered segments, not
unclaimed time. Fix at the mix: all time not claimed by a placed clip now carries
the original VOCALS stem (unity gain, edge fades), so any detection failure
degrades to "original audio plays", never to a muted speaker. (Vocals used as
audio, not as an oracle — invariant 3 intact.) Follow-up candidate: widen/step the
_sounds_foreign judgement window so an edge fragment cannot veto a long passage.

## Round 8 — the ASR-garble frontier: two rejected fixes (2026-08-08)

The dominant remaining error class (garbled Hebrew → fluent hallucinated English,
~20/112 severe on the HaMezach drama) resisted both candidate mechanisms:
1. **Word-confidence gating: REJECTED by calibration.** Whisper mishears
   phonetically-plausible words at full confidence — the worst garbles score
   p 0.94–0.998 ("ציפור על חודרנית" 0.941) while clean rows dip lower (0.871).
   No usable threshold exists.
2. **"Repair the plausible phrase" prompt instruction: REJECTED by A/B** (112 rows,
   judged): 0/11 known garbles repaired; on garbled lines the model produced more
   confident fabrications ("Shlomo from the Paratroopers" for an unread מ"פ);
   one clean-line regression (platoon→company). The instruction license is spent
   on confabulation, not recovery.
Still-open candidates for this class: ensemble ASR (ivrit + vanilla turbo
reconciliation), surfacing ASR n-best alternatives to the translator, richer
--context. Nothing ships until one of them A/Bs clean.

## Round 9 — 6:36 "female voice": verification cannot fix a reference problem (2026-08-08)

Speaker-similarity verification was calibrated on 147 clip/ref pairs and REJECTED:
Qwen's own encoder is saturated (cross-speaker 0.93+); WeSpeaker separates speakers
cleanly but scores the reported clip mid-population — because the clip faithfully
matches its REFERENCE. The reference window itself (vocals 391.9–396.4s) measures
F0 177 Hz while the male voice lives in the adjacent span; the defect is in
reference selection, not synthesis. Candidate fix (unimplemented): flag a ref
window that embeds far from the speaker's other windows / straddles a diarization
boundary, and fall back to the canonical speaker ref. Pending: user listening to
the original span to establish ground truth (female voice present in source?).

Also this round: hop-2 `preceding` REMOVED from the pivot (translate/v23) — it
deterministically caused an entity swap ("Jabhat al-Nusra" → "al-Qaeda's Front");
clean English needs no disambiguation. Verified fixed in rebuild. Numbers-as-words,
military-terms note (מודיעין, חגורת נפץ), and the recovered al-Julani all verified
in the rebuilt ru run.

## Round 10 — numbers, register, drama profile (2026-08-08)

- **Model-side digit conversion REVERTED, num2words shipped** (translate/v24,
  dubbing/numwords.py). The v22 prompt approach let the model invent readings
  ("ב-504" → mid "five zero four" → ru "в две тысячи четвёртом году", the year
  2004). Now: hop 1 outputs digits, code spells them deterministically in English
  (years as years), the target hop only inflects existing words, and a target-side
  net spells any surviving digit. Viewer-proposed, measured, adopted.
- **Drama profile validated**: HaMezach rebuilt with --register dialogue + terms
  note scored ~75% vs ~60% (same rubric) — two-thirds of the drama gap was
  settings, now a per-run flag. Echelon ladder (מחלקה platoon, גדוד battalion…)
  added after the re-review caught unit-rank drift. The remaining ~25% is the ASR
  garble class, row-for-row unchanged — next candidate: a Hebrew-coherence gate
  (garble is incoherent HEBREW; different signal than the failed confidence gate).
- Pivot hop-2 `preceding` removal verified (Jabhat al-Nusra restored).

## Round 11 — fluency license, dub-foreign live, LID override (2026-08-08)

- **Guarded fluency instruction SHIPPED** (translate/v25) for the clean-English→target
  hop only. A/B round 1 (unguarded): 33 wins but coined fake proper names. Round 2
  (guards: no coined names, keep repetition/wordplay, syntax-not-semantics, literal on
  garble): 13 wins, 0 severe regressions, one base fidelity error repaired → ship.
- **Foreign-span detection hardened** (transcript/v37): long spans judged by interior
  (edge bleed can't veto) and a ≥0.95 LID label outranks the ASR read (the Hebrew
  fine-tune transliterates clear English at logprob above the fail bar). The 78.5–89s
  English interviewee is now detected, and with --dub-foreign speaks Russian.
- Ensemble-ASR probe on the בלאגן span: BOTH models mishear it (acoustic, music bed) —
  ensemble voting won't fix that sub-class; noted before anyone builds it.

## Round 12 — drama run audit (yt_keRQsy-rWxI) + parallel fixes (2026-08-08)

Agent audits of the drama run (119 segs, 8 speakers) and the ru run answered the user's
"drama settings or bugs?" question: **bugs**. Register handling is fine; the damage is
garble→hallucination (~14 severe), timing overload (69% clips over slot, 12 cross-speaker
talk-overs), and Q+A merged into one voice (8+ cases).

Shipped (agents, file-owned, tags pre-bumped):
- segments/v27 — split segments at diarization turn boundaries (one voice never speaks two
  characters); MIN_SEG_SEC=0.9 stubs merge into same-speaker substantial neighbors only
  (kills sub-second tts_failed keeps). 6 new tests.
- timeline/v9 — cross-speaker wall: clip must end before next speaker's source onset;
  LEAD_MAX=0.6 earlier start into free timeline, then compression to RATE_MAX; residual
  recorded as "overrun" in placement (never truncated/overlapped). Fixes the 1.15-rate
  give-up branch that fueled the 06:17–06:35 cascade. 6 new tests. Suite: 227 passed.

Queued (need GPU A/B, after ru rebuild):
- garble-aware translation contract (both audits' #1: "bird on the penetrator",
  "Джабхат аль-Нам") — phonetically-nearest reading, never invent.
- length-budgeted translation (slot duration in prompt; retranslate-shorter on overflow).
- ≥3s clone reference minimum + reference-window validation (busiest drama refs were 0.6–1.1s).

## Round 13 — gloss gating + garble-name note (2026-08-08)

Full-manifest ru audit (92 segs): 65% acceptable, 17% real meaning errors, 7 nonsense segs.
NEW regression found: global --context glosses contaminated unrelated segments (9:12
בורגנים→«Район хаоса», 9:14 חיה שמה→«зикит»). Fixed: translate/v26 `relevant_context`
gates gloss clauses (source-script tokens, edit-distance ≤1) per segment, both pivot hops,
gated against the ORIGINAL source text. 8 tests.

Garble-aware translation, probed honestly:
- Bare "may contain mishearings" instruction: NO-SHIP as general fix — 1/5 garbles fixed,
  pill case worse, Gemma can't do Hebrew phonetics on demand.
- WITH context entities: SHIP — reconciles garbled proper nouns to context names
  (חוקרי השאר→"al-Sharaa's researchers", השער→al-Sharaa, Nusra name consolidated;
  clean controls unchanged). Shipped as `asr_source` note in _translate_instruction,
  emitted only when context is non-empty and the hop reads raw ASR (hop 2 passes
  asr_source=False). 2 tests; suite 237.

User reports on drama (yt_keRQsy-rWxI): 0:56 "Amala" (garble class), 0:59 Q+A merge
(segments/v27 targets it), 3:40–6:00 timing+translation pile-up (blessing scene garbles,
sub-second TTS failures, cascade). Re-running drama (register=dialogue + context) then ru,
sequentially, with all of today's fixes: bcc8idy9l.

### Round 13 verification (both reruns)

ru: gloss gating confirmed — zero contamination, zero over-correction; Nusra name fixed;
зикит/хаос fixed. Garble-name note under-fires in-pipeline: השער→"gate", חוקרי השאר→"rest"
persist. Probes show the fix fires with an ENGLISH preceding line but not with the Hebrew
preceding the pipeline passes (and השאר is coherent Hebrew, so no garble signal at all).
Next experiment: A/B hop-1 preceding = previous segment's English translation.
New pivot errors: בורגנים→"burgers"→«Район бургеров»; עובד עלינו→«Работает ли он НА нас? Да.»
(inversion, worst error in file — idiom class).

drama: blessing + toenail scenes fixed by context glosses (even over broken ASR);
8:56 and 6:41 Q+A now split. 5 Q+A merges REMAIN — diarization yields no boundary inside
fast/whispered exchanges, splitter has nothing to cut (upstream diarization density, not
splitter logic). Cross-speaker overruns 16→5 (worst 1.19→0.56s), max drift 1.23→0.60s,
placed-vs-placed overlaps zero. tts_failed 4→5: the 6:41 split's 0.44s "Nachum." answer
failed synthesis (structural: sub-second TTS coin flip, no same-speaker neighbor to merge).

## Round 14 — the unified fix (designed from all three audits, 2026-08-08)

All three runs (he→en doc, he→ru doc, he→en drama) share the same 5 severe classes:
garbled names translated literally (biggest), name-spelling drift, literal idioms,
dropped datives (meaning-inverting, fluent-sounding), dangling fragments. Decision:
ONE change — two-pass translate-then-revise in translate.py:
  P1: as today + preceding = previous ENGLISH (not Hebrew) + auto-extracted entity
      table on every call + source-side dative/idiom instruction.
  P2: document-level revision of the numbered target draft (batched), conservative
      contract (fix broken-as-target-language + entity consistency only, never claims),
      same entity table.
  Riders: no-source-script-in-output guard; phonetic-Arabic segments must not be
      translated as Hebrew (lang-ID guard).
Entity table is COMPUTED from the run's own transcript/translations (general code,
not a per-video glossary). Validation = the two prompt-agents' problem/control sets
(staged, waiting on GPUs occupied by user training jobs).

## Round 15 — the muted target-language speaker, made visible (transcript/v39 segments/v34)

The user's top complaint ("he is already speaking English and he got muted") is one
class with three coinciding conditions (Rounds 3, 7, 11): no transcript words, no
span, no uncovered fill. mix/v7 made it audible; it was still invisible — no segment,
no subtitle, nothing the editor app can select or correct. Fixed at the two stages
that decide, all four conditions reproduced first as failing unit tests.

1. **The source ASR is only a negative witness.** A source-language fine-tune does not
   fail on target speech, it TRANSLITERATES it and reads its own transliteration at
   -0.38 — above `FOREIGN_SRC_LOGPROB`. `_judge_span` believed that whenever LID was
   under the 0.95 override. New `_reads_as_target`: the target-language ASR reads the
   same clip and must clear all three of an absolute floor (`TARGET_READ_LOGPROB`
   -0.35), a margin over the source model's read of that clip (`TARGET_READ_MARGIN`
   0.10 — whoever reads it better is right; a tie is no evidence) and ≥3 real words
   that are not an `_ASR_STOCK` hallucination. It is not asked where LID already named
   a different language at ≥0.85 (a target-forced decoder always returns target text,
   so it cannot rename the Arabic quote).
2. **A target verdict now routes into the TARGET branch** of
   `detect_spoken_target_spans`, so the passage gets subtitle text, edge refinement,
   `lang == target` and a `latin`/`target_lang` keep segment — visible, selectable,
   correctable — instead of an unnamed `foreign` keep with "…" or nothing at all.
3. **`fill_uncovered_audible` judged a whole gap by its first 4s.** A gap opening on a
   music sting and then carrying an English speaker was answered "not target" once and
   filled with nothing. It now walks the gap in classifier-sized windows and keeps each
   target-language run inside it, reason `spoken_target` (a named verdict, distinct
   from an unreadable stretch). Unjudged stretches are still left alone — a keep there
   would play the mix over the bed that already carries it, and mix/v7 is the floor.
4. **`speaker_en` over-fire.** It was a speaker-level rule with no appeal: one line over
   `SPEAKER_EN_RATIO` and every segment of that speaker was kept, including their real
   source-language lines, which then never got dubbed. `mark_keep(seg_lang=…)` lets a
   per-segment verdict of "source language" at ≥ `SPEAKER_EN_VETO_PROB` (0.85, clear of
   the 0.34-0.60 band where every documented mislabel sits) outvote the prior. No
   witness or an unsure one changes nothing, so the prior still does its original job.

Not re-proposed: word-confidence gating and "repair the phrase" prompting (Round 8,
both rejected by measurement). Nothing here is per-video: two model witnesses, a
window scan, and a confidence bar.

### Needs real-audio measurement (thresholds are argued, not yet calibrated)

- **The he→en doc run, 78.5–89.1s English interviewee, and the ru run's same span.**
  Expect: a span with `lang == "en"`, real subtitle text, a keep segment with reason
  `latin`; previously only the mix's vocals fill covered it. Log line to look for:
  `transcript: en-spoken 78.5-89.1s: …`.
- **`TARGET_READ_LOGPROB` / `TARGET_READ_MARGIN` calibration.** Dump, for every
  candidate span in a run, `(src_lp, tgt_lp, verdict)` and check the two populations
  separate: real target speech should sit well above -0.35 with a positive margin,
  source-language spans well below. If they overlap, raise the floor before the margin.
- **False-positive check on the archival Arabic (211.9–247.4s chanting).** It must stay
  `foreign`/uncovered — the ≥0.85 named-language guard should hold, but the base target
  model's behaviour on chanting is the thing to watch.
- **`speaker_en` veto rate.** Count segments that lost `speaker_en` in a run with a
  target-heavy speaker; each one should be a line where that speaker really speaks the
  source language. A veto on genuine target speech is the regression to look for.
