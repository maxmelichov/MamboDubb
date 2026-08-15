# Plan: Linux/CUDA port, 10-min bug-hunt run, and multilingual dubbing

Persistent tracker each phase is long-running. Update the checkboxes as work lands.
Owner: Claude + maxm. Started 2026-08-07.

## Goal

1. **Phase 0** pipeline runs on this Linux box (2× RTX 5090) *and* still on the Mac (MLX).
2. **Phase 1** dub first 10 min of `sYd4zgR7f6w` (he→en, `--duration 600`), hunt bugs, fix general causes.
3. **Phase 2** Hebrew → any Qwen3-TTS-supported target (not just English).
4. **Phase 3** any source → any target supported by Gemma + Qwen3-TTS (en→ru, ru→en, …).

## Phase 0 Linux/CUDA environment  `[in progress]`

- [x] Restore `third_party/Qwen3-TTS` (upstream `QwenLM/Qwen3-TTS`, pinned commit `022e286b` the
      gitlink was committed without `.gitmodules`; now cloned in place. TODO: add `.gitmodules` entry).
- [x] `pyproject.toml`: `mlx-lm` marked `sys_platform == 'darwin'` (Linux venv skips MLX).
- [x] `uv sync` clean; torch 2.13 cu130 sees both GPUs; 71/71 unit tests pass.
- [x] Models under `models/`:
  - [x] `ivrit-ai/whisper-large-v3-turbo-ct2` → `models/whisper-large-v3-turbo-ct2` (Hebrew ASR)
  - [x] `Systran/faster-whisper-base.en` → `models/faster-whisper-base.en` (**required** no hub fallback in `transcript.py:231`)
  - [x] `speechbrain/lang-id-voxlingua107-ecapa` → `models/lang-id-voxlingua107-ecapa` (**required** for LID; `load_lid()` returns None without it)
  - [x] `Qwen/Qwen3-TTS-12Hz-1.7B-Base` → `models/Qwen3-TTS-12Hz-1.7B-Base`
  - [x] Gemma 4 12B weights for CUDA (transformers format; MLX quant is Mac-only)
- [x] **CUDA translator backend.** `translate.py` is MLX-only, and the main venv pins
      `transformers==4.57.3` (qwen-tts) which predates Gemma 4. Design: an **isolated worker venv**
      (`translator/` subproject: transformers ≥5, torch CUDA) running Gemma 4 12B bf16 on one 5090,
      spoken to over a JSON-lines subprocess protocol. `translate.load()` picks backend:
      darwin → mlx-lm (unchanged), linux → subprocess worker. Same prompts, greedy decoding.
      *Landed:* `translator/{pyproject.toml,worker.py}` + `translate.WorkerHandle`; worker applies
      the chat template, `translate.py` keeps all prompt construction and post-processing shared.
- [x] CUDA hygiene (small, non-blocking): `tts.free()` + `translate.free()` add
      `torch.cuda.empty_cache()`; faster-whisper try CUDA (`float16`) with CPU fallback in
      `transcript.py` and `tts._load_asr`; lift `VERIFY_CPU_THREADS=2` cap off-Mac.
- [ ] `.env` with `HF_TOKEN` (Pyannote diarization is gated; without it → single-speaker fallback).
      **Needs user's token.**
- [ ] Docs: Linux setup section in `docs/setup.md` (+ fix stale `qat-4bit` → `6bit` quant name),
      device notes in AGENTS.md.

## Phase 1 10-minute he→en bug-hunt run  `[blocked on Phase 0]`

Target video: `sYd4zgR7f6w` (Enemies s5e5, Khamenei) **premieres in ~25h not downloadable yet**.
Using the newest published episode from the same playlist instead: `nNQhzD-T85M`
(s5e4, Ahmed al-Sharaa / al-Julani, 55 min). e1 (`Z2Uq4aX2KPs`, Qatar) was the tuning video.

- [x] `uv run python -m dubbing "https://www.youtube.com/watch?v=nNQhzD-T85M" --duration 600` (94/96 dubbed, all verified, max drift 0.36s)
- [x] Review `report.json` clean; the 35s uncovered span is archival Arabic + music (by design).
- [ ] Watch `preview.mp4`; note mistranslations, timing drift, speaker-clone quality, artifacts.
- [ ] For each bug: find the *general* cause in `dubbing/` (invariant 7: no per-video rules),
      add a unit test, bump the stage tag, re-run.
- [x] Findings log: `docs/bughunt-nNQhzD-T85M.md` (46 flagged; 2 general fixes landed in translate/v14; 4 experiments queued).

## Phase 2 Hebrew → any Qwen3-TTS target  `[done 2026-08-07]`

All blockers are target-side. The unifying fix: a new `dubbing/script.py` with a Unicode-block
table exposing `is_target_script(text, lang)` / `script_ratio(text, lang)`, replacing every
"Latin == target" heuristic. Then:

- [x] `translate.is_target_text` → script-aware per target (currently rejects any non-Latin output
      → every ru/ar/he target segment would flip to `mt_failed`).
- [x] `tts.prepare_en` → script-aware sanitizer (currently **deletes all non-Latin text** a
      Russian line becomes empty). Rename `prepare_text`.
- [x] `tts._tokens` / `word_overlap` → Unicode word regex (`\w`), no `.en` assumptions
      (currently every non-Latin clip fails verification → silent all-keep).
- [x] `tts._verify`: ASR `language="en"` → `language=tgt`; multilingual faster-whisper model
      (`.en` models can't do it); "source-script leak" check derived from src/tgt scripts.
- [x] `tts` synth call: `language="English"` → Qwen's name for tgt (`"Auto"` +
      `x_vector_only_mode=True` is the documented cross-lingual mode; validate per language against
      `model.get_supported_languages()`).
- [x] Thread `tgt_lang` into `tts.Engine` + add `tgt` to the `tts` stage params in `cli.py`
      (also fix latent bug: `segments` params `{}` but reads `tgt_lang` changing `--tgt`
      doesn't invalidate segments today).
- [x] `transcript.recover_gaps:699` + `foreign_spans:747`: `latin_ratio > 0.5` → target-script test.
- [x] `segments.mark_keep` + `script_of`/`latin_ratio`: two-script (Latin/Hebrew) world →
      script table; `keep_reason "latin"` → `"target_lang"`.
- [x] `translate` English-only guards degrade gracefully per target: `_NEGATIONS`, `_REPEAT_SKIP`,
      `_adjacent_repeat`, `_proper_nouns` (caseless scripts), contractions instruction only for en.
- [x] Per-language speech-rate constants (`clone_length_ok` 3 words/s, `max_new_tokens`,
      `CLONE_*_SEC_PER_WORD`, timeline shorten word budgets).
- [x] `mix.py`: SRT `language=eng` + `preview_en.srt` filename from tgt.
- [x] Manifest: keep key `text_en` as the opaque "target text" field (renaming = 27 sites +
      invalidates every manifest on disk); fix docstrings only.
- [x] Bump `STAGE_TAGS`: transcript, segments, translate, tts, mix. Update tests
      (`is_target_text`, `prepare_en("שלום")==""`, `keep_reason=="latin"` suites become
      target-parameterized).
- [x] Validate: he→ru clean (19/20, drift 0.03s), he→es clean (15/17, drift 0.23s).

## Phase 3 any source → any target  `[landed 2026-08-07]`

- [x] Source ASR table in `transcript.py`: `he` → ivrit-ai turbo (current), everything else →
      vanilla `large-v3-turbo` CT2 with `language=src`.
- [x] (noted in code, shared for now) Re-validate `FOREIGN_SRC_LOGPROB` / `SPAN_TAIL_LOGPROB` thresholds (calibrated on the
      Hebrew model) against the vanilla model.
- [x] `segments.py:616` lost-words filter `[^A-Za-z֐-׿]` → Unicode letters.
- [x] `fetch._lang_prefs` alias table mirrors `transcript._LID_ALIAS` (iw/in/ji).
- [x] **Same-script pairs** (en→es, ru→uk): script heuristics can't separate src from tgt —
      target-span detection must ride on VoxLingua LID (`detect_spoken_target_spans` already does);
      gate `foreign_spans`/`recover_gaps` script shortcuts to different-script pairs only.
- [x] Validate matrix: en→ru 21/22 ✓, ru→en 15/16 ✓, en→es 19/22 drift 0.95s ✓ (after three
      general fixes it exposed: source-echo/reasoning-dump detector in translate with one retry;
      TTS clip-vs-slot bound killing runaway clips; manifest reset now reclaims downstream
      tts_failed keeps so re-translation actually happens). he→en regression rerun pending.

## Language matrix (what "supported" means)

- **Gemma 4 12B** translates broadly; `translate._LANG_NAMES` currently he,en,ar,fr,es,de,ru,it,pt,tr,fa extend as needed.
- **Qwen3-TTS** supported set is read from the checkpoint at runtime (`model.get_supported_languages()`), not a static list; `language="Auto"` + x-vector cloning is the documented multilingual path.
- **faster-whisper large-v3** covers ~100 languages for ASR/verification.
- Practical first wave: he/en/ru/es/fr/de/ar/pt/it/tr.
