/**
 * Which languages a run can be started between, in one place.
 *
 * What can be HEARD is broader than what can be SPOKEN: the ASR and the
 * translator handle every source below, but the synthesizer voices Qwen3-TTS's
 * ten languages plus Hebrew (a LoRA over the same checkpoint; the server
 * refuses a Hebrew target with the download command if the adapter is not
 * installed). Arabic is the one language on the source list and off the target
 * one, because offering it as a target would create a project whose tts stage
 * can only fail. That is why the two lists stay deliberately different.
 *
 * The import screen picks a run's pair from these; the selection panel offers
 * the same codes again as per-segment overrides. They were two hand-kept copies
 * of the same policy, and a language added to one of them was a bug in the
 * other.
 */

/** Codes in the value, names on the screen, in the order they are offered. */
export const SOURCE_LANGUAGES = [
  ["he", "Hebrew"],
  ["en", "English"],
  ["ar", "Arabic"],
  ["ru", "Russian"],
  ["fr", "French"],
  ["es", "Spanish"],
  ["de", "German"],
  ["it", "Italian"],
  ["pt", "Portuguese"],
  ["zh", "Chinese"],
  ["ja", "Japanese"],
  ["ko", "Korean"],
] as const;

export const TARGET_LANGUAGES = [
  ["en", "English"],
  ["he", "Hebrew"],
  ["ru", "Russian"],
  ["fr", "French"],
  ["es", "Spanish"],
  ["de", "German"],
  ["it", "Italian"],
  ["pt", "Portuguese"],
  ["zh", "Chinese"],
  ["ja", "Japanese"],
  ["ko", "Korean"],
] as const;

export const SOURCE_LANG_CODES: string[] = SOURCE_LANGUAGES.map(([code]) => code);
export const TARGET_LANG_CODES: string[] = TARGET_LANGUAGES.map(([code]) => code);
