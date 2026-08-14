/** Timecodes are read against a video, so they are m:ss.d, never seconds. */
export function timecode(seconds: number, decimals = 1): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const whole = Math.floor(seconds);
  const m = Math.floor(whole / 60);
  const s = whole % 60;
  const frac = decimals > 0 ? (seconds - whole).toFixed(decimals).slice(1) : "";
  return `${m}:${String(s).padStart(2, "0")}${frac}`;
}

export function duration(seconds: number): string {
  return `${seconds.toFixed(2)}s`;
}

export function percent(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${Math.round(value * 100)}%`;
}

/**
 * `he` → `Hebrew`, for the one or two places a sentence has to name a language
 * rather than tag it. The platform ships the table; a hand-written map would be
 * eleven languages long and wrong on the twelfth.
 */
export function languageName(code: string | null | undefined): string {
  if (!code) return "the source language";
  try {
    return new Intl.DisplayNames(["en"], { type: "language" }).of(code) ?? code.toUpperCase();
  } catch {
    return code.toUpperCase();
  }
}

/** A speaker id is `SPEAKER_04`; the label that reads is `S4`. */
export function speakerLabel(speaker: string): string {
  const match = /^SPEAKER_(\d+)$/.exec(speaker);
  return match ? `S${Number(match[1])}` : speaker;
}
