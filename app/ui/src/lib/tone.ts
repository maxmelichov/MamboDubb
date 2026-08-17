/**
 * Synthetic audio for fixture mode.
 *
 * A/B playback is the one part of the editor that cannot be judged from a
 * screenshot, so the fixture ships audible clips rather than dead <audio>
 * elements: a formant-ish buzz whose pitch stands in for a speaker and whose
 * length matches the segment's real duration. It is obviously not speech, which
 * is the point nobody will mistake fixture mode for a working pipeline.
 */

const SAMPLE_RATE = 8000;
const cache = new Map<string, string>();

export type ToneSpec = { hz: number; dur: number; seed: number };

/** `fixture:tone?hz=…&dur=…&seed=…` a placeholder URL the fixture hands out. */
export function toneUrl(spec: ToneSpec): string {
  return `fixture:tone?hz=${spec.hz.toFixed(1)}&dur=${spec.dur.toFixed(3)}&seed=${spec.seed}`;
}

export function isToneUrl(url: string): boolean {
  return url.startsWith("fixture:tone?");
}

/**
 * The longest blob this will synthesize. A source track is minutes long and
 * every segment's Orig button is a window of it, so the ceiling has to clear a
 * whole run at 8 kHz that is 14 MB, and it is one blob per voice, not one
 * per line.
 */
const MAX_DUR_SEC = 900;

/**
 * Decode a tone URL into a playable blob URL, cached per spec.
 *
 * The `#t=start,end` fragment names a *window* of the file and not another
 * file, exactly as it does on the server's `source.wav#t=…`, so it is stripped
 * before the spec is read and before the cache is keyed one track, however
 * many lines take windows of it. `clipAudio` enforces the window itself.
 */
export function resolveToneUrl(url: string): string {
  const [spec] = url.split("#");
  const cached = cache.get(spec);
  if (cached) return cached;

  const params = new URLSearchParams(spec.slice("fixture:tone?".length));
  const hz = Number(params.get("hz")) || 180;
  const dur = Math.min(Number(params.get("dur")) || 1, MAX_DUR_SEC);
  const seed = Number(params.get("seed")) || 1;

  const blob = new Blob([encodeWav(render(hz, dur, seed))], { type: "audio/wav" });
  const objectUrl = URL.createObjectURL(blob);
  cache.set(spec, objectUrl);
  return objectUrl;
}

function render(hz: number, dur: number, seed: number): Int16Array {
  const n = Math.max(1, Math.round(dur * SAMPLE_RATE));
  const out = new Int16Array(n);
  let rnd = seed * 9301 + 49297;

  // Syllable-rate amplitude gating, so the clip has the cadence of a phrase
  // rather than a test tone and A/B differences are audible.
  const syllableHz = 3.4 + ((seed % 7) / 7) * 1.6;

  for (let i = 0; i < n; i += 1) {
    const t = i / SAMPLE_RATE;
    rnd = (rnd * 9301 + 49297) % 233280;
    const jitter = (rnd / 233280 - 0.5) * 0.04;
    const gate = 0.35 + 0.65 * Math.max(0, Math.sin(2 * Math.PI * syllableHz * t));
    const fade = Math.min(1, t / 0.02, (dur - t) / 0.05);
    const wave =
      Math.sin(2 * Math.PI * hz * t) * 0.6 +
      Math.sin(2 * Math.PI * hz * 2 * t) * 0.25 +
      Math.sin(2 * Math.PI * hz * 3 * t) * 0.12 +
      jitter;
    out[i] = Math.round(Math.max(-1, Math.min(1, wave * gate * fade * 0.5)) * 32767);
  }
  return out;
}

function encodeWav(samples: Int16Array): ArrayBuffer {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const write = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };

  write(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  write(8, "WAVE");
  write(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, SAMPLE_RATE, true);
  view.setUint32(28, SAMPLE_RATE * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  write(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i += 1) view.setInt16(44 + i * 2, samples[i], true);
  return buffer;
}
