//! Nemotron-3 diarization worker, compiled: wav path in, JSON turns file out.
//!
//! Drop-in for `diarizer/worker.py` (same argv, same JSON, same chunking
//! semantics), minus the NeMo venv. The model runs through ONNX Runtime via
//! parakeet-rs, CPU only: both GPUs belong to the user's training jobs.
//!
//! The ONNX file is parakeet-rs's own export (`nemotron3_diar_v3.onnx` from
//! hf.co/altunenes/parakeet-rs). The onnx-community/Nemotron-3-Diarization-ONNX
//! export cannot be used: its graph exposes `logits`/`chunk_embeds` for
//! transformers.js, while parakeet-rs needs the `preds_diar`/`preds_hires`
//! streaming-cache graph plus constants stored in ONNX metadata.
//!
//! Resampling is plain linear interpolation. The model reads 128-bin mel
//! energies at 10ms hops and emits speaker activity, so aliasing above 8 kHz
//! from a 44.1 kHz source is far below the feature resolution; rubato-grade
//! filtering would add a dependency for no measurable change in turns.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use parakeet_rs::sortformer::{Sortformer, SpeakerSegment};

const OVERLAP_SEC: f64 = 10.0;
const MODEL_SR: u32 = 16_000;
const MODEL_FILE: &str = "nemotron3_diar_v3.onnx";
const MODEL_DIR: &str = "models/nemotron-3-diarization-onnx";
const MODEL_URL: &str =
    "https://huggingface.co/altunenes/parakeet-rs/resolve/main/nemotron-3-diarization";

/// One diarized turn, worker.py's output shape.
struct Turn {
    speaker: String,
    start: f64,
    end: f64,
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

/// DUB_DIARIZER_ONNX_DIR wins; otherwise walk up from the binary looking for
/// the repo's models/ dir (the release binary sits four levels below the
/// repo root), falling back to the cwd-relative default.
fn model_path() -> PathBuf {
    if let Ok(dir) = std::env::var("DUB_DIARIZER_ONNX_DIR") {
        return Path::new(&dir).join(MODEL_FILE);
    }
    if let Ok(exe) = std::env::current_exe() {
        for dir in exe.ancestors().skip(1) {
            let candidate = dir.join(MODEL_DIR).join(MODEL_FILE);
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    Path::new(MODEL_DIR).join(MODEL_FILE)
}

/// Decode any PCM wav to f32 mono at its native rate.
fn read_wav_mono(path: &str) -> Result<(Vec<f32>, u32), String> {
    let mut reader =
        hound::WavReader::open(path).map_err(|e| format!("cannot open {path}: {e}"))?;
    let spec = reader.spec();
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<_, _>>()
            .map_err(|e| format!("bad float sample in {path}: {e}"))?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()
                .map_err(|e| format!("bad int sample in {path}: {e}"))?
        }
    };
    let ch = spec.channels as usize;
    let mono: Vec<f32> = if ch == 1 {
        interleaved
    } else {
        interleaved
            .chunks_exact(ch)
            .map(|frame| frame.iter().sum::<f32>() / ch as f32)
            .collect()
    };
    Ok((mono, spec.sample_rate))
}

/// Linear-interpolation resample (see the module docs for why this suffices).
fn resample(mono: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to {
        return mono.to_vec();
    }
    let ratio = from as f64 / to as f64;
    let out_len = ((mono.len() as f64) / ratio).floor() as usize;
    (0..out_len)
        .map(|i| {
            let pos = i as f64 * ratio;
            let idx = pos as usize;
            let frac = (pos - idx as f64) as f32;
            let a = mono[idx];
            let b = *mono.get(idx + 1).unwrap_or(&a);
            a + (b - a) * frac
        })
        .collect()
}

/// Model segments (16 kHz sample offsets) to worker.py turns, offset applied
/// before the 3-decimal rounding, exactly like worker.py's `_parse`.
fn to_turns(segments: &[SpeakerSegment], offset: f64) -> Vec<Turn> {
    segments
        .iter()
        .map(|s| Turn {
            speaker: format!("speaker_{}", s.speaker_id),
            start: round3(s.start as f64 / MODEL_SR as f64 + offset),
            end: round3(s.end as f64 / MODEL_SR as f64 + offset),
        })
        .collect()
}

/// worker.py's window starts: step = chunk - overlap, one window minimum,
/// and no window that would begin inside the last half second.
fn window_starts(total: f64, chunk_sec: f64) -> Vec<f64> {
    let step = chunk_sec - OVERLAP_SEC;
    let count = ((total / step).floor() as usize).saturating_add(1).max(1);
    (0..count)
        .map(|i| i as f64 * step)
        .enumerate()
        .filter(|&(i, t0)| t0 < total - 0.5 || i == 0)
        .map(|(_, t0)| t0)
        .collect()
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        return Err("usage: mambodubb-diarizer <input.wav> <out.json> [chunk_sec]".into());
    }
    let (wav, out) = (&args[1], &args[2]);
    let chunk_sec: f64 = match args.get(3) {
        Some(raw) => raw
            .parse()
            .map_err(|_| format!("chunk_sec not a number: {raw}"))?,
        None => 0.0,
    };
    if chunk_sec > 0.0 && chunk_sec <= OVERLAP_SEC {
        return Err(format!("chunk_sec must exceed the {OVERLAP_SEC}s overlap"));
    }

    let (mono, sr) = read_wav_mono(wav)?;
    // worker.py measures the file's length at its native rate; the ownership
    // windows below must agree with it to the sample.
    let total = mono.len() as f64 / sr as f64;
    let mono16k = resample(&mono, sr, MODEL_SR);
    drop(mono);
    eprintln!("diarizer: {total:.1}s of audio ({sr} Hz in, {MODEL_SR} Hz to the model)");

    let model = model_path();
    if !model.is_file() {
        return Err(format!(
            "model not found at {}; fetch {MODEL_FILE} from {MODEL_URL} \
             or point DUB_DIARIZER_ONNX_DIR at its directory",
            model.display()
        ));
    }
    let mut diarizer =
        Sortformer::new(&model).map_err(|e| format!("cannot load {}: {e}", model.display()))?;
    eprintln!("diarizer: model loaded from {}", model.display());

    let mut turns: Vec<Turn> = Vec::new();
    if chunk_sec <= 0.0 {
        let segments = diarizer
            .diarize(mono16k, MODEL_SR, 1)
            .map_err(|e| format!("diarization failed: {e}"))?;
        turns = to_turns(&segments, 0.0);
    } else {
        let starts = window_starts(total, chunk_sec);
        for (i, &t0) in starts.iter().enumerate() {
            let a = (t0 * MODEL_SR as f64) as usize;
            let b = (((t0 + chunk_sec) * MODEL_SR as f64) as usize).min(mono16k.len());
            // Each diarize() call resets the model's speaker slots, which is
            // the point of chunking: the 8-speaker cap applies per window.
            let segments = diarizer
                .diarize(mono16k[a..b].to_vec(), MODEL_SR, 1)
                .map_err(|e| format!("diarization failed in window {i}: {e}"))?;
            // Each window owns the region up to the midpoint of its overlap
            // with the next; a turn belongs to the window holding its start.
            let lo = if i == 0 { 0.0 } else { t0 + OVERLAP_SEC / 2.0 };
            let hi = if i == starts.len() - 1 {
                total
            } else {
                starts[i + 1] + OVERLAP_SEC / 2.0
            };
            for mut t in to_turns(&segments, t0) {
                if lo <= t.start && t.start < hi {
                    t.speaker = format!("w{i}_{}", t.speaker);
                    t.end = t.end.min(hi + OVERLAP_SEC / 2.0);
                    turns.push(t);
                }
            }
            eprintln!(
                "diarizer: window {}/{} [{t0:.0}s..) done, {} turn(s) so far",
                i + 1,
                starts.len(),
                turns.len()
            );
        }
    }

    let json: Vec<serde_json::Value> = turns
        .iter()
        .map(|t| serde_json::json!({"speaker": t.speaker, "start": t.start, "end": t.end}))
        .collect();
    let body = serde_json::to_string(&json).map_err(|e| format!("json encode failed: {e}"))?;
    std::fs::write(out, body).map_err(|e| format!("cannot write {out}: {e}"))?;

    let labels: std::collections::HashSet<&str> =
        turns.iter().map(|t| t.speaker.as_str()).collect();
    eprintln!(
        "diarizer: {} turn(s), {} label(s)",
        turns.len(),
        labels.len()
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("mambodubb-diarizer: {msg}");
            ExitCode::FAILURE
        }
    }
}
