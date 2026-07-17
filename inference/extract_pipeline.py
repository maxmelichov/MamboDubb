#!/usr/bin/env python3
"""Phase 1 & 2: extract audio, separate stems, diarize, and transcribe Hebrew speech.

Outputs a JSON timeline of:
  [{speaker_id, start, end, text}, ...]

Also writes vocals.wav / background.wav under the run output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Fix for Pyannote MPS fallback on Apple Silicon
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from faster_whisper import WhisperModel
    from pyannote.audio import Pipeline
except ImportError as e:
    print(f"Error importing dependencies: {e}", file=sys.stderr)
    print("Install with: uv sync", file=sys.stderr)
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = str(REPO_ROOT / "models" / "whisper-large-v3-turbo-ct2")
DEFAULT_HUB_ID = "ivrit-ai/whisper-large-v3-turbo-ct2"
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"


def download_youtube_video(url: str, output_dir: Path) -> Path:
    """Download a YouTube video to the output directory using yt-dlp."""
    try:
        import yt_dlp
    except ImportError as e:
        raise SystemExit(
            f"yt-dlp is required for YouTube URLs ({e}).\n"
            "Install with: uv sync   (or: uv add yt-dlp)"
        ) from e

    print(f"Downloading YouTube video from {url}...", file=sys.stderr)
    output_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
        "quiet": False,
        "no_warnings": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        path = Path(filename)
        if not path.is_file():
            # yt-dlp may rewrite extension after merge (e.g. .webm → .mp4)
            candidates = list(output_dir.glob(f"{Path(info.get('title', '')).name}*"))
            if candidates:
                return candidates[0]
            raise FileNotFoundError(f"yt-dlp finished but file not found: {path}")
        return path


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            "ffmpeg not found on PATH. Install it (e.g. `brew install ffmpeg`) and retry."
        )


def extract_audio(
    input_path: Path,
    output_wav: Path,
    sample_rate: int = 44100,
    max_duration: float | None = None,
) -> Path:
    """Strip audio from video (or re-encode audio) to a WAV for Demucs.

    If max_duration is set, only the first N seconds are extracted.
    """
    require_ffmpeg()
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
    ]
    if max_duration is not None and max_duration > 0:
        cmd.extend(["-t", str(max_duration)])
    cmd.extend(
        [
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            str(output_wav),
        ]
    )
    dur_msg = f" (first {max_duration:g}s)" if max_duration else ""
    print(f"Extracting audio{dur_msg} → {output_wav}", file=sys.stderr)
    subprocess.run(cmd, check=True, capture_output=True)
    if not output_wav.is_file():
        raise FileNotFoundError(f"ffmpeg did not produce {output_wav}")
    return output_wav


def separate_vocals(input_audio: Path, demucs_dir: Path) -> tuple[Path, Path]:
    """Run HTDemucs two-stem separation; return (vocals, background/no_vocals)."""
    demucs_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running Demucs (htdemucs, two-stems) on {input_audio}...", file=sys.stderr)
    cmd = [
        sys.executable,
        "-m",
        "demucs",
        "--two-stems=vocals",
        "-n",
        "htdemucs",
        "-o",
        str(demucs_dir),
        str(input_audio),
    ]
    subprocess.run(cmd, check=True)

    stem_dir = demucs_dir / "htdemucs" / input_audio.stem
    vocals_path = stem_dir / "vocals.wav"
    background_path = stem_dir / "no_vocals.wav"
    if not vocals_path.is_file():
        raise FileNotFoundError(f"Demucs failed to produce {vocals_path}")
    if not background_path.is_file():
        raise FileNotFoundError(f"Demucs failed to produce {background_path}")
    return vocals_path, background_path


def diarize_audio(vocals_path: Path, model_id: str = DEFAULT_DIARIZATION_MODEL) -> list[dict]:
    print(f"Running Pyannote diarization ({model_id}) on {vocals_path}...", file=sys.stderr)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print(
            "Warning: HF_TOKEN not set. Pyannote gated models need a token + accepted terms.",
            file=sys.stderr,
        )

    try:
        pipeline = Pipeline.from_pretrained(model_id, token=token)
    except TypeError:
        pipeline = Pipeline.from_pretrained(model_id, use_auth_token=token)

    if pipeline is None:
        raise SystemExit(
            "Failed to load pyannote pipeline. Set HF_TOKEN in .env and accept model "
            f"conditions at https://huggingface.co/{model_id}"
        )

    import torch

    if torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))
    elif torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    output = pipeline(str(vocals_path))
    # pyannote.audio 4.x returns DiarizeOutput; 3.x returned Annotation directly.
    annotation = getattr(output, "speaker_diarization", output)

    segments: list[dict] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append(
            {
                "speaker_id": speaker,
                "start": round(float(turn.start), 3),
                "end": round(float(turn.end), 3),
            }
        )
    return segments


def resolve_whisper_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import ctranslate2

        if "cuda" in ctranslate2.get_supported_compute_types("cuda"):
            return "cuda"
    except Exception:
        pass
    return "cpu"


def resolve_whisper_model(model: str) -> str:
    path = Path(model)
    if path.is_dir():
        return str(path)
    if model == DEFAULT_MODEL and not Path(DEFAULT_MODEL).is_dir():
        return DEFAULT_HUB_ID
    return model


def transcribe_and_merge(
    vocals_path: Path,
    segments: list[dict],
    model_name: str,
    language: str = "he",
    device: str = "auto",
    compute_type: str = "auto",
    max_pause: float = 0.75,
    phrase_pause: float = 0.18,
) -> list[dict]:
    from inference.segment_merge import words_to_utterances

    model_path = resolve_whisper_model(model_name)
    device = resolve_whisper_device(device)
    print(
        f"Transcribing {vocals_path} with faster-whisper ({model_path}) on {device}...",
        file=sys.stderr,
    )

    model = WhisperModel(model_path, device=device, compute_type=compute_type)
    whisper_segments, info = model.transcribe(
        str(vocals_path),
        language=language,
        word_timestamps=True,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": int(max_pause * 1000)},
    )

    words = []
    for ws in whisper_segments:
        if ws.words:
            words.extend(ws.words)

    results = words_to_utterances(
        words,
        segments,
        max_pause=max_pause,
        phrase_pause=phrase_pause,
    )
    print(
        f"Detected language={info.language} p={info.language_probability:.2f}; "
        f"{len(words)} words → {len(results)} utterances "
        f"(max_pause={max_pause}s, phrase_pause={phrase_pause}s)",
        file=sys.stderr,
    )
    return results, model


def detect_segment_languages(
    vocals_path: Path,
    segments: list[dict],
    model: WhisperModel,
    min_duration: float = 1.0,
) -> None:
    """Use Whisper to detect language for each utterance segment > min_duration."""
    print(f"Detecting language for {len(segments)} segments...", file=sys.stderr)
    import numpy as np

    for i, seg in enumerate(segments):
        start = float(seg["start"])
        end = float(seg["end"])
        dur = end - start
        
        # Only detect language on segments long enough to have confident speech
        if dur < min_duration:
            # Inherit from default or context
            seg["language"] = "he"
            continue
            
        cmd = [
            "ffmpeg", "-y", "-ss", str(start), "-t", str(dur), "-i", str(vocals_path),
            "-f", "f32le", "-acodec", "pcm_f32le", "-ar", "16000", "-ac", "1", "-"
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, check=True)
            audio = np.frombuffer(result.stdout, dtype=np.float32)
            lang, prob, _ = model.detect_language(audio)
            seg["language"] = lang
            print(f"  Segment {i:02d} [{start:.1f}-{end:.1f}s]: detected {lang} (prob: {prob:.2f})", file=sys.stderr)
        except Exception as e:
            print(f"  Segment {i:02d} [{start:.1f}-{end:.1f}s]: language detection failed ({e})", file=sys.stderr)
            seg["language"] = "he"



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 & 2: Demucs + Pyannote + faster-whisper (ivrit CT2) → Hebrew JSON."
    )
    parser.add_argument("video", type=str, help="Input video file path or YouTube URL.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: outputs/<run_id>/segments.json).",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Working directory for stems/audio (default: outputs/<run_id>/).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Local CT2 dir or Hub id (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--diarization-model",
        default=DEFAULT_DIARIZATION_MODEL,
        help=f"Pyannote model id (default: {DEFAULT_DIARIZATION_MODEL}).",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--compute-type", default="auto")
    parser.add_argument("--language", "-l", default="he")
    parser.add_argument(
        "--max-pause",
        type=float,
        default=1.0,
        help="Gap (seconds) that starts a new utterance. Shorter same-speaker gaps become TTS pauses.",
    )
    parser.add_argument(
        "--phrase-pause",
        type=float,
        default=0.18,
        help="Within an utterance, gaps >= this become internal pause markers for TTS.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Only process the first N seconds of audio (e.g. 60 for a 1-minute test).",
    )
    parser.add_argument(
        "--skip-demucs",
        action="store_true",
        help="Skip source separation; diarize/transcribe the extracted mix directly.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    is_url = args.video.startswith("http://") or args.video.startswith("https://")
    
    if not is_url:
        video_path = Path(args.video)
        if not video_path.is_file():
            raise SystemExit(f"File not found: {video_path}")
        stem_name = video_path.stem
    else:
        stem_name = "youtube_download"

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + stem_name
    workdir = args.workdir or (REPO_ROOT / "outputs" / run_id)
    workdir.mkdir(parents=True, exist_ok=True)
    
    if is_url:
        video_path = download_youtube_video(args.video, workdir)
    else:
        video_path = Path(args.video)

    source_wav = workdir / "source.wav"
    extract_audio(video_path, source_wav, max_duration=args.max_duration)

    if args.skip_demucs:
        vocals_path = source_wav
        background_path = None
    else:
        vocals_path, background_path = separate_vocals(source_wav, workdir / "demucs")

    # Copy stems to a stable location for later pipeline stages
    vocals_out = workdir / "vocals.wav"
    if vocals_path.resolve() != vocals_out.resolve():
        shutil.copy2(vocals_path, vocals_out)
        vocals_path = vocals_out
    if background_path is not None:
        background_out = workdir / "background.wav"
        shutil.copy2(background_path, background_out)
        background_path = background_out

    segments = diarize_audio(vocals_path, model_id=args.diarization_model)
    
    results, whisper_model = transcribe_and_merge(
        vocals_path,
        segments,
        args.model,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        max_pause=args.max_pause,
        phrase_pause=args.phrase_pause,
    )

    detect_segment_languages(vocals_path, results, whisper_model)

    payload = {
        "source": str(video_path.resolve()),
        "workdir": str(workdir.resolve()),
        "vocals": str(vocals_path.resolve()),
        "background": str(background_path.resolve()) if background_path else None,
        "language": args.language,
        "whisper_model": resolve_whisper_model(args.model),
        "diarization_model": args.diarization_model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "segments": results,
    }

    output_path = args.output or (workdir / "segments.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(
        f"Wrote {len(results)} segments → {output_path}",
        file=sys.stderr,
    )
    print(str(output_path))


if __name__ == "__main__":
    main()
