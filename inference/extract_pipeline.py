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


def _phrases_from_words(words: list, phrase_pause: float) -> list[dict]:
    """Split word stream into pause-aware phrases (absolute timestamps)."""
    if not words:
        return []
    phrases: list[dict] = []
    buf: list[str] = []
    p_start = float(words[0].start)
    p_end = float(words[0].end)

    def flush(pause_after: float = 0.0) -> None:
        nonlocal buf, p_start, p_end
        if not buf:
            return
        text = "".join(buf).strip()
        if text and " " not in text and len(buf) > 1:
            text = " ".join(w.strip() for w in buf if w.strip())
        if text:
            phrases.append(
                {
                    "text": text,
                    "start": round(p_start, 3),
                    "end": round(p_end, 3),
                    "pause_after": round(pause_after, 3),
                }
            )
        buf = []

    for i, w in enumerate(words):
        token = w.word or ""
        if i == 0:
            buf = [token]
            p_start = float(w.start)
            p_end = float(w.end)
            continue
        gap = float(w.start) - float(words[i - 1].end)
        if gap >= phrase_pause:
            flush(pause_after=gap)
            buf = [token]
            p_start = float(w.start)
            p_end = float(w.end)
        else:
            if not buf:
                p_start = float(w.start)
            buf.append(token)
            p_end = float(w.end)
    flush(0.0)
    if phrases:
        phrases[-1]["pause_after"] = 0.0
    return phrases


def transcribe_and_merge(
    vocals_path: Path,
    segments: list[dict],
    model_name: str,
    language: str = "auto",
    device: str = "auto",
    compute_type: str = "auto",
    max_pause: float = 0.75,
    phrase_pause: float = 0.18,
) -> tuple[list[dict], WhisperModel]:
    """Per-turn language detect (he/en/ar) + ASR; keep non-Hebrew as keep_original."""
    from inference.lang_detect import CANDIDATE_LANGS, detect_and_transcribe_turn
    from inference.segment_merge import merge_diarization_turns, merge_same_speaker_segments

    model_path = resolve_whisper_model(model_name)
    device = resolve_whisper_device(device)
    print(
        f"Language-aware ASR on {vocals_path} with faster-whisper ({model_path}) on {device}...",
        file=sys.stderr,
    )

    model = WhisperModel(model_path, device=device, compute_type=compute_type)
    turns = merge_diarization_turns(segments, max_pause=max_pause)
    print(f"Diarization turns: {len(segments)} → {len(turns)} after pause merge", file=sys.stderr)

    if language and language != "auto":
        candidates = (language,)
    else:
        candidates = CANDIDATE_LANGS

    results: list[dict] = []
    for i, turn in enumerate(turns):
        start = float(turn["start"])
        end = float(turn["end"])
        if end - start < 0.25:
            continue
        print(
            f"  Turn {i:02d} [{start:.1f}-{end:.1f}s] {turn['speaker_id']}…",
            file=sys.stderr,
        )
        det = detect_and_transcribe_turn(
            model,
            vocals_path,
            start,
            end,
            candidates=candidates,
        )
        lang = det["language"]
        phrases = _phrases_from_words(det["words"], phrase_pause)
        if not phrases and det["text"]:
            phrases = [
                {
                    "text": det["text"],
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "pause_after": 0.0,
                }
            ]
        if not phrases:
            print("    empty — skip", file=sys.stderr)
            continue

        # Drop low-confidence micro junk (common on music beds / breath)
        if (end - start) < 0.8 and float(det.get("language_score") or 0) < -0.5:
            print(
                f"    junk skip score={det.get('language_score')} dur={end - start:.2f}s",
                file=sys.stderr,
            )
            continue
        if len(det["text"].strip()) < 3:
            print("    too short — skip", file=sys.stderr)
            continue

        keep_original = lang != "he"
        # For non-Hebrew, expose original text as text_en for subs (no machine translate).
        if keep_original:
            for p in phrases:
                p["text_en"] = p["text"]

        seg = {
            "speaker_id": turn["speaker_id"],
            "start": round(float(phrases[0]["start"]), 3),
            "end": round(float(phrases[-1]["end"]), 3),
            "language": lang,
            "language_score": det.get("language_score"),
            "keep_original": keep_original,
            "phrases": phrases,
            "text": " ".join(p["text"] for p in phrases),
            "duration": 0.0,
            "pauses": [float(p["pause_after"]) for p in phrases],
        }
        seg["duration"] = round(seg["end"] - seg["start"], 3)
        if keep_original:
            seg["text_en"] = seg["text"]
        results.append(seg)
        flag = "KEEP" if keep_original else "DUB"
        print(
            f"    → {lang} ({flag}) score={det.get('language_score')} | {seg['text'][:70]}",
            file=sys.stderr,
        )

    results = merge_same_speaker_segments(results, max_pause=max_pause)
    n_keep = sum(1 for s in results if s.get("keep_original"))
    print(
        f"Utterances: {len(results)} ({n_keep} keep-original, {len(results) - n_keep} Hebrew→dub)",
        file=sys.stderr,
    )
    return results, model


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
    parser.add_argument(
        "--language",
        "-l",
        default="auto",
        help="Force ASR language (he/en/ar) or 'auto' to score he/en/ar per turn (default: auto).",
    )
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
    
    results, _whisper_model = transcribe_and_merge(
        vocals_path,
        segments,
        args.model,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        max_pause=args.max_pause,
        phrase_pause=args.phrase_pause,
    )

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
