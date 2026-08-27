# Credits

MamboDubb is a pipeline, not a model: all of the intelligence below is the work of the teams that built and released these models openly. Full credit to them:

| Model | By | Used for |
|---|---|---|
| [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | Qwen team, Alibaba Cloud | Speech synthesis + zero-shot voice cloning |
| [QwenTTS-he-1.7B](https://huggingface.co/notmax123/QwenTTS-he-1.7B) | Maxim Melichov | Hebrew LoRA over the same Qwen3-TTS checkpoint |
| [RenikudPlus](https://github.com/maxmelichov/RenikudPlus) | Maxim Melichov, Yakov Kolani, Morris Alper | Hebrew grapheme to stressed-IPA G2P feeding the Hebrew TTS |
| [Gemma 4 12B it](https://huggingface.co/mlx-community/gemma-4-12B-it-6bit) | Google DeepMind, 6-bit MLX quant by the mlx-community | Context-aware translation |
| [whisper-large-v3-turbo-ct2](https://huggingface.co/ivrit-ai/whisper-large-v3-turbo-ct2) | ivrit.ai, fine-tuning OpenAI Whisper | Hebrew transcription |
| [faster-whisper-large-v3-turbo-ct2](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2) | OpenAI Whisper, CT2 conversion by deepdml | Transcription for the other source languages |
| [faster-whisper base / base.en / tiny.en](https://huggingface.co/Systran) | OpenAI Whisper, converted by SYSTRAN | Verifying every synthesized take by ear |
| [Demucs (htdemucs_ft)](https://github.com/adefossez/demucs) | Alexandre Défossez et al., Meta AI | Separating voices from music and effects |
| [speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) | pyannote.audio, Hervé Bredin | Who speaks when. CC-BY-4.0, redistributed unmodified in `third_party/pyannote-speaker-diarization-community-1` (see its `NOTICE.md`), so no HF account is needed |
| [spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | SpeechBrain | Speaker embeddings for voice-consistent cloning |
| [lang-id-voxlingua107-ecapa](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa) | SpeechBrain | Spoken-language identification |
| [Silero VAD](https://github.com/snakers4/silero-vad) | Silero Team | Voice activity detection |

Running on [MLX](https://github.com/ml-explore/mlx) + [mlx-lm](https://github.com/ml-explore/mlx-lm) (Apple), [faster-whisper](https://github.com/SYSTRAN/faster-whisper) / CTranslate2, PyTorch, [qwen-tts](https://github.com/QwenLM/Qwen3-TTS), FFmpeg, SoX and yt-dlp.
