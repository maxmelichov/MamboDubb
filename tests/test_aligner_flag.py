"""--aligner: recorded, replayed, fingerprint-safe, and honest about scope.

Same contract as --separator/--diarizer: the default must fingerprint exactly
as before the flag existed. The aligner-specific rules: an unsupported source
language is a printed no-op, not an error, and span grouping caps the blast
radius of a bad CTC path at one silence-bounded stretch.
"""

from __future__ import annotations

from dubbing import align, cli


def _args(argv: list[str]):
    args = cli.parse_args(["input.mp4", *argv])
    args.src, args.tgt = "he", "en"
    return args


def test_the_default_fingerprints_exactly_like_before_the_flag_existed():
    args = _args([])
    cli.resolve_settings(args, {"source": {}})
    assert args.aligner == "none"
    params = cli.stage_params(args, {"source": {}})["transcript"]
    assert "aligner" not in params


def test_wav2vec2_lands_in_the_transcript_fingerprint():
    args = _args(["--aligner", "wav2vec2"])
    cli.resolve_settings(args, {"source": {}})
    params = cli.stage_params(args, {"source": {}})["transcript"]
    assert params["aligner"] == "wav2vec2"


def test_a_bare_rerun_keeps_the_aligner_the_run_recorded():
    args = _args([])
    cli.resolve_settings(args, {"source": {"aligner": "wav2vec2"}})
    assert args.aligner == "wav2vec2"


def test_an_unsupported_language_is_a_no_op_not_an_error(tmp_path):
    words = [{"t": 0.0, "end": 0.5, "text": "hello"}]
    assert align.refine_word_times(words, tmp_path / "x.wav", "ru") == 0
    assert words[0]["t"] == 0.0


def test_spans_break_at_silences_and_at_the_length_cap():
    words = ([{"t": float(i), "end": i + 0.5, "text": "w"} for i in range(3)]
             + [{"t": 10.0, "end": 10.5, "text": "w"}])
    spans = align.spans_of(words)
    assert [len(s) for s in spans] == [3, 1]
    long = [{"t": i * 0.6, "end": i * 0.6 + 0.5, "text": "w"} for i in range(40)]
    assert all(s[-1]["end"] - s[0]["t"] <= align.SPAN_MAX_SEC
               for s in align.spans_of(long))


def test_normalize_strips_niqqud_and_unspellable_marks():
    vocab = {ch: i for i, ch in enumerate("שלוםאבגדהחת")}
    assert align._normalize("שָׁלוֹם", vocab) == "שלום"
    assert align._normalize("hello", vocab) == ""


def test_the_editor_rerun_command_carries_the_aligner():
    from pathlib import Path

    from editor import jobs

    cmd = jobs.dub_command("in.mp4", Path("out"), opts={"aligner": "wav2vec2"})
    assert cmd[cmd.index("--aligner") + 1] == "wav2vec2"
