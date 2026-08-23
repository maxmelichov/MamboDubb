"""Gemma 4 CUDA translation worker JSON lines on stdin/stdout, logs on stderr.

Runs in its own uv venv (see translator/pyproject.toml) because the main venv pins
transformers==4.57.3, which predates Gemma 4's `gemma4_unified` architecture.
`dubbing/translate.py` spawns this as a subprocess and speaks the protocol below.

Protocol (one JSON object per line, stdout flushed after every line):
  worker → parent   {"ready": true}                       once, after the model loads
  parent → worker   {"id": n, "user_text": str, "max_new_tokens": int}
  parent → worker   {"batch": [{"id", "user_text", "max_new_tokens"}, ...]}
  worker → parent   {"id": n, "text": str}                 the raw decoded completion
  worker → parent   {"id": n|null, "error": str}           and the loop continues

A batch is answered with one reply line per item and nothing else. This worker has
nothing to gain from it it decodes the items one after another, which is what a
sequential loop of single requests already did but it must *understand* the
message, so a caller that batches works against either backend and the vLLM worker
(translator/worker_vllm.py), where continuous batching is the whole point, stays a
drop-in replacement rather than a fork of the protocol.

The worker applies the model's own chat template (one user turn) and greedy-decodes;
all prompt construction and post-processing stays in dubbing/translate.py so the two
backends share every rule.

TRANSLATOR_LOAD_4BIT=1 in the environment is low-VRAM mode: the same checkpoint,
loaded through bitsandbytes as NF4 with a bfloat16 compute dtype, which turns
~24 GB of weights into roughly 7 and puts the translator on an ordinary card.
The parent sets it from `dubbing.translate.low_vram`; a hand-launched worker can
set it too. It degrades to bfloat16, loudly, when bitsandbytes is not installed
or there is no CUDA device to quantise onto (see `quant_config`).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "models" / "gemma-4-12b-it-cuda"


def utf8_stdio() -> None:
    """The protocol is UTF-8 JSON lines; a Windows console's stdio is not.

    Spelled out here rather than imported from `dubbing.tools`: this file runs in
    its own venv, which does not have the pipeline package installed. The parent
    also sets PYTHONIOENCODING, so this is the belt to that suspenders — it costs
    nothing and it means a worker launched by hand behaves the same way.
    """
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def log(msg: str) -> None:
    print(f"  worker: {msg}", file=sys.stderr, flush=True)


def emit(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def want_4bit() -> bool:
    """True when the parent asked for low-VRAM mode (`dubbing.translate.low_vram`)."""
    return os.environ.get("TRANSLATOR_LOAD_4BIT", "").strip().lower() in (
        "1", "true", "yes", "on")


def quant_config(cuda: bool):
    """The bitsandbytes NF4 config, or None with a reason logged.

    NF4 with a bfloat16 compute dtype and double quantisation: ~4.1 bits per
    weight against the 16 the default path uses, so 12B of weights lands near
    7 GB instead of 24. Compute stays bfloat16, which is what keeps this a
    memory trade rather than a speed one — each block is dequantised on its way
    into the matmul.

    Two ways this is unavailable, and neither may be a crash. bitsandbytes is an
    optional extra (`--extra lowvram`), and it is a CUDA library: on a CPU-only
    box there is nothing for it to quantise onto. Either way the caller falls
    back to bfloat16, which is the behaviour that existed before this mode, and
    the message says what to install. A silent fallback would be worse than
    either outcome: a user who asked for 4-bit and got a 24 GB allocation
    deserves to be told before the OOM, not after.
    """
    if not cuda:
        log("low-VRAM mode asked for, but there is no CUDA device here: "
            "bitsandbytes only quantises onto a GPU. Loading bfloat16 instead.")
        return None
    try:
        import bitsandbytes  # noqa: F401
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        log(f"low-VRAM mode asked for, but bitsandbytes is unavailable ({exc}). "
            "Install it with `uv sync --project translator --extra lowvram`. "
            "Loading bfloat16 instead.")
        return None
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)


def load(path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    quant = quant_config(device.startswith("cuda")) if want_4bit() else None
    log(f"loading {path} on {device} ({'4-bit NF4' if quant else 'bfloat16'})")
    tokenizer = AutoTokenizer.from_pretrained(path)
    kwargs = {"dtype": torch.bfloat16}
    if quant is not None:
        # device_map places the weights as they are quantised, shard by shard, so
        # the 24 GB bf16 checkpoint never exists in full on either the GPU or in
        # RAM — which is the whole point on a card that could not hold it.
        kwargs.update(quantization_config=quant, device_map={"": 0})
    try:
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    except TypeError:  # transformers < 5 spells it torch_dtype
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    # A quantised model is already on its device and refuses to be moved: bnb
    # keeps the quantisation state beside the weights, and `.to()` raises rather
    # than carry it. device_map put it where it belongs.
    if quant is None:
        model.to(device)
    model.eval()
    log("model loaded")
    return tokenizer, model, device


def generate(tokenizer, model, device: str, user_text: str, max_new_tokens: int) -> str:
    import torch

    messages = [{"role": "user", "content": user_text}]
    try:
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                            enable_thinking=False, return_tensors="pt")
    except TypeError:  # template does not accept enable_thinking
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                            return_tensors="pt")
    if not torch.is_tensor(ids):  # transformers 5 returns a BatchEncoding by default
        ids = ids["input_ids"]
    ids = ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, attention_mask=torch.ones_like(ids),
                             max_new_tokens=max_new_tokens, do_sample=False)
    # Keep special tokens: Gemma sometimes opens a thought channel even though the
    # template pre-closed it, and the parent's shared post-processing splits on the
    # literal <channel|> / turn markers. skip_special_tokens=True would delete the
    # markers and leave the bare word "thought" in the spoken line.
    return tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=False).strip()


def parse(raw: str) -> list[dict]:
    """The requests carried by one protocol line single or batch. Raises on junk."""
    msg = json.loads(raw)
    items = msg["batch"] if isinstance(msg, dict) and "batch" in msg else [msg]
    if not isinstance(items, list) or not items:
        raise ValueError("empty batch")
    return [{"id": item["id"], "user_text": item["user_text"],
             "max_new_tokens": int(item.get("max_new_tokens", 400))} for item in items]


def main() -> None:
    utf8_stdio()
    path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TRANSLATOR_MODEL_PATH",
                                                                str(DEFAULT_MODEL))
    tokenizer, model, device = load(path)
    emit({"ready": True})
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            reqs = parse(raw)
        except Exception as exc:  # malformed line report and keep serving
            emit({"id": None, "error": f"bad request: {exc}"})
            continue
        # One reply line per request, whether the line held one or many. This
        # worker has no batching engine it decodes them in turn, exactly as a
        # sequence of single requests would.
        for req in reqs:
            try:
                emit({"id": req["id"], "text": generate(tokenizer, model, device,
                                                        req["user_text"],
                                                        req["max_new_tokens"])})
            except Exception as exc:
                emit({"id": req["id"], "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
