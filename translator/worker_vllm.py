"""Gemma 4 vLLM translation worker same JSON-lines protocol as worker.py.

Why a second worker rather than a flag on the first: vLLM owns its own scheduler,
KV cache and CUDA graphs, so "load the model" means something different enough that
sharing one `load()` would be a disguise, not a reuse. Everything *outside* the
load-and-decode step is identical on purpose the protocol below is the same
protocol, byte for byte, so `dubbing/translate.WorkerHandle` cannot tell which
worker it spawned, and every prompt and every post-processing rule still lives in
`dubbing/translate.py` where both backends share it.

Protocol (one JSON object per line, stdout flushed after every line):
  worker → parent   {"ready": true}                        once, after the model loads
  parent → worker   {"id": n, "user_text": str, "max_new_tokens": int}
  parent → worker   {"batch": [{"id", "user_text", "max_new_tokens"}, ...]}
  worker → parent   {"id": n, "text": str}                 the raw decoded completion
  worker → parent   {"id": n|null, "error": str}           and the loop continues

A batch is answered with one reply line per item and nothing else; ids are what
match a reply to its request, so replies may arrive in any order. That is the
whole point of this backend: vLLM batches continuously, and a batch of N prompts
costs far less than N sequential requests. The single-request form is kept
because the translate stage is mostly sequential (each segment is shown the
previous segment's *output*), and one request at a time through vLLM is still
faster than one at a time through transformers.

Decoding matches worker.py exactly: greedy (temperature 0), the model's own chat
template with `enable_thinking=False`, and special tokens KEPT in the decoded text —
the parent splits on the literal `<channel|>` / turn markers, and skipping them
would leave a bare "thought" line in the spoken text.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "models" / "gemma-4-12b-it-cuda"

# 12B bf16 is ~24 GB of weights; the rest of the card is KV cache. Both are
# env-tunable because the right numbers depend on the card, not on the code:
# a 32 GB GPU shared with nothing wants a high fraction, one shared with a
# display server wants less. The context bound keeps the KV cache affordable —
# a translated segment is a couple of hundred tokens, never a long document.
GPU_UTIL = float(os.environ.get("TRANSLATOR_VLLM_GPU_UTIL", "0.90"))
MAX_MODEL_LEN = int(os.environ.get("TRANSLATOR_VLLM_MAX_LEN", "8192"))


def reserve_stdout():
    """Take fd 1 for the protocol and point everything else at stderr.

    vLLM logs its engine banner to **stdout** ("INFO … non-default args: {…}"),
    which lands in the middle of a JSON-lines stream and desyncs the parent on the
    very first read the ready line it gets is a log line. Rather than chase every
    logger vLLM, ray, torch or a future dependency might install, fd 1 is duplicated
    away for this module's own use and the real fd 1 is replaced by fd 2. After
    this, anything anyone prints to stdout goes to stderr (where `WorkerHandle`
    already streams it through), and only `emit` can reach the parent.

    Called from `main`, never at import: importing this module must stay free of
    side effects (the tests import it to check that both workers parse one
    protocol, and stealing pytest's stdout would be a memorable way to fail).
    """
    global _PROTOCOL
    _PROTOCOL = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)
    sys.stdout = os.fdopen(os.dup(2), "w", encoding="utf-8", buffering=1)


_PROTOCOL = None


def log(msg: str) -> None:
    print(f"  worker: {msg}", file=sys.stderr, flush=True)


def emit(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False), file=_PROTOCOL or sys.stdout, flush=True)


def load(path: str):
    from vllm import LLM

    log(f"loading {path} on vLLM (bfloat16, gpu_util={GPU_UTIL}, "
        f"max_model_len={MAX_MODEL_LEN})")
    llm = LLM(model=path, dtype="bfloat16", gpu_memory_utilization=GPU_UTIL,
              max_model_len=MAX_MODEL_LEN, trust_remote_code=True)
    tokenizer = llm.get_tokenizer()
    log("model loaded")
    return llm, tokenizer


def prompt_ids(tokenizer, user_text: str) -> list[int]:
    """One user turn through the model's own chat template, as token ids.

    Tokenised rather than rendered to a string for the same reason the MLX path
    tokenises (see `dubbing/translate._prompt`): the template emits `<bos>`
    itself, and handing vLLM the string would prepend a second one.

    Unwrapping is written against the *shape*, not the class. transformers 5
    returns a `BatchEncoding`, which is a `UserDict` and therefore not a `dict` —
    an `isinstance(…, dict)` test silently misses it, `list()` then yields the
    encoding's *keys*, and vLLM fails deep inside with "'>' not supported between
    str and int". So anything that is not already a sequence is indexed for
    "input_ids", one level of batch nesting is peeled, and the ints are coerced,
    which turns any remaining surprise into an error naming this function.
    """
    messages = [{"role": "user", "content": user_text}]
    try:
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                            enable_thinking=False, tokenize=True)
    except TypeError:                       # template does not accept enable_thinking
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                            tokenize=True)
    if not isinstance(ids, (list, tuple)):  # BatchEncoding, dict, tensor-bearing mapping
        ids = ids["input_ids"]
    if len(ids) and not isinstance(ids[0], int):   # batched shape for one message list
        ids = ids[0]
    return [int(t) for t in ids]


def generate(llm, tokenizer, reqs: list[dict]) -> list[str]:
    """Greedy-decode every request in one vLLM call. Returns texts in `reqs` order."""
    from vllm import SamplingParams

    prompts = [{"prompt_token_ids": prompt_ids(tokenizer, r["user_text"])} for r in reqs]
    params = [SamplingParams(temperature=0.0, top_p=1.0,
                             max_tokens=int(r.get("max_new_tokens", 400)),
                             skip_special_tokens=False)
              for r in reqs]
    outs = llm.generate(prompts, params, use_tqdm=False)
    return [(out.outputs[0].text or "").strip() for out in outs]


def serve_one(llm, tokenizer, reqs: list[dict]) -> None:
    """Answer a batch (of one or many), one reply line per request.

    A failure is reported per request rather than as a batch-level error: the
    parent matches replies by id and would otherwise wait forever for the ones
    the exception swallowed.
    """
    try:
        for req, text in zip(reqs, generate(llm, tokenizer, reqs)):
            emit({"id": req["id"], "text": text})
    except Exception as exc:
        import traceback

        traceback.print_exc(file=sys.stderr)     # the parent streams stderr through
        for req in reqs:
            emit({"id": req["id"], "error": f"{type(exc).__name__}: {exc}"})


def parse(raw: str) -> list[dict]:
    """The requests carried by one protocol line single or batch. Raises on junk."""
    msg = json.loads(raw)
    items = msg["batch"] if isinstance(msg, dict) and "batch" in msg else [msg]
    if not isinstance(items, list) or not items:
        raise ValueError("empty batch")
    out = []
    for item in items:
        out.append({"id": item["id"], "user_text": item["user_text"],
                    "max_new_tokens": int(item.get("max_new_tokens", 400))})
    return out


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TRANSLATOR_MODEL_PATH",
                                                                str(DEFAULT_MODEL))
    reserve_stdout()                        # before vLLM logs its banner to fd 1
    llm, tokenizer = load(path)
    emit({"ready": True})
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            reqs = parse(raw)
        except Exception as exc:            # malformed line report and keep serving
            emit({"id": None, "error": f"bad request: {exc}"})
            continue
        serve_one(llm, tokenizer, reqs)


if __name__ == "__main__":
    main()
