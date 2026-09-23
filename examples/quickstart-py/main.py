"""Greedy text completion, one forward pass per token, driven from the host.

The shortest complete inferlet: the README's example. `text-completion-py`
is the same program with chunked prefill and a device loop-carried decode.
"""

from inferlet import chat, model
from inferlet.eta import ForwardPass, Pipeline, WorkingSet, intrinsics, reduce_argmax


async def main(input: dict) -> dict:
    tokens = chat.prefix() + model.encode(input.get("prompt", "The capital of France is"))
    max_tokens = int(input.get("max_tokens", 8))
    ws = WorkingSet(tokens=len(tokens) + max_tokens)  # KV pages (and recurrent state on a hybrid model)
    pipe = Pipeline()

    done = 0  # tokens already in the KV cache
    for _ in range(max_tokens):
        fwd = ForwardPass()
        fwd.embed(tokens[done:])                        # the new tokens
        fwd.bind_state(ws, ws.geometry(done, len(tokens)))
        out = fwd.epilogue(lambda: reduce_argmax(intrinsics.logits()))  # runs on the device
        pipe.submit(fwd)
        done = len(tokens)
        tokens.append(await out)
    pipe.close()

    return {"text": model.decode(tokens[-max_tokens:])}
