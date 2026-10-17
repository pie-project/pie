"""Greedy text completion, device-carried — the Python twin of `text-completion`.

Same program as `examples/text-completion/src/lib.rs`, traced from
Python: one prefill pass whose greedy pick seeds the decode once, then a
1-wide decode pass whose token is loop-carried on the device (`reduce_argmax`
in the epilogue, put straight back into the channel `embed` reads) with the
host draining a mirror behind the runtime's run-ahead window (`run_ahead`).
The traced container is byte-identical to the Rust inferlet's, so both share
one program-cache entry.
"""

from inferlet import chat, model
from inferlet.eta import (
    Channel,
    ForwardPass,
    Pipeline,
    StateGeometry,
    WorkingSet,
    dtype,
    intrinsics,
    channel_capacity,
    reduce_argmax,
    reshape,
)

def greedy(logits):
    """The greedy pick over a logits row, as a one-lane `[1]` i32 cell."""
    return reshape(reduce_argmax(logits), [1])

async def main(input: dict) -> dict:
    prompt_text = input.get("prompt", "The capital of France is")
    max_tokens = int(input.get("max_tokens", 8))

    ws = WorkingSet()

    if max_tokens == 0:
        return {"text": "", "count": 0, "tokens": []}

    # The model's opening (`<bos>` where it has one) before the raw text: a
    # gemma without it answers noise.
    prompt = chat.prefix() + model.encode(prompt_text)
    if not prompt:
        prompt = [0]
    n = len(prompt)

    pipe = Pipeline()
    generated: list[int] = []

    toks = Channel.from_(prompt, dtype.i32).named("toks_p")
    embed_indptr = Channel.from_([0, n], dtype.u32).named("embed_indptr_p")
    tok_out = Channel([1], dtype.i32).named("tok_out_p")

    fwd = ForwardPass()
    fwd.embed(toks, embed_indptr)
    fwd.state([ws], StateGeometry.folding())

    @fwd.epilogue
    def _prefill_epilogue():
        tok_out.put(greedy(intrinsics.logits()))

    pipe.submit(fwd)
    first = await tok_out.take_scalar()
    generated.append(first)

    # ── DECODE (1-wide, device loop-carried token) ─────────────────────────
    if len(generated) < max_tokens:
        tok_in = Channel.from_([first], dtype.i32).named("tok_in")
        tok_out = Channel([1], dtype.i32).capacity(channel_capacity()).named("tok_out")
        embed_indptr = Channel.from_([0, 1], dtype.u32).named("embed_indptr")

        fwd = ForwardPass()
        fwd.embed(tok_in, embed_indptr)
        fwd.state([ws], StateGeometry.folding())

        @fwd.epilogue
        def _decode_epilogue():
            token = greedy(intrinsics.logits())
            tok_in.put(token)
            tok_out.put(token)

        budget = max_tokens - 1

        async def on_token() -> bool:
            generated.append(await tok_out.take_scalar())
            return True

        await pipe.run_ahead(fwd, budget, on_token)
    pipe.close()

    return {
        "text": model.decode(generated),
        "count": len(generated),
        "tokens": generated,
    }
