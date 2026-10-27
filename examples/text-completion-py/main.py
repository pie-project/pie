"""Greedy text completion, device-carried — the Python twin of `text-completion`."""

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
    pipe = Pipeline()

    # The model's opening (`<bos>` where it has one) before the raw text: a
    # gemma without it answers noise.
    prompt = chat.prefix() + model.encode(prompt_text)
    if not prompt:
        raise ValueError("empty prompt")
    *head, last = prompt
    if head:
        fill = ForwardPass()
        fill.embed(Channel.from_(head, dtype.i32), Channel.from_([0, len(head)], dtype.u32))
        fill.state([ws], StateGeometry.folding())
        pipe.submit(fill)

    tok_in = Channel.from_([last], dtype.i32)
    tok_out = Channel([1], dtype.i32).capacity(channel_capacity())
    fwd = ForwardPass()
    fwd.embed(tok_in, Channel.from_([0, 1], dtype.u32))
    fwd.state([ws], StateGeometry.folding())

    @fwd.epilogue
    def _():
        token = greedy(intrinsics.output(0, "logits"))
        tok_in.put(token)
        tok_out.put(token)

    generated: list[int] = []

    async def on_token() -> bool:
        generated.append(await tok_out.take_scalar())
        return True

    await pipe.run_ahead(fwd, max_tokens, on_token)
    pipe.close()

    return {
        "text": model.decode(generated),
        "count": len(generated),
        "tokens": generated,
    }
