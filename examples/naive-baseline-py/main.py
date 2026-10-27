"""Naive text completion — the Python twin of `naive-baseline`.

An epilogue-free fill of the prompt head, then a device-carried decode loop
from the prompt's last token, driven by `run_ahead`, which keeps the runtime's
run-ahead window full ahead of the host drain. The epilogue temperature-scales
the logits and draws a Gumbel-max sample; `stats` adds the two extra `[1]` f32
drains the algorithm inferlets carry. Traces to the same container bytes as the Rust inferlet.
"""

from inferlet import chat, model
from inferlet.eta import (
    Channel,
    ForwardPass,
    Pipeline,
    StateGeometry,
    WorkingSet,
    cast,
    channel_capacity,
    dtype,
    gumbel_max,
    intrinsics,
    iota,
    reshape,
)


def step(logits, temperature: float, rng_state):
    """One sampling step: temperature, then a Gumbel-max draw over the vocab."""
    scaled = logits if temperature == 1.0 else logits / temperature
    return gumbel_max(scaled, rng_state)


async def main(input: dict) -> dict:
    prompt_text = input.get("prompt", "Write a short paragraph about naive sampling.")
    temperature = float(input.get("temperature", 1.0))
    max_tokens = int(input.get("max_tokens", 32))
    seed = int(input.get("seed", 0x7CE1))
    want_stats = bool(input.get("stats", False))

    if not (temperature > 0.0) or temperature == float("inf"):
        raise ValueError("temperature must be finite and greater than 0")

    ws = WorkingSet()
    pipe = Pipeline()

    # The model's opening (`<bos>` where it has one) before the raw text.
    prompt = list(chat.prefix()) + list(model.encode(prompt_text))
    if not prompt:
        raise ValueError("empty prompt")
    *head, last = prompt
    if head:
        fill = ForwardPass()
        fill.embed(Channel.from_(head, dtype.i32), Channel.from_([0, len(head)], dtype.u32))
        fill.state([ws], StateGeometry.folding())
        pipe.submit(fill)

    cap = channel_capacity()
    tok_in = Channel.from_([last], dtype.i32)
    rng = Channel.from_([seed, 0], dtype.u32)
    tok_out = Channel([1], dtype.i32).capacity(cap)
    s1_out = Channel([1], dtype.f32).capacity(cap)
    s2_out = Channel([1], dtype.f32).capacity(cap)

    fwd = ForwardPass()
    fwd.embed(tok_in, Channel.from_([0, 1], dtype.u32))
    fwd.state([ws], StateGeometry.folding())

    @fwd.epilogue
    def _():
        r = rng.take()
        logits = intrinsics.output(0, "logits")
        token = step(logits, temperature, r)
        r_next = r + iota(2)
        tok_in.put(token)
        tok_out.put(token)
        if want_stats:
            mirror = reshape(cast(token, dtype.f32), [1])
            s1_out.put(mirror)
            s2_out.put(mirror)
        rng.put(r_next)

    generated: list[int] = []

    async def on_token() -> bool:
        t = await tok_out.take_scalar()
        if want_stats:
            await s1_out.take_host()
            await s2_out.take_host()
        generated.append(t)
        return True

    await pipe.run_ahead(fwd, max_tokens, on_token)
    pipe.close()

    return {
        "sampler": "naive-baseline-py",
        "text": model.decode(generated),
        "tokens": generated,
        "count": len(generated),
        "stats": want_stats,
    }
