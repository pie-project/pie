"""Naive text completion — the Python twin of `naive-baseline`.

One N-wide prefill fire, then a device-carried decode loop driven by
`run_ahead`, which keeps the runtime's run-ahead window full ahead of the host
drain. The epilogue temperature-scales the logits and draws a Gumbel-max
sample; `stats` adds the two extra `[1]` f32 drains the algorithm inferlets
carry. Traces to the same container bytes as the Rust inferlet.
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

    if max_tokens == 0:
        return {"sampler": "naive-baseline-py", "text": "", "tokens": [], "count": 0, "stats": want_stats}

    # The model's opening (`<bos>` where it has one) before the raw text.
    prompt = list(chat.prefix()) + list(model.encode(prompt_text))
    if not prompt:
        prompt = [0]
    n = len(prompt)

    generated: list[int] = []
    pipe = Pipeline()

    toks_p = Channel.from_(prompt, dtype.i32).named("toks_p")
    embed_indptr_p = Channel.from_([0, n], dtype.u32).named("embed_indptr_p")
    rng_p = Channel.from_([seed, 0], dtype.u32).named("rng_p")
    tok_out_p = Channel([1], dtype.i32).named("tok_out_p")
    s1_out_p = Channel([1], dtype.f32).named("s1_out_p")
    s2_out_p = Channel([1], dtype.f32).named("s2_out_p")

    fwd_p = ForwardPass()
    fwd_p.embed(toks_p, embed_indptr_p)
    fwd_p.state([ws], StateGeometry.folding())

    @fwd_p.epilogue
    def _prefill_epilogue():
        r = rng_p.take()
        logits = intrinsics.logits()
        token = step(logits, temperature, r)
        r_next = r + iota(2)
        tok_out_p.put(token)
        if want_stats:
            mirror = reshape(cast(token, dtype.f32), [1])
            s1_out_p.put(mirror)
            s2_out_p.put(mirror)
        rng_p.put(r_next)

    pipe.submit(fwd_p)
    g0 = await tok_out_p.take_scalar()
    if want_stats:
        await s1_out_p.take_host()
        await s2_out_p.take_host()
    generated.append(g0)

    # ── DECODE LOOP (1-wide, run-ahead). ──
    if len(generated) < max_tokens:
        cap = channel_capacity()
        tok_in = Channel.from_([g0], dtype.i32).named("tok_in")
        rng = Channel.from_([seed ^ 0x5BD1, 0], dtype.u32).named("rng")
        tok_out = Channel([1], dtype.i32).capacity(cap).named("tok_out")
        s1_out = Channel([1], dtype.f32).capacity(cap).named("s1_out")
        s2_out = Channel([1], dtype.f32).capacity(cap).named("s2_out")
        lane1 = Channel.from_([0, 1], dtype.u32).named("embed_indptr")

        fwd = ForwardPass()
        fwd.embed(tok_in, lane1)
        fwd.state([ws], StateGeometry.folding())

        @fwd.epilogue
        def _decode_epilogue():
            r = rng.take()
            logits = intrinsics.logits()
            token = step(logits, temperature, r)

            r_next = r + iota(2)

            tok_in.put(token)
            tok_out.put(token)
            if want_stats:
                mirror = reshape(cast(token, dtype.f32), [1])
                s1_out.put(mirror)
                s2_out.put(mirror)
            rng.put(r_next)

        budget = max_tokens - 1

        async def on_token() -> bool:
            t = await tok_out.take_scalar()
            if want_stats:
                await s1_out.take_host()
                await s2_out.take_host()
            generated.append(t)
            return True

        await pipe.run_ahead(fwd, budget, on_token)
    pipe.close()

    return {
        "sampler": "naive-baseline-py",
        "text": model.decode(generated),
        "tokens": generated,
        "count": len(generated),
        "stats": want_stats,
    }
