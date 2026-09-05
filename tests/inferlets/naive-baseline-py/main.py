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
    ForwardKind,
    ForwardPass,
    KvGeometry,
    Pipeline,
    RsWorkingSet,
    WorkingSet,
    cast,
    channel_capacity,
    dtype,
    gumbel_max,
    indptr,
    intrinsics,
    iota,
    kv_page_size,
    prefill_chunks,
    reshape,
    run_ahead,
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
    max_layers = input.get("max_layers")

    if not (temperature > 0.0) or temperature == float("inf"):
        raise ValueError("temperature must be finite and greater than 0")

    ws = WorkingSet()
    # One recurrent working set for the whole generation on a hybrid model
    # (the engine requires one per request row); none on a pure-attention one.
    rs_ws = [RsWorkingSet()] if model.pass_kind() != ForwardKind.ATTENTION else []
    page_size = kv_page_size()

    if max_tokens == 0:
        return {"sampler": "naive-baseline-py", "text": "", "tokens": [], "count": 0, "stats": want_stats}

    # The model's opening (`<bos>` where it has one) before the raw text.
    prompt = list(chat.prefix()) + list(model.encode(prompt_text))
    if not prompt:
        prompt = [0]
    n = len(prompt)
    max_pages = max(-(-(n + max_tokens + 1) // page_size), 1)
    ws.reserve(max_pages)

    generated: list[int] = []
    pipe = Pipeline()

    # ── PREFILL (chunked, C-wide): first sampled token comes off the prompt.
    g0 = 0
    for base, end in prefill_chunks(n):
        length = end - base
        toks_p = Channel.from_(prompt[base:end], dtype.i32).named("toks_p")
        embed_indptr_p = Channel.from_([0, length], dtype.u32).named("embed_indptr_p")
        positions_p = Channel.from_(range(base, end), dtype.u32).named("positions_p")
        pages_p = Channel.from_(range(max_pages), dtype.u32).named("pages_p")
        page_indptr_p = Channel.from_([0, -(-end // page_size)], dtype.u32).named("page_indptr_p")
        w_slot_p = Channel.from_([p // page_size for p in range(base, end)], dtype.u32).named("w_slot_p")
        w_off_p = Channel.from_([p % page_size for p in range(base, end)], dtype.u32).named("w_off_p")
        kv_len_p = Channel.from_([end], dtype.u32).named("kv_len_p")
        rng_p = Channel.from_([seed, 0], dtype.u32).named("rng_p")
        tok_out_p = Channel([1], dtype.i32).named("tok_out_p")
        s1_out_p = Channel([1], dtype.f32).named("s1_out_p")
        s2_out_p = Channel([1], dtype.f32).named("s2_out_p")

        fwd_p = ForwardPass()
        if max_layers is not None:
            fwd_p.set_max_layers(int(max_layers))
        fwd_p.embed(toks_p, embed_indptr_p)
        fwd_p.bind_state(
            ws,
            KvGeometry(
                kv_len=kv_len_p,
                pages=pages_p,
                page_indptr=page_indptr_p,
                w_slot=w_slot_p,
                w_off=w_off_p,
                positions=positions_p,
            ),
            rs_ws,
        )

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

        fwd_p.submit(pipe)
        # Every chunk samples; only the last chunk's token continues the
        # prompt. The intermediate takes cannot be skipped.
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
        positions = Channel.from_([n], dtype.u32).named("positions")
        pages = Channel.from_(range(max_pages), dtype.u32).named("pages")
        page_indptr = Channel.from_([0, -(-(n + 1) // page_size)], dtype.u32).named("page_indptr")
        w_slot = Channel.from_([n // page_size], dtype.u32).named("w_slot")
        w_off = Channel.from_([n % page_size], dtype.u32).named("w_off")
        kv_len = Channel.from_([n + 1], dtype.u32).named("kv_len")

        fwd = ForwardPass()
        if max_layers is not None:
            fwd.set_max_layers(int(max_layers))
        fwd.embed(tok_in, lane1)
        fwd.bind_state(
            ws,
            KvGeometry(
                kv_len=kv_len,
                pages=pages,
                page_indptr=page_indptr,
                w_slot=w_slot,
                w_off=w_off,
                positions=positions,
                writable_pages=n // page_size,
            ),
            rs_ws,
        )

        @fwd.epilogue
        def _decode_epilogue():
            length = kv_len.take()
            r = rng.take()
            logits = intrinsics.logits()
            token = step(logits, temperature, r)

            r_next = r + iota(2)
            next_length = length + 1
            page_count = next_length.div_ceil(page_size)

            tok_in.put(token)
            kv_len.put(next_length)
            positions.put(length)
            w_slot.put(length // page_size)
            w_off.put(length % page_size)
            page_indptr.put(indptr(1, page_count))
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

        await run_ahead(pipe, fwd, budget, on_token)
    pipe.close()

    return {
        "sampler": "naive-baseline-py",
        "text": model.decode(generated),
        "tokens": generated,
        "count": len(generated),
        "stats": want_stats,
    }
