"""Greedy text completion, host-driven — the Python twin of `text-completion`.

Same program as `tests/inferlets/text-completion/src/lib.rs`, traced from
Python: one chunked prefill, then a 1-wide decode loop whose ONE host-driven
channel is the token (`reduce_argmax` on the device, back to the host, down
again as a host-writer cell). Everything else the fire reads is derived from
the KV length by device arithmetic in the epilogue. The traced container is
byte-identical to the Rust inferlet's, so both share one program-cache entry.
"""

from inferlet import model
from inferlet.eta import (
    Channel,
    ForwardKind,
    ForwardPass,
    KvGeometry,
    Pipeline,
    RsWorkingSet,
    WorkingSet,
    dtype,
    indptr,
    intrinsics,
    kv_page_size,
    prefill_chunks,
    reduce_argmax,
    reshape,
)


def greedy(logits):
    """The greedy pick over a logits row, as a one-lane `[1]` i32 cell."""
    return reshape(reduce_argmax(logits), [1])


async def main(input: dict) -> dict:
    prompt_text = input.get("prompt", "The capital of France is")
    max_tokens = int(input.get("max_tokens", 8))

    kind = model.pass_kind()
    if kind == ForwardKind.RECURRENT:
        raise RuntimeError("this program has no recurrent-only path")

    ws = WorkingSet()
    rs_ws = [RsWorkingSet()] if kind != ForwardKind.ATTENTION else []
    page_size = kv_page_size()

    if max_tokens == 0:
        return {"text": "", "count": 0, "tokens": []}

    prompt = model.encode(prompt_text)
    if not prompt:
        prompt = [0]
    n = len(prompt)
    max_pages = max(-(-(n + max_tokens + 1) // page_size), 1)
    ws.reserve(max_pages)

    pipe = Pipeline()
    generated: list[int] = []

    # ── PREFILL (chunked, C-wide) ─────────────────────────────────────────
    first = 0
    for base, end in prefill_chunks(n):
        length = end - base
        toks = Channel.from_(prompt[base:end], dtype.i32).named("toks_p")
        embed_indptr = Channel.from_([0, length], dtype.u32).named("embed_indptr_p")
        positions = Channel.from_(range(base, end), dtype.u32).named("positions_p")
        pages = Channel.from_(range(max_pages), dtype.u32).named("pages_p")
        page_indptr = Channel.from_([0, -(-end // page_size)], dtype.u32).named("page_indptr_p")
        w_slot = Channel.from_([p // page_size for p in range(base, end)], dtype.u32).named("w_slot_p")
        w_off = Channel.from_([p % page_size for p in range(base, end)], dtype.u32).named("w_off_p")
        kv_len = Channel.from_([end], dtype.u32).named("kv_len_p")
        tok_out = Channel([1], dtype.i32).named("tok_out_p")

        fwd = ForwardPass(kind)
        fwd.embed(toks, embed_indptr)
        fwd.bind_state(
            ws,
            KvGeometry(
                kv_len=kv_len,
                pages=pages,
                page_indptr=page_indptr,
                w_slot=w_slot,
                w_off=w_off,
                positions=positions,
            ),
            rs_ws,
        )

        @fwd.epilogue
        def _prefill_epilogue():
            tok_out.put(greedy(intrinsics.logits()))

        fwd.submit(pipe)
        # Every chunk samples and every sample must be drained.
        first = await tok_out.take_scalar()
    generated.append(first)

    # ── DECODE (1-wide, host-driven token) ────────────────────────────────
    if len(generated) < max_tokens:
        tok_in = Channel.from_([first], dtype.i32).named("tok_in")
        embed_indptr = Channel.from_([0, 1], dtype.u32).named("embed_indptr")
        positions = Channel.from_([n], dtype.u32).named("positions")
        pages = Channel.from_(range(max_pages), dtype.u32).named("pages")
        page_indptr = Channel.from_([0, -(-(n + 1) // page_size)], dtype.u32).named("page_indptr")
        w_slot = Channel.from_([n // page_size], dtype.u32).named("w_slot")
        w_off = Channel.from_([n % page_size], dtype.u32).named("w_off")
        kv_len = Channel.from_([n + 1], dtype.u32).named("kv_len")
        tok_out = Channel([1], dtype.i32).named("tok_out")

        fwd = ForwardPass(kind)
        fwd.embed(tok_in, embed_indptr)
        fwd.bind_state(
            ws,
            KvGeometry(
                kv_len=kv_len,
                pages=pages,
                page_indptr=page_indptr,
                w_slot=w_slot,
                w_off=w_off,
                positions=positions,
            ),
            rs_ws,
        )

        @fwd.epilogue
        def _decode_epilogue():
            # `length` is the readable extent this fire runs at, so it is
            # also the position the NEXT fire's token sits at.
            length = kv_len.take()
            next_length = length + 1
            page_count = next_length.div_ceil(page_size)
            kv_len.put(next_length)
            positions.put(length)
            w_slot.put(length // page_size)
            w_off.put(length % page_size)
            page_indptr.put(indptr(1, page_count))
            tok_out.put(greedy(intrinsics.logits()))

        while True:
            fwd.submit(pipe)
            token = await tok_out.take_scalar()
            generated.append(token)
            if len(generated) >= max_tokens:
                break
            tok_in.put([token])
    pipe.close()

    return {
        "text": model.decode(generated),
        "count": len(generated),
        "tokens": generated,
    }
