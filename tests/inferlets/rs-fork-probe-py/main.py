"""Recurrent-state fork probe: a minimal repro for run-to-run divergence after `RsWorkingSet.fork` on a hybrid model.

The common prompt is prefilled once. Two first-level branches fork that
working set (copy-on-write), append distinct text, and are each forked again
into two leaves; generation then continues independently from all four
shared-prefix leaves, one pipeline per leaf. Exercises `WorkingSet.fork`,
`page_len`/incremental `reserve`, multiple pipelines, and `run_ahead` with an
early stop — and, on a hybrid model, `RsWorkingSet.fork` beside it.
"""

from inferlet import chat, model
from wit_world.imports import monotonic_clock as _clock
from inferlet.eta import (
    Channel,
    ForwardKind,
    ForwardPass,
    KvGeometry,
    Pipeline,
    RsWorkingSet,
    WorkingSet,
    channel_capacity,
    dtype,
    indptr,
    intrinsics,
    kv_page_size,
    reduce_argmax,
    reshape,
    run_ahead,
)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


async def append_tokens(ws: WorkingSet, rs: list, pipeline: Pipeline, start: int, tokens: list[int]) -> int:
    if not tokens:
        raise ValueError("cannot append an empty token sequence")
    n = len(tokens)
    total = start + n
    page_size = kv_page_size()
    # Extend the (purely logical) lease so it covers the appended extent.
    max_pages = max(_ceil_div(total, page_size), 1)
    have = ws.page_len()
    if max_pages > have:
        ws.reserve(max_pages - have)
    token_input = Channel.from_(tokens, dtype.i32)
    embed_indptr = Channel.from_([0, n], dtype.u32).named("embed_indptr")
    positions = Channel.from_(range(start, total), dtype.u32).named("positions")
    pages = Channel.from_(range(ws.page_len()), dtype.u32).named("pages")
    page_indptr = Channel.from_([0, _ceil_div(total, page_size)], dtype.u32).named("page_indptr")
    w_slot = Channel.from_([p // page_size for p in range(start, total)], dtype.u32).named("w_slot")
    w_off = Channel.from_([p % page_size for p in range(start, total)], dtype.u32).named("w_off")
    next_token = Channel([1], dtype.i32).named("next_token")
    kv_len = Channel.from_([total], dtype.u32).named("kv_len")

    fwd = ForwardPass()
    fwd.embed(token_input, embed_indptr)
    fwd.bind_state(
        ws,
        KvGeometry(
            kv_len=kv_len,
            pages=pages,
            page_indptr=page_indptr,
            w_slot=w_slot,
            w_off=w_off,
            positions=positions,
            writable_pages=start // page_size,
        ),
        rs,
    )

    @fwd.epilogue
    def _():
        next_token.put(reshape(reduce_argmax(intrinsics.logits()), [1]))

    fwd.submit(pipeline)
    return await next_token.take_scalar()


async def generate(ws: WorkingSet, rs: list, pipeline: Pipeline, seq_len: int, first_token: int, max_tokens: int) -> list[int]:
    if max_tokens == 0:
        return []
    stop_tokens = set(chat.stop_tokens())
    generated: list[int] = []
    if first_token not in stop_tokens:
        generated.append(first_token)
    if len(generated) >= max_tokens or first_token in stop_tokens:
        return generated

    page_size = kv_page_size()
    max_pages = max(_ceil_div(seq_len + max_tokens + 1, page_size), 1)
    have = ws.page_len()
    if max_pages > have:
        ws.reserve(max_pages - have)
    token_in = Channel.from_([first_token], dtype.i32).named("token_in")
    embed_indptr = Channel.from_([0, 1], dtype.u32).named("embed_indptr")
    positions = Channel.from_([seq_len], dtype.u32).named("positions")
    pages = Channel.from_(range(max_pages), dtype.u32).named("pages")
    page_indptr = Channel.from_([0, _ceil_div(seq_len + 1, page_size)], dtype.u32).named("page_indptr")
    w_slot = Channel.from_([seq_len // page_size], dtype.u32).named("w_slot")
    w_off = Channel.from_([seq_len % page_size], dtype.u32).named("w_off")
    token_out = Channel([1], dtype.i32).capacity(channel_capacity()).named("token_out")
    kv_len = Channel.from_([seq_len + 1], dtype.u32).named("kv_len")

    fwd = ForwardPass()
    fwd.embed(token_in, embed_indptr)
    fwd.bind_state(
        ws,
        KvGeometry(
            kv_len=kv_len,
            pages=pages,
            page_indptr=page_indptr,
            w_slot=w_slot,
            w_off=w_off,
            positions=positions,
            writable_pages=seq_len // page_size,
        ),
        rs,
    )

    @fwd.epilogue
    def _():
        length = kv_len.take()
        token = reshape(reduce_argmax(intrinsics.logits()), [1])
        next_length = length + 1
        page_count = next_length.div_ceil(page_size)
        token_in.put(token)
        kv_len.put(next_length)
        positions.put(length)
        w_slot.put(length // page_size)
        w_off.put(length % page_size)
        page_indptr.put(indptr(1, page_count))
        token_out.put(token)

    budget = max_tokens - len(generated)

    async def on_token() -> bool:
        token = await token_out.take_scalar()
        if token in stop_tokens:
            return False
        generated.append(token)
        return True

    await run_ahead(pipeline, fwd, budget, on_token)
    return generated


async def main(input: dict) -> str:
    """Root prompt → `depth` levels of `leaves` forks, each appending a
    suffix → greedy decode at every leaf for `num_tokens`, the whole tree
    `repeats` times, reporting whether the tree reproduces itself.
    `depth=1` forks once and decodes; `depth=2` is the prefix-tree shape
    (fork, append, fork again, append, decode)."""
    num_tokens = int(input.get("num_tokens", 6))
    leaves = int(input.get("leaves", 2))
    depth = int(input.get("depth", 1))
    repeats = int(input.get("repeats", 2))
    # `align`: pad the root prompt to a KV page boundary, so a child's first
    # append lands on a FRESH page (no copy-on-write of a shared page; the
    # shared prefix pages are read-only). Separates "CoW copy" from "shared
    # page read" when a run is not reproducible.
    align = bool(input.get("align", False))
    # `regen_root`: after the leaves decode, decode from the (still live)
    # ROOT as well. The root's pages are shared read-only with every leaf;
    # if a leaf's copy-on-write freed or aliased one of them, the root's
    # continuation no longer matches a root-only run (`depth=0`).
    regen_root = bool(input.get("regen_root", False))
    # `settle_ms`: sleep this long after every append before forking/decoding
    # from it. If the divergence disappears, the fork copied rows the parent's
    # fire had not yet committed to the page.
    settle_ms = int(input.get("settle_ms", 0))

    async def settle():
        if settle_ms > 0:
            await _clock.wait_for(settle_ms * 1_000_000)
    hybrid = model.pass_kind() != ForwardKind.ATTENTION
    suffixes = [" in a city", " in a forest", " at sea", " at dawn"]
    lines = []
    for rep in range(repeats):
        root = WorkingSet()
        root_rs = [RsWorkingSet()] if hybrid else []
        tokens = model.encode(input.get("prompt", "Write a short scene set"))
        if align:
            page = kv_page_size()
            filler = model.encode(" and")[-1]
            tokens = tokens + [filler] * ((-len(tokens)) % page)
        pipe = Pipeline()
        first = await append_tokens(root, root_rs, pipe, 0, tokens)
        await settle()
        level = [(root, root_rs, len(tokens), first)]
        for _ in range(depth):
            nxt = []
            for ws, rs, seq_len, _first in level:
                for k in range(leaves):
                    child, child_rs = ws.fork(pipe), [r.fork(pipe) for r in rs]
                    suffix = model.encode(suffixes[k % len(suffixes)])
                    f = await append_tokens(child, child_rs, pipe, seq_len, suffix)
                    await settle()
                    nxt.append((child, child_rs, seq_len + len(suffix), f))
            level = nxt
        pipe.close()
        outs = []
        for ws, rs, seq_len, f in level:
            outs.append(await generate(ws, rs, Pipeline(), seq_len, f, num_tokens))
        if regen_root:
            outs.append(await generate(root, root_rs, Pipeline(), len(tokens), first, num_tokens))
        lines.append(f"rep{rep}: " + " | ".join(str(o) for o in outs))
    stable = len({line.split(": ", 1)[1] for line in lines}) == 1
    return "\n".join(lines + [f"stable={stable} page_size={kv_page_size()} root_len={len(tokens)}"])
