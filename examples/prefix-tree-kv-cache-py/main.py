"""Prefix-tree KV-cache sharing — the Python twin of `prefix-tree-kv-cache`.

The common prompt is prefilled once. Two first-level branches fork that
working set (copy-on-write), append distinct text, and are each forked again
into two leaves; generation then continues independently from all four
shared-prefix leaves, one pipeline per leaf. Exercises `WorkingSet.fork`,
`page_len`/incremental `reserve`, multiple pipelines, and `run_ahead` with an
early stop — and, on a hybrid model, `RsWorkingSet.fork` beside it.
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
    num_tokens = int(input.get("num_tokens", 32))
    hybrid = model.pass_kind() != ForwardKind.ATTENTION
    root = WorkingSet()
    root_rs = [RsWorkingSet()] if hybrid else []

    root_tokens = model.encode("Write a short scene set")
    if not root_tokens:
        raise ValueError("tokenizer produced an empty root prompt")

    tree_pipeline = Pipeline()
    await append_tokens(root, root_rs, tree_pipeline, 0, root_tokens)
    root_len = len(root_tokens)

    def fork(ws: WorkingSet, rs: list) -> tuple[WorkingSet, list]:
        return ws.fork(tree_pipeline), [r.fork(tree_pipeline) for r in rs]

    first_level = []
    for suffix in (" in a city", " in a forest"):
        child, child_rs = fork(root, root_rs)
        tokens = model.encode(suffix)
        await append_tokens(child, child_rs, tree_pipeline, root_len, tokens)
        first_level.append((suffix.strip(), child, child_rs, root_len + len(tokens)))

    leaves = []
    for label, parent, parent_rs, seq_len in first_level:
        for suffix in (" at dawn", " at night"):
            leaf, leaf_rs = fork(parent, parent_rs)
            tokens = model.encode(suffix)
            first = await append_tokens(leaf, leaf_rs, tree_pipeline, seq_len, tokens)
            leaves.append((f"{label} {suffix.strip()}", leaf, leaf_rs, seq_len + len(tokens), first))
    # The BUILD stream ends here.
    tree_pipeline.close()

    # One pipeline per leaf: each generation is its own sequential stream.
    outputs = []
    for label, ws, rs, seq_len, first in leaves:
        leaf_pipeline = Pipeline()
        generated = await generate(ws, rs, leaf_pipeline, seq_len, first, num_tokens)
        outputs.append(f"{label}: {model.decode(generated)}")
    return "\n".join(outputs)
