"""Recurrent-state fork probe: a minimal repro for run-to-run divergence after `WorkingSet.fork` on a hybrid model.

The common prompt is prefilled once. Two first-level branches fork that
working set, append distinct text, and are each forked again into two leaves;
generation then continues independently from all four shared-prefix leaves,
one pipeline per leaf.
"""

from inferlet import chat, model
from wit_world.imports import monotonic_clock as _clock
from inferlet.eta import (
    Channel,
    ForwardPass,
    Pipeline,
    StateGeometry,
    WorkingSet,
    channel_capacity,
    dtype,
    intrinsics,
    kv_page_size,
    reduce_argmax,
    reshape,
)


async def append_tokens(ws: WorkingSet, pipeline: Pipeline, tokens: list[int]) -> int:
    if not tokens:
        raise ValueError("cannot append an empty token sequence")
    n = len(tokens)
    token_input = Channel.from_(tokens, dtype.i32)
    embed_indptr = Channel.from_([0, n], dtype.u32).named("embed_indptr")
    next_token = Channel([1], dtype.i32).named("next_token")

    fwd = ForwardPass()
    fwd.embed(token_input, embed_indptr)
    fwd.state([ws], StateGeometry.folding())

    @fwd.epilogue
    def _():
        next_token.put(reshape(reduce_argmax(intrinsics.output(0, "logits")), [1]))

    pipeline.submit(fwd)
    return await next_token.take_scalar()


async def generate(ws: WorkingSet, pipeline: Pipeline, first_token: int, max_tokens: int) -> list[int]:
    if max_tokens == 0:
        return []
    stop_tokens = set(chat.stop_tokens())
    generated: list[int] = []
    if first_token not in stop_tokens:
        generated.append(first_token)
    if len(generated) >= max_tokens or first_token in stop_tokens:
        return generated

    token_in = Channel.from_([first_token], dtype.i32).named("token_in")
    embed_indptr = Channel.from_([0, 1], dtype.u32).named("embed_indptr")
    token_out = Channel([1], dtype.i32).capacity(channel_capacity()).named("token_out")

    fwd = ForwardPass()
    fwd.embed(token_in, embed_indptr)
    fwd.state([ws], StateGeometry.folding())

    @fwd.epilogue
    def _():
        token = reshape(reduce_argmax(intrinsics.output(0, "logits")), [1])
        token_in.put(token)
        token_out.put(token)

    budget = max_tokens - len(generated)

    async def on_token() -> bool:
        token = await token_out.take_scalar()
        if token in stop_tokens:
            return False
        generated.append(token)
        return True

    await pipeline.run_ahead(fwd, budget, on_token)
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
    # append starts a fresh page rather than sharing the root's last one.
    # Separates "partial shared page" from "whole shared pages" when a run is
    # not reproducible.
    align = bool(input.get("align", False))
    # `regen_root`: after the leaves decode, decode from the (still live)
    # ROOT as well. The root's frozen state is shared with every leaf; if a
    # leaf's writes reached it, the root's continuation no longer matches a
    # root-only run (`depth=0`).
    regen_root = bool(input.get("regen_root", False))
    # `settle_ms`: sleep this long after every append before forking/decoding
    # from it. If the divergence disappears, the fork copied rows the parent's
    # fire had not yet committed to the page.
    settle_ms = int(input.get("settle_ms", 0))

    async def settle():
        if settle_ms > 0:
            await _clock.wait_for(settle_ms * 1_000_000)
    suffixes = [" in a city", " in a forest", " at sea", " at dawn"]
    lines = []
    for rep in range(repeats):
        root = WorkingSet()
        tokens = model.encode(input.get("prompt", "Write a short scene set"))
        if align:
            page = kv_page_size()
            filler = model.encode(" and")[-1]
            tokens = tokens + [filler] * ((-len(tokens)) % page)
        pipe = Pipeline()
        first = await append_tokens(root, pipe, tokens)
        await settle()
        level = [(root, first)]
        for _ in range(depth):
            nxt = []
            for ws, _first in level:
                for k in range(leaves):
                    child = ws.fork(pipe)
                    suffix = model.encode(suffixes[k % len(suffixes)])
                    f = await append_tokens(child, pipe, suffix)
                    await settle()
                    nxt.append((child, f))
            level = nxt
        pipe.close()
        outs = []
        for ws, f in level:
            outs.append(await generate(ws, Pipeline(), f, num_tokens))
        if regen_root:
            outs.append(await generate(root, Pipeline(), first, num_tokens))
        lines.append(f"rep{rep}: " + " | ".join(str(o) for o in outs))
    stable = len({line.split(": ", 1)[1] for line in lines}) == 1
    return "\n".join(lines + [f"stable={stable} page_size={kv_page_size()} root_len={len(tokens)}"])
