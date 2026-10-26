"""Prefix-tree KV-cache sharing — the Python twin of `prefix-tree-kv-cache`.

The common prompt is prefilled once. Two first-level branches fork that
working set, append distinct text, and are each forked again into two leaves;
generation then continues independently from all four shared-prefix leaves,
one pipeline per leaf. Exercises `WorkingSet.fork`, multiple pipelines, and
`run_ahead` with an early stop.
"""

from inferlet import chat, model
from inferlet.eta import (
    Channel,
    ForwardPass,
    Pipeline,
    StateGeometry,
    WorkingSet,
    channel_capacity,
    dtype,
    intrinsics,
    reduce_argmax,
    reshape,
)


async def append_tokens(ws: WorkingSet, pipeline: Pipeline, tokens: list[int]) -> int:
    # Freezes what it appends, so a later fork shares it instead of copying.
    if not tokens:
        raise ValueError("cannot append an empty token sequence")
    n = len(tokens)
    token_input = Channel.from_(tokens, dtype.i32)
    embed_indptr = Channel.from_([0, n], dtype.u32)
    next_token = Channel([1], dtype.i32)

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

    token_in = Channel.from_([first_token], dtype.i32)
    embed_indptr = Channel.from_([0, 1], dtype.u32)
    token_out = Channel([1], dtype.i32).capacity(channel_capacity())

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
    num_tokens = int(input.get("num_tokens", 32))
    root = WorkingSet()

    root_tokens = model.encode("Write a short scene set")
    if not root_tokens:
        raise ValueError("tokenizer produced an empty root prompt")

    tree_pipeline = Pipeline()
    await append_tokens(root, tree_pipeline, root_tokens)

    first_level = []
    for suffix in (" in a city", " in a forest"):
        child = root.fork(tree_pipeline)
        await append_tokens(child, tree_pipeline, model.encode(suffix))
        first_level.append((suffix.strip(), child))

    leaves = []
    for label, parent in first_level:
        for suffix in (" at dawn", " at night"):
            leaf = parent.fork(tree_pipeline)
            first = await append_tokens(leaf, tree_pipeline, model.encode(suffix))
            leaves.append((f"{label} {suffix.strip()}", leaf, first))
    # The BUILD stream ends here.
    tree_pipeline.close()

    # One pipeline per leaf: each generation is its own sequential stream.
    outputs = []
    for label, ws, first in leaves:
        leaf_pipeline = Pipeline()
        generated = await generate(ws, leaf_pipeline, first, num_tokens)
        outputs.append(f"{label}: {model.decode(generated)}")
    return "\n".join(outputs)
