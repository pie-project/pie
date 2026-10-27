"""Prefix-tree KV-cache sharing — the Python twin of `prefix-tree-kv-cache`.

The common prompt is prefilled once. Two first-level branches fork that
working set, append distinct text, and are each forked again into two leaves;
generation then continues independently from all four shared-prefix leaves,
one decode lane per leaf. Exercises `WorkingSet.fork` and `run_ahead` with an
early stop.
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


def append_tokens(ws: WorkingSet, pipeline: Pipeline, tokens: list[int]) -> None:
    # Freezes what it appends, so a later fork shares it instead of copying.
    if not tokens:
        raise ValueError("cannot append an empty token sequence")
    fwd = ForwardPass()
    fwd.embed(Channel.from_(tokens, dtype.i32), Channel.from_([0, len(tokens)], dtype.u32))
    fwd.state([ws], StateGeometry.folding())
    pipeline.submit(fwd)


async def generate(leaves: list[WorkingSet], pipeline: Pipeline, last: list[int], max_tokens: int) -> list[list[int]]:
    # Greedy decode, one lane per leaf, each seeded with its suffix's last token.
    B = len(leaves)
    vocab = model.output_vocab_size()
    stop_tokens = set(chat.stop_tokens())
    token_in = Channel.from_(last, dtype.i32)
    token_out = Channel([B], dtype.i32).capacity(channel_capacity())

    fwd = ForwardPass()
    fwd.embed(token_in, Channel.from_(range(B + 1), dtype.u32))
    fwd.state(leaves, StateGeometry.folding())

    @fwd.epilogue
    def _():
        tokens = reshape(reduce_argmax(reshape(intrinsics.output(0, "logits"), [B, vocab])), [B])
        token_in.put(tokens)
        token_out.put(tokens)

    generated: list[list[int]] = [[] for _ in leaves]
    done = [False] * B

    async def on_token() -> bool:
        for lane, token in enumerate(await token_out.take_host()):
            if done[lane]:
                continue
            if token in stop_tokens:
                done[lane] = True
            else:
                generated[lane].append(token)
        return not all(done)

    await pipeline.run_ahead(fwd, max_tokens, on_token)
    return generated


async def main(input: dict) -> str:
    num_tokens = int(input.get("num_tokens", 32))
    root = WorkingSet()
    pipeline = Pipeline()
    append_tokens(root, pipeline, model.encode("Write a short scene set"))

    first_level = []
    for suffix in (" in a city", " in a forest"):
        child = root.fork(pipeline)
        append_tokens(child, pipeline, model.encode(suffix))
        first_level.append((suffix.strip(), child))

    labels, leaves, last = [], [], []
    for label, parent in first_level:
        for suffix in (" at dawn", " at night"):
            leaf = parent.fork(pipeline)
            tokens = model.encode(suffix)
            if not tokens:
                raise ValueError("empty leaf suffix")
            *head, tail = tokens
            if head:
                append_tokens(leaf, pipeline, head)
            labels.append(f"{label} {suffix.strip()}")
            leaves.append(leaf)
            last.append(tail)

    generated = await generate(leaves, pipeline, last, num_tokens)
    pipeline.close()
    return "\n".join(f"{label}: {model.decode(tokens)}" for label, tokens in zip(labels, generated))
