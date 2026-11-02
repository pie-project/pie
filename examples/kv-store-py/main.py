"""A KV store: a corpus read once, and many questions that each read only the pages they need.

The client sends the corpus as chunks of tokens and, per question, its tokens
and the chunks it keeps (the client ran the retrieval). The program:

- reads the corpus once into one working set, the store;
- forks the store once per question and, with `select`, forgets every chunk the
  question does not keep (`prefix`, the opening tokens, always stays); without
  `select` a fork reads the whole store, the way a full-context request does;
- asks every question as one lane of a single pass and decodes the lanes together.

No chunk is read twice and a fork copies nothing. A question's decode reads its
own chunks only, so what it costs does not grow with the store.
"""

import time

from inferlet import model
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


async def read(ws: WorkingSet, pipe: Pipeline, tokens: list[int], step: int) -> None:
    """Append `tokens` to `ws`, `step` at a time."""
    for at in range(0, len(tokens), step):
        part = tokens[at:at + step]
        fwd = ForwardPass()
        fwd.embed(part)
        fwd.state([ws], StateGeometry.folding())
        done = Channel([1], dtype.i32)

        @fwd.epilogue
        def _():
            done.put(reshape(reduce_argmax(intrinsics.output(0, "logits")), [1]))

        pipe.submit(fwd)
        await done.take_scalar()


async def main(input: dict) -> dict:
    prefix = [int(t) for t in input["prefix"]]
    chunks = [[int(t) for t in c] for c in input["chunks"]]
    questions = input["questions"]
    select, decode = bool(input.get("select", True)), int(input.get("decode", 32))
    stops = set(int(t) for t in input["stop_tokens"])
    lanes = len(questions)

    # the store: the prefix, then every chunk, read once
    bounds, at = [], len(prefix)
    for c in chunks:
        bounds.append((at, at + len(c)))
        at += len(c)
    n = at
    store, pipe = WorkingSet(), Pipeline()
    t0 = time.monotonic()
    await read(store, pipe, prefix + [t for c in chunks for t in c], int(input.get("step", 8192)))
    read_s = time.monotonic() - t0

    # a fork per question; with `select`, it forgets the chunks it does not keep
    t0 = time.monotonic()
    forks = [store.fork(pipe) for _ in range(lanes)]
    kept = []
    for q, fork in zip(questions, forks):
        keep = set(q["keep"]) if select else set(range(len(chunks)))
        start = None
        for i, (a, b) in enumerate(bounds + [(n, n)]):
            dropping = i < len(chunks) and i not in keep
            if dropping and start is None:
                start = a
            elif not dropping and start is not None:
                fork.forget(start, a)
                start = None
        kept.append(len(prefix) + sum(b - a for i, (a, b) in enumerate(bounds) if i in keep))
    fork_s = time.monotonic() - t0

    # every question as one lane of a pass, then the lanes decode together
    v = model.output_vocab_size()
    asks = [[int(t) for t in q["tokens"]] for q in questions]
    indptr = [0]
    for a in asks:
        indptr.append(indptr[-1] + len(a))
    t0 = time.monotonic()
    first = Channel([lanes], dtype.i32)
    ask = ForwardPass()
    ask.embed(Channel.from_([t for a in asks for t in a], dtype.i32), Channel.from_(indptr, dtype.u32))
    ask.state(forks, StateGeometry.folding())

    @ask.epilogue
    def _():
        first.put(reduce_argmax(reshape(intrinsics.output(0, "logits"), [lanes, v])))

    pipe.submit(ask)
    now = await first.take_host()
    ask_s = time.monotonic() - t0

    tok_in = Channel.from_(now, dtype.i32)
    tok_out = Channel([lanes], dtype.i32).capacity(channel_capacity())
    step = ForwardPass()
    step.embed(tok_in, Channel.from_(list(range(lanes + 1)), dtype.u32))
    step.state(forks, StateGeometry.folding())

    @step.epilogue
    def _():
        token = reduce_argmax(reshape(intrinsics.output(0, "logits"), [lanes, v]))
        tok_in.put(token)
        tok_out.put(token)

    out = [[t] for t in now]
    ended = [t in stops for t in now]

    async def on_step() -> bool:
        for lane, token in enumerate(await tok_out.take_host()):
            if not ended[lane]:
                if token in stops:
                    ended[lane] = True
                else:
                    out[lane].append(token)
        return not all(ended)

    t0 = time.monotonic()
    steps = await pipe.run_ahead(step, max(decode - 1, 1), on_step) if not all(ended) else 0
    decode_s = time.monotonic() - t0
    answers = [model.decode([t for t in row if t not in stops]) for row in out]
    return {
        "select": select, "questions": lanes, "store_tokens": n, "chunks": len(chunks),
        "kept_tokens_mean": round(sum(kept) / lanes), "read_s": round(read_s, 3),
        "fork_ms": round(fork_s * 1000, 2), "ask_s": round(ask_s, 3), "decode_s": round(decode_s, 3),
        "answer_s": round(fork_s + ask_s + decode_s, 3), "decode_steps": steps, "answers": answers,
    }
