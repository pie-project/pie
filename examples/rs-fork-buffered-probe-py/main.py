"""Fork probe with rows left unfrozen (#860).

Forks a hybrid model's working set while rows it wrote are not yet frozen
into its recurrent state.

Prefill a prompt (freeze everything), decode `at` tokens one at a time (each
fire freezes the last fire's row and leaves its own unfrozen), then at that
point:

  none    do nothing
  fork    fork the working set and keep the fork alive
  settle  freeze every unfrozen row (one fire, fold = survivors + 1), then fork
  child   fork, then continue on the child (parent dropped)

and decode `tail` more tokens greedily. Returns the tail's token ids.
"""

from inferlet import chat, model
from inferlet.eta import (
    Channel,
    ForwardPass,
    Pipeline,
    StateGeometry,
    WorkingSet,
    dtype,
    intrinsics,
    reduce_argmax,
    reshape,
)


async def fire(ws, pipe, tokens, fold):
    n = len(tokens)
    ids = Channel.from_(tokens, dtype.i32)
    embed_indptr = Channel.from_([0, n], dtype.u32).named("embed_indptr")
    out = Channel([1], dtype.i32).named("next_token")
    fold = "all" if fold is None else Channel.from_([fold], dtype.u32).named("fold_len")

    fwd = ForwardPass()
    fwd.embed(ids, embed_indptr)
    fwd.state([ws], StateGeometry(fold=fold))

    @fwd.epilogue
    def _():
        out.put(reshape(reduce_argmax(intrinsics.output(0, "logits")), [1]))

    pipe.submit(fwd)
    return await out.take_scalar()


class Seq:
    def __init__(self, ws, pending, survivors):
        self.ws, self.pending, self.survivors = ws, pending, survivors

    async def step(self, pipe):
        """One decode step: freeze the last row, leave this one unfrozen."""
        token = self.pending
        self.pending = await fire(self.ws, pipe, [token], self.survivors)
        self.survivors = 1
        return token

    async def settle(self, pipe):
        """Freeze the unfrozen rows and the pending one."""
        token = self.pending
        self.pending = await fire(self.ws, pipe, [token], self.survivors + 1)
        self.survivors = 0
        return token

    def fork(self, pipe):
        return Seq(self.ws.fork(pipe), self.pending, self.survivors)


async def main(input: dict) -> str:
    if model.rs_state_size() == 0:
        return "not hybrid"
    at = int(input.get("at", 33))
    tail = int(input.get("tail", 48))
    variant = input.get("variant", "none")
    prompt = input.get("prompt", "Explain how a refrigerator keeps food cold, step by step.")

    pipe = Pipeline()
    ws = WorkingSet()
    tokens = chat.prefix() + model.encode(prompt)
    seq = Seq(ws, await fire(ws, pipe, tokens, None), 0)
    for _ in range(at):
        await seq.step(pipe)

    keep, out = [], []
    if variant == "fork":
        keep.append(seq.fork(pipe))
    elif variant == "settle":
        out.append(await seq.settle(pipe))
        keep.append(seq.fork(pipe))
    elif variant == "child":
        seq = seq.fork(pipe)

    while len(out) < tail:
        out.append(await seq.step(pipe))
    return " ".join(str(t) for t in out)
