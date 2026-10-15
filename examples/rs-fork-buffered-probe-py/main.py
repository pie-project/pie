"""Recurrent-state fork probe with rows left in the buffer (#860).

Forks an `RsWorkingSet` while rows sit unfolded in its buffer.

Prefill a prompt (fold everything), decode `at` tokens one at a time with
buffered steps (each fire folds the previous step's one survivor and buffers
its own row), then at that point:

  none    do nothing
  fork    fork the kv and rs working sets and keep the forks alive
  settle  fold the buffer empty (one fire, fold = survivors + 1), then fork
  child   fork, then continue on the child (parent dropped)

and decode `tail` more tokens greedily. Returns the tail's token ids.
"""

from inferlet import chat, model
from inferlet.eta import (
    Channel,
    ForwardKind,
    ForwardPass,
    KvBinding,
    KvGeometry,
    Pipeline,
    RsGeometry,
    RsWorkingSet,
    WorkingSet,
    dtype,
    intrinsics,
    kv_page_size,
    reduce_argmax,
    reshape,
)


def _ceil(a: int, b: int) -> int:
    return -(-a // b)


async def fire(ws, rs, pipe, tokens, base, fold, buffer_pages):
    n = len(tokens)
    total = base + n
    ps = kv_page_size()
    need = max(_ceil(total, ps), 1)
    if need > ws.page_len():
        ws.reserve(need - ws.page_len())
    ids = Channel.from_(tokens, dtype.i32)
    embed_indptr = Channel.from_([0, n], dtype.u32).named("embed_indptr")
    positions = Channel.from_(range(base, total), dtype.u32).named("positions")
    pages = Channel.from_(range(ws.page_len()), dtype.u32).named("pages")
    page_indptr = Channel.from_([0, _ceil(total, ps)], dtype.u32).named("page_indptr")
    w_slot = Channel.from_([p // ps for p in range(base, total)], dtype.u32).named("w_slot")
    w_off = Channel.from_([p % ps for p in range(base, total)], dtype.u32).named("w_off")
    kv_len = Channel.from_([total], dtype.u32).named("kv_len")
    out = Channel([1], dtype.i32).named("next_token")
    fold_len = None if fold is None else Channel.from_([fold], dtype.u32).named("fold_len")

    fwd = ForwardPass()
    fwd.embed(ids, embed_indptr)
    fwd.bind_hybrid(
        KvBinding(ws, KvGeometry(kv_len=kv_len, pages=pages, page_indptr=page_indptr, w_slot=w_slot,
                                 w_off=w_off, positions=positions, writable_pages=base // ps)),
        rs,
        RsGeometry(fold_len=fold_len, buffer=(0, buffer_pages)),
    )

    @fwd.epilogue
    def _():
        out.put(reshape(reduce_argmax(intrinsics.logits()), [1]))

    pipe.submit(fwd)
    return await out.take_scalar()


class Seq:
    def __init__(self, ws, rs, base, pending, survivors):
        self.ws, self.rs, self.base, self.pending, self.survivors = ws, rs, base, pending, survivors

    def pages_for(self, rows):
        page = self.rs[0].buffer_page_size()
        need = max(_ceil(page - 1 + self.survivors + rows, page), 1)
        have = self.rs[0].buffer_size()
        if need > have:
            self.rs[0].alloc_buffer(need - have)
        return need

    async def step(self, pipe):
        """One buffered decode step: fold the survivors, buffer this row."""
        need = self.pages_for(1)
        token = self.pending
        nxt = await fire(self.ws, self.rs, pipe, [token], self.base, self.survivors, need)
        self.base += 1
        self.survivors = 1
        self.pending = nxt
        return token

    async def settle(self, pipe):
        """Fold everything buffered and the pending row: the buffer ends empty."""
        need = self.pages_for(1)
        nxt = await fire(self.ws, self.rs, pipe, [self.pending], self.base, self.survivors + 1, need)
        self.base += 1
        self.survivors = 0
        tok = self.pending
        self.pending = nxt
        return tok

    def fork(self, pipe):
        return Seq(self.ws.fork(pipe), [r.fork(pipe) for r in self.rs], self.base, self.pending, self.survivors)


async def main(input: dict) -> str:
    if model.pass_kind() != ForwardKind.HYBRID:
        return "not hybrid"
    at = int(input.get("at", 33))
    tail = int(input.get("tail", 48))
    variant = input.get("variant", "none")
    prompt = input.get("prompt", "Explain how a refrigerator keeps food cold, step by step.")

    pipe = Pipeline()
    ws, rs = WorkingSet(), [RsWorkingSet()]
    tokens = chat.prefix() + model.encode(prompt)
    first = await fire(ws, rs, pipe, tokens, 0, None, 0)
    seq = Seq(ws, rs, len(tokens), first, 0)
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
