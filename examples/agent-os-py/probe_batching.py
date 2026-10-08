"""Kernel probe: batched lanes over one arena of KV pages.

One WorkingSet is the arena. A shared prefix occupies its first P pages
(page-aligned, written once). Every lane owns a private run of tail pages
after it. A decode "wave" is ONE forward pass with one row per lane: each
row lists [shared prefix pages + its own tail pages], so a fork is just two
rows naming the same page ids -- no copy, no re-prefill. Hybrid models give
each lane its own RsWorkingSet, forked from the prefix's.

Checks (1) a 1-lane wave equals a plain greedy decode, (2) B-lane waves give
the same tokens per lane as B separate single-lane runs (batching is exact),
(3) aggregate tok/s as B grows.
"""

import time

from inferlet import chat, model, session
from inferlet.eta import (
    Channel, ForwardKind, ForwardPass, KvGeometry, Pipeline, RsWorkingSet,
    WorkingSet, channel_capacity, dtype, gather, gumbel_max,
    intrinsics, iota, kv_page_size, reduce_argmax, reshape,
)

PS = kv_page_size()
_T0 = time.perf_counter()


def say(msg):
    session.send(f"[{time.perf_counter() - _T0:7.2f}s] {msg}")


def cdiv(a, b):
    return -(-a // b)


class Arena:
    """One WorkingSet; page ids [0, P) are the shared prefix, then lane tails."""

    def __init__(self, prefix_pages, max_lanes, tail_pages):
        self.ws = WorkingSet()
        g = self.ws.reserve(prefix_pages + max_lanes * tail_pages)
        self.base = g.start
        self.P = prefix_pages
        self.tail_pages = tail_pages

    def row_pages(self, slot, end):
        """Page ids a row needs to cover positions [0, end)."""
        need = cdiv(end, PS)
        ids = [self.base + i for i in range(min(need, self.P))]
        if need > self.P:
            t0 = self.base + self.P + slot * self.tail_pages
            ids += [t0 + i for i in range(need - self.P)]
        return ids


async def feed(arena, rs, pipe, slot, start, tokens, seed, temperature, writable=None):
    """Run `tokens` at positions [start, start+m) for one lane/row; return the sampled next token."""
    m = len(tokens)
    end = start + m
    pages = arena.row_pages(slot, end)
    ch_tok = Channel.from_(tokens, dtype.i32)
    ch_ind = Channel.from_([0, m], dtype.u32).named("embed_indptr")
    ch_pos = Channel.from_(range(start, end), dtype.u32).named("positions")
    ch_pages = Channel.from_(pages, dtype.u32).named("pages")
    ch_pind = Channel.from_([0, len(pages)], dtype.u32).named("page_indptr")
    ch_ws = Channel.from_([pages[p // PS] for p in range(start, end)], dtype.u32).named("w_slot")
    ch_wo = Channel.from_([p % PS for p in range(start, end)], dtype.u32).named("w_off")
    ch_kv = Channel.from_([end], dtype.u32).named("kv_len")
    rng = Channel.from_([seed, 0], dtype.u32).named("rng_p")
    out = Channel([1], dtype.i32).named("next_token")
    fwd = ForwardPass()
    fwd.embed(ch_tok, ch_ind)
    fwd.bind_state(
        arena.ws,
        KvGeometry(kv_len=ch_kv, pages=ch_pages, page_indptr=ch_pind, w_slot=ch_ws, w_off=ch_wo,
                   positions=ch_pos, writable_pages=writable),
        rs,
    )

    @fwd.epilogue
    def _():
        r = rng.take()
        lg = intrinsics.logits()
        sc = lg if temperature == 1.0 else lg / temperature
        out.put(reshape(gumbel_max(sc, r), [1]) if temperature > 0 else reshape(reduce_argmax(lg), [1]))

    pipe.submit(fwd)
    return await out.take_scalar()


async def wave(arena, slots, cur, n0, rss, K, seed, temperature, stop):
    """Decode K steps for len(slots) lanes in ONE pass per step. Returns tokens[lane][K]."""
    B = len(slots)
    ends = [n0[i] + K for i in range(B)]
    rows = [arena.row_pages(slots[i], ends[i]) for i in range(B)]
    flat = [p for r in rows for p in r]
    ptr = [0]
    for r in rows:
        ptr.append(ptr[-1] + len(r))
    v = model.output_vocab_size()
    cap = channel_capacity()
    tok_in = Channel.from_(cur, dtype.i32).named("tok_in")
    lanes_ind = Channel.from_(range(B + 1), dtype.u32).named("embed_indptr")
    pos = Channel.from_(n0, dtype.u32).named("positions")
    kv_len = Channel.from_([n + 1 for n in n0], dtype.u32).named("kv_len")
    pages = Channel.from_(flat, dtype.u32).named("pages")
    pind = Channel.from_(ptr, dtype.u32).named("page_indptr")
    w_slot = Channel.from_([rows[i][n0[i] // PS] for i in range(B)], dtype.u32).named("w_slot")
    w_off = Channel.from_([n % PS for n in n0], dtype.u32).named("w_off")
    rng = Channel.from_([seed, 0], dtype.u32).named("rng")
    tok_out = Channel([B], dtype.i32).capacity(cap).named("tok_out")
    # per-row page-id lookup, to find each lane's write page as it advances
    row_ids = Channel.from_(flat, dtype.u32).named("row_ids")
    row_start = Channel.from_(ptr[:-1], dtype.u32).named("row_start")

    pipe = Pipeline()
    fwd = ForwardPass()
    fwd.embed(tok_in, lanes_ind)
    fwd.bind_state(
        arena.ws,
        KvGeometry(kv_len=kv_len, pages=pages, page_indptr=pind, w_slot=w_slot, w_off=w_off,
                   positions=pos, writable_pages=arena.P),
        rss,
    )

    @fwd.epilogue
    def _():
        length = kv_len.take()            # [B] kv length of this step
        r = rng.take()
        lg = reshape(intrinsics.logits(), [B, v])
        sc = lg if temperature == 1.0 else lg / temperature
        token = gumbel_max(sc, r) if temperature > 0 else reduce_argmax(lg)
        nxt = length + 1
        ids = row_ids.take()
        starts = row_start.take()
        tok_in.put(token)
        kv_len.put(nxt)
        pos.put(length)
        w_slot.put(gather(ids, starts + length // PS))
        w_off.put(length % PS)
        tok_out.put(token)
        rng.put(r + iota(2))
        row_ids.put(ids)
        row_start.put(starts)
        pages.put(pages.take())
        pind.put(pind.take())

    out = [[] for _ in range(B)]

    async def on_step():
        t = await tok_out.take_host()
        for i in range(B):
            out[i].append(t[i])
        return True

    await pipe.run_ahead(fwd, K, on_step)
    return out


async def main(input: dict) -> dict:
    K = int(input.get("tokens", 48))
    widths = [int(x) for x in str(input.get("widths", "1,4,8,16")).split(",")]
    temperature = float(input.get("temperature", 0.0))
    hybrid = model.pass_kind() != ForwardKind.ATTENTION
    prompt = list(chat.system_user("You are a careful math tutor.", "Janet has 3 apples and buys 5 more, then gives 2 away. How many are left? Think step by step.")) + list(chat.cue())
    P = (len(prompt) - 1) // PS            # shared, page-aligned prefix pages
    pre = P * PS
    suffix = prompt[pre:]                    # >=1 tokens each lane feeds itself
    maxB = max(widths)
    tail_pages = cdiv(len(suffix) + K + 2, PS) + 1
    arena = Arena(max(P, 0), maxB, tail_pages)
    root_rs = [RsWorkingSet()] if hybrid else []
    build = Pipeline()
    if pre:
        # prefill the shared prefix once (the sampled token is discarded)
        for a, b in [(i, min(i + 256, pre)) for i in range(0, pre, 256)]:
            await feed(arena, root_rs, build, 0, a, prompt[a:b], 1, 1.0)
    build.close()
    say(f"prefix {pre} tok / {P} pages, suffix {len(suffix)} tok, hybrid={hybrid}")

    async def lanes(B):
        build = Pipeline()
        rss, firsts = [], []
        for i in range(B):
            rs = [r.fork(build) for r in root_rs]
            rss.append(rs)
            firsts.append(await feed(arena, rs, build, i, pre, suffix, 100 + i, temperature, arena.P))
        build.close()
        return rss, firsts

    results = []
    ref = None
    for B in widths:
        rss, firsts = await lanes(B)
        t0 = time.perf_counter()
        n0 = [len(prompt)] * B
        outs = await wave(arena, list(range(B)), firsts, n0, [r for rs in rss for r in rs], K, 7, temperature, set())
        dt = time.perf_counter() - t0
        if ref is None:
            ref = outs[0]
        same = sum(1 for o in outs if o == ref) if temperature == 0 else None
        results.append({"B": B, "wave_s": round(dt, 3), "agg_tok_s": round(B * K / dt, 1),
                        "identical_to_lane0": same, "sample": model.decode(outs[0][:24])})
        say(f"B={B}: {B * K} tok in {dt:.3f}s = {B * K / dt:.0f} tok/s")
    return {"hybrid": hybrid, "results": results}
