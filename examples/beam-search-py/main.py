"""On-device beam search with logical ancestry masks — the Python twin of
`beam-search`.

The KV cache is a prefix tree over a shared page pool. Each surviving beam
appends its new token at the next free FLAT pool position (`wpos = fill +
lane`), and its ancestry is encoded in the per-beam attention mask — inherit
the parent's mask (`gather(mask, parent)`) then set the one new cell
(`eq(col, wpos)`). That is the entire fork/prune mechanism. Exercises the
`mask` port, `Channel.from_shaped`, `capacity`, `top_k` / `gather` / `or_`
over a `[B, V]` block, and `run_ahead` — plus, on a hybrid model, the
per-lane `RsWorkingSet.fork` and the rebind of an attached pass.

**IT DOES NOT READ `prompt`.** Every beam is seeded with `BOS` at pool
position 0 and decodes from there, as the Rust program does.
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
    broadcast,
    cast,
    channel_capacity,
    dtype,
    eq,
    gather,
    intrinsics,
    iota,
    log_softmax,
    or_,
    reduce_argmax,
    reshape,
    run_ahead,
    top_k,
)

POOL_PAGES = 8  # shared pool pages (over-allocated; compaction bounds this)
BOS = 1


def count_greedy_mismatches(picked: list[int], greedy: list[int], beams: int) -> int:
    return sum(1 for lane in range(beams) if picked[lane] != greedy[lane])


def advance_hypotheses(hypotheses: list[list[int]], picked: list[int], parents: list[int], beams: int) -> list[list[int]]:
    return [hypotheses[parents[lane]] + [picked[lane]] for lane in range(beams)]


async def main(input: dict) -> str:
    page_t = model.kv_page_size()
    pool_len = POOL_PAGES * page_t
    max_steps = int(input.get("max_tokens", 16))
    B = int(input.get("beams", 2))
    if B == 0:
        raise ValueError("beams must be at least 1")
    if B > pool_len - 1:
        raise ValueError(f"beams exceeds the fixed pool ({pool_len - 1} positions)")
    capacity = (pool_len - 1) // B
    if max_steps > capacity:
        raise ValueError(f"max_tokens exceeds fixed beam pool capacity ({capacity})")
    if max_steps == 0:
        return ""

    v = model.output_vocab_size()

    # A fixed logical page pool: flat position `wpos` maps to
    # `pool_ids[wpos // page_t]` at offset `wpos % page_t`.
    ws = WorkingSet()
    pool_ids = ws.reserve(POOL_PAGES).ids
    tiled = [pool_ids[0] for _ in range(B * POOL_PAGES)]
    pool0 = pool_ids[0]

    # Shared BOS at pool position 0: every beam attends it; fill = 1.
    init_mask = [p == 0 for _ in range(B) for p in range(pool_len)]

    # Loop-carried search and page geometry.
    mask = Channel.from_shaped([B, pool_len], init_mask).named("mask")
    initial_scores = [float("-inf")] * B
    initial_scores[0] = 0.0
    scores = Channel.from_(initial_scores).named("scores")
    toks = Channel.from_([BOS] * B, dtype.i32).named("toks")
    pos = Channel.from_([0] * B, dtype.u32).named("pos")
    fill = Channel.from_([1], dtype.u32).named("fill")
    klen = Channel.from_([1] * B, dtype.u32).named("klen")
    w_slot = Channel.from_([pool0] * B, dtype.u32).named("w_slot")
    w_off = Channel.from_([0] * B, dtype.u32).named("w_off")
    pages = Channel.from_(tiled, dtype.u32).named("pages")
    page_indptr = Channel.from_shaped([B + 1], list(range(B + 1)), dtype.u32).named("page_indptr")
    lanes_b = Channel.from_(range(B + 1), dtype.u32).named("embed_indptr")
    pool_ids_ch = Channel.from_(pool_ids, dtype.u32).named("pool_ids")
    cap = channel_capacity()
    out = Channel([B], dtype.i32).capacity(cap).named("out")
    out_par = Channel([B], dtype.u32).capacity(cap).named("out_par")
    out_scr = Channel([B], dtype.f32).capacity(cap).named("out_scr")
    # Independent per-lane greedy argmax over the RAW logits, beside the beam
    # pick: at `beams == 1` the two must agree on every step.
    out_greedy = Channel([B], dtype.i32).capacity(cap).named("out_greedy")

    pipeline = Pipeline()
    kind = model.pass_kind()
    if kind == ForwardKind.ATTENTION:
        rs_working_sets: list[RsWorkingSet] = []
    elif kind == ForwardKind.HYBRID:
        rs_working_sets = [RsWorkingSet() for _ in range(B)]
    elif kind == ForwardKind.RECURRENT:
        raise ValueError("beam-search has no recurrent-only path (no registered model reports that kind)")
    else:
        raise ValueError("this program decodes a token at a time; a diffusion model wants a canvas loop")

    fwd = ForwardPass(kind)
    geometry = KvGeometry(
        kv_len=klen, pages=pages, page_indptr=page_indptr, w_slot=w_slot, w_off=w_off, positions=pos, mask=mask,
    )

    def bind_state(rs: list[RsWorkingSet]) -> None:
        # Beams never buffer: every fire folds its one token straight into
        # the recurrence, which is what makes a fork a plain state copy.
        fwd.bind_state(ws, geometry, rs)

    bind_state(rs_working_sets)
    fwd.embed(toks, lanes_b)

    @fwd.epilogue
    def _step():
        # 1. top-B over the flattened [B, V] candidate block. `logits()`
        # squeezes to `[v]` for a single read-out row, so reshape back.
        logits = reshape(intrinsics.logits(), [B, v])
        cand = broadcast(reshape(scores.take(), [B, 1]), [B, v]) + log_softmax(logits)
        s, i = top_k(reshape(cand, [B * v]), B)
        parent = i // v
        tok_i = cast(i % v, dtype.i32)

        # 2. flat tail-append positions: wpos = fill + lane.
        base = fill.take()
        lane = iota(B)
        base_b = broadcast(reshape(base, [1]), [B])
        wpos = base_b + lane

        # 3. mask evolution: inherit the parent's ancestry, OR the new position.
        inherited = gather(mask.take(), parent)
        col = broadcast(reshape(iota(pool_len), [1, pool_len]), [B, pool_len])
        wpos_b = broadcast(reshape(wpos, [B, 1]), [B, pool_len])
        newpos = eq(col, wpos_b)
        mask.put(or_(inherited, newpos))

        # 4. Explicit write descriptor for each surviving beam.
        pids = pool_ids_ch.take()
        logical_slot = wpos // page_t
        w_slot_v = gather(pids, logical_slot)
        w_off_v = wpos % page_t
        w_slot.put(w_slot_v)
        w_off.put(w_off_v)

        # KV span after this step's appends (the mask restricts attention).
        filled = base + B
        klen.put(broadcast(reshape(filled, [1]), [B]))

        pos.put(pos.take() + 1)
        fill.put(filled)
        scores.put(s)
        toks.put(tok_i)
        # Live page count for the NEXT fire, from that fire's klen; the pool
        # ids tiled B times at that stride, and the constant CSR re-emitted.
        page_count = filled.div_ceil(page_t)
        pages.put(gather(pids, iota(B * POOL_PAGES) % broadcast(page_count, [B * POOL_PAGES])))
        page_indptr.put(iota(B + 1) * broadcast(page_count, [B + 1]))

        out.put(tok_i)
        out_par.put(parent)
        out_scr.put(s)
        out_greedy.put(reshape(reduce_argmax(logits), [B]))
        pool_ids_ch.put(pids)

    hypotheses: list[list[int]] = [[] for _ in range(B)]
    final_scores = [float("-inf")] * B
    greedy_mismatches = 0

    async def drain() -> list[int]:
        """Take this step's four rows off the device; returns the parents."""
        nonlocal hypotheses, final_scores, greedy_mismatches
        picked = await out.take_host()
        parents = await out_par.take_host()
        final_scores = await out_scr.take_host()
        greedy = await out_greedy.take_host()
        greedy_mismatches += count_greedy_mismatches(picked, greedy, B)
        hypotheses = advance_hypotheses(hypotheses, picked, parents, B)
        return parents

    if not rs_working_sets:

        async def on_step() -> bool:
            await drain()
            return True

        await run_ahead(pipeline, fwd, max_steps, on_step)
    else:
        for _ in range(max_steps):
            fwd.submit(pipeline)
            parents = await drain()
            next_rs = [rs_working_sets[p].fork(pipeline) for p in parents]
            bind_state(next_rs)
            rs_working_sets = next_rs
    pipeline.close()

    # The last of the maximal lanes, as Rust's `Iterator::max_by` picks it —
    # a tie between beams is a real outcome (two hypotheses of equal mass).
    best_lane = 0
    for lane in range(1, B):
        if final_scores[lane] >= final_scores[best_lane]:
            best_lane = lane
    if B == 1 and greedy_mismatches != 0:
        raise RuntimeError(
            f"beam identity violated: width-1 beam search disagreed with greedy argmax on "
            f"{greedy_mismatches} of {max_steps} steps"
        )
    text = model.decode(hypotheses[best_lane])
    return f"{text}\n[beam] width={B} steps={max_steps} best_score={final_scores[best_lane]:.4f} greedy_mismatches={greedy_mismatches}"
