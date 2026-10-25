"""On-device beam search with logical ancestry masks — the Python twin of
`beam-search`.

**IT DOES NOT READ `prompt`.** Every beam is seeded with `BOS` at slot 0.
"""

from inferlet import model
from inferlet.eta import (
    Channel,
    ForwardPass,
    Pipeline,
    StateGeometry,
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
    top_k,
)

BOS = 1


def count_greedy_mismatches(picked: list[int], greedy: list[int], beams: int) -> int:
    return sum(1 for lane in range(beams) if picked[lane] != greedy[lane])


def advance_hypotheses(hypotheses: list[list[int]], picked: list[int], parents: list[int], beams: int) -> list[list[int]]:
    return [hypotheses[parents[lane]] + [picked[lane]] for lane in range(beams)]


async def main(input: dict) -> str:
    max_steps = int(input.get("max_tokens", 16))
    B = int(input.get("beams", 2))
    if B == 0:
        raise ValueError("beams must be at least 1")
    if max_steps == 0:
        return ""

    v = model.output_vocab_size()

    initial_scores = [float("-inf")] * B
    initial_scores[0] = 0.0
    scores = Channel.from_(initial_scores).named("scores")
    toks = Channel.from_([BOS] * B, dtype.i32).named("toks")
    cap = channel_capacity()
    out = Channel([B], dtype.i32).capacity(cap).named("out")
    out_par = Channel([B], dtype.u32).capacity(cap).named("out_par")
    out_scr = Channel([B], dtype.f32).capacity(cap).named("out_scr")
    # Independent per-lane greedy argmax over the RAW logits, beside the beam
    # pick: at `beams == 1` the two must agree on every step.
    out_greedy = Channel([B], dtype.i32).capacity(cap).named("out_greedy")
    lanes_b = Channel.from_(range(B + 1), dtype.u32).named("embed_indptr")

    def step():
        # `logits()` squeezes to `[v]` for a single read-out row, so reshape back.
        logits = reshape(intrinsics.logits(), [B, v])
        cand = broadcast(reshape(scores.take(), [B, 1]), [B, v]) + log_softmax(logits)
        s, i = top_k(reshape(cand, [B * v]), B)
        parent = i // v
        tok_i = cast(i % v, dtype.i32)
        scores.put(s)
        toks.put(tok_i)
        out.put(tok_i)
        out_par.put(parent)
        out_scr.put(s)
        out_greedy.put(reshape(reduce_argmax(logits), [B]))
        return parent

    pipeline = Pipeline()
    fwd = ForwardPass()
    fwd.embed(toks, lanes_b)

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

    pool_len = max_steps * B + 1
    if pool_len > 0xFFFF_FFFF:
        raise ValueError("beams * max_tokens exceeds the u32 range")
    ws = WorkingSet()
    # The device chooses the slots, so the pool is reserved up front.
    ws.reserve(pool_len)
    pool = [ws] * B

    init_mask = [p == 0 for _ in range(B) for p in range(pool_len)]
    mask = Channel.from_shaped([B, pool_len], init_mask).named("mask")
    slots = Channel.from_([0] * B, dtype.u32).named("slots")
    # A beam's RoPE position is its depth, not its pool slot.
    pos = Channel.from_([0] * B, dtype.u32).named("pos")
    fill = Channel.from_([1], dtype.u32).named("fill")
    kv_len = Channel.from_([1] * B, dtype.u32).named("kv_len")
    fwd.state(pool, StateGeometry(positions=pos, slots=slots, kv_len=kv_len))
    fwd.mask(mask)

    @fwd.epilogue
    def _pool_step():
        parent = step()

        base = fill.take()
        base_b = broadcast(reshape(base, [1]), [B])
        wpos = base_b + iota(B)

        inherited = gather(mask.take(), parent)
        col = broadcast(reshape(iota(pool_len), [1, pool_len]), [B, pool_len])
        wpos_b = broadcast(reshape(wpos, [B, 1]), [B, pool_len])
        newpos = eq(col, wpos_b)
        mask.put(or_(inherited, newpos))

        # Loop-carried: the host never drains these rings, so every fire re-puts them.
        filled = base + B
        slots.put(wpos)
        kv_len.put(broadcast(reshape(filled, [1]), [B]))
        pos.put(pos.take() + 1)
        fill.put(filled)

    async def on_step() -> bool:
        await drain()
        return True

    await pipeline.run_ahead(fwd, max_steps, on_step)
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
