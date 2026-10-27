"""On-device beam search with logical ancestry masks — the Python twin of
`beam-search`.

The chat-templated prompt (or `prompt_tokens`) is prefilled once; that fire
picks the first `beams` tokens, and the beams then share one working set,
each reading the pool up to its own slot under a mask that admits its own
ancestry.
"""

from inferlet import chat, model
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

DEFAULT_PROMPT = "Write a short paragraph about beam search."
DEFAULT_SYSTEM = "You are a helpful, respectful and honest assistant."


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
    if B > v:
        raise ValueError(f"beams ({B}) exceeds the vocabulary ({v})")

    if input.get("prompt_tokens") is not None:
        prompt = [int(t) for t in input["prompt_tokens"]]
    else:
        prompt = list(chat.system_user(input.get("system", DEFAULT_SYSTEM), input.get("prompt", DEFAULT_PROMPT)))
        prompt += list(chat.cue_thinking(not input.get("no_think", False)))
    if not prompt:
        raise ValueError("empty prompt")
    n = len(prompt)
    # The prompt's slots, then one slot per beam per decode step.
    decodes = max_steps - 1
    pool_len = n + decodes * B
    if pool_len > 0xFFFF_FFFF:
        raise ValueError("beams * max_tokens exceeds the u32 range")
    ws = WorkingSet()
    # The device chooses the beams' slots, so the pool is reserved up front.
    ws.reserve(pool_len)

    pipeline = Pipeline()

    # Step 0: the prompt's own next-token distribution; its top B are the first beams.
    toks_p = Channel.from_(prompt, dtype.i32)
    embed_indptr_p = Channel.from_([0, n], dtype.u32)
    out_p = Channel([B], dtype.i32)
    scr_p = Channel([B], dtype.f32)
    greedy_p = Channel([1], dtype.i32)
    fwd_p = ForwardPass()
    fwd_p.embed(toks_p, embed_indptr_p)
    fwd_p.state([ws], StateGeometry.folding())

    @fwd_p.epilogue
    def _prefill():
        logits = reshape(intrinsics.output(0, "logits"), [v])
        s, i = top_k(log_softmax(logits), B)
        out_p.put(cast(i, dtype.i32))
        scr_p.put(s)
        greedy_p.put(reshape(reduce_argmax(logits), [1]))

    pipeline.submit(fwd_p)
    first = await out_p.take_host()
    final_scores = await scr_p.take_host()
    greedy0 = await greedy_p.take_host()
    greedy_mismatches = int(first[0] != greedy0[0])
    hypotheses: list[list[int]] = [[t] for t in first]

    if decodes > 0:
        scores = Channel.from_(final_scores, dtype.f32)
        toks = Channel.from_(first, dtype.i32)
        cap = channel_capacity()
        out = Channel([B], dtype.i32).capacity(cap)
        out_par = Channel([B], dtype.u32).capacity(cap)
        out_scr = Channel([B], dtype.f32).capacity(cap)
        # Independent per-lane greedy argmax over the RAW logits, beside the beam
        # pick: at `beams == 1` the two must agree on every step.
        out_greedy = Channel([B], dtype.i32).capacity(cap)
        lanes_b = Channel.from_(range(B + 1), dtype.u32)

        def step():
            # `logits()` squeezes to `[v]` for a single read-out row, so reshape back.
            logits = reshape(intrinsics.output(0, "logits"), [B, v])
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

        fwd = ForwardPass()
        fwd.embed(toks, lanes_b)
        pool = [ws] * B

        # The first decode writes beam i's token at slot n + i; it sees the prompt and its own cell.
        init_mask = [p < n or p == n + lane for lane in range(B) for p in range(pool_len)]
        mask = Channel.from_shaped([B, pool_len], init_mask)
        slots = Channel.from_(range(n, n + B), dtype.u32)
        # A beam's RoPE position is its depth in the sequence, not its pool slot.
        pos = Channel.from_([n] * B, dtype.u32)
        # Where the fire after this one writes its first beam.
        fill = Channel.from_([n + B], dtype.u32)
        fwd.state(pool, StateGeometry(positions=pos, slots=slots))
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
            pos.put(pos.take() + 1)
            fill.put(filled)

        async def on_step() -> bool:
            nonlocal hypotheses, final_scores, greedy_mismatches
            picked = await out.take_host()
            parents = await out_par.take_host()
            final_scores = await out_scr.take_host()
            greedy = await out_greedy.take_host()
            greedy_mismatches += count_greedy_mismatches(picked, greedy, B)
            hypotheses = advance_hypotheses(hypotheses, picked, parents, B)
            return True

        await pipeline.run_ahead(fwd, decodes, on_step)
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
