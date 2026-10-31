"""Probe for WorkingSet.forget: does it read what a mask reads, and what does it save?

  --mode exact   three arms over one prompt, greedy: `mask` hides pages with an
                 attention mask, `forget` drops the same pages, `full` reads all.
                 `forget` must decode what `mask` decodes.
  --mode speed   one lane at a `context` of tokens with a `fraction` of its pages
                 forgotten: decode and tool-turn prefill tokens per second.
  --mode sweep   `speed` over comma-separated `contexts` and `fractions`.
"""

import random
import time

from inferlet import model, session
from inferlet.eta import (
    Channel,
    ForwardPass,
    Pipeline,
    StateGeometry,
    WorkingSet,
    dtype,
    intrinsics,
    kv_page_size,
    reduce_argmax,
    reshape,
)


def text_tokens(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    nouns = ["parser", "cache", "request", "module", "token", "queue", "field", "object"]
    verbs = ["returns", "raises", "stores", "reads", "checks", "drops", "writes", "skips"]
    out: list[int] = []
    while len(out) < n:
        out += model.encode(f"The {rng.choice(nouns)} {rng.choice(verbs)} the {rng.choice(nouns)} {rng.randrange(999)}. ")
    return out[:n]


async def step(ws: WorkingSet, pipe: Pipeline, tokens: list[int], mask=None) -> int:
    fwd = ForwardPass()
    fwd.embed(tokens)
    if mask is not None:
        fwd.mask(mask)
    fwd.state([ws], StateGeometry.folding())
    tok = Channel([1], dtype.i32)

    @fwd.epilogue
    def _():
        tok.put(reshape(reduce_argmax(intrinsics.output(0, "logits")), [1]))

    pipe.submit(fwd)
    return await tok.take_scalar()


def hidden_mask(at: int, span: int, hide: range) -> Channel:
    rows = [key <= at and key not in hide for key in range(span)]
    return Channel.from_shaped([1, span], rows, dtype.bool)


async def logits_of(ws: WorkingSet, pipe: Pipeline, tokens: list[int], mask=None, position=None) -> list[float]:
    fwd = ForwardPass()
    fwd.embed(tokens)
    if mask is not None:
        fwd.mask(mask)
    if position is None:
        fwd.state([ws], StateGeometry.folding())
    else:
        fwd.state([ws], StateGeometry.folding(positions=Channel.from_([position], dtype.u32)))
    out = Channel([model.output_vocab_size()], dtype.f32)

    @fwd.epilogue
    def _():
        out.put(reshape(intrinsics.output(0, "logits"), [model.output_vocab_size()]))

    pipe.submit(fwd)
    return await out.take_host()


def apart(a: list[float], b: list[float]) -> float:
    """The largest logit gap, after removing each row's mean."""
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    return max(abs((x - ma) - (y - mb)) for x, y in zip(a, b))


async def exact(input: dict) -> dict:
    n, page = input.get("context", 1024), kv_page_size()
    steps = input.get("steps", 48)
    prompt = text_tokens(n, 1)
    forced = text_tokens(steps, 2)
    first_page = n // (4 * page)
    last_page = 3 * n // (4 * page)
    hide = range(first_page * page, last_page * page)
    span = n + steps + 1
    arms = {}
    for arm in ("full", "mask", "mask0", "forget", "compact"):
        ws, pipe = WorkingSet(), Pipeline()
        await step(ws, pipe, prompt)
        dropped = ws.forget(hide.start, hide.stop) if arm in ("forget", "compact") else 0
        rows = []
        for i, token in enumerate(forced):
            at = n + i
            mask = None
            if arm == "mask":
                mask = hidden_mask(at, span, hide)
            elif arm == "mask0":
                mask = hidden_mask(at, span, range(0))
            position = at - len(hide) if arm == "compact" else None
            rows.append(await logits_of(ws, pipe, [token], mask, position))
        pipe.close()
        arms[arm] = {"rows": rows, "dropped": dropped}
    gap = lambda a, b: [round(apart(x, y), 4) for x, y in zip(arms[a]["rows"], arms[b]["rows"])]
    forget_mask, full_mask = gap("forget", "mask"), gap("full", "mask")
    top = lambda rows: [max(range(len(r)), key=r.__getitem__) for r in rows]
    ids = [sorted(range(len(r)), key=r.__getitem__)[-16:] for r in arms["full"]["rows"][:4]]
    return {
        "prompt": prompt, "forced": forced, "hide": [hide.start, hide.stop], "ids": ids,
        "values": {arm: [[v["rows"][i][t] for t in row] for i, row in enumerate(ids)] for arm, v in arms.items()},
        "page": page,
        "hidden_slots": len(hide),
        "dropped": arms["forget"]["dropped"],
        "mask0_vs_full_max": max(gap("mask0", "full")),
        "compact_vs_mask_max": max(gap("compact", "mask")),
        "compact_vs_forget_max": max(gap("compact", "forget")),
        "forget_vs_mask_max": max(forget_mask),
        "forget_vs_mask": forget_mask[:12],
        "full_vs_mask_max": max(full_mask),
        "full_vs_mask": full_mask[:12],
        "top_agree": sum(x == y for x, y in zip(top(arms["forget"]["rows"]), top(arms["mask"]["rows"]))),
        "steps": steps,
    }


async def speed(input: dict) -> dict:
    n, page = input["context"], kv_page_size()
    fraction = input.get("fraction", 0.0)
    decode, tool = input.get("decode", 64), input.get("tool", 512)
    ws, pipe = WorkingSet(), Pipeline()
    token = 0
    chunks = input.get("chunk", 2048)
    t0 = time.monotonic()
    made = 0
    while made < n:
        part = text_tokens(min(chunks, n - made), made + 1)
        token = await step(ws, pipe, part)
        made += len(part)
    prefill = time.monotonic() - t0
    keep = int(n * (1 - fraction)) // page * page
    dropped = ws.forget(page, n - keep + page) if fraction > 0 else 0
    t0 = time.monotonic()
    for _ in range(decode):
        token = await step(ws, pipe, [token])
    decoding = time.monotonic() - t0
    part = text_tokens(tool, 7)
    t0 = time.monotonic()
    token = await step(ws, pipe, part)
    turn = time.monotonic() - t0
    pipe.close()
    return {
        "context": n, "fraction": fraction, "dropped": dropped, "page": page,
        "prefill_tps": round(n / prefill, 1),
        "decode_tps": round(decode / decoding, 2),
        "tool_prefill_tps": round(tool / turn, 1),
    }


async def sweep(input: dict) -> dict:
    """`speed` over every context and fraction, in one engine."""
    rows = []
    for context in input["contexts"].split(","):
        for fraction in input["fractions"].split(","):
            row = await speed({**input, "context": int(context), "fraction": float(fraction)})
            rows.append(row)
            session.send(row)
    return {"rows": rows}


async def main(input: dict) -> dict:
    mode = input.get("mode", "exact")
    return await {"exact": exact, "speed": speed, "sweep": sweep}[mode](input)
