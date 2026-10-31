"""Probe for WorkingSet.forget: does it read what a mask reads, and what does it save?

  --mode exact   three arms over one prompt, greedy: `mask` hides pages with an
                 attention mask, `forget` drops the same pages, `full` reads all.
                 `forget` must decode what `mask` decodes.
  --mode speed   one lane at a `context` of tokens with a `fraction` of its pages
                 forgotten: decode and tool-turn prefill tokens per second.
  --mode sweep   `speed` over comma-separated `contexts` and `fractions`.
  --mode paths   greedy tokens past a forgotten span, host loop against run-ahead.
  --mode decl    what a plain lane reads out.
  --mode merged  chunks read apart and merged, against one sequence masked to match.
  --mode read    a codebase read in order, or in parallel and merged; then questions.
  --mode team    sub-agents forked from one context, each asked about its own module;
                 `focus` makes each forget the other modules first.
"""

import json
import random
import time

from inferlet import model, session
from inferlet.eta import (
    Channel,
    ForwardPass,
    Pipeline,
    StateGeometry,
    WorkingSet,
    channel_capacity,
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
    first_page = 0 if input.get("lead") else n // (4 * page)
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


async def ahead(ws: WorkingSet, pipe: Pipeline, first: int, budget: int) -> list[int]:
    """`budget` greedy tokens on the run-ahead path; closes `pipe`."""
    tok_in = Channel.from_([first], dtype.i32)
    tok_out = Channel([1], dtype.i32).capacity(channel_capacity())
    fwd = ForwardPass()
    fwd.embed(tok_in, Channel.from_([0, 1], dtype.u32))
    fwd.state([ws], StateGeometry.folding())

    @fwd.epilogue
    def _():
        token = reshape(reduce_argmax(intrinsics.output(0, "logits")), [1])
        tok_in.put(token)
        tok_out.put(token)

    out: list[int] = []

    async def on_token() -> bool:
        out.append(await tok_out.take_scalar())
        return len(out) < budget

    await pipe.run_ahead(fwd, budget, on_token)
    return out


async def paths(input: dict) -> dict:
    """Greedy tokens on the host loop and on the run-ahead path, past a forgotten span."""
    n, page, steps = input.get("context", 2048), kv_page_size(), input.get("steps", 64)
    prompt = text_tokens(n, 1)
    start, end = n // (4 * page) * page, 3 * n // (4 * page) * page
    runs = {}
    for arm in ("host", "ahead"):
        ws, pipe = WorkingSet(), Pipeline()
        token = await step(ws, pipe, prompt)
        if input.get("forget", 1):
            ws.forget(start, end)
        if arm == "ahead":
            runs[arm] = await ahead(ws, pipe, token, steps)
            continue
        out = []
        for _ in range(steps):
            token = await step(ws, pipe, [token])
            out.append(token)
        pipe.close()
        runs[arm] = out
    agree = sum(x == y for x, y in zip(runs["host"], runs["ahead"]))
    split = next((i for i, (x, y) in enumerate(zip(runs["host"], runs["ahead"])) if x != y), None)
    return {"same": runs["host"] == runs["ahead"], "agree": agree, "split": split, "steps": steps}


async def speed(input: dict) -> dict:
    n, page = input["context"], kv_page_size()
    fraction = input.get("fraction", 0.0)
    decode, tool = input.get("decode", 64), input.get("tool", 512)
    ws, pipe = WorkingSet(), Pipeline()
    chunks = input.get("chunk", 2048)
    parts = [text_tokens(min(chunks, n - at), at + 1) for at in range(0, n, chunks)]
    t0 = time.monotonic()
    for part in parts:
        token = await step(ws, pipe, part)
    prefill = time.monotonic() - t0
    keep = int(n * (1 - fraction)) // page * page
    dropped = ws.forget(page, n - keep + page) if fraction > 0 else 0
    part = text_tokens(tool, 7)
    t0 = time.monotonic()
    token = await step(ws, pipe, part)
    turn = time.monotonic() - t0
    t0 = time.monotonic()
    token = await step(ws, pipe, [token])
    t1 = time.monotonic()
    out = await ahead(ws, pipe, token, decode)
    decoding = time.monotonic() - t1
    return {
        "context": n, "fraction": fraction, "dropped": dropped, "page": page,
        "prefill_tps": round(n / prefill, 1),
        "decode_tps": round((len(out) - 1) / decoding, 2),
        "first_token_ms": round((t1 - t0) * 1000, 1),
        "tool_prefill_tps": round(tool / turn, 1),
    }


FILLER = [
    "The handler retries the request after the timeout expires and logs the failure.",
    "A cached entry is dropped when its owner closes the connection to the pool.",
    "Each worker reads one batch from the queue and writes its result to the store.",
    "The parser skips blank lines and keeps a count of the fields it has seen.",
    "On shutdown the scheduler waits for running jobs and then flushes the journal.",
    "The client signs every call and rejects a reply whose checksum does not match.",
]


def team_docs(agents: int, tokens_each: int) -> tuple[list[str], list[tuple[str, str]]]:
    """One module per agent, each hiding one constant, and the question that asks for it."""
    rng = random.Random(11)
    docs, asks = [], []
    for agent in range(agents):
        name = f"module_{agent:02d}"
        lines = [rng.choice(FILLER) for _ in range(tokens_each // 16)]
        secret = rng.randrange(1000, 9999)
        lines.insert(rng.randrange(len(lines) // 4, 3 * len(lines) // 4), f"In {name} the retry limit constant is set to {secret}.")
        docs.append(f"# {name}\n" + "\n".join(lines) + "\n")
        asks.append((f"\nQuestion: In {name}, what is the retry limit constant? Answer with the number only.\nAnswer:", str(secret)))
    return docs, asks


async def team(input: dict) -> dict:
    """A parent reads every module; each sub-agent forks from it and answers about its own.
    With `focus`, a sub-agent forgets the other modules first: no re-read, and each decodes
    over its own pages only, while the parent and the others keep theirs."""
    agents, each = input["agents"], input["tokens_each"]
    focus, decode = input.get("focus", 0), input.get("decode", 32)
    docs, asks = team_docs(agents, each)
    parts = [model.encode(doc) for doc in docs]
    tokens = [token for part in parts for token in part]
    ats = [0]
    for part in parts:
        ats.append(ats[-1] + len(part))
    n = len(tokens)
    ws, pipe = WorkingSet(), Pipeline()
    t0 = time.monotonic()
    for at in range(0, n, 4096):
        await step(ws, pipe, tokens[at:at + 4096])
    parent = time.monotonic() - t0
    t0 = time.monotonic()
    forks = [ws.fork(pipe) for _ in range(agents)]
    if focus:
        for agent, child in enumerate(forks):
            # the first tokens stay: a model reading none of them loses the thread (attention sinks)
            child.forget(input.get("sink", 64), ats[agent])
            child.forget(ats[agent + 1], n)
    forked = time.monotonic() - t0
    questions = [model.encode(ask) for ask, _ in asks]
    v = model.output_vocab_size()
    indptr = [0]
    for question in questions:
        indptr.append(indptr[-1] + len(question))
    t0 = time.monotonic()
    first = Channel([agents], dtype.i32)
    ask = ForwardPass()
    ask.embed(Channel.from_([t for question in questions for t in question], dtype.i32), Channel.from_(indptr, dtype.u32))
    ask.state(forks, StateGeometry.folding())

    @ask.epilogue
    def _():
        first.put(reduce_argmax(reshape(intrinsics.output(0, "logits"), [agents, v])))

    pipe.submit(ask)
    tokens_now = await first.take_host()
    ttft = time.monotonic() - t0
    tok_in = Channel.from_(tokens_now, dtype.i32)
    tok_out = Channel([agents], dtype.i32).capacity(channel_capacity())
    step_pass = ForwardPass()
    step_pass.embed(tok_in, Channel.from_(list(range(agents + 1)), dtype.u32))
    step_pass.state(forks, StateGeometry.folding())

    @step_pass.epilogue
    def _():
        token = reduce_argmax(reshape(intrinsics.output(0, "logits"), [agents, v]))
        tok_in.put(token)
        tok_out.put(token)

    out = [[t] for t in tokens_now]

    async def on_step() -> bool:
        for agent, token in enumerate(await tok_out.take_host()):
            out[agent].append(token)
        return True

    t1 = time.monotonic()
    await pipe.run_ahead(step_pass, decode, on_step)
    done = time.monotonic() - t1
    answers = [model.decode(row) for row in out]
    right = sum(answer in text for (_, answer), text in zip(asks, answers))
    return {
        "agents": agents, "tokens_each": each, "context": n, "focus": focus,
        "parent_s": round(parent, 3), "fork_ms": round(forked * 1000, 2),
        "first_token_s": round(ttft, 3), "decode_s": round(done, 3),
        "total_s": round(forked + ttft + done, 3),
        "tokens_per_s": round(agents * decode / done, 1),
        "right": right, "answers": [text[:16] for text in answers], "wanted": [answer for _, answer in asks],
    }


async def read_chunk(ws: WorkingSet, pipe: Pipeline, tokens: list[int], at: int, keep: range | None, start: int, span: int) -> None:
    """Read `tokens` at positions from `at`; with `keep`, as rows of one sequence that see only
    the header and their own chunk, which is what a fork reading the chunk alone computes."""
    fwd = ForwardPass()
    fwd.embed(tokens)
    if keep is not None:
        rows = [(key < keep.start or start <= key <= start + row) for row in range(len(tokens)) for key in range(span)]
        fwd.mask(Channel.from_shaped([len(tokens), span], rows, dtype.bool))
        fwd.state([ws], StateGeometry.folding())
    else:
        positions = Channel.from_(list(range(at, at + len(tokens))), dtype.u32)
        fwd.state([ws], StateGeometry.folding(positions=positions))
    tok = Channel([1], dtype.i32)

    @fwd.epilogue
    def _():
        tok.put(reshape(reduce_argmax(intrinsics.output(0, "logits")), [1]))

    pipe.submit(fwd)
    await tok.take_scalar()


async def merged(input: dict) -> dict:
    """`merge` against one sequence masked so each chunk saw only the header and itself: chunks
    read apart and merged must give the same logits."""
    page = kv_page_size()
    header_len, chunk_len = input.get("header", 8) * page, input.get("chunk", 16) * page
    chunks, steps = input.get("chunks", 4), input.get("steps", 8)
    header = text_tokens(header_len, 3)
    parts = [text_tokens(chunk_len, 20 + i) for i in range(chunks)]
    forced = text_tokens(steps, 2)
    total = header_len + chunks * chunk_len
    span = total + steps + 1
    arms = {}
    for arm in ("seen", "mask", "merge"):
        ws, pipe = WorkingSet(), Pipeline()
        await step(ws, pipe, header)
        readers = []
        for i, part in enumerate(parts):
            start = header_len + i * chunk_len
            if arm == "merge":
                child = ws.fork(pipe)
                await read_chunk(child, pipe, part, start, None, 0, 0)
                readers.append(child)
            elif arm == "mask":
                await read_chunk(ws, pipe, part, start, range(header_len, span), start, span)
            else:
                await step(ws, pipe, part)
        slots = sum(ws.merge(child, header_len, header_len + chunk_len) for child in readers)
        rows = [await logits_of(ws, pipe, [token]) for token in forced]
        pipe.close()
        arms[arm] = {"rows": rows, "slots": slots}
    gap = lambda a, b: [round(apart(x, y), 4) for x, y in zip(arms[a]["rows"], arms[b]["rows"])]
    return {
        "page": page, "total": total, "merged_slots": arms["merge"]["slots"],
        "merge_vs_mask_max": max(gap("merge", "mask")), "merge_vs_mask": gap("merge", "mask"),
        "seen_vs_mask_max": max(gap("seen", "mask")),
    }


async def read(input: dict) -> dict:
    """A codebase attached as `docs`. `seq` reads it in order, as one context. `par` forks a reader
    for each chunk, reads them side by side at the positions they would have had, and merges
    them. Then both answer the same questions, some of which need two chunks."""
    data = json.loads(session.file("docs"))
    header, chunks, questions = data["header"], data["chunks"], data["questions"]
    if input.get("kind"):
        questions = [q for q in questions if q["kind"] == input["kind"]]
    arm, decode = input.get("arm", "par"), input.get("decode", 12)
    h, each, count = len(header), len(chunks[0]), len(chunks)
    v = model.output_vocab_size()
    ws, pipe = WorkingSet(), Pipeline()
    await step(ws, pipe, header)
    t0 = time.monotonic()
    readers = []
    if arm == "seq":
        for part in chunks:
            for at in range(0, each, 1024):
                await step(ws, pipe, part[at:at + 1024])
    elif arm == "apart":
        readers = [ws.fork(pipe) for _ in chunks]
        for i, part in enumerate(chunks):
            for at in range(0, each, 1024):
                await read_chunk(readers[i], pipe, part[at:at + 1024], h + i * each + at, None, 0, 0)
    else:
        readers = [ws.fork(pipe) for _ in chunks]
        width = input.get("width", 1024 // count)
        for at in range(0, each, width):
            rows = [part[at:at + width] for part in chunks]
            fwd = ForwardPass()
            fwd.embed(
                Channel.from_([token for row in rows for token in row], dtype.i32),
                Channel.from_([i * len(rows[0]) for i in range(count + 1)], dtype.u32),
            )
            spots = [h + i * each + at + j for i in range(count) for j in range(len(rows[0]))]
            fwd.state(readers, StateGeometry.folding(positions=Channel.from_(spots, dtype.u32)))
            done = Channel([count], dtype.i32)

            @fwd.epilogue
            def _():
                done.put(reduce_argmax(reshape(intrinsics.output(0, "logits"), [count, v])))

            pipe.submit(fwd)
            await done.take_host()
    reading = time.monotonic() - t0
    t0 = time.monotonic()
    picked = input.get("picked", 0)
    merged_slots = 0 if picked else sum(ws.merge(reader, h, h + each) for reader in readers)
    merging = time.monotonic() - t0
    batch, answers = input.get("batch", 16), []
    t0 = time.monotonic()
    for at in range(0, len(questions), batch):
        group = questions[at:at + batch]
        forks = [ws.fork(pipe) for _ in group]
        lanes = len(group)
        indptr = [0]
        for lane, question in enumerate(group):
            if picked:  # only the chunks this question needs, the best a page selector could do
                for chunk in sorted(question["chunks"]):
                    forks[lane].merge(readers[chunk], h, h + each)
        for question in group:
            indptr.append(indptr[-1] + len(question["q"]))
        first = Channel([lanes], dtype.i32)
        ask = ForwardPass()
        ask.embed(Channel.from_([t for question in group for t in question["q"]], dtype.i32), Channel.from_(indptr, dtype.u32))
        total = h + each * count
        if picked:  # rows run on past every chunk, merged here or not
            spots = [total + j for question in group for j in range(len(question["q"]))]
            ask.state(forks, StateGeometry.folding(positions=Channel.from_(spots, dtype.u32)))
        else:
            ask.state(forks, StateGeometry.folding())

        @ask.epilogue
        def _():
            first.put(reduce_argmax(reshape(intrinsics.output(0, "logits"), [lanes, v])))

        pipe.submit(ask)
        heads = await first.take_host()
        if picked:
            out = [[t] for t in heads]
            for turn in range(decode):
                next_tokens = Channel([lanes], dtype.i32)
                go = ForwardPass()
                go.embed(Channel.from_([row[-1] for row in out], dtype.i32), Channel.from_(list(range(lanes + 1)), dtype.u32))
                spots = [total + len(question["q"]) + turn for question in group]
                go.state(forks, StateGeometry.folding(positions=Channel.from_(spots, dtype.u32)))

                @go.epilogue
                def _():
                    next_tokens.put(reduce_argmax(reshape(intrinsics.output(0, "logits"), [lanes, v])))

                pipe.submit(go)
                for lane, token in enumerate(await next_tokens.take_host()):
                    out[lane].append(token)
            answers += [model.decode(row) for row in out]
            continue
        tok_in = Channel.from_(heads, dtype.i32)
        tok_out = Channel([lanes], dtype.i32).capacity(channel_capacity())
        go = ForwardPass()
        go.embed(tok_in, Channel.from_(list(range(lanes + 1)), dtype.u32))
        go.state(forks, StateGeometry.folding())

        @go.epilogue
        def _():
            token = reduce_argmax(reshape(intrinsics.output(0, "logits"), [lanes, v]))
            tok_in.put(token)
            tok_out.put(token)

        out = [[t] for t in heads]

        async def on_step() -> bool:
            for lane, token in enumerate(await tok_out.take_host()):
                out[lane].append(token)
            return True

        await pipe.run_ahead(go, decode, on_step, end=False)
        answers += [model.decode(row) for row in out]
    asking = time.monotonic() - t0
    pipe.close()
    hit = lambda q, text: all(part in text for part in q["a"].split())
    tally = lambda kind: sum(hit(q, text) for q, text in zip(questions, answers) if q["kind"] == kind)
    return {
        "arm": arm, "chunks": count, "tokens": h + each * count, "merged_slots": merged_slots,
        "read_s": round(reading, 2), "read_tps": round(each * count / reading, 1), "merge_ms": round(merging * 1000, 2),
        "ask_s": round(asking, 2), "questions": len(questions),
        "right_one": tally("one"), "right_two": tally("two"), "right_pair": tally("pair"), "right_quad": tally("quad"),
        "right": [int(hit(q, text)) for q, text in zip(questions, answers)],
        "samples": [text[:14] for text in answers[:6]],
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
    if mode == "decl":
        return {"outputs": [[o.name, o.width, o.layers, o.readout] for o in model.lane([]).outputs], "page": kv_page_size()}
    return await {"exact": exact, "speed": speed, "sweep": sweep, "paths": paths, "team": team, "merged": merged, "read": read}[mode](input)
