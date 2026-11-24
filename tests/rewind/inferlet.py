from inferlet import eta as E
from inferlet import model as M


async def read(pipe, ws, toks, geometry, watch=(0,)):
    v, m = M.output_vocab_size(), len(watch)
    fwd = E.ForwardPass()
    fwd.embed(E.Channel.from_(toks, E.dtype.i32), E.Channel.from_([0, len(toks)], E.dtype.u32))
    fwd.state([ws], geometry)
    rows = E.Channel.from_([0] * m, E.dtype.u32)
    ids = E.Channel.from_(list(watch), E.dtype.u32)
    tok_out = E.Channel([1], E.dtype.i32)
    lp_out = E.Channel([1], E.dtype.f32)
    watch_out = E.Channel([m], E.dtype.f32)

    def body():
        x = E.reshape(E.intrinsics.output(0, "logits"), [1, v])
        tok = E.reshape(E.reduce_argmax(x), [1])
        lps = E.log_softmax(x)
        tok_out.put(tok)
        lp_out.put(E.scalar_gather(lps, E.cast(tok, E.dtype.u32)))
        watch_out.put(E.scalar_gather(E.gather(lps, rows.take()), ids.take()))

    fwd.epilogue(body)
    pipe.submit(fwd)
    tok, lp = (await tok_out.take_host())[0], (await lp_out.take_host())[0]
    return [int(tok), float(lp), [float(x) for x in await watch_out.take_host()]]


def at(start, rows):
    return E.StateGeometry.folding(slots=E.Channel.from_(list(range(start, start + rows)), E.dtype.u32))


async def fill(pipe, prompt, toks, fold):
    ws = E.WorkingSet()
    await read(pipe, ws, prompt, E.StateGeometry.folding())
    for t in toks:
        await read(pipe, ws, [t], E.StateGeometry.folding() if fold else E.StateGeometry())
    return ws


async def run_ahead(pipe, ws, prompt, n):
    out = [(await read(pipe, ws, prompt, E.StateGeometry.folding()))[0]]
    tok_in = E.Channel.from_(out, E.dtype.i32)
    tok_out = E.Channel([1], E.dtype.i32).capacity(E.channel_capacity())
    v = M.output_vocab_size()
    fwd = E.ForwardPass()
    fwd.embed(tok_in, E.Channel.from_([0, 1], E.dtype.u32))
    fwd.state([ws], E.StateGeometry())

    @fwd.epilogue
    def _():
        tok = E.reshape(E.reduce_argmax(E.reshape(E.intrinsics.output(0, "logits"), [1, v])), [1])
        tok_in.put(tok)
        tok_out.put(tok)

    async def on_token():
        out.append(await tok_out.take_scalar())
        return True

    await pipe.run_ahead(fwd, n - 1, on_token)
    return out


async def device(input):
    prompt, n, keep, nxt, watch = input["prompt"], input["n"], input["keep"], input["next"], input["watch"]
    ran = E.WorkingSet()
    drawn = await run_ahead(E.Pipeline(), ran, prompt, n)
    pipe = E.Pipeline()
    seq = [drawn[keep - 1]] + nxt
    got = await read(pipe, ran, seq, at(len(prompt) + keep - 1, len(seq)), watch)
    clean = await fill(pipe, prompt, [], True)
    want = await read(pipe, clean, drawn[:keep] + nxt, E.StateGeometry.folding(), watch)
    pipe.close()
    return {"drawn": drawn, "rewound": got, "clean": want}


async def main(input):
    if input.get("mode") == "device":
        return await device(input)
    prompt, over, keep, nxt = input["prompt"], input["overshoot"], input["keep"], input["next"]
    probe, watch = input["probe"], input["watch"]
    pipe = E.Pipeline()
    await fill(pipe, prompt, [], True)

    rewound = await fill(pipe, prompt, over, False)
    got = [await read(pipe, rewound, nxt, at(len(prompt) + keep, len(nxt)), watch),
           await read(pipe, rewound, [probe], E.StateGeometry.folding())]

    clean = await fill(pipe, prompt, over[:keep], True)
    want = [await read(pipe, clean, nxt, E.StateGeometry.folding(), watch),
            await read(pipe, clean, [probe], E.StateGeometry.folding())]

    stale = await fill(pipe, prompt, over, False)
    kept = await read(pipe, stale, nxt, E.StateGeometry.folding(), watch)

    frozen = await fill(pipe, prompt + over, [], True)
    try:
        await read(pipe, frozen, nxt, at(len(prompt) + keep, len(nxt)))
        refused = ""
    except E.InferletError as e:
        refused = str(e)[:400]
    pipe.close()
    return {"rewound": got, "clean": want, "stale": kept, "frozen": refused}
