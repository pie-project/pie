
from inferlet import eta as E
from inferlet import model as M


async def read(pipe, sets, seqs):
    toks, indptr, readout = [], [0], []
    for s in seqs:
        toks += s
        indptr.append(len(toks))
        readout.append(len(toks) - 1)
    k, v = len(sets), M.output_vocab_size()
    fwd = E.ForwardPass()
    fwd.embed(E.Channel.from_(toks, E.dtype.i32), E.Channel.from_(indptr, E.dtype.u32))
    fwd.state(list(sets), E.StateGeometry.folding())
    fwd.readout(E.Channel.from_(readout, E.dtype.u32))
    tok_out = E.Channel([k], E.dtype.i32)
    lp_out = E.Channel([k], E.dtype.f32)

    def body():
        x = E.reshape(E.intrinsics.output(0, "logits"), [k, v])
        tok = E.reshape(E.reduce_argmax(x), [k])
        tok_out.put(tok)
        lp_out.put(E.scalar_gather(E.log_softmax(x), E.cast(tok, E.dtype.u32)))

    fwd.epilogue(body)
    pipe.submit(fwd)
    toks = await tok_out.take_host()
    lps = await lp_out.take_host()
    return [[int(t), float(lp)] for t, lp in zip(toks, lps)]


async def main(input):
    prompts, chunks = input["prompts"], input["chunks"]
    pipe = E.Pipeline()
    sets = [E.WorkingSet() for _ in prompts]
    await read(pipe, sets, [p[:-1] for p in prompts])
    seqs = [[p[-1]] + c for p, c in zip(prompts, chunks)]
    lanes = range(len(sets))
    groups = {
        "mixed": [list(lanes)],
        "split": [[i for i in lanes if chunks[i]], [i for i in lanes if not chunks[i]]],
        "alone": [[i] for i in lanes],
    }[input["mode"]]
    out = [None] * len(sets)
    for g in filter(None, groups):
        for i, r in zip(g, await read(pipe, [sets[i] for i in g], [seqs[i] for i in g])):
            out[i] = r
    probe = [(await read(pipe, [ws], [[input["probe"]]]))[0] for ws in sets]
    pipe.close()
    return {"read": out, "probe": probe}
