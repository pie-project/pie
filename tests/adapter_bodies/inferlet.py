from inferlet import eta as E
from inferlet import model as M
from inferlet import session as S


async def read(pipe, sets, seqs, adapter):
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
    if adapter:
        fwd.adapter(adapter)
    tok_out, lp_out = E.Channel([k], E.dtype.i32), E.Channel([k], E.dtype.f32)

    def body():
        x = E.reshape(E.intrinsics.output(0, "logits"), [k, v])
        tok = E.reshape(E.reduce_argmax(x), [k])
        tok_out.put(tok)
        lp_out.put(E.scalar_gather(E.log_softmax(x), E.cast(tok, E.dtype.u32)))

    fwd.epilogue(body)
    pipe.submit(fwd)
    return [int(t) for t in await tok_out.take_host()], [float(x) for x in await lp_out.take_host()]


async def main(inp):
    adapter = E.Adapter.from_bundle(S.file("adapter.safetensors")) if inp["adapter"] else None
    pipe = E.Pipeline()
    sets = [E.WorkingSet() for _ in inp["prompts"]]
    toks, lps = [[] for _ in sets], [[] for _ in sets]
    t, lp = await read(pipe, sets, inp["prompts"], adapter)
    for _ in range(inp["decode"]):
        for i in range(len(sets)):
            toks[i].append(t[i]), lps[i].append(lp[i])
        t, lp = await read(pipe, sets, [[x] for x in t], adapter)
    pipe.close()
    return {"tokens": toks, "logprobs": lps}
