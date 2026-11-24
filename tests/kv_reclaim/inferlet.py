import random

from inferlet import eta as E
from inferlet import model as M


def tokens(seed, salt, n):
    rng = random.Random(seed * 1000003 + salt)
    return [rng.randrange(1000, 30000) for _ in range(n)]


async def read(pipe, sets, seqs, rows=None, targets=None):
    toks, indptr, readout, tgt = [], [0], [], []
    for i, s in enumerate(seqs):
        base = len(toks)
        toks += s
        indptr.append(len(toks))
        readout += [len(toks) - 1] if rows is None else [base + r for r in rows[i]]
        tgt += [] if rows is None else targets[i]
    k, v = len(readout), M.output_vocab_size()
    fwd = E.ForwardPass()
    fwd.embed(E.Channel.from_(toks, E.dtype.i32), E.Channel.from_(indptr, E.dtype.u32))
    fwd.state(list(sets), E.StateGeometry.folding())
    fwd.readout(E.Channel.from_(readout, E.dtype.u32))
    tok_out, lp_out = E.Channel([k], E.dtype.i32), E.Channel([k], E.dtype.f32)
    tgt_ch = E.Channel.from_(tgt or [0] * k, E.dtype.u32)

    def body():
        x = E.reshape(E.intrinsics.output(0, "logits"), [k, v])
        tok = E.reshape(E.reduce_argmax(x), [k])
        tok_out.put(tok)
        at = E.cast(tok, E.dtype.u32) if rows is None else tgt_ch.take()
        lp_out.put(E.scalar_gather(E.log_softmax(x), at))

    fwd.epilogue(body)
    pipe.submit(fwd)
    return [int(t) for t in await tok_out.take_host()], [float(x) for x in await lp_out.take_host()]


async def load(inp):
    seed, rng, pipe = inp["seed"], random.Random(inp["seed"]), E.Pipeline()
    trunk = tokens(seed, -1, inp["trunk"])
    root = E.WorkingSet()
    await read(pipe, [root], [trunk])
    sets = [root.fork(pipe) for _ in inp["lens"]]
    own = [tokens(seed, i, n) for i, n in enumerate(inp["lens"])]
    t, lp = await read(pipe, sets, own)
    segs = [[trunk + own[i], [t[i]], [lp[i]]] for i in range(len(sets))]
    due = [rng.randrange(40, 120) for _ in sets]
    for step in range(inp["decode"]):
        inj = [i for i in range(len(sets)) if step >= due[i]]
        if inj:
            extra = [tokens(seed, 10**6 + step * 1000 + i, rng.randrange(5, 30)) for i in inj]
            t, lp = await read(pipe, [sets[i] for i in inj], [[segs[i][-2][-1]] + x for i, x in zip(inj, extra)])
            for j, i in enumerate(inj):
                segs[i] += [extra[j], [t[j]], [lp[j]]]
                due[i] = step + rng.randrange(40, 120)
        rest = [i for i in range(len(sets)) if i not in inj]
        t, lp = await read(pipe, [sets[i] for i in rest], [[segs[i][-2][-1]] for i in rest])
        for j, i in enumerate(rest):
            segs[i][-2].append(t[j])
            segs[i][-1].append(lp[j])
    return {"segs": segs}


async def ref(inp):
    pipe, out = E.Pipeline(), []
    for seq, at in zip(inp["seqs"], inp["at"]):
        ws, got = E.WorkingSet(), []
        for start in range(0, len(seq), 2048):
            part = seq[start:start + 2048]
            rows = [p - 1 - start for p in at if start <= p - 1 < start + len(part)]
            _, lp = await read(pipe, [ws], [part], [rows or [0]], [[seq[start + r + 1] for r in rows] or [0]])
            got += lp[:len(rows)]
        out.append(got)
    return {"logprobs": out}


async def main(inp):
    return await (load(inp) if inp["mode"] == "load" else ref(inp))
