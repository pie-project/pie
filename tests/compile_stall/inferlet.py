import time
from inferlet import eta as E
from inferlet import model as M


def swarm_sampler(k, v, slots, temp, top_p, agent_ch, slot_ch, state_chs, tok_out, lp_out):
    def body():
        x = E.reshape(E.intrinsics.output(0, "logits"), [k, v])
        if temp != 1.0:
            x = x / temp
        if top_p < 1.0:
            x = E.mask_apply(x, E.pivot_threshold(E.softmax(x), E.cummass_le(top_p)))
        agent = agent_ch.take()
        slot = slot_ch.take()
        noise = None
        for s in range(slots):
            ns = E.gather(E.gumbel(state_chs[s].take(), [8, v]), agent)
            noise = ns if noise is None else E.select(E.broadcast(E.eq(slot, s), [k, v]), ns, noise)
        tok = E.reduce_argmax(x + noise)
        tok_out.put(tok)
        lp_out.put(E.scalar_gather(E.log_softmax(x), E.cast(tok, E.dtype.u32)))
    return body


def greedy(k, v, tok_out):
    def body():
        tok_out.put(E.reshape(E.reduce_argmax(E.reshape(E.intrinsics.output(0, "logits"), [k, v])), [k]))
    return body


def forward(sets, seqs):
    toks, indptr, readout = [], [0], []
    for s in seqs:
        toks += s
        indptr.append(len(toks))
        readout.append(len(toks) - 1)
    fwd = E.ForwardPass()
    fwd.embed(E.Channel.from_(toks, E.dtype.i32), E.Channel.from_(indptr, E.dtype.u32))
    fwd.state(list(sets), E.StateGeometry.folding())
    fwd.readout(E.Channel.from_(readout, E.dtype.u32))
    return fwd


async def shaped(pipe, sets, k, slots, temp, top_p, seed):
    v = M.output_vocab_size()
    t0 = time.monotonic()
    fwd = forward(sets[:k], [[42]] * k)
    agent = E.Channel.from_([i % 8 for i in range(k)], E.dtype.u32)
    slot = E.Channel.from_([i % slots for i in range(k)], E.dtype.u32)
    states = [E.Channel.from_([seed + s, 7], E.dtype.u32) for s in range(slots)]
    tok_out = E.Channel([k], E.dtype.i32)
    lp_out = E.Channel([k], E.dtype.f32)
    fwd.epilogue(swarm_sampler(k, v, slots, temp, top_p, agent, slot, states, tok_out, lp_out))
    pipe.submit(fwd)
    t1 = time.monotonic()
    await tok_out.take_host()
    await lp_out.take_host()
    return round(t1 - t0, 4), round(time.monotonic() - t1, 4)


async def main(input):
    pipe = E.Pipeline()
    v = M.output_vocab_size()
    if input["mode"] == "victim":
        ws = E.WorkingSet()
        tok_out = E.Channel([1], E.dtype.i32)
        fwd = forward([ws], [[1000, 2000, 3000]])
        fwd.epilogue(greedy(1, v, tok_out))
        pipe.submit(fwd)
        tok = int((await tok_out.take_host())[0])
        lat, end = [], time.monotonic() + float(input["seconds"])
        while time.monotonic() < end:
            t = time.monotonic()
            fwd = forward([ws], [[tok]])
            tok_out = E.Channel([1], E.dtype.i32)
            fwd.epilogue(greedy(1, v, tok_out))
            pipe.submit(fwd)
            tok = int((await tok_out.take_host())[0])
            lat.append(round(time.monotonic() - t, 4))
        return {"latency": lat}
    ks = list(input["ks"])
    sets = [E.WorkingSet() for _ in range(max(ks))]
    fwd = forward(sets, [[1000 + i, 2000 + i] for i in range(len(sets))])
    tok_out = E.Channel([len(sets)], E.dtype.i32)
    fwd.epilogue(greedy(len(sets), v, tok_out))
    pipe.submit(fwd)
    await tok_out.take_host()
    out = []
    for k in ks:
        a = await shaped(pipe, sets, k, int(input["slots"]), float(input["temp"]), float(input["top_p"]), int(input["seed"]))
        b = await shaped(pipe, sets, k, int(input["slots"]), float(input["temp"]), float(input["top_p"]), int(input["seed"]))
        out.append([k, a, b])
    return {"passes": out}
