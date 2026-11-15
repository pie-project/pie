
import time as _pie_time

try:  # lint / local only: in the uploaded program episode.py's text is above this line
    from episode import STOP, Episode  # noqa: F401
except ImportError:
    pass

from inferlet import chat as _pie_chat
from inferlet import eta as _pie_eta
from inferlet import model as _pie_model

PIE_MAX_ROWS = min(2048, _pie_model.max_pass_rows()) if hasattr(_pie_model, "max_pass_rows") else 2048
PIE_MAX_LANES = 64  # the server's max_forward_requests


class PieTeamEngine:

    def __init__(self, seed: int, temperature: float, top_p: float) -> None:
        self.pipe = _pie_eta.Pipeline()
        self.vocab = _pie_model.output_vocab_size()
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.key = (int(seed) * 0x9E3779B1 + 0x7F4A7C15) & 0xFFFFFFFF
        self.ctr = 0
        self.passes = 0
        self.decode_passes = 0
        self.prefill_passes = 0
        self.decode_rows = 0
        self.prefill_rows = 0
        self.max_lanes = 0
        self.lane_sum = 0  # sum of lanes over decode passes
        self.pass_s = 0.0  # host time inside passes: build + submit (trace) + wait
        self.build_s = 0.0
        self.submit_s = 0.0
        self.wait_s = 0.0

    def _sampler(self, k: int, rng: _pie_eta.Channel, tok_out: _pie_eta.Channel, lp_out: _pie_eta.Channel):
        v, temp, top_p = self.vocab, self.temperature, self.top_p

        def body():
            x = _pie_eta.reshape(_pie_eta.intrinsics.output(0, "logits"), [k, v])  # one readout row per lane
            if temp <= 0.0:  # greedy; logprob reported under the untempered distribution
                tok = _pie_eta.reshape(_pie_eta.reduce_argmax(x), [k])
                lp = _pie_eta.scalar_gather(_pie_eta.log_softmax(x), _pie_eta.cast(tok, _pie_eta.dtype.u32))
            else:
                if temp != 1.0:
                    x = x / temp
                if top_p < 1.0:  # nucleus: keep the head whose mass is within top_p (IR's expansion)
                    x = _pie_eta.mask_apply(x, _pie_eta.pivot_threshold(_pie_eta.softmax(x), _pie_eta.cummass_le(top_p)))
                tok = _pie_eta.gumbel_max(x, rng.take())
                lp = _pie_eta.scalar_gather(_pie_eta.log_softmax(x), _pie_eta.cast(tok, _pie_eta.dtype.u32))
            tok_out.put(tok)
            lp_out.put(lp)

        return body

    async def _pass(self, sets: list, seqs: list) -> tuple[list[int], list[float]]:
        toks, indptr, readout = [], [0], []
        for s in seqs:
            toks += s
            indptr.append(len(toks))
            readout.append(len(toks) - 1)
        k = len(sets)
        t_build = _pie_time.monotonic()
        fwd = _pie_eta.ForwardPass()
        fwd.embed(_pie_eta.Channel.from_(toks, _pie_eta.dtype.i32), _pie_eta.Channel.from_(indptr, _pie_eta.dtype.u32))
        fwd.state(list(sets), _pie_eta.StateGeometry.folding())
        fwd.readout(_pie_eta.Channel.from_(readout, _pie_eta.dtype.u32))
        rng = _pie_eta.Channel.from_([self.key ^ ((self.ctr * 0x85EBCA6B) & 0xFFFFFFFF), self.ctr], _pie_eta.dtype.u32)
        self.ctr += 1
        tok_out = _pie_eta.Channel([k], _pie_eta.dtype.i32)
        lp_out = _pie_eta.Channel([k], _pie_eta.dtype.f32)
        fwd.epilogue(self._sampler(k, rng, tok_out, lp_out))
        t = _pie_time.monotonic()
        self.pipe.submit(fwd)  # traces the epilogue and encodes the program, then submits
        t_sub = _pie_time.monotonic()
        nxt = await tok_out.take_host()
        lps = await lp_out.take_host()
        t_end = _pie_time.monotonic()
        self.build_s += t - t_build
        self.submit_s += t_sub - t
        self.wait_s += t_end - t_sub
        self.pass_s += t_end - t_build
        self.passes += 1
        if all(len(s) == 1 for s in seqs):
            self.decode_passes += 1
            self.decode_rows += len(toks)
            self.lane_sum += k
            self.max_lanes = max(self.max_lanes, k)
        else:
            self.prefill_passes += 1
            self.prefill_rows += len(toks)
        return [int(x) for x in nxt], [float(x) for x in lps]

    async def read(self, sets: list, seqs: list) -> tuple[list[int], list[float]]:
        seqs = [list(s) for s in seqs]
        for i, s in enumerate(seqs):  # a sequence longer than a pass: its head goes first, alone
            while len(s) > PIE_MAX_ROWS:
                await self._pass([sets[i]], [s[:PIE_MAX_ROWS]])
                s = s[PIE_MAX_ROWS:]
            seqs[i] = s
        out_t, out_l = [0] * len(sets), [0.0] * len(sets)
        i = 0
        while i < len(sets):
            j, rows = i, 0
            while j < len(sets) and j - i < PIE_MAX_LANES and (j == i or rows + len(seqs[j]) <= PIE_MAX_ROWS):
                rows += len(seqs[j])
                j += 1
            t, lp = await self._pass(sets[i:j], seqs[i:j])
            out_t[i:j], out_l[i:j] = t, lp
            i = j
        return out_t, out_l

    def fork(self, ws: _pie_eta.WorkingSet) -> _pie_eta.WorkingSet:
        return ws.fork(self.pipe)

    async def _score(self, ws, piece: list, rows: list) -> list:
        v, temp, top_p = self.vocab, self.temperature, self.top_p
        ro = [r for r, _ in rows] or [len(piece) - 1]
        n = len(ro)
        fwd = _pie_eta.ForwardPass()
        fwd.embed(_pie_eta.Channel.from_(piece, _pie_eta.dtype.i32),
                  _pie_eta.Channel.from_([0, len(piece)], _pie_eta.dtype.u32))
        fwd.state([ws], _pie_eta.StateGeometry.folding())
        fwd.readout(_pie_eta.Channel.from_(ro, _pie_eta.dtype.u32))
        tgt = _pie_eta.Channel.from_([t for _, t in rows] or [0], _pie_eta.dtype.u32)
        lp_out = _pie_eta.Channel([n], _pie_eta.dtype.f32)

        def body():
            x = _pie_eta.reshape(_pie_eta.intrinsics.output(0, "logits"), [n, v])
            if temp > 0.0 and temp != 1.0:
                x = x / temp
            if temp > 0.0 and top_p < 1.0:
                x = _pie_eta.mask_apply(x, _pie_eta.pivot_threshold(_pie_eta.softmax(x), _pie_eta.cummass_le(top_p)))
            lp_out.put(_pie_eta.scalar_gather(_pie_eta.log_softmax(x), tgt.take()))

        fwd.epilogue(body)
        self.pipe.submit(fwd)
        got = await lp_out.take_host()
        return [float(x) for x in got] if rows else []

    async def rescore(self, tokens: list, at: list) -> list:
        runs: list = []  # contiguous runs of indices, at most 64 long (one readout run per pass)
        for i in at:
            if i < 1 or i >= len(tokens):
                continue
            if runs and runs[-1][-1] == i - 1 and len(runs[-1]) < 64:
                runs[-1].append(i)
            else:
                runs.append([i])
        ws = _pie_eta.WorkingSet()
        out: list = []
        cur = 0  # tokens already in ws
        for run in runs:
            lo, end = run[0] - 1, run[-1]  # logits rows lo .. end-1 predict tokens run[0] .. run[-1]
            while end - cur > PIE_MAX_ROWS and lo > cur:  # a long stretch with nothing to score
                step = min(PIE_MAX_ROWS, lo - cur)
                await self._score(ws, tokens[cur:cur + step], [])
                cur += step
            out += await self._score(ws, tokens[cur:end], [(i - 1 - cur, tokens[i]) for i in run])
            cur = end
        pie_release(ws)
        return out


def pie_release(ws) -> None:
    try:
        ws.ws.__exit__(None, None, None)
    except Exception:  # noqa: BLE001 -- already released
        pass


class PieAgent:
    __slots__ = ("a", "ws", "pending", "chunk", "lps", "used", "traj", "decoding", "limit")

    def __init__(self, a: int) -> None:
        self.a = a
        self.ws = None
        self.pending = None  # last sampled token, not yet written to the working set
        self.chunk: list[int] = []  # tokens of the generation chunk in progress
        self.lps: list[float] = []
        self.used = 0  # generated tokens, all chunks
        self.traj: list[dict] = []
        self.decoding = False
        self.limit = 0  # this chunk's token cap


async def main(input: dict) -> dict:
    t0 = _pie_time.monotonic()
    task = input.get("task", "aggregate")
    n_agents = int(input.get("n_agents", 4))
    seed = int(input.get("seed", 0))
    budget = int(input.get("budget_tokens", 2048))
    chunk_tokens = int(input.get("chunk_tokens", 1024))
    kwargs = {"budget_tokens": budget, "chunk_tokens": chunk_tokens, "depth": int(input.get("depth", 2))}
    if input.get("lam") is not None:
        kwargs["lam"] = float(input["lam"])
    temperature = float(input.get("temperature", 1.0))
    top_p = float(input.get("top_p", 1.0))
    max_wall_s = float(input.get("max_wall_s", 600.0))

    ep = Episode(task, n_agents, seed, **kwargs)
    eng = PieTeamEngine(seed, temperature, top_p)
    eos = set(_pie_chat.stop_tokens())  # <|im_end|>, <|endoftext|> on Qwen3
    stop_ids = {}  # single-token stop strings, by id
    for s in STOP:
        ids = _pie_model.encode(s)
        if len(ids) == 1:
            stop_ids[ids[0]] = s
    tail_n = max(len(_pie_model.encode(s)) for s in STOP) + 4
    notes = []
    if temperature <= 0.0:
        notes.append("temperature<=0: greedy; logprobs are under the untempered distribution")
    stats = {"injected_tokens": 0, "steps": 0, "wakes": 0, "idle_decode_violations": 0,
             "inject_passes": 0, "timeout": False, "aborted": None}

    shared_text = ep.shared_prompt()
    shared = _pie_model.encode(shared_text)
    trunk = _pie_eta.WorkingSet()
    await eng.read([trunk], [shared])
    t_shared = _pie_time.monotonic() - t0

    agents = [PieAgent(a) for a in range(n_agents)]
    privs = []
    for ag in agents:
        ag.ws = eng.fork(trunk)
        ptext = ep.private_prompt(ag.a)
        ptoks = _pie_model.encode(ptext)
        privs.append(ptoks)
        ag.traj.append({"kind": "prompt", "text": shared_text + ptext, "tokens": shared + ptoks,
                        "logprobs": None, "mask": 0})
    first, first_lp = await eng.read([ag.ws for ag in agents], privs)

    def start_chunk(ag: PieAgent, tok: int, lp: float) -> None:
        ag.chunk, ag.lps = [tok], [lp]
        ag.pending = tok
        ag.decoding = True
        ag.limit = ep.max_new_tokens(ag.a)  # min(chunk_tokens, budget left)

    def chunk_end(ag: PieAgent):
        tok = ag.chunk[-1]
        if tok in stop_ids:
            return ("tool_call" if stop_ids[tok] == STOP[0] else "end_turn"), stop_ids[tok]
        if tok in eos:
            return "end_turn", None
        tail = _pie_model.decode(ag.chunk[-tail_n:])
        for s in STOP:
            if s in tail:
                return ("tool_call" if s == STOP[0] else "end_turn"), s
        if len(ag.chunk) >= ag.limit:
            return "length", None
        return None

    pending_inject: dict = {}  # agent -> text to append before it decodes again

    def finish_chunk(ag: PieAgent, reason: str, stop: str | None) -> None:
        text = _pie_model.decode(ag.chunk)
        if stop is not None and stop in text:  # a stop spelled over several tokens: cut after it
            text = text[: text.index(stop) + len(stop)]
        n = len(ag.chunk)
        ag.used += n
        ag.traj.append({"kind": "gen", "text": text, "tokens": list(ag.chunk), "logprobs": list(ag.lps),
                        "mask": 1})
        ag.chunk, ag.lps = [], []
        ag.decoding = False
        stats["steps"] += 1
        inj = ep.step(ag.a, text, reason, n)
        if inj is not None and ep.status(ag.a) == "running":
            if inj:
                pending_inject[ag.a] = inj
            elif ep.max_new_tokens(ag.a) > 0:
                ag.decoding = True
                ag.limit = ep.max_new_tokens(ag.a)
        if ep.status(ag.a) == "done":
            pie_release(ag.ws)
            ag.ws = None

    for ag, tok, lp in zip(agents, first, first_lp):
        start_chunk(ag, tok, lp)
        end = chunk_end(ag)
        if end:
            finish_chunk(ag, *end)

    async def team_loop() -> None:
        while not ep.done:
            if _pie_time.monotonic() - t0 > max_wall_s:
                stats["timeout"] = True
                notes.append(f"stopped at max_wall_s={max_wall_s}")
                return
            for ag in agents:
                if ag.a not in pending_inject and ep.status(ag.a) == "idle":
                    w = ep.wake(ag.a)
                    if w:
                        stats["wakes"] += 1
                        pending_inject[ag.a] = w
            acted = bool(pending_inject)
            if pending_inject:
                items = [(agents[a], txt) for a, txt in sorted(pending_inject.items()) if agents[a].ws is not None]
                pending_inject.clear()
                sets, seqs = [], []
                for ag, txt in items:
                    itoks = _pie_model.encode(txt)
                    ag.traj.append({"kind": "inject", "text": txt, "tokens": itoks, "logprobs": None, "mask": 0})
                    stats["injected_tokens"] += len(itoks)
                    sets.append(ag.ws)
                    seqs.append(([ag.pending] if ag.pending is not None else []) + itoks)
                if items:
                    stats["inject_passes"] += 1
                    toks, lps = await eng.read(sets, seqs)
                    for (ag, _), tok, lp in zip(items, toks, lps):
                        if ep.status(ag.a) != "running" or ep.max_new_tokens(ag.a) <= 0:
                            ag.pending = None  # its last token is written; it will not decode
                            continue
                        start_chunk(ag, tok, lp)
                        end = chunk_end(ag)
                        if end:
                            finish_chunk(ag, *end)
                if ep.done:
                    return
            live = [ag for ag in agents if ag.decoding and ag.a not in pending_inject]
            if not live:
                if acted or pending_inject:
                    continue  # an append may have ended a chunk at once, or left an agent to wake
                if not ep.done:
                    notes.append("loop ended with no runnable agent and no wake while episode not done")
                return
            for ag in live:
                if ep.status(ag.a) != "running":
                    stats["idle_decode_violations"] += 1
            toks, lps = await eng.read([ag.ws for ag in live], [[ag.pending] for ag in live])
            for ag, tok, lp in zip(live, toks, lps):
                ag.chunk.append(tok)
                ag.lps.append(lp)
                ag.pending = tok
                end = chunk_end(ag)
                if end:
                    finish_chunk(ag, *end)
    try:
        await team_loop()
    except _pie_eta.InferletError as e:  # e.g. the engine's KV pool is exhausted: end the episode,
        stats["aborted"] = str(e)[:400]  # keep what was generated, and say so
        notes.append("aborted: " + str(e)[:200])

    for ag in agents:
        if ag.chunk:
            ag.traj.append({"kind": "gen", "text": _pie_model.decode(ag.chunk), "tokens": list(ag.chunk),
                            "logprobs": list(ag.lps), "mask": 1})
            ag.used += len(ag.chunk)
            ag.chunk = []
    for ag in agents:
        if ag.ws is not None:
            pie_release(ag.ws)
    pie_release(trunk)
    wall = _pie_time.monotonic() - t0
    gen = sum(ag.used for ag in agents)

    verify = None
    if input.get("verify") and not stats["aborted"]:
        verify = []
        per_seg = int(input.get("verify_tokens", 4))
        for ag in agents:
            flat, at, rec, after_inj, bare, at_bare = [], [], [], [], [], []
            for i, seg in enumerate(ag.traj):
                if seg["kind"] == "gen":
                    for j in range(min(per_seg, len(seg["tokens"]))):
                        at.append(len(flat) + j)
                        rec.append(seg["logprobs"][j])
                        after_inj.append(i > 0 and ag.traj[i - 1]["kind"] == "inject")
                        if after_inj[-1]:
                            at_bare.append((len(bare) + j, len(rec) - 1))
                flat += seg["tokens"]
                if seg["kind"] != "inject":
                    bare += seg["tokens"]
            got = await eng.rescore(flat, at) if at else []
            d = [abs(a - b) for a, b in zip(got, rec)]
            di = [x for x, f in zip(d, after_inj) if f]
            row = {"agent": ag.a, "checked": len(d), "after_inject": len(di),
                   "max_abs_diff": round(max(d), 4) if d else 0.0,
                   "mean_abs_diff": round(sum(d) / len(d), 5) if d else 0.0,
                   "mean_abs_diff_after_inject": round(sum(di) / len(di), 5) if di else 0.0}
            if at_bare:
                ctl = await eng.rescore(bare, [i for i, _ in at_bare])
                dc = [abs(a - rec[k]) for a, (_, k) in zip(ctl, at_bare)]
                fin = [x for x in dc if x < 1e30]  # top-p: a token outside the control's nucleus is -inf
                row["control_mean_abs_diff_without_injects"] = round(sum(fin) / len(fin), 4) if fin else 0.0
                row["control_outside_nucleus"] = len(dc) - len(fin)
            verify.append(row)
    eng.pipe.close()
    out = {
        "driver": "pie",
        "result": ep.result(),
        "timing": {
            "wall_s": round(wall, 3),
            "gen_tokens": gen,
            "injected_tokens": stats["injected_tokens"],
            "forward_passes": eng.passes,
            "requests": stats["steps"],  # generation chunks (one HTTP request each in the baseline)
            "decode_passes": eng.decode_passes,
            "prefill_passes": eng.prefill_passes,
            "inject_passes": stats["inject_passes"],
            "decode_rows": eng.decode_rows,
            "prefill_rows": eng.prefill_rows,
            "shared_prompt_tokens": len(shared),
            "private_prompt_tokens": sum(len(p) for p in privs),
            "mean_lanes": round(eng.lane_sum / eng.decode_passes, 3) if eng.decode_passes else 0.0,
            "max_lanes": eng.max_lanes,
            "pass_s": round(eng.pass_s, 3),
            "pass_build_s": round(eng.build_s, 3),
            "pass_submit_s": round(eng.submit_s, 3),
            "pass_wait_s": round(eng.wait_s, 3),
            "shared_prefill_s": round(t_shared, 3),
            "wakes": stats["wakes"],
            "idle_decode_violations": stats["idle_decode_violations"],
            "timeout": stats["timeout"],
            "aborted": stats["aborted"],  # an engine refusal (e.g. KV pool exhausted) ended the episode early
            "temperature": temperature,
            "top_p": top_p,
            "notes": "; ".join(notes),
            **({"verify": verify} if verify is not None else {}),
        },
        "trajectories": [ag.traj for ag in agents],
    }
    if stats["aborted"]:
        out["aborted"] = stats["aborted"]
    return out
