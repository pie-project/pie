"""agent-os: an operating system for reasoning agents, as one Pie inferlet.

Pie is the kernel: it owns KV pages, copy-on-write state, paging and the
forward pass. This program is the user-space half. It runs many agents
("lanes") against one arena of KV pages and decides, every wave, which ones
run, which die, which are forked and what they share.

    kernel   Arena (refcounted KV pages)  Root (a shared, page-aligned prefix)
             Lane (an agent: its page chain + recurrent state)
             wave() (ONE forward pass that decodes every running lane)
    runtime  Scheduler: spawn / fork / kill / share, a token ledger, a trace
    brain    Policy: decides what to do at each wave boundary
    tasks    Solver and verifier prompts, answer extraction

Why waves. Lanes decoded on separate pipelines timeslice (aggregate tok/s is
flat); lanes decoded as rows of one pass batch (aggregate tok/s grows with
width). A wave is K tokens (one KV page) for every running lane, so a fork at
a wave boundary is exact and free: the child lists the parent's pages.
"""

import asyncio
import json
import math
import re
import time

from inferlet import chat, model, session
from inferlet.eta import (
    Channel,
    ForwardKind,
    ForwardPass,
    KvGeometry,
    Pipeline,
    RsWorkingSet,
    WorkingSet,
    channel_capacity,
    dtype,
    gather,
    gumbel_max,
    intrinsics,
    iota,
    kv_page_size,
    reduce_argmax,
    reshape,
)

PS = kv_page_size()
SLOTS_PER_RS = 2  # recurrent-state seats one RsWorkingSet holds (one per posted frame)
WAVE = PS  # tokens per wave: exactly one KV page, so lanes stay page-aligned
_T0 = time.perf_counter()


def now():
    return time.perf_counter() - _T0


def cdiv(a, b):
    return -(-a // b)


# ============================================================================
# KERNEL: arena, roots, lanes, waves
# ============================================================================


class Arena:
    """One WorkingSet of logical pages with reference counts. A page is
    free when no root or lane lists it, so a shared prefix is held once."""

    def __init__(self, pages):
        self.ws = WorkingSet()
        grant = self.ws.reserve(pages)
        self.base = grant.start
        self.free = list(range(self.base + pages - 1, self.base - 1, -1))
        self.ref = {}
        self.capacity = pages
        self.peak = 0
        self.frontier = 0       # highest page id ever handed out, +1: what the engine must back

    def alloc(self, n=1):
        if len(self.free) < n:
            raise MemoryError(f"arena out of pages ({self.capacity})")
        out = [self.free.pop() for _ in range(n)]
        for p in out:
            self.ref[p] = 1
        self.peak = max(self.peak, self.capacity - len(self.free))
        self.frontier = max(self.frontier, max(out) + 1 - self.base)
        return out

    def retain(self, pages):
        for p in pages:
            self.ref[p] += 1

    def release(self, pages):
        for p in pages:
            self.ref[p] -= 1
            if self.ref[p] == 0:
                del self.ref[p]
                self.free.append(p)

    def used(self):
        return self.capacity - len(self.free)


class Root:
    """A shared prefix: a page-aligned run of KV and the recurrent state after
    it. `cur` is the one prompt token not yet in the KV; lanes start by
    feeding it, which is also how each lane draws its own first token."""

    def __init__(self, name, pages, rs, n, cur, prompt_tokens):
        self.name, self.pages, self.rs, self.n, self.cur = name, pages, rs, n, cur
        self.prompt_tokens = prompt_tokens


class Lane:
    _next = 0

    def __init__(self, root, pages, rs, n, cur, kind, parent=None):
        Lane._next += 1
        self.id = Lane._next
        self.root, self.pages, self.rs, self.n, self.cur = root, pages, rs, n, cur
        self.kind, self.parent = kind, parent
        self.toks = []          # tokens generated so far
        self.done = False
        self.why_done = ""
        self.born_wave = 0
        self.rows_run = 0       # rows this lane occupied in waves (its compute)
        self.text = ""
        self.answer = None


def pad_to_page(encode, build, tag):
    """Token list for a prompt whose length-1 is a multiple of the page size,
    by padding the system text with newlines: the shared prefix is whole
    pages, so a fork never has to copy a partial one."""
    words = ["Work", "carefully", "and", "check", "each", "step", "before", "you", "answer", "the", "question",
             "with", "full", "attention", "to", "detail", "so", "that", "the", "result", "is", "correct"]
    for pad in range(0, 8 * PS):
        filler = (" " + " ".join(words[i % len(words)] for i in range(pad))) if pad else ""
        toks = encode(build(filler))
        if (len(toks) - 1) % PS == 0 and len(toks) > 1:
            return toks
    raise ValueError(f"could not page-align the {tag} prompt")


async def _feed_chunk(arena, pipe, rs, pages, start, tokens):
    """Run `tokens` at positions [start, start+m) of one sequence listing `pages`."""
    m = len(tokens)
    end = start + m
    need = cdiv(end, PS)
    row = pages[:need]
    ch_tok = Channel.from_(tokens, dtype.i32)
    ch_ind = Channel.from_([0, m], dtype.u32).named("embed_indptr")
    ch_pos = Channel.from_(range(start, end), dtype.u32).named("positions")
    ch_pages = Channel.from_(row, dtype.u32).named("pages")
    ch_pind = Channel.from_([0, len(row)], dtype.u32).named("page_indptr")
    ch_ws = Channel.from_([row[p // PS] for p in range(start, end)], dtype.u32).named("w_slot")
    ch_wo = Channel.from_([p % PS for p in range(start, end)], dtype.u32).named("w_off")
    ch_kv = Channel.from_([end], dtype.u32).named("kv_len")
    out = Channel([1], dtype.i32).named("next_token")
    written = [row[p // PS] for p in range(start, end)]
    fwd = ForwardPass()
    fwd.embed(ch_tok, ch_ind)
    fwd.bind_state(
        arena.ws,
        KvGeometry(kv_len=ch_kv, pages=ch_pages, page_indptr=ch_pind, w_slot=ch_ws,
                   w_off=ch_wo, positions=ch_pos, writable_pages=(min(written), max(written) + 1)),
        rs,
    )

    @fwd.epilogue
    def _():
        out.put(reshape(reduce_argmax(intrinsics.logits()), [1]))

    pipe.submit(fwd)
    return await out.take_scalar()


async def make_root(arena, name, tokens):
    """Prefill tokens[:-1] once into fresh shared pages; tokens[-1] stays pending."""
    n = len(tokens) - 1
    if n % PS:
        raise ValueError("root prefix must be page-aligned")
    pages = arena.alloc(n // PS)
    hybrid = model.pass_kind() != ForwardKind.ATTENTION
    rs = [RsWorkingSet()] if hybrid else []
    pipe = Pipeline()
    step = 8 * PS
    for a in range(0, n, step):
        b = min(a + step, n)
        await _feed_chunk(arena, pipe, rs, pages, a, tokens[a:b])
    pipe.close()
    return Root(name, pages, rs, n, tokens[-1], len(tokens))


def spawn_lane(arena, root, kind, setup_pipe, born_wave=0):
    arena.retain(root.pages)
    rs = [r.fork(setup_pipe) for r in root.rs]
    lane = Lane(root, list(root.pages), rs, root.n, root.cur, kind)
    lane.born_wave = born_wave
    return lane


def fork_lane(arena, parent, setup_pipe, born_wave=0):
    """A child that shares every page the parent has written (copy-on-write)."""
    arena.retain(parent.pages)
    rs = [r.fork(setup_pipe) for r in parent.rs]
    lane = Lane(parent.root, list(parent.pages), rs, parent.n, parent.cur, parent.kind, parent.id)
    lane.toks = list(parent.toks)
    lane.born_wave = born_wave
    return lane


def kill_lane(arena, lane, why):
    if lane.pages:
        arena.release(lane.pages)
        lane.pages = []
    lane.rs = []
    lane.done = True
    lane.why_done = lane.why_done or why


async def wave(arena, lanes, seed, temperature):
    """ONE forward pass per step decodes `WAVE` tokens for every lane in `lanes`.

    Each lane is a row listing its own page chain (shared prefix pages are the
    same ids across rows). Returns tokens[lane_index][WAVE].
    """
    B = len(lanes)
    K = WAVE
    for lane in lanes:
        extra = cdiv(lane.n + K, PS) - len(lane.pages)
        if extra > 0:
            lane.pages += arena.alloc(extra)
    flat = [p for lane in lanes for p in lane.pages]
    ptr = [0]
    for lane in lanes:
        ptr.append(ptr[-1] + len(lane.pages))
    v = model.output_vocab_size()
    cap = channel_capacity()
    tok_in = Channel.from_([lane.cur for lane in lanes], dtype.i32).named("tok_in")
    lanes_ind = Channel.from_(range(B + 1), dtype.u32).named("embed_indptr")
    pos = Channel.from_([lane.n for lane in lanes], dtype.u32).named("positions")
    kv_len = Channel.from_([lane.n + 1 for lane in lanes], dtype.u32).named("kv_len")
    pages = Channel.from_(flat, dtype.u32).named("pages")
    pind = Channel.from_(ptr, dtype.u32).named("page_indptr")
    w_slot = Channel.from_([lane.pages[lane.n // PS] for lane in lanes], dtype.u32).named("w_slot")
    w_off = Channel.from_([lane.n % PS for lane in lanes], dtype.u32).named("w_off")
    rng = Channel.from_([seed, 0], dtype.u32).named("rng")
    tok_out = Channel([B], dtype.i32).capacity(cap).named("tok_out")
    row_ids = Channel.from_(flat, dtype.u32).named("row_ids")
    row_start = Channel.from_(ptr[:-1], dtype.u32).named("row_start")
    rss = [r for lane in lanes for r in lane.rs]

    writes = [lane.pages[lane.n // PS] for lane in lanes]
    pipe = Pipeline()
    fwd = ForwardPass()
    fwd.embed(tok_in, lanes_ind)
    fwd.bind_state(
        arena.ws,
        KvGeometry(kv_len=kv_len, pages=pages, page_indptr=pind, w_slot=w_slot,
                   w_off=w_off, positions=pos, writable_pages=(min(writes), max(writes) + 1)),
        rss,
    )

    @fwd.epilogue
    def _():
        length = kv_len.take()
        r = rng.take()
        lg = reshape(intrinsics.logits(), [B, v])
        sc = lg if temperature == 1.0 else lg / temperature
        token = gumbel_max(sc, r) if temperature > 0 else reduce_argmax(lg)
        ids = row_ids.take()
        starts = row_start.take()
        tok_in.put(token)
        kv_len.put(length + 1)
        pos.put(length)
        w_slot.put(gather(ids, starts + length // PS))
        w_off.put(length % PS)
        tok_out.put(token)
        rng.put(r + iota(2))
        row_ids.put(ids)
        row_start.put(starts)
        pages.put(pages.take())
        pind.put(pind.take())

    out = [[] for _ in range(B)]

    async def on_step():
        t = await tok_out.take_host()
        for i in range(B):
            out[i].append(t[i])
        return True

    await pipe.run_ahead(fwd, K, on_step)
    for i, lane in enumerate(lanes):
        lane.n += K
        lane.cur = out[i][-1]
        lane.rows_run += K
    return out


# ============================================================================
# KERNEL SELF-TEST
# ============================================================================


async def kernel_test(inp):
    """Greedy checks: (1) two 32-token waves == one 64-token run of the same
    lanes; (2) a lane forked mid-generation continues exactly as its parent."""
    arena = Arena(512)
    sys_text = "You are a careful math tutor."
    user_text = "Janet has 3 apples and buys 5 more, then gives 2 away. How many are left? Think step by step."
    build = lambda pad: list(chat.system_user(sys_text + pad, user_text)) + list(chat.cue())
    toks = pad_to_page(lambda x: list(x), build, "test")
    root = await make_root(arena, "test", toks)
    res = {"prompt_tokens": len(toks), "prefix_pages": len(root.pages)}

    # (1) continuity: lane A runs two waves; lane B runs the same two waves
    setup = Pipeline()
    a = spawn_lane(arena, root, "solve", setup)
    b = spawn_lane(arena, root, "solve", setup)
    setup.close()
    t0 = time.perf_counter()
    o1 = await wave(arena, [a, b], 1, 0.0)
    o2 = await wave(arena, [a, b], 2, 0.0)
    res["continuity"] = {
        "waves_s": round(time.perf_counter() - t0, 3),
        "a_equals_b": o1[0] + o2[0] == o1[1] + o2[1],
        "sample": model.decode(o1[0] + o2[0])[:160],
    }

    # (2) fork at a wave boundary: the child must continue like the parent
    setup = Pipeline()
    child = fork_lane(arena, a, setup)
    setup.close()
    p3 = await wave(arena, [a], 3, 0.0)
    c3 = await wave(arena, [child], 3, 0.0)
    res["fork"] = {
        "child_equals_parent": p3[0] == c3[0],
        "shared_pages_at_fork": len(a.pages) - 1,
        "arena_pages_used": arena.used(),
    }
    # (3) batching: 8 identical greedy lanes in a wave all match each other
    setup = Pipeline()
    many = [spawn_lane(arena, root, "solve", setup) for _ in range(8)]
    setup.close()
    outs = await wave(arena, many, 4, 0.0)
    res["batch8_all_equal"] = all(o == outs[0] for o in outs)
    for ln in [a, b, child] + many:
        kill_lane(arena, ln, "test")
    res["pages_after_kill"] = arena.used()
    return res


# ============================================================================
# TASKS: prompts, answer extraction, normalization
# ============================================================================

SOLVER_SYS = {
    "math": "You are a careful problem solver. Solve the problem directly in at most 12 short lines. Do not second-guess or re-check a step you already did. Finish with the final answer as \\boxed{ANSWER}.",
    "mc": "You are a careful problem solver. Reason briefly, then end with a line of the form 'Answer: X' where X is the letter of the correct option.",
}
JUDGE_SYS = {
    "math": "You are a careful problem solver and referee. Independent solvers disagree. Work the problem out yourself, step by step but briefly, and give the final answer as \\boxed{ANSWER}.",
    "mc": "You are a careful problem solver and referee. Independent solvers disagree. Work the problem out yourself, briefly, and end with a line of the form 'Answer: X' where X is the letter of the correct option.",
}


def last_boxed(text):
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j = i + 7
    depth, k = 1, j
    while k < len(text) and depth:
        depth += (text[k] == "{") - (text[k] == "}")
        k += 1
    return text[j:k - 1] if depth == 0 else None


_MC = re.compile(r"Answer\s*[:：]\s*\(?\**([A-Ea-e])\b")


def extract(text, kind):
    if kind == "mc":
        m = _MC.findall(text)
        return m[-1].upper() if m else None
    return last_boxed(text)


def normalize(ans, kind):
    if ans is None:
        return None
    if kind == "mc":
        return ans.strip().upper()[:1]
    s = ans.strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    for a, b in (("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("\\left", ""), ("\\right", ""), ("\\!", ""),
                 ("\\,", ""), ("\\ ", ""), ("^\\circ", ""), ("^{\\circ}", ""), ("\\%", ""), ("$", ""), ("\\$", "")):
        s = s.replace(a, b)
    s = s.replace(" ", "").rstrip(".")
    m = re.fullmatch(r"[A-Za-z]=(.+)", s)
    if m:
        s = m.group(1)
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        f = float(s)
        s = str(int(f)) if f == int(f) else str(f)
    return s


def is_looping(toks, n=24, window=520, times=4):
    """True when the last `n` tokens already occurred `times`-1 more times in the
    recent window: a lane stuck rewriting the same thought."""
    if len(toks) < 160:
        return False
    tail = toks[-n:]
    win = toks[-window:]
    first = tail[0]
    hits = 0
    for i in range(len(win) - n + 1):
        if win[i] == first and win[i:i + n] == tail:
            hits += 1
            if hits >= times:
                return True
    return False


def _betacf(a, b, x):
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for k in range(1, 200):
        k2 = 2 * k
        aa = k * (b - k) * x / ((qam + k2) * (a + k2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + k) * (qab + k) * x / ((a + k2) * (qap + k2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-12:
            break
    return h


def betai(a, b, x):
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


PRIOR = 0.3   # sparse prior: a wrong answer is rarely repeated, so a repeat is real evidence


def leader_prob(c1, c2):
    """P(the top answer's share beats the runner-up's), with a sparse Dirichlet prior."""
    a, b = c1 + PRIOR, c2 + PRIOR
    return 1.0 - betai(a, b, 0.5)


# ============================================================================
# RUNTIME: problems, scheduler, ledger, trace
# ============================================================================


class Problem:
    def __init__(self, pid, spec, policy):
        self.pid, self.spec, self.policy = pid, spec, policy
        self.kind = spec.get("kind", "math")
        self.roots = {}
        self.lanes = []
        self.votes = {}            # normalized answer -> weight
        self.first_seen = {}       # normalized answer -> order of first vote
        self.finished = False
        self.final = None
        self.why_final = ""
        self.started = 0
        self.judged = False
        self.waves_alive = 0
        self.waiting = False
        self.spent = 0             # decode tokens charged to this problem

    def running(self):
        return [l for l in self.lanes if not l.done]

    def tally(self):
        ranked = sorted(self.votes.items(), key=lambda kv: (-kv[1], self.first_seen[kv[0]]))
        return ranked

    def confidence(self):
        r = self.tally()
        if not r:
            return 0.0
        c1 = r[0][1]
        c2 = r[1][1] if len(r) > 1 else 0
        return leader_prob(c1, c2)

    def vote(self, ans, weight=1.0):
        if ans is None:
            return
        if ans not in self.first_seen:
            self.first_seen[ans] = len(self.first_seen)
        self.votes[ans] = self.votes.get(ans, 0) + weight


class Fixed:
    """Baseline: N independent samples, then plurality vote. N=1 is a single agent."""

    one_shot = True

    def __init__(self, n):
        self.n = n
        self.name = f"fixed{n}"

    def decide(self, p, sup):
        if not p.lanes:
            return [("spawn", "solve", self.n, f"fixed width {self.n}")]
        if not p.running():
            return [("finish", "all lanes done")]
        return []


class Adaptive:
    """Bayesian consensus controller. Starts narrow, widens only while the
    answers disagree, stops the moment the leader is statistically clear,
    and hands a split vote to a few judge lanes that share one blackboard
    prefix holding the competing candidates."""

    def __init__(self, w0=2, step=2, wmax=12, delta=0.08, judge=True, judge_n=3, judge_at=8, giveup=6, straggler=True, vmin=2):
        self.w0, self.step, self.wmax, self.delta, self.giveup = w0, step, wmax, delta, giveup
        self.straggler, self.vmin = straggler, vmin
        self.judge, self.judge_n, self.judge_at = judge, judge_n, judge_at
        self.name = f"adaptive(w0={w0},max={wmax},d={delta}{',judge' if judge else ''})"

    def decide(self, p, sup):
        acts = []
        conf = p.confidence()
        tally = p.tally()
        done_n = len([l for l in p.lanes if l.done and l.kind == "solve" and l.answer is not None])
        if not p.lanes:
            return [("spawn", "solve", self.w0, f"open with {self.w0} agents")]
        if self.straggler:
            dl = sup.deadline()
            if dl is not None:
                for lane in p.running():
                    if len(lane.toks) > dl:
                        acts.append(("kill", lane, f"straggler: {len(lane.toks)} tokens with no answer, past the learned deadline of {dl} (answering lanes finish well before)"))
                if acts and not [l for l in p.running() if len(l.toks) <= dl]:
                    pass
        if done_n >= self.vmin and conf >= 1 - self.delta:
            lead = tally[0]
            return [("finish", f"consensus: '{lead[0]}' has {lead[1]:g} votes, P(clear leader)={conf:.2f}>={1 - self.delta:.2f}")]
        killed_now = {a[1].id for a in acts if a[0] == "kill"}
        running = [l for l in p.running() if l.id not in killed_now]
        if running:
            return acts
        if p.started >= self.giveup and not tally:
            return acts + [("finish", f"cut losses: {p.started} agents tried and none produced an answer; budget goes to winnable problems")]
        # everyone alive has finished and there is no consensus
        if self.judge and not p.judged and p.started >= self.judge_at and len(tally) >= 2:
            return acts + [("judge", self.judge_n, f"split vote after {p.started} solvers: {tally[0][0]}x{tally[0][1]:g} vs {tally[1][0]}x{tally[1][1]:g}")]
        if p.started >= self.wmax:
            return acts + [("finish", f"width cap {self.wmax} reached; plurality")]
        n = min(self.step, self.wmax - p.started)
        why = "no answer yet" if not tally else f"split vote ({tally[0][0]}x{tally[0][1]:g}, P={conf:.2f}): widen by {n}"
        return acts + [("spawn", "solve", n, why)]


class Council:
    """n independent solvers; if they do not agree, j judges read the blackboard (the
    candidate answers) from one shared prefix. Cooperation costs one prefill, not one per agent."""

    one_shot = True

    def __init__(self, n=4, j=3, delta=0.08, vmin=2):
        self.n, self.j, self.delta, self.vmin = n, j, delta, vmin
        self.name = f"council({n}+{j})"

    def decide(self, p, sup):
        if not p.lanes:
            return [("spawn", "solve", self.n, f"{self.n} independent solvers")]
        if p.running():
            return []
        tally = p.tally()
        done_n = len([l for l in p.lanes if l.kind == "solve" and l.answer is not None])
        if p.judged:
            return [("finish", "judges weighed in")]
        if done_n >= self.vmin and p.confidence() >= 1 - self.delta:
            return [("finish", f"solvers agree: '{tally[0][0]}' P(clear leader)={p.confidence():.2f}")]
        if len(tally) >= 2:
            return [("judge", self.j, f"solvers split: {tally[0][0]}x{tally[0][1]:g} vs {tally[1][0]}x{tally[1][1]:g}")]
        return [("finish", "no usable answers" if not tally else "single answer; nothing to referee")]


class Ledger:
    def __init__(self):
        self.decode_tokens = 0       # rows x WAVE, every row of every wave
        self.useful_tokens = 0       # tokens up to and including the answer / EOS
        self.prefill_tokens = 0      # tokens actually read into KV
        self.prefill_saved = 0       # tokens a non-sharing engine would have re-read
        self.waves = 0
        self.wave_seconds = 0.0
        self.spawned = self.forked = self.killed = self.judged = 0
        self.peak_rows = 0
        self.est_saved_by_kill = 0
        self.wave_log = []


class Supervisor:
    def __init__(self, arena, problems, budget, max_rows, temperature, max_new, stream, seed, state_slots=0):
        self.arena, self.problems = arena, problems
        self.state_slots = state_slots     # 0 = attention-only model, no recurrent state to count
        self.budget, self.max_rows, self.temperature = budget, max_rows, temperature
        self.max_new, self.stream, self.seed = max_new, stream, seed
        self.led = Ledger()
        self.trace = []
        self.stop = set(chat.stop_tokens())
        self.lens = []               # token lengths of finished lanes (for estimates)
        self.ans_lens = []           # token lengths of lanes that ended with an answer
        self.wave_i = 0

    # ---- bookkeeping -------------------------------------------------------
    def est_len(self):
        if len(self.lens) >= 3:
            s = sorted(self.lens)
            return s[len(s) // 2]
        return self.max_new * 0.6

    def remaining(self, lane):
        """Expected tokens a running lane still has to run: the median length of finished lanes
        that went at least as far, less what it has already used."""
        so_far = len(lane.toks)
        longer = sorted(n for n in self.lens if n >= so_far)
        if len(longer) >= 3:
            return max(longer[len(longer) // 2] - so_far, 0)
        return max(self.max_new - so_far, 0)

    def committed(self):
        """Decode tokens spent plus the expected tail of every lane still running."""
        est = self.est_len()
        tail = sum(max(est - len(l.toks), WAVE) for p in self.problems for l in p.running())
        return self.led.decode_tokens + tail

    def deadline(self):
        """Learned straggler deadline: a generous multiple of how long answering lanes take."""
        if len(self.ans_lens) < 6:
            return None
        s = sorted(self.ans_lens)
        q = s[min(len(s) - 1, int(len(s) * 0.85))]
        return max(int(q * 1.3), 4 * WAVE)

    def mem_fit(self):
        """How many more lanes the arena can promise to carry to max_new."""
        cap = cdiv(self.max_new, PS) + 2
        promised = 0
        for q in self.problems:
            if q.finished:
                continue
            for l in q.running():
                promised += max(cap - (len(l.pages) - len(l.root.pages)), 0)
        room = len(self.arena.free) - promised - 24      # 24 pages of headroom for roots
        fit = max(room // cap, 0)
        if self.state_slots:
            # every recurrent-state set (a root's, a lane's) holds SLOTS_PER_RS seats of a fixed pool
            held = 0
            for q in self.problems:
                if q.finished:
                    continue
                held += sum(len(r.rs) for r in q.roots.values()) + sum(len(l.rs) for l in q.running())
            fit = min(fit, max((self.state_slots - 8 - held * SLOTS_PER_RS) // SLOTS_PER_RS, 0))
        return fit

    def log(self, p, action, why, **kw):
        ev = {"t": round(now(), 2), "wave": self.wave_i, "problem": p.pid if p else None, "action": action, "why": why}
        ev.update(kw)
        self.trace.append(ev)
        if self.stream:
            session.send(json.dumps(ev))

    # ---- actions -----------------------------------------------------------
    async def make_root(self, p, name, system, user, tag):
        build = lambda pad: list(chat.system_user(system + pad, user)) + list(chat.cue())
        toks = pad_to_page(lambda x: x, build, tag)
        root = await make_root(self.arena, f"{p.pid}:{name}", toks)
        p.roots[name] = root
        self.led.prefill_tokens += root.n
        return root

    async def spawn(self, p, root_name, n, kind, why):
        if root_name not in p.roots:
            sys_text = (SOLVER_SYS if root_name == "solve" else JUDGE_SYS)[p.kind]
            await self.make_root(p, root_name, sys_text, p.spec["question"] if root_name == "solve" else p.judge_user, root_name)
        root = p.roots[root_name]
        setup = Pipeline()
        made = []
        for _ in range(n):
            lane = spawn_lane(self.arena, root, kind, setup, self.wave_i)
            p.lanes.append(lane)
            made.append(lane.id)
            p.started += 1 if kind == "solve" else 0
            self.led.spawned += 1
            self.led.prefill_saved += root.n
        setup.close()
        if getattr(p.policy, "one_shot", False):
            root.rs = []      # nothing will fork from this root again: free its state seats now
        self.log(p, "SPAWN", why, lanes=made, kind=kind, shared_prefix_tokens=root.n,
                 prefill_saved=root.n * n)

    def kill(self, p, lane, why, est_saved=0):
        kill_lane(self.arena, lane, why)
        self.led.killed += 1
        self.led.est_saved_by_kill += est_saved
        self.log(p, "KILL", why, lane=lane.id, tokens_so_far=len(lane.toks), est_saved_tokens=int(est_saved))

    def finish(self, p, why):
        if p.finished:
            return
        saved = 0
        for lane in p.running():
            rem = self.remaining(lane)
            saved += rem
            self.kill(p, lane, "problem finished", rem)
        tally = p.tally()
        p.final = tally[0][0] if tally else None
        p.finished = True
        p.why_final = why
        self.log(p, "FINISH", why, final=p.final, votes=[[a, round(w, 2)] for a, w in tally],
                 est_saved_tokens=int(saved))
        for root in p.roots.values():
            self.arena.release(root.pages)
            root.rs = []
        p.roots = {}

    async def apply(self, p, acts):
        for act in acts:
            if act[0] == "finish":
                self.finish(p, act[1])
                return
            if act[0] == "kill":
                lane = act[1]
                if not lane.done:
                    self.kill(p, lane, act[2], self.remaining(lane))
                continue
            if act[0] == "spawn":
                _, rname, n, why = act
                room = max(int((self.budget - self.committed()) // max(self.est_len(), 1)), 0)
                first = not p.lanes
                n = n if first else min(n, room)
                fit = self.mem_fit()
                if n <= 0:
                    self.log(p, "DENY", "global budget: no room for another lane", asked=act[2])
                    if not p.running():
                        self.finish(p, "budget exhausted; plurality so far")
                    return
                if fit < (n if first else 1):
                    if not p.waiting:
                        p.waiting = True
                        self.log(p, "WAIT", f"memory: {fit} lane(s) fit, {n} wanted; problem stays CREATED until pages free up")
                    return
                n = min(n, fit)
                p.waiting = False
                await self.spawn(p, rname, n, "solve", why)
            elif act[0] == "judge":
                _, n, why = act
                p.judged = True
                tally = p.tally()[:3]
                cand = "\n".join(f"- Solver answer {i + 1}: {a}  ({w:g} of {p.started} solvers)" for i, (a, w) in enumerate(tally))
                p.judge_user = p.spec["question"] + "\n\nOther solvers disagreed:\n" + cand + "\n\nDecide carefully which is right, or find the correct answer."
                room = max(int((self.budget - self.committed()) // max(self.est_len(), 1)), 0)
                n = min(n, room, self.mem_fit())
                if n <= 0:
                    self.log(p, "DENY", "no room for judges (budget or memory)")
                    self.finish(p, "budget exhausted; plurality so far")
                    return
                await self.spawn(p, "judge", n, "judge", f"blackboard: {why}")
                self.led.judged += n

    # ---- the wave loop -----------------------------------------------------
    def settle(self, p, lane, new_tokens):
        """Account one lane's wave; detect EOS and early answers."""
        cut = len(new_tokens)
        for i, t in enumerate(new_tokens):
            if t in self.stop:
                cut = i
                break
        used = cut if cut < len(new_tokens) else len(new_tokens)
        lane.toks += new_tokens[:cut]
        self.led.useful_tokens += used
        lane.text += model.decode(new_tokens[:cut]) if cut else ""
        hit_stop = cut < len(new_tokens)
        ans = extract(lane.text, p.kind)
        if ans is not None and re.sub(r"[^a-z]", "", ans.lower()) in ("answer", "finalanswer", "youranswer", "x"):
            lane.why_done = "placeholder"     # it wrote the template, not a result
            lane.done = True
        elif ans is not None and (p.kind == "mc" or last_boxed(lane.text) is not None):
            lane.answer = normalize(ans, p.kind)
            lane.why_done = "answered"
            lane.done = True
        elif hit_stop:
            lane.why_done = "eos"
            lane.done = True
        elif len(lane.toks) >= self.max_new:
            lane.why_done = "max length"
            lane.done = True
        elif is_looping(lane.toks):
            lane.why_done = "looping"
            lane.done = True
        if lane.done:
            self.lens.append(len(lane.toks))
            if lane.why_done == "answered":
                self.ans_lens.append(len(lane.toks))
            weight = 1.5 if lane.kind == "judge" else 1.0
            p.vote(lane.answer, weight)
            kill_lane(self.arena, lane, lane.why_done)
            self.log(p, "DONE", lane.why_done, lane=lane.id, kind=lane.kind, tokens=len(lane.toks), answer=lane.answer,
                     **({"tail": lane.text[-240:]} if lane.why_done != "answered" else {}))

    async def run(self):
        idle = 0
        while True:
            active = [p for p in self.problems if not p.finished]
            if not active:
                break
            # the strategist wakes: one decision pass per active problem, most uncertain first
            active.sort(key=lambda q: q.confidence())
            for p in active:
                await self.apply(p, p.policy.decide(p, self))
            rows = [(p, l) for p in self.problems if not p.finished for l in p.running()]
            if not rows:
                idle += 1
                if idle > 400:
                    for p in active:
                        self.finish(p, "no runnable lanes; plurality so far")
                continue
            idle = 0
            # time-slice: least-served lanes first; the rest are suspended this wave
            rows.sort(key=lambda pl: (pl[1].rows_run, pl[1].id))
            batch = rows[: self.max_rows]
            self.led.peak_rows = max(self.led.peak_rows, len(batch))
            t0 = time.perf_counter()
            outs = await wave(self.arena, [l for _, l in batch], self.seed + self.wave_i, self.temperature)
            dt = time.perf_counter() - t0
            self.led.waves += 1
            self.led.wave_log.append([len(batch), round(dt, 3)])
            self.led.wave_seconds += dt
            self.led.decode_tokens += len(batch) * WAVE
            self.wave_i += 1
            for (p, lane), toks in zip(batch, outs):
                p.spent += WAVE
                self.settle(p, lane, toks)
            if self.stream and self.wave_i % 4 == 0:
                session.send(json.dumps({"t": round(now(), 1), "wave": self.wave_i, "rows": len(batch),
                                         "tok_s": round(len(batch) * WAVE / dt), "pages": self.arena.used(),
                                         "active_problems": len([q for q in self.problems if not q.finished])}))
        return self


# ============================================================================
# ENTRY
# ============================================================================


def make_policy(spec):
    kind = spec.get("name", "adaptive")
    if kind == "fixed":
        return Fixed(int(spec.get("n", 4)))
    if kind == "council":
        return Council(int(spec.get("n", 4)), int(spec.get("j", 3)), float(spec.get("delta", 0.08)), int(spec.get("vmin", 2)))
    return Adaptive(w0=int(spec.get("w0", 2)), step=int(spec.get("step", 2)), wmax=int(spec.get("wmax", 12)),
                    delta=float(spec.get("delta", 0.08)), judge=bool(spec.get("judge", True)),
                    judge_n=int(spec.get("judge_n", 3)), judge_at=int(spec.get("judge_at", 8)),
                    giveup=int(spec.get("giveup", 6)), straggler=bool(spec.get("straggler", False)),
                    vmin=int(spec.get("vmin", 2)))


async def solve(inp):
    specs = inp["problems"]
    if inp.get("solver_sys"):
        SOLVER_SYS["math"] = inp["solver_sys"]
    pol = inp.get("policy", {"name": "adaptive"})
    max_new = int(inp.get("max_new", 320))
    max_rows = int(inp.get("max_rows", 32))
    temperature = float(inp.get("temperature", 0.7))
    pages = int(inp.get("arena_pages", 3800))
    per_problem = float(inp.get("budget_per_problem", 1e12))
    budget = per_problem * len(specs)
    arena = Arena(pages)
    problems = [Problem(i, s, make_policy(pol)) for i, s in enumerate(specs)]
    sup = Supervisor(arena, problems, budget, max_rows, temperature, max_new, bool(inp.get("stream", False)), int(inp.get("seed", 17)), int(inp.get("state_slots", 0)))
    t0 = time.perf_counter()
    await sup.run()
    wall = time.perf_counter() - t0
    out = []
    for p in problems:
        gold = normalize(p.spec.get("gold"), p.kind) if p.spec.get("gold") is not None else None
        out.append({"id": p.pid, "question": p.spec["question"][:400], "final": p.final, "gold": gold, "correct": (p.final == gold) if gold is not None else None,
                    "lanes": len(p.lanes), "tokens": p.spent, "votes": [[a, w] for a, w in p.tally()], "why": p.why_final,
                    **({"texts": [l.text for l in p.lanes[:2]]} if inp.get("texts") else {})})
    n = len(out)
    ok = sum(1 for o in out if o["correct"])
    led = sup.led
    summary = {
        "policy": problems[0].policy.name if problems else None, "problems": n, "correct": ok,
        "accuracy": round(ok / n, 4) if n else None, "wall_s": round(wall, 2),
        "decode_tokens": led.decode_tokens, "decode_tokens_per_problem": round(led.decode_tokens / n, 1) if n else None,
        "useful_tokens": led.useful_tokens, "prefill_tokens": led.prefill_tokens, "prefill_saved_by_sharing": led.prefill_saved,
        "waves": led.waves, "agg_tok_s": round(led.decode_tokens / led.wave_seconds, 1) if led.wave_seconds else None,
        "peak_rows": led.peak_rows, "spawned": led.spawned, "killed": led.killed, "judge_lanes": led.judged,
        "est_tokens_saved_by_kills": int(led.est_saved_by_kill), "arena_pages_peak": arena.peak, "arena_frontier_pages": arena.frontier, "wave_log": led.wave_log,
    }
    return {"summary": summary, "problems": out, "trace": sup.trace if inp.get("trace", True) else []}


async def main(input: dict):
    if isinstance(input.get("cfg"), str):          # `pie run -- --cfg '<json>'`: arguments arrive as strings
        input = {**input, **json.loads(input["cfg"])}
    mode = input.get("mode", "kernel_test")
    if mode == "kernel_test":
        return await kernel_test(input)
    if mode == "solve":
        return await solve(input)
    raise ValueError(f"unknown mode {mode}")
