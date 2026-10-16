"""Trace-driven policy simulator: evaluate strategies on RECORDED lanes, with no GPU.

Every fixed-width run in results/ logged each lane's length, how it ended and its
answer. This pools them per problem (real samples from the real model) and replays
a policy by drawing lanes without replacement. That gives thousands of resamples,
so quality-at-fixed-compute comes with error bars, and a strategy can be tuned
in seconds before it is spent on the engine.

    python3 simulate.py results/e1-fixed*.json
"""

import glob
import json
import math
import random
import statistics
import sys

HERE = __file__.rsplit("/", 1)[0] or "."
SRC = open(HERE + "/main.py").read()
_ns = {"math": math}
exec(SRC[SRC.index("def _betacf"):SRC.index("# ============================================================================\n# RUNTIME")], _ns)
leader_prob = _ns["leader_prob"]


def load_pool(paths):
    pool, gold = {}, {}
    for path in paths:
        r = json.load(open(path))
        for p in r["problems"]:
            gold[p["id"]] = p["gold"]
        for e in r["trace"]:
            if e["action"] == "DONE" and e.get("kind", "solve") == "solve":
                pool.setdefault(e["problem"], []).append((e["tokens"], e["why"], e["answer"]))
    return pool, gold


class Lane:
    __slots__ = ("cost", "ans")

    def __init__(self, cost, ans):
        self.cost, self.ans = cost, ans


def draw(rng, lanes, deadline):
    """A recorded lane as the policy would have run it under a straggler deadline."""
    toks, why, ans = lanes
    if deadline is not None and toks > deadline:
        return Lane(deadline, None)
    return Lane(toks, ans if why == "answered" else None)


def plurality(votes):
    return max(votes.items(), key=lambda kv: (kv[1], -list(votes).index(kv[0])))[0] if votes else None


def run_fixed(rng, lanes, n, deadline=None):
    picks = rng.sample(lanes, min(n, len(lanes)))
    got = [draw(rng, l, deadline) for l in picks]
    votes = {}
    for g in got:
        if g.ans is not None:
            votes[g.ans] = votes.get(g.ans, 0) + 1
    return plurality(votes), sum(g.cost for g in got)


def run_adaptive(rng, lanes, w0=2, step=2, wmax=12, delta=0.08, vmin=2, deadline=None, giveup=6):
    order = rng.sample(lanes, len(lanes))
    used, cost, votes = 0, 0, {}
    n = w0
    while True:
        batch = order[used:used + n]
        if not batch:
            break
        got = [draw(rng, l, deadline) for l in batch]
        used += len(batch)
        cost += sum(g.cost for g in got)
        for g in got:
            if g.ans is not None:
                votes[g.ans] = votes.get(g.ans, 0) + 1
        ranked = sorted(votes.values(), reverse=True)
        answered = sum(votes.values())
        if ranked and answered >= vmin:
            c1 = ranked[0]
            c2 = ranked[1] if len(ranked) > 1 else 0
            if leader_prob(c1, c2) >= 1 - delta:
                break
        if used >= wmax:
            break
        if used >= giveup and not votes:
            break
        n = min(step, wmax - used)
    return plurality(votes), cost


def evaluate(pool, gold, fn, trials=200, seed=1):
    rng = random.Random(seed)
    accs, costs = [], []
    ids = list(pool)
    for _ in range(trials):
        ok, tot = 0, 0
        for pid in ids:
            ans, cost = fn(rng, pool[pid])
            ok += 1 if ans is not None and ans == gold[pid] else 0
            tot += cost
        accs.append(ok / len(ids))
        costs.append(tot / len(ids))
    return statistics.mean(accs), statistics.pstdev(accs), statistics.mean(costs)


def main(paths):
    pool, gold = load_pool(paths)
    n = len(pool)
    sizes = sorted(len(v) for v in pool.values())
    print(f"{n} problems, lanes per problem: min {sizes[0]} median {sizes[len(sizes)//2]} max {sizes[-1]}")
    pts = []
    print("\nfixed width (plain majority vote)")
    for k in (1, 2, 3, 4, 6, 8):
        a, sd, c = evaluate(pool, gold, lambda r, l, k=k: run_fixed(r, l, k))
        pts.append((c, a, f"fixed{k}"))
        print(f"  N={k:2d}   acc {a*100:5.1f}% +-{sd*100:3.1f}   tokens/problem {c:7.1f}")
    print("\nfixed width + straggler deadline")
    for dl in (320, 384, 448, 512):
        for k in (3, 4, 6):
            a, sd, c = evaluate(pool, gold, lambda r, l, k=k, dl=dl: run_fixed(r, l, k, dl))
            pts.append((c, a, f"fixed{k}+dl{dl}"))
            print(f"  N={k} deadline {dl}   acc {a*100:5.1f}% +-{sd*100:3.1f}   tokens/problem {c:7.1f}")
    print("\nadaptive (sequential widening, sparse-prior consensus)")
    for dl in (None, 384, 448, 512):
        for w0, step, wmax in ((2, 2, 8), (2, 2, 12), (3, 2, 12)):
            for vmin, delta in ((2, 0.08), (3, 0.05)):
                a, sd, c = evaluate(pool, gold, lambda r, l: run_adaptive(r, l, w0, step, wmax, delta, vmin, dl))
                pts.append((c, a, f"adaptive w0={w0} step={step} max={wmax} vmin={vmin} d={delta} dl={dl}"))
                print(f"  dl={str(dl):4s} w0={w0} step={step} max={wmax:2d} vmin={vmin} d={delta}   acc {a*100:5.1f}% +-{sd*100:3.1f}   tokens/problem {c:7.1f}")
    pts.sort()
    front, best = [], -1
    for c, a, name in pts:
        if a > best + 1e-9:
            front.append((c, a, name))
            best = a
    print("\nPareto frontier (cheapest policy for each accuracy level)")
    for c, a, name in front:
        print(f"  {c:7.1f} tok/problem   {a*100:5.1f}%   {name}")


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(glob.glob(HERE + "/results/e1-fixed*.json")))
