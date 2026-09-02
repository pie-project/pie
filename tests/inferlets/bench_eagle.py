"""Wall-clock of the device-resident speculative loop against itself at
`k = 0` and against `naive-baseline`, warm, on whichever drafting artifact
`--model` names. Reports ms/token — decode rate is what a draft head buys
or costs; nothing here asserts.

    python tests/inferlets/bench_eagle.py --engine metal --model <overlay.zt> \
        [--engine-option ...]
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from conftest import run_inferlet, run_tests  # noqa: E402

PROMPT = "Write a long, detailed story about a dragon who learns to bake bread."
MAX_TOKENS = int(os.environ.get("BENCH_TOKENS", "160"))
REPEATS = int(os.environ.get("BENCH_REPEATS", "2"))
GREEDY = {"temperature": 0.01, "seed": 7}


async def _timed(client, args, name, inputs):
    started = time.perf_counter()
    out = await run_inferlet(client, name, inputs, timeout=args.timeout)
    took = time.perf_counter() - started
    r = json.loads(out[out.find("{"):])
    n = len(r["tokens"])
    return took, n, r


async def bench(client, args):
    spec = lambda k: {"prompt": PROMPT, "max_tokens": MAX_TOKENS, "k": k}
    # Warm: the artifact into the page cache, the seats filled.
    await _timed(client, args, "mtp-speculative-decoding", spec(0))
    depth = (await _timed(client, args, "mtp-speculative-decoding", spec(1)))[2]["depth"]
    arms = []
    # BENCH_ARMS="k=2;k=2,pad=1;k=0,pad=2": explicit arms instead of the ladder.
    if os.environ.get("BENCH_ARMS"):
        for arm in os.environ["BENCH_ARMS"].split(";"):
            kv = dict(p.split("=") for p in arm.split(","))
            arms.append((arm, "mtp-speculative-decoding", {**spec(int(kv.get("k", 0))), "pad": int(kv.get("pad", 0))}))
        rows = {label: [] for label, _, _ in arms}
        for _ in range(REPEATS):
            for label, name, inputs in arms:
                took, n, r = await _timed(client, args, name, inputs)
                rows[label].append((took, n, r))
                print(f"  {label:14s} {n:4d} tokens in {took:6.2f}s = {1000 * took / max(n, 1):6.1f} ms/token  rounds={r['rounds']} ({1000 * took / max(r['rounds'], 1):5.1f} ms/round) accepted={r['accepted']}/{r['drafted']}")
        return
    if os.environ.get("BENCH_BASELINE", "1") != "0":
        arms.append(("baseline", "naive-baseline", {"prompt": PROMPT, "max_tokens": MAX_TOKENS, **GREEDY}))
    for k in range(0, depth + 1):
        arms.append((f"loop k={k}", "mtp-speculative-decoding", spec(k)))
    # The fire SHAPE of `k = depth` with no draft accepted: what the width
    # itself costs, read against what the head's chain costs.
    arms.append((f"k=0 pad={depth}", "mtp-speculative-decoding", {**spec(0), "pad": depth}))
    print(f"  depth = {depth}, {MAX_TOKENS} tokens asked, {REPEATS} repeats each, interleaved")
    rows = {label: [] for label, _, _ in arms}
    for _ in range(REPEATS):
        for label, name, inputs in arms:
            took, n, r = await _timed(client, args, name, inputs)
            rows[label].append((took, n, r))
    for label, _, _ in arms:
        best = min(rows[label], key=lambda t: t[0] / max(t[1], 1))
        took, n, r = best
        extra = ""
        if "rounds" in r:
            extra = f"  rounds={r['rounds']} accepted={r['accepted']}/{r['drafted']}"
        print(f"  {label:10s} {n:4d} tokens in {took:6.2f}s  = {1000 * took / max(n, 1):6.1f} ms/token  ({n / took:5.2f} tok/s){extra}")
    print(f"  text (loop k=0): {rows['loop k=0'][0][2]['text'][:120]!r}")


if __name__ == "__main__":
    run_tests([bench], "Speculative loop wall-clock")
