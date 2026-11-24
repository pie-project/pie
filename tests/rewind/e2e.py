import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "packages/python/client/src"))
ap = argparse.ArgumentParser()
ap.add_argument("--uri", default="ws://127.0.0.1:8080")
ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
ap.add_argument("--tol", type=float, default=0.25)
args = ap.parse_args()

import torch
from pie_client import PieClient
from transformers import AutoModelForCausalLM, AutoTokenizer

dev = "cuda" if torch.cuda.is_available() else "cpu"
hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(dev).eval()
tok = AutoTokenizer.from_pretrained(args.model)
results = []


@torch.no_grad()
def ref(ctx):
    return torch.log_softmax(hf(torch.tensor([ctx], device=dev)).logits[0, -1].float(), -1).cpu()


def gap(ref_lp, watch, got, against=None):
    rows = [i for i, t in enumerate(watch) if i == 0 or float(ref_lp[t]) > -5.0]
    base = against if against is not None else [float(ref_lp[t]) for t in watch]
    return max(abs(got[i] - base[i]) for i in rows)


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name, detail, flush=True)


async def run(c, prog, inp):
    out = await (await c.launch_process(prog, input=inp)).result()
    return json.loads(out) if isinstance(out, str) else out


async def main():
    c = PieClient(args.uri)
    await c.connect()
    prog = await c.install_program_bytes((HERE / "inferlet.py").read_bytes(), "rewind.py")
    rng = random.Random(3)
    ids = lambda text: tok.encode(text, add_special_tokens=False)
    noise = lambda n: [rng.randrange(1000, 30000) for _ in range(n)]
    cases = [
        ("text", ids("The capital of France is"),
         ids(" Paris. The capital of Italy is Rome. The capital of Spain is"), 2, ids(" The")),
        ("code", ids("def add(a, b):\n    return"),
         ids(" a + b\n\ndef mul(a, b):\n    return a * b\n"), 4, ids("\n\nprint(add(2, 3")),
        ("noise, keep 0", noise(40), noise(9), 0, noise(3)),
        ("noise, long next", noise(17), noise(8), 3, noise(5)),
    ]
    for label, prompt, over, keep, nxt in cases:
        ctx = prompt + over[:keep] + nxt
        lps = [ref(ctx), ref(ctx + [11])]
        watch = torch.topk(lps[0], 16).indices.tolist()
        out = await run(c, prog, {"prompt": prompt, "overshoot": over, "keep": keep, "next": nxt,
                                  "probe": 11, "watch": watch})
        g, w = out["rewound"], out["clean"]
        same = all(x[0] == y[0] and abs(x[1] - y[1]) < 0.05 for x, y in zip(g, w))
        same = same and gap(lps[0], watch, g[0][2], w[0][2]) < 0.05
        check(f"{label}: rewound == never written", same, f"{[x[:2] for x in g]} {[y[:2] for y in w]}")
        err = max([abs(float(lp[x[0]]) - x[1]) for lp, x in zip(lps, g)] + [gap(lps[0], watch, g[0][2])])
        check(f"{label}: rewound == transformers", err < args.tol, f"{err:.3f}")
        stale = gap(lps[0], watch, out["stale"][2])
        check(f"{label}: overshoot left in place is visible", stale > args.tol, f"{stale:.3f}")
        check(f"{label}: rewind into the frozen prefix is refused", "frozen prefix" in out["frozen"],
              out["frozen"][:200])
    for label, prompt, n, keep, nxt in [
        ("run-ahead, text", ids("The three primary colors are"), 12, 3, ids(" and")),
        ("run-ahead, noise", noise(25), 9, 1, noise(4)),
    ]:
        inp = {"mode": "device", "prompt": prompt, "n": n, "keep": keep, "next": nxt}
        drawn = (await run(c, prog, dict(inp, watch=[0])))["drawn"]
        lp = ref(prompt + drawn[:keep] + nxt)
        watch = torch.topk(lp, 16).indices.tolist()
        out = await run(c, prog, dict(inp, watch=watch))
        g, w = out["rewound"], out["clean"]
        diff = gap(lp, watch, g[2], w[2])
        check(f"{label}: rewound == never ran past", g[0] == w[0] and diff < args.tol / 2,
              f"{g[:2]} {w[:2]} {diff:.3f}")
        err = gap(lp, watch, g[2])
        check(f"{label}: rewound == transformers", err < args.tol, f"{err:.3f}")
    await c.close()
    print(f"{sum(results)}/{len(results)} passed", flush=True)
    sys.exit(0 if all(results) else 1)


asyncio.run(main())
