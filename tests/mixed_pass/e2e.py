#!/usr/bin/env python3
import argparse, asyncio, json, random, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "packages/python/client/src"))
a = argparse.ArgumentParser()
a.add_argument("--uri", default="ws://127.0.0.1:8080")
a.add_argument("--model", default="Qwen/Qwen3-0.6B")
a.add_argument("--procs", type=int, default=4)
args = a.parse_args()

import torch
from transformers import AutoModelForCausalLM
from pie_client import PieClient

dev = "cuda" if torch.cuda.is_available() else "cpu"
hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(dev).eval()


@torch.no_grad()
def ref(ctx):
    return torch.log_softmax(hf(torch.tensor([ctx], device=dev)).logits[0, -1].float(), -1).cpu()


def case(seed, lens, chunk_lens):
    rng = random.Random(seed)
    draw = lambda n: [rng.randrange(1000, 30000) for _ in range(n)]
    return {"prompts": [draw(n) for n in lens], "chunks": [draw(m) for m in chunk_lens], "probe": 11}


CASES = [
    case(1, [21, 34, 27], [7, 0, 0]),
    case(2, [21, 34, 27, 19], [0, 9, 0, 0]),
    case(3, [21, 34, 27], [0, 0, 12]),
    case(4, [18, 25, 30, 22, 26], [0, 5, 0, 16, 0]),
    case(5, [12 + i for i in range(12)], [0, 0, 0, 31, 0, 0, 0, 0, 2, 0, 0, 0]),
]


async def main():
    c = PieClient(args.uri)
    await c.connect()
    prog = await c.install_program_bytes((HERE / "inferlet.py").read_bytes(), "mixed-pass.py")
    failed = 0
    for n, inp in enumerate(CASES):
        ctx = [p + ch for p, ch in zip(inp["prompts"], inp["chunks"])]
        want = [ref(x) for x in ctx]
        want_probe = [ref(x + [inp["probe"]]) for x in ctx]
        for mode in ("mixed", "split", "alone"):
            outs = await asyncio.gather(*[(await c.launch_process(prog, input=dict(inp, mode=mode))).result()
                                          for _ in range(args.procs)])
            gaps = [max(abs(float(want[i][t]) - lp), abs(float(want_probe[i][pt]) - plp))
                    for o in map(lambda o: json.loads(o) if isinstance(o, str) else o, outs)
                    for i, ((t, lp), (pt, plp)) in enumerate(zip(o["read"], o["probe"]))]
            bad = sum(g > 0.25 for g in gaps)
            failed += bad > 0
            print(f"{'FAIL' if bad else 'PASS'} case {n} {mode}: {bad}/{len(gaps)} lanes off, max gap {max(gaps):.3f}")
    await c.close()
    sys.exit(1 if failed else 0)


asyncio.run(main())
