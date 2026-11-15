#!/usr/bin/env python3
import argparse, asyncio, json, random, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "packages/python/client/src"))
from pie_client import PieClient

a = argparse.ArgumentParser()
a.add_argument("--uri", default="ws://127.0.0.1:8080")
a.add_argument("--runs", type=int, default=5)
a.add_argument("--procs", type=int, default=4)
a.add_argument("--lanes", type=int, default=8)
a.add_argument("--prompt", type=int, default=300)
a.add_argument("--decode", type=int, default=300)
args = a.parse_args()


async def run(c, prog, inp):
    out = await (await c.launch_process(prog, input=inp)).result()
    return json.loads(out) if isinstance(out, str) else out


async def main():
    c = PieClient(args.uri)
    await c.connect()
    prog = await c.install_program_bytes((HERE / "inferlet.py").read_bytes(), "kv-reclaim.py")
    failed = 0
    for r in range(args.runs):
        rng = random.Random(r)
        jobs = [{"mode": "load", "seed": 1000 * r + p, "trunk": 300, "decode": args.decode,
                 "lens": [rng.randrange(args.prompt // 3, args.prompt) for _ in range(args.lanes)]}
                for p in range(args.procs)]
        seqs, ats, lps = [], [], []
        for out in await asyncio.gather(*(run(c, prog, j) for j in jobs)):
            for segs in out["segs"]:
                seq, at, lp = [], [], []
                for i in range(0, len(segs), 3):
                    seq += segs[i]
                    at += range(len(seq), len(seq) + len(segs[i + 1]))
                    seq += segs[i + 1]
                    lp += segs[i + 2]
                seqs.append(seq), ats.append(at), lps.append(lp)
        parts = [list(range(i, len(seqs), args.procs)) for i in range(args.procs)]
        refs = await asyncio.gather(*(run(c, prog, {"mode": "ref", "seqs": [seqs[i] for i in p],
                                                    "at": [ats[i] for i in p]}) for p in parts))
        want = {i: x for p, rf in zip(parts, refs) for i, x in zip(p, rf["logprobs"])}
        bad = sum(not abs(x - y) <= 0.5 for i in range(len(seqs)) for x, y in zip(lps[i], want[i]))
        draws = sum(map(len, lps))
        failed += bad > 0
        print(f"{'FAIL' if bad else 'PASS'} run {r}: {bad}/{draws} draws off the sequence read alone", flush=True)
    await c.close()
    sys.exit(1 if failed else 0)


asyncio.run(main())
