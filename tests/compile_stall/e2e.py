#!/usr/bin/env python3
import argparse, asyncio, json, random, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "packages/python/client/src"))
from pie_client import PieClient

a = argparse.ArgumentParser()
a.add_argument("--uri", default="ws://127.0.0.1:8080")
a.add_argument("--stall", type=float, default=1.0, help="the longest a victim's pass may take, s")
args = a.parse_args()
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, detail, flush=True)


async def run(c, prog, inp):
    out = await (await c.launch_process(prog, input=inp)).result()
    return json.loads(out) if isinstance(out, str) else out


async def main():
    c = PieClient(args.uri)
    await c.connect()
    prog = await c.install_program_bytes((HERE / "inferlet.py").read_bytes(), "compile-stall.py")
    temp = round(random.uniform(0.5, 0.95), 6)  # a program no cache holds
    shapes = {"mode": "shapes", "ks": [3, 5], "slots": 2, "temp": temp, "top_p": 0.95, "seed": 5}
    victim = asyncio.ensure_future(run(c, prog, {"mode": "victim", "seconds": 45.0}))
    await asyncio.sleep(3.0)
    t = time.monotonic()
    out = await run(c, prog, shapes)
    wall = time.monotonic() - t
    lat = (await victim)["latency"]
    first = [p[1][0] for p in out["passes"]]
    again = [p[2][0] for p in out["passes"]]
    check("a new program takes seconds to compile (the test sees a compile)", min(first) > 2.0,
          f"first submits {first}s, {wall:.1f}s in all")
    check("passes decoding beside it do not wait on it", max(lat) < args.stall,
          f"{len(lat)} passes, the longest {max(lat):.3f}s, the median {sorted(lat)[len(lat) // 2]:.4f}s")
    check("the same program submitted again does not compile again", max(again) < 0.5,
          f"second submits {again}s")
    await c.close()
    print(f"{sum(results)}/{len(results)} passed", flush=True)
    sys.exit(0 if all(results) else 1)


asyncio.run(main())
