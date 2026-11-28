#!/usr/bin/env python3
"""A burst of 64 processes of the text-completion bench, as a census's c64
runs it (one tenant-shared working set, prompts of ~128 tokens with shared
prefixes, 256 tokens each): mixed fires of several prompts beside decodes
form, which Vulkan states eager at load by name. Asserts no request fails,
the engines count no unrecorded fire, and some fire ran eager as stated.
Run against a served Qwen3-0.6B (Vulkan is where it was found; minutes),
measuring on, on a home that has booted the model twice before: without the
fix a home's first two boots may pass, its third walked 5 to 12 fires
unrecorded. `--wasm` is examples' text-completion-bench built for
wasm32-wasip3."""
import argparse, asyncio, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "packages/python/client/src"))
from pie_client import PieClient

a = argparse.ArgumentParser()
a.add_argument("--uri", default="ws://127.0.0.1:8080")
a.add_argument("--wasm", required=True)
a.add_argument("--rounds", type=int, default=4)
a.add_argument("--lanes", type=int, default=64)
args = a.parse_args()

WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa".split()


def prompt(i):
    return f"Request {i}: continue this list forever. " + " ".join(WORDS[(i * 7 + k * 3) % len(WORDS)] for k in range(110))


async def main():
    c = PieClient(args.uri)
    await c.connect()
    prog = await c.install_program(Path(args.wasm))
    n = [0]

    async def one():
        n[0] += 1
        p = await c.launch_process(prog, input={"prompts": [prompt(n[0])], "max_tokens": 256, "ignore_eos": True,
                                                "return_text": False, "temperature": 0.0})
        return await asyncio.wait_for(p.result(), 600)

    failed = 0
    for r in range(args.rounds):
        got = await asyncio.gather(*(one() for _ in range(args.lanes)), return_exceptions=True)
        bad = [g for g in got if isinstance(g, Exception)]
        failed += len(bad)
        print(f"round {r}: {len(bad)} of {args.lanes} failed{': ' + str(bad[0])[:300] if bad else ''}", flush=True)
    ok, status = await c.query("model_status", "")
    status = json.loads(status) if isinstance(status, str) else status
    count = lambda name: sum(v for k, v in status.items() if k.endswith("." + name) and isinstance(v, (int, float)))
    unrecorded, eager = count("unrecorded"), count("eager")
    await c.close()
    bad = failed or unrecorded or not eager
    print(f"{'FAIL' if bad else 'PASS'}: {failed} failed requests, {unrecorded} unrecorded fires, {eager} stated eager", flush=True)
    sys.exit(1 if bad else 0)


asyncio.run(main())
