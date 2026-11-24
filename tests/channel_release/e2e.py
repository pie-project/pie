#!/usr/bin/env python3
"""A process that runs compat::generate turn after turn holds the same device
memory after its hundredth turn as after its tenth: each turn lets go of the
channels it made. Run beside a `pie serve` alone on its GPU:

    (cd inferlet && cargo build --release --target wasm32-wasip3)
    python e2e.py --uri ws://127.0.0.1:8080 --gpu 0
"""
import argparse, asyncio, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "packages/python/client/src"))
from pie_client import Event, PieClient

a = argparse.ArgumentParser()
a.add_argument("--uri", default="ws://127.0.0.1:8080")
a.add_argument("--gpu", type=int, default=0)
a.add_argument("--warm", type=int, default=16)
a.add_argument("--turns", type=int, default=128)
a.add_argument("--slack-mib", type=int, default=64)
a.add_argument("--wasm", default=str(HERE / "inferlet/target/wasm32-wasip3/release/channel_release.wasm"))
args = a.parse_args()


def used_mib():
    out = subprocess.run(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=memory.used",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


async def main():
    c = PieClient(args.uri)
    await c.connect()
    prog = await c.install_program(args.wasm)
    p = await c.launch_process(prog, {"turns": args.turns, "tokens": 16})
    marks = {}
    for turn in range(args.turns):
        ev, val = await p.recv()
        if ev != Event.Message:
            sys.exit(f"FAIL turn {turn}: {ev} {str(val)[:200]}")
        if turn + 1 in (args.warm, args.turns):
            await asyncio.sleep(1)
            marks[turn + 1] = used_mib()
        await p.signal("next")
    await p.result()
    await c.close()
    grown = marks[args.turns] - marks[args.warm]
    ok = grown <= args.slack_mib
    print(f"{'PASS' if ok else 'FAIL'}: {marks[args.warm]} MiB after turn {args.warm}, "
          f"{marks[args.turns]} MiB after turn {args.turns} ({grown:+d} MiB)", flush=True)
    sys.exit(0 if ok else 1)


asyncio.run(main())
