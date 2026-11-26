#!/usr/bin/env python3
"""A SIGTERM mid-serving ends with no faulted lanes and every instance
closed before the scheduler stops: the runtime ends each process as the host
and waits for its teardown before the schedulers stop. Boots `--pie serve`
itself on `--config` (a home under PIE_HOME that has the model imported),
launches `--lanes` processes of the text-completion bench (`--wasm`, built
for wasm32-wasip3) of long decodes, signals the server `--after` seconds in,
and reads its log past the signal. Without the fix a desktop Vulkan
Qwen3-0.6B run logs 16 instance closes refused ("scheduler shutting down")
and 8 channel closes refused ("has no scheduler"); the phone faulted lanes."""
import argparse, asyncio, re, signal, subprocess, sys, time, tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "packages/python/client/src"))
from pie_client import PieClient

a = argparse.ArgumentParser()
a.add_argument("--pie", required=True)
a.add_argument("--config", required=True)
a.add_argument("--wasm", required=True)
a.add_argument("--log", default="sigterm_drain.log")
a.add_argument("--lanes", type=int, default=8)
a.add_argument("--after", type=float, default=3.0)
args = a.parse_args()

REFUSED = ["a lane faulted", "ring is empty", "blocked on channel", "close_instance failed",
           "scheduler shutting down", "has no scheduler", "a guest envelope refused"]
CLOSED = "every process ended and its instances closed before the schedulers stop"


async def main():
    listen = tomllib.loads(Path(args.config).read_text())["server"]["listen"]
    log = Path(args.log)
    srv = subprocess.Popen([args.pie, "--config", args.config, "serve", "--no-measure"],
                           stdout=log.open("w"), stderr=subprocess.STDOUT)
    t0 = time.time()
    while "ready at" not in log.read_text(errors="ignore"):
        if srv.poll() is not None or time.time() - t0 > 600:
            sys.exit(f"FAIL: the server did not come up (see {log})")
        await asyncio.sleep(0.05)
    c = PieClient(f"ws://{listen}")
    await c.connect()
    prog = await c.install_program(Path(args.wasm))
    for i in range(args.lanes):
        await c.launch_process(prog, input={"prompts": [f"Request {i}: continue this list forever. alpha bravo"],
                                            "max_tokens": 4096, "ignore_eos": True, "return_text": False,
                                            "temperature": 0.0})
    await asyncio.sleep(args.after)
    srv.send_signal(signal.SIGTERM)
    t = time.time()
    while srv.poll() is None and time.time() - t < 120:
        await asyncio.sleep(0.05)
    if srv.poll() is None:
        srv.kill()
        sys.exit("FAIL: the server did not exit within 120 s of SIGTERM")
    text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text(errors="ignore"))
    after = text[text.find("received SIGTERM"):]
    counts = {k: after.count(k) for k in REFUSED}
    closed = [line for line in after.splitlines() if CLOSED in line]
    bad = srv.returncode != 0 or any(counts.values()) or not closed
    print(f"{'FAIL' if bad else 'PASS'}: exit {srv.returncode} in {time.time() - t:.2f} s; "
          f"{closed[0].split('processes=')[-1][:40] + ' processes ended first' if closed else 'no close line'}; "
          + ", ".join(f"{k}: {v}" for k, v in counts.items()), flush=True)
    sys.exit(1 if bad else 0)


asyncio.run(main())
