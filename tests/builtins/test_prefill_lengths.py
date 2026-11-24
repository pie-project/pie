#!/usr/bin/env python3
"""One program at every prompt length, on a served model.

    python tests/builtins/test_prefill_lengths.py --engine vulkan

Boots `pie serve` (`PIE_BIN`, default target/release/pie) with a fresh
PIE_HOME and CUDA_CACHE_PATH, then runs the built-in `prefill` program at
three prompt lengths (`--lengths`, default 37, 300 and 1500 tokens). After
each it reads the model status: the first request may register its program
(`programs.registered`), every later length registers none, and no request
builds anything after ready (`engine.built_at_register`,
`engine.built_in_fire`, `engine.compiled`, `engine.unrecorded`: each 0).
Exits non-zero, naming the counter, otherwise.
"""
import argparse, asyncio, json, os, shutil, signal, socket, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages/python/client/src"))
from pie_client import PieClient

a = argparse.ArgumentParser()
a.add_argument("--engine", default="vulkan", choices=("cuda", "rocm", "vulkan", "wgpu", "metal", "xla"))
a.add_argument("--model", default="Qwen/Qwen3-0.6B")
a.add_argument("--lengths", default="37,300,1500")
a.add_argument("--keep-home", action="store_true")
args = a.parse_args()
PIE = Path(os.environ.get("PIE_BIN", ROOT / "target/release/pie"))
BUILT = ("built_at_register", "built_in_fire", "compiled", "unrecorded")


async def main():
    home = Path(tempfile.mkdtemp(prefix="pie-prefill-lengths."))
    cache = Path(tempfile.mkdtemp(prefix="pie-prefill-lengths-cc."))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    cfg = home / "cfg.toml"
    cfg.write_text(f'[server]\nlisten = "127.0.0.1:{port}"\nverbose = false\n\n[model]\nname = "default"\n'
                   f'source = "{args.model}"\n\n[engine]\ntype = "{args.engine}"\ndevice = ["{args.engine}:0"]\n'
                   '\n[runtime]\n')
    env = dict(os.environ, PIE_HOME=str(home), CUDA_CACHE_PATH=str(cache))
    subprocess.run([str(PIE), "--config", str(cfg), "model", "import", args.model], env=env, check=True,
                   capture_output=True, timeout=1800)
    log_path = home / "serve.log"
    srv = subprocess.Popen([str(PIE), "--config", str(cfg), "serve"], stdout=open(log_path, "w"),
                           stderr=subprocess.STDOUT, env=env, start_new_session=True)
    failed = []
    try:
        t0 = time.monotonic()
        while "Server ready" not in log_path.read_text(errors="ignore"):
            if srv.poll() is not None or time.monotonic() - t0 > 2400:
                raise SystemExit(f"the server did not come up: see {log_path}")
            await asyncio.sleep(0.1)
        c = PieClient(f"ws://127.0.0.1:{port}")
        await asyncio.wait_for(c.connect(), 60)

        async def status():
            ok, st = await asyncio.wait_for(c.query("model_status", ""), 60)
            if not ok:
                raise SystemExit(f"model_status: {st}")
            st = json.loads(st)
            pick = lambda name: next(v for k, v in st.items() if k.endswith(name))
            return {"registered": pick(".programs.registered"),
                    **{name: pick(f".engine.{name}") for name in BUILT}}

        before = await status()
        print(json.dumps({"after": "ready", **before}), flush=True)
        for at, n in enumerate(int(x) for x in args.lengths.split(",")):
            tokens = [(1000 + 37 * j) % 30000 + 100 for j in range(n)]
            p = await asyncio.wait_for(c.launch_process("prefill", input={"tokens": tokens}), 600)
            out = await asyncio.wait_for(p.result(), 600)
            now = await status()
            registered = now["registered"] - before["registered"]
            print(json.dumps({"after": f"prefill {n}", "result": out, **now, "registered_by_it": registered}),
                  flush=True)
            if out.get("tokens") != n:
                failed.append(f"prefill {n} answered {out}")
            if registered > (1 if at == 0 else 0):
                failed.append(f"prefill {n} registered {registered} program(s)")
            failed += [f"prefill {n}: engine.{name} = {now[name]}" for name in BUILT if now[name]]
            before = now
        await asyncio.wait_for(c.close(), 30)
    finally:
        if srv.poll() is None:
            os.killpg(srv.pid, signal.SIGTERM)
            try:
                srv.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(srv.pid, signal.SIGKILL)
                srv.wait()
        shutil.rmtree(cache, ignore_errors=True)
        if not args.keep_home and not failed:
            shutil.rmtree(home, ignore_errors=True)
    print(json.dumps({"failed": failed, "log": None if not failed else str(log_path)}))
    sys.exit(1 if failed else 0)


asyncio.run(main())
