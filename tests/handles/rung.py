#!/usr/bin/env python3
"""The activation rung under induced critical pressure (SPEC decision 6).

    PIE_HOME=/tmp/pie-test python3 tests/handles/rung.py --log /tmp/r.log

The server runs with a governor that reads the system (`memory_tolerance =
"balanced"`) and `PIE_PRESSURE_FILE` standing in for the platform's reader.
Prompts and decodes run while the file says `critical`: fires are cut to the
smaller rung and keep going, the pool lowers its rung behind them. The file
says `calm`: the pool raises its rung (moving state out of the larger
span) and fires take their full width again. Every turn's tokens are
compared with the same turns run before any pressure.
"""
import argparse, asyncio, os, re, signal, socket, subprocess, sys, tempfile, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages/python/client/src"))
from pie_client import PieClient
PIE = Path(os.environ.get("PIE_BIN", ROOT / "target/release/pie"))
HOME = Path(os.environ["PIE_HOME"])
EX = ROOT / "examples/target/wasm32-wasip3/release"
a = argparse.ArgumentParser()
a.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
a.add_argument("--log", required=True)
a.add_argument("--kv-budget", default="")
a.add_argument("--len", type=int, default=3000)
a.add_argument("--hold", type=int, default=0, help="sets kept open at the smaller rung, so state takes the span a raise empties")
a.add_argument("--hold-len", type=int, default=0)
a.add_argument("--ctx", type=int, default=0)
args = a.parse_args()
results = []
def check(name, cond, detail=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name, detail, flush=True)

async def run(c, prog, ws, tokens, n, drawn=0, seed=7):
    p = await c.launch_process(prog, input={"ws": {"$handle": "ws"}, "tokens": tokens, "max_tokens": n, "seed": seed, "drawn": drawn},
                               handles={"ws": ws}, returns={"ws": ws})
    return (await p.result())["tokens"]

def prompt(base, n):
    return [(base + 37 * i) % 150000 + 10 for i in range(n)]

async def turns(c, prog, tag):
    """Three long prompts, then their decodes, one turn at a time (lanes
    sharing a fire batch by timing, which is no rung's to keep)."""
    out, hs = {}, []
    for i in range(3):
        h = await c.working_set(); hs.append(h)
        out[f"p{i}"] = await run(c, prog, h, prompt(1000 + 777 * i, args.len), 4)
    for i, h in enumerate(hs):
        out[f"d{i}"] = await run(c, prog, h, [out[f"p{i}"][-1]], 24, drawn=4)
    for h in hs: await h.close()
    return out

def lines(pattern):
    return [l for l in Path(args.log).read_text(errors="replace").splitlines() if re.search(pattern, l)]

async def main():
    with socket.socket() as s: s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    pressure = Path(tempfile.mkdtemp()) / "pressure"
    pressure.write_text("calm")
    cfg = Path(args.log).with_suffix(".toml")
    t = (HOME / "config.toml").read_text()
    t = re.sub(r'^source = "[^"]*"', f'source = "{args.model}"', t, flags=re.M)
    t = re.sub(r'^listen = "[^"]*"', f'listen = "127.0.0.1:{port}"', t, flags=re.M)
    t = t.replace('# memory_tolerance     = "dedicated"', 'memory_tolerance = "balanced"')
    if args.ctx: t = t.replace("# max_model_len   = 32768", f"max_model_len = {args.ctx}\n#")
    if args.kv_budget: t = t.replace('# device_kv_budget     = "20GiB"', f'device_kv_budget = "{args.kv_budget}"')
    cfg.write_text(t)
    log = open(args.log, "w")
    env = dict(os.environ, PIE_HOME=str(HOME), PIE_PRESSURE_FILE=str(pressure),
               RUST_LOG=os.environ.get("RUST_LOG", "info,runtime::store::shared=debug"))
    srv = subprocess.Popen([str(PIE), "--config", str(cfg), "serve"], stdout=log, stderr=subprocess.STDOUT,
                           env=env, start_new_session=True)
    try:
        c = None
        for _ in range(300):
            if srv.poll() is not None: raise SystemExit(f"server exited; see {args.log}")
            try: c = PieClient(f"ws://127.0.0.1:{port}"); await c.connect(); break
            except Exception: c = None; await asyncio.sleep(1)
        prog = await c.install_program(EX / "working_set_handles.wasm")
        rungs = lines(r"kv tiers|activation")
        calm = await turns(c, prog, "calm")
        pressure.write_text("critical")
        for _ in range(600):
            if lines(r"the activation rung is lowered"): break
            await asyncio.sleep(0.1)
        check("critical: fires are cut to a smaller rung", lines(r"fires cut to the activation rung"), str(lines(r"fires cut")[:2]))
        check("and the pool lowers its rung behind them", lines(r"the activation rung is lowered"), str(lines(r"is lowered")[:1]))
        hard = await asyncio.wait_for(turns(c, prog, "critical"), 300)
        check("fires keep going at the smaller rung", all(len(v) > 0 for v in hard.values()), str({k: len(v) for k, v in hard.items()}))
        # A prompt cut finer folds a recurrent model's state at other rows,
        # so what follows it need not be bit for bit what the full rung drew.
        same = [k for k in calm if calm[k] == hard[k]]
        print(f"INFO at the smaller rung {len(same)}/{len(calm)} turns drew what the full rung drew; differing: {sorted(set(calm) - set(same))}", flush=True)
        held, refs = [], []
        if args.hold:
            # Down to the smallest rung first: the most span state may take.
            stated = re.search(r"rungs\S*=\S*\[\((\d+),", " ".join(lines(r"a shared pool")))
            smallest = stated.group(1) if stated else "0"
            for _ in range(600):
                if any(re.search(rf"rung\S*=\S*{smallest}\b", l) for l in lines(r"is lowered")): break
                await asyncio.sleep(0.1)
            # State takes the span above the smaller rung; half of it goes, so
            # the raise has room below to move the rest to.
            for i in range(args.hold):
                h = await c.working_set()
                await run(c, prog, h, prompt(50000 + 313 * i, args.hold_len or args.len // 4), 2)
                held.append(h)
            for h in held[::2]: await h.close()
            held = held[1::2]
            for h in held:
                f = await h.fork()
                refs.append(await run(c, prog, f, [7], 8, drawn=2))
                await f.close()
        pressure.write_text("calm")
        for _ in range(300):
            if lines(r"the activation rung is raised"): break
            await asyncio.sleep(0.1)
        check("calm: the rung comes back", lines(r"the activation rung is raised"), str(lines(r"is raised")[:1]))
        if args.hold:
            moved = [l for l in lines(r"the activation rung is raised") if not re.search(r"moved\S*=\S*0\b", l)]
            # Whether any state lay in the span depends on how full the pool
            # got below it; the unit tests force the move.
            print(f"INFO the raise moved state: {bool(moved)} {lines(r'is raised')[-1:]}", flush=True)
            after = [await run(c, prog, h, [7], 8, drawn=2) for h in held]
            check("and what it moved reads as it did", after == refs, f"{sum(a == r for a, r in zip(after, refs))}/{len(refs)}")
            for h in held: await h.close()
        back = await asyncio.wait_for(turns(c, prog, "back"), 300)
        check("and fires run at full width as before", back == calm, str([k for k in calm if calm[k] != back[k]]))
        await c.close()
    finally:
        if srv.poll() is None:
            os.killpg(srv.pid, signal.SIGTERM)
            try: srv.wait(20)
            except subprocess.TimeoutExpired: os.killpg(srv.pid, signal.SIGKILL)
    print(f"{sum(results)}/{len(results)} passed")
asyncio.run(main())
