#!/usr/bin/env python3
"""The activation rung by demand (SPEC decision 6).

    PIE_HOME=/tmp/pie-test python3 tests/handles/rung.py --log /tmp/r.log

At rest the pool sits at its smallest rung: the room the larger rungs'
activation would take is the cache's. A long prompt asks for the rung its
pieces take and the pool raises it (moving what state lies in the larger
span down first); the prompt runs on at the smaller rung's width meanwhile
and takes the full width once the raise lands. With no long prompt left the
pool lowers again. `--hold N` keeps N sets open at the smallest rung first,
so state takes the span and the raise must move it (or have the planner
free room for it); what it moved must read as it did.
"""
import argparse, asyncio, os, re, signal, socket, subprocess, sys, time
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
a.add_argument("--budget-device", default="", help="[engine.memory] budget_device: weights and fires come off it before kv")
a.add_argument("--len", type=int, default=3000)
a.add_argument("--hold", type=int, default=0, help="sets kept open at the smallest rung, so state takes the span a raise empties")
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

async def turns(c, prog):
    """Three long prompts, then their decodes, one turn at a time."""
    out, hs = {}, []
    for i in range(3):
        h = await c.working_set(); hs.append(h)
        out[f"p{i}"] = await run(c, prog, h, prompt(1000 + 777 * i, args.len), 4)
    for i, h in enumerate(hs):
        out[f"d{i}"] = await run(c, prog, h, [out[f"p{i}"][-1]], 24, drawn=4)
    for h in hs: await h.close()
    return out

def lines(pattern):
    return [re.sub(r"\x1b\[[0-9;]*m", "", l) for l in Path(args.log).read_text(errors="replace").splitlines() if re.search(pattern, l)]

async def wait_for(pattern, count, seconds):
    for _ in range(int(seconds * 10)):
        if len(lines(pattern)) >= count: return True
        await asyncio.sleep(0.1)
    return False

def smallest():
    stated = re.search(r"rungs=\[\((\d+),", " ".join(lines(r"a shared pool")))
    return stated.group(1) if stated else None

def lowered_to(rung):
    return [l for l in lines(r"the activation rung is lowered") if re.search(rf"rung=\s*{rung}\b", l)]

async def main():
    with socket.socket() as s: s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    cfg = Path(args.log).with_suffix(".toml")
    t = (HOME / "config.toml").read_text()
    t = re.sub(r'^source = "[^"]*"', f'source = "{args.model}"', t, flags=re.M)
    t = re.sub(r'^listen = "[^"]*"', f'listen = "127.0.0.1:{port}"', t, flags=re.M)
    if args.ctx: t = t.replace("# max_context     = 32768", f"max_context = {args.ctx}\n#")
    if args.budget_device: t = t.replace('# budget_device         = "20GiB"', f'budget_device = "{args.budget_device}"')
    cfg.write_text(t)
    log = open(args.log, "w")
    env = dict(os.environ, PIE_HOME=str(HOME),
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
        low = smallest()
        await wait_for(r"the activation rung is lowered", 1, 30)
        check("at rest the pool sits at its smallest rung", low and lowered_to(low), str(lines(r"is lowered")[:1]))
        raised0 = len(lines(r"the activation rung is raised"))
        first = await asyncio.wait_for(turns(c, prog), 300)
        check("a long prompt raises the rung", len(lines(r"the activation rung is raised")) > raised0,
              str(lines(r"is raised")[raised0:raised0 + 1]))
        check("and its turns run", all(len(v) > 0 for v in first.values()), str({k: len(v) for k, v in first.items()}))
        lowers = len(lowered_to(low))
        check("with no long prompt left the rung lowers again", await wait_for(rf"the activation rung is lowered.*rung=\s*{low}\b", lowers + 1, 30) or len(lowered_to(low)) > 1,
              str(lowered_to(low)[-1:]))
        held, refs = [], []
        if args.hold:
            # State takes the span above the smallest rung; half of it goes, so
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
            raised1 = len(lines(r"the activation rung is raised"))
            again = await asyncio.wait_for(turns(c, prog), 300)
            check("a long prompt raises the rung over held state", len(lines(r"the activation rung is raised")) > raised1,
                  str(lines(r"is raised")[raised1:raised1 + 1]))
            moved = [l for l in lines(r"the activation rung is raised")[raised1:] if not re.search(r"moved\S*=\S*0\b", l)]
            print(f"INFO the raise moved state: {bool(moved)} {moved[:1]}", flush=True)
            print(f"INFO a raise short of room asked the planner: {bool(lines(r'the activation rung waits'))}", flush=True)
            # A raise short of room has the planner free it as for pages: the
            # policy may discard a held set (its key says it may), and what it
            # leaves must read as it did.
            after, gone = [], 0
            for h, ref in zip(held, refs):
                try:
                    after.append((await run(c, prog, h, [7], 8, drawn=2), ref))
                except Exception as e:
                    if "dead" not in str(e): raise
                    gone += 1
            print(f"INFO held sets the policy discarded to fund the raise: {gone}/{len(held)}", flush=True)
            check("and what the raise moved reads as it did", after and all(a == r for a, r in after), f"{sum(a == r for a, r in after)}/{len(after)}")
            check("and its turns run", all(len(v) > 0 for v in again.values()))
            for h in held:
                try: await h.close()
                except Exception: pass
        back = await asyncio.wait_for(turns(c, prog), 300)
        same = [k for k in first if first[k] == back[k]]
        print(f"INFO {len(same)}/{len(first)} turns drew what the first run drew (a recurrent model's prompt cut at other rows folds at other rows); differing: {sorted(set(first) - set(same))}", flush=True)
        check("fires keep running", all(len(v) > 0 for v in back.values()))
        await c.close()
    finally:
        if srv.poll() is None:
            os.killpg(srv.pid, signal.SIGTERM)
            try: srv.wait(20)
            except subprocess.TimeoutExpired: os.killpg(srv.pid, signal.SIGKILL)
    print(f"{sum(results)}/{len(results)} passed")
asyncio.run(main())
