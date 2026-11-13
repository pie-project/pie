#!/usr/bin/env python3
"""Handles on a real engine: resume, fork, eviction, quota, every SDK.

    PIE_HOME=/tmp/pie-test python3 tests/handles/e2e.py --log /tmp/e.log \\
        --ctx 512 --kv-budget 176MiB

Build first: `pie` for the engine, `working-set-handles` for wasm32-wasip3 in
examples/, and `--example handles` of pie-client for t_clients. A token here is drawn by position, so a turn split across
processes must draw exactly what one process does. A working set on the device
takes its kv pages, an rs list slot and its fixed record from one pool: on
Qwen3.5-0.8B 176MiB holds 9 or so of t_evict's 470-token sets, short of its 11.
A policy gets its own runs: `--policy keep-handles --only t_keep --ctx 512
--kv-budget 136MiB` (8 or so 300-token sets), and `--policy keep-handles --only
t_weak --ctx 512 --kv-budget 116MiB`. The host's limit (t_quota, t_close,
t_weak's discard) is the host tier, which `--kv-budget` does not reach:
`--host-budget 60MiB --only t_quota,t_close --ctx 512 --kv-budget 116MiB` sets
host_kv_budget so the sets at rest fill it.
"""
import argparse, asyncio, json, os, re, signal, socket, subprocess, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages/python/client/src"))
from pie_client import PieClient
from pie_client.client import HandleError
PIE = Path(os.environ.get("PIE_BIN", ROOT / "target/release/pie"))
HOME = Path(os.environ["PIE_HOME"])  # a home of its own: `pie config init`, then
                                     # `pie model import`; languages/ for t_script
EX = ROOT / "examples/target/wasm32-wasip3/release"
a = argparse.ArgumentParser()
a.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
a.add_argument("--ctx", type=int, default=0)
a.add_argument("--kv-budget", default=""); a.add_argument("--log", required=True)
a.add_argument("--only", default=""); a.add_argument("--policy", default="")
a.add_argument("--host-budget", default="")
args = a.parse_args()
PROMPT = list(range(1000, 1060))
results = []
def check(name, cond, detail=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name, detail, flush=True)

async def run(c, prog, ws, tokens, n, drawn=0, ret=None, seed=7):
    p = await c.launch_process(prog, input={"ws": {"$handle": "ws"}, "tokens": tokens, "max_tokens": n, "seed": seed, "drawn": drawn},
                               handles={"ws": ws}, returns={"ws": ret or ws})
    return (await p.result())["tokens"]

async def t_resume(c, prog):
    a = await c.working_set(); whole = await run(c, prog, a, PROMPT, 32)
    b = await c.working_set(); first = await run(c, prog, b, PROMPT, 12)
    st = await b.stat()
    rest = await run(c, prog, b, [first[-1]], 20, drawn=12)
    check("resume across processes equals one run", first + rest == whole, f"stat={st}")
    await a.close(); await b.close()
    # fork rolls back
    e = await c.working_set()
    f = await run(c, prog, e, PROMPT, 12)
    ck = await e.fork()
    r1 = await run(c, prog, e, [f[-1]], 20, drawn=12)
    r2 = await run(c, prog, ck, [f[-1]], 20, drawn=12)
    check("a fork rolls back", f + r1 == whole and r1 == r2)
    r3 = await run(c, prog, ck, [999], 8, drawn=99)
    check("a fork takes its own path", r3 != r1[:8], str(r3[:4]))
    te, tc = (await e.stat())["tokens"], (await ck.stat())["tokens"]
    check("and leaves the other where it stood", tc == te + 8, f"{te} {tc}")
    for h in (e, ck): await h.close()
    return whole

async def t_lazy(c, prog, whole):
    u = await c.working_set(); st = await u.stat()
    check("new is unborn", st["state"] == "unborn" and st["engine"] is None, str(st))
    v = await u.fork(); check("fork of unborn is unborn", (await v.stat())["state"] == "unborn")
    x = await run(c, prog, v, PROMPT, 32); check("a forked unborn runs", x == whole)
    await u.close()
    try: await u.stat(); check("closed is unknown", False)
    except HandleError as e: check("closed is unknown", e.code == "unknown", str(e))
    await v.close()

async def t_order(c, prog, whole):
    w = await c.working_set()
    t = asyncio.ensure_future(run(c, prog, w, PROMPT, 12))
    await asyncio.sleep(0)
    await asyncio.sleep(0.01)
    # sent while it runs: waits for the handle to land
    t0 = time.time(); ch = await w.fork(); waited = time.time() - t0
    first = await t
    st = await ch.stat()
    check("a fork of a lent handle waits for it", st["tokens"] >= 72 - 1, f"waited={waited:.3f}s stat={st}")
    r = await run(c, prog, ch, [first[-1]], 20, drawn=12)
    check("and sees what the process wrote", first + r == whole)
    await w.close(); await ch.close()

async def t_dead(c, prog, whole):
    w = await c.working_set()
    from pie_client.client import WorkingSet
    never = WorkingSet(c, c._next_handle())
    p = await c.launch_process(prog, input={"ws": {"$handle": "ws"}, "tokens": PROMPT, "max_tokens": 2, "seed": 7},
                               handles={"ws": w}, returns={"ws": never, "other": w})
    await p.result()
    try: await w.stat(); check("an ungiven return is dead", False)
    except HandleError as e: check("an ungiven return is dead", e.code == "dead", str(e))
    st = await never.stat(); check("what was given landed", st["state"] == "parked")
    try:
        await c.launch_process(prog, input={"ws": {"$handle": "ws"}, "tokens": PROMPT, "max_tokens": 1, "seed": 7}, handles={"ws": w}, returns={"ws": w})
        check("a dead handle refuses a launch", False)
    except HandleError as e: check("a dead handle refuses a launch", e.code == "dead", str(e)[:80])
    await never.close(); await w.close()
    # a launch that fails before running gives its handle back
    k = await c.working_set(); x = await run(c, prog, k, PROMPT, 12)
    try:
        await c.launch_process("no-such-program", input={}, handles={"ws": k}, returns={"ws": k}); 
    except Exception: pass
    check("a refused launch keeps the handle", (await k.stat())["state"] == "parked")
    await k.close()

async def t_evict(c, prog, whole):
    w = await c.working_set(); first = await run(c, prog, w, PROMPT, 12)
    pair = await w.fork()
    hogs = []
    for i in range(9):
        h = await c.working_set(); hogs.append(h)
        await run(c, prog, h, list(range(2000 + i, 2470 + i)), 4, seed=i)
    st = await w.stat()
    check("pressure moved the idle handle off the device", not st["resident"], f"{st}")
    check("a fork pair at rest does not pin what it shares", not (await pair.stat())["resident"])
    fk2 = await w.fork(); check("a fork of an evicted handle is evicted", not (await fk2.stat())["resident"])
    r = await run(c, prog, w, [first[-1]], 20, drawn=12)
    check("a restored handle resumes exactly", first + r == whole)
    r2 = await run(c, prog, fk2, [first[-1]], 20, drawn=12)
    check("so does its evicted fork", r2 == r)
    # A peer's restore moved pages under `pair`, which wakes with nothing owed.
    r3 = await run(c, prog, pair, [first[-1]], 20, drawn=12)
    check("nor does the pair lose it", r3 == r)
    for h in hogs + [w, fk2, pair]: await h.close()

async def t_quota(c, prog, whole):
    # Past the host tier (--host-budget) the host is short: with no worker
    # policy listening, the default responder lets sets at rest go, the one
    # keyed lowest first, so no launch starves and none waits. (A policy
    # that keeps them all instead has a launch fail at the seat deadline
    # with a "quota:" error naming host_kv_budget.)
    hs, refused = [], None
    for i in range(24):
        h = await c.working_set()
        try:
            await run(c, prog, h, PROMPT, 2); hs.append(h)
        except Exception as e:
            refused = str(e); await h.close(); break
    gone, live = [], []
    for h in hs:
        try: await h.stat(); live.append(h)
        except HandleError as e: gone.append(e)
    check("past the host no launch starves", refused is None, f"{len(hs)} rest; {refused and refused[:120]}")
    if args.host_budget:  # past what the host tier holds only where it is small
        check("the default responder lets sets at rest go, by name", gone and all(e.code == "dead" and "default responder" in str(e) for e in gone), f"{len(gone)}/{len(hs)} {gone[:1]}")
    x = await run(c, prog, live[0], [999], 2, drawn=50)
    check("what rests still runs", len(x) == 2)
    await live[0].close()
    y = await c.working_set(); z = await run(c, prog, y, PROMPT, 32)
    check("and closing one makes room", z == whole)
    for h in live[1:] + [y]: await h.close()

async def t_keep(c, prog, whole):
    N = 300
    kept = await c.working_set(); await run(c, prog, kept, list(range(3000, 3000 + N)), 2)
    await kept.set_props({"keep": 10}); await asyncio.sleep(0.2)
    hogs = []
    for i in range(10):
        h = await c.working_set(); hogs.append(h)
        await run(c, prog, h, list(range(4000 + 500 * i, 4000 + 500 * i + N)), 2)
        rest = [(await x.stat())["resident"] for x in hogs]
        if not all(rest): break
    check("the policy's key outranks age at rest", (await kept.stat())["resident"] and not all(rest), str(rest))
    await kept.set_props({"keep": -1, "head": 160}); await asyncio.sleep(0.2)
    before = await run(c, prog, kept, [1], 2, drawn=80)
    check("props stay through a lend", (await kept.stat())["resident"])
    h = await c.working_set(); hogs.append(h); await run(c, prog, h, list(range(9000, 9000 + N)), 2)
    st = await kept.stat()
    check("a key on a head outlasts its tail", 0 < st["swapped_pages"] <= (st["tokens"] - 160) // 16 + 1, str(st))
    z = await run(c, prog, kept, [1], 2, drawn=81)
    check("and a set split so resumes", len(z) == 2 and (await kept.stat())["resident"], str(z))
    bad = None
    try: await kept.set_props({"": 1})
    except HandleError as e: bad = e
    check("a bad prop is refused", bad is not None and bad.code == "invalid", str(bad))
    t = asyncio.ensure_future(run(c, prog, kept, [1], 2, drawn=40))
    await asyncio.sleep(0); await asyncio.sleep(0.005)
    await kept.set_props({"keep": 30}); await t
    check("sent while lent, they land with it", True)
    for h in hogs + [kept]: await h.close()

async def t_weak(c, prog, whole):
    strong = await c.working_set(); await run(c, prog, strong, PROMPT, 2)
    weak, refused = [], None
    for i in range(24):
        h = await c.working_set()
        try:
            await run(c, prog, h, PROMPT, 2); await h.set_props({"weak": True, "keep": i}); weak.append(h)
        except Exception as e: refused = str(e); break
        await asyncio.sleep(0.05)
    check("weak handles make room: no launch was refused", refused is None, str(refused))
    dead = []
    for h in weak:
        try: await h.stat()
        except HandleError as e: dead.append(e)
    check("a policy's discard is heard, by name", dead and all(e.code == "dead" and "keep-handles" in str(e) for e in dead), f"{len(dead)}/{len(weak)} {dead[:1]}")
    check("the rest stands", (await strong.stat())["state"] == "parked")
    y = await c.working_set(); z = await run(c, prog, y, PROMPT, 32)
    check("and runs on", z == whole)
    await strong.close(); await y.close()

async def t_snapshot(c, prog, whole):
    w = await c.working_set(); first = await run(c, prog, w, PROMPT, 12)
    saved = await w.save("s/one"); check("a save says what it kept", saved["tokens"] == (await w.stat())["tokens"], str(saved))
    second = await run(c, prog, w, [first[-1]], 8, drawn=12)
    await w.save("s/two")
    two = await c.load_snapshot("s/two"); st = await two.stat()
    check("a load comes off the device", st["state"] == "parked" and not st["resident"], str(st))
    rest = await run(c, prog, two, [second[-1]], 12, drawn=20)
    check("a loaded set resumes exactly", first + second + rest == whole)
    one = await c.load_snapshot("s/one"); r = await run(c, prog, one, [first[-1]], 20, drawn=12)
    check("and so does an older one", first + r == whole)
    try: await w.save("s/one"); check("a name taken is refused", False)
    except HandleError as e: check("a name taken is refused", e.code == "in_use", str(e))
    listed = await c.list_snapshots("s/")
    check("they list", [s["name"] for s in listed] == ["s/one", "s/two"], str(listed))
    await c.delete_snapshot("s/one"); await c.delete_snapshot("s/two")
    check("delete gives the name back", await c.list_snapshots("s/") == [])
    r2 = await run(c, prog, two, [1], 2, drawn=60); check("what was loaded stays", len(r2) == 2)
    for bad, code in (("s/none", "unknown"), ("x" * 101, "invalid")):
        try: await c.load_snapshot(bad); check("a bad load is refused", False)
        except HandleError as e: check("a bad load is refused", e.code == code, str(e))
    for h in (w, two, one): await h.close()

async def t_long_snapshot(c, prog, whole):
    # past a sliding window: what a ring retired stays out, what it holds comes back
    long = [1000 + (i * 37) % 3000 for i in range(args.ctx - 80 if args.ctx else 700)]
    # gpt-oss draws otherwise on the first prompt this long a server takes (on
    # main too); the second is what every later one draws
    cold = await c.working_set(); await run(c, prog, cold, long, 1); await cold.close()
    a = await c.working_set(); one = await run(c, prog, a, long, 48)
    b = await c.working_set(); first = await run(c, prog, b, long, 24)
    await b.save("l/one"); await b.close()
    back = await c.load_snapshot("l/one")
    rest = await run(c, prog, back, [first[-1]], 24, drawn=24)
    check("a long set saved and loaded resumes exactly", first + rest == one, f"{len(first + rest)}")
    await c.delete_snapshot("l/one")
    for h in (a, back): await h.close()

async def t_script(c, prog, whole):
    S = Path(__file__).parent
    for file in ("turn.py", "turn.js"):
        name = await c.install_program(S / file)
        a = await c.working_set(); b = await c.working_set()
        one = (await (await c.run(name, ws=a, tokens=PROMPT, max_tokens=16)).result())["tokens"]
        first = (await (await c.run(name, ws=b, tokens=PROMPT, max_tokens=6)).result())["tokens"]
        rest = (await (await c.run(name, ws=b, tokens=[first[-1]], max_tokens=10)).result())["tokens"]
        check(f"{file} resumes a handle by typed input", first + rest == one, str(one[:5]))
        ck = await a.fork(); st = await ck.stat()
        check(f"{file} gave it back", st["state"] == "parked" and st["tokens"] >= 75, str(st))
        for h in (a, b, ck): await h.close()

async def t_matcher(c, prog, whole):
    S = Path(__file__).parent
    name = await c.install_program(S / "m.py")
    m = await c.matcher(regex="[0-9]+")
    check("a matcher is unborn until bound", (await m.stat())["state"] == "unborn")
    r = await (await c.run(name, m=m, accept=[])).result()
    st = await m.stat()
    check("a matcher comes back", st["state"] == "parked" and r["mask"] > 0, f"{r} {st}")
    bad = await c.matcher(json_schema="{ not json")
    try:
        await (await c.run(name, m=bad, accept=[])).result(); check("a bad grammar fails its launch", False)
    except Exception as e: check("a bad grammar fails its launch", True, str(e)[:90])
    try: await bad.stat(); check("and the handle says why", False)
    except HandleError as e: check("and the handle says why", e.code == "dead", str(e)[:90])
    await m.close()
    # the builtin prefill fills a handle
    w = await c.working_set()
    p = await c.launch_process("prefill", input={"tokens": PROMPT}, handles={"ws": w}, returns={"ws": w})
    out = await p.result()
    st = await w.stat()
    check("prefill leaves a handle holding the prompt", st["state"] == "parked" and st["tokens"] >= 59, f"{out} {st}")
    await w.close()

async def t_media(c, prog, whole):
    S = Path(__file__).parent
    mk = await c.install_program(S / "mk.py")
    see = await c.install_program(S / "see.py")
    out, sound = await c.frames(), c.pcm()
    await (await c.run(mk, out=out, sound=sound, w=64, h=48)).result()
    st = await out.stat()
    check("a process's pictures land as a handle", st["state"] == "parked" and st["width"] == 64, str(st))
    check("and its sound", (await sound.stat())["samples"] == 400)
    png = await out.read("png")
    check("a client reads a handle as the format it asks", png[:8] == b"\x89PNG\r\n\x1a\n", f"{len(png)} bytes")
    wav = await sound.read("wav")
    check("sound reads too", wav[:4] == b"RIFF", f"{len(wav)} bytes")
    twin = await out.fork()
    r = await (await c.run(see, image=twin)).result()
    check("pictures move on to a model with no client between", r["tokens"] > 0, str(r))
    check("and come back", (await twin.read("png")) == png)
    up = await c.frames(png)
    check("an upload is unborn until bound", (await up.stat())["state"] == "unborn")
    u = await (await c.run(see, image=up)).result()
    check("an upload reads as what was uploaded", u == r, str(u))
    try:
        await out.read("tiff"); check("a format that is none is refused", False)
    except HandleError as e: check("a format that is none is refused", e.code == "invalid", str(e))
    bad = await c.frames(b"not a picture")
    try:
        await (await c.run(see, image=bad)).result(); check("bytes that decode to nothing fail their launch", False)
    except Exception as e: check("bytes that decode to nothing fail their launch", True, str(e)[:80])
    for h in (out, sound, twin, up): await h.close()

async def t_clients(c, prog, whole):
    import json, tempfile
    S = Path(__file__).parent
    mk = await c.install_program(S / "mk.py")
    see = await c.install_program(S / "see.py")
    out, sound = await c.frames(), c.pcm()
    await (await c.run(mk, out=out, sound=sound, w=64, h=48)).result()
    data = await out.read("png")
    want = await (await c.run(see, image=out)).result()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(data)
    spec = {"uri": URI[0], "prog": prog, "see": see, "png": f.name, "prompt": PROMPT}
    runs = {
        "js": ["node", str(S / "client.mjs"), json.dumps(spec)],
        "rust": [str(ROOT / "target/release/examples/handles"), URI[0], prog, see, f.name, json.dumps(PROMPT)],
    }
    for name, argv in runs.items():
        with tempfile.TemporaryFile() as err:
            done = await asyncio.create_subprocess_exec(*argv, stdout=subprocess.PIPE, stderr=err)
            try:
                stdout, _ = await asyncio.wait_for(done.communicate(), 60)
            except asyncio.TimeoutError:
                done.kill(); await done.wait(); err.seek(0)
                check(f"the {name} client", False, "hung after: " + err.read().decode()[-200:]); continue
            err.seek(0); stderr = err.read()
        if done.returncode != 0:
            check(f"the {name} client", False, stderr.decode()[-300:]); continue
        got = json.loads(stdout.decode().strip().splitlines()[-1])
        check(f"the {name} client resumes and forks", got["tokens"] == whole, str(got["stat"]))
        check(f"the {name} client hears a closed handle", "unknown" in got["closed"], got["closed"])
        check(f"the {name} client uploads and reads", got["seen"] == want and got["raw"] == 64 * 48 * 3, f"{got['seen']} {got['raw']}")
    await out.close(); await sound.close()

async def t_failed(c, prog, whole):
    S = Path(__file__).parent
    both = await c.install_program(S / "both.py")
    fail = await c.install_program(S / "fail.py")
    async def survives(name, w, first):
        check(f"{name}: the handle is as it was", (await w.stat())["state"] == "parked")
        check(f"{name}: and resumes exactly", first + await run(c, prog, w, [first[-1]], 20, drawn=12) == whole)
        await w.close()
    w = await c.working_set(); first = await run(c, prog, w, PROMPT, 12)
    try: await (await c.run(both, ws=w, matcher=await c.matcher(regex="("))).result(); check("a bad grammar fails its launch", False)
    except Exception as e: check("a bad grammar fails its launch", True, str(e)[:70])
    await survives("a bad grammar", w, first)
    w = await c.working_set(); first = await run(c, prog, w, PROMPT, 12)
    out = await (await c.run(fail, ws=w, tokens=[7] * 600)).result()
    check("a guest hears a refused fire", out["failed"], str(out)[:90])
    await survives("a refused fire", w, first)
    w = await c.working_set(); first = await run(c, prog, w, PROMPT, 12)
    p = await c.launch_process("prefill", input="not what it reads", handles={"ws": w}, returns={"ws": w})
    try: await p.result(); check("a guest that fails before it takes a handle fails", False)
    except Exception as e: check("a guest that fails before it takes a handle fails", True, str(e)[:70])
    await survives("a guest that never took it", w, first)

async def t_close(c, prog, whole):
    d = PieClient(URI[0]); await d.connect()
    made = 0
    for i in range(24):
        h = await d.working_set()
        try:
            await run(d, prog, h, PROMPT, 2); made += 1
        except Exception:
            await h.close(); break
    await d.close()
    await asyncio.sleep(1)
    e = PieClient(URI[0]); await e.connect()
    again = 0
    for i in range(made):
        h = await e.working_set()
        await run(e, prog, h, PROMPT, 2); again += 1
    await e.close()
    check("closing a session frees what it held", again == made and made > 0, f"{made} then {again}")

URI = [None]
async def main():
    with socket.socket() as s: s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    cfg = Path(args.log).with_suffix(".toml")
    t = (HOME / "config.toml").read_text()
    t = re.sub(r'^source = "[^"]*"', f'source = "{args.model}"', t, flags=re.M)
    t = re.sub(r'^listen = "[^"]*"', f'listen = "127.0.0.1:{port}"', t, flags=re.M)
    if args.kv_budget: t = t.replace('# device_kv_budget     = "20GiB"', f'device_kv_budget = "{args.kv_budget}"')
    if args.host_budget: t = t.replace('# host_kv_budget       = "32GiB"', f'host_kv_budget = "{args.host_budget}"')
    if args.policy: t = t.replace('# worker_policy            = "noop"', f'worker_policy = "{args.policy}"')
    if args.ctx: t = t.replace("# max_model_len   = 32768", f"max_model_len = {args.ctx}\n#")
    cfg.write_text(t)
    log = open(args.log, "w")
    srv = subprocess.Popen([str(PIE), "--config", str(cfg), "serve"], stdout=log, stderr=subprocess.STDOUT,
                           env=dict(os.environ, PIE_HOME=str(HOME), RUST_BACKTRACE="1"), start_new_session=True)
    try:
        c = None
        for _ in range(300):
            if srv.poll() is not None: raise SystemExit(f"server exited; see {args.log}")
            try:
                URI[0] = f"ws://127.0.0.1:{port}"; c = PieClient(URI[0]); await c.connect(); break
            except Exception: c = None; await asyncio.sleep(1)
        prog = await c.install_program(EX / "working_set_handles.wasm")
        whole = await t_resume(c, prog)
        for f in (t_lazy, t_order, t_dead, t_evict, t_quota, t_script, t_matcher, t_media, t_clients, t_failed, t_snapshot, t_long_snapshot, t_close, t_keep, t_weak):
            if args.only and f.__name__ not in args.only.split(","): continue
            if not args.only and f in (t_keep, t_weak): continue
            try: await asyncio.wait_for(f(c, prog, whole), 120)
            except Exception as e:
                check(f.__name__, False, repr(e)[:300]); traceback.print_exc()
        await c.close()
    finally:
        if srv.poll() is None:
            os.killpg(srv.pid, signal.SIGTERM)
            try: srv.wait(20)
            except subprocess.TimeoutExpired: os.killpg(srv.pid, signal.SIGKILL)
    print(f"{sum(results)}/{len(results)} passed")
asyncio.run(main())
