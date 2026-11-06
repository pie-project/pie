#!/usr/bin/env python3
"""Handles on a real engine: resume, fork, eviction, quota, every SDK.

    PIE_HOME=/tmp/pie-test python3 tests/handles/e2e.py --log /tmp/e.log \\
        --ctx 512 --slots 12 --kv-budget 216MiB

Build first: `pie` for the engine, `working-set-handles` for wasm32-wasip3 in
examples/, and `--example handles` of pie-client for t_clients. A token here is drawn by position, so a turn split across
processes must draw exactly what one process does. `--slots 4` with
`--kv-budget 116MiB --only t_quota,t_close` reaches the host's limit; `t_cost`
wants the default pool.
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
a.add_argument("--ctx", type=int, default=0); a.add_argument("--slots", type=int, default=0)
a.add_argument("--kv-budget", default=""); a.add_argument("--log", required=True)
a.add_argument("--only", default="")
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
    d = await c.working_set()
    x = await run(c, prog, d, PROMPT, 32); check("the run is repeatable", x == whole)
    await d.close()
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
    except HandleError as e: check("closed is unknown", "unknown" in str(e), str(e))
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
    except HandleError as e: check("an ungiven return is dead", "dead" in str(e), str(e))
    st = await never.stat(); check("what was given landed", st["state"] == "parked")
    try:
        await c.launch_process(prog, input={"ws": {"$handle": "ws"}, "tokens": PROMPT, "max_tokens": 1, "seed": 7}, handles={"ws": w}, returns={"ws": w})
        check("a dead handle refuses a launch", False)
    except Exception as e: check("a dead handle refuses a launch", "dead" in str(e), str(e)[:80])
    await never.close(); await w.close()
    # a launch that fails before running gives its handle back
    k = await c.working_set(); x = await run(c, prog, k, PROMPT, 12)
    try:
        await c.launch_process("no-such-program@9.9.9", input={}, handles={"ws": k}, returns={"ws": k}); 
    except Exception: pass
    check("a refused launch keeps the handle", (await k.stat())["state"] == "parked")
    await k.close()

async def t_many(c, prog, whole):
    hs = []
    for _ in range(10):
        h = await c.working_set(); await run(c, prog, h, PROMPT, 2); hs.append(h)
    w = await c.working_set(); x = await run(c, prog, w, PROMPT, 32)
    check("idle handles past the seats leave a fire room", x == whole, f"{len(hs)} parked")
    for h in hs + [w]: await h.close()

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
    check("nor does either lose it", r3 == r)
    for h in hogs + [w, fk2, pair]: await h.close()

async def t_quota(c, prog, whole):
    hs, refused = [], None
    for i in range(24):
        h = await c.working_set()
        try:
            await run(c, prog, h, PROMPT, 2); hs.append(h)
        except Exception as e:
            refused = str(e); await h.close(); break
    check("past the host a launch is refused, not starved", refused and "quota" in refused, f"{len(hs)} rest; {refused and refused[:90]}")
    x = await run(c, prog, hs[0], [999], 2, drawn=50)
    check("what rests still runs", len(x) == 2)
    # a fork shares its state until it writes, and then wants its own
    twin = await hs[1].fork()
    try:
        await run(c, prog, twin, [999], 2, drawn=50); check("a fork that would grow past the host is refused", False)
    except Exception as e:
        check("a fork that would grow past the host is refused", "quota" in str(e), str(e)[:90])
    check("and refused, keeps its handle", (await twin.stat())["state"] == "parked")
    await twin.close()
    check("closing the fork frees its twin", len(await run(c, prog, hs[1], [999], 2, drawn=50)) == 2)
    await hs[0].close()
    y = await c.working_set(); z = await run(c, prog, y, PROMPT, 32)
    check("and closing one makes room", z == whole)
    for h in hs[1:] + [y]: await h.close()

async def t_script(c, prog, whole):
    S = Path(__file__).parent
    for file in ("turn.py", "turn.js"):
        name = await c.install_program(S / file, force_overwrite=True)
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
    name = await c.install_program(S / "m.py", force_overwrite=True)
    m = await c.matcher(regex="[0-9]+")
    check("a matcher is unborn until bound", (await m.stat())["state"] == "unborn")
    r = await (await c.run(name, m=m, accept=[])).result()
    st = await m.stat()
    check("a matcher comes back", st["state"] == "parked" and r["mask"] > 0, f"{r} {st}")
    f = await m.fork()
    r2 = await (await c.run(name, m=f, accept=[])).result()
    check("a matcher forks", r2 == r)
    bad = await c.matcher(json_schema="{ not json")
    try:
        await (await c.run(name, m=bad, accept=[])).result(); check("a bad grammar fails its launch", False)
    except Exception as e: check("a bad grammar fails its launch", True, str(e)[:90])
    try: await bad.stat(); check("and the handle says why", False)
    except HandleError as e: check("and the handle says why", "dead" in str(e), str(e)[:90])
    for h in (m, f): await h.close()
    # the builtin prefill fills a handle
    w = await c.working_set()
    p = await c.launch_process("prefill", input={"tokens": PROMPT}, handles={"ws": w}, returns={"ws": w})
    out = await p.result()
    st = await w.stat()
    check("prefill leaves a handle holding the prompt", st["state"] == "parked" and st["tokens"] >= 59, f"{out} {st}")
    await w.close()

async def t_media(c, prog, whole):
    S = Path(__file__).parent
    mk = await c.install_program(S / "mk.py", force_overwrite=True)
    see = await c.install_program(S / "see.py", force_overwrite=True)
    out, sound = await c.frames(), c.pcm()
    await (await c.run(mk, out=out, sound=sound, w=64, h=48)).result()
    st = await out.stat()
    check("a process's pictures land as a handle", st["state"] == "parked" and st["width"] == 64, str(st))
    check("and its sound", (await sound.stat())["samples"] == 400)
    png = await out.read("png")
    check("a client reads a handle as the format it asks", png[:8] == b"\x89PNG\r\n\x1a\n", f"{len(png)} bytes")
    raw = await out.read("raw-rgb8")
    check("and the handle stays", len(raw) == 64 * 48 * 3, f"{len(raw)} bytes")
    wav = await sound.read("wav")
    check("sound reads too", wav[:4] == b"RIFF", f"{len(wav)} bytes")
    twin = await out.fork()
    r = await (await c.run(see, image=twin)).result()
    check("pictures move on to a model with no client between", r["tokens"] > 0, str(r))
    back = await twin.read("raw-rgb8")
    check("and come back", len(back) == len(raw) and back == raw)
    up = await c.frames(png)
    check("an upload is unborn until bound", (await up.stat())["state"] == "unborn")
    u = await (await c.run(see, image=up)).result()
    check("an upload reads as what was uploaded", u == r and (await up.read("raw-rgb8")) == raw, str(u))
    try:
        await out.read("tiff"); check("a format that is none is refused", False)
    except HandleError as e: check("a format that is none is refused", "tiff" in str(e), str(e))
    # Whichever way the encoder goes, a reader hears how it ended.
    try:
        video = await asyncio.wait_for(out.read("mp4"), 20)
        check("a reader hears how an encoder ended", len(video) > 0, f"{len(video)} bytes")
    except HandleError as e:
        check("a reader hears how an encoder ended", True, str(e)[:80])
    many = await asyncio.gather(*(out.read("png") for _ in range(4)))
    check("reads run side by side", all(b == png for b in many))
    r2 = await (await c.run(see, image=out)).result()
    check("the handle was free the whole time", r2 == r)
    bad = await c.frames(b"not a picture")
    try:
        await (await c.run(see, image=bad)).result(); check("bytes that decode to nothing fail their launch", False)
    except Exception as e: check("bytes that decode to nothing fail their launch", True, str(e)[:80])
    for h in (out, sound, twin, up): await h.close()

async def t_media_js(c, prog, whole):
    S = Path(__file__).parent
    mk = await c.install_program(S / "mk.js", force_overwrite=True)
    see = await c.install_program(S / "see.js", force_overwrite=True)
    out, sound = await c.frames(), c.pcm()
    await (await c.run(mk, out=out, sound=sound, w=64, h=48)).result()
    check("a JS guest gives pictures and sound", (await out.stat())["width"] == 64 and (await sound.stat())["samples"] == 400)
    r = await (await c.run(see, image=out)).result()
    check("and takes them by typed input", r["width"] == 64 and r["tokens"] > 0, str(r))
    await out.close(); await sound.close()

async def t_clients(c, prog, whole):
    import json, tempfile
    S = Path(__file__).parent
    mk = await c.install_program(S / "mk.py", force_overwrite=True)
    see = await c.install_program(S / "see.py", force_overwrite=True)
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

async def t_review(c, prog, whole):
    S = Path(__file__).parent
    both = await c.install_program(S / "both.py", force_overwrite=True)
    # a grammar that does not compile costs only itself
    w = await c.working_set(); first = await run(c, prog, w, PROMPT, 12)
    bad = await c.matcher(regex="(")
    try:
        await (await c.run(both, ws=w, matcher=bad)).result(); check("a bad grammar fails its launch", False)
    except Exception as e: check("a bad grammar fails its launch", True, str(e)[:70])
    check("and the working set beside it survives", (await w.stat())["state"] == "parked")
    r = await run(c, prog, w, [first[-1]], 20, drawn=12)
    check("and resumes exactly", first + r == whole)
    await w.close()
    # an empty prefill is a no-op, not a loss
    w = await c.working_set(); first = await run(c, prog, w, PROMPT, 12)
    p = await c.launch_process("prefill", input={"tokens": []}, handles={"ws": w}, returns={"ws": w})
    await p.result()
    check("an empty prefill gives the handle back", (await w.stat())["state"] == "parked")
    await w.close()
    # a refused read ends its turn: many in a row do not pile up
    o = await c.frames()
    mk = await c.install_program(S / "mk.py", force_overwrite=True)
    sound = c.pcm()
    await (await c.run(mk, out=o, sound=sound, w=8, h=8)).result()
    for _ in range(40):
        try: await o.read("tiff")
        except HandleError: pass
    check("refused reads end", len(await o.read("raw-rgb8")) == 192)
    await o.close(); await sound.close()
    # a launch waits its turn behind a lent handle
    w = await c.working_set(); first = await run(c, prog, w, PROMPT, 12)
    long = await c.launch_process(prog, input={"ws": {"$handle": "ws"}, "tokens": [first[-1]], "max_tokens": 200, "seed": 7, "drawn": 12},
                                  handles={"ws": w}, returns={"ws": w})
    queued = asyncio.ensure_future(c.launch_process(prog, input={"ws": {"$handle": "ws"}, "tokens": [1], "max_tokens": 5, "seed": 7},
                                                    handles={"ws": w}, returns={"ws": w}))
    await long.result()
    p2 = await asyncio.wait_for(queued, 30); await p2.result()
    check("a launch behind a lent handle runs in turn", (await w.stat())["state"] == "parked")
    await w.close()
    # a fire refused at its submit wrote nothing: the set is as it was
    fail = await c.install_program(S / "fail.py", force_overwrite=True)
    w = await c.working_set(); first = await run(c, prog, w, PROMPT, 12)
    before = (await w.stat())["tokens"]
    out = await (await c.run(fail, ws=w, tokens=[7] * 600)).result()
    check("a guest hears a refused fire", out["failed"], str(out)[:90])
    check("and the set it gave is untouched", (await w.stat())["tokens"] == before)
    r = await run(c, prog, w, [first[-1]], 20, drawn=12)
    check("and resumes exactly", first + r == whole)
    await w.close()
    # handles that cannot share the device are refused, and stay as they were
    hs = []
    for i in range(10):
        h = await c.working_set(); hs.append(h)
        await run(c, prog, h, list(range(3000 + i, 3460 + i)), 1, seed=i)
    moved = {f"h{i}": h for i, h in enumerate(hs)}
    try:
        p = await c.launch_process(prog, input={"tokens": [1], "max_tokens": 1, "seed": 7}, handles=moved, returns=moved)
        await asyncio.wait_for(p.result(), 30); check("handles past the device are refused", False)
    except asyncio.TimeoutError: check("handles past the device are refused", False, "it hung")
    except Exception as e: check("handles past the device are refused", "quota" in str(e), str(e)[:90])
    st = [(await h.stat())["state"] for h in hs]
    check("and each is as it was", st == ["parked"] * 10, str(set(st)))
    x = await run(c, prog, hs[0], [1], 2, drawn=50)
    check("and runs", len(x) == 2)
    for h in hs: await h.close()

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

async def t_cost(c, prog, whole):
    import statistics
    w = await c.working_set(); t = await run(c, prog, w, PROMPT, 1)
    lat = []
    for i in range(200):
        t0 = time.perf_counter(); t = await run(c, prog, w, [t[-1]], 1, drawn=i + 1); lat.append(time.perf_counter() - t0)
    n = await c.working_set()
    t0 = time.perf_counter(); await run(c, prog, n, PROMPT, 1); first = time.perf_counter() - t0
    lat.sort(); ms = lambda x: f"{x*1000:.2f}"
    print(f"COST one-token turn over a handle: p50 {ms(lat[100])} p95 {ms(lat[190])} max {ms(lat[-1])} ms; fresh 60-token turn {ms(first)} ms; tokens={t[0] if t else None}", flush=True)
    await w.close(); await n.close()

URI = [None]
async def main():
    with socket.socket() as s: s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    cfg = Path(args.log).with_suffix(".toml")
    t = (HOME / "config.toml").read_text()
    t = re.sub(r'^source = "[^"]*"', f'source = "{args.model}"', t, flags=re.M)
    t = re.sub(r'^listen = "[^"]*"', f'listen = "127.0.0.1:{port}"', t, flags=re.M)
    if args.kv_budget: t = t.replace('# device_kv_budget     = "20GiB"', f'device_kv_budget = "{args.kv_budget}"')
    if args.ctx: t = t.replace("# max_model_len   = 32768", f"max_model_len = {args.ctx}\n#")
    if args.slots: t = t.replace("# max_state_slots = 256", f"max_state_slots = {args.slots}\n#")
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
        prog = await c.install_program(EX / "working_set_handles.wasm", force_overwrite=True)
        whole = await t_resume(c, prog)
        for f in (t_lazy, t_order, t_dead, t_many, t_evict, t_quota, t_script, t_matcher, t_media, t_media_js, t_clients, t_review, t_close, t_cost):
            if args.only and f.__name__ not in args.only.split(","): continue
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
