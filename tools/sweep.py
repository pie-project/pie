#!/usr/bin/env python3
"""Fire-top sweep: prefill tok/s of long prompts at each stated fire top."""
import argparse, asyncio, glob, json, os, random, re, signal, statistics, subprocess, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--pie", required=True); ap.add_argument("--wasm", required=True)
ap.add_argument("--home", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--tops", default="2048,4096,8192,16384"); ap.add_argument("--conc", default="1")
ap.add_argument("--reps", type=int, default=6); ap.add_argument("--prompt", type=int, default=32768)
ap.add_argument("--context", type=int, default=40960); ap.add_argument("--devices", default="cuda:0")
ap.add_argument("--tp", type=int, default=1); ap.add_argument("--port", type=int, default=54199)
ap.add_argument("--client", default=None, help="the pie checkout's packages/python/client/src")
ap.add_argument("--device-id", default=None)
a = ap.parse_args()
if a.client: sys.path.insert(0, a.client)
from pie_client import PieClient

home = Path(a.home).resolve(); home.mkdir(parents=True, exist_ok=True)
work = home / "sweep"; work.mkdir(exist_ok=True)
devices = ", ".join(f'"{d}"' for d in a.devices.split(","))
cfg = work / "cfg.toml"
cfg.write_text(f'[server]\nlisten = "127.0.0.1:{a.port}"\nverbose = false\n\n[model]\nname = "default"\nsource = "{a.model}"\n\n'
               f'[engine]\ntype = "cuda"\ndevice = [{devices}]\ntensor_parallel_size = {a.tp}\nmax_context = {a.context}\n\n[runtime]\n')

def profiles():
    return glob.glob(str(home / "profiles" / "cuda" / "user" / "*.toml"))

def state_top(top):
    """The user profile states `fire_rows = top` (0: removed, the rule's)."""
    found = profiles()
    if top and not found and a.device_id:
        user = home / "profiles" / "cuda" / "user"; user.mkdir(parents=True, exist_ok=True)
        (user / f"{a.device_id}.toml").write_text("api = 1\n\n[engine]\nfire_rows = 0\n")
        found = profiles()
    if top and not found:
        sys.exit(f"no user profile under {home}/profiles/cuda/user to state fire_rows = {top} in: "
                 "the rates boot stored none, so a stated top would not be read; stopping rather than "
                 "sweeping the rule's top under the stated tops' names")
    for p in found:
        s = open(p).read()
        s = re.sub(r"(?m)^fire_rows = \d+\n", "", s)
        if top:
            if re.search(r"(?m)^\[engine\]\n", s):
                s = re.sub(r"(?m)^\[engine\]\n", f"[engine]\nfire_rows = {top}\n", s, count=1)
            else:
                s = s.replace("[provenance]", f"[engine]\nfire_rows = {top}\n\n[provenance]", 1) if "[provenance]" in s else s + f"\n[engine]\nfire_rows = {top}\n"
        open(p, "w").write(s)

def tokens(seed):
    x = seed * 7919 + 1; out = []
    for _ in range(a.prompt):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF; out.append(1000 + (x >> 8) % 90000)
    return out

async def one(c, prog, seed):
    p = await c.launch_process(prog, input={"prompt_tokens": tokens(seed), "max_tokens": 1, "ignore_eos": True,
                                            "return_text": False, "temperature": 0.0})
    r = await asyncio.wait_for(p.result(), 600); r = json.loads(r) if isinstance(r, str) else r
    return r["num_prompt_tokens"]

def serve(top, tag, reps):
    env = dict(os.environ, PIE_HOME=str(home), RUST_LOG="info", CUDA_CACHE_PATH=str(work / f"cc-{tag}"))
    log = work / f"{tag}.log"; t0 = time.time()
    srv = subprocess.Popen([a.pie, "--config", str(cfg), "serve"], stdout=open(log, "w"), stderr=subprocess.STDOUT,
                           env=env, start_new_session=True)
    out = {"model": a.model, "top": top, "tag": tag}
    async def run():
        while "Server ready" not in log.read_text(errors="ignore"):
            if srv.poll() is not None: out["refused"] = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text(errors="ignore"))[-600:]; return
            if time.time() > t0 + 900: out["refused"] = "not up in 900 s"; return
            await asyncio.sleep(0.05)
        out["boot_s"] = round(time.time() - t0, 2)
        c = PieClient(f"ws://127.0.0.1:{a.port}"); await c.connect(); prog = await c.install_program(a.wasm)
        seed = random.randrange(1 << 30)
        if reps:
            await one(c, prog, seed); await asyncio.gather(*[one(c, prog, seed + 1 + k) for k in range(2)])
            for n in [int(x) for x in a.conc.split(",")]:
                v = []
                for r in range(reps):
                    t = time.time(); got = await asyncio.gather(*[one(c, prog, seed + 100 * (r + 1) + k) for k in range(n)])
                    v.append(round(sum(got) / (time.time() - t)))
                out[f"c{n}"] = v; out[f"c{n}_mean"] = round(statistics.mean(v)); out[f"c{n}_range_pct"] = round(100 * (max(v) - min(v)) / statistics.mean(v), 2)
        await c.close()
    try: asyncio.run(run())
    finally:
        try: os.killpg(srv.pid, signal.SIGTERM); srv.wait(timeout=60)
        except Exception:
            try: os.killpg(srv.pid, signal.SIGKILL)
            except Exception: pass
    text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text(errors="ignore"))
    for key, pat in [("laid_top", r"fire limits: \d+ lanes, (\d+) rows"), ("device_pages", r"device_pages=(\d+)"),
                     ("bodies", r"bodies armed (\d+ of \d+)"), ("top_line", r"(top rung \d+[^;]*)")]:
        m = re.search(pat, text)
        if m: out[key] = m.group(1)
    print(json.dumps(out), flush=True)
    return out

state_top(0); serve(0, "rates", 0)          # measures the rates once, unless stored
tops = [int(t) for t in a.tops.split(",")]
for top in tops + tops[:1]:
    state_top(top); assert top == 0 or all(f"fire_rows = {top}" in open(p).read() for p in profiles()); serve(top, f"top{top}-{int(time.time())}", a.reps)
state_top(0)
