#!/usr/bin/env python3
"""give-run.py PIE CFG GUEST LOG CLIENT_SRC: one server, one guest that runs
the SDK's run_ahead to its end with device-counted folds and gives its set;
then the client stats, forks, saves and reopens what was given. No retries:
the server is ready at its own line, and every call is made once."""
import asyncio, json, os, signal, subprocess, sys, threading

pie, cfg, guest, log, client_src = sys.argv[1:6]
sys.path.insert(0, client_src)
from pie_client import PieClient

PROMPT, MAX = "The capital of France is", 32
READY = "worker registered with gateway"

srv = subprocess.Popen([pie, "-c", cfg, "--log-level", "info", "serve"], stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, errors="replace", start_new_session=True,
                       env=dict(os.environ, RUST_BACKTRACE="1"))
logf = open(log, "w")
ready = False
for line in srv.stdout:
    logf.write(line); logf.flush()
    if READY in line:
        ready = True
        break

def drain():
    for line in srv.stdout:
        logf.write(line); logf.flush()

async def main():
    c = PieClient("ws://127.0.0.1:18722")
    await c.connect()
    prog = await c.install_program(guest)
    ws = await c.working_set()
    p = await c.launch_process(prog, input={"ws": {"$handle": "ws"}, "prompt": PROMPT, "max_tokens": MAX},
                               handles={"ws": ws}, returns={"ws": ws})
    out = await p.result()
    if isinstance(out, str):
        out = json.loads(out)
    print("GUEST " + json.dumps(out), flush=True)
    stat = await ws.stat(); print("STAT " + json.dumps(stat), flush=True)
    child = await ws.fork()
    forked = await child.stat(); print("FORK " + json.dumps(forked), flush=True)
    saved = await ws.save("ordinary/one"); print("SAVE " + json.dumps(saved), flush=True)
    listed = await c.list_snapshots("ordinary/"); print("LIST " + json.dumps(listed), flush=True)
    back = await c.load_snapshot("ordinary/one")
    reopened = await back.stat(); print("REOPEN " + json.dumps(reopened), flush=True)
    await c.delete_snapshot("ordinary/one")
    expect = out["prompt_tokens"] + len(out["tokens"]) - 1
    got = {"expect": expect, "bound": out["bound"], "stat": stat.get("tokens"), "fork": forked.get("tokens"),
           "save": saved.get("tokens"), "list": [s.get("tokens") for s in listed], "reopen": reopened.get("tokens")}
    lengths = [got["stat"], got["fork"], got["save"], got["reopen"]] + got["list"]
    print("LENGTHS " + json.dumps(got), flush=True)
    print("VERDICT " + ("exact" if all(n == expect for n in lengths) else "MISMATCH"), flush=True)
    await c.close()

code = 1
try:
    if not ready:
        print(f"SERVER ended before it was ready: {srv.wait()}", flush=True)
    else:
        threading.Thread(target=drain, daemon=True).start()
        asyncio.run(asyncio.wait_for(main(), 120))
        code = 0
except BaseException as e:
    print(f"DRIVER {type(e).__name__}: {e}", flush=True)
finally:
    if srv.poll() is None:
        os.killpg(srv.pid, signal.SIGTERM)
        try:
            srv.wait(20)
        except subprocess.TimeoutExpired:
            os.killpg(srv.pid, signal.SIGKILL)
sys.exit(code)
