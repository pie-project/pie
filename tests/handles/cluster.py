#!/usr/bin/env python3
"""Handles across workers: a controller, a gateway and two real workers on one
device. The worker holding a session's handles is killed, and the session must
hear so: never a run again elsewhere, never an empty handle in the lost one's
place.

    PIE_HOME=/tmp/pie-test python3 tests/handles/cluster.py --log-dir /tmp/c

Build `pie` and `working-set-handles` first; PIE_HOME needs config.toml and the
model imported.
"""
import argparse, asyncio, os, re, signal, socket, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages/python/client/src"))
from pie_client import PieClient
from pie_client.client import HandleError

BIN = ROOT / "target/release"
HOME = Path(os.environ["PIE_HOME"])
EX = ROOT / "examples/target/wasm32-wasip3/release"
PROMPT = list(range(1000, 1060))
a = argparse.ArgumentParser()
a.add_argument("--log-dir", required=True)
a.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
args = a.parse_args()
LOGS = Path(args.log_dir); LOGS.mkdir(parents=True, exist_ok=True)
results = []


def check(name, cond, detail=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name, detail, flush=True)


def port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]


def spawn(name, argv, config=None):
    cmd = [str(argv[0])] + (["--config", str(config)] if config else []) + argv[1:]
    log = open(LOGS / f"{name}.log", "w")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                            env=dict(os.environ, PIE_HOME=str(HOME), RUST_BACKTRACE="1"))


def worker_config(i, controller, gateway, listen):
    t = (HOME / "config.toml").read_text()
    t = re.sub(r'^source = "[^"]*"', f'source = "{args.model}"', t, flags=re.M)
    t = re.sub(r'^listen = "[^"]*"', f'listen = "{listen}"', t, flags=re.M)
    t = t.replace("# max_context     = 32768", "max_context = 4096\n#")
    t += f'\n[cluster]\ncontroller = "{controller}"\nrole = "decode"\ngateways = ["{gateway}"]\n'
    path = LOGS / f"worker{i}.toml"; path.write_text(t); return path


async def run(c, prog, ws, tokens, n, drawn=0):
    p = await c.launch_process(prog, input={"ws": {"$handle": "ws"}, "tokens": tokens, "max_tokens": n, "seed": 7, "drawn": drawn},
                               handles={"ws": ws}, returns={"ws": ws})
    return (await p.result())["tokens"]


def cpu(p):
    """A process group leader's user and system ticks."""
    fields = Path(f"/proc/{p.pid}/stat").read_text().rsplit(")", 1)[1].split()
    return int(fields[11]) + int(fields[12])


async def alive(c, h):
    try:
        await asyncio.wait_for(h.stat(), 10); return True
    except HandleError as e:
        return str(e)
    except Exception as e:
        return f"{type(e).__name__}: {e}"


async def main():
    ctl, gw, gw_w = port(), port(), port()
    procs = {"controller": spawn("controller", [BIN / "pie-controller", "--metrics-listen", f"127.0.0.1:{port()}", "--listen", f"127.0.0.1:{ctl}"])}
    await asyncio.sleep(1)
    procs["gateway"] = spawn("gateway", [BIN / "pie-gateway", "--metrics-listen", f"127.0.0.1:{port()}", "--listen", f"127.0.0.1:{gw}",
                                         "--worker-listen", f"127.0.0.1:{gw_w}", "--controller", f"127.0.0.1:{ctl}"])
    await asyncio.sleep(1)
    workers = {}
    for i in range(2):
        cfg = worker_config(i, f"127.0.0.1:{ctl}", f"127.0.0.1:{gw_w}", f"127.0.0.1:{port()}")
        workers[i] = spawn(f"worker{i}", [BIN / "pie-worker", "--metrics-listen", f"127.0.0.1:{port()}"], cfg)
        await asyncio.sleep(25)  # one at a time: each sizes itself by what the device has free
    try:
        uri = f"ws://127.0.0.1:{gw}"
        c = None
        for _ in range(120):
            for name, p in {**procs, **{f"w{k}": v for k, v in workers.items()}}.items():
                if p.poll() is not None:
                    raise SystemExit(f"{name} exited {p.returncode}")
            try:
                c = PieClient(uri); await c.connect()
                prog = await c.install_program(EX / "working_set_handles.wasm")
                h = await c.working_set()
                base = await asyncio.wait_for(run(c, prog, h, PROMPT, 8), 30)
                break
            except Exception as e:
                last = e; c = None; await asyncio.sleep(2)
        if c is None:
            raise SystemExit(f"no cluster: {last}")
        await asyncio.sleep(6)
        rest = []
        t = base[-1]
        for i in range(30):
            t = (await run(c, prog, h, [t], 1, drawn=8 + i))[0]; rest.append(t)
        check("a session stays where its handles are for 30 turns", len(rest) == 30)

        # a turn running when its worker dies is told, and not run again
        long = asyncio.ensure_future(run(c, prog, h, [t], 1500, drawn=100))
        await asyncio.sleep(0.5)
        spent = {i: cpu(p) for i, p in workers.items()}
        await asyncio.sleep(1)
        spent = {i: cpu(p) - spent[i] for i, p in workers.items()}
        victim = max(spent, key=spent.get)
        check("the long turn is still running", not long.done(), f"cpu ticks {spent}")
        print(f"killing worker {victim}", flush=True)
        other = next(k for k in workers if k != victim)
        before = cpu(workers[other])
        os.killpg(workers[victim].pid, signal.SIGKILL)
        try:
            await asyncio.wait_for(long, 40); check("a lost turn is told", False, "it finished")
        except asyncio.TimeoutError:
            check("a lost turn is told", False, "hung")
        except Exception as e:
            check("a lost turn is told", "worker" in str(e) or "lost" in str(e), str(e)[:110])
        await asyncio.sleep(3)
        ran = cpu(workers[other]) - before
        check("and is not run again on the other worker", ran < spent[victim] * 3,
              f"{ran} ticks there in the seconds after, {spent[victim]} a second on the turn")
        await asyncio.sleep(7)
        # Every handle of a session is on one worker, so its loss is the
        # session's: the socket closes, saying why. Nothing can reach the name
        # again, least of all an empty set under it.
        state = await alive(c, h)
        check("a lost session is told why", isinstance(state, str) and "went away" in state, str(state)[:120])
        try:
            await asyncio.wait_for(run(c, prog, h, [t], 1), 15)
            check("nothing runs on its handles", False, "it ran")
        except Exception as e:
            check("nothing runs on its handles", True, str(e)[:90])
        d = PieClient(uri); await d.connect()
        fresh = await d.working_set()
        x = await asyncio.wait_for(run(d, prog, fresh, PROMPT, 8), 30)
        check("a new session runs on the worker left", x == base)
        # The worker dies with no turn on it: nothing closes the socket until
        # the session next asks, and then it hears.
        await asyncio.sleep(1)
        os.killpg(workers[other].pid, signal.SIGKILL)
        await asyncio.sleep(10)
        state = await alive(d, fresh)
        check("a worker lost between turns is heard at the next", isinstance(state, str) and "dead" in state, str(state)[:120])
        await c.close()
    finally:
        for p in list(workers.values()) + list(procs.values()):
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
    print(f"{sum(results)}/{len(results)} passed")


asyncio.run(main())
