#!/usr/bin/env python3
"""Guest programs beside the kv a pool maps in: a sequence that once let kv
pages land on guest state.

    PIE_HOME=/tmp/pie-test python3 tests/handles/guest_mix.py --log /tmp/g.log

Build first: `pie` for the engine and `text-completion-bench` for
wasm32-wasip3 in examples/. One server, the sequence a bench row runs, `--loops`
times: a warm-up (one lane, then 64 lanes of one prompt in one process), three
prompts of ~511 tokens, a single lane's 32 then 288 tokens, one process of 64
lanes for 32 then 160, and 64 concurrent processes for 32 then 160. The rung
falls to its smallest between them and rises for the long prompts, the 64
processes bind their guest programs one after another, and the kv supply maps
the lanes' shared prefixes in. Every request must complete with its tokens and
no epilogue may fault: a status the kernel did not write (two bf16 halves where
a status word should be) is guest state that kv or activation overwrote.

Where an engine laid its guests' batch tables under the activation's carve,
past what the span reserved, mapped-in kv pages overwrote them and every
process of the fire failed with "the kernel wrote a status this engine cannot
read". Guest state is the instance's and the pair's, in pool slots the runtime
takes; nothing guest lies in the span.
"""
import argparse, asyncio, os, re, signal, socket, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages/python/client/src"))
from pie_client import PieClient
PIE = Path(os.environ.get("PIE_BIN", ROOT / "target/release/pie"))
HOME = Path(os.environ["PIE_HOME"])  # a home of its own: `pie config init`, then `pie model import`
EX = ROOT / "examples/target/wasm32-wasip3/release"
a = argparse.ArgumentParser()
a.add_argument("--model", default="Qwen/Qwen3-0.6B")
a.add_argument("--log", required=True)
a.add_argument("--loops", type=int, default=3)
args = a.parse_args()
results = []
def check(name, cond, detail=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name, detail, flush=True)

WORDS = ("river town stone archive lantern harbor merchant ledger bridge "
         "orchard bell tower winter market clock garden mill road").split()

def prompt(seed, words):
    rng = (seed * 7919 + 17) % 100003
    out = []
    for _ in range(words):
        rng = (rng * 1103515245 + 12345) % 2147483648
        out.append(WORDS[rng % len(WORDS)])
    return f"Seed {seed}: " + " ".join(out)

def lines(pattern):
    plain = [re.sub(r"\x1b\[[0-9;]*m", "", l) for l in Path(args.log).read_text(errors="replace").splitlines()]
    return [l for l in plain if re.search(pattern, l)]

async def main():
    with socket.socket() as s: s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    cfg = Path(args.log).with_suffix(".toml")
    t = (HOME / "config.toml").read_text()
    t = re.sub(r'^source = "[^"]*"', f'source = "{args.model}"', t, flags=re.M)
    t = re.sub(r'^listen = "[^"]*"', f'listen = "127.0.0.1:{port}"', t, flags=re.M)
    cfg.write_text(t)
    log = open(args.log, "w")
    env = dict(os.environ, PIE_HOME=str(HOME), RUST_LOG=os.environ.get("RUST_LOG", "info"))
    srv = subprocess.Popen([str(PIE), "--config", str(cfg), "serve"], stdout=log, stderr=subprocess.STDOUT,
                           env=env, start_new_session=True)
    try:
        c = None
        for _ in range(300):
            if srv.poll() is not None: raise SystemExit(f"server exited; see {args.log}")
            try: c = PieClient(f"ws://127.0.0.1:{port}"); await c.connect(); break
            except Exception: c = None; await asyncio.sleep(1)
        prog = await c.install_program(EX / "text_completion_bench.wasm")

        async def run(body):
            body = dict({"ignore_eos": True, "return_text": False}, **body)
            p = await c.launch_process(prog, input=body)
            return await p.result()

        def tokens(r):
            if isinstance(r, str):
                import json
                r = json.loads([l for l in r.splitlines() if l.startswith("{")][-1])
            per = r.get("request_output_tokens") or []
            return sum(per) if per else r.get("num_output_tokens", 0)

        seed, failures, short = 0, [], 0
        for loop in range(args.loops):
            await run({"prompt": "Hi", "max_tokens": 4, "temperature": 0.0})
            await run({"prompts": ["Hi"] * 64, "batch_concurrency": 64, "max_tokens": 4})
            for _ in range(3):
                seed += 1
                await run({"prompt": prompt(seed, 380), "max_tokens": 1, "temperature": 0.0})
            for m in (32, 288):
                short += tokens(await run({"prompt": prompt(7, 80), "max_tokens": m, "temperature": 0.0})) != m
            for m in (32, 160):
                got = tokens(await run({"prompts": [prompt(100 + l, 80) for l in range(64)],
                                        "batch_concurrency": 64, "max_tokens": m}))
                short += got != 64 * m
            for m in (32, 160):
                seed += 1
                rs = await asyncio.gather(*(run({"prompt": prompt(1000 * seed + l, 80), "max_tokens": m})
                                            for l in range(64)), return_exceptions=True)
                failures += [r for r in rs if isinstance(r, Exception)]
                short += sum(tokens(r) != m for r in rs if not isinstance(r, Exception))
            print(f"INFO loop {loop}: {len(failures)} failed, {short} short so far", flush=True)
        faulted = lines(r"epilogue faulted|status this engine cannot read")
        check("every request completes", not failures, repr(failures[:1])[:300])
        check("with all its tokens", short == 0, f"{short} short")
        check("no epilogue faults", not faulted, str(faulted[:1])[:300])
        check("the kv supply mapped prefixes in meanwhile", lines(r"kv supply:.*mapped=[1-9]"),
              str(lines(r"kv supply:")[-1:])[:200])
        await c.close()
    finally:
        if srv.poll() is None:
            os.killpg(srv.pid, signal.SIGTERM)
            try: srv.wait(20)
            except subprocess.TimeoutExpired: os.killpg(srv.pid, signal.SIGKILL)
    print(f"{sum(results)}/{len(results)} passed")
asyncio.run(main())
