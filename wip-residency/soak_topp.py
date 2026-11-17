"""Soak: one `pie serve` of BINARY on HOME for MINUTES of mixed load, staggered
arrivals: top-p text-completion-bench (64 tokens), text_completion, beam w2,
naive_masked (a mask) and snapkv_eviction (a keep / score-capture program).
Every WINDOW seconds prints a JSON line: tokens/s of the bench in the window,
the server's phys_footprint (MiB), the engine tallies (model_status), and the
window's failures. Ends with a summary line.

    soak.py BINARY HOME EXAMPLES WASMDIR TAG MINUTES [WINDOW]
"""
import asyncio, json, os, random, re, subprocess, sys, threading, time

from pie_client.client import PieClient

S = os.environ.get("OUT", ".")
binary, home, examples, wasmdir, tag = sys.argv[1:6]
MINUTES = float(sys.argv[6])
WINDOW = float(sys.argv[7]) if len(sys.argv) > 7 else 300.0
WORDS = ("river town stone archive lantern harbor merchant ledger bridge "
         "orchard bell tower winter market clock garden mill road").split()


def prompt(seed, words):
    rng = (seed * 7919 + 17) % 100003
    out = []
    for _ in range(words):
        rng = (rng * 1103515245 + 12345) % 2147483648
        out.append(WORDS[rng % len(WORDS)])
    return f"Seed {seed}: " + " ".join(out)


def footprint(pid):
    try:
        out = subprocess.run(["footprint", "-p", str(pid)], capture_output=True, text=True,
                             timeout=30).stdout
        m = re.search(r"phys_footprint:\s*([\d.]+)\s*([KMG])B", out)
        if m:
            v = float(m.group(1))
            return round(v * {"K": 1 / 1024, "M": 1, "G": 1024}[m.group(2)], 1)
    except Exception:
        pass
    return None


async def main():
    env = dict(os.environ, PIE_HOME=home, PIE_INFERLETS=examples, PIE_CONTENTION_TRACE_MS="5000")
    log = open(f"{S}/soak-{tag}.log", "w")
    server = subprocess.Popen([binary, "--log-level", "info", "serve"], cwd=examples, env=env,
                              stdout=subprocess.PIPE, stderr=log, text=True)
    try:
        addr = None
        while addr is None:
            line = server.stdout.readline()
            if not line:
                raise SystemExit(f"{tag}: the server exited before it was ready")
            m = re.search(r"Server ready at (ws://\S+)", line)
            if m:
                addr = m.group(1)
        planner = {}
        def drain():
            for l in server.stdout:
                if l.startswith("[planner-trace]"):
                    for k in ("parks", "serves", "evictions", "restores", "restore_failures", "free"):
                        m = re.search(rf"\b{k}=(\d+)", l)
                        if m:
                            planner[k] = int(m.group(1))
        threading.Thread(target=drain, daemon=True).start()
        client = PieClient(f"{addr}/v1/ws")
        await client.connect()
        await client.authenticate("soak")
        prog = {}
        for name in ["text_completion_bench", "text_completion", "beam_search", "naive_masked",
                     "snapkv_eviction"]:
            path = f"{wasmdir}/{name}.wasm"
            prog[name] = await client.install_program_bytes(open(path, "rb").read(),
                                                            os.path.basename(path))

        stats = {"tokens": 0, "done": 0, "failed": 0, "errors": []}
        by_kind = {}

        async def one(kind, name, body):
            t0 = time.monotonic()
            try:
                process = await client.launch_process(prog[name], body)
                res = await process.result()
                if kind == "topp":
                    if isinstance(res, str):
                        res = json.loads([l for l in res.splitlines() if l.startswith("{")][-1])
                    stats["tokens"] += res.get("num_output_tokens", 0)
                stats["done"] += 1
                by_kind.setdefault(kind, [0, 0])[0] += 1
            except Exception as e:
                stats["failed"] += 1
                by_kind.setdefault(kind, [0, 0])[1] += 1
                if len(stats["errors"]) < 20:
                    stats["errors"].append(f"{kind}: {repr(e)[:300]}")
            return time.monotonic() - t0

        rnd = random.Random(7)
        seed = [0]

        async def arrivals(kind, name, body_of, mean_gap, cap):
            live = set()
            end = start + MINUTES * 60
            while time.monotonic() < end:
                await asyncio.sleep(rnd.expovariate(1.0 / mean_gap))
                live = {t for t in live if not t.done()}
                if len(live) >= cap:
                    continue
                seed[0] += 1
                live.add(asyncio.create_task(one(kind, name, body_of(seed[0]))))
            await asyncio.gather(*live, return_exceptions=True)

        async def model_status():
            ok, text = await client.query("model_status", "")
            try:
                d = json.loads(text)
                return {k.split(".engine.")[-1]: v for k, v in d.items()
                        if ".engine." in k and not k.endswith("_by_engine")}
            except Exception:
                return text[:200]

        start = time.monotonic()
        fp0 = footprint(server.pid)
        st0 = await model_status()
        print(json.dumps({"tag": tag, "t_min": 0, "footprint_mib": fp0, "tallies": st0}), flush=True)

        async def monitor():
            last_tokens, last_t = 0, time.monotonic()
            last_failed = 0
            while time.monotonic() < start + MINUTES * 60:
                await asyncio.sleep(WINDOW)
                now = time.monotonic()
                toks = stats["tokens"]
                print(json.dumps({
                    "tag": tag, "t_min": round((now - start) / 60, 1),
                    "topp_tok_s": round((toks - last_tokens) / (now - last_t), 1),
                    "done": stats["done"], "failed_window": stats["failed"] - last_failed,
                    "footprint_mib": footprint(server.pid), "tallies": await model_status(),
                    "by_kind": by_kind, "planner": dict(planner)}), flush=True)
                last_tokens, last_t, last_failed = toks, now, stats["failed"]

        await asyncio.gather(
            monitor(),
            arrivals("topp", "text_completion_bench",
                     lambda s: {"prompt": prompt(s, 60), "max_tokens": 64, "ignore_eos": True,
                                "return_text": False}, 0.15, int(os.environ.get("SOAK_CAP", "48"))),
        )
        print(json.dumps({"tag": tag, "final": True, "minutes": MINUTES,
                          "footprint_start_mib": fp0, "footprint_end_mib": footprint(server.pid),
                          "tallies_start": st0, "tallies_end": await model_status(),
                          "done": stats["done"], "failed": stats["failed"], "by_kind": by_kind,
                          "errors": stats["errors"]}), flush=True)
        await client.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()


asyncio.run(main())
