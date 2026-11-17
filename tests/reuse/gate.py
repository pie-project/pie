#!/usr/bin/env python3
"""Program reuse on a served model: requests that differ only in per-request
values (knobs, prompt length, max_tokens) register no program after the
first.

    PIE_HOME=/tmp/pie-reuse python3 tests/reuse/gate.py --engine cuda --log /tmp/reuse.log

`PIE_HOME` is a home of its own (`pie config init`, then `pie model import
Qwen/Qwen3-0.6B`). Build first: `pie` (`PIE_BIN`, default
target/release/pie) and, for wasm32-wasip3 in examples/,
`text-completion-bench`, `beam-search`, `naive-masked` and
`snapkv-eviction` (`--wasm`, default examples/target/wasm32-wasip3/release).

The runtime registers a program by the hash of its whole container; the
model status's `programs.misses` counts the programs registered anew (the
runtime's debug log names each one's channel shapes and op count). Each
phase sends its requests in turn and reads how many programs each one
registered: the first may register its kind's programs, every later one
must register none. A phase whose requests are distinct program kinds by
design lists them as such, and a phase the runtime cannot serve with one
program yet is an expected failure naming why: it is reported, and a pass
of one says it can become a plain phase.
"""
import argparse, asyncio, json, os, signal, socket, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages/python/client/src"))
from pie_client import PieClient

a = argparse.ArgumentParser()
a.add_argument("--engine", default="cuda", choices=("cuda", "vulkan", "wgpu", "metal", "rocm"))
a.add_argument("--model", default="Qwen/Qwen3-0.6B")
a.add_argument("--log", required=True)
a.add_argument("--wasm", default=str(ROOT / "examples/target/wasm32-wasip3/release"))
a.add_argument("--only", default="", help="comma-separated phase names")
args = a.parse_args()
PIE = Path(os.environ.get("PIE_BIN", ROOT / "target/release/pie"))
HOME = Path(os.environ["PIE_HOME"])

# Not yet one program, by design or until a later step: phase -> why.
TEMPERATURE = ("a sampled request's temperature is a program constant: a channel-fed divisor "
               "costs the fused sampler ~20% of a decode step until rank-0 channels lower as "
               "constants")
STEP4 = "a request's length shapes a key extent (step 4: dynamic key lengths)"
PROMPT = "a prompt's length is its prefill's token extent until prompts pad to the max pass rows"
EXPECTED = {
    "chat-prompt_len": PROMPT,
    "chat-greedy-prompt_len": PROMPT,
    "tcb-prompt_len": PROMPT,
    "chat-temperature": TEMPERATURE,
    "tcb-temperature": TEMPERATURE,
    "beam-max_tokens": STEP4 + ": the key pool is prompt + decodes",
    "beam-prompt_len": STEP4 + ": the key pool and its positions",
    "masked-max_tokens": STEP4 + ": the mask spans the prompt and max_tokens",
    "masked-prompt_len": STEP4 + ": the mask spans the prompt and max_tokens",
    "snapkv-filler": STEP4 + ": the page-mass rows are the prompt's pages",
}
# Distinct program kinds by design: each request may register its own.
KINDS = {"beam-width": "the beam width is the pass's lane count"}

M = "Write one sentence about the sea."
SEA = "The sea is"


def phases():
    chat = lambda **kw: ("chat", {"messages": [{"role": "user", "content": M}], "max_tokens": 8,
                                  "top_p": 0.9, "temperature": 0.7, **kw})
    tcb = lambda **kw: ("tcb", {"prompt": SEA, "max_tokens": 8, "temperature": 0.7, "top_p": 0.9,
                                "ignore_eos": True, **kw})
    beam = lambda **kw: ("beam", {"prompt": SEA, "max_tokens": 8, "beams": 2, **kw})
    masked = lambda **kw: ("masked", {"prompt": SEA, "max_tokens": 8, "temperature": 0.7, **kw})
    snapkv = lambda **kw: ("snapkv", {"filler": 20, **kw})
    very = lambda k: SEA + " very" * k
    return {
        "chat-temperature": [chat(temperature=t) for t in (0.6, 0.7, 0.8, 0.9)],
        "chat-top_p": [chat(top_p=p) for p in (0.5, 0.7, 0.8, 0.95)],
        "chat-max_tokens": [chat(max_tokens=n) for n in (4, 8, 12, 16)],
        "chat-prompt_len": [chat(messages=[{"role": "user", "content": M + " Please." * k}])
                            for k in (1, 5, 9, 13)],
        "chat-greedy-prompt_len": [chat(temperature=0, max_tokens=6,
                                        messages=[{"role": "user", "content": "Name a color." + " Thanks." * k}])
                                   for k in (1, 5, 9, 13)],
        "chat-logprobs": [chat(temperature=0, logprobs=True, top_logprobs=k) for k in (1, 2, 3)],
        "tcb-temperature": [tcb(temperature=t) for t in (0.6, 0.7, 0.8, 0.9)],
        "tcb-top_p": [tcb(top_p=p) for p in (0.5, 0.7, 0.8, 0.95)],
        "tcb-prompt_len": [tcb(prompt=very(k)) for k in (1, 5, 9, 13)],
        "tcb-max_tokens": [tcb(max_tokens=n) for n in (4, 8, 12, 16)],
        "beam-max_tokens": [beam(max_tokens=n) for n in (6, 8, 10, 12)],
        "beam-width": [beam(beams=b) for b in (2, 3, 4)],
        "beam-prompt_len": [beam(prompt=very(k)) for k in (1, 5, 9)],
        "masked-max_tokens": [masked(max_tokens=n) for n in (6, 8, 10)],
        "masked-temperature": [masked(temperature=t) for t in (0.6, 0.7, 0.8)],
        "masked-prompt_len": [masked(prompt=very(k)) for k in (1, 5, 9)],
        "snapkv-filler": [snapkv(filler=f) for f in (20, 40, 60)],
        "snapkv-same": [snapkv() for _ in range(3)],
    }


def http(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps({"model": "default", **body}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as resp:
        return resp.status


async def misses(c):
    ok, st = await asyncio.wait_for(c.query("model_status", ""), 60)
    st = json.loads(st) if ok else {}
    found = [v for k, v in st.items() if k.endswith(".programs.misses")]
    if not found:
        raise SystemExit("the model status counts no `programs.misses`: a runtime without the metric")
    return found[0]


async def main():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    cfg = HOME / "reuse.toml"
    cfg.write_text(f'[server]\nlisten = "127.0.0.1:{port}"\nverbose = false\n\n[model]\nname = "default"\n'
                   f'source = "{args.model}"\n\n[engine]\ntype = "{args.engine}"\ndevice = ["{args.engine}:0"]\n'
                   '\n[runtime]\n')
    log = open(args.log, "w")
    srv = subprocess.Popen([str(PIE), "--config", str(cfg), "serve"], stdout=log, stderr=subprocess.STDOUT,
                           env=dict(os.environ, PIE_HOME=str(HOME)), start_new_session=True)
    failed = []
    try:
        t0 = time.monotonic()
        while "Server ready" not in Path(args.log).read_text(errors="ignore"):
            if srv.poll() is not None or time.monotonic() - t0 > 1200:
                raise SystemExit("the server did not come up: see " + args.log)
            await asyncio.sleep(0.1)
        c = PieClient(f"ws://127.0.0.1:{port}")
        await asyncio.wait_for(c.connect(), 120)
        wasm = Path(args.wasm)
        programs = {name: await asyncio.wait_for(c.install_program(wasm / f"{file}.wasm"), 300)
                    for name, file in (("tcb", "text_completion_bench"), ("beam", "beam_search"),
                                       ("masked", "naive_masked"), ("snapkv", "snapkv_eviction"))}
        loop = asyncio.get_running_loop()
        only = set(filter(None, args.only.split(",")))
        for phase, requests in phases().items():
            if only and phase not in only:
                continue
            deltas, error = [], None
            for target, body in requests:
                before = await misses(c)
                try:
                    if target == "chat":
                        await loop.run_in_executor(None, http, port, body)
                    else:
                        p = await asyncio.wait_for(c.launch_process(programs[target], input=body), 900)
                        await asyncio.wait_for(p.result(), 900)
                except Exception as e:
                    error = f"{type(e).__name__}: {e}"[:300]
                deltas.append(await misses(c) - before)
            reused = error is None and all(d == 0 for d in deltas[1:])
            verdict = ("kinds" if phase in KINDS else
                       ("xpass" if reused else "xfail") if phase in EXPECTED else
                       "pass" if reused else "FAIL")
            if verdict in ("FAIL", "xpass"):
                failed.append(phase)
            print(json.dumps({"phase": phase, "verdict": verdict, "misses": deltas,
                              "why": EXPECTED.get(phase) or KINDS.get(phase), "error": error}), flush=True)
        await asyncio.wait_for(c.close(), 30)
    finally:
        if srv.poll() is None:
            os.killpg(srv.pid, signal.SIGTERM)
            try:
                srv.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(srv.pid, signal.SIGKILL)
                srv.wait()
    print(json.dumps({"failed": failed}))
    sys.exit(1 if failed else 0)


asyncio.run(main())
