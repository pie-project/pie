"""Throughput of a vLLM OpenAI-compatible server, in the shape of `bench_twins.py`.

n concurrent `/v1/completions` with one prompt, greedy, `ignore_eos` so every
request generates exactly `max_tokens` (what the pie twins do). Run beside
`bench_twins.py` on the same model, one server at a time on one GPU; pass
`--no-enable-prefix-caching` to vLLM for a prefill comparison, since the pie
twins share no prefix.

    vllm serve google/gemma-4-E4B --dtype bfloat16 --max-num-seqs 64 --port 8000
    python tests/examples/bench_vllm.py --n 1,8,16,32,64 --max-tokens 128
"""
import argparse, asyncio, json, statistics, sys, time
import aiohttp

async def run_one(session, url, model, prompt, max_tokens, timeout):
    t0 = time.perf_counter()
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
            "ignore_eos": True, "stream": False}
    async with session.post(f"{url}/v1/completions", json=body, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
        j = await r.json()
    if "usage" not in j:
        raise RuntimeError(json.dumps(j)[:300])
    return j["usage"]["completion_tokens"], time.perf_counter() - t0, j["choices"][0]["text"]

async def bench(session, url, model, prompt, n, max_tokens, timeout):
    t0 = time.perf_counter()
    res = await asyncio.gather(*[run_one(session, url, model, prompt, max_tokens, timeout) for _ in range(n)], return_exceptions=True)
    wall = time.perf_counter() - t0
    ok = [r for r in res if not isinstance(r, BaseException)]
    errs = [r for r in res if isinstance(r, BaseException)]
    toks = sum(c for c, _, _ in ok)
    lats = sorted(l for _, l, _ in ok)
    p50 = lats[len(lats)//2] if lats else 0; p99 = lats[int(len(lats)*0.99)] if lats else 0
    per = statistics.mean(c/l for c, l, _ in ok) if ok else 0
    print(f"vllm/{model.split('/')[-1]:18s} n={n:3d}  ok={len(ok):3d}/{n:<3d} wall={wall:7.2f}s  tok/s={toks/wall:8.1f}  lat p50={p50:6.2f}s p99={p99:6.2f}s  per-req tok/s={per:6.1f}")
    for e in errs[:2]: print("   error:", str(e)[:300])
    return ok[0][2] if ok else ""

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="google/gemma-4-E4B")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--n", default="1,8,16,32,64")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=600)
    a = ap.parse_args()
    async with aiohttp.ClientSession() as s:
        text = await run_one(s, a.url, a.model, a.prompt, a.max_tokens, a.timeout)  # warm-up
        print("sample:", repr(text[2][:80]))
        for n in [int(x) for x in a.n.split(",")]:
            for _ in range(a.repeat):
                await bench(s, a.url, a.model, a.prompt, n, a.max_tokens, a.timeout)
asyncio.run(main())
