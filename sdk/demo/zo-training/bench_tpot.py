"""Boot a local Pie server, init an ES adapter, and measure single-decode
TPOT/TTFT with the adapter ON vs OFF.

Output:  per-step latency, prefill (TTFT), and steady-state TPOT for both.
"""
import asyncio
import json
import os
import statistics
from contextlib import AsyncExitStack

import fire

from pie.config import AuthConfig, Config, ModelConfig
from pie.server import Server
from pie_client import Event


HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(HERE, "inferlets", "target", "wasm32-wasip2", "release")
MANIFEST_DIR = os.path.join(HERE, "inferlets")
ARTIFACTS = {
    "es-init@0.1.5":    (os.path.join(ARTIFACT_DIR, "es_init.wasm"),    os.path.join(MANIFEST_DIR, "es-init", "Pie.toml")),
    "bench-tpot@0.1.0": (os.path.join(ARTIFACT_DIR, "bench_tpot.wasm"), os.path.join(MANIFEST_DIR, "bench-tpot", "Pie.toml")),
}


async def launch_and_collect(client, inferlet, input_dict):
    process = await client.launch_process(inferlet, input=input_dict)
    result = None
    while True:
        event, message = await process.recv()
        if event in (Event.Return, Event.Message):
            result = message
            if event == Event.Return:
                break
        elif event == Event.Error:
            raise RuntimeError(f"inferlet {inferlet} failed: {message}")
    return result


async def _bench(model: str, max_tokens: int, prompt: str, adapter_name: str,
                rank: int, alpha: float, sigma: float, repeats: int):
    cfg = Config(
        port=0,
        auth=AuthConfig(enabled=False),
        max_concurrent_processes=4,
        models=[ModelConfig(
            hf_repo=model,
            device=["cuda:0"],
            gpu_mem_utilization=0.8,
            cpu_mem_budget_in_gb=12,
            default_token_budget=4096,
            max_batch_size=8,
        )],
    )

    async with AsyncExitStack() as stack:
        server = await stack.enter_async_context(Server(cfg))
        client = await server.connect()

        for name, (wasm, mf) in ARTIFACTS.items():
            await client.install_program(wasm, mf, force_overwrite=True)
        print(f"[bench] installed {list(ARTIFACTS.keys())}")

        # Init adapter (once); benchmark uses it for the ON case.
        await launch_and_collect(client, "es-init@0.1.5", {
            "name": adapter_name,
            "rank": rank,
            "alpha": alpha,
            "population_size": 16,
            "mu_fraction": 0.5,
            "initial_sigma": sigma,
            "upload": "",
        })
        print(f"[bench] adapter '{adapter_name}' initialized (rank={rank}, sigma={sigma})")

        async def run(label, with_adapter):
            inp = {
                "prompt": prompt,
                "system_prompt": "You are a helpful assistant.",
                "max_tokens": max_tokens,
                "warmup_tokens": 4,
                "adapter_name": adapter_name if with_adapter else "",
                "zo_seed": 12345 if with_adapter else 0,
            }
            samples = []
            for r in range(repeats):
                raw = await launch_and_collect(client, "bench-tpot@0.1.0", inp)
                samples.append(json.loads(raw))
                s = samples[-1]
                print(f"  [{label}] run {r+1}: tokens={s['tokens_total']}  "
                      f"TTFT={s['ttft_us']/1000:.2f} ms  "
                      f"TPOT_mean={s['tpot_mean_us']/1000:.2f} ms  "
                      f"P50={s['tpot_p50_us']/1000:.2f} ms  P90={s['tpot_p90_us']/1000:.2f} ms  "
                      f"min={s['tpot_min_us']/1000:.2f}  max={s['tpot_max_us']/1000:.2f}  "
                      f"steps={s['decode_steps']}")
            return samples

        print("\n=== Adapter OFF ===")
        off = await run("OFF", with_adapter=False)
        print("\n=== Adapter ON ===")
        on = await run("ON ", with_adapter=True)

        def summary(samples, label):
            mean_tpot = statistics.mean(s["tpot_mean_us"] for s in samples)
            p50 = statistics.median(s["tpot_p50_us"] for s in samples)
            p90 = statistics.median(s["tpot_p90_us"] for s in samples)
            ttft = statistics.median(s["ttft_us"] for s in samples) / 1000
            return mean_tpot, p50, p90, ttft

        on_tpot,  on_p50,  on_p90,  on_ttft  = summary(on,  "ON")
        off_tpot, off_p50, off_p90, off_ttft = summary(off, "OFF")
        print()
        print("=== Summary (median across runs) ===")
        print(f"           TTFT (ms)     TPOT mean (μs)   TPOT P50 (μs)   TPOT P90 (μs)")
        print(f"  ON    :  {on_ttft:8.2f}   {on_tpot:>14.1f}   {on_p50:>13.0f}   {on_p90:>13.0f}")
        print(f"  OFF   :  {off_ttft:8.2f}   {off_tpot:>14.1f}   {off_p50:>13.0f}   {off_p90:>13.0f}")
        print(f"  Δ     :  {on_ttft-off_ttft:+8.2f}   {on_tpot-off_tpot:>+14.1f}   "
              f"{on_p50-off_p50:>+13.0f}   {on_p90-off_p90:>+13.0f}")


def main(
    model: str = "meta-llama/Llama-3.2-1B-Instruct",
    max_tokens: int = 128,
    prompt: str = "Write a 200-word essay about the importance of curiosity in scientific research.",
    adapter_name: str = "tpot-bench",
    rank: int = 8,
    alpha: float = 16.0,
    sigma: float = 0.005,
    repeats: int = 3,
):
    asyncio.run(_bench(model, max_tokens, prompt, adapter_name,
                       rank, alpha, sigma, repeats))


if __name__ == "__main__":
    fire.Fire(main)
