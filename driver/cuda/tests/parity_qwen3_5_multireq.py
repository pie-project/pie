"""Multi-request parity for Qwen3.5 / Qwen3.6 linear-attn batching.

Drives the full runtime → cuda_native driver path with the linear-attn
schedule and asserts the multi-request layer does not crash, leak state,
or produce nonsense.

Procedure (one server / one model load):

  1. Submit a batch of distinct prompts SEQUENTIALLY (await each before
     the next) — drives the driver into pure R=1 fires for each request.
     This is the reference path (already covered by parity_qwen3.py vs
     HF transformers at cosine 0.999).

  2. Submit the SAME prompts CONCURRENTLY (asyncio.gather + greedy batch
     policy) — drives the driver into multi-request fires (R>1 prefill
     and/or R>1 decode), exercising the slot allocator, per-slot
     state-cache reset, and the batched linear-attn kernels.

  3. Pass criteria:
     a. No driver exceptions / no FP_NONE_FOR_DECODE retries.
     b. Every concurrent output is non-empty and within 4× the length of
        its sequential counterpart (sloppy bar — "model didn't go off
        the rails into a repetition loop or hit an early stop").
     c. Reports per-prompt full-match / prefix-match counts as a
        diagnostic — but does NOT fail on byte-level divergence: with
        bf16 + flashinfer's R-dependent prefill tiling, the very first
        decoded token can legitimately flip on prompts where the top-2
        candidates are close in logit space, even with greedy sampling.
        That is a property of the underlying attention kernels, not of
        the multi-request batching layer this test exists to verify.

A real bug in the slot allocator / per-slot reset / batched-kernel
offset arithmetic would manifest as nonsense output (not just an early
argmax flip): incoherent text, wrong language, or repetitive garbage
from reading another request's state.

Usage:

    uv run python driver/cuda/tests/parity_qwen3_5_multireq.py \\
        --model Qwen/Qwen3.5-0.8B-Base
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from pie_client import Event


PROMPTS = [
    "Write a haiku about autumn.",
    "Explain how a neural network works in one paragraph.",
    "List three reasons cats are popular pets.",
    "Describe the color blue to someone who has never seen it.",
    "What is the capital of France, and what is it famous for?",
    "Give me a one-sentence summary of the theory of relativity.",
    "Name five common kitchen utensils.",
    "Write the opening line of a mystery novel set in a library.",
]


# Longer prompts that take measurable prefill time. Used in the
# "stress" pass to encourage the runtime to actually batch — short
# prompts complete before the next gather()-launched task arrives at
# the driver.
STRESS_PROMPTS = [
    p + " " + ("This is a long preamble to make the prefill measurably slow "
               "so the runtime's adaptive scheduler will see a queue depth "
               "greater than one and pack the requests into a single fire.") * 3
    for p in PROMPTS
]


async def collect_one(client, inferlet_name, prompt: str, max_tokens: int) -> str:
    """Run a single inferlet invocation and return the concatenated output.

    `temperature=0` plus `top_p=1.0` selects the kernel's greedy branch
    end-to-end — no seed, no Gumbel noise, no nucleus truncation. Same
    logits → same token, every time. Without `top_p=1.0` the inferlet's
    default 0.95 routes through a seed-dependent sampler and the test
    would falsely flag RNG divergence as a slot-allocation bug."""
    process = await client.launch_process(
        inferlet_name,
        input={
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "system": "You are a helpful assistant.",
        },
    )
    out: list[str] = []
    while True:
        event, msg = await process.recv()
        if event == Event.Stdout:
            out.append(msg)
        elif event == Event.Return:
            out.append(msg)
            break
        elif event == Event.Error:
            raise RuntimeError(f"inferlet error: {msg}")
        # ignore Stderr
    return "".join(out)


async def run_test(args) -> int:
    from pie.server import Server
    from pie.config import (
        Config, ModelConfig, DriverConfig, AuthConfig, RuntimeConfig,
        SchedulerConfig, ServerConfig, TelemetryConfig,
    )

    script_dir = Path(__file__).parent.resolve()
    repo_root = script_dir.parent.parent.parent
    wasm_path = (repo_root / "inferlets" / "text-completion" / "target"
                 / "wasm32-wasip2" / "release" / "text_completion.wasm")
    manifest_path = repo_root / "inferlets" / "text-completion" / "Pie.toml"
    if not wasm_path.exists():
        print(f"text_completion.wasm not built at {wasm_path}")
        print("Run `cargo build --target wasm32-wasip2 --release` in inferlets/text-completion first.")
        return 2

    cfg = Config(
        server=ServerConfig(port=0),
        auth=AuthConfig(enabled=False),
        telemetry=TelemetryConfig(),
        runtime=RuntimeConfig(wasm_max_instances=4096),
        models=[
            ModelConfig(
                name="default",
                hf_repo=args.model,
                scheduler=SchedulerConfig(
                    # `greedy` packs as many ready requests as possible
                    # into each fire — needed to actually exercise R>1
                    # multi-request fires in the concurrent pass.
                    batch_policy="greedy",
                    default_token_limit=512,
                ),
                driver=DriverConfig(
                    type="cuda_native",
                    device=args.device.split(","),
                    tensor_parallel_size=args.tp_size,
                    options={
                        "max_batch_size": args.max_batch_size,
                        "max_num_kv_pages": args.kv_pages,
                    },
                ),
            ),
        ],
    )

    async with Server(cfg) as server:
        client = await server.connect()
        await client.install_program(wasm_path, manifest_path, force_overwrite=True)

        # Find the installed program name (matches tput.py's lookup pattern).
        import tomllib
        manifest = tomllib.loads(manifest_path.read_text())
        pkg_name = manifest["package"]["name"]
        version = manifest["package"]["version"]
        inferlet_name = f"{pkg_name}@{version}"

        n = len(PROMPTS)
        print(f"[parity-multireq] running {n} prompts greedily; max_tokens={args.max_tokens}")

        # Phase 1: sequential.
        print("[parity-multireq] sequential pass...", flush=True)
        seq_outputs: list[str] = []
        for i, p in enumerate(PROMPTS):
            text = await collect_one(client, inferlet_name, p, args.max_tokens)
            seq_outputs.append(text)
            print(f"  [{i}] {len(text)} chars", flush=True)

        # Phase 2: concurrent (one task per prompt, gather).
        print("[parity-multireq] concurrent pass...", flush=True)
        conc_outputs = await asyncio.gather(*[
            collect_one(client, inferlet_name, p, args.max_tokens)
            for p in PROMPTS
        ])
        for i, text in enumerate(conc_outputs):
            print(f"  [{i}] {len(text)} chars", flush=True)

        # Phase 2b: stress pass — long prompts launched together so the
        # runtime actually queues them and the scheduler packs R>1 fires.
        # Outputs aren't compared against anything (the prompts are
        # synthetic); the goal is to force the driver through the R>1
        # code path and assert it doesn't crash or emit garbage.
        print("[parity-multireq] stress pass (R>1 batching)...", flush=True)
        stress_outputs = await asyncio.gather(*[
            collect_one(client, inferlet_name, p, args.max_tokens)
            for p in STRESS_PROMPTS
        ])
        for i, text in enumerate(stress_outputs):
            print(f"  [{i}] {len(text)} chars", flush=True)

        # Phase 2c: eviction pass — submit more prompts than max_slots
        # so the LRU forces slot reassignment partway through. Tests
        # that reset_slot correctly zeros the victim's state before
        # the new context's prefill consumes it. With max_slots=N and
        # 2N+ concurrent contexts, at least N evictions must occur.
        eviction_n = args.max_batch_size * 2 + 8
        print(f"[parity-multireq] eviction pass ({eviction_n} contexts vs "
              f"{args.max_batch_size} slots)...", flush=True)
        # Use distinct, easy-to-evaluate prompts so each output is
        # independently verifiable. Cycle through the base prompt list.
        evict_prompts = [
            f"{PROMPTS[i % len(PROMPTS)]} (req {i})"
            for i in range(eviction_n)
        ]
        evict_outputs = await asyncio.gather(*[
            collect_one(client, inferlet_name, p, args.max_tokens)
            for p in evict_prompts
        ])
        empty_evict = sum(1 for o in evict_outputs if not o.strip())
        short_evict = sum(1 for o in evict_outputs if 0 < len(o) < 30)
        print(f"  total={len(evict_outputs)} empty={empty_evict} "
              f"short(<30ch)={short_evict}", flush=True)

    # Phase 3: assertions + per-prompt diagnostic.
    print()
    print("=== per-prompt diagnostic (informational; not pass/fail) ===")
    full_match = 0
    prefix_match = 0
    PREFIX_CHARS = max(20, args.prefix_chars)
    for i, (s, c) in enumerate(zip(seq_outputs, conc_outputs)):
        if s == c:
            full_match += 1
            print(f"  prompt[{i}]: FULL_MATCH ({len(s)}ch)")
        elif s[:PREFIX_CHARS] == c[:PREFIX_CHARS]:
            prefix_match += 1
            div = next((k for k, (a, b) in enumerate(zip(s, c)) if a != b),
                       min(len(s), len(c)))
            print(f"  prompt[{i}]: PREFIX_MATCH  (diverge at char {div}; "
                  f"seq {len(s)}ch / conc {len(c)}ch)")
        else:
            div = next((k for k, (a, b) in enumerate(zip(s, c)) if a != b),
                       min(len(s), len(c)))
            print(f"  prompt[{i}]: argmax-flipped  (diverge at char {div}; "
                  f"seq {len(s)}ch / conc {len(c)}ch)")
            print(f"    seq : {s[:60]!r}...")
            print(f"    conc: {c[:60]!r}...")

    print()
    print(f"[parity-multireq] full-match: {full_match}/{n}, "
          f"prefix-match (first {PREFIX_CHARS} chars): {prefix_match}/{n}")

    # Hard pass criteria: no crashes (we got this far), no empty outputs,
    # lengths within 4× of each other (model didn't enter a degenerate
    # repetition loop or terminate prematurely on one path only).
    failures: list[str] = []
    for i, (s, c) in enumerate(zip(seq_outputs, conc_outputs)):
        if not s.strip():
            failures.append(f"  prompt[{i}]: empty SEQUENTIAL output")
        if not c.strip():
            failures.append(f"  prompt[{i}]: empty CONCURRENT output")
        if s and c:
            ratio = max(len(s), len(c)) / max(1, min(len(s), len(c)))
            # 4× threshold is intentionally loose. With greedy decode at
            # bf16, a single argmax flip on a stop-adjacent token (e.g.
            # haiku endings, list closures) can switch a short response
            # into a longer one. That's not a multi-req bug — it's
            # legitimate stop-token sensitivity to the kernel pattern.
            if ratio > 4.0:
                failures.append(f"  prompt[{i}]: length ratio {ratio:.2f}x — "
                                "one path may have entered a repetition loop "
                                "or hit an early stop (seq={}ch, conc={}ch)"
                                .format(len(s), len(c)))
    # Stress-pass outputs must all be non-empty and have non-trivial
    # length (>30 chars). That rules out a "concurrent fire produced
    # garbage" failure mode where the model emits an immediate stop.
    for i, c in enumerate(stress_outputs):
        if not c.strip():
            failures.append(f"  stress[{i}]: empty output")
        elif len(c) < 30:
            failures.append(f"  stress[{i}]: suspiciously short output "
                            f"({len(c)}ch): {c!r}")
    # Eviction pass: every context must produce coherent output even
    # after its slot has been recycled. An evicted-and-returning ctx
    # always re-fires as a prefill (runtime invariant), and the slot
    # gets reset_slot'd before the new context consumes it — so a
    # broken reset_slot would manifest as nonsense / very short output.
    for i, c in enumerate(evict_outputs):
        if not c.strip():
            failures.append(f"  evict[{i}]: empty output (slot reset bug?)")
        elif len(c) < 30:
            failures.append(f"  evict[{i}]: suspiciously short output "
                            f"({len(c)}ch): {c!r}")

    if failures:
        print("=== HARD FAILURES ===")
        for f in failures:
            print(f)
        return 1
    print("[parity-multireq] OK — no crashes, no empty outputs, "
          "lengths within tolerance")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base",
                    help="HF model id (Qwen3.5 / 3.6 dense or MoE)")
    ap.add_argument("--device", default="cuda:0",
                    help="Comma-separated device list. With --tp-size N, N "
                         "devices must be listed (e.g. cuda:0,cuda:1).")
    ap.add_argument("--tp-size", type=int, default=1,
                    help="Tensor-parallel rank count. Verifies the slot "
                         "allocator + broadcast wiring under TP>1 by running "
                         "the same sequential-vs-concurrent comparison.")
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--max-batch-size", type=int, default=32,
                    help="Driver max_batch_size + slot allocator capacity. "
                         "32 is enough to fit the sequential-pass-then-"
                         "concurrent-pass slot demand without eviction.")
    ap.add_argument("--prefix-chars", type=int, default=20,
                    help="How many leading output chars must agree between "
                         "sequential and concurrent runs for a prompt to "
                         "count as 'prefix-match'. Beyond this depth, "
                         "flashinfer's R-dependent decode reductions can "
                         "produce bf16 argmax flips that aren't multi-req "
                         "bugs.")
    ap.add_argument("--kv-pages", type=int, default=2048)
    args = ap.parse_args()
    return asyncio.run(run_test(args))


if __name__ == "__main__":
    sys.exit(main())
