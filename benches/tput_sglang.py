"""Standalone SGLang throughput benchmark.

Mirrors `benches/tput.py`'s workload so pie native / dummy can be compared
fairly against sglang: applies the model's chat template (system + user
messages, same content as pie's text-completion inferlet) and uses
``temperature=0.6 + top_p=0.95`` to match the inferlet's sampler.

Run from the pie repo root with sglang installed in a separate venv::

    PATH=/path/to/sglang/.venv/bin:$PATH \\
        /path/to/sglang/.venv/bin/python benches/tput_sglang.py \\
        --n 2048 --max-tokens 100 --warmup-n 32
"""
import argparse
import time


def main():
    p = argparse.ArgumentParser(description="Standalone SGLang throughput benchmark")
    p.add_argument("--n", type=int, default=2048,
                   help="Total number of requests to time")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--mem-frac", type=float, default=0.8,
                   help="Fraction of GPU memory reserved for KV cache (sglang's mem_fraction_static)")
    p.add_argument("--max-tokens", type=int, default=100,
                   help="Max generated tokens per request")
    p.add_argument("--unique-prompts", action="store_true",
                   help="Append a per-request id to the prompt to defeat KV-cache reuse")
    p.add_argument("--warmup-n", type=int, default=0,
                   help="Run a warmup batch of this size before timing")
    p.add_argument("--cuda-graphs", action="store_true",
                   help="Enable CUDA graphs (sglang's default is enabled; this flag is for explicit-on)")
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--context-length", type=int, default=1024)
    p.add_argument("--disable-radix-cache", action="store_true",
                   help="Disable RadixAttention KV-prefix reuse (fair compare vs engines without prefix dedup)")
    p.add_argument("--ignore-eos", action="store_true",
                   help="Generate exactly max_tokens regardless of stop tokens")
    p.add_argument("--max-running-requests", type=int, default=None,
                   help="Cap concurrent in-flight requests (matches pie's max_concurrent_processes)")
    args = p.parse_args()

    import sglang as sgl

    engine = sgl.Engine(
        model_path=args.model,
        mem_fraction_static=args.mem_frac,
        disable_cuda_graph=not args.cuda_graphs,
        max_running_requests=args.max_running_requests,
        tp_size=args.tp_size,
        context_length=args.context_length,
        disable_radix_cache=args.disable_radix_cache,
    )

    base = "Write a short story about a robot."
    system = "You are a helpful benchmarking assistant."
    # Apply the model's chat template so the prompt token count matches
    # pie's inferlet (which calls ctx.system / ctx.user / ctx.cue).
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    def fmt(user_text: str) -> str:
        return tok.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": user_text}],
            tokenize=False, add_generation_prompt=True,
        )

    if args.unique_prompts:
        prompts = [fmt(f"{base} (Request #{i})") for i in range(args.n)]
    else:
        prompts = [fmt(base)] * args.n

    # Match pie's inferlet sampler: temperature=0.6 + top_p=0.95.
    sampling = {"temperature": 0.6, "top_p": 0.95,
                "max_new_tokens": args.max_tokens,
                "ignore_eos": args.ignore_eos}

    if args.warmup_n > 0:
        print(f"Warmup ({args.warmup_n} reqs)...", flush=True)
        engine.generate([fmt(base)] * args.warmup_n, sampling)
        print("Warmup done.", flush=True)

    start = time.time()
    outputs = engine.generate(prompts, sampling)
    duration = time.time() - start

    total_tokens = 0
    for o in outputs:
        meta = o.get("meta_info", {})
        total_tokens += int(meta.get("completion_tokens", 0))
    completed = len(outputs)

    print()
    print("─" * 40)
    print(f"sglang standalone: model={args.model} N={args.n} max_tokens={args.max_tokens}")
    print(f"Total Time:        {duration:.2f} s")
    print(f"Completed:         {completed}/{args.n}")
    print(f"Total Out Tokens:  {total_tokens}")
    print(f"Requests/sec:      {completed / duration:.2f}")
    print(f"Tokens/sec:        {total_tokens / duration:.2f}")
    print("─" * 40)

    engine.shutdown()


if __name__ == "__main__":
    main()
