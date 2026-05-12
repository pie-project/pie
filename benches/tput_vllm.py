"""Standalone vLLM throughput benchmark.

Mirrors `benches/tput.py`'s workload so pie native / dummy can be compared
fairly against vllm: applies the model's chat template (system + user
messages, same content as pie's text-completion inferlet) and uses
``temperature=0.6 + top_p=0.95`` to match the inferlet's sampler.

Run from the pie repo root with vllm installed in a separate venv::

    PATH=/path/to/vllm/.venv/bin:$PATH \\
        /path/to/vllm/.venv/bin/python benches/tput_vllm.py \\
        --n 2048 --max-tokens 100 --warmup-n 32
"""
import argparse
import time

from vllm import LLM, SamplingParams


def main():
    p = argparse.ArgumentParser(description="Standalone vLLM throughput benchmark")
    p.add_argument("--n", type=int, default=2048,
                   help="Total number of requests to time")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--gpu-mem-util", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=100,
                   help="Max generated tokens per request")
    p.add_argument("--max-num-seqs", type=int, default=512)
    p.add_argument("--max-model-len", type=int, default=None,
                   help="vLLM max_model_len. Pin to match pie's config when comparing engines "
                        "(default None = vllm auto from model config).")
    p.add_argument("--unique-prompts", action="store_true",
                   help="Append a per-request id to the prompt to defeat KV-cache reuse")
    p.add_argument("--warmup-n", type=int, default=0,
                   help="Run a warmup batch of this size before timing")
    p.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True,
                   help="Ignore stop tokens so every request consumes the full --max-tokens budget. "
                        "On by default to keep output-token totals identical across engines.")
    p.add_argument("--enforce-eager", action="store_true",
                   help="Run vLLM in eager mode (CUDA graphs disabled). Default off — "
                        "vLLM uses CUDA graphs, matching its production configuration. "
                        "Old default was True, which masked vLLM's primary optimization "
                        "and inflated pie-vs-vllm headlines (ticket #116 bug #2).")
    args = p.parse_args()

    llm_kwargs = dict(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=args.enforce_eager,
        max_num_seqs=args.max_num_seqs,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kwargs)

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
    sampling = SamplingParams(
        temperature=0.6, top_p=0.95, max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
    )

    if args.warmup_n > 0:
        print(f"Warmup ({args.warmup_n} reqs)...", flush=True)
        llm.generate([fmt(base)] * args.warmup_n, sampling)
        print("Warmup done.", flush=True)

    start = time.time()
    outputs = llm.generate(prompts, sampling)
    duration = time.time() - start

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    completed = len(outputs)
    print()
    print("─" * 40)
    print(f"vllm standalone: model={args.model} N={args.n} max_tokens={args.max_tokens}")
    print(f"Total Time:        {duration:.2f} s")
    print(f"Completed:         {completed}/{args.n}")
    print(f"Total Out Tokens:  {total_tokens}")
    print(f"Requests/sec:      {completed / duration:.2f}")
    print(f"Tokens/sec:        {total_tokens / duration:.2f}")
    print("─" * 40)


if __name__ == "__main__":
    main()
