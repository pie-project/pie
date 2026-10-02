#!/usr/bin/env python3
"""vLLM multimodal (image) latency/throughput benchmark.

Sibling of `vllm_bench.py` for image-input VLMs (the only modality with a true
Pie<->vLLM parity model: Qwen3-VL). Each request is the *same* local image plus
a text question, run with `ignore_eos` to a fixed `max_tokens` so token counts
match across engines. The timed path is: image preprocess (resize/patchify) +
vision encode + text prefill + decode — exactly what `pie_mm_bench.py` times on
the Pie side.

Modes (from common.add_mode_subcommands):
  latency  one request at a time; reports per-request wall latency.
  tput     all requests submitted together; reports req/s and output tok/s.

Run:
  /root/.venv/vllm/bin/python vllm_mm_bench.py latency \
      --model Qwen/Qwen3-VL-2B-Instruct --image assets/bench_image.png \
      --requests 16 --max-tokens 128 --json-out out/vllm_latency.json
"""
from __future__ import annotations

import argparse
import time

from common import (
    RequestResult,
    add_mode_subcommands,
    cuda_profiler_start,
    cuda_profiler_stop,
    finish,
    maybe_set_cpu_affinity,
    summarize,
    visible_cuda_devices,
)


def _load_image(path: str):
    from PIL import Image

    img = Image.open(path)
    img.load()
    return img.convert("RGB")


def _render_prompt(processor, system: str, question: str) -> str:
    # Mirror the Pie image-qa-bench scaffolding (system + "Here is an image:" +
    # <image> + question) so prompt-token counts line up across engines.
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here is an image:"},
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
    ]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _questions(args: argparse.Namespace, n: int) -> list[str]:
    if args.unique_prompts:
        return [f"{args.question} (Request #{i})" for i in range(n)]
    return [args.question for _ in range(n)]


def run(args: argparse.Namespace):
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    cpu_affinity = maybe_set_cpu_affinity(args, visible_cuda_devices(args.tp_size))
    n = args.requests if args.mode == "latency" else args.num_requests

    image = _load_image(args.image)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    questions = _questions(args, n + args.warmup)
    # One prompt dict per request: same image, per-request question text. The
    # image object is shared (read-only) across requests.
    requests = [
        {
            "prompt": _render_prompt(processor, args.system, q),
            "multi_modal_data": {"image": image},
        }
        for q in questions
    ]

    if args.mode == "latency":
        max_num_seqs = 1
    elif args.concurrency == 0:
        max_num_seqs = max(1, args.num_requests)
    else:
        max_num_seqs = args.concurrency

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_mem_util,
        max_num_seqs=max_num_seqs,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tp_size,
        limit_mm_per_prompt={"image": 1},
        enable_prefix_caching=False,
        disable_log_stats=True,
        enforce_eager=args.enforce_eager,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
    )
    if args.warmup:
        warmup_sampling = sampling
        if args.warmup_max_tokens is not None:
            warmup_sampling = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.warmup_max_tokens,
                ignore_eos=args.ignore_eos,
            )
        llm.generate(requests[: args.warmup], warmup_sampling)

    run_requests = requests[args.warmup :]
    results: list[RequestResult] = []
    cuda_profiler_start(args.cuda_profiler_capture)
    start = time.perf_counter()
    try:
        if args.mode == "latency":
            for req in run_requests:
                req_start = time.perf_counter()
                outputs = llm.generate([req], sampling)
                req_wall = time.perf_counter() - req_start
                for out in outputs:
                    results.append(
                        RequestResult(
                            True,
                            float(req_wall),
                            len(out.outputs[0].token_ids),
                            len(out.prompt_token_ids),
                        )
                    )
        else:
            outputs = llm.generate(run_requests, sampling)
            for out in outputs:
                results.append(
                    RequestResult(
                        True,
                        0.0,
                        len(out.outputs[0].token_ids),
                        len(out.prompt_token_ids),
                    )
                )
    finally:
        wall = time.perf_counter() - start
        cuda_profiler_stop(args.cuda_profiler_capture)

    summary = summarize(
        mode=args.mode,
        engine="vllm",
        model=args.model,
        results=results,
        wall_s=wall,
        config={
            "modality": "image",
            "image": args.image,
            "max_num_seqs": max_num_seqs,
            "max_model_len": args.max_model_len,
            "enforce_eager": args.enforce_eager,
            "enable_prefix_caching": False,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "ignore_eos": args.ignore_eos,
            "unique_prompts": args.unique_prompts,
            "cpu affinity": cpu_affinity,
            "warmup max tokens": args.warmup_max_tokens,
        },
    )
    return summary, results


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM multimodal (image) benchmark")
    add_mode_subcommands(parser)
    for sp in parser._subparsers._group_actions[0].choices.values():
        sp.add_argument(
            "--image",
            default="assets/bench_image.png",
            help="Local image path fed to every request.",
        )
        sp.add_argument(
            "--question",
            default="What is in this image? Answer in one sentence.",
            help="Question text (overrides --prompt for multimodal).",
        )
        sp.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()
    # Multimodal default model (overridable via --model).
    if args.model == "Qwen/Qwen3-0.6B":
        args.model = "Qwen/Qwen3-VL-2B-Instruct"
    summary, results = run(args)
    finish(summary, results, args.json_out)


if __name__ == "__main__":
    main()
