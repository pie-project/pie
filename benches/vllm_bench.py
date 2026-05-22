#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time

from common import (
    RequestResult,
    add_mode_subcommands,
    finish,
    hf_chat_prompts_and_counts,
    make_prompts,
    summarize,
)


def visible_tp_uses_system_topology(tp_size: int) -> bool:
    if tp_size <= 1:
        return False
    try:
        import torch
        import pynvml

        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            return False
        pynvml.nvmlInit()
        n = min(tp_size, torch.cuda.device_count())
        handles = []
        for i in range(n):
            prop = torch.cuda.get_device_properties(i)
            bus_id = (
                f"{prop.pci_domain_id:08x}:"
                f"{prop.pci_bus_id:02x}:"
                f"{prop.pci_device_id:02x}.0"
            )
            handles.append(pynvml.nvmlDeviceGetHandleByPciBusId(bus_id.encode()))
        for i in range(n):
            for j in range(i + 1, n):
                level = pynvml.nvmlDeviceGetTopologyCommonAncestor(
                    handles[i], handles[j],
                )
                if level >= pynvml.NVML_TOPOLOGY_SYSTEM:
                    return True
    except Exception:
        return False
    finally:
        try:
            pynvml.nvmlShutdown()  # type: ignore[name-defined]
        except Exception:
            pass
    return False


def maybe_disable_nccl_p2p_for_system_topology(tp_size: int) -> bool:
    system_topology = visible_tp_uses_system_topology(tp_size)
    if system_topology and os.environ.get("NCCL_P2P_DISABLE") is None:
        os.environ["NCCL_P2P_DISABLE"] = "1"
    return system_topology


def run(args: argparse.Namespace):
    system_topology = maybe_disable_nccl_p2p_for_system_topology(args.tp_size)

    from vllm import LLM, SamplingParams

    n = args.requests if args.mode == "latency" else args.num_requests
    prompts, prompt_counts = hf_chat_prompts_and_counts(
        args.model, args.system, make_prompts(args, n + args.warmup)
    )
    # Concurrency 0 means "no batch cap" — match pie's --concurrency 0 path.
    if args.mode == "latency":
        max_num_seqs = 1
    elif args.concurrency == 0:
        max_num_seqs = max(1, args.num_requests)
    else:
        max_num_seqs = args.concurrency
    llm_kwargs = {}
    if args.attention_backend:
        llm_kwargs["attention_config"] = {"backend": args.attention_backend}
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    if system_topology:
        llm_kwargs["disable_custom_all_reduce"] = True

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem_util,
        max_num_seqs=max_num_seqs,
        tensor_parallel_size=args.tp_size,
        max_model_len=args.max_model_len,
        enable_prefix_caching=False,
        **llm_kwargs,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
    )
    if args.warmup:
        llm.generate(prompts[: args.warmup], sampling)

    run_prompts = prompts[args.warmup:]
    run_prompt_counts = prompt_counts[args.warmup:]
    results: list[RequestResult] = []
    start = time.perf_counter()
    if args.mode == "latency":
        for p, prompt_count in zip(run_prompts, run_prompt_counts):
            req_start = time.perf_counter()
            outputs = llm.generate([p], sampling)
            req_wall = time.perf_counter() - req_start
            for out in outputs:
                results.append(
                    RequestResult(
                        True, float(req_wall), len(out.outputs[0].token_ids), prompt_count
                    )
                )
    else:
        outputs = llm.generate(run_prompts, sampling)
        for out, prompt_count in zip(outputs, run_prompt_counts):
            results.append(
                RequestResult(True, 0.0, len(out.outputs[0].token_ids), prompt_count)
            )
    wall = time.perf_counter() - start

    summary = summarize(
        mode=args.mode,
        engine="vllm",
        model=args.model,
        results=results,
        wall_s=wall,
        config={
            "enable_prefix_caching": False,
            "max_num_seqs": max_num_seqs,
            "attention_backend": args.attention_backend,
            "enforce_eager": args.enforce_eager,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "ignore_eos": args.ignore_eos,
            "unique_prompts": args.unique_prompts,
        },
    )
    return summary, results


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM canonical latency/throughput benchmark")
    add_mode_subcommands(parser)
    for sp in parser._subparsers._group_actions[0].choices.values():
        sp.add_argument("--attention-backend", default=None)
        sp.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()
    summary, results = run(args)
    finish(summary, results, args.json_out)


if __name__ == "__main__":
    main()
