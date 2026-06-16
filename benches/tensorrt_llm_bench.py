#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from common import (
    ROOT,
    RequestResult,
    add_mode_subcommands,
    finish,
    make_prompts,
    summarize,
)


DRIVER_SRC = ROOT / "driver" / "tensorrt_llm" / "src"
if str(DRIVER_SRC) not in sys.path:
    sys.path.insert(0, str(DRIVER_SRC))


def ensure_cuda_library_path() -> None:
    from pie_driver_tensorrt_llm.bootstrap import cuda_library_dirs

    if os.name != "posix":
        return

    env = os.environ.copy()
    venv_bin = Path(sys.prefix) / "bin"
    if venv_bin.is_dir():
        path_parts = [p for p in env.get("PATH", "").split(":") if p]
        if str(venv_bin) not in path_parts:
            env["PATH"] = ":".join([str(venv_bin), *path_parts])
            os.environ["PATH"] = env["PATH"]

    libs = cuda_library_dirs()
    parts = [p for p in env.get("LD_LIBRARY_PATH", "").split(":") if p]
    missing = [p for p in libs if p not in parts]
    if not missing or env.get("PIE_TRTLLM_BENCH_BOOTSTRAPPED") == "1":
        return

    env["PIE_TRTLLM_BENCH_BOOTSTRAPPED"] = "1"
    env["LD_LIBRARY_PATH"] = ":".join([*missing, *parts])
    os.execvpe(sys.executable, [sys.executable, __file__, *sys.argv[1:]], env)


def chat_prompts_token_ids(
    model: str, system: str, prompts: list[str]
) -> tuple[list[str], list[list[int]], list[int]]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    rendered = [
        tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]
    token_ids = [tok.encode(p, add_special_tokens=False) for p in rendered]
    return rendered, token_ids, [len(t) for t in token_ids]


def run_generate(args: argparse.Namespace, prompts: list[str], prompt_counts: list[int]):
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig

    llm_kwargs: dict[str, Any] = {
        "backend": args.backend,
        "max_batch_size": args.max_batch_size or _max_active(args),
        "max_seq_len": args.max_model_len,
        "trust_remote_code": True,
        "dtype": args.dtype,
    }
    if args.max_num_tokens is not None:
        llm_kwargs["max_num_tokens"] = args.max_num_tokens
    if args.kv_cache_free_gpu_mem_fraction is not None:
        llm_kwargs["kv_cache_config"] = KvCacheConfig(
            free_gpu_memory_fraction=args.kv_cache_free_gpu_mem_fraction
        )
    if args.enable_chunked_prefill is not None:
        llm_kwargs["enable_chunked_prefill"] = args.enable_chunked_prefill
    if args.single_process:
        llm_kwargs["env_overrides"] = {"TLLM_WORKER_USE_SINGLE_PROCESS": "1"}

    llm = LLM(model=args.model, tensor_parallel_size=args.tp_size, **llm_kwargs)
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
        detokenize=False,
        add_special_tokens=False,
    )

    if args.warmup:
        warmup_sampling = sampling
        if args.warmup_max_tokens is not None:
            warmup_sampling = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.warmup_max_tokens,
                ignore_eos=args.ignore_eos,
                detokenize=False,
                add_special_tokens=False,
            )
        llm.generate(prompts[: args.warmup], warmup_sampling, use_tqdm=False)

    run_prompts = prompts[args.warmup :]
    run_prompt_counts = prompt_counts[args.warmup :]
    results: list[RequestResult] = []
    start = time.perf_counter()
    if args.mode == "latency":
        for prompt, prompt_count in zip(run_prompts, run_prompt_counts):
            req_start = time.perf_counter()
            outputs = llm.generate([prompt], sampling, use_tqdm=False)
            req_wall = time.perf_counter() - req_start
            for output in outputs:
                results.append(
                    RequestResult(
                        True,
                        req_wall,
                        len(_extract_token_ids(output)),
                        prompt_count,
                    )
                )
    else:
        outputs = llm.generate(run_prompts, sampling, use_tqdm=False)
        results = [
            RequestResult(True, 0.0, len(_extract_token_ids(output)), prompt_count)
            for output, prompt_count in zip(outputs, run_prompt_counts)
        ]
    wall = time.perf_counter() - start
    return wall, results, {
        "backend": args.backend,
        "max_batch_size": llm_kwargs.get("max_batch_size"),
        "max_num_tokens": llm_kwargs.get("max_num_tokens"),
        "max_seq_len": llm_kwargs.get("max_seq_len"),
        "kv_cache_free_gpu_mem_fraction": args.kv_cache_free_gpu_mem_fraction,
        "single_process": args.single_process,
    }


def run_pyexecutor(
    args: argparse.Namespace, token_ids: list[list[int]], prompt_counts: list[int]
):
    from pie_driver_tensorrt_llm.config import (
        TensorRTLLMDriverConfig,
        TensorRTLLMRuntimeConfig,
    )
    from pie_driver_tensorrt_llm.engine import TensorRTLLMEngine
    from tensorrt_llm.llmapi import KvCacheConfig

    runtime_config = TensorRTLLMRuntimeConfig.from_args(
        args.model,
        tensor_parallel_size=args.tp_size,
        activation_dtype=args.dtype,
    )
    llm_kwargs: dict[str, Any] = {
        "max_batch_size": args.max_batch_size or _max_active(args),
    }
    if args.max_num_tokens is not None:
        llm_kwargs["max_num_tokens"] = args.max_num_tokens
    if args.max_model_len is not None:
        llm_kwargs["max_seq_len"] = args.max_model_len
    if args.kv_cache_free_gpu_mem_fraction is not None:
        llm_kwargs["kv_cache_config"] = KvCacheConfig(
            free_gpu_memory_fraction=args.kv_cache_free_gpu_mem_fraction
        )
    if args.enable_chunked_prefill is not None:
        llm_kwargs["enable_chunked_prefill"] = args.enable_chunked_prefill

    engine = TensorRTLLMEngine.load(
        runtime_config,
        TensorRTLLMDriverConfig(
            backend=args.backend,
            execution_mode="pyexecutor",
            pyexecutor_max_tokens=args.pyexecutor_max_tokens,
            pyexecutor_lookahead=args.pyexecutor_lookahead,
            pyexecutor_lookahead_min_batch_size=args.pyexecutor_lookahead_min_batch_size,
            pyexecutor_direct_token_limit=args.pyexecutor_direct_token_limit,
            pyexecutor_speculative_lookahead=args.pyexecutor_speculative_lookahead,
            lookahead_tokens=args.lookahead_tokens,
            llm_kwargs=llm_kwargs,
        ),
    )

    next_context_id = 1

    def run_slice(req_token_ids, req_prompt_counts, *, timed: bool):
        nonlocal next_context_id
        max_active = _max_active(args) if args.mode == "tput" else 1
        results: list[RequestResult] = []
        start = time.perf_counter()
        for offset in range(0, len(req_token_ids), max_active):
            wave_tokens = req_token_ids[offset : offset + max_active]
            wave_counts = req_prompt_counts[offset : offset + max_active]
            context_ids = list(
                range(next_context_id, next_context_id + len(wave_tokens))
            )
            next_context_id += len(wave_tokens)
            prev = None
            generated_counts = [0] * len(wave_tokens)
            req_start = time.perf_counter()
            step = 0
            while step < args.max_tokens:
                out = engine.fire_batch(
                    {
                        "batch": _pyexecutor_batch(
                            context_ids=context_ids,
                            prompt_token_ids=wave_tokens,
                            prev_tokens=prev,
                            step=step,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            output_spec_flags=args.pyexecutor_speculative_lookahead,
                        )
                    },
                    {"batch": None},
                )
                accepted = out.get("spec_accepted_tokens")
                if accepted:
                    emitted: list[list[int]] = []
                    for toks in accepted:
                        emitted.append([int(t) for t in (toks or [])])
                    emitted_counts = [len(toks) for toks in emitted]
                    if not emitted_counts or any(n <= 0 for n in emitted_counts):
                        raise RuntimeError(
                            "TensorRT-LLM direct-accepted path returned an empty token list"
                        )
                    if len(set(emitted_counts)) != 1:
                        raise RuntimeError(
                            "standalone direct-accepted benchmark requires uniform "
                            "accepted-token counts across the wave"
                        )
                    accepted_count = emitted_counts[0]
                    prev = [toks[-1] for toks in emitted]
                    for i, count in enumerate(emitted_counts):
                        generated_counts[i] += count
                    step += accepted_count
                else:
                    prev = [int(t) for t in out["tokens"]]
                    for i in range(len(prev)):
                        generated_counts[i] += 1
                    step += 1
            req_wall = time.perf_counter() - req_start
            results.extend(
                RequestResult(
                    True,
                    req_wall if args.mode == "latency" else 0.0,
                    n,
                    count,
                )
                for n, count in zip(generated_counts, wave_counts)
            )
            for context_id in context_ids:
                session = engine._pyexecutor_sessions.get(int(context_id))
                if session is not None:
                    engine._terminate_pyexecutor_session(int(context_id), session)
        return (time.perf_counter() - start if timed else 0.0), results

    if args.warmup:
        old_max_tokens = args.max_tokens
        if args.warmup_max_tokens is not None:
            args.max_tokens = args.warmup_max_tokens
        try:
            run_slice(token_ids[: args.warmup], prompt_counts[: args.warmup], timed=False)
        finally:
            args.max_tokens = old_max_tokens

    wall, results = run_slice(token_ids[args.warmup :], prompt_counts[args.warmup :], timed=True)
    return wall, results, {
        "backend": args.backend,
        "max_batch_size": llm_kwargs.get("max_batch_size"),
        "max_num_tokens": llm_kwargs.get("max_num_tokens"),
        "max_seq_len": llm_kwargs.get("max_seq_len"),
        "pyexecutor_max_tokens": args.pyexecutor_max_tokens,
        "pyexecutor_lookahead": args.pyexecutor_lookahead,
        "pyexecutor_lookahead_min_batch_size": args.pyexecutor_lookahead_min_batch_size,
        "pyexecutor_direct_token_limit": args.pyexecutor_direct_token_limit,
        "pyexecutor_speculative_lookahead": args.pyexecutor_speculative_lookahead,
        "lookahead_tokens": args.lookahead_tokens,
    }


def _pyexecutor_batch(
    *,
    context_ids: list[int],
    prompt_token_ids: list[list[int]],
    prev_tokens: list[int] | None,
    step: int,
    temperature: float,
    top_p: float,
    output_spec_flags: bool = False,
):
    if step == 0:
        token_ids: list[int] = []
        position_ids: list[int] = []
        qo_indptr = [0]
        for prompt in prompt_token_ids:
            token_ids.extend(int(t) for t in prompt)
            position_ids.extend(range(len(prompt)))
            qo_indptr.append(len(token_ids))
        indices_for_logits = [qo_indptr[i + 1] - 1 for i in range(len(prompt_token_ids))]
    else:
        assert prev_tokens is not None
        token_ids = [int(t) for t in prev_tokens]
        position_ids = [len(prompt) + step - 1 for prompt in prompt_token_ids]
        qo_indptr = list(range(len(prompt_token_ids) + 1))
        indices_for_logits = list(range(len(prompt_token_ids)))

    return SimpleNamespace(
        has_speculative_inputs=False,
        adapter_subpass_needed=False,
        sampling_masks=None,
        logit_masks=None,
        request_output_counts=[1] * len(context_ids),
        qo_indptr=qo_indptr,
        token_ids=token_ids,
        position_ids=position_ids,
        context_ids=context_ids,
        kv_page_indptr=[0] * (len(context_ids) + 1),
        kv_page_indices=[],
        sampler_types=[3] * len(context_ids),
        indices_for_logits=indices_for_logits,
        temperatures=[temperature] * len(context_ids),
        top_k_values=[0] * len(context_ids),
        top_p_values=[top_p] * len(context_ids),
        min_p_values=[0.0] * len(context_ids),
        sampler_seeds_arr=[0] * len(context_ids),
        output_spec_flags=[bool(output_spec_flags)] * len(context_ids),
    )


def _extract_token_ids(output: Any) -> list[int]:
    completions = getattr(output, "outputs", None)
    if completions:
        first = completions[0]
        token_ids = getattr(first, "token_ids", None)
        if token_ids:
            return [int(t) for t in token_ids]
        token_ids_diff = getattr(first, "token_ids_diff", None)
        if token_ids_diff:
            return [int(t) for t in token_ids_diff]
    return []


def _max_active(args: argparse.Namespace) -> int:
    if args.mode == "latency":
        return 1
    if args.concurrency == 0:
        return max(1, args.num_requests)
    return int(args.concurrency)


def run(args: argparse.Namespace):
    ensure_cuda_library_path()

    n = args.requests if args.mode == "latency" else args.num_requests
    prompts = make_prompts(args, n + args.warmup)
    rendered, token_ids, prompt_counts = chat_prompts_token_ids(
        args.model, args.system, prompts
    )
    if args.execution_mode == "generate":
        wall, results, config = run_generate(args, rendered, prompt_counts)
    else:
        wall, results, config = run_pyexecutor(args, token_ids, prompt_counts)

    summary = summarize(
        mode=args.mode,
        engine="tensorrt_llm",
        model=args.model,
        results=results,
        wall_s=wall,
        config={
            "execution_mode": args.execution_mode,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "ignore_eos": args.ignore_eos,
            "unique_prompts": args.unique_prompts,
            **config,
        },
    )
    return summary, results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TensorRT-LLM standalone canonical latency/throughput benchmark"
    )
    add_mode_subcommands(parser)
    for sp in parser._subparsers._group_actions[0].choices.values():
        sp.add_argument(
            "--execution-mode",
            choices=["generate", "pyexecutor"],
            default="generate",
        )
        sp.add_argument("--backend", default="pytorch")
        sp.add_argument("--dtype", default="bfloat16")
        sp.add_argument("--max-batch-size", type=int, default=None)
        sp.add_argument("--max-num-tokens", type=int, default=None)
        sp.add_argument("--pyexecutor-max-tokens", type=int, default=4096)
        sp.add_argument("--pyexecutor-lookahead", action="store_true")
        sp.add_argument("--pyexecutor-lookahead-min-batch-size", type=int, default=None)
        sp.add_argument("--pyexecutor-direct-token-limit", type=int, default=None)
        sp.add_argument("--pyexecutor-speculative-lookahead", action="store_true")
        sp.add_argument("--lookahead-tokens", type=int, default=16)
        sp.add_argument("--kv-cache-free-gpu-mem-fraction", type=float, default=0.90)
        sp.add_argument(
            "--enable-chunked-prefill",
            action=argparse.BooleanOptionalAction,
            default=None,
        )
        sp.add_argument("--single-process", action="store_true")
    args = parser.parse_args()
    summary, results = run(args)
    finish(summary, results, args.json_out)


if __name__ == "__main__":
    main()
