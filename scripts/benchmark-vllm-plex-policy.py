#!/usr/bin/env python3

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams


FAVORED_START = 8


def policy_facts(
    policy_id: str,
    index: int,
    favored_start: int = FAVORED_START,
) -> dict[str, Any]:
    favored = index >= favored_start
    facts: dict[str, Any] = {
        "weight": 8 if favored else 1,
        "application_id": f"application-{index}",
        "kv_overloaded": True,
        "interaction_in_progress": favored,
        "user_rpm_remaining": 1,
        "app_rpm_remaining": 1,
        "quantum": 1024 if favored else 0,
        "extend_tokens": 1,
        "worker_deficit": 1024 if favored else -1,
        "virtual_wait": 0 if favored else 1000,
        "slack_ms": 0 if favored else 1000,
        "waiting_ms": 10000 if favored else 0,
        "service_us": 0,
        "service_tokens": 0,
        "input_weight": 1,
        "output_weight": 1,
        "completed_branches": 0 if favored else 8,
        "group_service": 0 if favored else 1000,
        "workflow_rank": 0 if favored else 100,
        "dependency_ready": favored,
        "dependency_depth": 10 if favored else 1,
        "prefix_reuse_tokens": 4096 if favored else 0,
        "profiled_token_cost": 1 if favored else 1000,
        "ready": favored,
        "cache_ready": favored,
        "preempted": favored,
        "program_arrival": index,
        "resuming": favored,
        "expected_waste_tokens": 0 if favored else 1000,
        "fairness_threshold_ms": 1000,
        "demand_depth": 20 if favored else 1,
        "confidence_ppm": 700000,
        "stop_threshold_ppm": 900000,
        "progress_ppm": 900000 if favored else 100000,
        "queue_class": 0 if favored else 3,
        "weighted_size": 1,
        "queue_quota": 8,
        "upstream_elapsed_ms": 0,
        "current_queue_ms": 0,
        "current_execution_ms": 1,
        "downstream_queue_ms": 0,
        "downstream_execution_ms": 1,
        "downstream_batch_wait_p10_ms": 0,
        "deadline_ms": 1000,
        "excess_branch": False,
        "tool_ready": favored,
        "tool_failed": False,
        "migrate_target": None,
        "program_live": favored,
        "ttl_ms": 5000 if favored else 0,
        "ttl_expired": False,
        "reload_cost": 1000 if favored else 1,
        "fixed_prefix": favored,
        "steps_to_execution": 1 if favored else 100,
        "loading": False,
        "offloading": False,
        "prefetch": False,
        "reuse_probability_ppm": 900000 if favored else 10000,
        "recompute_flops": 1000,
        "leaf": True,
        "frequency": 100 if favored else 1,
        "recompute_cost": 1000 if favored else 1,
        "age": 100 if favored else 0,
        "expected_reuse_ms": 1 if favored else 1000,
        "recompute_ms": 1000,
        "pending_demand_depth": 100 if favored else 0,
        "adapter_hot": favored,
        "hotness": 100 if favored else 0,
        "next_use_step": 1 if favored else 1000,
        "workflow_ttl_ms": 5000 if favored else 0,
    }
    if policy_id == "parrot" and not favored:
        facts["dependency_ready"] = False
    return facts


def metadata(
    policy_id: str,
    index: int,
    prefix: str,
    favored_start: int = FAVORED_START,
) -> dict[str, Any]:
    group_id = (
        f"{policy_id}-group-{index}"
        if policy_id in {"agentix", "justitia", "qlm", "saga"}
        else None
    )
    return {
        "request_id": f"{prefix}-{index}",
        "principal_id": f"client-{index}",
        "group_id": group_id,
        "metadata": {
            "facts": policy_facts(policy_id, index, favored_start)
        },
    }


def sampling_params(
    policy_id: str,
    prefix: str,
    batch: int,
    output_tokens: int,
    favored_start: int = FAVORED_START,
) -> list[SamplingParams]:
    return [
        SamplingParams(
            temperature=0,
            max_tokens=output_tokens,
            ignore_eos=True,
            seed=0,
            extra_args={
                "plex": metadata(
                    policy_id, index, prefix, favored_start
                )
            },
        )
        for index in range(batch)
    ]


def request_metrics(outputs: list[Any]) -> list[dict[str, float]]:
    metrics = []
    for output in outputs:
        state = output.metrics
        if state is None:
            raise RuntimeError("vLLM request metrics are unavailable")
        generated = max(state.num_generation_tokens, 1)
        metrics.append(
            {
                "ttft_s": state.first_token_ts - state.queued_ts,
                "queue_s": state.scheduled_ts - state.queued_ts,
                "e2e_s": state.last_token_ts - state.queued_ts,
                "tpot_s": (
                    (state.last_token_ts - state.first_token_ts)
                    / max(generated - 1, 1)
                ),
            }
        )
    return metrics


def aggregate_metrics(
    values: list[dict[str, float]],
    favored_start: int,
) -> dict[str, Any]:
    favored = values[favored_start:]
    regular = values[:favored_start]

    def mean(items: list[dict[str, float]], key: str) -> float:
        return statistics.mean(item[key] for item in items)

    return {
        "mean_ttft_s": mean(values, "ttft_s"),
        "p95_ttft_s": sorted(item["ttft_s"] for item in values)[
            max(int(len(values) * 0.95) - 1, 0)
        ],
        "mean_e2e_s": mean(values, "e2e_s"),
        "mean_tpot_s": mean(values, "tpot_s"),
        "favored_mean_ttft_s": mean(favored, "ttft_s"),
        "regular_mean_ttft_s": mean(regular, "ttft_s"),
        "favored_mean_queue_s": mean(favored, "queue_s"),
        "regular_mean_queue_s": mean(regular, "queue_s"),
    }


def calibration(llm: LLM, policy_id: str) -> None:
    batch = FAVORED_START
    prompts = [
        f"Calibration request {index} with unique context {index}."
        for index in range(batch)
    ]
    llm.generate(
        prompts,
        sampling_params(policy_id, "calibration", batch, 16),
        use_tqdm=False,
    )


def run_schedule(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": args.model,
        "gpu_memory_utilization": 0.5,
        "max_num_seqs": args.batch,
        "max_num_batched_tokens": max(args.batch, 512),
        "disable_log_stats": False,
        "seed": 0,
    }
    if args.package is not None:
        kwargs["plex_policy"] = str(args.package)
    llm = LLM(**kwargs)
    calibration(llm, args.policy_id)
    runs = []
    for repeat in range(args.repeats):
        prompts = [
            (
                f"Repeat {repeat}, request {index}: explain deterministic "
                "scheduling briefly."
                + " context" * 64
            )
            for index in range(args.batch)
        ]
        started = time.perf_counter()
        outputs = llm.generate(
            prompts,
            sampling_params(
                args.policy_id,
                f"measured-{repeat}",
                args.batch,
                args.output_tokens,
            ),
            use_tqdm=False,
        )
        elapsed = time.perf_counter() - started
        total_tokens = sum(
            len(output.outputs[0].token_ids) for output in outputs
        )
        token_ids = [
            list(output.outputs[0].token_ids) for output in outputs
        ]
        metrics = request_metrics(outputs)
        runs.append(
            {
                "elapsed_s": elapsed,
                "throughput_tokens_per_second": total_tokens / elapsed,
                "request_metrics": metrics,
                "aggregate": aggregate_metrics(
                    metrics, FAVORED_START
                ),
                "token_sha256": hashlib.sha256(
                    json.dumps(
                        token_ids, separators=(",", ":")
                    ).encode()
                ).hexdigest(),
            }
        )
    return {
        "mode": "schedule",
        "policy_id": args.policy_id,
        "native": args.package is None,
        "batch": args.batch,
        "output_tokens": args.output_tokens,
        "repeats": args.repeats,
        "median_throughput_tokens_per_second": statistics.median(
            run["throughput_tokens_per_second"] for run in runs
        ),
        "runs": runs,
    }


def run_cache_pressure(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": args.model,
        "max_model_len": 1024,
        "kv_cache_memory_bytes": 13 * 1024 * 1024,
        "max_num_seqs": 8,
        "max_num_batched_tokens": 1024,
        "disable_log_stats": False,
        "seed": 0,
    }
    if args.package is not None:
        kwargs["plex_policy"] = str(args.package)
    llm = LLM(**kwargs)
    batch = 8
    runs = []
    for repeat in range(args.repeats):
        prompts = [
            (
                f"Cache repeat {repeat}, request {index}."
                + " token" * 512
            )
            for index in range(batch)
        ]
        started = time.perf_counter()
        outputs = llm.generate(
            prompts,
            sampling_params(
                args.policy_id,
                f"cache-{repeat}",
                batch,
                128,
                favored_start=4,
            ),
            use_tqdm=False,
        )
        elapsed = time.perf_counter() - started
        total_tokens = sum(
            len(output.outputs[0].token_ids) for output in outputs
        )
        token_ids = [
            list(output.outputs[0].token_ids) for output in outputs
        ]
        metrics = request_metrics(outputs)
        runs.append(
            {
                "elapsed_s": elapsed,
                "throughput_tokens_per_second": total_tokens / elapsed,
                "request_metrics": metrics,
                "aggregate": aggregate_metrics(metrics, 4),
                "token_sha256": hashlib.sha256(
                    json.dumps(
                        token_ids, separators=(",", ":")
                    ).encode()
                ).hexdigest(),
            }
        )
    return {
        "mode": "cache-pressure",
        "policy_id": args.policy_id,
        "native": args.package is None,
        "batch": batch,
        "output_tokens": 128,
        "repeats": args.repeats,
        "median_throughput_tokens_per_second": statistics.median(
            run["throughput_tokens_per_second"] for run in runs
        ),
        "runs": runs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--mode", choices=["schedule", "cache-pressure"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if args.batch <= FAVORED_START and args.mode == "schedule":
        raise SystemExit(f"--batch must be greater than {FAVORED_START}")
    result = (
        run_schedule(args)
        if args.mode == "schedule"
        else run_cache_pressure(args)
    )
    result["package_sha256"] = (
        hashlib.sha256(args.package.read_bytes()).hexdigest()
        if args.package is not None
        else None
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
