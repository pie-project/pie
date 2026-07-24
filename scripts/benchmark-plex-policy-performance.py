#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from pie_plex import Runtime


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGES = ROOT / "tests" / "policies" / "target" / "packages"
DEFAULT_TARGETS = ROOT / "tests" / "policies" / "performance-targets.json"
MECHANICS = [
    "request.cancel@1",
    "group.cancel@1",
    "cache.prefetch@1",
    "cache.swap@1",
    "request.rebalance@1",
    "schedule.atomic-enqueue@1",
]


class PolicyBench:
    def __init__(self, policy_id: str, package: Path) -> None:
        self.policy_id = policy_id
        self.runtime = Runtime(str(package), mechanics=MECHANICS)
        self.package_sha256 = hashlib.sha256(package.read_bytes()).hexdigest()
        self.sequence = 0
        self.latencies_us: list[float] = []
        self.operation_counts: Counter[str] = Counter()

    def next_id(self, label: str) -> str:
        self.sequence += 1
        return f"{self.policy_id}-{label}-{self.sequence}"

    def meta(self, label: str) -> dict[str, Any]:
        return {
            "opportunity_id": self.next_id(label),
            "snapshot": {"id": "host-filled", "revision": 0},
            "attempt": 0,
            "mechanics": [],
        }

    def invoke(
        self,
        operation: str,
        context: dict[str, Any],
        lifecycle: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "api_version": "pie.plex.engine@2",
            "operation": operation,
            "context": context,
        }
        if lifecycle:
            event["lifecycle"] = lifecycle
        started = time.perf_counter_ns()
        outcome = self.runtime.invoke(event)
        self.latencies_us.append((time.perf_counter_ns() - started) / 1000)
        self.operation_counts[operation] += 1
        if outcome.get("status") != "success":
            raise RuntimeError(
                f"{self.policy_id}/{operation} returned {outcome}"
            )
        return outcome

    def feedback(
        self,
        records: list[dict[str, Any]],
        lifecycle: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return self.invoke(
            "feedback",
            {
                "delivery_id": self.next_id("feedback"),
                "records": records,
            },
            lifecycle,
        )

    def latency_summary(self) -> dict[str, float]:
        ordered = sorted(self.latencies_us)
        return {
            "median_us": statistics.median(ordered),
            "p95_us": percentile(ordered, 0.95),
            "max_us": ordered[-1],
            "invocations": len(ordered),
        }


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    index = min(math.ceil(quantile * len(values)) - 1, len(values) - 1)
    return values[max(index, 0)]


def jain(values: list[float]) -> float:
    total = sum(values)
    squares = sum(value * value for value in values)
    return total * total / (len(values) * squares) if squares else 1.0


def request_ref(
    request_id: str,
    group_id: str | None = None,
    generation_id: int = 0,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "generation_id": generation_id,
        "group_id": group_id,
        "principal_id": "benchmark",
    }


def bootstrap_requests(
    bench: PolicyBench,
    label: str,
    facts: list[dict[str, Any]],
    *,
    status: str,
    group_keys: list[str | None] | None = None,
    group_facts: dict[str, dict[str, Any]] | None = None,
    generations: list[int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    token = bench.next_id(label)
    group_keys = group_keys or [None] * len(facts)
    generations = generations or [0] * len(facts)
    group_ids = {
        key: f"{token}-group-{key}" for key in group_keys if key is not None
    }
    lifecycle: list[dict[str, Any]] = []
    for key, group_id in group_ids.items():
        lifecycle.append(
            {
                "event": "create-group",
                "group_id": group_id,
                "principal_id": "benchmark",
                "limits": {
                    "max_members": max(len(facts), 1),
                    "max_scratch_bytes": 65536,
                },
                "facts": (group_facts or {}).get(key, {}),
            }
        )
    refs = []
    for index, (request_facts, key, generation_id) in enumerate(
        zip(facts, group_keys, generations)
    ):
        request_id = f"{token}-request-{index}"
        group_id = group_ids.get(key)
        lifecycle.append(
            {
                "event": "create-request",
                "request_id": request_id,
                "principal_id": "benchmark",
                "group_id": group_id,
                "fields": {"body": {}, "metadata": {}},
                "facts": request_facts,
            }
        )
        if status in {"admitted", "active"}:
            lifecycle.append(
                {"event": "admit-request", "request_id": request_id}
            )
        if status == "active":
            lifecycle.append(
                {"event": "activate-request", "request_id": request_id}
            )
        if generation_id > 0:
            for generation in range(1, generation_id + 1):
                lifecycle.append(
                    {
                        "event": "continue-request",
                        "request_id": request_id,
                        "fields": {"body": {}, "metadata": {}},
                        "facts": {
                            **request_facts,
                            "generation_id": generation,
                        },
                    }
                )
        refs.append(request_ref(request_id, group_id, generation_id))
    return refs, lifecycle


def invoke_schedule(
    bench: PolicyBench,
    refs: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    lifecycle: list[dict[str, Any]] | None,
    *,
    max_selections: int = 1,
    max_requests: int | None = None,
    token_budget: int = 1,
    capacity_facts: dict[str, Any] | None = None,
) -> tuple[list[int], dict[str, Any]]:
    request_limit = max_requests or max_selections
    meta = bench.meta("schedule")
    outcome = bench.invoke(
        "schedule",
        {
            "meta": meta,
            "cause": "capacity-changed",
            "runnable": [
                {
                    "request": ref,
                    "max_token_budget": token_budget,
                    "facts": item,
                }
                for ref, item in zip(refs, facts)
            ],
            "capacity": {
                "max_selections": max_selections,
                "max_requests": request_limit,
                "max_total_tokens": request_limit * token_budget,
                "facts": capacity_facts or {},
            },
        },
        lifecycle,
    )
    outcome["_benchmark_opportunity_id"] = meta["opportunity_id"]
    selections = outcome["plan"]["plan"]["selections"]
    selected = [
        request_index
        for selection in selections
        for request_index in selection["requests"]
    ]
    return selected, outcome


def enact_schedule(
    bench: PolicyBench,
    outcome: dict[str, Any],
    *,
    scheduled_tokens: int | None = None,
) -> None:
    opportunity_id = outcome["_benchmark_opportunity_id"]
    records = []
    for selection_index, selection in enumerate(
        outcome["plan"]["plan"]["selections"]
    ):
        requested = sum(selection["token_budgets"])
        enacted = requested if scheduled_tokens is None else min(
            requested, scheduled_tokens
        )
        records.append(
            {
                "subject": {
                    "kind": "schedule-selection",
                    "value": {
                        "opportunity_id": opportunity_id,
                        "selection_index": selection_index,
                    },
                },
                "outcome": "progress",
                "facts": {
                    "status": (
                        "enacted"
                        if enacted == requested
                        else "partially-enacted"
                        if enacted > 0
                        else "not-enacted"
                    ),
                    "requested_tokens": requested,
                    "scheduled_tokens": enacted,
                },
            }
        )
    if records:
        bench.feedback(records)


def invoke_admit(
    bench: PolicyBench,
    refs: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    lifecycle: list[dict[str, Any]],
    *,
    max_accepted: int,
) -> list[str]:
    outcome = bench.invoke(
        "admit",
        {
            "meta": bench.meta("admit"),
            "cause": "arrival",
            "candidates": [
                {"request": ref, "demand": [], "facts": item}
                for ref, item in zip(refs, facts)
            ],
            "capacity": {
                "max_accepted": max_accepted,
                "limits": [],
                "facts": {},
            },
        },
        lifecycle,
    )
    return outcome["plan"]["plan"]["decisions"]


def invoke_route(
    bench: PolicyBench,
    refs: list[dict[str, Any]],
    request_facts: list[dict[str, Any]],
    target_facts: list[dict[str, Any]],
    edge_facts: list[list[dict[str, Any]]],
    lifecycle: list[dict[str, Any]],
    *,
    target_capacity: int | None = None,
) -> tuple[list[int | None], dict[str, Any]]:
    meta = bench.meta("route")
    targets = [
        {
            "target_id": f"target-{index}",
            "max_assignments": target_capacity or len(refs),
            "capacity": [],
            "revision": 0,
            "facts": facts,
        }
        for index, facts in enumerate(target_facts)
    ]
    edges = [
        {
            "request_index": request_index,
            "target_index": target_index,
            "demand": [],
            "facts": facts,
        }
        for request_index, per_request in enumerate(edge_facts)
        for target_index, facts in enumerate(per_request)
    ]
    outcome = bench.invoke(
        "route",
        {
            "meta": meta,
            "cause": "admission",
            "requests": [
                {"request": ref, "facts": facts}
                for ref, facts in zip(refs, request_facts)
            ],
            "targets": targets,
            "feasible_edges": edges,
        },
        lifecycle,
    )
    outcome["_benchmark_opportunity_id"] = meta["opportunity_id"]
    assignments: list[int | None] = [None] * len(refs)
    for assignment in outcome["plan"]["plan"]["assignments"]:
        assignments[assignment["request_index"]] = assignment["target_index"]
    return assignments, outcome


def enact_route(bench: PolicyBench, outcome: dict[str, Any]) -> None:
    opportunity_id = outcome["_benchmark_opportunity_id"]
    records = [
        {
            "subject": {
                "kind": "route-assignment",
                "value": {
                    "opportunity_id": opportunity_id,
                    "request_index": assignment["request_index"],
                },
            },
            "outcome": "progress",
            "facts": {
                "status": "enacted",
                "target_index": assignment["target_index"],
            },
        }
        for assignment in outcome["plan"]["plan"]["assignments"]
    ]
    if records:
        bench.feedback(records)


def cache_object(
    object_id: str,
    facts: dict[str, Any],
    *,
    size_bytes: int = 1,
    beneficiary: str | None = None,
) -> dict[str, Any]:
    beneficiaries = (
        [{"kind": "request", "id": beneficiary}]
        if beneficiary is not None
        else []
    )
    return {
        "object_id": object_id,
        "size_bytes": size_bytes,
        "beneficiaries": beneficiaries,
        "beneficiary_count": len(beneficiaries),
        "facts": facts,
    }


def invoke_cache(
    bench: PolicyBench,
    residents: list[dict[str, Any]],
    prospective: list[dict[str, Any]],
    lifecycle: list[dict[str, Any]] | None,
    *,
    max_bytes: int,
    capacity_facts: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    meta = bench.meta("cache")
    outcome = bench.invoke(
        "cache",
        {
            "meta": meta,
            "cause": "pressure",
            "resident": [
                {"object": item, "reclaimable": True} for item in residents
            ],
            "prospective": prospective,
            "capacity": {
                "max_bytes": max_bytes,
                "fixed_bytes": 0,
                "facts": capacity_facts or {},
            },
            "episode": None,
        },
        lifecycle,
    )
    outcome["_benchmark_opportunity_id"] = meta["opportunity_id"]
    return outcome["plan"]["plan"], outcome


def complete_actions(bench: PolicyBench, outcome: dict[str, Any]) -> None:
    records = [
        {
            "subject": {"kind": "action", "value": action["id"]},
            "outcome": "action-succeeded",
            "facts": {
                "opportunity_id": outcome["_benchmark_opportunity_id"],
                "method": action["method"],
                "idempotency_key": action["args"]["idempotency_key"],
                "status": "succeeded",
                "details": {},
            },
        }
        for action in outcome.get("actions", [])
    ]
    if records:
        bench.feedback(records)


def exercise_feedback(
    bench: PolicyBench,
    label: str,
    facts: dict[str, Any] | None = None,
) -> None:
    refs, lifecycle = bootstrap_requests(
        bench,
        label,
        [facts or {}],
        status="active",
    )
    bench.feedback(
        [
            {
                "subject": {
                    "kind": "request",
                    "value": refs[0]["request_id"],
                },
                "outcome": "progress",
                "facts": facts or {},
            }
        ],
        lifecycle,
    )


def result(
    policy_values: list[float],
    baseline_values: list[float],
    *,
    direction: str,
    unit: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy_value = statistics.mean(policy_values)
    baseline_value = statistics.mean(baseline_values)
    if policy_value == 0 and baseline_value == 0:
        ratio = 1.0
    elif direction == "lower":
        ratio = baseline_value / policy_value if policy_value else math.inf
    else:
        ratio = policy_value / baseline_value if baseline_value else math.inf
    comparisons = [
        (policy < baseline if direction == "lower" else policy > baseline)
        - (policy > baseline if direction == "lower" else policy < baseline)
        for policy, baseline in zip(policy_values, baseline_values)
    ]
    wins = comparisons.count(1)
    losses = comparisons.count(-1)
    ties = comparisons.count(0)
    return {
        "direction": direction,
        "unit": unit,
        "policy_value": policy_value,
        "baseline_value": baseline_value,
        "improvement_ratio": ratio,
        "trend_reproduced": ratio >= 1.0,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": wins / len(comparisons),
        "details": details or {},
    }


def scenario_agentix(
    bench: PolicyBench, rng: random.Random, trials: int
) -> dict[str, Any]:
    policy_jct = []
    baseline_jct = []
    for trial in range(trials):
        programs = [
            [rng.randint(5, 7), rng.randint(3, 5), rng.randint(2, 4)],
            [rng.randint(5, 8), rng.randint(3, 5)],
            [rng.randint(1, 2)],
            [rng.randint(1, 2), rng.randint(1, 2)],
        ]
        flat_calls = [
            (program_index, call_index)
            for program_index, calls in enumerate(programs)
            for call_index in range(len(calls))
        ]
        base_facts = [
            {
                "agentix_mode": "plas",
                "call_arrival": index,
                "call_wait_us": 0,
                "queue_bounds_us": [1000, 3000, 7000],
                "queue_quanta_us": [1000, 2000, 4000, 8000],
                "starvation_ratio_ppm": 100_000_000,
            }
            for index in range(len(flat_calls))
        ]
        refs, lifecycle = bootstrap_requests(
            bench,
            f"agentix-{trial}",
            base_facts,
            status="active",
            group_keys=[str(program_index) for program_index, _ in flat_calls],
        )
        ref_index = {
            call: index for index, call in enumerate(flat_calls)
        }
        remaining = [calls.copy() for calls in programs]
        current_call = [0] * len(programs)
        ready_at = [0] * len(programs)
        completed_at = [0] * len(programs)
        step = 0
        first_invocation = True
        while any(
            current_call[index] < len(programs[index])
            for index in range(len(programs))
        ):
            ready = [
                (program_index, current_call[program_index])
                for program_index in range(len(programs))
                if current_call[program_index] < len(programs[program_index])
                and ready_at[program_index] <= step
            ]
            selected_positions, _ = invoke_schedule(
                bench,
                [refs[ref_index[call]] for call in ready],
                [
                    {
                        "agentix_mode": "plas",
                        "call_arrival": (
                            ready_at[program_index] * 100
                            + ref_index[(program_index, call_index)]
                        ),
                        "call_wait_us": (
                            step - ready_at[program_index]
                        )
                        * 1000,
                        "queue_bounds_us": [1000, 3000, 7000],
                        "queue_quanta_us": [1000, 2000, 4000, 8000],
                        "starvation_ratio_ppm": 100_000_000,
                    }
                    for program_index, call_index in ready
                ],
                lifecycle if first_invocation else None,
                max_selections=2,
                token_budget=1,
            )
            first_invocation = False
            if not selected_positions:
                raise RuntimeError("Agentix produced an empty runnable batch")
            records = []
            completed_calls = []
            for position in selected_positions:
                program_index, call_index = ready[position]
                remaining[program_index][call_index] -= 1
                request_id = refs[
                    ref_index[(program_index, call_index)]
                ]["request_id"]
                records.append(
                    {
                        "subject": {
                            "kind": "request",
                            "value": request_id,
                        },
                        "outcome": "progress",
                        "facts": {
                            "service_us": 1000,
                            "queue_bounds_us": [1000, 3000, 7000],
                            "queue_quanta_us": [1000, 2000, 4000, 8000],
                        },
                    }
                )
                if remaining[program_index][call_index] == 0:
                    completed_calls.append((program_index, request_id))
                    records.append(
                        {
                            "subject": {
                                "kind": "request",
                                "value": request_id,
                            },
                            "outcome": "completed",
                            "facts": {
                                "call_wait_us": (
                                    step - ready_at[program_index]
                                )
                                * 1000
                            },
                        }
                    )
            bench.feedback(records)
            for program_index, _ in completed_calls:
                current_call[program_index] += 1
                if current_call[program_index] == len(programs[program_index]):
                    completed_at[program_index] = step + 1
                else:
                    ready_at[program_index] = step + 1
            step += 1
            if step > 1000:
                raise RuntimeError("Agentix trace did not converge")
        policy_jct.append(statistics.mean(completed_at))

        remaining = [calls.copy() for calls in programs]
        current_call = [0] * len(programs)
        ready_at = [0] * len(programs)
        completed_at = [0] * len(programs)
        running: list[tuple[int, int]] = []
        queued: list[tuple[int, int]] = []
        step = 0
        while any(
            current_call[index] < len(programs[index])
            for index in range(len(programs))
        ):
            for program_index in range(len(programs)):
                call = (program_index, current_call[program_index])
                if (
                    call[1] < len(programs[program_index])
                    and ready_at[program_index] <= step
                    and call not in running
                    and call not in queued
                ):
                    queued.append(call)
            while queued and len(running) < 2:
                running.append(queued.pop(0))
            finished = []
            for program_index, call_index in running:
                remaining[program_index][call_index] -= 1
                if remaining[program_index][call_index] == 0:
                    finished.append((program_index, call_index))
            for call in finished:
                running.remove(call)
                program_index, _ = call
                current_call[program_index] += 1
                if current_call[program_index] == len(programs[program_index]):
                    completed_at[program_index] = step + 1
                else:
                    ready_at[program_index] = step + 1
            step += 1
        baseline_jct.append(statistics.mean(completed_at))
    return result(
        policy_jct,
        baseline_jct,
        direction="lower",
        unit="mean_program_jct_steps",
    )


def scenario_continuum(
    bench: PolicyBench, rng: random.Random, trials: int
) -> dict[str, Any]:
    policy_costs = []
    baseline_costs = []
    for trial in range(trials):
        history = sorted(rng.randint(100, 800) for _ in range(5))
        facts = [{"program_arrival": index} for index in range(4)]
        refs, lifecycle = bootstrap_requests(
            bench,
            f"continuum-{trial}",
            facts,
            status="active",
            group_keys=["a", "a", "b", "c"],
        )
        bench.feedback(
            [
                {
                    "subject": {
                        "kind": "request",
                        "value": refs[0]["request_id"],
                    },
                    "outcome": "progress",
                    "facts": {
                        "tool_id": "shell",
                        "tool_duration_ms": duration,
                    },
                }
                for duration in history
            ],
            lifecycle,
        )
        bench.feedback(
            [
                {
                    "subject": {
                        "kind": "request",
                        "value": refs[0]["request_id"],
                    },
                    "outcome": "completed",
                    "facts": {
                        "next_tool_id": "shell",
                        "now_ms": 1000,
                        "average_wait_ms": 500,
                        "memoryfulness_ppm": 1_000_000,
                        "prefill_reload_ms": 200,
                        "history_threshold": 1,
                        "program_arrival": 0,
                        "program_finished": False,
                    },
                }
            ]
        )
        schedule_facts = [
            {"now_ms": 1100, "program_arrival": 0},
            {"now_ms": 1100, "program_arrival": 1},
            {"now_ms": 1100, "program_arrival": 2},
        ]
        selected, _ = invoke_schedule(
            bench, refs[1:], schedule_facts, None
        )
        reload_costs = [rng.randint(100, 1000) for _ in range(3)]
        residents = [
            cache_object(
                f"continuum-{trial}-{index}",
                {},
                beneficiary=refs[index]["request_id"],
            )
            for index in range(3)
        ]
        plan, _ = invoke_cache(
            bench,
            residents,
            [],
            None,
            max_bytes=len(residents) - 1,
            capacity_facts={"now_ms": 1100},
        )
        policy_victim = plan["reclaim"][0]
        schedule_penalty = 0 if selected[0] == 0 else 5000
        policy_costs.append(
            schedule_penalty
            + reload_costs[policy_victim]
            + (10000 if policy_victim == 0 else 0)
        )
        baseline_costs.append(reload_costs[0] + 10000)
    return result(
        policy_costs,
        baseline_costs,
        direction="lower",
        unit="response_and_reload_cost",
    )


def scenario_kvflow(
    bench: PolicyBench, rng: random.Random, trials: int
) -> dict[str, Any]:
    policy_costs = []
    baseline_costs = []
    for trial in range(trials):
        facts = [{"required_objects": ["next"]}, {"cache_ready": True}]
        refs, lifecycle = bootstrap_requests(
            bench, f"kvflow-{trial}", facts, status="active"
        )
        residents = [
            cache_object(
                f"kvflow-{trial}-varying",
                {
                    "fixed_prefix": False,
                    "steps_to_execution": 0,
                    "cache_state": "gpu",
                },
                beneficiary=refs[0]["request_id"],
            ),
            cache_object(
                f"kvflow-{trial}-fixed",
                {
                    "fixed_prefix": True,
                    "beneficiary_steps": [2, 5],
                    "cache_state": "gpu",
                },
                beneficiary=refs[1]["request_id"],
            ),
        ]
        prospective = [
            cache_object(
                "next",
                {
                    "fixed_prefix": True,
                    "steps_to_execution": 1,
                    "cache_state": "cpu",
                    "prefetch": True,
                },
            )
        ]
        plan, cache_outcome = invoke_cache(
            bench,
            residents,
            prospective,
            lifecycle,
            max_bytes=2,
            capacity_facts={
                "prefetch_horizon_steps": 1,
                "max_concurrent_prefetches": 1,
            },
        )
        complete_actions(bench, cache_outcome)
        selected, _ = invoke_schedule(bench, refs, facts, None)
        victim = plan["reclaim"][0]
        harm = [1, 1000]
        policy_costs.append((0 if selected[0] == 0 else 500) + harm[victim])
        baseline_costs.append(500 + harm[0])
    return result(
        policy_costs,
        baseline_costs,
        direction="lower",
        unit="workflow_stall_cost",
    )


def scenario_preble(
    bench: PolicyBench, rng: random.Random, trials: int
) -> dict[str, Any]:
    policy_costs = []
    baseline_costs = []
    for trial in range(trials):
        exploit = trial % 2 == 0
        if exploit:
            remaining = 200
            cached = [800, 800, 0]
            rolling_load = [100, 120, 10]
            decoder_ratio = [900_000, 1_000_000, 800_000]
            total_cost = [
                100 + rng.randint(0, 10),
                80 + rng.randint(0, 10),
                1000 + rng.randint(0, 20),
            ]
        else:
            remaining = 500
            cached = [50, 50, 50]
            rolling_load = [100, 50, 10]
            decoder_ratio = [2_000_000, 1_000_000, 800_000]
            total_cost = [
                100 + rng.randint(0, 10),
                120 + rng.randint(0, 10),
                150 + rng.randint(0, 10),
            ]
        refs, lifecycle = bootstrap_requests(
            bench,
            f"preble-{trial}",
            [{"uncached_tokens": remaining}],
            status="admitted",
        )
        assignments, _ = invoke_route(
            bench,
            refs,
            [
                {
                    "uncached_tokens": remaining,
                    "decoder_imbalance_threshold_ppm": 1_500_000,
                    "balance_threshold_ppm": 10_000_000,
                }
            ],
            [
                {
                    "rolling_load_us": rolling_load[index],
                    "decoder_ratio_ppm": decoder_ratio[index],
                }
                for index in range(3)
            ],
            [
                [
                    {
                        "cached_tokens": cached[index],
                        "load_cost": total_cost[index] // 2,
                        "eviction_cost": total_cost[index] // 4,
                        "miss_prefill_cost": (
                            total_cost[index]
                            - total_cost[index] // 2
                            - total_cost[index] // 4
                        ),
                        "assignment_load_us": total_cost[index],
                    }
                    for index in range(3)
                ]
            ],
            lifecycle,
        )

        policy_costs.append(total_cost[assignments[0]])
        baseline_index = min(range(3), key=rolling_load.__getitem__)
        baseline_costs.append(total_cost[baseline_index])
    return result(
        policy_costs,
        baseline_costs,
        direction="lower",
        unit="prefix_route_cost",
    )


def scenario_helium(
    bench: PolicyBench, rng: random.Random, trials: int
) -> dict[str, Any]:
    policy_values = []
    baseline_values = []
    for trial in range(trials):
        costs = [rng.randint(10, 100) for _ in range(6)]
        workers = [index % 2 for index in range(6)]
        segments = [index // 2 for index in range(6)]
        facts = [
            {
                "workflow_id": f"helium-{trial}",
                "planned_worker_id": f"target-{workers[index]}",
                "worker_id": f"target-{workers[index]}",
                "segment_index": segments[index],
                "sequence_path": f"{workers[index]}/{segments[index]}",
                "dependency_ready": segments[index] == 0,
                "precedence_ready_at": segments[index],
                "now_token_step": 0,
                "critical_path_depth": 3 - segments[index],
                "earliest_start": segments[index],
            }
            for index in range(6)
        ]
        refs, lifecycle = bootstrap_requests(
            bench, f"helium-{trial}", facts, status="admitted"
        )
        assignments, _ = invoke_route(
            bench,
            refs,
            [
                {
                    "planned_worker_id": f"target-{workers[index]}"
                }
                for index in range(6)
            ],
            [{}, {}],
            [
                [
                    {
                        "planned_rank": (
                            0 if target == workers[index] else 1
                        ),
                        "estimated_token_steps": costs[index],
                    }
                    for target in range(2)
                ]
                for index in range(6)
            ],
            lifecycle,
            target_capacity=3,
        )
        worker_load = [0, 0]
        for index, target in enumerate(assignments):
            worker_load[target] += costs[index]
        policy_values.append(sum(costs) / max(worker_load))
        baseline_values.append(1.0)
        invoke_schedule(
            bench,
            refs,
            facts,
            [
                {
                    "event": "activate-request",
                    "request_id": ref["request_id"],
                }
                for ref in refs
            ],
            max_selections=2,
            max_requests=2,
        )
    return result(
        policy_values,
        baseline_values,
        direction="higher",
        unit="worker_parallelism_speedup",
    )


def scenario_vtc(
    bench: PolicyBench, _rng: random.Random, trials: int
) -> dict[str, Any]:
    clients = 4
    output_tokens = [1, 2, 4, 8]
    fair_weights = [1_000_000, 2_000_000, 4_000_000, 8_000_000]
    base_facts = [
        {
            "client_id": f"client-{index}",
            "queue_member": True,
            "dispatch_input_tokens": 0,
            "input_weight": 1,
            "input_price": 1,
            "fair_weight_ppm": fair_weights[index],
        }
        for index in range(clients)
    ]
    refs, lifecycle = bootstrap_requests(
        bench, "vtc", base_facts, status="active"
    )
    policy_service = [0.0] * clients
    baseline_service = [0.0] * clients
    policy_spans = []
    baseline_spans = []
    previous_active: set[int] = set()
    horizon = max(trials, 16) * clients
    for step in range(horizon):
        active = [0, 1]
        if step % 8 < 4:
            active.append(2)
        if step >= horizon // 2 and step % 6 < 3:
            active.append(3)
        facts = [
            {
                **base_facts[index],
                "client_became_active": index not in previous_active,
            }
            for index in active
        ]
        selected, schedule_outcome = invoke_schedule(
            bench,
            [refs[index] for index in active],
            facts,
            lifecycle if step == 0 else None,
        )
        if len(selected) != 1:
            raise RuntimeError("VTC did not remain work-conserving")
        selected_client = active[selected[0]]
        enact_schedule(bench, schedule_outcome)
        policy_service[selected_client] += 1
        baseline_service[active[0]] += 1
        bench.feedback(
            [
                {
                    "subject": {
                        "kind": "request",
                        "value": refs[selected_client]["request_id"],
                    },
                    "outcome": "progress",
                    "facts": {
                        "client_id": f"client-{selected_client}",
                        "input_tokens": 0,
                        "output_tokens": output_tokens[selected_client],
                        "output_weight": 1,
                        "output_price": 2,
                        "fair_weight_ppm": fair_weights[selected_client],
                    },
                }
            ]
        )
        policy_spans.append(abs(policy_service[0] - policy_service[1]))
        baseline_spans.append(abs(baseline_service[0] - baseline_service[1]))
        previous_active = set(active)
    return result(
        [max(policy_spans)],
        [max(baseline_spans)],
        direction="lower",
        unit="max_backlogged_service_span",
        details={
            "policy_service": policy_service,
            "baseline_service": baseline_service,
            "policy_jain": jain(policy_service[:2]),
            "baseline_jain": jain(baseline_service[:2]),
            "on_off_client": 2,
            "distribution_shift_client": 3,
        },
    )


def scenario_lmetric(
    bench: PolicyBench, rng: random.Random, trials: int
) -> dict[str, Any]:
    policy_costs = []
    baseline_costs = []
    for trial in range(trials):
        facts = [{"request_class": "hot", "window_id": f"window-{trial}"}] * 2
        refs, lifecycle = bootstrap_requests(
            bench, f"lmetric-{trial}", facts, status="admitted"
        )
        edge_facts = [
            {
                "cache_hit": True,
                "new_prefill_tokens": 1,
                "current_batch_size": rng.randint(0, 2),
            },
            {
                "cache_hit": False,
                "new_prefill_tokens": 10,
                "current_batch_size": rng.randint(0, 2),
            },
        ]
        invoke_route(
            bench,
            [refs[0]],
            [facts[0]],
            [{}, {}],
            [[edge_facts[0], edge_facts[1]]],
            lifecycle,
        )
        assignments, _ = invoke_route(
            bench,
            [refs[1]],
            [facts[1]],
            [{}, {}],
            [[edge_facts[0], edge_facts[1]]],
            [],
        )
        realized = [
            1000
            + edge_facts[0]["new_prefill_tokens"]
            * (edge_facts[0]["current_batch_size"] + 1),
            edge_facts[1]["new_prefill_tokens"]
            * (edge_facts[1]["current_batch_size"] + 1),
        ]
        policy_costs.append(realized[assignments[0]])
        baseline_costs.append(realized[0])
    return result(
        policy_costs,
        baseline_costs,
        direction="lower",
        unit="prefill_batch_product",
    )


def scenario_fairserve(
    bench: PolicyBench, _rng: random.Random, trials: int
) -> dict[str, Any]:
    weights = [1, 2, 4, 8]
    requests_per_user = min(max(trials, 16), 64)
    facts = []
    group_keys = []
    user_indices = []
    for user_index, weight in enumerate(weights):
        for request_index in range(requests_per_user):
            facts.append(
                {
                    "user_id": f"client-{user_index}",
                    "client_id": f"client-{user_index}",
                    "application_id": f"application-{user_index}",
                    "stage_id": "stage-1",
                    "expected_input_tokens": 0,
                    "expected_system_tokens": 0,
                    "expected_output_tokens": weight,
                    "input_weight": 1,
                    "system_weight": 2,
                    "output_weight": 1,
                    "user_priority_ppm": 1_000_000,
                    "kv_overloaded": True,
                    "interaction_in_progress": False,
                    "user_rpm_remaining": 1,
                    "app_rpm_remaining": 1,
                    "arrival_seq": request_index,
                }
            )
            group_keys.append(str(user_index))
            user_indices.append(user_index)
    refs, lifecycle = bootstrap_requests(
        bench,
        "fairserve",
        facts,
        status="active",
        group_keys=group_keys,
    )
    admit_facts = [
        {
            "user_id": f"admit-user-{index}",
            "client_id": f"admit-user-{index}",
            "application_id": "admit-app",
            "stage_id": "stage-1",
            "now_ms": index,
            "rpm_window_ms": 60_000,
            "user_rpm_limit": 1,
            "app_rpm_limit": len(weights),
            "kv_overloaded": True,
            "interaction_in_progress": index >= 2,
        }
        for index in range(len(weights))
    ]
    admit_refs, admit_lifecycle = bootstrap_requests(
        bench, "fairserve-admit", admit_facts, status="pending"
    )
    invoke_admit(
        bench,
        admit_refs,
        admit_facts,
        admit_lifecycle,
        max_accepted=len(facts),
    )
    policy_service = [0.0] * len(weights)
    baseline_service = [0.0] * len(weights)
    steps = requests_per_user * 2
    for step in range(steps):
        selected, _ = invoke_schedule(
            bench, refs, facts, lifecycle if step == 0 else None
        )
        selected_index = selected[0]
        user_index = user_indices[selected_index]
        policy_service[user_index] += 1
        baseline_service[step % len(weights)] += 1
        bench.feedback(
            [
                {
                    "subject": {
                        "kind": "request",
                        "value": refs[selected_index]["request_id"],
                    },
                    "outcome": "progress",
                    "facts": {"output_tokens": 1},
                },
                {
                    "subject": {
                        "kind": "request",
                        "value": refs[selected_index]["request_id"],
                    },
                    "outcome": "completed",
                    "facts": {
                        key: facts[selected_index][key]
                        for key in (
                            "user_id",
                            "client_id",
                            "application_id",
                            "stage_id",
                            "expected_input_tokens",
                            "expected_system_tokens",
                            "expected_output_tokens",
                            "input_weight",
                            "system_weight",
                            "output_weight",
                            "user_priority_ppm",
                        )
                    },
                },
            ]
        )
        del refs[selected_index]
        del facts[selected_index]
        del user_indices[selected_index]
    policy_normalized = [
        service / weight
        for service, weight in zip(policy_service, weights)
    ]
    baseline_normalized = [
        service / weight
        for service, weight in zip(baseline_service, weights)
    ]
    return result(
        [jain(policy_normalized)],
        [jain(baseline_normalized)],
        direction="higher",
        unit="weighted_jain_fairness",
        details={
            "weights": weights,
            "policy_service": policy_service,
            "baseline_service": baseline_service,
        },
    )


def scenario_marconi(
    bench: PolicyBench, rng: random.Random, trials: int
) -> dict[str, Any]:
    policy_values = []
    baseline_values = []
    for trial in range(trials):
        timestamps = rng.sample(range(1, 1000), 8)
        flops = [rng.randint(1, 10000) for _ in range(8)]
        alpha = 1_000_000
        residents = [
            cache_object(
                f"marconi-{trial}-{index}",
                {
                    "child_count": 0,
                    "last_access_us": timestamps[index],
                    "recompute_flops": flops[index],
                },
            )
            for index in range(8)
        ]
        plan, _ = invoke_cache(
            bench,
            residents,
            [],
            None,
            max_bytes=4,
            capacity_facts={"alpha_ppm": alpha},
        )
        reclaimed = set(plan["reclaim"])
        t_min, t_max = min(timestamps), max(timestamps)
        f_min, f_max = min(flops), max(flops)
        values = [
            (timestamps[index] - t_min) / max(t_max - t_min, 1)
            + (flops[index] - f_min) / max(f_max - f_min, 1)
            for index in range(8)
        ]
        policy_values.append(
            sum(value for index, value in enumerate(values) if index not in reclaimed)
        )
        lru_retained = sorted(range(8), key=timestamps.__getitem__)[-4:]
        baseline_values.append(sum(values[index] for index in lru_retained))
    exercise_feedback(bench, "marconi-feedback")
    return result(
        policy_values,
        baseline_values,
        direction="higher",
        unit="retained_recompute_value",
    )


def scenario_ragcache(
    bench: PolicyBench, rng: random.Random, trials: int
) -> dict[str, Any]:
    policy_costs = []
    baseline_costs = []
    for trial in range(trials):
        scores = []
        residents = []
        for index in range(6):
            frequency = rng.randint(1, 20)
            recompute = rng.randint(10, 1000)
            size = rng.randint(1, 8)
            unit_cost = max(recompute // size, 1)
            scores.append(unit_cost * frequency)
            residents.append(
                cache_object(
                    f"ragcache-{trial}-{index}",
                    {
                        "tier": "gpu",
                        "tier_child_count": 0,
                        "frequency": frequency,
                        "average_cost_per_new_token_fp": unit_cost,
                        "host_copy_exists": True,
                    },
                    size_bytes=size,
                )
            )
        plan, _ = invoke_cache(
            bench,
            residents,
            [],
            None,
            max_bytes=sum(item["size_bytes"] for item in residents) - 1,
        )
        policy_costs.append(scores[plan["reclaim"][0]])
        baseline_costs.append(scores[0])
    return result(
        policy_costs,
        baseline_costs,
        direction="lower",
        unit="evicted_gdsf_value",
    )


def scenario_dlpm(
    bench: PolicyBench, rng: random.Random, trials: int
) -> dict[str, Any]:
    policy_costs = []
    baseline_costs = []
    for trial in range(trials):
        client = f"client-{trial}"
        refs, lifecycle = bootstrap_requests(
            bench,
            f"dlpm-{trial}",
            [{"client_id": client}],
            status="admitted",
        )
        cached = [rng.randint(0, 4095) for _ in range(4)]
        for index in rng.sample(range(4), 2):
            cached[index] = 4096
        load = [rng.randint(1, 100) for _ in range(4)]
        longest = max(cached)
        assignments, _ = invoke_route(
            bench,
            refs,
            [
                {
                    "client_id": client,
                    "input_tokens": 1,
                    "input_weight": 1,
                }
            ],
            [{"worker_quantum": 100} for _ in range(4)],
            [
                [
                    {
                        "cached_tokens": cached[index],
                        "load": load[index],
                        "queue_size": load[index],
                        "longest_prefix_match": cached[index] == longest,
                    }
                    for index in range(4)
                ]
            ],
            lifecycle,
        )
        costs = [
            (longest - cached[index]) * 10 + load[index]
            for index in range(4)
        ]
        policy_costs.append(costs[assignments[0]])
        baseline_index = max(range(4), key=cached.__getitem__)
        baseline_costs.append(costs[baseline_index])
        _, schedule_outcome = invoke_schedule(
            bench,
            refs,
            [
                {
                    "client_id": client,
                    "queue_member": True,
                    "cached_tokens": longest,
                    "client_quantum": 100,
                    "extend_tokens": 1,
                    "extend_weight": 1,
                }
            ],
            [
                {
                    "event": "activate-request",
                    "request_id": refs[0]["request_id"],
                }
            ],
        )
        enact_schedule(bench, schedule_outcome)
        bench.feedback(
            [
                {
                    "subject": {
                        "kind": "request",
                        "value": refs[0]["request_id"],
                    },
                    "outcome": "progress",
                    "facts": {
                        "client_id": client,
                        "output_tokens": 1,
                        "output_weight": 1,
                    },
                }
            ]
        )
    return result(
        policy_costs,
        baseline_costs,
        direction="lower",
        unit="deficit_locality_cost",
    )


def scenario_infercept(
    bench: PolicyBench, rng: random.Random, trials: int
) -> dict[str, Any]:
    policy_values = []
    baseline_values = []
    for trial in range(trials):
        facts = [
            {
                "resuming": rng.random() < 0.5,
                "expected_waste_tokens": rng.randint(0, 1000),
            }
            for _ in range(5)
        ]
        refs, lifecycle = bootstrap_requests(
            bench, f"infercept-{trial}", facts, status="active"
        )
        selected, _ = invoke_schedule(bench, refs, facts, lifecycle)
        schedule_value = [
            (10000 if item["resuming"] else 0)
            + 1001
            - item["expected_waste_tokens"]
            for item in facts
        ]
        prospective = []
        saved = []
        for index in range(5):
            reuse = rng.randint(1, 1000)
            recompute = rng.randint(1, 1000)
            saved.append(recompute if reuse < recompute else 0)
            prospective.append(
                cache_object(
                    f"infercept-p-{trial}-{index}",
                    {
                        "expected_reuse_ms": reuse,
                        "recompute_ms": recompute,
                    },
                )
            )
        plan, _ = invoke_cache(
            bench, [], prospective, None, max_bytes=len(prospective)
        )
        policy_saved = sum(
            value
            for value, decision in zip(saved, plan["admissions"])
            if decision == "cache"
        )
        policy_values.append(schedule_value[selected[0]] + policy_saved)
        baseline_values.append(schedule_value[0])
    return result(
        policy_values,
        baseline_values,
        direction="higher",
        unit="resume_and_recompute_value",
    )


def scenario_peek(
    bench: PolicyBench, rng: random.Random, trials: int
) -> dict[str, Any]:
    policy_values = []
    baseline_values = []
    for trial in range(trials):
        facts = [
            {
                "lpm_hit_tokens": 0,
                "warm_threshold_tokens": 32,
                "deprio_prefix_id": "singleton",
                "cluster_id": "singleton",
                "cluster_size": 1,
                "ancestor_score": 0,
                "root_child_pending_count": 1,
                "arrival_seq": 0,
                "waiting_ms": 2000,
            },
            {
                "lpm_hit_tokens": 128,
                "warm_threshold_tokens": 32,
                "cluster_id": "warm",
                "cluster_size": 3,
                "ancestor_score": 100,
                "root_child_pending_count": 3,
                "arrival_seq": 1,
                "waiting_ms": 0,
            },
            {
                "lpm_hit_tokens": 0,
                "warm_threshold_tokens": 32,
                "deprio_prefix_id": "cold-a",
                "cluster_id": "cold-a",
                "cluster_size": 2,
                "ancestor_score": 80,
                "root_child_pending_count": 2,
                "arrival_seq": 2,
                "waiting_ms": 0,
            },
            {
                "lpm_hit_tokens": 0,
                "warm_threshold_tokens": 32,
                "deprio_prefix_id": "cold-a",
                "cluster_id": "cold-a",
                "cluster_size": 2,
                "ancestor_score": 80,
                "root_child_pending_count": 2,
                "arrival_seq": 3,
                "waiting_ms": 0,
            },
        ]
        refs, lifecycle = bootstrap_requests(
            bench, f"peek-{trial}", facts, status="active"
        )
        selected, _ = invoke_schedule(bench, refs, facts, lifecycle)
        utilities = [
            item["lpm_hit_tokens"]
            + item["ancestor_score"]
            + item["cluster_size"] * 10
            for item in facts
        ]
        pending_depth = [10, 0, 8, 8]
        residents = [
            cache_object(
                f"peek-{trial}-{index}",
                {
                    "ancestor_demands": [
                        {"pending_count": depth, "depth": 1}
                    ]
                },
                beneficiary=ref["request_id"],
            )
            for index, (ref, depth) in enumerate(zip(refs, pending_depth))
        ]
        plan, _ = invoke_cache(
            bench, residents, [], None, max_bytes=len(residents) - 1
        )
        policy_values.append(
            utilities[selected[0]]
            + (max(pending_depth) - pending_depth[plan["reclaim"][0]])
        )
        baseline_values.append(utilities[0])
    return result(
        policy_values,
        baseline_values,
        direction="higher",
        unit="queue_cache_priority",
    )


def scenario_qlm(
    bench: PolicyBench, rng: random.Random, trials: int
) -> dict[str, Any]:
    policy_values = []
    baseline_values = []
    for trial in range(trials):
        facts = [
            {
                "estimated_wait_ms": rng.randint(10, 500),
                "slo_ms": rng.randint(100, 350),
            }
            for _ in range(8)
        ]
        refs, lifecycle = bootstrap_requests(
            bench, f"qlm-{trial}", facts, status="pending"
        )
        decisions = invoke_admit(
            bench, refs, facts, lifecycle, max_accepted=len(facts)
        )
        feasible = [
            item["estimated_wait_ms"] <= item["slo_ms"] for item in facts
        ]
        policy_values.append(
            len(facts)
            + sum(
                1 if ok else -1
                for ok, decision in zip(feasible, decisions)
                if decision == "accept"
            )
        )
        baseline_values.append(
            len(facts) + sum(1 if ok else -1 for ok in feasible[:4])
        )
    route_refs, route_lifecycle = bootstrap_requests(
        bench, "qlm-route", [{}], status="admitted"
    )
    invoke_route(
        bench,
        route_refs,
        [{}],
        [{}, {}],
        [[{"estimated_wait_ms": 100}, {"estimated_wait_ms": 10}]],
        route_lifecycle,
    )
    schedule_refs, schedule_lifecycle = bootstrap_requests(
        bench,
        "qlm-schedule",
        [{"virtual_wait": 100}, {"virtual_wait": 0}],
        status="active",
        group_keys=["slow", "fast"],
    )
    invoke_schedule(
        bench,
        schedule_refs,
        [{"virtual_wait": 100}, {"virtual_wait": 0}],
        schedule_lifecycle,
    )
    exercise_feedback(bench, "qlm-feedback")
    return result(
        policy_values,
        baseline_values,
        direction="higher",
        unit="slo_goodput",
    )


def scenario_slos_serve(
    bench: PolicyBench, rng: random.Random, trials: int
) -> dict[str, Any]:
    policy_values = []
    baseline_values = []
    for trial in range(trials):
        facts = [
            {
                "predicted_total_ms": rng.randint(50, 600),
                "slo_ms": rng.randint(150, 450),
                "slack_ms": rng.randint(-100, 300),
            }
            for _ in range(8)
        ]
        refs, lifecycle = bootstrap_requests(
            bench, f"slos-{trial}", facts, status="pending"
        )
        decisions = invoke_admit(
            bench, refs, facts, lifecycle, max_accepted=len(facts)
        )
        feasible = [
            item["predicted_total_ms"] <= item["slo_ms"] for item in facts
        ]
        policy_values.append(
            len(facts)
            + sum(
                1 if ok else -1
                for ok, decision in zip(feasible, decisions)
                if decision == "accept"
            )
        )
        baseline_values.append(
            len(facts) + sum(1 if ok else -1 for ok in feasible[:4])
        )
    route_refs, route_lifecycle = bootstrap_requests(
        bench, "slos-route", [{}], status="admitted"
    )
    invoke_route(
        bench,
        route_refs,
        [{}],
        [{}, {}],
        [[{"stage_latency_ms": 100}, {"stage_latency_ms": 10}]],
        route_lifecycle,
    )
    schedule_refs, schedule_lifecycle = bootstrap_requests(
        bench,
        "slos-schedule",
        [{"slack_ms": 100}, {"slack_ms": 0}],
        status="active",
    )
    invoke_schedule(
        bench,
        schedule_refs,
        [{"slack_ms": 100}, {"slack_ms": 0}],
        schedule_lifecycle,
    )
    return result(
        policy_values,
        baseline_values,
        direction="higher",
        unit="multi_slo_goodput",
    )


def scenario_dynasor(
    bench: PolicyBench, rng: random.Random, trials: int
) -> dict[str, Any]:
    policy_values = []
    baseline_values = []
    for trial in range(trials):
        facts = [
            {
                "confidence_ppm": rng.randint(0, 1_000_000),
                "stop_threshold_ppm": 750_000,
                "progress_ppm": rng.randint(0, 1_000_000),
            }
            for _ in range(8)
        ]
        refs, lifecycle = bootstrap_requests(
            bench, f"dynasor-{trial}", facts, status="active"
        )
        selected, outcome = invoke_schedule(
            bench,
            refs,
            facts,
            lifecycle,
            max_selections=4,
        )
        useful = [
            item["confidence_ppm"] < item["stop_threshold_ppm"]
            for item in facts
        ]
        policy_useful = sum(useful[index] for index in selected)
        policy_compute = max(len(selected), 1)
        baseline_useful = sum(useful[:4])
        baseline_compute = 4
        cancelled = sum(
            action["method"] == "pie.request.cancel@1"
            for action in outcome["actions"]
        )
        policy_values.append(policy_useful / policy_compute)
        baseline_values.append(baseline_useful / baseline_compute)
        if cancelled != sum(not value for value in useful):
            raise RuntimeError("Dynasor cancellation count diverged")
    exercise_feedback(bench, "dynasor-feedback")
    return result(
        policy_values,
        baseline_values,
        direction="higher",
        unit="useful_work_fraction",
    )


def scenario_justitia(
    bench: PolicyBench, rng: random.Random, trials: int
) -> dict[str, Any]:
    policy_costs = []
    baseline_costs = []
    for trial in range(trials):
        costs = [rng.randint(50, 500) for _ in range(4)]
        now_us = trial * 1000
        facts = [
            {
                "ready": True,
                "now_us": now_us,
                "total_kv_tokens": 1000,
                "predicted_agent_kv_token_time": costs[group_index],
            }
            for group_index in range(4)
            for _ in range(2)
        ]
        group_keys = [
            str(group_index) for group_index in range(4) for _ in range(2)
        ]
        refs, lifecycle = bootstrap_requests(
            bench,
            f"justitia-{trial}",
            facts,
            status="active",
            group_keys=group_keys,
        )
        selected, schedule_outcome = invoke_schedule(
            bench,
            refs,
            facts,
            lifecycle,
            max_selections=1,
            max_requests=2,
        )
        selected_group = selected[0] // 2
        if any(index // 2 != selected_group for index in selected):
            raise RuntimeError("Justitia selected branches from multiple agents")
        policy_costs.append(costs[selected_group])
        baseline_costs.append(costs[0])
        enact_schedule(bench, schedule_outcome)
        group_ids = list(dict.fromkeys(ref["group_id"] for ref in refs))
        bench.feedback(
            [
                {
                    "subject": {"kind": "work-group", "value": group_id},
                    "outcome": "completed",
                    "facts": {
                        "now_us": now_us + 1,
                        "total_kv_tokens": 1000,
                    },
                }
                for group_id in group_ids
            ]
        )
    return result(
        policy_costs,
        baseline_costs,
        direction="lower",
        unit="predicted_kv_token_time",
    )


def scenario_chameleon(
    bench: PolicyBench, rng: random.Random, trials: int
) -> dict[str, Any]:
    policy_values = []
    baseline_values = []
    for trial in range(trials):
        hot = [rng.random() < 0.5 for _ in range(8)]
        facts = [
            {
                "weighted_size": rng.randint(1, 10),
                "queue_quota": rng.randint(4, 10),
                "queue_class": rng.randint(0, 3),
                "waiting_ms": rng.randint(0, 2000),
                "adapter_hot": value,
            }
            for value in hot
        ]
        refs, lifecycle = bootstrap_requests(
            bench, f"chameleon-{trial}", facts, status="pending"
        )
        decisions = invoke_admit(
            bench, refs, facts, lifecycle, max_accepted=4
        )
        prospective = [
            cache_object(
                f"chameleon-{trial}-{index}",
                {"adapter_hot": value},
            )
            for index, value in enumerate(hot)
        ]
        plan, _ = invoke_cache(
            bench, [], prospective, None, max_bytes=len(prospective)
        )
        policy_values.append(
            2 * len(facts)
            + sum(
                1 if fact["weighted_size"] <= fact["queue_quota"] else -1
                for fact, decision in zip(facts, decisions)
                if decision == "accept"
            )
            + sum(
                1 if is_hot else -1
                for is_hot, decision in zip(hot, plan["admissions"])
                if decision == "cache"
            )
        )
        baseline_values.append(
            2 * len(facts)
            + sum(
                1 if fact["weighted_size"] <= fact["queue_quota"] else -1
                for fact in facts[:4]
            )
            + sum(1 if value else -1 for value in hot[:4])
        )
        schedule_refs, schedule_lifecycle = bootstrap_requests(
            bench,
            f"chameleon-schedule-{trial}",
            facts[:4],
            status="active",
        )
        invoke_schedule(
            bench,
            schedule_refs,
            facts[:4],
            schedule_lifecycle,
        )
    return result(
        policy_values,
        baseline_values,
        direction="higher",
        unit="adapter_goodput_score",
    )


def scenario_hotprefix(
    bench: PolicyBench, rng: random.Random, trials: int
) -> dict[str, Any]:
    policy_values = []
    baseline_values = []
    for trial in range(trials):
        facts = [{} for _ in range(6)]
        refs, lifecycle = bootstrap_requests(
            bench, f"hotprefix-{trial}", facts, status="active"
        )
        hotness = [rng.randint(0, 20) for _ in refs]
        object_ids = [f"hotprefix-{trial}-{index}" for index in range(6)]
        bench.feedback(
            [
                {
                    "subject": {
                        "kind": "cache-object",
                        "value": object_id,
                    },
                    "outcome": "progress",
                    "facts": {"reuse_count": value},
                }
                for object_id, value in zip(object_ids, hotness)
            ],
            lifecycle,
        )
        prospective = [
            cache_object(
                object_id,
                {"hotness": value},
                beneficiary=ref["request_id"],
            )
            for object_id, value, ref in zip(object_ids, hotness, refs)
        ]
        plan, _ = invoke_cache(
            bench,
            [],
            prospective,
            None,
            max_bytes=len(prospective),
            capacity_facts={"hot_threshold": 10},
        )
        policy_values.append(
            sum(
                value
                for value, decision in zip(hotness, plan["admissions"])
                if decision == "cache"
            )
        )
        baseline_values.append(
            sum(value for value in hotness[:3] if value >= 10)
        )
    return result(
        policy_values,
        baseline_values,
        direction="higher",
        unit="retained_hotness",
    )


def scenario_pard(
    bench: PolicyBench, rng: random.Random, trials: int
) -> dict[str, Any]:
    policy_values = []
    baseline_values = []
    for trial in range(trials):
        high_load = trial % 2 == 0
        budgets = [10, 20, 100, 200, 300, 400]
        if not high_load:
            budgets.reverse()
        facts = [
            {
                "upstream_elapsed_ms": 100,
                "current_queue_ms": 0,
                "current_execution_ms": 0,
                "downstream_queue_ms": 0,
                "downstream_execution_ms": 0,
                "downstream_batch_wait_p10_ms": 0,
                "deadline_ms": 100 + budget,
                "cancel_supported": True,
            }
            for budget in budgets
        ] + [
            {
                "upstream_elapsed_ms": 100,
                "current_queue_ms": 0,
                "current_execution_ms": 0,
                "downstream_queue_ms": 0,
                "downstream_execution_ms": 0,
                "downstream_batch_wait_p10_ms": 0,
                "deadline_ms": 50,
                "cancel_supported": True,
            }
            for _ in range(2)
        ]
        refs, lifecycle = bootstrap_requests(
            bench, f"pard-{trial}", facts, status="active"
        )
        selected, _ = invoke_schedule(
            bench,
            refs,
            facts,
            lifecycle,
            max_selections=4,
            capacity_facts={
                "load_factor_ppm": (
                    1_300_000 if high_load else 700_000
                ),
                "hysteresis_epsilon_ppm": 100_000,
            },
        )
        feasible = [
            item["upstream_elapsed_ms"]
            + item["current_queue_ms"]
            + item["current_execution_ms"]
            + item["downstream_queue_ms"]
            + item["downstream_execution_ms"]
            + item["downstream_batch_wait_p10_ms"]
            <= item["deadline_ms"]
            for item in facts
        ]
        remaining = [
            max(
                item["deadline_ms"]
                - (
                    item["upstream_elapsed_ms"]
                    + item["current_queue_ms"]
                    + item["current_execution_ms"]
                    + item["downstream_queue_ms"]
                    + item["downstream_execution_ms"]
                    + item["downstream_batch_wait_p10_ms"]
                ),
                0,
            )
            for item in facts
        ]
        desired = sorted(
            (index for index in range(len(facts)) if feasible[index]),
            key=remaining.__getitem__,
            reverse=high_load,
        )[:4]
        policy_values.append(
            sum(feasible[index] for index in selected)
            + sum(index in desired for index in selected)
        )
        baseline_values.append(
            sum(feasible[:4]) + sum(index in desired for index in range(4))
        )
    exercise_feedback(bench, "pard-feedback")
    return result(
        policy_values,
        baseline_values,
        direction="higher",
        unit="deadline_goodput",
    )


def scenario_branch_regulation(
    bench: PolicyBench, rng: random.Random, trials: int
) -> dict[str, Any]:
    policy_values = []
    baseline_values = []
    for trial in range(trials):
        facts = [
            {
                "branch_limit": 2,
                "batch_interference": rng.randint(0, 10),
                "interference_limit": 5,
                "excess_branch": index % 4 >= 2,
            }
            for index in range(8)
        ]
        refs, lifecycle = bootstrap_requests(
            bench,
            f"branch-{trial}",
            facts,
            status="pending",
            group_keys=["a"] * 4 + ["b"] * 4,
        )
        decisions = invoke_admit(
            bench, refs, facts, lifecycle, max_accepted=8
        )
        policy_accepted = [
            index
            for index, decision in enumerate(decisions)
            if decision == "accept"
        ]
        policy_values.append(
            len(facts)
            + sum(facts[index]["batch_interference"] <= 5 for index in policy_accepted)
            - max(len(policy_accepted) - 4, 0)
        )
        baseline_values.append(
            len(facts)
            + sum(item["batch_interference"] <= 5 for item in facts)
            - 4
        )
        schedule_refs, schedule_lifecycle = bootstrap_requests(
            bench,
            f"branch-schedule-{trial}",
            [{"excess_branch": False}, {"excess_branch": True}],
            status="active",
        )
        invoke_schedule(
            bench,
            schedule_refs,
            [{"excess_branch": False}, {"excess_branch": True}],
            schedule_lifecycle,
        )
    return result(
        policy_values,
        baseline_values,
        direction="higher",
        unit="regulated_goodput",
    )


def scenario_dualmap(
    bench: PolicyBench, rng: random.Random, trials: int
) -> dict[str, Any]:
    policy_costs = []
    baseline_costs = []
    for trial in range(trials):
        prefix_hit = [rng.randint(0, 4096) for _ in range(4)]
        prefix_hit[rng.randrange(4)] = 4096
        ttft = [rng.randint(10, 200) for _ in range(4)]
        hash_choice = [rng.randint(0, 100) for _ in range(4)]
        slo_ms = rng.randint(50, 150)
        refs, lifecycle = bootstrap_requests(
            bench,
            f"dualmap-{trial}",
            [{"hotspot": rng.random() < 0.2, "slo_ms": slo_ms}],
            status="admitted",
        )
        assignments, _ = invoke_route(
            bench,
            refs,
            [{"hotspot": False, "slo_ms": slo_ms}],
            [{}, {}, {}, {}],
            [
                [
                    {
                        "hash_candidate": True,
                        "hash_choice": hash_choice[index],
                        "prefix_hit_tokens": prefix_hit[index],
                        "predicted_ttft_ms": ttft[index],
                    }
                    for index in range(4)
                ]
            ],
            lifecycle,
        )
        costs = [
            (max(prefix_hit) - prefix_hit[index]) * 10 + ttft[index]
            for index in range(4)
        ]
        policy_costs.append(costs[assignments[0]])
        baseline_index = max(range(4), key=hash_choice.__getitem__)
        baseline_costs.append(costs[baseline_index])
    return result(
        policy_costs,
        baseline_costs,
        direction="lower",
        unit="affinity_slo_cost",
    )


def scenario_llumnix(
    bench: PolicyBench, rng: random.Random, trials: int
) -> dict[str, Any]:
    policy_costs = []
    baseline_costs = []
    for trial in range(trials):
        usage = [rng.randint(1, 1000) for _ in range(4)]
        memory = [rng.randint(1000, 2000) for _ in range(4)]
        batch = [rng.randint(1, 16) for _ in range(4)]
        refs, lifecycle = bootstrap_requests(
            bench,
            f"llumnix-{trial}",
            [{"live_reschedule": False}],
            status="admitted",
        )
        assignments, _ = invoke_route(
            bench,
            refs,
            [{"live_reschedule": False}],
            [
                {
                    "memory_capacity": memory[index],
                    "batch_size": batch[index],
                }
                for index in range(4)
            ],
            [
                [
                    {"virtual_usage": value}
                    for value in usage
                ]
            ],
            lifecycle,
        )
        freeness = [
            max(memory[index] - usage[index], 0) / batch[index]
            for index in range(4)
        ]
        maximum = max(freeness)
        costs = [maximum - value + 1 for value in freeness]
        policy_costs.append(costs[assignments[0]])
        baseline_costs.append(costs[0])
    exercise_feedback(bench, "llumnix-feedback")
    return result(
        policy_costs,
        baseline_costs,
        direction="lower",
        unit="virtual_usage",
    )


def scenario_smetric(
    bench: PolicyBench, rng: random.Random, trials: int
) -> dict[str, Any]:
    policy_costs = []
    baseline_costs = []
    for trial in range(trials):
        mode = trial % 4
        if mode == 0:
            turn = 0
            affinity = [100, 10, 10, 10]
            load = [20, 2, 3, 4]
            estimated = [100] * 4
            overload = 2_000_000
        elif mode == 1:
            turn = 1
            affinity = [100, 20, 10, 5]
            load = [4, 2, 3, 5]
            estimated = [100] * 4
            overload = 2_000_000
        elif mode == 2:
            turn = 2
            affinity = [100, 20, 10, 5]
            load = [20, 2, 3, 4]
            estimated = [100] * 4
            overload = 1_500_000
        else:
            turn = 3
            affinity = [50, 20, 10, 5]
            load = [4, 2, 3, 5]
            estimated = [100] * 4
            overload = 10_000_000
        refs, lifecycle = bootstrap_requests(
            bench,
            f"smetric-{trial}",
            [
                {
                    "session_turn": turn,
                    "overload_ppm": overload,
                    "hit_ratio_ppm": 900_000,
                }
            ],
            status="admitted",
        )
        assignments, _ = invoke_route(
            bench,
            refs,
            [
                {
                    "session_turn": turn,
                    "overload_ppm": overload,
                    "hit_ratio_ppm": 900_000,
                }
            ],
            [{}, {}, {}, {}],
            [
                [
                    {
                        "cache_affinity": affinity[index],
                        "load": load[index],
                        "estimated_history_hit": estimated[index],
                    }
                    for index in range(4)
                ]
            ],
            lifecycle,
        )
        costs = [
            load[index]
            + (
                (max(affinity) - affinity[index]) * 2
                if turn > 0 and mode == 1
                else 0
            )
            for index in range(4)
        ]
        if mode in {2, 3}:
            costs[0] += 100
        policy_costs.append(costs[assignments[0]])
        baseline_index = max(range(4), key=affinity.__getitem__)
        baseline_costs.append(costs[baseline_index])
    return result(
        policy_costs,
        baseline_costs,
        direction="lower",
        unit="session_affinity_load_cost",
    )


def scenario_thunderagent(
    bench: PolicyBench, rng: random.Random, trials: int
) -> dict[str, Any]:
    policy_values = []
    baseline_values = []
    for trial in range(trials):
        facts = [
            {
                "tool_ready": rng.random() < 0.65,
                "tool_failed": rng.random() < 0.15,
                "migrate_target": None,
            }
            for _ in range(8)
        ]
        refs, lifecycle = bootstrap_requests(
            bench, f"thunder-{trial}", facts, status="active"
        )
        selected, _ = invoke_schedule(
            bench,
            refs,
            facts,
            lifecycle,
            max_selections=4,
        )
        useful = [
            item["tool_ready"] and not item["tool_failed"] for item in facts
        ]
        policy_values.append(sum(useful[index] for index in selected))
        baseline_values.append(sum(useful[:4]))
        prospective = [
            cache_object(
                f"thunder-{trial}-{index}",
                {"program_live": value},
            )
            for index, value in enumerate(useful)
        ]
        invoke_cache(
            bench,
            [],
            prospective,
            None,
            max_bytes=len(prospective),
        )
    exercise_feedback(bench, "thunder-feedback")
    return result(
        policy_values,
        baseline_values,
        direction="higher",
        unit="tool_ready_goodput",
    )


def scenario_pythia(
    bench: PolicyBench, rng: random.Random, trials: int
) -> dict[str, Any]:
    policy_costs = []
    baseline_costs = []
    for trial in range(trials):
        lookahead = [rng.randint(1, 1000) for _ in range(4)]
        ranks = [rng.randint(0, 20) for _ in range(4)]
        refs, lifecycle = bootstrap_requests(
            bench,
            f"pythia-{trial}",
            [{"workflow_rank": value} for value in ranks],
            status="admitted",
        )
        assignments, _ = invoke_route(
            bench,
            refs[:1],
            [{}],
            [{}, {}, {}, {}],
            [
                [
                    {"lookahead_cost": value}
                    for value in lookahead
                ]
            ],
            lifecycle,
        )
        selected, _ = invoke_schedule(
            bench,
            refs,
            [{"workflow_rank": value} for value in ranks],
            [
                {
                    "event": "activate-request",
                    "request_id": ref["request_id"],
                }
                for ref in refs
            ],
        )
        policy_costs.append(
            lookahead[assignments[0]] + ranks[selected[0]]
        )
        baseline_costs.append(lookahead[0] + ranks[0])
        residents = [
            cache_object(
                f"pythia-{trial}-{index}",
                {"next_use_step": rank},
            )
            for index, rank in enumerate(ranks)
        ]
        invoke_cache(
            bench,
            residents,
            [],
            None,
            max_bytes=len(residents) - 1,
        )
    exercise_feedback(bench, "pythia-feedback")
    return result(
        policy_costs,
        baseline_costs,
        direction="lower",
        unit="lookahead_workflow_cost",
    )


def scenario_goodserve(
    bench: PolicyBench, rng: random.Random, trials: int
) -> dict[str, Any]:
    policy_costs = []
    baseline_costs = []
    for trial in range(trials):
        costs = [10, 20, 40, 80]
        input_tokens = rng.randint(128, 2048)
        output_tokens = rng.randint(16, 256)
        deadline = rng.randint(500, 3000)
        queue = [rng.randint(0, 500) for _ in range(4)]
        cached = [rng.randint(0, input_tokens) for _ in range(4)]
        prefill = [5, 4, 2, 1]
        decode = [10, 7, 4, 2]
        capability = [1, 2, 3, 4]
        request_facts = {
            "risk_ppm": 0,
            "migration_threshold_ppm": 1_000_000,
            "deadline_ms": deadline,
            "input_tokens": input_tokens,
            "predicted_output_tokens": output_tokens,
        }
        refs, lifecycle = bootstrap_requests(
            bench,
            f"goodserve-{trial}",
            [request_facts],
            status="admitted",
        )
        assignments, _ = invoke_route(
            bench,
            refs,
            [request_facts],
            [{}, {}, {}, {}],
            [
                [
                    {
                        "queue_ms": queue[index],
                        "cached_tokens": cached[index],
                        "prefill_ms_per_token": prefill[index],
                        "decode_ms_per_token": decode[index],
                        "capability_rank": capability[index],
                        "cost": costs[index],
                    }
                    for index in range(4)
                ]
            ],
            lifecycle,
        )
        latency = [
            queue[index]
            + prefill[index] * (input_tokens - cached[index])
            + decode[index] * output_tokens
            for index in range(4)
        ]
        objective = [
            (
                0
                if latency[index] <= deadline
                else 100000 + latency[index]
            )
            + costs[index]
            for index in range(4)
        ]
        policy_costs.append(objective[assignments[0]])
        baseline_index = min(range(4), key=costs.__getitem__)
        baseline_costs.append(objective[baseline_index])
    exercise_feedback(bench, "goodserve-feedback")
    return result(
        policy_costs,
        baseline_costs,
        direction="lower",
        unit="e2e_slo_cost",
    )


def scenario_conserve(
    bench: PolicyBench, rng: random.Random, trials: int
) -> dict[str, Any]:
    policy_costs = []
    baseline_costs = []
    for trial in range(trials):
        followup = trial % 2 == 1
        bound_target = 2 if followup else None
        active_kv = [0, rng.randint(1, 100), rng.randint(1, 100), rng.randint(1, 100)]
        load = [rng.randint(1, 50) for _ in range(4)]
        request_facts = {
            "bound_target_id": (
                f"target-{bound_target}" if bound_target is not None else None
            )
        }
        refs, lifecycle = bootstrap_requests(
            bench,
            f"conserve-{trial}",
            [request_facts],
            status="admitted",
            generations=[1 if followup else 0],
        )
        assignments, _ = invoke_route(
            bench,
            refs,
            [request_facts],
            [
                {"prefiller": index == 0}
                for index in range(4)
            ],
            [
                [
                    {
                        "active_kv_bytes": active_kv[index],
                        "load": load[index],
                    }
                    for index in range(4)
                ]
            ],
            lifecycle,
        )
        costs = (
            [
                (
                    active_kv[index]
                    if index == bound_target
                    else 100000
                )
                for index in range(4)
            ]
            if followup
            else [0 if index == 0 else 100000 for index in range(4)]
        )
        policy_costs.append(costs[assignments[0]])
        baseline_index = min(range(4), key=load.__getitem__)
        baseline_costs.append(costs[baseline_index])
    return result(
        policy_costs,
        baseline_costs,
        direction="lower",
        unit="conversation_route_cost",
    )


def scenario_parrot(
    bench: PolicyBench, rng: random.Random, trials: int
) -> dict[str, Any]:
    policy_values = []
    baseline_values = []
    for trial in range(trials):
        facts = [
            {
                "dependency_ready": rng.random() < 0.5,
                "dependency_distance": rng.randint(0, 20),
            }
            for _ in range(6)
        ]
        facts[rng.randrange(6)]["dependency_ready"] = True
        refs, lifecycle = bootstrap_requests(
            bench, f"parrot-{trial}", facts, status="active"
        )
        selected, _ = invoke_schedule(
            bench,
            refs,
            facts,
            lifecycle,
            max_selections=3,
        )
        utility = [
            1000 - item["dependency_distance"]
            if item["dependency_ready"]
            else 0
            for item in facts
        ]
        policy_values.append(sum(utility[index] for index in selected))
        baseline_values.append(sum(utility[:3]))
        route_refs, route_lifecycle = bootstrap_requests(
            bench,
            f"parrot-route-{trial}",
            [{}],
            status="admitted",
        )
        invoke_route(
            bench,
            route_refs,
            [{}],
            [{}, {}],
            [
                [
                    {"dependency_distance": 100},
                    {"dependency_distance": 1},
                ]
            ],
            route_lifecycle,
        )
    return result(
        policy_values,
        baseline_values,
        direction="higher",
        unit="dependency_ready_work",
    )


def scenario_saga(
    bench: PolicyBench, rng: random.Random, trials: int
) -> dict[str, Any]:
    policy_costs = []
    baseline_costs = []
    for trial in range(trials):
        load = [5 + rng.randint(0, 2), 1 + rng.randint(0, 1)]
        refs, lifecycle = bootstrap_requests(
            bench,
            f"saga-{trial}",
            [
                {
                    "session_id": "session-a",
                    "affinity_target_id": "target-0",
                },
                {},
            ],
            status="admitted",
            group_keys=["a", "b"],
        )
        assignments, _ = invoke_route(
            bench,
            refs[:1],
            [
                {
                    "session_id": "session-a",
                    "affinity_target_id": "target-0",
                    "affinity_load_threshold_ppm": 800_000,
                }
            ],
            [{}, {}],
            [
                [
                    {
                        "cached": index == 0,
                        "cache_locality": 100 if index == 0 else 0,
                        "load": load[index],
                        "load_ppm": 500_000 if index == 0 else 100_000,
                    }
                    for index in range(2)
                ]
            ],
            lifecycle,
        )
        selected, _ = invoke_schedule(
            bench,
            refs,
            [
                {
                    "tenant_id": "tenant-a",
                    "remaining_work_ms": 100,
                    "deadline_ms": 1000,
                    "now_ms": 0,
                },
                {
                    "tenant_id": "tenant-b",
                    "remaining_work_ms": 100,
                    "deadline_ms": 200,
                    "now_ms": 0,
                },
            ],
            [
                {
                    "event": "activate-request",
                    "request_id": ref["request_id"],
                }
                for ref in refs
            ],
            max_selections=2,
            max_requests=2,
            token_budget=8,
        )
        costs = [load[0], load[1] + 100]
        urgency_penalty = [500, 100]
        policy_costs.append(costs[assignments[0]] + urgency_penalty[selected[0]])
        baseline_index = min(range(2), key=load.__getitem__)
        baseline_costs.append(costs[baseline_index] + urgency_penalty[0])
        prospective = [
            cache_object(
                f"saga-{trial}",
                {"tool_latency_samples_ms": [1000]},
            )
        ]
        invoke_cache(
            bench,
            [],
            prospective,
            None,
            max_bytes=1,
            capacity_facts={"now_ms": 0, "used_kv_ppm": 900_000},
        )
    return result(
        policy_costs,
        baseline_costs,
        direction="lower",
        unit="workflow_locality_load_cost",
    )


def scenario_routebalance(
    bench: PolicyBench, rng: random.Random, trials: int
) -> dict[str, Any]:
    policy_values = []
    baseline_values = []
    for trial in range(trials):
        outputs = [rng.randint(16, 256) for _ in range(6)]
        quality = [
            [rng.randint(100_000, 1_000_000) for _ in range(3)]
            for _ in range(6)
        ]
        costs = [
            [rng.randint(1, 100) for _ in range(3)] for _ in range(6)
        ]
        tpot = [rng.randint(500, 3000) for _ in range(3)]
        pending = [rng.randint(0, 200) for _ in range(3)]
        batch_size = [rng.randint(1, 16) for _ in range(3)]
        request_facts = [
            {
                "predicted_output_tokens": output,
                "cost_budget": 100,
                "quality_weight_ppm": 333_334,
                "cost_weight_ppm": 333_333,
                "latency_weight_ppm": 333_333,
            }
            for output in outputs
        ]
        refs, lifecycle = bootstrap_requests(
            bench,
            f"routebalance-{trial}",
            request_facts,
            status="admitted",
        )
        assignments, _ = invoke_route(
            bench,
            refs,
            request_facts,
            [
                {
                    "pending_decode_tokens": pending[index],
                    "decode_batch_size": batch_size[index],
                    "tpot_us": tpot[index],
                }
                for index in range(3)
            ],
            [
                [
                    {
                        "quality_ppm": quality[request_index][target_index],
                        "cost": costs[request_index][target_index],
                        "predicted_output_tokens": outputs[request_index],
                    }
                    for target_index in range(3)
                ]
                for request_index in range(6)
            ],
            lifecycle,
            target_capacity=2,
        )
        order = sorted(range(6), key=outputs.__getitem__, reverse=True)

        def score_batch(chosen: list[int | None]) -> float:
            dead = pending.copy()
            total = 0.0
            for request_index in order:
                target = chosen[request_index]
                if target is None:
                    continue
                latency = [
                    tpot[index]
                    * (dead[index] / batch_size[index] + outputs[request_index])
                    for index in range(3)
                ]
                max_cost = max(costs[request_index])
                max_latency = max(latency)
                total += (
                    quality[request_index][target] / 1_000_000
                    + 1
                    - costs[request_index][target] / max(max_cost, 1)
                    + 1
                    - latency[target] / max(max_latency, 1)
                ) / 3
                dead[target] += outputs[request_index]
            return total

        policy_values.append(score_batch(assignments))
        remaining = [2, 2, 2]
        baseline_assignments: list[int | None] = [None] * 6
        baseline_dead = pending.copy()
        for request_index in order:
            target = min(
                (
                    target
                    for target, slots in enumerate(remaining)
                    if slots > 0
                ),
                key=lambda target: (
                    baseline_dead[target] / batch_size[target],
                    target,
                ),
            )
            remaining[target] -= 1
            baseline_assignments[request_index] = target
            baseline_dead[target] += outputs[request_index]
        baseline_values.append(score_batch(baseline_assignments))
    return result(
        policy_values,
        baseline_values,
        direction="higher",
        unit="normalized_quality_latency_cost",
    )


SCENARIOS: dict[
    str, Callable[[PolicyBench, random.Random, int], dict[str, Any]]
] = {
    "agentix": scenario_agentix,
    "continuum": scenario_continuum,
    "kvflow": scenario_kvflow,
    "preble": scenario_preble,
    "helium": scenario_helium,
    "vtc": scenario_vtc,
    "lmetric": scenario_lmetric,
    "fairserve": scenario_fairserve,
    "marconi": scenario_marconi,
    "ragcache": scenario_ragcache,
    "dlpm": scenario_dlpm,
    "infercept": scenario_infercept,
    "peek": scenario_peek,
    "qlm": scenario_qlm,
    "slos-serve": scenario_slos_serve,
    "dynasor": scenario_dynasor,
    "justitia": scenario_justitia,
    "chameleon": scenario_chameleon,
    "hotprefix": scenario_hotprefix,
    "pard": scenario_pard,
    "branch-regulation": scenario_branch_regulation,
    "dualmap": scenario_dualmap,
    "llumnix": scenario_llumnix,
    "smetric": scenario_smetric,
    "thunderagent": scenario_thunderagent,
    "pythia": scenario_pythia,
    "goodserve": scenario_goodserve,
    "conserve": scenario_conserve,
    "parrot": scenario_parrot,
    "saga": scenario_saga,
    "routebalance": scenario_routebalance,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packages", type=Path, default=DEFAULT_PACKAGES)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--policy", action="append")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = json.loads(args.targets.read_text())
    trials = args.trials or targets["methodology"]["default_trials"]
    seed = args.seed or targets["methodology"]["default_seed"]
    selected = set(args.policy or SCENARIOS)
    unknown = selected - SCENARIOS.keys()
    if unknown:
        raise SystemExit(f"unknown policies: {sorted(unknown)}")
    target_by_id = {
        policy["id"]: policy for policy in targets["policies"]
    }
    results = []
    for index, policy_id in enumerate(
        policy["id"] for policy in targets["policies"] if policy["id"] in selected
    ):
        target = target_by_id[policy_id]
        package = (
            args.packages
            / f"plex_paper_{policy_id.replace('-', '_')}.plexpkg"
        )
        if not package.exists():
            raise SystemExit(f"missing package {package}")
        bench = PolicyBench(policy_id, package)
        rng = random.Random(seed + index)
        measured = SCENARIOS[policy_id](bench, rng, trials)
        results.append(
            {
                **target,
                "trials": trials,
                "seed": seed + index,
                "measurement": measured,
                "decision_latency": bench.latency_summary(),
                "operation_counts": dict(sorted(bench.operation_counts.items())),
                "package_sha256": bench.package_sha256,
            }
        )
        print(
            f"{policy_id}: ratio={measured['improvement_ratio']:.4f} "
            f"trend={measured['trend_reproduced']} "
            f"median={results[-1]['decision_latency']['median_us']:.1f}us"
        )
    report = {
        "schema_version": 1,
        "contract": {"major": 0, "minor": 6},
        "environment": {
            "python": sys.version.split()[0],
            "architecture": platform.machine(),
            "operating_system": platform.system(),
        },
        "seed": seed,
        "trials": trials,
        "policy_count": len(results),
        "trend_reproduced_count": sum(
            result["measurement"]["trend_reproduced"] for result in results
        ),
        "results": results,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded)


if __name__ == "__main__":
    main()
