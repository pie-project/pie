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
    token_budget: int = 1,
) -> tuple[list[int], dict[str, Any]]:
    outcome = bench.invoke(
        "schedule",
        {
            "meta": bench.meta("schedule"),
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
                "max_requests": max_selections,
                "max_total_tokens": max_selections * token_budget,
                "facts": {},
            },
        },
        lifecycle,
    )
    selections = outcome["plan"]["plan"]["selections"]
    selected = [
        request_index
        for selection in selections
        for request_index in selection["requests"]
    ]
    return selected, outcome


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
            "meta": bench.meta("route"),
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
    assignments: list[int | None] = [None] * len(refs)
    for assignment in outcome["plan"]["plan"]["assignments"]:
        assignments[assignment["request_index"]] = assignment["target_index"]
    return assignments, outcome


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
    outcome = bench.invoke(
        "cache",
        {
            "meta": bench.meta("cache"),
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
    return outcome["plan"]["plan"], outcome


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
    policy_values = []
    baseline_values = []
    for trial in range(trials):
        services = [rng.randint(10, 500) for _ in range(4)]
        waiting = [rng.randint(0, 2500) for _ in range(4)]
        facts = [{"waiting_ms": value} for value in waiting]
        refs, lifecycle = bootstrap_requests(
            bench,
            f"agentix-{trial}",
            facts,
            status="active",
            group_keys=[str(index) for index in range(4)],
        )
        bench.feedback(
            [
                {
                    "subject": {
                        "kind": "work-group",
                        "value": ref["group_id"],
                    },
                    "outcome": "progress",
                    "facts": {"service_us": service},
                }
                for ref, service in zip(refs, services)
            ],
            lifecycle,
        )
        selected, _ = invoke_schedule(bench, refs, facts, None)
        utilities = [
            (1_000_000 if wait * 1000 >= max(service, 1) * 4 else 0)
            + (500 - service)
            for wait, service in zip(waiting, services)
        ]
        policy_values.append(utilities[selected[0]])
        baseline_values.append(utilities[0])
    return result(
        policy_values,
        baseline_values,
        direction="higher",
        unit="starvation_adjusted_priority",
    )


def scenario_continuum(
    bench: PolicyBench, rng: random.Random, trials: int
) -> dict[str, Any]:
    policy_costs = []
    baseline_costs = []
    for trial in range(trials):
        pinned_index = rng.randrange(4)
        facts = [
            {
                "preempted": index == pinned_index,
                "program_arrival": rng.randint(0, 1000),
            }
            for index in range(4)
        ]
        refs, lifecycle = bootstrap_requests(
            bench, f"continuum-{trial}", facts, status="active"
        )
        bench.feedback(
            [
                {
                    "subject": {
                        "kind": "request",
                        "value": refs[pinned_index]["request_id"],
                    },
                    "outcome": "progress",
                    "facts": {"ttl_ms": 5000},
                }
            ],
            lifecycle,
        )
        selected, _ = invoke_schedule(bench, refs, facts, None)
        reload_costs = [rng.randint(10, 1000) for _ in refs]
        residents = [
            cache_object(
                f"continuum-{trial}-{index}",
                {"reload_cost": reload_cost},
                beneficiary=ref["request_id"],
            )
            for index, (ref, reload_cost) in enumerate(
                zip(refs, reload_costs)
            )
        ]
        plan, _ = invoke_cache(
            bench, residents, [], None, max_bytes=len(residents) - 1
        )
        policy_victim = plan["reclaim"][0]
        schedule_penalty = 0 if selected[0] == pinned_index else 5000
        policy_costs.append(
            schedule_penalty
            + reload_costs[policy_victim]
            + (10000 if policy_victim == pinned_index else 0)
        )
        baseline_costs.append(
            reload_costs[pinned_index] + 10000
        )
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
        ready = [rng.random() < 0.5 for _ in range(5)]
        ready[rng.randrange(5)] = True
        facts = [{"cache_ready": value} for value in ready]
        refs, lifecycle = bootstrap_requests(
            bench, f"kvflow-{trial}", facts, status="active"
        )
        selected, _ = invoke_schedule(bench, refs, facts, lifecycle)
        steps = [rng.randint(1, 20) for _ in refs]
        fixed = [rng.random() < 0.4 for _ in refs]
        residents = [
            cache_object(
                f"kvflow-{trial}-{index}",
                {
                    "fixed_prefix": fixed_prefix,
                    "steps_to_execution": distance,
                    "loading": False,
                    "offloading": False,
                },
                beneficiary=ref["request_id"],
            )
            for index, (ref, fixed_prefix, distance) in enumerate(
                zip(refs, fixed, steps)
            )
        ]
        plan, _ = invoke_cache(
            bench, residents, [], None, max_bytes=len(residents) - 1
        )
        victim = plan["reclaim"][0]
        harm = [
            (1000 if fixed_prefix else 0) + 100 / (distance + 1)
            for fixed_prefix, distance in zip(fixed, steps)
        ]
        policy_costs.append(
            (0 if ready[selected[0]] else 500) + harm[victim]
        )
        baseline_costs.append((0 if ready[0] else 500) + harm[0])
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
        remaining = rng.randint(128, 2048)
        cached = [rng.randint(0, 4096) for _ in range(3)]
        load = [rng.randint(10, 300) for _ in range(3)]
        eviction = [rng.randint(0, 300) for _ in range(3)]
        refs, lifecycle = bootstrap_requests(
            bench,
            f"preble-{trial}",
            [{"uncached_tokens": remaining}],
            status="admitted",
        )
        assignments, _ = invoke_route(
            bench,
            refs,
            [{"uncached_tokens": remaining}],
            [{}, {}, {}],
            [
                [
                    {
                        "cached_tokens": cached[index],
                        "load_cost": load[index],
                        "eviction_cost": eviction[index],
                        "miss_prefill_cost": max(
                            remaining - cached[index], 0
                        ),
                    }
                    for index in range(3)
                ]
            ],
            lifecycle,
        )
        exploit = max(cached) > remaining
        costs = [
            max(remaining - cached[index], 0) + 1
            if exploit
            else remaining + load[index] + eviction[index]
            for index in range(3)
        ]
        policy_costs.append(costs[assignments[0]])
        baseline_index = min(range(3), key=load.__getitem__)
        baseline_costs.append(costs[baseline_index])
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
        facts = [
            {
                "ready": rng.random() < 0.7,
                "dependency_depth": rng.randint(0, 10),
                "prefix_reuse_tokens": rng.randint(0, 4096),
                "earliest_start": rng.randint(0, 1000),
                "profiled_token_cost": rng.randint(1, 1000),
            }
            for _ in range(6)
        ]
        if not any(item["ready"] for item in facts) and rng.random() < 0.8:
            facts[rng.randrange(len(facts))]["ready"] = True
        refs, lifecycle = bootstrap_requests(
            bench, f"helium-{trial}", facts, status="active"
        )
        selected, _ = invoke_schedule(bench, refs, facts, lifecycle)
        if any(item["ready"] for item in facts):
            utilities = [
                (
                    item["dependency_depth"] * 1_000_000
                    + item["prefix_reuse_tokens"] * 100
                    + 1001
                    - item["profiled_token_cost"]
                    if item["ready"]
                    else 1
                )
                for item in facts
            ]
        else:
            utilities = [-item["earliest_start"] for item in facts]
        policy_values.append(utilities[selected[0]])
        baseline_values.append(utilities[0])
    return result(
        policy_values,
        baseline_values,
        direction="higher",
        unit="critical_path_priority",
    )


def scenario_vtc(
    bench: PolicyBench, _rng: random.Random, trials: int
) -> dict[str, Any]:
    clients = 4
    facts = [
        {
            "client_id": f"client-{index}",
            "dispatch_input_tokens": 0,
            "input_weight": 1,
        }
        for index in range(clients)
    ]
    refs, lifecycle = bootstrap_requests(
        bench, "vtc", facts, status="active"
    )
    policy_service = [0.0] * clients
    baseline_service = [0.0] * clients
    for step in range(max(trials, 16) * clients):
        selected, _ = invoke_schedule(
            bench, refs, facts, lifecycle if step == 0 else None
        )
        index = selected[0]
        policy_service[index] += 1
        baseline_service[0] += 1
        bench.feedback(
            [
                {
                    "subject": {
                        "kind": "request",
                        "value": refs[index]["request_id"],
                    },
                    "outcome": "progress",
                    "facts": {
                        "client_id": facts[index]["client_id"],
                        "input_tokens": 0,
                        "output_tokens": 1,
                        "output_weight": 1,
                    },
                }
            ]
        )
    return result(
        [jain(policy_service)],
        [jain(baseline_service)],
        direction="higher",
        unit="jain_fairness",
        details={
            "policy_service": policy_service,
            "baseline_service": baseline_service,
        },
    )


def scenario_lmetric(
    bench: PolicyBench, rng: random.Random, trials: int
) -> dict[str, Any]:
    policy_costs = []
    baseline_costs = []
    for trial in range(trials):
        new_tokens = [rng.randint(1, 4096) for _ in range(4)]
        batch_sizes = [rng.randint(1, 64) for _ in range(4)]
        hotspot = [rng.random() < 0.15 for _ in range(4)]
        hotspot[rng.randrange(4)] = False
        refs, lifecycle = bootstrap_requests(
            bench, f"lmetric-{trial}", [{}], status="admitted"
        )
        assignments, _ = invoke_route(
            bench,
            refs,
            [{}],
            [{"hotspot_confirmed": value} for value in hotspot],
            [
                [
                    {
                        "new_prefill_tokens": new_tokens[index],
                        "current_batch_size": batch_sizes[index],
                    }
                    for index in range(4)
                ]
            ],
            lifecycle,
        )
        costs = [
            new_tokens[index] * (batch_sizes[index] + 1)
            + (10**9 if hotspot[index] else 0)
            for index in range(4)
        ]
        policy_costs.append(costs[assignments[0]])
        baseline_index = min(range(4), key=batch_sizes.__getitem__)
        baseline_costs.append(costs[baseline_index])
    return result(
        policy_costs,
        baseline_costs,
        direction="lower",
        unit="prefill_batch_product",
    )


def scenario_fairserve(
    bench: PolicyBench, _rng: random.Random, trials: int
) -> dict[str, Any]:
    weights = [1.0, 2.0, 4.0, 8.0]
    facts = [
        {
            "client_id": f"client-{index}",
            "application_id": f"application-{index}",
            "weight": int(weight),
            "kv_overloaded": True,
            "interaction_in_progress": False,
            "user_rpm_remaining": 1,
            "app_rpm_remaining": 1,
        }
        for index, weight in enumerate(weights)
    ]
    refs, lifecycle = bootstrap_requests(
        bench, "fairserve", facts, status="active"
    )
    admit_facts = [
        {
            **item,
            "interaction_in_progress": index >= 2,
        }
        for index, item in enumerate(facts)
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
    steps = max(trials, 16) * len(weights)
    for step in range(steps):
        selected, _ = invoke_schedule(
            bench, refs, facts, lifecycle if step == 0 else None
        )
        index = selected[0]
        policy_service[index] += 1
        baseline_service[step % len(weights)] += 1
        bench.feedback(
            [
                {
                    "subject": {
                        "kind": "request",
                        "value": refs[index]["request_id"],
                    },
                    "outcome": "progress",
                    "facts": {
                        "client_id": facts[index]["client_id"],
                        "application_id": facts[index]["application_id"],
                        "service_tokens": 1,
                    },
                }
            ]
        )
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
        values = [rng.randint(1, 10000) for _ in range(8)]
        residents = [
            cache_object(
                f"marconi-{trial}-{index}",
                {
                    "reuse_probability_ppm": value,
                    "recompute_flops": 1000,
                },
            )
            for index, value in enumerate(values)
        ]
        plan, _ = invoke_cache(
            bench, residents, [], None, max_bytes=4
        )
        reclaimed = set(plan["reclaim"])
        policy_values.append(
            sum(value for index, value in enumerate(values) if index not in reclaimed)
        )
        baseline_values.append(sum(values[-4:]))
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
            age = rng.randint(0, 100)
            size = rng.randint(1, 8)
            scores.append(age + recompute * frequency / size)
            residents.append(
                cache_object(
                    f"ragcache-{trial}-{index}",
                    {
                        "leaf": True,
                        "frequency": frequency,
                        "recompute_cost": recompute,
                        "age": age,
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
        positive_deficit = trial % 2 == 0
        refs, lifecycle = bootstrap_requests(
            bench,
            f"dlpm-{trial}",
            [{"client_id": client}],
            status="admitted",
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
                        "client_id": client,
                        "output_tokens": 0 if positive_deficit else 100,
                    },
                }
            ],
            lifecycle,
        )
        cached = [rng.randint(0, 4096) for _ in range(4)]
        load = [rng.randint(1, 100) for _ in range(4)]
        assignments, _ = invoke_route(
            bench,
            refs,
            [{"client_id": client}],
            [{}, {}, {}, {}],
            [
                [
                    {
                        "cached_tokens": cached[index],
                        "load": load[index],
                        "worker_deficit": 100 if positive_deficit else -100,
                    }
                    for index in range(4)
                ]
            ],
            [],
        )
        max_cached = max(cached)
        costs = [
            (max_cached - cached[index]) * 10 + load[index]
            if positive_deficit
            else load[index]
            for index in range(4)
        ]
        policy_costs.append(costs[assignments[0]])
        baseline_index = max(range(4), key=cached.__getitem__)
        baseline_costs.append(costs[baseline_index])
        invoke_schedule(
            bench,
            refs,
            [
                {
                    "client_id": client,
                    "cached_tokens": max(cached),
                    "quantum": 100,
                    "extend_tokens": 1,
                }
            ],
            [
                {
                    "event": "activate-request",
                    "request_id": refs[0]["request_id"],
                }
            ],
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
                "waiting_ms": rng.randint(0, 2000),
                "fairness_threshold_ms": 1000,
                "demand_depth": rng.randint(0, 20),
            }
            for _ in range(6)
        ]
        refs, lifecycle = bootstrap_requests(
            bench, f"peek-{trial}", facts, status="active"
        )
        selected, _ = invoke_schedule(bench, refs, facts, lifecycle)
        utilities = [
            (10000 if item["waiting_ms"] >= item["fairness_threshold_ms"] else 0)
            + item["demand_depth"]
            for item in facts
        ]
        pending_depth = [rng.randint(0, 50) for _ in refs]
        residents = [
            cache_object(
                f"peek-{trial}-{index}",
                {"pending_demand_depth": depth},
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
    policy_values = []
    baseline_values = []
    for trial in range(trials):
        completed = [rng.randint(0, 8) for _ in range(4)]
        facts = [{} for _ in completed]
        refs, lifecycle = bootstrap_requests(
            bench,
            f"justitia-{trial}",
            facts,
            status="active",
            group_keys=[str(index) for index in range(4)],
        )
        records = []
        for ref, count in zip(refs, completed):
            records.extend(
                {
                    "subject": {
                        "kind": "work-group",
                        "value": ref["group_id"],
                    },
                    "outcome": "completed",
                    "facts": {},
                }
                for _ in range(count)
            )
        if records:
            bench.feedback(records, lifecycle)
            lifecycle = None
        selected, _ = invoke_schedule(bench, refs, facts, lifecycle)
        remaining = [8 - value for value in completed]
        policy_values.append(remaining[selected[0]])
        baseline_values.append(remaining[0])
    return result(
        policy_values,
        baseline_values,
        direction="higher",
        unit="remaining_branch_criticality",
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
        facts = [
            {
                "upstream_elapsed_ms": rng.randint(0, 500),
                "current_queue_ms": rng.randint(0, 200),
                "current_execution_ms": rng.randint(10, 200),
                "downstream_queue_ms": rng.randint(0, 200),
                "downstream_execution_ms": rng.randint(10, 200),
                "downstream_batch_wait_p10_ms": rng.randint(0, 100),
                "deadline_ms": rng.randint(250, 700),
            }
            for _ in range(8)
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
        policy_values.append(sum(feasible[index] for index in selected))
        baseline_values.append(sum(feasible[:4]))
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
        followup = trial % 2 == 1
        affinity = [rng.randint(0, 100) for _ in range(4)]
        load = [rng.randint(1, 100) for _ in range(4)]
        refs, lifecycle = bootstrap_requests(
            bench,
            f"smetric-{trial}",
            [{"overload_ppm": 1_500_000, "hit_ratio_ppm": 500_000}],
            status="admitted",
            generations=[1 if followup else 0],
        )
        assignments, _ = invoke_route(
            bench,
            refs,
            [{"overload_ppm": 1_500_000, "hit_ratio_ppm": 500_000}],
            [{}, {}, {}, {}],
            [
                [
                    {
                        "cache_affinity": affinity[index],
                        "load": load[index],
                        "estimated_history_hit": 50,
                    }
                    for index in range(4)
                ]
            ],
            lifecycle,
        )
        costs = [
            load[index]
            + (
                (max(affinity) - affinity[index]) * 200
                if followup
                else 0
            )
            for index in range(4)
        ]
        policy_costs.append(costs[assignments[0]])
        baseline_index = min(range(4), key=load.__getitem__)
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
        locality = [rng.randint(0, 100) for _ in range(4)]
        load = [rng.randint(1, 100) for _ in range(4)]
        group_service = [rng.randint(0, 1000) for _ in range(4)]
        refs, lifecycle = bootstrap_requests(
            bench,
            f"saga-{trial}",
            [
                {
                    "steal": False,
                    "group_service": service,
                }
                for service in group_service
            ],
            status="admitted",
            group_keys=[str(index) for index in range(4)],
        )
        assignments, _ = invoke_route(
            bench,
            refs[:1],
            [{"steal": False}],
            [{}, {}, {}, {}],
            [
                [
                    {
                        "cache_locality": locality[index],
                        "load": load[index],
                    }
                    for index in range(4)
                ]
            ],
            lifecycle,
        )
        selected, _ = invoke_schedule(
            bench,
            refs,
            [{"group_service": service} for service in group_service],
            [
                {
                    "event": "activate-request",
                    "request_id": ref["request_id"],
                }
                for ref in refs
            ],
        )
        costs = [
            (max(locality) - locality[index]) * 1000 + load[index]
            for index in range(4)
        ]
        policy_costs.append(
            costs[assignments[0]] + group_service[selected[0]]
        )
        baseline_index = min(range(4), key=load.__getitem__)
        baseline_costs.append(costs[baseline_index] + group_service[0])
        prospective = [
            cache_object(
                f"saga-{trial}-{index}",
                {"workflow_ttl_ms": 1000 if index >= 2 else 0},
            )
            for index in range(4)
        ]
        invoke_cache(
            bench,
            [],
            prospective,
            None,
            max_bytes=len(prospective),
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
        outputs = [rng.randint(16, 512) for _ in range(6)]
        quality = [
            [rng.randint(100_000, 1_000_000) for _ in range(3)]
            for _ in range(6)
        ]
        costs = [
            [rng.randint(1, 100) for _ in range(3)] for _ in range(6)
        ]
        latency = [
            [rng.randint(1, 100) for _ in range(3)] for _ in range(6)
        ]
        decode = [
            [rng.randint(1, 5) for _ in range(3)] for _ in range(6)
        ]
        queued = [rng.randint(0, 200) for _ in range(3)]
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
            [{"queued_tokens": value} for value in queued],
            [
                [
                    {
                        "quality_ppm": quality[request_index][target_index],
                        "cost": costs[request_index][target_index],
                        "latency_ms": latency[request_index][target_index],
                        "decode_ms_per_token": decode[request_index][target_index],
                    }
                    for target_index in range(3)
                ]
                for request_index in range(6)
            ],
            lifecycle,
            target_capacity=2,
        )
        utilities = []
        for request_index in range(6):
            q_row = quality[request_index]
            c_row = costs[request_index]
            t_row = [
                latency[request_index][target]
                + decode[request_index][target] * queued[target]
                for target in range(3)
            ]
            q_min, q_max = min(q_row), max(q_row)
            c_min, c_max = min(c_row), max(c_row)
            t_min, t_max = min(t_row), max(t_row)
            utilities.append(
                [
                    (
                        (q_row[target] - q_min)
                        / max(q_max - q_min, 1)
                        + (c_max - c_row[target])
                        / max(c_max - c_min, 1)
                        + (t_max - t_row[target])
                        / max(t_max - t_min, 1)
                    )
                    for target in range(3)
                ]
            )
        policy_values.append(
            sum(
                utilities[index][target]
                for index, target in enumerate(assignments)
                if target is not None
            )
        )
        remaining = [2, 2, 2]
        baseline = 0.0
        for request_index in range(6):
            target = min(
                (
                    target
                    for target, slots in enumerate(remaining)
                    if slots > 0
                ),
                key=lambda target: queued[target],
            )
            remaining[target] -= 1
            baseline += utilities[request_index][target]
        baseline_values.append(baseline)
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
