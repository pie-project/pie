from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from gpu_lr1.benchmark import Timing, machine_metadata, measure
from gpu_lr1.kernels import (
    make_lr1_step_workspace,
    triton_lr1_step_fused,
    triton_lr1_step_split,
)
from gpu_lr1.lr1 import (
    CanonicalLR1Compiler,
    LR1StepStatus,
    PackedLR1Tables,
    RaggedLR1Stacks,
    TorchRaggedLR1Stacks,
    pack_lr1_tables,
    select_and_step_lr1_cpu,
    step_lr1_terminals_cpu,
)
from gpu_lr1.lr1_workloads import NamedLR1Workload, benchmark_lr1_workloads


@dataclass(frozen=True)
class LR1BenchmarkProfile:
    batch_sizes: tuple[int, ...]
    mixes: tuple[str, ...]
    warmup: int
    iterations: int


PROFILES = {
    "quick": LR1BenchmarkProfile(
        batch_sizes=(1, 32, 512),
        mixes=("shift", "reduction", "depth", "mixed"),
        warmup=3,
        iterations=10,
    ),
    "full": LR1BenchmarkProfile(
        batch_sizes=(1, 8, 32, 128, 512, 2048),
        mixes=("shift", "reduction", "depth", "mixed"),
        warmup=8,
        iterations=30,
    ),
}


@dataclass(frozen=True)
class PreparedCase:
    name: str
    family: str
    grammar_id: int
    stack: RaggedLR1Stacks
    terminal: int
    expected_status: int
    expected_reductions: int

    @property
    def depth(self) -> int:
        return int(self.stack.pointers[0])

    @property
    def capacity(self) -> int:
        return int(self.stack.offsets[1])


class RotatingLR1Inputs:
    def __init__(
        self,
        logits: torch.Tensor,
        stacks: list[TorchRaggedLR1Stacks],
    ) -> None:
        self.logits = logits
        self.stacks = stacks
        self.index = 0

    def reset(self) -> None:
        self.index = 0

    def next(self) -> tuple[torch.Tensor, TorchRaggedLR1Stacks]:
        if self.index >= len(self.stacks):
            raise RuntimeError("LR(1) input pool exhausted")
        result = self.logits[self.index], self.stacks[self.index]
        self.index += 1
        return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark heterogeneous canonical LR(1) GPU execution"
    )
    parser.add_argument("--profile", choices=PROFILES, default="quick")
    parser.add_argument("--batch-sizes", type=int, nargs="+")
    parser.add_argument(
        "--mixes",
        nargs="+",
        choices=("shift", "reduction", "depth", "mixed"),
    )
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--logit-columns", type=int, default=32768)
    parser.add_argument("--max-reductions", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/lr1-benchmark.json"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("the LR(1) benchmark requires a CUDA GPU")
    base_profile = PROFILES[args.profile]
    profile = LR1BenchmarkProfile(
        batch_sizes=(
            tuple(args.batch_sizes)
            if args.batch_sizes is not None
            else base_profile.batch_sizes
        ),
        mixes=(
            tuple(args.mixes)
            if args.mixes is not None
            else base_profile.mixes
        ),
        warmup=args.warmup if args.warmup is not None else base_profile.warmup,
        iterations=(
            args.iterations
            if args.iterations is not None
            else base_profile.iterations
        ),
    )
    if args.logit_columns <= 0:
        raise ValueError("logit-columns must be positive")
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    workloads = benchmark_lr1_workloads()
    compile_started = time.perf_counter()
    compiled = [
        CanonicalLR1Compiler(workload.grammar).compile()
        for workload in workloads
    ]
    tables = pack_lr1_tables(compiled)
    compile_seconds = time.perf_counter() - compile_started
    if args.logit_columns < tables.num_terminals:
        raise ValueError(
            f"logit-columns must be at least {tables.num_terminals}"
        )
    tensors = tables.torch_tensors(device)
    cases = prepare_cases(workloads, tables)
    records = []

    for mix in profile.mixes:
        for batch_size in profile.batch_sizes:
            records.extend(
                benchmark_configuration(
                    tables,
                    tensors,
                    cases,
                    mix=mix,
                    batch_size=batch_size,
                    logit_columns=args.logit_columns,
                    max_reductions=args.max_reductions,
                    warmup=profile.warmup,
                    iterations=profile.iterations,
                    device=device,
                )
            )

    payload = {
        "metadata": machine_metadata(device),
        "config": {
            **vars(args),
            "output": str(args.output),
            "profile_details": asdict(profile),
        },
        "compile_seconds": compile_seconds,
        "tables": lr1_table_summary(tables),
        "grammars": [
            {
                "name": workload.name,
                "family": workload.family,
                "states": table.num_states,
                "productions": table.num_productions,
                "action_nnz": int(table.action_symbols.size),
                "max_action_row_nnz": int(table.action_row_nnz.max()),
                "prefix_tokens": len(workload.prefix),
                "prepared_depth": cases[index].depth,
                "next_reductions": cases[index].expected_reductions,
            }
            for index, (workload, table) in enumerate(zip(workloads, compiled))
        ],
        "benchmarks": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print_summary(payload)


def prepare_cases(
    workloads: list[NamedLR1Workload],
    tables: PackedLR1Tables,
) -> list[PreparedCase]:
    terminal_ids = {
        name: index for index, name in enumerate(tables.terminal_names)
    }
    cases = []
    for grammar_id, workload in enumerate(workloads):
        stack = RaggedLR1Stacks.initialize(
            [tables.start_states[grammar_id]],
            workload.stack_capacity,
        )
        for symbol in workload.prefix:
            statuses, _ = step_lr1_terminals_cpu(
                tables,
                stack,
                [terminal_ids[symbol]],
                max_reductions=10_000,
            )
            if int(statuses[0]) != int(LR1StepStatus.SHIFTED):
                raise AssertionError(
                    f"failed to prepare {workload.name}: "
                    f"prefix terminal {symbol!r} produced status {statuses[0]}"
                )

        terminal = terminal_ids[workload.next_terminal]
        logits = np.full(
            (1, tables.num_terminals),
            -20.0,
            dtype=np.float32,
        )
        logits[0, terminal] = 20.0
        expected_stack = stack.clone()
        result = select_and_step_lr1_cpu(
            logits,
            tables,
            expected_stack,
            max_reductions=10_000,
        )
        if int(result.terminals[0]) != terminal:
            raise AssertionError(
                f"prepared terminal is not allowed for {workload.name}"
            )
        cases.append(
            PreparedCase(
                name=workload.name,
                family=workload.family,
                grammar_id=grammar_id,
                stack=stack,
                terminal=terminal,
                expected_status=int(result.statuses[0]),
                expected_reductions=int(result.reductions[0]),
            )
        )
    return cases


def benchmark_configuration(
    tables,
    tensors,
    cases: list[PreparedCase],
    *,
    mix: str,
    batch_size: int,
    logit_columns: int,
    max_reductions: int,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> list[dict[str, object]]:
    selected_cases = select_cases(cases, mix, batch_size)
    initial, desired = assemble_batch(selected_cases)
    validation_logits = np.full(
        (batch_size, tables.num_terminals),
        -20.0,
        dtype=np.float32,
    )
    validation_logits[np.arange(batch_size), desired] = 20.0
    cpu_stacks = initial.clone()
    cpu_result = select_and_step_lr1_cpu(
        validation_logits,
        tables,
        cpu_stacks,
        max_reductions=max_reductions,
    )
    np.testing.assert_array_equal(cpu_result.terminals, desired)

    common = {
        "mix": mix,
        "batch_size": batch_size,
        "unique_grammars": len({case.grammar_id for case in selected_cases}),
        "initial_depth_min": min(case.depth for case in selected_cases),
        "initial_depth_mean": float(
            statistics.fmean(case.depth for case in selected_cases)
        ),
        "initial_depth_max": max(case.depth for case in selected_cases),
        "reductions_min": int(cpu_result.reductions.min()),
        "reductions_mean": float(cpu_result.reductions.mean()),
        "reductions_max": int(cpu_result.reductions.max()),
        "stack_pool_entries": int(initial.values.size),
    }
    records = []

    for name, step in (
        ("lr1_fused", triton_lr1_step_fused),
        ("lr1_split_fast_slow", triton_lr1_step_split),
    ):
        validation_stacks = initial.torch_tensors(device)
        validation_workspace = make_lr1_step_workspace(
            batch_size,
            device=device,
        )
        produced = step(
            torch.from_numpy(_pad_logits(validation_logits, logit_columns)).to(
                device
            ),
            tensors,
            validation_stacks,
            max_action_row_nnz=tables.max_action_row_nnz,
            max_goto_row_nnz=tables.max_goto_row_nnz,
            max_reductions=max_reductions,
            workspace=validation_workspace,
        )
        torch.cuda.synchronize()
        np.testing.assert_array_equal(
            produced[0].cpu().numpy(),
            cpu_result.terminals,
        )
        np.testing.assert_array_equal(
            produced[1].cpu().numpy(),
            cpu_result.statuses,
        )
        np.testing.assert_array_equal(
            produced[2].cpu().numpy(),
            cpu_result.reductions,
        )
        np.testing.assert_array_equal(
            validation_stacks.pointers.cpu().numpy(),
            cpu_stacks.pointers,
        )
        np.testing.assert_array_equal(
            validation_stacks.values.cpu().numpy(),
            cpu_stacks.values,
        )

        pool_count = warmup + 2 * iterations
        input_pool = make_input_pool(
            initial,
            desired,
            pool_count=pool_count,
            logit_columns=logit_columns,
            device=device,
        )
        workspace = make_lr1_step_workspace(batch_size, device=device)

        def function(
            step: Callable = step,
            pool: RotatingLR1Inputs = input_pool,
        ) -> torch.Tensor:
            logits, stacks = pool.next()
            return step(
                logits,
                tensors,
                stacks,
                max_action_row_nnz=tables.max_action_row_nnz,
                max_goto_row_nnz=tables.max_goto_row_nnz,
                max_reductions=max_reductions,
                workspace=workspace,
            )[1]

        input_pool.reset()
        timing = measure(
            function,
            warmup=warmup,
            iterations=iterations,
            measure_cuda=True,
        )
        records.append(
            {
                **common,
                "strategy": name,
                **asdict(timing),
                "sequences_per_second": float(
                    batch_size / (timing.cuda_mean_us * 1e-6)
                ),
            }
        )
        del input_pool
        torch.cuda.empty_cache()

    cpu_timing = measure_cpu_reference(
        validation_logits,
        tables,
        initial,
        max_reductions=max_reductions,
        warmup=warmup,
        iterations=iterations,
    )
    records.append(
        {
            **common,
            "strategy": "lr1_cpu_python_reference",
            **asdict(cpu_timing),
            "sequences_per_second": float(
                batch_size / (cpu_timing.wall_mean_us * 1e-6)
            ),
        }
    )
    return records


def select_cases(
    cases: list[PreparedCase],
    mix: str,
    batch_size: int,
) -> list[PreparedCase]:
    pool = cases if mix == "mixed" else [
        case for case in cases if case.family == mix
    ]
    if not pool:
        raise ValueError(f"no LR(1) workloads for mix {mix}")
    return [pool[index % len(pool)] for index in range(batch_size)]


def assemble_batch(
    cases: list[PreparedCase],
) -> tuple[RaggedLR1Stacks, np.ndarray]:
    capacities = np.asarray([case.capacity for case in cases], dtype=np.int32)
    offsets = np.empty(len(cases) + 1, dtype=np.int32)
    offsets[0] = 0
    offsets[1:] = np.cumsum(capacities, dtype=np.int64).astype(np.int32)
    values = np.full(int(offsets[-1]), -1, dtype=np.int32)
    pointers = np.empty(len(cases), dtype=np.int32)
    desired = np.empty(len(cases), dtype=np.int32)

    for index, case in enumerate(cases):
        base = int(offsets[index])
        capacity = case.capacity
        values[base : base + capacity] = case.stack.values
        pointers[index] = case.stack.pointers[0]
        desired[index] = case.terminal
    return RaggedLR1Stacks(values, offsets, pointers), desired


def make_input_pool(
    initial: RaggedLR1Stacks,
    desired: np.ndarray,
    *,
    pool_count: int,
    logit_columns: int,
    device: torch.device,
) -> RotatingLR1Inputs:
    logits = torch.randn(
        (pool_count, initial.batch_size, logit_columns),
        dtype=torch.float16,
        device=device,
    )
    pool_ids = torch.arange(pool_count, device=device)[:, None]
    batch_ids = torch.arange(initial.batch_size, device=device)[None, :]
    desired_device = torch.from_numpy(desired).to(device)[None, :]
    logits[pool_ids, batch_ids, desired_device] = 20.0

    base_values = torch.from_numpy(initial.values).to(device)
    base_pointers = torch.from_numpy(initial.pointers).to(device)
    offsets = torch.from_numpy(initial.offsets).to(device)
    value_pool = base_values.repeat(pool_count, 1)
    pointer_pool = base_pointers.repeat(pool_count, 1)
    stacks = [
        TorchRaggedLR1Stacks(
            values=value_pool[index],
            offsets=offsets,
            pointers=pointer_pool[index],
        )
        for index in range(pool_count)
    ]
    return RotatingLR1Inputs(logits, stacks)


def measure_cpu_reference(
    logits: np.ndarray,
    tables: PackedLR1Tables,
    initial: RaggedLR1Stacks,
    *,
    max_reductions: int,
    warmup: int,
    iterations: int,
) -> Timing:
    def function() -> None:
        select_and_step_lr1_cpu(
            logits,
            tables,
            initial.clone(),
            max_reductions=max_reductions,
        )

    for _ in range(warmup):
        function()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - started) / 1_000)
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return Timing(
        wall_p50_us=float(statistics.median(ordered)),
        wall_p95_us=float(ordered[p95_index]),
        wall_mean_us=float(statistics.fmean(ordered)),
        cuda_mean_us=None,
    )


def _pad_logits(logits: np.ndarray, columns: int) -> np.ndarray:
    if logits.shape[1] > columns:
        raise ValueError("logit column count is smaller than terminal count")
    padded = np.full(
        (logits.shape[0], columns),
        -20.0,
        dtype=np.float16,
    )
    padded[:, : logits.shape[1]] = logits
    return padded


def lr1_table_summary(tables: PackedLR1Tables) -> dict[str, object]:
    action_nnz = tables.action_row_nnz
    goto_nnz = tables.goto_row_nnz
    memory = tables.memory_bytes()
    return {
        "grammars": tables.num_grammars,
        "states": tables.num_states,
        "terminals": tables.num_terminals,
        "productions": tables.num_productions,
        "action_entries": int(tables.action_symbols.size),
        "goto_entries": int(tables.goto_symbols.size),
        "action_row_nnz": {
            "min": int(action_nnz.min()),
            "median": float(np.median(action_nnz)),
            "p95": float(np.percentile(action_nnz, 95)),
            "max": int(action_nnz.max()),
        },
        "goto_row_nnz": {
            "min": int(goto_nnz.min()),
            "median": float(np.median(goto_nnz)),
            "p95": float(np.percentile(goto_nnz, 95)),
            "max": int(goto_nnz.max()),
        },
        "memory_bytes": memory,
        "memory_total_bytes": int(sum(memory.values())),
    }


def print_summary(payload: dict[str, object]) -> None:
    metadata = payload["metadata"]
    tables = payload["tables"]
    print(
        f"GPU: {metadata['gpu']} | grammars={tables['grammars']} "
        f"states={tables['states']} terminals={tables['terminals']}"
    )
    print(
        f"compile={payload['compile_seconds']:.3f}s "
        f"table_bytes={tables['memory_total_bytes']}"
    )
    print(
        "strategy,mix,batch,depth_mean,reductions_mean,"
        "cuda_us,wall_p50_us,sequences_per_second"
    )
    for item in payload["benchmarks"]:
        cuda = item["cuda_mean_us"]
        cuda_text = "" if cuda is None else f"{cuda:.3f}"
        print(
            f"{item['strategy']},{item['mix']},{item['batch_size']},"
            f"{item['initial_depth_mean']:.2f},"
            f"{item['reductions_mean']:.2f},{cuda_text},"
            f"{item['wall_p50_us']:.3f},"
            f"{item['sequences_per_second']:.1f}"
        )


if __name__ == "__main__":
    main()
