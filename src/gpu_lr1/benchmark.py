from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from gpu_lr1.kernels import (
    make_argmax_workspace,
    triton_bitset_argmax,
    triton_bitset_mask_logits,
    triton_byte_dfa_advance,
    triton_csr_argmax,
    triton_csr_argmax_advance,
    triton_dense_advance,
    triton_dense_argmax,
    triton_dense_mask_logits,
)
from gpu_lr1.tables import (
    PackedTables,
    compile_named_schemas,
    compile_packed_tables,
    table_summary,
)
from gpu_lr1.vocab import Vocabulary
from gpu_lr1.workloads import benchmark_schemas


@dataclass(frozen=True)
class BenchmarkProfile:
    batch_sizes: tuple[int, ...]
    mixes: tuple[str, ...]
    warmup: int
    iterations: int


PROFILES = {
    "quick": BenchmarkProfile(
        batch_sizes=(1, 32),
        mixes=("hot", "mixed_all", "mixed_sparse", "mixed_dense"),
        warmup=3,
        iterations=10,
    ),
    "full": BenchmarkProfile(
        batch_sizes=(1, 8, 32, 128, 512),
        mixes=(
            "hot",
            "homogeneous",
            "mixed4",
            "mixed_all",
            "mixed_sparse",
            "mixed_dense",
        ),
        warmup=8,
        iterations=30,
    ),
}


@dataclass(frozen=True)
class Timing:
    wall_p50_us: float
    wall_p95_us: float
    wall_mean_us: float
    cuda_mean_us: float | None


class RotatingLogits:
    def __init__(self, values: torch.Tensor) -> None:
        self.values = values
        self.index = 0

    def reset(self, index: int = 0) -> None:
        self.index = index

    def next(self) -> torch.Tensor:
        if self.index >= self.values.shape[0]:
            raise RuntimeError("logits pool exhausted")
        value = self.values[self.index]
        self.index += 1
        return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark heterogeneous GPU JSON Schema table strategies"
    )
    parser.add_argument("--profile", choices=PROFILES, default="full")
    parser.add_argument("--batch-sizes", type=int, nargs="+")
    parser.add_argument(
        "--mixes",
        nargs="+",
        choices=(
            "hot",
            "homogeneous",
            "mixed4",
            "mixed_all",
            "mixed_sparse",
            "mixed_dense",
        ),
    )
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--schemas", type=int, default=14)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument(
        "--vocab",
        choices=("synthetic", "gpt2"),
        default="synthetic",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/benchmark.json"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("the benchmark requires a CUDA GPU")
    base_profile = PROFILES[args.profile]
    profile = BenchmarkProfile(
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
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    named_schemas = benchmark_schemas(args.schemas)
    compiled_schemas = compile_named_schemas(named_schemas)
    if args.vocab == "gpt2":
        vocabulary = Vocabulary.tiktoken("gpt2", args.vocab_size)
    else:
        vocabulary = Vocabulary.synthetic(
            args.vocab_size,
            [item.schema for item in named_schemas],
            seed=args.seed,
        )

    tables = compile_packed_tables(
        compiled_schemas,
        vocabulary,
        device=device,
        include_next_state=True,
    )
    tensors = tables.torch_tensors(device)
    results = []

    for mix in profile.mixes:
        for batch_size in profile.batch_sizes:
            rows_np = select_rows(
                tables,
                mix=mix,
                batch_size=batch_size,
                seed=args.seed,
            )
            rows = torch.from_numpy(rows_np).to(device)
            results.extend(
                benchmark_configuration(
                    tables,
                    tensors,
                    rows,
                    batch_size=batch_size,
                    mix=mix,
                    warmup=profile.warmup,
                    iterations=profile.iterations,
                )
            )

    payload = {
        "metadata": machine_metadata(device),
        "config": {
            **vars(args),
            "output": str(args.output),
            "profile_details": asdict(profile),
            "vocabulary_name": vocabulary.name,
            "max_token_bytes": vocabulary.max_token_bytes,
            "logits_buffering": "rotating buffers across timed iterations",
        },
        "schemas": [
            {
                "name": item.name,
                "family": item.family,
                "states": item.dfa.num_states,
            }
            for item in compiled_schemas
        ],
        "tables": table_summary(tables),
        "benchmarks": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print_summary(payload)


def benchmark_configuration(
    tables: PackedTables,
    tensors,
    rows: torch.Tensor,
    *,
    batch_size: int,
    mix: str,
    warmup: int,
    iterations: int,
) -> list[dict[str, object]]:
    vocab_size = tables.vocab_size
    logits_pool = RotatingLogits(
        torch.randn(
            (
                1 + warmup + 2 * iterations,
                batch_size,
                vocab_size,
            ),
            dtype=torch.float16,
            device=rows.device,
        )
    )
    validation_logits = logits_pool.values[0]
    rows_np = rows.cpu().numpy()
    row_nnz = tables.row_nnz[rows_np]
    common = {
        "mix": mix,
        "batch_size": batch_size,
        "unique_schemas": int(
            np.unique(
                np.searchsorted(tables.state_offsets[1:], rows_np, side="right")
            ).size
        ),
        "allowed_min": int(row_nnz.min()),
        "allowed_median": float(np.median(row_nnz)),
        "allowed_max": int(row_nnz.max()),
        "allowed_density_mean": float(row_nnz.mean() / vocab_size),
    }

    dense_output = torch.empty_like(validation_logits)
    bitset_output = torch.empty_like(validation_logits)
    dense_workspace = make_argmax_workspace(
        batch_size,
        vocab_size,
        device=rows.device,
    )
    bitset_workspace = make_argmax_workspace(
        batch_size,
        vocab_size,
        device=rows.device,
    )
    dense_tokens = torch.empty(
        batch_size,
        dtype=torch.int32,
        device=rows.device,
    )
    bitset_tokens = torch.empty_like(dense_tokens)
    csr_tokens = torch.empty_like(dense_tokens)
    next_states = torch.empty_like(rows)

    reference_masked = validation_logits.masked_fill(
        tensors.dense_mask[rows.long()].logical_not(),
        -float("inf"),
    )
    reference_tokens = torch.argmax(reference_masked, dim=1).to(torch.int32)

    strategies: list[
        tuple[str, Callable[[], torch.Tensor], bool, bool]
    ] = [
        (
            "mask_torch_dense",
            lambda: logits_pool.next().masked_fill(
                tensors.dense_mask[rows.long()].logical_not(),
                -float("inf"),
            ),
            True,
            True,
        ),
        (
            "mask_triton_dense",
            lambda: triton_dense_mask_logits(
                logits_pool.next(),
                tensors.dense_mask,
                rows,
                output=dense_output,
            ),
            True,
            True,
        ),
        (
            "mask_triton_bitset",
            lambda: triton_bitset_mask_logits(
                logits_pool.next(),
                tensors.bitset_mask,
                rows,
                output=bitset_output,
            ),
            True,
            True,
        ),
        (
            "select_torch_dense",
            lambda: torch.argmax(
                logits_pool.next().masked_fill(
                    tensors.dense_mask[rows.long()].logical_not(),
                    -float("inf"),
                ),
                dim=1,
            ).to(torch.int32),
            True,
            True,
        ),
        (
            "select_triton_dense_2stage",
            lambda: triton_dense_argmax(
                logits_pool.next(),
                tensors.dense_mask,
                rows,
                workspace=dense_workspace,
                output=dense_tokens,
            ),
            True,
            True,
        ),
        (
            "select_triton_bitset_2stage",
            lambda: triton_bitset_argmax(
                logits_pool.next(),
                tensors.bitset_mask,
                rows,
                workspace=bitset_workspace,
                output=bitset_tokens,
            ),
            True,
            True,
        ),
    ]

    strategies.append(
        (
            "select_triton_csr_1program",
            lambda: triton_csr_argmax(
                logits_pool.next(),
                tensors.csr_indptr,
                tensors.csr_indices,
                rows,
                max_row_nnz=vocab_size,
                output=csr_tokens,
            ),
            True,
            True,
        )
    )
    strategies.append(
        (
            "step_csr_fused",
            lambda: _csr_fused_step(
                logits_pool.next(),
                rows,
                tensors,
                vocab_size,
                csr_tokens,
                next_states,
            ),
            True,
            True,
        )
    )

    if tensors.next_state is None:
        raise RuntimeError("benchmark requires the dense next-state table")

    strategies.extend(
        [
            (
                "advance_triton_dense_next",
                lambda: triton_dense_advance(
                    rows,
                    reference_tokens,
                    tensors.next_state,
                    output=next_states,
                ),
                True,
                False,
            ),
            (
                "advance_triton_byte_dfa",
                lambda: triton_byte_dfa_advance(
                    rows,
                    reference_tokens,
                    tensors.token_bytes,
                    tensors.token_lengths,
                    tensors.byte_transitions,
                    output=next_states,
                ),
                True,
                False,
            ),
            (
                "step_bitset_dense_next",
                lambda: _bitset_dense_step(
                    logits_pool.next(),
                    rows,
                    tensors,
                    bitset_workspace,
                    bitset_tokens,
                    next_states,
                ),
                True,
                True,
            ),
            (
                "step_bitset_byte_dfa",
                lambda: _bitset_byte_step(
                    logits_pool.next(),
                    rows,
                    tensors,
                    bitset_workspace,
                    bitset_tokens,
                    next_states,
                ),
                True,
                True,
            ),
        ]
    )

    cpu_roundtrip = CpuBitsetRoundtrip(
        tables,
        logits_pool.next,
        batch_size,
        rows.device,
        rows,
        bitset_workspace,
    )
    strategies.append(
        ("cpu_sync_bitset_baseline", cpu_roundtrip, False, True)
    )

    records = []
    for name, function, measure_cuda, uses_logits in strategies:
        if uses_logits:
            logits_pool.reset(0)
        output = function()
        torch.cuda.synchronize()
        validate_strategy_output(
            name,
            output,
            reference_masked,
            reference_tokens,
            rows,
            tensors,
        )
        if uses_logits:
            logits_pool.reset(1)
        timing = measure(
            function,
            warmup=warmup,
            iterations=iterations,
            measure_cuda=measure_cuda,
        )
        effective_us = (
            timing.cuda_mean_us
            if timing.cuda_mean_us is not None
            else timing.wall_mean_us
        )
        records.append(
            {
                **common,
                "strategy": name,
                **asdict(timing),
                "sequences_per_second": float(
                    batch_size / (effective_us * 1e-6)
                ),
            }
        )
    return records


class CpuBitsetRoundtrip:
    """Optimistic host baseline: table lookup only, no CPU parser work."""

    def __init__(
        self,
        tables: PackedTables,
        logits_source: Callable[[], torch.Tensor],
        batch_size: int,
        device: torch.device,
        rows: torch.Tensor,
        workspace,
    ) -> None:
        self.logits_source = logits_source
        self.rows = rows
        self.host_table = torch.from_numpy(tables.bitset_mask)
        self.host_rows = torch.empty(
            batch_size,
            dtype=torch.int64,
            pin_memory=True,
        )
        self.host_bitset = torch.empty(
            (batch_size, tables.bitset_mask.shape[1]),
            dtype=torch.int32,
            pin_memory=True,
        )
        self.device_bitset = torch.empty_like(
            self.host_bitset,
            device=device,
        )
        self.local_rows = torch.arange(
            batch_size,
            dtype=torch.int32,
            device=device,
        )
        self.output = torch.empty(
            batch_size,
            dtype=torch.int32,
            device=device,
        )
        self.workspace = workspace

    def __call__(self) -> torch.Tensor:
        self.host_rows.copy_(self.rows, non_blocking=False)
        torch.index_select(
            self.host_table,
            0,
            self.host_rows,
            out=self.host_bitset,
        )
        self.device_bitset.copy_(self.host_bitset, non_blocking=True)
        return triton_bitset_argmax(
            self.logits_source(),
            self.device_bitset,
            self.local_rows,
            workspace=self.workspace,
            output=self.output,
        )


def _bitset_dense_step(
    logits,
    rows,
    tensors,
    workspace,
    tokens,
    next_states,
) -> torch.Tensor:
    triton_bitset_argmax(
        logits,
        tensors.bitset_mask,
        rows,
        workspace=workspace,
        output=tokens,
    )
    return triton_dense_advance(
        rows,
        tokens,
        tensors.next_state,
        output=next_states,
    )


def _bitset_byte_step(
    logits,
    rows,
    tensors,
    workspace,
    tokens,
    next_states,
) -> torch.Tensor:
    triton_bitset_argmax(
        logits,
        tensors.bitset_mask,
        rows,
        workspace=workspace,
        output=tokens,
    )
    return triton_byte_dfa_advance(
        rows,
        tokens,
        tensors.token_bytes,
        tensors.token_lengths,
        tensors.byte_transitions,
        output=next_states,
    )


def _csr_fused_step(
    logits,
    rows,
    tensors,
    max_row_nnz,
    tokens,
    next_states,
) -> torch.Tensor:
    triton_csr_argmax_advance(
        logits,
        tensors.csr_indptr,
        tensors.csr_indices,
        tensors.csr_next_state,
        rows,
        max_row_nnz=max_row_nnz,
        output_tokens=tokens,
        output_states=next_states,
    )
    return next_states


def validate_strategy_output(
    name: str,
    output: torch.Tensor,
    reference_masked: torch.Tensor,
    reference_tokens: torch.Tensor,
    rows: torch.Tensor,
    tensors,
) -> None:
    if name.startswith("mask_"):
        if not torch.equal(output, reference_masked):
            raise AssertionError(f"{name} produced an incorrect masked logits tensor")
        return
    expected_next = tensors.next_state[rows.long(), reference_tokens.long()]
    if name.startswith("advance_") or name.startswith("step_"):
        if not torch.equal(output, expected_next):
            raise AssertionError(f"{name} produced incorrect next states")
        return
    if not torch.equal(output.to(torch.int32), reference_tokens):
        raise AssertionError(f"{name} selected incorrect tokens")


def measure(
    function: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
    measure_cuda: bool,
) -> Timing:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    wall_samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        function()
        torch.cuda.synchronize()
        wall_samples.append((time.perf_counter_ns() - started) / 1_000)

    cuda_mean = None
    if measure_cuda:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(iterations):
            function()
        end_event.record()
        end_event.synchronize()
        cuda_mean = start_event.elapsed_time(end_event) * 1_000 / iterations

    ordered = sorted(wall_samples)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return Timing(
        wall_p50_us=float(statistics.median(ordered)),
        wall_p95_us=float(ordered[p95_index]),
        wall_mean_us=float(statistics.fmean(ordered)),
        cuda_mean_us=float(cuda_mean) if cuda_mean is not None else None,
    )


def select_rows(
    tables: PackedTables,
    *,
    mix: str,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed + batch_size * 17 + _mix_seed(mix))
    nnz = tables.row_nnz
    valid = np.flatnonzero(nnz > 0)
    if valid.size == 0:
        raise ValueError("compiled tables contain no usable states")

    if mix == "hot":
        median = np.median(nnz[valid])
        row = valid[np.argmin(np.abs(nnz[valid] - median))]
        selected = np.full(batch_size, row, dtype=np.int32)
    elif mix == "homogeneous":
        schema_sizes = np.diff(tables.state_offsets)
        schema_id = int(np.argmax(schema_sizes))
        pool = _schema_pool(tables, schema_id)
        selected = rng.choice(pool, size=batch_size, replace=True).astype(np.int32)
    elif mix in {"mixed4", "mixed_all"}:
        schema_count = min(4, tables.num_schemas) if mix == "mixed4" else tables.num_schemas
        schema_ids = np.arange(schema_count)
        selected = np.empty(batch_size, dtype=np.int32)
        for index in range(batch_size):
            schema_id = int(schema_ids[index % schema_count])
            selected[index] = rng.choice(_schema_pool(tables, schema_id))
    elif mix == "mixed_sparse":
        cutoff = max(8, int(np.percentile(nnz[valid], 50)))
        pool = valid[nnz[valid] <= cutoff]
        selected = rng.choice(pool, size=batch_size, replace=True).astype(np.int32)
    elif mix == "mixed_dense":
        cutoff = int(np.percentile(nnz[valid], 90))
        pool = valid[nnz[valid] >= cutoff]
        selected = rng.choice(pool, size=batch_size, replace=True).astype(np.int32)
    else:
        raise ValueError(f"unknown mix: {mix}")
    return np.ascontiguousarray(selected, dtype=np.int32)


def _schema_pool(tables: PackedTables, schema_id: int) -> np.ndarray:
    start = int(tables.state_offsets[schema_id])
    end = int(tables.state_offsets[schema_id + 1])
    rows = np.arange(start, end, dtype=np.int32)
    pool = rows[tables.row_nnz[start:end] > 0]
    if pool.size == 0:
        raise ValueError(f"schema {schema_id} has no usable states")
    return pool


def _mix_seed(mix: str) -> int:
    return {
        "hot": 1,
        "homogeneous": 2,
        "mixed4": 3,
        "mixed_all": 4,
        "mixed_sparse": 5,
        "mixed_dense": 6,
    }[mix]


def machine_metadata(device: torch.device) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": properties.name,
        "gpu_memory_bytes": properties.total_memory,
        "compute_capability": f"{properties.major}.{properties.minor}",
    }


def print_summary(payload: dict[str, object]) -> None:
    metadata = payload["metadata"]
    tables = payload["tables"]
    print(
        f"GPU: {metadata['gpu']} | schemas={tables['schemas']} "
        f"states={tables['states']} vocab={tables['vocab_size']}"
    )
    print(
        f"compile={tables['compile_seconds']:.2f}s "
        f"mean_density={tables['allowed_tokens']['mean_density']:.4%}"
    )
    print("strategy,mix,batch,cuda_us,wall_p50_us,sequences_per_second")
    for item in payload["benchmarks"]:
        cuda = item["cuda_mean_us"]
        cuda_text = "" if cuda is None else f"{cuda:.3f}"
        print(
            f"{item['strategy']},{item['mix']},{item['batch_size']},"
            f"{cuda_text},{item['wall_p50_us']:.3f},"
            f"{item['sequences_per_second']:.1f}"
        )


if __name__ == "__main__":
    main()
