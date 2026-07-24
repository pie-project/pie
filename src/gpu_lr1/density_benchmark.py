from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gpu_lr1.benchmark import RotatingLogits, machine_metadata, measure
from gpu_lr1.kernels import (
    make_argmax_workspace,
    triton_bitset_argmax,
    triton_csr_argmax,
    triton_dense_argmax,
)
from gpu_lr1.tables import (
    compile_named_schemas,
    compile_packed_tables,
    table_summary,
)
from gpu_lr1.vocab import Vocabulary
from gpu_lr1.workloads import benchmark_schemas


BINS = (
    ("1-8", 1, 8),
    ("9-64", 9, 64),
    ("65-256", 65, 256),
    ("257-1024", 257, 1024),
    ("1025-4096", 1025, 4096),
    ("4097-8192", 4097, 8192),
    ("8193+", 8193, 2**31 - 1),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure dense, bitset, and CSR crossover by mask density"
    )
    parser.add_argument("--schemas", type=int, default=14)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[32, 128, 512],
    )
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/density-crossover.json"),
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    named_schemas = benchmark_schemas(args.schemas)
    vocabulary = Vocabulary.tiktoken("gpt2", args.vocab_size)
    tables = compile_packed_tables(
        compile_named_schemas(named_schemas),
        vocabulary,
        device=device,
        include_next_state=False,
    )
    tensors = tables.torch_tensors(device)
    nnz = tables.row_nnz
    rng = np.random.default_rng(41)
    records = []

    for label, lower, upper in BINS:
        pool = np.flatnonzero((nnz >= lower) & (nnz <= upper))
        if pool.size == 0:
            continue
        for batch_size in args.batch_sizes:
            rows_np = rng.choice(pool, size=batch_size, replace=True).astype(np.int32)
            rows = torch.from_numpy(rows_np).to(device)
            logits_pool = RotatingLogits(
                torch.randn(
                    (
                        1 + args.warmup + 2 * args.iterations,
                        batch_size,
                        vocabulary.size,
                    ),
                    dtype=torch.float16,
                    device=device,
                )
            )
            validation_logits = logits_pool.values[0]
            reference = torch.argmax(
                validation_logits.masked_fill(
                    tensors.dense_mask[rows.long()].logical_not(),
                    -float("inf"),
                ),
                dim=1,
            ).to(torch.int32)
            workspace = make_argmax_workspace(
                batch_size,
                vocabulary.size,
                device=device,
            )
            output = torch.empty_like(reference)
            strategies = [
                (
                    "dense_2stage",
                    lambda: triton_dense_argmax(
                        logits_pool.next(),
                        tensors.dense_mask,
                        rows,
                        workspace=workspace,
                        output=output,
                    ),
                ),
                (
                    "bitset_2stage",
                    lambda: triton_bitset_argmax(
                        logits_pool.next(),
                        tensors.bitset_mask,
                        rows,
                        workspace=workspace,
                        output=output,
                    ),
                ),
            ]
            selected_max = int(nnz[rows_np].max())
            strategies.append(
                (
                    "csr_1program",
                    lambda: triton_csr_argmax(
                        logits_pool.next(),
                        tensors.csr_indptr,
                        tensors.csr_indices,
                        rows,
                        max_row_nnz=vocabulary.size,
                        output=output,
                    ),
                )
            )

            for strategy, function in strategies:
                logits_pool.reset(0)
                actual = function()
                torch.cuda.synchronize()
                if not torch.equal(actual, reference):
                    raise AssertionError(f"{strategy} selected incorrect tokens")
                logits_pool.reset(1)
                timing = measure(
                    function,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    measure_cuda=True,
                )
                records.append(
                    {
                        "bin": label,
                        "batch_size": batch_size,
                        "strategy": strategy,
                        "available_rows": int(pool.size),
                        "allowed_min": int(nnz[rows_np].min()),
                        "allowed_median": float(np.median(nnz[rows_np])),
                        "allowed_max": selected_max,
                        "cuda_mean_us": timing.cuda_mean_us,
                        "wall_p50_us": timing.wall_p50_us,
                        "sequences_per_second": float(
                            batch_size / (timing.cuda_mean_us * 1e-6)
                        ),
                    }
                )

    payload = {
        "metadata": machine_metadata(device),
        "config": {
            **vars(args),
            "output": str(args.output),
            "vocabulary_name": vocabulary.name,
            "logits_buffering": "rotating buffers across timed iterations",
        },
        "tables": table_summary(tables),
        "benchmarks": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("bin,batch,strategy,allowed_median,allowed_max,cuda_us")
    for item in records:
        print(
            f"{item['bin']},{item['batch_size']},{item['strategy']},"
            f"{item['allowed_median']:.1f},{item['allowed_max']},"
            f"{item['cuda_mean_us']:.3f}"
        )


if __name__ == "__main__":
    main()
