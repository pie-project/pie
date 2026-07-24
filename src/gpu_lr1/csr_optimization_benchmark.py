from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import triton

from gpu_lr1.benchmark import machine_metadata, measure
from gpu_lr1.kernels import (
    _csr_argmax_advance_kernel,
    make_ell_argmax_advance_plan,
    triton_csr_argmax_advance_packed,
)
from gpu_lr1.lr1_token_benchmark import (
    CyclingGraphs,
    CyclingLogits,
    benchmark_large_vocab_runtime,
    compile_qwen_probe,
    select_states,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep sparse CSR token-selection launch strategies"
    )
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 32, 128, 512, 2048],
    )
    parser.add_argument(
        "--row-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 16, 32, 64, 128, 256, 512, 1024, 4096, 8192],
    )
    parser.add_argument("--synthetic-vocab-size", type=int, default=32768)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--kernel-repetitions", type=int, default=100)
    parser.add_argument("--qwen-buffer-count", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/csr-optimization.json"),
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    qwen_records, qwen_tables = compile_qwen_probe(
        args.qwen_model,
        compile_timeout=10.0,
        max_configurations=20_000,
    )
    qwen_tensors = qwen_tables.torch_tensors(device)
    ell_lengths_np, ell_tokens_np, ell_next_np = qwen_tables.ell_tables()
    ell_lengths = torch.from_numpy(ell_lengths_np).to(device)
    ell_tokens = torch.from_numpy(ell_tokens_np).to(device)
    ell_next = torch.from_numpy(ell_next_np).to(device)
    ell_memory_bytes = int(
        ell_lengths.numel() * ell_lengths.element_size()
        + ell_tokens.numel() * ell_tokens.element_size()
        + ell_next.numel() * ell_next.element_size()
    )
    qwen_benchmarks = []
    for batch_size in args.batch_sizes:
        states = select_states(
            qwen_tables,
            mix="heterogeneous",
            batch_size=batch_size,
            seed=0,
        )
        qwen_benchmarks.extend(
            benchmark_large_vocab_runtime(
                qwen_tables,
                qwen_tensors,
                states,
                mix="qwen3_heterogeneous",
                buffer_count=args.qwen_buffer_count,
                warmup=args.warmup,
                iterations=args.iterations,
                device=device,
            )
        )
        qwen_benchmarks.extend(
            _benchmark_qwen_ell(
                qwen_tables,
                states,
                ell_lengths,
                ell_tokens,
                ell_next,
                buffer_count=args.qwen_buffer_count,
                warmup=args.warmup,
                iterations=args.iterations,
                device=device,
            )
        )

    sweep_started = time.perf_counter()
    sweep = []
    for row_size in args.row_sizes:
        if row_size > args.synthetic_vocab_size:
            raise ValueError("row size exceeds synthetic vocabulary")
        indptr = torch.tensor(
            [0, row_size],
            dtype=torch.int32,
            device=device,
        )
        indices = torch.arange(
            row_size,
            dtype=torch.int32,
            device=device,
        )
        next_states = indices + 100
        block_size = triton.next_power_of_2(row_size)
        for batch_size in args.batch_sizes:
            rows = torch.zeros(
                batch_size,
                dtype=torch.int32,
                device=device,
            )
            logits = torch.randn(
                (batch_size, args.synthetic_vocab_size),
                dtype=torch.float16,
                device=device,
            )
            output_tokens = torch.empty_like(rows)
            output_states = torch.empty_like(rows)
            expected_tokens = torch.argmax(
                logits[:, :row_size],
                dim=1,
            ).to(torch.int32)
            expected_states = expected_tokens + 100
            candidates = []

            for num_warps in (1, 2, 4, 8):
                def launch_single(num_warps: int = num_warps) -> None:
                    _csr_argmax_advance_kernel[(batch_size,)](
                        logits,
                        indptr,
                        indices,
                        next_states,
                        rows,
                        output_tokens,
                        output_states,
                        vocab_size=args.synthetic_vocab_size,
                        BLOCK_SIZE=block_size,
                        num_warps=num_warps,
                    )

                launch_single()
                torch.cuda.synchronize()
                _assert_outputs(
                    output_tokens,
                    output_states,
                    expected_tokens,
                    expected_states,
                )
                candidates.append(
                    {
                        "strategy": f"single_w{num_warps}",
                        "cuda_us": float(
                            triton.testing.do_bench(
                                launch_single,
                                warmup=25,
                                rep=args.kernel_repetitions,
                            )
                            * 1_000
                        ),
                    }
                )

            if row_size <= 64:
                for rows_per_program in (2, 4, 8, 16):
                    active_lanes = rows_per_program * block_size
                    recommended_warps = min(
                        8,
                        max(
                            1,
                            triton.next_power_of_2(
                                max(1, active_lanes // 32)
                            ),
                        ),
                    )
                    for num_warps in sorted(
                        {
                            recommended_warps,
                            max(1, recommended_warps // 2),
                        }
                    ):
                        def launch_packed(
                            rows_per_program: int = rows_per_program,
                            num_warps: int = num_warps,
                        ) -> None:
                            triton_csr_argmax_advance_packed(
                                logits,
                                indptr,
                                indices,
                                next_states,
                                rows,
                                max_row_nnz=row_size,
                                rows_per_program=rows_per_program,
                                num_warps=num_warps,
                                output_tokens=output_tokens,
                                output_states=output_states,
                            )

                        launch_packed()
                        torch.cuda.synchronize()
                        _assert_outputs(
                            output_tokens,
                            output_states,
                            expected_tokens,
                            expected_states,
                        )
                        candidates.append(
                            {
                                "strategy": (
                                    f"packed_r{rows_per_program}_w{num_warps}"
                                ),
                                "cuda_us": float(
                                    triton.testing.do_bench(
                                        launch_packed,
                                        warmup=25,
                                        rep=args.kernel_repetitions,
                                    )
                                    * 1_000
                                ),
                            }
                        )
            candidates.sort(key=lambda item: item["cuda_us"])
            sweep.append(
                {
                    "row_nnz": row_size,
                    "batch_size": batch_size,
                    "best_strategy": candidates[0]["strategy"],
                    "best_cuda_us": candidates[0]["cuda_us"],
                    "candidates": candidates,
                }
            )

    payload = {
        "metadata": machine_metadata(device),
        "config": {
            **vars(args),
            "output": str(args.output),
        },
        "qwen3": {
            "grammars": qwen_records,
            "tables": {
                "states": qwen_tables.num_states,
                "edges": int(qwen_tables.csr_indices.size),
                "max_row_nnz": qwen_tables.max_row_nnz,
                "vocab_size": qwen_tables.vocab_size,
                "csr_runtime_bytes": int(
                    qwen_tables.memory_bytes()["csr_tokens"]
                    + qwen_tables.memory_bytes()["csr_next_state"]
                    + qwen_tables.memory_bytes()["state_metadata"]
                ),
                "ell_runtime_bytes": ell_memory_bytes,
            },
            "benchmarks": qwen_benchmarks,
        },
        "row_sweep_seconds": time.perf_counter() - sweep_started,
        "row_sweep": sweep,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _print_summary(payload)


def _benchmark_qwen_ell(
    tables,
    states_np,
    row_lengths,
    token_table,
    next_state_table,
    *,
    buffer_count: int,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> list[dict[str, object]]:
    batch_size = states_np.size
    states = torch.from_numpy(states_np).to(device)
    logits = torch.randn(
        (buffer_count, batch_size, tables.vocab_size),
        dtype=torch.float16,
        device=device,
    )
    starts = tables.csr_indptr[states_np]
    desired = tables.csr_indices[starts]
    desired_next = tables.csr_next_state[starts]
    buffer_ids = torch.arange(buffer_count, device=device)[:, None]
    batch_ids = torch.arange(batch_size, device=device)[None, :]
    desired_device = torch.from_numpy(desired).to(device)[None, :]
    logits[buffer_ids, batch_ids, desired_device] = 20.0
    output_tokens = torch.empty(
        batch_size,
        dtype=torch.int32,
        device=device,
    )
    output_states = torch.empty_like(states)
    plan = make_ell_argmax_advance_plan(
        logits[0],
        row_lengths,
        token_table,
        next_state_table,
        states,
        output_tokens=output_tokens,
        output_states=output_states,
    )
    source = CyclingLogits(logits)
    source.reset()
    plan_timing = measure(
        lambda: plan(source.next())[1],
        warmup=warmup,
        iterations=iterations,
        measure_cuda=True,
    )
    graph_source = CyclingGraphs(
        [plan.capture(logits[index]) for index in range(buffer_count)]
    )
    graph_source.reset()
    graph_timing = measure(
        graph_source.next,
        warmup=warmup,
        iterations=iterations,
        measure_cuda=True,
    )
    if not torch.equal(output_tokens, desired_device[0]):
        raise AssertionError("ELL kernel selected an incorrect token")
    if not torch.equal(
        output_states,
        torch.from_numpy(desired_next).to(device),
    ):
        raise AssertionError("ELL kernel selected an incorrect next state")
    common = {
        "mix": "qwen3_heterogeneous",
        "batch_size": batch_size,
        "unique_grammars": 2,
        "row_nnz_min": int(tables.row_nnz[states_np].min()),
        "row_nnz_mean": float(tables.row_nnz[states_np].mean()),
        "row_nnz_max": int(tables.row_nnz[states_np].max()),
        "model_vocab_density_mean": float(
            tables.row_nnz[states_np].mean() / tables.vocab_size
        ),
        "config_depth_min": int(tables.config_depths[states_np].min()),
        "config_depth_mean": float(tables.config_depths[states_np].mean()),
        "config_depth_max": int(tables.config_depths[states_np].max()),
        "logits_buffer_count": buffer_count,
        "launch_strategy": plan.strategy,
    }
    return [
        {
            **common,
            "strategy": "bounded_lr1_qwen3_ell_plan",
            **plan_timing.__dict__,
            "sequences_per_second": float(
                batch_size / (plan_timing.cuda_mean_us * 1e-6)
            ),
        },
        {
            **common,
            "strategy": "bounded_lr1_qwen3_ell_cuda_graph",
            **graph_timing.__dict__,
            "sequences_per_second": float(
                batch_size / (graph_timing.cuda_mean_us * 1e-6)
            ),
        },
    ]


def _assert_outputs(
    tokens: torch.Tensor,
    states: torch.Tensor,
    expected_tokens: torch.Tensor,
    expected_states: torch.Tensor,
) -> None:
    if not torch.equal(tokens, expected_tokens):
        raise AssertionError("CSR kernel selected incorrect tokens")
    if not torch.equal(states, expected_states):
        raise AssertionError("CSR kernel selected incorrect states")


def _print_summary(payload: dict[str, object]) -> None:
    print("Qwen3 optimized launch paths")
    print("batch,strategy,launch,cuda_us,wall_p50_us")
    for item in payload["qwen3"]["benchmarks"]:
        print(
            f"{item['batch_size']},{item['strategy']},"
            f"{item['launch_strategy']},{item['cuda_mean_us']:.3f},"
            f"{item['wall_p50_us']:.3f}"
        )
    print("row_nnz,batch,best_strategy,best_cuda_us")
    for item in payload["row_sweep"]:
        print(
            f"{item['row_nnz']},{item['batch_size']},"
            f"{item['best_strategy']},{item['best_cuda_us']:.3f}"
        )


if __name__ == "__main__":
    main()
