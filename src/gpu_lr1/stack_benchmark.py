from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gpu_lr1.benchmark import machine_metadata, measure
from gpu_lr1.kernels import triton_stack_update


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark divergent GPU stack pointer layouts"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[32, 128, 512, 2048, 8192],
    )
    parser.add_argument("--max-depth", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/stack-layout.json"),
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    records = []

    for batch_size in args.batch_sizes:
        row_stack = torch.randint(
            0,
            64,
            (batch_size, args.max_depth),
            dtype=torch.int32,
            device=device,
        )
        depth_stack = row_stack.transpose(0, 1).contiguous()
        actions = torch.tensor(
            np.resize(np.asarray([-1, 0, 1], dtype=np.int32), batch_size),
            device=device,
        )
        push_symbols = torch.arange(
            batch_size,
            dtype=torch.int32,
            device=device,
        )
        output_pointers = torch.empty_like(actions)
        output_tops = torch.empty_like(actions)

        for distribution in ("same", "narrow", "wide"):
            pointers = make_stack_pointers(
                batch_size,
                args.max_depth,
                distribution,
                device,
            )
            expected_tops = row_stack[
                torch.arange(batch_size, device=device),
                pointers.long() - 1,
            ]
            expected_pointers = torch.clamp(
                pointers + actions,
                min=1,
                max=args.max_depth,
            )

            for layout, stack in (("row", row_stack), ("depth", depth_stack)):
                function = lambda layout=layout, stack=stack: triton_stack_update(
                    stack,
                    pointers,
                    actions,
                    push_symbols,
                    layout=layout,
                    output_pointers=output_pointers,
                    output_tops=output_tops,
                )
                produced_pointers, produced_tops = function()
                torch.cuda.synchronize()
                if not torch.equal(produced_pointers, expected_pointers):
                    raise AssertionError("stack pointer update mismatch")
                if not torch.equal(produced_tops, expected_tops):
                    raise AssertionError("stack top lookup mismatch")
                timing = measure(
                    function,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    measure_cuda=True,
                )
                records.append(
                    {
                        "batch_size": batch_size,
                        "max_depth": args.max_depth,
                        "distribution": distribution,
                        "layout": layout,
                        "wall_p50_us": timing.wall_p50_us,
                        "wall_p95_us": timing.wall_p95_us,
                        "cuda_mean_us": timing.cuda_mean_us,
                        "updates_per_second": float(
                            batch_size / (timing.cuda_mean_us * 1e-6)
                        ),
                    }
                )

    payload = {
        "metadata": machine_metadata(device),
        "config": {
            **vars(args),
            "output": str(args.output),
        },
        "benchmarks": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("batch,distribution,layout,cuda_us,wall_p50_us,updates_per_second")
    for item in records:
        print(
            f"{item['batch_size']},{item['distribution']},{item['layout']},"
            f"{item['cuda_mean_us']:.3f},{item['wall_p50_us']:.3f},"
            f"{item['updates_per_second']:.1f}"
        )


def make_stack_pointers(
    batch_size: int,
    max_depth: int,
    distribution: str,
    device: torch.device,
) -> torch.Tensor:
    if distribution == "same":
        return torch.full(
            (batch_size,),
            max_depth // 2,
            dtype=torch.int32,
            device=device,
        )
    generator = torch.Generator(device=device)
    generator.manual_seed(batch_size * 13)
    if distribution == "narrow":
        center = max_depth // 2
        return torch.randint(
            center - 4,
            center + 5,
            (batch_size,),
            dtype=torch.int32,
            device=device,
            generator=generator,
        )
    if distribution == "wide":
        return torch.randint(
            1,
            max_depth,
            (batch_size,),
            dtype=torch.int32,
            device=device,
            generator=generator,
        )
    raise ValueError(f"unknown distribution: {distribution}")


if __name__ == "__main__":
    main()

