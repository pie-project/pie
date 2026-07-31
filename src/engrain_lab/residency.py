"""What does one compiled schema cost to keep resident?

The case for device residency is paid for in memory: the translation from a
model's tokens to a grammar's terminals has to be there, where a host-side
matcher recomputes it every step and keeps only kilobytes. This measures the
price over the whole corpus, so it can be compared against XGrammar's compiler
cache rather than asserted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--instances", type=Path, default=Path("results/jsonschemabench-instances.json")
    )
    parser.add_argument(
        "--reference-kib",
        type=float,
        default=52.0,
        help="XGrammar's compiler cache, for comparison",
    )
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()

    import engrain
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(arguments.model)
    vocabulary: list[bytes] = []
    for token_id in range(len(tokenizer)):
        piece = tokenizer.convert_ids_to_tokens(token_id)
        try:
            vocabulary.append(tokenizer.convert_tokens_to_string([piece]).encode())
        except Exception:  # noqa: BLE001
            vocabulary.append(b"")

    compiler = engrain.Compiler(vocabulary)
    instances = json.loads(arguments.instances.read_text())["instances"]

    sizes: list[float] = []
    groups: list[int] = []
    for instance in instances:
        try:
            compiled = compiler.compile_json_schema(instance["schema"])
        except Exception:  # noqa: BLE001
            continue
        sizes.append(compiled.resident_bytes / 1024.0)
        groups.append(compiled.num_groups)

    ordered = np.array(sorted(sizes))
    print(
        f"compiled {len(ordered)} of {len(instances)} schemas "
        f"against a {len(vocabulary)}-token vocabulary"
    )
    print("resident size per schema, KiB:")
    percentiles = {}
    for quantile in (25, 50, 75, 90, 99):
        value = float(np.percentile(ordered, quantile))
        percentiles[quantile] = value
        print(f"  p{quantile:<2}: {value:>10.1f}")
    print(f"  max: {ordered.max():>10.1f}")
    print(f"  all {len(ordered)} together: {ordered.sum() / 1024:.1f} MiB")
    within = int((ordered <= arguments.reference_kib).sum())
    print(
        f"\nat or below XGrammar's ~{arguments.reference_kib:.0f} KiB cache: "
        f"{within} of {len(ordered)} ({100.0 * within / len(ordered):.0f}%)"
    )

    if arguments.output:
        arguments.output.write_text(
            json.dumps(
                {
                    "model": arguments.model,
                    "vocab_size": len(vocabulary),
                    "compiled": len(ordered),
                    "schemas": len(instances),
                    "resident_kib": {
                        "p25": percentiles[25],
                        "p50": percentiles[50],
                        "p75": percentiles[75],
                        "p90": percentiles[90],
                        "p99": percentiles[99],
                        "max": float(ordered.max()),
                        "total_mib": float(ordered.sum() / 1024),
                    },
                    "median_groups": int(np.median(groups)),
                    "within_reference": within,
                },
                indent=2,
            )
        )
        print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
