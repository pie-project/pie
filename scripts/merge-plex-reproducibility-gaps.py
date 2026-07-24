#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "tests" / "policies" / "replications" / "index.json"
PRIMITIVE_ALIASES = {
    "cache.transfer-batch@1": "cache.move@1",
    "cache.atomic-tier-exchange@1": "cache.move@1",
    "request.preempt-requeue@1": "request.pause-resume@1",
    "request.pause@1": "request.pause-resume@1",
    "request.resume@1": "request.pause-resume@1",
    "model.activate@1": "target.provision@1",
    "model.scale@1": "target.provision@1",
    "schedule.exact-forward-batch@1": "schedule.co-execute@1",
    "request.early-complete@1": "request.complete-with-output@1",
    "target.membership-control@1": "target.provision@1",
    "request.migration-lifecycle@1": "request.migrate@1",
    "tool.environment.lifecycle@1": "tool.resource@1",
    "workflow.dependency-runtime@1": "workflow.graph@1",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    by_id = {}
    for path in args.input:
        for entry in read_json(path)["entries"]:
            policy_id = entry["id"]
            if policy_id in by_id:
                raise SystemExit(f"duplicate policy {policy_id}")
            normalized = []
            seen_primitives = set()
            for primitive in entry["new_primitives_required"]:
                primitive["primitive"] = PRIMITIVE_ALIASES.get(
                    primitive["primitive"],
                    primitive["primitive"],
                )
                if primitive["primitive"] in seen_primitives:
                    continue
                seen_primitives.add(primitive["primitive"])
                normalized.append(primitive)
            entry["new_primitives_required"] = normalized
            by_id[policy_id] = entry
    order = read_json(INDEX)["replications"]
    missing = set(order) - by_id.keys()
    extra = by_id.keys() - set(order)
    if missing or extra:
        raise SystemExit(
            f"gap matrix mismatch: missing={sorted(missing)} extra={sorted(extra)}"
        )
    entries = [by_id[policy_id] for policy_id in order]
    ceilings = Counter(
        entry["best_achievable_evidence_with_current_model"]
        for entry in entries
    )
    report = {
        "schema_version": 1,
        "policy_count": len(entries),
        "evidence_ceiling_counts": dict(sorted(ceilings.items())),
        "entries": entries,
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
