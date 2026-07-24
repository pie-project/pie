#!/usr/bin/env python3

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def build_report(
    taxonomy: dict[str, Any],
    matrix: dict[str, Any],
) -> dict[str, Any]:
    entries = matrix["entries"]
    evidence = Counter(
        entry["best_achievable_evidence_with_current_model"]
        for entry in entries
    )
    primitive_policies: dict[str, set[str]] = defaultdict(set)
    for entry in entries:
        for primitive in entry["new_primitives_required"]:
            primitive_policies[primitive["primitive"]].add(entry["id"])
    primitive_catalog = []
    known = {
        primitive["id"]: primitive
        for primitive in taxonomy["candidate_primitives"]
    }
    for primitive_id, policies in sorted(
        primitive_policies.items(),
        key=lambda item: (-len(item[1]), item[0]),
    ):
        primitive_catalog.append(
            {
                **known.get(
                    primitive_id,
                    {
                        "id": primitive_id,
                        "kind": "policy-proposed",
                        "minimal_contract": "See per-policy analysis.",
                        "why": "Proposed by independent fidelity review.",
                    },
                ),
                "policies": sorted(policies),
                "policy_count": len(policies),
            }
        )
    return {
        "schema_version": 1,
        "policy_count": len(entries),
        "current_model_only_count": sum(
            not entry["new_primitives_required"] for entry in entries
        ),
        "new_primitive_count": len(primitive_catalog),
        "evidence_ceiling_counts": dict(sorted(evidence.items())),
        "taxonomy": taxonomy["axes"],
        "current_capabilities": taxonomy["current_capabilities"],
        "primitive_catalog": primitive_catalog,
        "entries": entries,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PLEX 31-policy reproducibility roadmap",
        "",
        "This roadmap separates three causes of missing fidelity:",
        "",
        "1. **Current-model improvement**: rewrite the policy, facts, state loop,",
        "   adapter, or benchmark without changing the PLEX contract.",
        "2. **New primitive required**: add authority, atomicity, lifecycle, or an",
        "   enacted mechanism that the current contract cannot safely express.",
        "3. **Implementation gap**: a bug, simplification, missing state transition,",
        "   wrong equation, or non-diagnostic benchmark.",
        "",
        f"- Policies: {report['policy_count']}",
        f"- Policies needing no new primitive for their best current-model evidence: "
        f"{report['current_model_only_count']}",
        f"- Deduplicated proposed primitives: {report['new_primitive_count']}",
        "",
        "## Evidence ceiling with the current programming model",
        "",
    ]
    for level, count in report["evidence_ceiling_counts"].items():
        lines.append(f"- `{level}`: {count}")
    lines.extend(
        [
            "",
            "## Primitive roadmap",
            "",
            "| Primitive | Kind | Policies | Minimal contract |",
            "|---|---|---:|---|",
        ]
    )
    for primitive in report["primitive_catalog"]:
        contract = primitive["minimal_contract"].replace("|", "\\|")
        lines.append(
            f"| `{primitive['id']}` | {primitive['kind']} | "
            f"{primitive['policy_count']} | {contract} |"
        )
    lines.extend(
        [
            "",
            "## Per-policy summary",
            "",
            "| Policy | Current-model improvements | New primitives | P0 gaps | Current-model ceiling |",
            "|---|---:|---|---:|---|",
        ]
    )
    for entry in report["entries"]:
        primitives = ", ".join(
            f"`{primitive['primitive']}`"
            for primitive in entry["new_primitives_required"]
        ) or "-"
        p0 = sum(
            gap["severity"] == "P0"
            for gap in entry["implementation_gaps"]
        )
        lines.append(
            f"| `{entry['id']}` | {len(entry['current_model_improvements'])} | "
            f"{primitives} | {p0} | "
            f"`{entry['best_achievable_evidence_with_current_model']}` |"
        )
    for entry in report["entries"]:
        lines.extend(
            [
                "",
                f"## {entry['id']}",
                "",
                "### Improve with the current model",
                "",
            ]
        )
        if entry["current_model_improvements"]:
            for item in entry["current_model_improvements"]:
                lines.append(
                    f"- **{item['gap']}** — {item['change']} "
                    f"({item['why_expressible_now']})"
                )
        else:
            lines.append("- None identified.")
        lines.extend(["", "### New primitives", ""])
        if entry["new_primitives_required"]:
            for item in entry["new_primitives_required"]:
                lines.append(
                    f"- **`{item['primitive']}`** — {item['why_required']} "
                    f"Minimal contract: {item['minimal_contract']}"
                )
        else:
            lines.append("- No new primitive is required for the stated evidence ceiling.")
        lines.extend(["", "### Implementation gaps", ""])
        for gap in entry["implementation_gaps"]:
            lines.append(
                f"- **{gap['severity']} — {gap['gap']}**: {gap['fix']}"
            )
        lines.extend(
            [
                "",
                f"**Best evidence without new primitives:** "
                f"`{entry['best_achievable_evidence_with_current_model']}`",
                "",
                "**Recommended sequence:**",
            ]
        )
        for step in entry["recommended_sequence"]:
            lines.append(f"1. {step}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        read_json(args.taxonomy),
        read_json(args.matrix),
    )
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    args.output_md.write_text(markdown(report))


if __name__ == "__main__":
    main()
