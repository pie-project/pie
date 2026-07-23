#!/usr/bin/env python3

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPLICATIONS = ROOT / "tests" / "policies" / "replications"
FIDELITY = ROOT / "tests" / "policies" / "fidelity-audit.json"
REPORT_JSON = ROOT / "tests" / "policies" / "replication-report.json"
REPORT_MD = ROOT / "plex_replication_report.md"
REQUIRED = {
    "id",
    "title",
    "paper_version",
    "source_url",
    "source_license",
    "component",
    "implements",
    "policy_kernel",
    "required_facts",
    "required_mechanics",
    "deferred_mechanics",
    "evidence_level",
    "fixture_provenance",
    "validation_status",
}
EVIDENCE = {
    "end-to-end-source-replication",
    "decision-trace-parity-with-deferred-mechanics",
    "policy-kernel-reproduction",
    "inspired-adaptation",
}


def main() -> None:
    index = read_json(REPLICATIONS / "index.json")
    fidelity = {
        entry["id"]: entry for entry in read_json(FIDELITY)["entries"]
    }
    entries = []
    for slug in index["replications"]:
        metadata = read_json(REPLICATIONS / slug / "metadata.json")
        missing = REQUIRED - metadata.keys()
        if missing:
            raise SystemExit(f"{slug}: missing metadata fields {sorted(missing)}")
        if metadata["id"] != slug:
            raise SystemExit(f"{slug}: metadata id mismatch")
        if metadata["evidence_level"] not in EVIDENCE:
            raise SystemExit(f"{slug}: unknown evidence level")
        if metadata["validation_status"] != "passing":
            raise SystemExit(f"{slug}: validation is not passing")
        if slug not in fidelity:
            raise SystemExit(f"{slug}: missing independent fidelity audit")
        read_json(REPLICATIONS / slug / "cases" / "basic.json")
        read_json(REPLICATIONS / slug / "expected" / "basic.json")
        entries.append(
            {
                **metadata,
                "fidelity": {
                    key: fidelity[slug][key]
                    for key in ("classification", "confidence", "summary")
                },
                "case": f"tests/policies/replications/{slug}/cases/basic.json",
                "expected": f"tests/policies/replications/{slug}/expected/basic.json",
            }
        )

    evidence = Counter(entry["evidence_level"] for entry in entries)
    operations = Counter(
        operation for entry in entries for operation in entry["implements"]
    )
    report = {
        "contract": index["contract"],
        "candidate_count": len(entries),
        "smoke_passing_count": sum(
            entry["validation_status"] == "passing" for entry in entries
        ),
        "evidence_counts": dict(sorted(evidence.items())),
        "fidelity_counts": dict(
            sorted(
                Counter(
                    entry["fidelity"]["classification"]
                    for entry in entries
                ).items()
            )
        ),
        "operation_counts": dict(sorted(operations.items())),
        "entries": entries,
    }
    write_json(REPORT_JSON, report)
    REPORT_MD.write_text(markdown(report))


def markdown(report: dict) -> str:
    lines = [
        "# PLEX v0.6 Replication and Fidelity Report",
        "",
        "This report is generated from committed replication metadata by",
        "`scripts/generate-plex-replication-report.py`.",
        "",
        f"- Candidates: {report['candidate_count']}",
        f"- Runtime smoke passing: {report['smoke_passing_count']}",
        f"- Contract: `{report['contract']['major']}.{report['contract']['minor']}`",
        "",
        "## Evidence",
        "",
    ]
    for level, count in report["evidence_counts"].items():
        lines.append(f"- `{level}`: {count}")
    lines.extend(
        [
            "",
            "## Independent fidelity",
            "",
        ]
    )
    for classification, count in report["fidelity_counts"].items():
        lines.append(f"- `{classification}`: {count}")
    lines.extend(
        [
            "",
            "## Candidates",
            "",
            "| ID | Title | Operations | Evidence | Fidelity | Deferred mechanics |",
            "|---|---|---|---|---|---:|",
        ]
    )
    for entry in report["entries"]:
        title = entry["title"].replace("|", "\\|")
        operations = ", ".join(f"`{operation}`" for operation in entry["implements"])
        lines.append(
            f"| `{entry['id']}` | {title} | {operations} | "
            f"`{entry['evidence_level']}` | "
            f"`{entry['fidelity']['classification']}` | "
            f"{len(entry['deferred_mechanics'])} |"
        )
    lines.extend(
        [
            "",
            "Runtime smoke means the package loads and its committed fixture path is",
            "covered by the release suite. It is not a claim of paper fidelity.",
            "Independent reviewers found no faithful or faithful-with-deferred-mechanics",
            "implementation; all entries are therefore classified as inspired",
            "adaptations until paper/artifact differential traces pass.",
            "",
        ]
    )
    return "\n".join(lines)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
