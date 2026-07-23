#!/usr/bin/env python3

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPLICATIONS = ROOT / "tests" / "policies" / "replications"
PAPERS = ROOT / "plex-serving-policy-wiki" / "papers"
CATALOG = ROOT / "plex-serving-policy-wiki" / "catalog.json"
START = "<!-- plex-v0.6-replication:start -->"
END = "<!-- plex-v0.6-replication:end -->"
SLUGS = {
    "agentix": "autellix-agentix",
    "continuum": "continuum",
    "kvflow": "kvflow",
    "preble": "preble",
    "helium": "helium",
    "vtc": "vtc",
    "lmetric": "lmetric",
    "fairserve": "fairserve",
    "marconi": "marconi",
    "ragcache": "ragcache",
    "dlpm": "dlpm-d2lpm",
    "infercept": "infercept",
    "peek": "peek",
    "qlm": "qlm",
    "slos-serve": "slos-serve",
    "dynasor": "certaindex-dynasor",
    "justitia": "justitia",
    "chameleon": "chameleon",
    "hotprefix": "hotprefix",
    "pard": "pard",
    "branch-regulation": "regulating-branch-parallelism-in-llm-serving",
    "dualmap": "dualmap",
    "llumnix": "llumnix",
    "smetric": "smetric",
    "thunderagent": "thunderagent",
    "pythia": "pythia",
    "goodserve": "goodserve",
    "conserve": "observation-not-prediction-conserve",
    "parrot": "parrot",
    "saga": "saga",
    "routebalance": "routebalance",
}


def main() -> None:
    index = read_json(REPLICATIONS / "index.json")
    catalog = read_json(CATALOG)
    catalog_by_slug = {entry["slug"]: entry for entry in catalog}
    for replication_id in index["replications"]:
        metadata = read_json(REPLICATIONS / replication_id / "metadata.json")
        paper_slug = SLUGS[replication_id]
        status = {
            "replication_id": replication_id,
            "component": metadata["component"],
            "operations": metadata["implements"],
            "evidence_level": metadata["evidence_level"],
            "validation_status": metadata["validation_status"],
            "deferred_mechanics": metadata["deferred_mechanics"],
            "metadata_path": f"tests/policies/replications/{replication_id}/metadata.json",
        }
        catalog_by_slug[paper_slug]["plex_v0_6_replication"] = status
        update_page(PAPERS / f"{paper_slug}.md", status)
    CATALOG.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n")


def update_page(path: Path, status: dict) -> None:
    text = path.read_text()
    block = "\n".join(
        [
            START,
            "## PLEX v0.6 replication status",
            "",
            f"- Component: `{status['component']}`",
            f"- Operations: {', '.join(f'`{op}`' for op in status['operations'])}",
            f"- Evidence: `{status['evidence_level']}`",
            f"- Validation: `{status['validation_status']}`",
            f"- Metadata: [`{status['metadata_path']}`](../../{status['metadata_path']})",
            "- Deferred mechanics: "
            + (
                "; ".join(status["deferred_mechanics"])
                if status["deferred_mechanics"]
                else "None"
            ),
            END,
            "",
            "",
        ]
    )
    if START in text:
        prefix, rest = text.split(START, 1)
        _, suffix = rest.split(END, 1)
        text = prefix + block + suffix.lstrip("\n")
    else:
        marker = "## Suggested citation"
        if marker not in text:
            raise SystemExit(f"{path}: missing insertion marker")
        text = text.replace(marker, block + marker, 1)
    path.write_text(text)


def read_json(path: Path):
    return json.loads(path.read_text())


if __name__ == "__main__":
    main()
