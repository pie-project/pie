#!/usr/bin/env python3

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPLICATIONS = ROOT / "tests" / "policies" / "replications"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: generate-plex-replication-artifacts.py definitions.json")
    definitions_path = Path(sys.argv[1])
    definitions = json.loads(definitions_path.read_text())
    index_path = REPLICATIONS / "index.json"
    index = json.loads(index_path.read_text())
    known = list(index["replications"])
    for definition in definitions:
        slug = definition["metadata"]["id"]
        root = REPLICATIONS / slug
        (root / "cases").mkdir(parents=True, exist_ok=True)
        (root / "expected").mkdir(parents=True, exist_ok=True)
        write_json(root / "metadata.json", definition["metadata"])
        write_json(root / "cases" / "basic.json", definition["case"])
        write_json(root / "expected" / "basic.json", definition["expected"])
        if slug not in known:
            known.append(slug)
    index["replications"] = known
    write_json(index_path, index)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
