#!/usr/bin/env python3
"""Every feature of `runtime` reaches `worker`, and every feature of `worker`
reaches `pie`: a flavor a library offers is only real once the binary can
select it, and the forwards are copied by hand across three manifests. Read
from `cargo metadata`, so this sees what Cargo resolved."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORWARDS = [("runtime", "worker"), ("worker", "pie")]
# `feature -> why it is not forwarded`; an entry that stops being true fails.
EXCUSED = {
    ("worker", "pie"): {
        "nixl": "transport's NIXL engine is a stub; forwarding it would put a flag on the CLI that turns on nothing",
    },
}


def main() -> int:
    out = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    features = {p["name"]: p["features"] for p in json.loads(out.stdout)["packages"]}
    problems = []
    for inner, outer in FORWARDS:
        forwarded = {
            edge.split("/", 1)[1]
            for values in features[outer].values()
            for edge in values
            if edge.startswith(f"{inner}/")
        }
        excused = EXCUSED.get((inner, outer), {})
        for feature in sorted(features[inner]):
            if feature != "default" and feature not in forwarded and feature not in excused:
                problems.append(f"`{inner}/{feature}` is not forwarded by `{outer}`")
        for feature in sorted(excused):
            if feature not in features[inner]:
                problems.append(f"`{inner}/{feature}` is excused but no longer declared; drop the entry")
            elif feature in forwarded:
                problems.append(f"`{inner}/{feature}` is excused and forwarded; drop the entry")
    for p in problems:
        print(f"feature-forward-audit: {p}", file=sys.stderr)
    if not problems:
        print("feature-forward-audit: every library feature reaches the binary")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
