"""Which points each SKU states, and which of them each plane claims.

WHY THIS IS A SCRIPT AND NOT A PARAGRAPH. This table was derived by hand
four times in one day and was wrong three of those times, in both
directions:

  * `vulkan qwen3.5 16/21` was reported for hours after an ssm merge had
    closed it to 21/21. A hand table goes stale the moment work lands and
    nothing tells you.
  * `cuda gemma4 22/23` was a REGEX artifact. A tier-2 point is a
    plane-private inherent method and is spelled `cuda::foo` in a plan
    where a tier-1 point is `family.point`; a pattern written for the
    second cannot see the first, and the hole it reports is in the tool.
    cuda covers gemma-4 completely, which its banked argmax already said.

Both failures have the same shape as the one this tree keeps finding in
its own tests -- something green that measured nothing -- so the fix is
the same: derive it, do not assert it.

WHERE THE TWO SIDES COME FROM.

  STATED: `model::catalog()`'s row for the SKU, traced at the plane, via
  `cargo run -p model --bin trace`. Every `ops[].kernel` in the resulting
  plan. This is the real plan the driver would walk, not a reading of the
  model text.

  CLAIMED: the `CLAIMED` and `TIER2` tables in each plane's generated
  `points_dispatch.rs`, which `points-dispatch` writes from the
  `#[claims] impl Family for Ctx<'_>` blocks. Parsed as text because
  these are four crates a Python script cannot link, and because the
  generated file is the same artifact the dispatch itself reads.

A HOLE HERE IS NOT A BUG. An unclaimed point is a default body returning
`Refusal::Absent`, which is this architecture's measured backlog: the row
refuses to bind, loudly, instead of binding and computing something else.
The number is the backlog's size, and the names are the work.
"""

import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

PLANES = ["cuda", "metal", "wgpu", "vulkan"]

# The three SKUs with banked argmaxes -- the ones a plane can be checked
# against rather than merely bound for. `scripts/banked-argmaxes.sh` is
# where those answers are gates.
BANKED = [
    ("qwen3.5", "qwen35-d0.8b-bf16-kv-bf16"),
    ("gpt-oss", "gptoss-20b-bf16-mxfp4-kv-bf16"),
    ("gemma-4", "gemma4-e4b-bf16-kv-bf16"),
]


def claimed(plane):
    """Every point name in one plane's generated dispatch tables."""
    text = (ROOT / f"crates/kernels-{plane}/src/points_dispatch.rs").read_text()
    names = set()
    for table in ("CLAIMED", "TIER2"):
        found = re.search(rf"pub const {table}[^=]*= &\[(.*?)\n\];", text, re.S)
        if found:
            names |= set(re.findall(r'\("([^"]+)"', found.group(1)))
    return names


def stated(sku, plane):
    """Every point the SKU's plan states at that plane."""
    run = subprocess.run(
        ["cargo", "run", "-q", "-p", "model", "--bin", "trace", "--", sku, plane],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    if run.returncode != 0:
        raise SystemExit(f"trace {sku} {plane} failed:\n{run.stderr}")
    return {op["kernel"] for op in json.loads(run.stdout)["ops"]}


def holes(points, names):
    """The stated points this plane does not claim.

    A tier-2 point is `<plane>::<name>` in a plan and bare in `TIER2`.
    Comparing the two spellings directly is the mistake this file's head
    describes, so the plane qualifier comes off first.
    """
    return sorted(p for p in points if p.split("::")[-1] not in names)


def main():
    skus = BANKED if len(sys.argv) == 1 else [(s, s) for s in sys.argv[1:]]
    tables = {plane: claimed(plane) for plane in PLANES}

    print(f"{'':8}" + "".join(f"{label:>12}" for label, _ in skus))
    backlog = {}
    for plane in PLANES:
        cells = []
        for label, sku in skus:
            points = stated(sku, plane)
            missing = holes(points, tables[plane])
            cells.append(f"{len(points) - len(missing)}/{len(points)}")
            if missing:
                backlog[(plane, label)] = missing
        print(f"{plane:8}" + "".join(f"{c:>12}" for c in cells))

    print()
    for plane in PLANES:
        print(f"{plane:8} {len(tables[plane])} claims")

    if not backlog:
        print("\nevery point every banked SKU states is claimed on every plane.")
        return
    print()
    for (plane, label), missing in sorted(backlog.items()):
        print(f"{plane} does not claim, for {label}:")
        for name in missing:
            print(f"    {name}")


if __name__ == "__main__":
    main()
