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

ASK IT ABOUT THE WHOLE CATALOG, not only the three banked rows. With no
arguments it covers the rows a plane can be CHECKED against an argmax; named
rows cover the rest, and that is where the work is. Measured over every
non-`tp2` row on 2026-08-25:

              dsv4  gemma4-e4b  gemma4-31b  glm5  gptoss-20b  gptoss-120b  kimik3  qwen35-a3b  qwen35-d3b  qwen35-d0.8b
  cuda       23/24       23/23       19/19 22/22       16/16        16/16   22/23       25/25       21/21         21/21
  metal      13/24       22/22       18/18 11/22       16/16        16/16   15/23       25/25       21/21         21/21
  wgpu       12/24       22/22       18/18 11/22       16/16        16/16   15/23       25/25       21/21         21/21
  vulkan      9/24       22/22       18/18 10/22       16/16        16/16   11/23       25/25       21/21         21/21

AND THE `-tp2` ROWS ARE ONE POINT AWAY, ALL SIX OF THEM. Every tensor-parallel
row states exactly one point its single-rank sibling does not, and it is the
same point every time:

  cuda    dsv4-tp2 2, gemma4-31b-tp2 1, glm5-tp2 1, gptoss-120b-tp2 1,
          kimik3-tp2 2, qwen35-a3b-tp2 1

-- the extras over the table above being `dist.all_reduce`, once per row. NO
PLANE CLAIMS IT: `kernels-{cuda,metal,wgpu}/src/dist.rs` are each
`impl kernels::points::Dist for Ctx<'_> {}`, an empty claim whose default body
returns `Refusal::Absent`. So a walk over any tp2 tower refuses at its first
all-reduce, on every plane, and one point stands between this tree and every
tensor-parallel row it lists.

WHAT EACH OF cuda's THREE HOLES ACTUALLY NEEDS, since "write the kernel" is
wrong for all three and the reason differs each time:

  * `dist.all_reduce` — the kernel EXISTS.
    `kernels_cuda::comm::all_reduce_bf16` is a whole custom all-reduce with a
    two-shot split and a residual-rmsnorm fusion, and
    `driver-cuda/src/fire/all_reduce.rs` drives it. The claim has one question
    to answer: the point is in-place (`InOut<Tensor<T>>`) and the comm entry
    point takes a separate input and output, so does the custom kernel tolerate
    aliasing them. Verifying the answer needs two ranks, and this box has one
    GPU.

  * `hc.collapse` — the POINT and the KERNEL disagree about OPERANDS.
    `norm/dsv4_hc.cuh`'s `hc_head_postprocess` computes
    `gate_i = sigmoid(mixes[n][i] * scale + base[i]) + eps`, and neither the
    point nor `model_dsl::kernels::hc::collapse` carries a `mixes [N, M]`. The
    sibling `hc.gates` does take it, as `normed`. So the declaration cannot
    express the kernel's arithmetic — `head_scale` has nothing to scale — and
    what `collapse` is meant to compute instead is not written down anywhere.

  * `norm.res_blend` — the POINT and the RECORDED STATEMENT disagree about
    ARITY, which `model_dsl::kernels::norm::res_blend` names in its own doc as
    an open ledger item: "`blocks` grows by one every layer that blends, so
    this records one value per block where `norm.res_blend` states the single
    concatenated rectangle its routine takes." No kernel is missing because no
    settled shape exists to write one against.

None of the three is closable by measuring something on this box. THE
CHECKPOINT IS NOT THE BLOCKER for two of them, which an earlier reading of this
had wrong: dsv4 and kimik3 being uncached matters for an end-to-end argmax and
not for these, since a point that is pure arithmetic can be checked against a
host reference — `driver-cuda/tests/moe_unrouted.rs` and
`driver-metal/tests/device_routing.rs` both do exactly that. What stops all
three is that the shape of the claim is not settled, and settling it by guessing
would be asserting rather than measuring.

Seven of the ten are complete on all four planes. The three that are not are
three FAMILIES rather than scattered points -- `hc.*` and `pool.*` for dsv4,
`mla.*` and `index.*` for glm5 and kimik3 -- which is what a backlog looks like
when it is whole subsystems and not drift. cuda's two are single points.

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

    # THE COLUMN IS AS WIDE AS ITS HEADING. Fixed at 12 this ran the catalog's
    # own row ids together into one unreadable line -- a tool whose whole job is
    # to report, reporting nothing legible, which is the same failure as a green
    # test that measured nothing.
    width = [max(len(label) + 2, 12) for label, _ in skus]
    print(f"{'':8}" + "".join(f"{l:>{w}}" for (l, _), w in zip(skus, width)))
    backlog = {}
    for plane in PLANES:
        cells = []
        for label, sku in skus:
            points = stated(sku, plane)
            missing = holes(points, tables[plane])
            cells.append(f"{len(points) - len(missing)}/{len(points)}")
            if missing:
                backlog[(plane, label)] = missing
        print(f"{plane:8}" + "".join(f"{c:>{w}}" for c, w in zip(cells, width)))

    print()
    for plane in PLANES:
        print(f"{plane:8} {len(tables[plane])} claims")

    if not backlog:
        print("\nevery point every SKU asked about is claimed on every plane.")
        return
    # THE SIZE OF THE BACKLOG, BEFORE ITS NAMES. Ten catalog rows across four
    # planes is more refusals than fit on a screen, and the first question about
    # a backlog is how big it is, not what the first line of it says.
    print()
    for plane in PLANES:
        rows = [(l, len(m)) for (p, l), m in sorted(backlog.items()) if p == plane]
        total = sum(n for _, n in rows)
        if not rows:
            print(f"{plane:8} claims every point every row above states")
            continue
        named = ", ".join(f"{l} {n}" for l, n in rows)
        print(f"{plane:8} {total} unclaimed over {len(rows)} row(s): {named}")
    print()
    for (plane, label), missing in sorted(backlog.items()):
        print(f"{plane} does not claim, for {label}:")
        for name in missing:
            print(f"    {name}")


if __name__ == "__main__":
    main()
