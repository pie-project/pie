#!/usr/bin/env python3
"""mkfull.py <outdir> <plan.json> [levers K:V,...]: the wafer program's whole build directory (layout.csl and one
source a PE kind), from a plan through the Rust `split-geo`, with the kernels in programs/split (or $SPLIT_SRC).

The levers and diagnostics (CYV, GELUT, WSPEC, SLW, RING, DIAG, TSTAMP, ...; `<kind>.K` for one kind) set the kernels'
params, defaulting to the shipped build. Everything is resolved here: link.sh compiles <outdir> with no shape params
and puts the artifact the engine runs (<outdir>/geo's program.json, host.json, bindings.json, rcfg arrays) beside the
compiled program. <outdir>/geo holds what the runners read (rcfg.npy, floor.json, host.json, bindings.json,
kernels.json, meta.json)."""
import os, shutil, sys
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import splitgeo


def build(out, plan, levers="", runner=None):
    """Writes the build directory <out> (and <out>/geo); `runner`: a runner script copied in beside
    splitgeo.py (the family payloads)."""
    os.makedirs(out, exist_ok=True)
    geo = os.path.join(out, "geo")
    splitgeo.build(plan, geo, levers=levers, kernels=os.environ.get("SPLIT_SRC"))
    lay = os.path.join(geo, "layout")
    for f in os.listdir(lay):
        shutil.copy(os.path.join(lay, f), os.path.join(out, f))
    for f in [os.path.join(H, "splitgeo.py")] + ([runner] if runner else []):
        shutil.copy(f, os.path.join(out, os.path.basename(f)))
    return splitgeo.load(geo)


def main(runner=None):
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    a = sys.argv[1:] + [""]
    g = build(a[0], a[1], a[2], runner)
    print("ok", a[0], "W", g.W, "H", g.H)


if __name__ == "__main__":
    main()
