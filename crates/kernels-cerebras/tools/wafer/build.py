#!/usr/bin/env python3
"""build.py <outdir> <plan.json> [levers K:V,...]: the wafer program's whole build directory (layout.csl and one
source a PE kind), from a plan through the Rust `wafer-compile`, with the kernels in programs/wafer (or $WAFER_KERNELS).

The levers and diagnostics (CYV, GELUT, WSPEC, SLW, RING, DIAG, TSTAMP, ...; `<kind>.K` for one kind) set the kernels'
params, defaulting to the shipped build. Everything is resolved here: link.sh compiles <outdir> with no shape params
and puts the artifact the engine runs (<outdir>/compiled's program.json, host.json, bindings.json, rcfg arrays) beside the
compiled program. <outdir>/compiled holds what the runners read (rcfg.npy, floor.json, host.json, bindings.json,
kernels.json, plan.json)."""
import os, shutil, sys
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import compiler


def build(out, plan, levers="", runner=None):
    """Writes the build directory <out> (and <out>/compiled); `runner`: a runner script copied in beside
    compiler.py (the family payloads)."""
    os.makedirs(out, exist_ok=True)
    compiled = os.path.join(out, "compiled")
    compiler.build(plan, compiled, levers=levers, kernels=os.environ.get("WAFER_KERNELS"))
    lay = os.path.join(compiled, "layout")
    for f in os.listdir(lay):
        shutil.copy(os.path.join(lay, f), os.path.join(out, f))
    for f in [os.path.join(H, "compiler.py")] + ([runner] if runner else []):
        shutil.copy(f, os.path.join(out, os.path.basename(f)))
    return compiler.load(compiled)


def main(runner=None):
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    a = sys.argv[1:] + [""]
    g = build(a[0], a[1], a[2], runner)
    print("ok", a[0], "W", g.W, "H", g.H)


if __name__ == "__main__":
    main()
