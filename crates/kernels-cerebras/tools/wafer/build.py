#!/usr/bin/env python3
"""build.py <outdir> <plan.json> [levers K:V,...]: the wafer program's whole build directory (layout.csl and one
source a PE kind), from a plan through the Rust `wafer-compile`, with the kernels in programs/wafer (or $WAFER_KERNELS),
and the bare host runner of the plan's family where it has one (run_<family>.py, with host.py and compiler.py: the
chip payload).

The levers and diagnostics (CYV, GELUT, WSPEC, SLW, RING, DIAG, TSTAMP, ...; `<kind>.K` for one kind) set the kernels'
params, defaulting to the shipped build. Everything is resolved here: link.sh compiles <outdir> with no shape params
and puts the artifact the engine runs (<outdir>/compiled's program.json, host.json, bindings.json, rcfg arrays) beside the
compiled program. <outdir>/compiled holds what the runners read (rcfg.npy, floor.json, host.json, bindings.json,
kernels.json, plan.json)."""
import json, os, shutil, sys
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import compiler

# each family's bare runner (the Qwen family has none: its programs run through the engine's ArtifactModel)
RUNNERS = {"Gemma": "gemma/run_gemma.py", "Gemma12": "gemma12/run_gemma12.py", "GptOss": "gptoss/run_gptoss.py"}


def build(out, plan, levers=""):
    """Writes the build directory <out> (and <out>/compiled), with the plan's family's runner."""
    os.makedirs(out, exist_ok=True)
    compiled = os.path.join(out, "compiled")
    compiler.build(plan, compiled, levers=levers, kernels=os.environ.get("WAFER_KERNELS"))
    lay = os.path.join(compiled, "layout")
    for f in os.listdir(lay):
        shutil.copy(os.path.join(lay, f), os.path.join(out, f))
    runner = RUNNERS.get(json.load(open(plan))["family"])
    if runner:
        for f in ("compiler.py", "host.py", runner):
            shutil.copy(os.path.join(H, f), os.path.join(out, os.path.basename(f)))
    return compiler.load(compiled)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    a = sys.argv[1:] + [""]
    g = build(a[0], a[1], a[2])
    print("ok", a[0], "W", g.W, "H", g.H)


if __name__ == "__main__":
    main()
