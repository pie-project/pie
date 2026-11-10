#!/usr/bin/env python3
"""build_gptoss.py <outdir> <plan.json> [levers K:V,...]: the gpt-oss wafer program payload: build.py's
build directory (layout.csl, one source a PE kind, compiled/ from wafer-compile) plus the host runner (run_gptoss.py) and compiler.py
it places with. The plan names the family; the geometry is the Rust one (src/spatial/family)."""
import os, sys
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, ".."))
import build

build.main(runner=os.path.join(H, "run_gptoss.py"))
