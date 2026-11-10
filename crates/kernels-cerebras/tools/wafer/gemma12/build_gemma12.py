#!/usr/bin/env python3
"""build_gemma12.py <outdir> <plan.json> [levers K:V,...]: the Gemma 4 12B wafer program payload: build.py's
build directory (layout.csl, one source a PE kind, compiled/ from wafer-compile) plus the host runner (run_gemma12.py) and compiler.py
it places with. The plan names the family; the geometry is the Rust one (src/spatial/family)."""
import os, sys
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, ".."))
import build

build.main(runner=os.path.join(H, "run_gemma12.py"))
