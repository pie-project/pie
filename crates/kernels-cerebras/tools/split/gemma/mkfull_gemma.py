#!/usr/bin/env python3
"""mkfull_gemma.py <outdir> <plan.json> [levers K:V,...]: the Gemma role-split payload: mkfull.py's
build directory (layout.csl, one source a PE kind, geo/ from split-geo) plus the host runner (run_gemma.py) and splitgeo.py
it places with. The plan names the family; the geometry is the Rust one (src/spatial/family)."""
import os, sys
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, ".."))
import mkfull

mkfull.main(runner=os.path.join(H, "run_gemma.py"))
