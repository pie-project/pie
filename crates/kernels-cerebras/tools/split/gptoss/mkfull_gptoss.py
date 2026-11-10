#!/usr/bin/env python3
"""mkfull_gptoss.py <outdir> <prep dir> <prefix> [overrides k=v,...] [levers K:V,...]: the gpt-oss role-split payload: mkfull.py's
build directory (layout.csl, one source a PE kind, geo/ from split-geo) plus the host runner (run_gptoss.py) and splitgeo.py
it places with. The family comes from the prep's meta; the geometry is the Rust one (src/spatial/family)."""
import os, sys
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, ".."))
import mkfull

mkfull.main(runner=os.path.join(H, "run_gptoss.py"))
