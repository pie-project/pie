#!/usr/bin/env python3
"""pie at the fixed points under a device profile with some decisions moved.

A variant is a home of its own (`--home`, its `artifacts` linked to the
measured home's) whose user profile is today's resolved one with the named
decisions set, so a run reads exactly what `pie profile dump` says plus the
change. The run is `pie_baseline.py` at the model's scoreboard flags, on an
idle device.

    python scripts/bench/profile_variant.py --model gptoss-120b --set kernel.moe.mxfp4_grouped_reuse=1 \\
        --home /root/pie-homes/variant-reuse1 --out DIR [--mode offline]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from common import ROOT, wait_for_idle_gpu


def profile_with(pie_bin: str, home: str, sets: list[tuple[str, str]]) -> str:
    """Today's resolved profile, with each `domain.key = value` replaced."""
    text = subprocess.run([pie_bin, "profile", "dump"], env={**os.environ, "PIE_HOME": home},
                          capture_output=True, text=True, check=True).stdout
    section = None
    out = []
    pending = dict(sets)
    for line in text.splitlines():
        head = re.match(r"^\[(\w+)\]$", line)
        if head:
            section = head.group(1)
        kv = re.match(r"^([\w.]+) = ", line)
        if kv and section and f"{section}.{kv.group(1)}" in pending:
            line = f"{kv.group(1)} = {pending.pop(f'{section}.{kv.group(1)}')}"
        out.append(line)
    if pending:
        raise SystemExit(f"no such decision in the profile: {sorted(pending)}")
    return "\n".join(out) + "\n"


def main() -> None:
    import scoreboard

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(scoreboard.MODELS))
    ap.add_argument("--set", action="append", default=[], metavar="ID=VALUE",
                    help="a decision and its TOML value, e.g. kernel.gemm.skinny_rows=0")
    ap.add_argument("--home", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="offline", choices=["offline", "serve"])
    ap.add_argument("--engine-option", action="append", default=[])
    ap.add_argument("--pie-bin", default=str(ROOT / "target" / "release" / "pie"))
    args = ap.parse_args()

    model = scoreboard.MODELS[args.model]
    base = os.environ.get("PIE_HOME", "/root/pie-homes/bench")
    home = Path(args.home)
    (home / "profiles" / "cuda" / "user").mkdir(parents=True, exist_ok=True)
    if not (home / "artifacts").exists():
        (home / "artifacts").symlink_to(Path(base) / "artifacts")
    sets = [tuple(s.split("=", 1)) for s in args.set]
    for f in (home / "profiles" / "cuda" / "user").glob("*.toml"):
        f.unlink()
    file = profile_with(args.pie_bin, base, sets)
    device = re.search(r"^# (.+)$", file, re.M).group(1)
    name = re.sub(r"[^a-z0-9]+", "-", device.lower()).strip("-")
    (home / "profiles" / "cuda" / "user" / f"{name}.toml").write_text(file)

    wait_for_idle_gpu()
    cmd = [sys.executable, "pie_baseline.py", args.mode, "--model", model.artifact(), "--hf", model.snapshot(),
           "--out", args.out, "--pie-bin", args.pie_bin]
    for opt in model.pie_options + args.engine_option:
        cmd += ["--engine-option", opt]
    subprocess.run(cmd, cwd=Path(__file__).parent, env={**os.environ, "PIE_HOME": str(home)}, check=True)


if __name__ == "__main__":
    main()
