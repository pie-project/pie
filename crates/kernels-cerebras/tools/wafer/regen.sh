#!/bin/sh
# regen.sh <out root> [config ...]
#
# Rebuilds the shipping wafer programs from their traces: for every full config of baseline/configs.json (or the
# named ones), `wafer-compile plan` (its deployment, preset and overrides: what the chip / Pie runs used) into
# <out root>/<config>/plan, then build.py (its levers) into <out root>/<config>/build, and with CSL_SDK set link.sh,
# which puts the artifact the engine runs in <out root>/<config>/build/out/artifact.
#
# The staged preps keep their tiles: the plans cut them as they were cut (the 9B / 4B / 0.8B at RLV 4, as their RoPE
# tables were staged). A program carries its artifact, and the engine opens nothing else. The preps these programs place (BENCH_JOB, default spatial/bench-job):
#   s9b (s9b_), s9b5 (s9b5_), s4b (s4b_), s08b (s08b_)   Qwen3.5 9B, 9B at RLV 5, 4B, 0.8B
#   q8p4 (q84_)                                          Qwen3-8B
#   ge2 (ge2_), ge4 (ge4_)                               Gemma 4 E2B, E4B
#   g12r (g12r_)                                         Gemma 4 12B
#   gpo (go_)                                            gpt-oss-20b
# Their <prefix>meta.json / params.txt are records nobody reads now.
# Llama 3.1 8B has no staged prep yet: plan it (`wafer-compile plan --model llama31-8b-bf16-kv-bf16`, the first preset
# that builds), cut the HF checkpoint with dense/prep_dense.py for that plan, and build / link as here.
#
# Env: PY (a python with numpy), WAFER_COMPILE (the binary; default: built into $CARGO_TARGET_DIR), CSL_SDK (link).
set -e
OUT=${1:?usage: regen.sh <out root> [config ...]}; shift
T=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python3}
if [ -z "$WAFER_COMPILE" ]; then
  cargo build -q --release --manifest-path "$T/../../Cargo.toml" --bin wafer-compile
  WAFER_COMPILE=${CARGO_TARGET_DIR:-$T/../../../../target}/release/wafer-compile
fi
export WAFER_COMPILE
"$PY" - "$T/baseline/configs.json" "$@" <<'EOF' | while IFS='	' read -r name model preset over levers; do
import json, sys
want = set(sys.argv[2:])
for c in json.load(open(sys.argv[1]))["configs"]:
    if c["kind"] == "full" and (not want or c["name"] in want):
        print("\t".join([c["name"], c["model"], c["preset"], c["set"] or "-", c["levers"] or "-"]))
EOF
  [ "$over" = "-" ] && over=""; [ "$levers" = "-" ] && levers=""
  d=$OUT/$name
  echo "== $name: $model, preset $preset${over:+, set $over}${levers:+, levers $levers}"
  "$WAFER_COMPILE" plan --model "$model" --preset "$preset" --set "$over" --out "$d/plan"
  "$PY" "$T/build.py" "$d/build" "$d/plan/plan.json" "$levers"
  [ -z "$CSL_SDK" ] || "$T/link.sh" "$d/build"
done
