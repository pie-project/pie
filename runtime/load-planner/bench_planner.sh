#!/usr/bin/env bash
set -euo pipefail

SNAPSHOT="${1:?usage: bench_planner.sh SNAPSHOT [BACKEND] [OUT]}"
BACKEND="${2:-cuda}"
OUT="${3:-/tmp/pie_load_plan.json}"

start_ns="$(date +%s%N)"
cargo run -q -p pie-load-planner --example plan_dump -- \
  "${SNAPSHOT}" "${BACKEND}" > "${OUT}"
end_ns="$(date +%s%N)"

elapsed_ms="$(( (end_ns - start_ns) / 1000000 ))"
bytes="$(wc -c < "${OUT}")"
printf 'pie-load-planner plan_dump snapshot=%s backend=%s elapsed_ms=%s dump_bytes=%s dump=%s\n' \
  "${SNAPSHOT}" "${BACKEND}" "${elapsed_ms}" "${bytes}" "${OUT}"
