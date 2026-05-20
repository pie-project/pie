#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TENSORS="${1:-1024}"
OUT="${2:-/tmp/pie_weight_loader_storage_program.json}"

start_ns="$(date +%s%N)"
cargo run -q -p pie-weight-loader --example compile_dump -- "${TENSORS}" > "${OUT}"
end_ns="$(date +%s%N)"

elapsed_ms="$(( (end_ns - start_ns) / 1000000 ))"
bytes="$(wc -c < "${OUT}")"
printf 'pie-weight-loader compile_dump tensors=%s elapsed_ms=%s dump_bytes=%s dump=%s\n' \
  "${TENSORS}" "${elapsed_ms}" "${bytes}" "${OUT}"
