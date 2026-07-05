#!/usr/bin/env bash
# One canonical pie build that aligns the CLI binary and the
# pie-server-py Python wheel to the SAME cargo feature set, so they
# share the cmake/cuda build dir. Bypassing this script with ad-hoc
# `cargo build --features X` invocations forces a fresh hash dir,
# which re-runs cmake configure and triggers a full 49-CU CUDA
# rebuild (~22 minutes from cold cache, ~2 min with ccache warm).
#
# Use:
#   scripts/build_pie_dev.sh           # cargo CLI + maturin wheel
#   scripts/build_pie_dev.sh cli       # cargo CLI only
#   scripts/build_pie_dev.sh wheel     # maturin wheel only
set -euo pipefail
cd "$(dirname "$0")/.."

# Match pie-server-py's feature set so both targets share build cache.
FEATURES="driver-cuda,driver-dummy"
TARGET="${1:-all}"

build_cli() {
  echo "▶ building pie CLI (--no-default-features --features $FEATURES)"
  cargo build --release -p pie-worker --bin pie \
    --no-default-features --features "$FEATURES"
}

build_wheel() {
  echo "▶ building pie-server-py wheel (matches feature set automatically)"
  (cd sdk/python-server && source .venv/bin/activate && \
   maturin develop --release)
}

case "$TARGET" in
  cli)   build_cli   ;;
  wheel) build_wheel ;;
  all)   build_cli && build_wheel ;;
  *) echo "usage: $0 [cli|wheel|all]"; exit 1 ;;
esac
