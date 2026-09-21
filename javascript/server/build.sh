#!/usr/bin/env bash
# Build the native addon and place it beside index.mjs as
# pie-server.<platform>-<arch>.node.
#
#   javascript/server/build.sh [release|dev] [--features <list>]
#
# The engine features default to this crate's defaults (`cuda`); pass e.g.
# `--features metal` on macOS or `--no-default-features` for an engine-less
# build.
set -euo pipefail
cd "$(dirname "$0")/../.."

profile="${1:-release}"
shift || true
case "$profile" in
  release) flag="--release"; dir=release ;;
  dev) flag=""; dir=debug ;;
  *) echo "usage: $0 [release|dev] [cargo flags...]" >&2; exit 2 ;;
esac

echo "== addon ($profile)"
cargo build -p pie-server-node $flag "$@"

case "$(uname -s)" in
  Linux)  lib="libpie_server.so" ;;
  Darwin) lib="libpie_server.dylib" ;;
  MINGW*|MSYS*|CYGWIN*) lib="pie_server.dll" ;;
  *) echo "unknown platform $(uname -s)" >&2; exit 1 ;;
esac
platform="$(node -p 'process.platform + "-" + process.arch')"
out="javascript/server/pie-server.$platform.node"
cp "target/$dir/$lib" "$out"
ls -la "$out"
echo "== done"
