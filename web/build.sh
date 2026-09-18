#!/usr/bin/env bash
# Build pie for the browser: the wasm32 host, its wasm-bindgen glue, and the
# sample inferlet. Output lands in web/site/{pkg,inferlets}; serve web/site
# (plus a models/ directory with a `.wgpu.zt` artifact) with any static server
# — `node web/tools/serve.mjs web/site <models-dir>` works.
set -euo pipefail
cd "$(dirname "$0")/.."

profile="${1:-release}"
case "$profile" in
  release) flag="--release"; dir=release ;;
  # The workspace's shipping profile: thin LTO and stripped symbols take the
  # wasm from 33 MB to 23 MB (gzip 7.4 → 6.4 MB) at the cost of a slower
  # build and bare panic messages in the console.
  min) flag="--profile release-min"; dir=release-min ;;
  dev) flag=""; dir=debug ;;
  *) echo "usage: $0 [release|min|dev]" >&2; exit 2 ;;
esac

echo "== host (wasm32-unknown-unknown, $profile)"
# shellcheck disable=SC2086
CARGO_TARGET_DIR=target-wasm cargo build --target wasm32-unknown-unknown -p pie-web $flag

echo "== wasm-bindgen"
rm -rf web/site/pkg
wasm-bindgen --target web --out-dir web/site/pkg --out-name pie_web \
  "target-wasm/wasm32-unknown-unknown/$dir/pie_web.wasm"
cp web/site/platform.mjs web/site/pkg/platform.mjs
# wasm-bindgen 0.2.127 writes a string-returning import's (ptr, len) through
# `setInt32(arg0 + 4 * k, …)` with `arg0` as the signed i32 the wasm passed,
# so an out-pointer above 2 GiB goes negative and the DataView throws.
# Coerce it unsigned, as the glue already does for pointers it reads.
n=$(grep -c 'setInt32(arg0 + 4 \* ' web/site/pkg/pie_web.js || true)
sed -i 's/setInt32(arg0 + 4 \* /setInt32((arg0 >>> 0) + 4 * /g' web/site/pkg/pie_web.js
echo "glue: $n out-pointer stores made unsigned"
ls -la web/site/pkg/pie_web_bg.wasm

echo "== inferlet (wasm32-wasip2)"
(cd tests/inferlets && cargo build -p text-completion --release --target wasm32-wasip2)
mkdir -p web/site/inferlets
cp tests/inferlets/target/wasm32-wasip2/release/text_completion.wasm web/site/inferlets/
cp tests/inferlets/text-completion/Pie.toml web/site/inferlets/text-completion.Pie.toml
echo "== done"
