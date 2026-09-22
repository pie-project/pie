#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

profile="${1:-release}"
case "$profile" in
  release) flag="--release"; dir=release ;;
  min) flag="--profile release-min"; dir=release-min ;;
  dev) flag=""; dir=debug ;;
  *) echo "usage: $0 [release|min|dev]" >&2; exit 2 ;;
esac

echo "== host (wasm32-unknown-unknown, $profile)"
CARGO_TARGET_DIR=target-wasm cargo build --target wasm32-unknown-unknown -p pie-browser $flag

echo "== wasm-bindgen"
rm -rf javascript/server-web/pkg
wasm-bindgen --target web --out-dir javascript/server-web/pkg --out-name pie_browser \
  "target-wasm/wasm32-unknown-unknown/$dir/pie_browser.wasm"
cp javascript/server-web/src/platform.mjs javascript/server-web/pkg/platform.mjs
# wasm-bindgen 0.2.128 stores a string-returning import's (ptr, len) through the
# signed i32 `arg0`, so an out-pointer above 2 GiB throws; coerce it unsigned.
n=$(grep -c 'setInt32(arg0 + 4 \* ' javascript/server-web/pkg/pie_browser.js || true)
sed -i 's/setInt32(arg0 + 4 \* /setInt32((arg0 >>> 0) + 4 * /g' javascript/server-web/pkg/pie_browser.js
echo "glue: $n out-pointer stores made unsigned"
ls -la javascript/server-web/pkg/pie_browser_bg.wasm

echo "== bundle"
[ -x javascript/node_modules/.bin/esbuild ] || npm --prefix javascript install --include=dev --no-audit --no-fund
npm --prefix javascript/server-web run -s bundle
ls -la javascript/server-web/dist/
echo "== done"
