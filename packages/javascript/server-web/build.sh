#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."

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
rm -rf packages/javascript/server-web/pkg
wasm-bindgen --target web --out-dir packages/javascript/server-web/pkg --out-name pie_browser \
  "target-wasm/wasm32-unknown-unknown/$dir/pie_browser.wasm"
cp packages/javascript/server-web/src/platform.mjs packages/javascript/server-web/pkg/platform.mjs
# wasm-bindgen 0.2.128 stores a string-returning import's (ptr, len) through the
# signed i32 `arg0`, so an out-pointer above 2 GiB throws; coerce it unsigned.
n=$(grep -c 'setInt32(arg0 + 4 \* ' packages/javascript/server-web/pkg/pie_browser.js || true)
sed -i.orig 's/setInt32(arg0 + 4 \* /setInt32((arg0 >>> 0) + 4 * /g' packages/javascript/server-web/pkg/pie_browser.js
rm packages/javascript/server-web/pkg/pie_browser.js.orig
echo "glue: $n out-pointer stores made unsigned"
ls -la packages/javascript/server-web/pkg/pie_browser_bg.wasm

echo "== bundle"
[ -x packages/javascript/node_modules/.bin/esbuild ] || npm --prefix packages/javascript install --include=dev --no-audit --no-fund
npm --prefix packages/javascript/server-web run -s bundle
ls -la packages/javascript/server-web/dist/
echo "== done"
