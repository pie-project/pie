#!/usr/bin/env bash
# Build the JavaScript language component: the one wasm every JavaScript
# inferlet runs in. It bundles StarlingMonkey (via componentize-js), the
# `@pie-project/inferlet` library and acorn, against the derived (synchronous)
# JavaScript world in `../wit`.
#
#   javascript/inferlet/language/build.sh [OUT]
#
#   OUT   defaults to $PIE_HOME/languages/javascript.wasm
#         (~/.pie/languages/javascript.wasm), where `pie` looks.
#
# Needs Node.js >= 18 and the `javascript/` workspace's devDependencies
# (`npm ci` runs there when node_modules is absent).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lib="$(cd "$here/.." && pwd)"
out="${1:-${PIE_HOME:-$HOME/.pie}/languages/javascript.wasm}"
case "$out" in
  /*) ;;
  *) out="$PWD/$out" ;;
esac

cd "$lib"
if [ ! -d ../node_modules ]; then
  echo "== npm ci (javascript workspace)"
  npm --prefix .. ci --include=dev --no-audit --no-fund
fi

echo "== bindings + library build"
npm run -s generate-bindings
npm run -s build

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

echo "== bundle (esbuild)"
npx --no-install esbuild "$here/app.js" --bundle --format=esm --platform=neutral \
  --target=es2022 --main-fields=module,main \
  --alias:@pie-project/inferlet="$lib/dist/index.js" \
  --external:'pie:*' --external:'wasi:*' \
  --outfile="$work/app.js" --log-level=warning

echo "== componentize (componentize-js)"
mkdir -p "$(dirname "$out")"
npx --no-install componentize-js "$work/app.js" -o "$out" --wit "$lib/wit" --world-name inferlet
ls -la "$out"
