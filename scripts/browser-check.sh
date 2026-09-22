#!/usr/bin/env bash
# The tab's crates on wasm32-unknown-unknown, the wgpu feature natively (the
# workspace jobs lint default features), and the pages' scripts.
set -euo pipefail
cd "$(dirname "$0")/.."

WASM_CRATES=(pie-browser runtime engine-wgpu kernels-wgpu web-std wasmtime-web
  checkpoint checkpoint-dsl ztensor ztensor-compat tokenizer grammar chat-template
  models model-ir model-dsl model-compiler engine model-exec eta-ir eta-dsl eta-compiler
  eta-exec waker ids dtype client-api)
NATIVE_CRATES=(pie-browser runtime engine-wgpu kernels-wgpu web-std wasmtime-web)

pkgs() { for c in "$@"; do printf -- '-p %s ' "$c"; done; }

echo "== wasm32: clippy"
CARGO_TARGET_DIR=target-wasm cargo clippy --target wasm32-unknown-unknown \
  $(pkgs "${WASM_CRATES[@]}") --features runtime/wgpu,engine-wgpu/wgpu -- -D warnings

echo "== native, wgpu feature: clippy + tests"
cargo clippy $(pkgs "${NATIVE_CRATES[@]}") --features runtime/wgpu,engine-wgpu/wgpu -- -D warnings
cargo test -q $(pkgs "${NATIVE_CRATES[@]}") --features runtime/wgpu,engine-wgpu/wgpu

echo "== javascript"
for f in javascript/browser/src/*.mjs tests/browser/*.mjs tests/browser/tools/*.mjs scripts/bench/browser/*.mjs; do
  node --check "$f"
done
for f in tests/browser/*.html crates/kernels-wgpu/tools/*.html; do
  node -e '
    const fs = require("fs");
    const m = fs.readFileSync(process.argv[1], "utf8").match(/<script type="module">([\s\S]*?)<\/script>/);
    if (!m) process.exit(0);
    const tmp = require("os").tmpdir() + "/pie-page-check.mjs";
    fs.writeFileSync(tmp, m[1]);
    const r = require("child_process").spawnSync(process.execPath, ["--check", tmp], { stdio: "inherit" });
    process.exit(r.status);
  ' "$f"
done
echo "== ok"
