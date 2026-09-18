#!/usr/bin/env bash
# The browser port's compile-and-lint gate, no GPU needed: every crate the tab
# links checks and clippies for wasm32, the same crates stay green natively,
# and the touched crates are rustfmt-clean. Run before `web/test.sh`.
set -euo pipefail
cd "$(dirname "$0")/.."

WASM_CRATES=(pie-web runtime engine-wgpu kernels-wgpu web-rt wasi-web wasmtime-web
  checkpoint checkpoint-dsl ztensor ztensor-compat tokenizer grammar chat-template
  models model-ir model-dsl model-compiler engine model-exec eta-ir eta-dsl eta-compiler
  eta-exec waker ids dtype client-api)
NATIVE_CRATES=(pie-web runtime engine-wgpu kernels-wgpu web-rt wasi-web wasmtime-web
  checkpoint ztensor ztensor-compat tokenizer grammar eta-compiler waker models)
FMT_CRATES=(pie-web runtime engine-wgpu kernels-wgpu web-rt wasi-web wasmtime-web
  checkpoint ztensor ztensor-compat tokenizer grammar eta-compiler waker models)

pkgs() { for c in "$@"; do printf -- '-p %s ' "$c"; done; }

echo "== wasm32: check + clippy"
# shellcheck disable=SC2046
CARGO_TARGET_DIR=target-wasm cargo clippy --target wasm32-unknown-unknown \
  $(pkgs "${WASM_CRATES[@]}") --features runtime/wgpu,engine-wgpu/wgpu -- -D warnings

echo "== native: check + clippy"
# shellcheck disable=SC2046
cargo clippy $(pkgs "${NATIVE_CRATES[@]}") --features runtime/wgpu,engine-wgpu/wgpu -- -D warnings

echo "== native: tests"
# shellcheck disable=SC2046
cargo test -q $(pkgs "${NATIVE_CRATES[@]}") --features runtime/wgpu,engine-wgpu/wgpu

echo "== rustfmt (touched crates only; the cuda crates are not rustfmt-clean upstream)"
# shellcheck disable=SC2046
cargo fmt $(pkgs "${FMT_CRATES[@]}") -- --check

echo "== javascript"
for f in web/site/*.mjs web/tools/*.mjs web/bench/*.mjs; do
  node --check "$f"
done
# The pages' inline module scripts, syntax-checked as modules.
for f in web/site/*.html web/tools/*.html; do
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
(cd sdk/client/javascript && node --test >/dev/null && echo "sdk tests ok")
echo "== ok"
