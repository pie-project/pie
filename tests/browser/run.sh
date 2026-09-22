#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

run() { # <label> <page-with-query> [timeout-ms]
  local label=$1 page=$2 timeout=${3:-600000}
  echo "== $label"
  node tests/browser/tools/headless.mjs . "tests/browser/$page" --timeout "$timeout" 2>&1 \
    | grep -v '^\s*at ' \
    | grep -E 'booted|returned|output:|process |^ok|^FAIL|^MATRIX|ping|launch without|headless\] status|failed' \
    | cut -c1-160
}

echo "== kernels through Tint"
cargo run -q -p kernels-wgpu --example dump_wgsl -- crates/kernels-wgpu/tools/wgsl
node tests/browser/tools/headless.mjs . crates/kernels-wgpu/tools/wgsl-check.html --timeout 300000 | tail -2

echo "== build the host"
./javascript/server-web/build.sh release > /dev/null
./tests/browser/tools/inferlets.sh

run "boot, install, run (worker)" "index.html?auto&max_tokens=${PIE_WEB_TOKENS:-16}&runs=${PIE_WEB_RUNS:-2}"
run "host corner cases" "corners.html"
run "transport (no engine)" "transport.html?n=100" 300000
run "client page" "client.html?max_tokens=8"
run "library corner cases (bench page)" "bench.html?what=corners"
run "chat-completion" "index.html?auto&runs=1&max_tokens=24&inferlet=inferlets/chat_completion.wasm&manifest=inferlets/chat-completion.Pie.toml&input=%7B%22prompt%22%3A%22What%20is%20the%20capital%20of%20France%3F%22%2C%22max_tokens%22%3A24%2C%22temperature%22%3A0%7D"
run "deterministic inferlets (matrix)" "matrix.html?only=sort-probe,sampling-primitives,prefill-rows,dry-repetition-penalty-cold,naive-baseline-cold,naive-baseline-js,top-a-sampling-cold,json-schema-constrained-decoding,token-healing,rs-window-decode,repetition-penalty,greenlist-watermarking,gumbel-watermark,attention-sink,sliding-window-attention,naive-masked-dense,naive-masked-structured,contrastive-decoding,text-completion-bench,beam-search-1&entry_timeout=240000" 1500000
echo "== page lifecycle"
node tests/browser/tools/lifecycle.mjs 2>&1 | grep -E 'ok  |FAIL|lifecycle\]' | cut -c1-140

for model in ${PIE_WEB_MODELS:-}; do
  cfg=${PIE_WEB_MODEL_CONFIG:-}
  run "model $model" "index.html?auto&max_tokens=8&model=$model&config=$(printf '%s' "$cfg" | sed 's/ /%20/g; s/=/%3D/g')" 900000
done
echo "== done"
