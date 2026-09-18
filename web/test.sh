#!/usr/bin/env bash
# The browser port's GPU suite: every kernel through Tint, the build, then the
# real thing in headless Chrome — a completion, the host
# corner cases, the transport measurement, the client SDK page, and the page
# lifecycle (reload from cache, hidden tab). Needs the 0.8B model linked as
# README says and Playwright's Chromium (web/tools/headless.mjs knows where).
#
# PIE_WEB_MODELS="models2b/…zt models9b/…zt" also boots each listed artifact
# once (with PIE_WEB_MODEL_CONFIG, e.g. "device_memory_mb = 20480,…").
set -euo pipefail
cd "$(dirname "$0")/.."

run() { # <label> <page-with-query> [timeout-ms]
  local label=$1 page=$2 timeout=${3:-600000}
  echo "== $label"
  node web/tools/headless.mjs web/site "$page" --timeout "$timeout" 2>&1 \
    | grep -v '^\s*at ' \
    | grep -E 'booted|returned|output:|process |^ok|^FAIL|^MATRIX|ping|launch without|headless\] status|failed' \
    | cut -c1-160
}

echo "== kernels through Tint"
cargo run -q -p kernels-wgpu --example dump_wgsl -- web/tools/wgsl
node web/tools/headless.mjs web/tools wgsl-check.html --timeout 300000 | tail -2

echo "== build the host"
./web/build.sh release > /dev/null

run "boot, install, run (worker)" "index.html?auto&max_tokens=${PIE_WEB_TOKENS:-16}&runs=${PIE_WEB_RUNS:-2}"
run "host corner cases" "corners.html"
run "transport (no engine)" "transport.html?n=100" 300000
run "client SDK page" "sdk.html?max_tokens=8"
run "SDK corner cases (bench page)" "bench.html?what=corners"
# A streaming chat inferlet (chat template, sampling, token-by-token events).
run "chat-completion" "index.html?auto&runs=1&max_tokens=24&inferlet=inferlets/chat_completion.wasm&manifest=inferlets/chat-completion.Pie.toml&input=%7B%22prompt%22%3A%22What%20is%20the%20capital%20of%20France%3F%22%2C%22max_tokens%22%3A24%2C%22temperature%22%3A0%7D"
# The inferlets of the matrix that answer deterministically (device-carried
# loops, masks, watermarks, grammar-constrained decoding, token healing, a
# beam of one, the epilogue-op probe, …), one tab.
echo "== inferlets (wasm32-wasip2)"
./web/tools/inferlets.sh
run "deterministic inferlets (matrix)" "matrix.html?only=sort-probe,sampling-primitives,prefill-rows,dry-repetition-penalty-cold,naive-baseline-cold,top-a-sampling-cold,json-schema-constrained-decoding,token-healing,rs-window-decode,repetition-penalty,greenlist-watermarking,gumbel-watermark,attention-sink,sliding-window-attention,naive-masked-dense,naive-masked-structured,contrastive-decoding,text-completion-bench,beam-search-1&entry_timeout=240000" 1500000
echo "== page lifecycle"
node web/tools/lifecycle.mjs web/site 2>&1 | grep -E 'ok  |FAIL|lifecycle\]' | cut -c1-140

for model in ${PIE_WEB_MODELS:-}; do
  cfg=${PIE_WEB_MODEL_CONFIG:-}
  run "model $model" "index.html?auto&max_tokens=8&model=$model&config=$(printf '%s' "$cfg" | sed 's/ /%20/g; s/=/%3D/g')" 900000
done
echo "== done"
