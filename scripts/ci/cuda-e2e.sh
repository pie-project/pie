#!/usr/bin/env bash
# pie on a real GPU, end to end: a debug `pie` with the CUDA engine serves
# Qwen3.5-0.8B; the compat API suite runs against it; an inferlet is
# installed, replaced and removed while it runs; `pie run` resolves a bare
# name. PIE_HOME holds the model between runs (the runner's volume).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PIE_HOME="${PIE_HOME:-$HOME/.pie}"
port=${PIE_E2E_PORT:-18517}
pie=${CARGO_TARGET_DIR:-target}/debug/pie

cargo build -p pie --features cuda --bin pie
(cd examples && cargo build -q --release --target wasm32-wasip2 -p text-completion -p naive-baseline)

"$pie" model list
if ! ls "$PIE_HOME"/models/Qwen--Qwen3.5-0.8B/*.cuda.zt >/dev/null 2>&1; then
  "$pie" model import Qwen/Qwen3.5-0.8B
fi
mkdir -p "$PIE_HOME"
cat > "$PIE_HOME/config.toml" <<TOML
[server]
host = "127.0.0.1"
port = $port
telemetry = false
[model]
name = "default"
model = "Qwen/Qwen3.5-0.8B"
[engine]
type = "cuda_native"
device = ["cuda:0"]
activation_dtype = "bfloat16"
gpu_mem_utilization = 0.85
[sandbox]
allow_fs = false
allow_network = true
network_allowed_hosts = ["*"]
TOML
"$pie" doctor || true

"$pie" inferlet remove text-completion@0.3.0 >/dev/null 2>&1 || true
log="${RUNNER_TEMP:-/tmp}/pie-serve.log"
"$pie" serve > "$log" 2>&1 &
serve=$!
trap 'kill $serve 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do grep -q "Server ready\|✗" "$log" && break; sleep 2; done
grep -q "Server ready" "$log" || { echo "the server did not come up"; tail -30 "$log"; exit 1; }

echo "== compat API suite"
uv run --with openai --with anthropic --with google-genai python tests/builtins/test_compat.py --base-url "http://127.0.0.1:$port"

echo "== install, replace and remove while serving"
submit() { uv run --project python/client python scripts/ci/launch.py "ws://127.0.0.1:$port" "$1" || true; }
wasm=$(cd examples && cargo metadata --format-version 1 --no-deps | jq -r .target_directory)/wasm32-wasip2/release
tc=$wasm/text_completion.wasm
nb=$wasm/naive_baseline.wasm
submit text-completion | grep -q '^error' || { echo "expected: not installed"; exit 1; }
"$pie" inferlet install "$tc"
submit text-completion | grep -q "^ok.*Paris" || { echo "expected: Paris"; exit 1; }
"$pie" inferlet install "$nb"
submit naive-baseline | grep -q "^ok.*sampler" || { echo "expected: the second program's output"; exit 1; }
"$pie" inferlet install "$tc" --force
submit text-completion | grep -q "^ok.*Paris" || { echo "expected: Paris after the replacement"; exit 1; }
"$pie" inferlet remove text-completion@0.3.0
submit text-completion | grep -q '^error' || { echo "expected: removed"; exit 1; }
"$pie" inferlet remove naive-baseline
if grep -q 'panicked at' "$log"; then echo "the server panicked"; grep -A3 'panicked at' "$log"; exit 1; fi
kill $serve; wait $serve 2>/dev/null || true

echo "== pie run by bare name"
"$pie" run text-completion | grep -q Paris
echo "cuda e2e: ok"
