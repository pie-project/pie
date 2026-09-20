#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

work=${PIE_LAVAPIPE_WORK:-$(mktemp -d)}
export PIE_HOME="$work/home"
mkdir -p "$PIE_HOME"
port=${PIE_LAVAPIPE_PORT:-28419}

echo "== a random-weight snapshot"
tok=()
if [ -n "${PIE_TOKENIZER_DIR:-}" ]; then tok=(--tokenizer "$PIE_TOKENIZER_DIR"); fi
python3 scripts/tiny_qwen35.py "$work/snapshot" "${tok[@]}"

echo "== pie with the wgpu engine"
cargo build --release -p pie --features wgpu
pie=./target/release/pie
echo "== inferlets (wasm32-wasip2)"
(cd tests/inferlets && cargo build -q --release --target wasm32-wasip2 -p text-completion -p naive-baseline -p sort-probe)

echo "== import"
"$pie" model import "$work/snapshot" --sku qwen35-tiny-u4g64-kv-bf16 --keep-source

cat > "$PIE_HOME/config.toml" <<TOML
[server]
host = "127.0.0.1"
port = $port
telemetry = false
[model]
name = "default"
model = "snapshot"
[engine]
type = "wgpu"
backends = "vulkan"
max_total_pages = 512
max_forward_tokens = 1024
max_forward_requests = 8
max_state_slots = 16
device = ["wgpu:0"]
[sandbox]
allow_fs = false
allow_network = true
network_allowed_hosts = ["*"]
TOML

echo "== serve on lavapipe"
icd=${PIE_VULKAN_ICD:-$(ls /usr/share/vulkan/icd.d/lvp_icd*.json 2>/dev/null | head -1)}
if [ -z "$icd" ]; then echo "no lavapipe ICD under /usr/share/vulkan/icd.d (install mesa-vulkan-drivers)"; exit 1; fi
VK_ICD_FILENAMES="$icd" VK_DRIVER_FILES="$icd" "$pie" serve > "$work/serve.log" 2>&1 &
server=$!
trap 'kill $server 2>/dev/null || true' EXIT
for _ in $(seq 1 300); do
  grep -q "Server ready" "$work/serve.log" && break
  if ! kill -0 $server 2>/dev/null; then sed 's/\x1b\[[0-9;]*m//g' "$work/serve.log" | grep -v "RPC{" | tail -20; exit 1; fi
  sleep 1
done
grep -q "Server ready" "$work/serve.log" || { echo "the server did not come up"; tail -20 "$work/serve.log"; exit 1; }

what=corners
if [ "${PIE_LAVAPIPE_BENCH:-0}" = "1" ]; then what=both; fi
echo "== $what"
PIE_BENCH_FIXTURE=1 node benches/web/native.mjs "ws://127.0.0.1:$port/v1/ws" "$what" | grep -v "^RESULT" | cut -c1-160
echo "== sort-probe"
PIE_MATRIX_ENTRY_TIMEOUT_MS=900000 node tests/web/tools/matrix-native.mjs "ws://127.0.0.1:$port/v1/ws" sort-probe-light > "$work/sort-probe.log" || true
grep "^MATRIX" "$work/sort-probe.log" | cut -f1,2,3 | sed 's/\\n/\n   /g' | cut -c1-160
grep -q $'^MATRIX sort-probe-light\tok\t' "$work/sort-probe.log" || { echo "sort-probe failed"; exit 1; }
sed 's/\x1b\[[0-9;]*m//g' "$work/serve.log" | grep -v "RPC{" | grep -E "ERROR|panic" | head -5 || true
echo "== done"
