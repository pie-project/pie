#!/usr/bin/env bash
# The engine end to end without a GPU: a random-weight miniature of Qwen3.5
# served by `pie` on Mesa's lavapipe (CPU Vulkan), then the inferlet library
# corner cases, sort-probe, one compat API call, and — when PIE_LANGUAGES_DIR
# holds the two pie-language-*.tar.gz — a Python and a JavaScript inferlet.
#
#   PIE_ENGINE          wgpu (default) or vulkan
#   PIE_TOKENIZER_DIR   a directory with Qwen3.5's tokenizer files
#   PIE_LANGUAGES_DIR   where pie-language-python.tar.gz and -javascript.tar.gz are
set -euo pipefail
cd "$(dirname "$0")/../.."

engine=${PIE_ENGINE:-wgpu}
work=${PIE_LAVAPIPE_WORK:-$(mktemp -d)}
export PIE_HOME="$work/home"
mkdir -p "$PIE_HOME"
port=${PIE_LAVAPIPE_PORT:-28419}
ws="ws://127.0.0.1:$port/v1/ws"
http="http://127.0.0.1:$port"

echo "== a random-weight snapshot"
tok=()
if [ -n "${PIE_TOKENIZER_DIR:-}" ]; then tok=(--tokenizer "$PIE_TOKENIZER_DIR"); fi
python3 scripts/tiny_qwen35.py "$work/snapshot" "${tok[@]}"

echo "== pie with the $engine engine"
cargo build --release -p pie --features "$engine"
pie=./target/release/pie
echo "== inferlets (wasm32-wasip2)"
(cd examples && cargo build -q --release --target wasm32-wasip2 -p text-completion -p naive-baseline -p sort-probe)

echo "== import"
"$pie" model import "$work/snapshot" --sku qwen35-tiny-u4g64-kv-bf16 --keep-source

case "$engine" in
  wgpu)   engine_toml=$'type = "wgpu"\nbackends = "vulkan"\ndevice = ["wgpu:0"]' ;;
  vulkan) engine_toml=$'type = "vulkan"\ndevice = ["vulkan:0"]\ndevice_index = 0' ;;
  *) echo "PIE_ENGINE must be wgpu or vulkan"; exit 2 ;;
esac
cat > "$PIE_HOME/config.toml" <<TOML
[server]
host = "127.0.0.1"
port = $port
telemetry = false
[model]
name = "default"
model = "snapshot"
[engine]
$engine_toml
max_total_pages = 512
max_forward_tokens = 1024
max_forward_requests = 8
max_state_slots = 16
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
PIE_BENCH_FIXTURE=1 node scripts/bench/browser/native.mjs "$ws" "$what" | grep -v "^RESULT" | cut -c1-160
echo "== sort-probe"
PIE_MATRIX_ENTRY_TIMEOUT_MS=900000 node tests/browser/tools/matrix-native.mjs "$ws" sort-probe-light > "$work/sort-probe.log" || true
grep "^MATRIX" "$work/sort-probe.log" | cut -f1,2,3 | sed 's/\\n/\n   /g' | cut -c1-160
grep -q $'^MATRIX sort-probe-light\tok\t' "$work/sort-probe.log" || { echo "sort-probe failed"; exit 1; }

echo "== compat API: one chat completion through the built-in inferlet"
curl -fsS -X POST "$http/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":4,"chat_template_kwargs":{"enable_thinking":false}}' \
  | python3 -c 'import json,sys; r=json.load(sys.stdin); assert r["choices"][0]["message"]["role"]=="assistant", r; print("chat completion:", json.dumps(r["choices"][0]["message"])[:120])'

if [ -n "${PIE_LANGUAGES_DIR:-}" ]; then
  echo "== a Python and a JavaScript inferlet, under their language components"
  for language in python javascript; do "$pie" language install "$PIE_LANGUAGES_DIR/pie-language-$language.tar.gz"; done
  "$pie" inferlet install examples/text-completion-py/main.py -m examples/text-completion-py/Pie.toml
  "$pie" inferlet install examples/text-completion-js/index.js -m examples/text-completion-js/Pie.toml
  for name in text-completion-py text-completion-js; do
    uv run --project python/client python scripts/ci/launch.py "$ws" "$name" | tee /dev/stderr | grep -q '^ok' || exit 1
  done
fi

sed 's/\x1b\[[0-9;]*m//g' "$work/serve.log" | grep -v "RPC{" | grep -E "ERROR|panic" | head -5 || true
echo "== done"
