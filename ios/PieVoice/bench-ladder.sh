#!/bin/bash
# Model-ladder benchmark for PieVoice.
#
#   bash bench-ladder.sh sim  <simulator-udid>
#   bash bench-ladder.sh dev  <device-udid>
#
# Installs the app once, then for each model: pushes the GGUF into the app's
# Documents container, launches in benchmark mode, and captures the
# PIEBENCH json lines. Results land in results/<target>-<timestamp>.jsonl
#
# Device runs need the app signed and installed (Xcode: pick your team, Run).
set -uo pipefail

MODE=${1:?usage: bench-ladder.sh sim|dev <udid>}
UDID=${2:?missing udid}
BUNDLE=org.pie-project.voice
MODELS_DIR=${MODELS_DIR:-/Users/aarushkandukoori/random/pie-models}
BUNDLED_MODEL=${BUNDLED_MODEL:-Qwen3-0.6B-Q4_K_M.gguf}
TURNS=${TURNS:-5}
OUT_DIR="$(dirname "$0")/results"
mkdir -p "$OUT_DIR"
STAMP=$(date +%Y%m%d-%H%M%S)
OUT="$OUT_DIR/$MODE-$STAMP.jsonl"

say(){ printf '\n=== %s ===\n' "$*"; }

container_docs() {
  if [ "$MODE" = sim ]; then
    local root
    root=$(xcrun simctl get_app_container "$UDID" "$BUNDLE" data 2>/dev/null) || return 1
    echo "$root/Documents"
  else
    echo "__DEVICE__"
  fi
}

push_model() {  # $1 = filename
  local src="$MODELS_DIR/$1"
  [ -f "$src" ] || { echo "skip $1 (not downloaded)"; return 1; }
  if [ "$MODE" = sim ]; then
    local docs; docs=$(container_docs) || return 1
    mkdir -p "$docs/qwen3-gguf"
    # Hardlink when possible: a 4.7 GB copy per rung is pure waste.
    ln -f "$src" "$docs/qwen3-gguf/$1" 2>/dev/null || cp "$src" "$docs/qwen3-gguf/$1"
  else
    xcrun devicectl device copy to --device "$UDID" --domain-type appDataContainer \
      --domain-identifier "$BUNDLE" --source "$src" \
      --destination "Documents/qwen3-gguf/$1" >/dev/null 2>&1 || {
        echo "copy failed for $1"; return 1; }
  fi
}

clear_models() {
  if [ "$MODE" = sim ]; then
    local docs; docs=$(container_docs) || return 0
    rm -rf "$docs/qwen3-gguf"
  fi
}

run_bench() {  # $1 = label  $2 = model filename
  local log; log=$(mktemp)
  if [ "$MODE" = sim ]; then
    xcrun simctl terminate "$UDID" "$BUNDLE" >/dev/null 2>&1
    sleep 1
    xcrun simctl launch --console-pty "$UDID" "$BUNDLE" \
      -PieBenchmark 1 -PieBenchmarkTurns "$TURNS" -PieModel "$2" -PieBenchPruneModels 1 >"$log" 2>&1 &
  else
    xcrun devicectl device process launch --device "$UDID" --console \
      "$BUNDLE" -PieBenchmark 1 -PieBenchmarkTurns "$TURNS" -PieModel "$2" -PieBenchPruneModels 1 >"$log" 2>&1 &
  fi
  local pid=$!
  # Wait for a terminal record, the process dying, or 15 minutes (8B is slow).
  local waited=0 died=0
  while [ $waited -lt 900 ]; do
    grep -qE '"event":"(done|model_mismatch)"' "$log" && break
    # A model too large for the device is killed by jetsam: the process
    # simply vanishes. Detect that instead of waiting out the timeout.
    if ! kill -0 $pid 2>/dev/null; then died=1; break; fi
    sleep 5; waited=$((waited+5))
  done
  kill $pid 2>/dev/null

  if [ $died -eq 1 ] && ! grep -q '"event":"done"' "$log"; then
    local turns; turns=$(grep -c '"event":"turn"' "$log")
    echo "  !! process died after $turns turn(s) — recording as did_not_fit"
    printf '{"event":"did_not_fit","model_file":"%s","turns_completed":%s,"waited_s":%s}\n' \
      "$2" "$turns" "$waited" >> "$OUT"
  fi
  grep '^PIEBENCH ' "$log" | sed 's/^PIEBENCH //' >> "$OUT"
  grep -c '^PIEBENCH ' "$log" | xargs -I{} echo "  captured {} records for $1"
  grep -q '"event":"model_mismatch"' "$log" && echo "  !! MODEL MISMATCH — $1 not loaded"
  rm -f "$log"
}

for MODEL in Qwen3-0.6B-Q4_K_M.gguf Qwen3-1.7B-Q4_K_M.gguf Qwen3-4B-Q4_K_M.gguf Qwen3-8B-Q4_K_M.gguf; do
  say "$MODEL"
  clear_models
  if [ "$MODEL" != "$BUNDLED_MODEL" ]; then
    push_model "$MODEL" || continue
  fi
  run_bench "$MODEL" "$MODEL"
done

say "results: $OUT"
python3 "$(dirname "$0")/bench-report.py" "$OUT" 2>/dev/null || cat "$OUT"
