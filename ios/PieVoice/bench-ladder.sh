#!/bin/bash
# Model-ladder benchmark for PieVoice.
#
#   bash bench-ladder.sh sim  <simulator-udid>
#   bash bench-ladder.sh dev  <device-udid>
#
# The app must already be installed (deploy-device.sh for a phone; Xcode or
# simctl for the Simulator). For each rung: pushes the model's artifact
# directory into the app's Documents/models container, launches in
# benchmark mode pinned to that rung, and captures the PIEBENCH json lines.
# Results land in results/<target>-<timestamp>.jsonl.
#
# Pie 0.5 ladder. A model is a directory under $PIE_STORE/models, named as
# `pie model list` prints it and holding one .zt artifact. Import each rung
# on the Mac first; a rung that is not imported is skipped:
#     pie model import Qwen/Qwen3.5-0.8B     # bundled by deploy-device.sh
#     pie model import Qwen/Qwen3.5-2B
#     pie model import Qwen/Qwen3.5-4B
# Only the 0.8B rung has been through the 0.5 deploy path so far; the 2B
# and 4B entries are the intended ladder, not measured ones.
set -uo pipefail

MODE=${1:?usage: bench-ladder.sh sim|dev <udid>}
UDID=${2:?missing udid}
BUNDLE=org.pie-project.voice
PIE_STORE=${PIE_STORE:-${PIE_HOME:-$HOME/random/pie-05-home}}
BUNDLED_MODEL=${BUNDLED_MODEL:-Qwen--Qwen3.5-0.8B}
LADDER=${LADDER:-"Qwen--Qwen3.5-0.8B Qwen--Qwen3.5-2B Qwen--Qwen3.5-4B"}
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

push_model() {  # $1 = artifact directory name
  local src="$PIE_STORE/models/$1"
  ls "$src"/*.zt >/dev/null 2>&1 || { echo "skip $1 (not imported under $PIE_STORE/models)"; return 1; }
  # Every file of the directory: a sharded artifact is a root plus
  # numbered shards and the engine expects them side by side.
  if [ "$MODE" = sim ]; then
    local docs; docs=$(container_docs) || return 1
    mkdir -p "$docs/models/$1"
    for f in "$src"/*; do
      [ -f "$f" ] || continue
      # Hardlink when possible: a multi-gigabyte copy per rung is pure waste.
      ln -f "$f" "$docs/models/$1/$(basename "$f")" 2>/dev/null || cp "$f" "$docs/models/$1/$(basename "$f")"
    done
  else
    for f in "$src"/*; do
      [ -f "$f" ] || continue
      xcrun devicectl device copy to --device "$UDID" --domain-type appDataContainer \
        --domain-identifier "$BUNDLE" --source "$f" \
        --destination "Documents/models/$1/$(basename "$f")" >/dev/null 2>&1 || {
          echo "copy failed for $1/$(basename "$f")"; return 1; }
    done
  fi
}

clear_models() {
  if [ "$MODE" = sim ]; then
    local docs; docs=$(container_docs) || return 0
    rm -rf "$docs/models"
  fi
}

run_bench() {  # $1 = label  $2 = artifact directory name
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
  # Wait for a terminal record, the process dying, or 15 minutes (the
  # largest rung is slow).
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
    printf '{"event":"did_not_fit","model_dir":"%s","turns_completed":%s,"waited_s":%s}\n' \
      "$2" "$turns" "$waited" >> "$OUT"
  fi
  grep '^PIEBENCH ' "$log" | sed 's/^PIEBENCH //' >> "$OUT"
  grep -c '^PIEBENCH ' "$log" | xargs -I{} echo "  captured {} records for $1"
  grep -q '"event":"model_mismatch"' "$log" && echo "  !! MODEL MISMATCH — $1 not loaded"
  rm -f "$log"
}

for MODEL in $LADDER; do
  say "$MODEL"
  clear_models
  if [ "$MODEL" != "$BUNDLED_MODEL" ]; then
    push_model "$MODEL" || continue
  fi
  run_bench "$MODEL" "$MODEL"
done

say "results: $OUT"
python3 "$(dirname "$0")/bench-report.py" "$OUT" 2>/dev/null || cat "$OUT"
