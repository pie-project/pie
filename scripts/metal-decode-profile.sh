#!/usr/bin/env bash
# Where a metal decode step goes, as a table rather than a paragraph.
#
# WHY THIS IS A SCRIPT. The accounting below was derived four times by hand
# across one session and two of those readings were wrong in the same way:
# a share of a tally multiplied by a per-unit guess. Both estimates missed by
# an order of magnitude -- "the staged in-place copies are worth 5.6 ms" (they
# were 0.46) and "arena renaming is worth 1.3 ms" (it was 0.0). The fix is the
# one this tree keeps arriving at: derive it, do not assert it.
#
# HOW IT MEASURES. `PIE_METAL_DROP=<symbols>` does not encode the dispatches
# whose symbol matches, and `PIE_METAL_DROP_AT=<indices>` does the same by
# position for families one symbol cannot tell apart. A dropped fire computes
# the WRONG ANSWER on purpose -- the same bargain `PIE_METAL_BARRIER_NONE`
# makes -- so what a family costs is the step without it. It is an UPPER BOUND
# on any fusion that would remove those dispatches: a fusion keeps the
# arithmetic and this does not.
#
# `PIE_METAL_TRACE_FIRE=1` splits each fire into its HOST half (the walk, the
# staging, the encode or ICB replay) and its GPU wait. Everything below is the
# wait, meaned over the last six decode fires of a short generation -- six
# because a longer run hits EOS and starts averaging the prefill fire in, which
# is how one earlier reading came back with a standard deviation of 72 ms.
#
# NEEDS A MAC and a release build with the metal driver:
#
#     cargo build --release -p pie --no-default-features --features driver-metal
#
# and an inferlet to drive it:
#
#     cd tests/inferlets/text-completion-bench
#     cargo build --target wasm32-wasip2 --release
#
# `.wiki/macos-bench.md` holds the readings this has produced and what they
# meant; this file only takes them.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIE="${PIE_BIN:-$ROOT/target/release/pie}"
BENCH="${PIE_INFERLET_DIR:-$ROOT/tests/inferlets/text-completion-bench}"
WASM="$BENCH/target/wasm32-wasip2/release/text_completion_bench.wasm"
CONFIG="${PIE_CONFIG:?set PIE_CONFIG to a toml whose [model] model is a snapshot directory}"

for f in "$PIE" "$WASM" "$BENCH/Pie.toml"; do
    [ -e "$f" ] || { echo "missing $f -- see this file's head" >&2; exit 1; }
done

step() {  # $1 = PIE_METAL_DROP value, $2 = label
    PIE_METAL_DROP="$1" PIE_METAL_TRACE_FIRE=1 "$PIE" --config "$CONFIG" run \
        --path "$WASM" --manifest "$BENCH/Pie.toml" -- \
        --prompt "The capital of France is" --max-tokens 16 >/tmp/pie-profile.txt 2>&1
    local n
    n=$(grep -m1 PIE_DROP /tmp/pie-profile.txt | awk '{print $2}' || true)
    grep "PIE_FIRE wait" /tmp/pie-profile.txt | tail -6 |
        awk -v l="$2" -v n="${n:-0}" -v b="${BASE:-0}" \
            '{s+=$3;c++} END{m=s/c; printf "  %-30s %8.3f ms  %7s  %s\n", l, m,
               (b>0 ? sprintf("%.3f", b-m) : "-"), (n>0 ? "["n"]" : "")}'
}

echo "the fire, and what each family of it costs"
echo "  (second column is the step WITHOUT that family; third is the family)"
echo
echo "  ONE RUN RESOLVES THE TOP TWO ROWS AND NOT THE REST. A fire's wait varies"
echo "  by about 0.15 ms between runs, so a family under half a millisecond is"
echo "  inside the spread and its number here is an order of magnitude, not a"
echo "  measurement -- run it several times before believing one. The two matvec"
echo "  families are 85% of the step and are resolved by any single run."
echo
BASE=$( { PIE_METAL_TRACE_FIRE=1 "$PIE" --config "$CONFIG" run --path "$WASM" \
    --manifest "$BENCH/Pie.toml" -- --prompt "The capital of France is" \
    --max-tokens 16 2>&1 || true; } | grep "PIE_FIRE wait" | tail -6 |
    awk '{s+=$3;c++} END{printf "%.6f", s/c}')
printf "  %-30s %8.3f ms\n\n" "baseline" "$BASE"

# The order is descending by what it cost when this was written, so a reader
# who stops early has read the part that matters.
step dense_gemv_t_bfloat16   "dense matvecs (qkv,o,router,lm)"
step mxfp4_qmv_routed        "mxfp4 expert matvecs"
step sdpa_paged              "paged attention"
step router_topk             "router top-k"
step add_bias_bfloat16       "add_bias"
step rms_single_row_bfloat16 "rmsnorm"
step neox_yarn               "rope"
step residual_add_bfloat16   "residual_add"
step expert_combine          "expert combine"
step packed_gptoss_swiglu    "swiglu"
step kv_append_paged         "kv append"
step split_qkv               "split_qkv"
step attn_sink_rescale       "attention sink"

echo
echo "and what the ordering costs, which no drop can show:"
for arm in "PIE_UNUSED=1:every edge (the tracker)" \
           "PIE_METAL_BARRIER_RAW_ONLY=1:RAW edges only" \
           "PIE_METAL_BARRIER_NONE=1:no ordering at all"; do
    env "${arm%%:*}" PIE_METAL_TRACE_FIRE=1 "$PIE" --config "$CONFIG" run \
        --path "$WASM" --manifest "$BENCH/Pie.toml" -- \
        --prompt "The capital of France is" --max-tokens 16 >/tmp/pie-profile.txt 2>&1
    grep "PIE_FIRE wait" /tmp/pie-profile.txt | tail -6 |
        awk -v l="${arm#*:}" '{s+=$3;c++} END{printf "  %-30s %8.3f ms\n", l, s/c}'
done
