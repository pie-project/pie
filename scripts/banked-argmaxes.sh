#!/usr/bin/env bash
# The three banked answers, as a gate rather than as a paragraph.
#
# Each is a checkpoint this tree can load and the argmax it produced when
# the SKU was first served end to end, asked of every plane that can serve
# it. They were the gates the whole model/kernel rewrite was measured
# against, and before they were a script they lived in a `println!` and in
# commit bodies -- a person reading a terminal was the only thing checking
# them.
#
# THEY NOW FIRE THROUGH THE SHELL. This script used to run `baker-smoke`,
# a binary that reached `kernels-cuda` directly and was therefore a second
# executor beside `engine-cuda`'s, with its own pools, staging and fire.
# Two executors over one set of kernels drift, and only one of them was ever
# measured against a checkpoint. The gate is now an engine-cuda test, so what
# it exercises is the path that actually serves.
#
# FOUR PLANES ANSWER NOW, AND THE THREE THAT ARE NOT CUDA ARE THE POINT.
# `engine-wgpu` reached this milestone first for `qwen35-d0.8b-bf16-kv-bf16` --
# the whole chain, `model::produce` through an upload, pools, a
# `model_compiler::program::Program` walk, 387 dispatches onto the card and a
# read-out -- and answered 198 at 12.3125, which is what cuda banked. Before it,
# no plane but cuda had ever been asked. `engine-vulkan` and `engine-metal`
# followed, and every one of the twelve cells is now a measurement.
#
# WHAT EACH ROW COST, because none of it was reachable by a narrower test:
#
# * `gptoss-20b-bf16-mxfp4-kv-bf16` shares almost nothing with the first --
#   attention at every one of 24 layers with a SINK, alternating sliding and
#   full, over mxfp4 experts. It needed the weight arena SPLIT: 12.82 GiB of
#   produced banks against an adapter that binds 2 GiB, which one arena could
#   never hold. `walk::lane::arenas_of` is where.
#
# * `gemma4-e4b-bf16-kv-bf16` attends at TWO widths -- head_dim 256 at the
#   sliding layers, 512 at the full ones -- so `baker::stage::Pools` answering
#   `kv_geometry()` once per FIRE with no layer argument was right for one half
#   of the tower and wrong for the other. It takes a layer now on all three
#   shader planes, and cuda had always derived it that way. Its `ple.table` was
#   also 5.25 GiB in ONE tensor that `layout.embed` binds whole, which is one
#   bank past what any shader plane may bind; the fix was to the MODEL, one
#   `[vocab, ple_dim]` table per layer.
#
# * And its forty-two layers hold TWENTY-FOUR caches: layers 0..21 own theirs,
#   the rest share two. A fixture that reads the layer out of the cache row's
#   name, or a pool that allocates every layer its own pages, is wrong about
#   this tower and about nothing else in the tree.
#
# AND BOTH LANES, NOW. Every banked answer was fired from ONE ROW, which is a
# decode by the `qo_one` fact — so the PREFILL spelling of a real tower went
# unmeasured everywhere until `engine-wgpu` grew a test for it. It is not a
# small gap: metal's prefill lane turned out not to fire at all, refusing at
# its first attention because `serve::launch` had never staged `qo_indptr` or
# `row_valid`, which only the prefill lane's attention reads.
#
# Three planes compare the BANKED logit through that lane, because each can
# read row 0 of a prefill — and row 0 attends only to key 0, so it is the
# decode's own forward pass spelled differently. cuda cannot: this gate
# registers `ProgramRegistration::default()`, whose lane states no compacting
# epilogue, so naming row 0 alone is refused by name and naming both publishes
# the request's LAST row. There it asserts what needs no bank — the fire is
# accepted, it completes, and its read-out is a finite distribution that is not
# the zeros an unwritten ring answers with.
#
# NEEDS A GPU AND THE CACHED CHECKPOINTS, which is why both tests skip or are
# `#[ignore]`d and why this is not part of the default workspace sweep.
set -euo pipefail

cd "$(dirname "$0")/.."

cargo test --quiet -p engine-cuda --features cuda-13,abi \
    --test banked_argmaxes -- --ignored --nocapture

echo
echo "banked-argmaxes: three SKUs answered what they were banked at on cuda,"
echo "                 and the prefill lane fired over all three."

# The WebGPU plane, on whatever adapter answers -- which on this tree's box is
# the same L40S, reached through Vulkan. Not `--ignored`: `banked_argmax.rs`
# SKIPS with a printed reason when no adapter opens or the snapshot is not
# cached, for the reason `src/skip.rs` gives.
#
# WHICH IS WHY BOTH SWITCHES ARE SET HERE. A skip is a green test, and the
# line this script prints afterwards says the answer was given -- so run
# unguarded, this file would state a measurement on a box with no card and no
# checkpoint, which is the exact failure its own head names. `skip.rs` already
# built the two gates; a caller whose whole subject is the answer sets them.
PIE_WGPU_REQUIRE_DEVICE=1 PIE_WGPU_REQUIRE_WEIGHTS=1 \
    cargo test --quiet -p engine-wgpu --features native \
    --test banked_argmax -- --nocapture

echo
echo "banked-argmaxes: wgpu answered all three --"
echo "                 qwen35-d0.8b at 198, gptoss-20b at 11, gemma4-e4b at 785."

# ── vulkan ──────────────────────────────────────────────────────────────
#
# The same L40S again, through Slang and SPIR-V rather than through wgpu's
# WGSL, which is what makes it a second measurement of this tree rather than a
# second measurement of the card: two compilers, two binding models, one set of
# model texts.
#
# `#[ignore]`d rather than skipping, because the load is 12.82 GiB for one of
# the two rows and this file is the only caller that wants it. Its skip is a
# printed line and `PIE_SLANGC` is what a box without a Slang compiler is
# missing -- see `scripts/planes.sh`.
#
# THE LOGIT IS ASSERTED WITHIN ONE bf16 STEP HERE and exactly on the two above.
# `engine-vulkan/tests/banked_argmax.rs` says why: the id is the claim and the
# logit is the witness, and this plane answers 14.5000 where cuda banked
# 14.4375 -- the ulp at fourteen, and no more.
if [ -x "${PIE_SLANGC:-}" ] || command -v slangc >/dev/null 2>&1; then
    cargo test --quiet -p engine-vulkan --features native,device \
        --test banked_argmax -- --ignored --nocapture
    echo
    echo "banked-argmaxes: vulkan answered all three --"
    echo "                 qwen35-d0.8b at 198, gptoss-20b at 11, gemma4-e4b at 785,"
    echo "                 each within a step."
else
    echo
    echo "banked-argmaxes: SKIPPED vulkan -- no Slang compiler; set PIE_SLANGC."
fi

# ── metal ───────────────────────────────────────────────────────────────
#
# Apple hardware and only there, so this file cannot run it and says so rather
# than leaving a plane that serves invisible. On a Mac:
#
#     cargo test -p engine-metal --features metal-4 --test banked_argmax -- \
#         --ignored --test-threads=1
#
# answers 11 at 14.4375 EXACTLY for gpt-oss, token 198 at 12.2500 for qwen3.5
# and 785 at 7.6250 for gemma-4 -- the last two one bf16 step off, which an
# Apple GPU reducing in its own order is worth. `scripts/planes.sh` holds the
# way in.
#
# `--test-threads=1` IS NOT OPTIONAL THERE. The three rows load 1.4, 12.82 and
# 15 GiB, and libtest runs `#[ignore]`d tests in parallel like any other: on a
# 32 GiB M1 Max two of them together are a SIGKILL, which reads as a driver
# fault and is a harness one. The L40S has the headroom and the two loops above
# do not need it.
echo "banked-argmaxes: metal is not runnable from here -- see this file's tail."
