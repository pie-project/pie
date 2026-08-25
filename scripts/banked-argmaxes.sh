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
# executor beside `driver-cuda`'s, with its own pools, staging and fire.
# Two executors over one set of kernels drift, and only one of them was ever
# measured against a checkpoint. The gate is now a driver-cuda test, so what
# it exercises is the path that actually serves.
#
# TWO PLANES ANSWER NOW, AND THE SECOND ONE IS THE POINT. `driver-wgpu`
# reached this milestone for `qwen35-d0.8b-bf16-kv-bf16` -- the whole chain,
# `model::produce` through an upload, pools, a `model_compiler::program::Program`
# walk, 387 dispatches onto the card and a read-out -- and answered 198 at
# 12.3125, which is what cuda banked. Before it, no plane but cuda had ever been
# asked.
#
# IT ANSWERS TWO OF THE THREE. `gptoss-20b-bf16-mxfp4-kv-bf16` followed and
# answered 11 at 14.4375 over 579 dispatches -- a tower that shares almost
# nothing with the first: attention at every one of 24 layers with a SINK,
# alternating sliding and full, over mxfp4 experts. Getting there needed the
# weight arena SPLIT: 12.82 GiB of produced banks against an adapter that binds
# 2 GiB, which one arena could never hold. `walk::lane::arenas_of` is where.
#
# The third is `gemma4-e4b-bf16-kv-bf16` and it is not here. Two things were in
# the way; one was real and is fixed, and the other is a bank.
#
# FIXED: it attends at two widths -- head_dim 256 at the sliding layers, 512 at
# the full ones -- and `baker::stage::Pools` answered `kv_geometry()` once per
# FIRE with no layer argument, so the strides a claim body read would have been
# right for one half of the tower and wrong for the other. It takes a layer now,
# on all three shader planes, and cuda had always derived it that way.
#
# NOT FIXED: `ple.table` is 5.25 GiB in ONE tensor and `layout.embed` binds it
# whole. That is not an arena problem -- it is one bank past what one allocation
# may be bound at, and no shader plane states a ceiling above 4 GiB. Closing it
# is a change to the POINT. `gemma4_attends_at_two_widths_and_is_blocked_on_one_
# bank` in `driver-wgpu/tests/banked_argmax.rs` measures both halves and fails
# the day either stops being true.
#
# It is a SEPARATE INVOCATION rather than more rows of the loop below because
# the two drivers are two crates with two feature sets, and because they do not
# cover the same set. A wgpu row for gemma-4 belongs here the day it fires, and
# its absence is meant to be visible.
#
# NEEDS A GPU AND THE CACHED CHECKPOINTS, which is why both tests skip or are
# `#[ignore]`d and why this is not part of the default workspace sweep.
set -euo pipefail

cd "$(dirname "$0")/.."

cargo test --quiet -p driver-cuda --features cuda-13,abi \
    --test banked_argmaxes -- --ignored --nocapture

echo
echo "banked-argmaxes: three SKUs answered what they were banked at on cuda."

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
    cargo test --quiet -p driver-wgpu --features native \
    --test banked_argmax -- --nocapture

echo
echo "banked-argmaxes: wgpu answered two of the three --"
echo "                 qwen35-d0.8b at 198/12.3125, gptoss-20b at 11/14.4375."
