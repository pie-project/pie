#!/usr/bin/env bash
# The three banked answers, as a gate rather than as a paragraph.
#
# Each is a checkpoint this tree can load and the argmax it produced when
# the SKU was first served end to end. They were the gates the whole
# model/kernel rewrite was measured against, and before they were a script
# they lived in a `println!` and in commit bodies -- a person reading a
# terminal was the only thing checking them.
#
# THEY NOW FIRE THROUGH THE SHELL. This script used to run `baker-smoke`,
# a binary that reached `kernels-cuda` directly and was therefore a second
# executor beside `driver-cuda`'s, with its own pools, staging and fire.
# Two executors over one set of kernels drift, and only one of them was ever
# measured against a checkpoint. The gate is now a driver-cuda test, so what
# it exercises is the path that actually serves.
#
# NEEDS A GPU AND THE CACHED CHECKPOINTS, which is why the test is
# `#[ignore]`d and why this is not part of the default workspace sweep.
set -euo pipefail

cd "$(dirname "$0")/.."

cargo test --quiet -p driver-cuda --features cuda-13,abi \
    --test banked_argmaxes -- --ignored --nocapture

echo
echo "banked-argmaxes: three SKUs answered what they were banked at."
