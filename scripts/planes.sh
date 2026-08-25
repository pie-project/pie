#!/usr/bin/env bash
#
# WHICH PLANES FIRE ON THIS BOX, AND HOW FAR EACH ONE GETS.
#
# There are four device planes and they are at four different distances from
# serving a model. Each has a gate; this runs the ones this machine can run
# and says plainly why it skipped the rest. That is the point: a plane that
# is merely absent from a run reads exactly like a plane that passed, and
# `scripts/ci-gate-audit.py` exists because that mistake was made once already.
#
#   cuda    SERVES ALL THREE. Three checkpoints go text -> Program -> GPU ->
#           argmax through the Shell that actually serves, and answer what
#           they were banked at. `scripts/banked-argmaxes.sh` is that gate;
#           this calls it.
#
#   wgpu    SERVES TWO OF THE THREE. `qwen35-d0.8b-bf16-kv-bf16` walks 381
#           statements into 387 dispatches on one submission and answers 198 at
#           12.3125 on the decode lane and the prefill lane both;
#           `gptoss-20b-bf16-mxfp4-kv-bf16` answers 11 at 14.4375 over 579.
#           Both are what cuda banked. It is the second plane ever asked, and
#           walking real towers is what found the two things nothing narrower
#           could: `attention.{decode,prefill,masked}` could not bind here at
#           all, and one weight arena cannot hold a model past about 2B
#           parameters on any shader plane.
#
#           gemma-4 is the one left and what stops it is ONE BANK. Its
#           `ple.table` is [262144, 10752] in bf16 -- 5.25 GiB in a single
#           tensor -- and `layout.embed` binds it whole. Splitting the arena
#           cannot help: this is one bank past what one allocation may be
#           BOUND at, and the largest ceiling any shader plane states is 4 GiB.
#           Closing it is a change to the POINT, not to a driver.
#
#   vulkan  COMPUTES THROUGH THE POINTS PATH, and has not been driven end to
#           end yet. A device opens, every module becomes a real pipeline, and
#           a `norm.rmsnorm` goes Plan -> Program -> walk -> generated dispatch
#           -> claim body -> vkCmdDispatch and lands inside one bf16 ulp of an
#           f64 reference. `attention.sink` here is BIT-IDENTICAL to the CUDA
#           twin over all 96 words, checked against a third harness built with
#           nvcc rather than against carried constants. What it is short of is
#           a `serve::run` of its own -- not a kernel.
#
#   metal   SERVES TWO OF THE THREE, on an Apple box and only there.
#           `gptoss-20b-bf16-mxfp4-kv-bf16` answers 11 at 14.4375 EXACTLY, and
#           `qwen35-d0.8b-bf16-kv-bf16` answers token 198 at 12.2500 -- the
#           banked token, one bf16 step under the banked logit, which is what
#           an Apple GPU reducing in its own order is worth. Both through
#           `serve::Shell`, the path a request actually takes.
#
#           Getting there took two defects nothing narrower could see: a
#           `kv_geometry` that stated two layouts in one pair, and every
#           `InOut` copy staged on the HOST before a single dispatch had run --
#           so the residual stream was added to zeros from layer 0 onward, in
#           silence.
#
#           It also COMPILES AND BUILDS PIPELINES.
#           `scripts/check-metal-4.sh` type-checks the Apple half against
#           `aarch64-apple-darwin` from Linux and that is all it does -- a
#           green metal column HERE is not a working plane. On a Mac the
#           suite is real: 366 tests plus 53 device-only, and every one of
#           the 293 declared entrypoints becomes a pipeline. Metal compiles
#           MSL at RUN time through the framework, so no `metal`/`metallib`
#           CLI and no Xcode is needed -- Command Line Tools and a cargo
#           toolchain are enough.
#
#           `tailscale ssh ingim@ins-mac-studio` reaches one. Plain TCP 22 is
#           filtered by the tailnet ACL and Tailscale's own SSH is not; that
#           distinction cost a day of believing the plane was untestable.
#
# `scripts/claims-census.py` is the OTHER table -- what each SKU states against
# what each plane claims -- DERIVED rather than written down, because the hand
# version was wrong three times in one day and in both directions. As of
# 2026-08-25 it prints: every point every banked SKU states is claimed on every
# plane. What separates the four is no longer arithmetic; it is SERVING.
#
# VULKAN NEEDS A SLANG COMPILER and this tree does not vendor one. Point
# `PIE_SLANGC` at a `slangc` binary, or put one on PATH. The releases live at
# https://github.com/shader-slang/slang/releases — this gate was measured
# against 2026.16, `slang-<ver>-linux-x86_64.tar.gz`, whose `bin/slangc` runs
# with no further install. Compiling the `.slang` tree takes about 3m20s cold.
set -uo pipefail

cd "$(dirname "$0")/.."

pass=0
skip=0
fail=0

# Run one plane's gate, or say why not. Never silently absent.
plane() {
    local name=$1 why=$2
    shift 2
    printf '\n== %s\n' "$name"
    if [ -n "$why" ]; then
        printf '   SKIPPED: %s\n' "$why"
        skip=$((skip + 1))
        return
    fi
    if "$@"; then
        printf '   %s: fired\n' "$name"
        pass=$((pass + 1))
    else
        printf '   %s: FAILED\n' "$name"
        fail=$((fail + 1))
    fi
}

have_slangc() {
    [ -x "${PIE_SLANGC:-}" ] || command -v slangc >/dev/null 2>&1
}

# ── cuda ────────────────────────────────────────────────────────────────
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    plane cuda "" bash scripts/banked-argmaxes.sh
else
    plane cuda "no CUDA device (nvidia-smi finds none)"
fi

# ── vulkan ──────────────────────────────────────────────────────────────
if ! have_slangc; then
    plane vulkan "no Slang compiler; set PIE_SLANGC (see this file's head)"
elif [ ! -e /lib/x86_64-linux-gnu/libvulkan.so.1 ] && ! ldconfig -p 2>/dev/null | grep -q libvulkan; then
    plane vulkan "no Vulkan loader on this box"
else
    plane vulkan "" cargo test --quiet -p driver-vulkan --features device
fi

# ── wgpu ────────────────────────────────────────────────────────────────
plane wgpu "" cargo test --quiet -p driver-wgpu --features native

# ── metal ───────────────────────────────────────────────────────────────
if [ "$(uname -s)" = "Darwin" ]; then
    plane metal "" cargo test --quiet -p driver-metal --features metal-4
else
    plane metal "not an Apple machine; \`scripts/check-metal-4.sh\` only TYPE-CHECKS the gated half"
fi

printf '\nplanes: %d fired, %d skipped, %d FAILED\n' "$pass" "$skip" "$fail"
[ "$fail" -eq 0 ]
