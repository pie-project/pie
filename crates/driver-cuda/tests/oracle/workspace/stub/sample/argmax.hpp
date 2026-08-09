#pragma once
// Stub for the `kArgmaxAccumSlots` that `kernels-cuda/csrc/src/sample/
// argmax.hpp` used to publish. §54 deleted that header along with its
// launchers, which became `driver-cuda/src/fire/lm_head_argmax.rs`.
//
// `workspace.cpp` takes one constant from it and none of the launchers. The
// value is NOT retyped here: `run.sh` derives it from `kernels-cuda-new/csrc/
// src/sample/argmax.cuh`'s `kAccumThreads` — which is what `kAccumWarps`, and
// therefore `kArgmaxAccumSlots`, was always defined as — and fails if it
// moves, so this stub cannot drift from the number the shipping build uses.
namespace pie_cuda_driver::kernels::sample {
constexpr int kArgmaxAccumSlots = PIE_ARGMAX_ACCUM_SLOTS;
}
