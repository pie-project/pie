#pragma once
// Stub for kernels-cuda/csrc/src/sample/argmax.hpp.
//
// `workspace.cpp` takes one constant from it and none of the launchers. The
// value is NOT retyped here: `run.sh` greps it out of the real header and
// fails if it moves, so this stub cannot drift from the number the shipping
// build uses.
namespace pie_cuda_driver::kernels::sample {
constexpr int kArgmaxAccumSlots = PIE_ARGMAX_ACCUM_SLOTS;
}
