#pragma once

// SHARED OP ARMS — the executor's body, for the ops whose execution is
// already family-blind.
//
// The audit that started this merge found 13 of 23 op kinds present in
// both family executors, with the bodies differing only by OPERAND
// CONVENTION (which workspace buffer plays each role) and by the weights
// struct — never by arithmetic. Step 1 removed the weights difference (a
// binder); step 2 removed the walk's; this file is where an arm lands the
// moment its operands stop being a family's private convention.
//
// It starts with the arms that were ALREADY identical, character for
// character, in both executors — the strongest possible evidence that the
// executor wanted to be one file. The rest follow as the SSA value arena
// (the trace already carries `inputs`/`outputs`; what it does not carry is
// a buffer, because buffer assignment is a backend job that was written as
// family convention) replaces the routing conventions.

#include <cstdint>

#include "kernels/swiglu.hpp"
#include "model/workspace.hpp"

namespace pie_cuda_driver::model::declared {

// `Swiglu`: the packed-bank form when the MLP's gate/up matmul landed in
// the fused bank, the two-buffer form otherwise. `gate_up_used_fused` is
// the Matmul arm's own decision, carried forward — the trace states ONE
// packed matmul either way (see the binder's `gate_up`).
//
// Both executors held this arm character-for-character identical.
// `dst` is the traced value's slot once the caller has moved this island
// onto the arena; a caller that has not keeps passing its convention.
inline void arm_swiglu(Workspace& ws,
                       bool gate_up_used_fused,
                       void* dst,
                       int n,
                       int intermediate,
                       cudaStream_t stream) {
    if (gate_up_used_fused) {
        kernels::launch_chunked_swiglu_bf16(
            ws.gate_up_fused.data(), dst, n, intermediate, stream);
    } else {
        kernels::launch_swiglu_bf16(
            ws.gate.data(), ws.up.data(), dst, n * intermediate, stream);
    }
}

}  // namespace pie_cuda_driver::model::declared
