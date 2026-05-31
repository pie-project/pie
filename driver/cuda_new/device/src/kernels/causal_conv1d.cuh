#pragma once

// Depthwise causal 1D convolution — the Mamba/SSM "conv" step (also used by
// GatedDeltaNet in-projection). This is the SELF-CONTAINED bf16 forward
// (full-prefill) variant lifted verbatim from
// driver/cuda/src/kernels/causal_conv1d.cu. The launch_ prefix is dropped and
// the namespace is pie_cuda_device::kernels.
//
// Only the single-request prefill/forward entry point is lifted here. The
// stateful single-token decode variants (update / update_batched) and the
// slot-cache batched prefill variant are NOT lifted: they require a persistent
// recurrent conv_state cache (per-slot [K, C] slabs, qo_indptr / slot_ids
// dispatch) rather than a self-contained forward. Those land with the rest of
// the Mamba/GatedDeltaNet state machinery in a focused pass.
//
//     y[t, c] = silu( sum_{k=0..K-1} W[c, k] * x[t - (K-1) + k, c]  + bias[c] )
//
// where x[t<0, c] is read from the prior state window (zeroed for a fresh
// prompt, which yields plain causal zero-padding).
//
// Memory layout (channels-last / time-major):
//
//     x       : [N, C]    bf16 — N tokens, C channels
//     weight  : [C, K]    bf16 — per-channel kernel of length K
//     bias    : [C]       bf16 — optional per-channel bias (nullptr = none)
//     state   : [K, C]    bf16 — last K input rows, oldest first (optional)
//     y       : [N, C]    bf16 — conv output

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_device::kernels {

// Single-request prefill. `state_out` may be nullptr to skip persisting
// the trailing K-window into a state buffer (and to treat the pre-window
// as zero-padding for a fresh prompt).
void causal_conv1d_prefill_bf16(
    const void* x,
    const void* weight,
    const void* bias,
    void*       y,
    void*       state_out,
    int N,
    int C,
    int K,
    cudaStream_t stream);

}  // namespace pie_cuda_device::kernels
