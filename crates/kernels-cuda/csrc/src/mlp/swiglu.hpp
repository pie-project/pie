#pragma once

// The two gated-MLP launchers the ahead-of-time archive still needs.
//
// This header used to declare sixteen — SwiGLU, GeGLU-tanh, SiTU, GPT-OSS
// and their clamped, strided and chunked forks. §43 deleted fourteen of
// them. TEN still have rows in `table::mlp` and `families::mlp`: every one
// of those rows is in `device.rs`'s `JIT_DISPATCHED`, so NVRTC compiles the
// kernel out of `mlp/swiglu.cuh` and the generated shim forwards to none of
// the launchers. The other four had no row anywhere and no caller in any
// language. The activations themselves are documented where they now run —
// `mlp/swiglu.cuh` and `families::mlp` — not here.
//
// Element-wise throughout. `gate`, `up`, `y` are bf16 row-major.

#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels::mlp {

// Elementwise `x[i] *= sigmoid(gate[i])`. Used by Qwen3.5 full-
// attention's per-token output gate (a' = a * σ(g)).
void sigmoid_gate_inplace_bf16(
    void*       x,      // bf16, in-place
    const void* gate,   // bf16, same shape as x
    int num_elements,
    cudaStream_t stream);

// Qwen3.6-MoE expert MLP fuses SwiGLU with the gate/up split — the
// expert's `gate_up_proj` GEMM produces a `[N, 2*I]` tensor where
// columns `[0, I)` are the gate features and `[I, 2*I)` are the up
// features. Read both halves of each row and emit `silu(gate) * up`
// directly into a `[N, I]` output, skipping the intermediate
// deinterleave that an unfused path would need.
//
//     y[n, i] = silu(packed[n, i]) * packed[n, I + i]
// `gate_second` selects the [linear|gate] order flashinfer's CUTLASS MoE
// requires; the default is HuggingFace's [gate|up].
void chunked_swiglu_bf16(
    const void* packed,  // [N, 2*I] bf16
    void*       y,       // [N, I]   bf16
    int N, int I,
    cudaStream_t stream,
    bool gate_second = false);

}  // namespace pie_cuda_driver::kernels::mlp
