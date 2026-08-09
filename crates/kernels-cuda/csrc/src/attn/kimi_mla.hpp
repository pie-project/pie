#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels::attn {

void kimi_split_q_b_bf16(
    const void* q_b,
    void* q_nope,
    void* q_pe,
    int tokens,
    int heads,
    int qk_nope_dim,
    int qk_rope_dim,
    cudaStream_t stream);

// `kimi_split_kv_a_norm_bf16` WAS declared here. Its launcher is deleted:
// the row is routed to the JIT (`kernels_cuda_new::device::JIT_DISPATCHED`),
// so no shim entry is generated for it and the shim entry was the whole of
// its consumer set. The kernel is unaffected -- `attn/kimi_mla.cuh`'s
// `split_kv_a_norm` is the same text NVRTC compiles and is now its only
// compiler.

}  // namespace pie_cuda_driver::kernels::attn
