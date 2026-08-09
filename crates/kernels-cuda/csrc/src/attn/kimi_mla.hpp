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

// The unfused `kimi_split_kv_a_bf16` this replaced was deleted once nothing
// called it: every caller wants the rmsnorm, and the fused form is the only
// one the table ever declared.
void kimi_split_kv_a_norm_bf16(
    const void* kv_a,
    const void* norm_weight,
    void* kv_c,
    void* k_pe,
    int tokens,
    int kv_lora_rank,
    int qk_rope_dim,
    float eps,
    cudaStream_t stream,
    // Row pitch of `kv_a`, for reading the kv half straight out of Kimi's
    // fused `q_a + kv_a` projection. 0 means the rows are exactly
    // `kv_lora_rank + qk_rope_dim` wide.
    int src_row_stride = 0);


}  // namespace pie_cuda_driver::kernels::attn
