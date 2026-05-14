#pragma once

// Token-id → hidden embedding lookup.
//
//     y[n, h] = weight[token_ids[n], h]
//
// `token_ids` is i32 (matches wire A_TOKEN_IDS dtype). `weight` and `y` are
// bf16 row-major.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

void launch_embed_bf16(
    const std::int32_t* token_ids,  // [num_tokens]
    const void* weight,             // [vocab, hidden]
    void* y,                        // [num_tokens, hidden]
    int num_tokens,
    int hidden,
    int vocab,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
