#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

// Average-pool consecutive tokens: out[i] = mean(in[i*ratio : (i+1)*ratio])
// Used by the compressor to reduce token count by compress_ratio.
void launch_average_pool_bf16(
    const void* input,       // [N, dim] BF16
    void* output,            // [N/ratio, dim] BF16
    int N,
    int dim,
    int ratio,
    cudaStream_t stream);

// Add Accumulated Positional Embedding (APE) to compressed KV.
// output[i] += ape[i % ratio]
void launch_add_ape_f32(
    void* data,              // [N_compressed, dim] BF16 (modified in-place)
    const float* ape,        // [ratio, dim] F32
    int N_compressed,
    int dim,
    int ratio,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
