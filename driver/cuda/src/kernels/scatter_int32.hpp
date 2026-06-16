#pragma once

// Scatter `num_samples` int32 values into a length-N output buffer at
// indices read from `d_indices`. Output positions not listed in
// `d_indices` are zeroed. Used by the topk+top-p sampling path to land
// per-sample-slot token IDs at their logit-row positions without a
// host roundtrip — keeps the whole sampling step graph-capturable.
//
// Semantics:
//   for (k = 0; k < num_samples; ++k):
//       d_out[d_indices[k]] = d_values[k]
//   (positions not in d_indices are left at the previously-written
//    zero from the zero pass.)

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

void launch_scatter_int32(
    std::int32_t* d_out,
    const std::int32_t* d_indices,
    const std::int32_t* d_values,
    int num_samples,
    int n_out,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
