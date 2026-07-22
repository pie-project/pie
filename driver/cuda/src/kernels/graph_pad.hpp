#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace pie_cuda_driver {

// Fill the persistent-input rows for the graph-lattice pad lanes
// [real_requests, real_requests + padding): one sacrificial `pad_page`
// each, one zero token, and row_valid = 0 (the KV-write kernels skip
// invalid rows). Reads kv_page_indptr[real_requests] ON DEVICE so the
// page CSR continues coherently for device-composed waves too. Ordered on
// `stream` after whatever wrote the real rows.
void launch_graph_pad_rows(
    std::uint32_t* qo_indptr,
    std::uint32_t* kv_page_indptr,
    std::uint32_t* kv_page_indices,
    std::uint32_t* kv_last_page_lens,
    std::uint32_t* tokens,
    std::uint32_t* positions,
    std::uint8_t* row_valid,
    std::uint8_t* custom_mask,
    std::int32_t* custom_mask_indptr,
    int real_mask_bytes,
    int real_requests,
    int real_tokens,
    int padding,
    std::uint32_t pad_page,
    cudaStream_t stream);

}  // namespace pie_cuda_driver
