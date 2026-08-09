//===-- graph_pad.cu - the graph-lattice pad launcher ----------------===//
//
// One host launcher and not one `__global__`: the device text is in
// `layout/graph_pad.cuh`, which this file includes so the archive and the JIT
// header set hold the SAME definition rather than two that drift.
//
//===----------------------------------------------------------------------===//

#include "layout/graph_pad.hpp"

#include "cuda_check.hpp"

// The `__global__` this launcher fires. ONE definition of it.
#include "layout/graph_pad.cuh"

namespace pie_cuda_driver {

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
    int pad_tokens,
    std::uint32_t pad_page,
    cudaStream_t stream) {
    if (padding <= 0 || pad_tokens < padding) return;
    kernels::layout::device::graph_pad_rows<<<1, padding, 0, stream>>>(
        qo_indptr,
        kv_page_indptr,
        kv_page_indices,
        kv_last_page_lens,
        tokens,
        positions,
        row_valid,
        custom_mask,
        custom_mask_indptr,
        real_mask_bytes,
        real_requests,
        real_tokens,
        padding,
        pad_tokens,
        pad_page);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace pie_cuda_driver
