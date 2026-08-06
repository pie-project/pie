#include "kernels/graph_pad.hpp"

#include "cuda_check.hpp"

namespace pie_cuda_driver {

namespace {

// Writes coherent CSR rows for the graph-lattice pad lanes [R, R+padding).
// Each pad lane gets one sacrificial page (`pad_page`), its share of
// `pad_tokens` (ids 0, positions 0..tok-1), and `row_valid = 0` so the
// KV-write kernels skip it. On the decode path `pad_tokens == padding` and
// every lane carries exactly one token, which is the original behaviour; a
// prefill wave padding N to the token lattice hands lanes up to a page each.
// The share is recomputed here from (pad_tokens, padding) with the same
// formula `frame.cpp` uses for the host CSR copy — the two describe the same
// wave to the flashinfer plan and to the attention kernel, so they must agree
// exactly rather than be passed as an array that could drift. The
// kv-page CSR CONTINUES from the device-resident kv_page_indptr[R] — for a
// device-composed wave that value is device-only knowledge, which is why
// this must be a kernel and not a host memcpy: a host-padded copy would
// leave the device rows stale, and a stale kv_page_indptr[R+1] below the
// real total makes the attention kernel's per-row page count wrap negative
// (the V6 iteration-8 hang).
__global__ void k_graph_pad_rows(
    std::uint32_t* __restrict__ qo_indptr,
    std::uint32_t* __restrict__ kv_page_indptr,
    std::uint32_t* __restrict__ kv_page_indices,
    std::uint32_t* __restrict__ kv_last_page_lens,
    std::uint32_t* __restrict__ tokens,
    std::uint32_t* __restrict__ positions,
    std::uint8_t* __restrict__ row_valid,
    std::uint8_t* __restrict__ custom_mask,
    std::int32_t* __restrict__ custom_mask_indptr,
    int real_mask_bytes,
    int real_requests,
    int real_tokens,
    int padding,
    int pad_tokens,
    std::uint32_t pad_page) {
    const int j = static_cast<int>(threadIdx.x);
    if (j >= padding) return;
    // Lane j's share, and where it starts — the same split frame.cpp writes
    // into the host CSR copy.
    const int base = pad_tokens / padding;
    const int extra = pad_tokens % padding;
    const int tok = base + (j < extra ? 1 : 0);
    const int off = j * base + (j < extra ? j : extra);

    const std::uint32_t page_base = kv_page_indptr[real_requests];
    qo_indptr[real_requests + 1 + j] =
        static_cast<std::uint32_t>(real_tokens + off + tok);
    kv_page_indptr[real_requests + 1 + j] =
        page_base + static_cast<std::uint32_t>(1 + j);
    kv_page_indices[page_base + j] = pad_page;
    // kv_len == qo_len for the lane; `tok <= page_size` by construction, so
    // one page holds it and the last page is `tok` long.
    kv_last_page_lens[real_requests + j] =
        static_cast<std::uint32_t>(tok);
    for (int t = 0; t < tok; ++t) {
        tokens[real_tokens + off + t] = 0;
        positions[real_tokens + off + t] = static_cast<std::uint32_t>(t);
        row_valid[real_tokens + off + t] = 0;
    }
    if (custom_mask != nullptr && custom_mask_indptr != nullptr) {
        // Reachable only on the decode path, where tok == 1 for every lane
        // (frame.cpp keeps custom-mask waves off the token lattice).
        custom_mask[real_mask_bytes + j] = 1;
        custom_mask_indptr[real_requests + 1 + j] =
            real_mask_bytes + 1 + j;
    }
}

}  // namespace

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
    k_graph_pad_rows<<<1, padding, 0, stream>>>(
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
