#include <cub/cub.cuh>

#include "attn/page_compact.hpp"

namespace pie_cuda_driver::kernels::attn {
namespace {

constexpr int kBlock = 256;

// Whether page `p` of a request survives. The last page is unconditional: it
// holds the token this fire is writing, and keeping it in place is what makes
// the compacted list's `last_page_len` -- and the
// `kv_len = (pages-1)*page_size + last_page_len` identity built on it -- carry
// over from the original CSR untouched.
//
// A slot past the end of the mask row keeps its page. That cannot happen while
// the stride bounds every request's page count, but the stride comes from a
// host CSR and the count from device geometry, so the check is what makes a
// disagreement an over-attend (a quality question) rather than an out-of-bounds
// read (a correctness one).
__device__ __forceinline__ bool page_survives(
    const std::uint8_t* __restrict__ keep,
    std::uint32_t row,
    std::uint32_t stride,
    std::uint32_t p,
    std::uint32_t pages) {
    if (p + 1 == pages) return true;
    if (p >= stride) return true;
    return keep[row + p] != 0;
}

__global__ void k_count_kept(
    const std::uint32_t* __restrict__ page_indptr_in,
    const std::uint8_t* __restrict__ keep,
    std::uint32_t keep_stride,
    int num_requests,
    std::uint32_t* __restrict__ counts) {
    const int r = static_cast<int>(blockIdx.x);
    if (r >= num_requests) return;
    const std::uint32_t beg = page_indptr_in[r];
    const std::uint32_t pages = page_indptr_in[r + 1] - beg;
    const std::uint32_t row = static_cast<std::uint32_t>(r) * keep_stride;

    std::uint32_t local = 0;
    for (std::uint32_t p = threadIdx.x; p < pages; p += blockDim.x) {
        if (page_survives(keep, row, keep_stride, p, pages)) ++local;
    }
    using Reduce = cub::BlockReduce<std::uint32_t, kBlock>;
    __shared__ typename Reduce::TempStorage tmp;
    const std::uint32_t total = Reduce(tmp).Sum(local);
    if (threadIdx.x == 0) counts[r] = total;
}

// Scan and scatter fused into one launch. The only thing block `r` needed from
// the separate scan pass was its own output base -- the exclusive prefix sum of
// the per-request counts -- and with one block per request that prefix is at
// most `num_requests` values long, so the block can just add them up itself.
// Recomputing an O(R) sum per block is far cheaper than the kernel launch it
// replaces, because this runs once per LAYER per fire.
__global__ void k_scan_and_scatter(
    const std::uint32_t* __restrict__ page_indices_in,
    const std::uint32_t* __restrict__ page_indptr_in,
    const std::uint32_t* __restrict__ last_page_lens_in,
    const std::uint8_t* __restrict__ keep,
    const std::uint32_t* __restrict__ counts,
    std::uint32_t keep_stride,
    int num_requests,
    std::uint32_t* __restrict__ page_indptr_out,
    std::uint32_t* __restrict__ last_page_lens_out,
    std::uint32_t* __restrict__ page_indices_out) {
    const int r = static_cast<int>(blockIdx.x);
    if (r >= num_requests) return;
    const std::uint32_t beg = page_indptr_in[r];
    const std::uint32_t pages = page_indptr_in[r + 1] - beg;
    const std::uint32_t row = static_cast<std::uint32_t>(r) * keep_stride;

    using Reduce = cub::BlockReduce<std::uint32_t, kBlock>;
    __shared__ typename Reduce::TempStorage red_tmp;
    std::uint32_t partial = 0;
    for (int i = static_cast<int>(threadIdx.x); i < r;
         i += static_cast<int>(blockDim.x)) {
        partial += counts[i];
    }
    const std::uint32_t base_sum = Reduce(red_tmp).Sum(partial);
    __shared__ std::uint32_t out_beg;
    if (threadIdx.x == 0) {
        out_beg = base_sum;
        if (r == 0) page_indptr_out[0] = 0;
        page_indptr_out[r + 1] = base_sum + counts[r];
        // Invariant 2 keeps the last page last, so the tail length is carried
        // through verbatim rather than recomputed.
        last_page_lens_out[r] = last_page_lens_in[r];
    }
    __syncthreads();

    using Scan = cub::BlockScan<std::uint32_t, kBlock>;
    __shared__ typename Scan::TempStorage tmp;
    __shared__ std::uint32_t running;
    if (threadIdx.x == 0) running = 0;
    __syncthreads();

    // Tiled so the emitted order matches the input order: within a tile the
    // exclusive scan ranks survivors, and `running` carries the count of
    // survivors from every earlier tile.
    for (std::uint32_t base = 0; base < pages; base += kBlock) {
        const std::uint32_t p = base + threadIdx.x;
        const std::uint32_t flag =
            (p < pages && page_survives(keep, row, keep_stride, p, pages))
                ? 1u
                : 0u;
        std::uint32_t excl = 0;
        std::uint32_t aggregate = 0;
        Scan(tmp).ExclusiveSum(flag, excl, aggregate);
        if (flag != 0) {
            page_indices_out[out_beg + running + excl] =
                page_indices_in[beg + p];
        }
        __syncthreads();
        if (threadIdx.x == 0) running += aggregate;
        __syncthreads();
    }
}

}  // namespace

void compact_page_csr(
    const std::uint32_t* page_indices_in,
    const std::uint32_t* page_indptr_in,
    const std::uint32_t* last_page_lens_in,
    const std::uint8_t* keep,
    std::uint32_t* scratch_counts,
    std::uint32_t keep_stride,
    int num_requests,
    std::uint32_t* page_indices_out,
    std::uint32_t* page_indptr_out,
    std::uint32_t* last_page_lens_out,
    cudaStream_t stream) {
    if (num_requests <= 0 || scratch_counts == nullptr) return;

    k_count_kept<<<num_requests, kBlock, 0, stream>>>(
        page_indptr_in, keep, keep_stride, num_requests, scratch_counts);
    k_scan_and_scatter<<<num_requests, kBlock, 0, stream>>>(
        page_indices_in, page_indptr_in, last_page_lens_in, keep,
        scratch_counts, keep_stride, num_requests, page_indptr_out,
        last_page_lens_out, page_indices_out);
}

}  // namespace pie_cuda_driver::kernels::attn
