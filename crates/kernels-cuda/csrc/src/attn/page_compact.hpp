#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels::attn {

/// Compact a fire's paged-KV CSR down to the pages a page mask keeps.
///
/// Quest (and any page-granular eviction policy) selects pages *for one
/// layer's attention*. FlashInfer already takes the page list as a launch
/// argument, so honouring a selection is a page-table gather -- not a kernel
/// change. This produces the gathered table.
///
/// Inputs are the fire's own CSR; outputs are a fresh, tight CSR. The input is
/// never modified: it remains the source of truth for the KV append and for
/// `kv_len` (`kernels/geometry.cu`), and compacting it in place would corrupt
/// the cache.
///
/// Three invariants the kernel enforces, each of which is a silent-corruption
/// bug if dropped:
///
///  1. **Order is preserved.** Pages are emitted in their original relative
///     order. FlashInfer derives a position's absolute index from its slot in
///     the page list, so reordering would relabel every key.
///
///  2. **The last page is always kept**, whatever the mask says. It holds the
///     token this fire is writing, and keeping it last is what makes the
///     compacted list's `last_page_len` -- and therefore the
///     `kv_len = (pages-1)*page_size + last_page_len` identity -- carry over
///     unchanged from the original CSR.
///
///  3. **A request never compacts to zero pages.** Attention over an empty
///     page list is undefined; invariant 2 already guarantees at least one.
///
/// `keep` is `[num_requests, keep_stride]` u8, row-major; entry `[r, p]`
/// governs slot `p` of request `r`'s page list, and `keep_stride` must be at
/// least every request's page count. Addressing by request rather than by
/// `page_indptr_in` is deliberate: the mask is written by the host, which on
/// the decode-envelope path holds only a bound on that CSR, so the two must not
/// share an addressing scheme.
void compact_page_csr(
    const std::uint32_t* page_indices_in,
    const std::uint32_t* page_indptr_in,
    const std::uint32_t* last_page_lens_in,
    const std::uint8_t* keep,
    // Caller-owned scratch, at least `num_requests` uint32. Owned by the caller
    // because this runs once per LAYER per fire, and a cudaMallocAsync/FreeAsync
    // pair costs more than both kernels put together at decode batch sizes.
    std::uint32_t* scratch_counts,
    std::uint32_t keep_stride,
    int num_requests,
    std::uint32_t* page_indices_out,
    std::uint32_t* page_indptr_out,
    std::uint32_t* last_page_lens_out,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels::attn
