// The host launcher, and nothing else. The two `__global__`s, the survival
// predicate they share and the two block collectives they fold through live
// in `attn/page_compact.cuh` -- ONE definition, read by nvcc here and by
// NVRTC from the same text at run time.
//
// `<cub/cub.cuh>` is gone with them. This was the only file in the tree that
// reached into CCCL, which is why `families/attn.rs` recorded it as one of
// the two that "were not split at all": CUB is 13.7 MB in 1,691 files and
// NVRTC answers no external include, so `BlockReduce`/`BlockScan` are written
// out in the header against `__shfl_down_sync`/`__shfl_up_sync`. Both fold
// `u32` under `+`, which is exact and associative modulo 2^32, so the
// rewrite is the same integer and not a close one.
//
// Both `<<<num_requests, 256>>>` are exactly as they were. Neither is a row
// yet: no ported rule opens a grid over REQUESTS, and the two are ordered on
// one stream besides -- `scan_and_scatter` reads the `counts` buffer
// `count_kept` fills. See the header.
//
// The launcher's own types stay `std::uint32_t`, which is what
// `page_compact.hpp` declares and what the generated shim compiles against.
// It is `unsigned int` here and `device::u32` is the same type, so the
// pointers reach the kernel with no cast: the header's spelling is a device
// vocabulary, not a different ABI.
#include "attn/page_compact.cuh"

#include "attn/page_compact.hpp"

namespace pie_cuda_driver::kernels::attn {

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

    device::count_kept<device::kBlock>
        <<<num_requests, device::kBlock, 0, stream>>>(
            page_indptr_in, keep, keep_stride, num_requests, scratch_counts);
    device::scan_and_scatter<device::kBlock>
        <<<num_requests, device::kBlock, 0, stream>>>(
            page_indices_in, page_indptr_in, last_page_lens_in, keep,
            scratch_counts, keep_stride, num_requests, page_indptr_out,
            last_page_lens_out, page_indices_out);
}

}  // namespace pie_cuda_driver::kernels::attn
