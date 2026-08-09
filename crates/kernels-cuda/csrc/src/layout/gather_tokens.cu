//===-- gather_tokens.cu - the compaction launchers ------------------===//
//
// Three host functions and not one `__global__`: the device text is in
// `layout/gather_tokens.cuh`, which the host header includes, so the archive
// and the JIT header set hold the SAME definitions rather than two that
// drift.
//
//===----------------------------------------------------------------------===//

// The scalar layer and the fixed-width integer names, out of the prelude.
#include "pie_device.cuh"

// The launcher declarations, and the forward declaration of `GatherTokenOp`
// the host shim compiles against.
#include "layout/gather_tokens.hpp"

// The `__global__`s these launchers fire, and the definition of
// `GatherTokenOp`. ONE definition of each, in the file NVRTC can also read.
#include "layout/gather_tokens.cuh"

#include <cstdint>
#include <stdexcept>

namespace pie_cuda_driver::kernels::layout {

namespace {

// Picks the vectorised copy when both strides are 8-bf16 aligned. A HOST
// test on a stride, which is why neither kernel is a row: no `Source`
// produces "is this pointer's row stride a multiple of eight".
void launch(
    device::u16* k_pages, device::u16* v_pages,
    const GatherTokenOp* ops, int num_ops,
    int num_layers, device::i64 layer_stride_elems,
    int page_size, int num_kv_heads, int head_dim,
    cudaStream_t stream)
{
    if (num_ops <= 0 || num_layers <= 0) return;
    const device::i64 token_stride =
        static_cast<device::i64>(num_kv_heads) * head_dim;
    const device::i64 page_stride = token_stride * page_size;
    const int threads = 256;
    const dim3 grid(static_cast<unsigned>(num_ops), 1u,
                    static_cast<unsigned>(num_layers));

    if (token_stride % 8 == 0 && layer_stride_elems % 8 == 0) {
        device::gather_i4<<<grid, threads, 0, stream>>>(
            reinterpret_cast<int4*>(k_pages),
            reinterpret_cast<int4*>(v_pages),
            ops,
            token_stride / 8, page_stride / 8, layer_stride_elems / 8);
    } else {
        device::gather_u16<<<grid, threads, 0, stream>>>(
            k_pages, v_pages, ops,
            token_stride, page_stride, layer_stride_elems);
    }
}

}  // namespace

// `launch_gather_tokens_bf16` and `launch_gather_tokens_bf16_layers` were
// deleted here by §43: two thin wrappers over `launch` above, with no
// `<<<>>>` of their own, no row, no shim entry and no caller in any language.
// `launch` itself stays -- `rope_write_kv_bf16` calls it four times.

}  // namespace pie_cuda_driver::kernels::layout
