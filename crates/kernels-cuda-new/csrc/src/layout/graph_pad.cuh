//===-- graph_pad.cuh - the graph-lattice pad lanes' CSR writer ------===//
//
// One `__global__`, and it is now JIT-ONLY. `graph_pad.cu` included this file
// and held `launch_graph_pad_rows`; §43 deleted the file whole, because that
// launcher had no row, therefore no shim entry, and no C++ caller either --
// the driver composed it from no statement. Exactly one definition still
// exists in the tree, and it is the one below.
//
// # Two spellings of the same integers
//
// The kernel was written against `<cstdint>` and now names the prelude's
// `u32`/`u8`/`i32` instead. That is a RENAME and not a conversion:
// `std::uint32_t` IS `unsigned int` on every target this tree builds for, so
// the launcher below goes on passing `std::uint32_t*` to a `u32*` parameter
// and the archive's object code is unchanged. The rename is forced -- NVRTC
// ships no standard library at all (the `stdlib_probe` measured 0 of 31
// standard headers answering), so a `.cuh` in the carried header set that
// said `std::uint32_t` would compile under nvcc and nowhere else.
//
// # Why it is not a row
//
// Geometrically it could be: `<<<1, padding>>>` with a `j >= padding` guard
// is `LaunchRule::RouteRows` at one row. It is not, because `RouteRows`
// launches one block PER ROW and reads the row count off the fire -- and this
// kernel must run exactly once, whatever the wave's shape, because every
// thread writes a different pad lane of one shared CSR. A fire with sixteen
// rows would run sixteen identical copies racing on the same words. There is
// no rule for "one block, whatever the rectangle", and §10.5 refuses an
// invented one.
//
// Nor is there a caller that could state it: `launch_graph_pad_rows` is
// composed by the driver while it builds a graph-captured wave, not by a
// model text, so no `Source` names its operands.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "pie_device.cuh"

namespace pie_cuda_driver::kernels::layout::device {

// The scalar layer is the PRELUDE's, not this family's. Named here so the
// kernels below read as they always did, so a row may keep spelling its
// element type `device::bf16`, and so the launchers in the enclosing
// namespace -- which write `device::` meaning the prelude's -- go on
// resolving to the same types through these declarations.
using ::pie_cuda_driver::kernels::device::i32;
using ::pie_cuda_driver::kernels::device::u32;
using ::pie_cuda_driver::kernels::device::u8;

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
__global__ void graph_pad_rows(
    u32* __restrict__ qo_indptr,
    u32* __restrict__ kv_page_indptr,
    u32* __restrict__ kv_page_indices,
    u32* __restrict__ kv_last_page_lens,
    u32* __restrict__ tokens,
    u32* __restrict__ positions,
    u8* __restrict__ row_valid,
    u8* __restrict__ custom_mask,
    i32* __restrict__ custom_mask_indptr,
    int real_mask_bytes,
    int real_requests,
    int real_tokens,
    int padding,
    int pad_tokens,
    u32 pad_page) {
    const int j = static_cast<int>(threadIdx.x);
    if (j >= padding) return;
    // Lane j's share, and where it starts — the same split frame.cpp writes
    // into the host CSR copy.
    const int base = pad_tokens / padding;
    const int extra = pad_tokens % padding;
    const int tok = base + (j < extra ? 1 : 0);
    const int off = j * base + (j < extra ? j : extra);

    const u32 page_base = kv_page_indptr[real_requests];
    qo_indptr[real_requests + 1 + j] =
        static_cast<u32>(real_tokens + off + tok);
    kv_page_indptr[real_requests + 1 + j] =
        page_base + static_cast<u32>(1 + j);
    kv_page_indices[page_base + j] = pad_page;
    // kv_len == qo_len for the lane; `tok <= page_size` by construction, so
    // one page holds it and the last page is `tok` long.
    kv_last_page_lens[real_requests + j] =
        static_cast<u32>(tok);
    for (int t = 0; t < tok; ++t) {
        tokens[real_tokens + off + t] = 0;
        positions[real_tokens + off + t] = static_cast<u32>(t);
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

}  // namespace pie_cuda_driver::kernels::layout::device
