//===-- gather_tokens.cuh - the compaction copy kernels --------------===//
//
// Two `__global__`s and the plan entry they read. `gather_tokens.cu` used to
// include this file and keep the host `launch` that picks between the two;
// `cd5cebd3d` deleted it, and the picking is `driver_internal::gather_tokens`
// now, over the root `src/layout.rs` declares for this file. One definition
// of each kernel, as before, and one launcher for both, as before.
//
// # Why neither is a row
//
// Both launch `dim3(num_ops, 1, num_layers)`: a THIRD grid axis over layers,
// and `blockIdx.x` indexes a plan entry rather than a row of a rectangle. No
// `LaunchRule` states a three-dimensional grid, and `new-horizon.md` §10.5
// refuses an invented one. The choice between them is a host test on stride
// alignment (`token_stride % 8 == 0`), which is the same shape of decision
// that keeps `layout::embed_bf16`'s vectorised form unmigrated -- an operand
// no `Source` can produce, because the value is read off a pointer the row
// does not have at planning time.
//
// Both reasons outlived the launcher that motivated them, which is why the
// Rust one is `driver_internal`'s and carries no `routine!` line: a routine is
// a symbol a trace may name, and neither of these has ever been nameable.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "prelude/device.cuh"

namespace pie::layout {

// One entry of the gather plan, mirroring the runtime `GatherOp` but over
// PHYSICAL page ids -- the host resolves slot id -> page id before launch.
// Copies `len` tokens from token offset `src_off` of page `src_page` to
// offset `dst_off` of page `dst_page`. `src_off + len <= page_size`, and the
// destination span stays inside one page because `compact` splits any run
// that would straddle a boundary.
//
// It lived HERE and not in `gather_tokens.hpp` because a kernel parameter
// type must be visible to whatever compiles the kernel, and NVRTC cannot open
// a host header: that file was `<cstdint>` and `<cuda_runtime.h>`, neither of
// which NVRTC ships, so it included this one instead. It went at `cd5cebd3d`
// with the `.cu`, and the argument survives both for the caller that is left:
// the DRIVER fills a device array of these records before the launch, and
// `driver_internal::gather_tokens` takes that buffer as an opaque address
// naming this declaration for its layout. A second copy of the struct spelled
// `std::uint32_t` -- or spelled again in Rust -- would be the same five words
// until someone reordered a field, and then it would be a silent mis-read of
// every op in the plan.
struct GatherTokenOp {
    u32 src_page;
    u32 src_off;
    u32 dst_page;
    u32 dst_off;
    u32 len;
};


// int4-vectorized copy (8 bf16 / 16 B per element): one block per (op, layer),
// grid-stride over the op's contiguous span for K and V. Used when every span
// base + length is 8-bf16-aligned (`token_stride % 8 == 0`), which holds for
// head_dim ∈ {64,128,256,512}.
__global__ void gather_i4(
    int4* __restrict__ k,
    int4* __restrict__ v,
    const GatherTokenOp* __restrict__ ops,
    i64 token_stride_i4,
    i64 page_stride_i4,
    i64 layer_stride_i4)
{
    const GatherTokenOp o = ops[blockIdx.x];
    const i64 layer_off = static_cast<i64>(blockIdx.z) * layer_stride_i4;
    const i64 span = static_cast<i64>(o.len) * token_stride_i4;
    const i64 sbase = layer_off +
        static_cast<i64>(o.src_page) * page_stride_i4 +
        static_cast<i64>(o.src_off) * token_stride_i4;
    const i64 dbase = layer_off +
        static_cast<i64>(o.dst_page) * page_stride_i4 +
        static_cast<i64>(o.dst_off) * token_stride_i4;
    for (i64 i = threadIdx.x; i < span; i += blockDim.x) {
        k[dbase + i] = k[sbase + i];
        v[dbase + i] = v[sbase + i];
    }
}

// Scalar bf16 fallback for a non-8-aligned token stride.
__global__ void gather_u16(
    u16* __restrict__ k,
    u16* __restrict__ v,
    const GatherTokenOp* __restrict__ ops,
    i64 token_stride,
    i64 page_stride,
    i64 layer_stride)
{
    const GatherTokenOp o = ops[blockIdx.x];
    const i64 layer_off = static_cast<i64>(blockIdx.z) * layer_stride;
    const i64 span = static_cast<i64>(o.len) * token_stride;
    const i64 sbase = layer_off +
        static_cast<i64>(o.src_page) * page_stride +
        static_cast<i64>(o.src_off) * token_stride;
    const i64 dbase = layer_off +
        static_cast<i64>(o.dst_page) * page_stride +
        static_cast<i64>(o.dst_off) * token_stride;
    for (i64 i = threadIdx.x; i < span; i += blockDim.x) {
        k[dbase + i] = k[sbase + i];
        v[dbase + i] = v[sbase + i];
    }
}

}  // namespace pie::layout
