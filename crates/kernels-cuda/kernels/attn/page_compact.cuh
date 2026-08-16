//===-- page_compact.cuh - the paged-KV CSR compactor's device text -------===//
//
// Two `__global__`s, the survival predicate they share, and the two block
// collectives they fold through. No host code: `page_compact.cu` includes
// this and keeps both `<<<>>>`, so the ahead-of-time build and NVRTC compile
// ONE text -- which is the whole point of the split, because two copies that
// agree today are two kernels that drift, each right for whichever half of
// the tree its tests exercise. `norm/altup_aux` shipped exactly that for a
// release with every test green.
//
// # What this computes
//
// Quest (and any page-granular eviction policy) selects pages *for one
// layer's attention*, and FlashInfer already takes the page list as a launch
// argument -- so honouring a selection is a page-table gather. `count_kept`
// counts each request's survivors; `scan_and_scatter` turns those counts into
// output bases and emits the surviving page ids in their original order. The
// three invariants are `page_compact.hpp`'s and unchanged: order is preserved,
// the last page is always kept, and no request compacts to zero pages.
//
// # Why `<cub/cub.cuh>` is gone, and what replaced it
//
// This is the one file in the tree that reached into CCCL, and it is the
// reason `families/attn.rs` recorded it as *"not split at all"*: CUB is
// 13.7 MB in 1,691 files, NVRTC answers no external include (measured: 0 of
// 31, not even `<cstdint>`), and §13.5 closed the door on carrying it. So the
// two collectives this file used -- `cub::BlockReduce<u32, 256>::Sum` and
// `cub::BlockScan<u32, 256>::ExclusiveSum` -- are written out below, in
// twenty-six lines, against `__shfl_down_sync` and `__shfl_up_sync`, which
// NVRTC has as builtins.
//
// **That is a body change, and §8 wants evidence for one.** The evidence is
// stronger here than a tolerance could ever be, and the reason is the element
// type: both collectives fold `u32` under `+`, which is exact and associative
// modulo 2^32, so ANY correct fold order produces the same bits. It is not
// "close enough to CUB"; it is the same integer. The harness measured it
// anyway -- see the report -- because an argument that a rewrite must agree
// is not a measurement that it does.
//
// Two contract details carried across rather than tidied:
//
//  * `block_sum_u32` leaves its result in **thread 0 only**, which is what
//    `BlockReduce::Sum` promises. Both call sites read it under
//    `if (threadIdx.x == 0)`, so widening the promise would be a helper doing
//    more than its callers ask and its name says.
//  * `block_exclusive_sum_u32` gives EVERY thread its exclusive prefix and
//    the block aggregate, which is what `BlockScan::ExclusiveSum` promises,
//    and it ends on a `__syncthreads()` so the caller may reuse the same
//    scratch on the next tile -- the loop in `scan_and_scatter` does exactly
//    that, once per 256 pages.
//
// # Which launcher becomes a row, and which does not
//
// Neither, yet, and the shape says why. Both launch `<<<num_requests,
// 256>>>`: one block per REQUEST, not per row of anything, with the block
// striding over that request's page list. No ported rule opens a grid over
// requests -- `Rms` is one block per row at 256 but asks for 32 bytes of
// shared memory and means a reduction, and `Elementwise` would open
// `ceil(n/256)` blocks, which for `scan_and_scatter`'s in-block running total
// is not a slower answer but a wrong one. Naming either would be inventing a
// rule under an existing name.
//
// The two are also ORDERED: `scan_and_scatter` reads the `counts` buffer
// `count_kept` fills, on the same stream. A row per kernel states two
// geometries and no dependency, so whatever states these will have to state
// that too.
//
// # Linkage: both are templates, and `BLOCK` is not a decoration
//
// §21.6 measured that a `.cuh` holding a NON-template `__global__` may be
// included by exactly one translation unit -- the host stub and the function
// both take external linkage, so a second includer is a hard `multiple
// definition` at link even when it never launches it. Both kernels here are
// `template <int BLOCK>`, so that constraint does not apply to this header
// and any number of units may carry it.
//
// The parameter is not invented for linkage. It was already there, spelled as
// a file-scope `constexpr int kBlock = 256` that reached the kernels as
// `cub::BlockReduce<u32, kBlock>` -- a COMPILE-TIME width, baked into the
// collectives' shared layout, distinct from the `blockDim.x` the stride loops
// use. Writing the collectives out is what turned that hidden argument into a
// stated one. There is no element type to abstract over -- every buffer here
// is `u32` or `u8` page-table metadata -- so `BLOCK` is the only parameter,
// and `<256>` is the only value any launcher instantiates.
//
// It is a value the kernel is compiled AGAINST, so §17.6's rule applies to
// whoever writes the row: take it from the ahead-of-time launcher and cite
// it. Compiled at 256 and launched at 128, `block_sum_u32` folds four warp
// partials that were never written and `scan_and_scatter` scatters against a
// tile offset that does not exist -- a plausible page list, not a fault. The
// negative control in the report measures exactly that.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "prelude/device.cuh"

namespace pie::attn {

/// The block width both kernels are compiled at and both launchers open.
///
/// Named here rather than in the `.cu` because it is a TEMPLATE ARGUMENT of
/// the two collectives below -- it sizes their shared scratch and fixes the
/// number of warp partials they fold -- so a launcher that opened a different
/// width would read scratch the kernel never wrote. `page_compact.cu` spells
/// it `kBlock` in both `<<<>>>`, which is the same constant.
constexpr int kBlock = 256;

/// Block-wide sum of `x`, **in thread 0**, through `scratch[BLOCK / 32]`.
///
/// `cub::BlockReduce<u32, BLOCK>::Sum` by hand. Thread 0 only, because that
/// is what CUB promises and both callers read it there.
///
/// The fold order is not part of the contract here, and this is the one place
/// in the tree where that sentence is true rather than a shortcut: `u32`
/// addition is exact and associative modulo 2^32, so warp-shuffle-then-serial
/// and CUB's raking layout sum to the same bits. A float reduction in this
/// shape would have to carry the original's order, which is why
/// `attn_res.cuh` does.
template <int BLOCK>
__device__ __forceinline__ u32 block_sum_u32(u32 x, u32* scratch) {
    constexpr int kWarps = BLOCK / 32;
    for (int off = 16; off > 0; off >>= 1) {
        x += __shfl_down_sync(0xffffffffu, x, off);
    }
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) scratch[warp] = x;
    __syncthreads();
    u32 total = 0;
    if (threadIdx.x == 0) {
        for (int w = 0; w < kWarps; ++w) total += scratch[w];
    }
    __syncthreads();
    return total;
}

/// Block-wide exclusive prefix sum of `x`, plus the block total, through
/// `scratch[BLOCK / 32 + 1]`.
///
/// `cub::BlockScan<u32, BLOCK>::ExclusiveSum(x, excl, aggregate)` by hand:
/// every thread gets its own `excl`, every thread gets the same `aggregate`.
///
/// The trailing `__syncthreads()` is load-bearing rather than defensive. The
/// caller tiles a page list 256 at a time and calls this once per tile with
/// the same `scratch`, so without it a fast warp's write of the next tile's
/// warp total could land before a slow warp has read this tile's offset.
template <int BLOCK>
__device__ __forceinline__ void block_exclusive_sum_u32(
    u32 x, u32& excl, u32& aggregate, u32* scratch) {
    constexpr int kWarps = BLOCK / 32;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;

    u32 inclusive = x;
    for (int off = 1; off < 32; off <<= 1) {
        const u32 up = __shfl_up_sync(0xffffffffu, inclusive, off);
        if (lane >= off) inclusive += up;
    }
    if (lane == 31) scratch[warp] = inclusive;
    __syncthreads();
    // `kWarps` is 8 at the one width this file is compiled for, so the serial
    // scan of the warp totals is eight adds on one thread -- cheaper than a
    // second shuffle pass and, being a single thread, trivially ordered.
    if (threadIdx.x == 0) {
        u32 running = 0;
        for (int w = 0; w < kWarps; ++w) {
            const u32 t = scratch[w];
            scratch[w] = running;
            running += t;
        }
        scratch[kWarps] = running;
    }
    __syncthreads();
    excl = inclusive - x + scratch[warp];
    aggregate = scratch[kWarps];
    __syncthreads();
}

/// Whether page `p` of a request survives.
///
/// The last page is unconditional: it holds the token this fire is writing,
/// and keeping it in place is what makes the compacted list's
/// `last_page_len` -- and the `kv_len = (pages-1)*page_size + last_page_len`
/// identity built on it -- carry over from the original CSR untouched.
///
/// A slot past the end of the mask row keeps its page. That cannot happen
/// while the stride bounds every request's page count, but the stride comes
/// from a host CSR and the count from device geometry, so the check is what
/// makes a disagreement an over-attend (a quality question) rather than an
/// out-of-bounds read (a correctness one).
__device__ __forceinline__ bool page_survives(
    const u8* __restrict__ keep,
    u32 row,
    u32 stride,
    u32 p,
    u32 pages) {
    if (p + 1 == pages) return true;
    if (p >= stride) return true;
    return keep[row + p] != 0;
}

/// One block per request: how many of its pages the mask keeps.
template <int BLOCK>
__global__ void count_kept(
    const u32* __restrict__ page_indptr_in,
    const u8* __restrict__ keep,
    u32 keep_stride,
    int num_requests,
    u32* __restrict__ counts) {
    const int r = static_cast<int>(blockIdx.x);
    if (r >= num_requests) return;
    const u32 beg = page_indptr_in[r];
    const u32 pages = page_indptr_in[r + 1] - beg;
    const u32 row = static_cast<u32>(r) * keep_stride;

    u32 local = 0;
    for (u32 p = threadIdx.x; p < pages; p += blockDim.x) {
        if (page_survives(keep, row, keep_stride, p, pages)) ++local;
    }
    __shared__ u32 tmp[BLOCK / 32];
    const u32 total = block_sum_u32<BLOCK>(local, tmp);
    if (threadIdx.x == 0) counts[r] = total;
}

/// Scan and scatter fused into one launch.
///
/// The only thing block `r` needed from the separate scan pass was its own
/// output base -- the exclusive prefix sum of the per-request counts -- and
/// with one block per request that prefix is at most `num_requests` values
/// long, so the block can just add them up itself. Recomputing an O(R) sum
/// per block is far cheaper than the kernel launch it replaces, because this
/// runs once per LAYER per fire.
template <int BLOCK>
__global__ void scan_and_scatter(
    const u32* __restrict__ page_indices_in,
    const u32* __restrict__ page_indptr_in,
    const u32* __restrict__ last_page_lens_in,
    const u8* __restrict__ keep,
    const u32* __restrict__ counts,
    u32 keep_stride,
    int num_requests,
    u32* __restrict__ page_indptr_out,
    u32* __restrict__ last_page_lens_out,
    u32* __restrict__ page_indices_out) {
    const int r = static_cast<int>(blockIdx.x);
    if (r >= num_requests) return;
    const u32 beg = page_indptr_in[r];
    const u32 pages = page_indptr_in[r + 1] - beg;
    const u32 row = static_cast<u32>(r) * keep_stride;

    __shared__ u32 red_tmp[BLOCK / 32];
    u32 partial = 0;
    for (int i = static_cast<int>(threadIdx.x); i < r;
         i += static_cast<int>(blockDim.x)) {
        partial += counts[i];
    }
    const u32 base_sum = block_sum_u32<BLOCK>(partial, red_tmp);
    __shared__ u32 out_beg;
    if (threadIdx.x == 0) {
        out_beg = base_sum;
        if (r == 0) page_indptr_out[0] = 0;
        page_indptr_out[r + 1] = base_sum + counts[r];
        // Invariant 2 keeps the last page last, so the tail length is carried
        // through verbatim rather than recomputed.
        last_page_lens_out[r] = last_page_lens_in[r];
    }
    __syncthreads();

    __shared__ u32 tmp[BLOCK / 32 + 1];
    __shared__ u32 running;
    if (threadIdx.x == 0) running = 0;
    __syncthreads();

    // Tiled so the emitted order matches the input order: within a tile the
    // exclusive scan ranks survivors, and `running` carries the count of
    // survivors from every earlier tile.
    for (u32 base = 0; base < pages; base += BLOCK) {
        const u32 p = base + threadIdx.x;
        const u32 flag =
            (p < pages && page_survives(keep, row, keep_stride, p, pages))
                ? 1u
                : 0u;
        u32 excl = 0;
        u32 aggregate = 0;
        block_exclusive_sum_u32<BLOCK>(flag, excl, aggregate, tmp);
        if (flag != 0) {
            page_indices_out[out_beg + running + excl] =
                page_indices_in[beg + p];
        }
        __syncthreads();
        if (threadIdx.x == 0) running += aggregate;
        __syncthreads();
    }
}

}  // namespace pie::attn
