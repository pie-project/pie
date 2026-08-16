//===-- dequant_wna16_tile.cuh - the INT4 decode, and the inferred verdict
//     that was wrong -----------------------------------------------------===//
//
// An ALTERNATIVE to `dequant_wna16.cuh`'s `dequant_wna16_int4b8`.
//
//     rows      out MB    tile        scalar      ratio
//        1        0.008   2.24 us      2.21 us    0.98x
//      128        1       2.36         4.52       1.91x
//    1,024        8       5.90        18.52       3.14x
//    8,192       67      35.05       123.05       3.51x
//   32,768      268     507.99       514.91       1.01x
//
// L40S sm_89, in_dim 4096, group 128, bf16 out. **Bit-identical at every
// size** -- 0 of 134,217,728 elements differ at the largest.
//
// # This bucket was the one the census only INFERRED, and the inference was
//   backwards
//
// `kernels/tile/alternatives.cuh` classified all 455 kernels into buckets
// with a measured representative each, except one: the quantiser bucket,
// where the verdict was "PRMT bit manipulation and M=1 GEMVs, both
// bandwidth-bound, so a wash". The file said it was inferred. It was, and it
// was wrong by 3.5x in the direction of doing nothing.
//
// The reason is visible once stated: **INT4 to bf16 is a 4x EXPANSION.** At
// 1,024 rows this reads 2 MB and writes 8 MB, and does a shift, a mask, a
// subtract and two converts per output element. It is neither read-bound nor
// arithmetic-free; it is exactly the shape `mlp/swiglu_tile.cuh` wins at,
// with more arithmetic. The scalar form gives each thread eight nibbles from
// one word and eight scattered stores; the tile form gives each lane one
// OUTPUT element, so the stores coalesce and the unpack is a tile op.
//
// The 1.01x at 32,768 rows is the roofline, at 268 MB written -- the fourth
// independent sighting of the line `swiglu_tile`, `rmsnorm_rasr_tile` and
// `topk_softmax_tile` already drew. The predicate below is bounded by it.
//
// # And two tile operations this tree had written off
//
// An earlier header in this spike said `operator<<` "is not among the tile
// builtins" and worked around it with a multiply by 65536. `cuda_tile.h` has
// `operator<<`, `operator>>`, `operator&`, `operator|` and `operator^`, and
// this kernel uses two of them. That claim was made once and repeated
// without being rechecked, which is how the shift ended up spelled as
// arithmetic three files away.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cuda_bf16.h>
#include <crt/cuda_tile.h>

namespace pie::quant {

namespace ct = ::cuda::tiles;

/// Whether to prefer this ALTERNATIVE to the scalar INT4 decode.
///
/// Up to the roofline, which here is the OUTPUT size: measured 3.51x at
/// 8,192 rows of in_dim 4096 (67 MB written) and 1.01x at 32,768 (268 MB).
/// The bound is the largest measured point where it is ahead, in elements,
/// for the reason `mlp/swiglu_tile.cuh` records -- a rounded byte figure
/// once excluded the very point it was measured at.
constexpr bool dequant_wna16_tile_preferred(long long rows, long long in_dim)
{
    return rows * in_dim <= (32LL << 20);
}

#ifndef BS
#define BS 1024
#endif
#ifndef IN_DIM
#define IN_DIM 4096
#endif
#ifndef GRP
#define GRP 128
#endif
using i32xBS = ct::tile<int, ct::shape<BS>>;
using f32xBS = ct::tile<float, ct::shape<BS>>;
using bfxBS  = ct::tile<__nv_bfloat16, ct::shape<BS>>;

/// One block per output row. Each lane owns one OUTPUT element and gathers
/// the word holding its nibble, so the unpack is a shift and a mask in the
/// tile domain rather than an eight-deep unrolled scalar loop.
__tile_global__ void dequant_wna16_tile(
    const int* __restrict__ _packed,
    const __nv_bfloat16* __restrict__ _scale,
    __nv_bfloat16* __restrict__ _out,
    int /*in_dim*/, int /*group_size*/)
{
    const int row = static_cast<int>(ct::bid().x);
    constexpr int words_per_row = IN_DIM / 8;
    constexpr int groups_per_row = IN_DIM / GRP;
    auto packed = ct::assume_aligned<16>(_packed + (long long)row * words_per_row);
    auto scale  = ct::assume_aligned<16>(_scale + (long long)row * groups_per_row);
    auto out    = ct::assume_aligned<16>(_out + (long long)row * IN_DIM);

    constexpr int nb = (IN_DIM + BS - 1) / BS;
    constexpr bool EVEN = (IN_DIM % BS) == 0;
    for (auto j : ct::irange(0, nb)) {
        auto k = ct::iota<i32xBS>() + j * BS;
        auto m = k < IN_DIM;
        // word index k/8, nibble index k%8 -- both are tile lanes.
        auto widx = k / ct::full<i32xBS>(8);
        auto lane = k - widx * ct::full<i32xBS>(8);
        i32xBS w;
        [[ using cutile : hint(1000, latency=1) ]]
        w = EVEN ? ct::load(packed + widx)
                 : ct::load_masked(packed + widx, m, ct::zeros<i32xBS>());
        auto nib = (w >> (lane * ct::full<i32xBS>(4))) & ct::full<i32xBS>(0xF);
        auto q = ct::element_cast<float>(nib - ct::full<i32xBS>(8));
        auto gidx = k / ct::full<i32xBS>(GRP);
        bfxBS sraw;
        [[ using cutile : hint(1000, latency=1) ]]
        sraw = EVEN ? ct::load(scale + gidx)
                    : ct::load_masked(scale + gidx, m, ct::zeros<bfxBS>());
        auto y = ct::element_cast<__nv_bfloat16>(q * ct::element_cast<float>(sraw));
        if constexpr (EVEN) ct::store(out + k, y);
        else ct::store_masked(out + k, y, m);
    }
}

}  // namespace pie::quant
