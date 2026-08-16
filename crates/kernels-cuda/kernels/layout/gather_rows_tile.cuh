//===-- gather_rows_tile.cuh - the row gather, and a measured WASH -------===//
//
// **This kernel is not faster and is carried to say so with numbers.** It is
// the measured representative of the COPY bucket -- 35 of the tree's 455
// `__global__`s -- and the whole point of having one is that "a gather is
// bandwidth-bound so CuTile cannot help" is a plausible claim that nobody
// had checked.
//
//     slots      tile        gather_rows (uint4)   ratio   differing
//         1      1.76 us        1.88 us            1.07x   0/4,096
//        16      1.86           1.97               1.06x   0/65,536
//       128      2.04           2.24               1.10x   0/524,288
//     1,024      5.34           5.18               0.97x   0/4,194,304
//     8,192    162.98         165.10               1.01x   0/33,554,432
//
// L40S sm_89, width 4096, bf16, gathering from a 65,536-row source.
// **Bit-identical at every size**, and within 10% either way at every size.
//
// The incumbent is already `uint4`-vectorised, and a pure copy does no
// arithmetic -- there is no latency to hide and no work to overlap, so the
// two kernels are the same memcpy expressed twice. That is different from
// `mlp/swiglu_tile.cuh`, which wins 1.53x while cached because it has
// arithmetic between the load and the store, and from
// `norm/rmsnorm_tile.cuh`, which wins because the scalar form walks the row
// twice.
//
// So the COPY bucket is a CODE argument only: forty lines of tile source
// against a kernel that has to branch on `(bytes & 15) == 0` and carry two
// loops. There is no speed in it, and a survey that assumed there might be
// would have put 35 kernels in the wrong column.
//
// # It is still an ALTERNATIVE, with an honest predicate
//
// `gather_rows_tile_preferred` answers FALSE. Not because the kernel is
// wrong -- it is bit-identical and within noise -- but because an
// alternative that is not faster should not displace a kernel that every
// toolchain can already compile. The predicate exists so the answer is
// stated rather than left to whoever reads the timing table.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cuda_bf16.h>
#include <crt/cuda_tile.h>

namespace pie::layout {

namespace ct = ::cuda::tiles;

/// Whether to prefer this ALTERNATIVE over `gather_rows`. It is a WASH
/// 1.10x, bit-identical -- so no.
constexpr bool gather_rows_tile_preferred(int /*slots*/, int /*width*/)
{
    return false;
}

#ifndef BS
#define BS 2048
#endif
#ifndef W
#define W 4096
#endif
using TxBS = ct::tile<__nv_bfloat16, ct::shape<BS>>;
using i32xBS = ct::tile<int, ct::shape<BS>>;

__tile_global__ void gather_rows_tile(
    const __nv_bfloat16* __restrict__ _src,
    const int* __restrict__ row_indices,
    __nv_bfloat16* __restrict__ _dst,
    int /*width*/)
{
    const int slot = static_cast<int>(ct::bid().x);
    const int row = row_indices[slot];
    auto src = ct::assume_aligned<16>(_src + (long long)row * W);
    auto dst = ct::assume_aligned<16>(_dst + (long long)slot * W);

    constexpr int nb = (W + BS - 1) / BS;
    constexpr bool EVEN = (W % BS) == 0;
    auto zero = ct::zeros<TxBS>();
    for (auto j : ct::irange(0, nb)) {
        auto cols = ct::iota<i32xBS>() + j * BS;
        TxBS v;
        if constexpr (EVEN) {
            [[ using cutile : hint(1000, latency=1) ]]
            v = ct::load(src + cols);
            ct::store(dst + cols, v);
        } else {
            auto m = cols < W;
            [[ using cutile : hint(1000, latency=1) ]]
            v = ct::load_masked(src + cols, m, zero);
            ct::store_masked(dst + cols, v, m);
        }
    }
}

}  // namespace pie::layout
