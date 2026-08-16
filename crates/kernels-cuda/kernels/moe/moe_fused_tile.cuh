//===-- moe_fused_tile.cuh - fc1 + swiglu + fc2 in one CuTile kernel ------===//
//
// An ALTERNATIVE to firing `moe_grouped_gemm_tile.cuh` twice with an
// activation between. Which of the two to prefer **flips with the grid**,
// and getting that wrong is what three earlier versions of this header did.
//
//     blocks   fused      two unfused GEMMs    ratio
//        106   0.989 ms         0.654 ms       1.51x  fused LOSES
//        212   1.360            1.315          1.03x  parity
//        424   1.796            2.594          0.69x  fused wins
//        848   2.638            5.145          0.51x
//      1,696   4.430           10.244          0.43x  fused 2.3x faster
//
// L40S sm_89, 142 SMs, 106 live experts, FM=32, three repeats each and
// stable to the third digit. `moe_fused_tile_preferred` below crosses at
// 212.
//
// # The earlier "fusion costs 1.5x" was one grid, and the grid was the cause
//
// This file used to open by saying it is a measured NEGATIVE result. That
// was taken at exactly one point -- 106 blocks -- and 106 blocks on a
// 142-SM part is **under one block per SM**. At that size the kernel is
// latency-bound and its larger per-block footprint has nothing to hide
// behind; it is not fusion that costs, it is having no parallelism.
//
// The per-block cost says it plainly:
//
//     blocks   us per block
//        106      9.34
//        212      6.53
//        424      4.22
//        848      3.15
//      1,696      2.60
//
// Still falling at twelve blocks per SM, which is what a latency-bound
// kernel looks like -- and the unfused pair, whose grid is
// `(N / kTileN, blocks)` and therefore 16x larger, was never in that regime
// at any of these points.
//
// What fusion buys is exactly what the CUTLASS island buys: `W2[e]` read
// once per block instead of once per output chunk, and the intermediate
// never round-tripping through HBM. Those are real and they show up as soon
// as there is enough work to expose them.
//
// # Against the island, which still wins at both ends
//
//     routed rows   island   fused tile   two unfused
//           2,544   0.581 ms   0.989 ms      0.654 ms
//          54,272   3.134      4.428        10.246
//
// So the island is 1.13x ahead of the best tile option at decode scale and
// 1.41x ahead at prefill scale -- and which tile option is "best" flips
// between them. Its lead over the UNFUSED pair meanwhile grows from 1.13x to
// 3.27x, which is the same story from the other side: batching is worth more
// the more rows there are.
//
// # Correct, and the error bound
//
// 0.42% worst relative error on positive data at the small end, 0.52% at the
// large -- 2^-8 is the bf16 rounding floor and the intermediate is stored
// bf16. The same harness reads 5.4% and 9.5% on SIGNED data, and that gap is
// conditioning rather than a defect: signed data cancels, the absolute error
// is unchanged, the relative error against a small result is amplified.
// Both are printed because either alone misleads.
//
// # `unsigned`, not `uint32_t`, and that is not style
//
// This file said `ct::extents<uint32_t, ...>` -- copied from NVIDIA's own
// `matmul.cuh` -- and compiled clean under `nvcc`, which force-includes
// `cuda_runtime.h` and so has `<cstdint>` transitively. **NVRTC does not.**
// Through the JIT path this crate would actually use, every one of those
// spellings was `error: identifier "uint32_t" is undefined`.
//
// The other tile kernels here already said `unsigned` and were unaffected.
// That is the whole value of compiling these through NVRTC rather than only
// through `nvcc`: an AOT build cannot see this class of defect at all.
//
// # Resource usage, and the reading trap it sets
//
//     unfused grouped GEMM      REG 174    SHARED 98,304
//     this kernel               REG 255    SHARED 98,312
//
// An earlier version of this header read that 98 KB as "one block per SM, so
// occupancy has collapsed". It is the tile compiler using the shared budget,
// not a symptom -- static shapes took this kernel from 1.778 ms to 0.984
// while leaving SHARED unchanged. The occupancy problem was real but it was
// the GRID, which is a launch parameter and not a resource figure.
//
// That header also asserted `cuda::tiles` "has no occupancy control". False:
// `[[using cutile: hint(1000, occupancy=N)]]` exists and NVIDIA's own
// `matmul.cuh` uses it. It does not happen to move this kernel.
//
// # Launch shape
//
// Tile kernels take `blockDim = (1,1,1)`; the thread count is the tile
// runtime's to choose. Passing 128 or 256 instead measures identical to
// three digits.
//
// Grid is `(num_blocks,)`, one expert block per CUDA block, each owning the
// WHOLE fc2 output panel -- which is what makes `W2[e]` a once-per-block
// read, and also what makes the grid small. Those are the same decision seen
// from two sides.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cuda_bf16.h>
#include <crt/cuda_tile.h>

namespace pie::moe {

namespace ct = ::cuda::tiles;

/// Whether to fuse, or to fire `moe_grouped_gemm_tile` twice with an
/// activation between.
///
/// Above 212 aligned blocks, which is where the grid stops being the
/// binding constraint on a 142-SM part. Measured: 1.51x behind at 106,
/// parity at 212, 0.69x at 424, 0.43x at 1,696.
///
/// The bound is a BLOCK count and not a row count on purpose -- what runs
/// out at the bottom is parallelism, and blocks are what the machine
/// schedules.
constexpr bool moe_fused_tile_preferred(int blocks)
{
    return blocks >= 212;
}

#ifndef FM
#define FM 32          // rows per block
#endif
#ifndef FI
#define FI 256         // inter_size; the intermediate is [FM, FI]
#endif
#ifndef FK
#define FK 2048        // hidden
#endif
#ifndef FIC
#define FIC 64         // column chunk for fc1, bounds the live accumulator
#endif
#ifndef FNK
#define FNK 64         // K chunk for fc1
#endif
#ifndef FN2
#define FN2 64         // N chunk for fc2's output panel
#endif

constexpr int kIC = FI / FIC;

template <int M, int N>
__tile__ auto silu(ct::tile<float, ct::shape<M, N>> x) {
    auto one = ct::full<ct::tile<float, ct::shape<M, N>>>(1.0f);
    auto z = ct::zeros<ct::tile<float, ct::shape<M, N>>>();
    return ct::div(x, ct::add(one, ct::exp(ct::sub(z, x))));
}

__tile_global__ void moe_fused_tile(
    const __nv_bfloat16* __restrict__ _a,
    const __nv_bfloat16* __restrict__ _w1,
    const __nv_bfloat16* __restrict__ _w2,
    __nv_bfloat16* __restrict__ _out,
    const int* __restrict__ ids,
    int /*k_unused*/)
{
    const auto* a = ct::assume_aligned<16>(_a);
    const auto* w1 = ct::assume_aligned<16>(_w1);
    const auto* w2 = ct::assume_aligned<16>(_w2);
    auto* out = ct::assume_aligned<16>(_out);

    const int b = static_cast<int>(ct::bid().x);
    const int e = ids[b];
    if (e < 0) return;

    const __nv_bfloat16* a_blk = a + static_cast<long long>(b) * FM * FK;
    __nv_bfloat16* o_blk = out + static_cast<long long>(b) * FM * FK;
    const __nv_bfloat16* w1_e = w1 + static_cast<long long>(e) * 2 * FI * FK;
    const __nv_bfloat16* w2_e = w2 + static_cast<long long>(e) * FK * FI;

    auto pA = ct::partition_view{
        ct::tensor_span{a_blk, ct::extents<unsigned, FM, FK>{}}, ct::shape<FM, FNK>{}};
    auto pW1 = ct::partition_view{
        ct::tensor_span<const __nv_bfloat16, ct::extents<unsigned, FK, 2 * FI>, ct::layout_left>{
            w1_e, ct::extents<unsigned, FK, 2 * FI>{}},
        ct::shape<FNK, FIC>{}};

    constexpr int kChunks = FK / FNK;
    auto chunk = [&](int c) {
        auto lin = ct::zeros<ct::tile<float, ct::shape<FM, FIC>>>();
        auto gate = ct::zeros<ct::tile<float, ct::shape<FM, FIC>>>();
        for (auto kt : ct::irange(0, kChunks)) {
            auto at = pA.load(0, kt);
            lin = ct::mma(at, pW1.load(kt, c), lin);
            gate = ct::mma(at, pW1.load(kt, kIC + c), gate);
        }
        return ct::element_cast<__nv_bfloat16>(ct::mul(silu<FM, FIC>(gate), lin));
    };
    static_assert(kIC == 4, "the chunk list is written for FI / FIC == 4");
    auto i0 = chunk(0); auto i1 = chunk(1);
    auto i2 = chunk(2); auto i3 = chunk(3);

    auto pW2 = ct::partition_view{
        ct::tensor_span<const __nv_bfloat16, ct::extents<unsigned, FI, FK>, ct::layout_left>{
            w2_e, ct::extents<unsigned, FI, FK>{}},
        ct::shape<FIC, FN2>{}};
    auto pO = ct::partition_view{
        ct::tensor_span{o_blk, ct::extents<unsigned, FM, FK>{}}, ct::shape<FM, FN2>{}};

    constexpr int nChunks = FK / FN2;
    for (auto n : ct::irange(0, nChunks)) {
        auto acc = ct::zeros<ct::tile<float, ct::shape<FM, FN2>>>();
        acc = ct::mma(i0, pW2.load(0, n), acc);
        acc = ct::mma(i1, pW2.load(1, n), acc);
        acc = ct::mma(i2, pW2.load(2, n), acc);
        acc = ct::mma(i3, pW2.load(3, n), acc);
        pO.store(ct::element_cast<__nv_bfloat16>(acc), 0, n);
    }
}

}  // namespace pie::moe
