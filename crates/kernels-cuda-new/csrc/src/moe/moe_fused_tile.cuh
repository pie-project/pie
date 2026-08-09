//===-- moe_fused_tile.cuh - fc1 + swiglu + fc2 in one CuTile kernel ------===//
//
// **READ THIS FIRST: this kernel is a measured NEGATIVE result.** It is
// correct and it is slower than not fusing. It is carried because the
// experiment cost a day and the answer should not have to be bought twice.
//
//     island   permute+fc1+act+fc2+finalise   0.581 ms   573 GB/s
//     two unfused tile GEMMs                  0.654      ~510
//     THIS kernel, best of the sweep          0.984      336
//
// L40S sm_89, 318 tokens / hidden 2048 / inter 256 / 256 experts / top-k 8,
// held at the decode census of 106 live experts. Correct to 0.42% worst
// relative error on positive data, which is 2^-8 and therefore the bf16
// rounding floor -- the intermediate is stored bf16. The same harness reads
// 5.36% on SIGNED data, and that gap is conditioning rather than a defect:
// signed data cancels, the absolute error is unchanged, the relative error
// against a small result is amplified.
//
// # Why it loses -- and NOT for the reason first published
//
// The first version of this header blamed shared memory: 92-99 KB of a
// 100 KB budget, "ONE block per SM, and 106 to 318 blocks each alone on an
// SM cannot hide HBM latency." That reads the number backwards. Making the
// extents and trip counts compile-time constants took this kernel from
// 1.778 ms to 0.984 while leaving SHARED at 98,312 -- the tile compiler
// stages deeply BECAUSE it can compute the addresses, and a full shared
// budget is the budget being used rather than a symptom. The unfused GEMM
// shows the same thing from the other side: 16 KB dynamic at 0.349 ms,
// 96 KB static at 0.324.
//
// That header also asserted `cuda::tiles` "has no occupancy control". False:
// `[[using cutile: hint(1000, occupancy=N)]]` exists and NVIDIA's own
// `matmul.cuh` uses it. It does not happen to move THIS kernel -- 98,312
// bytes of shared at occupancy 1, 2 and 4 alike -- but a wrong statement
// about an API is worse than a measurement that did not move, because it
// stops the next person looking.
//
// What is left, with both corrected, is a smaller and less explained gap:
// fusing costs 1.5x against not fusing, where the unfused pair does two of
// this kernel's three stages. The fc2 leg reads each `W2[e]` once per block
// either way; what fusion buys is not round-tripping the intermediate, and
// what it costs is that the four resident `[FM, FIC]` chunks are live across
// the whole fc2 loop. No measurement here separates those two, and the
// honest statement is that the direction is established and the cause is
// not.
//
// # What would change the answer
//
// Two things untried, both from NVIDIA's `matmul.cuh`: the `GROUP_SIZE_M`
// swizzle that groups blocks for L2 reuse, and `load_masked` with a
// `view_padding::zero` policy in place of the shape divisibility this file
// assumes. Neither is a tuning knob; both are structure.
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
// # Launch shape
//
// Tile kernels take `blockDim = (1,1,1)`; the thread count is the tile
// runtime's to choose. Passing 128 or 256 instead measures identical to
// three digits.
//
// Harness: `fused_bench.c` in the session workspace. Grid is
// `(num_blocks,)`, one expert block per CUDA block, each owning the WHOLE
// fc2 output panel -- which is what makes `W2[e]` a once-per-block read
// rather than once per output chunk, i.e. the island's access pattern
// stated directly. That part works; it is the occupancy that does not.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cuda_bf16.h>
#include <crt/cuda_tile.h>

namespace pie_cuda_driver::kernels::moe::device {

namespace ct = ::cuda::tiles;

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

}  // namespace pie_cuda_driver::kernels::moe::device
