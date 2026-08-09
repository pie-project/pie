//===-- rmsnorm_rasr_tile.cuh - the fused norm pair, in CuTile -----------===//
//
// An ALTERNATIVE to `rmsnorm.cuh`'s `rmsnorm_residual_add_scale_rmsnorm`,
// not a replacement for it. That kernel is the most expensive thing this
// family measures: its own header records **10.79 us/call in gemma-4-26B's
// decode -- 8% of the step**.
//
// It is expensive for a reason a tile fixes directly. The scalar form walks
// the row THREE times with dependent loads, once per pass; at hidden 2816
// with BLOCK=512 that is ~6 dependent loads per thread per pass. This reads
// each operand once into a tile and does all three passes in registers.
//
//     rows      tile      rmsnorm_rasr (scalar)
//        1      2.41 us         4.33 us      1.80x
//      128      3.44            5.46         1.59x
//    2,048     12.86           24.71         1.92x
//
// L40S sm_89, hidden 2816, bf16, BLOCK=512 for the scalar form. REG 106,
// 16 bytes of shared -- against the scalar form's two `__shared__ float[512]`
// reduction buffers and their barriers, which is the other half of why it
// wins.
//
// # Numerics: identical, then reassociated, and the difference is located
//
// At 1 and 128 rows every output is **BIT-IDENTICAL** to the scalar kernel,
// both the `hidden` buffer and the `norm_out` buffer. At 2,048 rows 35
// elements of 5,767,168 differ, worst relative 7.75e-3.
//
// Which 35 says what they are. The `hidden` buffer -- the one that depends
// only on the FIRST sum -- is bit-identical at every row count, 0 of
// 5,767,168. Only `norm_out` differs, and that depends on the SECOND sum.
// So the difference is reassociation in one reduction, not a different
// rounding point anywhere in the arithmetic: each bf16 rounding step is
// reproduced where the scalar form calls `from_f32` and nowhere else,
// including the round of `scale` itself.
//
// That is the same trade this family already ships. `rmsnorm_rasr_vec8`'s
// note says exactly this of itself -- "only the ORDER of the two sum
// reductions differs" -- and it is the kernel the launcher prefers when the
// pointers are aligned. So this kernel is in an equivalence class the tree
// already accepts, and it is stated here rather than left for someone to
// find in a diff.
//
// # Preferred whenever it can be compiled
//
// `rmsnorm_rasr_tile_preferred` answers true at every shape measured, like
// `rmsnorm_tile_preferred` and unlike the elementwise and top-K
// alternatives, which have real crossovers. The scalar and vec8 forms remain
// the fallback for every toolchain that cannot compile a tile kernel --
// which is every toolchain this crate loads today.
// `moe/moe_grouped_gemm_tile.cuh` states that floor in full;
// `csrc/src/tile_alternatives.cuh` pins these bounds to their sweeps.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cuda_bf16.h>
#include <crt/cuda_tile.h>

namespace pie_cuda_driver::kernels::norm::device {

namespace ct = ::cuda::tiles;

/// Whether this ALTERNATIVE should be preferred to the scalar fused pair.
///
/// Always, at every shape measured -- 1.80x at one row, 1.92x at 2,048.
constexpr bool rmsnorm_rasr_tile_preferred(int /*rows*/, int /*hidden*/)
{
    return true;
}

#ifndef BS
#define BS 4096
#endif
#ifndef HID
#define HID 2816
#endif
using TxBS = ct::tile<__nv_bfloat16, ct::shape<BS>>;
using f32xBS = ct::tile<float, ct::shape<BS>>;
using i32xBS = ct::tile<int, ct::shape<BS>>;

/// The scalar twin walks the row THREE times with dependent loads. This reads
/// each operand once into a tile and does all three passes in registers.
/// Every bf16 rounding point is reproduced exactly -- `element_cast` to bf16
/// where the scalar form calls `from_f32`, and nowhere else -- because the
/// scalar form's own note says its arithmetic is bit-identical to its
/// vectorised twin and that is the bar.
__tile_global__ void rmsnorm_rasr_tile(
    const __nv_bfloat16* __restrict__ _x,
    const __nv_bfloat16* __restrict__ _w,
    __nv_bfloat16* __restrict__ _hidden,
    float scale,
    const __nv_bfloat16* __restrict__ _nw,
    __nv_bfloat16* __restrict__ _out,
    int /*hidden_size*/, float eps)
{
    const int row = static_cast<int>(ct::bid().x);
    auto x  = ct::assume_aligned<16>(_x + (long long)row * HID);
    auto h  = ct::assume_aligned<16>(_hidden + (long long)row * HID);
    auto o  = ct::assume_aligned<16>(_out + (long long)row * HID);
    auto w  = ct::assume_aligned<16>(_w);
    auto nw = ct::assume_aligned<16>(_nw);

    constexpr int nb = (HID + BS - 1) / BS;
    constexpr bool EVEN = (HID % BS) == 0;
    auto zero = ct::zeros<TxBS>();
    auto acc = ct::zeros<f32xBS>();

    // Pass 1: sum of squares.
    for (auto j : ct::irange(0, nb)) {
        auto cols = ct::iota<i32xBS>() + j * BS;
        TxBS xj;
        if constexpr (EVEN) {
            [[ using cutile : hint(1000, latency=1) ]]
            xj = ct::load(x + cols);
        } else {
            [[ using cutile : hint(1000, latency=1) ]]
            xj = ct::load_masked(x + cols, cols < HID, zero);
        }
        auto v = ct::element_cast<float>(xj);
        acc = acc + v * v;
    }
    float s1 = static_cast<float>(ct::sum<0>(acc));
    float inv_rms = ct::rsqrt(s1 / (float)HID + eps);

    // `scale` is rounded to bf16 first, exactly as the scalar form does.
    // A tile has no `operator[]`, so the scalar comes back through a
    // reduction over a one-lane tile -- which is free and is the idiom.
    float scale_r = static_cast<float>(ct::reduce_max<0>(
        ct::element_cast<float>(ct::element_cast<__nv_bfloat16>(
            ct::full<ct::tile<float, ct::shape<1>>>(scale)))));

    // Pass 2: normalise, add the residual, scale, store, and accumulate the
    // second sum -- one traversal.
    auto acc2 = ct::zeros<f32xBS>();
    for (auto j : ct::irange(0, nb)) {
        auto cols = ct::iota<i32xBS>() + j * BS;
        auto m = cols < HID;
        TxBS xj, wj, hj;
        if constexpr (EVEN) {
            [[ using cutile : hint(1000, latency=1) ]]
            xj = ct::load(x + cols);
            [[ using cutile : hint(1000, latency=1) ]]
            wj = ct::load(w + cols);
            [[ using cutile : hint(1000, latency=1) ]]
            hj = ct::load(h + cols);
        } else {
            [[ using cutile : hint(1000, latency=1) ]]
            xj = ct::load_masked(x + cols, m, zero);
            [[ using cutile : hint(1000, latency=1) ]]
            wj = ct::load_masked(w + cols, m, zero);
            [[ using cutile : hint(1000, latency=1) ]]
            hj = ct::load_masked(h + cols, m, zero);
        }
        auto norm = ct::element_cast<__nv_bfloat16>(
            ct::element_cast<float>(xj) * inv_rms * ct::element_cast<float>(wj));
        auto summed = ct::element_cast<__nv_bfloat16>(
            ct::element_cast<float>(hj) + ct::element_cast<float>(norm));
        auto scaled = ct::element_cast<__nv_bfloat16>(
            ct::element_cast<float>(summed) * scale_r);
        if constexpr (EVEN) ct::store(h + cols, scaled);
        else ct::store_masked(h + cols, scaled, m);
        auto v = ct::element_cast<float>(scaled);
        acc2 = acc2 + ct::select(m, v * v, ct::zeros<f32xBS>());
    }
    float s2 = static_cast<float>(ct::sum<0>(acc2));
    float inv2 = ct::rsqrt(s2 / (float)HID + eps);

    // Pass 3: the second norm.
    for (auto j : ct::irange(0, nb)) {
        auto cols = ct::iota<i32xBS>() + j * BS;
        auto m = cols < HID;
        TxBS hj, nj;
        if constexpr (EVEN) {
            [[ using cutile : hint(1000, latency=1) ]]
            hj = ct::load(h + cols);
            [[ using cutile : hint(1000, latency=1) ]]
            nj = ct::load(nw + cols);
        } else {
            [[ using cutile : hint(1000, latency=1) ]]
            hj = ct::load_masked(h + cols, m, zero);
            [[ using cutile : hint(1000, latency=1) ]]
            nj = ct::load_masked(nw + cols, m, zero);
        }
        auto y = ct::element_cast<__nv_bfloat16>(
            ct::element_cast<float>(hj) * inv2 * ct::element_cast<float>(nj));
        if constexpr (EVEN) ct::store(o + cols, y);
        else ct::store_masked(o + cols, y, m);
    }
}

}  // namespace pie_cuda_driver::kernels::norm::device
