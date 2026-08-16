//===-- rmsnorm_tile.cuh - RMSNorm in CuTile, and it is faster -----------===//
//
// An ALTERNATIVE to `rmsnorm.cuh`'s scalar kernel, not a replacement for it.
// Both are carried; `rmsnorm_tile_preferred` below says which to fire, and
// the answer here happens to be "this one, always". The CuTile twin of
// `rmsnorm.cuh`'s scalar kernel: about forty lines, no
// block reduction, no `__shared__`, no warp shuffle, no launcher. Numerically
// IDENTICAL to it -- worst relative error **0.0000**, not "within tolerance"
// -- and faster:
//
//                        CuTile     rmsnorm.cuh    rmsnorm_vec8
//     H=4096, 1 row      1.94 us      2.93 us
//     H=7168, 1 row      2.42        3.84            2.38
//     H=4096, 2048 rows  9.66       12.37
//
// L40S sm_89, bf16, empty-grid floor 0.65 us. 1.51x and 1.59x over the
// scalar kernel; at H=7168 it TIES `rmsnorm_vec8`, the hand-vectorised path
// that requires 16-byte-aligned rows and a pitch check the launcher has to
// make -- a check this kernel does not need, because `assume_aligned` is a
// claim the JIT can make per row instead of a branch taken per fire.
//
// # An earlier CuTile RMSNorm was SLOWER, and the difference was the dialect
//
// The first attempt measured 3.84 us against the tree's 2.93 and was written
// off. It used `partition_view` over a 1-D row, `dynamic_extent` for the
// hidden size, a run-time trip count, and no hints. This one is NVIDIA's own
// idiom from `NVIDIA/TileGym`'s `rms_norm.cuh`:
//
//   * indices from `ct::iota<i32xBS>() + j` and a `ct::load(ptr + cols)`
//     gather, NOT a `partition_view` -- a row is not a tiled matrix;
//   * `[[using cutile: hint(1000, latency=1)]]` on every load;
//   * the hidden size a TEMPLATE parameter, so the trip count and the tail
//     predicate are compile-time;
//   * `ct::assume_aligned<16>` on every pointer;
//   * `ct::sum<0>` down to a plain `float`, after which it is ordinary C++.
//
// Half the runtime came back. `.wiki/driver/new-horizon.md` §23.20 has the
// before/after for every kernel in this spike, all of which moved.
//
// # The tail is masked, and that is not optional
//
// `HID % BS != 0` is the common case -- hidden 7168 against any power-of-two
// block -- and an unmasked version reads past the row and computes a wrong
// sum. It measured 0.1103 worst relative error at H=7168 while looking
// perfectly healthy at H=4096, which is the shape a careless bench picks.
// `EVEN_N` selects the masked path at compile time, so the cost is zero
// where the shapes divide.
//
// # Not a `Unit` yet, for the reasons `moe_grouped_gemm_tile.cuh` states
//
// Three pip wheels and a `CUDA_ROOT`: NVRTC 13.3, 13.3 runtime headers, and
// `tileiras` to assemble what NVRTC returns. That file's header carries the
// whole of it, including why a `Toolchain` floor alone is not enough for a
// tile unit.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cuda_bf16.h>
#include <crt/cuda_tile.h>

namespace pie::norm {

namespace ct = ::cuda::tiles;

/// Whether this ALTERNATIVE should be preferred to `rmsnorm.cuh`'s kernels.
///
/// Always, at every shape measured. Unlike the other tile alternatives this
/// one has no crossover: it wins by 1.51x at one row and converges to a tie
/// as the rows grow, never going behind.
///
///     rows      touched     tile       rmsnorm.cuh
///        1        0 MB      1.94 us      2.93 us
///      128        2         2.49         3.82
///    1,024       16         6.22         8.20
///    4,096       67        16.07        21.64
///   16,384      268       402.58       405.66
///   65,536    1,073      1590.55      1612.53
///
/// The parameters are taken anyway, so the shape of this predicate matches
/// the others and a caller does not have to know which ones have crossovers.
constexpr bool rmsnorm_tile_preferred(int /*rows*/, int /*hidden*/)
{
    return true;
}

#ifndef BS
#define BS 1024
#endif
#ifndef HID
#define HID 4096
#endif
using TxBS  = ct::tile<__nv_bfloat16, ct::shape<BS>>;
using f32xBS = ct::tile<float, ct::shape<BS>>;
using i32xBS = ct::tile<int, ct::shape<BS>>;

__tile_global__ void rmsnorm_tile(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ W,
    __nv_bfloat16* __restrict__ Y,
    int /*hidden*/, float eps)
{
    const int row = static_cast<int>(ct::bid().x);
    auto X_row = ct::assume_aligned<16>(X + (long long)row * HID);
    auto Y_row = ct::assume_aligned<16>(Y + (long long)row * HID);
    auto W_al  = ct::assume_aligned<16>(W);

    constexpr int num_blocks = (HID + BS - 1) / BS;
    constexpr bool EVEN_N = (HID % BS) == 0;
    auto zero_pad = ct::zeros<TxBS>();
    auto acc = ct::zeros<f32xBS>();
    for (auto j_idx : ct::irange(0, num_blocks)) {
        auto cols = ct::iota<i32xBS>() + j_idx * BS;
        TxBS xj;
        if constexpr (EVEN_N) {
            [[ using cutile : hint(1000, latency=1) ]]
            xj = ct::load(X_row + cols);
        } else {
            auto mask = cols < HID;
            [[ using cutile : hint(1000, latency=1) ]]
            xj = ct::load_masked(X_row + cols, mask, zero_pad);
        }
        auto x = ct::element_cast<float>(xj);
        acc = acc + x * x;
    }
    float sum_sq = static_cast<float>(ct::sum<0>(acc));
    constexpr float inv_N = 1.0f / (float)HID;
    float rms = ct::rsqrt(sum_sq * inv_N + eps);

    for (auto j_idx : ct::irange(0, num_blocks)) {
        auto cols = ct::iota<i32xBS>() + j_idx * BS;
        TxBS xj, wj;
        if constexpr (EVEN_N) {
            [[ using cutile : hint(1000, latency=1) ]]
            xj = ct::load(X_row + cols);
            [[ using cutile : hint(1000, latency=1) ]]
            wj = ct::load(W_al + cols);
        } else {
            auto m = cols < HID;
            [[ using cutile : hint(1000, latency=1) ]]
            xj = ct::load_masked(X_row + cols, m, zero_pad);
            [[ using cutile : hint(1000, latency=1) ]]
            wj = ct::load_masked(W_al + cols, m, zero_pad);
        }
        auto y = ct::element_cast<float>(xj) * rms * ct::element_cast<float>(wj);
        if constexpr (EVEN_N) {
            ct::store(Y_row + cols, ct::element_cast<__nv_bfloat16>(y));
        } else {
            ct::store_masked(Y_row + cols, ct::element_cast<__nv_bfloat16>(y), cols < HID);
        }
    }
}

}  // namespace pie::norm
