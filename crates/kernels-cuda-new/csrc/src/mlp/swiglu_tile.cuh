//===-- swiglu_tile.cuh - the MLP activation in CuTile ------------------===//
//
// An ALTERNATIVE to `swiglu.cuh`'s base kernel, not a replacement for it.
// Both are carried; `swiglu_tile_preferred` below says which to fire, and
// unlike the other tile alternatives this one has a real crossover. The
// CuTile twin of `swiglu.cuh`'s base kernel: fifteen lines against a
// thread-per-element `__global__`, BIT-EXACT against it (worst relative
// error 0.00000 over 4M elements), and its speed depends entirely on
// whether the working set fits L2 -- which is the finding, not a caveat.
//
//     elements     bytes    CuTile      swiglu.cuh
//      4,194,304    25 MB   0.008 ms    0.012 ms    3,273 vs 2,135 GB/s
//     33,554,432   201 MB   0.303       0.291         664 vs   693
//    134,217,728   805 MB   1.218       1.166         661 vs   691
//
// L40S sm_89, L2 is 48 MB, HBM peak ~864 GB/s. Inside L2 this is 1.53x
// faster; outside it both sit at 77-80% of peak and this is 4% behind.
//
// **An elementwise kernel at the memory roofline cannot be made faster by
// any programming model.** So the win in this class is a latency and
// occupancy win and it appears only where the data is cached -- which is
// not a niche, because a decode MLP activation is kilobytes and that is
// exactly where `norm/rmsnorm.cuh` measures a 2.20 us launch floor against
// a 2.38 us kernel.
//
// # The mask is what makes it general
//
// `n` is arbitrary, so every load and store is masked with
// `cols < n` and a zero pad. That is not a tail special case bolted on: it
// is the same `load_masked` / `store_masked` NVIDIA's own kernels use, it
// costs nothing measurable here, and without it this kernel would read past
// the buffer for any `n` not a multiple of `BS`. `norm/rmsnorm_tile.cuh`
// records what that looks like when it goes wrong -- healthy at one shape,
// 0.1103 relative error at another.
//
// # Not a `Unit` yet
//
// Three pip wheels and a `CUDA_ROOT`, per
// `moe/moe_grouped_gemm_tile.cuh`'s header. Nothing else.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cuda_bf16.h>
#include <crt/cuda_tile.h>

namespace pie_cuda_driver::kernels::mlp::device {

namespace ct = ::cuda::tiles;

/// Whether this ALTERNATIVE should be preferred to `swiglu.cuh`'s kernel.
///
/// Below the crossover, which is between 100 MB and 150 MB of TOUCHED bytes
/// -- three buffers of `n` bf16 elements, so `6 * n`. Above it both kernels
/// are at the memory roofline and this one is a few percent behind.
///
///     touched     tile        swiglu.cuh
///        6 MB     0.003 ms    0.004 ms
///       25        0.008       0.012
///       50        0.013       0.021
///      100        0.038       0.057
///      150        0.227       0.217     <- crossed
///      201        0.303       0.290
///
/// The bound is the largest MEASURED point where this kernel is ahead --
/// 16 Mi elements, 100.7 MB touched, where it is 1.5x ahead -- and not an
/// interpolation. The next point measured, 24 Mi / 151 MB, is 5% behind, and
/// anywhere between them is a guess. Stating it in elements rather than in
/// bytes is deliberate: a rounded byte figure excluded the very point that
/// was measured, which a `static_assert` against the measurements caught.
constexpr bool swiglu_tile_preferred(long long n)
{
    return n <= (16LL << 20);
}

#ifndef BS
#define BS 1024
#endif
using TxBS = ct::tile<__nv_bfloat16, ct::shape<BS>>;
using i32xBS = ct::tile<int, ct::shape<BS>>;

__tile_global__ void swiglu_tile(
    const __nv_bfloat16* __restrict__ _gate,
    const __nv_bfloat16* __restrict__ _up,
    __nv_bfloat16* __restrict__ _y,
    int n)
{
    auto gate = ct::assume_aligned<16>(_gate);
    auto up   = ct::assume_aligned<16>(_up);
    auto y    = ct::assume_aligned<16>(_y);

    auto cols = ct::iota<i32xBS>() + static_cast<int>(ct::bid().x) * BS;
    auto mask = cols < n;
    auto zero = ct::zeros<TxBS>();
    TxBS gj, uj;
    [[ using cutile : hint(1000, latency=1) ]]
    gj = ct::load_masked(gate + cols, mask, zero);
    [[ using cutile : hint(1000, latency=1) ]]
    uj = ct::load_masked(up + cols, mask, zero);
    auto g = ct::element_cast<float>(gj);
    auto u = ct::element_cast<float>(uj);
    auto one = ct::full<ct::tile<float, ct::shape<BS>>>(1.0f);
    auto z = ct::zeros<ct::tile<float, ct::shape<BS>>>();
    auto silu = g / (one + ct::exp(z - g));
    ct::store_masked(y + cols, ct::element_cast<__nv_bfloat16>(silu * u), mask);
}

}  // namespace pie_cuda_driver::kernels::mlp::device
