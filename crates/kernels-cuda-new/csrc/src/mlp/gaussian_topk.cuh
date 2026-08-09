//===-- gaussian_topk.cuh - AltUp's activation sparsity, as a template --===//
//
// One `__global__` template and one include. The launcher that used to wrap
// it is `LaunchRule::Rms` exactly -- `<<<N, 256, (256 / 32) * sizeof(float)>>>`
// -- so it says nothing a row does not, and the row is in
// `kernels_cuda_new::families::mlp`.
//
// # Why this file exists
//
// `gaussian_topk.cu` could not be handed to NVRTC: it includes
// `<cooperative_groups.h>`, and NVRTC on this machine answers 0 of 31
// standard headers with no include path. The device half moved here, the
// launcher stayed there, and the two now share ONE definition -- which is
// the property `kernels-cuda`'s `tests/sources.rs::no_global_is_defined_twice`
// exists to hold, after `norm/altup_aux` shipped two copies of a kernel for a
// release.
//
// # Cooperative groups, and why dropping them changed no bits
//
// The reduction was `cg::tiled_partition<32>(cg::this_thread_block())` and
// `tile.shfl_down`, twice: once for the row mean and once for the variance.
// The prelude's `block_sum` is that fold, line for line -- the same 16-to-1
// warp descent, the same `smem[warp]` handoff, the same second-level fold in
// warp 0, the same trailing `__syncthreads()` and the same broadcast through
// `smem[0]`. `pie_device.cuh` says so in as many words: `block_sum` is "what
// `cg::tiled_partition<32>(...).shfl_down(...)` lowered to before".
//
// That equality is the whole argument for the substitution. The fold ORDER is
// contract, not detail -- a different order sums the same values to a
// different last bit, and `driver-pipeline`'s tolerance contract holds argmax
// indices to zero -- so a reduction that merely computes the same sum would
// not have been good enough.
//
// # What the row recovers
//
// `N`. The grid was one block per row and the kernel reads `blockIdx.x`
// without a guard, so the token count was pure geometry and the rule states
// it. `dim` stayed: it is the loop bound AND the row stride, which is layout.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "pie_device.cuh"

namespace pie_cuda_driver::kernels::mlp::device {

// The scalar layer is the PRELUDE's. Named here so `gaussian_topk.cu`'s
// launcher -- which sits in `kernels::mlp` and says `device::bf16` on its
// cast -- resolves that spelling through this namespace to the same type it
// always meant.
using ::pie_cuda_driver::kernels::device::bf16;
using ::pie_cuda_driver::kernels::device::block_sum;
using ::pie_cuda_driver::kernels::device::Elem;
using ::pie_cuda_driver::kernels::device::f16;
using ::pie_cuda_driver::kernels::device::i32;

/// Zero everything below `mean + std_multiplier * stddev`, per row, in place.
///
/// One block per row. The threads reduce the mean and the variance in fp32 --
/// two passes, not Welford, because the row is read three times either way
/// and two exact passes are cheaper to reason about than one approximate one
/// -- then sweep the row again to apply the cutoff.
///
/// The subtraction in the last pass is not a clamp: what survives is
/// `x - cutoff`, shifted, which is what makes this activation SPARSITY rather
/// than a mask. Gemma-3n's AltUp states it that way.
template <class T>
__global__ void gaussian_topk(
    T* __restrict__ x,
    i32 dim,
    float std_multiplier)
{
    const i32 row = blockIdx.x;
    const i32 tid = threadIdx.x;

    T* row_ptr = x + static_cast<long long>(row) * dim;
    extern __shared__ float smem[];

    float local_sum = 0.f;
    for (i32 j = tid; j < dim; j += blockDim.x) {
        local_sum += Elem<T>::to_f32(row_ptr[j]);
    }
    const float mean = block_sum(local_sum, smem) / static_cast<float>(dim);

    float local_var = 0.f;
    for (i32 j = tid; j < dim; j += blockDim.x) {
        const float v = Elem<T>::to_f32(row_ptr[j]) - mean;
        local_var += v * v;
    }
    const float var = block_sum(local_var, smem) / static_cast<float>(dim);
    const float stddev = sqrtf(var);
    const float cutoff = mean + stddev * std_multiplier;

    for (i32 j = tid; j < dim; j += blockDim.x) {
        const float v = Elem<T>::to_f32(row_ptr[j]) - cutoff;
        row_ptr[j] = Elem<T>::from_f32(v > 0.f ? v : 0.f);
    }
}

}  // namespace pie_cuda_driver::kernels::mlp::device
