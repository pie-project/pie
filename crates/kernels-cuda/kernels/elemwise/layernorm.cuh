#pragma once

#include "prelude/device.cuh"
#include "elemwise/norm.cuh"

namespace pie::elemwise {

/// **THE CENTRED NORM, WITHOUT THE TWO VECTORS THAT BAKE**
/// (`.wiki/alto/multimodal.md` §6.1).
///
/// `y = (x - mean(x)) * rsqrt(var(x) + eps)` over a whole row, no scale and
/// no bias. Every qwen vision block is `nn.LayerNorm` — the checkpoints
/// publish `blocks.{l}.norm1.bias` beside `.weight`, and an RMSNorm has no
/// bias — and the two learned vectors fold into the GEMM that reads the norm:
/// `LN(x)*M^T = (c/rms(c))*diag(w)*M^T + b*M^T` for `c = x - mean(x)`. What
/// no import can fold away is the mean subtraction, and that is this kernel.
///
/// **TWO REDUCTIONS AND NOT ONE.** `var = E[x^2] - E[x]^2` would halve the
/// syncs and is the reason this kernel would be subtly wrong: a tower row
/// whose mean is large against its spread cancels catastrophically in f32,
/// and the failure shows up as a slightly wrong norm rather than as a NaN.
/// The mean is reduced, then the centred squares are reduced against it, the
/// way `torch.nn.LayerNorm` computes it.
///
/// `eps` sits INSIDE the root beside the variance, which is where LayerNorm
/// puts it and where the rms family next door puts it too.
template <class T, int BLOCK = 256>
__global__ void layernorm_no_scale(
    const T* __restrict__ x,
    T* __restrict__ y,
    int hidden,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const T* xr = x + static_cast<long long>(row) * hidden;
    T* yr = y + static_cast<long long>(row) * hidden;

    __shared__ float buf[BLOCK];

    float local = 0.f;
    for (int i = tid; i < hidden; i += BLOCK) {
        local += Elem<T>::to_f32(xr[i]);
    }
    const float mean = block_reduce_sum_exact<BLOCK>(local, buf) /
                       static_cast<float>(hidden);

    __syncthreads();

    float spread = 0.f;
    for (int i = tid; i < hidden; i += BLOCK) {
        const float c = Elem<T>::to_f32(xr[i]) - mean;
        spread += c * c;
    }
    const float inv = rsqrtf(block_reduce_sum_exact<BLOCK>(spread, buf) /
                                 static_cast<float>(hidden) +
                             eps);

    for (int i = tid; i < hidden; i += BLOCK) {
        yr[i] = Elem<T>::from_f32((Elem<T>::to_f32(xr[i]) - mean) * inv);
    }
}

}
