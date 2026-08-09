//===-- kimi_mla.cuh - kimi_k3's latent-attention preparation kernels ----===//
//
// Two `__global__` templates and nothing else. The `<<<>>>`s that used to sit
// beside them stay in `kimi_mla.cu`, which now `#include`s this header rather
// than defining what it launches -- so the ahead-of-time build and the JIT
// compile ONE text. A second copy is the failure this split exists to
// prevent: `norm/altup_aux` shipped two definitions of six kernels for a
// release, each correct for whichever half of the tests exercised it, and no
// test could see the disagreement because no test ran both.
//
// # What the launchers were doing, and what survives of them
//
// `kimi_split_q_b_bf16` was `total = tokens * heads * (nope + rope)`, an
// `if (total <= 0) return;`, and `<<<(total + 255) / 256, 256>>>` --
// `LaunchRule::Elementwise` verbatim, guard included: the rule answers
// `Ungeometric::Empty` for a zero extent and the binder refuses on it.
//
// `kimi_split_kv_a_norm_bf16` was `<<<tokens, 256>>>`: one block per token
// row, the row width read by a stride loop, a block-wide sum reduced in
// shared memory. That is `LaunchRule::Rms` -- a row-wise reduction, one block
// per row, 256 wide -- and it is `Rms` even though the algebra is a SPLIT
// with an RMSNorm inside it, because the rule names a geometry and not an
// operation. The 32 bytes of dynamic shared memory `Rms` requests go unused
// here; the reduction buffer is static and sized by `BLOCK_DIM`.
//
// # Why `BLOCK_DIM` is a defaulted template parameter
//
// `__shared__ float buf[BLOCK_DIM]` has to be sized at compile time, and the
// reduction's `off = BLOCK_DIM / 2` halving is only correct for the width the
// launch actually uses. The default is 256 because that is what `Rms`
// launches; a row states only the element type, so the default is what the
// JIT instantiates, and the ahead-of-time launcher passes the same 256 it
// always passed. Writing the width as a `constexpr` inside the body would
// have hidden the coupling instead of stating it.
//
// # Why they are templates when the originals were not
//
// The originals were `_bf16` and only `_bf16`. Under a JIT the element type
// is the row's, so a second numeric format costs a line in a table rather
// than a translation unit of `cicc` -- the same trade `norm/elementwise`
// measured. No such row is stated yet: kimi_k3 is bf16 everywhere, and a row
// nothing fires is a claim nothing checks.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "pie_device.cuh"

namespace pie_cuda_driver::kernels::attn::device {

using ::pie_cuda_driver::kernels::device::Elem;
using ::pie_cuda_driver::kernels::device::bf16;
using ::pie_cuda_driver::kernels::device::f16;

/// Split a fused `q_b` projection into its nope and rope halves.
///
/// `q_b` is `[tokens, heads, nope + rope]` and the two results are
/// `[tokens, heads, nope]` and `[tokens, heads, rope]`. One thread per source
/// element, which is why `total` is an argument: `Elementwise` covers the
/// grid but the kernel still has to know where the array ends, exactly as
/// `norm::device::tanh_inplace`'s `numel` does.
///
/// The `long long` casts on the destination indices are not decoration --
/// `tokens * heads * nope` overflows `int` on a long prefill at kimi_k3's
/// head count, and the product is formed before it is used as an index.
template <class T>
__global__ void split_q_b(
    const T* __restrict__ q_b,
    T* __restrict__ q_nope,
    T* __restrict__ q_pe,
    int total,
    int heads,
    int nope,
    int rope)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int per = nope + rope;
    const int d = i % per;
    const int h = (i / per) % heads;
    const int n = i / (heads * per);
    const T v = q_b[i];
    if (d < nope) {
        q_nope[(static_cast<long long>(n) * heads + h) * nope + d] = v;
    } else {
        q_pe[(static_cast<long long>(n) * heads + h) * rope + (d - nope)] = v;
    }
}

/// Split `kv_a` into a normalised latent and its rope-carrying companion.
///
/// One kernel rather than a split followed by an RMSNorm, because the latent
/// half is read twice by the norm and would otherwise be written to global
/// memory in between. `src_row_stride` is the SOURCE row width, which is
/// `kv_lora + rope` unless a caller hands a wider buffer -- the fused MLA
/// prepare does, which is why the stride is an operand and not a sum.
///
/// The `k_pe` copy is unnormalised on purpose: rope is applied to it later
/// and normalising a value that is about to be rotated changes the angle.
template <class T, int BLOCK_DIM = 256>
__global__ void split_kv_a_norm(
    const T* __restrict__ kv_a,
    const T* __restrict__ norm_weight,
    T* __restrict__ kv_c,
    T* __restrict__ k_pe,
    int kv_lora,
    int rope,
    int src_row_stride,
    float eps)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const T* row = kv_a + static_cast<long long>(n) * src_row_stride;

    for (int d = tid; d < rope; d += BLOCK_DIM) {
        k_pe[static_cast<long long>(n) * rope + d] = row[kv_lora + d];
    }

    float local = 0.f;
    for (int d = tid; d < kv_lora; d += BLOCK_DIM) {
        const float v = Elem<T>::to_f32(row[d]);
        local += v * v;
    }
    __shared__ float buf[BLOCK_DIM];
    buf[tid] = local;
    __syncthreads();
    for (int off = BLOCK_DIM / 2; off > 0; off >>= 1) {
        if (tid < off) buf[tid] += buf[tid + off];
        __syncthreads();
    }
    const float inv_rms = rsqrtf(buf[0] / static_cast<float>(kv_lora) + eps);
    for (int d = tid; d < kv_lora; d += BLOCK_DIM) {
        const float v = Elem<T>::to_f32(row[d]);
        const float w = Elem<T>::to_f32(norm_weight[d]);
        kv_c[static_cast<long long>(n) * kv_lora + d] = Elem<T>::from_f32(v * inv_rms * w);
    }
}

}  // namespace pie_cuda_driver::kernels::attn::device
