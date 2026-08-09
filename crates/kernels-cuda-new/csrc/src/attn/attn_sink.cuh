//===-- attn_sink.cuh - the sink rescale and the LSE rebase --------------===//
//
// Two `__global__` templates, no host code. GPT-OSS learns a per-head sink
// scalar and extends the softmax denominator with `exp(sink)`; flashinfer
// publishes its LSE in log2. Both facts are corrections applied AFTER an
// attention kernel has already written its output, which is why they are
// separate launches and not a fused epilogue -- the attention kernel is
// flashinfer's and cannot be edited.
//
// # What moved, and what did not
//
// `lse_log2_to_ln` was `<<<(n + 255) / 256, 256>>>` -- `LaunchRule::
// Elementwise` exactly, and it has a row. The `n <= 0` guard is the rule's
// `Ungeometric::Empty`.
//
// `attn_sink_rescale` was `<<<dim3(N, num_q_heads), clamp(head_dim, 32, 128)>>>`.
// That is `LaunchRule::PerHeadElementwise` and this crate's `runtime::launch`
// does not evaluate it yet: `Dims` carries `rows`, `width` and `in_width`,
// and a two-dimensional per-head grid needs a head count `Dims` has no field
// for. So the kernel is HERE -- one definition, NVRTC-clean, parsed by every
// probe -- and there is no row for it. Stating an unported rule would fail
// `runtime::launch`'s `every_stated_rule_is_ported`, and inventing a rule to
// fit is how a family acquires a geometry nothing else can read. When the
// rule lands, this file needs a row and not a line of CUDA.
//
// # The ln(2) is the bug this file exists to have fixed
//
// `state_t::get_lse()` returns `m + log2(d)` -- log base 2 of the softmax
// denominator, both in flashinfer's prefill and decode paths. The HF gpt-oss
// sink formulation is in natural log. Without the conversion the sigmoid
// argument was off by a factor of 0.693, which matched HF's top-1 on most
// prompts by accident and then drifted: greedy decoding degenerated after a
// few steps on some inputs. Both kernels below multiply by `kLn2` and the
// constant is spelled to full fp32 precision in each, because a rebased LSE
// and a rebasing rescale must agree on the same last bit or the two paths
// disagree on which token wins.
//
// # Why `lse_log2_to_ln` is a template over a type it only ever takes as f32
//
// A row names `template<Elem>` -- `DeviceKernel::instantiation()` formats
// exactly one type argument, always. The LSE buffer is fp32 by flashinfer's
// contract and there is no second format for it, so the type parameter buys
// nothing here; it is the shape the row table speaks. `f32` below is the
// alias the row names, and the body is written in plain float arithmetic
// rather than through `Elem<T>` because `Elem` has no `float`
// specialisation and adding one from this file would put a specialisation of
// a prelude template in a leaf header, where the next leaf to want it would
// collide.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "pie_device.cuh"

namespace pie_cuda_driver::kernels::attn::device {

using ::pie_cuda_driver::kernels::device::Elem;
using ::pie_cuda_driver::kernels::device::bf16;
using ::pie_cuda_driver::kernels::device::bf16_to_f32;
using ::pie_cuda_driver::kernels::device::f32_to_bf16;
using ::pie_cuda_driver::kernels::device::i32;
using ::pie_cuda_driver::kernels::device::usize;

/// The fp32 alias a row names when a kernel has no second format.
using f32 = float;

/// `log2(d) -> ln(d)`, in place, on the finite entries.
///
/// A causally-masked-out row carries `lse = -inf`; scaling an infinity is
/// still an infinity, but the guard is kept because it also covers the NaN a
/// zero-length row can produce, and because the combine downstream tests
/// finiteness rather than sign.
template <class T>
__global__ void lse_log2_to_ln(T* __restrict__ lse, usize n) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    constexpr float kLn2 = 0.69314718055994530942f;
    const float v = static_cast<float>(lse[i]);
    if (isfinite(v)) lse[i] = static_cast<T>(v * kLn2);
}

/// `o[t, h, :] *= sigmoid(ln_lse[t, h] - sink[h])`, in place.
///
/// Equivalent to a virtual KV slot with logit `sink[h]` and value zero: the
/// denominator grows by `exp(sink)` and the numerator does not, so every
/// component of the output shrinks by the same factor. Applied in place so
/// the o_proj GEMM and the residual add downstream read corrected
/// activations without a copy.
///
/// NO ROW STATES THIS KERNEL -- see the header. `N` is still an argument
/// because without a rule there is nothing to recover it from.
template <class T>
__global__ void attn_sink_rescale(
    T* __restrict__ o,
    const float* __restrict__ lse,
    const T* __restrict__ sinks,
    i32 N,
    i32 num_q_heads,
    i32 head_dim)
{
    const i32 t = static_cast<i32>(blockIdx.x);
    const i32 h = static_cast<i32>(blockIdx.y);
    if (t >= N || h >= num_q_heads) return;

    constexpr float kLn2 = 0.69314718055994530942f;
    const float lse_val = lse[t * num_q_heads + h];
    const float sink = Elem<T>::to_f32(sinks[h]);
    float r;
    if (!isfinite(lse_val)) {
        // `lse = -inf` on causal-masked-out rows; `o` is already zero there,
        // so the factor is don't-care and 1 is the cheapest don't-care.
        r = 1.0f;
    } else {
        const float diff = lse_val * kLn2 - sink;
        r = 1.0f / (1.0f + __expf(-diff));
    }

    const i32 row_stride = num_q_heads * head_dim;
    T* row = o + static_cast<long long>(t) * row_stride + h * head_dim;
    for (i32 d = static_cast<i32>(threadIdx.x); d < head_dim;
         d += static_cast<i32>(blockDim.x)) {
        row[d] = Elem<T>::from_f32(Elem<T>::to_f32(row[d]) * r);
    }
}

}  // namespace pie_cuda_driver::kernels::attn::device
