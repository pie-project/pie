//===-- dsv4_routing.cuh - DeepSeek-V4's two routers ----------------------===//
//
// Two `__global__` templates and the one `__device__` helper they share. No
// launcher: `dsv4_routing.cu` includes this file and keeps the `<<<>>>`, so
// the kernels have a single definition that nvcc and NVRTC both read.
//
// # Why both kernels are here and not one each
//
// They share `sqrtsoftplus`, and a shared helper is what decides the grain of
// a unit — `new-horizon.md` §10.5. Split across two headers, the helper is
// either duplicated (two definitions, drifting) or promoted to a third header
// that exists only to hold it. One unit, one compile, one copy of the maths
// that both routers agree on.
//
// # sqrt(softplus(x)) is the whole routing decision
//
// vLLM's `_topk_softplus_sqrt_torch` is what a DeepSeek-V4 checkpoint's
// weights were trained against, and the hash router below reproduces it
// exactly: the expert INDICES are a pure function of the token id, but the
// WEIGHTS still come from the logits, gathered at the hashed indices and
// renormalized across the K picks. A uniform `1/K` is a different model.
//
// The `x > 20` branch is not an optimisation. `expf(x)` overflows to inf a
// little past 88 and `log1pf(inf)` is inf, so a large logit would route with
// an infinite weight and renormalize every other expert to zero; past 20 the
// two expressions agree to well within bf16 anyway.
//
// # One row of the two, and the axis that separates them
//
// `topk_sqrtsoftplus` fires one block per token, 256 threads, with a staging
// loop that steps by `blockDim.x` — which is `LaunchRule::Rms` exactly, so
// it is rowed. The 32 bytes that rule hands the launch as dynamic shared
// memory are never read: this kernel reduces on thread 0 and never calls
// `device::block_sum`.
//
// `hash_route_lookup` fires one THREAD per token, `ceil(tokens / 256)`
// blocks. No ported rule produces that: `Elementwise` sizes on
// `rows · width`, and this statement's width is `top_k`, so a row would ask
// for `top_k` times the threads the launcher asks for and lean on the
// `n >= tokens` guard to discard the rest. It is carried as device text and
// left unrowed; the launcher below stays the only thing that fires it.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "pie_device.cuh"

namespace pie_cuda_driver::kernels::moe::device {

// The scalar layer is the PRELUDE's, named here so `device::i32` keeps its
// meaning inside `kernels::moe` once this nested namespace shadows it.
using ::pie_cuda_driver::kernels::device::Elem;
using ::pie_cuda_driver::kernels::device::bf16;
using ::pie_cuda_driver::kernels::device::f16;
using ::pie_cuda_driver::kernels::device::flt_max;
using ::pie_cuda_driver::kernels::device::i32;
using ::pie_cuda_driver::kernels::device::i64;

/// The widest router `topk_sqrtsoftplus`'s static shared arrays hold.
///
/// `[[maybe_unused]]` because a NVRTC unit instantiates only the templates
/// its rows name — this file's second kernel has no row — and a constant read
/// solely by an un-instantiated template is "declared but never referenced"
/// to the front end. Saying so here beats a `--diag-suppress` that would hide
/// the same warning about a constant nothing reads at all.
[[maybe_unused]] constexpr int kDsv4MaxExperts = 512;

/// `sqrt(log(1 + exp(x)))`, saturated at zero.
///
/// `__device__` explicitly, and every helper in this family is: nvcc infers
/// the execution space of a function a `__global__` calls, NVRTC does not,
/// and the diagnostic it gives instead names the CALL rather than the missing
/// annotation.
__device__ __forceinline__ float sqrtsoftplus(float x) {
    // For large x, softplus(x) == x to far better than bf16 -- and taking the
    // branch is what keeps expf() from overflowing to inf.
    const float sp = x > 20.f ? x : log1pf(expf(x));
    return sqrtf(fmaxf(sp, 0.f));
}

/// Top-`K` routing over `sqrt(softplus(logits))`, with an optional per-expert
/// correction bias.
///
/// One block per token, thread 0 running the K-pass selection scan. The bias
/// shifts the RANKING and not the published weight — `scores` is ranked,
/// `orig_scores` is published — which is the correction DeepSeek trains.
template <class T>
__global__ void topk_sqrtsoftplus(
    const T* __restrict__ logits,
    i32* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    const float* __restrict__ correction_bias,
    int E,
    int K,
    bool renormalize,
    float routed_scaling_factor)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const T* row = logits + static_cast<long long>(n) * E;
    __shared__ float scores[kDsv4MaxExperts];
    __shared__ float orig_scores[kDsv4MaxExperts];

    for (int e = tid; e < E; e += blockDim.x) {
        const float x = Elem<T>::to_f32(row[e]);
        const float s = sqrtsoftplus(x);
        orig_scores[e] = s;
        scores[e] = correction_bias != nullptr ? s + correction_bias[e] : s;
    }
    __syncthreads();

    if (tid == 0) {
        i32* idx = topk_idx + static_cast<long long>(n) * K;
        float* w = topk_w + static_cast<long long>(n) * K;
        float sum = 0.f;
        for (int k = 0; k < K; ++k) {
            int best_i = -1;
            float best_v = -flt_max();
            for (int e = 0; e < E; ++e) {
                const float v = scores[e];
                if (v > best_v) {
                    best_v = v;
                    best_i = e;
                }
            }
            idx[k] = best_i;
            w[k] = orig_scores[best_i];
            sum += orig_scores[best_i];
            scores[best_i] = -flt_max();
        }
        const float scale = renormalize && sum > 0.f
            ? routed_scaling_factor / sum
            : routed_scaling_factor;
        for (int k = 0; k < K; ++k) w[k] *= scale;
    }
}

/// DeepSeek-V4 hash routing: the experts come from a lookup table indexed by
/// token id, the weights from the logits at those experts.
///
/// One thread per token — the table read is the whole kernel and there is
/// nothing for a block to cooperate on. Both indices are clamped rather than
/// trusted: `token_ids` arrives from a request and `tid2eid` from a
/// checkpoint, and an out-of-range read here is a device-side fault whose
/// address says nothing about which of the two was wrong.
template <class T>
__global__ void hash_route_lookup(
    const i32* __restrict__ token_ids,
    const i64* __restrict__ tid2eid,
    const T* __restrict__ logits,
    i32* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    int tokens,
    int vocab_size,
    int E,
    int K,
    bool renormalize,
    float routed_scaling_factor)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= tokens) return;
    const int tok = token_ids[n];
    const int clamped = (tok >= 0 && tok < vocab_size) ? tok : 0;

    const i64* row = tid2eid + static_cast<long long>(clamped) * K;
    const T* lg = logits + static_cast<long long>(n) * E;
    i32* out_idx = topk_idx + static_cast<long long>(n) * K;
    float* out_w = topk_w + static_cast<long long>(n) * K;

    float sum = 0.f;
    for (int k = 0; k < K; ++k) {
        const int e = static_cast<int>(row[k]);
        const int ec = (e >= 0 && e < E) ? e : 0;
        out_idx[k] = ec;
        const float w = sqrtsoftplus(Elem<T>::to_f32(lg[ec]));
        out_w[k] = w;
        sum += w;
    }
    // A floor rather than a `sum > 0` branch: every weight here is a sqrt and
    // therefore non-negative, so the only way to reach zero is a row of zero
    // logits, and dividing that by 1e-20 publishes zeros instead of NaNs.
    const float scale = renormalize
        ? routed_scaling_factor / fmaxf(sum, 1e-20f)
        : routed_scaling_factor;
    for (int k = 0; k < K; ++k) out_w[k] *= scale;
}

}  // namespace pie_cuda_driver::kernels::moe::device
