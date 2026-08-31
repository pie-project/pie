#pragma once

#include "prelude/device.cuh"

#ifdef __CUDACC_RTC__

#include "prelude/mma.cuh"
#else

#include <cuda_bf16.h>
#include <mma.h>
#endif

namespace pie::linear {

namespace wmma = ::nvcuda::wmma;

[[maybe_unused]] constexpr int kDispatchBlock = 256;

[[maybe_unused]] constexpr int kMoeVecWidth = 8;

constexpr int kMaxTopK = 16;

constexpr int kGemvWarps = 4;

using f32 = float;

template <class T>
struct Logit {
    static __device__ __forceinline__ float to_f32(T v) { return Elem<T>::to_f32(v); }
};

template <>
struct Logit<f32> {
    static __device__ __forceinline__ float to_f32(f32 v) { return v; }
};

constexpr int kSoftmaxBlock = 64;

constexpr int kRouterMaxExperts = 512;

__device__ inline void block_argmax(
    const float* __restrict__ scores,
    int num_experts,
    float floor_value,
    float* __restrict__ value_buf,
    int* __restrict__ index_buf,
    float& best_value,
    int& best_index)
{
    const int tid = threadIdx.x;

    float local_v = floor_value;
    int local_i = -1;
    for (int j = tid; j < num_experts; j += kSoftmaxBlock) {
        const float v = scores[j];
        if (v > local_v) {
            local_v = v;
            local_i = j;
        }
    }
    value_buf[tid] = local_v;
    index_buf[tid] = local_i;
    __syncthreads();

    static_assert(kSoftmaxBlock == 64, "block_argmax folds exactly one upper warp");
    if (tid < 32) {
        float v = value_buf[tid];
        int i = index_buf[tid];

        auto take = [](float& v, int& i, float ov, int oi) {
            if (ov > v || (ov == v && oi >= 0 && (i < 0 || oi < i))) {
                v = ov;
                i = oi;
            }
        };
        take(v, i, value_buf[tid + 32], index_buf[tid + 32]);
        for (int off = 16; off > 0; off >>= 1) {
            take(v, i,
                 __shfl_down_sync(0xffffffffu, v, off),
                 __shfl_down_sync(0xffffffffu, i, off));
        }
        if (tid == 0) {
            value_buf[0] = v;
            index_buf[0] = i;
        }
    }
    __syncthreads();
    best_value = value_buf[0];
    best_index = index_buf[0];

}

// `per_expert_scale` is gemma 4's learned per-expert gain, applied to the
// selected weights after the softmax and gathered by the ids this body just
// chose. `nullptr` on every other family, which is the whole of the branch.
template <class T, bool FusedGemv>
__device__ __forceinline__ void moe_topk_softmax_body(
    const T* __restrict__ logits,
    const T* __restrict__ act,
    const T* __restrict__ bias,
    i32* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    int num_experts, int K, int hidden,
    const T* __restrict__ per_expert_scale = nullptr)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const T* row =
        FusedGemv ? logits
                  : logits + static_cast<long long>(n) * num_experts;

    __shared__ float probs[kRouterMaxExperts];
    __shared__ float buf[kSoftmaxBlock];
    __shared__ int ibuf[kSoftmaxBlock];

    float local_max = -flt_max();
    if constexpr (FusedGemv) {
        const T* x = act + static_cast<long long>(n) * hidden;
        const int warp = tid >> 5;
        const int lane = tid & 31;
        constexpr int kWarps = kSoftmaxBlock / 32;
        for (int e = warp; e < num_experts; e += kWarps) {
            const T* w = row + static_cast<long long>(e) * hidden;
            float acc = 0.f;
            for (int i = lane; i < hidden; i += 32) {
                acc += Logit<T>::to_f32(w[i]) * Logit<T>::to_f32(x[i]);
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                acc += __shfl_down_sync(0xffffffffu, acc, off);
            }
            if (lane == 0) {
                if (bias != nullptr) acc += Logit<T>::to_f32(bias[e]);
                probs[e] = acc;
            }
        }
        __syncthreads();
        for (int j = tid; j < num_experts; j += kSoftmaxBlock) {
            if (probs[j] > local_max) local_max = probs[j];
        }
    } else {
    for (int j = tid; j < num_experts; j += kSoftmaxBlock) {
        const float v = Logit<T>::to_f32(row[j]);
        probs[j] = v;
        if (v > local_max) local_max = v;
    }
    }
    buf[tid] = local_max;
    __syncthreads();
    for (int off = kSoftmaxBlock / 2; off > 0; off >>= 1) {
        if (tid < off) buf[tid] = fmaxf(buf[tid], buf[tid + off]);
        __syncthreads();
    }
    const float row_max = buf[0];
    __syncthreads();

    float local_sum = 0.f;
    for (int j = tid; j < num_experts; j += kSoftmaxBlock) {
        const float e = expf(probs[j] - row_max);
        probs[j] = e;
        local_sum += e;
    }
    buf[tid] = local_sum;
    __syncthreads();
    for (int off = kSoftmaxBlock / 2; off > 0; off >>= 1) {
        if (tid < off) buf[tid] += buf[tid + off];
        __syncthreads();
    }
    const float inv_Z = 1.f / buf[0];
    __syncthreads();

    for (int j = tid; j < num_experts; j += kSoftmaxBlock) probs[j] *= inv_Z;
    __syncthreads();

    i32* out_idx = topk_idx + static_cast<long long>(n) * K;
    float* out_w = topk_w + static_cast<long long>(n) * K;
    float w_sum = 0.f;
    for (int k = 0; k < K; ++k) {
        float best_v = -1.f;
        int best_i = -1;
        block_argmax(probs, num_experts, -1.f, buf, ibuf, best_v, best_i);

        const float w = best_i >= 0 ? best_v : 0.f;
        if (tid == 0) {
            out_idx[k] = best_i;
            out_w[k] = w;
            if (best_i >= 0) probs[best_i] = -1.f;
        }
        w_sum += w;
        __syncthreads();
    }
    if (tid == 0) {
        const float inv_w = 1.f / w_sum;
        for (int k = 0; k < K; ++k) {
            float w = out_w[k] * inv_w;
            if (per_expert_scale != nullptr && out_idx[k] >= 0) {
                w *= Logit<T>::to_f32(per_expert_scale[out_idx[k]]);
            }
            out_w[k] = w;
        }
    }
}

template <class T>
__global__ void moe_topk_softmax(
    const T* __restrict__ logits,
    const T* __restrict__ act,
    const T* __restrict__ bias,
    i32* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    int num_experts, int K, int hidden)
{
    moe_topk_softmax_body<T, false>(logits, act, bias, topk_idx, topk_w,
                                num_experts, K, hidden);
}

// **ADDITIVE, FOR GEMMA 4's MIXTURE.** The same softmax-then-renormalize the
// point above takes -- which is a softmax over the SELECTED k -- and then the
// learned gain of the expert each slot chose.
template <class T>
__global__ void moe_topk_softmax_scaled(
    const T* __restrict__ logits,
    const T* __restrict__ act,
    const T* __restrict__ per_expert_scale,
    i32* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    int num_experts, int K, int hidden)
{
    moe_topk_softmax_body<T, false>(logits, act, nullptr, topk_idx, topk_w,
                                num_experts, K, hidden, per_expert_scale);
}

template <class T>
__global__ void moe_topk_softmax_fused_gemv(
    const T* __restrict__ router_weight,
    const T* __restrict__ act,
    const T* __restrict__ bias,
    i32* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    int num_experts, int K, int hidden)
{
    moe_topk_softmax_body<T, true>(router_weight, act, bias, topk_idx, topk_w,
                               num_experts, K, hidden);
}

template <class T, int PerLane>
__device__ __forceinline__ void moe_topk_softmax_warp_body(
    const T* __restrict__ logits,
    i32* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    int num_experts, int K)
{
    const int n = blockIdx.x;
    const int lane = static_cast<int>(threadIdx.x);
    const T* row = logits + static_cast<long long>(n) * num_experts;

    int idx[PerLane];
    float val[PerLane];
#pragma unroll
    for (int i = 0; i < PerLane; ++i) {
        idx[i] = lane + i * 32;
        val[i] = idx[i] < num_experts ? Logit<T>::to_f32(row[idx[i]])
                                      : -flt_max();
    }

    i32* out_idx = topk_idx + static_cast<long long>(n) * K;
    float* out_w = topk_w + static_cast<long long>(n) * K;
    float best_w[8];
    int best_e[8];
    for (int k = 0; k < K; ++k) {

        float bv = -flt_max();
        int bi = -1;
#pragma unroll
        for (int i = 0; i < PerLane; ++i) {
            if (val[i] > bv || (val[i] == bv && idx[i] < bi)) {
                bv = val[i];
                bi = idx[i];
            }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (ov > bv || (ov == bv && oi >= 0 && (bi < 0 || oi < bi))) {
                bv = ov;
                bi = oi;
            }
        }
#pragma unroll
        for (int i = 0; i < PerLane; ++i) {
            if (idx[i] == bi) val[i] = -flt_max();
        }
        best_w[k] = bv;
        best_e[k] = bi;
    }
    if (lane == 0) {

        const float row_max = best_w[0];
        float w_sum = 0.f;
        for (int k = 0; k < K; ++k) {
            best_w[k] = expf(best_w[k] - row_max);
            w_sum += best_w[k];
        }
        const float inv_w = 1.f / w_sum;
        for (int k = 0; k < K; ++k) {
            out_idx[k] = best_e[k];
            out_w[k] = best_w[k] * inv_w;
        }
    }
}

template <class T>
__global__ void moe_topk_softmax_warp_x1(
    const T* __restrict__ logits, i32* __restrict__ topk_idx,
    float* __restrict__ topk_w, int num_experts, int K)
{
    moe_topk_softmax_warp_body<T, 1>(logits, topk_idx, topk_w, num_experts, K);
}

template <class T>
__global__ void moe_topk_softmax_warp_x2(
    const T* __restrict__ logits, i32* __restrict__ topk_idx,
    float* __restrict__ topk_w, int num_experts, int K)
{
    moe_topk_softmax_warp_body<T, 2>(logits, topk_idx, topk_w, num_experts, K);
}

template <class T>
__global__ void moe_topk_softmax_warp_x4(
    const T* __restrict__ logits, i32* __restrict__ topk_idx,
    float* __restrict__ topk_w, int num_experts, int K)
{
    moe_topk_softmax_warp_body<T, 4>(logits, topk_idx, topk_w, num_experts, K);
}

template <class T>
__global__ void moe_topk_softmax_warp_x8(
    const T* __restrict__ logits, i32* __restrict__ topk_idx,
    float* __restrict__ topk_w, int num_experts, int K)
{
    moe_topk_softmax_warp_body<T, 8>(logits, topk_idx, topk_w, num_experts, K);
}

template <class T>
__global__ void moe_topk_softmax_warp_x16(
    const T* __restrict__ logits, i32* __restrict__ topk_idx,
    float* __restrict__ topk_w, int num_experts, int K)
{
    moe_topk_softmax_warp_body<T, 16>(logits, topk_idx, topk_w, num_experts, K);
}

template <class T>
__global__ void apply_per_expert_scale(
    const i32* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    const T* __restrict__ per_expert_scale,
    int total)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    const int e = topk_idx[t];

    if (e < 0) return;
    const float s = Logit<T>::to_f32(per_expert_scale[e]);
    topk_w[t] *= s;
}

template <class T>
__global__ void moe_topk_sigmoid_bias(
    const T* __restrict__ logits,
    const float* __restrict__ correction_bias,
    i32* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    int num_experts,
    int K,
    int normalize,
    float routed_scaling_factor)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const T* row = logits + static_cast<long long>(n) * num_experts;

    __shared__ float probs[kRouterMaxExperts];
    __shared__ float choice[kRouterMaxExperts];
    __shared__ float buf[kSoftmaxBlock];
    __shared__ int ibuf[kSoftmaxBlock];

    for (int j = tid; j < num_experts; j += kSoftmaxBlock) {
        const float z = Logit<T>::to_f32(row[j]);
        const float p = 1.f / (1.f + __expf(-z));
        probs[j] = p;
        choice[j] = p + correction_bias[j];
    }
    __syncthreads();

    i32* out_idx = topk_idx + static_cast<long long>(n) * K;
    float* out_w = topk_w + static_cast<long long>(n) * K;
    float sum = 0.f;
    for (int k = 0; k < K; ++k) {
        float best_v = -flt_max();
        int best_i = -1;
        block_argmax(choice, num_experts, -flt_max(), buf, ibuf, best_v, best_i);
        const float weight = best_i >= 0 ? probs[best_i] : 0.f;
        if (tid == 0) {
            out_idx[k] = best_i;
            out_w[k] = weight;
            if (best_i >= 0) choice[best_i] = -flt_max();
        }
        sum += weight;
        __syncthreads();
    }
    if (tid == 0) {
        const float scale =
            normalize ? (routed_scaling_factor / (sum + 1e-20f))
                      : routed_scaling_factor;
        for (int k = 0; k < K; ++k) {
            out_w[k] *= scale;
        }
    }
}

template <class T>
__global__ void moe_topk_sigmoid(
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
    __shared__ float scores[kRouterMaxExperts];
    __shared__ float orig_scores[kRouterMaxExperts];

    for (int e = tid; e < E; e += blockDim.x) {
        const float x = Elem<T>::to_f32(row[e]);
        const float s = 1.f / (1.f + expf(-x));
        orig_scores[e] = s;
        scores[e] = correction_bias != nullptr ? s + correction_bias[e] : s;
    }
    __syncthreads();

    __shared__ bool taken[kRouterMaxExperts];
    for (int e = tid; e < E; e += blockDim.x) taken[e] = false;
    __syncthreads();

    if (tid == 0) {
        i32* idx = topk_idx + static_cast<long long>(n) * K;
        float* w = topk_w + static_cast<long long>(n) * K;
        float sum = 0.f;
        const int picks = K < E ? K : E;
        for (int k = 0; k < picks; ++k) {
            int best_i = -1;
            float best_v = -flt_max();
            for (int e = 0; e < E; ++e) {
                if (taken[e]) continue;
                const float v = scores[e];

                if (best_i < 0 || v > best_v) {
                    best_v = v;
                    best_i = e;
                }
            }
            idx[k] = best_i;
            w[k] = orig_scores[best_i];
            sum += orig_scores[best_i];
            taken[best_i] = true;
        }

        for (int k = picks; k < K; ++k) {
            idx[k] = 0;
            w[k] = 0.f;
        }
        const float scale = renormalize && sum > 0.f
            ? routed_scaling_factor / sum
            : routed_scaling_factor;
        for (int k = 0; k < K; ++k) w[k] *= scale;
    }
}

__device__ __forceinline__ float sqrt_softplus(float x) {

    const float sp = x > 20.f ? x : log1pf(expf(x));
    return sqrtf(fmaxf(sp, 0.f));
}

template <class T>
__global__ void moe_topk_sqrt_softplus(
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
    __shared__ float scores[kRouterMaxExperts];
    __shared__ float orig_scores[kRouterMaxExperts];

    for (int e = tid; e < E; e += blockDim.x) {
        const float x = Elem<T>::to_f32(row[e]);
        const float s = sqrt_softplus(x);
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
        const float w = sqrt_softplus(Elem<T>::to_f32(lg[ec]));
        out_w[k] = w;
        sum += w;
    }

    const float scale = renormalize
        ? routed_scaling_factor / fmaxf(sum, 1e-20f)
        : routed_scaling_factor;
    for (int k = 0; k < K; ++k) out_w[k] *= scale;
}

template <class T>
__global__ void build_dual_gemm_ptrs(
    const T* act,
    const T* w0,
    const T* w1,
    T* out0,
    T* out1,
    const T** act_ptrs,
    const T** w_ptrs,
    T** out_ptrs)
{
    act_ptrs[0] = act;
    act_ptrs[1] = act;
    w_ptrs[0] = w0;
    w_ptrs[1] = w1;
    out_ptrs[0] = out0;
    out_ptrs[1] = out1;
}

template <class T>
__global__ void build_moe_ptrs_decode(
    const i32* topk_idx,
    const float* topk_w,
    const T* gate_up_base,
    const T* down_base,
    const T* norm_x,
    T* expert_gate_up,
    T* expert_act,
    T* expert_out,
    const T** a_gu_ptrs,
    const T** b_gu_ptrs,
    T**       c_gu_ptrs,
    const T** a_dn_ptrs,
    const T** b_dn_ptrs,
    T**       c_dn_ptrs,
    float*    weights_out,
    int top_k, int H, int I_moe)
{
    const int k = threadIdx.x;
    if (k >= top_k) return;
    const long long stride_gu = 2LL * I_moe * H;
    const long long stride_dn = (long long)H * I_moe;

    int e = topk_idx[k];
    if (e < 0) e = 0;

    a_gu_ptrs[k] = gate_up_base + e * stride_gu;
    b_gu_ptrs[k] = norm_x;
    c_gu_ptrs[k] = expert_gate_up + (long long)k * 2 * I_moe;

    a_dn_ptrs[k] = down_base + e * stride_dn;
    b_dn_ptrs[k] = expert_act + (long long)k * I_moe;
    c_dn_ptrs[k] = expert_out + (long long)k * H;

    weights_out[k] = topk_w[k];
}

template <class T>
__global__ void build_moe_ptrs_decode_batched(
    const i32* topk_idx,
    const float* topk_w,
    const T* gate_up_base,
    const T* down_base,
    const T* norm_x,
    T* expert_gate_up,
    T* expert_act,
    T* expert_out,
    const T** a_gu_ptrs,
    const T** b_gu_ptrs,
    T**       c_gu_ptrs,
    const T** a_dn_ptrs,
    const T** b_dn_ptrs,
    T**       c_dn_ptrs,
    float*    weights_out,
    int num_tokens, int top_k, int H, int I_moe)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_tokens * top_k;
    if (r >= total) return;
    const long long stride_gu = 2LL * I_moe * H;
    const long long stride_dn = static_cast<long long>(H) * I_moe;
    const int token = r / top_k;

    int e = topk_idx[r];
    if (e < 0) e = 0;

    a_gu_ptrs[r] = gate_up_base + static_cast<long long>(e) * stride_gu;
    b_gu_ptrs[r] = norm_x + static_cast<long long>(token) * H;
    c_gu_ptrs[r] = expert_gate_up + static_cast<long long>(r) * 2 * I_moe;

    a_dn_ptrs[r] = down_base + static_cast<long long>(e) * stride_dn;
    b_dn_ptrs[r] = expert_act + static_cast<long long>(r) * I_moe;
    c_dn_ptrs[r] = expert_out + static_cast<long long>(r) * H;

    weights_out[r] = topk_w[r];
}

template <class T, bool ActByToken>
__device__ __forceinline__ void moe_matmul_select_wmma_body(
    const i32* __restrict__ topk_idx,
    const T* __restrict__ act,
    const T* __restrict__ weight_base,
    T* __restrict__ out,
    int top_k,
    int K,
    int N,
    long long expert_stride)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    static_assert(is_same<T, bf16>::value,
                  "pie_mma.cuh implements bf16 fragments only -- see its "
                  "static_assert, and do not extend it without a parity run");
    constexpr int N_TILE = 64;
    const int n0 = blockIdx.x * N_TILE;
    const int route = blockIdx.y;
    const int expert = topk_idx[route];

    if (n0 >= N) return;

    extern __shared__ __align__(16) unsigned char wmma_smem[];
    auto* a_tile = reinterpret_cast<T*>(wmma_smem);
    auto* c_tile = reinterpret_cast<float*>(a_tile + 16 * 16);
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int n_warp = n0 + warp_id * 16;
    if (expert < 0) {
        if (lane < 16) {
            out[static_cast<long long>(route) * N + n0 + warp_id * 16 + lane] =
                Elem<T>::from_f32(0.0f);
        }
        return;
    }

    const int token = route / top_k;
    const T* act_row = act + static_cast<long long>(ActByToken ? token : route) * K;
    const T* weight = weight_base + static_cast<long long>(expert) * expert_stride;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int i = threadIdx.x; i < 16 * 16; i += blockDim.x) {
            a_tile[i] = Elem<T>::from_f32(0.0f);
        }
        if (threadIdx.x < 16) {
            a_tile[threadIdx.x] = act_row[k0 + threadIdx.x];
        }
        __syncthreads();

        wmma::load_matrix_sync(
            a_frag, reinterpret_cast<const __nv_bfloat16*>(a_tile), 16);
        wmma::load_matrix_sync(
            b_frag,
            reinterpret_cast<const __nv_bfloat16*>(
                weight + static_cast<long long>(n_warp) * K + k0),
            K);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        __syncthreads();
    }

    wmma::store_matrix_sync(
        c_tile + warp_id * 16 * 16, acc_frag, 16, wmma::mem_row_major);
    __syncthreads();

    if (lane < 16) {
        const long long out_base = static_cast<long long>(route) * N + n0;
        out[out_base + warp_id * 16 + lane] =
            Elem<T>::from_f32(c_tile[warp_id * 16 * 16 + lane]);
    }
#endif
}

template <class T>
__global__ void moe_matmul_select_wmma_by_token(
    const i32* __restrict__ topk_idx,
    const T* __restrict__ act,
    const T* __restrict__ weight_base,
    T* __restrict__ out,
    int top_k, int K, int N, long long expert_stride)
{
    moe_matmul_select_wmma_body<T, true>(topk_idx, act, weight_base, out,
                                  top_k, K, N, expert_stride);
}

template <class T>
__global__ void moe_matmul_select_wmma_by_route(
    const i32* __restrict__ topk_idx,
    const T* __restrict__ act,
    const T* __restrict__ weight_base,
    T* __restrict__ out,
    int top_k, int K, int N, long long expert_stride)
{
    moe_matmul_select_wmma_body<T, false>(topk_idx, act, weight_base, out,
                                   top_k, K, N, expert_stride);
}

// **WHERE ONE EXPERT'S WEIGHTS ARE** (alto design §7, wave D2).
//
// `expert_table` is the bank's device-resident indirection table: entry `e`
// is the base address of expert `e`'s slot, wherever it lives — inside the
// device slab if it is resident, or in pinned host memory over UVA if it is
// not. It is DATA (article 5), so promoting an expert changes an entry and
// never an address a captured graph holds.
//
// `nullptr` is the fully-resident load, and it is not a slow arm of a fast
// one: the table is not read at all, the arithmetic is the
// `weight_base + expert * expert_stride` this kernel always did, and the
// generated code for that arm is what it was before D2 existed. That is
// dev's `place_all()` degeneration, spelled as a branch on a pointer the
// whole grid loads uniformly.
template <class T>
__device__ __forceinline__ const T* moe_expert_base(
    const T* __restrict__ weight_base,
    const T* const* __restrict__ expert_table,
    int expert,
    long long expert_stride)
{
    return expert_table != nullptr
        ? expert_table[expert]
        : weight_base + static_cast<long long>(expert) * expert_stride;
}

// **THE ONE STATISTIC THE FIRE PATH PUBLISHES** (article 3, applied to
// weights). One `atomicAdd` per routed expert per fire, into a device buffer
// at a fixed address; the host reads it between fires and promotes. No
// callback, no sync, and no host decision on the fire path — the count is
// device data that the settle-side readback carries out, exactly as the
// channel plane carries a commit word out.
//
// `nullptr` — the fully-resident load — costs the same uniform branch the
// table does and nothing else.
__device__ __forceinline__ void moe_note_expert(
    unsigned int* __restrict__ expert_hits, int expert)
{
    if (expert_hits != nullptr) {
        atomicAdd(&expert_hits[expert], 1u);
    }
}

template <class T, bool ActByToken, int kWarps, int kUnroll>
__device__ __forceinline__ void moe_matmul_select_gemv_body(
    const i32* __restrict__ topk_idx,
    const T* __restrict__ act,
    const T* __restrict__ weight_base,
    T* __restrict__ out,
    int top_k, int K, int N, long long expert_stride,
    const T* const* __restrict__ expert_table,
    unsigned int* __restrict__ expert_hits)
{
    const int route = blockIdx.y;
    const int row = blockIdx.x * kWarps + threadIdx.y;
    if (row >= N) return;
    const int lane = threadIdx.x;
    const int expert = topk_idx[route];

    if (expert < 0) {
        if (lane == 0) out[(long long)route * N + row] = Elem<T>::from_f32(0.f);
        return;
    }
    // One thread of one block per route does the counting: `blockIdx.x == 0`
    // picks the block that owns the first row tile and `threadIdx` picks its
    // first thread, so the count is per (route, fire) and not per block.
    if (blockIdx.x == 0 && threadIdx.y == 0 && lane == 0) {
        moe_note_expert(expert_hits, expert);
    }
    const T* w = moe_expert_base(weight_base, expert_table, expert, expert_stride)
        + (long long)row * K;
    const T* x = act + (long long)(ActByToken ? route / top_k : route) * K;

    float acc = 0.f;
    const int vec = K / kMoeVecWidth;
    const float4* w4 = reinterpret_cast<const float4*>(w);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    int i = lane;
    for (; i + 32 * (kUnroll - 1) < vec; i += 32 * kUnroll) {
        float4 wv[kUnroll];
        float4 xv[kUnroll];
        #pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            wv[u] = w4[i + 32 * u];
            xv[u] = x4[i + 32 * u];
        }
        #pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            const T* wb = reinterpret_cast<const T*>(&wv[u]);
            const T* xb = reinterpret_cast<const T*>(&xv[u]);
            #pragma unroll
            for (int j = 0; j < kMoeVecWidth; ++j) {
                acc += Elem<T>::to_f32(wb[j]) * Elem<T>::to_f32(xb[j]);
            }
        }
    }
    for (; i < vec; i += 32) {
        float4 wv = w4[i];
        float4 xv = x4[i];
        const T* wb = reinterpret_cast<const T*>(&wv);
        const T* xb = reinterpret_cast<const T*>(&xv);
        #pragma unroll
        for (int j = 0; j < kMoeVecWidth; ++j) {
            acc += Elem<T>::to_f32(wb[j]) * Elem<T>::to_f32(xb[j]);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) out[(long long)route * N + row] = Elem<T>::from_f32(acc);
}

template <class T, bool ActByToken, int kWarps, int kUnroll = 1>
__global__ void moe_matmul_select_gemv(
    const i32* __restrict__ topk_idx,
    const T* __restrict__ act,
    const T* __restrict__ weight_base,
    T* __restrict__ out,
    int top_k, int K, int N, long long expert_stride,
    const T* const* __restrict__ expert_table,
    unsigned int* __restrict__ expert_hits)
{
    moe_matmul_select_gemv_body<T, ActByToken, kWarps, kUnroll>(
        topk_idx, act, weight_base, out, top_k, K, N, expert_stride,
        expert_table, expert_hits);
}

template <class T>
__global__ void moe_matmul_select_gemv_by_token(
    const i32* __restrict__ topk_idx,
    const T* __restrict__ act,
    const T* __restrict__ weight_base,
    T* __restrict__ out,
    int top_k, int K, int N, long long expert_stride,
    const T* const* __restrict__ expert_table,
    unsigned int* __restrict__ expert_hits)
{
    moe_matmul_select_gemv_body<T, true, kGemvWarps, 1>(
        topk_idx, act, weight_base, out, top_k, K, N, expert_stride,
        expert_table, expert_hits);
}

template <class T>
__global__ void moe_matmul_select_gemv_by_route(
    const i32* __restrict__ topk_idx,
    const T* __restrict__ act,
    const T* __restrict__ weight_base,
    T* __restrict__ out,
    int top_k, int K, int N, long long expert_stride,
    const T* const* __restrict__ expert_table,
    unsigned int* __restrict__ expert_hits)
{
    moe_matmul_select_gemv_body<T, false, kGemvWarps, 1>(
        topk_idx, act, weight_base, out, top_k, K, N, expert_stride,
        expert_table, expert_hits);
}

template <class T>
__global__ void moe_align_decode(
    const T* __restrict__ topk_idx,
    T* __restrict__ sorted_route_ids,
    T* __restrict__ expert_ids,
    T* __restrict__ route_to_aligned_row,
    int num_routes,
    int num_experts,
    int block_size,
    int max_blocks,
    T* __restrict__ num_tokens_past_padded)
{
    static_assert(is_same<T, i32>::value, "the routing indices are i32");
    extern __shared__ i32 align_smem[];
    i32* counts = align_smem;
    i32* offsets = counts + num_experts;
    i32* fill = offsets + num_experts + 1;
    i32* warp_totals = fill + num_experts;
    i32* block_base = warp_totals + 32;

    const int aligned_rows = max_blocks * block_size;
    if (threadIdx.x == 0) *block_base = 0;
    for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
        counts[i] = 0;
        fill[i] = 0;
    }
    for (int i = threadIdx.x; i < aligned_rows; i += blockDim.x) {
        sorted_route_ids[i] = num_routes;
    }
    for (int i = threadIdx.x; i < max_blocks; i += blockDim.x) {
        expert_ids[i] = -1;
    }
    __syncthreads();

    for (int r = threadIdx.x; r < num_routes; r += blockDim.x) {
        const int e = topk_idx[r];
        if (0 <= e && e < num_experts) {
            atomicAdd(counts + e, 1);
        }
    }
    __syncthreads();

    {
        const int lane = threadIdx.x & 31;
        const int warp = static_cast<int>(threadIdx.x) >> 5;
        const int num_warps = static_cast<int>(blockDim.x) >> 5;
        for (int base = 0; base < num_experts; base += static_cast<int>(blockDim.x)) {
            const int e = base + static_cast<int>(threadIdx.x);
            int padded = 0;
            if (e < num_experts) {
                const int c = counts[e];
                padded = ((c + block_size - 1) / block_size) * block_size;
            }
            int value = padded;
            for (int off = 1; off < 32; off <<= 1) {
                const int n = __shfl_up_sync(0xffffffffu, value, off);
                if (lane >= off) value += n;
            }
            if (lane == 31) warp_totals[warp] = value;
            __syncthreads();
            if (warp == 0) {
                int t = (lane < num_warps) ? warp_totals[lane] : 0;
                for (int off = 1; off < 32; off <<= 1) {
                    const int n = __shfl_up_sync(0xffffffffu, t, off);
                    if (lane >= off) t += n;
                }
                if (lane < num_warps) warp_totals[lane] = t;
            }
            __syncthreads();
            const int warp_prefix = (warp == 0) ? 0 : warp_totals[warp - 1];
            if (e < num_experts) {

                offsets[e] = *block_base + warp_prefix + value - padded;
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                *block_base += warp_totals[num_warps - 1];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            offsets[num_experts] = *block_base;
#if defined(PIE_MOE_ALIGN_REPORT)

            printf("[moe-align] used=%d max=%d routes=%d experts=%d\n",
                   *block_base / block_size, max_blocks, num_routes,
                   num_experts);
#endif
        }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        const int begin = offsets[e];
        const int end = offsets[e + 1];
        for (int row = begin; row < end; row += block_size) {
            const int b = row / block_size;
            if (b < max_blocks) expert_ids[b] = e;
        }
    }
    __syncthreads();

    for (int r = threadIdx.x; r < num_routes; r += blockDim.x) {
        const int e = topk_idx[r];
        if (0 <= e && e < num_experts) {
            const int pos = atomicAdd(fill + e, 1);
            const int out = offsets[e] + pos;
            if (out < aligned_rows) {
                sorted_route_ids[out] = r;
                if (route_to_aligned_row != nullptr) {
                    route_to_aligned_row[r] = out;
                }
            }
        }
    }

    __syncthreads();
    if (num_tokens_past_padded != nullptr && threadIdx.x == 0) {
        *num_tokens_past_padded = *block_base;
    }
}

template <class T>
__global__ void moe_bucket_exact(
    const T* __restrict__ topk_idx,
    T* __restrict__ sorted_route_ids,
    T* __restrict__ route_to_sorted_row,
    T* __restrict__ counts_out,
    int num_routes,
    int num_experts)
{
    static_assert(is_same<T, i32>::value, "the routing indices are i32");
    extern __shared__ i32 bucket_smem[];
    i32* counts = bucket_smem;
    i32* offsets = counts + num_experts;
    i32* fill = offsets + num_experts + 1;

    for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
        counts[i] = 0;
        fill[i] = 0;
    }
    __syncthreads();

    for (int r = threadIdx.x; r < num_routes; r += blockDim.x) {
        const int e = topk_idx[r];
        if (0 <= e && e < num_experts) {
            atomicAdd(counts + e, 1);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int running = 0;
        for (int e = 0; e < num_experts; ++e) {
            offsets[e] = running;
            const int c = counts[e];
            counts_out[e] = c;
            running += c;
        }
        offsets[num_experts] = running;
    }
    __syncthreads();

    for (int r = threadIdx.x; r < num_routes; r += blockDim.x) {
        const int e = topk_idx[r];
        if (0 <= e && e < num_experts) {
            const int pos = atomicAdd(fill + e, 1);
            const int out = offsets[e] + pos;
            sorted_route_ids[out] = r;
            route_to_sorted_row[r] = out;
        }
    }
}

template <class T>
__global__ void gather_moe_aligned_inputs_vec(
    const T* __restrict__ norm_x,
    const i32* __restrict__ sorted_route_ids,
    T* __restrict__ aligned_in,
    int num_routes,
    int aligned_rows,
    int top_k,
    int hidden_vec,
    int shared_row_begin,
    int num_tokens)
{
    static_assert(sizeof(T) == 2, "kMoeVecWidth elements are one uint4");
    const int hv = blockIdx.y * blockDim.x + threadIdx.x;
    const int row = blockIdx.x;
    if (hv >= hidden_vec || row >= aligned_rows) return;
    int token = -1;
    if (shared_row_begin >= 0 && row >= shared_row_begin) {
        const int t = row - shared_row_begin;
        if (t < num_tokens) token = t;
    } else {
        const int route = sorted_route_ids[row];
        if (route < num_routes) token = route / top_k;
    }
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (token >= 0) {
        v = reinterpret_cast<const uint4*>(norm_x)[
            static_cast<long long>(token) * hidden_vec + hv];
    }
    reinterpret_cast<uint4*>(aligned_in)[
        static_cast<long long>(row) * hidden_vec + hv] = v;
}

template <class T>
__global__ void gather_moe_aligned_inputs(
    const T* __restrict__ norm_x,
    const i32* __restrict__ sorted_route_ids,
    T* __restrict__ aligned_in,
    int num_routes,
    int aligned_rows,
    int top_k,
    int hidden,
    int shared_row_begin,
    int num_tokens)
{
    const int h = blockIdx.y * blockDim.x + threadIdx.x;
    const int row = blockIdx.x;
    if (h >= hidden || row >= aligned_rows) return;
    int token = -1;
    if (shared_row_begin >= 0 && row >= shared_row_begin) {
        const int t = row - shared_row_begin;
        if (t < num_tokens) token = t;
    } else {
        const int route = sorted_route_ids[row];
        if (route < num_routes) token = route / top_k;
    }
    T v = Elem<T>::from_f32(0.0f);
    if (token >= 0) {
        v = norm_x[static_cast<long long>(token) * hidden + h];
    }
    aligned_in[static_cast<long long>(row) * hidden + h] = v;
}

template <class T>
__global__ void build_moe_ptrs_aligned(
    const i32* __restrict__ expert_ids,
    const T* __restrict__ gate_up_base,
    const T* __restrict__ down_base,
    const T* __restrict__ aligned_in,
    T* __restrict__ aligned_gate_up,
    T* __restrict__ aligned_act,
    T* __restrict__ aligned_out,
    const T** __restrict__ a_gu_ptrs,
    const T** __restrict__ b_gu_ptrs,
    T** __restrict__ c_gu_ptrs,
    const T** __restrict__ a_dn_ptrs,
    const T** __restrict__ b_dn_ptrs,
    T** __restrict__ c_dn_ptrs,
    int max_blocks,
    int block_size,
    int H,
    int I_moe,
    int routed_blocks,
    const T* __restrict__ shared_gate_up_base,
    const T* __restrict__ shared_down_base)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= max_blocks) return;
    const bool is_shared = (b >= routed_blocks);
    int e = is_shared ? 0 : expert_ids[b];
    if (e < 0) e = 0;
    const long long row = static_cast<long long>(b) * block_size;
    const long long stride_gu = 2LL * I_moe * H;
    const long long stride_dn = static_cast<long long>(H) * I_moe;

    a_gu_ptrs[b] = is_shared
        ? shared_gate_up_base
        : gate_up_base + static_cast<long long>(e) * stride_gu;
    b_gu_ptrs[b] = aligned_in + row * H;
    c_gu_ptrs[b] = aligned_gate_up + row * (2LL * I_moe);

    a_dn_ptrs[b] = is_shared
        ? shared_down_base
        : down_base + static_cast<long long>(e) * stride_dn;
    b_dn_ptrs[b] = aligned_act + row * I_moe;
    c_dn_ptrs[b] = aligned_out + row * H;
}

template <class T>
__global__ void add_moe_route_bias(
    T* __restrict__ out,
    const T* __restrict__ bias,
    const i32* __restrict__ topk_idx,
    int num_routes, int cols, int out_stride)
{
    const int route = blockIdx.x;
    if (route >= num_routes) return;
    const int e = topk_idx[route];
    if (e < 0) return;
    const T* b = bias + static_cast<long long>(e) * cols;
    T* o = out + static_cast<long long>(route) * out_stride;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        o[i] = Elem<T>::from_f32(Elem<T>::to_f32(o[i]) + Elem<T>::to_f32(b[i]));
    }
}

template <class T>
__global__ void transpose_expert_scales(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int n, int kg)
{
    const int e = blockIdx.z;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= kg) return;
    const long long base = static_cast<long long>(e) * n * kg;
    dst[base + static_cast<long long>(j) * n + i] =
        src[base + static_cast<long long>(i) * kg + j];
}

template <class T>
__global__ void moe_weighted_sum_scalar(
    T* __restrict__ out,
    const T* __restrict__ src,
    float weight, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float ov = Elem<T>::to_f32(out[i]);
    const float sv = Elem<T>::to_f32(src[i]);
    out[i] = Elem<T>::from_f32(ov + weight * sv);
}

template <class T>
__global__ void moe_weighted_sum_batched(
    T* __restrict__ out,
    const T* __restrict__ src,
    const float* __restrict__ weights,
    int batch, int hidden)
{
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= hidden) return;
    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < kMaxTopK; ++k) {
        if (k >= batch) break;
        const float v = Elem<T>::to_f32(src[(long long)k * hidden + h]);
        acc += weights[k] * v;
    }
    out[h] = Elem<T>::from_f32(acc);
}

template <class T>
__global__ void moe_weighted_sum(
    T* __restrict__ out,
    const T* __restrict__ src,
    const float* __restrict__ weights,
    int top_k, int hidden)
{
    const int n = blockIdx.x;
    const int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (h >= hidden) return;
    const long long base = static_cast<long long>(n) * top_k;
    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < kMaxTopK; ++k) {
        if (k >= top_k) break;
        const long long r = base + k;
        const float v = Elem<T>::to_f32(src[r * hidden + h]);
        acc += weights[r] * v;
    }
    out[static_cast<long long>(n) * hidden + h] = Elem<T>::from_f32(acc);
}

template <class T>
__global__ void moe_bias_sum(
    T* __restrict__ out,
    const T* __restrict__ x,
    const T* __restrict__ bias,
    const i32* __restrict__ topk_idx,
    const float* __restrict__ weights,
    int top_k, int hidden)
{
    const int n = blockIdx.x;
    const int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (h >= hidden) return;
    const long long base = static_cast<long long>(n) * top_k;
    const long long i = static_cast<long long>(n) * hidden + h;
    float acc = Elem<T>::to_f32(x[i]);
    for (int k = 0; k < top_k; ++k) {
        const long long r = base + k;
        const int e = topk_idx[r];

        if (e < 0) continue;
        const float b = Elem<T>::to_f32(
            bias[static_cast<long long>(e) * hidden + h]);
        acc += weights[r] * b;
    }
    out[i] = Elem<T>::from_f32(acc);
}

template <class T>
__global__ void moe_weighted_sum_add(
    T* __restrict__ out,
    const T* __restrict__ src,
    const float* __restrict__ weights,
    int top_k, int hidden)
{
    const int n = blockIdx.x;
    const int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (h >= hidden) return;
    const long long base = static_cast<long long>(n) * top_k;
    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < kMaxTopK; ++k) {
        if (k >= top_k) break;
        const long long r = base + k;
        const float v = Elem<T>::to_f32(src[r * hidden + h]);
        acc += weights[r] * v;
    }
    const long long out_idx = static_cast<long long>(n) * hidden + h;
    out[out_idx] = Elem<T>::from_f32(Elem<T>::to_f32(out[out_idx]) + acc);
}

template <class T>
__global__ void moe_weighted_sum_vec(
    T* __restrict__ out,
    const T* __restrict__ src,
    const float* __restrict__ weights,
    int top_k, int hidden_vec)
{
    static_assert(sizeof(T) == 2, "kMoeVecWidth elements are one uint4");
    const int n = blockIdx.x;
    const int hv = blockIdx.y * blockDim.x + threadIdx.x;
    if (hv >= hidden_vec) return;
    const long long base = static_cast<long long>(n) * top_k;
    float acc[kMoeVecWidth];
#pragma unroll
    for (int j = 0; j < kMoeVecWidth; ++j) acc[j] = 0.f;
    const uint4* sv = reinterpret_cast<const uint4*>(src);
    for (int k = 0; k < top_k; ++k) {
        const long long r = base + k;
        const uint4 v = sv[r * hidden_vec + hv];
        const auto* vh = reinterpret_cast<const T*>(&v);
        const float w = weights[r];
#pragma unroll
        for (int j = 0; j < kMoeVecWidth; ++j) acc[j] += w * Elem<T>::to_f32(vh[j]);
    }
    uint4 o;
    auto* oh = reinterpret_cast<T*>(&o);
#pragma unroll
    for (int j = 0; j < kMoeVecWidth; ++j) oh[j] = Elem<T>::from_f32(acc[j]);
    reinterpret_cast<uint4*>(out)[
        static_cast<long long>(n) * hidden_vec + hv] = o;
}

template <class T>
__global__ void moe_weighted_sum_add_vec(
    T* __restrict__ out,
    const T* __restrict__ src,
    const float* __restrict__ weights,
    int top_k, int hidden_vec)
{
    static_assert(sizeof(T) == 2, "kMoeVecWidth elements are one uint4");
    const int n = blockIdx.x;
    const int hv = blockIdx.y * blockDim.x + threadIdx.x;
    if (hv >= hidden_vec) return;
    const long long base = static_cast<long long>(n) * top_k;
    float acc[kMoeVecWidth];
#pragma unroll
    for (int j = 0; j < kMoeVecWidth; ++j) acc[j] = 0.f;
    const uint4* sv = reinterpret_cast<const uint4*>(src);
    for (int k = 0; k < top_k; ++k) {
        const long long r = base + k;
        const uint4 v = sv[r * hidden_vec + hv];
        const auto* vh = reinterpret_cast<const T*>(&v);
        const float w = weights[r];
#pragma unroll
        for (int j = 0; j < kMoeVecWidth; ++j) acc[j] += w * Elem<T>::to_f32(vh[j]);
    }
    const long long oi = static_cast<long long>(n) * hidden_vec + hv;
    uint4 o = reinterpret_cast<uint4*>(out)[oi];
    auto* oh = reinterpret_cast<T*>(&o);
#pragma unroll
    for (int j = 0; j < kMoeVecWidth; ++j) {
        oh[j] = Elem<T>::from_f32(Elem<T>::to_f32(oh[j]) + acc[j]);
    }
    reinterpret_cast<uint4*>(out)[oi] = o;
}

template <class T>
__global__ void moe_weighted_sum_aligned(
    T* __restrict__ out,
    const T* __restrict__ aligned_out,
    const float* __restrict__ weights,
    const i32* __restrict__ route_to_aligned_row,
    int top_k,
    int hidden)
{
    const int n = blockIdx.x;
    const int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (h >= hidden) return;
    const long long base = static_cast<long long>(n) * top_k;
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < kMaxTopK; ++k) {
        if (k >= top_k) break;
        const long long route = base + k;
        const int row = route_to_aligned_row[route];
        const float v = Elem<T>::to_f32(
            aligned_out[static_cast<long long>(row) * hidden + h]);
        acc += weights[route] * v;
    }
    out[static_cast<long long>(n) * hidden + h] = Elem<T>::from_f32(acc);
}

template <class T>
__global__ void reorder_moe_aligned_output_vec(
    const T* __restrict__ aligned_out,
    const i32* __restrict__ sorted_route_ids,
    T* __restrict__ route_out,
    int num_routes,
    int aligned_rows,
    int hidden_vec,
    int shared_row_begin,
    int num_tokens,
    T* __restrict__ shared_out)
{
    static_assert(sizeof(T) == 2, "kMoeVecWidth elements are one uint4");
    const int hv = blockIdx.y * blockDim.x + threadIdx.x;
    const int row = blockIdx.x;
    if (hv >= hidden_vec || row >= aligned_rows) return;
    const uint4 v = reinterpret_cast<const uint4*>(aligned_out)[
        static_cast<long long>(row) * hidden_vec + hv];
    if (shared_row_begin >= 0 && row >= shared_row_begin) {
        const int t = row - shared_row_begin;
        if (t < num_tokens) {
            reinterpret_cast<uint4*>(shared_out)[
                static_cast<long long>(t) * hidden_vec + hv] = v;
        }
        return;
    }
    const int route = sorted_route_ids[row];
    if (route >= num_routes) return;
    reinterpret_cast<uint4*>(route_out)[
        static_cast<long long>(route) * hidden_vec + hv] = v;
}

template <class T>
__global__ void reorder_moe_aligned_output(
    const T* __restrict__ aligned_out,
    const i32* __restrict__ sorted_route_ids,
    T* __restrict__ route_out,
    int num_routes,
    int aligned_rows,
    int hidden,
    int shared_row_begin,
    int num_tokens,
    T* __restrict__ shared_out)
{
    const int h = blockIdx.y * blockDim.x + threadIdx.x;
    const int row = blockIdx.x;
    if (h >= hidden || row >= aligned_rows) return;
    const T v = aligned_out[static_cast<long long>(row) * hidden + h];
    if (shared_row_begin >= 0 && row >= shared_row_begin) {
        const int t = row - shared_row_begin;
        if (t < num_tokens) {
            shared_out[static_cast<long long>(t) * hidden + h] = v;
        }
        return;
    }
    const int route = sorted_route_ids[row];
    if (route >= num_routes) return;
    route_out[static_cast<long long>(route) * hidden + h] = v;
}

}
