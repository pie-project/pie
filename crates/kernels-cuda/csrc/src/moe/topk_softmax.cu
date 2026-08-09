#include "moe/topk_softmax.hpp"

#include <cuda_bf16.h>
#include <cfloat>
#include <cstdlib>
#include <stdexcept>

namespace pie_cuda_driver::kernels::moe {

namespace {

constexpr int BLOCK = 64;
// Qwen3.6-35B-A3B uses 256 experts; Kimi K2.6 uses 384. Keep a single
// static shared-memory slab large enough for both. 512 floats == 2 KB.
constexpr int MAX_EXPERTS = 512;

// Block-wide argmax over `scores[0..num_experts)`, ties resolved to the
// LOWEST index — the same winner a serial `for (j) if (s[j] > best)` scan
// picks, so routing decisions stay bit-identical.
//
// The serial form cost K * num_experts iterations on thread 0 while the
// other 63 threads idled: at 256 experts and K = 8 that is 2048 dependent
// shared-memory reads, which measured 21 us per layer (7% of a Qwen3.6
// decode step). Strided scan plus a log-depth reduction is ~10 steps.
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
    // Strictly above `floor_value`, matching the serial scan's seed: an
    // already-picked expert is excluded by writing the floor back into
    // `scores`, and the floor itself must never win.
    float local_v = floor_value;
    int local_i = -1;
    for (int j = tid; j < num_experts; j += BLOCK) {
        const float v = scores[j];
        if (v > local_v) {
            local_v = v;
            local_i = j;
        }
    }
    value_buf[tid] = local_v;
    index_buf[tid] = local_i;
    __syncthreads();
    // A shared-memory tree over all BLOCK lanes costs log2(BLOCK)
    // __syncthreads PER ROUND, and there are K rounds. Fold the upper
    // warp once, then finish inside warp 0 with shuffles, which need no
    // barrier at all: 2 barriers per round instead of 8.
    static_assert(BLOCK == 64, "block_argmax folds exactly one upper warp");
    if (tid < 32) {
        float v = value_buf[tid];
        int i = index_buf[tid];
        // A strided scan gives thread t the indices t, t+BLOCK, ..., so the
        // lower index of a tie is not always in the lower lane: compare
        // indices explicitly rather than relying on lane order. This keeps
        // the winner identical to a serial `if (s[j] > best)` scan.
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
    // No trailing barrier: every caller syncs after acting on the winner,
    // which is what orders the next round's `value_buf` writes.
}

// One block per token. Phase 1: thread-local max-reduce + exp+sum-reduce
// for softmax. Phase 2: K iterations of argmax-with-exclusion to pick the
// top-K probs. Phase 3: thread 0 renormalizes and writes back.
// `FUSED_GEMV` computes the router logits here instead of reading them: one
// warp per expert, walking that expert's weight row. The router is a
// [num_experts, hidden] projection of ONE token, so as a standalone GEMV it is
// 32 blocks on 132 SMs and costs what a launch costs whatever it does -- the
// tuner measures 5.5 us for 0.18 MB. Folding it into the consumer that was
// going to read its output anyway removes the launch and the round trip
// through HBM, and the logits never leave shared memory.
template <bool FUSED_GEMV>
__global__ void topk_softmax_bf16_kernel(
    const __nv_bfloat16* __restrict__ logits,   // FUSED_GEMV: the weight
    const __nv_bfloat16* __restrict__ act,      // FUSED_GEMV only
    const __nv_bfloat16* __restrict__ bias,     // FUSED_GEMV only, may be null
    std::int32_t* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    int num_experts, int K, int hidden)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const __nv_bfloat16* row =
        FUSED_GEMV ? logits
                   : logits + static_cast<long long>(n) * num_experts;

    __shared__ float probs[MAX_EXPERTS];
    __shared__ float buf[BLOCK];
    __shared__ int ibuf[BLOCK];

    // 1. Stage row into shared memory + find max.
    float local_max = -FLT_MAX;
    if constexpr (FUSED_GEMV) {
        const __nv_bfloat16* x =
            act + static_cast<long long>(n) * hidden;
        const int warp = tid >> 5;
        const int lane = tid & 31;
        constexpr int kWarps = BLOCK / 32;
        for (int e = warp; e < num_experts; e += kWarps) {
            const __nv_bfloat16* w =
                row + static_cast<long long>(e) * hidden;
            float acc = 0.f;
            for (int i = lane; i < hidden; i += 32) {
                acc += __bfloat162float(w[i]) * __bfloat162float(x[i]);
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                acc += __shfl_down_sync(0xffffffffu, acc, off);
            }
            if (lane == 0) {
                if (bias != nullptr) acc += __bfloat162float(bias[e]);
                probs[e] = acc;
            }
        }
        __syncthreads();
        for (int j = tid; j < num_experts; j += BLOCK) {
            if (probs[j] > local_max) local_max = probs[j];
        }
    } else {
    for (int j = tid; j < num_experts; j += BLOCK) {
        const float v = __bfloat162float(row[j]);
        probs[j] = v;
        if (v > local_max) local_max = v;
    }
    }
    buf[tid] = local_max;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) buf[tid] = fmaxf(buf[tid], buf[tid + off]);
        __syncthreads();
    }
    const float row_max = buf[0];
    __syncthreads();

    // 2. exp + sum.
    float local_sum = 0.f;
    for (int j = tid; j < num_experts; j += BLOCK) {
        const float e = expf(probs[j] - row_max);
        probs[j] = e;
        local_sum += e;
    }
    buf[tid] = local_sum;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) buf[tid] += buf[tid + off];
        __syncthreads();
    }
    const float inv_Z = 1.f / buf[0];
    __syncthreads();

    // 3. Normalize in shared mem, then K block-wide argmaxes with exclusion.
    for (int j = tid; j < num_experts; j += BLOCK) probs[j] *= inv_Z;
    __syncthreads();

    std::int32_t* out_idx = topk_idx + static_cast<long long>(n) * K;
    float*        out_w   = topk_w   + static_cast<long long>(n) * K;
    float w_sum = 0.f;
    for (int k = 0; k < K; ++k) {
        float best_v = -1.f;
        int best_i = -1;
        block_argmax(probs, num_experts, -1.f, buf, ibuf, best_v, best_i);
        if (tid == 0) {
            out_idx[k] = best_i;
            out_w[k] = best_v;
            if (best_i >= 0) probs[best_i] = -1.f;  // exclude on next pass
        }
        w_sum += best_v;
        __syncthreads();
    }
    if (tid == 0) {
        const float inv_w = 1.f / w_sum;
        for (int k = 0; k < K; ++k) out_w[k] *= inv_w;
    }
}

// Single-warp top-K softmax: no shared memory, no __syncthreads.
//
// The block form above pays three block-wide reduction trees (max, sum, and
// K argmaxes) through shared memory. At BLOCK=64 each tree is 6 rounds and
// every round carries a __syncthreads, so routing one decode token through
// 32 experts runs ~36 barriers to do 32 exponentials. Measured on B200 with
// graph replay: 4.39 us/call against a 0.54 us empty-kernel floor, and it is
// called once per layer per token -- 105 us of a ~2.4 ms decode step.
//
// When the experts fit in a warp's registers (PER_LANE values per lane) the
// same reductions are __shfl_xor, which need no barrier and no shared
// traffic. Ties still resolve to the LOWEST index, so routing decisions stay
// identical to the block form -- that is a correctness requirement, not a
// nicety: a different expert choice is a different model.
//
// This is not a Blackwell path; warp shuffles are universal and this helps
// every architecture equally, so it is not gated on compute capability.
template <int PER_LANE>
__global__ void topk_softmax_warp_bf16_kernel(
    const __nv_bfloat16* __restrict__ logits,
    std::int32_t* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    int num_experts, int K)
{
    const int n = blockIdx.x;
    const int lane = static_cast<int>(threadIdx.x);
    const __nv_bfloat16* row =
        logits + static_cast<long long>(n) * num_experts;

    // The full softmax is never needed. The block form computes probs over
    // all experts, takes the top K, then renormalises them by their own sum --
    // so the partition function divides out exactly:
    //
    //   w_k = (e^{v_k - m} / Z) / sum_j (e^{v_j - m} / Z)
    //       =  e^{v_k - m}      / sum_j e^{v_j - m}      over the K winners
    //
    // and exp is monotonic, so the K winners are the same whether it is
    // applied or not. Selecting on the RAW logits therefore gives identical
    // routing while costing K exponentials instead of num_experts, and drops
    // a whole warp-sum reduction. At E=32 that is 32 expf calls saved.
    int idx[PER_LANE];
    float val[PER_LANE];
#pragma unroll
    for (int i = 0; i < PER_LANE; ++i) {
        idx[i] = lane + i * 32;
        val[i] = idx[i] < num_experts ? __bfloat162float(row[idx[i]])
                                      : -FLT_MAX;
    }

    // K rounds of warp argmax with exclusion. Every lane ends each round
    // holding the winner, so the running sum needs no broadcast.
    std::int32_t* out_idx = topk_idx + static_cast<long long>(n) * K;
    float* out_w = topk_w + static_cast<long long>(n) * K;
    float best_w[8];
    int best_e[8];
    for (int k = 0; k < K; ++k) {
        // Seeded below every representable logit, matching block_argmax's
        // "strictly above the floor" rule now that the scores are raw logits
        // rather than non-negative probabilities.
        float bv = -FLT_MAX;
        int bi = -1;
#pragma unroll
        for (int i = 0; i < PER_LANE; ++i) {
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
        for (int i = 0; i < PER_LANE; ++i) {
            if (idx[i] == bi) val[i] = -FLT_MAX;  // exclude on the next round
        }
        best_w[k] = bv;
        best_e[k] = bi;
    }
    if (lane == 0) {
        // best_w[0] is the row max by construction, so it is the shift that
        // keeps the exponentials in range -- the same one the block form
        // subtracts. Copy it out first: writing best_w[0] on the k=0
        // iteration would leave every later term shifted by exp(0)=1 instead
        // of by the max.
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

}  // namespace

void topk_softmax_bf16(
    const void* logits,
    std::int32_t* topk_idx, float* topk_w,
    int N, int num_experts, int K,
    cudaStream_t stream)
{
    if (N <= 0 || num_experts <= 0 || K <= 0) return;
    if (num_experts > MAX_EXPERTS) {
        throw std::runtime_error("topk_softmax_bf16: num_experts exceeds MAX_EXPERTS");
    }
    // PIE_TOPK_WARP=0 forces the block form, for A/B measurement.
    static const bool warp_ok = [] {
        const char* v = std::getenv("PIE_TOPK_WARP");
        return v == nullptr || v[0] != '0';
    }();
    topk_softmax_bf16_form(logits, topk_idx, topk_w, N, num_experts, K,
                                  warp_ok, stream);
}

void topk_softmax_bf16_form(
    const void* logits,
    std::int32_t* topk_idx, float* topk_w,
    int N, int num_experts, int K,
    bool use_warp,
    cudaStream_t stream)
{
    if (N <= 0 || num_experts <= 0 || K <= 0) return;
    if (num_experts > MAX_EXPERTS) {
        throw std::runtime_error("topk_softmax_bf16: num_experts exceeds MAX_EXPERTS");
    }
    // The warp form keeps the experts in registers, so it applies while they
    // fit (<= 512, which is MAX_EXPERTS) and while the K winners fit the small
    // result array (<= 8). Qwen3.6-35B-A3B routes through more than 128 and
    // was falling back to the block form at 7.56 us/call, 4.9% of its step.
    if (use_warp && K <= 8 && num_experts <= 512) {
        const auto* in = static_cast<const __nv_bfloat16*>(logits);
        if (num_experts <= 32) {
            topk_softmax_warp_bf16_kernel<1><<<N, 32, 0, stream>>>(
                in, topk_idx, topk_w, num_experts, K);
        } else if (num_experts <= 64) {
            topk_softmax_warp_bf16_kernel<2><<<N, 32, 0, stream>>>(
                in, topk_idx, topk_w, num_experts, K);
        } else if (num_experts <= 128) {
            topk_softmax_warp_bf16_kernel<4><<<N, 32, 0, stream>>>(
                in, topk_idx, topk_w, num_experts, K);
        } else if (num_experts <= 256) {
            topk_softmax_warp_bf16_kernel<8><<<N, 32, 0, stream>>>(
                in, topk_idx, topk_w, num_experts, K);
        } else {
            // 512 experts is MAX_EXPERTS; 16 values per lane is 32 registers
            // of scores plus indices, which still leaves the warp room.
            topk_softmax_warp_bf16_kernel<16><<<N, 32, 0, stream>>>(
                in, topk_idx, topk_w, num_experts, K);
        }
        return;
    }
    topk_softmax_bf16_kernel</*FUSED_GEMV=*/false><<<N, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits),
        nullptr, nullptr, topk_idx, topk_w,
        num_experts, K, 0);
}

void router_topk_softmax_bf16(
    const void* act,
    const void* router_weight,
    const void* router_bias,
    std::int32_t* topk_idx,
    float* topk_w,
    int N, int num_experts, int K, int hidden,
    cudaStream_t stream)
{
    if (N <= 0 || num_experts <= 0 || K <= 0 || hidden <= 0) return;
    if (num_experts > MAX_EXPERTS) {
        throw std::runtime_error(
            "router_topk_softmax_bf16: num_experts exceeds MAX_EXPERTS");
    }
    topk_softmax_bf16_kernel</*FUSED_GEMV=*/true><<<N, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(router_weight),
        static_cast<const __nv_bfloat16*>(act),
        static_cast<const __nv_bfloat16*>(router_bias),
        topk_idx, topk_w, num_experts, K, hidden);
}

namespace {

__global__ void apply_per_expert_scale_kernel(
    const std::int32_t* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    const __nv_bfloat16* __restrict__ per_expert_scale,
    int total)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    const int e = topk_idx[t];
    const float s = __bfloat162float(per_expert_scale[e]);
    topk_w[t] *= s;
}

}  // namespace

void apply_per_expert_scale_bf16(
    const std::int32_t* topk_idx,
    float* topk_w,
    const void* per_expert_scale_bf16,
    int N, int K,
    cudaStream_t stream)
{
    const int total = N * K;
    if (total <= 0) return;
    constexpr int BLOCK_T = 256;
    const int grid = (total + BLOCK_T - 1) / BLOCK_T;
    apply_per_expert_scale_kernel<<<grid, BLOCK_T, 0, stream>>>(
        topk_idx, topk_w,
        static_cast<const __nv_bfloat16*>(per_expert_scale_bf16),
        total);
}

namespace {

__global__ void topk_sigmoid_bias_bf16_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const float* __restrict__ correction_bias,
    std::int32_t* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    int num_experts,
    int K,
    int normalize,
    float routed_scaling_factor)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const __nv_bfloat16* row =
        logits + static_cast<long long>(n) * num_experts;

    __shared__ float probs[MAX_EXPERTS];
    __shared__ float choice[MAX_EXPERTS];
    __shared__ float buf[BLOCK];
    __shared__ int ibuf[BLOCK];

    for (int j = tid; j < num_experts; j += BLOCK) {
        const float z = __bfloat162float(row[j]);
        const float p = 1.f / (1.f + __expf(-z));
        probs[j] = p;
        choice[j] = p + correction_bias[j];
    }
    __syncthreads();

    std::int32_t* out_idx = topk_idx + static_cast<long long>(n) * K;
    float* out_w = topk_w + static_cast<long long>(n) * K;
    float sum = 0.f;
    for (int k = 0; k < K; ++k) {
        float best_v = -FLT_MAX;
        int best_i = -1;
        block_argmax(choice, num_experts, -FLT_MAX, buf, ibuf, best_v, best_i);
        const float weight = best_i >= 0 ? probs[best_i] : 0.f;
        if (tid == 0) {
            out_idx[k] = best_i;
            out_w[k] = weight;
            if (best_i >= 0) choice[best_i] = -FLT_MAX;
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

__global__ void topk_sigmoid_bias_fp32_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ correction_bias,
    std::int32_t* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    int num_experts,
    int K,
    int normalize,
    float routed_scaling_factor)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const float* row = logits + static_cast<long long>(n) * num_experts;

    __shared__ float probs[MAX_EXPERTS];
    __shared__ float choice[MAX_EXPERTS];
    __shared__ float buf[BLOCK];
    __shared__ int ibuf[BLOCK];

    for (int j = tid; j < num_experts; j += BLOCK) {
        const float z = row[j];
        const float p = 1.f / (1.f + __expf(-z));
        probs[j] = p;
        choice[j] = p + correction_bias[j];
    }
    __syncthreads();

    std::int32_t* out_idx = topk_idx + static_cast<long long>(n) * K;
    float* out_w = topk_w + static_cast<long long>(n) * K;
    float sum = 0.f;
    for (int k = 0; k < K; ++k) {
        float best_v = -FLT_MAX;
        int best_i = -1;
        block_argmax(choice, num_experts, -FLT_MAX, buf, ibuf, best_v, best_i);
        const float weight = best_i >= 0 ? probs[best_i] : 0.f;
        if (tid == 0) {
            out_idx[k] = best_i;
            out_w[k] = weight;
            if (best_i >= 0) choice[best_i] = -FLT_MAX;
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

}  // namespace

void topk_sigmoid_bias_bf16(
    const void* logits,
    const float* correction_bias,
    std::int32_t* topk_idx,
    float* topk_w,
    int N,
    int num_experts,
    int K,
    bool normalize,
    float routed_scaling_factor,
    cudaStream_t stream)
{
    if (N <= 0 || num_experts <= 0 || K <= 0) return;
    topk_sigmoid_bias_bf16_kernel<<<N, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits),
        correction_bias,
        topk_idx,
        topk_w,
        num_experts,
        K,
        normalize ? 1 : 0,
        routed_scaling_factor);
}

void topk_sigmoid_bias_fp32(
    const float* logits,
    const float* correction_bias,
    std::int32_t* topk_idx,
    float* topk_w,
    int N,
    int num_experts,
    int K,
    bool normalize,
    float routed_scaling_factor,
    cudaStream_t stream)
{
    if (N <= 0 || num_experts <= 0 || K <= 0) return;
    topk_sigmoid_bias_fp32_kernel<<<N, BLOCK, 0, stream>>>(
        logits,
        correction_bias,
        topk_idx,
        topk_w,
        num_experts,
        K,
        normalize ? 1 : 0,
        routed_scaling_factor);
}

}  // namespace pie_cuda_driver::kernels::moe
