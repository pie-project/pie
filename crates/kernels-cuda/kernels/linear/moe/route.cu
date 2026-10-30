#include "linear/codec.cuh"
#include "linear/act_quant.cuh"
#include "prim/device.cuh"
#ifdef __CUDACC_RTC__

#include "prim/mma_sm80.cuh"
#else

#include <cuda_bf16.h>
#include <mma.h>
#endif
#include "linear/moe_w4.cuh"
#include "linear/quant.cuh"

namespace pie::linear {

// Routes laid out by expert for the grouped GEMM: each routed expert's
// count, its rows' start (`align` rows apart, rounded up, for the contiguous
// layout, `m_max` apart for the masked one) and each route's row.
// Under a live window (`win`: `win[0]` tokens from token `win[1]`, `top`
// routes a token) only its routes are laid out; the rest are neither read
// into the GEMM nor written back.
__device__ __forceinline__ bool route_live(int r, int top, const u32* __restrict__ win)
{
    if (win == nullptr) return true;
    const int t = r / top - static_cast<int>(win[1]);
    return t >= 0 && t < static_cast<int>(win[0]);
}

// Routes counted and placed a block at a time: the block
// tallies its routes' experts in shared memory and adds each expert's tally
// to the global count (or takes its rows from the cursor) once, not once a
// route. A route's row within its expert is its rank in the block's tally
// past the block's take: the same rows, the same counts, in another order
// within an expert. `tally` is `2 * experts` ints of dynamic shared memory.
constexpr int kTallyRoutes = 2048;

__global__ void route_count_tally(
    const i32* __restrict__ routes,
    int count,
    int top,
    const u32* __restrict__ win,
    int experts,
    i32* __restrict__ counts)
{
    extern __shared__ int tally[];
    for (int e = threadIdx.x; e < experts; e += blockDim.x) tally[e] = 0;
    __syncthreads();
    const int r0 = blockIdx.x * kTallyRoutes;
    const int r1 = min(r0 + kTallyRoutes, count);
    for (int r = r0 + threadIdx.x; r < r1; r += blockDim.x) {
        const int e = routes[r];
        if (e >= 0 && e < experts && route_live(r, top, win)) atomicAdd(&tally[e], 1);
    }
    __syncthreads();
    for (int e = threadIdx.x; e < experts; e += blockDim.x)
        if (tally[e] != 0) atomicAdd(&counts[e], tally[e]);
}

__global__ void route_place_tally(
    const i32* __restrict__ routes,
    int count,
    int top,
    const u32* __restrict__ win,
    int experts,
    i32* __restrict__ cursor,
    i32* __restrict__ pos)
{
    extern __shared__ int tally[];
    int* base = tally + experts;
    for (int e = threadIdx.x; e < experts; e += blockDim.x) tally[e] = 0;
    __syncthreads();
    const int r0 = blockIdx.x * kTallyRoutes;
    const int r1 = min(r0 + kTallyRoutes, count);
    constexpr int kPer = kTallyRoutes / 256;
    int rank[kPer];
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int r = r0 + threadIdx.x + i * 256;
        rank[i] = -1;
        if (r < r1) {
            const int e = routes[r];
            if (e >= 0 && e < experts && route_live(r, top, win)) rank[i] = atomicAdd(&tally[e], 1);
        }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < experts; e += blockDim.x)
        base[e] = tally[e] != 0 ? atomicAdd(&cursor[e], tally[e]) : 0;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int r = r0 + threadIdx.x + i * 256;
        if (r < r1) pos[r] = rank[i] >= 0 ? base[routes[r]] + rank[i] : -1;
    }
}

// The experts' starts across a block: each thread sums a run of consecutive
// experts' rows, the block scans the runs, and each thread writes its run's
// starts. The same integers, without one thread walking every expert.
constexpr int kOffsetThreads = 1024;

// `route_count_tally`, `route_offsets_scan` and `route_place_tally` in one
// block, for a fire of at most `kOffsetThreads * kSmallPer` routes: the
// counts tallied in shared memory, scanned into the experts' starts, each
// route placed at its start past its rank. `shared` is `2 * experts` ints.
constexpr int kSmallPer = 16;

__global__ void __launch_bounds__(kOffsetThreads) route_lay_out_small(
    const i32* __restrict__ routes,
    int count,
    int top,
    const u32* __restrict__ win,
    int experts,
    int m_max,
    int align,
    i32* __restrict__ counts,
    i32* __restrict__ offsets,
    i32* __restrict__ pos,
    const float* __restrict__ input_scales,
    const float* __restrict__ tensor_scales,
    float* __restrict__ alpha,
    int alpha_rows,
    const float* __restrict__ input_scales2,
    const float* __restrict__ tensor_scales2,
    float* __restrict__ alpha2,
    int alpha_rows2)
{
    extern __shared__ int lay[];
    int* tally = lay;
    int* start = lay + experts;
    __shared__ int run_sum[kOffsetThreads];
    const int tid = static_cast<int>(threadIdx.x);
    for (int e = tid; e < experts; e += kOffsetThreads) tally[e] = 0;
    __syncthreads();
    int rank[kSmallPer];
#pragma unroll
    for (int i = 0; i < kSmallPer; ++i) {
        const int r = tid + i * kOffsetThreads;
        rank[i] = -1;
        if (r < count) {
            const int e = routes[r];
            if (e >= 0 && e < experts && route_live(r, top, win)) rank[i] = atomicAdd(&tally[e], 1);
        }
    }
    __syncthreads();
    const int per = (experts + kOffsetThreads - 1) / kOffsetThreads;
    const int lo = min(tid * per, experts), hi = min(lo + per, experts);
    auto rows = [&](int e) { return m_max > 0 ? m_max : (tally[e] + align - 1) / align * align; };
    int sum = 0;
    for (int e = lo; e < hi; ++e) sum += rows(e);
    // The runs scanned a warp at a time by shuffles, then the warps' totals
    // by the first warp: two barriers, not two a doubling step.
    const int lane = tid & 31, warp = tid >> 5;
    int incl = sum;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, incl, off);
        if (lane >= off) incl += y;
    }
    if (lane == 31) run_sum[warp] = incl;
    __syncthreads();
    if (warp == 0) {
        int w = run_sum[lane];
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, w, off);
            if (lane >= off) w += y;
        }
        run_sum[lane] = w;
    }
    __syncthreads();
    int at = (warp > 0 ? run_sum[warp - 1] : 0) + incl - sum;
    for (int e = lo; e < hi; ++e) {
        start[e] = at;
        offsets[e] = at;
        counts[e] = tally[e];
        at += rows(e);
    }
    if (tid == kOffsetThreads - 1) offsets[experts] = run_sum[31];
    // NVFP4's per-expert product scale (`expert_alpha`), of each leg asked.
    if (alpha != nullptr) {
        for (int e = tid; e < experts; e += kOffsetThreads)
            alpha[e] = input_scales[e] * tensor_scales[static_cast<long long>(e) * alpha_rows];
    }
    if (alpha2 != nullptr) {
        for (int e = tid; e < experts; e += kOffsetThreads)
            alpha2[e] = input_scales2[e] * tensor_scales2[static_cast<long long>(e) * alpha_rows2];
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kSmallPer; ++i) {
        const int r = tid + i * kOffsetThreads;
        if (r < count) pos[r] = rank[i] >= 0 ? start[routes[r]] + rank[i] : -1;
    }
}

// `route_lay_out_small` and the NVFP4 quantize of a narrow W4A4 fire (at
// most 1024 routes over 512 experts) in one launch: every block lays the
// routes out alike -- a route's rank its 32-route chunk's matches ahead of it
// past the chunks before -- so each lands the same rows; block 0 lands the
// layout and the experts' alphas, and each block quantizes its share of the
// routes' 16-wide blocks to their rows. Dynamic shared memory: `33 * experts`
// ints for the counts and starts, `count` for the rows.
__global__ void __launch_bounds__(256) route_lay_out_quant_nvfp4(
    const i32* __restrict__ routes,
    int count,
    int top,
    const u32* __restrict__ win,
    int experts,
    int align,
    i32* __restrict__ counts,
    i32* __restrict__ offsets,
    i32* __restrict__ pos,
    const float* __restrict__ tensor_scales,
    float* __restrict__ alpha,
    int alpha_rows,
    const float* __restrict__ tensor_scales2,
    const float* __restrict__ input_scales2,
    float* __restrict__ alpha2,
    int alpha_rows2,
    const bf16* __restrict__ x,
    const float* __restrict__ input_scales,
    u8* __restrict__ codes,
    u8* __restrict__ atoms,
    int k)
{
    // A programmatic dependent (`kernel.pdl`) reads nothing until its
    // producer is done; its consumer (the gate-up GEMM) may launch from here.
    pdl_wait();
    pdl_launch();
    extern __shared__ int lay[];
    int* cnt = lay;                     // [32 chunks][experts]
    int* start = lay + 32 * experts;    // [experts]
    int* row = start + experts;         // [count]
    __shared__ int run_sum[8];
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31, warp = tid >> 5;
    const int chunks = (count + 31) / 32;
    for (int i = tid; i < chunks * experts; i += 256) cnt[i] = 0;
    __syncthreads();
    int rank[4], ex[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c = warp + 8 * i;
        const int r = c * 32 + lane;
        int e = -1;
        if (r < count) {
            e = routes[r];
            if (e < 0 || e >= experts || !route_live(r, top, win)) e = -1;
        }
        const unsigned m = __match_any_sync(0xffffffffu, e);
        rank[i] = __popc(m & ((1u << lane) - 1u));
        ex[i] = e;
        if (c < chunks && e >= 0 && rank[i] == 0) cnt[c * experts + e] = __popc(m);
    }
    __syncthreads();
    // Each expert's chunks prefixed in place, its rows (two experts a
    // thread, experts <= 512).
    constexpr int kPer = 2;
    int rows_e[kPer], tally[kPer], sum = 0;
#pragma unroll
    for (int p = 0; p < kPer; ++p) {
        const int e = tid * kPer + p;
        tally[p] = 0;
        rows_e[p] = 0;
        if (e >= experts) continue;
        for (int c = 0; c < chunks; ++c) {
            const int v = cnt[c * experts + e];
            cnt[c * experts + e] = tally[p];
            tally[p] += v;
        }
        rows_e[p] = (tally[p] + align - 1) / align * align;
        sum += rows_e[p];
    }
    // The experts' rows scanned.
    int incl = sum;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, incl, off);
        if (lane >= off) incl += y;
    }
    if (lane == 31) run_sum[warp] = incl;
    __syncthreads();
    int before = 0;
    for (int w = 0; w < warp; ++w) before += run_sum[w];
    int at_e = before + incl - sum;
#pragma unroll
    for (int p = 0; p < kPer; ++p) {
        const int e = tid * kPer + p;
        if (e >= experts) continue;
        start[e] = at_e;
        if (blockIdx.x == 0) {
            offsets[e] = at_e;
            counts[e] = tally[p];
            if (alpha != nullptr)
                alpha[e] = input_scales[e] * tensor_scales[static_cast<long long>(e) * alpha_rows];
            if (alpha2 != nullptr)
                alpha2[e] = input_scales2[e] * tensor_scales2[static_cast<long long>(e) * alpha_rows2];
        }
        at_e += rows_e[p];
    }
    if (blockIdx.x == 0 && tid == 255) {
        int total = 0;
        for (int w = 0; w < 8; ++w) total += run_sum[w];
        offsets[experts] = total;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c = warp + 8 * i;
        const int r = c * 32 + lane;
        if (r < count) {
            const int o = ex[i] >= 0 ? start[ex[i]] + cnt[c * experts + ex[i]] + rank[i] : -1;
            row[r] = o;
            if (blockIdx.x == 0) pos[r] = o;
        }
    }
    __syncthreads();
    // A thread a route's 16-wide block, under its expert's global.
    const int blocks = k / 16;
    for (long long at = static_cast<long long>(blockIdx.x) * 256 + tid; at < static_cast<long long>(count) * blocks;
         at += static_cast<long long>(gridDim.x) * 256) {
        const int r = static_cast<int>(at / blocks), b = static_cast<int>(at % blocks);
        const int o = row[r];
        if (o < 0) continue;
        float v[16];
        const float amax = load_block16(x + static_cast<long long>(r / top) * k + b * 16, v);
        uint2 w;
        atoms[atom_offset(o, b, blocks)] = nvfp4_block16(v, amax, __fdiv_rn(1.f, input_scales[routes[r]]), w);
        *reinterpret_cast<uint2*>(codes + static_cast<long long>(o) * (k / 2) + b * 8) = w;
    }
}

__global__ void __launch_bounds__(kOffsetThreads) route_offsets_scan(
    const i32* __restrict__ counts,
    int experts,
    int m_max,
    int align,
    i32* __restrict__ offsets,
    i32* __restrict__ cursor)
{
    __shared__ int run_sum[kOffsetThreads];
    const int tid = static_cast<int>(threadIdx.x);
    const int per = (experts + kOffsetThreads - 1) / kOffsetThreads;
    const int lo = min(tid * per, experts), hi = min(lo + per, experts);
    auto rows = [&](int e) {
        return m_max > 0 ? m_max : (counts[e] + align - 1) / align * align;
    };
    int sum = 0;
    for (int e = lo; e < hi; ++e) sum += rows(e);
    run_sum[tid] = sum;
    __syncthreads();
    // Inclusive scan over the runs.
    for (int step = 1; step < kOffsetThreads; step <<= 1) {
        const int add = tid >= step ? run_sum[tid - step] : 0;
        __syncthreads();
        run_sum[tid] += add;
        __syncthreads();
    }
    int at = run_sum[tid] - sum;
    for (int e = lo; e < hi; ++e) {
        offsets[e] = at;
        cursor[e] = at;
        at += rows(e);
    }
    if (tid == kOffsetThreads - 1) offsets[experts] = run_sum[tid];
}

// `y[r] = grouped[pos[r]]`, a dropped route zeros, a route past the live
// window untouched; a thread a pair.
__global__ void route_gather(
    const bf16* __restrict__ grouped,
    const i32* __restrict__ pos,
    bf16* __restrict__ y,
    int count,
    int n,
    int top,
    const u32* __restrict__ win)
{
    const long long at = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= static_cast<long long>(count) * (n / 2)) return;
    const int r = static_cast<int>(at / (n / 2));
    if (!route_live(r, top, win)) return;
    const int c = static_cast<int>(at % (n / 2)) * 2;
    const int o = pos[r];
    bf16x2* dst = reinterpret_cast<bf16x2*>(y + static_cast<long long>(r) * n + c);
    if (o < 0) {
        *dst = f32_to_bf16x2(0.f, 0.f);
        return;
    }
    *dst = *reinterpret_cast<const bf16x2*>(grouped + static_cast<long long>(o) * n + c);
}

// `route_gather`, 16 bytes a thread: the same rows, written whole.
__global__ void route_gather_vec8(
    const bf16* __restrict__ grouped,
    const i32* __restrict__ pos,
    bf16* __restrict__ y,
    int count,
    int n,
    int top,
    const u32* __restrict__ win)
{
    const int runs = n / 8;
    const long long at = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= static_cast<long long>(count) * runs) return;
    const int r = static_cast<int>(at / runs);
    if (!route_live(r, top, win)) return;
    const int c = static_cast<int>(at % runs) * 8;
    const int o = pos[r];
    uint4* dst = reinterpret_cast<uint4*>(y + static_cast<long long>(r) * n + c);
    if (o < 0) {
        *dst = make_uint4(0u, 0u, 0u, 0u);
        return;
    }
    *dst = *reinterpret_cast<const uint4*>(grouped + static_cast<long long>(o) * n + c);
}

// Each expert's product scale: its activations' static global times its
// weight's tensor scale, the first row's standing for the expert's (vLLM
// reads the gate leg's for both of a gated bank's legs).
__global__ void expert_alpha(
    const float* __restrict__ input_scales,
    const float* __restrict__ tensor_scales,
    float* __restrict__ alpha,
    int experts,
    int rows)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= experts) return;
    alpha[e] = input_scales[e] * tensor_scales[static_cast<long long>(e) * rows];
}

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

template <class T, bool FusedGemv>
__device__ __forceinline__ void moe_topk_softmax_body(
    const T* __restrict__ logits,
    const T* __restrict__ act,
    const T* __restrict__ bias,
    i32* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    int num_experts, int K, int hidden,
    const u32* __restrict__ win,
    const T* __restrict__ per_expert_scale = nullptr)
{
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int plane_row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int tid = threadIdx.x;
    const T* row =
        FusedGemv ? logits
                  : logits + static_cast<long long>(plane_row) * num_experts;

    __shared__ float probs[kRouterMaxExperts];
    __shared__ float buf[kSoftmaxBlock];
    __shared__ int ibuf[kSoftmaxBlock];

    float local_max = -flt_max();
    if constexpr (FusedGemv) {
        const T* x = act + static_cast<long long>(plane_row) * hidden;
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

    i32* out_idx = topk_idx + static_cast<long long>(plane_row) * K;
    float* out_w = topk_w + static_cast<long long>(plane_row) * K;
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
    int num_experts, int K, int hidden,
    const u32* __restrict__ win)
{
    moe_topk_softmax_body<T, false>(logits, act, bias, topk_idx, topk_w,
                                num_experts, K, hidden, win);
}

template <class T>
__global__ void moe_topk_softmax_scaled(
    const T* __restrict__ logits,
    const T* __restrict__ act,
    const T* __restrict__ per_expert_scale,
    i32* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    int num_experts, int K, int hidden,
    const u32* __restrict__ win)
{
    moe_topk_softmax_body<T, false>(logits, act, nullptr, topk_idx, topk_w,
                                num_experts, K, hidden, win, per_expert_scale);
}

// A row's top `K` (at most 16) of its `num_experts` (at most `32 kPerLane`)
// logits and their softmax over those `K`, a warp a row: the full-softmax
// arm's weights (the top `K` of the softmax, renormalized, are the softmax
// of the top `K` logits) without its block reductions, under the fire's live
// window, each weight times `per_expert_scale` where it is given.
template <class T, int kPerLane>
__global__ void __launch_bounds__(32) moe_topk_softmax_lane(
    const T* __restrict__ logits,
    i32* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    int num_experts,
    int K,
    const u32* __restrict__ win,
    const T* __restrict__ per_expert_scale)
{
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    const int n = blockIdx.x;
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int row_at = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int lane = static_cast<int>(threadIdx.x);
    const T* row = logits + static_cast<long long>(row_at) * num_experts;
    float val[kPerLane];
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
        const int e = lane + i * 32;
        val[i] = e < num_experts ? Logit<T>::to_f32(row[e]) : -flt_max();
    }
    float best_w[16];
    int best_e[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        if (k >= K) break;
        float bv = -flt_max();
        int bi = 0x7fffffff;
#pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
            const int e = lane + i * 32;
            if (val[i] > bv) {
                bv = val[i];
                bi = e;
            }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
#pragma unroll
        for (int i = 0; i < kPerLane; ++i)
            if (lane + i * 32 == bi) val[i] = -flt_max();
        best_w[k] = bv;
        best_e[k] = bi < num_experts ? bi : -1;
    }
    if (lane == 0) {
        i32* out_idx = topk_idx + static_cast<long long>(row_at) * K;
        float* out_w = topk_w + static_cast<long long>(row_at) * K;
        float w_sum = 0.f;
        const float row_max = best_w[0];
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            if (k >= K) break;
            best_w[k] = best_e[k] >= 0 ? expf(best_w[k] - row_max) : 0.f;
            w_sum += best_w[k];
        }
        const float inv_w = 1.f / w_sum;
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            if (k >= K) break;
            float w = best_w[k] * inv_w;
            if (per_expert_scale != nullptr && best_e[k] >= 0) w *= Logit<T>::to_f32(per_expert_scale[best_e[k]]);
            out_idx[k] = best_e[k];
            out_w[k] = w;
        }
    }
}

template <class T>
__global__ void moe_topk_sigmoid_sink(
    const T* __restrict__ logits,
    i32* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    const float* __restrict__ correction_bias,
    const float* __restrict__ global_scale,
    int E,
    int S,
    int K,
    float scaling,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int plane_row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int tid = threadIdx.x;
    const int W = E + S;
    const T* row = logits + static_cast<long long>(plane_row) * W;
    __shared__ float scores[kRouterMaxExperts];
    __shared__ float orig_scores[kRouterMaxExperts];
    __shared__ bool taken[kRouterMaxExperts];

    for (int e = tid; e < W; e += blockDim.x) {
        const float x = Elem<T>::to_f32(row[e]);
        const float s = 1.f / (1.f + expf(-x));
        orig_scores[e] = s;
        scores[e] = (e < E && correction_bias != nullptr) ? s + correction_bias[e] : s;
        taken[e] = false;
    }
    __syncthreads();

    if (tid == 0) {
        i32* idx = topk_idx + static_cast<long long>(plane_row) * (K + S);
        float* w = topk_w + static_cast<long long>(plane_row) * (K + S);
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
            idx[k] = -1;
            w[k] = 0.f;
        }
        for (int s = 0; s < S; ++s) {
            idx[K + s] = E + s;
            w[K + s] = orig_scores[E + s];
            sum += orig_scores[E + s];
        }
        const float g = global_scale != nullptr ? global_scale[0] : 1.f;
        const float scale = scaling * g / (sum + 1e-20f);
        for (int k = 0; k < K + S; ++k) w[k] *= scale;
    }
}

__device__ __forceinline__ float sqrt_softplus(float x) {

    const float sp = x > 20.f ? x : log1pf(expf(x));
    return sqrtf(fmaxf(sp, 0.f));
}

// The ranked routers, a warp a row: the sigmoid (`kSoftplus` false) or
// DeepSeek-V4's sqrt-softplus of each logit, `K` picks by the biased score
// (the largest, the lowest expert on a tie), weighted by the unbiased one,
// renormalized and scaled. Lane `l` holds experts `l + 32 m` as
// order-preserving keys, `kPer` of them; a pick is two warp reductions. A
// score below -FLT_MAX never wins. Picks past the experts land expert 0 at
// weight 0 (sigmoid) or -1 (sqrt-softplus).
constexpr int kRankedRows = 4;

template <class T, bool kSoftplus, int kPer>
__global__ void __launch_bounds__(32 * kRankedRows) moe_topk_ranked(
    const T* __restrict__ logits,
    i32* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    const float* __restrict__ correction_bias,
    int rows,
    int E,
    int K,
    bool renormalize,
    float routed_scaling_factor,
    const u32* __restrict__ win)
{
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    const int n = static_cast<int>(blockIdx.x) * kRankedRows + static_cast<int>(threadIdx.x >> 5);
    if (n >= rows) return;
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int plane_row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int tid = static_cast<int>(threadIdx.x & 31);
    const T* row = logits + static_cast<long long>(plane_row) * E;

    float orig[kPer];
    unsigned key[kPer];
    // Every logit and bias out before any is scored: a lane's loads paid
    // once, not once an expert.
    float raw[kPer], bias[kPer];
#pragma unroll
    for (int m = 0; m < kPer; ++m) {
        const int e = tid + 32 * m;
        raw[m] = e < E ? Elem<T>::to_f32(row[e]) : 0.f;
        bias[m] = e < E && correction_bias != nullptr ? correction_bias[e] : 0.f;
    }
#pragma unroll
    for (int m = 0; m < kPer; ++m) {
        const int e = tid + 32 * m;
        key[m] = 0u;
        orig[m] = 0.f;
        if (e < E) {
            const float x = raw[m];
            const float sc = kSoftplus ? sqrt_softplus(x) : 1.f / (1.f + expf(-x));
            orig[m] = sc;
            const float v = correction_bias != nullptr ? sc + bias[m] : sc;
            if (v > -flt_max()) {
                const unsigned u = __float_as_uint(v);
                key[m] = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
            }
        }
    }
    i32* idx = topk_idx + static_cast<long long>(plane_row) * K;
    float* w = topk_w + static_cast<long long>(plane_row) * K;
    int my_e = -1;
    float my_w = 0.f;
    float sum = 0.f;
    for (int k = 0; k < K; ++k) {
        unsigned mine = 0u;
        int at = 0x7fffffff;
        float o_at = 0.f;
#pragma unroll
        for (int m = 0; m < kPer; ++m) {
            if (key[m] > mine) {
                mine = key[m];
                at = tid + 32 * m;
                o_at = orig[m];
            }
        }
        const unsigned top = __reduce_max_sync(0xffffffffu, mine);
        const int i = top == 0u ? -1
                                : static_cast<int>(__reduce_min_sync(
                                      0xffffffffu, mine == top ? static_cast<unsigned>(at) : 0x7fffffffu));
#pragma unroll
        for (int m = 0; m < kPer; ++m) {
            if (tid + 32 * m == i) key[m] = 0u;
        }
        if (i >= 0) {
            const float o = __shfl_sync(0xffffffffu, o_at, i & 31);
            sum += o;
            if (tid == (k & 31)) {
                my_e = i;
                my_w = o;
            }
        } else if (tid == (k & 31)) {
            my_e = kSoftplus ? -1 : 0;
            my_w = 0.f;
        }
        // More than 32 picks: the lane's first round lands before its
        // second is kept.
        if ((k & 31) == 31 || k == K - 1) {
            const int base = k & ~31;
            if (tid <= (k & 31)) {
                idx[base + tid] = my_e;
                if (!kSoftplus || my_e >= 0) w[base + tid] = my_w;
            }
        }
    }
    __syncwarp();
    const float scale = renormalize && sum > 0.f ? routed_scaling_factor / sum : routed_scaling_factor;
    for (int k = tid; k < K; k += 32) w[k] *= scale;
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
__global__ void build_moe_leg_ptrs(
    const i32* __restrict__ expert_ids,
    const T* __restrict__ bank_base,
    const T* __restrict__ aligned_in,
    T* __restrict__ aligned_out,
    const T** __restrict__ w_ptrs,
    const T** __restrict__ act_ptrs,
    T** __restrict__ out_ptrs,
    int max_blocks,
    int block_size,
    int K,
    int N)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= max_blocks) return;
    const int e = expert_ids[b];
    const long long row = static_cast<long long>(b) * block_size;
    w_ptrs[b] = bank_base + static_cast<long long>(e < 0 ? 0 : e)
                * static_cast<long long>(N) * static_cast<long long>(K);
    act_ptrs[b] = aligned_in + row * static_cast<long long>(K);
    out_ptrs[b] = aligned_out + row * static_cast<long long>(N);
}

__device__ __forceinline__ float hash_sqrt_softplus(float x) {
    const float sp = x > 20.f ? x : log1pf(expf(x));
    return sqrtf(fmaxf(sp, 0.f));
}

__global__ void hash_route_gather(
    const i32* __restrict__ token_ids,
    const i64* __restrict__ tid2eid,
    const bf16* __restrict__ logits,
    i32* __restrict__ routes,
    float* __restrict__ weights,
    int tokens,
    int vocab,
    int n_experts,
    int top_k,
    int renormalize,
    float scaling,
    const u32* __restrict__ win)
{
    const int n = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                + static_cast<int>(threadIdx.x);
    if (n >= tokens) return;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int plane_row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const i32 raw = token_ids[plane_row];
    const int tid = (raw >= 0 && raw < vocab) ? raw : 0;
    const i64* picks = tid2eid + static_cast<long long>(tid) * top_k;
    const bf16* score = logits + static_cast<long long>(plane_row) * n_experts;
    i32* ids = routes + static_cast<long long>(plane_row) * top_k;
    float* ws = weights + static_cast<long long>(plane_row) * top_k;
    float sum = 0.f;
    for (int r = 0; r < top_k; ++r) {
        const i64 e = picks[r];
        const float w = (e >= 0 && e < static_cast<i64>(n_experts))
            ? hash_sqrt_softplus(bf16_to_f32(score[static_cast<int>(e)])) : 0.f;
        ids[r] = static_cast<i32>(e);
        ws[r] = w;
        sum += w;
    }
    const float scale = (renormalize != 0 && sum > 0.f) ? scaling / sum : scaling;
    for (int r = 0; r < top_k; ++r) ws[r] *= scale;
}

__global__ void group_routes(
    i32* __restrict__ routes,
    int tokens,
    int groups,
    const u32* __restrict__ win)
{
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                  + static_cast<int>(threadIdx.x);
    if (idx >= tokens * groups) return;
    const int n = idx / groups;
    const int slot = idx - n * groups;
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int plane_row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    routes[static_cast<long long>(plane_row) * groups + slot] = slot;
}

/// `y[route] = x[route / act_div] . W[expert]^T (+ bias[expert])` for the
/// routes `order` lists by expert, `work` naming (expert, slice of
/// `slice_routes` routes) per CTA (from `moe_route_order`). A CTA covers 128
/// weight rows; `kNb` 8-route blocks a slice (`slice_routes` = 8 kNb).
/// With `kGlu`, the bank's `n` rows are `inter` gate rows and as many up
/// rows: a warp's two tiles are 16 gate rows and their up rows, and `out`
/// is the GLU, `inter` wide (a CTA 64 of it).
/// A block-wide exclusive scan of one value a thread (blockDim.x a whole
/// number of warps, at most 1024); the total lands in `*total`.
__device__ __forceinline__ int w4_block_scan(int v, int* warp_sums, int* total) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int warps = static_cast<int>(blockDim.x) >> 5;
    int x = v;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, off);
        if (lane >= off) x += y;
    }
    if (lane == 31) warp_sums[warp] = x;
    __syncthreads();
    if (warp == 0) {
        int w = lane < warps ? warp_sums[lane] : 0;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, w, off);
            if (lane >= off) w += y;
        }
        if (lane < warps) warp_sums[lane] = w;
        if (lane == 31) *total = w;
    }
    __syncthreads();
    const int before = warp > 0 ? warp_sums[warp - 1] : 0;
    return before + x - v;
}

/// `moe_route_order`'s product (routes sorted by expert, each expert's
/// offset, the work list of (expert, slice) items in expert order) with the
/// two serial walks over the experts done as block scans; `offsets` holds
/// `num_experts + 3`: the experts' starts and the dropped routes' (those
/// naming no expert), the work items' count, and the live routes' count.
__global__ void moe_route_order_scan(
    const i32* __restrict__ routes,
    i32* __restrict__ order,
    i32* __restrict__ offsets,
    i32* __restrict__ work,
    int work_cap,
    int group_routes,
    int top_k,
    int route_count,
    int num_experts,
    const u32* __restrict__ win)
{
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    extern __shared__ i32 order_smem[];
    i32* counts = order_smem;
    i32* fill = counts + num_experts + 1;
    __shared__ int warp_sums[32];
    __shared__ int total;
    const int live = win != nullptr
        ? min(static_cast<int>(win[0]) * top_k, route_count)
        : route_count;
    const int base = win != nullptr ? static_cast<int>(win[1]) * top_k : 0;
    const int tid = static_cast<int>(threadIdx.x);
    const int threads = static_cast<int>(blockDim.x);
    for (int e = tid; e <= num_experts; e += threads) {
        counts[e] = 0;
        fill[e] = 0;
    }
    __syncthreads();
    // A warp's routes to one expert add once (a 32-expert bank's 22 routes a token
    // contended on 33 counters, 121 us a 8192-token fire).
    for (int r = tid; r < live; r += threads) {
        int e = routes[base + r];
        if (e < 0 || e >= num_experts) e = num_experts;
        const unsigned peers = __match_any_sync(__activemask(), e);
        if ((tid & 31) == __ffs(peers) - 1) atomicAdd(counts + e, __popc(peers));
    }
    __syncthreads();
    // Each thread a contiguous run of experts: their routes and work items.
    const int per = (num_experts + 1 + threads - 1) / threads;
    const int e0 = min(tid * per, num_experts + 1);
    const int e1 = min(e0 + per, num_experts + 1);
    int mine = 0, items = 0;
    for (int e = e0; e < e1; ++e) {
        const int c = counts[e];
        mine += c;
        if (e < num_experts) items += (c + group_routes - 1) / group_routes;
    }
    int route_at = w4_block_scan(mine, warp_sums, &total);
    __syncthreads();
    int item_at = w4_block_scan(items, warp_sums, &total);
    const int items_total = total;
    __syncthreads();
    for (int e = e0; e < e1; ++e) {
        const int c = counts[e];
        counts[e] = route_at;
        offsets[e] = route_at;
        route_at += c;
        if (e < num_experts) {
            for (int g = 0; g * group_routes < c; ++g, ++item_at)
                if (item_at < work_cap) work[item_at] = e * 65536 + g;
        }
    }
    if (tid == 0) {
        offsets[num_experts + 1] = min(items_total, work_cap);
        // The dropped routes' bucket ends at the live routes' count.
        offsets[num_experts + 2] = live;
    }
    __syncthreads();
    for (int r = tid; r < live; r += threads) {
        int e = routes[base + r];
        if (e < 0 || e >= num_experts) e = num_experts;
        const unsigned peers = __match_any_sync(__activemask(), e);
        const int leader = __ffs(peers) - 1;
        int at = 0;
        if ((tid & 31) == leader) at = atomicAdd(fill + e, __popc(peers));
        at = __shfl_sync(peers, at, leader);
        const int pos = counts[e] + at + __popc(peers & ((1u << (tid & 31)) - 1u));
        order[pos] = base + r;
    }
}

__global__ void moe_route_order(
    const i32* __restrict__ routes,
    i32* __restrict__ order,
    i32* __restrict__ offsets,
    i32* __restrict__ work,
    int work_cap,
    int group_routes,
    int top_k,
    int route_count,
    int num_experts,
    const u32* __restrict__ win)
{
    extern __shared__ i32 order_smem[];
    i32* counts = order_smem;
    i32* fill = counts + num_experts + 1;
    const int live = win != nullptr
        ? min(static_cast<int>(win[0]) * top_k, route_count)
        : route_count;
    const int base = win != nullptr ? static_cast<int>(win[1]) * top_k : 0;
    for (int e = threadIdx.x; e <= num_experts; e += blockDim.x) {
        counts[e] = 0;
        fill[e] = 0;
    }
    __syncthreads();
    for (int r = threadIdx.x; r < live; r += blockDim.x) {
        int e = routes[base + r];
        if (e < 0 || e >= num_experts) e = num_experts;
        atomicAdd(counts + e, 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int running = 0;
        for (int e = 0; e <= num_experts; ++e) {
            const int c = counts[e];
            counts[e] = running;
            running += c;
        }
    }
    __syncthreads();

    for (int e = threadIdx.x; e <= num_experts; e += blockDim.x) offsets[e] = counts[e];
    for (int r = threadIdx.x; r < live; r += blockDim.x) {
        int e = routes[base + r];
        if (e < 0 || e >= num_experts) e = num_experts;
        const int pos = counts[e] + atomicAdd(fill + e, 1);
        order[pos] = base + r;
    }

    if (threadIdx.x == 0) {
        int w = 0;
        for (int e = 0; e < num_experts && w < work_cap; ++e) {
            const int c = counts[e + 1] - counts[e];
            for (int g = 0; g * group_routes < c && w < work_cap; ++g) work[w++] = e * 65536 + g;
        }
        offsets[num_experts + 1] = w;
    }
}

}
