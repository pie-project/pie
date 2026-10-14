#pragma once

// Mixture of experts: the routers (top-k over a token's expert scores), the
// dense expert matmuls (linear/moe_kit.hip), and the folds of the routed
// rows back into token rows. A route is (token, rank): route r of a fire is
// token r / top_k's pick r % top_k, and a window states live tokens.

#include "linear/moe_kit.hip"

namespace pie::linear {

// ------------------------------------------------------------- routers --

// A router holds a token's scores in registers, kRouterSlots a lane (score
// j in lane j % kWave, slot j / kWave), and one wavefront picks.
constexpr int kRouterMax = 512;
constexpr int kRouterSlots = kRouterMax / kWave;

struct Ranked {
    float rank[kRouterSlots];  // what orders the picks
    float keep[kRouterSlots];  // what a pick weighs
    u32 out;                   // slots out of the running: picked, or past the row
};

// (v, i) beats (bv, bi): a present candidate over none, a larger value, an
// equal value at a lower index.
__device__ __forceinline__ bool router_beats(float v, int i, float bv, int bi) {
    return i >= 0 && (bi < 0 || v > bv || (v == bv && i < bi));
}

// The best score still running, its weight, and takes it out of the running;
// -1 once none is left.
__device__ __forceinline__ int router_pick(Ranked& s, float& weight) {
    const int l = lane();
    float bv = 0.f, bw = 0.f;
    int bi = -1;
#pragma unroll
    for (int t = 0; t < kRouterSlots; ++t) {
        const int i = (s.out >> t) & 1u ? -1 : l + kWave * t;
        if (router_beats(s.rank[t], i, bv, bi)) {
            bv = s.rank[t];
            bw = s.keep[t];
            bi = i;
        }
    }
#pragma unroll
    for (int off = kWave / 2; off > 0; off >>= 1) {
        const float ov = __shfl_xor(bv, off, kWave), ow = __shfl_xor(bw, off, kWave);
        const int oi = __shfl_xor(bi, off, kWave);
        if (router_beats(ov, oi, bv, bi)) {
            bv = ov;
            bw = ow;
            bi = oi;
        }
    }
    if (bi >= 0 && bi % kWave == l) s.out |= 1u << (bi / kWave);
    weight = bw;
    return bi;
}

// Row `n` of a windowed router launch, or -1 when the window has it dead.
__device__ __forceinline__ long long router_row(const u32* win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return -1;
    return window.plane(blockIdx.x);
}

// Loads a row's `E` scores through f(x) -> (rank, keep).
template <class T, class F>
__device__ __forceinline__ Ranked router_load(const T* __restrict__ row, int E, F&& f) {
    Ranked s;
    s.out = 0;
    const int l = lane();
#pragma unroll
    for (int t = 0; t < kRouterSlots; ++t) {
        const int e = l + kWave * t;
        s.rank[t] = 0.f;
        s.keep[t] = 0.f;
        if (e < E) {
            f(e, to_f32(row[e]), s.rank[t], s.keep[t]);
        } else {
            s.out |= 1u << t;
        }
    }
    return s;
}

// softmax over the row, the top K by probability, renormalized over the
// picks (and scaled by `gain[expert]` when given). One wavefront a row.
template <class T>
__device__ __forceinline__ void topk_softmax_row(const T* __restrict__ logits, const T* __restrict__ gain,
                                                 i32* __restrict__ topk_idx, float* __restrict__ topk_w, int E, int K,
                                                 const u32* __restrict__ win) {
    if (threadIdx.x >= kWave) return;
    const long long row = router_row(win);
    if (row < 0) return;
    Ranked s = router_load(logits + row * E, E, [](int, float x, float& r, float& k) { r = k = x; });
    float m = neg_inf();
#pragma unroll
    for (int t = 0; t < kRouterSlots; ++t)
        if (!((s.out >> t) & 1u)) m = fmaxf(m, s.rank[t]);
    m = wave_max(m);
    float z = 0.f;
#pragma unroll
    for (int t = 0; t < kRouterSlots; ++t) {
        if (!((s.out >> t) & 1u)) {
            s.rank[t] = expf(s.rank[t] - m);
            z += s.rank[t];
        }
    }
    const float inv_z = 1.f / wave_sum(z);
#pragma unroll
    for (int t = 0; t < kRouterSlots; ++t) s.keep[t] = s.rank[t] = s.rank[t] * inv_z;

    const int l = lane();
    int mine = -1;
    float mine_w = 0.f, sum = 0.f;
    for (int k = 0; k < K; ++k) {
        float w;
        const int e = router_pick(s, w);
        if (e < 0) w = 0.f;
        sum += w;
        if (l == k) {
            mine = e;
            mine_w = w;
        }
    }
    if (l < K) {
        float w = mine_w * (1.f / sum);
        if (gain != nullptr && mine >= 0) w *= to_f32(gain[mine]);
        topk_idx[row * K + l] = mine;
        topk_w[row * K + l] = w;
    }
}

template <class T>
__global__ __launch_bounds__(64) void moe_topk_softmax(const T* __restrict__ logits, const T* __restrict__,
                                                       const T* __restrict__, i32* __restrict__ topk_idx,
                                                       float* __restrict__ topk_w, int num_experts, int K, int,
                                                       const u32* __restrict__ win) {
    topk_softmax_row<T>(logits, nullptr, topk_idx, topk_w, num_experts, K, win);
}

template <class T>
__global__ __launch_bounds__(64) void moe_topk_softmax_scaled(const T* __restrict__ logits, const T* __restrict__,
                                                              const T* __restrict__ per_expert_scale,
                                                              i32* __restrict__ topk_idx, float* __restrict__ topk_w,
                                                              int num_experts, int K, int,
                                                              const u32* __restrict__ win) {
    topk_softmax_row<T>(logits, per_expert_scale, topk_idx, topk_w, num_experts, K, win);
}

// The top K by score + bias (no bias: by score), weighed by the score;
// renormalized to `scaling` over the picks when asked, else scaled by it.
// Picks past E land expert 0 at weight 0.
template <class T, class F>
__device__ __forceinline__ void topk_biased_row(const T* __restrict__ logits, i32* __restrict__ topk_idx,
                                                float* __restrict__ topk_w, const float* __restrict__ bias, int E,
                                                int K, bool renormalize, float scaling, const u32* __restrict__ win,
                                                F&& score) {
    if (threadIdx.x >= kWave) return;
    const long long row = router_row(win);
    if (row < 0) return;
    Ranked s = router_load(logits + row * E, E, [&](int e, float x, float& r, float& k) {
        k = score(x);
        r = bias != nullptr ? k + bias[e] : k;
    });
    const int l = lane();
    int mine = 0;
    float mine_w = 0.f, sum = 0.f;
    const int picks = min(K, E);
    for (int k = 0; k < picks; ++k) {
        float w;
        const int e = router_pick(s, w);
        sum += w;
        if (l == k) {
            mine = e;
            mine_w = w;
        }
    }
    if (l < K) {
        const float scale = renormalize && sum > 0.f ? scaling / sum : scaling;
        topk_idx[row * K + l] = mine;
        topk_w[row * K + l] = mine_w * scale;
    }
}

__device__ __forceinline__ float router_sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

__device__ __forceinline__ float sqrt_softplus(float x) {
    const float sp = x > 20.f ? x : log1pf(expf(x));
    return sqrtf(fmaxf(sp, 0.f));
}

template <class T>
__global__ __launch_bounds__(64) void moe_topk_sigmoid(const T* __restrict__ logits, i32* __restrict__ topk_idx,
                                                       float* __restrict__ topk_w,
                                                       const float* __restrict__ correction_bias, int E, int K,
                                                       bool renormalize, float routed_scaling_factor,
                                                       const u32* __restrict__ win) {
    topk_biased_row(logits, topk_idx, topk_w, correction_bias, E, K, renormalize, routed_scaling_factor, win,
                    router_sigmoid);
}

template <class T>
__global__ __launch_bounds__(64) void moe_topk_sqrt_softplus(const T* __restrict__ logits, i32* __restrict__ topk_idx,
                                                             float* __restrict__ topk_w,
                                                             const float* __restrict__ correction_bias, int E, int K,
                                                             bool renormalize, float routed_scaling_factor,
                                                             const u32* __restrict__ win) {
    topk_biased_row(logits, topk_idx, topk_w, correction_bias, E, K, renormalize, routed_scaling_factor, win,
                    sqrt_softplus);
}

// A row of E routed + S sink scores: the top K routed by sigmoid + bias,
// then every sink, all weighed by sigmoid and normalized together to
// scaling · global_scale. Picks past E land expert -1 at weight 0.
template <class T>
__global__ __launch_bounds__(64) void moe_topk_sigmoid_sink(const T* __restrict__ logits, i32* __restrict__ topk_idx,
                                                            float* __restrict__ topk_w,
                                                            const float* __restrict__ correction_bias,
                                                            const float* __restrict__ global_scale, int E, int S,
                                                            int K, float scaling, const u32* __restrict__ win) {
    if (threadIdx.x >= kWave) return;
    const long long row = router_row(win);
    if (row < 0) return;
    const int W = E + S;
    const T* x = logits + row * W;
    Ranked s = router_load(x, E, [&](int e, float v, float& r, float& k) {
        k = router_sigmoid(v);
        r = correction_bias != nullptr ? k + correction_bias[e] : k;
    });
    const int l = lane();
    int mine = -1;
    float mine_w = 0.f, sum = 0.f;
    const int picks = min(K, E);
    for (int k = 0; k < picks; ++k) {
        float w;
        const int e = router_pick(s, w);
        sum += w;
        if (l == k) {
            mine = e;
            mine_w = w;
        }
    }
    // The sinks, in order after the picks, each lane the ones it lands.
    for (int i = 0; i < S; ++i) {
        const float w = router_sigmoid(to_f32(x[E + i]));
        sum += w;
        if (l == K + i) {
            mine = E + i;
            mine_w = w;
        }
    }
    if (l < K + S) {
        const float gs = global_scale != nullptr ? global_scale[0] : 1.f;
        const float scale = scaling * gs / (sum + 1e-20f);
        topk_idx[row * (K + S) + l] = mine;
        topk_w[row * (K + S) + l] = mine_w * scale;
    }
}

// ---------------------------------------------------------------- folds --

// y[n][h] = Σ_k weights[n][k] · routed[n·top_k + k][h], in rank order.
template <class T>
__global__ void moe_weighted_sum(T* __restrict__ out, const T* __restrict__ src, const float* __restrict__ weights,
                                 int top_k, int hidden, const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long n = window.plane(blockIdx.x);
    const int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (h >= hidden) return;
    const long long base = n * top_k;
    float acc = 0.f;
    for (int k = 0; k < top_k; ++k) acc += weights[base + k] * to_f32(src[(base + k) * hidden + h]);
    out[n * hidden + h] = from_f32<T>(acc);
}

// y[n][h] = x[n][h] + Σ_k weights[n][k] · bias[routes[n][k]][h]; a dead
// route (expert < 0) adds nothing.
template <class T>
__global__ void moe_bias_sum(T* __restrict__ out, const T* __restrict__ x, const T* __restrict__ bias,
                             const i32* __restrict__ topk_idx, const float* __restrict__ weights, int top_k,
                             int hidden, const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long n = window.plane(blockIdx.x);
    const int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (h >= hidden) return;
    const long long base = n * top_k, i = n * hidden + h;
    float acc = to_f32(x[i]);
    for (int k = 0; k < top_k; ++k) {
        const int e = topk_idx[base + k];
        if (e < 0) continue;
        acc += weights[base + k] * to_f32(bias[static_cast<long long>(e) * hidden + h]);
    }
    out[i] = from_f32<T>(acc);
}

// -------------------------------------------------- the aligned legs ----

// Routes bucketed by expert into blocks of `block_size` rows: expert e's
// routes fill rows [offsets[e], offsets[e+1]), each expert's run padded to
// whole blocks; expert_ids[b] names block b's expert (-1 past the used
// blocks) and an unfilled row holds `num_routes`. Within an expert, routes
// land in an order the atomics pick. One block; dynamic LDS of
// (3·experts + 34) ints.
template <class T>
__global__ __launch_bounds__(256) void moe_align_decode(const T* __restrict__ topk_idx, T* __restrict__ sorted_route_ids,
                                                        T* __restrict__ expert_ids, T* __restrict__ route_to_aligned_row,
                                                        int num_routes, int num_experts, int block_size, int max_blocks,
                                                        T* __restrict__ num_tokens_past_padded) {
    static_assert(is_same<T, i32>::value, "the routing indices are i32");
    extern __shared__ i32 align_smem[];
    i32* counts = align_smem;
    i32* offsets = counts + num_experts;
    i32* fill = offsets + num_experts + 1;
    i32* wave_totals = fill + num_experts;

    const int aligned_rows = max_blocks * block_size;
    for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
        counts[i] = 0;
        fill[i] = 0;
    }
    for (int i = threadIdx.x; i < aligned_rows; i += blockDim.x) sorted_route_ids[i] = num_routes;
    for (int i = threadIdx.x; i < max_blocks; i += blockDim.x) expert_ids[i] = -1;
    __syncthreads();
    for (int r = threadIdx.x; r < num_routes; r += blockDim.x) {
        const int e = topk_idx[r];
        if (0 <= e && e < num_experts) atomicAdd(counts + e, 1);
    }
    __syncthreads();

    int base = 0;
    for (int e0 = 0; e0 < num_experts; e0 += blockDim.x) {
        const int e = e0 + threadIdx.x;
        const int padded = e < num_experts ? (counts[e] + block_size - 1) / block_size * block_size : 0;
        int total;
        const int before = block_scan(padded, wave_totals, total);
        if (e < num_experts) offsets[e] = base + before;
        base += total;
    }
    if (threadIdx.x == 0) offsets[num_experts] = base;
    __syncthreads();

    for (int e = threadIdx.x; e < num_experts; e += blockDim.x)
        for (int row = offsets[e]; row < offsets[e + 1]; row += block_size) {
            const int b = row / block_size;
            if (b < max_blocks) expert_ids[b] = e;
        }
    for (int r = threadIdx.x; r < num_routes; r += blockDim.x) {
        const int e = topk_idx[r];
        if (0 <= e && e < num_experts) {
            const int out = offsets[e] + atomicAdd(fill + e, 1);
            if (out < aligned_rows) {
                sorted_route_ids[out] = r;
                if (route_to_aligned_row != nullptr) route_to_aligned_row[r] = out;
            }
        }
    }
    if (num_tokens_past_padded != nullptr && threadIdx.x == 0) *num_tokens_past_padded = base;
}

// aligned_in[row] = x[token of the row's route] (zeros for an unfilled
// row); rows from `shared_row_begin` (≥ 0) take tokens in order instead.
template <class T>
__global__ void gather_moe_aligned_inputs(const T* __restrict__ norm_x, const i32* __restrict__ sorted_route_ids,
                                          T* __restrict__ aligned_in, int num_routes, int aligned_rows, int top_k,
                                          int hidden, int shared_row_begin, int num_tokens) {
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
    aligned_in[static_cast<long long>(row) * hidden + h] =
        token >= 0 ? norm_x[static_cast<long long>(token) * hidden + h] : from_f32<T>(0.f);
}

// route_out[route of the row] = aligned_out[row]; rows from
// `shared_row_begin` land in shared_out by token.
template <class T>
__global__ void reorder_moe_aligned_output(const T* __restrict__ aligned_out, const i32* __restrict__ sorted_route_ids,
                                           T* __restrict__ route_out, int num_routes, int aligned_rows, int hidden,
                                           int shared_row_begin, int num_tokens, T* __restrict__ shared_out) {
    const int h = blockIdx.y * blockDim.x + threadIdx.x;
    const int row = blockIdx.x;
    if (h >= hidden || row >= aligned_rows) return;
    const T v = aligned_out[static_cast<long long>(row) * hidden + h];
    if (shared_row_begin >= 0 && row >= shared_row_begin) {
        const int t = row - shared_row_begin;
        if (t < num_tokens) shared_out[static_cast<long long>(t) * hidden + h] = v;
        return;
    }
    const int route = sorted_route_ids[row];
    if (route < num_routes) route_out[static_cast<long long>(route) * hidden + h] = v;
}

// The batched GEMM's pointer triples, one a block of aligned rows.
template <class T>
__global__ void build_moe_leg_ptrs(const i32* __restrict__ expert_ids, const T* __restrict__ bank_base,
                                   const T* __restrict__ aligned_in, T* __restrict__ aligned_out,
                                   const T** __restrict__ w_ptrs, const T** __restrict__ act_ptrs,
                                   T** __restrict__ out_ptrs, int max_blocks, int block_size, int K, int N) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= max_blocks) return;
    const int e = expert_ids[b];
    const long long row = static_cast<long long>(b) * block_size;
    w_ptrs[b] = bank_base + static_cast<long long>(e < 0 ? 0 : e) * N * K;
    act_ptrs[b] = aligned_in + row * K;
    out_ptrs[b] = aligned_out + row * N;
}

// ------------------------------------------------------ dense matmuls ---

// y[route] = bank[expert]·x[token or route] for a T bank (`table`, when
// given, names each expert's rows; `hits` counts each expert's routes).
// Grid (routes, ⌈N / (4·R)⌉), 256 threads; see stream_rows.
template <class T, bool BY_TOKEN>
__device__ __forceinline__ void select_gemv(const i32* __restrict__ topk_idx, const T* __restrict__ act,
                                            const T* __restrict__ weight_base, T* __restrict__ out, int top_k, int K,
                                            int N, long long expert_stride, const T* const* __restrict__ expert_table,
                                            unsigned int* __restrict__ expert_hits, int rows_per_wave,
                                            const u32* __restrict__ win) {
    extern __shared__ __align__(16) unsigned char stream_smem[];
    const int route = blockIdx.x;
    if (win != nullptr && route >= static_cast<int>(win[0]) * top_k) return;
    const long long plane_route = win != nullptr ? route + static_cast<long long>(win[1]) * top_k : route;
    const int expert = topk_idx[plane_route];
    T* y = out + plane_route * N;
    if (expert < 0) {
        stream_zero(y, N, rows_per_wave);
        return;
    }
    if (expert_hits != nullptr && blockIdx.y == 0 && threadIdx.x == 0) atomicAdd(&expert_hits[expert], 1u);
    const T* w = expert_table != nullptr ? expert_table[expert] : weight_base + expert * expert_stride;
    const T* x = act + (BY_TOKEN ? plane_route / top_k : plane_route) * K;
    stream_rows<StreamDense<T>, T>({w}, x, y, nullptr, N, K, rows_per_wave, stream_smem);
}

template <class T>
__global__ __launch_bounds__(256) void moe_matmul_select_gemv_by_token(
    const i32* __restrict__ topk_idx, const T* __restrict__ act, const T* __restrict__ weight_base,
    T* __restrict__ out, int top_k, int K, int N, long long expert_stride, const T* const* __restrict__ expert_table,
    unsigned int* __restrict__ expert_hits, int rows_per_wave, const u32* __restrict__ win) {
    select_gemv<T, true>(topk_idx, act, weight_base, out, top_k, K, N, expert_stride, expert_table, expert_hits,
                         rows_per_wave, win);
}

template <class T>
__global__ __launch_bounds__(256) void moe_matmul_select_gemv_by_route(
    const i32* __restrict__ topk_idx, const T* __restrict__ act, const T* __restrict__ weight_base,
    T* __restrict__ out, int top_k, int K, int N, long long expert_stride, const T* const* __restrict__ expert_table,
    unsigned int* __restrict__ expert_hits, int rows_per_wave, const u32* __restrict__ win) {
    select_gemv<T, false>(topk_idx, act, weight_base, out, top_k, K, N, expert_stride, expert_table, expert_hits,
                          rows_per_wave, win);
}

// The grouped dense leg: routes ordered by expert (moe_route_order in
// linear/quant.hip), one work item of up to MT a block. Grid (work items,
// ⌈N / (32·waves)⌉), kGroupWaves<MT> waves.
template <class T, int MT>
__global__ __launch_bounds__(kGroupWaves<MT>* kWave) void moe_matmul_select_grouped(const T* __restrict__ act, const i32* __restrict__ order,
                                                                 const i32* __restrict__ offsets,
                                                                 const i32* __restrict__ work,
                                                                 const T* __restrict__ bank, T* __restrict__ out,
                                                                 int act_div, int n, int k, int num_experts) {
    GroupItem item;
    if (!group_item(offsets, work, num_experts, MT, item)) return;
    const T* w = bank + static_cast<long long>(item.expert) * n * k;
    grouped_tile<GroupDense<T>, T, MT>(act, order, item.begin, item.count, {w}, nullptr, out, act_div, n, k);
}

// The batch dense leg (decode batches): one work item of up to MT routes a
// block, 64·NBW weight rows (grid (work cap, ⌈n / (64·NBW)⌉), 256 threads),
// a stage 128 elements of a row.
template <class T, int MT, int NBW>
__global__ __launch_bounds__(256) void moe_matmul_select_dense_batch(
    const T* __restrict__ act, const i32* __restrict__ order, const i32* __restrict__ offsets,
    const i32* __restrict__ work, const T* __restrict__ bank, T* __restrict__ out, int act_div, int n, int k,
    int num_experts) {
    GroupItem item;
    if (!group_item(offsets, work, num_experts, MT, item)) return;
    const T* w = bank + static_cast<long long>(item.expert) * n * k;
    batch_tile<BatchDense<T, MT >= 32 ? 128 : 256>, T, MT, NBW, 3>(act, order, item.begin, item.count, {w}, nullptr, out, act_div, n,
                                                  k, blockIdx.y);
}

constexpr int kDenseXcds = 8;

// The wide dense leg (wide decode batches): one work item of up to 4·MW
// routes a block, NT weight rows, KC elements a stage; k a multiple of KC and the activation
// under 4 GiB. One dimension of 8·per_xcd·tiles blocks: XCD x (block b runs
// on XCD b mod 8) takes items x, x + 8, ..., their tiles in turn, so the
// blocks one item's activation feeds run on one XCD at once and find it in
// its L2.
template <class T, int MW, int NT, int KC, int DEPTH, int ADEPTH>
__global__ __launch_bounds__(256) void moe_matmul_select_dense_wide(
    const T* __restrict__ act, const i32* __restrict__ order, const i32* __restrict__ offsets,
    const i32* __restrict__ work, const T* __restrict__ bank, T* __restrict__ out, int act_div, int n, int k,
    int num_experts, u32 act_bytes, int per_xcd) {
    const int b = blockIdx.x, slot = b / kDenseXcds;
    const int at = b % kDenseXcds + kDenseXcds * (slot % per_xcd), tile = slot / per_xcd;
    GroupItem item;
    if (!group_item(offsets, work, num_experts, 4 * MW, item, at)) return;
    const T* w = bank + static_cast<long long>(item.expert) * n * k;
    wide_tile<T, MW, NT, KC, DEPTH, ADEPTH>(act, act_bytes, order, item.begin, item.count, w, out, act_div, n, k, tile);
}

// The dense projection as one expert every row routes to: y[m, n] = act[m,
// k]·wᵀ (SWIGLU: h[m, n] = silu(act·gateᵀ)·(act·upᵀ) from a packed [gate |
// up] bank of 2·n rows), MT rows of act an item, 64 output rows a block. One
// dimension of blocks: each 64-row tile's `splits` slices of K, `items`
// items each, in turn, the tiles dealt round the eight XCDs (block b runs on
// XCD b mod 8) so a tile's items run on one XCD at once and their later
// reads of the weight (cached reads, past one item) find its L2. A slice is
// `per` stages; SPLIT lands each slice's f32 partial in `part` ([splits, m,
// n], 2·n under SWIGLU) for mgemm_finalize to sum in slice order.
template <int MT, bool SWIGLU, bool SPLIT, bool NT>
__global__ __launch_bounds__(256) void dense_batch_bf16(const bf16* __restrict__ act, const bf16* __restrict__ w,
                                                        bf16* __restrict__ y, float* __restrict__ part, int m, int n,
                                                        int k, int items, int tiles, int splits, int per) {
    constexpr int SBY = MT >= 32 ? 128 : 256;  // a stage of 64 or 128 elements
    const int b = blockIdx.x, xcd = b % kDenseXcds, slot = b / kDenseXcds;
    const int item = slot % items, rest = slot / items, split = rest % splits;
    const int tile = rest / splits * kDenseXcds + xcd;
    if (tile >= tiles) return;
    const int begin = item * MT;
    float* slice = SPLIT ? part + static_cast<long long>(split) * m * (SWIGLU ? 2 * n : n) : nullptr;
    batch_tile<BatchDense<bf16, SBY>, bf16, MT, SWIGLU ? 2 : 1, 3, SWIGLU, NT>(
        act, nullptr, begin, min(MT, m - begin), {w}, nullptr, y, 1, n, k, tile, split * per, (split + 1) * per,
        slice);
}

}  // namespace pie::linear
