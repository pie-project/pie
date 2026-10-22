#pragma once

// Mixture of experts: the routers (top-k over a token's expert scores), the
// dense expert matmuls (linear/moe_kit.hip), and the folds of the routed
// rows back into token rows. A route is (token, rank): route r of a fire is
// token r / top_k's pick r % top_k, and a window states live tokens.

#include "linear/moe_kit.hip"

namespace pie::linear {

// ------------------------------------------------------------- routers --

// A router holds a token's scores in registers, SL slots a lane (score j in
// lane j % kWave, slot j / kWave: SL whole waves cover the row), and one
// wavefront picks.

template <int SL>
struct Ranked {
    float rank[SL];  // what orders the picks
    float keep[SL];  // what a pick weighs
    u32 out;         // slots out of the running: picked, or past the row
};

// A candidate as one key whose max is the pick: a larger score, an equal
// one at a lower index, any over none (0). The score's bits made monotone
// (-0 as +0) above the complement of its index.
__device__ __forceinline__ unsigned long long router_key(float v, int i) {
    u32 b = __builtin_bit_cast(u32, v);
    b = b == 0x80000000u ? 0u : b;
    b = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    return (static_cast<unsigned long long>(b) << 32) | (0xFFFFFFFFu - static_cast<u32>(i));
}

// max(x, x of the lane DPP control CTRL names), rows outside ROWS keeping x.
template <int CTRL, int ROWS>
__device__ __forceinline__ u32 dpp_max(u32 x) {
    return max(x, static_cast<u32>(__builtin_amdgcn_update_dpp(static_cast<int>(x), static_cast<int>(x), CTRL, ROWS,
                                                               0xf, false)));
}

// The wavefront's largest value, every lane: a butterfly over quads, rows'
// halves and rows by DPP, rows' maxima carried up to lane 63.
__device__ __forceinline__ u32 wave_max_u32(u32 x) {
    if constexpr (kWave == 64) {
        x = dpp_max<0xB1, 0xf>(x);   // quad_perm [1, 0, 3, 2]
        x = dpp_max<0x4E, 0xf>(x);   // quad_perm [2, 3, 0, 1]
        x = dpp_max<0x141, 0xf>(x);  // row_half_mirror
        x = dpp_max<0x140, 0xf>(x);  // row_mirror
        x = dpp_max<0x142, 0xa>(x);  // row_bcast:15
        x = dpp_max<0x143, 0xc>(x);  // row_bcast:31
        return static_cast<u32>(__builtin_amdgcn_readlane(static_cast<int>(x), 63));
    } else {
#pragma unroll
        for (int off = kWave / 2; off > 0; off >>= 1)
            x = max(x, static_cast<u32>(__shfl_xor(static_cast<int>(x), off, kWave)));
        return x;
    }
}

// The picks over a row's keys: each lane's slots as keys (0 once out of the
// running) and the best of them, which only a pick's owner recomputes.
template <int SL>
struct Picks {
    unsigned long long key[SL];
    unsigned long long best;

    __device__ __forceinline__ void rank() {
        best = 0;
#pragma unroll
        for (int t = 0; t < SL; ++t) best = key[t] > best ? key[t] : best;
    }
};

template <int SL>
__device__ __forceinline__ Picks<SL> router_picks(const Ranked<SL>& s) {
    Picks<SL> p;
    const int l = lane();
#pragma unroll
    for (int t = 0; t < SL; ++t) p.key[t] = (s.out >> t) & 1u ? 0ull : router_key(s.rank[t], l + kWave * t);
    p.rank();
    return p;
}

// The best key still running, its weight, and takes it out of the running;
// -1 once none is left. The wave's largest high word first, then (only on a
// tie there) the largest low word among the lanes holding it.
template <int SL>
__device__ __forceinline__ int router_pick(const Ranked<SL>& s, Picks<SL>& p, float& weight) {
    const int l = lane();
    const u32 hi = static_cast<u32>(p.best >> 32), lo = static_cast<u32>(p.best);
    const u32 top = wave_max_u32(hi);
    const unsigned long long holders = __ballot(hi == top);
    const u32 low = __popcll(holders) == 1
                        ? static_cast<u32>(__builtin_amdgcn_readlane(static_cast<int>(lo), __ffsll(holders) - 1))
                        : wave_max_u32(hi == top ? lo : 0u);
    if (top == 0 && low == 0) {
        weight = 0.f;
        return -1;
    }
    const int bi = static_cast<int>(0xFFFFFFFFu - low);
    const int owner = bi % kWave, slot = bi / kWave;
    float kw = 0.f;
#pragma unroll
    for (int t = 0; t < SL; ++t) kw = t == slot ? s.keep[t] : kw;
    weight = __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, kw), owner));
    if (l == owner) {
#pragma unroll
        for (int t = 0; t < SL; ++t) p.key[t] = t == slot ? 0ull : p.key[t];
        p.rank();
    }
    return bi;
}

// Row `n` of a windowed router launch, or -1 when the window has it dead.
__device__ __forceinline__ long long router_row(const u32* win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return -1;
    return window.plane(blockIdx.x);
}

// Loads a row's `E` scores (and their `bias`, 0 without) through
// f(x, bias) -> (rank, keep). Every slot reads (past the row: the last
// score) so that the loads issue together.
template <int SL, class T, class F>
__device__ __forceinline__ Ranked<SL> router_load(const T* __restrict__ row, const float* __restrict__ bias, int E,
                                                  F&& f) {
    Ranked<SL> s;
    s.out = 0;
    const int l = lane();
    float x[SL], b[SL];
#pragma unroll
    for (int t = 0; t < SL; ++t) {
        const int e = min(l + kWave * t, E - 1);
        x[t] = to_f32(row[e]);
        b[t] = bias != nullptr ? bias[e] : 0.f;
    }
#pragma unroll
    for (int t = 0; t < SL; ++t) {
        s.rank[t] = 0.f;
        s.keep[t] = 0.f;
        if (l + kWave * t < E) {
            f(x[t], b[t], s.rank[t], s.keep[t]);
        } else {
            s.out |= 1u << t;
        }
    }
    return s;
}

// softmax over the row, the top K by probability, renormalized over the
// picks (and scaled by `gain[expert]` when given). One wavefront a row.
template <int SL, class T>
__device__ __forceinline__ void topk_softmax_row(const T* __restrict__ logits, const T* __restrict__ gain,
                                                 i32* __restrict__ topk_idx, float* __restrict__ topk_w, int E, int K,
                                                 const u32* __restrict__ win) {
    if (threadIdx.x >= kWave) return;
    const long long row = router_row(win);
    if (row < 0) return;
    Ranked<SL> s =
        router_load<SL>(logits + row * E, nullptr, E, [](float x, float, float& r, float& k) { r = k = x; });
    float m = neg_inf();
#pragma unroll
    for (int t = 0; t < SL; ++t)
        if (!((s.out >> t) & 1u)) m = fmaxf(m, s.rank[t]);
    m = wave_max(m);
    float z = 0.f;
#pragma unroll
    for (int t = 0; t < SL; ++t) {
        if (!((s.out >> t) & 1u)) {
            s.rank[t] = expf(s.rank[t] - m);
            z += s.rank[t];
        }
    }
    const float inv_z = 1.f / wave_sum(z);
#pragma unroll
    for (int t = 0; t < SL; ++t) s.keep[t] = s.rank[t] = s.rank[t] * inv_z;

    const int l = lane();
    int mine = -1;
    float mine_w = 0.f, sum = 0.f;
    Picks<SL> p = router_picks(s);
    for (int k = 0; k < K; ++k) {
        float w;
        const int e = router_pick(s, p, w);
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

template <class T, int SL>
__global__ __launch_bounds__(64) void moe_topk_softmax(const T* __restrict__ logits, const T* __restrict__,
                                                       const T* __restrict__, i32* __restrict__ topk_idx,
                                                       float* __restrict__ topk_w, int num_experts, int K, int,
                                                       const u32* __restrict__ win) {
    topk_softmax_row<SL, T>(logits, nullptr, topk_idx, topk_w, num_experts, K, win);
}

template <class T, int SL>
__global__ __launch_bounds__(64) void moe_topk_softmax_scaled(const T* __restrict__ logits, const T* __restrict__,
                                                              const T* __restrict__ per_expert_scale,
                                                              i32* __restrict__ topk_idx, float* __restrict__ topk_w,
                                                              int num_experts, int K, int,
                                                              const u32* __restrict__ win) {
    topk_softmax_row<SL, T>(logits, per_expert_scale, topk_idx, topk_w, num_experts, K, win);
}

// The top K by score + bias (no bias: by score), weighed by the score;
// renormalized to `scaling` over the picks when asked, else scaled by it.
// Picks past E land expert 0 at weight 0.
template <int SL, class T, class F>
__device__ __forceinline__ void topk_biased_row(const T* __restrict__ logits, i32* __restrict__ topk_idx,
                                                float* __restrict__ topk_w, const float* __restrict__ bias, int E,
                                                int K, bool renormalize, float scaling, const u32* __restrict__ win,
                                                F&& score) {
    if (threadIdx.x >= kWave) return;
    const long long row = router_row(win);
    if (row < 0) return;
    Ranked<SL> s = router_load<SL>(logits + row * E, bias, E, [&](float x, float b, float& r, float& k) {
        k = score(x);
        r = k + b;
    });
    const int l = lane();
    int mine = 0;
    float mine_w = 0.f, sum = 0.f;
    const int picks = min(K, E);
    Picks<SL> p = router_picks(s);
    for (int k = 0; k < picks; ++k) {
        float w;
        const int e = router_pick(s, p, w);
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

template <class T, int SL>
__global__ __launch_bounds__(64) void moe_topk_sigmoid(const T* __restrict__ logits, i32* __restrict__ topk_idx,
                                                       float* __restrict__ topk_w,
                                                       const float* __restrict__ correction_bias, int E, int K,
                                                       bool renormalize, float routed_scaling_factor,
                                                       const u32* __restrict__ win) {
    topk_biased_row<SL>(logits, topk_idx, topk_w, correction_bias, E, K, renormalize, routed_scaling_factor, win,
                        router_sigmoid);
}

template <class T, int SL>
__global__ __launch_bounds__(64) void moe_topk_sqrt_softplus(const T* __restrict__ logits, i32* __restrict__ topk_idx,
                                                             float* __restrict__ topk_w,
                                                             const float* __restrict__ correction_bias, int E, int K,
                                                             bool renormalize, float routed_scaling_factor,
                                                             const u32* __restrict__ win) {
    topk_biased_row<SL>(logits, topk_idx, topk_w, correction_bias, E, K, renormalize, routed_scaling_factor, win,
                        sqrt_softplus);
}

// A row of E routed + S sink scores: the top K routed by sigmoid + bias,
// then every sink, all weighed by sigmoid and normalized together to
// scaling · global_scale. Picks past E land expert -1 at weight 0.
template <class T, int SL>
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
    Ranked<SL> s = router_load<SL>(x, correction_bias, E, [](float v, float b, float& r, float& k) {
        k = router_sigmoid(v);
        r = k + b;
    });
    const int l = lane();
    int mine = -1;
    float mine_w = 0.f, sum = 0.f;
    const int picks = min(K, E);
    Picks<SL> p = router_picks(s);
    for (int k = 0; k < picks; ++k) {
        float w;
        const int e = router_pick(s, p, w);
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

// y[n][h] = Σ_k weights[n][k] · routed[n·top_k + k][h], in rank order;
// given `shared`, rounded and then shared[n][h]·sigmoid(gate[n·gate_stride])
// added, as `moe_sigmoid_gate_add` would. Given `gate_w` (one row of k), the
// gate is act[n] · gate_w, rounded, the block's own. VEC elements a thread
// (8: hidden a multiple of 8, the rows 16-byte aligned), K routes a token.
template <class T, int VEC, int K>
__global__ void moe_weighted_sum(T* __restrict__ out, const T* __restrict__ src, const float* __restrict__ weights,
                                 int hidden, const T* __restrict__ shared, const T* __restrict__ gate,
                                 int gate_stride, const u32* __restrict__ win, const T* __restrict__ gate_w,
                                 const T* __restrict__ act, int k) {
    using V = conditional<VEC == 8, Pack8<T>, T>::type;
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long n = window.plane(blockIdx.x);
    const int h = (blockIdx.y * blockDim.x + threadIdx.x) * VEC;
    // The routed rows, their weights and the shared row in flight before
    // the gate's dot waits on its own reads (past the row: its last column).
    const int hc = min(h, hidden - VEC);
    float w[K];
    V v[K], sh;
#pragma unroll
    for (int r = 0; r < K; ++r) {
        w[r] = weights[n * K + r];
        v[r] = *reinterpret_cast<const V*>(src + (n * K + r) * hidden + hc);
    }
    if (shared != nullptr) sh = *reinterpret_cast<const V*>(shared + n * hidden + hc);
    float logit = 0.f;
    if (gate_w != nullptr) {
        __shared__ float red[kWave];
        float dot = 0.f;
        for (int c = static_cast<int>(threadIdx.x) * 8; c < k; c += static_cast<int>(blockDim.x) * 8) {
            const Pack8<T> a = *reinterpret_cast<const Pack8<T>*>(act + n * k + c);
            const Pack8<T> g = *reinterpret_cast<const Pack8<T>*>(gate_w + c);
#pragma unroll
            for (int j = 0; j < 8; ++j) dot = fmaf(to_f32(a.v[j]), to_f32(g.v[j]), dot);
        }
        logit = to_f32(from_f32<T>(block_sum(dot, red)));
    } else if (shared != nullptr) {
        logit = to_f32(gate[n * gate_stride]);
    }
    if (h >= hidden) return;
    const T* vs = reinterpret_cast<const T*>(v);
    const T* shs = reinterpret_cast<const T*>(&sh);
    float acc[VEC] = {};
#pragma unroll
    for (int r = 0; r < K; ++r)
#pragma unroll
        for (int j = 0; j < VEC; ++j) acc[j] += w[r] * to_f32(vs[r * VEC + j]);
    const float sg = shared != nullptr ? 1.f / (1.f + __expf(-logit)) : 0.f;
    V o;
    T* os = reinterpret_cast<T*>(&o);
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
        float a = acc[j];
        if (shared != nullptr) a = to_f32(from_f32<T>(a)) + to_f32(shs[j]) * sg;
        os[j] = from_f32<T>(a);
    }
    *reinterpret_cast<V*>(out + n * hidden + h) = o;
}

// y[n][h] = Σ_k src[n·top_k + k][h] + shared[n][h], the routed rows already
// weighted, summed in f32 and rounded once; VEC columns a lane (8: one
// 16-byte pack of every row, the width a whole number of them).
template <class T, int VEC>
__global__ void moe_sum_add(T* __restrict__ out, const T* __restrict__ src, const T* __restrict__ shared,
                            int top_k, int hidden, const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long n = window.plane(blockIdx.x);
    const int h = (blockIdx.y * blockDim.x + threadIdx.x) * VEC;
    if (h >= hidden) return;
    const long long base = n * top_k;
    float acc[VEC] = {};
    if constexpr (VEC == 8) {
        for (int k = 0; k < top_k; ++k) {
            const Pack8<T> v = *reinterpret_cast<const Pack8<T>*>(src + (base + k) * hidden + h);
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[e] += to_f32(v.v[e]);
        }
        const Pack8<T> sh = *reinterpret_cast<const Pack8<T>*>(shared + n * hidden + h);
        Pack8<T> o;
#pragma unroll
        for (int e = 0; e < 8; ++e) o.v[e] = from_f32<T>(acc[e] + to_f32(sh.v[e]));
        *reinterpret_cast<Pack8<T>*>(out + n * hidden + h) = o;
    } else {
        for (int k = 0; k < top_k; ++k) acc[0] += to_f32(src[(base + k) * hidden + h]);
        out[n * hidden + h] = from_f32<T>(acc[0] + to_f32(shared[n * hidden + h]));
    }
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

// A one-token fire's routed MLP in two launches, a wave a row: each lane
// holds CH 16-byte chunks of its row (K = 512·CH) and the activation's
// matching chunks, so a row is one dot and one wave sum. Grid (routes or
// tokens, ⌈rows / (4·RPW)⌉), 256 threads.
constexpr int kDecodeWaves = 4;

// A dword's two 16-bit elements as f32.
template <class T>
__device__ __forceinline__ void halves(u32 v, float& lo, float& hi) {
    if constexpr (is_same<T, bf16>::value) {
        lo = __builtin_bit_cast(float, v << 16);
        hi = __builtin_bit_cast(float, v & 0xffff0000u);
    } else {
        lo = to_f32(__builtin_bit_cast(T, static_cast<unsigned short>(v)));
        hi = to_f32(__builtin_bit_cast(T, static_cast<unsigned short>(v >> 16)));
    }
}

template <class T>
__device__ __forceinline__ float dot8(moe_u4 w, b128 x) {
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float a0, a1, b0, b1;
        halves<T>(w[i], a0, a1);
        halves<T>(x.w[i], b0, b1);
        s = fmaf(a0, b0, s);
        s = fmaf(a1, b1, s);
    }
    return s;
}

template <int CH, class T>
__device__ __forceinline__ void decode_row(const T* row, moe_u4* w) {
    const moe_u4* p = reinterpret_cast<const moe_u4*>(row) + lane();
#pragma unroll
    for (int c = 0; c < CH; ++c) w[c] = __builtin_nontemporal_load(p + c * kWave);
}

// h[route][i] = silu(G·x)·(U·x), G row i and U row I + i of the route's
// packed [gate | up] expert; a dead route lands zeros.
template <class T, int CH, int RPW>
__global__ __launch_bounds__(kDecodeWaves* kWave) void moe_decode_glu(
    const i32* __restrict__ topk_idx, const T* __restrict__ act, const T* __restrict__ bank, T* __restrict__ h,
    int top_k, int I, long long expert_stride, const u32* __restrict__ win) {
    constexpr int K = 512 * CH;
    const int route = blockIdx.x;
    if (win != nullptr && route >= static_cast<int>(win[0]) * top_k) return;
    const long long plane_route = win != nullptr ? route + static_cast<long long>(win[1]) * top_k : route;
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / kWave);
    const int row0 = (blockIdx.y * kDecodeWaves + wave) * RPW;
    if (row0 >= I) return;
    const int l = lane();
    const int expert = topk_idx[plane_route];
    T* y = h + plane_route * I;
    if (expert < 0) {
        if (l < RPW && row0 + l < I) y[row0 + l] = from_f32<T>(0.f);
        return;
    }
    // The activation's reads issue first: a wait for them waits out no weight.
    const b128* x = reinterpret_cast<const b128*>(act + (plane_route / top_k) * K) + l;
    b128 xs[CH];
#pragma unroll
    for (int c = 0; c < CH; ++c) xs[c] = x[c * kWave];
    const T* w = bank + expert * expert_stride;
    moe_u4 g[RPW][CH], u[RPW][CH];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const long long row = min(row0 + r, I - 1);
        decode_row<CH>(w + row * K, g[r]);
        decode_row<CH>(w + (I + row) * K, u[r]);
    }
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        float a = 0.f, b = 0.f;
#pragma unroll
        for (int c = 0; c < CH; ++c) {
            a += dot8<T>(g[r][c], xs[c]);
            b += dot8<T>(u[r][c], xs[c]);
        }
        a = wave_sum(a);
        b = wave_sum(b);
        if (l == 0 && row0 + r < I) y[row0 + r] = from_f32<T>(a / (1.f + __expf(-a)) * b);
    }
}

// y[token][j] = Σ_k w_k · (D[e_k] row j)·h[route k], plus shared[token][j]·
// sigmoid(gate[token·gate_stride]) where `shared` is given; a dead route
// adds nothing.
template <class T, int CH, int TOPK, int RPW>
__global__ __launch_bounds__(kDecodeWaves* kWave) void moe_decode_down(
    const i32* __restrict__ topk_idx, const float* __restrict__ weights, const T* __restrict__ h,
    const T* __restrict__ bank, T* __restrict__ out, const T* __restrict__ shared, const T* __restrict__ gate,
    int gate_stride, int H, long long expert_stride, const u32* __restrict__ win) {
    constexpr int K = 512 * CH;
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long token = window.plane(blockIdx.x);
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / kWave);
    const int row0 = (blockIdx.y * kDecodeWaves + wave) * RPW;
    if (row0 >= H) return;
    const int l = lane();
    int expert[TOPK];
    float wk[TOPK];
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
        expert[k] = topk_idx[token * TOPK + k];
        wk[k] = weights[token * TOPK + k];
    }
    b128 hs[TOPK][CH];
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
        const b128* x = reinterpret_cast<const b128*>(h + (token * TOPK + k) * K) + l;
#pragma unroll
        for (int c = 0; c < CH; ++c) hs[k][c] = x[c * kWave];
    }
    moe_u4 dv[RPW][TOPK][CH];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const long long row = min(row0 + r, H - 1);
#pragma unroll
        for (int k = 0; k < TOPK; ++k)
            decode_row<CH>(bank + max(expert[k], 0) * expert_stride + row * K, dv[r][k]);
    }
    const float s = shared != nullptr ? 1.f / (1.f + __expf(-to_f32(gate[token * gate_stride]))) : 0.f;
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
            float d = 0.f;
#pragma unroll
            for (int c = 0; c < CH; ++c) d += dot8<T>(dv[r][k][c], hs[k][c]);
            acc = expert[k] < 0 ? acc : fmaf(wk[k], d, acc);
        }
        acc = wave_sum(acc);
        const int row = row0 + r;
        if (l == 0 && row < H) {
            if (shared != nullptr) acc = fmaf(to_f32(shared[token * H + row]), s, acc);
            out[token * H + row] = from_f32<T>(acc);
        }
    }
}

constexpr int kDenseXcds = 8;

// The grouped dense leg: routes ordered by expert (moe_route_order in
// linear/quant.hip), one work item of up to MT a block. Grid (work items,
// ⌈N / (32·waves)⌉), kGroupWaves<MT> waves. SORTED as grouped_tile's.
template <class T, int MT, int SORTED = 0>
__global__ __launch_bounds__(kGroupWaves<MT>* kWave) void moe_matmul_select_grouped(const T* __restrict__ act, const i32* __restrict__ order,
                                                                 const i32* __restrict__ offsets,
                                                                 const i32* __restrict__ work,
                                                                 const T* __restrict__ bank, T* __restrict__ out,
                                                                 int act_div, int n, int k, int num_experts) {
    // Each XCD (block b runs on XCD b mod 8) takes a run of the grid in
    // (tile, item) order: its CUs walk one weight tile's items together, so
    // the tile is read from its L2 once a wave of them, not once an XCD each.
    const int blocks = gridDim.x * gridDim.y;
    const int b = blockIdx.x + gridDim.x * blockIdx.y;
    const int x = b % kDenseXcds, q = blocks / kDenseXcds, r = blocks % kDenseXcds;
    const int at = x * q + min(x, r) + b / kDenseXcds;  // XCD x owns q (+1 below r) blocks
    GroupItem item;
    if (!group_item(offsets, work, num_experts, MT, item, at % gridDim.x)) return;
    const T* w = bank + static_cast<long long>(item.expert) * n * k;
    grouped_tile<GroupDense<T>, T, MT, 2, kGroupWaves<MT>, SORTED>(act, order, item.begin, item.count, {w},
                                                                  nullptr, out, act_div, n, k, at / gridDim.x);
}

// The grouped dense leg at a prefill's routes: NBW 16-row weight blocks a
// wave over WAVES waves (16·NBW·WAVES weight rows a block), so each
// activation piece a wave reads from LDS feeds NBW times the products.
template <class T, int MT, int NBW, int WAVES, int SORTED = 0, int GLU = 0>
__global__ __launch_bounds__(WAVES* kWave) void moe_matmul_select_grouped_tall(
    const T* __restrict__ act, const i32* __restrict__ order, const i32* __restrict__ offsets,
    const i32* __restrict__ work, const T* __restrict__ bank, T* __restrict__ out, int act_div, int n, int k,
    int num_experts, float limit) {
    const int blocks = gridDim.x * gridDim.y;
    const int b = blockIdx.x + gridDim.x * blockIdx.y;
    const int x = b % kDenseXcds, q = blocks / kDenseXcds, r = blocks % kDenseXcds;
    const int at = x * q + min(x, r) + b / kDenseXcds;
    GroupItem item;
    if (!group_item(offsets, work, num_experts, MT, item, at % gridDim.x)) return;
    const T* w = bank + static_cast<long long>(item.expert) * n * k;
    grouped_tile<GroupDense<T>, T, MT, NBW, WAVES, SORTED, GLU>(act, order, item.begin, item.count, {w}, nullptr,
                                                                out, act_div, n, k, at / gridDim.x, limit);
}

// The batch dense leg (decode batches): one work item of up to MT routes a
// block, 64·NBW weight rows (grid (shared items + work cap, ⌈n / (64·NBW)⌉),
// 256 threads), a stage 128 elements of a row. A shared expert (`shared` a
// dense bank every one of `shared_rows` rows takes) runs as the first
// ⌈shared_rows / MT⌉ items, from `shared_act` into `shared_out` by row.
template <class T, int MT, int NBW>
__global__ __launch_bounds__(256) void moe_matmul_select_dense_batch(
    const T* __restrict__ act, const i32* __restrict__ order, const i32* __restrict__ offsets,
    const i32* __restrict__ work, const T* __restrict__ bank, T* __restrict__ out, int act_div, int n, int k,
    int num_experts, const T* __restrict__ shared, const T* __restrict__ shared_act, T* __restrict__ shared_out,
    int shared_rows) {
    constexpr int SBY = MT >= 32 ? 128 : 256;
    const int items = (shared_rows + MT - 1) / MT, b = static_cast<int>(blockIdx.x);
    if (b < items) {
        batch_tile<BatchDense<T, SBY>, T, MT, NBW, 3>(shared_act, nullptr, b * MT, min(MT, shared_rows - b * MT),
                                                      {shared}, nullptr, shared_out, 1, n, k, blockIdx.y);
        return;
    }
    GroupItem item;
    if (!group_item(offsets, work, num_experts, MT, item, b - items)) return;
    const T* w = bank + static_cast<long long>(item.expert) * n * k;
    batch_tile<BatchDense<T, SBY>, T, MT, NBW, 3>(act, order, item.begin, item.count, {w}, nullptr, out, act_div, n, k,
                                                  blockIdx.y);
}


// The gate-up leg's batch with its swiglu: h[route] = silu(G·x)·(U·x) over
// a packed [gate | up] bank of 2n rows, as batch_tile's SWIGLU (grid
// (shared items + work cap, ⌈n / 32⌉), 256 threads); a shared
// expert as moe_matmul_select_dense_batch's, from the token rows `act`.
template <class T, int MT>
__global__ __launch_bounds__(256) void moe_matmul_select_dense_batch_swiglu(
    const T* __restrict__ act, const i32* __restrict__ order, const i32* __restrict__ offsets,
    const i32* __restrict__ work, const T* __restrict__ bank, T* __restrict__ h, int act_div, int n, int k,
    int num_experts, const T* __restrict__ shared, T* __restrict__ shared_h, int shared_rows) {
    constexpr int SBY = MT >= 32 ? 128 : 256;
    const int items = (shared_rows + MT - 1) / MT, b = static_cast<int>(blockIdx.x);
    if (b < items) {
        batch_tile<BatchDense<T, SBY>, T, MT, 1, 3, true>(act, nullptr, b * MT, min(MT, shared_rows - b * MT),
                                                          {shared}, nullptr, shared_h, 1, n, k, blockIdx.y);
        return;
    }
    GroupItem item;
    if (!group_item(offsets, work, num_experts, MT, item, b - items)) return;
    const T* w = bank + static_cast<long long>(item.expert) * 2 * n * k;
    batch_tile<BatchDense<T, SBY>, T, MT, 1, 3, true>(act, order, item.begin, item.count, {w}, nullptr, h, act_div, n,
                                                      k, blockIdx.y);
}

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

// A dense projection as one expert every row routes to (SWIGLU: a packed
// [gate | up] bank), MT rows an item, 64 output rows (SwiGLU: 32) a block, tiles dealt
// round the eight XCDs so a tile's items share one L2. SPLIT lands each K
// slice's f32 partial for `mgemm_finalize`.
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
    batch_tile<BatchDense<bf16, SBY>, bf16, MT, 1, 3, SWIGLU, NT>(
        act, nullptr, begin, min(MT, m - begin), {w}, nullptr, y, 1, n, k, tile, split * per, (split + 1) * per,
        slice);
}


// ------------------------------------------------ the ordered routed MLP --

// The routed MLP's activation over rows in expert order: h[i] = silu(g) · u
// of packed[i] = [g | u] (g capped above at `limit`, u held to ±limit, when
// CLAMP), for the live routes i < offsets[E]. Grid (route cap, ⌈I / (8 ·
// 256)⌉), 256 threads, eight elements a thread; I a multiple of 8.
template <class T, bool CLAMP>
__global__ __launch_bounds__(256) void moe_swiglu_ordered(const T* __restrict__ packed, T* __restrict__ h,
                                                          const i32* __restrict__ offsets, int num_experts, int I,
                                                          float limit) {
    const int i = blockIdx.x;
    if (i >= offsets[num_experts]) return;
    const int c = (blockIdx.y * 256 + threadIdx.x) * 8;
    if (c >= I) return;
    const long long row = static_cast<long long>(i);
    const Pack8<T> g = *reinterpret_cast<const Pack8<T>*>(packed + row * 2 * I + c);
    const Pack8<T> u = *reinterpret_cast<const Pack8<T>*>(packed + row * 2 * I + I + c);
    Pack8<T> o;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        float gv = to_f32(g.v[e]), uv = to_f32(u.v[e]);
        if constexpr (CLAMP) {
            gv = fminf(gv, limit);
            uv = fminf(fmaxf(uv, -limit), limit);
            o.v[e] = from_f32<T>(gv / (1.f + expf(-gv)) * uv);
        } else {
            o.v[e] = from_f32<T>(gv * (1.f / (1.f + __expf(-gv))) * uv);
        }
    }
    *reinterpret_cast<Pack8<T>*>(h + row * I + c) = o;
}

// The hipBLASLt grouped leg (hip/lt_grouped.rs): routes in expert order
// from moe_route_order. Each expert's `UserArguments` (49 words: m its
// weight rows, n its routes, k; a/b/d its planes) is written here, an n = 0
// sentinel after the last so a worst-case grid's spare groups end there.
// One UserArguments record: d[c, n] = b[c, k] · a[n, k]ᵀ (c = 0 with n = 0
// is the sentinel a worst-case grid's spare groups end at).
template <class T>
__device__ __forceinline__ void lt_record(u32* __restrict__ rec, u32 n, u32 c, u32 k, u64 d, u64 a, u64 b) {
    u32 w[49];
#pragma unroll
    for (int i = 0; i < 49; ++i) w[i] = 0;
    if (n != 0) {
        w[0] = n;
        w[1] = c;
        w[2] = 1;
        w[3] = k;
        w[4] = static_cast<u32>(d);
        w[5] = static_cast<u32>(d >> 32);
        w[6] = static_cast<u32>(d);
        w[7] = static_cast<u32>(d >> 32);
        w[8] = static_cast<u32>(a);
        w[9] = static_cast<u32>(a >> 32);
        w[10] = static_cast<u32>(b);
        w[11] = static_cast<u32>(b >> 32);
        w[12] = n;
        w[13] = n * c;
        w[14] = n;
        w[15] = n * c;
        w[16] = k;
        w[17] = n * k;
        w[18] = k;
        w[19] = k * c;
        w[20] = 0x3f800000u;  // alpha = 1.0f; beta (word 24) stays 0
    }
#pragma unroll
    for (int i = 0; i < 49; ++i) rec[i] = w[i];
}

template <class T>
__device__ __forceinline__ void lt_write_args(const i32* __restrict__ offsets, u32* __restrict__ args, u64 bank,
                                              u64 staged_in, u64 staged_out, int n, int k, int num_experts) {
    for (int e = threadIdx.x; e <= num_experts; e += blockDim.x) {
        u32* rec = args + static_cast<long long>(e) * 49;
        if (e == num_experts) {
            lt_record<T>(rec, 0, 0, 0, 0, 0, 0);
            continue;
        }
        const u32 off = static_cast<u32>(offsets[e]);
        const u32 c = static_cast<u32>(offsets[e + 1]) - off;
        lt_record<T>(rec, n, c, k, staged_out + static_cast<u64>(off) * n * sizeof(T),
                     bank + static_cast<u64>(e) * n * k * sizeof(T), staged_in + static_cast<u64>(off) * k * sizeof(T));
    }
}

// The routing scatter: the routes' rows gathered by expert into `staged`
// (row p = act[order[p] / act_div]) and, when `pos` is given, each route's
// row there (pos[order[p]] = p); block 0 writes the arguments as well. Rows
// move in 16-byte pieces (k a multiple of 8), a grid-stride over the live
// routes.
template <class T>
__global__ __launch_bounds__(256) void moe_lt_scatter(const T* __restrict__ act, const i32* __restrict__ order,
                                                      const i32* __restrict__ offsets, T* __restrict__ staged,
                                                      u32* __restrict__ args, i32* __restrict__ pos, u64 bank,
                                                      u64 staged_out, int act_div, int n, int k, int num_experts) {
    if (blockIdx.x == 0)
        lt_write_args<T>(offsets, args, bank, reinterpret_cast<u64>(staged), staged_out, n, k, num_experts);
    using V = uint4;
    const int pieces = k / 8;
    const long long routed = offsets[num_experts];
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    const long long first = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pos != nullptr)
        for (long long p = first; p < routed; p += stride) pos[order[p]] = static_cast<i32>(p);
    for (long long i = first; i < routed * pieces; i += stride) {
        const long long p = i / pieces;
        const int c = static_cast<int>(i - p * pieces);
        const long long src = static_cast<long long>(order[p] / act_div) * k;
        reinterpret_cast<V*>(staged + p * k)[c] = reinterpret_cast<const V*>(act + src)[c];
    }
}

// The arguments alone, for a leg whose rows are already by expert (the down
// projection of the sorted routed MLP). One block.
template <class T>
__global__ __launch_bounds__(256) void moe_lt_args(const i32* __restrict__ offsets, u32* __restrict__ args, u64 bank,
                                                   u64 staged_in, u64 staged_out, int n, int k, int num_experts) {
    lt_write_args<T>(offsets, args, bank, staged_in, staged_out, n, k, num_experts);
}

// A projection over the live rows of a padded plane (an lm head over a
// body's readouts): `live` names how many of `rows` are, and launch r of
// three grouped GEMMs takes them when they number in (bounds[r], bounds[r +
// 1]], the others none; a record and its sentinel a launch.
template <class T>
__global__ __launch_bounds__(64) void dense_lt_live_args(const u32* __restrict__ live, u32* __restrict__ args, u64 w,
                                                         u64 act, u64 y, int n, int k, int rows, int b1, int b2) {
    const int m = min(static_cast<int>(*live), rows);
    const int bounds[4] = {0, b1, b2, rows};
    for (int r = threadIdx.x; r < 6; r += blockDim.x) {
        const int at = r / 2;
        const u32 c = m > bounds[at] && m <= bounds[at + 1] ? static_cast<u32>(m) : 0u;
        lt_record<T>(args + r * 49, r % 2 == 0 ? static_cast<u32>(n) : 0u, c, k, y, w, act);
    }
}

// The sorted routed MLP's fold: y[n][h] = Σ_k weights[n][k] ·
// sorted[pos[n·top_k + k]][h], summed in f32 and rounded once; a dead route
// (expert < 0) adds nothing. As moe_weighted_sum, over the staged window.
template <class T>
__global__ void moe_lt_fold(T* __restrict__ out, const T* __restrict__ sorted, const i32* __restrict__ pos,
                            const i32* __restrict__ routes, const float* __restrict__ weights, int top_k, int hidden,
                            const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long n = window.plane(blockIdx.x);
    const int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (h >= hidden) return;
    const long long base = n * top_k;
    float acc = 0.f;
    for (int k = 0; k < top_k; ++k) {
        if (routes[base + k] < 0) continue;
        acc += weights[base + k] * to_f32(sorted[static_cast<long long>(pos[base + k]) * hidden + h]);
    }
    out[n * hidden + h] = from_f32<T>(acc);
}

}  // namespace pie::linear
