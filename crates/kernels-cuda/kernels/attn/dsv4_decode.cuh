#pragma once

// DeepSeek-V4's attention over a decode-width batch in one launch: a row's
// sliding window of the shared key/value plane (`prefill_lse` over the
// window), its compressed rows (`pool_lse`, or `pool_lse_selected` over the
// ids an indexer picked), the two folded (`merge_lse`) and the sinks (`sink`).
// Every query head reads the same keys (one kv head, keys and values one
// plane), so a block holds a row's 64 heads and walks the row's keys 32 at a
// time: the window's last `window` positions, then the compressed rows.
//
// Eight warps: warp w's 16 heads are the (w % 4)th, its half of the head
// width the (w / 4)th. S = Q K^T over a half lands in shared memory, the
// partner warp (the other half) adds its own, both keep the same online
// softmax, and O += P K over the warp's half of the width.
//
// A cluster of `ranks` blocks (along x) splits a row's keys; the blocks fold
// their running max, sum and output through distributed shared memory, each
// rank landing a slice of the head width with the sinks applied.

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <math_constants.h>

#include "prelude/device.cuh"

namespace pie::attn::dsv4 {

constexpr int kHeads = 64;
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kKeys = 32;
constexpr int kStages = 3;
constexpr float kLog2e = 1.44269504088896340736f;

template <int D>
struct Shape {
    static constexpr int kLd = D + 8;
    static constexpr int kQBytes = kHeads * kLd * 2;
    static constexpr int kTileBytes = kKeys * kLd * 2;
    static constexpr int kSBytes = kWarps * 16 * kKeys * 4;
    static constexpr int kBytes = kQBytes + kStages * kTileBytes + kSBytes;
    // The fold's output plane reuses Q and the ring.
    static_assert(kHeads * D * 4 <= kQBytes + kStages * kTileBytes, "the fold fits");
};

__device__ __forceinline__ std::uint32_t saddr(const void* p) {
    return static_cast<std::uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ std::uint32_t pack2(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const std::uint32_t*>(&v);
}

__device__ __forceinline__ void mma16816(float (&d)[4], const std::uint32_t (&a)[4],
                                         std::uint32_t b0, std::uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldsm4(std::uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(saddr(p)));
}

__device__ __forceinline__ void ldsm4t(std::uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(saddr(p)));
}

__device__ __forceinline__ float ld_rank(const float* at, unsigned rank) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    unsigned remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(saddr(at)), "r"(rank));
    float v;
    asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(remote) : "memory");
    return v;
#else
    return rank == 0 ? *at : 0.f;
#endif
}

__device__ __forceinline__ float4 ld_rank4(const float* at, unsigned rank) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    unsigned remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(saddr(at)), "r"(rank));
    float4 v;
    asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                 : "r"(remote)
                 : "memory");
    return v;
#else
    return rank == 0 ? *reinterpret_cast<const float4*>(at) : make_float4(0.f, 0.f, 0.f, 0.f);
#endif
}

__device__ __forceinline__ void sync_ranks() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::
                     : "memory");
#else
    __syncthreads();
#endif
}

__device__ __forceinline__ unsigned rank_in_cluster() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    unsigned r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
#else
    return 0;
#endif
}

__device__ __forceinline__ long long slot_of(const u32* __restrict__ indices,
                                             const u32* __restrict__ indptr, int req, int pos,
                                             int page_size) {
    const u32 page = indices[indptr[req] + pos / page_size];
    return static_cast<long long>(page) * page_size + (pos % page_size);
}

// `ratio` zero: no compressed rows (the window and the sinks alone).
// `selection` null with a ratio: every compressed row the query sees;
// otherwise the row's `top_k` ids, ended early by a negative one.
template <int D, class SinkT>
__global__ __launch_bounds__(kThreads, 1) void decode(
    const bf16* __restrict__ q,
    const bf16* __restrict__ win_keys,
    const u32* __restrict__ win_indices,
    const u32* __restrict__ win_indptr,
    long long win_stride,
    int win_page_size,
    int window,
    const bf16* __restrict__ pool_keys,
    const u32* __restrict__ pool_indices,
    const u32* __restrict__ pool_indptr,
    long long pool_stride,
    int pool_page_size,
    int ratio,
    const i32* __restrict__ selection,
    int top_k,
    const i32* __restrict__ positions,
    const i32* __restrict__ req_of_token,
    const SinkT* __restrict__ sinks,
    bf16* __restrict__ o,
    float scale,
    const u32* __restrict__ win)
{
    using S = Shape<D>;
    constexpr int kLd = S::kLd;
    constexpr int kHalf = D / 2;
    constexpr int kChunks = D / 8;            // 16-byte pieces a key row
    constexpr int kPerKey = kThreads / kKeys; // threads a key row
    static_assert(kChunks % kPerKey == 0, "a key row splits evenly");
    static_assert(kHalf % 16 == 0, "a half width is whole k16 steps");

    const int qi = static_cast<int>(blockIdx.y);
    if (win != nullptr && qi >= static_cast<int>(win[0])) return;
    const int row = win != nullptr ? qi + static_cast<int>(win[1]) : qi;
    const unsigned ranks = gridDim.x;
    const unsigned rank = ranks > 1 ? rank_in_cluster() : 0;

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int mt = warp & 3;
    const int dh = warp >> 2;
    const int g = lane >> 2, p = (lane & 3) << 1;

    extern __shared__ __align__(128) char smem_raw[];
    bf16* sQ = reinterpret_cast<bf16*>(smem_raw);
    bf16* sRing = reinterpret_cast<bf16*>(smem_raw + S::kQBytes);
    float* sS = reinterpret_cast<float*>(smem_raw + S::kQBytes + kStages * S::kTileBytes);
    __shared__ unsigned char s_valid[kStages][kKeys];
    __shared__ int s_end;
    __shared__ float s_ml[2][kHeads];

    const int req = req_of_token[row];
    const int qpos = positions[row];
    const int n_win = max(0, min(qpos + 1, window));
    const int num_visible = ratio > 0 ? max((qpos + 1) / ratio, 0) : 0;
    const i32* srow =
        selection != nullptr ? selection + static_cast<long long>(row) * top_k : nullptr;

    if (tid == 0) s_end = srow != nullptr ? top_k : num_visible;
    __syncthreads();
    if (srow != nullptr && num_visible > 0) {
        for (int n = tid; n < top_k; n += kThreads)
            if (srow[n] < 0) atomicMin(&s_end, n);
    }
    for (int c = tid; c < kHeads * kChunks; c += kThreads) {
        const int h = c / kChunks;
        const int d = (c % kChunks) * 8;
        __pipeline_memcpy_async(sQ + h * kLd + d,
                                q + (static_cast<long long>(row) * kHeads + h) * D + d, 16);
    }
    __pipeline_commit();
    __syncthreads();
    const int n_pool = num_visible > 0 ? s_end : 0;
    const int total = n_win + n_pool;
    const int tiles = (total + kKeys - 1) / kKeys;
    const int per = (tiles + static_cast<int>(ranks) - 1) / static_cast<int>(ranks);
    const int t0 = min(tiles, static_cast<int>(rank) * per);
    const int t1 = min(tiles, t0 + per);

    // Tile t's keys into ring slot `buf`: a key row per kPerKey threads.
    auto stage = [&](int t, int buf) {
        const int key = tid / kPerKey;
        const int part = tid % kPerKey;
        const int n = t * kKeys + key;
        const bf16* src = nullptr;
        if (n < n_win) {
            const int pos = qpos - n_win + 1 + n;
            src = win_keys + slot_of(win_indices, win_indptr, req, pos, win_page_size) * win_stride;
        } else if (n < total) {
            const int j = n - n_win;
            const int c = srow != nullptr ? srow[j] : j;
            if (c >= 0 && c < num_visible)
                src = pool_keys +
                      slot_of(pool_indices, pool_indptr, req, (c + 1) * ratio - 1, pool_page_size) *
                          pool_stride;
        }
        if (part == 0) s_valid[buf][key] = src != nullptr;
        bf16* dst = sRing + buf * kKeys * kLd + key * kLd;
#pragma unroll
        for (int i = 0; i < kChunks / kPerKey; ++i) {
            const int d = (part + i * kPerKey) * 8;
            if (src != nullptr) {
                __pipeline_memcpy_async(dst + d, src + d, 16);
            } else {
                *reinterpret_cast<int4*>(dst + d) = make_int4(0, 0, 0, 0);
            }
        }
    };

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (t0 + s < t1) stage(t0 + s, s);
        __pipeline_commit();
    }

    float oacc[kHalf / 8][4];
#pragma unroll
    for (int n = 0; n < kHalf / 8; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) oacc[n][i] = 0.f;
    // Log2 domain: scores carry scale * log2(e).
    float m_lo = -CUDART_INF_F, m_hi = -CUDART_INF_F, l_lo = 0.f, l_hi = 0.f;
    const float sl2 = scale * kLog2e;
    const bf16* qa = sQ + mt * 16 * kLd;
    float* mine = sS + warp * 16 * kKeys;
    const float* theirs = sS + (warp ^ 4) * 16 * kKeys;

    for (int t = t0; t < t1; ++t) {
        const int buf = (t - t0) % kStages;
        __pipeline_wait_prior(kStages - 2);
        __syncthreads();
        if (t + kStages - 1 < t1) stage(t + kStages - 1, (t - t0 + kStages - 1) % kStages);
        __pipeline_commit();
        const bf16* sK = sRing + buf * kKeys * kLd;

        // This warp's half of S: 16 heads x 32 keys.
        float sacc[4][4];
#pragma unroll
        for (int n = 0; n < 4; ++n)
#pragma unroll
            for (int i = 0; i < 4; ++i) sacc[n][i] = 0.f;
#pragma unroll 4
        for (int k0 = dh * kHalf; k0 < (dh + 1) * kHalf; k0 += 16) {
            std::uint32_t a[4];
            ldsm4(a, qa + (lane & 15) * kLd + k0 + ((lane & 16) ? 8 : 0));
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                const int j = lane >> 3;
                std::uint32_t b[4];
                ldsm4(b, sK + (np * 16 + (j >> 1) * 8 + (lane & 7)) * kLd + k0 + (j & 1) * 8);
                mma16816(sacc[2 * np], a, b[0], b[1]);
                mma16816(sacc[2 * np + 1], a, b[2], b[3]);
            }
        }
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            *reinterpret_cast<float2*>(mine + g * kKeys + nt * 8 + p) =
                make_float2(sacc[nt][0], sacc[nt][1]);
            *reinterpret_cast<float2*>(mine + (g + 8) * kKeys + nt * 8 + p) =
                make_float2(sacc[nt][2], sacc[nt][3]);
        }
        __syncthreads();
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const float2 lo = *reinterpret_cast<const float2*>(theirs + g * kKeys + nt * 8 + p);
            const float2 hi = *reinterpret_cast<const float2*>(theirs + (g + 8) * kKeys + nt * 8 + p);
            const bool v0 = s_valid[buf][nt * 8 + p], v1 = s_valid[buf][nt * 8 + p + 1];
            sacc[nt][0] = v0 ? (sacc[nt][0] + lo.x) * sl2 : -CUDART_INF_F;
            sacc[nt][1] = v1 ? (sacc[nt][1] + lo.y) * sl2 : -CUDART_INF_F;
            sacc[nt][2] = v0 ? (sacc[nt][2] + hi.x) * sl2 : -CUDART_INF_F;
            sacc[nt][3] = v1 ? (sacc[nt][3] + hi.y) * sl2 : -CUDART_INF_F;
        }
        float mx_lo = -CUDART_INF_F, mx_hi = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            mx_lo = fmaxf(mx_lo, fmaxf(sacc[nt][0], sacc[nt][1]));
            mx_hi = fmaxf(mx_hi, fmaxf(sacc[nt][2], sacc[nt][3]));
        }
#pragma unroll
        for (int sh = 1; sh < 4; sh <<= 1) {
            mx_lo = fmaxf(mx_lo, __shfl_xor_sync(0xffffffffu, mx_lo, sh));
            mx_hi = fmaxf(mx_hi, __shfl_xor_sync(0xffffffffu, mx_hi, sh));
        }
        const float mn_lo = fmaxf(m_lo, mx_lo), mn_hi = fmaxf(m_hi, mx_hi);
        const float b_lo = mn_lo == -CUDART_INF_F ? 0.f : mn_lo;
        const float b_hi = mn_hi == -CUDART_INF_F ? 0.f : mn_hi;
        const float c_lo = exp2f(m_lo - b_lo), c_hi = exp2f(m_hi - b_hi);
        m_lo = mn_lo;
        m_hi = mn_hi;
        float rl = 0.f, rh = 0.f;
        std::uint32_t pa[2][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const float p0 = exp2f(sacc[nt][0] - b_lo), p1 = exp2f(sacc[nt][1] - b_lo);
            const float p2 = exp2f(sacc[nt][2] - b_hi), p3 = exp2f(sacc[nt][3] - b_hi);
            rl += p0 + p1;
            rh += p2 + p3;
            pa[nt >> 1][(nt & 1) * 2] = pack2(p0, p1);
            pa[nt >> 1][(nt & 1) * 2 + 1] = pack2(p2, p3);
        }
#pragma unroll
        for (int sh = 1; sh < 4; sh <<= 1) {
            rl += __shfl_xor_sync(0xffffffffu, rl, sh);
            rh += __shfl_xor_sync(0xffffffffu, rh, sh);
        }
        l_lo = l_lo * c_lo + rl;
        l_hi = l_hi * c_hi + rh;
#pragma unroll
        for (int n = 0; n < kHalf / 8; ++n) {
            oacc[n][0] *= c_lo;
            oacc[n][1] *= c_lo;
            oacc[n][2] *= c_hi;
            oacc[n][3] *= c_hi;
        }
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            const int j = lane >> 3;
            const bf16* vrow = sK + (kk * 16 + (j & 1) * 8 + (lane & 7)) * kLd + dh * kHalf + (j >> 1) * 8;
#pragma unroll
            for (int np = 0; np < kHalf / 16; ++np) {
                std::uint32_t b[4];
                ldsm4t(b, vrow + np * 16);
                mma16816(oacc[2 * np], pa[kk], b[0], b[1]);
                mma16816(oacc[2 * np + 1], pa[kk], b[2], b[3]);
            }
        }
    }
    __pipeline_wait_prior(0);

    const float sink_lo = Elem<SinkT>::to_f32(sinks[mt * 16 + g]) * kLog2e;
    const float sink_hi = Elem<SinkT>::to_f32(sinks[mt * 16 + g + 8]) * kLog2e;
    const int h_lo = mt * 16 + g, h_hi = h_lo + 8;
    bf16* orow = o + static_cast<long long>(row) * kHeads * D;

    if (ranks == 1) {
        const float top_lo = fmaxf(m_lo, sink_lo), top_hi = fmaxf(m_hi, sink_hi);
        const float f_lo = exp2f(m_lo - top_lo), f_hi = exp2f(m_hi - top_hi);
        const float den_lo = l_lo * f_lo + exp2f(sink_lo - top_lo);
        const float den_hi = l_hi * f_hi + exp2f(sink_hi - top_hi);
        const float s_lo = den_lo > 0.f ? f_lo / den_lo : 0.f;
        const float s_hi = den_hi > 0.f ? f_hi / den_hi : 0.f;
#pragma unroll
        for (int n = 0; n < kHalf / 8; ++n) {
            const int d = dh * kHalf + n * 8 + p;
            *reinterpret_cast<std::uint32_t*>(orow + h_lo * D + d) =
                pack2(oacc[n][0] * s_lo, oacc[n][1] * s_lo);
            *reinterpret_cast<std::uint32_t*>(orow + h_hi * D + d) =
                pack2(oacc[n][2] * s_hi, oacc[n][3] * s_hi);
        }
        return;
    }

    // The ranks' fold: every rank's (max, sum) a head, then its output scaled
    // to the common max over the common sum, in its own shared memory.
    if (dh == 0 && (lane & 3) == 0) {
        s_ml[0][h_lo] = m_lo;
        s_ml[0][h_hi] = m_hi;
        s_ml[1][h_lo] = l_lo;
        s_ml[1][h_hi] = l_hi;
    }
    sync_ranks();
    float top_lo = sink_lo, top_hi = sink_hi;
    for (unsigned r = 0; r < ranks; ++r) {
        top_lo = fmaxf(top_lo, ld_rank(&s_ml[0][h_lo], r));
        top_hi = fmaxf(top_hi, ld_rank(&s_ml[0][h_hi], r));
    }
    float den_lo = exp2f(sink_lo - top_lo), den_hi = exp2f(sink_hi - top_hi);
    for (unsigned r = 0; r < ranks; ++r) {
        const float rm_lo = ld_rank(&s_ml[0][h_lo], r), rm_hi = ld_rank(&s_ml[0][h_hi], r);
        const float rl_lo = ld_rank(&s_ml[1][h_lo], r), rl_hi = ld_rank(&s_ml[1][h_hi], r);
        if (rl_lo > 0.f) den_lo += rl_lo * exp2f(rm_lo - top_lo);
        if (rl_hi > 0.f) den_hi += rl_hi * exp2f(rm_hi - top_hi);
    }
    const float s_lo = l_lo > 0.f ? exp2f(m_lo - top_lo) / den_lo : 0.f;
    const float s_hi = l_hi > 0.f ? exp2f(m_hi - top_hi) / den_hi : 0.f;
    float* sO = reinterpret_cast<float*>(smem_raw);
#pragma unroll
    for (int n = 0; n < kHalf / 8; ++n) {
        const int d = dh * kHalf + n * 8 + p;
        *reinterpret_cast<float2*>(sO + h_lo * D + d) = make_float2(oacc[n][0] * s_lo, oacc[n][1] * s_lo);
        *reinterpret_cast<float2*>(sO + h_hi * D + d) = make_float2(oacc[n][2] * s_hi, oacc[n][3] * s_hi);
    }
    sync_ranks();
    // This rank's slice of the width, summed over the ranks.
    const int slice = D / static_cast<int>(ranks);
    const int quads = kHeads * slice / 4;
    for (int i = tid; i < quads; i += kThreads) {
        const int h = i / (slice / 4);
        const int d = static_cast<int>(rank) * slice + (i % (slice / 4)) * 4;
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (unsigned r = 0; r < ranks; ++r) {
            const float4 v = ld_rank4(sO + h * D + d, r);
            acc.x += v.x;
            acc.y += v.y;
            acc.z += v.z;
            acc.w += v.w;
        }
        uint2 packed;
        packed.x = pack2(acc.x, acc.y);
        packed.y = pack2(acc.z, acc.w);
        *reinterpret_cast<uint2*>(orow + h * D + d) = packed;
    }
    // No block leaves while another still reads its shared memory.
    sync_ranks();
}

}  // namespace pie::attn::dsv4
