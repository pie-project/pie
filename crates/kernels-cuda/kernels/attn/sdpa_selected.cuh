#pragma once

#include <cstdint>
#include <cstring>

#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <math_constants.h>

#include "prim/device.cuh"

namespace pie::attn::sdpa_selected {

constexpr int kBlock = 256;
constexpr int kWarps = kBlock / 32;
constexpr int kMaxPer = 16;

// Paged GQA attention over the blocks a selection names, plus the open block
// that runs to the query. `attention.decode`/`attention.prefill` are FlashInfer
// and take no per-row selection, so this walk is written out: the reduction is
// `mla_naive_paged_kernel`'s, and only the key loop and the K/V addressing
// differ (two planes at `(slot * n_kv_heads + kv_head) * D`, V accumulated
// rather than K). With `per_head` the selection row holds one run of `top_k`
// blocks per kv head, and a head's queries read their kv head's run.
template <class T>
__global__ void sdpa_paged_selected_kernel(
    const T* __restrict__ queries,
    const T* __restrict__ k_pages,
    const T* __restrict__ v_pages,
    const u32* __restrict__ qo_indptr,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    const i32* __restrict__ selection,
    T* __restrict__ o,
    i32 R, i32 H, i32 KVH, i32 D, i32 page_size, float sm_scale,
    i32 top_k, i32 ratio, i32 G, i32 per_head,
    const u32* __restrict__ win)
{
    const int t = static_cast<int>(blockIdx.x);

    if (win != nullptr && t >= static_cast<int>(win[0])) return;

    const int t_row = win != nullptr ? t + static_cast<int>(win[1]) : t;

    if (win != nullptr && static_cast<int>(win[2]) < R) R = static_cast<int>(win[2]);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int K = kWarps / G;
    const int g = warp / K;
    const int s = warp % K;
    const int h = static_cast<int>(blockIdx.y) * G + g;
    const int kv_head = h / (H / KVH);
    const int per = D / 32;

    int lo = 0, hi = R - 1;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (t < static_cast<int>(qo_indptr[mid + 1])) hi = mid; else lo = mid + 1;
    }
    const int r = lo;
    const int qo_lo = static_cast<int>(qo_indptr[r]);
    const int new_tokens = static_cast<int>(qo_indptr[r + 1]) - qo_lo;
    const int pages_first = static_cast<int>(kv_page_indptr[r]);
    const int num_pages = static_cast<int>(kv_page_indptr[r + 1]) - pages_first;
    const int kv_len =
        (num_pages - 1) * page_size + static_cast<int>(kv_last_page_lens[r]);
    const int j_end = kv_len - new_tokens + (t - qo_lo) + 1;

    const int nblocks = (ratio > 0) ? j_end / ratio : 0;
    const int blocks = min(top_k, nblocks);
    const i32* srow = per_head
        ? selection + (static_cast<long long>(t_row) * KVH + kv_head) * top_k
        : selection + static_cast<long long>(t_row) * top_k;

    extern __shared__ float smem[];
    float* wacc = smem;
    float* wm   = wacc + kWarps * D;
    float* wl   = wm + kWarps;

    const T* qrow = queries + (static_cast<long long>(t_row) * H + h) * D;
    float qreg[kMaxPer];
    for (int i = 0; i < per; ++i) {
        qreg[i] = Elem<T>::to_f32(qrow[lane + i * 32]) * sm_scale;
    }

    float acc[kMaxPer];
    for (int i = 0; i < per; ++i) acc[i] = 0.f;
    float m = -CUDART_INF_F, lsum = 0.f;

    // A block below `nblocks` holds only cells below `nblocks * ratio <= j_end`,
    // so rejecting the block is the whole causal test.
    const int steps = blocks * ratio + (ratio - 1);
    for (int n = s; n < steps; n += K) {
        int j;
        if (n < blocks * ratio) {
            const int c = srow[n / ratio];
            if (c < 0 || c >= nblocks) continue;
            j = c * ratio + (n % ratio);
        } else {
            j = nblocks * ratio + (n - blocks * ratio);
            if (j >= j_end) continue;
        }
        const int page =
            static_cast<int>(kv_page_indices[pages_first + j / page_size]);
        const int off = j % page_size;
        const long long slot = static_cast<long long>(page) * page_size + off;
        const T* kj = k_pages + (slot * KVH + kv_head) * D;
        const T* vj = v_pages + (slot * KVH + kv_head) * D;

        float pd = 0.f;
        for (int i = 0; i < per; ++i) {
            pd += qreg[i] * Elem<T>::to_f32(kj[lane + i * 32]);
        }
        #pragma unroll
        for (int sh = 16; sh > 0; sh >>= 1) {
            pd += __shfl_xor_sync(0xffffffffu, pd, sh);
        }
        const float m_new = fmaxf(m, pd);
        const float corr = __expf(m - m_new);
        const float p = __expf(pd - m_new);
        lsum = lsum * corr + p;
        for (int i = 0; i < per; ++i) {
            acc[i] = acc[i] * corr + p * Elem<T>::to_f32(vj[lane + i * 32]);
        }
        m = m_new;
    }

    for (int i = 0; i < per; ++i) {
        wacc[warp * D + lane + i * 32] = acc[i];
    }
    if (lane == 0) { wm[warp] = m; wl[warp] = lsum; }
    __syncthreads();

    const int total_out = G * D;
    for (int idx = tid; idx < total_out; idx += kBlock) {
        const int gg = idx / D;
        const int d = idx % D;
        const int w0 = gg * K;
        float m_all = -CUDART_INF_F;
        for (int w = w0; w < w0 + K; ++w) m_all = fmaxf(m_all, wm[w]);
        float l_all = 0.f, v = 0.f;
        for (int w = w0; w < w0 + K; ++w) {
            if (wm[w] > -CUDART_INF_F) {
                const float e = __expf(wm[w] - m_all);
                l_all += wl[w] * e;
                v += wacc[w * D + d] * e;
            }
        }
        const float inv = (l_all > 0.f) ? (1.f / l_all) : 0.f;
        o[(static_cast<long long>(t_row) * H + static_cast<int>(blockIdx.y) * G + gg) * D + d] =
            Elem<T>::from_f32(v * inv);
    }
}

// The same walk on tensor cores: a block serves one query row's `G` heads of
// one kv head (`G` <= 16, an mma's M), each warp its own 16-key chunks of the
// row's selected and open keys, gathered into shared memory a chunk at a
// time, with its own running max, sum and output; the warps' partials fold
// at the end. A row reads exactly the keys the scalar walk reads.
namespace mma {

constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kKeys = 16;

__device__ __forceinline__ std::uint32_t pack2(float lo, float hi) {
    const __nv_bfloat16 l = __float2bfloat16(lo), h = __float2bfloat16(hi);
    std::uint16_t a, b;
    std::memcpy(&a, &l, 2);
    std::memcpy(&b, &h, 2);
    return static_cast<std::uint32_t>(a) | (static_cast<std::uint32_t>(b) << 16);
}

__device__ __forceinline__ void mma_m16n8k16(float (&d)[4], const std::uint32_t (&a)[4],
                                             const std::uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ std::uint32_t smem_addr(const void* p) {
    return static_cast<std::uint32_t>(__cvta_generic_to_shared(p));
}

// A: 16 rows of `base` (row pitch `ld`), 16 columns from `k0`.
__device__ __forceinline__ void ld_a(std::uint32_t (&a)[4], const __nv_bfloat16* base, int ld,
                                     int lane, int k0) {
    const __nv_bfloat16* r = base + (lane & 15) * ld + k0 + ((lane & 16) ? 8 : 0);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(smem_addr(r)));
}

// B of S = Q K^T: 8 keys (rows of `base`) by 16 dims from `k0`.
__device__ __forceinline__ void ld_b_k(std::uint32_t (&b)[2], const __nv_bfloat16* base, int ld,
                                       int lane, int k0) {
    const __nv_bfloat16* r = base + (lane & 7) * ld + k0 + ((lane & 8) ? 8 : 0);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(b[0]), "=r"(b[1])
                 : "r"(smem_addr(r)));
}

// B of O = P V: 16 keys (rows of `base`) by 8 dims from `n0`.
__device__ __forceinline__ void ld_b_v(std::uint32_t (&b)[2], const __nv_bfloat16* base, int ld,
                                       int lane, int n0) {
    const __nv_bfloat16* r = base + (lane & 15) * ld + n0;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(b[0]), "=r"(b[1])
                 : "r"(smem_addr(r)));
}

template <int D>
struct Shape {
    static constexpr int kLd = D + 8;
    static constexpr int kNTiles = D / 8;
    // Q, then each warp's K and V chunk (its partial output once it is done),
    // then each warp's running max and sum a row.
    static constexpr int kQBytes = 16 * kLd * 2;
    static constexpr int kWarpBytes = 2 * kKeys * kLd * 2;
    static_assert(kWarpBytes >= 16 * D * 4, "a warp's output fits where its chunks were");
    static constexpr int kBytes = kQBytes + kWarps * kWarpBytes + kWarps * 16 * 2 * 4;
};

template <int D>
__global__ __launch_bounds__(kThreads) void sdpa_paged_selected_mma_kernel(
    const __nv_bfloat16* __restrict__ queries,
    const __nv_bfloat16* __restrict__ k_pages,
    const __nv_bfloat16* __restrict__ v_pages,
    const u32* __restrict__ qo_indptr,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    const i32* __restrict__ selection,
    __nv_bfloat16* __restrict__ o,
    i32 R, i32 H, i32 KVH, i32 page_size, float sm_scale,
    i32 top_k, i32 ratio, i32 per_head,
    const u32* __restrict__ win)
{
    using S = Shape<D>;
    constexpr int kLd = S::kLd;
    constexpr int kNTiles = S::kNTiles;
    constexpr int kChunks = D / 8;
    const int t = static_cast<int>(blockIdx.x);
    if (win != nullptr && t >= static_cast<int>(win[0])) return;
    const int t_row = win != nullptr ? t + static_cast<int>(win[1]) : t;
    if (win != nullptr && static_cast<int>(win[2]) < R) R = static_cast<int>(win[2]);

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int kv_head = static_cast<int>(blockIdx.y);
    const int G = H / KVH;

    extern __shared__ __align__(16) char smem_raw[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    char* warp_base = smem_raw + S::kQBytes + warp * S::kWarpBytes;
    __nv_bfloat16* sK = reinterpret_cast<__nv_bfloat16*>(warp_base);
    __nv_bfloat16* sV = sK + kKeys * kLd;
    float* sML = reinterpret_cast<float*>(smem_raw + S::kQBytes + kWarps * S::kWarpBytes);
    __shared__ long long s_slot[kWarps][kKeys];

    int lo = 0, hi = R - 1;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (t < static_cast<int>(qo_indptr[mid + 1])) hi = mid; else lo = mid + 1;
    }
    const int r = lo;
    const int qo_lo = static_cast<int>(qo_indptr[r]);
    const int new_tokens = static_cast<int>(qo_indptr[r + 1]) - qo_lo;
    const int pages_first = static_cast<int>(kv_page_indptr[r]);
    const int num_pages = static_cast<int>(kv_page_indptr[r + 1]) - pages_first;
    const int kv_len = (num_pages - 1) * page_size + static_cast<int>(kv_last_page_lens[r]);
    const int j_end = kv_len - new_tokens + (t - qo_lo) + 1;
    const int nblocks = (ratio > 0) ? j_end / ratio : 0;
    const int blocks = min(top_k, nblocks);
    const i32* srow = per_head
        ? selection + (static_cast<long long>(t_row) * KVH + kv_head) * top_k
        : selection + static_cast<long long>(t_row) * top_k;
    const int steps = blocks * ratio + (ratio - 1);

    // The kv head's `G` query heads; the rest of the 16 rows zero.
    for (int c = tid; c < 16 * kChunks; c += kThreads) {
        const int row = c / kChunks;
        const int d = (c % kChunks) * 8;
        int4 v = make_int4(0, 0, 0, 0);
        if (row < G) {
            v = *reinterpret_cast<const int4*>(
                queries + (static_cast<long long>(t_row) * H + kv_head * G + row) * D + d);
        }
        *reinterpret_cast<int4*>(sQ + row * kLd + d) = v;
    }
    __syncthreads();

    const int g = lane >> 2, p = (lane & 3) << 1;
    float oacc[kNTiles][4];
#pragma unroll
    for (int n = 0; n < kNTiles; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) oacc[n][i] = 0.f;
    }
    float m_lo = -CUDART_INF_F, m_hi = -CUDART_INF_F, l_lo = 0.f, l_hi = 0.f;

    for (int base = warp * kKeys; base < steps; base += kWarps * kKeys) {
        if (lane < kKeys) {
            const int n = base + lane;
            int j = -1;
            if (n < blocks * ratio) {
                const int c = srow[n / ratio];
                if (c >= 0 && c < nblocks) j = c * ratio + (n % ratio);
            } else if (n < steps) {
                j = nblocks * ratio + (n - blocks * ratio);
                if (j >= j_end) j = -1;
            }
            s_slot[warp][lane] = j < 0
                ? -1
                : static_cast<long long>(kv_page_indices[pages_first + j / page_size]) * page_size
                      + j % page_size;
        }
        __syncwarp();
        for (int c = lane; c < kKeys * kChunks; c += 32) {
            const int key = c / kChunks;
            const int d = (c % kChunks) * 8;
            const long long slot = s_slot[warp][key];
            __nv_bfloat16* kd = sK + key * kLd + d;
            __nv_bfloat16* vd = sV + key * kLd + d;
            if (slot >= 0) {
                const long long at = (slot * KVH + kv_head) * D + d;
                __pipeline_memcpy_async(kd, k_pages + at, 16);
                __pipeline_memcpy_async(vd, v_pages + at, 16);
            } else {
                *reinterpret_cast<int4*>(kd) = make_int4(0, 0, 0, 0);
                *reinterpret_cast<int4*>(vd) = make_int4(0, 0, 0, 0);
            }
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncwarp();

        float sacc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll 4
        for (int k0 = 0; k0 < D; k0 += 16) {
            std::uint32_t a[4];
            ld_a(a, sQ, kLd, lane, k0);
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                std::uint32_t b[2];
                ld_b_k(b, sK + nt * 8 * kLd, kLd, lane, k0);
                mma_m16n8k16(sacc[nt], a, b);
            }
        }
        // Each thread holds keys `p`, `p + 1` of each 8-key tile, rows `g`
        // and `g + 8`.
        float mx_lo = -CUDART_INF_F, mx_hi = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const bool live = s_slot[warp][nt * 8 + p + i] >= 0;
                sacc[nt][i] = live ? sacc[nt][i] * sm_scale : -CUDART_INF_F;
                sacc[nt][2 + i] = live ? sacc[nt][2 + i] * sm_scale : -CUDART_INF_F;
                mx_lo = fmaxf(mx_lo, sacc[nt][i]);
                mx_hi = fmaxf(mx_hi, sacc[nt][2 + i]);
            }
        }
#pragma unroll
        for (int sh = 1; sh < 4; sh <<= 1) {
            mx_lo = fmaxf(mx_lo, __shfl_xor_sync(0xffffffffu, mx_lo, sh));
            mx_hi = fmaxf(mx_hi, __shfl_xor_sync(0xffffffffu, mx_hi, sh));
        }
        const float mn_lo = fmaxf(m_lo, mx_lo), mn_hi = fmaxf(m_hi, mx_hi);
        const float c_lo = mn_lo == -CUDART_INF_F ? 1.f : __expf(m_lo - mn_lo);
        const float c_hi = mn_hi == -CUDART_INF_F ? 1.f : __expf(m_hi - mn_hi);
        m_lo = mn_lo;
        m_hi = mn_hi;
        float pr[2][4];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                pr[nt][i] = mn_lo == -CUDART_INF_F ? 0.f : __expf(sacc[nt][i] - mn_lo);
                pr[nt][2 + i] = mn_hi == -CUDART_INF_F ? 0.f : __expf(sacc[nt][2 + i] - mn_hi);
            }
        }
        l_lo = l_lo * c_lo + pr[0][0] + pr[0][1] + pr[1][0] + pr[1][1];
        l_hi = l_hi * c_hi + pr[0][2] + pr[0][3] + pr[1][2] + pr[1][3];
        const std::uint32_t pa[4] = {
            pack2(pr[0][0], pr[0][1]),
            pack2(pr[0][2], pr[0][3]),
            pack2(pr[1][0], pr[1][1]),
            pack2(pr[1][2], pr[1][3]),
        };
#pragma unroll
        for (int n = 0; n < kNTiles; ++n) {
            oacc[n][0] *= c_lo;
            oacc[n][1] *= c_lo;
            oacc[n][2] *= c_hi;
            oacc[n][3] *= c_hi;
            std::uint32_t b[2];
            ld_b_v(b, sV, kLd, lane, n * 8);
            mma_m16n8k16(oacc[n], pa, b);
        }
        __syncwarp();
    }

    // Each warp's partial where its chunks were, then the fold across warps.
#pragma unroll
    for (int sh = 1; sh < 4; sh <<= 1) {
        l_lo += __shfl_xor_sync(0xffffffffu, l_lo, sh);
        l_hi += __shfl_xor_sync(0xffffffffu, l_hi, sh);
    }
    float* sO = reinterpret_cast<float*>(warp_base);
#pragma unroll
    for (int n = 0; n < kNTiles; ++n) {
        sO[g * D + n * 8 + p] = oacc[n][0];
        sO[g * D + n * 8 + p + 1] = oacc[n][1];
        sO[(g + 8) * D + n * 8 + p] = oacc[n][2];
        sO[(g + 8) * D + n * 8 + p + 1] = oacc[n][3];
    }
    if ((lane & 3) == 0) {
        sML[(warp * 16 + g) * 2] = m_lo;
        sML[(warp * 16 + g) * 2 + 1] = l_lo;
        sML[(warp * 16 + g + 8) * 2] = m_hi;
        sML[(warp * 16 + g + 8) * 2 + 1] = l_hi;
    }
    __syncthreads();
    for (int idx = tid; idx < G * D; idx += kThreads) {
        const int row = idx / D;
        const int d = idx % D;
        float m_all = -CUDART_INF_F;
        for (int w = 0; w < kWarps; ++w) m_all = fmaxf(m_all, sML[(w * 16 + row) * 2]);
        float l_all = 0.f, v = 0.f;
        if (m_all > -CUDART_INF_F) {
            for (int w = 0; w < kWarps; ++w) {
                const float mw = sML[(w * 16 + row) * 2];
                if (mw == -CUDART_INF_F) continue;
                const float e = __expf(mw - m_all);
                l_all += sML[(w * 16 + row) * 2 + 1] * e;
                const float* ow =
                    reinterpret_cast<const float*>(smem_raw + S::kQBytes + w * S::kWarpBytes);
                v += ow[row * D + d] * e;
            }
        }
        const float inv = l_all > 0.f ? 1.f / l_all : 0.f;
        o[(static_cast<long long>(t_row) * H + kv_head * G + row) * D + d] =
            __float2bfloat16(v * inv);
    }
}

// Rows that share their keys: a block serves `kTileRows` consecutive query
// rows of one request and one kv head, every (row, head) pair a row of the
// mma (`kTileRows * G` <= 16 * kTileWarps), and reads the union of the rows'
// selected blocks (and open blocks) once, a 16-key chunk at a time, each row
// masking the keys it did not select. A block a row names twice is read once.
template <int D>
struct Tile {
    static constexpr int kRows = 8;
    static constexpr int kWarps = 6;
    static constexpr int kThreads = kWarps * 32;
    static constexpr int kVectors = 16 * kWarps;
    static constexpr int kLd = D + 8;
    static constexpr int kNTiles = D / 8;
    static constexpr int kMaxBlocks = 4096;
    static constexpr int kWords = kMaxBlocks / 32;
    static constexpr int kQBytes = kVectors * kLd * 2;
    static constexpr int kStageBytes = 2 * kKeys * kLd * 2;
    // Q, two stages of keys and values, the rows' block maps, the union's
    // block list.
    static constexpr int kMapBytes = kRows * kWords * 4;
    static constexpr int kBytes = kQBytes + 2 * kStageBytes + kMapBytes + kMaxBlocks * 4;
};

template <int D>
__global__ __launch_bounds__(Tile<D>::kThreads) void sdpa_paged_selected_tile_kernel(
    const __nv_bfloat16* __restrict__ queries,
    const __nv_bfloat16* __restrict__ k_pages,
    const __nv_bfloat16* __restrict__ v_pages,
    const u32* __restrict__ qo_indptr,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    const i32* __restrict__ selection,
    __nv_bfloat16* __restrict__ o,
    i32 R, i32 H, i32 KVH, i32 page_size, float sm_scale,
    i32 top_k, i32 ratio, i32 per_head, i32 tokens,
    const u32* __restrict__ win)
{
    using S = Tile<D>;
    constexpr int kLd = S::kLd;
    constexpr int kNTiles = S::kNTiles;
    constexpr int kChunks = D / 8;
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int kv_head = static_cast<int>(blockIdx.y);
    const int G = H / KVH;
    const int live = win != nullptr ? min(tokens, static_cast<int>(win[0])) : tokens;
    const int shift = win != nullptr ? static_cast<int>(win[1]) : 0;
    if (win != nullptr && static_cast<int>(win[2]) < R) R = static_cast<int>(win[2]);
    const int t_first = static_cast<int>(blockIdx.x) * S::kRows;
    if (t_first >= live) return;
    const int t_last = min(t_first + S::kRows, live);

    extern __shared__ __align__(16) char smem_raw[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sKV = reinterpret_cast<__nv_bfloat16*>(smem_raw + S::kQBytes);
    auto rowbits = reinterpret_cast<u32 (*)[S::kWords]>(smem_raw + S::kQBytes + 2 * S::kStageBytes);
    int* list = reinterpret_cast<int*>(smem_raw + S::kQBytes + 2 * S::kStageBytes + S::kMapBytes);
    __shared__ u32 unionbits[S::kWords];
    __shared__ int word_start[S::kWords + 1];
    __shared__ int row_jend[S::kRows];
    __shared__ int g_req, g_end;
    __shared__ long long slot_s[2][kKeys];
    __shared__ int key_s[2][kKeys];

    const int g = lane >> 2, p = (lane & 3) << 1;

    // The tile's rows, a request at a time.
    for (int t0 = t_first; t0 < t_last;) {
        if (tid == 0) {
            int lo = 0, hi = R - 1;
            while (lo < hi) {
                const int mid = (lo + hi) >> 1;
                if (t0 < static_cast<int>(qo_indptr[mid + 1])) hi = mid; else lo = mid + 1;
            }
            g_req = lo;
            g_end = min(t_last, static_cast<int>(qo_indptr[lo + 1]));
        }
        __syncthreads();
        const int r = g_req, t1 = g_end, nrows = t1 - t0;
        const int qo_lo = static_cast<int>(qo_indptr[r]);
        const int new_tokens = static_cast<int>(qo_indptr[r + 1]) - qo_lo;
        const int pages_first = static_cast<int>(kv_page_indptr[r]);
        const int num_pages = static_cast<int>(kv_page_indptr[r + 1]) - pages_first;
        const int kv_len = (num_pages - 1) * page_size + static_cast<int>(kv_last_page_lens[r]);

        // Each row's blocks: the ones it selects, and its open one.
        for (int i = tid; i < S::kRows * S::kWords; i += S::kThreads) (&rowbits[0][0])[i] = 0u;
        __syncthreads();
        for (int rr = warp; rr < nrows; rr += S::kWarps) {
            const int t = t0 + rr;
            const int t_row = t + shift;
            const int j_end = kv_len - new_tokens + (t - qo_lo) + 1;
            const int nblocks = j_end / ratio;
            const int blocks = min(top_k, nblocks);
            const i32* srow = per_head
                ? selection + (static_cast<long long>(t_row) * KVH + kv_head) * top_k
                : selection + static_cast<long long>(t_row) * top_k;
            for (int n = lane; n < blocks; n += 32) {
                const int c = srow[n];
                if (c >= 0 && c < nblocks) atomicOr(&rowbits[rr][c >> 5], 1u << (c & 31));
            }
            if (lane == 0) {
                row_jend[rr] = j_end;
                if (nblocks * ratio < j_end) atomicOr(&rowbits[rr][nblocks >> 5], 1u << (nblocks & 31));
            }
        }
        __syncthreads();
        // The union, and where each word's blocks start in the list.
        for (int w = tid; w < S::kWords; w += S::kThreads) {
            u32 u = 0u;
            for (int rr = 0; rr < nrows; ++rr) u |= rowbits[rr][w];
            unionbits[w] = u;
        }
        __syncthreads();
        if (tid == 0) {
            int at = 0;
            for (int w = 0; w < S::kWords; ++w) {
                word_start[w] = at;
                at += __popc(unionbits[w]);
            }
            word_start[S::kWords] = at;
        }
        __syncthreads();
        for (int w = tid; w < S::kWords; w += S::kThreads) {
            u32 u = unionbits[w];
            int at = word_start[w];
            while (u) {
                const int b = __ffs(u) - 1;
                list[at++] = w * 32 + b;
                u &= u - 1;
            }
        }
        // The rows' queries: vector `v` is row `v / G`, head `v % G`.
        for (int c = tid; c < S::kVectors * kChunks; c += S::kThreads) {
            const int v = c / kChunks;
            const int d = (c % kChunks) * 8;
            int4 q = make_int4(0, 0, 0, 0);
            if (v < nrows * G) {
                const int rr = v / G, h = v % G;
                q = *reinterpret_cast<const int4*>(
                    queries + (static_cast<long long>(t0 + rr + shift) * H + kv_head * G + h) * D + d);
            }
            *reinterpret_cast<int4*>(sQ + v * kLd + d) = q;
        }
        __syncthreads();
        const int nkeys = word_start[S::kWords] * ratio;

        // The warp's 16 vectors' rows (their row in the tile, or -1).
        const int v_lo = warp * 16 + g, v_hi = v_lo + 8;
        const int row_lo = v_lo < nrows * G ? v_lo / G : -1;
        const int row_hi = v_hi < nrows * G ? v_hi / G : -1;

        float oacc[kNTiles][4];
#pragma unroll
        for (int n = 0; n < kNTiles; ++n) {
#pragma unroll
            for (int i = 0; i < 4; ++i) oacc[n][i] = 0.f;
        }
        float m_lo = -CUDART_INF_F, m_hi = -CUDART_INF_F, l_lo = 0.f, l_hi = 0.f;

        auto issue = [&](int base, int stage) {
            if (tid < kKeys) {
                const int n = base + tid;
                int j = -1;
                if (n < nkeys) j = list[n / ratio] * ratio + n % ratio;
                if (j >= kv_len) j = -1;
                key_s[stage][tid] = j;
                slot_s[stage][tid] = j < 0
                    ? -1
                    : static_cast<long long>(kv_page_indices[pages_first + j / page_size]) * page_size
                          + j % page_size;
            }
            __syncthreads();
            __nv_bfloat16* sK = sKV + stage * (2 * kKeys * kLd);
            __nv_bfloat16* sV = sK + kKeys * kLd;
            for (int c = tid; c < kKeys * kChunks; c += S::kThreads) {
                const int key = c / kChunks;
                const int d = (c % kChunks) * 8;
                const long long slot = slot_s[stage][key];
                if (slot >= 0) {
                    const long long at = (slot * KVH + kv_head) * D + d;
                    __pipeline_memcpy_async(sK + key * kLd + d, k_pages + at, 16);
                    __pipeline_memcpy_async(sV + key * kLd + d, v_pages + at, 16);
                } else {
                    *reinterpret_cast<int4*>(sK + key * kLd + d) = make_int4(0, 0, 0, 0);
                    *reinterpret_cast<int4*>(sV + key * kLd + d) = make_int4(0, 0, 0, 0);
                }
            }
            __pipeline_commit();
        };

        int stage = 0;
        if (nkeys > 0) issue(0, 0);
        for (int base = 0; base < nkeys; base += kKeys, stage ^= 1) {
            if (base + kKeys < nkeys) {
                issue(base + kKeys, stage ^ 1);
            } else {
                __pipeline_commit();
            }
            __pipeline_wait_prior(1);
            __syncthreads();
            const __nv_bfloat16* sK = sKV + stage * (2 * kKeys * kLd);
            const __nv_bfloat16* sV = sK + kKeys * kLd;

            float sacc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll 4
            for (int k0 = 0; k0 < D; k0 += 16) {
                std::uint32_t a[4];
                ld_a(a, sQ + warp * 16 * kLd, kLd, lane, k0);
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    std::uint32_t b[2];
                    ld_b_k(b, sK + nt * 8 * kLd, kLd, lane, k0);
                    mma_m16n8k16(sacc[nt], a, b);
                }
            }
            // A row reads a key of a block it chose, before its end.
            auto member = [&](int row, int col) {
                const int j = key_s[stage][col];
                if (row < 0 || j < 0 || j >= row_jend[row]) return false;
                const int b = j / ratio;
                return ((rowbits[row][b >> 5] >> (b & 31)) & 1u) != 0u;
            };
            float mx_lo = -CUDART_INF_F, mx_hi = -CUDART_INF_F;
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const int col = nt * 8 + p + i;
                    sacc[nt][i] = member(row_lo, col) ? sacc[nt][i] * sm_scale : -CUDART_INF_F;
                    sacc[nt][2 + i] = member(row_hi, col) ? sacc[nt][2 + i] * sm_scale : -CUDART_INF_F;
                    mx_lo = fmaxf(mx_lo, sacc[nt][i]);
                    mx_hi = fmaxf(mx_hi, sacc[nt][2 + i]);
                }
            }
#pragma unroll
            for (int sh = 1; sh < 4; sh <<= 1) {
                mx_lo = fmaxf(mx_lo, __shfl_xor_sync(0xffffffffu, mx_lo, sh));
                mx_hi = fmaxf(mx_hi, __shfl_xor_sync(0xffffffffu, mx_hi, sh));
            }
            const float mn_lo = fmaxf(m_lo, mx_lo), mn_hi = fmaxf(m_hi, mx_hi);
            const float c_lo = mn_lo == -CUDART_INF_F ? 1.f : __expf(m_lo - mn_lo);
            const float c_hi = mn_hi == -CUDART_INF_F ? 1.f : __expf(m_hi - mn_hi);
            m_lo = mn_lo;
            m_hi = mn_hi;
            float pr[2][4];
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    pr[nt][i] = mn_lo == -CUDART_INF_F ? 0.f : __expf(sacc[nt][i] - mn_lo);
                    pr[nt][2 + i] = mn_hi == -CUDART_INF_F ? 0.f : __expf(sacc[nt][2 + i] - mn_hi);
                }
            }
            l_lo = l_lo * c_lo + pr[0][0] + pr[0][1] + pr[1][0] + pr[1][1];
            l_hi = l_hi * c_hi + pr[0][2] + pr[0][3] + pr[1][2] + pr[1][3];
            const std::uint32_t pa[4] = {
                pack2(pr[0][0], pr[0][1]),
                pack2(pr[0][2], pr[0][3]),
                pack2(pr[1][0], pr[1][1]),
                pack2(pr[1][2], pr[1][3]),
            };
#pragma unroll
            for (int n = 0; n < kNTiles; ++n) {
                oacc[n][0] *= c_lo;
                oacc[n][1] *= c_lo;
                oacc[n][2] *= c_hi;
                oacc[n][3] *= c_hi;
                std::uint32_t b[2];
                ld_b_v(b, sV, kLd, lane, n * 8);
                mma_m16n8k16(oacc[n], pa, b);
            }
            __syncthreads();
        }

#pragma unroll
        for (int sh = 1; sh < 4; sh <<= 1) {
            l_lo += __shfl_xor_sync(0xffffffffu, l_lo, sh);
            l_hi += __shfl_xor_sync(0xffffffffu, l_hi, sh);
        }
        const float inv_lo = l_lo > 0.f ? 1.f / l_lo : 0.f;
        const float inv_hi = l_hi > 0.f ? 1.f / l_hi : 0.f;
        auto out_row = [&](int v) {
            return o + (static_cast<long long>(t0 + v / G + shift) * H + kv_head * G + v % G) * D;
        };
        if (row_lo >= 0) {
            __nv_bfloat16* dst = out_row(v_lo);
#pragma unroll
            for (int n = 0; n < kNTiles; ++n) {
                dst[n * 8 + p] = __float2bfloat16(oacc[n][0] * inv_lo);
                dst[n * 8 + p + 1] = __float2bfloat16(oacc[n][1] * inv_lo);
            }
        }
        if (row_hi >= 0) {
            __nv_bfloat16* dst = out_row(v_hi);
#pragma unroll
            for (int n = 0; n < kNTiles; ++n) {
                dst[n * 8 + p] = __float2bfloat16(oacc[n][2] * inv_hi);
                dst[n * 8 + p + 1] = __float2bfloat16(oacc[n][3] * inv_hi);
            }
        }
        __syncthreads();
        t0 = t1;
    }
}

}  // namespace mma

}
