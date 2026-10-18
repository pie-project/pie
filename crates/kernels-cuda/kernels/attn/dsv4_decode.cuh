#pragma once

// DeepSeek-V4's attention over a decode-width batch in one launch: a row's
// sliding window of the shared key/value plane (`prefill_lse` over the
// window), its compressed rows (`pool_lse`, or `pool_lse_selected` over the
// ids an indexer picked), the two folded (`merge_lse`) and the sinks (`sink`).
// Every query head reads the same keys (one kv head, keys and values one
// plane). A block holds a row's 16 heads (one of four groups) and walks its
// share of the row's keys 32 at a time: the window's last `window`
// positions, then the compressed rows.
//
// Eight warps. For S = Q K^T warp w takes keys 8(w % 4).. of the tile over
// half the head width (the (w / 4)th), its q fragments held in registers;
// the two halves meet in shared memory and every warp keeps the same online
// softmax over the whole tile. For O += P V warp w takes 64 columns.
//
// With formats WF and PF (`attn/dsv4_kv.cuh`; 0 is bf16) the window's and
// the pool's planes are DeepSeek's quantized rows: a tile's bytes land in a
// staging ring and are widened to bf16 before the products, as FlashMLA
// does on sm100. V4 keeps both planes in its fp8 rows; V4.1's FP4 cache its
// window in MXFP8 rows and its compressed rows in NVFP4 ones.
//
// A cluster of R blocks (along x) splits a row's keys; the blocks fold their
// running max, sum and output through distributed shared memory, each rank
// landing a slice of the head width with the sinks applied.

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <math_constants.h>

#include "attn/dsv4_kv.cuh"
#include "prelude/device.cuh"

namespace pie::attn::dsv4 {

constexpr int kGroup = 16;   // heads a block
constexpr int kGroups = 4;   // blocks a row's 64 heads
constexpr int kHeads = kGroup * kGroups;
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kKeys = 32;
// Tiles a block's key addresses are gathered for at once (a chunk).
constexpr int kChunkTiles = 32;
constexpr float kLog2e = 1.44269504088896340736f;

template <int D, int WF, int PF>
struct Shape {
    static constexpr bool kFp8 = WF != 0;
    static constexpr int kRawData = kFp8 ? (Rows<WF>::kData > Rows<PF>::kData ? Rows<WF>::kData : Rows<PF>::kData) : 0;
    static constexpr int kRawScale = kFp8 ? (Rows<WF>::kScale > Rows<PF>::kScale ? Rows<WF>::kScale : Rows<PF>::kScale) : 0;
    static constexpr int kLd = D + 8;              // a bf16 row, padded
    static constexpr int kSLd = kKeys + 8;         // a partial score row, padded
    static constexpr int kOLd = D + 8;             // the fold's f32 row, padded
    // Tiles in flight: fp8 rows are smaller, so more of them.
    static constexpr int kStages = kFp8 ? 6 : 4;
    static constexpr int kQBytes = kGroup * kLd * 2;
    static constexpr int kTileBytes = kKeys * kLd * 2;
    static constexpr int kRawBytes = kKeys * (kRawData + kRawScale);
    // bf16: the ring is the tiles; fp8: the ring is raw bytes, widened into
    // one tile.
    static constexpr int kRingBytes = kFp8 ? kStages * kRawBytes + kTileBytes : kStages * kTileBytes;
    static constexpr int kSBytes = 2 * kGroup * kSLd * 4;
    static constexpr int kBytes = kQBytes + kRingBytes + kSBytes;
    static_assert(kGroup * kOLd * 4 <= kQBytes + kRingBytes, "the fold fits where Q and the ring were");
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
    asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(remote));
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
                 : "r"(remote));
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

// `ratio` zero: no compressed rows (the window and the sinks alone).
// `selection` null with a ratio: every compressed row the query sees;
// otherwise the row's `top_k` ids, ended early by a negative one.
// bf16 planes are `stride` elements a position; fp8 planes are DeepSeek's
// 584-byte rows, page by page.
// Key `slot`'s raw bytes in format F into a ring key's data and scale
// bytes: `part` of the key's kPer threads.
template <int F, int kPer>
__device__ __forceinline__ void stage_raw(std::uint8_t* plane, long long slot, int ps, int part,
                                          std::uint8_t* data, std::uint8_t* sc) {
    using T = Rows<F>;
    const DsRow r = rows_at<F>(plane, slot / ps, static_cast<int>(slot % ps), ps);
    for (int i = part; i < T::kData / 16; i += kPer) __pipeline_memcpy_async(data + i * 16, r.data + i * 16, 16);
    if (part == 0) {
        if constexpr (T::kScale % 16 == 0) {
            for (int i = 0; i < T::kScale / 16; ++i) __pipeline_memcpy_async(sc + i * 16, r.scales + i * 16, 16);
        } else {
            __pipeline_memcpy_async(sc, r.scales, T::kScale);
        }
    }
}

template <int D, class SinkT, int R, int WF, int PF>
__global__ __launch_bounds__(kThreads, 1) void decode(
    const bf16* __restrict__ q,
    const void* __restrict__ win_keys,
    const u32* __restrict__ win_indices,
    const u32* __restrict__ win_indptr,
    long long win_stride,
    int win_page_size,
    int window,
    const void* __restrict__ pool_keys,
    const u32* __restrict__ pool_indices,
    const u32* __restrict__ pool_indptr,
    long long pool_stride,
    int pool_page_size,
    int ratio,
    const i32* __restrict__ selection,
    int top_k,
    const i32* __restrict__ positions,
    const i32* __restrict__ req_of_token,
    const i32* __restrict__ floors,
    const SinkT* __restrict__ sinks,
    bf16* __restrict__ o,
    float scale,
    Turn turn,
    int pool_compact,
    const u32* __restrict__ win)
{
    using S = Shape<D, WF, PF>;
    constexpr bool kFp8 = S::kFp8;
    static_assert((WF == 0) == (PF == 0), "both planes quantized, or neither");
    constexpr int kLd = S::kLd;
    constexpr int kSLd = S::kSLd;
    constexpr int kHalf = D / 2;
    constexpr int kCols = D / kWarps;          // PV columns a warp
    constexpr int kChunks = D / 8;             // 16-byte bf16 pieces a key row
    constexpr int kPerKey = kThreads / kKeys;  // threads a key row
    static_assert(!kFp8 || D == kDsDim, "an fp8 row is DeepSeek's 512");
    static_assert(kChunks % kPerKey == 0, "a key row splits evenly");
    static_assert(kHalf % 32 == 0 && kCols % 16 == 0, "whole k32 steps and n16 pairs");

    const int qi = static_cast<int>(blockIdx.z);
    if (win != nullptr && qi >= static_cast<int>(win[0])) return;
    const int row = win != nullptr ? qi + static_cast<int>(win[1]) : qi;
    const int group = static_cast<int>(blockIdx.y);
    const unsigned rank = R > 1 ? rank_in_cluster() : 0;

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int kw = warp & 3;  // QK: the tile's keys 8kw..8kw+7
    const int dh = warp >> 2; // QK: half the head width
    const int g = lane >> 2, p = (lane & 3) << 1;

    extern __shared__ __align__(128) char smem_raw[];
    bf16* sQ = reinterpret_cast<bf16*>(smem_raw);
    char* sRing = smem_raw + S::kQBytes;
    float* sS = reinterpret_cast<float*>(smem_raw + S::kQBytes + S::kRingBytes);
    constexpr int kStages = S::kStages;
    // A chunk's key slots (page * page size + offset; -1 reads nothing).
    __shared__ long long s_slot[kChunkTiles * kKeys];
    __shared__ int s_end;
    __shared__ float s_ml[2][kGroup];
    __shared__ float s_fold[2][kGroup];

    const int req = req_of_token[row];
    const int qpos = positions[row];
    // A replay floor: the window reads no key below the row's.
    const int floor = floors != nullptr ? floors[row] : 0;
    const int n_win = max(0, min(qpos + 1 - floor, window));
    const int num_visible = ratio > 0 ? max((qpos + 1) / ratio, 0) : 0;
    const i32* srow =
        selection != nullptr ? selection + static_cast<long long>(row) * top_k : nullptr;

    if (tid == 0) s_end = srow != nullptr ? top_k : num_visible;
    const bf16* qrow = q + (static_cast<long long>(row) * kHeads + group * kGroup) * D;
    for (int c = tid; c < kGroup * kChunks; c += kThreads) {
        const int h = c / kChunks;
        const int d = (c % kChunks) * 8;
        __pipeline_memcpy_async(sQ + h * kLd + d, qrow + h * D + d, 16);
    }
    __pipeline_commit();
    __syncthreads();
    if (srow != nullptr && num_visible > 0) {
        for (int n = tid; n < top_k; n += kThreads)
            if (srow[n] < 0) atomicMin(&s_end, n);
    }
    __syncthreads();
    const int n_pool = num_visible > 0 ? s_end : 0;
    const int total = n_win + n_pool;
    const int tiles = (total + kKeys - 1) / kKeys;
    const int per = (tiles + R - 1) / R;
    const int t0 = min(tiles, static_cast<int>(rank) * per);
    const int t1 = min(tiles, t0 + per);

    // Key n's slot in its plane: the window's first `n_win`, then the
    // compressed rows'; -1 for none.
    auto slot_of_key = [&](int n) -> long long {
        if (n < n_win) {
            const int pos = qpos - n_win + 1 + n;
            return static_cast<long long>(win_indices[win_indptr[req] + pos / win_page_size]) *
                       win_page_size +
                   pos % win_page_size;
        }
        if (n >= total) return -1;
        const int j = n - n_win;
        const int c = srow != nullptr ? srow[j] : j;
        if (c < 0 || c >= num_visible) return -1;
        // A compact pool's page holds its entries a row each (`pool_page_size`
        // of them); else entry c sits at its group's last position.
        const int pos = pool_compact ? c : (c + 1) * ratio - 1;
        return static_cast<long long>(pool_indices[pool_indptr[req] + pos / pool_page_size]) *
                   pool_page_size +
               pos % pool_page_size;
    };

    // Tile t's keys into ring slot `buf`: a key row per kPerKey threads.
    auto stage = [&](int t, int c0, int buf) {
        const int key = tid / kPerKey;
        const int part = tid % kPerKey;
        const int n = t * kKeys + key;
        const long long slot = s_slot[(t - c0) * kKeys + key];
        const bool win_key = n < n_win;
        if constexpr (kFp8) {
            std::uint8_t* raw = reinterpret_cast<std::uint8_t*>(sRing) + buf * S::kRawBytes;
            std::uint8_t* data = raw + key * S::kRawData;
            std::uint8_t* sc = raw + kKeys * S::kRawData + key * S::kRawScale;
            if (slot >= 0) {
                const int ps = win_key ? win_page_size : pool_page_size;
                std::uint8_t* plane = const_cast<std::uint8_t*>(
                    static_cast<const std::uint8_t*>(win_key ? win_keys : pool_keys));
                if (win_key) {
                    stage_raw<WF, kPerKey>(plane, slot, ps, part, data, sc);
                } else {
                    stage_raw<PF, kPerKey>(plane, slot, ps, part, data, sc);
                }
            }
        } else {
            bf16* dst = reinterpret_cast<bf16*>(sRing) + buf * kKeys * kLd + key * kLd;
            const bf16* src =
                slot >= 0 ? static_cast<const bf16*>(win_key ? win_keys : pool_keys) +
                                slot * (win_key ? win_stride : pool_stride)
                          : nullptr;
#pragma unroll
            for (int i = 0; i < kChunks / kPerKey; ++i) {
                const int d = (part + i * kPerKey) * 8;
                if (slot >= 0) {
                    __pipeline_memcpy_async(dst + d, src + d, 16);
                } else {
                    *reinterpret_cast<int4*>(dst + d) = make_int4(0, 0, 0, 0);
                }
            }
        }
    };

    // This warp's q fragments: 16 heads over its half of the width.
    std::uint32_t qa[kHalf / 16][4];
    __pipeline_wait_prior(0);
    __syncthreads();
#pragma unroll
    for (int k = 0; k < kHalf / 16; ++k)
        ldsm4(qa[k], sQ + (lane & 15) * kLd + dh * kHalf + k * 16 + ((lane & 16) ? 8 : 0));

    constexpr int kNT = kCols / 8;
    float oacc[kNT][4];
#pragma unroll
    for (int n = 0; n < kNT; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) oacc[n][i] = 0.f;
    // Log2 domain: scores carry scale * log2(e).
    float m_lo = -CUDART_INF_F, m_hi = -CUDART_INF_F, l_lo = 0.f, l_hi = 0.f;
    const float sl2 = scale * kLog2e;
    float* mine = sS + dh * kGroup * kSLd;
    const float* other = sS + (dh ^ 1) * kGroup * kSLd;
    bf16* sTile = reinterpret_cast<bf16*>(sRing + (kFp8 ? kStages * S::kRawBytes : 0));

    for (int c0 = t0; c0 < t1; c0 += kChunkTiles) {
    const int c1 = min(t1, c0 + kChunkTiles);
    // The chunk's key slots, every one in flight at once; the last chunk's
    // tiles are all read before they are overwritten.
    __syncthreads();
    for (int k = tid; k < (c1 - c0) * kKeys; k += kThreads) s_slot[k] = slot_of_key(c0 * kKeys + k);
    __syncthreads();
#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (c0 + s < c1) stage(c0 + s, c0, s);
        __pipeline_commit();
    }
    for (int t = c0; t < c1; ++t) {
        const int buf = (t - c0) % kStages;
        const long long* tslot = s_slot + (t - c0) * kKeys;
        __pipeline_wait_prior(kStages - 2);
        __syncthreads();
        if (t + kStages - 1 < c1) stage(t + kStages - 1, c0, (t - c0 + kStages - 1) % kStages);
        __pipeline_commit();
        const bf16* sK;
        if constexpr (kFp8) {
            // Widen the staged bytes into the bf16 tile.
            const std::uint8_t* raw = reinterpret_cast<const std::uint8_t*>(sRing) + buf * S::kRawBytes;
            for (int c = tid; c < kKeys * kChunks; c += kThreads) {
                const int key = c / kChunks;
                const int j = (c % kChunks) * 8;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (tslot[key] >= 0) {
                    const std::uint8_t* kd = raw + key * S::kRawData;
                    const std::uint8_t* ks = raw + kKeys * S::kRawData + key * S::kRawScale;
                    v = t * kKeys + key < n_win ? rows_load8<WF>(kd, ks, j) : rows_load8<PF>(kd, ks, j);
                }
                *reinterpret_cast<uint4*>(sTile + key * kLd + j) = v;
            }
            __syncthreads();
            sK = sTile;
        } else {
            sK = reinterpret_cast<const bf16*>(sRing) + buf * kKeys * kLd;
        }

        // S over this warp's 8 keys and half the width.
        float s0[4] = {0.f, 0.f, 0.f, 0.f}, s1[4] = {0.f, 0.f, 0.f, 0.f};
        {
            const int j = lane >> 3;
            const bf16* krow = sK + (kw * 8 + (lane & 7)) * kLd + dh * kHalf + j * 8;
#pragma unroll
            for (int k = 0; k < kHalf / 16; k += 2) {
                std::uint32_t b[4];
                ldsm4(b, krow + k * 16);
                mma16816(s0, qa[k], b[0], b[1]);
                mma16816(s1, qa[k + 1], b[2], b[3]);
            }
        }
        *reinterpret_cast<float2*>(mine + g * kSLd + kw * 8 + p) = make_float2(s0[0] + s1[0], s0[1] + s1[1]);
        *reinterpret_cast<float2*>(mine + (g + 8) * kSLd + kw * 8 + p) =
            make_float2(s0[2] + s1[2], s0[3] + s1[3]);
        __syncthreads();

        // The whole tile's scores, in every warp.
        float sacc[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const float2 a_lo = *reinterpret_cast<const float2*>(mine + g * kSLd + nt * 8 + p);
            const float2 b_lo = *reinterpret_cast<const float2*>(other + g * kSLd + nt * 8 + p);
            const float2 a_hi = *reinterpret_cast<const float2*>(mine + (g + 8) * kSLd + nt * 8 + p);
            const float2 b_hi = *reinterpret_cast<const float2*>(other + (g + 8) * kSLd + nt * 8 + p);
            const bool v0 = tslot[nt * 8 + p] >= 0, v1 = tslot[nt * 8 + p + 1] >= 0;
            sacc[nt][0] = v0 ? (a_lo.x + b_lo.x) * sl2 : -CUDART_INF_F;
            sacc[nt][1] = v1 ? (a_lo.y + b_lo.y) * sl2 : -CUDART_INF_F;
            sacc[nt][2] = v0 ? (a_hi.x + b_hi.x) * sl2 : -CUDART_INF_F;
            sacc[nt][3] = v1 ? (a_hi.y + b_hi.y) * sl2 : -CUDART_INF_F;
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
        for (int n = 0; n < kNT; ++n) {
            oacc[n][0] *= c_lo;
            oacc[n][1] *= c_lo;
            oacc[n][2] *= c_hi;
            oacc[n][3] *= c_hi;
        }
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            const int j = lane >> 3;
            const bf16* vrow = sK + (kk * 16 + (j & 1) * 8 + (lane & 7)) * kLd + warp * kCols + (j >> 1) * 8;
#pragma unroll
            for (int np = 0; np < kNT / 2; ++np) {
                std::uint32_t b[4];
                ldsm4t(b, vrow + np * 16);
                mma16816(oacc[2 * np], pa[kk], b[0], b[1]);
                mma16816(oacc[2 * np + 1], pa[kk], b[2], b[3]);
            }
        }
    }
    __pipeline_wait_prior(0);
    }

    const int h_lo = g, h_hi = g + 8;
    bf16* orow = o + (static_cast<long long>(row) * kHeads + group * kGroup) * D;
    // The turn's frequencies, once a block (every thread reaches here).
    __shared__ float s_turn[64];
    if (turn.dim > 0) {
        if (tid < turn.dim / 2) s_turn[tid] = turn_freq(tid, turn);
        __syncthreads();
    }

    if constexpr (R == 1) {
        const float sink_lo = Elem<SinkT>::to_f32(sinks[group * kGroup + h_lo]) * kLog2e;
        const float sink_hi = Elem<SinkT>::to_f32(sinks[group * kGroup + h_hi]) * kLog2e;
        const float top_lo = fmaxf(m_lo, sink_lo), top_hi = fmaxf(m_hi, sink_hi);
        const float f_lo = exp2f(m_lo - top_lo), f_hi = exp2f(m_hi - top_hi);
        const float den_lo = l_lo * f_lo + exp2f(sink_lo - top_lo);
        const float den_hi = l_hi * f_hi + exp2f(sink_hi - top_hi);
        const float s_lo = den_lo > 0.f ? f_lo / den_lo : 0.f;
        const float s_hi = den_hi > 0.f ? f_hi / den_hi : 0.f;
        const int turn_pos = turn.dim > 0 ? positions[row] : 0;
#pragma unroll
        for (int n = 0; n < kNT; ++n) {
            const int d = warp * kCols + n * 8 + p;
            float a0 = oacc[n][0] * s_lo, a1 = oacc[n][1] * s_lo;
            float b0 = oacc[n][2] * s_hi, b1 = oacc[n][3] * s_hi;
            turn_back(a0, a1, d, D, turn_pos, turn, s_turn);
            turn_back(b0, b1, d, D, turn_pos, turn, s_turn);
            *reinterpret_cast<std::uint32_t*>(orow + h_lo * D + d) = pack2(a0, a1);
            *reinterpret_cast<std::uint32_t*>(orow + h_hi * D + d) = pack2(b0, b1);
        }
    } else {
        // The ranks' fold: every rank's (max, sum) a head, then its output
        // scaled to the common max over the common sum, in its own shared
        // memory. Every rank's value is in flight before any is used.
        if (warp == 0 && (lane & 3) == 0) {
            s_ml[0][h_lo] = m_lo;
            s_ml[0][h_hi] = m_hi;
            s_ml[1][h_lo] = l_lo;
            s_ml[1][h_hi] = l_hi;
        }
        sync_ranks();
        if (tid < kGroup) {
            const float sink = Elem<SinkT>::to_f32(sinks[group * kGroup + tid]) * kLog2e;
            float rm[R], rl[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                rm[r] = ld_rank(&s_ml[0][tid], r);
                rl[r] = ld_rank(&s_ml[1][tid], r);
            }
            float top = sink;
#pragma unroll
            for (int r = 0; r < R; ++r) top = fmaxf(top, rm[r]);
            float den = exp2f(sink - top);
#pragma unroll
            for (int r = 0; r < R; ++r)
                if (rl[r] > 0.f) den += rl[r] * exp2f(rm[r] - top);
            s_fold[0][tid] = top;
            s_fold[1][tid] = den;
        }
        __syncthreads();
        const float s_lo = l_lo > 0.f ? exp2f(m_lo - s_fold[0][h_lo]) / s_fold[1][h_lo] : 0.f;
        const float s_hi = l_hi > 0.f ? exp2f(m_hi - s_fold[0][h_hi]) / s_fold[1][h_hi] : 0.f;
        constexpr int kOLd = S::kOLd;
        float* sO = reinterpret_cast<float*>(smem_raw);
#pragma unroll
        for (int n = 0; n < kNT; ++n) {
            const int d = warp * kCols + n * 8 + p;
            *reinterpret_cast<float2*>(sO + h_lo * kOLd + d) = make_float2(oacc[n][0] * s_lo, oacc[n][1] * s_lo);
            *reinterpret_cast<float2*>(sO + h_hi * kOLd + d) = make_float2(oacc[n][2] * s_hi, oacc[n][3] * s_hi);
        }
        sync_ranks();
        // This rank's slice of the width, summed over the ranks.
        constexpr int kSlice = D / R;
        constexpr int kQuads = kGroup * kSlice / 4;
#pragma unroll 1
        for (int i = tid; i < kQuads; i += kThreads) {
            const int h = i / (kSlice / 4);
            const int d = static_cast<int>(rank) * kSlice + (i % (kSlice / 4)) * 4;
            float4 v[R];
#pragma unroll
            for (int r = 0; r < R; ++r) v[r] = ld_rank4(sO + h * kOLd + d, r);
            float4 acc = v[0];
#pragma unroll
            for (int r = 1; r < R; ++r) {
                acc.x += v[r].x;
                acc.y += v[r].y;
                acc.z += v[r].z;
                acc.w += v[r].w;
            }
            if (turn.dim > 0) {
                const int turn_pos = positions[row];
                turn_back(acc.x, acc.y, d, D, turn_pos, turn, s_turn);
                turn_back(acc.z, acc.w, d + 2, D, turn_pos, turn, s_turn);
            }
            uint2 packed;
            packed.x = pack2(acc.x, acc.y);
            packed.y = pack2(acc.z, acc.w);
            *reinterpret_cast<uint2*>(orow + h * D + d) = packed;
        }
        // No block leaves while another still reads its shared memory.
        sync_ranks();
    }
}

}  // namespace pie::attn::dsv4
