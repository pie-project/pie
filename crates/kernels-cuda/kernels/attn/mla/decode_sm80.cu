#include <cstdint>
#include "prim/mma.cuh"
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_pipeline.h>
#include <math_constants.h>
#include "prim/device.cuh"
#include "prim/rope.cuh"
#include "attn/fa2/mla.cuh"
#include "attn/fa2/mla_params.cuh"

namespace pie::attn::mla {

constexpr int kMaxRopePairs = 128;

namespace mla_naive {

constexpr int kMlaNaiveBlock = 256;
constexpr int kMlaNaiveWarps = kMlaNaiveBlock / 32;
constexpr int kMlaNaiveMaxPePer = 4;

// `MaxPer` latent elements a lane holds: 16 for a 512-wide latent, 32 for
// a 1024-wide one.
template <int MaxPer>
__global__ void mla_naive_paged_kernel(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ ckv_pages,
    const __nv_bfloat16* __restrict__ kpe_pages,
    const std::uint32_t* __restrict__ qo_indptr,
    const std::uint32_t* __restrict__ kv_page_indices,
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
    __nv_bfloat16* __restrict__ o,
    float* __restrict__ lse,
    const std::int32_t* __restrict__ selection, int top_k,
    int R, int H, int CKV, int KPE, int page_size, float sm_scale, bool causal,
    int G, int window,
    const u32* __restrict__ win)
{
    const int t = blockIdx.x;

    if (win != nullptr && t >= static_cast<int>(win[0])) return;

    const int t_row = win != nullptr ? t + static_cast<int>(win[1]) : t;

    if (win != nullptr && static_cast<int>(win[2]) < R) R = static_cast<int>(win[2]);
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int K = kMlaNaiveWarps / G;
    const int g = warp / K;
    const int s = warp % K;
    const int h = blockIdx.y * G + g;
    const int per = CKV / 32;
    const int pper = KPE / 32;

    const std::int32_t* srow =
        (selection != nullptr) ? selection + static_cast<long long>(t_row) * top_k : nullptr;

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
    const int pre_kv = kv_len - new_tokens;
    const int abs_q = pre_kv + (t - qo_lo);
    const int j_end = causal ? (abs_q + 1) : kv_len;
    const int j_lo = window > 0 ? max(0, abs_q + 1 - window) : 0;

    extern __shared__ float smem[];
    float* wacc  = smem;
    float* wm    = wacc + kMlaNaiveWarps * CKV;
    float* wl    = wm + kMlaNaiveWarps;

    const __nv_bfloat16* qn =
        q_nope + (static_cast<long long>(t_row) * H + h) * CKV;
    const __nv_bfloat16* qp =
        q_pe + (static_cast<long long>(t_row) * H + h) * KPE;
    float qn_r[MaxPer];
    float qp_r[kMlaNaiveMaxPePer];
    for (int i = 0; i < per; ++i) qn_r[i] = __bfloat162float(qn[lane + i * 32]);
    for (int i = 0; i < pper; ++i) qp_r[i] = __bfloat162float(qp[lane + i * 32]);

    float acc[MaxPer];
    for (int i = 0; i < per; ++i) acc[i] = 0.f;
    float m = -CUDART_INF_F, lsum = 0.f;

    const int steps = (srow != nullptr) ? top_k : j_end;
    for (int n = (srow != nullptr ? 0 : j_lo) + s; n < steps; n += K) {
        int j = n;
        if (srow != nullptr) {
            j = srow[n];
            if (j < j_lo || j >= j_end) continue;
        }
        const int page =
            static_cast<int>(kv_page_indices[pages_first + j / page_size]);
        const int off = j % page_size;
        const __nv_bfloat16* ckv_j =
            ckv_pages + (static_cast<long long>(page) * page_size + off) * CKV;
        const __nv_bfloat16* kpe_j =
            kpe_pages + (static_cast<long long>(page) * page_size + off) * KPE;

        float kv[MaxPer];
        float pd = 0.f;
        for (int i = 0; i < per; ++i) {
            kv[i] = __bfloat162float(ckv_j[lane + i * 32]);
            pd += qn_r[i] * kv[i];
        }
        for (int i = 0; i < pper; ++i) {
            pd += qp_r[i] * __bfloat162float(kpe_j[lane + i * 32]);
        }
        pd = ::pie::warp_sum(pd);
        const float score = pd * sm_scale;
        const float m_new = fmaxf(m, score);
        const float corr = __expf(m - m_new);
        const float p = __expf(score - m_new);
        lsum = lsum * corr + p;
        for (int i = 0; i < per; ++i) {
            acc[i] = acc[i] * corr + p * kv[i];
        }
        m = m_new;
    }

    for (int i = 0; i < per; ++i) {
        wacc[warp * CKV + lane + i * 32] = acc[i];
    }
    if (lane == 0) { wm[warp] = m; wl[warp] = lsum; }
    __syncthreads();

    const int total_out = G * CKV;
    for (int idx = tid; idx < total_out; idx += kMlaNaiveBlock) {
        const int gg = idx / CKV;
        const int d = idx % CKV;
        const int w0 = gg * K;
        float m_all = -CUDART_INF_F;
        for (int w = w0; w < w0 + K; ++w) m_all = fmaxf(m_all, wm[w]);
        float l_all = 0.f, v = 0.f;
        for (int w = w0; w < w0 + K; ++w) {
            if (wm[w] > -CUDART_INF_F) {
                const float e = __expf(wm[w] - m_all);
                l_all += wl[w] * e;
                v += wacc[w * CKV + d] * e;
            }
        }
        const float inv = (l_all > 0.f) ? (1.f / l_all) : 0.f;
        o[(static_cast<long long>(t_row) * H + blockIdx.y * G + gg) * CKV + d] =
            __float2bfloat16(v * inv);
        if (lse != nullptr && d == 0) {
            lse[static_cast<long long>(t_row) * H + blockIdx.y * G + gg] =
                (l_all > 0.f) ? (m_all + logf(l_all)) * 1.4426950408889634f : -CUDART_INF_F;
        }
    }
}

namespace mma_detail {

#ifndef PIE_MLA_MMA_BK
#define PIE_MLA_MMA_BK 64
#endif
#ifndef PIE_MLA_MMA_WARPS
#define PIE_MLA_MMA_WARPS 8
#endif

#ifndef PIE_MLA_MMA_STAGES
#define PIE_MLA_MMA_STAGES 1
#endif

#ifndef PIE_MLA_MMA_MINBLK
#define PIE_MLA_MMA_MINBLK 2
#endif

constexpr int kBM = 16;
constexpr int kBK = PIE_MLA_MMA_BK;
constexpr int kWarps = PIE_MLA_MMA_WARPS;
constexpr int kStages = PIE_MLA_MMA_STAGES;
static_assert(kStages >= 1, "kStages must be at least 1");
constexpr int kThreads = kWarps * 32;
constexpr int kCkv = 512;
// `kKpe`/`kD`/`kLdD` live in the kernel: they follow its `KPE` template
// parameter (the rope width, 64 for a rotated part, 0 for NoPE latents).
constexpr int kLdP = kBK + 8;
constexpr int kDimsPerWarp = kCkv / kWarps;
constexpr int kNTiles = kDimsPerWarp / 8;
constexpr int kSNTiles = kBK / 8 / kWarps;
static_assert(kSNTiles >= 1, "kBK must be at least 8 * kWarps");
static_assert(kBK % 16 == 0, "kBK must be a multiple of the mma k step");

// With `kFp8` the pool is vLLM's fp8 MLA rows (`mla_kv_append_fp8`): one u8
// plane, a cell's latent and rotated e4m3 codes in one row under a scale of
// 1, at `ckv_pages`; each staged chunk is widened to bf16 as it lands (the
// tcgen05 sparse pass reads them otherwise).
template <int KPE, bool kFp8 = false>
__global__ __launch_bounds__(kThreads, PIE_MLA_MMA_MINBLK) void mla_mma_paged_kernel(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ ckv_pages,
    const __nv_bfloat16* __restrict__ kpe_pages,
    const std::uint32_t* __restrict__ qo_indptr,
    const std::uint32_t* __restrict__ kv_page_indices,
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
    __nv_bfloat16* __restrict__ o,
    float* __restrict__ lse,
    __nv_bfloat16* __restrict__ part_o,
    float* __restrict__ part_lse,
    const std::int32_t* __restrict__ selection, int top_k, int dense,
    int R, int H, int page_size, float sm_scale, bool causal,
    const u32* __restrict__ win)
{
    // `KPE` is the rope width: 64 with a rotated part, 0 for NoPE latents
    // (GLM-5.3-Flash's DSA). The `d < kCkv` loads already skip the pe part
    // when it is absent; q_pe/kpe_pages are then unread.
    constexpr int kKpe = KPE;
    constexpr int kD = kCkv + kKpe;
    constexpr int kLdD = kD + 8;
    extern __shared__ __align__(16) char smem_raw[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sKbuf = sQ + kBM * kLdD;
    __nv_bfloat16* sP = sKbuf + kStages * kBK * kLdD;
    float* sS = reinterpret_cast<float*>(sP + kBM * kLdP);
    float* sM = sS + kBM * kBK;
    float* sL = sM + kBM;
    float* sCorr = sL + kBM;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const int t = blockIdx.y;

    if (win != nullptr && t >= static_cast<int>(win[0])) return;

    const int t_row = win != nullptr ? t + static_cast<int>(win[1]) : t;

    if (win != nullptr && static_cast<int>(win[2]) < R) R = static_cast<int>(win[2]);
    const int h0 = blockIdx.x * kBM;

    __shared__ int s_req;
    if (tid == 0) s_req = 0;
    if (tid < kBM) { sM[tid] = -CUDART_INF_F; sL[tid] = 0.f; }
    __syncthreads();

    for (int i = tid; i < R; i += kThreads) {
        if (t >= static_cast<int>(qo_indptr[i]) &&
            t < static_cast<int>(qo_indptr[i + 1])) {
            s_req = i;
        }
    }
    __syncthreads();
    const int r = s_req;

    const int qo_lo = static_cast<int>(qo_indptr[r]);
    const int new_tokens = static_cast<int>(qo_indptr[r + 1]) - qo_lo;
    const int pages_first = static_cast<int>(kv_page_indptr[r]);
    const int num_pages = static_cast<int>(kv_page_indptr[r + 1]) - pages_first;
    const int kv_len =
        (num_pages - 1) * page_size + static_cast<int>(kv_last_page_lens[r]);
    const int abs_q = kv_len - new_tokens + (t - qo_lo);
    const int j_end = causal ? (abs_q + 1) : kv_len;
    // A selected row reads the positions its ranking kept, -1 anywhere among
    // them (a pooled index's dropped blocks): through its last id. A row
    // that sees no more keys than its selection holds reads them all in
    // order where `dense` says, as its index (which ranks all of them) kept.
    const std::int32_t* srow =
        selection != nullptr ? selection + static_cast<long long>(t_row) * top_k : nullptr;
    if (srow != nullptr && dense != 0 && j_end <= top_k) srow = nullptr;
    __shared__ int s_entries;
    if (srow != nullptr) {
        if (tid == 0) s_entries = 0;
        __syncthreads();
        int last = 0;
        for (int n = tid; n < top_k; n += kThreads) {
            if (srow[n] >= 0) last = n + 1;
        }
        if (last > 0) atomicMax(&s_entries, last);
        __syncthreads();
    }
    const int steps = srow != nullptr ? s_entries : j_end;
    // Split-K (`gridDim.z` > 1, a decode's few rows): this block reads its
    // share of the row's key tiles, its normalized output and base-2
    // log-sum-exp landing in `part_o`/`part_lse` for `mla_mma_fold`.
    const int splits = static_cast<int>(gridDim.z);
    const int per = ((steps + kBK - 1) / kBK + splits - 1) / splits * kBK;
    const int j_lo = min(steps, static_cast<int>(blockIdx.z) * per);
    const int j_hi = min(steps, j_lo + per);

    constexpr int kChunksPerRow = kD / 8;
    for (int c = tid; c < kBM * kChunksPerRow; c += kThreads) {
        const int row = c / kChunksPerRow;
        const int d = (c % kChunksPerRow) * 8;
        const long long qh = (static_cast<long long>(t_row) * H + h0 + row);
        const int4 v = (d < kCkv)
            ? *reinterpret_cast<const int4*>(q_nope + qh * kCkv + d)
            : *reinterpret_cast<const int4*>(q_pe + qh * kKpe + (d - kCkv));
        *reinterpret_cast<int4*>(sQ + row * kLdD + d) = v;
    }

    float oacc[kNTiles][4];
    #pragma unroll
    for (int n = 0; n < kNTiles; ++n) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) oacc[n][i] = 0.f;
    }
    const int g = lane >> 2, p = (lane & 3) << 1;
    const int dbase = warp * kDimsPerWarp;

    __shared__ long long s_slot[kBK];
    // Which of each stage's tile rows hold a key, for the scores' mask.
    __shared__ bool s_ok[kStages * kBK];
    auto issue_tile = [&](int j0, int stage) {
        for (int rr = tid; rr < kBK; rr += kThreads) {
            const int n = j0 + rr;
            int j = n;
            if (srow != nullptr) j = n < j_hi ? srow[n] : -1;
            const bool ok = n < j_hi && j >= 0 && j < j_end;
            s_ok[stage * kBK + rr] = ok;
            s_slot[rr] =
                ok
                    ? (static_cast<long long>(
                           kv_page_indices[pages_first + j / page_size]) *
                           page_size +
                       (j % page_size))
                    : -1;
        }
        __syncthreads();
        __nv_bfloat16* dst_base = sKbuf + stage * (kBK * kLdD);
        for (int c = tid; c < kBK * kChunksPerRow; c += kThreads) {
            const int row = c / kChunksPerRow;
            const int d = (c % kChunksPerRow) * 8;
            const long long slot = s_slot[row];
            __nv_bfloat16* dst = dst_base + row * kLdD + d;
            if (slot >= 0 && kFp8) {
                const uint2 codes = *reinterpret_cast<const uint2*>(
                    reinterpret_cast<const std::uint8_t*>(ckv_pages) + slot * kD + d);
                const unsigned short pairs[4] = {
                    static_cast<unsigned short>(codes.x), static_cast<unsigned short>(codes.x >> 16),
                    static_cast<unsigned short>(codes.y), static_cast<unsigned short>(codes.y >> 16)};
                int4 v;
                std::uint32_t* w = reinterpret_cast<std::uint32_t*>(&v);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float lo, hi;
                    ::pie::e4m3_pair_f32(pairs[i], lo, hi);
                    const __nv_bfloat162 b = __floats2bfloat162_rn(lo, hi);
                    w[i] = *reinterpret_cast<const std::uint32_t*>(&b);
                }
                *reinterpret_cast<int4*>(dst) = v;
            } else if (slot >= 0) {
                const void* src =
                    (d < kCkv)
                        ? static_cast<const void*>(ckv_pages + slot * kCkv + d)
                        : static_cast<const void*>(kpe_pages + slot * kKpe +
                                                   (d - kCkv));
                __pipeline_memcpy_async(dst, src, 16);
            } else {
                *reinterpret_cast<int4*>(dst) = make_int4(0, 0, 0, 0);
            }
        }
        __pipeline_commit();
    };

    #pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        const int j0 = j_lo + s * kBK;
        if (j0 < j_hi) issue_tile(j0, s);
    }

    int stage = 0;
    for (int j0 = j_lo; j0 < j_hi; j0 += kBK, stage = (stage + 1) % kStages) {

        __syncthreads();
        const int prefetch_j0 = j0 + (kStages - 1) * kBK;
        if (prefetch_j0 < j_hi) {
            issue_tile(prefetch_j0, (stage + kStages - 1) % kStages);
        } else {
            __pipeline_commit();
        }
        __pipeline_wait_prior(kStages - 1);
        __syncthreads();
        const __nv_bfloat16* sK = sKbuf + stage * (kBK * kLdD);

        float sacc[kSNTiles][4];
        #pragma unroll
        for (int n = 0; n < kSNTiles; ++n) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) sacc[n][i] = 0.f;
        }
        {

            const __nv_bfloat16* kbase =
                sK + static_cast<long long>(warp * 8 * kSNTiles) * kLdD;
            #pragma unroll 2
            for (int k0 = 0; k0 < kD; k0 += 16) {
                std::uint32_t a[4];
                ld_a(a, sQ + k0, kLdD, lane);
                #pragma unroll
                for (int n = 0; n < kSNTiles; ++n) {
                    std::uint32_t b[2];
                    ld_b(b, kbase + static_cast<long long>(n * 8) * kLdD + k0, kLdD, lane);
                    mma_m16n8k16(sacc[n], a, b);
                }
            }
        }
        #pragma unroll
        for (int n = 0; n < kSNTiles; ++n) {
            const int ncol = (warp * kSNTiles + n) * 8 + p;
            sS[g * kBK + ncol] = sacc[n][0];
            sS[g * kBK + ncol + 1] = sacc[n][1];
            sS[(g + 8) * kBK + ncol] = sacc[n][2];
            sS[(g + 8) * kBK + ncol + 1] = sacc[n][3];
        }
        __syncthreads();

        {
            constexpr int kSubs = kThreads / kBM;
            constexpr int kColsPerSub = kBK / kSubs;
            const int row = tid / kSubs;
            const int sub = tid % kSubs;
            float v[kColsPerSub];
            float local = -CUDART_INF_F;
            #pragma unroll
            for (int i = 0; i < kColsPerSub; ++i) {
                const int jj = sub * kColsPerSub + i;
                float s = -CUDART_INF_F;
                if (s_ok[stage * kBK + jj]) {
                    s = sS[row * kBK + jj] * sm_scale;
                }
                v[i] = s;
                local = fmaxf(local, s);
            }
            #pragma unroll
            for (int m = 1; m < kSubs; m <<= 1) {
                local = fmaxf(local, __shfl_xor_sync(0xffffffffu, local, m));
            }
            const float m_prev = sM[row];
            const float m_new = fmaxf(m_prev, local);
            const bool empty = (m_new == -CUDART_INF_F);
            const float corr =
                empty ? 1.f : ((m_prev == -CUDART_INF_F) ? 0.f : __expf(m_prev - m_new));
            float lsum = 0.f;
            #pragma unroll
            for (int i = 0; i < kColsPerSub; ++i) {
                v[i] = empty ? 0.f : __expf(v[i] - m_new);
                lsum += v[i];
            }
            #pragma unroll
            for (int m = 1; m < kSubs; m <<= 1) {
                lsum += __shfl_xor_sync(0xffffffffu, lsum, m);
            }
            if (sub == 0) {
                sM[row] = m_new;
                sL[row] = sL[row] * corr + lsum;
                sCorr[row] = corr;
            }
            #pragma unroll
            for (int i = 0; i < kColsPerSub; ++i) {
                sP[row * kLdP + sub * kColsPerSub + i] = __float2bfloat16(v[i]);
            }
        }
        __syncthreads();

        const float c0 = sCorr[g], c1 = sCorr[g + 8];
        #pragma unroll
        for (int n = 0; n < kNTiles; ++n) {
            oacc[n][0] *= c0; oacc[n][1] *= c0;
            oacc[n][2] *= c1; oacc[n][3] *= c1;
        }
        #pragma unroll
        for (int k0 = 0; k0 < kBK; k0 += 16) {
            std::uint32_t a[4];
            ld_a(a, sP + k0, kLdP, lane);
            #pragma unroll
            for (int n = 0; n < kNTiles; ++n) {
                std::uint32_t b[2];
                ld_b_trans(b, sK + static_cast<long long>(k0) * kLdD + dbase + n * 8, kLdD, lane);
                mma_m16n8k16(oacc[n], a, b);
            }
        }
    }

    __syncthreads();
    const bool split = splits > 1;
    float* lse_at = split ? part_lse : lse;
    __nv_bfloat16* o_at = split ? part_o : o;
    const long long unit =
        split ? static_cast<long long>(t_row) * splits + blockIdx.z : static_cast<long long>(t_row);
    if (lse_at != nullptr && tid < kBM) {
        const float l = sL[tid];
        lse_at[unit * H + h0 + tid] =
            (l > 0.f) ? (sM[tid] + logf(l)) * 1.4426950408889634f : -CUDART_INF_F;
    }
    const float l0 = sL[g], l1 = sL[g + 8];
    const float i0 = (l0 > 0.f) ? (1.f / l0) : 0.f;
    const float i1 = (l1 > 0.f) ? (1.f / l1) : 0.f;
    __nv_bfloat16* o0 = o_at + (unit * H + h0 + g) * kCkv;
    __nv_bfloat16* o1 = o_at + (unit * H + h0 + g + 8) * kCkv;
    #pragma unroll
    for (int n = 0; n < kNTiles; ++n) {
        const int d = dbase + n * 8 + p;
        o0[d] = __float2bfloat16(oacc[n][0] * i0);
        o0[d + 1] = __float2bfloat16(oacc[n][1] * i0);
        o1[d] = __float2bfloat16(oacc[n][2] * i1);
        o1[d + 1] = __float2bfloat16(oacc[n][3] * i1);
    }
}

// `mla_mma_paged_kernel`'s split units of each row folded: a block a (row,
// head), the units' outputs weighted by 2^(lse - max) over their sum (at
// most 32 units, a lane each), the row's output and base-2 log-sum-exp (where
// asked) landed.
__global__ __launch_bounds__(128) void mla_mma_fold(
    const __nv_bfloat16* __restrict__ part_o, const float* __restrict__ part_lse,
    __nv_bfloat16* __restrict__ o, float* __restrict__ lse, int splits, int H,
    const u32* __restrict__ win)
{
    const int t = static_cast<int>(blockIdx.x);
    if (win != nullptr && t >= static_cast<int>(win[0])) return;
    const int t_row = win != nullptr ? t + static_cast<int>(win[1]) : t;
    const int h = static_cast<int>(blockIdx.y);
    const long long unit0 = static_cast<long long>(t_row) * splits;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const float mine = lane < splits ? part_lse[(unit0 + lane) * H + h] : -CUDART_INF_F;
    float top = mine;
    top = ::pie::warp_max(top);
    const float e = mine > -CUDART_INF_F ? exp2f(mine - top) : 0.f;
    float sum = e;
    sum = ::pie::warp_sum(sum);
    const float share = sum > 0.f ? e / sum : 0.f;
    const int d = static_cast<int>(threadIdx.x) * 4;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int s = 0; s < splits; ++s) {
        const float w = __shfl_sync(0xffffffffu, share, s);
        if (w == 0.f) continue;
        const __nv_bfloat16* src = part_o + ((unit0 + s) * H + h) * kCkv + d;
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] = fmaf(w, __bfloat162float(src[i]), acc[i]);
    }
    __nv_bfloat16* dst = o + (static_cast<long long>(t_row) * H + h) * kCkv + d;
    #pragma unroll
    for (int i = 0; i < 4; ++i) dst[i] = __float2bfloat16(acc[i]);
    if (lse != nullptr && threadIdx.x == 0) {
        lse[static_cast<long long>(t_row) * H + h] = sum > 0.f ? top + __log2f(sum) : -CUDART_INF_F;
    }
}

}

}

namespace mla_fa2 {

using DTypeQ = __nv_bfloat16;
using DTypeKV = __nv_bfloat16;
using DTypeO = __nv_bfloat16;
using IdType = std::int32_t;

using Params = ::pie::attn::fa2::MLAParams<DTypeQ, DTypeKV, DTypeO, IdType>;

inline constexpr std::uint32_t HEAD_DIM_CKV = 512;
inline constexpr std::uint32_t HEAD_DIM_KPE = 64;
inline constexpr std::uint32_t CTA_TILE_Q = 64;

template <bool CAUSAL, std::uint32_t NUM_STAGES, bool QK_SHARD, std::uint32_t CTA_TILE_KV>
using Traits =
    ::pie::attn::fa2::mla::KernelTraits<CAUSAL, NUM_STAGES, QK_SHARD, HEAD_DIM_CKV, HEAD_DIM_KPE,
                                   CTA_TILE_Q, CTA_TILE_KV, DTypeQ, DTypeKV, DTypeO, IdType>;

template <class KTraits>
__device__ unsigned smem_bytes_mla =
    static_cast<unsigned>(sizeof(typename KTraits::SharedStorage));

}

}
