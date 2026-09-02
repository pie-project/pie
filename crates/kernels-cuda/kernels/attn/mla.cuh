#pragma once

#include <cstdint>
#include <cstring>

#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <math_constants.h>

#include "prelude/device.cuh"
#include "prelude/rope.cuh"

#include "flashinfer/attention/mla.cuh"
#include "flashinfer/attention/mla_params.cuh"

namespace pie::attn {

template <class T>
__global__ void mla_split_q_b(
    const T* __restrict__ q_b,
    T* __restrict__ q_nope,
    T* __restrict__ q_pe,
    int total,
    int heads,
    int nope,
    int rope,
    const u32* __restrict__ win)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int per = nope + rope;
    const int d = i % per;
    const int h = (i / per) % heads;
    const int n = i / (heads * per);
    // The staged-geometry seat (qkv_fused.cuh's idiom): a replay whose grid
    // was carved at a bucket retires its padded rows here, off a word the
    // fire staged, not a parameter the recording baked.
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    // And WHERE those rows begin: an armed seat's pointers are plane bases,
    // so `win[1]` is the plane row this launch's first element belongs to.
    // The flat index has to be re-laid on it — the cut reads and writes the
    // same row axis.
    const int n_row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const T v = q_b[(static_cast<long long>(n_row) * heads + h) * per + d];
    if (d < nope) {
        q_nope[(static_cast<long long>(n_row) * heads + h) * nope + d] = v;
    } else {
        q_pe[(static_cast<long long>(n_row) * heads + h) * rope + (d - nope)] = v;
    }
}

template <class T, int BLOCK_DIM = 256>
__global__ void mla_latents(
    const T* __restrict__ kv_a,
    const T* __restrict__ norm_weight,
    T* __restrict__ kv_c,
    T* __restrict__ k_pe,
    int kv_lora,
    int rope,
    int src_row_stride,
    float eps,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;
    // The staged-geometry seat (qkv_fused.cuh's idiom): a replay whose grid
    // was carved at a bucket retires its padded rows here, off a word the
    // fire staged, not a parameter the recording baked.
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    // And WHERE those rows begin: an armed seat's pointers are plane bases,
    // so `win[1]` is the plane row this launch's first block owns — the fused
    // projection and both halves it is cut into share that row axis.
    const int n_row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int tid = threadIdx.x;
    const T* row = kv_a + static_cast<long long>(n_row) * src_row_stride;

    for (int d = tid; d < rope; d += BLOCK_DIM) {
        k_pe[static_cast<long long>(n_row) * rope + d] = row[kv_lora + d];
    }

    float local = 0.f;
    for (int d = tid; d < kv_lora; d += BLOCK_DIM) {
        const float v = Elem<T>::to_f32(row[d]);
        local += v * v;
    }
    __shared__ float buf[BLOCK_DIM];
    buf[tid] = local;
    __syncthreads();
    for (int off = BLOCK_DIM / 2; off > 0; off >>= 1) {
        if (tid < off) buf[tid] += buf[tid + off];
        __syncthreads();
    }
    const float inv_rms = rsqrtf(buf[0] / static_cast<float>(kv_lora) + eps);
    for (int d = tid; d < kv_lora; d += BLOCK_DIM) {
        const float v = Elem<T>::to_f32(row[d]);
        const float w = Elem<T>::to_f32(norm_weight[d]);
        kv_c[static_cast<long long>(n_row) * kv_lora + d] = Elem<T>::from_f32(v * inv_rms * w);
    }
}

constexpr int kMaxRopePairs = 128;

__device__ __forceinline__ int mla_find_request(const u32* qo_indptr,
                                                int R,
                                                int token_idx) {
    for (int r = 0; r < R; ++r) {
        if (token_idx < static_cast<int>(qo_indptr[r + 1])) return r;
    }
    return R - 1;
}

__device__ __forceinline__ void mla_resolve_dst(
    const u32* __restrict__ qo_indptr,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    int R,
    int page_size,
    int token_idx,
    int& actual_page,
    int& offset_in_page)
{
    const int r = mla_find_request(qo_indptr, R, token_idx);
    const int qo_lo = qo_indptr[r];
    const int qo_hi = qo_indptr[r + 1];
    const int new_tokens_r = qo_hi - qo_lo;
    const int offset_in_new = token_idx - qo_lo;
    const int pages_first = kv_page_indptr[r];
    const int pages_last = kv_page_indptr[r + 1];
    const int num_pages_r = pages_last - pages_first;
    const int total_kv_after =
        (num_pages_r - 1) * page_size + kv_last_page_lens[r];
    const int pre_kv_len = total_kv_after - new_tokens_r;
    const int abs_kv_pos = pre_kv_len + offset_in_new;
    const int page_in_req = abs_kv_pos / page_size;
    offset_in_page = abs_kv_pos % page_size;
    actual_page = static_cast<int>(kv_page_indices[pages_first + page_in_req]);
}

__global__ void mla_kv_append(
    const bf16* __restrict__ ckv_curr,
    const bf16* __restrict__ kpe_curr,
    bf16* __restrict__ ckv_pages,
    bf16* __restrict__ kpe_pages,
    const u32* __restrict__ qo_indptr,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    const u8* __restrict__ row_valid,
    int R,
    int page_size,
    int kv_lora_rank,
    int qk_rope_head_dim)
{
    const int t = blockIdx.x;
    if (row_valid != nullptr && row_valid[t] == 0) return;
    int actual_page = 0;
    int offset_in_page = 0;
    mla_resolve_dst(qo_indptr, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens, R, page_size, t, actual_page,
                    offset_in_page);

    const long long ckv_src = static_cast<long long>(t) * kv_lora_rank;
    const long long ckv_dst =
        (static_cast<long long>(actual_page) * page_size + offset_in_page) *
        kv_lora_rank;
    for (int i = threadIdx.x; i < kv_lora_rank; i += blockDim.x) {
        ckv_pages[ckv_dst + i] = ckv_curr[ckv_src + i];
    }

    const long long kpe_src = static_cast<long long>(t) * qk_rope_head_dim;
    const long long kpe_dst =
        (static_cast<long long>(actual_page) * page_size + offset_in_page) *
        qk_rope_head_dim;
    for (int i = threadIdx.x; i < qk_rope_head_dim; i += blockDim.x) {
        kpe_pages[kpe_dst + i] = kpe_curr[kpe_src + i];
    }
}

template <int BLOCK_DIM>
__global__ void mla_prepare(
    const bf16* __restrict__ kv_a,
    const bf16* __restrict__ kv_a_norm_w,
    const bf16* __restrict__ q_b,
    bf16* __restrict__ kv_c,
    bf16* __restrict__ k_pe,
    bf16* __restrict__ q_nope,
    bf16* __restrict__ q_pe,
    bf16* __restrict__ ckv_pages,
    bf16* __restrict__ kpe_pages,
    const i32* __restrict__ positions,
    const u32* __restrict__ qo_indptr,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    const u8* __restrict__ row_valid,
    int R,
    int page_size,
    int heads,
    int kv_lora,
    int nope,
    int rope,
    int src_row_stride,
    float eps,
    float theta,
    bool interleaved,
    int heads_per_block,

    float yarn_factor,
    float yarn_low_dim,
    float yarn_high_dim,
    float yarn_mscale)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const int half = rope / 2;
    const int pos = positions[n];

    __shared__ float cs[2 * kMaxRopePairs];
    const int cached = half <= kMaxRopePairs ? half : 0;
    const bool yarn = yarn_factor > 0.f;
    auto angle = [&](int dp, float& c, float& s_) {
        if (yarn) {
            rope_cos_sin_yarn_original(theta, dp, rope, pos, yarn_factor,
                                       yarn_low_dim, yarn_high_dim,
                                       yarn_mscale, c, s_);
        } else {
            rope_cos_sin(theta, dp, rope, pos, c, s_);
        }
    };
    for (int dp = tid; dp < cached; dp += BLOCK_DIM) {
        angle(dp, cs[dp], cs[cached + dp]);
    }
    if (cached > 0) __syncthreads();

    if (blockIdx.y == 0) {
        const bf16* row =
            kv_a + static_cast<long long>(n) * src_row_stride;
        bf16* kpe_out = k_pe + static_cast<long long>(n) * rope;

        for (int dp = tid; dp < half; dp += BLOCK_DIM) {
            float cos_v, sin_v;
            if (dp < cached) { cos_v = cs[dp]; sin_v = cs[cached + dp]; }
            else angle(dp, cos_v, sin_v);
            if (interleaved) {
                rotate_pair_interleaved_to(row + kv_lora, kpe_out, dp, cos_v, sin_v);
            } else {
                rotate_pair_to(row + kv_lora, kpe_out, half, dp, cos_v, sin_v);
            }
        }

        float local = 0.f;
        for (int d = tid; d < kv_lora; d += BLOCK_DIM) {
            const float v = bf16_to_f32(row[d]);
            local += v * v;
        }
        __shared__ float buf[BLOCK_DIM];
        buf[tid] = local;
        __syncthreads();
        for (int off = BLOCK_DIM / 2; off > 0; off >>= 1) {
            if (tid < off) buf[tid] += buf[tid + off];
            __syncthreads();
        }
        const float inv_rms =
            rsqrtf(buf[0] / static_cast<float>(kv_lora) + eps);

        bf16* kvc_out = kv_c + static_cast<long long>(n) * kv_lora;
        for (int d = tid; d < kv_lora; d += BLOCK_DIM) {
            const float v = bf16_to_f32(row[d]);
            const float w = bf16_to_f32(kv_a_norm_w[d]);
            kvc_out[d] = f32_to_bf16(v * inv_rms * w);
        }

        if (row_valid != nullptr && row_valid[n] == 0) return;
        __syncthreads();

        int actual_page = 0;
        int offset_in_page = 0;
        mla_resolve_dst(qo_indptr, kv_page_indices, kv_page_indptr,
                        kv_last_page_lens, R, page_size, n,
                        actual_page, offset_in_page);
        const long long slot =
            static_cast<long long>(actual_page) * page_size + offset_in_page;
        bf16* ckv_dst = ckv_pages + slot * kv_lora;
        for (int d = tid; d < kv_lora; d += BLOCK_DIM) ckv_dst[d] = kvc_out[d];
        bf16* kpe_dst = kpe_pages + slot * rope;
        for (int d = tid; d < rope; d += BLOCK_DIM) kpe_dst[d] = kpe_out[d];
        return;
    }

    const int per = nope + rope;
    const int head_base = (blockIdx.y - 1) * heads_per_block;
    const int heads_here = min(heads_per_block, heads - head_base);
    if (heads_here <= 0) return;

    const bf16* qb_row =
        q_b + (static_cast<long long>(n) * heads + head_base) * per;
    bf16* qn_row =
        q_nope + (static_cast<long long>(n) * heads + head_base) * nope;
    for (int i = tid; i < heads_here * nope; i += BLOCK_DIM) {
        const int h = i / nope;
        qn_row[i] = qb_row[static_cast<long long>(h) * per + (i - h * nope)];
    }

    bf16* qp_row =
        q_pe + (static_cast<long long>(n) * heads + head_base) * rope;
    for (int i = tid; i < heads_here * half; i += BLOCK_DIM) {
        const int h = i / half;
        const int dp = i - h * half;
        float cos_v, sin_v;
        if (dp < cached) { cos_v = cs[dp]; sin_v = cs[cached + dp]; }
        else angle(dp, cos_v, sin_v);
        const bf16* src =
            qb_row + static_cast<long long>(h) * per + nope;
        bf16* dst = qp_row + static_cast<long long>(h) * rope;
        if (interleaved) rotate_pair_interleaved_to(src, dst, dp, cos_v, sin_v);
        else rotate_pair_to(src, dst, half, dp, cos_v, sin_v);
    }
}

namespace mla_naive {

constexpr int kMlaNaiveBlock = 256;
constexpr int kMlaNaiveWarps = kMlaNaiveBlock / 32;
constexpr int kMlaNaiveMaxPer = 16;
constexpr int kMlaNaiveMaxPePer = 4;

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
    const std::int32_t* __restrict__ selection, int top_k,
    int R, int H, int CKV, int KPE, int page_size, float sm_scale, bool causal,
    int G,
    const u32* __restrict__ win)
{
    const int t = blockIdx.x;
    // The staged-geometry seat (qkv_fused.cuh's idiom): a replay whose grid
    // was carved at a bucket retires its padded rows here, off a word the
    // fire staged, not a parameter the recording baked.
    if (win != nullptr && t >= static_cast<int>(win[0])) return;
    // And WHERE those rows begin: an armed seat's pointers are plane bases,
    // so `win[1]` is the plane row this launch's first block owns. Only the
    // PLANES move by it — the qo boundaries this searches are the window's
    // own, rebased to its zero, so the token ordinal they answer stays `t`.
    const int t_row = win != nullptr ? t + static_cast<int>(win[1]) : t;
    // `R` may be the key's lane ceiling; `win[2]` is the live request count.
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

    extern __shared__ float smem[];
    float* wacc  = smem;
    float* wm    = wacc + kMlaNaiveWarps * CKV;
    float* wl    = wm + kMlaNaiveWarps;

    const __nv_bfloat16* qn =
        q_nope + (static_cast<long long>(t_row) * H + h) * CKV;
    const __nv_bfloat16* qp =
        q_pe + (static_cast<long long>(t_row) * H + h) * KPE;
    float qn_r[kMlaNaiveMaxPer];
    float qp_r[kMlaNaiveMaxPePer];
    for (int i = 0; i < per; ++i) qn_r[i] = __bfloat162float(qn[lane + i * 32]);
    for (int i = 0; i < pper; ++i) qp_r[i] = __bfloat162float(qp[lane + i * 32]);

    float acc[kMlaNaiveMaxPer];
    for (int i = 0; i < per; ++i) acc[i] = 0.f;
    float m = -CUDART_INF_F, lsum = 0.f;

    const int steps = (srow != nullptr) ? top_k : j_end;
    for (int n = s; n < steps; n += K) {
        int j = n;
        if (srow != nullptr) {
            j = srow[n];
            if (j < 0 || j >= j_end) continue;
        }
        const int page =
            static_cast<int>(kv_page_indices[pages_first + j / page_size]);
        const int off = j % page_size;
        const __nv_bfloat16* ckv_j =
            ckv_pages + (static_cast<long long>(page) * page_size + off) * CKV;
        const __nv_bfloat16* kpe_j =
            kpe_pages + (static_cast<long long>(page) * page_size + off) * KPE;

        float kv[kMlaNaiveMaxPer];
        float pd = 0.f;
        for (int i = 0; i < per; ++i) {
            kv[i] = __bfloat162float(ckv_j[lane + i * 32]);
            pd += qn_r[i] * kv[i];
        }
        for (int i = 0; i < pper; ++i) {
            pd += qp_r[i] * __bfloat162float(kpe_j[lane + i * 32]);
        }
        #pragma unroll
        for (int sh = 16; sh > 0; sh >>= 1) {
            pd += __shfl_xor_sync(0xffffffffu, pd, sh);
        }
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
constexpr int kKpe = 64;
constexpr int kD = kCkv + kKpe;
constexpr int kLdD = kD + 8;
constexpr int kLdP = kBK + 8;
constexpr int kDimsPerWarp = kCkv / kWarps;
constexpr int kNTiles = kDimsPerWarp / 8;
constexpr int kSNTiles = kBK / 8 / kWarps;
static_assert(kSNTiles >= 1, "kBK must be at least 8 * kWarps");
static_assert(kBK % 16 == 0, "kBK must be a multiple of the mma k step");

__device__ __forceinline__ std::uint32_t pack2(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    std::uint32_t l, h;
    std::memcpy(&l, &lo, 2);
    std::memcpy(&h, &hi, 2);
    return (l & 0xffffu) | ((h & 0xffffu) << 16);
}

__device__ __forceinline__ void mma_m16n8k16(float (&d)[4], const std::uint32_t (&a)[4],
                                             const std::uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void ld_a(std::uint32_t (&a)[4], const __nv_bfloat16* base,
                                     int ld, int lane, int k0) {
    const __nv_bfloat16* r =
        base + static_cast<long long>(lane & 15) * ld + k0 + ((lane & 16) ? 8 : 0);
    const std::uint32_t addr =
        static_cast<std::uint32_t>(__cvta_generic_to_shared(r));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
        : "r"(addr));
}

__device__ __forceinline__ void ld_b_k(std::uint32_t (&b)[2], const __nv_bfloat16* base,
                                       int ld, int lane, int k0) {
    const __nv_bfloat16* q =
        base + static_cast<long long>(lane & 7) * ld + k0 + ((lane & 8) ? 8 : 0);
    const std::uint32_t addr =
        static_cast<std::uint32_t>(__cvta_generic_to_shared(q));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(b[0]), "=r"(b[1])
        : "r"(addr));
}

__device__ __forceinline__ void ld_b_v(std::uint32_t (&b)[2], const __nv_bfloat16* sK,
                                       int ld, int lane, int k0, int nbase) {

    const __nv_bfloat16* q =
        sK + static_cast<long long>(k0 + (lane & 15)) * ld + nbase;
    const std::uint32_t addr =
        static_cast<std::uint32_t>(__cvta_generic_to_shared(q));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(b[0]), "=r"(b[1])
        : "r"(addr));
}

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
    int R, int H, int page_size, float sm_scale, bool causal,
    const u32* __restrict__ win)
{
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
    // The staged-geometry seat (qkv_fused.cuh's idiom): a replay whose grid
    // was carved at a bucket retires its padded rows here, off a word the
    // fire staged, not a parameter the recording baked.
    if (win != nullptr && t >= static_cast<int>(win[0])) return;
    // And WHERE those rows begin: an armed seat's pointers are plane bases,
    // so `win[1]` is the plane row this launch's first block owns. Only the
    // PLANES move by it — the qo boundaries this scans are the window's own,
    // rebased to its zero, so the token ordinal they answer stays `t`.
    const int t_row = win != nullptr ? t + static_cast<int>(win[1]) : t;
    // `R` may be the key's lane ceiling; `win[2]` is the live request count.
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
    auto issue_tile = [&](int j0, int stage) {
        for (int rr = tid; rr < kBK; rr += kThreads) {
            const int j = j0 + rr;
            s_slot[rr] =
                (j < j_end)
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
            if (slot >= 0) {
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
        const int j0 = s * kBK;
        if (j0 < j_end) issue_tile(j0, s);
    }

    int stage = 0;
    for (int j0 = 0; j0 < j_end; j0 += kBK, stage = (stage + 1) % kStages) {

        __syncthreads();
        const int prefetch_j0 = j0 + (kStages - 1) * kBK;
        if (prefetch_j0 < j_end) {
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
                ld_a(a, sQ, kLdD, lane, k0);
                #pragma unroll
                for (int n = 0; n < kSNTiles; ++n) {
                    std::uint32_t b[2];
                    ld_b_k(b, kbase + static_cast<long long>(n * 8) * kLdD,
                           kLdD, lane, k0);
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
                const int j = j0 + jj;
                float s = -CUDART_INF_F;
                if (j < j_end) {
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
            ld_a(a, sP, kLdP, lane, k0);
            #pragma unroll
            for (int n = 0; n < kNTiles; ++n) {
                std::uint32_t b[2];
                ld_b_v(b, sK, kLdD, lane, k0, dbase + n * 8);
                mma_m16n8k16(oacc[n], a, b);
            }
        }
    }

    __syncthreads();
    const float l0 = sL[g], l1 = sL[g + 8];
    const float i0 = (l0 > 0.f) ? (1.f / l0) : 0.f;
    const float i1 = (l1 > 0.f) ? (1.f / l1) : 0.f;
    __nv_bfloat16* o0 = o + (static_cast<long long>(t_row) * H + h0 + g) * kCkv;
    __nv_bfloat16* o1 = o + (static_cast<long long>(t_row) * H + h0 + g + 8) * kCkv;
    #pragma unroll
    for (int n = 0; n < kNTiles; ++n) {
        const int d = dbase + n * 8 + p;
        o0[d] = __float2bfloat16(oacc[n][0] * i0);
        o0[d + 1] = __float2bfloat16(oacc[n][1] * i0);
        o1[d] = __float2bfloat16(oacc[n][2] * i1);
        o1[d + 1] = __float2bfloat16(oacc[n][3] * i1);
    }
}

}

}

namespace mla_fa2 {

using DTypeQ = __nv_bfloat16;
using DTypeKV = __nv_bfloat16;
using DTypeO = __nv_bfloat16;
using IdType = std::int32_t;

using Params = ::flashinfer::MLAParams<DTypeQ, DTypeKV, DTypeO, IdType>;

inline constexpr std::uint32_t HEAD_DIM_CKV = 512;
inline constexpr std::uint32_t HEAD_DIM_KPE = 64;
inline constexpr std::uint32_t CTA_TILE_Q = 64;

template <bool CAUSAL, std::uint32_t NUM_STAGES, bool QK_SHARD, std::uint32_t CTA_TILE_KV>
using Traits =
    ::flashinfer::mla::KernelTraits<CAUSAL, NUM_STAGES, QK_SHARD, HEAD_DIM_CKV, HEAD_DIM_KPE,
                                   CTA_TILE_Q, CTA_TILE_KV, DTypeQ, DTypeKV, DTypeO, IdType>;

template <class KTraits>
__device__ unsigned smem_bytes_mla =
    static_cast<unsigned>(sizeof(typename KTraits::SharedStorage));

}

}
