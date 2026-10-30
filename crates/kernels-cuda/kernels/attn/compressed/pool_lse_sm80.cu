// `pool_lse_paged` (both forms) on tensor cores. Every query
// head of a row reads the same compressed rows (one latent, its own key and
// value), so a block serves one row's 16 heads (an mma's M) and walks the
// row's compressed rows 16 at a time: gathered once into shared memory, then
// S = Q K^T with the head width split across the four warps and folded in
// shared memory, an online softmax every warp keeps alike, and O += P K with
// each warp owning a quarter of the head width. A row reads exactly the
// compressed rows the scalar walk reads; P meets the keys in bf16.

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <math_constants.h>

#include "attn/compressed/pool.cuh"
#include "attn/selected.cuh"

namespace pie::attn::compressed {

using namespace pie::attn::selected::mma;

constexpr int kPoolWarps = 4;
constexpr int kPoolThreads = kPoolWarps * 32;
constexpr int kPoolKeys = 16;
constexpr int kPoolHeads = 16;

template <int D>
struct PoolShape {
    static constexpr int kLd = D + 8;
    static constexpr int kQBytes = kPoolHeads * kLd * 2;
    static constexpr int kKBytes = kPoolKeys * kLd * 2;
    // Q, two key buffers, each warp's partial scores (16 x 16 f32).
    static constexpr int kSBytes = kPoolWarps * kPoolHeads * kPoolKeys * 4;
    static constexpr int kBytes = 2 * kKBytes + kSBytes;
};

// `kSelected`: the row's `top_k` ids in `selection`, ended early by a negative
// one; otherwise every compressed row the query sees.
template <int D, bool kSelected, class SinkT = bf16>
__global__ __launch_bounds__(kPoolThreads) void pool_lse_mma(
    const bf16* __restrict__ q,
    const bf16* __restrict__ comp_kv_pages,
    const i32* __restrict__ selection,
    bf16* __restrict__ o,
    float* __restrict__ lse_out,
    const i32* __restrict__ positions,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const i32* __restrict__ req_of_token,
    int num_q_heads,
    int ratio,
    int top_k,
    int page_size,
    float scale,
    int compact,
    const bf16* __restrict__ win_o,
    const float* __restrict__ win_lse,
    const SinkT* __restrict__ sinks,
    const u32* __restrict__ win)
{
    using S = PoolShape<D>;
    constexpr int kLd = S::kLd;
    constexpr int kChunks = D / 8;
    constexpr int kWarpDims = D / kPoolWarps;
    constexpr int kWarpTiles = kWarpDims / 8;
    static_assert(D % 64 == 0, "a warp's quarter of the head width is whole 16-wide steps");

    const int qi = static_cast<int>(blockIdx.x);
    if (win != nullptr && qi >= static_cast<int>(win[0])) return;
    const int qi_row = win != nullptr ? qi + static_cast<int>(win[1]) : qi;
    const int head0 = static_cast<int>(blockIdx.y) * kPoolHeads;
    const int heads = min(kPoolHeads, num_q_heads - head0);

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g = lane >> 2, p = (lane & 3) << 1;

    extern __shared__ __align__(16) char smem_raw[];
    bf16* sQ = reinterpret_cast<bf16*>(smem_raw);
    bf16* sK0 = reinterpret_cast<bf16*>(smem_raw);
    float* sS = reinterpret_cast<float*>(smem_raw + 2 * S::kKBytes);
    __shared__ long long s_slot[2][kPoolKeys];
    __shared__ int s_end;

    const int req = req_of_token[qi_row];
    const int qpos = positions[qi_row];
    const int num_visible =
        min((qpos + 1) / ratio, owned_positions(kv_page_indptr, req, page_size) / ratio);
    const i32* srow = kSelected ? selection + static_cast<long long>(qi_row) * top_k : nullptr;

    // The walk's end: the first negative id (selected), else every visible row.
    if (tid == 0) s_end = kSelected ? top_k : max(num_visible, 0);
    __syncthreads();
    if (kSelected) {
        for (int n = tid; n < top_k; n += kPoolThreads)
            if (srow[n] < 0) atomicMin(&s_end, n);
    }

    for (int c = tid; c < kPoolHeads * kChunks; c += kPoolThreads) {
        const int row = c / kChunks;
        const int d = (c % kChunks) * 8;
        int4 v = make_int4(0, 0, 0, 0);
        if (row < heads) {
            v = *reinterpret_cast<const int4*>(
                q + (static_cast<long long>(qi_row) * num_q_heads + head0 + row) * D + d);
        }
        *reinterpret_cast<int4*>(sQ + row * kLd + d) = v;
    }
    __syncthreads();
    std::uint32_t qa[kWarpDims / 16][4];
#pragma unroll
    for (int i = 0; i < kWarpDims / 16; ++i) ld_a(qa[i], sQ + warp * kWarpDims + i * 16, kLd, lane);
    __syncthreads();
    const int end = num_visible > 0 ? s_end : 0;

    // A chunk's compressed rows: their slots, then their latents.
    auto stage = [&](int base, int buf) {
        if (tid < kPoolKeys) {
            const int n = base + tid;
            int c = -1;
            if (n < end) {
                c = kSelected ? srow[n] : n;
                if (c < 0 || c >= num_visible) c = -1;
            }
            s_slot[buf][tid] = c < 0 ? -1
                : entry_slot(kv_page_indices, kv_page_indptr, req, c, ratio, page_size, compact);
        }
        __syncthreads();
        bf16* sK = sK0 + buf * kPoolKeys * kLd;
        for (int c = tid; c < kPoolKeys * kChunks; c += kPoolThreads) {
            const int key = c / kChunks;
            const int d = (c % kChunks) * 8;
            const long long slot = s_slot[buf][key];
            bf16* kd = sK + key * kLd + d;
            if (slot >= 0) {
                __pipeline_memcpy_async(kd, comp_kv_pages + slot * D + d, 16);
            } else {
                *reinterpret_cast<int4*>(kd) = make_int4(0, 0, 0, 0);
            }
        }
        __pipeline_commit();
    };

    float oacc[kWarpTiles][4];
#pragma unroll
    for (int n = 0; n < kWarpTiles; ++n)
#pragma unroll
        for (int i = 0; i < 4; ++i) oacc[n][i] = 0.f;
    float m_lo = -CUDART_INF_F, m_hi = -CUDART_INF_F, l_lo = 0.f, l_hi = 0.f;

    if (end > 0) stage(0, 0);
    int buf = 0;
    for (int base = 0; base < end; base += kPoolKeys, buf ^= 1) {
        if (base + kPoolKeys < end) {
            stage(base + kPoolKeys, buf ^ 1);
            __pipeline_wait_prior(1);
        } else {
            __pipeline_wait_prior(0);
        }
        __syncthreads();
        const bf16* sK = sK0 + buf * kPoolKeys * kLd;

        // This warp's quarter of the head width: partial S of 16 heads x 16 keys.
        float sacc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
        for (int k0 = warp * kWarpDims; k0 < (warp + 1) * kWarpDims; k0 += 16) {
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                std::uint32_t b[2];
                ld_b(b, sK + nt * 8 * kLd + k0, kLd, lane);
                mma_m16n8k16(sacc[nt], qa[(k0 - warp * kWarpDims) / 16], b);
            }
        }
        float* mine = sS + warp * kPoolHeads * kPoolKeys;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            mine[g * kPoolKeys + nt * 8 + p] = sacc[nt][0];
            mine[g * kPoolKeys + nt * 8 + p + 1] = sacc[nt][1];
            mine[(g + 8) * kPoolKeys + nt * 8 + p] = sacc[nt][2];
            mine[(g + 8) * kPoolKeys + nt * 8 + p + 1] = sacc[nt][3];
        }
        __syncthreads();
        // The whole score, in every warp's fragment layout.
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = g + (i >> 1) * 8;
                const int key = nt * 8 + p + (i & 1);
                float s = 0.f;
#pragma unroll
                for (int w = 0; w < kPoolWarps; ++w)
                    s += sS[(w * kPoolHeads + row) * kPoolKeys + key];
                sacc[nt][i] = s_slot[buf][key] >= 0 ? s * scale : -CUDART_INF_F;
            }
        }
        float mx_lo = -CUDART_INF_F, mx_hi = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            mx_lo = fmaxf(mx_lo, fmaxf(sacc[nt][0], sacc[nt][1]));
            mx_hi = fmaxf(mx_hi, fmaxf(sacc[nt][2], sacc[nt][3]));
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
        float rl = pr[0][0] + pr[0][1] + pr[1][0] + pr[1][1];
        float rh = pr[0][2] + pr[0][3] + pr[1][2] + pr[1][3];
#pragma unroll
        for (int sh = 1; sh < 4; sh <<= 1) {
            rl += __shfl_xor_sync(0xffffffffu, rl, sh);
            rh += __shfl_xor_sync(0xffffffffu, rh, sh);
        }
        l_lo = l_lo * c_lo + rl;
        l_hi = l_hi * c_hi + rh;
        const std::uint32_t pa[4] = {
            pack_bf16x2(pr[0][0], pr[0][1]),
            pack_bf16x2(pr[0][2], pr[0][3]),
            pack_bf16x2(pr[1][0], pr[1][1]),
            pack_bf16x2(pr[1][2], pr[1][3]),
        };
#pragma unroll
        for (int n = 0; n < kWarpTiles; ++n) {
            oacc[n][0] *= c_lo;
            oacc[n][1] *= c_lo;
            oacc[n][2] *= c_hi;
            oacc[n][3] *= c_hi;
            std::uint32_t b[2];
            ld_b_trans(b, sK + warp * kWarpDims + n * 8, kLd, lane);
            mma_m16n8k16(oacc[n], pa, b);
        }
        // Every warp is past this chunk's scores and keys before they are
        // overwritten.
        __syncthreads();
    }

    const float inv_lo = l_lo > 0.f ? 1.f / l_lo : 0.f;
    const float inv_hi = l_hi > 0.f ? 1.f / l_hi : 0.f;
    if (win_o != nullptr) {
        // The window's partial folded in as `merge_lse_combine` folds it and
        // the sinks applied as `sink_rescale` applies them, into `o`, each
        // rounded to bf16 where the three launches rounded it.
        struct Fold {
            int mode;  // 0: the window alone, 1: the pool alone, 2: both
            float w1, w2, inv_total, r;
        };
        auto fold = [&](int h, float l, float m) {
            constexpr float kLog2e = 1.44269504088896340736f;
            constexpr float kLn2 = 0.69314718055994530942f;
            const float lp = l > 0.f ? (logf(l) + m) * kLog2e : -CUDART_INF_F;
            const float lw = win_lse[qi_row * num_q_heads + head0 + h];
            Fold f{0, 0.f, 0.f, 0.f, 1.f};
            float lse = lw;
            if (isfinite(lp)) {
                if (!isfinite(lw)) {
                    f.mode = 1;
                    lse = lp;
                } else {
                    f.mode = 2;
                    const float lse_max = fmaxf(lw, lp);
                    f.w1 = exp2f(lw - lse_max);
                    f.w2 = exp2f(lp - lse_max);
                    f.inv_total = 1.0f / (f.w1 + f.w2);
                    lse = lse_max + log2f(f.w1 + f.w2);
                }
            }
            if (isfinite(lse)) {
                const float diff = lse * kLn2 - Elem<SinkT>::to_f32(sinks[head0 + h]);
                f.r = 1.0f / (1.0f + __expf(-diff));
            }
            return f;
        };
        auto land = [&](const Fold& f, long long at, float acc) {
            const float v1 = Elem<bf16>::to_f32(win_o[at]);
            const float v2 = Elem<bf16>::to_f32(f32_to_bf16(acc));
            const float merged = f.mode == 0 ? v1
                : f.mode == 1 ? v2
                : Elem<bf16>::to_f32(f32_to_bf16((v1 * f.w1 + v2 * f.w2) * f.inv_total));
            o[at] = f32_to_bf16(merged * f.r);
        };
        const Fold lo = g < heads ? fold(g, l_lo, m_lo) : Fold{};
        const Fold hi = g + 8 < heads ? fold(g + 8, l_hi, m_hi) : Fold{};
#pragma unroll
        for (int n = 0; n < kWarpTiles; ++n) {
            const int d = warp * kWarpDims + n * 8 + p;
            if (g < heads) {
                const long long at = (static_cast<long long>(qi_row) * num_q_heads + head0 + g) * D + d;
                land(lo, at, oacc[n][0] * inv_lo);
                land(lo, at + 1, oacc[n][1] * inv_lo);
            }
            if (g + 8 < heads) {
                const long long at = (static_cast<long long>(qi_row) * num_q_heads + head0 + g + 8) * D + d;
                land(hi, at, oacc[n][2] * inv_hi);
                land(hi, at + 1, oacc[n][3] * inv_hi);
            }
        }
        return;
    }
#pragma unroll
    for (int n = 0; n < kWarpTiles; ++n) {
        const int d = warp * kWarpDims + n * 8 + p;
        if (g < heads) {
            bf16* row = o + (static_cast<long long>(qi_row) * num_q_heads + head0 + g) * D + d;
            row[0] = f32_to_bf16(oacc[n][0] * inv_lo);
            row[1] = f32_to_bf16(oacc[n][1] * inv_lo);
        }
        if (g + 8 < heads) {
            bf16* row = o + (static_cast<long long>(qi_row) * num_q_heads + head0 + g + 8) * D + d;
            row[0] = f32_to_bf16(oacc[n][2] * inv_hi);
            row[1] = f32_to_bf16(oacc[n][3] * inv_hi);
        }
    }
    if (lse_out != nullptr && warp == 0 && (lane & 3) == 0) {
        constexpr float kLog2e = 1.44269504088896340736f;
        if (g < heads)
            lse_out[qi_row * num_q_heads + head0 + g] =
                l_lo > 0.f ? (logf(l_lo) + m_lo) * kLog2e : -CUDART_INF_F;
        if (g + 8 < heads)
            lse_out[qi_row * num_q_heads + head0 + g + 8] =
                l_hi > 0.f ? (logf(l_hi) + m_hi) * kLog2e : -CUDART_INF_F;
    }
}

}  // namespace pie::attn::compressed
