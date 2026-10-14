#pragma once

// Attention over the paged KV cache: decode (one query row per request) and
// prefill (a ragged block of query rows per request), driven by the work
// list the host scheduler (attn/sched_*.rs) staged. A work item is one
// (request, query tile, kv chunk); a split schedule writes f32 partials and
// their log-sum-exp per chunk, and `merge_partials` folds them.
//
// Scores live in the log2 domain: s = q·k · sm_scale · log2(e), so a
// partial's lse is m + log2(Σ 2^(s - m)), the convention every consumer of
// the lse plane reads.

#include "prelude/device.hip"
#include "prelude/mfma.hip"
#include "attn/kv.hip"

namespace pie::attn {

// Mirrors `attn::paged::Args` field for field; passed by value.
struct Paged {
    const bf16* q;
    const bf16* k;
    const bf16* v;
    const i32* page_indices;
    const i32* page_indptr;
    const i32* last_page_len;
    const i32* q_indptr;
    const i32* o_indptr;
    const i32* request_indices;
    const i32* qo_tile_indices;
    const i32* kv_tile_indices;
    const i32* kv_chunk_size;
    const u8* block_valid;
    void* o;
    float* lse;
    i32 num_qo_heads;
    i32 num_kv_heads;
    i32 page_size;
    i32 q_stride_n;
    i32 q_stride_h;
    i32 window_left;
    float scale_log2;
    i32 partition;
};

__device__ __forceinline__ int kv_length(const Paged& p, int req) {
    const int pages = p.page_indptr[req + 1] - p.page_indptr[req];
    return pages <= 0 ? 0 : (pages - 1) * p.page_size + p.last_page_len[req];
}

// The element offset of token `t`, head `h`, element `j` of request `req`.
template <bool HND, int D>
__device__ __forceinline__ long long kv_token(const Paged& p, int req, int t, int h, int j) {
    const long long page = p.page_indices[p.page_indptr[req] + t / p.page_size];
    return kv_at<HND>(page, t % p.page_size, h, j, p.page_size, p.num_kv_heads, D);
}

// ---------------------------------------------------------------- decode --

// One block per (work item, kv head), WAVES wavefronts splitting the chunk's
// tokens. Lane i of a wavefront scores token t0 + i for every query head of
// the GQA group, the wavefront keeps an online softmax per head, and each
// lane accumulates its D/kWave columns of the output.
template <int D, int G, int WAVES, bool HND>
__global__ __launch_bounds__(kWave* WAVES) void paged_decode(const Paged p) {
    constexpr int kCols = D / kWave;
    static_assert(D % kWave == 0, "a decode head is a whole number of lanes wide");
    __shared__ float qs[G][D];
    __shared__ float part_m[WAVES][G], part_l[WAVES][G];
    __shared__ float part_o[WAVES][G][D];

    const int bx = blockIdx.x;
    const int kv_head = blockIdx.y;
    if (p.block_valid != nullptr && p.block_valid[bx] == 0) return;
    const int req = p.request_indices[bx];
    const int kv_len = kv_length(p, req);
    const int chunk = *p.kv_chunk_size;
    const int start = p.partition ? p.kv_tile_indices[bx] * chunk : 0;
    const int end = p.partition ? min(start + chunk, kv_len) : kv_len;
    const int wave = threadIdx.x / kWave;
    const int ln = lane();

    // The window keeps keys at or after kv_len - 1 - window_left.
    const int lo = p.window_left >= 0 ? max(start, kv_len - 1 - p.window_left) : start;

    const bf16* qrow = p.q + static_cast<long long>(p.q_indptr[req]) * p.q_stride_n;
    for (int i = threadIdx.x; i < G * D; i += blockDim.x) {
        const int g = i / D, j = i % D;
        qs[g][j] = to_f32(qrow[(kv_head * G + g) * p.q_stride_h + j]) * p.scale_log2;
    }
    __syncthreads();

    float m[G], l[G], acc[G][kCols];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        m[g] = neg_inf();
        l[g] = 0.f;
#pragma unroll
        for (int c = 0; c < kCols; ++c) acc[g][c] = 0.f;
    }

    for (int t0 = lo + wave * kWave; t0 < end; t0 += WAVES * kWave) {
        const int t = t0 + ln;
        const bool live = t < end;
        float s[G];
#pragma unroll
        for (int g = 0; g < G; ++g) s[g] = 0.f;
        if (live) {
            const auto* kp = reinterpret_cast<const Pack8<bf16>*>(p.k + kv_token<HND, D>(p, req, t, kv_head, 0));
            for (int j8 = 0; j8 < D / 8; ++j8) {
                const Pack8<bf16> kv8 = kp[j8];
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float kf = to_f32(kv8.v[e]);
#pragma unroll
                    for (int g = 0; g < G; ++g) s[g] += qs[g][j8 * 8 + e] * kf;
                }
            }
        }
        float pr[G];
#pragma unroll
        for (int g = 0; g < G; ++g) {
            const float m_new = fmaxf(m[g], wave_max(live ? s[g] : neg_inf()));
            const float alpha = exp2f(m[g] - m_new);
            pr[g] = live ? exp2f(s[g] - m_new) : 0.f;
            l[g] = l[g] * alpha + wave_sum(pr[g]);
            m[g] = m_new;
#pragma unroll
            for (int c = 0; c < kCols; ++c) acc[g][c] *= alpha;
        }
        const int tiles = min(kWave, end - t0);
        for (int i = 0; i < tiles; ++i) {
            const bf16* vp = p.v + kv_token<HND, D>(p, req, t0 + i, kv_head, ln * kCols);
            float vf[kCols];
#pragma unroll
            for (int c = 0; c < kCols; ++c) vf[c] = to_f32(vp[c]);
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const float pi = __shfl(pr[g], i, kWave);
#pragma unroll
                for (int c = 0; c < kCols; ++c) acc[g][c] += pi * vf[c];
            }
        }
    }

    // Fold the wavefronts: each one's state rescaled to the block's max.
#pragma unroll
    for (int g = 0; g < G; ++g) {
        if (ln == 0) {
            part_m[wave][g] = m[g];
            part_l[wave][g] = l[g];
        }
#pragma unroll
        for (int c = 0; c < kCols; ++c) part_o[wave][g][ln * kCols + c] = acc[g][c];
    }
    __syncthreads();

    // A split writes its work item's partial row; an unsplit decode the
    // request's own query row of the plane.
    const long long row = static_cast<long long>(p.partition ? bx : p.q_indptr[req]) * p.num_qo_heads;
    for (int i = threadIdx.x; i < G * D; i += blockDim.x) {
        const int g = i / D, j = i % D;
        float big = neg_inf();
#pragma unroll
        for (int w = 0; w < WAVES; ++w) big = fmaxf(big, part_m[w][g]);
        float sum = 0.f, out = 0.f;
        if (big != neg_inf()) {
#pragma unroll
            for (int w = 0; w < WAVES; ++w) {
                const float f = exp2f(part_m[w][g] - big);
                sum += part_l[w][g] * f;
                out += part_o[w][g][j] * f;
            }
        }
        const float value = sum > 0.f ? out / sum : 0.f;
        const long long at = (row + kv_head * G + g) * D + j;
        if (p.partition) {
            static_cast<float*>(p.o)[at] = value;
        } else {
            static_cast<bf16*>(p.o)[at] = from_f32<bf16>(value);
        }
        if (j == 0 && p.lse != nullptr) {
            p.lse[row + kv_head * G + g] = sum > 0.f ? big + log2f(sum) : neg_inf();
        }
    }
}

// The most heads whose wavefront outputs fold through LDS at once: a
// divisor of G within 48 KiB.
__host__ __device__ constexpr int decode_fold_heads(int G, int D, int waves) {
    for (int h = G; h > 1; --h)
        if (G % h == 0 && waves * h * D * 4 <= 48 * 1024) return h;
    return 1;
}

// The decode on the matrix core. Per wavefront, SUB keys a round:
//
//   Sᵀ (16 keys × 16) = K · Qᵀ     A = K rows straight from the pages, B = Qᵀ
//                                   (column = query head, G of 16 live)
//
// so lane l holds keys 4(l/16)+t of each 16-key tile for head l%16, and the
// online softmax of head g lives in the lanes with l%16 = g. The weights go
// through LDS to the value side, where each lane owns 8 columns of the head
// for one of 512/D key parities and accumulates P·V on the vector units from
// 16-byte value reads. Eight wavefronts interleave their rounds over the
// chunk (a 256-key chunk is one round each) and fold through LDS at the end,
// as the other decode does.
// Eight wavefronts a block; four at 512-wide heads, whose round needs more
// than the 256 registers a wavefront gets at two a SIMD.
template <int D>
constexpr int kDecodeMfmaWaves = D >= 512 ? 4 : 8;

template <int D, int G, bool HND>
__global__ __launch_bounds__(64 * kDecodeMfmaWaves<D>) void paged_decode_mfma(const Paged p) {
    constexpr int WAVES = kDecodeMfmaWaves<D>, SUB = 16, KT = SUB / 16, KP = D / 32;
    constexpr int LPK = D / 8;           // lanes a key's value row spans
    constexpr int KPI = kWave / LPK;     // keys one value read covers
    constexpr int VR = SUB / KPI;        // value reads a round
    static_assert(G <= 16 && D % 32 == 0 && kWave % LPK == 0, "a group within one column tile");
    __shared__ float ps[WAVES][SUB][G];
    __shared__ float alph[WAVES][G];
    __shared__ float part_m[WAVES][G], part_l[WAVES][G];
    // The wavefronts' outputs fold through LDS HG heads at a time: all G at
    // once where they fit beside the rest (D = 512 does not).
    constexpr int HG = decode_fold_heads(G, D, WAVES);
    __shared__ float part_o[WAVES][HG][D];
    // The chunk's page table, a window of TABLE pages at a time: a key's
    // address then needs no read of global memory before its own.
    constexpr int TABLE = 512;
    __shared__ i32 table[TABLE];

    const int bx = blockIdx.x;
    const int kv_head = blockIdx.y;
    if (p.block_valid != nullptr && p.block_valid[bx] == 0) return;
    const int req = p.request_indices[bx];
    const int kv_len = kv_length(p, req);
    const int chunk = *p.kv_chunk_size;
    const int start = p.partition ? p.kv_tile_indices[bx] * chunk : 0;
    const int end = p.partition ? min(start + chunk, kv_len) : kv_len;
    const int wave = threadIdx.x / kWave, l = lane();
    const int lo = p.window_left >= 0 ? max(start, kv_len - 1 - p.window_left) : start;
    const int g = l % 16, quad = l / 16;

    // Qᵀ as B operands: this lane's column is head g, k pairs 8·quad. Held in
    // registers, except at 512-wide heads, where they would push the round's
    // reads into scratch: there every wavefront rereads them from LDS.
    constexpr bool QLDS = D >= 512;
    mfma_ab2 qf[QLDS ? 1 : KP];
    __shared__ mfma_ab2 qs[QLDS ? KP : 1][kWave];
    {
        const bf16* qrow = p.q + static_cast<long long>(p.q_indptr[req]) * p.q_stride_n +
                           static_cast<long long>(kv_head * G + min(g, G - 1)) * p.q_stride_h;
        if constexpr (QLDS) {
            if (wave == 0) {
#pragma unroll
                for (int st = 0; st < KP; ++st) qs[st][l] = g < G ? load8(qrow + st * 32 + 8 * quad) : mfma_ab2{};
            }
            __syncthreads();
        } else {
#pragma unroll
            for (int st = 0; st < KP; ++st) {
                qf[st] = load8(qrow + st * 32 + 8 * quad);
                if (g >= G) qf[st] = mfma_ab2{};
            }
        }
    }
    auto q_at = [&](int st) -> mfma_ab2 {
        if constexpr (QLDS) {
            return qs[st][l];
        } else {
            return qf[st];
        }
    };

    float m = neg_inf(), lsum = 0.f;
    float acc[G][8];
#pragma unroll
    for (int h = 0; h < G; ++h)
#pragma unroll
        for (int c = 0; c < 8; ++c) acc[h][c] = 0.f;
    const int col = 8 * (l % LPK), parity = l / LPK;

    const i32* pages = p.page_indices + p.page_indptr[req];
    const int ps_ = p.page_size;
    for (int page0 = lo / ps_; page0 * ps_ < end; page0 += TABLE) {
    const int w0 = max(lo, page0 * ps_), wend = min(end, (page0 + TABLE) * ps_);
    __syncthreads();
    for (int i = threadIdx.x; i < TABLE && page0 + i <= (wend - 1) / ps_; i += blockDim.x) table[i] = pages[page0 + i];
    __syncthreads();
    auto at = [&](int t, int j) {
        return kv_at<HND>(static_cast<long long>(table[t / ps_ - page0]), t % ps_, kv_head, j, ps_, p.num_kv_heads, D);
    };
    for (int t0 = w0 + wave * SUB; t0 < wend; t0 += WAVES * SUB) {
        // The round's key and value reads, all issued before any is used
        // (past the chunk they reread its last key, which weighs zero).
        // The key reads in halves (KH of them at once): a 512-wide head's
        // whole key row beside its value reads would not fit the registers.
        constexpr int KH = QLDS ? KP / 2 : KP;
        mfma_cd s[KT];
        mfma_ab2 kr[KT][KH];
        auto read_keys = [&](int half) {
#pragma unroll
            for (int kt = 0; kt < KT; ++kt) {
                const bf16* row = p.k + at(min(t0 + kt * 16 + g, wend - 1), 0) + 8 * quad + half * KH * 32;
#pragma unroll
                for (int st = 0; st < KH; ++st) kr[kt][st] = load8(row + st * 32);
            }
        };
        auto score_keys = [&](int half) {
#pragma unroll
            for (int kt = 0; kt < KT; ++kt)
#pragma unroll
                for (int st = 0; st < KH; ++st) s[kt] = mfma16x2(kr[kt][st], q_at(half * KH + st), s[kt]);
        };
#pragma unroll
        for (int kt = 0; kt < KT; ++kt) s[kt] = mfma_zero();
        read_keys(0);
        // The value reads ride ahead of the scores, except at 512-wide heads,
        // which read them in halves after the weights (registers again).
        auto value_at = [&](int i) {
            return *reinterpret_cast<const Pack8<bf16>*>(p.v + at(min(t0 + i * KPI + parity, wend - 1), col));
        };
        Pack8<bf16> vr[QLDS ? 1 : VR];
        if constexpr (!QLDS) {
#pragma unroll
            for (int i = 0; i < VR; ++i) vr[i] = value_at(i);
        }
        score_keys(0);
        if constexpr (KH < KP) {
            read_keys(1);
            score_keys(1);
        }
        float tmax = neg_inf();
#pragma unroll
        for (int kt = 0; kt < KT; ++kt)
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int key = t0 + kt * 16 + 4 * quad + t;
                s[kt][t] = key < wend ? s[kt][t] * p.scale_log2 : neg_inf();
                tmax = fmaxf(tmax, s[kt][t]);
            }
        tmax = fmaxf(tmax, __shfl_xor(tmax, 16, kWave));
        tmax = fmaxf(tmax, __shfl_xor(tmax, 32, kWave));
        const float m_new = fmaxf(m, tmax);
        const float alpha = m_new == neg_inf() ? 1.f : exp2f(m - m_new);
        float sum = 0.f;
#pragma unroll
        for (int kt = 0; kt < KT; ++kt)
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const float e = m_new == neg_inf() ? 0.f : exp2f(s[kt][t] - m_new);
                sum += e;
                if (g < G) ps[wave][kt * 16 + 4 * quad + t][g] = e;
            }
        lsum = lsum * alpha + sum;
        m = m_new;
        if (quad == 0 && g < G) alph[wave][g] = alpha;
        // (the wavefront reads back only its own weights)
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");

#pragma unroll
        for (int h = 0; h < G; ++h) {
            const float a = alph[wave][h];
#pragma unroll
            for (int c = 0; c < 8; ++c) acc[h][c] *= a;
        }
        auto weigh = [&](int i, const Pack8<bf16>& v) {
            const int kl = i * KPI + parity;
            float pv[G];
#pragma unroll
            for (int h = 0; h < G; ++h) pv[h] = ps[wave][kl][h];
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const float vf = to_f32(v.v[c]);
#pragma unroll
                for (int h = 0; h < G; ++h) acc[h][c] += pv[h] * vf;
            }
        };
        if constexpr (QLDS) {
            constexpr int VH = VR / 2;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                Pack8<bf16> vh[VH];
#pragma unroll
                for (int i = 0; i < VH; ++i) vh[i] = value_at(half * VH + i);
#pragma unroll
                for (int i = 0; i < VH; ++i) weigh(half * VH + i, vh[i]);
            }
        } else {
#pragma unroll
            for (int i = 0; i < VR; ++i) weigh(i, vr[i]);
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    }
    }

    // The head's sum over its four quads, the columns' over the parities.
    lsum += __shfl_xor(lsum, 16, kWave);
    lsum += __shfl_xor(lsum, 32, kWave);
#pragma unroll
    for (int off = LPK; off < kWave; off <<= 1)
#pragma unroll
        for (int h = 0; h < G; ++h)
#pragma unroll
            for (int c = 0; c < 8; ++c) acc[h][c] += __shfl_xor(acc[h][c], off, kWave);
    if (quad == 0 && g < G) {
        part_m[wave][g] = m;
        part_l[wave][g] = lsum;
    }

    // A split writes its work item's partial row; an unsplit decode the
    // request's own query row of the plane.
    const long long row = static_cast<long long>(p.partition ? bx : p.q_indptr[req]) * p.num_qo_heads;
#pragma unroll
    for (int h0 = 0; h0 < G; h0 += HG) {
        if (h0 > 0) __syncthreads();
        if (l < LPK)
#pragma unroll
            for (int h = 0; h < HG; ++h)
#pragma unroll
                for (int c = 0; c < 8; ++c) part_o[wave][h][col + c] = acc[h0 + h][c];
        __syncthreads();
        for (int i = threadIdx.x; i < HG * D; i += blockDim.x) {
            const int hl = i / D, h = h0 + hl, j = i % D;
            float big = neg_inf();
#pragma unroll
            for (int w = 0; w < WAVES; ++w) big = fmaxf(big, part_m[w][h]);
            float sum = 0.f, out = 0.f;
            if (big != neg_inf()) {
#pragma unroll
                for (int w = 0; w < WAVES; ++w) {
                    const float f = exp2f(part_m[w][h] - big);
                    sum += part_l[w][h] * f;
                    out += part_o[w][hl][j] * f;
                }
            }
            const float value = sum > 0.f ? out / sum : 0.f;
            const long long at = (row + kv_head * G + h) * D + j;
            if (p.partition) static_cast<float*>(p.o)[at] = value;
            else static_cast<bf16*>(p.o)[at] = from_f32<bf16>(value);
            if (j == 0 && p.lse != nullptr) p.lse[row + kv_head * G + h] = sum > 0.f ? big + log2f(sum) : neg_inf();
        }
    }
}

// --------------------------------------------------------------- prefill --
//
// MFMA flash attention for the widest heads (D = 512). A block owns 128
// packed query rows (row r is token r / G, head kv_head·G + r % G) as eight
// wavefronts of 16, and walks its kv
// chunk BC keys at a time through LDS (K as [key][d], V transposed to
// [d][key]). Per wavefront and tile:
//
//   Sᵀ (BC × 16) = K · Qᵀ        A = K rows,  B = Qᵀ held in registers
//   Oᵀ (D × 16) += Vᵀ · Pᵀ       A = Vᵀ rows, B = Pᵀ, the Sᵀ tile's own lanes
//
// v_mfma_f32_16x16x16_bf16 holds A as (row l%16, k 4(l/16)+t), B as
// (col l%16, k 4(l/16)+t) and D as (row 4(l/16)+t, col l%16): the Sᵀ result
// is already Pᵀ's operand layout, and every lane's output column is one
// query row, so the softmax rescale is the lane's own.
//
// The LDS rows are padded so an operand read's 16 rows fall in distinct
// banks (K rows by 32 bytes, Vᵀ rows by 8), and Vᵀ is staged with a wave's
// lanes on 16 consecutive keys of four columns, so its 2-byte stores of a
// column land side by side. The next tile's K and V are read into registers
// while this one multiplies. A wavefront whose 16 rows are all past the
// request's queries (a decode's 64-head group fills half the tile) only
// stages.

constexpr int kPrefillTileQ = 128;

// The keys a tile reads from its first under a window of `wl` keys behind a
// row (kv_len without one): the window, then its rows' tokens, one more
// where a tile starts mid-token (sched_prefill's `window_reach`).
__device__ __forceinline__ int window_reach(int wl, int G) {
    return wl + (kPrefillTileQ + G - 1) / G + 1;
}

template <int D>
constexpr int prefill_bc() {
    return D >= 512 ? 16 : 32;
}

template <int D, bool HND, bool CAUSAL>
__global__ __launch_bounds__(512) void paged_prefill_wide(const Paged p) {
    constexpr int BC = prefill_bc<D>();
    constexpr int KS = D / 16;   // k steps over the head for Sᵀ, d tiles for Oᵀ
    constexpr int KT = BC / 16;  // key tiles per kv tile
    constexpr int KPAD = D + 16, VPAD = BC + 4;
    constexpr int THREADS = 512, RD = BC * D / 8 / THREADS;  // 16-byte reads a thread a tile
    static_assert(RD * 8 * THREADS == BC * D && THREADS % BC == 0, "the tile splits evenly");
    __shared__ alignas(16) bf16 ks[BC][KPAD];
    __shared__ alignas(16) bf16 vt[D][VPAD];

    const int bx = blockIdx.x;
    const int kv_head = blockIdx.y;
    if (p.block_valid != nullptr && p.block_valid[bx] == 0) return;
    const int G = p.num_qo_heads / p.num_kv_heads;
    const int req = p.request_indices[bx];
    const int qo_tile = p.qo_tile_indices[bx];
    const int kv_tile = p.kv_tile_indices[bx];
    const int q_first = p.q_indptr[req];
    const int qo_len = p.q_indptr[req + 1] - q_first;
    const int kv_len = kv_length(p, req);
    const int chunk = *p.kv_chunk_size;
    const int wl = p.window_left >= 0 ? p.window_left : kv_len;

    const int kv_start = max(0, kv_len + (qo_tile * kPrefillTileQ) / G - qo_len - wl);
    const int chunk_start = p.partition ? min(kv_tile * chunk + kv_start, kv_len) : kv_start;
    int chunk_end = p.partition ? min((kv_tile + 1) * chunk + kv_start, kv_len) : kv_len;
    if (CAUSAL) {
        // No key past the tile's last query is seen by any row of it.
        const int last_token = min(qo_len, ((qo_tile + 1) * kPrefillTileQ + G - 1) / G) - 1;
        chunk_end = min(chunk_end, kv_len - qo_len + last_token + 1);
    }

    const int wave = threadIdx.x / kWave;
    const int ln = lane();
    const int packed = qo_tile * kPrefillTileQ + wave * 16 + (ln % 16);
    const int token = packed / G;
    const int head = kv_head * G + packed % G;
    const bool row_live = token < qo_len;
    const int quad = ln / 16;
    // Whether any of the wavefront's rows is a query (its first row is).
    const bool wave_live = (qo_tile * kPrefillTileQ + wave * 16) / G < qo_len;

    // Qᵀ as B operands: column = this lane's query row, k = 4·quad + t.
    mfma_ab qf[KS];
    {
        const bf16* qrow = p.q + static_cast<long long>(q_first + token) * p.q_stride_n +
                           static_cast<long long>(head) * p.q_stride_h;
#pragma unroll
        for (int s = 0; s < KS; ++s) qf[s] = row_live ? load4(qrow + s * 16 + 4 * quad) : mfma_ab{0, 0, 0, 0};
    }

    mfma_cd o[KS];
#pragma unroll
    for (int s = 0; s < KS; ++s) o[s] = mfma_cd{0.f, 0.f, 0.f, 0.f};
    float m = neg_inf(), l = 0.f;
    // The last key this row may see, and the first (the window).
    const int key_hi = CAUSAL ? kv_len - qo_len + token : kv_len - 1;
    const int key_lo = kv_len - qo_len + token - wl;

    // A thread's reads of a tile: K piece c = tid + r·THREADS as (key c /
    // (D/8), columns 8·(c % (D/8))), a wavefront one key's row; V piece c as
    // (key c % BC, columns 8·(c / BC)), a wavefront 16 keys of four columns.
    // Every read is issued (a key past the chunk reads the chunk's last,
    // and lands as zeros at the store), so no read waits on another.
    Pack8<bf16> kr[RD], vr[RD];
    auto fetch = [&](int kv0) {
#pragma unroll
        for (int r = 0; r < RD; ++r) {
            const int c = threadIdx.x + r * THREADS;
            const int kkey = kv0 + c / (D / 8), kj = (c % (D / 8)) * 8;
            const int vkey = kv0 + c % BC, vj = (c / BC) * 8;
            kr[r] = *reinterpret_cast<const Pack8<bf16>*>(
                p.k + kv_token<HND, D>(p, req, min(kkey, chunk_end - 1), kv_head, kj));
            vr[r] = *reinterpret_cast<const Pack8<bf16>*>(
                p.v + kv_token<HND, D>(p, req, min(vkey, chunk_end - 1), kv_head, vj));
        }
    };
    auto stage = [&](int kv0) {
#pragma unroll
        for (int r = 0; r < RD; ++r) {
            const int c = threadIdx.x + r * THREADS;
            const Pack8<bf16> k8 = kv0 + c / (D / 8) < chunk_end ? kr[r] : Pack8<bf16>{};
            const Pack8<bf16> v8 = kv0 + c % BC < chunk_end ? vr[r] : Pack8<bf16>{};
            *reinterpret_cast<Pack8<bf16>*>(&ks[c / (D / 8)][(c % (D / 8)) * 8]) = k8;
            const int vkey = c % BC, vj = (c / BC) * 8;
#pragma unroll
            for (int e = 0; e < 8; ++e) vt[vj + e][vkey] = v8.v[e];
        }
    };
    if (chunk_start < chunk_end) fetch(chunk_start);

    for (int kv0 = chunk_start; kv0 < chunk_end; kv0 += BC) {
        stage(kv0);
        __syncthreads();
        if (kv0 + BC < chunk_end) fetch(kv0 + BC);
        if (wave_live) {
            // Sᵀ: lane holds keys 4·quad + t of each 16-key tile, for its row;
            // two chains over the head, so a product waits on the one before
            // the last.
            mfma_cd s[KT];
#pragma unroll
            for (int kt = 0; kt < KT; ++kt) {
                mfma_cd s2 = mfma_cd{0.f, 0.f, 0.f, 0.f};
                s[kt] = mfma_cd{0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int st = 0; st < KS; st += 2) {
                    s[kt] = mfma16(load4(&ks[kt * 16 + ln % 16][st * 16 + 4 * quad]), qf[st], s[kt]);
                    s2 = mfma16(load4(&ks[kt * 16 + ln % 16][(st + 1) * 16 + 4 * quad]), qf[st + 1], s2);
                }
                s[kt] += s2;
            }
            float tile_max = neg_inf();
#pragma unroll
            for (int kt = 0; kt < KT; ++kt)
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const int key = kv0 + kt * 16 + 4 * quad + t;
                    const bool seen = row_live && key < chunk_end && key <= key_hi && key >= key_lo;
                    s[kt][t] = seen ? s[kt][t] * p.scale_log2 : neg_inf();
                    tile_max = fmaxf(tile_max, s[kt][t]);
                }
            tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 16, kWave));
            tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 32, kWave));
            const float m_new = fmaxf(m, tile_max);
            const float alpha = m_new == neg_inf() ? 1.f : exp2f(m - m_new);
            float sum = 0.f;
            mfma_ab pt[KT];
#pragma unroll
            for (int kt = 0; kt < KT; ++kt) {
                float e[4];
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    e[t] = m_new == neg_inf() ? 0.f : exp2f(s[kt][t] - m_new);
                    sum += e[t];
                }
                pt[kt] = pack4(e[0], e[1], e[2], e[3]);
            }
            l = l * alpha + sum;
            m = m_new;
#pragma unroll
            for (int dt = 0; dt < KS; ++dt) {
                o[dt] *= alpha;
#pragma unroll
                for (int kt = 0; kt < KT; ++kt) o[dt] = mfma16(load4(&vt[dt * 16 + ln % 16][kt * 16 + 4 * quad]), pt[kt], o[dt]);
            }
        }
        __syncthreads();
    }

    // Each lane summed only its own keys of the row; the row's four lanes
    // (ln, ln ± 16, ± 32) share m, so their sums add.
    l += __shfl_xor(l, 16, kWave);
    l += __shfl_xor(l, 32, kWave);
    if (!row_live) return;

    const int num_chunks = (min(kv_len, window_reach(wl, G)) + chunk - 1) / chunk;
    const long long row = p.partition
        ? static_cast<long long>(p.o_indptr[req]) + static_cast<long long>(token) * num_chunks + kv_tile
        : static_cast<long long>(p.o_indptr[req]) + token;
    const long long base = (row * p.num_qo_heads + head) * D;
    const float inv = l > 0.f ? 1.f / l : 0.f;
#pragma unroll
    for (int dt = 0; dt < KS; ++dt) {
        const int j = dt * 16 + 4 * quad;
        if (p.partition) {
            float* out = static_cast<float*>(p.o) + base + j;
#pragma unroll
            for (int t = 0; t < 4; ++t) out[t] = o[dt][t] * inv;
        } else {
            *reinterpret_cast<mfma_ab*>(static_cast<bf16*>(p.o) + base + j) =
                pack4(o[dt][0] * inv, o[dt][1] * inv, o[dt][2] * inv, o[dt][3] * inv);
        }
    }
    if (quad == 0 && p.lse != nullptr) p.lse[row * p.num_qo_heads + head] = l > 0.f ? m + log2f(l) : neg_inf();
}

// MFMA flash attention for heads up to 256 wide, the same products as the
// wide kernel (Sᵀ = K·Qᵀ, Oᵀ += Vᵀ·Pᵀ) with the staging built for the
// matrix core's appetite:
//
//   * the request's page table sits in LDS, so a key's address needs no
//     global read before its own;
//   * the next kv tile is read into registers while this one computes, and
//     only LDS is fenced between them (`lds_barrier`), so the reads stay in
//     flight across the barrier;
//   * K rows and Vᵀ rows are padded by 8 bytes, which puts the 16 rows an
//     MFMA operand read touches in distinct banks; Vᵀ is written two keys
//     at a time, 4 bytes per column.
//
// A block owns 128 packed query rows as eight wavefronts of 16 and walks
// its kv chunk 32 keys at a time.
template <int D, bool HND, bool CAUSAL>
__global__ __launch_bounds__(512) void paged_prefill(const Paged p) {
    constexpr int BC = 32, KT = BC / 16, KP = D / 32, DT = D / 16;
    constexpr int KPAD = D + 4, VPAD = BC + 4;
    constexpr int THREADS = 512, TABLE = 512;
    static_assert(D == 128 || D == 256, "the narrow prefill");
    // Per thread a round: D/128 16-byte K reads (key t/16, columns from
    // (t%16)·D/16) and, for the first 4·D threads, two V reads (keys 2(t%16)
    // and +1, columns 8·(t/16)).
    constexpr int KR = BC * D / 8 / THREADS;
    __shared__ alignas(16) bf16 ks[BC * KPAD];
    __shared__ alignas(16) bf16 vt[D * VPAD];
    __shared__ i32 table[TABLE];

    // Causal tiles grow with their position: the last of a request walks the
    // most keys, so the grid starts there and the short ones fill the tail.
    const int bx = CAUSAL ? gridDim.x - 1 - blockIdx.x : blockIdx.x;
    const int kv_head = blockIdx.y;
    if (p.block_valid != nullptr && p.block_valid[bx] == 0) return;
    const int G = p.num_qo_heads / p.num_kv_heads;
    const int req = p.request_indices[bx];
    const int qo_tile = p.qo_tile_indices[bx];
    const int kv_tile = p.kv_tile_indices[bx];
    const int q_first = p.q_indptr[req];
    const int qo_len = p.q_indptr[req + 1] - q_first;
    const int kv_len = kv_length(p, req);
    const int chunk = *p.kv_chunk_size;
    const int wl = p.window_left >= 0 ? p.window_left : kv_len;

    const int kv_start = max(0, kv_len + (qo_tile * kPrefillTileQ) / G - qo_len - wl);
    const int chunk_start = p.partition ? min(kv_tile * chunk + kv_start, kv_len) : kv_start;
    int chunk_end = p.partition ? min((kv_tile + 1) * chunk + kv_start, kv_len) : kv_len;
    if (CAUSAL) {
        const int last_token = min(qo_len, ((qo_tile + 1) * kPrefillTileQ + G - 1) / G) - 1;
        chunk_end = min(chunk_end, kv_len - qo_len + last_token + 1);
    }

    const int tid = threadIdx.x, wave = tid / kWave, ln = lane(), quad = ln / 16;
    const int packed = qo_tile * kPrefillTileQ + wave * 16 + (ln % 16);
    const int token = packed / G;
    const int head = kv_head * G + packed % G;
    const bool row_live = token < qo_len;

    // Qᵀ as B operands, k paired 8·quad.
    mfma_ab2 qf[KP];
    {
        const bf16* qrow = p.q + static_cast<long long>(q_first + min(token, qo_len - 1)) * p.q_stride_n +
                           static_cast<long long>(head) * p.q_stride_h + 8 * quad;
#pragma unroll
        for (int s = 0; s < KP; ++s) qf[s] = load8(qrow + s * 32);
    }

    mfma_cd o[DT];
#pragma unroll
    for (int s = 0; s < DT; ++s) o[s] = mfma_zero();
    float m = neg_inf(), l = 0.f;
    const int key_hi = CAUSAL ? kv_len - qo_len + token : kv_len - 1;
    const int key_lo = kv_len - qo_len + token - wl;

    const i32* pages = p.page_indices + p.page_indptr[req];
    const int ps = p.page_size;
    // This thread's reads of a round.
    const int k_key = tid / (D / 8 / KR), k_col = (tid % (D / 8 / KR)) * 8 * KR;
    const int v_pair = tid % (BC / 2), v_col = (tid / (BC / 2)) * 8;
    const bool v_live = v_col < D;
    Pack8<bf16> kreg[KR], vreg[2];
    int page0 = 0, wend = 0;
    auto fetch = [&](int kv0) {
        auto at = [&](int t, int j) {
            t = min(t, wend - 1);
            return kv_at<HND>(static_cast<long long>(table[t / ps - page0]), t % ps, kv_head, j, ps, p.num_kv_heads, D);
        };
#pragma unroll
        for (int r = 0; r < KR; ++r)
            kreg[r] = *reinterpret_cast<const Pack8<bf16>*>(p.k + at(kv0 + k_key, k_col + 8 * r));
        if (v_live) {
#pragma unroll
            for (int e = 0; e < 2; ++e) vreg[e] = *reinterpret_cast<const Pack8<bf16>*>(p.v + at(kv0 + 2 * v_pair + e, v_col));
        }
    };
    auto stage = [&](int kv0) {
        // Keys past the chunk land as zeros, so no garbage meets a weight.
#pragma unroll
        for (int r = 0; r < KR; ++r) {
            Pack8<bf16> kv = kreg[r];
            if (kv0 + k_key >= wend) kv = Pack8<bf16>{};
            *reinterpret_cast<Pack8<bf16>*>(&ks[k_key * KPAD + k_col + 8 * r]) = kv;
        }
        if (v_live) {
#pragma unroll
            for (int e = 0; e < 2; ++e)
                if (kv0 + 2 * v_pair + e >= wend) vreg[e] = Pack8<bf16>{};
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const bf16 pair[2] = {vreg[0].v[c], vreg[1].v[c]};
                *reinterpret_cast<u32*>(&vt[(v_col + c) * VPAD + 2 * v_pair]) = *reinterpret_cast<const u32*>(pair);
            }
        }
    };

    for (int pw = chunk_start / ps; pw * ps < chunk_end; pw += TABLE) {
        page0 = pw;
        const int w0 = max(chunk_start, pw * ps);
        wend = min(chunk_end, (pw + TABLE) * ps);
        lds_barrier();
        for (int i = tid; i < TABLE && pw + i <= (wend - 1) / ps; i += THREADS) table[i] = pages[pw + i];
        lds_barrier();
        if (w0 < wend) fetch(w0);
        for (int kv0 = w0; kv0 < wend; kv0 += BC) {
            lds_barrier();
            stage(kv0);
            lds_barrier();
            if (kv0 + BC < wend) fetch(kv0 + BC);

            mfma_cd s[KT];
#pragma unroll
            for (int kt = 0; kt < KT; ++kt) {
                s[kt] = mfma_zero();
#pragma unroll
                for (int st = 0; st < KP; ++st)
                    s[kt] = mfma16x2(load8(&ks[(kt * 16 + ln % 16) * KPAD + st * 32 + 8 * quad]), qf[st], s[kt]);
            }
            float tile_max = neg_inf();
#pragma unroll
            for (int kt = 0; kt < KT; ++kt)
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const int key = kv0 + kt * 16 + 4 * quad + t;
                    const bool seen = row_live && key < wend && key <= key_hi && key >= key_lo;
                    s[kt][t] = seen ? s[kt][t] * p.scale_log2 : neg_inf();
                    tile_max = fmaxf(tile_max, s[kt][t]);
                }
            tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 16, kWave));
            tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 32, kWave));
            const float m_new = fmaxf(m, tile_max);
            const float alpha = m_new == neg_inf() ? 1.f : exp2f(m - m_new);
            float sum = 0.f;
            mfma_ab pt[KT];
#pragma unroll
            for (int kt = 0; kt < KT; ++kt) {
                float e[4];
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    e[t] = m_new == neg_inf() ? 0.f : exp2f(s[kt][t] - m_new);
                    sum += e[t];
                }
                pt[kt] = pack4(e[0], e[1], e[2], e[3]);
            }
            l = l * alpha + sum;
            m = m_new;
#pragma unroll
            for (int dt = 0; dt < DT; ++dt) {
                o[dt] *= alpha;
#pragma unroll
                for (int kt = 0; kt < KT; ++kt)
                    o[dt] = mfma16(load4(&vt[(dt * 16 + ln % 16) * VPAD + kt * 16 + 4 * quad]), pt[kt], o[dt]);
            }
        }
    }

    l += __shfl_xor(l, 16, kWave);
    l += __shfl_xor(l, 32, kWave);
    if (!row_live) return;

    const int num_chunks = (min(kv_len, window_reach(wl, G)) + chunk - 1) / chunk;
    const long long row = p.partition
        ? static_cast<long long>(p.o_indptr[req]) + static_cast<long long>(token) * num_chunks + kv_tile
        : static_cast<long long>(p.o_indptr[req]) + token;
    const long long base = (row * p.num_qo_heads + head) * D;
    const float inv = l > 0.f ? 1.f / l : 0.f;
#pragma unroll
    for (int dt = 0; dt < DT; ++dt) {
        const int j = dt * 16 + 4 * quad;
        if (p.partition) {
            float* out = static_cast<float*>(p.o) + base + j;
            *reinterpret_cast<float4*>(out) = float4{o[dt][0] * inv, o[dt][1] * inv, o[dt][2] * inv, o[dt][3] * inv};
        } else {
            *reinterpret_cast<mfma_ab*>(static_cast<bf16*>(p.o) + base + j) =
                pack4(o[dt][0] * inv, o[dt][1] * inv, o[dt][2] * inv, o[dt][3] * inv);
        }
    }
    if (quad == 0 && p.lse != nullptr) p.lse[row * p.num_qo_heads + head] = l > 0.f ? m + log2f(l) : neg_inf();
}

// ----------------------------------------------------------------- merge --

// Folds partials: row `pos`'s parts are rows indptr[pos]..indptr[pos+1] of
// `parts` (f32, [part][heads][D]) with log2-domain lse `part_lse`. One block
// of four wavefronts per (row, head): the wavefronts split the parts, each
// lane owning D/64 columns, and fold through LDS. The window, when staged,
// caps the rows and offsets them.
template <int D>
__global__ __launch_bounds__(256) void merge_partials(const float* __restrict__ parts, const float* __restrict__ part_lse,
                               const i32* __restrict__ indptr, bf16* __restrict__ o,
                               float* __restrict__ lse, int max_rows, const u32* __restrict__ rows_ptr,
                               int heads, const u32* __restrict__ win) {
    constexpr int WAVES = 4, C = (D + kWave - 1) / kWave;
    __shared__ float red[WAVES];
    __shared__ float acc_s[WAVES][D];
    const int pos = blockIdx.x / heads;
    const int head = blockIdx.x % heads;
    const int staged = rows_ptr != nullptr ? static_cast<int>(*rows_ptr) : max_rows;
    const int rows = win != nullptr ? min(staged, static_cast<int>(win[0])) : staged;
    if (pos >= rows) return;
    const long long out_row = (win != nullptr ? static_cast<long long>(win[1]) : 0) + pos;
    const int first = indptr[pos], last = indptr[pos + 1];
    const int wave = threadIdx.x / kWave, l = lane();

    // The parts' max lse, then their weights' sum.
    float big = neg_inf();
    for (int i = first + threadIdx.x; i < last; i += blockDim.x) big = fmaxf(big, part_lse[static_cast<long long>(i) * heads + head]);
    big = wave_max(big);
    if (l == 0) red[wave] = big;
    __syncthreads();
    big = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
    __syncthreads();
    float sum = 0.f;
    if (big != neg_inf())
        for (int i = first + threadIdx.x; i < last; i += blockDim.x) sum += exp2f(part_lse[static_cast<long long>(i) * heads + head] - big);
    sum = wave_sum(sum);
    if (l == 0) red[wave] = sum;
    __syncthreads();
    sum = red[0] + red[1] + red[2] + red[3];

    float acc[C];
#pragma unroll
    for (int c = 0; c < C; ++c) acc[c] = 0.f;
    if (sum > 0.f)
        for (int i = first + wave; i < last; i += WAVES) {
            const long long at = static_cast<long long>(i) * heads + head;
            const float w = exp2f(part_lse[at] - big);
#pragma unroll
            for (int c = 0; c < C; ++c) {
                const int j = c * kWave + l;
                if (j < D) acc[c] += parts[at * D + j] * w;
            }
        }
#pragma unroll
    for (int c = 0; c < C; ++c) {
        const int j = c * kWave + l;
        if (j < D) acc_s[wave][j] = acc[c];
    }
    __syncthreads();
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        const float total = acc_s[0][j] + acc_s[1][j] + acc_s[2][j] + acc_s[3][j];
        o[(out_row * heads + head) * D + j] = from_f32<bf16>(sum > 0.f ? total / sum : 0.f);
    }
    if (threadIdx.x == 0 && lse != nullptr) lse[out_row * heads + head] = sum > 0.f ? big + log2f(sum) : neg_inf();
}

}  // namespace pie::attn
