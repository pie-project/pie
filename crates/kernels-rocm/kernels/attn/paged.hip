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

    const long long row = static_cast<long long>(bx) * p.num_qo_heads;
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

// --------------------------------------------------------------- prefill --
//
// MFMA flash attention. A block owns 64 packed query rows (row r is token
// r / G, head kv_head·G + r % G) as four wavefronts of 16, and walks its kv
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

typedef short mfma_ab __attribute__((ext_vector_type(4)));
typedef float mfma_cd __attribute__((ext_vector_type(4)));

__device__ __forceinline__ mfma_cd mfma16(mfma_ab a, mfma_ab b, mfma_cd c) {
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
}

__device__ __forceinline__ mfma_ab load4(const bf16* p) {
    return *reinterpret_cast<const mfma_ab*>(p);
}

__device__ __forceinline__ mfma_ab pack4(float a, float b, float c, float d) {
    const bf16 v[4] = {from_f32<bf16>(a), from_f32<bf16>(b), from_f32<bf16>(c), from_f32<bf16>(d)};
    return *reinterpret_cast<const mfma_ab*>(v);
}

constexpr int kPrefillTileQ = 64;

template <int D>
constexpr int prefill_bc() {
    return D >= 512 ? 16 : 32;
}

template <int D, bool HND, bool CAUSAL>
__global__ __launch_bounds__(256) void paged_prefill(const Paged p) {
    constexpr int BC = prefill_bc<D>();
    constexpr int KS = D / 16;   // k steps over the head for Sᵀ, d tiles for Oᵀ
    constexpr int KT = BC / 16;  // key tiles per kv tile
    __shared__ bf16 ks[BC][D];
    __shared__ bf16 vt[D][BC];

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

    for (int kv0 = chunk_start; kv0 < chunk_end; kv0 += BC) {
        // Stage K and Vᵀ: 16 bytes (8 elements of one key) per thread-step.
        for (int c = threadIdx.x; c < BC * D / 8; c += blockDim.x) {
            const int key = c / (D / 8), j = (c % (D / 8)) * 8;
            Pack8<bf16> kv8{}, vv8{};
            if (kv0 + key < chunk_end) {
                const long long at = kv_token<HND, D>(p, req, kv0 + key, kv_head, j);
                kv8 = *reinterpret_cast<const Pack8<bf16>*>(p.k + at);
                vv8 = *reinterpret_cast<const Pack8<bf16>*>(p.v + at);
            }
            *reinterpret_cast<Pack8<bf16>*>(&ks[key][j]) = kv8;
#pragma unroll
            for (int e = 0; e < 8; ++e) vt[j + e][key] = vv8.v[e];
        }
        __syncthreads();

        // Sᵀ: lane holds keys 4·quad + t of each 16-key tile, for its row.
        mfma_cd s[KT];
#pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
            s[kt] = mfma_cd{0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int st = 0; st < KS; ++st) s[kt] = mfma16(load4(&ks[kt * 16 + ln % 16][st * 16 + 4 * quad]), qf[st], s[kt]);
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
        __syncthreads();
    }

    // Each lane summed only its own keys of the row; the row's four lanes
    // (ln, ln ± 16, ± 32) share m, so their sums add.
    l += __shfl_xor(l, 16, kWave);
    l += __shfl_xor(l, 32, kWave);
    if (!row_live) return;

    const int num_chunks = (min(kv_len, wl + kPrefillTileQ) + chunk - 1) / chunk;
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

// ----------------------------------------------------------------- merge --

// Folds partials: row `pos`'s parts are rows indptr[pos]..indptr[pos+1] of
// `parts` (f32, [part][heads][D]) with log2-domain lse `part_lse`. One block
// per (row, head); the window, when staged, caps the rows and offsets them.
template <int D>
__global__ void merge_partials(const float* __restrict__ parts, const float* __restrict__ part_lse,
                               const i32* __restrict__ indptr, bf16* __restrict__ o,
                               float* __restrict__ lse, int max_rows, const u32* __restrict__ rows_ptr,
                               int heads, const u32* __restrict__ win) {
    const int pos = blockIdx.x / heads;
    const int head = blockIdx.x % heads;
    const int staged = rows_ptr != nullptr ? static_cast<int>(*rows_ptr) : max_rows;
    const int rows = win != nullptr ? min(staged, static_cast<int>(win[0])) : staged;
    if (pos >= rows) return;
    const long long out_row = (win != nullptr ? static_cast<long long>(win[1]) : 0) + pos;
    const int first = indptr[pos], last = indptr[pos + 1];
    float big = neg_inf();
    for (int i = first; i < last; ++i) big = fmaxf(big, part_lse[static_cast<long long>(i) * heads + head]);
    float sum = 0.f;
    if (big != neg_inf())
        for (int i = first; i < last; ++i) sum += exp2f(part_lse[static_cast<long long>(i) * heads + head] - big);
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float acc = 0.f;
        if (sum > 0.f)
            for (int i = first; i < last; ++i) {
                const long long at = static_cast<long long>(i) * heads + head;
                acc += parts[at * D + j] * exp2f(part_lse[at] - big);
            }
        o[(out_row * heads + head) * D + j] = from_f32<bf16>(sum > 0.f ? acc / sum : 0.f);
    }
    if (threadIdx.x == 0 && lse != nullptr) lse[out_row * heads + head] = sum > 0.f ? big + log2f(sum) : neg_inf();
}

}  // namespace pie::attn
