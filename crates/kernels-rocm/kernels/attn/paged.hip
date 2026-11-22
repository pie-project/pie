#pragma once

// Attention over the paged KV cache, decode and prefill, over the work list
// the host scheduler (attn/sched_*.rs) staged: a work item is a (request,
// query tile, kv chunk); a split schedule writes f32 partials and their lse
// for `merge_partials`. Scores are in the log2 domain (s = q·k·sm_scale·
// log2 e), the lse convention every reader of the plane takes. `FP8` reads
// an e4m3 pool (attn/kv.hip's `Kv`): the host folds its unit into
// `scale_log2`, the kernels into their output rows.

#include "prim/device.hip"
#include "prim/mfma.hip"
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
    // A learned relative-position bias (Inkling), null for none: the score
    // of query row r (position pos, head h) against key j gains
    // rel_bias[(r·H + h)·rel_extent + pos − j] where pos − j < rel_extent,
    // and grows by 1 + rel_log_alpha·ln((pos + 1) / rel_log_floor) past the
    // floor, before the softmax.
    const float* rel_bias;
    i32 rel_extent;
    i32 rel_log_floor;
    float rel_log_alpha;
    i32 rel_pad;
    // Elements from a page's rows to the next page's, keys and values alike
    // (`page_size * num_kv_heads * D` packed).
    i64 stride_page;
};

// The relative bias of one query row and head: `sc` scales a raw dot into
// the log2 domain, `bs` the bias, `row` its bias row (null: none).
struct RelRow {
    const float* row;
    float sc, bs;
    int pos, extent;

    __device__ __forceinline__ float score(float dot, int key) const {
        float x = dot * sc;
        if (row != nullptr) {
            const int back = pos - key;
            if (back >= 0 && back < extent) x += row[back] * bs;
        }
        return x;
    }
};

__device__ __forceinline__ RelRow rel_row(const Paged& p, long long q_row, int head, int pos) {
    if (p.rel_bias == nullptr) return {nullptr, p.scale_log2, 0.f, pos, 0};
    const float grow = p.rel_log_alpha != 0.f && p.rel_log_floor != 0 && pos + 1 > p.rel_log_floor
                           ? 1.f + p.rel_log_alpha * logf(static_cast<float>(pos + 1) / static_cast<float>(p.rel_log_floor))
                           : 1.f;
    return {p.rel_bias + (q_row * p.num_qo_heads + head) * static_cast<long long>(p.rel_extent),
            p.scale_log2 * grow, 1.4426950408889634f * grow, pos, p.rel_extent};
}

__device__ __forceinline__ int kv_length(const Paged& p, int req) {
    const int pages = p.page_indptr[req + 1] - p.page_indptr[req];
    return pages <= 0 ? 0 : (pages - 1) * p.page_size + p.last_page_len[req];
}

// The element offset of token `t`, head `h`, element `j` of request `req`.
template <bool HND, int D>
__device__ __forceinline__ long long kv_token(const Paged& p, int req, int t, int h, int j) {
    const long long page = p.page_indices[p.page_indptr[req] + t / p.page_size];
    return kv_at<HND>(page, t % p.page_size, h, j, p.page_size, p.num_kv_heads, D, p.stride_page);
}

// ---------------------------------------------------------------- decode --

// A fire's per-lane masks as a decode reads them (`engine-cuda`'s
// `mask::stage`): request r's word origin[r] (its boundary; bit 31, the wide
// bit, says nothing to a decode row, its lane's last) and its plane of bits
// from byte indptr[r]. Its single row reads key t where t < boundary, else
// bit t - boundary of its plane; a boundary at or past its keys (the word
// stating none, u32::MAX >> 1) reads no bit.
struct DecodeMask {
    const u8* bits;
    const i64* indptr;
    const u32* origin;
};

// One lane's mask, read once a block: its boundary, and its plane only when
// some key lies past it.
struct LaneMask {
    int boundary;
    const u8* plane;

    __device__ __forceinline__ bool reads(int t) const {
        return t < boundary || ((plane[(t - boundary) >> 3] >> ((t - boundary) & 7)) & 1u);
    }
};

template <bool MASK>
__device__ __forceinline__ LaneMask lane_mask(const DecodeMask& m, int req, int kv_len) {
    if constexpr (!MASK) {
        return LaneMask{0x7fffffff, nullptr};
    } else {
        const int boundary = static_cast<int>(m.origin[req] & 0x7fffffffu);
        return LaneMask{boundary, boundary < kv_len ? m.bits + m.indptr[req] : nullptr};
    }
}

// One block per (work item, kv head), WAVES wavefronts splitting the chunk's
// tokens. Lane i of a wavefront scores token t0 + i for every query head of
// the GQA group, the wavefront keeps an online softmax per head, and each
// lane accumulates its D/kWave columns of the output.
template <int D, int G, int WAVES, bool HND, bool FP8, bool MASK>
__global__ __launch_bounds__(kWave* WAVES) void paged_decode(const Paged p, const DecodeMask dm) {
    using KV = Kv<FP8>;
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
    const LaneMask mask = lane_mask<MASK>(dm, req, kv_len);

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
        const bool live = t < end && (!MASK || mask.reads(t));
        float s[G];
#pragma unroll
        for (int g = 0; g < G; ++g) s[g] = 0.f;
        if (live) {
            const long long kp = kv_token<HND, D>(p, req, t, kv_head, 0);
            for (int j8 = 0; j8 < D / 8; ++j8) {
                const Pack8<bf16> kv8 = KV::wide(KV::raw(p.k, kp + 8 * j8));
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
            // (a wavefront whose keys a lane's mask hides all of keeps none)
            const float alpha = m_new == neg_inf() ? 1.f : exp2f(m[g] - m_new);
            pr[g] = live ? exp2f(s[g] - m_new) : 0.f;
            l[g] = l[g] * alpha + wave_sum(pr[g]);
            m[g] = m_new;
#pragma unroll
            for (int c = 0; c < kCols; ++c) acc[g][c] *= alpha;
        }
        const int tiles = min(kWave, end - t0);
        for (int i = 0; i < tiles; ++i) {
            const long long vp = kv_token<HND, D>(p, req, t0 + i, kv_head, ln * kCols);
            float vf[kCols];
#pragma unroll
            for (int c = 0; c < kCols; ++c) vf[c] = KV::one(p.v, vp + c);
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
        const float value = sum > 0.f ? out / sum * KV::unit : 0.f;
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

// The decode on the matrix core: per wavefront SUB keys a round, Sᵀ = K·Qᵀ
// with K straight from the pages (lane l holds keys 4(l/16)+t for head
// l%16). The weights go through LDS to the value side, where a lane
// accumulates 8 columns of P·V from 16-byte value reads; the wavefronts
// fold through LDS at the end; QL keeps Qᵀ in LDS for the registers.

// With PVM the round's P·V runs on the matrix core too: the wavefront lays
// its 16 value rows in LDS and takes Oᵀ (16 columns × 16 heads) += Vᵀ · Pᵀ a
// column tile at a time, Pᵀ straight from the score registers (lane l holds
// keys 4(l/16)+t of head l%16, the B operand's layout), so a lane keeps D/16
// accumulator quads for its head instead of G × 8 floats, and a round costs
// D/16 MFMAs instead of G·D/8 FMAs a lane. The weights round to bf16 for the
// product, as a prefill's do.
template <int D, int G, bool HND, int WAVES, bool QL, bool PVM, bool FP8, bool MASK>
__global__ __launch_bounds__(64 * WAVES) void paged_decode_mfma(const Paged p, const DecodeMask dm) {
    using KV = Kv<FP8>;
    constexpr int SUB = 16, KT = SUB / 16, KP = D / 32;
    // Elements a value read takes: sixteen e4m3 codes (16 bytes) where the
    // values stage through LDS, else eight (a lane's accumulator columns).
    constexpr int RE = PVM && FP8 ? 16 : 8;
    constexpr int LPK = D / RE;          // lanes a key's value row spans
    constexpr int KPI = kWave / LPK;     // keys one value read covers
    constexpr int VR = SUB / KPI;        // value reads a round
    static_assert(G <= 16 && D % 32 == 0 && kWave % LPK == 0, "a group within one column tile");
    // Under PVM a round's value rows stage in LDS HV column halves at a
    // time: two past 256-wide heads, so four wavefronts' rows fit at 512.
    constexpr int HV = PVM && D > 256 ? 2 : 1, DH = D / HV;
    static_assert(!PVM || WAVES * SUB * (DH + 8) * 2 <= 40 * 1024,
                  "the staged value rows of a round within LDS");
    // A 512-wide head's key row, value reads and G·8 vector accumulators
    // spill past the registers into scratch: there the values are weighed on
    // the matrix core, whose accumulator tiles the AGPRs hold.
    static_assert(D < 512 || PVM, "512-wide heads weigh their values on the matrix core");
    constexpr int VLD = DH + 8;          // a staged value row's pitch
    __shared__ float ps[PVM ? 1 : WAVES][SUB][G];
    __shared__ float alph[WAVES][G];
    __shared__ float part_m[WAVES][G], part_l[WAVES][G];
    // The wavefronts' outputs fold through LDS HG heads at a time: all G at
    // once where they fit beside the rest. Under PVM the
    // fold reuses the rounds' value rows' LDS.
    constexpr int HG = decode_fold_heads(G, D, WAVES);
    constexpr int PO = WAVES * HG * D;
    constexpr int VB = PVM ? WAVES * SUB * VLD / 2 : 0;  // in floats
    __shared__ alignas(16) float po_lds[PO > VB ? PO : VB];
    auto part_o = [&](int w, int h, int j) -> float& { return po_lds[(w * HG + h) * D + j]; };
    bf16* vbuf = reinterpret_cast<bf16*>(po_lds);
    // The chunk's page table, a window of TABLE pages at a time: a key's
    // address then needs no read of global memory before its own.
    constexpr int TABLE = 512;
    __shared__ i32 table[TABLE];

    const int bx = blockIdx.x;
    const int kv_head = blockIdx.y;
    // (the work item read beside its mask: the staged lists span the padded grid)
    const int req = p.request_indices[bx];
    if (p.block_valid != nullptr && p.block_valid[bx] == 0) return;
    const int kv_len = kv_length(p, req);
    const int chunk = *p.kv_chunk_size;
    const int start = p.partition ? p.kv_tile_indices[bx] * chunk : 0;
    const int end = p.partition ? min(start + chunk, kv_len) : kv_len;
    const int wave = threadIdx.x / kWave, l = lane();
    const int lo = p.window_left >= 0 ? max(start, kv_len - 1 - p.window_left) : start;
    const int g = l % 16, quad = l / 16;
    const LaneMask mask = lane_mask<MASK>(dm, req, kv_len);

    // Qᵀ as B operands: this lane's column is head g, k pairs 8·quad. Held in
    // registers, except under QL: there every wavefront rereads them from
    // LDS.
    constexpr bool QLDS = QL;
    mfma_ab2 qf[QLDS ? 1 : KP];
    // (only the G live columns: a lane past them reads head G - 1's, whose
    // score columns no head's softmax reads)
    __shared__ mfma_ab2 qs[QLDS ? KP : 1][4][G];
    {
        const bf16* qrow = p.q + static_cast<long long>(p.q_indptr[req]) * p.q_stride_n +
                           static_cast<long long>(kv_head * G + min(g, G - 1)) * p.q_stride_h;
        if constexpr (QLDS) {
            if (wave == 0 && g < G) {
#pragma unroll
                for (int st = 0; st < KP; ++st) qs[st][quad][g] = load8(qrow + KV::col(st, quad));
            }
            __syncthreads();
        } else {
#pragma unroll
            for (int st = 0; st < KP; ++st) {
                qf[st] = load8(qrow + KV::col(st, quad));
                if (g >= G) qf[st] = mfma_ab2{};
            }
        }
    }
    auto q_at = [&](int st) -> mfma_ab2 {
        if constexpr (QLDS) {
            return qs[st][quad][min(g, G - 1)];
        } else {
            return qf[st];
        }
    };

    float m = neg_inf(), lsum = 0.f;
    float acc[PVM ? 1 : G][8];
#pragma unroll
    for (int h = 0; h < (PVM ? 1 : G); ++h)
#pragma unroll
        for (int c = 0; c < 8; ++c) acc[h][c] = 0.f;
    // Under PVM: Oᵀ tile mt holds columns 16mt + 4·quad + t of head g.
    mfma_cd ot[PVM ? D / 16 : 1];
#pragma unroll
    for (int mt = 0; mt < (PVM ? D / 16 : 1); ++mt) ot[mt] = mfma_zero();
    const int col = RE * (l % LPK), parity = l / LPK;

    const i32* pages = p.page_indices + p.page_indptr[req];
    const int ps_ = p.page_size;
    const RelRow rel = rel_row(p, p.q_indptr[req], kv_head * G + min(g, G - 1), kv_len - 1);
    for (int page0 = lo / ps_; page0 * ps_ < end; page0 += TABLE) {
    const int w0 = max(lo, page0 * ps_), wend = min(end, (page0 + TABLE) * ps_);
    __syncthreads();
    for (int i = threadIdx.x; i < TABLE && page0 + i <= (wend - 1) / ps_; i += blockDim.x) table[i] = pages[page0 + i];
    __syncthreads();
    auto at = [&](int t, int j) {
        return kv_at<HND>(static_cast<long long>(table[t / ps_ - page0]), t % ps_, kv_head, j, ps_, p.num_kv_heads, D, p.stride_page);
    };
    for (int t0 = w0 + wave * SUB; t0 < wend; t0 += WAVES * SUB) {
        // The round's key and value reads, all issued before any is used
        // (past the chunk they reread its last key, which weighs zero).
        mfma_cd s[KT];
        typename KV::Raw kr[KT][KP];
#pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
            KV::keys(p.k, at(min(t0 + kt * 16 + g, wend - 1), 0), quad, 0, kr[kt]);
        }
#pragma unroll
        for (int kt = 0; kt < KT; ++kt) s[kt] = mfma_zero();
        // The value reads ride ahead of the scores.
        using VRaw = typename conditional<RE == 16, Pack8<bf16>, typename KV::Raw>::type;
        auto value_at = [&](int i) -> VRaw {
            const long long v = at(min(t0 + i * KPI + parity, wend - 1), col);
            if constexpr (RE == 16) return *reinterpret_cast<const VRaw*>(reinterpret_cast<const u8*>(p.v) + v);
            else return KV::raw(p.v, v);
        };
        VRaw vr[VR];
#pragma unroll
        for (int i = 0; i < VR; ++i) vr[i] = value_at(i);
#pragma unroll
        for (int kt = 0; kt < KT; ++kt)
#pragma unroll
            for (int st = 0; st < KP; ++st) s[kt] = mfma16x2(KV::ab2(kr[kt][st]), q_at(st), s[kt]);
        float tmax = neg_inf();
#pragma unroll
        for (int kt = 0; kt < KT; ++kt)
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int key = t0 + kt * 16 + 4 * quad + t;
                s[kt][t] = key < wend && (!MASK || mask.reads(key)) ? rel.score(s[kt][t], key) : neg_inf();
                tmax = fmaxf(tmax, s[kt][t]);
            }
        tmax = fmaxf(tmax, __shfl_xor(tmax, 16, kWave));
        tmax = fmaxf(tmax, __shfl_xor(tmax, 32, kWave));
        const float m_new = fmaxf(m, tmax);
        const float alpha = m_new == neg_inf() ? 1.f : exp2f(m - m_new);
        float sum = 0.f;
        float ew[KT][4];
#pragma unroll
        for (int kt = 0; kt < KT; ++kt)
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const float e = m_new == neg_inf() ? 0.f : exp2f(s[kt][t] - m_new);
                sum += e;
                ew[kt][t] = e;
                if constexpr (!PVM) {
                    if (g < G) ps[wave][kt * 16 + 4 * quad + t][g] = e;
                }
            }
        lsum = lsum * alpha + sum;
        m = m_new;
        if constexpr (PVM) {
            static_assert(KT == 1, "one 16-key tile a round");
            // The round's value rows into this wavefront's LDS rows, a column
            // half at a time: each half's lanes stage theirs, then its tiles
            // take Vᵀ · Pᵀ.
            bf16* vs = vbuf + wave * SUB * VLD;
            const mfma_ab b = pack4(ew[0][0], ew[0][1], ew[0][2], ew[0][3]);
#pragma unroll
            for (int half = 0; half < HV; ++half) {
                if (col / DH == half) {
#pragma unroll
                    for (int i = 0; i < VR; ++i) {
                        bf16* row = vs + (i * KPI + parity) * VLD + (col - half * DH);
                        if constexpr (RE == 16) {
                            const auto w = __builtin_bit_cast(mfma_ab2, vr[i]);
                            *reinterpret_cast<mfma_ab2*>(row) = e4m3x8_bf16(__builtin_bit_cast(u64, w.lo));
                            *reinterpret_cast<mfma_ab2*>(row + 8) = e4m3x8_bf16(__builtin_bit_cast(u64, w.hi));
                        } else {
                            *reinterpret_cast<Pack8<bf16>*>(row) = KV::wide(vr[i]);
                        }
                    }
                }
                __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
                __builtin_amdgcn_wave_barrier();
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
#pragma unroll
                for (int mh = 0; mh < DH / 16; ++mh) {
                    const int mt = half * (DH / 16) + mh;
                    mfma_ab a;
#pragma unroll
                    for (int t = 0; t < 4; ++t)
                        a[t] = __builtin_bit_cast(short, vs[(4 * quad + t) * VLD + 16 * mh + g]);
#pragma unroll
                    for (int t = 0; t < 4; ++t) ot[mt][t] *= alpha;
                    ot[mt] = mfma16(a, b, ot[mt]);
                }
                __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
                __builtin_amdgcn_wave_barrier();
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
            }
            continue;
        }
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
        auto weigh = [&](int i, const auto& raw) {
            const Pack8<bf16> v = KV::wide(raw);
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
        if constexpr (PVM) {
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
    if constexpr (!PVM) {
#pragma unroll
        for (int off = LPK; off < kWave; off <<= 1)
#pragma unroll
            for (int h = 0; h < G; ++h)
#pragma unroll
                for (int c = 0; c < 8; ++c) acc[h][c] += __shfl_xor(acc[h][c], off, kWave);
    } else {
        // (the fold below reuses the value rows' LDS)
        __syncthreads();
    }
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
        if constexpr (PVM) {
            if (g >= h0 && g < h0 + HG && g < G)
#pragma unroll
                for (int mt = 0; mt < D / 16; ++mt)
#pragma unroll
                    for (int t = 0; t < 4; ++t) part_o(wave, g - h0, 16 * mt + 4 * quad + t) = ot[mt][t];
        } else if (l < LPK) {
#pragma unroll
            for (int h = 0; h < HG; ++h)
#pragma unroll
                for (int c = 0; c < 8; ++c) part_o(wave, h, col + c) = acc[h0 + h][c];
        }
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
                    out += part_o(w, hl, j) * f;
                }
            }
            const float value = sum > 0.f ? out / sum * KV::unit : 0.f;
            const long long at = (row + kv_head * G + h) * D + j;
            if (p.partition) static_cast<float*>(p.o)[at] = value;
            else static_cast<bf16*>(p.o)[at] = from_f32<bf16>(value);
            if (j == 0 && p.lse != nullptr) p.lse[row + kv_head * G + h] = sum > 0.f ? big + log2f(sum) : neg_inf();
        }
    }
}

// --------------------------------------------------------------- prefill --
//
// MFMA flash attention for D = 512. A block owns 128 packed query rows (row
// r is token r / G, head kv_head·G + r % G) as eight wavefronts of 16, and
// walks its chunk BC keys a tile through LDS (V transposed): Sᵀ = K·Qᵀ, Oᵀ
// += Vᵀ·Pᵀ. The Sᵀ result is already Pᵀ's operand layout and each lane's
// output column one query row, so the rescale is the lane's own. LDS rows
// are padded so an operand read's 16 rows hit distinct banks.

constexpr int kPrefillTileQ = 128;

// The keys a tile reads from its first under a window of `wl` keys behind a
// row (kv_len without one): the window, then its rows' tokens, one more
// where a tile starts mid-token (sched_prefill's `window_reach`).
__device__ __forceinline__ int window_reach(int wl, int G) {
    return wl + (kPrefillTileQ + G - 1) / G + 1;
}

// Keys a kv tile: a tile is whole 16-byte reads for every thread (64-wide
// heads take 64 keys).
template <int D>
constexpr int prefill_bc() {
    return D >= 512 ? 16 : D <= 64 ? 64 : 32;
}

// A prefill block's (work item, kv head): each XCD (block b runs on XCD b
// mod 8) takes a run of the grid in (head, item) order, so the items it runs
// at once are neighbouring query tiles of one head and find that head's keys
// and values in its L2; causal items walk each head from its last (the
// longest) down.
__device__ __forceinline__ void prefill_item(bool causal, int& bx, int& kv_head) {
    const int total = gridDim.x * gridDim.y, b = blockIdx.x + gridDim.x * blockIdx.y;
    const int xcd = b % 8, q = total / 8, rem = total % 8;
    const int at = xcd * q + min(xcd, rem) + b / 8;
    kv_head = at / gridDim.x;
    bx = at % gridDim.x;
    if (causal) bx = gridDim.x - 1 - bx;
}

// The masked form's two segments, a request each (`kernels_cuda`'s custom
// mask): below `origin` its committed run, every key read (and cut at the
// row's own position unless the word's wide bit is set); from `origin` on its
// buffer, key j of query row r read where bit r·(kv_len − origin) + (j −
// origin) of the request's plane is set, the plane from byte indptr[req],
// least bit first. No origin table: origin 0, wide.
struct PrefillMask {
    const u8* bits;
    const i64* indptr;
    const u32* origin;
};

template <int D, bool HND, bool CAUSAL, bool MASK, bool FP8>
__device__ __forceinline__ void paged_prefill_wide_body(const Paged& p, const PrefillMask& mask) {
    using KV = Kv<FP8>;
    constexpr int BC = prefill_bc<D>();
    constexpr int KS = D / 16;   // d tiles for Oᵀ
    constexpr int KP = D / 32;   // paired k steps over the head for Sᵀ
    constexpr int KT = BC / 16;  // key tiles per kv tile
    constexpr int KPAD = D + 8, VPAD = BC + 4;
    constexpr int THREADS = 512, RD = BC * D / 8 / THREADS;  // 16-byte reads a thread a tile
    static_assert(RD * 8 * THREADS == BC * D && THREADS % BC == 0, "the tile splits evenly");
    __shared__ alignas(16) bf16 ks[BC][KPAD];
    __shared__ alignas(16) bf16 vt[D][VPAD];

    int bx, kv_head;
    prefill_item(CAUSAL, bx, kv_head);
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
    // A masked request is cut at each row's position unless its word is wide.
    int origin = 0;
    bool causal = CAUSAL;
    const u8* bits = nullptr;
    if constexpr (MASK) {
        const u32 word = mask.origin != nullptr ? mask.origin[req] : 0x80000000u;
        origin = static_cast<int>(word & 0x7fffffffu);
        causal = (word >> 31) == 0;
        bits = mask.bits + mask.indptr[req];
    }

    const int kv_start = max(0, kv_len + (qo_tile * kPrefillTileQ) / G - qo_len - wl);
    const int chunk_start = p.partition ? min(kv_tile * chunk + kv_start, kv_len) : kv_start;
    int chunk_end = p.partition ? min((kv_tile + 1) * chunk + kv_start, kv_len) : kv_len;
    if (causal) {
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

    // Qᵀ as B operands, k paired 8·quad: column = this lane's query row.
    mfma_ab2 qf[KP];
    {
        const bf16* qrow = p.q + static_cast<long long>(q_first + min(token, qo_len - 1)) * p.q_stride_n +
                           static_cast<long long>(head) * p.q_stride_h + 8 * quad;
#pragma unroll
        for (int s = 0; s < KP; ++s) qf[s] = load8(qrow + s * 32);
    }

    mfma_cd o[KS];
#pragma unroll
    for (int s = 0; s < KS; ++s) o[s] = mfma_cd{0.f, 0.f, 0.f, 0.f};
    float m = neg_inf(), l = 0.f;
    // The last key this row may see, and the first (the window).
    const int key_hi = causal ? kv_len - qo_len + token : kv_len - 1;
    const int key_lo = kv_len - qo_len + token - wl;
    // This row's bits of the buffer: key j's at row_bits + (j − origin).
    const long long row_bits = static_cast<long long>(token) * (kv_len - origin);

    // A thread's reads of a tile: K piece c = tid + r·THREADS as (key c /
    // (D/8), columns 8·(c % (D/8))), a wavefront one key's row; V piece c as
    // (key c % BC, columns 8·(c / BC)), a wavefront 16 keys of four columns.
    // Every read is issued (a key past the chunk reads the chunk's last,
    // and lands as zeros at the store), so no read waits on another.
    // With an even RD a thread's V reads pair up: keys 2·(c % (BC/2)) and
    // +1 of columns 8·(c / (BC/2)), landed in Vᵀ four bytes a write.
    constexpr bool PAIRS = RD % 2 == 0;
    typename KV::Raw kr[RD], vr[RD];
    auto v_at = [&](int r, int& vkey, int& vj) {
        if constexpr (PAIRS) {
            const int c = threadIdx.x + (r / 2) * THREADS;
            vkey = 2 * (c % (BC / 2)) + r % 2;
            vj = (c / (BC / 2)) * 8;
        } else {
            const int c = threadIdx.x + r * THREADS;
            vkey = c % BC;
            vj = (c / BC) * 8;
        }
    };
    auto fetch = [&](int kv0) {
#pragma unroll
        for (int r = 0; r < RD; ++r) {
            const int c = threadIdx.x + r * THREADS;
            const int kkey = kv0 + c / (D / 8), kj = (c % (D / 8)) * 8;
            int vkey, vj;
            v_at(r, vkey, vj);
            kr[r] = KV::raw(p.k, kv_token<HND, D>(p, req, min(kkey, chunk_end - 1), kv_head, kj));
            vr[r] = KV::raw(p.v, kv_token<HND, D>(p, req, min(kv0 + vkey, chunk_end - 1), kv_head, vj));
        }
    };
    auto stage = [&](int kv0) {
#pragma unroll
        for (int r = 0; r < RD; ++r) {
            const int c = threadIdx.x + r * THREADS;
            const Pack8<bf16> k8 = kv0 + c / (D / 8) < chunk_end ? KV::wide(kr[r]) : Pack8<bf16>{};
            *reinterpret_cast<Pack8<bf16>*>(&ks[c / (D / 8)][(c % (D / 8)) * 8]) = k8;
        }
#pragma unroll
        for (int r = 0; r < RD; r += PAIRS ? 2 : 1) {
            int vkey, vj;
            v_at(r, vkey, vj);
            const Pack8<bf16> v0 = kv0 + vkey < chunk_end ? KV::wide(vr[r]) : Pack8<bf16>{};
            if constexpr (PAIRS) {
                const Pack8<bf16> v1 = kv0 + vkey + 1 < chunk_end ? KV::wide(vr[r + 1]) : Pack8<bf16>{};
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const bf16 pair[2] = {v0.v[e], v1.v[e]};
                    *reinterpret_cast<u32*>(&vt[vj + e][vkey]) = *reinterpret_cast<const u32*>(pair);
                }
            } else {
#pragma unroll
                for (int e = 0; e < 8; ++e) vt[vj + e][vkey] = v0.v[e];
            }
        }
    };
    // The next tile is read into registers across the tile's products, where
    // they have room: at D = 512 the output's (D/16 · 4) and Qᵀ's (D/32 · 4)
    // registers take 192 of the 256 a lane has at two wavefronts a SIMD, and
    // a tile held beside them spills to scratch. There the tile is read just
    // before it is staged.
    constexpr bool AHEAD = D < 512;
    if (AHEAD && chunk_start < chunk_end) fetch(chunk_start);

    for (int kv0 = chunk_start; kv0 < chunk_end; kv0 += BC) {
        if constexpr (!AHEAD) fetch(kv0);
        stage(kv0);
        lds_barrier();
        if (AHEAD && kv0 + BC < chunk_end) fetch(kv0 + BC);
        if (wave_live) {
            // Sᵀ: lane holds keys 4·quad + t of each 16-key tile, for its row;
            // two chains over the head, so a product waits on the one before
            // the last.
            mfma_cd s[KT];
#pragma unroll
            for (int kt = 0; kt < KT; ++kt) {
                mfma_cd c[2] = {mfma_zero(), mfma_zero()};
#pragma unroll
                for (int st = 0; st < KP; ++st)
                    c[st % 2] = mfma16x2(load8(&ks[kt * 16 + ln % 16][st * 32 + 8 * quad]), qf[st], c[st % 2]);
                s[kt] = c[0] + c[1];
            }
            float tile_max = neg_inf();
#pragma unroll
            for (int kt = 0; kt < KT; ++kt)
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const int key = kv0 + kt * 16 + 4 * quad + t;
                    bool seen = row_live && key < chunk_end && key <= key_hi && key >= key_lo;
                    if constexpr (MASK) {
                        if (seen && key >= origin) {
                            const long long bit = row_bits + (key - origin);
                            seen = (bits[bit / 8] >> (bit % 8)) & 1u;
                        }
                    }
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
        lds_barrier();
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
    const float inv = l > 0.f ? KV::unit / l : 0.f;
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

template <int D, bool HND, bool CAUSAL, bool FP8>
__global__ __launch_bounds__(512) void paged_prefill_wide(const Paged p) {
    paged_prefill_wide_body<D, HND, CAUSAL, false, FP8>(p, PrefillMask{});
}

template <int D, bool HND>
__global__ __launch_bounds__(512) void paged_prefill_wide_masked(const Paged p, const PrefillMask mask) {
    paged_prefill_wide_body<D, HND, false, true, false>(p, mask);
}

// MFMA flash attention for heads up to 256, the wide kernel's products with
// the page table in LDS, the next tile read into registers across an
// LDS-only barrier (`lds_barrier`), and K / Vᵀ rows padded 8 bytes for the
// banks. 128 packed query rows as eight wavefronts, keys 32 a tile. The
// selected (DSA / QSA) form masks each key by its row's LDS bitmap of the
// pooled blocks its selection names, so the tile's rows share K and V reads.
struct PrefillSel {
    const i32* selection;  // [row][top_k], the rows of q; per head [row][kv head][top_k]
    i32 top_k;
    i32 ratio;
    i32 per_head;
};
constexpr int kSelTokens = 16;
constexpr int kSelBlocks = 2048;
constexpr int kSelWords = kSelBlocks / 32;


// The narrow prefill's tiles in LDS (K, Vᵀ, the page table), one allocation
// a head width: every body of a width shares it (the masked prefill runs the
// unmasked body or the masked one in the same kernel).
template <int D>
__device__ __forceinline__ unsigned char* prefill_lds() {
    constexpr int BC = 32, KPAD = D + 4, VPAD = BC + 4, TABLE = 512;
    __shared__ alignas(16) unsigned char lds[(BC * KPAD + D * VPAD) * sizeof(bf16) + TABLE * sizeof(i32)];
    return lds;
}

template <int D, bool HND, bool CAUSAL, bool SEL, bool MASK, bool FP8>
__device__ __forceinline__ void paged_prefill_body(const Paged& p, const PrefillSel& sel,
                                                   const PrefillMask& mask) {
    using KV = Kv<FP8>;
    constexpr int BC = 32, KT = BC / 16, KP = D / 32, DT = D / 16;
    constexpr int KPAD = D + 4, VPAD = BC + 4;
    constexpr int THREADS = 512, TABLE = 512;
    static_assert(D == 128 || D == 256, "the narrow prefill");
    // Per thread a round: D/128 16-byte K reads (key t/16, columns from
    // (t%16)·D/16) and, for the first 4·D threads, two V reads (keys 2(t%16)
    // and +1, columns 8·(t/16)).
    constexpr int KR = BC * D / 8 / THREADS;
    bf16* const ks = reinterpret_cast<bf16*>(prefill_lds<D>());
    bf16* const vt = ks + BC * KPAD;
    i32* const table = reinterpret_cast<i32*>(vt + D * VPAD);
    static_assert((BC * KPAD * sizeof(bf16)) % 16 == 0 && (D * VPAD * sizeof(bf16)) % 16 == 0,
                  "each tile on a 16-byte bound");

    // Causal tiles grow with their position: the last of a request walks the
    // most keys, so the grid starts there and the short ones fill the tail.
    int bx, kv_head;
    prefill_item(CAUSAL, bx, kv_head);
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
    // A masked request is cut at each row's position unless its word is wide.
    int origin = 0;
    bool causal = CAUSAL;
    const u8* bits = nullptr;
    if constexpr (MASK) {
        const u32 word = mask.origin != nullptr ? mask.origin[req] : 0x80000000u;
        origin = static_cast<int>(word & 0x7fffffffu);
        causal = (word >> 31) == 0;
        bits = mask.bits + mask.indptr[req];
    }

    const int kv_start = max(0, kv_len + (qo_tile * kPrefillTileQ) / G - qo_len - wl);
    const int chunk_start = p.partition ? min(kv_tile * chunk + kv_start, kv_len) : kv_start;
    int chunk_end = p.partition ? min((kv_tile + 1) * chunk + kv_start, kv_len) : kv_len;
    if (causal) {
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
    const int key_hi = causal ? kv_len - qo_len + token : kv_len - 1;
    // This row's bits of the buffer: key j's at row_bits + (j − origin).
    const long long row_bits = static_cast<long long>(token) * (kv_len - origin);
    const int key_lo = kv_len - qo_len + token - wl;

    // The tile's tokens' selected blocks, and how many blocks each pools.
    __shared__ u32 sel_bits[SEL ? kSelTokens : 1][SEL ? kSelWords : 1];
    __shared__ int sel_blocks[SEL ? kSelTokens : 1];
    const int tok_lo = (qo_tile * kPrefillTileQ) / G;
    if constexpr (SEL) {
        const int tok_n = min(qo_len, ((qo_tile + 1) * kPrefillTileQ + G - 1) / G) - tok_lo;
        for (int i = tid; i < kSelTokens * kSelWords; i += THREADS) sel_bits[i / kSelWords][i % kSelWords] = 0;
        if (tid < kSelTokens) sel_blocks[tid] = tid < tok_n ? (kv_len - qo_len + tok_lo + tid + 1) / sel.ratio : 0;
        __syncthreads();
        for (int i = tid; i < tok_n * sel.top_k; i += THREADS) {
            const int r = i / sel.top_k, e = i % sel.top_k;
            const int nb = sel_blocks[r];
            if (e >= min(sel.top_k, nb)) continue;
            const long long row = q_first + tok_lo + r;
            const int c = sel.per_head
                              ? sel.selection[(row * p.num_kv_heads + kv_head) * sel.top_k + e]
                              : sel.selection[row * sel.top_k + e];
            if (c >= 0 && c < nb) atomicOr(&sel_bits[r][c / 32], 1u << (c % 32));
        }
        __syncthreads();
    }
    const int sel_row = token - tok_lo;
    const RelRow rel = rel_row(p, q_first + min(token, qo_len - 1), head, kv_len - qo_len + token);

    const i32* pages = p.page_indices + p.page_indptr[req];
    const int ps = p.page_size;
    // This thread's reads of a round.
    const int k_key = tid / (D / 8 / KR), k_col = (tid % (D / 8 / KR)) * 8 * KR;
    const int v_pair = tid % (BC / 2), v_col = (tid / (BC / 2)) * 8;
    const bool v_live = v_col < D;
    typename KV::Raw kreg[KR], vreg[2];
    int page0 = 0, wend = 0;
    auto fetch = [&](int kv0) {
        auto at = [&](int t, int j) {
            t = min(t, wend - 1);
            return kv_at<HND>(static_cast<long long>(table[t / ps - page0]), t % ps, kv_head, j, ps, p.num_kv_heads, D, p.stride_page);
        };
#pragma unroll
        for (int r = 0; r < KR; ++r)
            kreg[r] = KV::raw(p.k, at(kv0 + k_key, k_col + 8 * r));
        if (v_live) {
#pragma unroll
            for (int e = 0; e < 2; ++e) vreg[e] = KV::raw(p.v, at(kv0 + 2 * v_pair + e, v_col));
        }
    };
    auto stage = [&](int kv0) {
        // Keys past the chunk land as zeros, so no garbage meets a weight.
#pragma unroll
        for (int r = 0; r < KR; ++r) {
            Pack8<bf16> kv = KV::wide(kreg[r]);
            if (kv0 + k_key >= wend) kv = Pack8<bf16>{};
            *reinterpret_cast<Pack8<bf16>*>(&ks[k_key * KPAD + k_col + 8 * r]) = kv;
        }
        if (v_live) {
            Pack8<bf16> vw[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) vw[e] = kv0 + 2 * v_pair + e >= wend ? Pack8<bf16>{} : KV::wide(vreg[e]);
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const bf16 pair[2] = {vw[0].v[c], vw[1].v[c]};
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
                    bool seen = row_live && key < wend && key <= key_hi && key >= key_lo;
                    if constexpr (SEL) {
                        const int b = key / sel.ratio;
                        seen = seen && (b >= sel_blocks[sel_row] || ((sel_bits[sel_row][b / 32] >> (b % 32)) & 1u));
                    }
                    if constexpr (MASK) {
                        if (seen && key >= origin) {
                            const long long bit = row_bits + (key - origin);
                            seen = (bits[bit / 8] >> (bit % 8)) & 1u;
                        }
                    }
                    s[kt][t] = seen ? rel.score(s[kt][t], key) : neg_inf();
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
    const float inv = l > 0.f ? KV::unit / l : 0.f;
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

template <int D, bool HND, bool CAUSAL, bool FP8>
__global__ __launch_bounds__(512) void paged_prefill(const Paged p) {
    paged_prefill_body<D, HND, CAUSAL, false, false, FP8>(p, PrefillSel{}, PrefillMask{});
}

template <int D, bool HND>
__global__ __launch_bounds__(512) void paged_prefill_selected(const Paged p, const PrefillSel sel) {
    paged_prefill_body<D, HND, true, true, false, false>(p, sel, PrefillMask{});
}

// `CAUSAL`: the plan's (as the unmasked prefill's): its block order, the
// causal tiles' longest first. Each request is still cut at its rows'
// positions, or not, by its mask word's wide bit.
template <int D, bool HND, bool CAUSAL>
__global__ __launch_bounds__(512) void paged_prefill_masked(const Paged p, const PrefillMask mask) {
    // A request stating no mask (its word not wide, its boundary at or past
    // its keys: no bit is read) is the unmasked causal prefill's: the same
    // keys, cut at the same positions. Its block runs that body, the mask
    // and the cut compile-time, as the unmasked prefill does; any other
    // request runs the masked body. One body serves both kinds of lane.
    if constexpr (CAUSAL) {
        int bx, kv_head;
        prefill_item(CAUSAL, bx, kv_head);
        if (p.block_valid != nullptr && p.block_valid[bx] == 0) return;
        const int req = p.request_indices[bx];
        const u32 word = mask.origin != nullptr ? mask.origin[req] : 0x80000000u;
        if ((word >> 31) == 0 && static_cast<int>(word & 0x7fffffffu) >= kv_length(p, req)) {
            paged_prefill_body<D, HND, true, false, false, false>(p, PrefillSel{}, PrefillMask{});
            return;
        }
    }
    paged_prefill_body<D, HND, CAUSAL, false, true, false>(p, PrefillSel{}, mask);
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
