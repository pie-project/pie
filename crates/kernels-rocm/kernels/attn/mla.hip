#pragma once

// Multi-head latent attention (MLA) over the paged latent cache, and the ops
// that feed it. A cache row is a token's latent (`rank` wide, the keys' and
// values' shared part) beside its rotated half (`rope` wide, keys only),
// held in two page planes; the query arrives absorbed into the latent space,
// so every head scores q_nope·latent + q_pe·rope and reads the latent back.
//
// Scores live in the log2 domain (s = q·k · sm_scale · log2 e); a log-sum-
// exp is m + log2 Σ 2^(s − m), the convention every lse plane here carries.

#include "prelude/device.hip"
#include "prelude/mfma.hip"

namespace pie::attn {

// ---------------------------------------------------------------- feed --

// q_b's per-head [nope | rope] rows split into the two planes.
template <class T>
__global__ void mla_split_q_b(const T* __restrict__ q_b, T* __restrict__ q_nope, T* __restrict__ q_pe,
                              int total, int heads, int nope, int rope, const u32* __restrict__ win) {
    const Window window{win};
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int per = nope + rope;
    const int d = i % per;
    const int h = (i / per) % heads;
    const int n = i / (heads * per);
    if (!window.live(n)) return;
    const long long row = window.plane(n) * heads + h;
    const T v = q_b[row * per + d];
    if (d < nope) q_nope[row * nope + d] = v;
    else q_pe[row * rope + (d - nope)] = v;
}

// kv_a's [latent | rope] row: the latent RMS-normed into kv_c, the rope
// half copied into k_pe. One block per row.
template <class T, int BLOCK>
__global__ __launch_bounds__(BLOCK) void mla_latents(const T* __restrict__ kv_a, const T* __restrict__ weight,
                                                     T* __restrict__ kv_c, T* __restrict__ k_pe, int kv_lora,
                                                     int rope, int src_row_stride, float eps,
                                                     const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long n = window.plane(blockIdx.x);
    const T* row = kv_a + n * src_row_stride;
    for (int d = threadIdx.x; d < rope; d += BLOCK) k_pe[n * rope + d] = row[kv_lora + d];

    __shared__ float smem[BLOCK / kWave];
    float local = 0.f;
    for (int d = threadIdx.x; d < kv_lora; d += BLOCK) {
        const float v = to_f32(row[d]);
        local += v * v;
    }
    const float inv = rsqrtf(block_sum(local, smem) / static_cast<float>(kv_lora) + eps);
    for (int d = threadIdx.x; d < kv_lora; d += BLOCK)
        kv_c[n * kv_lora + d] = from_f32<T>(to_f32(row[d]) * inv * to_f32(weight[d]));
}

// The request a ragged row `t` belongs to: the last r with indptr[r] <= t.
__device__ __forceinline__ int request_of(const i32* __restrict__ indptr, int requests, int t) {
    int lo = 0, hi = requests - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (indptr[mid] <= t) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

__device__ __forceinline__ int paged_len(const i32* page_indptr, const i32* last_page_lens, int r,
                                         int page_size) {
    const int pages = page_indptr[r + 1] - page_indptr[r];
    return pages <= 0 ? 0 : (pages - 1) * page_size + last_page_lens[r];
}

template <class T>
__device__ __forceinline__ void copy_row(const T* __restrict__ src, T* __restrict__ dst, int n) {
    if (n % 8 == 0 && aligned16(src) && aligned16(dst)) {
        for (int i = threadIdx.x; i < n / 8; i += blockDim.x)
            reinterpret_cast<b128*>(dst)[i] = reinterpret_cast<const b128*>(src)[i];
    } else {
        for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
    }
}

// The step's new latent (and rope) rows into their cache slots: row t of
// request r lands at kv position (kv_len_r − new_r) + (t − qo_lo_r). One
// block per row; `kpe` null appends the latent plane alone.
__global__ void mla_kv_append(const bf16* __restrict__ ckv, const bf16* __restrict__ kpe,
                              bf16* __restrict__ ckv_pages, bf16* __restrict__ kpe_pages,
                              const i32* __restrict__ qo_indptr, const i32* __restrict__ page_indices,
                              const i32* __restrict__ page_indptr, const i32* __restrict__ last_page_lens,
                              const u8* __restrict__ row_valid, int requests, int page_size, int rank,
                              int rope) {
    const int t = blockIdx.x;
    if (requests <= 0 || t >= qo_indptr[requests]) return;
    if (row_valid != nullptr && row_valid[t] == 0) return;
    const int r = request_of(qo_indptr, requests, t);
    const int kv_len = paged_len(page_indptr, last_page_lens, r, page_size);
    const int pos = kv_len - (qo_indptr[r + 1] - qo_indptr[r]) + (t - qo_indptr[r]);
    if (pos < 0) return;
    const long long slot =
        static_cast<long long>(page_indices[page_indptr[r] + pos / page_size]) * page_size + pos % page_size;
    copy_row(ckv + static_cast<long long>(t) * rank, ckv_pages + slot * rank, rank);
    if (kpe != nullptr && rope > 0)
        copy_row(kpe + static_cast<long long>(t) * rope, kpe_pages + slot * rope, rope);
}

// ---------------------------------------------------------------- keys --

// The keys a query row walks: list entry n names key position
// p = sel ? sel[n] : n, visible when 0 <= p <= limit; key p's row is cell
// (p + 1)·ratio − 1 of the request's pages (ratio 1: the token itself; a
// compressed pool keeps its entry at the last cell of each block).
struct LatentKeys {
    const i32* sel;
    const i32* pages;
    int page_size;
    int ratio;
    int n0, n1;
    int limit;
    // The first position a windowed row still reads; 0 reads every row.
    int floor = 0;

    __device__ __forceinline__ int pos(int n) const { return sel != nullptr ? sel[n] : n; }
    __device__ __forceinline__ bool visible(int p) const { return p >= floor && p <= limit; }
    __device__ __forceinline__ int cell(int p) const { return (p + 1) * ratio - 1; }
    __device__ __forceinline__ int page(int p) const { return pages[cell(p) / page_size]; }
    __device__ __forceinline__ long long slot(int p, int page_id) const {
        return static_cast<long long>(page_id) * page_size + cell(p) % page_size;
    }
};

// --------------------------------------------------------------- naive --
//
// The reference walk, for any latent and rope width: a wavefront per
// (query row, key stride), lane i holding elements i + 64·j of the row, an
// online softmax per wavefront, the wavefronts of a row folded through LDS.

namespace mla_naive {

constexpr int kBlock = 256;
constexpr int kWaves = kBlock / kWave;
constexpr int kMaxPer = 16;  // latent ≤ 1024
constexpr int kMaxPe = 4;    // rope ≤ 256

// G query rows a block, kWaves/G wavefronts each; `out(g, d, value)` and
// `out_lse(g, lse)` land row g's results. `smem` holds kWaves·ckv + 2·kWaves
// floats. The key rows are `ckv` wide at `ckv_pages` (also the values) and
// `kpe` wide at `kpe_pages`.
template <class Out, class OutLse>
__device__ void attend(const bf16* __restrict__ qn, const bf16* __restrict__ qp,
                       const bf16* __restrict__ ckv_pages, const bf16* __restrict__ kpe_pages, int ckv,
                       int kpe, const LatentKeys& keys, float scale_log2, int G, float* smem, Out&& out,
                       OutLse&& out_lse) {
    const int wv = threadIdx.x / kWave, ln = lane();
    const int K = kWaves / G, sub = wv % K;
    float qr[kMaxPer], qpr[kMaxPe], acc[kMaxPer];
#pragma unroll
    for (int i = 0; i < kMaxPer; ++i) {
        const int d = ln + kWave * i;
        qr[i] = d < ckv ? to_f32(qn[d]) : 0.f;
        acc[i] = 0.f;
    }
#pragma unroll
    for (int i = 0; i < kMaxPe; ++i) {
        const int d = ln + kWave * i;
        qpr[i] = d < kpe ? to_f32(qp[d]) : 0.f;
    }
    float m = neg_inf(), l = 0.f;
    for (int n = keys.n0 + sub; n < keys.n1; n += K) {
        const int p = keys.pos(n);
        if (!keys.visible(p)) continue;
        const long long slot = keys.slot(p, keys.page(p));
        const bf16* kr = ckv_pages + slot * ckv;
        const bf16* pr = kpe_pages + slot * kpe;
        float kv[kMaxPer];
        float dot = 0.f;
#pragma unroll
        for (int i = 0; i < kMaxPer; ++i) {
            const int d = ln + kWave * i;
            kv[i] = d < ckv ? to_f32(kr[d]) : 0.f;
            dot += qr[i] * kv[i];
        }
#pragma unroll
        for (int i = 0; i < kMaxPe; ++i) {
            const int d = ln + kWave * i;
            if (d < kpe) dot += qpr[i] * to_f32(pr[d]);
        }
        const float s = wave_sum(dot) * scale_log2;
        const float m_new = fmaxf(m, s);
        const float corr = exp2f(m - m_new);
        const float e = exp2f(s - m_new);
        l = l * corr + e;
#pragma unroll
        for (int i = 0; i < kMaxPer; ++i) acc[i] = acc[i] * corr + e * kv[i];
        m = m_new;
    }

    float* wacc = smem;
    float* wm = wacc + kWaves * ckv;
    float* wl = wm + kWaves;
#pragma unroll
    for (int i = 0; i < kMaxPer; ++i) {
        const int d = ln + kWave * i;
        if (d < ckv) wacc[wv * ckv + d] = acc[i];
    }
    if (ln == 0) {
        wm[wv] = m;
        wl[wv] = l;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < G * ckv; i += kBlock) {
        const int g = i / ckv, d = i % ckv;
        float big = neg_inf();
        for (int w = g * K; w < g * K + K; ++w) big = fmaxf(big, wm[w]);
        float sum = 0.f, v = 0.f;
        if (big != neg_inf()) {
            for (int w = g * K; w < g * K + K; ++w) {
                const float f = exp2f(wm[w] - big);
                sum += wl[w] * f;
                v += wacc[w * ckv + d] * f;
            }
        }
        out(g, d, sum > 0.f ? v / sum : 0.f);
        if (d == 0) out_lse(g, sum > 0.f ? big + log2f(sum) : neg_inf());
    }
}

// One block per (token, group of G heads). With a selection, the token's
// keys are the positions its row lists (entries outside the causal range
// skipped); without, every key up to its causal bound. A `band` past 0
// keeps the last `band` positions through the row's own; `lse`, when not
// null, takes each head's base-2 log-sum-exp.
__global__ __launch_bounds__(kBlock) void mla_naive_paged_kernel(
    const bf16* __restrict__ q_nope, const bf16* __restrict__ q_pe, const bf16* __restrict__ ckv_pages,
    const bf16* __restrict__ kpe_pages, const i32* __restrict__ qo_indptr,
    const i32* __restrict__ page_indices, const i32* __restrict__ page_indptr,
    const i32* __restrict__ last_page_lens, bf16* __restrict__ o, float* __restrict__ lse,
    const i32* __restrict__ selection, int top_k, int requests, int heads, int ckv, int kpe,
    int page_size, float sm_scale, bool causal, int G, int band, const u32* __restrict__ win) {
    extern __shared__ float naive_smem[];
    const Window window{win};
    const int t = blockIdx.x;
    if (!window.live(t)) return;
    const long long t_row = window.plane(t);
    requests = window.lanes(requests);
    const int K = kWaves / G;
    const int h = blockIdx.y * G + (threadIdx.x / kWave) / K;

    const int r = request_of(qo_indptr, requests, t);
    const int kv_len = paged_len(page_indptr, last_page_lens, r, page_size);
    const int abs_q = kv_len - (qo_indptr[r + 1] - qo_indptr[r]) + (t - qo_indptr[r]);
    const int j_end = causal ? min(abs_q + 1, kv_len) : kv_len;
    const int floor = band > 0 ? max(0, abs_q + 1 - band) : 0;
    LatentKeys keys{selection != nullptr ? selection + t_row * top_k : nullptr,
                    page_indices + page_indptr[r],
                    page_size,
                    1,
                    selection != nullptr ? 0 : floor,
                    selection != nullptr ? min(top_k, max(j_end, 0)) : j_end,
                    j_end - 1,
                    floor};
    const long long head0 = t_row * heads + static_cast<long long>(blockIdx.y) * G;
    attend(
        q_nope + (t_row * heads + h) * ckv, q_pe + (t_row * heads + h) * kpe, ckv_pages, kpe_pages, ckv, kpe,
        keys, sm_scale * 1.4426950408889634f, G, naive_smem,
        [&](int g, int d, float v) { o[(head0 + g) * ckv + d] = from_f32<bf16>(v); },
        [&](int g, float v) {
            if (lse != nullptr) lse[head0 + g] = v;
        });
}

}  // namespace mla_naive

// ---------------------------------------------------------------- mfma --
//
// The latent-512 walk on the matrix core: eight wavefronts own 64 query
// rows and walk their keys 32 a tile through LDS, two tiles in flight in
// registers. Sᵀ = K·Qᵀ splits the key width between a row tile's two
// wavefronts, which trade half-width scores through LDS and run the same
// online softmax; Oᵀ += Vᵀ·Pᵀ gives wavefront w latent 64w... Lane l holds
// Sᵀ (key 4(l/16)+t, row l%16), so a row's rescale is per lane column. The
// keys' cache slots sit in an LDS table filled ahead of the walk.

namespace mla {

constexpr int kRank = 512;
constexpr int kBM = 64;
constexpr int kBK = 32;
// The fewest keys a split of the gathered walk takes.
constexpr int kMinShare = 64;
constexpr int kWaves = 8;
constexpr int kThreads = kWaves * kWave;
constexpr int kNoKey = 0x7fffffff;
constexpr int kTableTiles = 8;

// Sixteen bytes in registers (a vector, so the prefetch stays out of scratch).
typedef u32 chunk __attribute__((ext_vector_type(4)));

// RK the latent's width, BK the keys a tile: 32 at the 512 latent, 16 at
// a 1024 one (whose key rows would not fit LDS 32 at a time).
template <int R, int RK = kRank, int BK = kBK>
struct Smem {
    static constexpr int W = RK + R;     // a key row: latent, then rope
    static constexpr int LD = W + 8;     // its LDS pitch (16-byte rows, banks staggered)
    static constexpr int LP = BK + 4;    // the weights' pitch
    static constexpr int TABLE = kTableTiles * BK;
    bf16 kv[BK][LD];
    bf16 p[kBM][LP];
    mfma_cd part[kWaves][2][kWave];
    i32 slot[2][TABLE];
    i32 kpos[2][TABLE];
    float rm[kBM];
    float rl[kBM];
};

// One tile of a key row in registers: four latent chunks and one rope
// chunk a thread (sixteen threads a key; the upper eight repeat the lower
// eight's rope chunk).
struct Staged {
    chunk lat[4];
    chunk rp;
};

// The block's 64 query rows: `qn(r)`/`qp(r)` their latent and rope query
// rows, `limit(r)` the last key position row r sees (−1: a dead row), and
// `put(r, j, v)` / `put_lse(r, lse)` land 4 output columns from j / the lse.
// `attend` walks the keys and lands every live row: every column of the
// scores, and the 512 output columns from `voff` (a 1024 latent's output
// is two blocks' halves, each scoring the whole row).
template <int R, class Rows, int RK = kRank, int BK = kBK>
__device__ void attend(Smem<R, RK, BK>& sm, const Rows& rows, const LatentKeys& keys,
                       const bf16* __restrict__ ckv, const bf16* __restrict__ kpe, float scale_log2,
                       int voff = 0) {
    using SM = Smem<R, RK, BK>;
    constexpr int ST = SM::W / 32;       // 32-wide k pairs over the key row
    constexpr int SH = ST / 2;           // a wavefront's half
    constexpr int KT = BK / 16;          // 16-key halves a tile
    constexpr int TPK = kThreads / BK;   // threads a key row's read
    static_assert(ST % 2 == 0 && TPK * 32 == RK, "a key row split evenly");
    const int w = threadIdx.x / kWave, l = lane(), c = l % 16, g = l / 16;
    const int rt = w % 4, dh = w / 4;
    const int qrow = 16 * rt + c;

    // Qᵀ as B operands: this lane's column is query row qrow, its k pairs
    // the wavefront's half of the key row.
    mfma_ab2 qf[SH];
    {
        const bf16* qn = rows.qn(qrow);
        const bf16* qp = rows.qp(qrow);
#pragma unroll
        for (int j = 0; j < SH; ++j) {
            const int st = SH * dh + j;
            qf[j] = load8(st < RK / 32 ? qn + 32 * st + 8 * g : qp + 32 * (st - RK / 32) + 8 * g);
        }
    }
    const int lim = rows.limit(qrow);

    const int count = keys.n1 - keys.n0;
    const int tiles = keys.limit < 0 || count <= 0 ? 0 : (count + BK - 1) / BK;

    // Table half h holds tiles [8k, 8k + 8) for k ≡ h (mod 2).
    auto fill = [&](int tile0) {
        const int h = (tile0 / kTableTiles) % 2;
        for (int e = threadIdx.x; e < SM::TABLE; e += kThreads) {
            const int n = keys.n0 + tile0 * BK + e;
            const int p = keys.pos(min(n, keys.n1 - 1));
            const bool vis = n < keys.n1 && keys.visible(p);
            const int at = vis ? p : 0;
            sm.kpos[h][e] = vis ? p : kNoKey;
            sm.slot[h][e] = static_cast<i32>(keys.slot(at, keys.page(at)));
        }
    };
    const int key = threadIdx.x / TPK, part = threadIdx.x % TPK;
    auto issue = [&](Staged& s, int tile) {
        tile = min(tile, tiles - 1);  // past the walk: the last tile again, from L2
        const long long slot = sm.slot[(tile / kTableTiles) % 2][(tile % kTableTiles) * BK + key];
        const bf16* src = ckv + slot * RK;
#pragma unroll
        for (int i = 0; i < 4; ++i) s.lat[i] = *reinterpret_cast<const chunk*>(src + 8 * (part + TPK * i));
        if constexpr (R > 0) s.rp = *reinterpret_cast<const chunk*>(kpe + slot * R + 8 * (part % 8));
    };
    auto store = [&](const Staged& s) {
        bf16* dst = &sm.kv[key][0];
#pragma unroll
        for (int j = 0; j < 4; ++j) *reinterpret_cast<chunk*>(dst + 8 * (part + TPK * j)) = s.lat[j];
        if constexpr (R > 0) *reinterpret_cast<chunk*>(dst + RK + 8 * (part % 8)) = s.rp;
    };

    float m_run = neg_inf(), l_run = 0.f;
    mfma_cd acc[4][4];
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) acc[mt][nt] = mfma_zero();

    auto step = [&](Staged& buf, int i) {
        lds_barrier();
        store(buf);
        const int ahead = i + 2;
        if (ahead < tiles && ahead % kTableTiles == 0) {
            fill(ahead);
            lds_barrier();
        }
        issue(buf, ahead);
        lds_barrier();

        // This wavefront's half-width Sᵀ over the tile's two 16-key halves.
        mfma_cd s0 = mfma_zero(), s1 = mfma_zero();
        {
            const bf16* k0 = &sm.kv[c][8 * g];
            const bf16* k1 = &sm.kv[KT > 1 ? 16 + c : c][8 * g];
#pragma unroll
            for (int j = 0; j < SH; ++j) {
                const int st = SH * dh + j;
                s0 = mfma16x2(load8(k0 + 32 * st), qf[j], s0);
                if constexpr (KT > 1) s1 = mfma16x2(load8(k1 + 32 * st), qf[j], s1);
            }
        }
        sm.part[w][0][l] = s0;
        if constexpr (KT > 1) sm.part[w][1][l] = s1;
        lds_barrier();
        const mfma_cd o0 = sm.part[w ^ 4][0][l];
        const mfma_cd o1 = KT > 1 ? sm.part[w ^ 4][1][l] : mfma_zero();
        const i32* kp = &sm.kpos[(i / kTableTiles) % 2][(i % kTableTiles) * BK + 4 * g];
        const int4 kp0 = *reinterpret_cast<const int4*>(kp);
        const int4 kp1 = KT > 1 ? *reinterpret_cast<const int4*>(kp + 16) : int4{kNoKey, kNoKey, kNoKey, kNoKey};
        const int ks[8] = {kp0.x, kp0.y, kp0.z, kp0.w, kp1.x, kp1.y, kp1.z, kp1.w};
        float x[8];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            x[t] = ks[t] <= lim ? (s0[t] + o0[t]) * scale_log2 : neg_inf();
            x[4 + t] = KT > 1 && ks[4 + t] <= lim ? (s1[t] + o1[t]) * scale_log2 : neg_inf();
        }
        float mx = x[0];
#pragma unroll
        for (int t = 1; t < 8; ++t) mx = fmaxf(mx, x[t]);
        mx = fmaxf(mx, __shfl_xor(mx, 16, kWave));
        mx = fmaxf(mx, __shfl_xor(mx, 32, kWave));
        // A row that has seen no key yet keeps m = −inf; it exponentiates
        // against 0 instead, so its weights are 0 and its rescale is 0 (of
        // an accumulator that is still 0).
        const float m_new = fmaxf(m_run, mx);
        const float m_use = m_new == neg_inf() ? 0.f : m_new;
        const float alpha = __builtin_amdgcn_exp2f(m_run - m_use);
        float e[8], sum = 0.f;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            e[t] = __builtin_amdgcn_exp2f(x[t] - m_use);
            sum += e[t];
        }
        l_run = l_run * alpha + sum;
        m_run = m_new;
        // (a 16-key tile's weights are one half, which both waves hold)
        if (KT > 1 || dh == 0)
            *reinterpret_cast<mfma_ab*>(&sm.p[qrow][16 * dh + 4 * g]) =
                dh == 0 ? pack4(e[0], e[1], e[2], e[3]) : pack4(e[4], e[5], e[6], e[7]);
        if (dh == 0 && g == 0) sm.rm[qrow] = alpha;
        lds_barrier();

        // Oᵀ += Vᵀ · Pᵀ over this wavefront's 64 latent columns.
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const float a = sm.rm[16 * nt + c];
#pragma unroll
            for (int mt = 0; mt < 4; ++mt)
#pragma unroll
                for (int t = 0; t < 4; ++t) acc[mt][nt][t] *= a;
        }
#pragma unroll
        for (int kk = 0; kk < KT; ++kk) {
            mfma_ab b[4];
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
                b[nt] = *reinterpret_cast<const mfma_ab*>(&sm.p[16 * nt + c][16 * kk + 4 * g]);
#pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                const int col = voff + 64 * w + 16 * mt + c;
                const int r0 = 16 * kk + 4 * g;
                mfma_ab a;
#pragma unroll
                for (int t = 0; t < 4; ++t) a[t] = __builtin_bit_cast(short, sm.kv[r0 + t][col]);
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) acc[mt][nt] = mfma16(a, b[nt], acc[mt][nt]);
            }
        }
    };

    if (tiles > 0) {
        Staged a, b;
        fill(0);
        lds_barrier();
        issue(a, 0);
        issue(b, 1);
        // Pairs, so the loop's back edge always leaves b's reads the
        // younger: the wait for a's then lets b's stay in flight.
        int i = 0;
        for (; i + 1 < tiles; i += 2) {
            step(a, i);
            step(b, i + 1);
        }
        if (i < tiles) step(a, i);
    }

    // A row's sum over its four lane quads (both halves hold all its keys).
    l_run += __shfl_xor(l_run, 16, kWave);
    l_run += __shfl_xor(l_run, 32, kWave);
    lds_barrier();
    if (dh == 0 && g == 0) {
        sm.rm[qrow] = m_run;
        sm.rl[qrow] = l_run;
    }
    lds_barrier();
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
        const int row = 16 * nt + c;
        const float sum = sm.rl[row];
        const float inv = sum > 0.f ? 1.f / sum : 0.f;
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
            const float v[4] = {acc[mt][nt][0] * inv, acc[mt][nt][1] * inv, acc[mt][nt][2] * inv,
                                acc[mt][nt][3] * inv};
            rows.put(row, voff + 64 * w + 16 * mt + 4 * g, v);
        }
        if (w == 0 && g == 0 && voff == 0) rows.put_lse(row, sum > 0.f ? sm.rm[row] + log2f(sum) : neg_inf());
    }
    lds_barrier();
}

__device__ __forceinline__ void store4(bf16* dst, const float (&v)[4]) {
    *reinterpret_cast<mfma_ab*>(dst) = pack4(v[0], v[1], v[2], v[3]);
}
__device__ __forceinline__ void store4(float* dst, const float (&v)[4]) {
    *reinterpret_cast<float4*>(dst) = make_float4(v[0], v[1], v[2], v[3]);
}

// ---------------------------------------------------------- planned walk --

// Mirrors `attn::mla::Work` field for field; passed by value. The work list
// is the MLA scheduler's (attn/sched_mla.rs): cluster y's items run on the
// blocks (·, y), block x taking packed rows q_start + 64x of each item.
struct Work {
    const bf16* q_nope;
    const bf16* q_pe;
    const bf16* ckv;
    const bf16* kpe;
    const i32* kv_indices;
    const i32* q_indptr;
    const i32* kv_indptr;
    const i32* partial_indptr;
    const i32* q_len;
    const i32* kv_len;
    const i32* q_start;
    const i32* kv_start;
    const i32* kv_end;
    const i32* work_indptr;
    bf16* o;
    float* lse;
    bf16* partial_o;
    float* partial_lse;
    i32 num_heads;
    i32 page_size;
    float scale_log2;
    i32 causal;
};

// Packed row r of a work item is (token (base + r) / H, head (base + r) % H)
// of the request; it lands in o, or in the item's partial rows.
template <int R>
struct PlanRows {
    const bf16* q_nope;
    const bf16* q_pe;
    bf16* o;
    float* lse;
    bf16* part_o;
    float* part_lse;
    int heads, base, q_len, kv_len, token0;
    bool causal;

    __device__ __forceinline__ int token(int r) const { return (base + r) / heads; }
    __device__ __forceinline__ long long qrow(int r) const {
        const int at = base + r;
        return static_cast<long long>(token0 + min(at / heads, q_len - 1)) * heads + at % heads;
    }
    __device__ __forceinline__ const bf16* qn(int r) const { return q_nope + qrow(r) * kRank; }
    __device__ __forceinline__ const bf16* qp(int r) const { return q_pe + qrow(r) * R; }
    __device__ __forceinline__ int limit(int r) const {
        const int t = token(r);
        if (t >= q_len) return -1;
        return causal ? kv_len - q_len + t : kv_len - 1;
    }
    __device__ __forceinline__ void put(int r, int j, const float (&v)[4]) const {
        if (token(r) >= q_len) return;
        if (part_o != nullptr) store4(part_o + static_cast<long long>(r) * kRank + j, v);
        else store4(o + qrow(r) * kRank + j, v);
    }
    __device__ __forceinline__ void put_lse(int r, float v) const {
        if (token(r) >= q_len) return;
        if (part_lse != nullptr) part_lse[r] = v;
        else if (lse != nullptr) lse[qrow(r)] = v;
    }
};

template <int R>
__global__ __launch_bounds__(kThreads) void mla_paged(const Work p) {
    __shared__ Smem<R> sm;
    const int heads = p.num_heads;
    for (int item = p.work_indptr[blockIdx.y]; item < p.work_indptr[blockIdx.y + 1]; ++item) {
        const int q_len = p.q_len[item], kv_len = p.kv_len[item];
        const int base = p.q_start[item] + static_cast<int>(blockIdx.x) * kBM;
        if (base / heads >= q_len) continue;
        const int kv_start = p.kv_start[item];
        int kv_end = p.kv_end[item];
        if (p.causal) kv_end = min(kv_end, kv_len - q_len + min(q_len - 1, (base + kBM - 1) / heads) + 1);
        const int partial = p.partial_indptr[item];
        const long long part_row = static_cast<long long>(partial) + static_cast<long long>(blockIdx.x) * kBM;
        PlanRows<R> rows{p.q_nope,
                         p.q_pe,
                         p.o,
                         p.lse,
                         partial < 0 ? nullptr : p.partial_o + part_row * kRank,
                         partial < 0 ? nullptr : p.partial_lse + part_row,
                         heads,
                         base,
                         q_len,
                         kv_len,
                         p.q_indptr[item],
                         p.causal != 0};
        const LatentKeys keys{nullptr, p.kv_indices + p.kv_indptr[item], p.page_size, 1,
                              kv_start, max(kv_end, kv_start), max(kv_end, kv_start) - 1};
        attend<R>(sm, rows, keys, p.ckv, p.kpe, p.scale_log2);
    }
}

// A running fold of normalized partial rows, each weighted by 2^lse: eight
// columns a lane.
struct Fold {
    float m = neg_inf(), d = 0.f, o[8] = {};

    __device__ __forceinline__ void add(float s, float w, const float (&v)[8]) {
        if (s == neg_inf()) return;
        const float m_new = fmaxf(m, s);
        const float a = exp2f(m - m_new), f = w * exp2f(s - m_new);
        d = d * a + f;
#pragma unroll
        for (int j = 0; j < 8; ++j) o[j] = o[j] * a + f * v[j];
        m = m_new;
    }
    __device__ __forceinline__ void add(const Fold& x) {
        float v[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = x.d > 0.f ? x.o[j] / x.d : 0.f;
        add(x.m, x.d, v);
    }
    // Partials k0, k0 + step, ... below end, eight in flight: `load(k)`
    // reads partial k's raw columns, `widen(raw, v)` turns them into eight
    // floats once the whole group has arrived (the empty asm takes them all,
    // so no read is issued just before its use).
    template <class Raw, class Load, class Widen>
    __device__ __forceinline__ void walk(const float* lse, int k0, int step, int end, Load&& load, Widen&& widen) {
        constexpr int N = 8;
        for (int k = k0; k < end; k += N * step) {
            float s[N];
            Raw raw[N];
#pragma unroll
            for (int j = 0; j < N; ++j) {
                const int kj = k + j * step;
                const int at = kj < end ? kj : k;
                s[j] = lse[at];
                // An empty partial (lse −∞) is skipped unread.
                raw[j] = s[j] == neg_inf() ? Raw{} : load(at);
            }
#pragma unroll
            for (int j = 0; j < N; ++j) asm volatile("" ::"v"(raw[j]));
#pragma unroll
            for (int j = 0; j < N; ++j) {
                float v[8];
                widen(raw[j], v);
                add(k + j * step < end ? s[j] : neg_inf(), 1.f, v);
            }
        }
    }
    __device__ __forceinline__ float lse() const { return d > 0.f ? m + log2f(d) : neg_inf(); }
};

// Mirrors `attn::mla::Merge`. Merge slot b folds the split rows it owns:
// packed rows [start, end) of the batch, partial row k·stride + i of
// [pstart, pend) for local row i. 64 threads a row, 8 columns each; a slot
// of fewer than four rows splits each row's partials over its wavefronts.
struct Merge {
    const i32* start;
    const i32* end;
    const i32* pstart;
    const i32* pend;
    const i32* stride;
    const bf16* partial_o;
    const float* partial_lse;
    bf16* o;
    float* lse;
    i32 num_heads;
};

__global__ __launch_bounds__(256) void mla_merge(const Merge p) {
    __shared__ float fo[4][kWave][8];
    __shared__ float fm[4], fd[4];
    const int b = blockIdx.x;
    const int start = p.start[b], len = p.end[b] - start;
    const int pstart = p.pstart[b], pend = p.pend[b], stride = p.stride[b];
    const int wv = threadIdx.x / kWave, col = 8 * lane();
    auto load = [&](int k) { return *reinterpret_cast<const chunk*>(p.partial_o + static_cast<long long>(k) * kRank + col); };
    auto widen = [](const chunk& x, float (&v)[8]) {
        const Pack8<bf16> h = __builtin_bit_cast(Pack8<bf16>, x);
#pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = to_f32(h.v[j]);
    };
    auto land = [&](int i, const Fold& f) {
        const long long row = start + i;
        const float inv = f.d > 0.f ? 1.f / f.d : 0.f;
        Pack8<bf16> out;
#pragma unroll
        for (int j = 0; j < 8; ++j) out.v[j] = from_f32<bf16>(f.o[j] * inv);
        *reinterpret_cast<Pack8<bf16>*>(p.o + row * kRank + col) = out;
        if (p.lse != nullptr && col == 0) p.lse[row] = f.lse();
    };
    if (len >= 4) {
        for (int i = wv; i < len; i += 4) {
            Fold f;
            f.walk<chunk>(p.partial_lse, pstart + i, stride, pend, load, widen);
            land(i, f);
        }
        return;
    }
    for (int i = 0; i < len; ++i) {
        Fold f;
        f.walk<chunk>(p.partial_lse, pstart + i + wv * stride, 4 * stride, pend, load, widen);
#pragma unroll
        for (int j = 0; j < 8; ++j) fo[wv][lane()][j] = f.o[j];
        if (lane() == 0) {
            fm[wv] = f.m;
            fd[wv] = f.d;
        }
        __syncthreads();
        if (wv == 0) {
            Fold all;
            for (int w = 0; w < 4; ++w) {
                Fold x;
                x.m = fm[w];
                x.d = fd[w];
#pragma unroll
                for (int j = 0; j < 8; ++j) x.o[j] = fo[w][lane()][j];
                all.add(x);
            }
            land(i, all);
        }
        __syncthreads();
    }
}

// --------------------------------------------------------- gathered walk --

// Mirrors `attn::mla::Gather` (by value). A block a (64 heads, token,
// split): the token's keys are its selection row or every visible position;
// several splits land f32 partials `latent_merge` folds. Visibility is the
// request's (causal or not) or, with `positions`, a compressed pool's: a
// token at q sees the (q + 1) / ratio entries before it.
struct Gather {
    const bf16* q_nope;
    const bf16* q_pe;
    const bf16* ckv;
    const bf16* kpe;
    const i32* page_indices;
    const i32* page_indptr;
    const i32* last_page_lens;
    const i32* qo_indptr;
    const i32* positions;
    const i32* req_of_token;
    const i32* selection;
    bf16* o;
    float* lse;
    float* part_o;
    float* part_lse;
    const u32* win;
    i32 top_k;
    i32 num_requests;
    i32 num_heads;
    i32 page_size;
    i32 ratio;
    i32 splits;
    i32 causal;
    float scale_log2;
    // Past 0, the last `window` positions through the row's own (a
    // request-ordered walk only).
    i32 window;
    // A compressed pool keeps its entries compact, a row each.
    i32 compact;
};

template <int R, int RK = kRank>
struct GatherRows {
    const bf16* q_nope;
    const bf16* q_pe;
    bf16* o;
    float* lse;
    float* part_o;
    float* part_lse;
    long long t_row, t;
    int heads, h0, last, splits, split;

    __device__ __forceinline__ long long qrow(int r) const { return t_row * heads + min(h0 + r, heads - 1); }
    __device__ __forceinline__ long long prow(int r) const {
        return (t * heads + h0 + r) * splits + split;
    }
    __device__ __forceinline__ const bf16* qn(int r) const { return q_nope + qrow(r) * RK; }
    __device__ __forceinline__ const bf16* qp(int r) const { return q_pe + qrow(r) * R; }
    __device__ __forceinline__ int limit(int r) const { return h0 + r < heads ? last : -1; }
    __device__ __forceinline__ void put(int r, int j, const float (&v)[4]) const {
        if (h0 + r >= heads) return;
        if (splits > 1) store4(part_o + prow(r) * RK + j, v);
        else store4(o + qrow(r) * RK + j, v);
    }
    __device__ __forceinline__ void put_lse(int r, float v) const {
        if (h0 + r >= heads) return;
        if (splits > 1) part_lse[prow(r)] = v;
        else if (lse != nullptr) lse[qrow(r)] = v;
    }
};

template <int R, int RK = kRank>
__global__ __launch_bounds__(kThreads) void latent_gather(const Gather p) {
    constexpr int BK = RK > kRank ? 16 : kBK;
    constexpr int HALVES = RK / kRank;  // blocks that split a row's output columns
    __shared__ Smem<R, RK, BK> sm;
    const int split = static_cast<int>(blockIdx.z) / HALVES;
    const int voff = (static_cast<int>(blockIdx.z) % HALVES) * kRank;
    const Window window{p.win};
    const int t = blockIdx.y;
    if (!window.live(t)) return;
    const long long t_row = window.plane(t);

    const i32* pages;
    int limit;
    int ratio = 1;
    int floor = 0;
    if (p.positions != nullptr) {
        const int req = p.req_of_token[t_row];
        pages = p.page_indices + p.page_indptr[req];
        ratio = p.ratio;
        limit = (p.positions[t_row] + 1) / ratio - 1;
    } else {
        const int requests = window.lanes(p.num_requests);
        const int r = request_of(p.qo_indptr, requests, t);
        const int kv_len = paged_len(p.page_indptr, p.last_page_lens, r, p.page_size);
        const int abs_q = kv_len - (p.qo_indptr[r + 1] - p.qo_indptr[r]) + (t - p.qo_indptr[r]);
        pages = p.page_indices + p.page_indptr[r];
        limit = (p.causal ? min(abs_q + 1, kv_len) : kv_len) - 1;
        if (p.window > 0) floor = max(0, abs_q + 1 - p.window);
    }
    const i32* sel = p.selection != nullptr ? p.selection + t_row * p.top_k : nullptr;
    // Without a selection the walk's list is the visible positions from the
    // window's floor.
    const int first = sel != nullptr ? 0 : floor;
    // A selection lists its picks first, ascending, −1 past them
    // (`index_topk`): a row that sees `limit + 1` positions has at most that
    // many, so the walk stops there rather than at the budget.
    const int count = sel != nullptr ? min(p.top_k, max(limit + 1, 0)) : max(limit + 1 - floor, 0);
    // A split takes at least kMinShare keys: past the row's last, a split
    // folds nothing and says so with an empty lse alone (the merge reads no
    // more of it), so a short row costs a few splits, not all of them.
    const int share = max(((count + p.splits - 1) / p.splits + BK - 1) / BK * BK, kMinShare);
    const int n0 = min(count, split * share);
    if (split > 0 && n0 >= count) {
        const int h0 = static_cast<int>(blockIdx.x) * kBM;
        if (voff == 0)
            for (int r = threadIdx.x; r < kBM && h0 + r < p.num_heads; r += kThreads)
                p.part_lse[(t * p.num_heads + h0 + r) * p.splits + split] = neg_inf();
        return;
    }
    const bool compact = p.positions != nullptr && p.compact;
    const LatentKeys keys{sel,     pages, compact ? p.page_size / ratio : p.page_size, compact ? 1 : ratio,
                          first + n0, first + min(count, n0 + share), limit, floor};
    const GatherRows<R, RK> rows{p.q_nope, p.q_pe, p.o, p.lse, p.part_o, p.part_lse, t_row, t,
                                 p.num_heads, static_cast<int>(blockIdx.x) * kBM, limit, p.splits, split};
    attend<R, GatherRows<R, RK>, RK, BK>(sm, rows, keys, p.ckv, p.kpe, p.scale_log2, voff);
}

// The splits of row (t, h) folded into o (and lse). One wavefront a row,
// eight columns a lane.
template <int RK>
__global__ __launch_bounds__(kWave) void latent_merge(const float* __restrict__ part_o,
                                                      const float* __restrict__ part_lse, bf16* __restrict__ o,
                                                      float* __restrict__ lse, int heads, int splits,
                                                      const u32* __restrict__ win) {
    const Window window{win};
    const int t = blockIdx.x;
    if (!window.live(t)) return;
    const long long row = window.plane(t) * heads + blockIdx.y;
    const long long prow = (static_cast<long long>(t) * heads + blockIdx.y) * splits;
    for (int c0 = 0; c0 < RK; c0 += 8 * kWave) {
    const int col = c0 + 8 * lane();
    Fold f;
    typedef float f32x8 __attribute__((ext_vector_type(8)));
    f.walk<f32x8>(
        part_lse + prow, 0, 1, splits,
        [&](int s) { return *reinterpret_cast<const f32x8*>(part_o + (prow + s) * RK + col); },
        [](const f32x8& x, float (&v)[8]) {
#pragma unroll
            for (int j = 0; j < 8; ++j) v[j] = x[j];
        });
    const float inv = f.d > 0.f ? 1.f / f.d : 0.f;
    Pack8<bf16> out;
#pragma unroll
    for (int j = 0; j < 8; ++j) out.v[j] = from_f32<bf16>(f.o[j] * inv);
    *reinterpret_cast<Pack8<bf16>*>(o + row * RK + col) = out;
    if (lse != nullptr && col == 0) lse[row] = f.lse();
    }
}

}  // namespace mla

}  // namespace pie::attn
