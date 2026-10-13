#pragma once

// Paged GQA attention over the key blocks a per-row selection names (DSA /
// QSA sparse attention), plus the open block that runs up to the query:
//
//   keys of row t = ⋃_{c ∈ selection[t], 0 ≤ c < ⌊j_end / ratio⌋} [c·ratio, (c+1)·ratio)
//                 ∪ [⌊j_end / ratio⌋·ratio, j_end)
//
// with j_end the row's causal end. A block below ⌊j_end / ratio⌋ holds only
// keys before j_end, so rejecting a block id is the whole causal test.
//
// One block per (token row, group of G query heads): 256 threads as eight
// 32-lane groups, K = 8 / G of them per head walking its keys round-robin
// with an online softmax, folded through LDS at the end. A 32-lane group
// (half a wavefront) dots one key, D / 32 elements a lane, so a wavefront
// reads two keys at once.

#include "prelude/device.hip"
#include "prelude/mfma.hip"

namespace pie::attn::sdpa_selected {

constexpr int kBlock = 256;
constexpr int kLanes = 32;
constexpr int kGroups = kBlock / kLanes;
constexpr int kMaxPer = 16;

template <class T>
__global__ __launch_bounds__(kBlock) void sdpa_paged_selected_kernel(
    const T* __restrict__ queries, const T* __restrict__ k_pages, const T* __restrict__ v_pages,
    const u32* __restrict__ qo_indptr, const u32* __restrict__ page_indices,
    const u32* __restrict__ page_indptr, const u32* __restrict__ last_page_lens,
    const i32* __restrict__ selection, T* __restrict__ o, i32 R, i32 H, i32 KVH, i32 D,
    i32 page_size, float sm_scale, i32 top_k, i32 ratio, i32 G, const u32* __restrict__ win) {
    const Window window{win};
    const int t = static_cast<int>(blockIdx.x);
    if (!window.live(t)) return;
    const long long t_row = window.plane(t);
    R = window.lanes(R);

    const int sub = static_cast<int>(threadIdx.x) % kLanes;
    const int group = static_cast<int>(threadIdx.x) / kLanes;
    const int K = kGroups / G;
    const int g = group / K;
    const int s = group % K;
    const int h = static_cast<int>(blockIdx.y) * G + g;
    const int kv_head = h / (H / KVH);
    const int per = D / kLanes;

    // The request of launch row t.
    int lo = 0, hi = R - 1;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (t < static_cast<int>(qo_indptr[mid + 1])) hi = mid;
        else lo = mid + 1;
    }
    const int qo_lo = static_cast<int>(qo_indptr[lo]);
    const int fresh = static_cast<int>(qo_indptr[lo + 1]) - qo_lo;
    const int pages_first = static_cast<int>(page_indptr[lo]);
    const int pages = static_cast<int>(page_indptr[lo + 1]) - pages_first;
    const int kv_len = (pages - 1) * page_size + static_cast<int>(last_page_lens[lo]);
    const int j_end = kv_len - fresh + (t - qo_lo) + 1;

    const int nblocks = ratio > 0 ? j_end / ratio : 0;
    const int blocks = min(top_k, nblocks);
    const i32* srow = selection + t_row * top_k;

    extern __shared__ float smem[];
    float* gacc = smem;
    float* gm = gacc + kGroups * D;
    float* gl = gm + kGroups;

    const T* qrow = queries + (t_row * H + h) * D;
    float qreg[kMaxPer], acc[kMaxPer];
#pragma unroll
    for (int i = 0; i < kMaxPer; ++i) {
        qreg[i] = i < per ? to_f32(qrow[sub + i * kLanes]) * sm_scale : 0.f;
        acc[i] = 0.f;
    }
    float m = neg_inf(), l = 0.f;

    const int steps = blocks * ratio + (ratio - 1);
    for (int n = s; n < steps; n += K) {
        int j;
        if (n < blocks * ratio) {
            const int c = srow[n / ratio];
            if (c < 0 || c >= nblocks) continue;
            j = c * ratio + n % ratio;
        } else {
            j = nblocks * ratio + (n - blocks * ratio);
            if (j >= j_end) continue;
        }
        const long long slot = static_cast<long long>(page_indices[pages_first + j / page_size]) * page_size + j % page_size;
        const T* kj = k_pages + (slot * KVH + kv_head) * D;
        const T* vj = v_pages + (slot * KVH + kv_head) * D;

        float pd = 0.f;
#pragma unroll
        for (int i = 0; i < kMaxPer; ++i)
            if (i < per) pd += qreg[i] * to_f32(kj[sub + i * kLanes]);
#pragma unroll
        for (int off = kLanes / 2; off > 0; off >>= 1) pd += __shfl_xor(pd, off, kLanes);
        const float m_new = fmaxf(m, pd);
        const float corr = __expf(m - m_new);
        const float p = __expf(pd - m_new);
        l = l * corr + p;
#pragma unroll
        for (int i = 0; i < kMaxPer; ++i)
            if (i < per) acc[i] = acc[i] * corr + p * to_f32(vj[sub + i * kLanes]);
        m = m_new;
    }

#pragma unroll
    for (int i = 0; i < kMaxPer; ++i)
        if (i < per) gacc[group * D + sub + i * kLanes] = acc[i];
    if (sub == 0) {
        gm[group] = m;
        gl[group] = l;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < G * D; idx += kBlock) {
        const int gg = idx / D;
        const int d = idx % D;
        const int first = gg * K;
        float m_all = neg_inf();
        for (int w = first; w < first + K; ++w) m_all = fmaxf(m_all, gm[w]);
        float l_all = 0.f, v = 0.f;
        for (int w = first; w < first + K; ++w)
            if (gm[w] > neg_inf()) {
                const float e = __expf(gm[w] - m_all);
                l_all += gl[w] * e;
                v += gacc[w * D + d] * e;
            }
        const float inv = l_all > 0.f ? 1.f / l_all : 0.f;
        o[(t_row * H + static_cast<int>(blockIdx.y) * G + gg) * D + d] = from_f32<T>(v * inv);
    }
}

// The selected attention on the matrix core, for GQA groups of up to 16
// heads at D = 64, 128 or 256: one block per (token row, kv head, split of
// the row's entries), so the group's heads share every key and value read.
// The row's keys are listed into LDS a window at a time: the selection
// entries (and the open block, as one more entry of j_end mod ratio keys)
// are scanned for the keys each contributes and their pool slots written
// densely, so a refused entry costs no round. The rounds over a window are
// `attn/paged.hip`'s `paged_decode_mfma`: per wavefront 16 keys, Sᵀ = K·Qᵀ
// on the matrix core, the weights through LDS to the value side, vector
// value reads, and the wavefronts folded through LDS at the end. Past the
// window's keys a lane rereads its last one, which weighs zero.
//
// Four wavefronts a block: at a 12-head group the round's reads and sums
// need more than the 256 registers two wavefronts a SIMD leave (eight
// spilled to scratch and ran 30% slower). Rows too few to fill the device
// split their entries over gridDim.z blocks, `sdpa_selected_merge` folding
// the splits.
constexpr int kSelWaves = 4;
constexpr int kSelThreads = kSelWaves * kWave;
constexpr int kSelTable = 1024;

// The most heads whose wavefront outputs fold through LDS at once: a
// divisor of G within 16 KiB.
__host__ __device__ constexpr int fold_heads(int G, int D, int waves) {
    for (int h = G; h > 1; --h)
        if (G % h == 0 && waves * h * D * 4 <= 16 * 1024) return h;
    return 1;
}

// The block's exclusive prefix of `v`, and in `total` its sum; every thread
// calls it. `wsum` holds a word a wavefront.
__device__ __forceinline__ int block_exclusive_scan(int v, int* wsum, int& total) {
    const int l = lane(), w = static_cast<int>(threadIdx.x) / kWave;
    int inc = v;
#pragma unroll
    for (int off = 1; off < kWave; off <<= 1) {
        const int n = __shfl_up(inc, off, kWave);
        if (l >= off) inc += n;
    }
    __syncthreads();
    if (l == kWave - 1) wsum[w] = inc;
    __syncthreads();
    int before = 0;
    total = 0;
    for (int i = 0; i < kSelWaves; ++i) {
        const int s = wsum[i];
        if (i < w) before += s;
        total += s;
    }
    return before + inc - v;
}

// N consecutive bf16 of a value row, read as one access.
template <int N>
struct alignas(2 * N) ValuePack {
    bf16 v[N];
};

template <int D, int G>
__global__ __launch_bounds__(kSelThreads) void sdpa_selected_mfma(
    const bf16* __restrict__ queries, const bf16* __restrict__ k_pages,
    const bf16* __restrict__ v_pages, const u32* __restrict__ qo_indptr,
    const u32* __restrict__ page_indices, const u32* __restrict__ page_indptr,
    const u32* __restrict__ last_page_lens, const i32* __restrict__ selection,
    bf16* __restrict__ o, float* __restrict__ part, i32 R, i32 H, i32 KVH, i32 page_size,
    float scale_log2, i32 top_k, i32 ratio, const u32* __restrict__ win) {
    constexpr int WAVES = kSelWaves, SUB = 16, KP = D / 32;
    // Columns a lane accumulates: four in a wide group, whose eight would not
    // fit the registers beside the round's reads.
    constexpr int COLS = G > 8 ? 4 : 8;
    constexpr int LPK = D / COLS;     // lanes a key's value row spans
    constexpr int KPI = kWave / LPK;  // keys one value read covers
    constexpr int VR = SUB / KPI;     // value reads a round
    static_assert(G <= 16 && D % 32 == 0 && LPK <= kWave && kWave % LPK == 0, "a group within one tile");
    constexpr int HG = fold_heads(G, D, WAVES);
    __shared__ float ps[WAVES][SUB][G];
    __shared__ float alph[WAVES][G];
    __shared__ float part_m[WAVES][G], part_l[WAVES][G];
    __shared__ float part_o[WAVES][HG][D];
    __shared__ i32 table[kSelTable];
    __shared__ int wsum[WAVES];

    const Window window{win};
    const int t = static_cast<int>(blockIdx.x);
    if (!window.live(t)) return;
    const long long t_row = window.plane(t);
    R = window.lanes(R);
    const int kv_head = static_cast<int>(blockIdx.y);
    const int wave = static_cast<int>(threadIdx.x) / kWave, l = lane();
    const int g = l % 16, quad = l / 16;

    // The request of launch row t.
    int lo = 0, hi = R - 1;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (t < static_cast<int>(qo_indptr[mid + 1])) hi = mid;
        else lo = mid + 1;
    }
    const int qo_lo = static_cast<int>(qo_indptr[lo]);
    const int fresh = static_cast<int>(qo_indptr[lo + 1]) - qo_lo;
    const int pages_first = static_cast<int>(page_indptr[lo]);
    const int pages = static_cast<int>(page_indptr[lo + 1]) - pages_first;
    const int kv_len = (pages - 1) * page_size + static_cast<int>(last_page_lens[lo]);
    const int j_end = kv_len - fresh + (t - qo_lo) + 1;
    const int nblocks = j_end / ratio;
    const int blocks = min(top_k, nblocks);
    const int entries = blocks + 1;  // the open block last
    const i32* srow = selection + t_row * top_k;
    const u32* pidx = page_indices + pages_first;

    // Qᵀ as B operands (this lane's column is head g, k pairs 8·quad), in
    // LDS: held in registers they would push the round's reads into scratch.
    // (The window's scan below orders these writes before any read.)
    __shared__ mfma_ab2 qs[KP][kWave];
    if (wave == 0) {
        const bf16* qrow = queries + (t_row * H + kv_head * G + min(g, G - 1)) * D;
#pragma unroll
        for (int st = 0; st < KP; ++st) qs[st][l] = g < G ? load8(qrow + st * 32 + 8 * quad) : mfma_ab2{};
    }

    float m = neg_inf(), lsum = 0.f;
    float acc[G][COLS];
#pragma unroll
    for (int h = 0; h < G; ++h)
#pragma unroll
        for (int c = 0; c < COLS; ++c) acc[h][c] = 0.f;
    const int col = COLS * (l % LPK), parity = l / LPK;
    const long long head_off = static_cast<long long>(kv_head) * D;
    const long long pitch = static_cast<long long>(KVH) * D;

    // This block's share of the entries: split z of gridDim.z.
    const int splits = static_cast<int>(gridDim.z), z = static_cast<int>(blockIdx.z);
    const int share = (entries + splits - 1) / splits;
    const int e_begin = z * share, e_end = min(entries, e_begin + share);
    const int chunk = min(kSelThreads, kSelTable / ratio);
    for (int e0 = e_begin; e0 < e_end; e0 += chunk) {
        // The window's keys: entry e0 + tid's count, scanned to its offset.
        const int e = e0 + static_cast<int>(threadIdx.x);
        int count = 0, j0 = 0;
        if (static_cast<int>(threadIdx.x) < chunk && e < e_end) {
            if (e < blocks) {
                const int c = srow[e];
                if (c >= 0 && c < nblocks) {
                    count = ratio;
                    j0 = c * ratio;
                }
            } else {
                j0 = nblocks * ratio;
                count = j_end - j0;
            }
        }
        int n = 0;
        const int at0 = block_exclusive_scan(count, wsum, n);
        for (int r = 0; r < count; ++r) {
            const int j = j0 + r;
            table[at0 + r] = static_cast<i32>(pidx[j / page_size]) * page_size + j % page_size;
        }
        __syncthreads();
        auto kv_row = [&](int key) {
            return static_cast<long long>(table[min(key, n - 1)]) * pitch + head_off;
        };

        for (int t0 = wave * SUB; t0 < n; t0 += WAVES * SUB) {
            // The round's key and value reads, all issued before any is used
            // (past the window they reread its last key, which weighs zero).
            mfma_ab2 kr[KP];
            {
                const bf16* row = k_pages + kv_row(t0 + g) + 8 * quad;
#pragma unroll
                for (int st = 0; st < KP; ++st) kr[st] = load8(row + st * 32);
            }
            ValuePack<COLS> vr[VR];
#pragma unroll
            for (int i = 0; i < VR; ++i)
                vr[i] = *reinterpret_cast<const ValuePack<COLS>*>(v_pages + kv_row(t0 + i * KPI + parity) + col);
            mfma_cd s = mfma_zero();
#pragma unroll
            for (int st = 0; st < KP; ++st) s = mfma16x2(kr[st], qs[st][l], s);
            float tmax = neg_inf();
#pragma unroll
            for (int q4 = 0; q4 < 4; ++q4) {
                const int key = t0 + 4 * quad + q4;
                s[q4] = key < n ? s[q4] * scale_log2 : neg_inf();
                tmax = fmaxf(tmax, s[q4]);
            }
            tmax = fmaxf(tmax, __shfl_xor(tmax, 16, kWave));
            tmax = fmaxf(tmax, __shfl_xor(tmax, 32, kWave));
            const float m_new = fmaxf(m, tmax);
            const float alpha = m_new == neg_inf() ? 1.f : exp2f(m - m_new);
            float sum = 0.f;
#pragma unroll
            for (int q4 = 0; q4 < 4; ++q4) {
                const float w = m_new == neg_inf() ? 0.f : exp2f(s[q4] - m_new);
                sum += w;
                if (g < G) ps[wave][4 * quad + q4][g] = w;
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
                for (int c = 0; c < COLS; ++c) acc[h][c] *= a;
            }
#pragma unroll
            for (int i = 0; i < VR; ++i) {
                const int kl = i * KPI + parity;
                float pv[G];
#pragma unroll
                for (int h = 0; h < G; ++h) pv[h] = ps[wave][kl][h];
#pragma unroll
                for (int c = 0; c < COLS; ++c) {
                    const float vf = to_f32(vr[i].v[c]);
#pragma unroll
                    for (int h = 0; h < G; ++h) acc[h][c] += pv[h] * vf;
                }
            }
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        }
        __syncthreads();
    }

    // The head's sum over its four quads, the columns' over the parities.
    lsum += __shfl_xor(lsum, 16, kWave);
    lsum += __shfl_xor(lsum, 32, kWave);
#pragma unroll
    for (int off = LPK; off < kWave; off <<= 1)
#pragma unroll
        for (int h = 0; h < G; ++h)
#pragma unroll
            for (int c = 0; c < COLS; ++c) acc[h][c] += __shfl_xor(acc[h][c], off, kWave);
    if (quad == 0 && g < G) {
        part_m[wave][g] = m;
        part_l[wave][g] = lsum;
    }

    // Split, the block leaves its unnormalized sum and its (max, total) per
    // head for `sdpa_selected_merge`: [z][launch row][head] planes.
    const long long slot = (static_cast<long long>(z) * gridDim.x + t) * H + kv_head * G;
    float* pacc = part == nullptr ? nullptr : part + slot * D;
    float* pml = part == nullptr ? nullptr : part + static_cast<long long>(splits) * gridDim.x * H * D + slot * 2;
    bf16* orow = o + (t_row * H + kv_head * G) * D;
#pragma unroll
    for (int h0 = 0; h0 < G; h0 += HG) {
        if (h0 > 0) __syncthreads();
        if (l < LPK)
#pragma unroll
            for (int h = 0; h < HG; ++h)
#pragma unroll
                for (int c = 0; c < COLS; ++c) part_o[wave][h][col + c] = acc[h0 + h][c];
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
            if (pacc == nullptr) {
                orow[h * D + j] = from_f32<bf16>(sum > 0.f ? out / sum : 0.f);
            } else {
                pacc[h * D + j] = out;
                if (j == 0) {
                    pml[2 * h] = big;
                    pml[2 * h + 1] = sum;
                }
            }
        }
    }
}

// The splits of `sdpa_selected_mfma` folded: per (launch row, head) the
// splits' sums rescaled to their common max, in split order. One block per
// (launch row, head), a thread a column.
constexpr int kMergeSplits = 16;

__global__ __launch_bounds__(256) void sdpa_selected_merge(const float* __restrict__ part,
                                                           bf16* __restrict__ o, i32 H, i32 D,
                                                           i32 splits, const u32* __restrict__ win) {
    const Window window{win};
    const int t = static_cast<int>(blockIdx.x), h = static_cast<int>(blockIdx.y);
    if (!window.live(t)) return;
    const long long t_row = window.plane(t);
    const long long rows = gridDim.x;
    const float* ml = part + splits * rows * H * D;
    float mz[kMergeSplits];
    float big = neg_inf();
#pragma unroll
    for (int z = 0; z < kMergeSplits; ++z) {
        mz[z] = z < splits ? ml[((z * rows + t) * H + h) * 2] : neg_inf();
        big = fmaxf(big, mz[z]);
    }
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float sum = 0.f, out = 0.f;
        if (big != neg_inf()) {
#pragma unroll
            for (int z = 0; z < kMergeSplits; ++z) {
                if (z >= splits || mz[z] == neg_inf()) continue;
                const long long at = (z * rows + t) * H + h;
                const float f = exp2f(mz[z] - big);
                sum += ml[at * 2 + 1] * f;
                out += part[at * D + j] * f;
            }
        }
        o[(t_row * H + h) * D + j] = from_f32<bf16>(sum > 0.f ? out / sum : 0.f);
    }
}

}  // namespace pie::attn::sdpa_selected
