#pragma once

// Gated DeltaNet: the causal depthwise conv in front of it, the gates, and
// the delta-rule recurrence over a per-slot state; and Kimi Delta Attention
// (at the end), the same with a decay per key channel. Value head h's state
// S[i][j] is bf16 at the row's head h (`StateRows::atom`) + i·V_d + j; a token
// stores S' = bf16(S·g) + k̂·δ (δ = (v − Sᵀgk̂)·β) and reads o = S'ᵀq̂ before
// S' rounds. A slot's conv state holds its last (K−1)·dil + 1 inputs, oldest
// first, cut along channels: a piece holds `[window][rows.atoms]`. A
// captured launch reads its extent from the window.

#include "prim/device.hip"
#include "prim/mfma.hip"
#include "prim/state_rows.hip"
#include "ssm/carry.hip"

namespace pie::attn {

using f32x4 = float __attribute__((ext_vector_type(4)));
using f32x2 = float __attribute__((ext_vector_type(2)));

__device__ __forceinline__ float silu(float z) { return z / (1.f + __expf(-z)); }

__device__ __forceinline__ bool folds(bool write_state, const u8* mask, int r) {
    return write_state && (mask == nullptr || mask[r] != 0);
}

// `folds` through a scalar load of the aligned word that holds the byte
// (a uniform byte load is a vector load, and waiting on it waits on every
// load issued before it): for a row a block, read where the stores need it.
__device__ __forceinline__ bool folds_uniform(bool write_state, const u8* mask, int r) {
    if (!write_state) return false;
    if (mask == nullptr) return true;
    const int off = static_cast<int>((reinterpret_cast<usize>(mask) + static_cast<usize>(r)) & 3);
    const u32 word = *reinterpret_cast<const u32*>(mask + r - off);
    return ((word >> (8 * off)) & 0xffu) != 0;
}

template <bool SILU, bool RESIDUAL>
__device__ __forceinline__ float conv_out(float acc, float x) {
    if constexpr (RESIDUAL) return acc + x;
    if constexpr (SILU) return silu(acc);
    return acc;
}

// One request's span of rows after its begin offset and commit cap.
struct Span {
    int t0;
    int n;
};

__device__ __forceinline__ Span span_of(const u32* indptr, int r, int rl, const int* commit_len,
                                        const int* begin_at) {
    Span s{static_cast<int>(indptr[r]), static_cast<int>(indptr[r + 1] - indptr[r])};
    if (begin_at != nullptr) {
        const int b = min(begin_at[rl], s.n);
        if (b > 0) {
            s.t0 += b;
            s.n -= b;
        }
    }
    if (commit_len != nullptr) s.n = min(s.n, commit_len[rl]);
    return s;
}

// A lane's state snapshots: entries snap[rl]..snap[rl + 1] of snap_rows (rows
// inside the lane, ascending) and snap_slots, each the state after that many
// rows landed in a slot of its own. None where snap is null.
struct Snaps {
    int e0;
    int e1;
};

__device__ __forceinline__ Snaps snaps_of(const int* snap, int rl) {
    return snap == nullptr ? Snaps{0, 0} : Snaps{snap[rl], snap[rl + 1]};
}


// The decode steps' state traffic: read once and written once a step, and
// not again until the next step, past every other layer's. Non-temporal, so
// it streams past the caches it would only evict from.
using u32x4 = u32 __attribute__((ext_vector_type(4)));
__device__ __forceinline__ u32x4 state_load(const bf16* p) {
    return __builtin_nontemporal_load(reinterpret_cast<const u32x4*>(p));
}
__device__ __forceinline__ void state_store(bf16* p, u32x4 v) {
    __builtin_nontemporal_store(v, reinterpret_cast<u32x4*>(p));
}
__device__ __forceinline__ f32x4 state_load(const f32x4* p) { return __builtin_nontemporal_load(p); }
__device__ __forceinline__ void state_store(f32x4* p, f32x4 v) { __builtin_nontemporal_store(v, p); }

// bf16 pairs packed in a word, low element first.
__device__ __forceinline__ float bf16_lo(u32 w) { return __builtin_bit_cast(float, w << 16); }
__device__ __forceinline__ float bf16_hi(u32 w) { return __builtin_bit_cast(float, w & 0xffff0000u); }

// Round to nearest even into the upper half, as `from_f32<bf16>` rounds every
// finite value and the NaNs arithmetic makes (canonical, or widened from
// bf16): two integer ops where the conversion spends five, and a pair packs
// in one more.
__device__ __forceinline__ u32 bf16_rne_bits(float x) {
    const u32 u = __builtin_bit_cast(u32, x);
    return u + 0x7fffu + ((u >> 16) & 1u);
}
__device__ __forceinline__ float bf16_round(float x) {
    return __builtin_bit_cast(float, bf16_rne_bits(x) & 0xffff0000u);
}
__device__ __forceinline__ u32 bf16_pack_rne(float lo, float hi) {
    return __builtin_amdgcn_perm(bf16_rne_bits(hi), bf16_rne_bits(lo), 0x07060302u);
}

constexpr int GDN_CH = 64;

struct GdnLane {
    int t0;
    int T;
    int rl;
    // The lane's first chunk id: negative where it holds no state.
    int slot;
    Snaps sn;
};

__device__ __forceinline__ GdnLane gdn_lane(int r, const u32* indptr, const StateRows& rows, const int* commit_len,
                                            const int* begin_at, const int* snap, const u32* win) {
    GdnLane lane;
    lane.rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const Span sp = span_of(indptr, r, lane.rl, commit_len, begin_at);
    lane.t0 = sp.t0;
    lane.T = sp.n;
    lane.slot = lane.T > 0 ? rows.first(lane.rl) : -1;
    lane.sn = snaps_of(snap, lane.rl);
    return lane;
}

// A snapshot row is a chunk boundary: the rows between two are chunked from
// the first, as a pass cut there would chunk them, so a request takes at most
// one more chunk a snapshot.
__device__ __forceinline__ int gdn_chunk_base(int r, const u32* indptr, const int* begin_at, const int* snap,
                                              const u32* win) {
    int t0 = static_cast<int>(indptr[r]);
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    if (begin_at != nullptr) {
        const int b = min(begin_at[rl], static_cast<int>(indptr[r + 1]) - t0);
        if (b > 0) t0 += b;
    }
    return t0 / GDN_CH + r + (snap != nullptr ? snap[rl] : 0);
}

// The request whose chunk slots hold s.
__device__ __forceinline__ int gdn_slot_lane(int s, int lanes, const u32* indptr, const int* begin_at,
                                             const int* snap, const u32* win) {
    int lo = 0, hi = lanes - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (gdn_chunk_base(mid, indptr, begin_at, snap, win) <= s) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

// Chunk j of a request: its first row, its rows (none past the last chunk),
// and the snapshot its end lands (−1: none).
struct GdnChunk {
    int ta;
    int L;
    int snap;
};

__device__ __forceinline__ GdnChunk gdn_chunk(const GdnLane& ln, int j, const int* snap_rows) {
    int a = 0;
    for (int e = ln.sn.e0; e < ln.sn.e1; ++e) {
        const int b = snap_rows[e];
        if (b >= ln.T) break;
        const int n = (b - a + GDN_CH - 1) / GDN_CH;
        if (j < n) return {ln.t0 + a + j * GDN_CH, min(GDN_CH, b - a - j * GDN_CH), j == n - 1 ? e : -1};
        j -= n;
        a = b;
    }
    return {ln.t0 + a + j * GDN_CH, min(GDN_CH, ln.T - a - j * GDN_CH), -1};
}

__device__ __forceinline__ int gdn_chunks(const GdnLane& ln, const int* snap_rows) {
    int a = 0, n = 0;
    for (int e = ln.sn.e0; e < ln.sn.e1 && snap_rows[e] < ln.T; ++e) {
        n += (snap_rows[e] - a + GDN_CH - 1) / GDN_CH;
        a = snap_rows[e];
    }
    return n + (ln.T - a + GDN_CH - 1) / GDN_CH;
}

// T = (I + A)⁻¹ for a strictly lower CH x CH A in LDS, in 16 x 16 blocks:
// each diagonal block by forward substitution, T[i][c] = δ_ic − Σ_{j<i}
// A[i][j]·T[j][c] with lane c keeping column c in registers; then the blocks
// below it a block-row at a time, T_ij = −T_ii·Σ_{j≤k<i} A_ik·T_kj, in f32
// on the matrix core. T overwrites A in place (the blocks above the diagonal
// are left as they are). X is CH/16 16 x 16 f32 tiles of LDS scratch, one a
// wavefront; every thread of a block of at least CH/16 wavefronts calls it.
template <int CH>
__device__ __forceinline__ void unit_lower_inverse(float (*A)[CH], float* X) {
    constexpr int NB = CH / 16;
    const int wave = threadIdx.x / kWave, l = lane();
    if (wave < NB && l < 16) {
        const int d = 16 * wave;
        float col[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            float t = i == l ? 1.f : 0.f;
#pragma unroll
            for (int jj = 0; jj < i; ++jj) t -= A[d + i][d + jj] * col[jj];
            col[i] = t;
            A[d + i][d + l] = t;
        }
    }
    lds_barrier();
    // 16 x 16 · 16 x 16 in f32 from LDS rows at stride `ld`.
    auto product = [&](const float* m, int ldm, const float* n, int ldn, mfma_cd c) {
#pragma unroll
        for (int k0 = 0; k0 < 16; k0 += 4) {
            const int k = k0 + l / 16;
            c = mfma4_f32(m[(l % 16) * ldm + k], n[k * ldn + l % 16], c);
        }
        return c;
    };
    float* Xw = X + (wave < NB ? wave : 0) * 256;
    for (int bi = 1; bi < NB; ++bi) {
        const int bj = wave;  // the block-row's blocks, one per wavefront
        mfma_cd t = mfma_zero();
        const bool mine = bj < bi;
        if (mine) {
            mfma_cd x = mfma_zero();
            for (int bk = bj; bk < bi; ++bk) x = product(&A[16 * bi][16 * bk], CH, &A[16 * bk][16 * bj], CH, x);
            mfma_each(x, [&](int i, int c, float v) { Xw[i * 16 + c] = v; });
            // (the wavefront reads back only what it wrote)
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
            t = product(&A[16 * bi][16 * bi], CH, Xw, 16, mfma_zero());
        }
        lds_barrier();
        if (mine) mfma_each(t, [&](int i, int c, float v) { A[16 * bi + i][16 * bj + c] = -v; });
        lds_barrier();
    }
}

}  // namespace pie::attn
