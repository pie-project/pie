#pragma once

#include "attn/ssm.cuh"
#include "prelude/tc5.cuh"

// The chunked gated delta rule's prefill on tcgen05, one block a (request,
// value head) walking the request's 64-token chunks in order, the state in
// TMEM throughout.
//
// Per chunk, with K and Q its raw (unnormalized) key and query rows, rk and
// rq their inverse norms (rq carrying the query scale), G the decay's
// running sum within the chunk and beta the write strength:
//
//   A   = tril(beta_i rk_i rk_j (k_i . k_j) e^{G_i - G_j}, -1)
//   T'  = (I + A)^-1 diag(beta)
//   NV  = T' (V - diag(e^G rk) K S^T)        (S^T: the state, value-major)
//   O   = diag(e^G rq) Q S^T + tril(rq_i rk_j (q_i . k_j) e^{G_i - G_j}) NV
//   S^T = e^{G_last} S^T + (diag(rk e^{G_last - G}) NV)^T K
//
// Two prep groups of four warps (0-3 and 10-13), one row a thread, take
// alternate chunks: each reads k k^T and q k^T off the tensor cores, the
// key norms off its diagonal and the query norms off the rows, forms A and
// the decayed causal scores, and inverts I + A. Warps 4-7, the state group,
// one value column a thread, keep the state: they hand its bf16 copy to the
// products that read it, decay it, scale the products' outputs by the
// per-row factors the prep groups land, and store the output. Warp 8 loads
// K, Q and V by TMA; warp 9 owns TMEM and issues every MMA.

namespace pie::attn::gdn {

using tc5::u32;
using tc5::u64;

constexpr int CH = 64;

struct Params {
    const float* gates;
    __nv_bfloat16* state;
    const int* slot_ids;
    const u32* qo_indptr;
    const int* commit_len;
    const int* begin_at;
    const u8* write_state_mask;
    float* out;
    const u32* win;
    long long slot_stride;
    int lanes;
    int K_h;
    int V_h;
    int write_state;
    float q_scale;
};

// Shared memory, from a 1024-aligned base.
constexpr u32 kKQ = 0;                  // 2 stages of [K;Q]: 128 rows, two 64-wide atoms
constexpr u32 kKQStage = 32768;
constexpr u32 kV = kKQ + 2 * kKQStage;  // 2 stages of V: 64 rows, two atoms
constexpr u32 kVStage = 16384;
constexpr u32 kT = kV + 2 * kVStage;    // 2 stages of T': 64 x 64 bf16
constexpr u32 kQK = kT + 2 * 8192;      // 2 stages of the decayed scores
constexpr int LDA = 68;
constexpr u32 kPrep = kQK + 2 * 8192;   // per prep group: A and T in f32, the merge scratch
constexpr u32 kPrepA = 0;
constexpr u32 kPrepT = CH * LDA * 4;
constexpr u32 kPrepX = 2 * CH * LDA * 4;
constexpr u32 kPrepBytes = kPrepX + 4096;
// Per stage, 64 floats each: G (log2), beta, rk, rq, and the per-row
// factors the state group reads, e^G rk, e^G rq and rk e^(G_last - G)
// (zero past the chunk), then e^G_last.
constexpr u32 kRing = kPrep + 2 * kPrepBytes;
constexpr int kRingF = 8 * CH;
constexpr u32 kBar = kRing + 2 * kRingF * 4;
constexpr u32 kSmem = kBar + 512;

__host__ __device__ constexpr u32 smem_bytes() { return kSmem + 1024; }

// TMEM columns.
constexpr u32 kColS = 0;     // S^T, f32: lane = value column, column = key column
constexpr u32 kColKS = 128;  // K S^T, then NV
constexpr u32 kColQS = 192;  // Q S^T, then O
constexpr u32 kColKK = 256;  // [K;Q] K^T, one buffer a prep group
constexpr u32 kColSI = 384;  // S^T in bf16, the A operand of [K;Q] S^T
constexpr u32 kColU = 448;   // -diag(e^G rk) K S^T, then NV, in bf16
constexpr u32 kColD = 480;   // the decayed NV in bf16

// MN-major, 128B-swizzled operand: atoms of 64 MN elements by 8 K rows,
// `lbo` bytes from one 64-wide MN atom to the next, 1024 bytes from one
// 8-row K group to the next.
__device__ __forceinline__ u64 sdesc_sw128_mn(u32 saddr, u32 lbo) {
    u64 d = 0;
    d |= static_cast<u64>((saddr & 0x3FFFFu) >> 4);
    d |= static_cast<u64>(lbo >> 4) << 16;
    d |= static_cast<u64>(1024 >> 4) << 32;
    d |= static_cast<u64>(1) << 46;
    d |= static_cast<u64>(2) << 61;
    return d;
}

__device__ __forceinline__ void named_sync(u32 id, u32 count) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
}

__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ u32 tf32(float x) {
    u32 r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ void mma_tf32(float (&d)[4], const u32 (&a)[4], u32 b0, u32 b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// D (16 x 8) += X[r0.., xk0..] . Y[yk0.., n0..] over `ks` steps of eight,
// both f32 row-major, in tf32.
__device__ __forceinline__ void tile_mm(float (&d)[4], const float* X, int ldx, int r0, int xk0,
                                        const float* Y, int ldy, int yk0, int n0, int ks, int l) {
    const int g = l >> 2;
    const int t = l & 3;
    for (int k = 0; k < ks; ++k) {
        const int xk = xk0 + 8 * k;
        const int yk = yk0 + 8 * k;
        u32 a[4];
        a[0] = tf32(X[(r0 + g) * ldx + xk + t]);
        a[1] = tf32(X[(r0 + g + 8) * ldx + xk + t]);
        a[2] = tf32(X[(r0 + g) * ldx + xk + t + 4]);
        a[3] = tf32(X[(r0 + g + 8) * ldx + xk + t + 4]);
        const u32 b0 = tf32(Y[(yk + t) * ldy + n0 + g]);
        const u32 b1 = tf32(Y[(yk + t + 4) * ldy + n0 + g]);
        mma_tf32(d, a, b0, b1);
    }
}

// Lands a 16 x 8 accumulator tile, scaled, at Z[r0.., n0..].
__device__ __forceinline__ void tile_put(const float (&d)[4], float* Z, int ldz, int r0, int n0,
                                         float scale, int l) {
    const int g = l >> 2;
    const int t = l & 3;
    Z[(r0 + g) * ldz + n0 + 2 * t] = d[0] * scale;
    Z[(r0 + g) * ldz + n0 + 2 * t + 1] = d[1] * scale;
    Z[(r0 + g + 8) * ldz + n0 + 2 * t] = d[2] * scale;
    Z[(r0 + g + 8) * ldz + n0 + 2 * t + 1] = d[3] * scale;
}

// The walk every role takes: the block's (request, value head) tiles, each
// its chunks in order.
struct Walk {
    int tile;
    int r;
    int h;
    GdnLane lane;
    int nch;
    int j;
};

__device__ __forceinline__ bool next_tile(const Params& p, int lanes, Walk& w) {
    for (; w.tile < lanes * p.V_h; w.tile += static_cast<int>(gridDim.x)) {
        w.r = w.tile / p.V_h;
        w.h = w.tile % p.V_h;
        w.lane = gdn_lane(w.r, p.qo_indptr, p.slot_ids, p.commit_len, p.begin_at, p.win);
        if (w.lane.T <= 0 || w.lane.slot < 0) continue;
        w.nch = (w.lane.T + CH - 1) / CH;
        w.j = 0;
        return true;
    }
    return false;
}

__device__ __forceinline__ bool advance(const Params& p, int lanes, Walk& w) {
    if (++w.j < w.nch) return true;
    w.tile += static_cast<int>(gridDim.x);
    return next_tile(p, lanes, w);
}

__global__ void __launch_bounds__(448, 1) gdn_chunked_tc5(
    const __grid_constant__ tc5::TmaMap map_qkv, const Params p) {
    using namespace tc5;
    extern __shared__ unsigned char smem_raw[];
    const u32 raw = smem_addr(smem_raw);
    unsigned char* smem = smem_raw + (((raw + 1023u) & ~1023u) - raw);
    const u32 sbase = smem_addr(smem);

    u64* bars = reinterpret_cast<u64*>(smem + kBar);
    u64* kq_full = bars + 0;
    u64* kq_empty = bars + 2;
    u64* v_full = bars + 4;
    u64* v_empty = bars + 6;
    u64* t_full = bars + 8;
    u64* t_empty = bars + 10;
    u64* g_full = bars + 12;
    u64* cg1_done = bars + 14;
    u64* kk_full = bars + 16;
    u64* kk_free = bars + 18;
    u64* si_full = bars + 20;
    u64* ksqs_full = bars + 21;
    u64* u_full = bars + 22;
    u64* nv_full = bars + 23;
    u64* nvb_full = bars + 24;
    u64* o_full = bars + 25;
    u64* o_free = bars + 26;
    u64* kv_done = bars + 27;
    u32* tmem_slot = reinterpret_cast<u32*>(bars + 28);

    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int l = tid & 31;
    int lanes = p.lanes;
    if (p.win != nullptr) lanes = min(lanes, static_cast<int>(p.win[2]));
    const int row0 = p.win != nullptr ? static_cast<int>(p.win[1]) : 0;
    const int group = p.V_h / p.K_h;
    const int K_dim = p.K_h * 128;

    if (tid == 0) {
        for (int s = 0; s < 2; ++s) {
            bar_init(&kq_full[s], 1);
            bar_init(&kq_empty[s], 1);
            bar_init(&v_full[s], 1);
            bar_init(&v_empty[s], 1);
            bar_init(&t_full[s], 128);
            bar_init(&t_empty[s], 1);
            bar_init(&g_full[s], 128);
            bar_init(&cg1_done[s], 128);
            bar_init(&kk_full[s], 1);
            bar_init(&kk_free[s], 128);
        }
        bar_init(si_full, 128);
        bar_init(ksqs_full, 1);
        bar_init(u_full, 128);
        bar_init(nv_full, 1);
        bar_init(nvb_full, 128);
        bar_init(o_full, 1);
        bar_init(o_free, 128);
        bar_init(kv_done, 1);
        bar_init_fence();
    }
    if (warp == 9) {
        tmem_alloc(tmem_slot, 512);
        tmem_relinquish();
    }
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const u32 tmem = *tmem_slot;

    Walk w{static_cast<int>(blockIdx.x), 0, 0, {}, 0, 0};
    bool live = next_tile(p, lanes, w);

    if (warp == 8) {
        if (l == 0 && live) {
            tma_prefetch(&map_qkv);
            int c = 0;
            do {
                const int s = c & 1;
                const u32 ph = (c >> 1) & 1;
                const int hk = w.h / group;
                const int y = row0 + w.lane.t0 + w.j * CH;
                unsigned char* kqb = smem + kKQ + s * kKQStage;
                bar_wait(&kq_empty[s], ph ^ 1);
                bar_arrive_expect_tx(&kq_full[s], kKQStage);
                tma_load_3d(kqb, &kq_full[s], &map_qkv, K_dim + hk * 128, y, 0);
                tma_load_3d(kqb + 8192, &kq_full[s], &map_qkv, hk * 128, y, 0);
                tma_load_3d(kqb + 16384, &kq_full[s], &map_qkv, K_dim + hk * 128 + 64, y, 0);
                tma_load_3d(kqb + 24576, &kq_full[s], &map_qkv, hk * 128 + 64, y, 0);
                unsigned char* vb = smem + kV + s * kVStage;
                bar_wait(&v_empty[s], ph ^ 1);
                bar_arrive_expect_tx(&v_full[s], kVStage);
                tma_load_3d(vb, &v_full[s], &map_qkv, 2 * K_dim + w.h * 128, y, 0);
                tma_load_3d(vb + 8192, &v_full[s], &map_qkv, 2 * K_dim + w.h * 128 + 64, y, 0);
                ++c;
            } while (advance(p, lanes, w));
        }
    } else if (warp == 9) {
        if (l == 0 && live) {
            const u32 id128 = idesc_dense(kF16Bf16, kF16Bf16, 128, 128);
            const u32 id64 = idesc_dense(kF16Bf16, kF16Bf16, 128, 64);
            const u32 idkv = id128 | (1u << 16);
            const u32 idv = id64 | (1u << 15);
            // [K;Q] K^T of chunk c into its prep group's buffer.
            auto kk = [&](int c) {
                const int s = c & 1;
                const u32 kqb = sbase + kKQ + s * kKQStage;
                bar_wait(&kq_full[s], (c >> 1) & 1);
                bar_wait(&kk_free[s], ((c >> 1) & 1) ^ 1);
                tc_fence_after();
#pragma unroll
                for (u32 k = 0; k < 8; ++k) {
                    const u64 d = sdesc_sw128(kqb + (k / 4) * 16384) + 2 * (k % 4);
                    mma_f16(tmem + kColKK + s * 64, d, d, id64, k > 0);
                }
                tc_commit(&kk_full[s]);
            };
            kk(0);
            int c = 0;
            bool more = true;
            while (more) {
                const int s = c & 1;
                const u32 ph = (c >> 1) & 1;
                const u32 kqb = sbase + kKQ + s * kKQStage;
                more = advance(p, lanes, w);
                // [K;Q] S^T, the state's bf16 copy in TMEM as A.
                bar_wait(si_full, c & 1);
                if (c > 0) bar_wait(o_free, (c - 1) & 1);
                tc_fence_after();
#pragma unroll
                for (u32 k = 0; k < 8; ++k) {
                    mma_f16_ts(tmem + kColKS, tmem + kColSI + k * 8,
                               sdesc_sw128(kqb + (k / 4) * 16384) + 2 * (k % 4), id128, k > 0);
                }
                tc_commit(ksqs_full);
                // The next chunk's product, for the other prep group to work
                // beside this one.
                if (more) kk(c + 1);
                // NV = T' V + T' (-diag(e^G rk) K S^T).
                // Its D overwrites K S^T: the state group must have read it.
                bar_wait(u_full, c & 1);
                bar_wait(&t_full[s], ph);
                bar_wait(&v_full[s], ph);
                tc_fence_after();
                const u32 tb = sbase + kT + s * 8192;
                const u32 vb = sbase + kV + s * kVStage;
#pragma unroll
                for (u32 k = 0; k < 4; ++k) {
                    mma_f16(tmem + kColKS, sdesc_sw128_mn(vb + k * 2048, 8192),
                            sdesc_sw128(tb) + 2 * k, idv, k > 0);
                }
#pragma unroll
                for (u32 k = 0; k < 4; ++k) {
                    mma_f16_ts(tmem + kColKS, tmem + kColU + k * 8, sdesc_sw128(tb) + 2 * k, id64,
                               true);
                }
                tc_commit(nv_full);
                tc_commit(&v_empty[s]);
                // O += P NV, then S^T += (decayed NV)^T K.
                bar_wait(nvb_full, c & 1);
                tc_fence_after();
                const u32 qkb = sbase + kQK + s * 8192;
#pragma unroll
                for (u32 k = 0; k < 4; ++k) {
                    mma_f16_ts(tmem + kColQS, tmem + kColU + k * 8, sdesc_sw128(qkb) + 2 * k, id64,
                               true);
                }
                tc_commit(o_full);
                tc_commit(&t_empty[s]);
#pragma unroll
                for (u32 k = 0; k < 4; ++k) {
                    mma_f16_ts(tmem + kColS, tmem + kColD + k * 8,
                               sdesc_sw128_mn(kqb + k * 2048, 16384), idkv, true);
                }
                tc_commit(kv_done);
                tc_commit(&kq_empty[s]);
                ++c;
            }
        }
    } else if (warp < 4 || warp >= 10) {
        // A prep group: the chunks of its parity. Thread i (by its TMEM lane)
        // holds row i of [K;Q] K^T: k k^T for i < 64, q k^T past it.
        if (live) {
            const int gq = warp < 4 ? 0 : 1;
            const u32 quad = static_cast<u32>(warp & 3);
            const int i128 = static_cast<int>(quad) * 32 + l;
            const u32 bar_id = 1 + gq * 2;
            unsigned char* prep = smem + kPrep + gq * kPrepBytes;
            float* sA = reinterpret_cast<float*>(prep + kPrepA);
            float* Tm = reinterpret_cast<float*>(prep + kPrepT);
            for (int e = i128; e < CH * LDA; e += 128) Tm[e] = 0.f;
            named_sync(bar_id, 128);
            const u32 lane_base = (quad * 32u) << 16;
            int c = 0;
            do {
                if ((c & 1) != gq) {
                    ++c;
                    continue;
                }
                const int s = gq;
                const u32 ph = (c >> 1) & 1;
                float* G = reinterpret_cast<float*>(smem + kRing) + s * kRingF;
                float* B = G + CH;
                float* RK = B + CH;
                float* RQ = RK + CH;
                float* FK = RQ + CH;
                float* FQ = FK + CH;
                float* FD = FQ + CH;
                const int ta = w.lane.t0 + w.j * CH;
                const int L = min(CH, w.lane.t0 + w.lane.T - ta);
                bar_wait(&cg1_done[s], ph ^ 1);
                bar_wait(&t_empty[s], ph ^ 1);
                if (quad == 0) {
                    float gv[2], bv[2];
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const int i = 2 * l + e;
                        const bool in = i < L;
                        const float* grow =
                            p.gates + static_cast<long long>(row0 + ta + (in ? i : 0)) * 2 * p.V_h;
                        gv[e] = in ? grow[w.h] : 0.f;
                        bv[e] = in ? grow[p.V_h + w.h] : 0.f;
                    }
                    const float b2 = gv[0] + gv[1];
                    float run = b2;
#pragma unroll
                    for (int off = 1; off < 32; off <<= 1) {
                        const float up = __shfl_up_sync(0xffffffffu, run, off);
                        if (l >= off) run += up;
                    }
                    G[2 * l] = (run - b2 + gv[0]) * 1.4426950408889634f;
                    G[2 * l + 1] = run * 1.4426950408889634f;
                    B[2 * l] = bv[0];
                    B[2 * l + 1] = bv[1];
                }
                // The query norms off the rows themselves.
                bar_wait(&kq_full[s], ph);
                if (i128 >= CH) {
                    const int i = i128 - CH;
                    const unsigned char* kqb = smem + kKQ + s * kKQStage;
                    float ss = 0.f;
#pragma unroll
                    for (int a = 0; a < 2; ++a) {
#pragma unroll
                        for (int q = 0; q < 8; ++q) {
                            const uint4 x = *reinterpret_cast<const uint4*>(
                                kqb + a * 16384 + (CH + i) * 128 + q * 16);
                            const u32 xs[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
                            for (int e = 0; e < 4; ++e) {
                                const float lo = __uint_as_float(xs[e] << 16);
                                const float hi = __uint_as_float(xs[e] & 0xffff0000u);
                                ss = __fmaf_rn(lo, lo, ss);
                                ss = __fmaf_rn(hi, hi, ss);
                            }
                        }
                    }
                    RQ[i] = rsqrtf(ss + 1e-6f) * p.q_scale;
                }
                bar_wait(&kk_full[s], ph);
                tc_fence_after();
                u32 r0[32], r1[32];
                tmem_ld32(tmem + lane_base + kColKK + s * 64, r0);
                tmem_ld32(tmem + lane_base + kColKK + s * 64 + 32, r1);
                tmem_wait_ld();
                tc_fence_before();
                bar_arrive(&kk_free[s]);
                if (i128 < CH) {
                    float dg = 0.f;
#pragma unroll
                    for (int q = 0; q < 32; ++q) {
                        const u32 x = quad == 0 ? r0[q] : r1[q];
                        if (q == l) dg = __uint_as_float(x);
                    }
                    RK[i128] = rsqrtf(dg + 1e-6f);
                }
                named_sync(bar_id, 128);
                if (i128 < CH) {
                    const float gt = G[i128];
                    const float gl = G[L - 1];
                    const float et = ex2(gt);
                    FK[i128] = -et * RK[i128];
                    FQ[i128] = et * RQ[i128];
                    FD[i128] = i128 < L ? RK[i128] * ex2(gl - gt) : 0.f;
                    if (i128 == 0) FD[CH] = ex2(gl);
                }
                named_sync(bar_id, 128);
                bar_arrive(&g_full[s]);
                if (i128 < CH) {
                    const int i = i128;
                    const float gi = G[i];
                    const float f = B[i] * RK[i];
#pragma unroll
                    for (int jb = 0; jb < 4; ++jb) {
                        if (jb * 16 >= i) break;
                        float gj[16], rj[16], av[16];
#pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            const float4 g4 = reinterpret_cast<const float4*>(G)[jb * 4 + q];
                            const float4 r4 = reinterpret_cast<const float4*>(RK)[jb * 4 + q];
                            gj[4 * q] = g4.x; gj[4 * q + 1] = g4.y; gj[4 * q + 2] = g4.z; gj[4 * q + 3] = g4.w;
                            rj[4 * q] = r4.x; rj[4 * q + 1] = r4.y; rj[4 * q + 2] = r4.z; rj[4 * q + 3] = r4.w;
                        }
#pragma unroll
                        for (int u = 0; u < 16; ++u) {
                            const int jj = jb * 16 + u;
                            const float kv = __uint_as_float(jj < 32 ? r0[jj] : r1[jj - 32]);
                            av[u] = (jj < i && i < L) ? kv * f * rj[u] * ex2(gi - gj[u]) : 0.f;
                        }
#pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            reinterpret_cast<float4*>(sA + i * LDA + jb * 16)[q] =
                                make_float4(av[4 * q], av[4 * q + 1], av[4 * q + 2], av[4 * q + 3]);
                        }
                    }
                } else {
                    const int i = i128 - CH;
                    const float gi = G[i];
                    const float f = RQ[i];
                    unsigned char* qkb = smem + kQK + s * 8192;
#pragma unroll
                    for (int jb = 0; jb < 4; ++jb) {
                        u32 pk[8];
                        if (jb * 16 > i) {
#pragma unroll
                            for (int e = 0; e < 8; ++e) pk[e] = 0u;
                        } else {
                            float gj[16], rj[16];
#pragma unroll
                            for (int q = 0; q < 4; ++q) {
                                const float4 g4 = reinterpret_cast<const float4*>(G)[jb * 4 + q];
                                const float4 r4 = reinterpret_cast<const float4*>(RK)[jb * 4 + q];
                                gj[4 * q] = g4.x; gj[4 * q + 1] = g4.y; gj[4 * q + 2] = g4.z; gj[4 * q + 3] = g4.w;
                                rj[4 * q] = r4.x; rj[4 * q + 1] = r4.y; rj[4 * q + 2] = r4.z; rj[4 * q + 3] = r4.w;
                            }
#pragma unroll
                            for (int e = 0; e < 8; ++e) {
                                float v2[2];
#pragma unroll
                                for (int x = 0; x < 2; ++x) {
                                    const int u = 2 * e + x;
                                    const int jj = jb * 16 + u;
                                    const float qk = __uint_as_float(jj < 32 ? r0[jj] : r1[jj - 32]);
                                    v2[x] = (jj <= i && i < L) ? qk * f * rj[u] * ex2(gi - gj[u]) : 0.f;
                                }
                                pk[e] = gdn_pack(v2[0], v2[1]);
                            }
                        }
#pragma unroll
                        for (int h2 = 0; h2 < 2; ++h2) {
                            const int cj = jb * 2 + h2;
                            *reinterpret_cast<uint4*>(qkb + i * 128 + (((cj ^ (i & 7))) << 4)) =
                                make_uint4(pk[4 * h2], pk[4 * h2 + 1], pk[4 * h2 + 2], pk[4 * h2 + 3]);
                        }
                    }
                    fence_async_shared();
                }
                named_sync(bar_id, 128);

                // T = (I + A)^-1: each warp inverts one sixteen-row diagonal
                // block by forward substitution in f32, then two tf32 merge
                // levels form the blocks below the diagonal.
                if (l < 16) {
                    const int d0 = static_cast<int>(quad) * 16;
                    float tcol[16];
#pragma unroll
                    for (int rr = 0; rr < 16; ++rr) {
                        const float* ar = sA + (d0 + rr) * LDA + d0;
                        float sum = 0.f;
#pragma unroll
                        for (int m = 0; m < rr; ++m) sum = __fmaf_rn(ar[m], tcol[m], sum);
                        tcol[rr] = rr == l ? 1.f : (rr < l ? 0.f : -sum);
                    }
#pragma unroll
                    for (int rr = 0; rr < 16; ++rr) Tm[(d0 + rr) * LDA + d0 + l] = tcol[rr];
                }
                named_sync(bar_id, 128);
                {
                    // 16 -> 32: T10 = -D1 (A10 D0) and T32 = -D3 (A32 D2),
                    // a warp an eight-column half of one.
                    float* P = reinterpret_cast<float*>(prep + kPrepX) + quad * 128;
                    const int b = static_cast<int>(quad >> 1) * 32;
                    const int n0 = static_cast<int>(quad & 1) * 8;
                    float d[4] = {0.f, 0.f, 0.f, 0.f};
                    tile_mm(d, sA, LDA, b + 16, b, Tm, LDA, b, b + n0, 2, l);
                    tile_put(d, P, 8, 0, 0, 1.f, l);
                    __syncwarp();
                    float e[4] = {0.f, 0.f, 0.f, 0.f};
                    tile_mm(e, Tm, LDA, b + 16, b + 16, P, 8, 0, 0, 2, l);
                    tile_put(e, Tm, LDA, b + 16, b + n0, -1.f, l);
                }
                named_sync(bar_id, 128);
                {
                    // 32 -> 64: T[32:, :32] = -D23 (A[32:, :32] D01), a warp
                    // a 16 x 16 quarter of each product.
                    float* P = reinterpret_cast<float*>(prep + kPrepX);
                    const int r = 32 + static_cast<int>(quad >> 1) * 16;
                    const int nb = static_cast<int>(quad & 1) * 16;
#pragma unroll
                    for (int h2 = 0; h2 < 2; ++h2) {
                        float d[4] = {0.f, 0.f, 0.f, 0.f};
                        tile_mm(d, sA, LDA, r, 0, Tm, LDA, 0, nb + 8 * h2, 4, l);
                        tile_put(d, P, 32, r - 32, nb + 8 * h2, 1.f, l);
                    }
                    named_sync(bar_id, 128);
#pragma unroll
                    for (int h2 = 0; h2 < 2; ++h2) {
                        float d[4] = {0.f, 0.f, 0.f, 0.f};
                        tile_mm(d, Tm, LDA, r, 32, P, 32, 0, nb + 8 * h2, 4, l);
                        tile_put(d, Tm, LDA, r, nb + 8 * h2, -1.f, l);
                    }
                }
                named_sync(bar_id, 128);
                // T' = T diag(beta) in bf16, the B operand of NV = T' U.
                if (i128 < CH) {
                    const int i = i128;
                    unsigned char* tb = smem + kT + s * 8192;
#pragma unroll
                    for (int cj = 0; cj < 8; ++cj) {
                        u32 pk[4] = {0u, 0u, 0u, 0u};
                        if (cj * 8 <= i) {
                            const float4 t0 = reinterpret_cast<const float4*>(Tm + i * LDA)[2 * cj];
                            const float4 t1 = reinterpret_cast<const float4*>(Tm + i * LDA)[2 * cj + 1];
                            const float4 b0 = reinterpret_cast<const float4*>(B)[2 * cj];
                            const float4 b1 = reinterpret_cast<const float4*>(B)[2 * cj + 1];
                            pk[0] = gdn_pack(t0.x * b0.x, t0.y * b0.y);
                            pk[1] = gdn_pack(t0.z * b0.z, t0.w * b0.w);
                            pk[2] = gdn_pack(t1.x * b1.x, t1.y * b1.y);
                            pk[3] = gdn_pack(t1.z * b1.z, t1.w * b1.w);
                        }
                        *reinterpret_cast<uint4*>(tb + i * 128 + (((cj ^ (i & 7))) << 4)) =
                            make_uint4(pk[0], pk[1], pk[2], pk[3]);
                    }
                    fence_async_shared();
                }
                bar_arrive(&t_full[s]);
                ++c;
            } while (advance(p, lanes, w));
        }
    } else {
        // The state group: thread v holds value column v of S^T and of
        // every product's output.
        if (live) {
            const int v = tid - 128;
            const u32 lane_base = (static_cast<u32>(warp - 4) * 32u) << 16;
            const u32 tb = tmem + lane_base;
            const int stride_row = p.V_h * 128;
            int c = 0;
            do {
                const int s = c & 1;
                const u32 ph = (c >> 1) & 1;
                const float* G = reinterpret_cast<const float*>(smem + kRing) + s * kRingF;
                const float* FK = G + 4 * CH;
                const float* FQ = G + 5 * CH;
                const float* FD = G + 6 * CH;
                const int ta = w.lane.t0 + w.j * CH;
                const int L = min(CH, w.lane.t0 + w.lane.T - ta);
                __nv_bfloat16* st = p.state + static_cast<long long>(w.lane.slot) * p.slot_stride +
                                    static_cast<long long>(w.h) * 128 * 128;
                bar_wait(kv_done, (c & 1) ^ 1);
                tc_fence_after();
                if (w.j == 0) {
                    // The request's state, [key][value] in bf16, into S^T.
#pragma unroll 1
                    for (int q = 0; q < 4; ++q) {
                        u32 x[32];
#pragma unroll
                        for (int i = 0; i < 32; ++i) {
                            const unsigned short b = reinterpret_cast<const unsigned short*>(
                                st)[(q * 32 + i) * 128 + v];
                            x[i] = static_cast<u32>(b) << 16;
                        }
                        tmem_st32(tb + kColS + q * 32, x);
                    }
                    tmem_wait_st();
                }
                // Its bf16 copy for [K;Q] S^T first, then the decay.
#pragma unroll 1
                for (int hq = 0; hq < 2; ++hq) {
                    u32 a[32], b[32], pk[32];
                    tmem_ld32(tb + kColS + hq * 64, a);
                    tmem_ld32(tb + kColS + hq * 64 + 32, b);
                    tmem_wait_ld();
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        pk[i] = gdn_pack(__uint_as_float(a[2 * i]), __uint_as_float(a[2 * i + 1]));
                        pk[16 + i] =
                            gdn_pack(__uint_as_float(b[2 * i]), __uint_as_float(b[2 * i + 1]));
                    }
                    tmem_st32(tb + kColSI + hq * 32, pk);
                }
                tmem_wait_st();
                tc_fence_before();
                bar_arrive(si_full);
                bar_wait(&g_full[s], ph);
                {
                    const float el = FD[CH];
#pragma unroll 1
                    for (int q = 0; q < 4; ++q) {
                        u32 a[32];
                        tmem_ld32(tb + kColS + q * 32, a);
                        tmem_wait_ld();
#pragma unroll
                        for (int i = 0; i < 32; ++i) a[i] = __float_as_uint(__uint_as_float(a[i]) * el);
                        tmem_st32(tb + kColS + q * 32, a);
                    }
                }

                // -diag(e^G rk) K S^T in bf16, and Q S^T scaled to the
                // output's first part.
                bar_wait(ksqs_full, c & 1);
                tc_fence_after();
                {
                    u32 k0[32], k1[32], pk[32];
                    tmem_ld32(tb + kColKS, k0);
                    tmem_ld32(tb + kColKS + 32, k1);
                    tmem_wait_ld();
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        pk[i] = gdn_pack(__uint_as_float(k0[2 * i]) * FK[2 * i],
                                         __uint_as_float(k0[2 * i + 1]) * FK[2 * i + 1]);
                        pk[16 + i] = gdn_pack(__uint_as_float(k1[2 * i]) * FK[32 + 2 * i],
                                              __uint_as_float(k1[2 * i + 1]) * FK[32 + 2 * i + 1]);
                    }
                    tmem_st32(tb + kColU, pk);
                    tmem_wait_st();
                    tc_fence_before();
                    bar_arrive(u_full);
                    tmem_ld32(tb + kColQS, k0);
                    tmem_ld32(tb + kColQS + 32, k1);
                    tmem_wait_ld();
#pragma unroll
                    for (int i = 0; i < 32; ++i) {
                        k0[i] = __float_as_uint(__uint_as_float(k0[i]) * FQ[i]);
                        k1[i] = __float_as_uint(__uint_as_float(k1[i]) * FQ[32 + i]);
                    }
                    tmem_st32(tb + kColQS, k0);
                    tmem_st32(tb + kColQS + 32, k1);
                }

                // NV into bf16 for O += P NV, and its decayed, normed rows for
                // the state's update.
                bar_wait(nv_full, c & 1);
                tc_fence_after();
                {
                    u32 n0[32], n1[32], pk[32], pd[32];
                    tmem_ld32(tb + kColKS, n0);
                    tmem_ld32(tb + kColKS + 32, n1);
                    tmem_wait_ld();
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const float a0 = __uint_as_float(n0[2 * i]);
                        const float a1 = __uint_as_float(n0[2 * i + 1]);
                        const float c0 = __uint_as_float(n1[2 * i]);
                        const float c1 = __uint_as_float(n1[2 * i + 1]);
                        pk[i] = gdn_pack(a0, a1);
                        pk[16 + i] = gdn_pack(c0, c1);
                        pd[i] = gdn_pack(a0 * FD[2 * i], a1 * FD[2 * i + 1]);
                        pd[16 + i] = gdn_pack(c0 * FD[32 + 2 * i], c1 * FD[32 + 2 * i + 1]);
                    }
                    tmem_st32(tb + kColU, pk);
                    tmem_st32(tb + kColD, pd);
                    tmem_wait_st();
                }
                tc_fence_before();
                bar_arrive(nvb_full);

                // The output.
                bar_wait(o_full, c & 1);
                tc_fence_after();
                {
                    u32 o0[32], o1[32];
                    tmem_ld32(tb + kColQS, o0);
                    tmem_ld32(tb + kColQS + 32, o1);
                    tmem_wait_ld();
                    tc_fence_before();
                    bar_arrive(o_free);
                    float* orow = p.out + static_cast<long long>(row0 + ta) * stride_row + w.h * 128 + v;
#pragma unroll
                    for (int t = 0; t < 32; ++t) {
                        if (t < L) orow[static_cast<long long>(t) * stride_row] = __uint_as_float(o0[t]);
                    }
#pragma unroll
                    for (int t = 0; t < 32; ++t) {
                        if (32 + t < L)
                            orow[static_cast<long long>(32 + t) * stride_row] = __uint_as_float(o1[t]);
                    }
                }
                bar_arrive(&cg1_done[s]);

                if (w.j == w.nch - 1) {
                    // The request's last chunk folded: its state back out.
                    bar_wait(kv_done, c & 1);
                    tc_fence_after();
                    if (p.write_state && row_persists(p.write_state_mask, w.lane.rl)) {
#pragma unroll 1
                        for (int q = 0; q < 4; ++q) {
                            u32 x[32];
                            tmem_ld32(tb + kColS + q * 32, x);
                            tmem_wait_ld();
#pragma unroll
                            for (int i = 0; i < 32; ++i) {
                                const u32 bits = gdn_pack(__uint_as_float(x[i]), 0.f);
                                reinterpret_cast<unsigned short*>(st)[(q * 32 + i) * 128 + v] =
                                    static_cast<unsigned short>(bits & 0xffffu);
                            }
                        }
                    }
                }
                ++c;
            } while (advance(p, lanes, w));
        }
    }
    tc_fence_before();
    __syncthreads();
    if (warp == 9) {
        tc_fence_after();
        tmem_dealloc(tmem, 512);
    }
}

}  // namespace pie::attn::gdn
