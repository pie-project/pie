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
// Two prep groups of four warps (0-3 and 8-11), one row a thread, take
// alternate chunks: each issues its own k k^T and q k^T, reads them off the
// tensor cores, the key norms off its diagonal and the query norms off the
// rows, forms A and the decayed causal scores, and inverts I + A. Warps 4-7,
// the state group, one value column a thread, keep the state: they hand its
// bf16 copy to the products that read it, decay it, scale the products'
// outputs by the per-row factors the prep groups land, and store the output;
// their thread 0 issues the chain's products and the TMA loads, a chunk
// ahead. Twelve warps leave each thread 168 registers.

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
constexpr u32 kPrep = kQK + 2 * 8192;   // per prep group: A and T in f16, the merge scratch
constexpr u32 kPrepA = 0;
constexpr u32 kPrepT = CH * 72 * 2;
constexpr u32 kPrepX = 2 * CH * 72 * 2;
constexpr u32 kPrepBytes = kPrepX + 4096;
// A whole chunk's output, [token][value] in f32, for one TMA store.
constexpr u32 kO = kPrep + 2 * kPrepBytes;
// Per stage, 64 floats each: G (log2), beta, rk, rq, and the per-row
// factors the state group reads, e^G rk, e^G rq and rk e^(G_last - G)
// (zero past the chunk), then e^G_last.
constexpr u32 kRing = kO + CH * 128 * 4;
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

// f16 pairs and the f16 tensor-core product the inverse runs on: its
// operands keep a tf32's ten mantissa bits, and ldmatrix lands them.
constexpr int LDH = 72;  // an f16 64-wide row padded so ldmatrix's eight rows miss each other's banks

__device__ __forceinline__ u32 h2(float lo, float hi) {
    u32 r;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

__device__ __forceinline__ float2 h2f(u32 v) {
    float lo, hi;
    asm("{\n\t.reg .f16 a, b;\n\tmov.b32 {a, b}, %2;\n\tcvt.f32.f16 %0, a;\n\tcvt.f32.f16 %1, b;\n\t}"
        : "=f"(lo), "=f"(hi)
        : "r"(v));
    return make_float2(lo, hi);
}

__device__ __forceinline__ float hf(const unsigned short* s) {
    float f;
    asm("{\n\t.reg .f16 a;\n\tmov.b16 a, %1;\n\tcvt.f32.f16 %0, a;\n\t}" : "=f"(f) : "h"(*s));
    return f;
}

__device__ __forceinline__ unsigned short f2h(float f) {
    unsigned short r;
    asm("{\n\t.reg .f16 a;\n\tcvt.rn.f16.f32 a, %1;\n\tmov.b16 %0, a;\n\t}" : "=h"(r) : "f"(f));
    return r;
}

__device__ __forceinline__ void mma_h(float (&d)[4], const u32 (&a)[4], u32 b0, u32 b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldsm4(u32 (&r)[4], const unsigned short* at) {
    const u32 addr = static_cast<u32>(__cvta_generic_to_shared(at));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void ldsm4t(u32 (&r)[4], const unsigned short* at) {
    const u32 addr = static_cast<u32>(__cvta_generic_to_shared(at));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

// A 16 x 16 A operand at rows m0.., columns k0.. of a row-major f16 tile.
__device__ __forceinline__ void frag_a(u32 (&a)[4], const unsigned short* s, int ld, int m0, int k0, int l) {
    ldsm4(a, s + (m0 + (l & 15)) * ld + k0 + ((l >> 4) << 3));
}

// The B operand of two eight-wide column tiles n0, n0 + 8 over k0..k0+16 of
// a row-major [k][n] f16 tile: b[0], b[1] the first, b[2], b[3] the second.
__device__ __forceinline__ void frag_b(u32 (&b)[4], const unsigned short* s, int ld, int k0, int n0, int l) {
    ldsm4t(b, s + (k0 + (l & 7) + (((l >> 3) & 1) << 3)) * ld + n0 + ((l >> 4) << 3));
}

// A 16 x 8 accumulator tile, negated, as f16 at Z[r0.., n0..].
__device__ __forceinline__ void put_h(const float (&d)[4], unsigned short* Z, int ld, int r0, int n0, float sc, int l) {
    const int g = l >> 2;
    const int t = l & 3;
    *reinterpret_cast<u32*>(Z + (r0 + g) * ld + n0 + 2 * t) = h2(d[0] * sc, d[1] * sc);
    *reinterpret_cast<u32*>(Z + (r0 + g + 8) * ld + n0 + 2 * t) = h2(d[2] * sc, d[3] * sc);
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

__device__ __forceinline__ void tma_store_3d(const tc5::TmaMap* map, const void* src, int x, int y) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%2, %3, %4}], [%1];" ::"l"(
            reinterpret_cast<tc5::u64>(map)),
        "r"(tc5::smem_addr(src)), "r"(x), "r"(y), "r"(0)
        : "memory");
}

__global__ void __launch_bounds__(384, 1) gdn_chunked_tc5(
    const __grid_constant__ tc5::TmaMap map_qkv, const __grid_constant__ tc5::TmaMap map_out,
    const Params p) {
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
    // How many chunks' K S^T products the state group has issued: a prep
    // group issues chunk c's k k^T after chunk c - 1's, as one MMA warp did.
    volatile int* ks_issued = reinterpret_cast<volatile int*>(bars + 30);

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
        *ks_issued = 0;
        bar_init_fence();
    }
    if (warp == 0) {
        tmem_alloc(tmem_slot, 512);
        tmem_relinquish();
    }
    tc_fence_before();
    __syncthreads();
    tc_fence_after();
    const u32 tmem = *tmem_slot;

    Walk w{static_cast<int>(blockIdx.x), 0, 0, {}, 0, 0};
    bool live = next_tile(p, lanes, w);

    const u32 id128 = idesc_dense(kF16Bf16, kF16Bf16, 128, 128);
    const u32 id64 = idesc_dense(kF16Bf16, kF16Bf16, 128, 64);
    const u32 idkv = id128 | (1u << 16);
    const u32 idv = id64 | (1u << 15);
    // Chunk c's K, Q and V by TMA into stage c & 1.
    auto load = [&](const Walk& wl, int c) {
        const int s = c & 1;
        const int hk = wl.h / group;
        const int y = row0 + wl.lane.t0 + wl.j * CH;
        unsigned char* kqb = smem + kKQ + s * kKQStage;
        bar_arrive_expect_tx(&kq_full[s], kKQStage);
        tma_load_3d(kqb, &kq_full[s], &map_qkv, K_dim + hk * 128, y, 0);
        tma_load_3d(kqb + 8192, &kq_full[s], &map_qkv, hk * 128, y, 0);
        tma_load_3d(kqb + 16384, &kq_full[s], &map_qkv, K_dim + hk * 128 + 64, y, 0);
        tma_load_3d(kqb + 24576, &kq_full[s], &map_qkv, hk * 128 + 64, y, 0);
        unsigned char* vb = smem + kV + s * kVStage;
        bar_arrive_expect_tx(&v_full[s], kVStage);
        tma_load_3d(vb, &v_full[s], &map_qkv, 2 * K_dim + wl.h * 128, y, 0);
        tma_load_3d(vb + 8192, &v_full[s], &map_qkv, 2 * K_dim + wl.h * 128 + 64, y, 0);
    };
    if (warp < 4 || warp >= 8) {
        // A prep group: the chunks of its parity. Thread i (by its TMEM lane)
        // holds row i of [K;Q] K^T: k k^T for i < 64, q k^T past it.
        if (live) {
            const int gq = warp < 4 ? 0 : 1;
            const u32 quad = static_cast<u32>(warp & 3);
            const int i128 = static_cast<int>(quad) * 32 + l;
            const u32 bar_id = 1 + gq * 2;
            unsigned char* prep = smem + kPrep + gq * kPrepBytes;
            unsigned short* Ah = reinterpret_cast<unsigned short*>(prep + kPrepA);
            unsigned short* Th = reinterpret_cast<unsigned short*>(prep + kPrepT);
            for (int e = i128; e < CH * LDH; e += 128) Th[e] = 0;
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
                if (i128 == 0) {
                    // [K;Q] K^T into this group's buffer, once the chunk lands.
                    const u32 kqb = sbase + kKQ + s * kKQStage;
                    bar_wait(&kq_full[s], ph);
                    while (*ks_issued < c) {
                    }
                    tc_fence_after();
#pragma unroll
                    for (u32 k = 0; k < 8; ++k) {
                        const u64 d = sdesc_sw128(kqb + (k / 4) * 16384) + 2 * (k % 4);
                        mma_f16(tmem + kColKK + s * 64, d, d, id64, k > 0);
                    }
                    tc_commit(&kk_full[s]);
                }
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
                        for (int q = 0; q < 2; ++q) {
                            reinterpret_cast<uint4*>(Ah + i * LDH + jb * 16)[q] = make_uint4(
                                h2(av[8 * q], av[8 * q + 1]), h2(av[8 * q + 2], av[8 * q + 3]),
                                h2(av[8 * q + 4], av[8 * q + 5]), h2(av[8 * q + 6], av[8 * q + 7]));
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

                // T = (I + A)^-1 in f16 (its products accumulate in f32):
                // the eight eight-row diagonal blocks by forward substitution,
                // then three doubling levels, X = A_lo T_hi and T_lo = -T_d X.
                if (quad < 2) {
                    const int L64 = static_cast<int>(quad) * 32 + l;
                    const int b0 = (L64 >> 3) * 8;
                    const int col = L64 & 7;
                    float a[28];
#pragma unroll
                    for (int r = 1, e = 0; r < 8; ++r) {
#pragma unroll
                        for (int m = 0; m < r; ++m, ++e) a[e] = hf(Ah + (b0 + r) * LDH + b0 + m);
                    }
                    float t[8];
#pragma unroll
                    for (int r = 0, e = 0; r < 8; ++r) {
                        float sum = 0.f;
#pragma unroll
                        for (int m = 0; m < r; ++m, ++e) sum = __fmaf_rn(a[e], t[m], sum);
                        t[r] = r == col ? 1.f : (r < col ? 0.f : -sum);
                    }
#pragma unroll
                    for (int r = 0; r < 8; ++r) Th[(b0 + r) * LDH + b0 + col] = f2h(t[r]);
                }
                named_sync(bar_id, 128);
                {
                    // 8 -> 16, a warp a sixteen-row block, in f32 FMAs.
                    const int d0 = static_cast<int>(quad) * 16;
                    float* X = reinterpret_cast<float*>(prep + kPrepX) + quad * 64;
#pragma unroll
                    for (int h2i = 0; h2i < 2; ++h2i) {
                        const int e = l + 32 * h2i;
                        const int r = e >> 3;
                        const int c = e & 7;
                        float sum = 0.f;
#pragma unroll
                        for (int m = 0; m < 8; ++m) {
                            if (m >= c) {
                                sum = __fmaf_rn(hf(Ah + (d0 + 8 + r) * LDH + d0 + m),
                                                hf(Th + (d0 + m) * LDH + d0 + c), sum);
                            }
                        }
                        X[e] = sum;
                    }
                    __syncwarp();
#pragma unroll
                    for (int h2i = 0; h2i < 2; ++h2i) {
                        const int e = l + 32 * h2i;
                        const int r = e >> 3;
                        const int c = e & 7;
                        float sum = 0.f;
#pragma unroll
                        for (int m = 0; m < 8; ++m) {
                            if (m <= r) {
                                sum = __fmaf_rn(hf(Th + (d0 + 8 + r) * LDH + d0 + 8 + m), X[m * 8 + c], sum);
                            }
                        }
                        Th[(d0 + 8 + r) * LDH + d0 + c] = f2h(-sum);
                    }
                }
                named_sync(bar_id, 128);
                unsigned short* Xh = reinterpret_cast<unsigned short*>(prep + kPrepX);
                if (quad < 2) {
                    // 16 -> 32: warp b the block at rows 32b, X = A10 T0 then
                    // T10 = -T1 X, sixteen by sixteen each.
                    const int b0 = static_cast<int>(quad) * 32;
                    unsigned short* Xb = Xh + quad * (16 * 24);
                    u32 fa[4], fb[4];
                    float d0[4] = {0.f, 0.f, 0.f, 0.f}, d1[4] = {0.f, 0.f, 0.f, 0.f};
                    frag_a(fa, Ah, LDH, b0 + 16, b0, l);
                    frag_b(fb, Th, LDH, b0, b0, l);
                    mma_h(d0, fa, fb[0], fb[1]);
                    mma_h(d1, fa, fb[2], fb[3]);
                    put_h(d0, Xb, 24, 0, 0, 1.f, l);
                    put_h(d1, Xb, 24, 0, 8, 1.f, l);
                    __syncwarp();
                    float e0[4] = {0.f, 0.f, 0.f, 0.f}, e1[4] = {0.f, 0.f, 0.f, 0.f};
                    frag_a(fa, Th, LDH, b0 + 16, b0 + 16, l);
                    frag_b(fb, Xb, 24, 0, 0, l);
                    mma_h(e0, fa, fb[0], fb[1]);
                    mma_h(e1, fa, fb[2], fb[3]);
                    put_h(e0, Th, LDH, b0 + 16, b0, -1.f, l);
                    put_h(e1, Th, LDH, b0 + 16, b0 + 8, -1.f, l);
                }
                named_sync(bar_id, 128);
                {
                    // 32 -> 64: warp w the X tile at rows 16(w>>1), columns
                    // 16(w&1), then the same tile of T_lo.
                    const int mt = static_cast<int>(quad >> 1) * 16;
                    const int nh = static_cast<int>(quad & 1) * 16;
                    u32 fa[4], fb[4];
                    float d0[4] = {0.f, 0.f, 0.f, 0.f}, d1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                    for (int k = 0; k < 2; ++k) {
                        frag_a(fa, Ah, LDH, 32 + mt, 16 * k, l);
                        frag_b(fb, Th, LDH, 16 * k, nh, l);
                        mma_h(d0, fa, fb[0], fb[1]);
                        mma_h(d1, fa, fb[2], fb[3]);
                    }
                    put_h(d0, Xh, 40, mt, nh, 1.f, l);
                    put_h(d1, Xh, 40, mt, nh + 8, 1.f, l);
                    named_sync(bar_id, 128);
                    float e0[4] = {0.f, 0.f, 0.f, 0.f}, e1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                    for (int k = 0; k < 2; ++k) {
                        frag_a(fa, Th, LDH, 32 + mt, 32 + 16 * k, l);
                        frag_b(fb, Xh, 40, 16 * k, nh, l);
                        mma_h(e0, fa, fb[0], fb[1]);
                        mma_h(e1, fa, fb[2], fb[3]);
                    }
                    put_h(e0, Th, LDH, 32 + mt, nh, -1.f, l);
                    put_h(e1, Th, LDH, 32 + mt, nh + 8, -1.f, l);
                }
                named_sync(bar_id, 128);
                {
                    // T' = T diag(beta) in bf16, the B operand of NV = T' U:
                    // two threads a row, 32 columns each.
                    const int i = i128 >> 1;
                    const int hh = i128 & 1;
                    unsigned char* tb = smem + kT + s * 8192;
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int cj = hh * 4 + q;
                        const uint4 t4 = *reinterpret_cast<const uint4*>(Th + i * LDH + cj * 8);
                        const float4 b0 = reinterpret_cast<const float4*>(B)[2 * cj];
                        const float4 b1 = reinterpret_cast<const float4*>(B)[2 * cj + 1];
                        const float2 x0 = h2f(t4.x), x1 = h2f(t4.y), x2 = h2f(t4.z), x3 = h2f(t4.w);
                        *reinterpret_cast<uint4*>(tb + i * 128 + (((cj ^ (i & 7))) << 4)) = make_uint4(
                            gdn_pack(x0.x * b0.x, x0.y * b0.y), gdn_pack(x1.x * b0.z, x1.y * b0.w),
                            gdn_pack(x2.x * b1.x, x2.y * b1.y), gdn_pack(x3.x * b1.z, x3.y * b1.w));
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
            // Thread 0 loads one chunk ahead: chunks 0 and 1 now, chunk
            // c + 1 once chunk c - 1's products are done with its stage.
            Walk wl = w;
            bool lmore = true;
            if (v == 0) {
                tma_prefetch(&map_qkv);
                load(wl, 0);
                lmore = advance(p, lanes, wl);
                if (lmore) load(wl, 1);
            }
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
                if (v == 0 && c >= 1 && lmore) {
                    lmore = advance(p, lanes, wl);
                    if (lmore) load(wl, c + 1);
                }
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
                named_sync(5, 128);
                if (v == 0) {
                    // [K;Q] S^T, the state's bf16 copy in TMEM as A.
                    const u32 kqb = sbase + kKQ + s * kKQStage;
                    tc_fence_after();
                    bar_wait(&kq_full[s], ph);
#pragma unroll
                    for (u32 k = 0; k < 8; ++k) {
                        mma_f16_ts(tmem + kColKS, tmem + kColSI + k * 8,
                                   sdesc_sw128(kqb + (k / 4) * 16384) + 2 * (k % 4), id128, k > 0);
                    }
                    tc_commit(ksqs_full);
                    *ks_issued = c + 1;
                }
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
                    named_sync(5, 128);
                    if (v == 0) {
                        // NV = T' V + T' (-diag(e^G rk) K S^T), over K S^T.
                        tc_fence_after();
                        bar_wait(&t_full[s], ph);
                        bar_wait(&v_full[s], ph);
                        const u32 tbs = sbase + kT + s * 8192;
                        const u32 vb = sbase + kV + s * kVStage;
#pragma unroll
                        for (u32 k = 0; k < 4; ++k) {
                            mma_f16(tmem + kColKS, sdesc_sw128_mn(vb + k * 2048, 8192),
                                    sdesc_sw128(tbs) + 2 * k, idv, k > 0);
                        }
#pragma unroll
                        for (u32 k = 0; k < 4; ++k) {
                            mma_f16_ts(tmem + kColKS, tmem + kColU + k * 8, sdesc_sw128(tbs) + 2 * k,
                                       id64, true);
                        }
                        tc_commit(nv_full);
                    }
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
                named_sync(5, 128);
                if (v == 0) {
                    // S^T += (decayed NV)^T K, then O += P NV.
                    tc_fence_after();
                    const u32 kqb = sbase + kKQ + s * kKQStage;
                    const u32 qkb = sbase + kQK + s * 8192;
#pragma unroll
                    for (u32 k = 0; k < 4; ++k) {
                        mma_f16_ts(tmem + kColS, tmem + kColD + k * 8,
                                   sdesc_sw128_mn(kqb + k * 2048, 16384), idkv, true);
                    }
                    tc_commit(kv_done);
#pragma unroll
                    for (u32 k = 0; k < 4; ++k) {
                        mma_f16_ts(tmem + kColQS, tmem + kColU + k * 8, sdesc_sw128(qkb) + 2 * k,
                                   id64, true);
                    }
                    tc_commit(o_full);
                    tc_commit(&t_empty[s]);
                }

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
                    if (L == CH) {
                        // A whole chunk: staged [token][value] and stored by
                        // TMA, the last chunk's store done reading first.
                        if (v == 0) tc5::bulk_wait_read<0>();
                        named_sync(5, 128);
                        float* ob = reinterpret_cast<float*>(smem + kO);
#pragma unroll
                        for (int t = 0; t < 32; ++t) ob[t * 128 + v] = __uint_as_float(o0[t]);
#pragma unroll
                        for (int t = 0; t < 32; ++t) ob[(32 + t) * 128 + v] = __uint_as_float(o1[t]);
                        fence_async_shared();
                        named_sync(5, 128);
                        if (v == 0) {
                            tma_store_3d(&map_out, ob, w.h * 128, row0 + ta);
                            tc5::bulk_commit();
                        }
                    } else {
                        float* orow =
                            p.out + static_cast<long long>(row0 + ta) * stride_row + w.h * 128 + v;
#pragma unroll
                        for (int t = 0; t < 32; ++t) {
                            if (t < L) orow[static_cast<long long>(t) * stride_row] = __uint_as_float(o0[t]);
                        }
#pragma unroll
                        for (int t = 0; t < 32; ++t) {
                            if (32 + t < L)
                                orow[static_cast<long long>(32 + t) * stride_row] =
                                    __uint_as_float(o1[t]);
                        }
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
    if (tid == 128) tc5::bulk_wait_all();
    tc_fence_before();
    __syncthreads();
    if (warp == 0) {
        tc_fence_after();
        tmem_dealloc(tmem, 512);
    }
}

}  // namespace pie::attn::gdn
