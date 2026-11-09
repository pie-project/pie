#include "linear/codec.cuh"
#include "linear/moe_w4.cuh"
#include "linear/moe/table.cuh"
#include "linear/quant.cuh"
#include "prim/mma.cuh"

namespace pie::linear {

namespace w4cvt {

/// The four code bytes of `q` as f16 pairs, the low nibble the low half.
__device__ __forceinline__ void e2m1x8(unsigned q, unsigned (&r)[4]) {
    asm("{ .reg .b8 b0, b1, b2, b3;\n"
        "mov.b32 {b0, b1, b2, b3}, %4;\n"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
        "cvt.rn.f16x2.e2m1x2 %1, b1;\n"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\n"
        "cvt.rn.f16x2.e2m1x2 %3, b3; }\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(q));
}

__device__ __forceinline__ unsigned hmul2(unsigned a, unsigned b) {
    unsigned d;
    asm("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(d) : "r"(a), "r"(b));
    return d;
}

/// A bf16 pair as an f16 pair (exact for |x| < 2^15 down to f16's
/// subnormals).
__device__ __forceinline__ unsigned f16_of_bf16(unsigned x) {
    const float lo = __uint_as_float(x << 16);
    const float hi = __uint_as_float(x & 0xFFFF0000u);
    unsigned d;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(hi), "f"(lo));
    return d;
}

/// `at` in cluster rank `rank`'s shared memory, read.
__device__ __forceinline__ float ld_rank(const float* at, unsigned rank) {
    unsigned remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(remote)
                 : "r"(static_cast<unsigned>(__cvta_generic_to_shared(at))), "r"(rank));
    float v;
    asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(remote));
    return v;
}

__device__ __forceinline__ void sync_ranks() {
    asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::
                     : "memory");
}

__device__ __forceinline__ unsigned rank_in_cluster() {
    unsigned r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

/// 16 bytes from `gmem` into `smem` by cp.async, or 16 zeros (nothing
/// read) where `live` is false.
__device__ __forceinline__ void cp16_or_zero(void* smem, const void* gmem, bool live) {
    const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(gmem),
                 "r"(live ? 16 : 0));
}

__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }

/// Every cp.async group of this thread's but the newest `kPending` landed.
template <int kPending>
__device__ __forceinline__ void wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(kPending));
}

/// A chunk's codes and scales a lane holds.
template <int kTiles>
struct Codes {
    uint4 w[kTiles][2];
    unsigned s[kTiles][2];
};

/// Whether a chunk's scales keep to f16: each 2^-14 .. 2^13, an f16
/// normal that no code times overflows.
template <int kTiles>
__device__ __forceinline__ bool fits(const Codes<kTiles>& cc) {
    bool ok = true;
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
#pragma unroll
        for (int h = 0; h < 2; ++h) ok &= cc.s[t][h] - 113u <= 27u;
    return ok;
}

/// One chunk on the f16 path: the codes from registers, the activations
/// from the window (`xs` at the chunk's place, its words `kWin * 4 * 32`
/// apart a block of 8 routes).
template <int kTiles, int kNb, int kWin>
__device__ __forceinline__ void compute(const Codes<kTiles>& cc, const uint4* xs, int lane,
                                        float (&acc)[kTiles][kNb][4]) {
    // 2^(s - 127) as an f16 pair: exponent field s - 112.
    unsigned sc[kTiles][2];
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const unsigned e = (cc.s[t][h] - 112u) << 10;
            sc[t][h] = e | (e << 16);
        }
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        uint4 xf[kNb];
#pragma unroll
        for (int b = 0; b < kNb; ++b) xf[b] = xs[(b * kWin * 4 + w) * 32 + lane];
#pragma unroll
        for (int t = 0; t < kTiles; ++t) {
            unsigned r0[4], r1[4];
            e2m1x8((&cc.w[t][0].x)[w], r0);
            e2m1x8((&cc.w[t][1].x)[w], r1);
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const unsigned a0 = hmul2(r0[2 * j], sc[t][0]);
                const unsigned a1 = hmul2(r1[2 * j], sc[t][1]);
                const unsigned a2 = hmul2(r0[2 * j + 1], sc[t][0]);
                const unsigned a3 = hmul2(r1[2 * j + 1], sc[t][1]);
#pragma unroll
                for (int b = 0; b < kNb; ++b)
                    mma_m16n8k16_f16(acc[t][b], {a0, a1, a2, a3}, {j == 0 ? xf[b].x : xf[b].z, j == 0 ? xf[b].y : xf[b].w});
            }
        }
    }
}

}

/// `moe_select_mxfp4_mma` with the e2m1 converts: the same operands (the
/// scales row-major, or with `scale_atoms` in tcgen05's atoms), grid and
/// landing, and `kNb * kWin * 4 KiB` of dynamic shared memory (two
/// windows of `kNb` 8-route blocks' f16 activations).
///
/// With `kKSplit` above 1 a work item's K splits across a cluster of that
/// many CTAs along x (rank r its r-th share of the chunks), and rank 0 sums
/// the others' through the cluster's shared memory before it lands the
/// rows: more CTAs streaming the bank where a decode's few routes leave
/// most SMs idle.
template <int kNb, int kGlu, int kDepth, int kWin, int kMinBlocks, int kKSplit = 1>
__global__ void __launch_bounds__(128, kMinBlocks) moe_select_mxfp4_cvt(
    const bf16* __restrict__ act,
    const i32* __restrict__ order,
    const i32* __restrict__ offsets,
    const i32* __restrict__ work,
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    const bf16* __restrict__ bias,
    bf16* __restrict__ out,
    int act_div,
    int n,
    int k,
    int num_experts,
    int inter,
    float limit,
    float galpha,
    int scale_atoms,
    const u64* __restrict__ expert_table)
{
    constexpr int kTiles = 2;
    constexpr int kRowsWarp = kGlu != 0 ? 16 : kTiles * 16;
    constexpr int kRowsCta = 4 * kRowsWarp;
    constexpr int kSlice = 8 * kNb;
    // A window's entries: (block, chunk, word) x 32 lanes, 16 bytes each.
    constexpr int kWinWords = kNb * kWin * 4 * 32;
    static_assert(kWin % kDepth == 0, "a window holds whole rings");
    extern __shared__ uint4 w4cvt_xs[];

    // A route that names no expert lands zeros (the first CTA writes them).
    if (blockIdx.x == 0 && blockIdx.y == 0) {
        const int width = kGlu != 0 ? inter : n;
        for (int i = offsets[num_experts]; i < offsets[num_experts + 2]; ++i) {
            bf16* row = out + static_cast<long long>(order[i]) * width;
            for (int j = static_cast<int>(threadIdx.x); j < width; j += blockDim.x)
                row[j] = f32_to_bf16(0.f);
        }
    }
    const int item_at = static_cast<int>(blockIdx.x) / kKSplit;
    if (item_at >= offsets[num_experts + 1]) return;
    const int item = work[item_at];
    const int expert = item >> 16;
    const int slice = item & 0xFFFF;
    const int begin = offsets[expert] + slice * kSlice;
    const int count = min(offsets[expert + 1] - begin, kSlice);
    if (count <= 0) return;

    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int g = lane >> 2;
    const int c = lane & 3;
    const int row_base = static_cast<int>(blockIdx.y) * kRowsCta + warp * kRowsWarp;
    const int span = kGlu != 0 ? inter : n;
    auto tile_row = [&](int t) { return kGlu != 0 ? row_base + t * inter : row_base + 16 * t; };
    // A warp past the bank's rows still lands and converts its share of the
    // activations: the window barriers count it.
    const bool rows = row_base < span;

    const int g_all = k >> 5;
    const int chunks_all = (g_all + 3) >> 2;
    // This rank's share of the chunks, from chunk `cb` on.
    const int rank = kKSplit > 1 ? static_cast<int>(w4cvt::rank_in_cluster()) : 0;
    const int share = (chunks_all + kKSplit - 1) / kKSplit;
    const int cb = min(rank * share, chunks_all);
    const int chunks = min(chunks_all - cb, share);
    // The code groups from the share's first chunk on.
    const int groups = g_all - cb * 4;
    const int windows = (chunks + kWin - 1) / kWin;
    const long long row_bytes = k >> 1;
    const u8* w_e = expert_plane(codes, 1LL * n * row_bytes, expert_table, 3, 0, expert);
    const u8* s_e = expert_plane(scales, 1LL * n * g_all, expert_table, 3, 1, expert);
    // A chunk's scales: the next 4 bytes of a row-major plane, or the next
    // 512-byte atom column of the tcgen05 layout (`scale_atoms`, the bank's
    // rows counted across its experts).
    const int s_step = scale_atoms != 0 ? 512 : 4;

    const u8* wrow[kTiles][2];
    const u8* srow[kTiles][2];
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = min(tile_row(t) + 8 * h + g, n - 1);
            wrow[t][h] = w_e + r * row_bytes + c * 16;
            if (scale_atoms != 0) {
                // Atoms over the bank's rows; a table's entry is the expert's
                // first atom (its rows a whole number of 128).
                const long long at = expert_table != nullptr
                    ? static_cast<long long>(r)
                    : static_cast<long long>(expert) * n + r;
                srow[t][h] = (expert_table != nullptr ? s_e : scales)
                    + (at / 128) * ((g_all + 3) / 4) * 512 + (at % 32) * 16
                    + ((at / 32) % 4) * 4 + c;
            } else {
                srow[t][h] = s_e + static_cast<long long>(r) * g_all + c;
            }
            // From this rank's first chunk on.
            wrow[t][h] += cb * 64;
            srow[t][h] += cb * s_step;
        }
    const bf16* xrow[kNb];
#pragma unroll
    for (int b = 0; b < kNb; ++b) {
        const int tok = 8 * b + g;
        xrow[b] = tok < count
            ? act + static_cast<long long>(order[begin + tok] / act_div) * k + c * 32 + cb * 128
            : nullptr;
    }

    using Codes = w4cvt::Codes<kTiles>;
    auto load = [&](int ch, Codes& cc) {
        const bool live = ch * 4 + c < groups;
#pragma unroll
        for (int t = 0; t < kTiles; ++t)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                cc.w[t][h] = live ? w4mma::ld_stream(wrow[t][h] + ch * 64) : make_uint4(0, 0, 0, 0);
                cc.s[t][h] = live ? w4mma::ld_scale(srow[t][h] + ch * s_step) : 127u;
            }
    };
    // This thread's entries of a window: word `warp` of each chunk and block,
    // for its own lane (the lane that reads them back is another warp's).
    auto stage = [&](int wi) {
        uint4* buf = w4cvt_xs + (wi & 1) * kWinWords;
#pragma unroll
        for (int b = 0; b < kNb; ++b)
#pragma unroll
            for (int i = 0; i < kWin; ++i) {
                const int ch = wi * kWin + i;
                const bool live = xrow[b] != nullptr && ch < chunks && ch * 4 + c < groups;
                w4cvt::cp16_or_zero(buf + ((b * kWin + i) * 4 + warp) * 32 + lane,
                                    live ? xrow[b] + ch * 128 + warp * 8 : act, live);
            }
        w4cvt::commit();
    };
    auto convert = [&](int wi) {
        uint4* buf = w4cvt_xs + (wi & 1) * kWinWords;
#pragma unroll
        for (int b = 0; b < kNb; ++b)
#pragma unroll
            for (int i = 0; i < kWin; ++i) {
                uint4* at = buf + ((b * kWin + i) * 4 + warp) * 32 + lane;
                uint4 v = *at;
                v.x = w4cvt::f16_of_bf16(v.x);
                v.y = w4cvt::f16_of_bf16(v.y);
                v.z = w4cvt::f16_of_bf16(v.z);
                v.w = w4cvt::f16_of_bf16(v.w);
                *at = v;
            }
    };
    // The bf16 path for one chunk, its activations read again.
    auto fallback = [&](int ch, const Codes& cc, float (&acc)[kTiles][kNb][4]) {
        w4mma::Chunk<kTiles, kNb> full;
        const bool live = ch * 4 + c < groups;
#pragma unroll
        for (int t = 0; t < kTiles; ++t)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                full.w[t][h] = cc.w[t][h];
                full.s[t][h] = cc.s[t][h];
            }
#pragma unroll
        for (int b = 0; b < kNb; ++b)
#pragma unroll
            for (int q = 0; q < 4; ++q)
                full.x[b][q] = live && xrow[b] != nullptr
                    ? w4mma::ld_act(xrow[b] + ch * 128 + q * 8)
                    : make_uint4(0, 0, 0, 0);
        bool wide = false;
#pragma unroll
        for (int t = 0; t < kTiles; ++t) wide |= cc.s[t][0] > 128u || cc.s[t][1] > 128u;
        if (__any_sync(0xffffffffu, wide)) {
            w4mma::compute<true, false>(full, acc);
        } else {
            w4mma::compute<false, false>(full, acc);
        }
    };

    float acc[kTiles][kNb][4];
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
#pragma unroll
        for (int b = 0; b < kNb; ++b)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[t][b][e] = 0.f;

    Codes ring[kDepth];
    if (rows) {
#pragma unroll
        for (int d = 0; d < kDepth; ++d)
            if (d < chunks) load(d, ring[d]);
    }
    stage(0);
    for (int wi = 0; wi < windows; ++wi) {
        // This thread's copies of window wi have landed (the only group in
        // flight); converted, a barrier hands them to the other warps and
        // frees the other buffer for window wi + 1.
        w4cvt::wait<0>();
        convert(wi);
        __syncthreads();
        if (wi + 1 < windows) stage(wi + 1);
        if (!rows) continue;
        const uint4* buf = w4cvt_xs + (wi & 1) * kWinWords;
        const int last = min(chunks, (wi + 1) * kWin);
        // Chunk ch's ring slot is ch % kDepth: a window of whole rings keeps
        // it static.
        for (int ch0 = wi * kWin; ch0 < last; ch0 += kDepth) {
#pragma unroll
            for (int d = 0; d < kDepth; ++d) {
                const int ch = ch0 + d;
                if (ch >= last) break;
                Codes& cc = ring[d];
                if (__all_sync(0xffffffffu, w4cvt::fits(cc))) {
                    w4cvt::compute<kTiles, kNb, kWin>(cc, buf + (ch - wi * kWin) * 4 * 32, lane,
                                                      acc);
                } else {
                    fallback(ch, cc, acc);
                }
                if (ch + kDepth < chunks) load(ch + kDepth, cc);
            }
        }
    }
    w4cvt::wait<0>();

    // An activation past f16's range turns the sums non-finite: the warp
    // walks K again on the bf16 path alone.
    bool finite = true;
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
#pragma unroll
        for (int b = 0; b < kNb; ++b)
#pragma unroll
            for (int e = 0; e < 4; ++e) finite &= isfinite(acc[t][b][e]);
    if (!__all_sync(0xffffffffu, finite)) {
#pragma unroll
        for (int t = 0; t < kTiles; ++t)
#pragma unroll
            for (int b = 0; b < kNb; ++b)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[t][b][e] = 0.f;
        for (int ch = 0; ch < chunks; ++ch) {
            Codes cc;
            load(ch, cc);
            fallback(ch, cc, acc);
        }
    }

    // A split K's shares: each rank's sums in its shared memory (the
    // windows', all read), rank 0 adding the others' and landing the rows,
    // the others held until it has read them.
    if constexpr (kKSplit > 1) {
        __syncthreads();
        constexpr int kPer = kTiles * kNb * 4;
        float* red = reinterpret_cast<float*>(w4cvt_xs) + warp * kPer * 32;
        if (rows) {
#pragma unroll
            for (int t = 0; t < kTiles; ++t)
#pragma unroll
                for (int b = 0; b < kNb; ++b)
#pragma unroll
                    for (int e = 0; e < 4; ++e) red[((t * kNb + b) * 4 + e) * 32 + lane] = acc[t][b][e];
        }
        w4cvt::sync_ranks();
        if (rank == 0 && rows) {
#pragma unroll
            for (int t = 0; t < kTiles; ++t)
#pragma unroll
                for (int b = 0; b < kNb; ++b)
#pragma unroll
                    for (int e = 0; e < 4; ++e)
#pragma unroll
                        for (unsigned q = 1; q < kKSplit; ++q)
                            acc[t][b][e] += w4cvt::ld_rank(red + ((t * kNb + b) * 4 + e) * 32 + lane, q);
        }
        w4cvt::sync_ranks();
        if (rank != 0) return;
    }
    if (!rows) return;

    const bf16* b_e = bias != nullptr ? expert_plane(bias, 1LL * n, expert_table, 3, 2, expert) : nullptr;
    w4mma::w4_land<kGlu, kTiles, kNb>(acc, [](int) { return 1.f; }, b_e, order, begin, count, out,
                                      n, inter, row_base, g, c, limit, galpha);
}

// `moe_matmul_select_mxfp4` (the decode arm: a block a route, `kRows` rows a
// warp) with each 32-code block widened by the e2m1 converts rather than
// byte permutes: the same f16 codes and products in the same order, so the
// same bits, at fewer ALU slots a code (gpt-oss-20b's routed projections at
// one token on a B200: 18.2 and 10.7 us to 14.5 and 8.6).
template <class T, int kRows>
__global__ void moe_matmul_select_mxfp4_cvt(
    const T* __restrict__ act,
    const i32* __restrict__ routes,
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    const T* __restrict__ bias,
    T* __restrict__ out,
    int top_k,
    int act_div,
    int n,
    int k,
    const MoeGroupBases* __restrict__ bases,
    unsigned int* __restrict__ group_hits,
    const u32* __restrict__ win,
    const u64* __restrict__ expert_table)
{
    const int route = blockIdx.x;

    if (win != nullptr && route >= static_cast<int>(win[0]) * top_k) return;

    const int plane_route = win != nullptr ? route + static_cast<int>(win[1]) * top_k : route;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int row0 = (blockIdx.y * (blockDim.x >> 5) + warp_in_block) * kRows;
    if (row0 >= n) return;
    const int expert = routes[plane_route];

    moe_note_group(group_hits);

    const int groups_per_row = k / 32;

    const u8* codes_at = codes;
    const u8* scales_at = scales;
    if (bases != nullptr) {
        const MoeGroupBases seat = *bases;
        codes_at = seat.codes;
        scales_at = seat.scales;
    }

    const u8* w = expert_plane(codes_at, 1LL * n * (k / 2), expert_table, 3, 0, expert);
    const u8* s = expert_plane(scales_at, 1LL * n * groups_per_row, expert_table, 3, 1, expert);

    const T* x = act + static_cast<long long>(plane_route / act_div) * k;

    int row_of[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) row_of[r] = min(row0 + r, n - 1);

    constexpr int kU = 2;
    const uint4* wq = reinterpret_cast<const uint4*>(w);
    const uint4* x4 = reinterpret_cast<const uint4*>(x);

    float acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] = 0.f;

    for (int g0 = lane_id; g0 < groups_per_row; g0 += 32 * kU) {
        uint4 ww[kU][kRows];
        u8 sc[kU][kRows];
#pragma unroll
        for (int u = 0; u < kU; ++u) {
            const int g = min(g0 + 32 * u, groups_per_row - 1);
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                const long long at = static_cast<long long>(row_of[r]) * groups_per_row + g;
                ww[u][r] = wq[at];
                sc[u][r] = s[at];
            }
        }
#pragma unroll
        for (int u = 0; u < kU; ++u) {
            const int g = g0 + 32 * u;
            if (g >= groups_per_row) break;
            float part[kRows];
#pragma unroll
            for (int r = 0; r < kRows; ++r) part[r] = 0.f;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint4 xq = x4[g * 4 + q];
                const T* xe = reinterpret_cast<const T*>(&xq);
                float xv[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) xv[j] = Elem<T>::to_f32(xe[j]);
#pragma unroll
                for (int r = 0; r < kRows; ++r) {
                    unsigned qd[4];
                    w4cvt::e2m1x8((&ww[u][r].x)[q], qd);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&qd[j]));
                        part[r] = fmaf(f.x, xv[2 * j], part[r]);
                        part[r] = fmaf(f.y, xv[2 * j + 1], part[r]);
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < kRows; ++r)
                acc[r] = fmaf(part[r], mxfp4_block_scale(sc[u][r]), acc[r]);
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int r = 0; r < kRows; ++r)
            acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], off);
    }
    if (lane_id == 0) {
        const T* b = bias != nullptr
            ? expert_plane(bias, 1LL * n, expert_table, 3, 2, expert)
            : nullptr;
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const int row = row0 + r;
            if (row >= n) break;
            float v = acc[r];
            if (b != nullptr) v += Elem<T>::to_f32(b[row]);
            out[static_cast<long long>(plane_route) * n + row] = Elem<T>::from_f32(v);
        }
    }
}

// `moe_matmul_select_nvfp4` (a decode's few routes an expert, and NVFP4's
// dense product) with each code word widened by the e2m1 converts, as
// `moe_matmul_select_mxfp4_cvt` widens MXFP4's: the same bits.
template <class T, int kRowsT>
__global__ void moe_matmul_select_nvfp4_cvt(
    const T* __restrict__ act,
    const i32* __restrict__ routes,
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    const float* __restrict__ tensor_scales,
    T* __restrict__ out,
    int top_k,
    int act_div,
    int n,
    int k,
    const MoeGroupBases* __restrict__ bases,
    unsigned int* __restrict__ group_hits,
    const u32* __restrict__ win,
    const u64* __restrict__ expert_table)
{
    constexpr int kRows = kRowsT;
    const int route = blockIdx.x;

    if (win != nullptr && route >= static_cast<int>(win[0]) * top_k) return;

    const int plane_route = win != nullptr ? route + static_cast<int>(win[1]) * top_k : route;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int row0 = (blockIdx.y * (blockDim.x >> 5) + warp_in_block) * kRows;
    if (row0 >= n) return;
    const int expert = routes != nullptr ? routes[plane_route] : 0;
    T* y = out + static_cast<long long>(plane_route) * n;
    if (expert < 0) {
        if (lane_id == 0) {
#pragma unroll
            for (int r = 0; r < kRows; ++r)
                if (row0 + r < n) y[row0 + r] = Elem<T>::from_f32(0.f);
        }
        return;
    }

    moe_note_group(group_hits);

    // A 16-byte load is 32 codes, two groups of a row.
    const int chunks = k / 32;
    const int groups = k / 16;

    const u8* codes_at = codes;
    const u8* scales_at = scales;
    const float* tensor_at = tensor_scales;
    if (bases != nullptr) {
        const MoeGroupBases seat = *bases;
        codes_at = seat.codes;
        scales_at = seat.scales;
        tensor_at = reinterpret_cast<const float*>(seat.biases);
    }

    const uint4* wq = reinterpret_cast<const uint4*>(
        expert_plane(codes_at, 1LL * n * (k / 2), expert_table, 3, 0, expert));
    const unsigned short* s2 = reinterpret_cast<const unsigned short*>(
        expert_plane(scales_at, 1LL * n * groups, expert_table, 3, 1, expert));
    const float* g = expert_plane(tensor_at, 1LL * n, expert_table, 3, 2, expert);

    const T* x = act + static_cast<long long>(plane_route / act_div) * k;
    const uint4* x4 = reinterpret_cast<const uint4*>(x);

    int row_of[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) row_of[r] = min(row0 + r, n - 1);

    constexpr int kU = 2;
    float acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] = 0.f;

    for (int c0 = lane_id; c0 < chunks; c0 += 32 * kU) {
        uint4 ww[kU][kRows];
        unsigned short sc[kU][kRows];
#pragma unroll
        for (int u = 0; u < kU; ++u) {
            const int c = min(c0 + 32 * u, chunks - 1);
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                const long long at = static_cast<long long>(row_of[r]) * chunks + c;
                ww[u][r] = wq[at];
                sc[u][r] = s2[at];
            }
        }
#pragma unroll
        for (int u = 0; u < kU; ++u) {
            const int c = c0 + 32 * u;
            if (c >= chunks) break;
            float part[2][kRows];
#pragma unroll
            for (int r = 0; r < kRows; ++r) part[0][r] = part[1][r] = 0.f;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint4 xq = x4[c * 4 + q];
                const T* xe = reinterpret_cast<const T*>(&xq);
                float xv[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) xv[j] = Elem<T>::to_f32(xe[j]);
#pragma unroll
                for (int r = 0; r < kRows; ++r) {
                    unsigned qd[4];
                    w4cvt::e2m1x8((&ww[u][r].x)[q], qd);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&qd[j]));
                        part[q >> 1][r] = fmaf(f.x, xv[2 * j], part[q >> 1][r]);
                        part[q >> 1][r] = fmaf(f.y, xv[2 * j + 1], part[q >> 1][r]);
                    }
                }
            }
            constexpr float kBack = 1.f;
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                acc[r] = fmaf(part[0][r], nvfp4_scale(static_cast<u8>(sc[u][r] & 0xFFu)) * kBack, acc[r]);
                acc[r] = fmaf(part[1][r], nvfp4_scale(static_cast<u8>(sc[u][r] >> 8)) * kBack, acc[r]);
            }
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int r = 0; r < kRows; ++r)
            acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], off);
    }
    if (lane_id == 0) {
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const int row = row0 + r;
            if (row >= n) break;
            y[row] = Elem<T>::from_f32(acc[r] * g[row]);
        }
    }
}

}
