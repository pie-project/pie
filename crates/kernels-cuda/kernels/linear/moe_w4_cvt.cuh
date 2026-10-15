#pragma once

// The register-dequant mxfp4 select (`linear/moe_w4.cuh`) on the datacenter
// Blackwells' e2m1 converts: each code byte becomes an f16 pair in one
// `cvt.rn.f16x2.e2m1x2`, the byte picked by the instruction, and one f16 mul
// lays the block's 2^(s - 127) on it, where the bf16 path spends four
// integer ops and a bf16 mul a pair. The activations meet them as f16.
//
// The lanes own the same code groups as there (lane (g, c) the group c of
// rows g and g + 8 in each 128-code chunk), but a byte's two codes are
// adjacent, so k-step (w, j) takes bytes 2j and 2j + 1 of word w and the
// activations land in their own order, an f16 pair from a bf16 pair.
//
// The activations turn f16 once a CTA, not once a warp: a window of
// `kWin` chunks lands in shared memory by cp.async in the order the lanes
// read it back (one 16-byte read a lane a word, no bank conflicts), each
// thread converts the entries it copied, and one barrier a window hands them
// over while the next window lands in the other buffer. The codes stream
// into registers `kDepth` chunks ahead, a warp's own (a ring in shared
// memory by cp.async measured slower).
//
// f16 holds what bf16 does only within range: a chunk whose scales leave
// 2^-14 .. 2^13 runs the bf16 path's arithmetic instead, its activations
// read again from global memory (gpt-oss's scales sit in 2^-16 .. 2^11, all
// but one block in ten million inside), and a warp whose activations pass
// f16's walks again on the bf16 path alone.

#include "linear/moe_w4.cuh"

namespace pie::linear {

namespace w4cvt {

__device__ __forceinline__ void mma(float (&d)[4], unsigned a0, unsigned a1, unsigned a2,
                                    unsigned a3, unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

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
                    mma(acc[t][b], a0, a1, a2, a3, j == 0 ? xf[b].x : xf[b].z,
                        j == 0 ? xf[b].y : xf[b].w);
            }
        }
    }
}

}  // namespace w4cvt

/// `moe_select_mxfp4_mma` with the e2m1 converts: the same operands (the
/// scales row-major, or with `scale_atoms` in tcgen05's atoms), grid and
/// landing, and `kNb * kWin * 4 KiB` of dynamic shared memory (two
/// windows of `kNb` 8-route blocks' f16 activations).
template <int kNb, int kGlu, int kDepth, int kWin, int kMinBlocks>
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
    int scale_atoms)
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
    if (static_cast<int>(blockIdx.x) >= offsets[num_experts + 1]) return;
    const int item = work[blockIdx.x];
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

    const int groups = k >> 5;
    const int chunks = (groups + 3) >> 2;
    const int windows = (chunks + kWin - 1) / kWin;
    const long long row_bytes = k >> 1;
    const u8* w_e = codes + static_cast<long long>(expert) * n * row_bytes;
    const u8* s_e = scales + static_cast<long long>(expert) * n * groups;
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
                const long long at = static_cast<long long>(expert) * n + r;
                srow[t][h] = scales + (at / 128) * ((groups + 3) / 4) * 512 + (at % 32) * 16
                    + ((at / 32) % 4) * 4 + c;
            } else {
                srow[t][h] = s_e + static_cast<long long>(r) * groups + c;
            }
        }
    const bf16* xrow[kNb];
#pragma unroll
    for (int b = 0; b < kNb; ++b) {
        const int tok = 8 * b + g;
        xrow[b] = tok < count
            ? act + static_cast<long long>(order[begin + tok] / act_div) * k + c * 32
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
    if (!rows) return;

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

    const bf16* b_e = bias != nullptr ? bias + static_cast<long long>(expert) * n : nullptr;
    if constexpr (kGlu != 0) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = row_base + 8 * h + g;
            if (col >= inter) continue;
            const float bg = b_e != nullptr ? bf16_to_f32(b_e[col]) : 0.f;
            const float bu = b_e != nullptr ? bf16_to_f32(b_e[inter + col]) : 0.f;
#pragma unroll
            for (int b = 0; b < kNb; ++b)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int tok = 8 * b + 2 * c + e;
                    if (tok < count) {
                        const long long route = order[begin + tok];
                        out[route * inter + col] = f32_to_bf16(w4mma::glu<kGlu>(
                            acc[0][b][2 * h + e] + bg, acc[1][b][2 * h + e] + bu, limit, galpha));
                    }
                }
        }
        return;
    }
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = row_base + 16 * t + 8 * h + g;
            if (row >= n) continue;
            const float add = b_e != nullptr ? bf16_to_f32(b_e[row]) : 0.f;
#pragma unroll
            for (int b = 0; b < kNb; ++b)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int tok = 8 * b + 2 * c + e;
                    if (tok < count) {
                        const long long route = order[begin + tok];
                        out[route * n + row] = f32_to_bf16(acc[t][b][2 * h + e] + add);
                    }
                }
        }
}

}  // namespace pie::linear
