#pragma once

// A 3-D convolution over a voxel grid as an implicit GEMM: an output row (a
// voxel) against an output channel, K running over the taps and, within a
// tap, the input channels. The weight is `[C_out, taps·C_in]`, taps-major
// (`conv_weight_taps_major` relabels a checkpoint's channel-major one). A
// tap past the frame reads zero, or under `causal_t` with a cache the
// cache's frame, or under `replicate` the edge frame. bf16 in and out, f32
// accumulation, an optional f32 bias.
//
// Two kernels: `conv3d_direct`, f32 FMA over 64x64 tiles for any channel
// count; and `conv3d_mfma`, the CDNA matrix core over 64x64 tiles, for input
// channels in whole 8-element chunks on 16-byte planes.

#include "prim/device.hip"
#include "prim/mfma.hip"
#include "prim/voxels.hip"

namespace pie::conv {

using namespace ::pie::voxels;

struct ConvGeom {
    int c_in;
    int c_out;
    int kt;
    int kh;
    int kw;
    int st;
    int sh;
    int sw;
    // Front pads; the back pads only shape the output grid.
    int pt;
    int ph;
    int pw;
    // A frame before the first reads the cache (when one is bound) and
    // nothing past the last.
    int causal;
    // A frame outside the clip reads its edge frame instead of zero.
    int replicate;
    int lanes;
    int rows_out;

    // The rows `grid` and `o_grid` address `x`'s and `o`'s row 0 at: a
    // window of a segmented fire starts its tensors at its own rows while
    // the fire's grids hold fire-wide ones.
    int x_base;
    int o_base;
};

struct OutRow {
    Lane in;
    Voxel o;
    // The cache row of this lane's first cached frame.
    int cache_base;
    bool live;
};

__device__ __forceinline__ OutRow out_row(const ConvGeom& g, const int* __restrict__ grid,
                                          const int* __restrict__ o_grid, int m) {
    OutRow r;
    r.live = false;
    r.cache_base = 0;
    if (m >= g.rows_out) return r;
    Lane og;
    const int l = lane_of(o_grid, g.lanes, m + g.o_base, og);
    if (l < 0) return r;
    r.in = lane_at(grid, l);
    r.in.off -= g.x_base;
    r.o = unravel(og, m + g.o_base - og.off);
    // The cache holds `pt` frames a lane, lane after lane.
    int planes = 0;
    for (int j = 0; j < l; ++j) planes += lane_at(grid, j).plane();
    r.cache_base = planes * g.pt;
    r.live = true;
    return r;
}

// The input row tap (it, ih, iw) of output row `r` reads, or -1 for zero.
__device__ __forceinline__ int tap_row(const ConvGeom& g, const OutRow& r, int it, int ih, int iw, bool has_cache,
                                       bool& from_cache) {
    from_cache = false;
    const int hi = r.o.h * g.sh - g.ph + ih;
    const int wi = r.o.w * g.sw - g.pw + iw;
    if (hi < 0 || hi >= r.in.h || wi < 0 || wi >= r.in.w) return -1;
    int ti = r.o.t * g.st - g.pt + it;
    if (ti < 0) {
        if (g.causal && has_cache) {
            from_cache = true;
            return r.cache_base + ((ti + g.pt) * r.in.h + hi) * r.in.w + wi;
        }
        if (!g.replicate) return -1;
        ti = 0;
    }
    if (ti >= r.in.t) {
        // A causal convolution reads no frame past the clip's last.
        if (g.causal || !g.replicate) return -1;
        ti = r.in.t - 1;
    }
    return ravel(r.in, ti, hi, wi);
}

__device__ __forceinline__ void tap_of(const ConvGeom& g, int tap, int& it, int& ih, int& iw) {
    const int plane = g.kh * g.kw;
    it = tap / plane;
    const int rest = tap - it * plane;
    ih = rest / g.kw;
    iw = rest - ih * g.kw;
}

constexpr int kDirectTM = 64;
constexpr int kDirectTN = 64;
constexpr int kDirectBK = 32;
constexpr int kDirectThreads = 256;

// Eight channels from `c` of a row, zero past `c_in`; VEC reads them as one
// 16-byte access (C_in a multiple of 8 on 16-byte planes).
template <bool VEC>
__device__ __forceinline__ void gather8(const bf16* __restrict__ row, int c, int c_in, float (&out)[8]) {
    if constexpr (VEC) {
        if (c < c_in) {
            const Pack8<bf16> v = *reinterpret_cast<const Pack8<bf16>*>(row + c);
#pragma unroll
            for (int j = 0; j < 8; ++j) out[j] = to_f32(v.v[j]);
        }
    } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (c + j < c_in) out[j] = to_f32(row[c + j]);
        }
    }
}

// A 64x64 output tile a block of 256 threads, 4x4 a thread; K staged 32 at a
// time through LDS as f32.
template <bool VEC>
__global__ __launch_bounds__(kDirectThreads) void conv3d_direct(const bf16* __restrict__ x,
                                                                const int* __restrict__ grid,
                                                                const bf16* __restrict__ w,
                                                                const float* __restrict__ bias,
                                                                const bf16* __restrict__ cache, bf16* __restrict__ o,
                                                                const int* __restrict__ o_grid, ConvGeom g) {
    __shared__ __attribute__((aligned(16))) float As[kDirectBK][kDirectTM];
    __shared__ __attribute__((aligned(16))) float Bs[kDirectBK][kDirectTN];

    const int tid = static_cast<int>(threadIdx.x);
    const int m0 = static_cast<int>(blockIdx.x) * kDirectTM;
    const int n0 = static_cast<int>(blockIdx.y) * kDirectTN;

    // Staging: a thread loads 8 consecutive k of one A row and one B row.
    const int s_row = tid % 64;
    const int s_k = (tid / 64) * 8;
    const OutRow r = out_row(g, grid, o_grid, m0 + s_row);
    const bool has_cache = cache != nullptr;
    const int taps = g.kt * g.kh * g.kw;
    const int cchunks = (g.c_in + kDirectBK - 1) / kDirectBK;
    const int steps = taps * cchunks;
    const bool n_live = (n0 + s_row) < g.c_out;
    const bf16* wrow = w + static_cast<long long>(n_live ? n0 + s_row : 0) * taps * g.c_in;

    const int ty = tid / 16;
    const int tx = tid % 16;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
    }

    int src_row = -1;
    bool from_cache = false;
    for (int s = 0; s < steps; ++s) {
        const int tap = s / cchunks;
        const int c0 = (s - tap * cchunks) * kDirectBK;
        if (c0 == 0 && r.live) {
            int it, ih, iw;
            tap_of(g, tap, it, ih, iw);
            src_row = tap_row(g, r, it, ih, iw, has_cache, from_cache);
        }

        float a[8];
        float b[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            a[j] = 0.f;
            b[j] = 0.f;
        }
        if (r.live && src_row >= 0) {
            const bf16* row = (from_cache ? cache : x) + static_cast<long long>(src_row) * g.c_in;
            gather8<VEC>(row, c0 + s_k, g.c_in, a);
        }
        if (n_live) {
            gather8<VEC>(wrow + static_cast<long long>(tap) * g.c_in, c0 + s_k, g.c_in, b);
        }

        __syncthreads();
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            As[s_k + j][s_row] = a[j];
            Bs[s_k + j][s_row] = b[j];
        }
        __syncthreads();

#pragma unroll 8
        for (int k = 0; k < kDirectBK; ++k) {
            const float4 av = *reinterpret_cast<const float4*>(&As[k][ty * 4]);
            const float4 bv = *reinterpret_cast<const float4*>(&Bs[k][tx * 4]);
            const float ar[4] = {av.x, av.y, av.z, av.w};
            const float br[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(ar[i], br[j], acc[i][j]);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int m = m0 + ty * 4 + i;
        if (m >= g.rows_out) continue;
        Lane box;
        const bool live = lane_of(o_grid, g.lanes, m + g.o_base, box) >= 0;
        bf16* orow = o + static_cast<long long>(m) * g.c_out;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n = n0 + tx * 4 + j;
            if (n >= g.c_out) continue;
            float v = 0.f;
            if (live) {
                v = acc[i][j];
                if (bias != nullptr) v += bias[n];
            }
            orow[n] = from_f32<bf16>(v);
        }
    }
}

constexpr int kMfmaTM = 64;
constexpr int kMfmaTN = 64;
constexpr int kMfmaBK = 32;
constexpr int kMfmaThreads = 256;
// 8 elements of padding a row: an MFMA operand read's 16 rows land in
// distinct banks, and a row stays 16-byte aligned for the staging stores.
constexpr int kMfmaLd = kMfmaBK + 8;

// A 64x64 output tile a block of four wavefronts, 32x32 (2x2 MFMA tiles) a
// wavefront; K staged 32 at a time through LDS as bf16. Needs C_in % 8 == 0
// and 16-byte planes: an 8-channel chunk is then wholly in or out of range.
__global__ __launch_bounds__(kMfmaThreads) void conv3d_mfma(const bf16* __restrict__ x, const int* __restrict__ grid,
                                                            const bf16* __restrict__ w,
                                                            const float* __restrict__ bias,
                                                            const bf16* __restrict__ cache, bf16* __restrict__ o,
                                                            const int* __restrict__ o_grid, ConvGeom g) {
    __shared__ __attribute__((aligned(16))) bf16 As[kMfmaTM * kMfmaLd];
    __shared__ __attribute__((aligned(16))) bf16 Bs[kMfmaTN * kMfmaLd];
    __shared__ int row_live[kMfmaTM];

    const int tid = static_cast<int>(threadIdx.x);
    const int wave = tid / kWave;
    const int wave_m = wave >> 1;
    const int wave_n = wave & 1;
    const int m0 = static_cast<int>(blockIdx.x) * kMfmaTM;
    const int n0 = static_cast<int>(blockIdx.y) * kMfmaTN;
    const bool has_cache = cache != nullptr;
    const int taps = g.kt * g.kh * g.kw;
    const int cchunks = (g.c_in + kMfmaBK - 1) / kMfmaBK;
    const int steps = taps * cchunks;

    // Staging: a thread moves 8 consecutive k of one A row and one B row.
    const int s_row = tid >> 2;
    const int s_k = (tid & 3) * 8;
    const OutRow r = out_row(g, grid, o_grid, m0 + s_row);
    if ((tid & 3) == 0) row_live[s_row] = r.live ? 1 : 0;
    const bool n_live = (n0 + s_row) < g.c_out;
    const bf16* wrow = w + static_cast<long long>(n_live ? n0 + s_row : 0) * taps * g.c_in;

    mfma_cd acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) acc[i][j] = mfma_zero();
    }

    int src_row = -1;
    bool from_cache = false;
    for (int s = 0; s < steps; ++s) {
        const int tap = s / cchunks;
        const int c0 = (s - tap * cchunks) * kMfmaBK;
        if (c0 == 0 && r.live) {
            int it, ih, iw;
            tap_of(g, tap, it, ih, iw);
            src_row = tap_row(g, r, it, ih, iw, has_cache, from_cache);
        }
        const int c = c0 + s_k;
        const bool k_live = c < g.c_in;
        b128 a{{0u, 0u, 0u, 0u}};
        b128 b{{0u, 0u, 0u, 0u}};
        if (k_live && r.live && src_row >= 0) {
            a = *reinterpret_cast<const b128*>((from_cache ? cache : x) +
                                               static_cast<long long>(src_row) * g.c_in + c);
        }
        if (k_live && n_live) {
            b = *reinterpret_cast<const b128*>(wrow + static_cast<long long>(tap) * g.c_in + c);
        }

        __syncthreads();
        *reinterpret_cast<b128*>(As + s_row * kMfmaLd + s_k) = a;
        *reinterpret_cast<b128*>(Bs + s_row * kMfmaLd + s_k) = b;
        __syncthreads();

#pragma unroll
        for (int i = 0; i < 2; ++i) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                acc[i][j] = mfma_tile<kMfmaBK>(As + (wave_m * 32 + i * 16) * kMfmaLd, kMfmaLd,
                                               Bs + (wave_n * 32 + j * 16) * kMfmaLd, kMfmaLd, acc[i][j]);
            }
        }
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            mfma_each(acc[i][j], [&](int rr, int cc, float v) {
                const int tm = wave_m * 32 + i * 16 + rr;
                const int m = m0 + tm;
                const int n = n0 + wave_n * 32 + j * 16 + cc;
                if (m >= g.rows_out || n >= g.c_out) return;
                float out = 0.f;
                if (row_live[tm]) out = v + (bias != nullptr ? bias[n] : 0.f);
                o[static_cast<long long>(m) * g.c_out + n] = from_f32<bf16>(out);
            });
        }
    }
}

// A checkpoint's `[C_out, C_in·taps]` (channel-major) weight relabelled to
// `[C_out, taps·C_in]` (taps-major), the order the kernels above read.
__global__ void conv_weight_taps_major(const bf16* __restrict__ src, bf16* __restrict__ dst, int c_out, int c_in,
                                       int taps) {
    const long long total = static_cast<long long>(c_out) * c_in * taps;
    const long long e = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= total) return;
    const long long pitch = static_cast<long long>(c_in) * taps;
    const int n = static_cast<int>(e / pitch);
    const int k = static_cast<int>(e - n * pitch);
    const int tap = k / c_in;
    const int c = k - tap * c_in;
    dst[e] = src[n * pitch + static_cast<long long>(c) * taps + tap];
}

}  // namespace pie::conv
