#pragma once

// Four-bit MLX affine planes repacked in tile order (`U4g64tiled`, s·c + b,
// bf16 activations): rows padded to 16-row bands, a band's 64-code K step
// 512 contiguous bytes, word L = 4·col + q's 32-bit word kt holding nibble s
// + 4h at row 16·band + col + 8·(s ≥ 2), k 64·step + 16·kt + 2q + 8·(s & 1)
// + h. On the 16x16x16 core that is B's own layout up to the k order within
// a tile. Factors bf16 [band][group][16].

#include "linear/affine.hip"

namespace pie::linear {

constexpr int kTiledK = 64;     // codes a step
constexpr int kTiledBand = 16;  // rows a band

// The prefill stage: a 64-code step of eight bands (128 weight rows), one
// thread a 16-byte word, decoded into the tile.
template <int kGroup>
struct TiledStage {
    const u8* codes;
    const bf16* scales;
    const bf16* biases;
    int band, col, q, steps, groups;
    Words<4> word;
    Factor f[4][2];  // a k tile's factors for the word's two rows

    __device__ __forceinline__ TiledStage(const u8* c, const u8* s, const u8* b, int n, int k)
        : codes(c), scales(reinterpret_cast<const bf16*>(s)), biases(reinterpret_cast<const bf16*>(b)) {
        const int bands = (n + kTiledBand - 1) / kTiledBand;
        band = min(static_cast<int>(blockIdx.y) * (kGemmBN / kTiledBand) + static_cast<int>(threadIdx.x) / 32,
                   bands - 1);
        col = threadIdx.x % 32 / 4;
        q = threadIdx.x % 4;
        steps = k / kTiledK;
        groups = k / kGroup;
    }

    __device__ __forceinline__ void load(int c) {
        word = *reinterpret_cast<const Words<4>*>(codes + ((static_cast<long long>(band) * steps + c) * 32 +
                                                           threadIdx.x % 32) * 16);
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            if (kGroup >= kTiledK && kt > 0) {
                f[kt][0] = f[0][0];
                f[kt][1] = f[0][1];
                continue;
            }
            const long long at = (static_cast<long long>(band) * groups + (c * kTiledK + kt * 16) / kGroup) * kTiledBand;
#pragma unroll
            for (int half = 0; half < 2; ++half)
                f[kt][half] = {to_f32(scales[at + col + 8 * half]), to_f32(biases[at + col + 8 * half])};
        }
    }

    __device__ __forceinline__ void store(bf16* ws) {
        bf16* tile = ws + (threadIdx.x / 32 * kTiledBand + col) * kGemmLd + 2 * q;
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            const u32 w = word[kt];
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const Factor fs = f[kt][s / 2];
                const float c0 = static_cast<float>((w >> (4 * s)) & 15u);
                const float c1 = static_cast<float>((w >> (4 * (s + 4))) & 15u);
                *reinterpret_cast<u32*>(tile + (s / 2) * 8 * kGemmLd + kt * 16 + 8 * (s & 1)) =
                    pack_pair<bf16>(dequant<kOffPost>(c0, fs), dequant<kOffPost>(c1, fs));
            }
        }
    }
};

// y = act · Wᵀ over a tiled plane, BM activation rows a block (the GEMM of
// linear/affine.hip).
template <int kGroup, int BM>
__global__ __launch_bounds__(kGemmThreads) void matmul_affine_tiled(const bf16* __restrict__ act,
                                                                    const u8* __restrict__ codes,
                                                                    const u8* __restrict__ scales,
                                                                    const u8* __restrict__ biases,
                                                                    bf16* __restrict__ out, float* __restrict__ part,
                                                                    int per, int m, int n, int k,
                                                                    const u32* __restrict__ win) {
    static_assert(kGroup % 16 == 0, "a k tile sits in one group");
    const int rows = win == nullptr ? m : min(m, static_cast<int>(win[0]));
    if (static_cast<int>(blockIdx.x) * BM >= rows) return;
    TiledStage<kGroup> stage(codes, scales, biases, n, k);
    affine_gemm<bf16, BM>(act, out, part, per, rows, n, k, stage);
}

// The decode over a tiled plane (≤ 16 rows): SPLIT wavefronts a band, every
// SPLIT-th K step each, a step one read a wavefront; raw codes (128 + c)
// fold a group at a time, Σ x·w = s·Σ x·(c + 128) + (b − 128·s)·Σ x.
template <int kGroup, int SPLIT>
__global__ __launch_bounds__(kWave* SPLIT) void gemv_affine_tiled(const bf16* __restrict__ act,
                                                                  const u8* __restrict__ codes,
                                                                  const u8* __restrict__ scales,
                                                                  const u8* __restrict__ biases,
                                                                  bf16* __restrict__ out, int m, int n, int k,
                                                                  const u32* __restrict__ win) {
    using U = Unpack<bf16, 4>;
    static_assert(kWave == 64, "a 16 x 16 x 16 core of 64 lanes");
    static_assert(kGroup % 16 == 0, "a k tile sits in one group");
    constexpr int kTiles = kGroup < kTiledK ? kGroup / 16 : 4;  // k tiles a fold

    const int rows = win == nullptr ? m : min(m, static_cast<int>(win[0]));
    if (rows <= 0) return;
    const int l = lane(), wave = threadIdx.x / kWave;
    const int c = l % 16, g = l / 16;
    const int band = blockIdx.x;
    const int steps = k / kTiledK, groups = k / kGroup;
    const int s0 = c >= 8 ? 8 : 0;  // the nibble shift of this row's first pair

    const u8* wp = codes + (static_cast<long long>(band) * steps * 32 + 4 * (c & 7) + g) * 16;
    const bf16* xp = act + static_cast<long long>(min(c, rows - 1)) * k + 2 * g;
    const bf16* sp = reinterpret_cast<const bf16*>(scales) + static_cast<long long>(band) * groups * kTiledBand + c;
    const bf16* bp = reinterpret_cast<const bf16*>(biases) + static_cast<long long>(band) * groups * kTiledBand + c;
    const Words<2> ones{kOnes<bf16>, kOnes<bf16>};

    mfma_cd acc = mfma_zero();
    for (int step = wave; step < steps; step += SPLIT) {
        const Words<4> w = __builtin_nontemporal_load(reinterpret_cast<const Words<4>*>(wp + step * 32 * 16));
        u32 x[4][2];
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            x[kt][0] = *reinterpret_cast<const u32*>(xp + step * kTiledK + kt * 16);
            x[kt][1] = *reinterpret_cast<const u32*>(xp + step * kTiledK + kt * 16 + 8);
        }
#pragma unroll
        for (int f0 = 0; f0 < 4; f0 += kTiles) {
            const long long fx = static_cast<long long>((step * kTiledK + f0 * 16) / kGroup) * kTiledBand;
            const float s = to_f32(sp[fx]);
            const float o = to_f32(bp[fx]) - U::kMagic * s;
            mfma_cd d = mfma_zero(), sx = mfma_zero();
#pragma unroll
            for (int kt = f0; kt < f0 + kTiles; ++kt) {
                const Words<2> a{x[kt][0], x[kt][1]};
                const Words<2> b{((w[kt] >> s0) & 0x000f000fu) | U::kHigh, ((w[kt] >> (s0 + 4)) & 0x000f000fu) | U::kHigh};
                d = mfma16x16<bf16>(a, b, d);
                sx = mfma16x16<bf16>(a, ones, sx);
            }
#pragma unroll
            for (int t = 0; t < 4; ++t) acc[t] = fmaf(s, d[t], fmaf(o, sx[t], acc[t]));
        }
    }

    // Lane l's cells are rows 4g + t of column c; the wavefronts add up in LDS.
    __shared__ float part[SPLIT][kWave * 4];
#pragma unroll
    for (int t = 0; t < 4; ++t) part[wave][t * kWave + l] = acc[t];
    __syncthreads();
    for (int cell = threadIdx.x; cell < kWave * 4; cell += blockDim.x) {
        float v = 0.f;
#pragma unroll
        for (int i = 0; i < SPLIT; ++i) v += part[i][cell];
        const int ll = cell % kWave, row = 4 * (ll / 16) + cell / kWave, col = band * kTiledBand + ll % 16;
        if (row < rows && col < n) out[static_cast<long long>(row) * n + col] = from_f32<bf16>(v);
    }
}

// ---------------------------------------------------------------- repack --

// The row-major four-bit plane in tile order: one thread an output word.
template <int kBits>
__global__ void repack_affine_tiled(const u8* __restrict__ codes, u32* __restrict__ out, int n, int k) {
    static_assert(kBits == 4, "this pass repacks the four-bit code plane only");
    const int steps = k / kTiledK;
    const int bands = (n + kTiledBand - 1) / kTiledBand;
    const long long at = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= static_cast<long long>(bands) * steps * 32 * 4) return;
    const int kt = static_cast<int>(at % 4);
    const int word = static_cast<int>(at / 4 % 32);
    const long long tile = at / 128;
    const int step = static_cast<int>(tile % steps), band = static_cast<int>(tile / steps);
    const int col = word / 4, q = word % 4;
    u32 res = 0u;
#pragma unroll
    for (int s = 0; s < 4; ++s)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = band * kTiledBand + col + (s >= 2 ? 8 : 0);
            const int kk = step * kTiledK + kt * 16 + 2 * q + 8 * (s & 1) + h;
            u32 code = 0u;
            if (row < n) {
                const u8 byte = codes[static_cast<long long>(row) * (k / 2) + kk / 2];
                code = kk & 1 ? byte >> 4 : byte & 15u;
            }
            res |= code << (4 * (s + 4 * h));
        }
    out[at] = res;
}

// The factor planes in tile order, [band][group][16]; rows past n are zero.
__global__ void repack_factors_tiled(const u8* __restrict__ scales, const u8* __restrict__ biases,
                                     u8* __restrict__ out_scales, u8* __restrict__ out_biases, int n, int groups) {
    const int bands = (n + kTiledBand - 1) / kTiledBand;
    const long long at = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= static_cast<long long>(bands) * groups * kTiledBand) return;
    const int j = static_cast<int>(at % kTiledBand);
    const long long rest = at / kTiledBand;
    const int g = static_cast<int>(rest % groups), band = static_cast<int>(rest / groups);
    const int row = band * kTiledBand + j;
    u16 s = 0, b = 0;
    if (row < n) {
        s = reinterpret_cast<const u16*>(scales)[static_cast<long long>(row) * groups + g];
        b = reinterpret_cast<const u16*>(biases)[static_cast<long long>(row) * groups + g];
    }
    reinterpret_cast<u16*>(out_scales)[at] = s;
    reinterpret_cast<u16*>(out_biases)[at] = b;
}

}  // namespace pie::linear
