
#pragma once

#include "prelude/device.cuh"
#include "prelude/mma.cuh"

#include <cuda_bf16.h>
#include <cuda_pipeline.h>

namespace pie::linear {


constexpr int kMmaK = 16;
constexpr int kMmaM = 16;
constexpr int kMmaN = 8;

constexpr int kTiledK = 64;

constexpr int kTiledQuad = kTiledK / kMmaK;

constexpr int kTiledLdA = kTiledK + 8;

constexpr int kTiledBand = kMmaK;


template <int lut>
__device__ __forceinline__ int pie_lop3(int a, int b, int c) {
    int res;
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(res)
                 : "r"(a), "r"(b), "r"(c), "n"(lut));
    return res;
}

__device__ __forceinline__ unsigned dequant_u4_bf16x2(unsigned q) {
    constexpr int kLo = 0x000f000f;
    constexpr int kEx = 0x43004300;

    return static_cast<unsigned>(
        pie_lop3<(0xf0 & 0xcc) | 0xaa>(static_cast<int>(q), kLo, kEx));
}


constexpr unsigned kTiledMagic = 0x43004300u;

__device__ __forceinline__ __nv_bfloat162 as_pair(unsigned bits) {
    __nv_bfloat162 out;
    out.x.raw = static_cast<unsigned short>(bits & 0xffffu);
    out.y.raw = static_cast<unsigned short>(bits >> 16);
    return out;
}

__device__ __forceinline__ unsigned as_bits(__nv_bfloat162 v) {
    return (static_cast<unsigned>(v.y.raw) << 16) | static_cast<unsigned>(v.x.raw);
}

__device__ __forceinline__ unsigned splat(bf16 v) {
    const unsigned h = v.raw;
    return (h << 16) | h;
}

__device__ __forceinline__ unsigned fold_post(unsigned dq, unsigned s2, unsigned b2) {
    const __nv_bfloat162 code = __hsub2(as_pair(dq), as_pair(kTiledMagic));
    return as_bits(__hfma2(code, as_pair(s2), as_pair(b2)));
}

__device__ __forceinline__ void ldmatrix_a(unsigned (&reg)[4], const bf16* at) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(at));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
                 : "r"(addr));
}


template <class T, int kBits, int kGroup, int kM, int kN, int kThreads, int kStages>
__global__ __launch_bounds__(kThreads) void matmul_affine_tiled(
    const T* __restrict__ act,
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    const u8* __restrict__ biases,
    T* __restrict__ out,
    int m,
    int n,
    int k,
    const u32* __restrict__ win)
{
    static_assert(is_same<T, bf16>::value,
                  "this point is bf16 activations only -- the mma wrapper it uses "
                  "is bf16, and an f16 twin needs its own parity run");
    static_assert(kBits == 4, "this point serves the four-bit code plane only");
    static_assert(kGroup % kMmaK == 0,
                  "a 16-wide k tile must sit inside one group, or a lane's two "
                  "factors would not cover the codes it holds");

    constexpr int kWarps = kThreads / 32;
    static_assert(kWarps * 32 == kThreads, "a block is a whole number of warps");
    static_assert(kN == kWarps * kTiledBand,
                  "one warp owns one 16-column band, because one repacked word is "
                  "a whole B fragment for two columns eight apart");
    constexpr int kMFrags = kM / kMmaM;
    static_assert(kMFrags * kMmaM == kM, "the row tile is whole mma tiles");
    constexpr int kNSubs = kTiledBand / kMmaN;
    static_assert((kStages & (kStages - 1)) == 0 && kStages >= 2,
                  "the stage index is masked, so the depth is a power of two");

    constexpr int kStageElems = kM * kTiledLdA;

    constexpr int kLdC = kN + 8;
    constexpr int kDrainChunks = kM * kN / 8;

    static_assert((kM * (kTiledK / 8)) % kThreads == 0,
                  "the activation staging map is a whole number of 16-byte chunks "
                  "per thread");
    static_assert((kStages * kStageElems / 2) % kThreads == 0,
                  "the staging buffer zeroes in whole words per thread");
    static_assert(kDrainChunks % kThreads == 0 && (kM * kN) % kThreads == 0,
                  "the epilogue drains in whole chunks per thread, vector and "
                  "scalar alike");

    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int m0 = static_cast<int>(blockIdx.x) * kM;
    const int n0 = static_cast<int>(blockIdx.y) * kN;

    int rows = m;
    if (win != nullptr) {
        const int staged = static_cast<int>(win[0]);
        if (staged < rows) rows = staged;
    }
    if (m0 >= rows) return;

    extern __shared__ __align__(16) unsigned char tiled_smem[];
    bf16* a_tile = reinterpret_cast<bf16*>(tiled_smem);

    if (m0 + kM > rows) {
        u32* z = reinterpret_cast<u32*>(tiled_smem);
        constexpr int kWords = kStages * kStageElems / 2;
#pragma unroll
        for (int i = 0; i < kWords / kThreads; ++i) {
            z[tid + i * kThreads] = 0u;
        }
        __syncthreads();
    }

    constexpr int kChunksPerRow = kTiledK / 8;
    constexpr int kChunks = kM * kChunksPerRow;
    constexpr int kChunksPerThread = kChunks / kThreads;

    const int steps = k / kTiledK;
    const int groups = k / kGroup;

    const int band = (n0 / kTiledBand) + warp;
    const bool live = band * kTiledBand < n;
    const int col_of = lane >> 2;
    const int row_of = lane & 3;

    const uint4* wq = reinterpret_cast<const uint4*>(codes)
        + static_cast<long long>(band) * steps * 32 + lane;
    const bf16* sf = reinterpret_cast<const bf16*>(scales);
    const bf16* bp = reinterpret_cast<const bf16*>(biases);

    float acc[kMFrags][kNSubs][4];
#pragma unroll
    for (int mt = 0; mt < kMFrags; ++mt) {
#pragma unroll
        for (int ns = 0; ns < kNSubs; ++ns) {
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[mt][ns][i] = 0.f;
        }
    }

    unsigned s2[kNSubs];
    unsigned b2[kNSubs];
#pragma unroll
    for (int ns = 0; ns < kNSubs; ++ns) {
        s2[ns] = 0u;
        b2[ns] = 0u;
    }
    int held = -1;

    auto stage = [&](int buf, int k0) {
        bf16* dst = a_tile + buf * kStageElems;
#pragma unroll
        for (int i = 0; i < kChunksPerThread; ++i) {
            const int chunk = tid + i * kThreads;
            const int r = chunk / kChunksPerRow;
            const int col = (chunk % kChunksPerRow) * 8;
            if (m0 + r < rows) {
                __pipeline_memcpy_async(
                    dst + r * kTiledLdA + col,
                    act + static_cast<long long>(m0 + r) * k + k0 + col,
                    16);
            }
        }
        __pipeline_commit();
    };

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < steps) {
            stage(s, s * kTiledK);
        } else {
            __pipeline_commit();
        }
    }

    uint4 q4 = make_uint4(0u, 0u, 0u, 0u);
    if (live) q4 = wq[0];

    for (int step = 0; step < steps; ++step) {
        const int fetch = step + kStages - 1;
        if (fetch < steps) {
            stage(fetch & (kStages - 1), fetch * kTiledK);
        } else {
            __pipeline_commit();
        }

        const unsigned qs[kTiledQuad] = {q4.x, q4.y, q4.z, q4.w};
        if (live && step + 1 < steps) q4 = wq[(step + 1) * 32];

        __pipeline_wait_prior(kStages - 1);
        __syncthreads();

        const bf16* src = a_tile + (step & (kStages - 1)) * kStageElems;

#pragma unroll
        for (int kk = 0; kk < kTiledQuad; ++kk) {
            const int kt = step * kTiledQuad + kk;

            const int g = (kt * kMmaK) / kGroup;
            if (live && g != held) {
                held = g;
                const long long at =
                    (static_cast<long long>(band) * groups + g) * kTiledBand + col_of;
#pragma unroll
                for (int ns = 0; ns < kNSubs; ++ns) {
                    s2[ns] = splat(sf[at + ns * 8]);
                    b2[ns] = splat(bp[at + ns * 8]);
                }
            }

            const unsigned q = qs[kk];
            unsigned frag[kNSubs][2];
#pragma unroll
            for (int ns = 0; ns < kNSubs; ++ns) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    frag[ns][h] = fold_post(
                        dequant_u4_bf16x2(q >> (4 * (2 * ns + h))), s2[ns], b2[ns]);
                }
            }

#pragma unroll
            for (int mt = 0; mt < kMFrags; ++mt) {
                unsigned a[4];
                ldmatrix_a(
                    a,
                    src + (mt * kMmaM + (lane & 15)) * kTiledLdA + kk * kMmaK
                        + ((lane >> 4) << 3));
#pragma unroll
                for (int ns = 0; ns < kNSubs; ++ns) {
                    ::nvcuda::wmma::detail::mma_m16n8k16(
                        acc[mt][ns], a, frag[ns], acc[mt][ns]);
                }
            }
        }
        __syncthreads();
    }

    __pipeline_wait_prior(0);
    __syncthreads();
    bf16* c_tile = reinterpret_cast<bf16*>(tiled_smem);
    {

        u32* c_word = reinterpret_cast<u32*>(tiled_smem);
        const int col = warp * kTiledBand + 2 * row_of;
#pragma unroll
        for (int mt = 0; mt < kMFrags; ++mt) {
#pragma unroll
            for (int ns = 0; ns < kNSubs; ++ns) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const int row = mt * kMmaM + col_of + 8 * half;
                    const bf16 lo = Elem<T>::from_f32(acc[mt][ns][2 * half]);
                    const bf16 hi = Elem<T>::from_f32(acc[mt][ns][2 * half + 1]);
                    c_word[(row * kLdC + col + ns * kMmaN) >> 1] =
                        (static_cast<u32>(hi.raw) << 16) | static_cast<u32>(lo.raw);
                }
            }
        }
    }
    __syncthreads();

    if ((n & 7) == 0 && n0 + kN <= n) {
        constexpr int kChunksOfRow = kN / 8;
#pragma unroll
        for (int i = 0; i < kDrainChunks / kThreads; ++i) {
            const int chunk = tid + i * kThreads;
            const int r = chunk / kChunksOfRow;
            const int c = (chunk % kChunksOfRow) * 8;
            if (m0 + r < rows) {
                const uint4 v = *reinterpret_cast<const uint4*>(c_tile + r * kLdC + c);
                *reinterpret_cast<uint4*>(
                    out + static_cast<long long>(m0 + r) * n + n0 + c) = v;
            }
        }
    } else {
#pragma unroll
        for (int i = 0; i < (kM * kN) / kThreads; ++i) {
            const int at = tid + i * kThreads;
            const int r = at / kN;
            const int c = at % kN;
            if (m0 + r < rows && n0 + c < n) {
                out[static_cast<long long>(m0 + r) * n + n0 + c] = c_tile[r * kLdC + c];
            }
        }
    }
}

template <class T, int kBits, int kGroup, int kRowsT, int kBands, int kSplit>
__global__ __launch_bounds__(32 * kBands * kSplit) void gemv_affine_tiled(
    const T* __restrict__ act,
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    const u8* __restrict__ biases,
    T* __restrict__ out,
    int m,
    int n,
    int k,
    const u32* __restrict__ win)
{
    static_assert(is_same<T, bf16>::value,
                  "this point is bf16 activations only -- it reads two of them as one "
                  "32-bit word, and an f16 twin needs its own parity run");
    static_assert(kBits == 4, "this point serves the four-bit code plane only");
    static_assert(kGroup % kMmaK == 0,
                  "a 16-wide k tile must sit inside one group, or a lane's two "
                  "factors would not cover the codes it holds");
    static_assert(kRowsT >= 1 && kRowsT <= kMmaM,
                  "the decode point holds its accumulators in registers, and above one "
                  "mma tile of rows the tiled GEMM is the point");
    static_assert(kBands >= 1 && kSplit >= 1, "a block is at least one warp");

    constexpr int kNSubs = kTiledBand / kMmaN;

    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int col_of = lane >> 2;
    const int row_of = lane & 3;
    const int slice = warp % kSplit;

    int rows = m;
    if (win != nullptr) {
        const int staged = static_cast<int>(win[0]);
        if (staged < rows) rows = staged;
    }

    const int bands = (n + kTiledBand - 1) / kTiledBand;
    const int band = static_cast<int>(blockIdx.x) * kBands + (warp / kSplit);
    const int steps = k / kTiledK;
    const int groups = k / kGroup;

    float acc[kRowsT][kNSubs];
#pragma unroll
    for (int r = 0; r < kRowsT; ++r) {
#pragma unroll
        for (int ns = 0; ns < kNSubs; ++ns) acc[r][ns] = 0.f;
    }

    if (band < bands && rows > 0) {

        const uint4* wq = reinterpret_cast<const uint4*>(codes)
            + static_cast<long long>(band) * steps * 32 + lane;
        const bf16* sf = reinterpret_cast<const bf16*>(scales);
        const bf16* bp = reinterpret_cast<const bf16*>(biases);

        const u32* xw = reinterpret_cast<const u32*>(act);

        unsigned s2[kNSubs];
        unsigned b2[kNSubs];
#pragma unroll
        for (int ns = 0; ns < kNSubs; ++ns) {
            s2[ns] = 0u;
            b2[ns] = 0u;
        }
        int held = -1;

        for (int step = slice; step < steps; step += kSplit) {
            const uint4 q4 = wq[static_cast<long long>(step) * 32];
            const unsigned qs[kTiledQuad] = {q4.x, q4.y, q4.z, q4.w};
#pragma unroll
            for (int kk = 0; kk < kTiledQuad; ++kk) {
                const int kt = step * kTiledQuad + kk;
                const int g = (kt * kMmaK) / kGroup;
                if (g != held) {
                    held = g;
                    const long long at =
                        (static_cast<long long>(band) * groups + g) * kTiledBand + col_of;
#pragma unroll
                    for (int ns = 0; ns < kNSubs; ++ns) {
                        s2[ns] = splat(sf[at + ns * 8]);
                        b2[ns] = splat(bp[at + ns * 8]);
                    }
                }

                const unsigned q = qs[kk];
                float wf[kNSubs][2][2];
#pragma unroll
                for (int ns = 0; ns < kNSubs; ++ns) {
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const __nv_bfloat162 pair = as_pair(fold_post(
                            dequant_u4_bf16x2(q >> (4 * (2 * ns + h))), s2[ns], b2[ns]));
                        wf[ns][h][0] = bf16_to_f32(pair.x);
                        wf[ns][h][1] = bf16_to_f32(pair.y);
                    }
                }

                const long long xat = static_cast<long long>(kt) * 8 + row_of;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
#pragma unroll
                    for (int r = 0; r < kRowsT; ++r) {
                        if (r < rows) {
                            const __nv_bfloat162 xv = as_pair(
                                xw[static_cast<long long>(r) * (k / 2) + xat + 4 * h]);
                            const float x0 = bf16_to_f32(xv.x);
                            const float x1 = bf16_to_f32(xv.y);
#pragma unroll
                            for (int ns = 0; ns < kNSubs; ++ns) {
                                acc[r][ns] = fmaf(wf[ns][h][0], x0, acc[r][ns]);
                                acc[r][ns] = fmaf(wf[ns][h][1], x1, acc[r][ns]);
                            }
                        }
                    }
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < kRowsT; ++r) {
#pragma unroll
        for (int ns = 0; ns < kNSubs; ++ns) {
            acc[r][ns] += __shfl_xor_sync(0xffffffffu, acc[r][ns], 1);
            acc[r][ns] += __shfl_xor_sync(0xffffffffu, acc[r][ns], 2);
        }
    }

    if constexpr (kSplit > 1) {
        extern __shared__ __align__(16) unsigned char gemv_tiled_smem[];
        float* red = reinterpret_cast<float*>(gemv_tiled_smem);
        if (row_of == 0) {
#pragma unroll
            for (int r = 0; r < kRowsT; ++r) {
#pragma unroll
                for (int ns = 0; ns < kNSubs; ++ns) {
                    red[(warp * kRowsT + r) * kTiledBand + col_of + ns * 8] = acc[r][ns];
                }
            }
        }
        __syncthreads();
        if (slice != 0) return;
#pragma unroll
        for (int r = 0; r < kRowsT; ++r) {
#pragma unroll
            for (int ns = 0; ns < kNSubs; ++ns) {
                float sum = 0.f;
#pragma unroll
                for (int s = 0; s < kSplit; ++s) {
                    sum += red[((warp + s) * kRowsT + r) * kTiledBand + col_of + ns * 8];
                }
                acc[r][ns] = sum;
            }
        }
    }

    if (row_of != 0 || band >= bands) return;
#pragma unroll
    for (int ns = 0; ns < kNSubs; ++ns) {
        const int col = band * kTiledBand + col_of + ns * 8;
        if (col < n) {
#pragma unroll
            for (int r = 0; r < kRowsT; ++r) {
                if (r < rows) {
                    out[static_cast<long long>(r) * n + col] =
                        Elem<T>::from_f32(acc[r][ns]);
                }
            }
        }
    }
}

template <int kBits>
__global__ void repack_affine_tiled(
    const u8* __restrict__ codes,
    u32* __restrict__ out,
    int n,
    int k)
{
    static_assert(kBits == 4, "this pass repacks the four-bit code plane only");
    const int quads = k / kTiledK;
    const int bands = (n + kTiledBand - 1) / kTiledBand;
    const long long total = static_cast<long long>(bands) * quads * 32 * kTiledQuad;
    const long long at = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= total) return;

    const int word = static_cast<int>(at & (kTiledQuad - 1));
    const long long rest = at / kTiledQuad;
    const int lane = static_cast<int>(rest & 31);
    const long long tile = rest >> 5;
    const int kq = static_cast<int>(tile % quads);
    const int band = static_cast<int>(tile / quads);
    const int kt = kq * kTiledQuad + word;

    const int col_of = lane >> 2;
    const int k_base = kt * kMmaK + 2 * (lane & 3);
    const int row_bytes = k / 2;

    u32 res = 0u;
#pragma unroll
    for (int s = 0; s < 4; ++s) {

        const int col = band * kTiledBand + col_of + ((s >= 2) ? 8 : 0);
        const int k_off = (s & 1) ? 8 : 0;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int kk = k_base + k_off + h;
            u32 code = 0u;
            if (col < n) {
                const u8 byte = codes[static_cast<long long>(col) * row_bytes + (kk >> 1)];
                code = (kk & 1) ? static_cast<u32>(byte >> 4) : static_cast<u32>(byte & 0xFu);
            }
            res |= code << (4 * (s + 4 * h));
        }
    }
    out[at] = res;
}

__global__ void repack_factors_tiled(
    const u8* __restrict__ scales,
    const u8* __restrict__ biases,
    u8* __restrict__ out_scales,
    u8* __restrict__ out_biases,
    int n,
    int groups)
{
    const int bands = (n + kTiledBand - 1) / kTiledBand;
    const long long total = static_cast<long long>(bands) * groups * kTiledBand;
    const long long at = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= total) return;

    const int j = static_cast<int>(at % kTiledBand);
    const long long rest = at / kTiledBand;
    const int g = static_cast<int>(rest % groups);
    const int band = static_cast<int>(rest / groups);
    const int row = band * kTiledBand + j;

    bf16 sv = u16_as_bf16(0);
    bf16 bv = u16_as_bf16(0);
    if (row < n) {
        const long long from = static_cast<long long>(row) * groups + g;
        sv = reinterpret_cast<const bf16*>(scales)[from];
        bv = reinterpret_cast<const bf16*>(biases)[from];
    }
    reinterpret_cast<bf16*>(out_scales)[at] = sv;
    reinterpret_cast<bf16*>(out_biases)[at] = bv;
}

}
