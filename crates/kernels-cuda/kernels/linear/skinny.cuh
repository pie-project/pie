
#pragma once

#include "prelude/device.cuh"
#include "prelude/mma.cuh"

#include <cuda_bf16.h>
#include <cuda_pipeline.h>

namespace pie::linear {

constexpr int kSkinnyM = 64;

template <int kBK>
__host__ __device__ constexpr int skinny_ld() {
    return kBK + 8;
}

__device__ __forceinline__ void skinny_ldmatrix_x4(unsigned (&reg)[4], const bf16* at) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(at));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
                 : "r"(addr));
}

__device__ __forceinline__ unsigned long long skinny_policy(bool keep) {
    unsigned long long policy;
    if (keep) {
        asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(policy));
    } else {
        asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(policy));
    }
    return policy;
}

__device__ __forceinline__ void skinny_cp_async16(void* dst, const void* src, unsigned long long policy) {
    const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n"
                 :
                 : "r"(d), "l"(src), "l"(policy)
                 : "memory");
}

__device__ __forceinline__ float skinny_round(float v) {
    return Elem<bf16>::to_f32(Elem<bf16>::from_f32(v));
}

__device__ __forceinline__ float skinny_geglu(float g, float u) {
    constexpr float kAlpha = 0.7978845608028654f;
    constexpr float kBeta = 0.044715f;
    const float inner = kAlpha * (g + kBeta * g * g * g);
    const float gelu = 0.5f * g * (1.f + tanhf(inner));
    return gelu * u;
}

template <int kWarps, int kStages, int kBK, int kEpilogue>
__global__ __launch_bounds__(kWarps * 32) void skinny_bf16_kernel(
    const bf16* __restrict__ w,
    const bf16* __restrict__ act,
    bf16* __restrict__ out,
    int m,
    int n,
    int k,
    float cap)
{
    constexpr bool kGeglu = kEpilogue == 2;
    constexpr int kThreads = kWarps * 32;
    constexpr int kSkinnyLd = skinny_ld<kBK>();
    static_assert(kBK % 64 == 0, "a step is whole 128-byte row pieces");
    static_assert(!kGeglu || kWarps % 2 == 0, "geglu halves the block's rows between gate and up");

    constexpr int kN = kWarps * 16;

    constexpr int kCols = kGeglu ? kN / 2 : kN;

    constexpr int kNSubs = kSkinnyM / 8;

    constexpr int kQuad = kBK / 16;
    constexpr int kWStage = kN * kSkinnyLd;
    constexpr int kAStage = kSkinnyM * kSkinnyLd;
    constexpr int kStageElems = kWStage + kAStage;
    constexpr int kChunksPerRow = kBK / 8;
    constexpr int kWChunks = kN * kChunksPerRow;
    constexpr int kAChunks = kSkinnyM * kChunksPerRow;

    constexpr int kLdC = kN + 8;
    static_assert(kStages >= 2, "one stage in flight while one is multiplied, at least");
    static_assert(kWChunks % kThreads == 0 && kAChunks % kThreads == 0,
                  "the staging map is a whole number of 16-byte chunks per thread");
    static_assert(kSkinnyM * kLdC <= kStageElems, "the epilogue tile fits in one stage of the ring");
    static_assert(kCols % 8 == 0, "the drain stores whole 16-byte chunks");

    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int n0 = static_cast<int>(blockIdx.x) * kCols;

    extern __shared__ __align__(16) unsigned char skinny_smem[];
    bf16* ring = reinterpret_cast<bf16*>(skinny_smem);

    if (m < kSkinnyM) {
        u32* z = reinterpret_cast<u32*>(skinny_smem);
        constexpr int kWords = kStages * kStageElems / 2;
        for (int i = tid; i < kWords; i += kThreads) z[i] = 0u;
        __syncthreads();
    }

    auto weight_row = [&](int r) -> int {
        if constexpr (kGeglu) {
            return r < kCols ? n0 + r : n + n0 + (r - kCols);
        } else {
            return n0 + r;
        }
    };

    const unsigned long long stream_policy = skinny_policy(false);
    const unsigned long long keep_policy = skinny_policy(true);

    auto stage = [&](int buf, int k0) {
        bf16* wdst = ring + buf * kStageElems;
        bf16* adst = wdst + kWStage;
#pragma unroll
        for (int i = 0; i < kWChunks / kThreads; ++i) {
            const int chunk = tid + i * kThreads;
            const int r = chunk / kChunksPerRow;
            const int c = (chunk % kChunksPerRow) * 8;
            skinny_cp_async16(
                wdst + r * kSkinnyLd + c,
                w + static_cast<long long>(weight_row(r)) * k + k0 + c,
                stream_policy);
        }
#pragma unroll
        for (int i = 0; i < kAChunks / kThreads; ++i) {
            const int chunk = tid + i * kThreads;
            const int r = chunk / kChunksPerRow;
            const int c = (chunk % kChunksPerRow) * 8;
            if (r < m) {
                skinny_cp_async16(
                    adst + r * kSkinnyLd + c,
                    act + static_cast<long long>(r) * k + k0 + c,
                    keep_policy);
            }
        }
        __pipeline_commit();
    };

    const int steps = k / kBK;

    float acc[kNSubs][4];
#pragma unroll
    for (int s = 0; s < kNSubs; ++s) {
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[s][i] = 0.f;
    }

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < steps) {
            stage(s, s * kBK);
        } else {
            __pipeline_commit();
        }
    }

    const int a_row = lane & 15;
    const int a_col = (lane >> 4) << 3;

    for (int step = 0; step < steps; ++step) {

        const int fetch = step + kStages - 1;
        if (fetch < steps) {
            stage(fetch % kStages, fetch * kBK);
        } else {
            __pipeline_commit();
        }
        __pipeline_wait_prior(kStages - 1);
        __syncthreads();

        const bf16* wsrc = ring + (step % kStages) * kStageElems + warp * 16 * kSkinnyLd;
        const bf16* asrc = ring + (step % kStages) * kStageElems + kWStage;
#pragma unroll
        for (int kk = 0; kk < kQuad; ++kk) {
            unsigned a[4];
            skinny_ldmatrix_x4(a, wsrc + a_row * kSkinnyLd + kk * 16 + a_col);
#pragma unroll
            for (int pair = 0; pair < kNSubs / 2; ++pair) {
                unsigned b[4];
                skinny_ldmatrix_x4(b, asrc + (pair * 16 + a_row) * kSkinnyLd + kk * 16 + a_col);
                const unsigned b_lo[2] = {b[0], b[2]};
                const unsigned b_hi[2] = {b[1], b[3]};
                ::nvcuda::wmma::detail::mma_m16n8k16(acc[2 * pair], a, b_lo, acc[2 * pair]);
                ::nvcuda::wmma::detail::mma_m16n8k16(acc[2 * pair + 1], a, b_hi, acc[2 * pair + 1]);
            }
        }
        __syncthreads();
    }

    __pipeline_wait_prior(0);
    __syncthreads();
    bf16* c_tile = ring;
    {
        const int g = lane >> 2;
        const int t = lane & 3;
#pragma unroll
        for (int s = 0; s < kNSubs; ++s) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int col = warp * 16 + g + 8 * half;
                const int row = s * 8 + 2 * t;
                c_tile[row * kLdC + col] = Elem<bf16>::from_f32(acc[s][2 * half]);
                c_tile[(row + 1) * kLdC + col] = Elem<bf16>::from_f32(acc[s][2 * half + 1]);
            }
        }
    }
    __syncthreads();

    constexpr int kChunksOfRow = kCols / 8;
    constexpr int kDrain = kSkinnyM * kChunksOfRow;
#pragma unroll
    for (int i = 0; i < (kDrain + kThreads - 1) / kThreads; ++i) {
        const int chunk = tid + i * kThreads;
        if (chunk < kDrain) {
            const int r = chunk / kChunksOfRow;
            const int c = (chunk % kChunksOfRow) * 8;
            if (r < m) {
                bf16* dst = out + static_cast<long long>(r) * n + n0 + c;
                const bf16* src = c_tile + r * kLdC + c;
                if constexpr (kEpilogue == 0) {
                    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
                } else {
                    __align__(16) bf16 piece[8];
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        float v;
                        if constexpr (kEpilogue == 1) {
                            v = cap * tanhf(Elem<bf16>::to_f32(src[e]) * (1.f / cap));
                        } else {
                            v = skinny_geglu(
                                Elem<bf16>::to_f32(src[e]),
                                Elem<bf16>::to_f32(src[kCols + e]));
                        }
                        piece[e] = Elem<bf16>::from_f32(v);
                    }
                    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(piece);
                }
            }
        }
    }
}

}
