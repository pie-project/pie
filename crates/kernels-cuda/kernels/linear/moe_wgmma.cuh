#pragma once

// A routed select over a 128x128-block FP8 bank against rows quantized to
// e4m3 in 1x128 blocks (W8A8) on Hopper's wgmma (sm_90a): a CTA, one
// warpgroup, takes 128 of an expert's weight rows (one block's) and a slice
// of `kR` of its routes. Each 128-wide K step's codes and route rows stage
// through a cp.async ring into shared memory in the 128-byte swizzle the
// MMA's descriptors read (K-major, 16-byte chunk c of row r at chunk
// c ^ (r % 8)); two m64nRk32 MMAs a 32-code step sum the block, whose sums
// then land in the accumulators under its two exponents. The accumulators
// lie as mma.sync's m16n8 ones do: warp w's rows 16 w + g (+ 8) of each
// m64, columns 8 j + 2 c (+ 1).

#include "linear/grouped_fp8.cuh"

namespace pie::linear {

namespace wgfp8 {

__device__ __forceinline__ unsigned long long desc(const void* smem) {
    const unsigned at = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    unsigned long long d = (at & 0x3FFFFu) >> 4;   // start address
    d |= 1ull << 16;                               // leading offset (unread: swizzled K-major)
    d |= static_cast<unsigned long long>(1024 >> 4) << 32;  // 8 rows of 128 bytes
    d |= 1ull << 62;                               // 128-byte swizzle
    return d;
}

__device__ __forceinline__ void cp16(void* dst, const void* src) {
    const unsigned at = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(at), "l"(src));
}

template <int R>
struct Mma;

template <>
struct Mma<32> {
    static __device__ __forceinline__ void run(float (&d)[16], unsigned long long a,
                                               unsigned long long b, int acc) {
        asm volatile(
            "{\n.reg .pred p;\nsetp.ne.b32 p, %18, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, p, 1, 1;\n}\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
              "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]),
              "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
            : "l"(a), "l"(b), "r"(acc));
    }
};

template <>
struct Mma<64> {
    static __device__ __forceinline__ void run(float (&d)[32], unsigned long long a,
                                               unsigned long long b, int acc) {
        asm volatile(
            "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,"
            "%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1;\n}\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
              "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]),
              "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]),
              "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
              "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]),
              "+f"(d[31])
            : "l"(a), "l"(b), "r"(acc));
    }
};

}  // namespace wgfp8

// The ring's depth, and the bytes a stage holds before the 1024-byte
// alignment the swizzle asks of its base.
constexpr int kWgStages = 3;

/// `y[route] = q(x[route / act_div]) . (W[expert] * 2^(s - 127))^T`, the rows
/// quantized by `quantize_rows_e4m3b128` (`xq`, `xe`), the bank's exponents
/// spread to its rows in tcgen05's atoms (a weight row an atom row), the
/// routes laid out by expert (`order`, `offsets`, `work` as `moe_route_order`
/// lays them, slices of `kR`). `n` and `k` whole 128s. Grid `[work_cap, n /
/// 128]`, 128 threads, `kWgStages (128 + kR) 128 + 1024` bytes of shared memory.
template <int kR>
__global__ void __launch_bounds__(128) moe_select_fp8a8_wgmma(
    const u8* __restrict__ xq, const u8* __restrict__ xe, const i32* __restrict__ order,
    const i32* __restrict__ offsets, const i32* __restrict__ work, const u8* __restrict__ weight,
    const u8* __restrict__ atoms, bf16* __restrict__ out, int act_div, int n, int k,
    int num_experts) {
    constexpr int kA = 128 * 128;
    constexpr int kStage = kA + kR * 128;
    constexpr int kF = kR / 2;  // a thread's sums of one m64 x kR

    if (static_cast<int>(blockIdx.x) >= offsets[num_experts + 1]) return;
    const int item = work[blockIdx.x];
    const int expert = item >> 16;
    const int begin = offsets[expert] + (item & 0xFFFF) * kR;
    const int count = min(offsets[expert + 1] - begin, kR);
    if (count <= 0) return;

    extern __shared__ uint4 wgfp8_smem[];
    u8* smem = reinterpret_cast<u8*>(
        (reinterpret_cast<unsigned long long>(wgfp8_smem) + 1023) & ~1023ull);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    const int row0 = static_cast<int>(blockIdx.y) * 128;
    const int blocks = k / 128;
    const u8* w_e = weight + (static_cast<long long>(expert) * n + row0) * k;
    const u8* atoms_at = atoms + atom_offset(expert * n + row0, 0, blocks);

    auto load = [&](int kb, int st) {
        u8* a = smem + st * kStage;
        u8* b = a + kA;
        const int k0 = kb * 128;
        for (int i = tid; i < 128 * 8; i += 128) {
            const int r = i >> 3, ch = i & 7;
            wgfp8::cp16(a + r * 128 + ((ch ^ (r & 7)) << 4),
                        w_e + static_cast<long long>(r) * k + k0 + ch * 16);
        }
        for (int i = tid; i < kR * 8; i += 128) {
            const int r = i >> 3, ch = i & 7;
            u8* dst = b + r * 128 + ((ch ^ (r & 7)) << 4);
            if (r < count) {
                const long long row = order[begin + r] / act_div;
                wgfp8::cp16(dst, xq + row * k + k0 + ch * 16);
            } else {
                *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
            }
        }
    };

    // This lane's routes' activation rows (C's columns 8 j + 2 c, + 1), -1
    // past the slice: their exponents scale the block sums.
    int xrow[kR / 8][2];
#pragma unroll
    for (int j = 0; j < kR / 8; ++j)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int tok = 8 * j + 2 * c + h;
            xrow[j][h] = tok < count ? order[begin + tok] / act_div : -1;
        }

    float acc[2][kF], blk[2][kF];
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int i = 0; i < kF; ++i) acc[t][i] = 0.f;

#pragma unroll
    for (int s = 0; s < kWgStages - 1; ++s) {
        if (s < blocks) load(s, s);
        asm volatile("cp.async.commit_group;\n" ::);
    }
    for (int kb = 0; kb < blocks; ++kb) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(kWgStages - 2));
        __syncthreads();
        // The stage's generic-proxy writes, seen by the MMA's async proxy.
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        const int next = kb + kWgStages - 1;
        if (next < blocks) load(next, next % kWgStages);
        asm volatile("cp.async.commit_group;\n" ::);

        const u8* a = smem + (kb % kWgStages) * kStage;
        const u8* b = a + kA;
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            const unsigned long long db = wgfp8::desc(b + kk * 32);
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                wgfp8::Mma<kR>::run(blk[t], wgfp8::desc(a + t * 64 * 128 + kk * 32), db, kk > 0);
            }
        }
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
        const float sw = gfp8::exp2_biased(atoms_at[(kb / 4) * 512 + kb % 4]);
#pragma unroll
        for (int j = 0; j < kR / 8; ++j) {
            const float s0 = xrow[j][0] >= 0 ? sw * gfp8::exp2_biased(xe[xrow[j][0] * blocks + kb]) : 0.f;
            const float s1 = xrow[j][1] >= 0 ? sw * gfp8::exp2_biased(xe[xrow[j][1] * blocks + kb]) : 0.f;
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                acc[t][4 * j + 0] = fmaf(blk[t][4 * j + 0], s0, acc[t][4 * j + 0]);
                acc[t][4 * j + 1] = fmaf(blk[t][4 * j + 1], s1, acc[t][4 * j + 1]);
                acc[t][4 * j + 2] = fmaf(blk[t][4 * j + 2], s0, acc[t][4 * j + 2]);
                acc[t][4 * j + 3] = fmaf(blk[t][4 * j + 3], s1, acc[t][4 * j + 3]);
            }
        }
        // The stage just read is the one a later step refills.
        __syncthreads();
    }

#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int j = 0; j < kR / 8; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int tok = 8 * j + 2 * c + (q & 1);
                if (tok >= count) continue;
                const int row = row0 + t * 64 + 16 * warp + g + (q >= 2 ? 8 : 0);
                const long long route = order[begin + tok];
                out[route * n + row] = f32_to_bf16(acc[t][4 * j + q]);
            }
}


namespace wgbf16 {

template <int R>
struct Mma;

template <>
struct Mma<32> {
    static __device__ __forceinline__ void run(float (&d)[16], unsigned long long a,
                                               unsigned long long b) {
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, 1, 1, 1, 0, 0;\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
              "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]),
              "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
            : "l"(a), "l"(b));
    }
};

template <>
struct Mma<16> {
    static __device__ __forceinline__ void run(float (&d)[8], unsigned long long a,
                                               unsigned long long b) {
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, 1, 1, 1, 0, 0;\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
              "+f"(d[7])
            : "l"(a), "l"(b));
    }
};

}  // namespace wgbf16

/// `y[route] = x[route / act_div] . W[expert]^T` for a bf16 bank on Hopper's
/// wgmma: `moe_select_fp8a8_wgmma`'s ring and swizzle over bf16, a stage 64 of
/// K (a 128-byte row), four m64nRk16 MMAs a stage into the sums, no block
/// scales. `k` whole 64s, `n` whole 128s. Grid `[work_cap, n / 128]`, 128
/// threads, `kWgStages (128 + kR) 128 + 1024` bytes of shared memory.
template <int kR>
__global__ void __launch_bounds__(128) moe_select_bf16_wgmma(
    const bf16* __restrict__ act, const i32* __restrict__ order, const i32* __restrict__ offsets,
    const i32* __restrict__ work, const bf16* __restrict__ weight, bf16* __restrict__ out,
    int act_div, int n, int k, int num_experts) {
    constexpr int kA = 128 * 128;
    constexpr int kStage = kA + kR * 128;
    constexpr int kF = kR / 2;

    if (static_cast<int>(blockIdx.x) >= offsets[num_experts + 1]) return;
    const int item = work[blockIdx.x];
    const int expert = item >> 16;
    const int begin = offsets[expert] + (item & 0xFFFF) * kR;
    const int count = min(offsets[expert + 1] - begin, kR);
    if (count <= 0) return;

    extern __shared__ uint4 wgbf16_smem[];
    u8* smem = reinterpret_cast<u8*>(
        (reinterpret_cast<unsigned long long>(wgbf16_smem) + 1023) & ~1023ull);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    const int row0 = static_cast<int>(blockIdx.y) * 128;
    const int steps = k / 64;
    const bf16* w_e = weight + (static_cast<long long>(expert) * n + row0) * k;

    auto load = [&](int kb, int st) {
        u8* a = smem + st * kStage;
        u8* b = a + kA;
        const int k0 = kb * 64;
        for (int i = tid; i < 128 * 8; i += 128) {
            const int r = i >> 3, ch = i & 7;
            wgfp8::cp16(a + r * 128 + ((ch ^ (r & 7)) << 4),
                        w_e + static_cast<long long>(r) * k + k0 + ch * 8);
        }
        for (int i = tid; i < kR * 8; i += 128) {
            const int r = i >> 3, ch = i & 7;
            u8* dst = b + r * 128 + ((ch ^ (r & 7)) << 4);
            if (r < count) {
                const long long row = order[begin + r] / act_div;
                wgfp8::cp16(dst, act + row * k + k0 + ch * 8);
            } else {
                *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
            }
        }
    };

    float acc[2][kF];
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int i = 0; i < kF; ++i) acc[t][i] = 0.f;

#pragma unroll
    for (int s = 0; s < kWgStages - 1; ++s) {
        if (s < steps) load(s, s);
        asm volatile("cp.async.commit_group;\n" ::);
    }
    for (int kb = 0; kb < steps; ++kb) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(kWgStages - 2));
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        const int next = kb + kWgStages - 1;
        if (next < steps) load(next, next % kWgStages);
        asm volatile("cp.async.commit_group;\n" ::);

        const u8* a = smem + (kb % kWgStages) * kStage;
        const u8* b = a + kA;
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            const unsigned long long db = wgfp8::desc(b + kk * 32);
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                wgbf16::Mma<kR>::run(acc[t], wgfp8::desc(a + t * 64 * 128 + kk * 32), db);
            }
        }
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
        __syncthreads();
    }

#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int j = 0; j < kR / 8; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int tok = 8 * j + 2 * c + (q & 1);
                if (tok >= count) continue;
                const int row = row0 + t * 64 + 16 * warp + g + (q >= 2 ? 8 : 0);
                const long long route = order[begin + tok];
                out[route * n + row] = f32_to_bf16(acc[t][4 * j + q]);
            }
}

}  // namespace pie::linear
