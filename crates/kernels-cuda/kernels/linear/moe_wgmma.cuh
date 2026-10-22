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
#include "linear/moe_w4.cuh"
#include "linear/w8a16.cuh"

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
/// wgmma: `moe_select_fp8a8_wgmma`'s ring and swizzle over bf16, a stage
/// `kAtoms` 64-wide K atoms (128-byte rows), one m64 tile of the expert's
/// rows a CTA (a decode's few routes stream more rows' CTAs at once), the
/// MMAs summing straight into the accumulators (no block scales). `k` whole
/// `64 kAtoms`s, `n` whole 64s. Grid `[work_cap, n / 64]`, 128 threads,
/// `kWgStages kAtoms (64 + kR) 128 + 1024` bytes of shared memory.
template <int kR, int kAtoms = 1>
__global__ void __launch_bounds__(128) moe_select_bf16_wgmma(
    const bf16* __restrict__ act, const i32* __restrict__ order, const i32* __restrict__ offsets,
    const i32* __restrict__ work, const bf16* __restrict__ weight, bf16* __restrict__ out,
    int act_div, int n, int k, int num_experts) {
    constexpr int kT = 1;
    constexpr int kRows = 64 * kT;
    // `kAtoms` 64-wide K atoms a stage (64 or 128 of K).
    constexpr int kA = kAtoms * kRows * 128;
    constexpr int kStage = kA + kAtoms * kR * 128;
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
    const int row0 = static_cast<int>(blockIdx.y) * kRows;
    const int steps = k / (64 * kAtoms);
    const bf16* w_e = weight + (static_cast<long long>(expert) * n + row0) * k;

    auto load = [&](int kb, int st) {
        u8* a = smem + st * kStage;
        u8* b = a + kA;
        const int k0 = kb * 64 * kAtoms;
        for (int i = tid; i < kRows * 8 * kAtoms; i += 128) {
            const int r = i / (8 * kAtoms), q = i % (8 * kAtoms), at = q >> 3, ch = q & 7;
            wgfp8::cp16(a + at * kRows * 128 + r * 128 + ((ch ^ (r & 7)) << 4),
                        w_e + static_cast<long long>(r) * k + k0 + q * 8);
        }
        for (int i = tid; i < kR * 8 * kAtoms; i += 128) {
            const int r = i / (8 * kAtoms), q = i % (8 * kAtoms), at = q >> 3, ch = q & 7;
            u8* dst = b + at * kR * 128 + r * 128 + ((ch ^ (r & 7)) << 4);
            if (r < count) {
                const long long row = order[begin + r] / act_div;
                wgfp8::cp16(dst, act + row * k + k0 + q * 8);
            } else {
                *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
            }
        }
    };

    float acc[kT][kF];
#pragma unroll
    for (int t = 0; t < kT; ++t)
#pragma unroll
        for (int i = 0; i < kF; ++i) acc[t][i] = 0.f;

#pragma unroll
    for (int s = 0; s < kWgStages - 1; ++s) {
        if (s < steps) load(s, s);
        asm volatile("cp.async.commit_group;\n" ::);
    }
    // A step's MMAs stay in flight past it: the stage after next loads where
    // the step before read, once its MMAs land (the warpgroup's, one CTA).
    for (int kb = 0; kb < steps; ++kb) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(kWgStages - 2));
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

        const u8* a = smem + (kb % kWgStages) * kStage;
        const u8* b = a + kA;
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
        for (int kk = 0; kk < 4 * kAtoms; ++kk) {
            const int at = kk >> 2, off = (kk & 3) * 32;
            const unsigned long long db = wgfp8::desc(b + at * kR * 128 + off);
#pragma unroll
            for (int t = 0; t < kT; ++t) {
                wgbf16::Mma<kR>::run(acc[t], wgfp8::desc(a + at * kRows * 128 + t * 64 * 128 + off), db);
            }
        }
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
        const int next = kb + kWgStages - 1;
        if (next < steps) load(next, next % kWgStages);
        asm volatile("cp.async.commit_group;\n" ::);
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");

#pragma unroll
    for (int t = 0; t < kT; ++t)
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

namespace wgfp4 {

// m64n64k16 with A, the warp's 16 rows, from registers: the a0..a3 of
// mma.sync's m16n8k16, so the register dequant feeds it as it feeds mma.sync.
__device__ __forceinline__ void mma(float (&d)[32], const unsigned (&a)[4],
                                    unsigned long long b) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,"
        "%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, 1, 1, 1, 0;\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
          "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]),
          "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]),
          "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
          "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]),
          "+f"(d[31])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b));
}

// Holds `a`'s registers to here: an MMA in flight still reads them.
__device__ __forceinline__ void hold(unsigned (&a)[4]) {
    asm volatile("" : "+r"(a[0]), "+r"(a[1]), "+r"(a[2]), "+r"(a[3])::"memory");
}

}  // namespace wgfp4

namespace wgfp4 {

// The byte of `w` that `sel` (`0x4440 | c`) picks: its two e2m1 codes as
// `w4mma::e2m1_pair` lands them (the low code the pair's low bf16).
__device__ __forceinline__ unsigned byte_codes(unsigned w, unsigned sel) {
    const unsigned b = __byte_perm(w, 0u, sel);
    return ((b * 0x01001000u) & 0x80008000u) | ((b * 0x00040040u) & 0x01C001C0u);
}

}  // namespace wgfp4

/// `moe_select_mxfp4_mma_staged` on Hopper's wgmma (sm_90a), for a slice of
/// 64 routes an expert: a CTA, one warpgroup, two m64 tiles of the expert's
/// rows (128 rows, or a GLU's 64 gate and 64 up rows). Each 128-code K
/// chunk's route rows (K-major, 128-byte swizzled, the MMAs' B) and the CTA's
/// codes and scales stage through a `kStages` cp.async ring. A lane
/// dequantizes the A fragment mma.sync's m16n8k16 would hold of its warp's
/// 16 rows of each tile, k-step `s`'s codes `16 s + 2 c (+ 1)` and `+ 8` the
/// bytes `8 s + c` and `8 s + 4 + c` of the row's chunk, and the warpgroup's
/// m64n64k16 MMAs read them from registers. `k` whole 64s: a last chunk of
/// 64 codes stages its half (the rest zero) and steps its first four k-steps
/// alone. Grid `[work_cap, rows / 128]` (64 GLU columns), 128 threads,
/// `kStages 25600 + 5120` bytes of shared memory past a 1024-byte alignment.

template <int kGlu, bool kNv, int kStages>
__global__ void __launch_bounds__(128, 3) moe_select_fp4_wgmma(
    const bf16* __restrict__ act, const i32* __restrict__ order, const i32* __restrict__ offsets,
    const i32* __restrict__ work, const u8* __restrict__ codes, const u8* __restrict__ scales,
    const bf16* __restrict__ bias, bf16* __restrict__ out, int act_div, int n, int k,
    int num_experts, int inter, float limit, float galpha, const float* __restrict__ tscale) {
    constexpr int kSlice = 64;
    constexpr int kT = 2;
    constexpr int kThreads = 128;
    // A stage: the routes' two K-major atoms (128 codes), then the CTA's
    // rows' codes (64 bytes a row, its 16-byte runs swizzled by row pairs)
    // and scales (8 bytes a row).
    constexpr int kAtom = 64 * 128;
    constexpr int kRows = 64 * kT;
    constexpr int kB = 2 * kAtom;
    // Whole 1024s, each stage's atoms on the swizzle's alignment.
    constexpr int kStage = (kB + kRows * (64 + 8) + 1023) / 1024 * 1024;
    // A thread's 16-byte route runs, code runs and scale runs a chunk.
    constexpr int kRuns = 64 * 16 / kThreads;
    constexpr int kSegs = kRows * 4 / kThreads;
    constexpr int kSfs = (kRows + kThreads - 1) / kThreads;

    if (static_cast<int>(blockIdx.x) >= offsets[num_experts + 1]) return;
    const int item = work[blockIdx.x];
    const int expert = item >> 16;
    const int begin = offsets[expert] + (item & 0xFFFF) * kSlice;
    const int count = min(offsets[expert + 1] - begin, kSlice);
    if (count <= 0) return;

    // The ring on the swizzle's 1024-byte alignment, kept a shared-memory
    // pointer (an offset from the array, not a cast integer). Past it, a
    // warp's 32 rows' NVFP4 scales a chunk as bf16 pairs.
    extern __shared__ uint4 wgfp4_smem[];
    const unsigned base = static_cast<unsigned>(__cvta_generic_to_shared(wgfp4_smem));
    u8* ring = reinterpret_cast<u8*>(wgfp4_smem) + ((1024u - (base & 1023u)) & 1023u);
    unsigned* conv = reinterpret_cast<unsigned*>(ring + kStages * kStage);

    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    // Warp w's 16 rows of each m64 tile.
    constexpr int kTileRows = kGlu != 0 ? 64 : 64 * kT;
    const int row_base = static_cast<int>(blockIdx.y) * kTileRows + 16 * warp;
    const int span = kGlu != 0 ? inter : n;
    // A CTA row `r` (tile `r / 64`, row `r % 64` of it): its bank row.
    auto bank_row = [&](int r) {
        const int t = r / 64, base = static_cast<int>(blockIdx.y) * kTileRows + r % 64;
        return kGlu != 0 ? base + t * inter : base + 64 * t;
    };

    const int chunks = (k + 127) >> 7;
    // A last chunk of 64 codes: the K steps it holds.
    const int tail_steps = (k & 127) != 0 ? (k & 127) >> 4 : 8;
    const long long row_bytes = k >> 1;
    const u8* w_e = codes + static_cast<long long>(expert) * n * row_bytes;
    const int sf_row = kNv ? k >> 4 : k >> 5;
    const u8* s_e = scales + static_cast<long long>(expert) * n * sf_row;

    // This thread's route runs' rows (-1 past the slice), and its code and
    // scale runs' bank rows, clamped into the bank (a row past it lands
    // nowhere).
    int src[kRuns], wr[kSegs], sr[kSfs];
#pragma unroll
    for (int j = 0; j < kRuns; ++j) {
        const int row = (tid + kThreads * j) >> 4;
        src[j] = row < count ? order[begin + row] / act_div : -1;
    }
#pragma unroll
    for (int j = 0; j < kSegs; ++j) wr[j] = min(bank_row((tid + kThreads * j) >> 2), n - 1);
#pragma unroll
    for (int j = 0; j < kSfs; ++j) sr[j] = min(bank_row(min(tid + kThreads * j, kRows - 1)), n - 1);

    auto cp = [](void* dst, const void* from, int live) {
        const unsigned at = static_cast<unsigned>(__cvta_generic_to_shared(dst));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(at), "l"(from), "r"(live));
    };
    // Where a bank row's scales are not whole cp.async words apart (gpt-oss's
    // 90 bytes a row) or a last chunk holds half of them, a chunk's scales
    // come through registers a stage ahead, read in halves and words.
    const bool sf_regs = (sf_row & (kNv ? 7 : 3)) != 0 || tail_steps < 8;
    uint2 sf_next[kSfs];
    auto sf_read = [&](int ch) {
#pragma unroll
        for (int j = 0; j < kSfs; ++j) {
            sf_next[j] = make_uint2(0u, 0u);
            if (ch >= chunks || tid + kThreads * j >= kRows) continue;
            const bool half = ch == chunks - 1 && tail_steps < 8;
            const u8* from = s_e + static_cast<long long>(sr[j]) * sf_row + ch * (kNv ? 8 : 4);
            if constexpr (kNv) {
                const unsigned* w = reinterpret_cast<const unsigned*>(from);
                sf_next[j] = make_uint2(w[0], half ? 0u : w[1]);
            } else {
                const unsigned short* h = reinterpret_cast<const unsigned short*>(from);
                sf_next[j].x = h[0] | (half ? 0u : static_cast<unsigned>(h[1]) << 16);
            }
        }
    };
    if (sf_regs) sf_read(0);
    auto stage = [&](int ch) {
        if (ch < chunks) {
            u8* st = ring + (ch % kStages) * kStage;
            // A chunk of 64 codes (the last, where `k` is no whole 128s)
            // stages its half; the codes and routes past `k` land as zero.
            const bool half = ch == chunks - 1 && tail_steps < 8;
#pragma unroll
            for (int j = 0; j < kRuns; ++j) {
                const int u = tid + kThreads * j, row = u >> 4, q = u & 15;
                const bool live = src[j] >= 0 && (!half || q < 8);
                const bf16* from = live ? act + static_cast<long long>(src[j]) * k + ch * 128 + q * 8 : act;
                cp(st + (q >> 3) * kAtom + row * 128 + (((q & 7) ^ (row & 7)) << 4), from, live ? 16 : 0);
            }
#pragma unroll
            for (int j = 0; j < kSegs; ++j) {
                const int u = tid + kThreads * j, r = u >> 2, q = u & 3;
                const bool live = !half || q < 2;
                cp(st + kB + r * 64 + ((q ^ ((r >> 1) & 3)) << 4),
                   live ? w_e + wr[j] * row_bytes + ch * 64 + q * 16 : w_e, live ? 16 : 0);
            }
            if (sf_regs) {
#pragma unroll
                for (int j = 0; j < kSfs; ++j) {
                    const int r = tid + kThreads * j;
                    if (r < kRows) *reinterpret_cast<uint2*>(st + kB + kRows * 64 + r * 8) = sf_next[j];
                }
                sf_read(ch + 1);
            } else {
#pragma unroll
                for (int j = 0; j < kSfs; ++j) {
                    const int r = tid + kThreads * j;
                    if (r >= kRows) break;
                    const u8* from = s_e + static_cast<long long>(sr[j]) * sf_row + ch * (kNv ? 8 : 4);
                    const unsigned at = static_cast<unsigned>(__cvta_generic_to_shared(st + kB + kRows * 64 + r * 8));
                    if constexpr (kNv) {
                        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(at), "l"(from));
                    } else {
                        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(at), "l"(from));
                    }
                }
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    float acc[kT][8][4];
#pragma unroll
    for (int t = 0; t < kT; ++t)
#pragma unroll
        for (int b = 0; b < 8; ++b)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[t][b][e] = 0.f;

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) stage(s);

    for (int ch = 0; ch < chunks; ++ch) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(kStages - 2));
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        stage(ch + kStages - 1);

        const u8* st = ring + (ch % kStages) * kStage;
        // The lane's rows (h: g, g + 8) of each tile in the stage.
        int rr[kT][2];
#pragma unroll
        for (int t = 0; t < kT; ++t)
#pragma unroll
            for (int h = 0; h < 2; ++h) rr[t][h] = t * 64 + 16 * warp + 8 * h + g;
        const u8* sfs = st + kB + kRows * 64;
        // NVFP4: lane `l` converts its warp's row `l % 16` of tile `l / 16`'s
        // eight block scales; MXFP4 reads its four a step.
        unsigned* cw = conv + warp * kT * 16 * 8;
        uint2 sc[kT][2];
        bool wide = false;
        if constexpr (kNv) {
            __syncwarp();
#pragma unroll
            for (int j = 0; j < (kT * 16 + 31) / 32; ++j) {
                const int l = lane + 32 * j;
                if (l < kT * 16) {
                    const int r = (l / 16) * 64 + 16 * warp + l % 16;
                    const uint2 v = *reinterpret_cast<const uint2*>(sfs + r * 8);
                    unsigned o[8];
#pragma unroll
                    for (int q = 0; q < 8; ++q) {
                        const unsigned e = w4mma::nvfp4_scale_bf16(((q < 4 ? v.x : v.y) >> (8 * (q & 3))) & 0xFFu);
                        o[q] = e | (e << 16);
                    }
                    uint4* d = reinterpret_cast<uint4*>(cw + l * 8);
                    d[0] = make_uint4(o[0], o[1], o[2], o[3]);
                    d[1] = make_uint4(o[4], o[5], o[6], o[7]);
                }
            }
            __syncwarp();
        } else {
#pragma unroll
            for (int t = 0; t < kT; ++t)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    sc[t][h] = *reinterpret_cast<const uint2*>(sfs + rr[t][h] * 8);
                    wide |= __vcmpgtu4(sc[t][h].x, 0x80808080u) != 0u;
                }
            wide = __any_sync(0xffffffffu, wide);
        }
        const unsigned sel = 0x4440u | static_cast<unsigned>(c);
        const int steps = ch == chunks - 1 ? tail_steps : 8;

        unsigned a[2][kT][4];
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if (s >= steps) break;
            unsigned (&as)[kT][4] = a[s & 1];
#pragma unroll
            for (int t = 0; t < kT; ++t) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int r = rr[t][h];
                    const uint2 w = *reinterpret_cast<const uint2*>(
                        st + kB + r * 64 + (((s >> 1) ^ ((r >> 1) & 3)) << 4) + (s & 1) * 8);
                    // The step's scale, a bf16 pair: NVFP4's e4m3 a 16 codes
                    // (block `s`, converted), MXFP4's e8m0 a 32 (block `s / 2`).
                    unsigned x, x2 = 0u;
                    if constexpr (kNv) {
                        x = cw[(16 * t + 8 * h + g) * 8 + s];
                    } else {
                        const unsigned b = (sc[t][h].x >> (8 * (s >> 1))) & 0xFFu;
                        x = !wide || b <= 128u ? (b + 126u) << 7 : 253u << 7;
                        x2 = b <= 128u ? 0x3F80u : b << 7;
                        x2 |= x2 << 16;
                        x |= x << 16;
                    }
                    unsigned lo = w4mma::mul2(wgfp4::byte_codes(w.x, sel), x);
                    unsigned hi = w4mma::mul2(wgfp4::byte_codes(w.y, sel), x);
                    if (!kNv && wide) {
                        lo = w4mma::mul2(lo, x2);
                        hi = w4mma::mul2(hi, x2);
                    }
                    as[t][h] = lo;
                    as[t][2 + h] = hi;
                }
            }
            const unsigned long long db = wgfp8::desc(st + (s >> 2) * kAtom + (s & 3) * 32);
            asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
            for (int t = 0; t < kT; ++t) wgfp4::mma(reinterpret_cast<float(&)[32]>(acc[t]), as[t], db);
            asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
            // The step before's A registers are free once its MMAs land.
            asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
#pragma unroll
            for (int t = 0; t < kT; ++t) wgfp4::hold(as[t]);
        }
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    }
    asm volatile("cp.async.wait_group 0;\n" ::);

    const bf16* b_e = bias != nullptr ? bias + static_cast<long long>(expert) * n : nullptr;
    const float* t_e = kNv ? tscale + static_cast<long long>(expert) * n : nullptr;
    auto factor = [&](int row) { return kNv ? t_e[row] * 256.f : 1.f; };
    if (row_base >= span) return;
    w4mma::w4_land<kGlu, kT, 8, 64>(acc, factor, b_e, order, begin, count, out, n, inter, row_base,
                                    g, c, limit, galpha);
}


/// A block-FP8 projection (`n x k` e4m3 codes under ue8m0 exponents a row a
/// `kGran`, 32 or 128, columns, laid in tcgen05's atoms) against a batch's
/// bf16 rows (W8A16, at most 64 of them) on Hopper's wgmma (sm_90a), the
/// weights as A: a CTA, one warpgroup, takes two m64 tiles of the plane's
/// rows, the batch's rows the MMAs' n64 B. Each 128-code K chunk's batch
/// rows (two K-major 128-byte-swizzled atoms), the tiles' codes (128 bytes a
/// row, its 16-byte runs swizzled by row) and their four exponent bytes
/// stage through a `kStages` cp.async ring; a lane widens the A fragment
/// mma.sync's m16n8k16 would hold of its warp's 16 rows of each tile, codes
/// `16 s + 2 c (+ 1)` and `+ 8` of k-step `s`, into bf16 under its block's
/// exponent, each code once. `k` whole 128s. With `splits` > 1 a CTA sums
/// the `blockIdx.z`-th run of chunks into `part` (`[splits, m, n]` f32) for
/// `fp8_w8a16_fold`. Grid `[n / 128, 1, splits]`, 128 threads,
/// `kStages * 33792 + 1024` bytes of shared memory.
template <int kGran, int kStages>
__global__ void __launch_bounds__(128) fp8_w8a16_wgmma(
    const bf16* __restrict__ x, int x_stride, const u8* __restrict__ codes,
    const u8* __restrict__ atoms, bf16* __restrict__ y, int y_stride, int m, int n, int k,
    const u32* __restrict__ win, float* __restrict__ part, int splits) {
    static_assert(kGran == 32 || kGran == 128, "a block is 32 or 128 columns of K");
    constexpr int kT = 2;
    constexpr int kThreads = 128;
    constexpr int kAtom = 64 * 128;
    constexpr int kRows = 64 * kT;
    constexpr int kB = 2 * kAtom;
    constexpr int kCodes = kRows * 128;
    constexpr int kStage = (kB + kCodes + kRows * 4 + 1023) / 1024 * 1024;
    constexpr int kRuns = 64 * 16 / kThreads;
    constexpr int kSegs = kRows * 8 / kThreads;

    int rows = m;
    if (win != nullptr) rows = min(rows, static_cast<int>(win[0]));
    if (rows <= 0) return;

    extern __shared__ uint4 wgw8_smem[];
    const unsigned base = static_cast<unsigned>(__cvta_generic_to_shared(wgw8_smem));
    u8* ring = reinterpret_cast<u8*>(wgw8_smem) + ((1024u - (base & 1023u)) & 1023u);

    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    const int n0 = static_cast<int>(blockIdx.x) * kRows;
    const int kblocks = k / kGran;
    // This CTA's run of K's 128-code chunks.
    const int all = k >> 7;
    const int per = (all + splits - 1) / splits;
    const int first = static_cast<int>(blockIdx.z) * per;
    const int chunks = max(0, min(all, first + per) - first);

    int wr[kSegs];
#pragma unroll
    for (int j = 0; j < kSegs; ++j) wr[j] = min(n0 + ((tid + kThreads * j) >> 3), n - 1);
    const int sr = min(n0 + tid, n - 1);

    auto cp = [](void* dst, const void* from, int live) {
        const unsigned at = static_cast<unsigned>(__cvta_generic_to_shared(dst));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(at), "l"(from), "r"(live));
    };
    auto stage = [&](int i) {
        if (i < chunks) {
            const int ch = first + i;
            u8* st = ring + (i % kStages) * kStage;
#pragma unroll
            for (int j = 0; j < kRuns; ++j) {
                const int u = tid + kThreads * j, row = u >> 4, q = u & 15;
                const bool live = row < rows;
                const bf16* from = x + static_cast<long long>(live ? row : 0) * x_stride + ch * 128 + q * 8;
                cp(st + (q >> 3) * kAtom + row * 128 + (((q & 7) ^ (row & 7)) << 4), from, live ? 16 : 0);
            }
#pragma unroll
            for (int j = 0; j < kSegs; ++j) {
                const int u = tid + kThreads * j, r = u >> 3, q = u & 7;
                cp(st + kB + r * 128 + ((q ^ (r & 7)) << 4), codes + static_cast<long long>(wr[j]) * k + ch * 128 + q * 16, 16);
            }
            // The row's exponents of the chunk: four of an atom's adjacent
            // bytes (32-wide blocks), or the one of them a 128 is.
            const long long at = w8a16::atom_offset(sr, (ch * 128 / kGran) & ~3, kblocks);
            const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(st + kB + kCodes + tid * 4));
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(dst), "l"(atoms + at));
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    float acc[kT][8][4];
#pragma unroll
    for (int t = 0; t < kT; ++t)
#pragma unroll
        for (int b = 0; b < 8; ++b)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[t][b][e] = 0.f;

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) stage(s);

    // The lane's rows (h: g, g + 8) of each tile.
    int rr[kT][2];
#pragma unroll
    for (int t = 0; t < kT; ++t)
#pragma unroll
        for (int h = 0; h < 2; ++h) rr[t][h] = t * 64 + 16 * warp + 8 * h + g;
    // A pair of codes `2c, 2c + 1` of a 16-byte run: its word, and the
    // selector `w8a16::widen2` lays them by.
    const unsigned sel = (c & 1) ? 0xB3A2u : 0x9180u;

    for (int i = 0; i < chunks; ++i) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(kStages - 2));
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        stage(i + kStages - 1);

        const u8* st = ring + (i % kStages) * kStage;
        unsigned ex[kT][2];
#pragma unroll
        for (int t = 0; t < kT; ++t)
#pragma unroll
            for (int h = 0; h < 2; ++h) ex[t][h] = *reinterpret_cast<const unsigned*>(st + kB + kCodes + rr[t][h] * 4);
        const int sub = kGran == 128 ? (first + i) & 3 : 0;

        unsigned a[2][kT][4];
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            unsigned (&as)[kT][4] = a[s & 1];
#pragma unroll
            for (int t = 0; t < kT; ++t) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int r = rr[t][h];
                    const uint4 v = *reinterpret_cast<const uint4*>(st + kB + r * 128 + ((s ^ (r & 7)) << 4));
                    const unsigned e = (ex[t][h] >> (8 * (kGran == 32 ? s >> 1 : sub))) & 0xFFu;
                    const unsigned f2 = w8a16::factor2(e);
                    as[t][h] = w8a16::widen2((c >> 1) ? v.y : v.x, sel, f2);
                    as[t][2 + h] = w8a16::widen2((c >> 1) ? v.w : v.z, sel, f2);
                }
            }
            const unsigned long long db = wgfp8::desc(st + (s >> 2) * kAtom + (s & 3) * 32);
            asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
            for (int t = 0; t < kT; ++t) wgfp4::mma(reinterpret_cast<float(&)[32]>(acc[t]), as[t], db);
            asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
            asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
#pragma unroll
            for (int t = 0; t < kT; ++t) wgfp4::hold(as[t]);
        }
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    }
    asm volatile("cp.async.wait_group 0;\n" ::);

#pragma unroll
    for (int t = 0; t < kT; ++t)
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int tok = 8 * j + 2 * c + (q & 1);
                const int col = n0 + t * 64 + 16 * warp + g + (q >= 2 ? 8 : 0);
                if (tok >= rows || col >= n) continue;
                if (part != nullptr) {
                    part[(static_cast<long long>(blockIdx.z) * m + tok) * n + col] = acc[t][j][q];
                } else {
                    y[static_cast<long long>(tok) * y_stride + col] = f32_to_bf16(acc[t][j][q]);
                }
            }
}


namespace wgbf16gemm {

// m64n256k16 bf16, A and B from shared memory, accumulating into `d`.
__device__ __forceinline__ void mma(float (&d)[128], unsigned long long a, unsigned long long b) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, %128, %129, 1, 1, 1, 0, 0;\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
        : "l"(a), "l"(b));
}

__device__ __forceinline__ void bar_init(unsigned long long* bar, unsigned count) {
    const unsigned at = static_cast<unsigned>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(at), "r"(count));
}

__device__ __forceinline__ void bar_arrive(unsigned long long* bar) {
    const unsigned at = static_cast<unsigned>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(at) : "memory");
}

__device__ __forceinline__ void bar_wait(unsigned long long* bar, unsigned parity) {
    const unsigned at = static_cast<unsigned>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n.reg .pred done;\nwait_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;\n"
        "@!done bra wait_%=;\n}\n" ::"r"(at), "r"(parity)
        : "memory");
}

}  // namespace wgbf16gemm

/// A prefill's routed bf16 select on Hopper's wgmma, warp-specialized:
/// `y[route] = x[route / act_div] . W[expert]^T`, or with a GLU (`kGlu` 1..5)
/// `glu(x . gate, x . up)` over `inter` columns, the up rows `inter` past the
/// gate's. A CTA takes 128 of an expert's routes and 256 of its weight rows
/// (128 gate and their 128 up rows with a GLU): a producer warpgroup gathers
/// the routes' rows and the weight rows, 64 of K a stage, into a `kStages`
/// ring by cp.async (128-byte swizzled, K-major), each stage handed over and
/// back by mbarriers; two consumer warpgroups each run m64n256k16 MMAs over
/// 64 of the routes. A GLU's gate and up columns lie in the same lane's
/// accumulators (columns `x` and `128 + x`). `k` whole 64s. Grid
/// `[work_cap, ceil(cols / (kGlu ? 128 : 256))]`, 384 threads, `kStages *
/// 49152 + 1024 + 128` bytes of shared memory.
template <int kStages, int kGlu>
__global__ void __launch_bounds__(384, 1) moe_select_bf16_wsgemm(
    const bf16* __restrict__ act, const i32* __restrict__ order, const i32* __restrict__ offsets,
    const i32* __restrict__ work, const bf16* __restrict__ weight, bf16* __restrict__ out,
    int act_div, int n, int k, int num_experts, int inter, float limit, float galpha) {
    constexpr int kSlice = 128;
    constexpr int kA = kSlice * 128;   // 128 routes x 64 bf16
    constexpr int kB = 256 * 128;      // 256 weight rows x 64 bf16
    constexpr int kStage = kA + kB;

    if (static_cast<int>(blockIdx.x) >= offsets[num_experts + 1]) return;
    const int item = work[blockIdx.x];
    const int expert = item >> 16;
    const int begin = offsets[expert] + (item & 0xFFFF) * kSlice;
    const int count = min(offsets[expert + 1] - begin, kSlice);
    if (count <= 0) return;

    extern __shared__ uint4 wsgemm_smem[];
    const unsigned base = static_cast<unsigned>(__cvta_generic_to_shared(wsgemm_smem));
    u8* ring = reinterpret_cast<u8*>(wsgemm_smem) + ((1024u - (base & 1023u)) & 1023u);
    unsigned long long* full = reinterpret_cast<unsigned long long*>(ring + kStages * kStage);
    unsigned long long* empty = full + kStages;

    const int tid = static_cast<int>(threadIdx.x);
    const int wg = tid >> 7;
    const int steps = k >> 6;
    // The CTA's first output column: 128 GLU columns, or 256 rows.
    const int col0 = static_cast<int>(blockIdx.y) * (kGlu != 0 ? 128 : 256);
    if (tid == 0) {
        for (int s = 0; s < kStages; ++s) {
            wgbf16gemm::bar_init(full + s, 128);
            wgbf16gemm::bar_init(empty + s, 8);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();

    if (wg == 0) {
        // The producer: its rows of each stage, gathered.
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
        const bf16* w_e = weight + static_cast<long long>(expert) * n * k;
        const bf16* asrc[8];
        int alive[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int r = (tid + 128 * j) >> 3;
            alive[j] = r < count;
            asrc[j] = act + static_cast<long long>(alive[j] ? order[begin + r] / act_div : 0) * k;
        }
        const bf16* bsrc[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int r = (tid + 128 * j) >> 3;
            const int row = kGlu != 0 ? (r < 128 ? col0 + r : inter + col0 + (r - 128)) : col0 + r;
            const int clamp = kGlu != 0 ? (r < 128 ? min(row, inter - 1) : min(row, 2 * inter - 1)) : min(row, n - 1);
            bsrc[j] = w_e + static_cast<long long>(clamp) * k;
        }
        auto cp = [](void* dst, const void* from, int live) {
            const unsigned at = static_cast<unsigned>(__cvta_generic_to_shared(dst));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(at), "l"(from), "r"(live));
        };
        constexpr int kLag = 1;
        for (int kb = 0; kb < steps; ++kb) {
            const int s = kb % kStages;
            if (kb >= kStages) wgbf16gemm::bar_wait(empty + s, ((kb / kStages) - 1) & 1);
            u8* st = ring + s * kStage;
            const int k0 = kb * 64;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int u = tid + 128 * j, r = u >> 3, q = u & 7;
                cp(st + r * 128 + ((q ^ (r & 7)) << 4), asrc[j] + k0 + q * 8, alive[j] ? 16 : 0);
            }
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int u = tid + 128 * j, r = u >> 3, q = u & 7;
                cp(st + kA + r * 128 + ((q ^ (r & 7)) << 4), bsrc[j] + k0 + q * 8, 16);
            }
            asm volatile("cp.async.commit_group;\n" ::);
            if (kb >= kLag) {
                asm volatile("cp.async.wait_group %0;\n" ::"n"(kLag));
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                wgbf16gemm::bar_arrive(full + (kb - kLag) % kStages);
            }
        }
        asm volatile("cp.async.wait_group 0;\n" ::);
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        for (int kb = max(0, steps - kLag); kb < steps; ++kb) wgbf16gemm::bar_arrive(full + kb % kStages);
        return;
    }

    // The consumers: warpgroup `c` takes routes 64 c .. 64 c + 63.
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
    const int c_wg = wg - 1;
    const int lt = tid & 127, warp = lt >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    float acc[128];
#pragma unroll
    for (int i = 0; i < 128; ++i) acc[i] = 0.f;
    for (int kb = 0; kb < steps; ++kb) {
        const int s = kb % kStages;
        wgbf16gemm::bar_wait(full + s, (kb / kStages) & 1);
        const u8* st = ring + s * kStage;
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            wgbf16gemm::mma(acc, wgfp8::desc(st + c_wg * 64 * 128 + kk * 32), wgfp8::desc(st + kA + kk * 32));
        }
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
        // The stage before is read: hand it back.
        if (kb > 0 && lane == 0) wgbf16gemm::bar_arrive(empty + (kb - 1) % kStages);
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");

    // acc[4 j + q]: route row 16 warp + g (+ 8 for q >= 2) of the
    // warpgroup's 64, column 8 j + 2 c (+ 1 for odd q) of the 256.
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int r = c_wg * 64 + 16 * warp + g + 8 * h;
        if (r >= count) continue;
        const long long route = order[begin + r];
        if constexpr (kGlu != 0) {
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int col = col0 + 8 * j + 2 * c;
                if (col >= inter) continue;
                const float o0 = glu_fn::glu<kGlu>(acc[4 * j + 2 * h], acc[4 * (j + 16) + 2 * h], limit, galpha);
                const float o1 = glu_fn::glu<kGlu>(acc[4 * j + 2 * h + 1], acc[4 * (j + 16) + 2 * h + 1], limit, galpha);
                bf16* o = out + route * inter + col;
                if (col + 1 < inter) {
                    *reinterpret_cast<unsigned*>(o) = pack_bf16x2(o0, o1);
                } else {
                    o[0] = f32_to_bf16(o0);
                }
            }
        } else {
#pragma unroll
            for (int j = 0; j < 32; ++j) {
                const int col = col0 + 8 * j + 2 * c;
                if (col >= n) continue;
                bf16* o = out + route * n + col;
                if (col + 1 < n) {
                    *reinterpret_cast<unsigned*>(o) = pack_bf16x2(acc[4 * j + 2 * h], acc[4 * j + 2 * h + 1]);
                } else {
                    o[0] = f32_to_bf16(acc[4 * j + 2 * h]);
                }
            }
        }
    }
}

}  // namespace pie::linear
