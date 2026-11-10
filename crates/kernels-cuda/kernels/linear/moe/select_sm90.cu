#include "linear/codec.cuh"
#include "linear/wgmma.cuh"
#include "linear/grouped_fp8.cuh"
#include "linear/moe_w4.cuh"
#include "linear/moe/table.cuh"
#include "linear/w8a16.cuh"

namespace pie::linear {


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
    int num_experts, const u64* __restrict__ expert_table) {
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
    // The expert table ([codes, atoms] an expert), resolved once a tile.
    const u8* w_e = expert_plane(weight, static_cast<long long>(n) * k, expert_table, 2, 0, expert)
        + static_cast<long long>(row0) * k;
    const ExpertAtoms ea = expert_atoms(atoms, expert_table, expert, n);
    const u8* atoms_at = ea.at + atom_offset(ea.row0 + row0, 0, blocks);

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

}


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
    int act_div, int n, int k, int num_experts, const u64* __restrict__ expert_table) {
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
    // The expert table (one plane an expert), resolved once a tile.
    const bf16* w_e = expert_plane(weight, static_cast<long long>(n) * k, expert_table, 1, 0, expert)
        + static_cast<long long>(row0) * k;

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
/// `kSlice` 128 and `kWg` 2: 128 routes against two warpgroups' tiles.

template <int kGlu, bool kNv, int kStages, int kSlice = 64, int kWg = 1>
__global__ void __launch_bounds__(128 * kWg, kWg == 2 ? 1 : kSlice == 64 ? 3 : 2) moe_select_fp4_wgmma(
    const bf16* __restrict__ act, const i32* __restrict__ order, const i32* __restrict__ offsets,
    const i32* __restrict__ work, const u8* __restrict__ codes, const u8* __restrict__ scales,
    const bf16* __restrict__ bias, bf16* __restrict__ out, int act_div, int n, int k,
    int num_experts, int inter, float limit, float galpha, const float* __restrict__ tscale,
    const u64* __restrict__ expert_table) {
    constexpr int kT = 2;
    constexpr int kThreads = 128 * kWg;
    // A stage: the routes' two K-major atoms (128 codes), then the CTA's
    // rows' codes (64 bytes a row, its 16-byte runs swizzled by row pairs)
    // and scales (8 bytes a row).
    constexpr int kAtom = kSlice * 128;
    constexpr int kRows = 64 * kT * kWg;
    constexpr int kB = 2 * kAtom;
    // Whole 1024s, each stage's atoms on the swizzle's alignment.
    constexpr int kStage = (kB + kRows * (64 + 8) + 1023) / 1024 * 1024;
    // A thread's 16-byte route runs, code runs and scale runs a chunk.
    constexpr int kRuns = kSlice * 16 / kThreads;
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
    // Warp w's 16 rows of each m64 tile, its warpgroup's tiles past `wg0`.
    const int wq = warp & 3, wg0 = (warp >> 2) * 64 * kT;
    constexpr int kTileRows = kGlu != 0 ? 64 : 64 * kT;
    const int row_base = (static_cast<int>(blockIdx.y) * kWg + (warp >> 2)) * kTileRows + 16 * wq;
    const int span = kGlu != 0 ? inter : n;
    // A CTA row `r` (tile `r / 64`, row `r % 64` of it): its bank row.
    auto bank_row = [&](int r) {
        const int w = r / (64 * kT), t = r % (64 * kT) / 64;
        const int base = (static_cast<int>(blockIdx.y) * kWg + w) * kTileRows + r % 64;
        return kGlu != 0 ? base + t * inter : base + 64 * t;
    };

    const int chunks = (k + 127) >> 7;
    // A last chunk of 64 codes: the K steps it holds.
    const int tail_steps = (k & 127) != 0 ? (k & 127) >> 4 : 8;
    const long long row_bytes = k >> 1;
    const u8* w_e = expert_plane(codes, 1LL * n * row_bytes, expert_table, 3, 0, expert);
    const int sf_row = kNv ? k >> 4 : k >> 5;
    const u8* s_e = expert_plane(scales, 1LL * n * sf_row, expert_table, 3, 1, expert);

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

    float acc[kT][kSlice / 8][4];
#pragma unroll
    for (int t = 0; t < kT; ++t)
#pragma unroll
        for (int b = 0; b < kSlice / 8; ++b)
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
            for (int h = 0; h < 2; ++h) rr[t][h] = wg0 + t * 64 + 16 * wq + 8 * h + g;
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
                    const int r = wg0 + (l / 16) * 64 + 16 * wq + l % 16;
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
            for (int t = 0; t < kT; ++t) wgfp4::mma(reinterpret_cast<float(&)[kSlice / 2]>(acc[t]), as[t], db);
            asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
            // The step before's A registers are free once its MMAs land.
            asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
#pragma unroll
            for (int t = 0; t < kT; ++t) wgfp4::hold(as[t]);
        }
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    }
    asm volatile("cp.async.wait_group 0;\n" ::);

    const bf16* b_e = bias != nullptr ? expert_plane(bias, 1LL * n, expert_table, 3, 2, expert) : nullptr;
    const float* t_e = kNv ? expert_plane(tscale, 1LL * n, expert_table, 3, 2, expert) : nullptr;
    auto factor = [&](int row) { return kNv ? t_e[row] * 256.f : 1.f; };
    if (row_base >= span) return;
    w4mma::w4_land<kGlu, kT, kSlice / 8, 64>(acc, factor, b_e, order, begin, count, out, n, inter, row_base,
                                    g, c, limit, galpha);
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

}


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
    const i32* __restrict__ work, const bf16* __restrict__ weight, const bf16* __restrict__ bias,
    bf16* __restrict__ out, int act_div, int n, int k, int num_experts, int inter, float limit,
    float galpha, const u64* __restrict__ expert_table) {
    // A GLU's gate and up rows side by side; relu^2 (6) is no GLU, its rows as they are.
    constexpr bool kGated = kGlu != 0 && kGlu != 6;
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
    const int col0 = static_cast<int>(blockIdx.y) * (kGated ? 128 : 256);
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
        // The expert table (one plane an expert), resolved once a tile.
        const bf16* w_e = expert_plane(weight, static_cast<long long>(n) * k, expert_table, 1, 0, expert);
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
            const int row = kGated ? (r < 128 ? col0 + r : inter + col0 + (r - 128)) : col0 + r;
            const int clamp = kGated ? (r < 128 ? min(row, inter - 1) : min(row, 2 * inter - 1)) : min(row, n - 1);
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

    // An expert's bias (a GLU's gate rows, then its up rows), where it has one.
    const bf16* b_e = bias != nullptr ? bias + static_cast<long long>(expert) * n : nullptr;
    // acc[4 j + q]: route row 16 warp + g (+ 8 for q >= 2) of the
    // warpgroup's 64, column 8 j + 2 c (+ 1 for odd q) of the 256.
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int r = c_wg * 64 + 16 * warp + g + 8 * h;
        if (r >= count) continue;
        const long long route = order[begin + r];
        if constexpr (kGated) {
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int col = col0 + 8 * j + 2 * c;
                if (col >= inter) continue;
                float bg0 = 0.f, bg1 = 0.f, bu0 = 0.f, bu1 = 0.f;
                if (b_e != nullptr) {
                    bg0 = bf16_to_f32(b_e[col]);
                    bu0 = bf16_to_f32(b_e[inter + col]);
                    if (col + 1 < inter) {
                        bg1 = bf16_to_f32(b_e[col + 1]);
                        bu1 = bf16_to_f32(b_e[inter + col + 1]);
                    }
                }
                const float o0 = act::glu<kGlu>(acc[4 * j + 2 * h] + bg0, acc[4 * (j + 16) + 2 * h] + bu0, limit, galpha);
                const float o1 = act::glu<kGlu>(acc[4 * j + 2 * h + 1] + bg1, acc[4 * (j + 16) + 2 * h + 1] + bu1, limit, galpha);
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
                const float b0 = b_e != nullptr ? bf16_to_f32(b_e[col]) : 0.f;
                const float b1 = b_e != nullptr && col + 1 < n ? bf16_to_f32(b_e[col + 1]) : 0.f;
                bf16* o = out + route * n + col;
                float v0 = acc[4 * j + 2 * h] + b0, v1 = acc[4 * j + 2 * h + 1] + b1;
                if constexpr (kGlu == 6) {
                    v0 = fmaxf(v0, 0.f) * fmaxf(v0, 0.f);
                    v1 = fmaxf(v1, 0.f) * fmaxf(v1, 0.f);
                }
                if (col + 1 < n) {
                    *reinterpret_cast<unsigned*>(o) = pack_bf16x2(v0, v1);
                } else {
                    o[0] = f32_to_bf16(v0);
                }
            }
        }
    }
}


namespace wgfp8ws {

// m64n128k32 e4m3, A and B from shared memory, `d` overwritten (`acc` 0) or summed into.
__device__ __forceinline__ void mma(float (&d)[64], unsigned long long a, unsigned long long b, int acc) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, p, 1, 1;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "l"(a), "l"(b), "r"(acc));
}

}


/// `moe_select_fp8a8_wgmma` warp-specialized, for a prefill's routes: a CTA
/// takes 128 of an expert's routes and `128 kHalves` of its weight rows (a
/// block of the bank's scales a half). A producer warpgroup gathers the
/// routes' code rows and the weight rows, 128 of K a stage, into a `kStages`
/// ring by cp.async (128-byte swizzled, K-major), each stage handed over and
/// back by mbarriers; two consumer warpgroups each run m64n128k32 MMAs over
/// 64 of the routes, a half at a time, and scale a block's sums into their
/// accumulators while the other warpgroup's MMAs run. `n` whole `128
/// kHalves`, `k` whole 128s. Grid `[work_cap, n / (128 kHalves)]`, 384
/// threads, `kStages * 16384 (1 + kHalves) + 1024 + 128` bytes of shared
/// memory.
template <int kStages, int kHalves = 1>
__global__ void __launch_bounds__(384, 1) moe_select_fp8a8_wsgemm(
    const u8* __restrict__ xq, const u8* __restrict__ xe, const i32* __restrict__ order,
    const i32* __restrict__ offsets, const i32* __restrict__ work, const u8* __restrict__ weight,
    const u8* __restrict__ atoms, bf16* __restrict__ out, int act_div, int n, int k,
    int num_experts, const u64* __restrict__ expert_table) {
    constexpr int kSlice = 128;
    constexpr int kA = kSlice * 128;        // 128 routes x 128 codes
    constexpr int kB = 128 * 128 * kHalves; // the weight rows x 128 codes
    constexpr int kStage = kA + kB;

    if (static_cast<int>(blockIdx.x) >= offsets[num_experts + 1]) return;
    const int item = work[blockIdx.x];
    const int expert = item >> 16;
    const int begin = offsets[expert] + (item & 0xFFFF) * kSlice;
    const int count = min(offsets[expert + 1] - begin, kSlice);
    if (count <= 0) return;

    extern __shared__ uint4 wsfp8_smem[];
    const unsigned base = static_cast<unsigned>(__cvta_generic_to_shared(wsfp8_smem));
    u8* ring = reinterpret_cast<u8*>(wsfp8_smem) + ((1024u - (base & 1023u)) & 1023u);
    unsigned long long* full = reinterpret_cast<unsigned long long*>(ring + kStages * kStage);
    unsigned long long* empty = full + kStages;

    const int tid = static_cast<int>(threadIdx.x);
    const int wg = tid >> 7;
    const int blocks = k >> 7;
    const int row0 = static_cast<int>(blockIdx.y) * 128 * kHalves;
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
        // The expert table ([codes, atoms] an expert), resolved once a tile.
        const u8* w_e = expert_plane(weight, static_cast<long long>(n) * k, expert_table, 2, 0, expert)
            + static_cast<long long>(row0) * k;
        const u8* asrc[8];
        int alive[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int r = (tid + 128 * j) >> 3;
            alive[j] = r < count;
            asrc[j] = xq + static_cast<long long>(alive[j] ? order[begin + r] / act_div : 0) * k;
        }
        auto cp = [](void* dst, const void* from, int live) {
            const unsigned at = static_cast<unsigned>(__cvta_generic_to_shared(dst));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(at), "l"(from), "r"(live));
        };
        constexpr int kLag = 1;
        for (int kb = 0; kb < blocks; ++kb) {
            const int s = kb % kStages;
            if (kb >= kStages) wgbf16gemm::bar_wait(empty + s, ((kb / kStages) - 1) & 1);
            u8* st = ring + s * kStage;
            const int k0 = kb * 128;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int u = tid + 128 * j, r = u >> 3, q = u & 7;
                cp(st + r * 128 + ((q ^ (r & 7)) << 4), asrc[j] + k0 + q * 16, alive[j] ? 16 : 0);
            }
#pragma unroll
            for (int j = 0; j < 8 * kHalves; ++j) {
                const int u = tid + 128 * j, r = u >> 3, q = u & 7;
                cp(st + kA + r * 128 + ((q ^ (r & 7)) << 4), w_e + static_cast<long long>(r) * k + k0 + q * 16, 16);
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
        for (int kb = max(0, blocks - kLag); kb < blocks; ++kb) wgbf16gemm::bar_arrive(full + kb % kStages);
        return;
    }

    // The consumers: warpgroup `c` takes routes 64 c .. 64 c + 63.
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
    const int c_wg = wg - 1;
    const int lt = tid & 127, warp = lt >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    // acc[hf][4 j + q]: route row 16 warp + g (+ 8 for q >= 2) of the
    // warpgroup's 64, weight row 128 hf + 8 j + 2 c (+ 1 for odd q).
    int xr[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int r = c_wg * 64 + 16 * warp + g + 8 * h;
        xr[h] = r < count ? order[begin + r] / act_div : -1;
    }
    const ExpertAtoms ea = expert_atoms(atoms, expert_table, expert, n);
    const u8* atoms_at = ea.at + atom_offset(ea.row0 + row0, 0, blocks);
    auto scale = [&](int kb, int h) {
        return xr[h] >= 0 ? static_cast<int>(xe[static_cast<long long>(xr[h]) * blocks + kb]) : -1;
    };
    // A half's weight block exponent: its first row's, 128 rows a block.
    auto wexp = [&](int kb, int hf) { return static_cast<int>(atoms_at[hf * 512 * ((blocks + 3) / 4) + (kb / 4) * 512 + kb % 4]); };
    int e0 = scale(0, 0), e1 = scale(0, 1);
    int ew[kHalves];
#pragma unroll
    for (int hf = 0; hf < kHalves; ++hf) ew[hf] = wexp(0, hf);
    float acc[kHalves][64], blk[64];
#pragma unroll
    for (int hf = 0; hf < kHalves; ++hf)
#pragma unroll
        for (int i = 0; i < 64; ++i) acc[hf][i] = 0.f;
    for (int kb = 0; kb < blocks; ++kb) {
        const int s = kb % kStages;
        // The next block's exponents, in flight under this block's MMAs.
        const int nk = kb + 1 < blocks ? kb + 1 : kb;
        const int n0 = scale(nk, 0), n1 = scale(nk, 1);
        int nw[kHalves];
#pragma unroll
        for (int hf = 0; hf < kHalves; ++hf) nw[hf] = wexp(nk, hf);
        wgbf16gemm::bar_wait(full + s, (kb / kStages) & 1);
        const u8* st = ring + s * kStage;
#pragma unroll
        for (int hf = 0; hf < kHalves; ++hf) {
            asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                wgfp8ws::mma(blk, wgfp8::desc(st + c_wg * 64 * 128 + kk * 32),
                             wgfp8::desc(st + kA + hf * 128 * 128 + kk * 32), kk > 0);
            }
            asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
            asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
            // The stage is read: hand it back before the last half's scaling.
            if (hf == kHalves - 1 && lane == 0) wgbf16gemm::bar_arrive(empty + s);
            const float sw = gfp8::exp2_biased(ew[hf]);
            const float s0 = e0 >= 0 ? sw * gfp8::exp2_biased(e0) : 0.f;
            const float s1 = e1 >= 0 ? sw * gfp8::exp2_biased(e1) : 0.f;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                acc[hf][4 * j + 0] = fmaf(blk[4 * j + 0], s0, acc[hf][4 * j + 0]);
                acc[hf][4 * j + 1] = fmaf(blk[4 * j + 1], s0, acc[hf][4 * j + 1]);
                acc[hf][4 * j + 2] = fmaf(blk[4 * j + 2], s1, acc[hf][4 * j + 2]);
                acc[hf][4 * j + 3] = fmaf(blk[4 * j + 3], s1, acc[hf][4 * j + 3]);
            }
        }
        e0 = n0;
        e1 = n1;
#pragma unroll
        for (int hf = 0; hf < kHalves; ++hf) ew[hf] = nw[hf];
    }

#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int r = c_wg * 64 + 16 * warp + g + 8 * h;
        if (r >= count) continue;
        const long long route = order[begin + r];
#pragma unroll
        for (int hf = 0; hf < kHalves; ++hf) {
            bf16* o = out + route * n + row0 + 128 * hf + 2 * c;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                *reinterpret_cast<unsigned*>(o + 8 * j) = pack_bf16x2(acc[hf][4 * j + 2 * h], acc[hf][4 * j + 2 * h + 1]);
            }
        }
    }
}


namespace wgfp8tma {

// CUtensorMap: 128 opaque bytes the host encodes, passed __grid_constant__.
struct alignas(64) TmaMap {
    unsigned long long raw[16];
};

__device__ __forceinline__ unsigned saddr(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void expect(unsigned long long* bar, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(saddr(bar)), "r"(bytes)
                 : "memory");
}

// An arrival on `bar` once this thread's cp.asyncs so far have landed.
__device__ __forceinline__ void landed(unsigned long long* bar) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(saddr(bar)) : "memory");
}

// A box of `map` at (x = inner, y = outer) into `dst`, completing `bar`.
__device__ __forceinline__ void tma(void* dst, unsigned long long* bar, const TmaMap* map, int x, int y) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4}], [%2];\n" ::"r"(saddr(dst)),
        "l"(reinterpret_cast<unsigned long long>(map)), "r"(saddr(bar)), "r"(x), "r"(y)
        : "memory");
}

}

/// `moe_select_fp8a8_wsgemm<kStages, 2>` persistent, its weight rows landed
/// by TMA (`map_w`: the bank's codes, `[experts n][k]`, 128-byte by 128-row
/// boxes, 128B-swizzled) and the gathered route rows tracked by the stage's
/// mbarrier, so no thread waits on its own copies. A CTA takes (item,
/// 256-row tile) jobs in order, the tiles fastest (an item's route rows read
/// once from memory, an expert's weight rows while its items run), its first
/// its index and each later one off `ticket` (zero at launch), so an SM
/// done early takes more; the producer lands the next job's stages under
/// this one's last MMAs and stores, which leave through shared memory in
/// whole 16-byte chunks; a consumer whose 64 routes are all past the slice
/// runs no MMAs. `n` whole 256s, `k` whole 128s. Grid up to the SMs, 384
/// threads, `kStages * 49152 + 1024 + 192 + 32768` bytes of shared memory.
template <int kStages>
__global__ void __launch_bounds__(384, 1) moe_select_fp8a8_tma(
    const __grid_constant__ wgfp8tma::TmaMap map_w, const u8* __restrict__ xq,
    const u8* __restrict__ xe, const i32* __restrict__ order, const i32* __restrict__ offsets,
    const i32* __restrict__ work, const u8* __restrict__ atoms, bf16* __restrict__ out, int act_div,
    int n, int k, int num_experts, unsigned* __restrict__ ticket,
    const u64* __restrict__ expert_table, u64 window) {
    constexpr int kA = 128 * 128;  // 128 routes x 128 codes
    constexpr int kB = 256 * 128;  // 256 weight rows x 128 codes
    constexpr int kStage = kA + kB;

    const int tiles = n >> 8;
    const int jobs = offsets[num_experts + 1] * tiles;
    if (static_cast<int>(blockIdx.x) >= jobs) return;

    extern __shared__ uint4 tmafp8_smem[];
    const unsigned base = static_cast<unsigned>(__cvta_generic_to_shared(tmafp8_smem));
    u8* ring = reinterpret_cast<u8*>(tmafp8_smem) + ((1024u - (base & 1023u)) & 1023u);
    unsigned long long* full = reinterpret_cast<unsigned long long*>(ring + kStages * kStage);
    unsigned long long* empty = full + kStages;
    // The next two jobs' indices, handed from the producer to the consumers.
    unsigned long long* jfull = empty + kStages;
    unsigned long long* jempty = jfull + 2;
    int* jobq = reinterpret_cast<int*>(jempty + 2);
    // Each consumer's 64 routes by 128 weight rows of bf16 on their way out,
    // 16-byte chunks swizzled by row.
    u8* staged = reinterpret_cast<u8*>(jobq + 4);

    const int tid = static_cast<int>(threadIdx.x);
    const int wg = tid >> 7;
    const int blocks = k >> 7;
    if (tid == 0) {
        for (int s = 0; s < kStages; ++s) {
            // The producer's 128 threads' copies and the weight boxes' bytes.
            wgbf16gemm::bar_init(full + s, 129);
            // Each consumer warp reading the stage.
            wgbf16gemm::bar_init(empty + s, 8);
        }
        for (int q = 0; q < 2; ++q) {
            wgbf16gemm::bar_init(jfull + q, 1);
            wgbf16gemm::bar_init(jempty + q, 8);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();

    // A job's expert, first route and routes, and first weight row.
    auto job = [&](int j, int& expert, int& begin, int& count, int& row0) {
        const int item = work[j / tiles];
        expert = item >> 16;
        begin = offsets[expert] + (item & 0xFFFF) * 128;
        count = min(offsets[expert + 1] - begin, 128);
        row0 = (j % tiles) * 256;
    };

    if (wg == 0) {
        // The producer: the route rows by cp.async, the weight rows by TMA.
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
        int it = 0;
        for (int m = 0;; ++m) {
            // A CTA's first job its index, each later one the next ticket
            // past the grid's: an SM that finishes early takes more.
            if (tid == 0) {
                if (m >= 2) wgbf16gemm::bar_wait(jempty + (m & 1), ((m >> 1) - 1) & 1);
                jobq[m & 1] = m == 0 ? static_cast<int>(blockIdx.x)
                                     : static_cast<int>(gridDim.x + atomicAdd(ticket, 1u));
                wgbf16gemm::bar_arrive(jfull + (m & 1));
            }
            asm volatile("bar.sync 1, 128;\n" ::: "memory");
            const int j = jobq[m & 1];
            if (j >= jobs) break;
            int expert, begin, count, row0;
            job(j, expert, begin, count, row0);
            const u8* asrc[8];
            int alive[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int r = (tid + 128 * i) >> 3;
                alive[i] = r < count;
                asrc[i] = xq + static_cast<long long>(alive[i] ? order[begin + r] / act_div : 0) * k;
            }
            // The expert's first row in the map: its table entry past the
            // window, or `expert n` in the bank.
            const int wrow = static_cast<int>(expert_row(expert_table, 2, 0, expert, n, window, k)) + row0;
            for (int kb = 0; kb < blocks; ++kb, ++it) {
                const int s = it % kStages;
                if (it >= kStages) wgbf16gemm::bar_wait(empty + s, ((it / kStages) - 1) & 1);
                u8* st = ring + s * kStage;
                const int k0 = kb * 128;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int u = tid + 128 * i, r = u >> 3, q = u & 7;
                    const unsigned at_s = wgfp8tma::saddr(st + r * 128 + ((q ^ (r & 7)) << 4));
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(at_s),
                                 "l"(asrc[i] + k0 + q * 16), "r"(alive[i] ? 16 : 0));
                }
                wgfp8tma::landed(full + s);
                if (tid == 0) {
                    wgfp8tma::expect(full + s, kB);
                    wgfp8tma::tma(st + kA, full + s, &map_w, k0, wrow);
                    wgfp8tma::tma(st + kA + 16384, full + s, &map_w, k0, wrow + 128);
                }
            }
        }
        return;
    }

    // The consumers: warpgroup `c` takes routes 64 c .. 64 c + 63.
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
    const int c_wg = wg - 1;
    const int lt = tid & 127, warp = lt >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    int it = 0;
    for (int m = 0;; ++m) {
        wgbf16gemm::bar_wait(jfull + (m & 1), (m >> 1) & 1);
        const int j = jobq[m & 1];
        if (lane == 0) wgbf16gemm::bar_arrive(jempty + (m & 1));
        if (j >= jobs) break;
        int expert, begin, count, row0;
        job(j, expert, begin, count, row0);
        const bool live = count > 64 * c_wg;
        int xr[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = c_wg * 64 + 16 * warp + g + 8 * h;
            xr[h] = r < count ? order[begin + r] / act_div : -1;
        }
        const ExpertAtoms ea = expert_atoms(atoms, expert_table, expert, n);
        const u8* atoms_at = ea.at + atom_offset(ea.row0 + row0, 0, blocks);
        auto scale = [&](int kb, int h) {
            return xr[h] >= 0 ? static_cast<int>(xe[static_cast<long long>(xr[h]) * blocks + kb]) : -1;
        };
        auto wexp = [&](int kb, int hf) {
            return static_cast<int>(atoms_at[hf * 512 * ((blocks + 3) / 4) + (kb / 4) * 512 + kb % 4]);
        };
        int e0 = scale(0, 0), e1 = scale(0, 1);
        int ew[2] = {wexp(0, 0), wexp(0, 1)};
        float acc[2][64], blk[64];
#pragma unroll
        for (int hf = 0; hf < 2; ++hf)
#pragma unroll
            for (int i = 0; i < 64; ++i) acc[hf][i] = 0.f;
        for (int kb = 0; kb < blocks; ++kb, ++it) {
            const int s = it % kStages;
            const int nk = kb + 1 < blocks ? kb + 1 : kb;
            const int n0 = scale(nk, 0), n1 = scale(nk, 1);
            const int nw[2] = {wexp(nk, 0), wexp(nk, 1)};
            wgbf16gemm::bar_wait(full + s, (it / kStages) & 1);
            if (live) {
                // The cp.async rows, seen by the MMA's async proxy.
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                const u8* st = ring + s * kStage;
#pragma unroll
                for (int hf = 0; hf < 2; ++hf) {
                    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
                    for (int kk = 0; kk < 4; ++kk) {
                        wgfp8ws::mma(blk, wgfp8::desc(st + c_wg * 64 * 128 + kk * 32),
                                     wgfp8::desc(st + kA + hf * 128 * 128 + kk * 32), kk > 0);
                    }
                    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
                    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
                    // The stage is read: hand it back before the last half's scaling.
                    if (hf == 1 && lane == 0) wgbf16gemm::bar_arrive(empty + s);
                    const float sw = gfp8::exp2_biased(ew[hf]);
                    const float s0 = e0 >= 0 ? sw * gfp8::exp2_biased(e0) : 0.f;
                    const float s1 = e1 >= 0 ? sw * gfp8::exp2_biased(e1) : 0.f;
#pragma unroll
                    for (int q = 0; q < 16; ++q) {
                        acc[hf][4 * q + 0] = fmaf(blk[4 * q + 0], s0, acc[hf][4 * q + 0]);
                        acc[hf][4 * q + 1] = fmaf(blk[4 * q + 1], s0, acc[hf][4 * q + 1]);
                        acc[hf][4 * q + 2] = fmaf(blk[4 * q + 2], s1, acc[hf][4 * q + 2]);
                        acc[hf][4 * q + 3] = fmaf(blk[4 * q + 3], s1, acc[hf][4 * q + 3]);
                    }
                }
            } else if (lane == 0) {
                wgbf16gemm::bar_arrive(empty + s);
            }
            e0 = n0;
            e1 = n1;
            ew[0] = nw[0];
            ew[1] = nw[1];
        }
        if (!live) continue;
        // The sums through shared memory, so each route's row leaves in
        // whole 16-byte chunks.
        u8* tile = staged + c_wg * 16384;
        long long route[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = (lt >> 4) + 8 * i;
            route[i] = c_wg * 64 + r < count ? order[begin + c_wg * 64 + r] : -1;
        }
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = 16 * warp + g + 8 * h;
#pragma unroll
                for (int q = 0; q < 16; ++q) {
                    *reinterpret_cast<unsigned*>(tile + r * 256 + ((q ^ (r & 7)) << 4) + 4 * c) =
                        pack_bf16x2(acc[hf][4 * q + 2 * h], acc[hf][4 * q + 2 * h + 1]);
                }
            }
            asm volatile("bar.sync %0, 128;\n" ::"r"(2 + c_wg) : "memory");
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int r = (lt >> 4) + 8 * i, q = lt & 15;
                if (route[i] < 0) continue;
                *reinterpret_cast<uint4*>(out + route[i] * n + row0 + 128 * hf + 8 * q) =
                    *reinterpret_cast<const uint4*>(tile + r * 256 + ((q ^ (r & 7)) << 4));
            }
            asm volatile("bar.sync %0, 128;\n" ::"r"(2 + c_wg) : "memory");
        }
    }
}
}
