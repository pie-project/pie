#include "linear/codec.cuh"
#include "linear/wgmma.cuh"
#include "linear/grouped_fp8.cuh"
#include "linear/moe_w4.cuh"
#include "linear/w8a16.cuh"

namespace pie::linear {

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

}
