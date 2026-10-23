#include "linear/w8a16.cuh"

namespace pie::linear {


template <int kMT, int kNSub, int kWarpsN, int kWarpsK, int kStages, int kGran>
__global__ void __launch_bounds__(kWarpsN * kWarpsK * 32) fp8_w8a16_mma(
    const bf16* __restrict__ x,
    int x_stride,
    const u8* __restrict__ codes,
    const u8* __restrict__ atoms,
    bf16* __restrict__ y,
    int y_stride,
    int m,
    int n,
    int k,
    const u32* __restrict__ win)
{
    using namespace w8a16;
    static_assert(kGran == 32 || kGran == 128, "a block is 32 or 128 columns of K");
    static_assert(kWarpsK <= 15, "a K group's barrier id is 1 + its index");
    constexpr int kBM = 16 * kMT;
    constexpr int kBN = 8 * kNSub * kWarpsN;
    constexpr int kGroupThreads = 32 * kWarpsN;
    constexpr int kStageElems = kBM * 64;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int wn = warp % kWarpsN;
    const int wk = warp / kWarpsN;
    const int g = lane >> 2;
    const int t = lane & 3;

    int rows = m;
    if (win != nullptr) rows = min(rows, static_cast<int>(win[0]));
    // A tile of `kBM` rows a grid row: the rows past it a later tile's.
    const int m0 = static_cast<int>(blockIdx.y) * kBM;
    if (rows <= m0) return;
    rows -= m0;
    x += static_cast<long long>(m0) * x_stride;
    y += static_cast<long long>(m0) * y_stride;

    extern __shared__ __align__(16) unsigned char w8a16_smem[];
    bf16* ring = reinterpret_cast<bf16*>(w8a16_smem) + wk * kStages * kStageElems;

    const int n0 = static_cast<int>(blockIdx.x) * kBN + wn * 8 * kNSub;
    const int steps = k >> 6;
    const int kblocks = k / kGran;
    const int iters = steps > wk ? (steps - wk + kWarpsK - 1) / kWarpsK : 0;
    const int gtid = wn * 32 + lane;

    // The group's share of a step's activation tile: `kBM` rows of eight
    // 16-byte chunks, chunk `c` of row `r` at `c ^ (r & 7)`.
    auto issue = [&](int it) {
        if (it < iters) {
            const int k0 = (wk + it * kWarpsK) * 64;
            bf16* dst = ring + (it % kStages) * kStageElems;
#pragma unroll
            for (int i = gtid; i < kBM * 8; i += kGroupThreads) {
                const int r = i >> 3;
                const int c = i & 7;
                const bf16* src = x + static_cast<long long>(min(r, rows - 1)) * x_stride + k0 + c * 8;
                cp_async16(smem_addr(dst + r * 64 + ((c ^ (r & 7)) * 8)), src, r < rows ? 16 : 0);
            }
        }
        cp_commit();
    };

    // A lane's columns' codes and exponents for a step.
    auto fetch = [&](int it, uint4 (&b)[kNSub], unsigned (&f)[kNSub]) {
        if (it >= iters) return;
        const int k0 = (wk + it * kWarpsK) * 64;
        const int kb = kGran == 128 ? (k0 >> 7) : ((k0 + 16 * t) >> 5);
#pragma unroll
        for (int j = 0; j < kNSub; ++j) {
            const int col = min(n0 + 8 * j + g, n - 1);
            b[j] = ld_nc(codes + static_cast<long long>(col) * k + k0 + 16 * t);
            f[j] = factor2(atoms[atom_offset(col, kb, kblocks)]);
        }
    };

    float acc[kMT][kNSub][4];
#pragma unroll
    for (int i = 0; i < kMT; ++i)
#pragma unroll
        for (int j = 0; j < kNSub; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) issue(s);

    // The codes and exponents `kDepth` steps ahead, a ring of registers a
    // step refilled as it is spent.
    constexpr int kDepth = 3;
    uint4 bring[kDepth][kNSub];
    unsigned fring[kDepth][kNSub];
#pragma unroll
    for (int d = 0; d < kDepth; ++d) fetch(d, bring[d], fring[d]);

    for (int it0 = 0; it0 < iters; it0 += kDepth) {
#pragma unroll
        for (int d = 0; d < kDepth; ++d) {
            const int it = it0 + d;
            if (it >= iters) break;
            cp_wait<kStages - 2>();
            group_sync(1 + wk, kGroupThreads);
            const bf16* tile = ring + (it % kStages) * kStageElems;
#pragma unroll
            for (int mt = 0; mt < kMT; ++mt) {
                const int ra = mt * 16 + g;
                const int rb = ra + 8;
                const uint4 a_lo = *reinterpret_cast<const uint4*>(tile + ra * 64 + (((2 * t) ^ (ra & 7)) * 8));
                const uint4 a_hi = *reinterpret_cast<const uint4*>(tile + ra * 64 + (((2 * t + 1) ^ (ra & 7)) * 8));
                const uint4 b_lo = *reinterpret_cast<const uint4*>(tile + rb * 64 + (((2 * t) ^ (rb & 7)) * 8));
                const uint4 b_hi = *reinterpret_cast<const uint4*>(tile + rb * 64 + (((2 * t + 1) ^ (rb & 7)) * 8));
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const uint4& ar = c < 2 ? a_lo : a_hi;
                    const uint4& br = c < 2 ? b_lo : b_hi;
                    const unsigned a[4] = {
                        (c & 1) ? ar.z : ar.x,
                        (c & 1) ? br.z : br.x,
                        (c & 1) ? ar.w : ar.y,
                        (c & 1) ? br.w : br.y,
                    };
#pragma unroll
                    for (int j = 0; j < kNSub; ++j) {
                        const unsigned w = word_of(bring[d][j], c);
                        const unsigned bf[2] = {widen2(w, 0x9180u, fring[d][j]), widen2(w, 0xB3A2u, fring[d][j])};
                        nvcuda::wmma::detail::mma_m16n8k16(acc[mt][j], a, bf, acc[mt][j]);
                    }
                }
            }
            issue(it + kStages - 1);
            fetch(it + kDepth, bring[d], fring[d]);
        }
    }
    cp_wait<0>();

    if constexpr (kWarpsK == 1) {
#pragma unroll
        for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
            for (int j = 0; j < kNSub; ++j) {
                const int col = n0 + 8 * j + 2 * t;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int r = mt * 16 + g + 8 * h;
                    if (r >= rows) continue;
                    bf16* out = y + static_cast<long long>(r) * y_stride + col;
                    if (col + 1 < n) {
                        *reinterpret_cast<unsigned*>(out) = pack_bf16x2(acc[mt][j][2 * h], acc[mt][j][2 * h + 1]);
                    } else if (col < n) {
                        out[0] = f32_to_bf16(acc[mt][j][2 * h]);
                    }
                }
            }
    } else {
        __syncthreads();
        float* red = reinterpret_cast<float*>(w8a16_smem);
#pragma unroll
        for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
            for (int j = 0; j < kNSub; ++j) {
                const int cl = wn * 8 * kNSub + 8 * j + 2 * t;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int r = mt * 16 + g + 8 * h;
                    float* at = red + (wk * kBM + r) * kBN + cl;
                    at[0] = acc[mt][j][2 * h];
                    at[1] = acc[mt][j][2 * h + 1];
                }
            }
        __syncthreads();
        const int live = min(rows, kBM);
        for (int i = threadIdx.x; i < live * kBN; i += kWarpsN * kWarpsK * 32) {
            const int r = i / kBN;
            const int cl = i % kBN;
            const int col = static_cast<int>(blockIdx.x) * kBN + cl;
            if (col >= n) continue;
            float sum = 0.f;
#pragma unroll
            for (int w = 0; w < kWarpsK; ++w) sum += red[(w * kBM + r) * kBN + cl];
            y[static_cast<long long>(r) * y_stride + col] = f32_to_bf16(sum);
        }
    }
}

// The same product at a wide fire's rows as a tiled GEMM: a CTA `kBM`
// activation rows by 128 weight rows, its eight warps 2 x 4 over the tile
// (a warp `kBM / 2` rows by 32 columns), both operands staged through a
// `kStages` cp.async ring 64 columns of K at a time -- the activations as
// `fp8_w8a16_mma` lays them, the codes a weight row's 64 bytes a stage row
// (a quarter-warp's 16-byte reads land on 32 banks unswizzled) -- and each
// warp widening its columns' codes from shared memory as it reads them.
// With `splits` > 1 a CTA takes the `blockIdx.z`-th of K's steps and lays
// its f32 sums in `part` (a `[splits, m, n]` slab) for `fp8_w8a16_fold`:
// a plane too narrow to fill the card at a batch's rows, split along K.
template <int kBM, int kStages, int kGran>
__global__ void __launch_bounds__(256) fp8_w8a16_gemm(
    const bf16* __restrict__ x,
    int x_stride,
    const u8* __restrict__ codes,
    const u8* __restrict__ atoms,
    bf16* __restrict__ y,
    int y_stride,
    int m,
    int n,
    int k,
    const u32* __restrict__ win,
    float* __restrict__ part,
    int splits)
{
    using namespace w8a16;
    static_assert(kGran == 32 || kGran == 128, "a block is 32 or 128 columns of K");
    static_assert(kBM == 64 || kBM == 128, "a CTA is 64 or 128 rows");
    constexpr int kBN = 128;
    constexpr int kMT = kBM / 32;  // m16 tiles a warp
    constexpr int kNSub = 4;        // n8 tiles a warp
    constexpr int kABytes = kBM * 128;
    constexpr int kStageBytes = kABytes + kBN * 64;
    constexpr int kACopies = kBM * 8 / 256;
    constexpr int kBCopies = kBN * 4 / 256;

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp >> 2;
    const int wn = warp & 3;
    const int g = lane >> 2;
    const int t = lane & 3;

    int rows = m;
    if (win != nullptr) rows = min(rows, static_cast<int>(win[0]));
    const int m0 = static_cast<int>(blockIdx.y) * kBM;
    if (rows <= m0) return;
    rows -= m0;
    x += static_cast<long long>(m0) * x_stride;
    y += static_cast<long long>(m0) * y_stride;
    const int n0 = static_cast<int>(blockIdx.x) * kBN;
    // This CTA's run of K's 64-column steps.
    const int per = ((k >> 6) + splits - 1) / splits;
    const int first = static_cast<int>(blockIdx.z) * per;
    const int steps = max(0, min(k >> 6, first + per) - first);
    const int kblocks = k / kGran;
    // The blocks this CTA's steps read, from `kb0`.
    const int kb0 = first * 64 / kGran;
    const int spans = steps > 0 ? ((first + steps) * 64 - 1) / kGran + 1 - kb0 : 0;

    // The stages, then the CTA's block exponents (`kBN` rows of `spans`),
    // laid once.
    extern __shared__ __align__(16) unsigned char w8a16_gemm_smem[];
    u8* exps = w8a16_gemm_smem + kStages * kStageBytes;
    for (int i = tid; i < kBN * spans; i += 256) {
        const int r = i / spans;
        exps[i] = atoms[atom_offset(min(n0 + r, n - 1), kb0 + i - r * spans, kblocks)];
    }
    const unsigned smem0 = smem_addr(w8a16_gemm_smem);

    // Each thread's copies a stage: their sources (advanced 64 columns a
    // step) and destinations within a stage.
    const bf16* a_src[kACopies];
    unsigned a_dst[kACopies];
    int a_bytes[kACopies];
#pragma unroll
    for (int j = 0; j < kACopies; ++j) {
        const int i = tid + 256 * j;
        const int r = i >> 3;
        const int c = i & 7;
        a_src[j] = x + static_cast<long long>(min(r, rows - 1)) * x_stride + c * 8;
        a_dst[j] = (r * 64 + ((c ^ (r & 7)) * 8)) * 2;
        a_bytes[j] = r < rows ? 16 : 0;
    }
    const u8* b_src[kBCopies];
    unsigned b_dst[kBCopies];
#pragma unroll
    for (int j = 0; j < kBCopies; ++j) {
        const int i = tid + 256 * j;
        const int r = i >> 2;
        const int c = i & 3;
        b_src[j] = codes + static_cast<long long>(min(n0 + r, n - 1)) * k + c * 16;
        b_dst[j] = kABytes + r * 64 + c * 16;
    }
    int issued = 0;
    auto issue = [&]() {
        if (issued < steps) {
            const unsigned base = smem0 + (issued % kStages) * kStageBytes;
            const long long ka = static_cast<long long>(first + issued) * 64;
#pragma unroll
            for (int j = 0; j < kACopies; ++j) cp_async16(base + a_dst[j], a_src[j] + ka, a_bytes[j]);
#pragma unroll
            for (int j = 0; j < kBCopies; ++j) cp_async16(base + b_dst[j], b_src[j] + ka, 16);
        }
        ++issued;
        cp_commit();
    };

    // A lane's shared-memory reads a stage, as byte offsets in it.
    unsigned a_at[kMT][4];
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt) {
        const int ra = wm * (kBM / 2) + mt * 16 + g;
        const int rb = ra + 8;
        a_at[mt][0] = (ra * 64 + (((2 * t) ^ (ra & 7)) * 8)) * 2;
        a_at[mt][1] = (ra * 64 + (((2 * t + 1) ^ (ra & 7)) * 8)) * 2;
        a_at[mt][2] = (rb * 64 + (((2 * t) ^ (rb & 7)) * 8)) * 2;
        a_at[mt][3] = (rb * 64 + (((2 * t + 1) ^ (rb & 7)) * 8)) * 2;
    }
    const unsigned b_at = kABytes + (wn * 32 + g) * 64 + t * 16;
    const u8* my_exps = exps + (wn * 32 + g) * spans - kb0;

    float acc[kMT][kNSub][4];
#pragma unroll
    for (int i = 0; i < kMT; ++i)
#pragma unroll
        for (int j = 0; j < kNSub; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) issue();
    __syncthreads();  // the exponents laid

    int slot = 0;
    for (int step = 0; step < steps; ++step) {
        cp_wait<kStages - 2>();
        __syncthreads();
        const unsigned char* stage = w8a16_gemm_smem + slot * kStageBytes;
        const int kb = kGran == 128 ? ((first + step) >> 1) : (((first + step) * 64 + 16 * t) >> 5);
        unsigned f[kNSub];
        uint4 bw[kNSub];
#pragma unroll
        for (int j = 0; j < kNSub; ++j) {
            f[j] = factor2(my_exps[8 * j * spans + kb]);
            bw[j] = *reinterpret_cast<const uint4*>(stage + b_at + j * 8 * 64);
        }
#pragma unroll
        for (int mt = 0; mt < kMT; ++mt) {
            const uint4 a_lo = *reinterpret_cast<const uint4*>(stage + a_at[mt][0]);
            const uint4 a_hi = *reinterpret_cast<const uint4*>(stage + a_at[mt][1]);
            const uint4 b_lo = *reinterpret_cast<const uint4*>(stage + a_at[mt][2]);
            const uint4 b_hi = *reinterpret_cast<const uint4*>(stage + a_at[mt][3]);
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const uint4& ar = c < 2 ? a_lo : a_hi;
                const uint4& br = c < 2 ? b_lo : b_hi;
                const unsigned a[4] = {
                    (c & 1) ? ar.z : ar.x,
                    (c & 1) ? br.z : br.x,
                    (c & 1) ? ar.w : ar.y,
                    (c & 1) ? br.w : br.y,
                };
#pragma unroll
                for (int j = 0; j < kNSub; ++j) {
                    const unsigned w = word_of(bw[j], c);
                    const unsigned bf[2] = {widen2(w, 0x9180u, f[j]), widen2(w, 0xB3A2u, f[j])};
                    nvcuda::wmma::detail::mma_m16n8k16(acc[mt][j], a, bf, acc[mt][j]);
                }
            }
        }
        issue();
        slot = slot + 1 == kStages ? 0 : slot + 1;
    }
    cp_wait<0>();

#pragma unroll
    for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
        for (int j = 0; j < kNSub; ++j) {
            const int col = n0 + wn * 32 + 8 * j + 2 * t;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = wm * (kBM / 2) + mt * 16 + g + 8 * h;
                if (r >= rows) continue;
                if (part != nullptr) {
                    float* out = part + (static_cast<long long>(blockIdx.z) * m + m0 + r) * n + col;
                    if (col + 1 < n) {
                        *reinterpret_cast<float2*>(out) = make_float2(acc[mt][j][2 * h], acc[mt][j][2 * h + 1]);
                    } else if (col < n) {
                        out[0] = acc[mt][j][2 * h];
                    }
                    continue;
                }
                bf16* out = y + static_cast<long long>(r) * y_stride + col;
                if (col + 1 < n) {
                    *reinterpret_cast<unsigned*>(out) = pack_bf16x2(acc[mt][j][2 * h], acc[mt][j][2 * h + 1]);
                } else if (col < n) {
                    out[0] = f32_to_bf16(acc[mt][j][2 * h]);
                }
            }
        }
}

// `fp8_w8a16_gemm`'s K splits summed: `y = sum_z part[z]`, a thread a pair
// of columns (`n` even).
__global__ void __launch_bounds__(256) fp8_w8a16_fold(
    const float* __restrict__ part,
    bf16* __restrict__ y,
    int y_stride,
    int m,
    int n,
    int splits,
    const u32* __restrict__ win)
{
    int rows = m;
    if (win != nullptr) rows = min(rows, static_cast<int>(win[0]));
    const long long i = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) * 2;
    if (i >= static_cast<long long>(rows) * n) return;
    const long long plane = static_cast<long long>(m) * n;
    float2 sum = *reinterpret_cast<const float2*>(part + i);
    for (int z = 1; z < splits; ++z) {
        const float2 v = *reinterpret_cast<const float2*>(part + z * plane + i);
        sum.x += v.x;
        sum.y += v.y;
    }
    const int r = static_cast<int>(i / n);
    const int c = static_cast<int>(i % n);
    *reinterpret_cast<unsigned*>(y + static_cast<long long>(r) * y_stride + c) = pack_bf16x2(sum.x, sum.y);
}

// The same plane widened whole to bf16 (`out`, `[n, k]`), for a wide fire's
// cuBLAS GEMM: a thread sixteen codes.
template <int kGran>
__global__ void __launch_bounds__(256) fp8_atoms_to_bf16(
    const u8* __restrict__ codes,
    const u8* __restrict__ atoms,
    bf16* __restrict__ out,
    int n,
    int k)
{
    using namespace w8a16;
    const long long chunk = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int per_row = k >> 4;
    if (chunk >= static_cast<long long>(n) * per_row) return;
    const int row = static_cast<int>(chunk / per_row);
    const int c0 = static_cast<int>(chunk % per_row) * 16;
    const uint4 v = *reinterpret_cast<const uint4*>(codes + static_cast<long long>(row) * k + c0);
    const unsigned f2 = factor2(atoms[atom_offset(row, c0 / kGran, k / kGran)]);
    uint4 lo, hi;
    lo.x = widen2(v.x, 0x9180u, f2);
    lo.y = widen2(v.x, 0xB3A2u, f2);
    lo.z = widen2(v.y, 0x9180u, f2);
    lo.w = widen2(v.y, 0xB3A2u, f2);
    hi.x = widen2(v.z, 0x9180u, f2);
    hi.y = widen2(v.z, 0xB3A2u, f2);
    hi.z = widen2(v.w, 0x9180u, f2);
    hi.w = widen2(v.w, 0xB3A2u, f2);
    uint4* dst = reinterpret_cast<uint4*>(out + static_cast<long long>(row) * k + c0);
    dst[0] = lo;
    dst[1] = hi;
}

// `y[m] = x[m] . W^T` for an affine 4-bit plane (MLX's: `[n, k / 2]` codes
// low nibble first, a bf16 scale and bias a row a 64 columns, `w = s q + b`)
// on mma.sync, for a decode's few rows: `fp8_w8a16_mma`'s walk, each lane's
// sixteen columns of a step its row's eight bytes, widened in registers.
template <int kMT, int kNSub, int kWarpsN, int kWarpsK, int kStages, int kDepth>
__global__ void __launch_bounds__(kWarpsN * kWarpsK * 32) affine_w4a16_mma(
    const bf16* __restrict__ x,
    const u8* __restrict__ codes,
    const bf16* __restrict__ scales,
    const bf16* __restrict__ biases,
    bf16* __restrict__ y,
    int m,
    int n,
    int k,
    const u32* __restrict__ win)
{
    using namespace w8a16;
    static_assert(kWarpsK <= 15, "a K group's barrier id is 1 + its index");
    constexpr int kBM = 16 * kMT;
    constexpr int kBN = 8 * kNSub * kWarpsN;
    constexpr int kGroupThreads = 32 * kWarpsN;
    constexpr int kStageElems = kBM * 64;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int wn = warp % kWarpsN;
    const int wk = warp / kWarpsN;
    const int g = lane >> 2;
    const int t = lane & 3;

    int rows = m;
    if (win != nullptr) rows = min(rows, static_cast<int>(win[0]));
    const int m0 = static_cast<int>(blockIdx.y) * kBM;
    if (rows <= m0) return;
    rows -= m0;
    x += static_cast<long long>(m0) * k;
    y += static_cast<long long>(m0) * n;

    extern __shared__ __align__(16) unsigned char w8a16_smem[];
    bf16* ring = reinterpret_cast<bf16*>(w8a16_smem) + wk * kStages * kStageElems;

    const int n0 = static_cast<int>(blockIdx.x) * kBN + wn * 8 * kNSub;
    const int steps = k >> 6;
    const int iters = steps > wk ? (steps - wk + kWarpsK - 1) / kWarpsK : 0;
    const int gtid = wn * 32 + lane;
    const long long row_bytes = k >> 1;

    auto issue = [&](int it) {
        if (it < iters) {
            const int k0 = (wk + it * kWarpsK) * 64;
            bf16* dst = ring + (it % kStages) * kStageElems;
#pragma unroll
            for (int i = gtid; i < kBM * 8; i += kGroupThreads) {
                const int r = i >> 3;
                const int c = i & 7;
                const bf16* src = x + static_cast<long long>(min(r, rows - 1)) * k + k0 + c * 8;
                cp_async16(smem_addr(dst + r * 64 + ((c ^ (r & 7)) * 8)), src, r < rows ? 16 : 0);
            }
        }
        cp_commit();
    };

    // A lane's columns' codes, scales and biases for a step (its group).
    auto fetch = [&](int it, uint2 (&b)[kNSub], unsigned (&s)[kNSub], unsigned (&z)[kNSub]) {
        if (it >= iters) return;
        const int step = wk + it * kWarpsK;
#pragma unroll
        for (int j = 0; j < kNSub; ++j) {
            const int col = min(n0 + 8 * j + g, n - 1);
            b[j] = *reinterpret_cast<const uint2*>(codes + col * row_bytes + step * 32 + 8 * t);
            const long long at = static_cast<long long>(col) * steps + step;
            const unsigned sv = scales[at].raw;
            const unsigned bv = biases[at].raw;
            s[j] = sv | (sv << 16);
            z[j] = bv | (bv << 16);
        }
    };

    float acc[kMT][kNSub][4];
#pragma unroll
    for (int i = 0; i < kMT; ++i)
#pragma unroll
        for (int j = 0; j < kNSub; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) issue(s);

    // The codes, scales and biases `kDepth` steps ahead (deeper spills).
    uint2 bring[kDepth][kNSub];
    unsigned sring[kDepth][kNSub];
    unsigned zring[kDepth][kNSub];
#pragma unroll
    for (int d = 0; d < kDepth; ++d) fetch(d, bring[d], sring[d], zring[d]);

    for (int it0 = 0; it0 < iters; it0 += kDepth) {
#pragma unroll
        for (int d = 0; d < kDepth; ++d) {
            const int it = it0 + d;
            if (it >= iters) break;
            cp_wait<kStages - 2>();
            group_sync(1 + wk, kGroupThreads);
            const bf16* tile = ring + (it % kStages) * kStageElems;
#pragma unroll
            for (int mt = 0; mt < kMT; ++mt) {
                const int ra = mt * 16 + g;
                const int rb = ra + 8;
                const uint4 a_lo = *reinterpret_cast<const uint4*>(tile + ra * 64 + (((2 * t) ^ (ra & 7)) * 8));
                const uint4 a_hi = *reinterpret_cast<const uint4*>(tile + ra * 64 + (((2 * t + 1) ^ (ra & 7)) * 8));
                const uint4 b_lo = *reinterpret_cast<const uint4*>(tile + rb * 64 + (((2 * t) ^ (rb & 7)) * 8));
                const uint4 b_hi = *reinterpret_cast<const uint4*>(tile + rb * 64 + (((2 * t + 1) ^ (rb & 7)) * 8));
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const uint4& ar = c < 2 ? a_lo : a_hi;
                    const uint4& br = c < 2 ? b_lo : b_hi;
                    const unsigned a[4] = {
                        (c & 1) ? ar.z : ar.x,
                        (c & 1) ? br.z : br.x,
                        (c & 1) ? ar.w : ar.y,
                        (c & 1) ? br.w : br.y,
                    };
#pragma unroll
                    for (int j = 0; j < kNSub; ++j) {
                        // Chunk c's four codes: the lane's columns 16t + 4c .. + 3.
                        const unsigned word = c < 2 ? bring[d][j].x : bring[d][j].y;
                        const unsigned run = (word >> (16 * (c & 1))) & 0xFFFFu;
                        const unsigned bf[2] = {affine2(run, 0, sring[d][j], zring[d][j]),
                                                affine2(run, 1, sring[d][j], zring[d][j])};
                        nvcuda::wmma::detail::mma_m16n8k16(acc[mt][j], a, bf, acc[mt][j]);
                    }
                }
            }
            issue(it + kStages - 1);
            fetch(it + kDepth, bring[d], sring[d], zring[d]);
        }
    }
    cp_wait<0>();

    __syncthreads();
    float* red = reinterpret_cast<float*>(w8a16_smem);
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
        for (int j = 0; j < kNSub; ++j) {
            const int cl = wn * 8 * kNSub + 8 * j + 2 * t;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = mt * 16 + g + 8 * h;
                float* at = red + (wk * kBM + r) * kBN + cl;
                at[0] = acc[mt][j][2 * h];
                at[1] = acc[mt][j][2 * h + 1];
            }
        }
    __syncthreads();
    const int live = min(rows, kBM);
    for (int i = threadIdx.x; i < live * kBN; i += kWarpsN * kWarpsK * 32) {
        const int r = i / kBN;
        const int cl = i % kBN;
        const int col = static_cast<int>(blockIdx.x) * kBN + cl;
        if (col >= n) continue;
        float sum = 0.f;
#pragma unroll
        for (int w = 0; w < kWarpsK; ++w) sum += red[(w * kBM + r) * kBN + cl];
        y[static_cast<long long>(r) * n + col] = f32_to_bf16(sum);
    }
}

}
