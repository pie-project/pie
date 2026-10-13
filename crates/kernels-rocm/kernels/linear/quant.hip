#pragma once

// Quantized linear kernels: the dense affine projections and the casts a
// load runs, then the MoE expert banks.

#include "linear/affine.hip"

namespace pie::linear {

// ------------------------------------------------------------- affine --

// A factor read as its bits: F's 16 bits, or a zero point's byte.
template <class F, int kBits, int kOffset>
__device__ __forceinline__ Factor factor_bits(u32 s, u32 off) {
    auto real = [](u32 h) {
        if constexpr (is_same<F, bf16>::value) {
            return __builtin_bit_cast(float, h << 16);
        } else {
            return static_cast<float>(__builtin_bit_cast(f16, static_cast<u16>(h)));
        }
    };
    if constexpr (kOffset == kOffPost || kOffset == kOffPreReal) {
        return {real(s), real(off)};
    } else if constexpr (kOffset == kOffPreInt) {
        return {real(s), static_cast<float>(off)};
    } else {
        return {real(s), static_cast<float>(1 << (kBits - 1))};
    }
}

// y[m, n] = act[m] · W[n] over an affine-quantized W, for the few rows of a
// decode: bound by streaming the codes once. The products run on the
// matrix cores with the activation as A and the codes as B, both raw, so a
// product sums Σ x·(c + magic); each group of codes then folds into f32 as
//
//   Σ x·w = s·Σ x·(c + magic) + (o − magic·s)·Σ x
//
// the row sums Σ x from one more product against ones.
//
// SHAPE 4, the 4 x 4 x 4 core (one to four activation rows): sixteen blocks
// of four lanes, block b's lanes holding four weight rows and four
// activation rows over the same LANE codes at k = 16·LANE·p + LANE·b, so a
// read is four weight rows of 16·LANE codes each. A block sums its own k;
// the sixteen blocks add up at the end.
//
// SHAPE 16, the 16 x 16 x 16 core (up to 64 rows): lane group g of the
// four holds LANE codes of sixteen weight rows at k = 4·LANE·p + LANE·g,
// one group of codes between them (4·LANE codes at most a group), and
// the core sums the four.
//
// A wavefront owns NT tiles of SHAPE weight rows and RS tiles of SHAPE
// activation rows (from a launch row of blocks blockIdx.y) and walks a
// SPLIT-th of K, R pieces at a time: a round's reads are all issued by
// hand before the first is waited on, and each piece multiplies as it
// lands. A piece past K (or past the wavefront's share) rereads live codes
// and counts as zero.
template <class T, class F, int kBits, int kOffset, int kGroup, int SHAPE, int LANE, int NT, int RS, int SPLIT, int R>
__global__ __launch_bounds__(kWave* SPLIT) void matmul_affine_decode(
    const T* __restrict__ act, const u8* __restrict__ codes, const u8* __restrict__ scales,
    const u8* __restrict__ biases, T* __restrict__ out, int m, int n, int k, const AffineSeat* __restrict__ seat,
    const u32* __restrict__ win) {
    using U = Unpack<T, kBits>;
    constexpr int kSlots = kWave / SHAPE;     // lane groups along k
    constexpr int kBytes = LANE * kBits / 8;  // codes a lane reads a piece
    constexpr int kWords = kBytes / 4;
    constexpr int kPiece = kSlots * LANE;
    constexpr int kXBytes = 2 * LANE < 16 ? 2 * LANE : 16;  // activation bytes a read
    constexpr int kXReads = 2 * LANE / kXBytes;
    constexpr bool kOffsets = kOffset != kOffPreConst;
    constexpr int kReads = NT * (kOffsets ? 3 : 2) + RS * kXReads;  // a piece's reads
    static_assert(kWave == 64 && (SHAPE == 4 || SHAPE == 16), "a 4 x 4 x 4 or 16 x 16 x 16 core");
    static_assert(kBytes == 4 || kBytes == 8 || kBytes == 16, "a lane reads one 4-, 8- or 16-byte access");
    static_assert(kGroup % (SHAPE == 4 ? LANE : 4 * LANE) == 0 && LANE % 4 == 0,
                  "a product's codes are whole 4-code steps of one group");
    static_assert(R >= 1 && R <= 4 && R * kReads < 64, "a round's reads fit the count");

    const int rows = win == nullptr ? m : min(m, static_cast<int>(win[0]));
    const int r0 = blockIdx.y * SHAPE * RS;
    if (r0 >= rows) return;
    const AffinePlanes planes = seated(codes, scales, biases, seat);

    const int l = lane(), wave = threadIdx.x / kWave;
    const int across = l % SHAPE, slot = l / SHAPE;  // a lane's row in a tile, and its k
    const int col0 = blockIdx.x * SHAPE * NT;
    const int groups = k / kGroup;
    const int pieces = (k + kPiece - 1) / kPiece;
    // The pieces this wavefront walks, a whole number of rounds.
    const int per = ((pieces + SPLIT - 1) / SPLIT + R - 1) / R * R;
    const int mine = max(0, min(per, pieces - wave * per));

    const u8* wp[NT];
    long long fp[NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        const long long row = min(col0 + SHAPE * nt + across, n - 1);
        wp[nt] = planes.codes + row * k * kBits / 8;
        fp[nt] = row * groups;
    }
    const T* xp[RS];
#pragma unroll
    for (int rs = 0; rs < RS; ++rs)
        xp[rs] = act + static_cast<long long>(min(r0 + SHAPE * rs + across, rows - 1)) * k;

    struct Piece {
        Words<kWords> w[NT];
        Words<kXBytes / 4> x[RS][kXReads];
        u32 s[NT], o[NT];
    };
    auto first = [&](int r) { return (wave * per + r) * kPiece + LANE * slot; };
    auto issue = [&](Piece& at, int r) {
        const int kc = min(first(min(r, mine - 1)), k - LANE);
        const int group = kc / kGroup;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            issue_read<kBytes>(at.w[nt], wp[nt] + kc * kBits / 8);
            issue_read_small(at.s[nt], planes.scales + (fp[nt] + group) * 2);
            if constexpr (kOffsets) {
                constexpr bool kByte = kOffset == kOffPreInt;
                issue_read_small<kByte>(at.o[nt], planes.biases + (fp[nt] + group) * (kByte ? 1 : 2));
            }
        }
#pragma unroll
        for (int rs = 0; rs < RS; ++rs)
#pragma unroll
            for (int q = 0; q < kXReads; ++q) issue_read<kXBytes>(at.x[rs][q], xp[rs] + kc + q * kXBytes / 2);
    };
    auto settle = [&](Piece& at) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            hold(at.w[nt]);
            hold(at.s[nt]);
            if constexpr (kOffsets) hold(at.o[nt]);
        }
#pragma unroll
        for (int rs = 0; rs < RS; ++rs)
#pragma unroll
            for (int q = 0; q < kXReads; ++q) hold(at.x[rs][q]);
    };

    mfma_cd acc[NT][RS];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int rs = 0; rs < RS; ++rs) acc[nt][rs] = mfma_zero();
    const Words<2> ones{kOnes<T>, kOnes<T>};
    auto product = [](Words<2> a, Words<2> b, mfma_cd c) {
        if constexpr (SHAPE == 4) {
            return mfma4x4<T>(a, b, c);
        } else {
            return mfma16x16<T>(a, b, c);
        }
    };

    auto multiply = [&](Piece& at, int r) {
        const bool live = r < mine && first(r) < k;
        u32 x[RS][LANE / 2];
#pragma unroll
        for (int rs = 0; rs < RS; ++rs)
#pragma unroll
            for (int i = 0; i < LANE / 2; ++i)
                x[rs][i] = live ? at.x[rs][i / (kXBytes / 4)][i % (kXBytes / 4)] : 0u;
        mfma_cd d[NT][RS], sx[RS];
#pragma unroll
        for (int rs = 0; rs < RS; ++rs) {
            sx[rs] = mfma_zero();
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) d[nt][rs] = mfma_zero();
        }
#pragma unroll
        for (int w = 0; w < kWords; ++w) {
            u32 p[NT][U::kPairs];
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) U::pairs(word(at.w[nt], w), p[nt]);
#pragma unroll
            for (int i = 0; i < U::kPairs; i += 2) {
                const int q = (w * U::kPairs + i) / 2;  // the 4-code step
#pragma unroll
                for (int rs = 0; rs < RS; ++rs) {
                    const Words<2> a{x[rs][2 * q], x[rs][2 * q + 1]};
                    sx[rs] = product(a, ones, sx[rs]);
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) d[nt][rs] = product(a, Words<2>{p[nt][i], p[nt][i + 1]}, d[nt][rs]);
                }
            }
        }
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const Factor f = factor_bits<F, kBits, kOffset>(at.s[nt], at.o[nt]);
            const float o = additive<kOffset>(f) - U::kMagic * f.s;
#pragma unroll
            for (int rs = 0; rs < RS; ++rs)
#pragma unroll
                for (int t = 0; t < 4; ++t) acc[nt][rs][t] = fmaf(f.s, d[nt][rs][t], fmaf(o, sx[rs][t], acc[nt][rs][t]));
        }
    };

    for (int r = 0; r < mine; r += R) {
        Piece at[R];
#pragma unroll
        for (int i = 0; i < R; ++i) issue(at[i], r + i);
#pragma unroll
        for (int i = 0; i < R; ++i) {
            // Piece i lands with the R − 1 − i behind it still in flight.
            if (i == 0) wait_reads<(R - 1) * kReads>();
            if (i == 1) wait_reads<(R > 2 ? R - 2 : 0) * kReads>();
            if (i == 2) wait_reads<(R > 3 ? R - 3 : 0) * kReads>();
            if (i == 3) wait_reads<0>();
            settle(at[i]);
            multiply(at[i], r + i);
        }
    }

    // A lane's D[t] is row `down(t)` and column `across` of its tile pair.
    // The 4 x 4 x 4 core's sixteen blocks each hold their own k: sum the
    // four of a 16-lane row (lanes j, j + 4, j + 8, j + 12, which a rotate
    // by 4 and by 8 pairs) here, the rows below with the wavefronts.
    auto down = [&](int t) { return SHAPE == 4 ? t : 4 * slot + t; };
    if constexpr (SHAPE == 4) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int rs = 0; rs < RS; ++rs)
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    float v = acc[nt][rs][t];
                    v += __builtin_bit_cast(  // row_ror:4
                        float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), 0x124, 0xf, 0xf, false));
                    v += __builtin_bit_cast(  // row_ror:8
                        float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), 0x128, 0xf, 0xf, false));
                    acc[nt][rs][t] = v;
                }
    }
    constexpr int kCells = NT * RS * SHAPE * SHAPE;           // the block's outputs
    constexpr int kParts = SHAPE == 4 ? 4 * SPLIT : SPLIT;  // partial sums a cell
    __shared__ float part[kParts][kCells];
    if (SHAPE == 16 || l % 16 < 4) {
        const int at = SHAPE == 4 ? 4 * wave + l / 16 : wave;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int rs = 0; rs < RS; ++rs)
#pragma unroll
                for (int t = 0; t < 4; ++t)
                    part[at][((nt * RS + rs) * SHAPE + down(t)) * SHAPE + across] = acc[nt][rs][t];
    }
    __syncthreads();
    for (int cell = threadIdx.x; cell < kCells; cell += blockDim.x) {
        float v = 0.f;
#pragma unroll
        for (int i = 0; i < kParts; ++i) v += part[i][cell];
        const int c = cell % SHAPE, rr = cell / SHAPE % SHAPE, rs = cell / (SHAPE * SHAPE) % RS,
                  nt = cell / (SHAPE * SHAPE * RS);
        const int row = r0 + SHAPE * rs + rr, col = col0 + SHAPE * nt + c;
        if (row < rows && col < n) out[static_cast<long long>(row) * n + col] = from_f32<T>(v);
    }
}

// The prefill arm: a 64-code stage of 128 weight rows, two threads a row,
// each decoding 32 codes into the tile.
template <class T, class F, int kBits, int kOffset, int kGroup>
struct AffineRows {
    static constexpr int kCodes = 32;  // codes a thread a stage
    static constexpr int kPerWord = 32 / kBits;
    static constexpr int kWords = kCodes / kPerWord;
    static constexpr bool kOneGroup = kGroup % kCodes == 0;
    static constexpr int kFactors = kOneGroup ? 1 : kWords;

    AffinePlanes planes;
    int row, half, wrow, k, groups;
    Words<kWords < 4 ? kWords : 4> words[kWords < 4 ? 1 : kWords / 4];
    Factor f[kFactors];

    __device__ __forceinline__ AffineRows(AffinePlanes p, int n, int k_) : planes(p), k(k_) {
        row = threadIdx.x / 2;
        half = threadIdx.x % 2;
        wrow = min(static_cast<int>(blockIdx.y) * kGemmBN + row, n - 1);
        groups = k / kGroup;
    }

    __device__ __forceinline__ void load(int c) {
        const int k0 = c * kGemmKC + half * kCodes;
        const u8* at = planes.codes + static_cast<long long>(wrow) * k * kBits / 8 + k0 * kBits / 8;
        constexpr int W = kWords < 4 ? kWords : 4;
#pragma unroll
        for (int i = 0; i < (kWords < 4 ? 1 : kWords / 4); ++i)
            words[i] = *reinterpret_cast<const Words<W>*>(at + 16 * i);
#pragma unroll
        for (int i = 0; i < kFactors; ++i)
            f[i] = factor_at<F, kBits, kOffset>(planes.scales, planes.biases,
                                                static_cast<long long>(wrow) * groups + (k0 + i * kPerWord) / kGroup);
    }

    __device__ __forceinline__ void store(T* ws) {
        u32 out[kCodes / 2];
#pragma unroll
        for (int w = 0; w < kWords; ++w) {
            float c[kPerWord];
            codes_f32<kBits>(words[w / 4][w % 4], c);
            const Factor fw = f[kOneGroup ? 0 : w];
#pragma unroll
            for (int i = 0; i < kPerWord; i += 2)
                out[(w * kPerWord + i) / 2] = pack_pair<T>(dequant<kOffset>(c[i], fw), dequant<kOffset>(c[i + 1], fw));
        }
#pragma unroll
        for (int i = 0; i < kCodes / 8; ++i)
            *reinterpret_cast<Words<4>*>(ws + row * kGemmLd + half * kCodes + 8 * i) =
                Words<4>{out[4 * i], out[4 * i + 1], out[4 * i + 2], out[4 * i + 3]};
    }
};

// y = act · Wᵀ over an affine-quantized W, BM activation rows a block.
// K is a whole number of 64-code stages.
template <class T, class F, int kBits, int kOffset, int kGroup, int BM>
__global__ __launch_bounds__(kGemmThreads) void matmul_affine_gemm(
    const T* __restrict__ act, const u8* __restrict__ codes, const u8* __restrict__ scales,
    const u8* __restrict__ biases, T* __restrict__ out, float* __restrict__ part, int per, int m, int n, int k,
    const AffineSeat* __restrict__ seat, const u32* __restrict__ win) {
    const int rows = win == nullptr ? m : min(m, static_cast<int>(win[0]));
    if (static_cast<int>(blockIdx.x) * BM >= rows) return;
    AffineRows<T, F, kBits, kOffset, kGroup> stage(seated(codes, scales, biases, seat), n, k);
    affine_gemm<T, BM>(act, out, part, per, rows, n, k, stage);
}

// The plane decoded to bf16, one thread a 32-bit code word, as the dense
// GEMM then reads it.
template <class F, int kBits, int kOffset, int kGroup>
__global__ void dequant_affine(const u8* __restrict__ codes, const u8* __restrict__ scales,
                               const u8* __restrict__ biases, bf16* __restrict__ out, int n, int k) {
    constexpr int kPerWord = 32 / kBits;
    const int words_per_row = k / kPerWord;
    const long long at = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= static_cast<long long>(n) * words_per_row) return;
    const int row = static_cast<int>(at / words_per_row);
    const int word = static_cast<int>(at % words_per_row);
    const Factor f = factor_at<F, kBits, kOffset>(scales, biases,
                                                  static_cast<long long>(row) * (k / kGroup) +
                                                      word * kPerWord / kGroup);
    float c[kPerWord];
    codes_f32<kBits>(reinterpret_cast<const u32*>(codes)[at], c);
    u32* dst = reinterpret_cast<u32*>(out + static_cast<long long>(row) * k + word * kPerWord);
#pragma unroll
    for (int i = 0; i < kPerWord; i += 2) dst[i / 2] = pack_pair<bf16>(dequant<kOffset>(c[i], f), dequant<kOffset>(c[i + 1], f));
}

// --------------------------------------------------------------- casts --

template <class T>
__global__ void cast_f32_to(const float* __restrict__ src, T* __restrict__ dst, usize n) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = from_f32<T>(src[i]);
}

// buf[row] *= l, elementwise: one block a row.
template <class T>
__global__ void scale_rows(T* __restrict__ buf, const T* __restrict__ l, int width) {
    T* row = buf + static_cast<usize>(blockIdx.x) * width;
    for (int c = threadIdx.x; c < width; c += blockDim.x) row[c] = from_f32<T>(to_f32(row[c]) * to_f32(l[c]));
}

}  // namespace pie::linear

// ================================================================ MoE ====
//
// Packed expert banks (mxfp4, MLX affine) for the routed matmul. The decode
// legs stream (linear/moe_kit.hip); the grouped legs take routes ordered by
// expert from `moe_route_order`. A seat's `bases`, when given, names the
// planes in place of the resident ones; `group_hits` counts the routes read.

#include "linear/moe_kit.hip"

namespace pie::linear {

// Routes in expert order: order[offsets[e] .. offsets[e+1]) are expert e's
// routes (plane route indices), offsets[num_experts] starts the routes no
// expert takes, and `work` lists one item (e << 16 | slice) for each
// `group_routes` routes of an expert, at most `work_cap`, their count in
// offsets[num_experts + 1]. One block of 256 threads; dynamic LDS of
// 2·(num_experts + 1) ints. A thread holds its first kOrderHeld routes in
// registers across the two passes over them, and owns a run of the buckets
// for the prefix, so the block meets at four barriers.
constexpr int kOrderThreads = 1024;
constexpr int kOrderHeld = 2;

__global__ __launch_bounds__(kOrderThreads) void moe_route_order(const i32* __restrict__ routes,
                                                                 i32* __restrict__ order, i32* __restrict__ offsets,
                                                                 i32* __restrict__ work, int work_cap,
                                                                 int group_routes, int top_k, int route_count,
                                                                 int num_experts, const u32* __restrict__ win) {
    extern __shared__ i32 order_smem[];
    __shared__ i32 wave_totals[2][kOrderThreads / kWave];
    i32* counts = order_smem;
    i32* fill = counts + num_experts + 1;
    const int tid = threadIdx.x, l = lane(), wave = tid / kWave;
    const int live = win != nullptr ? min(static_cast<int>(win[0]) * top_k, route_count) : route_count;
    const int base = win != nullptr ? static_cast<int>(win[1]) * top_k : 0;
    auto bucket = [&](int r) {
        const int e = routes[base + r];
        return e < 0 || e >= num_experts ? num_experts : e;
    };
    int held[kOrderHeld];
#pragma unroll
    for (int i = 0; i < kOrderHeld; ++i) {
        const int r = tid + i * kOrderThreads;
        held[i] = r < live ? bucket(r) : -1;
    }
    for (int e = tid; e <= num_experts; e += kOrderThreads) {
        counts[e] = 0;
        fill[e] = 0;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kOrderHeld; ++i)
        if (held[i] >= 0) atomicAdd(counts + held[i], 1);
    for (int r = tid + kOrderHeld * kOrderThreads; r < live; r += kOrderThreads) atomicAdd(counts + bucket(r), 1);
    __syncthreads();

    // Thread t's buckets [t·per, +per): their routes and items, prefixed
    // over the block (a wave scan, then the waves' totals).
    const int per = (num_experts + kOrderThreads) / kOrderThreads;
    const int e0 = min(tid * per, num_experts + 1), e1 = min(e0 + per, num_experts + 1);
    int mine = 0, items = 0;
    for (int e = e0; e < e1; ++e) {
        const int c = counts[e];
        mine += c;
        if (e < num_experts) items += (c + group_routes - 1) / group_routes;
    }
    int run = mine, first = items;
#pragma unroll
    for (int off = 1; off < kWave; off <<= 1) {
        const int a = __shfl_up(run, off, kWave), b = __shfl_up(first, off, kWave);
        if (l >= off) {
            run += a;
            first += b;
        }
    }
    if (l == kWave - 1) {
        wave_totals[0][wave] = run;
        wave_totals[1][wave] = first;
    }
    __syncthreads();
    run -= mine;
    first -= items;
    int total = 0;
#pragma unroll
    for (int w = 0; w < kOrderThreads / kWave; ++w) {
        if (w < wave) {
            run += wave_totals[0][w];
            first += wave_totals[1][w];
        }
        total += wave_totals[1][w];
    }
    for (int e = e0; e < e1; ++e) {
        const int c = counts[e];
        counts[e] = run;
        offsets[e] = run;
        const int slices = e < num_experts ? (c + group_routes - 1) / group_routes : 0;
        for (int s = 0; s < slices && first + s < work_cap; ++s) work[first + s] = e * 65536 + s;
        run += c;
        first += slices;
    }
    if (tid == 0) offsets[num_experts + 1] = min(total, work_cap);
    __syncthreads();

#pragma unroll
    for (int i = 0; i < kOrderHeld; ++i)
        if (held[i] >= 0) order[counts[held[i]] + atomicAdd(fill + held[i], 1)] = base + tid + i * kOrderThreads;
    for (int r = tid + kOrderHeld * kOrderThreads; r < live; r += kOrderThreads) {
        const int e = bucket(r);
        order[counts[e] + atomicAdd(fill + e, 1)] = base + r;
    }
}

// The decode legs: y[route] = W[expert(route)]·act[route / act_div], one
// route a block row (grid (routes, ⌈n / (4·R)⌉), 256 threads). A dead route
// (expert < 0) lands zeros.
template <class C, class T, class F>
__device__ __forceinline__ void select_stream(const T* __restrict__ act, const i32* __restrict__ routes,
                                              T* __restrict__ out, int top_k, int act_div, int n, int k,
                                              unsigned int* __restrict__ group_hits, int rows_per_wave,
                                              const u32* __restrict__ win, F&& planes) {
    extern __shared__ __align__(16) unsigned char stream_smem[];
    const int route = blockIdx.x;
    if (win != nullptr && route >= static_cast<int>(win[0]) * top_k) return;
    const long long plane_route = win != nullptr ? route + static_cast<long long>(win[1]) * top_k : route;
    const int expert = routes[plane_route];
    T* y = out + plane_route * n;
    if (expert < 0) {
        stream_zero(y, n, rows_per_wave);
        return;
    }
    if (group_hits != nullptr && blockIdx.y == 0 && threadIdx.x == 0) atomicAdd(group_hits, 1u);
    const T* bias = nullptr;
    const typename C::Planes p = planes(expert, bias);
    stream_rows<C, T>(p, act + plane_route / act_div * k, y, bias, n, k, rows_per_wave, stream_smem);
}

template <class T>
__global__ __launch_bounds__(256) void moe_matmul_select_mxfp4(
    const T* __restrict__ act, const i32* __restrict__ routes, const u8* __restrict__ codes,
    const u8* __restrict__ scales, const T* __restrict__ bias, T* __restrict__ out, int top_k, int act_div, int n,
    int k, const MoeGroupBases* __restrict__ bases, unsigned int* __restrict__ group_hits, int rows_per_wave,
    const u32* __restrict__ win) {
    select_stream<StreamMxfp4<T>, T>(
        act, routes, out, top_k, act_div, n, k, group_hits, rows_per_wave, win, [&](int e, const T*& b) {
            const u8* c = bases != nullptr ? bases->codes : codes;
            const u8* s = bases != nullptr ? bases->scales : scales;
            const long long rows = static_cast<long long>(e) * n;
            if (bias != nullptr) b = bias + rows;
            return typename StreamMxfp4<T>::Planes{c + rows * (k / 2), s + rows * (k / 32)};
        });
}

template <class T, int BITS, int GROUP>
__global__ __launch_bounds__(256) void moe_matmul_select_mlxu4(
    const T* __restrict__ act, const i32* __restrict__ routes, const u8* __restrict__ codes,
    const u8* __restrict__ scales, const u8* __restrict__ biases, T* __restrict__ out, int top_k, int act_div, int n,
    int k, const MoeGroupBases* __restrict__ bases, unsigned int* __restrict__ group_hits, int rows_per_wave,
    const u32* __restrict__ win) {
    using C = StreamAffine<T, BITS, GROUP>;
    select_stream<C, T>(act, routes, out, top_k, act_div, n, k, group_hits, rows_per_wave, win,
                        [&](int e, const T*&) {
                            const u8* c = bases != nullptr ? bases->codes : codes;
                            const u8* s = bases != nullptr ? bases->scales : scales;
                            const u8* b = bases != nullptr ? bases->biases : biases;
                            const long long rows = static_cast<long long>(e) * n;
                            const long long factors = rows * (k / GROUP);
                            return typename C::Planes{reinterpret_cast<const u32*>(c) + rows * (k * BITS / 32),
                                                      reinterpret_cast<const bf16*>(s) + factors,
                                                      reinterpret_cast<const bf16*>(b) + factors};
                        });
}

// The grouped legs on the matrix core (bf16): one work item of up to MT
// routes a block, 32 weight rows a wave (grid (work cap, ⌈n / (32·waves)⌉),
// kGroupWaves<MT> waves).
template <int MT>
__global__ __launch_bounds__(kGroupWaves<MT>* kWave) void moe_matmul_select_mxfp4_wmma(
    const bf16* __restrict__ act, const i32* __restrict__ order, const i32* __restrict__ offsets,
    const i32* __restrict__ work, const u8* __restrict__ codes, const u8* __restrict__ scales,
    const bf16* __restrict__ bias, bf16* __restrict__ out, int act_div, int n, int k, int num_experts,
    const MoeGroupBases* __restrict__ bases, unsigned int* __restrict__ group_hits) {
    GroupItem item;
    if (!group_item(offsets, work, num_experts, MT, item)) return;
    if (group_hits != nullptr && blockIdx.y == 0 && threadIdx.x == 0)
        atomicAdd(group_hits, static_cast<unsigned>(item.count));
    const u8* c = bases != nullptr ? bases->codes : codes;
    const u8* s = bases != nullptr ? bases->scales : scales;
    const long long rows = static_cast<long long>(item.expert) * n;
    grouped_tile<GroupMxfp4, bf16, MT>(act, order, item.begin, item.count, {c + rows * (k / 2), s + rows * (k / 32)},
                                   bias != nullptr ? bias + rows : nullptr, out, act_div, n, k);
}

template <int BITS, int GROUP, int MT>
__global__ __launch_bounds__(kGroupWaves<MT>* kWave) void moe_matmul_select_mlxu4_wmma(
    const bf16* __restrict__ act, const i32* __restrict__ order, const i32* __restrict__ offsets,
    const i32* __restrict__ work, const u8* __restrict__ codes, const u8* __restrict__ scales,
    const u8* __restrict__ biases, bf16* __restrict__ out, int act_div, int n, int k, int num_experts,
    const MoeGroupBases* __restrict__ bases, unsigned int* __restrict__ group_hits) {
    GroupItem item;
    if (!group_item(offsets, work, num_experts, MT, item)) return;
    if (group_hits != nullptr && blockIdx.y == 0 && threadIdx.x == 0)
        atomicAdd(group_hits, static_cast<unsigned>(item.count));
    const u8* c = bases != nullptr ? bases->codes : codes;
    const u8* s = bases != nullptr ? bases->scales : scales;
    const u8* b = bases != nullptr ? bases->biases : biases;
    const long long rows = static_cast<long long>(item.expert) * n;
    const long long factors = rows * (k / GROUP);
    grouped_tile<GroupAffine<BITS, GROUP>, bf16, MT>(
        act, order, item.begin, item.count,
        {reinterpret_cast<const u32*>(c) + rows * (k * BITS / 32), reinterpret_cast<const bf16*>(s) + factors,
         reinterpret_cast<const bf16*>(b) + factors},
        nullptr, out, act_div, n, k);
}

// The batch legs (decode batches, bf16): one work item of up to MT routes a
// block, 64·NBW weight rows (grid (work cap, ⌈n / (64·NBW)⌉), 256 threads).
// A stage is 256 codes a row short of 32 routes, 128 at 32, so the
// activation's two LDS buffers stay near 17 KB.
constexpr int kBatchDepth = 3;

template <int MT, int NBW>
__global__ __launch_bounds__(256) void moe_matmul_select_mxfp4_batch(
    const bf16* __restrict__ act, const i32* __restrict__ order, const i32* __restrict__ offsets,
    const i32* __restrict__ work, const u8* __restrict__ codes, const u8* __restrict__ scales,
    const bf16* __restrict__ bias, bf16* __restrict__ out, int act_div, int n, int k, int num_experts,
    const MoeGroupBases* __restrict__ bases, unsigned int* __restrict__ group_hits) {
    GroupItem item;
    if (!group_item(offsets, work, num_experts, MT, item)) return;
    if (group_hits != nullptr && blockIdx.y == 0 && threadIdx.x == 0)
        atomicAdd(group_hits, static_cast<unsigned>(item.count));
    const u8* c = bases != nullptr ? bases->codes : codes;
    const u8* s = bases != nullptr ? bases->scales : scales;
    const long long rows = static_cast<long long>(item.expert) * n;
    batch_tile<BatchMxfp4<MT == 32 ? 64 : 128>, bf16, MT, NBW, kBatchDepth>(
        act, order, item.begin, item.count, {c + rows * (k / 2), s + rows * (k / 32)},
        bias != nullptr ? bias + rows : nullptr, out, act_div, n, k);
}

template <int BITS, int GROUP, int MT, int NBW>
__global__ __launch_bounds__(256) void moe_matmul_select_mlxu4_batch(
    const bf16* __restrict__ act, const i32* __restrict__ order, const i32* __restrict__ offsets,
    const i32* __restrict__ work, const u8* __restrict__ codes, const u8* __restrict__ scales,
    const u8* __restrict__ biases, bf16* __restrict__ out, int act_div, int n, int k, int num_experts,
    const MoeGroupBases* __restrict__ bases, unsigned int* __restrict__ group_hits) {
    static_assert(BITS == 4, "the batch leg reads four-bit codes");
    constexpr int ROW_BYTES = MT < 32 || GROUP == 128 ? 128 : 64;
    GroupItem item;
    if (!group_item(offsets, work, num_experts, MT, item)) return;
    if (group_hits != nullptr && blockIdx.y == 0 && threadIdx.x == 0)
        atomicAdd(group_hits, static_cast<unsigned>(item.count));
    const u8* c = bases != nullptr ? bases->codes : codes;
    const u8* s = bases != nullptr ? bases->scales : scales;
    const u8* b = bases != nullptr ? bases->biases : biases;
    const long long rows = static_cast<long long>(item.expert) * n;
    const long long factors = rows * (k / GROUP);
    batch_tile<BatchAffine<GROUP, ROW_BYTES>, bf16, MT, NBW, kBatchDepth>(
        act, order, item.begin, item.count,
        {reinterpret_cast<const u32*>(c) + rows * (k / 8), reinterpret_cast<const bf16*>(s) + factors,
         reinterpret_cast<const bf16*>(b) + factors},
        nullptr, out, act_div, n, k);
}

// The grouped legs off the matrix core, any T, f32 throughout: one expert a
// block (grid (experts, ⌈n / 128⌉), 256 threads), its routes 16 at a time
// against a 128 x 64 tile of the dequantized weight in LDS. Each thread
// owns two weight rows and four of the batch's routes.
constexpr int kSimtTileN = 128;
constexpr int kSimtTileK = 64;
constexpr int kSimtBatch = 16;

__constant__ float kFp4Values[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                                     -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

// `dequant(row, kk, w)` writes codes kk .. kk + 7 of weight row `row`.
template <class T, class D>
__device__ __forceinline__ void grouped_simt(const T* __restrict__ act, const i32* __restrict__ order, int begin,
                                             int end, const T* __restrict__ bias, T* __restrict__ out, int act_div,
                                             int n, int k, D&& dequant) {
    __shared__ float w_tile[kSimtTileN][kSimtTileK + 1];
    __shared__ float x_tile[kSimtBatch][kSimtTileK];
    const int row0 = blockIdx.y * kSimtTileN;
    const int tid = threadIdx.x;
    const int pair = tid % (kSimtTileN / 2), slot = tid / (kSimtTileN / 2);
    const int ra = 2 * pair, rb = ra + 1;
    const float bias_a = bias != nullptr && row0 + ra < n ? to_f32(bias[row0 + ra]) : 0.f;
    const float bias_b = bias != nullptr && row0 + rb < n ? to_f32(bias[row0 + rb]) : 0.f;
    for (int b0 = begin; b0 < end; b0 += kSimtBatch) {
        const int batch = min(kSimtBatch, end - b0);
        float acc[2][kSimtBatch / 4] = {};
        for (int k0 = 0; k0 < k; k0 += kSimtTileK) {
            for (int idx = tid; idx < kSimtTileN * kSimtTileK / 8; idx += blockDim.x) {
                const int r = idx / (kSimtTileK / 8), kk = (idx % (kSimtTileK / 8)) * 8;
                float w[8] = {};
                if (k0 + kk < k) dequant(min(row0 + r, n - 1), k0 + kk, w);
#pragma unroll
                for (int j = 0; j < 8; ++j) w_tile[r][kk + j] = w[j];
            }
            for (int idx = tid; idx < batch * kSimtTileK; idx += blockDim.x) {
                const int t = idx / kSimtTileK, kk = idx % kSimtTileK;
                const T* x = act + static_cast<long long>(order[b0 + t] / act_div) * k;
                x_tile[t][kk] = k0 + kk < k ? to_f32(x[k0 + kk]) : 0.f;
            }
            __syncthreads();
            for (int kk = 0; kk < kSimtTileK; ++kk) {
                const float wa = w_tile[ra][kk], wb = w_tile[rb][kk];
#pragma unroll
                for (int j = 0; j < kSimtBatch / 4; ++j) {
                    const float xv = x_tile[slot + 4 * j][kk];
                    acc[0][j] = fmaf(wa, xv, acc[0][j]);
                    acc[1][j] = fmaf(wb, xv, acc[1][j]);
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int j = 0; j < kSimtBatch / 4; ++j) {
            const int t = slot + 4 * j;
            if (t >= batch) continue;
            T* o = out + static_cast<long long>(order[b0 + t]) * n + row0;
            if (row0 + ra < n) o[ra] = from_f32<T>(acc[0][j] + bias_a);
            if (row0 + rb < n) o[rb] = from_f32<T>(acc[1][j] + bias_b);
        }
    }
}

template <class T>
__global__ __launch_bounds__(256) void moe_matmul_select_mxfp4_grouped(
    const T* __restrict__ act, const i32* __restrict__ order, const i32* __restrict__ offsets,
    const u8* __restrict__ codes, const u8* __restrict__ scales, const T* __restrict__ bias, T* __restrict__ out,
    int act_div, int n, int k, int num_experts, const MoeGroupBases* __restrict__ bases,
    unsigned int* __restrict__ group_hits) {
    const int expert = blockIdx.x;
    if (expert >= num_experts || static_cast<int>(blockIdx.y) * kSimtTileN >= n) return;
    const int begin = offsets[expert], end = offsets[expert + 1];
    if (begin >= end) return;
    if (group_hits != nullptr && blockIdx.y == 0 && threadIdx.x == 0)
        atomicAdd(group_hits, static_cast<unsigned>(end - begin));
    const long long rows = static_cast<long long>(expert) * n;
    const u32* w32 = reinterpret_cast<const u32*>((bases != nullptr ? bases->codes : codes) + rows * (k / 2));
    const u8* s8 = (bases != nullptr ? bases->scales : scales) + rows * (k / 32);
    grouped_simt<T>(act, order, begin, end, bias != nullptr ? bias + rows : nullptr, out, act_div, n, k,
                    [&](int row, int kk, float* w) {
                        const float sc = mxfp4_scale(s8[static_cast<long long>(row) * (k / 32) + kk / 32]);
                        const u32 word = w32[static_cast<long long>(row) * (k / 8) + kk / 8];
#pragma unroll
                        for (int j = 0; j < 8; ++j) w[j] = kFp4Values[(word >> (4 * j)) & 15u] * sc;
                    });
}

template <class T, int BITS, int GROUP>
__global__ __launch_bounds__(256) void moe_matmul_select_mlxu4_grouped(
    const T* __restrict__ act, const i32* __restrict__ order, const i32* __restrict__ offsets,
    const u8* __restrict__ codes, const u8* __restrict__ scales, const u8* __restrict__ biases, T* __restrict__ out,
    int act_div, int n, int k, int num_experts, const MoeGroupBases* __restrict__ bases,
    unsigned int* __restrict__ group_hits) {
    const int expert = blockIdx.x;
    if (expert >= num_experts || static_cast<int>(blockIdx.y) * kSimtTileN >= n) return;
    const int begin = offsets[expert], end = offsets[expert + 1];
    if (begin >= end) return;
    if (group_hits != nullptr && blockIdx.y == 0 && threadIdx.x == 0)
        atomicAdd(group_hits, static_cast<unsigned>(end - begin));
    const long long rows = static_cast<long long>(expert) * n;
    const long long factors = rows * (k / GROUP);
    const u32* w32 = reinterpret_cast<const u32*>(bases != nullptr ? bases->codes : codes) + rows * (k * BITS / 32);
    const bf16* s16 = reinterpret_cast<const bf16*>(bases != nullptr ? bases->scales : scales) + factors;
    const bf16* b16 = reinterpret_cast<const bf16*>(bases != nullptr ? bases->biases : biases) + factors;
    grouped_simt<T>(act, order, begin, end, nullptr, out, act_div, n, k, [&](int row, int kk, float* w) {
        const long long f = static_cast<long long>(row) * (k / GROUP) + kk / GROUP;
        const float sv = to_f32(s16[f]), bv = to_f32(b16[f]);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const long long bit = (static_cast<long long>(row) * k + kk + j) * BITS;
            const u32 code = (w32[bit / 32] >> (bit % 32)) & ((1u << BITS) - 1u);
            w[j] = fmaf(static_cast<float>(code), sv, bv);
        }
    });
}

}  // namespace pie::linear
