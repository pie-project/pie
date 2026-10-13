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

// y[m, n] = act[m] · W[n] for the rows of a decode (up to eight),
// streaming the codes once. The products run on the 4 x 4 x 4 core:
// sixteen blocks of four lanes, lane 4b + j of block b holding weight row j
// and activation row j of a tile over the same LANE codes, at
// k = 16·LANE·p + LANE·b of piece p. So one read is four weight rows of
// 16·LANE codes each, and no lane exchanges codes.
//
// A block of SPLIT wavefronts is persistent over tiles of 4·NT weight
// rows (blockIdx.x, then every gridDim.x-th). Wavefront w owns pieces
// w + SPLIT·i, i < P, of every tile, so the block reads whole rows at once,
// and holds their activation (RS tiles of four rows, E of the four live)
// for the kernel's life, as matrix-core operands in `Quads` order with Σ x
// of each, so a tile's only reads are its codes and factors, D tiles in
// flight. The codes go to the core raw, so a product sums Σ x·(c + magic),
// and each piece folds into f32 as
//
//   Σ x·w = s·Σ x·(c + magic) + (o − magic·s)·Σ x
//
// (LANE codes are one group's). A piece's factors for the 4·NT rows are
// one 4-byte read a lane of each plane, handed to the lanes that use them
// by `ds_bpermute`. The wavefronts' sums meet in LDS. K is whole pieces.
template <class T, class F, int kBits, int kOffset, int kGroup, int LANE, int NT, int RS, int E, int P, int SPLIT,
          int D>
__global__ __launch_bounds__(kWave* SPLIT) void matmul_affine_stream(
    const T* __restrict__ act, const u8* __restrict__ codes, const u8* __restrict__ scales,
    const u8* __restrict__ biases, T* __restrict__ out, float* __restrict__ slices, int m, int n, int k,
    const AffineSeat* __restrict__ seat, const u32* __restrict__ win) {
    using Q = Quads<T, kBits>;
    constexpr int kBytes = LANE * kBits / 8;  // code bytes a lane reads a piece
    constexpr int kWords = kBytes / 4;
    constexpr int kPiece = 16 * LANE;
    constexpr int kPieceGroups = kPiece / kGroup;
    constexpr bool kOffsets = kOffset != kOffPreConst;
    constexpr int kOffBytes = kOffset == kOffPreInt ? 1 : 2;
    constexpr int kRows = 4 * NT;                          // weight rows a tile
    constexpr int kScaleWords = kPieceGroups * 2 / 4;       // a row's factor words a piece
    constexpr int kOffWords = kPieceGroups * kOffBytes / 4;
    static_assert(kWave == 64, "sixteen blocks of four lanes");
    static_assert(kBytes == 4 || kBytes == 8 || kBytes == 16, "a lane reads one 4-, 8- or 16-byte access");
    static_assert(kGroup % LANE == 0 && kPiece % kGroup == 0 && LANE >= 8,
                  "a lane's codes are whole words of one group");
    static_assert(kScaleWords >= 1 && kRows * kScaleWords <= kWave, "a piece's scales are a word a lane");
    static_assert(!kOffsets || (kOffWords >= 1 && kRows * kOffWords <= kWave), "a piece's offsets are a word a lane");

    const int rows = win == nullptr ? m : min(m, static_cast<int>(win[0]));
    if (rows <= 0) return;
    const AffinePlanes planes = seated(codes, scales, biases, seat);
    const int l = lane(), j = l % 4, b = l / 4;
    const int wave = __builtin_amdgcn_readfirstlane(static_cast<int>(threadIdx.x) / kWave);
    const int groups = k / kGroup;
    const int tiles = (n + kRows - 1) / kRows;
    const long long row_bytes = static_cast<long long>(k) * kBits / 8;
    // K past what a block's wavefronts hold is sliced over blockIdx.y, a
    // slice of SPLIT·P pieces, each slice's f32 sums landing in `slices`
    // (slice-major, [rows][n] each) for `affine_gemm_finalize` to add up.
    const int base = blockIdx.y * SPLIT * P, pieces = min(k / kPiece, base + SPLIT * P);

    // This wavefront's pieces: piece i is base + wave + SPLIT·i, live below
    // `pieces` (a dead one reads the last live piece and multiplies nothing).
    auto piece = [&](int i) { return min(base + wave + SPLIT * i, pieces - 1); };
    auto alive = [&](int i) { return base + wave + SPLIT * i < pieces; };

    // The activation, once: operand h of word w of piece i, tile rs, and Σ x.
    // Its reads go out behind the first tile's, so the codes are in flight
    // first.
    Words<4> xr[P][RS][LANE / 8];
    auto issue_act = [&]() {
#pragma unroll
        for (int i = 0; i < P; ++i)
#pragma unroll
            for (int rs = 0; rs < RS; ++rs) {
                const u8* x = reinterpret_cast<const u8*>(act + piece(i) * kPiece);
                const u32 off = 2 * (static_cast<u32>(min(4 * rs + j, rows - 1)) * k + LANE * b);
#pragma unroll
                for (int q = 0; q < LANE / 8; ++q) xr[i][rs][q] = read_global<Words<4>>(x + off + 16 * q);
            }
    };
    Words<2> xa[P][RS][kWords][Q::kSteps];
    mfma_cd sx[P][RS];
    auto prime = [&]() {
        const Words<2> ones{kOnes<T>, kOnes<T>};
#pragma unroll
        for (int i = 0; i < P; ++i)
#pragma unroll
            for (int rs = 0; rs < RS; ++rs) {
                u32 d[LANE / 2];
#pragma unroll
                for (int q = 0; q < LANE / 8; ++q) {
#pragma unroll
                    for (int e = 0; e < 4; ++e) d[4 * q + e] = alive(i) ? xr[i][rs][q][e] : 0u;
                }
                sx[i][rs] = mfma_zero();
#pragma unroll
                for (int w = 0; w < kWords; ++w)
#pragma unroll
                    for (int h = 0; h < Q::kSteps; ++h) {
                        xa[i][rs][w][h] = Q::act(d + w * (Q::kPer / 2), h);
                        sx[i][rs] = mfma4x4<T>(xa[i][rs][w][h], ones, sx[i][rs]);
                    }
            }
    };
    // Factor reads: lane s < kRows·words reads word s / kRows of row
    // s % kRows; lane 4b + j takes its group's from the lane holding row
    // 4·nt + j, as a byte address for `ds_bpermute`, and a shift. Where both
    // planes fit half a wavefront, one read takes them, the upper half the
    // offsets.
    constexpr bool kPacked = kOffsets && kRows * kScaleWords <= kWave / 2 && kRows * kOffWords <= kWave / 2;
    const int my_group = LANE * b / kGroup;  // within a piece
    auto factor_lane = [&](int words) { return min(kPacked ? l % (kWave / 2) : l, kRows * words - 1); };
    auto factor_src = [&](int nt, int bytes, bool off) {
        return 4 * ((my_group * bytes / 4) * kRows + 4 * nt + j + (kPacked && off ? kWave / 2 : 0));
    };
    const int scale_shift = my_group * 2 % 4 * 8, off_shift = my_group * kOffBytes % 4 * 8;

    struct Tile {
        Words<kWords> w[P][NT];
        u32 s[P], o[P];
    };
    auto issue = [&](Tile& tl, int t) {
        const int row0 = t * kRows;
        u32 co[NT];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
            co[nt] = static_cast<u32>((min(row0 + 4 * nt + j, n - 1) - row0) * row_bytes) + b * kBytes;
        const int fs = factor_lane(kScaleWords), fo = factor_lane(kOffWords);
        const u32 so = static_cast<u32>((min(row0 + fs % kRows, n - 1) - row0) * groups * 2) + fs / kRows * 4;
        const u32 oo = static_cast<u32>((min(row0 + fo % kRows, n - 1) - row0) * groups * kOffBytes) + fo / kRows * 4;
#pragma unroll
        for (int i = 0; i < P; ++i) {
            const int pc = piece(i);
            const u8* cb = planes.codes + row0 * row_bytes + static_cast<long long>(pc) * kPiece * kBits / 8;
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) tl.w[i][nt] = read_global<Words<kWords>, true>(cb + co[nt]);
            const long long f0 = static_cast<long long>(row0) * groups + pc * kPieceGroups;
            if constexpr (kPacked) {
                tl.s[i] = read_global<u32>(l < kWave / 2 ? planes.scales + f0 * 2 + so : planes.biases + f0 * kOffBytes + oo);
                tl.o[i] = tl.s[i];
            } else {
                tl.s[i] = read_global<u32>(planes.scales + f0 * 2 + so);
                if constexpr (kOffsets) {
                    tl.o[i] = read_global<u32>(planes.biases + f0 * kOffBytes + oo);
                } else {
                    tl.o[i] = 0;
                }
            }
        }
    };

    // The wavefronts' partial sums, two tiles' worth so a tile's writes
    // need no barrier behind the previous tile's reads.
    constexpr int kCells = NT * RS * 16;
    __shared__ float part[2][kCells][SPLIT];

    auto land = [&](const Tile& tl, int t, int slot) {
        mfma_cd acc[NT][RS];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int rs = 0; rs < RS; ++rs) acc[nt][rs] = mfma_zero();
#pragma unroll
        for (int i = 0; i < P; ++i) {
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const u32 sw = __builtin_amdgcn_ds_bpermute(factor_src(nt, 2, false), tl.s[i]) >> scale_shift;
                u32 ow = 0;
                if constexpr (kOffsets)
                    ow = __builtin_amdgcn_ds_bpermute(factor_src(nt, kOffBytes, true), tl.o[i]) >> off_shift;
                const Factor f = factor_bits<F, kBits, kOffset>(sw & 0xffffu, ow & (kOffBytes == 1 ? 0xffu : 0xffffu));
                const float o = additive<kOffset>(f) - Q::kMagic * f.s;
                mfma_cd d[RS];
#pragma unroll
                for (int rs = 0; rs < RS; ++rs) d[rs] = mfma_zero();
#pragma unroll
                for (int w = 0; w < kWords; ++w) {
                    Words<2> c[Q::kSteps];
                    Q::codes(word(tl.w[i][nt], w), c);
#pragma unroll
                    for (int h = 0; h < Q::kSteps; ++h)
#pragma unroll
                        for (int rs = 0; rs < RS; ++rs) d[rs] = mfma4x4<T>(xa[i][rs][w][h], c[h], d[rs]);
                }
#pragma unroll
                for (int rs = 0; rs < RS; ++rs)
#pragma unroll
                    for (int e = 0; e < E; ++e)
                        acc[nt][rs][e] = fmaf(f.s, d[rs][e], fmaf(o, sx[i][rs][e], acc[nt][rs][e]));
            }
        }
        // Lane 4b + j's D[e] is activation row e, weight row j, summed over
        // its block's k: sum the sixteen blocks (a rotate by 4 and by 8
        // within a 16-lane row, then across the rows), and the wavefronts'
        // sums in LDS, in wavefront order.
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int rs = 0; rs < RS; ++rs)
#pragma unroll
                for (int e = 0; e < E; ++e) {
                    float v = acc[nt][rs][e];
                    v += __builtin_bit_cast(  // row_ror:4
                        float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), 0x124, 0xf, 0xf, true));
                    v += __builtin_bit_cast(  // row_ror:8
                        float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), 0x128, 0xf, 0xf, true));
                    v += __builtin_bit_cast(  // lane ^ 16
                        float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, v), 0x401f));
                    v += __shfl_xor(v, 32, kWave);
                    acc[nt][rs][e] = v;
                }
        if (l < 4) {
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int rs = 0; rs < RS; ++rs)
#pragma unroll
                    for (int e = 0; e < E; ++e) part[slot][((nt * RS + rs) * 4 + e) * 4 + j][wave] = acc[nt][rs][e];
        }
        lds_barrier();
        for (int cell = threadIdx.x; cell < kCells; cell += blockDim.x) {
            float v = 0.f;
#pragma unroll
            for (int q = 0; q < SPLIT; ++q) v += part[slot][cell][q];
            const int c = cell % 4, r = cell / 4 % 4, rs = cell / 16 % RS, nt = cell / (16 * RS);
            const int row = 4 * rs + r, col = t * kRows + 4 * nt + c;
            if (row >= rows || col >= n) continue;
            if (slices != nullptr) {
                slices[(static_cast<long long>(blockIdx.y) * rows + row) * n + col] = v;
            } else {
                out[static_cast<long long>(row) * n + col] = from_f32<T>(v);
            }
        }
    };

    // D tiles in flight: slot d lands tile i + d, then reads tile i + d + D.
    // The first tile's codes go out before the activation's reads, the
    // rest behind them. Every read in the steady loop is unconditional, so
    // the compiler's waits count the slots behind the one landing (a read
    // under a branch makes it wait for all of them); the scheduling
    // barriers keep each slot's reads where they are issued, as it would
    // sink them to their use. A block of fewer than D tiles lands them one
    // at a time.
    const int step = gridDim.x, first = blockIdx.x;
    if (first >= tiles) return;
    const int count = (tiles - first + step - 1) / step;
    static_assert(D >= 1 && D <= 4, "tiles in flight");
    auto tile = [&](int at) { return first + at * step; };
    Tile ring[D];
    if (count < D) {
        issue(ring[0], tile(0));
        issue_act();
        prime();
        for (int at = 0; at < count; ++at) {
            if (at > 0) issue(ring[0], tile(at));
            land(ring[0], tile(at), at & 1);
        }
        return;
    }
    issue(ring[0], tile(0));
    __builtin_amdgcn_sched_barrier(0);
    issue_act();
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int d = 1; d < D; ++d) issue(ring[d], tile(d));
    __builtin_amdgcn_sched_barrier(0);
    prime();
    // Steady: every slot reads its next tile.
    int i = 0;
    for (; i + 2 * D <= count; i += D) {
#pragma unroll
        for (int d = 0; d < D; ++d) {
            land(ring[d], tile(i + d), (i + d) & 1);
            __builtin_amdgcn_sched_barrier(0);
            issue(ring[d], tile(i + d + D));
            __builtin_amdgcn_sched_barrier(0);
        }
    }
    // The last D to 2D − 1 tiles: D in flight, the rest read as slots free.
#pragma unroll
    for (int d = 0; d < D; ++d) {
        land(ring[d], tile(i + d), (i + d) & 1);
        if (i + d + D < count) issue(ring[d], tile(i + d + D));
    }
#pragma unroll
    for (int d = 0; d < D; ++d)
        if (i + D + d < count) land(ring[d], tile(i + D + d), (i + D + d) & 1);
}

// y = act · Wᵀ over an affine-quantized W, from a few rows past a decode
// to a prefill: a GEMM whose weight tile stays packed until a lane takes it
// to the matrix core. A block of four wavefronts lands a BM x BN tile of
// y, WM x WN wavefronts of (16·RS) x (16·NT) each, walking K in stages of
// KS codes; a grid short of the device slices K over blockIdx.z, `per`
// stages a slice, each slice's f32 partial landing in `part` (slice-major,
// [rows][n] each) for `affine_gemm_finalize` to sum in order.
//
// A stage reads, coalesced along the rows, the packed codes of BN weight
// rows, their factors, and BM activation rows into registers, and lands
// them in LDS: the codes as they are, the factors as two planes, the
// activation shuffled into `Quads` order a word of codes at a time. The
// next stage's reads are in flight while the matrix cores take this one.
//
// The products run on the 16 x 16 x 16 core with the activation as A and
// the raw codes as B (lane group g of the four holds a quarter of each
// group of codes, so a product never mixes two groups), and a group folds
// into f32 as Σ x·w = s·Σ x·(c + magic) + (o − magic·s)·Σ x, with Σ x summed
// as the stage lands. K is a whole number of stages.
template <class T, class F, int kBits, int kOffset, int kGroup, int KS, int WM, int RS, int NT, int DEPTH>
__global__ __launch_bounds__(4 * kWave) void matmul_affine_tile(
    const T* __restrict__ act, const u8* __restrict__ codes, const u8* __restrict__ scales,
    const u8* __restrict__ biases, T* __restrict__ out, float* __restrict__ part, int per, int m, int n, int k,
    const AffineSeat* __restrict__ seat, const u32* __restrict__ win) {
    using Q = Quads<T, kBits>;
    constexpr int kThreads = 4 * kWave;
    constexpr int WN = 4 / WM;
    constexpr int BM = WM * 16 * RS, BN = WN * 16 * NT;
    constexpr int kGroups = KS / kGroup;      // groups a stage
    constexpr int kRowBytes = KS * kBits / 8;  // a weight row's codes a stage
    constexpr int kLaneBytes = kGroup * kBits / 32;  // a lane group's quarter of a group
    constexpr int kLaneWords = kLaneBytes / 4;
    constexpr int kSteps = kGroup / 16;  // 16 x 16 x 16 products a group
    constexpr int kCodePitch = kRowBytes + 16;  // LDS row strides, off the bank period
    constexpr int kActPitch = KS + 8;
    constexpr bool kOffsets = kOffset != kOffPreConst;
    constexpr int kOffBytes = kOffset == kOffPreInt ? 1 : 2;
    // Reads a thread a stage: 16-byte code chunks, 4-byte factor words,
    // 16-byte activation chunks (two a 16-code segment).
    constexpr int kCodeReads = BN * kRowBytes / 16 / kThreads;
    constexpr int kScaleWords = BN * kGroups * 2 / 4, kOffWords = BN * kGroups * kOffBytes / 4;
    constexpr int kScaleReads = (kScaleWords + kThreads - 1) / kThreads;
    constexpr int kOffReads = kOffsets ? (kOffWords + kThreads - 1) / kThreads : 0;
    constexpr int kActSegs = BM * KS / 16 / kThreads;
    static_assert(kWave == 64 && (WM == 1 || WM == 2 || WM == 4), "four 64-lane wavefronts");
    static_assert(kLaneBytes == 4 || kLaneBytes == 8 || kLaneBytes == 16, "a lane group takes whole words of a group");
    static_assert(KS % kGroup == 0 && kCodeReads * 16 * kThreads == BN * kRowBytes, "a stage is whole reads");
    static_assert(kActSegs * 16 * kThreads == BM * KS && (kGroups * 2) % 4 == 0 && (kGroups * kOffBytes) % 4 == 0,
                  "a stage's activation and factors are whole reads");
    static_assert(16 % (Q::kPer) == 0 || Q::kPer == 16, "a 16-code segment is whole words of codes");

    // A stage in LDS: the codes, the activation, the factor planes, and Σ x
    // of each group of codes of each activation row.
    __shared__ __align__(16) u8 ws[BN * kCodePitch];
    __shared__ __align__(16) T xs[BM * kActPitch];
    __shared__ __align__(16) u16 fs[BN * kGroups];
    __shared__ __align__(16) u8 fo[kOffsets ? BN * kGroups * kOffBytes : 4];
    __shared__ __align__(16) float sxs[kGroups][BM];

    const int rows = win == nullptr ? m : min(m, static_cast<int>(win[0]));
    const int m0 = blockIdx.x * BM;
    if (m0 >= rows) return;
    const AffinePlanes planes = seated(codes, scales, biases, seat);
    const int n0 = blockIdx.y * BN;
    const int tid = threadIdx.x, l = lane(), wave = tid / kWave;
    const int wm = wave / WN, wn = wave % WN;
    const int r16 = l % 16, g = l / 16;
    const int stages = k / KS;
    const int first = blockIdx.z * per, last = min(stages, first + per);
    if (first >= last) return;
    const int groups = k / kGroup;

    // A thread's reads, as byte offsets from the block's bases (wavefront-
    // uniform, so the loop walks scalar registers): its code chunks, factor
    // words and activation segments, rows past the live ones clamped to the
    // last.
    const long long row_bytes = static_cast<long long>(k) * kBits / 8;
    const u8* code_base = planes.codes + n0 * row_bytes;
    const u8* scale_base = planes.scales + static_cast<long long>(n0) * groups * 2;
    const u8* off_base = planes.biases + static_cast<long long>(n0) * groups * kOffBytes;
    const u8* act_base = reinterpret_cast<const u8*>(act + static_cast<long long>(m0) * k);
    u32 co[kCodeReads], so[kScaleReads], oo[kOffReads > 0 ? kOffReads : 1], xo[kActSegs];
#pragma unroll
    for (int i = 0; i < kCodeReads; ++i) {
        const int c = i * kThreads + tid, row = c / (kRowBytes / 16), col = c % (kRowBytes / 16);
        co[i] = static_cast<u32>((min(n0 + row, n - 1) - n0) * row_bytes) + col * 16;
    }
    auto factor_at = [&](int i, int words, int bytes) {  // word i·threads + tid of a factor plane
        const int w = min(i * kThreads + tid, words - 1);
        const int row = w * 4 / (kGroups * bytes), at = w * 4 % (kGroups * bytes);
        return static_cast<u32>((min(n0 + row, n - 1) - n0) * groups * bytes + at);
    };
#pragma unroll
    for (int i = 0; i < kScaleReads; ++i) so[i] = factor_at(i, kScaleWords, 2);
#pragma unroll
    for (int i = 0; i < kOffReads; ++i) oo[i] = factor_at(i, kOffWords, kOffBytes);
#pragma unroll
    for (int i = 0; i < kActSegs; ++i) {
        const int s = i * kThreads + tid, row = s / (KS / 16), col = s % (KS / 16);
        xo[i] = static_cast<u32>((min(m0 + row, rows - 1) - m0) * k * 2) + col * 32;
    }
    auto code_lds = [&](int i) {
        const int c = i * kThreads + tid;
        return c / (kRowBytes / 16) * kCodePitch + c % (kRowBytes / 16) * 16;
    };
    auto act_lds = [&](int i) {
        const int s = i * kThreads + tid;
        return s / (KS / 16) * kActPitch + s % (KS / 16) * 16;
    };

    struct Stage {
        Words<4> c[kCodeReads];
        u32 s[kScaleReads], o[kOffReads > 0 ? kOffReads : 1];
        Words<4> x[kActSegs][2];
    };
    auto load = [&](Stage& st, int sg) {
        const u8* cb = code_base + sg * kRowBytes;
        const u8* sb = scale_base + sg * kGroups * 2;
        const u8* ob = off_base + sg * kGroups * kOffBytes;
        const u8* xb = act_base + sg * KS * 2;
#pragma unroll
        for (int i = 0; i < kCodeReads; ++i) st.c[i] = read_global<Words<4>, true>(cb + co[i]);
#pragma unroll
        for (int i = 0; i < kScaleReads; ++i) st.s[i] = read_global<u32>(sb + so[i]);
#pragma unroll
        for (int i = 0; i < kOffReads; ++i) st.o[i] = read_global<u32>(ob + oo[i]);
#pragma unroll
        for (int i = 0; i < kActSegs; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) st.x[i][h] = read_global<Words<4>>(xb + xo[i] + 16 * h);
    };
    auto store = [&](const Stage& st) {
#pragma unroll
        for (int i = 0; i < kCodeReads; ++i) *reinterpret_cast<Words<4>*>(ws + code_lds(i)) = st.c[i];
#pragma unroll
        for (int i = 0; i < kScaleReads; ++i) {
            const int w = i * kThreads + tid;
            if (kScaleWords % kThreads == 0 || w < kScaleWords) reinterpret_cast<u32*>(fs)[w] = st.s[i];
        }
#pragma unroll
        for (int i = 0; i < kOffReads; ++i) {
            const int w = i * kThreads + tid;
            if (kOffWords % kThreads == 0 || w < kOffWords) reinterpret_cast<u32*>(fo)[w] = st.o[i];
        }
        // Sixteen activations in k order to `Quads` order, word of codes
        // by word of codes, and the sum of each group of them.
        constexpr int kSegs = kGroup / 16;  // segments a group
        static_assert(kSegs == 2 || kSegs == 4 || kSegs == 8, "a group is two, four or eight segments");
#pragma unroll
        for (int i = 0; i < kActSegs; ++i) {
            u32 d[8], q[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) d[e] = st.x[i][e / 4][e % 4];
            // The sum on the 4 x 4 x 4 core against ones: lane 4b + j's D[t]
            // is lane 4b + t's, a block being four adjacent segments of a row.
            const Words<2> ones{kOnes<T>, kOnes<T>};
            mfma_cd four = mfma_zero();
#pragma unroll
            for (int w = 0; w < 4; ++w) four = mfma4x4<T>(Words<2>{d[2 * w], d[2 * w + 1]}, ones, four);
            const int seg = i * kThreads + tid, row = seg / (KS / 16), col = seg % (KS / 16);
            float sum;
            if constexpr (kSegs == 2) {
                sum = col % 4 < 2 ? four[0] + four[1] : four[2] + four[3];
            } else {
                sum = (four[0] + four[1]) + (four[2] + four[3]);
                if constexpr (kSegs == 8) sum += __shfl_xor(sum, 4, kWave);
            }
            if (col % kSegs == 0) sxs[col / kSegs][row] = sum;
#pragma unroll
            for (int c = 0; c < 8; c += Q::kPer / 2)
#pragma unroll
                for (int h = 0; h < Q::kSteps; ++h) {
                    const Words<2> a = Q::act(d + c, h);
                    q[c + 2 * h] = a[0];
                    q[c + 2 * h + 1] = a[1];
                }
#pragma unroll
            for (int h = 0; h < 2; ++h)
                *reinterpret_cast<Words<4>*>(xs + act_lds(i) + 8 * h) =
                    Words<4>{q[4 * h], q[4 * h + 1], q[4 * h + 2], q[4 * h + 3]};
        }
    };

    mfma_cd acc[NT][RS];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int rs = 0; rs < RS; ++rs) acc[nt][rs] = mfma_zero();

    auto compute = [&]() {
#pragma unroll 2
        for (int gi = 0; gi < kGroups; ++gi) {
            // This lane group's quarter of group gi: its codes for NT weight
            // tiles, its activation for RS tiles; every tile's product of a
            // step issued together, so the cores have independent work.
            Words<kLaneWords> c[NT];
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int row = wn * 16 * NT + nt * 16 + r16;
                c[nt] = *reinterpret_cast<const Words<kLaneWords>*>(ws + row * kCodePitch + gi * (kGroup * kBits / 8) +
                                                                     g * kLaneBytes);
            }
            Words<4> a4[RS][kSteps / 2];
#pragma unroll
            for (int rs = 0; rs < RS; ++rs) {
                const int row = wm * 16 * RS + rs * 16 + r16;
                const T* xr = xs + row * kActPitch + gi * kGroup + g * (kGroup / 4);
#pragma unroll
                for (int s0 = 0; s0 < kSteps / 2; ++s0) a4[rs][s0] = *reinterpret_cast<const Words<4>*>(xr + 8 * s0);
            }
            mfma_cd d[NT][RS], sx[RS];
#pragma unroll
            for (int rs = 0; rs < RS; ++rs) {
                // Σ x of activation rows 4g .. 4g + 3 of the tile, as D lays them.
                sx[rs] = *reinterpret_cast<const mfma_cd*>(&sxs[gi][wm * 16 * RS + rs * 16 + 4 * g]);
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) d[nt][rs] = mfma_zero();
            }
#pragma unroll
            for (int s = 0; s < kSteps; ++s) {
                Words<2> b[NT];
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    Words<2> q[Q::kSteps];
                    Q::codes(word(c[nt], s / Q::kSteps), q);
                    b[nt] = q[s % Q::kSteps];
                }
#pragma unroll
                for (int rs = 0; rs < RS; ++rs) {
                    const Words<2> a{a4[rs][s / 2][2 * (s % 2)], a4[rs][s / 2][2 * (s % 2) + 1]};
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) d[nt][rs] = mfma16x16<T>(a, b[nt], d[nt][rs]);
                }
            }
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int row = wn * 16 * NT + nt * 16 + r16;
                u32 ob = 0;
                if constexpr (kOffset == kOffPreInt) {
                    ob = fo[row * kGroups + gi];
                } else if constexpr (kOffsets) {
                    ob = reinterpret_cast<const u16*>(fo)[row * kGroups + gi];
                }
                const Factor f = factor_bits<F, kBits, kOffset>(fs[row * kGroups + gi], ob);
                const float o = additive<kOffset>(f) - Q::kMagic * f.s;
#pragma unroll
                for (int rs = 0; rs < RS; ++rs)
#pragma unroll
                    for (int t = 0; t < 4; ++t)
                        acc[nt][rs][t] = fmaf(f.s, d[nt][rs][t], fmaf(o, sx[rs][t], acc[nt][rs][t]));
            }
        }
    };

    // DEPTH stages in flight: slot d lands stage first + i + d, then reads
    // stage first + i + d + DEPTH into itself. The steady loop's reads are
    // unconditional, so the compiler's waits count the slots behind the one
    // landing (a read under a branch makes it wait for all of them); the
    // last DEPTH to 2·DEPTH − 1 stages drain behind it.
    const int C = last - first;
    auto at = [&](int i) { return first + i; };
    auto stage = [&](const Stage& s) {
        store(s);
        lds_barrier();
        compute();
        lds_barrier();
    };
    Stage st[DEPTH];
    if (C < DEPTH) {
        for (int i = 0; i < C; ++i) {
            load(st[0], at(i));
            stage(st[0]);
        }
    } else {
#pragma unroll
        for (int d = 0; d < DEPTH; ++d) load(st[d], at(d));
        int i = 0;
        for (; i + 2 * DEPTH <= C; i += DEPTH) {
#pragma unroll
            for (int d = 0; d < DEPTH; ++d) {
                store(st[d]);
                lds_barrier();
                load(st[d], at(i + d + DEPTH));
                compute();
                lds_barrier();
            }
        }
#pragma unroll
        for (int d = 0; d < DEPTH; ++d) {
            store(st[d]);
            lds_barrier();
            if (i + d + DEPTH < C) load(st[d], at(i + d + DEPTH));
            compute();
            lds_barrier();
        }
#pragma unroll
        for (int d = 0; d < DEPTH; ++d)
            if (i + DEPTH + d < C) stage(st[d]);
    }

    // Lane (r16, g)'s D[t] is activation row 4g + t and weight row r16 of
    // its tile pair.
#pragma unroll
    for (int rs = 0; rs < RS; ++rs)
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int row = m0 + wm * 16 * RS + rs * 16 + 4 * g + t;
            if (row >= rows) continue;
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int col = n0 + wn * 16 * NT + nt * 16 + r16;
                if (col >= n) continue;
                if (part != nullptr) {
                    part[(static_cast<long long>(blockIdx.z) * rows + row) * n + col] = acc[nt][rs][t];
                } else {
                    out[static_cast<long long>(row) * n + col] = from_f32<T>(acc[nt][rs][t]);
                }
            }
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
