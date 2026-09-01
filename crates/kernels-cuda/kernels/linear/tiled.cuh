// **THE TILED AFFINE POINT** — `linear.matmul` over the post-affine W4A16
// form (`g{N}_u4_bf16_b_bf16`, MLX's affine triplet at four bits), folded
// inside an m16n8k16 tensor-core mainloop instead of re-read per activation
// row.
//
// This is §J4's hybrid, PHASE A (correctness, one fixed config) as tuned by
// PHASE B (the epilogue, the stage depth, the weight superword, and a second
// config tuple for the tall-skinny direction). `linear/quant.cuh` keeps both
// of the arms a caller has today — the fused GEMV (`matmul_affine`, one
// block column per token, parity with cuBLAS at one token and ~100x slower
// over a prefill) and the interim decoded-tile arm (`dequant_affine` into
// scratch, then cuBLAS). This point is the third: the weight is read ONCE,
// as nibbles, and never materialises.
//
// # What is vendored, and what is written fresh
//
// TWO THINGS come from vllm/marlin (`origin/dev`,
// `driver/cuda/third_party/marlin/`), because they are the two pieces the
// §J4 recon judged hard to reinvent and easy to get subtly wrong:
//
//   1. `dequant_u4_bf16x2` below is `dequant.h:174-200`'s int4->bf16 lop3
//      trick — the `0x000f000f` / `0x43004300` magic-constant pair that
//      turns four nibbles into two bf16 lanes with one instruction.
//      MODIFICATIONS: the `vllm::ScalarTypeId` template axis, the fp16 and
//      fp8 and fp4 specialisations, the `skip_flop` axis and the
//      `scalar_type.hpp` dependency are all gone; what is left is the one
//      specialisation this point fires (`kU4`, no bias) with the shifts
//      written out at the call site rather than folded into `q >>= 4`.
//   2. The B-tile ownership `repack_affine_tiled` writes is
//      `gptq_marlin_repack.cu:124-204`'s, transcribed: k-offsets
//      {0, 1, 8, 9} off `2*(lane%4)`, two n-columns eight apart off
//      `lane/4`, and the nibble interleave `{0, 2, 4, 6, 1, 3, 5, 7}`. That
//      permutation is not decorative — it is exactly the m16n8k16
//      B-fragment ownership, which is why ONE 32-bit word holds a whole
//      lane's B fragment for two columns and why the dequant above can be
//      applied to it four times at shifts {0, 4, 8, 12} and land in
//      fragment order with no shuffle.
//
// Everything else is this tree's: `prelude/mma.cuh`'s m16n8k16 wrapper,
// `<cuda_pipeline.h>`'s cp.async, the shim's `__hfma2`/`__hsub2`, and the
// `win` guard convention every new point takes. No marlin host C++, no
// `marlin_template.h`, no global locks, no split-K.
//
// # The fold is the reason this is not just a dequant
//
// A four-bit code `c` under the post-affine arm decodes to `s*c + b`. The
// lop3 lands `128 + c` in a bf16 lane (`0x4300` is 128.0, and the low four
// mantissa bits ARE the code, so 128..143 are exact); `__hsub2` takes the
// 128 back off exactly; `__hfma2` then folds `s*c + b` with ONE rounding.
// That single rounding is what makes this point's oracle the same host fold
// the interim arm's golden already uses — see `tests/tiled_matmul.rs`. It
// survived every phase-B move below, and it is the property that pins them:
// an optimisation that rounds twice is not this kernel.
//
// The alternative — folding the bias into a per-group constant
// `b - 128*s` so the subtraction disappears — was rejected: that constant
// has to be rounded to bf16 itself, and it is four times the weight's own
// magnitude, so the rounding lands ~0.8% of an element instead of ~0.2%.
// One extra instruction per pair buys back a bit-exact reference.
//
// # The config is a TUPLE NOW, and there are two of them
//
// Phase A had one, chosen for sm_89 and not searched: 128 threads (4
// warps), a 64x64 output tile over a 64-wide contraction step, two cp.async
// stages. Phase B made `(M, N, threads, stages)` template parameters and
// swept them on both projection directions a serving transformer fires
// (`n >= k`: up and gate, k 2048 by n 10240; `k > n`: down, k 10240 by n
// 2048) at nine row counts. Two things came out of it:
//
//   * EIGHT WARPS, not four. A 64x128 tile over 256 threads is 0.118 ms at
//     512x2048x10240 against the phase-A tile's 0.140, and it is ahead on
//     both directions at every row count from 512 up. Doubling the column
//     tile doubles what one staged activation row is multiplied by, and
//     eight warps is what covers the dequant behind the mma once the
//     weight superword below has cut the load count.
//   * THE SECOND TUPLE IS A SHORTER ROW TILE, and the axis it turns on is
//     the ROW COUNT and not the aspect ratio. A down projection is sixteen
//     column tiles wide, so at 128 rows a 64-row tile carves 32 blocks over
//     an L40S's 142 SMs; halving the row tile doubles the grid and takes
//     0.147 ms to 0.096. Above 512 rows there are blocks to spare and the
//     taller tile's reuse wins instead — in BOTH directions, which is why
//     `src/linear/tiled.rs` picks on rows alone and says so.
//
// # What phase B changed inside the mainloop, and what it bought
//
//   (i)   THE EPILOGUE GOES THROUGH SHARED MEMORY. The accumulator lane map
//         is eight rows by two columns per instruction, so a direct store
//         wrote eight 16-byte segments and ran at half transaction
//         efficiency over 10.5MB of output. The tile is staged into the
//         (now dead) activation buffer and drained 16 bytes a lane, so a
//         warp's store is four whole 128-byte rows. Worth 10-14% where the
//         output is the big rectangle (`n >= k`) and inside the noise where
//         it is not.
//   (ii)  A K-MAJOR WEIGHT SUPERWORD: the repacked plane groups four k tiles
//         as `[lane][4]`, so a lane pulls a whole 64-wide contraction step's
//         B fragments as ONE `uint4` instead of four strided words, one step
//         ahead of the mma that reads them. This is a REPACK LAYOUT CHANGE;
//         `repack_affine_tiled` is its only writer and
//         `tests/tiled_matmul.rs`'s host un-repack is its guard. It is
//         neutral at four warps and it is 1.4x at eight, on the long
//         contraction — which is the whole reason the eight-warp tile is
//         reachable at all.
//   (iii) FOUR cp.async STAGES, WHICH LOST. It was on the list and it is
//         recorded here because it lost: a 64-row stage is 9KB, four of them
//         is 36KB, and an L40S then holds two blocks an SM instead of five.
//         0.160 ms against 0.140 at the phase-A tile, same sign everywhere
//         else. Four stages ships only under the short-row tuple, whose
//         stage is half the size.
//
// # What phase B did not do, and §J4b did
//
// **THE ARM IS WIRED NOW.** Phase B's blocker was that the tiled point reads
// a REPACKED plane while dispatch held the plane the checkpoint landed, and
// firing one on the other is unrefusable — same rectangles, wrong answer.
// Three things closed it: `gemv_affine_tiled` below, so a decode step has a
// reader of this layout too; `Expr::Repack` at the CONVERT target, so the
// relabelling is paid once at `pie model import`; and
// `dtype::Dtype::MlxU4Tiled` on the weight's declaration, so the engine's
// `WeightRow` can say which order it holds. `src/linear/tiled.rs` carries
// the numbers all three stand on.
#pragma once

#include "prelude/device.cuh"
#include "prelude/mma.cuh"

#include <cuda_bf16.h>
#include <cuda_pipeline.h>

namespace pie::linear {

// ─── what every tuple shares ───────────────────────────────────────────────

/// The mma shape, which is the whole file's unit of account: a k step is
/// sixteen wide, an n subtile is eight, and the repacked plane's tile is
/// 16 by 16 because that is what one warp's 32 words cover.
constexpr int kMmaK = 16;
constexpr int kMmaM = 16;
constexpr int kMmaN = 8;

/// The contraction step, and it is NOT a tuple axis. Four 16-wide mma k
/// tiles is exactly the `uint4` superword a lane pulls in one instruction,
/// so a step that is not 64 wide would either split that load or leave part
/// of it unused. Both tuples walk k sixty-four at a time.
constexpr int kTiledK = 64;

/// The k tiles one step covers, which is the superword's word count.
constexpr int kTiledQuad = kTiledK / kMmaK;

/// The staged activation tile's row stride, in bf16.
///
/// Eight elements of padding, which is sixteen bytes: a row is then 144
/// bytes and `row * 144 / 16 mod 8` walks 0..7 as `row` does, so the eight
/// addresses one `ldmatrix` issues land in eight different shared-memory
/// segments. Unpadded (128 bytes a row) they would all land in one.
constexpr int kTiledLdA = kTiledK + 8;

/// A warp's own column band, which is the mma B fragment's n extent doubled
/// — one repacked word serves column `c` and column `c + 8`, so sixteen
/// columns is the smallest unit a warp can own without splitting a word.
constexpr int kTiledBand = kMmaK;

// ─── vendored: the lop3 dequant ────────────────────────────────────────────

/// Lookup-table three-input logical op.
///
/// VENDORED from marlin `dequant.h:73-81` unchanged. It is written as inline
/// PTX rather than left to the compiler for the reason upstream states: the
/// `(a & b) | c` idiom below is not reliably recognised as a single LOP3.
template <int lut>
__device__ __forceinline__ int pie_lop3(int a, int b, int c) {
    int res;
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(res)
                 : "r"(a), "r"(b), "r"(c), "n"(lut));
    return res;
}

/// Two four-bit codes out of `q`, as a packed bf16 pair holding
/// `(128 + code_lo, 128 + code_hi)`.
///
/// VENDORED from marlin `dequant.h:174-186` — the `kU4B8`/`kU4` bf16
/// specialisation with `skip_flop = true`. MODIFICATIONS: the ScalarType
/// template axis and the `q >>= 4` sequencing are dropped, so a caller
/// picks its own shift and gets one pair per call.
///
/// The magic is `0x43004300`: bf16 `0x4300` is 128.0, whose mantissa bits
/// are all zero, so OR-ing a nibble into the low four of them lands exactly
/// `128 + nibble` — every one of 128..143 fits bf16's seven mantissa bits.
/// The low nibble of each 16-bit half becomes that half's code, which is
/// why the two codes a call answers are `q & 0xF` and `(q >> 16) & 0xF`.
__device__ __forceinline__ unsigned dequant_u4_bf16x2(unsigned q) {
    constexpr int kLo = 0x000f000f;
    constexpr int kEx = 0x43004300;
    // Guarantee that the `(a & b) | c` operation is a LOP3.
    return static_cast<unsigned>(
        pie_lop3<(0xf0 & 0xcc) | 0xaa>(static_cast<int>(q), kLo, kEx));
}

// ─── this tree's: packing, the fold, and the A fragment ────────────────────

/// The 128.0 pair `dequant_u4_bf16x2` biases by, as bits.
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

/// One bf16 in both lanes of a pair, as bits. Both halves of a B-fragment
/// register are the same OUTPUT COLUMN at two contraction positions, so one
/// group's scale and bias broadcast across them and no per-lane selection
/// is needed anywhere in the mainloop.
__device__ __forceinline__ unsigned splat(bf16 v) {
    const unsigned h = v.raw;
    return (h << 16) | h;
}

/// `s*c + b` for two codes at once — the post-affine fold, in registers,
/// with ONE rounding.
///
/// `dq` is `dequant_u4_bf16x2`'s answer, so it holds `128 + c`; the
/// subtraction is exact and the `__hfma2` is the only place a bit is lost.
/// This is §J4's `scale_and_sub` written the other way up: marlin subtracts
/// a zero point before scaling and needs a `__hneg2` to do it, and a
/// post-offset bias is ADDED, so the negation never appears.
__device__ __forceinline__ unsigned fold_post(unsigned dq, unsigned s2, unsigned b2) {
    const __nv_bfloat162 code = __hsub2(as_pair(dq), as_pair(kTiledMagic));
    return as_bits(__hfma2(code, as_pair(s2), as_pair(b2)));
}

/// One m16n8k16 A fragment out of shared memory, in one instruction.
///
/// `ldmatrix.x4` loads four 8x8 tiles: lane `l` hands it the address of row
/// `l % 16` at column `(l / 16) * 8` of the 16x16 tile, and the four
/// registers come back as the fragment's `a0..a3` already — rows `g` and
/// `g + 8` at columns `2t..2t+1` and `2t+8..2t+9`, which is
/// `prelude/mma.cuh`'s `load_matrix_sync` lane map with the eight scalar
/// loads it spends replaced by one.
__device__ __forceinline__ void ldmatrix_a(unsigned (&reg)[4], const bf16* at) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(at));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
                 : "r"(addr));
}

// ─── the point ─────────────────────────────────────────────────────────────

// **THE TILED POST-AFFINE PROJECTION**: `y = act x W^T`, `W[n][k] =
// s[n][g]*code[n][k] + b[n][g]`, over the REPACKED planes
// `repack_affine_tiled` and `repack_factors_tiled` write.
//
// The stored form does not change. What changes is the ORDER the same bits
// sit in, decided once at load time, so that a lane's whole contraction
// step of B fragments is one aligned `uint4` and its two factors are two
// adjacent halves. `src/linear/tiled.rs` states the layout;
// `repack_affine_tiled` below is its only writer and
// `tests/tiled_matmul.rs` un-repacks it bit-exactly.
//
// **THE PLANES, AS THIS POINT READS THEM**
//
//   codes    a `uint4` sequence of QUADS — four 16(k) x 16(n) tiles at a
//            time, 32 superwords each, in (n band, k quad) order.
//            Superword `lane` of quad `(nb, kq)` is at
//            `((nb * (k / 64)) + kq) * 32 + lane`, and its four words are
//            the four k tiles `4*kq .. 4*kq+3` in order. Each word's eight
//            nibbles are marlin's interleave, so the four shifts
//            {0, 4, 8, 12} answer (b0, b1) for column `lane/4` and then for
//            column `lane/4 + 8`.
//   scales   bf16, `[n band][group][16]` — the sixteen columns of a band
//            adjacent, so a lane's two (eight apart) are two loads.
//   biases   the same rectangle, the same order.
//
// Both are padded up to a whole 16-column band with zero codes and zero
// factors, which decode to a zero weight, so the tail columns compute a
// harmless zero and the epilogue simply does not write them.
//
// **THE `win` GUARD** is `matmul_affine`'s, generalised from a token to a
// tile: a replay whose grid was carved at a bucket retires its padded rows
// by taking `win[0]` as the row count, which masks both the activation
// staging and the epilogue. `nullptr` is the resident case.
//
// **NO STREAMED SEAT.** A plane that moves between fires has no fixed
// rectangle to have been repacked into; `src/linear/tiled.rs` refuses one,
// on the same ground `matmul_via_dense` does.
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

    /// One stage of the staged activation tile, in bf16 elements.
    constexpr int kStageElems = kM * kTiledLdA;
    /// The epilogue tile's row stride. Same eight elements of padding, and
    /// they do the same job one step later: the accumulator's eight rows
    /// land in eight different banks instead of one.
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

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
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

    // **THE M EDGE IS A ZERO, WRITTEN ONCE — AND ONLY BY THE BLOCK THAT HAS
    // ONE.** cp.async never touches a row past `rows`, so zeroing the
    // staging buffer is the entire edge handling on the activation side.
    // A block whose rows are all live never reads an unwritten byte, so it
    // skips a zeroing that at four stages is 36KB of shared memory.
    if (m0 + kM > rows) {
        u32* z = reinterpret_cast<u32*>(tiled_smem);
        constexpr int kWords = kStages * kStageElems / 2;
#pragma unroll
        for (int i = 0; i < kWords / kThreads; ++i) {
            z[tid + i * kThreads] = 0u;
        }
        __syncthreads();
    }

    // The staging map: one thread moves 16 bytes (8 bf16) per chunk, and
    // eight adjacent threads cover one row of the step, so a warp's fetch
    // is four whole rows and each of them is contiguous.
    constexpr int kChunksPerRow = kTiledK / 8;
    constexpr int kChunks = kM * kChunksPerRow;
    constexpr int kChunksPerThread = kChunks / kThreads;

    const int steps = k / kTiledK;
    const int groups = k / kGroup;

    // The warp's own 16-column band of the block's `kN`. A band wholly past
    // `n` is a padded one: its codes and factors are zero, so it walks the
    // mainloop with the others, accumulates zero, and stages a zero the
    // epilogue does not write back.
    const int band = (n0 / kTiledBand) + warp;
    const bool live = band * kTiledBand < n;
    const int col_of = lane >> 2;
    const int row_of = lane & 3;

    // ONE SUPERWORD IS ONE CONTRACTION STEP. The repacked plane groups four
    // k tiles as `[lane][4]`, so the four B-fragment words a lane spends a
    // step on arrive in a single 16-byte load and a warp's fetch is 512
    // contiguous bytes.
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

    // The two columns' factors, held across k tiles and reloaded only when
    // the group turns over — at `kGroup >= 64` that is once a step or less.
    unsigned s2[kNSubs];
    unsigned b2[kNSubs];
#pragma unroll
    for (int ns = 0; ns < kNSubs; ++ns) {
        s2[ns] = 0u;
        b2[ns] = 0u;
    }
    int held = -1;

    // One stage's worth of activation rows, issued and left in flight.
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

    // **THE PROLOGUE IS `kStages - 1` GROUPS, REAL OR EMPTY.** A commit
    // happens whether or not there was a step to stage, so the group count
    // is uniform and `wait_prior(kStages - 1)` means the same thing on
    // every iteration including the last `kStages - 1` of them.
#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < steps) {
            stage(s, s * kTiledK);
        } else {
            __pipeline_commit();
        }
    }

    // The superword for step 0, pulled before the loop so that every
    // iteration's weight load is one step ahead of the mma that reads it.
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

            // ONE word is this lane's whole B fragment for both n subtiles:
            // shifts {0, 4} are (b0, b1) for column `col_of`, shifts
            // {8, 12} the same for column `col_of + 8`. That is marlin's
            // interleave paying for itself.
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

    // **THE EPILOGUE GOES THROUGH SHARED MEMORY**, and the buffer it goes
    // through is the activation staging one — the mainloop is over, the
    // last `__syncthreads` above is the fence, and the tile it wants is
    // never larger than one stage plus change.
    //
    // The reason is transaction efficiency. The accumulator's lane map is
    // `prelude/mma.cuh`'s `store_matrix_sync` one — `row_of` picks the
    // column pair, `col_of` the row — so a warp's direct store is eight
    // rows by four bytes, which is eight 16-byte pieces of eight different
    // 32-byte sectors. Staged and re-read, a warp's store is 32 lanes by 16
    // bytes over four whole 128-byte rows, and every sector is full.
    __pipeline_wait_prior(0);
    __syncthreads();
    bf16* c_tile = reinterpret_cast<bf16*>(tiled_smem);
    {
        // Two accumulator registers are two ADJACENT columns of one row, so
        // the staging store is one 32-bit word and the padded stride makes
        // the eight rows of a fragment eight different banks.
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

    // **THE DRAIN, IN ONE OF TWO WIDTHS.** A block whose column tile is
    // whole and whose `n` is a multiple of eight stores 16 bytes a lane;
    // anything ragged — the last column tile of a projection whose `n` is
    // not a whole tile, or an `n` a 16-byte store would straddle — stores
    // one element a lane, still with `tid` walking columns so the warp is
    // contiguous. Both read the same staged tile, so the ragged path is the
    // same numbers and not a second epilogue.
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
#else
    (void)act;
    (void)codes;
    (void)scales;
    (void)biases;
    (void)out;
    (void)m;
    (void)n;
    (void)k;
    (void)win;
    __trap();
#endif
}

// **THE SAME REPACKED PLANES AT A DECODE STEP** — `y = act x W^T` over
// `repack_affine_tiled`'s output at the row counts a serving step brings,
// where the mma tile above has nothing to multiply and the cost is the
// weight's own bytes.
//
// It is here and not in `linear/quant.cuh` for the reason that file's fused
// GEMV is there: a point belongs beside the LAYOUT it reads. The repack is
// this file's, its only writer is thirty lines below, and a decode point
// that gathered through it from another translation unit would be the third
// place the nibble map is written down. The two readers and the writer sit
// together, and `tests/tiled_matmul.rs` holds all three against one fold.
//
// **WHY NOT JUST FIRE `matmul_affine_tiled` WITH A ONE-TILE ROW COUNT.**
// Because the tile point is arranged for ARITHMETIC and a decode is
// arranged for BYTES IN FLIGHT. Its grid is one block per (row tile,
// 128-column tile), so a 10240-column projection carves eighty blocks over
// an L40S's 142 SMs, each warp holding one 16-byte superword and the next
// one prefetched: about 10KB of the machine's ~640KB bandwidth-delay
// product is ever outstanding. At 512 rows that is invisible behind the
// mma; at one row it is the whole cost, and the point would run at a
// fraction of the memory it is bound by.
//
// **SO THE GRID IS CARVED ON THE WEIGHT AND NOT ON THE OUTPUT.** A warp
// owns one 16-column band and a slice of the contraction; a block is
// `kBands` bands by `kSplit` slices, and the two are a config tuple because
// the two projection directions want opposite carvings — a wide `n` has
// bands to spare and splits nothing, a tall `k` has few bands and must
// split the contraction to fill the machine. `src/linear/tiled.rs` picks.
//
// **THE LANE MAP IS THE REPACK'S OWN, READ THE OTHER WAY.** Lane `l` of a
// superword holds, at nibble `s + 4h`, the code at
// `k = 16*kt + 2*(l%4) + 8*(s&1) + h` and `n = 16*band + l/4 + 8*(s>=2)`.
// So `dequant_u4_bf16x2(q >> 4*(2*ns + h))` answers the SAME bf16 pair the
// mainloop above calls a B fragment — column `l/4 + 8*ns`, contraction
// positions `16*kt + 2*(l%4) + 8*h` and that plus one — and the two
// activations it multiplies are ONE 32-bit load, because those two
// positions are adjacent. Four lanes (`l%4 = 0..3`) share a column, so the
// dot closes with two `__shfl_xor`s and nothing else.
//
// **THE FOLD IS `matmul_affine_tiled`'s, ELEMENT FOR ELEMENT.** `fold_post`
// materialises `s*c + b` as a bf16 with one rounding; this point then
// widens it and multiplies in f32, which is what the host oracle
// (`tests/tiled_matmul.rs`'s `fold_decoded`) computes. So the two readings
// of the tiled layout agree to the accumulation order and no further, which
// is the same standing the tile point has against the fused GEMV.
//
// **THE `win` GUARD** is the tile point's: `win[0]` is the live row count a
// replay staged, `nullptr` the resident case. **NO STREAMED SEAT**, on the
// same ground — a plane that moves between fires was never repacked.
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

    /// The n subtiles one lane's word covers -- columns `l/4` and `l/4 + 8`.
    constexpr int kNSubs = kTiledBand / kMmaN;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
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
        // ONE SUPERWORD IS ONE CONTRACTION STEP, and this warp takes every
        // `kSplit`-th of them, so the `kSplit` warps of one band issue
        // adjacent 512-byte fetches and the block's read of a step is
        // contiguous.
        const uint4* wq = reinterpret_cast<const uint4*>(codes)
            + static_cast<long long>(band) * steps * 32 + lane;
        const bf16* sf = reinterpret_cast<const bf16*>(scales);
        const bf16* bp = reinterpret_cast<const bf16*>(biases);
        // The activations, as the PAIRS a fragment register multiplies:
        // positions `2t` and `2t+1` of a 16-wide k tile are adjacent, so one
        // 32-bit load is one lane's whole half-fragment of one row.
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

                // The four fragment halves this lane's word holds, folded
                // once and widened once -- they are multiplied by `kRowsT`
                // activation rows below, so the dequant is paid per WEIGHT
                // and not per row. That is the whole difference from the
                // fused GEMV, which pays it per row.
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

                // `kt * 8 + row_of` is the 32-bit word holding activations
                // `16*kt + 2*row_of` and the next; `+ 4` is the same pair
                // eight positions along, which is the fragment's other half.
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

    // **THE DOT CLOSES ACROSS FOUR LANES**, because `l % 4` is the only axis
    // of the repack's lane map that walks the contraction: lanes `4c..4c+3`
    // hold four disjoint quarters of column `16*band + c`.
#pragma unroll
    for (int r = 0; r < kRowsT; ++r) {
#pragma unroll
        for (int ns = 0; ns < kNSubs; ++ns) {
            acc[r][ns] += __shfl_xor_sync(0xffffffffu, acc[r][ns], 1);
            acc[r][ns] += __shfl_xor_sync(0xffffffffu, acc[r][ns], 2);
        }
    }

    // **AND ACROSS THE SLICES, THROUGH SHARED MEMORY** — only when there
    // are slices. At `kSplit == 1` a warp owns its band's whole contraction
    // and this whole stage, its barrier and its shared memory, compiles out.
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
#else
    (void)act;
    (void)codes;
    (void)scales;
    (void)biases;
    (void)out;
    (void)m;
    (void)n;
    (void)k;
    (void)win;
    (void)kNSubs;
    __trap();
#endif
}

// **THE RELABELLING, AND IT IS ONLY A RELABELLING** — the load-time pass
// that puts a dense `[n, k]` four-bit plane into the order
// `matmul_affine_tiled` reads. Shifts, masks and ors; no arithmetic touches
// a code, so the served row is the stored row and serve-as-stored holds the
// way it holds for `dequant_affine`'s scratch decode.
//
// The layout it writes is `gptq_marlin_repack.cu:124-204`'s B tile, which
// is the m16n8k16 B-fragment ownership and nothing else:
//
//   lane `l` of a 16(k) x 16(n) tile owns `t = l % 4`, `c = l / 4`, and its
//   eight nibbles are, in order 0..7:
//
//     (k = 2t+0, n = c)   (k = 2t+8, n = c)   (k = 2t+0, n = c+8)
//     (k = 2t+8, n = c+8) (k = 2t+1, n = c)   (k = 2t+9, n = c)
//     (k = 2t+1, n = c+8) (k = 2t+9, n = c+8)
//
// — upstream's `tc_offsets = {0, 1, 8, 9}` against `tc_row = 2*(th_id%4)`,
// its two columns eight apart off `th_id/4`, and its
// `pack_idx = {0, 2, 4, 6, 1, 3, 5, 7}`. Read the other way: nibble
// `s/4 + 4h` is the `h`-th bf16 lane of the pair the dequant answers at
// shift `s`, and the four shifts walk (b0, b1) of column `c` and then of
// column `c + 8`.
//
// WHAT IS THIS TREE'S AND NOT UPSTREAM'S is the tile ORDER, and phase B
// made it one order stronger. Upstream keys a tile by `(k_tile, n_tile)`
// with `n` innermost; phase A keyed it by `(n band, k tile)` with `k`
// innermost, so a warp's k walk down one band was a sequential stream of
// 128-byte loads. Phase B groups FOUR k tiles — one whole contraction step
// — as `[lane][4]`, so that stream is 512-byte loads and a lane spends one
// instruction per step instead of four. Nothing about the fragment depends
// on either choice; both are prefetch decisions, and the second is why
// `matmul_affine_tiled` reads `uint4`.
//
// ONE THREAD PER OUTPUT WORD, which is eight input nibbles gathered from
// eight places. This runs once per weight at load and is not a shape the
// mainloop's economics apply to.
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

    // The superword is the innermost axis now: four words of one lane, then
    // the next lane, then the next quad, then the next band.
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
        // Shifts 0 and 4 serve column `c`, shifts 8 and 12 column `c + 8`;
        // within a column, shift 4 is the `+8` half of the k quad.
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

// **THE FACTOR PLANES' HALF OF THE RELABELLING** — `[n][group]` becomes
// `[n band][group][16]`, which is a transpose of the (column, group)
// rectangle inside each band and nothing more.
//
// The permutation is chosen to be the SIMPLEST one that makes a lane's read
// short: it needs the factors of columns `c` and `c + 8` of one band at one
// group, and this puts a band's sixteen columns adjacent, so those are two
// halves eight apart in one 32-byte run. §J4's recon reports dev's scale
// permute as an 8x8 transpose per 64-wide row; the fragment ownership here
// is 16 columns wide, so 16 is the run.
//
// A band's columns past `n` are written as a zero factor, which with the
// zero code `repack_affine_tiled` writes there makes the padded weight
// exactly zero.
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

}  // namespace pie::linear
