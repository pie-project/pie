#include "linear/codec.cuh"
#include "prim/device.cuh"
#include "prim/mma_sm100.cuh"
#include "prim/act.cuh"

namespace pie::linear {

namespace tc5k {

__host__ __device__ constexpr unsigned stage_k(int kind) {
    return kind == 0 ? 64u : kind == 3 ? 256u : 128u;
}

__host__ __device__ constexpr unsigned atoms_k(int kind, int gran) {
    return kind < 2 ? 0u : (stage_k(kind) / gran / 4 > 0 ? stage_k(kind) / gran / 4 : 1u);
}

__host__ __device__ constexpr unsigned pow2_cols(unsigned n) {
    unsigned c = 32;
    while (c < n) c <<= 1;
    return c;
}

}


// What the epilogue does past the product (host mirror: tc5.rs `EpiArgs`).
struct Epi {
    int inter;               // a GLU's output columns (its gate rows; up rows follow)
    float limit, galpha;     // the clamp variants' limit, clamp_alpha's alpha
    unsigned char* q;        // the quantized output, [rows][inter] (fp4: inter / 2)
    unsigned char* qs;       // its scale atoms, cols inter / 128 (fp8) or / 16 (fp4)
    float gscale;            // NVFP4's global scale, 1 / input_scale
    const float* row_scale;  // a weight each output row (routed), or null
    int pad;                 // masked: scale rows an expert; 0 otherwise
    const bf16* bias;        // a bias each weight row, added before the GLU, or null
    const unsigned char* wsf;  // a W4A16 weight's scales, row-major [rows][k / gran]
    const bf16* wzero;         // an affine W4A16 weight's zeros, beside its scales
    const float* wscale;       // an f32 scale each weight row (NVFP4's tensor scale), or null
    const int* live;           // a contiguous group's live rows an expert, or null
    // An NVFP4 store's input scale an expert (its global 1 / that), or null
    // for `gscale`; with it, a value is rounded to bf16 before it is
    // quantized, as the GLU's bf16 row would be.
    const float* in_scale;
    // A dense launch's P tiles a raster band (`kernel.gemm.tc5_raster`).
    int raster;
    // A split's own slab of `part`: split s's sums at part + s * slab, each
    // summed in split order by whoever lands the tile (deterministic, no
    // float atomics).
    long long slab;
    // A split tile's arrival count a CTA (`kernel.gemm.tc5_split_tickets`),
    // zero between launches, or null for the finalize's launch: the last of
    // a tile's splits to land sums the slabs and rounds y itself.
    unsigned* tickets;
    // Swapped: the weights, their scales and the group were landed before
    // the launch this one depends on (a routed down leg after its gate-up).
    int weights_ahead;
    // Grouped and swapped: from this many waves (0, never) the units dealt
    // over the fewest CTAs that keep them, the same count each
    // (`kernel.moe.even_waves_from`).
    int even_waves;
    // The routing entries shared memory holds (a grouped launch's experts and
    // one, at most 1028; `kernel.moe.routes_sized`), or 0 for 1028.
    int route_cap;
};

namespace tc5k {

using ::pie::glu_fn::glu;

// DeepSeek's 1x128 ue8m0 scale for a block whose largest magnitude is
// amax: 2^ceil(log2(max(amax, 1e-4) / 448)), as its exponent.
__device__ __forceinline__ int ue8m0_exponent(float amax) {
    int e;
    const float m = frexpf(fmaxf(amax, 1e-4f) / 448.f, &e);
    return m == 0.5f ? e - 1 : e;
}

// A W4A16 weight's transform warps.
constexpr unsigned kXWarps = 8;
constexpr unsigned kXGroupsMax = 2;

// Eight e2m1 codes (element i in bits 4i..4i+3) as bf16 times a block
// scale: the magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6 by byte permutes of their
// bf16 high and low bytes, the sign or'd in, then one exact bf16 multiply a
// pair (mxfp4: ue8m0 2^(b - 127), 0 standing for 2^-127; nvfp4: an e4m3).
template <int kMix>
__device__ __forceinline__ uint4 dequant8(unsigned code, unsigned char sb) {
    auto four = [](unsigned nib, unsigned& a, unsigned& b) {
        const unsigned sel = nib & 0x7777u;
        const unsigned hi = __byte_perm(0x3F3F3F00u, 0x40404040u, sel);
        const unsigned lo = __byte_perm(0xC0800000u, 0xC0804000u, sel);
        a = __byte_perm(lo, hi, 0x5140u) | ((nib << 12) & 0x8000u) | ((nib << 24) & 0x80000000u);
        b = __byte_perm(lo, hi, 0x7362u) | ((nib << 4) & 0x8000u) | ((nib << 16) & 0x80000000u);
    };
    unsigned scale;
    if constexpr (kMix == 1) {
        scale = sb == 0 ? 0x0040u : static_cast<unsigned>(sb) << 7;
    } else {
        const unsigned e = (sb >> 3) & 15u, m = sb & 7u;
        float f = e == 0 ? static_cast<float>(m) * 0.001953125f
                         : __int_as_float(static_cast<int>(((e + 120u) << 23) | (m << 20)));
        if (e == 15 && m == 7) f = __int_as_float(0x7fc00000);
        scale = (__float_as_uint(f) >> 16) | ((sb & 0x80u) << 8);
    }
    scale |= scale << 16;
    unsigned w[4];
    four(code, w[0], w[1]);
    four(code >> 16, w[2], w[3]);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(w[i]) : "r"(w[i]), "r"(scale));
    }
    return make_uint4(w[0], w[1], w[2], w[3]);
}

// Two and four bytes likewise.
__device__ __forceinline__ unsigned ld_nc_u16(const unsigned char* p) {
    unsigned short v;
    asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ unsigned ld_nc_u32(const unsigned char* p) {
    unsigned v;
    asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

// One byte from global memory through the non-coherent path, where the
// asm stands.
__device__ __forceinline__ unsigned ld_nc_u8(const unsigned char* p) {
    unsigned short v;
    asm volatile("ld.global.nc.u8 %0, [%1];" : "=h"(v) : "l"(p));
    return v;
}

// Eight unsigned 4-bit codes (element i in bits 4i..4i+3) as bf16 of
// fma(code, scale, zero), rounded once as pie's affine kernels round them.
__device__ __forceinline__ uint4 dequant8_affine(unsigned code, float scale, float zero) {
    unsigned w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float a = __int_as_float(0x4b000000 | ((code >> (8 * i)) & 15u)) - 8388608.f;
        const float b = __int_as_float(0x4b000000 | ((code >> (8 * i + 4)) & 15u)) - 8388608.f;
        w[i] = pack_bf16x2(fmaf(a, scale, zero), fmaf(b, scale, zero));
    }
    return make_uint4(w[0], w[1], w[2], w[3]);
}

__device__ __forceinline__ float bf16_bits_f32(unsigned bits) {
    return __uint_as_float(bits << 16);
}

// 2^-e for a block exponent: exact, and in range from the 1e-4 floor up.
__device__ __forceinline__ float pow2_neg(int e) {
    return __int_as_float((127 - e) << 23);
}

__device__ __forceinline__ unsigned short e4m3x2(float lo, float hi) {
    unsigned short r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
    return r;
}

__device__ __forceinline__ unsigned e2m1x8(const float (&v)[8]) {
    unsigned r;
    asm("{\n\t.reg .b8 b0, b1, b2, b3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n\t"
        "mov.b32 %0, {b0, b1, b2, b3};\n\t}"
        : "=r"(r)
        : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]));
    return r;
}

__device__ __forceinline__ float rcp_approx(float a) {
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}

// NVFP4's block scale (vLLM / modelopt): e4m3(gscale * amax / 6), and the
// factor its 16 values are multiplied by before rounding to e2m1.
__device__ __forceinline__ unsigned char nvfp4_scale(float amax, float gscale, float& out) {
    const float sf = gscale * (amax * rcp_approx(6.f));
    const unsigned short pair = e4m3x2(sf, 0.f);
    const unsigned char code = static_cast<unsigned char>(pair & 0xff);
    unsigned short wide = code;
    float back;
    asm("{\n\t.reg .b32 h;\n\t.reg .b16 lo, hi;\n\tcvt.rn.f16x2.e4m3x2 h, %1;\n\t"
        "mov.b32 {lo, hi}, h;\n\tcvt.f32.f16 %0, lo;\n\t}"
        : "=f"(back)
        : "h"(wide));
    out = back != 0.f ? rcp_approx(back * rcp_approx(gscale)) : 0.f;
    return code;
}

// The byte of scale (row, col) in an atom-layout tensor of `cols` columns.
__device__ __forceinline__ size_t atom_at(int row, int col, int cols) {
    const int atoms_k = (cols + 3) / 4;
    return static_cast<size_t>((row / 128) * atoms_k + col / 4) * 512 + (row % 32) * 16 +
           ((row / 32) % 4) * 4 + col % 4;
}

}


template <int kKind, int kFmtP, int kFmtQ, int kGranP, int kGranQ, int kBN, int kStages,
          int kSwap, int kPair, int kGroup, int kGlu, int kOut, int kMix>
__global__ __launch_bounds__(kMix != 0 ? 224 + 32 * tc5k::kXWarps : 224, 1) void tc5_gemm(
    const __grid_constant__ tc5::TmaMap map_p, const __grid_constant__ tc5::TmaMap map_q,
    const __grid_constant__ tc5::TmaMap map_y, const __grid_constant__ tc5::TmaMap map_sfp,
    const __grid_constant__ tc5::TmaMap map_sfq, bf16* __restrict__ y, int rows_p, int rows_q,
    int k, int ldy, float alpha, float* __restrict__ part, int splits,
    const int* __restrict__ group, int experts, int n_per, int m_max,
    const float* __restrict__ alpha_e, Epi epi) {
    using namespace tc5;
    static_assert(kBN % 32 == 0 && kBN >= 32 && kBN <= 256, "the Q tile is 32..256 rows");
    static_assert(kKind >= 0 && kKind <= 3, "four kinds");
    static_assert(kPair == 1 || (kPair == 2 && !kSwap && kBN >= 64), "a pair splits a wide Q");
    static_assert(kGroup == 0 ||
                      (((kGroup == 1) == !kSwap || (kGroup == 1 && (kMix != 0 || kKind == 2 || kKind == 3))) &&
                       (kGroup == 1 || kPair == 1)),
                  "contiguous groups run unswapped (a W4A16 or NVFP4 one swapped too), masked "
                  "swapped on lone CTAs");
    // A GLU reads gate and up rows; relu2 (6) the one value it activates.
    constexpr bool kFused = kGlu != 0 && kGlu != 6;
    static_assert(!kFused || kSwap || kBN == 256, "an unswapped GLU tile is gate 128 | up 128");
    static_assert(!kFused || !kSwap || kBN <= 128, "a swapped GLU tile holds two accumulators");
    static_assert(kMix == 0 || (kKind == 0 && kPair == 1), "W4A16 runs kind::f16 on lone CTAs");
    static_assert(kMix == 0 || !kSwap || kStages % 2 == 0,
                  "a swapped W4A16 weight's transform groups own alternate slots");
    static_assert(kOut == 0 || kGlu != 0, "a quantized store follows a GLU");

    // A GLU tile reads the weight's gate rows and the matching up rows `inter`
    // further on, side by side: unswapped as the two halves of the Q tile,
    // swapped as two P tiles into two accumulators.
    constexpr u32 kPT = kFused && kSwap ? 2 : 1;
    constexpr u32 kAccN = kBN * kPT;
    constexpr u32 kQStep = kFused && !kSwap ? 128 : kBN;  // output columns a Q tile
    const int inter = epi.inter;
    constexpr u32 kM = 128 * kPair;
    constexpr u32 kQRows = kBN / kPair;
    // A swapped block-scaled launch of one P tile (a decode streaming its
    // weight) reads two 128-byte boxes a tile a stage (512 K NVFP4, 256 K
    // mx), four MMAs a box: half the stages, and each weight row's two
    // boxes adjacent in memory.
    constexpr u32 kBoxes = kKind >= 2 && kSwap && kPT == 1 ? 2 : 1;
    constexpr u32 kStageK = tc5k::stage_k(kKind) * kBoxes;
    constexpr u32 kInstrK = tc5k::stage_k(kKind) / 4;
    constexpr u32 kInner = kKind == 0 ? 64 : 128;  // TMA box width, in the map's elements
    constexpr bool kScaled = kKind >= 2;
    constexpr u32 kBoxP = 128 * 128;
    constexpr u32 kBoxQ = kQRows * 128;
    constexpr u32 kTileP = kBoxP * kBoxes;
    constexpr u32 kTileQ = kBoxQ * kBoxes;
    // TMA counts the bytes it reads: an fp4 operand unpacked to a byte an
    // element (16U4_ALIGN16B) fills its tile from half as many.
    constexpr u32 kTxP = kKind == 2 && kFmtP == kE2M1 ? kTileP / 2 : kTileP;
    constexpr u32 kTxQ = kKind == 2 && kFmtQ == kE2M1 ? kTileQ / 2 : kTileQ;
    constexpr u32 kAtomsP = tc5k::atoms_k(kKind, kGranP) * kBoxes;
    constexpr u32 kAtomsQ = tc5k::atoms_k(kKind, kGranQ) * kBoxes;
    constexpr u32 kRowAtomsQ = (kBN + 127) / 128;
    // A contiguous P tile starts at either 64-row half: its scales span two
    // row blocks.
    constexpr u32 kRowAtomsP = kGroup == 1 && kScaled && !kSwap ? 2 : 1;
    constexpr u32 kSfP = kAtomsP * 512 * kPT * kRowAtomsP;  // this CTA's P rows (each P tile's)
    constexpr u32 kSfQ = kAtomsQ * kRowAtomsQ * 512;  // all of Q's rows, in each CTA
    // W4A16 (kMix 1 mxfp4: e2m1 under a ue8m0 a 32; 2 nvfp4: under an e4m3
    // a 16): the weight's codes land raw, 32 bytes a row a stage, and four
    // transform warps write them as bf16 into the tile the MMA reads.
    constexpr bool kMixed = kMix != 0;
    // Two groups of transform warps take turns a stage where the ring is
    // even (each group's slots its own); an odd ring (the Q side's widest
    // tiles) keeps one group of all eight.
    constexpr bool kMixP = kMixed && kSwap;
    constexpr bool kMixQ = kMixed && !kSwap;
    constexpr u32 kXGroups = kStages % 2 == 0 ? tc5k::kXGroupsMax : 1;
    constexpr u32 kRawRows = kMixP ? 128 * kPT : kMixQ ? kQRows : 0;
    constexpr u32 kRawBytes = kRawRows * 32;
    constexpr u32 kTx = kPair * ((kMixP ? kRawBytes : kTxP * kPT) + (kMixQ ? kRawBytes : kTxQ) +
                                 kSfP + kSfQ);
    constexpr u32 kSfCols = 4 * kAtomsP * kPT * kRowAtomsP + 4 * kAtomsQ * kRowAtomsQ;
    // Unswapped, the MMA thread copies each step's scales to TMEM itself,
    // into one slot: its tensor pipe runs them in order with the MMAs on
    // either side, so they wait on no barrier and leave TMEM to the
    // accumulators. Swapped, they go from their own warp, up to four K steps
    // ahead in their own slots.
    constexpr bool kSelfSf = kScaled && !kSwap;
    constexpr u32 kSfSlots = kSelfSf ? 1 : kStages < 4 / kBoxes ? kStages : 4 / kBoxes;
    // A swapped W4A16 weight lands in TMEM instead: an A slot a ring stage of
    // each P tile's 64 K, past the accumulators.
    constexpr u32 kASlots = kStages;
    constexpr u32 kSfAll = kScaled ? kSfSlots * kSfCols : kMixP ? kASlots * kPT * 32 : 0;
    static_assert(!kMixP || kASlots % kXGroups == 0, "W4A16's groups own alternate A slots");
    // Two accumulator stages; when they and the scale slots overrun TMEM the
    // stages share kOv columns, and a tile's first MMA waits on the previous
    // tile's epilogue to read them.
    constexpr u32 kOvRaw = 2 * kAccN + kSfAll > 512 ? 2 * kAccN + kSfAll - 512 : 0;
    constexpr u32 kOv = (kOvRaw + 15) / 16 * 16;
    constexpr u32 kOvChunks = (kOv + 31) / 32;
    constexpr u32 kStride = kAccN - kOv;
    constexpr u32 kCols = tc5k::pow2_cols(kStride + kAccN + kSfAll);
    static_assert(kCols <= 512, "TMEM is 512 columns");

    constexpr u32 kOffQ = kMixP ? 0 : kStages * kTileP * kPT;
    constexpr u32 kOffSfP = kOffQ + kStages * kTileQ;
    constexpr u32 kOffSfQ = kOffSfP + kStages * kSfP;
    // Each epilogue warp stages two 32 x 32 bf16 boxes for the TMA store.
    constexpr u32 kOffRaw = (kOffSfQ + kStages * kSfQ + 127) / 128 * 128;
    constexpr u32 kOffOut = (kOffRaw + kStages * kRawBytes + 1023) / 1024 * 1024;
    // A swapped GLU quantizing over the P rows reduces its amax through
    // shared memory instead.
    // A swapped NVFP4 store turns each warp's 32 x 32 values through a
    // padded tile instead.
    constexpr u32 kOutBytes = kSwap ? (kOut == 2 ? 4 * 32 * 33 * 4 : kOut != 0 ? 1024 : 0)
                                    : 4 * 2 * 2048;
    constexpr u32 kOffBar = kOffOut + kOutBytes;
    // A grouped launch holds the routing (offsets or counts, and a swapped
    // contiguous launch's live rows) in shared memory, read once a unit.
    // Its entries: the bank's experts and one where the launch says
    // (`route_cap`), else the most a launch routes.
    constexpr bool kLives = kGroup == 1 && kSwap;
    const u32 cap = epi.route_cap > 0 ? static_cast<u32>(epi.route_cap) : 1028u;
    const u32 group_ints = kGroup == 0 ? 0u : (kLives ? 3u : 2u) * cap;
    const u32 bytes = kOffBar + (3 * kStages + 16) * 8 + 8 + group_ints * 4;

    extern __shared__ unsigned char smem_raw[];
    const u32 raw = smem_addr(smem_raw);
    unsigned char* smem = smem_raw + (((raw + 1023u) & ~1023u) - raw);

    u64* full = reinterpret_cast<u64*>(smem + kOffBar);
    u64* empty = full + kStages;
    u64* xfull = empty + kStages;  // a W4A16 stage's weight tile written
    u64* tfull = xfull + kStages;
    u64* tempty = tfull + 2;
    u64* tover = tempty + 2;
    u64* sf_ready = tover + 2;
    u32* tmem_slot = reinterpret_cast<u32*>(sf_ready + 8);
    int* routes = reinterpret_cast<int*>(tmem_slot + 2);
    int* starts = routes + cap;
    int* lives = starts + cap;  // kLives only

    const u32 warp = threadIdx.x / 32;
    const u32 lane = threadIdx.x % 32;
    const u32 rank = kPair == 2 ? cluster_rank() : 0;
    const bool leader = rank == 0;

    if (threadIdx.x == 0) {
        u32 have;
        asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(have));
        // A launch's routing past its entries (more experts than it holds)
        // would overrun them, as one short of its shared memory would.
        if (have < bytes + 1024 || (kGroup != 0 && static_cast<u32>(experts) + 1 > cap)) {
            asm volatile("trap;");
        }
    }
    if (warp == 4 && lane == 0) {
        tma_prefetch(&map_p);
        tma_prefetch(&map_q);
        if constexpr (kScaled) {
            tma_prefetch(&map_sfp);
            tma_prefetch(&map_sfq);
        }
        for (u32 s = 0; s < kStages; ++s) {
            bar_init(&full[s], 1);
            bar_init(&empty[s], 1);
            bar_init(&xfull[s], tc5k::kXWarps / kXGroups);
        }
        for (u32 a = 0; a < 2; ++a) {
            bar_init(&tfull[a], 1);
            bar_init(&tempty[a], 4 * kPair);
            bar_init(&tover[a], 4 * kPair);
        }
        for (u32 a = 0; a < kSfSlots; ++a) {
            bar_init(&sf_ready[a], 1);
        }
        bar_init_fence();
    }
    if (warp == 5) {
        tmem_alloc<kPair>(tmem_slot, kCols);
        tmem_relinquish<kPair>();
    }
    // As a programmatic dependent (`kernel.pdl`) the set-up above overlapped
    // the producer's tail; wait for it before touching memory. With the
    // weights ahead the group is read first, the weight loader never waits,
    // and every other role waits below.
    const bool ahead = kSwap && epi.weights_ahead != 0;
    if (!ahead) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    // A masked launch counts each expert's tiles up front, so units map onto
    // the busy experts alone. An expert's output columns: its bank rows, or a
    // GLU's gate rows.
    const int span = kFused ? inter : n_per;
    const int masked_p = (span + 127) / 128;
    const int cont_q = (span + static_cast<int>(kQStep) - 1) / static_cast<int>(kQStep);
    if (kGroup != 0 && warp == 0) {
        for (int i = static_cast<int>(lane); i < experts + (kGroup == 1 ? 1 : 0); i += 32) {
            routes[i] = __ldg(&group[i]);
        }
        __syncwarp();
        if constexpr (kLives) {
            // An expert's rows, but for its block's padding past the live ones.
            for (int i = static_cast<int>(lane); i < experts; i += 32) {
                const int rows = routes[i + 1] - routes[i];
                lives[i] = epi.live != nullptr ? min(rows, __ldg(&epi.live[i])) : rows;
            }
        }
        {
            __syncwarp();
            const int per = (experts + 31) / 32;
            const int lo = min(experts, static_cast<int>(lane) * per);
            const int hi = min(experts, lo + per);
            // An expert's tiles: its rows' 128-row tiles by the Q tiles.
            auto tiles_of = [&](int e) {
                if constexpr (kGroup == 1 && kSwap) {
                    // Swapped (W4A16, a decode's NVFP4): its live rows'
                    // kBN-row Q tiles by its weight's 128-row P tiles.
                    return (lives[e] + kBN - 1) / kBN * masked_p;
                } else if constexpr (kGroup == 1) {
                    return (routes[e + 1] - routes[e] + kM - 1) / kM * cont_q;
                } else {
                    return (min(routes[e], m_max) + kBN - 1) / kBN * masked_p;
                }
            };
            int sum = 0;
            for (int i = lo; i < hi; ++i) {
                sum += tiles_of(i);
            }
            int incl = sum;
            for (int d = 1; d < 32; d <<= 1) {
                const int up = __shfl_up_sync(0xffffffffu, incl, d);
                incl += lane >= static_cast<u32>(d) ? up : 0;
            }
            int run = incl - sum;
            for (int i = lo; i < hi; ++i) {
                starts[i] = run;
                run += tiles_of(i);
            }
            if (lane == 31) {
                starts[experts] = incl;
            }
        }
    }
    tc_fence_before();
    if constexpr (kPair == 2) {
        cluster_sync();
    } else {
        __syncthreads();
    }
    tc_fence_after();
    const u32 tmem = *tmem_slot;
    // Every CTA is resident and holds its TMEM, so the dependent may launch
    // now (not earlier: its CTA could take an SM's TMEM first).
    asm volatile("griddepcontrol.launch_dependents;");
    if (ahead && !(warp == 4 && lane == 0)) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    // Tiles along P and Q (of one expert, when grouped), and the experts.
    // Contiguous P tiles stop at the routed rows.
    const int num_p = kGroup == 2   ? masked_p
                      : kGroup == 1 ? (routes[experts] + static_cast<int>(kM) - 1) / static_cast<int>(kM)
                                    : (rows_p + static_cast<int>(kM) - 1) / static_cast<int>(kM);
    const int num_q = kGroup == 0 ? (rows_q + static_cast<int>(kQStep) - 1) / static_cast<int>(kQStep)
                      : kGroup == 1 ? (span + static_cast<int>(kQStep) - 1) / static_cast<int>(kQStep)
                                    : (m_max + kBN - 1) / kBN;
    const int num_kb = (k + static_cast<int>(kStageK) - 1) / static_cast<int>(kStageK);
    // A unit is one tile's share of K: `splits` of them a tile when a
    // decode's few tiles would leave SMs idle, summed in f32 at `part`.
    const int kb_per = (num_kb + splits - 1) / splits;
    const int tiles = (kGroup != 0 ? starts[experts] : num_p * num_q) * splits;
    int first_tile = static_cast<int>(blockIdx.x) / kPair;
    int tile_step = static_cast<int>(gridDim.x) / kPair;
    // A decode's routed units, few and equal, would leave the last wave part
    // full: from `even_waves` waves the fewest CTAs that keep the waves take
    // the same count each, and the rest retire (their SMs free for what
    // follows).
    if (kGroup != 0 && kSwap && epi.even_waves > 0 && tiles > 0) {
        const int waves = (tiles + tile_step - 1) / tile_step;
        const int even = (tiles + waves - 1) / waves;
        if (waves >= epi.even_waves && even < tile_step) {
            tile_step = even;
            if (first_tile >= even) {
                first_tile = tiles;
            }
        }
    }

    // Where a unit reads and lands. D's tile-local [i][j] lands at y[out_p
    // + i][out_q + j] unswapped, y[out_q + j][out_p + i] swapped, for i <
    // lim_p and j < lim_q.
    struct Unit {
        int p_row, q_row, p_block, q_block;
        u32 quarter, pquarter;
        int kb0, kb1;
        float alpha;
        int out_p, out_q, lim_p, lim_q, s_row;
        int e;     // the expert (grouped), else 0
        int ks;  // which of its tile's K splits
    };
    // Every role walks the same units and skips the same empty ones (a
    // contiguous tile past the routed rows, a masked tile past its count).
    auto unit_of = [&](int u, Unit& x) -> bool {
        const int t = u / splits;
        x.ks = u - t * splits;
        const int ks = u - t * splits;
        x.kb0 = ks * kb_per;
        x.kb1 = min(num_kb, x.kb0 + kb_per);
        x.alpha = alpha;
        int e = 0, pt, qt;
        x.pquarter = 0;
        x.e = 0;
        if constexpr (kGroup != 0) {
            // The last expert whose tiles start at or before t: an empty
            // expert shares its start with the next.
            int lo = 0, hi = experts - 1;
            while (lo < hi) {
                const int mid = (lo + hi + 1) / 2;
                if (starts[mid] <= t) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            e = lo;
            x.e = e;
            const int r = t - starts[e];
            if constexpr (kGroup == 1 && kSwap) {
                // An expert's row tiles walk fastest, sharing its weight tile.
                const int q_tiles = (lives[e] + kBN - 1) / kBN;
                pt = r / q_tiles;
                qt = r - pt * q_tiles;
            } else if constexpr (kGroup == 1) {
                // Bands of an expert's row tiles walk its weight tiles together,
                // so a wave shares both operands (as a dense launch's).
                const int m_tiles = (routes[e + 1] - routes[e] + static_cast<int>(kM) - 1) / static_cast<int>(kM);
                const int kRaster = epi.raster;
                const int band = r / (kRaster * num_q);
                const int first = band * kRaster;
                const int size = min(m_tiles - first, kRaster);
                const int rr = r - band * kRaster * num_q;
                pt = first + rr % size;
                qt = rr / size;
            } else {
                const int q_tiles = (min(routes[e], m_max) + kBN - 1) / kBN;
                pt = r / q_tiles;
                qt = r - pt * q_tiles;
            }
        } else {
            // Groups of 8 P tiles walk Q together, so a wave shares both
            // operands.
            const int kRaster = epi.raster;
            const int band = t / (kRaster * num_q);
            const int first = band * kRaster;
            const int size = min(num_p - first, kRaster);
            const int r = t - band * kRaster * num_q;
            pt = first + r % size;
            qt = r / size;
        }
        if constexpr (kGroup == 0) {
            x.p_row = pt * static_cast<int>(kM);
            x.q_row = qt * static_cast<int>(kQStep);
            x.p_block = pt * kPair;
            x.out_p = x.p_row;
            x.out_q = x.q_row;
            x.lim_p = rows_p - x.p_row;
            x.lim_q = rows_q - x.q_row;
        } else if constexpr (kGroup == 1 && kSwap) {
            // Swapped (W4A16): the expert's weight rows on P, its routed rows
            // from its offset on Q.
            x.p_row = e * n_per + pt * 128;
            x.q_row = routes[e] + qt * kBN;
            x.p_block = x.p_row / 128;
            x.out_p = pt * 128;
            x.out_q = x.q_row;
            x.lim_p = span - pt * 128;
            // Past the expert's live rows its block is padding.
            x.lim_q = lives[e] - qt * kBN;
        } else if constexpr (kGroup == 1) {
            // A tile starts at its expert's rows, either 64-row half (the
            // MMA faults on A-side scales at an odd quarter): its
            // rows past the expert's are another's, and land nowhere.
            if (routes[e] % 64 != 0) {
                asm volatile("trap;");
            }
            x.p_row = routes[e] + pt * static_cast<int>(kM);
            x.q_row = e * n_per + qt * static_cast<int>(kQStep);
            x.p_block = x.p_row / 128;
            x.pquarter = static_cast<u32>(x.p_row % 128) / 32;
            x.out_p = x.p_row;
            x.out_q = qt * static_cast<int>(kQStep);
            x.lim_p = min(static_cast<int>(kM), routes[e + 1] - x.p_row);
            // Past the expert's live rows its block is padding: those rows
            // land nowhere, and a warp wholly past them stores nothing.
            if (epi.live != nullptr) {
                x.lim_p = min(x.lim_p, __ldg(&epi.live[e]) - pt * static_cast<int>(kM));
            }
            x.lim_q = span - qt * static_cast<int>(kQStep);
        } else {
            const int count = min(routes[e], m_max);
            if (qt * kBN >= count) {
                return false;
            }
            x.p_row = e * n_per + pt * 128;
            x.q_row = e * m_max + qt * kBN;
            x.p_block = x.p_row / 128;
            x.out_p = pt * 128;
            x.out_q = x.q_row;
            x.lim_p = span - pt * 128;
            x.lim_q = count - qt * kBN;
        }
        // A Q tile narrower than its 128-row scale atom reads the atom's
        // columns from its own 32-row quarter on; masked rows' scales are
        // padded to whole row blocks an expert.
        const int q_scale_row =
            kGroup == 2 ? e * ((m_max + 127) / 128) * 128 + qt * kBN : x.q_row;
        x.q_block = q_scale_row / 128;
        x.quarter = static_cast<u32>(q_scale_row % 128) / 32;
        // The scale row a quantized store gives y's first row of this tile.
        x.s_row = kSwap ? (kGroup == 2 ? e * epi.pad + qt * kBN : x.out_q) : x.out_p;
        if (kGroup != 0 && alpha_e != nullptr) {
            x.alpha *= __ldg(&alpha_e[e]);
        }
        return true;
    };

    // Every arrival a pair counts lands on rank 0's barrier.
    auto arrive_leader = [&](u64* bar) {
        if constexpr (kPair == 2) {
            bar_arrive_at(bar, 0);
        } else {
            bar_arrive(bar);
        }
    };
    auto commit = [&](u64* bar) {
        if constexpr (kPair == 2) {
            tc_commit_pair(bar);
        } else {
            tc_commit(bar);
        }
    };

    if (warp == 4) {
        if (lane == 0) {
            u32 s = 0, phase = 0;
            for (int t = first_tile; t < tiles; t += tile_step) {
                Unit x;
                if (!unit_of(t, x)) {
                    continue;
                }
                const int p0 = x.p_row + static_cast<int>(rank) * 128;
                const int q0 = x.q_row + static_cast<int>(rank * kQRows);
                for (int kb = x.kb0; kb < x.kb1; ++kb) {
                    bar_wait(&empty[s], phase ^ 1);
                    if (leader) {
                        bar_arrive_expect_tx(&full[s], kTx);
                    }
                    unsigned char* tile_p = smem + s * kTileP * kPT;
                    unsigned char* tile_q = smem + kOffQ + s * kTileQ;
                    // Each of the stage's boxes, `stride` bytes apart.
                    auto load2 = [&](void* dst, const TmaMap* map, int row, u32 stride) {
#pragma unroll
                        for (u32 b = 0; b < kBoxes; ++b) {
                            unsigned char* at = static_cast<unsigned char*>(dst) + b * stride;
                            const int x = (kb * static_cast<int>(kBoxes) + static_cast<int>(b)) *
                                          static_cast<int>(kInner);
                            if constexpr (kPair == 2) {
                                tma_load_2d_pair(at, &full[s], map, x, row);
                            } else {
                                tma_load_2d(at, &full[s], map, x, row);
                            }
                        }
                    };
                    auto load3 = [&](void* dst, const TmaMap* map, int atom, int block) {
                        if constexpr (kPair == 2) {
                            tma_load_3d_pair(dst, &full[s], map, 0, atom, block);
                        } else {
                            tma_load_3d(dst, &full[s], map, 0, atom, block);
                        }
                    };
                    // A GLU's second P tile, or the up half of its Q tile,
                    // reads the rows `inter` past the gate's.
                    // A W4A16 weight's codes land raw, 32 bytes a row.
                    unsigned char* raw_s = smem + kOffRaw + s * kRawBytes;
                    auto load_raw = [&](void* dst, const TmaMap* map, int row) {
                        tma_load_2d(dst, &full[s], map, kb * 32, row);
                    };
                    for (u32 h = 0; h < kPT; ++h) {
                        if constexpr (kMixP) {
                            load_raw(raw_s + h * 128 * 32, &map_p, p0 + static_cast<int>(h) * inter);
                        } else {
                            load2(tile_p + h * kTileP, &map_p, p0 + static_cast<int>(h) * inter, kBoxP);
                        }
                    }
                    if (ahead) {
                        // The activations: lane 1's.
                    } else if constexpr (kMixQ && kFused) {
                        load_raw(raw_s, &map_q, x.q_row);
                        load_raw(raw_s + 128 * 32, &map_q, x.q_row + inter);
                    } else if constexpr (kMixQ) {
                        load_raw(raw_s, &map_q, q0);
                    } else if constexpr (kFused && !kSwap && kPair == 1) {
                        load2(tile_q, &map_q, x.q_row, kBoxQ);
                        load2(tile_q + 128 * 128, &map_q, x.q_row + inter, kBoxQ);
                    } else if constexpr (kFused && !kSwap) {
                        load2(tile_q, &map_q, x.q_row + static_cast<int>(rank) * inter, kBoxQ);
                    } else {
                        load2(tile_q, &map_q, q0, kBoxQ);
                    }
                    if constexpr (kScaled) {
                        const int atom_p = static_cast<int>(kb * kStageK / kGranP / 4);
                        const int atom_q = static_cast<int>(kb * kStageK / kGranQ / 4);
                        unsigned char* sf_p = smem + kOffSfP + s * kSfP;
                        unsigned char* sf_q = smem + kOffSfQ + s * kSfQ;
                        for (u32 h = 0; h < kPT; ++h) {
                            load3(sf_p + h * kAtomsP * 512, &map_sfp, atom_p,
                                  x.p_block + static_cast<int>(rank) +
                                      static_cast<int>(h) * inter / 128);
                        }
                        if (ahead) {
                        } else if constexpr (kFused && !kSwap) {
                            for (int h = 0; h < 2; ++h) {
                                load3(sf_q + h * kAtomsQ * 512, &map_sfq, atom_q,
                                      (x.q_row + h * inter) / 128);
                            }
                        } else {
                            load3(sf_q, &map_sfq, atom_q, x.q_block);
                        }
                    }
                    if (++s == kStages) {
                        s = 0;
                        phase ^= 1;
                    }
                }
            }
        } else if (lane == 1 && ahead) {
            // The weights ahead, the activation rows and their scales load
            // here once the producer kernel is done (waited above), each into
            // its stage once free, under the stage's one expect_tx (lane 0's):
            // so the weight loader runs a ring ahead and they never queue
            // behind it. Swapped launches are lone CTAs.
            u32 s = 0, phase = 0;
            for (int t = first_tile; t < tiles; t += tile_step) {
                Unit x;
                if (!unit_of(t, x)) {
                    continue;
                }
                for (int kb = x.kb0; kb < x.kb1; ++kb) {
                    bar_wait(&empty[s], phase ^ 1);
#pragma unroll
                    for (u32 b = 0; b < kBoxes; ++b) {
                        tma_load_2d(smem + kOffQ + s * kTileQ + b * kBoxQ, &full[s], &map_q,
                                    (kb * static_cast<int>(kBoxes) + static_cast<int>(b)) *
                                        static_cast<int>(kInner),
                                    x.q_row);
                    }
                    if constexpr (kScaled) {
                        tma_load_3d(smem + kOffSfQ + s * kSfQ, &full[s], &map_sfq, 0,
                                    static_cast<int>(kb * kStageK / kGranQ / 4), x.q_block);
                    }
                    if (++s == kStages) {
                        s = 0;
                        phase ^= 1;
                    }
                }
            }
        }
    } else if (warp == 6) {
        // The swapped scale copier: each K step's scales into slot (step % kSfSlots) once
        // the MMAs kSfSlots steps back have read it. Shared memory holds
        // [row block][atom]; TMEM takes [atom][row block]. The warp runs the
        // loop together and one lane issues, as in the MMA warp.
        if (kScaled && !kSelfSf && leader) {
            u32 s = 0, phase = 0, step = 0;
            for (int t = first_tile; t < tiles; t += tile_step) {
                Unit x;
                if (!unit_of(t, x)) {
                    continue;
                }
                for (int kb = x.kb0; kb < x.kb1; ++kb, ++step) {
                    const u32 slot = step % kSfSlots;
                    // A slot is free once the MMAs of the step that last read it
                    // are done: the commit that empties that step's stage says so.
                    if (step >= kSfSlots) {
                        const u32 last = step - kSfSlots;
                        bar_wait(&empty[last % kStages], (last / kStages) & 1);
                    }
                    bar_wait(&full[s], phase);
                    tc_fence_after();
                    __syncwarp();
                    if (elect_one()) {
                        const u32 sf_tp = tmem + kStride + kAccN + slot * kSfCols;
                        const u32 sf_tq = sf_tp + 4 * kAtomsP * kPT * kRowAtomsP;
                        const u32 sf_p = smem_addr(smem + kOffSfP + s * kSfP);
                        const u32 sf_q = smem_addr(smem + kOffSfQ + s * kSfQ);
                        // P's scales as Q's: [row block][atom] in, [atom][row block]
                        // out, each P tile's after the last's.
                        for (u32 h = 0; h < kPT; ++h) {
                            for (u32 r = 0; r < kRowAtomsP; ++r) {
                                for (u32 a = 0; a < kAtomsP; ++a) {
                                    const u32 base = h * kAtomsP * kRowAtomsP;
                                    tc_cp_sf_atom<kPair>(
                                        sf_tp + 4 * (base + a * kRowAtomsP + r),
                                        sdesc_sf(sf_p + (base + r * kAtomsP + a) * 512));
                                }
                            }
                        }
                        for (u32 r = 0; r < kRowAtomsQ; ++r) {
                            for (u32 a = 0; a < kAtomsQ; ++a) {
                                tc_cp_sf_atom<kPair>(sf_tq + 4 * (a * kRowAtomsQ + r),
                                                     sdesc_sf(sf_q + (r * kAtomsQ + a) * 512));
                            }
                        }
                        tc_commit<kPair>(&sf_ready[slot]);
                    }
                    if (++s == kStages) {
                        s = 0;
                        phase ^= 1;
                    }
                }
            }
        }
    } else if (warp == 5) {
        // The MMA warp: all of it walks the steps, the elected lane issues.
        if (leader) {
            u32 s = 0, phase = 0, step = 0;
            int it = 0;
            for (int t = first_tile; t < tiles; t += tile_step) {
                Unit x;
                if (!unit_of(t, x)) {
                    continue;
                }
                const int kb0 = x.kb0;
                const u32 acc = static_cast<u32>(it) % 2;
                bar_wait(&tempty[acc], ((static_cast<u32>(it) / 2) & 1) ^ 1);
                tc_fence_after();
                const u32 d0 = tmem + acc * kStride;
                const u32 quarter_q = x.quarter;
                for (int kb = kb0; kb < x.kb1; ++kb, ++step) {
                    // A W4A16 stage is ready once its weight tile is written.
                    bar_wait(kMixed ? &xfull[s] : &full[s], phase);
                    const u32 slot = step % kSfSlots;
                    if constexpr (kScaled && !kSelfSf) {
                        bar_wait(&sf_ready[slot], (step / kSfSlots) & 1);
                    }
                    tc_fence_after();
                    const u32 tile_p = smem_addr(smem + s * kTileP * kPT);
                    const u32 tile_q = smem_addr(smem + kOffQ + s * kTileQ);
                    const u32 sf_tp0 = tmem + kStride + kAccN + slot * kSfCols;
                    const u32 sf_tq = sf_tp0 + 4 * kAtomsP * kPT * kRowAtomsP;
                    if (kOv > 0 && kb == kb0 && it > 0) {
                        const u32 prev = static_cast<u32>(it - 1);
                        bar_wait(&tover[prev % 2], (prev / 2) & 1);
                        tc_fence_after();
                    }
                    __syncwarp();
                    if (elect_one()) {
                        if constexpr (kSelfSf) {
                            // P's scales as Q's: [row block][atom] in, [atom][row
                            // block] out, each P tile's after the last's.
                            const u32 sf_p = smem_addr(smem + kOffSfP + s * kSfP);
                            const u32 sf_q = smem_addr(smem + kOffSfQ + s * kSfQ);
                            for (u32 h = 0; h < kPT; ++h) {
                                for (u32 r = 0; r < kRowAtomsP; ++r) {
                                    for (u32 a = 0; a < kAtomsP; ++a) {
                                        const u32 base = h * kAtomsP * kRowAtomsP;
                                        tc_cp_sf_atom<kPair>(
                                            sf_tp0 + 4 * (base + a * kRowAtomsP + r),
                                            sdesc_sf(sf_p + (base + r * kAtomsP + a) * 512));
                                    }
                                }
                            }
                            for (u32 r = 0; r < kRowAtomsQ; ++r) {
                                for (u32 a = 0; a < kAtomsQ; ++a) {
                                    tc_cp_sf_atom<kPair>(sf_tq + 4 * (a * kRowAtomsQ + r),
                                                         sdesc_sf(sf_q + (r * kAtomsQ + a) * 512));
                                }
                            }
                        }
                        const u64 desc_q = sdesc_sw128(tile_q);
                        // MMA j reads box j / 4 of each tile.
                        constexpr u64 kBoxDescP = kBoxP >> 4, kBoxDescQ = kBoxQ >> 4;
                        // A W4A16 A side's slot: this step's, in TMEM.
                        const u32 a_slot = tmem + kStride + kAccN + s * kPT * 32;
#pragma unroll
                        for (u32 h = 0; h < kPT; ++h) {
                        const u64 desc_p = sdesc_sw128(tile_p + h * kTileP);
                        const u32 d = d0 + h * kBN;
                        const u32 sf_tp = sf_tp0 + 4 * h * kAtomsP * kRowAtomsP;
#pragma unroll
                        for (u32 j = 0; j < 4 * kBoxes; ++j) {
                            const u64 a = desc_p + (j / 4) * kBoxDescP + 2 * (j % 4);
                            const u64 b = desc_q + (j / 4) * kBoxDescQ + 2 * (j % 4);
                            const bool accumulate = kb > kb0 || j > 0;
                            if constexpr (kMixP) {
                                mma_f16_ts(d, a_slot + h * 32 + j * 8, b,
                                           idesc_dense(kF16Bf16, kF16Bf16, kM, kBN), accumulate);
                            } else if constexpr (kKind == 0) {
                                mma_f16<kPair>(d, a, b, idesc_dense(kF16Bf16, kF16Bf16, kM, kBN),
                                               accumulate);
                            } else if constexpr (kKind == 1) {
                                mma_f8f6f4<kPair>(d, a, b, idesc_dense(kFmtP, kFmtQ, kM, kBN),
                                                  accumulate);
                            } else {
                                const u32 k0 = static_cast<u32>(kb) * kStageK + j * kInstrK;
                                const u32 cp = k0 / kGranP;
                                const u32 cq = k0 / kGranQ;
                                const u32 col_p =
                                    sf_tp +
                                    4 * kRowAtomsP *
                                        (cp / 4 - static_cast<u32>(kb) * kStageK / kGranP / 4) +
                                    x.pquarter;
                                const u32 col_q =
                                    sf_tq +
                                    4 * kRowAtomsQ *
                                        (cq / 4 - static_cast<u32>(kb) * kStageK / kGranQ / 4) +
                                    quarter_q;
                                const u32 desc = idesc_with_ids(
                                    idesc_scaled(kFmtP, kFmtQ, kM, kBN, kKind == 2), cp % 4, cq % 4);
                                if constexpr (kKind == 2) {
                                    mma_mxf8f6f4<kPair>(d, a, b, desc, accumulate, col_p, col_q);
                                } else {
                                    mma_nvf4<kPair>(d, a, b, desc, accumulate, col_p, col_q);
                                }
                            }
                        }
                        }
                        commit(&empty[s]);
                    }
                    if (++s == kStages) {
                        s = 0;
                        phase ^= 1;
                    }
                }
                __syncwarp();
                if (elect_one()) {
                    commit(&tfull[acc]);
                }
                ++it;
            }
            // Both CTAs' epilogues have read the last accumulator, so every
            // commit has landed before either CTA retires its barriers.
            if (it > 0) {
                const u32 last = static_cast<u32>(it - 1);
                bar_wait(&tempty[last % 2], (last / 2) & 1);
            }
        }
    } else if (kMixed && warp >= 7) {
        // The transform warps: each stage's raw codes as bf16 in the swizzled tile
        // TMA would have landed, a thread one 16-byte chunk of every 16th row, its
        // scales row-major from global memory. Two groups of warps take turns a
        // stage, each waiting on its own slots' barriers in order.
        constexpr u32 kGroupWarps = tc5k::kXWarps / kXGroups;
        const u32 xgroup = (warp - 7) / kGroupWarps;
        const u32 t = (threadIdx.x - 224) % (32 * kGroupWarps);
        const u32 c = t % 8;
        u32 step = 0;
        // The weight's factor group: its scales' gran (mxfp4 32, nvfp4 16,
        // affine 32, 64 or 128).
        constexpr int kWGran = kSwap ? kGranP : kGranQ;
        constexpr bool kAffine = kMix == 3;
        // The weight's rows: a dense GLU's extent is its output columns,
        // its up rows past them; a bank's extent is all of its rows.
        const int weight_rows = (kMixP ? rows_p : rows_q) * (kFused && kGroup == 0 ? 2 : 1);
        const int sf_cols = k / kWGran;
        u32 s = 0, phase = 0;
        for (int u = first_tile; u < tiles; u += tile_step) {
            Unit x;
            if (!unit_of(u, x)) {
                continue;
            }
            // A raw row's weight row: a GLU's up rows `inter` past its gate's.
            auto wrow = [&](u32 r) -> int {
                const int ri = static_cast<int>(r);
                if constexpr (kMixP) {
                    return x.p_row + ri / 128 * inter + ri % 128;
                } else if constexpr (kFused) {
                    return x.q_row + (ri >= 128 ? inter + ri - 128 : ri);
                } else {
                    return x.q_row + ri;
                }
            };
            constexpr u32 kRowStep = 4 * kGroupWarps;
            constexpr u32 kIters = kRawRows / kRowStep > 0 ? kRawRows / kRowStep : 1;
            // A stage's scales load before its wait, by volatile loads the
            // compiler keeps where they stand (a plain load would sink
            // past the barrier wait to its use, an L2 trip a row).
            auto load_sb = [&](int kb, unsigned (&sb)[kIters]) {
                const int col = (kb * 64 + static_cast<int>(c) * 8) / kWGran;
#pragma unroll
                for (u32 i = 0; i < kIters; ++i) {
                    const int g = wrow(t / 8 + kRowStep * i);
                    if constexpr (kAffine) {
                        const size_t at = static_cast<size_t>(g) * sf_cols + col;
                        const unsigned char* zeros = reinterpret_cast<const unsigned char*>(epi.wzero);
                        sb[i] = g < weight_rows
                                    ? tc5k::ld_nc_u16(epi.wsf + 2 * at) |
                                          (tc5k::ld_nc_u16(zeros + 2 * at) << 16)
                                    : 0u;
                    } else {
                        sb[i] = g < weight_rows
                                    ? tc5k::ld_nc_u8(epi.wsf + static_cast<size_t>(g) * sf_cols + col)
                                    : 0u;
                    }
                }
            };
            if constexpr (kMixP) {
                // A thread owns a row of each P tile: its lane quarter's
                // row of TMEM, 64 K a stage as 32 bf16 pairs.
                const u32 r = (warp % 4) * 32 + lane;
                constexpr int kSfStage = 64 / kWGran;  // scales a row a stage (not affine)
                // An affine row's groups a stage: two of 32, else one.
                constexpr int kGroups = kWGran < 64 ? 2 : 1;
                for (int kb = x.kb0; kb < x.kb1; ++kb, ++step) {
                    if (step % kXGroups != xgroup) {
                        if (++s == kStages) {
                            s = 0;
                            phase ^= 1;
                        }
                        continue;
                    }
                    unsigned sc[kPT];
                    unsigned az[kPT][kGroups];  // affine: scale | zero << 16 a group
#pragma unroll
                    for (u32 h = 0; h < kPT; ++h) {
                        const int g = wrow(h * 128 + r);
                        if constexpr (kAffine) {
                            sc[h] = 0;
                            const unsigned char* zeros =
                                reinterpret_cast<const unsigned char*>(epi.wzero);
#pragma unroll
                            for (int q = 0; q < kGroups; ++q) {
                                const size_t at = static_cast<size_t>(g) * sf_cols +
                                                  (kb * 64 + q * 32) / kWGran;
                                az[h][q] = g >= weight_rows
                                               ? 0u
                                               : tc5k::ld_nc_u16(epi.wsf + 2 * at) |
                                                     (tc5k::ld_nc_u16(zeros + 2 * at) << 16);
                            }
                        } else {
                            const unsigned char* at =
                                epi.wsf + static_cast<size_t>(g) * sf_cols + kb * kSfStage;
                            sc[h] = g >= weight_rows ? 0u
                                    : kSfStage == 2  ? tc5k::ld_nc_u16(at)
                                                     : tc5k::ld_nc_u32(at);
                        }
                    }
                    bar_wait(&full[s], phase);
                    tc_fence_after();
                    const unsigned char* raw_s = smem + kOffRaw + s * kRawBytes;
                    // Its stage's slot: free once `full` says the producer saw
                    // the stage emptied by the MMAs that read it last.
                    const u32 a_slot = tmem + kStride + kAccN + s * kPT * 32 +
                                       (((warp % 4) * 32) << 16);
#pragma unroll
                    for (u32 h = 0; h < kPT; ++h) {
                        const uint4* row = reinterpret_cast<const uint4*>(raw_s + (h * 128 + r) * 32);
                        const uint4 c0 = row[0], c1 = row[1];
                        const unsigned code[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
                        u32 v[32];
#pragma unroll
                        for (int w = 0; w < 8; ++w) {
                            uint4 d;
                            if constexpr (kAffine) {
                                const unsigned f = az[h][kGroups == 2 ? w / 4 : 0];
                                d = tc5k::dequant8_affine(code[w], tc5k::bf16_bits_f32(f & 0xffffu),
                                                          tc5k::bf16_bits_f32(f >> 16));
                            } else {
                                const unsigned char b = static_cast<unsigned char>(
                                    sc[h] >> (8 * (w * 8 / kWGran)));
                                d = tc5k::dequant8<kMix>(code[w], b);
                            }
                            v[4 * w] = d.x;
                            v[4 * w + 1] = d.y;
                            v[4 * w + 2] = d.z;
                            v[4 * w + 3] = d.w;
                        }
                        tmem_st32(a_slot + h * 32, v);
                    }
                    tmem_wait_st();
                    tc_fence_before();
                    __syncwarp();
                    if (lane == 0) {
                        bar_arrive(&xfull[s]);
                    }
                    if (++s == kStages) {
                        s = 0;
                        phase ^= 1;
                    }
                }
                continue;
            }
            for (int kb = x.kb0; kb < x.kb1; ++kb, ++step) {
                if (step % kXGroups != xgroup) {
                    if (++s == kStages) {
                        s = 0;
                        phase ^= 1;
                    }
                    continue;
                }
                unsigned sb[kIters];
                load_sb(kb, sb);
                bar_wait(&full[s], phase);
                const unsigned char* raw_s = smem + kOffRaw + s * kRawBytes;
                unsigned char* tile = kMixP ? smem + s * kTileP * kPT : smem + kOffQ + s * kTileQ;
                // Every code first: a store to the tile may not pass a load
                // from the raw slot the compiler cannot tell apart.
                unsigned code[kIters];
#pragma unroll
                for (u32 i = 0; i < kIters; ++i) {
                    code[i] = *reinterpret_cast<const unsigned*>(raw_s + (t / 8 + kRowStep * i) * 32 + c * 4);
                }
#pragma unroll
                for (u32 i = 0; i < kIters; ++i) {
                    const u32 r = t / 8 + kRowStep * i;
                    uint4 d;
                    if constexpr (kAffine) {
                        d = tc5k::dequant8_affine(code[i], tc5k::bf16_bits_f32(sb[i] & 0xffffu),
                                                  tc5k::bf16_bits_f32(sb[i] >> 16));
                    } else {
                        d = tc5k::dequant8<kMix>(code[i], static_cast<unsigned char>(sb[i]));
                    }
                    *reinterpret_cast<uint4*>(tile + r * 128 + ((c ^ (r % 8)) * 16)) = d;
                }
                fence_async_shared();
                __syncwarp();
                if (lane == 0) {
                    bar_arrive(&xfull[s]);
                }
                if (++s == kStages) {
                    s = 0;
                    phase ^= 1;
                }
            }
        }
    } else {
        int it = 0;
        const u32 row = warp * 32 + lane;
        u32 out_buf = 0;
        for (int t = first_tile; t < tiles; t += tile_step) {
            Unit x;
            if (!unit_of(t, x)) {
                continue;
            }
            const u32 acc = static_cast<u32>(it) % 2;
            // An unswapped warp wholly past the tile's rows (a decode
            // expert's few routes fill one warp's) reads nothing: it only
            // releases the accumulator.
            const bool dead = !kSwap && static_cast<int>(rank * 128 + warp * 32) >= x.lim_p;
            bar_wait(&tfull[acc], (static_cast<u32>(it) / 2) & 1);
            tc_fence_after();
            ++it;
            const float alpha = x.alpha;
            // This thread's tile-local P row, and where it lands.
            const int i = static_cast<int>(rank * 128 + row);
            const bool live = i < x.lim_p;
            const int p = x.out_p + i;
            const u32 taddr = tmem + acc * kStride + ((warp * 32) << 16);
            // A GLU reads its gate columns and, kUp further on, its up ones.
            constexpr u32 kChunks = (kFused ? (kSwap ? kBN : 128) : kBN) / 32;
            constexpr u32 kUp = kSwap ? kBN : 128;
            const float rs_row =
                !kSwap && epi.row_scale != nullptr && live ? epi.row_scale[p] : 1.f;
            // This tile's 32 values a chunk, past alpha, the GLU and the
            // routed weight.
            // A chunk's raw columns (a GLU's gate and up), then its values.
            auto fetch = [&](u32 c0, u32 (&v)[32], u32 (&u)[32]) {
                tmem_ld32(taddr + c0, v);
                if constexpr (kFused) {
                    tmem_ld32(taddr + kUp + c0, u);
                }
            };
            // A bias each weight row (gate, and up `inter` on), added before
            // the GLU: a swapped thread's row holds one, an unswapped chunk a
            // column each. Of a tile's K splits, the first adds it.
            const bool biased = epi.bias != nullptr && x.kb0 == 0;  // one split adds it
            const float bias_row = kSwap && biased && live ? bf16_to_f32(epi.bias[x.p_row + i]) : 0.f;
            const float bias_row_up =
                kSwap && kFused && biased && live ? bf16_to_f32(epi.bias[x.p_row + inter + i]) : 0.f;
            // An unswapped chunk's column factors, a lane each column: the
            // chunk's values take them by shuffle (one load a lane, not 32
            // a thread).
            struct Cols {
                float ws, ws_up, b, b_up;
            };
            auto cols_of = [&](u32 c0) {
                Cols f{1.f, 1.f, 0.f, 0.f};
                if constexpr (!kSwap) {
                    const int c = static_cast<int>(c0 + lane);
                    if (c < x.lim_q) {
                        const int at = x.q_row + c;
                        if (epi.wscale != nullptr) {
                            f.ws = epi.wscale[at];
                            if constexpr (kFused) {
                                f.ws_up = epi.wscale[at + inter];
                            }
                        }
                        if (epi.bias != nullptr && x.kb0 == 0) {
                            f.b = bf16_to_f32(epi.bias[at]);
                            if constexpr (kFused) {
                                f.b_up = bf16_to_f32(epi.bias[at + inter]);
                            }
                        }
                    }
                }
                return f;
            };
            auto bias_col = [&](const Cols& f, int j, bool up) -> float {
                if (!biased) {
                    return 0.f;
                }
                if constexpr (kSwap) {
                    return up ? bias_row_up : bias_row;
                } else {
                    return __shfl_sync(0xffffffffu, up ? f.b_up : f.b, j);
                }
            };
            // An f32 scale each weight row (NVFP4's tensor scale), on the
            // product before the bias.
            const bool wscaled = epi.wscale != nullptr;
            const float ws_row = kSwap && wscaled && live ? epi.wscale[x.p_row + i] : 1.f;
            const float ws_row_up =
                kSwap && kFused && wscaled && live ? epi.wscale[x.p_row + inter + i] : 1.f;
            auto ws_col = [&](const Cols& f, int j, bool up) -> float {
                if (!wscaled) {
                    return 1.f;
                }
                if constexpr (kSwap) {
                    return up ? ws_row_up : ws_row;
                } else {
                    return __shfl_sync(0xffffffffu, up ? f.ws_up : f.ws, j);
                }
            };
            // No factor past alpha (a W4A4 select's store): its chunk
            // takes the plain path, not the per-column guards below (the
            // same sums: a zero bias's add turns -0 to +0 there too).
            const bool plain = !wscaled && !biased && epi.row_scale == nullptr;
            // A SwiGLU landing an NVFP4 operand under each expert's global takes gate
            // and up rounded to bf16, as apart; its quotient stays the fast one.
            auto glu_of = [&](float g, float u) -> float {
                if constexpr (kOut == 2 && kGlu == 1) {
                    if (epi.in_scale != nullptr) {
                        g = bf16_to_f32(f32_to_bf16(g));
                        u = bf16_to_f32(f32_to_bf16(u));
                        return __fdividef(g, 1.f + __expf(-g)) * u;
                    }
                }
                return tc5k::glu<kGlu>(g, u, epi.limit, epi.galpha);
            };
            // An ungated activation (relu2) of a value, or the value itself;
            // landing an NVFP4 operand under each expert's global, it takes the
            // value rounded to bf16, as apart.
            auto act = [&](float g) -> float {
                if constexpr (kGlu == 0) {
                    return g;
                } else {
                    if constexpr (kOut == 2) {
                        if (epi.in_scale != nullptr) {
                            g = bf16_to_f32(f32_to_bf16(g));
                        }
                    }
                    return tc5k::glu<kGlu>(g, 0.f, 0.f, 0.f);
                }
            };
            auto finish = [&](u32 c0, const Cols& f, const u32 (&v)[32], const u32 (&u)[32],
                              float (&h)[32]) {
                if (plain) {
#pragma unroll
                    for (int j = 0; j < 32; ++j) {
                        if constexpr (kFused) {
                            h[j] = glu_of(__uint_as_float(v[j]) * alpha + 0.f,
                                          __uint_as_float(u[j]) * alpha + 0.f);
                        } else {
                            h[j] = act(__uint_as_float(v[j]) * alpha + 0.f);
                        }
                    }
                    return;
                }
                if constexpr (kFused) {
#pragma unroll
                    for (int j = 0; j < 32; ++j) {
                        h[j] = glu_of(
                            __uint_as_float(v[j]) * alpha * ws_col(f, j, false) +
                                bias_col(f, j, false),
                            __uint_as_float(u[j]) * alpha * ws_col(f, j, true) +
                                bias_col(f, j, true));
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < 32; ++j) {
                        h[j] = act(__uint_as_float(v[j]) * alpha * ws_col(f, j, false) +
                                   bias_col(f, j, false));
                    }
                }
#pragma unroll
                for (int j = 0; j < 32; ++j) {
                    float rs = rs_row;
                    if constexpr (kSwap) {
                        const int c = static_cast<int>(c0) + j;
                        rs = epi.row_scale != nullptr && c < x.lim_q
                                 ? epi.row_scale[x.out_q + c]
                                 : 1.f;
                    }
                    h[j] *= rs;
                }
            };
            auto product = [&](u32 c0, float (&h)[32]) {
                // A swapped chunk wholly past the tile's rows lands nothing.
                if (dead || (kSwap && static_cast<int>(c0) >= x.lim_q)) {
#pragma unroll
                    for (int j = 0; j < 32; ++j) {
                        h[j] = 0.f;
                    }
                    return;
                }
                // The factors' loads go out before the accumulator's read.
                const Cols f = plain ? Cols{1.f, 1.f, 0.f, 0.f} : cols_of(c0);
                u32 v[32], u[32];
                fetch(c0, v, u);
                tmem_wait_ld();
                finish(c0, f, v, u, h);
            };
            // The columns the next tile's MMA shares drain first; a GLU reads them all
            // before any of its math, so the MMA is released early.
            const bool reverse = kOv > 0 && acc == 0;
            auto col = [&](u32 chunk) {
                return reverse ? 32 * (kChunks - 1 - chunk) : 32 * chunk;
            };
            constexpr bool kEarly = kFused && kOv > 0;
            constexpr u32 kPre = kEarly ? kOvChunks : 1;
            u32 pre_v[kPre][32], pre_u[kPre][32];
            if constexpr (kEarly) {
#pragma unroll
                for (u32 c = 0; c < kPre; ++c) {
                    fetch(col(c), pre_v[c], pre_u[c]);
                }
                tmem_wait_ld();
                tc_fence_before();
                __syncwarp();
                if (lane == 0) {
                    arrive_leader(&tover[acc]);
                }
            }
            auto value = [&](u32 chunk, float (&h)[32]) {
                if (kEarly && chunk < kPre) {
                    finish(col(chunk), plain ? Cols{1.f, 1.f, 0.f, 0.f} : cols_of(col(chunk)),
                           pre_v[chunk], pre_u[chunk], h);
                } else {
                    product(col(chunk), h);
                }
                if (!kEarly && kOv > 0 && chunk + 1 == kOvChunks) {
                    tc_fence_before();
                    __syncwarp();
                    if (lane == 0) {
                        arrive_leader(&tover[acc]);
                    }
                }
                if (chunk + 1 == kChunks) {
                    tc_fence_before();
                    __syncwarp();
                    if (lane == 0) {
                        arrive_leader(&tempty[acc]);
                    }
                }
            };
            // DeepSeek's 1x128 block of an unswapped row is the tile's four
            // chunks: held in registers, its amax known once they are read.
            if constexpr (kOut == 1 && !kSwap) {
                float hall[kChunks][32];
                float amax = 0.f;
#pragma unroll
                for (u32 chunk = 0; chunk < kChunks; ++chunk) {
                    value(chunk, hall[chunk]);
#pragma unroll
                    for (int j = 0; j < 32; ++j) {
                        amax = fmaxf(amax, fabsf(hall[chunk][j]));
                    }
                }
                if (live && x.lim_q > 0) {
                    const int ce = tc5k::ue8m0_exponent(amax);
                    // satfinite clamps to +-448.
                    const float down = tc5k::pow2_neg(ce);
#pragma unroll
                    for (u32 chunk = 0; chunk < kChunks; ++chunk) {
                        const float(&h)[32] = hall[chunk];
                        u32 w[8];
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            float e[4];
#pragma unroll
                            for (int b = 0; b < 4; ++b) {
                                e[b] = h[4 * j + b] * down;
                            }
                            w[j] = static_cast<u32>(tc5k::e4m3x2(e[0], e[1])) |
                                   (static_cast<u32>(tc5k::e4m3x2(e[2], e[3])) << 16);
                        }
                        uint4* out = reinterpret_cast<uint4*>(
                            epi.q + static_cast<size_t>(p) * inter + x.out_q + col(chunk));
                        out[0] = make_uint4(w[0], w[1], w[2], w[3]);
                        out[1] = make_uint4(w[4], w[5], w[6], w[7]);
                    }
                    epi.qs[tc5k::atom_at(x.s_row + i, x.out_q / 128, inter / 128)] =
                        static_cast<unsigned char>(ce + 127);
                }
                continue;
            }
#pragma unroll(kEarly ? kChunks : 1)
            for (u32 chunk = 0; chunk < kChunks; ++chunk) {
                const u32 c0 = col(chunk);
                float h[32];
                value(chunk, h);
                const int j0 = static_cast<int>(c0);
                const int q0 = x.out_q + j0;
                if constexpr (kSwap && kOut == 1) {
                    // A token's 1x128 block is this tile's 128 P rows: its
                    // amax reduces across the four warps.
                    float* red = reinterpret_cast<float*>(smem + kOffOut);
                    float mine = 0.f;
#pragma unroll
                    for (int j = 0; j < 32; ++j) {
                        float m = live ? fabsf(h[j]) : 0.f;
#pragma unroll
                        for (int d = 16; d > 0; d >>= 1) {
                            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, d));
                        }
                        mine = lane == static_cast<u32>(j) ? m : mine;
                    }
                    asm volatile("bar.sync 1, 128;" ::: "memory");
                    red[warp * 32 + lane] = mine;
                    asm volatile("bar.sync 1, 128;" ::: "memory");
#pragma unroll
                    for (int j = 0; j < 32; ++j) {
                        const float amax = fmaxf(fmaxf(red[j], red[32 + j]),
                                                 fmaxf(red[64 + j], red[96 + j]));
                        const int ce = tc5k::ue8m0_exponent(amax);
                        if (j0 + j < x.lim_q) {
                            const float q = h[j] * tc5k::pow2_neg(ce);
                            if (live) {
                                epi.q[static_cast<size_t>(q0 + j) * inter + p] =
                                    static_cast<unsigned char>(tc5k::e4m3x2(q, 0.f) & 0xff);
                            }
                            if (row == 0) {
                                epi.qs[tc5k::atom_at(x.s_row + j0 + j, x.out_p / 128,
                                                     inter / 128)] =
                                    static_cast<unsigned char>(ce + 127);
                            }
                        }
                    }
                } else if constexpr (kSwap && kOut == 2) {
                    // A token's 16-wide NVFP4 block is 16 lanes' rows: the
                    // warp's 32 x 32 values turn through its padded tile, a
                    // lane then a (token, block) of them, landing eight bytes
                    // and a scale. Under an input scale an expert, the tile's
                    // expert's global, each value rounded to bf16. A dead row
                    // is a zero its block's amax sees and no byte it lands.
                    float gs = epi.gscale;
                    if (epi.in_scale != nullptr) {
                        gs = __fdiv_rn(1.f, __ldg(&epi.in_scale[x.e]));
#pragma unroll
                        for (int j = 0; j < 32; ++j) {
                            h[j] = bf16_to_f32(f32_to_bf16(h[j]));
                        }
                    }
                    float* tile = reinterpret_cast<float*>(smem + kOffOut) + warp * 32 * 33;
                    __syncwarp();
#pragma unroll
                    for (int j = 0; j < 32; ++j) {
                        tile[j * 33 + lane] = live ? h[j] : 0.f;
                    }
                    __syncwarp();
                    const int t = static_cast<int>(lane);
                    const int first = static_cast<int>(rank * 128 + warp * 32);
                    if (j0 + t < x.lim_q) {
                        const size_t r = static_cast<size_t>(q0 + t);
#pragma unroll
                        for (int b = 0; b < 2; ++b) {
                            float v[16];
                            float m = 0.f;
#pragma unroll
                            for (int k = 0; k < 16; ++k) {
                                v[k] = tile[t * 33 + 16 * b + k];
                                m = fmaxf(m, fabsf(v[k]));
                            }
                            float scale;
                            const unsigned char code = tc5k::nvfp4_scale(m, gs, scale);
                            const int row0 = first + 16 * b;
                            const int pb = x.out_p + row0;
                            float e[2][8];
#pragma unroll
                            for (int k = 0; k < 16; ++k) {
                                e[k / 8][k % 8] = v[k] * scale;
                            }
                            const u32 lo = tc5k::e2m1x8(e[0]), hi = tc5k::e2m1x8(e[1]);
                            unsigned char* dst = epi.q + r * (inter / 2) + pb / 2;
                            if (row0 + 16 <= x.lim_p) {
                                *reinterpret_cast<uint2*>(dst) = make_uint2(lo, hi);
                            } else {
#pragma unroll
                                for (int k = 0; k < 8; ++k) {
                                    if (row0 + 2 * k < x.lim_p) {
                                        dst[k] = static_cast<unsigned char>(
                                            (k < 4 ? lo : hi) >> (8 * (k % 4)));
                                    }
                                }
                            }
                            if (row0 < x.lim_p) {
                                epi.qs[tc5k::atom_at(x.s_row + j0 + t, pb / 16, inter / 16)] =
                                    code;
                            }
                        }
                    }
                } else if constexpr (kSwap) {
                    // A warp's lanes are 32 consecutive weight rows: each
                    // column is one coalesced 64-byte run of y.
                    if (live && splits > 1) {
#pragma unroll
                        for (int j = 0; j < 32; ++j) {
                            if (j0 + j < x.lim_q) {
                                part[static_cast<size_t>(x.ks) * epi.slab +
                                     static_cast<size_t>(q0 + j) * ldy + p] = h[j];
                            }
                        }
                    } else if (live) {
#pragma unroll
                        for (int j = 0; j < 32; ++j) {
                            if (j0 + j < x.lim_q) {
                                y[static_cast<size_t>(q0 + j) * ldy + p] = f32_to_bf16(h[j]);
                            }
                        }
                    }
                } else if constexpr (kOut == 2) {
                    // A row's two 16-wide blocks: 16 bytes and two scales.
                    if (live && j0 < x.lim_q) {
                        float gs = epi.gscale;
                        if (epi.in_scale != nullptr) {
                            gs = __fdiv_rn(1.f, __ldg(&epi.in_scale[x.e]));
#pragma unroll
                            for (int j = 0; j < 32; ++j) {
                                h[j] = bf16_to_f32(f32_to_bf16(h[j]));
                            }
                        }
                        u32 w[4];
#pragma unroll
                        for (int b = 0; b < 2; ++b) {
                            float m = 0.f;
#pragma unroll
                            for (int j = 0; j < 16; ++j) {
                                m = fmaxf(m, fabsf(h[16 * b + j]));
                            }
                            float scale;
                            const unsigned char code = tc5k::nvfp4_scale(m, gs, scale);
#pragma unroll
                            for (int g = 0; g < 2; ++g) {
                                float e[8];
#pragma unroll
                                for (int j = 0; j < 8; ++j) {
                                    e[j] = h[16 * b + 8 * g + j] * scale;
                                }
                                w[2 * b + g] = tc5k::e2m1x8(e);
                            }
                            epi.qs[tc5k::atom_at(x.s_row + i, q0 / 16 + b, inter / 16)] = code;
                        }
                        *reinterpret_cast<uint4*>(epi.q + static_cast<size_t>(p) * (inter / 2) +
                                                  q0 / 2) = make_uint4(w[0], w[1], w[2], w[3]);
                    }
                } else if (splits > 1) {
                    // A thread's 32 sums of its row, four at a time.
                    if (live) {
                        float* at = part + static_cast<size_t>(x.ks) * epi.slab +
                                    static_cast<size_t>(p) * ldy + q0;
#pragma unroll
                        for (int j = 0; j < 32; j += 4) {
                            if (j0 + j < x.lim_q) {
                                *reinterpret_cast<float4*>(at + j) =
                                    make_float4(h[j], h[j + 1], h[j + 2], h[j + 3]);
                            }
                        }
                    }
                } else if (static_cast<int>(rank * 128 + warp * 32) < x.lim_p) {
                    // The warp's 32 x 32 box goes out by TMA from a 64-byte
                    // swizzled stage, which also drops rows and columns past y
                    // (a contiguous tile's warps past its expert land nothing).
                    unsigned char* box = smem + kOffOut + (warp * 2 + out_buf) * 2048;
                    if (lane == 0) {
                        bulk_wait_read<1>();
                    }
                    __syncwarp();
#pragma unroll
                    for (u32 j = 0; j < 4; ++j) {
                        uint4 w;
                        w.x = pack_bf16x2(h[8 * j + 0], h[8 * j + 1]);
                        w.y = pack_bf16x2(h[8 * j + 2], h[8 * j + 3]);
                        w.z = pack_bf16x2(h[8 * j + 4], h[8 * j + 5]);
                        w.w = pack_bf16x2(h[8 * j + 6], h[8 * j + 7]);
                        *reinterpret_cast<uint4*>(box + lane * 64 + 16 * (j ^ ((lane >> 1) & 3))) =
                            w;
                    }
                    fence_async_shared();
                    __syncwarp();
                    if (lane == 0) {
                        tma_store_2d(&map_y, box, q0,
                                     x.out_p + static_cast<int>(rank * 128 + warp * 32));
                        bulk_commit();
                    }
                    out_buf ^= 1;
                }
            }
            // A split tile's last split sums every split's slab in split order and
            // rounds y, as the finalize did; each thread sums what it stored, and the
            // count is left at zero for the next launch.
            if constexpr (kOut == 0 && !kFused) {
                if (splits > 1 && epi.tickets != nullptr) {
                    __threadfence();
                    const u32 at_ticket =
                        static_cast<u32>(t / splits) * kPair + static_cast<u32>(rank);
                    u32 mine = 0;
                    asm volatile("bar.sync 2, 128;" ::: "memory");
                    if (row == 0) {
                        const u32 seen = atomicAdd(&epi.tickets[at_ticket], 1u);
                        mine = seen + 1 == static_cast<u32>(splits) ? 1u : 0u;
                        if (mine != 0) {
                            epi.tickets[at_ticket] = 0;
                        }
                    }
                    u32 last;
                    asm volatile(
                        "{\n\t.reg .pred q, l;\n\t"
                        "setp.ne.u32 q, %1, 0;\n\t"
                        "bar.red.or.pred l, 2, 128, q;\n\t"
                        "selp.u32 %0, 1, 0, l;\n\t}"
                        : "=r"(last)
                        : "r"(mine)
                        : "memory");
                    if (last != 0 && live) {
                        __threadfence();
                        const int lim = min(x.lim_q, static_cast<int>(kBN));
                        const size_t slab = static_cast<size_t>(epi.slab);
                        // Four of the thread's values at a time, each split's
                        // four loaded before any is added (the adds stay in
                        // split order): loads in flight, not one at a time.
                        if constexpr (kSwap) {
                            for (int j0 = 0; j0 < lim; j0 += 4) {
                                size_t at[4];
                                float v[4];
#pragma unroll
                                for (int u = 0; u < 4; ++u) {
                                    const int j = min(j0 + u, lim - 1);
                                    at[u] = static_cast<size_t>(x.out_q + j) * ldy + p;
                                    v[u] = __ldcg(part + at[u]);
                                }
#pragma unroll 4
                                for (int sp = 1; sp < splits; ++sp) {
                                    float w[4];
#pragma unroll
                                    for (int u = 0; u < 4; ++u) {
                                        w[u] = __ldcg(part + static_cast<size_t>(sp) * slab + at[u]);
                                    }
#pragma unroll
                                    for (int u = 0; u < 4; ++u) v[u] += w[u];
                                }
#pragma unroll
                                for (int u = 0; u < 4; ++u) {
                                    if (j0 + u < lim) y[at[u]] = f32_to_bf16(v[u]);
                                }
                            }
                        } else {
                            for (int j0 = 0; j0 < lim; j0 += 8) {
                                size_t at[2];
                                float4 v[2];
#pragma unroll
                                for (int u = 0; u < 2; ++u) {
                                    const int j = min(j0 + 4 * u, (lim - 1) / 4 * 4);
                                    at[u] = static_cast<size_t>(p) * ldy + x.out_q + j;
                                    v[u] = __ldcg(reinterpret_cast<const float4*>(part + at[u]));
                                }
#pragma unroll 4
                                for (int sp = 1; sp < splits; ++sp) {
                                    float4 w[2];
#pragma unroll
                                    for (int u = 0; u < 2; ++u) {
                                        w[u] = __ldcg(reinterpret_cast<const float4*>(
                                            part + static_cast<size_t>(sp) * slab + at[u]));
                                    }
#pragma unroll
                                    for (int u = 0; u < 2; ++u) {
                                        v[u].x += w[u].x;
                                        v[u].y += w[u].y;
                                        v[u].z += w[u].z;
                                        v[u].w += w[u].w;
                                    }
                                }
#pragma unroll
                                for (int u = 0; u < 2; ++u) {
                                    if (j0 + 4 * u < lim) {
                                        *reinterpret_cast<uint2*>(y + at[u]) = make_uint2(
                                            pack_bf16x2(v[u].x, v[u].y), pack_bf16x2(v[u].z, v[u].w));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if (!kSwap && lane == 0) {
            bulk_wait_all();
        }
    }

    tc_fence_before();
    if constexpr (kPair == 2) {
        cluster_sync();
    } else {
        __syncthreads();
    }
    if (warp == 5) {
        tc_fence_after();
        tmem_dealloc<kPair>(tmem, kCols);
    }
}

// The split-K sums in f32, four at a time, to bf16: each split's slab
// added in split order.
__global__ __launch_bounds__(256) void tc5_finalize(const float4* __restrict__ part,
                                                    uint2* __restrict__ y, int quads, int splits,
                                                    long long slab_quads) {
    // Launched as the GEMM's programmatic dependent (`kernel.pdl`): its sums
    // are read only once that grid has finished and flushed them. Without
    // the attribute the wait returns at once.
    asm volatile("griddepcontrol.wait;" ::: "memory");
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < quads) {
        // Each split's slab in split order: the same sum every launch.
        float4 v = part[i];
        for (int s = 1; s < splits; ++s) {
            const float4 w = part[static_cast<long long>(s) * slab_quads + i];
            v.x += w.x;
            v.y += w.y;
            v.z += w.z;
            v.w += w.w;
        }
        y[i] = make_uint2(pack_bf16x2(v.x, v.y), pack_bf16x2(v.z, v.w));
    }
}

// As tc5_finalize for a grouped launch's routed rows alone (below
// group[experts] when contiguous, under each count when masked; the rest
// were never summed into): a block a row of `n`.
__global__ __launch_bounds__(128) void tc5_finalize_routed(const float4* __restrict__ part,
                                                           uint2* __restrict__ y,
                                                           const int* __restrict__ counts,
                                                           int experts, int m_max, int n,
                                                           int splits, long long slab_quads) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    const int row = static_cast<int>(blockIdx.x);
    if (m_max == 0 ? row >= __ldg(&counts[experts]) : row % m_max >= __ldg(&counts[row / m_max])) {
        return;
    }
    const size_t base = static_cast<size_t>(row) * (n / 4);
    for (int i = static_cast<int>(threadIdx.x); i < n / 4; i += blockDim.x) {
        float4 v = part[base + i];
        for (int s = 1; s < splits; ++s) {
            const float4 w = part[static_cast<long long>(s) * slab_quads + base + i];
            v.x += w.x;
            v.y += w.y;
            v.z += w.z;
            v.w += w.w;
        }
        y[base + i] = make_uint2(pack_bf16x2(v.x, v.y), pack_bf16x2(v.z, v.w));
    }
}

}
