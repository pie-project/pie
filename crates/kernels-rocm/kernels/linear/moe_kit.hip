#pragma once

// The expert matmuls every MoE bank shares, dense (moe.hip) or packed
// (quant.hip): y[route] = act[token(route)] · W[expert(route)]ᵀ. A bank
// differs only in how a run of its codes becomes bf16 / f16 operands, which
// a codec spells; the two engines here do the rest.
//
// The stream (decode: each expert sees a route or two) reads an expert's rows
// as one flat run of 16-byte units, 64 lanes side by side, so every wave
// instruction reads a kilobyte of contiguous weight whatever K is. Each unit
// is dotted with its slice of the activation (staged in LDS) on the 4 x 4 x 4
// matrix core, the lane's own dot taken from the block diagonal; the dots
// gather in LDS and fold into rows at the end.
//
// The grouped engine (prefill: an expert sees many routes) takes a work item
// of up to 32, 64 or 128 of one expert's routes, gathers their activation
// rows into LDS and multiplies them against 128 or 256 weight rows a block on
// the 16 x 16 x 16 matrix core, the weight decoded from registers straight
// into the operand.
//
// The batch engine (decode batches: an expert sees 1 to 32 routes) reads an
// expert's weight once for all its routes, streaming it into registers
// several stages ahead by hand, and multiplies 4 to 32 routes at a time.
//
// All rely on the matrix core reading its k positions in any order so long
// as both operands agree: a codec decodes codes in whatever pairing is
// cheapest, and the activation is staged in that same order.

#include "linear/mgemm.hip"

namespace pie::linear {

typedef short moe_s4 __attribute__((ext_vector_type(4)));
typedef _Float16 moe_h4 __attribute__((ext_vector_type(4)));
typedef unsigned int moe_u2 __attribute__((ext_vector_type(2)));
typedef unsigned int moe_u4 __attribute__((ext_vector_type(4)));

template <int V>
struct MoeConst {
    static constexpr int value = V;
};

// D += A·B over four k, a lane's four elements of A and of B as two dwords.
// On the 4 x 4 x 4 core (16 blocks a wave) block b's row i is lane 4b+i's A
// and its column j lane 4b+j's B, so D[i][i] (lane 4b+i, register i) is the
// lane's own dot of four.
template <class T>
__device__ __forceinline__ mfma_cd mma4x4(u32 a0, u32 a1, u32 b0, u32 b1, mfma_cd c) {
    const moe_u2 a{a0, a1}, b{b0, b1};
    if constexpr (is_same<T, f16>::value) {
        return __builtin_amdgcn_mfma_f32_4x4x4f16(__builtin_bit_cast(moe_h4, a), __builtin_bit_cast(moe_h4, b), c, 0,
                                                  0, 0);
    } else {
        return __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(__builtin_bit_cast(moe_s4, a), __builtin_bit_cast(moe_s4, b), c,
                                                      0, 0, 0);
    }
}

// The 16 x 16 x 16 step (prelude/mfma.hip has the lane layout) in T.
template <class T>
__device__ __forceinline__ mfma_cd mma16x16(u32 a0, u32 a1, u32 b0, u32 b1, mfma_cd c) {
    const moe_u2 a{a0, a1}, b{b0, b1};
    if constexpr (is_same<T, f16>::value) {
        return __builtin_amdgcn_mfma_f32_16x16x16f16(__builtin_bit_cast(moe_h4, a), __builtin_bit_cast(moe_h4, b), c,
                                                     0, 0, 0);
    } else {
        return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(__builtin_bit_cast(moe_s4, a), __builtin_bit_cast(moe_s4, b),
                                                         c, 0, 0, 0);
    }
}

// The lane's own dot of WORDS dwords (2·WORDS elements) of `w` against as
// many of `x`.
template <class T, int WORDS>
__device__ __forceinline__ float own_dot(const u32* w, const u32* x) {
    mfma_cd d = mfma_zero();
#pragma unroll
    for (int i = 0; i < WORDS; i += 2) d = mma4x4<T>(x[i], x[i + 1], w[i], w[i + 1], d);
    const int i = lane() & 3;
    return i == 0 ? d[0] : i == 1 ? d[1] : i == 2 ? d[2] : d[3];
}

template <class T>
__device__ __forceinline__ u32 pack2(float a, float b) {
    const T v[2] = {from_f32<T>(a), from_f32<T>(b)};
    return __builtin_bit_cast(u32, v);
}

// ---------------------------------------------------------------- mxfp4 --

// An e8m0 block scale: 2^(b - 127), and 2^-127 for b = 0.
__device__ __forceinline__ float mxfp4_scale(u32 b) {
    return __builtin_bit_cast(float, b == 0 ? 0x00400000u : b << 23);
}

// The eight fp4 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6} as T, split into a
// table of high bytes and one of low bytes (two dwords each), so a code's
// three magnitude bits select its two bytes with one byte permute each.
struct Fp4Tables {
    u32 h0, h1, l0, l1;
};

template <class T>
__device__ __forceinline__ Fp4Tables fp4_tables() {
    if constexpr (is_same<T, f16>::value) {
        return {0x3E3C3800u, 0x46444240u, 0u, 0u};
    } else {
        return {0x3F3F3F00u, 0x40404040u, 0xC0800000u, 0xC0804000u};
    }
}

// The same magnitudes in bf16 already scaled by 2^(b - 127): a scaled fp4
// value is exact in bf16, so folding the scale into the table changes no
// product. The exponent clamps to [1, 252], past which the scaled magnitudes
// leave bf16's range.
__device__ __forceinline__ Fp4Tables fp4_tables_scaled(u32 b) {
    const u32 e = min(max(b, 1u), 252u) << 7;
    const u32 both = e | (e << 16);
    // Entries (m, m+1) as 16-bit halves: m = 0 is zero, m = 1 one exponent
    // down, the rest one mantissa bit (64) or exponent (128) apart.
    const u32 p01 = (e - 128u) << 16;
    const u32 p23 = both + (64u << 16);
    const u32 p45 = both + (128u | (192u << 16));
    const u32 p67 = both + (256u | (320u << 16));
    return {__builtin_amdgcn_perm(p23, p01, 0x07050301u), __builtin_amdgcn_perm(p67, p45, 0x07050301u),
            __builtin_amdgcn_perm(p23, p01, 0x06040200u), __builtin_amdgcn_perm(p67, p45, 0x06040200u)};
}

// Eight fp4 codes (code j in bits 4j..4j+3, its sign the top bit) as four
// dwords of pairs, ordered (c0,c2) (c4,c6) (c1,c3) (c5,c7).
__device__ __forceinline__ void fp4_word(u32 w, const Fp4Tables& t, u32* d) {
    const u32 se = w & 0x07070707u, so = (w >> 4) & 0x07070707u;
    const u32 he = __builtin_amdgcn_perm(t.h1, t.h0, se) | ((w << 4) & 0x80808080u);
    const u32 ho = __builtin_amdgcn_perm(t.h1, t.h0, so) | (w & 0x80808080u);
    const u32 le = __builtin_amdgcn_perm(t.l1, t.l0, se), lo = __builtin_amdgcn_perm(t.l1, t.l0, so);
    d[0] = __builtin_amdgcn_perm(he, le, 0x05010400u);
    d[1] = __builtin_amdgcn_perm(he, le, 0x07030602u);
    d[2] = __builtin_amdgcn_perm(ho, lo, 0x05010400u);
    d[3] = __builtin_amdgcn_perm(ho, lo, 0x07030602u);
}

// Eight activation elements (four dwords of pairs) into fp4_word's order.
__device__ __forceinline__ b128 fp4_shuffle(b128 x) {
    return {{__builtin_amdgcn_perm(x.w[1], x.w[0], 0x05040100u), __builtin_amdgcn_perm(x.w[3], x.w[2], 0x05040100u),
             __builtin_amdgcn_perm(x.w[1], x.w[0], 0x07060302u), __builtin_amdgcn_perm(x.w[3], x.w[2], 0x07060302u)}};
}

// The pointers a packed bank's seat names when its experts stream from a
// group buffer instead of the resident planes.
struct alignas(16) MoeGroupBases {
    const u8* codes;
    const u8* scales;
    const u8* biases;
    const u8* pad;
};

// ------------------------------------------------------------ stream ----

constexpr int kStreamWaves = 4;
constexpr int kStreamUnroll = 8;

// A codec for the stream names its unit (kCodes codes in one fetch), the
// planes of one expert it reads, how a unit's dot comes out of the matrix
// core, and how eight natural activation elements land in its order
// (`stage`). kXsum: the value wants the sum of the unit's activation slice.
//
// The staged activation is chunk-major: 16-byte chunk q of unit g sits at
// chunk q·G + g (G units a row), so the 64 lanes of a wave, each on its own
// unit, read 64 neighbouring chunks and no two share an LDS bank.

// Where staged chunk q of unit g starts, in elements.
__device__ __forceinline__ int chunk_at(int g, int q, int G) { return 8 * (q * G + g); }

// A unit's activation slice, chunk by chunk, as dwords of pairs.
template <int WORDS>
__device__ __forceinline__ void slice_words(const u32* x, int stride, u32* w) {
#pragma unroll
    for (int q = 0; q < WORDS / 4; ++q) {
        const b128 c = *reinterpret_cast<const b128*>(x + q * stride);
#pragma unroll
        for (int i = 0; i < 4; ++i) w[4 * q + i] = c.w[i];
    }
}

// Two dwords of pairs (a0,a1) (b0,b1) as (a0,b0) and (a1,b1).
__device__ __forceinline__ u32 pair_lo(u32 a, u32 b) { return __builtin_amdgcn_perm(b, a, 0x05040100u); }
__device__ __forceinline__ u32 pair_hi(u32 a, u32 b) { return __builtin_amdgcn_perm(b, a, 0x07060302u); }

// A wave's run of a plane as a buffer resource: reads past its `bytes`
// land zeros and touch no memory, so a round's fetches need no clamp, and
// their offsets are 32-bit.
__device__ __forceinline__ __amdgpu_buffer_rsrc_t moe_run(const void* base, u32 bytes) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(base), 0, static_cast<int>(bytes), 0x00020000);
}

// Buffer-load cache policy: nt, a streamed read no later read finds.
constexpr int kStreamNt = 2;

// A dense T bank: a unit is eight elements, in their own order.
template <class T>
struct StreamDense {
    static constexpr int kCodes = 8;
    static constexpr int kUnroll = kStreamUnroll;
    static constexpr bool kXsum = false;
    struct Planes {
        const T* w;
    };
    struct Run {
        __amdgpu_buffer_rsrc_t w;
    };
    struct Unit {
        moe_u4 w;
    };
    static __device__ __forceinline__ void stage(b128 x, int e0, T* xs, int) {
        *reinterpret_cast<b128*>(xs + e0) = x;
    }
    static __device__ __forceinline__ Run run(const Planes& p, long long first, int units) {
        return {moe_run(p.w + first * 8, units * 16)};
    }
    static __device__ __forceinline__ Unit take(const Run& r, int u) {
        return {__builtin_amdgcn_raw_buffer_load_b128(r.w, u * 16, 0, kStreamNt)};
    }
    static __device__ __forceinline__ float value(const Unit& u, const u32* x, int, float) {
        const u32 w[4] = {u.w[0], u.w[1], u.w[2], u.w[3]};
        return own_dot<T, 4>(w, x);
    }
};

// An mxfp4 bank: 32 fp4 codes (16 bytes) and the block's e8m0 scale. The
// dot runs on the unscaled magnitudes and takes the scale once.
template <class T>
struct StreamMxfp4 {
    static constexpr int kCodes = 32;
    static constexpr int kUnroll = kStreamUnroll / 2;
    static constexpr bool kXsum = false;
    struct Planes {
        const u8* codes;
        const u8* scales;
    };
    struct Run {
        __amdgpu_buffer_rsrc_t codes, scales;
    };
    struct Unit {
        moe_u4 w;
        u32 scale;
    };
    static __device__ __forceinline__ void stage(b128 x, int e0, T* xs, int G) {
        *reinterpret_cast<b128*>(xs + chunk_at(e0 / 32, e0 % 32 / 8, G)) = fp4_shuffle(x);
    }
    static __device__ __forceinline__ Run run(const Planes& p, long long first, int units) {
        return {moe_run(p.codes + first * 16, units * 16), moe_run(p.scales + first, units)};
    }
    static __device__ __forceinline__ Unit take(const Run& r, int u) {
        return {__builtin_amdgcn_raw_buffer_load_b128(r.codes, u * 16, 0, kStreamNt),
                __builtin_amdgcn_raw_buffer_load_b8(r.scales, u, 0, kStreamNt)};
    }
    static __device__ __forceinline__ float value(const Unit& u, const u32* x, int stride, float) {
        const Fp4Tables t = fp4_tables<T>();
        u32 d[16], xw[16];
#pragma unroll
        for (int q = 0; q < 4; ++q) fp4_word(u.w[q], t, d + 4 * q);
        slice_words<16>(x, stride, xw);
        return own_dot<T, 16>(d, xw) * mxfp4_scale(u.scale);
    }
};

// An affine bank (MLX): BITS-bit codes in little-endian u32 words, value
// code·scale + bias with a bf16 scale and bias per GROUP codes. A unit is 32
// codes (BITS words), inside one group. The codes enter the matrix core as
// M + code, exact in T (M = 128 in bf16, 1024 in f16, the value whose ulp is
// one): the dot is then Σ code·x + M·Σx, and the unit's value
// (Σ code·x)·scale + (Σx)·bias.
template <class T, int BITS, int GROUP>
struct StreamAffine {
    static_assert(BITS == 2 || BITS == 4 || BITS == 8, "two-, four- or eight-bit codes");
    static_assert(GROUP % 32 == 0, "a unit lies inside one group");
    static constexpr int kCodes = 32;
    static constexpr int kUnroll = BITS == 8 ? kStreamUnroll / 4 : kStreamUnroll / 2;
    static constexpr bool kXsum = true;
    static constexpr int kPerFactor = GROUP / 32;  // units a scale covers
    static constexpr float kMagic = is_same<T, f16>::value ? 1024.f : 128.f;
    static constexpr u32 kMagic2 = is_same<T, f16>::value ? 0x64006400u : 0x43004300u;
    struct Planes {
        const u32* codes;
        const bf16* scales;
        const bf16* biases;
    };
    struct Run {
        __amdgpu_buffer_rsrc_t codes, scales, biases;
    };
    struct Unit {
        u32 w[BITS];
        float scale, bias;
    };
    // Four-bit words pair (c_j, c_j+4), two-bit (c_j, c_j+8); an eight-bit
    // word is two nibble planes, each pairing (c0, c2) (c1, c3).
    static __device__ __forceinline__ void stage(b128 x, int e0, T* xs, int G) {
        const int g = e0 / 32;
        if constexpr (BITS == 4) {
            *reinterpret_cast<b128*>(xs + chunk_at(g, e0 % 32 / 8, G)) = {
                {pair_lo(x.w[0], x.w[2]), pair_hi(x.w[0], x.w[2]), pair_lo(x.w[1], x.w[3]), pair_hi(x.w[1], x.w[3])}};
        } else if constexpr (BITS == 8) {
            *reinterpret_cast<b128*>(xs + chunk_at(g, e0 % 32 / 8, G)) = {
                {pair_lo(x.w[0], x.w[1]), pair_hi(x.w[0], x.w[1]), pair_lo(x.w[2], x.w[3]), pair_hi(x.w[2], x.w[3])}};
        } else {
            // Two-bit pairs reach across chunks: element by element.
            const T* v = reinterpret_cast<const T*>(&x);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int j = (e0 + i) % 32, h = j & 15;
                const int at = (j & 16) + (h < 8 ? 2 * h : 2 * (h - 8) + 1);
                xs[chunk_at(g, at / 8, G) + at % 8] = v[i];
            }
        }
    }
    // Pointer reads, for the grouped engine.
    static __device__ __forceinline__ Unit fetch(const Planes& p, long long u) {
        Unit v;
        const u32* at = p.codes + u * BITS;
        if constexpr (BITS == 2) {
            const moe_u2 w = __builtin_nontemporal_load(reinterpret_cast<const moe_u2*>(at));
            v.w[0] = w[0];
            v.w[1] = w[1];
        } else {
#pragma unroll
            for (int h = 0; h < BITS / 4; ++h) {
                const moe_u4 w = __builtin_nontemporal_load(reinterpret_cast<const moe_u4*>(at) + h);
#pragma unroll
                for (int i = 0; i < 4; ++i) v.w[4 * h + i] = w[i];
            }
        }
        const long long f = u / kPerFactor;
        v.scale = to_f32(p.scales[f]);
        v.bias = to_f32(p.biases[f]);
        return v;
    }
    // A run starts on a row, so on a group.
    static __device__ __forceinline__ Run run(const Planes& p, long long first, int units) {
        const long long f = first / kPerFactor;
        const int factors = (units + kPerFactor - 1) / kPerFactor;
        return {moe_run(p.codes + first * BITS, units * BITS * 4), moe_run(p.scales + f, factors * 2),
                moe_run(p.biases + f, factors * 2)};
    }
    static __device__ __forceinline__ Unit take(const Run& r, int u) {
        Unit v;
        if constexpr (BITS == 2) {
            const moe_u2 w = __builtin_amdgcn_raw_buffer_load_b64(r.codes, u * 8, 0, kStreamNt);
            v.w[0] = w[0];
            v.w[1] = w[1];
        } else {
#pragma unroll
            for (int h = 0; h < BITS / 4; ++h) {
                const moe_u4 w = __builtin_amdgcn_raw_buffer_load_b128(r.codes, u * BITS * 4 + 16 * h, 0, kStreamNt);
#pragma unroll
                for (int i = 0; i < 4; ++i) v.w[4 * h + i] = w[i];
            }
        }
        const int f = 2 * (u / kPerFactor);
        v.scale = __builtin_bit_cast(float, static_cast<u32>(__builtin_amdgcn_raw_buffer_load_b16(r.scales, f, 0, 0))
                                                << 16);
        v.bias = __builtin_bit_cast(float, static_cast<u32>(__builtin_amdgcn_raw_buffer_load_b16(r.biases, f, 0, 0))
                                               << 16);
        return v;
    }
    static __device__ __forceinline__ float value(const Unit& u, const u32* x, int stride, float xsum) {
        u32 xw[16];
        slice_words<16>(x, stride, xw);
        float part;
        if constexpr (BITS == 8) {
            u32 lo[16], hi[16];
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                lo[2 * q] = (u.w[q] & 0x000F000Fu) | kMagic2;
                lo[2 * q + 1] = ((u.w[q] >> 8) & 0x000F000Fu) | kMagic2;
                hi[2 * q] = ((u.w[q] >> 4) & 0x000F000Fu) | kMagic2;
                hi[2 * q + 1] = ((u.w[q] >> 12) & 0x000F000Fu) | kMagic2;
            }
            const float l = fmaf(-kMagic, xsum, own_dot<T, 16>(lo, xw));
            const float h = fmaf(-kMagic, xsum, own_dot<T, 16>(hi, xw));
            part = fmaf(16.f, h, l);
        } else {
            constexpr int PER = 32 / BITS;  // codes a word
            constexpr u32 MASK = ((1u << BITS) - 1u) * 0x00010001u;
            u32 d[16];
#pragma unroll
            for (int q = 0; q < BITS; ++q)
#pragma unroll
                for (int j = 0; j < PER / 2; ++j) d[q * (PER / 2) + j] = ((u.w[q] >> (BITS * j)) & MASK) | kMagic2;
            part = fmaf(-kMagic, xsum, own_dot<T, 16>(d, xw));
        }
        return fmaf(xsum, u.bias, part * u.scale);
    }
};

// Activation packs a thread stages ahead of the weight's first reads; a
// longer row stages the rest after them.
constexpr int kStreamEarlyPacks = 4;

// The unit-dot slots a stream row holds in LDS: one a unit for a row of
// fewer units than lanes, else one a lane (each lane's running sum).
__device__ __forceinline__ int stream_slots(int G) { return min(G, kWave); }

// One route's rows [blockIdx.y·waves·R, +waves·R) of y = W·x, W the expert's
// planes (`p`), R = `rows_per_wave` (≤ 64). Dynamic LDS holds the staged
// activation, its unit sums, and R·stream_slots(G) dots a wave.
//
// A wave's run is R·G units (G = k / kCodes a row), fetched U a lane a
// round, unit u0 + i·64 + lane; a fetch past the run lands zeros, so every
// round issues all its reads unconditionally. The first round is in flight
// before the activation is staged, and each later round while the one
// before it is multiplied.
template <class C, class T>
__device__ __forceinline__ void stream_rows(const typename C::Planes& p, const T* __restrict__ x, T* __restrict__ y,
                                            const T* __restrict__ bias, int n, int k, int rows_per_wave,
                                            unsigned char* smem) {
    constexpr int CODES = C::kCodes;
    constexpr int U = C::kUnroll;
    constexpr int ROUND = kWave * U;
    const int G = k / CODES, S = stream_slots(G);
    T* xs = reinterpret_cast<T*>(smem);
    float* xsum = reinterpret_cast<float*>(smem + ((2 * k + 15) & ~15));
    float* part = xsum + (C::kXsum ? (G + 3) & ~3 : 0);

    const int tid = threadIdx.x, l = lane();
    const int wave = __builtin_amdgcn_readfirstlane(tid / kWave);
    const int R = rows_per_wave;
    const int row0 = (blockIdx.y * kStreamWaves + wave) * R;
    const int rows = max(0, min(R, n - row0));
    const int units = rows * G;
    const typename C::Run run = C::run(p, rows > 0 ? static_cast<long long>(row0) * G : 0, units);

    typename C::Unit w[2][U];
    auto fetch = [&](auto slot, int u0) {
        constexpr int B = decltype(slot)::value;
#pragma unroll
        for (int i = 0; i < U; ++i) w[B][i] = C::take(run, u0 + i * kWave + l);
    };

    // The activation's first packs are read before the weight's, so their
    // stores wait out none of it.
    const int packs = k / 8;
    const b128* x8 = reinterpret_cast<const b128*>(x);
    b128 early[kStreamEarlyPacks];
#pragma unroll
    for (int i = 0; i < kStreamEarlyPacks; ++i) early[i] = x8[min(tid + i * kStreamWaves * kWave, packs - 1)];
    fetch(MoeConst<0>{}, 0);
#pragma unroll
    for (int i = 0; i < kStreamEarlyPacks; ++i) {
        const int at = tid + i * kStreamWaves * kWave;
        if (at < packs) C::stage(early[i], 8 * at, xs, G);
    }
    for (int at = tid + kStreamEarlyPacks * kStreamWaves * kWave; at < packs; at += kStreamWaves * kWave)
        C::stage(x8[at], 8 * at, xs, G);
    float* mine = part + wave * R * S;
    for (int i = l; i < R * S; i += kWave) mine[i] = 0.f;
    lds_barrier();
    if constexpr (C::kXsum) {
        // A unit's sum over its staged slice (a permutation of its natural
        // elements).
        for (int u = tid; u < G; u += kStreamWaves * kWave) {
            float s = 0.f;
#pragma unroll
            for (int q = 0; q < CODES / 8; ++q)
#pragma unroll
                for (int j = 0; j < 8; ++j) s += to_f32(xs[chunk_at(u, q, G) + j]);
            xsum[u] = s;
        }
        lds_barrier();
    }
    if (rows == 0) return;

    // The lane's unit is unit g of row r; a step of 64 units moves it 64 / G
    // rows and 64 mod G units on. Its dot adds to the lane's running sum for
    // the row (slot `l`), or (a row shorter than the wave) lands in the
    // unit's own slot.
    const int step = kWave % G, hop = kWave / G;
    int g = l % G, r = l / G, last = -1;
    float acc = 0.f;
    auto multiply = [&](auto slot, int u0) {
        constexpr int B = decltype(slot)::value;
#pragma unroll
        for (int i = 0; i < U; ++i) {
            const int u = u0 + i * kWave + l;
            const float v = C::value(w[B][i], reinterpret_cast<const u32*>(xs + chunk_at(g, 0, G)), 4 * G,
                                     C::kXsum ? xsum[g] : 0.f);
            acc = r == last ? acc + v : v;
            last = r;
            if (u < units) mine[r * S + (G < kWave ? g : l)] = acc;
            g += step;
            r += hop;
            if (g >= G) {
                g -= G;
                ++r;
            }
        }
    };
    if (units <= ROUND) {
        multiply(MoeConst<0>{}, 0);
    } else {
        for (int u0 = 0; u0 < units; u0 += 2 * ROUND) {
            fetch(MoeConst<1>{}, u0 + ROUND);
            multiply(MoeConst<0>{}, u0);
            fetch(MoeConst<0>{}, u0 + 2 * ROUND);
            multiply(MoeConst<1>{}, u0 + ROUND);
        }
    }

    // Fold: ⌊64 / R⌋ lanes a row, each summing a stride of its row's slots,
    // then down the lanes.
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront", "local");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront", "local");
    const int lanes = kWave / R;
    const int fr = l / lanes, j0 = l % lanes;
    float sum = 0.f;
    if (fr < rows)
        for (int j = j0; j < S; j += lanes) sum += mine[fr * S + j];
    for (int off = 1; off < lanes; off <<= 1) {
        const float o = __shfl_down(sum, off, kWave);
        if (j0 + off < lanes) sum += o;
    }
    if (j0 == 0 && fr < rows) {
        if (bias != nullptr) sum += to_f32(bias[row0 + fr]);
        y[row0 + fr] = from_f32<T>(sum);
    }
}

// The rows of this block a dead route (expert < 0) lands as zeros.
template <class T>
__device__ __forceinline__ void stream_zero(T* __restrict__ y, int n, int rows_per_wave) {
    const int per_block = kStreamWaves * rows_per_wave;
    const int row0 = blockIdx.y * per_block;
    for (int i = threadIdx.x; i < per_block && row0 + i < n; i += blockDim.x) y[row0 + i] = from_f32<T>(0.f);
}

// ----------------------------------------------------------- grouped ----

// A work item is up to MT routes of one expert: 32, or 128 for an expert
// with routes enough to fill it, which reads its weight a quarter as often.
constexpr int kGroupThreads = 256;
// Weight rows are 32 a wave; a 64- or 128-route item takes eight waves, 256
// rows, so two waves a SIMD share the one activation tile the LDS holds.
template <int MT>
constexpr int kGroupWaves = MT >= 64 ? 8 : 4;

// A codec for the grouped engine decodes a chunk (kCodes codes, one fetch)
// into kCodes / 2 dwords of bf16 / f16 pairs, the exact operand, in the
// order `shuffle` puts eight natural activation elements into. A stage is
// kKU chunks a lane (half that at 128 routes, whose activation tile is four
// times as tall); lane group g (lane / 16) takes the stage's chunks g, g + 4,
// ..., so the four groups read neighbouring chunks of a row.

template <class T>
struct GroupDense {
    static constexpr int kCodes = 8;
    static constexpr int kKU = 4;
    struct Planes {
        const T* w;
    };
    struct Chunk {
        moe_u4 w;
    };
    static __device__ __forceinline__ Chunk fetch(const Planes& p, long long row, int k, int c) {
        return {*(reinterpret_cast<const moe_u4*>(p.w + row * k) + c)};
    }
    static __device__ __forceinline__ void decode(const Chunk& c, u32* d) {
#pragma unroll
        for (int i = 0; i < 4; ++i) d[i] = c.w[i];
    }
    static __device__ __forceinline__ b128 shuffle(b128 x) { return x; }
};

// mxfp4 in bf16: the block scale folds into the decode table.
struct GroupMxfp4 {
    static constexpr int kCodes = 32;
    static constexpr int kKU = 2;
    struct Planes {
        const u8* codes;
        const u8* scales;
    };
    struct Chunk {
        moe_u4 w;
        u32 scale;
    };
    static __device__ __forceinline__ Chunk fetch(const Planes& p, long long row, int k, int c) {
        const long long u = row * (k / 32) + c;
        return {__builtin_nontemporal_load(reinterpret_cast<const moe_u4*>(p.codes) + u), p.scales[u]};
    }
    static __device__ __forceinline__ void decode(const Chunk& c, u32* d) {
        const Fp4Tables t = fp4_tables_scaled(c.scale);
#pragma unroll
        for (int q = 0; q < 4; ++q) fp4_word(c.w[q], t, d + 4 * q);
    }
    static __device__ __forceinline__ b128 shuffle(b128 x) { return fp4_shuffle(x); }
};

// An affine bank in bf16: code·scale + bias rounded to bf16, as a
// dequantized bank would hold it, in natural order.
template <int BITS, int GROUP>
struct GroupAffine {
    static_assert(GROUP % 32 == 0, "a chunk lies inside one group");
    static constexpr int kCodes = 32;
    static constexpr int kKU = 2;
    using Planes = typename StreamAffine<bf16, BITS, GROUP>::Planes;
    using Chunk = typename StreamAffine<bf16, BITS, GROUP>::Unit;
    static __device__ __forceinline__ Chunk fetch(const Planes& p, long long row, int k, int c) {
        return StreamAffine<bf16, BITS, GROUP>::fetch(p, row * (k / 32) + c);
    }
    static __device__ __forceinline__ void decode(const Chunk& c, u32* d) {
        constexpr int PER = 32 / BITS;
        constexpr u32 MASK = (1u << BITS) - 1u;
#pragma unroll
        for (int q = 0; q < BITS; ++q)
#pragma unroll
            for (int j = 0; j < PER; j += 2) {
                const float a = static_cast<float>((c.w[q] >> (BITS * j)) & MASK);
                const float b = static_cast<float>((c.w[q] >> (BITS * (j + 1))) & MASK);
                d[(q * PER + j) / 2] = pack2<bf16>(fmaf(a, c.scale, c.bias), fmaf(b, c.scale, c.bias));
            }
    }
    static __device__ __forceinline__ b128 shuffle(b128 x) { return x; }
};

// y[route] = W·act[route / act_div] (+ bias) for routes order[begin ..
// begin + count) (count ≤ MT) of one expert, weight rows [blockIdx.y·32·
// waves, +32·waves). Rounds once on the store, after the bias.
template <class C, class T, int MT, int NBW = 2, int WAVES = kGroupWaves<MT>>
__device__ __forceinline__ void grouped_tile(const T* __restrict__ act, const i32* __restrict__ order, int begin,
                                             int count, const typename C::Planes& p, const T* __restrict__ bias,
                                             T* __restrict__ out, int act_div, int n, int k, int tile = -1) {
    static_assert(MT == 32 || MT == 64 || MT == 128, "a work item is 32, 64 or 128 routes");
    constexpr int CODES = C::kCodes, KU = MT == 32 || C::kKU == 1 ? C::kKU : C::kKU / 2;
    constexpr int KC = 4 * KU * CODES;  // codes a stage
    // LDS rows unpadded, their 16-byte chunks XOR-swizzled by row (chunk c of
    // row r at c ^ (r % 8)): the 16 rows a read touches fill the banks
    // twice over instead of colliding (the padded pitch left the dense leg's
    // LDS reads half in bank conflicts).
    constexpr int LD = KC;
    auto swz = [](int row, int e) { return row * LD + ((((e >> 3) ^ (row & 7)) << 3) | (e & 7)); };
    constexpr int THREADS = WAVES * kWave, TILE_N = 16 * NBW * WAVES;
    constexpr int MB = MT / 16, NB = NBW;
    constexpr int NA = MT * KC / 8 / THREADS;  // 16-byte activation reads a thread a stage
    constexpr int WORDS = CODES / 2;
    static_assert(NA * 8 * THREADS == MT * KC, "the activation tile splits evenly");
    __shared__ __align__(16) T xt[MT * LD];

    const int tid = threadIdx.x, wave = tid / kWave, l = lane(), r16 = l % 16, g = l / 16;
    const int row0 = (tile < 0 ? static_cast<int>(blockIdx.y) : tile) * TILE_N + wave * NB * 16;
    const int chunks = k / CODES;
    const int stages = (k + KC - 1) / KC;

    // The routes' activation rows; a route past the item reads the first
    // (its pieces land as zeros).
    const T* src[NA];
    int col[NA], at[NA];
    bool live[NA];
#pragma unroll
    for (int i = 0; i < NA; ++i) {
        const int piece = tid + i * THREADS;
        const int m = piece / (KC / 8);
        live[i] = m < count;
        src[i] = act + static_cast<long long>(order[begin + (live[i] ? m : 0)] / act_div) * k;
        col[i] = (piece % (KC / 8)) * 8;
        at[i] = swz(m, col[i]);
    }
    long long wrow[NB];
#pragma unroll
    for (int nb = 0; nb < NB; ++nb) wrow[nb] = min(row0 + nb * 16 + r16, n - 1);

    typename C::Chunk wv[2][NB][KU];
    b128 av[NA];
    mfma_cd acc[NB][MB];
#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
#pragma unroll
        for (int mb = 0; mb < MB; ++mb) acc[nb][mb] = mfma_zero();

    // Stage s's reads: the activation first, then the weight, so the store
    // of the one waits out none of the other. Past the last stage the reads
    // repeat it and its activation lands as zeros.
    bool xlive[NA];
    auto fetch = [&](auto slot, int s) {
        constexpr int S = decltype(slot)::value;
        const int sc = min(s, stages - 1);
#pragma unroll
        for (int i = 0; i < NA; ++i) {
            const int c = s * KC + col[i];
            xlive[i] = live[i] && c < k;
            av[i] = *reinterpret_cast<const b128*>(src[i] + min(sc * KC + col[i], k - 8));
        }
#pragma unroll
        for (int nb = 0; nb < NB; ++nb)
#pragma unroll
            for (int j = 0; j < KU; ++j)
                wv[S][nb][j] = C::fetch(p, wrow[nb], k, min(sc * 4 * KU + 4 * j + g, chunks - 1));
    };
    auto store = [&]() {
#pragma unroll
        for (int i = 0; i < NA; ++i)
            *reinterpret_cast<b128*>(xt + at[i]) = xlive[i] ? C::shuffle(av[i]) : b128{{0u, 0u, 0u, 0u}};
    };
    auto multiply = [&](auto slot) {
        constexpr int S = decltype(slot)::value;
#pragma unroll
        for (int j = 0; j < KU; ++j) {
            u32 d[NB][WORDS];
#pragma unroll
            for (int nb = 0; nb < NB; ++nb) C::decode(wv[S][nb][j], d[nb]);
            const int c0 = (4 * j + g) * CODES;
#pragma unroll
            for (int q = 0; q < WORDS; q += 4) {
                b128 xv[MB];
#pragma unroll
                for (int mb = 0; mb < MB; ++mb)
                    xv[mb] = *reinterpret_cast<const b128*>(xt + swz(mb * 16 + r16, c0 + 2 * q));
                // The two k-steps of the read in separate sweeps, so no
                // product waits on the one just before it.
#pragma unroll
                for (int h = 0; h < 4; h += 2)
#pragma unroll
                    for (int nb = 0; nb < NB; ++nb)
#pragma unroll
                        for (int mb = 0; mb < MB; ++mb)
                            acc[nb][mb] = mma16x16<T>(d[nb][q + h], d[nb][q + h + 1], xv[mb].w[h], xv[mb].w[h + 1],
                                                      acc[nb][mb]);
            }
        }
    };

    fetch(MoeConst<0>{}, 0);
    store();
    lds_barrier();
    for (int s = 0; s < stages; s += 2) {
        fetch(MoeConst<1>{}, s + 1);
        multiply(MoeConst<0>{});
        lds_barrier();
        store();
        lds_barrier();
        fetch(MoeConst<0>{}, s + 2);
        multiply(MoeConst<1>{});
        lds_barrier();
        store();
        lds_barrier();
    }

    // D cell (row 4g + t, column r16) of block (nb, mb): weight row, route.
#pragma unroll
    for (int mb = 0; mb < MB; ++mb) {
        const int m = mb * 16 + r16;
        if (m >= count) continue;
        T* o = out + static_cast<long long>(order[begin + m]) * n;
#pragma unroll
        for (int nb = 0; nb < NB; ++nb) {
            const int c = row0 + nb * 16 + 4 * g;
            T v[4];
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const float b = bias != nullptr && c + t < n ? to_f32(bias[c + t]) : 0.f;
                v[t] = from_f32<T>(acc[nb][mb][t] + b);
            }
            if (c + 3 < n && n % 4 == 0) {
                *reinterpret_cast<moe_u2*>(o + c) = __builtin_bit_cast(moe_u2, v);
            } else {
#pragma unroll
                for (int t = 0; t < 4; ++t)
                    if (c + t < n) o[c + t] = v[t];
            }
        }
    }
}

// The work item a grouped block takes (item `at`, by default the block's
// x): `work` lists (expert << 16 | slice) and offsets[experts + 1] holds how
// many; false past the list.
struct GroupItem {
    int expert, begin, count;
};

__device__ __forceinline__ bool group_item(const i32* __restrict__ offsets, const i32* __restrict__ work,
                                           int num_experts, int routes, GroupItem& item, int at = -1) {
    if (at < 0) at = static_cast<int>(blockIdx.x);
    if (at >= offsets[num_experts + 1]) return false;
    const int w = work[at];
    item.expert = w / 65536;
    item.begin = offsets[item.expert] + (w % 65536) * routes;
    item.count = min(offsets[item.expert + 1] - item.begin, routes);
    return item.count > 0;
}

// ------------------------------------------------------------- batch ----

// A work item of up to MT (4, 8, 16 or 32) routes of one expert against
// 64·NBW weight rows a block, NBW 16-row blocks a wave. A stage of the weight
// goes from memory to registers DEPTH stages ahead of its use, by hand-issued
// buffer reads waited by hand (as mgemm's); each wave read takes whole rows
// of the stage (kRowBytes a row, 128 bytes or more on 8 lanes), which the
// card streams near its peak where the matrix core's own layout (16 rows of
// 32 or 64 bytes a read) would not. At its turn the stage goes through the wave's own LDS,
// whence each lane reads its operand pieces; the routes' activation is
// staged through LDS for the block, double-buffered, one barrier a stage.
// The matrix core takes the activation as A and the decoded weight as B, so
// a lane's D cells are its own weight row against four routes.
//
// A codec reads kPieces pieces a lane a stage (kStageK codes of a row over
// the four lane groups), lane group g's piece j at byte poff(j, g) of the
// row's stage and element kpos(j, g) of the activation's; each of a piece's
// kWords code words decodes into four dwords of pairs, the order `shuffle`
// puts eight natural activation elements into. A kFold codec (affine)
// multiplies raw codes and applies a piece's scale and bias after its
// products: its piece is a fourth of one factor group, so the four lane
// groups of one product share their factors.

typedef unsigned int moe_rsrc __attribute__((ext_vector_type(4)));
typedef int moe_i2 __attribute__((ext_vector_type(2)));

// A plane as a raw buffer descriptor: reads at or past `bytes` land zeros
// and touch no memory.
__device__ __forceinline__ moe_rsrc batch_plane(const void* base, unsigned long long bytes) {
    const unsigned long long a = reinterpret_cast<unsigned long long>(base);
    return moe_rsrc{static_cast<u32>(__builtin_amdgcn_readfirstlane(static_cast<int>(a))),
                    static_cast<u32>(__builtin_amdgcn_readfirstlane(static_cast<int>(a >> 32))) & 0xffffu,
                    static_cast<u32>(bytes < 0xfffff000ull ? bytes : 0xfffff000ull), 0x00020000u};
}

// The offset a read past every plane takes: a stage past the last.
constexpr u32 kBatchVoid = 0xfffff000u;

// Buffer reads the compiler neither counts nor moves: 16 bytes nt (the
// weight, read once, whole lines), and 1, 2, 4 or 8 bytes cached (a factor
// run, whose line the neighbouring stages read).
__device__ __forceinline__ void bload_nt(i32x4& d, moe_rsrc r, u32 v) {
    asm volatile("buffer_load_dwordx4 %0, %1, %2, 0 offen nt" : "=v"(d) : "v"(v), "s"(r) : "memory");
}
// The same cached: a weight other blocks read again.
__device__ __forceinline__ void bload_kept(i32x4& d, moe_rsrc r, u32 v) {
    asm volatile("buffer_load_dwordx4 %0, %1, %2, 0 offen" : "=v"(d) : "v"(v), "s"(r) : "memory");
}
template <int BYTES>
__device__ __forceinline__ void bload_run(moe_i2& d, moe_rsrc r, u32 v) {
    static_assert(BYTES == 1 || BYTES == 2 || BYTES == 4 || BYTES == 8, "a run of 1, 2, 4 or 8 bytes");
    if constexpr (BYTES == 8) {
        asm volatile("buffer_load_dwordx2 %0, %1, %2, 0 offen" : "=v"(d) : "v"(v), "s"(r) : "memory");
    } else if constexpr (BYTES == 4) {
        asm volatile("buffer_load_dword %0, %1, %2, 0 offen" : "=v"(d[0]) : "v"(v), "s"(r) : "memory");
    } else if constexpr (BYTES == 2) {
        asm volatile("buffer_load_ushort %0, %1, %2, 0 offen" : "=v"(d[0]) : "v"(v), "s"(r) : "memory");
    } else {
        asm volatile("buffer_load_ubyte %0, %1, %2, 0 offen" : "=v"(d[0]) : "v"(v), "s"(r) : "memory");
    }
}

// The same through a lane's own address, cached.
template <int BYTES>
__device__ __forceinline__ void gload_run(moe_i2& d, const void* p) {
    static_assert(BYTES == 2 || BYTES == 4 || BYTES == 8, "a run of 2, 4 or 8 bytes");
    if constexpr (BYTES == 8) {
        asm volatile("global_load_dwordx2 %0, %1, off" : "=v"(d) : "v"(p) : "memory");
    } else if constexpr (BYTES == 4) {
        asm volatile("global_load_dword %0, %1, off" : "=v"(d[0]) : "v"(p) : "memory");
    } else {
        asm volatile("global_load_ushort %0, %1, off" : "=v"(d[0]) : "v"(p) : "memory");
    }
}

template <class V>
__device__ __forceinline__ void batch_hold(V& v) {
    asm volatile("" : "+v"(v));
}

// An operand piece of BYTES bytes from LDS, as dwords.
template <int BYTES>
struct BatchPiece {
    u32 w[BYTES / 4];
};

template <int BYTES>
__device__ __forceinline__ BatchPiece<BYTES> lds_piece(const u8* p) {
    BatchPiece<BYTES> v;
    if constexpr (BYTES == 16) {
        const b128 x = *reinterpret_cast<const b128*>(p);
#pragma unroll
        for (int i = 0; i < 4; ++i) v.w[i] = x.w[i];
    } else if constexpr (BYTES == 8) {
        const moe_u2 x = *reinterpret_cast<const moe_u2*>(p);
        v.w[0] = x[0];
        v.w[1] = x[1];
    } else {
        v.w[0] = *reinterpret_cast<const u32*>(p);
    }
    return v;
}

// A dense T bank: SBY / 2 elements a row a stage, a piece one 16-byte chunk
// (chunk 4j + g).
template <class T, int SBY>
struct BatchDense {
    static_assert(SBY == 128 || SBY == 256, "a stage of 64 or 128 elements");
    static constexpr int kRowBytes = SBY, kPad = 16, kStageK = SBY / 2, kPieces = SBY / 64, kWords = 1;
    static constexpr int kPieceBytes = 16, kFactorLoads = 0, kFactorLds = 0;
    static constexpr bool kFold = false;
    struct Planes {
        const T* w;
    };
    struct Run {
        moe_rsrc w;
        u32 row;
    };
    struct Factors {};
    struct Piece {};
    static __device__ __forceinline__ Run run(const Planes& p, int n, int k) {
        return {batch_plane(p.w, 2ull * n * k), 2u * k};
    }
    static __device__ __forceinline__ moe_rsrc codes(const Run& r) { return r.w; }
    static __device__ __forceinline__ u32 row_bytes(const Run& r) { return r.row; }
    static constexpr __device__ int poff(int j, int g) { return 16 * (4 * j + g); }
    static constexpr __device__ int kpos(int j, int g) { return 8 * (4 * j + g); }
    static __device__ __forceinline__ void issue_factors(Factors&, const Run&, int, int, bool) {}
    static __device__ __forceinline__ void tie_factors(Factors&) {}
    static __device__ __forceinline__ void stage_factors(const Factors&, u8*) {}
    static __device__ __forceinline__ Piece prep(const Factors&, int) { return {}; }
    static __device__ __forceinline__ void decode(const Piece&, const BatchPiece<16>& c, int, u32* d) {
#pragma unroll
        for (int i = 0; i < 4; ++i) d[i] = c.w[i];
    }
    static __device__ __forceinline__ b128 shuffle(b128 x) { return x; }
};

// An mxfp4 bank in bf16: a stage is SBY / 16 blocks of 32 codes a row,
// each block's scale folded into its decode table. Lane group g takes KU of
// them, [KU·g, +KU), so its KU scale bytes are one read of its own.
template <int SBY>
struct BatchMxfp4 {
    static_assert(SBY == 64 || SBY == 128, "a stage of four or eight blocks");
    static constexpr int KU = SBY / 64;
    static constexpr int kRowBytes = SBY, kPad = 16, kStageK = 2 * SBY, kPieces = KU, kWords = 4;
    static constexpr int kPieceBytes = 16, kFactorLoads = 1, kFactorLds = 0;
    static constexpr bool kFold = false;
    struct Planes {
        const u8* codes;
        const u8* scales;
    };
    struct Run {
        moe_rsrc codes, scales;
        u32 code_row, scale_row;
    };
    struct Factors {
        moe_i2 scale;
    };
    using Piece = Fp4Tables;
    static __device__ __forceinline__ Run run(const Planes& p, int n, int k) {
        return {batch_plane(p.codes, 1ull * n * k / 2), batch_plane(p.scales, 1ull * n * k / 32), k / 2u, k / 32u};
    }
    static __device__ __forceinline__ moe_rsrc codes(const Run& r) { return r.codes; }
    static __device__ __forceinline__ u32 row_bytes(const Run& r) { return r.code_row; }
    static constexpr __device__ int poff(int j, int g) { return 16 * (KU * g + j); }
    static constexpr __device__ int kpos(int j, int g) { return 32 * (KU * g + j); }
    static __device__ __forceinline__ void issue_factors(Factors& f, const Run& run, int row, int s, bool live) {
        const u32 v = live ? row * run.scale_row + s * (4 * KU) + lane() / 16 * KU : kBatchVoid;
        bload_run<KU>(f.scale, run.scales, v);
    }
    static __device__ __forceinline__ void tie_factors(Factors& f) { batch_hold(f.scale); }
    static __device__ __forceinline__ void stage_factors(const Factors&, u8*) {}
    static __device__ __forceinline__ Piece prep(const Factors& f, int j) {
        return fp4_tables_scaled((static_cast<u32>(f.scale[0]) >> (8 * j)) & 0xffu);
    }
    static __device__ __forceinline__ void decode(const Piece& t, const BatchPiece<16>& c, int q, u32* d) {
        fp4_word(c.w[q], t, d);
    }
    static __device__ __forceinline__ b128 shuffle(b128 x) { return fp4_shuffle(x); }
};

// A four-bit affine bank (MLX) in bf16: lane group g's piece j is codes
// [GROUP/4·g, +GROUP/4) of the stage's group j, so the product of one piece
// is one group's, and the piece's value is (Σ code·x)·scale + (Σx)·bias. The
// codes enter as 128 + code (exact), so the product is Σ code·x + 128·Σx and
// the bias taken is bias − 128·scale. A stage's factors of a row (KU scales,
// then KU biases) are read by the lane groups, two a plane, and meet in the
// wave's LDS.
template <int GROUP, int SBY>
struct BatchAffine {
    static_assert(GROUP == 32 || GROUP == 64 || GROUP == 128, "a 32-, 64- or 128-code group");
    static constexpr int KU = 2 * SBY / GROUP;  // groups a stage
    static_assert(KU == 2 || KU == 4 || KU == 8, "two to eight groups a stage");
    static constexpr int LB = KU;  // bytes a lane group reads
    static constexpr int kRowBytes = SBY, kStageK = 2 * SBY, kPieces = KU, kWords = GROUP / 32;
    static constexpr int kPieceBytes = GROUP / 8, kPad = kPieceBytes == 8 ? 8 : 16;
    static constexpr int kFactorLoads = 1, kFactorLds = 4 * KU;  // LDS a row
    static constexpr bool kFold = true;
    using Planes = typename StreamAffine<bf16, 4, GROUP>::Planes;
    struct Run {
        moe_rsrc codes;
        const u8* factors;  // the lane's plane: scales or biases
        u32 code_row, factor_row, last;
    };
    struct Factors {
        moe_i2 f;
    };
    struct Piece {};
    static __device__ __forceinline__ Run run(const Planes& p, int n, int k) {
        const u32 factor_row = 2u * (k / GROUP);
        const void* plane = lane() / 16 < 2 ? static_cast<const void*>(p.scales) : static_cast<const void*>(p.biases);
        return {batch_plane(p.codes, 1ull * n * k / 2), static_cast<const u8*>(plane), k / 2u, factor_row,
                static_cast<u32>(n) * factor_row - LB};
    }
    static __device__ __forceinline__ moe_rsrc codes(const Run& r) { return r.codes; }
    static __device__ __forceinline__ u32 row_bytes(const Run& r) { return r.code_row; }
    static constexpr __device__ int poff(int j, int g) { return GROUP / 2 * j + GROUP / 8 * g; }
    static constexpr __device__ int kpos(int j, int g) { return GROUP * j + GROUP / 4 * g; }
    // Lane group g reads half g % 2 of the stage's scales (g < 2) or biases;
    // past the planes' end it rereads their last run (its products are
    // zeros).
    static __device__ __forceinline__ u32 half(int g) { return static_cast<u32>(g % 2 * KU); }
    static __device__ __forceinline__ void issue_factors(Factors& f, const Run& run, int row, int s, bool) {
        const u32 v = min(row * run.factor_row + s * (2 * KU) + half(lane() / 16), run.last);
        gload_run<LB>(f.f, run.factors + v);
    }
    static __device__ __forceinline__ void tie_factors(Factors& f) { batch_hold(f.f); }
    // `at` is the lane's row of the wave's factor LDS.
    static __device__ __forceinline__ void stage_factors(const Factors& f, u8* at) {
        const int g = lane() / 16;
        u8* p = at + (g < 2 ? 0 : 2 * KU) + half(g);
        if constexpr (LB == 8) {
            *reinterpret_cast<moe_u2*>(p) = __builtin_bit_cast(moe_u2, f.f);
        } else if constexpr (LB == 4) {
            *reinterpret_cast<u32*>(p) = static_cast<u32>(f.f[0]);
        } else {
            *reinterpret_cast<u16*>(p) = static_cast<u16>(f.f[0]);
        }
    }
    static __device__ __forceinline__ Piece prep(const Factors&, int) { return {}; }
    // Eight codes as (c0,c4) (c1,c5) (c2,c6) (c3,c7), each 128 + code.
    static __device__ __forceinline__ void decode(const Piece&, const BatchPiece<GROUP / 8>& c, int q, u32* d) {
        const u32 w = c.w[q];
#pragma unroll
        for (int i = 0; i < 4; ++i) d[i] = ((w >> (4 * i)) & 0x000F000Fu) | 0x43004300u;
    }
    static __device__ __forceinline__ b128 shuffle(b128 x) {
        return {{pair_lo(x.w[0], x.w[2]), pair_hi(x.w[0], x.w[2]), pair_lo(x.w[1], x.w[3]), pair_hi(x.w[1], x.w[3])}};
    }
    // Group j's scale and its bias less 128 scales, from the lane's row of
    // the factor LDS.
    static __device__ __forceinline__ void factors(const u8* at, int j, float& scale, float& bias) {
        const u16 s = reinterpret_cast<const u16*>(at)[j], b = reinterpret_cast<const u16*>(at)[KU + j];
        scale = __builtin_bit_cast(float, static_cast<u32>(s) << 16);
        bias = fmaf(-128.f, scale, __builtin_bit_cast(float, static_cast<u32>(b) << 16));
    }
};

// y[route] = W·act[route / act_div] (+ bias) for routes order[begin ..
// begin + count) (count ≤ MT) of one expert (routes begin .. begin + count
// themselves without an order), weight rows [tile·64·NBW, +64·NBW). Rounds
// once on the store, after the bias. Short of 16 routes the activation's
// rows past MT read one zero row. SWIGLU reads a packed [gate | up] bank of
// 2·n rows and lands silu(gate)·up over n: a wave's two blocks are its 16
// gate rows and the matching up rows, the products rounded to T first, as
// the unfused projection lands them. A block may take stages [s0, s1) of K
// alone and land its f32 partial in `part` (rows of n, 2·n under SWIGLU,
// gate then up) for a later sum, unbiased and unrounded. NT reads the weight
// as no other block reads it again.
template <class C, class T, int MT, int NBW, int DEPTH, bool SWIGLU = false, bool NT = true>
__device__ __forceinline__ void batch_tile(const T* __restrict__ act, const i32* __restrict__ order, int begin,
                                           int count, const typename C::Planes& p, const T* __restrict__ bias,
                                           T* __restrict__ out, int act_div, int n, int k, int tile, int s0 = 0,
                                           int s1 = -1, float* __restrict__ part = nullptr) {
    static_assert(!SWIGLU || NBW == 2, "a SwiGLU wave takes a gate block and its up block");
    static_assert(MT == 4 || MT == 8 || MT == 16 || MT == 32 || MT == 64, "a batch item is 4 to 64 routes");
    constexpr int MB = MT < 16 ? 1 : MT / 16, KC = C::kStageK, LD = KC + 8, R = DEPTH + 1;
    constexpr int RT = MT < 16 ? MT + 1 : MT;  // LDS rows a buffer
    constexpr int WAVES = kGroupThreads / kWave;
    constexpr int RW = 16 * NBW;                         // weight rows a wave
    constexpr int SBY = C::kRowBytes, LPR = SBY / 16;    // a row's stage, its 16-byte reads
    constexpr int NL = RW * SBY / (16 * kWave);          // 16-byte weight reads a lane a stage
    constexpr int PITCH = SBY + C::kPad, FPITCH = C::kFactorLds;
    constexpr int PA = MT * KC / 8;                                // 16-byte activation pieces a stage
    constexpr int NA = (PA + kGroupThreads - 1) / kGroupThreads;  // reads a thread
    constexpr int NW = NL + NBW * C::kFactorLoads;                // weight reads a lane a stage
    static_assert(PA % kGroupThreads == 0 || PA < kGroupThreads, "the activation stage splits evenly");
    static_assert(NL >= 1 && NL * 16 * kWave == RW * SBY && kWave % LPR == 0, "a wave's stage is whole reads");
    static_assert(DEPTH >= 2 && NW + (DEPTH - 1) * (NA + NW) < 64, "the reads in flight fit the count");
    __shared__ __align__(16) T xs[2 * RT * LD];
    __shared__ __align__(16) u8 wl[WAVES][RW * (PITCH + FPITCH)];

    const int tid = threadIdx.x, l = lane(), r16 = l % 16, g = l / 16;
    const int wave = __builtin_amdgcn_readfirstlane(tid / kWave);
    const int row0 = (tile * WAVES + wave) * (SWIGLU ? 16 : 16 * NBW);
    const int all = (k + KC - 1) / KC, first = s0;
    const int stages = (s1 < 0 ? all : min(s1, all)) - first;
    // The item's 16-route blocks that hold a route: the rest neither read
    // their activation nor multiply.
    const int live_mb = MB == 1 ? 1 : __builtin_amdgcn_readfirstlane((count + 15) / 16);
    const typename C::Run run = C::run(p, SWIGLU ? 2 * n : n, k);
    // The bank row of the wave's local row r; a row past the bank reads the
    // last.
    auto bank_row = [&](int r) { return SWIGLU ? min(row0 + r % 16, n - 1) + r / 16 * n : min(row0 + r, n - 1); };
    auto route = [&](int m) { return order != nullptr ? order[begin + m] : begin + m; };
    const moe_rsrc wr = C::codes(run);

    // The routes' activation rows; a route past the item reads the first and
    // a piece past K the row's last (both land as zeros). Short of a thread a
    // piece, the threads past the stage reread a piece and store nothing.
    const T* src[NA];
    int col[NA], at[NA];
    bool live[NA], mine[NA];
#pragma unroll
    for (int i = 0; i < NA; ++i) {
        const int piece = (tid + i * kGroupThreads) % PA;
        const int m = piece / (KC / 8);
        mine[i] = tid + i * kGroupThreads < PA;
        live[i] = m < count;
        src[i] = act + static_cast<long long>(route(live[i] ? m : 0) / act_div) * k;
        col[i] = (piece % (KC / 8)) * 8;
        at[i] = m * LD + col[i];
    }
    // A wave's read i takes 16 bytes of its stage's flat run of rows, the
    // (64·i + l)th.
    u32 wv[NL];
    int frow[NBW];
#pragma unroll
    for (int nb = 0; nb < NBW; ++nb) frow[nb] = bank_row(nb * 16 + r16);
#pragma unroll
    for (int i = 0; i < NL; ++i) {
        const int flat = kWave * i + l;
        wv[i] = static_cast<u32>(bank_row(flat / LPR)) * C::row_bytes(run) + flat % LPR * 16;
    }

    i32x4 w[R][NL];
    typename C::Factors f[R][NBW];
    i32x4 a[R][NA];
    mfma_cd acc[NBW][MB];
#pragma unroll
    for (int nb = 0; nb < NBW; ++nb)
#pragma unroll
        for (int mb = 0; mb < MB; ++mb) acc[nb][mb] = mfma_zero();

    // Stage s's reads, the activation's then the weight's; past the last
    // stage the activation rereads the last and the weight reads nothing.
    auto issue = [&](auto slot, int s) {
        constexpr int S = decltype(slot)::value;
        const int sc = first + min(s, stages - 1), sa = first + s;
        const bool on = s < stages;
#pragma unroll
        for (int i = 0; i < NA; ++i) gload16(a[S][i], src[i] + min(sc * KC + col[i], k - 8));
#pragma unroll
        for (int i = 0; i < NL; ++i) {
            const u32 v = on ? wv[i] + sa * SBY : kBatchVoid;
            if constexpr (NT) {
                bload_nt(w[S][i], wr, v);
            } else {
                bload_kept(w[S][i], wr, v);
            }
        }
#pragma unroll
        for (int nb = 0; nb < NBW; ++nb) C::issue_factors(f[S][nb], run, frow[nb], sa, on);
    };
    auto store = [&](auto slot, int buf, int s) {
        constexpr int S = decltype(slot)::value;
        T* st = xs + buf * (RT * LD);
#pragma unroll
        for (int i = 0; i < NA; ++i) {
            const bool on = live[i] && (first + s) * KC + col[i] < k;
            if (PA % kGroupThreads == 0 || mine[i])
                *reinterpret_cast<b128*>(st + at[i]) =
                    on ? C::shuffle(__builtin_bit_cast(b128, a[S][i])) : b128{{0u, 0u, 0u, 0u}};
        }
    };
    // The stage's weight into the wave's LDS, as it was read.
    auto stage_weight = [&](auto slot) {
        constexpr int S = decltype(slot)::value;
#pragma unroll
        for (int i = 0; i < NL; ++i) {
            const int flat = kWave * i + l;
            u8* q = wl[wave] + flat / LPR * PITCH + flat % LPR * 16;
            if constexpr (PITCH % 16 == 0) {
                *reinterpret_cast<i32x4*>(q) = w[S][i];
            } else {
                reinterpret_cast<moe_u2*>(q)[0] = moe_u2{static_cast<u32>(w[S][i][0]), static_cast<u32>(w[S][i][1])};
                reinterpret_cast<moe_u2*>(q)[1] = moe_u2{static_cast<u32>(w[S][i][2]), static_cast<u32>(w[S][i][3])};
            }
        }
#pragma unroll
        for (int nb = 0; nb < NBW; ++nb) C::stage_factors(f[S][nb], wl[wave] + RW * PITCH + (nb * 16 + r16) * FPITCH);
    };
    // Every LDS read of the stage is issued before the first product, so the
    // products wait out the reads once, not piece by piece.
    auto multiply = [&](auto slot, const T* xb) {
        constexpr int S = decltype(slot)::value;
        constexpr int KU = C::kPieces, W = C::kWords;
        constexpr u32 kOnes = is_same<T, f16>::value ? 0x3C003C00u : 0x3F803F80u;
        BatchPiece<C::kPieceBytes> cw[KU][NBW];
        b128 xv[KU][W][MB];
        float scale[KU][NBW], shift[KU][NBW];
#pragma unroll
        for (int j = 0; j < KU; ++j)
#pragma unroll
            for (int nb = 0; nb < NBW; ++nb)
                cw[j][nb] = lds_piece<C::kPieceBytes>(wl[wave] + (nb * 16 + r16) * PITCH + C::poff(j, g));
#pragma unroll
        for (int j = 0; j < KU; ++j)
#pragma unroll
            for (int q = 0; q < W; ++q)
#pragma unroll
                for (int mb = 0; mb < MB; ++mb)
                    if (mb < live_mb)
                        xv[j][q][mb] = *reinterpret_cast<const b128*>(xb + mb * 16 * LD + C::kpos(j, g) + 8 * q);
        if constexpr (C::kFold) {
#pragma unroll
            for (int j = 0; j < KU; ++j)
#pragma unroll
                for (int nb = 0; nb < NBW; ++nb)
                    C::factors(wl[wave] + RW * PITCH + (nb * 16 + r16) * FPITCH, j, scale[j][nb], shift[j][nb]);
        }
#pragma unroll
        for (int j = 0; j < KU; ++j) {
            typename C::Piece pc[NBW];
#pragma unroll
            for (int nb = 0; nb < NBW; ++nb) pc[nb] = C::prep(f[S][nb], j);
            mfma_cd part[NBW][MB], xsum[MB];
            if constexpr (C::kFold) {
#pragma unroll
                for (int mb = 0; mb < MB; ++mb) {
                    xsum[mb] = mfma_zero();
#pragma unroll
                    for (int nb = 0; nb < NBW; ++nb) part[nb][mb] = mfma_zero();
                }
            }
#pragma unroll
            for (int q = 0; q < W; ++q) {
#pragma unroll
                for (int nb = 0; nb < NBW; ++nb) {
                    u32 d[4];
                    C::decode(pc[nb], cw[j][nb], q, d);
#pragma unroll
                    for (int mb = 0; mb < MB; ++mb) {
                        if (mb >= live_mb) continue;
                        const b128& x = xv[j][q][mb];
                        mfma_cd& c = C::kFold ? part[nb][mb] : acc[nb][mb];
                        c = mma16x16<T>(x.w[0], x.w[1], d[0], d[1], c);
                        c = mma16x16<T>(x.w[2], x.w[3], d[2], d[3], c);
                    }
                }
                if constexpr (C::kFold) {
#pragma unroll
                    for (int mb = 0; mb < MB; ++mb) {
                        if (mb >= live_mb) continue;
                        const b128& x = xv[j][q][mb];
                        xsum[mb] = mma16x16<T>(x.w[0], x.w[1], kOnes, kOnes, xsum[mb]);
                        xsum[mb] = mma16x16<T>(x.w[2], x.w[3], kOnes, kOnes, xsum[mb]);
                    }
                }
            }
            if constexpr (C::kFold) {
#pragma unroll
                for (int nb = 0; nb < NBW; ++nb)
#pragma unroll
                    for (int mb = 0; mb < MB; ++mb)
#pragma unroll
                        for (int t = 0; t < 4; ++t)
                            acc[nb][mb][t] = fmaf(scale[j][nb], part[nb][mb][t],
                                                  fmaf(shift[j][nb], xsum[mb][t], acc[nb][mb][t]));
            }
        }
    };

    Unroll<DEPTH, 0>::run([&](auto slot, int s) { issue(slot, s); }, 0);
    // The first stage's activation: younger than it are its own weight and
    // the DEPTH - 1 stages behind.
    vm_wait<NW + (DEPTH - 1) * (NA + NW)>();
    tie(a[0]);
    store(Const<0>{}, 0, 0);
    if constexpr (MT < 16)
        for (int i = tid; i < LD; i += kGroupThreads) xs[MT * LD + i] = xs[(RT + MT) * LD + i] = from_f32<T>(0.f);
    lds_barrier();

    const T* xrow = xs + (MT < 16 ? min(r16, MT) : r16) * LD;
    auto stage_at = [&](auto cur, int s) {
        constexpr int CU = decltype(cur)::value;
        constexpr int NEXT = (CU + 1) % R;
        if (s >= stages) return;
        // Stage s + 1's activation, and so (older still) stage s's weight.
        vm_wait<NW + (DEPTH - 2) * (NA + NW)>();
        tie(a[NEXT]);
        tie(w[CU]);
#pragma unroll
        for (int nb = 0; nb < NBW; ++nb) C::tie_factors(f[CU][nb]);
        store(Const<NEXT>{}, (s + 1) & 1, s + 1);
        stage_weight(Const<CU>{});
        issue(Const<(CU + DEPTH) % R>{}, s + DEPTH);
        multiply(Const<CU>{}, xrow + (s & 1) * (RT * LD));
        lds_barrier();
    };
    for (int s = 0; s < stages; s += R) Unroll<R, 0>::run(stage_at, s);
    // The reads past the end land before their registers are given up.
    vm_wait<0>();
#pragma unroll
    for (int r = 0; r < R; ++r) {
        tie(a[r]);
        tie(w[r]);
#pragma unroll
        for (int nb = 0; nb < NBW; ++nb) C::tie_factors(f[r][nb]);
    }

    // D cell (route 4g + t, weight row r16) of block (nb, mb).
#pragma unroll
    for (int mb = 0; mb < MB; ++mb)
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int m = mb * 16 + 4 * g + t;
            if (m >= count) continue;
            if (part != nullptr) {
                float* q = part + static_cast<long long>(route(m)) * (SWIGLU ? 2 * n : n);
#pragma unroll
                for (int nb = 0; nb < NBW; ++nb) {
                    const int c = row0 + (SWIGLU ? 0 : nb * 16) + r16;
                    if (c < n) q[(SWIGLU ? nb * n : 0) + c] = acc[nb][mb][t];
                }
                continue;
            }
            T* o = out + static_cast<long long>(route(m)) * n;
            if constexpr (SWIGLU) {
                const int c = row0 + r16;
                if (c >= n) continue;
                const float gt = to_f32(from_f32<T>(acc[0][mb][t])), up = to_f32(from_f32<T>(acc[1][mb][t]));
                o[c] = from_f32<T>(gt * (1.f / (1.f + __expf(-gt))) * up);
            } else {
#pragma unroll
                for (int nb = 0; nb < NBW; ++nb) {
                    const int c = row0 + nb * 16 + r16;
                    if (c >= n) continue;
                    const float b = bias != nullptr ? to_f32(bias[c]) : 0.f;
                    o[c] = from_f32<T>(acc[nb][mb][t] + b);
                }
            }
        }
}

// -------------------------------------------------------------- wide ----

// A dense 16-bit bank at wide decode batches (an expert sees tens to a couple
// of hundred routes): a work item is all of an expert's routes up to 4·MW,
// against NT weight rows a block. The block's four waves share the weight
// and split the routes, wave w taking the item's 16-route blocks w, w + 4,
// ..., so an item covers its expert's routes at once and the weight is read
// from memory once; the tiles are narrow enough that blocks run several to
// a CU and the grid's last round is a small part of it.
//
// A stage is KC (64 or 128) elements of a row (128 or 256 bytes on 8 or 16
// lanes, whole rows a wave read); the weight goes memory → registers (DEPTH stages ahead, hand-issued
// buffer reads) → the block's LDS (double-buffered, one barrier a stage) →
// each wave's operand pieces. A wave's activation goes memory → registers
// (ADEPTH stages ahead) straight into the operand: lane (r16, g) holds route
// r16's 16 bytes at element 8·(4j + g) of the stage, and its weight piece is
// the same elements of row r16, so the matrix core's k order agrees. The
// weight is A and the activation B: a lane's D cells are four neighbouring
// weight rows of one route, stored as one 8-byte write.

// y[route] = W·act[route / act_div] for routes order[begin .. begin + count)
// (count ≤ 4·MW) of one expert (planes `w`, n x k, k a multiple of KC),
// weight rows [tile·NT, +NT). `act_bytes` bounds the activation's buffer.
template <class T, int MW, int NT, int KC, int DEPTH, int ADEPTH>
__device__ __forceinline__ void wide_tile(const T* __restrict__ act, unsigned long long act_bytes,
                                          const i32* __restrict__ order, int begin, int count,
                                          const T* __restrict__ w, T* __restrict__ out, int act_div, int n, int k,
                                          int tile) {
    static_assert(MW % 16 == 0 && MW >= 16 && MW <= 64, "a wave takes one to four 16-route blocks");
    static_assert(NT == 32 || NT == 64 || NT == 128, "a block takes 32, 64 or 128 weight rows");
    static_assert(KC == 64 || KC == 128, "a stage of 64 or 128 elements");
    constexpr int PITCH = 2 * KC + 16;
    static_assert(DEPTH % ADEPTH == 0 && ADEPTH >= 1 && DEPTH >= 2, "the activation ring divides the weight's");
    constexpr int WAVES = kGroupThreads / kWave, MB = MW / 16, NB = NT / 16;
    constexpr int KU = KC / 32;                    // 16-byte pieces a lane a 16-route block
    constexpr int NW = NT * 2 * KC / (16 * kGroupThreads);  // weight reads a thread a stage
    constexpr int NA = MB * KU;                        // activation reads a lane a stage
    constexpr int LPR = 2 * KC / 16;               // lanes a row's stage
    static_assert(NW >= 1 && NW * 16 * kGroupThreads == NT * 2 * KC, "the stage is whole reads");
    static_assert((ADEPTH - 1) * (NW + NA) + NW * DEPTH + NA * ADEPTH < 64, "the reads in flight fit the count");
    __shared__ __align__(16) u8 wl[2][NT * PITCH];

    const int tid = threadIdx.x, l = lane(), r16 = l % 16, g = l / 16;
    const int wave = __builtin_amdgcn_readfirstlane(tid / kWave);
    const int row0 = tile * NT;
    const int stages = k / KC;
    const int blocks = (count + 15) / 16;
    // The wave's live 16-route blocks: wave + 4·i for i < live.
    const int live = __builtin_amdgcn_readfirstlane(blocks > wave ? (blocks - wave + WAVES - 1) / WAVES : 0);
    const moe_rsrc wr = batch_plane(w, 2ull * n * k);
    const moe_rsrc ar = batch_plane(act, act_bytes);

    // A read's offset: the weight's flat run of rows (thread tid's read i is
    // 16 bytes (256·i + tid) of the stage); the activation's route row.
    u32 wo[NW];
#pragma unroll
    for (int i = 0; i < NW; ++i) {
        const int flat = kGroupThreads * i + tid;
        wo[i] = static_cast<u32>(min(row0 + flat / LPR, n - 1)) * (2u * k) + flat % LPR * 16;
    }
    u32 ao[MB];
    bool alive[MB];
#pragma unroll
    for (int i = 0; i < MB; ++i) {
        const int m = (wave + WAVES * i) * 16 + r16;
        alive[i] = m < count;
        ao[i] = alive[i] ? static_cast<u32>(order[begin + m] / act_div) * (2u * k) + 16 * g : kBatchVoid;
    }

    i32x4 wv[DEPTH][NW];
    i32x4 av[ADEPTH][NA];
    mfma_cd acc[NB][MB];
#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
#pragma unroll
        for (int mb = 0; mb < MB; ++mb) acc[nb][mb] = mfma_zero();

    // Every read is issued, a dead one at the void offset (it lands zeros and
    // touches nothing), so the count of reads behind each stays fixed.
    auto issue_w = [&](auto slot, int s) {
        constexpr int S = decltype(slot)::value;
        const bool on = s < stages;
#pragma unroll
        for (int i = 0; i < NW; ++i) bload_nt(wv[S][i], wr, on ? wo[i] + s * (2 * KC) : kBatchVoid);
    };
    auto issue_a = [&](auto slot, int s) {
        constexpr int S = decltype(slot)::value;
        const bool on = s < stages;
#pragma unroll
        for (int i = 0; i < MB; ++i)
#pragma unroll
            for (int j = 0; j < KU; ++j)
                bload_kept(av[S][i * KU + j], ar,
                           on && alive[i] ? ao[i] + s * (2 * KC) + 64 * j : kBatchVoid);
    };
    auto stage_w = [&](auto slot, int buf) {
        constexpr int S = decltype(slot)::value;
#pragma unroll
        for (int i = 0; i < NW; ++i) {
            const int flat = kGroupThreads * i + tid;
            *reinterpret_cast<i32x4*>(wl[buf] + flat / LPR * PITCH + flat % LPR * 16) = wv[S][i];
        }
    };
    auto multiply = [&](auto aslot, int buf) {
        constexpr int A = decltype(aslot)::value;
#pragma unroll
        for (int j = 0; j < KU; ++j) {
            b128 wp[NB];
#pragma unroll
            for (int nb = 0; nb < NB; ++nb)
                wp[nb] = *reinterpret_cast<const b128*>(wl[buf] + (nb * 16 + r16) * PITCH + 16 * (4 * j + g));
#pragma unroll
            for (int h = 0; h < 4; h += 2)
#pragma unroll
                for (int mb = 0; mb < MB; ++mb) {
                    if (mb >= live) continue;
                    const i32x4& x = av[A][mb * KU + j];
#pragma unroll
                    for (int nb = 0; nb < NB; ++nb)
                        acc[nb][mb] = mma16x16<T>(wp[nb].w[h], wp[nb].w[h + 1], static_cast<u32>(x[h]),
                                                  static_cast<u32>(x[h + 1]), acc[nb][mb]);
                }
        }
    };

    // The prologue issues as the steady state would have: virtual stage t
    // (t from -DEPTH) issues the weight of t + DEPTH and the activation of
    // t + ADEPTH.
    Unroll<DEPTH, 0>::run(
        [&](auto slot, int t) {
            constexpr int S = decltype(slot)::value;
            issue_w(slot, t);
            if constexpr (S >= DEPTH - ADEPTH) issue_a(Const<(S - (DEPTH - ADEPTH)) % ADEPTH>{}, t - (DEPTH - ADEPTH));
        },
        0);

    auto stage_at = [&](auto cur, int s) {
        constexpr int CU = decltype(cur)::value;
        constexpr int AC = CU % ADEPTH;
        if (s >= stages) return;
        // Stage s's activation, and (older) its weight; younger are the
        // ADEPTH - 1 stages behind.
        vm_wait<(ADEPTH - 1) * (NW + NA)>();
        tie(wv[CU]);
        tie(av[AC]);
        stage_w(Const<CU>{}, s & 1);
        issue_w(Const<CU>{}, s + DEPTH);
        lds_barrier();
        multiply(Const<AC>{}, s & 1);
        issue_a(Const<AC>{}, s + ADEPTH);
    };
    for (int s = 0; s < stages; s += DEPTH) Unroll<DEPTH, 0>::run(stage_at, s);
    vm_wait<0>();
#pragma unroll
    for (int r = 0; r < DEPTH; ++r) tie(wv[r]);
#pragma unroll
    for (int r = 0; r < ADEPTH; ++r) tie(av[r]);

    // D cell (weight row 4g + t, route r16) of block (nb, mb).
#pragma unroll
    for (int mb = 0; mb < MB; ++mb) {
        if (mb >= live || !alive[mb]) continue;
        T* o = out + static_cast<long long>(order[begin + (wave + WAVES * mb) * 16 + r16]) * n;
#pragma unroll
        for (int nb = 0; nb < NB; ++nb) {
            const int c = row0 + nb * 16 + 4 * g;
            T v[4];
#pragma unroll
            for (int t = 0; t < 4; ++t) v[t] = from_f32<T>(acc[nb][mb][t]);
            if (c + 3 < n && n % 4 == 0) {
                *reinterpret_cast<moe_u2*>(o + c) = __builtin_bit_cast(moe_u2, v);
            } else {
#pragma unroll
                for (int t = 0; t < 4; ++t)
                    if (c + t < n) o[c + t] = v[t];
            }
        }
    }
}

// ----------------------------------------------------------- ordering ---

// Exclusive prefix of one int a thread over the block (whole wavefronts,
// at most 16 of them); `total` gets the block's sum. Every thread calls it.
__device__ __forceinline__ int block_scan(int v, int* wave_totals, int& total) {
    const int l = lane(), w = threadIdx.x / kWave, waves = blockDim.x / kWave;
    int inc = v;
#pragma unroll
    for (int off = 1; off < kWave; off <<= 1) {
        const int up = __shfl_up(inc, off, kWave);
        if (l >= off) inc += up;
    }
    if (l == kWave - 1) wave_totals[w] = inc;
    __syncthreads();
    int before = 0;
    total = 0;
    for (int i = 0; i < waves; ++i) {
        const int t = wave_totals[i];
        if (i < w) before += t;
        total += t;
    }
    __syncthreads();
    return before + inc - v;
}

}  // namespace pie::linear
