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
// of up to 32 or 128 of one expert's routes, gathers their activation rows
// into LDS and multiplies them against 128 weight rows a block on the 16 x 16
// x 16 matrix core, the weight decoded from registers straight into the
// operand.
//
// Both rely on the matrix core reading its k positions in any order so long
// as both operands agree: a codec decodes codes in whatever pairing is
// cheapest, and the activation is staged in that same order.

#include "prelude/mfma.hip"

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
constexpr int kGroupTileN = 128;  // weight rows a block (N), 32 a wave
constexpr int kGroupThreads = 256;

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
        return {__builtin_nontemporal_load(reinterpret_cast<const moe_u4*>(p.w + row * k) + c)};
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
// begin + count) (count ≤ MT) of one expert, weight rows [blockIdx.y·128,
// +128). Rounds once on the store, after the bias.
template <class C, class T, int MT>
__device__ __forceinline__ void grouped_tile(const T* __restrict__ act, const i32* __restrict__ order, int begin,
                                             int count, const typename C::Planes& p, const T* __restrict__ bias,
                                             T* __restrict__ out, int act_div, int n, int k) {
    static_assert(MT == 32 || MT == 128, "a work item is 32 or 128 routes");
    constexpr int CODES = C::kCodes, KU = MT == 32 || C::kKU == 1 ? C::kKU : C::kKU / 2;
    constexpr int KC = 4 * KU * CODES;  // codes a stage
    constexpr int LD = KC + 8;          // LDS row pitch, elements
    constexpr int MB = MT / 16, NB = kGroupTileN / (16 * (kGroupThreads / kWave));
    constexpr int NA = MT * KC / 8 / kGroupThreads;  // 16-byte activation reads a thread a stage
    constexpr int WORDS = CODES / 2;
    static_assert(NA * 8 * kGroupThreads == MT * KC, "the activation tile splits evenly");
    __shared__ __align__(16) T xt[MT * LD];

    const int tid = threadIdx.x, wave = tid / kWave, l = lane(), r16 = l % 16, g = l / 16;
    const int row0 = blockIdx.y * kGroupTileN + wave * NB * 16;
    const int chunks = k / CODES;
    const int stages = (k + KC - 1) / KC;

    // The routes' activation rows; a route past the item reads the first
    // (its pieces land as zeros).
    const T* src[NA];
    int col[NA], at[NA];
    bool live[NA];
#pragma unroll
    for (int i = 0; i < NA; ++i) {
        const int piece = tid + i * kGroupThreads;
        const int m = piece / (KC / 8);
        live[i] = m < count;
        src[i] = act + static_cast<long long>(order[begin + (live[i] ? m : 0)] / act_div) * k;
        col[i] = (piece % (KC / 8)) * 8;
        at[i] = m * LD + col[i];
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
                    xv[mb] = *reinterpret_cast<const b128*>(xt + (mb * 16 + r16) * LD + c0 + 2 * q);
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

// The work item a grouped block takes: `work` lists (expert << 16 | slice)
// and offsets[experts + 1] holds how many; false past the list.
struct GroupItem {
    int expert, begin, count;
};

__device__ __forceinline__ bool group_item(const i32* __restrict__ offsets, const i32* __restrict__ work,
                                           int num_experts, int routes, GroupItem& item) {
    if (static_cast<int>(blockIdx.x) >= offsets[num_experts + 1]) return false;
    const int w = work[blockIdx.x];
    item.expert = w / 65536;
    item.begin = offsets[item.expert] + (w % 65536) * routes;
    item.count = min(offsets[item.expert + 1] - item.begin, routes);
    return item.count > 0;
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
