#pragma once

// Affine-quantized weights, as MLX packs them: a weight row of K codes at
// BITS (2, 4 or 8) bits, packed little-end first into 32-bit words, with one
// factor pair a GROUP of codes. A code c of factor (s, off) stands for
//
//   post offset     s·c + off
//   pre offset      s·(c − off)      off an integer byte, a real, or 2^(BITS−1)
//
// The factors are F (bf16 or f16), one row of K / GROUP a weight row.

#include "prim/mfma.hip"

namespace pie::linear {

// W 32-bit words moved as one access. A vector, not a struct: a struct
// read from global memory lands in scratch and is copied out of it.
template <int W>
struct WordsOf {
    typedef u32 type __attribute__((ext_vector_type(W)));
};
template <>
struct WordsOf<1> {
    typedef u32 type;  // a register operand binds a scalar, not a 1-vector
};
template <int W>
using Words = typename WordsOf<W>::type;

// Word i of a read.
template <class V>
__device__ __forceinline__ u32 word(V v, int i) {
    return v[i];
}
__device__ __forceinline__ u32 word(u32 v, int) { return v; }

// A read the compiler knows is of global memory: through a generic pointer
// it issues a flat read, which also counts against LDS waits, so a loop's
// LDS traffic would wait on its prefetch. NT marks codes streamed once.
template <class V, bool NT = false>
__device__ __forceinline__ V read_global(const void* p) {
    const auto* g = (const __attribute__((address_space(1))) V*)(p);
    if constexpr (NT) {
        return __builtin_nontemporal_load(g);
    } else {
        return *g;
    }
}

// Global reads issued and waited by hand, for a loop that keeps a round of
// reads in flight across its arithmetic: the compiler, counting its own
// reads, sinks each to its use and waits there for it and every read
// behind it. `wait_reads<N>` waits until at most N reads are pending (the
// count drains in issue order); `hold` ties a read's registers to the
// point after the wait, so no use of them moves above it. NT marks a read
// no later read finds in cache (the streamed codes).
template <int BYTES, bool NT = false>
__device__ __forceinline__ void issue_read(Words<BYTES / 4>& dst, const void* p) {
    static_assert(BYTES == 4 || BYTES == 8 || BYTES == 16, "a read is one, two or four words");
    if constexpr (BYTES == 16) {
        if constexpr (NT) {
            asm volatile("global_load_dwordx4 %0, %1, off nt" : "=v"(dst) : "v"(p) : "memory");
        } else {
            asm volatile("global_load_dwordx4 %0, %1, off" : "=v"(dst) : "v"(p) : "memory");
        }
    } else if constexpr (BYTES == 8) {
        if constexpr (NT) {
            asm volatile("global_load_dwordx2 %0, %1, off nt" : "=v"(dst) : "v"(p) : "memory");
        } else {
            asm volatile("global_load_dwordx2 %0, %1, off" : "=v"(dst) : "v"(p) : "memory");
        }
    } else {
        if constexpr (NT) {
            asm volatile("global_load_dword %0, %1, off nt" : "=v"(dst) : "v"(p) : "memory");
        } else {
            asm volatile("global_load_dword %0, %1, off" : "=v"(dst) : "v"(p) : "memory");
        }
    }
}

// One 16-bit (or, BYTE, 8-bit) value, zero-extended.
template <bool BYTE = false>
__device__ __forceinline__ void issue_read_small(u32& dst, const void* p) {
    if constexpr (BYTE) {
        asm volatile("global_load_ubyte %0, %1, off" : "=v"(dst) : "v"(p) : "memory");
    } else {
        asm volatile("global_load_ushort %0, %1, off" : "=v"(dst) : "v"(p) : "memory");
    }
}

template <int N>
__device__ __forceinline__ void wait_reads() {
    static_assert(N >= 0 && N < 64, "vmcnt counts to 63");
    asm volatile("s_waitcnt vmcnt(%0)" ::"n"(N) : "memory");
}

template <class R>
__device__ __forceinline__ void hold(R& r) {
    asm volatile("" : "+v"(r));
}

constexpr int kOffPost = 0;
constexpr int kOffPreInt = 1;
constexpr int kOffPreReal = 2;
constexpr int kOffPreConst = 3;

// Where a streaming tier seated a projection's planes this step: the fire
// binds the slot, the tier fills it before the step runs.
struct alignas(16) AffineSeat {
    const u8* codes;
    const u8* scales;
    const u8* biases;
    const u8* pad;
};

struct AffinePlanes {
    const u8* codes;
    const u8* scales;
    const u8* biases;
};

__device__ __forceinline__ AffinePlanes seated(const u8* codes, const u8* scales, const u8* biases,
                                               const AffineSeat* seat) {
    if (seat == nullptr) return {codes, scales, biases};
    const AffineSeat at = *seat;
    return {at.codes, at.scales, at.biases};
}

// One group's factor pair: the scale and the offset as the format states it.
struct Factor {
    float s, off;
};

template <class F, int kBits, int kOffset>
__device__ __forceinline__ Factor factor_at(const u8* scales, const u8* biases, long long fx) {
    const float s = to_f32(reinterpret_cast<const F*>(scales)[fx]);
    if constexpr (kOffset == kOffPost || kOffset == kOffPreReal) {
        return {s, to_f32(reinterpret_cast<const F*>(biases)[fx])};
    } else if constexpr (kOffset == kOffPreInt) {
        return {s, static_cast<float>(biases[fx])};
    } else {
        return {s, static_cast<float>(1 << (kBits - 1))};
    }
}

// The weight code c stands for.
template <int kOffset>
__device__ __forceinline__ float dequant(float c, Factor f) {
    if constexpr (kOffset == kOffPost) {
        return fmaf(c, f.s, f.off);
    } else {
        return f.s * (c - f.off);
    }
}

// The weight as s·c + o: o is the offset a post-offset factor adds, or the
// −s·z a pre-offset one subtracts.
template <int kOffset>
__device__ __forceinline__ float additive(Factor f) {
    return kOffset == kOffPost ? f.off : -f.s * f.off;
}

// The codes of one 32-bit word as matrix-core operand pairs in k order:
// pair j holds codes 2j and 2j + 1 as two 16-bit T, each `kMagic + c`
// exactly. A product with them sums Σ x·(c + kMagic); the kMagic·Σ x it
// carries comes off with the factors. The magic is an exponent whose ulp
// is 1, so the code is OR-ed into the mantissa: 128 in bf16 (codes below
// 128), 1024 in f16. Eight-bit codes in bf16 overflow 128's mantissa and
// are converted outright.
template <class T, int kBits>
struct Unpack {
    static constexpr int kPerWord = 32 / kBits;
    static constexpr int kPairs = kPerWord / 2;
    static constexpr bool kF16 = is_same<T, f16>::value;
    static constexpr bool kConvert = !kF16 && kBits == 8;
    static constexpr float kMagic = kF16 ? 1024.f : (kConvert ? 0.f : 128.f);
    static constexpr u32 kHigh = kF16 ? 0x64006400u : 0x43004300u;
    static_assert(kBits == 2 || kBits == 4 || kBits == 8, "codes are two, four or eight bits");

    // Byte j of `lo` and byte j of `hi` as the two 16-bit halves' low bytes.
    __device__ static __forceinline__ u32 spread(u32 hi, u32 lo, int j) {
        return __builtin_amdgcn_perm(hi, lo, 0x0c000c00u | static_cast<u32>(j) | (static_cast<u32>(4 + j) << 16));
    }

    __device__ static __forceinline__ void pairs(u32 q, u32 (&p)[kPairs]) {
        if constexpr (kBits == 4) {
            const u32 even = q & 0x0f0f0f0fu, odd = (q >> 4) & 0x0f0f0f0fu;
#pragma unroll
            for (int j = 0; j < 4; ++j) p[j] = spread(odd, even, j) | kHigh;
        } else if constexpr (kBits == 2) {
            u32 at[4];
#pragma unroll
            for (int s = 0; s < 4; ++s) at[s] = (q >> (2 * s)) & 0x03030303u;  // byte j: code 4j + s
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int t = 0; t < 2; ++t) p[2 * j + t] = spread(at[2 * t + 1], at[2 * t], j) | kHigh;
        } else if constexpr (kConvert) {
            u32 f[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) f[i] = __builtin_bit_cast(u32, static_cast<float>((q >> (8 * i)) & 0xffu));
            // A small integer's f32 is its bf16 in the upper half, exactly.
#pragma unroll
            for (int j = 0; j < 2; ++j) p[j] = __builtin_amdgcn_perm(f[2 * j + 1], f[2 * j], 0x07060302u);
        } else {
#pragma unroll
            for (int j = 0; j < 2; ++j)
                p[j] = __builtin_amdgcn_perm(q, kHigh, 0x01000100u | static_cast<u32>(4 + 2 * j) |
                                                           (static_cast<u32>(5 + 2 * j) << 16));
        }
    }
};

// (x & mask) | bits as one instruction: with both constants literal the
// compiler splits it in two (a three-operand instruction takes no literal
// on gfx9), so they reach it opaque, from registers. (Not the instruction
// itself in asm: the hazard checks cannot see into asm next to the
// matrix cores.)
__device__ __forceinline__ u32 and_or(u32 x, u32 mask, u32 bits) {
    asm("" : "+v"(mask));
    asm("" : "+s"(bits));
    return (x & mask) | bits;
}

// A 32-bit word of P codes as MFMA operands in the cheapest order: pair i
// holds codes i and i + P/2 (one shift and one and-or, the magic exponent
// filling both halves), and `act` lays the word's activations out the same
// way. Eight-bit codes in bf16 convert outright, in that order.
template <class T, int kBits>
struct Quads {
    static constexpr int kPer = 32 / kBits;  // codes a word
    static constexpr int kSteps = kPer / 4;  // 4-code operands a word
    static constexpr bool kF16 = is_same<T, f16>::value;
    static constexpr bool kConvert = !kF16 && kBits == 8;
    static constexpr float kMagic = kF16 ? 1024.f : (kConvert ? 0.f : 128.f);
    static constexpr u32 kHigh = kF16 ? 0x64006400u : 0x43004300u;
    static constexpr u32 kMask = ((1u << kBits) - 1) * 0x00010001u;
    static_assert(kBits == 2 || kBits == 4 || kBits == 8, "codes are two, four or eight bits");

    // Operand h of word q, as B.
    __device__ static __forceinline__ void codes(u32 q, Words<2> (&b)[kSteps]) {
        if constexpr (kConvert) {
            u32 f[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) f[i] = __builtin_bit_cast(u32, static_cast<float>((q >> (8 * i)) & 0xffu));
            // A small integer's f32 is its bf16 in the upper half, exactly.
            b[0] = Words<2>{__builtin_amdgcn_perm(f[2], f[0], 0x07060302u),
                            __builtin_amdgcn_perm(f[3], f[1], 0x07060302u)};
        } else {
#pragma unroll
            for (int h = 0; h < kSteps; ++h)
                b[h] = Words<2>{and_or(q >> (kBits * 2 * h), kMask, kHigh),
                                and_or(q >> (kBits * (2 * h + 1)), kMask, kHigh)};
        }
    }

    // Operand h of the activations d[0 .. P/2) under the same codes, as A.
    __device__ static __forceinline__ Words<2> act(const u32* d, int h) {
        const u32 lo = d[h], hi = d[h + kPer / 4];
        return Words<2>{__builtin_amdgcn_perm(hi, lo, 0x05040100u), __builtin_amdgcn_perm(hi, lo, 0x07060302u)};
    }
};

// Two 16-bit operands of type T, one per half, of 1.0: the B operand that
// makes a product the row sums of A.
template <class T>
constexpr u32 kOnes = is_same<T, f16>::value ? 0x3c003c00u : 0x3f803f80u;

// The 4 x 4 x 4 matrix core, sixteen blocks a wavefront: lane 4b + j holds
// row j of block b's A and column j of its B (k 0..3 in its four halves),
// and D[t] of lane 4b + j is Σ_k A[4b + t][k] · B[4b + j][k].
template <class T>
__device__ __forceinline__ mfma_cd mfma4x4(Words<2> a, Words<2> b, mfma_cd c) {
    if constexpr (is_same<T, f16>::value) {
        typedef _Float16 h4 __attribute__((ext_vector_type(4)));
        return __builtin_amdgcn_mfma_f32_4x4x4f16(__builtin_bit_cast(h4, a), __builtin_bit_cast(h4, b), c, 0, 0, 0);
    } else {
        return __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(__builtin_bit_cast(mfma_ab, a), __builtin_bit_cast(mfma_ab, b), c,
                                                       0, 0, 0);
    }
}

// The 16 x 16 x 16 matrix core over T, laid out as `mfma16` (prelude).
template <class T>
__device__ __forceinline__ mfma_cd mfma16x16(Words<2> a, Words<2> b, mfma_cd c) {
    if constexpr (is_same<T, f16>::value) {
        typedef _Float16 h4 __attribute__((ext_vector_type(4)));
        return __builtin_amdgcn_mfma_f32_16x16x16f16(__builtin_bit_cast(h4, a), __builtin_bit_cast(h4, b), c, 0, 0, 0);
    } else {
        return mfma16(__builtin_bit_cast(mfma_ab, a), __builtin_bit_cast(mfma_ab, b), c);
    }
}

// Two f32 as two 16-bit T in one word, low first, rounded to nearest even.
// bf16 rounds by hand: the conversion's NaN guard costs more than the
// rounding, and a decoded weight is finite (a NaN factor still lands NaN).
template <class T>
__device__ __forceinline__ u32 pack_pair(float lo, float hi) {
    if constexpr (is_same<T, bf16>::value) {
        auto round = [](float v) {
            const u32 b = __builtin_bit_cast(u32, v);
            return b + 0x7fffu + ((b >> 16) & 1u);
        };
        return __builtin_amdgcn_perm(round(hi), round(lo), 0x07060302u);
    } else {
        struct Two {
            T v[2];
        };
        return __builtin_bit_cast(u32, Two{{from_f32<T>(lo), from_f32<T>(hi)}});
    }
}

// The codes of one 32-bit word as f32, in k order.
template <int kBits>
__device__ __forceinline__ void codes_f32(u32 q, float (&c)[32 / kBits]) {
    if constexpr (kBits == 8) {
#pragma unroll
        for (int i = 0; i < 4; ++i) c[i] = static_cast<float>((q >> (8 * i)) & 0xffu);
    } else if constexpr (kBits == 4) {
        // Codes as bytes, so each converts in one byte-to-float.
        const u32 even = q & 0x0f0f0f0fu, odd = (q >> 4) & 0x0f0f0f0fu;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            c[2 * i] = static_cast<float>((even >> (8 * i)) & 0xffu);
            c[2 * i + 1] = static_cast<float>((odd >> (8 * i)) & 0xffu);
        }
    } else {
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const u32 at = (q >> (2 * s)) & 0x03030303u;
#pragma unroll
            for (int j = 0; j < 4; ++j) c[4 * j + s] = static_cast<float>((at >> (8 * j)) & 0xffu);
        }
    }
}

// --------------------------------------------------------------- GEMM --
//
// y = act · Wᵀ over a quantized W at prefill: four wavefronts land a BM x
// 128 tile, each 64-wide K stage's codes decoded once into an LDS weight
// tile (rounded as a decoded plane lands them), the next stage in flight
// across the multiply. A short grid slices K over blockIdx.z into f32
// partials `affine_gemm_finalize` sums in slice order.

constexpr int kGemmKC = 64;
constexpr int kGemmLd = kGemmKC + 8;  // row stride of an LDS tile: off the bank period
constexpr int kGemmBN = 128;
constexpr int kGemmThreads = 4 * 64;

template <class T, int BM, class Stage>
__device__ __forceinline__ void affine_gemm(const T* __restrict__ act, T* __restrict__ out, float* __restrict__ part,
                                            int per, int rows, int n, int k, Stage& stage) {
    static_assert(BM % 32 == 0 && BM <= 128, "two wavefronts along m, whole 16-row tiles each");
    static_assert(kWave == 64, "the block is four 64-lane wavefronts");
    constexpr int MT = BM / 32;  // 16-row tiles of the activation a wavefront
    constexpr int NT = 4;        // 16-row tiles of the weight a wavefront (two along n)
    constexpr int XP = BM * kGemmKC / 8 / kGemmThreads;  // 16-byte activation reads a thread a stage
    __shared__ __align__(16) T ws[kGemmBN * kGemmLd];
    __shared__ __align__(16) T xs[BM * kGemmLd];

    const int tid = threadIdx.x, l = lane(), wave = tid / kWave;
    const int r16 = l % 16, g = l / 16;
    const int wn = wave % 2, wm = wave / 2;
    const int m0 = blockIdx.x * BM;
    const int first = blockIdx.z * per, chunks = min(k / kGemmKC, first + per);
    if (first >= chunks) return;

    // Rows past the live ones read the last live row; nothing stores them.
    const T* xrow[XP];
#pragma unroll
    for (int i = 0; i < XP; ++i) {
        const int pack = tid + i * kGemmThreads;
        xrow[i] = act + static_cast<long long>(min(m0 + pack / 8, rows - 1)) * k + pack % 8 * 8;
    }
    Words<4> xr[XP];
#pragma unroll
    for (int i = 0; i < XP; ++i) xr[i] = *reinterpret_cast<const Words<4>*>(xrow[i] + first * kGemmKC);
    stage.load(first);

    mfma_cd acc[NT][MT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) acc[nt][mt] = mfma_zero();

    for (int c = first; c < chunks; ++c) {
#pragma unroll
        for (int i = 0; i < XP; ++i) {
            const int pack = tid + i * kGemmThreads;
            *reinterpret_cast<Words<4>*>(xs + pack / 8 * kGemmLd + pack % 8 * 8) = xr[i];
        }
        stage.store(ws);
        lds_barrier();
        // The last stage rereads itself, so the read is unconditional.
        const int next = min(c + 1, chunks - 1);
#pragma unroll
        for (int i = 0; i < XP; ++i) xr[i] = *reinterpret_cast<const Words<4>*>(xrow[i] + next * kGemmKC);
        stage.load(next);
#pragma unroll
        for (int kk = 0; kk < kGemmKC; kk += 32) {
            // Lane group g takes k 8g..8g+7 of the 32: two k-steps from one
            // 16-byte read per operand.
            Words<4> a[NT], b[MT];
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
                a[nt] = *reinterpret_cast<const Words<4>*>(ws + (wn * 64 + nt * 16 + r16) * kGemmLd + kk + 8 * g);
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
                b[mt] = *reinterpret_cast<const Words<4>*>(xs + (wm * (BM / 2) + mt * 16 + r16) * kGemmLd + kk + 8 * g);
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    acc[nt][mt] = mfma16x16<T>(Words<2>{a[nt].x, a[nt].y}, Words<2>{b[mt].x, b[mt].y}, acc[nt][mt]);
                    acc[nt][mt] = mfma16x16<T>(Words<2>{a[nt].z, a[nt].w}, Words<2>{b[mt].z, b[mt].w}, acc[nt][mt]);
                }
        }
        lds_barrier();
    }

    const int n0 = blockIdx.y * kGemmBN;
    const bool quads = n % 4 == 0;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int m = m0 + wm * (BM / 2) + mt * 16 + r16;
        if (m >= rows) continue;
        T* y = out + static_cast<long long>(m) * n;
        float* p = part + (static_cast<long long>(blockIdx.z) * rows + m) * n;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int col = n0 + wn * 64 + nt * 16 + 4 * g;
            if (part != nullptr) {
#pragma unroll
                for (int t = 0; t < 4; ++t)
                    if (col + t < n) p[col + t] = acc[nt][mt][t];
            } else if (quads && col < n) {
                *reinterpret_cast<Words<2>*>(y + col) =
                    Words<2>{pack_pair<T>(acc[nt][mt][0], acc[nt][mt][1]), pack_pair<T>(acc[nt][mt][2], acc[nt][mt][3])};
            } else {
#pragma unroll
                for (int t = 0; t < 4; ++t)
                    if (col + t < n) y[col + t] = from_f32<T>(acc[nt][mt][t]);
            }
        }
    }
}

// y = Σ part over `splits` slices, in slice order, rounded once.
template <class T>
__global__ void affine_gemm_finalize(const float* __restrict__ part, T* __restrict__ out, int m, int n, int splits,
                                     const u32* __restrict__ win) {
    const int rows = win == nullptr ? m : min(m, static_cast<int>(win[0]));
    const long long cells = static_cast<long long>(rows) * n;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < cells;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        float v = 0.f;
        for (int s = 0; s < splits; ++s) v += part[s * cells + i];
        out[i] = from_f32<T>(v);
    }
}

}  // namespace pie::linear
