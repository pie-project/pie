#pragma once

// The paged KV cache's writes. A page holds `page_size` slots of `h_kv`
// heads × `d` elements, NHD (slot → head → d) or HND (head → slot → d):
// bf16, or OCP e4m3fn codes under a unit scale (an e4m3 pool, vLLM's `fp8`
// KV), which the attention reads as they are.
//
// gfx942 converts fnuz, under whose bias an OCP code holds half its value
// (but for 0x80, fnuz's NaN and OCP's −0, and 0x7f / 0xff, OCP's NaNs). The
// writer stores fnuz(x / 2) saturated to OCP's 448, so −0 lands as 0x00 and
// no NaN is made; a reader converts in registers to bf16 of half the value
// and folds `kE4m3Unit` into its scales.

#include "prim/device.hip"
#include "prim/mfma.hip"

namespace pie::attn {

// The element offset of (page, slot, head, j) in either layout.
template <bool HND>
__device__ __forceinline__ long long kv_at(long long page, int slot, int head, int j, int page_size,
                                           int h_kv, int d) {
    return HND ? ((page * h_kv + head) * page_size + slot) * d + j
               : ((page * page_size + slot) * h_kv + head) * d + j;
}

constexpr float kE4m3Unit = 2.f;

// Four values as OCP e4m3fn codes, round to nearest even, saturated.
__device__ __forceinline__ u32 e4m3x4(float a, float b, float c, float d) {
    auto half = [](float x) { return fminf(fmaxf(x * 0.5f, -224.f), 224.f); };
    const u32 lo = __builtin_amdgcn_cvt_pk_fp8_f32(half(a), half(b), 0, false);
    return __builtin_amdgcn_cvt_pk_fp8_f32(half(c), half(d), lo, true);
}

// Four e4m3fn codes as bf16 of half their value (exact: bf16 holds e4m3).
__device__ __forceinline__ mfma_ab e4m3x4_bf16(u32 w) {
    const auto lo = __builtin_amdgcn_cvt_pk_f32_fp8(w, false);
    const auto hi = __builtin_amdgcn_cvt_pk_f32_fp8(w, true);
    constexpr u32 kHighs = 0x07060302u;  // the high halves of two f32
    const u32 a = __builtin_amdgcn_perm(__float_as_uint(lo[1]), __float_as_uint(lo[0]), kHighs);
    const u32 b = __builtin_amdgcn_perm(__float_as_uint(hi[1]), __float_as_uint(hi[0]), kHighs);
    return __builtin_bit_cast(mfma_ab, (u64{b} << 32) | a);
}

__device__ __forceinline__ mfma_ab2 e4m3x8_bf16(u64 w) {
    return {e4m3x4_bf16(static_cast<u32>(w)), e4m3x4_bf16(static_cast<u32>(w >> 32))};
}

// A pool's elements as its readers take them: `FP8` an e4m3 pool's codes
// (at the element offsets a bf16 pool's would be), read as bf16 of `1 /
// unit` their value. A read lands raw (`Raw` eight elements, `Raw4` four)
// and widens where it is used, so a prefetch stays in flight. A matrix-core
// operand's k step `st` of lane group `quad` sits at column `col(st, quad)`
// (which k a step takes is the operands' own business): sixteen bytes a
// lane a read either way, `keys` the reads of N steps from `st0` (even).
template <bool FP8>
struct Kv {
    using Raw = Pack8<bf16>;
    struct alignas(8) Raw4 {
        bf16 v[4];
    };
    static constexpr float unit = 1.f;
    __device__ static Raw raw(const bf16* base, long long at) { return *reinterpret_cast<const Raw*>(base + at); }
    __device__ static Raw4 raw4(const bf16* base, long long at) { return *reinterpret_cast<const Raw4*>(base + at); }
    __device__ static Pack8<bf16> wide(const Raw& r) { return r; }
    __device__ static mfma_ab2 ab2(const Raw& r) { return __builtin_bit_cast(mfma_ab2, r); }
    __device__ static float f(const Raw4& r, int c) { return to_f32(r.v[c]); }
    __device__ static float one(const bf16* base, long long at) { return to_f32(base[at]); }
    __device__ static int col(int st, int quad) { return 32 * st + 8 * quad; }
    template <int N>
    __device__ static void keys(const bf16* base, long long row, int quad, int st0, Raw (&out)[N]) {
#pragma unroll
        for (int i = 0; i < N; ++i) out[i] = raw(base, row + col(st0 + i, quad));
    }
};

template <>
struct Kv<true> {
    using Raw = u64;
    using Raw4 = u32;
    static constexpr float unit = kE4m3Unit;
    __device__ static const u8* bytes(const bf16* base) { return reinterpret_cast<const u8*>(base); }
    __device__ static Raw raw(const bf16* base, long long at) { return *reinterpret_cast<const Raw*>(bytes(base) + at); }
    __device__ static Raw4 raw4(const bf16* base, long long at) { return *reinterpret_cast<const Raw4*>(bytes(base) + at); }
    __device__ static Pack8<bf16> wide(Raw r) { return __builtin_bit_cast(Pack8<bf16>, e4m3x8_bf16(r)); }
    __device__ static mfma_ab2 ab2(Raw r) { return e4m3x8_bf16(r); }
    __device__ static float f(Raw4 r, int c) { return __builtin_amdgcn_cvt_f32_fp8(r >> (8 * c), 0); }
    __device__ static float one(const bf16* base, long long at) { return __builtin_amdgcn_cvt_f32_fp8(bytes(base)[at], 0); }
    __device__ static int col(int st, int quad) { return 64 * (st >> 1) + 16 * quad + 8 * (st & 1); }
    template <int N>
    __device__ static void keys(const bf16* base, long long row, int quad, int st0, Raw (&out)[N]) {
        static_assert(N % 2 == 0, "steps in pairs");
#pragma unroll
        for (int i = 0; i < N; i += 2) {
            const Pack8<bf16> w = *reinterpret_cast<const Pack8<bf16>*>(bytes(base) + row + col(st0 + i, quad));
            const auto pair = __builtin_bit_cast(mfma_ab2, w);
            out[i] = __builtin_bit_cast(u64, pair.lo);
            out[i + 1] = __builtin_bit_cast(u64, pair.hi);
        }
    }
};

// Token row b's k and v (h_kv·d each) into page w_page[b], slot w_off[b],
// as bf16 or (FP8) e4m3 codes. A row whose slot is out of range, or that
// `row_valid` marks dead, is skipped. One block per token row; 8 elements a
// thread when d allows.
template <bool HND, bool FP8>
__global__ void kv_append_explicit(const bf16* __restrict__ k, const bf16* __restrict__ v,
                                   bf16* __restrict__ k_pages, bf16* __restrict__ v_pages,
                                   const u32* __restrict__ w_page, const u32* __restrict__ w_off,
                                   const u8* __restrict__ row_valid, int B, int page_size,
                                   int h_kv, int d, const u32* __restrict__ win) {
    const Window window{win};
    if (static_cast<int>(blockIdx.x) >= B || !window.live(blockIdx.x)) return;
    const long long b = window.plane(blockIdx.x);
    if (row_valid != nullptr && row_valid[b] == 0) return;
    const long long page = w_page[b];
    const int slot = static_cast<int>(w_off[b]);
    if (slot < 0 || slot >= page_size) return;
    const int row = h_kv * d;
    const bf16* ks = k + b * row;
    const bf16* vs = v + b * row;
    if constexpr (FP8) {
        auto codes = [](const Pack8<bf16>& x) {
            return u64{e4m3x4(to_f32(x.v[0]), to_f32(x.v[1]), to_f32(x.v[2]), to_f32(x.v[3]))} |
                   u64{e4m3x4(to_f32(x.v[4]), to_f32(x.v[5]), to_f32(x.v[6]), to_f32(x.v[7]))} << 32;
        };
        u8* kp = reinterpret_cast<u8*>(k_pages);
        u8* vp = reinterpret_cast<u8*>(v_pages);
        if (d % 8 == 0 && aligned16(ks) && aligned16(vs)) {
            for (int i = threadIdx.x * 8; i < row; i += blockDim.x * 8) {
                const long long dst = kv_at<HND>(page, slot, i / d, i % d, page_size, h_kv, d);
                *reinterpret_cast<u64*>(kp + dst) = codes(*reinterpret_cast<const Pack8<bf16>*>(ks + i));
                *reinterpret_cast<u64*>(vp + dst) = codes(*reinterpret_cast<const Pack8<bf16>*>(vs + i));
            }
            return;
        }
        for (int i = threadIdx.x; i < row; i += blockDim.x) {
            const long long dst = kv_at<HND>(page, slot, i / d, i % d, page_size, h_kv, d);
            kp[dst] = static_cast<u8>(e4m3x4(to_f32(ks[i]), 0.f, 0.f, 0.f));
            vp[dst] = static_cast<u8>(e4m3x4(to_f32(vs[i]), 0.f, 0.f, 0.f));
        }
        return;
    }
    if (d % 8 == 0 && aligned16(ks) && aligned16(vs) && aligned16(k_pages) && aligned16(v_pages)) {
        for (int i = threadIdx.x * 8; i < row; i += blockDim.x * 8) {
            const long long dst = kv_at<HND>(page, slot, i / d, i % d, page_size, h_kv, d);
            *reinterpret_cast<b128*>(k_pages + dst) = *reinterpret_cast<const b128*>(ks + i);
            *reinterpret_cast<b128*>(v_pages + dst) = *reinterpret_cast<const b128*>(vs + i);
        }
        return;
    }
    for (int i = threadIdx.x; i < row; i += blockDim.x) {
        const long long dst = kv_at<HND>(page, slot, i / d, i % d, page_size, h_kv, d);
        k_pages[dst] = ks[i];
        v_pages[dst] = vs[i];
    }
}

}  // namespace pie::attn
